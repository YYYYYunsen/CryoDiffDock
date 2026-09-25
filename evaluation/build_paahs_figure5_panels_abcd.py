#!/usr/bin/env python3
"""Build Figure 5 panels A-D and their supplements from the PAAHS bundle.

Panels A and B are recomputed on the final PAAHS poses.  The outcome is
always whether the Run-1 PAAHS pose fails (symmetry-aware RMSD >= 2 A).
Two different uncertainty features share the outcome:

  * budget 1 uses within-run dispersion of the 40-pose Run-1 sampling
    pool, one minus its largest symmetry-aware 2 A cluster fraction;
  * budgets 2, 4 and 8 use cross-run disagreement, one minus the
    fraction of the first k reported Top-1 poses that fall in the 2 A
    cluster containing the Run-1 pose.

They are not the same quantity and are drawn and captioned separately.

Panels C and D are read from the frozen budget and joint density/PB
tables.  Panel C's cost axis reuses the Figure 2 runtime definition so
the 1-run operating point matches Figure 2 panel D exactly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

CROSS_RUN_BUDGETS = (2, 4, 8)
ALL_BUDGETS = (1, 2, 4, 8)
CLUSTER_CUTOFF_A = 2.0
SUCCESS_CUTOFF_A = 2.0
TIMING_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
POOL_FEATURE = "one_minus_largest_cluster_fraction_at_2p0A"
CROSS_RUN_FEATURE = "one_minus_run1_cluster_fraction_at_2p0A"
LOCAL_RUNTIME_METHOD = "CryoDiffDock-Local V2 selected"


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def bootstrap_ci(
    labels: np.ndarray, scores: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    count = len(labels)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, count, count)
        value = binary_auc(labels[index], scores[index])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return 100.0 * max(0.0, center - spread), 100.0 * min(1.0, center + spread)


def distance_matrix(frame: pd.DataFrame, replicates: list[str]) -> np.ndarray:
    count = len(replicates)
    order = {name: position for position, name in enumerate(replicates)}
    distances = np.zeros((count, count), dtype=float)
    seen = np.zeros((count, count), dtype=bool)
    for row in frame.itertuples(index=False):
        left, right = order[row.replicate_left], order[row.replicate_right]
        distances[left, right] = distances[right, left] = float(row.pairwise_rmsd_A)
        seen[left, right] = seen[right, left] = True
    np.fill_diagonal(seen, True)
    if not seen.all():
        raise ValueError("Incomplete pairwise RMSD matrix")
    return distances


def selected_cluster_fraction(distances: np.ndarray, selected_index: int) -> float:
    count = len(distances)
    if count == 1:
        return 1.0
    hierarchy = linkage(squareform(distances, checks=False), method="complete")
    labels = fcluster(hierarchy, t=CLUSTER_CUTOFF_A, criterion="distance")
    return float(np.sum(labels == labels[selected_index]) / count)


def stable_labels(distances: np.ndarray) -> np.ndarray:
    """Complete-linkage 2 A labels, relabelled by each cluster's first member."""
    count = len(distances)
    if count == 1:
        return np.ones(1, dtype=int)
    hierarchy = linkage(squareform(distances, checks=False), method="complete")
    raw = fcluster(hierarchy, t=CLUSTER_CUTOFF_A, criterion="distance")
    ordered = sorted(set(raw), key=lambda value: int(np.flatnonzero(raw == value)[0]))
    relabel = {value: index + 1 for index, value in enumerate(ordered)}
    return np.asarray([relabel[value] for value in raw], dtype=int)


def medoid_index(distances: np.ndarray, members: np.ndarray) -> int:
    members = np.asarray(members, dtype=int)
    submatrix = distances[np.ix_(members, members)]
    return int(members[int(np.argmin(submatrix.mean(axis=1)))])


def build_features(bundle: Path, pool_path: Path) -> pd.DataFrame:
    modes = bundle / "04_figure5_paahs_pose_modes_v1"
    pairwise = pd.read_csv(modes / "pairwise_pose_rmsd.csv")
    summary = pd.read_csv(modes / "per_case_mode_summary.csv")
    records = []
    for case, group in pairwise.groupby("complex_name", sort=True):
        replicates = sorted(
            set(group["replicate_left"]) | set(group["replicate_right"]),
            key=lambda name: int(name.replace("run", "")),
        )
        if replicates[0] != "run1":
            raise ValueError(f"{case}: Run 1 is not the first replicate")
        distances = distance_matrix(group, replicates)
        record = {"complex_name": case, "run_count": len(replicates)}
        for budget in CROSS_RUN_BUDGETS:
            if budget > len(replicates):
                raise ValueError(f"{case}: budget {budget} exceeds run count")
            fraction = selected_cluster_fraction(distances[:budget, :budget], 0)
            record[f"disagreement_at_{budget}"] = 1.0 - fraction
        records.append(record)
    features = pd.DataFrame.from_records(records)
    features = features.merge(
        summary[["complex_name", "run1_rmsd_A", "run1_success", "run1_disagreement"]],
        on="complex_name",
        how="inner",
        validate="one_to_one",
    )
    if len(features) != len(summary):
        raise ValueError("Case coverage mismatch between pairwise RMSD and summary")
    mismatch = np.abs(features["disagreement_at_8"] - features["run1_disagreement"])
    if float(mismatch.max()) > 1e-9:
        raise ValueError(
            "Recomputed 8-run disagreement disagrees with the frozen audit column"
        )
    features["run1_failure"] = features["run1_rmsd_A"] >= SUCCESS_CUTOFF_A
    if not (features["run1_failure"] != features["run1_success"]).all():
        raise ValueError("Run-1 failure label disagrees with the frozen success column")

    pool = pd.read_csv(pool_path)
    if set(pool["complex_name"]) != set(features["complex_name"]):
        raise ValueError("Run-1 sampling pool does not cover the same cases")
    if int(pool["pose_count"].nunique()) != 1:
        raise ValueError("Run-1 sampling pool has an inhomogeneous pose count")
    pool = pool.assign(
        disagreement_at_1=1.0 - pool["largest_cluster_fraction_at_2p0A"],
        pool_pose_count=pool["pose_count"],
    )
    features = features.merge(
        pool[["complex_name", "disagreement_at_1", "pool_pose_count"]],
        on="complex_name",
        how="inner",
        validate="one_to_one",
    )
    return features


def build_panel_a(features: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    labels = features["run1_failure"].to_numpy(bool)
    rows = []
    for offset, budget in enumerate(ALL_BUDGETS):
        scores = features[f"disagreement_at_{budget}"].to_numpy(float)
        auroc = binary_auc(labels, scores)
        low, high = bootstrap_ci(labels, scores, repeats, seed + offset)
        rows.append({
            "budget_runs": budget,
            "feature_family": "within_run_pool" if budget == 1 else "cross_run",
            "score_column": (
                POOL_FEATURE if budget == 1
                else f"{CROSS_RUN_FEATURE}_budget{budget}"
            ),
            "n": int(len(labels)),
            "failure_n": int(labels.sum()),
            "success_n": int((~labels).sum()),
            "failure_auroc": auroc,
            "failure_auroc_95ci_low": low,
            "failure_auroc_95ci_high": high,
            "distinct_score_values": int(len(np.unique(scores))),
        })
    return pd.DataFrame(rows)


def build_panel_b(features: pd.DataFrame) -> pd.DataFrame:
    labels = features["run1_failure"].to_numpy(bool)
    total = len(labels)
    rows = []
    for budget in ALL_BUDGETS:
        scores = features[f"disagreement_at_{budget}"].to_numpy(float)
        for threshold in np.unique(scores):
            retained = scores <= threshold
            retained_n = int(retained.sum())
            failures = int(labels[retained].sum())
            low, high = wilson_interval(failures, retained_n)
            rows.append({
                "budget_runs": budget,
                "feature_family": "within_run_pool" if budget == 1 else "cross_run",
                "uncertainty_threshold": float(threshold),
                "retained_n": retained_n,
                "coverage_pct": 100.0 * retained_n / total,
                "failure_n": failures,
                "failure_pct": 100.0 * failures / retained_n,
                "failure_wilson_95ci_low_pct": low,
                "failure_wilson_95ci_high_pct": high,
            })
    return pd.DataFrame(rows)


def replicate_wall_clock_minutes(bundle: Path) -> tuple[float, list[dict]]:
    records = []
    for path in sorted(bundle.glob("replicate*/replicate_summary.json")):
        payload = json.loads(path.read_text())
        started, completed = payload.get("started_utc"), payload.get("completed_utc")
        if not started or not completed:
            records.append({
                "replicate": path.parent.name,
                "minutes_per_case": None,
                "timing": "reused sharded run; not a single-GPU measurement",
            })
            continue
        seconds = (
            dt.datetime.strptime(completed, TIMING_FORMAT)
            - dt.datetime.strptime(started, TIMING_FORMAT)
        ).total_seconds()
        records.append({
            "replicate": path.parent.name,
            "minutes_per_case": seconds / float(payload["case_count"]) / 60.0,
            "timing": "measured one-GPU wall clock",
        })
    measured = [r["minutes_per_case"] for r in records if r["minutes_per_case"]]
    if not measured:
        raise ValueError("No measured replicate timing found")
    return statistics.median(measured), records


def figure2_runtime_minutes(runtime_path: Path) -> float:
    runtime = pd.read_csv(runtime_path)
    local = runtime.loc[runtime["method"] == LOCAL_RUNTIME_METHOD]
    if local.empty:
        raise ValueError(f"{LOCAL_RUNTIME_METHOD!r} missing from {runtime_path}")
    return float(local["runtime_minutes"].mean())


SCORE_COLUMNS = (
    "final_selection_score",
    "multiscale_ncc",
    "ncc",
    "ncc_sigma_1p5",
    "ncc_sigma_2p25",
    "confidence",
    "selection_score",
    "expanded_density_z",
    "basin_q25_ncc",
    "interaction_geometry_quality",
)
PRIMARY_SCORE = "final_selection_score"


def case_blocks(bundle: Path, features: pd.DataFrame):
    """Yield (case, distance matrix, per-run frame) for each case, runs in order."""
    modes = bundle / "04_figure5_paahs_pose_modes_v1"
    pairwise = pd.read_csv(modes / "pairwise_pose_rmsd.csv")
    assignments = pd.read_csv(modes / "pose_mode_assignments.csv")
    cases = set(features["complex_name"])
    for case, group in pairwise.groupby("complex_name", sort=True):
        if case not in cases:
            continue
        replicates = sorted(
            set(group["replicate_left"]) | set(group["replicate_right"]),
            key=lambda name: int(name.replace("run", "")),
        )
        poses = (
            assignments.loc[assignments["complex_name"] == case]
            .sort_values("replicate_order")
            .reset_index(drop=True)
        )
        if list(poses["replicate"]) != replicates:
            raise ValueError(f"{case}: replicate order mismatch")
        yield case, distance_matrix(group, replicates), poses


def rerank_selection(
    bundle: Path, features: pd.DataFrame, score_column: str
) -> pd.DataFrame:
    """Cross-run reranking of the k reported poses by one selection score.

    The candidate set is exactly the k run-level PAAHS Top-1 poses.  One
    reported pose is the highest-scoring of them.  Two reported poses adds the
    highest-scoring pose at least the clustering cutoff away from the first, so
    the second answer is a genuinely different placement rather than a near
    duplicate.  Neither rule reads the reference.
    """
    records = []
    for case, distances, poses in case_blocks(bundle, features):
        scores = poses[score_column].to_numpy(float)
        rmsd = poses["reference_rmsd_A"].to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError(f"{case}: non-finite {score_column}")
        for budget in ALL_BUDGETS:
            block = distances[:budget, :budget]
            top = int(np.argmax(scores[:budget]))
            far = np.flatnonzero(block[top] >= CLUSTER_CUTOFF_A)
            if len(far):
                second = int(far[int(np.argmax(scores[far]))])
                best_of_two = min(float(rmsd[top]), float(rmsd[second]))
            else:
                second = -1
                best_of_two = float(rmsd[top])
            records.append({
                "complex_name": case,
                "score_column": score_column,
                "budget_runs": budget,
                "chosen_run": top + 1,
                "second_run": second + 1 if second >= 0 else 0,
                "second_available": second >= 0,
                "top1_rmsd_A": float(rmsd[top]),
                "top2_distinct_best_rmsd_A": best_of_two,
            })
    return pd.DataFrame.from_records(records)


def mode_medoid_selection(bundle: Path, features: pd.DataFrame) -> pd.DataFrame:
    """Alternative consensus rule: medoid of the best reproducible mode."""
    run1_rmsd = features.set_index("complex_name")["run1_rmsd_A"]
    records = []
    for case, distances, poses in case_blocks(bundle, features):
        ncc = poses["multiscale_ncc"].to_numpy(float)
        rmsd = poses["reference_rmsd_A"].to_numpy(float)
        fallback = float(run1_rmsd.loc[case])
        for budget in ALL_BUDGETS:
            block = distances[:budget, :budget]
            labels = stable_labels(block)
            ranked = []
            for mode in np.unique(labels):
                members = np.flatnonzero(labels == mode)
                if len(members) < 2:
                    continue
                ranked.append({
                    "median_ncc": float(np.median(ncc[members])),
                    "size": int(len(members)),
                    "first": int(members[0]),
                    "medoid": medoid_index(block, members),
                })
            ranked.sort(
                key=lambda item: (-item["median_ncc"], -item["size"], item["first"])
            )
            if ranked:
                top1 = float(rmsd[ranked[0]["medoid"]])
                top2 = min(float(rmsd[item["medoid"]]) for item in ranked[:2])
            else:
                top1 = top2 = fallback
            records.append({
                "complex_name": case,
                "budget_runs": budget,
                "reproducible_mode_n": len(ranked),
                "top1_medoid_rmsd_A": top1,
                "top2_medoid_best_rmsd_A": top2,
                "used_run1_fallback": not ranked,
            })
    return pd.DataFrame.from_records(records)


def score_sweep(bundle: Path, features: pd.DataFrame) -> pd.DataFrame:
    """Success of cross-run reranking under every candidate selection score."""
    cohort_n = int(features.shape[0])
    rows = []
    for column in SCORE_COLUMNS:
        selection = rerank_selection(bundle, features, column)
        for budget in ALL_BUDGETS:
            block = selection.loc[selection["budget_runs"] == budget]
            for target, answers in (
                ("top1_rmsd_A", 1),
                ("top2_distinct_best_rmsd_A", 2),
            ):
                success = int((block[target] < SUCCESS_CUTOFF_A).sum())
                low, high = wilson_interval(success, cohort_n)
                rows.append({
                    "score_column": column,
                    "is_frozen_pipeline_score": column == PRIMARY_SCORE,
                    "budget_runs": budget,
                    "reported_answers_per_case": answers,
                    "success_n": success,
                    "cohort_n": cohort_n,
                    "success_pct": 100.0 * success / cohort_n,
                    "success_95ci_low_pct": low,
                    "success_95ci_high_pct": high,
                })
    return pd.DataFrame(rows)


def score_discrimination(bundle: Path, features: pd.DataFrame) -> pd.DataFrame:
    """Within-case Spearman correlation between each score and pose RMSD."""
    from scipy.stats import spearmanr

    records = {column: [] for column in SCORE_COLUMNS}
    for _, _, poses in case_blocks(bundle, features):
        for column in SCORE_COLUMNS:
            values = poses[column].to_numpy(float)
            rmsd = poses["reference_rmsd_A"].to_numpy(float)
            if len(np.unique(values[np.isfinite(values)])) < 2:
                continue
            correlation = spearmanr(values, rmsd).correlation
            if np.isfinite(correlation):
                records[column].append(float(correlation))
    rows = []
    for column, values in records.items():
        array = np.asarray(values, dtype=float)
        rows.append({
            "score_column": column,
            "is_frozen_pipeline_score": column == PRIMARY_SCORE,
            "cases_n": int(len(array)),
            "mean_spearman_vs_rmsd": float(array.mean()),
            "median_spearman_vs_rmsd": float(np.median(array)),
        })
    return pd.DataFrame(rows).sort_values("mean_spearman_vs_rmsd")


def build_panel_c(
    bundle: Path,
    features: pd.DataFrame,
    inference_minutes: float,
    paahs_seconds: float,
    speed_accuracy_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    selection = rerank_selection(bundle, features, PRIMARY_SCORE)
    budget_table = pd.read_csv(
        bundle / "04_figure5_paahs_pose_modes_v1"
        / "mode_discovery_and_rescue_by_budget.csv"
    ).set_index("budget_runs")
    per_run_minutes = inference_minutes + paahs_seconds / 60.0
    cohort_n = int(features.shape[0])

    rows = []
    for budget in ALL_BUDGETS:
        block = selection.loc[selection["budget_runs"] == budget]
        if len(block) != cohort_n:
            raise ValueError(f"budget {budget}: {len(block)} cases, expected {cohort_n}")
        minutes = budget * per_run_minutes
        for column, answers, name in (
            ("top1_rmsd_A", 1, "One reported pose"),
            ("top2_distinct_best_rmsd_A", 2, "Two distinct reported poses"),
        ):
            if answers == 2 and budget == 1:
                continue
            success = int((block[column] < SUCCESS_CUTOFF_A).sum())
            low, high = wilson_interval(success, cohort_n)
            rows.append({
                "series": name,
                "kind": "deployable",
                "budget_runs": budget,
                "reported_answers_per_case": answers,
                "success_n": success,
                "cohort_n": cohort_n,
                "success_pct": 100.0 * success / cohort_n,
                "success_95ci_low_pct": low,
                "success_95ci_high_pct": high,
                "minutes_per_case": minutes,
                "second_pose_available_n": int(block["second_available"].sum()),
                "reported_a_later_run_n": int((block["chosen_run"] != 1).sum()),
            })
        oracle = int(budget_table.loc[budget, "best_of_budget_success_n"])
        low, high = wilson_interval(oracle, cohort_n)
        rows.append({
            "series": "Best of the k runs",
            "kind": "retrospective",
            "budget_runs": budget,
            "reported_answers_per_case": budget,
            "success_n": oracle,
            "cohort_n": cohort_n,
            "success_pct": 100.0 * oracle / cohort_n,
            "success_95ci_low_pct": low,
            "success_95ci_high_pct": high,
            "minutes_per_case": minutes,
            "second_pose_available_n": 0,
            "reported_a_later_run_n": 0,
        })

    speed = pd.read_csv(speed_accuracy_path)
    chemem = speed.loc[speed["method"] == "ChemEM"]
    if chemem.empty:
        raise ValueError(f"ChemEM missing from {speed_accuracy_path}")
    chemem = chemem.iloc[0]
    chemem_n = int(round(float(chemem["success_pct"]) / 100.0 * cohort_n))
    rows.append({
        "series": "ChemEM",
        "kind": "deployable",
        "budget_runs": 1,
        "reported_answers_per_case": 1,
        "success_n": chemem_n,
        "cohort_n": cohort_n,
        "success_pct": float(chemem["success_pct"]),
        "success_95ci_low_pct": float(chemem["wilson_95ci_low_pct"]),
        "success_95ci_high_pct": float(chemem["wilson_95ci_high_pct"]),
        "minutes_per_case": float(chemem["mean_runtime_min"]),
        "second_pose_available_n": 0,
        "reported_a_later_run_n": 0,
    })
    panel = pd.DataFrame(rows)

    frozen = speed.loc[speed["method"] == "CryoDiffDock-Local"]
    consistency = {}
    if not frozen.empty:
        frozen = frozen.iloc[0]
        budget1 = panel.loc[
            (panel["series"] == "One reported pose") & (panel["budget_runs"] == 1)
        ].iloc[0]
        consistency = {
            "figure2_local_minutes_per_case": float(frozen["mean_runtime_min"]),
            "panel_c_budget1_minutes_per_case": float(budget1["minutes_per_case"]),
            "figure2_local_success_pct": float(frozen["success_pct"]),
            "panel_c_budget1_success_pct": float(budget1["success_pct"]),
        }
        if abs(float(frozen["mean_runtime_min"]) - float(budget1["minutes_per_case"])) > 1e-3:
            raise ValueError(
                "Panel C 1-run cost disagrees with the frozen Figure 2 operating point"
            )
        if abs(float(frozen["success_pct"]) - float(budget1["success_pct"])) > 1e-6:
            raise ValueError(
                "Panel C 1-run accuracy disagrees with the frozen Figure 2 point"
            )
    return panel, selection, consistency


def build_panel_d(bundle: Path) -> pd.DataFrame:
    joint = bundle / "08_figure5_paahs_joint_density_pb_multimodal72_v2"
    counts = pd.read_csv(joint / "panel_e_multimodal_joint_counts.csv")
    rows = []
    for row in counts.itertuples(index=False):
        low, high = wilson_interval(int(row.cases_n), int(row.multimodal_n))
        rows.append({
            "category": row.category,
            "cases_n": int(row.cases_n),
            "multimodal_n": int(row.multimodal_n),
            "pct_of_multimodal": float(row.pct_of_multimodal),
            "pct_95ci_low": low,
            "pct_95ci_high": high,
            "ncc_tolerance": float(row.ncc_tolerance),
        })
    return pd.DataFrame(rows)


def build_supplements(bundle: Path, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    modes = bundle / "04_figure5_paahs_pose_modes_v1"
    joint = bundle / "08_figure5_paahs_joint_density_pb_multimodal72_v2"

    cohort = pd.read_csv(joint / "panel_e_joint_density_pb_counts.csv")
    rows = []
    for row in cohort.itertuples(index=False):
        low, high = wilson_interval(int(row.cases_n), int(row.cohort_n))
        rows.append({
            "category": row.category,
            "cases_n": int(row.cases_n),
            "cohort_n": int(row.cohort_n),
            "pct_of_cohort": float(row.pct_of_cohort),
            "pct_95ci_low": low,
            "pct_95ci_high": high,
            "ncc_tolerance": float(row.ncc_tolerance),
        })
    full_cohort = pd.DataFrame(rows)

    tolerance = pd.read_csv(joint / "supplement_ncc_tolerance_sensitivity.csv")

    per_mode = pd.read_csv(joint / "per_mode_joint_density_pb.csv")
    keep = [
        "complex_name", "mode_rank", "mode_id", "mode_size", "medoid_replicate",
        "medoid_final_multiscale_ncc", "density_ncc_gap_from_best",
        "near_best_density", "pb_valid", "joint_density_pb_supported",
    ]
    medoids = per_mode[keep].copy()
    wide = medoids.pivot(index="complex_name", columns="mode_rank")
    wide.columns = [f"mode{int(rank)}_{name}" for name, rank in wide.columns]
    wide = wide.reset_index()
    wide["joint_supported_modes"] = (
        medoids.groupby("complex_name")["joint_density_pb_supported"].sum().values
    )

    geometry = pd.read_csv(modes / "mode_pair_geometry.csv")
    ranked = medoids.pivot(index="complex_name", columns="mode_rank", values="mode_id")
    ranked.columns = [f"mode_id_rank{int(rank)}" for rank in ranked.columns]
    ranked = ranked.reset_index()
    ranked["mode_left"] = ranked[["mode_id_rank1", "mode_id_rank2"]].min(axis=1)
    ranked["mode_right"] = ranked[["mode_id_rank1", "mode_id_rank2"]].max(axis=1)
    separation = ranked.merge(
        geometry[[
            "complex_name", "mode_left", "mode_right", "medoid_symmetry_rmsd_A",
            "medoid_centroid_distance_A", "absolute_delta_multiscale_ncc_median",
        ]],
        on=["complex_name", "mode_left", "mode_right"],
        how="left",
        validate="one_to_one",
    )
    if separation["medoid_symmetry_rmsd_A"].isna().any():
        missing = separation.loc[
            separation["medoid_symmetry_rmsd_A"].isna(), "complex_name"
        ].tolist()
        raise ValueError(f"No mode-pair geometry for {missing}")
    separation = separation.merge(
        wide[["complex_name", "joint_supported_modes"]], on="complex_name", how="left"
    )

    discovery = pd.read_csv(modes / "mode_discovery_and_rescue_by_budget.csv")
    agreement = pd.read_csv(modes / "agreement_level_rescue_summary.csv")

    labels = features["run1_failure"].to_numpy(bool)
    rows = []
    for budget in ALL_BUDGETS:
        scores = features[f"disagreement_at_{budget}"].to_numpy(float)
        order = np.argsort(scores, kind="stable")
        quartiles = np.array_split(order, 4)
        for index, block in enumerate(quartiles, start=1):
            failures = int(labels[block].sum())
            low, high = wilson_interval(failures, len(block))
            rows.append({
                "budget_runs": budget,
                "feature_family": "within_run_pool" if budget == 1 else "cross_run",
                "uncertainty_quartile": index,
                "cases_n": int(len(block)),
                "failure_n": failures,
                "failure_pct": 100.0 * failures / len(block),
                "failure_95ci_low_pct": low,
                "failure_95ci_high_pct": high,
                "mean_uncertainty": float(np.mean(scores[block])),
            })
    quartile_risk = pd.DataFrame(rows)

    return {
        "s_full_cohort_joint_modes": full_cohort,
        "s_ncc_tolerance_sensitivity": tolerance,
        "s_mode_medoid_ncc_pairs": wide,
        "s_top_two_mode_separation": separation,
        "s_mode_discovery_by_budget": discovery,
        "s_agreement_level_rescue": agreement,
        "s_uncertainty_quartile_risk": quartile_risk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--single-run-pool", type=Path, required=True)
    parser.add_argument("--figure2-runtime", type=Path, required=True)
    parser.add_argument("--figure2-speed-accuracy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paahs-seconds-per-case", type=float, default=21.28)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    arguments = parser.parse_args()

    bundle = arguments.bundle.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "supplements").mkdir(exist_ok=True)

    features = build_features(bundle, arguments.single_run_pool.resolve())
    panel_a = build_panel_a(features, arguments.bootstrap_repeats, arguments.seed)
    panel_b = build_panel_b(features)
    inference_minutes = figure2_runtime_minutes(arguments.figure2_runtime.resolve())
    wall_clock_minutes, timing_records = replicate_wall_clock_minutes(bundle)
    panel_c, selection, consistency = build_panel_c(
        bundle,
        features,
        inference_minutes,
        arguments.paahs_seconds_per_case,
        arguments.figure2_speed_accuracy.resolve(),
    )
    panel_d = build_panel_d(bundle)
    supplements = build_supplements(bundle, features)

    features.to_csv(output / "per_case_disagreement_features.csv", index=False)
    panel_a.to_csv(output / "panel_a_failure_auroc.csv", index=False)
    panel_b.to_csv(output / "panel_b_risk_coverage.csv", index=False)
    panel_c.to_csv(output / "panel_c_accuracy_vs_compute.csv", index=False)
    selection.to_csv(
        output / "supplements" / "s_per_case_cross_run_reranking.csv", index=False
    )
    mode_medoid_selection(bundle, features).to_csv(
        output / "supplements" / "s_per_case_mode_medoid_selection.csv", index=False
    )
    score_sweep(bundle, features).to_csv(
        output / "supplements" / "s_cross_run_reranking_score_sweep.csv", index=False
    )
    score_discrimination(bundle, features).to_csv(
        output / "supplements" / "s_score_within_case_discrimination.csv", index=False
    )
    panel_d.to_csv(output / "panel_d_joint_density_pb_modes.csv", index=False)
    for name, frame in supplements.items():
        frame.to_csv(output / "supplements" / f"{name}.csv", index=False)

    provenance = {
        "bundle": str(bundle),
        "cohort_n": int(len(features)),
        "run1_failure_n": int(features["run1_failure"].sum()),
        "budgets": list(ALL_BUDGETS),
        "cluster_cutoff_A": CLUSTER_CUTOFF_A,
        "success_cutoff_A": SUCCESS_CUTOFF_A,
        "outcome": "Run-1 PAAHS pose symmetry-aware RMSD >= 2 A",
        "budget1_feature": {
            "name": POOL_FEATURE,
            "definition": (
                "one minus the largest symmetry-aware 2 A cluster fraction of the "
                "40-pose Run-1 sampling pool"
            ),
            "scope": "within a single run; no second run is required",
            "source": str(arguments.single_run_pool.resolve()),
            "pool_pose_count": int(features["pool_pose_count"].iloc[0]),
        },
        "cross_run_feature": {
            "name": CROSS_RUN_FEATURE,
            "definition": (
                "one minus the fraction of the first k reported PAAHS Top-1 poses in "
                "the symmetry-aware complete-linkage 2 A cluster containing the "
                "Run-1 pose"
            ),
            "scope": "budgets 2, 4 and 8",
            "note": (
                "undefined at budget 1, where the Run-1 pose is trivially alone; "
                "the budget-1 point therefore uses the within-run pool feature and "
                "is not the same quantity"
            ),
        },
        "inference_minutes_per_case": inference_minutes,
        "inference_timing_source": str(arguments.figure2_runtime.resolve()),
        "replicate_wall_clock_minutes_per_case_median": wall_clock_minutes,
        "replicate_wall_clock_records": timing_records,
        "paahs_seconds_per_case": arguments.paahs_seconds_per_case,
        "paahs_timing_note": (
            "fixed average per case; the measured PAAHS wall clock was disturbed by "
            "a server fault and is not used"
        ),
        "figure2_consistency_check": consistency,
        "bootstrap_repeats": arguments.bootstrap_repeats,
        "seed": arguments.seed,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(panel_a.to_string(index=False))
    print()
    print(panel_c.to_string(index=False))


if __name__ == "__main__":
    main()
