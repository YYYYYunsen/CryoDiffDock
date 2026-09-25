#!/usr/bin/env python3
"""Audit reproducible pose modes across independent Local V2 runs.

This analysis separates two questions that must not be conflated:

1. Reference-free: do independently selected Top-1 poses form one or more
   reproducible structural modes, and do those modes receive comparable
   density/physics support?
2. Post-reveal: when the first run fails the deposited-reference RMSD
   criterion, does a later run recover a successful pose or pose cluster?

The frozen defaults are complete-linkage clustering at 2 A symmetry-aware
heavy-atom RMSD and at least two independent runs per reproducible mode.
Reference coordinates are never used for clustering, mode quality, or the
reference-free candidate ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_multirun_false_consensus import (
    INTERACTION_NUMERIC_COLUMNS,
    RANKING_BOOLEAN_COLUMNS,
    RANKING_NUMERIC_COLUMNS,
    as_bool,
    select_ranking_row,
)
from evaluation.audit_pose_ensemble_uncertainty import (
    ordered_pairwise_rmsd,
    symmetry_pairwise_rmsd,
)
from evaluation.evaluate_ligand_accuracy import (
    centroid_distance,
    read_heavy_molecule,
    symmetry_rmsd,
)


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
REFERENCE_COLUMNS = (
    "reference_ligand",
    "reference_ligand_path",
    "ground_truth_ligand",
)
EXTRA_RANKING_NUMERIC_COLUMNS = (
    "selection_score",
    "ncc_sigma_1p5",
    "ncc_sigma_2p25",
)
QUALITY_NUMERIC_COLUMNS = tuple(dict.fromkeys(
    (*RANKING_NUMERIC_COLUMNS, *EXTRA_RANKING_NUMERIC_COLUMNS,
     *INTERACTION_NUMERIC_COLUMNS)
))


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; found {list(columns)}")


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return values


def parse_budgets(value: str) -> list[int]:
    budgets = [int(item) for item in value.split(",") if item.strip()]
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] != 1:
        raise argparse.ArgumentTypeError(
            "budgets must be sorted unique positive integers beginning with 1"
        )
    return budgets


def stable_complete_linkage_labels(
    distances: np.ndarray, cutoff_A: float
) -> np.ndarray:
    """Return 1-based labels ordered by each cluster's earliest member.

    Stable relabelling makes outputs deterministic even though scipy's raw
    cluster integers are implementation details.
    """
    distances = np.asarray(distances, dtype=float)
    count = len(distances)
    if distances.shape != (count, count) or count == 0:
        raise ValueError("Pairwise distance matrix must be non-empty and square")
    if not np.isfinite(distances).all():
        raise ValueError("Pairwise distance matrix contains non-finite values")
    if cutoff_A <= 0:
        raise ValueError("Cluster cutoff must be positive")
    if count == 1:
        return np.ones(1, dtype=int)
    hierarchy = linkage(squareform(distances, checks=False), method="complete")
    raw = fcluster(hierarchy, t=cutoff_A, criterion="distance")
    ordered_raw = sorted(set(raw), key=lambda value: int(np.flatnonzero(raw == value)[0]))
    relabel = {value: index + 1 for index, value in enumerate(ordered_raw)}
    return np.asarray([relabel[value] for value in raw], dtype=int)


def medoid_index(distances: np.ndarray, member_indices: np.ndarray) -> int:
    member_indices = np.asarray(member_indices, dtype=int)
    if not len(member_indices):
        raise ValueError("Cannot choose a medoid from an empty cluster")
    submatrix = distances[np.ix_(member_indices, member_indices)]
    mean_distance = submatrix.mean(axis=1)
    return int(member_indices[int(np.argmin(mean_distance))])


def numeric_summary(values: pd.Series, prefix: str) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return {
            f"{prefix}_n": 0,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
        }
    return {
        f"{prefix}_n": int(len(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
    }


def load_references(path: Path, cohort: set[str]) -> dict[str, Path]:
    table = pd.read_csv(path, dtype=str)
    id_column = choose_column(table.columns, ID_COLUMNS, "reference-manifest ID")
    reference_column = choose_column(
        table.columns, REFERENCE_COLUMNS, "reference ligand"
    )
    table = table.copy()
    table["_complex_name"] = table[id_column].map(normalize_id)
    if table["_complex_name"].duplicated().any():
        duplicate = sorted(
            table.loc[table["_complex_name"].duplicated(False), "_complex_name"].unique()
        )
        raise ValueError(f"Reference manifest contains duplicate IDs: {duplicate[:20]}")
    missing = sorted(cohort - set(table["_complex_name"]))
    if missing:
        raise ValueError(f"Reference manifest lacks cohort cases: {missing[:20]}")
    result = {}
    for _, row in table.iterrows():
        complex_name = row["_complex_name"]
        if complex_name in cohort:
            result[complex_name] = resolve_path(row[reference_column], path.parent)
    return result


def augment_selected_rows(
    selected_path: Path,
    cohort_ids: list[str],
    budget: int,
    score_interactions: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    selected = pd.read_csv(selected_path, dtype=str)
    required = {"complex_name", "replicate", "selected_pose_path", "ranking_path"}
    missing_columns = sorted(required - set(selected.columns))
    if missing_columns:
        raise ValueError(f"Selected-pose table lacks columns: {missing_columns}")
    selected["complex_name"] = selected["complex_name"].map(normalize_id)
    if "replicate_order" not in selected.columns:
        order = {
            name: index + 1
            for index, name in enumerate(selected["replicate"].drop_duplicates())
        }
        selected["replicate_order"] = selected["replicate"].map(order)
    selected["replicate_order"] = pd.to_numeric(
        selected["replicate_order"], errors="raise"
    ).astype(int)
    selected = selected[selected["replicate_order"].le(budget)].copy()
    cohort = set(cohort_ids)
    found = set(selected["complex_name"])
    if found != cohort:
        raise ValueError(
            "Selected-pose table does not match cohort: "
            f"missing={sorted(cohort-found)[:20]}, extra={sorted(found-cohort)[:20]}"
        )
    counts = selected.groupby("complex_name").size()
    bad = counts[counts.ne(budget)]
    if len(bad):
        raise ValueError(
            "Selected-pose replicate count mismatch: "
            + ", ".join(f"{name}={value}" for name, value in bad.head(20).items())
        )

    interaction_failures: list[dict[str, str]] = []
    protein_cache: dict[str, object] = {}
    if score_interactions:
        from cryodiffdock_pocket.interactions import (
            read_protein_interaction_sites,
            score_pose_interactions,
        )
    records = []
    for raw in selected.to_dict("records"):
        ranking_path = resolve_path(raw["ranking_path"], selected_path.parent)
        pose_path = resolve_path(raw["selected_pose_path"], selected_path.parent)
        if not pose_path.is_file():
            raise FileNotFoundError(f"Selected pose does not exist: {pose_path}")
        row = select_ranking_row(ranking_path)
        if normalize_id(row.get("complex_name")) != raw["complex_name"]:
            raise ValueError(f"Ranking case mismatch: {ranking_path}")
        record = dict(raw)
        record["ranking_path"] = str(ranking_path)
        record["selected_pose_path"] = str(pose_path)
        for column in QUALITY_NUMERIC_COLUMNS:
            # A post-processing audit may provide recomputed values (for
            # example NCC for the final PAAHS coordinates) directly in the
            # selected-pose table. Prefer those over the historical ranking
            # row while preserving the original behavior when absent.
            value = raw.get(column, np.nan)
            if pd.isna(value):
                value = row.get(column, np.nan)
            record[column] = pd.to_numeric(value, errors="coerce")
        for column in RANKING_BOOLEAN_COLUMNS:
            record[column] = as_bool(row.get(column, np.nan))
        protein_path = resolve_path(row.get("protein_path", ""), ranking_path.parent)
        record["protein_path"] = str(protein_path)
        if score_interactions:
            try:
                cache_key = str(protein_path)
                if cache_key not in protein_cache:
                    protein_cache[cache_key] = read_protein_interaction_sites(protein_path)
                scores = score_pose_interactions(
                    protein_cache[cache_key], pose_path
                )
                for column in INTERACTION_NUMERIC_COLUMNS:
                    record[column] = pd.to_numeric(
                        scores.get(column, np.nan), errors="coerce"
                    )
            except Exception as exc:
                interaction_failures.append({
                    "complex_name": raw["complex_name"],
                    "replicate": str(raw["replicate"]),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                for column in INTERACTION_NUMERIC_COLUMNS:
                    record[column] = np.nan
        records.append(record)
    result = pd.DataFrame(records)
    cohort_order = {name: index for index, name in enumerate(cohort_ids)}
    result["cohort_order"] = result["complex_name"].map(cohort_order)
    result = result.sort_values(
        ["cohort_order", "replicate_order"], kind="stable"
    ).reset_index(drop=True)
    return result, interaction_failures


def summarize_agreement_rescue(
    per_case: pd.DataFrame, budget: int
) -> pd.DataFrame:
    rows = []
    for cluster_size, group in per_case.groupby("run1_cluster_size", sort=True):
        failures = group[~group["run1_success"]]
        rows.append({
            "run1_cluster_size": int(cluster_size),
            "agreement_label": f"{int(cluster_size)}/{budget}",
            "case_n": int(len(group)),
            "run1_failure_n": int((~group["run1_success"]).sum()),
            "run1_failure_pct": float(100.0 * (~group["run1_success"]).mean()),
            "later_run_rescue_n": int(failures["rescued_by_later_run"].sum()),
            "later_run_rescue_pct_among_run1_failures": (
                float(100.0 * failures["rescued_by_later_run"].mean())
                if len(failures) else np.nan
            ),
            "all_runs_failure_n": int(group["all_runs_failure"].sum()),
            "median_reproducible_mode_count": float(
                group["reproducible_mode_count"].median()
            ),
        })
    return pd.DataFrame(rows).sort_values("run1_cluster_size", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-pose-replicates", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", type=parse_budgets, default=[1, 2, 4, 8])
    parser.add_argument("--cluster-cutoff", type=float, default=2.0)
    parser.add_argument("--minimum-replicates-per-mode", type=int, default=2)
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument(
        "--pairwise-rmsd-mode", choices=("ordered", "symmetry"), default="symmetry"
    )
    parser.add_argument("--score-interactions", action="store_true")
    parser.add_argument("--expected-n", type=int)
    args = parser.parse_args()

    if not np.isclose(args.cluster_cutoff, 2.0):
        raise ValueError("This frozen audit requires the prespecified 2.0 A cutoff")
    if args.minimum_replicates_per_mode != 2:
        raise ValueError("This frozen audit defines a reproducible mode as >=2 runs")
    if args.success_cutoff <= 0:
        raise ValueError("success-cutoff must be positive")
    budget = max(args.budgets)
    selected_path = args.selected_pose_replicates.expanduser().resolve()
    cohort_path = args.cohort.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not selected_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError("Selected-pose table or cohort does not exist")
    cohort_ids = read_ids(cohort_path)
    if args.expected_n is not None and len(cohort_ids) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cases, found {len(cohort_ids)}")
    if budget < 2:
        raise ValueError("At least two runs are required for a mode audit")

    references = None
    reference_path = None
    if args.reference_manifest is not None:
        reference_path = args.reference_manifest.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(reference_path)
        references = load_references(reference_path, set(cohort_ids))

    selected, interaction_failures = augment_selected_rows(
        selected_path, cohort_ids, budget, args.score_interactions
    )
    pose_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    cluster_pair_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    budget_case_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for complex_name in cohort_ids:
        group = selected[selected["complex_name"].eq(complex_name)].sort_values(
            "replicate_order", kind="stable"
        ).reset_index(drop=True)
        try:
            molecules = [
                read_heavy_molecule(Path(path)) for path in group["selected_pose_path"]
            ]
            distances = (
                symmetry_pairwise_rmsd(molecules)
                if args.pairwise_rmsd_mode == "symmetry"
                else ordered_pairwise_rmsd(molecules)
            )
            labels = stable_complete_linkage_labels(distances, args.cluster_cutoff)
            group = group.copy()
            group["mode_id"] = labels
            reference = (
                read_heavy_molecule(references[complex_name])
                if references is not None else None
            )
            if reference is not None:
                group["reference_rmsd_A"] = [
                    symmetry_rmsd(reference, molecule) for molecule in molecules
                ]
                group["reference_success"] = group["reference_rmsd_A"].lt(
                    args.success_cutoff
                )
            else:
                group["reference_rmsd_A"] = np.nan
                group["reference_success"] = pd.NA

            for left in range(budget):
                for right in range(left + 1, budget):
                    pair_rows.append({
                        "complex_name": complex_name,
                        "replicate_left": group.iloc[left]["replicate"],
                        "replicate_right": group.iloc[right]["replicate"],
                        "mode_left": int(labels[left]),
                        "mode_right": int(labels[right]),
                        "same_mode": bool(labels[left] == labels[right]),
                        "pairwise_rmsd_A": float(distances[left, right]),
                    })

            local_cluster_rows = []
            medoid_by_mode: dict[int, int] = {}
            for mode_id in sorted(set(labels)):
                member_indices = np.flatnonzero(labels == mode_id)
                medoid = medoid_index(distances, member_indices)
                medoid_by_mode[int(mode_id)] = medoid
                subset = group.iloc[member_indices]
                record: dict[str, object] = {
                    "complex_name": complex_name,
                    "mode_id": int(mode_id),
                    "mode_size": int(len(member_indices)),
                    "mode_fraction": float(len(member_indices) / budget),
                    "reproducible_mode": bool(
                        len(member_indices) >= args.minimum_replicates_per_mode
                    ),
                    "replicates": "|".join(subset["replicate"].astype(str)),
                    "replicate_orders": "|".join(
                        subset["replicate_order"].astype(int).astype(str)
                    ),
                    "medoid_replicate": group.iloc[medoid]["replicate"],
                    "medoid_pose_path": group.iloc[medoid]["selected_pose_path"],
                    "within_mode_pairwise_rmsd_max_A": (
                        float(distances[np.ix_(member_indices, member_indices)].max())
                    ),
                }
                for column in QUALITY_NUMERIC_COLUMNS:
                    if column in subset.columns:
                        record.update(numeric_summary(subset[column], column))
                for column in RANKING_BOOLEAN_COLUMNS:
                    values = subset[column].dropna().astype(bool)
                    record[f"{column}_n"] = int(len(values))
                    record[f"{column}_fraction"] = (
                        float(values.mean()) if len(values) else np.nan
                    )
                if reference is not None:
                    rmsd_values = pd.to_numeric(
                        subset["reference_rmsd_A"], errors="coerce"
                    )
                    record.update({
                        "reference_min_rmsd_A": float(rmsd_values.min()),
                        "reference_median_rmsd_A": float(rmsd_values.median()),
                        "medoid_reference_rmsd_A": float(
                            group.iloc[medoid]["reference_rmsd_A"]
                        ),
                        "contains_reference_success": bool(
                            rmsd_values.lt(args.success_cutoff).any()
                        ),
                    })
                local_cluster_rows.append(record)
                cluster_rows.append(record)

            cluster_table = pd.DataFrame(local_cluster_rows)
            cluster_lookup = cluster_table.set_index("mode_id")
            for left_mode in sorted(medoid_by_mode):
                for right_mode in sorted(medoid_by_mode):
                    if right_mode <= left_mode:
                        continue
                    left_index = medoid_by_mode[left_mode]
                    right_index = medoid_by_mode[right_mode]
                    left_molecule = molecules[left_index]
                    right_molecule = molecules[right_index]
                    pair_record: dict[str, object] = {
                        "complex_name": complex_name,
                        "mode_left": left_mode,
                        "mode_right": right_mode,
                        "medoid_symmetry_rmsd_A": float(
                            symmetry_rmsd(left_molecule, right_molecule)
                        ),
                        "medoid_centroid_distance_A": float(
                            centroid_distance(left_molecule, right_molecule)
                        ),
                        "medoid_centered_rmsd_A": float(
                            symmetry_rmsd(
                                left_molecule, right_molecule, center=True
                            )
                        ),
                        "medoid_rigid_aligned_rmsd_A": float(
                            symmetry_rmsd(
                                left_molecule,
                                right_molecule,
                                center=True,
                                minimize=True,
                            )
                        ),
                    }
                    for quality_column in (
                        "multiscale_ncc_median",
                        "ncc_median",
                        "final_selection_score_median",
                        "in_bounds_fraction_median",
                        "relative_physics_pass_fraction",
                    ):
                        if quality_column not in cluster_lookup.columns:
                            continue
                        left_value = pd.to_numeric(
                            cluster_lookup.at[left_mode, quality_column],
                            errors="coerce",
                        )
                        right_value = pd.to_numeric(
                            cluster_lookup.at[right_mode, quality_column],
                            errors="coerce",
                        )
                        pair_record[f"left_{quality_column}"] = left_value
                        pair_record[f"right_{quality_column}"] = right_value
                        pair_record[f"absolute_delta_{quality_column}"] = (
                            float(abs(left_value - right_value))
                            if np.isfinite(left_value) and np.isfinite(right_value)
                            else np.nan
                        )
                    cluster_pair_rows.append(pair_record)

            reproducible = cluster_table[cluster_table["reproducible_mode"]].copy()
            run1_mode = int(labels[0])
            run1_cluster_size = int(np.sum(labels == run1_mode))
            case: dict[str, object] = {
                "complex_name": complex_name,
                "run_count": budget,
                "mode_count": int(len(cluster_table)),
                "reproducible_mode_count": int(len(reproducible)),
                "singleton_mode_count": int((cluster_table["mode_size"] == 1).sum()),
                "multiple_reproducible_modes": bool(len(reproducible) >= 2),
                "largest_mode_size": int(cluster_table["mode_size"].max()),
                "largest_mode_fraction": float(cluster_table["mode_fraction"].max()),
                "run1_mode_id": run1_mode,
                "run1_cluster_size": run1_cluster_size,
                "run1_cluster_fraction": float(run1_cluster_size / budget),
                "run1_disagreement": float(1.0 - run1_cluster_size / budget),
            }
            for column in ("multiscale_ncc", "ncc", "final_selection_score"):
                median_column = f"{column}_median"
                if median_column in reproducible.columns:
                    values = pd.to_numeric(
                        reproducible[median_column], errors="coerce"
                    ).dropna().sort_values(ascending=False)
                    case[f"best_reproducible_mode_{column}_median"] = (
                        float(values.iloc[0]) if len(values) else np.nan
                    )
                    case[f"second_reproducible_mode_{column}_median"] = (
                        float(values.iloc[1]) if len(values) >= 2 else np.nan
                    )
                    case[f"top_two_reproducible_mode_{column}_gap"] = (
                        float(values.iloc[0] - values.iloc[1])
                        if len(values) >= 2 else np.nan
                    )
            if "in_bounds_fraction_median" in reproducible.columns:
                case["minimum_reproducible_mode_median_in_bounds_fraction"] = float(
                    pd.to_numeric(
                        reproducible["in_bounds_fraction_median"], errors="coerce"
                    ).min()
                )
            if "relative_physics_pass_fraction" in reproducible.columns:
                case["minimum_reproducible_mode_physics_pass_fraction"] = float(
                    pd.to_numeric(
                        reproducible["relative_physics_pass_fraction"], errors="coerce"
                    ).min()
                )
            if reference is not None:
                rmsds = pd.to_numeric(group["reference_rmsd_A"], errors="raise")
                run1_success = bool(rmsds.iloc[0] < args.success_cutoff)
                later_success = bool(rmsds.iloc[1:].lt(args.success_cutoff).any())
                best_index = int(rmsds.to_numpy().argmin())
                case.update({
                    "run1_rmsd_A": float(rmsds.iloc[0]),
                    "run1_success": run1_success,
                    "best_of_runs_rmsd_A": float(rmsds.iloc[best_index]),
                    "best_run_replicate": group.iloc[best_index]["replicate"],
                    "any_run_success": bool(rmsds.lt(args.success_cutoff).any()),
                    "rescued_by_later_run": bool(not run1_success and later_success),
                    "all_runs_failure": bool(rmsds.ge(args.success_cutoff).all()),
                    "successful_run_count": int(rmsds.lt(args.success_cutoff).sum()),
                    "successful_mode_count": int(
                        cluster_table.get(
                            "contains_reference_success", pd.Series(dtype=bool)
                        ).fillna(False).astype(bool).sum()
                    ),
                    "earliest_successful_run": (
                        int(group.loc[rmsds.lt(args.success_cutoff), "replicate_order"].min())
                        if rmsds.lt(args.success_cutoff).any() else np.nan
                    ),
                })
            else:
                for column in (
                    "run1_success", "any_run_success", "rescued_by_later_run",
                    "all_runs_failure",
                ):
                    case[column] = pd.NA
            case_rows.append(case)

            for _, row in group.iterrows():
                pose_rows.append(row.to_dict())

            for current_budget in args.budgets:
                subset_labels = labels[:current_budget]
                discovered = set(int(value) for value in subset_labels)
                full_reproducible_modes = set(
                    reproducible["mode_id"].astype(int).tolist()
                )
                row: dict[str, object] = {
                    "complex_name": complex_name,
                    "budget_runs": current_budget,
                    "full_budget_modes_discovered": int(len(discovered)),
                    "full_budget_reproducible_modes_discovered": int(
                        len(discovered & full_reproducible_modes)
                    ),
                    "all_full_budget_modes_discovered": bool(
                        len(discovered) == len(cluster_table)
                    ),
                    "all_full_budget_reproducible_modes_discovered": bool(
                        full_reproducible_modes.issubset(discovered)
                    ),
                }
                if reference is not None:
                    prefix_rmsd = pd.to_numeric(
                        group.iloc[:current_budget]["reference_rmsd_A"], errors="raise"
                    )
                    row.update({
                        "best_of_budget_rmsd_A": float(prefix_rmsd.min()),
                        "any_success_by_budget": bool(
                            prefix_rmsd.lt(args.success_cutoff).any()
                        ),
                        "run1_failure_rescued_by_budget": bool(
                            group.iloc[0]["reference_rmsd_A"] >= args.success_cutoff
                            and prefix_rmsd.iloc[1:].lt(args.success_cutoff).any()
                        ),
                    })
                budget_case_rows.append(row)
        except Exception as exc:
            failures.append({
                "complex_name": complex_name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures, columns=["complex_name", "error"]).to_csv(
        output / "failures.csv", index=False
    )
    pd.DataFrame(interaction_failures).to_csv(
        output / "interaction_scoring_failures.csv", index=False
    )
    if failures:
        raise RuntimeError(f"Mode audit failed for {len(failures)} cases")

    pose_table = pd.DataFrame(pose_rows)
    cluster_table = pd.DataFrame(cluster_rows)
    cluster_pair_table = pd.DataFrame(cluster_pair_rows)
    pair_table = pd.DataFrame(pair_rows)
    per_case = pd.DataFrame(case_rows)
    budget_cases = pd.DataFrame(budget_case_rows)
    pose_table.to_csv(output / "pose_mode_assignments.csv", index=False)
    cluster_table.to_csv(output / "mode_quality_summary.csv", index=False)
    cluster_pair_table.to_csv(output / "mode_pair_geometry.csv", index=False)
    pair_table.to_csv(output / "pairwise_pose_rmsd.csv", index=False)
    per_case.to_csv(output / "per_case_mode_summary.csv", index=False)
    budget_cases.to_csv(output / "per_case_mode_discovery_by_budget.csv", index=False)

    reference_free_candidates = per_case[
        per_case["multiple_reproducible_modes"]
    ].copy()
    sort_columns = []
    ascending = []
    ncc_gap = "top_two_reproducible_mode_multiscale_ncc_gap"
    if ncc_gap in reference_free_candidates.columns:
        sort_columns.append(ncc_gap)
        ascending.append(True)
    sort_columns.extend(["reproducible_mode_count", "largest_mode_fraction"])
    ascending.extend([False, True])
    reference_free_candidates = reference_free_candidates.sort_values(
        sort_columns, ascending=ascending, kind="stable", na_position="last"
    )
    reference_free_candidates.to_csv(
        output / "reference_free_multimodal_candidates.csv", index=False
    )

    budget_summary_rows = []
    for current_budget, group in budget_cases.groupby("budget_runs", sort=True):
        record: dict[str, object] = {
            "budget_runs": int(current_budget),
            "case_n": int(len(group)),
            "mean_full_budget_modes_discovered": float(
                group["full_budget_modes_discovered"].mean()
            ),
            "median_full_budget_modes_discovered": float(
                group["full_budget_modes_discovered"].median()
            ),
            "all_full_budget_modes_discovered_n": int(
                group["all_full_budget_modes_discovered"].sum()
            ),
            "all_full_budget_reproducible_modes_discovered_n": int(
                group["all_full_budget_reproducible_modes_discovered"].sum()
            ),
        }
        if references is not None:
            run1_failures = int((~per_case["run1_success"]).sum())
            rescue_n = int(group["run1_failure_rescued_by_budget"].sum())
            record.update({
                "best_of_budget_success_n": int(group["any_success_by_budget"].sum()),
                "best_of_budget_success_pct": float(
                    100.0 * group["any_success_by_budget"].mean()
                ),
                "run1_failure_n_fixed": run1_failures,
                "run1_failure_rescued_by_budget_n": rescue_n,
                "run1_failure_rescued_by_budget_pct": (
                    100.0 * rescue_n / run1_failures if run1_failures else np.nan
                ),
            })
        budget_summary_rows.append(record)
    pd.DataFrame(budget_summary_rows).to_csv(
        output / "mode_discovery_and_rescue_by_budget.csv", index=False
    )

    agreement_rescue = None
    if references is not None:
        agreement_rescue = summarize_agreement_rescue(per_case, budget)
        agreement_rescue.to_csv(output / "agreement_level_rescue_summary.csv", index=False)
        per_case[per_case["rescued_by_later_run"]].to_csv(
            output / "run1_failures_rescued_by_later_runs.csv", index=False
        )
        per_case[per_case["all_runs_failure"]].to_csv(
            output / "all_runs_rmsd_failures.csv", index=False
        )

    multiple_n = int(per_case["multiple_reproducible_modes"].sum())
    summary: dict[str, object] = {
        "analysis_role": "post_frozen_multirun_pose_mode_audit",
        "requested_cases_n": len(cohort_ids),
        "run_count": budget,
        "cluster_definition": (
            "complete linkage on symmetry-aware heavy-atom RMSD at 2.0 A"
        ),
        "reproducible_mode_definition": "cluster represented by at least 2 independent runs",
        "mode_count_median": float(per_case["mode_count"].median()),
        "reproducible_mode_count_median": float(
            per_case["reproducible_mode_count"].median()
        ),
        "multiple_reproducible_modes_n": multiple_n,
        "multiple_reproducible_modes_pct": 100.0 * multiple_n / len(per_case),
        "reference_free_candidate_ranking": (
            "Cases with >=2 reproducible modes; prioritize smaller Top-2 mode "
            "median multiscale-NCC gap, then more reproducible modes. No RMSD label used."
        ),
        "nested_budget_mode_discovery_policy": (
            "Full-run 2 A mode labels are fixed using all runs; each nested prefix "
            "reports which final modes had already been observed. This is a "
            "retrospective mode-discovery curve, not an online stopping rule."
        ),
        "interpretation_boundary": (
            "Reproducible density-compatible prediction modes are model-generated "
            "structural hypotheses, not a Bayesian posterior or proof of biological heterogeneity."
        ),
        "uses_reference_for_clustering_or_candidate_ranking": False,
        "reference_reveal_performed": references is not None,
        "interaction_geometry_scored": args.score_interactions,
        "interaction_scoring_failures_n": len(interaction_failures),
        "inputs": {
            "selected_pose_replicates": {
                "path": str(selected_path), "sha256": sha256(selected_path)
            },
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
        },
    }
    if references is not None:
        run1_failure_n = int((~per_case["run1_success"]).sum())
        rescue_n = int(per_case["rescued_by_later_run"].sum())
        summary.update({
            "run1_success_n": int(per_case["run1_success"].sum()),
            "run1_failure_n": run1_failure_n,
            "run1_failures_rescued_by_later_runs_n": rescue_n,
            "run1_failures_rescued_by_later_runs_pct": (
                100.0 * rescue_n / run1_failure_n if run1_failure_n else np.nan
            ),
            "any_run_success_n": int(per_case["any_run_success"].sum()),
            "all_runs_failure_n": int(per_case["all_runs_failure"].sum()),
            "reference_usage": (
                "Reference RMSD is computed only after pose clusters and the "
                "reference-free candidate ordering are fixed. Best-of-run results "
                "are retrospective candidate coverage, not operational Top-1."
            ),
        })
        summary["inputs"]["reference_manifest"] = {
            "path": str(reference_path), "sha256": sha256(reference_path)
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if agreement_rescue is not None:
        print("\nAGREEMENT-LEVEL RESCUE")
        print(agreement_rescue.to_string(index=False))
    print(f"\nMULTIRUN_POSE_MODE_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
