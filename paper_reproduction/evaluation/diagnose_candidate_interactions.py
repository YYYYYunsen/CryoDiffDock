#!/usr/bin/env python3
"""Retrospective validation diagnostic for interaction-aware pose selection.

RMSD is read only after all deployment-safe geometric interaction features are
computed.  Outputs are exploratory and must not be presented as an independent
test-set result.  No inference or candidate generation is performed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.interactions import (
    read_protein_interaction_sites,
    score_pose_interactions,
)


FEATURES = (
    "hbond_geometry_quality",
    "aromatic_geometry_quality",
    "salt_bridge_geometry_quality",
    "interaction_geometry_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-root", required=True, type=Path, action="append",
        help="Candidate search root; repeat for non-overlapping completion roots.",
    )
    parser.add_argument("--cohort", type=Path, help="Optional validation ID list")
    parser.add_argument(
        "--expected-missing-ids", default="",
        help=(
            "Comma-separated cohort IDs known to lack candidate pools. They remain "
            "failures in the fixed denominator; any other missing ID is an error."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pool-condition", choices=("auto", "interaction_pool", "expanded_pool"),
        default="auto", help="Materialized candidate-pool directory to analyze.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--interaction-weights", default="0,0.025,0.05,0.1,0.2,0.35,0.5,0.75,1.0"
    )
    parser.add_argument(
        "--score-column", default="auto",
        help="Current selector score, or auto to prefer final_selection_score.",
    )
    parser.add_argument(
        "--selector-scope", choices=("auto", "finalists", "eligible", "all"),
        default="auto",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def read_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)) or not values:
        raise ValueError(f"Cohort must contain unique nonempty IDs: {path}")
    return values


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def choose_pool_condition(search_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    for condition in ("interaction_pool", "expanded_pool"):
        if (search_root / condition).is_dir():
            return condition
    raise FileNotFoundError(f"No interaction_pool or expanded_pool under {search_root}")


def load_candidates(
    search_root: Path, cohort: list[str] | None, pool_condition: str,
) -> pd.DataFrame:
    ranking_files = sorted((search_root / pool_condition).glob("index*___*/ranking.csv"))
    if not ranking_files:
        raise FileNotFoundError(
            f"No {pool_condition} ranking.csv files under {search_root}"
        )
    rankings = pd.concat(
        [pd.read_csv(path, dtype={"complex_name": str}) for path in ranking_files],
        ignore_index=True,
    )
    required = {"row_index", "complex_name", "candidate_id", "materialized_rank", "materialized_path", "protein_path"}
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")
    rankings["complex_name"] = rankings["complex_name"].astype(str).str.upper()
    if cohort is not None:
        rankings = rankings[rankings["complex_name"].isin(cohort)].copy()

    metrics_path = search_root / f"metrics_{pool_condition}" / "per_pose.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    metric_columns = ["row_index", "rank", "ligand_rmsd_A", "centroid_distance_A", "centered_rmsd_A", "rigid_aligned_rmsd_A"]
    missing = sorted(set(metric_columns) - set(metrics.columns))
    if missing:
        raise ValueError(f"Expanded-pool metrics are missing columns: {missing}")
    candidates = rankings.merge(
        metrics[metric_columns], left_on=["row_index", "materialized_rank"],
        right_on=["row_index", "rank"], how="inner", validate="one_to_one",
    )
    if len(candidates) != len(rankings):
        raise ValueError(f"Only {len(candidates)}/{len(rankings)} candidates matched RMSD")
    return candidates.sort_values(["complex_name", "candidate_id"], kind="stable").reset_index(drop=True)


def choose_score_column(candidates: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in candidates:
            raise ValueError(f"Missing requested score column: {requested}")
        return requested
    for column in ("final_selection_score", "expanded_density_score", "selection_score", "s0_selection_score"):
        if column in candidates and pd.to_numeric(candidates[column], errors="coerce").notna().any():
            return column
    raise ValueError("Could not find a selector score column")


def selector_mask(candidates: pd.DataFrame, scope: str, score_column: str) -> tuple[pd.Series, str]:
    if scope in {"auto", "finalists"} and "v2_finalist" in candidates:
        mask = bool_series(candidates["v2_finalist"])
        if mask.any():
            return mask & pd.to_numeric(candidates[score_column], errors="coerce").notna(), "v2_finalists"
    if scope == "finalists":
        raise ValueError("--selector-scope finalists requested but v2_finalist is unavailable")
    if scope in {"auto", "eligible"} and "density_gate_pass" in candidates:
        mask = bool_series(candidates["density_gate_pass"])
        if mask.any():
            return mask & pd.to_numeric(candidates[score_column], errors="coerce").notna(), "density_gate_pass"
    if scope == "eligible":
        raise ValueError("--selector-scope eligible requested but density_gate_pass is unavailable")
    return pd.to_numeric(candidates[score_column], errors="coerce").notna(), "all_scored_candidates"


def score_complex(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    protein_cache: dict[str, object] = {}
    for row in rows:
        base = {"complex_name": row["complex_name"], "candidate_id": row["candidate_id"]}
        try:
            protein_path = str(row["protein_path"])
            if protein_path not in protein_cache:
                protein_cache[protein_path] = read_protein_interaction_sites(protein_path)
            result = score_pose_interactions(protein_cache[protein_path], str(row["materialized_path"]))
            features.append({**base, **result})
        except Exception as exc:
            failures.append({**base, "error": f"{type(exc).__name__}: {exc}"})
    return features, failures


def compute_features(candidates: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [group.to_dict("records") for _, group in candidates.groupby("complex_name", sort=False)]
    feature_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    if workers <= 1:
        for task in tasks:
            features, failures = score_complex(task)
            feature_rows.extend(features); failure_rows.extend(failures)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score_complex, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                features, failures = future.result()
                feature_rows.extend(features); failure_rows.extend(failures)
                if index % 10 == 0 or index == len(futures):
                    print(f"Scored {index}/{len(futures)} complexes")
    return pd.DataFrame(feature_rows), pd.DataFrame(failure_rows)


def within_complex_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    def transform(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        scale = float(numeric.std(ddof=0))
        if not np.isfinite(scale) or scale <= 1e-12:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (numeric - float(numeric.mean())) / scale
    return frame.groupby("complex_name", sort=False)[column].transform(transform)


def auc_probability(success: np.ndarray, failure: np.ndarray) -> float:
    if len(success) == 0 or len(failure) == 0:
        return float("nan")
    delta = success[:, None] - failure[None, :]
    return float(((delta > 0).sum() + 0.5 * (delta == 0).sum()) / delta.size)


def feature_associations(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_complex = []
    for complex_name, group in candidates.groupby("complex_name", sort=False):
        success = group["ligand_rmsd_A"] < 2.0
        for feature in FEATURES:
            values = pd.to_numeric(group[feature], errors="coerce")
            finite = values.notna() & group["ligand_rmsd_A"].notna()
            rho = float("nan")
            if finite.sum() >= 3 and values[finite].nunique() > 1:
                rho = float(spearmanr(values[finite], -group.loc[finite, "ligand_rmsd_A"]).statistic)
            auc = auc_probability(values[success & values.notna()].to_numpy(), values[~success & values.notna()].to_numpy())
            per_complex.append({"complex_name": complex_name, "feature": feature, "candidate_n": int(finite.sum()), "success_n": int((success & finite).sum()), "spearman_with_negative_rmsd": rho, "success_auc": auc})
    detailed = pd.DataFrame(per_complex)
    summary = []
    for feature, group in detailed.groupby("feature", sort=False):
        rho = group["spearman_with_negative_rmsd"].dropna()
        auc = group["success_auc"].dropna()
        summary.append({
            "feature": feature,
            "complexes_with_spearman": int(len(rho)),
            "median_within_complex_spearman": float(rho.median()) if len(rho) else np.nan,
            "positive_spearman_pct": float(100.0 * (rho > 0).mean()) if len(rho) else np.nan,
            "complexes_with_success_auc": int(len(auc)),
            "mean_within_complex_success_auc": float(auc.mean()) if len(auc) else np.nan,
        })
    return pd.DataFrame(summary), detailed


def select_rows(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    work = frame.copy()
    work["_rerank_score"] = score
    selected = work.sort_values(
        ["complex_name", "_rerank_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).groupby("complex_name", as_index=False).first()
    return selected


def metrics(
    selected: pd.DataFrame, fixed_denominator_n: int | None = None,
) -> dict[str, object]:
    rmsd = selected["ligand_rmsd_A"].to_numpy(dtype=float)
    denominator = len(selected) if fixed_denominator_n is None else fixed_denominator_n
    if denominator < len(selected) or denominator <= 0:
        raise ValueError("Fixed denominator must be at least the scored complex count")
    return {
        "complex_n": int(len(selected)),
        "fixed_denominator_n": int(denominator),
        "top1_lt2_n": int((rmsd < 2.0).sum()),
        "top1_lt2_pct": float(100.0 * (rmsd < 2.0).sum() / denominator),
        "top1_lt5_pct": float(100.0 * (rmsd < 5.0).sum() / denominator),
        "median_rmsd_A": float(np.median(rmsd)),
        "mean_rmsd_A": float(np.mean(rmsd)),
    }


def reranking_sweep(
    candidates: pd.DataFrame,
    score_column: str,
    mask: pd.Series,
    weights: list[float],
    fixed_denominator_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = candidates.loc[mask].copy()
    base_score = pd.to_numeric(eligible[score_column], errors="coerce").to_numpy(float)
    baseline = select_rows(eligible, base_score)
    baseline_metrics = metrics(baseline, fixed_denominator_n)
    rows = []
    selections = []
    for feature in FEATURES:
        z = eligible[f"{feature}_z"].to_numpy(float)
        for weight in weights:
            selected = select_rows(eligible, base_score + float(weight) * z)
            result = metrics(selected, fixed_denominator_n)
            paired_columns = [
                "complex_name", "ligand_rmsd_A", "candidate_id",
                "materialized_path", score_column, feature,
            ]
            paired = baseline[paired_columns].merge(
                selected[paired_columns],
                on="complex_name", suffixes=("_baseline", "_reranked"), validate="one_to_one",
            )
            before = paired["ligand_rmsd_A_baseline"] < 2.0
            after = paired["ligand_rmsd_A_reranked"] < 2.0
            result.update({
                "feature": feature, "interaction_weight": float(weight),
                "delta_top1_lt2_pp": result["top1_lt2_pct"] - baseline_metrics["top1_lt2_pct"],
                "delta_median_rmsd_A": result["median_rmsd_A"] - baseline_metrics["median_rmsd_A"],
                "gained_n": int((~before & after).sum()), "lost_n": int((before & ~after).sum()),
                "changed_n": int((paired["candidate_id_baseline"] != paired["candidate_id_reranked"]).sum()),
            })
            rows.append(result)
            for _, item in paired.iterrows():
                selections.append({"feature": feature, "interaction_weight": float(weight), **item.to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(selections)


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main() -> int:
    args = parse_args()
    search_roots = [path.expanduser().resolve() for path in args.search_root]
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    cohort = read_ids(args.cohort.expanduser().resolve() if args.cohort else None)
    expected_missing = {
        value.strip().upper()
        for value in args.expected_missing_ids.split(",")
        if value.strip()
    }
    if expected_missing and cohort is None:
        raise ValueError("--expected-missing-ids requires --cohort")
    unknown_expected_missing = expected_missing - set(cohort or [])
    if unknown_expected_missing:
        raise ValueError(
            "Expected-missing IDs are outside the cohort: "
            f"{sorted(unknown_expected_missing)}"
        )
    candidate_frames = []
    pool_conditions = {}
    for search_root in search_roots:
        pool_condition = choose_pool_condition(search_root, args.pool_condition)
        frame = load_candidates(search_root, cohort, pool_condition)
        if frame.empty:
            continue
        frame["source_search_root"] = str(search_root)
        frame["source_pool_condition"] = pool_condition
        candidate_frames.append(frame)
        pool_conditions[str(search_root)] = pool_condition
    if not candidate_frames:
        raise ValueError("No requested cohort candidates were found in any search root")
    candidates = pd.concat(candidate_frames, ignore_index=True)
    if cohort is not None:
        absent = sorted(set(cohort) - set(candidates["complex_name"]))
        if set(absent) != expected_missing:
            raise ValueError(
                "Combined candidate-pool missing IDs do not match the explicit "
                f"expectation: observed={absent}, expected={sorted(expected_missing)}"
            )
    else:
        absent = []
    root_coverage = candidates[["complex_name", "source_search_root"]].drop_duplicates()
    overlapping = (
        root_coverage.groupby("complex_name")["source_search_root"].nunique()
    )
    overlapping = overlapping[overlapping > 1]
    if len(overlapping):
        raise ValueError(
            "Complexes occur in multiple search roots: "
            f"{overlapping.index.tolist()[:30]}"
        )
    if candidates.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Combined roots contain duplicate candidate identifiers")
    score_column = choose_score_column(candidates, args.score_column)
    mask, scope = selector_mask(candidates, args.selector_scope, score_column)
    features, failures = compute_features(candidates, args.workers)
    candidates = candidates.merge(features, on=["complex_name", "candidate_id"], how="left", validate="one_to_one")
    feature_missing = candidates["interaction_geometry_quality"].isna()
    if feature_missing.any():
        mask = mask & ~feature_missing
    for feature in FEATURES:
        candidates[f"{feature}_z"] = within_complex_zscore(candidates, feature)
    association, association_per_complex = feature_associations(candidates.loc[~feature_missing])
    weights = [float(value) for value in args.interaction_weights.split(",") if value.strip()]
    if 0.0 not in weights or any(value < 0 for value in weights):
        raise ValueError("Interaction weights must be nonnegative and include zero")
    fixed_denominator_n = len(cohort) if cohort is not None else None
    sweep, paired = reranking_sweep(
        candidates, score_column, mask, weights, fixed_denominator_n,
    )
    nonzero = sweep[sweep["interaction_weight"] > 0].copy()
    best = nonzero.sort_values(
        ["top1_lt2_n", "median_rmsd_A", "interaction_weight"],
        ascending=[False, True, True], kind="stable",
    ).iloc[0]
    best_paired = paired[
        paired["feature"].eq(best["feature"])
        & paired["interaction_weight"].eq(best["interaction_weight"])
    ].copy()
    best_paired["delta_rmsd_A"] = best_paired["ligand_rmsd_A_reranked"] - best_paired["ligand_rmsd_A_baseline"]
    best_paired["outcome"] = np.select(
        [
            (best_paired["ligand_rmsd_A_baseline"] >= 2) & (best_paired["ligand_rmsd_A_reranked"] < 2),
            (best_paired["ligand_rmsd_A_baseline"] < 2) & (best_paired["ligand_rmsd_A_reranked"] >= 2),
            best_paired["candidate_id_baseline"] != best_paired["candidate_id_reranked"],
        ], ["gained", "lost", "changed_same_threshold"], default="unchanged",
    )
    baseline_row = sweep[sweep["interaction_weight"].eq(0.0)].iloc[0]
    exploratory_benefit = bool(
        best["top1_lt2_n"] > baseline_row["top1_lt2_n"]
        or (
            best["top1_lt2_n"] == baseline_row["top1_lt2_n"]
            and best["median_rmsd_A"] < baseline_row["median_rmsd_A"]
        )
    )

    outputs = {
        "candidate_interaction_features.csv": candidates,
        "feature_association_summary.csv": association,
        "feature_association_per_complex.csv": association_per_complex,
        "reranking_weight_sweep.csv": sweep,
        "best_exploratory_rerank_per_complex.csv": best_paired,
        "feature_failures.csv": failures,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    validations = pd.DataFrame([
        {"check": "candidate_rows_loaded", "observed": len(candidates), "expected": "> 0", "passed": len(candidates) > 0},
        {"check": "scored_complexes_loaded", "observed": candidates["complex_name"].nunique(), "expected": len(cohort) - len(expected_missing) if cohort else "> 0", "passed": candidates["complex_name"].nunique() == len(cohort) - len(expected_missing) if cohort else candidates["complex_name"].nunique() > 0},
        {"check": "fixed_denominator", "observed": fixed_denominator_n or candidates["complex_name"].nunique(), "expected": len(cohort) if cohort else candidates["complex_name"].nunique(), "passed": (fixed_denominator_n or candidates["complex_name"].nunique()) == (len(cohort) if cohort else candidates["complex_name"].nunique())},
        {"check": "expected_missing_ids", "observed": ",".join(absent), "expected": ",".join(sorted(expected_missing)), "passed": set(absent) == expected_missing},
        {"check": "unique_candidate_keys", "observed": int(candidates.duplicated(["complex_name", "candidate_id"]).sum()), "expected": 0, "passed": not candidates.duplicated(["complex_name", "candidate_id"]).any()},
        {"check": "interaction_feature_coverage_pct", "observed": 100.0 * (~feature_missing).mean(), "expected": 100.0, "passed": not feature_missing.any()},
        {"check": "selector_scope_has_each_complex", "observed": candidates.loc[mask, "complex_name"].nunique(), "expected": candidates["complex_name"].nunique(), "passed": candidates.loc[mask, "complex_name"].nunique() == candidates["complex_name"].nunique()},
        {"check": "weight_zero_is_invariant", "observed": int(sweep.loc[sweep["interaction_weight"].eq(0), "changed_n"].max()), "expected": 0, "passed": int(sweep.loc[sweep["interaction_weight"].eq(0), "changed_n"].max()) == 0},
    ])
    validations.to_csv(output / "validation_checks.csv", index=False)
    summary = {
        "analysis": "retrospective validation candidate-pool interaction diagnostic",
        "interpretation": "exploratory; select and freeze a rule before independent test evaluation",
        "search_roots": [str(path) for path in search_roots],
        "cohort": str(args.cohort) if args.cohort else None,
        "expected_missing_ids": sorted(expected_missing),
        "scored_complex_n": candidates["complex_name"].nunique(),
        "fixed_denominator_n": fixed_denominator_n or candidates["complex_name"].nunique(),
        "pool_conditions": pool_conditions,
        "candidate_n": len(candidates), "complex_n": candidates["complex_name"].nunique(),
        "selector_score_column": score_column, "selector_scope": scope,
        "feature_failure_n": int(feature_missing.sum()),
        "best_exploratory_configuration": best.to_dict(),
        "best_configuration_improves_success_or_median": exploratory_benefit,
        "deployment_decision": (
            "candidate rule for freezing and independent evaluation"
            if exploratory_benefit
            else "no interaction reranking benefit detected; retain frozen selector"
        ),
        "validation_passed": bool(validations["passed"].all()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    index = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            index.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(index).to_csv(output / "file_index.csv", index=False)
    print("\nFEATURE ASSOCIATION")
    print(association.to_string(index=False))
    print("\nRERANKING SWEEP")
    print(sweep.to_string(index=False))
    print("\nBEST EXPLORATORY CONFIGURATION")
    print(best.to_string())
    print(
        "\nDECISION: "
        + (
            "candidate rule found; freeze before any independent evaluation"
            if exploratory_benefit
            else "no benefit detected; keep the existing selector"
        )
    )
    print(f"\nINTERACTION_DIAGNOSTIC_OK: {output}")
    if args.strict and not validations["passed"].all():
        print(validations.loc[~validations["passed"]].to_string(index=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
