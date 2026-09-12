#!/usr/bin/env python3
"""Merge Figure 3 ablation shards and run paired analyses against main V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ABLATIONS = (
    ("no_density_gradients", "No density gradients"),
    ("no_hierarchical_composition", "No hierarchical composition (depth 1)"),
    ("no_combined_rigid", "No combined rigid moves"),
    ("no_torsion", "No torsional moves"),
    ("no_coarse_orientations", "No coarse orientations"),
    ("no_basin_robustness", "No basin robustness"),
    ("no_clash_penalty", "No clash penalty"),
)

AUDIT_COLUMNS = (
    "num_original",
    "num_parents",
    "num_generated",
    "num_candidates",
    "num_eligible_generated",
    "num_finalists",
    "selected_depth",
    "selected_branch_type",
    "selected_move_type",
    "selected_move_sequence",
    "selection_ready_seconds",
    "deployment_runtime_seconds",
    "case_runtime_seconds",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [line.strip().upper() for line in path.read_text().splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return values


def load_metric(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"complex_name": str})
    required = {"complex_name", "top1_rmsd_A"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks columns {missing}: {path}")
    frame = frame.copy()
    frame["complex_name"] = frame["complex_name"].str.strip().str.upper()
    if frame["complex_name"].duplicated().any():
        duplicates = frame.loc[
            frame["complex_name"].duplicated(False), "complex_name"
        ].unique().tolist()
        raise ValueError(f"{label} contains duplicate IDs: {duplicates[:20]}")
    values = pd.to_numeric(frame["top1_rmsd_A"], errors="coerce")
    if not np.isfinite(values.to_numpy(float)).all() or (values < 0).any():
        raise ValueError(f"{label}.top1_rmsd_A contains invalid values")
    frame["top1_rmsd_A"] = values.astype(float)
    return frame


def holm_adjust(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=float)
    result = np.full(len(array), np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(array))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(array[finite], kind="stable")]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (count - rank) * float(array[index]))
        running = max(running, adjusted)
        result[index] = running
    return result.tolist()


def exact_mcnemar(main_success: np.ndarray, ablated_success: np.ndarray) -> tuple[float, int, int]:
    gained = int((~main_success & ablated_success).sum())
    lost = int((main_success & ~ablated_success).sum())
    discordant = gained + lost
    p_value = float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0
    return p_value, gained, lost


def bootstrap_intervals(
    main: np.ndarray,
    ablated: np.ndarray,
    *,
    fixed_failure_count: int,
    seed: int,
    samples: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(main), size=(samples, len(main)))
    delta = ablated - main
    success_delta = (ablated < 2.0).astype(float) - (main < 2.0).astype(float)
    fixed_success_delta = np.concatenate([
        success_delta,
        np.zeros(fixed_failure_count, dtype=float),
    ])
    fixed_indices = rng.integers(
        0, len(fixed_success_delta), size=(samples, len(fixed_success_delta))
    )
    mean_samples = delta[indices].mean(axis=1)
    median_samples = np.median(delta[indices], axis=1)
    paired_rate_samples = 100.0 * success_delta[indices].mean(axis=1)
    fixed_rate_samples = 100.0 * fixed_success_delta[fixed_indices].mean(axis=1)
    values = {}
    for name, sampled in (
        ("mean_delta_rmsd_A", mean_samples),
        ("median_delta_rmsd_A", median_samples),
        ("paired_lt2_difference_pp", paired_rate_samples),
        ("fixed_lt2_difference_pp", fixed_rate_samples),
    ):
        low, high = np.quantile(sampled, [0.025, 0.975])
        values[f"{name}_ci95_low"] = float(low)
        values[f"{name}_ci95_high"] = float(high)
    return values


def parse_source(value: str, stage: str) -> dict[str, object]:
    source_id, separator, raw_root = value.partition("=")
    if not separator or not source_id.strip() or not raw_root.strip():
        raise argparse.ArgumentTypeError("source must be SOURCE_ID=/path/to/root")
    return {
        "source_id": source_id.strip(),
        "stage": stage,
        "root": Path(raw_root).expanduser().resolve(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--main-metrics", required=True, type=Path)
    parser.add_argument(
        "--technical-failure",
        action="append",
        default=[],
        help="Formal-cohort ID with no evaluable pose; retained as failure in rates",
    )
    parser.add_argument(
        "--stage1-source",
        action="append",
        default=[],
        metavar="SOURCE_ID=ROOT",
        help="Exploratory shard root containing one subdirectory per ablation",
    )
    parser.add_argument(
        "--stage2-source",
        action="append",
        default=[],
        metavar="SOURCE_ID=ROOT",
        help="Confirmatory shard root containing one subdirectory per ablation",
    )
    parser.add_argument("--expected-stage1", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")

    cohort_path = args.cohort.expanduser().resolve()
    cohort = read_ids(cohort_path)
    cohort_set = set(cohort)
    technical_failures = [str(value).strip().upper() for value in args.technical_failure]
    if len(technical_failures) != len(set(technical_failures)):
        raise ValueError("Duplicate --technical-failure IDs")
    unknown_failures = sorted(set(technical_failures) - cohort_set)
    if unknown_failures:
        raise ValueError(f"Technical failures are outside the formal cohort: {unknown_failures}")
    evaluable = [name for name in cohort if name not in set(technical_failures)]
    evaluable_set = set(evaluable)
    sources = [
        *[parse_source(value, "stage1_exploratory") for value in args.stage1_source],
        *[parse_source(value, "stage2_confirmatory") for value in args.stage2_source],
    ]
    if not args.stage1_source or not args.stage2_source:
        raise ValueError("At least one stage1 and one stage2 source are required")
    source_ids = [str(source["source_id"]) for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source IDs must be unique")

    main_path = args.main_metrics.expanduser().resolve()
    main = load_metric(main_path, "main V2")
    main_ids = set(main["complex_name"])
    missing_main = [name for name in evaluable if name not in main_ids]
    if missing_main:
        raise ValueError(f"Main V2 metrics miss cohort IDs: {missing_main[:20]}")
    main = main.set_index("complex_name").loc[evaluable].reset_index()
    main_values = main.set_index("complex_name")["top1_rmsd_A"]

    long_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {
        "cohort": sha256(cohort_path),
        "main_metrics": sha256(main_path),
    }
    for order, (key, label) in enumerate(ABLATIONS):
        key_frames = []
        for source in sources:
            root = Path(source["root"])
            metric_path = root / key / "metrics_v2_selected/per_complex.csv"
            audit_path = root / key / "per_complex_audit.csv"
            metric = load_metric(metric_path, f"{source['source_id']}:{key}")
            metric_total_n = len(metric)
            metric_ids = set(metric["complex_name"])
            outside = sorted(metric_ids - cohort_set)
            metric = metric[metric["complex_name"].isin(evaluable_set)].copy()
            metric = metric[["complex_name", "top1_rmsd_A"]].rename(
                columns={"top1_rmsd_A": "ablation_rmsd_A"}
            )
            metric["source_id"] = source["source_id"]
            metric["stage"] = source["stage"]
            if audit_path.is_file():
                audit = pd.read_csv(audit_path, dtype={"complex_name": str})
                audit["complex_name"] = audit["complex_name"].str.strip().str.upper()
                if audit["complex_name"].duplicated().any():
                    raise ValueError(f"Duplicate IDs in {audit_path}")
                keep = ["complex_name", *[column for column in AUDIT_COLUMNS if column in audit]]
                metric = metric.merge(audit[keep], on="complex_name", how="left", validate="one_to_one")
                input_hashes[f"{source['source_id']}:{key}:audit"] = sha256(audit_path)
            input_hashes[f"{source['source_id']}:{key}:metrics"] = sha256(metric_path)
            coverage_rows.append({
                "ablation_key": key,
                "display_label": label,
                "source_id": source["source_id"],
                "stage": source["stage"],
                "source_total_n": metric_total_n,
                "analysis_cohort_n": len(metric),
                "filtered_out_n": len(outside),
                "filtered_out_ids": ";".join(outside),
                "metrics_path": str(metric_path),
                "metrics_sha256": sha256(metric_path),
                "audit_path": str(audit_path) if audit_path.is_file() else "",
            })
            key_frames.append(metric)
        merged = pd.concat(key_frames, ignore_index=True, sort=False)
        duplicates = sorted(
            merged.loc[merged["complex_name"].duplicated(False), "complex_name"].unique()
        )
        missing = [name for name in evaluable if name not in set(merged["complex_name"])]
        if duplicates or missing:
            raise ValueError(
                f"{key} coverage failure: duplicates={duplicates[:20]}, missing={missing[:20]}"
            )
        merged["display_order"] = order
        merged["ablation_key"] = key
        merged["display_label"] = label
        merged["main_rmsd_A"] = merged["complex_name"].map(main_values)
        merged["delta_rmsd_A"] = merged["ablation_rmsd_A"] - merged["main_rmsd_A"]
        merged["main_lt2"] = merged["main_rmsd_A"] < 2.0
        merged["ablation_lt2"] = merged["ablation_rmsd_A"] < 2.0
        merged["main_lt5"] = merged["main_rmsd_A"] < 5.0
        merged["ablation_lt5"] = merged["ablation_rmsd_A"] < 5.0
        long_frames.append(merged)

    per_complex = pd.concat(long_frames, ignore_index=True, sort=False)
    stage1_ids = set(per_complex.loc[
        per_complex["stage"].eq("stage1_exploratory"), "complex_name"
    ])
    stage2_ids = set(per_complex.loc[
        per_complex["stage"].eq("stage2_confirmatory"), "complex_name"
    ])
    if stage1_ids & stage2_ids:
        raise ValueError(f"Stage1 and stage2 overlap: {sorted(stage1_ids & stage2_ids)[:20]}")
    if args.expected_stage1 is not None and len(stage1_ids) != args.expected_stage1:
        raise ValueError(f"Expected stage1 n={args.expected_stage1}, got {len(stage1_ids)}")
    if stage1_ids | stage2_ids != evaluable_set:
        raise ValueError("Stage1 and stage2 do not reconstruct the evaluable cohort")

    analysis_sets = (
        ("stage1_exploratory", "exploratory", stage1_ids, 0),
        (
            "stage2_confirmatory", "confirmatory", stage2_ids,
            len(technical_failures),
        ),
        ("pooled", "overall_estimate", evaluable_set, len(technical_failures)),
    )
    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for set_index, (analysis_set, role, ids, fixed_failure_count) in enumerate(analysis_sets):
        for order, (key, label) in enumerate(ABLATIONS):
            group = per_complex[
                per_complex["ablation_key"].eq(key)
                & per_complex["complex_name"].isin(ids)
            ].copy()
            group = group.sort_values("complex_name")
            main_array = group["main_rmsd_A"].to_numpy(float)
            ablated_array = group["ablation_rmsd_A"].to_numpy(float)
            delta = ablated_array - main_array
            main_success = main_array < 2.0
            ablated_success = ablated_array < 2.0
            mcnemar_p, gained, lost = exact_mcnemar(main_success, ablated_success)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                try:
                    wilcoxon_p = float(wilcoxon(ablated_array, main_array).pvalue)
                except ValueError:
                    wilcoxon_p = 1.0
            intervals = bootstrap_intervals(
                main_array,
                ablated_array,
                fixed_failure_count=fixed_failure_count,
                seed=args.seed + set_index * 1000 + order * 17,
                samples=args.bootstrap_samples,
            )
            fixed_denominator = len(group) + fixed_failure_count
            paired_main_lt2_pct = 100.0 * float(main_success.mean())
            paired_ablation_lt2_pct = 100.0 * float(ablated_success.mean())
            fixed_main_lt2_pct = 100.0 * float(main_success.sum()) / fixed_denominator
            fixed_ablation_lt2_pct = 100.0 * float(ablated_success.sum()) / fixed_denominator
            summary_rows.append({
                "analysis_set": analysis_set,
                "analysis_role": role,
                "display_order": order,
                "ablation_key": key,
                "display_label": label,
                "paired_evaluable_n": len(group),
                "technical_failure_n": fixed_failure_count,
                "fixed_denominator_n": fixed_denominator,
                "changed_n": int((np.abs(delta) > 1e-8).sum()),
                "improved_n": int((delta < -1e-8).sum()),
                "worsened_n": int((delta > 1e-8).sum()),
                "main_median_rmsd_A": float(np.median(main_array)),
                "ablation_median_rmsd_A": float(np.median(ablated_array)),
                "median_delta_rmsd_A": float(np.median(delta)),
                "mean_delta_rmsd_A": float(np.mean(delta)),
                "main_lt2_pct_paired_evaluable": paired_main_lt2_pct,
                "ablation_lt2_pct_paired_evaluable": paired_ablation_lt2_pct,
                "paired_lt2_difference_pp": paired_ablation_lt2_pct - paired_main_lt2_pct,
                "main_lt2_pct_fixed_denominator": fixed_main_lt2_pct,
                "ablation_lt2_pct_fixed_denominator": fixed_ablation_lt2_pct,
                "fixed_lt2_difference_pp": fixed_ablation_lt2_pct - fixed_main_lt2_pct,
                "main_lt5_pct_paired_evaluable": 100.0 * float((main_array < 5.0).mean()),
                "ablation_lt5_pct_paired_evaluable": 100.0 * float((ablated_array < 5.0).mean()),
                "paired_lt5_difference_pp": 100.0 * float(
                    (ablated_array < 5.0).mean() - (main_array < 5.0).mean()
                ),
                "main_lt5_pct_fixed_denominator": (
                    100.0 * float((main_array < 5.0).sum()) / fixed_denominator
                ),
                "ablation_lt5_pct_fixed_denominator": (
                    100.0 * float((ablated_array < 5.0).sum()) / fixed_denominator
                ),
                "lt2_success_gained_n": gained,
                "lt2_success_lost_n": lost,
                **intervals,
            })
            test_rows.extend([
                {
                    "analysis_set": analysis_set,
                    "analysis_role": role,
                    "display_order": order,
                    "ablation_key": key,
                    "display_label": label,
                    "test_family": "RMSD_distribution",
                    "test": "two-sided Wilcoxon signed-rank",
                    "effect": "median delta RMSD (ablation - main), A",
                    "effect_value": float(np.median(delta)),
                    "paired_evaluable_n": len(group),
                    "fixed_denominator_n": fixed_denominator,
                    "p_value": wilcoxon_p,
                },
                {
                    "analysis_set": analysis_set,
                    "analysis_role": role,
                    "display_order": order,
                    "ablation_key": key,
                    "display_label": label,
                    "test_family": "Top1_lt2",
                    "test": "exact McNemar at 2 A",
                    "effect": "fixed-denominator success-rate difference (ablation - main), pp",
                    "effect_value": fixed_ablation_lt2_pct - fixed_main_lt2_pct,
                    "paired_effect_value": paired_ablation_lt2_pct - paired_main_lt2_pct,
                    "paired_evaluable_n": len(group),
                    "fixed_denominator_n": fixed_denominator,
                    "p_value": mcnemar_p,
                },
            ])

    summary = pd.DataFrame(summary_rows).sort_values(
        ["analysis_set", "display_order"], kind="stable"
    )
    tests = pd.DataFrame(test_rows)
    tests["holm_adjusted_p_value"] = np.nan
    for (_analysis_set, _family), indices in tests.groupby(
        ["analysis_set", "test_family"], sort=False
    ).groups.items():
        adjusted = holm_adjust(tests.loc[indices, "p_value"].tolist())
        tests.loc[indices, "holm_adjusted_p_value"] = adjusted
    tests = tests.sort_values(["analysis_set", "display_order", "test"], kind="stable")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_complex.to_csv(output / "ablation_per_complex.csv", index=False)
    summary.to_csv(output / "ablation_summary.csv", index=False)
    tests.to_csv(output / "ablation_paired_tests.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(output / "ablation_coverage_audit.csv", index=False)

    audit = {
        "analysis": "Figure 3 V2 component ablations",
        "cohort": str(cohort_path),
        "formal_cohort_n": len(cohort),
        "paired_evaluable_n": len(evaluable),
        "technical_failure_n": len(technical_failures),
        "technical_failure_ids": technical_failures,
        "stage1_n": len(stage1_ids),
        "stage2_n": len(stage2_ids),
        "main_metrics": str(main_path),
        "ablation_keys": [key for key, _label in ABLATIONS],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "delta_definition": "ablation minus main V2; positive RMSD is worse",
        "fixed_denominator_policy": (
            "Technical failures are retained as failures in success rates and are "
            "excluded only from paired RMSD/McNemar inputs; concordant failures do "
            "not change McNemar discordance counts."
        ),
        "technical_failure_stage_assignment": "stage2_confirmatory",
        "multiplicity": "Holm correction across seven ablations within each analysis set and test family",
        "source_roots": [
            {key: str(value) for key, value in source.items()} for source in sources
        ],
        "input_sha256": input_hashes,
        "outputs": [
            "ablation_per_complex.csv",
            "ablation_summary.csv",
            "ablation_paired_tests.csv",
            "ablation_coverage_audit.csv",
        ],
    }
    (output / "ablation_analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    columns = [
        "display_label", "paired_evaluable_n", "fixed_denominator_n",
        "ablation_lt2_pct_fixed_denominator", "fixed_lt2_difference_pp",
        "median_delta_rmsd_A", "mean_delta_rmsd_A", "changed_n",
    ]
    for analysis_set, _role, _ids, _failure_count in analysis_sets:
        print(f"\n{analysis_set.upper()}")
        print(summary.loc[summary["analysis_set"].eq(analysis_set), columns].to_string(index=False))
        print("\nPaired tests (Holm within family)")
        print(tests.loc[
            tests["analysis_set"].eq(analysis_set),
            ["display_label", "test", "effect_value", "p_value", "holm_adjusted_p_value"],
        ].to_string(index=False))
    print(f"\nFIGURE3_ABLATION_ANALYSIS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
