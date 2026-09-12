#!/usr/bin/env python3
"""Finalize the paired Local benchmark after documented technical recovery.

The script merges historical and recovery metric tables by complex ID, checks
that every technically evaluable case occurs exactly once, and reports both
the available-case analysis and the formal fixed-denominator analysis in which
documented technical failures count as unsuccessful predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0].strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicates: {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_sources(
    paths: list[Path], cohort: list[str], label: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    wanted = set(cohort)
    frames = []
    audit = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"complex_name": str})
        if "complex_name" not in frame:
            raise ValueError(f"{path} has no complex_name column")
        frame = frame.copy()
        frame["complex_name"] = frame["complex_name"].str.strip().str.upper()
        if frame["complex_name"].duplicated().any():
            raise ValueError(f"Duplicate complex IDs within {path}")
        selected = frame[frame["complex_name"].isin(wanted)].copy()
        selected["metric_source"] = str(path)
        frames.append(selected)
        audit.append({
            "metric_group": label,
            "source": str(path),
            "source_sha256": sha256(path),
            "source_rows": len(frame),
            "evaluable_cohort_rows_contributed": len(selected),
        })
    merged = pd.concat(frames, ignore_index=True, sort=False)
    duplicates = sorted(
        merged.loc[
            merged["complex_name"].duplicated(False), "complex_name"
        ].unique()
    )
    missing = [name for name in cohort if name not in set(merged["complex_name"])]
    if duplicates or missing:
        raise ValueError(
            f"{label} coverage failure: duplicates={duplicates[:20]}, "
            f"missing={missing[:20]}"
        )
    indexed = merged.set_index("complex_name")
    result = indexed.loc[cohort].reset_index()
    result["row_index"] = np.arange(len(result), dtype=int)
    return result, audit


def finite_metric(frame: pd.DataFrame, column: str, label: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"{label} lacks {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{label}.{column} contains invalid RMSD values")
    return values


def bootstrap_rate(
    success: np.ndarray, sample_indices: np.ndarray
) -> tuple[float, float, float]:
    estimate = 100.0 * float(success.mean())
    sampled = 100.0 * success[sample_indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return estimate, float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-cohort", required=True, type=Path)
    parser.add_argument("--evaluable-cohort", required=True, type=Path)
    parser.add_argument("--original-metrics", action="append", required=True, type=Path)
    parser.add_argument("--v1-selected-metrics", action="append", required=True, type=Path)
    parser.add_argument("--v1-oracle-metrics", action="append", required=True, type=Path)
    parser.add_argument("--v2-selected-metrics", action="append", required=True, type=Path)
    parser.add_argument("--v2-oracle-metrics", action="append", required=True, type=Path)
    parser.add_argument("--expected-formal", type=int, default=190)
    parser.add_argument("--expected-evaluable", type=int, default=189)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    formal_path = args.formal_cohort.expanduser().resolve()
    evaluable_path = args.evaluable_cohort.expanduser().resolve()
    formal = read_ids(formal_path)
    evaluable = read_ids(evaluable_path)
    if len(formal) != args.expected_formal:
        raise ValueError(f"Expected {args.expected_formal} formal cases, found {len(formal)}")
    if len(evaluable) != args.expected_evaluable:
        raise ValueError(
            f"Expected {args.expected_evaluable} evaluable cases, found {len(evaluable)}"
        )
    if not set(evaluable) <= set(formal):
        raise ValueError("Evaluable cohort is not a subset of formal cohort")
    technical_failures = [name for name in formal if name not in set(evaluable)]

    specifications = {
        "original": args.original_metrics,
        "v1_selected": args.v1_selected_metrics,
        "v1_oracle": args.v1_oracle_metrics,
        "v2_selected": args.v2_selected_metrics,
        "v2_oracle": args.v2_oracle_metrics,
    }
    merged: dict[str, pd.DataFrame] = {}
    source_audit: list[dict[str, object]] = []
    for label, paths in specifications.items():
        merged[label], audit = merge_sources(paths, evaluable, label)
        source_audit.extend(audit)

    values = {
        ("CryoDiffDock-Pocket", "Top-1"): finite_metric(
            merged["original"], "top1_rmsd_A", "original"
        ),
        ("CryoDiffDock-Pocket", "Best-of-40"): finite_metric(
            merged["original"], "best_of_n_rmsd_A", "original"
        ),
        ("CryoDiffDock-Local V1", "Top-1"): finite_metric(
            merged["v1_selected"], "top1_rmsd_A", "v1_selected"
        ),
        ("CryoDiffDock-Local V1", "Oracle"): finite_metric(
            merged["v1_oracle"], "best_of_n_rmsd_A", "v1_oracle"
        ),
        ("CryoDiffDock-Local V2", "Top-1"): finite_metric(
            merged["v2_selected"], "top1_rmsd_A", "v2_selected"
        ),
        ("CryoDiffDock-Local V2", "Oracle"): finite_metric(
            merged["v2_oracle"], "best_of_n_rmsd_A", "v2_oracle"
        ),
    }

    per_complex = pd.DataFrame({"complex_name": evaluable})
    per_complex["original_top1_rmsd_A"] = values[("CryoDiffDock-Pocket", "Top-1")]
    per_complex["original_best_of_40_rmsd_A"] = values[
        ("CryoDiffDock-Pocket", "Best-of-40")
    ]
    per_complex["v1_top1_rmsd_A"] = values[("CryoDiffDock-Local V1", "Top-1")]
    per_complex["v1_oracle_rmsd_A"] = values[("CryoDiffDock-Local V1", "Oracle")]
    per_complex["v2_top1_rmsd_A"] = values[("CryoDiffDock-Local V2", "Top-1")]
    per_complex["v2_oracle_rmsd_A"] = values[("CryoDiffDock-Local V2", "Oracle")]

    rng = np.random.default_rng(args.seed)
    sample_indices = {
        "technically_evaluable": rng.integers(
            0, len(evaluable), size=(args.bootstrap_repeats, len(evaluable))
        ),
        "formal_fixed_denominator": rng.integers(
            0, len(formal), size=(args.bootstrap_repeats, len(formal))
        ),
    }
    rate_rows = []
    for (method, endpoint), rmsd in values.items():
        for cutoff in (1.0, 2.0, 5.0):
            observed_success = rmsd < cutoff
            for analysis_set, denominator in (
                ("technically_evaluable", len(evaluable)),
                ("formal_fixed_denominator", len(formal)),
            ):
                success = np.zeros(denominator, dtype=bool)
                success[: len(evaluable)] = observed_success
                estimate, low, high = bootstrap_rate(
                    success, sample_indices[analysis_set]
                )
                rate_rows.append({
                    "analysis_set": analysis_set,
                    "method": method,
                    "endpoint": endpoint,
                    "cutoff_A": cutoff,
                    "success_count": int(observed_success.sum()),
                    "denominator": denominator,
                    "success_pct": estimate,
                    "bootstrap_95ci_low_pct": low,
                    "bootstrap_95ci_high_pct": high,
                    "scored_n": len(evaluable),
                    "technical_failures_counted_as_failure": (
                        analysis_set == "formal_fixed_denominator"
                    ),
                    "median_rmsd_A_among_scored": float(np.median(rmsd)),
                })

    comparison_rows = []
    comparisons = (
        ("CryoDiffDock-Pocket", "CryoDiffDock-Local V1", "Top-1", "Top-1"),
        ("CryoDiffDock-Pocket", "CryoDiffDock-Local V2", "Top-1", "Top-1"),
        ("CryoDiffDock-Local V1", "CryoDiffDock-Local V2", "Top-1", "Top-1"),
        ("CryoDiffDock-Pocket", "CryoDiffDock-Local V1", "Best-of-40", "Oracle"),
        ("CryoDiffDock-Pocket", "CryoDiffDock-Local V2", "Best-of-40", "Oracle"),
        ("CryoDiffDock-Local V1", "CryoDiffDock-Local V2", "Oracle", "Oracle"),
    )
    pair_indices = rng.integers(
        0, len(evaluable), size=(args.bootstrap_repeats, len(evaluable))
    )
    for left_method, right_method, left_endpoint, right_endpoint in comparisons:
        left = values[(left_method, left_endpoint)]
        right = values[(right_method, right_endpoint)]
        try:
            wilcoxon_p = float(wilcoxon(right, left).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        for cutoff in (1.0, 2.0, 5.0):
            left_success = left < cutoff
            right_success = right < cutoff
            right_only = int((~left_success & right_success).sum())
            left_only = int((left_success & ~right_success).sum())
            discordant = right_only + left_only
            delta = right_success.astype(float) - left_success.astype(float)
            sampled = 100.0 * delta[pair_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(sampled, [0.025, 0.975])
            comparison_rows.append({
                "left_method": left_method,
                "left_endpoint": left_endpoint,
                "right_method": right_method,
                "right_endpoint": right_endpoint,
                "cutoff_A": cutoff,
                "paired_n": len(evaluable),
                "delta_right_minus_left_percentage_points": 100.0 * float(delta.mean()),
                "paired_bootstrap_95ci_low_pp": float(ci_low),
                "paired_bootstrap_95ci_high_pp": float(ci_high),
                "right_only_successes": right_only,
                "left_only_successes": left_only,
                "exact_mcnemar_pvalue": (
                    float(binomtest(right_only, discordant, 0.5).pvalue)
                    if discordant else 1.0
                ),
                "paired_rmsd_wilcoxon_pvalue": wilcoxon_p,
            })

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_complex.to_csv(output / "local189_per_complex.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(output / "success_rates.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(source_audit).to_csv(output / "source_coverage_audit.csv", index=False)
    for label, frame in merged.items():
        frame.to_csv(output / f"{label}_merged_per_complex.csv", index=False)

    summary = {
        "formal_cohort_n": len(formal),
        "technically_evaluable_n": len(evaluable),
        "technical_failure_n": len(technical_failures),
        "technical_failure_ids": technical_failures,
        "primary_fixed_denominator": len(formal),
        "paired_pose_quality_denominator": len(evaluable),
        "technical_failures_count_as_unsuccessful_in_primary_rates": True,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "formal_cohort": str(formal_path),
        "formal_cohort_sha256": sha256(formal_path),
        "evaluable_cohort": str(evaluable_path),
        "evaluable_cohort_sha256": sha256(evaluable_path),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nSOURCE COVERAGE")
    print(pd.DataFrame(source_audit).to_string(index=False))
    print("\nTOP-1 <2 A")
    rates = pd.DataFrame(rate_rows)
    print(rates[
        rates["endpoint"].eq("Top-1") & rates["cutoff_A"].eq(2.0)
    ].to_string(index=False))
    print(f"LOCAL_FORMAL_BENCHMARK_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
