#!/usr/bin/env python3
"""Create a paired baseline-versus-guided ligand docking summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline per_complex.csv")
    parser.add_argument("--guided", required=True, type=Path, help="Guided per_complex.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    pairing_group = parser.add_mutually_exclusive_group()
    pairing_group.add_argument(
        "--allow-baseline-superset",
        action="store_true",
        help=(
            "Allow the baseline table to contain additional complexes, while "
            "requiring every guided complex to have exactly one baseline match. "
            "Useful for comparing a pilot subset against frozen full-cohort metrics."
        ),
    )
    pairing_group.add_argument(
        "--allow-intersection",
        action="store_true",
        help=(
            "Pair the intersection of the two tables and report unmatched IDs. "
            "Useful when frozen methods have different technical-failure sets."
        ),
    )
    args = parser.parse_args(argv)

    baseline = pd.read_csv(args.baseline.expanduser().resolve())
    guided = pd.read_csv(args.guided.expanduser().resolve())
    key = "complex_name"
    for label, table in (("baseline", baseline), ("guided", guided)):
        if key not in table.columns:
            raise ValueError(f"{label} table has no {key} column")
        if table[key].astype(str).duplicated().any():
            duplicates = table.loc[
                table[key].astype(str).duplicated(keep=False), key
            ].astype(str).unique().tolist()
            raise ValueError(f"{label} has duplicate complex_name values: {duplicates[:10]}")
        table[key] = table[key].astype(str)
    primary_metrics = ["top1_rmsd_A", "best_of_n_rmsd_A"]
    diagnostic_metrics = [
        "top1_centroid_distance_A",
        "top1_centered_rmsd_A",
        "top1_rigid_aligned_rmsd_A",
        "best_rmsd_centroid_distance_A",
        "best_rmsd_centered_rmsd_A",
        "best_rmsd_rigid_aligned_rmsd_A",
        "best_of_n_centroid_distance_A",
    ]
    metrics = primary_metrics + [
        metric for metric in diagnostic_metrics
        if metric in baseline.columns and metric in guided.columns
    ]
    identity_columns = [key]
    if "row_index" in baseline.columns:
        identity_columns.insert(0, "row_index")
    guided_identity_columns = [key]
    if "row_index" in guided.columns:
        guided_identity_columns.insert(0, "row_index")
    paired = baseline[identity_columns + metrics].merge(
        guided[guided_identity_columns + metrics],
        on=key,
        suffixes=("_baseline", "_guided"),
        validate="one_to_one",
    )
    baseline_keys = set(baseline[key])
    guided_keys = set(guided[key])
    baseline_only_keys = baseline_keys - guided_keys
    guided_only_keys = guided_keys - baseline_keys
    if args.allow_intersection:
        pairing_valid = len(paired) > 0
    elif args.allow_baseline_superset:
        pairing_valid = not guided_only_keys and len(paired) == len(guided)
    else:
        pairing_valid = (
            not baseline_only_keys
            and not guided_only_keys
            and len(paired) == len(baseline) == len(guided)
        )
    if not pairing_valid:
        raise ValueError(
            f"Conditions are not paired: baseline={len(baseline)}, guided={len(guided)}, overlap={len(paired)}"
        )

    summary = {
        "num_paired_complexes": len(paired),
        "pairing": {
            "num_baseline_complexes": len(baseline),
            "num_guided_complexes": len(guided),
            "num_baseline_only": len(baseline_only_keys),
            "num_guided_only": len(guided_only_keys),
            "baseline_only_complexes": sorted(baseline_only_keys),
            "guided_only_complexes": sorted(guided_only_keys),
        },
        "metrics": {},
    }
    for metric in metrics:
        baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
        guided_values = paired[f"{metric}_guided"].to_numpy(dtype=float)
        delta = guided_values - baseline_values
        paired[f"{metric}_delta_guided_minus_baseline"] = delta
        try:
            pvalue = float(wilcoxon(guided_values, baseline_values).pvalue)
        except ValueError:
            pvalue = float("nan")
        metric_summary = {
            "median_baseline_A": float(np.median(baseline_values)),
            "median_guided_A": float(np.median(guided_values)),
            "median_delta_A": float(np.median(delta)),
            "mean_delta_A": float(np.mean(delta)),
            "guided_better_pct": float(100.0 * np.mean(delta < 0.0)),
            "guided_worse_pct": float(100.0 * np.mean(delta > 0.0)),
            "wilcoxon_pvalue": pvalue,
        }
        # Preserve the original top-level schema for the two headline metrics.
        if metric in primary_metrics:
            summary[metric] = metric_summary
        else:
            summary["metrics"][metric] = metric_summary

    for cutoff in (2.0, 5.0):
        for prefix, metric in (("top1", "top1_rmsd_A"), ("best_of_n", "best_of_n_rmsd_A")):
            baseline_success = paired[f"{metric}_baseline"].to_numpy(dtype=float) < cutoff
            guided_success = paired[f"{metric}_guided"].to_numpy(dtype=float) < cutoff
            summary[f"{prefix}_lt{int(cutoff)}A"] = {
                "baseline_pct": float(100.0 * baseline_success.mean()),
                "guided_pct": float(100.0 * guided_success.mean()),
                "gained_complexes": int(np.sum(~baseline_success & guided_success)),
                "lost_complexes": int(np.sum(baseline_success & ~guided_success)),
            }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_complexes.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
