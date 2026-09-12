#!/usr/bin/env python3
"""Summarize a once-only locked test reveal with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=/path/per_complex.csv")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("condition must be NAME=/path/per_complex.csv")
    return name.strip(), Path(path).expanduser().resolve()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [100.0 * (centre - half), 100.0 * (centre + half)]


def bootstrap_ci(values: np.ndarray, statistic, rng, repeats: int) -> list[float]:
    n = len(values)
    sampled = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled[index] = statistic(values[rng.integers(0, n, n)])
    return [float(value) for value in np.quantile(sampled, [0.025, 0.975])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", required=True, type=parse_condition)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    if args.bootstrap_repeats <= 0:
        raise ValueError("bootstrap-repeats must be positive")

    tables = {}
    for name, path in args.condition:
        if name in tables:
            raise ValueError(f"Duplicate condition: {name}")
        frame = pd.read_csv(path, dtype={"complex_name": str})
        required = {"complex_name", "top1_rmsd_A", "best_of_n_rmsd_A"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        if frame["complex_name"].duplicated().any():
            raise ValueError(f"{path} contains duplicate complex names")
        tables[name] = frame
    if args.baseline not in tables:
        raise ValueError(f"Unknown baseline condition: {args.baseline}")
    common = set.intersection(*(set(table["complex_name"]) for table in tables.values()))
    if not common:
        raise ValueError("Conditions have no common complexes")
    for name, table in tables.items():
        missing = sorted(set(table["complex_name"]) - common)
        if missing:
            raise ValueError(
                f"Conditions are not exactly paired; {name} has {len(missing)} extra complexes"
            )

    rng = np.random.default_rng(args.seed)
    summary = {
        "num_paired_complexes": len(common),
        "baseline": args.baseline,
        "conditions": {},
        "paired_vs_baseline": {},
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "uses_ground_truth": True,
        "stage": "single locked test reveal",
    }
    ordered_names = sorted(common)
    aligned = {
        name: table.set_index("complex_name").loc[ordered_names].reset_index()
        for name, table in tables.items()
    }
    for name, table in aligned.items():
        top1 = table["top1_rmsd_A"].to_numpy(float)
        best = table["best_of_n_rmsd_A"].to_numpy(float)
        top2 = int((top1 < 2).sum())
        top5 = int((top1 < 5).sum())
        summary["conditions"][name] = {
            "n": len(top1),
            "median_top1_rmsd_A": float(np.median(top1)),
            "median_top1_rmsd_95pct_bootstrap_ci_A": bootstrap_ci(
                top1, np.median, rng, args.bootstrap_repeats
            ),
            "top1_lt2A_count": top2,
            "top1_lt2A_pct": 100.0 * top2 / len(top1),
            "top1_lt2A_95pct_wilson_ci": wilson(top2, len(top1)),
            "top1_lt5A_count": top5,
            "top1_lt5A_pct": 100.0 * top5 / len(top1),
            "top1_lt5A_95pct_wilson_ci": wilson(top5, len(top1)),
            "median_best_of_n_rmsd_A": float(np.median(best)),
            "best_of_n_lt2A_pct": float(100.0 * np.mean(best < 2)),
        }

    baseline = aligned[args.baseline]["top1_rmsd_A"].to_numpy(float)
    for name, table in aligned.items():
        if name == args.baseline:
            continue
        values = table["top1_rmsd_A"].to_numpy(float)
        delta = values - baseline
        baseline_success = baseline < 2
        selected_success = values < 2
        gained = int((~baseline_success & selected_success).sum())
        lost = int((baseline_success & ~selected_success).sum())
        discordant = gained + lost
        success_delta = selected_success.astype(float) - baseline_success.astype(float)
        try:
            wilcoxon_p = float(wilcoxon(values, baseline).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        summary["paired_vs_baseline"][name] = {
            "median_rmsd_delta_A": float(np.median(delta)),
            "median_rmsd_delta_95pct_bootstrap_ci_A": bootstrap_ci(
                delta, np.median, rng, args.bootstrap_repeats
            ),
            "top1_lt2A_delta_percentage_points": float(100.0 * success_delta.mean()),
            "top1_lt2A_delta_95pct_paired_bootstrap_ci": bootstrap_ci(
                success_delta, lambda x: 100.0 * np.mean(x), rng,
                args.bootstrap_repeats,
            ),
            "gained_lt2A": gained,
            "lost_lt2A": lost,
            "exact_mcnemar_pvalue": (
                float(binomtest(gained, discordant, 0.5).pvalue)
                if discordant else 1.0
            ),
            "paired_wilcoxon_rmsd_pvalue": wilcoxon_p,
        }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "locked_test_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    rows = []
    for name, result in summary["conditions"].items():
        rows.append({"condition": name, **result})
    pd.DataFrame(rows).to_csv(output / "locked_test_table.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
