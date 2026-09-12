#!/usr/bin/env python3
"""Stratify paired docking performance by pretraining sequence exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def find_id_column(table: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in table.columns:
            raise ValueError(f"Requested ID column {requested!r} is absent")
        return requested
    for candidate in ("pdb_id", "complex_name", "name"):
        if candidate in table.columns:
            return candidate
    raise ValueError("Could not infer ID column; use --*-id-column")


def read_metric(
    path: Path, requested_id_column: str | None, metric_column: str, output_name: str
) -> pd.DataFrame:
    table = pd.read_csv(path)
    id_column = find_id_column(table, requested_id_column)
    if metric_column not in table.columns:
        raise ValueError(f"{path} lacks metric column {metric_column!r}")
    result = table[[id_column, metric_column]].copy()
    result.columns = ["pdb_id", output_name]
    result["pdb_id"] = normalize_ids(result["pdb_id"])
    result[output_name] = pd.to_numeric(result[output_name], errors="coerce")
    if result["pdb_id"].duplicated().any():
        raise ValueError(f"Duplicate complex IDs in {path}")
    return result


def subgroup_masks(identity: pd.Series) -> dict[str, pd.Series]:
    # Missing identity means no coverage-qualified MMseqs hit and is therefore
    # sequence-novel. Unresolved extraction must be excluded before this call.
    filled = pd.to_numeric(identity, errors="coerce").fillna(-np.inf)
    return {
        "all": pd.Series(True, index=identity.index),
        "identity_lt90": filled.lt(90.0),
        "identity_lt50": filled.lt(50.0),
        "identity_lt30": filled.lt(30.0),
    }


def bootstrap_delta(
    baseline: np.ndarray, guided: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(baseline)
    indices = rng.integers(0, n, size=(repeats, n))
    delta = guided.astype(float) - baseline.astype(float)
    values = 100.0 * delta[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", required=True, type=Path)
    parser.add_argument(
        "--scope", choices=("all_receptor", "training_contact"), default="all_receptor"
    )
    parser.add_argument("--baseline-table", required=True, type=Path)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--baseline-metric", default="top1_rmsd_A")
    parser.add_argument("--baseline-id-column")
    parser.add_argument("--guided-table", required=True, type=Path)
    parser.add_argument("--guided-name", required=True)
    parser.add_argument("--guided-metric", default="top1_rmsd_A")
    parser.add_argument("--guided-id-column")
    parser.add_argument("--rmsd-cutoff", type=float, default=2.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument(
        "--allow-unresolved-exposure",
        action="store_true",
        help="Exclude unresolved exposure rows instead of failing (sensitivity analysis only).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.rmsd_cutoff <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("cutoff and bootstrap repeats must be positive")

    exposure = pd.read_csv(args.exposure)
    if "pdb_id" not in exposure.columns:
        raise ValueError("Exposure table requires pdb_id")
    identity_column = f"{args.scope}_fident_pct"
    class_column = f"{args.scope}_exposure_class"
    required = {identity_column, class_column}
    if not required <= set(exposure.columns):
        raise ValueError(f"Exposure table lacks {sorted(required - set(exposure.columns))}")
    exposure["pdb_id"] = normalize_ids(exposure["pdb_id"])
    if exposure["pdb_id"].duplicated().any():
        raise ValueError("Exposure table contains duplicate complex IDs")
    baseline = read_metric(
        args.baseline_table, args.baseline_id_column, args.baseline_metric, "baseline_rmsd_A"
    )
    guided = read_metric(
        args.guided_table, args.guided_id_column, args.guided_metric, "guided_rmsd_A"
    )
    joined = exposure.merge(baseline, on="pdb_id", how="left", validate="one_to_one")
    joined = joined.merge(guided, on="pdb_id", how="left", validate="one_to_one")
    joined["baseline_success"] = (
        np.isfinite(joined["baseline_rmsd_A"])
        & joined["baseline_rmsd_A"].lt(args.rmsd_cutoff)
    )
    joined["guided_success"] = (
        np.isfinite(joined["guided_rmsd_A"])
        & joined["guided_rmsd_A"].lt(args.rmsd_cutoff)
    )
    joined["unresolved_exposure"] = joined[class_column].astype(str).eq("unresolved")

    unresolved_n = int(joined["unresolved_exposure"].sum())
    if unresolved_n and not args.allow_unresolved_exposure:
        raise ValueError(
            f"Exposure audit has {unresolved_n} unresolved complexes; resolve extraction "
            "failures or explicitly use --allow-unresolved-exposure for a sensitivity analysis"
        )

    resolved = joined[~joined["unresolved_exposure"]].copy()
    masks = subgroup_masks(resolved[identity_column])
    rows: list[dict[str, object]] = []
    for index, (subgroup, mask) in enumerate(masks.items()):
        group = resolved[mask].copy()
        n = len(group)
        if not n:
            continue
        baseline_success = group["baseline_success"].to_numpy(bool)
        guided_success = group["guided_success"].to_numpy(bool)
        gained = int((~baseline_success & guided_success).sum())
        lost = int((baseline_success & ~guided_success).sum())
        try:
            from scipy.stats import binomtest

            mcnemar = float(
                binomtest(gained, gained + lost, p=0.5).pvalue
                if gained + lost else 1.0
            )
        except ImportError:  # pragma: no cover
            mcnemar = np.nan
        low, high = bootstrap_delta(
            baseline_success,
            guided_success,
            args.bootstrap_repeats,
            args.seed + index,
        )
        baseline_count = int(baseline_success.sum())
        guided_count = int(guided_success.sum())
        rows.append({
            "scope": args.scope,
            "subgroup": subgroup,
            "num_complexes": n,
            "baseline_method": args.baseline_name,
            "guided_method": args.guided_name,
            "rmsd_cutoff_A": args.rmsd_cutoff,
            "baseline_success_count": baseline_count,
            "baseline_success_pct": 100.0 * baseline_count / n,
            "guided_success_count": guided_count,
            "guided_success_pct": 100.0 * guided_count / n,
            "delta_guided_minus_baseline_pp": 100.0 * (guided_count - baseline_count) / n,
            "paired_bootstrap_95ci_low_pp": low,
            "paired_bootstrap_95ci_high_pp": high,
            "gained": gained,
            "lost": lost,
            "exact_mcnemar_pvalue": mcnemar,
            "baseline_scored_n": int(np.isfinite(group["baseline_rmsd_A"]).sum()),
            "guided_scored_n": int(np.isfinite(group["guided_rmsd_A"]).sum()),
        })

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "performance_by_exposure_per_complex.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "performance_by_exposure_summary.csv", index=False)
    metadata = {
        "scope": args.scope,
        "exposure_table": str(args.exposure.resolve()),
        "baseline": args.baseline_name,
        "guided": args.guided_name,
        "rmsd_cutoff_A": args.rmsd_cutoff,
        "fixed_denominator_n": len(joined),
        "unresolved_exposure_n": unresolved_n,
        "missing_predictions_count_as_failure": True,
        "subsets": {
            "identity_lt90": "maximum coverage-qualified identity <90% or no hit",
            "identity_lt50": "maximum coverage-qualified identity <50% or no hit",
            "identity_lt30": "maximum coverage-qualified identity <30% or no hit",
        },
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"\nPRETRAINING_EXPOSURE_PERFORMANCE_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
