#!/usr/bin/env python3
"""Stratify frozen-cohort docking accuracy by PDBBind ligand exposure.

The authoritative cohort is always the left table. Missing predictions remain
in subgroup denominators and count as failures for RMSD success endpoints.
Median RMSD is reported among scored cases, together with its scored count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd


ID_CANDIDATES = ("pdb_id", "complex_name", "complex_id", "name")
REQUIRED_EXPOSURE_COLUMNS = {
    "pdb_id",
    "exact_inchikey_in_pdbbind_train",
    "connectivity_in_pdbbind_train",
    "scaffold_evaluable",
    "murcko_scaffold_in_pdbbind_train",
    "max_ecfp4_tanimoto_to_pdbbind_train",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", required=True, type=Path)
    parser.add_argument("--performance", required=True, type=Path)
    parser.add_argument("--authoritative-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method-name", default="CryoDiffDock-Local")
    parser.add_argument("--id-column")
    parser.add_argument("--metric-column", default="top1_rmsd_A")
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper()


def truthy(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value is True
        or str(value).strip().lower() in {"1", "true", "yes", "y"}
    )


def read_ids(path: Path) -> list[str]:
    ids = [
        line.split("#", 1)[0].strip().upper()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    ids = [value for value in ids if value]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return ids


def infer_id_column(table: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in table.columns:
            raise ValueError(f"requested ID column {requested!r} is absent")
        return requested
    found = next((column for column in ID_CANDIDATES if column in table), None)
    if found is None:
        raise ValueError(f"could not infer ID column from {list(table.columns)}")
    return found


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    return 100.0 * (center - half), 100.0 * (center + half)


def bootstrap_median_interval(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    medians = np.median(values[indices], axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def subgroup_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    exact = truthy(table["exact_inchikey_in_pdbbind_train"])
    connectivity = truthy(table["connectivity_in_pdbbind_train"])
    scaffold_evaluable = truthy(table["scaffold_evaluable"])
    shared_scaffold = truthy(table["murcko_scaffold_in_pdbbind_train"])
    similarity = pd.to_numeric(
        table["max_ecfp4_tanimoto_to_pdbbind_train"], errors="coerce"
    )
    return {
        "all": pd.Series(True, index=table.index),
        "no_exact_ligand_identity": ~exact,
        "no_connectivity_match": ~connectivity,
        "no_shared_murcko_scaffold": scaffold_evaluable & ~shared_scaffold,
        "max_tanimoto_lt_0.5": similarity.lt(0.5),
        "max_tanimoto_lt_0.7": similarity.lt(0.7),
        "exact_ligand_identity": exact,
        "connectivity_match": connectivity,
        "shared_murcko_scaffold": scaffold_evaluable & shared_scaffold,
        "max_tanimoto_ge_0.5": similarity.ge(0.5),
        "max_tanimoto_ge_0.7": similarity.ge(0.7),
    }


def summarize(
    joined: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    membership = joined.copy()
    masks = subgroup_masks(joined)
    for subgroup, mask in masks.items():
        membership[f"subgroup__{subgroup}"] = mask
    for index, (subgroup, mask) in enumerate(masks.items()):
        group = joined.loc[mask].copy()
        rmsd = pd.to_numeric(group["top1_rmsd_A"], errors="coerce")
        scored = rmsd[np.isfinite(rmsd)].to_numpy(float)
        n = len(group)
        record: dict[str, object] = {
            "method": args.method_name,
            "subgroup": subgroup,
            "n_cohort": n,
            "n_scored": len(scored),
            "n_missing_counted_as_failure": n - len(scored),
            "median_rmsd_A_among_scored": float(np.median(scored)) if len(scored) else math.nan,
            "q25_rmsd_A_among_scored": float(np.quantile(scored, 0.25)) if len(scored) else math.nan,
            "q75_rmsd_A_among_scored": float(np.quantile(scored, 0.75)) if len(scored) else math.nan,
        }
        median_low, median_high = bootstrap_median_interval(
            scored, args.bootstrap_repeats, args.seed + index
        )
        record["median_rmsd_ci95_low_A"] = median_low
        record["median_rmsd_ci95_high_A"] = median_high
        for cutoff, label in ((1.0, "lt1"), (2.0, "lt2")):
            # Missing RMSD remains False and therefore counts as failure.
            successes = int((np.isfinite(rmsd) & rmsd.lt(cutoff)).sum())
            low, high = wilson_interval(successes, n)
            record[f"top1_{label}_count"] = successes
            record[f"top1_{label}_pct"] = 100.0 * successes / n if n else math.nan
            record[f"top1_{label}_ci95_low_pct"] = low
            record[f"top1_{label}_ci95_high_pct"] = high
        summary_rows.append(record)
    return pd.DataFrame(summary_rows), membership


def make_preview(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        "all",
        "max_tanimoto_lt_0.5",
        "max_tanimoto_lt_0.7",
        "no_connectivity_match",
        "no_shared_murcko_scaffold",
    ]
    labels = {
        "all": "All",
        "max_tanimoto_lt_0.5": "Max Tanimoto <0.5",
        "max_tanimoto_lt_0.7": "Max Tanimoto <0.7",
        "no_connectivity_match": "No connectivity match",
        "no_shared_murcko_scaffold": "No shared scaffold",
    }
    frame = summary.set_index("subgroup").loc[selected].reset_index()
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "top1_lt1", "Top-1 RMSD <1 Å"),
        (axes[1], "top1_lt2", "Top-1 RMSD <2 Å"),
    ):
        value = frame[f"{prefix}_pct"].to_numpy(float)
        low = frame[f"{prefix}_ci95_low_pct"].to_numpy(float)
        high = frame[f"{prefix}_ci95_high_pct"].to_numpy(float)
        axis.errorbar(
            value, y, xerr=np.vstack([value - low, high - value]), fmt="o",
            color="#0072B2", ecolor="#7A8793", capsize=3,
        )
        axis.set_xlim(0, 100)
        axis.set_xlabel("Success rate (%)")
        axis.set_title(title)
        axis.grid(axis="x", color="#E1E4E8", linewidth=0.7)
    median = frame["median_rmsd_A_among_scored"].to_numpy(float)
    low = frame["median_rmsd_ci95_low_A"].to_numpy(float)
    high = frame["median_rmsd_ci95_high_A"].to_numpy(float)
    axes[2].errorbar(
        median, y, xerr=np.vstack([median - low, high - median]), fmt="o",
        color="#D55E00", ecolor="#7A8793", capsize=3,
    )
    axes[2].set_xlabel("Median RMSD (Å)")
    axes[2].set_title("Median Top-1 RMSD")
    axes[2].grid(axis="x", color="#E1E4E8", linewidth=0.7)
    tick_labels = [f"{labels[row.subgroup]}\n(n={row.n_cohort})" for row in frame.itertuples()]
    axes[0].set_yticks(y, tick_labels)
    axes[0].invert_yaxis()
    for axis in axes[1:]:
        axis.set_yticks(y, [])
        axis.invert_yaxis()
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.expected_n < 1 or args.bootstrap_repeats < 100:
        raise ValueError("expected cohort and bootstrap repeats must be positive")
    exposure_path = args.exposure.expanduser().resolve()
    performance_path = args.performance.expanduser().resolve()
    ids_path = args.authoritative_ids.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (exposure_path, performance_path, ids_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    cohort_ids = read_ids(ids_path)
    if len(cohort_ids) != args.expected_n:
        raise ValueError(f"expected {args.expected_n} IDs, found {len(cohort_ids)}")
    order = pd.DataFrame({"cohort_order": range(len(cohort_ids)), "pdb_id": cohort_ids})

    exposure = pd.read_csv(exposure_path, keep_default_na=False)
    missing_columns = sorted(REQUIRED_EXPOSURE_COLUMNS - set(exposure.columns))
    if missing_columns:
        raise ValueError(f"exposure table lacks columns: {missing_columns}")
    exposure = exposure.copy()
    exposure["pdb_id"] = normalize_ids(exposure["pdb_id"])
    exposure = exposure.loc[exposure["pdb_id"].isin(cohort_ids)]
    if exposure["pdb_id"].duplicated().any():
        raise ValueError("exposure table contains duplicate authoritative IDs")

    performance = pd.read_csv(performance_path, keep_default_na=False)
    id_column = infer_id_column(performance, args.id_column)
    if args.metric_column not in performance:
        raise ValueError(f"performance table lacks {args.metric_column!r}")
    performance = performance[[id_column, args.metric_column]].copy()
    performance.columns = ["pdb_id", "top1_rmsd_A"]
    performance["pdb_id"] = normalize_ids(performance["pdb_id"])
    performance = performance.loc[performance["pdb_id"].isin(cohort_ids)]
    if performance["pdb_id"].duplicated().any():
        raise ValueError("performance table contains duplicate authoritative IDs")
    performance["top1_rmsd_A"] = pd.to_numeric(
        performance["top1_rmsd_A"], errors="coerce"
    )

    joined = order.merge(exposure, on="pdb_id", how="left", validate="one_to_one")
    joined = joined.merge(performance, on="pdb_id", how="left", validate="one_to_one")
    missing_exposure = joined.loc[
        joined["max_ecfp4_tanimoto_to_pdbbind_train"].eq(""), "pdb_id"
    ].tolist()
    # After a CSV merge, absent rows may also be represented by NaN.
    missing_exposure += joined.loc[
        joined["max_ecfp4_tanimoto_to_pdbbind_train"].isna(), "pdb_id"
    ].tolist()
    missing_exposure = sorted(set(missing_exposure))
    summary, membership = summarize(joined, args)
    summary.to_csv(output / "ligand_exposure_performance_summary.csv", index=False)
    membership.to_csv(output / "ligand_exposure_performance_per_complex.csv", index=False)
    if not args.skip_plot:
        make_preview(summary, output / "ligand_exposure_performance_preview.png")

    expected_counts = {
        "all": 174,
        "max_tanimoto_lt_0.5": 152,
        "max_tanimoto_lt_0.7": 162,
        "no_connectivity_match": 161,
        "no_shared_murcko_scaffold": 139,
    }
    observed_counts = summary.set_index("subgroup")["n_cohort"].astype(int).to_dict()
    checks = [
        {"check": "authoritative_cohort_count", "observed": len(cohort_ids),
         "expected": args.expected_n, "passed": len(cohort_ids) == args.expected_n},
        {"check": "exposure_rows_complete", "observed": len(cohort_ids) - len(missing_exposure),
         "expected": len(cohort_ids), "passed": not missing_exposure},
    ]
    for subgroup, expected in expected_counts.items():
        checks.append({
            "check": f"subgroup_count__{subgroup}",
            "observed": observed_counts.get(subgroup, -1),
            "expected": expected,
            "passed": observed_counts.get(subgroup) == expected,
        })
    validations = pd.DataFrame(checks)
    validations.to_csv(output / "validation_checks.csv", index=False)
    metadata = {
        "analysis": "Frozen Local accuracy stratified by PDBBind ligand exposure",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": args.method_name,
        "fixed_denominator_policy": "authoritative cohort is left table; missing predictions count as RMSD-threshold failures",
        "median_policy": "median and bootstrap CI among scored cases; n_scored reported",
        "scaffold_policy": "no-shared-scaffold subgroup excludes scaffold-unevaluable acyclic ligands",
        "inputs": {
            "exposure": {"path": str(exposure_path), "sha256": sha256_file(exposure_path)},
            "performance": {"path": str(performance_path), "sha256": sha256_file(performance_path)},
            "authoritative_ids": {"path": str(ids_path), "sha256": sha256_file(ids_path)},
        },
        "parameters": {"bootstrap_repeats": args.bootstrap_repeats, "seed": args.seed},
        "missing_exposure_ids": missing_exposure,
        "missing_prediction_ids": membership.loc[
            ~np.isfinite(pd.to_numeric(membership["top1_rmsd_A"], errors="coerce")), "pdb_id"
        ].tolist(),
        "validation_passed": bool(validations["passed"].all()),
    }
    (output / "ligand_exposure_performance_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"\nLIGAND_EXPOSURE_PERFORMANCE_COMPLETE: {output}")
    if args.strict and not metadata["validation_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
