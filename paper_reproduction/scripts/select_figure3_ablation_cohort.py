#!/usr/bin/env python3
"""Freeze a result-blind, covariate-stratified Figure 3 ablation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = (
    "ligand_heavy_atoms",
    "ligand_rotatable_bonds",
    "num_flexible_sidechains",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tertile(values: pd.Series) -> pd.Series:
    percentile = values.rank(method="average", pct=True)
    return np.minimum(np.floor(percentile.to_numpy(float) * 3.0), 2).astype(int)


def proportional_allocations(counts: pd.Series, sample_size: int) -> pd.Series:
    raw = counts.astype(float) * float(sample_size) / float(counts.sum())
    allocation = np.floor(raw).astype(int)
    remaining = int(sample_size - allocation.sum())
    order = sorted(counts.index, key=lambda key: (-(raw[key] - allocation[key]), str(key)))
    for key in order[:remaining]:
        allocation[key] += 1
    if int(allocation.sum()) != sample_size or (allocation > counts).any():
        raise AssertionError("Invalid stratified allocation")
    return allocation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--covariates", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    args = parser.parse_args(argv)

    cohort_path = args.cohort.expanduser().resolve()
    covariates_path = args.covariates.expanduser().resolve()
    cohort = [x.strip().upper() for x in cohort_path.read_text().splitlines() if x.strip()]
    if len(cohort) != len(set(cohort)) or not 0 < args.sample_size <= len(cohort):
        raise ValueError("Cohort must be unique and sample-size must be in range")

    columns = ["complex_name", *FEATURES]
    table = pd.read_csv(covariates_path, usecols=columns, dtype={"complex_name": str})
    table["complex_name"] = table["complex_name"].str.upper()
    if table["complex_name"].duplicated().any():
        raise ValueError("Covariate table has duplicate complex_name values")
    indexed = table.set_index("complex_name")
    missing = [name for name in cohort if name not in indexed.index]
    if missing:
        raise ValueError(f"Covariates missing cohort IDs: {missing[:10]}")
    table = indexed.loc[cohort].reset_index()
    for feature in FEATURES:
        table[feature] = pd.to_numeric(table[feature], errors="raise")
        if not np.isfinite(table[feature]).all():
            raise ValueError(f"Non-finite selection covariate: {feature}")
        table[f"{feature}_tertile"] = tertile(table[feature])

    table["stratum"] = (
        "H" + table["ligand_heavy_atoms_tertile"].astype(str)
        + "_R" + table["ligand_rotatable_bonds_tertile"].astype(str)
        + "_S" + table["num_flexible_sidechains_tertile"].astype(str)
    )
    counts = table.groupby("stratum", sort=True).size()
    allocations = proportional_allocations(counts, args.sample_size)
    table["selection_hash"] = table["complex_name"].map(
        lambda name: hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest()
    )
    table["stratum_size"] = table["stratum"].map(counts).astype(int)
    table["stratum_allocation"] = table["stratum"].map(allocations).astype(int)
    table["selected"] = False
    for stratum, group in table.groupby("stratum", sort=True):
        chosen = group.sort_values(["selection_hash", "complex_name"]).head(
            int(allocations[stratum])
        ).index
        table.loc[chosen, "selected"] = True

    selected = table.loc[table["selected"]].sort_values(
        ["stratum", "selection_hash", "complex_name"]
    )
    if len(selected) != args.sample_size:
        raise AssertionError("Selected cohort size differs from request")

    output = args.output.expanduser().resolve()
    audit_csv = args.audit_csv.expanduser().resolve()
    audit_json = args.audit_json.expanduser().resolve()
    for path in (output, audit_csv, audit_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{name}\n" for name in selected["complex_name"]))
    table.to_csv(audit_csv, index=False)

    balance = {}
    for feature in FEATURES:
        full = table[feature].to_numpy(float)
        subset = selected[feature].to_numpy(float)
        pooled_sd = float(np.std(full, ddof=1))
        balance[feature] = {
            "full_mean": float(np.mean(full)),
            "subset_mean": float(np.mean(subset)),
            "full_median": float(np.median(full)),
            "subset_median": float(np.median(subset)),
            "standardized_mean_difference": (
                float((np.mean(subset) - np.mean(full)) / pooled_sd) if pooled_sd else 0.0
            ),
        }
    audit = {
        "selection_policy": "proportional 3x3x3 covariate-stratified hash sample",
        "selection_is_result_blind": True,
        "prohibited_outcomes_used": [],
        "selection_features": list(FEATURES),
        "seed": args.seed,
        "full_cohort_n": len(cohort),
        "ablation_cohort_n": len(selected),
        "cohort_source": str(cohort_path),
        "covariate_source": str(covariates_path),
        "cohort_source_sha256": file_sha256(cohort_path),
        "covariate_source_sha256": file_sha256(covariates_path),
        "selected_ids_sha256": file_sha256(output),
        "stratum_full_counts": {str(key): int(value) for key, value in counts.items()},
        "stratum_selected_counts": {
            str(key): int(value) for key, value in allocations.items()
        },
        "balance": balance,
    }
    audit_json.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"FIGURE3_ABLATION_COHORT_OK: {output} ({len(selected)} complexes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
