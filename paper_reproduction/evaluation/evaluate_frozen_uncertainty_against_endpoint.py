#!/usr/bin/env python3
"""Evaluate a frozen uncertainty feature against a different fixed endpoint.

This analysis never recomputes, selects, or tunes the uncertainty feature.  It
joins an existing per-complex uncertainty table to a prespecified endpoint
table, then evaluates failure discrimination and selective-prediction curves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex_id", "id")
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "ok"}


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def choose_column(columns, candidates, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; found {list(columns)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    ids = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return ids


def unique_id_table(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    id_column = choose_column(table.columns, ID_COLUMNS, f"{label} ID")
    table = table.copy()
    table["complex_name"] = table[id_column].map(normalize_id)
    duplicates = sorted(
        table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate IDs: {duplicates[:20]}")
    return table


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        value = binary_auc(labels[indices], scores[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-table", required=True, type=Path)
    parser.add_argument("--endpoint-table", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument(
        "--largest-cluster-column",
        default="largest_cluster_fraction_at_2p0A",
    )
    parser.add_argument("--endpoint-rmsd-column", default="v2_rmsd_A")
    parser.add_argument("--endpoint-status-column", default="v2_scored")
    parser.add_argument("--endpoint-name", default="CryoDiffDock-Local V2 selected Top-1")
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--missing-counts-as-failure", action="store_true")
    args = parser.parse_args()

    if args.expected_n <= 0 or args.bootstrap_repeats <= 0 or args.success_cutoff <= 0:
        raise ValueError("expected-n, bootstrap-repeats, and success-cutoff must be positive")

    uncertainty_path = args.uncertainty_table.expanduser().resolve()
    endpoint_path = args.endpoint_table.expanduser().resolve()
    cohort_path = args.cohort.expanduser().resolve()
    uncertainty = unique_id_table(uncertainty_path, "uncertainty table")
    endpoint = unique_id_table(endpoint_path, "endpoint table")
    cohort_ids = read_ids(cohort_path)
    if len(cohort_ids) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cohort IDs, found {len(cohort_ids)}")

    required_uncertainty = {args.largest_cluster_column}
    missing_uncertainty_columns = sorted(required_uncertainty - set(uncertainty.columns))
    if missing_uncertainty_columns:
        raise ValueError(f"Uncertainty table missing columns: {missing_uncertainty_columns}")
    if args.endpoint_rmsd_column not in endpoint.columns:
        raise ValueError(f"Endpoint table lacks {args.endpoint_rmsd_column!r}")
    if args.endpoint_status_column and args.endpoint_status_column not in endpoint.columns:
        raise ValueError(f"Endpoint table lacks {args.endpoint_status_column!r}")

    order = pd.DataFrame({
        "cohort_order": np.arange(len(cohort_ids)),
        "complex_name": cohort_ids,
    })
    uncertainty_ids = set(uncertainty["complex_name"])
    endpoint_ids = set(endpoint["complex_name"])
    coverage = {
        "cohort_n": len(cohort_ids),
        "uncertainty_rows_in_cohort": len(set(cohort_ids) & uncertainty_ids),
        "endpoint_rows_in_cohort": len(set(cohort_ids) & endpoint_ids),
        "missing_uncertainty_ids": sorted(set(cohort_ids) - uncertainty_ids),
        "missing_endpoint_ids": sorted(set(cohort_ids) - endpoint_ids),
        "extra_uncertainty_ids": sorted(uncertainty_ids - set(cohort_ids)),
        "extra_endpoint_ids": sorted(endpoint_ids - set(cohort_ids)),
    }
    if coverage["missing_uncertainty_ids"]:
        raise ValueError(f"Uncertainty coverage is incomplete: {coverage}")

    uncertainty_selected = uncertainty[
        uncertainty["complex_name"].isin(cohort_ids)
    ].copy()
    if "top1_failure" in uncertainty_selected:
        uncertainty_selected = uncertainty_selected.rename(
            columns={"top1_failure": "guided40_top1_failure"}
        )
    if "top1_success" in uncertainty_selected:
        uncertainty_selected = uncertainty_selected.rename(
            columns={"top1_success": "guided40_top1_success"}
        )
    if "selected_top1_rmsd_A" in uncertainty_selected:
        uncertainty_selected = uncertainty_selected.rename(
            columns={"selected_top1_rmsd_A": "guided40_selected_top1_rmsd_A"}
        )

    endpoint_selected = endpoint[[
        "complex_name", args.endpoint_rmsd_column, args.endpoint_status_column
    ]].copy()
    endpoint_selected = endpoint_selected.rename(columns={
        args.endpoint_rmsd_column: "endpoint_top1_rmsd_A",
        args.endpoint_status_column: "endpoint_reported_scored",
    })
    endpoint_selected["endpoint_top1_rmsd_A"] = pd.to_numeric(
        endpoint_selected["endpoint_top1_rmsd_A"], errors="coerce"
    )
    endpoint_selected["endpoint_reported_scored"] = (
        endpoint_selected["endpoint_reported_scored"]
        .astype(str).str.strip().str.lower().isin(TRUE_VALUES)
    )
    endpoint_selected["endpoint_scored"] = (
        endpoint_selected["endpoint_reported_scored"]
        & np.isfinite(endpoint_selected["endpoint_top1_rmsd_A"])
    )

    cases = order.merge(
        uncertainty_selected, on="complex_name", how="left", validate="one_to_one"
    ).merge(
        endpoint_selected, on="complex_name", how="left", validate="one_to_one"
    )
    cases["endpoint_scored"] = cases["endpoint_scored"].fillna(False).astype(bool)
    cases["uncertainty"] = 1.0 - pd.to_numeric(
        cases[args.largest_cluster_column], errors="coerce"
    )
    if not np.isfinite(cases["uncertainty"]).all():
        raise ValueError("Frozen uncertainty contains missing or non-finite values")
    cases["top1_success"] = (
        cases["endpoint_scored"]
        & (cases["endpoint_top1_rmsd_A"] < args.success_cutoff)
    )
    cases["top1_failure"] = ~cases["top1_success"]

    analyzed = cases if args.missing_counts_as_failure else cases[cases["endpoint_scored"]].copy()
    labels = analyzed["top1_failure"].to_numpy(bool)
    scores = analyzed["uncertainty"].to_numpy(float)
    auc = binary_auc(labels, scores)
    low, high = bootstrap_auc(labels, scores, args.bootstrap_repeats, args.seed)

    ordered = np.argsort(scores, kind="stable")
    risk_rows = []
    for retained_n in range(1, len(analyzed) + 1):
        retained = ordered[:retained_n]
        risk_rows.append({
            "retained_n": retained_n,
            "coverage": retained_n / len(analyzed),
            "success_count": int((~labels[retained]).sum()),
            "success_pct": 100.0 * float((~labels[retained]).mean()),
            "failure_pct": 100.0 * float(labels[retained].mean()),
            "maximum_retained_uncertainty": float(scores[retained].max()),
        })
    risk = pd.DataFrame(risk_rows)
    half_n = int(math.ceil(0.5 * len(analyzed)))
    half = risk.loc[risk["retained_n"].eq(half_n)].iloc[0]

    quartile_rows = []
    for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
        quartile_rows.append({
            "uncertainty_quartile": quartile,
            "n": len(indices),
            "success_count": int((~labels[indices]).sum()),
            "success_pct": 100.0 * float((~labels[indices]).mean()),
            "failure_pct": 100.0 * float(labels[indices].mean()),
            "median_endpoint_top1_rmsd_A_among_scored": float(
                analyzed.iloc[indices]["endpoint_top1_rmsd_A"].median()
            ),
            "median_uncertainty": float(np.median(scores[indices])),
        })
    quartiles = pd.DataFrame(quartile_rows)
    failure_rates = quartiles["failure_pct"].to_numpy(float)
    monotonic = bool(np.all(np.diff(failure_rates) >= -1e-12))

    scored_rmsd = analyzed["endpoint_top1_rmsd_A"].to_numpy(float)
    finite_rmsd = np.isfinite(scored_rmsd)
    correlation = spearmanr(
        scores[finite_rmsd], scored_rmsd[finite_rmsd]
    ).statistic if finite_rmsd.sum() >= 2 else float("nan")
    full_success_pct = 100.0 * float((~labels).mean())
    summary = {
        "analysis_role": "frozen_cross_endpoint_one_shot_check",
        "cohort_n": len(cases),
        "analyzed_n": len(analyzed),
        "endpoint_name": args.endpoint_name,
        "endpoint_success_definition": f"Top-1 RMSD < {args.success_cutoff:g} A",
        "endpoint_scored_n": int(cases["endpoint_scored"].sum()),
        "endpoint_missing_or_failed_n": int((~cases["endpoint_scored"]).sum()),
        "missing_counts_as_failure": bool(args.missing_counts_as_failure),
        "success_count": int((~labels).sum()),
        "success_pct": full_success_pct,
        "failure_count": int(labels.sum()),
        "failure_pct": 100.0 * float(labels.mean()),
        "frozen_uncertainty_feature": "1 - largest cluster fraction at 2 A",
        "feature_recomputed_or_retuned": False,
        "uses_reference_for_uncertainty_feature": False,
        "failure_auroc": auc,
        "failure_auroc_95ci": [low, high],
        "spearman_uncertainty_vs_endpoint_top1_rmsd_among_scored": float(correlation),
        "most_certain_50pct_n": half_n,
        "most_certain_50pct_success_pct": float(half["success_pct"]),
        "most_certain_50pct_gain_pp": float(half["success_pct"] - full_success_pct),
        "failure_rate_monotonic_across_uncertainty_quartiles": monotonic,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "inputs": {
            "uncertainty_table": {"path": str(uncertainty_path), "sha256": sha256(uncertainty_path)},
            "endpoint_table": {"path": str(endpoint_path), "sha256": sha256(endpoint_path)},
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
        },
        "interpretation": (
            "The previously frozen guided-ensemble uncertainty feature is evaluated "
            "against the final V2 selected-pose endpoint. No uncertainty parameter is "
            "selected using this endpoint."
        ),
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output / "per_complex_uncertainty_v2_labels.csv", index=False)
    risk.to_csv(output / "risk_coverage.csv", index=False)
    quartiles.to_csv(output / "uncertainty_quartiles.csv", index=False)
    (output / "coverage_audit.json").write_text(json.dumps(coverage, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("COVERAGE AUDIT")
    print(json.dumps(coverage, indent=2))
    print("\nFROZEN UNCERTAINTY AGAINST V2 ENDPOINT")
    print(json.dumps(summary, indent=2))
    print("\nUNCERTAINTY QUARTILES")
    print(quartiles.to_string(index=False))
    print(f"\nFROZEN_UNCERTAINTY_ENDPOINT_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
