#!/usr/bin/env python3
"""Exactly extract an authoritative cohort from completed Local V2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "complex_id")


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID list is empty or contains duplicates: {path}")
    return values


def read_table(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={column: str for column in ID_COLUMNS})
    id_column = next((column for column in ID_COLUMNS if column in table.columns), None)
    if id_column is None:
        raise ValueError(f"{label} has no ID column: {list(table.columns)}")
    table = table.copy()
    table["complex_name"] = table[id_column].astype(str).str.strip().str.upper()
    table["source_file"] = str(path)
    return table


def collect(paths: list[Path], label: str) -> pd.DataFrame:
    tables = []
    for priority, path in enumerate(paths):
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        table = read_table(resolved, label)
        table["source_priority"] = priority
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)


def check_duplicate_metrics(
    table: pd.DataFrame, columns: tuple[str, ...], label: str, tolerance: float = 1e-6
) -> None:
    duplicate_ids = table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    conflicts = []
    for complex_name in duplicate_ids:
        group = table[table["complex_name"].eq(complex_name)]
        for column in columns:
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna().to_numpy()
            if len(values) > 1 and float(np.max(values) - np.min(values)) > tolerance:
                conflicts.append({
                    "complex_name": complex_name,
                    "column": column,
                    "values": values.tolist(),
                    "sources": group["source_file"].tolist(),
                })
    if conflicts:
        raise ValueError(f"Conflicting duplicate {label} rows: {conflicts[:10]}")


def choose_first(table: pd.DataFrame, cohort: set[str]) -> pd.DataFrame:
    return (
        table[table["complex_name"].isin(cohort)]
        .sort_values("source_priority", kind="stable")
        .drop_duplicates("complex_name", keep="first")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--pose-scores", action="append", required=True, type=Path)
    parser.add_argument("--v2-audit", action="append", required=True, type=Path)
    parser.add_argument("--v2-metrics", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    args = parser.parse_args()

    cohort_path = args.cohort.expanduser().resolve()
    cohort_ids = read_ids(cohort_path)
    if len(cohort_ids) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cohort IDs, found {len(cohort_ids)}")
    cohort_set = set(cohort_ids)
    order = pd.DataFrame({
        "authoritative_index": range(len(cohort_ids)),
        "complex_name": cohort_ids,
    })

    manifest = collect(args.manifest, "manifest")
    manifest = choose_first(manifest, cohort_set)
    selected_manifest = order.merge(
        manifest, on="complex_name", how="left",
        validate="one_to_one", sort=False,
    )
    absent_manifest = [value for value in cohort_ids if value not in set(manifest["complex_name"])]

    scores = pd.concat(
        [read_table(path.expanduser().resolve(), "pose scores") for path in args.pose_scores],
        ignore_index=True, sort=False,
    )
    selected_scores = scores[scores["complex_name"].isin(cohort_set)].copy()
    score_counts = selected_scores.groupby("complex_name").size().to_dict()
    absent_scores = [value for value in cohort_ids if score_counts.get(value, 0) == 0]
    unexpected_score_counts = {
        value: int(score_counts.get(value, 0))
        for value in cohort_ids if score_counts.get(value, 0) != 80
    }

    audits_all = collect(args.v2_audit, "V2 audit")
    check_duplicate_metrics(
        audits_all,
        ("selection_ready_seconds", "top1_materialization_seconds", "deployment_runtime_seconds"),
        "V2 audit",
        tolerance=1e-3,
    )
    audits = choose_first(audits_all, cohort_set)
    if "deployment_runtime_seconds" not in audits:
        if {"selection_ready_seconds", "top1_materialization_seconds"}.issubset(audits.columns):
            audits["deployment_runtime_seconds"] = (
                pd.to_numeric(audits["selection_ready_seconds"], errors="coerce")
                + pd.to_numeric(audits["top1_materialization_seconds"], errors="coerce")
            )
        else:
            raise ValueError(
                "V2 audit lacks deployment_runtime_seconds and its two components"
            )
    audits["deployment_runtime_seconds"] = pd.to_numeric(
        audits["deployment_runtime_seconds"], errors="coerce"
    )
    absent_audit = [value for value in cohort_ids if value not in set(audits["complex_name"])]
    missing_deployment_runtime = audits.loc[
        audits["deployment_runtime_seconds"].isna(), "complex_name"
    ].tolist()

    metrics_all = collect(args.v2_metrics, "V2 metrics")
    check_duplicate_metrics(metrics_all, ("top1_rmsd_A",), "V2 metrics")
    metrics = choose_first(metrics_all, cohort_set)
    if "top1_rmsd_A" not in metrics:
        raise ValueError("V2 metrics lacks top1_rmsd_A")
    metrics["top1_rmsd_A"] = pd.to_numeric(metrics["top1_rmsd_A"], errors="coerce")
    absent_metrics = [value for value in cohort_ids if value not in set(metrics["complex_name"])]
    invalid_metrics = metrics.loc[~np.isfinite(metrics["top1_rmsd_A"]), "complex_name"].tolist()

    fatal = {
        "manifest_missing_ids": absent_manifest,
        "pose_scores_missing_ids": absent_scores,
        "pose_score_count_not_80": unexpected_score_counts,
        "v2_audit_missing_ids": absent_audit,
        "v2_deployment_runtime_missing_ids": missing_deployment_runtime,
        "v2_metrics_missing_ids": absent_metrics,
        "v2_top1_rmsd_invalid_ids": invalid_metrics,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_manifest.to_csv(output / "local174_manifest_exact.csv", index=False)
    selected_scores.to_csv(output / "local174_pose_scores_exact.csv", index=False)
    order.merge(audits, on="complex_name", how="left", validate="one_to_one").to_csv(
        output / "local174_v2_audit_exact.csv", index=False
    )
    order.merge(metrics, on="complex_name", how="left", validate="one_to_one").to_csv(
        output / "local174_v2_metrics_exact.csv", index=False
    )
    summary = {
        "cohort_n": len(cohort_ids),
        "manifest_source_n": len(manifest),
        "pose_score_source_complex_n": int(scores["complex_name"].nunique()),
        "v2_audit_source_files": [str(path.expanduser().resolve()) for path in args.v2_audit],
        "v2_metrics_source_files": [str(path.expanduser().resolve()) for path in args.v2_metrics],
        "selection": "exact complex_name match in authoritative cohort order",
        "runtime_definition": "selection_ready_seconds + Top-1 materialization only",
        "best_of_n_materialization_excluded": True,
        "fatal_checks": fatal,
        "complete": not any(bool(value) for value in fatal.values()),
    }
    (output / "local174_collection_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        raise SystemExit("FIGURE2_EXISTING_LOCAL174_INCOMPLETE")
    print(f"FIGURE2_EXISTING_LOCAL174_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
