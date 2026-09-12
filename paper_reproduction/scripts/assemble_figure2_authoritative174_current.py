#!/usr/bin/env python3
"""Assemble current, possibly incomplete Figure 2 authoritative-174 inputs.

This script performs no docking.  It prepares three derived inputs used by the
existing Figure 2 preview/statistics programs:

* a cohort table compatible with the ChemEM evaluator;
* a historical ChemEM table plus a recovery-aware current work table;
* a recovery-aware EMERALD table with exactly one row per authoritative ID.

Completed recovery results take precedence over their original failed rows.
Missing and failed rows are retained so fixed-denominator reporting can count
them as failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "complex_id", "pdb_id_upper")
TRUE_VALUES = {
    "1", "true", "yes", "y", "complete", "completed", "success", "succeeded", "ok"
}


def read_ids(path: Path, expected_n: int) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != expected_n or len(set(values)) != expected_n:
        raise ValueError(
            f"Expected {expected_n} unique IDs in {path}, found {len(values)}"
        )
    return values


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator, dtype=str)


def normalize(table: pd.DataFrame, label: str) -> pd.DataFrame:
    id_column = next((column for column in ID_COLUMNS if column in table.columns), None)
    if id_column is None:
        raise ValueError(f"{label} has no recognized ID column: {list(table.columns)}")
    result = table.copy()
    result["complex_name"] = result[id_column].astype(str).str.strip().str.upper()
    result = result[result["complex_name"].ne("") & result["complex_name"].ne("NAN")]
    return result


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def build_preview_cohort(
    ids: list[str], manifest_path: Path, output_path: Path
) -> pd.DataFrame:
    manifest = normalize(read_table(manifest_path), "authoritative manifest")
    if manifest["complex_name"].duplicated().any():
        raise ValueError("Authoritative manifest contains duplicate IDs")
    selected = pd.DataFrame({"complex_name": ids}).merge(
        manifest, on="complex_name", how="left", validate="one_to_one", sort=False,
        suffixes=("", "_manifest"),
    )
    if "ligand" not in selected.columns:
        raise ValueError("Authoritative manifest lacks ligand")
    if selected["ligand"].isna().any():
        missing = selected.loc[selected["ligand"].isna(), "complex_name"].tolist()
        raise ValueError(f"Manifest is missing authoritative IDs: {missing}")
    if "resolution_A" not in selected.columns:
        selected["resolution_A"] = np.nan
    if "resolution_stratum" not in selected.columns:
        selected["resolution_stratum"] = "not_stratified"
    else:
        selected["resolution_stratum"] = selected["resolution_stratum"].fillna(
            "not_stratified"
        )
    if "selection_phase" not in selected.columns:
        selected["selection_phase"] = "authoritative174"
    else:
        selected["selection_phase"] = selected["selection_phase"].fillna(
            "authoritative174"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return selected


def build_historical_chemem(input_path: Path, output_path: Path) -> pd.DataFrame:
    source = normalize(read_table(input_path), "historical ChemEM benchmark")
    if source["complex_name"].duplicated().any():
        raise ValueError("Historical ChemEM benchmark contains duplicate IDs")
    top1_column = next(
        (column for column in ("chemem_top1_rmsd_A", "top1_rmsd_A") if column in source),
        None,
    )
    oracle_column = next(
        (column for column in ("chemem_oracle_rmsd_A", "best_rmsd_A") if column in source),
        None,
    )
    if top1_column is None or oracle_column is None:
        raise ValueError("Historical ChemEM benchmark lacks Top-1/oracle RMSD columns")
    top1 = finite(source[top1_column])
    oracle = finite(source[oracle_column])
    if "chemem_complete" in source:
        complete = truthy(source["chemem_complete"]) & top1.notna() & oracle.notna()
    elif "status" in source:
        complete = truthy(source["status"]) & top1.notna() & oracle.notna()
    else:
        complete = top1.notna() & oracle.notna()
    result = pd.DataFrame({
        "complex_name": source["complex_name"],
        "status": np.where(complete, "complete", "pending_or_failed"),
        "top1_rmsd_A": top1.where(complete),
        "best_rmsd_A": oracle.where(complete),
    })
    aliases = {
        "num_ranked": ("chemem_num_ranked", "num_ranked"),
        "num_evaluated": ("chemem_num_evaluated", "num_evaluated"),
        "num_pose_failures": ("chemem_num_pose_failures", "num_pose_failures"),
        "top1_iteration": ("chemem_top1_iteration", "top1_iteration"),
        "top1_score": ("chemem_top1_score", "top1_score"),
        "best_rank": ("chemem_oracle_rank", "best_rank"),
        "best_iteration": ("chemem_oracle_iteration", "best_iteration"),
    }
    for output_column, candidates in aliases.items():
        input_column = next((column for column in candidates if column in source), None)
        result[output_column] = source[input_column] if input_column else np.nan
    result.to_csv(output_path, index=False)
    return result


def build_current_chemem_cases(
    primary_path: Path, recovery_path: Path | None, output_path: Path
) -> pd.DataFrame:
    primary = normalize(read_table(primary_path), "current ChemEM cases")
    primary["source_priority"] = 0
    primary["source_cases_file"] = str(primary_path)
    tables = [primary]
    if recovery_path is not None and recovery_path.is_file():
        recovery = normalize(read_table(recovery_path), "ChemEM recovery cases")
        recovery["source_priority"] = 1
        recovery["source_cases_file"] = str(recovery_path)
        tables.append(recovery)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    if "stage1_output" not in combined or "case_group" not in combined:
        raise ValueError("ChemEM cases require stage1_output and case_group")
    combined = (
        combined.sort_values("source_priority", kind="stable")
        .drop_duplicates("complex_name", keep="last")
        .sort_values(["source_priority", "complex_name"], kind="stable")
    )
    combined.to_csv(output_path, index=False)
    return combined


def emerald_summary_files(
    old_summary: Path, expansion_summary: Path, recovery_roots: list[Path]
) -> list[tuple[int, Path]]:
    files = [(0, old_summary), (1, expansion_summary)]
    for root_index, recovery_root in enumerate(recovery_roots, start=2):
        if recovery_root.is_dir():
            for path in sorted(recovery_root.rglob("emerald_paper_summary.tsv")):
                files.append((root_index, path.resolve()))
    return files


def build_emerald(
    ids: list[str], old_summary: Path, expansion_summary: Path,
    recovery_roots: list[Path], timeout_ids: set[str], timeout_cap_s: float,
    output_path: Path,
) -> tuple[pd.DataFrame, list[str]]:
    tables = []
    source_files = emerald_summary_files(
        old_summary, expansion_summary, recovery_roots
    )
    for priority, path in source_files:
        table = normalize(read_table(path), f"EMERALD summary {path}")
        if "status" not in table and "state" in table:
            table["status"] = table["state"]
        if "status" not in table or "top1_rmsd_A" not in table:
            raise ValueError(f"EMERALD summary lacks status/top1_rmsd_A: {path}")
        table["source_priority"] = priority
        table["source_summary"] = str(path)
        table["top1_numeric"] = finite(table["top1_rmsd_A"])
        table["scored"] = truthy(table["status"]) & table["top1_numeric"].notna()
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined = combined[combined["complex_name"].isin(ids)].copy()
    # A completed, scored row always beats a failed row. Among equal states, a
    # later recovery source takes precedence over the original attempt.
    selected = (
        combined.sort_values(
            ["complex_name", "scored", "source_priority"], kind="stable"
        )
        .drop_duplicates("complex_name", keep="last")
    )
    output = pd.DataFrame({"complex_name": ids}).merge(
        selected.drop(columns=["top1_numeric", "scored"], errors="ignore"),
        on="complex_name", how="left", validate="one_to_one", sort=False,
    )
    output["pdb_id"] = output["complex_name"]
    output["status"] = output["status"].fillna("missing")
    if "end_to_end_wall_s" not in output:
        output["end_to_end_wall_s"] = np.nan
    output["timeout_cap_assigned"] = False
    currently_scored = truthy(output["status"]) & finite(output["top1_rmsd_A"]).notna()
    timeout_mask = output["complex_name"].isin(timeout_ids) & ~currently_scored
    output.loc[timeout_mask, "status"] = "timeout"
    output.loc[timeout_mask, "end_to_end_wall_s"] = timeout_cap_s
    output.loc[timeout_mask, "timeout_cap_assigned"] = True
    output.to_csv(output_path, sep="\t", index=False)
    return output, [str(path) for _, path in source_files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-ids", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-chemem-benchmark", required=True, type=Path)
    parser.add_argument("--new-chemem-cases", required=True, type=Path)
    parser.add_argument("--chemem-recovery-cases", type=Path)
    parser.add_argument("--old-emerald-summary", required=True, type=Path)
    parser.add_argument("--new-emerald-summary", required=True, type=Path)
    parser.add_argument("--emerald-recovery-root", action="append", default=[], type=Path)
    parser.add_argument("--emerald-timeout-id", action="append", default=[])
    parser.add_argument("--emerald-timeout-cap-s", type=float, default=43200.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    args = parser.parse_args()

    cohort_ids_path = require_file(args.cohort_ids, "cohort IDs")
    manifest_path = require_file(args.manifest, "authoritative manifest")
    old_chemem_path = require_file(args.old_chemem_benchmark, "historical ChemEM benchmark")
    new_chemem_path = require_file(args.new_chemem_cases, "new ChemEM cases")
    old_emerald_path = require_file(args.old_emerald_summary, "historical EMERALD summary")
    new_emerald_path = require_file(args.new_emerald_summary, "new EMERALD summary")
    recovery_cases = (
        args.chemem_recovery_cases.expanduser().resolve()
        if args.chemem_recovery_cases else None
    )
    emerald_recoveries = [
        path.expanduser().resolve() for path in args.emerald_recovery_root
    ]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ids = read_ids(cohort_ids_path, args.expected_n)
    timeout_ids = {
        value.strip().upper() for value in args.emerald_timeout_id if value.strip()
    }
    unknown_timeout_ids = sorted(timeout_ids - set(ids))
    if unknown_timeout_ids:
        raise ValueError(f"EMERALD timeout IDs are outside the cohort: {unknown_timeout_ids}")

    cohort = build_preview_cohort(ids, manifest_path, output / "preview_cohort.csv")
    historical = build_historical_chemem(
        old_chemem_path, output / "chemem_historical_for_evaluator.csv"
    )
    current = build_current_chemem_cases(
        new_chemem_path, recovery_cases, output / "chemem_current_cases.csv"
    )
    emerald, emerald_sources = build_emerald(
        ids, old_emerald_path, new_emerald_path, emerald_recoveries,
        timeout_ids, args.emerald_timeout_cap_s,
        output / "emerald_current_174.tsv",
    )
    emerald_scored = truthy(emerald["status"]) & finite(emerald["top1_rmsd_A"]).notna()
    audit = {
        "cohort_n": len(cohort),
        "historical_chemem_rows": len(historical),
        "current_chemem_case_rows": len(current),
        "chemem_recovery_cases_used": bool(recovery_cases and recovery_cases.is_file()),
        "emerald_scored_n": int(emerald_scored.sum()),
        "emerald_missing_or_failed_n": int(len(emerald) - emerald_scored.sum()),
        "emerald_timeout_ids": sorted(timeout_ids),
        "emerald_timeout_cap_s": args.emerald_timeout_cap_s,
        "emerald_source_files": emerald_sources,
        "fixed_denominator_policy": "all 174 cases; missing/failed count as failures",
        "paired_policy": "strict intersection with finite Top-1 RMSD for all three methods",
        "docking_or_inference_run": False,
    }
    (output / "assembly_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"FIGURE2_AUTHORITATIVE174_CURRENT_ASSEMBLED: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
