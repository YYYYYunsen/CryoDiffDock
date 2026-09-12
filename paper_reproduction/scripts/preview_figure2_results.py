#!/usr/bin/env python3
"""Preview and audit a frozen Figure 2 comparison before final bundling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "succeeded", "ok"}
METHODS = ("ChemEM Stage 1", "CryoDiffDock-Local V2", "EMERALD")


def read_ids(path: Path, expected_n: int) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != expected_n or len(set(values)) != expected_n:
        raise ValueError(
            f"Frozen cohort must contain {expected_n} unique IDs, found {len(values)}"
        )
    return values


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator)


def normalize(table: pd.DataFrame, id_candidates: tuple[str, ...], label: str) -> pd.DataFrame:
    id_column = next((column for column in id_candidates if column in table.columns), None)
    if id_column is None:
        raise ValueError(f"{label} has no ID column; columns={sorted(table.columns)}")
    result = table.copy()
    result["complex_name"] = result[id_column].astype(str).str.strip().str.upper()
    duplicates = sorted(result.loc[result["complex_name"].duplicated(False), "complex_name"].unique())
    if duplicates:
        raise ValueError(f"{label} contains duplicate IDs: {duplicates[:20]}")
    return result


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def quantiles(values: pd.Series) -> tuple[float, float, float, float]:
    values = finite(values).dropna()
    if values.empty:
        return (float("nan"),) * 4
    return (
        float(values.median()),
        float(values.quantile(0.25)),
        float(values.quantile(0.75)),
        float(values.mean()),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--chemem", required=True, type=Path)
    parser.add_argument("--v2", required=True, type=Path)
    parser.add_argument("--emerald", required=True, type=Path)
    parser.add_argument("--v2-runtime", type=Path)
    parser.add_argument("--chemem-runtime", type=Path)
    parser.add_argument("--emerald-runtime", type=Path)
    parser.add_argument("--old-v2", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cohort-n", type=int, default=99)
    parser.add_argument(
        "--allow-v2-incomplete",
        action="store_true",
        help=(
            "Retain missing/technical-failure V2 cases as failures in the fixed "
            "denominator and exclude them from strict paired-complete analyses."
        ),
    )
    args = parser.parse_args()

    cohort_ids = read_ids(args.cohort.expanduser().resolve(), args.expected_cohort_n)
    cohort_n = len(cohort_ids)
    order = pd.DataFrame({"frozen_index": np.arange(cohort_n), "complex_name": cohort_ids})

    chemem = normalize(read_table(args.chemem), ("complex_name", "pdb_id"), "ChemEM")
    if "chemem_top1_rmsd_A" not in chemem or "chemem_complete" not in chemem:
        raise ValueError("ChemEM table requires chemem_complete and chemem_top1_rmsd_A")
    chemem = chemem[["complex_name", "chemem_complete", "chemem_top1_rmsd_A"]].copy()
    chemem["chemem_rmsd_A"] = finite(chemem["chemem_top1_rmsd_A"])
    chemem["chemem_scored"] = truthy(chemem["chemem_complete"]) & chemem["chemem_rmsd_A"].notna()

    v2 = normalize(read_table(args.v2), ("complex_name", "pdb_id"), "V2")
    if "top1_rmsd_A" not in v2:
        raise ValueError("V2 table requires top1_rmsd_A")
    v2 = v2[["complex_name", "top1_rmsd_A"]].copy()
    v2["v2_rmsd_A"] = finite(v2["top1_rmsd_A"])
    v2["v2_scored"] = v2["v2_rmsd_A"].notna()

    emerald = normalize(read_table(args.emerald), ("pdb_id", "complex_name"), "EMERALD")
    if "top1_rmsd_A" not in emerald or "status" not in emerald:
        raise ValueError("EMERALD table requires status and top1_rmsd_A")
    emerald = emerald[["complex_name", "status", "top1_rmsd_A", "end_to_end_wall_s"]].copy()
    emerald["emerald_rmsd_A"] = finite(emerald["top1_rmsd_A"])
    emerald["emerald_scored"] = truthy(emerald["status"]) & emerald["emerald_rmsd_A"].notna()
    emerald["emerald_runtime_min"] = finite(emerald["end_to_end_wall_s"]) / 60.0

    joined = order.merge(
        chemem[["complex_name", "chemem_scored", "chemem_rmsd_A"]],
        on="complex_name", how="left", validate="one_to_one",
    ).merge(
        v2[["complex_name", "v2_scored", "v2_rmsd_A"]],
        on="complex_name", how="left", validate="one_to_one",
    ).merge(
        emerald[["complex_name", "emerald_scored", "emerald_rmsd_A", "status", "emerald_runtime_min"]],
        on="complex_name", how="left", validate="one_to_one",
    )
    for column in ("chemem_scored", "v2_scored", "emerald_scored"):
        joined[column] = joined[column].fillna(False).astype(bool)
    joined["three_software_paired_complete"] = joined[
        ["chemem_scored", "v2_scored", "emerald_scored"]
    ].all(axis=1)

    if args.chemem_runtime:
        chemem_runtime = normalize(
            read_table(args.chemem_runtime), ("complex_name", "pdb_id"), "ChemEM runtime"
        )
        if "runtime_minutes" not in chemem_runtime:
            raise ValueError("ChemEM runtime table requires runtime_minutes")
        chemem_runtime["chemem_runtime_min"] = finite(chemem_runtime["runtime_minutes"])
        joined = joined.merge(
            chemem_runtime[["complex_name", "chemem_runtime_min"]],
            on="complex_name", how="left", validate="one_to_one",
        )

    if args.v2_runtime:
        v2_runtime = normalize(
            read_table(args.v2_runtime), ("complex_name", "pdb_id"), "V2 runtime"
        )
        required = {"runtime_seconds", "v2_expansion_runtime_source_column", "pose_score_rows"}
        missing = sorted(required - set(v2_runtime.columns))
        if missing:
            raise ValueError(f"V2 runtime table lacks columns: {missing}")
        v2_runtime["v2_runtime_min"] = finite(v2_runtime["runtime_seconds"]) / 60.0
        joined = joined.merge(
            v2_runtime[[
                "complex_name", "v2_runtime_min", "pose_score_rows",
                "v2_expansion_runtime_source_column",
            ]],
            on="complex_name", how="left", validate="one_to_one",
        )

    if args.emerald_runtime:
        emerald_runtime = normalize(
            read_table(args.emerald_runtime),
            ("complex_name", "pdb_id"),
            "EMERALD runtime",
        )
        if "runtime_minutes" in emerald_runtime:
            emerald_runtime["emerald_runtime_min_collected"] = finite(
                emerald_runtime["runtime_minutes"]
            )
        elif "runtime_seconds" in emerald_runtime:
            emerald_runtime["emerald_runtime_min_collected"] = (
                finite(emerald_runtime["runtime_seconds"]) / 60.0
            )
        else:
            raise ValueError(
                "EMERALD runtime table requires runtime_minutes or runtime_seconds"
            )
        joined = joined.merge(
            emerald_runtime[["complex_name", "emerald_runtime_min_collected"]],
            on="complex_name", how="left", validate="one_to_one",
        )
        joined["emerald_runtime_min"] = joined[
            "emerald_runtime_min_collected"
        ].combine_first(joined["emerald_runtime_min"])
        joined = joined.drop(columns=["emerald_runtime_min_collected"])

    checks: dict[str, object] = {
        "frozen_n": len(joined),
        "unique_frozen_n": int(joined["complex_name"].nunique()),
        "completed_counts": {
            "ChemEM Stage 1": int(joined["chemem_scored"].sum()),
            "CryoDiffDock-Local V2": int(joined["v2_scored"].sum()),
            "EMERALD": int(joined["emerald_scored"].sum()),
        },
        "strict_three_software_paired_n": int(joined["three_software_paired_complete"].sum()),
        "v2_incomplete_allowed": bool(args.allow_v2_incomplete),
    }

    rmsd_columns = ("chemem_rmsd_A", "v2_rmsd_A", "emerald_rmsd_A")
    invalid_rmsd = {
        column: int((finite(joined[column]).dropna() < 0).sum()) for column in rmsd_columns
    }
    checks["negative_rmsd_counts"] = invalid_rmsd

    if args.v2_runtime:
        pose_score_counts = finite(joined["pose_score_rows"])
        accepted_v2_runtime_sources = {
            "deployment_runtime_seconds",
            "selection_ready_plus_top1_materialization_seconds",
            "case_runtime_seconds_minus_all_condition_materialization_seconds",
            "legacy_run_selection_ready_seconds_amortized",
            "selection_ready_seconds",
            "not_attempted_due_upstream_technical_failure",
        }
        checks["v2_runtime_finite_n"] = int(joined["v2_runtime_min"].notna().sum())
        checks["v2_pose_score_80_n"] = int(pose_score_counts.eq(80).sum())
        checks["v2_pose_score_zero_n"] = int(pose_score_counts.eq(0).sum())
        checks["v2_pose_score_unexpected_n"] = int(
            (~pose_score_counts.isin((0, 80))).sum()
        )
        checks["v2_deployment_runtime_source_n"] = int(
            joined["v2_expansion_runtime_source_column"]
            .isin(accepted_v2_runtime_sources)
            .sum()
        )
    if args.emerald_runtime:
        checks["emerald_runtime_finite_n"] = int(
            joined["emerald_runtime_min"].notna().sum()
        )

    if args.old_v2:
        old = normalize(read_table(args.old_v2), ("complex_name", "pdb_id"), "old V2")
        if "top1_rmsd_A" not in old:
            raise ValueError("Old V2 table requires top1_rmsd_A")
        old["old_v2_rmsd_A"] = finite(old["top1_rmsd_A"])
        parity = joined[["complex_name", "v2_rmsd_A"]].merge(
            old[["complex_name", "old_v2_rmsd_A"]],
            on="complex_name", how="inner", validate="one_to_one",
        ).dropna()
        differences = (parity["v2_rmsd_A"] - parity["old_v2_rmsd_A"]).abs()
        checks["old_new_v2_parity"] = {
            "common_scored_n": len(parity),
            "max_absolute_rmsd_difference_A": float(differences.max()) if len(differences) else None,
            "different_over_1e_6_n": int((differences > 1e-6).sum()),
        }

    summary_rows: list[dict[str, object]] = []
    analysis_sets = (
        (f"frozen_{cohort_n}_missing_as_failure", joined),
        (
            "three_software_paired_complete",
            joined[joined["three_software_paired_complete"]],
        ),
    )
    method_specs = (
        ("ChemEM Stage 1", "chemem_scored", "chemem_rmsd_A"),
        ("CryoDiffDock-Local V2", "v2_scored", "v2_rmsd_A"),
        ("EMERALD", "emerald_scored", "emerald_rmsd_A"),
    )
    for analysis_set, frame in analysis_sets:
        for method, complete_column, rmsd_column in method_specs:
            values = finite(frame.loc[frame[complete_column], rmsd_column]).dropna()
            for cutoff in (2.0, 5.0):
                summary_rows.append({
                    "analysis_set": analysis_set,
                    "method": method,
                    "cutoff_A": cutoff,
                    "success_count": int((values < cutoff).sum()),
                    "denominator": len(frame),
                    "success_pct": 100.0 * int((values < cutoff).sum()) / len(frame),
                    "scored_n": len(values),
                    "median_rmsd_A_among_scored": float(values.median()),
                })
    summary = pd.DataFrame(summary_rows)

    runtime_rows: list[dict[str, object]] = []
    for method, column, success_column in (
        ("ChemEM Stage 1", "chemem_runtime_min", "chemem_scored"),
        ("CryoDiffDock-Local V2", "v2_runtime_min", "v2_scored"),
        ("EMERALD", "emerald_runtime_min", "emerald_scored"),
    ):
        if column not in joined:
            continue
        observed = finite(joined[column]).dropna()
        successful = finite(joined.loc[joined[success_column], column]).dropna()
        median, q1, q3, mean = quantiles(observed)
        runtime_rows.append({
            "method": method,
            "frozen_n": cohort_n,
            "observed_attempt_runtime_n": len(observed),
            "missing_runtime_n": cohort_n - len(observed),
            "successful_output_with_runtime_n": len(successful),
            "failed_or_incomplete_with_runtime_n": len(observed) - len(successful),
            "median_runtime_min": median,
            "q1_runtime_min": q1,
            "q3_runtime_min": q3,
            "mean_runtime_min": mean,
        })
    runtime_summary = pd.DataFrame(runtime_rows)

    fatal = []
    if (
        checks["completed_counts"]["CryoDiffDock-Local V2"] != cohort_n
        and not args.allow_v2_incomplete
    ):
        fatal.append(f"V2 does not have {cohort_n} scored Top-1 results")
    if any(invalid_rmsd.values()):
        fatal.append("negative RMSD detected")
    if args.v2_runtime:
        if checks["v2_runtime_finite_n"] != cohort_n:
            fatal.append(f"V2 runtime is not complete for all {cohort_n} cases")
        if checks["v2_pose_score_unexpected_n"]:
            fatal.append("V2 pose-score count is neither 0 nor 80 for one or more cases")
        if checks["v2_pose_score_zero_n"] and not args.allow_v2_incomplete:
            fatal.append("one or more frozen cases have no inference pose scores")
        if checks["v2_deployment_runtime_source_n"] != cohort_n:
            fatal.append("V2 expansion runtime did not use deployment_runtime_seconds")
    parity_check = checks.get("old_new_v2_parity")
    if parity_check and parity_check["different_over_1e_6_n"]:
        fatal.append("new Top-1-only V2 RMSD differs from the original V2 ranking")
    checks["fatal_checks"] = fatal
    checks["ready_for_bundle"] = not fatal

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "preview_per_complex.csv", index=False)
    summary.to_csv(output / "preview_accuracy_summary.csv", index=False)
    runtime_summary.to_csv(output / "preview_runtime_summary.csv", index=False)
    (output / "preview_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")

    print(f"\nFROZEN-{cohort_n} COMPLETENESS")
    print(pd.DataFrame([
        {"method": method, "scored": count, "missing_or_failed": cohort_n - count}
        for method, count in checks["completed_counts"].items()
    ]).to_string(index=False))
    print(f"\nStrict three-software paired N: {checks['strict_three_software_paired_n']}")
    print("\nACCURACY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if not runtime_summary.empty:
        print("\nRUNTIME (observed attempts; minutes)")
        print(runtime_summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if parity_check:
        print("\nOLD/NEW V2 TOP-1 PARITY")
        print(json.dumps(parity_check, indent=2, sort_keys=True))
    print("\nLONGEST V2 CASES")
    if "v2_runtime_min" in joined:
        print(
            joined.nlargest(10, "v2_runtime_min")[
                ["complex_name", "v2_runtime_min", "v2_rmsd_A"]
            ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
    else:
        print("V2 runtime was not supplied")
    print(f"\nReady for bundle: {not fatal}")
    if fatal:
        for message in fatal:
            print(f"ERROR: {message}")
    print(f"Preview files: {output}")
    return 2 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
