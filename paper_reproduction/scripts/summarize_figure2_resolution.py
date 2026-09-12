#!/usr/bin/env python3
"""Resolution-stratified current Figure 2 accuracy and paired tests."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


LABELS = ("higher_resolution", "middle_resolution", "lower_resolution")
METHODS = (
    ("ChemEM Stage 1", "chemem_scored", "chemem_rmsd_A"),
    ("CryoDiffDock-Local V2", "v2_scored", "v2_rmsd_A"),
    ("EMERALD", "emerald_scored", "emerald_rmsd_A"),
)
CUTOFFS = (2.0, 5.0)
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "succeeded", "ok"}


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def normalize_ids(table: pd.DataFrame, label: str) -> pd.DataFrame:
    column = next(
        (candidate for candidate in ("complex_name", "pdb_id", "complex_id", "pdb_id_upper")
         if candidate in table.columns),
        None,
    )
    if column is None:
        raise ValueError(f"{label} has no recognized ID column")
    result = table.copy()
    result["complex_name"] = result[column].astype(str).str.strip().str.upper()
    if result["complex_name"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    return result


def stable_resolution_tertiles(metadata: pd.DataFrame) -> pd.DataFrame:
    ordered = metadata.sort_values(
        ["resolution_A", "complex_name"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(ordered)
    ordered["resolution_rank"] = np.arange(n)
    ordered["resolution_stratum"] = [
        LABELS[min(2, (rank * 3) // n)] for rank in range(n)
    ]
    return ordered


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)


def holm(values: list[float]) -> list[float]:
    pvalues = np.asarray(values, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def add_family_holm(
    rows: list[dict[str, object]], pvalue_column: str, family_columns: list[str]
) -> None:
    table = pd.DataFrame(rows)
    adjusted = pd.Series(index=table.index, dtype=float)
    for _, indices in table.groupby(family_columns, sort=False).groups.items():
        index_list = list(indices)
        values = table.loc[index_list, pvalue_column].astype(float).tolist()
        adjusted.loc[index_list] = holm(values)
    for index, value in adjusted.items():
        rows[int(index)][f"{pvalue_column}_holm_family"] = float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-per-complex", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    args = parser.parse_args()

    preview = normalize_ids(pd.read_csv(args.preview_per_complex), "preview")
    manifest = normalize_ids(pd.read_csv(args.manifest), "manifest")
    if len(preview) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} preview rows, found {len(preview)}")
    if "resolution_A" not in manifest:
        raise ValueError("Manifest lacks resolution_A")
    manifest["resolution_A"] = pd.to_numeric(manifest["resolution_A"], errors="coerce")
    metadata = manifest[manifest["complex_name"].isin(preview["complex_name"])][
        ["complex_name", "resolution_A"]
    ].copy()
    if len(metadata) != args.expected_n or metadata["resolution_A"].isna().any():
        raise ValueError("Resolution metadata does not cover the complete cohort")
    metadata = stable_resolution_tertiles(metadata)
    cases = preview.merge(metadata, on="complex_name", how="left", validate="one_to_one")
    for _, scored_column, rmsd_column in METHODS:
        cases[scored_column] = truthy(cases[scored_column])
        cases[rmsd_column] = pd.to_numeric(cases[rmsd_column], errors="coerce")
        cases[scored_column] &= np.isfinite(cases[rmsd_column])
    cases["three_software_paired_complete"] = cases[
        [spec[1] for spec in METHODS]
    ].all(axis=1)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output / "resolution_per_complex.csv", index=False)

    bin_rows = []
    for label in LABELS:
        group = cases[cases["resolution_stratum"].eq(label)]
        bin_rows.append({
            "resolution_stratum": label,
            "cohort_n": len(group),
            "paired_n": int(group["three_software_paired_complete"].sum()),
            "resolution_min_A": float(group["resolution_A"].min()),
            "resolution_max_A": float(group["resolution_A"].max()),
            "resolution_median_A": float(group["resolution_A"].median()),
        })
    pd.DataFrame(bin_rows).to_csv(output / "resolution_bin_audit.csv", index=False)

    accuracy_rows = []
    for label in LABELS:
        fixed = cases[cases["resolution_stratum"].eq(label)]
        paired = fixed[fixed["three_software_paired_complete"]]
        for analysis_set, frame, missing_as_failure in (
            ("fixed_stratum_missing_as_failure", fixed, True),
            ("three_software_paired_complete", paired, False),
        ):
            for method, scored_column, rmsd_column in METHODS:
                values = frame.loc[frame[scored_column], rmsd_column].to_numpy(float)
                for cutoff in CUTOFFS:
                    success = int((values < cutoff).sum())
                    accuracy_rows.append({
                        "resolution_stratum": label,
                        "analysis_set": analysis_set,
                        "method": method,
                        "cutoff_A": cutoff,
                        "success_count": success,
                        "denominator": len(frame),
                        "success_pct": 100.0 * success / len(frame) if len(frame) else np.nan,
                        "scored_n": len(values),
                        "missing_or_failed_n": len(frame) - len(values),
                        "median_rmsd_A_among_scored": float(np.median(values)) if len(values) else np.nan,
                        "missing_counted_as_failure": missing_as_failure,
                    })
    pd.DataFrame(accuracy_rows).to_csv(output / "resolution_accuracy.csv", index=False)

    success_rows: list[dict[str, object]] = []
    rmsd_rows: list[dict[str, object]] = []
    for label in LABELS:
        paired = cases[
            cases["resolution_stratum"].eq(label)
            & cases["three_software_paired_complete"]
        ]
        for left, right in itertools.combinations(METHODS, 2):
            left_method, _, left_column = left
            right_method, _, right_column = right
            left_rmsd = paired[left_column].to_numpy(float)
            right_rmsd = paired[right_column].to_numpy(float)
            try:
                rmsd_pvalue = float(wilcoxon(right_rmsd, left_rmsd).pvalue)
            except ValueError:
                rmsd_pvalue = 1.0
            rmsd_rows.append({
                "resolution_stratum": label,
                "left_method": left_method,
                "right_method": right_method,
                "paired_n": len(paired),
                "median_delta_right_minus_left_rmsd_A": float(
                    np.median(right_rmsd - left_rmsd)
                ) if len(paired) else np.nan,
                "wilcoxon_pvalue": rmsd_pvalue,
            })
            for cutoff in CUTOFFS:
                left_success = left_rmsd < cutoff
                right_success = right_rmsd < cutoff
                left_only = int((left_success & ~right_success).sum())
                right_only = int((~left_success & right_success).sum())
                success_rows.append({
                    "resolution_stratum": label,
                    "cutoff_A": cutoff,
                    "left_method": left_method,
                    "right_method": right_method,
                    "paired_n": len(paired),
                    "left_success_count": int(left_success.sum()),
                    "right_success_count": int(right_success.sum()),
                    "delta_right_minus_left_percentage_points": 100.0 * (
                        float(right_success.mean()) - float(left_success.mean())
                    ) if len(paired) else np.nan,
                    "left_only_successes": left_only,
                    "right_only_successes": right_only,
                    "exact_mcnemar_pvalue": exact_mcnemar(left_only, right_only),
                })

    # Each endpoint family contains all three method pairs across all three
    # resolution strata (nine tests), giving a conservative subgroup analysis.
    add_family_holm(success_rows, "exact_mcnemar_pvalue", ["cutoff_A"])
    add_family_holm(rmsd_rows, "wilcoxon_pvalue", ["resolution_stratum"])
    pd.DataFrame(success_rows).to_csv(output / "resolution_paired_success_tests.csv", index=False)
    pd.DataFrame(rmsd_rows).to_csv(output / "resolution_paired_rmsd_tests.csv", index=False)

    audit = {
        "cohort_n": len(cases),
        "stratification": (
            "deterministic rank tertiles of resolution_A over authoritative-174; "
            "ascending resolution with PDB ID tie-break"
        ),
        "labels": list(LABELS),
        "bin_counts": {row["resolution_stratum"]: row["cohort_n"] for row in bin_rows},
        "primary_subgroup_analysis": "strict three-software paired complete",
        "fixed_stratum_sensitivity": "technical failures retained in each 58-case denominator",
        "success_test": "exact McNemar",
        "success_multiplicity": "Holm across 3 method pairs x 3 strata within each cutoff",
        "rmsd_test": "paired Wilcoxon; Holm across 3 method pairs within each stratum",
    }
    (output / "resolution_analysis_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print("\nRESOLUTION BINS")
    print(pd.DataFrame(bin_rows).to_string(index=False))
    print("\nPAIRED-COMPLETE ACCURACY")
    accuracy = pd.DataFrame(accuracy_rows)
    print(accuracy[accuracy["analysis_set"].eq("three_software_paired_complete")].to_string(index=False))
    print("\nPAIRED SUCCESS TESTS")
    print(pd.DataFrame(success_rows).to_string(index=False))
    print(f"\nFIGURE2_RESOLUTION_SUMMARY_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
