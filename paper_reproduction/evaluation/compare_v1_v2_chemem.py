#!/usr/bin/env python3
"""Compare ChemEM Stage-1, frozen Local V1, and Local V2 on one intersection.

The primary comparison is operational Top-1:

* ChemEM: ``chemem_top1_rmsd_A`` from ``per_complex_benchmark.csv``;
* V1: ``top1_rmsd_A`` from the frozen-S0 metrics;
* V2: ``top1_rmsd_A`` from the frozen V2-selected metrics.

An optional, explicitly separate oracle analysis compares the minimum RMSD in
each method's evaluated candidate pool.  Oracle results are coverage ceilings,
not deployment accuracy, and pool sizes need not be compute matched.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex", "name")
CUTOFFS_A = (1.0, 2.0, 5.0)
PREFERRED_RESOLUTION_STRATA = (
    "higher_resolution",
    "middle_resolution",
    "lower_resolution",
)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def load_unique(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    id_column = next((column for column in ID_COLUMNS if column in table), None)
    if id_column is None:
        raise ValueError(f"{label} has no complex ID column: {path}")
    table["complex_name"] = table[id_column].map(normalize_id)
    duplicates = sorted(
        table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {duplicates[:20]}")
    return table


def load_metric(
    path: Path, source_column: str, output_column: str, label: str
) -> pd.DataFrame:
    table = load_unique(path, label)
    if source_column not in table:
        raise ValueError(f"{label} lacks {source_column}: {path}")
    table[output_column] = pd.to_numeric(table[source_column], errors="coerce")
    return table[["complex_name", output_column]]


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def wilson_interval(successes: int, denominator: int) -> tuple[float, float]:
    if denominator <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / denominator
    scale = 1.0 + z * z / denominator
    center = (proportion + z * z / (2.0 * denominator)) / scale
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / denominator
        + z * z / (4.0 * denominator * denominator)
    ) / scale
    return 100.0 * (center - half), 100.0 * (center + half)


def exact_mcnemar(left_only: int, right_only: int) -> float:
    from scipy.stats import binomtest

    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)


def safe_wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    try:
        return float(wilcoxon(right, left).pvalue)
    except ValueError:
        return 1.0


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def cochran_q(binary: np.ndarray) -> tuple[float, float]:
    """Cochran's Q for matched binary outcomes (rows=cases, columns=methods)."""
    from scipy.stats import chi2

    binary = np.asarray(binary, dtype=float)
    method_totals = binary.sum(axis=0)
    case_totals = binary.sum(axis=1)
    k = binary.shape[1]
    denominator = k * case_totals.sum() - np.square(case_totals).sum()
    if denominator <= 0:
        return 0.0, 1.0
    numerator = (k - 1) * (
        k * np.square(method_totals).sum() - np.square(method_totals.sum())
    )
    statistic = float(numerator / denominator)
    return statistic, float(chi2.sf(statistic, k - 1))


def method_summary(
    frame: pd.DataFrame, methods: list[tuple[str, str]], endpoint: str
) -> pd.DataFrame:
    rows = []
    for method, column in methods:
        values = frame[column].to_numpy(float)
        row: dict[str, object] = {
            "endpoint": endpoint,
            "method": method,
            "num_complexes": len(values),
            "median_rmsd_A": float(np.median(values)),
            "mean_rmsd_A": float(np.mean(values)),
        }
        for cutoff in CUTOFFS_A:
            successes = int((values < cutoff).sum())
            low, high = wilson_interval(successes, len(values))
            tag = f"lt{cutoff:g}A"
            row[f"{tag}_count"] = successes
            row[f"{tag}_pct"] = 100.0 * successes / len(values)
            row[f"{tag}_ci95_low_pct"] = low
            row[f"{tag}_ci95_high_pct"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(
    frame: pd.DataFrame, methods: list[tuple[str, str]], endpoint: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    arrays = {method: frame[column].to_numpy(float) for method, column in methods}
    pairs = list(combinations(methods, 2))
    rows = []
    for (left_method, _), (right_method, _) in pairs:
        left = arrays[left_method]
        right = arrays[right_method]
        rmsd_pvalue = safe_wilcoxon(left, right)
        for cutoff in CUTOFFS_A:
            left_success = left < cutoff
            right_success = right < cutoff
            left_only = int((left_success & ~right_success).sum())
            right_only = int((~left_success & right_success).sum())
            rows.append({
                "endpoint": endpoint,
                "left_method": left_method,
                "right_method": right_method,
                "num_paired": len(frame),
                "cutoff_A": cutoff,
                "median_delta_right_minus_left_A": float(np.median(right - left)),
                "mean_delta_right_minus_left_A": float(np.mean(right - left)),
                "wilcoxon_pvalue": rmsd_pvalue,
                "left_success_count": int(left_success.sum()),
                "right_success_count": int(right_success.sum()),
                "delta_right_minus_left_percentage_points": float(
                    100.0 * (right_success.mean() - left_success.mean())
                ),
                "left_only_successes": left_only,
                "right_only_successes": right_only,
                "mcnemar_exact_pvalue": exact_mcnemar(left_only, right_only),
            })
    result = pd.DataFrame(rows)
    # Correct the three pairwise tests separately for continuous RMSD and for
    # each binary threshold.
    unique_pairs = result.drop_duplicates(["left_method", "right_method"])
    rmsd_adjusted = holm_adjust(unique_pairs["wilcoxon_pvalue"].tolist())
    rmsd_map = {
        (row.left_method, row.right_method): adjusted
        for row, adjusted in zip(unique_pairs.itertuples(), rmsd_adjusted)
    }
    result["wilcoxon_holm_pvalue"] = [
        rmsd_map[(left, right)]
        for left, right in zip(result["left_method"], result["right_method"])
    ]
    result["mcnemar_holm_pvalue"] = float("nan")
    for cutoff in CUTOFFS_A:
        mask = result["cutoff_A"].eq(cutoff)
        result.loc[mask, "mcnemar_holm_pvalue"] = holm_adjust(
            result.loc[mask, "mcnemar_exact_pvalue"].tolist()
        )

    values = [arrays[method] for method, _ in methods]
    try:
        friedman = friedmanchisquare(*values)
        friedman_statistic = float(friedman.statistic)
        friedman_pvalue = float(friedman.pvalue)
    except ValueError:
        friedman_statistic, friedman_pvalue = 0.0, 1.0
    omnibus: dict[str, object] = {
        "endpoint": endpoint,
        "num_paired": len(frame),
        "friedman_statistic": friedman_statistic,
        "friedman_pvalue": friedman_pvalue,
        "cochran_q": {},
    }
    for cutoff in CUTOFFS_A:
        binary = np.column_stack([values_array < cutoff for values_array in values])
        statistic, pvalue = cochran_q(binary)
        omnibus["cochran_q"][f"lt{cutoff:g}A"] = {
            "statistic": statistic,
            "pvalue": pvalue,
        }
    return result, omnibus


def resolution_stratified_analysis(
    frame: pd.DataFrame,
    methods: list[tuple[str, str]],
    endpoint: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if "resolution_stratum" not in frame:
        return pd.DataFrame(), pd.DataFrame(), {}
    available = frame["resolution_stratum"].dropna().astype(str).unique().tolist()
    ordered = [value for value in PREFERRED_RESOLUTION_STRATA if value in available]
    ordered += sorted(set(available) - set(ordered))
    summary_tables = []
    paired_tables = []
    omnibus: dict[str, object] = {}
    for stratum in ordered:
        subset = frame.loc[
            frame["resolution_stratum"].astype(str).eq(stratum)
        ].copy()
        method_table = method_summary(subset, methods, endpoint)
        method_table.insert(0, "resolution_stratum", stratum)
        method_table.insert(1, "stratum_num_complexes", len(subset))
        if "resolution_A" in subset:
            resolution = pd.to_numeric(subset["resolution_A"], errors="coerce")
            method_table.insert(2, "median_resolution_A", float(resolution.median()))
            method_table.insert(3, "min_resolution_A", float(resolution.min()))
            method_table.insert(4, "max_resolution_A", float(resolution.max()))
        summary_tables.append(method_table)

        pairwise, tests = paired_summary(subset, methods, endpoint)
        pairwise.insert(0, "resolution_stratum", stratum)
        paired_tables.append(pairwise)
        omnibus[stratum] = tests
    return (
        pd.concat(summary_tables, ignore_index=True),
        pd.concat(paired_tables, ignore_index=True),
        omnibus,
    )


def complete_intersection(
    chemem: pd.DataFrame,
    metric_tables: list[pd.DataFrame],
    required_columns: list[str],
) -> pd.DataFrame:
    merged = chemem.copy()
    for table in metric_tables:
        merged = merged.merge(table, on="complex_name", how="inner", validate="one_to_one")
    finite = np.ones(len(merged), dtype=bool)
    for column in required_columns:
        finite &= np.isfinite(pd.to_numeric(merged[column], errors="coerce"))
    return merged.loc[finite].sort_values("complex_name").reset_index(drop=True)


def finite_ids(table: pd.DataFrame, column: str) -> set[str]:
    values = pd.to_numeric(table[column], errors="coerce")
    return set(table.loc[np.isfinite(values), "complex_name"].astype(str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemem-benchmark", required=True, type=Path)
    parser.add_argument("--v1-selected", required=True, type=Path)
    parser.add_argument("--v2-selected", required=True, type=Path)
    parser.add_argument("--v1-oracle", type=Path)
    parser.add_argument("--v2-oracle", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int)
    args = parser.parse_args()
    if (args.v1_oracle is None) != (args.v2_oracle is None):
        raise ValueError("Provide both --v1-oracle and --v2-oracle, or neither")

    chemem_path = args.chemem_benchmark.expanduser().resolve()
    chemem = load_unique(chemem_path, "ChemEM benchmark")
    required_chemem = {"chemem_top1_rmsd_A", "chemem_oracle_rmsd_A"}
    missing = sorted(required_chemem - set(chemem.columns))
    if missing:
        raise ValueError(f"ChemEM benchmark lacks columns: {missing}")
    for column in required_chemem:
        chemem[column] = pd.to_numeric(chemem[column], errors="coerce")
    if "chemem_complete" in chemem:
        chemem = chemem.loc[truthy(chemem["chemem_complete"])]
    metadata_columns = [
        column for column in ("resolution_A", "resolution_stratum", "selection_phase")
        if column in chemem
    ]
    chemem = chemem[["complex_name", *sorted(required_chemem), *metadata_columns]]
    if "resolution_A" in chemem:
        chemem["resolution_A"] = pd.to_numeric(
            chemem["resolution_A"], errors="coerce"
        )

    v1_selected = load_metric(
        args.v1_selected.expanduser().resolve(),
        "top1_rmsd_A", "v1_top1_rmsd_A", "V1 selected metrics",
    )
    v2_selected = load_metric(
        args.v2_selected.expanduser().resolve(),
        "top1_rmsd_A", "v2_top1_rmsd_A", "V2 selected metrics",
    )
    operational_columns = [
        "chemem_top1_rmsd_A", "v1_top1_rmsd_A", "v2_top1_rmsd_A"
    ]
    operational = complete_intersection(
        chemem, [v1_selected, v2_selected], operational_columns
    )
    chemem_ids = finite_ids(chemem, "chemem_top1_rmsd_A")
    v1_ids = finite_ids(v1_selected, "v1_top1_rmsd_A")
    v2_ids = finite_ids(v2_selected, "v2_top1_rmsd_A")
    common_ids = chemem_ids & v1_ids & v2_ids
    intersection_audit = {
        "chemem_complete_with_finite_top1_n": len(chemem_ids),
        "v1_selected_with_finite_top1_n": len(v1_ids),
        "v2_selected_with_finite_top1_n": len(v2_ids),
        "three_method_common_n": len(common_ids),
        "chemem_complete_absent_or_invalid_in_v1_n": len(chemem_ids - v1_ids),
        "chemem_complete_absent_or_invalid_in_v1": sorted(chemem_ids - v1_ids),
        "chemem_complete_absent_or_invalid_in_v2_n": len(chemem_ids - v2_ids),
        "chemem_complete_absent_or_invalid_in_v2": sorted(chemem_ids - v2_ids),
        "common_complexes": sorted(common_ids),
    }
    if args.expected_n is not None and len(operational) != args.expected_n:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "intersection_audit.json").write_text(
            json.dumps(intersection_audit, indent=2) + "\n"
        )
        print("INTERSECTION AUDIT")
        print(json.dumps(intersection_audit, indent=2))
        raise ValueError(
            f"Expected {args.expected_n} common completed cases, found "
            f"{len(operational)}. See {output / 'intersection_audit.json'}."
        )

    operational_methods = [
        ("ChemEM Stage 1", "chemem_top1_rmsd_A"),
        ("CryoDiffDock-Local V1 frozen S0", "v1_top1_rmsd_A"),
        ("CryoDiffDock-Local V2 selected", "v2_top1_rmsd_A"),
    ]
    summaries = [method_summary(operational, operational_methods, "top1")]
    paired_tables = []
    paired, operational_omnibus = paired_summary(
        operational, operational_methods, "top1"
    )
    paired_tables.append(paired)
    omnibus = {"top1": operational_omnibus}
    resolution_summaries = []
    resolution_pairwise_tables = []
    resolution_omnibus: dict[str, object] = {}
    resolution_summary, resolution_pairwise, resolution_tests = (
        resolution_stratified_analysis(
            operational, operational_methods, "top1"
        )
    )
    if not resolution_summary.empty:
        resolution_summaries.append(resolution_summary)
        resolution_pairwise_tables.append(resolution_pairwise)
        resolution_omnibus["top1"] = resolution_tests

    oracle = None
    if args.v1_oracle is not None and args.v2_oracle is not None:
        v1_oracle = load_metric(
            args.v1_oracle.expanduser().resolve(),
            "best_of_n_rmsd_A", "v1_oracle_rmsd_A", "V1 oracle metrics",
        )
        v2_oracle = load_metric(
            args.v2_oracle.expanduser().resolve(),
            "best_of_n_rmsd_A", "v2_oracle_rmsd_A", "V2 oracle metrics",
        )
        oracle_columns = [
            "chemem_oracle_rmsd_A", "v1_oracle_rmsd_A", "v2_oracle_rmsd_A"
        ]
        oracle_base_columns = [
            "complex_name", "chemem_oracle_rmsd_A", *metadata_columns
        ]
        oracle = complete_intersection(
            operational[oracle_base_columns],
            [v1_oracle, v2_oracle],
            oracle_columns,
        )
        if len(oracle) != len(operational):
            raise ValueError(
                f"Oracle intersection ({len(oracle)}) differs from Top-1 "
                f"intersection ({len(operational)})"
            )
        oracle_methods = [
            ("ChemEM Stage 1", "chemem_oracle_rmsd_A"),
            ("CryoDiffDock-Local V1 pool", "v1_oracle_rmsd_A"),
            ("CryoDiffDock-Local V2 pool", "v2_oracle_rmsd_A"),
        ]
        summaries.append(method_summary(oracle, oracle_methods, "oracle"))
        paired, oracle_omnibus = paired_summary(oracle, oracle_methods, "oracle")
        paired_tables.append(paired)
        omnibus["oracle"] = oracle_omnibus
        resolution_summary, resolution_pairwise, resolution_tests = (
            resolution_stratified_analysis(oracle, oracle_methods, "oracle")
        )
        if not resolution_summary.empty:
            resolution_summaries.append(resolution_summary)
            resolution_pairwise_tables.append(resolution_pairwise)
            resolution_omnibus["oracle"] = resolution_tests

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    operational.to_csv(output / "top1_per_complex_common.csv", index=False)
    if oracle is not None:
        oracle.to_csv(output / "oracle_per_complex_common.csv", index=False)
    summary_table = pd.concat(summaries, ignore_index=True)
    pairwise_table = pd.concat(paired_tables, ignore_index=True)
    summary_table.to_csv(output / "method_summary.csv", index=False)
    pairwise_table.to_csv(output / "pairwise_comparisons.csv", index=False)
    if resolution_summaries:
        resolution_summary_table = pd.concat(
            resolution_summaries, ignore_index=True
        )
        resolution_pairwise_table = pd.concat(
            resolution_pairwise_tables, ignore_index=True
        )
        resolution_summary_table.to_csv(
            output / "resolution_method_summary.csv", index=False
        )
        resolution_pairwise_table.to_csv(
            output / "resolution_pairwise_comparisons.csv", index=False
        )
        (output / "resolution_omnibus_tests.json").write_text(
            json.dumps(resolution_omnibus, indent=2) + "\n"
        )
    else:
        resolution_summary_table = pd.DataFrame()
        resolution_pairwise_table = pd.DataFrame()
    metadata = {
        "num_common_completed_cases": len(operational),
        "chemem_benchmark": str(chemem_path),
        "v1_selected": str(args.v1_selected.expanduser().resolve()),
        "v2_selected": str(args.v2_selected.expanduser().resolve()),
        "v1_oracle": (
            str(args.v1_oracle.expanduser().resolve()) if args.v1_oracle else None
        ),
        "v2_oracle": (
            str(args.v2_oracle.expanduser().resolve()) if args.v2_oracle else None
        ),
        "intersection_rule": (
            "ChemEM complete with finite ChemEM, V1, and V2 metrics"
        ),
        "intersection_audit": intersection_audit,
        "top1_primary_analysis": True,
        "oracle_warning": (
            "Oracle values are candidate-pool coverage ceilings. Do not compare "
            "them as deployment accuracy without compute-matched pool sizes."
        ),
        "omnibus_tests": omnibus,
        "resolution_strata_source": (
            "preassigned ChemEM benchmark resolution_stratum"
            if "resolution_stratum" in operational else None
        ),
        "resolution_omnibus_tests": resolution_omnibus,
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")

    display_columns = [
        "endpoint", "method", "num_complexes", "median_rmsd_A",
        "lt2A_count", "lt2A_pct", "lt5A_count", "lt5A_pct",
    ]
    print(f"COMMON COMPLETED CASES: {len(operational)}")
    print("\nTHREE-METHOD SUMMARY")
    print(summary_table[display_columns].to_string(index=False))
    print("\nPAIRWISE TESTS AT 2 A")
    print(pairwise_table.loc[
        pairwise_table["cutoff_A"].eq(2.0),
        [
            "endpoint", "left_method", "right_method", "num_paired",
            "median_delta_right_minus_left_A",
            "delta_right_minus_left_percentage_points",
            "wilcoxon_pvalue", "wilcoxon_holm_pvalue",
            "mcnemar_exact_pvalue", "mcnemar_holm_pvalue",
        ],
    ].to_string(index=False))
    if not resolution_summary_table.empty:
        print("\nRESOLUTION-STRATIFIED SUMMARY")
        print(resolution_summary_table[[
            "resolution_stratum", "endpoint", "method",
            "stratum_num_complexes", "median_resolution_A",
            "median_rmsd_A", "lt2A_count", "lt2A_pct",
            "lt5A_count", "lt5A_pct",
        ]].to_string(index=False))
    print(f"\nV1_V2_CHEMEM_COMPARISON_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
