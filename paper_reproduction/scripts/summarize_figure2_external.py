#!/usr/bin/env python3
"""Generate the frozen-99 and strict-paired Figure 2 statistics."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, friedmanchisquare, wilcoxon


EXTERNAL_METHODS = (
    "ChemEM Stage 1",
    "CryoDiffDock-Local V2 selected",
    "EMERALD",
)
CUTOFFS = (2.0, 5.0)


def boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "complete", "completed", "success", "succeeded", "ok"}
    )


def holm(values: Iterable[float]) -> list[float]:
    pvalues = np.asarray(list(values), dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * pvalues[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    values = matrix.astype(float)
    k = values.shape[1]
    column_sums = values.sum(axis=0)
    row_sums = values.sum(axis=1)
    total = column_sums.sum()
    denominator = k * total - np.square(row_sums).sum()
    if denominator <= 0:
        return 0.0, 1.0
    statistic = (k - 1.0) * (k * np.square(column_sums).sum() - total * total) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def bootstrap_ci(values: np.ndarray, statistic, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    sampled = statistic(values[indices], axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def write(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-long", required=True, type=Path)
    parser.add_argument("--runtime-long", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-frozen-n", type=int, default=99)
    parser.add_argument("--expected-three-software-n", type=int)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument(
        "--overwrite-output", action="store_true",
        help="Allow replacing this script's derived tables in a non-empty output directory.",
    )
    args = parser.parse_args()

    accuracy_path = args.accuracy_long.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite_output:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(accuracy_path, dtype={"complex_id": str})
    required = {"complex_id", "method", "completed", "top1_symmetry_corrected_rmsd_A"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Accuracy ledger lacks columns: {missing}")
    table["complex_id"] = table["complex_id"].str.strip().str.upper()
    if table.duplicated(["complex_id", "method"]).any():
        raise ValueError("Accuracy ledger has duplicate complex/method rows")
    cohort_ids = table["complex_id"].drop_duplicates().tolist()
    if len(cohort_ids) != args.expected_frozen_n:
        raise ValueError(f"Frozen cohort contains {len(cohort_ids)} IDs, expected {args.expected_frozen_n}")
    external = table[table["method"].isin(EXTERNAL_METHODS)].copy()
    counts = external.groupby("complex_id")["method"].nunique()
    if len(counts) != len(cohort_ids) or not counts.eq(len(EXTERNAL_METHODS)).all():
        raise ValueError("Every frozen ID must have one ledger row for each external method")
    external["completed_bool"] = boolean(external["completed"])
    external["rmsd_A"] = pd.to_numeric(external["top1_symmetry_corrected_rmsd_A"], errors="coerce")
    external["scored"] = external["completed_bool"] & np.isfinite(external["rmsd_A"])

    complete_matrix = external.pivot(index="complex_id", columns="method", values="scored")
    paired_ids = [complex_id for complex_id in cohort_ids if bool(complete_matrix.loc[complex_id, list(EXTERNAL_METHODS)].all())]
    if args.expected_three_software_n is not None and len(paired_ids) != args.expected_three_software_n:
        raise ValueError(
            f"Strict three-software paired N is {len(paired_ids)}, expected {args.expected_three_software_n}"
        )
    (output / "three_software_common_ids.txt").write_text("".join(f"{value}\n" for value in paired_ids))

    accuracy_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for analysis_set, ids, missing_as_failure in (
        (f"frozen_{len(cohort_ids)}_missing_as_failure", cohort_ids, True),
        ("three_software_paired_complete", paired_ids, False),
    ):
        subset = external[external["complex_id"].isin(ids)]
        for method in EXTERNAL_METHODS:
            method_rows = subset[subset["method"].eq(method)]
            values = method_rows.loc[method_rows["scored"], "rmsd_A"].to_numpy(float)
            denominator = len(ids)
            distribution_rows.append({
                "analysis_set": analysis_set,
                "method": method,
                "frozen_or_paired_n": denominator,
                "completed_scored_n": len(values),
                "missing_or_failed_n": denominator - len(values),
                "median_rmsd_A_among_scored": float(np.median(values)) if len(values) else float("nan"),
                "q1_rmsd_A_among_scored": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
                "q3_rmsd_A_among_scored": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
                "missing_counted_as_failure_for_success_rates": missing_as_failure,
            })
            for cutoff in CUTOFFS:
                success = int((values < cutoff).sum())
                accuracy_rows.append({
                    "analysis_set": analysis_set,
                    "method": method,
                    "cutoff_A": cutoff,
                    "success_count": success,
                    "denominator": denominator,
                    "success_pct": 100.0 * success / denominator if denominator else float("nan"),
                    "completed_scored_n": len(values),
                    "missing_or_failed_n": denominator - len(values),
                    "missing_counted_as_failure": missing_as_failure,
                })
    write(output / "accuracy_success_summary.csv", accuracy_rows)
    write(output / "accuracy_distribution_summary.csv", distribution_rows)

    paired = external[external["complex_id"].isin(paired_ids)].pivot(
        index="complex_id", columns="method", values="rmsd_A"
    ).loc[paired_ids, list(EXTERNAL_METHODS)]
    rng = np.random.default_rng(args.seed)
    continuous_rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(EXTERNAL_METHODS, 2):
        delta = paired[right].to_numpy(float) - paired[left].to_numpy(float)
        try:
            pvalue = float(wilcoxon(paired[right], paired[left]).pvalue)
        except ValueError:
            pvalue = 1.0
        low, high = bootstrap_ci(delta, np.median, args.bootstrap_repeats, rng)
        continuous_rows.append({
            "left_method": left,
            "right_method": right,
            "paired_n": len(paired),
            "median_delta_right_minus_left_rmsd_A": float(np.median(delta)),
            "bootstrap_95ci_low_A": low,
            "bootstrap_95ci_high_A": high,
            "wilcoxon_pvalue": pvalue,
        })
    adjusted = holm(row["wilcoxon_pvalue"] for row in continuous_rows)
    for row, value in zip(continuous_rows, adjusted):
        row["wilcoxon_holm_pvalue"] = value
    write(output / "paired_rmsd_tests.csv", continuous_rows)

    binary_rows: list[dict[str, object]] = []
    omnibus_rows: list[dict[str, object]] = []
    if len(paired):
        statistic, pvalue = friedmanchisquare(*(paired[method].to_numpy(float) for method in EXTERNAL_METHODS))
        omnibus_rows.append({
            "endpoint": "top1_rmsd_A",
            "cutoff_A": "",
            "test": "Friedman",
            "paired_n": len(paired),
            "statistic": float(statistic),
            "pvalue": float(pvalue),
        })
    for cutoff in CUTOFFS:
        success = paired.lt(cutoff)
        q_stat, q_p = cochran_q(success.to_numpy(bool))
        omnibus_rows.append({
            "endpoint": "top1_success",
            "cutoff_A": cutoff,
            "test": "Cochran_Q",
            "paired_n": len(paired),
            "statistic": q_stat,
            "pvalue": q_p,
        })
        cutoff_rows = []
        for left, right in itertools.combinations(EXTERNAL_METHODS, 2):
            left_values = success[left].to_numpy(bool)
            right_values = success[right].to_numpy(bool)
            left_only = int((left_values & ~right_values).sum())
            right_only = int((~left_values & right_values).sum())
            delta = 100.0 * (right_values.astype(float) - left_values.astype(float))
            low, high = bootstrap_ci(delta, np.mean, args.bootstrap_repeats, rng)
            cutoff_rows.append({
                "cutoff_A": cutoff,
                "left_method": left,
                "right_method": right,
                "paired_n": len(paired),
                "left_success_count": int(left_values.sum()),
                "right_success_count": int(right_values.sum()),
                "delta_right_minus_left_percentage_points": float(delta.mean()),
                "bootstrap_95ci_low_pp": low,
                "bootstrap_95ci_high_pp": high,
                "left_only_successes": left_only,
                "right_only_successes": right_only,
                "exact_mcnemar_pvalue": exact_mcnemar(left_only, right_only),
            })
        adjusted = holm(row["exact_mcnemar_pvalue"] for row in cutoff_rows)
        for row, value in zip(cutoff_rows, adjusted):
            row["exact_mcnemar_holm_pvalue"] = value
        binary_rows.extend(cutoff_rows)
    write(output / "paired_success_tests.csv", binary_rows)
    write(output / "omnibus_tests.csv", omnibus_rows)

    resolution_rows: list[dict[str, object]] = []
    if "resolution_stratum" in external.columns:
        paired_long = external[external["complex_id"].isin(paired_ids)].copy()
        paired_long["resolution_stratum"] = paired_long["resolution_stratum"].fillna("").astype(str)
        for stratum in sorted(value for value in paired_long["resolution_stratum"].unique() if value):
            stratum_rows = paired_long[paired_long["resolution_stratum"].eq(stratum)]
            stratum_ids = stratum_rows["complex_id"].nunique()
            for method in EXTERNAL_METHODS:
                values = stratum_rows[stratum_rows["method"].eq(method)]["rmsd_A"].to_numpy(float)
                for cutoff in CUTOFFS:
                    resolution_rows.append({
                        "resolution_stratum": stratum,
                        "method": method,
                        "paired_n": stratum_ids,
                        "cutoff_A": cutoff,
                        "success_count": int((values < cutoff).sum()),
                        "success_pct": 100.0 * float((values < cutoff).mean()) if len(values) else float("nan"),
                        "median_rmsd_A": float(np.median(values)) if len(values) else float("nan"),
                    })
    write(output / "resolution_stratified_accuracy.csv", resolution_rows)

    runtime_rows: list[dict[str, object]] = []
    runtime_observed_common_n = None
    accuracy_runtime_common_n = None
    if args.runtime_long:
        runtime = pd.read_csv(args.runtime_long.expanduser().resolve(), dtype={"complex_id": str})
        runtime["complex_id"] = runtime["complex_id"].str.strip().str.upper()
        if runtime.duplicated(["complex_id", "method"]).any():
            raise ValueError("Runtime ledger has duplicate complex/method rows")
        runtime["wall_s_numeric"] = pd.to_numeric(runtime["wall_s"], errors="coerce")
        runtime["runtime_observed"] = np.isfinite(runtime["wall_s_numeric"])
        runtime["run_completed"] = boolean(runtime["completed"])
        runtime = runtime[runtime["complex_id"].isin(cohort_ids)]
        accuracy_completion = table.copy()
        accuracy_completion["accuracy_complete"] = boolean(accuracy_completion["completed"]) & np.isfinite(
            pd.to_numeric(accuracy_completion["top1_symmetry_corrected_rmsd_A"], errors="coerce")
        )
        completion_lookup = accuracy_completion.set_index(["complex_id", "method"])["accuracy_complete"]
        runtime["successful_output"] = [
            bool(completion_lookup.get((complex_id, method), False))
            for complex_id, method in zip(runtime["complex_id"], runtime["method"])
        ]
        runtime_observed_matrix = runtime[runtime["method"].isin(EXTERNAL_METHODS)].pivot_table(
            index="complex_id", columns="method", values="runtime_observed", aggfunc="max"
        ).reindex(columns=EXTERNAL_METHODS, fill_value=False).fillna(False)
        runtime_observed_common_ids = [
            value for value in cohort_ids
            if value in runtime_observed_matrix.index and bool(runtime_observed_matrix.loc[value].all())
        ]
        accuracy_runtime_common_ids = [
            value for value in paired_ids if value in set(runtime_observed_common_ids)
        ]
        runtime_observed_common_n = len(runtime_observed_common_ids)
        accuracy_runtime_common_n = len(accuracy_runtime_common_ids)
        runtime_sets = (
            (
                f"frozen_{len(cohort_ids)}_all_observed_attempts",
                cohort_ids,
                tuple(dict.fromkeys(runtime["method"].tolist())),
                False,
            ),
            (
                f"frozen_{len(cohort_ids)}_successful_outputs_only",
                cohort_ids,
                tuple(dict.fromkeys(runtime["method"].tolist())),
                True,
            ),
            (
                "three_software_runtime_observed_common",
                runtime_observed_common_ids,
                EXTERNAL_METHODS,
                False,
            ),
            (
                "three_software_accuracy_and_runtime_common",
                accuracy_runtime_common_ids,
                EXTERNAL_METHODS,
                False,
            ),
        )
        for analysis_set, ids, methods, completed_only in runtime_sets:
            eligible = runtime[runtime["complex_id"].isin(ids)]
            subset = eligible[eligible["runtime_observed"]]
            if completed_only:
                subset = subset[subset["successful_output"]]
            for method in methods:
                values = subset.loc[subset["method"].eq(method), "wall_s_numeric"].to_numpy(float)
                if not len(values):
                    continue
                method_eligible = eligible[eligible["method"].eq(method)]
                observed = method_eligible[method_eligible["runtime_observed"]]
                runtime_rows.append({
                    "analysis_set": analysis_set,
                    "method": method,
                    "eligible_cohort_n": len(ids),
                    "runtime_n": len(values),
                    "runtime_observed_n": int(method_eligible["runtime_observed"].sum()),
                    "runtime_missing_n": len(ids) - int(method_eligible["runtime_observed"].sum()),
                    "successful_output_with_runtime_n": int((observed["successful_output"]).sum()),
                    "failed_or_incomplete_output_with_runtime_n": int((~observed["successful_output"]).sum()),
                    "median_wall_s": float(np.median(values)),
                    "q1_wall_s": float(np.quantile(values, 0.25)),
                    "q3_wall_s": float(np.quantile(values, 0.75)),
                    "mean_wall_s": float(np.mean(values)),
                    "timing_note": (
                        "method-native observed attempts, including failed/incomplete runs with finite wall time"
                        if not completed_only else
                        "method-native successful outputs only"
                    ),
                })
    write(output / "runtime_summary.csv", runtime_rows)

    summary = {
        "frozen_external_n": len(cohort_ids),
        "method_completed_n": {
            method: int(external[external["method"].eq(method)]["scored"].sum())
            for method in EXTERNAL_METHODS
        },
        "strict_three_software_paired_n": len(paired_ids),
        "strict_runtime_observed_common_n": runtime_observed_common_n,
        "strict_accuracy_and_runtime_common_n": accuracy_runtime_common_n,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "primary_external_methods": list(EXTERNAL_METHODS),
        "runtime_interpretation": "method-native measured runtime, not an identical end-to-end boundary",
    }
    (output / "figure2_statistics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "README.txt").write_text(
        "Primary accuracy: three_software_paired_complete rows and paired tests.\n"
        f"Sensitivity: frozen_{len(cohort_ids)}_missing_as_failure success rates.\n"
        "Pairwise p-values are Holm-adjusted within endpoint/cutoff families.\n"
        "Primary runtime summary includes every observed attempt, including failures; completed-only is secondary.\n"
        "Runtime is method-native and should not be labelled identical end-to-end.\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"FIGURE2_STATISTICS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
