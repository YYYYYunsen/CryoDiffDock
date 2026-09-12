#!/usr/bin/env python3
"""Assemble known-site GNINA/Vina, DiffDock-Pocket, and Local V2 comparisons."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHODS = (
    "GNINA executable (Vina scoring)",
    "DiffDock-Pocket",
    "CryoDiffDock-Local V2 selected",
)
ID_COLUMNS = ("complex_name", "complex_id", "pdb_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--accuracy-ledger", required=True, type=Path)
    parser.add_argument("--runtime-long", required=True, type=Path)
    parser.add_argument("--cryodiffdock-runtime-raw", required=True, type=Path)
    parser.add_argument("--gnina-summary", required=True, type=Path)
    parser.add_argument("--diffdock-metrics-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--expected-diffdock-scored-n", type=int, default=172)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate cohort IDs in {path}")
    return values


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator)


def normalize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    column = next((name for name in ID_COLUMNS if name in frame), None)
    if column is None:
        raise ValueError(f"{label} lacks a recognized ID column")
    result = frame.copy()
    result["complex_name"] = result[column].astype(str).str.strip().str.upper()
    return result


def discover_diffdock_metrics(roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file():
            candidates.append(resolved)
        elif resolved.is_dir():
            candidates.extend(sorted(resolved.rglob("per_complex.csv")))
    selected: list[Path] = []
    for path in candidates:
        try:
            frame = read_table(path)
        except Exception:
            continue
        if "top1_rmsd_A" not in frame:
            continue
        condition = frame.get("condition", pd.Series(dtype=str)).astype(str).str.lower()
        path_text = str(path).lower()
        if "raw_baseline" in path_text or (len(condition) and condition.eq("raw_baseline").all()):
            selected.append(path)
    if not selected:
        raise FileNotFoundError("No raw_baseline per_complex.csv found under DiffDock metric roots")
    return selected


def wilson(success: int, denominator: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if denominator <= 0:
        return math.nan, math.nan
    p = success / denominator
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return 100 * (center - half), 100 * (center + half)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> int:
    args = parse_args()
    cohort = read_ids(args.cohort.expanduser().resolve())
    if len(cohort) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cohort IDs, found {len(cohort)}")
    order = {case_id: index for index, case_id in enumerate(cohort)}
    ledger = normalize(read_table(args.accuracy_ledger.expanduser().resolve()), "accuracy ledger")
    ledger = ledger.set_index("complex_name").reindex(cohort)
    local_rmsd_column = next(
        column for column in ("cryodiffdock_local_v2_top1_rmsd_A", "v2_rmsd_A", "top1_rmsd_A")
        if column in ledger
    )
    chemem_rmsd_column = next(
        column for column in ("chemem_top1_rmsd_A", "chemem_rmsd_A") if column in ledger
    )

    gnina = normalize(read_table(args.gnina_summary.expanduser().resolve()), "GNINA summary")
    gnina["top1_rmsd_A"] = pd.to_numeric(gnina["top1_rmsd_A"], errors="coerce")
    gnina["elapsed_s"] = pd.to_numeric(gnina["elapsed_s"], errors="coerce")
    gnina = gnina[gnina["complex_name"].isin(cohort)].sort_values("complex_name")
    gnina = gnina.drop_duplicates("complex_name", keep="last").set_index("complex_name")

    metric_paths = discover_diffdock_metrics(args.diffdock_metrics_root)
    metric_frames = []
    for priority, path in enumerate(metric_paths):
        frame = normalize(read_table(path), str(path))
        frame["source_file"] = str(path)
        frame["source_priority"] = priority
        metric_frames.append(frame)
    diffdock = pd.concat(metric_frames, ignore_index=True, sort=False)
    diffdock["top1_rmsd_A"] = pd.to_numeric(diffdock["top1_rmsd_A"], errors="coerce")
    diffdock = (
        diffdock[diffdock["complex_name"].isin(cohort)]
        .sort_values("source_priority", ascending=False, kind="stable")
        .drop_duplicates("complex_name", keep="first")
        .set_index("complex_name")
    )
    missing_diffdock_rows = [case_id for case_id in cohort if case_id not in diffdock.index]
    diffdock_scored_ids = set(
        diffdock.loc[np.isfinite(diffdock["top1_rmsd_A"])]
        .index.astype(str)
    )
    diffdock_technical_failure_ids = [
        case_id for case_id in cohort if case_id not in diffdock_scored_ids
    ]
    if len(diffdock_scored_ids) != args.expected_diffdock_scored_n:
        raise ValueError(
            "Expected exactly "
            f"{args.expected_diffdock_scored_n} scored DiffDock-Pocket cases, "
            f"found {len(diffdock_scored_ids)}; technical failures="
            f"{diffdock_technical_failure_ids}"
        )

    raw_runtime = normalize(
        read_table(args.cryodiffdock_runtime_raw.expanduser().resolve()), "CryoDiffDock runtime raw"
    ).set_index("complex_name")
    required_runtime = "baseline_inference_seconds_reconstructed"
    if required_runtime not in raw_runtime:
        raise ValueError(
            f"{required_runtime} is absent. Rerun prepare_figure2_authoritative174_runtime.sh "
            "with the updated extractor before packaging."
        )
    runtime_long = normalize(
        read_table(args.runtime_long.expanduser().resolve()), "runtime ledger"
    )
    runtime_long["method"] = runtime_long["method"].replace(
        {"CryoDiffDock-Local V2": "CryoDiffDock-Local V2 selected"}
    )
    runtime_long["runtime_seconds"] = pd.to_numeric(runtime_long["wall_s"], errors="coerce")
    method_runtime = runtime_long.set_index(["complex_name", "method"])["runtime_seconds"]

    rows: list[dict[str, object]] = []
    for case_id in cohort:
        sources = (
            (
                METHODS[0],
                gnina.loc[case_id, "top1_rmsd_A"] if case_id in gnina.index else math.nan,
                gnina.loc[case_id, "elapsed_s"] if case_id in gnina.index else math.nan,
                str(args.gnina_summary.expanduser().resolve()),
                "known pocket defined from deposited ligand; GNINA executable with Vina scoring",
            ),
            (
                METHODS[1],
                diffdock.loc[case_id, "top1_rmsd_A"] if case_id in diffdock.index else math.nan,
                raw_runtime.loc[case_id, required_runtime] if case_id in raw_runtime.index else math.nan,
                diffdock.loc[case_id, "source_file"] if case_id in diffdock.index else "",
                "density-free baseline arm; known local pocket; Top-1 confidence pose",
            ),
            (
                METHODS[2],
                ledger.loc[case_id, local_rmsd_column],
                method_runtime.get((case_id, METHODS[2]), math.nan),
                str(args.accuracy_ledger.expanduser().resolve()),
                "known local pocket plus cryo-EM density; reference-free V2 Top-1",
            ),
        )
        for method, rmsd_value, runtime_value, source, protocol in sources:
            rmsd = pd.to_numeric(pd.Series([rmsd_value]), errors="coerce").iloc[0]
            runtime_s = pd.to_numeric(pd.Series([runtime_value]), errors="coerce").iloc[0]
            scored = bool(np.isfinite(rmsd))
            rows.append({
                "cohort_order": order[case_id],
                "complex_name": case_id,
                "method": method,
                "top1_rmsd_A": rmsd if scored else math.nan,
                "scored": scored,
                "success_lt2_A": scored and rmsd < 2.0,
                "success_lt5_A": scored and rmsd < 5.0,
                "missing_counted_as_failure": not scored,
                "runtime_seconds": runtime_s,
                "runtime_minutes": runtime_s / 60.0 if np.isfinite(runtime_s) else math.nan,
                "source_file": source,
                "protocol": protocol,
            })
    per_case = pd.DataFrame(rows)

    summary_rows = []
    for method in METHODS:
        frame = per_case[per_case["method"].eq(method)]
        scored = pd.to_numeric(frame.loc[frame["scored"], "top1_rmsd_A"], errors="coerce").dropna()
        runtime = pd.to_numeric(frame["runtime_minutes"], errors="coerce").dropna()
        for cutoff in (2.0, 5.0):
            success = int(frame[f"success_lt{int(cutoff)}_A"].sum())
            low, high = wilson(success, len(cohort))
            summary_rows.append({
                "method": method,
                "cutoff_A": cutoff,
                "success_count": success,
                "denominator": len(cohort),
                "success_pct": 100 * success / len(cohort),
                "wilson_95ci_low_pct": low,
                "wilson_95ci_high_pct": high,
                "scored_n": len(scored),
                "missing_or_failed_n": len(cohort) - len(scored),
                "median_top1_rmsd_A_among_scored": float(scored.median()) if len(scored) else math.nan,
                "mean_top1_rmsd_A_among_scored": float(scored.mean()) if len(scored) else math.nan,
                "runtime_n": len(runtime),
                "median_runtime_min": float(runtime.median()) if len(runtime) else math.nan,
                "mean_runtime_min": float(runtime.mean()) if len(runtime) else math.nan,
            })
    summary = pd.DataFrame(summary_rows)

    wide = per_case.pivot(index="complex_name", columns="method", values="top1_rmsd_A")
    test_rows = []
    for cutoff in (2.0, 5.0):
        for left_index in range(len(METHODS)):
            for right_index in range(left_index + 1, len(METHODS)):
                left, right = METHODS[left_index], METHODS[right_index]
                pair = wide[[left, right]].dropna()
                left_success = pair[left].lt(cutoff)
                right_success = pair[right].lt(cutoff)
                right_only = int((~left_success & right_success).sum())
                left_only = int((left_success & ~right_success).sum())
                discordant = right_only + left_only
                pvalue = float(stats.binomtest(right_only, discordant, 0.5).pvalue) if discordant else 1.0
                wilcoxon = stats.wilcoxon(pair[right] - pair[left], zero_method="wilcox")
                test_rows.append({
                    "cutoff_A": cutoff,
                    "left_method": left,
                    "right_method": right,
                    "paired_n": len(pair),
                    "left_success_count": int(left_success.sum()),
                    "right_success_count": int(right_success.sum()),
                    "delta_right_minus_left_success_pp": 100 * (right_success.mean() - left_success.mean()),
                    "right_only_successes": right_only,
                    "left_only_successes": left_only,
                    "exact_mcnemar_pvalue": pvalue,
                    "paired_rmsd_wilcoxon_pvalue": float(wilcoxon.pvalue),
                })
    tests = pd.DataFrame(test_rows)
    tests["mcnemar_holm_pvalue_within_cutoff"] = math.nan
    tests["wilcoxon_holm_pvalue_within_cutoff"] = math.nan
    for cutoff in (2.0, 5.0):
        mask = tests["cutoff_A"].eq(cutoff)
        tests.loc[mask, "mcnemar_holm_pvalue_within_cutoff"] = holm(
            tests.loc[mask, "exact_mcnemar_pvalue"].tolist()
        )
        tests.loc[mask, "wilcoxon_holm_pvalue_within_cutoff"] = holm(
            tests.loc[mask, "paired_rmsd_wilcoxon_pvalue"].tolist()
        )

    # Accuracy–runtime positioning includes the two density-aware refinement
    # references, but they are intentionally not part of the conventional
    # known-site accuracy bars.  The explicit panel_role keeps these protocol
    # families visually distinct in the figure.
    speed_accuracy = summary.drop_duplicates("method").copy()
    speed_accuracy["panel_role"] = speed_accuracy["method"].map({
        METHODS[0]: "density_free_known_site_baseline",
        METHODS[1]: "density_free_known_site_baseline",
        METHODS[2]: "proposed_density_conditioned_method",
    })
    chemem_accuracy = ledger[chemem_rmsd_column].pipe(pd.to_numeric, errors="coerce")
    chemem_runtime = runtime_long.loc[runtime_long["method"].eq("ChemEM Stage 1"), "runtime_seconds"].dropna() / 60
    emerald_rmsd_column = next(
        column for column in ("emerald_top1_rmsd_A", "emerald_rmsd_A") if column in ledger
    )
    emerald_accuracy = ledger[emerald_rmsd_column].pipe(pd.to_numeric, errors="coerce")
    emerald_runtime = runtime_long.loc[
        runtime_long["method"].eq("EMERALD"), "runtime_seconds"
    ].dropna() / 60

    reference_rows = []
    for method, accuracy, runtime in (
        ("ChemEM Stage 1", chemem_accuracy, chemem_runtime),
        ("EMERALD", emerald_accuracy, emerald_runtime),
    ):
        success_count = int(accuracy.lt(2).sum())
        low, high = wilson(success_count, len(cohort))
        reference_rows.append({
            "method": method,
            "cutoff_A": 2.0,
            "success_count": success_count,
            "denominator": len(cohort),
            "success_pct": 100 * success_count / len(cohort),
            "wilson_95ci_low_pct": low,
            "wilson_95ci_high_pct": high,
            "scored_n": int(accuracy.notna().sum()),
            "missing_or_failed_n": int(accuracy.isna().sum()),
            "median_top1_rmsd_A_among_scored": float(accuracy.median()),
            "mean_top1_rmsd_A_among_scored": float(accuracy.mean()),
            "runtime_n": len(runtime),
            "median_runtime_min": float(runtime.median()),
            "mean_runtime_min": float(runtime.mean()),
            "panel_role": "density_aware_refinement_reference",
        })
    speed_accuracy = pd.concat(
        [speed_accuracy, pd.DataFrame(reference_rows)], ignore_index=True, sort=False
    )

    protocols = pd.DataFrame([
        {"method": METHODS[0], "density_used": False, "site_information": "deposited-ligand-defined known pocket", "top1_selection": "Vina score", "runtime_definition": "GNINA/Vina docking elapsed_s", "interpretation": "privileged known-pocket conventional docking baseline"},
        {"method": METHODS[1], "density_used": False, "site_information": "known local pocket", "top1_selection": "DiffDock confidence", "runtime_definition": "baseline inference arm including amortized preparation/loading", "interpretation": "density-free learned docking baseline"},
        {"method": METHODS[2], "density_used": True, "site_information": "same known local pocket", "top1_selection": "reference-free density/physics V2 selector", "runtime_definition": "baseline+guided inference plus V2 search/selection and Top-1 materialization", "interpretation": "proposed density-conditioned method"},
        {"method": "ChemEM Stage 1", "density_used": True, "site_information": "known density site", "top1_selection": "first ranked Stage-1 fit", "runtime_definition": "summed iteration runtime", "interpretation": "expensive density-refinement reference; speed–accuracy panel only"},
        {"method": "EMERALD", "density_used": True, "site_information": "known density site", "top1_selection": "EMERALD ranked Top-1", "runtime_definition": "recorded per-case wall time with timeout cap", "interpretation": "density-refinement reference; speed–accuracy panel only"},
    ])

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(output / "known_site_baselines_per_complex.csv", index=False)
    summary.to_csv(output / "known_site_baselines_summary.csv", index=False)
    speed_accuracy.to_csv(output / "known_site_speed_accuracy.csv", index=False)
    tests.to_csv(output / "known_site_baselines_paired_tests.csv", index=False)
    protocols.to_csv(output / "known_site_baselines_protocols.csv", index=False)
    audit = {
        "cohort_n": len(cohort),
        "diffdock_metric_files": [str(path) for path in metric_paths],
        "diffdock_scored_n": len(diffdock_scored_ids),
        "missing_diffdock_metric_ids": missing_diffdock_rows,
        "diffdock_technical_failure_ids": diffdock_technical_failure_ids,
        "gnina_label": "GNINA executable (Vina scoring)",
        "gnina_is_independent_docking": True,
        "gnina_candidate_score_only_outputs_used": False,
        "gnina_pocket_is_privileged": True,
        "fixed_denominator_missing_as_failure": True,
        "median_rmsd_uses_scored_cases_only": True,
    }
    (output / "known_site_baselines_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"FIGURE2_KNOWN_SITE_BASELINES_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
