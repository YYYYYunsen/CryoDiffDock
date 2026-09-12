#!/usr/bin/env python3
"""Merge historical/new ChemEM Stage-1 results and compare with Pocket.

The command is safe to rerun while new ChemEM jobs are finishing. Historical
successful results are retained. A newly completed repair/addition result is
evaluated from its ranked SDF files and takes precedence over the historical
row. The frozen cohort itself is never changed.

Two analyses are reported:

* frozen_attempted: the full frozen cohort is the denominator; missing/failed
  ChemEM predictions count as failures.
* paired_complete: only cases with valid ChemEM and Pocket results are compared.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex", "name")
RESULT_LINE = re.compile(
    r"^\s*(?P<iteration>\d+)\s*:\s*(?P<score>[-+0-9.eE]+)\s*$"
)
CUTOFFS_A = (1.0, 2.0, 5.0)
METHOD_ENDPOINTS = (
    ("ChemEM Stage 1", "Top-1", "chemem_top1_rmsd_A"),
    ("CryoDiffDock-Pocket S0", "Top-1", "pocket_s0_top1_rmsd_A"),
    ("ChemEM Stage 1", "Oracle", "chemem_oracle_rmsd_A"),
    ("CryoDiffDock-Pocket", "Oracle", "pocket_oracle_rmsd_A"),
)
PAIRED_COMPARISONS = (
    (
        "CryoDiffDock-Pocket S0_vs_ChemEM Stage 1",
        "chemem_top1_rmsd_A",
        "pocket_s0_top1_rmsd_A",
    ),
    (
        "CryoDiffDock-Pocket oracle_vs_ChemEM Stage 1 oracle",
        "chemem_oracle_rmsd_A",
        "pocket_oracle_rmsd_A",
    ),
)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def choose_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column found; columns={list(columns)}")


def load_unique_table(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str)
    id_column = choose_column(table.columns, ID_COLUMNS, f"{label} ID")
    table["complex_name"] = table[id_column].map(normalize_id)
    duplicates = sorted(
        table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:20]}")
    return table


def finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def resolve_path(value: object, table_parent: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = table_parent / path
    return path.resolve()


def read_heavy_molecule(path: Path):
    from rdkit import Chem

    if path.suffix.lower() in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif path.suffix.lower() == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand: {path}")
    return Chem.RemoveHs(molecule, sanitize=False)


def symmetry_rmsd(reference, prediction) -> float:
    from spyrmsd import molecule, rmsd

    reference_spy = molecule.Molecule.from_rdkit(reference)
    prediction_spy = molecule.Molecule.from_rdkit(prediction)
    value = rmsd.symmrmsd(
        np.asarray(reference.GetConformer().GetPositions()),
        np.asarray(prediction.GetConformer().GetPositions()),
        reference_spy.atomicnums,
        prediction_spy.atomicnums,
        reference_spy.adjacency_matrix,
        prediction_spy.adjacency_matrix,
    )
    return float(np.asarray(value).reshape(-1)[0])


def parse_ranked_results(results_path: Path) -> list[tuple[int, int, float, Path]]:
    ranked = []
    for rank, line in enumerate(results_path.read_text(errors="replace").splitlines(), 1):
        match = RESULT_LINE.match(line)
        if match is None:
            continue
        iteration = int(match.group("iteration"))
        score = float(match.group("score"))
        pose = results_path.parent / f"Ligand_{iteration}_refined.sdf"
        ranked.append((rank, iteration, score, pose))
    if not ranked:
        raise ValueError(f"No ranked result records in {results_path}")
    return ranked


def evaluate_new_case(
    task: tuple[str, str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    complex_name, reference_text, output_text = task
    reference_path = Path(reference_text)
    results_path = Path(output_text) / "fitting" / "results.txt"
    base = {
        "complex_name": complex_name,
        "chemem_source": "new_stage1_output",
        "chemem_results_path": str(results_path),
    }
    if not results_path.is_file() or results_path.stat().st_size == 0:
        return {
            **base,
            "chemem_source_status": "pending_or_failed",
            "chemem_evaluation_error": "results.txt missing or empty",
        }, []
    try:
        reference = read_heavy_molecule(reference_path)
        ranked = parse_ranked_results(results_path)
    except Exception as exc:
        return {
            **base,
            "chemem_source_status": "evaluation_failed",
            "chemem_evaluation_error": f"{type(exc).__name__}: {exc}",
        }, []

    poses = []
    errors = []
    for rank, iteration, score, pose_path in ranked:
        try:
            prediction = read_heavy_molecule(pose_path)
            poses.append({
                "complex_name": complex_name,
                "rank": rank,
                "iteration": iteration,
                "chemem_score": score,
                "rmsd_A": symmetry_rmsd(reference, prediction),
                "prediction": str(pose_path),
                "reference_ligand": str(reference_path),
            })
        except Exception as exc:
            errors.append(
                f"rank={rank}, iteration={iteration}: {type(exc).__name__}: {exc}"
            )
    if not poses:
        return {
            **base,
            "chemem_source_status": "evaluation_failed",
            "chemem_num_ranked": len(ranked),
            "chemem_num_evaluated": 0,
            "chemem_num_pose_failures": len(errors),
            "chemem_evaluation_error": " | ".join(errors[:5]),
        }, []
    pose_table = pd.DataFrame(poses).sort_values("rank")
    top1 = pose_table.iloc[0]
    if int(top1["rank"]) != 1:
        return {
            **base,
            "chemem_source_status": "evaluation_failed",
            "chemem_num_ranked": len(ranked),
            "chemem_num_evaluated": len(pose_table),
            "chemem_num_pose_failures": len(errors),
            "chemem_evaluation_error": "ChemEM rank-1 pose could not be evaluated",
        }, poses
    best = pose_table.loc[pose_table["rmsd_A"].idxmin()]
    return {
        **base,
        "chemem_source_status": "complete",
        "chemem_num_ranked": len(ranked),
        "chemem_num_evaluated": len(pose_table),
        "chemem_num_pose_failures": len(errors),
        "chemem_top1_iteration": int(top1["iteration"]),
        "chemem_top1_score": float(top1["chemem_score"]),
        "chemem_top1_rmsd_A": float(top1["rmsd_A"]),
        "chemem_oracle_rmsd_A": float(best["rmsd_A"]),
        "chemem_oracle_rank": int(best["rank"]),
        "chemem_oracle_iteration": int(best["iteration"]),
        "chemem_evaluation_error": " | ".join(errors[:5]),
    }, poses


def historical_result(row: pd.Series) -> dict[str, object]:
    top1 = finite_number(row.get("top1_rmsd_A"))
    oracle = finite_number(row.get("best_rmsd_A"))
    complete = str(row.get("status", "")).strip().lower() == "complete"
    complete = complete and math.isfinite(top1) and math.isfinite(oracle)
    return {
        "complex_name": normalize_id(row["complex_name"]),
        "chemem_source": "historical_results_table",
        "chemem_source_status": (
            "complete" if complete else str(row.get("status", "pending_or_failed"))
        ),
        "chemem_num_ranked": finite_number(row.get("num_ranked")),
        "chemem_num_evaluated": finite_number(row.get("num_evaluated")),
        "chemem_num_pose_failures": finite_number(row.get("num_pose_failures")),
        "chemem_top1_iteration": finite_number(row.get("top1_iteration")),
        "chemem_top1_score": finite_number(row.get("top1_score")),
        "chemem_top1_rmsd_A": top1 if complete else float("nan"),
        "chemem_oracle_rmsd_A": oracle if complete else float("nan"),
        "chemem_oracle_rank": finite_number(row.get("best_rank")),
        "chemem_oracle_iteration": finite_number(row.get("best_iteration")),
        "chemem_evaluation_error": "",
    }


def load_pocket(path: Path, value_column: str, output_column: str) -> pd.DataFrame:
    table = load_unique_table(path, "Pocket metrics")
    if value_column not in table.columns:
        raise ValueError(f"{path} lacks {value_column}")
    table[output_column] = pd.to_numeric(table[value_column], errors="coerce")
    return table[["complex_name", output_column]]


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


def build_rate_rows(
    cases: pd.DataFrame, deployment_only: bool = False,
) -> list[dict[str, object]]:
    rows = []
    paired = cases[cases["paired_complete"]]
    for analysis_set, frame, missing_as_failure in (
        ("frozen_attempted", cases, True),
        ("paired_complete", paired, False),
    ):
        method_endpoints = METHOD_ENDPOINTS[:2] if deployment_only else METHOD_ENDPOINTS
        for method, endpoint, column in method_endpoints:
            all_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(all_values)
            values = all_values[finite]
            denominator = len(frame) if missing_as_failure else len(values)
            median = float(np.median(values)) if len(values) else float("nan")
            for cutoff in CUTOFFS_A:
                successes = int((values < cutoff).sum())
                low, high = wilson_interval(successes, denominator)
                rows.append({
                    "analysis_set": analysis_set,
                    "method": method,
                    "endpoint": endpoint,
                    "cutoff_A": cutoff,
                    "success_count": successes,
                    "denominator": denominator,
                    "success_pct": 100.0 * successes / denominator if denominator else float("nan"),
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                    "num_scored": int(finite.sum()),
                    "median_rmsd_A_among_scored": median,
                    "missing_counted_as_failure": missing_as_failure,
                })
    return rows


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    from scipy.stats import binomtest

    return float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)


def build_paired_rows(
    cases: pd.DataFrame, bootstrap_repeats: int, seed: int,
    deployment_only: bool = False,
) -> list[dict[str, object]]:
    from scipy.stats import wilcoxon

    paired = cases[cases["paired_complete"]]
    rows = []
    rng = np.random.default_rng(seed)
    comparisons = PAIRED_COMPARISONS[:1] if deployment_only else PAIRED_COMPARISONS
    for comparison, left_column, right_column in comparisons:
        left = paired[left_column].to_numpy(float)
        right = paired[right_column].to_numpy(float)
        n = len(paired)
        try:
            wilcoxon_p = float(wilcoxon(right, left).pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        for cutoff in CUTOFFS_A:
            left_success = left < cutoff
            right_success = right < cutoff
            left_only = int((left_success & ~right_success).sum())
            right_only = int((~left_success & right_success).sum())
            delta = right_success.astype(float) - left_success.astype(float)
            if n:
                indices = rng.integers(0, n, size=(bootstrap_repeats, n))
                sampled = 100.0 * delta[indices].mean(axis=1)
                low, high = np.quantile(sampled, [0.025, 0.975])
            else:
                low = high = float("nan")
            rows.append({
                "comparison": comparison,
                "cutoff_A": cutoff,
                "num_paired": n,
                "left_method_column": left_column,
                "right_method_column": right_column,
                "left_success_count": int(left_success.sum()),
                "left_success_pct": 100.0 * left_success.mean() if n else float("nan"),
                "right_success_count": int(right_success.sum()),
                "right_success_pct": 100.0 * right_success.mean() if n else float("nan"),
                "delta_right_minus_left_percentage_points": (
                    100.0 * delta.mean() if n else float("nan")
                ),
                "delta_95ci_low_pp": float(low),
                "delta_95ci_high_pp": float(high),
                "right_only_successes": right_only,
                "left_only_successes": left_only,
                "exact_mcnemar_pvalue": exact_mcnemar(left_only, right_only),
                "paired_rmsd_wilcoxon_pvalue": wilcoxon_p,
            })
    return rows


def build_resolution_rows(
    cases: pd.DataFrame, deployment_only: bool = False,
) -> list[dict[str, object]]:
    rows = []
    available = cases["resolution_stratum"].dropna().astype(str).unique().tolist()
    preferred = ["higher_resolution", "middle_resolution", "lower_resolution"]
    strata = ["Overall"] + [value for value in preferred if value in available]
    strata += sorted(set(available) - set(strata))
    for stratum in strata:
        group = cases if stratum == "Overall" else cases[
            cases["resolution_stratum"].eq(stratum)
        ]
        paired = group[group["paired_complete"]]
        method_endpoints = METHOD_ENDPOINTS[:2] if deployment_only else METHOD_ENDPOINTS
        for method, endpoint, column in method_endpoints:
            values = paired[column].to_numpy(float)
            row = {
                "resolution_stratum": stratum,
                "resolution_median_A": float(group["resolution_A"].median()),
                "frozen_n": len(group),
                "chemem_complete_n": int(group["chemem_complete"].sum()),
                "paired_n": len(paired),
                "method": method,
                "endpoint": endpoint,
                "median_rmsd_A": float(np.median(values)) if len(values) else float("nan"),
            }
            for cutoff in CUTOFFS_A:
                row[f"lt{cutoff:g}A_count"] = int((values < cutoff).sum())
                row[f"lt{cutoff:g}A_pct"] = (
                    100.0 * float((values < cutoff).mean())
                    if len(values) else float("nan")
                )
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-chemem-results", required=True, type=Path)
    parser.add_argument("--new-cases", required=True, type=Path)
    parser.add_argument("--pocket-s0", required=True, type=Path)
    parser.add_argument("--pocket-oracle", type=Path)
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Report only selected Top-1 endpoints; exclude all Oracle comparisons.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--expected-cohort-n", type=int, default=100,
        help="Required frozen cohort size (default preserves ChemEM100).",
    )
    args = parser.parse_args()
    if args.workers <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("workers and bootstrap-repeats must be positive")

    cohort_path = args.cohort.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    old_path = args.old_chemem_results.expanduser().resolve()
    new_cases_path = args.new_cases.expanduser().resolve()
    cohort = load_unique_table(cohort_path, "cohort")
    if len(cohort) != args.expected_cohort_n:
        raise ValueError(
            f"Expected frozen cohort size {args.expected_cohort_n}, "
            f"found {len(cohort)} rows"
        )
    for column in ("resolution_A", "resolution_stratum", "selection_phase"):
        if column not in cohort.columns:
            raise ValueError(f"Cohort lacks {column}")
    cohort["resolution_A"] = pd.to_numeric(cohort["resolution_A"], errors="coerce")

    manifest = load_unique_table(manifest_path, "manifest")
    if "ligand" not in manifest.columns:
        raise ValueError("Manifest lacks ligand reference path")
    manifest_index = manifest.set_index("complex_name")
    missing_manifest = sorted(set(cohort["complex_name"]) - set(manifest_index.index))
    if missing_manifest:
        raise ValueError(f"Cohort IDs absent from manifest: {missing_manifest}")

    old = load_unique_table(old_path, "historical ChemEM results")
    old_results = {
        result["complex_name"]: result
        for _, row in old.iterrows()
        for result in [historical_result(row)]
    }
    new_cases = load_unique_table(new_cases_path, "new ChemEM cases")
    missing_new = sorted({"stage1_output", "case_group"} - set(new_cases.columns))
    if missing_new:
        raise ValueError(f"New cases table lacks columns: {missing_new}")

    tasks = []
    new_metadata = {}
    for _, row in new_cases.iterrows():
        complex_name = row["complex_name"]
        reference = resolve_path(
            manifest_index.loc[complex_name, "ligand"], manifest_path.parent
        )
        output = resolve_path(row["stage1_output"], new_cases_path.parent)
        tasks.append((complex_name, str(reference), str(output)))
        new_metadata[complex_name] = {
            "chemem_case_group": str(row["case_group"]),
            "chemem_stage1_output": str(output),
        }

    evaluated_new = {}
    pose_rows = []
    if args.workers == 1:
        results = map(evaluate_new_case, tasks)
        for result, poses in results:
            evaluated_new[result["complex_name"]] = result
            pose_rows.extend(poses)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(evaluate_new_case, task) for task in tasks]
            for future in as_completed(futures):
                result, poses = future.result()
                evaluated_new[result["complex_name"]] = result
                pose_rows.extend(poses)

    merged_chemem = []
    for complex_name in cohort["complex_name"]:
        historical = old_results.get(complex_name)
        new = evaluated_new.get(complex_name)
        if new is not None and new.get("chemem_source_status") == "complete":
            selected = dict(new)
        elif historical is not None and historical.get("chemem_source_status") == "complete":
            selected = dict(historical)
            if new is not None:
                selected["chemem_retry_status"] = new.get("chemem_source_status")
                selected["chemem_retry_error"] = new.get("chemem_evaluation_error", "")
        elif new is not None:
            selected = dict(new)
        elif historical is not None:
            selected = dict(historical)
        else:
            selected = {
                "complex_name": complex_name,
                "chemem_source": "none",
                "chemem_source_status": "not_started",
                "chemem_evaluation_error": "No historical or new result",
            }
        selected.update(new_metadata.get(complex_name, {}))
        merged_chemem.append(selected)

    cases = cohort.merge(
        pd.DataFrame(merged_chemem), on="complex_name", how="left", validate="one_to_one"
    )
    cases = cases.merge(
        load_pocket(
            args.pocket_s0.expanduser().resolve(),
            "top1_rmsd_A", "pocket_s0_top1_rmsd_A",
        ),
        on="complex_name", how="left", validate="one_to_one",
    )
    if args.deployment_only:
        cases["pocket_oracle_rmsd_A"] = float("nan")
    else:
        if args.pocket_oracle is None:
            raise ValueError("--pocket-oracle is required unless --deployment-only is set")
        cases = cases.merge(
            load_pocket(
                args.pocket_oracle.expanduser().resolve(),
                "best_of_n_rmsd_A", "pocket_oracle_rmsd_A",
            ),
            on="complex_name", how="left", validate="one_to_one",
        )
    numeric_columns = (
        "chemem_top1_rmsd_A", "chemem_oracle_rmsd_A",
        "pocket_s0_top1_rmsd_A", "pocket_oracle_rmsd_A",
    )
    for column in numeric_columns:
        if column not in cases.columns:
            cases[column] = float("nan")
        cases[column] = pd.to_numeric(cases[column], errors="coerce")
    cases["chemem_complete"] = (
        cases["chemem_source_status"].eq("complete")
        & np.isfinite(cases["chemem_top1_rmsd_A"])
        & np.isfinite(cases["chemem_oracle_rmsd_A"])
    )
    cases["pocket_complete"] = np.isfinite(cases["pocket_s0_top1_rmsd_A"])
    if not args.deployment_only:
        cases["pocket_complete"] &= np.isfinite(cases["pocket_oracle_rmsd_A"])
    cases["paired_complete"] = cases["chemem_complete"] & cases["pocket_complete"]
    chemem_success = cases["chemem_top1_rmsd_A"] < 2.0
    pocket_success = cases["pocket_s0_top1_rmsd_A"] < 2.0
    cases["top1_transition_lt2A"] = np.select(
        [
            ~cases["paired_complete"],
            chemem_success & pocket_success,
            chemem_success & ~pocket_success,
            ~chemem_success & pocket_success,
        ],
        ["not_paired", "both_success", "chemem_only", "pocket_only"],
        default="both_failure",
    )

    rates = pd.DataFrame(build_rate_rows(cases, args.deployment_only))
    comparisons = pd.DataFrame(
        build_paired_rows(
            cases, args.bootstrap_repeats, args.seed, args.deployment_only
        )
    )
    resolution = pd.DataFrame(build_resolution_rows(cases, args.deployment_only))
    completion = (
        cases.groupby(["selection_phase", "resolution_stratum"], dropna=False)
        .agg(
            frozen_n=("complex_name", "size"),
            chemem_complete_n=("chemem_complete", "sum"),
            pocket_complete_n=("pocket_complete", "sum"),
            paired_complete_n=("paired_complete", "sum"),
        )
        .reset_index()
    )
    completion["chemem_completion_pct"] = (
        100.0 * completion["chemem_complete_n"] / completion["frozen_n"]
    )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output / "per_complex_benchmark.csv", index=False)
    if pose_rows:
        pose_table = pd.DataFrame(pose_rows).sort_values(["complex_name", "rank"])
    else:
        pose_table = pd.DataFrame(columns=[
            "complex_name", "rank", "iteration", "chemem_score", "rmsd_A",
            "prediction", "reference_ligand",
        ])
    pose_table.to_csv(output / "new_chemem_per_pose.csv", index=False)
    rates.to_csv(output / "success_rates.csv", index=False)
    comparisons.to_csv(output / "paired_threshold_comparisons.csv", index=False)
    resolution.to_csv(output / "resolution_stratified_metrics.csv", index=False)
    completion.to_csv(output / "completion_audit.csv", index=False)

    cohort_ids = set(cases["complex_name"])
    summary = {
        "frozen_cohort": str(cohort_path),
        "frozen_cohort_n": len(cases),
        "historical_complete_n": int(sum(
            result.get("chemem_source_status") == "complete"
            for result in old_results.values()
            if result["complex_name"] in cohort_ids
        )),
        "new_work_cases_n": len(new_cases),
        "new_outputs_complete_n": int(sum(
            result.get("chemem_source_status") == "complete"
            for result in evaluated_new.values()
        )),
        "combined_chemem_complete_n": int(cases["chemem_complete"].sum()),
        "combined_chemem_failed_or_pending_n": int((~cases["chemem_complete"]).sum()),
        "pocket_complete_n": int(cases["pocket_complete"].sum()),
        "paired_complete_n": int(cases["paired_complete"].sum()),
        "new_results_override_historical_failures": True,
        "frozen_attempted_missing_counted_as_failure": True,
        "paired_complete_missing_excluded": True,
        "chemem_top1_definition": "first ranked entry in fitting/results.txt",
        "chemem_oracle_definition": "minimum RMSD among all evaluable ranked ChemEM poses",
        "pocket_top1_definition": "frozen S0 selected pose",
        "deployment_only": args.deployment_only,
        "oracle_endpoints_reported": not args.deployment_only,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nPAIRED-COMPLETE PERFORMANCE")
    display = rates[
        rates["analysis_set"].eq("paired_complete")
        & rates["cutoff_A"].isin([2.0, 5.0])
    ][[
        "method", "endpoint", "cutoff_A", "success_count", "denominator",
        "success_pct", "median_rmsd_A_among_scored",
    ]]
    print(display.to_string(index=False))
    print("\nTOP-1 PAIRED COMPARISON AT 2 A")
    print(comparisons[
        comparisons["comparison"].str.contains("S0_vs")
        & comparisons["cutoff_A"].eq(2.0)
    ].to_string(index=False))
    print(f"CHEMEM_POCKET_BENCHMARK_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
