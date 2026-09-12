#!/usr/bin/env python3
"""Evaluate paired ligand/side-chain candidate coverage from existing results.

The input manifest must contain ``complex_name``, ``experimental_protein``,
``reference_ligand`` and ``flexible_sidechains``.  If ``reference_protein`` or
``input_protein`` columns are absent, ``experimental_protein`` is used.

This distinction matters: when the input protein is also the deposited
reference, its side-chain RMSD is already approximately zero.  Such a dataset
can measure retention and joint candidate coverage, but it cannot establish
improvement over the input conformation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.evaluation import compare_sidechains  # noqa: E402
from evaluation.evaluate_ligand_accuracy import (  # noqa: E402
    locate_result_directory,
    ranked_predictions,
    read_heavy_molecule,
    resolve_path,
    symmetry_rmsd,
)


REQUIRED_COLUMNS = {
    "complex_name",
    "experimental_protein",
    "reference_ligand",
    "flexible_sidechains",
}


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _path_from_row(row: pd.Series, column: str, fallback: str, base: Path) -> Path:
    value = row[column] if column in row and _present(row[column]) else row[fallback]
    return resolve_path(value, base)


def paired_protein_path(ligand_path: Path, rank: int) -> Path:
    """Locate the protein PDB paired with one confidence-bearing ligand SDF."""
    exact = ligand_path.with_name(f"{ligand_path.stem}_protein.pdb")
    if exact.is_file():
        return exact
    if rank == 1:
        alias = ligand_path.with_name("rank1_protein.pdb")
        if alias.is_file():
            return alias
    raise FileNotFoundError(f"No paired protein PDB for {ligand_path}")


def _finite_values(values) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _safe_median(values) -> Optional[float]:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite else None


def _safe_max(values) -> Optional[float]:
    finite = _finite_values(values)
    return float(np.max(finite)) if finite else None


def summarize_complex(
    candidates: list[dict[str, object]],
    input_metrics: dict[str, object],
    *,
    sidechain_threshold_A: float,
    ligand_threshold_A: float,
    reference_identity_tolerance_A: float,
) -> dict[str, object]:
    if not candidates:
        raise ValueError("At least one candidate is required")
    candidates = sorted(candidates, key=lambda row: int(row["rank"]))
    top = candidates[0]
    ligand_best = min(candidates, key=lambda row: float(row["ligand_rmsd_A"]))
    sidechain_best = min(candidates, key=lambda row: float(row["sidechain_rmsd_A"]))
    joint = [
        row for row in candidates
        if float(row["ligand_rmsd_A"]) < ligand_threshold_A
        and float(row["sidechain_rmsd_A"]) < sidechain_threshold_A
    ]
    joint_best = min(
        joint,
        key=lambda row: (float(row["ligand_rmsd_A"]), float(row["sidechain_rmsd_A"])),
    ) if joint else None
    input_rmsd = float(input_metrics["sidechain_rmsd_A"])
    sidechain_best_rmsd = float(sidechain_best["sidechain_rmsd_A"])
    input_reference_like = input_rmsd <= reference_identity_tolerance_A
    return {
        "num_predictions": len(candidates),
        "num_evaluated_residues": int(input_metrics["num_evaluated_residues"]),
        "input_sidechain_rmsd_A": input_rmsd,
        "input_mean_chi_error_deg": input_metrics["mean_chi_error_deg"],
        "input_chi_accuracy_20deg": input_metrics["chi_accuracy_20deg"],
        "input_reference_like": bool(input_reference_like),
        "improvement_gate_applicable": not input_reference_like,
        "confidence_top1_rank": int(top["rank"]),
        "confidence_top1_ligand_rmsd_A": float(top["ligand_rmsd_A"]),
        "confidence_top1_sidechain_rmsd_A": float(top["sidechain_rmsd_A"]),
        "confidence_top1_chi_accuracy_20deg": top["chi_accuracy_20deg"],
        "ligand_best_rank": int(ligand_best["rank"]),
        "ligand_best_rmsd_A": float(ligand_best["ligand_rmsd_A"]),
        "ligand_best_sidechain_rmsd_A": float(ligand_best["sidechain_rmsd_A"]),
        "sidechain_best_rank": int(sidechain_best["rank"]),
        "sidechain_best_rmsd_A": sidechain_best_rmsd,
        "sidechain_best_ligand_rmsd_A": float(sidechain_best["ligand_rmsd_A"]),
        "sidechain_best_chi_accuracy_20deg": sidechain_best["chi_accuracy_20deg"],
        "best_of_n_chi_accuracy_20deg": _safe_max(
            row["chi_accuracy_20deg"] for row in candidates
        ),
        "sidechain_oracle_delta_vs_input_A": sidechain_best_rmsd - input_rmsd,
        "sidechain_oracle_improved_vs_input": bool(sidechain_best_rmsd < input_rmsd),
        "sidechain_retained_lt_threshold": bool(
            sidechain_best_rmsd < sidechain_threshold_A
        ),
        "joint_coverage": bool(joint),
        "num_joint_candidates": len(joint),
        "joint_best_rank": int(joint_best["rank"]) if joint_best else None,
        "joint_best_ligand_rmsd_A": (
            float(joint_best["ligand_rmsd_A"]) if joint_best else None
        ),
        "joint_best_sidechain_rmsd_A": (
            float(joint_best["sidechain_rmsd_A"]) if joint_best else None
        ),
    }


def summarize_dataset(
    per_complex: pd.DataFrame,
    *,
    sidechain_threshold_A: float,
    ligand_threshold_A: float,
    reference_identity_tolerance_A: float,
    num_failures: int,
) -> dict[str, object]:
    applicable = per_complex[per_complex["improvement_gate_applicable"].astype(bool)]
    median_delta = _safe_median(applicable["sidechain_oracle_delta_vs_input_A"])
    if len(applicable) == 0:
        improvement_gate = "NOT_APPLICABLE_INPUT_IS_REFERENCE"
    else:
        improvement_gate = "GO" if median_delta is not None and median_delta < 0 else "NO_GO"
    return {
        "num_complexes": int(len(per_complex)),
        "num_failures": int(num_failures),
        "num_improvement_gate_applicable": int(len(applicable)),
        "improvement_gate": improvement_gate,
        "reference_identity_tolerance_A": reference_identity_tolerance_A,
        "sidechain_threshold_A": sidechain_threshold_A,
        "ligand_threshold_A": ligand_threshold_A,
        "median_input_sidechain_rmsd_A": _safe_median(
            per_complex["input_sidechain_rmsd_A"]
        ),
        "median_confidence_top1_sidechain_rmsd_A": _safe_median(
            per_complex["confidence_top1_sidechain_rmsd_A"]
        ),
        "median_ligand_best_sidechain_rmsd_A": _safe_median(
            per_complex["ligand_best_sidechain_rmsd_A"]
        ),
        "median_sidechain_best_rmsd_A": _safe_median(
            per_complex["sidechain_best_rmsd_A"]
        ),
        "median_sidechain_oracle_delta_vs_input_A_applicable": median_delta,
        "sidechain_oracle_improved_vs_input_pct_applicable": (
            float(100.0 * applicable["sidechain_oracle_improved_vs_input"].mean())
            if len(applicable) else None
        ),
        "sidechain_retention_lt_threshold_pct": float(
            100.0 * per_complex["sidechain_retained_lt_threshold"].mean()
        ),
        "joint_coverage_pct": float(100.0 * per_complex["joint_coverage"].mean()),
        "median_num_joint_candidates": _safe_median(per_complex["num_joint_candidates"]),
        "interpretation": (
            "Current data measure retention/joint coverage only; use distinct apo/input "
            "and holo/reference proteins to claim side-chain improvement."
            if improvement_gate.startswith("NOT_APPLICABLE")
            else "Improvement gate is evaluated only on complexes whose input differs from reference."
        ),
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument("--sidechain-threshold", type=float, default=1.0)
    parser.add_argument("--ligand-threshold", type=float, default=2.0)
    parser.add_argument("--reference-identity-tolerance", type=float, default=0.05)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.num_predictions <= 0:
        raise ValueError("num-predictions must be positive")
    if min(
        args.sidechain_threshold,
        args.ligand_threshold,
        args.reference_identity_tolerance,
    ) < 0:
        raise ValueError("thresholds must be non-negative")

    manifest = args.manifest.expanduser().resolve()
    results = args.results.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(manifest)
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    candidate_rows: list[dict[str, object]] = []
    complex_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row_index, row in table.iterrows():
        name = str(row["complex_name"])
        try:
            reference_protein = _path_from_row(
                row, "reference_protein", "experimental_protein", manifest.parent
            )
            input_protein = _path_from_row(
                row, "input_protein", "experimental_protein", manifest.parent
            )
            reference_ligand_path = resolve_path(row["reference_ligand"], manifest.parent)
            residues = str(row["flexible_sidechains"]).strip()
            for path in (reference_protein, input_protein, reference_ligand_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            reference_ligand = read_heavy_molecule(reference_ligand_path)
            input_metrics = compare_sidechains(
                reference_protein, input_protein, residue_ids=residues
            )
            result_directory = locate_result_directory(results, int(row_index))
            predictions = ranked_predictions(result_directory, args.num_predictions)
            if not predictions:
                raise FileNotFoundError(f"No ranked ligand poses in {result_directory}")

            current_candidates: list[dict[str, object]] = []
            for rank, confidence, ligand_path in predictions:
                protein_path = paired_protein_path(ligand_path, rank)
                prediction_ligand = read_heavy_molecule(ligand_path)
                ligand_rmsd = symmetry_rmsd(reference_ligand, prediction_ligand)
                sidechain = compare_sidechains(
                    reference_protein, protein_path, residue_ids=residues
                )
                candidate = {
                    "row_index": int(row_index),
                    "complex_name": name,
                    "rank": int(rank),
                    "confidence": float(confidence),
                    "ligand_rmsd_A": float(ligand_rmsd),
                    "sidechain_rmsd_A": float(sidechain["sidechain_rmsd_A"]),
                    "mean_chi_error_deg": sidechain["mean_chi_error_deg"],
                    "chi_accuracy_20deg": sidechain["chi_accuracy_20deg"],
                    "chi_accuracy_40deg": sidechain["chi_accuracy_40deg"],
                    "num_evaluated_residues": sidechain["num_evaluated_residues"],
                    "num_chi_angles": sidechain["num_chi_angles"],
                    "ligand_path": str(ligand_path),
                    "protein_path": str(protein_path),
                    "reference_protein": str(reference_protein),
                    "input_protein": str(input_protein),
                    "flexible_sidechains": residues,
                }
                current_candidates.append(candidate)
                candidate_rows.append(candidate)
            summary = summarize_complex(
                current_candidates,
                input_metrics,
                sidechain_threshold_A=args.sidechain_threshold,
                ligand_threshold_A=args.ligand_threshold,
                reference_identity_tolerance_A=args.reference_identity_tolerance,
            )
            summary.update({
                "row_index": int(row_index),
                "complex_name": name,
                "reference_protein": str(reference_protein),
                "input_protein": str(input_protein),
                "reference_and_input_same_path": reference_protein == input_protein,
                "flexible_sidechains": residues,
            })
            complex_rows.append(summary)
            print(
                f"[{row_index + 1}/{len(table)}] {name} "
                f"inputSC={summary['input_sidechain_rmsd_A']:.3f} "
                f"top1SC={summary['confidence_top1_sidechain_rmsd_A']:.3f} "
                f"bestSC={summary['sidechain_best_rmsd_A']:.3f} "
                f"ligbestSC={summary['ligand_best_sidechain_rmsd_A']:.3f} "
                f"joint={int(summary['joint_coverage'])}"
            )
        except Exception as exc:
            failure = {
                "row_index": int(row_index),
                "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{row_index + 1}/{len(table)}] FAILED {name}: {exc}")
            if args.fail_on_error:
                raise

    if not complex_rows:
        raise RuntimeError("No complexes were evaluated")
    candidates_frame = pd.DataFrame(candidate_rows)
    complexes_frame = pd.DataFrame(complex_rows)
    candidates_frame.to_csv(output_dir / "per_candidate.csv", index=False)
    complexes_frame.to_csv(output_dir / "per_complex.csv", index=False)
    pd.DataFrame(failures, columns=["row_index", "complex_name", "error"]).to_csv(
        output_dir / "failures.csv", index=False
    )
    summary = summarize_dataset(
        complexes_frame,
        sidechain_threshold_A=args.sidechain_threshold,
        ligand_threshold_A=args.ligand_threshold,
        reference_identity_tolerance_A=args.reference_identity_tolerance,
        num_failures=len(failures),
    )
    summary.update({
        "manifest": str(manifest),
        "results": str(results),
        "num_predictions_requested": args.num_predictions,
        "reference_protein_rule": (
            "reference_protein column, otherwise experimental_protein"
        ),
        "input_protein_rule": "input_protein column, otherwise experimental_protein",
    })
    text = json.dumps(summary, indent=2, default=_json_default) + "\n"
    (output_dir / "summary.json").write_text(text)
    print(text, end="")
    return 1 if failures and args.fail_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
