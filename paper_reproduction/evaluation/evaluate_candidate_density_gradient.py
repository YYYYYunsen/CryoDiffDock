#!/usr/bin/env python3
"""G0.5B: test one raw NCC correction on generated ligand candidates.

This validation-only diagnostic reads deposited reference ligands. It does not
rerun diffusion and it does not use score norm matching. Each existing pose is
corrected by a fixed physical translation, rotation, or their combination,
then evaluated for density and ground-truth geometry changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_density_gradient_informativeness import (
    Objective,
    resolve_path,
    rotate_about_centroid,
)
from evaluation.evaluate_ligand_accuracy import (
    centroid_distance,
    locate_result_directory,
    ranked_predictions,
    read_heavy_molecule,
    symmetry_rmsd,
)
from cryodiffdock_pocket.density_guidance import load_density_crop


def molecule_coordinates(molecule) -> np.ndarray:
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)


def atomic_numbers(molecule) -> np.ndarray:
    return np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.float32
    )


def molecule_with_coordinates(molecule, coordinates: np.ndarray):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(
            f"Coordinate shape {coordinates.shape} does not match "
            f"{molecule.GetNumAtoms()} ligand atoms"
        )
    updated = Chem.Mol(molecule)
    conformer = updated.GetConformer()
    for index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(
            index, Point3D(float(coordinate[0]), float(coordinate[1]), float(coordinate[2]))
        )
    return updated


def geometry_metrics(reference, prediction) -> dict[str, float]:
    return {
        "ligand_rmsd_A": symmetry_rmsd(reference, prediction),
        "centroid_distance_A": centroid_distance(reference, prediction),
        "centered_rmsd_A": symmetry_rmsd(reference, prediction, center=True),
        "rigid_aligned_rmsd_A": symmetry_rmsd(
            reference, prediction, center=True, minimize=True
        ),
    }


def normalized(vector: torch.Tensor) -> torch.Tensor | None:
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= 1e-12:
        return None
    return vector / norm


def correction_coordinates(
    coordinates: torch.Tensor,
    translation_gradient: torch.Tensor,
    rotation_gradient: torch.Tensor,
    translation_step_A: float,
    rotation_step_degrees: float,
) -> dict[str, torch.Tensor]:
    translation_direction = normalized(translation_gradient)
    rotation_direction = normalized(rotation_gradient)

    translated = coordinates.clone()
    if translation_direction is not None:
        translated = translated + float(translation_step_A) * translation_direction[None, :]

    rotated = coordinates.clone()
    if rotation_direction is not None:
        rotated = rotate_about_centroid(
            rotated, rotation_direction, np.deg2rad(float(rotation_step_degrees))
        )

    joint = translated.clone()
    if rotation_direction is not None:
        joint = rotate_about_centroid(
            joint, rotation_direction, np.deg2rad(float(rotation_step_degrees))
        )
    return {"translation": translated, "rotation": rotated, "joint": joint}


def rmsd_bin(value: float) -> str:
    if value < 2.0:
        return "lt2"
    if value < 5.0:
        return "2to5"
    return "ge5"


def summarize_group(group: pd.DataFrame) -> dict[str, float | int]:
    tolerance = 1e-7
    ncc_improved = group["delta_ncc"] > tolerance
    rmsd_improved = group["delta_ligand_rmsd_A"] < -tolerance
    centroid_improved = group["delta_centroid_distance_A"] < -tolerance
    centered_improved = group["delta_centered_rmsd_A"] < -tolerance
    return {
        "num_poses": int(len(group)),
        "ncc_improved_pct": float(ncc_improved.mean() * 100.0),
        "rmsd_improved_pct": float(rmsd_improved.mean() * 100.0),
        "ncc_and_rmsd_improved_pct": float((ncc_improved & rmsd_improved).mean() * 100.0),
        "centroid_improved_pct": float(centroid_improved.mean() * 100.0),
        "centered_rmsd_improved_pct": float(centered_improved.mean() * 100.0),
        "median_delta_ncc": float(group["delta_ncc"].median()),
        "median_delta_ligand_rmsd_A": float(group["delta_ligand_rmsd_A"].median()),
        "median_delta_centroid_distance_A": float(
            group["delta_centroid_distance_A"].median()
        ),
        "median_delta_centered_rmsd_A": float(group["delta_centered_rmsd_A"].median()),
        "median_delta_rigid_aligned_rmsd_A": float(
            group["delta_rigid_aligned_rmsd_A"].median()
        ),
    }


def grouped_summary(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in table.groupby(keys, sort=False, observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({
            **{column: value for column, value in zip(keys, key)},
            **summarize_group(group),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument("--limit-complexes", type=int, default=5)
    parser.add_argument("--translation-step", type=float, default=0.02)
    parser.add_argument("--rotation-step-degrees", type=float, default=0.5)
    parser.add_argument("--ncc-sigma", type=float, default=1.5)
    parser.add_argument("--ncc-mask-radius", type=float, default=10.0)
    parser.add_argument("--ncc-chunk-size", type=int, default=8192)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.num_predictions <= 0:
        raise ValueError("--num-predictions must be positive")
    if args.translation_step <= 0 or args.rotation_step_degrees <= 0:
        raise ValueError("Correction step sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    manifest = args.manifest.expanduser().resolve()
    candidate_root = args.candidate_root.expanduser().resolve()
    table = pd.read_csv(manifest)
    required = {"complex_name", "reference_ligand", "ligand_density_crop"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if args.limit_complexes > 0:
        table = table.head(args.limit_complexes)
    if not candidate_root.is_dir():
        raise FileNotFoundError(candidate_root)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for position, (row_index, row) in enumerate(table.iterrows(), start=1):
        name = str(row["complex_name"])
        try:
            reference_path = resolve_path(row["reference_ligand"], manifest.parent)
            crop_path = resolve_path(row["ligand_density_crop"], manifest.parent)
            reference = read_heavy_molecule(reference_path)
            crop = load_density_crop(crop_path)
            result_directory = locate_result_directory(candidate_root, int(row_index))
            predictions = ranked_predictions(result_directory, args.num_predictions)
            if not predictions:
                raise FileNotFoundError(f"No ranked ligand poses under {result_directory}")

            evaluator = None
            expected_atomic_numbers = None
            for rank, confidence, prediction_path in predictions:
                prediction = read_heavy_molecule(prediction_path)
                prediction_atomic_numbers = atomic_numbers(prediction)
                if evaluator is None:
                    expected_atomic_numbers = prediction_atomic_numbers
                    evaluator = Objective(
                        "map_ncc", crop, prediction_atomic_numbers, device,
                        args.ncc_sigma, args.ncc_mask_radius, args.ncc_chunk_size,
                    )
                elif not np.array_equal(prediction_atomic_numbers, expected_atomic_numbers):
                    raise ValueError(
                        f"Candidate atom ordering/composition differs at {prediction_path}"
                    )

                coordinates = torch.as_tensor(
                    molecule_coordinates(prediction), dtype=torch.float32, device=device
                )
                before_metrics = geometry_metrics(reference, prediction)
                translation_gradient, rotation_gradient, audit = evaluator.scores(coordinates)
                corrected_sets = correction_coordinates(
                    coordinates, translation_gradient, rotation_gradient,
                    args.translation_step, args.rotation_step_degrees,
                )
                for correction, corrected_coordinates in corrected_sets.items():
                    corrected_np = corrected_coordinates.detach().cpu().numpy()
                    corrected_molecule = molecule_with_coordinates(prediction, corrected_np)
                    after_metrics = geometry_metrics(reference, corrected_molecule)
                    corrected_ncc = evaluator.value(corrected_coordinates)
                    record = {
                        "row_index": int(row_index),
                        "complex_name": name,
                        "rank": int(rank),
                        "confidence": float(confidence),
                        "correction": correction,
                        "initial_rmsd_bin": rmsd_bin(before_metrics["ligand_rmsd_A"]),
                        "prediction_path": str(prediction_path.resolve()),
                        "crop_path": str(crop_path),
                        "ncc_before": float(audit["density_objective"]),
                        "ncc_after": corrected_ncc,
                        "delta_ncc": corrected_ncc - float(audit["density_objective"]),
                        "coordinate_gradient_norm": audit["coordinate_gradient_norm"],
                        "translation_gradient_norm": audit["translation_gradient_norm"],
                        "rotation_gradient_norm": audit["rotation_gradient_norm"],
                        "in_bounds_fraction": audit["in_bounds_fraction"],
                    }
                    for metric, before_value in before_metrics.items():
                        record[f"before_{metric}"] = before_value
                        record[f"after_{metric}"] = after_metrics[metric]
                        record[f"delta_{metric}"] = after_metrics[metric] - before_value
                    pose_rows.append(record)
            print(f"[{position}/{len(table)}] {name}: {len(predictions)} poses OK")
        except Exception as exc:
            failures.append({
                "row_index": int(row_index), "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{position}/{len(table)}] FAILED {name}: {exc}")
            if args.fail_on_error:
                raise

    if not pose_rows:
        raise RuntimeError("No candidate gradient trials completed")
    poses = pd.DataFrame(pose_rows)
    by_bin = grouped_summary(poses, ["correction", "initial_rmsd_bin"])
    by_complex = grouped_summary(poses, ["complex_name", "correction"])
    overall = grouped_summary(poses, ["correction"])
    poses.to_csv(output / "pose_trials.csv", index=False)
    by_bin.to_csv(output / "summary_by_rmsd_bin.csv", index=False)
    by_complex.to_csv(output / "per_complex_summary.csv", index=False)
    overall.to_csv(output / "summary_overall.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)

    report = {
        "manifest": str(manifest),
        "candidate_root": str(candidate_root),
        "device": str(device),
        "num_complexes": int(poses["row_index"].nunique()),
        "num_input_poses": int(poses[["row_index", "rank"]].drop_duplicates().shape[0]),
        "num_corrected_trials": int(len(poses)),
        "num_failures": int(len(failures)),
        "translation_step_A": float(args.translation_step),
        "rotation_step_degrees": float(args.rotation_step_degrees),
        "ncc_sigma_A": float(args.ncc_sigma),
        "ncc_mask_radius_A": float(args.ncc_mask_radius),
        "uses_ground_truth": True,
        "uses_diffdock_norm_matching": False,
        "reruns_diffusion": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nOverall")
    print(overall.to_string(index=False))
    print("\nBy initial RMSD bin")
    print(by_bin.to_string(index=False))
    print("\n" + json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
