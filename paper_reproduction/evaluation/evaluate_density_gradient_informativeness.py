#!/usr/bin/env python3
"""G0.5: test whether raw density gradients locally restore a known ligand pose.

This is a validation-only diagnostic: it deliberately reads ``reference_ligand``.
For each native ligand, it applies controlled rigid translations and rotations,
computes the *unscaled* atom-density or Gaussian-NCC gradient, and takes one
fixed physical step along that gradient.  It therefore tests gradient
informativeness without DiffDock score norm matching or reverse diffusion.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_guidance import (
    gaussian_ncc,
    load_density_crop,
    rigid_density_scores,
    rigid_ncc_scores,
    trilinear_density,
)


AXES = np.eye(3, dtype=np.float32)


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_heavy_ligand(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read reference ligand: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float32
    )
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.float32
    )
    return coordinates, atomic_numbers


def parse_positive_csv(value: str, label: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label} must be a comma-separated list of positive values")
    return values


def rotation_matrix(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    x, y, z = axis
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    angle = torch.as_tensor(angle_rad, dtype=axis.dtype, device=axis.device)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def rotate_about_centroid(
    coordinates: torch.Tensor, axis: torch.Tensor, angle_rad: float
) -> torch.Tensor:
    center = coordinates.mean(dim=0, keepdim=True)
    matrix = rotation_matrix(axis, angle_rad)
    return (coordinates - center) @ matrix.T + center


def direct_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(((left - right) ** 2).sum(dim=-1).mean()).cpu())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((torch.dot(left, right) / denominator).cpu())


class Objective:
    def __init__(
        self,
        name: str,
        crop,
        atomic_numbers: np.ndarray,
        device: torch.device,
        sigma_A: float,
        mask_radius_A: float,
        chunk_size: int,
    ) -> None:
        self.name = name
        self.crop = crop
        self.device = device
        self.density = torch.as_tensor(crop.density, dtype=torch.float32, device=device)
        self.origin = torch.as_tensor(crop.origin, dtype=torch.float32, device=device)
        self.spacing = torch.as_tensor(crop.spacing, dtype=torch.float32, device=device)
        self.atomic_weights = torch.as_tensor(
            atomic_numbers, dtype=torch.float32, device=device
        )
        self.sigma_A = float(sigma_A)
        self.chunk_size = int(chunk_size)
        self.sample_points = None
        self.target_density = None
        if name == "map_ncc":
            axes = [
                self.origin[axis]
                + torch.arange(
                    self.density.shape[axis], dtype=torch.float32, device=device
                ) * self.spacing[axis]
                for axis in range(3)
            ]
            mesh = torch.meshgrid(*axes, indexing="ij")
            points = torch.stack(mesh, dim=-1).reshape(-1, 3)
            center = self.origin + 0.5 * (
                torch.as_tensor(
                    self.density.shape, dtype=torch.float32, device=device
                ) - 1.0
            ) * self.spacing
            mask = torch.linalg.vector_norm(points - center[None, :], dim=-1) \
                <= float(mask_radius_A)
            if int(mask.sum()) < 64:
                raise ValueError(f"NCC mask has only {int(mask.sum())} voxels")
            self.sample_points = points[mask]
            self.target_density = (
                (self.density - float(crop.norm_center)) / float(crop.norm_scale)
            ).reshape(-1)[mask]

    def scores(self, coordinates: torch.Tensor):
        if self.name == "atom_density":
            return rigid_density_scores(
                coordinates,
                self.density,
                self.origin,
                self.spacing,
                self.crop.norm_center,
                self.crop.norm_scale,
            )
        return rigid_ncc_scores(
            coordinates_abs=coordinates,
            atomic_weights=self.atomic_weights,
            sample_points=self.sample_points,
            target_density=self.target_density,
            sigma_A=self.sigma_A,
            crop_origin=self.origin,
            crop_spacing=self.spacing,
            crop_shape=self.density.shape,
            chunk_size=self.chunk_size,
        )

    def value(self, coordinates: torch.Tensor) -> float:
        with torch.no_grad():
            if self.name == "atom_density":
                values, _ = trilinear_density(
                    density=self.density,
                    origin=self.origin,
                    spacing=self.spacing,
                    points=coordinates,
                    fill_value=self.crop.norm_center,
                )
                value = (
                    (values - float(self.crop.norm_center))
                    / float(self.crop.norm_scale)
                ).mean()
            else:
                value = gaussian_ncc(
                    sample_points=self.sample_points,
                    coordinates=coordinates,
                    atomic_weights=self.atomic_weights,
                    target_density=self.target_density,
                    sigma_A=self.sigma_A,
                    chunk_size=self.chunk_size,
                )
        return float(value.cpu())


def summarize(
    records: pd.DataFrame, *, per_complex: bool = False
) -> list[dict[str, object]]:
    summaries = []
    keys = (["row_index", "complex_name"] if per_complex else []) + [
        "objective", "perturbation", "magnitude"
    ]
    for key, group in records.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        offset = 2 if per_complex else 0
        finite_cosine = group["restoration_cosine"].replace([np.inf, -np.inf], np.nan)
        record = {
            "objective": key[offset],
            "perturbation": key[offset + 1],
            "magnitude": float(key[offset + 2]),
            "num_trials": int(len(group)),
            "median_coordinate_gradient_norm": float(group["coordinate_gradient_norm"].median()),
            "median_translation_gradient_norm": float(group["translation_gradient_norm"].median()),
            "median_rotation_gradient_norm": float(group["rotation_gradient_norm"].median()),
            "median_restoration_cosine": float(finite_cosine.median()),
            "restorative_direction_pct": float((finite_cosine > 0).mean() * 100.0),
            "objective_improved_pct": float((group["objective_delta"] > 0).mean() * 100.0),
            "geometry_improved_pct": float((group["rmsd_delta_A"] < 0).mean() * 100.0),
            "both_improved_pct": float(
                ((group["objective_delta"] > 0) & (group["rmsd_delta_A"] < 0)).mean()
                * 100.0
            ),
            "median_objective_delta": float(group["objective_delta"].median()),
            "median_rmsd_delta_A": float(group["rmsd_delta_A"].median()),
        }
        if per_complex:
            record = {
                "row_index": int(key[0]),
                "complex_name": str(key[1]),
                **record,
            }
        summaries.append(record)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--objectives", default="atom_density,map_ncc")
    parser.add_argument("--translation-magnitudes", default="0.1,0.5,1.0")
    parser.add_argument("--rotation-degrees", default="2,10,30")
    parser.add_argument("--translation-step", type=float, default=0.1,
                        help="Fixed raw-gradient ascent step in Angstrom")
    parser.add_argument("--rotation-step-degrees", type=float, default=2.0,
                        help="Fixed raw-gradient ascent step in degrees")
    parser.add_argument("--ncc-sigma", type=float, default=1.5)
    parser.add_argument("--ncc-mask-radius", type=float, default=10.0)
    parser.add_argument("--ncc-chunk-size", type=int, default=8192)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    objectives = [item.strip() for item in args.objectives.split(",") if item.strip()]
    unknown = sorted(set(objectives) - {"atom_density", "map_ncc"})
    if not objectives or unknown:
        raise ValueError(f"Unknown objectives: {unknown}")
    translation_magnitudes = parse_positive_csv(
        args.translation_magnitudes, "--translation-magnitudes"
    )
    rotation_degrees = parse_positive_csv(args.rotation_degrees, "--rotation-degrees")
    if args.translation_step <= 0 or args.rotation_step_degrees <= 0:
        raise ValueError("Fixed step sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    required = {"complex_name", "reference_ligand", "ligand_density_crop"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if args.limit > 0:
        table = table.head(args.limit)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    positive_axes = torch.as_tensor(AXES, dtype=torch.float32, device=device)
    directions = [
        (f"{label}{sign_label}", float(sign) * axis)
        for label, axis in zip("xyz", positive_axes)
        for sign_label, sign in (("+", 1.0), ("-", -1.0))
    ]

    for position, (row_index, row) in enumerate(table.iterrows(), start=1):
        name = str(row["complex_name"])
        try:
            ligand_path = resolve_path(row["reference_ligand"], manifest.parent)
            crop_path = resolve_path(row["ligand_density_crop"], manifest.parent)
            native_np, atomic_numbers = read_heavy_ligand(ligand_path)
            native = torch.as_tensor(native_np, dtype=torch.float32, device=device)
            crop = load_density_crop(crop_path)
            evaluators = [
                Objective(
                    objective, crop, atomic_numbers, device, args.ncc_sigma,
                    args.ncc_mask_radius, args.ncc_chunk_size,
                )
                for objective in objectives
            ]
            for evaluator in evaluators:
                native_objective = evaluator.value(native)
                # Test both signs so a systematic map offset cannot
                # masquerade as a restorative gradient.
                for axis_label, axis in directions:
                    for magnitude in translation_magnitudes:
                        perturbed = native + float(magnitude) * axis[None, :]
                        tr_gradient, rot_gradient, audit = evaluator.scores(perturbed)
                        norm = torch.linalg.vector_norm(tr_gradient)
                        if float(norm) > 1e-12:
                            corrected = perturbed + float(args.translation_step) \
                                * (tr_gradient / norm)[None, :]
                        else:
                            corrected = perturbed
                        before = direct_rmsd(perturbed, native)
                        after = direct_rmsd(corrected, native)
                        corrected_objective = evaluator.value(corrected)
                        rows.append({
                            "row_index": int(row_index), "complex_name": name,
                            "objective": evaluator.name,
                            "native_objective": native_objective,
                            "perturbation": "translation_A", "axis": axis_label,
                            "magnitude": float(magnitude),
                            "fixed_step": float(args.translation_step),
                            "perturbed_objective": audit["density_objective"],
                            "corrected_objective": corrected_objective,
                            "objective_delta": corrected_objective - audit["density_objective"],
                            "coordinate_gradient_norm": audit["coordinate_gradient_norm"],
                            "translation_gradient_norm": audit["translation_gradient_norm"],
                            "rotation_gradient_norm": audit["rotation_gradient_norm"],
                            "restoration_cosine": cosine(tr_gradient, -axis),
                            "rmsd_before_A": before, "rmsd_after_A": after,
                            "rmsd_delta_A": after - before,
                            "in_bounds_fraction": audit["in_bounds_fraction"],
                        })

                    for magnitude in rotation_degrees:
                        perturbed = rotate_about_centroid(
                            native, axis, np.deg2rad(float(magnitude))
                        )
                        tr_gradient, rot_gradient, audit = evaluator.scores(perturbed)
                        norm = torch.linalg.vector_norm(rot_gradient)
                        if float(norm) > 1e-12:
                            corrected = rotate_about_centroid(
                                perturbed,
                                rot_gradient / norm,
                                np.deg2rad(float(args.rotation_step_degrees)),
                            )
                        else:
                            corrected = perturbed
                        before = direct_rmsd(perturbed, native)
                        after = direct_rmsd(corrected, native)
                        corrected_objective = evaluator.value(corrected)
                        rows.append({
                            "row_index": int(row_index), "complex_name": name,
                            "objective": evaluator.name,
                            "native_objective": native_objective,
                            "perturbation": "rotation_deg", "axis": axis_label,
                            "magnitude": float(magnitude),
                            "fixed_step": float(args.rotation_step_degrees),
                            "perturbed_objective": audit["density_objective"],
                            "corrected_objective": corrected_objective,
                            "objective_delta": corrected_objective - audit["density_objective"],
                            "coordinate_gradient_norm": audit["coordinate_gradient_norm"],
                            "translation_gradient_norm": audit["translation_gradient_norm"],
                            "rotation_gradient_norm": audit["rotation_gradient_norm"],
                            "restoration_cosine": cosine(rot_gradient, -axis),
                            "rmsd_before_A": before, "rmsd_after_A": after,
                            "rmsd_delta_A": after - before,
                            "in_bounds_fraction": audit["in_bounds_fraction"],
                        })
            print(f"[{position}/{len(table)}] {name}: {len(evaluators)} objectives OK")
        except Exception as exc:
            failures.append({
                "row_index": int(row_index), "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{position}/{len(table)}] FAILED {name}: {exc}")
            if args.fail_on_error:
                raise

    if not rows:
        raise RuntimeError("No gradient trials completed")
    trials = pd.DataFrame(rows)
    summary_rows = summarize(trials)
    summary_table = pd.DataFrame(summary_rows)
    per_complex_table = pd.DataFrame(summarize(trials, per_complex=True))
    trials.to_csv(output / "gradient_trials.csv", index=False)
    summary_table.to_csv(output / "summary.csv", index=False)
    per_complex_table.to_csv(output / "per_complex_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    report = {
        "manifest": str(manifest),
        "device": str(device),
        "num_complexes": int(trials["row_index"].nunique()),
        "num_trials": int(len(trials)),
        "num_failures": int(len(failures)),
        "objectives": objectives,
        "ncc_sigma_A": float(args.ncc_sigma),
        "ncc_mask_radius_A": float(args.ncc_mask_radius),
        "translation_step_A": float(args.translation_step),
        "rotation_step_degrees": float(args.rotation_step_degrees),
        "uses_ground_truth": True,
        "uses_diffdock_norm_matching": False,
        "summary": summary_rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary_table.to_string(index=False))
    print(json.dumps({key: value for key, value in report.items() if key != "summary"}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
