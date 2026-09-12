#!/usr/bin/env python3
"""Validation-only coverage audit for pocket-centred density segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import (
    CartesianDensityCrop,
    fixed_crop_mask,
    render_gaussian_ligand_density,
)


def resolve_path(value, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def read_reference(path: Path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read reference ligand: {path}")
    coordinates, atomic_numbers = [], []
    conformer = molecule.GetConformer()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append((point.x, point.y, point.z))
        atomic_numbers.append(atom.GetAtomicNum())
    return np.asarray(coordinates, dtype=np.float32), np.asarray(atomic_numbers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gaussian-sigma", type=float, default=1.5)
    parser.add_argument("--ncc-mask-radius", type=float, default=10.0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    rows, failures = [], []
    for row_index, row in table.iterrows():
        try:
            crop_path = resolve_path(row["ligand_density_crop"], manifest.parent)
            reference_path = resolve_path(row["reference_ligand"], manifest.parent)
            coordinates, atomic_numbers = read_reference(reference_path)
            with np.load(crop_path, allow_pickle=False) as archive:
                density = np.asarray(archive["density"], dtype=np.float32)
                origin = np.asarray(archive["origin"], dtype=np.float32).reshape(3)
                spacing = np.asarray(archive["spacing"], dtype=np.float32)
                if spacing.ndim == 0:
                    spacing = np.repeat(float(spacing), 3)
                spacing = spacing.reshape(3)
                support = np.asarray(
                    archive["density_component_support"], dtype=bool
                )
                status = str(np.asarray(archive["density_component_status"]).reshape(()))
                norm_center = float(np.asarray(archive["norm_center"]).reshape(()))
                norm_scale = float(np.asarray(archive["norm_scale"]).reshape(()))
            if support.shape != density.shape:
                raise ValueError("component support and density shapes differ")

            crop = CartesianDensityCrop(
                density, origin, spacing, norm_center, norm_scale, str(crop_path)
            )
            ncc_mask = fixed_crop_mask(crop, radius_A=args.ncc_mask_radius)
            indices = np.rint((coordinates - origin) / spacing).astype(int)
            valid = ((indices >= 0) & (indices < np.asarray(density.shape))).all(axis=1)
            atom_supported = np.zeros(len(coordinates), dtype=bool)
            if valid.any():
                selected = indices[valid]
                atom_supported[valid] = support[
                    selected[:, 0], selected[:, 1], selected[:, 2]
                ]

            gaussian = render_gaussian_ligand_density(
                crop, coordinates, atomic_numbers, sigma_A=args.gaussian_sigma
            )
            gaussian_total = float(gaussian.sum())
            gaussian_retained = (
                float((gaussian * support).sum()) / gaussian_total
                if gaussian_total > 1e-12 else float("nan")
            )
            rows.append({
                "row_index": row_index,
                "complex_name": row["complex_name"],
                "status": status,
                "fallback": status.startswith("fallback_"),
                "native_heavy_atom_support_fraction": float(atom_supported.mean()),
                "native_gaussian_support_fraction": gaussian_retained,
                "support_fraction_of_ncc_mask": float(
                    (support & ncc_mask).sum() / ncc_mask.sum()
                ),
            })
        except Exception as exc:
            failures.append({
                "row_index": row_index, "complex_name": row.get("complex_name", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })
            if args.fail_on_error:
                raise

    result = pd.DataFrame(rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "per_complex.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    summary = {
        "num_complexes": len(result),
        "num_failures": len(failures),
        "fallback_pct": float(100 * result["fallback"].mean()),
        "median_native_heavy_atom_support_fraction": float(
            result["native_heavy_atom_support_fraction"].median()
        ),
        "p10_native_heavy_atom_support_fraction": float(
            result["native_heavy_atom_support_fraction"].quantile(0.1)
        ),
        "median_native_gaussian_support_fraction": float(
            result["native_gaussian_support_fraction"].median()
        ),
        "p10_native_gaussian_support_fraction": float(
            result["native_gaussian_support_fraction"].quantile(0.1)
        ),
        "median_support_fraction_of_ncc_mask": float(
            result["support_fraction_of_ncc_mask"].median()
        ),
        "uses_ground_truth": True,
        "ground_truth_use": "coverage_audit_only_not_segmentation",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
