#!/usr/bin/env python3
"""Validation-only NCC scoring of deposited ligands against prepared crops."""

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
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
    sigma_column,
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sigmas", default="1.5")
    parser.add_argument("--mask-radius", type=float, default=10.0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    sigmas = [float(item) for item in args.sigmas.split(",") if item.strip()]
    rows, failures = [], []
    for row_index, row in table.iterrows():
        try:
            crop_path = resolve_path(row["ligand_density_crop"], manifest.parent)
            reference_path = resolve_path(row["reference_ligand"], manifest.parent)
            crop = load_cartesian_density_crop(crop_path)
            mask = fixed_crop_mask(crop, radius_A=args.mask_radius)
            coordinates, atomic_numbers = read_reference(reference_path)
            result = {
                "row_index": row_index,
                "complex_name": row["complex_name"],
                "crop_path": str(crop_path),
                "reference_ligand": str(reference_path),
            }
            for sigma in sigmas:
                score = score_gaussian_density(
                    crop, coordinates, atomic_numbers, sigma, mask
                )
                result[sigma_column("ncc", sigma)] = score.ncc
                result[sigma_column("cosine", sigma)] = score.cosine_overlap
                result["ligand_in_bounds_fraction"] = score.ligand_in_bounds_fraction
            rows.append(result)
            print(f"[{row_index + 1}/{len(table)}] {row['complex_name']}: OK")
        except Exception as exc:
            failures.append({
                "row_index": row_index,
                "complex_name": row.get("complex_name", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })
            if args.fail_on_error:
                raise

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(failures).to_csv(output.with_suffix(".failures.csv"), index=False)
    summary = {
        "manifest": str(manifest),
        "num_complexes": len(rows),
        "num_failures": len(failures),
        "uses_ground_truth": True,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
