#!/usr/bin/env python3
"""Create a synthetic ligand-density crop for engineering smoke tests only.

This utility renders Gaussian density from a reference ligand and therefore
contains ground-truth coordinate leakage. Its output must never be used for a
benchmark, validation result, hyperparameter choice, or paper table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem


def read_heavy_coordinates(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format for smoke crop: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand coordinates: {path}")
    conformer = molecule.GetConformer()
    coordinates = []
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        point = conformer.GetAtomPosition(atom.GetIdx())
        coordinates.append((point.x, point.y, point.z))
    if not coordinates:
        raise ValueError(f"Ligand contains no heavy atoms: {path}")
    return np.asarray(coordinates, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spacing", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=5.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    if args.spacing <= 0 or args.margin <= 0 or args.sigma <= 0:
        raise ValueError("spacing, margin, and sigma must be positive")

    coordinates = read_heavy_coordinates(args.ligand.resolve())
    origin = coordinates.min(axis=0) - float(args.margin)
    upper = coordinates.max(axis=0) + float(args.margin)
    shape = np.ceil((upper - origin) / float(args.spacing)).astype(int) + 1
    axes = [origin[i] + np.arange(shape[i], dtype=np.float32) * args.spacing for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    density = np.zeros(tuple(shape), dtype=np.float32)
    denominator = 2.0 * float(args.sigma) ** 2
    for coordinate in coordinates:
        squared_distance = np.square(grid - coordinate).sum(axis=-1)
        density += np.exp(-squared_distance / denominator).astype(np.float32)

    norm_center = float(np.median(density))
    norm_scale = float(np.std(density))
    if norm_scale <= 1e-6:
        norm_scale = 1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        density=density,
        origin=origin.astype(np.float32),
        spacing=np.repeat(np.float32(args.spacing), 3),
        norm_center=np.float32(norm_center),
        norm_scale=np.float32(norm_scale),
        center=coordinates.mean(axis=0).astype(np.float32),
        box_size=np.float32((upper - origin).max()),
        synthetic_smoke_only=np.asarray(True),
        source_ligand=np.asarray(str(args.ligand.resolve())),
    )
    print(
        f"SYNTHETIC_SMOKE_CROP_ONLY: {args.output.resolve()} "
        f"shape={tuple(shape)} spacing={args.spacing}"
    )


if __name__ == "__main__":
    main()

