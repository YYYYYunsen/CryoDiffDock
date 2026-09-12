#!/usr/bin/env python3
"""Resample a deployment-available MRC/CCP4 map into a guidance crop.

The output is an axis-aligned Cartesian ``.npz`` consumed by
``--ligand_density_crop``.  The crop center must come from the predefined
pocket or a density-only detector; this script never reads reference ligand
coordinates.

For density-guided docking, ``--map`` should contain segmented ligand density
or protein-subtracted residual density. A center derived from deposited ligand
coordinates must be labeled as an oracle-pocket/local-docking experiment; it
is not an end-to-end pocket-finding result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_preparation import cartesian_axes, normalization, sample_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="Segmented or residual MRC/CCP4 map")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--center-x", required=True, type=float)
    parser.add_argument("--center-y", required=True, type=float)
    parser.add_argument("--center-z", required=True, type=float)
    parser.add_argument("--box-size", type=float, default=24.0, help="Cartesian cube side in Angstrom")
    parser.add_argument("--spacing", type=float, default=0.75, help="Output grid spacing in Angstrom")
    args = parser.parse_args()
    if args.box_size <= 0 or args.spacing <= 0:
        raise ValueError("box-size and spacing must be positive")
    if not args.map.is_file():
        raise FileNotFoundError(args.map)

    center = np.asarray([args.center_x, args.center_y, args.center_z], dtype=np.float32)
    origin, axes = cartesian_axes(center, args.box_size, args.spacing)
    sampled = sample_map(args.map.resolve(), axes)
    density, norm_center, norm_scale = normalization(sampled)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        density=density.astype(np.float32),
        origin=origin,
        spacing=np.repeat(np.float32(args.spacing), 3),
        norm_center=np.float32(norm_center),
        norm_scale=np.float32(norm_scale),
        center=center,
        box_size=np.float32(args.box_size),
        source_map=np.asarray(str(args.map.resolve())),
        reference_ligand_coordinates_used=np.asarray(False),
    )
    finite_fraction = float(np.isfinite(sampled).mean())
    print(
        f"DENSITY_GUIDANCE_CROP_OK: {args.output.resolve()} "
        f"shape={density.shape} spacing={args.spacing:.3f} "
        f"finite_fraction={finite_fraction:.4f}"
    )


if __name__ == "__main__":
    main()
