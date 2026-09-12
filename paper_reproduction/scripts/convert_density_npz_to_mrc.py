#!/usr/bin/env python3
"""Convert one CryoDiffDock Cartesian density crop from NPZ to MRC.

CryoDiffDock crops store density as ``density[X, Y, Z]`` with absolute
Cartesian ``origin[X, Y, Z]`` and ``spacing[X, Y, Z]`` in Angstroms.  The
``mrcfile`` array convention is ``data[section, row, column]`` or ``[Z, Y, X]``.
The transpose below is therefore required to preserve the physical frame in
ChimeraX and other MRC readers.

The conversion does not renormalize, threshold, mask, or otherwise change the
density values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np


def load_crop(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"density", "origin", "spacing"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        density_xyz = np.asarray(archive["density"], dtype=np.float32)
        origin_xyz = np.asarray(archive["origin"], dtype=np.float32).reshape(3)
        spacing_xyz = np.asarray(archive["spacing"], dtype=np.float32)
        if spacing_xyz.ndim == 0:
            spacing_xyz = np.repeat(spacing_xyz, 3)
        spacing_xyz = spacing_xyz.reshape(3)

    if density_xyz.ndim != 3 or min(density_xyz.shape) < 2:
        raise ValueError(f"density must be a 3D array, got {density_xyz.shape}")
    if not np.isfinite(density_xyz).all():
        raise ValueError("density contains non-finite values")
    if not np.isfinite(origin_xyz).all():
        raise ValueError(f"origin contains non-finite values: {origin_xyz}")
    if not np.isfinite(spacing_xyz).all() or np.any(spacing_xyz <= 0):
        raise ValueError(f"spacing must be finite and positive: {spacing_xyz}")
    return density_xyz, origin_xyz, spacing_xyz


def write_mrc(
    output: Path,
    density_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
    overwrite: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    density_zyx = np.ascontiguousarray(
        np.transpose(density_xyz, (2, 1, 0)), dtype=np.float32
    )
    with mrcfile.new(output, overwrite=overwrite) as mrc:
        mrc.set_data(density_zyx)
        mrc.voxel_size = tuple(float(value) for value in spacing_xyz)
        mrc.header.origin.x = float(origin_xyz[0])
        mrc.header.origin.y = float(origin_xyz[1])
        mrc.header.origin.z = float(origin_xyz[2])
        mrc.header.mapc = 1
        mrc.header.mapr = 2
        mrc.header.maps = 3
        mrc.header.ispg = 1
        mrc.update_header_stats()
        mrc.flush()


def verify_roundtrip(
    output: Path,
    density_xyz: np.ndarray,
    origin_xyz: np.ndarray,
    spacing_xyz: np.ndarray,
) -> None:
    with mrcfile.open(output, permissive=False) as mrc:
        recovered_xyz = np.transpose(np.asarray(mrc.data), (2, 1, 0))
        recovered_origin = np.asarray(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z],
            dtype=np.float32,
        )
        recovered_spacing = np.asarray(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z],
            dtype=np.float32,
        )
        if recovered_xyz.shape != density_xyz.shape:
            raise RuntimeError(
                f"round-trip shape mismatch: {recovered_xyz.shape} != {density_xyz.shape}"
            )
        if not np.array_equal(recovered_xyz, density_xyz):
            raise RuntimeError("round-trip density values are not identical")
        if not np.allclose(recovered_origin, origin_xyz, atol=1e-5, rtol=0):
            raise RuntimeError(
                f"round-trip origin mismatch: {recovered_origin} != {origin_xyz}"
            )
        if not np.allclose(recovered_spacing, spacing_xyz, atol=1e-6, rtol=0):
            raise RuntimeError(
                f"round-trip spacing mismatch: {recovered_spacing} != {spacing_xyz}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CryoDiffDock density-crop NPZ")
    parser.add_argument("output", type=Path, help="Output MRC path")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file"
    )
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {destination}; pass --overwrite to replace it"
        )

    density, origin, spacing = load_crop(source)
    write_mrc(destination, density, origin, spacing, args.overwrite)
    verify_roundtrip(destination, density, origin, spacing)

    upper = origin + (np.asarray(density.shape, dtype=np.float32) - 1) * spacing
    center = (origin + upper) / 2
    print(f"input:   {source}")
    print(f"output:  {destination}")
    print(f"shape XYZ: {tuple(int(value) for value in density.shape)}")
    print(f"spacing A: {spacing.tolist()}")
    print(f"origin A:  {origin.tolist()}")
    print(f"center A:  {center.tolist()}")
    print("round-trip verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
