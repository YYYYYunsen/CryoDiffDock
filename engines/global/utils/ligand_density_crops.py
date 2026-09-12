"""Utilities for ligand-centered cryo-EM density crop features.

The crop format used here is intentionally simple: a Cartesian, regularly
spaced density box saved as ``.npz`` with ``density``, ``origin``, ``spacing``,
``norm_center`` and ``norm_scale``.  Coordinates passed to these helpers are
absolute PDB/map coordinates in Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


_AXIS_DIRECTIONS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)


def fibonacci_sphere(n_sphere: int = 26) -> np.ndarray:
    """Approximately uniform unit directions on a sphere."""
    n_sphere = int(n_sphere)
    if n_sphere <= 0:
        raise ValueError(f"n_sphere must be positive, got {n_sphere}")
    if n_sphere == 1:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    i = np.arange(n_sphere, dtype=np.float32)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / float(n_sphere)
    radius = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = golden_angle * i
    dirs = np.stack([np.cos(theta) * radius, np.sin(theta) * radius, z], axis=1)
    return dirs.astype(np.float32)


@dataclass(frozen=True)
class LigandDensityCrop:
    density: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    norm_center: float
    norm_scale: float
    center: np.ndarray | None = None
    box_size: float | None = None
    pdb_id: str | None = None
    source_map_path: str | None = None


def _safe_scale(value: float, fallback: float = 1.0) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 1e-6:
        return float(fallback)
    return value


def atom_feature_dim(shell_radii: Sequence[float]) -> int:
    """Per-atom feature dimension: center(1) + shells(4*n) + contrast(1)."""
    return 2 + 4 * len(tuple(shell_radii))


def summary_feature_dim() -> int:
    """Per-pose summary feature dimension."""
    return 6


def find_ligand_density_crop_file(pdb_id: str, crop_dir: str | Path) -> Path | None:
    """Find a ligand-density crop by the common project naming conventions."""
    root = Path(crop_dir)
    pdb_id = pdb_id.upper()
    candidates = [
        root / pdb_id / f"{pdb_id}_ligand_density_crop.npz",
        root / pdb_id / "ligand_density_crop.npz",
        root / f"{pdb_id}_ligand_density_crop.npz",
        root / f"{pdb_id}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@lru_cache(maxsize=256)
def load_ligand_density_crop(path: str | Path) -> LigandDensityCrop:
    """Load one ligand density crop from disk."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        density = np.asarray(data["density"], dtype=np.float32)
        origin = np.asarray(data["origin"], dtype=np.float32).reshape(3)
        spacing = np.asarray(data["spacing"], dtype=np.float32)
        if spacing.ndim == 0:
            spacing = np.repeat(float(spacing), 3).astype(np.float32)
        spacing = spacing.reshape(3)
        center = np.asarray(data["center"], dtype=np.float32).reshape(3) if "center" in data else None
        box_size = float(np.asarray(data["box_size"]).reshape(())) if "box_size" in data else None
        pdb_id = str(np.asarray(data["pdb_id"]).reshape(())) if "pdb_id" in data else None
        source_map_path = (
            str(np.asarray(data["source_map_path"]).reshape(())) if "source_map_path" in data else None
        )
        norm_center = float(np.asarray(data["norm_center"]).reshape(()))
        norm_scale = _safe_scale(float(np.asarray(data["norm_scale"]).reshape(())))

    if density.ndim != 3:
        raise ValueError(f"Expected 3D density crop in {path}, got shape {density.shape}")
    return LigandDensityCrop(
        density=density,
        origin=origin,
        spacing=spacing,
        norm_center=norm_center,
        norm_scale=norm_scale,
        center=center,
        box_size=box_size,
        pdb_id=pdb_id,
        source_map_path=source_map_path,
    )


def interp_crop_points(crop: LigandDensityCrop, xyz: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Tri-linear interpolation in a Cartesian density crop.

    Args:
        crop: Loaded ligand-density crop.
        xyz: Absolute coordinates with shape ``(N, 3)``.
        fill_value: Value returned for points outside the crop.
    """
    coords = np.asarray(xyz, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected xyz with shape (N,3), got {coords.shape}")

    density = crop.density
    grid_coords = (coords - crop.origin[None, :]) / crop.spacing[None, :]
    nx, ny, nz = density.shape
    valid = (
        (grid_coords[:, 0] >= 0.0)
        & (grid_coords[:, 1] >= 0.0)
        & (grid_coords[:, 2] >= 0.0)
        & (grid_coords[:, 0] <= nx - 1)
        & (grid_coords[:, 1] <= ny - 1)
        & (grid_coords[:, 2] <= nz - 1)
    )

    out = np.full(coords.shape[0], fill_value, dtype=np.float32)
    if not np.any(valid):
        return out

    base = np.floor(grid_coords[valid]).astype(np.int64)
    b = np.stack(
        [
            np.clip(base[:, 0], 0, nx - 2),
            np.clip(base[:, 1], 0, ny - 2),
            np.clip(base[:, 2], 0, nz - 2),
        ],
        axis=1,
    )
    f = grid_coords[valid] - b.astype(np.float32)
    x0, y0, z0 = b[:, 0], b[:, 1], b[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    wx, wy, wz = f[:, 0], f[:, 1], f[:, 2]

    c000 = density[x0, y0, z0]
    c100 = density[x1, y0, z0]
    c010 = density[x0, y1, z0]
    c110 = density[x1, y1, z0]
    c001 = density[x0, y0, z1]
    c101 = density[x1, y0, z1]
    c011 = density[x0, y1, z1]
    c111 = density[x1, y1, z1]

    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    out[valid] = (c0 * (1.0 - wz) + c1 * wz).astype(np.float32)
    return out


def compute_crop_pose_density_features(
    crop: LigandDensityCrop,
    ligand_coords_abs: np.ndarray,
    shell_radii: Sequence[float] = (0.8, 1.6, 2.4),
    vector_shell_radii: Sequence[float] | None = None,
    n_vector_sphere: int = 26,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-atom scalar, per-atom vector, and per-pose density features.

    The scalar features describe local density occupancy. The vector features
    are density-weighted directions on spherical shells, giving an l=1
    gradient-like signal in the same Cartesian frame as the model coordinates.
    """
    coords = np.asarray(ligand_coords_abs, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected ligand coords with shape (N,3), got {coords.shape}")

    norm_center = crop.norm_center
    norm_scale = _safe_scale(crop.norm_scale)
    radii = tuple(float(r) for r in shell_radii)
    vector_radii = tuple(float(r) for r in (vector_shell_radii if vector_shell_radii is not None else radii))

    center_vals = interp_crop_points(crop, coords)
    center_vals = np.nan_to_num(center_vals, nan=norm_center, posinf=norm_center, neginf=norm_center)
    center_vals = np.clip((center_vals - norm_center) / norm_scale, -8.0, 8.0)

    feat_chunks = [center_vals[:, None]]
    shell_means = []
    for radius in radii:
        sample_pts = coords[:, None, :] + float(radius) * _AXIS_DIRECTIONS[None, :, :]
        shell_vals = interp_crop_points(crop, sample_pts.reshape(-1, 3)).reshape(len(coords), -1)
        shell_vals = np.nan_to_num(shell_vals, nan=norm_center, posinf=norm_center, neginf=norm_center)
        shell_vals = np.clip((shell_vals - norm_center) / norm_scale, -8.0, 8.0)
        shell_mean = shell_vals.mean(axis=1, keepdims=True)
        shell_max = shell_vals.max(axis=1, keepdims=True)
        shell_min = shell_vals.min(axis=1, keepdims=True)
        shell_std = shell_vals.std(axis=1, keepdims=True)
        feat_chunks.extend([shell_mean, shell_max, shell_min, shell_std])
        shell_means.append(shell_mean[:, 0])

    outer_shell_mean = shell_means[-1] if shell_means else np.zeros(len(coords), dtype=np.float32)
    local_contrast = (center_vals - outer_shell_mean)[:, None]
    atom_features = np.concatenate(feat_chunks + [local_contrast], axis=1).astype(np.float32)

    sphere_dirs = fibonacci_sphere(n_vector_sphere)
    atom_vectors = np.zeros((len(coords), len(vector_radii), 3), dtype=np.float32)
    if len(coords) and vector_radii:
        for shell_idx, radius in enumerate(vector_radii):
            sample_pts = coords[:, None, :] + radius * sphere_dirs[None, :, :]
            shell_vals = interp_crop_points(crop, sample_pts.reshape(-1, 3)).reshape(len(coords), -1)
            shell_vals = np.nan_to_num(shell_vals, nan=norm_center, posinf=norm_center, neginf=norm_center)
            shell_vals = np.clip((shell_vals - norm_center) / norm_scale, -8.0, 8.0)
            weights = np.clip(shell_vals, 0.0, None)
            weight_sum = weights.sum(axis=1, keepdims=True)
            vec = (weights[:, :, None] * sphere_dirs[None, :, :]).sum(axis=1)
            # Average by total positive density so empty/flat shells stay near 0.
            atom_vectors[:, shell_idx, :] = vec / (weight_sum + 1e-6)

    p25 = float(np.percentile(center_vals, 25)) if center_vals.size else 0.0
    pose_summary = np.asarray(
        [
            float(center_vals.mean()) if center_vals.size else 0.0,
            float(center_vals.min()) if center_vals.size else 0.0,
            p25,
            float(local_contrast[:, 0].mean()) if local_contrast.size else 0.0,
            float(center_vals.max()) if center_vals.size else 0.0,
            float(center_vals.std()) if center_vals.size else 0.0,
        ],
        dtype=np.float32,
    )
    return atom_features, atom_vectors.astype(np.float32), pose_summary
