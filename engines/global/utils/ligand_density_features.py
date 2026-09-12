"""Ligand-local cryo-EM density features for pose-dependent confidence models."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


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


def load_map_mapping(csv_path: str) -> Dict[str, str]:
    """Load ``PDB_ID -> map_path`` from a 04_maps-like CSV."""
    df = pd.read_csv(csv_path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).strip().upper()
        map_path = str(row["map_path"]).strip()
        if pdb_id and map_path and map_path != "nan":
            mapping[pdb_id] = map_path
    return mapping


def _get_map_utils():
    from utils.maps import open_map, interp_points

    return open_map, interp_points


@lru_cache(maxsize=32)
def get_grid(map_path: str):
    """Cached GEMMI grid loader."""
    open_map, _ = _get_map_utils()
    _, grid = open_map(map_path)
    return grid


def atom_feature_dim(shell_radii: Sequence[float]) -> int:
    """Per-atom feature dimension: center(1) + shells(4*n) + contrast(1)."""
    return 2 + 4 * len(tuple(shell_radii))


def summary_feature_dim() -> int:
    """Per-pose summary feature dimension."""
    return 6  # mean, min, p25, contrast, max, std


def _safe_scale(value: float, fallback: float = 1.0) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 1e-6:
        return float(fallback)
    return value


@lru_cache(maxsize=32)
def get_grid_normalization_stats(map_path: str) -> Tuple[float, float]:
    """Return cached global normalization stats for one EM map."""
    grid = get_grid(map_path)
    grid_values = np.asarray(grid, dtype=np.float32)
    finite = grid_values[np.isfinite(grid_values)]
    if finite.size == 0:
        return 0.0, 1.0

    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = _safe_scale(1.4826 * mad, fallback=float(np.std(finite)))
    return center, scale


def _safe_stats(values: np.ndarray) -> Tuple[float, float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.max(values)),
        float(np.min(values)),
        float(np.std(values)),
    )


def compute_pose_density_features(
    grid,
    ligand_coords_abs: np.ndarray,
    shell_radii: Sequence[float] = (0.8, 1.6, 2.4),
    normalization_stats: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-atom features and per-pose summary for one ligand pose.

    Per-atom features (atom_feature_dim):
    - center density, shell mean/max/min/std at each radius, local contrast

    Per-pose summary (summary_feature_dim = 6):
    - mean, min, p25, contrast_mean, max, std of center_vals
    """
    coords = np.asarray(ligand_coords_abs, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected ligand coords with shape (N,3), got {coords.shape}")

    norm_center, norm_scale = normalization_stats
    norm_scale = _safe_scale(norm_scale)

    _, interp_points = _get_map_utils()
    center_vals = interp_points(grid, coords)
    center_vals = np.nan_to_num(center_vals, nan=norm_center, posinf=norm_center, neginf=norm_center)
    center_vals = (center_vals - norm_center) / norm_scale
    center_vals = np.clip(center_vals, -8.0, 8.0)

    feat_chunks = [center_vals[:, None]]
    shell_means = []
    for radius in shell_radii:
        sample_pts = coords[:, None, :] + float(radius) * _AXIS_DIRECTIONS[None, :, :]
        shell_vals = interp_points(grid, sample_pts.reshape(-1, 3)).reshape(len(coords), -1)
        shell_vals = np.nan_to_num(shell_vals, nan=norm_center, posinf=norm_center, neginf=norm_center)
        shell_vals = (shell_vals - norm_center) / norm_scale
        shell_vals = np.clip(shell_vals, -8.0, 8.0)
        shell_mean = shell_vals.mean(axis=1, keepdims=True)
        shell_max = shell_vals.max(axis=1, keepdims=True)
        shell_min = shell_vals.min(axis=1, keepdims=True)
        shell_std = shell_vals.std(axis=1, keepdims=True)
        feat_chunks.extend([shell_mean, shell_max, shell_min, shell_std])
        shell_means.append(shell_mean[:, 0])

    outer_shell_mean = shell_means[-1] if shell_means else np.zeros(len(coords), dtype=np.float32)
    local_contrast = (center_vals - outer_shell_mean)[:, None]

    atom_features = np.concatenate(feat_chunks + [local_contrast], axis=1).astype(np.float32)

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
    return atom_features, pose_summary


def compute_pose_cache_density_features(
    map_path: str,
    original_center: np.ndarray,
    pose_positions_centered: np.ndarray,
    shell_radii: Sequence[float] = (0.8, 1.6, 2.4),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute density features for every pose in a cached pose tensor."""
    grid = get_grid(map_path)
    centered = np.asarray(pose_positions_centered, dtype=np.float32)
    center = np.asarray(original_center, dtype=np.float32).reshape(1, 1, 3)
    abs_positions = centered + center
    normalization_stats = get_grid_normalization_stats(map_path)

    atom_features = []
    summaries = []
    for pose in abs_positions:
        atom_feat, summary = compute_pose_density_features(
            grid,
            pose,
            shell_radii=shell_radii,
            normalization_stats=normalization_stats,
        )
        atom_features.append(atom_feat)
        summaries.append(summary)

    return np.stack(atom_features, axis=0), np.stack(summaries, axis=0)