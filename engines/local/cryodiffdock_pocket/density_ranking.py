"""Ground-truth-free Gaussian ligand-density scoring for pose ranking.

The experimental crop is fixed for a complex.  Every candidate is rendered on
the same Cartesian grid and evaluated on the same crop-centred mask, so scores
are comparable within that complex.  Reference ligand coordinates are never
read here; they belong only in evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class DensityCropLike(Protocol):
    density: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    norm_center: float
    norm_scale: float


@dataclass(frozen=True)
class CartesianDensityCrop:
    density: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    norm_center: float
    norm_scale: float
    path: str


def load_cartesian_density_crop(path: str | Path) -> CartesianDensityCrop:
    """Load the ranking crop without importing PyTorch sampling code."""
    crop_path = Path(path).expanduser().resolve()
    with np.load(crop_path, allow_pickle=False) as archive:
        required = {"density", "origin", "spacing", "norm_center", "norm_scale"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"Density crop {crop_path} is missing keys: {missing}")
        density = np.asarray(archive["density"], dtype=np.float32)
        origin = np.asarray(archive["origin"], dtype=np.float32).reshape(3)
        spacing = np.asarray(archive["spacing"], dtype=np.float32)
        if spacing.ndim == 0:
            spacing = np.repeat(float(spacing), 3)
        spacing = spacing.reshape(3)
        norm_center = float(np.asarray(archive["norm_center"]).reshape(()))
        norm_scale = float(np.asarray(archive["norm_scale"]).reshape(()))
    if density.ndim != 3 or min(density.shape) < 2:
        raise ValueError(f"Density crop must be 3D with axes >=2, got {density.shape}")
    if not np.isfinite(density).all() or not np.isfinite(origin).all():
        raise ValueError(f"Density crop contains non-finite values: {crop_path}")
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Density crop has invalid spacing: {crop_path}")
    if not np.isfinite(norm_center) or not np.isfinite(norm_scale) or norm_scale <= 1e-6:
        raise ValueError(f"Density crop has invalid normalization: {crop_path}")
    return CartesianDensityCrop(
        density, origin, spacing, norm_center, norm_scale, str(crop_path)
    )


@dataclass(frozen=True)
class GaussianDensityScore:
    sigma_A: float
    ncc: float
    cosine_overlap: float
    mask_voxels: int
    ligand_in_bounds_fraction: float


def crop_center(crop: DensityCropLike) -> np.ndarray:
    """Return the geometric centre of a Cartesian crop in Angstrom."""
    extent = (np.asarray(crop.density.shape, dtype=np.float32) - 1.0) * crop.spacing
    return crop.origin + 0.5 * extent


def normalized_target(crop: DensityCropLike) -> np.ndarray:
    return (
        (np.asarray(crop.density, dtype=np.float32) - float(crop.norm_center))
        / float(crop.norm_scale)
    ).astype(np.float32)


def fixed_crop_mask(
    crop: DensityCropLike,
    radius_A: float = 10.0,
    target_floor: float | None = None,
    min_voxels: int = 64,
) -> np.ndarray:
    """Build one pose-independent mask around the supplied pocket/crop centre.

    A pose-dependent support would make scores incomparable and can reward a
    candidate merely by choosing an easier subset of voxels.  ``target_floor``
    is optional and disabled by default so missing-density/background voxels
    still penalize an incorrectly shaped ligand envelope.
    """
    if radius_A <= 0:
        raise ValueError(f"mask radius must be positive, got {radius_A}")
    axes = [
        crop.origin[axis]
        + np.arange(crop.density.shape[axis], dtype=np.float32) * crop.spacing[axis]
        for axis in range(3)
    ]
    center = crop_center(crop)
    squared = None
    for axis, values in enumerate(axes):
        shape = [1, 1, 1]
        shape[axis] = len(values)
        term = (values.reshape(shape) - center[axis]) ** 2
        squared = term if squared is None else squared + term
    mask = squared <= float(radius_A) ** 2
    if target_floor is not None:
        thresholded = mask & (normalized_target(crop) >= float(target_floor))
        if int(thresholded.sum()) >= int(min_voxels):
            mask = thresholded
    if int(mask.sum()) < int(min_voxels):
        raise ValueError(
            f"NCC mask contains only {int(mask.sum())} voxels; require {min_voxels}"
        )
    return mask


def coordinates_in_bounds_fraction(coordinates: np.ndarray, crop: DensityCropLike) -> float:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    upper = crop.origin + (np.asarray(crop.density.shape) - 1.0) * crop.spacing
    inside = ((coordinates >= crop.origin) & (coordinates <= upper)).all(axis=1)
    return float(inside.mean()) if len(inside) else 0.0


def render_gaussian_ligand_density(
    crop: DensityCropLike,
    coordinates: np.ndarray,
    atomic_numbers: Sequence[float],
    sigma_A: float,
    cutoff_sigma: float = 4.0,
) -> np.ndarray:
    """Render a Z-weighted ligand envelope using local voxel bounding boxes."""
    if sigma_A <= 0 or cutoff_sigma <= 0:
        raise ValueError("sigma_A and cutoff_sigma must be positive")
    coordinates = np.asarray(coordinates, dtype=np.float32)
    weights = np.asarray(atomic_numbers, dtype=np.float32).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must be (N,3), got {coordinates.shape}")
    if len(weights) != len(coordinates):
        raise ValueError(
            f"atomic number count {len(weights)} != coordinate count {len(coordinates)}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")
    if not np.isfinite(weights).all():
        raise ValueError("atomic numbers contain non-finite values")
    rendered = np.zeros(crop.density.shape, dtype=np.float32)
    cutoff_A = float(cutoff_sigma) * float(sigma_A)
    inv_two_sigma2 = 1.0 / (2.0 * float(sigma_A) ** 2)
    shape = np.asarray(crop.density.shape, dtype=int)
    crop_lower = np.asarray(crop.origin, dtype=np.float32)
    crop_upper = crop_lower + (shape - 1) * np.asarray(crop.spacing, dtype=np.float32)

    for atom, weight in zip(coordinates, weights):
        # Reject Gaussian supports that do not intersect the crop before
        # converting Cartesian coordinates to integer voxel indices.  Very
        # distant but finite diffusion outliers can exceed the platform integer
        # range during astype(int); they correctly contribute zero density.
        if np.any(atom + cutoff_A < crop_lower) or np.any(atom - cutoff_A > crop_upper):
            continue
        lower = np.floor((atom - cutoff_A - crop.origin) / crop.spacing).astype(int)
        upper = np.ceil((atom + cutoff_A - crop.origin) / crop.spacing).astype(int) + 1
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, shape)
        if np.any(lower >= upper):
            continue
        axes = [
            crop.origin[axis]
            + np.arange(lower[axis], upper[axis], dtype=np.float32) * crop.spacing[axis]
            for axis in range(3)
        ]
        dx2 = (axes[0] - atom[0])[:, None, None] ** 2
        dy2 = (axes[1] - atom[1])[None, :, None] ** 2
        dz2 = (axes[2] - atom[2])[None, None, :] ** 2
        gaussian = np.exp(-(dx2 + dy2 + dz2) * inv_two_sigma2)
        slices = tuple(slice(lower[axis], upper[axis]) for axis in range(3))
        rendered[slices] += float(max(weight, 1.0)) * gaussian.astype(np.float32)
    return rendered


def normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return float("nan")
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def cosine_overlap(x: np.ndarray, y: np.ndarray) -> float:
    x = np.maximum(np.asarray(x, dtype=np.float64).reshape(-1), 0.0)
    y = np.maximum(np.asarray(y, dtype=np.float64).reshape(-1), 0.0)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float("nan")


def score_gaussian_density(
    crop: DensityCropLike,
    coordinates: np.ndarray,
    atomic_numbers: Sequence[float],
    sigma_A: float,
    mask: np.ndarray,
    cutoff_sigma: float = 4.0,
) -> GaussianDensityScore:
    if mask.shape != crop.density.shape:
        raise ValueError(f"mask shape {mask.shape} != crop shape {crop.density.shape}")
    model = render_gaussian_ligand_density(
        crop, coordinates, atomic_numbers, sigma_A, cutoff_sigma=cutoff_sigma
    )
    target = normalized_target(crop)
    return GaussianDensityScore(
        sigma_A=float(sigma_A),
        ncc=normalized_cross_correlation(model[mask], target[mask]),
        cosine_overlap=cosine_overlap(model[mask], target[mask]),
        mask_voxels=int(mask.sum()),
        ligand_in_bounds_fraction=coordinates_in_bounds_fraction(coordinates, crop),
    )


def zscore(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    output = np.zeros_like(array)
    if not finite.any():
        return output
    mean = float(array[finite].mean())
    std = float(array[finite].std())
    if std > 1e-12:
        output[finite] = (array[finite] - mean) / std
    output[~finite] = -10.0
    return output


def sigma_column(prefix: str, sigma_A: float) -> str:
    label = f"{float(sigma_A):g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_sigma_{label}"
