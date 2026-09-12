"""Training-free cryo-EM ligand-density guidance for DiffDock-Pocket.

The pretrained DiffDock-Pocket score is treated as a structural prior.  A
differentiable likelihood score from a ligand-density crop is projected onto
the ligand translation and rotation degrees of freedom and composed with the
learned score during reverse diffusion.

Density crops use the project ``.npz`` schema:

``density`` (X,Y,Z), ``origin`` (3,), ``spacing`` (scalar or 3,),
``norm_center`` and ``norm_scale``.  Coordinates are absolute Angstroms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from utils.geometry import axis_angle_to_matrix


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if bool(np.isnan(value)):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text and text.lower() not in {"none", "nan"} else None


@dataclass(frozen=True)
class DensityCrop:
    density: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    norm_center: float
    norm_scale: float
    path: str


def load_density_crop(path: str | Path) -> DensityCrop:
    """Load and validate one Cartesian ligand-density crop."""
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
        spacing = spacing.astype(np.float32).reshape(3)
        norm_center = float(np.asarray(archive["norm_center"]).reshape(()))
        norm_scale = float(np.asarray(archive["norm_scale"]).reshape(()))

    if density.ndim != 3 or min(density.shape) < 2:
        raise ValueError(f"Density crop must be a 3D grid with every axis >=2, got {density.shape}")
    if not np.isfinite(density).all():
        raise ValueError(f"Density crop contains non-finite voxels: {crop_path}")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Density crop has invalid origin/spacing: {crop_path}")
    if not np.isfinite(norm_center):
        raise ValueError(f"Density crop has invalid norm_center: {crop_path}")
    if not np.isfinite(norm_scale) or norm_scale <= 1e-6:
        raise ValueError(f"Density crop has invalid norm_scale: {crop_path}")
    return DensityCrop(density, origin, spacing, norm_center, norm_scale, str(crop_path))


def trilinear_density(
    density: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
    points: torch.Tensor,
    fill_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiably sample ``density[X,Y,Z]`` at absolute coordinates.

    Points outside the crop receive ``fill_value`` and zero coordinate
    gradient.  The returned mask is true for points inside the crop.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shaped (N,3), got {tuple(points.shape)}")
    grid = (points - origin) / spacing
    shape = torch.as_tensor(density.shape, device=points.device, dtype=points.dtype)
    valid = ((grid >= 0.0) & (grid <= shape - 1.0)).all(dim=-1)

    # Clamp only for safe indexing. Multiplication by ``valid`` below makes
    # out-of-box values constant, and therefore gives them zero gradient.
    max_base = torch.as_tensor(
        [density.shape[0] - 2, density.shape[1] - 2, density.shape[2] - 2],
        device=points.device,
        dtype=torch.long,
    )
    base = torch.floor(grid).long()
    base = torch.maximum(base, torch.zeros_like(base))
    base = torch.minimum(base, max_base)
    frac = grid - base.to(grid.dtype)
    frac = torch.clamp(frac, 0.0, 1.0)

    x0, y0, z0 = base.unbind(dim=-1)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    wx, wy, wz = frac.unbind(dim=-1)
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
    sampled = c0 * (1.0 - wz) + c1 * wz
    fill = torch.full_like(sampled, float(fill_value))
    return torch.where(valid, sampled, fill), valid


def rigid_density_scores(
    coordinates_abs: torch.Tensor,
    density: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
    norm_center: float,
    norm_scale: float,
    return_coordinate_score: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Return translation and rotation likelihood scores for one ligand.

    Rotation is represented as the infinitesimal axis-angle torque around the
    ligand centroid, matching DiffDock-Pocket's three-component SO(3) update.
    """
    with torch.enable_grad():
        coords = coordinates_abs.detach().clone().requires_grad_(True)
        values, valid = trilinear_density(
            density=density,
            origin=origin,
            spacing=spacing,
            points=coords,
            fill_value=norm_center,
        )
        # Only the gradient direction is used downstream (its norm is matched
        # to the learned score), so do not clip high-density voxels here.  A
        # hard clamp can erase the gradient precisely around a strong peak.
        normalized = (values - float(norm_center)) / float(norm_scale)
        objective = normalized.mean()
        coordinate_score = torch.autograd.grad(objective, coords, create_graph=False)[0]

    translation, rotation = project_rigid_scores(coords.detach(), coordinate_score.detach())
    audit = {
        "density_objective": float(objective.detach().cpu()),
        "in_bounds_fraction": float(valid.float().mean().detach().cpu()),
        "coordinate_gradient_norm": float(coordinate_score.norm().detach().cpu()),
        "translation_gradient_norm": float(translation.norm().detach().cpu()),
        "rotation_gradient_norm": float(rotation.norm().detach().cpu()),
    }
    if return_coordinate_score:
        return translation.detach(), rotation.detach(), coordinate_score.detach(), audit
    return translation.detach(), rotation.detach(), audit


def gaussian_ncc(
    sample_points: torch.Tensor,
    coordinates: torch.Tensor,
    atomic_weights: torch.Tensor,
    target_density: torch.Tensor,
    sigma_A: float,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Differentiable NCC between a Z-weighted ligand envelope and a map.

    ``sample_points`` and ``target_density`` define one fixed, pose-independent
    crop mask.  Fixing the support is essential: a pose must not improve its
    score merely by moving the region on which correlation is evaluated.
    """
    if sigma_A <= 0 or chunk_size <= 0:
        raise ValueError("sigma_A and chunk_size must be positive")
    if sample_points.ndim != 2 or sample_points.shape[-1] != 3:
        raise ValueError(f"sample_points must be (M,3), got {sample_points.shape}")
    if coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must be (N,3), got {coordinates.shape}")
    if atomic_weights.numel() != coordinates.shape[0]:
        raise ValueError("atomic weight count must match coordinate count")
    if target_density.numel() != sample_points.shape[0]:
        raise ValueError("target density count must match sample point count")

    weights = atomic_weights.to(coordinates).reshape(1, -1)
    weights = weights / torch.clamp(weights.mean(), min=1.0)
    model_chunks = []
    inv_two_sigma2 = 1.0 / (2.0 * float(sigma_A) ** 2)
    for start in range(0, sample_points.shape[0], int(chunk_size)):
        points = sample_points[start:start + int(chunk_size)]
        squared = ((points[:, None, :] - coordinates[None, :, :]) ** 2).sum(dim=-1)
        model_chunks.append((torch.exp(-squared * inv_two_sigma2) * weights).sum(dim=1))
    model = torch.cat(model_chunks, dim=0)
    target = target_density.to(model)
    model_centered = model - model.mean()
    target_centered = target - target.mean()
    denominator = torch.sqrt(
        (model_centered.square().mean()).clamp_min(1e-8)
        * (target_centered.square().mean()).clamp_min(1e-8)
    )
    return (model_centered * target_centered).mean() / denominator


def gaussian_ncc_batch(
    sample_points: torch.Tensor,
    coordinates: torch.Tensor,
    atomic_weights: torch.Tensor,
    target_density: torch.Tensor,
    sigma_A: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Batched Gaussian ligand-map NCC for equal-composition poses.

    ``coordinates`` has shape ``(B,N,3)`` and the returned tensor has shape
    ``(B,)``. Each pose is independent, so differentiating the sum of returned
    NCCs gives the per-pose Cartesian gradient without cross-pose coupling.
    """
    if sigma_A <= 0 or chunk_size <= 0:
        raise ValueError("sigma_A and chunk_size must be positive")
    if sample_points.ndim != 2 or sample_points.shape[-1] != 3:
        raise ValueError(f"sample_points must be (M,3), got {sample_points.shape}")
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError(f"coordinates must be (B,N,3), got {coordinates.shape}")
    if atomic_weights.numel() != coordinates.shape[1]:
        raise ValueError("atomic weight count must match ligand atom count")
    if target_density.numel() != sample_points.shape[0]:
        raise ValueError("target density count must match sample point count")

    weights = atomic_weights.to(coordinates).reshape(1, 1, -1)
    weights = weights / torch.clamp(weights.mean(), min=1.0)
    inv_two_sigma2 = 1.0 / (2.0 * float(sigma_A) ** 2)
    model_chunks = []
    for start in range(0, sample_points.shape[0], int(chunk_size)):
        points = sample_points[start:start + int(chunk_size)]
        squared = (
            (points[None, :, None, :] - coordinates[:, None, :, :]) ** 2
        ).sum(dim=-1)
        model_chunks.append(
            (torch.exp(-squared * inv_two_sigma2) * weights).sum(dim=-1)
        )
    model = torch.cat(model_chunks, dim=1)
    target = target_density.to(model)
    model_centered = model - model.mean(dim=1, keepdim=True)
    target_centered = target - target.mean()
    denominator = torch.sqrt(
        model_centered.square().mean(dim=1).clamp_min(1e-8)
        * target_centered.square().mean().clamp_min(1e-8)
    )
    return (model_centered * target_centered[None, :]).mean(dim=1) / denominator


def rigid_ncc_scores(
    coordinates_abs: torch.Tensor,
    atomic_weights: torch.Tensor,
    sample_points: torch.Tensor,
    target_density: torch.Tensor,
    sigma_A: float,
    crop_origin: torch.Tensor,
    crop_spacing: torch.Tensor,
    crop_shape: Sequence[int],
    chunk_size: int = 8192,
    return_coordinate_score: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Return rigid translation/rotation gradients of Gaussian map NCC."""
    with torch.enable_grad():
        coords = coordinates_abs.detach().clone().requires_grad_(True)
        objective = gaussian_ncc(
            sample_points=sample_points,
            coordinates=coords,
            atomic_weights=atomic_weights,
            target_density=target_density,
            sigma_A=sigma_A,
            chunk_size=chunk_size,
        )
        coordinate_score = torch.autograd.grad(objective, coords, create_graph=False)[0]

    translation, rotation = project_rigid_scores(coords.detach(), coordinate_score.detach())
    shape = torch.as_tensor(crop_shape, device=coords.device, dtype=coords.dtype)
    upper = crop_origin + (shape - 1.0) * crop_spacing
    valid = ((coords.detach() >= crop_origin) & (coords.detach() <= upper)).all(dim=-1)
    audit = {
        "density_objective": float(objective.detach().cpu()),
        "in_bounds_fraction": float(valid.float().mean().detach().cpu()),
        "coordinate_gradient_norm": float(coordinate_score.norm().detach().cpu()),
        "translation_gradient_norm": float(translation.norm().detach().cpu()),
        "rotation_gradient_norm": float(rotation.norm().detach().cpu()),
    }
    if return_coordinate_score:
        return translation.detach(), rotation.detach(), coordinate_score.detach(), audit
    return translation.detach(), rotation.detach(), audit


def project_rigid_scores(
    coordinates: torch.Tensor,
    coordinate_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project Cartesian atom scores onto translation and axis-angle rotation."""
    if coordinates.shape != coordinate_score.shape or coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"coordinates and coordinate_score must both be (N,3), got "
            f"{tuple(coordinates.shape)} and {tuple(coordinate_score.shape)}"
        )
    translation = coordinate_score.sum(dim=0)
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    rotation = torch.cross(centered, coordinate_score, dim=-1).sum(dim=0)
    return translation, rotation


def project_torsion_scores(
    coordinates: torch.Tensor,
    coordinate_score: torch.Tensor,
    edge_index: torch.Tensor,
    mask_rotate: torch.Tensor,
) -> torch.Tensor:
    """Project Cartesian atom scores onto ligand rotatable-bond angles.

    ``edge_index`` and ``mask_rotate`` use the same convention as
    ``utils.torsion.modify_conformer_torsion_angles``: the first atom is the
    fixed side of the bond, the second atom belongs to the rotated component.
    The returned value is the infinitesimal derivative with respect to the
    positive angle convention used by the sampler.
    """
    if coordinates.shape != coordinate_score.shape or coordinates.ndim != 2 or coordinates.shape[-1] != 3:
        raise ValueError(
            f"coordinates and coordinate_score must both be (N,3), got "
            f"{tuple(coordinates.shape)} and {tuple(coordinate_score.shape)}"
        )
    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError(f"edge_index must be (K,2), got {tuple(edge_index.shape)}")
    if mask_rotate.shape != (edge_index.shape[0], coordinates.shape[0]):
        raise ValueError(
            f"mask_rotate must be (K,N), got {tuple(mask_rotate.shape)} for "
            f"K={edge_index.shape[0]}, N={coordinates.shape[0]}"
        )

    edge_index = edge_index.to(device=coordinates.device, dtype=torch.long)
    mask_rotate = mask_rotate.to(device=coordinates.device, dtype=torch.bool)
    torsion = torch.zeros(edge_index.shape[0], dtype=coordinates.dtype, device=coordinates.device)
    for index, (u, v) in enumerate(edge_index):
        axis = coordinates[u] - coordinates[v]
        axis_norm = torch.linalg.vector_norm(axis)
        if float(axis_norm.detach()) <= 1e-8:
            continue
        axis = axis / axis_norm
        relative = coordinates[mask_rotate[index]] - coordinates[v]
        torque = torch.cross(relative, coordinate_score[mask_rotate[index]], dim=-1).sum(dim=0)
        torsion[index] = torch.dot(axis, torque)
    return torsion


def _apply_torsion_updates(
    coordinates: torch.Tensor,
    edge_index: torch.Tensor,
    mask_rotate: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Apply angle updates with the same convention as the native sampler."""
    updated = coordinates.detach().clone()
    edge_index = edge_index.to(device=updated.device, dtype=torch.long)
    mask_rotate = mask_rotate.to(device=updated.device, dtype=torch.bool)
    updates = updates.to(device=updated.device, dtype=updated.dtype).reshape(-1)
    if updates.numel() != edge_index.shape[0]:
        raise ValueError("torsion update count must match edge_index")
    for index, (u, v) in enumerate(edge_index):
        if abs(float(updates[index])) <= 1e-12:
            continue
        axis = updated[u] - updated[v]
        axis_norm = torch.linalg.vector_norm(axis)
        if float(axis_norm) <= 1e-8:
            continue
        rot_mat = axis_angle_to_matrix(axis / axis_norm * updates[index])
        rotated = (updated[mask_rotate[index]] - updated[v]) @ rot_mat.T + updated[v]
        updated[mask_rotate[index]] = rotated
    return updated


def estimate_x0_coordinates(
    coordinates: torch.Tensor,
    translation_score: torch.Tensor,
    rotation_score: torch.Tensor,
    tr_sigma: float,
    rot_sigma: float,
    torsion_score: Optional[torch.Tensor] = None,
    tor_sigma: Optional[float] = None,
    edge_index: Optional[torch.Tensor] = None,
    mask_rotate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """First-order Tweedie estimate of clean ligand coordinates.

    Translation and SO(3) use ``x0 ~= xt + sigma**2 * score``.  Torsion uses
    the analogous local-angle approximation and is applied after the rigid
    update.  This is intentionally an explicit approximation, not a claim of
    an exact wrapped-SO(3)/torus posterior mean.
    """
    estimated = coordinates.detach().clone()
    estimated = estimated + float(tr_sigma) ** 2 * translation_score.to(estimated)
    rotation_update = float(rot_sigma) ** 2 * rotation_score.to(estimated)
    center = estimated.mean(dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix(rotation_update)
    estimated = (estimated - center) @ rot_mat.T + center

    if torsion_score is not None:
        if tor_sigma is None or edge_index is None or mask_rotate is None:
            raise ValueError("torsion x0 estimation requires tor_sigma, edge_index, and mask_rotate")
        estimated = _apply_torsion_updates(
            estimated,
            edge_index=edge_index,
            mask_rotate=mask_rotate,
            updates=float(tor_sigma) ** 2 * torsion_score,
        )
    return estimated


def guidance_weight(progress: float, start_fraction: float) -> float:
    """Linear late-step ramp in [0,1]."""
    progress = float(np.clip(progress, 0.0, 1.0))
    start_fraction = float(start_fraction)
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError(f"density guidance start must be in [0,1), got {start_fraction}")
    if progress <= start_fraction:
        return 0.0
    return float((progress - start_fraction) / (1.0 - start_fraction))


def uncertainty_matched_sigma(
    base_sigma_A: float,
    tr_sigma: Optional[float],
    rot_sigma: Optional[float],
    radius_of_gyration_A: float,
    noise_scale: float,
    sigma_max_A: float,
) -> float:
    """Widen the ligand envelope by the pose uncertainty at the current step.

    At noise level ``t`` the pose is only known to within roughly ``tr_sigma``
    in translation and ``rot_sigma * Rg`` in rotation.  The density such a pose
    is *expected* to explain is therefore its envelope convolved with that
    positional spread, and convolving two Gaussians adds their variances::

        sigma_eff**2 = base**2 + (noise_scale * sigma_pos)**2

    This replaces a fixed 1.5 A envelope, which measures the wrong quantity
    when compared against a pose that is still several Angstrom and tens of
    degrees away from its denoised target.  ``noise_scale = 0`` reproduces the
    fixed-sigma behaviour exactly.
    """
    if noise_scale <= 0.0:
        return float(base_sigma_A)
    if base_sigma_A <= 0 or sigma_max_A <= 0:
        raise ValueError("base sigma and sigma cap must be positive")
    translation = float(tr_sigma) if tr_sigma is not None else 0.0
    rotation = (
        float(rot_sigma) * float(radius_of_gyration_A) if rot_sigma is not None else 0.0
    )
    positional = math.sqrt(translation ** 2 + rotation ** 2)
    widened = math.sqrt(
        float(base_sigma_A) ** 2 + (float(noise_scale) * positional) ** 2
    )
    return float(min(widened, float(sigma_max_A)))


def compose_score(
    model_score: torch.Tensor,
    likelihood_score: torch.Tensor,
    weight: float,
    clip_ratio: float,
    minimum_reference_norm: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add a direction-only likelihood score at a controlled model-score scale.

    ``weight=0`` is an exact identity.  For positive weights the likelihood
    direction is norm-matched to the corresponding learned score, avoiding a
    unit mismatch between density derivatives and diffusion scores.  Its norm
    is capped at ``clip_ratio`` times the reference score norm.
    """
    if float(weight) == 0.0:
        return model_score, torch.zeros_like(model_score)
    if clip_ratio <= 0:
        raise ValueError(f"density guidance clip ratio must be positive, got {clip_ratio}")
    guide_norm = torch.linalg.vector_norm(likelihood_score, dim=-1, keepdim=True)
    reference_norm = torch.linalg.vector_norm(model_score, dim=-1, keepdim=True)
    usable = guide_norm > 1e-12
    reference_scale = torch.clamp(reference_norm, min=float(minimum_reference_norm))
    addition_norm = torch.minimum(
        reference_scale * abs(float(weight)),
        reference_scale * float(clip_ratio),
    )
    direction = likelihood_score / torch.clamp(guide_norm, min=1e-12)
    addition = torch.where(usable, direction * addition_norm, torch.zeros_like(direction))
    if float(weight) < 0:
        addition = -addition
    return model_score + addition, addition


@dataclass
class LigandDensityGuidance:
    crop: DensityCrop
    lambda_translation: float = 0.5
    lambda_rotation: float = 0.5
    lambda_torsion: float = 0.5
    start_fraction: float = 0.6
    clip_ratio: float = 1.0
    objective: str = "atom_density"
    ncc_sigma_A: float = 1.5
    ncc_sigma_noise_scale: float = 0.0
    ncc_sigma_max_A: float = 4.0
    ncc_mask_radius_A: float = 10.0
    ncc_chunk_size: int = 8192
    use_x0_estimate: bool = False
    use_torsion_projection: bool = False
    _device_cache: dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _audit_steps: list[Mapping[str, float]] = field(default_factory=list, init=False, repr=False)
    _ncc_cache: dict[str, Tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.lambda_translation < 0 or self.lambda_rotation < 0 or self.lambda_torsion < 0:
            raise ValueError("Density guidance weights must be non-negative")
        if not 0.0 <= self.start_fraction < 1.0:
            raise ValueError(
                f"density guidance start must be in [0,1), got {self.start_fraction}"
            )
        if self.clip_ratio <= 0:
            raise ValueError(
                f"density guidance clip ratio must be positive, got {self.clip_ratio}"
            )
        if self.objective not in {"atom_density", "map_ncc"}:
            raise ValueError(f"Unknown density guidance objective: {self.objective}")
        if self.ncc_sigma_A <= 0 or self.ncc_mask_radius_A <= 0 or self.ncc_chunk_size <= 0:
            raise ValueError("NCC sigma, mask radius, and chunk size must be positive")
        if self.ncc_sigma_noise_scale < 0:
            raise ValueError(
                f"NCC sigma noise scale must be non-negative, got {self.ncc_sigma_noise_scale}"
            )
        if self.ncc_sigma_max_A < self.ncc_sigma_A:
            raise ValueError(
                "NCC sigma cap must be at least the base sigma "
                f"({self.ncc_sigma_max_A} < {self.ncc_sigma_A})"
            )
        if self.ncc_sigma_noise_scale > 0 and self.objective != "map_ncc":
            raise ValueError(
                "The noise-matched blur schedule only applies to the map_ncc "
                "objective; atom_density reads the map by trilinear "
                "interpolation and has no envelope width."
            )

    def _tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                torch.as_tensor(self.crop.density, dtype=torch.float32, device=device),
                torch.as_tensor(self.crop.origin, dtype=torch.float32, device=device),
                torch.as_tensor(self.crop.spacing, dtype=torch.float32, device=device),
            )
        return self._device_cache[key]

    def _ncc_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._ncc_cache:
            density, origin, spacing = self._tensors(device)
            axes = [
                origin[axis]
                + torch.arange(density.shape[axis], dtype=density.dtype, device=device)
                * spacing[axis]
                for axis in range(3)
            ]
            xx, yy, zz = torch.meshgrid(*axes, indexing="ij")
            points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
            center = origin + 0.5 * (
                torch.as_tensor(density.shape, dtype=density.dtype, device=device) - 1.0
            ) * spacing
            mask = torch.linalg.vector_norm(points - center.reshape(1, 3), dim=-1) \
                <= float(self.ncc_mask_radius_A)
            if int(mask.sum().item()) < 64:
                raise ValueError(f"NCC mask contains only {int(mask.sum().item())} voxels")
            target = ((density - float(self.crop.norm_center)) / float(self.crop.norm_scale)) \
                .reshape(-1)[mask]
            self._ncc_cache[key] = (points[mask], target)
        return self._ncc_cache[key]

    def apply(
        self,
        data_list: Sequence[Any],
        translation_score: torch.Tensor,
        rotation_score: torch.Tensor,
        torsion_score: Optional[torch.Tensor],
        tr_sigma: Optional[float],
        rot_sigma: Optional[float],
        tor_sigma: Optional[float],
        step_index: int,
        inference_steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        progress = float(step_index + 1) / float(inference_steps)
        ramp = guidance_weight(progress, self.start_fraction)
        if ramp == 0.0 or (
            self.lambda_translation == 0.0
            and self.lambda_rotation == 0.0
            and (not self.use_torsion_projection or self.lambda_torsion == 0.0)
        ):
            return translation_score, rotation_score, torsion_score
        if self.use_x0_estimate and (tr_sigma is None or rot_sigma is None):
            raise ValueError("x0 density guidance requires tr_sigma and rot_sigma")
        if self.ncc_sigma_noise_scale > 0 and (tr_sigma is None or rot_sigma is None):
            raise ValueError(
                "the noise-matched blur schedule requires tr_sigma and rot_sigma"
            )
        if self.use_torsion_projection and torsion_score is None:
            raise ValueError("torsion density guidance requires torsion_score")

        density, origin, spacing = self._tensors(device)
        if self.objective == "map_ncc":
            ncc_points, ncc_target = self._ncc_tensors(device)
        tr_likelihood = []
        rot_likelihood = []
        tor_likelihood = []
        effective_sigmas = []
        audits = []
        torsion_offset = 0
        for graph_index, graph in enumerate(data_list):
            coordinates_abs = (
                graph["ligand"].pos.detach().to(device)
                + graph.original_center.detach().to(device)
            )
            edge_mask = graph["ligand"].edge_mask
            edge_index = graph["ligand", "ligand"].edge_index.T[edge_mask]
            mask_rotate = graph["ligand"].mask_rotate
            if not isinstance(mask_rotate, torch.Tensor):
                mask_rotate = torch.as_tensor(mask_rotate)
            if mask_rotate.ndim == 3:
                mask_rotate = mask_rotate[0]
            tor_count = int(edge_index.shape[0])
            graph_torsion_score = None
            if torsion_score is not None and tor_count > 0:
                graph_torsion_score = torsion_score[
                    torsion_offset:torsion_offset + tor_count
                ].to(device)
            torsion_offset += tor_count

            density_coordinates = coordinates_abs
            if self.use_x0_estimate:
                density_coordinates = estimate_x0_coordinates(
                    coordinates_abs,
                    translation_score[graph_index].to(device),
                    rotation_score[graph_index].to(device),
                    tr_sigma=float(tr_sigma),
                    rot_sigma=float(rot_sigma),
                    torsion_score=graph_torsion_score if self.use_torsion_projection else None,
                    tor_sigma=float(tor_sigma) if tor_sigma is not None else None,
                    edge_index=edge_index,
                    mask_rotate=mask_rotate,
                )
            coordinate_score_full = None
            if self.objective == "map_ncc":
                atomic_index = graph["ligand"].x[:, 0].detach().to(device).long()
                heavy = atomic_index != 0  # index 0 is atomic number 1 (hydrogen)
                if not bool(heavy.any()):
                    raise ValueError("NCC guidance requires at least one ligand heavy atom")
                coordinates_ncc = density_coordinates[heavy]
                atomic_index = atomic_index[heavy]
                atomic_weights = torch.where(
                    atomic_index < 118,
                    atomic_index + 1,
                    torch.full_like(atomic_index, 6),
                ).to(torch.float32)
                effective_sigma = self.ncc_sigma_A
                if self.ncc_sigma_noise_scale > 0:
                    centered = coordinates_ncc - coordinates_ncc.mean(dim=0, keepdim=True)
                    radius_of_gyration = float(
                        torch.sqrt(
                            centered.square().sum(dim=-1).mean().clamp_min(0.0)
                        ).detach().cpu()
                    )
                    effective_sigma = uncertainty_matched_sigma(
                        base_sigma_A=self.ncc_sigma_A,
                        tr_sigma=tr_sigma,
                        rot_sigma=rot_sigma,
                        radius_of_gyration_A=radius_of_gyration,
                        noise_scale=self.ncc_sigma_noise_scale,
                        sigma_max_A=self.ncc_sigma_max_A,
                    )
                effective_sigmas.append(effective_sigma)
                ncc_result = rigid_ncc_scores(
                    coordinates_abs=coordinates_ncc,
                    atomic_weights=atomic_weights,
                    sample_points=ncc_points,
                    target_density=ncc_target,
                    sigma_A=effective_sigma,
                    crop_origin=origin,
                    crop_spacing=spacing,
                    crop_shape=density.shape,
                    chunk_size=self.ncc_chunk_size,
                    return_coordinate_score=self.use_torsion_projection,
                )
                if self.use_torsion_projection:
                    tr_value, rot_value, coordinate_score, audit = ncc_result
                    coordinate_score_full = torch.zeros_like(density_coordinates)
                    coordinate_score_full[heavy] = coordinate_score
                else:
                    tr_value, rot_value, audit = ncc_result
            else:
                density_result = rigid_density_scores(
                    density_coordinates,
                    density,
                    origin,
                    spacing,
                    self.crop.norm_center,
                    self.crop.norm_scale,
                    return_coordinate_score=self.use_torsion_projection,
                )
                if self.use_torsion_projection:
                    tr_value, rot_value, coordinate_score_full, audit = density_result
                else:
                    tr_value, rot_value, audit = density_result
            tr_likelihood.append(tr_value.cpu())
            rot_likelihood.append(rot_value.cpu())
            if self.use_torsion_projection and tor_count > 0:
                tor_likelihood.append(
                    project_torsion_scores(
                        density_coordinates,
                        coordinate_score_full,
                        edge_index,
                        mask_rotate,
                    ).cpu()
                )
            audits.append(audit)

        tr_likelihood_tensor = torch.stack(tr_likelihood, dim=0).to(translation_score)
        rot_likelihood_tensor = torch.stack(rot_likelihood, dim=0).to(rotation_score)
        translation_score, tr_addition = compose_score(
            translation_score,
            tr_likelihood_tensor,
            ramp * self.lambda_translation,
            self.clip_ratio,
        )
        rotation_score, rot_addition = compose_score(
            rotation_score,
            rot_likelihood_tensor,
            ramp * self.lambda_rotation,
            self.clip_ratio,
        )
        torsion_score_out = torsion_score
        torsion_addition_norm = 0.0
        if self.use_torsion_projection and torsion_score is not None:
            additions = []
            offset = 0
            for likelihood in tor_likelihood:
                count = int(likelihood.numel())
                model_part = torsion_score[offset:offset + count].reshape(1, -1)
                likelihood_part = likelihood.to(model_part).reshape(1, -1)
                combined, addition = compose_score(
                    model_part,
                    likelihood_part,
                    ramp * self.lambda_torsion,
                    self.clip_ratio,
                )
                additions.append(combined.reshape(-1))
                torsion_addition_norm += float(addition.norm().detach().cpu())
                offset += count
            if additions:
                torsion_score_out = torch.cat(additions, dim=0)
        self._audit_steps.append({
            "step": int(step_index + 1),
            "progress": progress,
            "ramp": ramp,
            "translation_weight": ramp * float(self.lambda_translation),
            "rotation_weight": ramp * float(self.lambda_rotation),
            "torsion_weight": ramp * float(self.lambda_torsion),
            "x0_estimate": bool(self.use_x0_estimate),
            "torsion_projection": bool(self.use_torsion_projection),
            "mean_ncc_sigma_A": (
                float(np.mean(effective_sigmas)) if effective_sigmas else None
            ),
            "mean_density_objective": float(np.mean([row["density_objective"] for row in audits])),
            "mean_in_bounds_fraction": float(np.mean([row["in_bounds_fraction"] for row in audits])),
            "mean_coordinate_gradient_norm": float(np.mean([row["coordinate_gradient_norm"] for row in audits])),
            "mean_translation_addition_norm": float(tr_addition.norm(dim=-1).mean()),
            "mean_rotation_addition_norm": float(rot_addition.norm(dim=-1).mean()),
            "torsion_addition_norm": torsion_addition_norm,
        })
        return translation_score, rotation_score, torsion_score_out

    def summary(self) -> Mapping[str, Any]:
        return {
            "enabled": True,
            "crop_path": self.crop.path,
            "objective": self.objective,
            "lambda_translation": float(self.lambda_translation),
            "lambda_rotation": float(self.lambda_rotation),
            "lambda_torsion": float(self.lambda_torsion),
            "start_fraction": float(self.start_fraction),
            "clip_ratio": float(self.clip_ratio),
            "use_x0_estimate": bool(self.use_x0_estimate),
            "use_torsion_projection": bool(self.use_torsion_projection),
            "ncc_sigma_A": float(self.ncc_sigma_A),
            "ncc_sigma_noise_scale": float(self.ncc_sigma_noise_scale),
            "ncc_sigma_max_A": float(self.ncc_sigma_max_A),
            "ncc_mask_radius_A": float(self.ncc_mask_radius_A),
            "ncc_chunk_size": int(self.ncc_chunk_size),
            "steps": list(self._audit_steps),
        }

    def write_audit(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary(), indent=2) + "\n")


@dataclass
class LigandDensityTranslationCorrector:
    """Gated post-model NCC translation correction during reverse diffusion.

    Unlike :class:`LigandDensityGuidance`, this mechanism never rescales a
    likelihood gradient to the learned score norm. After the normal model
    tr/rot/torsion update, it proposes one fixed physical translation along the
    raw Gaussian-NCC gradient and accepts it only when observable gates pass.
    """

    crop: DensityCrop
    step_cap_A: float = 0.02
    start_fraction: float = 0.7
    min_ncc_gain: float = 0.0
    min_in_bounds_fraction: float = 0.0
    protected_fraction: float = 0.0
    ncc_sigma_A: float = 1.5
    ncc_mask_radius_A: float = 10.0
    ncc_chunk_size: int = 512
    _helper: LigandDensityGuidance = field(init=False, repr=False)
    _num_particles: Optional[int] = field(default=None, init=False, repr=False)
    _protected_count: int = field(default=0, init=False, repr=False)
    _cumulative_path_A: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _net_vectors_A: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _accepted_steps: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _active_steps: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _cumulative_ncc_gain: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _last_ncc: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _step_audit: list[Mapping[str, Any]] = field(default_factory=list, init=False, repr=False)
    _particle_step_audit: list[Mapping[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.step_cap_A <= 0:
            raise ValueError("density translation step cap must be positive")
        if not 0.0 <= self.start_fraction < 1.0:
            raise ValueError("density translation start must be in [0,1)")
        if self.min_ncc_gain < 0:
            raise ValueError("minimum NCC gain must be non-negative")
        if not 0.0 <= self.min_in_bounds_fraction <= 1.0:
            raise ValueError("minimum in-bounds fraction must be in [0,1]")
        if not 0.0 <= self.protected_fraction < 1.0:
            raise ValueError("protected fraction must be in [0,1)")
        if self.ncc_sigma_A <= 0 or self.ncc_mask_radius_A <= 0:
            raise ValueError("NCC sigma and mask radius must be positive")
        if self.ncc_chunk_size <= 0:
            raise ValueError("NCC chunk size must be positive")
        self._helper = LigandDensityGuidance(
            crop=self.crop,
            lambda_translation=0.0,
            lambda_rotation=0.0,
            start_fraction=0.0,
            objective="map_ncc",
            ncc_sigma_A=self.ncc_sigma_A,
            ncc_mask_radius_A=self.ncc_mask_radius_A,
            ncc_chunk_size=self.ncc_chunk_size,
        )

    def _initialize_particles(self, count: int) -> None:
        if self._num_particles is not None:
            if self._num_particles != int(count):
                raise ValueError(
                    f"Density corrector particle count changed from "
                    f"{self._num_particles} to {count}"
                )
            return
        self._num_particles = int(count)
        self._protected_count = int(round(float(count) * self.protected_fraction))
        self._cumulative_path_A = np.zeros(count, dtype=np.float64)
        self._net_vectors_A = np.zeros((count, 3), dtype=np.float64)
        self._accepted_steps = np.zeros(count, dtype=np.int64)
        self._active_steps = np.zeros(count, dtype=np.int64)
        self._cumulative_ncc_gain = np.zeros(count, dtype=np.float64)
        self._last_ncc = np.full(count, np.nan, dtype=np.float64)

    @staticmethod
    def _heavy_coordinates_and_weights(graph: Any, device: torch.device):
        coordinates_abs = (
            graph["ligand"].pos.detach().to(device)
            + graph.original_center.detach().to(device)
        )
        atomic_index = graph["ligand"].x[:, 0].detach().to(device).long()
        heavy = atomic_index != 0
        if not bool(heavy.any()):
            raise ValueError("NCC translation correction requires ligand heavy atoms")
        weights = torch.where(
            atomic_index[heavy] < 118,
            atomic_index[heavy] + 1,
            torch.full_like(atomic_index[heavy], 6),
        ).to(torch.float32)
        return coordinates_abs[heavy], weights

    def apply(
        self,
        data_list: Sequence[Any],
        model_translation_perturb: torch.Tensor,
        step_index: int,
        inference_steps: int,
        device: torch.device,
    ) -> Sequence[Any]:
        progress = float(step_index + 1) / float(inference_steps)
        if progress <= self.start_fraction:
            return data_list

        self._initialize_particles(len(data_list))
        coordinates_list = []
        atomic_weights = None
        for graph in data_list:
            coordinates, weights = self._heavy_coordinates_and_weights(graph, device)
            if atomic_weights is None:
                atomic_weights = weights
            elif not torch.equal(weights, atomic_weights):
                raise ValueError("Particle ligand atom composition/order is inconsistent")
            coordinates_list.append(coordinates)
        if len({int(value.shape[0]) for value in coordinates_list}) != 1:
            raise ValueError("Particle ligand heavy-atom counts are inconsistent")
        coordinates = torch.stack(coordinates_list, dim=0)
        ncc_points, ncc_target = self._helper._ncc_tensors(device)
        density, origin, spacing = self._helper._tensors(device)

        with torch.enable_grad():
            differentiable = coordinates.detach().clone().requires_grad_(True)
            before = gaussian_ncc_batch(
                ncc_points,
                differentiable,
                atomic_weights,
                ncc_target,
                sigma_A=self.ncc_sigma_A,
                chunk_size=self.ncc_chunk_size,
            )
            coordinate_gradient = torch.autograd.grad(
                before.sum(), differentiable, create_graph=False
            )[0]
        translation_gradient = coordinate_gradient.detach().sum(dim=1)
        gradient_norm = torch.linalg.vector_norm(translation_gradient, dim=1)
        usable = gradient_norm > 1e-12
        direction = translation_gradient / gradient_norm.clamp_min(1e-12)[:, None]
        proposal = coordinates + float(self.step_cap_A) * direction[:, None, :]

        with torch.no_grad():
            proposal_ncc = gaussian_ncc_batch(
                ncc_points,
                proposal,
                atomic_weights,
                ncc_target,
                sigma_A=self.ncc_sigma_A,
                chunk_size=self.ncc_chunk_size,
            )
            ncc_gain = proposal_ncc - before.detach()
            shape = torch.as_tensor(
                density.shape, dtype=proposal.dtype, device=proposal.device
            )
            upper = origin + (shape - 1.0) * spacing
            proposal_in_bounds = (
                ((proposal >= origin) & (proposal <= upper)).all(dim=-1)
                .float().mean(dim=1)
            )
            protected = torch.zeros(len(data_list), dtype=torch.bool, device=device)
            if self._protected_count:
                protected[:self._protected_count] = True
            accepted = (
                (~protected)
                & usable
                & (ncc_gain > float(self.min_ncc_gain))
                & (proposal_in_bounds >= float(self.min_in_bounds_fraction))
            )
            corrections = (
                accepted.to(direction.dtype)[:, None]
                * float(self.step_cap_A)
                * direction
            )

        model_translation_perturb = model_translation_perturb.detach().cpu()
        if model_translation_perturb.shape != (len(data_list), 3):
            raise ValueError(
                f"Expected model translation perturb shape {(len(data_list), 3)}, "
                f"got {tuple(model_translation_perturb.shape)}"
            )
        for particle_index, (graph, correction) in enumerate(zip(data_list, corrections)):
            graph["ligand"].pos = graph["ligand"].pos + correction.to(
                device=graph["ligand"].pos.device,
                dtype=graph["ligand"].pos.dtype,
            )
            correction_np = correction.detach().cpu().numpy().astype(np.float64)
            accepted_value = int(accepted[particle_index].detach().cpu())
            gain_value = float(ncc_gain[particle_index].detach().cpu()) if accepted_value else 0.0
            self._active_steps[particle_index] += int(not bool(protected[particle_index]))
            self._accepted_steps[particle_index] += accepted_value
            self._cumulative_path_A[particle_index] += float(np.linalg.norm(correction_np))
            self._net_vectors_A[particle_index] += correction_np
            self._cumulative_ncc_gain[particle_index] += gain_value
            self._last_ncc[particle_index] = float(
                (proposal_ncc if accepted_value else before.detach())[particle_index].cpu()
            )
            self._particle_step_audit.append({
                "step": int(step_index + 1),
                "progress": progress,
                "particle_index": int(particle_index),
                "protected": bool(protected[particle_index].detach().cpu()),
                "accepted": bool(accepted_value),
                "ncc_before": float(before[particle_index].detach().cpu()),
                "ncc_proposal": float(proposal_ncc[particle_index].detach().cpu()),
                "ncc_gain_proposal": float(ncc_gain[particle_index].detach().cpu()),
                "translation_gradient_norm": float(gradient_norm[particle_index].cpu()),
                "model_translation_perturb_norm_A": float(
                    torch.linalg.vector_norm(model_translation_perturb[particle_index])
                ),
                "correction_norm_A": float(np.linalg.norm(correction_np)),
                "cumulative_path_A": float(self._cumulative_path_A[particle_index]),
                "net_translation_A": float(
                    np.linalg.norm(self._net_vectors_A[particle_index])
                ),
                "proposal_in_bounds_fraction": float(
                    proposal_in_bounds[particle_index].detach().cpu()
                ),
            })

        model_norms = torch.linalg.vector_norm(model_translation_perturb, dim=1).numpy()
        correction_norms = torch.linalg.vector_norm(corrections, dim=1).cpu().numpy()
        accepted_np = accepted.cpu().numpy()
        active_np = (~protected).cpu().numpy()
        self._step_audit.append({
            "step": int(step_index + 1),
            "progress": progress,
            "num_particles": int(len(data_list)),
            "num_protected": int(protected.sum().cpu()),
            "num_active": int(active_np.sum()),
            "num_accepted": int(accepted_np.sum()),
            "acceptance_pct_of_active": float(
                100.0 * accepted_np.sum() / max(active_np.sum(), 1)
            ),
            "mean_ncc_before": float(before.detach().mean().cpu()),
            "mean_ncc_gain_proposal": float(ncc_gain.detach().mean().cpu()),
            "mean_translation_gradient_norm": float(gradient_norm.mean().cpu()),
            "mean_model_translation_perturb_norm_A": float(model_norms.mean()),
            "mean_correction_norm_A": float(correction_norms.mean()),
            "mean_correction_to_model_ratio": float(
                np.mean(correction_norms / np.maximum(model_norms, 1e-8))
            ),
            "mean_cumulative_path_A": float(self._cumulative_path_A.mean()),
            "mean_net_translation_A": float(
                np.linalg.norm(self._net_vectors_A, axis=1).mean()
            ),
            "mean_proposal_in_bounds_fraction": float(
                proposal_in_bounds.mean().cpu()
            ),
        })
        return data_list

    def summary(self) -> Mapping[str, Any]:
        particles = []
        if self._num_particles is not None:
            for particle_index in range(self._num_particles):
                particles.append({
                    "particle_index": particle_index,
                    "protected": particle_index < self._protected_count,
                    "active_steps": int(self._active_steps[particle_index]),
                    "accepted_steps": int(self._accepted_steps[particle_index]),
                    "cumulative_path_A": float(self._cumulative_path_A[particle_index]),
                    "net_translation_vector_A": self._net_vectors_A[particle_index].tolist(),
                    "net_translation_A": float(
                        np.linalg.norm(self._net_vectors_A[particle_index])
                    ),
                    "cumulative_ncc_gain": float(
                        self._cumulative_ncc_gain[particle_index]
                    ),
                    "last_ncc": float(self._last_ncc[particle_index]),
                })
        return {
            "enabled": True,
            "mechanism": "post_model_fixed_step_translation_ncc",
            "crop_path": self.crop.path,
            "step_cap_A": float(self.step_cap_A),
            "start_fraction": float(self.start_fraction),
            "min_ncc_gain": float(self.min_ncc_gain),
            "min_in_bounds_fraction": float(self.min_in_bounds_fraction),
            "protected_fraction": float(self.protected_fraction),
            "protected_count": int(self._protected_count),
            "ncc_sigma_A": float(self.ncc_sigma_A),
            "ncc_mask_radius_A": float(self.ncc_mask_radius_A),
            "ncc_chunk_size": int(self.ncc_chunk_size),
            "num_active_diffusion_steps": len(self._step_audit),
            "steps": list(self._step_audit),
            "particles": particles,
            "particle_steps": list(self._particle_step_audit),
        }

    def write_audit(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary(), indent=2) + "\n")


def crop_path_from_row(row: Mapping[str, Any], cli_path: Optional[str] = None) -> Optional[str]:
    """Resolve single-input CLI or batch-CSV crop path without reading GT."""
    if _optional_text(cli_path):
        return _optional_text(cli_path)
    for column in ("ligand_density_crop", "ligand_density_crop_path", "density_crop_path"):
        try:
            value = row[column]
        except (KeyError, TypeError):
            continue
        text = _optional_text(value)
        if text:
            return text
    return None
