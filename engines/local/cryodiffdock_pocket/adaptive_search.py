"""Density-conditioned hierarchical pose search primitives.

This module is the V2 counterpart to :mod:`cryodiffdock_pocket.discrete_search`.
V1 enumerates a fixed one-move library and uses density only after generation.
V2 keeps the deterministic and ground-truth-free character of that search, but
uses differentiable multi-scale Gaussian NCC to choose translation, rotation,
and torsion proposal directions.  Proposals can be composed over several beam
iterations without materialising an exponentially large Cartesian product.

The module contains no reference-pose or RMSD code.  RMSD belongs exclusively
to downstream evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from cryodiffdock_pocket.density_guidance import gaussian_ncc_batch
from cryodiffdock_pocket.density_ranking import (
    coordinates_in_bounds_fraction,
    fixed_crop_mask,
    normalized_target,
)
from cryodiffdock_pocket.discrete_search import (
    principal_axes,
    proper_signed_permutation_rotations,
    rotate_about_point,
    rotate_torsion_component,
    rotatable_bonds,
)
from cryodiffdock_pocket.physics import atom_radii


@dataclass
class AdaptivePose:
    """One node in the bounded V2 search tree."""

    candidate_id: str
    root_id: str
    parent_id: str
    coordinates: np.ndarray
    depth: int
    move_type: str
    move_magnitude: float
    axis_label: str = ""
    bond: tuple[int, int] | None = None
    branch_type: str = "local"
    move_sequence: tuple[str, ...] = field(default_factory=tuple)
    primary_ncc: float = float("nan")
    multiscale_ncc: float = float("nan")
    ncc_by_sigma: tuple[float, ...] = field(default_factory=tuple)
    in_bounds_fraction: float = float("nan")
    translation_gradient_norm: float = float("nan")
    rotation_gradient_norm: float = float("nan")
    maximum_torsion_gradient: float = float("nan")
    severe_clash_count: int = 0
    clashing_ligand_atom_fraction: float = 0.0
    selection_score: float = float("nan")
    basin_q25_ncc: float = float("nan")
    basin_support_fraction: float = float("nan")

    def copy_with(self, **updates) -> "AdaptivePose":
        values = dict(self.__dict__)
        values.update(updates)
        return AdaptivePose(**values)


def coordinate_fingerprint(
    coordinates: np.ndarray,
    tolerance_A: float = 1e-3,
) -> bytes:
    """Return a stable quantised fingerprint for exact/near-exact duplicates."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(f"coordinates must be (N,3), got {coordinates.shape}")
    if tolerance_A <= 0:
        raise ValueError("fingerprint tolerance must be positive")
    quantised = np.rint(coordinates / float(tolerance_A)).astype(np.int64)
    return quantised.tobytes()


def deduplicate_poses(
    poses: Iterable[AdaptivePose],
    tolerance_A: float = 1e-3,
) -> list[AdaptivePose]:
    """Deduplicate coordinates within each independent root hypothesis."""

    result: list[AdaptivePose] = []
    seen: set[tuple[str, bytes]] = set()
    for pose in poses:
        key = (pose.root_id, coordinate_fingerprint(pose.coordinates, tolerance_A))
        if key in seen:
            continue
        seen.add(key)
        result.append(pose)
    return result


class MultiScaleNCCEvaluator:
    """Batched differentiable NCC evaluator on one fixed, pose-independent mask."""

    def __init__(
        self,
        crop,
        atomic_numbers: Sequence[float],
        *,
        sigmas_A: Sequence[float] = (1.5, 2.25),
        mask_radius_A: float = 10.0,
        target_floor: float | None = None,
        min_mask_voxels: int = 64,
        device: torch.device | str = "cpu",
        voxel_chunk_size: int = 512,
        pose_batch_size: int = 64,
    ) -> None:
        sigmas = tuple(float(value) for value in sigmas_A)
        if not sigmas or any(value <= 0 for value in sigmas):
            raise ValueError("sigmas_A must contain positive values")
        if voxel_chunk_size <= 0 or pose_batch_size <= 0:
            raise ValueError("chunk and batch sizes must be positive")
        self.crop = crop
        self.sigmas_A = sigmas
        self.device = torch.device(device)
        self.voxel_chunk_size = int(voxel_chunk_size)
        self.pose_batch_size = int(pose_batch_size)
        self.atomic_weights = torch.as_tensor(
            np.asarray(atomic_numbers, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        mask = fixed_crop_mask(
            crop,
            radius_A=float(mask_radius_A),
            target_floor=target_floor,
            min_voxels=int(min_mask_voxels),
        )
        indices = np.argwhere(mask).astype(np.float32)
        points = np.asarray(crop.origin, dtype=np.float32)[None, :] + (
            indices * np.asarray(crop.spacing, dtype=np.float32)[None, :]
        )
        target = normalized_target(crop)[mask]
        self.sample_points = torch.as_tensor(
            points, dtype=torch.float32, device=self.device
        )
        self.target_density = torch.as_tensor(
            target, dtype=torch.float32, device=self.device
        )

    @property
    def primary_sigma_A(self) -> float:
        return self.sigmas_A[0]

    def _tensor(self, coordinates: Sequence[np.ndarray] | np.ndarray) -> torch.Tensor:
        array = np.asarray(coordinates, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"coordinates must be (B,N,3), got {array.shape}")
        if array.shape[1] != self.atomic_weights.numel():
            raise ValueError(
                f"coordinate atom count {array.shape[1]} does not match "
                f"weight count {self.atomic_weights.numel()}"
            )
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _score_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        columns = [
            gaussian_ncc_batch(
                self.sample_points,
                coordinates,
                self.atomic_weights,
                self.target_density,
                sigma_A=sigma,
                chunk_size=self.voxel_chunk_size,
            )
            for sigma in self.sigmas_A
        ]
        return torch.stack(columns, dim=1)

    def score(self, coordinates: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Return an array shaped ``(B, num_sigmas)``."""

        tensor = self._tensor(coordinates)
        chunks = []
        with torch.no_grad():
            for start in range(0, len(tensor), self.pose_batch_size):
                chunks.append(
                    self._score_tensor(tensor[start:start + self.pose_batch_size])
                    .detach().cpu().numpy()
                )
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(self.sigmas_A)))

    def score_and_gradient(
        self,
        coordinates: Sequence[np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-sigma NCC and Cartesian gradient of their arithmetic mean."""

        tensor = self._tensor(coordinates)
        score_chunks: list[np.ndarray] = []
        gradient_chunks: list[np.ndarray] = []
        for start in range(0, len(tensor), self.pose_batch_size):
            local = tensor[start:start + self.pose_batch_size].detach().clone()
            local.requires_grad_(True)
            scores = self._score_tensor(local)
            gradient = torch.autograd.grad(scores.mean(dim=1).sum(), local)[0]
            score_chunks.append(scores.detach().cpu().numpy())
            gradient_chunks.append(gradient.detach().cpu().numpy())
        if not score_chunks:
            return (
                np.empty((0, len(self.sigmas_A)), dtype=np.float32),
                np.empty_like(np.asarray(coordinates, dtype=np.float32)),
            )
        return np.concatenate(score_chunks), np.concatenate(gradient_chunks)

    def in_bounds_fraction(self, coordinates: np.ndarray) -> float:
        return coordinates_in_bounds_fraction(coordinates, self.crop)


class ProteinClashIndex:
    """Exact severe-clash counts using a reusable protein spatial index.

    V1 constructs the complete protein-by-ligand distance matrix for every
    candidate.  Radius queries return the same severe-pair count while avoiding
    work on protein atoms that cannot possibly clash with a ligand atom.
    """

    def __init__(self, coordinates: np.ndarray, elements: Sequence[str]) -> None:
        self.coordinates = np.asarray(coordinates, dtype=np.float64)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("protein coordinates must be (N,3)")
        self.radii = atom_radii(elements)
        if len(self.radii) != len(self.coordinates):
            raise ValueError("protein element and coordinate counts differ")
        self.tree = cKDTree(self.coordinates)
        self.maximum_radius = float(self.radii.max())

    def score(
        self,
        ligand_coordinates: np.ndarray,
        ligand_elements: Sequence[str],
        *,
        severe_overlap_A: float = 0.8,
    ) -> tuple[int, float]:
        coordinates = np.asarray(ligand_coordinates, dtype=np.float64)
        ligand_radii = atom_radii(ligand_elements)
        if coordinates.shape != (len(ligand_radii), 3):
            raise ValueError("ligand element and coordinate counts differ")
        severe_count = 0
        clashing = np.zeros(len(coordinates), dtype=bool)
        for atom_index, (coordinate, radius) in enumerate(zip(coordinates, ligand_radii)):
            broad_radius = max(
                0.0,
                self.maximum_radius + float(radius) - float(severe_overlap_A),
            )
            neighbors = self.tree.query_ball_point(coordinate, broad_radius)
            if not neighbors:
                continue
            neighbor_indices = np.asarray(neighbors, dtype=int)
            distances = np.linalg.norm(
                self.coordinates[neighbor_indices] - coordinate[None, :], axis=1
            )
            thresholds = (
                self.radii[neighbor_indices] + float(radius) - float(severe_overlap_A)
            )
            count = int(np.sum(distances < thresholds))
            severe_count += count
            clashing[atom_index] = count > 0
        fraction = float(clashing.mean()) if len(clashing) else 0.0
        return severe_count, fraction


def rigid_gradient_components(
    coordinates: np.ndarray,
    coordinate_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a Cartesian NCC gradient onto translation and rotation tangent space."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    gradient = np.asarray(coordinate_gradient, dtype=np.float64)
    if coordinates.shape != gradient.shape or coordinates.ndim != 2:
        raise ValueError("coordinates and gradient must have identical (N,3) shape")
    translation = gradient.sum(axis=0)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    rotation = np.cross(centered, gradient).sum(axis=0)
    return translation, rotation


def torsion_gradient_components(
    coordinates: np.ndarray,
    coordinate_gradient: np.ndarray,
    torsions: Sequence[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Project a Cartesian gradient onto this module's positive torsion convention."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    gradient = np.asarray(coordinate_gradient, dtype=np.float64)
    values = []
    for fixed, moving, mask in torsions:
        axis = coordinates[moving] - coordinates[fixed]
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            values.append(0.0)
            continue
        axis /= norm
        relative = coordinates[np.asarray(mask, dtype=bool)] - coordinates[moving]
        torque = np.cross(relative, gradient[np.asarray(mask, dtype=bool)]).sum(axis=0)
        values.append(float(np.dot(axis, torque)))
    return np.asarray(values, dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    return None if norm <= 1e-12 else vector / norm


def coarse_orientation_proposals(
    parent: AdaptivePose,
    *,
    include_cube_group: bool = True,
) -> list[AdaptivePose]:
    """Generate the density-independent coverage branch used before screening."""

    if not include_cube_group:
        return []
    coordinates = np.asarray(parent.coordinates, dtype=np.float64)
    center = coordinates.mean(axis=0)
    centered = coordinates - center
    result = []
    for index, matrix in enumerate(proper_signed_permutation_rotations()[1:], start=1):
        move_id = f"coarse{index:02d}"
        result.append(AdaptivePose(
            candidate_id=f"{parent.candidate_id}__{move_id}",
            root_id=parent.root_id,
            parent_id=parent.candidate_id,
            coordinates=centered @ matrix.T + center,
            depth=parent.depth + 1,
            move_type="coarse_orientation",
            move_magnitude=float(index),
            axis_label="cartesian_cube_group",
            branch_type="coarse",
            move_sequence=(*parent.move_sequence, move_id),
        ))
    return result


def density_conditioned_proposals(
    parent: AdaptivePose,
    coordinate_gradient: np.ndarray,
    molecule,
    *,
    translation_steps_A: Sequence[float] = (0.25, 0.5),
    rotation_steps_degrees: Sequence[float] = (10.0, 20.0),
    torsion_steps_degrees: Sequence[float] = (30.0, 60.0),
    max_torsion_bonds: int = 8,
    top_torsion_gradients: int = 2,
    include_combined_rigid: bool = True,
) -> tuple[list[AdaptivePose], dict[str, float]]:
    """Propose ascent moves along NCC translation, torque, and torsion gradients."""

    coordinates = np.asarray(parent.coordinates, dtype=np.float64)
    translation, rotation = rigid_gradient_components(coordinates, coordinate_gradient)
    translation_direction = _unit(translation)
    rotation_axis = _unit(rotation)
    torsions = rotatable_bonds(molecule)[: int(max_torsion_bonds)]
    torsion_values = torsion_gradient_components(
        coordinates, coordinate_gradient, torsions
    ) if torsions else np.empty(0, dtype=np.float64)
    diagnostics = {
        "translation_gradient_norm": float(np.linalg.norm(translation)),
        "rotation_gradient_norm": float(np.linalg.norm(rotation)),
        "maximum_torsion_gradient": (
            float(np.max(np.abs(torsion_values))) if len(torsion_values) else 0.0
        ),
    }
    proposals: list[AdaptivePose] = []

    def append(
        move_id: str,
        move_type: str,
        updated: np.ndarray,
        magnitude: float,
        axis_label: str,
        bond: tuple[int, int] | None = None,
    ) -> None:
        proposals.append(AdaptivePose(
            candidate_id=f"{parent.candidate_id}__{move_id}",
            root_id=parent.root_id,
            parent_id=parent.candidate_id,
            coordinates=np.asarray(updated, dtype=np.float64),
            depth=parent.depth + 1,
            move_type=move_type,
            move_magnitude=float(magnitude),
            axis_label=axis_label,
            bond=bond,
            branch_type=parent.branch_type,
            move_sequence=(*parent.move_sequence, move_id),
            **diagnostics,
        ))

    if translation_direction is not None:
        for step in translation_steps_A:
            step = float(step)
            append(
                f"gtr_{step:g}", "translation",
                coordinates + step * translation_direction[None, :], step,
                "density_gradient",
            )

    rotated_by_step: dict[float, np.ndarray] = {}
    if rotation_axis is not None:
        center = coordinates.mean(axis=0)
        for angle in rotation_steps_degrees:
            angle = float(angle)
            rotated = rotate_about_point(coordinates, rotation_axis, angle, center)
            rotated_by_step[angle] = rotated
            append(
                f"grot_{angle:g}", "rotation", rotated, angle,
                "density_torque",
            )

    if include_combined_rigid and translation_direction is not None and rotated_by_step:
        for angle, rotated in rotated_by_step.items():
            for step in translation_steps_A:
                step = float(step)
                append(
                    f"grigid_r{angle:g}_t{step:g}", "combined_rigid",
                    rotated + step * translation_direction[None, :],
                    float(np.hypot(angle, step)), "density_gradient+torque",
                )

    if len(torsion_values) and top_torsion_gradients > 0:
        order = np.argsort(-np.abs(torsion_values))[: int(top_torsion_gradients)]
        for torsion_index in order:
            derivative = float(torsion_values[torsion_index])
            if abs(derivative) <= 1e-12:
                continue
            fixed, moving, mask = torsions[int(torsion_index)]
            sign = 1.0 if derivative > 0 else -1.0
            for magnitude in torsion_steps_degrees:
                # +180 and -180 are identical; a gradient sign remains useful
                # in the path audit but only one geometry is generated.
                signed = sign * abs(float(magnitude))
                updated = rotate_torsion_component(
                    coordinates, fixed, moving, mask, signed
                )
                append(
                    f"gtor_b{int(torsion_index)}_{signed:+g}", "torsion",
                    updated, signed, f"density_torsion_{int(torsion_index)}",
                    (int(fixed), int(moving)),
                )

    return deduplicate_poses(proposals), diagnostics


def fixed_direction_proposals(
    parent: AdaptivePose,
    molecule,
    *,
    translation_steps_A: Sequence[float] = (0.25, 0.5),
    rotation_steps_degrees: Sequence[float] = (10.0, 20.0),
    torsion_steps_degrees: Sequence[float] = (30.0, 60.0),
    max_torsion_bonds: int = 8,
    top_torsion_gradients: int = 2,
    include_combined_rigid: bool = True,
) -> tuple[list[AdaptivePose], dict[str, float]]:
    """Generate a proposal-count-matched, density-gradient-free control.

    The first two deterministic ligand principal axes replace the NCC
    translation and torque directions.  The first rotatable bonds replace the
    gradient-ranked torsions.  Candidate scoring and beam selection may still
    use density; only the *proposal directions* are ablated.  For a molecule
    with enough rotors, this emits the same number of local proposals as
    :func:`density_conditioned_proposals`.
    """

    coordinates = np.asarray(parent.coordinates, dtype=np.float64)
    axes = principal_axes(coordinates)
    translation_direction = axes[0]
    rotation_axis = axes[1]
    torsions = rotatable_bonds(molecule)[: int(max_torsion_bonds)]
    diagnostics = {
        "translation_gradient_norm": 0.0,
        "rotation_gradient_norm": 0.0,
        "maximum_torsion_gradient": 0.0,
    }
    proposals: list[AdaptivePose] = []

    def append(
        move_id: str,
        move_type: str,
        updated: np.ndarray,
        magnitude: float,
        axis_label: str,
        bond: tuple[int, int] | None = None,
    ) -> None:
        proposals.append(AdaptivePose(
            candidate_id=f"{parent.candidate_id}__{move_id}",
            root_id=parent.root_id,
            parent_id=parent.candidate_id,
            coordinates=np.asarray(updated, dtype=np.float64),
            depth=parent.depth + 1,
            move_type=move_type,
            move_magnitude=float(magnitude),
            axis_label=axis_label,
            bond=bond,
            branch_type=parent.branch_type,
            move_sequence=(*parent.move_sequence, move_id),
            **diagnostics,
        ))

    for step_value in translation_steps_A:
        step = float(step_value)
        append(
            f"ftr_{step:g}", "translation",
            coordinates + step * translation_direction[None, :], step,
            "fixed_principal_0",
        )

    center = coordinates.mean(axis=0)
    rotated_by_step: dict[float, np.ndarray] = {}
    for angle_value in rotation_steps_degrees:
        angle = float(angle_value)
        rotated = rotate_about_point(coordinates, rotation_axis, angle, center)
        rotated_by_step[angle] = rotated
        append(
            f"frot_{angle:g}", "rotation", rotated, angle,
            "fixed_principal_1",
        )

    if include_combined_rigid:
        for angle, rotated in rotated_by_step.items():
            for step_value in translation_steps_A:
                step = float(step_value)
                append(
                    f"frigid_r{angle:g}_t{step:g}", "combined_rigid",
                    rotated + step * translation_direction[None, :],
                    float(np.hypot(angle, step)),
                    "fixed_principal_0+1",
                )

    torsion_limit = min(len(torsions), max(0, int(top_torsion_gradients)))
    for torsion_index in range(torsion_limit):
        fixed, moving, mask = torsions[torsion_index]
        for magnitude_value in torsion_steps_degrees:
            magnitude = abs(float(magnitude_value))
            updated = rotate_torsion_component(
                coordinates, fixed, moving, mask, magnitude
            )
            append(
                f"ftor_b{torsion_index}_{magnitude:+g}", "torsion",
                updated, magnitude, f"fixed_torsion_{torsion_index}",
                (int(fixed), int(moving)),
            )

    return deduplicate_poses(proposals), diagnostics


def assign_density_scores(
    poses: Sequence[AdaptivePose],
    evaluator: MultiScaleNCCEvaluator,
) -> None:
    """Populate density and spatial-support fields in-place."""

    if not poses:
        return
    scores = evaluator.score([pose.coordinates for pose in poses])
    for pose, values in zip(poses, scores):
        pose.ncc_by_sigma = tuple(float(value) for value in values)
        pose.primary_ncc = float(values[0])
        pose.multiscale_ncc = float(np.mean(values))
        pose.in_bounds_fraction = evaluator.in_bounds_fraction(pose.coordinates)


def local_basin_statistics(
    pose: AdaptivePose,
    evaluator: MultiScaleNCCEvaluator,
    *,
    translation_step_A: float = 0.20,
    rotation_step_degrees: float = 7.5,
    support_tolerance: float = 0.01,
) -> tuple[float, float]:
    """Measure whether a candidate lies in a stable NCC basin rather than a spike."""

    return local_basin_statistics_batch(
        [pose], evaluator,
        translation_step_A=translation_step_A,
        rotation_step_degrees=rotation_step_degrees,
        support_tolerance=support_tolerance,
    )[pose.candidate_id]


def local_basin_statistics_batch(
    poses: Sequence[AdaptivePose],
    evaluator: MultiScaleNCCEvaluator,
    *,
    translation_step_A: float = 0.20,
    rotation_step_degrees: float = 7.5,
    support_tolerance: float = 0.01,
) -> dict[str, tuple[float, float]]:
    """Batched local-basin statistics for a finalist set."""

    neighbors: list[np.ndarray] = []
    slices: list[tuple[AdaptivePose, int, int]] = []
    for pose in poses:
        coordinates = np.asarray(pose.coordinates, dtype=np.float64)
        center = coordinates.mean(axis=0)
        start = len(neighbors)
        for axis in np.eye(3, dtype=np.float64):
            for sign in (-1.0, 1.0):
                neighbors.append(
                    coordinates + sign * float(translation_step_A) * axis[None, :]
                )
                neighbors.append(rotate_about_point(
                    coordinates, axis, sign * float(rotation_step_degrees), center
                ))
        slices.append((pose, start, len(neighbors)))
    if not neighbors:
        return {}
    all_scores = evaluator.score(neighbors).mean(axis=1)
    result = {}
    for pose, start, end in slices:
        scores = all_scores[start:end]
        q25 = float(np.quantile(scores, 0.25))
        support = float(np.mean(
            scores >= float(pose.multiscale_ncc) - support_tolerance
        ))
        result[pose.candidate_id] = (q25, support)
    return result
