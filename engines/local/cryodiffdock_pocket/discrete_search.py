"""Deterministic local pose proposals for density-guided ligand search.

The helpers in this module deliberately do not read a reference pose or a
density map.  They only expand an existing RDKit conformer by applying
controlled rigid-body and single-bond torsion moves.  Scoring and deployment
gates live in ``scripts/run_discrete_density_search.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LocalVariant:
    """One deterministic local proposal around a parent conformer."""

    variant_id: str
    move_type: str
    coordinates: np.ndarray
    magnitude: float
    axis_label: str = ""
    bond: tuple[int, int] | None = None


def _as_coordinates(coordinates: np.ndarray) -> np.ndarray:
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"coordinates must have shape (N,3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("coordinates contain non-finite values")
    return array


def axis_angle_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Return a right-handed Rodrigues rotation matrix."""

    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("rotation axis has near-zero norm")
    x, y, z = axis / norm
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.deg2rad(float(angle_degrees))
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotate_about_point(
    coordinates: np.ndarray,
    axis: np.ndarray,
    angle_degrees: float,
    centre: np.ndarray,
) -> np.ndarray:
    coordinates = _as_coordinates(coordinates)
    centre = np.asarray(centre, dtype=np.float64).reshape(1, 3)
    matrix = axis_angle_matrix(axis, angle_degrees)
    return ((coordinates - centre) @ matrix.T + centre).astype(np.float64)


def principal_axes(coordinates: np.ndarray) -> np.ndarray:
    """Return three deterministic orthonormal body axes.

    Singular vectors have an arbitrary sign.  Canonicalising the largest
    absolute component makes proposal identifiers and coordinates reproducible.
    The search includes both positive and negative angles, so the convention
    does not privilege a physical direction.
    """

    coordinates = _as_coordinates(coordinates)
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=True)
    axes = vh.copy()
    for index in range(3):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0:
            axes[index] *= -1.0
    # Restore a right-handed frame after independent sign canonicalisation.
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1.0
    return axes


def proper_signed_permutation_rotations() -> list[np.ndarray]:
    """Return the 24 proper rotations of the Cartesian cube.

    This inexpensive orientation library supplies large, discrete basin jumps
    that a weak local NCC gradient cannot make.  The identity is returned
    first and callers can omit it when the unmodified parent is already kept.
    """

    import itertools

    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            for row, column in enumerate(permutation):
                matrix[row, column] = signs[row]
            if np.linalg.det(matrix) > 0.5:
                matrices.append(matrix)
    identity_index = next(
        index for index, matrix in enumerate(matrices)
        if np.allclose(matrix, np.eye(3))
    )
    identity = matrices.pop(identity_index)
    matrices.sort(key=lambda matrix: tuple(matrix.reshape(-1).tolist()))
    return [identity, *matrices]


def direct_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    left = _as_coordinates(left)
    right = _as_coordinates(right)
    if left.shape != right.shape:
        raise ValueError(f"coordinate shapes differ: {left.shape} != {right.shape}")
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def centroid_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = _as_coordinates(left)
    right = _as_coordinates(right)
    return float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))


def _component_after_cut(molecule, start: int, blocked: int) -> set[int]:
    """Return the atom component reached from ``start`` without crossing a bond."""

    visited: set[int] = set()
    stack = [int(start)]
    while stack:
        atom_index = stack.pop()
        if atom_index in visited:
            continue
        visited.add(atom_index)
        atom = molecule.GetAtomWithIdx(atom_index)
        for neighbour in atom.GetNeighbors():
            neighbour_index = int(neighbour.GetIdx())
            if {atom_index, neighbour_index} == {int(start), int(blocked)}:
                continue
            if neighbour_index not in visited:
                stack.append(neighbour_index)
    return visited


def rotatable_bonds(molecule) -> list[tuple[int, int, np.ndarray]]:
    """Return acyclic heavy-atom rotors and the smaller rotated component."""

    from rdkit import Chem

    prepared = Chem.Mol(molecule)
    try:
        prepared.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(prepared)
    except Exception:
        pass

    bonds: list[tuple[int, int, np.ndarray]] = []
    for bond in prepared.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.GetIsAromatic():
            continue
        try:
            if bond.IsInRing():
                continue
        except RuntimeError:
            continue
        u, v = int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx())
        atom_u, atom_v = prepared.GetAtomWithIdx(u), prepared.GetAtomWithIdx(v)
        if atom_u.GetAtomicNum() <= 1 or atom_v.GetAtomicNum() <= 1:
            continue
        if atom_u.GetDegree() <= 1 or atom_v.GetDegree() <= 1:
            continue

        side_v = _component_after_cut(prepared, v, u)
        side_u = _component_after_cut(prepared, u, v)
        if u in side_v or v in side_u:
            # The edge was not a graph bridge (normally a ring missed above).
            continue
        if len(side_v) <= len(side_u):
            fixed, moving, component = u, v, side_v
        else:
            fixed, moving, component = v, u, side_u
        # The axis atom itself need not be rewritten, but retaining it in the
        # mask is harmless because it lies on the rotation axis.
        mask = np.zeros(prepared.GetNumAtoms(), dtype=bool)
        mask[list(component)] = True
        bonds.append((fixed, moving, mask))
    return bonds


def rotate_torsion_component(
    coordinates: np.ndarray,
    fixed_atom: int,
    moving_atom: int,
    rotate_mask: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    coordinates = _as_coordinates(coordinates)
    rotate_mask = np.asarray(rotate_mask, dtype=bool)
    if rotate_mask.shape != (len(coordinates),):
        raise ValueError("rotate_mask must contain one value per atom")
    axis = coordinates[moving_atom] - coordinates[fixed_atom]
    updated = coordinates.copy()
    updated[rotate_mask] = rotate_about_point(
        coordinates[rotate_mask],
        axis=axis,
        angle_degrees=angle_degrees,
        centre=coordinates[moving_atom],
    )
    return updated


def _signed(values: Sequence[float]) -> list[float]:
    result = []
    for value in values:
        value = abs(float(value))
        if value <= 0:
            raise ValueError("proposal magnitudes must be positive")
        result.extend((-value, value))
    return result


def generate_local_variants(
    molecule,
    *,
    rotation_degrees: Sequence[float] = (15.0, 30.0, 60.0),
    translation_distances_A: Sequence[float] = (0.5, 1.0),
    torsion_degrees: Sequence[float] = (60.0, 120.0, 180.0),
    max_torsion_bonds: int = 8,
    extra_translation_direction: np.ndarray | None = None,
    include_coarse_orientations: bool = True,
) -> list[LocalVariant]:
    """Enumerate one-move proposals around the molecule's current conformer."""

    if molecule.GetNumConformers() == 0:
        raise ValueError("molecule has no conformer")
    if max_torsion_bonds < 0:
        raise ValueError("max_torsion_bonds must be non-negative")
    coordinates = _as_coordinates(molecule.GetConformer().GetPositions())
    centre = coordinates.mean(axis=0)
    variants = [LocalVariant("original", "original", coordinates.copy(), 0.0)]

    axes = list(principal_axes(coordinates))
    for axis_index, axis in enumerate(axes):
        for angle in _signed(rotation_degrees):
            variants.append(LocalVariant(
                variant_id=f"rot_a{axis_index}_{angle:+g}",
                move_type="rotation",
                coordinates=rotate_about_point(coordinates, axis, angle, centre),
                magnitude=float(angle),
                axis_label=f"principal_{axis_index}",
            ))

    if include_coarse_orientations:
        centered = coordinates - centre.reshape(1, 3)
        for orientation_index, matrix in enumerate(
            proper_signed_permutation_rotations()[1:], start=1
        ):
            variants.append(LocalVariant(
                variant_id=f"orient_cube_{orientation_index:02d}",
                move_type="coarse_orientation",
                coordinates=centered @ matrix.T + centre.reshape(1, 3),
                magnitude=float(orientation_index),
                axis_label="cartesian_cube_group",
            ))

    translation_axes = [np.eye(3, dtype=np.float64)[index] for index in range(3)]
    if extra_translation_direction is not None:
        direction = np.asarray(extra_translation_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction /= norm
            if all(abs(float(np.dot(direction, axis))) < 0.999 for axis in translation_axes):
                translation_axes.append(direction)
    for axis_index, axis in enumerate(translation_axes):
        for distance in _signed(translation_distances_A):
            variants.append(LocalVariant(
                variant_id=f"tr_a{axis_index}_{distance:+g}",
                move_type="translation",
                coordinates=coordinates + float(distance) * axis.reshape(1, 3),
                magnitude=float(distance),
                axis_label=(
                    f"cartesian_{axis_index}" if axis_index < 3 else "toward_crop_center"
                ),
            ))

    for bond_index, (fixed, moving, mask) in enumerate(
        rotatable_bonds(molecule)[: int(max_torsion_bonds)]
    ):
        for angle in _signed(torsion_degrees):
            variants.append(LocalVariant(
                variant_id=f"tor_b{bond_index}_{fixed}-{moving}_{angle:+g}",
                move_type="torsion",
                coordinates=rotate_torsion_component(
                    coordinates, fixed, moving, mask, angle
                ),
                magnitude=float(angle),
                axis_label=f"bond_{bond_index}",
                bond=(int(fixed), int(moving)),
            ))
    return variants


def greedy_diverse_indices(
    coordinate_sets: Sequence[np.ndarray],
    scores: Sequence[float],
    *,
    count: int,
    top_score_count: int,
    minimum_rmsd_A: float,
) -> list[int]:
    """Select high-score parents, then add geometrically distinct parents."""

    if count <= 0 or top_score_count < 0 or minimum_rmsd_A < 0:
        raise ValueError("invalid diversity-selection parameters")
    if len(coordinate_sets) != len(scores):
        raise ValueError("coordinate and score counts differ")
    if not coordinate_sets:
        return []
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    selected = order[: min(int(top_score_count), int(count), len(order))]
    for index in order:
        if index in selected:
            continue
        if all(
            direct_rmsd(coordinate_sets[index], coordinate_sets[other])
            >= float(minimum_rmsd_A)
            for other in selected
        ):
            selected.append(index)
            if len(selected) >= min(int(count), len(order)):
                return selected
    for index in order:
        if index not in selected:
            selected.append(index)
            if len(selected) >= min(int(count), len(order)):
                break
    return selected
