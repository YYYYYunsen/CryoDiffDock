"""Clash-aware local proposal and ranking helpers.

The functions here are reference-free: they use only the predicted ligand and
the supplied receptor.  They are kept separate from the frozen v1 repair so a
new validation-tuned method cannot silently change published results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from cryodiffdock_pocket.discrete_search import (
    LocalVariant,
    direct_rmsd,
    rotate_about_point,
    rotate_torsion_component,
    rotatable_bonds,
)
from cryodiffdock_pocket.physics import atom_radii


@dataclass(frozen=True)
class ClashField:
    """Smooth steric diagnostics and ligand-atom escape vectors."""

    energy: float
    severe_count: int
    clashing_ligand_atom_fraction: float
    max_overlap_A: float
    min_distance_A: float
    atom_forces: np.ndarray
    net_force: np.ndarray
    torque: np.ndarray


def clash_field(
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_coordinates: np.ndarray,
    ligand_elements,
    *,
    severe_overlap_A: float = 0.8,
    softness_A: float = 0.10,
) -> ClashField:
    """Return a differentiable-like clash energy and geometric escape field.

    The force is the negative gradient direction of a squared soft overlap
    penalty.  Only nearby positive overlaps contribute to proposal directions;
    the severe threshold controls the reported discrete clash count.
    """

    protein = np.asarray(protein_coordinates, dtype=np.float64)
    ligand = np.asarray(ligand_coordinates, dtype=np.float64)
    delta = ligand[None, :, :] - protein[:, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    safe_distance = np.maximum(distance, 1e-8)
    overlap = (
        atom_radii(protein_elements)[:, None]
        + atom_radii(ligand_elements)[None, :]
        - distance
    )
    excess = overlap - float(severe_overlap_A)
    severe = excess > 0.0
    softness = max(float(softness_A), 1e-6)
    # Stable softplus, measured in Angstrom, followed by a squared penalty.
    scaled = np.clip(excess / softness, -60.0, 60.0)
    soft = softness * np.logaddexp(0.0, scaled)
    energy = float(np.sum(soft * soft))
    sigmoid = np.where(
        scaled >= 0.0,
        1.0 / (1.0 + np.exp(-scaled)),
        np.exp(scaled) / (1.0 + np.exp(scaled)),
    )
    magnitude = 2.0 * soft * sigmoid
    unit = delta / safe_distance[:, :, None]
    atom_forces = np.sum(magnitude[:, :, None] * unit, axis=0)
    net_force = atom_forces.sum(axis=0)
    centre = ligand.mean(axis=0)
    torque = np.cross(ligand - centre, atom_forces).sum(axis=0)
    return ClashField(
        energy=energy,
        severe_count=int(severe.sum()),
        clashing_ligand_atom_fraction=float(severe.any(axis=0).mean()),
        max_overlap_A=float(overlap.max()),
        min_distance_A=float(distance.min()),
        atom_forces=atom_forces,
        net_force=net_force,
        torque=torque,
    )


def _unit(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-10 else None


def _append_distinct_direction(
    directions: list[np.ndarray], vector: np.ndarray, *, cosine_cutoff: float = 0.97
) -> None:
    direction = _unit(vector)
    if direction is None:
        return
    if all(abs(float(np.dot(direction, other))) < cosine_cutoff for other in directions):
        directions.append(direction)


def directed_local_variants(
    molecule,
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_elements,
    *,
    translation_distances_A: Sequence[float] = (0.05, 0.10, 0.20, 0.35),
    rotation_degrees: Sequence[float] = (1.0, 2.0, 5.0),
    torsion_degrees: Sequence[float] = (2.0, 5.0, 10.0),
    max_contact_directions: int = 4,
    max_torsion_bonds: int = 8,
    severe_overlap_A: float = 0.8,
) -> list[LocalVariant]:
    """Generate translations, rotations, and torsions aimed away from clashes."""

    coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    field = clash_field(
        protein_coordinates,
        protein_elements,
        coordinates,
        ligand_elements,
        severe_overlap_A=severe_overlap_A,
    )
    variants: list[LocalVariant] = []
    directions: list[np.ndarray] = []
    _append_distinct_direction(directions, field.net_force)
    atom_order = np.argsort(-np.linalg.norm(field.atom_forces, axis=1))
    for atom_index in atom_order:
        _append_distinct_direction(directions, field.atom_forces[int(atom_index)])
        if len(directions) >= int(max_contact_directions):
            break
    for direction_index, direction in enumerate(directions):
        for distance in translation_distances_A:
            distance = abs(float(distance))
            variants.append(LocalVariant(
                variant_id=f"clash_tr_d{direction_index}_{distance:g}",
                move_type="clash_translation",
                coordinates=coordinates + distance * direction.reshape(1, 3),
                magnitude=distance,
                axis_label=f"clash_escape_{direction_index}",
            ))

    torque_axis = _unit(field.torque)
    if torque_axis is not None:
        centre = coordinates.mean(axis=0)
        for angle in rotation_degrees:
            angle = abs(float(angle))
            # The positive torque direction is preferred, but its opposite is
            # retained because multiple contacts can make the local field noisy.
            for signed in (angle, -angle):
                variants.append(LocalVariant(
                    variant_id=f"clash_rot_{signed:+g}",
                    move_type="clash_rotation",
                    coordinates=rotate_about_point(
                        coordinates, torque_axis, signed, centre
                    ),
                    magnitude=signed,
                    axis_label="clash_torque",
                ))

    rotors = rotatable_bonds(molecule)
    scored_rotors = []
    for fixed, moving, mask in rotors:
        relevance = float(np.linalg.norm(field.atom_forces[mask], axis=1).sum())
        scored_rotors.append((relevance, fixed, moving, mask))
    scored_rotors.sort(key=lambda item: (-item[0], item[1], item[2]))
    for rank, (_, fixed, moving, mask) in enumerate(
        scored_rotors[: int(max_torsion_bonds)]
    ):
        for angle in torsion_degrees:
            angle = abs(float(angle))
            for signed in (-angle, angle):
                variants.append(LocalVariant(
                    variant_id=f"clash_tor_r{rank}_{fixed}-{moving}_{signed:+g}",
                    move_type="clash_torsion",
                    coordinates=rotate_torsion_component(
                        coordinates, fixed, moving, mask, signed
                    ),
                    magnitude=signed,
                    axis_label=f"clash_relevant_bond_{rank}",
                    bond=(int(fixed), int(moving)),
                ))
    return variants


def diverse_beam_indices(
    candidates,
    *,
    beam_size: int,
    minimum_rmsd_A: float = 0.04,
) -> list[int]:
    """Choose good candidates while preventing near-duplicate beam collapse."""

    if beam_size < 1:
        raise ValueError("beam_size must be positive")
    order = sorted(range(len(candidates)), key=lambda index: candidates[index].selection_key())
    selected: list[int] = []
    for index in order:
        if all(
            direct_rmsd(candidates[index].coordinates, candidates[other].coordinates)
            >= float(minimum_rmsd_A)
            for other in selected
        ):
            selected.append(index)
            if len(selected) >= min(beam_size, len(order)):
                return selected
    for index in order:
        if index not in selected:
            selected.append(index)
            if len(selected) >= min(beam_size, len(order)):
                break
    return selected
