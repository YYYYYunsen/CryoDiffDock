"""Reference-free rigid micro-refinement primitives for CAAHS-C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from cryodiffdock_pocket.clash_repair import clash_field
from cryodiffdock_pocket.discrete_search import direct_rmsd
from cryodiffdock_pocket.physics import rdkit_vdw_radii, relative_vdw_clashes


@dataclass(frozen=True)
class CaahsCCandidate:
    candidate_id: str
    coordinates: np.ndarray
    relative_clash_count: int
    minimum_relative_distance: float
    target_deficit_energy: float
    severe_clash_count: int
    displacement_A: float
    multiscale_ncc: float
    ncc_change: float
    optimizer_success: bool


def apply_rigid_parameters(coordinates: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("CAAHS-C rigid parameter vector must contain six values")
    centre = xyz.mean(axis=0)
    matrix = Rotation.from_rotvec(values[3:6]).as_matrix()
    return (xyz - centre) @ matrix.T + centre + values[:3]


def relative_deficit_energy(
    protein_coordinates: np.ndarray,
    protein_elements: Sequence[str],
    ligand_coordinates: np.ndarray,
    ligand_elements: Sequence[str],
    *,
    target_relative_distance: float,
) -> float:
    protein = np.asarray(protein_coordinates, dtype=np.float64)
    ligand = np.asarray(ligand_coordinates, dtype=np.float64)
    distances = np.linalg.norm(protein[:, None, :] - ligand[None, :, :], axis=-1)
    radius_sum = (
        rdkit_vdw_radii(protein_elements)[:, None]
        + rdkit_vdw_radii(ligand_elements)[None, :]
    )
    deficit = np.maximum(
        0.0, float(target_relative_distance) * radius_sum - distances
    )
    return float(np.sum(deficit * deficit))


def escape_directions(
    protein_coordinates: np.ndarray,
    protein_elements: Sequence[str],
    ligand_coordinates: np.ndarray,
    ligand_elements: Sequence[str],
    *,
    target_relative_distance: float,
    maximum_directions: int = 4,
) -> list[np.ndarray]:
    protein = np.asarray(protein_coordinates, dtype=np.float64)
    ligand = np.asarray(ligand_coordinates, dtype=np.float64)
    delta = ligand[None, :, :] - protein[:, None, :]
    distances = np.linalg.norm(delta, axis=-1)
    safe = np.maximum(distances, 1e-8)
    radius_sum = (
        rdkit_vdw_radii(protein_elements)[:, None]
        + rdkit_vdw_radii(ligand_elements)[None, :]
    )
    deficit = np.maximum(0.0, float(target_relative_distance) * radius_sum - distances)
    atom_force = np.sum(2.0 * deficit[:, :, None] * delta / safe[:, :, None], axis=0)
    proposals = [atom_force.sum(axis=0)]
    proposals.extend(atom_force[index] for index in np.argsort(-np.linalg.norm(atom_force, axis=1)))
    directions: list[np.ndarray] = []
    for vector in proposals:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-10:
            continue
        direction = np.asarray(vector, dtype=np.float64) / norm
        if all(float(np.dot(direction, old)) < 0.97 for old in directions):
            directions.append(direction)
        if len(directions) >= int(maximum_directions):
            break
    return directions


def refine_rigid_pose(
    *,
    ligand_coordinates: np.ndarray,
    ligand_elements: Sequence[str],
    protein_coordinates: np.ndarray,
    protein_elements: Sequence[str],
    density_scorer: Callable[[np.ndarray], float],
    maximum_displacement_A: float,
    maximum_ncc_drop: float = 0.02,
    target_relative_distance: float = 0.77,
    optimization_relative_distance: float | None = None,
    acceptance_relative_distance: float = 0.75,
    severe_overlap_A: float = 0.8,
    rotation_bound_degrees: float = 5.0,
    maximum_iterations: int = 30,
    number_of_starts: int = 4,
) -> list[CaahsCCandidate]:
    """Return baseline and deterministic SLSQP/path-backtracking candidates."""

    root = np.asarray(ligand_coordinates, dtype=np.float64)
    root_ncc = float(density_scorer(root))
    optimization_target = (
        float(target_relative_distance) + 0.01
        if optimization_relative_distance is None
        else float(optimization_relative_distance)
    )
    if optimization_target < float(target_relative_distance):
        raise ValueError("optimization target cannot be below the selection target")
    baseline_relative_count, baseline_minimum_relative, _ = relative_vdw_clashes(
        protein_coordinates,
        protein_elements,
        root,
        ligand_elements,
        minimum_relative_distance=acceptance_relative_distance,
    )
    baseline_steric = clash_field(
        protein_coordinates,
        protein_elements,
        root,
        ligand_elements,
        severe_overlap_A=severe_overlap_A,
    )
    if baseline_relative_count == 0 and baseline_steric.severe_count == 0:
        return [CaahsCCandidate(
            candidate_id="baseline",
            coordinates=root.copy(),
            relative_clash_count=0,
            minimum_relative_distance=baseline_minimum_relative,
            target_deficit_energy=relative_deficit_energy(
                protein_coordinates,
                protein_elements,
                root,
                ligand_elements,
                target_relative_distance=target_relative_distance,
            ),
            severe_clash_count=0,
            displacement_A=0.0,
            multiscale_ncc=root_ncc,
            ncc_change=0.0,
            optimizer_success=True,
        )]
    rotation_bound = np.deg2rad(float(rotation_bound_degrees))
    bounds = (
        [(-float(maximum_displacement_A), float(maximum_displacement_A))] * 3
        + [(-rotation_bound, rotation_bound)] * 3
    )

    starts = [np.zeros(6, dtype=np.float64)]
    for direction in escape_directions(
        protein_coordinates, protein_elements, root, ligand_elements,
        target_relative_distance=optimization_target,
        maximum_directions=max(0, int(number_of_starts) - 1),
    ):
        values = np.zeros(6, dtype=np.float64)
        values[:3] = min(0.10, maximum_displacement_A / 2.0) * direction
        starts.append(values)
    starts = starts[: max(1, int(number_of_starts))]

    raw: list[tuple[str, np.ndarray, bool]] = [("baseline", root, True)]
    for start_index, initial in enumerate(starts):
        def objective(parameters: np.ndarray) -> float:
            xyz = apply_rigid_parameters(root, parameters)
            displacement = direct_rmsd(root, xyz)
            return (
                relative_deficit_energy(
                    protein_coordinates, protein_elements, xyz, ligand_elements,
                    target_relative_distance=optimization_target,
                )
                + 0.02 * displacement * displacement
                + 1e-4 * float(np.sum(parameters[3:6] ** 2))
            )

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=[{
                "type": "ineq",
                "fun": lambda values: float(maximum_displacement_A)
                - direct_rmsd(root, apply_rigid_parameters(root, values)),
            }],
            options={"maxiter": int(maximum_iterations), "ftol": 1e-8, "disp": False},
        )
        for scale in (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125):
            xyz = apply_rigid_parameters(root, np.asarray(result.x) * scale)
            raw.append((f"start{start_index}_scale{scale:g}", xyz, bool(result.success)))

    candidates: list[CaahsCCandidate] = []
    seen: set[bytes] = set()
    for candidate_id, xyz, optimizer_success in raw:
        fingerprint = np.rint(np.asarray(xyz) / 1e-4).astype(np.int64).tobytes()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        displacement = direct_rmsd(root, xyz)
        ncc = float(density_scorer(xyz))
        relative_count, minimum_relative, _ = relative_vdw_clashes(
            protein_coordinates, protein_elements, xyz, ligand_elements,
            minimum_relative_distance=acceptance_relative_distance,
        )
        steric = clash_field(
            protein_coordinates, protein_elements, xyz, ligand_elements,
            severe_overlap_A=severe_overlap_A,
        )
        candidates.append(CaahsCCandidate(
            candidate_id=candidate_id,
            coordinates=np.asarray(xyz, dtype=np.float64),
            relative_clash_count=relative_count,
            minimum_relative_distance=minimum_relative,
            target_deficit_energy=relative_deficit_energy(
                protein_coordinates, protein_elements, xyz, ligand_elements,
                target_relative_distance=target_relative_distance,
            ),
            severe_clash_count=steric.severe_count,
            displacement_A=displacement,
            multiscale_ncc=ncc,
            ncc_change=ncc - root_ncc,
            optimizer_success=optimizer_success,
        ))
    candidates.sort(key=lambda item: (
        item.relative_clash_count,
        item.severe_clash_count,
        item.target_deficit_energy,
        item.displacement_A,
        -item.multiscale_ncc,
        item.candidate_id,
    ))
    return candidates


def candidate_passes(
    candidate: CaahsCCandidate,
    *,
    maximum_displacement_A: float,
    maximum_ncc_drop: float,
) -> bool:
    return bool(
        candidate.relative_clash_count == 0
        and candidate.severe_clash_count == 0
        and candidate.displacement_A <= float(maximum_displacement_A) + 1e-8
        and candidate.ncc_change >= -float(maximum_ncc_drop) - 1e-8
    )
