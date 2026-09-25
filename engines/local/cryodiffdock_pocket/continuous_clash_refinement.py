"""Reference-free continuous refinement for residual protein--ligand clashes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from cryodiffdock_pocket.clash_repair import ClashField, clash_field
from cryodiffdock_pocket.discrete_search import (
    direct_rmsd,
    rotate_torsion_component,
)


@dataclass(frozen=True)
class RefinementEvaluation:
    coordinates: np.ndarray
    clash: ClashField
    displacement_A: float
    multiscale_ncc: float
    ncc_change: float


@dataclass(frozen=True)
class RefinementResult:
    coordinates: np.ndarray
    evaluation: RefinementEvaluation
    success: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    parameter_vector: np.ndarray


def apply_pose_parameters(
    seed_coordinates: np.ndarray,
    parameters: np.ndarray,
    rotors: Sequence[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    """Apply torsions, an axis-angle rotation, and translation to one seed."""

    seed = np.asarray(seed_coordinates, dtype=np.float64)
    values = np.asarray(parameters, dtype=np.float64)
    expected = 6 + len(rotors)
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} pose parameters, got {values.shape}")
    updated = seed.copy()
    for angle, (fixed, moving, mask) in zip(values[6:], rotors):
        updated = rotate_torsion_component(
            updated, int(fixed), int(moving), np.asarray(mask, dtype=bool),
            float(angle),
        )
    centre = updated.mean(axis=0)
    matrix = Rotation.from_rotvec(values[3:6]).as_matrix()
    updated = (updated - centre) @ matrix.T + centre
    updated += values[:3].reshape(1, 3)
    return updated


class CachedPoseEvaluator:
    """Cache expensive NCC evaluations shared by SLSQP callbacks."""

    def __init__(
        self,
        *,
        root_coordinates: np.ndarray,
        seed_coordinates: np.ndarray,
        rotors: Sequence[tuple[int, int, np.ndarray]],
        protein_coordinates: np.ndarray,
        protein_elements,
        ligand_elements,
        root_ncc: float,
        density_scorer: Callable[[np.ndarray], float],
        severe_overlap_A: float,
    ) -> None:
        self.root_coordinates = np.asarray(root_coordinates, dtype=np.float64)
        self.seed_coordinates = np.asarray(seed_coordinates, dtype=np.float64)
        self.rotors = list(rotors)
        self.protein_coordinates = np.asarray(protein_coordinates, dtype=np.float64)
        self.protein_elements = list(protein_elements)
        self.ligand_elements = list(ligand_elements)
        self.root_ncc = float(root_ncc)
        self.density_scorer = density_scorer
        self.severe_overlap_A = float(severe_overlap_A)
        self._cache: dict[bytes, RefinementEvaluation] = {}

    def evaluate(self, parameters: np.ndarray) -> RefinementEvaluation:
        values = np.asarray(parameters, dtype=np.float64)
        key = np.round(values, 10).tobytes()
        if key not in self._cache:
            coordinates = apply_pose_parameters(
                self.seed_coordinates, values, self.rotors
            )
            steric = clash_field(
                self.protein_coordinates,
                self.protein_elements,
                coordinates,
                self.ligand_elements,
                severe_overlap_A=self.severe_overlap_A,
            )
            ncc = float(self.density_scorer(coordinates))
            self._cache[key] = RefinementEvaluation(
                coordinates=coordinates,
                clash=steric,
                displacement_A=direct_rmsd(self.root_coordinates, coordinates),
                multiscale_ncc=ncc,
                ncc_change=ncc - self.root_ncc,
            )
        return self._cache[key]


class CachedGeometryEvaluator:
    """Cheap steric/displacement evaluator for density-free inner loops."""

    def __init__(
        self,
        *,
        root_coordinates: np.ndarray,
        seed_coordinates: np.ndarray,
        rotors: Sequence[tuple[int, int, np.ndarray]],
        protein_coordinates: np.ndarray,
        protein_elements,
        ligand_elements,
        severe_overlap_A: float,
    ) -> None:
        self.root_coordinates = np.asarray(root_coordinates, dtype=np.float64)
        self.seed_coordinates = np.asarray(seed_coordinates, dtype=np.float64)
        self.rotors = list(rotors)
        self.protein_coordinates = np.asarray(protein_coordinates, dtype=np.float64)
        self.protein_elements = list(protein_elements)
        self.ligand_elements = list(ligand_elements)
        self.severe_overlap_A = float(severe_overlap_A)
        self._cache: dict[bytes, tuple[np.ndarray, ClashField, float]] = {}

    def evaluate(self, parameters: np.ndarray) -> tuple[np.ndarray, ClashField, float]:
        values = np.asarray(parameters, dtype=np.float64)
        key = np.round(values, 10).tobytes()
        if key not in self._cache:
            coordinates = apply_pose_parameters(
                self.seed_coordinates, values, self.rotors
            )
            steric = clash_field(
                self.protein_coordinates,
                self.protein_elements,
                coordinates,
                self.ligand_elements,
                severe_overlap_A=self.severe_overlap_A,
            )
            displacement = direct_rmsd(self.root_coordinates, coordinates)
            self._cache[key] = (coordinates, steric, displacement)
        return self._cache[key]


def refine_seed_fast_slsqp(
    *,
    root_coordinates: np.ndarray,
    seed_coordinates: np.ndarray,
    rotors: Sequence[tuple[int, int, np.ndarray]],
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_elements,
    root_ncc: float,
    density_scorer: Callable[[np.ndarray], float],
    severe_overlap_A: float = 0.8,
    max_displacement_A: float = 0.75,
    max_ncc_drop: float = 0.02,
    translation_bound_A: float = 0.25,
    rotation_bound_deg: float = 7.5,
    torsion_bound_deg: float = 15.0,
    max_iterations: int = 30,
    ftol: float = 1e-7,
) -> RefinementResult:
    """Optimize sterics cheaply, then enforce density with path backtracking.

    Exact multiscale NCC is deliberately absent from the finite-difference
    inner loop.  It is evaluated only at a small fixed set of points on the
    path from the feasible seed to the optimized pose.  Final acceptance uses
    the same exact hard gates as the slower constrained implementation.
    """

    evaluator = CachedGeometryEvaluator(
        root_coordinates=root_coordinates,
        seed_coordinates=seed_coordinates,
        rotors=rotors,
        protein_coordinates=protein_coordinates,
        protein_elements=protein_elements,
        ligand_elements=ligand_elements,
        severe_overlap_A=severe_overlap_A,
    )
    count = 6 + len(rotors)
    initial = np.zeros(count, dtype=np.float64)
    rotation_bound = np.deg2rad(float(rotation_bound_deg))
    bounds = (
        [(-float(translation_bound_A), float(translation_bound_A))] * 3
        + [(-rotation_bound, rotation_bound)] * 3
        + [(-float(torsion_bound_deg), float(torsion_bound_deg))] * len(rotors)
    )

    def objective(parameters: np.ndarray) -> float:
        _, steric, displacement = evaluator.evaluate(parameters)
        torsion_regularizer = float(np.sum(np.asarray(parameters[6:]) ** 2))
        return steric.energy + 0.02 * displacement ** 2 + 1e-6 * torsion_regularizer

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[{
            "type": "ineq",
            "fun": lambda values: float(max_displacement_A)
            - evaluator.evaluate(values)[2],
        }],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )

    # A fixed schedule is deterministic and caps exact density work at nine
    # evaluations per start.  Include the original seed (scale zero), which is
    # known to be feasible for the cascade input.
    scales = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0)
    feasible: list[tuple[tuple[float, ...], RefinementEvaluation, np.ndarray]] = []
    for scale in scales:
        parameters = np.asarray(result.x, dtype=np.float64) * float(scale)
        coordinates, steric, displacement = evaluator.evaluate(parameters)
        ncc = float(density_scorer(coordinates))
        evaluation = RefinementEvaluation(
            coordinates=coordinates,
            clash=steric,
            displacement_A=displacement,
            multiscale_ncc=ncc,
            ncc_change=ncc - float(root_ncc),
        )
        if (displacement <= float(max_displacement_A) + 1e-8
                and evaluation.ncc_change >= -float(max_ncc_drop) - 1e-8):
            key = (
                float(steric.severe_count), steric.energy,
                steric.clashing_ligand_atom_fraction, displacement,
            )
            feasible.append((key, evaluation, parameters))
    if not feasible:
        raise RuntimeError("No density/displacement-feasible point, including seed")
    feasible.sort(key=lambda item: item[0])
    _, final, final_parameters = feasible[0]
    return RefinementResult(
        coordinates=final.coordinates,
        evaluation=final,
        success=True,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        parameter_vector=final_parameters,
    )


def refine_seed_slsqp(
    *,
    root_coordinates: np.ndarray,
    seed_coordinates: np.ndarray,
    rotors: Sequence[tuple[int, int, np.ndarray]],
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_elements,
    root_ncc: float,
    density_scorer: Callable[[np.ndarray], float],
    severe_overlap_A: float = 0.8,
    max_displacement_A: float = 0.75,
    max_ncc_drop: float = 0.02,
    translation_bound_A: float = 0.25,
    rotation_bound_deg: float = 7.5,
    torsion_bound_deg: float = 15.0,
    max_iterations: int = 80,
    ftol: float = 1e-7,
) -> RefinementResult:
    """Refine one feasible seed with exact displacement and NCC constraints."""

    evaluator = CachedPoseEvaluator(
        root_coordinates=root_coordinates,
        seed_coordinates=seed_coordinates,
        rotors=rotors,
        protein_coordinates=protein_coordinates,
        protein_elements=protein_elements,
        ligand_elements=ligand_elements,
        root_ncc=root_ncc,
        density_scorer=density_scorer,
        severe_overlap_A=severe_overlap_A,
    )
    count = 6 + len(rotors)
    initial = np.zeros(count, dtype=np.float64)
    rotation_bound = np.deg2rad(float(rotation_bound_deg))
    bounds = (
        [(-float(translation_bound_A), float(translation_bound_A))] * 3
        + [(-rotation_bound, rotation_bound)] * 3
        + [(-float(torsion_bound_deg), float(torsion_bound_deg))] * len(rotors)
    )

    def objective(parameters: np.ndarray) -> float:
        evaluation = evaluator.evaluate(parameters)
        torsion_regularizer = float(np.sum(np.asarray(parameters[6:]) ** 2))
        return (
            evaluation.clash.energy
            + 0.02 * evaluation.displacement_A ** 2
            + 1e-6 * torsion_regularizer
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda values: float(max_displacement_A)
            - evaluator.evaluate(values).displacement_A,
        },
        {
            "type": "ineq",
            "fun": lambda values: evaluator.evaluate(values).ncc_change
            + float(max_ncc_drop),
        },
    ]
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": int(max_iterations),
            "ftol": float(ftol),
            "disp": False,
        },
    )
    final = evaluator.evaluate(np.asarray(result.x, dtype=np.float64))
    gates_pass = bool(
        final.displacement_A <= float(max_displacement_A) + 1e-8
        and final.ncc_change >= -float(max_ncc_drop) - 1e-8
    )
    return RefinementResult(
        coordinates=final.coordinates,
        evaluation=final,
        success=gates_pass,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(getattr(result, "nit", 0)),
        function_evaluations=int(getattr(result, "nfev", 0)),
        parameter_vector=np.asarray(result.x, dtype=np.float64),
    )
