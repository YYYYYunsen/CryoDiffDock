"""Restrained, reference-free ligand internal-geometry refinement.

The geometry checks reproduce the distance-geometry part of PoseBusters 0.6.5
``dock.yml`` without importing PoseBusters.  Candidate coordinates are produced
with position-restrained MMFF94s (UFF fallback) and rigidly aligned back onto
the input pose so the optimizer changes internal geometry rather than docking
placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InternalGeometryMetrics:
    internal_clash_count: int
    shortest_noncovalent_relative_distance: float
    bond_outlier_count: int
    angle_outlier_count: int
    internal_steric_clash_pass: bool
    bond_lengths_pass: bool
    bond_angles_pass: bool


@dataclass(frozen=True)
class RestrainedGeometryCandidate:
    candidate_id: str
    molecule: object
    force_field: str
    minimize_status: int
    energy: float
    displacement_A: float
    maximum_atom_displacement_A: float
    geometry: InternalGeometryMetrics


def _angle_end_pairs(molecule) -> set[tuple[int, int]]:
    neighbours = {
        atom.GetIdx(): [item.GetIdx() for item in atom.GetNeighbors()]
        for atom in molecule.GetAtoms()
    }
    pairs: set[tuple[int, int]] = set()
    for adjacent in neighbours.values():
        for first_index, first in enumerate(adjacent):
            for second in adjacent[first_index + 1:]:
                pairs.add(tuple(sorted((int(first), int(second)))))
    return pairs


def _symmetrized_terminal_bond_bounds(molecule, bounds) -> dict[tuple[int, int], tuple[float, float]]:
    """Match PoseBusters' conjugated terminal O/N bond-bound symmetrization."""

    from rdkit import Chem

    query = Chem.MolFromSmarts(
        "[O,N;D1;$([O,N;D1]-[*]=[O,N;D1]),$([O,N;D1]=[*]-[O,N;D1])]~[*]"
    )
    matches = [tuple(sorted(match)) for match in molecule.GetSubstructMatches(query)]
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for pair in matches:
        elements = tuple(sorted(
            molecule.GetAtomWithIdx(index).GetSymbol() for index in pair
        ))
        grouped.setdefault(elements, []).append(pair)
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for pairs in grouped.values():
        lower = min(float(bounds[second, first]) for first, second in pairs)
        upper = max(float(bounds[first, second]) for first, second in pairs)
        result.update({pair: (lower, upper) for pair in pairs})
    return result


def posebusters_internal_geometry(
    molecule,
    *,
    threshold_bad_bond_length: float = 0.25,
    threshold_bad_angle: float = 0.25,
    threshold_clash: float = 0.30,
) -> InternalGeometryMetrics:
    """Evaluate the PoseBusters 0.6.5 distance-geometry binary checks."""

    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

    mol = Chem.Mol(molecule)
    Chem.SanitizeMol(mol)
    if mol.GetNumConformers() == 0:
        raise ValueError("Molecule does not have a conformer")
    try:
        bounds = GetMoleculeBoundsMatrix(
            mol,
            set15bounds=True,
            scaleVDW=True,
            doTriangleSmoothing=True,
            useMacrocycle14config=False,
        )
    except (TypeError, RuntimeError):
        # RDKit 2023.03 (used by the main DiffDock environment) predates the
        # Python keyword while PoseBusters' newer environment exposes it.  The
        # older implementation has the equivalent non-macrocycle default.
        bounds = GetMoleculeBoundsMatrix(
            mol,
            set15bounds=True,
            scaleVDW=True,
            doTriangleSmoothing=True,
        )
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    bonds = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
    }
    angle_pairs = _angle_end_pairs(mol)
    symmetrized_bounds = _symmetrized_terminal_bond_bounds(mol, bounds)

    bond_outliers = 0
    angle_outliers = 0
    internal_clashes = 0
    shortest_relative = float("inf")
    for first in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(first).GetAtomicNum() == 1:
            continue
        for second in range(first + 1, mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(second).GetAtomicNum() == 1:
                continue
            pair = (first, second)
            lower, upper = symmetrized_bounds.get(
                pair,
                (float(bounds[second, first]), float(bounds[first, second])),
            )
            observed = float(distances[first, second])
            if pair in bonds:
                if observed < lower:
                    error = (lower - observed) / lower
                elif observed > upper:
                    error = (observed - upper) / upper
                else:
                    error = 0.0
                bond_outliers += int(error > float(threshold_bad_bond_length))
            elif pair in angle_pairs:
                if observed < lower:
                    error = (lower - observed) / lower
                elif observed > upper:
                    error = (observed - upper) / upper
                else:
                    error = 0.0
                angle_outliers += int(error > float(threshold_bad_angle))
            else:
                relative = observed / lower
                shortest_relative = min(shortest_relative, relative)
                internal_clashes += int(relative < 1.0 - float(threshold_clash))

    return InternalGeometryMetrics(
        internal_clash_count=int(internal_clashes),
        shortest_noncovalent_relative_distance=float(shortest_relative),
        bond_outlier_count=int(bond_outliers),
        angle_outlier_count=int(angle_outliers),
        internal_steric_clash_pass=internal_clashes == 0,
        bond_lengths_pass=bond_outliers == 0,
        bond_angles_pass=angle_outliers == 0,
    )


def _align_to_reference(
    coordinates: np.ndarray,
    reference: np.ndarray,
    alignment_indices: Sequence[int],
) -> np.ndarray:
    """Remove net rigid drift using an atom-mapped Kabsch alignment."""

    moving = np.asarray(coordinates, dtype=np.float64)
    fixed = np.asarray(reference, dtype=np.float64)
    indices = np.asarray(alignment_indices, dtype=int)
    moving_centre = moving[indices].mean(axis=0)
    fixed_centre = fixed[indices].mean(axis=0)
    x = moving[indices] - moving_centre
    y = fixed[indices] - fixed_centre
    left, _, right_t = np.linalg.svd(x.T @ y)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return (moving - moving_centre) @ rotation + fixed_centre


def _force_field(prepared, method: str):
    from rdkit.Chem import AllChem

    if method == "MMFF94s":
        if not AllChem.MMFFHasAllMoleculeParams(prepared):
            return None
        properties = AllChem.MMFFGetMoleculeProperties(
            prepared, mmffVariant="MMFF94s"
        )
        return AllChem.MMFFGetMoleculeForceField(prepared, properties, confId=0)
    if method == "UFF":
        if not AllChem.UFFHasAllMoleculeParams(prepared):
            return None
        return AllChem.UFFGetMoleculeForceField(prepared, confId=0)
    raise ValueError(f"Unsupported force field: {method}")


def restrained_internal_geometry_candidates(
    molecule,
    *,
    maximum_atom_displacements_A: Sequence[float],
    force_constants: Sequence[float],
    maximum_iterations: int = 200,
    force_fields: Sequence[str] = ("MMFF94s", "UFF"),
) -> list[RestrainedGeometryCandidate]:
    """Generate baseline and restrained local-minimization candidates."""

    from rdkit import Chem

    root = Chem.Mol(molecule)
    Chem.SanitizeMol(root)
    if root.GetNumConformers() == 0:
        raise ValueError("Molecule does not have a conformer")
    root_xyz = np.asarray(root.GetConformer().GetPositions(), dtype=np.float64)
    original_atoms_n = root.GetNumAtoms()
    heavy_indices = [
        atom.GetIdx() for atom in root.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    baseline_geometry = posebusters_internal_geometry(root)
    candidates = [RestrainedGeometryCandidate(
        candidate_id="baseline",
        molecule=Chem.Mol(root),
        force_field="none",
        minimize_status=0,
        energy=float("nan"),
        displacement_A=0.0,
        maximum_atom_displacement_A=0.0,
        geometry=baseline_geometry,
    )]
    if baseline_geometry.internal_steric_clash_pass:
        return candidates
    seen: set[bytes] = {np.rint(root_xyz / 1e-5).astype(np.int64).tobytes()}

    for method in force_fields:
        method_available = False
        for maximum_displacement in maximum_atom_displacements_A:
            for force_constant in force_constants:
                prepared = Chem.AddHs(Chem.Mol(root), addCoords=True)
                field = _force_field(prepared, str(method))
                if field is None:
                    continue
                method_available = True
                add_constraint = (
                    field.MMFFAddPositionConstraint
                    if method == "MMFF94s"
                    else field.UFFAddPositionConstraint
                )
                for index in heavy_indices:
                    add_constraint(
                        int(index), float(maximum_displacement), float(force_constant)
                    )
                field.Initialize()
                status = int(field.Minimize(maxIts=int(maximum_iterations)))
                energy = float(field.CalcEnergy())
                minimized = np.asarray(
                    prepared.GetConformer().GetPositions(), dtype=np.float64
                )[:original_atoms_n]
                aligned = _align_to_reference(minimized, root_xyz, heavy_indices)
                fingerprint = np.rint(aligned / 1e-5).astype(np.int64).tobytes()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                output = Chem.Mol(root)
                conformer = output.GetConformer()
                for index, position in enumerate(aligned):
                    conformer.SetAtomPosition(index, tuple(float(x) for x in position))
                delta = aligned[np.asarray(heavy_indices)] - root_xyz[np.asarray(heavy_indices)]
                atom_displacement = np.linalg.norm(delta, axis=1)
                candidates.append(RestrainedGeometryCandidate(
                    candidate_id=(
                        f"{method.lower()}_d{maximum_displacement:g}_"
                        f"k{force_constant:g}"
                    ),
                    molecule=output,
                    force_field=str(method),
                    minimize_status=status,
                    energy=energy,
                    displacement_A=float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
                    maximum_atom_displacement_A=float(atom_displacement.max()),
                    geometry=posebusters_internal_geometry(output),
                ))
        if method_available:
            break
    return candidates
