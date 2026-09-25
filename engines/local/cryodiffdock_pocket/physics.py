"""Deployment-safe physical diagnostics for protein--ligand candidate poses."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import PDBParser


# Match the radii used by datasets/steric_clash.py without importing RDKit at
# module import time.  This keeps geometry-only tests usable in CPU-only envs.
VDW_RADII = {
    "H": 1.10, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Na": 2.27, "Mg": 1.73, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "K": 2.75, "Ca": 2.31, "Ni": 1.63, "Cu": 1.40,
    "Zn": 1.39, "Se": 1.90, "Br": 1.83, "Pd": 1.63, "Ag": 1.72,
    "Cd": 1.58, "I": 1.98, "Pt": 1.75, "Au": 1.66, "Hg": 1.55,
    "U": 1.86,
}

# Values returned by RDKit's PeriodicTable.GetRvdw for the elements expected
# in drug-like ligands and standard protein ATOM records.  PoseBusters uses
# this convention for its relative minimum-distance check.  This deliberately
# differs from the historical project table above (for example O: 1.55 versus
# 1.52 and F: 1.50 versus 1.47).
RDKIT_VDW_RADII = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.60, "O": 1.55,
    "F": 1.50, "Si": 2.10, "P": 1.95, "S": 1.80, "Cl": 1.80,
    "Br": 1.90, "I": 2.10,
}


def _normalise_element(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "default"
    return text[0].upper() + text[1:].lower()


def atom_radii(elements) -> np.ndarray:
    return np.asarray(
        [VDW_RADII.get(_normalise_element(element), 2.0) for element in elements],
        dtype=np.float64,
    )


def rdkit_vdw_radii(elements) -> np.ndarray:
    """Return the RDKit periodic-table vdW radii used by PoseBusters.

    Keep this separate from :func:`atom_radii`: the latter is the historical
    project clash definition and must remain frozen for published endpoints.
    """

    values = []
    table = None
    for element in elements:
        symbol = _normalise_element(element)
        if symbol in RDKIT_VDW_RADII:
            values.append(RDKIT_VDW_RADII[symbol])
            continue
        if table is None:
            try:
                from rdkit import Chem
            except ImportError as error:
                raise ValueError(
                    f"No frozen PoseBusters/RDKit vdW radius for {symbol!r} "
                    "and RDKit is unavailable"
                ) from error
            table = Chem.GetPeriodicTable()
        atomic_number = int(table.GetAtomicNumber(symbol))
        if atomic_number <= 0:
            raise ValueError(f"Unknown element for RDKit vdW radius: {element!r}")
        values.append(float(table.GetRvdw(atomic_number)))
    return np.asarray(values, dtype=np.float64)


def relative_vdw_clashes(
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_coordinates: np.ndarray,
    ligand_elements,
    *,
    minimum_relative_distance: float = 0.75,
) -> tuple[int, float, np.ndarray]:
    """Return element-pair-scaled protein--ligand contact violations.

    Relative distance is ``d_ij / (r_i + r_j)``.  Values below the requested
    lower bound are violations.  The default 0.75 is the frozen CAAHS-C gate.
    """

    protein_coordinates = np.asarray(protein_coordinates, dtype=np.float64)
    ligand_coordinates = np.asarray(ligand_coordinates, dtype=np.float64)
    distance = np.linalg.norm(
        protein_coordinates[:, None, :] - ligand_coordinates[None, :, :], axis=-1
    )
    radius_sum = (
        rdkit_vdw_radii(protein_elements)[:, None]
        + rdkit_vdw_radii(ligand_elements)[None, :]
    )
    relative = np.divide(
        distance,
        radius_sum,
        out=np.full_like(distance, np.inf),
        where=radius_sum > 0,
    )
    violations = relative < float(minimum_relative_distance)
    minimum = float(relative.min()) if relative.size else float("inf")
    return int(violations.sum()), minimum, violations


def read_protein_heavy_atoms(path: str | Path):
    """Read polymer protein heavy atoms, excluding HETATM and hydrogens."""
    model = next(PDBParser(QUIET=True).get_structure("protein", str(path)).get_models())
    coordinates = []
    elements = []
    for chain in model:
        for residue in chain:
            if str(residue.id[0]).strip():
                continue
            for atom in residue:
                element = _normalise_element(atom.element or atom.name[:1])
                if element in {"H", "D"}:
                    continue
                coordinates.append(np.asarray(atom.coord, dtype=np.float64))
                elements.append(element)
    if not coordinates:
        raise ValueError(f"No polymer protein heavy atoms in {path}")
    return np.asarray(coordinates, dtype=np.float64), elements


def read_ligand_pose(path: str | Path):
    """Return an unsanitized RDKit molecule plus heavy-atom geometry."""
    from rdkit import Chem

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand pose: {path}")
    all_coordinates = np.asarray(
        molecule.GetConformer().GetPositions(), dtype=np.float64
    )
    keep = [atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()]
    coordinates = all_coordinates[np.asarray(keep, dtype=bool)]
    elements = [
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    if len(coordinates) == 0:
        raise ValueError(f"No ligand heavy atoms in {path}")
    return molecule, coordinates, elements


def clash_metrics(
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_coordinates: np.ndarray,
    ligand_elements,
    *,
    severe_overlap_A: float = 0.8,
) -> dict[str, object]:
    """Compute van-der-Waals overlap diagnostics.

    ``severe_overlap_A=0.8`` reproduces the historical DiffDock-Pocket clash
    definition: distance < r1 + r2 - 2*0.4 A.
    """
    protein_coordinates = np.asarray(protein_coordinates, dtype=np.float64)
    ligand_coordinates = np.asarray(ligand_coordinates, dtype=np.float64)
    if protein_coordinates.ndim != 2 or protein_coordinates.shape[1] != 3:
        raise ValueError("protein_coordinates must have shape [N,3]")
    if ligand_coordinates.ndim != 2 or ligand_coordinates.shape[1] != 3:
        raise ValueError("ligand_coordinates must have shape [M,3]")
    delta = protein_coordinates[:, None, :] - ligand_coordinates[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    radius_sum = (
        atom_radii(protein_elements)[:, None]
        + atom_radii(ligand_elements)[None, :]
    )
    overlap = radius_sum - distances
    severe = overlap > float(severe_overlap_A)
    clashing_ligand_atoms = severe.any(axis=0)
    return {
        "severe_overlap_cutoff_A": float(severe_overlap_A),
        "severe_clash_count": int(severe.sum()),
        "num_clashing_ligand_atoms": int(clashing_ligand_atoms.sum()),
        "clashing_ligand_atom_fraction": float(clashing_ligand_atoms.mean()),
        "max_vdw_overlap_A": float(overlap.max()),
        "min_protein_ligand_distance_A": float(distances.min()),
    }


def ligand_strain_metrics(
    molecule,
    *,
    variant: str = "MMFF94s",
    max_iterations: int = 500,
) -> dict[str, object]:
    """Estimate bound-conformer strain as single-point minus relaxed energy."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    result = {
        "strain_available": False,
        "strain_method": variant,
        "pose_energy_kcal_mol": None,
        "relaxed_energy_kcal_mol": None,
        "strain_energy_kcal_mol": None,
        "strain_minimize_status": None,
        "strain_error": "",
    }
    try:
        prepared = Chem.Mol(molecule)
        Chem.SanitizeMol(prepared)
        prepared = Chem.AddHs(prepared, addCoords=True)
        if not AllChem.MMFFHasAllMoleculeParams(prepared):
            result["strain_error"] = "MMFF parameters unavailable"
            return result
        properties = AllChem.MMFFGetMoleculeProperties(
            prepared, mmffVariant=variant
        )
        force_field = AllChem.MMFFGetMoleculeForceField(
            prepared, properties, confId=0
        )
        pose_energy = float(force_field.CalcEnergy())

        relaxed = copy.deepcopy(prepared)
        relaxed_properties = AllChem.MMFFGetMoleculeProperties(
            relaxed, mmffVariant=variant
        )
        relaxed_force_field = AllChem.MMFFGetMoleculeForceField(
            relaxed, relaxed_properties, confId=0
        )
        status = int(relaxed_force_field.Minimize(maxIts=int(max_iterations)))
        relaxed_energy = float(relaxed_force_field.CalcEnergy())
        result.update({
            "strain_available": True,
            "pose_energy_kcal_mol": pose_energy,
            "relaxed_energy_kcal_mol": relaxed_energy,
            "strain_energy_kcal_mol": max(0.0, pose_energy - relaxed_energy),
            "strain_minimize_status": status,
        })
    except Exception as exc:
        result["strain_error"] = f"{type(exc).__name__}: {exc}"
    return result


def score_ligand_pose_physics(
    protein_coordinates: np.ndarray,
    protein_elements,
    ligand_path: str | Path,
    *,
    severe_overlap_A: float = 0.8,
    calculate_strain: bool = True,
    strain_variant: str = "MMFF94s",
    strain_max_iterations: int = 500,
) -> dict[str, object]:
    molecule, ligand_coordinates, ligand_elements = read_ligand_pose(ligand_path)
    result = clash_metrics(
        protein_coordinates,
        protein_elements,
        ligand_coordinates,
        ligand_elements,
        severe_overlap_A=severe_overlap_A,
    )
    result["num_ligand_heavy_atoms"] = int(len(ligand_coordinates))
    if calculate_strain:
        result.update(ligand_strain_metrics(
            molecule,
            variant=strain_variant,
            max_iterations=strain_max_iterations,
        ))
    else:
        result.update({
            "strain_available": False,
            "strain_method": strain_variant,
            "pose_energy_kcal_mol": None,
            "relaxed_energy_kcal_mol": None,
            "strain_energy_kcal_mol": None,
            "strain_minimize_status": None,
            "strain_error": "disabled",
        })
    return result


def paired_protein_for_pose(ligand_path: str | Path) -> Optional[Path]:
    ligand_path = Path(ligand_path)
    candidate = ligand_path.with_name(f"{ligand_path.stem}_protein.pdb")
    return candidate if candidate.is_file() else None
