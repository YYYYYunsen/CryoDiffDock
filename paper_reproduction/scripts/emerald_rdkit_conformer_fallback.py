#!/usr/bin/env python3
"""Repair an invalid all-zero EMERALD MOL2 conformer from SMILES.

The input MOL2 remains the authority for atom order, atom types, bonds, and
partial charges.  RDKit supplies only new Cartesian coordinates.  No deposited
or native ligand coordinates are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Mol2Atom:
    line_index: int
    fields: tuple[str, ...]

    @property
    def element(self) -> str:
        value = self.fields[5].split(".", 1)[0]
        if value.upper() in {"CL", "BR"}:
            return value.title()
        return value[:1].upper() + value[1:].lower()

    @property
    def xyz(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.fields[2:5])  # type: ignore[return-value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mol2(path: Path) -> tuple[list[str], list[Mol2Atom]]:
    lines = path.read_text().splitlines()
    atoms: list[Mol2Atom] = []
    in_atoms = False
    for line_index, line in enumerate(lines):
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip().upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not line.strip():
            continue
        fields = tuple(line.split())
        if len(fields) < 9:
            raise ValueError(
                f"MOL2 atom line {line_index + 1} has fewer than 9 fields: {line!r}"
            )
        atoms.append(Mol2Atom(line_index=line_index, fields=fields))
    if not atoms:
        raise ValueError(f"no MOL2 atoms found in {path}")
    return lines, atoms


def _zero_coordinate_count(atoms: Sequence[Mol2Atom]) -> int:
    return sum(all(value == 0.0 for value in atom.xyz) for atom in atoms)


def _read_smiles(path: Path) -> str:
    for line in path.read_text().splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            return value.split()[0]
    raise ValueError(f"no SMILES found in {path}")


def _element_sequence_from_rdkit(molecule) -> list[str]:
    return [atom.GetSymbol() for atom in molecule.GetAtoms()]


def _prepare_rdkit_molecule(input_mol2: Path, smiles: str, expected: list[str]):
    from rdkit import Chem

    try:
        molecule = Chem.MolFromMol2File(
            str(input_mol2),
            sanitize=False,
            removeHs=False,
            cleanupSubstructures=False,
        )
    except TypeError:
        molecule = Chem.MolFromMol2File(
            str(input_mol2), sanitize=False, removeHs=False
        )

    source = "input_mol2_graph"
    if molecule is not None:
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            molecule = None

    if molecule is None or _element_sequence_from_rdkit(molecule) != expected:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("RDKit could not parse the fallback SMILES")
        molecule = Chem.AddHs(molecule)
        source = "smiles_graph"

    observed = _element_sequence_from_rdkit(molecule)
    if observed != expected:
        raise ValueError(
            "RDKit/input-MOL2 atom-order mismatch; refusing an unsafe coordinate "
            f"transfer (expected {expected}, observed {observed})"
        )

    molecule.RemoveAllConformers()
    molecule.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(molecule)
    return molecule, source


def _embed(molecule, seed: int) -> tuple[list[tuple[float, float, float]], dict]:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    ring_sizes = [len(ring) for ring in Chem.GetSymmSSSR(molecule)]
    maximum_ring_size = max(ring_sizes, default=0)
    is_macrocycle = maximum_ring_size >= 12

    conformer_id = -1
    attempt_used = None
    used_random_coordinates = False
    for attempt in range(4):
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = (int(seed) + attempt) % 2147483647
        parameters.maxIterations = 2000
        parameters.useMacrocycleTorsions = bool(is_macrocycle)
        parameters.useMacrocycle14config = bool(is_macrocycle)
        parameters.useRandomCoords = attempt == 3
        conformer_id = AllChem.EmbedMolecule(molecule, parameters)
        if conformer_id >= 0:
            attempt_used = attempt + 1
            used_random_coordinates = bool(parameters.useRandomCoords)
            break
    if conformer_id < 0:
        raise RuntimeError("RDKit conformer generation failed after four attempts")

    optimization = "none"
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule):
            AllChem.MMFFOptimizeMolecule(molecule, confId=conformer_id, maxIters=500)
            optimization = "MMFF"
        elif AllChem.UFFHasAllMoleculeParams(molecule):
            AllChem.UFFOptimizeMolecule(molecule, confId=conformer_id, maxIters=500)
            optimization = "UFF"
    except Exception as exc:
        optimization = f"failed_kept_embedded_coordinates:{type(exc).__name__}"

    conformer = molecule.GetConformer(conformer_id)
    coordinates = [
        (float(position.x), float(position.y), float(position.z))
        for position in (
            conformer.GetAtomPosition(index)
            for index in range(molecule.GetNumAtoms())
        )
    ]
    if not all(math.isfinite(value) for xyz in coordinates for value in xyz):
        raise RuntimeError("RDKit generated non-finite coordinates")
    if sum(all(value == 0.0 for value in xyz) for xyz in coordinates) > 1:
        raise RuntimeError("RDKit fallback still contains multiple all-zero atoms")

    return coordinates, {
        "rdkit_version": rdBase.rdkitVersion,
        "embedding": "ETKDGv3",
        "random_seed_base": int(seed),
        "embedding_attempt": attempt_used,
        "used_random_coordinates": used_random_coordinates,
        "maximum_ring_size": maximum_ring_size,
        "macrocycle_parameters_enabled": is_macrocycle,
        "optimization": optimization,
    }


def _format_atom_line(atom: Mol2Atom, xyz: tuple[float, float, float]) -> str:
    fields = atom.fields
    atom_id = int(fields[0])
    suffix = ""
    if len(fields) > 9:
        suffix = " " + " ".join(fields[9:])
    return (
        f"{atom_id:7d} {fields[1]:<8}"
        f"{xyz[0]:10.4f}{xyz[1]:10.4f}{xyz[2]:10.4f} "
        f"{fields[5]:<8} {fields[6]:>4} {fields[7]:<8} {fields[8]:>10}"
        f"{suffix}"
    )


def _write_coordinates(
    lines: list[str],
    atoms: Sequence[Mol2Atom],
    coordinates: Sequence[tuple[float, float, float]],
    output_path: Path,
) -> None:
    if len(atoms) != len(coordinates):
        raise ValueError(
            f"coordinate count mismatch: {len(coordinates)} coordinates for "
            f"{len(atoms)} MOL2 atoms"
        )
    output = list(lines)
    for atom, xyz in zip(atoms, coordinates):
        output[atom.line_index] = _format_atom_line(atom, xyz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mol2", type=Path, required=True)
    parser.add_argument("--smiles-file", type=Path, required=True)
    parser.add_argument("--output-mol2", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    lines, atoms = _read_mol2(args.input_mol2)
    zero_count = _zero_coordinate_count(atoms)
    if zero_count <= 1:
        raise ValueError(
            "fallback was requested for a MOL2 without multiple all-zero atoms"
        )

    smiles = _read_smiles(args.smiles_file)
    expected_elements = [atom.element for atom in atoms]
    molecule, molecule_source = _prepare_rdkit_molecule(
        args.input_mol2, smiles, expected_elements
    )
    coordinates, embedding_audit = _embed(molecule, args.seed)
    _write_coordinates(lines, atoms, coordinates, args.output_mol2)

    _, repaired_atoms = _read_mol2(args.output_mol2)
    original_noncoordinate_fields = [
        (atom.fields[0], atom.fields[1], *atom.fields[5:]) for atom in atoms
    ]
    repaired_noncoordinate_fields = [
        (atom.fields[0], atom.fields[1], *atom.fields[5:])
        for atom in repaired_atoms
    ]
    if repaired_noncoordinate_fields != original_noncoordinate_fields:
        raise RuntimeError(
            "fallback changed an atom ID, name, type, residue, charge, or "
            "other non-coordinate MOL2 field"
        )
    repaired_zero_count = _zero_coordinate_count(repaired_atoms)
    if repaired_zero_count > 1:
        raise RuntimeError("written fallback MOL2 failed zero-coordinate validation")

    audit = {
        "trigger": "multiple_all_zero_atom_coordinates",
        "input_mol2": str(args.input_mol2),
        "input_mol2_sha256": _sha256(args.input_mol2),
        "smiles_file": str(args.smiles_file),
        "smiles_file_sha256": _sha256(args.smiles_file),
        "native_coordinates_used": False,
        "molecule_source": molecule_source,
        "atom_count": len(atoms),
        "original_zero_coordinate_atom_count": zero_count,
        "repaired_zero_coordinate_atom_count": repaired_zero_count,
        "atom_order_preserved": True,
        "atom_types_preserved": True,
        "bond_table_preserved": True,
        "partial_charge_strings_preserved": True,
        "output_mol2": str(args.output_mol2),
        "output_mol2_sha256": _sha256(args.output_mol2),
        "fallback_runtime_seconds": time.perf_counter() - started,
        **embedding_audit,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
