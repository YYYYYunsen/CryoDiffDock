#!/usr/bin/env python3
"""Prepare a non-destructive retry table for 12 recoverable ChemEM100 cases.

The running 9D6C case is intentionally excluded.  Four cases only need their
Stage-1 output root created, six receptor failures receive a hydrogen-free copy
of the already PDBFixer/Amber-validated receptor, and two ligand failures get
isomeric SMILES reassigned from their deposited 3D SDF coordinates and checked
with OpenFF.  Original configs are backed up before targeted key replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


PREPROCESS_CASES = ("7YBU", "9KUF", "9BC6", "9KDH")
RECEPTOR_CASES = ("9EO4", "9MNM", "9KDM", "8Y0H", "6O9G", "9MB9")
STEREO_CASES = ("9CF5", "7LO6")

# Interleave failure types so four 3-case GPU jobs have balanced workloads.
RETRY_ORDER = (
    "7YBU", "9EO4", "9CF5",
    "9KUF", "9MNM", "7LO6",
    "9BC6", "9KDM", "8Y0H",
    "9KDH", "6O9G", "9MB9",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def backup_once(path: Path, backup_root: Path) -> Path:
    target = backup_root / path.name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return target


def replace_config_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    matches = [index for index, line in enumerate(lines) if pattern.match(line)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}= line in {path}, found {len(matches)}")
    lines[matches[0]] = f"{key} = {value}"
    path.write_text("\n".join(lines) + "\n")


def pdb_atom_is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    atom_name = line[12:16].strip().upper() if len(line) >= 16 else ""
    normalized_name = atom_name.lstrip("0123456789")
    return element == "H" or (not element and normalized_name.startswith("H"))


def write_hydrogen_free_pdb(source: Path, output: Path) -> dict[str, int]:
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept_atoms = 0
    removed_hydrogens = 0
    lines = []
    for raw in source.read_text(errors="replace").splitlines():
        record = raw[:6].strip().upper()
        if record in {"ATOM", "HETATM"}:
            if pdb_atom_is_hydrogen(raw):
                removed_hydrogens += 1
                continue
            kept_atoms += 1
            lines.append(raw)
        elif record == "TER":
            lines.append("TER")
    while lines and lines[-1] == "TER":
        lines.pop()
    lines.extend(["TER", "END"])
    if kept_atoms == 0:
        raise ValueError(f"No heavy atoms retained from {source}")
    if removed_hydrogens == 0:
        raise ValueError(f"No hydrogen atoms found in receptor {source}")
    output.write_text("\n".join(lines) + "\n")
    return {
        "kept_receptor_heavy_atoms": kept_atoms,
        "removed_receptor_hydrogens": removed_hydrogens,
    }


def stereo_from_3d_sdf(path: Path) -> tuple[str, dict[str, object]]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("Run this script in the ChemEM environment with RDKit") from exc

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if not molecules:
        raise ValueError(f"No readable ligand in {path}")
    molecule = molecules[0]
    if molecule.GetNumConformers() == 0 or not molecule.GetConformer().Is3D():
        raise ValueError(f"Ligand lacks a 3D conformer: {path}")

    before = [
        str(info)
        for info in Chem.FindPotentialStereo(molecule)
        if "Unspecified" in str(info.specified)
    ]
    Chem.AssignStereochemistryFrom3D(
        molecule, confId=0, replaceExistingTags=True
    )
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    after = [
        str(info)
        for info in Chem.FindPotentialStereo(molecule)
        if "Unspecified" in str(info.specified)
    ]
    if after:
        raise ValueError(
            f"3D assignment left {len(after)} undefined stereo elements: {after[:5]}"
        )
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    if not smiles:
        raise ValueError(f"Empty isomeric SMILES from {path}")

    try:
        try:
            from openff.toolkit import Molecule
        except ImportError:
            from openff.toolkit.topology import Molecule
        Molecule.from_smiles(smiles, allow_undefined_stereo=False)
    except Exception as exc:
        raise ValueError(f"OpenFF rejected reassigned SMILES: {exc}") from exc

    return smiles, {
        "ligand_heavy_atoms": molecule.GetNumHeavyAtoms(),
        "undefined_stereo_before": len(before),
        "undefined_stereo_after": len(after),
        "isomeric_smiles": smiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cases_path = args.cases.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    backup_root = output / "original_config_backups"

    rows = read_rows(cases_path)
    indexed = {row["complex_name"].strip().upper(): dict(row) for row in rows}
    missing = sorted(set(RETRY_ORDER) - set(indexed))
    if missing:
        raise ValueError(f"Retry cases absent from work table: {missing}")
    if "9D6C" in RETRY_ORDER:
        raise RuntimeError("Running case 9D6C must not enter the retry set")

    repaired_rows = []
    audit_rows = []
    for retry_index, complex_name in enumerate(RETRY_ORDER):
        row = indexed[complex_name]
        config = Path(row["stage1_config"])
        stage1_output = Path(row["stage1_output"])
        stage1_output.mkdir(parents=True, exist_ok=True)
        backup = backup_once(config, backup_root)
        action = "create_stage1_output_root"
        details: dict[str, object] = {}

        if complex_name in RECEPTOR_CASES:
            source = Path(row["protein_fixed"])
            no_h = source.with_name(source.stem + "_noH.pdb")
            details = write_hydrogen_free_pdb(source, no_h)
            replace_config_value(config, "protein", str(no_h))
            row["protein_fixed"] = str(no_h)
            action = "hydrogen_free_receptor"
        elif complex_name in STEREO_CASES:
            ligand_source = Path(row["ligand_source"])
            smiles, details = stereo_from_3d_sdf(ligand_source)
            replace_config_value(config, "ligand", smiles)
            row["ligand_smiles"] = smiles
            action = "stereo_from_3d_sdf"
        elif complex_name not in PREPROCESS_CASES:
            raise RuntimeError(f"Unclassified retry case: {complex_name}")

        row["original_work_index"] = row["work_index"]
        row["work_index"] = str(retry_index)
        row["repair_action"] = action
        repaired_rows.append(row)
        audit_rows.append({
            "retry_index": retry_index,
            "complex_name": complex_name,
            "case_group": row["case_group"],
            "repair_action": action,
            "config": str(config),
            "config_backup": str(backup),
            "stage1_output": str(stage1_output),
            **details,
        })

    retry_path = output / "retry12_cases.csv"
    fields = list(repaired_rows[0])
    with retry_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(repaired_rows)
    audit_path = output / "repair_audit.csv"
    audit_fields = sorted({key for row in audit_rows for key in row})
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    summary = {
        "source_cases": str(cases_path),
        "retry_cases": str(retry_path),
        "num_retry_cases": len(repaired_rows),
        "running_cases_excluded": ["9D6C"],
        "preprocessing_directory_repairs": list(PREPROCESS_CASES),
        "hydrogen_free_receptor_repairs": list(RECEPTOR_CASES),
        "stereo_from_3d_repairs": list(STEREO_CASES),
        "retry_order": list(RETRY_ORDER),
        "original_outputs_deleted": False,
        "source_cases_sha256": sha256(cases_path),
        "retry_cases_sha256": sha256(retry_path),
        "repair_audit_sha256": sha256(audit_path),
    }
    (output / "repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CHEMEM100_REPAIR_OK: {retry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
