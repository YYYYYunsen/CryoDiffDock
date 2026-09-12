#!/usr/bin/env python3
"""Compatibility wrapper for the publication EMERALD ligand utilities.

The Rosetta AM1 reader parses MOL2 atom names and coordinates from fixed
columns.  The current ``normalize-mol2-atom-names`` implementation in the
external benchmark tools rebuilds atom records with ``" ".join(fields)``,
which destroys those columns.  This wrapper also repairs the exceptional
all-zero OpenBabel conformers from the same input SMILES.  Handle only that
subcommand here while delegating every other subcommand unchanged.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def _die(message: str) -> "NoReturn":
    raise SystemExit(f"ERROR: {message}")


def _element_from_fields(fields: list[str], line: str) -> str:
    atom_type = fields[5].split(".", 1)[0]
    letters = "".join(char for char in atom_type if char.isalpha())
    if not letters:
        letters = "".join(char for char in fields[1] if char.isalpha())
    if not letters:
        _die(f"cannot infer MOL2 element from atom record: {line!r}")
    if letters[:2].upper() in {"CL", "BR"}:
        return letters[:2].upper()
    return letters[0].upper()


def _normalize_atom_names(input_path: Path, output_path: Path) -> None:
    output: list[str] = []
    in_atom_section = False
    counts: dict[str, int] = {}
    renamed = 0

    for line_number, line in enumerate(input_path.read_text().splitlines(), 1):
        if line.startswith("@<TRIPOS>"):
            in_atom_section = line.strip().upper() == "@<TRIPOS>ATOM"
            output.append(line)
            continue
        if not in_atom_section or not line.strip():
            output.append(line)
            continue

        tokens = list(re.finditer(r"\S+", line))
        fields = [token.group(0) for token in tokens]
        if len(fields) < 6:
            _die(f"malformed MOL2 atom record at line {line_number}: {line!r}")

        element = _element_from_fields(fields, line)
        counts[element] = counts.get(element, 0) + 1
        atom_name = f"{element}{counts[element]}"

        # Keep the first coordinate at its original byte offset.  Rosetta's
        # AM1.py reads name=line[7:11] and XYZ from line[16:26], [26:36],
        # [36:46], so changing the overall atom-name field width is invalid.
        name_start = tokens[1].start()
        coordinate_start = tokens[2].start()
        available_width = coordinate_start - name_start
        if len(atom_name) >= available_width:
            _die(
                f"atom name {atom_name!r} does not fit MOL2 field at line "
                f"{line_number}: {line!r}"
            )
        normalized = (
            line[:name_start]
            + atom_name.ljust(available_width)
            + line[coordinate_start:]
        )

        try:
            float(normalized[16:26])
            float(normalized[26:36])
            float(normalized[36:46])
        except ValueError as exc:
            _die(
                "input MOL2 is not in the fixed-column layout required by "
                f"Rosetta AM1.py at line {line_number}: {normalized!r} ({exc})"
            )
        if not normalized[7:11].strip():
            _die(f"Rosetta atom-name columns are empty at line {line_number}")

        output.append(normalized)
        renamed += 1

    if renamed == 0:
        _die(f"no MOL2 atom records found in {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output) + "\n")


def _count_all_zero_atoms(path: Path) -> int:
    in_atom_section = False
    count = 0
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if line.startswith("@<TRIPOS>"):
            in_atom_section = line.strip().upper() == "@<TRIPOS>ATOM"
            continue
        if not in_atom_section or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 9:
            _die(f"malformed MOL2 atom record at line {line_number}: {line!r}")
        try:
            xyz = tuple(float(value) for value in fields[2:5])
        except ValueError as exc:
            _die(f"invalid MOL2 coordinates at line {line_number}: {exc}")
        if all(value == 0.0 for value in xyz):
            count += 1
    return count


def _apply_rdkit_fallback(input_path: Path, output_path: Path) -> Path:
    enabled = os.environ.get("EMERALD_ENABLE_RDKIT_FALLBACK", "0") == "1"
    zero_count = _count_all_zero_atoms(input_path)
    if zero_count <= 1:
        return input_path
    if not enabled:
        _die(
            f"input MOL2 contains {zero_count} all-zero-coordinate atoms and "
            "EMERALD_ENABLE_RDKIT_FALLBACK is not enabled"
        )

    helper = Path(os.environ.get("EMERALD_RDKIT_FALLBACK_SCRIPT", ""))
    python = Path(os.environ.get("EMERALD_RDKIT_PYTHON", ""))
    smiles_file = input_path.with_name("ligand.smi")
    for label, path in (
        ("RDKit fallback helper", helper),
        ("RDKit fallback Python", python),
        ("fallback SMILES", smiles_file),
    ):
        if not path.is_file():
            _die(f"missing {label}: {path}")

    repaired_path = output_path.with_name("ligand_raw_rdkit_fallback.mol2")
    audit_path = output_path.with_name("rdkit_conformer_fallback_audit.json")
    command = [
        str(python),
        str(helper),
        "--input-mol2",
        str(input_path),
        "--smiles-file",
        str(smiles_file),
        "--output-mol2",
        str(repaired_path),
        "--audit",
        str(audit_path),
        "--seed",
        os.environ.get("EMERALD_RDKIT_FALLBACK_SEED", "2026"),
    ]
    print(
        f"[compat] detected {zero_count} all-zero-coordinate atoms; "
        "applying deterministic SMILES-only RDKit fallback",
        file=sys.stderr,
    )
    subprocess.run(command, check=True)
    repaired_zero_count = _count_all_zero_atoms(repaired_path)
    if repaired_zero_count > 1:
        _die(
            f"RDKit fallback left {repaired_zero_count} all-zero-coordinate atoms"
        )
    return repaired_path


def _run_normalize(argv: list[str]) -> int:
    input_value = None
    output_value = None
    index = 0
    while index < len(argv):
        if argv[index] == "--input-mol2" and index + 1 < len(argv):
            input_value = argv[index + 1]
            index += 2
        elif argv[index] == "--output-mol2" and index + 1 < len(argv):
            output_value = argv[index + 1]
            index += 2
        else:
            _die(f"unrecognized normalize argument: {argv[index]}")
    if not input_value or not output_value:
        _die("normalize-mol2-atom-names requires --input-mol2 and --output-mol2")
    input_path = Path(input_value)
    output_path = Path(output_value)
    coordinate_source = _apply_rdkit_fallback(input_path, output_path)
    _normalize_atom_names(coordinate_source, output_path)
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "normalize-mol2-atom-names":
        return _run_normalize(sys.argv[2:])

    real_utility = os.environ.get("EMERALD_REAL_LIGAND_UTIL_PY", "")
    if not real_utility or not Path(real_utility).is_file():
        _die(
            "EMERALD_REAL_LIGAND_UTIL_PY must point to the original "
            "ligand_pose_utils.py"
        )
    os.execv(sys.executable, [sys.executable, real_utility, *sys.argv[1:]])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
