#!/usr/bin/env python3
"""Prepare an auditable EMERALD receptor compatibility input for 9NR7.

The deposited full-polymer receptor for 9NR7 contains two chains made
entirely from ``UNK`` residues.  Rosetta cannot construct those residues.
This helper removes only those explicitly requested all-UNK chains after
verifying that every removed atom is farther than a conservative distance
from the supplied docking-site centre.  All retained PDB records are copied
byte-for-byte, and a one-row manifest substitution plus a JSON audit are
written for an isolated recovery run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ATOM_RECORDS = ("ATOM  ", "HETATM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-receptor", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receptor", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--case-id", default="9NR7")
    parser.add_argument("--remove-chain", action="append", required=True)
    parser.add_argument("--minimum-distance-A", type=float, default=20.0)
    return parser.parse_args()


def read_manifest(path: Path, case_id: str) -> tuple[list[dict[str, str]], list[str], int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        rows = list(reader)
        fields = list(reader.fieldnames)

    required = {"complex_name", "mask_structure", "pocket_center_x", "pocket_center_y", "pocket_center_z"}
    missing = sorted(required - set(fields))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    matching = [
        index
        for index, row in enumerate(rows)
        if row["complex_name"].strip().upper() == case_id
    ]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one {case_id} manifest row, found {len(matching)}")
    return rows, fields, matching[0]


def main() -> int:
    args = parse_args()
    case_id = args.case_id.strip().upper()
    remove_chains = set(args.remove_chain)
    if case_id != "9NR7":
        raise ValueError("this compatibility repair is intentionally restricted to 9NR7")
    if remove_chains != {"E", "G"}:
        raise ValueError("9NR7 compatibility repair requires exactly chains E and G")
    if args.minimum_distance_A < 20.0:
        raise ValueError("minimum-distance safety threshold must be at least 20 A")
    for path in (args.input_manifest, args.source_receptor):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows, fields, row_index = read_manifest(args.input_manifest, case_id)
    row = rows[row_index]
    center = tuple(
        float(row[column])
        for column in ("pocket_center_x", "pocket_center_y", "pocket_center_z")
    )

    lines = args.source_receptor.read_text().splitlines(keepends=True)
    chain_residues: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    chain_resnames: dict[str, Counter[str]] = defaultdict(Counter)
    chain_atoms: Counter[str] = Counter()
    chain_minimum_distance: dict[str, float] = defaultdict(lambda: math.inf)
    removed_serials: set[int] = set()
    retained_atom_count = 0

    for line_number, line in enumerate(lines, 1):
        if not line.startswith(ATOM_RECORDS):
            continue
        if len(line) < 54:
            raise ValueError(f"short PDB atom record at line {line_number}")
        chain = line[21:22]
        residue = (line[22:26], line[26:27], line[17:20].strip())
        resname = residue[2]
        xyz = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
        distance = math.sqrt(sum((value - origin) ** 2 for value, origin in zip(xyz, center)))
        chain_residues[chain].add(residue)
        chain_resnames[chain][resname] += 1
        chain_atoms[chain] += 1
        chain_minimum_distance[chain] = min(chain_minimum_distance[chain], distance)
        if chain in remove_chains:
            removed_serials.add(int(line[6:11]))
        else:
            retained_atom_count += 1

    if retained_atom_count == 0:
        raise ValueError("repair would leave no receptor atoms")
    unexpected_unk_chains = sorted(
        chain for chain, names in chain_resnames.items()
        if "UNK" in names and chain not in remove_chains
    )
    if unexpected_unk_chains:
        raise ValueError(f"UNK residues also occur outside E/G: {unexpected_unk_chains}")

    chain_audit: dict[str, dict[str, object]] = {}
    for chain in sorted(remove_chains):
        names = chain_resnames.get(chain, Counter())
        if not names:
            raise ValueError(f"requested chain {chain} is absent")
        if set(names) != {"UNK"}:
            raise ValueError(f"chain {chain} is not entirely UNK: {dict(names)}")
        minimum = chain_minimum_distance[chain]
        if minimum < args.minimum_distance_A:
            raise ValueError(
                f"chain {chain} approaches the site at {minimum:.3f} A, below "
                f"the {args.minimum_distance_A:.3f} A safety threshold"
            )
        chain_audit[chain] = {
            "atom_count": chain_atoms[chain],
            "residue_count": len(chain_residues[chain]),
            "residue_names": dict(names),
            "minimum_distance_to_site_A": minimum,
        }

    output_lines: list[str] = []
    removed_atom_count = 0
    removed_anisou_count = 0
    removed_conect_count = 0
    for line in lines:
        if line.startswith(ATOM_RECORDS) and line[21:22] in remove_chains:
            removed_atom_count += 1
            continue
        if line.startswith("ANISOU") and line[21:22] in remove_chains:
            removed_anisou_count += 1
            continue
        if line.startswith("CONECT"):
            serials = []
            for start in range(6, len(line.rstrip("\n")), 5):
                value = line[start : start + 5].strip()
                if value.isdigit():
                    serials.append(int(value))
            if any(serial in removed_serials for serial in serials):
                removed_conect_count += 1
                continue
        output_lines.append(line)

    if removed_atom_count != sum(chain_atoms[chain] for chain in remove_chains):
        raise AssertionError("removed-atom accounting mismatch")

    args.output_receptor.parent.mkdir(parents=True, exist_ok=True)
    args.output_receptor.write_text("".join(output_lines))
    retained_lines = [line for line in output_lines if line.startswith(ATOM_RECORDS)]
    if any(line[17:20].strip() == "UNK" for line in retained_lines):
        raise AssertionError("sanitized receptor still contains UNK atoms")
    if len(retained_lines) != retained_atom_count:
        raise AssertionError("retained-atom accounting mismatch")

    original_mask_structure = row["mask_structure"]
    rows[row_index] = dict(row)
    rows[row_index]["mask_structure"] = str(args.output_receptor.resolve())
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "case_id": case_id,
        "repair": "remove_remote_all_UNK_chains",
        "outcome_or_native_ligand_information_used": False,
        "site_center_A": center,
        "minimum_distance_safety_threshold_A": args.minimum_distance_A,
        "removed_chains": sorted(remove_chains),
        "removed_chain_details": chain_audit,
        "removed_atom_count": removed_atom_count,
        "removed_anisou_count": removed_anisou_count,
        "removed_conect_count": removed_conect_count,
        "retained_atom_count": retained_atom_count,
        "retained_records_copied_byte_for_byte": True,
        "original_mask_structure_manifest_value": original_mask_structure,
        "source_receptor": str(args.source_receptor.resolve()),
        "source_receptor_sha256": sha256(args.source_receptor),
        "output_receptor": str(args.output_receptor.resolve()),
        "output_receptor_sha256": sha256(args.output_receptor),
        "input_manifest": str(args.input_manifest.resolve()),
        "input_manifest_sha256": sha256(args.input_manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"EMERALD_9NR7_RECEPTOR_COMPAT_OK: {args.output_receptor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
