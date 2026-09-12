#!/usr/bin/env python3
"""Build an oracle-pocket source manifest from the CryoDiffDock dataset view."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_preparation import (  # noqa: E402
    STANDARD_PROTEIN_RESIDUES,
    read_ligand_heavy_coordinates,
)


def read_ids(path: Path):
    return [
        line.strip().split()[0].upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def map_lookup(path: Path):
    lookup = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pdb_id = str(row.get("pdb_id") or row.get("complex_name") or "").strip().upper()
            value = str(row.get("map_path") or "").strip()
            status = str(row.get("status") or "ok").strip().lower()
            if not pdb_id or not value or status not in {"ok", "success", "downloaded"}:
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            lookup[pdb_id] = candidate.resolve()
    return lookup


def flexible_residues(protein_path: Path, ligand_coordinates: np.ndarray, cutoff: float) -> str:
    import gemmi

    structure = gemmi.read_structure(str(protein_path))
    selected = []
    seen = set()
    cutoff_squared = float(cutoff) ** 2
    backbone_names = {"N", "CA", "C", "O", "OXT"}
    no_flexible_chi = {"ALA", "GLY", "PRO"}
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_name = residue.name.upper()
                if (
                    residue_name not in STANDARD_PROTEIN_RESIDUES
                    or residue_name in no_flexible_chi
                ):
                    continue
                coordinates = np.asarray(
                    [
                        (atom.pos.x, atom.pos.y, atom.pos.z)
                        for atom in residue
                        if atom.element.name not in {"H", "D"}
                        and atom.name.strip().upper() not in backbone_names
                    ],
                    dtype=np.float32,
                )
                if coordinates.size == 0:
                    continue
                delta = coordinates[:, None, :] - ligand_coordinates[None, :, :]
                if float(np.square(delta).sum(axis=-1).min()) > cutoff_squared:
                    continue
                key = f"{chain.name}:{residue.seqid.num}"
                if key not in seen:
                    seen.add(key)
                    selected.append(key)
        break
    if not selected:
        raise ValueError(f"No protein residues within {cutoff:.1f} A of reference ligand")
    return "-".join(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--map-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5, help="Number of valid cases; 0 means all")
    parser.add_argument("--flexible-distance", type=float, default=3.5)
    parser.add_argument(
        "--no-flexible-policy",
        choices=("error", "rigid"),
        default="error",
        help=(
            "Action when no torsion-bearing protein side chain is within the "
            "frozen cutoff. 'error' preserves the original behavior; 'rigid' "
            "writes an explicit zero-flexible-residue sentinel."
        ),
    )
    parser.add_argument("--protein-suffix", default="protein_processed")
    parser.add_argument("--ligand-suffix", default="ligand")
    parser.add_argument("--reference-suffix", default="ligand_deposited")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.limit < 0 or args.flexible_distance <= 0:
        raise ValueError("limit must be non-negative and flexible-distance must be positive")

    split = args.split.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve()
    map_csv = args.map_csv.expanduser().resolve()
    maps = map_lookup(map_csv)
    rows = []
    failures = []
    for pdb_id in read_ids(split):
        try:
            complex_dir = data_dir / pdb_id
            protein_path = complex_dir / f"{pdb_id}_{args.protein_suffix}.pdb"
            ligand_path = complex_dir / f"{pdb_id}_{args.ligand_suffix}.sdf"
            reference_path = reference_dir / pdb_id / f"{pdb_id}_{args.reference_suffix}.sdf"
            map_path = maps.get(pdb_id)
            for path in (protein_path, ligand_path, reference_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            if map_path is None or not map_path.is_file():
                raise FileNotFoundError(f"No readable map for {pdb_id} in {map_csv}")
            reference_coordinates = read_ligand_heavy_coordinates(reference_path)
            try:
                flexible = flexible_residues(
                    protein_path, reference_coordinates, args.flexible_distance
                )
                sidechain_policy = "reference_ligand_atoms"
                num_flexible = len(flexible.split("-"))
            except ValueError:
                if args.no_flexible_policy == "error":
                    raise
                # Keep predefined side-chain selection active without selecting
                # any receptor atoms.  The model has an explicit zero-flexible-
                # bond path; the same sentinel is used by the frozen cascade.
                flexible = "__NONE__:0"
                sidechain_policy = "rigid_receptor_fallback"
                num_flexible = 0
            rows.append({
                "complex_name": pdb_id,
                "experimental_protein": str(protein_path.resolve()),
                "ligand": str(ligand_path.resolve()),
                "reference_ligand": str(reference_path.resolve()),
                "map_path": str(map_path),
                "flexible_sidechains": flexible,
                "sidechain_policy": sidechain_policy,
                "num_flexible_sidechains": num_flexible,
            })
            print(f"OK {pdb_id}: {flexible} ({sidechain_policy})")
            if args.limit and len(rows) >= args.limit:
                break
        except Exception as exc:
            failures.append({"complex_name": pdb_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"SKIP {pdb_id}: {exc}")
            if args.fail_on_error:
                raise

    if not rows:
        raise RuntimeError("No complete cases were found")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    pd.DataFrame(failures).to_csv(output.with_suffix(".failures.csv"), index=False)
    print(f"ORACLE_SOURCE_MANIFEST_OK: {len(rows)} cases -> {output}")
    print(f"Skipped cases: {len(failures)} -> {output.with_suffix('.failures.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
