#!/usr/bin/env python3
"""Prepare and preflight a frozen ChemEM Stage-1 work set.

The work set contains the 42 newly selected cases plus explicitly requested
technical retries from the original 58-case cohort. It never reads RMSD or
docking scores. Receptors are rebuilt deterministically from the processed
protein PDB with PDBFixer and checked against the Amber protein force field
before any expensive fitting job is submitted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Iterable


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex")
REPAIR14 = (
    "7OCE", "8C1P", "8JD0", "8JD3", "8YHQ", "8ZOS", "8ZP0",
    "9EO4", "9MNM", "9P9B", "9P9E", "9P9F", "9RMW", "9Z6V",
)
DO_NOT_RETRY = ("9D6D",)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def choose_id_column(columns: Iterable[str], path: Path) -> str:
    for column in ID_COLUMNS:
        if column in columns:
            return column
    raise ValueError(f"No complex ID column in {path}; columns={list(columns)}")


def index_rows(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    if not rows:
        raise ValueError(f"Empty table: {path}")
    id_column = choose_id_column(rows[0], path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        complex_name = normalize_id(row[id_column])
        if complex_name in indexed:
            raise ValueError(f"Duplicate complex ID {complex_name} in {path}")
        indexed[complex_name] = row
    return indexed


def read_ids(path: Path) -> list[str]:
    values = [normalize_id(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(row: dict[str, str], columns: tuple[str, ...]) -> list[float]:
    values = []
    for column in columns:
        if column not in row:
            raise ValueError(f"Missing {column}")
        value = float(row[column])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {column}={row[column]!r}")
        values.append(value)
    return values


def ligand_smiles(path: Path) -> tuple[str, int, int]:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required; run in the ChemEM environment") from exc
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    molecules = [molecule for molecule in supplier if molecule is not None]
    if not molecules:
        raise ValueError(f"RDKit could not read a valid ligand from {path}")
    molecule = molecules[0]
    smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
    if not smiles:
        raise ValueError(f"RDKit produced empty SMILES for {path}")
    return smiles, molecule.GetNumHeavyAtoms(), len(molecules)


def check_map(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    if opener is gzip.open:
        with gzip.open(path, "rb") as handle:
            header = handle.read(1024)
    else:
        with path.open("rb") as handle:
            header = handle.read(1024)
    if len(header) < 1024:
        raise ValueError(f"Map header is shorter than 1024 bytes: {path}")
    return path.stat().st_size


def canonicalize_pdb_records(source: Path, output: Path) -> dict[str, int]:
    """Write a single-model PDB without orphan records that break OpenMM.

    Some processed structures contain TER records before the first coordinate
    record in a chain. OpenMM's PDB reader then attempts to terminate a chain
    that has not been created. Coordinates remain unchanged; malformed atom
    records are omitted so PDBFixer can reconstruct the missing atom.
    """
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    saw_model = False
    inside_first_model = True
    first_model_finished = False
    chain_has_atoms = False
    atom_records = 0
    skipped_orphan_ter = 0
    skipped_invalid_coordinates = 0
    with source.open(errors="replace") as input_handle, output.open("w") as output_handle:
        for raw_line in input_handle:
            line = raw_line.rstrip("\r\n")
            record = line[:6].strip().upper()
            if record == "MODEL":
                if saw_model:
                    inside_first_model = False
                else:
                    saw_model = True
                    inside_first_model = True
                continue
            if record == "ENDMDL":
                if saw_model and inside_first_model:
                    first_model_finished = True
                    break
                continue
            if first_model_finished or not inside_first_model:
                continue
            if record in {"ATOM", "HETATM"}:
                try:
                    coordinates = [float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54))]
                except (ValueError, IndexError):
                    skipped_invalid_coordinates += 1
                    continue
                if not all(math.isfinite(value) for value in coordinates):
                    skipped_invalid_coordinates += 1
                    continue
                output_handle.write(line + "\n")
                atom_records += 1
                chain_has_atoms = True
            elif record == "TER":
                if chain_has_atoms:
                    output_handle.write("TER\n")
                    chain_has_atoms = False
                else:
                    skipped_orphan_ter += 1
            # Header, ANISOU, CONECT, MASTER and duplicated END records are not
            # needed for topology reconstruction and are intentionally omitted.
        if chain_has_atoms:
            output_handle.write("TER\n")
        output_handle.write("END\n")
    if atom_records == 0:
        raise ValueError(f"No valid coordinate records in {source}")
    return {
        "source_atom_records": atom_records,
        "skipped_orphan_ter": skipped_orphan_ter,
        "skipped_invalid_coordinates": skipped_invalid_coordinates,
    }


def fix_and_validate_receptor(source: Path, output: Path, ph: float) -> dict[str, int]:
    try:
        from pdbfixer import PDBFixer
        from openmm.app import ForceField, HBonds, NoCutoff, PDBFile
    except ImportError as exc:
        raise RuntimeError(
            "PDBFixer and OpenMM are required. Install pdbfixer from conda-forge "
            "inside the ChemEM environment if it is missing."
        ) from exc

    canonical = output.with_name(output.stem + "_canonical_input.pdb")
    cleanup = canonicalize_pdb_records(source, canonical)
    fixer = PDBFixer(filename=str(canonical))
    fixer.removeHeterogens(False)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    # Avoid modelling unresolved loops unrelated to this docking benchmark.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=False)

    pdb = PDBFile(str(output))
    forcefield = ForceField("amber14/protein.ff14SB.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    atom_count = sum(1 for _atom in pdb.topology.atoms())
    residue_count = sum(1 for _residue in pdb.topology.residues())
    if atom_count == 0 or residue_count == 0 or system.getNumParticles() != atom_count:
        raise ValueError(f"Invalid fixed receptor produced from {source}")
    return {
        "receptor_atoms": atom_count,
        "receptor_residues": residue_count,
        **cleanup,
    }


def write_config(
    path: Path,
    protein: Path,
    smiles: str,
    centroid: list[float],
    output: Path,
    densmap: Path,
    resolution: float,
    platform: str,
    n_cpu: int,
) -> None:
    text = f"""protein = {protein}
ligand = {smiles}
centroid = ({centroid[0]}, {centroid[1]}, {centroid[2]})
output = {output}
densmap = {densmap}
resolution = {resolution}
pre_process = 1
pre_process_split_density = 1
auto_split_point = 1
auto_split_zone = 0
fitting = 1
dock_only = 0
post_process = 0
rescore = 0
protonation = 0
chirality = 1
rings = 1
platform = {platform}
n_cpu = {n_cpu}
overwrite = 1
segment_dimensions = (30, 30, 30)
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--new-case-ids", required=True, type=Path)
    parser.add_argument("--local-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--platform", default="OpenCL")
    parser.add_argument("--n-cpu", type=int, default=8)
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--repair-id", action="append", default=[])
    parser.add_argument("--skip-id", action="append", default=[])
    parser.add_argument(
        "--no-default-repairs", action="store_true",
        help="Do not add the historical ChemEM100 14-case repair list.",
    )
    parser.add_argument(
        "--no-default-skips", action="store_true",
        help=(
            "Do not add the historical ChemEM100 do-not-retry list. Use this "
            "for a new outcome-independent cohort in which every member must "
            "be attempted and technical failures retained."
        ),
    )
    parser.add_argument(
        "--expected-new-count", type=int, default=42,
        help="Required size of --new-case-ids (default preserves ChemEM100).",
    )
    parser.add_argument(
        "--expected-repair-count", type=int, default=14,
        help="Required repair-list size (default preserves ChemEM100).",
    )
    parser.add_argument("--new-case-group", default="new42")
    parser.add_argument("--repair-case-group", default="repair14")
    args = parser.parse_args()

    cohort_path = args.cohort.expanduser().resolve()
    new_ids_path = args.new_case_ids.expanduser().resolve()
    manifest_path = args.local_manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cohort = index_rows(read_csv(cohort_path), cohort_path)
    manifest = index_rows(read_csv(manifest_path), manifest_path)
    new_ids = read_ids(new_ids_path)
    if args.no_default_repairs:
        repair_source = args.repair_id
    else:
        repair_source = args.repair_id or REPAIR14
    repair_ids = [normalize_id(value) for value in repair_source]
    if args.no_default_skips:
        skip_source = args.skip_id
    else:
        skip_source = args.skip_id or DO_NOT_RETRY
    skip_ids = {normalize_id(value) for value in skip_source}

    if args.expected_new_count < 0 or args.expected_repair_count < 0:
        raise ValueError("Expected counts must be non-negative")
    if len(new_ids) != args.expected_new_count:
        raise ValueError(
            f"Expected {args.expected_new_count} new cases, found {len(new_ids)}"
        )
    if (
        len(repair_ids) != args.expected_repair_count
        or len(repair_ids) != len(set(repair_ids))
    ):
        raise ValueError(
            f"Expected {args.expected_repair_count} unique repair cases, "
            f"found {len(set(repair_ids))}"
        )
    if set(new_ids) & set(repair_ids):
        raise ValueError("New and repair case lists overlap")
    if skip_ids & (set(new_ids) | set(repair_ids)):
        raise ValueError("A skipped ID entered the work set")

    work = [(complex_name, args.repair_case_group) for complex_name in repair_ids]
    work.extend((complex_name, args.new_case_group) for complex_name in new_ids)
    expected_work_count = args.expected_repair_count + args.expected_new_count
    if len(work) != expected_work_count:
        raise RuntimeError(
            f"Expected {expected_work_count} work cases, found {len(work)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for work_index, (complex_name, case_group) in enumerate(work):
        print(
            f"[{work_index + 1}/{expected_work_count}] "
            f"{complex_name} {case_group}: START",
            flush=True,
        )
        try:
            if complex_name not in cohort:
                raise ValueError(f"Absent from frozen ChemEM100 cohort: {complex_name}")
            if complex_name not in manifest:
                raise ValueError(f"Absent from local manifest: {complex_name}")
            source = manifest[complex_name]
            protein_source = Path(source["experimental_protein"]).expanduser().resolve()
            ligand_source = Path(source["ligand"]).expanduser().resolve()
            map_source = Path(source["map_path"]).expanduser().resolve()
            centroid = require_finite(
                source, ("pocket_center_x", "pocket_center_y", "pocket_center_z")
            )
            resolution = float(cohort[complex_name]["resolution_A"])
            if not math.isfinite(resolution) or resolution <= 0:
                raise ValueError(f"Invalid resolution: {resolution}")

            case_input = output_root / "inputs" / complex_name
            receptor = case_input / f"{complex_name}_protein_fixed.pdb"
            densmap = case_input / f"{complex_name}_experimental.map"
            config = output_root / "configs" / f"{complex_name}_stage1.conf"
            stage1_output = output_root / "runs" / complex_name / "stage1"
            log_path = output_root / "logs" / f"{complex_name}_stage1_opencl.log"

            smiles, ligand_heavy_atoms, ligand_records = ligand_smiles(ligand_source)
            map_compressed_bytes = check_map(map_source)
            receptor_counts = fix_and_validate_receptor(protein_source, receptor, args.ph)
            write_config(
                config, receptor, smiles, centroid, stage1_output, densmap,
                resolution, args.platform, args.n_cpu,
            )
            rows.append({
                "work_index": work_index,
                "complex_name": complex_name,
                "case_group": case_group,
                "resolution_A": resolution,
                "protein_source": str(protein_source),
                "protein_fixed": str(receptor),
                "protein_fixed_sha256": sha256(receptor),
                "receptor_atoms": receptor_counts["receptor_atoms"],
                "receptor_residues": receptor_counts["receptor_residues"],
                "source_atom_records": receptor_counts["source_atom_records"],
                "skipped_orphan_ter": receptor_counts["skipped_orphan_ter"],
                "skipped_invalid_coordinates": receptor_counts["skipped_invalid_coordinates"],
                "ligand_source": str(ligand_source),
                "ligand_smiles": smiles,
                "ligand_heavy_atoms": ligand_heavy_atoms,
                "ligand_sdf_records": ligand_records,
                "map_source": str(map_source),
                "map_compressed_bytes": map_compressed_bytes,
                "decompressed_map": str(densmap),
                "stage1_config": str(config),
                "stage1_output": str(stage1_output),
                "case_log": str(log_path),
                "preflight_status": "pass",
            })
            print(
                f"[{work_index + 1}/{expected_work_count}] "
                f"{complex_name} {case_group}: PASS",
                flush=True,
            )
        except Exception as exc:
            error_log = output_root / "preflight_error_logs" / f"{complex_name}.log"
            error_log.parent.mkdir(parents=True, exist_ok=True)
            error_log.write_text(traceback.format_exc())
            failures.append({
                "work_index": str(work_index),
                "complex_name": complex_name,
                "case_group": case_group,
                "error": f"{type(exc).__name__}: {exc}",
                "error_log": str(error_log),
            })
            print(
                f"[{work_index + 1}/{expected_work_count}] "
                f"{complex_name} {case_group}: FAILED: {exc}",
                flush=True,
            )

    table_path = output_root / "cases_to_run.csv"
    fields = list(rows[0]) if rows else ["work_index", "complex_name", "case_group"]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failure_path = output_root / "preflight_failures.csv"
    with failure_path.open("w", newline="") as handle:
        fields = ["work_index", "complex_name", "case_group", "error", "error_log"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)

    summary = {
        "frozen_cohort": str(cohort_path),
        "new_case_ids": str(new_ids_path),
        "local_manifest": str(manifest_path),
        "requested_new_cases": args.expected_new_count,
        "requested_repair_cases": args.expected_repair_count,
        "do_not_retry": sorted(skip_ids),
        "requested_work_cases": expected_work_count,
        "preflight_passed": len(rows),
        "preflight_failed": len(failures),
        "uses_docking_outcomes": False,
        "receptor_policy": (
            "PDBFixer on processed protein; no unresolved-loop modelling; "
            "add missing atoms and pH 7 hydrogens; validate Amber14 protein parameterization"
        ),
        "input_sha256": {
            "cohort": sha256(cohort_path),
            "new_case_ids": sha256(new_ids_path),
            "local_manifest": sha256(manifest_path),
        },
    }
    (output_root / "preflight_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_root / "do_not_retry_ids.txt").write_text("\n".join(sorted(skip_ids)) + "\n")
    (output_root / "repair14_ids.txt").write_text("\n".join(repair_ids) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        print(f"CHEMEM100_PREFLIGHT_FAILED: see {failure_path}")
        return 1
    print(f"CHEMEM100_PREFLIGHT_OK: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
