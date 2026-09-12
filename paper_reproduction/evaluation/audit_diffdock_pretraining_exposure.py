#!/usr/bin/env python3
"""Audit CryoLigandDB test contact proteins against DiffDock pretraining.

The audit deliberately separates three questions:

1. Does a test PDB identifier occur verbatim in the training split?
2. Is a ligand-contacting test protein entity homologous to *any* receptor
   chain presented to the pretrained model (the conservative primary audit)?
3. Is it homologous specifically to a ligand-contacting training chain?

Reference ligand coordinates are used only to define ligand-contacting protein
entities.  Docking predictions, scores and RMSDs are never read by this script.
The script can prepare FASTA files, run MMseqs2, and summarize the matches in a
single invocation, or those stages can be run separately on a compute server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import gemmi
except ImportError as exc:  # pragma: no cover - environment preflight
    raise SystemExit("gemmi is required (the project environment pins gemmi==0.7.3)") from exc

MMSEQS_COLUMNS = [
    "query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"
]
DEFAULT_PROTEIN_PATTERNS = (
    "{pdb_id_lower}/{pdb_id_lower}_protein_esmfold_aligned_tr_fix.pdb",
    "{pdb_id}/{pdb_id}_protein_esmfold_aligned_tr_fix.pdb",
    "{pdb_id_lower}/{pdb_id_lower}_protein_processed_fix.pdb",
    "{pdb_id}/{pdb_id}_protein_processed_fix.pdb",
    "{pdb_id}/{pdb_id}_protein_processed.pdb",
    "{pdb_id_lower}/{pdb_id_lower}_protein_processed.pdb",
    "{pdb_id_lower}/{pdb_id_lower}_protein_obabel_reduce.pdb",
    "{pdb_id}/{pdb_id}_protein_obabel_reduce.pdb",
    "{pdb_id}/{pdb_id}_protein.pdb",
    "{pdb_id_lower}/{pdb_id_lower}_protein.pdb",
)
DEFAULT_LIGAND_PATTERNS = (
    "{pdb_id}/{pdb_id}_ligand.sdf",
    "{pdb_id_lower}/{pdb_id_lower}_ligand.sdf",
    "{pdb_id}/{pdb_id}_ligand.mol2",
    "{pdb_id_lower}/{pdb_id_lower}_ligand.mol2",
)


@dataclass(frozen=True)
class ChainRecord:
    pdb_id: str
    chain_id: str
    sequence: str
    is_contact: bool
    minimum_ligand_distance_A: float
    protein_path: str
    ligand_path: str


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip().split()[0].upper()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"ID list is empty: {path}")
    if len(values) != len(set(values)):
        duplicate = next(value for value, n in Counter(values).items() if n > 1)
        raise ValueError(f"Duplicate ID {duplicate!r} in {path}")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_existing(root: Path, pdb_id: str, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        candidate = root / pattern.format(pdb_id=pdb_id, pdb_id_lower=pdb_id.lower())
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_raw_cif(root: Path, pdb_id: str) -> Path | None:
    return first_existing(
        root,
        pdb_id,
        (
            "{pdb_id}/{pdb_id}.cif",
            "{pdb_id}/{pdb_id_lower}.cif",
            "{pdb_id}.cif",
            "{pdb_id_lower}.cif",
        ),
    )


def read_ligand_heavy_coordinates(path: Path, *, sanitize: bool = False) -> np.ndarray:
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
    except ImportError as exc:  # pragma: no cover - environment preflight
        raise RuntimeError(
            "RDKit is required to read deposited ligand coordinates"
        ) from exc
    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sanitize)
        molecule = next((mol for mol in supplier if mol is not None), None)
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=sanitize)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Unreadable ligand or missing conformer: {path}")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
    heavy = np.asarray([atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()])
    coordinates = coordinates[heavy]
    if len(coordinates) == 0 or not np.isfinite(coordinates).all():
        raise ValueError(f"Ligand has no finite heavy-atom coordinates: {path}")
    return coordinates


def minimum_distance(points: np.ndarray, ligand: np.ndarray) -> float:
    if not len(points) or not len(ligand):
        return math.inf
    minimum_squared = math.inf
    for start in range(0, len(points), 4096):
        block = points[start : start + 4096]
        delta = block[:, None, :] - ligand[None, :, :]
        minimum_squared = min(minimum_squared, float(np.square(delta).sum(axis=2).min()))
    return math.sqrt(minimum_squared)


def clean_one_letter_sequence(value: str) -> str:
    # pdbx_seq_one_letter_code_can is canonical and may contain line wrapping.
    return "".join(character for character in value.upper() if "A" <= character <= "Z")


def cif_values(block: gemmi.cif.Block, tag: str) -> list[str]:
    return [str(value) for value in block.find_values(tag)]


def extract_test_contact_entities(
    pdb_id: str,
    raw_root: Path,
    ligand_root: Path,
    ligand_pattern: str,
    cutoff: float,
    minimum_sequence_length: int,
) -> tuple[list[dict[str, object]], dict[str, str] | None]:
    cif_path = find_raw_cif(raw_root, pdb_id)
    ligand_path = first_existing(ligand_root, pdb_id, (ligand_pattern,))
    if cif_path is None:
        return [], {"pdb_id": pdb_id, "stage": "test", "error": "raw_mmcif_missing"}
    if ligand_path is None:
        return [], {"pdb_id": pdb_id, "stage": "test", "error": "ligand_sdf_missing"}
    try:
        ligand = read_ligand_heavy_coordinates(ligand_path)
        block = gemmi.cif.read_file(str(cif_path)).sole_block()

        entity_ids = cif_values(block, "_entity_poly.entity_id")
        entity_types = cif_values(block, "_entity_poly.type")
        canonical = cif_values(block, "_entity_poly.pdbx_seq_one_letter_code_can")
        if not (len(entity_ids) == len(entity_types) == len(canonical)):
            raise ValueError("incomplete _entity_poly sequence table")
        sequences = {
            entity_id: clean_one_letter_sequence(sequence)
            for entity_id, entity_type, sequence in zip(entity_ids, entity_types, canonical)
            if "polypeptide" in entity_type.lower()
        }

        xs = cif_values(block, "_atom_site.Cartn_x")
        ys = cif_values(block, "_atom_site.Cartn_y")
        zs = cif_values(block, "_atom_site.Cartn_z")
        atom_entities = cif_values(block, "_atom_site.label_entity_id")
        elements = cif_values(block, "_atom_site.type_symbol")
        models = cif_values(block, "_atom_site.pdbx_PDB_model_num")
        n = len(xs)
        if not n or not all(len(values) == n for values in (ys, zs, atom_entities)):
            raise ValueError("incomplete _atom_site coordinates or entity identifiers")
        if len(elements) != n:
            elements = [""] * n
        if len(models) != n:
            models = [""] * n
        valid_models = [value for value in models if value not in {"", ".", "?"}]
        selected_model = "1" if "1" in valid_models else (valid_models[0] if valid_models else "")

        coordinates: dict[str, list[tuple[float, float, float]]] = {
            entity_id: [] for entity_id in sequences
        }
        for index in range(n):
            entity_id = atom_entities[index]
            if entity_id not in sequences:
                continue
            if selected_model and models[index] not in {selected_model, "", ".", "?"}:
                continue
            if elements[index].strip().upper() in {"H", "D"}:
                continue
            try:
                coordinate = (float(xs[index]), float(ys[index]), float(zs[index]))
            except ValueError:
                continue
            if np.isfinite(coordinate).all():
                coordinates[entity_id].append(coordinate)

        records: list[dict[str, object]] = []
        for entity_id, sequence in sorted(sequences.items()):
            if len(sequence) < minimum_sequence_length or not coordinates[entity_id]:
                continue
            distance = minimum_distance(np.asarray(coordinates[entity_id]), ligand)
            if distance <= cutoff:
                records.append({
                    "pdb_id": pdb_id,
                    "entity_id": entity_id,
                    "entity_key": f"{pdb_id}_{entity_id}",
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "minimum_ligand_distance_A": distance,
                    "raw_cif": str(cif_path),
                    "ligand_path": str(ligand_path),
                })
        if not records:
            raise ValueError(f"no protein entity within {cutoff:g} A of the ligand")
        return records, None
    except Exception as exc:
        return [], {
            "pdb_id": pdb_id,
            "stage": "test",
            "error": f"{type(exc).__name__}: {exc}",
        }


def residue_one_letter(residue: gemmi.Residue) -> str | None:
    info = gemmi.find_tabulated_residue(residue.name)
    if not info.is_amino_acid():
        return None
    code = str(info.one_letter_code).upper()
    return code if len(code) == 1 and code.isalpha() else "X"


def extract_training_chains(
    pdb_id: str,
    receptor_root: Path,
    protein_patterns: Sequence[str],
    ligand_patterns: Sequence[str],
    cutoff: float,
    minimum_sequence_length: int,
    require_training_ligand: bool,
) -> tuple[list[ChainRecord], dict[str, str] | None]:
    ligand_path: Path | None = None
    ligand: np.ndarray | None = None
    ligand_errors: list[str] = []
    for pattern in ligand_patterns:
        candidate = receptor_root / pattern.format(
            pdb_id=pdb_id, pdb_id_lower=pdb_id.lower()
        )
        if not candidate.is_file():
            continue
        try:
            ligand = read_ligand_heavy_coordinates(
                candidate, sanitize=require_training_ligand
            )
            ligand_path = candidate.resolve()
            break
        except Exception as exc:
            ligand_errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    ligand_error = "" if ligand is not None else (
        "; ".join(ligand_errors) if ligand_errors else "ligand_missing"
    )
    if require_training_ligand and ligand is None:
        return [], {
            "pdb_id": pdb_id,
            "stage": "training_excluded",
            "error": ligand_error,
        }

    protein_path = first_existing(receptor_root, pdb_id, protein_patterns)
    if protein_path is None:
        return [], {"pdb_id": pdb_id, "stage": "training", "error": "protein_missing"}
    try:
        structure = gemmi.read_structure(str(protein_path))
        if len(structure) == 0:
            raise ValueError("protein structure has no model")
        model = structure[0]
        records: list[ChainRecord] = []
        for chain_index, chain in enumerate(model):
            sequence: list[str] = []
            coordinates: list[tuple[float, float, float]] = []
            for residue in chain:
                code = residue_one_letter(residue)
                if code is None:
                    continue
                sequence.append(code)
                for atom in residue:
                    element = str(atom.element.name).upper()
                    if element in {"H", "D"}:
                        continue
                    position = atom.pos
                    coordinate = (float(position.x), float(position.y), float(position.z))
                    if np.isfinite(coordinate).all():
                        coordinates.append(coordinate)
            joined = "".join(sequence)
            if len(joined) < minimum_sequence_length:
                continue
            distance = (
                minimum_distance(np.asarray(coordinates), ligand)
                if ligand is not None and coordinates
                else math.nan
            )
            chain_id = str(chain.name).strip() or f"index{chain_index}"
            records.append(ChainRecord(
                pdb_id=pdb_id,
                chain_id=chain_id,
                sequence=joined,
                is_contact=bool(np.isfinite(distance) and distance <= cutoff),
                minimum_ligand_distance_A=distance,
                protein_path=str(protein_path),
                ligand_path=str(ligand_path or ""),
            ))
        if not records:
            raise ValueError("no protein chain with a usable sequence")
        failure = None
        if ligand_error:
            failure = {
                "pdb_id": pdb_id,
                "stage": "training_contact_annotation",
                "error": ligand_error,
            }
        return records, failure
    except Exception as exc:
        return [], {
            "pdb_id": pdb_id,
            "stage": "training",
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_fasta(path: Path, rows: pd.DataFrame, id_column: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows.itertuples(index=False):
            identifier = str(getattr(row, id_column))
            sequence = str(row.sequence)
            handle.write(f">{identifier}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier: str | None = None
    pieces: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, clean_one_letter_sequence("".join(pieces))))
            identifier = line[1:].split()[0]
            pieces = []
        else:
            if identifier is None:
                raise ValueError(f"FASTA sequence occurs before a header in {path}")
            pieces.append(line)
    if identifier is not None:
        records.append((identifier, clean_one_letter_sequence("".join(pieces))))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def deduplicate_training_sequences(chains: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, (sequence, group) in enumerate(chains.groupby("sequence", sort=True), start=1):
        contacts = group["is_contact"].astype(bool)
        finite_distances = pd.to_numeric(
            group["minimum_ligand_distance_A"], errors="coerce"
        ).dropna()
        records.append({
            "target_id": f"T{index:06d}",
            "sequence": sequence,
            "sequence_length": len(sequence),
            "occurrence_count": len(group),
            "pdb_ids": ";".join(sorted(set(group["pdb_id"].astype(str)))),
            "chain_keys": ";".join(sorted(set(group["chain_key"].astype(str)))),
            "any_ligand_contact": bool(contacts.any()),
            "contact_occurrence_count": int(contacts.sum()),
            "minimum_ligand_distance_A": (
                float(finite_distances.min()) if len(finite_distances) else np.nan
            ),
        })
    return pd.DataFrame(records)


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort = read_ids(args.cohort.resolve())
    training = read_ids(args.train_split.resolve())

    intended_exact = sorted(set(cohort) & set(training))
    (output / "intended_split_exact_pdb_overlaps.txt").write_text(
        "".join(f"{pdb_id}\n" for pdb_id in intended_exact), encoding="utf-8"
    )

    test_records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(
            lambda pdb_id: extract_test_contact_entities(
                pdb_id,
                args.test_raw_root.resolve(),
                args.test_ligand_root.resolve(),
                args.test_ligand_pattern,
                args.contact_cutoff,
                args.minimum_sequence_length,
            ),
            cohort,
        )
        for index, (records, failure) in enumerate(iterator, start=1):
            test_records.extend(records)
            if failure:
                failures.append(failure)
            if index % 25 == 0 or index == len(cohort):
                print(f"test contact entities: {index}/{len(cohort)}", flush=True)

    training_records: list[ChainRecord] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        iterator = executor.map(
            lambda pdb_id: extract_training_chains(
                pdb_id,
                args.train_receptor_root.resolve(),
                args.train_protein_pattern or DEFAULT_PROTEIN_PATTERNS,
                args.train_ligand_pattern or DEFAULT_LIGAND_PATTERNS,
                args.contact_cutoff,
                args.minimum_sequence_length,
                args.require_training_ligand,
            ),
            training,
        )
        for index, (records, failure) in enumerate(iterator, start=1):
            training_records.extend(records)
            if failure:
                failures.append(failure)
            if index % 500 == 0 or index == len(training):
                print(f"training receptors: {index}/{len(training)}", flush=True)

    test = pd.DataFrame(test_records)
    if not test.empty:
        test.insert(0, "query_id", [f"Q{index:05d}" for index in range(1, len(test) + 1)])
    else:
        test = pd.DataFrame(columns=[
            "query_id", "pdb_id", "entity_id", "entity_key", "sequence",
            "sequence_length", "minimum_ligand_distance_A", "raw_cif", "ligand_path",
        ])
    chains = pd.DataFrame([record.__dict__ for record in training_records])
    if not chains.empty:
        chains["chain_key"] = chains["pdb_id"] + "_" + chains["chain_id"]
        chains = chains[[
            "pdb_id", "chain_id", "chain_key", "sequence", "is_contact",
            "minimum_ligand_distance_A", "protein_path", "ligand_path",
        ]]
    else:
        chains = pd.DataFrame(columns=[
            "pdb_id", "chain_id", "chain_key", "sequence", "is_contact",
            "minimum_ligand_distance_A", "protein_path", "ligand_path",
        ])
    auxiliary_rows: list[dict[str, object]] = []
    for fasta_path in args.additional_training_fasta or []:
        resolved_fasta = fasta_path.resolve()
        for identifier, sequence in read_fasta(resolved_fasta):
            if len(sequence) < args.minimum_sequence_length:
                continue
            auxiliary_rows.append({
                "pdb_id": "",
                "chain_id": identifier,
                "chain_key": f"{resolved_fasta.name}:{identifier}",
                "sequence": sequence,
                "is_contact": False,
                "minimum_ligand_distance_A": np.nan,
                "protein_path": str(resolved_fasta),
                "ligand_path": "",
            })
    if auxiliary_rows:
        chains = pd.concat([chains, pd.DataFrame(auxiliary_rows)], ignore_index=True)
    effective_training_ids = set(
        chains.loc[chains["pdb_id"].astype(str) != "", "pdb_id"].astype(str)
    ) if len(chains) else set()
    exact = sorted(set(cohort) & effective_training_ids)
    (output / "exact_pdb_overlaps.txt").write_text(
        "".join(f"{pdb_id}\n" for pdb_id in exact), encoding="utf-8"
    )
    targets = deduplicate_training_sequences(chains) if len(chains) else pd.DataFrame(columns=[
        "target_id", "sequence", "sequence_length", "occurrence_count", "pdb_ids",
        "chain_keys", "any_ligand_contact", "contact_occurrence_count",
        "minimum_ligand_distance_A",
    ])

    test.to_csv(output / "test_contact_entities.csv", index=False)
    chains.to_csv(output / "training_receptor_chains.csv", index=False)
    targets.to_csv(output / "training_unique_sequences.csv", index=False)
    pd.DataFrame(failures, columns=["pdb_id", "stage", "error"]).to_csv(
        output / "extraction_failures.csv", index=False
    )
    write_fasta(output / "test_contact_entities.fasta", test, "query_id")
    write_fasta(output / "training_receptor_sequences.fasta", targets, "target_id")

    test_failures = [row for row in failures if row["stage"] == "test"]
    training_failures = [row for row in failures if row["stage"] == "training"]
    contact_failures = [row for row in failures if row["stage"] == "training_contact_annotation"]
    training_exclusions = [row for row in failures if row["stage"] == "training_excluded"]
    pd.DataFrame(
        training_exclusions, columns=["pdb_id", "stage", "error"]
    ).to_csv(output / "training_exclusions.csv", index=False)
    provenance = {
        "schema_version": "diffdock-pretraining-exposure-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "",
        "checkpoint_sha256": (
            sha256_file(args.checkpoint.resolve()) if args.checkpoint and args.checkpoint.is_file() else ""
        ),
        "cohort": str(args.cohort.resolve()),
        "cohort_sha256": sha256_file(args.cohort.resolve()),
        "cohort_n": len(cohort),
        "train_split": str(args.train_split.resolve()),
        "train_split_sha256": sha256_file(args.train_split.resolve()),
        "train_complex_n": len(training),
        "train_receptor_root": str(args.train_receptor_root.resolve()),
        "train_protein_patterns": list(args.train_protein_pattern or DEFAULT_PROTEIN_PATTERNS),
        "train_ligand_patterns": list(args.train_ligand_pattern or DEFAULT_LIGAND_PATTERNS),
        "test_raw_root": str(args.test_raw_root.resolve()),
        "test_ligand_root": str(args.test_ligand_root.resolve()),
        "test_ligand_pattern": args.test_ligand_pattern,
        "contact_cutoff_A": args.contact_cutoff,
        "minimum_sequence_length": args.minimum_sequence_length,
        "intended_train_split_exact_pdb_overlap_n": len(intended_exact),
        "exact_pdb_overlap_n": len(exact),
        "test_contact_entity_n": len(test),
        "test_extraction_failure_n": len(test_failures),
        "training_receptor_chain_n": len(chains),
        "effective_training_complex_n": len(effective_training_ids),
        "training_excluded_before_model_input_n": len(training_exclusions),
        "training_requires_readable_ligand": args.require_training_ligand,
        "training_unique_sequence_n": len(targets),
        "training_contact_chain_n": int(chains["is_contact"].sum()) if len(chains) else 0,
        "training_structure_failure_n": len(training_failures),
        "training_contact_annotation_failure_n": len(contact_failures),
        "additional_training_fastas": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path.resolve()),
            }
            for path in (args.additional_training_fasta or [])
        ],
        "additional_training_sequence_n": len(auxiliary_rows),
        "training_source_notes": args.training_source_note or [],
        "contact_to_all_receptor_audit_ready": not test_failures and not training_failures,
        "contact_to_training_contact_audit_ready": (
            not test_failures and not training_failures and not contact_failures
        ),
        "uses_docking_predictions": False,
        "uses_reference_ligand_only_for_contact_definition": True,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    if args.require_complete and (
        test_failures or training_failures or contact_failures
    ):
        raise RuntimeError(
            "Extraction was incomplete; inspect extraction_failures.csv. "
            "No sequence-novel claim is valid until all required rows are resolved."
        )


def run_mmseqs(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    query = output / "test_contact_entities.fasta"
    target = output / "training_receptor_sequences.fasta"
    result = output / "mmseqs_all_hits.tsv"
    temporary = output / "mmseqs_tmp"
    if not query.is_file() or not target.is_file():
        raise FileNotFoundError("Run --stage prepare before MMseqs2 search")
    executable = shutil.which(args.mmseqs_bin) or args.mmseqs_bin
    command = [
        executable, "easy-search", str(query), str(target), str(result), str(temporary),
        "--format-output", ",".join(MMSEQS_COLUMNS),
        "--threads", str(args.mmseqs_threads),
        "-s", str(args.mmseqs_sensitivity),
        "--max-seqs", str(args.mmseqs_max_seqs),
        "-e", str(args.mmseqs_evalue),
    ]
    (output / "mmseqs_command.txt").write_text(" ".join(command) + "\n")
    if shutil.which(args.mmseqs_bin) is None and not Path(args.mmseqs_bin).is_file():
        raise FileNotFoundError(
            f"Could not find {args.mmseqs_bin!r}; install MMseqs2. The exact command "
            f"has been recorded in {output / 'mmseqs_command.txt'}"
        )
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def percent_value(value: object) -> float:
    number = float(value)
    return 100.0 * number if 0.0 <= number <= 1.0 else number


def exposure_class(identity: float) -> str:
    if not np.isfinite(identity):
        return "sequence_novel_lt30"
    if identity >= 99.9:
        return "identity_ge99.9_not_full_exact"
    if identity >= 90.0:
        return "near_identical_90_99.9"
    if identity >= 50.0:
        return "close_homologue_50_90"
    if identity >= 30.0:
        return "remote_homologue_30_50"
    return "sequence_novel_lt30"


def choose_best_hits(
    queries: pd.DataFrame,
    hits: pd.DataFrame,
    targets: pd.DataFrame,
    target_scope: str,
    minimum_query_coverage: float,
    minimum_target_coverage: float,
    minimum_alignment_length: int,
) -> pd.DataFrame:
    if target_scope == "training_contact":
        contact_flags = targets["any_ligand_contact"]
        if not pd.api.types.is_bool_dtype(contact_flags):
            contact_flags = contact_flags.astype(str).str.strip().str.lower().isin(
                {"true", "1", "yes"}
            )
        allowed_targets = set(
            targets.loc[contact_flags, "target_id"].astype(str)
        )
        scoped = hits[hits["target"].isin(allowed_targets)].copy()
    elif target_scope == "all_receptor":
        scoped = hits.copy()
    else:  # pragma: no cover - internal misuse
        raise ValueError(target_scope)
    scoped = scoped[
        scoped["qcov_pct"].ge(minimum_query_coverage)
        & scoped["tcov_pct"].ge(minimum_target_coverage)
        & scoped["alnlen"].ge(minimum_alignment_length)
    ].copy()
    scoped = scoped.sort_values(
        ["query", "fident_pct", "bits", "qcov_pct", "tcov_pct", "target"],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    )
    best = scoped.drop_duplicates("query", keep="first")
    target_columns = targets.rename(columns={
        "target_id": "target",
        "sequence": "best_training_sequence",
        "pdb_ids": "best_training_pdb_ids",
        "chain_keys": "best_training_chain_keys",
        "any_ligand_contact": "best_target_has_contact_occurrence",
    })[[
        "target", "best_training_sequence", "best_training_pdb_ids", "best_training_chain_keys",
        "best_target_has_contact_occurrence", "occurrence_count",
    ]]
    best = best.merge(target_columns, on="target", how="left", validate="many_to_one")
    result = queries[[
        "query_id", "pdb_id", "entity_id", "entity_key", "sequence_length",
        "minimum_ligand_distance_A", "sequence",
    ]].rename(columns={"sequence": "query_sequence"})
    result = result.merge(best, left_on="query_id", right_on="query", how="left")
    result["exposure_class"] = result["fident_pct"].map(exposure_class)
    result["strict_full_sequence_match"] = (
        result["query_sequence"].notna()
        & result["best_training_sequence"].notna()
        & result["query_sequence"].eq(result["best_training_sequence"])
    )
    result.loc[
        result["strict_full_sequence_match"], "exposure_class"
    ] = "exact_full_sequence"
    result = result.drop(columns=["query_sequence", "best_training_sequence"])
    result.insert(0, "target_scope", target_scope)
    return result


def summarize(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    query_path = output / "test_contact_entities.csv"
    target_path = output / "training_unique_sequences.csv"
    hit_path = output / "mmseqs_all_hits.tsv"
    if not all(path.is_file() for path in (query_path, target_path, hit_path)):
        raise FileNotFoundError("Preparation tables or mmseqs_all_hits.tsv are missing")
    queries = pd.read_csv(query_path, dtype={"query_id": str, "pdb_id": str})
    targets = pd.read_csv(target_path, dtype={"target_id": str})
    if hit_path.stat().st_size:
        hits = pd.read_csv(hit_path, sep="\t", names=MMSEQS_COLUMNS, header=None)
    else:
        hits = pd.DataFrame(columns=MMSEQS_COLUMNS)
    for column in ("fident", "alnlen", "qcov", "tcov", "evalue", "bits"):
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits["fident_pct"] = hits["fident"].map(percent_value)
    hits["qcov_pct"] = hits["qcov"].map(percent_value)
    hits["tcov_pct"] = hits["tcov"].map(percent_value)
    hits.to_csv(output / "mmseqs_all_hits_normalized.csv", index=False)

    entity_tables: list[pd.DataFrame] = []
    complex_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    cohort = read_ids(args.cohort.resolve())
    for scope in ("all_receptor", "training_contact"):
        entity = choose_best_hits(
            queries,
            hits,
            targets,
            scope,
            args.minimum_query_coverage,
            args.minimum_target_coverage,
            args.minimum_alignment_length,
        )
        entity_tables.append(entity)
        best = entity.sort_values(
            ["pdb_id", "fident_pct", "bits"],
            ascending=[True, False, False],
            na_position="last",
            kind="stable",
        ).drop_duplicates("pdb_id", keep="first")
        prefix = "all_receptor" if scope == "all_receptor" else "training_contact"
        rename = {
            column: f"{prefix}_{column}"
            for column in best.columns
            if column not in {"pdb_id", "target_scope"}
        }
        best = best.rename(columns=rename).drop(columns="target_scope")
        complexes = pd.DataFrame({"pdb_id": cohort}).merge(best, on="pdb_id", how="left")
        class_column = f"{prefix}_exposure_class"
        identity_column = f"{prefix}_fident_pct"
        complexes[class_column] = complexes[class_column].fillna("unresolved")
        complex_tables.append(complexes)
        counts = complexes[class_column].value_counts(dropna=False)
        for category, count in counts.items():
            summary_rows.append({
                "target_scope": scope,
                "category": category,
                "count": int(count),
                "denominator": len(cohort),
                "percent": 100.0 * int(count) / len(cohort),
            })
        for threshold in (30.0, 50.0, 90.0, 99.9):
            exposed = pd.to_numeric(complexes[identity_column], errors="coerce").ge(threshold)
            summary_rows.append({
                "target_scope": scope,
                "category": f"identity_ge_{threshold:g}_bidirectional_coverage",
                "count": int(exposed.sum()),
                "denominator": len(cohort),
                "percent": 100.0 * float(exposed.mean()),
            })

    entities = pd.concat(entity_tables, ignore_index=True)
    complexes = complex_tables[0].merge(complex_tables[1], on="pdb_id", how="outer")
    entities.to_csv(output / "per_test_entity_best_hits.csv", index=False)
    complexes.to_csv(output / "per_complex_best_hits.csv", index=False)
    summary_table = pd.DataFrame(summary_rows)
    summary_table.to_csv(output / "overlap_summary.csv", index=False)

    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    unresolved = int(complexes["all_receptor_exposure_class"].eq("unresolved").sum())
    summary = {
        **provenance,
        "mmseqs_result": str(hit_path),
        "mmseqs_result_sha256": sha256_file(hit_path),
        "primary_definition": (
            "maximum sequence identity between any ligand-contacting test entity and "
            "any receptor chain presented during pretraining, subject to bidirectional coverage"
        ),
        "minimum_query_coverage_pct": args.minimum_query_coverage,
        "minimum_target_coverage_pct": args.minimum_target_coverage,
        "minimum_alignment_length": args.minimum_alignment_length,
        "unresolved_primary_complex_n": unresolved,
        "primary_audit_complete": unresolved == 0,
        "terminology": (
            "Similarity is reported as pretraining sequence exposure; only exact duplicate "
            "structures or sequences should be described as direct leakage."
        ),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nPRETRAINING EXPOSURE SUMMARY")
    print(summary_table.to_string(index=False))
    print(f"\nPRETRAINING_EXPOSURE_AUDIT_OK: {output}")
    if args.require_complete and unresolved:
        raise RuntimeError(f"Primary audit has {unresolved} unresolved test complexes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "search", "summarize", "all"), default="all")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--train-split", required=True, type=Path)
    parser.add_argument("--train-receptor-root", required=True, type=Path)
    parser.add_argument(
        "--train-protein-pattern", action="append",
        help=("Relative pattern below the training root; repeat for fallbacks. "
              "Tokens: {pdb_id}, {pdb_id_lower}."),
    )
    parser.add_argument(
        "--train-ligand-pattern", action="append",
        help="Relative ligand pattern below the training root; repeat for fallbacks.",
    )
    parser.add_argument("--test-raw-root", required=True, type=Path)
    parser.add_argument("--test-ligand-root", required=True, type=Path)
    parser.add_argument(
        "--test-ligand-pattern",
        default="{pdb_id}/{pdb_id}_ligand_deposited.sdf",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--additional-training-fasta",
        action="append",
        type=Path,
        help=("Additional receptor sequences seen during training (for example a "
              "self-distillation or structural augmentation source); repeat as needed."),
    )
    parser.add_argument(
        "--training-source-note",
        action="append",
        help="Free-text model provenance note; repeat as needed.",
    )
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    parser.add_argument("--minimum-sequence-length", type=int, default=20)
    parser.add_argument(
        "--require-training-ligand",
        action="store_true",
        help=("Exclude intended split entries without a readable ligand, matching "
              "PDBBind preprocessing before model input construction."),
    )
    parser.add_argument("--minimum-query-coverage", type=float, default=80.0)
    parser.add_argument("--minimum-target-coverage", type=float, default=80.0)
    parser.add_argument("--minimum-alignment-length", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--mmseqs-threads", type=int, default=16)
    parser.add_argument("--mmseqs-sensitivity", type=float, default=7.5)
    parser.add_argument("--mmseqs-max-seqs", type=int, default=10000)
    parser.add_argument("--mmseqs-evalue", type=float, default=10.0)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.contact_cutoff <= 0:
        raise ValueError("--contact-cutoff must be positive")
    if args.minimum_sequence_length < 1 or args.minimum_alignment_length < 1:
        raise ValueError("minimum sequence/alignment lengths must be positive")
    if args.workers < 1 or args.mmseqs_threads < 1:
        raise ValueError("worker/thread counts must be positive")
    for coverage in (args.minimum_query_coverage, args.minimum_target_coverage):
        if not 0 <= coverage <= 100:
            raise ValueError("coverage thresholds must be in [0, 100]")
    if args.stage in {"prepare", "all"}:
        required_files = {
            "Cohort": args.cohort,
            "Training split": args.train_split,
        }
        required_directories = {
            "Training receptor root": args.train_receptor_root,
            "Test raw-structure root": args.test_raw_root,
            "Test reference-ligand root": args.test_ligand_root,
        }
        for label, path in required_files.items():
            if not path.expanduser().is_file():
                raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
        for label, path in required_directories.items():
            if not path.expanduser().is_dir():
                raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    if args.checkpoint is not None and not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    for path in args.additional_training_fasta or []:
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"Additional training FASTA does not exist: {path}")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    if args.stage in {"prepare", "all"}:
        prepare(args)
    if args.stage in {"search", "all"}:
        run_mmseqs(args)
    if args.stage in {"summarize", "all"}:
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
