#!/usr/bin/env python3
"""Prepare an isolated ChemEM recovery for N-terminal PRO H2/H3 mapping.

The installed ChemEM environment is never modified.  A private copy of the
ChemEM Python package is placed under the recovery root, and only the PRO
hydrogen-name dictionary is extended.  Coordinates, ligand identity, maps,
and docking parameters are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_CASE_IDS = ("9KDF", "9Y7J")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_package(chemem_python: Path) -> Path:
    command = [
        str(chemem_python),
        "-c",
        "import pathlib, ChemEM; print(pathlib.Path(ChemEM.__file__).resolve().parent)",
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    package = Path(result.stdout.strip()).resolve()
    if not (package / "tools" / "openff_tools.py").is_file():
        raise FileNotFoundError(f"invalid ChemEM package directory: {package}")
    return package


def read_rows(
    paths: list[Path], case_ids: tuple[str, ...]
) -> tuple[list[str], dict[str, dict[str, str]]]:
    fields: list[str] = []
    indexed: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            for field in reader.fieldnames or []:
                if field not in fields:
                    fields.append(field)
        for row in rows:
            case_id = str(row.get("complex_name", "")).strip().upper()
            if not case_id:
                continue
            if case_id in indexed:
                raise ValueError(
                    f"duplicate case {case_id} across source tables; refusing an "
                    "ambiguous recovery input"
                )
            indexed[case_id] = row
    missing = [case_id for case_id in case_ids if case_id not in indexed]
    if missing:
        raise ValueError(f"recovery cases absent from source CSVs: {missing}")
    return fields, indexed


def rewrite_config(source: Path, target: Path, output: Path, densmap: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    rewritten: list[str] = []
    replaced = set()
    for line in source.read_text().splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key == "output":
            line = f"output = {output}"
            replaced.add("output")
        elif key == "densmap":
            line = f"densmap = {densmap}"
            replaced.add("densmap")
        rewritten.append(line)
    if replaced != {"output", "densmap"}:
        raise ValueError(f"config lacks output/densmap assignments: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(rewritten) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-cases", required=True, type=Path, action="append",
        help="ChemEM cases CSV; repeat when recovery cases span work sets",
    )
    parser.add_argument(
        "--case-id", action="append", dest="case_ids",
        help=(
            "case to include in the recovery table; repeat for multiple cases. "
            "Defaults to the original 9KDF/9Y7J pair when omitted"
        ),
    )
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--chemem-python", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_cases = [path.expanduser().resolve() for path in args.source_cases]
    case_ids = tuple(
        str(case_id).strip().upper()
        for case_id in (args.case_ids or DEFAULT_CASE_IDS)
    )
    if not case_ids or any(not case_id for case_id in case_ids):
        raise ValueError("at least one nonempty recovery case ID is required")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"duplicate recovery case IDs: {case_ids}")
    recovery_root = args.recovery_root.expanduser().resolve()
    chemem_python = args.chemem_python.expanduser().resolve()
    for required in (*source_cases, chemem_python):
        if not required.is_file():
            raise FileNotFoundError(required)

    fields, indexed = read_rows(source_cases, case_ids)
    installed_package = locate_package(chemem_python)
    compat_python = recovery_root / "compat_python"
    compat_package = compat_python / "ChemEM"
    compat_python.mkdir(parents=True, exist_ok=True)
    shutil.copytree(installed_package, compat_package, dirs_exist_ok=True)

    installed_pro_json = installed_package / "tools" / "amino_acids" / "PRO_hydrogens.json"
    compat_pro_json = compat_package / "tools" / "amino_acids" / "PRO_hydrogens.json"
    before = json.loads(installed_pro_json.read_text())
    after = dict(before)
    for atom_name in ("H2", "H3"):
        existing = after.get(atom_name)
        if existing not in (None, "N"):
            raise ValueError(
                f"refusing to replace unexpected PRO mapping {atom_name}={existing!r}"
            )
        after[atom_name] = "N"
    compat_pro_json.write_text(json.dumps(after, sort_keys=True) + "\n")

    recovery_rows: list[dict[str, str]] = []
    for recovery_index, case_id in enumerate(case_ids):
        row = dict(indexed[case_id])
        output = recovery_root / "runs" / case_id / "stage1"
        densmap = recovery_root / "inputs" / case_id / f"{case_id}_experimental.map"
        config = recovery_root / "configs" / f"{case_id}_stage1.conf"
        log = recovery_root / "logs" / f"{case_id}_stage1_opencl.log"
        rewrite_config(Path(row["stage1_config"]), config, output, densmap)
        row.update(
            {
                "work_index": str(recovery_index),
                "case_group": "pro_h2_compat_recovery",
                "stage1_config": str(config),
                "stage1_output": str(output),
                "case_log": str(log),
                "decompressed_map": str(densmap),
            }
        )
        recovery_rows.append(row)

    recovery_cases = recovery_root / "recovery_cases.csv"
    recovery_root.mkdir(parents=True, exist_ok=True)
    with recovery_cases.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(recovery_rows)
    (recovery_root / "recovery_ids.txt").write_text("".join(f"{x}\n" for x in case_ids))

    audit = {
        "case_ids": list(case_ids),
        "selection_rule": (
            "all explicitly listed authoritative cases with the same logged "
            "N-terminal PRO/H2 compatibility failure"
        ),
        "outcome_independent_selection": True,
        "native_ligand_coordinates_used": False,
        "coordinates_changed": False,
        "docking_parameters_changed": False,
        "installed_chemem_package": str(installed_package),
        "compat_chemem_package": str(compat_package),
        "installed_PRO_hydrogens_json_sha256": sha256(installed_pro_json),
        "compat_PRO_hydrogens_json_sha256": sha256(compat_pro_json),
        "mapping_before": before,
        "mapping_after": after,
        "source_cases": [str(path) for path in source_cases],
        "source_cases_sha256": {
            str(path): sha256(path) for path in source_cases
        },
        "recovery_cases": str(recovery_cases),
        "recovery_cases_sha256": sha256(recovery_cases),
    }
    audit_path = recovery_root / "pro_h2_compat_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"CHEMEM_PRO_H2_RECOVERY_PREPARED: {recovery_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
