#!/usr/bin/env python3
"""Prepare an audited private Rosetta generic-potential compatibility copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


OLD_TOKEN = "a.mol.atms"
NEW_TOKEN = "a.mol.atoms['name']"
AUDIT_NAME = "figure2_am1bcc_compat_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_files(directory: Path) -> dict[str, Path]:
    return {
        name: directory / name
        for name in ("mol2genparams.py", "Molecule.py", "AM1.py")
    }


def validate_files(directory: Path) -> dict[str, Path]:
    files = required_files(directory)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise SystemExit("ERROR: missing required Rosetta files: " + ", ".join(missing))
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    source_files = validate_files(source)

    if output.exists():
        audit_path = output / AUDIT_NAME
        output_files = validate_files(output)
        molecule_text = output_files["Molecule.py"].read_text()
        if OLD_TOKEN not in molecule_text and NEW_TOKEN in molecule_text and audit_path.is_file():
            print(f"EMERALD_ROSETTA_AM1BCC_COMPAT_ALREADY_PREPARED: {output}")
            print(audit_path.read_text(), end="")
            return 0
        raise SystemExit(
            f"ERROR: output directory exists but is not a validated compatibility copy: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=output.parent))
    try:
        shutil.copytree(
            source,
            staging,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        staged_files = validate_files(staging)
        molecule_path = staged_files["Molecule.py"]
        original_text = molecule_path.read_text()
        replacement_count = original_text.count(OLD_TOKEN)
        if replacement_count == 0:
            raise SystemExit(
                f"ERROR: expected token {OLD_TOKEN!r} was not found in {source_files['Molecule.py']}"
            )
        patched_text = original_text.replace(OLD_TOKEN, NEW_TOKEN)
        molecule_path.write_text(patched_text)

        audit = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(source),
            "output_dir": str(output),
            "patch": {"from": OLD_TOKEN, "to": NEW_TOKEN, "replacement_count": replacement_count},
            "source_sha256": {name: sha256(path) for name, path in source_files.items()},
            "patched_sha256": {name: sha256(path) for name, path in staged_files.items()},
        }
        (staging / AUDIT_NAME).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"EMERALD_ROSETTA_AM1BCC_COMPAT_PREPARED: {output}")
    print((output / AUDIT_NAME).read_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
