#!/usr/bin/env python3
"""Add complete deposited polymer structure paths to an existing manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def find_structure(raw_root: Path, complex_name: str) -> Optional[Path]:
    pdb_id = str(complex_name).strip().split("_")[0].upper()
    directories = (raw_root / pdb_id, raw_root / pdb_id.lower())
    names = (
        f"{pdb_id}.cif", f"{pdb_id.lower()}.cif",
        f"{pdb_id}.mmcif", f"{pdb_id.lower()}.mmcif", "model.cif",
    )
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    matches = sorted(
        path
        for directory in directories if directory.is_dir()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".cif", ".mmcif"}
    )
    return max(matches, key=lambda path: path.stat().st_size).resolve() if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--column", default="mask_structure")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve()
    table = pd.read_csv(manifest)
    if "complex_name" not in table.columns:
        raise ValueError("Manifest is missing complex_name")

    paths = []
    failures = []
    for row_index, row in table.iterrows():
        path = find_structure(raw_root, row["complex_name"])
        paths.append(str(path) if path else "")
        if path is None:
            failures.append({
                "row_index": row_index,
                "complex_name": row["complex_name"],
                "error": "complete PDB/mmCIF not found",
            })
    table[args.column] = paths
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    pd.DataFrame(
        failures, columns=("row_index", "complex_name", "error")
    ).to_csv(output.with_suffix(".failures.csv"), index=False)
    print(f"FULL_POLYMER_MANIFEST: {len(table) - len(failures)} / {len(table)} -> {output}")
    print(f"Failures: {len(failures)} -> {output.with_suffix('.failures.csv')}")
    if failures and args.fail_on_error:
        raise RuntimeError(f"Missing complete structures for {len(failures)} cases")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
