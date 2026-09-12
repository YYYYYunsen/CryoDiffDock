#!/usr/bin/env python3
"""Subset a CSV/TSV by an ordered text ID list."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ID_COLUMNS = ("complex_name", "pdb_id", "complex_id", "id")


def delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ids = [line.split("#", 1)[0].strip().upper() for line in args.ids.read_text().splitlines()]
    ids = [value for value in ids if value]
    if len(ids) != len(set(ids)):
        raise ValueError("ID list contains duplicates")
    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter(args.input))
        if reader.fieldnames is None:
            raise ValueError("Input table has no header")
        id_column = next((column for column in ID_COLUMNS if column in reader.fieldnames), None)
        if id_column is None:
            raise ValueError(f"No ID column found; columns={reader.fieldnames}")
        indexed = {}
        for row in reader:
            key = str(row.get(id_column, "")).strip().upper()
            if key in indexed:
                raise ValueError(f"Duplicate input ID: {key}")
            indexed[key] = row
    missing = [value for value in ids if value not in indexed]
    if missing:
        raise ValueError(f"IDs absent from input table: {missing}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, delimiter=delimiter(output))
        writer.writeheader()
        writer.writerows(indexed[value] for value in ids)
    print(f"rows: {len(ids)}")
    print(f"SUBSET_TABLE_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
