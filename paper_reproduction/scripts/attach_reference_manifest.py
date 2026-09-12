#!/usr/bin/env python3
"""Attach reveal-only reference columns to a row-compatible blind manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--columns", default="reference_ligand")
    args = parser.parse_args(argv)
    blind_path = args.blind.expanduser().resolve()
    reference_path = args.reference.expanduser().resolve()
    output = args.output.expanduser().resolve()
    blind = pd.read_csv(blind_path, dtype={"complex_name": str})
    reference = pd.read_csv(reference_path, dtype={"complex_name": str})
    columns = [value.strip() for value in args.columns.split(",") if value.strip()]
    required = {"complex_name", *columns}
    if "complex_name" not in blind or not required <= set(reference.columns):
        raise ValueError(
            f"Blind requires complex_name and reference requires {sorted(required)}"
        )
    for label, table in (("blind", blind), ("reference", reference)):
        table["complex_name"] = table["complex_name"].astype(str).str.upper()
        if table["complex_name"].duplicated().any():
            raise ValueError(f"{label} manifest contains duplicate complex_name values")
    merged = blind.merge(
        reference[["complex_name", *columns]],
        on="complex_name", how="left", sort=False, validate="one_to_one",
    )
    missing = merged[columns].isna().any(axis=1)
    if missing.any():
        names = merged.loc[missing, "complex_name"].tolist()
        raise ValueError(f"Reference join failed for: {names[:10]}")
    if merged["complex_name"].tolist() != blind["complex_name"].tolist():
        raise AssertionError("Reference join changed blind-manifest row order")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    print(f"REFERENCE_MANIFEST_OK: {output} ({len(merged)} complexes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
