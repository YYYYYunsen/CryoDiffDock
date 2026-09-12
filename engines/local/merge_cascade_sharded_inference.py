#!/usr/bin/env python3
"""Merge row-indexed inference shards into one full-manifest result root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "complex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--conditions", default="baseline,guided")
    args = parser.parse_args()

    full = pd.read_csv(args.full_manifest.expanduser().resolve(), dtype=str)
    if full["complex_name"].duplicated().any():
        raise ValueError("Full manifest complex_name values must be unique")
    full_lookup = {name: index for index, name in enumerate(full["complex_name"])}
    shard_root = args.shard_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    conditions = [item.strip() for item in args.conditions.split(",") if item.strip()]
    linked = {condition: 0 for condition in conditions}

    for shard_index in range(args.num_shards):
        shard_manifest = shard_root / f"shard{shard_index}" / "manifest.csv"
        shard = pd.read_csv(shard_manifest, dtype=str)
        for local_index, row in shard.iterrows():
            name = str(row["complex_name"])
            full_index = full_lookup[name]
            for condition in conditions:
                source_parent = shard_root / f"shard{shard_index}" / "results" / condition
                matches = sorted(source_parent.glob(f"index{local_index}___*"))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one {condition} result for shard {shard_index} "
                        f"row {local_index}, found {len(matches)}"
                    )
                target_parent = output_root / condition
                target_parent.mkdir(parents=True, exist_ok=True)
                target = target_parent / f"index{full_index}___{safe_name(name)}"
                if target.exists() or target.is_symlink():
                    raise FileExistsError(target)
                target.symlink_to(matches[0].resolve(), target_is_directory=True)
                linked[condition] += 1

    summary = {
        "full_manifest": str(args.full_manifest.expanduser().resolve()),
        "num_rows": len(full),
        "num_shards": args.num_shards,
        "linked_result_directories": linked,
        "mode": "symbolic_links",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_SHARDS_MERGED: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
