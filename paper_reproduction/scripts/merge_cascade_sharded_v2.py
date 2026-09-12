#!/usr/bin/env python3
"""Merge sharded Local-V2 cascade Top-1 outputs into full branch row order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


CONFIG_KEYS = (
    "method",
    "density_sigmas_A",
    "density_weight",
    "basin_weight",
    "anchor_penalty",
    "coarse_anchor_penalty",
    "coarse_penalty",
    "clash_penalty",
    "search_depth",
    "beam_width",
    "beam_width_per_parent",
    "coarse_beam_fraction",
    "parent_count",
    "top_score_parents",
    "parent_score_column",
    "parent_diversity_rmsd_A",
    "translation_steps_A",
    "rotation_steps_degrees",
    "torsion_steps_degrees",
    "maximum_torsion_bonds",
    "top_torsion_gradients",
    "minimum_step_ncc_gain",
    "coarse_minimum_ncc_gain",
    "maximum_centroid_shift_A",
    "minimum_in_bounds_fraction",
    "clash_tolerance_count",
    "clash_tolerance_fraction",
    "severe_overlap_A",
    "mask_radius_A",
    "target_floor",
    "minimum_mask_voxels",
    "finalist_count",
    "basin_translation_step_A",
    "basin_rotation_step_degrees",
    "basin_support_tolerance",
    "coarse_orientation_library",
    "combined_rigid_moves",
    "basin_robust_selection",
    "uses_ground_truth",
    "materialization_mode",
)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "complex"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("num-shards must be positive")

    manifest_path = args.full_manifest.expanduser().resolve()
    shard_root = args.shard_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    full = pd.read_csv(manifest_path, dtype={"complex_name": str})
    if "complex_name" not in full.columns:
        raise ValueError("Full manifest lacks complex_name")
    if full["complex_name"].isna().any() or full["complex_name"].duplicated().any():
        raise ValueError("Full manifest complex_name values must be nonempty and unique")
    full_lookup = {
        str(name): index for index, name in enumerate(full["complex_name"].astype(str))
    }

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to merge into nonempty directory: {output_root}")
    selected_root = output_root / "v2_selected"
    selected_root.mkdir(parents=True, exist_ok=True)

    reference_config: dict | None = None
    audit_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    linked_names: set[str] = set()
    shard_summaries: list[dict] = []

    for shard_index in range(args.num_shards):
        search_root = shard_root / f"shard{shard_index}" / "search"
        shard_manifest_path = search_root / "evaluation_manifest.csv"
        summary_path = search_root / "summary.json"
        audit_path = search_root / "per_complex_audit.csv"
        for required in (shard_manifest_path, summary_path, audit_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        shard_manifest = pd.read_csv(
            shard_manifest_path, dtype={"complex_name": str}
        )
        if shard_manifest["complex_name"].duplicated().any():
            raise ValueError(f"Shard {shard_index} contains duplicate complex names")
        unknown = sorted(set(shard_manifest["complex_name"]) - set(full_lookup))
        if unknown:
            raise ValueError(f"Shard {shard_index} has names outside full manifest: {unknown}")

        summary = load_json(summary_path)
        config = {key: summary.get(key) for key in CONFIG_KEYS}
        if summary.get("materialization_mode") != "top1":
            raise ValueError(f"Shard {shard_index} was not run in top1 mode")
        if summary.get("uses_ground_truth") is not False:
            raise ValueError(f"Shard {shard_index} is not marked ground-truth-free")
        if reference_config is None:
            reference_config = config
        elif config != reference_config:
            raise ValueError(f"Frozen V2 configuration differs in shard {shard_index}")
        shard_summaries.append(summary)

        audit = pd.read_csv(audit_path, dtype={"complex_name": str})
        if len(audit) != len(shard_manifest):
            raise ValueError(
                f"Shard {shard_index} audit has {len(audit)} rows for "
                f"{len(shard_manifest)} manifest rows"
            )
        if set(audit["complex_name"]) != set(shard_manifest["complex_name"]):
            raise ValueError(f"Shard {shard_index} audit/manifest names differ")
        audit["shard_index"] = shard_index
        audit["shard_row_index"] = pd.to_numeric(
            audit["row_index"], errors="raise"
        ).astype(int)
        audit["row_index"] = audit["complex_name"].map(full_lookup).astype(int)
        source_indices = pd.to_numeric(
            audit["source_row_index"], errors="raise"
        ).astype(int)
        if not source_indices.equals(audit["row_index"]):
            raise ValueError(
                f"Shard {shard_index} did not preserve full-manifest source indices"
            )
        audit_frames.append(audit)

        failures_path = search_root / "failures.csv"
        if failures_path.is_file() and failures_path.stat().st_size:
            try:
                failures = pd.read_csv(failures_path)
            except pd.errors.EmptyDataError:
                failures = pd.DataFrame()
            if len(failures):
                failures["shard_index"] = shard_index
                failure_frames.append(failures)

        for local_index, row in shard_manifest.reset_index(drop=True).iterrows():
            name = str(row["complex_name"])
            if name in linked_names:
                raise ValueError(f"Complex appears in more than one shard: {name}")
            source_matches = sorted(
                (search_root / "v2_selected").glob(f"index{local_index}___*")
            )
            if len(source_matches) != 1:
                raise FileNotFoundError(
                    f"Expected one V2 Top-1 directory for shard {shard_index} "
                    f"row {local_index}, found {len(source_matches)}"
                )
            full_index = full_lookup[name]
            target = selected_root / f"index{full_index}___{safe_name(name)}"
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            target.symlink_to(source_matches[0].resolve(), target_is_directory=True)
            linked_names.add(name)

    missing = sorted(set(full_lookup) - linked_names)
    if missing:
        raise RuntimeError(f"V2 merge is missing full-manifest branches: {missing[:20]}")

    audit = pd.concat(audit_frames, ignore_index=True).sort_values("row_index")
    if audit["row_index"].tolist() != list(range(len(full))):
        raise RuntimeError("Merged V2 audit is not aligned to full branch row order")
    failures = (
        pd.concat(failure_frames, ignore_index=True)
        if failure_frames
        else pd.DataFrame(
            columns=["row_index", "source_row_index", "complex_name", "error", "shard_index"]
        )
    )

    full.to_csv(output_root / "evaluation_manifest.csv", index=False)
    audit.to_csv(output_root / "per_complex_audit.csv", index=False)
    failures.to_csv(output_root / "failures.csv", index=False)

    timing_keys = sorted(
        set().union(*(summary.get("timing", {}).keys() for summary in shard_summaries))
    )
    timing_sum = {
        key: float(sum(float(summary.get("timing", {}).get(key, 0.0)) for summary in shard_summaries))
        for key in timing_keys
    }
    summary = {
        "method": "CryoDiffDock-Local V2",
        "full_manifest": str(manifest_path),
        "num_branches": len(full),
        "num_shards": args.num_shards,
        "num_failures": len(failures),
        "linked_v2_selected_directories": len(linked_names),
        "materialization_mode": "top1",
        "uses_ground_truth": False,
        "frozen_config": reference_config,
        "sum_shard_runtime_seconds": float(
            sum(float(item["runtime_seconds"]) for item in shard_summaries)
        ),
        "maximum_shard_runtime_seconds": float(
            max(float(item["runtime_seconds"]) for item in shard_summaries)
        ),
        "summed_component_timing_seconds": timing_sum,
        "unmerged_large_tables": [
            str(shard_root / f"shard{index}" / "search" / "candidates.csv")
            for index in range(args.num_shards)
        ],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_V2_SHARDS_MERGED: {selected_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
