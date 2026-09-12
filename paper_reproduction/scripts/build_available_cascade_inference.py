#!/usr/bin/env python3
"""Build a reindexed cascade-inference view from complete sharded branches.

The original frozen branch manifest remains unchanged. Branches without all
requested ligand poses in every condition, or without paired flexible-protein
outputs when flexibility was requested, are recorded in an audit and omitted
from the runnable view. Result directories are exposed through symbolic links;
large inference outputs are not copied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


POSE_RE = re.compile(r"^rank(?P<rank>\d+)_confidence[-+0-9.eE]+\.sdf$")
PROTEIN_RE = re.compile(r"^rank\d+_confidence[-+0-9.eE]+_protein\.pdb$")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "complex"


def count_outputs(directory: Path, validate_poses: bool) -> tuple[int, int, list[str]]:
    files = list(directory.iterdir())
    pose_paths = sorted(
        (
            path for path in files
            if path.is_file() and POSE_RE.match(path.name) is not None
        ),
        key=lambda path: int(POSE_RE.match(path.name).group("rank")),
    )
    proteins = sum(
        path.is_file()
        and path.stat().st_size > 0
        and PROTEIN_RE.match(path.name) is not None
        for path in files
    )
    invalid: list[str] = []
    if validate_poses:
        import numpy as np
        from rdkit import Chem

        for path in pose_paths:
            try:
                supplier = Chem.SDMolSupplier(
                    str(path), removeHs=False, sanitize=False
                )
                molecule = supplier[0] if supplier and len(supplier) else None
                if molecule is None or molecule.GetNumConformers() == 0:
                    raise ValueError("RDKit could not read a conformer")
                coordinates = np.asarray(
                    molecule.GetConformer().GetPositions(), dtype=float
                )
                if coordinates.size == 0 or not np.isfinite(coordinates).all():
                    raise ValueError("coordinates are empty or non-finite")
                if not any(atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()):
                    raise ValueError("pose contains no heavy atoms")
            except Exception as exc:
                invalid.append(f"{path.name}:{type(exc).__name__}:{exc}")
    return len(pose_paths), proteins, invalid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--conditions", default="baseline,guided")
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument("--expected-available", type=int, default=None)
    parser.add_argument("--expected-missing", type=int, default=None)
    parser.add_argument(
        "--validate-poses", action="store_true",
        help="Require every ranked SDF to be RDKit-readable with finite coordinates.",
    )
    args = parser.parse_args()

    if args.num_shards <= 0 or args.num_predictions <= 0:
        raise ValueError("num-shards and num-predictions must be positive")
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be a nonempty unique comma-separated list")

    manifest_path = args.full_manifest.expanduser().resolve()
    shard_root = args.shard_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")

    full = pd.read_csv(manifest_path, dtype=str)
    if "complex_name" not in full:
        raise ValueError("Full manifest lacks complex_name")
    if full["complex_name"].isna().any() or full["complex_name"].duplicated().any():
        raise ValueError("Full manifest complex_name values must be nonempty and unique")
    full_names = full["complex_name"].astype(str).tolist()
    full_lookup = {name: index for index, name in enumerate(full_names)}

    source_rows: dict[str, dict[str, Path]] = {}
    audit_rows: list[dict[str, object]] = []
    seen_names: set[str] = set()

    for shard_index in range(args.num_shards):
        shard_manifest_path = shard_root / f"shard{shard_index}" / "manifest.csv"
        shard_manifest = pd.read_csv(shard_manifest_path, dtype=str)
        for local_index, row in shard_manifest.iterrows():
            name = str(row["complex_name"])
            if name not in full_lookup:
                raise ValueError(f"Shard branch is absent from full manifest: {name}")
            if name in seen_names:
                raise ValueError(f"Branch occurs in multiple shards: {name}")
            seen_names.add(name)
            rigid_fallback = row.get("sidechain_policy", "") == "rigid_receptor_fallback"
            condition_paths: dict[str, Path] = {}
            complete = True
            reasons: list[str] = []
            record: dict[str, object] = {
                "full_row_index": full_lookup[name],
                "shard_index": shard_index,
                "shard_row_index": int(local_index),
                "complex_name": name,
                "parent_complex_name": row.get("parent_complex_name", ""),
                "global_rank": row.get("global_rank", ""),
                "sidechain_policy": row.get("sidechain_policy", ""),
            }
            for condition in conditions:
                parent = shard_root / f"shard{shard_index}" / "results" / condition
                matches = sorted(path for path in parent.glob(f"index{local_index}___*") if path.is_dir())
                pose_count = protein_count = 0
                invalid_poses: list[str] = []
                if len(matches) != 1:
                    complete = False
                    reasons.append(f"{condition}:result_directories={len(matches)}")
                else:
                    pose_count, protein_count, invalid_poses = count_outputs(
                        matches[0], args.validate_poses
                    )
                    condition_paths[condition] = matches[0]
                    if pose_count != args.num_predictions:
                        complete = False
                        reasons.append(f"{condition}:poses={pose_count}")
                    if not rigid_fallback and protein_count != args.num_predictions:
                        complete = False
                        reasons.append(f"{condition}:proteins={protein_count}")
                    if invalid_poses:
                        complete = False
                        reasons.append(
                            f"{condition}:invalid_poses={len(invalid_poses)}"
                        )
                record[f"{condition}_pose_count"] = pose_count
                record[f"{condition}_protein_count"] = protein_count
                record[f"{condition}_invalid_pose_count"] = len(invalid_poses)
                record[f"{condition}_invalid_pose_details"] = "|".join(invalid_poses)
                record[f"{condition}_result_dir"] = (
                    str(matches[0].resolve()) if len(matches) == 1 else ""
                )
            record["available"] = complete
            record["reason"] = ";".join(reasons)
            audit_rows.append(record)
            if complete:
                source_rows[name] = condition_paths

    missing_from_shards = sorted(set(full_names) - seen_names)
    if missing_from_shards:
        raise RuntimeError(f"Full-manifest branches absent from shards: {missing_from_shards[:20]}")

    available_indices = [index for index, name in enumerate(full_names) if name in source_rows]
    available = full.iloc[available_indices].copy().reset_index(drop=True)
    available.insert(0, "frozen_full_row_index", available_indices)
    missing_count = len(full) - len(available)
    if args.expected_available is not None and len(available) != args.expected_available:
        raise RuntimeError(
            f"Expected {args.expected_available} available branches, found {len(available)}"
        )
    if args.expected_missing is not None and missing_count != args.expected_missing:
        raise RuntimeError(
            f"Expected {args.expected_missing} missing branches, found {missing_count}"
        )

    output.mkdir(parents=True, exist_ok=True)
    available.to_csv(output / "inference_blind_available.csv", index=False)
    audit = pd.DataFrame(audit_rows).sort_values("full_row_index")
    audit.to_csv(output / "branch_availability_audit.csv", index=False)

    for available_index, row in available.iterrows():
        name = str(row["complex_name"])
        for condition in conditions:
            target_parent = output / condition
            target_parent.mkdir(parents=True, exist_ok=True)
            target = target_parent / f"index{available_index}___{safe_name(name)}"
            target.symlink_to(source_rows[name][condition].resolve(), target_is_directory=True)

    parent_column = "parent_complex_name" if "parent_complex_name" in full else None
    summary = {
        "frozen_full_manifest": str(manifest_path),
        "frozen_branch_count": len(full),
        "frozen_parent_count": (
            int(full[parent_column].nunique()) if parent_column is not None else None
        ),
        "available_branch_count": len(available),
        "missing_branch_count": missing_count,
        "conditions": conditions,
        "num_predictions_per_condition": args.num_predictions,
        "semantic_pose_validation": bool(args.validate_poses),
        "missing_branches": audit.loc[~audit["available"], "complex_name"].tolist(),
        "cohort_policy": (
            "Frozen parents retained. Missing branches are not replaced; downstream "
            "complete-Top-K endpoints count an incomplete parent as failure."
        ),
        "mode": "reindexed symbolic-link view",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_AVAILABLE_VIEW_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
