#!/usr/bin/env python3
"""Audit and assemble parent-sharded validation cascade outputs without RMSD."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


TABLES = (
    "global/global_top20_scores.csv",
    "00_source/cascade_source_blind.csv",
    "00_source/cascade_source_reveal.csv",
    "01_prepared/inference_blind.csv",
    "01_prepared/evaluation_manifest.csv",
    "02_inference/available/inference_blind_available.csv",
    "02_inference/available/evaluation_manifest_available.csv",
    "03_ncc/pose_scores.csv",
)
BRANCH_TABLES = TABLES[1:7]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0].strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate cohort IDs: {path}")
    return values


def parent_column(table: pd.DataFrame) -> str:
    for name in ("parent_complex_name", "complex", "pdb_id", "complex_name", "complex_key"):
        if name in table.columns:
            return name
    raise ValueError(f"Cannot identify parent column from {table.columns.tolist()}")


def normalized_parent(series: pd.Series, column: str) -> pd.Series:
    values = series.astype(str).str.strip().str.upper()
    if column == "complex_name":
        values = values.str.replace(r"__GLOBAL_RANK\d+$", "", regex=True)
    return values


def branch_name_map(selected_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    if not selected_root.is_dir():
        raise FileNotFoundError(selected_root)
    for directory in selected_root.iterdir():
        if not directory.is_dir() or "___" not in directory.name:
            continue
        name = directory.name.split("___", 1)[1]
        if name in output:
            raise ValueError(f"Duplicate V2 branch directory: {name}")
        ranking = directory / "ranking.csv"
        poses = [
            path for path in directory.glob("rank1*.sdf")
            if re.fullmatch(r"rank1_confidence[-+0-9.eE]+\.sdf", path.name)
        ]
        if not ranking.is_file() or len(poses) != 1:
            raise ValueError(f"Invalid V2 selected directory: {directory}")
        ranked = pd.read_csv(ranking)
        if len(ranked) != 1 or str(ranked.iloc[0].get("complex_name", "")) != name:
            raise ValueError(f"V2 ranking identity mismatch: {directory}")
        output[name] = directory.resolve()
    return output


def canonical_rows(table: pd.DataFrame, cohort: list[str], expected_per_parent: int, label: str) -> pd.DataFrame:
    key = parent_column(table)
    work = table.copy()
    work["_parent"] = normalized_parent(work[key], key)
    unknown = sorted(set(work["_parent"]) - set(cohort))
    if unknown:
        raise ValueError(f"{label} contains parents outside the cohort: {unknown[:10]}")
    counts = work.groupby("_parent").size().reindex(cohort, fill_value=0)
    if not counts.eq(expected_per_parent).all():
        raise ValueError(f"{label} count mismatch: {counts[counts.ne(expected_per_parent)].to_dict()}")
    work["_parent_order"] = work["_parent"].map({name: index for index, name in enumerate(cohort)})
    order = ["_parent_order"]
    if "global_rank" in work.columns:
        work["_global_rank_order"] = pd.to_numeric(work["global_rank"], errors="raise")
        order.append("_global_rank_order")
    work = work.sort_values(order, kind="stable")
    return work.drop(columns=[name for name in ("_parent", "_parent_order", "_global_rank_order") if name in work])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument("--expected-parents", type=int, default=187)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    cohort_path = args.cohort.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    cohort = sorted(read_ids(cohort_path))
    if len(cohort) != args.expected_parents or args.num_shards < 1:
        raise ValueError(f"Expected {args.expected_parents} parents, found {len(cohort)}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")

    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    selected_by_name: dict[str, Path] = {}
    shard_records: list[dict[str, object]] = []
    reference_code_hashes: dict[str, str] | None = None
    seen: set[str] = set()
    migration_path = run_root / "migrated_to_h100.tsv"
    migrated_indices: set[int] = set()
    if migration_path.is_file():
        migration = pd.read_csv(migration_path, sep="\t")
        if "shard_index" not in migration:
            raise ValueError(f"Invalid migration provenance: {migration_path}")
        migrated_indices = set(pd.to_numeric(migration.shard_index, errors="raise").astype(int))

    for index in range(args.num_shards):
        shard = run_root / f"shard{index:02d}of{args.num_shards:02d}"
        success_path = shard / "AUTO_SUCCESS.json"
        lock_path = shard / "protocol_lock.json"
        for required in (success_path, lock_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        success = json.loads(success_path.read_text())
        lock = json.loads(lock_path.read_text())
        expected_ids = cohort[index::args.num_shards]
        observed_ids = [str(value).upper() for value in success.get("parent_ids", [])]
        if observed_ids != expected_ids:
            raise ValueError(f"Shard {index} parent assignment differs: {observed_ids} != {expected_ids}")
        if seen & set(observed_ids):
            raise ValueError(f"Parents repeated across shards: {sorted(seen & set(observed_ids))}")
        seen.update(observed_ids)
        expected_branches = 5 * len(expected_ids)
        required_success = {
            "mode": "full",
            "status": "complete_through_v2",
            "parents": len(expected_ids),
            "requested_branches": expected_branches,
            "available_branches": expected_branches,
            "selected_v2_branches": expected_branches,
            "rmsd_evaluated": False,
            "job_shards": args.num_shards,
            "job_shard_index": index,
            "gpus_per_job": 1,
        }
        mismatches = {key: (success.get(key), value) for key, value in required_success.items() if success.get(key) != value}
        if mismatches:
            raise ValueError(f"Shard {index} AUTO_SUCCESS mismatch: {mismatches}")
        code_hashes = lock.get("code_hashes")
        if not isinstance(code_hashes, dict) or not code_hashes:
            raise ValueError(f"Shard {index} lacks code hashes")
        if reference_code_hashes is None:
            reference_code_hashes = code_hashes
        elif code_hashes != reference_code_hashes:
            raise ValueError(f"Frozen code hashes differ in shard {index}")

        for relative in TABLES:
            path = shard / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            frames[relative].append(pd.read_csv(path))
        selected = branch_name_map(shard / "04_v2/merged/v2_selected")
        overlap = set(selected) & set(selected_by_name)
        if overlap:
            raise ValueError(f"V2 branches repeated across shards: {sorted(overlap)[:10]}")
        selected_by_name.update(selected)
        shard_records.append({
            "shard_index": index,
            "path": str(shard),
            "parents": len(expected_ids),
            "branches": expected_branches,
            "auto_success_sha256": sha256(success_path),
            "protocol_lock_sha256": sha256(lock_path),
            "execution_gpu_class": "h100" if index in migrated_indices else "A6000",
        })

    if seen != set(cohort):
        raise ValueError(f"Shard union differs from cohort; missing={sorted(set(cohort)-seen)}")

    assembled: dict[str, pd.DataFrame] = {}
    for relative, pieces in frames.items():
        columns = pieces[0].columns.tolist()
        for piece in pieces[1:]:
            if piece.columns.tolist() != columns:
                raise ValueError(f"Schema differs across shards: {relative}")
        table = pd.concat(pieces, ignore_index=True)
        expected = 20 if relative.startswith("global/") else 5
        if relative == "03_ncc/pose_scores.csv":
            # pose_scores contains every baseline/guided candidate, not one row per branch.
            key = parent_column(table)
            parents = set(normalized_parent(table[key], key))
            if parents != set(cohort):
                raise ValueError("NCC pose-score parent coverage differs from cohort")
            assembled[relative] = table
        else:
            assembled[relative] = canonical_rows(table, cohort, expected, relative)

    blind = assembled["01_prepared/inference_blind.csv"]
    if "complex_name" not in blind or blind["complex_name"].duplicated().any():
        raise ValueError("Assembled branch manifest has duplicate/missing complex_name")
    branch_names = blind["complex_name"].astype(str).tolist()
    if set(branch_names) != set(selected_by_name):
        raise ValueError(
            f"V2 selection coverage mismatch: missing={sorted(set(branch_names)-set(selected_by_name))[:10]}, "
            f"extra={sorted(set(selected_by_name)-set(branch_names))[:10]}"
        )
    if args.preflight_only:
        print(f"VALIDATION_SHARD_PREFLIGHT_OK: parents={len(cohort)}, branches={len(branch_names)}, global_rows={len(assembled['global/global_top20_scores.csv'])}")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir()
    (output / "inputs/cohort.txt").write_text("".join(f"{name}\n" for name in cohort))
    for relative, table in assembled.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
    selected_root = output / "04_v2/merged/v2_selected"
    selected_root.mkdir(parents=True)
    for row_index, name in enumerate(branch_names):
        target = selected_root / f"index{row_index}___{name}"
        os.symlink(selected_by_name[name], target, target_is_directory=True)

    provenance = {
        "status": "assembled_blind_no_rmsd_evaluated",
        "run_root": str(run_root),
        "cohort_source": str(cohort_path),
        "cohort_sha256": sha256(cohort_path),
        "cohort_n": len(cohort),
        "global_top20_rows": len(assembled["global/global_top20_scores.csv"]),
        "branch_rows": len(branch_names),
        "num_parent_shards": args.num_shards,
        "code_hashes": reference_code_hashes,
        "shards": shard_records,
        "source_table_hashes": {
            relative: [sha256(run_root / f"shard{index:02d}of{args.num_shards:02d}" / relative) for index in range(args.num_shards)]
            for relative in TABLES
        },
        "migration_record": (
            {"path": str(migration_path), "sha256": sha256(migration_path)}
            if migration_path.is_file() else None
        ),
        "assembler_sha256": sha256(Path(__file__)),
    }
    migration = run_root / "migrated_to_h100.tsv"
    if migration.is_file():
        provenance["migration_record"] = str(migration)
        provenance["migration_record_sha256"] = sha256(migration)
    (output / "assembly_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pd.DataFrame(shard_records).to_csv(output / "shard_audit.csv", index=False)
    (output / ".stage_assemble_complete").write_text("complete\n")
    print(f"VALIDATION_SHARDS_ASSEMBLED: {output}; parents={len(cohort)}, branches={len(branch_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
