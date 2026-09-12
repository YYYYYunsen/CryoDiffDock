#!/usr/bin/env python3
"""Audit technical availability for a fixed CryoDiffDock-Local cohort.

The audit deliberately reads no RMSD values.  It follows each requested case
through the frozen Local pipeline: source manifest, prepared manifest, the two
40-pose inference arms, NCC scoring, and frozen V1 selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id_upper", "pdb_id", "complex", "name", "id")


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        duplicates = pd.Series(values)[pd.Series(values).duplicated(False)].unique()
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:20].tolist()}")
    return values


def read_table(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        table = pd.read_csv(path, dtype=str)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    id_column = next((column for column in ID_COLUMNS if column in table.columns), None)
    if id_column is None:
        return None
    table = table.copy()
    table["_audit_id"] = table[id_column].fillna("").map(normalize_id)
    return table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_ranked_poses(directory: Path) -> int:
    confidence_poses = list(directory.glob("rank*_confidence*.sdf"))
    if confidence_poses:
        return len(confidence_poses)
    return int((directory / "rank1.sdf").is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--test-split", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-poses-per-arm", type=int, default=40)
    args = parser.parse_args()

    cohort_path = args.cohort.expanduser().resolve()
    test_root = args.test_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.expected_poses_per_arm <= 0:
        raise ValueError("--expected-poses-per-arm must be positive")

    requested = read_ids(cohort_path)
    split_ids: set[str] | None = None
    split_path: Path | None = None
    if args.test_split is not None:
        split_path = args.test_split.expanduser().resolve()
        split_ids = set(read_ids(split_path))

    source_root = test_root / "01_source"
    prepared_root = test_root / "02_prepared"
    inference_root = test_root / "03_inference"
    ncc_root = test_root / "04_ncc"
    s0_root = test_root / "06_s0_frozen"
    metrics_root = test_root / "11_metrics_reveal"

    source_tables: list[tuple[Path, pd.DataFrame]] = []
    for path in (
        source_root / "test_source.csv",
        source_root / "test_source_full_polymer.csv",
    ):
        table = read_table(path)
        if table is not None:
            source_tables.append((path, table))

    prepared_path = prepared_root / "inference.csv"
    prepared = read_table(prepared_path)
    preparation_report = read_table(prepared_root / "preparation_report.csv")
    ncc_path = ncc_root / "pose_scores.csv"
    ncc = read_table(ncc_path)

    v1_tables: list[tuple[Path, pd.DataFrame]] = []
    for path in (
        s0_root / "best_selected_candidates.csv",
        metrics_root / "frozen_s0" / "per_complex.csv",
    ):
        table = read_table(path)
        if table is not None:
            v1_tables.append((path, table))

    diagnostic_tables: list[tuple[Path, pd.DataFrame]] = []
    for path in sorted(test_root.rglob("*.csv")):
        lowered = path.name.lower()
        if "fail" not in lowered and "report" not in lowered:
            continue
        table = read_table(path)
        if table is not None:
            diagnostic_tables.append((path, table))

    rows: list[dict[str, object]] = []
    expected_ncc = 2 * args.expected_poses_per_arm
    for complex_name in requested:
        source_present = any(
            table["_audit_id"].eq(complex_name).any()
            for _path, table in source_tables
        )

        prepared_index: int | None = None
        if prepared is not None:
            hits = prepared[prepared["_audit_id"].eq(complex_name)]
            if not hits.empty:
                prepared_index = int(hits.index[0])

        preparation_status = ""
        preparation_error = ""
        if preparation_report is not None:
            hits = preparation_report[
                preparation_report["_audit_id"].eq(complex_name)
            ]
            if not hits.empty:
                last = hits.iloc[-1]
                preparation_status = str(last.get("status", "")).strip()
                preparation_error = str(last.get("error", "")).strip()

        arm_counts: dict[str, int] = {}
        for arm in ("baseline", "guided"):
            directories = sorted(
                path
                for path in (inference_root / arm).glob(f"index*___{complex_name}*")
                if path.is_dir()
            )
            arm_counts[f"{arm}_result_directories"] = len(directories)
            arm_counts[f"{arm}_pose_count"] = sum(
                count_ranked_poses(directory) for directory in directories
            )
            arm_counts[f"{arm}_protein_count"] = sum(
                len(list(directory.glob("rank*_protein*.pdb")))
                for directory in directories
            )

        ncc_pose_rows = 0 if ncc is None else int(ncc["_audit_id"].eq(complex_name).sum())
        v1_available = any(
            table["_audit_id"].eq(complex_name).any()
            for _path, table in v1_tables
        )

        messages: list[str] = []
        for path, table in diagnostic_tables:
            hits = table[table["_audit_id"].eq(complex_name)]
            for _index, hit in hits.iterrows():
                status = str(hit.get("status", "")).strip()
                error = str(hit.get("error", "")).strip()
                if status or error:
                    messages.append(
                        f"{path.relative_to(test_root)}: status={status}; error={error}"
                    )

        in_split = None if split_ids is None else complex_name in split_ids
        baseline_complete = (
            arm_counts["baseline_pose_count"] == args.expected_poses_per_arm
        )
        guided_complete = arm_counts["guided_pose_count"] == args.expected_poses_per_arm
        if v1_available:
            inferred_stage = "V1_COMPLETE"
        elif prepared_index is None:
            if in_split is False:
                inferred_stage = "ABSENT_FROM_ORIGINAL_TEST_SPLIT"
            elif source_present:
                inferred_stage = "PREPARATION_MISSING_OR_FAILED"
            else:
                inferred_stage = "SOURCE_MISSING_OR_FAILED"
        elif not baseline_complete or not guided_complete:
            inferred_stage = "INFERENCE_INCOMPLETE"
        elif ncc_pose_rows != expected_ncc:
            inferred_stage = "NCC_INCOMPLETE"
        else:
            inferred_stage = "V1_SEARCH_OR_SELECTION_INCOMPLETE"

        rows.append({
            "complex_name": complex_name,
            "in_original_test_split": in_split,
            "in_source_manifest": source_present,
            "in_prepared_manifest": prepared_index is not None,
            "prepared_row_index": prepared_index,
            "preparation_status": preparation_status,
            "preparation_error": preparation_error,
            **arm_counts,
            "ncc_pose_rows": ncc_pose_rows,
            "v1_selected_available": v1_available,
            "inferred_failure_stage": inferred_stage,
            "historical_errors": " | ".join(dict.fromkeys(messages)),
        })

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    counts = result["inferred_failure_stage"].value_counts().rename_axis("stage")

    summary = {
        "cohort": str(cohort_path),
        "requested_n": len(requested),
        "test_root": str(test_root),
        "test_split": str(split_path) if split_path is not None else None,
        "expected_poses_per_arm": args.expected_poses_per_arm,
        "expected_ncc_pose_rows": expected_ncc,
        "stage_counts": {str(key): int(value) for key, value in counts.items()},
        "uses_rmsd_or_docking_outcomes": False,
        "input_sha256": {
            "cohort": sha256(cohort_path),
            "prepared_manifest": sha256(prepared_path) if prepared_path.is_file() else None,
            "ncc_scores": sha256(ncc_path) if ncc_path.is_file() else None,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("LOCAL COMPLETION GAP AUDIT")
    print(counts.to_string())
    print()
    print(result.to_string(index=False))
    print()
    print(f"AUDIT OUTPUT: {output_path}")
    print(f"SUMMARY: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
