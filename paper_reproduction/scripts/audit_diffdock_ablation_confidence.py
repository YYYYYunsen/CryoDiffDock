#!/usr/bin/env python3
"""Audit DiffDock-L ablation confidence-ranking coverage on a fixed cohort.

The script is read-only.  It compares the raw ablation pose count with the
confidence-ranked pose count for every cohort member and writes an explicit
missing/incomplete list that can be used for a resume run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--raw-poses", required=True, type=Path)
    parser.add_argument("--confidence-poses", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-poses", type=int, default=40,
        help="Maximum number of raw poses expected to be confidence-ranked.",
    )
    return parser.parse_args()


def cohort_ids(path: Path) -> list[str]:
    result = [
        line.split()[0].strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not result:
        raise ValueError(f"Empty cohort: {path}")
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate IDs in cohort: {path}")
    return result


def sdf_files(case_dir: Path) -> list[Path]:
    if not case_dir.is_dir():
        return []
    direct = sorted(case_dir.glob("*.sdf"))
    return direct if direct else sorted(case_dir.rglob("*.sdf"))


def main() -> int:
    args = parse_args()
    if args.max_poses <= 0:
        raise ValueError("--max-poses must be positive")
    for path in (args.cohort, args.raw_poses, args.confidence_poses):
        if not path.exists():
            raise FileNotFoundError(path)

    ids = cohort_ids(args.cohort)
    rows: list[dict[str, object]] = []
    for pdb_id in ids:
        raw_dir = args.raw_poses / pdb_id
        scored_dir = args.confidence_poses / pdb_id
        raw = sdf_files(raw_dir)
        scored = sdf_files(scored_dir)
        expected = min(len(raw), args.max_poses)
        rank1 = next(
            (path for path in scored if path.name.lower().startswith("rank1_")),
            None,
        )
        if not raw_dir.is_dir():
            status = "missing_raw_directory"
        elif not raw:
            status = "no_raw_poses"
        elif not scored_dir.is_dir():
            status = "missing_confidence_directory"
        elif not scored:
            status = "no_confidence_poses"
        elif len(scored) < expected:
            status = "incomplete_confidence"
        elif rank1 is None:
            status = "missing_rank1_confidence_pose"
        else:
            status = "complete"
        rows.append({
            "pdb_id": pdb_id,
            "status": status,
            "raw_pose_n": len(raw),
            "expected_confidence_pose_n": expected,
            "confidence_pose_n": len(scored),
            "missing_confidence_pose_n": max(0, expected - len(scored)),
            "rank1_present": rank1 is not None,
            "rank1_pose": str(rank1) if rank1 is not None else "",
            "raw_case_dir": str(raw_dir),
            "confidence_case_dir": str(scored_dir),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "ablation_confidence_audit.csv"
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    incomplete = [row for row in rows if row["status"] != "complete"]
    missing_rank1 = [row for row in rows if not row["rank1_present"]]
    resume_ids = [str(row["pdb_id"]) for row in incomplete]
    (args.output_dir / "resume_ids.txt").write_text(
        "".join(f"{pdb_id}\n" for pdb_id in resume_ids)
    )
    summary = {
        "cohort": str(args.cohort.resolve()),
        "raw_poses": str(args.raw_poses.resolve()),
        "confidence_poses": str(args.confidence_poses.resolve()),
        "cohort_n": len(rows),
        "complete_n": len(rows) - len(incomplete),
        "incomplete_n": len(incomplete),
        "rank1_present_n": len(rows) - len(missing_rank1),
        "rank1_missing_n": len(missing_rank1),
        "raw_pose_total": sum(int(row["raw_pose_n"]) for row in rows),
        "confidence_pose_total": sum(int(row["confidence_pose_n"]) for row in rows),
        "audit_csv": str(audit_path.resolve()),
        "resume_ids": str((args.output_dir / "resume_ids.txt").resolve()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if incomplete:
        print("INCOMPLETE CASES")
        for row in incomplete:
            print(
                f"{row['pdb_id']}\t{row['status']}\t"
                f"raw={row['raw_pose_n']}\tscored={row['confidence_pose_n']}\t"
                f"expected={row['expected_confidence_pose_n']}"
            )
        print(f"ABLATION_CONFIDENCE_INCOMPLETE: {len(incomplete)}/{len(rows)}")
        return 1
    print(f"ABLATION_CONFIDENCE_COMPLETE: {len(rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
