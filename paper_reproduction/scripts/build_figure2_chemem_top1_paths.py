#!/usr/bin/env python3
"""Create one ChemEM rank-1 pose-path row per completed ChemEM100 case."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def norm(value: object) -> str:
    return str(value).strip().upper()


def integer(value: object) -> Optional[int]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def find_old_pose(root: Path, complex_id: str, iteration: int) -> Optional[Path]:
    fitting = root / "runs" / complex_id / "stage1" / "fitting"
    exact = fitting / f"Ligand_{iteration}_refined.sdf"
    if exact.is_file():
        return exact.resolve()
    matches = sorted(fitting.glob(f"Ligand_{iteration}_*_refined.sdf"))
    return matches[0].resolve() if len(matches) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--new-per-pose", required=True, type=Path)
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    benchmark_path = args.benchmark.expanduser().resolve()
    new_pose_path = args.new_per_pose.expanduser().resolve()
    historical_root = args.historical_root.expanduser().resolve()
    benchmark = read_csv(benchmark_path)
    new_pose_rows = read_csv(new_pose_path)
    new_top1: dict[str, str] = {}
    for row in new_pose_rows:
        if integer(row.get("rank")) != 1:
            continue
        complex_id = norm(row.get("complex_name", ""))
        pose = str(row.get("prediction", row.get("prediction_path", ""))).strip()
        if complex_id and pose:
            new_top1[complex_id] = pose

    output_rows: list[dict[str, object]] = []
    for row in benchmark:
        complex_id = norm(row.get("complex_name", ""))
        complete = str(row.get("chemem_complete", "")).strip().lower() in {"1", "true", "yes"}
        source = str(row.get("chemem_source", "")).strip()
        pose = new_top1.get(complex_id, "")
        if not pose and complete:
            iteration = integer(row.get("chemem_top1_iteration"))
            if iteration is not None:
                found = find_old_pose(historical_root, complex_id, iteration)
                pose = str(found) if found else ""
        output_rows.append({
            "complex_name": complex_id,
            "rank": 1,
            "prediction_path": pose,
            "chemem_source": source,
            "chemem_complete": complete,
            "pose_path_exists": bool(pose and Path(pose).expanduser().is_file()),
            "chemem_top1_iteration": row.get("chemem_top1_iteration", ""),
            "chemem_top1_rmsd_A": row.get("chemem_top1_rmsd_A", ""),
        })

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    missing = [row["complex_name"] for row in output_rows if row["chemem_complete"] and not row["pose_path_exists"]]
    print(f"completed ChemEM cases: {sum(bool(row['chemem_complete']) for row in output_rows)}")
    print(f"completed cases missing rank-1 file: {len(missing)}")
    print(f"missing IDs: {','.join(str(value) for value in missing) if missing else 'none'}")
    print(f"CHEMEM_TOP1_PATHS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
