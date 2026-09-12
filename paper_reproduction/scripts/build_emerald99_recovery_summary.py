#!/usr/bin/env python3
"""Build a recovery-aware EMERALD frozen-cohort summary.

The original chunk merger only sees ``run_root/chunks``.  This builder uses
``audit_emerald99_recovery.py``'s selected attempt for every frozen ID, so
successful recovery-round outputs are included without duplicating attempts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Optional


TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", re.MULTILINE)


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Frozen ID list is empty or contains duplicates: {path}")
    return values


def read_table(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return list(reader)


def metric_table(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return {
        str(row.get("metric", "")).strip(): str(row.get("value", "")).strip()
        for row in read_table(path)
        if str(row.get("metric", "")).strip()
    }


def finite(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_metric(values: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if finite(values.get(name)) is not None:
            return values[name]
    return ""


def ranked_top1(case_dir: Path) -> str:
    path = case_dir / "results" / "emerald_rmsd_ranked.tsv"
    if not path.is_file():
        return ""
    rows = read_table(path)
    return str(rows[0].get("pdb_path", "")).strip() if rows else ""


def observed_failure_wall(case_dir: Optional[Path]) -> tuple[str, str]:
    """Return a filesystem-bounded elapsed time for an unfinished attempt."""
    if case_dir is None or not case_dir.is_dir():
        return "", ""
    driver = case_dir / "driver.log"
    if not driver.is_file():
        return "", ""
    match = TIMESTAMP.search(driver.read_text(errors="replace"))
    if not match:
        return "", ""
    started = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    mtimes = [path.stat().st_mtime for path in case_dir.rglob("*") if path.is_file()]
    if not mtimes:
        return "", ""
    elapsed = max(mtimes) - started
    if elapsed < 0:
        return "", ""
    return f"{elapsed:.6f}", "driver_first_timestamp_to_last_case_file_mtime"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-audit", required=True, type=Path)
    parser.add_argument("--frozen-ids", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-completed", type=int)
    args = parser.parse_args()

    audit_path = args.recovery_audit.expanduser().resolve()
    frozen_path = args.frozen_ids.expanduser().resolve()
    frozen_ids = read_ids(frozen_path)
    audit_rows = read_table(audit_path)
    indexed = {str(row.get("pdb_id", "")).strip().upper(): row for row in audit_rows}
    if len(indexed) != len(audit_rows):
        raise ValueError("Recovery audit contains blank or duplicate PDB IDs")
    if set(indexed) != set(frozen_ids):
        raise ValueError(
            f"Recovery audit/frozen mismatch: missing={sorted(set(frozen_ids) - set(indexed))}, "
            f"unexpected={sorted(set(indexed) - set(frozen_ids))}"
        )

    output_rows: list[dict[str, object]] = []
    completed_errors: list[str] = []
    for frozen_index, pdb_id in enumerate(frozen_ids):
        audit = indexed[pdb_id]
        state = str(audit.get("state", "missing")).strip().lower() or "missing"
        case_text = str(audit.get("selected_case_dir", "")).strip()
        case_dir = Path(case_text).expanduser() if case_text else None
        evaluation = metric_table(case_dir / "results" / "emerald_eval_summary.tsv") if case_dir else {}
        runtime = metric_table(case_dir / "results" / "runtime.tsv") if case_dir else {}
        end_to_end = first_metric(runtime, ("end_to_end_wall_s",))
        runtime_source = "runtime.tsv:end_to_end_wall_s" if end_to_end else ""
        if not end_to_end and state != "completed":
            end_to_end, runtime_source = observed_failure_wall(case_dir)
        top1 = first_metric(evaluation, ("top1_ligand_rmsd_A",)) or str(audit.get("top1_rmsd_A", ""))
        topk = first_metric(
            evaluation,
            tuple(sorted(key for key in evaluation if key.startswith("top") and key.endswith("_best_ligand_rmsd_A"))),
        ) or str(audit.get("topk_best_rmsd_A", ""))
        oracle = first_metric(evaluation, ("oracle_best_ligand_rmsd_A",)) or str(audit.get("oracle_rmsd_A", ""))
        top1_pose = ranked_top1(case_dir) if case_dir else ""
        row: dict[str, object] = {
            "frozen_index": frozen_index,
            "pdb_id": pdb_id,
            "status": "completed" if state == "completed" else state,
            "state": state,
            "stage": audit.get("stage", ""),
            "attempt_count": audit.get("attempt_count", ""),
            "exit_code": audit.get("exit_code", ""),
            "top1_rmsd_A": top1,
            "topk_best_rmsd_A": topk,
            "oracle_rmsd_A": oracle,
            "num_ranked_poses": first_metric(evaluation, ("num_ranked_poses",)) or audit.get("num_ranked_poses", ""),
            "completed_trajectories": first_metric(runtime, ("completed_trajectories",)) or audit.get("completed_trajectories", ""),
            "map_resolution_A": first_metric(runtime, ("map_resolution_A",)),
            "prep_wall_s": first_metric(runtime, ("prep_wall_s",)),
            "docking_wall_s": first_metric(runtime, ("docking_wall_s",)),
            "end_to_end_wall_s": end_to_end,
            "runtime_source": runtime_source,
            "cpu_core_s": first_metric(runtime, ("cpu_core_s_estimate", "cpu_core_s")),
            "gpu_s": first_metric(runtime, ("gpu_s",)),
            "max_concurrent_cpu_cores": first_metric(runtime, ("max_concurrent_cpu_cores",)),
            "gpu_count": first_metric(runtime, ("gpu_count",)),
            "workdir": str(case_dir.resolve()) if case_dir else "",
            "top1_pose_path": top1_pose,
            "log": str((case_dir / "driver.log").resolve()) if case_dir and (case_dir / "driver.log").is_file() else "",
            "last_status": audit.get("last_status", ""),
            "source_recovery_audit": str(audit_path),
        }
        if state == "completed":
            missing = [
                name for name in ("top1_rmsd_A", "oracle_rmsd_A", "end_to_end_wall_s")
                if finite(row[name]) is None
            ]
            if not top1_pose or not Path(top1_pose).is_file():
                missing.append("top1_pose_path")
            if missing:
                completed_errors.append(f"{pdb_id}:{','.join(missing)}")
        output_rows.append(row)

    counts: dict[str, int] = {}
    for row in output_rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    completed_n = counts.get("completed", 0)
    if args.expected_completed is not None and completed_n != args.expected_completed:
        raise ValueError(f"Completed count is {completed_n}, expected {args.expected_completed}")
    if completed_errors:
        raise ValueError(f"Completed attempts lack required Figure 2 assets: {completed_errors}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "frozen_n": len(frozen_ids),
        "status_counts": counts,
        "completed_required_assets_valid": not completed_errors,
        "recovery_audit": str(audit_path),
        "frozen_ids": str(frozen_path),
        "output": str(output),
    }
    summary_path = output.with_name(output.stem + "_audit.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"EMERALD_RECOVERY_SUMMARY_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
