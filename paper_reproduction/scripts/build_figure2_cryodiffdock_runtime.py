#!/usr/bin/env python3
"""Build auditable CryoDiffDock-Local V2 method-native runtimes for Figure 2.

The frozen inference run is batched and has no legacy per-case timer.  This
script therefore amortizes one measured cohort-level inference wall time over
the timed cohort, then adds each case's already-recorded V2 adaptive-search,
selection, and materialization time from ``per_complex_audit.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Optional


ID_COLUMNS = ("complex_name", "pdb_id", "complex_id", "id")
REAL_LINE = re.compile(r"^real\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)


def delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter(path)))


def id_column(row: dict[str, str]) -> str:
    for candidate in ID_COLUMNS:
        if candidate in row:
            return candidate
    raise ValueError(f"No ID column found; columns={sorted(row)}")


def read_cohort(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        table = rows(path)
        if not table:
            raise ValueError(f"Empty cohort: {path}")
        column = id_column(table[0])
        values = [str(row[column]).strip().upper() for row in table]
    else:
        values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in cohort: {path}")
    return values


def finite(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_time_file(path: Path) -> float:
    text = path.read_text(errors="replace")
    match = REAL_LINE.search(text)
    if not match:
        raise ValueError(f"No '/usr/bin/time -p' real-seconds line in {path}")
    value = float(match.group(1))
    if value <= 0:
        raise ValueError(f"Non-positive elapsed time in {path}: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--v2-audit", required=True, type=Path)
    timing = parser.add_mutually_exclusive_group(required=True)
    timing.add_argument("--inference-time-file", type=Path, help="Output of /usr/bin/time -p for baseline+guided inference")
    timing.add_argument("--inference-total-seconds", type=float, help="Measured baseline+guided inference wall seconds")
    parser.add_argument("--inference-case-count", type=int, help="Cases included in the inference timing; defaults to cohort size")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--v1-expansion-time-file", type=Path, help="Optional /usr/bin/time -p output for frozen V1 expansion+selection")
    parser.add_argument("--v1-output", type=Path, help="Optional V1 runtime CSV; requires --v1-expansion-time-file")
    args = parser.parse_args()
    if bool(args.v1_expansion_time_file) != bool(args.v1_output):
        raise ValueError("--v1-expansion-time-file and --v1-output must be supplied together")

    cohort = read_cohort(args.cohort.expanduser().resolve())
    audit_path = args.v2_audit.expanduser().resolve()
    audit_rows = rows(audit_path)
    if not audit_rows:
        raise ValueError(f"Empty V2 audit: {audit_path}")
    audit_id = id_column(audit_rows[0])
    if "case_runtime_seconds" not in audit_rows[0]:
        raise ValueError(f"V2 audit lacks case_runtime_seconds: {audit_path}")
    audit: dict[str, float] = {}
    for row in audit_rows:
        complex_id = str(row.get(audit_id, "")).strip().upper()
        value = finite(row.get("case_runtime_seconds"))
        if complex_id and value is not None:
            if complex_id in audit:
                raise ValueError(f"Duplicate V2 audit ID: {complex_id}")
            audit[complex_id] = value

    if args.inference_time_file:
        inference_total = parse_time_file(args.inference_time_file.expanduser().resolve())
        inference_source = str(args.inference_time_file.expanduser().resolve())
    else:
        inference_total = float(args.inference_total_seconds)
        inference_source = "command_line_inference_total_seconds"
    count = args.inference_case_count or len(cohort)
    if count <= 0 or inference_total <= 0:
        raise ValueError("Inference seconds and case count must be positive")
    inference_per_case = inference_total / count

    output_rows: list[dict[str, object]] = []
    for complex_id in cohort:
        expansion = audit.get(complex_id)
        output_rows.append({
            "complex_name": complex_id,
            "method": "CryoDiffDock-Local V2 selected",
            "status": "completed" if expansion is not None else "missing_v2_audit",
            "runtime_seconds": inference_per_case + expansion if expansion is not None else "",
            "inference_seconds_amortized": inference_per_case,
            "expansion_selection_materialization_seconds": expansion if expansion is not None else "",
            "inference_total_seconds": inference_total,
            "inference_case_count": count,
            "inference_source": inference_source,
            "expansion_source": str(audit_path),
            "timing_boundary": "amortized baseline+guided inference wall + per-case V2 expansion/selection/materialization",
        })

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    missing = [row["complex_name"] for row in output_rows if row["status"] != "completed"]
    print(f"cohort cases: {len(cohort)}")
    print(f"inference total seconds: {inference_total}")
    print(f"inference seconds/case (amortized): {inference_per_case}")
    print(f"V2 runtime rows completed: {len(cohort) - len(missing)}")
    print(f"missing V2 audit IDs: {','.join(missing) if missing else 'none'}")
    print(f"CRYODIFFDOCK_FIGURE2_RUNTIME_OK: {output}")

    if args.v1_expansion_time_file and args.v1_output:
        v1_time_path = args.v1_expansion_time_file.expanduser().resolve()
        v1_total = parse_time_file(v1_time_path)
        v1_per_case = v1_total / count
        v1_rows = [{
            "complex_name": complex_id,
            "method": "CryoDiffDock-Local V1 frozen S0",
            "status": "completed",
            "runtime_seconds": inference_per_case + v1_per_case,
            "inference_seconds_amortized": inference_per_case,
            "expansion_selection_seconds_amortized": v1_per_case,
            "inference_total_seconds": inference_total,
            "expansion_selection_total_seconds": v1_total,
            "timing_case_count": count,
            "inference_source": inference_source,
            "expansion_selection_source": str(v1_time_path),
            "timing_boundary": "amortized baseline+guided inference wall + amortized frozen V1 expansion/selection wall",
        } for complex_id in cohort]
        v1_output = args.v1_output.expanduser().resolve()
        v1_output.parent.mkdir(parents=True, exist_ok=True)
        with v1_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(v1_rows[0]))
            writer.writeheader()
            writer.writerows(v1_rows)
        print(f"V1 expansion+selection total seconds: {v1_total}")
        print(f"V1 runtime seconds/case (fully amortized): {inference_per_case + v1_per_case}")
        print(f"CRYODIFFDOCK_V1_FIGURE2_RUNTIME_OK: {v1_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
