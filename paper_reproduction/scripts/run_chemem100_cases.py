#!/usr/bin/env python3
"""Run a contiguous chunk of preflighted ChemEM100 Stage-1 cases."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["work_index"]))
    return rows


def materialize_map(source: Path, target: Path) -> bool:
    if target.is_file() and target.stat().st_size > 1024:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    if source.suffix.lower() == ".gz":
        input_handle = gzip.open(source, "rb")
    else:
        input_handle = source.open("rb")
    with input_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
    temporary.replace(target)
    return True


def run_case(row: dict[str, str], chemem_bin: str, timeout_hours: float) -> dict[str, object]:
    complex_name = row["complex_name"]
    config = Path(row["stage1_config"])
    output = Path(row["stage1_output"])
    results = output / "fitting" / "results.txt"
    log_path = Path(row["case_log"])
    source_map = Path(row["map_source"])
    densmap = Path(row["decompressed_map"])
    status_path = log_path.with_suffix(".status.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if results.is_file() and results.stat().st_size > 0:
        status = {
            "complex_name": complex_name,
            "case_group": row["case_group"],
            "status": "already_complete",
            "returncode": 0,
            "results": str(results),
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"SKIP {complex_name}: already complete", flush=True)
        return status

    # ChemEM creates stage-specific children such as ``preprocessing`` but
    # assumes the configured output root already exists.
    output.mkdir(parents=True, exist_ok=True)
    generated_map = materialize_map(source_map, densmap)
    command = [
        "timeout", f"{timeout_hours:g}h", "stdbuf", "-oL", "-eL",
        chemem_bin, str(config),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    started_utc = datetime.now(timezone.utc)
    started_clock = time.monotonic()
    print(f"START {complex_name} group={row['case_group']}", flush=True)
    print("COMMAND " + " ".join(command), flush=True)
    with log_path.open("w") as log_handle:
        log_handle.write(
            f"START: {complex_name} at {started_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        )
        log_handle.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
        returncode = process.wait()
        ended_utc = datetime.now(timezone.utc)
        elapsed_seconds = time.monotonic() - started_clock
        log_handle.write(
            f"END: {complex_name} at {ended_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
        )
        log_handle.write(f"WRAPPER_ELAPSED_SECONDS: {elapsed_seconds:.6f}\n")
        log_handle.flush()

    complete = results.is_file() and results.stat().st_size > 0
    if generated_map and densmap.is_file():
        densmap.unlink()
    status = {
        "complex_name": complex_name,
        "case_group": row["case_group"],
        "status": "complete" if complete else ("timeout" if returncode == 124 else "failed"),
        "returncode": returncode,
        "results": str(results),
        "log": str(log_path),
        "start_time_utc": started_utc.isoformat(),
        "end_time_utc": ended_utc.isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "timeout_hours": timeout_hours,
    }
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"END {complex_name}: {status['status']} returncode={returncode}", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--start", required=True, type=int, help="Inclusive zero-based work index")
    parser.add_argument("--end", required=True, type=int, help="Exclusive zero-based work index")
    parser.add_argument("--chemem-bin", default="chemem")
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.start < 0 or args.end <= args.start:
        raise ValueError("Require 0 <= start < end")

    rows = read_rows(args.cases.expanduser().resolve())
    selected = [row for row in rows if args.start <= int(row["work_index"]) < args.end]
    print(f"CHEMEM100 CHUNK start={args.start} end={args.end} selected={len(selected)}", flush=True)
    statuses = [run_case(row, args.chemem_bin, args.timeout_hours) for row in selected]
    completed = sum(status["status"] in {"complete", "already_complete"} for status in statuses)
    print(f"CHEMEM100 CHUNK COMPLETE completed={completed}/{len(statuses)}", flush=True)
    # A failed case must not stop subsequent cases or terminate the Slurm shell.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
