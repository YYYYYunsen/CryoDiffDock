#!/usr/bin/env python3
"""Audit ChemEM100 completion and repaired-case readiness without running ChemEM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def load_status(log_path: Path) -> tuple[str, int | None, Path | None]:
    status_path = log_path.with_suffix(".status.json")
    if not nonempty(status_path):
        return "", None, None
    try:
        payload = json.loads(status_path.read_text())
        return str(payload.get("status", "")), payload.get("returncode"), status_path
    except (OSError, json.JSONDecodeError):
        return "unreadable_status_json", None, status_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--retry-cases", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    analysis_path = args.analysis.expanduser().resolve()
    cases_path = args.cases.expanduser().resolve()
    retry_path = args.retry_cases.expanduser().resolve()
    analysis = {
        row["complex_name"].strip().upper(): row
        for row in read_rows(analysis_path)
    }
    cases = {
        row["complex_name"].strip().upper(): row
        for row in read_rows(cases_path)
    }
    retry = {
        row["complex_name"].strip().upper(): row
        for row in read_rows(retry_path)
    }
    retry_prepared_mtime = retry_path.stat().st_mtime

    rows: list[dict[str, object]] = []
    all_ids = sorted(analysis)
    for complex_name in all_ids:
        metric = analysis[complex_name]
        analyzed_complete = truthy(metric.get("chemem_complete", ""))
        case = retry.get(complex_name, cases.get(complex_name, {}))
        output_text = case.get("stage1_output", "")
        log_text = case.get("case_log", "")
        result_path = (
            Path(output_text) / "fitting" / "results.txt"
            if output_text else Path("/nonexistent")
        )
        result_exists = nonempty(result_path)
        status, returncode, status_path = (
            load_status(Path(log_text)) if log_text else ("", None, None)
        )
        status_after_repair = bool(
            status_path is not None
            and status_path.stat().st_mtime >= retry_prepared_mtime
        )
        config_exists = nonempty(Path(case["stage1_config"])) if case.get("stage1_config") else False

        if analyzed_complete:
            state = "COMPLETE_IN_ANALYSIS"
        elif result_exists:
            state = "RESULT_EXISTS_REANALYZE"
        elif complex_name == "9D6D":
            state = "INTENTIONAL_NO_RETRY"
        elif (
            complex_name in retry
            and config_exists
            and status_after_repair
            and status in {"failed", "timeout"}
        ):
            state = "REPAIR_ATTEMPT_FAILED"
        elif complex_name in retry and config_exists:
            state = "REPAIRED_READY_LOCAL"
        elif complex_name == "9D6C":
            state = "9D6C_NEEDS_RECHECK"
        else:
            state = "INCOMPLETE_NOT_IN_RETRY_TABLE"

        rows.append({
            "complex_name": complex_name,
            "selection_phase": metric.get("selection_phase", ""),
            "resolution_stratum": metric.get("resolution_stratum", ""),
            "state": state,
            "repair_action": retry.get(complex_name, {}).get("repair_action", ""),
            "last_runner_status": status,
            "last_returncode": "" if returncode is None else returncode,
            "status_after_repair": status_after_repair,
            "results_exists": result_exists,
            "config_exists": config_exists,
            "results_path": str(result_path) if output_text else "",
        })

    incomplete = [row for row in rows if row["state"] != "COMPLETE_IN_ANALYSIS"]
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row["state"])
        counts[state] = counts.get(state, 0) + 1

    print("CHEMEM100 STATUS COUNTS")
    for state, count in sorted(counts.items()):
        print(f"{state:32s} {count:3d}")
    print("\nINCOMPLETE / STALE CASES")
    fields = (
        "complex_name", "selection_phase", "resolution_stratum", "state",
        "repair_action", "last_runner_status", "last_returncode",
    )
    print("\t".join(fields))
    for row in incomplete:
        print("\t".join(str(row[field]) for field in fields))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nAUDIT OUTPUT: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
