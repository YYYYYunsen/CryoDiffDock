#!/usr/bin/env python3
"""Assemble method-native runtime measurements for Figure 2 authoritative-174."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


TRUE = {"1", "true", "yes", "complete", "completed", "success", "succeeded", "ok"}
ITERATION = re.compile(
    r"Iteration\s+\d+\s*,.*?Runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*min",
    re.IGNORECASE,
)
WRAPPER = re.compile(r"WRAPPER_ELAPSED_SECONDS:\s*([0-9]+(?:\.[0-9]+)?)")
STAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def ids(path: Path, expected: int) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if len(values) != expected or len(set(values)) != expected:
        raise ValueError(f"expected {expected} unique IDs, found {len(values)}")
    return values


def finite(value: object) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def truth(value: object) -> bool:
    return str(value).strip().lower() in TRUE


def normalized(table: pd.DataFrame) -> pd.DataFrame:
    column = next(
        (name for name in ("complex_name", "pdb_id", "complex_id", "id") if name in table),
        None,
    )
    if column is None:
        raise ValueError(f"no ID column: {list(table.columns)}")
    result = table.copy()
    result["complex_name"] = result[column].astype(str).str.strip().str.upper()
    duplicates = sorted(
        result.loc[result["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"duplicate complex IDs: {duplicates[:20]}")
    return result


def metric_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    table = pd.read_csv(path, sep="\t", dtype=str)
    if not {"metric", "value"}.issubset(table.columns):
        return {}
    return dict(zip(table["metric"].astype(str), table["value"].astype(str)))


def index_files(roots: list[Path], pattern: str, parent_is_id: bool = False) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(pattern):
            if parent_is_id:
                case_id = path.parent.name.upper()
            else:
                match = re.search(r"(?<![A-Z0-9])([0-9][A-Z0-9]{3})(?![A-Z0-9])", path.name.upper())
                if not match:
                    continue
                case_id = match.group(1)
            result.setdefault(case_id, []).append(path.resolve())
    return result


def parse_chemem(path: Path, case_id: str) -> tuple[float, str, bool]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    has_end = re.search(rf"(?mi)^END:\s+{re.escape(case_id)}(?:\s|$)", text) is not None
    iterations = [float(value) for value in ITERATION.findall(text)]
    if iterations:
        return 60.0 * sum(iterations), "summed_iteration_runtime", has_end
    wrapper = WRAPPER.findall(text)
    if wrapper:
        return float(wrapper[-1]), "wrapper_elapsed_seconds", has_end
    return math.nan, "missing", has_end


def status_elapsed(paths: list[Path]) -> tuple[float, str, Path | None]:
    candidates = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        value = finite(payload.get("elapsed_seconds"))
        if math.isfinite(value) and value >= 0:
            candidates.append((path.stat().st_mtime, value, path))
    if not candidates:
        return math.nan, "missing", None
    _, value, path = max(candidates)
    return value, "case_status_elapsed_seconds", path


def chemem_runtime(
    cohort: list[str], results: pd.DataFrame, roots: list[Path], timeout_cap: float,
) -> pd.DataFrame:
    logs = index_files(roots, "*.log")
    statuses = index_files(roots, "*.status.json")
    indexed = results.set_index("complex_name")
    rows = []
    for case_id in cohort:
        record = indexed.loc[case_id] if case_id in indexed.index else pd.Series(dtype=object)
        completed = truth(record.get("chemem_complete", False))
        options = []
        for log in logs.get(case_id, []):
            value, source, has_end = parse_chemem(log, case_id)
            if math.isfinite(value) and value >= 0:
                options.append((has_end, log.stat().st_mtime, value, source, log))
        if options:
            if completed and any(option[0] for option in options):
                options = [option for option in options if option[0]]
            _, _, wall, source, source_path = max(options)
        else:
            wall, source, source_path = status_elapsed(statuses.get(case_id, []))
        source_status = str(record.get("chemem_source_status", "missing")).lower()
        if not math.isfinite(wall) and "timeout" in source_status:
            wall, source, source_path = timeout_cap, "protocol_timeout_cap", None
        rows.append({
            "complex_name": case_id,
            "method": "ChemEM Stage 1",
            "status": "completed" if completed else source_status,
            "completed": completed,
            "runtime_seconds": wall,
            "runtime_minutes": wall / 60.0 if math.isfinite(wall) else math.nan,
            "runtime_source": source,
            "runtime_source_path": "" if source_path is None else str(source_path),
        })
    return pd.DataFrame(rows)


def emerald_case_runtime(case_dir: Path) -> tuple[float, str]:
    metrics = metric_file(case_dir / "results" / "runtime.tsv")
    direct = finite(metrics.get("end_to_end_wall_s"))
    if math.isfinite(direct) and direct >= 0:
        return direct, "runtime.tsv:end_to_end_wall_s"
    driver = case_dir / "driver.log"
    if not driver.is_file():
        return math.nan, "missing"
    text = driver.read_text(errors="replace")
    stamps = [
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
        for value in STAMP.findall(text)
    ]
    if len(stamps) >= 2 and stamps[-1] >= stamps[0]:
        return stamps[-1] - stamps[0], "driver_first_to_last_timestamp"
    if stamps:
        mtimes = [path.stat().st_mtime for path in case_dir.rglob("*") if path.is_file()]
        if mtimes and max(mtimes) >= stamps[0]:
            return max(mtimes) - stamps[0], "driver_start_to_last_case_file_mtime"
    return math.nan, "missing"


def emerald_runtime(
    cohort: list[str], results: pd.DataFrame, roots: list[Path], timeout_cap: float,
) -> pd.DataFrame:
    logs = index_files(roots, "driver.log", parent_is_id=True)
    indexed = results.set_index("complex_name")
    rows = []
    for case_id in cohort:
        record = indexed.loc[case_id] if case_id in indexed.index else pd.Series(dtype=object)
        status = str(record.get("status", "missing")).strip().lower()
        rmsd = finite(record.get("top1_rmsd_A"))
        completed = truth(status) and math.isfinite(rmsd)
        wall = finite(record.get("end_to_end_wall_s"))
        source = "selected_summary:end_to_end_wall_s" if math.isfinite(wall) else "missing"
        source_path = ""
        candidates: list[Path] = []
        raw_log = str(record.get("log", "")).strip()
        if raw_log and Path(raw_log).is_file():
            candidates.append(Path(raw_log).resolve())
        raw_summary = str(record.get("source_summary", "")).strip()
        if raw_summary:
            candidate = Path(raw_summary).parent / case_id / "driver.log"
            if candidate.is_file():
                candidates.append(candidate.resolve())
        selected_candidates = set(candidates)
        candidates.extend(logs.get(case_id, []))
        if not math.isfinite(wall):
            options = []
            for log in dict.fromkeys(candidates):
                value, candidate_source = emerald_case_runtime(log.parent)
                if math.isfinite(value) and value >= 0:
                    has_final = (
                        (log.parent / "results" / "runtime.tsv").is_file()
                        and (log.parent / "results" / "emerald_eval_summary.tsv").is_file()
                    )
                    options.append((
                        log in selected_candidates,
                        has_final,
                        log.stat().st_mtime,
                        value,
                        candidate_source,
                        log,
                    ))
            if options:
                if any(option[0] for option in options):
                    options = [option for option in options if option[0]]
                if completed and any(option[1] for option in options):
                    options = [option for option in options if option[1]]
                _, _, _, wall, source, source_log = max(options)
                source_path = str(source_log)
        if status == "timeout" and not math.isfinite(wall):
            wall, source = timeout_cap, "protocol_timeout_cap"
        rows.append({
            "complex_name": case_id,
            "method": "EMERALD",
            "status": status,
            "completed": completed,
            "runtime_seconds": wall,
            "runtime_minutes": wall / 60.0 if math.isfinite(wall) else math.nan,
            "runtime_source": source,
            "runtime_source_path": source_path,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--chemem-results", required=True, type=Path)
    parser.add_argument("--chemem-root", action="append", default=[], type=Path)
    parser.add_argument("--v2-runtime", required=True, type=Path)
    parser.add_argument("--emerald-results", required=True, type=Path)
    parser.add_argument("--emerald-root", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--timeout-cap-s", type=float, default=43200.0)
    args = parser.parse_args()

    cohort = ids(args.cohort.expanduser().resolve(), args.expected_n)
    chemem_results = normalized(pd.read_csv(args.chemem_results.expanduser().resolve()))
    emerald_results = normalized(pd.read_csv(args.emerald_results.expanduser().resolve(), sep="\t"))
    chemem = chemem_runtime(
        cohort, chemem_results,
        [path.expanduser().resolve() for path in args.chemem_root],
        args.timeout_cap_s,
    )
    emerald = emerald_runtime(
        cohort, emerald_results,
        [path.expanduser().resolve() for path in args.emerald_root],
        args.timeout_cap_s,
    )

    v2_raw = normalized(pd.read_csv(args.v2_runtime.expanduser().resolve()))
    v2_index = v2_raw.set_index("complex_name")
    v2_rows = []
    for case_id in cohort:
        record = v2_index.loc[case_id] if case_id in v2_index.index else pd.Series(dtype=object)
        wall = finite(record.get("runtime_seconds"))
        if not math.isfinite(wall):
            wall = finite(record.get("observed_runtime_seconds"))
        status = str(record.get("status", "missing"))
        v2_rows.append({
            "complex_name": case_id,
            "method": "CryoDiffDock-Local V2 selected",
            "status": status,
            "completed": status.strip().lower() == "completed",
            "runtime_seconds": wall,
            "runtime_minutes": wall / 60.0 if math.isfinite(wall) else math.nan,
            "runtime_source": "reconstructed_inference_plus_recorded_v2_attempt",
            "runtime_source_path": str(args.v2_runtime.expanduser().resolve()),
        })
    v2 = pd.DataFrame(v2_rows)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chemem.to_csv(output / "chemem_authoritative174_runtime.csv", index=False)
    v2.to_csv(output / "cryodiffdock_authoritative174_runtime.csv", index=False)
    emerald.to_csv(output / "emerald_authoritative174_runtime.csv", index=False)
    long = pd.concat([chemem, v2, emerald], ignore_index=True)
    if len(long) != 3 * len(cohort):
        raise RuntimeError(f"runtime ledger row count mismatch: {len(long)}")
    if long.duplicated(["complex_name", "method"]).any():
        raise RuntimeError("runtime ledger has duplicate complex/method rows")
    long.rename(columns={"complex_name": "complex_id", "runtime_seconds": "wall_s"}).to_csv(
        output / "runtime_per_complex_long.csv", index=False
    )
    audit = {
        "cohort_n": len(cohort),
        "runtime_finite_n": {
            method: int(group["runtime_seconds"].notna().sum())
            for method, group in (("ChemEM Stage 1", chemem), ("CryoDiffDock-Local V2 selected", v2), ("EMERALD", emerald))
        },
        "runtime_missing_ids": {
            method: group.loc[group["runtime_seconds"].isna(), "complex_name"].tolist()
            for method, group in (("ChemEM Stage 1", chemem), ("CryoDiffDock-Local V2 selected", v2), ("EMERALD", emerald))
        },
        "runtime_nonpositive_ids": {
            method: group.loc[
                group["runtime_seconds"].notna() & group["runtime_seconds"].le(0),
                "complex_name",
            ].tolist()
            for method, group in (("ChemEM Stage 1", chemem), ("CryoDiffDock-Local V2 selected", v2), ("EMERALD", emerald))
        },
        "runtime_source_counts": {
            method: group["runtime_source"].value_counts(dropna=False).to_dict()
            for method, group in (("ChemEM Stage 1", chemem), ("CryoDiffDock-Local V2 selected", v2), ("EMERALD", emerald))
        },
        "timeout_cap_s": args.timeout_cap_s,
        "output": str(output / "runtime_per_complex_long.csv"),
    }
    (output / "runtime_collection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"FIGURE2_AUTHORITATIVE174_RUNTIME_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
