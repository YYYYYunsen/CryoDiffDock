#!/usr/bin/env python3
"""Recover frozen-cohort CryoDiffDock inference runtimes from legacy tqdm logs.

The legacy baseline and guided runs were executed on a larger manifest.  Their
``Docking inference`` progress records contain both the source row index and
complex ID, so docking wall time can be assigned back to individual cases to
the one-second resolution of tqdm's elapsed clock.  The remaining preprocessing
and graph-loading phases do not print case IDs; those measured phase totals are
therefore amortized over the full source manifest and labelled as such.

Every requested cohort ID is retained, including cases for which inference or
downstream V2 expansion failed.  Missing downstream audit rows are reported,
not silently dropped.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path
from typing import Optional


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DOCKING = re.compile(
    r"Docking inference:.*?\|\s*(?P<count>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[0-9:]+)<[^\]\r\n]*?"
    r"Row\s+(?P<row>\d+),\s*complex\s+(?P<complex>[^\],\r\n]+)"
)
LOADING = re.compile(
    r"Loading complexes:.*?\|\s*(?P<count>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[0-9:]+)<"
)
UNNAMED = re.compile(r"(?m)^\s*(?P<count>\d+)it\s+\[(?P<elapsed>[0-9:]+),")
COMPLETED = re.compile(r"Completed\s+(?P<count>\d+)\s*/\s*(?P<total>\d+)\s+inferences")
V2_TOTAL_RUNTIME = re.compile(
    r"total\s+V2\s+search\s+runtime\s*\(s\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
V2_TIMING_BREAKDOWN = re.compile(r"timing\s+breakdown\s*:\s*(\{[^\r\n]+\})", re.IGNORECASE)
ID_COLUMNS = ("complex_name", "pdb_id", "complex_id", "id")


def seconds(value: str) -> float:
    fields = [int(part) for part in value.split(":")]
    if len(fields) == 2:
        minutes, secs = fields
        return float(60 * minutes + secs)
    if len(fields) == 3:
        hours, minutes, secs = fields
        return float(3600 * hours + 60 * minutes + secs)
    raise ValueError(f"Unsupported tqdm elapsed time: {value!r}")


def delimiter(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter(path))
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return list(reader)


def find_id_column(rows: list[dict[str, str]], label: str) -> str:
    if not rows:
        raise ValueError(f"Empty {label} table")
    column = next((name for name in ID_COLUMNS if name in rows[0]), None)
    if column is None:
        raise ValueError(f"No ID column in {label}; columns={sorted(rows[0])}")
    return column


def read_ids(path: Path) -> list[str]:
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    values = [value for value in values if value]
    if not values:
        raise ValueError(f"No IDs in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate cohort IDs in {path}")
    return values


def finite(value: object) -> Optional[float]:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_log_text(
    text: str, manifest_ids: list[str], label: str,
) -> dict[str, object]:
    events = list(DOCKING.finditer(text))
    if not events:
        raise ValueError(f"No 'Docking inference' row records in {label}")

    totals = {int(match.group("total")) for match in events}
    if totals != {len(manifest_ids)}:
        raise ValueError(
            f"Docking log total/manifest mismatch for {label}: "
            f"log totals={sorted(totals)}, manifest={len(manifest_ids)}"
        )

    starts: dict[int, float] = {}
    logged_ids: dict[int, str] = {}
    for match in events:
        row = int(match.group("row"))
        elapsed = seconds(match.group("elapsed"))
        complex_id = match.group("complex").strip().upper()
        if row < 0 or row >= len(manifest_ids):
            raise ValueError(f"Log row outside manifest: row={row}, source={label}")
        if complex_id != manifest_ids[row]:
            raise ValueError(
                f"Log/manifest ID mismatch at row {row}: "
                f"log={complex_id}, manifest={manifest_ids[row]}, source={label}"
            )
        starts[row] = min(starts.get(row, elapsed), elapsed)
        logged_ids[row] = complex_id

    missing_rows = sorted(set(range(len(manifest_ids))) - set(starts))
    if missing_rows:
        preview = ",".join(str(value) for value in missing_rows[:20])
        raise ValueError(
            f"Cannot recover per-case docking time: {len(missing_rows)} row starts "
            f"are absent from {label}; first missing rows={preview}"
        )

    completed = [
        match for match in events
        if int(match.group("count")) == int(match.group("total"))
    ]
    if not completed:
        raise ValueError(f"No completed docking progress record in {label}")
    docking_total = max(seconds(match.group("elapsed")) for match in completed)

    docking_by_row: dict[int, float] = {}
    for row in range(len(manifest_ids)):
        end = starts[row + 1] if row + 1 < len(manifest_ids) else docking_total
        duration = end - starts[row]
        if duration < 0:
            raise ValueError(f"Negative recovered duration at row {row} in {label}")
        docking_by_row[row] = duration

    loading_events = list(LOADING.finditer(text))
    loading_total = 0.0
    loading_end = -1
    loading_start = len(text)
    if loading_events:
        loading_start = min(match.start() for match in loading_events)
        loading_end = max(match.end() for match in loading_events)
        final_loading = [
            match for match in loading_events
            if int(match.group("count")) == int(match.group("total"))
        ]
        if final_loading:
            loading_total = max(seconds(match.group("elapsed")) for match in final_loading)

    first_docking = min(match.start() for match in events)
    unnamed = list(UNNAMED.finditer(text))
    pre_total = max(
        (seconds(match.group("elapsed")) for match in unnamed
         if match.start() < loading_start and int(match.group("count")) == len(manifest_ids)),
        default=0.0,
    )
    post_total = max(
        (seconds(match.group("elapsed")) for match in unnamed
         if loading_end < match.start() < first_docking
         and int(match.group("count")) == len(manifest_ids)),
        default=0.0,
    )
    nondocking_total = pre_total + loading_total + post_total

    return {
        "docking_by_row": docking_by_row,
        "docking_total": docking_total,
        "preprocessing_total": pre_total,
        "loading_total": loading_total,
        "postprocessing_total": post_total,
        "nondocking_total": nondocking_total,
        "nondocking_per_case": nondocking_total / len(manifest_ids),
    }


def parse_log(path: Path, manifest_ids: list[str]) -> dict[str, object]:
    text = ANSI.sub("", path.read_text(errors="replace").replace("\r", "\n"))
    return parse_log_text(text, manifest_ids, str(path))


def parse_combined_log(path: Path, manifest_ids: list[str]) -> list[dict[str, object]]:
    """Split a sequential baseline+guided log at completed-run markers."""
    text = ANSI.sub("", path.read_text(errors="replace").replace("\r", "\n"))
    markers = [
        match for match in COMPLETED.finditer(text)
        if int(match.group("count")) == len(manifest_ids)
        and int(match.group("total")) == len(manifest_ids)
    ]
    sections: list[dict[str, object]] = []
    start = 0
    for arm_index, marker in enumerate(markers, start=1):
        section = text[start:marker.end()]
        start = marker.end()
        if DOCKING.search(section):
            sections.append(
                parse_log_text(
                    section,
                    manifest_ids,
                    f"{path} arm {arm_index}",
                )
            )
    if len(sections) != 2:
        raise ValueError(
            f"Expected exactly two completed inference arms in {path}, "
            f"found {len(sections)}"
        )
    return sections


def legacy_v2_selection_ready(
    audit_path: Path, run_log_path: Path,
) -> dict[str, object]:
    """Recover a run-level selection-ready mean from a legacy V2 run.

    The legacy audit records per-case total wall time but only the run log
    records the aggregate all-condition materialization block.  Consequently
    its deployment-oriented search/selection time is identifiable only as a
    run-level mean, not as an exact per-case value.
    """
    rows = read_table(audit_path)
    case_runtimes = [finite(row.get("case_runtime_seconds")) for row in rows]
    case_runtimes = [value for value in case_runtimes if value is not None]
    if not case_runtimes:
        raise ValueError(f"Legacy V2 audit has no finite case_runtime_seconds: {audit_path}")

    text = run_log_path.read_text(errors="replace")
    breakdowns = V2_TIMING_BREAKDOWN.findall(text)
    if not breakdowns:
        raise ValueError(f"No V2 timing breakdown found in {run_log_path}")
    try:
        breakdown = ast.literal_eval(breakdowns[-1])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Cannot parse V2 timing breakdown in {run_log_path}") from exc
    materialization = finite(breakdown.get("materialization_seconds"))
    if materialization is None:
        raise ValueError(f"V2 timing breakdown lacks materialization_seconds: {run_log_path}")

    audit_total = sum(case_runtimes)
    logged_totals = [float(value) for value in V2_TOTAL_RUNTIME.findall(text)]
    if logged_totals:
        logged_total = logged_totals[-1]
        tolerance = max(1.0, 0.001 * logged_total)
        if abs(audit_total - logged_total) > tolerance:
            raise ValueError(
                "Legacy V2 audit/log runtime mismatch: "
                f"audit_sum={audit_total}, logged_total={logged_total}"
            )
    selection_total = audit_total - materialization
    if selection_total <= 0:
        raise ValueError(
            f"Non-positive legacy selection-ready total: {selection_total}"
        )
    return {
        "case_count": len(case_runtimes),
        "case_runtime_total": audit_total,
        "all_condition_materialization_total": materialization,
        "selection_ready_total": selection_total,
        "selection_ready_per_case": selection_total / len(case_runtimes),
        "audit_path": str(audit_path),
        "run_log_path": str(run_log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frozen-ids", required=True, type=Path)
    parser.add_argument("--baseline-log", type=Path)
    parser.add_argument("--guided-log", type=Path)
    parser.add_argument(
        "--combined-baseline-guided-log",
        type=Path,
        help="Sequential two-arm log containing baseline then guided inference.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pose-scores", type=Path)
    parser.add_argument("--matched-manifest-output", type=Path)
    parser.add_argument("--pose-score-output", type=Path)
    parser.add_argument("--v2-audit", type=Path)
    parser.add_argument("--v2-metrics", type=Path)
    parser.add_argument("--legacy-v2-audit", type=Path)
    parser.add_argument("--legacy-v2-run-log", type=Path)
    parser.add_argument(
        "--run-label",
        default="",
        help="Auditable label for the inference run represented by this input set.",
    )
    args = parser.parse_args()
    if args.pose_score_output and not args.pose_scores:
        raise ValueError("--pose-score-output requires --pose-scores")
    if bool(args.legacy_v2_audit) != bool(args.legacy_v2_run_log):
        raise ValueError("Supply both --legacy-v2-audit and --legacy-v2-run-log")
    separate_logs = args.baseline_log is not None or args.guided_log is not None
    if separate_logs and not (args.baseline_log and args.guided_log):
        raise ValueError("Supply both --baseline-log and --guided-log")
    if separate_logs == (args.combined_baseline_guided_log is not None):
        raise ValueError(
            "Supply either separate baseline/guided logs or one combined log"
        )

    manifest = read_table(args.manifest.expanduser().resolve())
    manifest_id_column = find_id_column(manifest, "manifest")
    manifest_ids = [str(row[manifest_id_column]).strip().upper() for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise ValueError("Manifest IDs are not unique")
    frozen_ids = read_ids(args.frozen_ids.expanduser().resolve())
    row_by_id = {complex_id: row for row, complex_id in enumerate(manifest_ids)}
    absent = [complex_id for complex_id in frozen_ids if complex_id not in row_by_id]
    if absent:
        raise ValueError(f"Frozen IDs absent from manifest: {absent}")

    if args.combined_baseline_guided_log:
        baseline, guided = parse_combined_log(
            args.combined_baseline_guided_log.expanduser().resolve(),
            manifest_ids,
        )
    else:
        baseline = parse_log(args.baseline_log.expanduser().resolve(), manifest_ids)
        guided = parse_log(args.guided_log.expanduser().resolve(), manifest_ids)

    pose_counts: dict[str, int] = {}
    pose_rows_by_id: dict[str, list[dict[str, str]]] = {}
    if args.pose_scores:
        pose_rows = read_table(args.pose_scores.expanduser().resolve())
        pose_id_column = find_id_column(pose_rows, "pose-score")
        for row in pose_rows:
            complex_id = str(row[pose_id_column]).strip().upper()
            pose_counts[complex_id] = pose_counts.get(complex_id, 0) + 1
            pose_rows_by_id.setdefault(complex_id, []).append(row)

    v2_scored_ids: Optional[set[str]] = None
    if args.v2_metrics:
        metric_rows = read_table(args.v2_metrics.expanduser().resolve())
        metric_id_column = find_id_column(metric_rows, "V2 metrics")
        if "top1_rmsd_A" not in metric_rows[0]:
            raise ValueError("V2 metrics lacks top1_rmsd_A")
        v2_scored_ids = {
            str(row[metric_id_column]).strip().upper()
            for row in metric_rows
            if finite(row.get("top1_rmsd_A")) is not None
        }

    legacy_timing: Optional[dict[str, object]] = None
    if args.legacy_v2_audit:
        legacy_timing = legacy_v2_selection_ready(
            args.legacy_v2_audit.expanduser().resolve(),
            args.legacy_v2_run_log.expanduser().resolve(),
        )

    expansion: dict[str, tuple[float, str]] = {}
    if args.v2_audit:
        audit_rows = read_table(args.v2_audit.expanduser().resolve())
        audit_id_column = find_id_column(audit_rows, "V2 audit")
        for row in audit_rows:
            complex_id = str(row[audit_id_column]).strip().upper()
            direct = finite(row.get("deployment_runtime_seconds"))
            selection_ready = finite(row.get("selection_ready_seconds"))
            top1_materialization = finite(row.get("top1_materialization_seconds"))
            case_runtime = finite(row.get("case_runtime_seconds"))
            all_materialization = finite(row.get("materialization_seconds"))
            if direct is not None:
                # In a materialization_mode=top1 timing run this includes only
                # search/selection and writing the single selected pose.
                value = direct
                source = "deployment_runtime_seconds"
            elif selection_ready is not None:
                value = selection_ready
                source = "selection_ready_seconds"
            elif case_runtime is not None and all_materialization is not None:
                # Legacy V2 wrote every condition/oracle pose and only recorded
                # a single materialization total.  Subtracting that block gives
                # the measured search-through-selection boundary without
                # charging deployment for best-of-N pose materialization.
                value = case_runtime - all_materialization
                source = "case_runtime_seconds_minus_all_condition_materialization_seconds"
            elif case_runtime is not None and legacy_timing is not None:
                value = float(legacy_timing["selection_ready_per_case"])
                source = "legacy_run_selection_ready_seconds_amortized"
            else:
                value = None
                source = "missing"
            if value is not None and value < 0:
                raise ValueError(
                    f"Negative V2 expansion/selection runtime for {complex_id}: {value}"
                )
            if complex_id and value is not None:
                if complex_id in expansion:
                    raise ValueError(f"Duplicate V2 audit ID: {complex_id}")
                expansion[complex_id] = (value, source)

    output_rows: list[dict[str, object]] = []
    for frozen_index, complex_id in enumerate(frozen_ids):
        source_row = row_by_id[complex_id]
        baseline_docking = baseline["docking_by_row"][source_row]
        guided_docking = guided["docking_by_row"][source_row]
        baseline_nondocking = float(baseline["nondocking_per_case"])
        guided_nondocking = float(guided["nondocking_per_case"])
        baseline_inference = baseline_docking + baseline_nondocking
        guided_inference = guided_docking + guided_nondocking
        nondocking = baseline_nondocking + guided_nondocking
        inference = baseline_inference + guided_inference
        expansion_record = expansion.get(complex_id)
        expansion_seconds = expansion_record[0] if expansion_record is not None else None
        expansion_runtime_source = expansion_record[1] if expansion_record is not None else "missing"
        upstream_failure = (
            expansion_seconds is None
            and v2_scored_ids is not None
            and complex_id not in v2_scored_ids
        )
        if upstream_failure:
            expansion_seconds = 0.0
            expansion_runtime_source = "not_attempted_due_upstream_technical_failure"
        status = (
            "failed_before_v2_expansion" if upstream_failure else
            "completed" if expansion_seconds is not None else
            "incomplete_missing_v2_expansion"
        )
        observed = inference + (expansion_seconds or 0.0)
        output_rows.append({
            "frozen_index": frozen_index,
            "source_row_index": source_row,
            "complex_name": complex_id,
            "inference_run_label": args.run_label,
            "inference_manifest": str(args.manifest.expanduser().resolve()),
            "method": "CryoDiffDock-Local V2 selected",
            "status": status,
            # A technically failed inference may legitimately have zero pose
            # rows.  Keep it in the runtime denominator and encode the count
            # numerically so the audit itself cannot crash on the failure.
            "pose_score_rows": pose_counts.get(complex_id, 0) if args.pose_scores else "",
            "baseline_docking_seconds_log_delta": baseline_docking,
            "guided_docking_seconds_log_delta": guided_docking,
            "baseline_non_docking_seconds_amortized": baseline_nondocking,
            "guided_non_docking_seconds_amortized": guided_nondocking,
            "baseline_inference_seconds_reconstructed": baseline_inference,
            "guided_inference_seconds_reconstructed": guided_inference,
            "non_docking_seconds_amortized": nondocking,
            "inference_seconds_reconstructed": inference,
            "v2_expansion_selection_materialization_seconds": (
                expansion_seconds if expansion_seconds is not None else ""
            ),
            "v2_expansion_runtime_source_column": expansion_runtime_source,
            "v2_expansion_status": (
                "not_attempted_upstream_failure" if upstream_failure else
                "completed" if expansion_seconds is not None else
                "not_recorded"
            ),
            "legacy_v2_selection_ready_seconds_amortized": (
                legacy_timing["selection_ready_per_case"] if legacy_timing is not None else ""
            ),
            "legacy_v2_timing_case_count": (
                legacy_timing["case_count"] if legacy_timing is not None else ""
            ),
            "observed_runtime_seconds": observed,
            # A failed/missing V2 expansion still consumed the reconstructed
            # inference time.  Preserve that observed attempt runtime instead
            # of turning it into a missing timing value.
            "runtime_seconds": observed,
            "full_v2_runtime_seconds": observed if expansion_seconds is not None else "",
            "timing_note": (
                "baseline+guided docking from ID-labelled tqdm row deltas (1 s clock); "
                "non-docking phase totals amortized across source manifest"
            ),
        })

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    if args.matched_manifest_output:
        matched_path = args.matched_manifest_output.expanduser().resolve()
        matched_path.parent.mkdir(parents=True, exist_ok=True)
        original_fields = list(manifest[0])
        prefix_fields = ["frozen_index", "source_row_index"]
        matched_fields = prefix_fields + [
            field for field in original_fields if field not in prefix_fields
        ]
        with matched_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=matched_fields)
            writer.writeheader()
            for frozen_index, complex_id in enumerate(frozen_ids):
                source_row = row_by_id[complex_id]
                writer.writerow({
                    **manifest[source_row],
                    "frozen_index": frozen_index,
                    "source_row_index": source_row,
                })
        print(f"MATCHED_FROZEN_MANIFEST_OK: {matched_path}")

    if args.pose_score_output:
        pose_path = args.pose_score_output.expanduser().resolve()
        pose_path.parent.mkdir(parents=True, exist_ok=True)
        first_pose = next((rows[0] for rows in pose_rows_by_id.values() if rows), None)
        if first_pose is None:
            raise ValueError("Pose-score table is empty")
        original_fields = list(first_pose)
        prefix_fields = ["frozen_index", "source_row_index"]
        pose_fields = prefix_fields + [
            field for field in original_fields if field not in prefix_fields
        ]
        with pose_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=pose_fields)
            writer.writeheader()
            for frozen_index, complex_id in enumerate(frozen_ids):
                source_row = row_by_id[complex_id]
                for pose_row in pose_rows_by_id.get(complex_id, []):
                    writer.writerow({
                        **pose_row,
                        "frozen_index": frozen_index,
                        "source_row_index": source_row,
                    })
        print(f"MATCHED_FROZEN_POSE_SCORES_OK: {pose_path}")

    missing_pose = [
        row["complex_name"] for row in output_rows
        if args.pose_scores and finite(row["pose_score_rows"]) == 0
    ]
    missing_expansion = [
        row["complex_name"] for row in output_rows
        if row["v2_expansion_status"] == "not_recorded"
    ]
    upstream_failures = [
        row["complex_name"] for row in output_rows
        if row["v2_expansion_status"] == "not_attempted_upstream_failure"
    ]
    print(f"source manifest cases: {len(manifest_ids)}")
    print(f"frozen cases matched exactly: {len(output_rows)}")
    print(f"frozen pose-score-missing IDs: {','.join(missing_pose) if missing_pose else 'none'}")
    print(
        "frozen V2 expansion-audit-missing IDs: "
        f"{','.join(missing_expansion) if missing_expansion else 'none'}"
    )
    print(
        "frozen V2 expansion not attempted after upstream technical failure: "
        f"{','.join(upstream_failures) if upstream_failures else 'none'}"
    )
    if legacy_timing is not None:
        print(
            "legacy V2 selection-ready timing (s): "
            f"total={legacy_timing['selection_ready_total']}, "
            f"cases={legacy_timing['case_count']}, "
            f"amortized_per_case={legacy_timing['selection_ready_per_case']}"
        )
    for label, result in (("baseline", baseline), ("guided", guided)):
        print(
            f"{label} measured phase totals (s): "
            f"pre={result['preprocessing_total']}, load={result['loading_total']}, "
            f"post={result['postprocessing_total']}, docking={result['docking_total']}"
        )
    print(f"FIGURE2_FROZEN_INFERENCE_RUNTIME_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
