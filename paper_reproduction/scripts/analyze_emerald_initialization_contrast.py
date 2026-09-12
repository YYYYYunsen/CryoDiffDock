#!/usr/bin/env python3
"""Build the paired EMERALD initialization supplementary analysis.

This analysis contrasts the privileged ``sdf + native_perturbed`` protocol
with the deployment-like ``smiles_3d + site_center`` protocol.  Protocols are
verified from each selected driver log.  The result is intentionally called a
protocol contrast, not a single-factor ablation, because conformer source and
initial placement differ and the recorded charge protocol may also differ.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, wilcoxon


TRUE_STATUS = {"complete", "completed", "passed", "success", "succeeded", "ok"}
ID_COLUMNS = ("complex_name", "pdb_id", "case_id", "complex_id", "id")
METRIC_ALIASES = {
    "top1_rmsd_A": ("top1_rmsd_A", "top1_ligand_rmsd_A"),
    "top5_best_rmsd_A": (
        "top5_best_rmsd_A", "topk_best_rmsd_A", "top5_best_ligand_rmsd_A"
    ),
    "oracle_rmsd_A": ("oracle_rmsd_A", "oracle_best_ligand_rmsd_A"),
}
RUNTIME_ALIASES = ("end_to_end_wall_s", "elapsed_s", "runtime_s")
CONFORMER_PATTERN = re.compile(r"input conformer mode:\s*([^\s;]+)")
START_PATTERN = re.compile(r"start position mode:\s*([^\s;]+)")
CHARGE_PATTERN = re.compile(r"AM1-BCC:\s*([^\s;]+)")
TRAJECTORY_PATTERN = re.compile(r"trajectories:\s*(\d+)")
TRANSLATION_PATTERN = re.compile(r"maximum initial translation:\s*([0-9.]+)\s*A")
TIMESTAMP_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


@dataclass(frozen=True)
class Record:
    complex_name: str
    source_group: str
    source_priority: int
    status: str
    top1_rmsd_A: float
    top5_best_rmsd_A: float
    oracle_rmsd_A: float
    runtime_s: float
    runtime_source: str
    conformer_mode: str
    start_position_mode: str
    am1bcc: str
    trajectories: str
    maximum_initial_translation_A: str
    log: str
    workdir: str
    source_data: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_labeled(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/or/relative/path")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("source label cannot be empty")
    return label, Path(raw_path).expanduser().resolve()


def finite(value: object) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def first_value(row: dict[str, object], names: Iterable[str]) -> object:
    for name in names:
        value = row.get(name, "")
        if str(value).strip() not in {"", "nan", "NaN", "None"}:
            return value
    return ""


def normalized_id(row: dict[str, object]) -> str:
    return str(first_value(row, ID_COLUMNS)).strip().upper()


def protocol_from_log(path: Path | None) -> tuple[str, str, str, str, str]:
    if path is None or not path.is_file():
        return ("missing_log",) * 5
    text = path.read_text(errors="replace")
    conformer = CONFORMER_PATTERN.search(text)
    start = START_PATTERN.search(text)
    charge = CHARGE_PATTERN.search(text)
    trajectories = TRAJECTORY_PATTERN.search(text)
    translation = TRANSLATION_PATTERN.search(text)
    return (
        conformer.group(1) if conformer else "unrecorded",
        start.group(1) if start else "unrecorded",
        charge.group(1) if charge else "unrecorded",
        trajectories.group(1) if trajectories else "unrecorded",
        translation.group(1) if translation else "unrecorded",
    )


def recover_runtime_s(log: Path | None, workdir: Path | None) -> tuple[float, str]:
    case_dir = workdir or (log.parent if log is not None else None)
    if case_dir is not None:
        runtime_path = case_dir / "results" / "runtime.tsv"
        if runtime_path.is_file():
            metrics = read_key_value_metrics(runtime_path)
            value = finite(first_value(metrics, RUNTIME_ALIASES))
            if math.isfinite(value) and value >= 0:
                return value, "runtime.tsv"
    if log is None or not log.is_file():
        return math.nan, "missing"
    text = log.read_text(errors="replace")
    timestamps = [
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S").timestamp()
        for value in TIMESTAMP_PATTERN.findall(text)
    ]
    if len(timestamps) >= 2 and timestamps[-1] >= timestamps[0]:
        return timestamps[-1] - timestamps[0], "driver_first_to_last_timestamp"
    if timestamps and case_dir is not None:
        mtimes = [path.stat().st_mtime for path in case_dir.rglob("*") if path.is_file()]
        if mtimes and max(mtimes) >= timestamps[0]:
            return (
                max(mtimes) - timestamps[0],
                "driver_start_to_last_case_file_mtime",
            )
    return math.nan, "missing"


def resolve_log(row: dict[str, object], source: Path) -> Path | None:
    raw_log = str(row.get("log", "")).strip()
    if raw_log:
        path = Path(raw_log).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        if path.is_file():
            return path.resolve()
    raw_workdir = str(row.get("workdir", "")).strip()
    if raw_workdir:
        workdir = Path(raw_workdir).expanduser()
        if not workdir.is_absolute():
            workdir = source.parent / workdir
        candidate = workdir / "driver.log"
        if candidate.is_file():
            return candidate.resolve()
    return None


def build_record(
    row: dict[str, object], source_group: str, priority: int, source: Path,
    log: Path | None, workdir: Path | None = None,
) -> Record:
    conformer, start, charge, trajectories, translation = protocol_from_log(log)
    metrics = {
        output: finite(first_value(row, aliases))
        for output, aliases in METRIC_ALIASES.items()
    }
    runtime_s = finite(first_value(row, RUNTIME_ALIASES))
    runtime_source = "summary_runtime_column" if math.isfinite(runtime_s) else "missing"
    if not math.isfinite(runtime_s):
        runtime_s, runtime_source = recover_runtime_s(log, workdir)
    return Record(
        complex_name=normalized_id(row),
        source_group=source_group,
        source_priority=priority,
        status=str(row.get("status", "completed")).strip().lower(),
        runtime_s=runtime_s,
        runtime_source=runtime_source,
        conformer_mode=conformer,
        start_position_mode=start,
        am1bcc=charge,
        trajectories=trajectories,
        maximum_initial_translation_A=translation,
        log="" if log is None else str(log),
        workdir=str(workdir or str(row.get("workdir", ""))),
        source_data=str(source),
        **metrics,
    )


def read_summary(label: str, path: Path, priority: int) -> list[Record]:
    if not path.is_file():
        raise FileNotFoundError(path)
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    records: list[Record] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"summary has no header: {path}")
        for raw in reader:
            row: dict[str, object] = dict(raw)
            case_id = normalized_id(row)
            if not case_id:
                continue
            log = resolve_log(row, path)
            if log is None:
                # Per-case recovery summaries live directly beside the case
                # directory: chunks/<chunk>/emerald_paper_summary.tsv and
                # chunks/<chunk>/<CASE>/driver.log.  Those summaries do not
                # consistently record a log/workdir column.
                for candidate in (
                    path.parent / case_id / "driver.log",
                    path.parent / case_id.lower() / "driver.log",
                ):
                    if candidate.is_file():
                        log = candidate.resolve()
                        break
            records.append(build_record(row, label, priority, path, log))
    return records


def read_key_value_metrics(path: Path) -> dict[str, object]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if rows and [value.strip().lower() for value in rows[0][:2]] == ["metric", "value"]:
        rows = rows[1:]
    return {row[0].strip(): row[1].strip() for row in rows if len(row) >= 2}


def read_case_root(label: str, root: Path, priority: int) -> list[Record]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: list[Record] = []
    # Completed EMERALD paper runs write the full Top-1/Top-k/oracle metrics to
    # ``emerald_eval_summary.tsv``.  Some older attempts additionally contain
    # ``ranking_metrics.tsv``.  Index case directories from either filename,
    # then merge both tables with the evaluation summary taking precedence.
    workdirs = {
        metrics_path.parent.parent.resolve()
        for pattern in (
            "results/emerald_eval_summary.tsv",
            "results/ranking_metrics.tsv",
        )
        for metrics_path in root.rglob(pattern)
    }
    for workdir in sorted(workdirs):
        case_id = workdir.name.upper()
        ranking_path = workdir / "results" / "ranking_metrics.tsv"
        evaluation_path = workdir / "results" / "emerald_eval_summary.tsv"
        row = {
            **(read_key_value_metrics(ranking_path) if ranking_path.is_file() else {}),
            **(read_key_value_metrics(evaluation_path) if evaluation_path.is_file() else {}),
        }
        row["complex_name"] = case_id
        row["status"] = "completed"
        log = workdir / "driver.log"
        source = evaluation_path if evaluation_path.is_file() else ranking_path
        records.append(
            build_record(
                row, label, priority, source.resolve(),
                log.resolve() if log.is_file() else None, workdir,
            )
        )
    return records


def read_source(label: str, path: Path, priority: int) -> list[Record]:
    if path.is_file():
        return read_summary(label, path, priority)
    if not path.is_dir():
        raise FileNotFoundError(path)
    records: list[Record] = []
    summaries = sorted(path.rglob("emerald_paper_summary.tsv"))
    for summary in summaries:
        records.extend(read_summary(label, summary, priority))
    records.extend(read_case_root(label, path, priority))
    return records


def completed(record: Record) -> bool:
    return (
        record.status in TRUE_STATUS
        and all(
            math.isfinite(getattr(record, column))
            for column in METRIC_ALIASES
        )
    )


def same_metrics(records: list[Record]) -> bool:
    signatures = {
        tuple(round(getattr(record, column), 8) for column in METRIC_ALIASES)
        for record in records
    }
    return len(signatures) == 1


def select_records(
    records: list[Record], protocol: tuple[str, str], label: str,
) -> dict[str, Record]:
    eligible = [
        record for record in records
        if completed(record)
        and (record.conformer_mode, record.start_position_mode) == protocol
    ]
    grouped: dict[str, list[Record]] = {}
    for record in eligible:
        grouped.setdefault(record.complex_name, []).append(record)

    selected: dict[str, Record] = {}
    for case_id, options in grouped.items():
        highest = max(record.source_priority for record in options)
        finalists = [record for record in options if record.source_priority == highest]
        # Aggregate and chunk summaries can point to the exact same case log.
        by_log: dict[str, Record] = {}
        for record in finalists:
            by_log[record.log or record.workdir] = record
        finalists = list(by_log.values())
        if len(finalists) > 1 and not same_metrics(finalists):
            details = [(record.source_data, record.log) for record in finalists]
            raise ValueError(
                f"ambiguous {label} attempts for {case_id} at equal source "
                f"priority; specify a narrower source: {details}"
            )
        selected[case_id] = sorted(
            finalists, key=lambda record: (record.source_data, record.log)
        )[-1]
    return selected


def mcnemar_exact(native_success: np.ndarray, site_success: np.ndarray) -> tuple[int, int, float]:
    native_only = int(np.sum(native_success & ~site_success))
    site_only = int(np.sum(~native_success & site_success))
    discordant = native_only + site_only
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(native_only, site_only), discordant, 0.5).pvalue
    )
    return native_only, site_only, pvalue


def wilcoxon_pvalues(native: np.ndarray, site: np.ndarray) -> tuple[float, float]:
    difference = site - native
    if np.allclose(difference, 0):
        return 1.0, 1.0
    return (
        float(wilcoxon(site, native, alternative="two-sided", zero_method="wilcox").pvalue),
        float(wilcoxon(site, native, alternative="greater", zero_method="wilcox").pvalue),
    )


def bootstrap_delta(
    native: np.ndarray, site: np.ndarray, repeats: int, seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(native)
    indexes = rng.integers(0, n, size=(repeats, n))
    differences = site - native
    mean_samples = differences[indexes].mean(axis=1)
    median_samples = np.median(differences[indexes], axis=1)
    return (
        float(np.quantile(mean_samples, 0.025)),
        float(np.quantile(mean_samples, 0.975)),
        float(np.quantile(median_samples, 0.025)),
        float(np.quantile(median_samples, 0.975)),
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-summary", action="append", default=[], type=parse_labeled)
    parser.add_argument("--native-case-root", action="append", default=[], type=parse_labeled)
    parser.add_argument("--sitecenter-source", action="append", default=[], type=parse_labeled)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-native", type=int, default=15)
    parser.add_argument("--expected-historical", type=int, default=12)
    parser.add_argument("--expected-pilot", type=int, default=3)
    parser.add_argument("--pilot-id", action="append", default=[])
    parser.add_argument("--allow-incomplete-sitecenter", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.native_summary or not args.native_case_root or not args.sitecenter_source:
        raise ValueError("native summary, native case root, and site-center sources are required")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    native_records: list[Record] = []
    priority = 0
    for label, path in args.native_summary:
        native_records.extend(read_summary(label, path, priority))
        priority += 1
    for label, path in args.native_case_root:
        native_records.extend(read_case_root(label, path, priority))
        priority += 1
    native_protocol_records = [
        record for record in native_records
        if completed(record)
        and (record.conformer_mode, record.start_position_mode)
        == ("sdf", "native_perturbed")
    ]
    native_ids_by_input_group = {
        group: sorted({
            record.complex_name for record in native_protocol_records
            if record.source_group == group
        })
        for group in sorted({record.source_group for record in native_protocol_records})
    }
    native_memberships: dict[str, list[str]] = {}
    for group, case_ids in native_ids_by_input_group.items():
        for case_id in case_ids:
            native_memberships.setdefault(case_id, []).append(group)
    overlapping_native_ids = {
        case_id: groups for case_id, groups in native_memberships.items()
        if len(groups) > 1
    }
    native = select_records(native_records, ("sdf", "native_perturbed"), "native")
    if len(native) != args.expected_native:
        raise ValueError(
            f"expected {args.expected_native} completed native-perturbed cases, "
            f"found {len(native)}: {sorted(native)}"
        )
    pilot_ids = {value.strip().upper() for value in args.pilot_id if value.strip()}
    if len(pilot_ids) != args.expected_pilot:
        raise ValueError(
            f"expected {args.expected_pilot} unique --pilot-id values, "
            f"found {len(pilot_ids)}: {sorted(pilot_ids)}"
        )
    pilot_missing_from_native = sorted(pilot_ids - set(native))
    if pilot_missing_from_native:
        raise ValueError(
            "explicit pilot IDs are absent from the completed native-perturbed "
            f"set: {pilot_missing_from_native}"
        )
    case_groups = {
        case_id: "pilot3" if case_id in pilot_ids else "historical12"
        for case_id in native
    }
    group_counts = Counter(case_groups.values())
    if group_counts.get("historical12", 0) != args.expected_historical:
        raise ValueError(f"historical12 count mismatch: {dict(group_counts)}")
    if group_counts.get("pilot3", 0) != args.expected_pilot:
        raise ValueError(f"pilot3 count mismatch: {dict(group_counts)}")

    site_records: list[Record] = []
    for priority, (label, path) in enumerate(args.sitecenter_source):
        site_records.extend(read_source(label, path, priority))
    site_all = select_records(
        site_records, ("smiles_3d", "site_center"), "site-center"
    )
    # Keep the inferential table restricted to the native-perturbed target
    # cohort.  Other valid site-center cases are useful for source auditing but
    # must not inflate the reported counterpart count.
    site = {
        case_id: site_all[case_id]
        for case_id in native
        if case_id in site_all
    }

    ordered_ids = [
        case_id for group in ("historical12", "pilot3")
        for case_id in sorted(
            case_id for case_id in native if case_groups[case_id] == group
        )
    ]
    missing_site = [case_id for case_id in ordered_ids if case_id not in site]
    if missing_site and not args.allow_incomplete_sitecenter:
        raise ValueError(
            f"site-center counterparts are incomplete ({len(missing_site)}/"
            f"{len(ordered_ids)} missing): {missing_site}; rerun with "
            "--allow-incomplete-sitecenter only for a provisional preview"
        )

    per_case: list[dict[str, object]] = []
    for case_id in ordered_ids:
        left = native[case_id]
        right = site.get(case_id)
        row: dict[str, object] = {
            "complex_name": case_id,
            "cohort_group": case_groups[case_id],
            "native_selected_source_group": left.source_group,
            "paired_complete": right is not None,
            "native_conformer_mode": left.conformer_mode,
            "native_start_position_mode": left.start_position_mode,
            "native_am1bcc": left.am1bcc,
            "native_trajectories": left.trajectories,
            "native_maximum_initial_translation_A": left.maximum_initial_translation_A,
            "native_runtime_s": left.runtime_s,
            "native_runtime_source": left.runtime_source,
            "native_log": left.log,
            "native_source_data": left.source_data,
        }
        for endpoint in METRIC_ALIASES:
            row[f"native_{endpoint}"] = getattr(left, endpoint)
        if right is None:
            row.update({
                "sitecenter_status": "missing_or_incomplete",
                "sitecenter_conformer_mode": "",
                "sitecenter_start_position_mode": "",
                "sitecenter_am1bcc": "",
                "sitecenter_trajectories": "",
                "sitecenter_maximum_initial_translation_A": "",
                "sitecenter_runtime_s": math.nan,
                "sitecenter_runtime_source": "missing",
                "sitecenter_log": "",
                "sitecenter_source_data": "",
                "charge_protocol_matched": False,
            })
            for endpoint in METRIC_ALIASES:
                row[f"sitecenter_{endpoint}"] = math.nan
                row[f"delta_sitecenter_minus_native_{endpoint}"] = math.nan
        else:
            row.update({
                "sitecenter_status": right.status,
                "sitecenter_conformer_mode": right.conformer_mode,
                "sitecenter_start_position_mode": right.start_position_mode,
                "sitecenter_am1bcc": right.am1bcc,
                "sitecenter_trajectories": right.trajectories,
                "sitecenter_maximum_initial_translation_A": right.maximum_initial_translation_A,
                "sitecenter_runtime_s": right.runtime_s,
                "sitecenter_runtime_source": right.runtime_source,
                "sitecenter_log": right.log,
                "sitecenter_source_data": right.source_data,
                "charge_protocol_matched": left.am1bcc == right.am1bcc,
            })
            for endpoint in METRIC_ALIASES:
                native_value = getattr(left, endpoint)
                site_value = getattr(right, endpoint)
                row[f"sitecenter_{endpoint}"] = site_value
                row[f"delta_sitecenter_minus_native_{endpoint}"] = site_value - native_value
        per_case.append(row)

    paired = [row for row in per_case if row["paired_complete"]]
    native_only_rmsd_rows: list[dict[str, object]] = []
    native_only_threshold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    overall_group = f"all{args.expected_native}"

    # Native-perturbed results are already complete and can be inspected while
    # the site-center counterpart jobs are still running.  Keep these tables
    # descriptive: inferential tests below remain restricted to paired cases.
    for group in (overall_group, "historical12", "pilot3"):
        subset = per_case if group == overall_group else [
            row for row in per_case if row["cohort_group"] == group
        ]
        if not subset:
            continue
        runtimes = np.array(
            [float(row["native_runtime_s"]) for row in subset], dtype=float
        )
        finite_runtimes = runtimes[np.isfinite(runtimes)]
        for endpoint in METRIC_ALIASES:
            values = np.array(
                [float(row[f"native_{endpoint}"]) for row in subset], dtype=float
            )
            native_only_rmsd_rows.append({
                "cohort_group": group,
                "endpoint": endpoint,
                "n": len(values),
                "mean_rmsd_A": float(np.mean(values)),
                "median_rmsd_A": float(np.median(values)),
                "standard_deviation_rmsd_A": float(np.std(values, ddof=1))
                if len(values) > 1 else math.nan,
                "minimum_rmsd_A": float(np.min(values)),
                "maximum_rmsd_A": float(np.max(values)),
                "runtime_n": len(finite_runtimes),
                "mean_runtime_s": float(np.mean(finite_runtimes))
                if len(finite_runtimes) else math.nan,
                "median_runtime_s": float(np.median(finite_runtimes))
                if len(finite_runtimes) else math.nan,
            })
            for cutoff in (2.0, 5.0):
                successes = values < cutoff
                native_only_threshold_rows.append({
                    "cohort_group": group,
                    "endpoint": endpoint,
                    "cutoff_A": cutoff,
                    "n": len(values),
                    "success_count": int(successes.sum()),
                    "success_pct": float(100 * successes.mean()),
                })

    for group in (overall_group, "historical12", "pilot3"):
        subset = paired if group == overall_group else [
            row for row in paired if row["cohort_group"] == group
        ]
        if not subset:
            continue
        for endpoint_index, endpoint in enumerate(METRIC_ALIASES):
            native_values = np.array(
                [float(row[f"native_{endpoint}"]) for row in subset], dtype=float
            )
            site_values = np.array(
                [float(row[f"sitecenter_{endpoint}"]) for row in subset], dtype=float
            )
            delta = site_values - native_values
            two_sided, site_greater = wilcoxon_pvalues(native_values, site_values)
            mean_low, mean_high, median_low, median_high = bootstrap_delta(
                native_values, site_values, args.bootstrap_repeats,
                args.seed + endpoint_index,
            )
            summary_rows.append({
                "cohort_group": group,
                "endpoint": endpoint,
                "paired_n": len(subset),
                "native_median_rmsd_A": float(np.median(native_values)),
                "sitecenter_median_rmsd_A": float(np.median(site_values)),
                "median_delta_sitecenter_minus_native_A": float(np.median(delta)),
                "median_delta_95ci_low_A": median_low,
                "median_delta_95ci_high_A": median_high,
                "mean_delta_sitecenter_minus_native_A": float(np.mean(delta)),
                "mean_delta_95ci_low_A": mean_low,
                "mean_delta_95ci_high_A": mean_high,
                "sitecenter_worse_count": int(np.sum(delta > 0)),
                "native_worse_count": int(np.sum(delta < 0)),
                "tie_count": int(np.sum(np.isclose(delta, 0))),
                "wilcoxon_two_sided_pvalue": two_sided,
                "wilcoxon_sitecenter_greater_pvalue": site_greater,
            })
            for cutoff in (2.0, 5.0):
                native_success = native_values < cutoff
                site_success = site_values < cutoff
                native_only, site_only, pvalue = mcnemar_exact(
                    native_success, site_success
                )
                threshold_rows.append({
                    "cohort_group": group,
                    "endpoint": endpoint,
                    "cutoff_A": cutoff,
                    "paired_n": len(subset),
                    "native_success_count": int(native_success.sum()),
                    "native_success_pct": float(100 * native_success.mean()),
                    "sitecenter_success_count": int(site_success.sum()),
                    "sitecenter_success_pct": float(100 * site_success.mean()),
                    "delta_sitecenter_minus_native_percentage_points": float(
                        100 * (site_success.mean() - native_success.mean())
                    ),
                    "native_only_successes": native_only,
                    "sitecenter_only_successes": site_only,
                    "exact_mcnemar_pvalue": pvalue,
                })

    per_case_path = output / "supplementary_emerald_initialization_per_case.csv"
    native_only_rmsd_path = (
        output / "supplementary_emerald_native_perturbed_rmsd_summary.csv"
    )
    native_only_threshold_path = (
        output / "supplementary_emerald_native_perturbed_success_summary.csv"
    )
    summary_path = output / "supplementary_emerald_initialization_rmsd_summary.csv"
    threshold_path = output / "supplementary_emerald_initialization_success_summary.csv"
    protocol_path = output / "supplementary_emerald_initialization_protocol_audit.csv"
    write_csv(per_case_path, per_case)
    write_csv(native_only_rmsd_path, native_only_rmsd_rows)
    write_csv(native_only_threshold_path, native_only_threshold_rows)
    write_csv(summary_path, summary_rows)
    write_csv(threshold_path, threshold_rows)
    write_csv(
        protocol_path,
        [
            {"analysis_arm": arm, **asdict(record)}
            for arm, selected in (("native_perturbed", native), ("site_center", site))
            for record in selected.values()
            if record.complex_name in set(ordered_ids)
        ],
    )

    charge_matched = sum(bool(row.get("charge_protocol_matched")) for row in paired)
    audit = {
        "analysis_name": "EMERALD privileged-initialization protocol contrast",
        "interpretation": (
            "This is not a single-factor initialization ablation: the arms "
            "differ in conformer source and starting placement, and charge "
            "protocol equality is reported per case."
        ),
        "native_protocol_required": "sdf + native_perturbed",
        "sitecenter_protocol_required": "smiles_3d + site_center",
        "native_completed_n": len(native),
        "native_protocol_ids_by_input_group": native_ids_by_input_group,
        "native_ids_present_in_multiple_input_sources": overlapping_native_ids,
        "explicit_pilot_ids": sorted(pilot_ids),
        "cohort_group_assignment_rule": (
            "pilot3 is assigned by the three explicit pilot case IDs; all "
            "remaining unique native-perturbed cases are historical12. The "
            "selected recovery-aware data source does not define cohort group."
        ),
        "sitecenter_completed_n": len(site),
        "sitecenter_completed_n_within_target15": len(site),
        "sitecenter_completed_n_across_all_scanned_sources": len(site_all),
        "paired_complete_n": len(paired),
        "missing_sitecenter_ids": missing_site,
        "group_counts": dict(group_counts),
        "paired_charge_protocol_matched_n": charge_matched,
        "paired_charge_protocol_mismatched_n": len(paired) - charge_matched,
        "rmsd_success_definition": "strictly less than cutoff",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "nominal_pvalues_unadjusted_for_multiple_endpoints": True,
        "outputs": {
            "per_case": str(per_case_path),
            "native_only_rmsd_summary": str(native_only_rmsd_path),
            "native_only_success_summary": str(native_only_threshold_path),
            "rmsd_summary": str(summary_path),
            "success_summary": str(threshold_path),
            "protocol_audit": str(protocol_path),
        },
    }
    audit_path = output / "supplementary_emerald_initialization_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    for path in (
        per_case_path, native_only_rmsd_path, native_only_threshold_path,
        summary_path, threshold_path, protocol_path,
    ):
        audit.setdefault("sha256", {})[path.name] = sha256(path)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print(json.dumps(audit, indent=2, sort_keys=True))
    print("\nNATIVE-PERTURBED DESCRIPTIVE RMSD SUMMARY")
    for row in native_only_rmsd_rows:
        if row["cohort_group"] == overall_group:
            print(
                f"{row['endpoint']}: n={row['n']} "
                f"mean={row['mean_rmsd_A']:.3f} A, "
                f"median={row['median_rmsd_A']:.3f} A, "
                f"range={row['minimum_rmsd_A']:.3f}-"
                f"{row['maximum_rmsd_A']:.3f} A"
            )
    print("\nNATIVE-PERTURBED DESCRIPTIVE SUCCESS SUMMARY")
    for row in native_only_threshold_rows:
        if row["cohort_group"] == overall_group:
            print(
                f"{row['endpoint']} <{row['cutoff_A']:.0f} A: "
                f"{row['success_count']}/{row['n']} "
                f"({row['success_pct']:.1f}%)"
            )
    print("\nPAIRED RMSD SUMMARY")
    for row in summary_rows:
        if row["cohort_group"] == overall_group:
            print(
                f"{row['endpoint']}: n={row['paired_n']} "
                f"native median={row['native_median_rmsd_A']:.3f} A, "
                f"site-center median={row['sitecenter_median_rmsd_A']:.3f} A, "
                f"delta={row['median_delta_sitecenter_minus_native_A']:.3f} A, "
                f"Wilcoxon(two-sided)={row['wilcoxon_two_sided_pvalue']:.4g}"
            )
    print("\nPAIRED SUCCESS SUMMARY")
    for row in threshold_rows:
        if row["cohort_group"] == overall_group and row["endpoint"] == "top1_rmsd_A":
            print(
                f"Top-1 <{row['cutoff_A']:.0f} A: "
                f"native {row['native_success_count']}/{row['paired_n']} "
                f"({row['native_success_pct']:.1f}%), "
                f"site-center {row['sitecenter_success_count']}/{row['paired_n']} "
                f"({row['sitecenter_success_pct']:.1f}%), "
                f"McNemar={row['exact_mcnemar_pvalue']:.4g}"
            )
    print(f"EMERALD_INITIALIZATION_SUPPLEMENT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
