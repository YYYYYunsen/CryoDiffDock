#!/usr/bin/env python3
"""Prepare auditable EMERALD site-center recovery worklists.

The deployment worklist contains two outcome-independent groups:

* completed historical cases whose selected run explicitly used the
  privileged ``sdf + native_perturbed`` protocol and therefore need a paired
  ``smiles_3d + site_center`` counterpart;
* cases whose site-center docking was interrupted or killed by resource
  contention and can be rerun without changing molecular inputs.

Cases requiring receptor edits are held behind a separate compatibility
worklist.  Unsafe ligand reconstruction failures are retained as failures.
No RMSD or docking score is read when constructing any worklist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


RESOURCE_OR_INTERRUPTED = (
    "6O9G",
    "9C6R",
    "9PND",
    "9Z6U",
    "9Z6W",
    "9ZRQ",
)
INPUT_COMPATIBILITY = ("8KDO", "9NR7")
RETAINED_FAILURES = ("8Y2G",)

CONFORMER_PATTERN = re.compile(r"input conformer mode:\s*([^\s;]+)")
START_PATTERN = re.compile(r"start position mode:\s*([^\s;]+)")
CHARGE_PATTERN = re.compile(r"AM1-BCC:\s*([^\s;]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path, expected_n: int) -> list[str]:
    values = [
        line.split("#", 1)[0].strip().upper()
        for line in path.read_text().splitlines()
    ]
    values = [value for value in values if value]
    if len(values) != expected_n or len(set(values)) != expected_n:
        raise ValueError(
            f"expected {expected_n} unique authoritative IDs in {path}, "
            f"found {len(values)}"
        )
    return values


def read_table(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"table has no header: {path}")
        return list(reader)


def normalized_id(row: dict[str, str]) -> str:
    for column in ("pdb_id", "complex_name", "case_id", "id"):
        value = str(row.get(column, "")).strip().upper()
        if value:
            return value
    return ""


def protocol_from_log(path: Path) -> tuple[str, str, str]:
    text = path.read_text(errors="replace")
    conformer = CONFORMER_PATTERN.search(text)
    start = START_PATTERN.search(text)
    charge = CHARGE_PATTERN.search(text)
    return (
        conformer.group(1) if conformer else "unrecorded",
        start.group(1) if start else "unrecorded",
        charge.group(1) if charge else "unrecorded",
    )


def selected_log(row: dict[str, str]) -> Path | None:
    log_text = str(row.get("log", "")).strip()
    if log_text:
        return Path(log_text).expanduser()
    workdir = str(row.get("workdir", "")).strip()
    return Path(workdir).expanduser() / "driver.log" if workdir else None


def write_ids(path: Path, preferred_order: list[str], selected: set[str]) -> None:
    ordered = [case_id for case_id in preferred_order if case_id in selected]
    if len(ordered) != len(selected):
        missing = sorted(selected - set(ordered))
        raise ValueError(f"worklist IDs absent from the prepared ID universe: {missing}")
    path.write_text("".join(f"{case_id}\n" for case_id in ordered))


def build_recovery_manifest(
    authoritative_rows: list[dict[str, str]],
    historical_rows: list[dict[str, str]],
    ordered_ids: list[str],
    output: Path,
) -> None:
    fields: list[str] = []
    for rows in (authoritative_rows, historical_rows):
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    if "complex_name" not in fields:
        fields.insert(0, "complex_name")

    historical = {
        normalized_id(row): dict(row) for row in historical_rows if normalized_id(row)
    }
    authoritative = {
        normalized_id(row): dict(row) for row in authoritative_rows if normalized_id(row)
    }
    selected_rows: list[dict[str, str]] = []
    required = (
        "experimental_protein", "ligand", "reference_ligand", "map_path",
        "mask_structure", "pocket_center_x", "pocket_center_y", "pocket_center_z",
    )
    for case_id in ordered_ids:
        row = dict(historical.get(case_id, {}))
        row.update(authoritative.get(case_id, {}))
        if not row:
            raise ValueError(f"no manifest row for recovery case {case_id}")
        row["complex_name"] = case_id
        missing = [field for field in required if not str(row.get(field, "")).strip()]
        if missing:
            raise ValueError(f"recovery manifest row {case_id} lacks {missing}")
        selected_rows.append(row)

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-ids", required=True, type=Path)
    parser.add_argument("--authoritative-manifest", required=True, type=Path)
    parser.add_argument("--historical-manifest", required=True, type=Path)
    parser.add_argument("--historical-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--expected-near-native", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ids_path = args.authoritative_ids.expanduser().resolve()
    manifest_path = args.authoritative_manifest.expanduser().resolve()
    historical_manifest_path = args.historical_manifest.expanduser().resolve()
    summary_path = args.historical_summary.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (ids_path, manifest_path, historical_manifest_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    authoritative = read_ids(ids_path, args.expected_n)
    authoritative_set = set(authoritative)
    manifest_rows = read_table(manifest_path, ",")
    manifest_ids = {normalized_id(row) for row in manifest_rows}
    missing_manifest = sorted(authoritative_set - manifest_ids)
    if missing_manifest:
        raise ValueError(f"authoritative manifest is missing IDs: {missing_manifest}")
    historical_manifest_rows = read_table(historical_manifest_path, ",")

    summary_rows = read_table(summary_path, "\t")
    near_native: list[str] = []
    protocol_audit: list[dict[str, str]] = []
    for row in summary_rows:
        if str(row.get("status", "")).strip().lower() != "completed":
            continue
        case_id = normalized_id(row)
        log = selected_log(row)
        if log is None or not log.is_file():
            protocol_audit.append({
                "complex_name": case_id,
                "conformer_mode": "missing_log",
                "start_position_mode": "missing_log",
                "am1bcc": "missing_log",
                "log": "" if log is None else str(log),
            })
            continue
        conformer, start, charge = protocol_from_log(log)
        protocol_audit.append({
            "complex_name": case_id,
            "conformer_mode": conformer,
            "start_position_mode": start,
            "am1bcc": charge,
            "log": str(log.resolve()),
        })
        if conformer == "sdf" and start == "native_perturbed":
            near_native.append(case_id)

    if len(near_native) != args.expected_near_native:
        raise ValueError(
            f"expected {args.expected_near_native} explicit sdf/native_perturbed "
            f"completed cases, found {len(near_native)}: {sorted(near_native)}"
        )
    if len(near_native) != len(set(near_native)):
        raise ValueError("historical summary selected duplicate near-native cases")

    standard = set(near_native) | set(RESOURCE_OR_INTERRUPTED)
    compatibility = set(INPUT_COMPATIBILITY)
    retained = set(RETAINED_FAILURES)
    groups = {
        "standard": standard,
        "input_compatibility": compatibility,
        "retained_failure": retained,
    }
    names = list(groups)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise ValueError(f"overlap between {left}/{right}: {sorted(overlap)}")

    output.mkdir(parents=True, exist_ok=True)
    near_native_order = [case_id for case_id in near_native]
    recovery_order = authoritative + [
        case_id for case_id in near_native_order if case_id not in authoritative_set
    ]
    near_path = output / "near_native15_ids.txt"
    standard_path = output / "sitecenter_standard_ids.txt"
    compatibility_path = output / "sitecenter_input_compat_ids.txt"
    retained_path = output / "sitecenter_retained_failure_ids.txt"
    all_safe_path = output / "sitecenter_all_safe_attempt_ids.txt"
    write_ids(near_path, recovery_order, set(near_native))
    write_ids(standard_path, recovery_order, standard)
    write_ids(compatibility_path, recovery_order, compatibility)
    write_ids(retained_path, recovery_order, retained)
    write_ids(all_safe_path, recovery_order, standard | compatibility)
    recovery_manifest_path = output / "sitecenter_recovery_manifest.csv"
    build_recovery_manifest(
        manifest_rows,
        historical_manifest_rows,
        [case_id for case_id in recovery_order if case_id in standard | compatibility],
        recovery_manifest_path,
    )

    audit_fields = (
        "complex_name", "conformer_mode", "start_position_mode", "am1bcc", "log"
    )
    protocol_path = output / "historical_selected_protocol_audit.csv"
    with protocol_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(protocol_audit)

    audit = {
        "authoritative_n": len(authoritative),
        "near_native_completed_n": len(near_native),
        "near_native_completed_ids": [
            case_id for case_id in recovery_order if case_id in set(near_native)
        ],
        "near_native_in_authoritative_n": len(set(near_native) & authoritative_set),
        "near_native_outside_authoritative_ids": [
            case_id for case_id in near_native if case_id not in authoritative_set
        ],
        "standard_sitecenter_submission_n": len(standard),
        "standard_sitecenter_submission_ids": [
            case_id for case_id in recovery_order if case_id in standard
        ],
        "resource_or_interrupted_ids": list(RESOURCE_OR_INTERRUPTED),
        "input_compatibility_gate_ids": list(INPUT_COMPATIBILITY),
        "retained_failure_ids": list(RETAINED_FAILURES),
        "main_protocol": {
            "input_conformer_mode": "smiles_3d",
            "start_position_mode": "site_center",
            "am1bcc": 0,
            "trajectories": 3,
            "random_seed": 2026,
        },
        "uses_rmsd_or_docking_outcome_for_selection": False,
        "authoritative_ids_sha256": sha256(ids_path),
        "authoritative_manifest_sha256": sha256(manifest_path),
        "historical_manifest_sha256": sha256(historical_manifest_path),
        "sitecenter_recovery_manifest_sha256": sha256(recovery_manifest_path),
        "historical_summary_sha256": sha256(summary_path),
        "protocol_audit_sha256": sha256(protocol_path),
    }
    audit_path = output / "sitecenter_recovery_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"EMERALD_SITECENTER_RECOVERY_PREPARED: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
