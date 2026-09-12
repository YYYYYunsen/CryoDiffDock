#!/usr/bin/env python3
"""Freeze formal and technically evaluable Local benchmark cohorts.

This utility deliberately does not read RMSD values or docking outcomes.  It
starts from a canonical chemistry-defined cohort, removes documented chemistry
exclusions to form the formal cohort, and records technical failures without
silently removing them from the formal reporting denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"No IDs found in {path}")
    if len(values) != len(set(values)):
        duplicates = pd.Series(values)[pd.Series(values).duplicated(False)].unique()
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:20].tolist()}")
    return values


def parse_failure(value: str) -> tuple[str, str]:
    complex_name, separator, reason = value.partition("=")
    complex_name = normalize_id(complex_name)
    reason = reason.strip()
    if not separator or not complex_name or not reason:
        raise argparse.ArgumentTypeError(
            "technical failure must be COMPLEX_ID=REASON"
        )
    return complex_name, reason


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-cohort", required=True, type=Path)
    parser.add_argument("--chemistry-exclusions", required=True, type=Path)
    parser.add_argument(
        "--technical-failure",
        action="append",
        default=[],
        type=parse_failure,
        metavar="ID=REASON",
    )
    parser.add_argument("--expected-canonical", type=int)
    parser.add_argument("--expected-formal", type=int)
    parser.add_argument("--expected-evaluable", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    canonical_path = args.canonical_cohort.expanduser().resolve()
    exclusions_path = args.chemistry_exclusions.expanduser().resolve()
    for path in (canonical_path, exclusions_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical = read_ids(canonical_path)
    exclusions = read_ids(exclusions_path)
    canonical_set = set(canonical)
    exclusion_set = set(exclusions)
    unknown_exclusions = sorted(exclusion_set - canonical_set)
    if unknown_exclusions:
        raise ValueError(
            f"Chemistry exclusions are absent from canonical cohort: {unknown_exclusions}"
        )

    failure_pairs = list(args.technical_failure)
    failure_ids = [complex_name for complex_name, _reason in failure_pairs]
    if len(failure_ids) != len(set(failure_ids)):
        raise ValueError("Duplicate --technical-failure complex IDs")
    failure_reasons = dict(failure_pairs)

    formal = [value for value in canonical if value not in exclusion_set]
    formal_set = set(formal)
    unknown_failures = sorted(set(failure_ids) - formal_set)
    if unknown_failures:
        raise ValueError(
            f"Technical failures are absent from formal cohort: {unknown_failures}"
        )
    evaluable = [value for value in formal if value not in failure_reasons]

    expected = {
        "canonical": args.expected_canonical,
        "formal": args.expected_formal,
        "evaluable": args.expected_evaluable,
    }
    observed = {
        "canonical": len(canonical),
        "formal": len(formal),
        "evaluable": len(evaluable),
    }
    mismatches = {
        key: (expected[key], observed[key])
        for key in expected
        if expected[key] is not None and expected[key] != observed[key]
    }
    if mismatches:
        raise ValueError(f"Cohort count mismatch: {mismatches}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    formal_path = output / "formal_cohort_ids.txt"
    evaluable_path = output / "technically_evaluable_ids.txt"
    exclusions_output = output / "chemistry_exclusions.csv"
    failures_output = output / "technical_failures.csv"
    status_output = output / "cohort_status.csv"
    summary_output = output / "summary.json"
    checksums_output = output / "SHA256SUMS"

    write_ids(formal_path, formal)
    write_ids(evaluable_path, evaluable)
    pd.DataFrame({
        "complex_name": exclusions,
        "status": "chemistry_exclusion",
    }).to_csv(exclusions_output, index=False)
    pd.DataFrame([
        {
            "complex_name": complex_name,
            "status": "technical_failure",
            "failure_reason": reason,
            "retained_in_formal_denominator": True,
        }
        for complex_name, reason in failure_pairs
    ], columns=[
        "complex_name", "status", "failure_reason",
        "retained_in_formal_denominator",
    ]).to_csv(failures_output, index=False)

    status_rows = []
    for complex_name in canonical:
        if complex_name in exclusion_set:
            status = "chemistry_exclusion"
            reason = "documented_non_drug_like_or_peptide_like_exclusion"
            in_formal = False
            technically_evaluable = False
        elif complex_name in failure_reasons:
            status = "technical_failure"
            reason = failure_reasons[complex_name]
            in_formal = True
            technically_evaluable = False
        else:
            status = "technically_evaluable"
            reason = ""
            in_formal = True
            technically_evaluable = True
        status_rows.append({
            "complex_name": complex_name,
            "status": status,
            "reason": reason,
            "in_formal_cohort": in_formal,
            "in_technically_evaluable_set": technically_evaluable,
            "counts_as_failure_in_fixed_denominator_analysis": bool(
                in_formal and not technically_evaluable
            ),
        })
    pd.DataFrame(status_rows).to_csv(status_output, index=False)

    summary = {
        "canonical_cohort_n": len(canonical),
        "chemistry_exclusion_n": len(exclusions),
        "formal_cohort_n": len(formal),
        "technical_failure_n": len(failure_pairs),
        "technically_evaluable_n": len(evaluable),
        "formal_primary_denominator": len(formal),
        "available_case_sensitivity_denominator": len(evaluable),
        "primary_reporting_policy": (
            "Retain technical failures in the formal cohort and count them as "
            "failures for fixed-denominator success rates."
        ),
        "sensitivity_reporting_policy": (
            "Report technically evaluable cases separately as an available-case "
            "sensitivity analysis."
        ),
        "uses_rmsd_or_docking_accuracy_for_cohort_definition": False,
        "inputs": {
            "canonical_cohort": str(canonical_path),
            "chemistry_exclusions": str(exclusions_path),
        },
        "input_sha256": {
            "canonical_cohort": sha256(canonical_path),
            "chemistry_exclusions": sha256(exclusions_path),
        },
        "technical_failures": [
            {"complex_name": name, "reason": reason}
            for name, reason in failure_pairs
        ],
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")

    checksum_paths = [
        formal_path, evaluable_path, exclusions_output, failures_output,
        status_output, summary_output,
    ]
    checksums_output.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in checksum_paths
    ))

    print(json.dumps(summary, indent=2))
    print(f"FORMAL COHORT: {formal_path}")
    print(f"TECHNICALLY EVALUABLE SET: {evaluable_path}")
    print(f"STATUS TABLE: {status_output}")
    print(f"LOCAL_TECHNICAL_COHORT_FREEZE_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
