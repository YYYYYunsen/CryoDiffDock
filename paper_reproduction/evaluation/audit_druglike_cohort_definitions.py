#!/usr/bin/env python3
"""Build one authoritative drug-like test cohort and audit derived ID sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ID_COLUMNS = ("pdb_id_upper", "pdb_id", "complex_name", "complex", "name", "id")


def normalize(value: object) -> str:
    return str(value).strip().upper()


def choose(columns, candidates, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; columns={list(columns)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("cohort must be NAME=/path/to/ids.txt-or.csv")
    return name.strip(), Path(path).expanduser().resolve()


def read_id_set(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep=None, engine="python", dtype=str)
        column = choose(table.columns, ID_COLUMNS, "complex ID")
        values = [normalize(value) for value in table[column].dropna()]
    else:
        values = [
            normalize(line.split()[0])
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if any(not value for value in values):
        raise ValueError(f"Empty normalized ID in {path}")
    if len(values) != len(set(values)):
        duplicates = pd.Series(values)[pd.Series(values).duplicated(False)].unique()
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:20].tolist()}")
    return set(values)


def write_ids(path: Path, values: set[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in sorted(values)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemistry-manifest", required=True, type=Path)
    parser.add_argument("--cohort", action="append", default=[], type=parse_named_path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-membership", default="em_test")
    parser.add_argument("--class-column", default="rcsb_class")
    parser.add_argument("--drug-like-label", default="drug_like")
    parser.add_argument("--expected-canonical", type=int, default=None)
    args = parser.parse_args()

    names = [name for name, _path in args.cohort]
    if len(names) != len(set(names)):
        raise ValueError("Cohort names must be unique")
    chemistry_path = args.chemistry_manifest.expanduser().resolve()
    chemistry = pd.read_csv(chemistry_path, dtype=str)
    id_column = choose(chemistry.columns, ID_COLUMNS, "chemistry complex ID")
    for column in (args.class_column, "split_membership"):
        if column not in chemistry.columns:
            raise ValueError(f"Chemistry manifest lacks {column!r}")
    chemistry["_pdb_id"] = chemistry[id_column].map(normalize)
    if chemistry["_pdb_id"].duplicated().any():
        duplicates = chemistry.loc[
            chemistry["_pdb_id"].duplicated(False), "_pdb_id"
        ].unique()
        raise ValueError(f"Chemistry manifest contains duplicate IDs: {duplicates[:20]}")
    canonical = set(
        chemistry.loc[
            chemistry["split_membership"].eq(args.split_membership)
            & chemistry[args.class_column].eq(args.drug_like_label),
            "_pdb_id",
        ]
    )
    if args.expected_canonical is not None and len(canonical) != args.expected_canonical:
        raise RuntimeError(
            f"Canonical cohort changed: {len(canonical)} != {args.expected_canonical}"
        )

    cohorts = {name: read_id_set(path) for name, path in args.cohort}
    universe = set(canonical)
    for values in cohorts.values():
        universe.update(values)
    ledger = pd.DataFrame({"pdb_id": sorted(universe)})
    ledger["in_canonical_drug_like_test"] = ledger["pdb_id"].isin(canonical)
    for name, values in cohorts.items():
        ledger[f"in_{name}"] = ledger["pdb_id"].isin(values)

    output = args.output_dir.expanduser().resolve()
    differences = output / "differences"
    differences.mkdir(parents=True, exist_ok=True)
    canonical_path = output / "canonical_authoritative_drug_like_test_ids.txt"
    write_ids(canonical_path, canonical)
    ledger.to_csv(output / "cohort_membership_ledger.csv", index=False)

    count_rows = [{
        "cohort": "canonical_authoritative_drug_like_test",
        "raw_count": len(canonical),
        "canonical_overlap_count": len(canonical),
        "extra_not_canonical_count": 0,
        "missing_from_canonical_count": 0,
        "is_subset_of_canonical": True,
        "is_equal_to_canonical": True,
        "source": str(chemistry_path),
        "source_sha256": sha256(chemistry_path),
    }]
    for name, path in args.cohort:
        values = cohorts[name]
        extra = values - canonical
        missing = canonical - values
        write_ids(differences / f"{name}__extra_not_canonical.txt", extra)
        write_ids(differences / f"{name}__missing_from_set.txt", missing)
        count_rows.append({
            "cohort": name,
            "raw_count": len(values),
            "canonical_overlap_count": len(values & canonical),
            "extra_not_canonical_count": len(extra),
            "missing_from_canonical_count": len(missing),
            "is_subset_of_canonical": len(extra) == 0,
            "is_equal_to_canonical": values == canonical,
            "source": str(path),
            "source_sha256": sha256(path),
        })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(output / "cohort_counts.csv", index=False)

    pairwise_rows = []
    all_sets = {"canonical_authoritative_drug_like_test": canonical, **cohorts}
    for left_name, left in all_sets.items():
        for right_name, right in all_sets.items():
            pairwise_rows.append({
                "left_cohort": left_name,
                "right_cohort": right_name,
                "intersection_count": len(left & right),
                "left_only_count": len(left - right),
                "right_only_count": len(right - left),
            })
    pd.DataFrame(pairwise_rows).to_csv(output / "pairwise_cohort_counts.csv", index=False)

    summary = {
        "canonical_definition": (
            f"{args.split_membership} AND {args.class_column} == {args.drug_like_label}"
        ),
        "canonical_count": len(canonical),
        "canonical_ids": str(canonical_path),
        "chemistry_manifest": str(chemistry_path),
        "chemistry_manifest_sha256": sha256(chemistry_path),
        "derived_cohorts_are_coverage_sets_not_alternative_sample_definitions": True,
        "extra_policy": "exclude IDs outside canonical cohort",
        "missing_policy": "retain canonical ID; rerun if feasible, otherwise count as failure",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output_files = [
        canonical_path,
        output / "cohort_membership_ledger.csv",
        output / "cohort_counts.csv",
        output / "pairwise_cohort_counts.csv",
        output / "summary.json",
    ]
    with (output / "SHA256SUMS").open("w") as handle:
        for path in output_files:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(counts.to_string(index=False))
    print(f"CANONICAL_DRUGLIKE_COHORT_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
