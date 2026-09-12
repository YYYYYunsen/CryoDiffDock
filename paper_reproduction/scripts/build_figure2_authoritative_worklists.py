#!/usr/bin/env python3
"""Freeze Figure 2 work lists and enrich the prepared local-docking manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex_id")


def read_ids(path: Path) -> list[str]:
    values = [
        line.split("#", 1)[0].strip().upper()
        for line in path.read_text().splitlines()
    ]
    values = [value for value in values if value]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID list is empty or contains duplicates: {path}")
    return values


def normalize_table(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str)
    id_column = next((column for column in ID_COLUMNS if column in table.columns), None)
    if id_column is None:
        raise ValueError(f"{label} has no recognized ID column: {list(table.columns)}")
    table = table.copy()
    table["complex_name"] = table[id_column].str.strip().str.upper()
    if table["complex_name"].isna().any() or table["complex_name"].duplicated().any():
        raise ValueError(f"{label} contains blank or duplicate IDs")
    return table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_ids(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-ids", required=True, type=Path)
    parser.add_argument("--authoritative-cohort", required=True, type=Path)
    parser.add_argument("--previous-ids", required=True, type=Path)
    parser.add_argument(
        "--previous-chemem-cohort", required=True, type=Path,
        help="Historical ChemEM cohort; attempted failures are reusable attempts.",
    )
    parser.add_argument("--prepared-manifest", required=True, type=Path)
    parser.add_argument(
        "--resolution-table", required=True, type=Path,
        help="Outcome-independent metadata table containing resolution_A.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-authoritative-n", type=int, default=174)
    parser.add_argument("--expected-previous-n", type=int, default=99)
    parser.add_argument("--expected-reusable-n", type=int, default=94)
    parser.add_argument("--expected-new-n", type=int, default=80)
    parser.add_argument("--expected-excluded-previous-n", type=int, default=5)
    parser.add_argument("--expected-chemem-reusable-n", type=int, default=95)
    parser.add_argument("--expected-chemem-new-n", type=int, default=79)
    args = parser.parse_args()

    authoritative_path = args.authoritative_ids.expanduser().resolve()
    cohort_path = args.authoritative_cohort.expanduser().resolve()
    previous_path = args.previous_ids.expanduser().resolve()
    previous_chemem_path = args.previous_chemem_cohort.expanduser().resolve()
    manifest_path = args.prepared_manifest.expanduser().resolve()
    resolution_path = args.resolution_table.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    authoritative = read_ids(authoritative_path)
    previous = read_ids(previous_path)
    if len(authoritative) != args.expected_authoritative_n:
        raise ValueError(
            f"Expected {args.expected_authoritative_n} authoritative IDs, "
            f"found {len(authoritative)}"
        )
    if len(previous) != args.expected_previous_n:
        raise ValueError(
            f"Expected {args.expected_previous_n} previous IDs, found {len(previous)}"
        )

    authoritative_set = set(authoritative)
    previous_set = set(previous)
    reusable = [value for value in authoritative if value in previous_set]
    expansion = [value for value in authoritative if value not in previous_set]
    excluded_previous = [value for value in previous if value not in authoritative_set]
    observed_counts = (len(reusable), len(expansion), len(excluded_previous))
    expected_counts = (
        args.expected_reusable_n,
        args.expected_new_n,
        args.expected_excluded_previous_n,
    )
    if observed_counts != expected_counts:
        raise ValueError(
            "Unexpected authoritative/previous overlap: "
            f"reusable,new,excluded_previous={observed_counts}; expected={expected_counts}"
        )

    previous_chemem = normalize_table(previous_chemem_path, "historical ChemEM cohort")
    previous_chemem_ids = set(previous_chemem["complex_name"])
    chemem_reusable = [value for value in authoritative if value in previous_chemem_ids]
    chemem_expansion = [value for value in authoritative if value not in previous_chemem_ids]
    if (
        len(chemem_reusable) != args.expected_chemem_reusable_n
        or len(chemem_expansion) != args.expected_chemem_new_n
    ):
        raise ValueError(
            "Unexpected authoritative/ChemEM overlap: "
            f"reusable,new={(len(chemem_reusable), len(chemem_expansion))}; "
            f"expected={(args.expected_chemem_reusable_n, args.expected_chemem_new_n)}"
        )

    cohort = normalize_table(cohort_path, "authoritative cohort")
    manifest = normalize_table(manifest_path, "prepared manifest")
    resolution = normalize_table(resolution_path, "resolution table")
    if set(authoritative) != set(cohort["complex_name"]):
        raise ValueError("Authoritative IDs and cohort CSV membership differ")
    if set(authoritative) != set(manifest["complex_name"]):
        missing = sorted(authoritative_set - set(manifest["complex_name"]))
        extra = sorted(set(manifest["complex_name"]) - authoritative_set)
        raise ValueError(f"Prepared manifest mismatch: missing={missing}, extra={extra}")
    resolution = resolution[resolution["complex_name"].isin(authoritative_set)].copy()
    if "resolution_A" not in resolution:
        raise ValueError("Resolution table lacks resolution_A")
    resolution["resolution_A"] = pd.to_numeric(resolution["resolution_A"], errors="coerce")
    if resolution["complex_name"].duplicated().any():
        conflicts = resolution.groupby("complex_name")["resolution_A"].nunique(dropna=False)
        if (conflicts > 1).any():
            raise ValueError("Resolution table has conflicting duplicate PDB rows")
        resolution = resolution.drop_duplicates("complex_name")
    if set(resolution["complex_name"]) != authoritative_set:
        raise ValueError("Resolution table does not cover the complete authoritative cohort")

    order = pd.DataFrame({
        "authoritative_index": range(len(authoritative)),
        "complex_name": authoritative,
    })
    cohort_fields = [
        column for column in (
            "complex_name", "resolution_A", "resolution_stratum", "split_membership",
            "n_heavy_atoms", "n_rotatable_bonds", "has_metal",
        ) if column in cohort.columns
    ]
    merged = order.merge(
        manifest.drop(columns=[c for c in ID_COLUMNS if c != "complex_name" and c in manifest]),
        on="complex_name", how="left", validate="one_to_one", sort=False,
    ).merge(
        cohort[cohort_fields].drop_duplicates("complex_name"),
        on="complex_name", how="left", validate="one_to_one", sort=False,
        suffixes=("", "_cohort"),
    ).merge(
        resolution[["complex_name", "resolution_A"]],
        on="complex_name", how="left", validate="one_to_one", sort=False,
        suffixes=("", "_metadata"),
    )
    if "resolution_A_cohort" in merged:
        if "resolution_A" not in merged:
            merged["resolution_A"] = merged["resolution_A_cohort"]
        else:
            merged["resolution_A"] = merged["resolution_A"].fillna(
                merged["resolution_A_cohort"]
            )
        merged = merged.drop(columns="resolution_A_cohort")
    if "resolution_A_metadata" in merged:
        if "resolution_A" not in merged:
            merged["resolution_A"] = merged["resolution_A_metadata"]
        else:
            merged["resolution_A"] = merged["resolution_A"].fillna(
                merged["resolution_A_metadata"]
            )
        merged = merged.drop(columns="resolution_A_metadata")
    required = {
        "experimental_protein", "ligand", "reference_ligand", "map_path",
        "mask_structure", "pocket_center_x", "pocket_center_y", "pocket_center_z",
        "resolution_A",
    }
    missing_columns = sorted(required - set(merged.columns))
    if missing_columns:
        raise ValueError(f"Enriched manifest lacks required columns: {missing_columns}")
    if merged[list(required)].isna().any().any():
        bad = merged.loc[merged[list(required)].isna().any(axis=1), "complex_name"].tolist()
        raise ValueError(f"Required manifest values are blank for: {bad}")

    merged.to_csv(output / "authoritative174_manifest.csv", index=False)
    merged.drop(columns=["reference_ligand"]).to_csv(
        output / "authoritative174_manifest_blind.csv", index=False
    )
    write_ids(output / "authoritative174_ids.txt", authoritative)
    write_ids(output / "reusable_from_frozen99_94_ids.txt", reusable)
    write_ids(output / "run_new80_ids.txt", expansion)
    write_ids(output / "run_emerald_new80_ids.txt", expansion)
    write_ids(output / "chemem_reusable95_ids.txt", chemem_reusable)
    write_ids(output / "run_chemem_new79_ids.txt", chemem_expansion)
    write_ids(output / "excluded_from_frozen99_5_ids.txt", excluded_previous)

    audit = {
        "authoritative_n": len(authoritative),
        "previous_n": len(previous),
        "reusable_previous_n": len(reusable),
        "new_work_n": len(expansion),
        "excluded_previous_n": len(excluded_previous),
        "reusable_previous_ids": reusable,
        "new_work_ids": expansion,
        "excluded_previous_ids": excluded_previous,
        "chemem_historical_cohort_n": len(previous_chemem_ids),
        "chemem_reusable_attempt_n": len(chemem_reusable),
        "chemem_new_work_n": len(chemem_expansion),
        "chemem_reusable_attempt_ids": chemem_reusable,
        "chemem_new_work_ids": chemem_expansion,
        "outcome_independent_selection": True,
        "input_sha256": {
            "authoritative_ids": sha256(authoritative_path),
            "authoritative_cohort": sha256(cohort_path),
            "previous_ids": sha256(previous_path),
            "previous_chemem_cohort": sha256(previous_chemem_path),
            "prepared_manifest": sha256(manifest_path),
            "resolution_table": sha256(resolution_path),
        },
    }
    (output / "worklist_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"FIGURE2_AUTHORITATIVE_WORKLISTS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
