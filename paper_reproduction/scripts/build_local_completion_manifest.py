#!/usr/bin/env python3
"""Build an auditable inference manifest for missing Local benchmark cases.

Cases already present in the frozen prepared manifest retain the random seed
they would have received in the original run (base seed plus prepared-row
index).  Newly recovered preparation rows use their position in the original
test split.  This makes technical recovery deterministic without consulting
any docking outcome or RMSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
PATH_COLUMNS = ("experimental_protein", "ligand", "ligand_density_crop")


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
        raise ValueError(f"Duplicate IDs found in {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, label: str) -> tuple[pd.DataFrame, str]:
    table = pd.read_csv(path, dtype=str)
    id_column = next((column for column in ID_COLUMNS if column in table.columns), None)
    if id_column is None:
        raise ValueError(f"{path} has no complex-ID column")
    table = table.copy()
    table["_completion_id"] = table[id_column].map(normalize_id)
    duplicates = table.loc[
        table["_completion_id"].duplicated(False), "_completion_id"
    ].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate IDs in {path}: {duplicates[:20].tolist()}")
    table["_completion_source"] = label
    table["_completion_source_row"] = range(len(table))
    return table, id_column


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", required=True, type=Path)
    parser.add_argument("--original-prepared", required=True, type=Path)
    parser.add_argument("--recovered-prepared", required=True, type=Path)
    parser.add_argument("--test-split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-seed", type=int, default=20260813)
    parser.add_argument("--skip-file-validation", action="store_true")
    args = parser.parse_args()

    inputs = {
        "worklist": args.worklist.expanduser().resolve(),
        "original_prepared": args.original_prepared.expanduser().resolve(),
        "recovered_prepared": args.recovered_prepared.expanduser().resolve(),
        "test_split": args.test_split.expanduser().resolve(),
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    requested = read_ids(inputs["worklist"])
    split = read_ids(inputs["test_split"])
    split_index = {complex_name: index for index, complex_name in enumerate(split)}
    missing_from_split = sorted(set(requested) - set(split_index))
    if missing_from_split:
        raise ValueError(f"Worklist cases absent from original test split: {missing_from_split}")

    original, original_id_column = load_manifest(
        inputs["original_prepared"], "original_frozen_prepared"
    )
    recovered, recovered_id_column = load_manifest(
        inputs["recovered_prepared"], "rigid_fallback_recovered_prepared"
    )
    original_lookup = original.set_index("_completion_id", drop=False)
    recovered_lookup = recovered.set_index("_completion_id", drop=False)

    rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    for complex_name in requested:
        if complex_name in original_lookup.index:
            row = original_lookup.loc[complex_name].copy()
            if isinstance(row, pd.DataFrame):
                raise AssertionError(f"Ambiguous original row for {complex_name}")
            source_row = int(row["_completion_source_row"])
            inference_seed = args.base_seed + source_row
            seed_policy = "base_seed_plus_original_prepared_row"
        elif complex_name in recovered_lookup.index:
            row = recovered_lookup.loc[complex_name].copy()
            if isinstance(row, pd.DataFrame):
                raise AssertionError(f"Ambiguous recovered row for {complex_name}")
            source_row = int(row["_completion_source_row"])
            inference_seed = args.base_seed + split_index[complex_name]
            seed_policy = "base_seed_plus_original_test_split_row"
        else:
            raise KeyError(f"No prepared manifest row for {complex_name}")

        row_id_column = (
            original_id_column
            if row["_completion_source"] == "original_frozen_prepared"
            else recovered_id_column
        )
        row["complex_name"] = complex_name
        if row_id_column != "complex_name":
            row[row_id_column] = complex_name
        row["completion_source"] = row["_completion_source"]
        row["inference_seed"] = inference_seed
        row["completion_seed_policy"] = seed_policy
        rows.append(row)
        audit_rows.append({
            "complex_name": complex_name,
            "completion_source": row["_completion_source"],
            "source_row_index": source_row,
            "original_test_split_row_index": split_index[complex_name],
            "inference_seed": inference_seed,
            "seed_policy": seed_policy,
            "flexible_sidechains": row.get("flexible_sidechains", ""),
            "sidechain_policy": row.get("sidechain_policy", ""),
        })

    output = pd.DataFrame(rows).drop(
        columns=["_completion_id", "_completion_source", "_completion_source_row"],
        errors="ignore",
    )
    if output["complex_name"].tolist() != requested:
        raise AssertionError("Output row order differs from the worklist")
    if output["complex_name"].duplicated().any():
        raise AssertionError("Output contains duplicate complex names")
    required_columns = {"complex_name", "flexible_sidechains", *PATH_COLUMNS}
    missing_columns = sorted(required_columns - set(output.columns))
    if missing_columns:
        raise ValueError(f"Combined manifest lacks required columns: {missing_columns}")

    if not args.skip_file_validation:
        missing_files: list[str] = []
        for _, row in output.iterrows():
            for column in PATH_COLUMNS:
                path = Path(str(row[column])).expanduser()
                if not path.is_file():
                    missing_files.append(f"{row['complex_name']}:{column}:{path}")
        if missing_files:
            raise FileNotFoundError(
                "Prepared manifest contains missing files:\n" + "\n".join(missing_files[:30])
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reveal_path = output_dir / "inference_reveal.csv"
    blind_path = output_dir / "inference_blind.csv"
    audit_path = output_dir / "completion_manifest_audit.csv"
    summary_path = output_dir / "summary.json"
    output.to_csv(reveal_path, index=False)
    output.drop(columns=["reference_ligand"], errors="ignore").to_csv(
        blind_path, index=False
    )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    source_counts = pd.Series(
        [row["completion_source"] for row in audit_rows]
    ).value_counts().to_dict()
    summary = {
        "requested_n": len(requested),
        "output_n": len(output),
        "base_seed": args.base_seed,
        "uses_docking_outcomes": False,
        "uses_rmsd": False,
        "row_order": "worklist order",
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "seed_policy": {
            "original_prepared_rows": "base seed plus frozen prepared-row index",
            "newly_recovered_rows": "base seed plus original test-split row index",
        },
        "inputs": {name: str(path) for name, path in inputs.items()},
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"LOCAL_COMPLETION_MANIFEST_OK: {blind_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
