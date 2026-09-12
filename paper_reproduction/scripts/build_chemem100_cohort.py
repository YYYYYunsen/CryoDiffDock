#!/usr/bin/env python3
"""Build a frozen ChemEM cohort without reading docking outcomes.

The intended paper workflow keeps the already attempted ChemEM cases fixed and
adds chemistry-only cases until a requested total is reached.  Selection is
balanced over resolution, ligand size, and ligand flexibility.  Reference RMSD,
docking scores, and ChemEM completion status are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


ID_COLUMNS = ("pdb_id_upper", "pdb_id", "complex_name", "complex", "name")
RESOLUTION_COLUMNS = ("resolution_A", "resolution", "map_resolution_A")
RESOLUTION_LABELS = ("higher_resolution", "middle_resolution", "lower_resolution")


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def choose_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column found; columns={list(columns)}")


def read_id_file(path: Path) -> list[str]:
    values = [normalize_id(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank_bins(
    frame: pd.DataFrame,
    value_column: str,
    labels: tuple[str, ...],
) -> pd.Series:
    """Assign deterministic, near-equal rank bins with ties broken by PDB ID."""
    ordered = frame[[value_column]].copy()
    ordered["_complex_sort"] = frame["complex_name"].astype(str).to_numpy()
    ordered = ordered.sort_values(
        [value_column, "_complex_sort"], kind="mergesort"
    )
    result = pd.Series(index=frame.index, dtype=object)
    n = len(ordered)
    for rank, index in enumerate(ordered.index):
        bin_index = min(len(labels) - 1, (rank * len(labels)) // n)
        result.loc[index] = labels[bin_index]
    return result


def bounded_largest_remainder(
    total: int,
    sizes: dict[object, int],
    lower_bounds: Optional[dict[object, int]] = None,
) -> dict[object, int]:
    """Allocate integer targets proportional to sizes under lower/upper bounds."""
    keys = list(sizes)
    lower = {key: int((lower_bounds or {}).get(key, 0)) for key in keys}
    if any(lower[key] > sizes[key] for key in keys):
        raise ValueError("A lower bound exceeds the available stratum size")
    if sum(lower.values()) > total or total > sum(sizes.values()):
        raise ValueError("Requested total is incompatible with allocation bounds")
    size_sum = float(sum(sizes.values()))
    desired = {key: total * sizes[key] / size_sum for key in keys}
    allocation = dict(lower)
    while sum(allocation.values()) < total:
        available = [key for key in keys if allocation[key] < sizes[key]]
        if not available:
            raise RuntimeError("No capacity remains while allocating cohort targets")
        key = max(
            available,
            key=lambda item: (
                desired[item] - allocation[item],
                sizes[item] - allocation[item],
                str(item),
            ),
        )
        allocation[key] += 1
    return allocation


def numeric_values(value: object) -> list[float]:
    values = value if isinstance(value, (list, tuple)) else [value]
    parsed = []
    for item in values:
        token = str(item).strip().strip("'\"")
        if token in {"", ".", "?"}:
            continue
        try:
            number = float(token)
        except ValueError:
            continue
        if math.isfinite(number) and number > 0:
            parsed.append(number)
    return parsed


def resolution_from_cif(path: Path) -> float:
    try:
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    except ImportError as exc:
        raise RuntimeError(
            "Biopython is required to read resolution from mmCIF. Alternatively, "
            "provide --resolution-table."
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    cif = MMCIF2Dict(str(path))
    for key in (
        "_em_3d_reconstruction.resolution",
        "_refine.ls_d_res_high",
        "_reflns.d_resolution_high",
    ):
        values = numeric_values(cif.get(key, []))
        if values:
            return min(values)
    raise ValueError(f"No experimental resolution found in {path}")


def load_resolution_table(path: Path) -> pd.Series:
    table = pd.read_csv(path, dtype=str)
    id_column = choose_column(table.columns, ID_COLUMNS, "complex ID")
    resolution_column = choose_column(table.columns, RESOLUTION_COLUMNS, "resolution")
    table["complex_name"] = table[id_column].map(normalize_id)
    table["resolution_A"] = pd.to_numeric(table[resolution_column], errors="coerce")
    if table["complex_name"].duplicated().any():
        raise ValueError(f"Duplicate complex IDs in {path}")
    return table.set_index("complex_name")["resolution_A"]


def load_optional_ids(path: Optional[Path]) -> set[str]:
    return set(read_id_file(path)) if path is not None else set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-manifest", required=True, type=Path)
    parser.add_argument("--local-metrics", required=True, type=Path)
    parser.add_argument("--chemistry-manifest", required=True, type=Path)
    parser.add_argument("--existing-cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution-table", type=Path)
    parser.add_argument("--global-available-cohort", type=Path)
    parser.add_argument("--legacy-local-cohort", type=Path)
    parser.add_argument("--original-cohort", type=Path)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--split-membership", default="em_test")
    parser.add_argument("--class-column", default="rcsb_class")
    parser.add_argument("--drug-like-label", default="drug_like")
    parser.add_argument("--expected-test-drug-like", type=int)
    parser.add_argument("--expected-local-drug-like", type=int)
    args = parser.parse_args()

    manifest = pd.read_csv(args.local_manifest, dtype=str)
    manifest_id = choose_column(manifest.columns, ID_COLUMNS, "manifest complex ID")
    manifest["complex_name"] = manifest[manifest_id].map(normalize_id)
    if manifest["complex_name"].duplicated().any():
        raise ValueError("Local manifest contains duplicate complex IDs")

    metrics = pd.read_csv(args.local_metrics, dtype=str)
    metrics_id = choose_column(metrics.columns, ID_COLUMNS, "metrics complex ID")
    local_evaluable = set(metrics[metrics_id].map(normalize_id))

    chemistry = pd.read_csv(args.chemistry_manifest, dtype=str)
    chemistry_id = choose_column(chemistry.columns, ID_COLUMNS, "chemistry complex ID")
    chemistry["complex_name"] = chemistry[chemistry_id].map(normalize_id)
    if chemistry["complex_name"].duplicated().any():
        duplicated = chemistry.loc[
            chemistry["complex_name"].duplicated(False), "complex_name"
        ].unique()
        raise ValueError(f"Chemistry manifest has duplicate IDs: {duplicated[:20]}")
    if args.class_column not in chemistry.columns:
        raise ValueError(f"Missing chemistry class column {args.class_column!r}")
    if "split_membership" not in chemistry.columns:
        raise ValueError("Chemistry manifest lacks split_membership")

    test_drug_like = set(
        chemistry.loc[
            chemistry["split_membership"].eq(args.split_membership)
            & chemistry[args.class_column].eq(args.drug_like_label),
            "complex_name",
        ]
    )
    authoritative_local = test_drug_like & local_evaluable
    if args.expected_test_drug_like is not None and len(test_drug_like) != args.expected_test_drug_like:
        raise ValueError(
            f"Authoritative test drug-like count changed: {len(test_drug_like)} "
            f"!= {args.expected_test_drug_like}"
        )
    if args.expected_local_drug_like is not None and len(authoritative_local) != args.expected_local_drug_like:
        raise ValueError(
            f"Authoritative local drug-like count changed: {len(authoritative_local)} "
            f"!= {args.expected_local_drug_like}"
        )

    excluded = {normalize_id(value) for value in args.exclude_id}
    eligible_ids = authoritative_local - excluded
    existing_order = read_id_file(args.existing_cohort)
    existing = set(existing_order)
    if not existing <= eligible_ids:
        raise ValueError(
            "Existing cohort contains ineligible IDs: "
            f"{sorted(existing - eligible_ids)}"
        )
    if len(existing) > args.target_size:
        raise ValueError("Existing cohort is larger than target size")
    if args.target_size > len(eligible_ids):
        raise ValueError("Target size exceeds eligible cohort")

    manifest_indexed = manifest.set_index("complex_name")
    missing_manifest = sorted(eligible_ids - set(manifest_indexed.index))
    if missing_manifest:
        raise ValueError(f"Eligible IDs absent from local manifest: {missing_manifest}")

    chem_indexed = chemistry.set_index("complex_name")
    feature = manifest_indexed.loc[sorted(eligible_ids)].copy()
    feature.index.name = None
    feature["complex_name"] = feature.index
    for source, destination in (
        ("n_heavy_atoms", "ligand_heavy_atoms"),
        ("n_rotatable_bonds", "ligand_rotatable_bonds"),
        ("rcsb_matched_ccd_id", "ccd_id"),
        ("rcsb_matched_name", "ligand_name"),
        ("rcsb_matched_inchikey", "inchikey"),
        (args.class_column, "authoritative_class"),
    ):
        if source in chem_indexed.columns:
            feature[destination] = chem_indexed.loc[feature.index, source]
    for column in ("ligand_heavy_atoms", "ligand_rotatable_bonds"):
        feature[column] = pd.to_numeric(feature[column], errors="coerce")
        if feature[column].isna().any():
            missing = feature.loc[feature[column].isna(), "complex_name"].tolist()
            raise ValueError(f"Missing {column} for {missing}")

    if args.resolution_table is not None:
        resolution = load_resolution_table(args.resolution_table)
        feature["resolution_A"] = resolution.reindex(feature.index)
    else:
        if "mask_structure" not in feature.columns:
            raise ValueError("Local manifest lacks mask_structure for resolution parsing")
        values = []
        for complex_name, path in feature["mask_structure"].items():
            try:
                values.append(resolution_from_cif(Path(path)))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to determine resolution for {complex_name}: {exc}"
                ) from exc
        feature["resolution_A"] = values
    if feature["resolution_A"].isna().any():
        missing = feature.loc[feature["resolution_A"].isna(), "complex_name"].tolist()
        raise ValueError(f"Missing resolution for {missing}")

    feature["resolution_stratum"] = stable_rank_bins(
        feature, "resolution_A", RESOLUTION_LABELS
    )
    feature["heavy_atom_stratum"] = stable_rank_bins(
        feature, "ligand_heavy_atoms", ("small", "medium", "large")
    )
    feature["rotatable_bond_stratum"] = stable_rank_bins(
        feature, "ligand_rotatable_bonds", ("low", "medium", "high")
    )
    feature["complexity_cell"] = (
        feature["heavy_atom_stratum"] + "__" + feature["rotatable_bond_stratum"]
    )
    feature["existing_case"] = feature["complex_name"].isin(existing)

    resolution_sizes = feature["resolution_stratum"].value_counts().to_dict()
    existing_resolution = (
        feature.loc[feature["existing_case"], "resolution_stratum"].value_counts().to_dict()
    )
    resolution_targets = bounded_largest_remainder(
        args.target_size, resolution_sizes, existing_resolution
    )

    rng = np.random.default_rng(args.seed)
    feature["selection_random_key"] = rng.random(len(feature))
    selected_additions: list[str] = []
    for resolution_label in RESOLUTION_LABELS:
        group = feature[feature["resolution_stratum"].eq(resolution_label)]
        target = resolution_targets[resolution_label]
        existing_group = group[group["existing_case"]]
        cell_sizes = group["complexity_cell"].value_counts().to_dict()
        lower = existing_group["complexity_cell"].value_counts().to_dict()
        cell_targets = bounded_largest_remainder(target, cell_sizes, lower)
        for cell, cell_target in cell_targets.items():
            already = int(lower.get(cell, 0))
            need = cell_target - already
            if need <= 0:
                continue
            candidates = group[
                group["complexity_cell"].eq(cell) & (~group["existing_case"])
            ].sort_values(["selection_random_key", "complex_name"])
            if len(candidates) < need:
                raise RuntimeError(f"Insufficient candidates in {resolution_label}/{cell}")
            selected_additions.extend(candidates.head(need)["complex_name"].tolist())

    if len(existing) + len(selected_additions) != args.target_size:
        raise RuntimeError("Balanced selection did not produce the requested cohort size")
    if len(selected_additions) != len(set(selected_additions)):
        raise RuntimeError("Duplicate additions were selected")
    selected = existing | set(selected_additions)

    additions = feature.loc[selected_additions].sort_values(
        ["resolution_stratum", "resolution_A", "complex_name"]
    )
    combined_order = existing_order + additions["complex_name"].tolist()
    cohort = feature.loc[combined_order].copy()
    cohort["selection_phase"] = np.where(
        cohort["complex_name"].isin(existing), "existing_58", "confirmatory_addition_42"
    )
    cohort.insert(0, "chemem_index", range(len(cohort)))

    global_available = load_optional_ids(args.global_available_cohort)
    legacy_local = load_optional_ids(args.legacy_local_cohort)
    original = load_optional_ids(args.original_cohort)
    membership = chemistry.loc[
        chemistry["complex_name"].isin(test_drug_like),
        [
            column
            for column in (
                "complex_name",
                args.class_column,
                "rcsb_matched_ccd_id",
                "rcsb_matched_name",
                "n_heavy_atoms",
                "n_rotatable_bonds",
            )
            if column in chemistry.columns
        ],
    ].copy()
    membership["local_evaluable"] = membership["complex_name"].isin(local_evaluable)
    membership["global_available"] = membership["complex_name"].isin(global_available)
    membership["legacy_local_drug_like"] = membership["complex_name"].isin(legacy_local)
    membership["known_chemistry_exclusion"] = membership["complex_name"].isin(excluded)
    membership["original_chemem_cohort"] = membership["complex_name"].isin(original)
    membership["existing_chemem58"] = membership["complex_name"].isin(existing)
    membership["selected_chemem100"] = membership["complex_name"].isin(selected)
    membership["selection_phase"] = np.select(
        [
            membership["existing_chemem58"],
            membership["selected_chemem100"],
            membership["known_chemistry_exclusion"],
            ~membership["local_evaluable"],
        ],
        [
            "existing_58",
            "confirmatory_addition_42",
            "chemistry_exclusion",
            "local_technical_exclusion",
        ],
        default="eligible_not_selected",
    )
    membership = membership.sort_values("complex_name")

    counts = [
        ("authoritative_test_drug_like", len(test_drug_like)),
        ("local_evaluable_all_classes", len(local_evaluable)),
        ("authoritative_local_drug_like", len(authoritative_local)),
        ("legacy_local_drug_like", len(legacy_local)),
        ("authoritative_global_available_drug_like", len(test_drug_like & global_available)),
        ("authoritative_global_local_common", len(authoritative_local & global_available)),
        ("legacy_global_local_common", len(legacy_local & global_available)),
        ("known_chemistry_exclusions", len(authoritative_local & excluded)),
        ("chemem_eligible_after_exclusions", len(eligible_ids)),
        ("original_chemem_cohort", len(original)),
        ("existing_chemem_cohort", len(existing)),
        ("confirmatory_additions", len(selected_additions)),
        ("frozen_chemem100", len(selected)),
    ]

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort.reset_index(drop=True).to_csv(output / "chemem100_cohort.csv", index=False)
    membership.to_csv(output / "cohort_membership_audit.csv", index=False)
    pd.DataFrame(counts, columns=["definition", "count"]).to_csv(
        output / "cohort_definition_counts.csv", index=False
    )
    (output / "chemem100_ids.txt").write_text("\n".join(combined_order) + "\n")
    (output / "chemem42_additional_ids.txt").write_text(
        "\n".join(additions["complex_name"].tolist()) + "\n"
    )
    (output / "authoritative_local_drug_like_ids.txt").write_text(
        "\n".join(sorted(authoritative_local)) + "\n"
    )

    input_paths = {
        "local_manifest": args.local_manifest.resolve(),
        "local_metrics": args.local_metrics.resolve(),
        "chemistry_manifest": args.chemistry_manifest.resolve(),
        "existing_cohort": args.existing_cohort.resolve(),
    }
    if args.resolution_table is not None:
        input_paths["resolution_table"] = args.resolution_table.resolve()
    if args.global_available_cohort is not None:
        input_paths["global_available_cohort"] = args.global_available_cohort.resolve()
    if args.legacy_local_cohort is not None:
        input_paths["legacy_local_cohort"] = args.legacy_local_cohort.resolve()
    if args.original_cohort is not None:
        input_paths["original_cohort"] = args.original_cohort.resolve()

    summary = {
        "selection_rule": (
            "Keep existing cases fixed; add cases without reading any docking outcome. "
            "Balance rank-based resolution tertiles and ligand heavy-atom/rotatable-bond tertiles."
        ),
        "target_size": args.target_size,
        "seed": args.seed,
        "class_column": args.class_column,
        "split_membership": args.split_membership,
        "known_chemistry_exclusions": sorted(excluded),
        "counts": dict(counts),
        "resolution_targets": resolution_targets,
        "existing_resolution_counts": existing_resolution,
        "selected_resolution_counts": cohort["resolution_stratum"].value_counts().to_dict(),
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "uses_docking_outcomes_for_selection": False,
        "missing_or_failed_cases_may_not_be_replaced": True,
    }
    (output / "cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    output_files = [
        output / "chemem100_cohort.csv",
        output / "cohort_membership_audit.csv",
        output / "cohort_definition_counts.csv",
        output / "chemem100_ids.txt",
        output / "chemem42_additional_ids.txt",
        output / "authoritative_local_drug_like_ids.txt",
        output / "cohort_summary.json",
    ]
    with (output / "SHA256SUMS").open("w") as handle:
        for path in output_files:
            handle.write(f"{sha256(path)}  {path.name}\n")

    print(pd.DataFrame(counts, columns=["definition", "count"]).to_string(index=False))
    print("\nSelected additions by resolution stratum:")
    print(additions["resolution_stratum"].value_counts().to_string())
    print(f"CHEMEM100_COHORT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
