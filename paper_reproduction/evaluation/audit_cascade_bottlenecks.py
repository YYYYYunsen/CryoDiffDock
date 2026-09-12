#!/usr/bin/env python3
"""Decompose a Global-to-Local cascade into site and Local-docking bottlenecks.

The source reveal manifest is used only after inference has finished.  Global
site error is the distance between the Global-pose heavy-atom centroid (the
actual Local crop centre) and the reference-ligand heavy-atom centroid.  Local
success is evaluated on the frozen selected output for the same branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError("Requested cohort is empty")
    if len(values) != len(set(values)):
        raise ValueError("Requested cohort contains duplicate IDs")
    return values


def parse_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(item <= 0 for item in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be positive comma-separated numbers")
    if len(thresholds) != len(set(thresholds)):
        raise argparse.ArgumentTypeError("thresholds must be unique")
    return thresholds


def resolve_path(value: object, parent: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = parent / path
    return path.resolve()


def heavy_atom_centroid(path: Path) -> np.ndarray:
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported reference-ligand format: {path}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read reference ligand: {path}")
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    heavy = np.asarray([atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()])
    if not heavy.any() or coordinates.shape[0] != len(heavy):
        raise ValueError(f"Reference ligand has no usable heavy atoms: {path}")
    coordinates = coordinates[heavy]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Reference ligand has non-finite coordinates: {path}")
    return coordinates.mean(axis=0)


def classify(site_hit: bool, same_branch_success: bool, any_success: bool, available: bool) -> str:
    if site_hit and same_branch_success:
        return "site_hit_local_success"
    if site_hit and any_success:
        return "site_hit_success_only_on_other_branch"
    if site_hit:
        return "site_hit_local_failure" if available else "site_hit_local_unavailable"
    if any_success:
        return "site_miss_local_rescue"
    return "site_miss_local_failure" if available else "site_miss_local_unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-reveal", required=True, type=Path)
    parser.add_argument("--available-manifest", required=True, type=Path)
    parser.add_argument("--branch-metrics", required=True, type=Path)
    parser.add_argument("--requested-cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--site-thresholds", type=parse_thresholds, default=parse_thresholds("2,4,6"))
    parser.add_argument("--local-rmsd-cutoff", type=float, default=2.0)
    parser.add_argument("--final-centroid-cutoff", type=float, default=2.0)
    args = parser.parse_args()
    if args.top_k <= 0 or args.local_rmsd_cutoff <= 0 or args.final_centroid_cutoff <= 0:
        raise ValueError("top-k and cutoffs must be positive")

    source_path = args.source_reveal.expanduser().resolve()
    available_path = args.available_manifest.expanduser().resolve()
    metrics_path = args.branch_metrics.expanduser().resolve()
    source = pd.read_csv(source_path, dtype=str)
    available = pd.read_csv(available_path, dtype=str)
    metrics = pd.read_csv(metrics_path, dtype=str)

    required_source = {
        "complex_name", "parent_complex_name", "global_rank", "reference_ligand",
        "pocket_center_x", "pocket_center_y", "pocket_center_z",
    }
    missing = sorted(required_source - set(source.columns))
    if missing:
        raise ValueError(f"Source reveal manifest lacks columns: {missing}")
    if "complex_name" not in available.columns:
        raise ValueError("Available manifest lacks 'complex_name'")
    required_metrics = {"row_index", "top1_rmsd_A"}
    missing = sorted(required_metrics - set(metrics.columns))
    if missing:
        raise ValueError(f"Branch metrics lacks columns: {missing}")
    for label, table in (("source", source), ("available", available)):
        if table["complex_name"].isna().any() or table["complex_name"].duplicated().any():
            raise ValueError(f"{label} complex_name values must be nonempty and unique")

    available = available.reset_index().rename(columns={"index": "row_index"})
    available["row_index"] = available["row_index"].astype(int)
    metrics["row_index"] = pd.to_numeric(metrics["row_index"], errors="raise").astype(int)
    if metrics["row_index"].duplicated().any():
        raise ValueError("Branch metrics contains duplicate row_index values")
    if not metrics["row_index"].between(0, len(available) - 1).all():
        raise ValueError("Branch metrics row_index is outside the available manifest")

    metric_columns = [
        column for column in (
            "row_index", "top1_rmsd_A", "top1_centroid_distance_A",
            "top1_centered_rmsd_A", "top1_rigid_aligned_rmsd_A",
        ) if column in metrics.columns
    ]
    local = available[["row_index", "complex_name"]].merge(
        metrics[metric_columns], on="row_index", how="left", validate="one_to_one"
    )
    for column in metric_columns:
        if column != "row_index":
            local[column] = pd.to_numeric(local[column], errors="coerce")
    local = local.drop(columns="row_index")

    source["complex_name"] = source["complex_name"].astype(str)
    source["parent_complex_name"] = source["parent_complex_name"].map(normalize)
    source["global_rank"] = pd.to_numeric(source["global_rank"], errors="raise").astype(int)
    for column in ("pocket_center_x", "pocket_center_y", "pocket_center_z"):
        source[column] = pd.to_numeric(source[column], errors="raise")
    requested = read_ids(args.requested_cohort.expanduser().resolve())
    requested_set = set(requested)
    source = source[source["parent_complex_name"].isin(requested_set)].copy()
    source = source[source["global_rank"].le(args.top_k)].copy()

    reference_cache: dict[Path, np.ndarray] = {}
    reference_errors: list[dict[str, str]] = []
    distances: list[float] = []
    reference_paths: list[str] = []
    for _, row in source.iterrows():
        path = resolve_path(row["reference_ligand"], source_path.parent)
        reference_paths.append(str(path))
        try:
            if path not in reference_cache:
                reference_cache[path] = heavy_atom_centroid(path)
            predicted = row[["pocket_center_x", "pocket_center_y", "pocket_center_z"]].to_numpy(float)
            distances.append(float(np.linalg.norm(predicted - reference_cache[path])))
        except Exception as exc:
            distances.append(np.nan)
            reference_errors.append({
                "complex_name": str(row["complex_name"]),
                "reference_ligand": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            })
    source["reference_ligand_resolved"] = reference_paths
    source["global_center_distance_A"] = distances
    branch = source.merge(local, on="complex_name", how="left", validate="one_to_one")
    branch = branch.rename(columns={
        "top1_rmsd_A": "local_selected_rmsd_A",
        "top1_centroid_distance_A": "local_selected_centroid_distance_A",
        "top1_centered_rmsd_A": "local_selected_centered_rmsd_A",
        "top1_rigid_aligned_rmsd_A": "local_selected_rigid_aligned_rmsd_A",
    })
    branch["local_available"] = np.isfinite(branch["local_selected_rmsd_A"])
    branch["local_success"] = (
        branch["local_available"]
        & branch["local_selected_rmsd_A"].lt(args.local_rmsd_cutoff)
    )
    if "local_selected_centroid_distance_A" in branch.columns:
        branch["local_centroid_hit_but_rmsd_failure"] = (
            branch["local_selected_centroid_distance_A"].lt(args.final_centroid_cutoff)
            & ~branch["local_success"]
        )
        branch["centroid_improvement_A"] = (
            branch["global_center_distance_A"]
            - branch["local_selected_centroid_distance_A"]
        )
    for threshold in args.site_thresholds:
        label = f"{threshold:g}A"
        branch[f"global_site_hit_{label}"] = branch["global_center_distance_A"].le(threshold)
        branch[f"same_branch_success_{label}"] = (
            branch[f"global_site_hit_{label}"] & branch["local_success"]
        )

    parent_rows: list[dict[str, object]] = []
    expected_ranks = set(range(1, args.top_k + 1))
    for pdb_id in requested:
        group = branch[branch["parent_complex_name"].eq(pdb_id)].sort_values("global_rank")
        source_ranks = set(group["global_rank"].tolist())
        rank1 = group[group["global_rank"].eq(1)]
        record: dict[str, object] = {
            "pdb_id": pdb_id,
            "source_branch_count": int(len(group)),
            "available_local_branch_count": int(group["local_available"].sum()) if len(group) else 0,
            "complete_global_top_k": source_ranks >= expected_ranks,
            "complete_local_top_k": (
                source_ranks >= expected_ranks
                and set(group.loc[group["local_available"], "global_rank"].tolist()) >= expected_ranks
            ),
            "rank1_global_center_distance_A": (
                float(rank1.iloc[0]["global_center_distance_A"]) if len(rank1) == 1 else np.nan
            ),
            "rank1_local_selected_rmsd_A": (
                float(rank1.iloc[0]["local_selected_rmsd_A"]) if len(rank1) == 1 else np.nan
            ),
            "best_global_center_distance_top_k_A": (
                float(group["global_center_distance_A"].min()) if len(group) else np.nan
            ),
            "best_available_local_rmsd_top_k_A": (
                float(group.loc[group["local_available"], "local_selected_rmsd_A"].min())
                if group["local_available"].any() else np.nan
            ),
        }
        if len(group) and group["global_center_distance_A"].notna().any():
            record["closest_global_site_rank"] = int(
                group.loc[group["global_center_distance_A"].idxmin(), "global_rank"]
            )
        else:
            record["closest_global_site_rank"] = np.nan
        if len(group) and group["local_available"].any():
            valid_local = group[group["local_available"]]
            record["best_available_local_rank"] = int(
                valid_local.loc[valid_local["local_selected_rmsd_A"].idxmin(), "global_rank"]
            )
        else:
            record["best_available_local_rank"] = np.nan

        for threshold in args.site_thresholds:
            label = f"{threshold:g}A"
            site_column = f"global_site_hit_{label}"
            same_column = f"same_branch_success_{label}"
            r1_site = bool(rank1.iloc[0][site_column]) if len(rank1) == 1 else False
            r1_available = bool(rank1.iloc[0]["local_available"]) if len(rank1) == 1 else False
            r1_success = bool(rank1.iloc[0]["local_success"]) if len(rank1) == 1 else False
            any_site = bool(group[site_column].any()) if len(group) else False
            any_same_success = bool(group[same_column].any()) if len(group) else False
            any_local_success = bool(group["local_success"].any()) if len(group) else False
            any_local_available = bool(group["local_available"].any()) if len(group) else False
            record[f"rank1_site_hit_{label}"] = r1_site
            record[f"rank1_local_success_{label}"] = r1_success
            record[f"rank1_same_branch_success_{label}"] = r1_site and r1_success
            record[f"rank1_outcome_{label}"] = classify(
                r1_site, r1_site and r1_success, r1_success, r1_available
            )
            record[f"top{args.top_k}_any_site_hit_{label}"] = any_site
            record[f"top{args.top_k}_any_available_local_success_{label}"] = any_local_success
            record[f"top{args.top_k}_same_branch_success_{label}"] = any_same_success
            record[f"top{args.top_k}_official_success_{label}"] = (
                bool(record["complete_local_top_k"]) and any_local_success
            )
            record[f"top{args.top_k}_outcome_{label}"] = classify(
                any_site, any_same_success, any_local_success, any_local_available
            )
        parent_rows.append(record)

    parents = pd.DataFrame(parent_rows)
    denominator = len(parents)
    summary_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    for endpoint in ("rank1", f"top{args.top_k}"):
        for threshold in args.site_thresholds:
            label = f"{threshold:g}A"
            if endpoint == "rank1":
                site = parents[f"rank1_site_hit_{label}"]
                success = parents[f"rank1_local_success_{label}"]
                same = parents[f"rank1_same_branch_success_{label}"]
                outcome = parents[f"rank1_outcome_{label}"]
                complete = parents["rank1_local_selected_rmsd_A"].notna()
            else:
                site = parents[f"top{args.top_k}_any_site_hit_{label}"]
                success = parents[f"top{args.top_k}_official_success_{label}"]
                same = parents[f"top{args.top_k}_same_branch_success_{label}"]
                outcome = parents[f"top{args.top_k}_outcome_{label}"]
                complete = parents["complete_local_top_k"]
            site_n = int(site.sum())
            same_n = int(same.sum())
            summary_rows.append({
                "endpoint": endpoint,
                "site_threshold_A": threshold,
                "requested_n": denominator,
                "complete_local_n": int(complete.sum()),
                "global_site_hit_count": site_n,
                "global_site_hit_pct_fixed_denominator": 100.0 * site_n / denominator,
                "local_success_count_fixed_denominator": int(success.sum()),
                "local_success_pct_fixed_denominator": 100.0 * int(success.sum()) / denominator,
                "same_branch_site_hit_and_local_success_count": same_n,
                "conditional_local_success_pct_given_site_hit": (
                    100.0 * same_n / site_n if site_n else np.nan
                ),
                "site_hit_without_same_branch_local_success_count": int((site & ~same).sum()),
                "site_miss_but_local_success_count": int((~site & success).sum()),
            })
            counts = outcome.value_counts(dropna=False)
            for category, count in counts.items():
                category_rows.append({
                    "endpoint": endpoint,
                    "site_threshold_A": threshold,
                    "category": category,
                    "count": int(count),
                    "pct_fixed_denominator": 100.0 * int(count) / denominator,
                })

    branch_summary_rows: list[dict[str, object]] = []
    for threshold in args.site_thresholds:
        label = f"{threshold:g}A"
        site = branch[f"global_site_hit_{label}"]
        local_available = branch["local_available"]
        local_success = branch["local_success"]
        row = {
            "site_threshold_A": threshold,
            "branch_n": len(branch),
            "local_available_n": int(local_available.sum()),
            "site_hit_branch_n": int(site.sum()),
            "site_hit_and_local_success_n": int((site & local_success).sum()),
            "conditional_success_pct_given_site_hit": (
                100.0 * int((site & local_success).sum()) / int(site.sum())
                if site.any() else np.nan
            ),
            "site_hit_but_local_failure_n": int((site & local_available & ~local_success).sum()),
            "site_miss_but_local_success_n": int((~site & local_success).sum()),
        }
        if "local_centroid_hit_but_rmsd_failure" in branch.columns:
            row["final_centroid_hit_but_rmsd_failure_n"] = int(
                branch["local_centroid_hit_but_rmsd_failure"].sum()
            )
        branch_summary_rows.append(row)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch.to_csv(output / "cascade_bottleneck_per_branch.csv", index=False)
    parents.to_csv(output / "cascade_bottleneck_per_parent.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "cascade_bottleneck_summary.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output / "cascade_bottleneck_categories.csv", index=False)
    pd.DataFrame(branch_summary_rows).to_csv(output / "cascade_bottleneck_branch_summary.csv", index=False)
    pd.DataFrame(reference_errors, columns=["complex_name", "reference_ligand", "error"]).to_csv(
        output / "reference_read_failures.csv", index=False
    )
    summary = {
        "requested_cohort_n": denominator,
        "source_branch_n": len(branch),
        "available_local_branch_n": int(branch["local_available"].sum()),
        "top_k": args.top_k,
        "site_thresholds_A": args.site_thresholds,
        "local_success_definition": f"frozen selected Local V2 pose RMSD < {args.local_rmsd_cutoff:g} A",
        "global_site_definition": "Global predicted ligand heavy-atom centroid to reference ligand heavy-atom centroid",
        "missing_policy": "requested parents retained; unavailable Local branches count as failures",
        "reference_read_failures_n": len(reference_errors),
        "uses_reference_only_after_inference": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nPARENT-LEVEL BOTTLENECK SUMMARY")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nBRANCH-LEVEL DIAGNOSTIC")
    print(pd.DataFrame(branch_summary_rows).to_string(index=False))
    print(f"CASCADE_BOTTLENECK_AUDIT_OK: {output}")
    return 0 if not reference_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
