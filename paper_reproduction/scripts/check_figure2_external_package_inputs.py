#!/usr/bin/env python3
"""Validate every external Figure 2 package input before derived analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "complex_id", "pdb_id", "id")


def table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=separator)


def ids(frame: pd.DataFrame, label: str) -> pd.Series:
    column = next((name for name in ID_COLUMNS if name in frame), None)
    if column is None:
        raise ValueError(f"{label} lacks an ID column: {list(frame.columns)}")
    return frame[column].astype(str).str.strip().str.upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--gnina-summary", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    parser.add_argument("--local-metrics", required=True, type=Path)
    parser.add_argument("--local-manifest", required=True, type=Path)
    parser.add_argument("--ligand-map-manifest", required=True, type=Path)
    parser.add_argument("--chemem-benchmark", required=True, type=Path)
    parser.add_argument("--resolution-per-complex", required=True, type=Path)
    parser.add_argument("--initialization-per-complex", required=True, type=Path)
    parser.add_argument("--v2-selected-root", required=True, type=Path)
    parser.add_argument("--diffdock-primary-metrics", required=True, type=Path)
    parser.add_argument("--completion-manifest", required=True, type=Path)
    parser.add_argument("--completion-baseline-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--expected-local-runnable-n", type=int, default=173)
    parser.add_argument("--expected-map-paired-n", type=int, default=165)
    args = parser.parse_args()

    paths = {
        name: value.expanduser().resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if name != "output" and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 2 inputs:\n" + "\n".join(missing))

    cohort = [line.split("#", 1)[0].strip().upper() for line in paths["cohort"].read_text().splitlines()]
    cohort = [value for value in cohort if value]
    cohort_set = set(cohort)
    checks: list[dict[str, object]] = []

    def record(name: str, observed: object, expected: object, passed: bool) -> None:
        checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed)})

    record("cohort_unique_n", len(cohort_set), args.expected_n, len(cohort) == len(cohort_set) == args.expected_n)

    sources = {
        "gnina": table(paths["gnina_summary"]),
        "preview": table(paths["preview"]),
        "local_metrics": table(paths["local_metrics"]),
        "local_manifest": table(paths["local_manifest"]),
        "ligand_map_manifest": table(paths["ligand_map_manifest"]),
        "chemem_benchmark": table(paths["chemem_benchmark"]),
        "resolution": table(paths["resolution_per_complex"]),
        "completion_manifest": table(paths["completion_manifest"]),
        "diffdock_primary": table(paths["diffdock_primary_metrics"]),
    }
    normalized_ids: dict[str, pd.Series] = {}
    for label, frame in sources.items():
        series = ids(frame, label)
        normalized_ids[label] = series
        overlap = int(series.isin(cohort_set).sum())
        unique_overlap = int(series[series.isin(cohort_set)].nunique())
        full_cohort_sources = {
            "gnina", "preview", "local_metrics", "local_manifest",
            "chemem_benchmark", "resolution",
        }
        if label == "ligand_map_manifest":
            expected_overlap: object = args.expected_local_runnable_n
            overlap_passed = unique_overlap == args.expected_local_runnable_n
        elif label in full_cohort_sources:
            expected_overlap = args.expected_n
            overlap_passed = unique_overlap == args.expected_n
        else:
            expected_overlap = "informational"
            overlap_passed = True
        record(
            f"{label}_authoritative_unique_overlap",
            unique_overlap,
            expected_overlap,
            overlap_passed,
        )
        record(f"{label}_duplicate_authoritative_rows", overlap - unique_overlap, 0, overlap == unique_overlap)

    chemem = sources["chemem_benchmark"].copy()
    chemem["_id"] = normalized_ids["chemem_benchmark"]
    local_metrics = sources["local_metrics"].copy()
    local_metrics["_id"] = normalized_ids["local_metrics"]
    chemem_rmsd = pd.to_numeric(chemem["chemem_top1_rmsd_A"], errors="coerce")
    local_rmsd = pd.to_numeric(local_metrics["top1_rmsd_A"], errors="coerce")
    chemem_finite_ids = set(chemem.loc[np.isfinite(chemem_rmsd), "_id"]) & cohort_set
    local_finite_ids = set(local_metrics.loc[np.isfinite(local_rmsd), "_id"]) & cohort_set
    paired_ids = chemem_finite_ids & local_finite_ids
    record(
        "ligand_map_expected_paired_n",
        len(paired_ids),
        args.expected_map_paired_n,
        len(paired_ids) == args.expected_map_paired_n,
    )

    map_manifest = sources["ligand_map_manifest"].copy()
    map_manifest["_id"] = normalized_ids["ligand_map_manifest"]
    map_manifest = map_manifest[map_manifest["_id"].isin(cohort_set)].set_index("_id")
    missing_paired_manifest = sorted(paired_ids - set(map_manifest.index))
    record(
        "ligand_map_paired_manifest_coverage",
        len(paired_ids) - len(missing_paired_manifest),
        len(paired_ids),
        not missing_paired_manifest,
    )
    crop_column = next(
        (name for name in ("ligand_density_crop", "density_crop", "crop_npz") if name in map_manifest),
        None,
    )
    existing_crop_n = 0
    if crop_column is not None and not missing_paired_manifest:
        for case_id in sorted(paired_ids):
            value = str(map_manifest.loc[case_id, crop_column]).strip()
            crop_path = Path(value).expanduser()
            if not crop_path.is_absolute():
                crop_path = paths["ligand_map_manifest"].parent / crop_path
            existing_crop_n += int(crop_path.is_file())
    record(
        "ligand_map_paired_existing_density_crops",
        existing_crop_n,
        len(paired_ids),
        crop_column is not None and existing_crop_n == len(paired_ids),
    )

    gnina = sources["gnina"].copy()
    gnina["_id"] = normalized_ids["gnina"]
    gnina = gnina[gnina["_id"].isin(cohort_set)]
    gnina_rmsd = pd.to_numeric(gnina["top1_rmsd_A"], errors="coerce")
    gnina_runtime = pd.to_numeric(gnina["elapsed_s"], errors="coerce")
    status = gnina.get("status", pd.Series("", index=gnina.index)).astype(str).str.lower()
    record("gnina_finite_top1_n", int(np.isfinite(gnina_rmsd).sum()), args.expected_n, int(np.isfinite(gnina_rmsd).sum()) == args.expected_n)
    record("gnina_finite_positive_runtime_n", int((np.isfinite(gnina_runtime) & gnina_runtime.gt(0)).sum()), args.expected_n, int((np.isfinite(gnina_runtime) & gnina_runtime.gt(0)).sum()) == args.expected_n)
    record("gnina_passed_status_n", int(status.isin({"passed", "complete", "completed", "success"}).sum()), args.expected_n, int(status.isin({"passed", "complete", "completed", "success"}).sum()) == args.expected_n)

    resolution = sources["resolution"].copy()
    stratum_column = next((name for name in ("resolution_stratum", "stratum", "resolution_group") if name in resolution), None)
    if stratum_column is None:
        record("resolution_strata", "missing column", "three 58-case strata", False)
    else:
        resolution["_id"] = normalized_ids["resolution"]
        counts = sorted(resolution.loc[resolution["_id"].isin(cohort_set), stratum_column].value_counts().astype(int).tolist())
        record("resolution_stratum_sizes", counts, [58, 58, 58], counts == [58, 58, 58])

    initialization = table(paths["initialization_per_complex"])
    record("emerald_initialization_rows", len(initialization), 15, len(initialization) == 15)
    selected_dirs = [path for path in paths["v2_selected_root"].glob("index*___*") if path.is_dir()]
    selected_ids = {path.name.split("___", 1)[1].split("_protein", 1)[0].upper() for path in selected_dirs if "___" in path.name}
    record("v2_selected_authoritative_directories", len(selected_ids & cohort_set), 173, len(selected_ids & cohort_set) == 173)
    completion_dirs = [path for path in paths["completion_baseline_root"].glob("index*___*") if path.is_dir()]
    record("completion_baseline_result_directories", len(completion_dirs), ">= 11", len(completion_dirs) >= 11)

    result = pd.DataFrame(checks)
    output = paths["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    failures = result.loc[~result["passed"]]
    if len(failures):
        raise SystemExit(f"FIGURE2_EXTERNAL_PREFLIGHT_FAILED: {len(failures)} checks")
    print(f"FIGURE2_EXTERNAL_PREFLIGHT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
