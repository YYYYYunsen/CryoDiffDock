#!/usr/bin/env python3
"""Collect the raw tables and representative structures needed for Figure 2.

The collector is intentionally analysis-light.  It preserves the supplied raw
tables, builds auditable long-form ledgers, identifies the exact comparison
intersections, and copies only the structural assets needed for representative
cases.  It does not silently drop failed or missing cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
from pathlib import Path
from typing import Iterable, Optional


METHODS = (
    "ChemEM Stage 1",
    "CryoDiffDock-Local V1 frozen S0",
    "CryoDiffDock-Local V2 selected",
    "EMERALD",
)
V1_METHOD = "CryoDiffDock-Local V1 frozen S0"
V2_METHOD = "CryoDiffDock-Local V2 selected"
EXTERNAL_METHODS = ("ChemEM Stage 1", V2_METHOD, "EMERALD")
V1_V2_CHEMEM_METHODS = METHODS[:3]
ID_COLUMNS = ("complex_name", "pdb_id", "complex_id", "complex", "target", "id", "name")
COMPLETE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "passed", "success", "succeeded", "ok"}
WALL_COLUMNS = (
    "end_to_end_wall_s",
    "case_runtime_seconds",
    "runtime_seconds",
    "elapsed_seconds",
    "elapsed_s",
    "wall_time_s",
    "runtime_s",
)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def truthy(value: object) -> bool:
    return str(value).strip().lower() in COMPLETE_VALUES


def delimiter_for(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter_for(path))
        if reader.fieldnames is None:
            raise ValueError(f"Table has no header: {path}")
        return list(reader)


def choose_column(columns: Iterable[str], candidates: Iterable[str], label: str, required: bool = True) -> Optional[str]:
    available = set(columns)
    selected = next((candidate for candidate in candidates if candidate in available), None)
    if required and selected is None:
        raise ValueError(f"No {label} column found; candidates={tuple(candidates)} columns={sorted(available)}")
    return selected


def index_unique(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    if not rows:
        return {}
    id_column = choose_column(rows[0], ID_COLUMNS, f"{label} ID")
    assert id_column is not None
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        complex_id = normalize_id(row.get(id_column, ""))
        if not complex_id:
            continue
        if complex_id in indexed:
            raise ValueError(f"Duplicate {label} ID {complex_id}")
        indexed[complex_id] = row
    return indexed


def read_cohort(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        rows = read_rows(path)
        if not rows:
            raise ValueError(f"Cohort table is empty: {path}")
        id_column = choose_column(rows[0], ID_COLUMNS, "cohort ID")
        assert id_column is not None
        ids = [normalize_id(row.get(id_column, "")) for row in rows]
    else:
        ids = [normalize_id(line.split("#", 1)[0]) for line in path.read_text().splitlines()]
    ids = [value for value in ids if value]
    if not ids:
        raise ValueError(f"Cohort has no IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Cohort contains duplicate IDs: {path}")
    return ids


def write_csv(path: Path, rows: list[dict[str, object]], fields: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_source(path: Path, destination: Path, label: str, audit: list[dict[str, object]]) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{label}{path.suffix.lower()}"
    shutil.copy2(path, target)
    audit.append({
        "bundle_path": str(target),
        "source_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
    return target


def method_records(
    method: str,
    path: Path,
) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    rows = read_rows(path)
    indexed = index_unique(rows, method)
    if not rows:
        return {}, rows
    if method == "ChemEM Stage 1":
        metric_column = "chemem_top1_rmsd_A"
        complete_column = "chemem_complete" if "chemem_complete" in rows[0] else None
        status_column = "chemem_source_status" if "chemem_source_status" in rows[0] else None
        failure_column = next((c for c in ("chemem_failure_category", "chemem_failure_text", "chemem_evaluation_error") if c in rows[0]), None)
    else:
        metric_column = "top1_rmsd_A"
        complete_column = None
        status_column = next((column for column in ("status", "state") if column in rows[0]), None)
        failure_column = next((c for c in ("failure_reason", "error", "log") if c in rows[0]), None)
    if metric_column not in rows[0]:
        raise ValueError(f"{method} table lacks {metric_column}: {path}")

    records: dict[str, dict[str, object]] = {}
    for complex_id, row in indexed.items():
        rmsd = finite_float(row.get(metric_column))
        if complete_column:
            complete = truthy(row.get(complete_column, "")) and rmsd is not None
        elif status_column:
            complete = truthy(row.get(status_column, "")) and rmsd is not None
        else:
            complete = rmsd is not None
        records[complex_id] = {
            "observed": True,
            "completed": complete,
            "status": row.get(status_column, "") if status_column else ("completed" if complete else "incomplete"),
            "failure_reason": row.get(failure_column, "") if failure_column else "",
            "top1_rmsd_A": rmsd,
            "source_table": str(path.expanduser().resolve()),
            "source_metric_column": metric_column,
        }
    return records, rows


def resolve_metadata(manifest_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, Optional[str]]]:
    indexed = index_unique(manifest_rows, "input manifest")
    columns = manifest_rows[0].keys() if manifest_rows else []
    mapping = {
        "receptor": choose_column(columns, ("mask_structure", "experimental_protein", "protein_path", "receptor_path", "protein"), "receptor", False),
        "reference_ligand": choose_column(columns, ("reference_ligand", "ligand", "ligand_path", "ligand_sdf"), "reference ligand", False),
        "map": choose_column(columns, ("map_path", "em_map", "density_map", "map"), "map", False),
        "resolution_A": choose_column(columns, ("resolution_A", "map_resolution_A", "resolution", "em_resolution"), "resolution", False),
        "resolution_stratum": choose_column(columns, ("resolution_stratum", "resolution_bin", "resolution_group"), "resolution stratum", False),
        "center_x": choose_column(columns, ("pocket_center_x", "center_x", "site_x"), "center x", False),
        "center_y": choose_column(columns, ("pocket_center_y", "center_y", "site_y"), "center y", False),
        "center_z": choose_column(columns, ("pocket_center_z", "center_z", "site_z"), "center z", False),
    }
    return indexed, mapping


def merge_manifest_rows(
    metadata_rows: list[dict[str, str]],
    asset_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Merge manifests by ID, preferring non-empty asset-manifest values."""
    metadata = index_unique(metadata_rows, "metadata manifest") if metadata_rows else {}
    assets = index_unique(asset_rows, "input manifest")
    ids = list(dict.fromkeys([*metadata, *assets]))
    merged: list[dict[str, str]] = []
    for complex_id in ids:
        row = dict(metadata.get(complex_id, {}))
        for column, value in assets.get(complex_id, {}).items():
            if str(value).strip():
                row[column] = value
        row["complex_name"] = complex_id
        merged.append(row)
    return merged


def pose_index(path: Optional[Path], label: str) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_rows(path)
    if not rows:
        return {}
    id_column = choose_column(rows[0], ID_COLUMNS, f"{label} pose ID")
    path_column = choose_column(
        rows[0],
        ("prediction_path", "prediction", "materialized_path", "pdb_path", "pose_path", "top1_pose_path"),
        f"{label} pose path",
    )
    rank_column = choose_column(rows[0], ("rank", "materialized_rank"), f"{label} rank", False)
    assert id_column is not None and path_column is not None
    result: dict[str, str] = {}
    for row in rows:
        complex_id = normalize_id(row.get(id_column, ""))
        if not complex_id:
            continue
        rank = finite_float(row.get(rank_column)) if rank_column else 1.0
        if rank != 1.0:
            continue
        value = str(row.get(path_column, "")).strip()
        if value and complex_id not in result:
            result[complex_id] = value
    return result


def emerald_pose_index(summary_rows: list[dict[str, str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    indexed = index_unique(summary_rows, "EMERALD summary")
    for complex_id, row in indexed.items():
        explicit = str(row.get("top1_pose_path", "")).strip()
        if explicit and Path(explicit).is_file():
            result[complex_id] = explicit
            continue
        workdir = str(row.get("workdir", row.get("selected_case_dir", ""))).strip()
        if not workdir:
            continue
        ranked = Path(workdir) / "results" / "emerald_rmsd_ranked.tsv"
        if not ranked.is_file():
            continue
        ranked_rows = read_rows(ranked)
        if ranked_rows and str(ranked_rows[0].get("pdb_path", "")).strip():
            result[complex_id] = str(ranked_rows[0]["pdb_path"]).strip()
    return result


def runtime_rows(method: str, path: Path) -> list[dict[str, object]]:
    rows = read_rows(path)
    if not rows:
        return []
    id_column = choose_column(rows[0], ID_COLUMNS, f"{method} runtime ID")
    wall_column = choose_column(rows[0], WALL_COLUMNS, f"{method} wall time")
    status_column = choose_column(rows[0], ("status", "state", "job_status"), f"{method} runtime status", False)
    assert id_column is not None and wall_column is not None
    output: list[dict[str, object]] = []
    for row in rows:
        complex_id = normalize_id(row.get(id_column, ""))
        if not complex_id:
            continue
        wall = finite_float(row.get(wall_column))
        status = str(row.get(status_column, "")).strip() if status_column else ("completed" if wall is not None else "missing")
        output.append({
            "complex_id": complex_id,
            "method": method,
            "status": status,
            "completed": truthy(status) if status_column else wall is not None,
            "wall_s": wall,
            "wall_source_column": wall_column,
            "prep_wall_s": finite_float(row.get("prep_wall_s")),
            "docking_wall_s": finite_float(row.get("docking_wall_s")),
            "cpu_core_s": finite_float(row.get("cpu_core_s")),
            "gpu_s": finite_float(row.get("gpu_s")),
            "max_concurrent_cpu_cores": finite_float(row.get("max_concurrent_cpu_cores")),
            "gpu_count": finite_float(row.get("gpu_count")),
            "source_table": str(path.expanduser().resolve()),
        })
    return output


def category_candidates(ids: list[str], accuracy: dict[str, dict[str, dict[str, object]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    available_methods = METHODS
    for complex_id in ids:
        values = {
            method: accuracy[method][complex_id]["top1_rmsd_A"]
            for method in available_methods
            if complex_id in accuracy[method] and accuracy[method][complex_id]["completed"]
        }
        if not all(method in values for method in EXTERNAL_METHODS):
            continue
        external_values = [float(values[method]) for method in EXTERNAL_METHODS]
        categories: list[tuple[str, float]] = []
        if all(value < 2.0 for value in external_values):
            categories.append(("agreement", max(external_values)))
        if float(values[V2_METHOD]) < 2.0 and float(values["ChemEM Stage 1"]) >= 5.0:
            categories.append(("cryodiffdock_rescue", -(float(values["ChemEM Stage 1"]) - float(values[V2_METHOD]))))
        if all(value >= 5.0 for value in external_values):
            categories.append(("shared_failure", sum(external_values) / len(external_values)))
        for category, priority in categories:
            rows.append({
                "category": category,
                "complex_id": complex_id,
                "selection_priority": priority,
                "chemem_top1_rmsd_A": values.get("ChemEM Stage 1"),
                "v1_top1_rmsd_A": values.get("CryoDiffDock-Local V1 frozen S0"),
                "v2_top1_rmsd_A": values.get("CryoDiffDock-Local V2 selected"),
                "emerald_top1_rmsd_A": values.get("EMERALD"),
                "three_software_complete": all(method in values for method in EXTERNAL_METHODS),
                "all_four_complete": len(values) == len(METHODS),
            })
    return sorted(
        rows,
        key=lambda row: (
            str(row["category"]),
            float(row["selection_priority"]),
            str(row["complex_id"]),
        ),
    )


def select_cases(candidates: list[dict[str, object]], overrides: list[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for text in overrides:
        if "=" not in text:
            raise ValueError(f"--case expects CATEGORY=PDB_ID, got {text!r}")
        category, complex_id = text.split("=", 1)
        category = category.strip()
        if category not in {"agreement", "cryodiffdock_rescue", "shared_failure"}:
            raise ValueError(f"Unknown representative category: {category}")
        selected[category] = normalize_id(complex_id)
    for row in candidates:
        selected.setdefault(str(row["category"]), str(row["complex_id"]))
    return selected


def copy_case_assets(
    selected: dict[str, str],
    output: Path,
    manifest: dict[str, dict[str, str]],
    metadata_columns: dict[str, Optional[str]],
    poses: dict[str, dict[str, str]],
    accuracy: dict[str, dict[str, dict[str, object]]],
    audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    case_rows: list[dict[str, object]] = []
    method_slug = {
        "ChemEM Stage 1": "chemem_top1",
        "CryoDiffDock-Local V1 frozen S0": "v1_top1",
        "CryoDiffDock-Local V2 selected": "v2_top1",
        "EMERALD": "emerald_top1",
    }
    for category, complex_id in selected.items():
        case_dir = output / "representative_cases" / f"{category}__{complex_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        source_row = manifest.get(complex_id, {})
        metadata: dict[str, object] = {"category": category, "complex_id": complex_id}
        for field, column in metadata_columns.items():
            metadata[field] = source_row.get(column, "") if column else ""
        for method in METHODS:
            metadata[f"{method_slug[method]}_rmsd_A"] = accuracy.get(method, {}).get(complex_id, {}).get("top1_rmsd_A")

        assets = {
            "receptor": source_row.get(metadata_columns["receptor"], "") if metadata_columns["receptor"] else "",
            "reference_ligand": source_row.get(metadata_columns["reference_ligand"], "") if metadata_columns["reference_ligand"] else "",
            "map": source_row.get(metadata_columns["map"], "") if metadata_columns["map"] else "",
        }
        for method, index in poses.items():
            assets[method_slug[method]] = index.get(complex_id, "")

        for label, source_text in assets.items():
            metadata[f"{label}_source"] = source_text
            if not source_text:
                metadata[f"{label}_bundle"] = ""
                continue
            source = Path(source_text).expanduser()
            if not source.is_file():
                metadata[f"{label}_bundle"] = "MISSING"
                continue
            destination = case_dir / f"{label}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            metadata[f"{label}_bundle"] = str(destination.relative_to(output))
            audit.append({
                "bundle_path": str(destination),
                "source_path": str(source.resolve()),
                "size_bytes": source.stat().st_size,
                "sha256": sha256(source),
            })
        case_rows.append(metadata)
        write_csv(case_dir / "case_metadata.csv", [metadata])
    return case_rows


def protocol_template() -> list[dict[str, object]]:
    fields = (
        "method", "software_version", "code_commit", "task_mode", "protein_input",
        "ligand_input", "map_input", "pocket_center_source", "reference_pose_visible_during_inference",
        "method_specific_preprocessing", "candidate_budget", "top1_ranking_rule", "hardware",
        "allocated_cpu_cores", "allocated_gpu_count", "process_count", "timing_boundary",
    )
    return [{field: (method if field == "method" else "FILL_ON_SERVER") for field in fields} for method in METHODS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path, help="Frozen ChemEM cohort ID list or table")
    parser.add_argument("--input-manifest", required=True, type=Path, help="Per-case receptor/ligand/map/center manifest")
    parser.add_argument("--metadata-manifest", type=Path, help="Optional cohort metadata table (for example resolution)")
    parser.add_argument("--chemem-benchmark", required=True, type=Path)
    parser.add_argument("--v1-metrics", required=True, type=Path)
    parser.add_argument("--v2-metrics", required=True, type=Path)
    parser.add_argument("--emerald-summary", required=True, type=Path)
    parser.add_argument("--chemem-per-pose", type=Path)
    parser.add_argument("--v1-per-pose", type=Path)
    parser.add_argument("--v2-per-pose", type=Path)
    parser.add_argument("--chemem-runtime", type=Path)
    parser.add_argument("--v1-runtime", type=Path)
    parser.add_argument("--v2-runtime", type=Path)
    parser.add_argument("--protocol-audit", type=Path)
    parser.add_argument("--protocol-file", action="append", default=[], type=Path)
    parser.add_argument("--extra-file", action="append", default=[], type=Path)
    parser.add_argument("--case", action="append", default=[], help="Override representative case as CATEGORY=PDB_ID")
    parser.add_argument("--expected-primary-n", type=int)
    parser.add_argument("--expected-three-software-n", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()

    paths = {
        "cohort": args.cohort,
        "input_manifest": args.input_manifest,
        "chemem_benchmark": args.chemem_benchmark,
        "v1_metrics": args.v1_metrics,
        "v2_metrics": args.v2_metrics,
        "emerald_summary": args.emerald_summary,
    }
    if args.metadata_manifest is not None:
        paths["metadata_manifest"] = args.metadata_manifest
    for label, path in paths.items():
        paths[label] = path.expanduser().resolve()
        if not paths[label].is_file():
            raise FileNotFoundError(f"Missing required {label}: {paths[label]}")

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_audit: list[dict[str, object]] = []
    raw_dir = output / "raw_sources"
    for label, path in paths.items():
        copy_source(path, raw_dir, label, source_audit)
    optional_sources = {
        "chemem_per_pose": args.chemem_per_pose,
        "v1_per_pose": args.v1_per_pose,
        "v2_per_pose": args.v2_per_pose,
        "chemem_runtime": args.chemem_runtime,
        "v1_runtime": args.v1_runtime,
        "v2_runtime": args.v2_runtime,
    }
    for label, path in optional_sources.items():
        if path is not None:
            copy_source(path, raw_dir, label, source_audit)

    cohort_ids = read_cohort(paths["cohort"])
    asset_manifest_rows = read_rows(paths["input_manifest"])
    metadata_manifest_rows = read_rows(paths["metadata_manifest"]) if "metadata_manifest" in paths else []
    manifest_rows = merge_manifest_rows(metadata_manifest_rows, asset_manifest_rows)
    manifest, metadata_columns = resolve_metadata(manifest_rows)

    accuracy: dict[str, dict[str, dict[str, object]]] = {}
    accuracy["ChemEM Stage 1"], _ = method_records("ChemEM Stage 1", paths["chemem_benchmark"])
    accuracy["CryoDiffDock-Local V1 frozen S0"], _ = method_records("CryoDiffDock-Local V1 frozen S0", paths["v1_metrics"])
    accuracy["CryoDiffDock-Local V2 selected"], _ = method_records("CryoDiffDock-Local V2 selected", paths["v2_metrics"])
    accuracy["EMERALD"], emerald_rows = method_records("EMERALD", paths["emerald_summary"])

    primary_ids = [complex_id for complex_id in cohort_ids if all(accuracy[m].get(complex_id, {}).get("completed") for m in V1_V2_CHEMEM_METHODS)]
    three_software_ids = [complex_id for complex_id in cohort_ids if all(accuracy[m].get(complex_id, {}).get("completed") for m in EXTERNAL_METHODS)]
    all_four_ids = [complex_id for complex_id in cohort_ids if all(accuracy[m].get(complex_id, {}).get("completed") for m in METHODS)]
    emerald_matched_ids = [complex_id for complex_id in cohort_ids if all(accuracy[m].get(complex_id, {}).get("completed") for m in ("CryoDiffDock-Local V2 selected", "EMERALD"))]
    if args.expected_primary_n is not None and len(primary_ids) != args.expected_primary_n:
        raise ValueError(
            f"Primary ChemEM/V1/V2 intersection is {len(primary_ids)}, expected {args.expected_primary_n}. "
            "The output was left in place for audit."
        )
    if args.expected_three_software_n is not None and len(three_software_ids) != args.expected_three_software_n:
        raise ValueError(
            f"Strict ChemEM/V2/EMERALD intersection is {len(three_software_ids)}, "
            f"expected {args.expected_three_software_n}. The output was left in place for audit."
        )

    ledger: list[dict[str, object]] = []
    for complex_id in cohort_ids:
        meta = manifest.get(complex_id, {})
        resolution_column = metadata_columns["resolution_A"]
        for method in METHODS:
            record = accuracy[method].get(complex_id, {})
            rmsd = record.get("top1_rmsd_A")
            ledger.append({
                "complex_id": complex_id,
                "method": method,
                "in_frozen_cohort": True,
                "observed": bool(record.get("observed", False)),
                "completed": bool(record.get("completed", False)),
                "status": record.get("status", "missing"),
                "failure_reason": record.get("failure_reason", ""),
                "top1_symmetry_corrected_rmsd_A": rmsd,
                "top1_lt2A": (float(rmsd) < 2.0) if rmsd is not None else "",
                "top1_lt5A": (float(rmsd) < 5.0) if rmsd is not None else "",
                "resolution_A": meta.get(resolution_column, "") if resolution_column else "",
                "resolution_stratum": meta.get(metadata_columns["resolution_stratum"], "") if metadata_columns["resolution_stratum"] else "",
                "in_chemem_v1_v2_common": complex_id in primary_ids,
                "in_three_software_common": complex_id in three_software_ids,
                "in_v2_emerald_matched": complex_id in emerald_matched_ids,
                "in_all_four_common": complex_id in all_four_ids,
                "source_table": record.get("source_table", ""),
                "source_metric_column": record.get("source_metric_column", ""),
            })
    write_csv(output / "tables" / "accuracy_per_complex_long.csv", ledger)

    cohort_ledger: list[dict[str, object]] = []
    for complex_id in cohort_ids:
        row: dict[str, object] = {
            "complex_id": complex_id,
            "in_chemem_v1_v2_common": complex_id in primary_ids,
            "in_three_software_common": complex_id in three_software_ids,
            "in_v2_emerald_matched": complex_id in emerald_matched_ids,
            "in_all_four_common": complex_id in all_four_ids,
        }
        for method, prefix in zip(METHODS, ("chemem", "v1", "v2", "emerald")):
            record = accuracy[method].get(complex_id, {})
            row[f"{prefix}_observed"] = bool(record.get("observed", False))
            row[f"{prefix}_completed"] = bool(record.get("completed", False))
            row[f"{prefix}_status"] = record.get("status", "missing")
            row[f"{prefix}_failure_reason"] = record.get("failure_reason", "")
        cohort_ledger.append(row)
    write_csv(output / "tables" / "cohort_ledger.csv", cohort_ledger)
    (output / "tables" / "chemem_v1_v2_common_ids.txt").write_text("\n".join(primary_ids) + "\n")
    (output / "tables" / "three_software_common_ids.txt").write_text("\n".join(three_software_ids) + "\n")
    (output / "tables" / "v2_emerald_matched_ids.txt").write_text("\n".join(emerald_matched_ids) + "\n")
    (output / "tables" / "all_four_common_ids.txt").write_text("\n".join(all_four_ids) + "\n")

    runtime: list[dict[str, object]] = []
    for method, path in (
        ("ChemEM Stage 1", args.chemem_runtime),
        ("CryoDiffDock-Local V1 frozen S0", args.v1_runtime),
        ("CryoDiffDock-Local V2 selected", args.v2_runtime),
        ("EMERALD", paths["emerald_summary"]),
    ):
        if path is not None:
            runtime.extend(runtime_rows(method, Path(path).expanduser().resolve()))
    write_csv(output / "tables" / "runtime_per_complex_long.csv", runtime, fields=[
        "complex_id", "method", "status", "completed", "wall_s", "wall_source_column",
        "prep_wall_s", "docking_wall_s", "cpu_core_s", "gpu_s",
        "max_concurrent_cpu_cores", "gpu_count", "source_table",
    ])

    protocol_dir = output / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    if args.protocol_audit:
        copy_source(args.protocol_audit, protocol_dir, "protocol_audit", source_audit)
    else:
        template = protocol_template()
        write_csv(protocol_dir / "protocol_audit_FILL_ON_SERVER.csv", template)
    for index, path in enumerate(args.protocol_file, start=1):
        copy_source(path, protocol_dir, f"protocol_file_{index:02d}_{path.stem}", source_audit)
    for index, path in enumerate(args.extra_file, start=1):
        copy_source(path, output / "extra", f"extra_file_{index:02d}_{path.stem}", source_audit)

    candidates = category_candidates(three_software_ids, accuracy)
    write_csv(output / "tables" / "representative_case_candidates.csv", candidates, fields=[
        "category", "complex_id", "selection_priority", "chemem_top1_rmsd_A",
        "v1_top1_rmsd_A", "v2_top1_rmsd_A", "emerald_top1_rmsd_A",
        "three_software_complete", "all_four_complete",
    ])
    selected = select_cases(candidates, args.case)
    poses = {
        "ChemEM Stage 1": pose_index(args.chemem_per_pose, "ChemEM"),
        "CryoDiffDock-Local V1 frozen S0": pose_index(args.v1_per_pose, "V1"),
        "CryoDiffDock-Local V2 selected": pose_index(args.v2_per_pose, "V2"),
        "EMERALD": emerald_pose_index(emerald_rows),
    }
    copied_cases = copy_case_assets(selected, output, manifest, metadata_columns, poses, accuracy, source_audit)

    write_csv(output / "source_file_audit.csv", source_audit)
    summary = {
        "frozen_cohort_n": len(cohort_ids),
        "method_completed_n": {
            method: sum(bool(accuracy[method].get(complex_id, {}).get("completed")) for complex_id in cohort_ids)
            for method in METHODS
        },
        "chemem_v1_v2_common_n": len(primary_ids),
        "three_software_common_n": len(three_software_ids),
        "v2_emerald_matched_n": len(emerald_matched_ids),
        "all_four_common_n": len(all_four_ids),
        "selected_representative_cases": selected,
        "representative_case_count": len(copied_cases),
        "runtime_rows": len(runtime),
        "warning": (
            "The primary external comparison is the strict ChemEM/V2/EMERALD paired set. Also report the "
            "full frozen-cohort missing-as-failure result. Runtime rows are method-native unless the protocol "
            "audit proves a shared end-to-end boundary; do not call them identical end-to-end timings."
        ),
    }
    (output / "bundle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "README.txt").write_text(
        "Figure 2 external Local benchmark raw-data bundle\n\n"
        "tables/accuracy_per_complex_long.csv powers panels C and D.\n"
        "tables/cohort_ledger.csv powers panel B.\n"
        "tables/runtime_per_complex_long.csv powers panel E.\n"
        "protocol/ powers panel A.\n"
        "representative_cases/ powers panel F.\n\n"
        "Use tables/three_software_common_ids.txt for the primary paired external comparison.\n"
        "Do not replace missing/failed cases or mix denominators without qualification.\n"
        "Check bundle_summary.json and source_file_audit.csv before transfer.\n"
    )

    if args.archive:
        archive = output.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(output, arcname=output.name)
        print(f"ARCHIVE: {archive}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"FIGURE2_BUNDLE_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
