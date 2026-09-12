#!/usr/bin/env python3
"""Evaluate five cascade branches per parent with cached symmetry mappings."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_ligand_accuracy import (
    centroid_distance,
    locate_result_directory,
    ranked_predictions,
    read_heavy_molecule,
    resolve_path,
    symmetry_rmsd,
)


def rmsd_to_reference_from_isomorphisms(
    reference_coordinates: np.ndarray,
    prediction_coordinates: np.ndarray,
    isomorphisms,
) -> tuple[np.ndarray, int]:
    """Return exact unaligned symmetry RMSD for many poses from one mapping pass."""
    reference_coordinates = np.asarray(reference_coordinates, dtype=float)
    prediction_coordinates = np.asarray(prediction_coordinates, dtype=float)
    if reference_coordinates.ndim != 2 or reference_coordinates.shape[-1] != 3:
        raise ValueError("reference_coordinates must have shape (atoms, 3)")
    if prediction_coordinates.ndim != 3 or prediction_coordinates.shape[-1] != 3:
        raise ValueError("prediction_coordinates must have shape (poses, atoms, 3)")
    if not len(reference_coordinates) or not len(prediction_coordinates):
        raise ValueError("reference and predictions must be nonempty")
    minimum_squared = np.full(len(prediction_coordinates), np.inf, dtype=float)
    mapping_count = 0
    for reference_indices, prediction_indices in isomorphisms:
        reference_indices = np.asarray(reference_indices, dtype=int)
        prediction_indices = np.asarray(prediction_indices, dtype=int)
        if (
            len(reference_indices) != len(reference_coordinates)
            or len(prediction_indices) != prediction_coordinates.shape[1]
            or len(reference_indices) != len(prediction_indices)
        ):
            raise ValueError("isomorphism does not cover every heavy atom")
        difference = (
            reference_coordinates[reference_indices][None, :, :]
            - prediction_coordinates[:, prediction_indices, :]
        )
        squared = np.mean(np.sum(difference * difference, axis=-1), axis=-1)
        minimum_squared = np.minimum(minimum_squared, np.maximum(squared, 0.0))
        mapping_count += 1
    if mapping_count == 0 or not np.isfinite(minimum_squared).all():
        raise ValueError("no usable reference-to-prediction graph isomorphisms were generated")
    return np.sqrt(minimum_squared), mapping_count


def symmetry_rmsd_many(reference, predictions) -> tuple[np.ndarray, int, str]:
    """Enumerate reference-to-pose atom mappings once and reuse them for all poses."""
    from spyrmsd import graph, molecule

    if not predictions:
        raise ValueError("At least one prediction is required")
    converted_reference = molecule.Molecule.from_rdkit(reference)
    converted_predictions = [molecule.Molecule.from_rdkit(value) for value in predictions]
    prediction_atomicnums = np.asarray(converted_predictions[0].atomicnums)
    prediction_adjacency = np.asarray(converted_predictions[0].adjacency_matrix)
    homogeneous = all(
        np.array_equal(np.asarray(value.atomicnums), prediction_atomicnums)
        and np.array_equal(np.asarray(value.adjacency_matrix), prediction_adjacency)
        for value in converted_predictions[1:]
    )
    if not homogeneous:
        return (
            np.asarray([symmetry_rmsd(reference, value) for value in predictions], dtype=float),
            len(predictions),
            "legacy_per_pose_heterogeneous_graph_fallback",
        )
    reference_graph = graph.graph_from_adjacency_matrix(
        np.asarray(converted_reference.adjacency_matrix),
        np.asarray(converted_reference.atomicnums),
    )
    prediction_graph = graph.graph_from_adjacency_matrix(
        prediction_adjacency,
        prediction_atomicnums,
    )
    isomorphisms = graph.match_graphs(reference_graph, prediction_graph)
    reference_coordinates = np.asarray(reference.GetConformer().GetPositions(), dtype=float)
    prediction_coordinates = np.stack([
        np.asarray(value.GetConformer().GetPositions(), dtype=float)
        for value in predictions
    ])
    values, mapping_count = rmsd_to_reference_from_isomorphisms(
        reference_coordinates, prediction_coordinates, isomorphisms
    )
    return values, mapping_count, "cached_reference_to_prediction_isomorphisms"


def evaluate_parent(task):
    parent_id, records, manifest_parent, results_root, condition, legacy_check = task
    started = time.perf_counter()
    try:
        records = sorted(records, key=lambda value: int(value["global_rank"]))
        references = {
            str(resolve_path(value["reference_ligand"], Path(manifest_parent)))
            for value in records
        }
        if len(references) != 1:
            raise ValueError(f"{parent_id}: branches do not share one reference ligand")
        reference_path = Path(references.pop())
        reference = read_heavy_molecule(reference_path)
        predictions = []
        prediction_paths = []
        for record in records:
            directory = locate_result_directory(Path(results_root), int(record["row_index"]))
            ranked = ranked_predictions(directory, 1)
            if len(ranked) != 1:
                raise FileNotFoundError(f"Expected one selected pose in {directory}")
            prediction_path = ranked[0][2]
            predictions.append(read_heavy_molecule(prediction_path))
            prediction_paths.append(prediction_path)
        rmsd_started = time.perf_counter()
        rmsds, mapping_count, mapping_mode = symmetry_rmsd_many(reference, predictions)
        legacy_value = np.nan
        if legacy_check:
            legacy_value = symmetry_rmsd(reference, predictions[0])
            if not np.isclose(rmsds[0], legacy_value, rtol=1e-7, atol=1e-7):
                raise ValueError(
                    f"cached/legacy symmetry RMSD mismatch: {rmsds[0]} != {legacy_value}"
                )
        rmsd_seconds = time.perf_counter() - rmsd_started
        pose_rows = []
        complex_rows = []
        for record, prediction, prediction_path, value in zip(records, predictions, prediction_paths, rmsds):
            row_index = int(record["row_index"])
            complex_name = str(record["complex_name"])
            center = centroid_distance(reference, prediction)
            pose_rows.append({
                "condition": condition,
                "row_index": row_index,
                "complex_name": complex_name,
                "rank": 1,
                "confidence": np.nan,
                "ligand_rmsd_A": float(value),
                "centroid_distance_A": center,
                "prediction_path": str(prediction_path),
                "reference_ligand": str(reference_path),
            })
            complex_rows.append({
                "condition": condition,
                "row_index": row_index,
                "complex_name": complex_name,
                "num_predictions": 1,
                "top1_rmsd_A": float(value),
                "top1_centroid_distance_A": center,
                "best_of_n_rmsd_A": float(value),
                "best_of_n_rank": 1,
                "top1_lt2A": bool(value < 2.0),
                "top1_lt5A": bool(value < 5.0),
                "best_of_n_lt2A": bool(value < 2.0),
                "best_of_n_lt5A": bool(value < 5.0),
            })
        timing = {
            "parent_complex_name": parent_id,
            "branch_count": len(records),
            "symmetry_mapping_count": mapping_count,
            "mapping_mode": mapping_mode,
            "legacy_equivalence_checked": bool(legacy_check),
            "legacy_first_branch_rmsd_A": float(legacy_value),
            "symmetry_rmsd_seconds": rmsd_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        return pose_rows, complex_rows, timing, None
    except Exception as exc:
        return [], [], None, {
            "parent_complex_name": parent_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--expected-parents", type=int, required=True)
    parser.add_argument("--branches-per-parent", type=int, default=5)
    parser.add_argument(
        "--legacy-check-parents", type=int, default=3,
        help="Compare the first cached result with legacy symmrmsd for this many parents.",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.expected_parents < 1 or args.branches_per_parent < 1 or args.legacy_check_parents < 0:
        raise ValueError("workers and expected counts must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    results_root = args.results.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}")
    manifest = pd.read_csv(manifest_path)
    required = {"complex_name", "parent_complex_name", "global_rank", "reference_ligand"}
    if required - set(manifest) or manifest.complex_name.astype(str).duplicated().any():
        raise ValueError("Invalid branch evaluation manifest")
    manifest["parent_complex_name"] = manifest.parent_complex_name.astype(str).str.upper()
    manifest["global_rank"] = pd.to_numeric(manifest.global_rank, errors="raise").astype(int)
    manifest["row_index"] = np.arange(len(manifest))
    grouped = []
    for parent_id, group in manifest.groupby("parent_complex_name", sort=False):
        if len(group) != args.branches_per_parent or set(group.global_rank) != set(range(1, args.branches_per_parent + 1)):
            raise ValueError(f"{parent_id}: incomplete branch set")
        grouped.append((
            parent_id, group.to_dict("records"), str(manifest_path.parent),
            str(results_root), args.condition, len(grouped) < args.legacy_check_parents,
        ))
    if len(grouped) != args.expected_parents:
        raise ValueError(f"Expected {args.expected_parents} parents, found {len(grouped)}")

    executor = None
    results = map(evaluate_parent, grouped)
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(evaluate_parent, grouped, chunksize=1)
    pose_rows, complex_rows, timing_rows, failures = [], [], [], []
    try:
        for index, (poses, complexes, timing, failure) in enumerate(results, 1):
            pose_rows.extend(poses)
            complex_rows.extend(complexes)
            if timing is not None:
                timing_rows.append(timing)
                print(
                    f"[{index}/{len(grouped)}] {timing['parent_complex_name']} "
                    f"branches={timing['branch_count']} mappings={timing['symmetry_mapping_count']} "
                    f"rmsd={timing['symmetry_rmsd_seconds']:.3f}s",
                    flush=True,
                )
            else:
                failures.append(failure)
                print(f"[{index}/{len(grouped)}] FAILED {failure}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pose_rows).sort_values(["row_index", "rank"]).to_csv(output / "per_pose.csv", index=False)
    pd.DataFrame(complex_rows).sort_values("row_index").to_csv(output / "per_complex.csv", index=False)
    pd.DataFrame(timing_rows).to_csv(output / "per_parent_timing.csv", index=False)
    pd.DataFrame(failures, columns=["parent_complex_name", "error"]).to_csv(output / "failures.csv", index=False)
    values = pd.DataFrame(complex_rows)
    expected_branches = args.expected_parents * args.branches_per_parent
    summary = {
        "condition": args.condition,
        "num_parents": len(timing_rows),
        "num_complexes": len(complex_rows),
        "expected_complexes": expected_branches,
        "num_failures": len(failures),
        "rmsd_definition": "heavy-atom symmetry-corrected; center=False; minimize=False",
        "symmetry_mapping_policy": "enumerate reference-to-prediction mappings once per parent and reuse across five branches",
        "legacy_equivalence_checked_parents": int(sum(row["legacy_equivalence_checked"] for row in timing_rows)),
        "median_top1_rmsd_A": float(values.top1_rmsd_A.median()) if len(values) else None,
        "top1_lt2A_pct_fixed_denominator": float(100 * values.top1_lt2A.sum() / expected_branches) if len(values) else 0.0,
        "top1_lt5A_pct_fixed_denominator": float(100 * values.top1_lt5A.sum() / expected_branches) if len(values) else 0.0,
        "total_symmetry_rmsd_seconds": float(sum(row["symmetry_rmsd_seconds"] for row in timing_rows)),
        "maximum_parent_symmetry_rmsd_seconds": float(max((row["symmetry_rmsd_seconds"] for row in timing_rows), default=0.0)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures or len(complex_rows) != expected_branches:
        raise RuntimeError(f"Cached branch RMSD incomplete; inspect {output / 'failures.csv'}")
    print(f"CACHED_CASCADE_BRANCH_RMSD_OK: {expected_branches}/{expected_branches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
