#!/usr/bin/env python3
"""Expand an existing pose pool by deterministic local moves and rank by NCC.

This is a deployment-safe generation stage.  It never reads a reference
ligand or an RMSD.  Every original pose is retained, while new rigid-body and
single-torsion variants must improve their parent's density score.  The
optional physical gate is relative to the parent rather than an absolute
crystal-structure-derived threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import (
    crop_center,
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
    zscore,
)
from cryodiffdock_pocket.discrete_search import (
    centroid_distance,
    direct_rmsd,
    generate_local_variants,
    greedy_diverse_indices,
)
from cryodiffdock_pocket.physics import (
    clash_metrics,
    paired_protein_for_pose,
    read_protein_heavy_atoms,
)
from cryodiffdock_pocket.search_budget import (
    allocate_total_candidate_budget,
    per_complex_wall_clock_budget,
)


def comma_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_molecule(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read pose: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    try:
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
    except Exception:
        pass
    return molecule


def molecule_coordinates(molecule) -> np.ndarray:
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)


def molecule_with_coordinates(template, coordinates: np.ndarray):
    molecule = Chem.Mol(template)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (molecule.GetNumAtoms(), 3):
        raise ValueError(
            f"coordinate shape {coordinates.shape} does not match "
            f"{molecule.GetNumAtoms()} atoms"
        )
    conformer = molecule.GetConformer()
    for atom_index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, tuple(float(value) for value in xyz))
    return molecule


def write_sdf(molecule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in text)


def choose_protein_path(pose_path: Path, manifest_row: pd.Series, manifest_base: Path) -> Path:
    paired = paired_protein_for_pose(pose_path)
    if paired is not None:
        return paired.resolve()
    for column in ("experimental_protein", "protein_path", "protein"):
        if column in manifest_row and pd.notna(manifest_row[column]):
            candidate = resolve_path(manifest_row[column], manifest_base)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No paired or manifest protein found for {pose_path}")


def rank_records(records: list[dict], key: str, eligible_key: str | None = None):
    eligible = records if eligible_key is None else [row for row in records if row[eligible_key]]
    if not eligible:
        raise RuntimeError(f"No candidates eligible for {key}")
    return sorted(
        eligible,
        key=lambda row: (-float(row[key]), -float(row["confidence"]), row["candidate_id"]),
    )


def materialize(
    root: Path,
    row_index: int,
    complex_name: str,
    ranked: list[dict],
    molecules: dict[str, object],
    score_key: str,
    limit: int | None = None,
    copy_protein: bool = False,
) -> None:
    directory = root / f"index{row_index}___{safe_name(complex_name)}"
    directory.mkdir(parents=True, exist_ok=True)
    selected = ranked if limit is None else ranked[:limit]
    ranking_rows = []
    for rank, record in enumerate(selected, start=1):
        score = float(record[score_key])
        destination = directory / f"rank{rank}_confidence{score:.8f}.sdf"
        if record["is_original"]:
            shutil.copy2(record["prediction_path"], destination)
        else:
            write_sdf(molecules[record["candidate_id"]], destination)
        protein_path = Path(record["protein_path"])
        if copy_protein and protein_path.is_file():
            shutil.copy2(protein_path, destination.with_name(f"{destination.stem}_protein.pdb"))
        ranking_rows.append({**record, "materialized_rank": rank, "materialized_path": str(destination)})
    pd.DataFrame(ranking_rows).to_csv(directory / "ranking.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path,
                        help="pose_scores.csv from rerank_ligand_by_density_ncc.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", default="v1_fixed")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N manifest rows (smoke tests)")
    parser.add_argument("--complexes", default=None,
                        help="Comma-separated complex names for a fixed pilot subset")
    parser.add_argument("--density-column", default="ncc_sigma_1p5")
    parser.add_argument("--density-sigma", type=float, default=1.5)
    parser.add_argument("--density-weight", type=float, default=2.0)
    parser.add_argument("--parent-score-column", default="confidence",
                        choices=("confidence", "combined_score"))
    parser.add_argument("--parent-count", type=int, default=6)
    parser.add_argument("--top-score-parents", type=int, default=3)
    parser.add_argument("--parent-diversity-rmsd", type=float, default=1.5)
    parser.add_argument("--rotation-degrees", type=comma_floats, default=comma_floats("15,30,60"))
    parser.add_argument("--translation-distances", type=comma_floats, default=comma_floats("0.5,1.0"))
    parser.add_argument("--torsion-degrees", type=comma_floats, default=comma_floats("60,120,180"))
    parser.add_argument("--max-torsion-bonds", type=int, default=8)
    parser.add_argument("--no-coarse-orientations", action="store_true",
                        help="Disable the 23-member coarse rigid orientation library")
    parser.add_argument("--minimum-ncc-gain", type=float, default=0.0)
    parser.add_argument("--anchor-penalty", type=float, default=0.05,
                        help="Penalty per Angstrom raw RMSD from the parent")
    parser.add_argument("--max-centroid-shift", type=float, default=1.5)
    parser.add_argument("--min-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--clash-tolerance-count", type=int, default=0)
    parser.add_argument("--clash-tolerance-fraction", type=float, default=0.0)
    parser.add_argument("--severe-overlap", type=float, default=0.8)
    parser.add_argument("--mask-radius", type=float, default=10.0)
    parser.add_argument("--target-floor", type=float, default=None)
    parser.add_argument("--min-mask-voxels", type=int, default=64)
    parser.add_argument("--cutoff-sigma", type=float, default=4.0)
    parser.add_argument("--materialize-expanded-limit", type=int, default=0,
                        help="0 writes every expanded candidate for oracle evaluation")
    parser.add_argument(
        "--total-candidate-budget",
        type=int,
        default=None,
        help=(
            "Cohort-wide cap including original poses. Remaining candidates are "
            "allocated evenly in manifest order so the requested total is exact."
        ),
    )
    parser.add_argument(
        "--total-wall-clock-budget-seconds",
        type=float,
        default=None,
        help=(
            "Cohort-wide expansion ceiling split evenly across complexes. Final "
            "ranking/materialization still completes and observed time is recorded."
        ),
    )
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.parent_count <= 0 or args.top_score_parents > args.parent_count:
        raise ValueError("Require 0 <= top-score-parents <= parent-count and parent-count > 0")
    if args.max_torsion_bonds < 0:
        raise ValueError("max-torsion-bonds must be non-negative")

    run_started = time.perf_counter()
    manifest_path = args.manifest.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    scores = pd.read_csv(scores_path)
    required_manifest = {"complex_name", "ligand_density_crop"}
    required_scores = {
        "row_index", "complex_name", "source", "source_rank", "confidence",
        "prediction_path", args.density_column,
    }
    missing = sorted(required_manifest - set(manifest.columns))
    missing_scores = sorted(required_scores - set(scores.columns))
    if missing or missing_scores:
        raise ValueError(f"Missing manifest columns {missing}; score columns {missing_scores}")

    manifest = manifest.copy()
    manifest["_source_row_index"] = np.arange(len(manifest), dtype=int)
    if args.complexes:
        requested = [value.strip() for value in args.complexes.split(",") if value.strip()]
        requested_set = set(requested)
        available = set(manifest["complex_name"].astype(str))
        absent = sorted(requested_set - available)
        if absent:
            raise ValueError(f"Requested complexes are absent from manifest: {absent}")
        order = {name: index for index, name in enumerate(requested)}
        manifest = manifest[manifest["complex_name"].astype(str).isin(requested_set)]
        manifest = manifest.assign(
            _requested_order=manifest["complex_name"].astype(str).map(order)
        ).sort_values("_requested_order", kind="stable").drop(columns="_requested_order")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest.head(args.limit)
    manifest = manifest.reset_index(drop=True)
    score_row_indices = pd.to_numeric(scores["row_index"], errors="raise").astype(int)
    original_counts = [
        int(score_row_indices.eq(int(source_index)).sum())
        for source_index in manifest["_source_row_index"]
    ]
    if any(count == 0 for count in original_counts):
        missing_rows = [
            int(source_index)
            for source_index, count in zip(manifest["_source_row_index"], original_counts)
            if count == 0
        ]
        raise FileNotFoundError(f"No scored poses for source rows: {missing_rows[:10]}")
    case_candidate_budgets = allocate_total_candidate_budget(
        args.total_candidate_budget, original_counts
    )
    case_wall_clock_budget = per_complex_wall_clock_budget(
        args.total_wall_clock_budget_seconds, len(manifest)
    )
    manifest.drop(columns="_source_row_index").to_csv(
        output / "evaluation_manifest.csv", index=False
    )

    roots = {
        "original_confidence": output / "original_confidence",
        "original_density": output / "original_density",
        "expanded_confidence": output / "expanded_confidence",
        "expanded_density": output / "expanded_density",
        "expanded_density_physics": output / "expanded_density_physics",
        "expanded_pool": output / "expanded_pool",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    selected_records: list[dict] = []
    audit_rows: list[dict] = []
    failures: list[dict] = []
    protein_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    for row_index, manifest_row in manifest.iterrows():
        source_row_index = int(manifest_row["_source_row_index"])
        complex_name = str(manifest_row["complex_name"])
        case_started = time.perf_counter()
        case_candidate_budget = case_candidate_budgets[row_index]
        case_deadline = (
            None
            if case_wall_clock_budget is None
            else case_started + case_wall_clock_budget
        )
        try:
            group = scores[
                scores["row_index"].astype(int) == source_row_index
            ].copy()
            if group.empty:
                raise FileNotFoundError(
                    f"No scored candidates for source row {source_row_index}"
                )
            crop_path = resolve_path(manifest_row["ligand_density_crop"], manifest_path.parent)
            crop = load_cartesian_density_crop(crop_path)
            mask = fixed_crop_mask(
                crop, radius_A=args.mask_radius, target_floor=args.target_floor,
                min_voxels=args.min_mask_voxels,
            )

            originals = []
            molecule_by_id: dict[str, object] = {}
            coordinates = []
            for _, source_row in group.iterrows():
                pose_path = Path(str(source_row["prediction_path"])).expanduser().resolve()
                molecule = read_molecule(pose_path)
                xyz = molecule_coordinates(molecule)
                candidate_id = (
                    f"orig_{safe_name(source_row['source'])}_"
                    f"{int(source_row['source_rank']):03d}"
                )
                if candidate_id in molecule_by_id:
                    candidate_id += f"_{len(originals)}"
                protein_path = choose_protein_path(pose_path, manifest_row, manifest_path.parent)
                record = {
                    "row_index": int(row_index),
                    "source_row_index": source_row_index,
                    "complex_name": complex_name,
                    "candidate_id": candidate_id,
                    "is_original": True,
                    "source": str(source_row["source"]),
                    "source_rank": int(source_row["source_rank"]),
                    "parent_id": candidate_id,
                    "move_type": "original",
                    "move_magnitude": 0.0,
                    "axis_label": "",
                    "bond": "",
                    "confidence": float(source_row["confidence"]),
                    "input_ncc": float(source_row[args.density_column]),
                    "anchor_rmsd_A": 0.0,
                    "centroid_shift_A": 0.0,
                    "prediction_path": str(pose_path),
                    "protein_path": str(protein_path),
                }
                originals.append(record)
                molecule_by_id[candidate_id] = molecule
                coordinates.append(xyz)

                atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
                density = score_gaussian_density(
                    crop, xyz, atomic_numbers, args.density_sigma, mask,
                    cutoff_sigma=args.cutoff_sigma,
                )
                record["ncc"] = float(density.ncc)
                record["in_bounds_fraction"] = float(
                    density.ligand_in_bounds_fraction
                )

            confidence_z = zscore([record["confidence"] for record in originals])
            density_z = zscore([record["ncc"] for record in originals])
            for index, record in enumerate(originals):
                record["confidence_z"] = float(confidence_z[index])
                record["original_density_z"] = float(density_z[index])
                record["original_confidence_score"] = float(confidence_z[index])
                record["original_density_score"] = float(
                    confidence_z[index] + args.density_weight * density_z[index]
                )

            parent_scores = [record[args.parent_score_column] if args.parent_score_column in record else (
                record["original_density_score"] if args.parent_score_column == "combined_score"
                else record["confidence"]
            ) for record in originals]
            parent_indices = greedy_diverse_indices(
                coordinates, parent_scores, count=args.parent_count,
                top_score_count=args.top_score_parents,
                minimum_rmsd_A=args.parent_diversity_rmsd,
            )
            parent_ids = {originals[index]["candidate_id"] for index in parent_indices}

            parent_physics: dict[str, dict] = {}
            stop_expansion = False
            for parent_index in parent_indices:
                if stop_expansion:
                    break
                parent = originals[parent_index]
                molecule = molecule_by_id[parent["candidate_id"]]
                xyz = molecule_coordinates(molecule)
                protein_path = parent["protein_path"]
                if protein_path not in protein_cache:
                    protein_cache[protein_path] = read_protein_heavy_atoms(protein_path)
                protein_xyz, protein_elements = protein_cache[protein_path]
                ligand_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                parent_physics[parent["candidate_id"]] = clash_metrics(
                    protein_xyz, protein_elements, xyz, ligand_elements,
                    severe_overlap_A=args.severe_overlap,
                )

                direction = crop_center(crop) - xyz.mean(axis=0)
                for variant in generate_local_variants(
                    molecule,
                    rotation_degrees=args.rotation_degrees,
                    translation_distances_A=args.translation_distances,
                    torsion_degrees=args.torsion_degrees,
                    max_torsion_bonds=args.max_torsion_bonds,
                    extra_translation_direction=direction,
                    include_coarse_orientations=not args.no_coarse_orientations,
                ):
                    if variant.move_type == "original":
                        continue
                    candidate_budget_full = (
                        case_candidate_budget is not None
                        and len(originals) >= int(case_candidate_budget)
                    )
                    deadline_reached = (
                        case_deadline is not None
                        and time.perf_counter() >= case_deadline
                    )
                    if candidate_budget_full or deadline_reached:
                        stop_expansion = True
                        break
                    variant_id = f"{parent['candidate_id']}__{variant.variant_id}"
                    variant_molecule = molecule_with_coordinates(molecule, variant.coordinates)
                    atomic_numbers = [atom.GetAtomicNum() for atom in variant_molecule.GetAtoms()]
                    density = score_gaussian_density(
                        crop, variant.coordinates, atomic_numbers,
                        args.density_sigma, mask, cutoff_sigma=args.cutoff_sigma,
                    )
                    physics = clash_metrics(
                        protein_xyz, protein_elements, variant.coordinates,
                        ligand_elements, severe_overlap_A=args.severe_overlap,
                    )
                    parent_metric = parent_physics[parent["candidate_id"]]
                    relative_physics_pass = (
                        physics["severe_clash_count"]
                        <= parent_metric["severe_clash_count"] + args.clash_tolerance_count
                        and physics["clashing_ligand_atom_fraction"]
                        <= parent_metric["clashing_ligand_atom_fraction"]
                        + args.clash_tolerance_fraction
                    )
                    anchor_rmsd = direct_rmsd(xyz, variant.coordinates)
                    shift = centroid_distance(xyz, variant.coordinates)
                    record = {
                        "row_index": int(row_index),
                        "source_row_index": source_row_index,
                        "complex_name": complex_name,
                        "candidate_id": variant_id,
                        "is_original": False,
                        "source": parent["source"],
                        "source_rank": parent["source_rank"],
                        "parent_id": parent["candidate_id"],
                        "move_type": variant.move_type,
                        "move_magnitude": variant.magnitude,
                        "axis_label": variant.axis_label,
                        "bond": "" if variant.bond is None else f"{variant.bond[0]}-{variant.bond[1]}",
                        "confidence": parent["confidence"],
                        "confidence_z": parent["confidence_z"],
                        "original_density_z": parent["original_density_z"],
                        "ncc": float(density.ncc),
                        "parent_ncc": parent["ncc"],
                        "ncc_gain": float(density.ncc - parent["ncc"]),
                        "in_bounds_fraction": float(density.ligand_in_bounds_fraction),
                        "anchor_rmsd_A": anchor_rmsd,
                        "centroid_shift_A": shift,
                        "severe_clash_count": physics["severe_clash_count"],
                        "parent_severe_clash_count": parent_metric["severe_clash_count"],
                        "clashing_ligand_atom_fraction": physics["clashing_ligand_atom_fraction"],
                        "parent_clashing_ligand_atom_fraction": parent_metric["clashing_ligand_atom_fraction"],
                        "relative_physics_pass": bool(relative_physics_pass),
                        "prediction_path": "",
                        "protein_path": protein_path,
                    }
                    molecule_by_id[variant_id] = variant_molecule
                    originals.append(record)

            records = originals
            for record in records:
                if record["is_original"]:
                    metric = parent_physics.get(record["candidate_id"])
                    record.update({
                        "parent_ncc": record["ncc"],
                        "ncc_gain": 0.0,
                        "severe_clash_count": None if metric is None else metric["severe_clash_count"],
                        "parent_severe_clash_count": None if metric is None else metric["severe_clash_count"],
                        "clashing_ligand_atom_fraction": None if metric is None else metric["clashing_ligand_atom_fraction"],
                        "parent_clashing_ligand_atom_fraction": None if metric is None else metric["clashing_ligand_atom_fraction"],
                        "relative_physics_pass": True,
                    })
                record["parent_selected"] = record["parent_id"] in parent_ids
                record["density_gate_pass"] = bool(
                    record["is_original"] or (
                        record["ncc_gain"] >= args.minimum_ncc_gain
                        and record["in_bounds_fraction"] >= args.min_in_bounds_fraction
                        and record["centroid_shift_A"] <= args.max_centroid_shift
                    )
                )
                record["density_physics_gate_pass"] = bool(
                    record["density_gate_pass"] and record["relative_physics_pass"]
                )

            original_records = [record for record in records if record["is_original"]]
            # Keep the original-pool calibration fixed.  Re-standardising on
            # the larger variant pool would silently change the effective NCC
            # weight and could alter the original ranking even when no new
            # variant is selected, confounding expansion with score scaling.
            original_ncc = np.asarray(
                [record["ncc"] for record in original_records], dtype=np.float64
            )
            finite_original_ncc = np.isfinite(original_ncc)
            if not finite_original_ncc.any():
                raise ValueError("All original-pool NCC values are non-finite")
            original_ncc_mean = float(original_ncc[finite_original_ncc].mean())
            original_ncc_std = float(original_ncc[finite_original_ncc].std())
            expanded_density_z = []
            for record in records:
                value = float(record["ncc"])
                if not np.isfinite(value):
                    standardized = -10.0
                elif original_ncc_std > 1e-12:
                    standardized = (value - original_ncc_mean) / original_ncc_std
                else:
                    standardized = 0.0
                expanded_density_z.append(float(standardized))
            for index, record in enumerate(records):
                record["expanded_density_z"] = float(expanded_density_z[index])
                record["expanded_confidence_score"] = float(
                    record["confidence_z"] - args.anchor_penalty * record["anchor_rmsd_A"]
                )
                record["expanded_density_score"] = float(
                    record["confidence_z"]
                    + args.density_weight * expanded_density_z[index]
                    - args.anchor_penalty * record["anchor_rmsd_A"]
                )

            ranked_sets = {
                "original_confidence": rank_records(original_records, "original_confidence_score"),
                "original_density": rank_records(original_records, "original_density_score"),
                "expanded_confidence": rank_records(records, "expanded_confidence_score"),
                "expanded_density": rank_records(records, "expanded_density_score", "density_gate_pass"),
                "expanded_density_physics": rank_records(
                    records, "expanded_density_score", "density_physics_gate_pass"
                ),
                "expanded_pool": rank_records(records, "expanded_density_score"),
            }
            score_keys = {
                "original_confidence": "original_confidence_score",
                "original_density": "original_density_score",
                "expanded_confidence": "expanded_confidence_score",
                "expanded_density": "expanded_density_score",
                "expanded_density_physics": "expanded_density_score",
                "expanded_pool": "expanded_density_score",
            }
            selection_ready_seconds = time.perf_counter() - case_started
            for condition, ranked in ranked_sets.items():
                if condition == "expanded_pool":
                    limit = (
                        None if args.materialize_expanded_limit == 0
                        else args.materialize_expanded_limit
                    )
                elif condition == "original_confidence":
                    # This is the unexpanded pool used for the matched oracle
                    # control as well as confidence top-1.
                    limit = None
                else:
                    limit = 1
                materialize(
                    roots[condition], row_index, complex_name, ranked,
                    molecule_by_id, score_keys[condition], limit=limit,
                    copy_protein=limit == 1,
                )
                if condition != "expanded_pool":
                    selected_records.append({"condition": condition, **ranked[0]})

            generated = [record for record in records if not record["is_original"]]
            generated_density_pass = [row["density_gate_pass"] for row in generated]
            generated_density_physics_pass = [
                row["density_physics_gate_pass"] for row in generated
            ]
            generated_ncc_gain = [float(row["ncc_gain"]) for row in generated]
            audit_rows.append({
                "row_index": row_index,
                "source_row_index": source_row_index,
                "complex_name": complex_name,
                "num_original": len(original_records),
                "num_parents": len(parent_indices),
                "num_generated": len(generated),
                "num_candidates": len(records),
                "candidate_budget": case_candidate_budget,
                "candidate_budget_exhausted": bool(
                    case_candidate_budget is not None
                    and len(records) >= int(case_candidate_budget)
                ),
                "generated_density_pass_pct": (
                    100.0 * float(np.mean(generated_density_pass))
                    if generated else float("nan")
                ),
                "generated_density_physics_pass_pct": (
                    100.0 * float(np.mean(generated_density_physics_pass))
                    if generated else float("nan")
                ),
                "generated_ncc_improved_pct": (
                    100.0 * float(np.mean([value >= 0 for value in generated_ncc_gain]))
                    if generated else float("nan")
                ),
                "median_generated_ncc_gain": (
                    float(np.median(generated_ncc_gain))
                    if generated else float("nan")
                ),
                "best_generated_ncc_gain": (
                    float(max(generated_ncc_gain))
                    if generated else float("nan")
                ),
                "expanded_density_selected_new": not ranked_sets["expanded_density"][0]["is_original"],
                "expanded_density_physics_selected_new": not ranked_sets["expanded_density_physics"][0]["is_original"],
                "selection_ready_seconds": selection_ready_seconds,
                "wall_clock_budget_seconds": case_wall_clock_budget,
                "wall_clock_budget_exhausted": bool(
                    case_deadline is not None and time.perf_counter() >= case_deadline
                ),
                "case_runtime_seconds": time.perf_counter() - case_started,
            })
            all_records.extend(records)
            print(
                f"[{row_index + 1}/{len(manifest)}] {complex_name}: "
                f"{len(original_records)} original + {len(generated)} local variants"
            )
        except Exception as exc:
            failure = {"row_index": row_index, "source_row_index": source_row_index,
                       "complex_name": complex_name,
                       "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[{row_index + 1}/{len(manifest)}] FAILED {complex_name}: {exc}")
            if args.fail_on_error:
                raise

    if not all_records:
        raise RuntimeError("No complexes were processed")
    actual_total_candidates = sum(int(row["num_candidates"]) for row in audit_rows)
    if (
        args.total_candidate_budget is not None
        and actual_total_candidates != int(args.total_candidate_budget)
    ):
        raise RuntimeError(
            "Requested cohort candidate budget was not attained: "
            f"requested={args.total_candidate_budget}, actual={actual_total_candidates}. "
            "Use a budget no larger than the native candidate capacity."
        )
    pd.DataFrame(all_records).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(selected_records).to_csv(output / "selected.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "per_complex_audit.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    summary = {
        "experiment_id": args.experiment_id,
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "num_complexes": len(audit_rows),
        "num_failures": len(failures),
        "num_candidates": len(all_records),
        "requested_total_candidate_budget": args.total_candidate_budget,
        "actual_total_candidates": int(actual_total_candidates),
        "requested_total_wall_clock_budget_seconds": args.total_wall_clock_budget_seconds,
        "per_complex_wall_clock_budget_seconds": case_wall_clock_budget,
        "evaluation_manifest": str(output / "evaluation_manifest.csv"),
        "parent_score_column": args.parent_score_column,
        "parent_count": args.parent_count,
        "rotation_degrees": args.rotation_degrees,
        "translation_distances_A": args.translation_distances,
        "torsion_degrees": args.torsion_degrees,
        "max_torsion_bonds": args.max_torsion_bonds,
        "coarse_orientations": not args.no_coarse_orientations,
        "density_sigma_A": args.density_sigma,
        "density_weight": args.density_weight,
        "minimum_ncc_gain": args.minimum_ncc_gain,
        "relative_physics_gate": True,
        "uses_ground_truth": False,
        "runtime_definition": (
            "search/scoring/ranking plus benchmark materialization; per-complex "
            "selection_ready_seconds excludes materialization"
        ),
        "runtime_seconds": time.perf_counter() - run_started,
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "conditions": {name: str(path) for name, path in roots.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
