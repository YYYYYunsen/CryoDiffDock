#!/usr/bin/env python3
"""Run CryoDiffDock-Local V2 density-conditioned hierarchical search.

V2 preserves the V1 input contract and evaluation directory layout, but it
replaces fixed one-step enumeration with a bounded coarse-to-fine beam over
composable translation, rotation, and torsion moves.  Proposal directions are
computed from differentiable multi-scale Gaussian NCC.  No reference ligand or
RMSD is read anywhere in this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.adaptive_search import (
    AdaptivePose,
    MultiScaleNCCEvaluator,
    ProteinClashIndex,
    coarse_orientation_proposals,
    coordinate_fingerprint,
    deduplicate_poses,
    density_conditioned_proposals,
    fixed_direction_proposals,
    local_basin_statistics_batch,
)
from cryodiffdock_pocket.density_ranking import load_cartesian_density_crop, zscore
from cryodiffdock_pocket.clash_repair import directed_local_variants
from cryodiffdock_pocket.discrete_search import (
    centroid_distance,
    direct_rmsd,
    greedy_diverse_indices,
)
from cryodiffdock_pocket.physics import paired_protein_for_pose, read_protein_heavy_atoms
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


def safe_name(value: object) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    )


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
        raise ValueError("coordinate and molecule atom counts differ")
    conformer = molecule.GetConformer()
    for atom_index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, tuple(float(value) for value in xyz))
    return molecule


def write_sdf(molecule, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    try:
        # Keep large-coordinate cryo-EM structures parseable by avoiding the
        # fixed-width V2000 atom-coordinate fields.
        xyz = molecule_coordinates(molecule)
        if not np.isfinite(xyz).all():
            raise ValueError("Cannot write nonfinite ligand coordinates")
        if np.max(np.abs(xyz)) >= 900.0:
            writer.SetForceV3000(True)
        writer.write(molecule)
    finally:
        writer.close()


def choose_protein_path(pose_path: Path, manifest_row: pd.Series, base: Path) -> Path:
    paired = paired_protein_for_pose(pose_path)
    if paired is not None:
        return paired.resolve()
    for column in ("experimental_protein", "protein_path", "protein"):
        if column in manifest_row and pd.notna(manifest_row[column]):
            candidate = resolve_path(manifest_row[column], base)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No paired or manifest protein found for {pose_path}")


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if value == "cuda" or (value == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def select_beam(
    poses: list[AdaptivePose],
    eligible: dict[str, bool],
    *,
    beam_width: int,
    per_root_width: int,
    coarse_fraction: float,
    density_eligible: dict[str, bool] | None = None,
    steric_rescue_eligible: dict[str, bool] | None = None,
    steric_fraction: float = 0.0,
) -> list[AdaptivePose]:
    """Select a bounded beam with optional density/steric dual quotas.

    With ``steric_fraction == 0`` this follows the original V2 implementation
    byte-for-byte at the decision level.  CAAHS-A reserves part of the beam for
    candidates that reduce the severe-clash count while satisfying the density
    and displacement guards encoded in ``steric_rescue_eligible``.
    """

    ordered = sorted(
        (pose for pose in poses if eligible.get(pose.candidate_id, False)),
        key=lambda pose: (-float(pose.selection_score), pose.candidate_id),
    )
    if steric_fraction <= 0.0:
        root_counts: dict[str, int] = {}
        root_limited = []
        for pose in ordered:
            count = root_counts.get(pose.root_id, 0)
            if count >= int(per_root_width):
                continue
            root_counts[pose.root_id] = count + 1
            root_limited.append(pose)

        coarse_budget = min(
            int(beam_width), max(0, int(round(float(beam_width) * coarse_fraction)))
        )
        coarse = [pose for pose in root_limited if pose.branch_type == "coarse"]
        selected = coarse[:coarse_budget]
        selected_ids = {pose.candidate_id for pose in selected}
        for pose in root_limited:
            if pose.candidate_id in selected_ids:
                continue
            selected.append(pose)
            selected_ids.add(pose.candidate_id)
            if len(selected) >= int(beam_width):
                break
        return selected

    density_eligible = density_eligible or {}
    steric_rescue_eligible = steric_rescue_eligible or {}
    density_ordered = [
        pose for pose in ordered if density_eligible.get(pose.candidate_id, False)
    ]
    steric_ordered = sorted(
        (
            pose for pose in poses
            if steric_rescue_eligible.get(pose.candidate_id, False)
        ),
        key=lambda pose: (
            int(pose.severe_clash_count),
            float(pose.clashing_ligand_atom_fraction),
            -float(pose.selection_score),
            pose.candidate_id,
        ),
    )

    steric_budget = min(
        int(beam_width), max(0, int(round(float(beam_width) * steric_fraction)))
    )
    density_budget = int(beam_width) - steric_budget
    coarse_budget = min(
        density_budget, max(0, int(round(float(beam_width) * coarse_fraction)))
    )
    selected: list[AdaptivePose] = []
    selected_ids: set[str] = set()
    root_counts: dict[str, int] = {}

    def add_from(pool: list[AdaptivePose], limit: int) -> None:
        for pose in pool:
            if len(selected) >= limit or len(selected) >= int(beam_width):
                break
            if pose.candidate_id in selected_ids:
                continue
            count = root_counts.get(pose.root_id, 0)
            if count >= int(per_root_width):
                continue
            selected.append(pose)
            selected_ids.add(pose.candidate_id)
            root_counts[pose.root_id] = count + 1

    add_from(
        [pose for pose in density_ordered if pose.branch_type == "coarse"],
        coarse_budget,
    )
    add_from(density_ordered, density_budget)
    add_from(steric_ordered, density_budget + steric_budget)
    # If one quota cannot be filled, release the unused capacity to either path.
    add_from(ordered, int(beam_width))
    return selected


def clash_aware_parent_indices(
    originals: list[AdaptivePose],
    baseline_indices: list[int],
    *,
    parent_count: int,
    steric_slots: int,
    maximum_ncc_drop: float,
    minimum_rmsd_A: float,
) -> list[int]:
    """Reserve parent slots for low-clash, density-supported hypotheses."""

    if steric_slots <= 0:
        return list(baseline_indices)
    best_ncc = max(float(pose.multiscale_ncc) for pose in originals)
    guarded = [
        index for index, pose in enumerate(originals)
        if pose.multiscale_ncc >= best_ncc - float(maximum_ncc_drop)
    ]
    guarded.sort(key=lambda index: (
        int(originals[index].severe_clash_count),
        float(originals[index].clashing_ligand_atom_fraction),
        -float(originals[index].multiscale_ncc),
        -float(originals[index].selection_score),
        originals[index].candidate_id,
    ))
    reserve = min(int(parent_count), int(steric_slots), len(guarded))
    keep_density = max(0, int(parent_count) - reserve)
    selected = list(baseline_indices[:keep_density])

    def diverse(index: int) -> bool:
        xyz = originals[index].coordinates
        return all(
            direct_rmsd(xyz, originals[other].coordinates)
            >= float(minimum_rmsd_A)
            for other in selected
        )

    for index in guarded:
        if index not in selected and diverse(index):
            selected.append(index)
        if len(selected) >= int(parent_count):
            break
    for index in list(baseline_indices) + guarded:
        if index not in selected:
            selected.append(index)
        if len(selected) >= int(parent_count):
            break
    return selected


def clash_aware_final_ranking(
    final_ranked: list[dict],
    *,
    enabled: bool,
    maximum_ncc_drop: float,
    maximum_anchor_rmsd_A: float,
) -> tuple[list[dict], bool]:
    """Conservatively replace a clashing Top-1 only with a guarded clash-free pose."""

    if not enabled or not final_ranked or int(final_ranked[0]["severe_clash_count"]) == 0:
        return final_ranked, False
    baseline = final_ranked[0]
    rescues = [
        row for row in final_ranked
        if int(row["severe_clash_count"]) == 0
        and float(row["multiscale_ncc"])
        >= float(baseline["multiscale_ncc"]) - float(maximum_ncc_drop)
        and float(row["anchor_rmsd_A"]) <= float(maximum_anchor_rmsd_A)
    ]
    if not rescues:
        return final_ranked, False
    rescue = min(
        rescues,
        key=lambda row: (
            -float(row["final_selection_score"]),
            float(row["anchor_rmsd_A"]),
            row["candidate_id"],
        ),
    )
    if rescue["candidate_id"] == baseline["candidate_id"]:
        return final_ranked, False
    return [rescue] + [
        row for row in final_ranked if row["candidate_id"] != rescue["candidate_id"]
    ], True


def clash_directed_adaptive_proposals(
    parent: AdaptivePose,
    molecule,
    protein_coordinates: np.ndarray,
    protein_elements: list[str],
    ligand_elements: list[str],
    *,
    translation_distances_A: list[float],
    rotation_degrees: list[float],
    torsion_degrees: list[float],
    max_contact_directions: int,
    max_torsion_bonds: int,
    severe_overlap_A: float,
) -> list[AdaptivePose]:
    """Convert clash-field escape moves into hierarchical-search proposals."""

    parent_molecule = molecule_with_coordinates(molecule, parent.coordinates)
    variants = directed_local_variants(
        parent_molecule,
        protein_coordinates,
        protein_elements,
        ligand_elements,
        translation_distances_A=translation_distances_A,
        rotation_degrees=rotation_degrees,
        torsion_degrees=torsion_degrees,
        max_contact_directions=max_contact_directions,
        max_torsion_bonds=max_torsion_bonds,
        severe_overlap_A=severe_overlap_A,
    )
    proposals = []
    for variant in variants:
        move_id = f"b_{variant.variant_id}"
        proposals.append(AdaptivePose(
            candidate_id=f"{parent.candidate_id}__{move_id}",
            root_id=parent.root_id,
            parent_id=parent.candidate_id,
            coordinates=np.asarray(variant.coordinates, dtype=np.float64),
            depth=parent.depth + 1,
            move_type=variant.move_type,
            move_magnitude=float(variant.magnitude),
            axis_label=variant.axis_label,
            bond=variant.bond,
            branch_type="steric",
            move_sequence=(*parent.move_sequence, move_id),
        ))
    return deduplicate_poses(proposals)


def materialize(
    root: Path,
    row_index: int,
    complex_name: str,
    ranked: list[dict],
    root_templates: dict[str, object],
    pose_lookup: dict[str, AdaptivePose],
    *,
    score_key: str,
    limit: int | None,
    copy_protein: bool = False,
) -> None:
    directory = root / f"index{row_index}___{safe_name(complex_name)}"
    directory.mkdir(parents=True, exist_ok=True)
    selected = ranked if limit is None else ranked[:limit]
    rows = []
    for rank, record in enumerate(selected, start=1):
        score = float(record[score_key])
        destination = directory / f"rank{rank}_confidence{score:.8f}.sdf"
        if bool(record["is_original"]):
            shutil.copy2(record["prediction_path"], destination)
        else:
            pose = pose_lookup[record["candidate_id"]]
            write_sdf(
                molecule_with_coordinates(root_templates[pose.root_id], pose.coordinates),
                destination,
            )
        if copy_protein:
            protein = Path(str(record["protein_path"]))
            if protein.is_file():
                shutil.copy2(
                    protein,
                    destination.with_name(f"{destination.stem}_protein.pdb"),
                )
        rows.append({
            **record,
            "materialized_rank": int(rank),
            "materialized_path": str(destination),
        })
    pd.DataFrame(rows).to_csv(directory / "ranking.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path,
                        help="V1 pose_scores.csv for the baseline40+guided40 union")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", default="v2_main")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--complexes", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--density-sigmas", type=comma_floats,
                        default=comma_floats("1.5,2.25"))
    parser.add_argument("--density-weight", type=float, default=3.0)
    parser.add_argument("--basin-weight", type=float, default=0.25)
    parser.add_argument("--anchor-penalty", type=float, default=0.10)
    parser.add_argument("--coarse-anchor-penalty", type=float, default=0.0)
    parser.add_argument("--coarse-penalty", type=float, default=0.10)
    parser.add_argument("--clash-penalty", type=float, default=0.02)
    parser.add_argument(
        "--clash-aware-search", action="store_true",
        help="Enable Clash-Aware Adaptive Hierarchical Search, phase A (CAAHS-A)",
    )
    parser.add_argument("--steric-parent-slots", type=int, default=2)
    parser.add_argument("--steric-parent-max-ncc-drop", type=float, default=0.02)
    parser.add_argument("--steric-beam-fraction", type=float, default=0.25)
    parser.add_argument("--steric-rescue-max-ncc-drop", type=float, default=0.02)
    parser.add_argument("--steric-rescue-max-anchor-rmsd-A", type=float, default=0.75)
    parser.add_argument("--clash-aware-final-max-ncc-drop", type=float, default=0.02)
    parser.add_argument("--clash-aware-final-max-anchor-rmsd-A", type=float, default=0.75)
    parser.add_argument(
        "--clash-directed-proposals", action="store_true",
        help=(
            "Enable CAAHS-B clash-field translation, torque, and local torsion "
            "proposals in addition to the CAAHS-A density-guided proposals"
        ),
    )
    parser.add_argument(
        "--clash-translation-steps", type=comma_floats,
        default=comma_floats("0.10,0.20,0.35,0.50"),
    )
    parser.add_argument(
        "--clash-rotation-steps", type=comma_floats,
        default=comma_floats("2,5,10"),
    )
    parser.add_argument(
        "--clash-torsion-steps", type=comma_floats,
        default=comma_floats("5,10,20"),
    )
    parser.add_argument("--clash-contact-directions", type=int, default=4)
    parser.add_argument("--parent-count", type=int, default=6)
    parser.add_argument("--top-score-parents", type=int, default=3)
    parser.add_argument("--parent-score-column", choices=("confidence", "combined_score"),
                        default="confidence")
    parser.add_argument("--parent-diversity-rmsd", type=float, default=1.5)
    parser.add_argument("--search-depth", type=int, default=3,
                        help="Maximum number of composable moves after each root")
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--beam-width-per-parent", type=int, default=10)
    parser.add_argument("--coarse-beam-fraction", type=float, default=0.25)
    parser.add_argument("--translation-steps", type=comma_floats,
                        default=comma_floats("0.25,0.5"))
    parser.add_argument("--rotation-steps", type=comma_floats,
                        default=comma_floats("10,20"))
    parser.add_argument("--torsion-steps", type=comma_floats,
                        default=comma_floats("30,60"))
    parser.add_argument("--max-torsion-bonds", type=int, default=8)
    parser.add_argument("--top-torsion-gradients", type=int, default=2)
    parser.add_argument("--no-coarse-orientations", action="store_true")
    parser.add_argument("--no-combined-rigid", action="store_true")
    parser.add_argument(
        "--no-density-gradients",
        action="store_true",
        help=(
            "Replace NCC-gradient proposal directions with deterministic ligand "
            "principal-axis and first-rotor controls while retaining density scoring"
        ),
    )
    parser.add_argument("--minimum-step-ncc-gain", type=float, default=0.0)
    parser.add_argument("--coarse-minimum-ncc-gain", type=float, default=-0.01,
                        help="Small NCC tolerance that lets near-support coarse basins enter refinement")
    parser.add_argument("--max-centroid-shift", type=float, default=1.5)
    parser.add_argument("--min-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--clash-tolerance-count", type=int, default=0)
    parser.add_argument("--clash-tolerance-fraction", type=float, default=0.0)
    parser.add_argument("--severe-overlap", type=float, default=0.8)
    parser.add_argument("--mask-radius", type=float, default=10.0)
    parser.add_argument("--target-floor", type=float, default=None)
    parser.add_argument("--min-mask-voxels", type=int, default=64)
    parser.add_argument("--voxel-chunk-size", type=int, default=512)
    parser.add_argument("--pose-batch-size", type=int, default=32)
    parser.add_argument("--finalist-count", type=int, default=48)
    parser.add_argument("--basin-translation-step", type=float, default=0.20)
    parser.add_argument("--basin-rotation-step", type=float, default=7.5)
    parser.add_argument("--basin-support-tolerance", type=float, default=0.01)
    parser.add_argument("--materialize-expanded-limit", type=int, default=0,
                        help="0 writes the complete scored V2 pool for oracle evaluation")
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
        "--candidate-budget-reference-audit",
        type=Path,
        default=None,
        help=(
            "Optional full-run per_complex_audit.csv. Its native per-complex "
            "candidate counts are scaled by largest remainder so the matched "
            "cohort total preserves adaptive allocation without using RMSD."
        ),
    )
    parser.add_argument(
        "--total-wall-clock-budget-seconds",
        type=float,
        default=None,
        help=(
            "Cohort-wide search ceiling split evenly across complexes. The deadline "
            "stops new proposal stages; deterministic final ranking/materialization "
            "still completes and its observed time is recorded."
        ),
    )
    parser.add_argument(
        "--materialization-mode",
        choices=("benchmark", "top1", "interaction_diagnostic"),
        default="benchmark",
        help=(
            "benchmark writes all legacy ablation/oracle outputs; top1 writes only "
            "the final selected pose for deployment-style runtime measurement; "
            "interaction_diagnostic writes Top-1 plus every final selector finalist"
        ),
    )
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.parent_count <= 0 or not 0 <= args.top_score_parents <= args.parent_count:
        raise ValueError("invalid parent selection settings")
    if args.search_depth < 0 or args.beam_width <= 0 or args.beam_width_per_parent <= 0:
        raise ValueError("search depth must be non-negative and beam widths must be positive")
    if args.max_torsion_bonds < 0 or args.top_torsion_gradients < 0:
        raise ValueError("torsion limits must be non-negative")
    if not 0.0 <= args.coarse_beam_fraction <= 1.0:
        raise ValueError("coarse beam fraction must be in [0,1]")
    if not 0.0 <= args.steric_beam_fraction <= 1.0:
        raise ValueError("steric beam fraction must be in [0,1]")
    if not 0 <= args.steric_parent_slots <= args.parent_count:
        raise ValueError("steric parent slots must be between 0 and parent count")
    if min(
        args.steric_parent_max_ncc_drop,
        args.steric_rescue_max_ncc_drop,
        args.clash_aware_final_max_ncc_drop,
        args.steric_rescue_max_anchor_rmsd_A,
        args.clash_aware_final_max_anchor_rmsd_A,
    ) < 0:
        raise ValueError("CAAHS-A density/displacement guards must be non-negative")
    if not 0.0 <= args.basin_weight <= 1.0:
        raise ValueError("basin weight must be in [0,1]")
    if args.finalist_count <= 0:
        raise ValueError("finalist count must be positive")
    if args.clash_directed_proposals and not args.clash_aware_search:
        raise ValueError("clash-directed proposals require --clash-aware-search")
    if args.clash_contact_directions < 1:
        raise ValueError("clash-contact-directions must be positive")

    started = time.perf_counter()
    device = choose_device(args.device)
    manifest_path = args.manifest.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for label, path in (("manifest", manifest_path), ("scores", scores_path)):
        if not path.is_file():
            kind = "directory" if path.is_dir() else "missing path"
            raise FileNotFoundError(
                f"--{label} must point to a CSV file, but received {kind}: {path}. "
                f"If a shell variable was used, verify it with: printf '<%s>\\n' "
                f"\"${{{label.upper()}}}\""
            )
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    scores = pd.read_csv(scores_path)
    required_manifest = {"complex_name", "ligand_density_crop"}
    required_scores = {
        "row_index", "complex_name", "source", "source_rank", "confidence",
        "prediction_path",
    }
    missing = sorted(required_manifest - set(manifest.columns))
    missing_scores = sorted(required_scores - set(scores.columns))
    if missing or missing_scores:
        raise ValueError(f"Missing manifest columns {missing}; score columns {missing_scores}")

    manifest = manifest.copy()
    manifest["_source_row_index"] = np.arange(len(manifest), dtype=int)
    if args.complexes:
        requested = [item.strip() for item in args.complexes.split(",") if item.strip()]
        order = {name: index for index, name in enumerate(requested)}
        absent = sorted(set(requested) - set(manifest["complex_name"].astype(str)))
        if absent:
            raise ValueError(f"Requested complexes are absent: {absent}")
        manifest = manifest[manifest["complex_name"].astype(str).isin(requested)]
        manifest = manifest.assign(
            _requested_order=manifest["complex_name"].astype(str).map(order)
        ).sort_values("_requested_order", kind="stable").drop(columns="_requested_order")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        manifest = manifest.head(args.limit)
    manifest = manifest.reset_index(drop=True)
    score_row_indices = pd.to_numeric(scores["row_index"], errors="raise").astype(int)
    manifest_names = manifest["complex_name"].astype(str)
    score_names = scores["complex_name"].astype(str)
    row_index_alignment_valid = True
    for source_index, complex_name in zip(
        manifest["_source_row_index"].astype(int), manifest_names
    ):
        matched_names = score_names[score_row_indices.eq(int(source_index))]
        if matched_names.empty or not matched_names.eq(str(complex_name)).all():
            row_index_alignment_valid = False
            break
    name_alignment_valid = bool(
        not manifest_names.duplicated().any()
        and set(manifest_names).issubset(set(score_names))
    )
    if row_index_alignment_valid:
        score_match_mode = "row_index"
        original_counts = [
            int(score_row_indices.eq(int(source_index)).sum())
            for source_index in manifest["_source_row_index"]
        ]
    elif name_alignment_valid:
        score_match_mode = "complex_name"
        score_name_counts = score_names.value_counts()
        original_counts = [int(score_name_counts.get(name, 0)) for name in manifest_names]
    else:
        raise ValueError(
            "Scores cannot be aligned to the manifest by row_index or unique "
            "complex_name. The merged file may need an explicit branch identifier."
        )
    if any(count == 0 for count in original_counts):
        missing_rows = [
            int(source_index)
            for source_index, count in zip(manifest["_source_row_index"], original_counts)
            if count == 0
        ]
        raise FileNotFoundError(f"No scored poses for source rows: {missing_rows[:10]}")
    native_capacities = None
    candidate_budget_reference = None
    if args.candidate_budget_reference_audit is not None:
        if args.total_candidate_budget is None:
            raise ValueError(
                "candidate-budget-reference-audit requires total-candidate-budget"
            )
        candidate_budget_reference = (
            args.candidate_budget_reference_audit.expanduser().resolve()
        )
        reference_audit = pd.read_csv(
            candidate_budget_reference, dtype={"complex_name": str}
        )
        if "complex_name" not in reference_audit:
            raise ValueError("Candidate-budget reference audit lacks complex_name")
        reference_audit["complex_name"] = (
            reference_audit["complex_name"].astype(str).str.upper()
        )
        if reference_audit["complex_name"].duplicated().any():
            raise ValueError("Candidate-budget reference audit has duplicate complexes")
        if "num_candidates" not in reference_audit:
            if {"num_original", "num_generated"} <= set(reference_audit.columns):
                reference_audit["num_candidates"] = (
                    pd.to_numeric(reference_audit["num_original"], errors="raise")
                    + pd.to_numeric(reference_audit["num_generated"], errors="raise")
                )
            else:
                raise ValueError(
                    "Candidate-budget reference audit needs num_candidates or "
                    "num_original+num_generated"
                )
        reference_index = reference_audit.set_index("complex_name")
        manifest_names = manifest["complex_name"].astype(str).str.upper().tolist()
        missing_reference = [name for name in manifest_names if name not in reference_index.index]
        if missing_reference:
            raise ValueError(
                "Candidate-budget reference audit is missing complexes: "
                f"{missing_reference[:10]}"
            )
        native_capacities = pd.to_numeric(
            reference_index.loc[manifest_names, "num_candidates"], errors="raise"
        ).astype(int).tolist()
    case_candidate_budgets = allocate_total_candidate_budget(
        args.total_candidate_budget, original_counts, capacities=native_capacities
    )
    case_wall_clock_budget = per_complex_wall_clock_budget(
        args.total_wall_clock_budget_seconds, len(manifest)
    )
    manifest.drop(columns="_source_row_index").to_csv(
        output / "evaluation_manifest.csv", index=False
    )

    if args.materialization_mode == "interaction_diagnostic":
        condition_names = ("interaction_pool", "v2_selected")
    else:
        condition_names = (
            "original_confidence", "original_density", "expanded_confidence",
            "expanded_density", "expanded_density_physics", "expanded_pool",
            "v2_selected",
        )
    roots = {name: output / name for name in condition_names}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    selected_records: list[dict] = []
    audit_rows: list[dict] = []
    failures: list[dict] = []
    protein_cache: dict[str, tuple[np.ndarray, list[str], ProteinClashIndex]] = {}
    timing = {"density_seconds": 0.0, "gradient_seconds": 0.0,
              "physics_seconds": 0.0, "basin_seconds": 0.0,
              "materialization_seconds": 0.0,
              "top1_materialization_seconds": 0.0,
              "oracle_control_materialization_seconds": 0.0}

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
            if score_match_mode == "row_index":
                group = scores[score_row_indices.eq(source_row_index)].copy()
            else:
                group = scores[score_names.eq(complex_name)].copy()
            if group.empty:
                raise FileNotFoundError(
                    f"No scored poses for {complex_name} using {score_match_mode}"
                )
            crop = load_cartesian_density_crop(resolve_path(
                manifest_row["ligand_density_crop"], manifest_path.parent
            ))

            root_templates: dict[str, object] = {}
            root_metadata: dict[str, dict] = {}
            originals: list[AdaptivePose] = []
            source_coordinates = []
            atomic_numbers = None
            ligand_elements = None
            for source_position, (_, source) in enumerate(group.iterrows()):
                pose_path = Path(str(source["prediction_path"])).expanduser().resolve()
                molecule = read_molecule(pose_path)
                xyz = molecule_coordinates(molecule)
                numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
                elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                if atomic_numbers is None:
                    atomic_numbers, ligand_elements = numbers, elements
                elif numbers != atomic_numbers:
                    raise ValueError("original candidate atom ordering/composition differs")
                candidate_id = (
                    f"orig_{safe_name(source['source'])}_{int(source['source_rank']):03d}"
                )
                if candidate_id in root_templates:
                    candidate_id += f"_{source_position}"
                protein_path = choose_protein_path(pose_path, manifest_row, manifest_path.parent)
                root_templates[candidate_id] = molecule
                root_metadata[candidate_id] = {
                    "source": str(source["source"]),
                    "source_rank": int(source["source_rank"]),
                    "confidence": float(source["confidence"]),
                    "prediction_path": str(pose_path),
                    "protein_path": str(protein_path),
                }
                originals.append(AdaptivePose(
                    candidate_id=candidate_id,
                    root_id=candidate_id,
                    parent_id=candidate_id,
                    coordinates=xyz,
                    depth=0,
                    move_type="original",
                    move_magnitude=0.0,
                    branch_type="original",
                ))
                source_coordinates.append(xyz)

            evaluator = MultiScaleNCCEvaluator(
                crop, atomic_numbers, sigmas_A=args.density_sigmas,
                mask_radius_A=args.mask_radius, target_floor=args.target_floor,
                min_mask_voxels=args.min_mask_voxels, device=device,
                voxel_chunk_size=args.voxel_chunk_size,
                pose_batch_size=args.pose_batch_size,
            )
            density_started = time.perf_counter()
            original_scores = evaluator.score(source_coordinates)
            timing["density_seconds"] += time.perf_counter() - density_started
            for pose, values in zip(originals, original_scores):
                pose.ncc_by_sigma = tuple(float(value) for value in values)
                pose.primary_ncc = float(values[0])
                pose.multiscale_ncc = float(np.mean(values))
                pose.in_bounds_fraction = evaluator.in_bounds_fraction(pose.coordinates)

            confidences = np.asarray(
                [root_metadata[pose.root_id]["confidence"] for pose in originals],
                dtype=np.float64,
            )
            confidence_z = zscore(confidences)
            original_density = np.asarray(
                [pose.multiscale_ncc for pose in originals], dtype=np.float64
            )
            density_mean = float(np.mean(original_density))
            density_std = float(np.std(original_density))
            if density_std <= 1e-12:
                density_std = 1.0
            density_z = (original_density - density_mean) / density_std
            for pose, confidence_value, density_value in zip(
                originals, confidence_z, density_z
            ):
                root_metadata[pose.root_id]["confidence_z"] = float(confidence_value)
                pose.selection_score = float(
                    confidence_value + args.density_weight * density_value
                )

            if args.parent_score_column == "combined_score":
                parent_scores = [pose.selection_score for pose in originals]
            else:
                parent_scores = confidences.tolist()
            baseline_parent_indices = greedy_diverse_indices(
                source_coordinates, parent_scores, count=args.parent_count,
                top_score_count=args.top_score_parents,
                minimum_rmsd_A=args.parent_diversity_rmsd,
            )

            clash_indices: dict[str, ProteinClashIndex] = {}
            root_clashes: dict[str, tuple[int, float]] = {}
            physics_started = time.perf_counter()
            roots_to_score = (
                originals
                if args.clash_aware_search
                else [originals[index] for index in baseline_parent_indices]
            )
            for root_pose in roots_to_score:
                protein_path = root_metadata[root_pose.root_id]["protein_path"]
                if protein_path not in protein_cache:
                    protein_xyz, protein_elements = read_protein_heavy_atoms(protein_path)
                    protein_cache[protein_path] = (
                        protein_xyz, protein_elements,
                        ProteinClashIndex(protein_xyz, protein_elements),
                    )
                clash_index = protein_cache[protein_path][2]
                clash_indices[root_pose.root_id] = clash_index
                root_clashes[root_pose.root_id] = clash_index.score(
                    root_pose.coordinates, ligand_elements,
                    severe_overlap_A=args.severe_overlap,
                )
                root_pose.severe_clash_count, root_pose.clashing_ligand_atom_fraction = (
                    root_clashes[root_pose.root_id]
                )
            timing["physics_seconds"] += time.perf_counter() - physics_started

            parent_indices = clash_aware_parent_indices(
                originals,
                baseline_parent_indices,
                parent_count=args.parent_count,
                steric_slots=(args.steric_parent_slots if args.clash_aware_search else 0),
                maximum_ncc_drop=args.steric_parent_max_ncc_drop,
                minimum_rmsd_A=args.parent_diversity_rmsd,
            )
            parents = [originals[index] for index in parent_indices]
            # A diversity fallback can add an unscored root only in legacy mode;
            # make its spatial index available before proposal evaluation.
            for parent in parents:
                if parent.root_id in root_clashes:
                    continue
                protein_path = root_metadata[parent.root_id]["protein_path"]
                if protein_path not in protein_cache:
                    protein_xyz, protein_elements = read_protein_heavy_atoms(protein_path)
                    protein_cache[protein_path] = (
                        protein_xyz, protein_elements,
                        ProteinClashIndex(protein_xyz, protein_elements),
                    )
                clash_indices[parent.root_id] = protein_cache[protein_path][2]
                root_clashes[parent.root_id] = clash_indices[parent.root_id].score(
                    parent.coordinates, ligand_elements,
                    severe_overlap_A=args.severe_overlap,
                )
                parent.severe_clash_count, parent.clashing_ligand_atom_fraction = (
                    root_clashes[parent.root_id]
                )

            pose_lookup = {pose.candidate_id: pose for pose in originals}
            all_generated: list[AdaptivePose] = []
            eligible: dict[str, bool] = {pose.candidate_id: True for pose in originals}
            density_eligible: dict[str, bool] = {
                pose.candidate_id: True for pose in originals
            }
            steric_rescue_eligible: dict[str, bool] = {
                pose.candidate_id: False for pose in originals
            }
            seen = {
                (pose.root_id, coordinate_fingerprint(pose.coordinates))
                for pose in originals
            }

            def process_proposals(proposals: list[AdaptivePose]) -> list[AdaptivePose]:
                unique = []
                for pose in deduplicate_poses(proposals):
                    key = (pose.root_id, coordinate_fingerprint(pose.coordinates))
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(pose)
                if case_candidate_budget is not None:
                    remaining = max(
                        0,
                        int(case_candidate_budget) - len(originals) - len(all_generated),
                    )
                    unique = unique[:remaining]
                if not unique:
                    return []
                density_start = time.perf_counter()
                values = evaluator.score([pose.coordinates for pose in unique])
                timing["density_seconds"] += time.perf_counter() - density_start
                physics_start = time.perf_counter()
                for pose, local_values in zip(unique, values):
                    pose.ncc_by_sigma = tuple(float(value) for value in local_values)
                    pose.primary_ncc = float(local_values[0])
                    pose.multiscale_ncc = float(np.mean(local_values))
                    pose.in_bounds_fraction = evaluator.in_bounds_fraction(pose.coordinates)
                    pose.severe_clash_count, pose.clashing_ligand_atom_fraction = (
                        clash_indices[pose.root_id].score(
                            pose.coordinates, ligand_elements,
                            severe_overlap_A=args.severe_overlap,
                        )
                    )
                    parent = pose_lookup[pose.parent_id]
                    root = pose_lookup[pose.root_id]
                    step_gain = pose.multiscale_ncc - parent.multiscale_ncc
                    minimum_gain = (
                        args.coarse_minimum_ncc_gain
                        if pose.move_type == "coarse_orientation"
                        else args.minimum_step_ncc_gain
                    )
                    anchor = direct_rmsd(root.coordinates, pose.coordinates)
                    centroid = centroid_distance(root.coordinates, pose.coordinates)
                    root_count, root_fraction = root_clashes[pose.root_id]
                    excess = max(pose.severe_clash_count - root_count, 0)
                    anchor_weight = (
                        args.coarse_anchor_penalty
                        if pose.branch_type == "coarse" else args.anchor_penalty
                    )
                    density_value = (pose.multiscale_ncc - density_mean) / density_std
                    pose.selection_score = float(
                        root_metadata[pose.root_id]["confidence_z"]
                        + args.density_weight * density_value
                        - anchor_weight * anchor
                        - args.coarse_penalty * float(pose.branch_type == "coarse")
                        - args.clash_penalty * math.log1p(excess)
                    )
                    density_path_pass = bool(
                        step_gain >= minimum_gain
                        and centroid <= args.max_centroid_shift
                        and pose.in_bounds_fraction >= args.min_in_bounds_fraction
                    )
                    steric_path_pass = bool(
                        args.clash_aware_search
                        and pose.severe_clash_count < parent.severe_clash_count
                        and pose.clashing_ligand_atom_fraction
                        <= parent.clashing_ligand_atom_fraction
                        and pose.multiscale_ncc
                        >= root.multiscale_ncc - args.steric_rescue_max_ncc_drop
                        and anchor <= args.steric_rescue_max_anchor_rmsd_A
                        and centroid <= args.max_centroid_shift
                        and pose.in_bounds_fraction >= args.min_in_bounds_fraction
                    )
                    density_eligible[pose.candidate_id] = density_path_pass
                    steric_rescue_eligible[pose.candidate_id] = steric_path_pass
                    eligible[pose.candidate_id] = bool(
                        density_path_pass or steric_path_pass
                    )
                    pose_lookup[pose.candidate_id] = pose
                timing["physics_seconds"] += time.perf_counter() - physics_start
                all_generated.extend(unique)
                return unique

            def caahs_b_proposals(parent: AdaptivePose) -> list[AdaptivePose]:
                if not args.clash_directed_proposals:
                    return []
                if int(parent.severe_clash_count) <= 0:
                    return []
                protein_path = root_metadata[parent.root_id]["protein_path"]
                protein_xyz, protein_elements, _ = protein_cache[protein_path]
                return clash_directed_adaptive_proposals(
                    parent,
                    root_templates[parent.root_id],
                    protein_xyz,
                    protein_elements,
                    ligand_elements,
                    translation_distances_A=args.clash_translation_steps,
                    rotation_degrees=args.clash_rotation_steps,
                    torsion_degrees=args.clash_torsion_steps,
                    max_contact_directions=args.clash_contact_directions,
                    max_torsion_bonds=args.max_torsion_bonds,
                    severe_overlap_A=args.severe_overlap,
                )

            first_proposals = []
            if args.search_depth >= 1 and (
                case_deadline is None or time.perf_counter() < case_deadline
            ):
                if args.no_density_gradients:
                    parent_gradients = [None] * len(parents)
                else:
                    gradient_started = time.perf_counter()
                    _, parent_gradients = evaluator.score_and_gradient(
                        [parent.coordinates for parent in parents]
                    )
                    timing["gradient_seconds"] += time.perf_counter() - gradient_started
                for parent, gradient in zip(parents, parent_gradients):
                    proposal_kwargs = dict(
                        translation_steps_A=args.translation_steps,
                        rotation_steps_degrees=args.rotation_steps,
                        torsion_steps_degrees=args.torsion_steps,
                        max_torsion_bonds=args.max_torsion_bonds,
                        top_torsion_gradients=args.top_torsion_gradients,
                        include_combined_rigid=not args.no_combined_rigid,
                    )
                    if args.no_density_gradients:
                        guided, diagnostics = fixed_direction_proposals(
                            parent, root_templates[parent.root_id], **proposal_kwargs
                        )
                    else:
                        guided, diagnostics = density_conditioned_proposals(
                            parent, gradient, root_templates[parent.root_id],
                            **proposal_kwargs,
                        )
                    parent.translation_gradient_norm = diagnostics["translation_gradient_norm"]
                    parent.rotation_gradient_norm = diagnostics["rotation_gradient_norm"]
                    parent.maximum_torsion_gradient = diagnostics["maximum_torsion_gradient"]
                    first_proposals.extend(guided)
                    first_proposals.extend(caahs_b_proposals(parent))
                    first_proposals.extend(coarse_orientation_proposals(
                        parent, include_cube_group=not args.no_coarse_orientations
                    ))
            stage = process_proposals(first_proposals)
            beam = select_beam(
                stage, eligible, beam_width=args.beam_width,
                per_root_width=args.beam_width_per_parent,
                coarse_fraction=args.coarse_beam_fraction,
                density_eligible=density_eligible,
                steric_rescue_eligible=steric_rescue_eligible,
                steric_fraction=(
                    args.steric_beam_fraction if args.clash_aware_search else 0.0
                ),
            )
            stage_sizes = [len(stage)] if args.search_depth >= 1 else []
            beam_sizes = [len(beam)] if args.search_depth >= 1 else []

            for _depth in range(2, int(args.search_depth) + 1):
                budget_full = (
                    case_candidate_budget is not None
                    and len(originals) + len(all_generated) >= int(case_candidate_budget)
                )
                deadline_reached = (
                    case_deadline is not None and time.perf_counter() >= case_deadline
                )
                if not beam or budget_full or deadline_reached:
                    stage_sizes.append(0)
                    beam_sizes.append(0)
                    break
                if args.no_density_gradients:
                    gradients = [None] * len(beam)
                else:
                    gradient_started = time.perf_counter()
                    _, gradients = evaluator.score_and_gradient(
                        [pose.coordinates for pose in beam]
                    )
                    timing["gradient_seconds"] += time.perf_counter() - gradient_started
                proposals = []
                for parent, gradient in zip(beam, gradients):
                    proposal_kwargs = dict(
                        translation_steps_A=args.translation_steps,
                        rotation_steps_degrees=args.rotation_steps,
                        torsion_steps_degrees=args.torsion_steps,
                        max_torsion_bonds=args.max_torsion_bonds,
                        top_torsion_gradients=args.top_torsion_gradients,
                        include_combined_rigid=not args.no_combined_rigid,
                    )
                    if args.no_density_gradients:
                        guided, diagnostics = fixed_direction_proposals(
                            parent, root_templates[parent.root_id], **proposal_kwargs
                        )
                    else:
                        guided, diagnostics = density_conditioned_proposals(
                            parent, gradient, root_templates[parent.root_id],
                            **proposal_kwargs,
                        )
                    parent.translation_gradient_norm = diagnostics["translation_gradient_norm"]
                    parent.rotation_gradient_norm = diagnostics["rotation_gradient_norm"]
                    parent.maximum_torsion_gradient = diagnostics["maximum_torsion_gradient"]
                    proposals.extend(guided)
                    proposals.extend(caahs_b_proposals(parent))
                stage = process_proposals(proposals)
                beam = select_beam(
                    stage, eligible, beam_width=args.beam_width,
                    per_root_width=args.beam_width_per_parent,
                    coarse_fraction=args.coarse_beam_fraction,
                    density_eligible=density_eligible,
                    steric_rescue_eligible=steric_rescue_eligible,
                    steric_fraction=(
                        args.steric_beam_fraction if args.clash_aware_search else 0.0
                    ),
                )
                stage_sizes.append(len(stage))
                beam_sizes.append(len(beam))

            candidate_pool = originals + all_generated
            preliminary = sorted(
                (pose for pose in candidate_pool if eligible.get(pose.candidate_id, False)),
                key=lambda pose: (-float(pose.selection_score), pose.candidate_id),
            )
            if args.clash_aware_search:
                steric_finalist_budget = min(
                    args.finalist_count,
                    max(0, int(round(args.finalist_count * args.steric_beam_fraction))),
                )
                density_finalists = [
                    pose for pose in preliminary
                    if density_eligible.get(pose.candidate_id, False)
                ][: args.finalist_count - steric_finalist_budget]
                finalist_ids = {pose.candidate_id for pose in density_finalists}
                steric_finalists = sorted(
                    (
                        pose for pose in candidate_pool
                        if steric_rescue_eligible.get(pose.candidate_id, False)
                        and pose.candidate_id not in finalist_ids
                    ),
                    key=lambda pose: (
                        int(pose.severe_clash_count),
                        float(pose.clashing_ligand_atom_fraction),
                        -float(pose.selection_score),
                        pose.candidate_id,
                    ),
                )[:steric_finalist_budget]
                finalists = density_finalists + steric_finalists
                finalist_ids = {pose.candidate_id for pose in finalists}
                for pose in preliminary:
                    if len(finalists) >= args.finalist_count:
                        break
                    if pose.candidate_id not in finalist_ids:
                        finalists.append(pose)
                        finalist_ids.add(pose.candidate_id)
            else:
                finalists = preliminary[: args.finalist_count]
            best_original = max(originals, key=lambda pose: pose.selection_score)
            if best_original.candidate_id not in {pose.candidate_id for pose in finalists}:
                finalists.append(best_original)
            basin_started = time.perf_counter()
            basin_values = (
                local_basin_statistics_batch(
                    finalists, evaluator,
                    translation_step_A=args.basin_translation_step,
                    rotation_step_degrees=args.basin_rotation_step,
                    support_tolerance=args.basin_support_tolerance,
                )
                if args.basin_weight > 0
                else {
                    pose.candidate_id: (pose.multiscale_ncc, 1.0)
                    for pose in finalists
                }
            )
            for pose in finalists:
                pose.basin_q25_ncc, pose.basin_support_fraction = (
                    basin_values[pose.candidate_id]
                )
            timing["basin_seconds"] += time.perf_counter() - basin_started

            final_scores: dict[str, float] = {}
            for pose in finalists:
                central_z = (pose.multiscale_ncc - density_mean) / density_std
                basin_z = (pose.basin_q25_ncc - density_mean) / density_std
                final_scores[pose.candidate_id] = float(
                    pose.selection_score
                    + args.density_weight * args.basin_weight * (basin_z - central_z)
                )

            records = []
            for pose in candidate_pool:
                metadata = root_metadata[pose.root_id]
                root = pose_lookup[pose.root_id]
                parent = pose_lookup[pose.parent_id]
                root_count, root_fraction = root_clashes.get(
                    pose.root_id,
                    (pose.severe_clash_count, pose.clashing_ligand_atom_fraction),
                )
                if pose.move_type == "original" and pose.root_id not in root_clashes:
                    protein_path = metadata["protein_path"]
                    if protein_path not in protein_cache:
                        protein_xyz, protein_elements = read_protein_heavy_atoms(protein_path)
                        protein_cache[protein_path] = (
                            protein_xyz, protein_elements,
                            ProteinClashIndex(protein_xyz, protein_elements),
                        )
                    pose.severe_clash_count, pose.clashing_ligand_atom_fraction = (
                        protein_cache[protein_path][2].score(
                            pose.coordinates, ligand_elements,
                            severe_overlap_A=args.severe_overlap,
                        )
                    )
                    root_count, root_fraction = (
                        pose.severe_clash_count, pose.clashing_ligand_atom_fraction
                    )
                relative_physics = bool(
                    pose.severe_clash_count <= root_count + args.clash_tolerance_count
                    and pose.clashing_ligand_atom_fraction
                    <= root_fraction + args.clash_tolerance_fraction
                )
                multiscale_z = (pose.multiscale_ncc - density_mean) / density_std
                anchor = direct_rmsd(root.coordinates, pose.coordinates)
                centroid = centroid_distance(root.coordinates, pose.coordinates)
                step_gain = pose.multiscale_ncc - parent.multiscale_ncc
                total_gain = pose.multiscale_ncc - root.multiscale_ncc
                record = {
                    "row_index": int(row_index),
                    "source_row_index": source_row_index,
                    "complex_name": complex_name,
                    "candidate_id": pose.candidate_id,
                    "root_id": pose.root_id,
                    "parent_id": pose.parent_id,
                    "is_original": pose.move_type == "original",
                    "source": metadata["source"],
                    "source_rank": metadata["source_rank"],
                    "depth": pose.depth,
                    "branch_type": pose.branch_type,
                    "move_type": pose.move_type,
                    "move_magnitude": pose.move_magnitude,
                    "axis_label": pose.axis_label,
                    "bond": "" if pose.bond is None else f"{pose.bond[0]}-{pose.bond[1]}",
                    "move_sequence": "|".join(pose.move_sequence),
                    "confidence": metadata["confidence"],
                    "confidence_z": metadata["confidence_z"],
                    "ncc": pose.primary_ncc,
                    "multiscale_ncc": pose.multiscale_ncc,
                    "expanded_density_z": multiscale_z,
                    "root_ncc": root.multiscale_ncc,
                    "parent_ncc": parent.multiscale_ncc,
                    "ncc_gain": total_gain,
                    "step_ncc_gain": step_gain,
                    "in_bounds_fraction": pose.in_bounds_fraction,
                    "anchor_rmsd_A": anchor,
                    "centroid_shift_A": centroid,
                    "translation_gradient_norm": pose.translation_gradient_norm,
                    "rotation_gradient_norm": pose.rotation_gradient_norm,
                    "maximum_torsion_gradient": pose.maximum_torsion_gradient,
                    "severe_clash_count": pose.severe_clash_count,
                    "step_parent_severe_clash_count": parent.severe_clash_count,
                    "parent_severe_clash_count": root_count,
                    "clashing_ligand_atom_fraction": pose.clashing_ligand_atom_fraction,
                    "step_parent_clashing_ligand_atom_fraction": (
                        parent.clashing_ligand_atom_fraction
                    ),
                    "parent_clashing_ligand_atom_fraction": root_fraction,
                    "relative_physics_pass": relative_physics,
                    "density_gate_pass": density_eligible.get(
                        pose.candidate_id, False
                    ),
                    "steric_rescue_gate_pass": steric_rescue_eligible.get(
                        pose.candidate_id, False
                    ),
                    "clash_aware_gate_pass": eligible.get(pose.candidate_id, False),
                    "density_physics_gate_pass": (
                        density_eligible.get(pose.candidate_id, False)
                        and relative_physics
                    ),
                    "selection_score": pose.selection_score,
                    "expanded_density_score": pose.selection_score,
                    "expanded_confidence_score": (
                        metadata["confidence_z"] - args.anchor_penalty * anchor
                    ),
                    "basin_q25_ncc": pose.basin_q25_ncc,
                    "basin_support_fraction": pose.basin_support_fraction,
                    "v2_finalist": pose.candidate_id in final_scores,
                    "final_selection_score": final_scores.get(pose.candidate_id, float("nan")),
                    "prediction_path": metadata["prediction_path"] if pose.move_type == "original" else "",
                    "protein_path": metadata["protein_path"],
                }
                for sigma, value in zip(args.density_sigmas, pose.ncc_by_sigma):
                    label = f"{float(sigma):g}".replace(".", "p")
                    record[f"ncc_sigma_{label}"] = value
                records.append(record)

            original_records = [record for record in records if record["is_original"]]
            finalists_records = [
                record for record in records if record["candidate_id"] in final_scores
            ]
            for record in finalists_records:
                record["final_selection_score"] = final_scores[record["candidate_id"]]
            final_ranked = sorted(
                finalists_records,
                key=lambda row: (-float(row["final_selection_score"]), row["candidate_id"]),
            )
            physics_ranked = sorted(
                [row for row in finalists_records if row["relative_physics_pass"] or row["is_original"]],
                key=lambda row: (-float(row["final_selection_score"]), row["candidate_id"]),
            )
            if not final_ranked or not physics_ranked:
                raise RuntimeError("V2 produced no final eligible candidates")
            baseline_selected = final_ranked[0]
            clash_aware_ranked, clash_aware_switched = clash_aware_final_ranking(
                final_ranked,
                enabled=args.clash_aware_search,
                maximum_ncc_drop=args.clash_aware_final_max_ncc_drop,
                maximum_anchor_rmsd_A=args.clash_aware_final_max_anchor_rmsd_A,
            )
            clash_aware_selected = clash_aware_ranked[0]
            for record in records:
                record["baseline_v2_selected"] = bool(
                    record["candidate_id"] == baseline_selected["candidate_id"]
                )
                record["caahs_a_selected"] = bool(
                    record["candidate_id"] == clash_aware_selected["candidate_id"]
                )
                record["caahs_b_selected"] = bool(
                    args.clash_directed_proposals
                    and record["candidate_id"] == clash_aware_selected["candidate_id"]
                )
            ranked_sets = {
                "original_confidence": sorted(
                    original_records, key=lambda row: (-row["confidence"], row["candidate_id"])
                ),
                "original_density": sorted(
                    original_records, key=lambda row: (-row["selection_score"], row["candidate_id"])
                ),
                "expanded_confidence": sorted(
                    records, key=lambda row: (-row["expanded_confidence_score"], row["candidate_id"])
                ),
                "expanded_density": final_ranked,
                "expanded_density_physics": physics_ranked,
                "expanded_pool": sorted(
                    records, key=lambda row: (-row["selection_score"], row["candidate_id"])
                ),
                "interaction_pool": final_ranked,
                # Keep the historical directory name so all downstream evaluation
                # scripts remain compatible; summary/audit identify CAAHS-A.
                "v2_selected": clash_aware_ranked,
            }
            score_keys = {
                "original_confidence": "confidence",
                "original_density": "selection_score",
                "expanded_confidence": "expanded_confidence_score",
                "expanded_density": "final_selection_score",
                "expanded_density_physics": "final_selection_score",
                "expanded_pool": "selection_score",
                "interaction_pool": "final_selection_score",
                "v2_selected": "final_selection_score",
            }
            selection_ready_seconds = time.perf_counter() - case_started
            materialize_started = time.perf_counter()
            top1_materialization_seconds = 0.0
            oracle_control_materialization_seconds = 0.0
            if args.materialization_mode == "top1":
                materialization_items = [("v2_selected", ranked_sets["v2_selected"])]
            elif args.materialization_mode == "interaction_diagnostic":
                materialization_items = [
                    ("interaction_pool", ranked_sets["interaction_pool"]),
                    ("v2_selected", ranked_sets["v2_selected"]),
                ]
            else:
                materialization_items = [
                    (condition, ranked_sets[condition]) for condition in condition_names
                ]
            for condition, ranked in materialization_items:
                if condition == "interaction_pool":
                    limit = None
                elif condition == "expanded_pool":
                    limit = None if args.materialize_expanded_limit == 0 else args.materialize_expanded_limit
                elif condition == "original_confidence":
                    limit = None
                else:
                    limit = 1
                condition_materialize_started = time.perf_counter()
                materialize(
                    roots[condition], row_index, complex_name, ranked,
                    root_templates, pose_lookup, score_key=score_keys[condition],
                    limit=limit, copy_protein=limit == 1,
                )
                condition_materialize_seconds = (
                    time.perf_counter() - condition_materialize_started
                )
                if condition == "v2_selected":
                    top1_materialization_seconds += condition_materialize_seconds
                else:
                    oracle_control_materialization_seconds += condition_materialize_seconds
                if condition not in {"expanded_pool", "v2_selected"}:
                    selected_records.append({"condition": condition, **ranked[0]})
            case_materialization_seconds = time.perf_counter() - materialize_started
            timing["materialization_seconds"] += case_materialization_seconds
            timing["top1_materialization_seconds"] += top1_materialization_seconds
            timing["oracle_control_materialization_seconds"] += (
                oracle_control_materialization_seconds
            )

            generated_eligible = [
                pose for pose in all_generated if eligible.get(pose.candidate_id, False)
            ]
            selected_pose = pose_lookup[clash_aware_selected["candidate_id"]]
            audit_rows.append({
                "row_index": int(row_index),
                "source_row_index": source_row_index,
                "complex_name": complex_name,
                "num_original": len(originals),
                "num_parents": len(parents),
                "num_generated": len(all_generated),
                "num_candidates": len(originals) + len(all_generated),
                "candidate_budget": case_candidate_budget,
                "candidate_budget_exhausted": bool(
                    case_candidate_budget is not None
                    and len(originals) + len(all_generated) >= int(case_candidate_budget)
                ),
                "num_eligible_generated": len(generated_eligible),
                "generated_eligible_pct": 100.0 * len(generated_eligible) / max(len(all_generated), 1),
                "stage_sizes": json.dumps(stage_sizes),
                "beam_sizes": json.dumps(beam_sizes),
                "num_finalists": len(finalists),
                "baseline_selected_candidate_id": baseline_selected["candidate_id"],
                "baseline_selected_severe_clash_count": int(
                    baseline_selected["severe_clash_count"]
                ),
                "selected_candidate_id": clash_aware_selected["candidate_id"],
                "selected_severe_clash_count": int(
                    clash_aware_selected["severe_clash_count"]
                ),
                "caahs_a_switched_top1": clash_aware_switched,
                "selected_new": selected_pose.move_type != "original",
                "selected_depth": selected_pose.depth,
                "selected_branch_type": selected_pose.branch_type,
                "selected_move_type": selected_pose.move_type,
                "selected_move_sequence": "|".join(selected_pose.move_sequence),
                "selected_multiscale_ncc": selected_pose.multiscale_ncc,
                "selection_ready_seconds": selection_ready_seconds,
                "top1_materialization_seconds": top1_materialization_seconds,
                "oracle_control_materialization_seconds": oracle_control_materialization_seconds,
                "deployment_runtime_seconds": (
                    selection_ready_seconds + top1_materialization_seconds
                ),
                "wall_clock_budget_seconds": case_wall_clock_budget,
                "wall_clock_budget_exhausted": bool(
                    case_deadline is not None and time.perf_counter() >= case_deadline
                ),
                "materialization_mode": args.materialization_mode,
                "case_runtime_seconds": time.perf_counter() - case_started,
            })
            all_records.extend(records)
            print(
                f"[{row_index + 1}/{len(manifest)}] {complex_name}: "
                f"{len(originals)} original + {len(all_generated)} adaptive variants; "
                f"selected depth={selected_pose.depth} {selected_pose.move_type}"
            )
        except Exception as exc:
            failures.append({
                "row_index": int(row_index), "source_row_index": source_row_index,
                "complex_name": complex_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
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
        "method": (
            "Clash-Aware Adaptive Hierarchical Search (CAAHS-B)"
            if args.clash_directed_proposals
            else (
                "Clash-Aware Adaptive Hierarchical Search (CAAHS-A)"
                if args.clash_aware_search else "CryoDiffDock-Local V2"
            )
        ),
        "search": (
            "density/steric dual-path hierarchical beam search over SE(3) x T^K"
            if args.clash_aware_search
            else "density-conditioned hierarchical beam search over SE(3) x T^K"
        ),
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "score_manifest_match_mode": score_match_mode,
        "evaluation_manifest": str(output / "evaluation_manifest.csv"),
        "num_complexes": len(audit_rows),
        "num_failures": len(failures),
        "num_candidates": len(all_records),
        "device": str(device),
        "density_sigmas_A": args.density_sigmas,
        "density_weight": args.density_weight,
        "density_gradient_proposals": not args.no_density_gradients,
        "basin_weight": args.basin_weight,
        "anchor_penalty": args.anchor_penalty,
        "coarse_anchor_penalty": args.coarse_anchor_penalty,
        "coarse_penalty": args.coarse_penalty,
        "clash_penalty": args.clash_penalty,
        "clash_aware_search": args.clash_aware_search,
        "caahs_phase": (
            "B" if args.clash_directed_proposals
            else ("A" if args.clash_aware_search else None)
        ),
        "clash_directed_proposals": args.clash_directed_proposals,
        "clash_translation_steps_A": args.clash_translation_steps,
        "clash_rotation_steps_degrees": args.clash_rotation_steps,
        "clash_torsion_steps_degrees": args.clash_torsion_steps,
        "clash_contact_directions": args.clash_contact_directions,
        "steric_parent_slots": args.steric_parent_slots,
        "steric_parent_max_ncc_drop": args.steric_parent_max_ncc_drop,
        "steric_beam_fraction": args.steric_beam_fraction,
        "steric_rescue_max_ncc_drop": args.steric_rescue_max_ncc_drop,
        "steric_rescue_max_anchor_rmsd_A": args.steric_rescue_max_anchor_rmsd_A,
        "clash_aware_final_max_ncc_drop": args.clash_aware_final_max_ncc_drop,
        "clash_aware_final_max_anchor_rmsd_A": (
            args.clash_aware_final_max_anchor_rmsd_A
        ),
        "search_depth": args.search_depth,
        "beam_width": args.beam_width,
        "beam_width_per_parent": args.beam_width_per_parent,
        "coarse_beam_fraction": args.coarse_beam_fraction,
        "parent_count": args.parent_count,
        "top_score_parents": args.top_score_parents,
        "parent_score_column": args.parent_score_column,
        "parent_diversity_rmsd_A": args.parent_diversity_rmsd,
        "translation_steps_A": args.translation_steps,
        "rotation_steps_degrees": args.rotation_steps,
        "torsion_steps_degrees": args.torsion_steps,
        "maximum_torsion_bonds": args.max_torsion_bonds,
        "top_torsion_gradients": args.top_torsion_gradients,
        "minimum_step_ncc_gain": args.minimum_step_ncc_gain,
        "coarse_minimum_ncc_gain": args.coarse_minimum_ncc_gain,
        "maximum_centroid_shift_A": args.max_centroid_shift,
        "minimum_in_bounds_fraction": args.min_in_bounds_fraction,
        "clash_tolerance_count": args.clash_tolerance_count,
        "clash_tolerance_fraction": args.clash_tolerance_fraction,
        "severe_overlap_A": args.severe_overlap,
        "mask_radius_A": args.mask_radius,
        "target_floor": args.target_floor,
        "minimum_mask_voxels": args.min_mask_voxels,
        "finalist_count": args.finalist_count,
        "basin_translation_step_A": args.basin_translation_step,
        "basin_rotation_step_degrees": args.basin_rotation_step,
        "basin_support_tolerance": args.basin_support_tolerance,
        "coarse_orientation_library": not args.no_coarse_orientations,
        "combined_rigid_moves": not args.no_combined_rigid,
        "basin_robust_selection": args.basin_weight > 0,
        "uses_ground_truth": False,
        "materialization_mode": args.materialization_mode,
        "requested_total_candidate_budget": args.total_candidate_budget,
        "candidate_budget_allocation": (
            "native-capacity proportional largest remainder"
            if native_capacities is not None else "even additional candidates in manifest order"
        ),
        "candidate_budget_reference_audit": (
            str(candidate_budget_reference) if candidate_budget_reference else None
        ),
        "actual_total_candidates": int(actual_total_candidates),
        "requested_total_wall_clock_budget_seconds": args.total_wall_clock_budget_seconds,
        "per_complex_wall_clock_budget_seconds": case_wall_clock_budget,
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
        },
        "runtime_definition": (
            "search/scoring/selection plus final Top-1 materialization"
            if args.materialization_mode == "top1"
            else (
                "search/scoring/selection plus final-selector-pool materialization"
                if args.materialization_mode == "interaction_diagnostic"
                else "search/scoring/selection plus legacy benchmark/oracle materialization"
            )
        ),
        "runtime_seconds": time.perf_counter() - started,
        "timing": timing,
        "conditions": {name: str(path) for name, path in roots.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
