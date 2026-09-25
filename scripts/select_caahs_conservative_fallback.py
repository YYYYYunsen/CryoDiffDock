#!/usr/bin/env python3
"""Apply the frozen CAAHS-A rescue-only selector without native structures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.adaptive_search import MultiScaleNCCEvaluator, ProteinClashIndex
from cryodiffdock_pocket.density_ranking import load_cartesian_density_crop
from cryodiffdock_pocket.physics import paired_protein_for_pose, read_protein_heavy_atoms
from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd
from scripts.run_adaptive_density_search import molecule_coordinates, read_molecule


FORBIDDEN = ("reference", "rmsd_before", "rmsd_after", "native", "ground_truth")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0].strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return values


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if value == "cuda" or (value == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_protein(pose_path: Path, fallback: Path) -> Path:
    paired = paired_protein_for_pose(pose_path)
    return paired.resolve() if paired is not None else fallback.resolve()


def clash_score(
    pose_path: Path,
    protein_path: Path,
    cache: dict[str, ProteinClashIndex],
    severe_overlap_A: float,
) -> tuple[int, float]:
    key = str(protein_path.resolve())
    if key not in cache:
        xyz, elements = read_protein_heavy_atoms(protein_path)
        cache[key] = ProteinClashIndex(xyz, elements)
    molecule = read_molecule(pose_path)
    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    count, fraction = cache[key].score(
        molecule_coordinates(molecule),
        elements,
        severe_overlap_A=severe_overlap_A,
    )
    return int(count), float(fraction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--selected-poses", required=True, type=Path)
    parser.add_argument("--full-manifest", required=True, type=Path)
    parser.add_argument("--branch-manifest", required=True, type=Path)
    parser.add_argument("--caahs-root", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    inputs = [
        args.cohort, args.selected_poses, args.full_manifest,
        args.branch_manifest, args.config_json,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.caahs_root.is_dir():
        raise FileNotFoundError(args.caahs_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")

    config = json.loads(args.config_json.read_text())
    frozen = config.get("conservative_fallback", {})
    if frozen.get("status") != "development_frozen":
        raise ValueError("CAAHS conservative fallback is not development-frozen")
    maximum_displacement = float(frozen["maximum_old_to_new_displacement_A"])
    maximum_ncc_drop = float(frozen["maximum_ncc_drop"])
    if frozen.get("require_new_clash_free") is not True:
        raise ValueError("Frozen selector must require a clash-free new pose")
    if frozen.get("uses_reference_or_rmsd_to_native_for_selection") is not False:
        raise ValueError("Frozen selector must prohibit native/reference selection")
    severe_overlap = float(config["severe_overlap_A"])
    device = choose_device(args.device)

    cohort = read_ids(args.cohort)
    if len(cohort) != args.expected_cases:
        raise ValueError(f"Expected {args.expected_cases} IDs, found {len(cohort)}")

    selected = pd.read_csv(
        args.selected_poses,
        usecols=["pdb_id", "selected_pose_path", "selected_global_rank"],
    )
    selected["pdb_id"] = selected["pdb_id"].astype(str).str.upper()
    selected["selected_global_rank"] = pd.to_numeric(
        selected["selected_global_rank"], errors="raise"
    ).astype(int)
    selected = selected[selected["pdb_id"].isin(cohort)].copy()
    if len(selected) != len(cohort) or selected["pdb_id"].duplicated().any():
        raise ValueError("Selected-pose identities do not match the cohort")
    selected = selected.set_index("pdb_id")

    full = pd.read_csv(
        args.full_manifest,
        usecols=["parent_complex_name", "global_rank", "experimental_protein"],
    )
    full["parent_complex_name"] = full["parent_complex_name"].astype(str).str.upper()
    full["global_rank"] = pd.to_numeric(full["global_rank"], errors="raise").astype(int)

    branch = pd.read_csv(args.branch_manifest)
    forbidden = [column for column in branch if any(token in column.lower() for token in FORBIDDEN)]
    if forbidden:
        raise ValueError(f"Blind branch manifest contains forbidden columns: {forbidden}")
    required_branch = {
        "complex_name", "parent_complex_name", "global_rank", "ligand_density_crop"
    }
    if required_branch - set(branch):
        raise ValueError(f"Branch manifest lacks {sorted(required_branch - set(branch))}")
    branch["parent_complex_name"] = branch["parent_complex_name"].astype(str).str.upper()
    branch["global_rank"] = pd.to_numeric(branch["global_rank"], errors="raise").astype(int)
    if branch["parent_complex_name"].duplicated().any():
        raise ValueError("Selected-branch manifest has duplicate parent IDs")
    branch_index = {
        row.parent_complex_name: (index, row)
        for index, row in enumerate(branch.itertuples(index=False))
    }

    protein_cache: dict[str, ProteinClashIndex] = {}
    rows: list[dict] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_root = args.output_dir / "selected_results"

    for order, pdb_id in enumerate(cohort):
        choice = selected.loc[pdb_id]
        old_pose = Path(str(choice["selected_pose_path"])).expanduser().resolve()
        metadata = full[full["parent_complex_name"].eq(pdb_id)]
        if metadata.empty:
            raise ValueError(f"{pdb_id}: missing full-manifest metadata")
        fallback_protein = Path(str(metadata.iloc[0]["experimental_protein"]))
        old_protein = resolve_protein(old_pose, fallback_protein)
        old_count, old_fraction = clash_score(
            old_pose, old_protein, protein_cache, severe_overlap
        )

        global_rank = int(choice["selected_global_rank"])
        new_pose = old_pose
        new_protein = old_protein
        new_count, new_fraction = old_count, old_fraction
        displacement = 0.0
        old_ncc = float("nan")
        new_ncc = float("nan")
        delta_ncc = float("nan")
        accepted = False
        reason = "frozen_global_pose_unchanged" if global_rank == 0 else "retain_old"
        branch_name = ""

        if global_rank != 0:
            if pdb_id not in branch_index:
                raise ValueError(f"{pdb_id}: selected local branch is absent")
            row_index, branch_row = branch_index[pdb_id]
            if int(branch_row.global_rank) != global_rank:
                raise ValueError(f"{pdb_id}: selected global rank differs from branch manifest")
            branch_name = str(branch_row.complex_name)
            result_dir = args.caahs_root / "v2_selected" / f"index{row_index}___{branch_name}"
            ranking_path = result_dir / "ranking.csv"
            ranking = pd.read_csv(ranking_path)
            if len(ranking) != 1:
                raise ValueError(f"{pdb_id}: expected one CAAHS Top-1 in {ranking_path}")
            new_pose = Path(str(ranking.iloc[0]["materialized_path"])).resolve()
            new_protein = Path(str(ranking.iloc[0]["protein_path"])).resolve()
            new_count, new_fraction = clash_score(
                new_pose, new_protein, protein_cache, severe_overlap
            )

            old_heavy = read_heavy_molecule(old_pose)
            new_heavy = read_heavy_molecule(new_pose)
            displacement = float(symmetry_rmsd(old_heavy, new_heavy))

            old_molecule = read_molecule(old_pose)
            new_molecule = read_molecule(new_pose)
            atomic_numbers = [atom.GetAtomicNum() for atom in old_molecule.GetAtoms()]
            if atomic_numbers != [atom.GetAtomicNum() for atom in new_molecule.GetAtoms()]:
                raise ValueError(f"{pdb_id}: old/new ligand atom ordering differs")
            evaluator = MultiScaleNCCEvaluator(
                load_cartesian_density_crop(Path(str(branch_row.ligand_density_crop))),
                atomic_numbers,
                sigmas_A=tuple(float(value) for value in config["density_sigmas_A"])
                if "density_sigmas_A" in config else (1.5, 2.25),
                mask_radius_A=float(config.get("mask_radius_A", 10.0)),
                min_mask_voxels=int(config.get("minimum_mask_voxels", 64)),
                device=device,
                voxel_chunk_size=512,
                pose_batch_size=2,
            )
            scores = evaluator.score([
                molecule_coordinates(old_molecule), molecule_coordinates(new_molecule)
            ])
            old_ncc = float(scores[0].mean())
            new_ncc = float(scores[1].mean())
            delta_ncc = new_ncc - old_ncc
            accepted = bool(
                old_count > 0
                and new_count == 0
                and displacement <= maximum_displacement
                and delta_ncc >= -maximum_ncc_drop
            )
            if accepted:
                reason = "accepted_complete_clash_rescue_within_frozen_gates"
            elif old_count == 0:
                reason = "retain_old_already_clash_free"
            elif new_count != 0:
                reason = "retain_old_new_not_clash_free"
            elif displacement > maximum_displacement:
                reason = "retain_old_displacement_gate"
            else:
                reason = "retain_old_ncc_gate"

        final_pose = new_pose if accepted else old_pose
        final_protein = new_protein if accepted else old_protein
        final_count = new_count if accepted else old_count
        final_fraction = new_fraction if accepted else old_fraction
        destination = selected_root / pdb_id
        destination.mkdir(parents=True, exist_ok=True)
        ranked_pose = destination / "rank1_confidence0.00000000.sdf"
        shutil.copy2(final_pose, ranked_pose)
        shutil.copy2(final_pose, destination / "rank1.sdf")
        shutil.copy2(final_protein, destination / "rank1_protein.pdb")
        rows.append({
            "cohort_order": order,
            "pdb_id": pdb_id,
            "selected_global_rank": global_rank,
            "branch_name": branch_name,
            "caahs_attempted": global_rank != 0,
            "caahs_accepted": accepted,
            "selection_reason": reason,
            "old_pose_path": str(old_pose),
            "caahs_pose_path": str(new_pose),
            "selected_pose_path": str(ranked_pose.resolve()),
            "old_severe_clash_count": old_count,
            "caahs_severe_clash_count": new_count,
            "selected_severe_clash_count": final_count,
            "old_clashing_atom_fraction": old_fraction,
            "caahs_clashing_atom_fraction": new_fraction,
            "selected_clashing_atom_fraction": final_fraction,
            "old_to_new_displacement_A": displacement,
            "old_multiscale_ncc": old_ncc,
            "caahs_multiscale_ncc": new_ncc,
            "delta_ncc": delta_ncc,
        })

    table = pd.DataFrame(rows)
    selection_path = args.output_dir / "selected_poses_blind.csv"
    table.to_csv(selection_path, index=False)
    summary = {
        "status": "selection_sealed_before_outcome_evaluation",
        "method": "CAAHS-A conservative fallback",
        "cases_n": len(table),
        "caahs_attempted_n": int(table["caahs_attempted"].sum()),
        "caahs_accepted_n": int(table["caahs_accepted"].sum()),
        "old_clash_free_n": int(table["old_severe_clash_count"].eq(0).sum()),
        "selected_clash_free_n": int(table["selected_severe_clash_count"].eq(0).sum()),
        "maximum_old_to_new_displacement_A": maximum_displacement,
        "maximum_ncc_drop": maximum_ncc_drop,
        "uses_reference_or_native_rmsd_for_selection": False,
        "selection_sha256": sha256(selection_path),
        "config_sha256": sha256(args.config_json),
        "input_hashes": {str(path.resolve()): sha256(path) for path in inputs},
        "caahs_summary_sha256": sha256(args.caahs_root / "summary.json"),
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {"platform": platform.platform(), "device": str(device)},
    }
    (args.output_dir / "selection_seal.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"CAAHS_CONSERVATIVE_SELECTION_SEALED: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
