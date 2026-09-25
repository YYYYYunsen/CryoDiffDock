#!/usr/bin/env python3
"""Run one CAAHS-C rigid or joint ligand--side-chain rescue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.caahs_c_refinement import (  # noqa: E402
    CaahsCCandidate,
    candidate_passes,
    refine_rigid_pose,
)
from cryodiffdock_pocket.density_ranking import (  # noqa: E402
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
)
from cryodiffdock_pocket.interactions import (  # noqa: E402
    read_protein_interaction_sites,
    score_pose_interactions,
)
from cryodiffdock_pocket.physics import read_protein_heavy_atoms  # noqa: E402
from scripts.run_adaptive_density_search import (  # noqa: E402
    molecule_coordinates,
    molecule_with_coordinates,
    read_molecule,
    write_sdf,
)


def interaction_quality(protein: Path, ligand: Path) -> float | None:
    try:
        sites = read_protein_interaction_sites(protein)
        return float(score_pose_interactions(sites, ligand)["interaction_geometry_quality"])
    except Exception:
        return None


def candidate_record(candidate: CaahsCCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "relative_vdw_clash_count": candidate.relative_clash_count,
        "minimum_relative_vdw_distance": candidate.minimum_relative_distance,
        "target_deficit_energy": candidate.target_deficit_energy,
        "severe_clash_count": candidate.severe_clash_count,
        "ligand_displacement_A": candidate.displacement_A,
        "multiscale_ncc": candidate.multiscale_ncc,
        "delta_ncc": candidate.ncc_change,
        "optimizer_success": candidate.optimizer_success,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--density-crop", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("rigid", "joint"), required=True)
    parser.add_argument("--maximum-ligand-displacement-A", required=True, type=float)
    parser.add_argument("--target-relative-vdw-distance", type=float, default=0.77)
    parser.add_argument("--optimization-relative-vdw-distance", type=float, default=0.78)
    parser.add_argument("--acceptance-relative-vdw-distance", type=float, default=0.75)
    parser.add_argument("--maximum-ncc-drop", type=float, default=0.02)
    parser.add_argument("--maximum-interaction-drop", type=float, default=0.05)
    parser.add_argument("--severe-overlap-A", type=float, default=0.8)
    parser.add_argument("--rotation-bound-degrees", type=float, default=5.0)
    parser.add_argument("--maximum-iterations", type=int, default=30)
    parser.add_argument("--number-of-starts", type=int, default=4)
    parser.add_argument("--joint-top-k", type=int, default=4)
    parser.add_argument("--chi-offsets", default="-60,-30,30,60")
    parser.add_argument("--sidechain-beam-width", type=int, default=16)
    parser.add_argument("--sidechain-coordinate-rounds", type=int, default=2)
    parser.add_argument("--maximum-target-residues", type=int, default=3)
    parser.add_argument("--maximum-sidechain-displacement-A", type=float, default=1.5)
    parser.add_argument("--sigmas-A", default="1.0,1.5,2.0")
    parser.add_argument("--mask-radius-A", type=float, default=10.0)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    for path in (args.ligand, args.protein, args.density_crop):
        if not path.is_file():
            raise FileNotFoundError(path)

    molecule = read_molecule(args.ligand)
    root_xyz = molecule_coordinates(molecule)
    ligand_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.float32
    )
    protein_xyz, protein_elements = read_protein_heavy_atoms(args.protein)
    crop = load_cartesian_density_crop(args.density_crop)
    mask = fixed_crop_mask(crop, radius_A=args.mask_radius_A, min_voxels=64)
    sigmas = tuple(float(value) for value in args.sigmas_A.split(","))

    def density_scorer(xyz: np.ndarray) -> float:
        return float(np.mean([
            score_gaussian_density(
                crop, np.asarray(xyz, dtype=np.float32), atomic_numbers,
                sigma, mask, cutoff_sigma=4.0,
            ).ncc
            for sigma in sigmas
        ]))

    candidates = refine_rigid_pose(
        ligand_coordinates=root_xyz,
        ligand_elements=ligand_elements,
        protein_coordinates=protein_xyz,
        protein_elements=protein_elements,
        density_scorer=density_scorer,
        maximum_displacement_A=args.maximum_ligand_displacement_A,
        maximum_ncc_drop=args.maximum_ncc_drop,
        target_relative_distance=args.target_relative_vdw_distance,
        optimization_relative_distance=args.optimization_relative_vdw_distance,
        acceptance_relative_distance=args.acceptance_relative_vdw_distance,
        severe_overlap_A=args.severe_overlap_A,
        rotation_bound_degrees=args.rotation_bound_degrees,
        maximum_iterations=args.maximum_iterations,
        number_of_starts=args.number_of_starts,
    )
    baseline = next(item for item in candidates if item.candidate_id == "baseline")
    baseline_interaction = interaction_quality(args.protein, args.ligand)
    rigid_dir = args.output_dir / "rigid_candidates"
    rigid_dir.mkdir()
    records = []
    admissible: list[dict[str, object]] = []

    for index, candidate in enumerate(candidates):
        pose_path = rigid_dir / f"candidate_{index:03d}.sdf"
        write_sdf(molecule_with_coordinates(molecule, candidate.coordinates), pose_path)
        record = candidate_record(candidate)
        record["pose_path"] = str(pose_path)
        record["candidate_type"] = "rigid"
        record["interaction_geometry_quality"] = None
        record["interaction_geometry_change"] = None
        record["hard_gates_pass"] = False
        if candidate_passes(
            candidate,
            maximum_displacement_A=args.maximum_ligand_displacement_A,
            maximum_ncc_drop=args.maximum_ncc_drop,
        ) and candidate.minimum_relative_distance >= (
            args.target_relative_vdw_distance - 1e-8
        ):
            quality = interaction_quality(args.protein, pose_path)
            change = (
                quality - baseline_interaction
                if quality is not None and baseline_interaction is not None else None
            )
            interaction_pass = bool(
                quality is not None
                and (baseline_interaction is None or change >= -args.maximum_interaction_drop)
            )
            record["interaction_geometry_quality"] = quality
            record["interaction_geometry_change"] = change
            record["hard_gates_pass"] = interaction_pass
            if interaction_pass:
                admissible.append({
                    **record,
                    "protein_path": str(args.protein),
                    "sidechain_displacement_A": 0.0,
                })
        records.append(record)

    if args.mode == "joint" and not admissible:
        eligible = [
            item for item in candidates
            if item.displacement_A <= args.maximum_ligand_displacement_A + 1e-8
            and item.ncc_change >= -args.maximum_ncc_drop - 1e-8
            and (
                item.relative_clash_count < baseline.relative_clash_count
                or item.minimum_relative_distance > baseline.minimum_relative_distance + 1e-8
            )
        ][: args.joint_top_k]
        sidechain_script = Path(__file__).resolve().with_name(
            "refine_caahs_a_sidechain_rotamers.py"
        )
        for joint_index, candidate in enumerate(eligible):
            ligand_path = args.output_dir / "joint_inputs" / f"candidate_{joint_index:03d}.sdf"
            write_sdf(molecule_with_coordinates(molecule, candidate.coordinates), ligand_path)
            joint_output = args.output_dir / "joint_candidates" / f"candidate_{joint_index:03d}"
            command = [
                sys.executable, str(sidechain_script),
                "--ligand", str(ligand_path),
                "--protein", str(args.protein),
                "--output-dir", str(joint_output),
                "--selection-mode", "interaction",
                f"--chi-offsets={args.chi_offsets}",
                "--beam-width", str(args.sidechain_beam_width),
                "--coordinate-rounds", str(args.sidechain_coordinate_rounds),
                "--max-residues", str(args.maximum_target_residues),
                "--max-sidechain-displacement-A", str(args.maximum_sidechain_displacement_A),
                "--maximum-interaction-drop", str(args.maximum_interaction_drop),
                "--severe-overlap-A", str(args.severe_overlap_A),
                "--minimum-relative-vdw-distance", str(args.target_relative_vdw_distance),
                "--method-name", "CAAHS-C",
            ]
            process = subprocess.run(command, capture_output=True, text=True, check=False)
            if process.returncode != 0:
                records.append({
                    **candidate_record(candidate),
                    "candidate_id": f"joint_{joint_index}",
                    "candidate_type": "joint_failed",
                    "hard_gates_pass": False,
                    "error": process.stdout + process.stderr,
                })
                continue
            summary = json.loads((joint_output / "summary.json").read_text())
            final_protein = joint_output / "selected_protein.pdb"
            quality = interaction_quality(final_protein, ligand_path)
            change = (
                quality - baseline_interaction
                if quality is not None and baseline_interaction is not None else None
            )
            interaction_pass = bool(
                quality is not None
                and (baseline_interaction is None or change >= -args.maximum_interaction_drop)
            )
            passes = bool(
                summary["selected_relative_vdw_clash_count"] == 0
                and summary["selected_minimum_relative_vdw_distance"]
                >= args.target_relative_vdw_distance - 1e-8
                and summary["selected_severe_clash_count"] == 0
                and candidate.displacement_A <= args.maximum_ligand_displacement_A + 1e-8
                and candidate.ncc_change >= -args.maximum_ncc_drop - 1e-8
                and interaction_pass
            )
            record = {
                **candidate_record(candidate),
                "candidate_id": f"joint_{joint_index}",
                "candidate_type": "joint",
                "relative_vdw_clash_count": summary["selected_relative_vdw_clash_count"],
                "minimum_relative_vdw_distance": summary["selected_minimum_relative_vdw_distance"],
                "severe_clash_count": summary["selected_severe_clash_count"],
                "interaction_geometry_quality": quality,
                "interaction_geometry_change": change,
                "sidechain_displacement_A": summary["sidechain_displacement_A"],
                "pose_path": str(ligand_path),
                "protein_path": str(final_protein),
                "hard_gates_pass": passes,
                "sidechain_status": summary["status"],
            }
            records.append(record)
            if passes:
                admissible.append(record)

    baseline_pass = candidate_passes(
        baseline,
        maximum_displacement_A=args.maximum_ligand_displacement_A,
        maximum_ncc_drop=args.maximum_ncc_drop,
    )
    selected: dict[str, object] | None = None
    if not baseline_pass and admissible:
        admissible.sort(key=lambda item: (
            float(item["ligand_displacement_A"]),
            float(item.get("sidechain_displacement_A", 0.0)),
            -float(item["interaction_geometry_quality"]),
            -float(item["multiscale_ncc"]),
            str(item["candidate_id"]),
        ))
        selected = admissible[0]

    if selected is None:
        shutil.copy2(args.ligand, args.output_dir / "selected_pose.sdf")
        shutil.copy2(args.protein, args.output_dir / "selected_protein.pdb")
        final_relative_count = baseline.relative_clash_count
        final_minimum_relative = baseline.minimum_relative_distance
        final_severe_count = baseline.severe_clash_count
        final_displacement = 0.0
        final_ncc_change = 0.0
        final_interaction_change = 0.0
        final_sidechain_displacement = 0.0
        selected_method = "baseline_retained"
        status = "no_action_already_pass" if baseline_pass else "no_admissible_rescue"
    else:
        shutil.copy2(Path(str(selected["pose_path"])), args.output_dir / "selected_pose.sdf")
        shutil.copy2(Path(str(selected["protein_path"])), args.output_dir / "selected_protein.pdb")
        final_relative_count = int(selected["relative_vdw_clash_count"])
        final_minimum_relative = float(selected["minimum_relative_vdw_distance"])
        final_severe_count = int(selected["severe_clash_count"])
        final_displacement = float(selected["ligand_displacement_A"])
        final_ncc_change = float(selected["delta_ncc"])
        final_interaction_change = selected["interaction_geometry_change"]
        final_sidechain_displacement = float(selected.get("sidechain_displacement_A", 0.0))
        selected_method = f"caahs_c_{selected['candidate_type']}"
        status = "resolved_minimum_distance"

    pd.DataFrame(records).to_csv(args.output_dir / "candidate_audit.csv", index=False)
    summary = {
        "method": "CAAHS-C",
        "mode": args.mode,
        "status": status,
        "selected_method": selected_method,
        "baseline_relative_vdw_clash_count": baseline.relative_clash_count,
        "selected_relative_vdw_clash_count": final_relative_count,
        "baseline_minimum_relative_vdw_distance": baseline.minimum_relative_distance,
        "selected_minimum_relative_vdw_distance": final_minimum_relative,
        "baseline_severe_clash_count": baseline.severe_clash_count,
        "selected_severe_clash_count": final_severe_count,
        "ligand_displacement_A": final_displacement,
        "sidechain_displacement_A": final_sidechain_displacement,
        "delta_ncc": final_ncc_change,
        "interaction_geometry_change": final_interaction_change,
        "target_relative_vdw_distance": args.target_relative_vdw_distance,
        "optimization_relative_vdw_distance": args.optimization_relative_vdw_distance,
        "acceptance_relative_vdw_distance": args.acceptance_relative_vdw_distance,
        "maximum_ligand_displacement_A": args.maximum_ligand_displacement_A,
        "maximum_ncc_drop": args.maximum_ncc_drop,
        "maximum_interaction_drop": args.maximum_interaction_drop,
        "ligand_coordinates_changed": selected is not None,
        "uses_reference_or_native_rmsd": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
