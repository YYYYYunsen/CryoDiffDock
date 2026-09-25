#!/usr/bin/env python3
"""Run one restrained ligand internal-geometry rescue after CAAHS-C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.clash_repair import clash_field  # noqa: E402
from cryodiffdock_pocket.density_ranking import (  # noqa: E402
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
)
from cryodiffdock_pocket.interactions import (  # noqa: E402
    read_protein_interaction_sites,
    score_pose_interactions,
)
from cryodiffdock_pocket.internal_geometry_refinement import (  # noqa: E402
    restrained_internal_geometry_candidates,
)
from cryodiffdock_pocket.physics import (  # noqa: E402
    read_protein_heavy_atoms,
    relative_vdw_clashes,
)
from scripts.run_adaptive_density_search import (  # noqa: E402
    molecule_coordinates,
    read_molecule,
    write_sdf,
)


def parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def interaction_quality(protein: Path, ligand: Path) -> float | None:
    try:
        sites = read_protein_interaction_sites(protein)
        return float(score_pose_interactions(sites, ligand)["interaction_geometry_quality"])
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--density-crop", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-atom-displacements-A", required=True)
    parser.add_argument("--force-constants", required=True)
    parser.add_argument("--maximum-iterations", type=int, default=200)
    parser.add_argument("--maximum-ligand-rmsd-A", type=float, default=0.35)
    parser.add_argument("--minimum-relative-vdw-distance", type=float, default=0.77)
    parser.add_argument("--maximum-ncc-drop", type=float, default=0.02)
    parser.add_argument("--maximum-interaction-drop", type=float, default=0.05)
    parser.add_argument("--severe-overlap-A", type=float, default=0.8)
    parser.add_argument("--sigmas-A", default="1.0,1.5,2.0")
    parser.add_argument("--mask-radius-A", type=float, default=10.0)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    molecule = read_molecule(args.ligand)
    root_xyz = molecule_coordinates(molecule)
    heavy_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    ligand_elements = [
        atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    atomic_numbers = np.asarray(
        [molecule.GetAtomWithIdx(index).GetAtomicNum() for index in heavy_indices],
        dtype=np.float32,
    )
    protein_xyz, protein_elements = read_protein_heavy_atoms(args.protein)
    crop = load_cartesian_density_crop(args.density_crop)
    mask = fixed_crop_mask(crop, radius_A=args.mask_radius_A, min_voxels=64)
    sigmas = parse_floats(args.sigmas_A)

    def density_score(xyz: np.ndarray) -> float:
        heavy_xyz = np.asarray(xyz, dtype=np.float64)[np.asarray(heavy_indices)]
        return float(np.mean([
            score_gaussian_density(
                crop, heavy_xyz.astype(np.float32), atomic_numbers,
                sigma, mask, cutoff_sigma=4.0,
            ).ncc
            for sigma in sigmas
        ]))

    baseline_ncc = density_score(root_xyz)
    baseline_interaction = interaction_quality(args.protein, args.ligand)
    maximum_atom_displacements = parse_floats(
        args.maximum_atom_displacements_A
    )
    maximum_atom_displacement_gate = max(maximum_atom_displacements)
    candidates = restrained_internal_geometry_candidates(
        molecule,
        maximum_atom_displacements_A=maximum_atom_displacements,
        force_constants=parse_floats(args.force_constants),
        maximum_iterations=args.maximum_iterations,
    )
    candidate_dir = args.output_dir / "candidates"
    candidate_dir.mkdir()
    records = []
    admissible = []
    for index, candidate in enumerate(candidates):
        pose_path = candidate_dir / f"candidate_{index:03d}.sdf"
        write_sdf(candidate.molecule, pose_path)
        xyz = np.asarray(candidate.molecule.GetConformer().GetPositions(), dtype=np.float64)
        heavy_xyz = xyz[np.asarray(heavy_indices)]
        ncc = density_score(xyz)
        relative_count, minimum_relative, _ = relative_vdw_clashes(
            protein_xyz, protein_elements, heavy_xyz, ligand_elements,
            minimum_relative_distance=args.minimum_relative_vdw_distance,
        )
        severe = clash_field(
            protein_xyz, protein_elements, heavy_xyz, ligand_elements,
            severe_overlap_A=args.severe_overlap_A,
        )
        quality = interaction_quality(args.protein, pose_path)
        interaction_change = (
            quality - baseline_interaction
            if quality is not None and baseline_interaction is not None else None
        )
        interaction_pass = bool(
            quality is not None
            and (
                baseline_interaction is None
                or interaction_change >= -args.maximum_interaction_drop - 1e-8
            )
        )
        geometry = candidate.geometry
        hard_pass = bool(
            candidate.candidate_id != "baseline"
            and geometry.internal_steric_clash_pass
            and geometry.bond_lengths_pass
            and geometry.bond_angles_pass
            and relative_count == 0
            and minimum_relative >= args.minimum_relative_vdw_distance - 1e-8
            and severe.severe_count == 0
            and candidate.displacement_A <= args.maximum_ligand_rmsd_A + 1e-8
            and candidate.maximum_atom_displacement_A
            <= maximum_atom_displacement_gate + 1e-8
            and ncc - baseline_ncc >= -args.maximum_ncc_drop - 1e-8
            and interaction_pass
        )
        record = {
            "candidate_id": candidate.candidate_id,
            "force_field": candidate.force_field,
            "minimize_status": candidate.minimize_status,
            "energy": candidate.energy,
            "internal_clash_count": geometry.internal_clash_count,
            "shortest_noncovalent_relative_distance": (
                geometry.shortest_noncovalent_relative_distance
            ),
            "bond_outlier_count": geometry.bond_outlier_count,
            "angle_outlier_count": geometry.angle_outlier_count,
            "ligand_internal_rmsd_A": candidate.displacement_A,
            "maximum_atom_displacement_A": candidate.maximum_atom_displacement_A,
            "protein_relative_vdw_clash_count": relative_count,
            "protein_minimum_relative_vdw_distance": minimum_relative,
            "severe_clash_count": severe.severe_count,
            "multiscale_ncc": ncc,
            "delta_ncc": ncc - baseline_ncc,
            "interaction_geometry_quality": quality,
            "interaction_geometry_change": interaction_change,
            "hard_gates_pass": hard_pass,
            "pose_path": str(pose_path),
        }
        records.append(record)
        if hard_pass:
            admissible.append(record)

    baseline = records[0]
    selected = None
    if int(baseline["internal_clash_count"]) > 0 and admissible:
        admissible.sort(key=lambda item: (
            int(item["internal_clash_count"]),
            int(item["bond_outlier_count"]) + int(item["angle_outlier_count"]),
            float(item["ligand_internal_rmsd_A"]),
            -float(item["shortest_noncovalent_relative_distance"]),
            -float(item["multiscale_ncc"]),
            str(item["candidate_id"]),
        ))
        selected = admissible[0]

    if selected is None:
        shutil.copy2(args.ligand, args.output_dir / "selected_pose.sdf")
        status = (
            "no_action_internal_geometry_pass"
            if int(baseline["internal_clash_count"]) == 0
            else "no_admissible_internal_geometry_rescue"
        )
        selected_method = "baseline_retained"
        final = baseline
    else:
        shutil.copy2(Path(str(selected["pose_path"])), args.output_dir / "selected_pose.sdf")
        status = "resolved_internal_steric_clash"
        selected_method = "restrained_internal_geometry"
        final = selected
    shutil.copy2(args.protein, args.output_dir / "selected_protein.pdb")

    pd.DataFrame(records).to_csv(args.output_dir / "candidate_audit.csv", index=False)
    summary = {
        "method": "CAAHS-C restrained ligand internal-geometry refinement",
        "status": status,
        "selected_method": selected_method,
        "baseline_internal_clash_count": int(baseline["internal_clash_count"]),
        "selected_internal_clash_count": int(final["internal_clash_count"]),
        "baseline_shortest_noncovalent_relative_distance": float(
            baseline["shortest_noncovalent_relative_distance"]
        ),
        "selected_shortest_noncovalent_relative_distance": float(
            final["shortest_noncovalent_relative_distance"]
        ),
        "selected_bond_outlier_count": int(final["bond_outlier_count"]),
        "selected_angle_outlier_count": int(final["angle_outlier_count"]),
        "ligand_internal_rmsd_A": float(final["ligand_internal_rmsd_A"]),
        "maximum_atom_displacement_A": float(final["maximum_atom_displacement_A"]),
        "protein_relative_vdw_clash_count": int(
            final["protein_relative_vdw_clash_count"]
        ),
        "baseline_severe_clash_count": int(baseline["severe_clash_count"]),
        "selected_severe_clash_count": int(final["severe_clash_count"]),
        "protein_minimum_relative_vdw_distance": float(
            final["protein_minimum_relative_vdw_distance"]
        ),
        "delta_ncc": float(final["delta_ncc"]),
        "interaction_geometry_change": final["interaction_geometry_change"],
        "maximum_ligand_rmsd_A": args.maximum_ligand_rmsd_A,
        "maximum_atom_displacement_gate_A": maximum_atom_displacement_gate,
        "minimum_relative_vdw_distance": args.minimum_relative_vdw_distance,
        "maximum_ncc_drop": args.maximum_ncc_drop,
        "maximum_interaction_drop": args.maximum_interaction_drop,
        "ligand_coordinates_changed": selected is not None,
        "protein_coordinates_changed": False,
        "uses_reference_or_native_rmsd": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
