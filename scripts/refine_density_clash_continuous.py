#!/usr/bin/env python3
"""Continuously refine one unresolved V2 repair under frozen hard gates."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.clash_repair import (  # noqa: E402
    clash_field,
    directed_local_variants,
)
from cryodiffdock_pocket.continuous_clash_refinement import (  # noqa: E402
    RefinementEvaluation,
    refine_seed_fast_slsqp,
    refine_seed_slsqp,
)
from cryodiffdock_pocket.density_ranking import (  # noqa: E402
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
)
from cryodiffdock_pocket.discrete_search import direct_rmsd, rotatable_bonds  # noqa: E402
from cryodiffdock_pocket.physics import read_protein_heavy_atoms  # noqa: E402


def read_heavy_molecule(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand: {path}")
    editable = Chem.RWMol(molecule)
    for index in reversed(range(editable.GetNumAtoms())):
        if editable.GetAtomWithIdx(index).GetAtomicNum() == 1:
            editable.RemoveAtom(index)
    result = editable.GetMol()
    result.UpdatePropertyCache(strict=False)
    try:
        Chem.GetSymmSSSR(result)
    except Exception:
        pass
    return result


def coordinates(molecule) -> np.ndarray:
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)


def molecule_with_coordinates(template, xyz: np.ndarray):
    molecule = Chem.Mol(template)
    conformer = molecule.GetConformer()
    for index, point in enumerate(np.asarray(xyz, dtype=np.float64)):
        conformer.SetAtomPosition(index, tuple(float(value) for value in point))
    return molecule


def write_sdf(molecule, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        writer.write(molecule)
    finally:
        writer.close()


def evaluation_record(evaluation: RefinementEvaluation) -> dict[str, object]:
    return {
        "severe_clash_count": evaluation.clash.severe_count,
        "clashing_ligand_atom_fraction": evaluation.clash.clashing_ligand_atom_fraction,
        "max_vdw_overlap_A": evaluation.clash.max_overlap_A,
        "min_protein_ligand_distance_A": evaluation.clash.min_distance_A,
        "smooth_clash_energy": evaluation.clash.energy,
        "displacement_A": evaluation.displacement_A,
        "multiscale_ncc": evaluation.multiscale_ncc,
        "ncc_change": evaluation.ncc_change,
    }


def selection_key(record: dict[str, object]) -> tuple[float, ...]:
    clashes = int(record["severe_clash_count"])
    if clashes == 0:
        return (
            0.0, float(record["displacement_A"]),
            -float(record["multiscale_ncc"]),
            float(record["smooth_clash_energy"]),
        )
    return (
        float(clashes), float(record["smooth_clash_energy"]),
        float(record["clashing_ligand_atom_fraction"]),
        float(record["displacement_A"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-starts", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-torsion-bonds", type=int, default=8)
    parser.add_argument(
        "--optimization-mode", choices=["fast", "exact"], default="fast",
        help="fast keeps exact NCC out of numerical-gradient inner loops",
    )
    parser.add_argument("--translation-bound-A", type=float, default=0.25)
    parser.add_argument("--rotation-bound-deg", type=float, default=7.5)
    parser.add_argument("--torsion-bound-deg", type=float, default=15.0)
    parser.add_argument("--minimum-seed-rmsd-A", type=float, default=0.04)
    parser.add_argument(
        "--max-displacement-A", type=float,
        help="Override the input repair displacement gate for development sweeps",
    )
    parser.add_argument(
        "--max-ncc-drop", type=float,
        help="Override the input repair NCC-drop gate for development sweeps",
    )
    parser.add_argument(
        "--restart-from-original", action="store_true",
        help=(
            "Generate continuous starts around the upstream original pose while "
            "retaining the discrete selected pose as the safe baseline"
        ),
    )
    args = parser.parse_args()
    if args.num_starts < 1 or args.max_iterations < 1 or args.max_torsion_bonds < 0:
        raise ValueError("starts/iterations must be positive and torsion count non-negative")

    started = time.perf_counter()
    case_dir = args.v2_case_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    summary_path = case_dir / "summary.json"
    root_path = case_dir / "original_pose.sdf"
    selected_path = case_dir / "selected_pose.sdf"
    for path in (summary_path, root_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    if summary.get("reference_or_rmsd_label_used_for_search") is not False:
        raise RuntimeError(f"Unsafe V2 input: {summary_path}")

    gates = summary.get("parameters", {})
    max_displacement = float(gates.get("max_displacement_A", 0.75))
    max_ncc_drop = float(gates.get("max_ncc_drop", 0.02))
    if args.max_displacement_A is not None:
        if args.max_displacement_A <= 0:
            raise ValueError("--max-displacement-A must be positive")
        max_displacement = float(args.max_displacement_A)
    if args.max_ncc_drop is not None:
        if args.max_ncc_drop < 0:
            raise ValueError("--max-ncc-drop must be non-negative")
        max_ncc_drop = float(args.max_ncc_drop)
    severe_overlap = float(gates.get("severe_overlap_A", 0.8))
    sigmas = tuple(float(value) for value in gates.get("sigmas_A", (1.0, 1.5, 2.0)))
    mask_radius = float(gates.get("mask_radius_A", 10.0))
    protein_path = Path(summary["protein"])
    density_path = Path(summary["density_crop"])

    root_molecule = read_heavy_molecule(root_path)
    selected_molecule = read_heavy_molecule(selected_path)
    root_xyz = coordinates(root_molecule)
    selected_xyz = coordinates(selected_molecule)
    elements = [atom.GetSymbol() for atom in root_molecule.GetAtoms()]
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in root_molecule.GetAtoms()], dtype=np.float32
    )
    protein_xyz, protein_elements = read_protein_heavy_atoms(protein_path)
    crop = load_cartesian_density_crop(density_path)
    mask = fixed_crop_mask(crop, radius_A=mask_radius, min_voxels=64)

    def density_ncc(xyz: np.ndarray) -> float:
        return float(np.mean([
            score_gaussian_density(
                crop, np.asarray(xyz, dtype=np.float32), atomic_numbers,
                sigma, mask, cutoff_sigma=4.0,
            ).ncc
            for sigma in sigmas
        ]))

    root_ncc = float(summary["original"]["multiscale_ncc"])

    def evaluate(xyz: np.ndarray) -> RefinementEvaluation:
        steric = clash_field(
            protein_xyz, protein_elements, xyz, elements,
            severe_overlap_A=severe_overlap,
        )
        ncc = density_ncc(xyz)
        return RefinementEvaluation(
            coordinates=np.asarray(xyz, dtype=np.float64),
            clash=steric,
            displacement_A=direct_rmsd(root_xyz, xyz),
            multiscale_ncc=ncc,
            ncc_change=ncc - root_ncc,
        )

    base_evaluation = evaluate(selected_xyz)
    restart_xyz = root_xyz if args.restart_from_original else selected_xyz
    restart_molecule = root_molecule if args.restart_from_original else selected_molecule
    restart_evaluation = evaluate(restart_xyz)
    restart_id = "original_restart" if args.restart_from_original else "v2_selected"
    seeds = [(restart_id, restart_xyz, restart_evaluation)]
    variants = directed_local_variants(
        restart_molecule, protein_xyz, protein_elements, elements,
        translation_distances_A=(0.03, 0.06, 0.10),
        rotation_degrees=(1.0, 2.0),
        torsion_degrees=(2.0, 5.0),
        max_contact_directions=4,
        max_torsion_bonds=args.max_torsion_bonds,
        severe_overlap_A=severe_overlap,
    )
    feasible = []
    for variant in variants:
        candidate = evaluate(variant.coordinates)
        if (candidate.displacement_A <= max_displacement + 1e-12
                and candidate.ncc_change >= -max_ncc_drop - 1e-12):
            feasible.append((variant.variant_id, variant.coordinates, candidate))
    feasible.sort(key=lambda item: selection_key(evaluation_record(item[2])))
    if len(seeds) < args.num_starts:
        for item in feasible:
            if all(
                direct_rmsd(item[1], existing[1]) >= args.minimum_seed_rmsd_A
                for existing in seeds
            ):
                seeds.append(item)
                if len(seeds) >= args.num_starts:
                    break

    candidate_rows = [{
        "candidate_id": "v2_selected",
        "seed_id": "v2_selected",
        "candidate_type": "baseline",
        "accepted_optimizer_result": True,
        **evaluation_record(base_evaluation),
    }]
    coordinate_candidates = [(candidate_rows[0], selected_xyz)]
    start_rows = []
    for start_index, (seed_id, seed_xyz, seed_evaluation) in enumerate(seeds):
        seed_molecule = molecule_with_coordinates(root_molecule, seed_xyz)
        field = seed_evaluation.clash
        ranked_rotors = []
        for fixed, moving, rotor_mask in rotatable_bonds(seed_molecule):
            relevance = float(np.linalg.norm(field.atom_forces[rotor_mask], axis=1).sum())
            ranked_rotors.append((relevance, fixed, moving, rotor_mask))
        ranked_rotors.sort(key=lambda item: (-item[0], item[1], item[2]))
        rotors = [
            (fixed, moving, rotor_mask)
            for _, fixed, moving, rotor_mask
            in ranked_rotors[: args.max_torsion_bonds]
        ]
        refinement_function = (
            refine_seed_fast_slsqp
            if args.optimization_mode == "fast"
            else refine_seed_slsqp
        )
        result = refinement_function(
            root_coordinates=root_xyz,
            seed_coordinates=seed_xyz,
            rotors=rotors,
            protein_coordinates=protein_xyz,
            protein_elements=protein_elements,
            ligand_elements=elements,
            root_ncc=root_ncc,
            density_scorer=density_ncc,
            severe_overlap_A=severe_overlap,
            max_displacement_A=max_displacement,
            max_ncc_drop=max_ncc_drop,
            translation_bound_A=args.translation_bound_A,
            rotation_bound_deg=args.rotation_bound_deg,
            torsion_bound_deg=args.torsion_bound_deg,
            max_iterations=args.max_iterations,
        )
        candidate_id = f"slsqp_start{start_index}"
        record = {
            "candidate_id": candidate_id,
            "seed_id": seed_id,
            "candidate_type": "continuous_refinement",
            "accepted_optimizer_result": bool(result.success),
            **evaluation_record(result.evaluation),
        }
        candidate_rows.append(record)
        if result.success:
            coordinate_candidates.append((record, result.coordinates))
        start_rows.append({
            "start_index": start_index,
            "seed_id": seed_id,
            "seed_severe_clash_count": seed_evaluation.clash.severe_count,
            "seed_displacement_A": seed_evaluation.displacement_A,
            "seed_ncc_change": seed_evaluation.ncc_change,
            "num_rotors": len(rotors),
            "optimizer_success": result.optimizer_success,
            "hard_gates_pass": result.success,
            "optimizer_status": result.optimizer_status,
            "optimizer_message": result.optimizer_message,
            "iterations": result.iterations,
            "function_evaluations": result.function_evaluations,
        })
        if result.success and result.evaluation.clash.severe_count == 0:
            break

    # Continuous refinement is a fallback for the discrete severe-clash
    # endpoint.  Do not move the ligand merely for a lower smooth surrogate
    # when the reported severe-clash count is unchanged.
    strict_improvements = [
        item for item in coordinate_candidates
        if item[0]["candidate_id"] == "v2_selected"
        or int(item[0]["severe_clash_count"])
        < base_evaluation.clash.severe_count
    ]
    strict_improvements.sort(key=lambda item: selection_key(item[0]))
    selected_record, selected_coordinates = strict_improvements[0]
    for row in candidate_rows:
        row["selected"] = row["candidate_id"] == selected_record["candidate_id"]
    output.mkdir(parents=True)
    write_sdf(molecule_with_coordinates(root_molecule, selected_coordinates),
              output / "selected_pose.sdf")
    shutil.copy2(root_path, output / "original_pose.sdf")
    shutil.copy2(selected_path, output / "v2_selected_pose.sdf")
    pd.DataFrame(candidate_rows).to_csv(output / "candidate_audit.csv", index=False)
    pd.DataFrame(start_rows).to_csv(output / "optimizer_starts.csv", index=False)
    final_summary = {
        "analysis_role": (
            "parallel_original_restart_clash_refinement_v6_development"
            if args.restart_from_original
            else "continuous_constrained_clash_refinement_v3_development"
        ),
        "input_v2_case_dir": str(case_dir),
        "status": (
            "resolved_clash_free" if int(selected_record["severe_clash_count"]) == 0
            else "partial_clash_reduction"
            if int(selected_record["severe_clash_count"])
            < base_evaluation.clash.severe_count
            else "no_additional_improvement"
        ),
        "original": summary["original"],
        "v2_selected": evaluation_record(base_evaluation),
        "selected": selected_record,
        "num_requested_starts": args.num_starts,
        "num_feasible_starts": len(seeds),
        "max_iterations": args.max_iterations,
        "optimization_mode": args.optimization_mode,
        "restart_from_original": bool(args.restart_from_original),
        "restart_seed": restart_id,
        "hard_gates": {
            "maximum_direct_displacement_A": max_displacement,
            "maximum_multiscale_ncc_drop": max_ncc_drop,
            "displacement_gate_overridden": args.max_displacement_A is not None,
            "ncc_gate_overridden": args.max_ncc_drop is not None,
        },
        "reference_or_rmsd_label_used_for_search": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    print(json.dumps(final_summary, indent=2))
    print(f"CONTINUOUS_CLASH_REFINEMENT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
