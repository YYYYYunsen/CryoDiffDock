#!/usr/bin/env python3
"""Validation-stage clash-directed, density-preserving ligand repair v2."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.clash_repair import (  # noqa: E402
    clash_field,
    directed_local_variants,
    diverse_beam_indices,
)
from cryodiffdock_pocket.density_ranking import (  # noqa: E402
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
)
from cryodiffdock_pocket.discrete_search import (  # noqa: E402
    direct_rmsd,
    generate_local_variants,
)
from cryodiffdock_pocket.physics import read_protein_heavy_atoms  # noqa: E402


def comma_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated values")
    return values


def read_heavy_molecule(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand pose: {path}")
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
        # V2000 uses fixed-width coordinate fields and can silently produce an
        # unreadable mol block for structures with large absolute coordinates.
        xyz = coordinates(molecule)
        if not np.isfinite(xyz).all():
            raise ValueError("Cannot write nonfinite ligand coordinates")
        if np.max(np.abs(xyz)) >= 900.0:
            writer.SetForceV3000(True)
        writer.write(molecule)
    finally:
        writer.close()


@dataclass
class Candidate:
    candidate_id: str
    parent_id: str
    depth: int
    move_type: str
    coordinates: np.ndarray
    displacement_A: float
    multiscale_ncc: float
    ncc_change: float
    severe_clash_count: int
    clashing_ligand_atom_fraction: float
    max_vdw_overlap_A: float
    min_protein_ligand_distance_A: float
    smooth_clash_energy: float
    passes_displacement_gate: bool
    passes_density_gate: bool
    passes_search_displacement_gate: bool
    passes_search_density_gate: bool

    def selection_key(self) -> tuple[float, ...]:
        ncc = self.multiscale_ncc if math.isfinite(self.multiscale_ncc) else -math.inf
        if self.severe_clash_count == 0:
            return (
                0.0, self.displacement_A, -ncc,
                self.smooth_clash_energy, float(self.depth),
            )
        return (
            float(self.severe_clash_count),
            self.smooth_clash_energy,
            self.clashing_ligand_atom_fraction,
            self.max_vdw_overlap_A,
            -ncc,
            self.displacement_A,
        )


def _records(candidates: list[Candidate], selected: Candidate) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        row = asdict(candidate)
        row.pop("coordinates")
        row["selected"] = candidate.candidate_id == selected.candidate_id
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--density-crop", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-displacement-A", type=float, default=0.75)
    parser.add_argument("--max-ncc-drop", type=float, default=0.02)
    parser.add_argument("--intermediate-displacement-slack-A", type=float, default=0.05)
    parser.add_argument("--intermediate-ncc-slack", type=float, default=0.005)
    parser.add_argument("--severe-overlap-A", type=float, default=0.8)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-size", type=int, default=12)
    parser.add_argument("--minimum-beam-rmsd-A", type=float, default=0.04)
    parser.add_argument("--translations-A", type=comma_floats,
                        default=comma_floats("0.1,0.25,0.5"))
    parser.add_argument("--rotations-deg", type=comma_floats,
                        default=comma_floats("2,5,10"))
    parser.add_argument("--torsions-deg", type=comma_floats,
                        default=comma_floats("5,10,20"))
    parser.add_argument("--directed-translations-A", type=comma_floats,
                        default=comma_floats("0.03,0.06,0.1,0.2,0.35"))
    parser.add_argument("--directed-rotations-deg", type=comma_floats,
                        default=comma_floats("1,2,5"))
    parser.add_argument("--directed-torsions-deg", type=comma_floats,
                        default=comma_floats("2,5,10"))
    parser.add_argument("--max-contact-directions", type=int, default=4)
    parser.add_argument("--max-torsion-bonds", type=int, default=8)
    parser.add_argument("--sigmas-A", type=comma_floats,
                        default=comma_floats("1,1.5,2"))
    parser.add_argument("--mask-radius-A", type=float, default=10.0)
    parser.add_argument("--generic-proposals", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--adaptive-early-stop", action="store_true")
    parser.add_argument(
        "--density-prescreen-multiplier", type=int, default=0,
        help="If positive, exact-NCC score only this many beam-size multiples per depth",
    )
    parser.add_argument(
        "--protein-neighbor-radius-A", type=float, default=0.0,
        help="Use only protein atoms within this root-ligand radius during search",
    )
    parser.add_argument("--method-version", default="clash_directed_v2_development")
    args = parser.parse_args()

    started = time.perf_counter()
    ligand_path = args.ligand.expanduser().resolve()
    protein_path = args.protein.expanduser().resolve()
    crop_path = args.density_crop.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for path in (ligand_path, protein_path, crop_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.search_depth < 1 or args.beam_size < 1:
        raise ValueError("search depth and beam size must be positive")
    if args.intermediate_displacement_slack_A < 0 or args.intermediate_ncc_slack < 0:
        raise ValueError("intermediate slacks must be non-negative")

    output.mkdir(parents=True)
    molecule = read_heavy_molecule(ligand_path)
    root_xyz = coordinates(molecule)
    elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.float32
    )
    full_protein_xyz, full_protein_elements = read_protein_heavy_atoms(protein_path)
    protein_xyz, protein_elements = full_protein_xyz, full_protein_elements
    empty_neighbor_list_fallback = False
    if args.protein_neighbor_radius_A > 0:
        tree = cKDTree(full_protein_xyz)
        indices = sorted({
            index
            for atom in root_xyz
            for index in tree.query_ball_point(atom, args.protein_neighbor_radius_A)
        })
        # The radius is only a search acceleration.  A pose with no nearby
        # protein atoms still has a well-defined full-receptor steric score;
        # use that exact score instead of failing the whole case.
        if indices:
            protein_xyz = full_protein_xyz[np.asarray(indices, dtype=int)]
            protein_elements = [full_protein_elements[index] for index in indices]
        else:
            empty_neighbor_list_fallback = True
    crop = load_cartesian_density_crop(crop_path)
    mask = fixed_crop_mask(crop, radius_A=args.mask_radius_A, min_voxels=64)

    def density_ncc(xyz: np.ndarray) -> float:
        return float(np.mean([
            score_gaussian_density(
                crop, xyz.astype(np.float32), atomic_numbers, sigma, mask,
                cutoff_sigma=4.0,
            ).ncc
            for sigma in args.sigmas_A
        ]))

    root_ncc = density_ncc(root_xyz)

    def evaluate(candidate_id: str, parent_id: str, depth: int,
                 move_type: str, xyz: np.ndarray, *, score_density: bool = True) -> Candidate:
        xyz = np.asarray(xyz, dtype=np.float64)
        displacement = direct_rmsd(root_xyz, xyz)
        search_displacement = (
            displacement <= args.max_displacement_A
            + args.intermediate_displacement_slack_A + 1e-12
        )
        steric = clash_field(
            protein_xyz, protein_elements, xyz, elements,
            severe_overlap_A=args.severe_overlap_A,
        )
        # An over-budget pose cannot be selected.  Avoid the three expensive
        # density renders, but retain its steric diagnostics in the audit.
        ncc = density_ncc(xyz) if search_displacement and score_density else float("nan")
        change = ncc - root_ncc if math.isfinite(ncc) else float("nan")
        passes_displacement = displacement <= args.max_displacement_A + 1e-12
        passes_density = math.isfinite(change) and change >= -args.max_ncc_drop - 1e-12
        search_density = (
            math.isfinite(change)
            and change >= -args.max_ncc_drop - args.intermediate_ncc_slack - 1e-12
        )
        return Candidate(
            candidate_id, parent_id, depth, move_type, xyz,
            displacement, ncc, change,
            steric.severe_count,
            steric.clashing_ligand_atom_fraction,
            steric.max_overlap_A,
            steric.min_distance_A,
            steric.energy, passes_displacement, passes_density,
            search_displacement, search_density,
        )

    def add_density(candidate: Candidate) -> Candidate:
        if math.isfinite(candidate.multiscale_ncc):
            return candidate
        ncc = density_ncc(candidate.coordinates)
        candidate.multiscale_ncc = ncc
        candidate.ncc_change = ncc - root_ncc
        candidate.passes_density_gate = (
            candidate.ncc_change >= -args.max_ncc_drop - 1e-12
        )
        candidate.passes_search_density_gate = (
            candidate.ncc_change
            >= -args.max_ncc_drop - args.intermediate_ncc_slack - 1e-12
        )
        return candidate

    root = evaluate("root", "", 0, "original", root_xyz)
    root.multiscale_ncc = root_ncc
    root.ncc_change = 0.0
    root.passes_density_gate = True
    root.passes_search_density_gate = True
    all_candidates = [root]
    selected = root
    status = "no_action_clash_free"

    if root.severe_clash_count > 0:
        beam = [root]
        seen = {np.round(root_xyz, 4).tobytes()}
        for depth in range(1, args.search_depth + 1):
            children: list[Candidate] = []
            for parent in beam:
                parent_molecule = molecule_with_coordinates(molecule, parent.coordinates)
                variants = directed_local_variants(
                    parent_molecule, protein_xyz, protein_elements, elements,
                    translation_distances_A=args.directed_translations_A,
                    rotation_degrees=args.directed_rotations_deg,
                    torsion_degrees=args.directed_torsions_deg,
                    max_contact_directions=args.max_contact_directions,
                    max_torsion_bonds=args.max_torsion_bonds,
                    severe_overlap_A=args.severe_overlap_A,
                )
                if args.generic_proposals:
                    variants.extend(generate_local_variants(
                        parent_molecule,
                        rotation_degrees=args.rotations_deg,
                        translation_distances_A=args.translations_A,
                        torsion_degrees=args.torsions_deg,
                        max_torsion_bonds=args.max_torsion_bonds,
                        include_coarse_orientations=False,
                    )[1:])
                for variant in variants:
                    signature = np.round(variant.coordinates, 4).tobytes()
                    if signature in seen:
                        continue
                    seen.add(signature)
                    child = evaluate(
                        f"d{depth}_{parent.candidate_id}_{variant.variant_id}",
                        parent.candidate_id, depth, variant.move_type,
                        variant.coordinates,
                        score_density=args.density_prescreen_multiplier <= 0,
                    )
                    all_candidates.append(child)
                    if child.passes_search_displacement_gate:
                        children.append(child)
            if not children:
                break
            if args.density_prescreen_multiplier > 0:
                children.sort(key=lambda candidate: candidate.selection_key())
                limit = max(
                    args.beam_size,
                    args.beam_size * args.density_prescreen_multiplier,
                )
                children = [add_density(candidate) for candidate in children[:limit]]
            children = [
                candidate for candidate in children
                if candidate.passes_search_density_gate
            ]
            if not children:
                break
            indices = diverse_beam_indices(
                children, beam_size=args.beam_size,
                minimum_rmsd_A=args.minimum_beam_rmsd_A,
            )
            beam = [children[index] for index in indices]
            if args.adaptive_early_stop and any(
                candidate.severe_clash_count == 0
                and candidate.passes_displacement_gate
                and candidate.passes_density_gate
                for candidate in beam
            ):
                break

        acceptable = [
            candidate for candidate in all_candidates
            if candidate.passes_displacement_gate
            and candidate.passes_density_gate
            and (
                candidate.severe_clash_count < root.severe_clash_count
                or candidate.smooth_clash_energy < root.smooth_clash_energy - 1e-12
            )
        ]
        if acceptable:
            acceptable.sort(key=lambda candidate: candidate.selection_key())
            selected = acceptable[0]
            status = (
                "resolved_clash_free" if selected.severe_clash_count == 0
                else "partial_clash_reduction"
            )
        else:
            status = "unresolved_no_acceptable_improvement"

    # The neighbor list is only a search acceleration.  Recompute the selected
    # steric endpoint against the complete receptor before writing output.
    full_steric = clash_field(
        full_protein_xyz, full_protein_elements, selected.coordinates, elements,
        severe_overlap_A=args.severe_overlap_A,
    )
    selected.severe_clash_count = full_steric.severe_count
    selected.clashing_ligand_atom_fraction = full_steric.clashing_ligand_atom_fraction
    selected.max_vdw_overlap_A = full_steric.max_overlap_A
    selected.min_protein_ligand_distance_A = full_steric.min_distance_A
    selected.smooth_clash_energy = full_steric.energy
    status = (
        "no_action_clash_free" if root.severe_clash_count == 0
        else "resolved_clash_free" if selected.severe_clash_count == 0
        else "partial_clash_reduction"
        if selected.severe_clash_count < root.severe_clash_count
        else "unresolved_no_acceptable_improvement"
    )

    write_sdf(molecule_with_coordinates(molecule, selected.coordinates),
              output / "selected_pose.sdf")
    shutil.copy2(ligand_path, output / "original_pose.sdf")
    audit = _records(all_candidates, selected)
    audit.to_csv(output / "candidate_audit.csv", index=False)
    clash_free = audit["severe_clash_count"] == 0
    runtime_seconds = time.perf_counter() - started
    summary = {
        "analysis_role": "validation_tuned_clash_directed_density_preserving_repair_v2",
        "method_version": args.method_version,
        "status": status,
        "input_ligand": str(ligand_path),
        "protein": str(protein_path),
        "density_crop": str(crop_path),
        "original": {key: value for key, value in asdict(root).items()
                     if key != "coordinates"},
        "selected": {key: value for key, value in asdict(selected).items()
                     if key != "coordinates"},
        "candidate_oracle": {
            "any_clash_free": bool(clash_free.any()),
            "clash_free_with_displacement_gate": bool(
                (clash_free & audit["passes_displacement_gate"]).any()
            ),
            "clash_free_with_density_gate": bool(
                (clash_free & audit["passes_density_gate"]).any()
            ),
            "clash_free_with_both_final_gates": bool(
                (clash_free & audit["passes_displacement_gate"]
                 & audit["passes_density_gate"]).any()
            ),
        },
        "parameters": vars(args) | {
            "ligand": str(ligand_path), "protein": str(protein_path),
            "density_crop": str(crop_path), "output_dir": str(output),
        },
        "reference_or_rmsd_label_used_for_search": False,
        "candidate_count": len(all_candidates),
        "density_scored_candidate_count": int(audit["multiscale_ncc"].notna().sum()),
        "search_protein_atom_count": len(protein_xyz),
        "full_protein_atom_count": len(full_protein_xyz),
        "empty_neighbor_list_full_protein_fallback": empty_neighbor_list_fallback,
        "full_receptor_final_verification": True,
        "runtime_seconds": runtime_seconds,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"DENSITY_PRESERVING_CLASH_REPAIR_V2_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
