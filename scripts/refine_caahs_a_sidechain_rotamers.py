#!/usr/bin/env python3
"""Rescue residual CAAHS-A clashes by conservative local side-chain rotations.

The ligand and protein backbone are immutable.  Only chi bonds in side chains
that directly clash with the ligand are rotated.  A candidate is accepted only
when it removes every severe ligand--protein clash, introduces no new severe
protein--protein contact, stays inside the side-chain displacement gate, and
(in interaction mode) preserves interaction-geometry quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.discrete_search import rotate_about_point
from cryodiffdock_pocket.interactions import (
    read_protein_interaction_sites,
    score_pose_interactions,
)
from cryodiffdock_pocket.physics import atom_radii, relative_vdw_clashes
from scripts.run_adaptive_density_search import molecule_coordinates, read_molecule


BACKBONE = {"N", "CA", "C", "O", "OXT"}

# (axis atom 1, axis atom 2, atoms downstream of axis atom 2).  These standard
# heavy-atom chi definitions preserve bond lengths and the fixed backbone.
CHI_DEFINITIONS: dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    "ARG": (("CA", "CB", ("CG", "CD", "NE", "CZ", "NH1", "NH2")),
            ("CB", "CG", ("CD", "NE", "CZ", "NH1", "NH2")),
            ("CG", "CD", ("NE", "CZ", "NH1", "NH2")),
            ("CD", "NE", ("CZ", "NH1", "NH2"))),
    "ASN": (("CA", "CB", ("CG", "OD1", "ND2")),
            ("CB", "CG", ("OD1", "ND2"))),
    "ASP": (("CA", "CB", ("CG", "OD1", "OD2")),
            ("CB", "CG", ("OD1", "OD2"))),
    "CYS": (("CA", "CB", ("SG",)),),
    "GLN": (("CA", "CB", ("CG", "CD", "OE1", "NE2")),
            ("CB", "CG", ("CD", "OE1", "NE2")),
            ("CG", "CD", ("OE1", "NE2"))),
    "GLU": (("CA", "CB", ("CG", "CD", "OE1", "OE2")),
            ("CB", "CG", ("CD", "OE1", "OE2")),
            ("CG", "CD", ("OE1", "OE2"))),
    "HIS": (("CA", "CB", ("CG", "ND1", "CD2", "CE1", "NE2")),
            ("CB", "CG", ("ND1", "CD2", "CE1", "NE2"))),
    "ILE": (("CA", "CB", ("CG1", "CG2", "CD1")),
            ("CB", "CG1", ("CD1",))),
    "LEU": (("CA", "CB", ("CG", "CD1", "CD2")),
            ("CB", "CG", ("CD1", "CD2"))),
    "LYS": (("CA", "CB", ("CG", "CD", "CE", "NZ")),
            ("CB", "CG", ("CD", "CE", "NZ")),
            ("CG", "CD", ("CE", "NZ")),
            ("CD", "CE", ("NZ",))),
    "MET": (("CA", "CB", ("CG", "SD", "CE")),
            ("CB", "CG", ("SD", "CE")),
            ("CG", "SD", ("CE",))),
    "PHE": (("CA", "CB", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")),
            ("CB", "CG", ("CD1", "CD2", "CE1", "CE2", "CZ"))),
    "SER": (("CA", "CB", ("OG",)),),
    "THR": (("CA", "CB", ("OG1", "CG2")),),
    "TRP": (("CA", "CB", ("CG", "CD1", "CD2", "NE1", "CE2", "CE3",
                              "CZ2", "CZ3", "CH2")),
            ("CB", "CG", ("CD1", "CD2", "NE1", "CE2", "CE3",
                             "CZ2", "CZ3", "CH2"))),
    "TYR": (("CA", "CB", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH")),
            ("CB", "CG", ("CD1", "CD2", "CE1", "CE2", "CZ", "OH"))),
    "VAL": (("CA", "CB", ("CG1", "CG2")),),
}


@dataclass(frozen=True)
class AtomRecord:
    line_index: int
    atom_name: str
    residue_name: str
    residue_key: tuple[str, int, str]
    element: str


@dataclass
class State:
    coordinates: np.ndarray
    ligand_clashes: int
    ligand_clashing_fraction: float
    relative_vdw_clashes: int
    minimum_relative_vdw_distance: float
    introduced_internal_clashes: int
    sidechain_displacement_A: float
    angle_penalty: float
    moves: tuple[str, ...]


def comma_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated angles")
    return values


def read_pdb(path: Path) -> tuple[list[str], list[AtomRecord], np.ndarray, list[str]]:
    lines = path.read_text().splitlines(keepends=True)
    records: list[AtomRecord] = []
    coordinates = []
    elements = []
    seen_altloc: set[tuple[str, int, str, str]] = set()
    for line_index, line in enumerate(lines):
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A"}:
            continue
        atom_name = line[12:16].strip()
        chain = line[21:22].strip() or "_"
        resnum = int(line[22:26])
        icode = line[26:27].strip()
        identity = (chain, resnum, icode, atom_name)
        if identity in seen_altloc:
            continue
        seen_altloc.add(identity)
        element = (line[76:78].strip() if len(line) >= 78 else "") or atom_name[:1]
        element = element.upper()
        if element in {"H", "D"}:
            continue
        records.append(AtomRecord(
            line_index=line_index,
            atom_name=atom_name,
            residue_name=line[17:20].strip().upper(),
            residue_key=(chain, resnum, icode),
            element=element,
        ))
        coordinates.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elements.append(element)
    if not records:
        raise ValueError(f"No protein ATOM coordinates in {path}")
    return lines, records, np.asarray(coordinates, dtype=np.float64), elements


def write_pdb(lines: list[str], records: list[AtomRecord], xyz: np.ndarray, path: Path) -> None:
    updated = list(lines)
    for record, coordinate in zip(records, xyz):
        line = updated[record.line_index]
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        body = body.ljust(80)
        body = (
            body[:30]
            + f"{coordinate[0]:8.3f}{coordinate[1]:8.3f}{coordinate[2]:8.3f}"
            + body[54:]
        )
        updated[record.line_index] = body.rstrip() + ending
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(updated))


def ligand_clashes(
    protein_xyz: np.ndarray,
    protein_elements: list[str],
    ligand_xyz: np.ndarray,
    ligand_elements: list[str],
    severe_overlap_A: float,
) -> tuple[int, float, np.ndarray]:
    distance = np.linalg.norm(
        protein_xyz[:, None, :] - ligand_xyz[None, :, :], axis=-1
    )
    overlap = (
        atom_radii(protein_elements)[:, None]
        + atom_radii(ligand_elements)[None, :]
        - distance
    )
    severe = overlap > float(severe_overlap_A)
    return int(severe.sum()), float(severe.any(axis=0).mean()), severe


def sidechain_internal_clash_pairs(
    xyz: np.ndarray,
    elements: list[str],
    records: list[AtomRecord],
    moving_indices: set[int],
    severe_overlap_A: float,
) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    radii = atom_radii(elements)
    for left in moving_indices:
        delta = xyz - xyz[left]
        distance = np.linalg.norm(delta, axis=1)
        overlap = radii + radii[left] - distance
        for right in np.flatnonzero(overlap > float(severe_overlap_A)):
            right = int(right)
            if right == left or records[right].residue_key == records[left].residue_key:
                continue
            result.add(tuple(sorted((int(left), right))))
    return result


def sidechain_rmsd(xyz: np.ndarray, baseline: np.ndarray, indices: set[int]) -> float:
    if not indices:
        return 0.0
    chosen = np.asarray(sorted(indices), dtype=int)
    return float(np.sqrt(np.mean(np.sum((xyz[chosen] - baseline[chosen]) ** 2, axis=1))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-mode", choices=("steric", "interaction"), default="steric")
    parser.add_argument("--chi-offsets", type=comma_floats, default=comma_floats("-60,-30,30,60"))
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--coordinate-rounds", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=3)
    parser.add_argument("--max-sidechain-displacement-A", type=float, default=1.5)
    parser.add_argument("--maximum-interaction-drop", type=float, default=0.05)
    parser.add_argument("--severe-overlap-A", type=float, default=0.8)
    parser.add_argument("--minimum-relative-vdw-distance", type=float)
    parser.add_argument("--method-name", default="CAAHS-A+SC")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.ligand, args.protein):
        if not path.is_file():
            raise FileNotFoundError(path)

    ligand = read_molecule(args.ligand)
    ligand_xyz = molecule_coordinates(ligand)
    ligand_elements = [atom.GetSymbol() for atom in ligand.GetAtoms()]
    lines, records, baseline_xyz, protein_elements = read_pdb(args.protein)
    original_count, original_fraction, original_pairs = ligand_clashes(
        baseline_xyz, protein_elements, ligand_xyz, ligand_elements, args.severe_overlap_A
    )
    if args.minimum_relative_vdw_distance is None:
        original_relative_count = 0
        original_minimum_relative = float("nan")
        original_relative_pairs = np.zeros_like(original_pairs, dtype=bool)
    else:
        (
            original_relative_count,
            original_minimum_relative,
            original_relative_pairs,
        ) = relative_vdw_clashes(
            baseline_xyz,
            protein_elements,
            ligand_xyz,
            ligand_elements,
            minimum_relative_distance=args.minimum_relative_vdw_distance,
        )

    target_pairs = original_pairs | original_relative_pairs
    protein_indices = np.flatnonzero(target_pairs.any(axis=1))
    backbone_contacts = [
        int(index) for index in protein_indices
        if records[int(index)].atom_name in BACKBONE
    ]
    residue_keys = sorted({records[int(index)].residue_key for index in protein_indices})
    residue_atoms: dict[tuple[str, int, str], dict[str, int]] = {}
    residue_names: dict[tuple[str, int, str], str] = {}
    for index, record in enumerate(records):
        residue_atoms.setdefault(record.residue_key, {})[record.atom_name] = index
        residue_names[record.residue_key] = record.residue_name

    supported = []
    unsupported = []
    for key in residue_keys:
        name = residue_names[key]
        definitions = CHI_DEFINITIONS.get(name, ())
        atoms = residue_atoms[key]
        usable = [definition for definition in definitions if all(
            atom in atoms for atom in (definition[0], definition[1], *definition[2])
        )]
        if usable:
            supported.append((key, name, usable))
        else:
            unsupported.append((key, name))

    status = "search_not_run"
    selected = None
    baseline_interaction = None
    selected_interaction = None
    moved_indices = {
        residue_atoms[key][atom]
        for key, _, definitions in supported
        for _, _, moving in definitions
        for atom in moving
    }
    baseline_internal = sidechain_internal_clash_pairs(
        baseline_xyz, protein_elements, records, moved_indices, args.severe_overlap_A
    )

    def evaluate(xyz: np.ndarray, penalty: float, moves: tuple[str, ...]) -> State:
        count, fraction, _ = ligand_clashes(
            xyz, protein_elements, ligand_xyz, ligand_elements, args.severe_overlap_A
        )
        internal = sidechain_internal_clash_pairs(
            xyz, protein_elements, records, moved_indices, args.severe_overlap_A
        )
        if args.minimum_relative_vdw_distance is None:
            relative_count = 0
            minimum_relative = float("nan")
        else:
            relative_count, minimum_relative, _ = relative_vdw_clashes(
                xyz,
                protein_elements,
                ligand_xyz,
                ligand_elements,
                minimum_relative_distance=args.minimum_relative_vdw_distance,
            )
        return State(
            coordinates=xyz,
            ligand_clashes=count,
            ligand_clashing_fraction=fraction,
            relative_vdw_clashes=relative_count,
            minimum_relative_vdw_distance=minimum_relative,
            introduced_internal_clashes=len(internal - baseline_internal),
            sidechain_displacement_A=sidechain_rmsd(xyz, baseline_xyz, moved_indices),
            angle_penalty=float(penalty),
            moves=moves,
        )

    initial = evaluate(baseline_xyz.copy(), 0.0, ())
    states = [initial]
    if original_count == 0 and original_relative_count == 0:
        status = "no_action_clash_free"
        selected = initial
    elif backbone_contacts:
        status = "ineligible_backbone_clash"
        selected = initial
    elif unsupported:
        status = "ineligible_unsupported_residue"
        selected = initial
    elif not 1 <= len(supported) <= args.max_residues:
        status = "ineligible_residue_count"
        selected = initial
    else:
        for _ in range(args.coordinate_rounds):
            for key, residue_name, definitions in supported:
                atoms = residue_atoms[key]
                for chi_index, (axis_left, axis_right, moving_names) in enumerate(definitions, 1):
                    expanded = list(states)
                    for state in states:
                        left = state.coordinates[atoms[axis_left]]
                        right = state.coordinates[atoms[axis_right]]
                        axis = right - left
                        moving = np.asarray([atoms[name] for name in moving_names], dtype=int)
                        for angle in args.chi_offsets:
                            xyz = state.coordinates.copy()
                            xyz[moving] = rotate_about_point(
                                xyz[moving], axis, float(angle), right
                            )
                            label = (
                                f"{key[0]}:{key[1]}{key[2]}:{residue_name}:"
                                f"chi{chi_index}:{float(angle):+g}"
                            )
                            expanded.append(evaluate(
                                xyz,
                                state.angle_penalty + (abs(float(angle)) / 60.0) ** 2,
                                (*state.moves, label),
                            ))
                    expanded.sort(key=lambda state: (
                        state.relative_vdw_clashes,
                        state.ligand_clashes,
                        state.introduced_internal_clashes,
                        state.angle_penalty,
                        state.sidechain_displacement_A,
                        state.moves,
                    ))
                    states = expanded[: args.beam_width]

        valid = [
            state for state in states
            if state.ligand_clashes == 0
            and state.relative_vdw_clashes == 0
            and state.introduced_internal_clashes == 0
            and state.sidechain_displacement_A <= args.max_sidechain_displacement_A
        ]
        candidate_dir = args.output_dir / "interaction_candidates"
        baseline_path = candidate_dir / "baseline.pdb"
        write_pdb(lines, records, baseline_xyz, baseline_path)
        try:
            baseline_sites = read_protein_interaction_sites(baseline_path)
            baseline_interaction = float(
                score_pose_interactions(baseline_sites, args.ligand)[
                    "interaction_geometry_quality"
                ]
            )
        except Exception:
            baseline_interaction = None

        scored = []
        for index, state in enumerate(valid):
            interaction = None
            if args.selection_mode == "interaction":
                protein_path = candidate_dir / f"candidate_{index:03d}.pdb"
                write_pdb(lines, records, state.coordinates, protein_path)
                try:
                    sites = read_protein_interaction_sites(protein_path)
                    interaction = float(
                        score_pose_interactions(sites, args.ligand)[
                            "interaction_geometry_quality"
                        ]
                    )
                except Exception:
                    interaction = None
            scored.append((state, interaction))

        if args.selection_mode == "interaction":
            scored = [
                item for item in scored
                if item[1] is not None
                and (
                    baseline_interaction is None
                    or item[1] >= baseline_interaction - args.maximum_interaction_drop
                )
            ]
            scored.sort(key=lambda item: (
                -float(item[1]),
                item[0].angle_penalty,
                item[0].sidechain_displacement_A,
                item[0].moves,
            ))
        else:
            scored.sort(key=lambda item: (
                item[0].angle_penalty,
                item[0].sidechain_displacement_A,
                item[0].moves,
            ))
        if scored:
            selected, selected_interaction = scored[0]
            status = "resolved_clash_free"
        else:
            selected = initial
            status = "no_admissible_clash_free_rotamer"

        # In the steric control, measure interaction geometry only after the
        # selection decision so it remains an outcome rather than a selector.
        if (
            args.selection_mode == "steric"
            and status == "resolved_clash_free"
            and baseline_interaction is not None
        ):
            audit_path = candidate_dir / "steric_selected_audit.pdb"
            write_pdb(lines, records, selected.coordinates, audit_path)
            try:
                selected_sites = read_protein_interaction_sites(audit_path)
                selected_interaction = float(
                    score_pose_interactions(selected_sites, args.ligand)[
                        "interaction_geometry_quality"
                    ]
                )
            except Exception:
                selected_interaction = None

    shutil.copy2(args.ligand, args.output_dir / "selected_pose.sdf")
    write_pdb(lines, records, selected.coordinates, args.output_dir / "selected_protein.pdb")
    summary = {
        "status": status,
        "method": args.method_name,
        "selection_mode": args.selection_mode,
        "original_severe_clash_count": original_count,
        "selected_severe_clash_count": selected.ligand_clashes,
        "original_clashing_ligand_atom_fraction": original_fraction,
        "selected_clashing_ligand_atom_fraction": selected.ligand_clashing_fraction,
        "original_relative_vdw_clash_count": original_relative_count,
        "selected_relative_vdw_clash_count": selected.relative_vdw_clashes,
        "original_minimum_relative_vdw_distance": (
            original_minimum_relative
            if args.minimum_relative_vdw_distance is not None else None
        ),
        "selected_minimum_relative_vdw_distance": (
            selected.minimum_relative_vdw_distance
            if args.minimum_relative_vdw_distance is not None else None
        ),
        "minimum_relative_vdw_distance_gate": args.minimum_relative_vdw_distance,
        "introduced_internal_protein_clashes": selected.introduced_internal_clashes,
        "sidechain_displacement_A": selected.sidechain_displacement_A,
        "angle_penalty": selected.angle_penalty,
        "selected_moves": list(selected.moves),
        "target_residues": [
            {"chain": key[0], "residue_number": key[1], "insertion_code": key[2],
             "residue_name": name, "num_chi": len(definitions)}
            for key, name, definitions in supported
        ],
        "unsupported_residues": [
            {"chain": key[0], "residue_number": key[1], "insertion_code": key[2],
             "residue_name": name}
            for key, name in unsupported
        ],
        "backbone_clash_atom_count": len(backbone_contacts),
        "num_valid_clash_free_candidates": len(valid) if 'valid' in locals() else 0,
        "baseline_interaction_geometry_quality": baseline_interaction,
        "selected_interaction_geometry_quality": selected_interaction,
        "interaction_geometry_change": (
            selected_interaction - baseline_interaction
            if selected_interaction is not None and baseline_interaction is not None
            else None
        ),
        "maximum_interaction_drop": args.maximum_interaction_drop,
        "maximum_sidechain_displacement_A": args.max_sidechain_displacement_A,
        "ligand_coordinates_changed": False,
        "backbone_coordinates_changed": False,
        "uses_reference_or_native_rmsd": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
