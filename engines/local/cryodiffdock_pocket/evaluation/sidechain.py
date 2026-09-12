"""Side-chain accuracy metrics for DiffDock-Pocket protein conformers.

The evaluator matches residues by chain, residue number and insertion code,
aligns the prediction to the reference with common backbone atoms, and reports
both side-chain heavy-atom RMSD and circular chi-angle errors.  Symmetric
terminal atoms are handled by choosing the lower-RMSD equivalent naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from Bio.PDB import PDBParser


BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

# Standard heavy-atom definitions. A missing atom makes only the affected chi
# unavailable; it does not discard the whole residue from atom RMSD.
CHI_ATOMS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
            ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
            ("CB", "CG", "SD", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}

# Each permutation must be applied as a group. This preserves the geometry of
# phenyl rings while allowing chemically indistinguishable atom naming.
SYMMETRIC_SWAPS: dict[str, tuple[tuple[tuple[str, str], ...], ...]] = {
    "ARG": ((("NH1", "NH2"),),),
    "ASP": ((("OD1", "OD2"),),),
    "GLU": ((("OE1", "OE2"),),),
    "LEU": ((("CD1", "CD2"),),),
    "PHE": ((("CD1", "CD2"), ("CE1", "CE2")),),
    "TYR": ((("CD1", "CD2"), ("CE1", "CE2")),),
    "VAL": ((("CG1", "CG2"),),),
}

# Terminal symmetry makes these torsions 180-degree periodic.
CHI_PERIOD_180 = {("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2), ("ARG", 4)}


@dataclass(frozen=True, order=True)
class ResidueId:
    chain: str
    number: int
    insertion_code: str = ""

    def label(self) -> str:
        return f"{self.chain}:{self.number}{self.insertion_code}".rstrip()


@dataclass
class ResidueRecord:
    residue_id: ResidueId
    name: str
    atoms: dict[str, np.ndarray]


def _load_residues(path: str | Path) -> dict[ResidueId, ResidueRecord]:
    structure = PDBParser(QUIET=True).get_structure("protein", str(path))
    model = next(structure.get_models())
    records: dict[ResidueId, ResidueRecord] = {}
    for chain in model:
        for residue in chain:
            hetero, number, insertion = residue.id
            if hetero.strip():
                continue
            residue_id = ResidueId(chain.id.strip() or "_", int(number), insertion.strip())
            atoms: dict[str, np.ndarray] = {}
            for atom in residue:
                name = atom.name.strip()
                element = (atom.element or "").strip().upper()
                if element == "H" or name.startswith("H"):
                    continue
                atoms[name] = np.asarray(atom.coord, dtype=np.float64)
            records[residue_id] = ResidueRecord(residue_id, residue.resname.strip().upper(), atoms)
    return records


def parse_residue_ids(values: str | Iterable[str] | None) -> set[ResidueId] | None:
    """Parse ``A:123-A:124B`` or an iterable of ``chain:number`` labels."""
    if values is None:
        return None
    tokens = values.split("-") if isinstance(values, str) else list(values)
    parsed: set[ResidueId] = set()
    for raw in tokens:
        raw = str(raw).strip()
        if not raw:
            continue
        chain, sep, number_text = raw.partition(":")
        if not sep:
            raise ValueError(f"Invalid residue label {raw!r}; expected CHAIN:NUMBER")
        digits = "".join(char for char in number_text if char in "-0123456789")
        insertion = number_text[len(digits):].strip()
        parsed.add(ResidueId(chain.strip() or "_", int(digits), insertion))
    return parsed


def _backbone_alignment(
    reference: Mapping[ResidueId, ResidueRecord],
    prediction: Mapping[ResidueId, ResidueRecord],
) -> tuple[np.ndarray, np.ndarray, int]:
    ref_points: list[np.ndarray] = []
    pred_points: list[np.ndarray] = []
    for residue_id in sorted(set(reference) & set(prediction)):
        ref = reference[residue_id]
        pred = prediction[residue_id]
        if ref.name != pred.name:
            continue
        for atom_name in ("N", "CA", "C", "O"):
            if atom_name in ref.atoms and atom_name in pred.atoms:
                ref_points.append(ref.atoms[atom_name])
                pred_points.append(pred.atoms[atom_name])
    if len(ref_points) < 3:
        return np.eye(3), np.zeros(3), len(ref_points)
    ref_array = np.asarray(ref_points)
    pred_array = np.asarray(pred_points)
    ref_center = ref_array.mean(axis=0)
    pred_center = pred_array.mean(axis=0)
    covariance = (pred_array - pred_center).T @ (ref_array - ref_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = ref_center - rotation @ pred_center
    return rotation, translation, len(ref_points)


def _transform_records(
    records: Mapping[ResidueId, ResidueRecord], rotation: np.ndarray, translation: np.ndarray
) -> dict[ResidueId, ResidueRecord]:
    return {
        residue_id: ResidueRecord(
            residue_id,
            record.name,
            {name: rotation @ coord + translation for name, coord in record.atoms.items()},
        )
        for residue_id, record in records.items()
    }


def _dihedral(points: Sequence[np.ndarray]) -> float:
    p0, p1, p2, p3 = (np.asarray(point, dtype=np.float64) for point in points)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-8:
        return float("nan")
    b1 /= norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _circular_error(reference: float, prediction: float, period: float = 360.0) -> float:
    return float(abs((prediction - reference + period / 2.0) % period - period / 2.0))


def _sidechain_atom_pairs(
    reference: ResidueRecord, prediction: ResidueRecord
) -> tuple[np.ndarray, np.ndarray, bool]:
    names = sorted(
        (set(reference.atoms) & set(prediction.atoms)) - BACKBONE_ATOMS
    )
    if not names:
        return np.empty((0, 3)), np.empty((0, 3)), False
    reference_coords = np.asarray([reference.atoms[name] for name in names])
    base_prediction = {name: prediction.atoms[name] for name in names}
    candidates = [base_prediction]
    for swap_group in SYMMETRIC_SWAPS.get(reference.name, ()):
        swapped = dict(base_prediction)
        valid = True
        for left, right in swap_group:
            if left not in swapped or right not in swapped:
                valid = False
                break
            swapped[left], swapped[right] = swapped[right], swapped[left]
        if valid:
            candidates.append(swapped)
    best = min(
        candidates,
        key=lambda candidate: float(np.square(
            np.asarray([candidate[name] for name in names]) - reference_coords
        ).sum()),
    )
    return reference_coords, np.asarray([best[name] for name in names]), best is not base_prediction


def compare_sidechains(
    reference_path: str | Path,
    prediction_path: str | Path,
    residue_ids: str | Iterable[str] | None = None,
    align_backbone: bool = True,
) -> dict[str, object]:
    """Compare one predicted protein conformer with a deposited reference."""
    reference = _load_residues(reference_path)
    prediction = _load_residues(prediction_path)
    selected = parse_residue_ids(residue_ids)
    rotation, translation, alignment_atoms = _backbone_alignment(reference, prediction)
    if align_backbone:
        prediction = _transform_records(prediction, rotation, translation)

    rows: list[dict[str, object]] = []
    all_squared_errors: list[float] = []
    all_chi_errors: list[float] = []
    skipped_mismatch = 0
    common_ids = sorted(set(reference) & set(prediction))
    for residue_id in common_ids:
        if selected is not None and residue_id not in selected:
            continue
        ref = reference[residue_id]
        pred = prediction[residue_id]
        if ref.name != pred.name:
            skipped_mismatch += 1
            continue
        ref_coords, pred_coords, symmetry_swapped = _sidechain_atom_pairs(ref, pred)
        if len(ref_coords) == 0:
            continue
        squared = np.square(pred_coords - ref_coords).sum(axis=1)
        all_squared_errors.extend(float(value) for value in squared)
        row: dict[str, object] = {
            "residue": residue_id.label(),
            "resname": ref.name,
            "num_matched_sidechain_heavy_atoms": int(len(ref_coords)),
            "sidechain_rmsd_A": float(np.sqrt(squared.mean())),
            "symmetry_swap_used": symmetry_swapped,
        }
        for chi_index, atom_names in enumerate(CHI_ATOMS.get(ref.name, ()), start=1):
            if not all(name in ref.atoms and name in pred.atoms for name in atom_names):
                continue
            ref_angle = _dihedral([ref.atoms[name] for name in atom_names])
            pred_angle = _dihedral([pred.atoms[name] for name in atom_names])
            period = 180.0 if (ref.name, chi_index) in CHI_PERIOD_180 else 360.0
            error = _circular_error(ref_angle, pred_angle, period)
            row[f"chi{chi_index}_error_deg"] = error
            all_chi_errors.append(error)
        rows.append(row)

    aggregate_rmsd = (
        float(np.sqrt(np.mean(all_squared_errors))) if all_squared_errors else float("nan")
    )
    return {
        "reference_path": str(reference_path),
        "prediction_path": str(prediction_path),
        "aligned_on_backbone": bool(align_backbone),
        "num_alignment_atoms": int(alignment_atoms),
        "num_evaluated_residues": len(rows),
        "num_matched_sidechain_heavy_atoms": len(all_squared_errors),
        "sidechain_rmsd_A": aggregate_rmsd,
        "sidechain_rmsd_lt_1A": bool(aggregate_rmsd < 1.0) if all_squared_errors else False,
        "mean_chi_error_deg": float(np.mean(all_chi_errors)) if all_chi_errors else float("nan"),
        "chi_accuracy_20deg": float(np.mean(np.asarray(all_chi_errors) < 20.0)) if all_chi_errors else float("nan"),
        "chi_accuracy_40deg": float(np.mean(np.asarray(all_chi_errors) < 40.0)) if all_chi_errors else float("nan"),
        "num_chi_angles": len(all_chi_errors),
        "num_residue_name_mismatches": skipped_mismatch,
        "per_residue": rows,
    }
