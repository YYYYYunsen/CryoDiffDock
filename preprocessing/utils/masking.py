"""Utilities for building reproducible cryo-EM density masks.

The feature structure and mask structure intentionally have different roles:

* the feature structure defines the atoms stored in ``ring_features.npz``;
* the mask structure defines which modeled density is removed from the map.

Keeping these operations separate lets DiffDock retain its cropped receptor
graph while the map is masked with the complete deposited mmCIF model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import gemmi
import numpy as np

from utils.maps import is_nucleic_residue, is_protein_residue, is_water_or_ion


ResidueKey = tuple[str, str, int, str]


# Categories that can never be the docking target: the dataset filter in
# ``02_filter_candidates.py`` rejects ions, glycans and lipids as ligands.
# Masking them therefore uses only comp_id class knowledge, which a prospective
# user has for their own model, and never uses knowledge of the answer.
POLYMER_CATEGORIES = frozenset({"protein_polymer", "nucleic_acid", "other_polymer"})
KNOWN_NONTARGET_CATEGORIES = frozenset(
    {"water", "ion", "glycan", "lipid", "detergent"}
)


# This list is deliberately conservative.  Unknown non-polymers remain
# ``other_nonpolymer`` rather than being guessed to be lipid/detergent.
LIPID_RESNAMES = {
    "CLR", "CHL", "CHD", "Y01", "OHC", "BCN", "OLA", "PAM", "PLM",
    "MYR", "STE", "LDA", "ACD", "EIC", "OCT", "PC1", "PEE", "PLC",
    "PTD", "P6L", "PGW", "L2L", "CER", "SPH", "MC3", "POPC", "POPE",
    "POPG", "DOPC", "DOPE", "DOPS", "DPPC", "DMPC", "DLPC", "DSPC",
    "CHS", "LMT", "LMU",
}

DETERGENT_RESNAMES = {
    "DDM", "LDAO", "BOG", "BNG", "OG", "NG", "DM", "UDM", "C8E",
    "C10E", "C12E", "LMPG", "LMNG", "GDN", "DNG", "HEC", "FC12",
    "FOS", "TRS", "CYO", "BAM", "DAO", "DPC",
}

GLYCAN_RESNAMES = {
    "NAG", "NDG", "BMA", "MAN", "GAL", "GLC", "FUC", "SIA", "NAN",
    "BGC", "A2G", "A3G", "A4G", "A6P", "G6D", "XYS", "XYP", "FUL",
}


def residue_key(chain: gemmi.Chain, residue: gemmi.Residue) -> ResidueKey:
    return (
        chain.name.strip(),
        residue.name.strip(),
        int(residue.seqid.num),
        residue.seqid.icode.strip(),
    )


def _entity_type_name(residue: gemmi.Residue) -> str:
    try:
        return str(residue.entity_type).split(".")[-1]
    except Exception:
        return "Unknown"


def classify_residue(residue: gemmi.Residue) -> str:
    """Classify one mmCIF residue for masking and attribution."""
    name = residue.name.strip().upper()
    entity_type = _entity_type_name(residue)

    if entity_type == "Water" or name in {"HOH", "WAT", "DOD", "H2O"}:
        return "water"

    # Name-based assignments precede entity fallbacks. Gemmi may infer an
    # isolated HETATM ligand chain in a PDB file as Polymer, while deposited
    # mmCIF files normally carry authoritative entity typing.
    if is_nucleic_residue(residue):
        return "nucleic_acid"
    if is_protein_residue(residue):
        return "protein_polymer"
    if name in LIPID_RESNAMES:
        return "lipid"
    if name in DETERGENT_RESNAMES:
        return "detergent"
    if name in GLYCAN_RESNAMES:
        return "glycan"
    if entity_type == "Branched":
        return "glycan"
    if entity_type == "Polymer" and residue.het_flag != "H":
        return "other_polymer"
    if is_water_or_ion(residue):
        if name in {"HOH", "WAT", "DOD", "H2O"}:
            return "water"
        return "ion"

    heavy_atoms = [
        atom for atom in residue
        if atom.element.name.upper() not in {"H", "D", "T"}
    ]
    if len(heavy_atoms) == 1 and heavy_atoms[0].element.is_metal:
        return "ion"
    return "other_nonpolymer"


def read_heavy_atom_coordinates(path: str | Path) -> np.ndarray:
    """Read heavy-atom coordinates from a PDB/mmCIF ligand extract."""
    structure = gemmi.read_structure(str(path))
    structure.remove_alternative_conformations()
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element.name.upper() in {"H", "D", "T"}:
                        continue
                    coordinates.append([atom.pos.x, atom.pos.y, atom.pos.z])
        break
    if not coordinates:
        raise ValueError(f"no heavy atoms found in target ligand: {path}")
    return np.asarray(coordinates, dtype=np.float64)


def locate_target_residue_keys(
    structure: gemmi.Structure,
    target_ligand_path: str | Path,
    coordinate_tolerance: float = 0.15,
    minimum_match_fraction: float = 0.90,
    all_copies: bool = False,
) -> set[ResidueKey]:
    """Locate the extracted target ligand instance in a complete mmCIF model.

    Coordinate matching is used instead of chain/residue identifiers because
    ligand extraction can rename the residue (commonly to chain L, residue 1).

    With ``all_copies``, every residue sharing the matched comp_id is returned.
    Homo-oligomers commonly carry several symmetry-related copies of the same
    ligand; masking the copies that are not the deposited ground truth would use
    knowledge of which copy is the answer, which a prospective user lacks.
    """
    target = read_heavy_atom_coordinates(target_ligand_path)
    if not len(structure):
        raise ValueError("mask structure contains no model")

    candidates: list[tuple[int, float, ResidueKey]] = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            if classify_residue(residue) in {"water", "ion"}:
                continue
            xyz = np.asarray(
                [
                    [atom.pos.x, atom.pos.y, atom.pos.z]
                    for atom in residue
                    if atom.element.name.upper() not in {"H", "D", "T"}
                ],
                dtype=np.float64,
            )
            if not len(xyz):
                continue
            distances = np.linalg.norm(target[:, None, :] - xyz[None, :, :], axis=2)
            nearest = distances.min(axis=1)
            matched = int(np.count_nonzero(nearest <= coordinate_tolerance))
            if matched:
                candidates.append((matched, float(nearest.mean()), residue_key(chain, residue)))

    if not candidates:
        raise ValueError(
            f"target ligand coordinates did not match any residue in the mask structure: "
            f"{target_ligand_path}"
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched, mean_distance, key = candidates[0]
    fraction = matched / len(target)
    if fraction < minimum_match_fraction:
        raise ValueError(
            f"best target ligand match is incomplete: matched={matched}/{len(target)} "
            f"({fraction:.1%}), mean nearest distance={mean_distance:.3f} A, key={key}"
        )
    if not all_copies:
        return {key}

    comp_id = key[1]
    return {
        residue_key(chain, residue)
        for chain in model
        for residue in chain
        if residue.name.strip() == comp_id
    }


def prepare_mask_model(
    feature_structure_path: str | Path,
    mask_structure_path: Optional[str | Path] = None,
    mask_policy: str = "feature_structure",
    target_ligand_path: Optional[str | Path] = None,
    target_all_copies: bool = False,
) -> tuple[gemmi.Model, dict[str, object]]:
    """Create a model whose non-mask atoms have zero occupancy.

    Policies
    --------
    ``feature_structure``
        Backward-compatible behavior: mask every atom in the cropped feature
        structure. ``mask_structure_path`` is ignored.
    ``full_polymer``
        Mask all protein, nucleic-acid and other polymer atoms in the complete
        mask structure. Non-polymers remain visible.
    ``polymer_plus_known_nontarget``
        Additionally mask the non-polymer classes the dataset filter excludes
        from being a target (water, ion, glycan, lipid, detergent). Only
        comp_id class knowledge is used, so no target identification is needed
        and the policy is reproducible at inference time.
    ``all_known_nontarget``
        Mask every modeled atom except the located target ligand. Requires
        ``target_ligand_path``; see ``target_all_copies``.
    """
    valid_policies = {
        "feature_structure",
        "full_polymer",
        "polymer_plus_known_nontarget",
        "all_known_nontarget",
    }
    if mask_policy not in valid_policies:
        raise ValueError(f"unknown mask policy {mask_policy!r}; choose {sorted(valid_policies)}")

    if mask_policy == "feature_structure":
        selected_path = Path(feature_structure_path)
    else:
        if mask_structure_path is None:
            raise ValueError(f"mask_structure_path is required for mask policy {mask_policy}")
        selected_path = Path(mask_structure_path)

    structure = gemmi.read_structure(str(selected_path))
    structure.remove_alternative_conformations()
    try:
        structure.setup_entities()
        structure.add_entity_types(overwrite=False)
    except Exception:
        # mmCIF normally already supplies entity typing; residue-name fallbacks
        # in classify_residue cover structures where setup is incomplete.
        pass
    if not len(structure):
        raise ValueError(f"mask structure contains no model: {selected_path}")

    target_keys: set[ResidueKey] = set()
    if mask_policy == "all_known_nontarget":
        if target_ligand_path is None:
            raise ValueError("target_ligand_path is required for all_known_nontarget")
        target_keys = locate_target_residue_keys(
            structure, target_ligand_path, all_copies=target_all_copies
        )

    model = structure[0].clone()
    category_atoms: dict[str, int] = {}
    masked_atoms = excluded_atoms = 0
    for chain in model:
        for residue in chain:
            category = classify_residue(residue)
            key = residue_key(chain, residue)
            heavy_count = sum(
                atom.element.name.upper() not in {"H", "D", "T"}
                for atom in residue
            )
            category_atoms[category] = category_atoms.get(category, 0) + heavy_count

            if mask_policy == "feature_structure":
                should_mask = True
            elif mask_policy == "full_polymer":
                should_mask = category in POLYMER_CATEGORIES
            elif mask_policy == "polymer_plus_known_nontarget":
                should_mask = category in (
                    POLYMER_CATEGORIES | KNOWN_NONTARGET_CATEGORIES
                )
            else:
                should_mask = key not in target_keys

            for atom in residue:
                if should_mask:
                    masked_atoms += int(atom.element.name.upper() not in {"H", "D", "T"})
                else:
                    atom.occ = 0.0
                    excluded_atoms += int(atom.element.name.upper() not in {"H", "D", "T"})

    metadata: dict[str, object] = {
        "policy": mask_policy,
        "feature_structure_path": str(Path(feature_structure_path).resolve()),
        "mask_structure_path": str(selected_path.resolve()),
        "target_ligand_path": (
            str(Path(target_ligand_path).resolve()) if target_ligand_path is not None else ""
        ),
        "target_residue_keys": [list(key) for key in sorted(target_keys)],
        "target_all_copies": bool(target_all_copies),
        "n_target_residues_excluded": len(target_keys),
        "masked_heavy_atoms": masked_atoms,
        "excluded_heavy_atoms": excluded_atoms,
        "category_heavy_atoms": category_atoms,
    }
    return model, metadata


def iter_model_atoms(
    structure: gemmi.Structure,
) -> Iterable[tuple[np.ndarray, str, str, ResidueKey]]:
    """Yield heavy-atom coordinate, category, comp_id and residue key."""
    if not len(structure):
        return
    for chain in structure[0]:
        for residue in chain:
            category = classify_residue(residue)
            key = residue_key(chain, residue)
            for atom in residue:
                if atom.element.name.upper() in {"H", "D", "T"}:
                    continue
                yield (
                    np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64),
                    category,
                    residue.name.strip().upper(),
                    key,
                )
