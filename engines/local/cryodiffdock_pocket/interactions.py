"""Deployment-safe geometric protein--ligand interaction features.

The functions in this module never read a reference ligand or RMSD.  They are
intended for retrospective validation and, only after freezing, deployment
reranking.  Hydrogen-bond features are deliberately named ``hbond_geometry``:
without explicit hydrogens they represent compatible donor/acceptor distance,
not proof of a hydrogen bond.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


AROMATIC_ATOMS = {
    "PHE": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TYR": ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"),
    "TRP": ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    "HIS": ("CG", "ND1", "CE1", "NE2", "CD2"),
}

SIDECHAIN_DONORS = {
    "ARG": ("NE", "NH1", "NH2"), "ASN": ("ND2",), "GLN": ("NE2",),
    "HIS": ("ND1", "NE2"), "LYS": ("NZ",), "SER": ("OG",),
    "THR": ("OG1",), "TRP": ("NE1",), "TYR": ("OH",), "CYS": ("SG",),
}
SIDECHAIN_ACCEPTORS = {
    "ASN": ("OD1",), "ASP": ("OD1", "OD2"), "GLN": ("OE1",),
    "GLU": ("OE1", "OE2"), "HIS": ("ND1", "NE2"), "SER": ("OG",),
    "THR": ("OG1",), "TYR": ("OH",), "CYS": ("SG",),
}
# Histidine is deliberately omitted from formal-charge sites because its
# protonation state is not reliably available in a processed heavy-atom PDB.
POSITIVE_ATOMS = {"ARG": ("CZ",), "LYS": ("NZ",)}
NEGATIVE_ATOMS = {"ASP": ("CG",), "GLU": ("CD",)}


@dataclass(frozen=True)
class ProteinInteractionSites:
    donor_xyz: np.ndarray
    acceptor_xyz: np.ndarray
    positive_xyz: np.ndarray
    negative_xyz: np.ndarray
    aromatic_centers: np.ndarray
    aromatic_normals: np.ndarray


def _array(values: list[np.ndarray]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape((-1, 3))


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    return center, vh[-1] / max(np.linalg.norm(vh[-1]), 1e-12)


def read_protein_interaction_sites(path: str | Path) -> ProteinInteractionSites:
    """Read standard-amino-acid interaction sites from the first PDB model."""
    model = next(PDBParser(QUIET=True).get_structure("protein", str(path)).get_models())
    donors: list[np.ndarray] = []
    acceptors: list[np.ndarray] = []
    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    ring_centers: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []
    for chain in model:
        for residue in chain:
            if str(residue.id[0]).strip():
                continue
            name = str(residue.resname).upper()
            atoms = {
                str(atom.name).strip(): np.asarray(atom.coord, dtype=np.float64)
                for atom in residue
                if str(atom.element or atom.name[:1]).upper() not in {"H", "D"}
            }
            # Backbone amide nitrogen is a donor except in proline; carbonyl
            # oxygen is an acceptor.  Terminal-state refinements are omitted.
            if name != "PRO" and "N" in atoms:
                donors.append(atoms["N"])
            for atom_name in ("O", "OXT"):
                if atom_name in atoms:
                    acceptors.append(atoms[atom_name])
            for atom_name in SIDECHAIN_DONORS.get(name, ()):
                if atom_name in atoms:
                    donors.append(atoms[atom_name])
            for atom_name in SIDECHAIN_ACCEPTORS.get(name, ()):
                if atom_name in atoms:
                    acceptors.append(atoms[atom_name])
            for atom_name in POSITIVE_ATOMS.get(name, ()):
                if atom_name in atoms:
                    positive.append(atoms[atom_name])
            for atom_name in NEGATIVE_ATOMS.get(name, ()):
                if atom_name in atoms:
                    negative.append(atoms[atom_name])
            ring_names = AROMATIC_ATOMS.get(name)
            if ring_names and all(atom_name in atoms for atom_name in ring_names):
                center, normal = _plane(np.asarray([atoms[a] for a in ring_names]))
                ring_centers.append(center)
                ring_normals.append(normal)
    if not donors or not acceptors:
        raise ValueError(f"No protein donor/acceptor sites parsed from {path}")
    return ProteinInteractionSites(
        donor_xyz=_array(donors), acceptor_xyz=_array(acceptors),
        positive_xyz=_array(positive), negative_xyz=_array(negative),
        aromatic_centers=_array(ring_centers), aromatic_normals=_array(ring_normals),
    )


def _rdkit_feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    return ChemicalFeatures.BuildFeatureFactory(
        str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
    )


_FEATURE_FACTORY = None


def _ligand_features(path: str | Path) -> dict[str, object]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=True)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not sanitize ligand pose: {path}")
    global _FEATURE_FACTORY
    if _FEATURE_FACTORY is None:
        _FEATURE_FACTORY = _rdkit_feature_factory()
    conformer = molecule.GetConformer()
    features = _FEATURE_FACTORY.GetFeaturesForMol(molecule)

    result: dict[str, list[np.ndarray]] = {
        "Donor": [], "Acceptor": [], "PosIonizable": [], "NegIonizable": [],
    }
    aromatic_centers: list[np.ndarray] = []
    aromatic_normals: list[np.ndarray] = []
    for feature in features:
        family = feature.GetFamily()
        atom_ids = tuple(feature.GetAtomIds())
        points = np.asarray(
            [list(conformer.GetAtomPosition(int(atom_id))) for atom_id in atom_ids],
            dtype=np.float64,
        )
        if family == "Aromatic" and len(points) >= 3:
            center, normal = _plane(points)
            aromatic_centers.append(center)
            aromatic_normals.append(normal)
        elif family in result:
            result[family].append(points.mean(axis=0))
    return {
        "donor_xyz": _array(result["Donor"]),
        "acceptor_xyz": _array(result["Acceptor"]),
        "positive_xyz": _array(result["PosIonizable"]),
        "negative_xyz": _array(result["NegIonizable"]),
        "aromatic_centers": _array(aromatic_centers),
        "aromatic_normals": _array(aromatic_normals),
        "num_heavy_atoms": int(molecule.GetNumHeavyAtoms()),
    }


def _distance_quality(
    left: np.ndarray,
    right: np.ndarray,
    *,
    ideal_A: float,
    width_A: float,
    cutoff_A: float,
) -> tuple[float, int, float]:
    if len(left) == 0 or len(right) == 0:
        return 0.0, 0, float("nan")
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    best = distances.min(axis=1)
    quality = np.exp(-0.5 * ((best - ideal_A) / width_A) ** 2)
    quality[best > cutoff_A] = 0.0
    return float(quality.mean()), int((best <= cutoff_A).sum()), float(best.min())


def aromatic_geometry_score(
    ligand_centers: np.ndarray,
    ligand_normals: np.ndarray,
    protein_centers: np.ndarray,
    protein_normals: np.ndarray,
) -> dict[str, float | int]:
    """Return soft parallel/T-shaped aromatic-contact geometry quality."""
    if len(ligand_centers) == 0 or len(protein_centers) == 0:
        return {
            "aromatic_geometry_quality": 0.0, "aromatic_contact_count": 0,
            "best_aromatic_centroid_distance_A": float("nan"),
            "best_aromatic_plane_angle_deg": float("nan"),
        }
    distances = np.linalg.norm(
        ligand_centers[:, None, :] - protein_centers[None, :, :], axis=-1
    )
    cosine = np.abs(ligand_normals @ protein_normals.T).clip(0.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    parallel = np.exp(-0.5 * ((distances - 4.8) / 0.9) ** 2) * np.exp(
        -0.5 * (angles / 25.0) ** 2
    )
    tshape = np.exp(-0.5 * ((distances - 5.2) / 1.0) ** 2) * np.exp(
        -0.5 * ((angles - 90.0) / 25.0) ** 2
    )
    quality = np.maximum(parallel, tshape)
    quality[(distances < 3.0) | (distances > 7.0)] = 0.0
    best_index = np.unravel_index(int(np.argmax(quality)), quality.shape)
    best_per_ligand = quality.max(axis=1)
    return {
        "aromatic_geometry_quality": float(best_per_ligand.mean()),
        "aromatic_contact_count": int((best_per_ligand >= 0.25).sum()),
        "best_aromatic_centroid_distance_A": float(distances[best_index]),
        "best_aromatic_plane_angle_deg": float(angles[best_index]),
    }


def score_pose_interactions(
    protein: ProteinInteractionSites,
    ligand_path: str | Path,
) -> dict[str, object]:
    """Compute normalized, low-assumption interaction geometry features."""
    ligand = _ligand_features(ligand_path)
    donor_quality, donor_count, donor_min = _distance_quality(
        ligand["donor_xyz"], protein.acceptor_xyz,
        ideal_A=2.9, width_A=0.45, cutoff_A=3.6,
    )
    acceptor_quality, acceptor_count, acceptor_min = _distance_quality(
        ligand["acceptor_xyz"], protein.donor_xyz,
        ideal_A=2.9, width_A=0.45, cutoff_A=3.6,
    )
    polar_sites = len(ligand["donor_xyz"]) + len(ligand["acceptor_xyz"])
    hbond_quality = (
        donor_quality * len(ligand["donor_xyz"])
        + acceptor_quality * len(ligand["acceptor_xyz"])
    ) / max(polar_sites, 1)
    salt_pos, salt_pos_count, salt_pos_min = _distance_quality(
        ligand["positive_xyz"], protein.negative_xyz,
        ideal_A=3.2, width_A=0.7, cutoff_A=4.5,
    )
    salt_neg, salt_neg_count, salt_neg_min = _distance_quality(
        ligand["negative_xyz"], protein.positive_xyz,
        ideal_A=3.2, width_A=0.7, cutoff_A=4.5,
    )
    charged_sites = len(ligand["positive_xyz"]) + len(ligand["negative_xyz"])
    salt_quality = (
        salt_pos * len(ligand["positive_xyz"])
        + salt_neg * len(ligand["negative_xyz"])
    ) / max(charged_sites, 1)
    aromatic = aromatic_geometry_score(
        ligand["aromatic_centers"], ligand["aromatic_normals"],
        protein.aromatic_centers, protein.aromatic_normals,
    )
    interaction_quality = (
        0.50 * hbond_quality
        + 0.35 * float(aromatic["aromatic_geometry_quality"])
        + 0.15 * salt_quality
    )
    return {
        "num_ligand_heavy_atoms": ligand["num_heavy_atoms"],
        "num_ligand_donor_sites": len(ligand["donor_xyz"]),
        "num_ligand_acceptor_sites": len(ligand["acceptor_xyz"]),
        "num_ligand_aromatic_rings": len(ligand["aromatic_centers"]),
        "num_ligand_charged_sites": charged_sites,
        "hbond_geometry_quality": float(hbond_quality),
        "hbond_geometry_count": int(donor_count + acceptor_count),
        "best_hbond_distance_A": float(np.nanmin([donor_min, acceptor_min]))
        if np.isfinite([donor_min, acceptor_min]).any() else float("nan"),
        **aromatic,
        "salt_bridge_geometry_quality": float(salt_quality),
        "salt_bridge_geometry_count": int(salt_pos_count + salt_neg_count),
        "best_salt_bridge_distance_A": float(np.nanmin([salt_pos_min, salt_neg_min]))
        if np.isfinite([salt_pos_min, salt_neg_min]).any() else float("nan"),
        "interaction_geometry_quality": float(interaction_quality),
    }
