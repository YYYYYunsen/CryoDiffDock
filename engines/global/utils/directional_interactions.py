"""Differentiable directional protein--ligand interaction features.

The terms in this module are deliberately lightweight.  They are intended for
pose discrimination and local refinement, not as binding free energies.  The
same implementation is used by the offline scoring script and by
``density_gradient_refine.py`` so a feature that helps reranking can be tested
as a coordinate gradient without changing its definition.

Hydrogen-bond directions are inferred from heavy-atom geometry.  Ambiguous
protonation/tautomer states therefore remain an important limitation; the
scores should be treated as soft evidence and combined with a density anchor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import gemmi
import numpy as np
import torch


_PROTEIN_DONORS = {
    ("*", "N"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"), ("GLN", "NE2"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"),
    ("TYR", "OH"), ("CYS", "SG"),
}
_PROTEIN_ACCEPTORS = {
    ("*", "O"), ("*", "OXT"),
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"), ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"), ("TYR", "OH"), ("CYS", "SG"),
}
_PROTEIN_POSITIVE = {
    ("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
}
_PROTEIN_NEGATIVE = {
    ("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
}
_METALS = {"ZN", "MG", "CA", "MN", "FE", "CO", "NI", "CU"}
_METAL_IDEAL_DISTANCE = {
    "ZN": 2.20, "MG": 2.10, "CA": 2.40, "MN": 2.20,
    "FE": 2.20, "CO": 2.15, "NI": 2.10, "CU": 2.10,
}


def _empty_np() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


@dataclass(frozen=True)
class ProteinInteractionSites:
    donor_pos: np.ndarray
    donor_dir: np.ndarray
    acceptor_pos: np.ndarray
    acceptor_dir: np.ndarray
    positive_pos: np.ndarray
    negative_pos: np.ndarray
    metal_pos: np.ndarray
    metal_ideal_distance: np.ndarray

    def subset_near(self, ligand_coords: np.ndarray, radius: float = 12.0) -> "ProteinInteractionSites":
        ligand_coords = np.asarray(ligand_coords, dtype=np.float32)

        def keep(pos: np.ndarray) -> np.ndarray:
            if len(pos) == 0 or len(ligand_coords) == 0:
                return np.zeros(len(pos), dtype=bool)
            d2 = ((pos[:, None, :] - ligand_coords[None, :, :]) ** 2).sum(axis=-1)
            return d2.min(axis=1) <= float(radius) ** 2

        kd, ka = keep(self.donor_pos), keep(self.acceptor_pos)
        kp, kn, km = keep(self.positive_pos), keep(self.negative_pos), keep(self.metal_pos)
        return ProteinInteractionSites(
            self.donor_pos[kd], self.donor_dir[kd],
            self.acceptor_pos[ka], self.acceptor_dir[ka],
            self.positive_pos[kp], self.negative_pos[kn],
            self.metal_pos[km], self.metal_ideal_distance[km],
        )


@dataclass(frozen=True)
class LigandInteractionSites:
    donor_indices: tuple[int, ...]
    acceptor_indices: tuple[int, ...]
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    neighbor_indices: tuple[tuple[int, ...], ...]


def _as_xyz(rows: Iterable[np.ndarray]) -> np.ndarray:
    rows = list(rows)
    return np.asarray(rows, dtype=np.float32).reshape((-1, 3)) if rows else _empty_np()


def _site_key(resname: str, atom_name: str, table: set[tuple[str, str]]) -> bool:
    return (resname, atom_name) in table or ("*", atom_name) in table


def _infer_outward_direction(atom_idx: int, elements: list[str], coords: np.ndarray) -> np.ndarray:
    """Approximate an H/lone-pair direction as pointing away from bonded atoms."""
    center = coords[atom_idx]
    neighbors = []
    # Protein PDB files do not always contain reliable CONECT records.  Within
    # one residue, 1.15--1.95 A is a useful heavy-atom covalent-distance proxy.
    for j, element in enumerate(elements):
        if j == atom_idx or element in _METALS:
            continue
        d = float(np.linalg.norm(coords[j] - center))
        if 1.10 <= d <= 1.95:
            neighbors.append(coords[j])
    if not neighbors:
        return np.zeros(3, dtype=np.float32)
    vec = center - np.mean(neighbors, axis=0)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-6 else np.zeros(3, dtype=np.float32)


@lru_cache(maxsize=256)
def parse_protein_interaction_sites(pdb_path: str) -> ProteinInteractionSites:
    """Parse standard-residue interaction sites from a PDB/mmCIF receptor."""
    st = gemmi.read_structure(os.fspath(pdb_path))
    st.remove_alternative_conformations()
    donors, donor_dirs, acceptors, acceptor_dirs = [], [], [], []
    positives, negatives, metals, metal_distances = [], [], [], []

    for model in st:
        for chain in model:
            for residue in chain:
                resname = residue.name.strip().upper()
                atoms = [a for a in residue if a.element.name != "H"]
                if not atoms:
                    continue
                coords = np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in atoms], dtype=np.float32)
                elements = [a.element.name.upper() for a in atoms]
                for i, atom in enumerate(atoms):
                    atom_name = atom.name.strip().upper()
                    element = elements[i]
                    pos = coords[i]
                    if element in _METALS:
                        metals.append(pos)
                        metal_distances.append(_METAL_IDEAL_DISTANCE[element])
                        continue
                    if _site_key(resname, atom_name, _PROTEIN_DONORS) and not (
                        resname == "PRO" and atom_name == "N"
                    ):
                        donors.append(pos)
                        donor_dirs.append(_infer_outward_direction(i, elements, coords))
                    if _site_key(resname, atom_name, _PROTEIN_ACCEPTORS):
                        acceptors.append(pos)
                        acceptor_dirs.append(_infer_outward_direction(i, elements, coords))
                    if (resname, atom_name) in _PROTEIN_POSITIVE:
                        positives.append(pos)
                    if (resname, atom_name) in _PROTEIN_NEGATIVE:
                        negatives.append(pos)
        break  # docking uses the first structural model

    return ProteinInteractionSites(
        _as_xyz(donors), _as_xyz(donor_dirs),
        _as_xyz(acceptors), _as_xyz(acceptor_dirs),
        _as_xyz(positives), _as_xyz(negatives), _as_xyz(metals),
        np.asarray(metal_distances, dtype=np.float32),
    )


@lru_cache(maxsize=1)
def _feature_factory():
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    return ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))


def _fallback_ligand_polar_atoms(mol) -> tuple[set[int], set[int]]:
    donors, acceptors = set(), set()
    for atom in mol.GetAtoms():
        idx, z = atom.GetIdx(), atom.GetAtomicNum()
        if z not in (7, 8, 16):
            continue
        charge = atom.GetFormalCharge()
        try:
            n_h = atom.GetTotalNumHs(includeNeighbors=True)
        except Exception:
            n_h = 0
        if n_h > 0 or (z == 7 and charge > 0):
            donors.add(idx)
        if charge <= 0:
            # Exclude amide-like nitrogens from the permissive fallback.
            amide_n = z == 7 and any(
                nb.GetAtomicNum() == 6 and any(
                    b.GetBondTypeAsDouble() >= 1.9 and b.GetOtherAtom(nb).GetAtomicNum() == 8
                    for b in nb.GetBonds()
                )
                for nb in atom.GetNeighbors()
            )
            if not amide_n:
                acceptors.add(idx)
    return donors, acceptors


def build_ligand_interaction_sites(mol) -> LigandInteractionSites:
    """Type ligand atoms once; directions are recomputed from current coordinates."""
    donors, acceptors = set(), set()
    try:
        from rdkit import Chem
        typed_mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(typed_mol)
        except Exception:
            typed_mol = mol
        typed = _feature_factory().GetFeaturesForMol(typed_mol)
        for feature in typed:
            family = feature.GetFamily()
            ids = feature.GetAtomIds()
            if family == "Donor":
                donors.update(int(i) for i in ids)
            elif family == "Acceptor":
                acceptors.update(int(i) for i in ids)
    except Exception:
        donors, acceptors = _fallback_ligand_polar_atoms(mol)

    positive, negative = set(), set()
    neighbors = []
    for atom in mol.GetAtoms():
        idx, charge = atom.GetIdx(), atom.GetFormalCharge()
        if charge > 0:
            positive.add(idx)
        elif charge < 0:
            negative.add(idx)
        neighbors.append(tuple(int(nb.GetIdx()) for nb in atom.GetNeighbors()))

    return LigandInteractionSites(
        tuple(sorted(donors)), tuple(sorted(acceptors)),
        tuple(sorted(positive)), tuple(sorted(negative)), tuple(neighbors),
    )


def _tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(array):
        return array.to(dtype=like.dtype, device=like.device)
    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def protein_sites_to_device(sites: ProteinInteractionSites, like: torch.Tensor) -> ProteinInteractionSites:
    """Copy static protein fields once before an iterative GPU refinement."""
    return ProteinInteractionSites(*(
        _tensor(getattr(sites, field), like)
        for field in (
            "donor_pos", "donor_dir", "acceptor_pos", "acceptor_dir",
            "positive_pos", "negative_pos", "metal_pos", "metal_ideal_distance",
        )
    ))


def _ligand_directions(coords: torch.Tensor, indices: tuple[int, ...],
                       neighbors: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    dirs = []
    for idx in indices:
        nbs = neighbors[idx]
        if nbs:
            vec = coords[idx] - coords[list(nbs)].mean(dim=0)
            dirs.append(vec / vec.norm().clamp_min(1e-6))
        else:
            dirs.append(torch.zeros(3, dtype=coords.dtype, device=coords.device))
    return torch.stack(dirs) if dirs else coords.new_zeros((0, 3))


def _angular_gate(cosine: torch.Tensor, minimum_cosine: float = 0.35,
                  softness: float = 0.12) -> torch.Tensor:
    return torch.sigmoid((cosine - float(minimum_cosine)) / float(softness))


def _hbond_score(lig_pos: torch.Tensor, lig_dir: torch.Tensor,
                 prot_pos: torch.Tensor, prot_dir: torch.Tensor,
                 ideal_distance: float = 2.9, distance_sigma: float = 0.45,
                 average_over: str = "ligand") -> torch.Tensor:
    if lig_pos.shape[0] == 0 or prot_pos.shape[0] == 0:
        return lig_pos.new_zeros(())
    delta = prot_pos[None, :, :] - lig_pos[:, None, :]
    dist = torch.linalg.norm(delta, dim=-1).clamp_min(1e-6)
    unit_lp = delta / dist[..., None]
    distance_term = torch.exp(-0.5 * ((dist - float(ideal_distance)) / float(distance_sigma)) ** 2)
    lig_cos = (lig_dir[:, None, :] * unit_lp).sum(dim=-1)
    # prot_dir points away from its bonded neighbors, so a compatible ligand
    # approaches along the opposite pair vector (protein -> ligand).
    prot_cos = (prot_dir[None, :, :] * -unit_lp).sum(dim=-1)
    lig_known = torch.linalg.norm(lig_dir, dim=-1)[:, None] > 0.5
    prot_known = torch.linalg.norm(prot_dir, dim=-1)[None, :] > 0.5
    lig_gate = torch.where(lig_known, _angular_gate(lig_cos), torch.ones_like(lig_cos))
    prot_gate = torch.where(prot_known, _angular_gate(prot_cos), torch.ones_like(prot_cos))
    pair_score = distance_term * lig_gate * prot_gate
    if average_over == "ligand":
        return pair_score.max(dim=1).values.mean()
    if average_over == "protein":
        return pair_score.max(dim=0).values.mean()
    raise ValueError(f"unknown average_over={average_over!r}")


def _distance_complement_score(lig_pos: torch.Tensor, prot_pos: torch.Tensor,
                               ideal: float, sigma: float) -> torch.Tensor:
    if lig_pos.shape[0] == 0 or prot_pos.shape[0] == 0:
        return lig_pos.new_zeros(())
    dist = torch.cdist(lig_pos, prot_pos)
    score = torch.exp(-0.5 * ((dist - float(ideal)) / float(sigma)) ** 2)
    return score.max(dim=1).values.mean()


def _mean_available(terms: list[tuple[torch.Tensor, bool]], like: torch.Tensor) -> torch.Tensor:
    available = [value for value, is_available in terms if is_available]
    return torch.stack(available).mean() if available else like.new_zeros(())


def directional_interaction_scores(
    ligand_coords: torch.Tensor,
    ligand_sites: LigandInteractionSites,
    protein_sites: ProteinInteractionSites,
) -> dict[str, torch.Tensor]:
    """Return differentiable, approximately [0, 1] pose compatibility terms."""
    c = ligand_coords
    donor_idx = list(ligand_sites.donor_indices)
    acceptor_idx = list(ligand_sites.acceptor_indices)
    positive_idx = list(ligand_sites.positive_indices)
    negative_idx = list(ligand_sites.negative_indices)

    lig_donor_pos = c[donor_idx] if donor_idx else c.new_zeros((0, 3))
    lig_acceptor_pos = c[acceptor_idx] if acceptor_idx else c.new_zeros((0, 3))
    lig_donor_dir = _ligand_directions(c, ligand_sites.donor_indices, ligand_sites.neighbor_indices)
    lig_acceptor_dir = _ligand_directions(c, ligand_sites.acceptor_indices, ligand_sites.neighbor_indices)

    # ligand donor -> protein acceptor
    hbond_donor = _hbond_score(
        lig_donor_pos, lig_donor_dir,
        _tensor(protein_sites.acceptor_pos, c), _tensor(protein_sites.acceptor_dir, c),
    )
    # protein donor -> ligand acceptor.  _hbond_score expects its first argument
    # to be the donor, so swap the roles and keep ligand coordinates in the
    # differentiable protein-position slot.
    hbond_acceptor = _hbond_score(
        _tensor(protein_sites.donor_pos, c), _tensor(protein_sites.donor_dir, c),
        lig_acceptor_pos, lig_acceptor_dir,
        average_over="protein",
    )
    hbond = _mean_available([
        (hbond_donor, bool(donor_idx) and len(protein_sites.acceptor_pos) > 0),
        (hbond_acceptor, bool(acceptor_idx) and len(protein_sites.donor_pos) > 0),
    ], c)

    salt_positive = _distance_complement_score(
        c[positive_idx] if positive_idx else c.new_zeros((0, 3)),
        _tensor(protein_sites.negative_pos, c), 3.2, 0.65,
    )
    salt_negative = _distance_complement_score(
        c[negative_idx] if negative_idx else c.new_zeros((0, 3)),
        _tensor(protein_sites.positive_pos, c), 3.2, 0.65,
    )
    salt_bridge = _mean_available([
        (salt_positive, bool(positive_idx) and len(protein_sites.negative_pos) > 0),
        (salt_negative, bool(negative_idx) and len(protein_sites.positive_pos) > 0),
    ], c)

    metal_pos = _tensor(protein_sites.metal_pos, c)
    if lig_acceptor_pos.shape[0] and metal_pos.shape[0]:
        dist = torch.cdist(lig_acceptor_pos, metal_pos)
        ideal = _tensor(protein_sites.metal_ideal_distance, c)[None, :]
        metal_pair = torch.exp(-0.5 * ((dist - ideal) / 0.30) ** 2)
        # Each receptor metal needs at least one compatible ligand acceptor;
        # do not dilute a correct chelating atom by all non-coordinating polar atoms.
        metal_coordination = metal_pair.max(dim=0).values.mean()
    else:
        metal_coordination = c.new_zeros(())

    return {
        "hbond": hbond,
        "hbond_ligand_donor": hbond_donor,
        "hbond_ligand_acceptor": hbond_acceptor,
        "salt_bridge": salt_bridge,
        "metal_coordination": metal_coordination,
    }


def score_pose_numpy(mol, coords: np.ndarray, protein_sites: ProteinInteractionSites) -> dict[str, float]:
    """Convenience wrapper for non-gradient scoring and CSV export."""
    ligand_sites = build_ligand_interaction_sites(mol)
    c = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
    with torch.no_grad():
        scores = directional_interaction_scores(c, ligand_sites, protein_sites)
    return {name: float(value) for name, value in scores.items()}
