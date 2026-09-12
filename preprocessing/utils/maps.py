#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, gzip, logging
from typing import Dict, Tuple, List, Optional
import numpy as np
import gemmi

log = logging.getLogger(__name__)

__all__ = [
    "open_map", "interp_points", "sphere_mean",
    "load_structure_atoms", "compute_ncc",
]

def open_map(path: str) -> Tuple[gemmi.Ccp4Map, gemmi.FloatGrid]:
    """Read .map or .map.gz to gemmi and return (ccp4, grid) with setup()."""
    if path.endswith(".gz"):
        # gemmi can read gz directly if file object supports .read? Safer: temp
        tmp = path[:-3]
        if not os.path.exists(tmp):
            import gzip, shutil
            with gzip.open(path, 'rb') as f_in, open(tmp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        path = tmp
    m = gemmi.read_ccp4_map(path)
    m.setup(float('nan'))  # expand & set unit cell for interpolation
    return m, m.grid

def _to_fractional(grid: gemmi.FloatGrid, xyz: np.ndarray) -> List[gemmi.Fractional]:
    uc = grid.unit_cell
    fracs = []
    for x, y, z in xyz.tolist():
        fr = uc.fractionalize(gemmi.Position(float(x), float(y), float(z)))
        fracs.append(fr)
    return fracs

def interp_points(grid: gemmi.FloatGrid, xyz: np.ndarray) -> np.ndarray:
    """Tri-linear interpolation of grid at Cartesian Å coords.
    Returns values (N,) with NaN if outside grid.
    """
    fracs = _to_fractional(grid, xyz)
    vals = []
    for fr in fracs:
        try:
            vals.append(grid.interpolate_value(fr))
        except Exception:
            vals.append(float('nan'))
    return np.asarray(vals, dtype=np.float32)

def sphere_mean(grid: gemmi.FloatGrid, center_xyz: np.ndarray, radius: float,
                directions: np.ndarray) -> float:
    """Mean grid value on a sphere of radius (Å) around center.
    directions: (D,3) unit vectors.
    Returns scalar mean (ignores NaN).
    """
    pts = center_xyz[None, :] + radius * directions
    vals = interp_points(grid, pts)
    if np.all(np.isnan(vals)):
        return float('nan')
    return float(np.nanmean(vals))

def load_structure_atoms(cif_path: str, heavy_only: bool = True, protein_only: bool = False) -> Tuple[np.ndarray, List[Dict]]:
    """Load atoms from mmCIF via gemmi. Returns (coords Å, metadata list).
    
    Args:
        cif_path: Path to mmCIF file
        heavy_only: If True, exclude hydrogen atoms
        protein_only: If True, only load standard amino acids (ATOM), exclude HETATM
    """
    st = gemmi.read_structure(cif_path)
    st.remove_alternative_conformations()
    coords, meta = [], []
    
    # Standard amino acid names
    standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    
    for model in st:
        for chain in model:
            for res in chain:
                # Check if this is a standard amino acid
                is_protein = res.name in standard_aa
                
                # If protein_only=True, skip non-protein residues
                if protein_only and not is_protein:
                    continue
                
                for atom in res:
                    ele = atom.element.name
                    if heavy_only and ele == 'H':
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    meta.append({
                        'element': ele,
                        'atom': atom.name,
                        'res': res.name,
                        'seqid': res.seqid.num,
                        'chain': chain.name,
                        'occ': atom.occ,
                        'is_protein': is_protein,  # Add protein flag
                    })
    return np.asarray(coords, dtype=np.float32), meta

def compute_ncc(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized cross-correlation between two 1D arrays; ignores NaN.
    NCC = cov(x,y) / (std(x)*std(y))
    """
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float('nan')
    xx, yy = x[mask], y[mask]
    sx, sy = np.std(xx), np.std(yy)
    if sx < 1e-6 or sy < 1e-6:
        return float('nan')
    return float(np.cov(xx, yy, bias=True)[0,1] / (sx * sy))

AA3 = {
    # 20 standard amino acids
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    # Rare/extended
    'SEC','PYL','MSE','CSO','CME','KCX','PCA','MLE','ORN',
    # Protonation/deprotonation/aliases
    'HID','HIE','HIP','HSD','HSE','HSP','ASH','GLH','LYN',
    # Ambiguous/mixed labels
    'ASX','GLX','UNK','CYX',  # CYX commonly used for disulfide-bonded CYS
    # Phosphorylation
    'SEP','TPO','PTR'
}

# --- Common nucleic acid residues (RNA 1-letter; DNA 2-letter D*; plus common modifications) ---
RNA_1 = {'A','C','G','U','I'}
DNA_2 = {'DA','DC','DG','DT','DI'}
NA_MOD = {
    'PSU','H2U','1MA','M2G','7MG','OMG','5MC','5MU','MIA','2MG','Y',  # common modifications
    'ADE','GUA','CYT','THY','URI'  # occasionally seen 3-letter representations
}

WATER = {'HOH','WAT','H2O'}
COMMON_IONS = {
    'NA','K','CL','MG','MN','CA','ZN','FE','FE2','CU','CO','NI','CD','HG','CS','SR','BR','IOD'
}

def is_protein_residue(res):
    name = res.name.strip().upper()
    if name in AA3:
        return True
    # Some PDB files label unknown amino acids as UNK
    if name == 'UNK':
        return True
    return False

def is_nucleic_residue(res):
    name = res.name.strip().upper()
    if name in NA_MOD:
        return True
    if len(name) == 1 and name in RNA_1:
        return True
    if len(name) == 2 and name in DNA_2:
        return True
    # Rare 3-letter nucleoside labels
    if name in {'ADE','GUA','CYT','THY','URI'}:
        return True
    return False

def is_water_or_ion(res):
    name = res.name.strip().upper()
    if name in WATER: return True
    if name in COMMON_IONS: return True
    # Ions often have only 1 atom; this heuristic can be enabled if needed
    # if len(res) == 1 and res.atom(0).element.is_metal():
    #     return True
    return False

def zero_occupancy_ligand(model):
    for chain in model:
        for res in chain:
            is_ligand = not (is_protein_residue(res) or is_nucleic_residue(res) or is_water_or_ion(res))
            for at in res:
                at.occ = 0.0 if is_ligand else 1.0