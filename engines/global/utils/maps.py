#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import gemmi

__all__ = [
    "open_map",
    "interp_points",
    "sphere_mean",
    "load_structure_atoms",
    "compute_ncc",
]


def _import_gemmi():
    import gemmi

    return gemmi


def open_map(path: str) -> Tuple["gemmi.Ccp4Map", "gemmi.FloatGrid"]:
    """Read a CCP4/MRC map and return (ccp4, grid) with setup()."""
    gemmi = _import_gemmi()
    if path.endswith(".gz"):
        tmp = path[:-3]
        if not os.path.exists(tmp):
            import shutil

            with open(tmp, "wb") as f_out:
                import gzip

                with gzip.open(path, "rb") as f_in:
                    shutil.copyfileobj(f_in, f_out)
        path = tmp
    m = gemmi.read_ccp4_map(path)
    m.setup(float("nan"))
    return m, m.grid


def _to_fractional(grid: "gemmi.FloatGrid", xyz: np.ndarray) -> List["gemmi.Fractional"]:
    gemmi = _import_gemmi()
    uc = grid.unit_cell
    fracs = []
    for x, y, z in xyz.tolist():
        fr = uc.fractionalize(gemmi.Position(float(x), float(y), float(z)))
        fracs.append(fr)
    return fracs


def interp_points(grid: "gemmi.FloatGrid", xyz: np.ndarray) -> np.ndarray:
    """Tri-linear interpolation of a grid at Cartesian coordinates."""
    fracs = _to_fractional(grid, xyz)
    vals = []
    for fr in fracs:
        try:
            vals.append(grid.interpolate_value(fr))
        except Exception:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=np.float32)


def sphere_mean(
    grid: "gemmi.FloatGrid", center_xyz: np.ndarray, radius: float, directions: np.ndarray
) -> float:
    """Mean grid value on a sphere of radius (A) around center."""
    pts = center_xyz[None, :] + radius * directions
    vals = interp_points(grid, pts)
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def load_structure_atoms(
    cif_path: str, heavy_only: bool = True, protein_only: bool = False
) -> Tuple[np.ndarray, List[Dict]]:
    """Load atoms from a structure file via gemmi."""
    gemmi = _import_gemmi()
    st = gemmi.read_structure(cif_path)
    st.remove_alternative_conformations()
    coords, meta = [], []

    standard_aa = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }

    for model in st:
        for chain in model:
            for res in chain:
                is_protein = res.name in standard_aa
                if protein_only and not is_protein:
                    continue
                for atom in res:
                    ele = atom.element.name
                    if heavy_only and ele == "H":
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    meta.append(
                        {
                            "element": ele,
                            "atom": atom.name,
                            "res": res.name,
                            "seqid": res.seqid.num,
                            "chain": chain.name,
                            "occ": atom.occ,
                            "is_protein": is_protein,
                        }
                    )
    return np.asarray(coords, dtype=np.float32), meta


def compute_ncc(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized cross-correlation between two 1D arrays; ignores NaN."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return float("nan")
    xx, yy = x[mask], y[mask]
    sx, sy = np.std(xx), np.std(yy)
    if sx < 1e-6 or sy < 1e-6:
        return float("nan")
    return float(np.cov(xx, yy, bias=True)[0, 1] / (sx * sy))
