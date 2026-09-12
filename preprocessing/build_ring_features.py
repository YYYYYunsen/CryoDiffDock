#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute per-atom ring features for a single structure-map pair.

For the provided structure (.cif/.pdb) and EM map, this script:
- masks protein regions in the map (by setting their grid values to a constant),
- samples points on spheres around each atom at given radii,
- computes a simple statistic of the sampled values (max),
- and writes one feature file (.npz) plus a small .json sidecar.
"""
from __future__ import annotations
import os, sys, json, argparse, logging, pathlib
from typing import List, Dict, Optional, Tuple
import numpy as np
import gemmi

from utils.maps import open_map, load_structure_atoms, interp_points
from utils.masking import POLYMER_CATEGORIES, classify_residue, prepare_mask_model

# Policies whose mask includes modeled non-polymer atoms, which may warrant a
# larger mask radius than the one used for polymers.
NONPOLYMER_MASKING_POLICIES = frozenset(
    {'polymer_plus_known_nontarget', 'all_known_nontarget'}
)
try:
    from scipy.spatial import cKDTree as _KDTree
except Exception:
    _KDTree = None
from utils.geom import unit_vectors_on_sphere

log = logging.getLogger(__name__)

PERIODIC_TABLE = {
    'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'CL':17,'K':19,'CA':20,
    'MG':12,'ZN':30,'FE':26,'MN':25,'CU':29,'NA':11,'NI':28,'CO':27,
    'BR':35,'I':53,'SE':34,'B':5
}

def element_to_Z(sym: str) -> int:
    s = sym.strip().upper()
    return int(PERIODIC_TABLE.get(s, 0))


def build_kdtree(atom_xyz: np.ndarray):
    return _KDTree(atom_xyz) if (_KDTree is not None and len(atom_xyz) > 0) else None


def preprocess_protein_mask(grid, model, ignore_hydrogen=True, ignore_zero_occupancy_atoms=True, radius=2.5, mask_value=None):
    """Mask protein regions in the EM map by setting grid points near protein atoms to a constant value."""
    if mask_value is None:
        mask_value = grid.sum()/grid.point_count
    grid.mask_points_in_constant_radius(model=model, radius=radius, value=mask_value,
                                        ignore_hydrogen=ignore_hydrogen, ignore_zero_occupancy_atoms=ignore_zero_occupancy_atoms)
    return grid


def ring_max_at_point(grid, center_xyz: np.ndarray, radius: float, directions: np.ndarray) -> float:
    """Compute max interpolated EM density on a sphere around center_xyz at given radius.

    After protein masking, protein regions contribute the mask value (mean density),
    while ligand/solvent regions keep their original density.
    """
    pts = center_xyz[None, :] + float(radius) * directions
    emv = interp_points(grid, pts)
    return float(np.max(emv))


def _normalize_percentile(
    rings: np.ndarray,
    ring_vectors: np.ndarray,
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Per-channel percentile normalization for rings and ring_vectors.

    rings has shape (N_atoms, n_radii * 4) where the last axis is
    [max_r0, mean_r0, std_r0, max_r1, mean_r1, std_r1, ..., l2norm_r0, ..., l2norm_rN].
    Each channel is independently clipped to [low_pct, high_pct] and scaled to [0, 1].

    ring_vectors has shape (N_atoms, n_radii, 3). Each radius channel is scaled
    by the same factor as the corresponding max channel (index i*3) so that
    the vector magnitudes stay consistent with the scalar features.

    Args:
        rings: (N_atoms, n_radii * 4) raw density statistics
        ring_vectors: (N_atoms, n_radii, 3) density-weighted direction vectors
        low_pct: Lower percentile for clipping (default 1)
        high_pct: Upper percentile for clipping (default 99)

    Returns:
        Tuple of (normalized_rings, normalized_ring_vectors, stats_dict)
    """
    if not (0.0 <= low_pct < high_pct <= 100.0):
        raise ValueError(
            f"Invalid percentile range: low_pct={low_pct}, high_pct={high_pct}. "
            "Expected 0 <= low_pct < high_pct <= 100."
        )

    rings_out = rings.copy().astype(np.float32)
    vecs_out = ring_vectors.copy().astype(np.float32)
    n_channels = rings.shape[1]
    n_radii = ring_vectors.shape[1]
    lows, highs = [], []

    for i in range(n_channels):
        col = rings[:, i]
        low = float(np.percentile(col, low_pct))
        high = float(np.percentile(col, high_pct))
        scale = high - low if high - low > 1e-8 else 1.0
        rings_out[:, i] = np.clip((col - low) / scale, 0.0, 1.0)
        lows.append(low)
        highs.append(high)

    # Scale ring_vectors by the same factor as the corresponding max channel (i*3)
    for ri in range(n_radii):
        max_channel_idx = ri * 3  # max is the first stat for each radius
        scale = highs[max_channel_idx] - lows[max_channel_idx]
        if scale > 1e-8:
            vecs_out[:, ri, :] = ring_vectors[:, ri, :] / scale

    stats = {'method': 'percentile', 'lows': lows, 'highs': highs,
             'low_pct': low_pct, 'high_pct': high_pct}
    return rings_out, vecs_out, stats


def _normalize_upper_percentile(
    rings: np.ndarray,
    ring_vectors: np.ndarray,
    high_pct: float = 99.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Upper-only percentile normalization for rings and ring_vectors.

    This mode preserves weak signals by not lifting the lower tail:
      - rings: clip only values above the high percentile, then scale by that high value
      - ring_vectors: scale each radius vector channel by the corresponding max-channel high value
    """
    if not (0.0 < high_pct <= 100.0):
        raise ValueError(
            f"Invalid upper percentile: high_pct={high_pct}. Expected 0 < high_pct <= 100."
        )

    rings_out = rings.copy().astype(np.float32)
    vecs_out = ring_vectors.copy().astype(np.float32)
    n_channels = rings.shape[1]
    n_radii = ring_vectors.shape[1]
    highs = []
    scales = []
    min_scale = 1e-3

    for i in range(n_channels):
        col = rings[:, i]
        high = float(np.percentile(col, high_pct))
        # Use a strictly positive robust scale to avoid exploding values when
        # high is near zero or negative (which can happen for signed channels).
        abs_high = float(np.percentile(np.abs(col), high_pct))
        scale = max(abs_high, min_scale)
        rings_out[:, i] = np.minimum(col, high) / scale
        highs.append(high)
        scales.append(scale)

    # Scale ring_vectors by the corresponding max channel (i*3)
    for ri in range(n_radii):
        max_channel_idx = ri * 3
        scale = scales[max_channel_idx]
        vecs_out[:, ri, :] = ring_vectors[:, ri, :] / scale

    stats = {
        'method': 'upper_percentile',
        'highs': highs,
        'scales': scales,
        'high_pct': high_pct,
        'min_scale': min_scale,
    }
    return rings_out, vecs_out, stats


def _spherical_harmonics_l2(dirs: np.ndarray) -> np.ndarray:
    """Compute real l=2 spherical harmonics for unit direction vectors.

    Uses Racah normalization consistent with e3nn 'component' normalization.
    For unit vector d = (x, y, z), returns 5-component tensor:
        [sqrt(3)*xy, sqrt(3)*yz, (2z²-x²-y²)/2, sqrt(3)*xz, sqrt(3)/2*(x²-y²)]

    Args:
        dirs: (N, 3) unit direction vectors

    Returns:
        (N, 5) real l=2 spherical harmonics
    """
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    sqrt3 = np.sqrt(3.0)
    return np.stack([
        sqrt3 * x * y,                      # m=-2
        sqrt3 * y * z,                      # m=-1
        0.5 * (2.0 * z**2 - x**2 - y**2),  # m=0
        sqrt3 * x * z,                      # m=1
        0.5 * sqrt3 * (x**2 - y**2),        # m=2
    ], axis=1).astype(np.float32)


def _compute_density_center_densest_blob(
    xyz: np.ndarray,
    score: np.ndarray,
    top_pct: float = 95.0,
    blob_radius: float = 15.0,
) -> Optional[np.ndarray]:
    """Estimate a single density center by locking onto the densest local blob.

    Steps:
      1) Keep high-density candidates (top percentile by score)
      2) Find the peak candidate (highest score)
      3) Keep only candidates within blob_radius of that peak (single blob)
      4) Compute weighted centroid inside this blob
    """
    if xyz.shape[0] == 0:
        return None

    threshold = np.percentile(score, top_pct)
    high_mask = score >= threshold
    if not high_mask.any():
        return None

    sub_xyz = xyz[high_mask]
    sub_score = score[high_mask]

    peak_idx = int(np.argmax(sub_score))
    peak_xyz = sub_xyz[peak_idx]

    dists = np.linalg.norm(sub_xyz - peak_xyz[None, :], axis=1)
    blob_mask = dists <= float(blob_radius)

    if not blob_mask.any():
        return peak_xyz.astype(np.float32)

    blob_xyz = sub_xyz[blob_mask]
    blob_score = sub_score[blob_mask]

    # Soft weighting relative to threshold to favor stronger local points.
    w = np.maximum(blob_score - threshold + 1e-3, 0.0) ** 2
    if np.all(w <= 1e-12):
        return blob_xyz.mean(axis=0).astype(np.float32)
    w = w / (w.sum() + 1e-8)
    center = np.sum(blob_xyz * w[:, None], axis=0)
    return center.astype(np.float32)


def _compute_secondary_density_center(
    xyz: np.ndarray,
    score: np.ndarray,
    primary_center: Optional[np.ndarray],
    top_pct: float = 95.0,
    blob_radius: float = 15.0,
    exclusion_ratio: float = 0.8,
) -> Optional[np.ndarray]:
    """Estimate a secondary density center from a non-overlapping peak/blob.

    This avoids the previous "mask to global min" behavior, which can be unstable.
    We first remove atoms around the primary center, then run the same densest-blob
    logic on the remaining atoms to get a true second peak candidate.
    """
    if primary_center is None or xyz.shape[0] == 0:
        return None

    exclusion_r = float(blob_radius) * float(exclusion_ratio)
    d = np.linalg.norm(xyz - primary_center[None, :], axis=1)
    remain = d > exclusion_r
    if int(np.sum(remain)) < 20:
        return None

    xyz2 = xyz[remain]
    score2 = score[remain]
    return _compute_density_center_densest_blob(
        xyz=xyz2,
        score=score2,
        top_pct=top_pct,
        blob_radius=blob_radius,
    )


def compute_ring_features(
    structure_path: str,
    map_path: str,
    radii: List[float] = [2.5, 3.5, 4.5, 5.5, 6.5, 8.0],
    n_dir: int = 96,
    atom_selector: str = 'heavy',
    protein_cutoff: float = 3.0,
    pdb_id: Optional[str] = None,
    normalize: str = 'percentile',
    norm_low_pct: float = 1.0,
    norm_high_pct: float = 99.0,
    center_top_pct: float = 95.0,
    center_blob_radius: float = 15.0,
    mask_structure_path: Optional[str] = None,
    mask_policy: str = 'feature_structure',
    target_ligand_path: Optional[str] = None,
    target_all_copies: bool = False,
    save_raw_features: bool = False,
    nonpolymer_cutoff: Optional[float] = None,
) -> Dict[str, object]:
    """Compute ring features for a single structure-map pair.

    Args:
        structure_path: Path to structure file (.cif or .pdb)
        map_path: Path to EM map file (.map/.mrc/.ccp4[.gz])
        radii: Radii at which to sample density
        n_dir: Number of Fibonacci-sphere sampling directions
        atom_selector: 'heavy' or 'all'
        protein_cutoff: Radius for protein masking in Angstroms
        pdb_id: Identifier to record in output
        normalize: Normalization method: 'percentile', 'upper_percentile', or 'none'
        norm_low_pct: Lower percentile used by percentile normalization
        norm_high_pct: Upper percentile used by percentile normalization
        center_top_pct: Top percentile threshold used to select high-density candidates
        center_blob_radius: Radius (A) around the peak candidate to define the main blob
        mask_structure_path: Complete model used only for density masking
        mask_policy: ``feature_structure``, ``full_polymer``,
            ``polymer_plus_known_nontarget``, or ``all_known_nontarget``
        target_ligand_path: Extracted target ligand used for coordinate-based
            exclusion under ``all_known_nontarget``
        target_all_copies: Exclude every copy of the target comp_id rather than
            only the deposited ground-truth instance
        save_raw_features: Also store pre-normalization scalar/vector features
        nonpolymer_cutoff: Optional larger mask radius for modeled non-polymers
            under the policies that mask non-polymers

    Returns:
        Dict with keys: coords, atomZ, rings, ring_vectors, radii, pdb_id,
                       chain_ids, res_ids, res_names, atom_names,
                       norm_method, norm_stats (if normalized)

    Notes:
        rings layout: [max_r0, mean_r0, std_r0, ..., max_rN, mean_rN, std_rN,
                       l2norm_r0, l2norm_r1, ..., l2norm_rN]
        Total scalar channels: n_radii * 4 (3 density stats + 1 l=2 norm per radius)
    """
    dirs = unit_vectors_on_sphere(n_dir, seed=123)
    pid = pdb_id or os.path.splitext(os.path.basename(structure_path))[0]

    _ccp4, grid = open_map(map_path)

    xyz, meta = load_structure_atoms(structure_path, heavy_only=(atom_selector == 'heavy'))
    if len(xyz) == 0:
        raise ValueError(f'No atoms loaded from structure: {structure_path}')

    model, mask_metadata = prepare_mask_model(
        feature_structure_path=structure_path,
        mask_structure_path=mask_structure_path,
        mask_policy=mask_policy,
        target_ligand_path=target_ligand_path,
        target_all_copies=target_all_copies,
    )
    grid.normalize()
    mask_value = grid.sum() / grid.point_count
    grid = preprocess_protein_mask(
        grid, model,
        ignore_hydrogen=True,
        ignore_zero_occupancy_atoms=True,
        radius=protein_cutoff,
        mask_value=mask_value
    )
    if nonpolymer_cutoff is not None:
        if nonpolymer_cutoff < protein_cutoff:
            raise ValueError('nonpolymer_cutoff must be >= protein_cutoff')
        if mask_policy in NONPOLYMER_MASKING_POLICIES and nonpolymer_cutoff > protein_cutoff:
            nonpolymer_model = model.clone()
            for chain in nonpolymer_model:
                for residue in chain:
                    if classify_residue(residue) in POLYMER_CATEGORIES:
                        for atom in residue:
                            atom.occ = 0.0
            grid = preprocess_protein_mask(
                grid, nonpolymer_model,
                ignore_hydrogen=True,
                ignore_zero_occupancy_atoms=True,
                radius=nonpolymer_cutoff,
                mask_value=mask_value,
            )
    mask_metadata['protein_cutoff'] = float(protein_cutoff)
    mask_metadata['requested_nonpolymer_cutoff'] = (
        float(nonpolymer_cutoff) if nonpolymer_cutoff is not None else None
    )
    mask_metadata['effective_nonpolymer_cutoff'] = (
        float(nonpolymer_cutoff)
        if mask_policy in NONPOLYMER_MASKING_POLICIES and nonpolymer_cutoff is not None
        else float(protein_cutoff)
    )

    # rings stores [max, mean, std] per radius: shape (N_atoms, n_radii * 3)
    # channel layout: [max_r0, mean_r0, std_r0, max_r1, mean_r1, std_r1, ...]
    # After loop, l=2 norms are appended: [l2norm_r0, ..., l2norm_rN]
    # Final rings shape: (N_atoms, n_radii * 4)
    rings = np.zeros((len(xyz), len(radii) * 3), dtype=np.float32)
    ring_vectors = np.zeros((len(xyz), len(radii), 3), dtype=np.float32)
    ring_l2_norms = np.zeros((len(xyz), len(radii)), dtype=np.float32)
    # Precompute l=2 spherical harmonics for all directions (n_dir, 5)
    sh_l2 = _spherical_harmonics_l2(dirs)
    for ai in range(len(xyz)):
        for ri, rad in enumerate(radii):
            pts = xyz[ai][None, :] + float(rad) * dirs
            emv = interp_points(grid, pts)  # (n_dir,)
            rings[ai, ri * 3 + 0] = float(np.max(emv))
            rings[ai, ri * 3 + 1] = float(np.mean(emv))
            rings[ai, ri * 3 + 2] = float(np.std(emv))
            # Density-weighted direction vector (l=1 feature)
            ring_vectors[ai, ri] = np.sum(emv[:, None] * dirs, axis=0) / len(dirs)
            # l=2 density-weighted tensor squared norm (rotation-invariant scalar)
            T_r = np.sum(emv[:, None] * sh_l2, axis=0)  # (5,)
            ring_l2_norms[ai, ri] = float(np.sum(T_r ** 2))

    # Append l=2 norms to rings: final shape (N_atoms, n_radii * 4)
    rings = np.concatenate([rings, ring_l2_norms], axis=1)
    rings_raw = rings.copy()
    ring_vectors_raw = ring_vectors.copy()

    atomZ = np.array([element_to_Z(m['element']) for m in meta], dtype=np.int16)

    # Compute density_center: centroid of top-10% atoms by raw max density at first radius.
    # Uses raw (pre-normalization) values so the center reflects absolute density signal.
    # After protein masking, high values indicate nearby non-protein (ligand) density.
    
    # Density center from densest single blob (no model architecture change).
    # Multi-radius weighted scoring emphasizes local signal.
    base_weights = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    n_center_radii = min(3, len(radii))
    max_channel_indices = [ri * 3 for ri in range(n_center_radii)]
    w = base_weights[:n_center_radii]
    w = w / (w.sum() + 1e-8)  # renormalize when fewer than 3 radii are available
    score = (rings[:, max_channel_indices] * w[None, :]).sum(axis=1)

    # Geometric constraints for center selection
    prot_com = xyz.mean(axis=0).astype(np.float32)
    dists_to_com = np.linalg.norm(xyz - prot_com[None, :], axis=1)
    prot_radius = float(np.percentile(dists_to_com, 98))  # 98th percentile roughly bounds the protein

    # Candidate 1: Dense Center (Absolute densest blob, top 5%)
    dense_center = _compute_density_center_densest_blob(
        xyz=xyz, score=score, top_pct=center_top_pct, blob_radius=center_blob_radius
    )

    # Candidate 2: secondary peak/blob center (non-overlapping with primary dense blob).
    weak_center = _compute_secondary_density_center(
        xyz=xyz,
        score=score,
        primary_center=dense_center,
        top_pct=center_top_pct,
        blob_radius=center_blob_radius,
    )

    dist_dense = float(np.linalg.norm(dense_center - prot_com)) if dense_center is not None else float('inf')
    dist_weak = float(np.linalg.norm(weak_center - prot_com)) if weak_center is not None else float('inf')

    # Gating: if the densest blob is hanging too far outside the main body (> prot_radius),
    # it's likely an artifact (lipid/micelle). Fallback to the weaker, more embedded blob.
    if dist_dense <= prot_radius:
        density_center = dense_center
    elif dist_weak <= prot_radius:
        density_center = weak_center
    else:
        density_center = None

    # Extract atom identity metadata for direct lookup (avoids coordinate matching)
    chain_ids = np.array([m['chain'] for m in meta], dtype='U4')
    res_ids = np.array([m['seqid'] for m in meta], dtype=np.int32)
    res_names = np.array([m['res'] for m in meta], dtype='U4')
    atom_names = np.array([m['atom'] for m in meta], dtype='U4')

    # Normalize features
    norm_stats = {'method': normalize}
    if normalize == 'percentile':
        rings, ring_vectors, norm_stats = _normalize_percentile(
            rings, ring_vectors, low_pct=norm_low_pct, high_pct=norm_high_pct
        )
    elif normalize == 'upper_percentile':
        rings, ring_vectors, norm_stats = _normalize_upper_percentile(
            rings, ring_vectors, high_pct=norm_high_pct
        )
    elif normalize != 'none':
        raise ValueError(
            f"Unknown normalize method '{normalize}'. "
            "Use 'percentile', 'upper_percentile', or 'none'."
        )

    res = {
        'coords': xyz.astype(np.float32),
        'atomZ': atomZ,
        'rings': rings,
        'ring_vectors': ring_vectors,
        'radii': np.array(radii, dtype=np.float32),
        'pdb_id': pid,
        'norm_method': normalize,
        'norm_stats': norm_stats,
        'mask_policy': mask_policy,
        'mask_metadata': mask_metadata,
        # Atom identity for direct lookup
        'chain_ids': chain_ids,
        'res_ids': res_ids,
        'res_names': res_names,
        'atom_names': atom_names,
    }
    if density_center is not None:
        res['density_center'] = density_center.astype(np.float32)
    if save_raw_features:
        res['rings_raw'] = rings_raw
        res['ring_vectors_raw'] = ring_vectors_raw
    return res


def save_ring_features(result: Dict, out_path: str):
    """Save ring features dict to .npz and .json sidecar."""
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(out_path, **result)

    # Count unique residues and chains
    n_residues = 0
    n_chains = 0
    if 'chain_ids' in result and 'res_ids' in result:
        unique_residues = set(zip(result['chain_ids'], result['res_ids']))
        n_residues = len(unique_residues)
        n_chains = len(set(result['chain_ids']))

    with open(out_path + '.json', 'w') as f:
        dc = result.get('density_center')
        json.dump({
            'pdb_id': str(result['pdb_id']),
            'n_atoms': int(len(result['coords'])),
            'n_residues': n_residues,
            'n_chains': n_chains,
            'radii': [float(x) for x in result['radii']],
            'has_atom_identity': 'chain_ids' in result,
            'has_ring_vectors': 'ring_vectors' in result,
            'has_raw_features': 'rings_raw' in result,
            'mask_policy': str(result.get('mask_policy', 'feature_structure')),
            'mask_metadata': result.get('mask_metadata', {}),
            'density_center': [float(x) for x in dc] if dc is not None else None,
        }, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser()
    ap.add_argument('--structure', type=str, required=True, help='Structure file path (.cif or .pdb)')
    ap.add_argument('--map', dest='emap', type=str, required=True, help='EM map path (.map/.mrc/.ccp4[.gz])')
    ap.add_argument('--out', type=str, required=True, help='Output .npz filepath')
    ap.add_argument('--pdb-id', type=str, default=None, help='Identifier to record in output; defaults to structure basename')
    ap.add_argument('--radii', type=float, nargs='+', default=[2.5, 3.5, 4.5, 5.5, 6.5, 8.0])
    ap.add_argument('--n-dir', type=int, default=96)
    ap.add_argument('--atom-selector', type=str, default='heavy', choices=['heavy', 'all'])
    ap.add_argument('--protein-cutoff', type=float, default=3, help='Distance cutoff (Angstrom) to define protein regions')
    ap.add_argument('--normalize', type=str, default='percentile', choices=['percentile', 'upper_percentile', 'none'],
                    help='Normalization method (default: percentile)')
    ap.add_argument('--norm-low-pct', type=float, default=1.0,
                    help='Lower percentile for percentile normalization (default: 1.0)')
    ap.add_argument('--norm-high-pct', type=float, default=99.0,
                    help='Upper percentile for percentile normalization (default: 99.0)')
    ap.add_argument('--center-top-pct', type=float, default=95.0,
                    help='Top percentile to select high-density candidates for center (default: 95.0)')
    ap.add_argument('--center-blob-radius', type=float, default=15.0,
                    help='Blob radius (A) around peak candidate for center estimation (default: 15.0)')
    ap.add_argument('--mask-structure', type=str, default=None,
                    help='Complete PDB/mmCIF used only for masking')
    ap.add_argument('--mask-policy', type=str, default='feature_structure',
                    choices=['feature_structure', 'full_polymer',
                             'polymer_plus_known_nontarget', 'all_known_nontarget'])
    ap.add_argument('--target-ligand', type=str, default=None,
                    help='Extracted target ligand; required for all_known_nontarget')
    ap.add_argument('--target-all-copies', action='store_true',
                    help='Leave every copy of the target comp_id unmasked, not only '
                         'the deposited ground-truth instance')
    ap.add_argument('--save-raw-features', action='store_true',
                    help='Store pre-normalization rings_raw and ring_vectors_raw')
    ap.add_argument('--nonpolymer-cutoff', type=float, default=None,
                    help='Optional larger mask radius for modeled non-polymers')
    args = ap.parse_args()

    if not os.path.exists(args.structure):
        logging.error('Structure file not found: %s', args.structure)
        sys.exit(1)
    if not os.path.exists(args.emap):
        logging.error('Map file not found: %s', args.emap)
        sys.exit(1)
    if args.mask_policy != 'feature_structure' and not args.mask_structure:
        ap.error('--mask-structure is required unless --mask-policy=feature_structure')
    if args.mask_structure and not os.path.exists(args.mask_structure):
        ap.error(f'--mask-structure not found: {args.mask_structure}')
    if args.mask_policy == 'all_known_nontarget' and not args.target_ligand:
        ap.error('--target-ligand is required for all_known_nontarget')
    if args.target_ligand and not os.path.exists(args.target_ligand):
        ap.error(f'--target-ligand not found: {args.target_ligand}')

    try:
        result = compute_ring_features(
            structure_path=args.structure,
            map_path=args.emap,
            radii=args.radii,
            n_dir=args.n_dir,
            atom_selector=args.atom_selector,
            protein_cutoff=args.protein_cutoff,
            pdb_id=args.pdb_id,
            normalize=args.normalize,
            norm_low_pct=args.norm_low_pct,
            norm_high_pct=args.norm_high_pct,
            center_top_pct=args.center_top_pct,
            center_blob_radius=args.center_blob_radius,
            mask_structure_path=args.mask_structure,
            mask_policy=args.mask_policy,
            target_ligand_path=args.target_ligand,
            target_all_copies=args.target_all_copies,
            save_raw_features=args.save_raw_features,
            nonpolymer_cutoff=args.nonpolymer_cutoff,
        )
        save_ring_features(result, args.out)
    except Exception as e:
        logging.error('Failed: %s', e)
        sys.exit(1)

if __name__ == '__main__':
    main()
