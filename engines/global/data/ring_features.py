"""Ring feature loading and alignment utilities for DiffDock-EM.

This module handles loading ring feature .npz files and aligning them with
DiffDock's receptor graph representations (both residue-level for CGModel
and atom-level for AAModel).

Ring features are per-heavy-atom max EM density values at radii [4,5,6,7] Å,
computed from the cryo-EM map with protein masking.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

# Import constants for residue classification
try:
    from datasets.constants import AMINO_ACIDS, NUCLEOTIDES, CA_ATOM_INDEX
except ImportError:
    # Fallback definitions if constants not available
    AMINO_ACIDS = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'MSE', 'PYL', 'SEC',
    }
    NUCLEOTIDES = {
        'DA', 'DC', 'DG', 'DT', 'DU', 'A', 'C', 'G', 'U',
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
    }
    CA_ATOM_INDEX = 1


def load_ring_features(npz_path: Union[str, Path]) -> dict:
    """Load ring features from .npz file.

    Args:
        npz_path: Path to the .npz file containing ring features

    Returns:
        Dict with keys:
            - coords: (N_atoms, 3) float32 - atom coordinates
            - atomZ: (N_atoms,) int16 - atomic numbers
            - rings: (N_atoms, n_radii) float32 - max density at each radius
            - radii: (n_radii,) float32 - radii used (typically [4,5,6,7])
            - pdb_id: str - PDB identifier
            - chain_ids: (N_atoms,) str - chain identifiers (if available)
            - res_ids: (N_atoms,) int32 - residue numbers (if available)
            - res_names: (N_atoms,) str - residue names (if available)
            - atom_names: (N_atoms,) str - atom names (if available)
    """
    data = np.load(npz_path, allow_pickle=True)
    result = {
        'coords': data['coords'],
        'atomZ': data['atomZ'],
        'rings': data['rings'],
        'radii': data['radii'],
        'pdb_id': str(data['pdb_id']),
    }
    # Load optional atom identity metadata (new format)
    for key in ['chain_ids', 'res_ids', 'res_names', 'atom_names']:
        if key in data:
            result[key] = data[key]
    # Load optional l=1 directional ring vectors (backward-compatible)
    result['ring_vectors'] = data['ring_vectors'] if 'ring_vectors' in data else None
    # Load density_center: centroid of high-density region in PDB coordinate space
    result['density_center'] = data['density_center'] if 'density_center' in data else None
    return result


def has_atom_identity(ring_data: dict) -> bool:
    """Check if ring feature data has atom identity metadata for direct lookup."""
    return all(k in ring_data for k in ['chain_ids', 'res_ids', 'res_names', 'atom_names'])


def align_ring_features_by_identity(
    ring_data: dict,
    chain_ids: List[str],
    res_ids: List[int],
    ref_atom_name: str = 'CA',
    ring_dim: int = 4,
) -> np.ndarray:
    """Align ring features using atom identity metadata (fast direct lookup).

    This is the preferred method when ring features have atom identity metadata.
    Instead of coordinate matching, directly looks up ring features by
    (chain_id, res_id, atom_name).

    Args:
        ring_data: Dict from load_ring_features() with atom identity metadata
        chain_ids: List of chain IDs for each residue in DiffDock graph
        res_ids: List of residue numbers for each residue in DiffDock graph
        ref_atom_name: Reference atom name to use ('CA' for proteins, 'C1\'' for nucleotides)
        ring_dim: Number of ring feature channels

    Returns:
        (N_res, ring_dim) ring features per residue
    """
    if not has_atom_identity(ring_data):
        raise ValueError("Ring data does not have atom identity metadata")

    n_res = len(chain_ids)
    ring_values = ring_data['rings']

    # Handle ring_dim mismatch
    if ring_values.shape[1] != ring_dim:
        if ring_values.shape[1] > ring_dim:
            ring_values = ring_values[:, :ring_dim]
        else:
            padded = np.zeros((ring_values.shape[0], ring_dim), dtype=np.float32)
            padded[:, :ring_values.shape[1]] = ring_values
            ring_values = padded

    # Build lookup index: (chain, res_id, atom_name) -> ring feature index
    ring_chain_ids = ring_data['chain_ids']
    ring_res_ids = ring_data['res_ids']
    ring_atom_names = ring_data['atom_names']

    lookup = {}
    for idx in range(len(ring_chain_ids)):
        key = (str(ring_chain_ids[idx]), int(ring_res_ids[idx]), str(ring_atom_names[idx]))
        lookup[key] = idx

    # Look up ring features for each residue
    result = np.zeros((n_res, ring_dim), dtype=np.float32)
    for i in range(n_res):
        key = (str(chain_ids[i]), int(res_ids[i]), ref_atom_name)
        if key in lookup:
            result[i] = ring_values[lookup[key]]

    return result


def align_ring_features_residue(
    ring_coords: np.ndarray,
    ring_features: np.ndarray,
    residue_atom_coords: np.ndarray,
    match_threshold: float = 0.5,
    aggregation: str = 'mean',
    ring_vectors: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Align ring features to residue-level representation for CGModel.

    For each residue, finds heavy atoms from the ring feature set that match
    the residue's atom coordinates, then aggregates their ring features.

    Args:
        ring_coords: (N_ring_atoms, 3) coordinates from ring feature .npz
        ring_features: (N_ring_atoms, n_radii) ring feature values
        residue_atom_coords: (N_res, 14, 3) atom coordinates per residue
            (from DiffDock's get_coords, NaN for missing atoms)
        match_threshold: Max distance in Å to consider atoms as matching
        aggregation: 'mean' or 'max' for combining matched atoms
        ring_vectors: (N_ring_atoms, n_radii, 3) directional ring vectors (optional)

    Returns:
        If ring_vectors is None: (N_res, n_radii) aggregated ring features per residue
        If ring_vectors is provided: tuple of ((N_res, n_radii), (N_res, n_radii, 3))
    """
    if cKDTree is None:
        raise ImportError("scipy is required for KD-tree alignment")

    n_res = residue_atom_coords.shape[0]
    n_scalar = ring_features.shape[1]  # scalar channels: n_radii * n_stats (e.g. 12 = 4 radii * 3 stats)
    has_vectors = ring_vectors is not None
    n_vec_radii = ring_vectors.shape[1] if has_vectors else None  # actual number of radii (e.g. 4)

    # Build KD-tree from ring feature coordinates
    tree = cKDTree(ring_coords)

    result = np.zeros((n_res, n_scalar), dtype=np.float32)
    result_vectors = np.zeros((n_res, n_vec_radii, 3), dtype=np.float32) if has_vectors else None

    for res_idx in range(n_res):
        # Get all valid (non-NaN) atom coords for this residue
        res_atoms = residue_atom_coords[res_idx]  # (14, 3)
        valid_mask = ~np.isnan(res_atoms).any(axis=1)
        valid_coords = res_atoms[valid_mask]

        if len(valid_coords) == 0:
            continue

        # Find matching ring feature atoms for each residue atom
        matched_features = []
        matched_vectors = [] if has_vectors else None
        for coord in valid_coords:
            # Query for nearest ring feature atom
            dist, idx = tree.query(coord, k=1)
            if dist <= match_threshold:
                matched_features.append(ring_features[idx])
                if has_vectors:
                    matched_vectors.append(ring_vectors[idx])

        if len(matched_features) > 0:
            matched_features = np.stack(matched_features, axis=0)
            if aggregation == 'mean':
                result[res_idx] = matched_features.mean(axis=0)
            elif aggregation == 'max':
                result[res_idx] = matched_features.max(axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            if has_vectors:
                matched_vectors = np.stack(matched_vectors, axis=0)
                # Always mean for vectors (max doesn't make sense for directions)
                result_vectors[res_idx] = matched_vectors.mean(axis=0)

    if has_vectors:
        return result, result_vectors
    return result


def align_ring_features_to_residues(
    ring_npz_path: Union[str, Path],
    all_coords: np.ndarray,
    residue_names: Optional[List[str]] = None,
    chain_ids: Optional[List[str]] = None,
    res_ids: Optional[List[int]] = None,
    ring_dim: int = 4,
    match_threshold: float = 0.5,
) -> np.ndarray:
    """Align ring features to residues.

    Uses direct lookup by atom identity (chain, res_id, atom_name) if available,
    otherwise falls back to coordinate-based matching.

    Args:
        ring_npz_path: Path to .npz with 'coords' and 'rings'
        all_coords: [N_res, 14, 3] - residue atom coordinates (index 1 = Cα)
        residue_names: List of residue names to detect protein vs DNA/RNA
                      If None, assumes all are proteins
        chain_ids: List of chain IDs for each residue (for direct lookup)
        res_ids: List of residue numbers for each residue (for direct lookup)
        ring_dim: Number of ring feature channels
        match_threshold: Distance threshold for coordinate matching (fallback)

    Returns:
        [N_res, ring_dim] - ring features per residue
    """
    ring_data = load_ring_features(ring_npz_path)

    # Try direct lookup if atom identity metadata is available
    if has_atom_identity(ring_data) and chain_ids is not None and res_ids is not None:
        # Use direct lookup by (chain, res_id, CA)
        return align_ring_features_by_identity(
            ring_data, chain_ids, res_ids, ref_atom_name='CA', ring_dim=ring_dim
        )

    # Fallback to coordinate-based matching
    if cKDTree is None:
        raise ImportError("scipy is required for KD-tree alignment (fallback)")

    ring_coords = ring_data['coords']  # [N_ring_atoms, 3]
    ring_values = ring_data['rings']   # [N_ring_atoms, ring_dim]

    # Handle ring_dim mismatch
    if ring_values.shape[1] != ring_dim:
        if ring_values.shape[1] > ring_dim:
            ring_values = ring_values[:, :ring_dim]
        else:
            # Pad with zeros
            padded = np.zeros((ring_values.shape[0], ring_dim), dtype=np.float32)
            padded[:, :ring_values.shape[1]] = ring_values
            ring_values = padded

    # Build KD-tree on ring feature atoms
    tree = cKDTree(ring_coords)
    n_res = len(all_coords)
    residue_rings = np.zeros((n_res, ring_dim), dtype=np.float32)

    for i in range(n_res):
        res_name = residue_names[i] if residue_names else 'ALA'

        # Determine reference atom coordinate
        if res_name in NUCLEOTIDES:
            # For nucleotides, find C1' by searching for carbon in sugar ring region
            # C1' is typically the first non-backbone carbon - use mean of all atoms
            # as fallback since exact atom ordering varies
            valid_mask = ~np.isnan(all_coords[i]).any(axis=1)
            valid_coords = all_coords[i, valid_mask]
            if len(valid_coords) > 0:
                ref_coord = valid_coords.mean(axis=0)
            else:
                continue
        else:
            # For proteins, use Cα (index 1)
            ref_coord = all_coords[i, CA_ATOM_INDEX, :]

        # Skip if reference coordinate is invalid
        if np.isnan(ref_coord).any():
            # Fallback: use centroid of valid atoms
            valid_mask = ~np.isnan(all_coords[i]).any(axis=1)
            valid_coords = all_coords[i, valid_mask]
            if len(valid_coords) > 0:
                ref_coord = valid_coords.mean(axis=0)
            else:
                continue

        # Query nearest ring feature atom
        dist, idx = tree.query(ref_coord)
        if dist < match_threshold:
            residue_rings[i] = ring_values[idx]

    return residue_rings


def align_ring_features_atom(
    ring_coords: np.ndarray,
    ring_features: np.ndarray,
    atom_coords: np.ndarray,
    match_threshold: float = 0.5,
    ring_vectors: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Align ring features to atom-level representation for AAModel.

    For each atom in the DiffDock graph, finds the nearest matching atom
    in the ring feature set.

    Args:
        ring_coords: (N_ring_atoms, 3) coordinates from ring feature .npz
        ring_features: (N_ring_atoms, n_radii) ring feature values
        atom_coords: (N_atoms, 3) atom coordinates from DiffDock
        match_threshold: Max distance in Å to consider atoms as matching
        ring_vectors: (N_ring_atoms, n_radii, 3) directional ring vectors (optional)

    Returns:
        If ring_vectors is None: (N_atoms, n_radii) ring features per atom (zeros for unmatched)
        If ring_vectors is provided: tuple of ((N_atoms, n_radii), (N_atoms, n_radii, 3))
    """
    if cKDTree is None:
        raise ImportError("scipy is required for KD-tree alignment")

    n_atoms = atom_coords.shape[0]
    n_radii = ring_features.shape[1]
    has_vectors = ring_vectors is not None

    # Build KD-tree from ring feature coordinates
    tree = cKDTree(ring_coords)

    result = np.zeros((n_atoms, n_radii), dtype=np.float32)
    result_vectors = np.zeros((n_atoms, n_radii, 3), dtype=np.float32) if has_vectors else None

    # Query all atoms at once
    dists, indices = tree.query(atom_coords, k=1)

    # Only keep matches within threshold
    valid_mask = dists <= match_threshold
    result[valid_mask] = ring_features[indices[valid_mask]]
    if has_vectors:
        result_vectors[valid_mask] = ring_vectors[indices[valid_mask]]

    if has_vectors:
        return result, result_vectors
    return result


def align_ring_features_atoms_by_identity(
    ring_data: dict,
    chain_ids: List[str],
    res_ids: List[int],
    atom_names: List[str],
    ring_dim: int = 4,
) -> np.ndarray:
    """Align ring features to atoms using identity metadata (fast direct lookup).

    Args:
        ring_data: Dict from load_ring_features() with atom identity metadata
        chain_ids: List of chain IDs for each atom in DiffDock graph
        res_ids: List of residue numbers for each atom
        atom_names: List of atom names for each atom
        ring_dim: Number of ring feature channels

    Returns:
        (N_atoms, ring_dim) ring features per atom
    """
    if not has_atom_identity(ring_data):
        raise ValueError("Ring data does not have atom identity metadata")

    n_atoms = len(chain_ids)
    ring_values = ring_data['rings']

    # Handle ring_dim mismatch
    if ring_values.shape[1] != ring_dim:
        if ring_values.shape[1] > ring_dim:
            ring_values = ring_values[:, :ring_dim]
        else:
            padded = np.zeros((ring_values.shape[0], ring_dim), dtype=np.float32)
            padded[:, :ring_values.shape[1]] = ring_values
            ring_values = padded

    # Build lookup index
    ring_chain_ids = ring_data['chain_ids']
    ring_res_ids = ring_data['res_ids']
    ring_atom_names = ring_data['atom_names']

    lookup = {}
    for idx in range(len(ring_chain_ids)):
        key = (str(ring_chain_ids[idx]), int(ring_res_ids[idx]), str(ring_atom_names[idx]))
        lookup[key] = idx

    # Look up ring features for each atom
    result = np.zeros((n_atoms, ring_dim), dtype=np.float32)
    for i in range(n_atoms):
        key = (str(chain_ids[i]), int(res_ids[i]), str(atom_names[i]))
        if key in lookup:
            result[i] = ring_values[lookup[key]]

    return result


def align_ring_features_to_atoms(
    ring_npz_path: Union[str, Path],
    atom_coords: np.ndarray,
    chain_ids: Optional[List[str]] = None,
    res_ids: Optional[List[int]] = None,
    atom_names: Optional[List[str]] = None,
    ring_dim: int = 4,
    match_threshold: float = 0.5,
) -> np.ndarray:
    """Atom-level alignment for AAModel.

    Uses direct lookup by atom identity if available, otherwise falls back
    to coordinate-based matching.

    Args:
        ring_npz_path: Path to .npz with 'coords' and 'rings'
        atom_coords: [N_atoms, 3] - atom coordinates from DiffDock
        chain_ids: List of chain IDs for each atom (for direct lookup)
        res_ids: List of residue numbers for each atom (for direct lookup)
        atom_names: List of atom names for each atom (for direct lookup)
        ring_dim: Number of ring feature channels
        match_threshold: Distance threshold for coordinate matching (fallback)

    Returns:
        [N_atoms, ring_dim] - ring features per atom
    """
    ring_data = load_ring_features(ring_npz_path)

    # Try direct lookup if atom identity metadata is available
    if (has_atom_identity(ring_data) and
        chain_ids is not None and res_ids is not None and atom_names is not None):
        return align_ring_features_atoms_by_identity(
            ring_data, chain_ids, res_ids, atom_names, ring_dim=ring_dim
        )

    # Fallback to coordinate-based matching
    if cKDTree is None:
        raise ImportError("scipy is required for KD-tree alignment (fallback)")

    ring_coords = ring_data['coords']
    ring_values = ring_data['rings']

    # Handle ring_dim mismatch
    if ring_values.shape[1] != ring_dim:
        if ring_values.shape[1] > ring_dim:
            ring_values = ring_values[:, :ring_dim]
        else:
            padded = np.zeros((ring_values.shape[0], ring_dim), dtype=np.float32)
            padded[:, :ring_values.shape[1]] = ring_values
            ring_values = padded

    tree = cKDTree(ring_coords)

    dists, indices = tree.query(atom_coords)
    atom_rings = np.zeros((len(atom_coords), ring_dim), dtype=np.float32)

    valid = dists < match_threshold
    atom_rings[valid] = ring_values[indices[valid]]

    return atom_rings


def get_ring_features_for_complex(
    ring_npz_path: Union[str, Path],
    residue_atom_coords: Optional[np.ndarray] = None,
    atom_coords: Optional[np.ndarray] = None,
    match_threshold: float = 0.5,
    aggregation: str = 'mean',
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load and align ring features for a complex.

    Args:
        ring_npz_path: Path to ring feature .npz file
        residue_atom_coords: (N_res, 14, 3) for CGModel alignment
        atom_coords: (N_atoms, 3) for AAModel alignment
        match_threshold: Max distance for atom matching
        aggregation: Aggregation method for residue-level features

    Returns:
        Tuple of (residue_ring_features, atom_ring_features)
        Each is a torch.Tensor or None if corresponding coords not provided
    """
    ring_data = load_ring_features(ring_npz_path)

    residue_features = None
    atom_features = None

    if residue_atom_coords is not None:
        res_np = align_ring_features_residue(
            ring_data['coords'],
            ring_data['rings'],
            residue_atom_coords,
            match_threshold=match_threshold,
            aggregation=aggregation,
        )
        residue_features = torch.from_numpy(res_np).float()

    if atom_coords is not None:
        atom_np = align_ring_features_atom(
            ring_data['coords'],
            ring_data['rings'],
            atom_coords,
            match_threshold=match_threshold,
        )
        atom_features = torch.from_numpy(atom_np).float()

    return residue_features, atom_features


def find_ring_feature_file(
    pdb_id: str,
    ring_features_dir: Union[str, Path],
) -> Optional[Path]:
    """Find ring feature .npz file for a PDB ID.

    Searches for files matching patterns:
    - {ring_features_dir}/{pdb_id}/{pdb_id}_ring_features.npz
    - {ring_features_dir}/{pdb_id}_ring_features.npz
    - {ring_features_dir}/processed/{pdb_id}/{pdb_id}_ring_features.npz

    Args:
        pdb_id: PDB identifier (case-insensitive)
        ring_features_dir: Root directory containing ring feature files

    Returns:
        Path to .npz file if found, None otherwise
    """
    ring_features_dir = Path(ring_features_dir)
    pdb_id_upper = pdb_id.upper()
    pdb_id_lower = pdb_id.lower()

    candidates = [
        # Standard patterns
        ring_features_dir / pdb_id_upper / f'{pdb_id_upper}_ring_features.npz',
        ring_features_dir / f'{pdb_id_upper}_ring_features.npz',
        ring_features_dir / pdb_id_lower / f'{pdb_id_lower}_ring_features.npz',
        ring_features_dir / f'{pdb_id_lower}_ring_features.npz',
        # Under processed/ subdirectory
        ring_features_dir / 'processed' / pdb_id_upper / f'{pdb_id_upper}_ring_features.npz',
        ring_features_dir / 'processed' / pdb_id_lower / f'{pdb_id_lower}_ring_features.npz',
    ]

    for path in candidates:
        if path.exists():
            return path

    return None
