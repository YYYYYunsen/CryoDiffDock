"""EM Dataset for DiffDock-EM finetuning.

Loads cryo-EM structures with ring features for training/validation.
Each sample includes:
- Protein structure (receptor graph)
- Ligand structure
- Ring features (per-residue or per-atom max EM density at multiple radii)
"""

import copy
import multiprocessing
import os
import pickle
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — no file locking needed for single-process use
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import RemoveAllHs
from torch_geometric.data import Dataset, HeteroData
from tqdm import tqdm

from datasets.process_mols import (
    read_molecule,
    get_lig_graph_with_matching,
    generate_conformer,
    moad_extract_receptor_structure,
)
from datasets.parse_chi import get_coords
from data.ring_features import (
    load_ring_features,
    align_ring_features_residue,
    align_ring_features_atom,
    find_ring_feature_file,
)
from utils.utils import read_strings_from_txt


_WORKER_ID_TO_EMBEDDINGS = None


def _init_process_complex_worker(id_to_embeddings):
    """Initialize per-process shared state for preprocessing workers."""
    global _WORKER_ID_TO_EMBEDDINGS
    _WORKER_ID_TO_EMBEDDINGS = id_to_embeddings


def _fit_ring_scalar_dim(features: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad/truncate scalar ring feature channels to target_dim."""
    cur_dim = features.shape[1]
    if cur_dim == target_dim:
        return features
    if cur_dim > target_dim:
        return features[:, :target_dim]
    padded = np.zeros((features.shape[0], target_dim), dtype=np.float32)
    padded[:, :cur_dim] = features
    return padded


def _fit_ring_vector_dim(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad/truncate vector ring channels (axis=1) to target_dim."""
    cur_dim = vectors.shape[1]
    if cur_dim == target_dim:
        return vectors
    if cur_dim > target_dim:
        return vectors[:, :target_dim]
    padded = np.zeros((vectors.shape[0], target_dim, 3), dtype=np.float32)
    padded[:, :cur_dim] = vectors
    return padded


def _clip_ring_vector_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip per-vector L2 norm in ring_vectors to improve numeric stability.

    Args:
        vectors: (N, R, 3)
        max_norm: maximum allowed L2 norm for each 3D vector.
    """
    if max_norm is None or max_norm <= 0:
        return vectors
    out = vectors.astype(np.float32, copy=True)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)  # (N, R, 1)
    scale = np.minimum(1.0, max_norm / (norms + 1e-8))
    out *= scale
    return out


def _process_complex_worker(args):
    """Module-level worker for parallel preprocessing (must be picklable)."""
    pdb_id, cfg = args
    id_to_embeddings = _WORKER_ID_TO_EMBEDDINGS
    try:
        from datasets.process_mols import (
            get_lig_graph_with_matching,
            moad_extract_receptor_structure,
        )
        from datasets.parse_chi import get_coords
        from data.ring_features import (
            load_ring_features,
            align_ring_features_residue,
            align_ring_features_atom,
            find_ring_feature_file,
        )
        import prody as pr

        complex_dir = os.path.join(cfg['data_dir'], pdb_id)
        if not os.path.exists(complex_dir):
            raise FileNotFoundError(f'Complex directory not found: {complex_dir}')

        lig = read_mol(cfg['data_dir'], pdb_id, suffix=cfg['ligand_file'], remove_hs=False)
        if lig is None:
            raise ValueError(f'Could not read ligand for {pdb_id}')
        if cfg['max_lig_size'] is not None and lig.GetNumHeavyAtoms() > cfg['max_lig_size']:
            raise ValueError(f'Ligand too large: {lig.GetNumHeavyAtoms()} > {cfg["max_lig_size"]}')

        protein_path = None
        for suffix in [f'{pdb_id}_{cfg["protein_file"]}.pdb', f'{pdb_id}_protein.pdb', f'{pdb_id}.pdb']:
            path = os.path.join(complex_dir, suffix)
            if os.path.exists(path):
                protein_path = path
                break
        if protein_path is None:
            raise FileNotFoundError(f'Protein file not found for {pdb_id}')

        lm_embeddings = None
        if id_to_embeddings is not None:
            lm_embeddings = id_to_embeddings.get(pdb_id.upper()) or id_to_embeddings.get(pdb_id.lower())

        complex_graph = HeteroData()
        complex_graph['name'] = pdb_id

        get_lig_graph_with_matching(
            lig, complex_graph,
            popsize=cfg['popsize'], maxiter=cfg['maxiter'],
            matching=cfg['matching'], keep_original=True,
            num_conformers=1, remove_hs=cfg['remove_hs'],
        )

        pdb_struct = pr.parsePDB(protein_path)
        residue_atom_coords = get_coords(pdb_struct)

        moad_extract_receptor_structure(
            path=protein_path, complex_graph=complex_graph,
            neighbor_cutoff=cfg['receptor_radius'],
            max_neighbors=cfg['c_alpha_max_neighbors'],
            lm_embeddings=lm_embeddings,
            knn_only_graph=cfg['knn_only_graph'],
            all_atoms=cfg['all_atoms'],
            atom_cutoff=cfg['atom_radius'] if cfg['all_atoms'] else None,
            atom_max_neighbors=cfg['atom_max_neighbors'] if cfg['all_atoms'] else None,
        )

        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if cfg['all_atoms'] and hasattr(complex_graph['atom'], 'pos'):
            complex_graph['atom'].pos -= protein_center
        complex_graph.original_center = protein_center

        if isinstance(complex_graph['ligand'].pos, list):
            for i in range(len(complex_graph['ligand'].pos)):
                complex_graph['ligand'].pos[i] -= protein_center
        else:
            complex_graph['ligand'].pos -= protein_center

        ring_path = find_ring_feature_file(pdb_id, cfg['ring_features_dir'])
        if ring_path is not None:
            ring_data = load_ring_features(ring_path)
            ring_vectors = ring_data.get('ring_vectors')

            align_result = align_ring_features_residue(
                ring_data['coords'], ring_data['rings'], residue_atom_coords,
                match_threshold=cfg['ring_match_threshold'],
                aggregation=cfg['ring_aggregation'],
                ring_vectors=ring_vectors,
            )
            if ring_vectors is not None:
                residue_ring_features, residue_ring_vectors = align_result
                residue_ring_vectors = _fit_ring_vector_dim(residue_ring_vectors, cfg['ring_vec_dim'])
                residue_ring_vectors = _clip_ring_vector_norm(
                    residue_ring_vectors, cfg['ring_vector_clip_max_norm']
                )
                complex_graph['receptor'].ring_vectors = torch.from_numpy(residue_ring_vectors).float()
            else:
                residue_ring_features = align_result
                n_res = complex_graph['receptor'].pos.shape[0]
                complex_graph['receptor'].ring_vectors = torch.zeros(n_res, cfg['ring_vec_dim'], 3)

            residue_ring_features = _fit_ring_scalar_dim(residue_ring_features, cfg['ring_dim'])
            complex_graph['receptor'].ring_features = torch.from_numpy(residue_ring_features).float()

            if cfg['all_atoms'] and hasattr(complex_graph, 'atom') and hasattr(complex_graph['atom'], 'pos'):
                atom_coords = complex_graph['atom'].pos.cpu().numpy()
                atom_align_result = align_ring_features_atom(
                    ring_data['coords'], ring_data['rings'], atom_coords,
                    match_threshold=cfg['ring_match_threshold'],
                    ring_vectors=ring_vectors,
                )
                if ring_vectors is not None:
                    atom_ring_features, atom_ring_vectors = atom_align_result
                    atom_ring_vectors = _fit_ring_vector_dim(atom_ring_vectors, cfg['ring_vec_dim'])
                    atom_ring_vectors = _clip_ring_vector_norm(
                        atom_ring_vectors, cfg['ring_vector_clip_max_norm']
                    )
                    complex_graph['atom'].ring_vectors = torch.from_numpy(atom_ring_vectors).float()
                else:
                    atom_ring_features = atom_align_result
                    n_atoms = complex_graph['atom'].pos.shape[0]
                    complex_graph['atom'].ring_vectors = torch.zeros(n_atoms, cfg['ring_vec_dim'], 3)
                atom_ring_features = _fit_ring_scalar_dim(atom_ring_features, cfg['ring_dim'])
                complex_graph['atom'].ring_features = torch.from_numpy(atom_ring_features).float()
        else:
            n_res = complex_graph['receptor'].pos.shape[0]
            complex_graph['receptor'].ring_features = torch.zeros(n_res, cfg['ring_dim'])
            complex_graph['receptor'].ring_vectors = torch.zeros(n_res, cfg['ring_vec_dim'], 3)
            if cfg['all_atoms'] and hasattr(complex_graph, 'atom'):
                n_atoms = complex_graph['atom'].pos.shape[0]
                complex_graph['atom'].ring_features = torch.zeros(n_atoms, cfg['ring_dim'])
                complex_graph['atom'].ring_vectors = torch.zeros(n_atoms, cfg['ring_vec_dim'], 3)

        return pdb_id, [complex_graph], [lig], None
    except Exception as e:
        return pdb_id, None, None, str(e)


def print_statistics(complex_graphs):
    """Print dataset statistics (same pattern as pdbbind.py)."""
    statistics = ([], [], [], [], [], [])
    for complex_graph in complex_graphs:
        lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
        radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
        molecule_center = torch.mean(lig_pos, dim=0)
        radius_molecule = torch.max(
            torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
        distance_center = torch.linalg.vector_norm(molecule_center)
        statistics[0].append(radius_protein)
        statistics[1].append(radius_molecule)
        statistics[2].append(distance_center)
        if "rmsd_matching" in complex_graph:
            statistics[3].append(complex_graph.rmsd_matching)
        else:
            statistics[3].append(0)
        statistics[4].append(int(complex_graph.random_coords) if "random_coords" in complex_graph else -1)
        if "random_coords" in complex_graph and complex_graph.random_coords and "rmsd_matching" in complex_graph:
            statistics[5].append(complex_graph.rmsd_matching)
    if len(statistics[5]) == 0:
        statistics[5].append(-1)
    name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching', 'random coordinates', 'random rmsd matching']
    print('Number of complexes: ', len(complex_graphs))
    for i in range(len(name)):
        array = np.asarray(statistics[i])
        print(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def read_mol(data_dir, name, suffix='ligand', remove_hs=False):
    """Read ligand molecule from SDF or MOL2 file."""
    for ext in ['.sdf', '.mol2']:
        path = os.path.join(data_dir, name, f'{name}_{suffix}{ext}')
        if os.path.exists(path):
            mol = Chem.MolFromMolFile(path, removeHs=remove_hs) if ext == '.sdf' else Chem.MolFromMol2File(path, removeHs=remove_hs)
            if mol is not None:
                return mol
    return None


class EMDataset(Dataset):
    """Dataset for cryo-EM structures with ring features.

    Args:
        root: Root directory containing structure folders
        split_path: Path to text file listing PDB IDs to include
        ring_features_dir: Directory containing ring feature .npz files
        transform: PyTorch Geometric transform to apply
        cache_path: Directory for caching processed graphs
        receptor_radius: Radius for receptor graph construction
        c_alpha_max_neighbors: Max neighbors for Cα graph
        all_atoms: Whether to use all-atom representation (for AAModel)
        atom_radius: Radius for atom graph (if all_atoms=True)
        atom_max_neighbors: Max neighbors for atom graph
        esm_embeddings_path: Path to precomputed ESM embeddings
        limit_complexes: Limit number of complexes (0 = no limit)
        remove_hs: Whether to remove hydrogens from ligand
        num_conformers: Number of ligand conformers to generate
        max_lig_size: Maximum ligand size (heavy atoms)
        protein_file: Protein file suffix (default: protein_processed)
        ligand_file: Ligand file suffix (default: ligand)
        knn_only_graph: Use KNN-only graph construction
        ring_aggregation: How to aggregate ring features per residue ('mean' or 'max')
        ring_match_threshold: Distance threshold for atom matching (Å)
    """

    def __init__(
        self,
        root: str,
        split_path: str,
        ring_features_dir: str,
        transform=None,
        cache_path: str = 'data/cache_em',
        receptor_radius: float = 30,
        c_alpha_max_neighbors: Optional[int] = None,
        all_atoms: bool = False,
        atom_radius: float = 5,
        atom_max_neighbors: Optional[int] = None,
        esm_embeddings_path: Optional[str] = None,
        limit_complexes: int = 0,
        remove_hs: bool = False,
        num_conformers: int = 1,
        max_lig_size: Optional[int] = None,
        protein_file: str = 'protein_processed',
        ligand_file: str = 'ligand',
        knn_only_graph: bool = True,
        ring_aggregation: str = 'mean',
        ring_match_threshold: float = 0.5,
        ring_dim: int = 13,
        ring_vec_dim: int = 6,
        ring_vector_clip_max_norm: float = 20.0,
        num_workers: int = 1,
        matching: bool = True,
        popsize: int = 15,
        maxiter: int = 15,
        blacklist_path: Optional[str] = 'nan_complexes.txt',
    ):
        super(EMDataset, self).__init__(root, transform)

        self.data_dir = root
        self.split_path = split_path
        self.ring_features_dir = ring_features_dir
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.all_atoms = all_atoms
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors
        self.esm_embeddings_path = esm_embeddings_path
        self.limit_complexes = limit_complexes
        self.remove_hs = remove_hs
        self.num_conformers = num_conformers
        self.max_lig_size = max_lig_size
        self.protein_file = protein_file
        self.ligand_file = ligand_file
        self.knn_only_graph = knn_only_graph
        self.ring_aggregation = ring_aggregation
        self.ring_match_threshold = ring_match_threshold
        self.ring_dim = ring_dim
        self.ring_vec_dim = ring_vec_dim
        self.ring_vector_clip_max_norm = ring_vector_clip_max_norm
        self.num_workers = num_workers
        self.matching = matching
        self.popsize = popsize
        self.maxiter = maxiter
        self.blacklist_path = blacklist_path
        # Build cache path
        cache_suffix = (
            f'_recRad{receptor_radius}'
            f'_recMax{c_alpha_max_neighbors}'
            + ('_allatoms' if all_atoms else '')
            + (f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}' if all_atoms else '')
            + ('_esm' if esm_embeddings_path else '')
            + f'_ring{ring_aggregation}'
            + f'_rvClip{ring_vector_clip_max_norm:g}'
        )
        split_name = os.path.splitext(os.path.basename(split_path))[0]
        self.full_cache_path = os.path.join(cache_path, f'em_{split_name}{cache_suffix}')

        # Load or preprocess — file lock ensures only one process builds the cache
        os.makedirs(self.full_cache_path, exist_ok=True)
        if _HAS_FCNTL:
            lock_path = self.full_cache_path + '.lock'
            with open(lock_path, 'w') as _lock_f:
                fcntl.flock(_lock_f, fcntl.LOCK_EX)  # blocks until acquired; safe for parallel shards
                if not self._check_cache():
                    self._preprocess()
                # lock released on context exit
        else:
            if not self._check_cache():
                self._preprocess()

        self.complex_graphs, self.rdkit_ligands = self._load_cache()

        # Filter blacklisted complexes (applies to both train and val)
        if blacklist_path and os.path.exists(blacklist_path):
            with open(blacklist_path) as f:
                blacklist = {line.split('\t')[0].strip() for line in f if line.strip()}
            before = len(self.complex_graphs)
            keep = [i for i, g in enumerate(self.complex_graphs) if g.name not in blacklist]
            self.complex_graphs = [self.complex_graphs[i] for i in keep]
            self.rdkit_ligands = [self.rdkit_ligands[i] for i in keep]
            print(f'Blacklist filtered {before - len(self.complex_graphs)} complexes ({len(blacklist)} in blacklist)')

        print(f'Loaded {len(self.complex_graphs)} complexes from {self.full_cache_path}')

    def _check_cache(self) -> bool:
        """Check if cache exists and is complete (done marker must exist)."""
        return os.path.exists(os.path.join(self.full_cache_path, 'done'))

    def _load_cache(self) -> Tuple[List, List]:
        """Load all cached complexes."""
        complex_graphs = []
        rdkit_ligands = []

        i = 0
        while True:
            graph_path = os.path.join(self.full_cache_path, f'heterographs{i}.pkl')
            lig_path = os.path.join(self.full_cache_path, f'rdkit_ligands{i}.pkl')
            if not os.path.exists(graph_path):
                break
            with open(graph_path, 'rb') as f:
                complex_graphs.extend(pickle.load(f))
            with open(lig_path, 'rb') as f:
                rdkit_ligands.extend(pickle.load(f))
            i += 1

        return complex_graphs, rdkit_ligands

    def _preprocess(self):
        """Preprocess all complexes and cache them."""
        print(f'Preprocessing EM complexes from {self.split_path}')

        # Remove stale partial cache files from interrupted preprocessing runs.
        for stale in Path(self.full_cache_path).glob('heterographs*.pkl'):
            stale.unlink()
        for stale in Path(self.full_cache_path).glob('rdkit_ligands*.pkl'):
            stale.unlink()
        done_path = os.path.join(self.full_cache_path, 'done')
        if os.path.exists(done_path):
            os.remove(done_path)

        # Read PDB IDs from split file
        pdb_ids = read_strings_from_txt(self.split_path)
        if self.limit_complexes > 0:
            pdb_ids = pdb_ids[:self.limit_complexes]
        print(f'Processing {len(pdb_ids)} complexes')

        # Load ESM embeddings if provided
        id_to_embeddings = None
        if self.esm_embeddings_path is not None:
            raw_embeddings = torch.load(self.esm_embeddings_path)

            # Reorganize embeddings by PDB ID - group chains together as lists
            # moad_extract_receptor_structure expects lm_embeddings as a list of per-chain tensors
            from collections import defaultdict

            id_to_embeddings = defaultdict(list)
            chain_indices = defaultdict(list)

            for key, embedding in raw_embeddings.items():
                if '_chain_' in key:
                    pdb_id = key.split('_chain_')[0]
                    chain_idx = int(key.split('_chain_')[1])
                else:
                    pdb_id = key
                    chain_idx = 0
                id_to_embeddings[pdb_id.upper()].append(embedding)
                chain_indices[pdb_id.upper()].append(chain_idx)

            # Reorder chains by index to ensure correct order
            for pdb_id in list(id_to_embeddings.keys()):
                order = np.argsort(chain_indices[pdb_id])
                id_to_embeddings[pdb_id] = [id_to_embeddings[pdb_id][i] for i in order]

            id_to_embeddings = dict(id_to_embeddings)
            total_chains = sum(len(v) for v in id_to_embeddings.values())
            print(f'Loaded ESM embeddings for {len(id_to_embeddings)} complexes ({total_chains} chains)')

        # Process each complex (parallel when num_workers > 1). Cache results
        # incrementally to avoid holding all ESM-backed graphs in memory.
        failed = []
        processed_count = 0
        complex_graphs_buffer = []
        rdkit_ligands_buffer = []
        cache_batch_idx = 0
        cache_batch_size = 1000

        def _flush_cache_buffer(force=False):
            nonlocal cache_batch_idx, complex_graphs_buffer, rdkit_ligands_buffer
            if not force and len(complex_graphs_buffer) < cache_batch_size:
                return
            if len(complex_graphs_buffer) == 0:
                return
            graph_path = os.path.join(self.full_cache_path, f'heterographs{cache_batch_idx}.pkl')
            lig_path = os.path.join(self.full_cache_path, f'rdkit_ligands{cache_batch_idx}.pkl')
            with open(graph_path, 'wb') as f:
                pickle.dump(complex_graphs_buffer, f)
            with open(lig_path, 'wb') as f:
                pickle.dump(rdkit_ligands_buffer, f)
            cache_batch_idx += 1
            complex_graphs_buffer = []
            rdkit_ligands_buffer = []

        def _record_success(graphs, ligs):
            nonlocal processed_count
            complex_graphs_buffer.extend(graphs)
            rdkit_ligands_buffer.extend(ligs)
            processed_count += len(graphs)
            _flush_cache_buffer(force=False)

        cfg = {
            'data_dir': self.data_dir,
            'ring_features_dir': self.ring_features_dir,
            'ring_dim': self.ring_dim,
            'ring_vec_dim': self.ring_vec_dim,
            'ring_aggregation': self.ring_aggregation,
            'ring_match_threshold': self.ring_match_threshold,
            'receptor_radius': self.receptor_radius,
            'c_alpha_max_neighbors': self.c_alpha_max_neighbors,
            'all_atoms': self.all_atoms,
            'atom_radius': self.atom_radius,
            'atom_max_neighbors': self.atom_max_neighbors,
            'knn_only_graph': self.knn_only_graph,
            'matching': self.matching,
            'popsize': self.popsize,
            'maxiter': self.maxiter,
            'remove_hs': self.remove_hs,
            'protein_file': self.protein_file,
            'ligand_file': self.ligand_file,
            'max_lig_size': self.max_lig_size,
            'ring_vector_clip_max_norm': self.ring_vector_clip_max_norm,
        }

        if self.num_workers > 1:
            print(f'Preprocessing with {self.num_workers} parallel workers')
            worker_args = iter((pid, cfg) for pid in pdb_ids)
            max_pending = max(self.num_workers * 2, self.num_workers)
            mp_ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp_ctx,
                initializer=_init_process_complex_worker,
                initargs=(id_to_embeddings,),
            ) as executor:
                futures = {}

                def _submit_next():
                    try:
                        arg = next(worker_args)
                    except StopIteration:
                        return False
                    futures[executor.submit(_process_complex_worker, arg)] = arg[0]
                    return True

                for _ in range(min(max_pending, len(pdb_ids))):
                    _submit_next()

                with tqdm(total=len(pdb_ids), desc='Processing complexes') as pbar:
                    while futures:
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for future in done:
                            pid = futures.pop(future)
                            pbar.update(1)
                            try:
                                pid, graphs, ligs, err = future.result()
                            except Exception as e:
                                print(f'Failed to process {pid}: {e}')
                                failed.append(pid)
                            else:
                                if err is not None:
                                    print(f'Failed to process {pid}: {err}')
                                    failed.append(pid)
                                else:
                                    _record_success(graphs, ligs)
                            _submit_next()
        else:
            for pdb_id in tqdm(pdb_ids, desc='Processing complexes'):
                try:
                    graphs, ligs = self._process_complex(pdb_id, id_to_embeddings)
                    _record_success(graphs, ligs)
                except Exception as e:
                    print(f'Failed to process {pdb_id}: {e}')
                    failed.append(pdb_id)

        _flush_cache_buffer(force=True)

        print(f'Successfully processed {processed_count} complexes, {len(failed)} failed')
        open(os.path.join(self.full_cache_path, 'done'), 'w').close()

    def _process_complex(
        self,
        pdb_id: str,
        id_to_embeddings: Optional[Dict] = None,
    ) -> Tuple[List[HeteroData], List]:
        """Process a single complex."""
        complex_dir = os.path.join(self.data_dir, pdb_id)
        if not os.path.exists(complex_dir):
            raise FileNotFoundError(f'Complex directory not found: {complex_dir}')

        # Read ligand
        lig = read_mol(self.data_dir, pdb_id, suffix=self.ligand_file, remove_hs=False)
        if lig is None:
            raise ValueError(f'Could not read ligand for {pdb_id}')

        if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
            raise ValueError(f'Ligand too large: {lig.GetNumHeavyAtoms()} > {self.max_lig_size}')

        # Find protein file
        protein_path = None
        for suffix in [f'{pdb_id}_{self.protein_file}.pdb', f'{pdb_id}_protein.pdb', f'{pdb_id}.pdb']:
            path = os.path.join(complex_dir, suffix)
            if os.path.exists(path):
                protein_path = path
                break

        if protein_path is None:
            raise FileNotFoundError(f'Protein file not found for {pdb_id}')

        # Get ESM embeddings for this complex
        lm_embeddings = None
        if id_to_embeddings is not None:
            # Try to find embeddings for this PDB ID
            lm_embeddings = id_to_embeddings.get(pdb_id.upper())
            if lm_embeddings is None:
                lm_embeddings = id_to_embeddings.get(pdb_id.lower())

        # Create complex graph
        complex_graph = HeteroData()
        complex_graph['name'] = pdb_id

        # IMPORTANT: Build ligand graph FIRST before receptor
        # This ensures transformation mask is computed on ligand nodes only
        get_lig_graph_with_matching(
            lig,
            complex_graph,
            popsize=self.popsize,
            maxiter=self.maxiter,
            matching=self.matching,
            keep_original=True,
            num_conformers=1,
            remove_hs=self.remove_hs,
        )

        # Load prody data for residue coordinates (needed for ring feature alignment)
        import prody as pr
        pdb = pr.parsePDB(protein_path)
        residue_atom_coords = get_coords(pdb)  # (N_res, 14, 3)

        # Build receptor graph
        moad_extract_receptor_structure(
            path=protein_path,
            complex_graph=complex_graph,
            neighbor_cutoff=self.receptor_radius,
            max_neighbors=self.c_alpha_max_neighbors,
            lm_embeddings=lm_embeddings,
            knn_only_graph=self.knn_only_graph,
            all_atoms=self.all_atoms,
            atom_cutoff=self.atom_radius if self.all_atoms else None,
            atom_max_neighbors=self.atom_max_neighbors if self.all_atoms else None,
        )

        # Center protein and save original center (same as moad.py lines 405-409)
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms and hasattr(complex_graph['atom'], 'pos'):
            complex_graph['atom'].pos -= protein_center
        complex_graph.original_center = protein_center

        # Center ligand positions
        if isinstance(complex_graph['ligand'].pos, list):
            for i in range(len(complex_graph['ligand'].pos)):
                complex_graph['ligand'].pos[i] -= protein_center
        else:
            complex_graph['ligand'].pos -= protein_center

        # Load and align ring features
        ring_path = find_ring_feature_file(pdb_id, self.ring_features_dir)
        if ring_path is not None:
            ring_data = load_ring_features(ring_path)
            ring_vectors = ring_data.get('ring_vectors')

            # Align ring features to residue level
            align_result = align_ring_features_residue(
                ring_data['coords'],
                ring_data['rings'],
                residue_atom_coords,
                match_threshold=self.ring_match_threshold,
                aggregation=self.ring_aggregation,
                ring_vectors=ring_vectors,
            )
            if ring_vectors is not None:
                residue_ring_features, residue_ring_vectors = align_result
                residue_ring_vectors = _fit_ring_vector_dim(residue_ring_vectors, self.ring_vec_dim)
                residue_ring_vectors = _clip_ring_vector_norm(
                    residue_ring_vectors, self.ring_vector_clip_max_norm
                )
                complex_graph['receptor'].ring_vectors = torch.from_numpy(residue_ring_vectors).float()
            else:
                residue_ring_features = align_result
                n_res = complex_graph['receptor'].pos.shape[0]
                complex_graph['receptor'].ring_vectors = torch.zeros(n_res, self.ring_vec_dim, 3)

            residue_ring_features = _fit_ring_scalar_dim(residue_ring_features, self.ring_dim)
            complex_graph['receptor'].ring_features = torch.from_numpy(residue_ring_features).float()

            # If all-atoms mode, also align at atom level
            if self.all_atoms and hasattr(complex_graph, 'atom') and hasattr(complex_graph['atom'], 'pos'):
                atom_coords = complex_graph['atom'].pos.cpu().numpy()
                atom_align_result = align_ring_features_atom(
                    ring_data['coords'],
                    ring_data['rings'],
                    atom_coords,
                    match_threshold=self.ring_match_threshold,
                    ring_vectors=ring_vectors,
                )
                if ring_vectors is not None:
                    atom_ring_features, atom_ring_vectors = atom_align_result
                    atom_ring_vectors = _fit_ring_vector_dim(atom_ring_vectors, self.ring_vec_dim)
                    atom_ring_vectors = _clip_ring_vector_norm(
                        atom_ring_vectors, self.ring_vector_clip_max_norm
                    )
                    complex_graph['atom'].ring_vectors = torch.from_numpy(atom_ring_vectors).float()
                else:
                    atom_ring_features = atom_align_result
                    n_atoms = complex_graph['atom'].pos.shape[0]
                    complex_graph['atom'].ring_vectors = torch.zeros(n_atoms, self.ring_vec_dim, 3)
                atom_ring_features = _fit_ring_scalar_dim(atom_ring_features, self.ring_dim)
                complex_graph['atom'].ring_features = torch.from_numpy(atom_ring_features).float()
        else:
            print(f'Warning: No ring features found for {pdb_id}')
            n_res = complex_graph['receptor'].pos.shape[0]
            complex_graph['receptor'].ring_features = torch.zeros(n_res, self.ring_dim)
            complex_graph['receptor'].ring_vectors = torch.zeros(n_res, self.ring_vec_dim, 3)
            if self.all_atoms and hasattr(complex_graph, 'atom'):
                n_atoms = complex_graph['atom'].pos.shape[0]
                complex_graph['atom'].ring_features = torch.zeros(n_atoms, self.ring_dim)
                complex_graph['atom'].ring_vectors = torch.zeros(n_atoms, self.ring_vec_dim, 3)

        return [complex_graph], [lig]

    def len(self):
        return len(self.complex_graphs)

    def get(self, idx):
        complex_graph = copy.deepcopy(self.complex_graphs[idx])
        complex_graph.mol = RemoveAllHs(copy.deepcopy(self.rdkit_ligands[idx]))

        # Clean up temporary attributes
        for attr in ['random_coords', 'coords', 'seq', 'sequence', 'mask', 'rmsd_matching', 'cluster', 'orig_seq', 'to_keep', 'chain_ids']:
            if hasattr(complex_graph, attr):
                delattr(complex_graph, attr)
            if hasattr(complex_graph['receptor'], attr):
                delattr(complex_graph['receptor'], attr)

        return complex_graph
