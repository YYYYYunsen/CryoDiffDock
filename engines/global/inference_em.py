"""Inference script for DiffDock-EM.

Runs inference with a DiffDock-EM model that incorporates cryo-EM ring features.
Supports both standard finetuned models and LoRA-finetuned models.
"""

import copy
import logging
import os
import random
import warnings
from argparse import ArgumentParser, FileType, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.parse_chi import get_coords
from datasets.process_mols import write_mol_with_coords
from data.ring_features import (
    load_ring_features,
    align_ring_features_residue,
    align_ring_features_atom,
    find_ring_feature_file,
)
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_em_model, load_pretrained_weights
from utils.visualise import PDBFile

# Optional LoRA imports
try:
    from models.lora import apply_lora_to_model, load_lora_state_dict
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

if os.name != 'nt':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_nofile = min(64000, rlimit[1])
    if rlimit[0] < target_nofile:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_nofile, rlimit[1]))

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser(description='DiffDock-EM Inference')

    # Input/output
    parser.add_argument('--protein_ligand_csv', type=str, default=None,
                        help='CSV file with protein_path, ligand_description, [ring_features_path] columns')
    parser.add_argument('--protein_path', type=str, default=None,
                        help='Path to protein PDB file')
    parser.add_argument('--ligand_description', type=str, default=None,
                        help='SMILES string or path to ligand file')
    parser.add_argument('--ring_features_path', type=str, default=None,
                        help='Path to ring features .npz file')
    parser.add_argument('--ring_features_dir', type=str, default=None,
                        help='Directory containing ring feature files (alternative to --ring_features_path)')
    parser.add_argument('--ablate_ring_features', type=str, default='none',
                        choices=['none', 'all', 'scalar', 'vector'],
                        help='Ablation for measuring how much the model relies on the '
                             'density channels: zero all ring inputs, only the l=0 '
                             'scalars, or only the l=1 vectors. Not a normal inference mode.')
    parser.add_argument('--complex_name', type=str, default=None,
                        help='Name for the complex')
    parser.add_argument('--out_dir', type=str, default='results/em_inference',
                        help='Output directory')

    # Model
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing trained EM model')
    parser.add_argument('--ckpt', type=str, default='best_ema_model.pt',
                        help='Checkpoint file name')
    parser.add_argument('--ring_dim', type=int, default=None,
                        help='Ring feature dimension (default: from model config or 4)')
    parser.add_argument('--ring_vec_dim', type=int, default=None,
                        help='Number of l=1 vector ring feature channels / radii count (default: from model config or 6)')
    parser.add_argument('--ring_vec_scale', type=float, default=None,
                        help='Input scale for the l=1 ring embedding. Must match training; '
                             'read from the model config.yml by default.')
    # LoRA options
    parser.add_argument('--use_lora', action='store_true',
                        help='Model was trained with LoRA')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank (must match training)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha (must match training)')
    parser.add_argument('--lora_weights', type=str, default=None,
                        help='Path to LoRA-only weights (alternative to full checkpoint)')

    # Sampling
    parser.add_argument('--samples_per_complex', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for sampling')
    parser.add_argument('--inference_steps', type=int, default=20,
                        help='Number of diffusion steps')
    parser.add_argument('--actual_steps', type=int, default=None,
                        help='Actual number of steps (if different from inference_steps)')
    parser.add_argument('--no_final_step_noise', action='store_true',
                        help='No noise in final step (default: SDE mode with noise in all steps)')

    # Temperature
    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

    # Pretrained model (for restoring batch norm stats with LoRA)
    parser.add_argument('--pretrained_score_model', type=str, default=None,
                        help='Path to pretrained score model checkpoint. Used to restore batch norm '
                             'running stats when using LoRA (auto-detected from config if not specified)')

    # Other
    parser.add_argument('--save_visualisation', action='store_true',
                        help='Save visualization of diffusion steps')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible representative trajectories')
    parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0)
    parser.add_argument('--choose_residue', action='store_true')
    parser.add_argument('--pocket_center', type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help='Local docking anchor in original protein/map coordinates')
    parser.add_argument('--use_density_center', action='store_true',
                        help='Use density_center from ring_features.npz as the local docking anchor')
    parser.add_argument('--local_tr_sigma', type=float, default=None,
                        help='Gaussian translation noise around the local anchor in Angstrom')
    parser.add_argument('--torsion_noise_scale', type=float, default=None,
                        help='Std (rad) of Gaussian torsion noise around input conformer. '
                             'None = full uniform random (global docking). '
                             'Use ~0.5 for local docking with RDKit-generated conformer as start.')
    parser.add_argument('--start_t', type=float, default=1.0,
                        help='Start reverse diffusion from this t value. Use <1 for local refinement, e.g. 0.4')

    return parser


def _freeze_bn(model):
    """Put all BatchNorm layers into eval mode (use frozen running stats).
    Called after model.train() to keep BN frozen while avoiding e3nn eval-mode NaN issues."""
    m = model.module if hasattr(model, 'module') else model
    for module in m.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()


def restore_pretrained_bn_stats(model, pretrained_ckpt_path):
    """Restore batch norm running statistics from pretrained checkpoint.

    LoRA only wraps Linear layers, not BatchNorm. But during finetuning, batch norm
    running_mean and running_var get updated via exponential moving average of batch
    statistics. When the finetuning dataset is small, these stats become noisy and can
    cause NaN outputs for some inputs. Restoring pretrained stats fixes this.
    """
    pretrained_state = torch.load(pretrained_ckpt_path, map_location='cpu')
    model_state = model.state_dict()

    restored = 0
    for key in pretrained_state:
        if 'batch_norm.running_mean' in key or 'batch_norm.running_var' in key or 'batch_norm.num_batches_tracked' in key:
            if key in model_state:
                model_state[key] = pretrained_state[key]
                restored += 1

    model.load_state_dict(model_state)
    logger.info(f"Restored {restored} batch norm running stats from pretrained model")
    return model


def _backfill_pretrained_config(model_config, model_dir):
    """Backfill missing architecture keys from pretrained model_parameters.yml.

    Older finetuned config.yml files may lack keys like embedding_scale (1000)
    and embedding_type (sinusoidal). Without these, get_em_model uses wrong
    defaults (e.g. embedding_scale=10000), causing incorrect timestep embeddings
    and NaN scores during inference.
    """
    arch_keys = [
        'embedding_type', 'embedding_scale',
        'not_fixed_center_conv', 'not_fixed_knn_radius_graph',
        'no_differentiate_convolutions', 'tp_weights_layers',
    ]
    missing = [k for k in arch_keys if k not in model_config]
    if not missing:
        return

    # Find pretrained model_parameters.yml
    pretrained_path = model_config.get('pretrained_model', '')
    if pretrained_path:
        pretrained_dir = os.path.dirname(pretrained_path)
        params_path = os.path.join(pretrained_dir, 'model_parameters.yml')
        if not os.path.exists(params_path):
            # Try relative paths
            for candidate_dir in [
                os.path.join(model_dir, '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model'),
                os.path.join(model_dir, '..', '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model'),
            ]:
                p = os.path.normpath(os.path.join(candidate_dir, 'model_parameters.yml'))
                if os.path.exists(p):
                    params_path = p
                    break
    else:
        params_path = None
        for candidate_dir in [
            os.path.join(model_dir, '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model'),
            os.path.join(model_dir, '..', '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model'),
        ]:
            p = os.path.normpath(os.path.join(candidate_dir, 'model_parameters.yml'))
            if os.path.exists(p):
                params_path = p
                break

    if params_path and os.path.exists(params_path):
        with open(params_path) as f:
            pretrained_config = yaml.safe_load(f)
        filled = []
        for key in missing:
            if key in pretrained_config:
                model_config[key] = pretrained_config[key]
                filled.append(f"{key}={pretrained_config[key]}")
        if filled:
            logger.info(f"Backfilled from pretrained config: {', '.join(filled)}")
    else:
        logger.warning(f"Config missing architecture keys {missing} and pretrained "
                       f"model_parameters.yml not found — model may use wrong defaults")


def _find_pretrained_ckpt(model_config, model_dir):
    """Try to find the pretrained checkpoint path from config or known locations."""
    # Check config for pretrained_model path
    pretrained_path = model_config.get('pretrained_model')
    if pretrained_path and os.path.exists(pretrained_path):
        return pretrained_path

    # Check common relative locations
    candidates = [
        os.path.join(model_dir, '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model', 'best_ema_inference_epoch_model.pt'),
        os.path.join(model_dir, '..', '..', '..', 'DiffDock', 'workdir', 'v1.1', 'score_model', 'best_ema_inference_epoch_model.pt'),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    return None


def _zero_ring_channels(complex_graph, ablate, all_atoms):
    """Zero the ablated ring channels in place after alignment.

    Zeroing reproduces exactly what the model sees for a complex with no ring
    features, so the comparison isolates the contribution of the density
    channels without changing any other input.
    """
    if ablate not in ('scalar', 'vector'):
        return
    node_types = ['receptor'] + (['atom'] if all_atoms else [])
    attribute = 'ring_features' if ablate == 'scalar' else 'ring_vectors'
    for node_type in node_types:
        if node_type in complex_graph.node_types and hasattr(complex_graph[node_type], attribute):
            tensor = complex_graph[node_type][attribute]
            complex_graph[node_type][attribute] = torch.zeros_like(tensor)


def add_ring_features_to_complex(
    complex_graph,
    ring_features_path=None,
    ring_features_dir=None,
    pdb_id=None,
    all_atoms=False,
    match_threshold=0.5,
    ring_dim=4,
    ring_vec_dim=4,
    protein_path=None,
    ablate='none',
):
    """Add ring features to a complex graph.

    Args:
        complex_graph: The complex graph to modify
        ring_features_path: Direct path to .npz file
        ring_features_dir: Directory to search for ring features
        pdb_id: PDB ID for looking up ring features
        all_atoms: Whether model uses all-atoms
        match_threshold: Distance threshold for atom matching
        ring_dim: Number of ring feature channels
        protein_path: Path to protein PDB file (for residue-level alignment)
        ablate: 'none', 'all', 'scalar' (zero l=0), or 'vector' (zero l=1).
            Ablation is for measuring how much the model relies on the density
            channels; it is not a normal inference mode.
    """
    # Find ring features file
    ring_path = ring_features_path
    if ring_path is None and ring_features_dir is not None and pdb_id is not None:
        ring_path = find_ring_feature_file(pdb_id, ring_features_dir)

    if ablate == 'all':
        ring_path = None
        logger.warning("Ring feature ablation: using zeros for all channels")

    if ring_path is None or not os.path.exists(str(ring_path)):
        # No ring features - use zeros
        n_res = complex_graph['receptor'].pos.shape[0]
        complex_graph['receptor'].ring_features = torch.zeros(n_res, ring_dim)
        complex_graph['receptor'].ring_vectors = torch.zeros(n_res, ring_vec_dim, 3)
        if all_atoms and 'atom' in complex_graph.node_types:
            n_atoms = complex_graph['atom'].pos.shape[0]
            complex_graph['atom'].ring_features = torch.zeros(n_atoms, ring_dim)
            complex_graph['atom'].ring_vectors = torch.zeros(n_atoms, ring_vec_dim, 3)
        logger.warning(f"No ring features found, using zeros")
        return

    # Load ring features
    ring_data = load_ring_features(ring_path)
    ring_vectors = ring_data.get('ring_vectors')
    logger.info(f"Loaded ring features from {ring_path}: {ring_data['rings'].shape}")
    if ring_data['rings'].shape[1] != ring_dim:
        logger.info(
            "Ring scalar dim mismatch: data=%d, target=%d (will pad/truncate as needed)",
            ring_data['rings'].shape[1], ring_dim
        )
    if ring_vectors is not None and ring_vectors.shape[1] != ring_vec_dim:
        logger.info(
            "Ring vector dim mismatch: data=%d, target=%d (will pad/truncate as needed)",
            ring_vectors.shape[1], ring_vec_dim
        )

    # Use the same residue-level alignment as training (prody + get_coords)
    n_res = complex_graph['receptor'].pos.shape[0]
    base_scalar_dim = min(ring_dim, ring_data['rings'].shape[1])
    actual_vec_dim = min(ring_vec_dim, ring_vectors.shape[1]) if ring_vectors is not None else 0
    residue_ring_features = np.zeros((n_res, ring_dim), dtype=np.float32)
    residue_ring_vectors = np.zeros((n_res, ring_vec_dim, 3), dtype=np.float32)

    residue_atom_coords = None
    if protein_path is not None and os.path.exists(protein_path):
        import prody
        prody.confProDy(verbosity='none')
        structure = prody.parsePDB(protein_path)
        residue_atom_coords = get_coords(structure)  # (N_res, 14, 3)

        ring_feats_slice = ring_data['rings'][:, :base_scalar_dim]
        ring_vecs_slice = ring_vectors[:, :actual_vec_dim] if (ring_vectors is not None and actual_vec_dim > 0) else None

        result = align_ring_features_residue(
            ring_data['coords'], ring_feats_slice,
            residue_atom_coords, match_threshold=match_threshold,
            ring_vectors=ring_vecs_slice,
        )
        if ring_vecs_slice is not None:
            result_feats, result_vecs = result
            residue_ring_features[:, :base_scalar_dim] = result_feats[:n_res]
            residue_ring_vectors[:, :actual_vec_dim] = result_vecs[:n_res]
        else:
            residue_ring_features[:, :base_scalar_dim] = result[:n_res]
    else:
        logger.warning("No protein_path provided, falling back to Cα ball query for ring alignment")
        from scipy.spatial import cKDTree
        receptor_pos = complex_graph['receptor'].pos.cpu().numpy()
        tree = cKDTree(ring_data['coords'])
        for i, ca_pos in enumerate(receptor_pos):
            indices = tree.query_ball_point(ca_pos, r=4.0)
            if len(indices) > 0:
                residue_ring_features[i, :base_scalar_dim] = ring_data['rings'][indices, :base_scalar_dim].mean(axis=0)
                if ring_vectors is not None and actual_vec_dim > 0:
                    residue_ring_vectors[i, :actual_vec_dim] = ring_vectors[indices, :actual_vec_dim].mean(axis=0)

    complex_graph['receptor'].ring_features = torch.from_numpy(residue_ring_features).float()
    complex_graph['receptor'].ring_vectors = torch.from_numpy(residue_ring_vectors).float()

    # Atom-level features for AAModel
    if all_atoms and 'atom' in complex_graph.node_types:
        atom_pos = complex_graph['atom'].pos.cpu().numpy()
        result = align_ring_features_atom(
            ring_data['coords'],
            ring_data['rings'][:, :base_scalar_dim],
            atom_pos,
            match_threshold=match_threshold,
            ring_vectors=ring_vectors[:, :actual_vec_dim] if (ring_vectors is not None and actual_vec_dim > 0) else None,
        )
        atom_ring_features = np.zeros((len(atom_pos), ring_dim), dtype=np.float32)
        atom_ring_vectors = np.zeros((len(atom_pos), ring_vec_dim, 3), dtype=np.float32)
        if ring_vectors is not None and actual_vec_dim > 0:
            atom_feats, atom_vecs = result
            atom_ring_features[:, :base_scalar_dim] = atom_feats
            atom_ring_vectors[:, :actual_vec_dim] = atom_vecs
        else:
            atom_ring_features[:, :base_scalar_dim] = result
        complex_graph['atom'].ring_features = torch.from_numpy(atom_ring_features).float()
        complex_graph['atom'].ring_vectors = torch.from_numpy(atom_ring_vectors).float()

    _zero_ring_channels(complex_graph, ablate, all_atoms)


def get_ring_density_center(ring_features_path):
    """Read density_center from a ring features npz file.

    NOTE: The current npz format (coords, rings, ring_vectors, radii, atomZ, pdb_id)
    does not include a density_center field. --use_density_center requires npz files
    produced with a density_center key (e.g. centroid of high-density voxels).
    If the key is absent this function returns None and inference_em.py will raise
    a clear ValueError so the user knows the feature is not yet available.
    """
    if ring_features_path is None or not os.path.exists(str(ring_features_path)):
        return None
    ring_data = load_ring_features(ring_features_path)
    density_center = ring_data.get('density_center')
    if density_center is None:
        return None
    return np.asarray(density_center, dtype=np.float32)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Using random seed: {args.seed}")
    if not (0.0 < args.start_t <= 1.0):
        raise ValueError(f"--start_t must be in (0, 1], got {args.start_t}")
    if args.pocket_center is not None and args.use_density_center:
        raise ValueError("Use either --pocket_center or --use_density_center, not both")
    if args.ablate_ring_features != 'none' and args.use_density_center:
        raise ValueError(
            "--use_density_center reads the anchor from the ring feature npz, so it "
            "would keep leaking density information into an ablated run"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load model config
    config_path = os.path.join(args.model_dir, 'config.yml')
    model_params_path = os.path.join(args.model_dir, 'model_parameters.yml')

    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = yaml.safe_load(f)
    elif os.path.exists(model_params_path):
        with open(model_params_path) as f:
            model_config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"No config found in {args.model_dir}")

    # Backfill missing architecture keys from pretrained model config.
    # Older finetuned config.yml files may be missing keys like embedding_scale,
    # embedding_type, not_fixed_center_conv that are needed for correct model
    # construction (e.g. embedding_scale defaults to 10000 but pretrained uses 1000).
    _backfill_pretrained_config(model_config, args.model_dir)

    model_args = Namespace(**model_config)

    # Check if model was trained with LoRA (from config)
    use_lora = args.use_lora or model_config.get('use_lora', False)
    lora_r = args.lora_r if args.lora_r != 8 else model_config.get('lora_r', 8)
    lora_alpha = args.lora_alpha if args.lora_alpha != 16 else model_config.get('lora_alpha', 16)

    if use_lora and not LORA_AVAILABLE:
        raise ImportError("LoRA module not available. Make sure models/lora.py exists.")

    # Parse input
    if args.protein_ligand_csv is not None:
        df = pd.read_csv(args.protein_ligand_csv)
        complex_name_list = set_nones(df['complex_name'].tolist()) if 'complex_name' in df else [None] * len(df)
        protein_path_list = set_nones(df['protein_path'].tolist())
        ligand_description_list = set_nones(df['ligand_description'].tolist())
        ring_features_list = set_nones(df['ring_features_path'].tolist()) if 'ring_features_path' in df else [None] * len(df)
    else:
        complex_name_list = [args.complex_name or 'complex_0']
        protein_path_list = [args.protein_path]
        ligand_description_list = [args.ligand_description]
        ring_features_list = [args.ring_features_path]

    complex_name_list = [name if name else f"complex_{i}" for i, name in enumerate(complex_name_list)]

    # Create output directories
    for name in complex_name_list:
        os.makedirs(os.path.join(args.out_dir, name), exist_ok=True)

    # Check if model uses ESM embeddings based on training config
    # If the finetuned model was trained without ESM, we must also skip ESM during inference
    use_lm_embeddings = (
        hasattr(model_args, 'esm_embeddings_path') and model_args.esm_embeddings_path is not None
    ) or (
        hasattr(model_args, 'esm_embeddings_model') and model_args.esm_embeddings_model is not None
    ) or (
        hasattr(model_args, 'pdbbind_esm_embeddings_path') and model_args.pdbbind_esm_embeddings_path is not None
    ) or (
        hasattr(model_args, 'moad_esm_embeddings_path') and model_args.moad_esm_embeddings_path is not None
    )

    if use_lm_embeddings:
        # Set the embeddings path to trigger 'precomputed' mode in get_em_model
        model_args.esm_embeddings_path = "precomputed_by_inference"
        logger.info("Using ESM embeddings (model was trained with them)")
    else:
        logger.info("Skipping ESM embeddings (model was trained without them)")

    # Create dataset
    logger.info("Preprocessing complexes...")
    test_dataset = InferenceDataset(
        out_dir=args.out_dir,
        complex_names=complex_name_list,
        protein_files=protein_path_list,
        ligand_descriptions=ligand_description_list,
        protein_sequences=[None] * len(protein_path_list),
        lm_embeddings=use_lm_embeddings,
        receptor_radius=model_args.receptor_radius if hasattr(model_args, 'receptor_radius') else 30,
        remove_hs=model_args.remove_hs if hasattr(model_args, 'remove_hs') else False,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors if hasattr(model_args, 'c_alpha_max_neighbors') else None,
        all_atoms=model_args.all_atoms if hasattr(model_args, 'all_atoms') else False,
        atom_radius=model_args.atom_radius if hasattr(model_args, 'atom_radius') else 5,
        atom_max_neighbors=model_args.atom_max_neighbors if hasattr(model_args, 'atom_max_neighbors') else None,
        knn_only_graph=True,
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Create model

    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    ring_dim = args.ring_dim if args.ring_dim is not None else model_config.get('ring_dim', 4)
    ring_vec_dim = args.ring_vec_dim if args.ring_vec_dim is not None else model_config.get('ring_vec_dim', 6)
    # Must match training: models trained before --ring_vec_scale existed default to 1.0.
    ring_vec_scale = args.ring_vec_scale if args.ring_vec_scale is not None else model_config.get('ring_vec_scale', 1.0)
    if ring_vec_scale != 1.0:
        logger.info(f"Using ring_vec_scale={ring_vec_scale} for the l=1 ring embedding")
    model = get_em_model(model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, ring_dim=ring_dim,
                         ring_vec_dim=ring_vec_dim, ring_vec_scale=ring_vec_scale)

    # Apply LoRA if needed (before loading weights)
    if use_lora:
        logger.info(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
        model, _, _ = apply_lora_to_model(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=0.0,  # No dropout during inference
            verbose=False,
        )

    # Load weights
    ckpt_path = os.path.join(args.model_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        alt = os.path.join(args.model_dir, 'best_ema_inference_epoch_model.pt')
        if os.path.exists(alt):
            logger.info(f"Checkpoint {args.ckpt} not found, using {alt}")
            ckpt_path = alt
    logger.info(f"Loading model from {ckpt_path}")

    if use_lora and args.lora_weights:
        # Load base model weights first, then LoRA weights separately
        base_ckpt = os.path.join(args.model_dir, 'pretrained_model.pt')
        if os.path.exists(base_ckpt):
            load_pretrained_weights(model, base_ckpt, strict=False, verbose=False)
        load_lora_state_dict(model, torch.load(args.lora_weights, map_location='cpu'))
        logger.info(f"Loaded LoRA weights from {args.lora_weights}")
    else:
        # Load full checkpoint (includes LoRA weights if applicable)
        load_pretrained_weights(model, ckpt_path, strict=False, verbose=True)

    # Restore pretrained batch norm running stats when using LoRA.
    # LoRA doesn't modify BatchNorm layers, but finetuning on a small dataset
    # shifts running_mean/running_var, causing NaN for some inputs.
    if use_lora:
        pretrained_ckpt = args.pretrained_score_model or _find_pretrained_ckpt(model_config, args.model_dir)
        if pretrained_ckpt:
            restore_pretrained_bn_stats(model, pretrained_ckpt)
        else:
            logger.warning("Could not find pretrained checkpoint to restore batch norm stats. "
                           "Use --pretrained_score_model to specify it.")

    model = model.to(device)

    if use_lora:
        from models.lora import merge_lora_weights
        logger.info("Merging LoRA weights for inference...")
        model = merge_lora_weights(model)

    # Use train mode + frozen BN instead of model.eval(): e3nn/LoRA fine-tuned models
    # can produce all-NaN outputs in eval mode. BN is frozen so running stats are unchanged;
    # backbone dropout=0 so train/eval makes no practical difference for predictions.
    model.train()
    _freeze_bn(model)

    # Sampling schedule
    tr_schedule = get_t_schedule(
        inference_steps=args.inference_steps,
        sigma_schedule='expbeta',
        t_max=args.start_t,
    )

    # Run inference
    N = args.samples_per_complex
    failures, skipped = 0, 0

    logger.info(f"Running inference on {len(test_dataset)} complexes")
    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        if not orig_complex_graph.success[0]:
            skipped += 1
            logger.warning(f"Skipping {complex_name_list[idx]} - preprocessing failed")
            continue

        try:
            # Add ring features to original graph once, then deepcopy
            ring_path = ring_features_list[idx] if idx < len(ring_features_list) else None
            pdb_id = complex_name_list[idx]
            protein_path = protein_path_list[idx]

            add_ring_features_to_complex(
                orig_complex_graph,
                ring_features_path=ring_path,
                ring_features_dir=args.ring_features_dir,
                pdb_id=pdb_id,
                all_atoms=model_args.all_atoms if hasattr(model_args, 'all_atoms') else False,
                ring_dim=ring_dim,
                ring_vec_dim=ring_vec_dim,
                protein_path=protein_path,
                ablate=args.ablate_ring_features,
            )

            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            local_center = None
            if args.pocket_center is not None:
                local_center = np.asarray(args.pocket_center, dtype=np.float32)
            elif args.use_density_center:
                local_center = get_ring_density_center(ring_path)
                if local_center is None:
                    raise ValueError(
                        f"--use_density_center was requested, but no density_center was found in {ring_path}"
                    )

            if local_center is not None:
                local_center = (
                    torch.from_numpy(local_center).float().view(3)
                    - orig_complex_graph.original_center.cpu().view(3)
                )
                logger.info(
                    "Local docking anchor for %s: centered coords=(%.3f, %.3f, %.3f), start_t=%.3f, local_tr_sigma=%s",
                    complex_name_list[idx],
                    float(local_center[0]),
                    float(local_center[1]),
                    float(local_center[2]),
                    args.start_t,
                    "default" if args.local_tr_sigma is None else f"{args.local_tr_sigma:.3f}",
                )

            # Randomize initial positions
            randomize_position(
                data_list,
                model_args.no_torsion if hasattr(model_args, 'no_torsion') else False,
                False,
                model_args.tr_sigma_max if hasattr(model_args, 'tr_sigma_max') else 19.0,
                initial_noise_std_proportion=args.initial_noise_std_proportion,
                choose_residue=args.choose_residue,
                pocket_center=local_center,
                local_tr_sigma=args.local_tr_sigma,
                torsion_noise_scale=args.torsion_noise_scale,
            )

            lig = orig_complex_graph.mol[0]

            # Run sampling
            visualization_list = None
            if args.save_visualisation:
                visualization_list = []
                for graph in data_list:
                    trajectory_pdb = PDBFile(lig)
                    trajectory_pdb.add(
                        (graph['ligand'].pos + graph.original_center).detach().cpu(),
                        part=1,
                        order=1,
                    )
                    visualization_list.append(trajectory_pdb)

            data_list, _ = sampling(
                data_list=data_list,
                model=model,
                inference_steps=args.actual_steps or args.inference_steps,
                tr_schedule=tr_schedule,
                rot_schedule=tr_schedule,
                tor_schedule=tr_schedule,
                device=device,
                t_to_sigma=t_to_sigma,
                model_args=model_args,
                no_final_step_noise=args.no_final_step_noise,
                ode=False,
                visualization_list=visualization_list,
                batch_size=args.batch_size,
                temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor],
            )

            # Save results
            write_dir = os.path.join(args.out_dir, complex_name_list[idx])

            for rank, data in enumerate(data_list):
                mol_pred = copy.deepcopy(lig)
                if model_args.remove_hs if hasattr(model_args, 'remove_hs') else False:
                    mol_pred = RemoveAllHs(mol_pred)

                ligand_pos = data['ligand'].pos.cpu().numpy()
                if hasattr(data, 'original_center'):
                    ligand_pos += data.original_center.cpu().numpy()

                if mol_pred.GetNumAtoms() != ligand_pos.shape[0]:
                    raise ValueError(
                        f"Output atom count mismatch for {complex_name_list[idx]} "
                        f"rank {rank + 1}: molecule has {mol_pred.GetNumAtoms()} "
                        f"atoms but sampled coordinates have {ligand_pos.shape[0]}"
                    )

                write_mol_with_coords(
                    mol_pred,
                    ligand_pos,
                    os.path.join(write_dir, f'rank{rank+1}.sdf'),
                )

            if args.save_visualisation:
                for rank, trajectory_pdb in enumerate(visualization_list):
                    trajectory_pdb.write(
                        os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb')
                    )
                logger.info(
                    f"Saved {len(visualization_list)} diffusion trajectories "
                    f"with {args.actual_steps or args.inference_steps} denoising steps each"
                )

            logger.info(f"Saved {N} samples for {complex_name_list[idx]}")

        except Exception as e:
            failures += 1
            logger.error(f"Failed on {complex_name_list[idx]}: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info(f"Completed: {len(test_dataset) - failures - skipped} successful, {failures} failed, {skipped} skipped")


if __name__ == '__main__':
    main()
