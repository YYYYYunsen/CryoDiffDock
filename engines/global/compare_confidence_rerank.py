#!/usr/bin/env python3
"""Rerank pretrained/ablation/finetuned poses with confidence models and compare them."""

import copy
import csv
import logging
import os
import re
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.process_mols import write_mol_with_coords
from evaluate_em import compute_rmsds, load_test_ids
from inference_em import add_ring_features_to_complex, _backfill_pretrained_config
from models.lora import apply_lora_to_model, merge_lora_weights
try:
    from models.old_aa_model_ligand_density import AAOldLigandDensityModel
except ImportError:
    # The paper's primary path uses the original DiffDock confidence model. The
    # experimental ligand-density confidence architecture was never part of the
    # frozen release source; fail only if that optional branch is requested.
    AAOldLigandDensityModel = None
from utils.diffusion_utils import get_timestep_embedding
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.ligand_density_features import (
    atom_feature_dim,
    compute_pose_density_features,
    get_grid,
    get_grid_normalization_stats,
    load_map_mapping,
    summary_feature_dim,
)
from utils.inference_utils import InferenceDataset, set_nones
from utils.utils import get_em_model, get_model, load_pretrained_weights


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL_KINDS = ('orig', 'em', 'ligdens')
MODEL_PRETTY_NAMES = {
    'orig': 'Original confidence',
    'em': 'Cryo-DiffDock confidence',
    'ligdens': 'Ligand-density confidence',
}


def _freeze_bn(model):
    """Put all BatchNorm layers into eval mode (use frozen running stats)."""
    m = model.module if hasattr(model, 'module') else model
    for module in m.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()


def get_parser():
    parser = ArgumentParser(description='Compare confidence-model reranking variants')
    parser.add_argument('--protein_ligand_csv', required=True)
    parser.add_argument('--test_split', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--map_csv', default=None,
                        help='04_maps-style CSV required for ligand-density confidence scoring.')
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--pretrained_poses', required=True)
    parser.add_argument('--ablation_poses', required=True)
    parser.add_argument('--finetuned_poses', required=True)
    parser.add_argument('--orig_conf_dir', required=True)
    parser.add_argument('--orig_conf_ckpt', default='best_model.pt')
    parser.add_argument('--em_conf_dir', required=True)
    parser.add_argument('--em_conf_ckpt', default='best_model.pt')
    parser.add_argument('--ligdens_conf_dir', default=None)
    parser.add_argument('--ligdens_conf_ckpt', default='best_model.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_poses', type=int, default=None)
    # Parallel execution: restrict to a single (pose_set, conf_model) task per process
    parser.add_argument('--pose_set', choices=['pretrained', 'ablation', 'finetuned'], default=None,
                        help='Single pose set to score. If omitted, all three are processed.')
    parser.add_argument('--conf_model', choices=['orig', 'em', 'ligdens'], default=None,
                        help='Single confidence model to use. If omitted, all configured models are used.')
    parser.add_argument('--report_only', action='store_true',
                        help='Skip scoring; read existing reranked poses and write summary reports.')
    parser.add_argument('--native_comparison', action='store_true',
                        help='With --report_only, compare pretrained poses (already natively '
                             'confidence-ranked) against ablation_orig_conf and finetuned_orig_conf.')
    parser.add_argument('--esm_embeddings_path', default=None,
                        help='Precomputed ESM2 embeddings .pt file for the EM confidence model.')
    return parser


def patch_original_conf_args(model_args):
    defaults = {
        'use_original_model_cache': True,
        'esm_embeddings_path': None,
        'num_classification_bins': 2,
        'not_fixed_knn_radius_graph': True,
        'not_knn_only_graph': True,
        'shell_radii': [0.8, 1.6, 2.4],
        'use_density_summary': False,
        'use_lora': False,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(model_args, key):
            setattr(model_args, key, value)
    if not hasattr(model_args, 'confidence_dropout'):
        model_args.confidence_dropout = getattr(model_args, 'dropout', 0.0)
    if not hasattr(model_args, 'confidence_no_batchnorm'):
        model_args.confidence_no_batchnorm = False
    return model_args


def load_model_args(model_dir, kind):
    config_path = os.path.join(model_dir, 'config.yml')
    params_path = os.path.join(model_dir, 'model_parameters.yml')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif os.path.exists(params_path):
        with open(params_path) as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f'No config.yml or model_parameters.yml found in {model_dir}')

    if kind == 'em':
        # Older finetuned configs may miss a few architecture keys.
        # Fill the commonly expected defaults directly so rerank evaluation
        # does not depend on a pretrained config file being present.
        config.setdefault('embedding_type', 'sinusoidal')
        config.setdefault('embedding_scale', 1000)
        config.setdefault('not_fixed_center_conv', False)
        config.setdefault('not_fixed_knn_radius_graph', True)
        config.setdefault('no_differentiate_convolutions', False)
        config.setdefault('tp_weights_layers', 2)
        _backfill_pretrained_config(config, model_dir)
        return Namespace(**config)
    if kind == 'ligdens':
        return patch_original_conf_args(Namespace(**config))
    return patch_original_conf_args(Namespace(**config))


def infer_use_lm_embeddings(model_args):
    return any(
        getattr(model_args, attr, None) is not None
        for attr in [
            'esm_embeddings_path',
            'esm_embeddings_model',
            'pdbbind_esm_embeddings_path',
            'moad_esm_embeddings_path',
        ]
    )


def build_dataset(df, model_args, out_dir, esm_embeddings_path=None):
    complex_names = set_nones(df['complex_name'].tolist()) if 'complex_name' in df else [None] * len(df)
    protein_paths = set_nones(df['protein_path'].tolist())
    ligand_descriptions = set_nones(df['ligand_description'].tolist())
    ring_paths = set_nones(df['ring_features_path'].tolist()) if 'ring_features_path' in df else [None] * len(df)
    complex_names = [name if name else f'complex_{i}' for i, name in enumerate(complex_names)]

    # Use LM embeddings if the model config requests them OR if a path was explicitly provided.
    # Note: patch_original_conf_args sets esm_embeddings_path=None in model_args, so we must
    # also check the externally supplied path to avoid missing embeddings for the original model.
    use_lm_embeddings = infer_use_lm_embeddings(model_args) or (esm_embeddings_path is not None)
    precomputed_lm_embeddings = None

    if use_lm_embeddings:
        if esm_embeddings_path is not None:
            logger.info(f'Loading precomputed ESM embeddings from {esm_embeddings_path}')
            raw = torch.load(esm_embeddings_path, map_location='cpu')
            # Group chains by PDB ID (keys are "{PDB_ID}_chain_{idx}" or just "{PDB_ID}")
            id_to_embeddings = {}
            for key, emb in raw.items():
                pdb_id = key.split('_chain_')[0].upper() if '_chain_' in key else key.upper()
                id_to_embeddings.setdefault(pdb_id, []).append(emb)
            precomputed_lm_embeddings = [
                id_to_embeddings.get(name.upper()) for name in complex_names
            ]
            n_missing = sum(1 for e in precomputed_lm_embeddings if e is None)
            if n_missing:
                logger.warning(
                    f'{n_missing}/{len(complex_names)} complexes not found in ESM embeddings file; '
                    'those complexes will be skipped during scoring.'
                )
            # InferenceDataset triggers on-the-fly ESM computation if the first element is None
            # (and protein_sequences are all None, which would crash). Fall back safely.
            if precomputed_lm_embeddings[0] is None:
                logger.warning(
                    f'First complex "{complex_names[0]}" not found in ESM embeddings file. '
                    'Disabling LM embeddings for this dataset build — check PDB ID format in the .pt file.'
                )
                use_lm_embeddings = False
                precomputed_lm_embeddings = None
        else:
            logger.info('Computing ESM embeddings from the input receptor structures.')

    dataset = InferenceDataset(
        out_dir=out_dir,
        complex_names=complex_names,
        protein_files=protein_paths,
        ligand_descriptions=ligand_descriptions,
        protein_sequences=[None] * len(protein_paths),
        lm_embeddings=use_lm_embeddings,
        precomputed_lm_embeddings=precomputed_lm_embeddings,
        receptor_radius=model_args.receptor_radius,
        remove_hs=model_args.remove_hs,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        all_atoms=model_args.all_atoms,
        atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        knn_only_graph=not getattr(model_args, 'not_knn_only_graph', False),
    )
    return dataset, complex_names, protein_paths, ring_paths


def build_model(model_args, kind, model_dir, ckpt, device):
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    if kind == 'em':
        model = get_em_model(
            model_args,
            device,
            t_to_sigma=t_to_sigma,
            no_parallel=True,
            confidence_mode=True,
            ring_dim=getattr(model_args, 'ring_dim', 4),
            ring_vec_dim=getattr(model_args, 'ring_vec_dim', 6),
        )
        # If the checkpoint was saved from a LoRA-trained model, apply LoRA
        # before loading so that keys match (original_layer.*, lora_A, lora_B).
        if getattr(model_args, 'use_lora', False):
            lora_r = getattr(model_args, 'lora_r', 8)
            lora_alpha = getattr(model_args, 'lora_alpha', 16)
            lora_dropout = getattr(model_args, 'lora_dropout', 0.0)
            logger.info(f'Applying LoRA (r={lora_r}, alpha={lora_alpha}) before loading checkpoint')
            # Move to CPU first so LoRA tensors and model weights are on the same device
            model = model.cpu()
            model, n_layers, n_params = apply_lora_to_model(
                model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, verbose=False)
            logger.info(f'  {n_layers} LoRA layers ({n_params:,} params)')
            load_pretrained_weights(model, os.path.join(model_dir, ckpt), strict=False, verbose=True)
            # Merge LoRA into base weights so inference runs without LoRA wrapper overhead
            logger.info('Merging LoRA weights into base layers for inference')
            model = merge_lora_weights(model)
        else:
            load_pretrained_weights(model, os.path.join(model_dir, ckpt), strict=False, verbose=True)
    elif kind == 'ligdens':
        timestep_emb_func = get_timestep_embedding(
            getattr(model_args, 'embedding_type', 'sinusoidal'),
            model_args.sigma_embed_dim,
            getattr(model_args, 'embedding_scale', 10000),
        )
        if AAOldLigandDensityModel is None:
            raise ImportError(
                "The optional ligand-density confidence architecture is unavailable; "
                "use the frozen original-confidence path (conf_model=orig)."
            )
        model = AAOldLigandDensityModel(
            t_to_sigma=t_to_sigma,
            device=device,
            timestep_emb_func=timestep_emb_func,
            lm_embedding_type='esm' if getattr(model_args, 'esm_embeddings_path', None) is not None else None,
            no_torsion=getattr(model_args, 'no_torsion', False),
            num_conv_layers=model_args.num_conv_layers,
            lig_max_radius=model_args.max_radius,
            scale_by_sigma=model_args.scale_by_sigma,
            norm_by_sigma=getattr(model_args, 'norm_by_sigma', True),
            sigma_embed_dim=model_args.sigma_embed_dim,
            ns=model_args.ns,
            nv=model_args.nv,
            distance_embed_dim=model_args.distance_embed_dim,
            cross_distance_embed_dim=model_args.cross_distance_embed_dim,
            batch_norm=not getattr(model_args, 'no_batch_norm', False),
            dropout=getattr(model_args, 'dropout', 0.0),
            use_second_order_repr=getattr(model_args, 'use_second_order_repr', False),
            cross_max_distance=model_args.cross_max_distance,
            dynamic_max_cross=getattr(model_args, 'dynamic_max_cross', False),
            confidence_mode=True,
            confidence_dropout=getattr(model_args, 'confidence_dropout', getattr(model_args, 'dropout', 0.0)),
            confidence_no_batchnorm=getattr(model_args, 'confidence_no_batchnorm', False),
            num_confidence_outputs=getattr(model_args, 'num_classification_bins', 2) - 1,
            no_aminoacid_identities=False,
            use_old_atom_encoder=getattr(model_args, 'use_old_atom_encoder', True),
            ligand_density_dim=atom_feature_dim(getattr(model_args, 'shell_radii', [0.8, 1.6, 2.4])),
            density_summary_dim=summary_feature_dim() if getattr(model_args, 'use_density_summary', False) else 0,
        )
        if getattr(model_args, 'use_lora', False):
            lora_r = getattr(model_args, 'lora_r', 8)
            lora_alpha = getattr(model_args, 'lora_alpha', 16)
            lora_dropout = getattr(model_args, 'lora_dropout', 0.0)
            logger.info(f'Applying LoRA (r={lora_r}, alpha={lora_alpha}) before loading checkpoint')
            model = model.cpu()
            model, n_layers, n_params = apply_lora_to_model(
                model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, verbose=False)
            logger.info(f'  {n_layers} LoRA layers ({n_params:,} params)')
            load_pretrained_weights(model, os.path.join(model_dir, ckpt), strict=False, verbose=True)
            logger.info('Merging LoRA weights into base layers for inference')
            model = merge_lora_weights(model)
        else:
            load_pretrained_weights(model, os.path.join(model_dir, ckpt), strict=False, verbose=True)
    else:
        model = get_model(
            model_args,
            device,
            t_to_sigma=t_to_sigma,
            no_parallel=True,
            confidence_mode=True,
            old=True,
        )
        load_pretrained_weights(model, os.path.join(model_dir, ckpt), strict=False, verbose=True)
    model = model.to(device)
    # Initial state: train mode so e3nn equivariant layers don't produce NaN.
    # BN calibration (called separately after build_model) will accumulate proper
    # running stats and then switch the model to eval mode.
    model.train()
    if kind == 'em':
        # Sanity-check: report any NaN weights so we can pinpoint the source if scores are NaN.
        nan_params = [(n, p) for n, p in model.named_parameters() if torch.isnan(p).any()]
        if nan_params:
            logger.warning(f'WARNING: {len(nan_params)} parameters still contain NaN after loading:')
            for pn, pv in nan_params[:10]:
                n_nan = int(torch.isnan(pv).sum())
                logger.warning(f'  {pn}: {n_nan}/{pv.numel()} NaN values')
        else:
            logger.info('NaN weight check passed: no NaN parameters detected')
    return model


def zero_time_fields(graph):
    zeros = lambda n: torch.zeros(n)
    for node_type in ['ligand', 'receptor', 'atom', 'misc_atom']:
        if node_type in graph.node_types:
            graph[node_type].node_t = {k: zeros(graph[node_type].num_nodes) for k in ('tr', 'rot', 'tor')}
    graph.complex_t = {k: zeros(1) for k in ('tr', 'rot', 'tor')}
    return graph


def find_pose_files(complex_dir, max_poses=None):
    seen_ranks = {}  # rank -> path; prefer plain rankN.sdf over rankN_confidence*.sdf
    for path in complex_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != '.sdf':
            continue
        match = re.match(r'rank(\d+)(?:_confidence[-\d.]+)?\.sdf$', path.name)
        if not match:
            continue
        rank = int(match.group(1))
        has_conf_in_name = '_confidence' in path.name
        if rank not in seen_ranks or (not has_conf_in_name):
            seen_ranks[rank] = path
    pose_files = sorted(seen_ranks.items())  # list of (rank, path)
    if max_poses is not None:
        pose_files = pose_files[:max_poses]
    return pose_files


def load_pose(path, remove_hs):
    mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError(f'Failed to read pose from {path}')
    if remove_hs:
        mol = RemoveAllHs(mol)
    return mol, mol.GetConformer().GetPositions()


def attach_ligand_density_features(graph, pose_coords_abs, map_path, shell_radii, use_density_summary):
    atom_features, summary = compute_pose_density_features(
        get_grid(map_path),
        np.asarray(pose_coords_abs, dtype=np.float32),
        shell_radii=shell_radii,
        normalization_stats=get_grid_normalization_stats(map_path),
    )
    if atom_features.shape[0] != graph['ligand'].num_nodes:
        raise ValueError(
            f'Density feature atom count mismatch: {atom_features.shape[0]} vs graph {graph["ligand"].num_nodes}'
        )
    graph['ligand'].density_features = torch.from_numpy(atom_features).float()
    if use_density_summary:
        graph.density_summary = torch.from_numpy(summary).float().unsqueeze(0)


def score_and_write_pose_set(
    pose_set_name,
    pose_dir,
    out_dir,
    dataset,
    complex_names,
    protein_paths,
    ring_paths,
    model_args,
    model,
    kind,
    batch_size,
    max_poses,
    data_dir,
    map_mapping=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = skipped = failed = 0

    for idx, complex_name in enumerate(tqdm(complex_names, desc=f'Rerank {pose_set_name} with {kind}')):
        complex_pose_dir = Path(pose_dir) / complex_name
        if not complex_pose_dir.is_dir():
            skipped += 1
            continue
        pose_files = find_pose_files(complex_pose_dir, max_poses=max_poses)
        if not pose_files:
            skipped += 1
            continue

        try:
            base_graph = dataset[idx]
            if not getattr(base_graph, 'success', False):
                skipped += 1
                continue

            if kind == 'em':
                add_ring_features_to_complex(
                    base_graph,
                    ring_features_path=ring_paths[idx],
                    ring_features_dir=data_dir,
                    pdb_id=complex_name,
                    all_atoms=model_args.all_atoms,
                    ring_dim=getattr(model_args, 'ring_dim', 4),
                    ring_vec_dim=getattr(model_args, 'ring_vec_dim', 6),
                    protein_path=protein_paths[idx],
                )
            elif kind == 'ligdens':
                if map_mapping is None:
                    raise ValueError('map_mapping is required for ligand-density confidence scoring')
                if complex_name.upper() not in map_mapping:
                    raise KeyError(f'No EM map found for {complex_name} in map CSV')
                map_path = map_mapping[complex_name.upper()]

            batch_graphs = []
            records = []
            center = base_graph.original_center.cpu().numpy()
            for original_rank, pose_path in pose_files:
                pose_mol, pose_coords = load_pose(pose_path, model_args.remove_hs)
                if pose_coords.shape[0] != base_graph['ligand'].pos.shape[0]:
                    raise ValueError(
                        f'Atom count mismatch for {pose_path}: pose has {pose_coords.shape[0]}, '
                        f'graph has {base_graph["ligand"].pos.shape[0]}'
                    )
                # Shallow-copy the graph and only clone the tensors we mutate (ligand pos + time
                # fields). This avoids a full deepcopy of the large receptor sub-graph, which is
                # the main CPU bottleneck when processing 40 poses per complex.
                graph = copy.copy(base_graph)
                graph._node_store_dict = dict(base_graph._node_store_dict)
                lig_store = copy.copy(base_graph['ligand'])
                lig_store.pos = torch.from_numpy(pose_coords - center).float()
                graph._node_store_dict['ligand'] = lig_store
                if kind == 'ligdens':
                    attach_ligand_density_features(
                        graph,
                        pose_coords_abs=pose_coords,
                        map_path=map_path,
                        shell_radii=getattr(model_args, 'shell_radii', [0.8, 1.6, 2.4]),
                        use_density_summary=getattr(model_args, 'use_density_summary', False),
                    )
                zero_time_fields(graph)
                batch_graphs.append(graph)
                records.append((original_rank, pose_path, pose_mol, pose_coords))

            loader = DataLoader(batch_graphs, batch_size=batch_size, shuffle=False)
            scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(model_args.device)
                    pred = model(batch)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    if pred.ndim > 1:
                        pred = pred[:, 0]
                    scores.extend(pred.detach().cpu().view(-1).tolist())

            scored = []
            for (original_rank, pose_path, pose_mol, pose_coords), score in zip(records, scores):
                scored.append({
                    'original_rank': original_rank,
                    'pose_path': pose_path,
                    'pose_mol': pose_mol,
                    'pose_coords': pose_coords,
                    'confidence': float(score),
                })
            scored.sort(key=lambda x: x['confidence'], reverse=True)

            write_dir = out_dir / complex_name
            write_dir.mkdir(parents=True, exist_ok=True)
            summary = ['new_rank,original_rank,confidence,source_file,ranked_file']
            for new_rank, item in enumerate(scored, start=1):
                out_name = f'rank{new_rank}_confidence{item["confidence"]:.4f}.sdf'
                write_mol_with_coords(item['pose_mol'], item['pose_coords'], str(write_dir / out_name))
                summary.append(
                    f'{new_rank},{item["original_rank"]},{item["confidence"]:.6f},{item["pose_path"].name},{out_name}'
                )
            (write_dir / 'confidence_ranking.csv').write_text('\n'.join(summary) + '\n')
            processed += 1
        except Exception as exc:
            logger.error(f'Failed on {pose_set_name}/{complex_name} with {kind}: {exc}', exc_info=True)
            failed += 1

    logger.info('%s + %s: processed=%d skipped=%d failed=%d', pose_set_name, kind, processed, skipped, failed)


def summarize_rmsds(rmsds):
    rows = []
    for pdb_id, pose_rmsds in sorted(rmsds.items()):
        valid = [r for r in pose_rmsds if r < float('inf')]
        if not valid:
            continue
        rows.append({
            'pdb_id': pdb_id,
            'top1': pose_rmsds[0] if pose_rmsds else float('inf'),
            'top5': min(pose_rmsds[:min(len(pose_rmsds), 5)]),
            'top10': min(pose_rmsds[:min(len(pose_rmsds), 10)]),
            'oracle': min(valid),
        })

    def pct(key, cutoff):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.mean(vals < cutoff) * 100) if len(vals) else float('nan')

    def mean(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.mean(vals)) if len(vals) else float('nan')

    def median(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(np.median(vals)) if len(vals) else float('nan')

    return {
        'n': len(rows),
        'top1_lt2': pct('top1', 2.0),
        'top1_lt3_5': pct('top1', 3.5),
        'top1_lt5': pct('top1', 5.0),
        'top5_lt2': pct('top5', 2.0),
        'top10_lt2': pct('top10', 2.0),
        'oracle_lt2': pct('oracle', 2.0),
        'top1_mean': mean('top1'),
        'top1_median': median('top1'),
        'oracle_mean': mean('oracle'),
    }


def _detect_report_models(results, sources):
    found = []
    for kind in MODEL_KINDS:
        if all(f'{kind}_{source}' in results for source in sources):
            found.append(kind)
    return found


def write_reports(results, out_dir):
    sources = [
        source for source in ('pretrained', 'ablation', 'finetuned')
        if f'baseline_{source}' in results
    ]
    if not sources:
        raise ValueError('No baseline pose sets were available for the report')
    suffix = f'_{sources[0]}' if len(sources) == 1 else ''
    summary_csv = out_dir / f'confidence_rerank_summary{suffix}.csv'
    compare_csv = out_dir / f'confidence_rerank_comparison{suffix}.csv'
    report_txt = out_dir / f'confidence_rerank_report{suffix}.txt'
    model_kinds = _detect_report_models(results, sources)

    summary_fields = [
        'label', 'n',
        'top1_mean', 'top1_median',
        'top1_lt2', 'top1_lt3_5', 'top1_lt5',
        'top5_lt2', 'top10_lt2', 'oracle_lt2',
        'oracle_mean',
    ]
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for label, metrics in results.items():
            row = {'label': label}
            row.update(metrics)
            writer.writerow(row)

    compare_rows = []
    for source in sources:
        baseline = results[f'baseline_{source}']
        row = {
            'source': source,
            'baseline_top1_lt2': baseline['top1_lt2'],
            'baseline_top5_lt2': baseline['top5_lt2'],
            'baseline_top10_lt2': baseline['top10_lt2'],
            'baseline_top1_mean': baseline['top1_mean'],
            'baseline_top1_median': baseline['top1_median'],
            'baseline_top1_lt3_5': baseline['top1_lt3_5'],
            'baseline_top1_lt5': baseline['top1_lt5'],
        }
        for kind in model_kinds:
            metrics = results[f'{kind}_{source}']
            row[f'{kind}_conf_top1_lt2'] = metrics['top1_lt2']
            row[f'{kind}_minus_baseline'] = metrics['top1_lt2'] - baseline['top1_lt2']
            row[f'{kind}_conf_top5_lt2'] = metrics['top5_lt2']
            row[f'{kind}_conf_top10_lt2'] = metrics['top10_lt2']
            row[f'{kind}_conf_top1_mean'] = metrics['top1_mean']
            row[f'{kind}_conf_top1_median'] = metrics['top1_median']
            row[f'{kind}_conf_top1_lt3_5'] = metrics['top1_lt3_5']
            row[f'{kind}_conf_top1_lt5'] = metrics['top1_lt5']
        compare_rows.append(row)

    compare_fields = list(compare_rows[0].keys())
    with open(compare_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields)
        writer.writeheader()
        for row in compare_rows:
            writer.writerow(row)

    lines = ['Confidence Rerank Comparison', '=' * 60, '', 'Summary metrics (% complexes with RMSD < 2A)', '']
    ordered = []
    for source in sources:
        ordered.append(f'baseline_{source}')
        for kind in model_kinds:
            ordered.append(f'{kind}_{source}')
    for label in ordered:
        m = results[label]
        lines.append(
            f'{label:<22} top1={m["top1_lt2"]:5.1f}%  top5={m["top5_lt2"]:5.1f}%  '
            f'top10={m["top10_lt2"]:5.1f}%  oracle={m["oracle_lt2"]:5.1f}%'
        )

    lines.extend(['', 'Top-1 RMSD comparison', '', '=' * 86])
    lines.append(
        f'  {"Condition":<20} {"N":>4} {"Mean":>8} {"Median":>8} {"<2A":>7} {"<3.5A":>7} {"<5A":>7}'
    )
    lines.append(
        f'  {"-" * 20} {"-" * 4} {"-" * 8} {"-" * 8} {"-" * 7} {"-" * 7} {"-" * 7}'
    )
    comparison_labels = []
    for source in sources:
        comparison_labels.append((f'{source} raw sample 1', f'baseline_{source}'))
        for kind in model_kinds:
            comparison_labels.append(
                (f'{source} + {MODEL_PRETTY_NAMES.get(kind, kind)}', f'{kind}_{source}')
            )
    for pretty, key in comparison_labels:
        m = results[key]
        lines.append(
            f'  {pretty:<20} {m["n"]:>4d} {m["top1_mean"]:>7.2f}A {m["top1_median"]:>7.2f}A '
            f'{m["top1_lt2"]:>6.1f}% {m["top1_lt3_5"]:>6.1f}% {m["top1_lt5"]:>6.1f}%'
        )
    lines.append('=' * 86)
    lines.append('')
    lines.append('Paired comparison by pose source')
    lines.append('')
    for row in compare_rows:
        parts = [f'{row["source"]:<10} baseline top1={row["baseline_top1_lt2"]:5.1f}%']
        delta_parts = []
        for kind in model_kinds:
            parts.append(f'{kind}_conf={row[f"{kind}_conf_top1_lt2"]:5.1f}%')
            delta_parts.append(f'delta({kind}-baseline)={row[f"{kind}_minus_baseline"]:+5.1f}%')
        lines.append('  '.join(parts))
        lines.append('           ' + '  '.join(delta_parts))

    report_txt.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved summary to: {summary_csv}')
    print(f'Saved comparison to: {compare_csv}')
    print(f'Saved report to: {report_txt}')


def write_native_comparison(results, out_dir):
    """Write one table for all pose generators under native DiffDock confidence."""
    csv_path = out_dir / 'native_diffdock_confidence_comparison.csv'
    report_path = out_dir / 'native_diffdock_confidence_comparison.txt'
    fields = [
        'condition', 'n', 'top1_mean', 'top1_median',
        'top1_lt2', 'top1_lt3_5', 'top1_lt5',
        'top5_lt2', 'top10_lt2', 'oracle_lt2', 'oracle_mean',
    ]
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in ('pretrained', 'ablation', 'finetuned'):
            writer.writerow({'condition': condition, **results[condition]})

    lines = [
        'NATIVE DIFFDOCK CONFIDENCE — UNIFIED COMPARISON',
        '=' * 104,
        ('  Condition        N    Top1 mean  Top1 median    Top1<2A  '
         'Top1<3.5A  Top1<5A  Top5<2A  Top10<2A  Oracle<2A'),
        '  ' + '-' * 100,
    ]
    for condition in ('pretrained', 'ablation', 'finetuned'):
        m = results[condition]
        lines.append(
            f'  {condition:<12} {m["n"]:>4d}  {m["top1_mean"]:>9.2f}A '
            f'{m["top1_median"]:>11.2f}A  {m["top1_lt2"]:>8.1f}% '
            f'{m["top1_lt3_5"]:>10.1f}% {m["top1_lt5"]:>8.1f}% '
            f'{m["top5_lt2"]:>8.1f}% {m["top10_lt2"]:>9.1f}% '
            f'{m["oracle_lt2"]:>10.1f}%'
        )
    lines.extend([
        '=' * 104,
        'pretrained uses its inference-time native confidence order; ablation and finetuned',
        'use the corresponding *_orig_conf reranked directories.',
        'Oracle should remain unchanged by reranking because the pose set is unchanged.',
    ])
    report_path.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'\nSaved comparison to: {csv_path}')
    print(f'Saved report to: {report_path}')


def main():
    args = get_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.protein_ligand_csv)
    pdb_ids = load_test_ids(args.test_split)

    pose_sets_all = {
        'pretrained': Path(args.pretrained_poses),
        'ablation': Path(args.ablation_poses),
        'finetuned': Path(args.finetuned_poses),
    }

    # ---- Report-only mode: aggregate existing scoring outputs ----
    if args.report_only:
        logger.info('Report-only mode: aggregating existing outputs.')
        if args.native_comparison:
            effective_paths = {
                'pretrained': pose_sets_all['pretrained'],
                'ablation': out_dir / 'ablation_orig_conf',
                'finetuned': out_dir / 'finetuned_orig_conf',
            }
            missing = [str(path) for path in effective_paths.values() if not path.is_dir()]
            if missing:
                raise FileNotFoundError(
                    'Native comparison requires all three pose directories; missing: '
                    + ', '.join(missing)
                )
            results = {
                source: summarize_rmsds(compute_rmsds(pdb_ids, args.data_dir, str(path)))
                for source, path in effective_paths.items()
            }
            write_native_comparison(results, out_dir)
            return
        results = {}
        report_pose_sets = (
            {args.pose_set: pose_sets_all[args.pose_set]}
            if args.pose_set is not None
            else pose_sets_all
        )
        candidate_models = [
            kind for kind in MODEL_KINDS
            if all((out_dir / f'{source}_{kind}_conf').exists() for source in report_pose_sets)
        ]
        for source, pose_dir in report_pose_sets.items():
            results[f'baseline_{source}'] = summarize_rmsds(
                compute_rmsds(pdb_ids, args.data_dir, str(pose_dir)))
            for kind in candidate_models:
                results[f'{kind}_{source}'] = summarize_rmsds(
                    compute_rmsds(pdb_ids, args.data_dir, str(out_dir / f'{source}_{kind}_conf')))
        write_reports(results, out_dir)
        return

    # ---- Scoring mode ----
    # Determine which subset of pose_sets and models to run
    pose_sets = (
        {args.pose_set: pose_sets_all[args.pose_set]}
        if args.pose_set is not None
        else pose_sets_all
    )
    run_orig = args.conf_model in (None, 'orig')
    run_em = args.conf_model in (None, 'em')
    run_ligdens = args.conf_model in (None, 'ligdens')

    # Load only the models we actually need
    orig_dataset = em_dataset = ligdens_dataset = None
    orig_args = em_args = ligdens_args = None
    orig_model = em_model = ligdens_model = None
    complex_names = protein_paths = ring_paths = None
    map_mapping = None

    if run_orig:
        logger.info('Building dataset for original confidence model')
        orig_args = load_model_args(args.orig_conf_dir, 'orig')
        orig_args.device = device
        orig_dataset, complex_names, protein_paths, ring_paths = build_dataset(
            df, orig_args, str(out_dir / 'tmp_orig'),
            esm_embeddings_path=args.esm_embeddings_path)
        logger.info('Loading original confidence model')
        orig_model = build_model(orig_args, 'orig', args.orig_conf_dir, args.orig_conf_ckpt, device)
        _freeze_bn(orig_model)

    if run_em:
        logger.info('Building dataset for EM confidence model')
        em_args = load_model_args(args.em_conf_dir, 'em')
        em_args.device = device
        em_dataset, em_cn, em_pp, em_rp = build_dataset(
            df, em_args, str(out_dir / 'tmp_em'),
            esm_embeddings_path=args.esm_embeddings_path,
        )
        if complex_names is None:
            complex_names, protein_paths, ring_paths = em_cn, em_pp, em_rp
        logger.info('Loading EM confidence model')
        em_model = build_model(em_args, 'em', args.em_conf_dir, args.em_conf_ckpt, device)
        _freeze_bn(em_model)

    if run_ligdens:
        if not args.ligdens_conf_dir:
            raise ValueError('--ligdens_conf_dir is required when scoring ligand-density confidence')
        if not args.map_csv:
            raise ValueError('--map_csv is required when scoring ligand-density confidence')
        logger.info('Building dataset for ligand-density confidence model')
        ligdens_args = load_model_args(args.ligdens_conf_dir, 'ligdens')
        ligdens_args.device = device
        ligdens_dataset, ld_cn, ld_pp, ld_rp = build_dataset(
            df, ligdens_args, str(out_dir / 'tmp_ligdens'),
            esm_embeddings_path=args.esm_embeddings_path,
        )
        if complex_names is None:
            complex_names, protein_paths, ring_paths = ld_cn, ld_pp, ld_rp
        map_mapping = load_map_mapping(args.map_csv)
        logger.info('Loading ligand-density confidence model')
        ligdens_model = build_model(
            ligdens_args, 'ligdens', args.ligdens_conf_dir, args.ligdens_conf_ckpt, device)
        _freeze_bn(ligdens_model)

    for name, pose_dir in pose_sets.items():
        if run_orig:
            score_and_write_pose_set(
                name, pose_dir, out_dir / f'{name}_orig_conf',
                orig_dataset, complex_names, protein_paths, ring_paths, orig_args,
                orig_model, 'orig', args.batch_size, args.max_poses, args.data_dir,
            )
        if run_em:
            score_and_write_pose_set(
                name, pose_dir, out_dir / f'{name}_em_conf',
                em_dataset, complex_names, protein_paths, ring_paths, em_args,
                em_model, 'em', args.batch_size, args.max_poses, args.data_dir,
            )
        if run_ligdens:
            score_and_write_pose_set(
                name, pose_dir, out_dir / f'{name}_ligdens_conf',
                ligdens_dataset, complex_names, protein_paths, ring_paths, ligdens_args,
                ligdens_model, 'ligdens', args.batch_size, args.max_poses, args.data_dir,
                map_mapping=map_mapping,
            )

    # Auto-generate full report only when running all 6 tasks in one process
    if args.pose_set is None and args.conf_model is None:
        results = {}
        for source, pose_dir in pose_sets_all.items():
            results[f'baseline_{source}'] = summarize_rmsds(
                compute_rmsds(pdb_ids, args.data_dir, str(pose_dir)))
            if run_orig:
                results[f'orig_{source}'] = summarize_rmsds(
                    compute_rmsds(pdb_ids, args.data_dir, str(out_dir / f'{source}_orig_conf')))
            if run_em:
                results[f'em_{source}'] = summarize_rmsds(
                    compute_rmsds(pdb_ids, args.data_dir, str(out_dir / f'{source}_em_conf')))
            if run_ligdens:
                results[f'ligdens_{source}'] = summarize_rmsds(
                    compute_rmsds(pdb_ids, args.data_dir, str(out_dir / f'{source}_ligdens_conf')))
        write_reports(results, out_dir)


if __name__ == '__main__':
    main()
