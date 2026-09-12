"""EM-aware Confidence Dataset for DiffDock-EM.

Generates training data for the EM confidence model:
1. Builds all-atom complex graphs with ring features (EMDataset, all_atoms=True)
2. Runs the finetuned EM score model (CGModel, all_atoms=False) to generate N poses
3. Computes RMSD vs crystal structure → binary label (RMSD < cutoff)
4. Returns (AA_graph + ring_features, label) pairs for confidence model training

Ring feature dimensions match the score model training:
  ring_dim=24   — l=0 scalar ring features (24 channels)
  ring_vec_dim=6 — l=1 vector ring features (6 radii × 3D vectors)
"""

import copy
import math
import os
import pickle
import random
from argparse import Namespace
from functools import partial

import numpy as np
import torch
import yaml
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from em_dataset import EMDataset
from utils.diffusion_utils import get_t_schedule, t_to_sigma as t_to_sigma_compl
from utils.sampling import randomize_position, sampling
from utils.utils import get_em_model, load_pretrained_weights
from models.lora import apply_lora_to_model, merge_lora_weights


class ListDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class EMConfidenceDataset(Dataset):
    """Confidence dataset using cryo-EM ring features.

    The score model (CGModel, all_atoms=False) generates N poses per complex
    during preprocessing. The confidence model (AAModel, all_atoms=True) is
    trained to rank those poses using all-atom graphs with ring features.

    Pose positions are in graph-centered coordinates (ligand.pos), consistent
    between CG and AA representations of the same complex.
    """

    def __init__(
        self,
        # Data paths
        data_dir: str,
        split_path: str,
        ring_features_dir: str,
        cache_path: str,
        # Score model (CGModel, finetuned on EM data)
        score_model_dir: str,
        score_model_ckpt: str = 'best_ema_model.pt',
        # Pose generation
        inference_steps: int = 20,
        samples_per_complex: int = 10,
        # Confidence model graph construction (all-atom)
        all_atoms: bool = True,
        receptor_radius: float = 15.0,
        c_alpha_max_neighbors: int = 24,
        atom_radius: float = 5.0,
        atom_max_neighbors: int = 8,
        ring_dim: int = 24,
        ring_vec_dim: int = 6,
        remove_hs: bool = True,
        max_lig_size=None,
        matching: bool = True,
        popsize: int = 20,
        maxiter: int = 20,
        num_conformers: int = 1,
        knn_only_graph: bool = True,
        esm_embeddings_path=None,
        protein_file: str = 'protein_processed',
        num_workers: int = 1,
        # Confidence training
        rmsd_classification_cutoff=2.0,
        balance: bool = False,
        limit_complexes: int = 0,
        cache_creation_id=None,
        cache_ids_to_combine=None,
        device='cpu',
        exclude_complexes=None,
    ):
        super().__init__()

        self.all_atoms = all_atoms
        self.balance = balance
        self.rmsd_classification_cutoff = rmsd_classification_cutoff
        self.samples_per_complex = samples_per_complex
        self.cache_creation_id = cache_creation_id
        self.cache_ids_to_combine = cache_ids_to_combine
        self.device = device
        self.ring_dim = ring_dim
        self.ring_vec_dim = ring_vec_dim

        # --- Build AA EM dataset (all_atoms=True) for confidence model ---
        # These graphs have both receptor-level and atom-level ring features
        aa_cache = os.path.join(cache_path, 'aa_graphs')
        self.aa_dataset = EMDataset(
            root=data_dir,
            split_path=split_path,
            ring_features_dir=ring_features_dir,
            transform=None,        # no noise transform; confidence uses t=0
            cache_path=aa_cache,
            all_atoms=True,
            receptor_radius=receptor_radius,
            c_alpha_max_neighbors=c_alpha_max_neighbors,
            atom_radius=atom_radius,
            atom_max_neighbors=atom_max_neighbors,
            ring_dim=ring_dim,
            ring_vec_dim=ring_vec_dim,
            remove_hs=remove_hs,
            max_lig_size=max_lig_size,
            matching=matching,
            popsize=popsize,
            maxiter=maxiter,
            num_conformers=num_conformers,
            knn_only_graph=knn_only_graph,
            esm_embeddings_path=esm_embeddings_path,
            protein_file=protein_file,
            num_workers=num_workers,
            blacklist_path=None,   # confidence model has its own error handling
        )

        self.complex_graph_dict = {g.name: g for g in self.aa_dataset.complex_graphs}
        if limit_complexes > 0:
            keys = list(self.complex_graph_dict.keys())[:limit_complexes]
            self.complex_graph_dict = {k: self.complex_graph_dict[k] for k in keys}
        if exclude_complexes:
            exclude_set = set(exclude_complexes)
            before = len(self.complex_graph_dict)
            self.complex_graph_dict = {k: v for k, v in self.complex_graph_dict.items()
                                       if k not in exclude_set}
            removed = before - len(self.complex_graph_dict)
            if removed:
                print(f'Excluded {removed} complexes from dataset (exclude_complexes list)')

        # --- Load score model config ---
        for cfg_name in ('config.yml', 'model_parameters.yml'):
            cfg_path = os.path.join(score_model_dir, cfg_name)
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    self.score_model_args = Namespace(**yaml.safe_load(f))
                break
        else:
            raise FileNotFoundError(
                f'No config.yml or model_parameters.yml found in {score_model_dir}'
            )

        # --- Cache path for ligand positions ---
        split_name = os.path.splitext(os.path.basename(split_path))[0]
        score_name = os.path.basename(score_model_dir.rstrip('/'))
        self.full_cache_path = os.path.join(
            cache_path,
            f'poses_{score_name}_split_{split_name}_limit{limit_complexes}',
        )

        # --- Preprocessing: generate poses if not cached ---
        id_suffix = '' if cache_creation_id is None else f'_id{cache_creation_id}'

        if cache_ids_to_combine is None:
            pkl_path = os.path.join(self.full_cache_path, f'ligand_positions{id_suffix}.pkl')
            if not os.path.exists(pkl_path):
                os.makedirs(self.full_cache_path, exist_ok=True)
                self._preprocessing(
                    score_model_dir=score_model_dir,
                    score_model_ckpt=score_model_ckpt,
                    inference_steps=inference_steps,
                    data_dir=data_dir,
                    split_path=split_path,
                    ring_features_dir=ring_features_dir,
                    cache_path=cache_path,
                    remove_hs=remove_hs,
                    num_workers=num_workers,
                )
            with open(pkl_path, 'rb') as f:
                self.full_ligand_positions, self.rmsds = pickle.load(f)
            names_path = os.path.join(self.full_cache_path, f'complex_names{id_suffix}.pkl')
            with open(names_path, 'rb') as f:
                generated_names = pickle.load(f)

        else:
            # Combine multiple cache shards
            all_pos, all_rms, all_names = [], [], []
            for cid in cache_ids_to_combine:
                p = os.path.join(self.full_cache_path, f'ligand_positions_id{cid}.pkl')
                n = os.path.join(self.full_cache_path, f'complex_names_id{cid}.pkl')
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f'Ligand positions cache not found for id {cid}: {p}'
                    )
                with open(p, 'rb') as f:
                    pos, rms = pickle.load(f)
                with open(n, 'rb') as f:
                    names = pickle.load(f)
                all_pos.append(pos)
                all_rms.append(rms)
                all_names.append(names)

            generated_names = list(set(sum(all_names, [])))
            name_to_pos, name_to_rmsd = {}, {}
            for positions, rmsds, names in zip(all_pos, all_rms, all_names):
                for name, pos, rmsd in zip(names, positions, rmsds):
                    name_to_pos.setdefault(name, []).append(pos)
                    name_to_rmsd.setdefault(name, []).append(rmsd)
            self.full_ligand_positions = [
                np.concatenate(name_to_pos[n], axis=0) for n in generated_names
            ]
            self.rmsds = [
                np.concatenate(name_to_rmsd[n], axis=0) for n in generated_names
            ]

        self.all_samples_per_complex = samples_per_complex * (
            1 if cache_ids_to_combine is None else len(cache_ids_to_combine)
        )
        self.positions_rmsds_dict = {
            name: (pos, rmsd)
            for name, pos, rmsd in zip(generated_names, self.full_ligand_positions, self.rmsds)
        }

        # --- Filter out bad poses (NaN/Inf coords, NaN/Inf/huge RMSD) ---
        # Preprocessing sometimes produces numerically exploded poses (RMSD up to 1e13 Å).
        # Feeding these into the confidence forward pass triggers NaN activations.
        # Compute a "good index" mask per complex and sample only from it in get().
        self.bad_rmsd_max = 100.0  # any pose with RMSD > 100 Å is considered corrupt
        self.good_indices_dict = {}
        total_poses = 0
        total_bad = 0
        for name, (pos, rmsd) in self.positions_rmsds_dict.items():
            rmsd_arr = np.asarray(rmsd)
            pos_arr = np.asarray(pos)
            # pos_arr shape: (n_samples, n_atoms, 3) -> per-sample finite check
            pos_finite = np.isfinite(pos_arr).reshape(pos_arr.shape[0], -1).all(axis=1)
            rmsd_ok = np.isfinite(rmsd_arr) & (rmsd_arr < self.bad_rmsd_max)
            good = pos_finite & rmsd_ok
            self.good_indices_dict[name] = np.flatnonzero(good)
            total_poses += good.size
            total_bad += int((~good).sum())

        # Drop complexes with zero good poses (can't train on them at all)
        candidate_names = set(self.positions_rmsds_dict.keys()) & set(self.complex_graph_dict.keys())
        n_all_bad = sum(1 for n in candidate_names if len(self.good_indices_dict[n]) == 0)
        self.dataset_names = [n for n in candidate_names if len(self.good_indices_dict[n]) > 0]

        print(f'EMConfidenceDataset: {len(self.dataset_names)} complexes, '
              f'up to {self.all_samples_per_complex} poses each')
        print(f'  bad poses filtered: {total_bad}/{total_poses} '
              f'({100 * total_bad / max(total_poses, 1):.2f}%)  '
              f'[NaN/Inf coords or RMSD ≥ {self.bad_rmsd_max} Å]')
        if n_all_bad:
            print(f'  dropped {n_all_bad} complexes with 0 good poses')
        if self.dataset_names:
            cutoff = (self.rmsd_classification_cutoff[0]
                      if isinstance(self.rmsd_classification_cutoff, list)
                      else self.rmsd_classification_cutoff)
            pos_samples = 0
            good_total = 0
            for n in self.dataset_names:
                _, rmsd = self.positions_rmsds_dict[n]
                gi = self.good_indices_dict[n]
                pos_samples += int(np.sum(np.asarray(rmsd)[gi] < cutoff))
                good_total += gi.size
            print(f'Positive samples (RMSD < {cutoff} Å, good poses only): '
                  f'{pos_samples}/{good_total} = {100 * pos_samples / max(good_total, 1):.1f}%')

    # -------------------------------------------------------------------------

    def len(self):
        return len(self.dataset_names)

    def get(self, idx):
        name = self.dataset_names[idx]
        complex_graph = copy.deepcopy(self.complex_graph_dict[name])
        positions, rmsds = self.positions_rmsds_dict[name]
        good_idx = self.good_indices_dict[name]
        # Restrict to good (non-exploded, finite-RMSD) poses
        good_positions = positions[good_idx]
        good_rmsds = np.asarray(rmsds)[good_idx]
        n_good = good_idx.size

        if self.balance:
            if isinstance(self.rmsd_classification_cutoff, list):
                raise ValueError(
                    'List rmsd_classification_cutoff is not supported with balance=True'
                )
            label = random.randint(0, 1)
            success = good_rmsds < self.rmsd_classification_cutoff
            n_success = int(np.count_nonzero(success))
            if label == 0 and n_success != n_good:
                sample = random.randint(0, n_good - n_success - 1)
                complex_graph['ligand'].pos = torch.from_numpy(good_positions[~success][sample])
            else:
                if n_success > 0:
                    sample = random.randint(0, n_success - 1)
                    complex_graph['ligand'].pos = torch.from_numpy(good_positions[success][sample])
                else:
                    # No positives available — fall back to a random negative so the batch is valid
                    sample = random.randint(0, n_good - 1)
                    complex_graph['ligand'].pos = torch.from_numpy(good_positions[sample])
                    label = 0
            complex_graph.y = torch.tensor(label).float().unsqueeze(0)
        else:
            sample = random.randint(0, n_good - 1)
            complex_graph['ligand'].pos = torch.from_numpy(good_positions[sample])
            if isinstance(self.rmsd_classification_cutoff, list):
                complex_graph.y_binned = torch.tensor(
                    np.logical_and(
                        good_rmsds[sample] < self.rmsd_classification_cutoff + [math.inf],
                        good_rmsds[sample] >= [0] + self.rmsd_classification_cutoff,
                    ),
                    dtype=torch.float,
                ).unsqueeze(0)
                complex_graph.y = torch.tensor(
                    good_rmsds[sample] < self.rmsd_classification_cutoff[0]
                ).unsqueeze(0).float()
            else:
                complex_graph.y = torch.tensor(
                    good_rmsds[sample] < self.rmsd_classification_cutoff
                ).float().unsqueeze(0)
            complex_graph.rmsd = torch.tensor(good_rmsds[sample]).unsqueeze(0).float()

        # Confidence model evaluates at t=0 (no noise)
        zeros = lambda n: torch.zeros(n)
        complex_graph['ligand'].node_t = {
            k: zeros(complex_graph['ligand'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        complex_graph['receptor'].node_t = {
            k: zeros(complex_graph['receptor'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        if self.all_atoms and hasattr(complex_graph, 'atom') and hasattr(complex_graph['atom'], 'x'):
            complex_graph['atom'].node_t = {
                k: zeros(complex_graph['atom'].num_nodes) for k in ('tr', 'rot', 'tor')
            }
        complex_graph.complex_t = {k: zeros(1) for k in ('tr', 'rot', 'tor')}

        return complex_graph

    # -------------------------------------------------------------------------

    def build_graph_with_pose(self, name, pose_position):
        """Construct a complex graph with a specific pose position placed on the ligand.

        Used by EMConfidencePairDataset to build (good, bad) pose pairs for
        pairwise ranking loss. Mirrors the t=0 setup from get() but skips the
        balance/label sampling logic — caller chooses which pose to place.
        """
        complex_graph = copy.deepcopy(self.complex_graph_dict[name])
        complex_graph['ligand'].pos = torch.from_numpy(pose_position)

        zeros = lambda n: torch.zeros(n)
        complex_graph['ligand'].node_t = {
            k: zeros(complex_graph['ligand'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        complex_graph['receptor'].node_t = {
            k: zeros(complex_graph['receptor'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        if self.all_atoms and hasattr(complex_graph, 'atom') and hasattr(complex_graph['atom'], 'x'):
            complex_graph['atom'].node_t = {
                k: zeros(complex_graph['atom'].num_nodes) for k in ('tr', 'rot', 'tor')
            }
        complex_graph.complex_t = {k: zeros(1) for k in ('tr', 'rot', 'tor')}
        return complex_graph

    # -------------------------------------------------------------------------

    def _preprocessing(
        self,
        score_model_dir,
        score_model_ckpt,
        inference_steps,
        data_dir,
        split_path,
        ring_features_dir,
        cache_path,
        remove_hs,
        num_workers,
    ):
        """Run the finetuned EM score model to generate poses and cache them.

        The score model is a CGModel (all_atoms=False). We build CG EM graphs
        to feed into it, then store predicted ligand positions in graph-centered
        coordinates — compatible with the AA graphs used during training.
        """
        score_args = self.score_model_args
        t_to_sigma = partial(t_to_sigma_compl, args=score_args)

        # Ring dims used by the score model
        score_ring_dim = getattr(score_args, 'ring_dim', self.ring_dim)
        score_ring_vec_dim = getattr(score_args, 'ring_vec_dim', self.ring_vec_dim)

        # Load finetuned EM score model (CGModel)
        score_model = get_em_model(
            score_args,
            self.device,
            t_to_sigma,
            ring_dim=score_ring_dim,
            ring_vec_dim=score_ring_vec_dim,
            no_parallel=True,
        )

        # Apply LoRA if the score model was trained with it (auto-detect from config)
        use_lora = getattr(score_args, 'use_lora', False)
        if use_lora:
            lora_r = getattr(score_args, 'lora_r', 8)
            lora_alpha = getattr(score_args, 'lora_alpha', 16)
            print(f'  Applying LoRA to score model (r={lora_r}, alpha={lora_alpha})')
            score_model, _, _ = apply_lora_to_model(
                score_model, r=lora_r, alpha=lora_alpha, dropout=0.0, verbose=False,
            )

        ckpt_path = os.path.join(score_model_dir, score_model_ckpt)
        load_pretrained_weights(score_model, ckpt_path)
        score_model = score_model.to(self.device)

        if use_lora:
            # Restore pretrained BN stats (LoRA finetuning corrupts them)
            pretrained_model_path = getattr(score_args, 'pretrained_model', None)
            if pretrained_model_path and os.path.exists(pretrained_model_path):
                from inference_em import restore_pretrained_bn_stats
                restore_pretrained_bn_stats(score_model, pretrained_model_path)
            else:
                print(f'  WARNING: pretrained_model not found; BN stats not restored')

            # Merge LoRA into base weights for faster inference
            print(f'  Merging LoRA weights...')
            score_model = merge_lora_weights(score_model)

            # train() + freeze BN: needed for merged LoRA models (eval mode causes NaN)
            score_model.train()
            for module in score_model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
        else:
            # Use train() + frozen BN instead of eval(): e3nn finetuned models
            # produce degraded scores in eval mode (same fix as inference_em.py).
            score_model.train()
            for module in score_model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()

        # Build CG EM dataset for score model inference (all_atoms=False)
        cg_dataset = EMDataset(
            root=data_dir,
            split_path=split_path,
            ring_features_dir=ring_features_dir,
            transform=None,
            cache_path=os.path.join(cache_path, 'cg_graphs'),
            all_atoms=False,
            receptor_radius=getattr(score_args, 'receptor_radius', 15.0),
            c_alpha_max_neighbors=getattr(score_args, 'c_alpha_max_neighbors', 24),
            ring_dim=score_ring_dim,
            ring_vec_dim=score_ring_vec_dim,
            remove_hs=remove_hs,
            matching=not getattr(score_args, 'no_torsion', False),
            popsize=getattr(score_args, 'matching_popsize', 20),
            maxiter=getattr(score_args, 'matching_maxiter', 20),
            num_conformers=getattr(score_args, 'num_conformers', 1),
            knn_only_graph=not getattr(score_args, 'not_knn_only_graph', False),
            esm_embeddings_path=getattr(score_args, 'esm_embeddings_path', None),
            num_workers=num_workers,
            blacklist_path=None,   # confidence model has its own error handling
        )
        cg_graph_dict = {g.name: g for g in cg_dataset.complex_graphs}

        common_names = list(
            set(self.complex_graph_dict.keys()) & set(cg_graph_dict.keys())
        )
        print(f'Preprocessing {len(common_names)} complexes, '
              f'{self.samples_per_complex} poses each...')

        sigma_schedule = getattr(score_args, 'sigma_schedule', 'expbeta')
        tr_schedule = get_t_schedule(sigma_schedule=sigma_schedule, inference_steps=inference_steps)

        rmsds_all, positions_all, names_all = [], [], []
        for name in tqdm(common_names):
            cg_graph = cg_graph_dict[name]

            data_list = [copy.deepcopy(cg_graph) for _ in range(self.samples_per_complex)]
            # sampling.py does mask_rotate[0] expecting a list-wrapped 2D array.
            # process_mols.py stores it as a bare 2D array, so wrap it here.
            for d in data_list:
                mr = d['ligand'].mask_rotate
                if not isinstance(mr, list):
                    d['ligand'].mask_rotate = [mr]
            randomize_position(
                data_list,
                score_args.no_torsion,
                False,
                score_args.tr_sigma_max,
            )

            predictions_list = None
            failed_counter = 0
            skip_complex = False
            while predictions_list is None:
                try:
                    predictions_list, _ = sampling(
                        data_list=data_list,
                        model=score_model,
                        inference_steps=inference_steps,
                        tr_schedule=tr_schedule,
                        rot_schedule=tr_schedule,
                        tor_schedule=tr_schedule,
                        device=self.device,
                        t_to_sigma=t_to_sigma,
                        model_args=score_args,
                        no_random=True,
                    )
                except Exception as e:
                    if 'failed to converge' in str(e):
                        failed_counter += 1
                        if failed_counter > 5:
                            print(f'WARNING: SVD failed 5x for {name}, using last state')
                            predictions_list = data_list
                            break
                        print(f'WARNING: SVD failed for {name}, retrying...')
                    elif 'No edges and no nodes' in str(e):
                        print(f'WARNING: No edges/nodes for {name}, skipping complex.')
                        skip_complex = True
                        break
                    else:
                        print(f'WARNING: Unexpected error for {name}: {e}, skipping complex.')
                        skip_complex = True
                        break
            if skip_complex:
                continue

            # Compute RMSD vs crystal structure
            if score_args.no_torsion:
                cg_graph['ligand'].orig_pos = (
                    cg_graph['ligand'].pos.cpu().numpy()
                    + cg_graph.original_center.cpu().numpy()
                )

            # Filter out hydrogens when computing RMSD
            filterHs = torch.not_equal(
                predictions_list[0]['ligand'].x[:, 0], 0
            ).cpu().numpy()

            orig_pos = cg_graph['ligand'].orig_pos
            if isinstance(orig_pos, list):
                orig_pos = orig_pos[0]

            ligand_pos = np.asarray([
                g['ligand'].pos.cpu().numpy()[filterHs] for g in predictions_list
            ])
            orig_ligand_pos = np.expand_dims(
                orig_pos[filterHs] - cg_graph.original_center.cpu().numpy(),
                axis=0,
            )
            rmsd = np.sqrt(
                ((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1)
            )

            rmsds_all.append(rmsd)
            positions_all.append(np.asarray([
                g['ligand'].pos.cpu().numpy() for g in predictions_list
            ]))
            names_all.append(name)

        # Save to cache
        id_suffix = (
            '' if self.cache_creation_id is None
            else f'_id{self.cache_creation_id}'
        )
        with open(
            os.path.join(self.full_cache_path, f'ligand_positions{id_suffix}.pkl'), 'wb'
        ) as f:
            pickle.dump((positions_all, rmsds_all), f)
        with open(
            os.path.join(self.full_cache_path, f'complex_names{id_suffix}.pkl'), 'wb'
        ) as f:
            pickle.dump(names_all, f)

        del score_model, cg_dataset
        torch.cuda.empty_cache()
        print(f'Preprocessing complete. Cache: {self.full_cache_path}')


# ===========================================================================
# Pairwise dataset for within-complex ranking loss
# ===========================================================================

class EMConfidencePairDataset(torch.utils.data.Dataset):
    """Returns (good_pose, bad_pose) pairs from the SAME complex for pairwise
    ranking loss.

    Why this exists:
        Pose-level BCE (single pose per sample) doesn't give the model any
        explicit incentive to rank poses *within* a complex. With ~32%
        positives, BCE learns a global "good pose" classifier that ends up
        producing the same within-complex ordering as the pretrained orig
        confidence model (100% top-1 agreement observed in diagnostics).

        Pairwise training — for each complex, force good_score > bad_score
        by a margin — directly optimises within-complex reranking, which is
        what top_k_hit_rate actually measures.

    Pair-construction strategy:
        Any two poses with RMSD gap >= ``min_rmsd_gap`` form a valid pair;
        the lower-RMSD pose is labelled "good", the higher "bad". This lets
        us use complexes where ALL poses are <2Å (pair 0.5Å vs 1.8Å) or ALL
        poses are ≥2Å (pair 3Å vs 8Å) — teaching the model a continuous
        quality ordering, not a binary classifier.

        With the original binary strategy (one <2Å + one ≥2Å required),
        ~48% of complexes were dropped. Relaxing to continuous RMSD keeps
        the vast majority of complexes usable.

    Each __getitem__ returns a list [good_graph, bad_graph]. Downstream
    (pairwise_collate / train_epoch_pairwise) flattens a batch to
    2*batch_size graphs and splits predictions back for margin_ranking_loss.

    The ``label_cutoff`` is kept only for the auxiliary BCE calibration term
    (graph.y = 1 if RMSD<cutoff else 0). It does NOT gate which pairs are
    formed.
    """

    def __init__(self, em_dataset, rmsd_cutoff=2.0, min_rmsd_gap=1.0,
                 max_resample_tries=10, seed=None):
        self.em_dataset = em_dataset
        self.label_cutoff = (
            rmsd_cutoff[0] if isinstance(rmsd_cutoff, list) else rmsd_cutoff
        )
        self.min_rmsd_gap = float(min_rmsd_gap)
        self.max_resample_tries = int(max_resample_tries)

        # A complex is usable if it has at least 2 poses whose RMSDs differ by
        # >= min_rmsd_gap. We pre-check this once so __getitem__ can't hit a
        # complex with no valid pair.
        self.valid_names = []
        self._pose_rmsds = {}   # name -> 1D np.array of RMSDs (good poses only)
        for name in em_dataset.dataset_names:
            _, rmsds = em_dataset.positions_rmsds_dict[name]
            good_idx = em_dataset.good_indices_dict[name]
            good_rmsds = np.asarray(rmsds)[good_idx]
            if good_rmsds.size < 2:
                continue
            if (good_rmsds.max() - good_rmsds.min()) < self.min_rmsd_gap:
                continue
            self.valid_names.append(name)
            self._pose_rmsds[name] = good_rmsds

        self._rng = random.Random(seed) if seed is not None else random

        n_total = len(em_dataset.dataset_names)
        n_valid = len(self.valid_names)
        n_dropped = n_total - n_valid

        # Classify the valid complexes to verify the distribution.
        n_mixed = n_allgood = n_allbad = 0
        for name in self.valid_names:
            r = self._pose_rmsds[name]
            n_pos = int((r < self.label_cutoff).sum())
            n_neg = int((r >= self.label_cutoff).sum())
            if n_pos > 0 and n_neg > 0:
                n_mixed += 1
            elif n_pos > 0:
                n_allgood += 1
            else:
                n_allbad += 1
        print(f'EMConfidencePairDataset: {n_valid}/{n_total} complexes usable '
              f'(min RMSD gap >= {self.min_rmsd_gap:.1f} Å, dropped {n_dropped})')
        print(f'  Breakdown of usable complexes: '
              f'mixed={n_mixed}  all<{self.label_cutoff:.1f}Å={n_allgood}  '
              f'all≥{self.label_cutoff:.1f}Å={n_allbad}')

    def __len__(self):
        return len(self.valid_names)

    def _sample_pair_indices(self, rmsds):
        """Return (low_idx, high_idx) such that rmsds[high] - rmsds[low] >= gap.

        Uses rejection sampling; falls back to (argmin, argmax) if rejection
        fails. Since every ``valid_names`` complex has max-min >= gap by
        construction, the fallback is guaranteed to satisfy the gap constraint.
        """
        n = rmsds.size
        for _ in range(self.max_resample_tries):
            i, j = self._rng.randrange(n), self._rng.randrange(n)
            if i == j:
                continue
            if rmsds[i] > rmsds[j]:
                i, j = j, i
            if rmsds[j] - rmsds[i] >= self.min_rmsd_gap:
                return i, j
        # Fallback: pick the extremes — guaranteed gap >= min_gap.
        return int(np.argmin(rmsds)), int(np.argmax(rmsds))

    def __getitem__(self, idx):
        name = self.valid_names[idx]
        positions, _ = self.em_dataset.positions_rmsds_dict[name]
        good_idx = self.em_dataset.good_indices_dict[name]
        good_positions = positions[good_idx]
        good_rmsds = self._pose_rmsds[name]

        low_pi, high_pi = self._sample_pair_indices(good_rmsds)

        good_graph = self.em_dataset.build_graph_with_pose(
            name, good_positions[low_pi]
        )
        bad_graph = self.em_dataset.build_graph_with_pose(
            name, good_positions[high_pi]
        )
        # Label by absolute RMSD cutoff (for auxiliary BCE). Note that in the
        # "all_good" case both y=1; in "all_bad" both y=0. Pairwise loss uses
        # the implicit ranking (low-rmsd > high-rmsd) not these labels.
        good_graph.y = torch.tensor(
            [1.0 if good_rmsds[low_pi] < self.label_cutoff else 0.0]
        ).float()
        bad_graph.y = torch.tensor(
            [1.0 if good_rmsds[high_pi] < self.label_cutoff else 0.0]
        ).float()
        # Expose the RMSDs in case downstream code wants to weight pairs by gap.
        good_graph.pair_rmsd = torch.tensor([float(good_rmsds[low_pi])]).float()
        bad_graph.pair_rmsd = torch.tensor([float(good_rmsds[high_pi])]).float()
        return [good_graph, bad_graph]


def pairwise_collate(batch):
    """Flatten list of pairs into a single list of graphs: [g0_good, g0_bad, g1_good, g1_bad, ...].

    Use with torch.utils.data.DataLoader(collate_fn=pairwise_collate). The
    resulting flat list is consumed by DataParallel / DataListLoader-style
    model forwards; train_epoch splits predictions via pred[0::2] / pred[1::2].
    """
    flat = []
    for pair in batch:
        flat.extend(pair)
    return flat
