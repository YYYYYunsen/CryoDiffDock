"""Training script for DiffDock-EM Confidence Model.

Finetunes a pretrained DiffDock AA confidence model with cryo-EM ring features.
Uses AAModel (all_atoms=True) + ring feature injection for improved pose ranking.

Ring feature dimensions (must match score model training):
  ring_dim=24   — l=0 scalar ring features (24 channels per atom)
  ring_vec_dim=6 — l=1 vector ring features (6 radii × 3D per atom)

Workflow:
  1. EMConfidenceDataset builds all-atom EM graphs and generates poses via
     the finetuned EM score model (preprocessing, cached).
  2. AAModel(confidence_mode=True, ring_dim=24, ring_vec_dim=6) is initialized
     from the pretrained DiffDock AA confidence weights.
  3. Binary cross-entropy training: label = (RMSD < cutoff).
"""

import copy
import gc
import math
import os
from argparse import ArgumentParser, FileType, Namespace
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm

from confidence.em_dataset import EMConfidenceDataset, EMConfidencePairDataset, pairwise_collate
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.training import AverageMeter
from utils.utils import (
    ExponentialMovingAverage,
    get_em_model,
    load_pretrained_weights,
    save_yaml_file,
)
from models.lora import (
    apply_lora_to_model,
    freeze_non_lora_params,
    get_lora_state_dict,
    print_trainable_params,
)

torch.multiprocessing.set_sharing_strategy('file_system')

# Per-sample loss thresholds (mirroring score model constants in utils/training.py)
# CONF_BATCH_LOSS_MAX is intentionally large: as training progresses the model becomes
# more confident and BCE for hard examples grows naturally (logit=15 → BCE≈15).
# The real protection is CONF_SAMPLE_LOSS_CLAMP, which caps the gradient contribution
# per sample before the mean. CONF_BATCH_LOSS_MAX only guards against genuine NaN/Inf.
CONF_BATCH_LOSS_MAX  = 1e6   # effectively disabled; real guard is CONF_SAMPLE_LOSS_CLAMP
CONF_SAMPLE_LOSS_CLAMP = 3.0 # soft cap before mean: noisy samples still train, just down-weighted
PARAM_MAX_THRESHOLD  = 2e1   # params > 20 are considered corrupted (large-but-finite); skip EMA update


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_parser():
    parser = ArgumentParser(description='DiffDock-EM Confidence Model Finetuning')
    parser.add_argument('--config', type=FileType(mode='r'), default=None)

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed EM structures')
    parser.add_argument('--split_train', type=str, required=True,
                        help='Path to training split file')
    parser.add_argument('--split_val', type=str, required=True,
                        help='Path to validation split file')
    parser.add_argument('--ring_features_dir', type=str, required=True,
                        help='Directory containing ring feature .npz files')
    parser.add_argument('--esm_embeddings_path', type=str, default=None)
    parser.add_argument('--cache_path', type=str, default='data/cache_em_confidence',
                        help='Root cache directory')

    # Pretrained models
    parser.add_argument('--pretrained_confidence_model', type=str, required=True,
                        help='Path to pretrained DiffDock AA confidence model weights (.pt)')
    parser.add_argument('--pretrained_confidence_config', type=str, default=None,
                        help='Path to pretrained confidence model config YAML '
                             '(auto-detected from parent dir if not set)')
    parser.add_argument('--score_model_dir', type=str, required=True,
                        help='Directory of finetuned EM score model (has config.yml + ckpt)')
    parser.add_argument('--score_model_ckpt', type=str, default='best_ema_model.pt',
                        help='Score model checkpoint filename')

    # Ring features — must match score model training
    parser.add_argument('--ring_dim', type=int, default=24,
                        help='l=0 scalar ring feature channels (default: 24)')
    parser.add_argument('--ring_vec_dim', type=int, default=6,
                        help='l=1 vector ring feature radii (default: 6)')
    parser.add_argument('--ring_lr_multiplier', type=float, default=10.0,
                        help='LR multiplier for ring embedding parameters')

    # Pose generation
    parser.add_argument('--inference_steps', type=int, default=20,
                        help='Denoising steps for pose generation')
    parser.add_argument('--samples_per_complex', type=int, default=10,
                        help='Number of poses per complex')
    parser.add_argument('--cache_creation_id', type=int, default=None,
                        help='Shard ID for parallel cache creation')
    parser.add_argument('--cache_ids_to_combine', nargs='+', type=int, default=None,
                        help='Cache shard IDs to combine into one dataset')

    # Dataset construction
    parser.add_argument('--receptor_radius', type=float, default=15.0)
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=24)
    parser.add_argument('--atom_radius', type=float, default=5.0)
    parser.add_argument('--atom_max_neighbors', type=int, default=8)
    parser.add_argument('--remove_hs', action='store_true', default=True)
    parser.add_argument('--keep_hs', action='store_true')
    parser.add_argument('--max_lig_size', type=int, default=None)
    parser.add_argument('--matching_popsize', type=int, default=20)
    parser.add_argument('--matching_maxiter', type=int, default=20)
    parser.add_argument('--num_conformers', type=int, default=1)
    parser.add_argument('--not_knn_only_graph', action='store_true')
    parser.add_argument('--protein_file', type=str, default='protein_processed')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--limit_complexes', type=int, default=0)
    parser.add_argument('--exclude_complexes', nargs='+', type=str, default=None,
                        help='PDB IDs to exclude from training and validation datasets')

    # Confidence loss
    parser.add_argument('--rmsd_classification_cutoff', nargs='+', type=float, default=2.0,
                        help='RMSD cutoff (Å) for positive label')
    parser.add_argument('--balance', action='store_true', default=False,
                        help='Balance positive/negative samples per batch')
    parser.add_argument('--rmsd_prediction', action='store_true', default=False,
                        help='Regress RMSD instead of binary classification')

    # Model architecture (loaded from pretrained confidence config)
    parser.add_argument('--ns', type=int, default=16)
    parser.add_argument('--nv', type=int, default=4)
    parser.add_argument('--num_conv_layers', type=int, default=2)
    parser.add_argument('--num_prot_emb_layers', type=int, default=0)
    parser.add_argument('--sigma_embed_dim', type=int, default=32)
    parser.add_argument('--distance_embed_dim', type=int, default=32)
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32)
    parser.add_argument('--max_radius', type=float, default=5.0)
    parser.add_argument('--cross_max_distance', type=float, default=80.0)
    parser.add_argument('--dynamic_max_cross', action='store_true', default=False)
    parser.add_argument('--sh_lmax', type=int, default=2)
    parser.add_argument('--use_second_order_repr', action='store_true')
    parser.add_argument('--reduce_pseudoscalars', action='store_true')
    parser.add_argument('--embed_also_ligand', action='store_true')
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--confidence_dropout', type=float, default=0.0)
    parser.add_argument('--confidence_no_batchnorm', action='store_true')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True)
    parser.add_argument('--no_torsion', action='store_true')
    parser.add_argument('--all_atoms', action='store_true', default=True,
                        help='Use all-atom AAModel for confidence (default: True)')
    parser.add_argument('--embedding_type', type=str, default='sinusoidal')
    parser.add_argument('--embedding_scale', type=int, default=10000)

    # Diffusion sigma params (needed for t_to_sigma, even though confidence uses t=0)
    parser.add_argument('--tr_sigma_min', type=float, default=0.1)
    parser.add_argument('--tr_sigma_max', type=float, default=19.0)
    parser.add_argument('--rot_sigma_min', type=float, default=0.03)
    parser.add_argument('--rot_sigma_max', type=float, default=1.55)
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314)
    parser.add_argument('--tor_sigma_max', type=float, default=3.14)

    # Training
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w_decay', type=float, default=0.0)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (default: 1.0)')
    parser.add_argument('--scheduler', type=str, default='plateau')
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--ema_rate', type=float, default=0.999)
    parser.add_argument('--freeze_pretrained_epochs', type=int, default=0,
                        help='Freeze pretrained params for N warmup epochs')

    # LoRA
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default=None)

    # Logging
    parser.add_argument('--run_name', type=str, default='em_confidence_finetune')
    parser.add_argument('--log_dir', type=str, default='workdir_em_confidence')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='diffdock-em-confidence')
    parser.add_argument('--model_save_frequency', type=int, default=0)
    parser.add_argument('--best_model_save_frequency', type=int, default=0)
    parser.add_argument('--main_metric', type=str, default='accuracy',
                        help='Metric for early stopping: loss | accuracy | ROC AUC')
    parser.add_argument('--main_metric_goal', type=str, default='max',
                        help='min | max')
    parser.add_argument('--rerank_eval_frequency', type=int, default=5,
                        help='Run true reranking evaluation every N epochs (0 to disable)')

    # Pairwise ranking loss — for learning within-complex ranking directly
    # (instead of the default pose-level BCE, which produces the same ranking
    # as the pretrained orig confidence model on EM data).
    parser.add_argument('--pairwise_loss', action='store_true',
                        help='Use pairwise margin ranking loss instead of pose-level BCE. '
                             'Each training sample is a (good, bad) pose pair from the same '
                             'complex; loss forces good_score > bad_score + margin.')
    parser.add_argument('--pairwise_margin', type=float, default=1.0,
                        help='Margin for pairwise ranking loss (default: 1.0)')
    parser.add_argument('--pairwise_bce_weight', type=float, default=0.1,
                        help='Auxiliary BCE weight added to pairwise loss to keep score '
                             'calibration sane (default: 0.1; set to 0 for pure ranking)')
    parser.add_argument('--pairwise_min_rmsd_gap', type=float, default=1.0,
                        help='Minimum RMSD gap (Å) between the two poses in a pair. '
                             'Pairs are formed by continuous RMSD ranking (not the binary '
                             '<2Å/≥2Å cutoff), so complexes with all-good or all-bad poses '
                             'still contribute. Lower values → more pairs available but '
                             'noisier ranking signal (default: 1.0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume, or "auto"')
    parser.add_argument('--preprocessing_only', action='store_true', default=False,
                        help='Only run dataset preprocessing (cache poses), then exit. '
                             'Used for parallel cache creation with --cache_creation_id.')

    return parser


# ---------------------------------------------------------------------------
# BatchNorm helpers
# ---------------------------------------------------------------------------

_BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def _lock_bn_frozen(model, verbose=False):
    """Permanently lock all BN layers so running stats cannot be written.

    Three overlapping safeguards — any one is enough, together they are belt-
    and-braces against the 'poisoned BN stats' failure mode we saw last run:

      1. eval() — BN reads stored running_mean/running_var, not batch stats.
      2. momentum = 0.0 — EMA update rule becomes `new = old`. Even if train()
         is accidentally called later, running stats cannot shift. This is
         the strongest protection.
      3. Sanity reset — if pretrained/resumed stats already contain NaN/Inf
         or negative variance, replace with canonical (0, 1) instead of
         silently propagating poison. Prefer clean-ish stats over poisoned.

    Call once right after model construction, and again at the start of each
    train/test epoch (model.train() resets BN submodules to training mode).
    """
    m = model.module if hasattr(model, 'module') else model
    n_locked = n_reset = 0
    for name, mod in m.named_modules():
        if not isinstance(mod, _BN_TYPES):
            continue
        bad = (
            not torch.isfinite(mod.running_mean).all()
            or not torch.isfinite(mod.running_var).all()
            or (mod.running_var < 0).any()
        )
        if bad:
            if verbose:
                print(f'[BN] Resetting poisoned stats for {name} → mean=0, var=1')
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
            if mod.num_batches_tracked is not None:
                mod.num_batches_tracked.zero_()
            n_reset += 1
        mod.eval()
        mod.momentum = 0.0
        n_locked += 1
    return n_locked, n_reset


def _assert_bn_finite(model, ctx=''):
    """Return True iff every BN buffer is finite. Prints offenders if not."""
    m = model.module if hasattr(model, 'module') else model
    bad = []
    for name, mod in m.named_modules():
        if not isinstance(mod, _BN_TYPES):
            continue
        if not torch.isfinite(mod.running_mean).all():
            bad.append(f'{name}.running_mean')
        if not torch.isfinite(mod.running_var).all() or (mod.running_var < 0).any():
            bad.append(f'{name}.running_var')
    if bad:
        print(f'[BN] Non-finite BN buffers detected ({ctx}): {bad[:10]}'
              f'{" …" if len(bad) > 10 else ""}')
        return False
    return True


# Back-compat alias kept so existing call sites still work.
_freeze_bn = _lock_bn_frozen



# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _conf_loss_per_sample(pred, data_list, device, rmsd_prediction, rmsd_classification_cutoff):
    """Compute per-sample confidence loss (reduction='none').

    data_list: always a Python list of individual HeteroData graphs.
    Returns: loss tensor of shape (N,)
    """
    if isinstance(pred, tuple):
        pred = pred[0]
    if rmsd_prediction:
        labels = torch.cat([g.rmsd for g in data_list]).to(device)
        return F.mse_loss(pred, labels, reduction='none')
    elif isinstance(rmsd_classification_cutoff, list):
        labels = torch.cat([g.y_binned for g in data_list]).to(device)
        return F.cross_entropy(pred, labels, reduction='none')
    else:
        labels = torch.cat([g.y for g in data_list]).to(device)
        return F.binary_cross_entropy_with_logits(pred, labels, reduction='none')



# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, rmsd_prediction, rmsd_classification_cutoff,
                ema_weights=None, max_grad_norm=1.0):
    model.train()
    # Keep all BN layers in eval mode throughout training.
    # BN in eval mode uses frozen pretrained running stats and never updates them,
    # so NaN batches cannot corrupt BN state.  Scores produced at inference will
    # be normalised by the same global baseline → reliable inter-complex ranking.
    _freeze_bn(model)

    meter = AverageMeter(['confidence_loss'])
    nan_log = os.environ.get('NAN_LOG_FILE', 'nan_complexes.txt')

    for data in tqdm(loader, total=len(loader)):
        if isinstance(data, list):
            if len(data) < 2:
                continue
            data = [d.to(device) for d in data]
        else:
            if data.num_graphs < 2:
                continue
            data = data.to(device)

        optimizer.zero_grad()
        try:
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]  # AAModel returns (confidence, atom_confidence)

            # ── Step 1: check NaN in raw model output ──────────────────────
            if not torch.isfinite(pred).all():
                names = [d.name for d in data] if isinstance(data, list) else [data.name]
                print(f'Train: NaN in raw model output for {names}, skipping batch')
                with open(nan_log, 'a') as f:
                    for name in names:
                        f.write(f'{name}\tnan_output\n')
                optimizer.zero_grad()
                continue

            # ── Step 2: per-sample loss with soft clamping ─────────────────
            data_list = data if isinstance(data, list) else [data]
            per_sample_loss = _conf_loss_per_sample(
                pred, data_list, device, rmsd_prediction, rmsd_classification_cutoff
            )
            # Soft cap: noisy samples still train but at reduced weight
            loss = per_sample_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()

            # ── Step 3: check bad batch loss ───────────────────────────────
            loss_is_bad = (
                not torch.isfinite(per_sample_loss).all() or
                per_sample_loss.abs().max().item() > CONF_BATCH_LOSS_MAX
            )
            if loss_is_bad:
                names = [d.name for d in data_list]
                print(f'Train: bad loss for {names} '
                      f'(max={per_sample_loss.abs().max().item():.3e}), skipping batch')
                with open(nan_log, 'a') as f:
                    for name in names:
                        f.write(f'{name}\tbad_loss\n')
                optimizer.zero_grad()
                continue

            # ── Step 4: normal backward + grad clip ───────────────────────
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if has_nan_grad:
                names = [d.name for d in data_list]
                print(f'WARNING: NaN/Inf gradient for {names}, skipping batch')
                optimizer.zero_grad()
                continue

            optimizer.step()
            has_nan_param = any(
                not torch.isfinite(p).all()
                for p in model.parameters()
            )
            has_large_param = any(
                p.abs().max().item() > PARAM_MAX_THRESHOLD
                for p in model.parameters()
                if p.requires_grad
            )
            if has_nan_param or has_large_param:
                names = [d.name for d in data_list]
                reason = 'NaN/Inf' if has_nan_param else f'large (>{PARAM_MAX_THRESHOLD:.0e})'
                print(f'WARNING: {reason} model params after optimizer.step() for {names}, '
                      'restoring from EMA and skipping EMA update')
                optimizer.zero_grad()
                if ema_weights is not None:
                    ema_weights.copy_to(model.parameters())
                    optimizer.state.clear()  # clear Adam momentum to prevent re-corruption
                # BN is frozen (eval mode) so no BN state needs restoring
            else:
                if ema_weights is not None:
                    ema_weights.update(model.parameters())
                meter.add([loss.cpu().detach()])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: OOM, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                gc.collect()
            elif 'Input mismatch' in str(e):
                print('WARNING: torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
            else:
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

    return meter.summary()


def train_epoch_pairwise(model, loader, optimizer, device, margin=1.0,
                         bce_weight=0.1, ema_weights=None, max_grad_norm=1.0):
    """Pairwise ranking training step.

    Each item in ``loader`` is a flat list [g0_good, g0_bad, g1_good, g1_bad, ...]
    (produced by pairwise_collate). Forward all 2N graphs through the model, then
    split predictions into good_scores (even indices) and bad_scores (odd indices)
    and apply margin_ranking_loss.

    Optionally adds a small BCE term to keep scores calibrated (prevents the
    model from collapsing to arbitrary absolute values).
    """
    model.train()
    _freeze_bn(model)

    meter = AverageMeter(['confidence_loss', 'rank_acc'])
    nan_log = os.environ.get('NAN_LOG_FILE', 'nan_complexes.txt')

    for data in tqdm(loader, total=len(loader)):
        if isinstance(data, list):
            if len(data) < 4 or len(data) % 2 != 0:
                # Need at least 2 pairs for a meaningful batch, even count required.
                continue
            data = [d.to(device) for d in data]
            data_list = data
        else:
            # DataLoader fallback: already-batched flat graphs; split by batch.
            if data.num_graphs < 4 or data.num_graphs % 2 != 0:
                continue
            data = data.to(device)
            data_list = data.to_data_list() if hasattr(data, 'to_data_list') else [data]

        optimizer.zero_grad()
        try:
            pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.view(-1)

            if not torch.isfinite(pred).all():
                names = [getattr(d, 'name', '?') for d in data_list]
                print(f'Pairwise: NaN in raw model output for {names}, skipping batch')
                with open(nan_log, 'a') as f:
                    for name in names:
                        f.write(f'{name}\tnan_output\n')
                optimizer.zero_grad()
                continue

            # Split predictions: even indices = good, odd indices = bad
            good_scores = pred[0::2]
            bad_scores  = pred[1::2]
            if good_scores.shape != bad_scores.shape or good_scores.numel() == 0:
                continue

            # Margin ranking: target=1 means good should exceed bad by margin
            target = torch.ones_like(good_scores)
            rank_loss = F.margin_ranking_loss(good_scores, bad_scores, target,
                                              margin=margin, reduction='none')
            # Clamp to prevent a single degenerate pair from dominating the batch
            rank_loss_mean = rank_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()

            # Optional auxiliary BCE for score calibration
            if bce_weight > 0:
                good_y = torch.ones_like(good_scores)
                bad_y  = torch.zeros_like(bad_scores)
                labels = torch.cat([good_y, bad_y])
                preds  = torch.cat([good_scores, bad_scores])
                bce = F.binary_cross_entropy_with_logits(
                    preds, labels, reduction='none'
                ).clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()
                loss = rank_loss_mean + bce_weight * bce
            else:
                loss = rank_loss_mean

            if not torch.isfinite(loss):
                names = [getattr(d, 'name', '?') for d in data_list]
                print(f'Pairwise: bad loss for {names}, skipping batch')
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if has_nan_grad:
                print(f'WARNING: NaN/Inf gradient, skipping batch')
                optimizer.zero_grad()
                continue

            optimizer.step()
            has_nan_param = any(not torch.isfinite(p).all() for p in model.parameters())
            has_large_param = any(
                p.abs().max().item() > PARAM_MAX_THRESHOLD
                for p in model.parameters() if p.requires_grad
            )
            if has_nan_param or has_large_param:
                reason = 'NaN/Inf' if has_nan_param else f'large (>{PARAM_MAX_THRESHOLD:.0e})'
                print(f'WARNING: {reason} params after step, restoring from EMA')
                optimizer.zero_grad()
                if ema_weights is not None:
                    ema_weights.copy_to(model.parameters())
                    optimizer.state.clear()
            else:
                if ema_weights is not None:
                    ema_weights.update(model.parameters())
                # Ranking accuracy: fraction of pairs where good > bad
                rank_acc = (good_scores.detach() > bad_scores.detach()).float().mean()
                meter.add([loss.cpu().detach(), rank_acc.cpu().detach()])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: OOM, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                gc.collect()
            else:
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()

    return meter.summary()


def test_epoch(model, loader, device, rmsd_prediction, rmsd_classification_cutoff):
    # Use train mode + frozen BN for validation.
    # model.eval() causes all-NaN val outputs in practice (likely due to e3nn/backbone
    # numerical behaviour differences in eval mode). With _freeze_bn, BN uses the same
    # frozen pretrained running stats as during train_epoch, making metrics consistent.
    model.train()
    _freeze_bn(model)

    if rmsd_prediction:
        meter = AverageMeter(['loss'], unpooled_metrics=True)
    else:
        meter = AverageMeter(['confidence_loss', 'accuracy'], unpooled_metrics=True)

    all_labels = []
    all_preds = []
    # Per-complex tracking for top-1 hit rate (the actual reranking metric we care about)
    complex_best = {}   # name -> (best_score, label_at_best_score)

    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():
                pred = model(data)
            if isinstance(pred, tuple):
                pred = pred[0]

            if not torch.isfinite(pred).all():
                # Batch has NaN — try each complex individually to salvage valid predictions.
                data_list = data if isinstance(data, list) else [data]
                core = model.module if hasattr(model, 'module') else model
                salvaged_preds, salvaged_data = [], []
                for d in data_list:
                    try:
                        with torch.no_grad():
                            p = core(d.to(device))
                        if isinstance(p, tuple):
                            p = p[0]
                        if torch.isfinite(p).all():
                            salvaged_preds.append(p)
                            salvaged_data.append(d)
                    except Exception:
                        pass
                if not salvaged_preds:
                    print('WARNING: NaN/Inf in val batch, all complexes failed, skipping')
                    continue
                print(f'WARNING: NaN/Inf in val batch, salvaged {len(salvaged_preds)}/{len(data_list)} complexes')
                pred = torch.cat(salvaged_preds, dim=0)
                data = salvaged_data

            if rmsd_prediction:
                labels = (torch.cat([g.rmsd for g in data]) if isinstance(data, list)
                          else data.rmsd).to(device)
                per_sample_loss = _conf_loss_per_sample(
                    pred, data if isinstance(data, list) else [data],
                    device, rmsd_prediction, rmsd_classification_cutoff,
                )
                loss = per_sample_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()
                if torch.isfinite(loss):
                    meter.add([loss.cpu().detach()])
            else:
                data_list = data if isinstance(data, list) else [data]
                if isinstance(rmsd_classification_cutoff, list):
                    labels = torch.cat([g.y_binned for g in data_list]).to(device)
                    per_sample_loss = _conf_loss_per_sample(
                        pred, data_list, device, rmsd_prediction, rmsd_classification_cutoff,
                    )
                    loss = per_sample_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()
                    accuracy = torch.tensor(0.0)
                else:
                    labels = torch.cat([g.y for g in data_list]).to(device)
                    per_sample_loss = _conf_loss_per_sample(
                        pred, data_list, device, rmsd_prediction, rmsd_classification_cutoff,
                    )
                    loss = per_sample_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()
                    accuracy = torch.mean((labels == (pred > 0).float()).float())

                if torch.isfinite(loss):
                    meter.add([loss.cpu().detach(), accuracy.cpu().detach()])
                    all_preds.append(pred.detach().cpu())
                    all_labels.append(labels.cpu())

                    # Track per-complex best score for top-1 hit rate
                    pred_cpu = pred.detach().cpu()
                    labels_cpu = labels.cpu()
                    for i, g in enumerate(data_list):
                        name = getattr(g, 'name', f'unknown_{i}')
                        score = float(pred_cpu[i].item())
                        label = float(labels_cpu[i].item())
                        if name not in complex_best or score > complex_best[name][0]:
                            complex_best[name] = (score, label)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('WARNING: OOM during eval, skipping batch')
                torch.cuda.empty_cache()
            else:
                raise e

    if not all_labels:
        print('WARNING: No valid validation samples (all batches had NaN/Inf predictions). '
              'Returning dummy metrics.')
        return {'confidence_loss': float('nan'), 'accuracy': 0.0,
                'ROC AUC': 0.0, 'top1_hit_rate': 0.0, 'baseline_metric': 0.0}
    all_labels = torch.cat(all_labels)
    if rmsd_prediction:
        baseline = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline = all_labels.sum() / len(all_labels)

    results = meter.summary()
    results['baseline_metric'] = baseline

    # Compute ROC AUC over full validation set (not per-batch average)
    if not rmsd_prediction and all_preds and not isinstance(rmsd_classification_cutoff, list):
        all_preds_cat = torch.cat(all_preds).numpy()
        try:
            results['ROC AUC'] = roc_auc_score(all_labels.numpy(), all_preds_cat)
        except ValueError:
            results['ROC AUC'] = 0.0
    else:
        results['ROC AUC'] = 0.0

    # Per-complex top-1 hit rate: for each complex, does the top-scored pose
    # have RMSD < 2 Å?  This directly measures reranking quality.
    if complex_best and not rmsd_prediction:
        hits = sum(1 for (_, l) in complex_best.values() if l > 0.5)
        results['top1_hit_rate'] = hits / len(complex_best)
    else:
        results['top1_hit_rate'] = 0.0

    return results


# ---------------------------------------------------------------------------
# True reranking evaluation (score ALL poses per complex, 8-GPU parallel)
# ---------------------------------------------------------------------------

class _RerankDataset(torch.utils.data.Dataset):
    """Flat iteration over (complex, pose) pairs for reranking evaluation.

    Each ``__getitem__`` returns a single HeteroData graph with one pose placed,
    ready for the model. RMSD and complex name are attached for per-entry
    tracking via the dataset index (index-based tracking is used downstream
    instead of graph attributes, which are fragile through Batch.from_data_list).
    """
    def __init__(self, em_dataset):
        self.em_dataset = em_dataset
        self.entries = []    # list of (complex_name, pose_idx_within_good)
        self.entry_rmsds = []
        for name in em_dataset.dataset_names:
            _, rmsds = em_dataset.positions_rmsds_dict[name]
            good_idx = em_dataset.good_indices_dict[name]
            good_rmsds = np.asarray(rmsds)[good_idx]
            for pi in range(len(good_idx)):
                self.entries.append((name, pi))
                self.entry_rmsds.append(float(good_rmsds[pi]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        name, pi = self.entries[idx]
        positions, _ = self.em_dataset.positions_rmsds_dict[name]
        good_idx = self.em_dataset.good_indices_dict[name]
        good_positions = positions[good_idx]

        g = copy.deepcopy(self.em_dataset.complex_graph_dict[name])
        g['ligand'].pos = torch.from_numpy(good_positions[pi])
        zeros = lambda n: torch.zeros(n)
        g['ligand'].node_t = {
            k: zeros(g['ligand'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        g['receptor'].node_t = {
            k: zeros(g['receptor'].num_nodes) for k in ('tr', 'rot', 'tor')
        }
        if self.em_dataset.all_atoms and hasattr(g, 'atom') and hasattr(g['atom'], 'x'):
            g['atom'].node_t = {
                k: zeros(g['atom'].num_nodes) for k in ('tr', 'rot', 'tor')
            }
        g.complex_t = {k: zeros(1) for k in ('tr', 'rot', 'tor')}
        return g


def rerank_evaluation(model, dataset, device, rmsd_cutoff=2.0, batch_size=8, num_workers=4):
    """Score all cached poses for each complex, pick highest-scored, check RMSD.

    Unlike test_epoch() which samples ONE random pose per complex, this
    evaluates true reranking quality: all ~40 cached poses per complex are
    scored, and the top-1 is checked against ``rmsd_cutoff``.

    Uses the same DataListLoader + DataParallel pattern as training — poses
    are sharded across all available GPUs for parallel forward passes.
    """
    rerank_ds = _RerankDataset(dataset)
    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    loader_class = DataListLoader if use_multi_gpu else DataLoader
    loader = loader_class(
        rerank_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
    )
    core = model.module if hasattr(model, 'module') else model

    # name -> (best_score, best_rmsd)
    complex_best = {}
    # Track by global index so we don't rely on name/rmsd attribute preservation.
    global_idx = 0

    for data in tqdm(loader, desc='Reranking'):
        try:
            if isinstance(data, list):
                # DataListLoader: list of graphs; DataParallel handles internal batching.
                data = [d.to(device) for d in data]
                with torch.no_grad():
                    pred = model(data)
                batch_len = len(data)
            else:
                # Single-GPU DataLoader path: Batch.from_data_list strips dict
                # attributes (complex_t, node_t), so restore them post-batch.
                data = data.to(device)
                data.complex_t = {
                    k: torch.zeros(data.num_graphs, device=device)
                    for k in ('tr', 'rot', 'tor')
                }
                for nt in data.node_types:
                    n = data[nt].num_nodes
                    data[nt].node_t = {
                        k: torch.zeros(n, device=device) for k in ('tr', 'rot', 'tor')
                    }
                with torch.no_grad():
                    pred = core(data)
                batch_len = data.num_graphs

            if isinstance(pred, tuple):
                pred = pred[0]
            pred_cpu = pred.detach().cpu().view(-1)

            for i in range(batch_len):
                entry_idx = global_idx + i
                if entry_idx >= len(rerank_ds.entries):
                    break
                name, _ = rerank_ds.entries[entry_idx]
                rmsd = rerank_ds.entry_rmsds[entry_idx]
                score = float(pred_cpu[i].item())
                if name not in complex_best or score > complex_best[name][0]:
                    complex_best[name] = (score, rmsd)
            global_idx += batch_len

        except RuntimeError as e:
            if 'out of memory' in str(e):
                torch.cuda.empty_cache()
                print('WARNING: OOM during reranking, skipping batch')
                # Still advance global_idx to keep indexing consistent
                # (use expected batch size; last partial batch handled by len check)
                global_idx += batch_size
                continue
            raise

    hits = sum(1 for (_, rmsd) in complex_best.values() if rmsd < rmsd_cutoff)
    total = len(complex_best)
    top1_rmsds = [rmsd for (_, rmsd) in complex_best.values()]

    return {
        'rerank_top1_hit_rate': hits / max(total, 1),
        'rerank_top1_median_rmsd': float(np.median(top1_rmsds)) if top1_rmsds else float('inf'),
        'rerank_top1_mean_rmsd': float(np.mean(top1_rmsds)) if top1_rmsds else float('inf'),
        'rerank_total_complexes': total,
        'rerank_hits': hits,
    }


# ---------------------------------------------------------------------------
# Optimizer with differential LR for ring params
# ---------------------------------------------------------------------------

def get_optimizer(model, base_lr, ring_lr_multiplier, weight_decay=0.0, use_lora=False):
    ring_params, lora_params, pretrained_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'atom_ring_feature_embedding' in name or 'atom_ring_vector_embedding' in name \
                or 'ring_feature_embedding' in name or 'ring_vector_embedding' in name:
            ring_params.append(param)
        elif 'lora_' in name:
            lora_params.append(param)
        else:
            pretrained_params.append(param)

    groups = []
    if pretrained_params:
        groups.append({'params': pretrained_params, 'lr': base_lr, 'name': 'pretrained'})
    if ring_params:
        groups.append({'params': ring_params, 'lr': base_lr * ring_lr_multiplier, 'name': 'ring'})
    if lora_params:
        groups.append({'params': lora_params, 'lr': base_lr * ring_lr_multiplier, 'name': 'lora'})

    if not groups:
        raise ValueError('No trainable parameters found.')
    return torch.optim.Adam(groups, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        config_dict = yaml.safe_load(args.config)
        for key, value in config_dict.items():
            if isinstance(value, list) and isinstance(getattr(args, key, None), list):
                getattr(args, key).extend(value)
            else:
                setattr(args, key, value)

    assert args.main_metric_goal in ('max', 'min')

    # Override --remove_hs with --keep_hs
    if args.keep_hs:
        args.remove_hs = False

    # --- Load pretrained confidence model architecture config ---
    if args.pretrained_confidence_config:
        conf_cfg_path = args.pretrained_confidence_config
    else:
        # Auto-detect from parent dir of the checkpoint
        for cfg_name in ('model_parameters.yml', 'config.yml'):
            candidate = os.path.join(
                os.path.dirname(args.pretrained_confidence_model), cfg_name
            )
            if os.path.exists(candidate):
                conf_cfg_path = candidate
                break
        else:
            conf_cfg_path = None

    if conf_cfg_path and os.path.exists(conf_cfg_path):
        print(f'Loading pretrained confidence model config from {conf_cfg_path}')
        with open(conf_cfg_path) as f:
            conf_cfg = yaml.safe_load(f)
        # Copy architecture params (these must match the pretrained weights)
        arch_keys = [
            'ns', 'nv', 'num_conv_layers', 'num_prot_emb_layers',
            'sigma_embed_dim', 'distance_embed_dim', 'cross_distance_embed_dim',
            'max_radius', 'cross_max_distance', 'dynamic_max_cross',
            'sh_lmax', 'use_second_order_repr', 'reduce_pseudoscalars',
            'embed_also_ligand', 'no_batch_norm', 'dropout',
            'scale_by_sigma', 'no_torsion', 'all_atoms',
            'embedding_type', 'embedding_scale',
            'tr_sigma_min', 'tr_sigma_max',
            'rot_sigma_min', 'rot_sigma_max',
            'tor_sigma_min', 'tor_sigma_max',
            'not_fixed_center_conv', 'no_differentiate_convolutions', 'tp_weights_layers',
            'rmsd_classification_cutoff', 'atom_rmsd_classification_cutoff',
        ]
        for key in arch_keys:
            if key in conf_cfg:
                setattr(args, key, conf_cfg[key])
        # Ensure all_atoms=True for AAModel
        if 'all_atoms' in conf_cfg:
            args.all_atoms = conf_cfg['all_atoms']
        else:
            args.all_atoms = True
    else:
        print('WARNING: No pretrained confidence config found, using default args.')
        args.all_atoms = True

    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_yaml_file(os.path.join(run_dir, 'model_parameters.yml'), vars(args))

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # --- Build datasets ---
    common_dataset_args = dict(
        data_dir=args.data_dir,
        ring_features_dir=args.ring_features_dir,
        cache_path=args.cache_path,
        score_model_dir=args.score_model_dir,
        score_model_ckpt=args.score_model_ckpt,
        inference_steps=args.inference_steps,
        samples_per_complex=args.samples_per_complex,
        all_atoms=args.all_atoms,
        receptor_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors,
        ring_dim=args.ring_dim,
        ring_vec_dim=args.ring_vec_dim,
        remove_hs=args.remove_hs,
        max_lig_size=args.max_lig_size,
        matching=not args.no_torsion,
        popsize=args.matching_popsize,
        maxiter=args.matching_maxiter,
        num_conformers=args.num_conformers,
        knn_only_graph=not args.not_knn_only_graph,
        esm_embeddings_path=args.esm_embeddings_path,
        protein_file=args.protein_file,
        num_workers=args.num_workers,
        rmsd_classification_cutoff=args.rmsd_classification_cutoff,
        balance=args.balance,
        limit_complexes=args.limit_complexes,
        cache_creation_id=args.cache_creation_id,
        cache_ids_to_combine=args.cache_ids_to_combine,
        device=device,
        exclude_complexes=args.exclude_complexes,
    )

    print('Building training dataset...')
    train_dataset = EMConfidenceDataset(split_path=args.split_train, **common_dataset_args)
    print('Building validation dataset...')
    val_dataset = EMConfidenceDataset(split_path=args.split_val, **common_dataset_args)

    if args.preprocessing_only:
        print('Preprocessing complete. Exiting (--preprocessing_only).')
        return

    use_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
    loader_class = DataListLoader if use_multi_gpu else DataLoader

    if args.pairwise_loss:
        # Pairwise training: each sample is a (good, bad) pose pair from the
        # same complex. Use a plain torch DataLoader with pairwise_collate so we
        # get a flat list [g0_good, g0_bad, g1_good, g1_bad, ...] — compatible
        # with DataParallel (which expects a list of graphs).
        pair_train_dataset = EMConfidencePairDataset(
            train_dataset,
            rmsd_cutoff=(args.rmsd_classification_cutoff[0]
                         if isinstance(args.rmsd_classification_cutoff, list)
                         else args.rmsd_classification_cutoff),
            min_rmsd_gap=args.pairwise_min_rmsd_gap,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=pair_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
            collate_fn=pairwise_collate,
        )
        print(f'Pairwise training: batch_size={args.batch_size} pairs '
              f'= {2 * args.batch_size} graphs/batch  |  '
              f'margin={args.pairwise_margin}  bce_weight={args.pairwise_bce_weight}')
    else:
        train_loader = loader_class(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
        )

    # Val loader always uses the pose-level dataset — test_epoch samples
    # one random pose per complex for loss/ROC, and rerank_evaluation scores
    # ALL poses for the true top-1 metric.
    val_loader = loader_class(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Build model ---
    t_to_sigma = partial(t_to_sigma_compl, args=args)

    model = get_em_model(
        args,
        device,
        t_to_sigma,
        ring_dim=args.ring_dim,
        ring_vec_dim=args.ring_vec_dim,
        no_parallel=True,
        confidence_mode=True,
    )

    # Load pretrained AA confidence weights (ring layers initialize near-zero)
    print(f'Loading pretrained confidence weights from {args.pretrained_confidence_model}')
    missing, unexpected = load_pretrained_weights(model, args.pretrained_confidence_model, strict=False)
    ring_keys = [k for k in missing if 'ring' in k]
    if ring_keys:
        print(f'New ring embedding parameters (will be trained): {ring_keys}')

    # Apply LoRA if requested
    if args.use_lora:
        print(f'Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...')
        target_modules = (
            set(m.strip() for m in args.lora_target_modules.split(','))
            if args.lora_target_modules else None
        )
        model, _, _ = apply_lora_to_model(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            verbose=True,
        )
        freeze_non_lora_params(model, verbose=True)
        # confidence_predictor is randomly initialized (not in pretrained weights),
        # so it must always be trainable regardless of LoRA freeze state.
        for name, param in model.named_parameters():
            if 'confidence_predictor' in name:
                param.requires_grad = True
        print_trainable_params(model)

    model = model.to(device)
    if use_multi_gpu:
        from torch_geometric.nn.data_parallel import DataParallel
        model = DataParallel(model)

    # Lock BN before any forward can happen: resets poisoned pretrained stats
    # (should be none here) and sets momentum=0 so running stats are write-once
    # from this point on. _lock_bn_frozen is idempotent; we re-call it each
    # epoch inside train_epoch/test_epoch to re-enforce eval mode after
    # model.train().
    n_locked, n_reset = _lock_bn_frozen(model, verbose=True)
    print(f'[BN] Locked {n_locked} BatchNorm layers (reset {n_reset} poisoned at load)')
    if not _assert_bn_finite(model, ctx='post-load'):
        raise RuntimeError('BN buffers contain NaN/Inf after pretrained load + reset. Aborting.')

    numel = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {numel:,} total params, {trainable:,} trainable '
          f'({100 * trainable / numel:.2f}%)')

    optimizer = get_optimizer(
        model.module if hasattr(model, 'module') else model,
        base_lr=args.lr,
        ring_lr_multiplier=args.ring_lr_multiplier,
        weight_decay=args.w_decay,
        use_lora=args.use_lora,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min' if args.main_metric_goal == 'min' else 'max',
        factor=0.7,
        patience=args.scheduler_patience,
        min_lr=args.lr / 100,
    ) if args.scheduler == 'plateau' else None

    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == 'auto':
            ckpt_path = os.path.join(run_dir, 'last_model.pt')
        if os.path.exists(ckpt_path):
            print(f'Resuming from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=device)
            m = model.module if hasattr(model, 'module') else model
            m.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if scheduler and ckpt.get('scheduler'):
                scheduler.load_state_dict(ckpt['scheduler'])
            if 'ema_weights' in ckpt:
                ema_weights.load_state_dict(ckpt['ema_weights'], device=device)
            start_epoch = ckpt['epoch'] + 1
            print(f'Resumed at epoch {ckpt["epoch"]}, continuing from {start_epoch}')

    # --- Training loop ---
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0.0
    best_epoch = 0

    print('Starting confidence model finetuning...')
    for epoch in range(start_epoch, args.n_epochs):

        # Warmup: optionally freeze pretrained params
        # Works for both LoRA and non-LoRA modes:
        #   Phase 1 (epoch < freeze_pretrained_epochs): only ring params trainable
        #   Phase 2 (epoch >= freeze_pretrained_epochs): ring + LoRA (or all) trainable
        if args.freeze_pretrained_epochs > 0:
            if epoch < args.freeze_pretrained_epochs:
                for name, param in model.named_parameters():
                    param.requires_grad = ('ring' in name or 'confidence_predictor' in name)
                if epoch == 0:
                    print(f'Ring+head warmup: freezing all non-ring/non-head params for '
                          f'{args.freeze_pretrained_epochs} epochs')
                    # Reinit EMA to match new requires_grad state.
                    # EMA was initialized before warmup freeze, so it tracked LoRA params too.
                    # After freeze, LoRA requires_grad=False → EMA.update() would zip shadow
                    # (includes LoRA) against current params (ring+head only) → size mismatch.
                    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
            elif epoch == args.freeze_pretrained_epochs:
                print(f'Unfreezing at epoch {epoch}')
                if args.use_lora:
                    # Re-apply LoRA freeze: non-LoRA non-ring params stay frozen
                    for name, param in model.named_parameters():
                        param.requires_grad = ('ring' in name or 'lora_' in name
                                               or 'confidence_predictor' in name)
                else:
                    for param in model.parameters():
                        param.requires_grad = True
                # Reinit EMA after unfreeze so shadow_params matches new requires_grad set.
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

        if args.pairwise_loss:
            train_metrics = train_epoch_pairwise(
                model, train_loader, optimizer, device,
                margin=args.pairwise_margin, bce_weight=args.pairwise_bce_weight,
                ema_weights=ema_weights, max_grad_norm=args.grad_clip_norm,
            )
            print(f'Epoch {epoch}: Train pairwise_loss {train_metrics["confidence_loss"]:.4f}  '
                  f'rank_acc {train_metrics["rank_acc"]:.4f}')
        else:
            train_metrics = train_epoch(
                model, train_loader, optimizer, device,
                args.rmsd_prediction, args.rmsd_classification_cutoff,
                ema_weights=ema_weights, max_grad_norm=args.grad_clip_norm,
            )
            print(f'Epoch {epoch}: Train confidence_loss {train_metrics["confidence_loss"]:.4f}')

        # EMA validation
        ema_weights.store(model.parameters())
        ema_weights.copy_to(model.parameters())
        # Safety check: if EMA params OR BN buffers have NaN/Inf, fall back
        ema_has_nan = any(
            not torch.isfinite(p).all()
            for p in model.parameters()
        ) or any(
            not torch.isfinite(buf).all()
            for m in model.modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d))
            for buf in (m.running_mean, m.running_var)
        )
        if ema_has_nan:
            print(f'WARNING: EMA weights or BN buffers contain NaN/Inf at epoch {epoch}, '
                  'falling back to current model weights for validation.')
            ema_weights.restore(model.parameters())

        val_metrics = test_epoch(
            model, val_loader, device,
            args.rmsd_prediction, args.rmsd_classification_cutoff,
        )
        if args.rmsd_prediction:
            print(f'Epoch {epoch}: Val loss {val_metrics["confidence_loss"]:.4f}')
        else:
            # Note: ``top1_hit_rate`` from test_epoch() is misleading — it only
            # scores ONE random pose per complex, so it tracks the positive-sample
            # rate (~32%), not reranking quality. The true reranking metric is
            # ``rerank_top1_hit_rate`` below, computed every ``rerank_eval_frequency``
            # epochs by scoring ALL poses per complex and taking the top-1.
            print(f'Epoch {epoch}: Val loss {val_metrics["confidence_loss"]:.4f}  '
                  f'accuracy {val_metrics["accuracy"]:.4f}  '
                  f'ROC AUC {val_metrics["ROC AUC"]:.4f}')

        # True reranking evaluation (score ALL poses per complex)
        if (args.rerank_eval_frequency > 0
                and (epoch + 1) % args.rerank_eval_frequency == 0):
            rerank_metrics = rerank_evaluation(
                model, val_dataset, device,
                rmsd_cutoff=(args.rmsd_classification_cutoff[0]
                             if isinstance(args.rmsd_classification_cutoff, list)
                             else args.rmsd_classification_cutoff),
            )
            print(f'Epoch {epoch}: Rerank top1_hit_rate {rerank_metrics["rerank_top1_hit_rate"]:.4f}  '
                  f'median_rmsd {rerank_metrics["rerank_top1_median_rmsd"]:.2f}  '
                  f'({rerank_metrics["rerank_hits"]}/{rerank_metrics["rerank_total_complexes"]} hits)')
            val_metrics.update(rerank_metrics)

        ema_state_dict = copy.deepcopy(
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        )
        ema_weights.restore(model.parameters())

        if args.wandb:
            logs = {}
            logs.update({f'train_{k}': v for k, v in train_metrics.items()})
            logs.update({f'val_{k}': v for k, v in val_metrics.items()})
            for i, pg in enumerate(optimizer.param_groups):
                logs[f'lr_{pg.get("name", i)}'] = pg['lr']
            wandb.log(logs, step=epoch + 1)

        # Scheduler / best-model use main_metric; if it's a rerank metric that
        # wasn't computed this epoch, fall back to ROC AUC for scheduler and
        # skip best-model update.
        metric_available = args.main_metric in val_metrics
        scheduler_metric = val_metrics.get(args.main_metric, val_metrics.get('ROC AUC', 0.0))
        if scheduler:
            scheduler.step(scheduler_metric)

        state_dict = (
            model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        )

        # Save best model
        improved = (
            metric_available
            and ((args.main_metric_goal == 'min' and val_metrics[args.main_metric] < best_val_metric)
                 or (args.main_metric_goal == 'max' and val_metrics[args.main_metric] > best_val_metric))
        )
        if improved:
            # Refuse to overwrite best_* with a poisoned checkpoint — momentum=0
            # should prevent BN drift entirely, but verify anyway so a single
            # bug can never silently corrupt the artifact used for inference.
            bn_ok = _assert_bn_finite(model, ctx=f'epoch {epoch} pre-save')
            if bn_ok:
                best_val_metric = val_metrics[args.main_metric]
                best_epoch = epoch
                torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))
                if args.use_lora:
                    lora_sd = get_lora_state_dict(
                        model.module if hasattr(model, 'module') else model
                    )
                    torch.save(lora_sd, os.path.join(run_dir, 'best_lora_weights.pt'))
            else:
                print(f'[BN] WARNING: skipping best_model save at epoch {epoch} '
                      'because BN buffers are non-finite. Investigate immediately.')

        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch+1}.pt'))
        if (args.best_model_save_frequency > 0
                and (epoch + 1) % args.best_model_save_frequency == 0):
            import shutil
            shutil.copyfile(
                os.path.join(run_dir, 'best_model.pt'),
                os.path.join(run_dir, f'best_model_epoch{epoch+1}.pt'),
            )

        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'ema_weights': ema_weights.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print(f'Best {args.main_metric} = {best_val_metric:.4f} at epoch {best_epoch}')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
