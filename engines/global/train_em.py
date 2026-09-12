"""Training script for DiffDock-EM finetuning.

Finetunes a pretrained DiffDock model with cryo-EM ring features.
Supports both full finetuning and LoRA-based parameter-efficient finetuning.
"""

import argparse
import copy
import math
import os
import random
import shutil
from functools import partial

import numpy as np
import torch
import yaml
import wandb

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
target_nofile = min(64000, rlimit[1])
if rlimit[0] < target_nofile:
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_nofile, rlimit[1]))
    except (ValueError, OSError):
        pass

from datasets.dataloader import DataLoader, DataListLoader

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.training import (
    BATCH_LOSS_MAX,
    VAL_LOSS_CLAMP,
    train_epoch,
    test_epoch,
    inference_epoch_fix,
    loss_function,
)
from utils.utils import save_yaml_file, get_em_model, load_pretrained_weights, ExponentialMovingAverage
from datasets.pdbbind import NoiseTransform
from em_dataset import EMDataset, print_statistics
from models.lora import (
    apply_lora_to_model,
    freeze_non_lora_params,
    get_lora_state_dict,
    print_trainable_params,
)


def get_parser():
    """Return the configured ArgumentParser (without calling parse_args).

    Shared by train_em.py and scan_nan_samples.py so both scripts always use
    the same architecture / dataset argument definitions.
    """
    parser = argparse.ArgumentParser(description='DiffDock-EM Finetuning')

    # Data arguments
    parser.add_argument('--train_split', type=str, required=True,
                        help='Path to training split file (list of PDB IDs)')
    parser.add_argument('--val_split', type=str, required=True,
                        help='Path to validation split file (list of PDB IDs)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing processed structures')
    parser.add_argument('--ring_features_dir', type=str, required=True,
                        help='Directory containing ring feature .npz files')
    parser.add_argument('--esm_embeddings_path', type=str, default=None,
                        help='Path to precomputed ESM embeddings')
    parser.add_argument('--val_data_dir', type=str, default=None,
                        help='Optional validation data directory override. '
                             'Defaults to --data_dir.')
    parser.add_argument('--val_ring_features_dir', type=str, default=None,
                        help='Optional validation ring feature directory override. '
                             'Defaults to --ring_features_dir.')
    parser.add_argument('--val_esm_embeddings_path', type=str, default=None,
                        help='Optional validation ESM embeddings override. '
                             'Defaults to --esm_embeddings_path.')
    parser.add_argument('--cache_path', type=str, default='data/cache_em',
                        help='Cache directory for processed graphs')

    # Pretrained model arguments
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained DiffDock model weights')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config YAML (uses pretrained defaults if not specified)')

    # Ring feature arguments
    parser.add_argument('--ring_dim', type=int, default=4,
                        help='Number of ring feature channels (default: 4)')
    parser.add_argument('--ring_vec_dim', type=int, default=6,
                        help='Number of l=1 vector ring feature channels / radii count (default: 6)')
    parser.add_argument('--ring_vector_clip_max_norm', type=float, default=20.0,
                        help='Clip max L2 norm for each ring vector (default: 20.0; <=0 disables)')
    parser.add_argument('--ring_vec_scale', type=float, default=1.0,
                        help='Multiply ring_vectors before the l=1 embedding (CGModel only). '
                             'ring_vectors arrive ~50x smaller than the percentile-normalised '
                             'ring_features, which starves the gradient of the single zero-init '
                             'l=1 linear. Try 30-50 to give the vector pathway a comparable '
                             'gradient scale. 1.0 = original behaviour.')
    parser.add_argument('--ring_lr_multiplier', type=float, default=10.0,
                        help='Learning rate multiplier for ring embedding (default: 10x)')
    parser.add_argument('--freeze_pretrained_epochs', type=int, default=0,
                        help='Number of epochs to freeze pretrained params (warmup ring embedding)')

    # LoRA arguments
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for parameter-efficient finetuning')
    parser.add_argument('--lora_r', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA scaling factor (default: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout probability (default: 0.1)')
    parser.add_argument('--lora_lr_multiplier', type=float, default=1.0,
                        help='LR multiplier specifically for LoRA weights')
    parser.add_argument('--lora_target_modules', type=str, default=None,
                        help='Comma-separated list of module name patterns to apply LoRA '
                             '(default: conv_layers,rec_emb_layers,final_conv,lig_emb_layers)')

    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--ema_rate', type=float, default=0.999,
                        help='EMA decay rate')
    parser.add_argument('--scheduler_patience', type=int, default=20,
                        help='Patience for LR scheduler')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--initial_eval_train_batches', type=int, default=64,
                        help='Run no-update sanity check on first N train batches before training; <=0 disables train check')
    parser.add_argument('--initial_eval_val_batches', type=int, default=-1,
                        help='Run no-update sanity check on first N val batches before training; -1 means full val, 0 disables')
    # Sampling-based validation. Score-matching val loss tracks docking accuracy
    # poorly, so this can also be used as the checkpoint-selection metric.
    parser.add_argument('--inference_epoch_freq', type=int, default=0,
                        help='Run sampling + symmetry-corrected RMSD on validation every N '
                             'epochs (0 disables). Costs roughly one extra forward pass per '
                             'sample per step, so keep N >= 5 unless the subset is small')
    parser.add_argument('--inference_complexes', type=int, default=100,
                        help='Validation complexes used by the inference epoch, taken in sorted '
                             'name order so the metric is measured on the same set every time; '
                             '<=0 uses all of them')
    parser.add_argument('--inference_samples', type=int, default=10,
                        help='Poses sampled per complex during an inference epoch')
    parser.add_argument('--inference_steps', type=int, default=20,
                        help='Reverse diffusion steps during an inference epoch')
    parser.add_argument('--inference_seed', type=int, default=0,
                        help='Seed reapplied at every inference epoch so each measurement '
                             'draws the same starting poses; epoch-to-epoch differences then '
                             'reflect the model instead of sampling noise')
    parser.add_argument('--selection_metric', choices=['val_loss', 'min_rmsds_lt2',
                                                       'min_rmsds_lt5', 'median_min_rmsd'],
                        default='val_loss',
                        help='Metric used for best_model.pt and LR scheduling. RMSD metrics '
                             'require --inference_epoch_freq > 0; success rates are maximised '
                             'and median_min_rmsd is minimised.')
    # Loss weights
    parser.add_argument('--tr_weight', type=float, default=1.0,
                        help='Translation loss weight')
    parser.add_argument('--rot_weight', type=float, default=1.0,
                        help='Rotation loss weight')
    parser.add_argument('--tor_weight', type=float, default=1.0,
                        help='Torsion loss weight')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization) to prevent weight drift')
    parser.add_argument('--backbone_loss_weight', type=float, default=0.0,
                        help='Backbone loss weight')
    parser.add_argument('--sidechain_loss_weight', type=float, default=0.0,
                        help='Sidechain loss weight')

    # Dataset parameters (loaded from pretrained model_parameters.yml)
    parser.add_argument('--receptor_radius', type=float, default=15.0,
                        help='Radius for receptor graph construction')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=24,
                        help='Max neighbors for Cα graph')
    parser.add_argument('--remove_hs', action='store_true', default=True,
                        help='Remove hydrogens from ligand')
    parser.add_argument('--keep_hs', action='store_true',
                        help='Keep hydrogens (overrides --remove_hs)')
    parser.add_argument('--max_lig_size', type=int, default=None,
                        help='Maximum ligand size (heavy atoms)')
    parser.add_argument('--max_receptor_size', type=int, default=10000,
                        help='Maximum receptor nodes (prevent OOM on A100)')
    parser.add_argument('--matching_popsize', type=int, default=20,
                        help='Conformer matching popsize')
    parser.add_argument('--matching_maxiter', type=int, default=20,
                        help='Conformer matching maxiter')
    parser.add_argument('--matching_tries', type=int, default=10,
                        help='Conformer matching tries')
    parser.add_argument('--not_knn_only_graph', action='store_true',
                        help='Use radius graph instead of KNN-only')
    parser.add_argument('--include_miscellaneous_atoms', action='store_true',
                        help='Include miscellaneous atoms')
    parser.add_argument('--num_conformers', type=int, default=1,
                        help='Number of ligand conformers')
    parser.add_argument('--protein_file', type=str, default='protein_processed',
                        help='Protein file suffix')
    parser.add_argument('--chain_cutoff', type=float, default=10,
                        help='Chain cutoff distance')
    parser.add_argument('--atom_radius', type=float, default=5,
                        help='Atom radius for all-atom model')
    parser.add_argument('--atom_max_neighbors', type=int, default=8,
                        help='Atom max neighbors for all-atom model')

    # Transform / sampling parameters
    parser.add_argument('--sampling_alpha', type=float, default=1,
                        help='Beta distribution alpha for time sampling')
    parser.add_argument('--sampling_beta', type=float, default=1,
                        help='Beta distribution beta for time sampling')
    parser.add_argument('--minimum_t', type=float, default=0.0,
                        help='Lower bound for sampled diffusion time t')
    parser.add_argument('--crop_beyond', type=float, default=20,
                        help='Crop receptor residues beyond this distance from ligand')

    # Model architecture (should match pretrained)
    parser.add_argument('--ns', type=int, default=48,
                        help='Number of scalar features')
    parser.add_argument('--nv', type=int, default=10,
                        help='Number of vector features')
    parser.add_argument('--num_conv_layers', type=int, default=11,
                        help='Number of convolution layers')
    parser.add_argument('--num_prot_emb_layers', type=int, default=3,
                        help='Number of protein embedding layers')
    parser.add_argument('--sigma_embed_dim', type=int, default=64,
                        help='Sigma embedding dimension')
    parser.add_argument('--distance_embed_dim', type=int, default=64,
                        help='Distance embedding dimension')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=64,
                        help='Cross distance embedding dimension')
    parser.add_argument('--max_radius', type=float, default=5.0,
                        help='Max ligand radius')
    parser.add_argument('--no_torsion', action='store_true',
                        help='Disable torsion prediction')
    parser.add_argument('--all_atoms', action='store_true',
                        help='Use all-atom model (AAModel)')
    parser.add_argument('--sh_lmax', type=int, default=2,
                        help='Spherical harmonics lmax')
    parser.add_argument('--use_second_order_repr', action='store_true',
                        help='Use second order representations')
    parser.add_argument('--reduce_pseudoscalars', action='store_true',
                        help='Reduce pseudoscalars')
    parser.add_argument('--embed_also_ligand', action='store_true',
                        help='Embed also ligand')
    parser.add_argument('--cross_max_distance', type=float, default=80,
                        help='Max cross distance')
    parser.add_argument('--dynamic_max_cross', action='store_true',
                        help='Dynamic max cross distance')
    parser.add_argument('--no_batch_norm', action='store_true',
                        help='Disable batch normalization')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--scale_by_sigma', action='store_true', default=True,
                        help='Scale by sigma')

    # Diffusion arguments
    parser.add_argument('--tr_sigma_min', type=float, default=0.1,
                        help='Translation sigma min')
    parser.add_argument('--tr_sigma_max', type=float, default=19.0,
                        help='Translation sigma max')
    parser.add_argument('--rot_sigma_min', type=float, default=0.03,
                        help='Rotation sigma min')
    parser.add_argument('--rot_sigma_max', type=float, default=1.55,
                        help='Rotation sigma max')
    parser.add_argument('--tor_sigma_min', type=float, default=0.0314,
                        help='Torsion sigma min')
    parser.add_argument('--tor_sigma_max', type=float, default=3.14,
                        help='Torsion sigma max')

    # Logging
    parser.add_argument('--run_name', type=str, default='diffdock_em_finetune',
                        help='Name of the run')
    parser.add_argument('--log_dir', type=str, default='workdir_em',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='diffdock-em',
                        help='Wandb project name')
    parser.add_argument('--save_model_freq', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from, or "auto" to find last_model.pt in run_dir')
    parser.add_argument('--resume_best_val', type=float, default=None,
                        help='Best val loss so far, for resuming a run whose checkpoint predates '
                             'best_val_loss tracking; prevents overwriting a better best_model.pt')

    return parser


def parse_args():
    return get_parser().parse_args()


def get_optimizer_with_ring_lr(model, base_lr, ring_lr_multiplier, lora_lr_multiplier=1.0, weight_decay=0.0, use_lora=False):
    """Create optimizer with different learning rates for pretrained, ring, and LoRA params."""
    ring_params = []
    lora_params = []
    pretrained_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'ring_feature_embedding' in name or 'atom_ring_feature_embedding' in name \
                or 'ring_vector_embedding' in name or 'atom_ring_vector_embedding' in name:
            ring_params.append(param)
        elif 'lora_' in name:
            lora_params.append(param)
        else:
            pretrained_params.append(param)

    param_groups = []

    # Pretrained params (only included if not using LoRA, or if some are trainable)
    if pretrained_params:
        param_groups.append({
            'params': pretrained_params,
            'lr': base_lr,
            'name': 'pretrained',
        })

    # Ring embedding params (higher LR)
    if ring_params:
        param_groups.append({
            'params': ring_params,
            'lr': base_lr * ring_lr_multiplier,
            'name': 'ring_embedding',
        })

    # LoRA params (same LR as ring, or can be tuned separately)
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': base_lr * lora_lr_multiplier,
            'name': 'lora',
        })

    if not param_groups:
        raise ValueError("No trainable parameters found!")

    return torch.optim.Adam(param_groups, weight_decay=weight_decay)


def _is_data_parallel_model(model):
    return hasattr(model, 'module') and type(model).__name__ == 'DataParallel'


def _split_to_samples(data):
    if isinstance(data, (list, tuple)):
        return list(data)
    if hasattr(data, 'to_data_list'):
        return data.to_data_list()
    return [data]


def _forward_single(model, sample):
    if _is_data_parallel_model(model):
        return model([sample])
    return model(sample)


def initial_no_update_eval(model, loader, device, t_to_sigma, loss_fn, split_name, max_batches=-1, freeze_bn=True):
    """Evaluate initial loss without backward/optimizer updates.

    This is intentionally separate from train_epoch/test_epoch so we can tell
    whether bad losses exist before the first optimizer step.
    """
    if max_batches == 0:
        return

    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    was_training = model.training
    model.eval()
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    sums = {'loss': 0.0, 'loss_clipped': 0.0, 'tr_loss': 0.0, 'rot_loss': 0.0, 'tor_loss': 0.0}
    count = 0
    batch_count = 0
    raw_nan_batches = 0
    extreme_loss_batches = 0
    fallback_samples = 0
    skipped_samples = 0

    def _add_loss_tuple(loss_tuple):
        nonlocal count, extreme_loss_batches
        if loss_tuple is None:
            return
        loss_val = loss_tuple[0].detach().cpu()
        if loss_val.numel() == 0 or torch.any(torch.isnan(loss_val)) or torch.any(torch.isinf(loss_val)):
            return
        if loss_val.abs().max().item() > BATCH_LOSS_MAX:
            extreme_loss_batches += 1
            return
        n = int(loss_val.numel())
        sums['loss'] += float(loss_val.sum().item())
        sums['loss_clipped'] += float(loss_val.clamp(max=VAL_LOSS_CLAMP).sum().item())
        for key, value in zip(['tr_loss', 'rot_loss', 'tor_loss'], loss_tuple[1:4]):
            value = value.detach().cpu()
            if value.numel() == 1 and n > 1:
                sums[key] += float(value.item()) * n
            else:
                sums[key] += float(value.mean().item()) * n
        count += n

    with torch.no_grad():
        for data in loader:
            if max_batches > 0 and batch_count >= max_batches:
                break
            batch_count += 1

            if isinstance(data, (list, tuple)):
                data = [d.to(device) for d in data]
            else:
                data = data.to(device)

            try:
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
                raw_has_nan = (
                    torch.any(torch.isnan(tr_pred)) or
                    torch.any(torch.isnan(rot_pred)) or
                    (tor_pred is not None and torch.any(torch.isnan(tor_pred)))
                )
                if raw_has_nan:
                    raw_nan_batches += 1
                    for sample in _split_to_samples(data):
                        try:
                            pred = _forward_single(model, sample)
                            sample_has_nan = any(torch.any(torch.isnan(t)) for t in pred if t is not None)
                            if sample_has_nan:
                                skipped_samples += 1
                                continue
                            fallback_samples += 1
                            _add_loss_tuple(loss_fn(*pred, data=[sample], t_to_sigma=t_to_sigma,
                                                     device=device, apply_mean=False))
                        except Exception:
                            skipped_samples += 1
                    continue

                loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred,
                                     data=data, t_to_sigma=t_to_sigma, device=device, apply_mean=False)
                _add_loss_tuple(loss_tuple)
            except RuntimeError as exc:
                print(f"Initial {split_name}: RuntimeError in no-update eval batch {batch_count}: {exc}")
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                continue

    if was_training:
        model.train()
    random.setstate(py_rng_state)
    np.random.set_state(np_rng_state)
    torch.set_rng_state(torch_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    if count == 0:
        print(f"Initial {split_name} no-update: no valid samples "
              f"(batches={batch_count}, raw_nan_batches={raw_nan_batches}, skipped_samples={skipped_samples})")
        return

    print(f"Initial {split_name} no-update: "
          f"loss {sums['loss'] / count:.4f} "
          f"clipped {sums['loss_clipped'] / count:.4f} "
          f"tr {sums['tr_loss'] / count:.4f} "
          f"rot {sums['rot_loss'] / count:.4f} "
          f"tor {sums['tor_loss'] / count:.4f} "
          f"valid_samples {count} "
          f"batches {batch_count} "
          f"raw_nan_batches {raw_nan_batches} "
          f"extreme_loss_batches {extreme_loss_batches} "
          f"fallback_samples {fallback_samples} "
          f"skipped_samples {skipped_samples}")


def _capture_rng_state():
    """Snapshot every RNG the sampling loop can advance.

    An inference epoch draws random starting poses and reverse-diffusion noise.
    Without restoring the streams afterwards, enabling it would shift the noise
    the next training epoch sees, so a run with the metric would follow a
    different trajectory than one without it and the two would not be comparable.
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _select_inference_complexes(val_loader, limit):
    """Pick a fixed subset of validation graphs holding their ground-truth poses.

    ``dataset.get(i)`` is called instead of indexing the dataset: it attaches the
    RDKit molecule and strips the temporary attributes that would break
    collation, while skipping the NoiseTransform that ``__getitem__`` applies, so
    the ligand keeps its ground-truth pose.

    The subset is taken in sorted name order rather than sampled, so the metric is
    measured on the same complexes every epoch and across runs.
    """
    dataset = val_loader.dataset
    graphs = getattr(dataset, 'complex_graphs', None)
    if graphs is None:
        raise AttributeError(
            f"validation dataset {type(dataset).__name__} has no complex_graphs; "
            "cannot run an inference epoch"
        )
    order = sorted(range(len(graphs)), key=lambda i: getattr(graphs[i], 'name', ''))
    if limit > 0:
        order = order[:limit]
    return [dataset.get(i) for i in order]


def train_em(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir,
             start_epoch=0, best_val_loss=math.inf, best_epoch=0,
             best_selection_score=None, best_selection_epoch=0):
    """Main training loop for EM finetuning."""

    loss_fn = partial(
        loss_function,
        tr_weight=args.tr_weight,
        rot_weight=args.rot_weight,
        tor_weight=args.tor_weight,
        no_torsion=args.no_torsion,
        backbone_weight=args.backbone_loss_weight,
        sidechain_weight=args.sidechain_loss_weight,
    )

    if args.selection_metric != 'val_loss' and not args.inference_epoch_freq:
        raise ValueError(
            f'--selection_metric={args.selection_metric} requires '
            '--inference_epoch_freq > 0'
        )

    # The score-matching loss remains the differentiable training objective.
    # selection_metric controls checkpoint selection and LR scheduling.
    if start_epoch > 0 and math.isfinite(best_val_loss):
        print(f"Carrying over best val loss {best_val_loss:.4f} from epoch {best_epoch}")

    if best_selection_score is None:
        best_selection_score = (
            math.inf
            if args.selection_metric in ('val_loss', 'median_min_rmsd')
            else -math.inf
        )
    print(f"Checkpoint selection metric: {args.selection_metric}")

    # Keep the inference-specific legacy checkpoint tracking separately from
    # the configurable selection metric used by best_model.pt.
    inference_complex_graphs = []
    best_inference_score = -math.inf
    best_inference_epoch = -1
    if args.inference_epoch_freq:
        inference_complex_graphs = _select_inference_complexes(
            val_loader, args.inference_complexes
        )
        print(f"Inference epoch enabled: every {args.inference_epoch_freq} epoch(s) on "
              f"{len(inference_complex_graphs)} validation complexes, "
              f"{args.inference_samples} samples x {args.inference_steps} steps")

    # Mixed precision training - DISABLED due to NaN issues
    scaler = None
    print("Mixed precision DISABLED (FP32 training)")

    if start_epoch == 0:
        print("\nInitial no-update sanity check (before any optimizer.step)")
        initial_no_update_eval(
            model, train_loader, device, t_to_sigma, loss_fn,
            split_name=f"train first {args.initial_eval_train_batches} batches",
            max_batches=args.initial_eval_train_batches,
            freeze_bn=True,
        )
        initial_no_update_eval(
            model, val_loader, device, t_to_sigma, loss_fn,
            split_name="val" if args.initial_eval_val_batches < 0 else f"val first {args.initial_eval_val_batches} batches",
            max_batches=args.initial_eval_val_batches,
            freeze_bn=True,
        )
        print("End initial no-update sanity check\n")

    print("Starting EM finetuning...")
    for epoch in range(start_epoch, args.n_epochs):
        ema_state_dict = None
        # Warmup phase: only ring embedding trains; LoRA (and pretrained) params are frozen
        if args.freeze_pretrained_epochs > 0:
            if epoch < args.freeze_pretrained_epochs:
                for name, param in model.named_parameters():
                    is_ring = ('ring_feature_embedding' in name or 'atom_ring_feature_embedding' in name
                               or 'ring_vector_embedding' in name or 'atom_ring_vector_embedding' in name)
                    if not is_ring:
                        param.requires_grad = False
                if epoch == 0:
                    print(f"[Warmup] Frozen all params except ring embedding "
                          f"for first {args.freeze_pretrained_epochs} epochs")
            elif epoch == args.freeze_pretrained_epochs:
                print(f"[Warmup done] Unfreezing trainable parameters at epoch {epoch}")
                if args.use_lora:
                    for name, param in model.named_parameters():
                        if 'lora_' in name:
                            param.requires_grad = True
                else:
                    for param in model.parameters():
                        param.requires_grad = True

        # Training
        train_losses = train_epoch(
            model, train_loader, optimizer, device, t_to_sigma, loss_fn,
            ema_weights if (not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs) else None,
            scaler=scaler,
            freeze_bn=True,
        )
        lr_info = " ".join(
            f"{pg.get('name', f'group_{i}')}_lr {pg['lr']:.6f}"
            for i, pg in enumerate(optimizer.param_groups)
        )
        print(f"Epoch {epoch}: Train loss {train_losses['loss']:.4f} "
              f"tr {train_losses['tr_loss']:.4f} "
              f"rot {train_losses['rot_loss']:.4f} "
              f"tor {train_losses['tor_loss']:.4f} "
              f"{lr_info}")

        # Validation with EMA weights
        if not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())

        val_losses = test_epoch(
            model, val_loader, device, t_to_sigma, loss_fn,
            test_sigma_intervals=True, use_amp=False, freeze_bn=True
        )
        _unclipped = val_losses.get('loss_unclipped', float('nan'))
        print(f"Epoch {epoch}: Val loss {val_losses['loss']:.4f} (unclipped {_unclipped:.4f}) "
              f"tr {val_losses['tr_loss']:.4f} "
              f"rot {val_losses['rot_loss']:.4f} "
              f"tor {val_losses['tor_loss']:.4f}")

        # Sampling-based RMSD, measured with the same EMA weights as the val loss.
        # Score-matching loss is a weak proxy for docking accuracy — upstream
        # DiffDock selects checkpoints on this metric instead, which is why the
        # pretrained weights are named best_ema_inference_epoch_model.pt. A failure
        # here must not kill training, so it only costs the metric for this epoch.
        inference_metrics = {}
        if inference_complex_graphs and (epoch + 1) % args.inference_epoch_freq == 0:
            rng_state = _capture_rng_state()
            try:
                # Common random numbers: reseeding identically every time means the
                # same starting poses and reverse-diffusion noise are drawn at each
                # measurement, so epoch-to-epoch differences reflect the model
                # rather than sampling luck. Without this, the metric's own noise
                # is large enough that picking the best epoch mostly picks the
                # luckiest one.
                random.seed(args.inference_seed)
                np.random.seed(args.inference_seed)
                torch.manual_seed(args.inference_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.inference_seed)
                inference_metrics = inference_epoch_fix(
                    model, inference_complex_graphs, device, t_to_sigma, args
                )
                print(f"Epoch {epoch}: Inference "
                      f"min_rmsd<2A {inference_metrics['min_rmsds_lt2']:.2f}% "
                      f"min_rmsd<5A {inference_metrics['min_rmsds_lt5']:.2f}% "
                      f"median_min_rmsd {inference_metrics['median_min_rmsd']:.3f}A "
                      f"all_rmsd<2A {inference_metrics['rmsds_lt2']:.2f}% "
                      f"(n={len(inference_complex_graphs)})")
            except Exception as exc:
                print(f"Epoch {epoch}: inference epoch failed, skipping metric: "
                      f"{type(exc).__name__}: {exc}")
            finally:
                _restore_rng_state(rng_state)

        if not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs:
            ema_state_dict = copy.deepcopy(
                model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            )
            ema_weights.restore(model.parameters())

        # Logging
        if args.wandb:
            logs = {}
            logs.update({'train_' + k: v for k, v in train_losses.items()})
            logs.update({'val_' + k: v for k, v in val_losses.items()})
            logs.update({'inference_' + k: v for k, v in inference_metrics.items()})
            for i, pg in enumerate(optimizer.param_groups):
                name = pg.get('name', f'group_{i}')
                logs[f'lr_{name}'] = pg['lr']
            wandb.log(logs, step=epoch + 1)

        # Select the checkpoint using either validation loss or the sampling
        # RMSD metric. RMSD is not backpropagated; it only selects checkpoints
        # and optionally drives the plateau scheduler.
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if val_losses['loss'] < best_val_loss:
            best_val_loss = val_losses['loss']
            best_epoch = epoch

        if args.selection_metric == 'val_loss':
            selection_score = float(val_losses['loss'])
            is_better = selection_score < best_selection_score
        else:
            selection_score = float(inference_metrics.get(args.selection_metric, math.nan))
            if not math.isfinite(selection_score):
                is_better = False
            elif args.selection_metric == 'median_min_rmsd':
                is_better = selection_score < best_selection_score
            else:
                is_better = selection_score > best_selection_score

        if is_better:
            best_selection_score = selection_score
            best_selection_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
            if not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_ema_model.pt'))

            # Also save LoRA-only weights if using LoRA
            if args.use_lora:
                lora_state = get_lora_state_dict(model.module if hasattr(model, 'module') else model)
                torch.save(lora_state, os.path.join(run_dir, 'best_lora_weights.pt'))

        # Keep a separately named RMSD-selected copy for backwards compatibility
        # and for comparison when selection_metric=val_loss.
        if inference_metrics and inference_metrics['min_rmsds_lt2'] > best_inference_score:
            best_inference_score = inference_metrics['min_rmsds_lt2']
            best_inference_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_inference_model.pt'))
            if not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs:
                torch.save(ema_state_dict, os.path.join(run_dir, 'best_inference_ema_model.pt'))
            if args.use_lora:
                torch.save(
                    get_lora_state_dict(model.module if hasattr(model, 'module') else model),
                    os.path.join(run_dir, 'best_inference_lora_weights.pt'),
                )

        # Periodic save
        if args.save_model_freq and (epoch + 1) % args.save_model_freq == 0:
            torch.save(state_dict, os.path.join(run_dir, f'epoch{epoch+1}_model.pt'))
            if ema_state_dict is not None:
                torch.save(
                    ema_state_dict,
                    os.path.join(run_dir, f'epoch{epoch+1}_ema_model.pt'),
                )
            if args.use_lora:
                lora_state = get_lora_state_dict(model.module if hasattr(model, 'module') else model)
                torch.save(lora_state, os.path.join(run_dir, f'epoch{epoch+1}_lora_weights.pt'))
            # Save full checkpoint for warm restart
            torch.save({
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'ema_weights': ema_weights.state_dict(),
                'use_lora': args.use_lora,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'best_selection_score': best_selection_score,
                'best_selection_epoch': best_selection_epoch,
                'selection_metric': args.selection_metric,
                'lora_r': args.lora_r if args.use_lora else None,
                'lora_alpha': args.lora_alpha if args.use_lora else None,
            }, os.path.join(run_dir, f'checkpoint_epoch{epoch+1}.pt'))

        # LR scheduler step
        if scheduler and (not args.freeze_pretrained_epochs or epoch >= args.freeze_pretrained_epochs):
            if args.selection_metric == 'val_loss':
                scheduler.step(val_losses['loss'])
            elif inference_metrics and math.isfinite(selection_score):
                # For success-rate metrics, ReduceLROnPlateau must maximise;
                # the scheduler itself is configured accordingly in main().
                scheduler.step(selection_score)

        # Save last model
        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'ema_weights': ema_weights.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'best_selection_score': best_selection_score,
            'best_selection_epoch': best_selection_epoch,
            'selection_metric': args.selection_metric,
            'use_lora': args.use_lora,
            'lora_r': args.lora_r if args.use_lora else None,
            'lora_alpha': args.lora_alpha if args.use_lora else None,
        }, os.path.join(run_dir, 'last_model.pt'))

    print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best {args.selection_metric} {best_selection_score:.4f} at epoch {best_selection_epoch}")
    if best_inference_epoch >= 0:
        print(f"Best sampled min_rmsd<2A {best_inference_score:.2f}% at epoch {best_inference_epoch} "
              f"-> best_inference_ema_model.pt")


def main():
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    args = parse_args()

    # Load pretrained model's config to ensure architecture matches.
    # Original DiffDock checkpoints store model_parameters.yml, while local
    # finetuning runs save config.yml next to the checkpoint.
    pretrained_dir = os.path.dirname(args.pretrained_model)
    pretrained_config_path = None
    for candidate in ('model_parameters.yml', 'config.yml'):
        candidate_path = os.path.join(pretrained_dir, candidate)
        if os.path.exists(candidate_path):
            pretrained_config_path = candidate_path
            break

    if pretrained_config_path is not None:
        print(f"Loading pretrained model config from {pretrained_config_path}")
        with open(pretrained_config_path) as f:
            pretrained_config = yaml.safe_load(f)
        # Copy architecture-related parameters from pretrained model
        arch_params = [
            'ns', 'nv', 'num_conv_layers', 'num_prot_emb_layers', 'sigma_embed_dim',
            'distance_embed_dim', 'cross_distance_embed_dim', 'sh_lmax',
            'use_second_order_repr', 'reduce_pseudoscalars', 'embed_also_ligand',
            'scale_by_sigma', 'max_radius', 'cross_max_distance', 'dynamic_max_cross',
            'no_batch_norm', 'dropout', 'no_torsion', 'all_atoms',
            'backbone_loss_weight', 'sidechain_loss_weight',
            'embedding_type', 'embedding_scale',
            'not_fixed_center_conv', 'not_fixed_knn_radius_graph',
            'no_differentiate_convolutions', 'tp_weights_layers',
        ]
        for key in arch_params:
            if key in pretrained_config:
                setattr(args, key, pretrained_config[key])

        # Copy dataset-related parameters from pretrained model
        data_params = [
            'receptor_radius', 'c_alpha_max_neighbors', 'remove_hs',
            'max_lig_size', 'matching_popsize', 'matching_maxiter', 'matching_tries',
            'num_conformers', 'atom_radius', 'atom_max_neighbors',
            'sampling_alpha', 'sampling_beta', 'crop_beyond',
        ]
        for key in data_params:
            if key in pretrained_config:
                setattr(args, key, pretrained_config[key])

    # Clear pretrained ESM paths - we use esm_embeddings_path for EM data
    # The pretrained config may have paths like pdbbind_esm_embeddings_path that
    # trigger lm_embedding_type='precomputed' even when those files don't exist
    if not hasattr(args, 'esm_embeddings_path') or args.esm_embeddings_path is None:
        # No EM ESM embeddings provided - clear pretrained paths too
        for esm_path_key in ['moad_esm_embeddings_path', 'pdbbind_esm_embeddings_path',
                             'pdbsidechain_esm_embeddings_path']:
            if hasattr(args, esm_path_key):
                setattr(args, esm_path_key, None)

    # Load additional model config if provided (overrides pretrained config)
    if args.model_config:
        with open(args.model_config) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    # Setup run directory
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config
    save_yaml_file(os.path.join(run_dir, 'config.yml'), vars(args))

    # Wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    # Create t_to_sigma function
    t_to_sigma = partial(t_to_sigma_compl, args=args)

    # Handle --keep_hs overriding --remove_hs
    if args.keep_hs:
        args.remove_hs = False

    # Create noise transform
    transform = NoiseTransform(
        t_to_sigma=t_to_sigma,
        no_torsion=args.no_torsion,
        all_atom=args.all_atoms,
        alpha=args.sampling_alpha,
        beta=args.sampling_beta,
        minimum_t=args.minimum_t,
        include_miscellaneous_atoms=args.include_miscellaneous_atoms,
        crop_beyond_cutoff=args.crop_beyond,
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset_args = dict(
        root=args.data_dir,
        ring_features_dir=args.ring_features_dir,
        transform=transform,
        cache_path=args.cache_path,
        esm_embeddings_path=args.esm_embeddings_path,
        all_atoms=args.all_atoms,
        receptor_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        remove_hs=args.remove_hs,
        max_lig_size=args.max_lig_size,
        matching=not args.no_torsion,
        popsize=args.matching_popsize,
        maxiter=args.matching_maxiter,
        num_conformers=args.num_conformers,
        knn_only_graph=not args.not_knn_only_graph,
        protein_file=args.protein_file,
        num_workers=args.num_workers,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors,
        ring_dim=args.ring_dim,
        ring_vec_dim=args.ring_vec_dim,
        ring_vector_clip_max_norm=args.ring_vector_clip_max_norm,
    )
    val_dataset_args = dict(train_dataset_args)
    val_dataset_args.update(
        root=args.val_data_dir or args.data_dir,
        ring_features_dir=args.val_ring_features_dir or args.ring_features_dir,
        esm_embeddings_path=(
            args.val_esm_embeddings_path
            if args.val_esm_embeddings_path is not None
            else args.esm_embeddings_path
        ),
    )

    if args.val_data_dir or args.val_ring_features_dir or args.val_esm_embeddings_path:
        print("Validation dataset overrides:")
        print(f"  val_data_dir: {val_dataset_args['root']}")
        print(f"  val_ring_features_dir: {val_dataset_args['ring_features_dir']}")
        print(f"  val_esm_embeddings_path: {val_dataset_args['esm_embeddings_path']}")

    train_dataset = EMDataset(
        split_path=args.train_split,
        **train_dataset_args,
    )

    val_dataset = EMDataset(
        split_path=args.val_split,
        **val_dataset_args,
    )

    print_statistics(train_dataset.complex_graphs)

    # Create data loaders
    # DataListLoader requires DataParallel wrapping (multi-GPU); use DataLoader for single GPU
    use_data_parallel = torch.cuda.is_available() and torch.cuda.device_count() > 1
    loader_class = DataListLoader if use_data_parallel else DataLoader
    train_loader = loader_class(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = loader_class(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Create model
    print("Creating model...")
    model = get_em_model(args, device, t_to_sigma, ring_dim=args.ring_dim, ring_vec_dim=args.ring_vec_dim,
                         ring_vec_scale=args.ring_vec_scale, no_parallel=True)

    # Apply LoRA BEFORE loading weights so checkpoint keys match LoRA-wrapped layer names
    if args.use_lora:
        print(f"\nApplying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")

        # Parse target modules
        target_modules = None
        if args.lora_target_modules:
            target_modules = set(m.strip() for m in args.lora_target_modules.split(','))

        model, lora_layers, lora_params = apply_lora_to_model(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            verbose=True,
        )

    # Load pretrained weights (LoRA already applied, so lora_A/lora_B keys match checkpoint)
    print(f"Loading pretrained weights from {args.pretrained_model}")
    missing_keys, unexpected_keys = load_pretrained_weights(model, args.pretrained_model)

    # Verify ring embedding is new
    ring_keys = [k for k in missing_keys if 'ring' in k]
    if ring_keys:
        print(f"Ring embedding parameters (new, will be trained): {ring_keys}")

    if args.use_lora:
        # Freeze non-LoRA parameters
        freeze_non_lora_params(model, verbose=True)
        print_trainable_params(model)
        print("BatchNorm running stats will be frozen during training (eval mode BN)")

    # Move to device and wrap with DataParallel if multi-GPU
    model = model.to(device)
    if use_data_parallel:
        from torch_geometric.nn.data_parallel import DataParallel
        model = DataParallel(model)

    numel = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {numel:,} total, {trainable:,} trainable ({100*trainable/numel:.2f}%)")

    # Create optimizer with differential LR
    optimizer = get_optimizer_with_ring_lr(
        model.module if hasattr(model, 'module') else model,
        base_lr=args.lr,
        ring_lr_multiplier=args.ring_lr_multiplier,
        lora_lr_multiplier=args.lora_lr_multiplier,
        weight_decay=args.weight_decay,
        use_lora=args.use_lora,
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min' if args.selection_metric in ('val_loss', 'median_min_rmsd') else 'max',
        factor=0.7,
        patience=args.scheduler_patience,
        min_lr=args.lr / 100,
    )

    # EMA (only for trainable parameters)
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    best_selection_score = None
    best_selection_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if ckpt_path == 'auto':
            ckpt_path = os.path.join(run_dir, 'last_model.pt')
        if os.path.exists(ckpt_path):
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            m = model.module if hasattr(model, 'module') else model
            m.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('scheduler') is not None and scheduler is not None:
                scheduler.load_state_dict(ckpt['scheduler'])
            if 'ema_weights' in ckpt:
                ema_weights.load_state_dict(ckpt['ema_weights'], device)
            start_epoch = ckpt['epoch'] + 1
            best_val_loss = ckpt.get('best_val_loss', math.inf)
            best_epoch = ckpt.get('best_epoch', 0)
            checkpoint_metric = ckpt.get('selection_metric')
            if checkpoint_metric and checkpoint_metric != args.selection_metric:
                print(
                    f"WARNING: checkpoint selection metric was {checkpoint_metric}, "
                    f"current run requests {args.selection_metric}; starting current "
                    "selection tracking from the next measured epoch."
                )
            best_selection_score = ckpt.get('best_selection_score')
            best_selection_epoch = ckpt.get('best_selection_epoch', 0)
            print(f"Resumed from epoch {ckpt['epoch']}, continuing from epoch {start_epoch}")
            if not math.isfinite(best_val_loss):
                # Checkpoint predates best_val_loss tracking. Without it the next
                # epoch would overwrite an existing best_model.pt unconditionally.
                print("WARNING: this checkpoint has no best_val_loss; back up "
                      "best_model.pt / best_ema_model.pt or pass --resume_best_val "
                      "with the best value from the log, or they will be overwritten.")
            if args.resume_best_val is not None:
                best_val_loss = args.resume_best_val
                print(f"Overriding best val loss from --resume_best_val: {best_val_loss:.4f}")
        else:
            print(f"WARNING: checkpoint not found at {ckpt_path}, starting from scratch")

    # Train
    train_em(args, model, optimizer, scheduler, ema_weights, train_loader, val_loader, t_to_sigma, run_dir,
             start_epoch=start_epoch, best_val_loss=best_val_loss, best_epoch=best_epoch,
             best_selection_score=best_selection_score,
             best_selection_epoch=best_selection_epoch)

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
