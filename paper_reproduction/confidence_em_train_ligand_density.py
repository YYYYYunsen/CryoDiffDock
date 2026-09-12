"""Train a ligand-density confidence model from external pose caches.

Uses AAOldLigandDensityModel which inherits from the same AAOldModel that
DiffDock-L's official confidence model uses. By default this script enables
only the per-atom ligand-density injection path, keeping the pretrained
confidence head intact. An optional density-summary branch can be enabled
explicitly, but that branch rebuilds the confidence head and is therefore a
less faithful pretrained initialization.
"""

from __future__ import annotations

import gc
import os
from argparse import ArgumentParser, FileType
from functools import partial

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from confidence.em_dataset_ligand_density import (
    LigandDensityConfidenceAllPosesDataset,
    LigandDensityConfidenceDataset,
    LigandDensityConfidencePairDataset,
    pairwise_collate,
)
from models.lora import apply_lora_to_model, freeze_non_lora_params, print_trainable_params
from models.old_aa_model_ligand_density import AAOldLigandDensityModel
from utils.diffusion_utils import get_timestep_embedding, t_to_sigma as t_to_sigma_compl
from utils.training import AverageMeter
from utils.utils import ExponentialMovingAverage, save_yaml_file

torch.multiprocessing.set_sharing_strategy("file_descriptor")


_BN_TYPES = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
CONF_SAMPLE_LOSS_CLAMP = 3.0
PARAM_MAX_THRESHOLD = 2e1


def get_parser():
    parser = ArgumentParser(description="Ligand-density confidence model training")
    parser.add_argument("--config", type=FileType(mode="r"), default=None)

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split_train", type=str, required=True)
    parser.add_argument("--split_val", type=str, required=True)
    parser.add_argument("--map_csv", type=str, required=True)
    parser.add_argument("--pose_cache_train", type=str, required=True)
    parser.add_argument("--pose_cache_val", type=str, required=True)
    parser.add_argument("--graph_cache_path", type=str, default="data/cache_confidence_ligand_density_graphs")
    parser.add_argument("--graph_cache_train", type=str, default=None)
    parser.add_argument("--graph_cache_val", type=str, default=None)
    parser.add_argument("--density_cache_root", type=str, default="data/cache_confidence_ligand_density_features")
    parser.add_argument("--density_cache_train", type=str, default=None,
                        help="Override density cache dir for train split (default: density_cache_root/train)")
    parser.add_argument("--density_cache_val", type=str, default=None,
                        help="Override density cache dir for val split (default: density_cache_root/val)")
    parser.add_argument("--cache_ids_train", nargs="+", type=int, default=None)
    parser.add_argument("--cache_ids_val", nargs="+", type=int, default=None)
    parser.add_argument("--pretrained_confidence_model", type=str, required=True)
    parser.add_argument("--pretrained_confidence_config", type=str, default=None)

    parser.add_argument("--shell_radii", nargs="+", type=float, default=[0.8, 1.6, 2.4])
    parser.add_argument("--density_lr_multiplier", type=float, default=3.0)
    parser.add_argument("--rmsd_classification_cutoff", nargs="+", type=float, default=[2.0])
    parser.add_argument("--pairwise_loss", action="store_true")
    parser.add_argument("--pairwise_margin", type=float, default=1.0)
    parser.add_argument("--pairwise_bce_weight", type=float, default=0.1)
    parser.add_argument("--pairwise_min_rmsd_gap", type=float, default=1.0)
    parser.add_argument("--pairwise_hard_negative_max_rmsd", type=float, default=4.0)
    parser.add_argument("--pairwise_hard_fraction", type=float, default=0.8)
    parser.add_argument("--balance", action="store_true", default=False)
    parser.add_argument("--val_max_poses_per_complex", type=int, default=0)
    parser.add_argument("--val_exclude_oracle_pose", action="store_true")
    parser.add_argument("--max_train_pairs_per_epoch", type=int, default=0,
                        help="Pairwise mode only: cap the number of complexes/pairs per epoch. 0 uses all.")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Run validation every N epochs. 0 disables validation during training.")
    parser.add_argument("--fast_val_every", type=int, default=1,
                        help="Run lightweight validation every N epochs. 0 disables fast validation.")
    parser.add_argument("--fast_val_max_poses_per_complex", type=int, default=3,
                        help="Fast validation only: cap poses per complex. 0 uses all poses.")

    parser.add_argument("--receptor_radius", type=float, default=15.0)
    parser.add_argument("--c_alpha_max_neighbors", type=int, default=24)
    parser.add_argument("--atom_radius", type=float, default=5.0)
    parser.add_argument("--atom_max_neighbors", type=int, default=8)
    parser.add_argument("--remove_hs", action="store_true", default=True)
    parser.add_argument("--keep_hs", action="store_true")
    parser.add_argument("--max_lig_size", type=int, default=None)
    parser.add_argument("--matching_popsize", type=int, default=20)
    parser.add_argument("--matching_maxiter", type=int, default=20)
    parser.add_argument("--num_conformers", type=int, default=1)
    parser.add_argument("--not_knn_only_graph", action="store_true")
    parser.add_argument("--protein_file", type=str, default="protein_processed")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit_complexes", type=int, default=0)
    parser.add_argument("--exclude_complexes", nargs="+", type=str, default=None)
    parser.add_argument("--max_ligand_nodes", type=int, default=0)
    parser.add_argument("--max_receptor_nodes", type=int, default=0)
    parser.add_argument("--max_atom_nodes", type=int, default=0)
    parser.add_argument("--max_total_nodes", type=int, default=0)
    parser.add_argument("--esm_embeddings_path", type=str, default=None)

    # Model architecture — defaults match DiffDock-L confidence model; overridden by
    # pretrained config at runtime so these only matter if no config is found.
    parser.add_argument("--ns", type=int, default=48)
    parser.add_argument("--nv", type=int, default=10)
    parser.add_argument("--num_conv_layers", type=int, default=6)
    parser.add_argument("--sigma_embed_dim", type=int, default=64)
    parser.add_argument("--distance_embed_dim", type=int, default=64)
    parser.add_argument("--cross_distance_embed_dim", type=int, default=64)
    parser.add_argument("--max_radius", type=float, default=5.0)
    parser.add_argument("--cross_max_distance", type=float, default=80.0)
    parser.add_argument("--dynamic_max_cross", action="store_true", default=False)
    parser.add_argument("--sh_lmax", type=int, default=2)
    parser.add_argument("--use_second_order_repr", action="store_true")
    parser.add_argument("--no_batch_norm", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--confidence_dropout", type=float, default=0.0)
    parser.add_argument("--confidence_no_batchnorm", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--freeze_pretrained_epochs", type=int, default=0)
    parser.add_argument("--scale_by_sigma", action="store_true", default=True)
    parser.add_argument("--norm_by_sigma", dest="norm_by_sigma", action="store_true")
    parser.add_argument("--no_norm_by_sigma", dest="norm_by_sigma", action="store_false")
    parser.set_defaults(norm_by_sigma=True)
    parser.add_argument("--no_torsion", action="store_true")
    parser.add_argument("--all_atoms", action="store_true", default=True)
    parser.add_argument("--use_old_atom_encoder", dest="use_old_atom_encoder", action="store_true")
    parser.add_argument("--no_use_old_atom_encoder", dest="use_old_atom_encoder", action="store_false")
    parser.set_defaults(use_old_atom_encoder=True)
    parser.add_argument(
        "--use_density_summary",
        action="store_true",
        default=False,
        help="Append per-pose density summary to the confidence head. "
             "Disabled by default to preserve the pretrained confidence head.",
    )
    parser.add_argument("--embedding_type", type=str, default="sinusoidal")
    parser.add_argument("--embedding_scale", type=int, default=10000)

    parser.add_argument("--tr_sigma_min", type=float, default=0.1)
    parser.add_argument("--tr_sigma_max", type=float, default=19.0)
    parser.add_argument("--rot_sigma_min", type=float, default=0.03)
    parser.add_argument("--rot_sigma_max", type=float, default=1.55)
    parser.add_argument("--tor_sigma_min", type=float, default=0.0314)
    parser.add_argument("--tor_sigma_max", type=float, default=3.14)

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--scheduler_patience", type=int, default=20)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--run_name", type=str, default="ligand_density_confidence")
    parser.add_argument("--log_dir", type=str, default="workdir_em_confidence_ligand_density")
    parser.add_argument("--main_metric", type=str, default="top1_hit_rate")
    parser.add_argument("--main_metric_goal", type=str, default="max")
    parser.add_argument("--scheduler_metric", type=str, default="ROC AUC")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_model.pt in the run directory")
    return parser


def _freeze_bn(model):
    module = model.module if hasattr(model, "module") else model
    for submodule in module.modules():
        if isinstance(submodule, _BN_TYPES):
            submodule.eval()
            submodule.momentum = 0.0


def _get_state_dict(obj):
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj


def _load_pretrained_backbone(model, ckpt_path):
    state = _get_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model_state = model.state_dict()

    filtered_state = {}
    skipped_shape = []
    for key, value in state.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        filtered_state[key] = value

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded pretrained weights from {ckpt_path}")
    if skipped_shape:
        print(f"  skipped shape-mismatched keys: {len(skipped_shape)}")
        for key, src_shape, dst_shape in skipped_shape[:20]:
            print(f"    {key}: ckpt{src_shape} != model{dst_shape}")
        if len(skipped_shape) > 20:
            print(f"    ... and {len(skipped_shape) - 20} more")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")


def _is_data_parallel(model):
    return hasattr(model, "module") and type(model).__name__ in {"DataParallel", "DistributedDataParallel"}


def _core_model(model):
    return model.module if _is_data_parallel(model) else model


def _state_dict(model):
    return _core_model(model).state_dict()


def _ddp_enabled():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank():
    return int(os.environ.get("RANK", "0"))


def _is_main_process():
    return _rank() == 0


def _setup_distributed():
    if not _ddp_enabled():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch.device(f"cuda:{local_rank}")


def _sync_skip_decision(local_skip: bool, device):
    """Return True if any DDP rank needs to skip the current step."""
    if not _ddp_enabled():
        return bool(local_skip)
    skip_tensor = torch.tensor([1 if local_skip else 0], device=device, dtype=torch.int)
    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
    return bool(skip_tensor.item())


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_density_layer(name):
    """Parameters that should train from epoch 0 (warmup phase).

    Excludes confidence_predictor intentionally: the pretrained head last layer
    is shape-mismatched (ckpt: 2 outputs, model: 1 output) and randomly
    initialised.  Training it during warmup produces exploding val loss before
    the backbone has learned to support it.  The head is unfrozen together with
    LoRA once freeze_pretrained_epochs is over.
    """
    return "density" in name


def _is_density_or_head(name):
    """Parameters that stay trainable after warmup (LoRA + density + head)."""
    return _is_density_layer(name) or "confidence_predictor" in name


def _forward_graphs(model, graphs_or_batch, device):
    if isinstance(graphs_or_batch, list):
        data = Batch.from_data_list(graphs_or_batch).to(device)
        return model(data), data
    data = graphs_or_batch.to(device)
    return model(data), data


def get_optimizer(model, base_lr, density_lr_multiplier, weight_decay=0.0):
    density_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "density" in name or "ligand_density_embedding" in name:
            density_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": base_lr})
    if density_params:
        groups.append({"params": density_params, "lr": base_lr * density_lr_multiplier})
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    cutoff,
    max_grad_norm=1.0,
    ema_weights=None,
    gradient_accumulation_steps=1,
    show_progress=True,
):
    model.train()
    _freeze_bn(model)
    meter = AverageMeter(["confidence_loss"])
    optimizer.zero_grad()
    accumulation_steps = max(int(gradient_accumulation_steps), 1)

    for step, batch in enumerate(tqdm(loader, total=len(loader), disable=not show_progress), start=1):
        pred, batch = _forward_graphs(model, batch, device)
        pred = pred[0] if isinstance(pred, tuple) else pred

        data_list = batch if isinstance(batch, list) else batch.to_data_list()
        if isinstance(cutoff, list):
            labels = torch.cat([g.y_binned for g in data_list]).to(device)
            loss = F.cross_entropy(pred, labels)
        else:
            labels = torch.cat([g.y for g in data_list]).to(device)
            loss = F.binary_cross_entropy_with_logits(pred, labels)

        if _sync_skip_decision(not torch.isfinite(loss), device):
            optimizer.zero_grad()
            continue
        (loss / accumulation_steps).backward()
        if step % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if _sync_skip_decision(has_nan_grad, device):
                optimizer.zero_grad()
                continue
            optimizer.step()
            optimizer.zero_grad()
            if ema_weights is not None:
                ema_weights.update(model.parameters())
        meter.add([loss.detach().cpu()])

    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not _sync_skip_decision(has_nan_grad, device):
            optimizer.step()
            if ema_weights is not None:
                ema_weights.update(model.parameters())
        optimizer.zero_grad()

    return meter.summary()


def train_epoch_pairwise(
    model,
    loader,
    optimizer,
    device,
    margin=1.0,
    bce_weight=0.1,
    max_grad_norm=1.0,
    ema_weights=None,
    gradient_accumulation_steps=1,
    show_progress=True,
):
    model.train()
    _freeze_bn(model)
    meter = AverageMeter(["confidence_loss", "rank_acc"])
    optimizer.zero_grad()
    accumulation_steps = max(int(gradient_accumulation_steps), 1)

    for step, flat_batch in enumerate(tqdm(loader, total=len(loader), disable=not show_progress), start=1):
        if len(flat_batch) < 2 or len(flat_batch) % 2 != 0:
            continue
        try:
            pred, _ = _forward_graphs(model, flat_batch, device)
            pred = pred[0] if isinstance(pred, tuple) else pred
            pred = pred.view(-1)
            if _sync_skip_decision(not torch.isfinite(pred).all(), device):
                print("Pairwise: NaN/Inf model output, skipping batch")
                optimizer.zero_grad()
                continue

            good_scores = pred[0::2]
            bad_scores = pred[1::2]
            if good_scores.shape != bad_scores.shape or good_scores.numel() == 0:
                continue
            target = torch.ones_like(good_scores)
            rank_loss = F.margin_ranking_loss(
                good_scores, bad_scores, target, margin=margin, reduction="none"
            )
            rank_loss_mean = rank_loss.clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()

            if bce_weight > 0:
                logits = torch.cat([good_scores, bad_scores], dim=0)
                good_labels = torch.cat([g.y for g in flat_batch[0::2]]).to(device).view(-1)
                bad_labels = torch.cat([g.y for g in flat_batch[1::2]]).to(device).view(-1)
                labels = torch.cat([good_labels, bad_labels], dim=0)
                bce = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                ).clamp(max=CONF_SAMPLE_LOSS_CLAMP).mean()
                loss = rank_loss_mean + bce_weight * bce
            else:
                loss = rank_loss_mean

            if _sync_skip_decision(not torch.isfinite(loss), device):
                optimizer.zero_grad()
                continue
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                has_nan_grad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
                if _sync_skip_decision(has_nan_grad, device):
                    print("WARNING: NaN/Inf gradient, skipping optimizer step")
                    optimizer.zero_grad()
                    continue
                optimizer.step()
                has_bad_param = any(
                    not torch.isfinite(p).all()
                    or (p.requires_grad and p.abs().max().item() > PARAM_MAX_THRESHOLD)
                    for p in model.parameters()
                )
                if _sync_skip_decision(has_bad_param, device):
                    print("WARNING: bad params after step, restoring from EMA")
                    optimizer.zero_grad()
                    if ema_weights is not None:
                        ema_weights.copy_to(model.parameters())
                        optimizer.state.clear()
                else:
                    optimizer.zero_grad()
                    if ema_weights is not None:
                        ema_weights.update(model.parameters())
            rank_acc = (good_scores.detach() > bad_scores.detach()).float().mean()
            meter.add([loss.detach().cpu(), rank_acc.detach().cpu()])

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                names = [getattr(graph, "name", "?") for graph in flat_batch]
                sizes = []
                for graph in flat_batch:
                    n_lig = int(graph["ligand"].num_nodes) if hasattr(graph["ligand"], "num_nodes") else 0
                    n_rec = int(graph["receptor"].num_nodes) if hasattr(graph["receptor"], "num_nodes") else 0
                    n_atom = int(graph["atom"].num_nodes) if hasattr(graph, "atom") else 0
                    sizes.append(f"{getattr(graph, 'name', '?')}:lig={n_lig},rec={n_rec},atom={n_atom}")
                print(f"WARNING: OOM on batch names={names}")
                print("WARNING: OOM graph sizes: " + " | ".join(sizes))
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if _ddp_enabled():
                    raise
                continue
            raise
        except Exception:
            optimizer.zero_grad()
            raise

    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if not _sync_skip_decision(has_nan_grad, device):
            optimizer.step()
            if ema_weights is not None:
                ema_weights.update(model.parameters())
        optimizer.zero_grad()

    return meter.summary()


def test_epoch(model, loader, device, cutoff):
    model.eval()
    meter = AverageMeter(["confidence_loss", "accuracy"], unpooled_metrics=True)
    all_labels = []
    all_preds = []
    pose_records = []
    complex_best = {}
    complex_has_good = {}

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), disable=not _is_main_process()):
            pred, batch = _forward_graphs(model, batch, device)
            pred = pred[0] if isinstance(pred, tuple) else pred
            data_list = batch if isinstance(batch, list) else batch.to_data_list()

            if isinstance(cutoff, list):
                labels = torch.cat([g.y_binned for g in data_list]).to(device)
                loss = F.cross_entropy(pred, labels)
                accuracy = torch.tensor(0.0, device=device)
            else:
                labels = torch.cat([g.y for g in data_list]).to(device)
                loss = F.binary_cross_entropy_with_logits(pred, labels)
                accuracy = ((pred > 0).float() == labels).float().mean()
                all_preds.append(pred.detach().cpu())
                all_labels.append(labels.detach().cpu())

                for i, graph in enumerate(data_list):
                    name = getattr(graph, "name", f"graph_{i}")
                    score = float(pred[i].item())
                    label = float(labels[i].item())
                    pose_records.append((name, score, label))
                    complex_has_good[name] = complex_has_good.get(name, False) or label > 0.5
                    if name not in complex_best or score > complex_best[name][0]:
                        complex_best[name] = (score, label)

            meter.add([loss.detach().cpu(), accuracy.detach().cpu()])

    results = meter.summary()
    if dist.is_available() and dist.is_initialized():
        local_payload = {
            "labels": torch.cat(all_labels).tolist() if all_labels else [],
            "preds": torch.cat(all_preds).tolist() if all_preds else [],
            "pose_records": pose_records,
            "loss_sum": float(meter.acc["confidence_loss"].item()),
            "acc_sum": float(meter.acc["accuracy"].item()),
            "count": int(meter.count),
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_payload)

        all_labels = []
        all_preds = []
        pose_records = []
        loss_sum = 0.0
        acc_sum = 0.0
        count = 0
        for payload in gathered:
            all_labels.extend(payload["labels"])
            all_preds.extend(payload["preds"])
            pose_records.extend(payload["pose_records"])
            loss_sum += payload["loss_sum"]
            acc_sum += payload["acc_sum"]
            count += payload["count"]
        results["confidence_loss"] = loss_sum / max(count, 1)
        results["accuracy"] = acc_sum / max(count, 1)

        complex_best = {}
        complex_has_good = {}
        for name, score, label in pose_records:
            complex_has_good[name] = complex_has_good.get(name, False) or label > 0.5
            if name not in complex_best or score > complex_best[name][0]:
                complex_best[name] = (score, label)

    if all_labels:
        if isinstance(all_labels[0], torch.Tensor):
            labels_cat = torch.cat(all_labels).numpy()
            preds_cat = torch.cat(all_preds).numpy()
        else:
            labels_cat = all_labels
            preds_cat = all_preds
        try:
            results["ROC AUC"] = roc_auc_score(labels_cat, preds_cat)
        except ValueError:
            results["ROC AUC"] = 0.0
    else:
        results["ROC AUC"] = 0.0
    results["top1_hit_rate"] = (
        sum(1 for _, label in complex_best.values() if label > 0.5) / max(len(complex_best), 1)
    )
    results["oracle_top1_hit_rate"] = (
        sum(1 for has_good in complex_has_good.values() if has_good) / max(len(complex_has_good), 1)
    )
    oracle = results["oracle_top1_hit_rate"]
    results["recoverable_top1_hit_rate"] = (
        results["top1_hit_rate"] / oracle if oracle > 0 else 0.0
    )
    results["top1_gap_to_oracle"] = oracle - results["top1_hit_rate"]
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.config:
        config_dict = yaml.safe_load(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)

    if args.keep_hs:
        args.remove_hs = False
    args.config = args.config.name if args.config is not None else None
    # Convert single-element list to scalar so num_confidence_outputs=1 and the
    # pairwise train loop (pred.view(-1)) stays correct.  The confidence head's
    # last layer is intentionally rebuilt from scratch (1 output vs pretrained's
    # 2); the backbone weights are what matter and they load cleanly.
    args.rmsd_classification_cutoff = (
        args.rmsd_classification_cutoff[0]
        if isinstance(args.rmsd_classification_cutoff, list) and len(args.rmsd_classification_cutoff) == 1
        else args.rmsd_classification_cutoff
    )

    if args.pretrained_confidence_config:
        cfg_path = args.pretrained_confidence_config
    else:
        cfg_path = None
        for name in ("model_parameters.yml", "config.yml"):
            candidate = os.path.join(os.path.dirname(args.pretrained_confidence_model), name)
            if os.path.exists(candidate):
                cfg_path = candidate
                break
    if cfg_path and os.path.exists(cfg_path):
        print(f"Loading pretrained confidence model config from {cfg_path}")
        with open(cfg_path) as handle:
            pretrained_cfg = yaml.safe_load(handle)
        # Architecture keys that must match the pretrained AAOldModel exactly.
        # Only override if the user hasn't changed the arg from its parser default.
        arch_keys = [
            "ns", "nv", "num_conv_layers",
            "sigma_embed_dim", "distance_embed_dim", "cross_distance_embed_dim",
            "max_radius", "cross_max_distance", "dynamic_max_cross",
            "sh_lmax", "use_second_order_repr", "no_batch_norm", "dropout",
            "scale_by_sigma", "norm_by_sigma", "no_torsion",
            "use_old_atom_encoder",
            "embedding_type", "embedding_scale",
            "tr_sigma_min", "tr_sigma_max",
            "rot_sigma_min", "rot_sigma_max",
            "tor_sigma_min", "tor_sigma_max",
            "rmsd_classification_cutoff",
        ]
        for key in arch_keys:
            if key in pretrained_cfg and hasattr(args, key):
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, pretrained_cfg[key])
    else:
        print("WARNING: No pretrained confidence config found. Using default arch args.")

    device = torch.device(args.device) if args.device else _setup_distributed()
    is_distributed = _ddp_enabled()
    run_dir = os.path.join(args.log_dir, args.run_name)
    if _is_main_process():
        os.makedirs(run_dir, exist_ok=True)
        save_yaml_file(os.path.join(run_dir, "model_parameters.yml"), vars(args))

    graph_cache_train = args.graph_cache_train or os.path.join(args.graph_cache_path, "train")
    graph_cache_val = args.graph_cache_val or os.path.join(args.graph_cache_path, "val")

    train_dataset = LigandDensityConfidenceDataset(
        data_dir=args.data_dir,
        split_path=args.split_train,
        map_csv=args.map_csv,
        pose_cache_dir=args.pose_cache_train,
        graph_cache_path=graph_cache_train,
        density_cache_dir=args.density_cache_train or os.path.join(args.density_cache_root, "train"),
        shell_radii=args.shell_radii,
        cache_ids_to_combine=args.cache_ids_train,
        all_atoms=True,
        receptor_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors,
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
        exclude_complexes=args.exclude_complexes,
        max_ligand_nodes=args.max_ligand_nodes,
        max_receptor_nodes=args.max_receptor_nodes,
        max_atom_nodes=args.max_atom_nodes,
        max_total_nodes=args.max_total_nodes,
    )
    val_dataset = LigandDensityConfidenceDataset(
        data_dir=args.data_dir,
        split_path=args.split_val,
        map_csv=args.map_csv,
        pose_cache_dir=args.pose_cache_val,
        graph_cache_path=graph_cache_val,
        density_cache_dir=args.density_cache_val or os.path.join(args.density_cache_root, "val"),
        shell_radii=args.shell_radii,
        cache_ids_to_combine=args.cache_ids_val,
        all_atoms=True,
        receptor_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors,
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
        balance=False,
        limit_complexes=args.limit_complexes,
        exclude_complexes=args.exclude_complexes,
        max_ligand_nodes=args.max_ligand_nodes,
        max_receptor_nodes=args.max_receptor_nodes,
        max_atom_nodes=args.max_atom_nodes,
        max_total_nodes=args.max_total_nodes,
    )

    timestep_emb_func = get_timestep_embedding(args.embedding_type, args.sigma_embed_dim, args.embedding_scale)
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    model = AAOldLigandDensityModel(
        t_to_sigma=t_to_sigma,
        device=device,
        timestep_emb_func=timestep_emb_func,
        lm_embedding_type="esm" if args.esm_embeddings_path is not None else None,
        no_torsion=args.no_torsion,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        norm_by_sigma=args.norm_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        ns=args.ns,
        nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        confidence_mode=True,
        confidence_dropout=args.confidence_dropout,
        confidence_no_batchnorm=args.confidence_no_batchnorm,
        num_confidence_outputs=(
            len(args.rmsd_classification_cutoff) + 1
            if isinstance(args.rmsd_classification_cutoff, list) else 1
        ),
        no_aminoacid_identities=False,
        use_old_atom_encoder=args.use_old_atom_encoder,
        ligand_density_dim=train_dataset.atom_feature_dim,
        density_summary_dim=train_dataset.summary_feature_dim if args.use_density_summary else 0,
    ).to(device)
    _load_pretrained_backbone(model, args.pretrained_confidence_model)

    if args.use_lora:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        target_modules = (
            set(m.strip() for m in args.lora_target_modules.split(","))
            if args.lora_target_modules
            else None
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
        for name, param in model.named_parameters():
            if _is_density_or_head(name):
                param.requires_grad = True
        print_trainable_params(model)

    _freeze_bn(model)
    model = model.to(device)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        if _is_main_process():
            print(f"Using DistributedDataParallel with world_size={dist.get_world_size()}")

    optimizer = get_optimizer(model, args.lr, args.density_lr_multiplier, args.w_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if args.main_metric_goal == "max" else "min",
        factor=0.7,
        patience=args.scheduler_patience,
        min_lr=args.lr / 100,
    )
    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

    start_epoch = 0
    best_metric = float("-inf") if args.main_metric_goal == "max" else float("inf")
    best_scheduler_metric = float("-inf") if args.main_metric_goal == "max" else float("inf")
    best_epoch = -1

    if args.resume:
        ckpt_path = os.path.join(run_dir, "last_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            _core_model(model).load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            ema_weights.load_state_dict(ckpt["ema_weights"], device)
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", best_metric)
            best_scheduler_metric = ckpt.get("best_scheduler_metric", best_scheduler_metric)
            best_epoch = ckpt.get("best_epoch", best_epoch)
            if _is_main_process():
                print(f"Resumed from {ckpt_path} (epoch {ckpt['epoch']} → starting epoch {start_epoch})")
        else:
            if _is_main_process():
                print(f"WARNING: --resume set but {ckpt_path} not found, starting from scratch")

    if args.pairwise_loss:
        train_pair_dataset = LigandDensityConfidencePairDataset(
            train_dataset,
            rmsd_cutoff=args.rmsd_classification_cutoff,
            min_rmsd_gap=args.pairwise_min_rmsd_gap,
            hard_negative_max_rmsd=args.pairwise_hard_negative_max_rmsd,
            hard_pair_fraction=args.pairwise_hard_fraction,
        )
        if args.max_train_pairs_per_epoch > 0 and len(train_pair_dataset) > args.max_train_pairs_per_epoch:
            train_pair_dataset.valid_names = train_pair_dataset.valid_names[: args.max_train_pairs_per_epoch]
            if _is_main_process():
                print(f"Capped pairwise training to {len(train_pair_dataset)} pairs/epoch")
        train_sampler = (
            DistributedSampler(train_pair_dataset, shuffle=True, drop_last=True)
            if is_distributed
            else None
        )
        train_loader = TorchDataLoader(
            train_pair_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=pairwise_collate,
            drop_last=True,
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
            drop_last=True,
        )
    val_pose_dataset = LigandDensityConfidenceAllPosesDataset(
        val_dataset,
        max_poses_per_complex=args.val_max_poses_per_complex,
        include_best_pose=not args.val_exclude_oracle_pose,
    )
    fast_val_pose_dataset = LigandDensityConfidenceAllPosesDataset(
        val_dataset,
        max_poses_per_complex=args.fast_val_max_poses_per_complex,
        include_best_pose=not args.val_exclude_oracle_pose,
    )
    if _is_main_process():
        val_pose_mode = (
            "all poses"
            if not args.val_max_poses_per_complex
            else f"up to {args.val_max_poses_per_complex} poses/complex"
        )
        fast_val_pose_mode = (
            "all poses"
            if not args.fast_val_max_poses_per_complex
            else f"up to {args.fast_val_max_poses_per_complex} poses/complex"
        )
        print(
            f"Fast validation uses {fast_val_pose_mode}: {len(fast_val_pose_dataset)} poses "
            f"from {len(val_dataset)} complexes"
        )
        print(
            f"Full validation uses {val_pose_mode}: {len(val_pose_dataset)} poses "
            f"from {len(val_dataset)} complexes"
        )
    fast_val_sampler = (
        DistributedSampler(fast_val_pose_dataset, shuffle=False, drop_last=False)
        if is_distributed
        else None
    )
    fast_val_loader = DataLoader(
        fast_val_pose_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=fast_val_sampler,
        num_workers=args.num_workers,
    )
    val_sampler = (
        DistributedSampler(val_pose_dataset, shuffle=False, drop_last=False)
        if is_distributed
        else None
    )
    val_loader = DataLoader(
        val_pose_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
    )

    for epoch in range(start_epoch, args.n_epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.freeze_pretrained_epochs > 0:
            if epoch < args.freeze_pretrained_epochs:
                for name, param in model.named_parameters():
                    param.requires_grad = _is_density_layer(name)
                if epoch == 0:
                    print(
                        "Density-only warmup: freezing backbone, LoRA, and confidence head for "
                        f"{args.freeze_pretrained_epochs} epochs"
                    )
                    ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
            elif epoch == args.freeze_pretrained_epochs:
                print(f"Unfreezing confidence head + LoRA at epoch {epoch}")
                if args.use_lora:
                    for name, param in model.named_parameters():
                        param.requires_grad = _is_density_or_head(name) or "lora_" in name
                else:
                    for param in model.parameters():
                        param.requires_grad = True
                ema_weights = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)

        if args.pairwise_loss:
            train_metrics = train_epoch_pairwise(
                model,
                train_loader,
                optimizer,
                device,
                margin=args.pairwise_margin,
                bce_weight=args.pairwise_bce_weight,
                max_grad_norm=args.grad_clip_norm,
                ema_weights=ema_weights,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                show_progress=_is_main_process(),
            )
        else:
            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cutoff=args.rmsd_classification_cutoff,
                max_grad_norm=args.grad_clip_norm,
                ema_weights=ema_weights,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                show_progress=_is_main_process(),
            )
        if _is_main_process():
            print(f"Epoch {epoch}: train {train_metrics}")

        state_dict = _state_dict(model)
        ema_state = ema_weights.state_dict()
        epoch_num = epoch + 1
        run_fast_validation = args.fast_val_every > 0 and (
            epoch_num % args.fast_val_every == 0 or epoch_num == args.n_epochs
        )
        run_full_validation = args.val_every > 0 and (
            epoch_num % args.val_every == 0 or epoch_num == args.n_epochs
        )
        if run_fast_validation or run_full_validation:
            ema_weights.store(model.parameters())
            ema_weights.copy_to(model.parameters())
            ema_model_state = _state_dict(model)

            fast_val_metrics = None
            if run_fast_validation:
                fast_val_metrics = test_epoch(
                    _core_model(model), fast_val_loader, device, args.rmsd_classification_cutoff
                )
                if _is_main_process():
                    print(f"Epoch {epoch}: fast_val {fast_val_metrics}")

            val_metrics = None
            if run_full_validation:
                val_metrics = test_epoch(
                    _core_model(model), val_loader, device, args.rmsd_classification_cutoff
                )
                if _is_main_process():
                    print(f"Epoch {epoch}: full_val {val_metrics}")
            ema_weights.restore(model.parameters())

            # Keep checkpoint selection and LR scheduling aligned:
            # use full validation as the authoritative signal whenever available,
            # and fall back to fast validation only when full validation is disabled.
            metric_source = val_metrics if val_metrics is not None else fast_val_metrics
            metric = metric_source.get(args.main_metric, 0.0) if metric_source is not None else 0.0
            scheduler_source = val_metrics if val_metrics is not None else fast_val_metrics
            scheduler_metric_name = args.scheduler_metric or args.main_metric
            scheduler_metric = (
                scheduler_source.get(scheduler_metric_name, metric)
                if scheduler_source is not None
                else metric
            )
            if is_distributed:
                metric_tensor = torch.tensor([float(metric)], device=device)
                dist.broadcast(metric_tensor, src=0)
                metric = float(metric_tensor.item())
                scheduler_metric_tensor = torch.tensor([float(scheduler_metric)], device=device)
                dist.broadcast(scheduler_metric_tensor, src=0)
                scheduler_metric = float(scheduler_metric_tensor.item())
            if scheduler_source is not None:
                scheduler.step(scheduler_metric)

            if val_metrics is not None:
                improved = (
                    metric > best_metric if args.main_metric_goal == "max" else metric < best_metric
                )
                tied_primary = abs(metric - best_metric) <= 1e-8
                improved_secondary = (
                    scheduler_metric > best_scheduler_metric
                    if args.main_metric_goal == "max"
                    else scheduler_metric < best_scheduler_metric
                )
                if _is_main_process() and improved:
                    best_metric = metric
                    best_scheduler_metric = scheduler_metric
                    best_epoch = epoch
                    torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))
                    torch.save(ema_model_state, os.path.join(run_dir, "best_ema_model.pt"))
                elif _is_main_process() and tied_primary and improved_secondary:
                    best_scheduler_metric = scheduler_metric
                    best_epoch = epoch
                    torch.save(state_dict, os.path.join(run_dir, "best_model.pt"))
                    torch.save(ema_model_state, os.path.join(run_dir, "best_ema_model.pt"))
        elif _is_main_process():
            fast_reason = "disabled" if args.fast_val_every <= 0 else f"fast_val_every={args.fast_val_every}"
            full_reason = "disabled" if args.val_every <= 0 else f"val_every={args.val_every}"
            print(f"Epoch {epoch}: fast validation skipped ({fast_reason})")
            print(f"Epoch {epoch}: full validation skipped ({full_reason})")

        if _is_main_process():
            torch.save(
                {
                    "epoch": epoch,
                    "model": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema_weights": ema_state,
                    "best_metric": best_metric,
                    "best_scheduler_metric": best_scheduler_metric,
                    "best_epoch": best_epoch,
                },
                os.path.join(run_dir, "last_model.pt"),
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_distributed:
            dist.barrier()

    if _is_main_process():
        print(
            f"Best {args.main_metric}: {best_metric:.4f} at epoch {best_epoch} "
            f"(scheduler_metric={args.scheduler_metric}, best_scheduler={best_scheduler_metric:.4f})"
        )
    _cleanup_distributed()


if __name__ == "__main__":
    main()
