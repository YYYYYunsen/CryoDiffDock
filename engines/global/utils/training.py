import copy
import os
import traceback
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

SINGLE_LOSS_MAX = 10.0   # per-sample loss threshold for fallback training
BATCH_LOSS_MAX  = 75.0   # batch-level bad-loss detection threshold
SAMPLE_LOSS_CLAMP = 50.0  # soft per-sample loss cap for training: clamp before batch mean (reduces weight of noisy samples without skipping them)
VAL_LOSS_CLAMP = 10.0     # stricter clamp for val: exclude outlier samples from val metric (normal val loss ~1.8-2.2, so 3.0 = ~1.5x normal max)


_SCORE_NORM_DEBUG_PRINTED = 0

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule


def loss_function(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False):
    global _SCORE_NORM_DEBUG_PRINTED
    # Check if data is a list (DataParallel multi-GPU) or single batch
    is_data_list = isinstance(data, (list, tuple))
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if is_data_list else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    # Normalize to 2D (N, 3) — defensive guard against 1D (3,) shapes from some data paths
    if is_data_list:
        tr_score = torch.cat([d.tr_score.reshape(-1, 3) for d in data], dim=0)
    else:
        tr_score = data.tr_score.reshape(-1, 3) if data.tr_score.dim() == 1 else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component
    if is_data_list:
        rot_score = torch.cat([d.rot_score.reshape(-1, 3) for d in data], dim=0)
    else:
        rot_score = data.rot_score.reshape(-1, 3) if data.rot_score.dim() == 1 else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1).clamp(min=1e-6)
    if os.environ.get("DEBUG_SCORE_NORM") and _SCORE_NORM_DEBUG_PRINTED < 20:
        print(
            "score_norm_debug | "
            f"rot_sigma_min={rot_sigma.min().item():.4e} "
            f"rot_sigma_max={rot_sigma.max().item():.4e} "
            f"rot_score_norm_min={rot_score_norm.min().item():.4e} "
            f"rot_score_norm_max={rot_score_norm.max().item():.4e}"
        )
        _SCORE_NORM_DEBUG_PRINTED += 1
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if is_data_list else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if is_data_list else data.tor_score
        if edge_tor_sigma.numel() == 0:
            # Rigid ligand(s) with no rotatable bonds — skip torsion loss
            num_graphs = len(data) if is_data_list else data.num_graphs
            if apply_mean:
                tor_loss = torch.zeros(1, dtype=torch.float)
                tor_base_loss = torch.zeros(1, dtype=torch.float)
            else:
                tor_loss = torch.zeros(num_graphs, dtype=torch.float)
                tor_base_loss = torch.zeros(num_graphs, dtype=torch.float)
        else:
            tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float().clamp(min=1e-6)
            if os.environ.get("DEBUG_SCORE_NORM") and _SCORE_NORM_DEBUG_PRINTED < 20:
                print(
                    "score_norm_debug | "
                    f"tor_sigma_min={edge_tor_sigma.min().item():.4e} "
                    f"tor_sigma_max={edge_tor_sigma.max().item():.4e} "
                    f"tor_score_norm_min={tor_score_norm2.min().item():.4e} "
                    f"tor_score_norm_max={tor_score_norm2.max().item():.4e}"
                )
                _SCORE_NORM_DEBUG_PRINTED += 1
            tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
            tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        if apply_mean and edge_tor_sigma.numel() > 0:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        elif edge_tor_sigma.numel() > 0:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if is_data_list else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            index = index.cpu()  # Ensure index is on CPU for aggregation
            num_graphs = len(data) if is_data_list else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if backbone_weight > 0:
        backbone_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data], dim=0) if is_data_list else data['receptor'].side_chain_vecs
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)], dim=0).long() if is_data_list else data['receptor'].batch
            index = index.cpu()  # Ensure index is on CPU for aggregation
            num_graphs = len(data) if is_data_list else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    if sidechain_weight > 0:
        sidechain_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data],
                                  dim=0) if is_data_list else data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)],
                              dim=0).long() if is_data_list else data['receptor'].batch
            index = index.cpu()  # Ensure index is on CPU for aggregation
            num_graphs = len(data) if is_data_list else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           tr_base_loss, rot_base_loss, tor_base_loss, backbone_base_loss, sidechain_base_loss


class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                v = v.cpu()  # Ensure on CPU for aggregation
                idx = interval_idx[type_idx].cpu()  # Ensure index on CPU
                self.count[type_idx].index_add_(0, idx, torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, idx, v)

    def summary(self):
        if self.intervals == 1:
            if self.count == 0:
                # No samples processed, return NaN values
                print("WARNING: No samples processed in this epoch (all batches may have been skipped due to OOM)")
                out = {k: float('nan') for k in self.acc.keys()}
            else:
                out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out


def _freeze_bn(model):
    """Set all BatchNorm modules to eval mode (freeze running stats).

    Used during LoRA finetuning to prevent running_mean/running_var from
    drifting on the small finetuning dataset. Both e3nn.nn.BatchNorm and
    torch.nn.BatchNorm* are handled since they're all nn.Module subclasses.
    """
    for module in model.modules():
        if 'BatchNorm' in type(module).__name__:
            module.eval()


def _is_data_parallel_model(model):
    """Best-effort check for PyG DataParallel wrapper."""
    return hasattr(model, 'module') and type(model).__name__ == 'DataParallel'


def _diagnose_shape_mismatch(data_list):
    """Print which attribute has inconsistent tensor ndim across samples in data_list."""
    if not isinstance(data_list, (list, tuple)) or len(data_list) < 2:
        return

    def _collect_attrs(d):
        """Return dict of attr_path -> tensor shape for all tensors in a HeteroData."""
        attrs = {}
        # Top-level graph attributes (tr_score, rot_score, complex_t, etc.)
        for k, v in d._global_store.items():
            if torch.is_tensor(v):
                attrs[k] = tuple(v.shape)
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    if torch.is_tensor(dv):
                        attrs[f'{k}[{dk!r}]'] = tuple(dv.shape)
        # Node/edge stores
        for store_name, store in list(d._node_store_dict.items()) + list(d._edge_store_dict.items()):
            for k, v in store.items():
                if torch.is_tensor(v):
                    attrs[f'{store_name}.{k}'] = tuple(v.shape)
                elif isinstance(v, dict):
                    for dk, dv in v.items():
                        if torch.is_tensor(dv):
                            attrs[f'{store_name}.{k}[{dk!r}]'] = tuple(dv.shape)
        return attrs

    all_attrs = [_collect_attrs(d) for d in data_list]
    all_keys = set().union(*[a.keys() for a in all_attrs])
    names = [getattr(d, 'name', f'sample_{i}') for i, d in enumerate(data_list)]

    mismatch_found = False
    for key in sorted(all_keys):
        shapes = [a.get(key) for a in all_attrs]
        dims = [len(s) if s is not None else None for s in shapes]
        if len(set(d for d in dims if d is not None)) > 1:
            print(f'  SHAPE MISMATCH attr={key}:')
            for name, shape in zip(names, shapes):
                print(f'    {name}: {shape}')
            mismatch_found = True
    if not mismatch_found:
        print(f'  No ndim mismatch found among {[names]}')


def _split_to_samples(data):
    """Return a list of per-complex Data objects from batch/list inputs."""
    if isinstance(data, (list, tuple)):
        return list(data)
    if hasattr(data, 'to_data_list'):
        return data.to_data_list()
    return [data]


def _batch_single_sample(sample):
    """Wrap a single HeteroData sample as a PyG Batch so node stores get .batch."""
    return Batch.from_data_list([sample])


def _forward_single(model, sample):
    """Run model on one sample for probing/fallback in both single- and multi-GPU."""
    single_batch = _batch_single_sample(sample)
    if _is_data_parallel_model(model):
        return model.module(single_batch)
    return model(single_batch)


def _train_single(model, sample, optimizer, device, t_to_sigma, loss_fn, ema_weights, scaler, max_grad_norm, use_amp, meter, freeze_bn=False):
    """Train on one sample (batch_size=1 fallback)."""
    if freeze_bn:
        _freeze_bn(model)
    try:
        optimizer.zero_grad()
        if use_amp:
            with autocast(enabled=True):
                _p = _forward_single(model, sample)
            _lt = loss_fn(*_p, data=[sample], t_to_sigma=t_to_sigma, device=device, apply_mean=False)
            if _lt is None or torch.any(torch.isnan(_lt[0])):
                return
            _loss_scalar = _lt[0].mean()
            scaler.scale(_loss_scalar).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            _p = _forward_single(model, sample)
            _lt = loss_fn(*_p, data=[sample], t_to_sigma=t_to_sigma, device=device, apply_mean=False)
            if _lt is None or torch.any(torch.isnan(_lt[0])) or _lt[0].abs().max().item() > SINGLE_LOSS_MAX:
                if _lt is not None and not torch.any(torch.isnan(_lt[0])):
                    print(f"  -> _train_single: loss too large ({_lt[0].abs().max().item():.3e}) for {sample.name}, skipping")
                return
            _lt[0].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                print(f"  -> _train_single: NaN/Inf gradient for {sample.name}, skipping update")
                optimizer.zero_grad()
                return
            optimizer.step()
        if ema_weights is not None:
            ema_weights.update(model.parameters())
        # apply_mean=False → _lt[1:] are per-sample vectors; reduce to scalars for meter
        _log_vals = [_lt[0].mean().cpu().detach()]
        for _v in _lt[1:]:
            _log_vals.append(_v.mean().cpu().detach() if _v.dim() > 0 else _v.cpu().detach())
        meter.add(_log_vals)
    except Exception as _fe:
        print(f"  -> fallback train ERROR {sample.name}: {_fe}")



def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights, scaler=None, freeze_bn=False, max_grad_norm=1.0):
    model.train()
    if freeze_bn:
        _freeze_bn(model)
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'])

    use_amp = scaler is not None

    for data in tqdm(loader, total=len(loader)):
        # Handle both list (DataParallel multi-GPU) and single batch (single GPU)
        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                print("Skipping batch of size 1 since otherwise batchnorm would not work.")
                continue
            data = [d.to(device) for d in data]
        else:
            if data.num_graphs == 1:
                print("Skipping batch of size 1 since otherwise batchnorm would not work.")
                continue
            data = data.to(device)
        optimizer.zero_grad()
        try:
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
                # Detect NaN in raw outputs BEFORE nan_to_num — NaN here means NaN gradients
                # will flow back and corrupt model weights even if loss looks fine after nan_to_num
                raw_has_nan = (
                    torch.any(torch.isnan(tr_pred)) or
                    torch.any(torch.isnan(rot_pred)) or
                    (tor_pred is not None and torch.any(torch.isnan(tor_pred)))
                )
                if raw_has_nan:
                    names = [d.name for d in data] if isinstance(data, (list, tuple)) else data.name
                    print(f"Train: NaN in raw model output for {names}, probing to find culprit...")
                    nan_log = os.environ.get('NAN_LOG_FILE', 'nan_complexes.txt')
                    data_list = _split_to_samples(data)
                    ok_samples = []
                    for _single in data_list:
                        try:
                            with torch.no_grad():
                                _p = _forward_single(model, _single)
                            _single_has_nan = (
                                torch.any(torch.isnan(_p[0])) or
                                torch.any(torch.isnan(_p[1])) or
                                (_p[2] is not None and torch.any(torch.isnan(_p[2])))
                            )
                            if _single_has_nan:
                                print(f"  -> CULPRIT: {_single.name}")
                                with open(nan_log, 'a') as _f:
                                    _f.write(f"{_single.name}\tnan_in=[raw_model_output]\n")
                            else:
                                print(f"  -> OK: {_single.name}")
                                ok_samples.append(_single)
                        except Exception as _e:
                            print(f"  -> ERROR probing {_single.name}: {_e}")
                            with open(nan_log, 'a') as _f:
                                _f.write(f"{_single.name}\terror={_e}\n")
                    if ok_samples:
                        print(f"  -> Fallback: training {len(ok_samples)} OK samples individually")
                        for _s in ok_samples:
                            _train_single(model, _s, optimizer, device, t_to_sigma, loss_fn,
                                          ema_weights, scaler, max_grad_norm, use_amp, meter,
                                          freeze_bn=freeze_bn)
                    continue
                tr_pred = torch.nan_to_num(tr_pred, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
                rot_pred = torch.nan_to_num(rot_pred, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
                if tor_pred is not None:
                    tor_pred = torch.nan_to_num(tor_pred, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
                loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, device=device, apply_mean=False)

            if loss_tuple is None:
                print("None loss tuple, skipping")
                continue
            per_sample_loss = loss_tuple[0]
            # Soft per-sample clamping: cap each sample's loss before averaging.
            # Samples with loss > SAMPLE_LOSS_CLAMP still contribute (not skipped)
            # but at reduced weight, preventing a few noisy samples from dominating.
            clamped = per_sample_loss.clamp(max=SAMPLE_LOSS_CLAMP)
            loss = clamped.mean()

            loss_is_bad = (torch.any(torch.isnan(per_sample_loss)) or
                           torch.any(torch.isinf(per_sample_loss)) or
                           per_sample_loss.abs().max().item() > BATCH_LOSS_MAX)
            if loss_is_bad:
                names = [d.name for d in data] if isinstance(data, (list, tuple)) else data.name
                _loss_names = ['tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss']
                _nan_components = [
                    name for name, val in zip(_loss_names, loss_tuple[1:6])
                    if torch.any(torch.isnan(val)) or torch.any(torch.isinf(val)) or val.abs().max().item() > BATCH_LOSS_MAX
                ]
                print(f"Bad loss | complexes={names} | bad_in={_nan_components} | max_sample_loss={per_sample_loss.abs().max().item():.3e}")

                nan_log = os.environ.get('NAN_LOG_FILE', 'nan_complexes.txt')
                data_list = _split_to_samples(data)
                ok_samples = []
                for _single in data_list:
                    try:
                        with torch.no_grad():
                            _pred = _forward_single(model, _single)
                            _lt = loss_fn(*_pred, data=[_single], t_to_sigma=t_to_sigma, device=device)
                        _single_loss = _lt[0]
                        _single_nan = [
                            name for name, val in zip(_loss_names, _lt[1:6])
                            if torch.any(torch.isnan(val)) or val.abs().max().item() > BATCH_LOSS_MAX
                        ]
                        _single_vals = {name: val.mean().item() for name, val in zip(_loss_names, _lt[1:6])}
                        _single_is_bad = (torch.any(torch.isnan(_single_loss)) or
                                          torch.any(torch.isinf(_single_loss)) or
                                          _single_loss.abs().max().item() > BATCH_LOSS_MAX)
                        if _single_is_bad:
                            print(f"  -> CULPRIT: {_single.name} | nan_in={_single_nan} | values={_single_vals}")
                            with open(nan_log, 'a') as _f:
                                _f.write(f"{_single.name}\tnan_in={_single_nan}\tvalues={_single_vals}\n")
                        else:
                            print(f"  -> OK: {_single.name} | values={_single_vals}")
                            ok_samples.append(_single)
                    except Exception as _e:
                        print(f"  -> ERROR probing {_single.name}: {_e}")
                        with open(nan_log, 'a') as _f:
                            _f.write(f"{_single.name}\terror={_e}\n")
                if ok_samples:
                    print(f"  -> Fallback: training {len(ok_samples)} OK samples individually")
                    for _s in ok_samples:
                        _train_single(model, _s, optimizer, device, t_to_sigma, loss_fn,
                                      ema_weights, scaler, max_grad_norm, use_amp, meter,
                                      freeze_bn=freeze_bn)
                continue

            # Mixed precision backward pass
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            if ema_weights is not None:
                ema_weights.update(model.parameters())

            # loss_tuple[1:] are per-sample vectors (apply_mean=False); reduce to scalars for meter
            _log_vals = [loss.cpu().detach()]
            for _v in loss_tuple[1:]:
                _log_vals.append(_v.mean().cpu().detach() if _v.dim() > 0 else _v.cpu().detach())
            meter.add(_log_vals)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            else:
                print(f'| WARNING: RuntimeError in train_epoch: {e}')
                traceback.print_exc()
                if 'same number of dimensions' in str(e) and isinstance(data, (list, tuple)):
                    print('  Diagnosing shape mismatch...')
                    try:
                        _diagnose_shape_mismatch(data)
                    except Exception as _de:
                        print(f'  Diagnosis failed: {_de}')
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue

    return meter.summary()


def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False, use_amp=True, freeze_bn=False):
    model.eval()
    if not freeze_bn:
        # Legacy: use train mode BN when not explicitly freezing BN stats,
        # to match the old behavior where running stats drifted during training.
        for module in model.modules():
            if 'BatchNorm' in type(module).__name__:
                module.train()
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
                         unpooled_metrics=True)
    # Track "real" val loss (before VAL_LOSS_CLAMP) on the same valid samples
    # that are included in meter. This helps compare clipped vs unclipped trends.
    unclipped_loss_sum = 0.0
    unclipped_loss_count = 0

    if test_sigma_intervals:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        # Handle both list (DataParallel multi-GPU) and single batch (single GPU)
        if isinstance(data, (list, tuple)):
            if len(data) == 1:
                print("Skipping val batch of size 1 since otherwise batchnorm would not work.")
                continue
            data = [d.to(device) for d in data]
        else:
            if data.num_graphs == 1:
                print("Skipping val batch of size 1 since otherwise batchnorm would not work.")
                continue
            data = data.to(device)
        try:
            # Debug: check for NaN in input data
            if isinstance(data, (list, tuple)):
                for d in data:
                    if hasattr(d, 'ring_features') and d.ring_features is not None:
                        if torch.any(torch.isnan(d.ring_features)):
                            print(f"Val: NaN in input ring_features for {d.name}")
                    if hasattr(d['receptor'], 'x') and torch.any(torch.isnan(d['receptor'].x)):
                        print(f"Val: NaN in receptor features for {d.name}")
            else:
                if hasattr(data, 'ring_features') and data.ring_features is not None:
                    if torch.any(torch.isnan(data.ring_features)):
                        print(f"Val: NaN in input ring_features")
                if hasattr(data['receptor'], 'x') and torch.any(torch.isnan(data['receptor'].x)):
                    print(f"Val: NaN in receptor features")

            with torch.no_grad(), autocast(enabled=use_amp):
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)

            # Check for NaN in predictions — fall back to per-sample eval
            _raw_nan = (torch.any(torch.isnan(tr_pred)) or torch.any(torch.isnan(rot_pred)) or
                        (tor_pred is not None and torch.any(torch.isnan(tor_pred))))
            if _raw_nan:
                names = [d.name for d in data] if isinstance(data, (list, tuple)) else data.name
                print(f"Val: NaN in predictions for {names}, falling back to per-sample eval")
                data_list = _split_to_samples(data)
                for _single in data_list:
                    try:
                        with torch.no_grad(), autocast(enabled=use_amp):
                            _tp = _forward_single(model, _single)
                        if any(torch.any(torch.isnan(t)) for t in _tp if t is not None):
                            print(f"  -> Val CULPRIT (NaN pred): {_single.name}, skipping")
                            continue
                        _lt = loss_fn(*_tp, data=[_single], t_to_sigma=t_to_sigma, apply_mean=False, device=device)
                        if _lt is None:
                            continue
                        _lv = _lt[0].cpu().detach()
                        if _lv.numel() == 0 or torch.any(torch.isnan(_lv)):
                            print(f"  -> Val bad loss for {_single.name}, skipping")
                            continue
                        if _lv.abs().max().item() > BATCH_LOSS_MAX:
                            print(f"  -> Val loss too large ({_lv.abs().max().item():.3e}) for {_single.name}, skipping")
                            continue
                        _lv_max = _lv.abs().max().item()
                        if _lv_max > VAL_LOSS_CLAMP:
                            print(f"  -> Val high loss ({_lv_max:.3f}) for {_single.name}, clamping to {VAL_LOSS_CLAMP}")
                        unclipped_loss_sum += float(_lv.sum().item())
                        unclipped_loss_count += int(_lv.numel())
                        meter.add([_lv.clamp(max=VAL_LOSS_CLAMP), *_lt[1:]])
                        print(f"  -> Val OK: {_single.name}")
                    except RuntimeError as _se:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                    except Exception as _se:
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                continue

            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            if loss_tuple is None:
                print("Val: loss_tuple is None, skipping")
                continue
            # Check for empty, NaN, or extreme values in validation loss — fall back to per-sample eval
            loss_val = loss_tuple[0].cpu().detach()
            if loss_val.numel() == 0 or torch.any(torch.isnan(loss_val)) or loss_val.abs().max().item() > BATCH_LOSS_MAX:
                names = [d.name for d in data] if isinstance(data, (list, tuple)) else data.name
                print(f"Val: NaN/empty loss for {names}, falling back to per-sample eval")
                data_list = _split_to_samples(data)
                for _single in data_list:
                    try:
                        with torch.no_grad(), autocast(enabled=use_amp):
                            _tp = _forward_single(model, _single)
                        _lt = loss_fn(*_tp, data=[_single], t_to_sigma=t_to_sigma, apply_mean=False, device=device)
                        if _lt is None:
                            continue
                        _lv = _lt[0].cpu().detach()
                        if _lv.numel() == 0 or torch.any(torch.isnan(_lv)):
                            print(f"  -> Val CULPRIT (NaN loss): {_single.name}, skipping")
                            continue
                        if _lv.abs().max().item() > BATCH_LOSS_MAX:
                            print(f"  -> Val CULPRIT (extreme loss {_lv.abs().max().item():.3e}): {_single.name}, skipping")
                            continue
                        _lv_max = _lv.abs().max().item()
                        if _lv_max > VAL_LOSS_CLAMP:
                            print(f"  -> Val high loss ({_lv_max:.3f}) for {_single.name}, clamping to {VAL_LOSS_CLAMP}")
                        unclipped_loss_sum += float(_lv.sum().item())
                        unclipped_loss_count += int(_lv.numel())
                        meter.add([_lv.clamp(max=VAL_LOSS_CLAMP), *_lt[1:]])
                        print(f"  -> Val OK: {_single.name}")
                    except RuntimeError as _se:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                    except Exception as _se:
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                continue
            # Per-sample eval for high-loss batches (5~10 range) to identify culprits
            loss_val_max = loss_val.abs().max().item()
            if loss_val_max > VAL_LOSS_CLAMP:
                names = [d.name for d in data] if isinstance(data, (list, tuple)) else [data.name]
                print(f"Val high loss ({loss_val_max:.3f}) in batch {names}, probing per-sample...")
                data_list = _split_to_samples(data)
                for _single in data_list:
                    try:
                        with torch.no_grad(), autocast(enabled=use_amp):
                            _tp = _forward_single(model, _single)
                        _lt = loss_fn(*_tp, data=[_single], t_to_sigma=t_to_sigma, apply_mean=False, device=device)
                        if _lt is None:
                            continue
                        _lv = _lt[0].cpu().detach()
                        if _lv.numel() == 0 or torch.any(torch.isnan(_lv)):
                            print(f"  -> Val CULPRIT (NaN loss): {_single.name}, skipping")
                            continue
                        _lv_max = _lv.abs().max().item()
                        if _lv_max > BATCH_LOSS_MAX:
                            print(f"  -> Val CULPRIT (extreme loss {_lv_max:.3e}): {_single.name}, skipping")
                            continue
                        if _lv_max > VAL_LOSS_CLAMP:
                            print(f"  -> Val CULPRIT (high loss {_lv_max:.3f}): {_single.name}, clamping")
                        unclipped_loss_sum += float(_lv.sum().item())
                        unclipped_loss_count += int(_lv.numel())
                        meter.add([_lv.clamp(max=VAL_LOSS_CLAMP), *_lt[1:]])
                    except RuntimeError as _se:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                    except Exception as _se:
                        print(f"  -> Val ERROR {_single.name}: {_se}")
                continue
            unclipped_loss_sum += float(loss_val.sum().item())
            unclipped_loss_count += int(loss_val.numel())
            meter.add([loss_val.clamp(max=VAL_LOSS_CLAMP), *loss_tuple[1:]])

            if test_sigma_intervals > 0:
                is_data_list = isinstance(data, (list, tuple))
                complex_t_tr, complex_t_rot, complex_t_tor = [
                    torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) if is_data_list
                    else data.complex_t[noise_type]
                    for noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr])

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue
            else:
                print(f'| WARNING: RuntimeError: {e}, skipping batch')
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue

    if meter.count == 0:
        print(f"WARNING: No validation samples were processed! Check for NaN losses or batch size issues.")
    out = meter.summary()
    out['loss_unclipped'] = float('nan') if unclipped_loss_count == 0 else (unclipped_loss_sum / unclipped_loss_count)
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out


def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds, min_rmsds = [], []

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        # EMDataset never stores orig_pos, so derive it the way evaluate.py does.
        # orig_complex_graph is untouched by sampling (randomize_position works on
        # the deepcopied data_list), so its pos is still the centred ground truth.
        if args.no_torsion or not hasattr(orig_complex_graph['ligand'], 'orig_pos'):
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),
              'mean_min_rmsd': float(np.mean(min_rmsds)),
              'median_min_rmsd': float(np.median(min_rmsds)),}
    return losses
