import os
import subprocess
import warnings
from datetime import datetime
from typing import List

import numpy
import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile
from torch import nn, Tensor
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.utils import degree, subgraph

from models.aa_model import AAModel
from models.cg_model import CGModel
from models.old_aa_model import AAOldModel
from models.old_cg_model import CGOldModel
from utils.diffusion_utils import get_timestep_embedding


def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)


def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


def unbatch(src, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    if isinstance(src, numpy.ndarray):
        return np.split(src, np.array(sizes).cumsum()[:-1], axis=dim)
    else:
        return src.split(sizes, dim)


def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)


def unbatch_edge_attributes(edge_attributes, edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attributes.split(sizes, dim=0)


def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def unfreeze_layer(model):
    for name, child in (model.named_children()):
        #print(name, child.parameters())
        for param in child.parameters():
            param.requires_grad = True


def get_optimizer_and_scheduler(args, model, scheduler_mode='min', step=0, optimizer=None):
    if args.scheduler == 'layer_linear_warmup':
        if step == 0:
            for name, child in (model.named_children()):
                if name.find('batch_norm') == -1:
                    for name, param in child.named_parameters():
                        if name.find('batch_norm') == -1:
                            param.requires_grad = False

            for l in [model.center_edge_embedding, model.final_conv, model.tr_final_layer, model.rot_final_layer,
                      model.final_edge_embedding, model.final_tp_tor, model.tor_bond_conv, model.tor_final_layer]:
                unfreeze_layer(l)

        elif 0 < step <= args.num_conv_layers:
            unfreeze_layer(model.conv_layers[-step])

        elif step == args.num_conv_layers + 1:
            for l in [model.lig_node_embedding, model.lig_edge_embedding, model.rec_node_embedding, model.rec_edge_embedding,
                      model.rec_sigma_embedding, model.cross_edge_embedding, model.rec_emb_layers, model.lig_emb_layers]:
                unfreeze_layer(l)

    if step == 0 or args.scheduler == 'layer_linear_warmup':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7, patience=args.scheduler_patience, min_lr=args.lr / 100)
    if args.scheduler == 'plateau':
        scheduler = scheduler_plateau
    elif args.scheduler == 'linear_warmup' or args.scheduler == 'layer_linear_warmup':
        if (args.scheduler == 'linear_warmup' and step < 1) or \
                (args.scheduler == 'layer_linear_warmup' and step <= args.num_conv_layers + 1):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_start_factor, end_factor=1.0,
                                                       total_iters=args.warmup_dur)
        else:
            scheduler = scheduler_plateau
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler


def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, old=False):

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type if 'embedding_type' in args else 'sinusoidal',
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale if 'embedding_type' in args else 10000)

    if old:
        if 'all_atoms' in args and args.all_atoms:
            model_class = AAOldModel
        else:
            model_class = CGOldModel

        lm_embedding_type = None
        if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

        model = model_class(t_to_sigma=t_to_sigma,
                            device=device,
                            no_torsion=args.no_torsion,
                            timestep_emb_func=timestep_emb_func,
                            num_conv_layers=args.num_conv_layers,
                            lig_max_radius=args.max_radius,
                            scale_by_sigma=args.scale_by_sigma,
                            sigma_embed_dim=args.sigma_embed_dim,
                            norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
                            ns=args.ns, nv=args.nv,
                            distance_embed_dim=args.distance_embed_dim,
                            cross_distance_embed_dim=args.cross_distance_embed_dim,
                            batch_norm=not args.no_batch_norm,
                            dropout=args.dropout,
                            use_second_order_repr=args.use_second_order_repr,
                            cross_max_distance=args.cross_max_distance,
                            dynamic_max_cross=args.dynamic_max_cross,
                            smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
                            odd_parity=args.odd_parity if "odd_parity" in args else False,
                            lm_embedding_type=lm_embedding_type,
                            confidence_mode=confidence_mode,
                            affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
                            parallel=args.parallel if "parallel" in args else 1,
                            num_confidence_outputs=len(
                                args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                args.rmsd_classification_cutoff, list) else 1,
                            parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
                            fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
                            no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
                            include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
                            use_old_atom_encoder=args.use_old_atom_encoder if hasattr(args, 'use_old_atom_encoder') else True)

    else:
        if 'all_atoms' in args and args.all_atoms:
            model_class = AAModel
        else:
            model_class = CGModel

        lm_embedding_type = None
        if ('moad_esm_embeddings_path' in args and args.moad_esm_embeddings_path is not None) or \
            ('pdbbind_esm_embeddings_path' in args and args.pdbbind_esm_embeddings_path is not None) or \
            ('pdbsidechain_esm_embeddings_path' in args and args.pdbsidechain_esm_embeddings_path is not None) or \
            ('esm_embeddings_path' in args and args.esm_embeddings_path is not None):
            lm_embedding_type = 'precomputed'
        if 'esm_embeddings_model' in args and args.esm_embeddings_model is not None: lm_embedding_type = args.esm_embeddings_model

        model = model_class(t_to_sigma=t_to_sigma,
                            device=device,
                            no_torsion=args.no_torsion,
                            timestep_emb_func=timestep_emb_func,
                            num_conv_layers=args.num_conv_layers,
                            lig_max_radius=args.max_radius,
                            scale_by_sigma=args.scale_by_sigma,
                            sigma_embed_dim=args.sigma_embed_dim,
                            norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
                            ns=args.ns, nv=args.nv,
                            distance_embed_dim=args.distance_embed_dim,
                            cross_distance_embed_dim=args.cross_distance_embed_dim,
                            batch_norm=not args.no_batch_norm,
                            dropout=args.dropout,
                            use_second_order_repr=args.use_second_order_repr,
                            cross_max_distance=args.cross_max_distance,
                            dynamic_max_cross=args.dynamic_max_cross,
                            smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
                            odd_parity=args.odd_parity if "odd_parity" in args else False,
                            lm_embedding_type=lm_embedding_type,
                            confidence_mode=confidence_mode,
                            affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
                            parallel=args.parallel if "parallel" in args else 1,
                            num_confidence_outputs=len(
                                args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                args.rmsd_classification_cutoff, list) else 1,
                            atom_num_confidence_outputs=len(
                                args.atom_rmsd_classification_cutoff) + 1 if 'atom_rmsd_classification_cutoff' in args and isinstance(
                                args.atom_rmsd_classification_cutoff, list) else 1,
                            parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
                            fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
                            no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
                            include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
                            sh_lmax=args.sh_lmax if 'sh_lmax' in args else 2,
                            differentiate_convolutions=not args.no_differentiate_convolutions if "no_differentiate_convolutions" in args else True,
                            tp_weights_layers=args.tp_weights_layers if "tp_weights_layers" in args else 2,
                            num_prot_emb_layers=args.num_prot_emb_layers if "num_prot_emb_layers" in args else 0,
                            reduce_pseudoscalars=args.reduce_pseudoscalars if "reduce_pseudoscalars" in args else False,
                            embed_also_ligand=args.embed_also_ligand if "embed_also_ligand" in args else False,
                            atom_confidence=args.atom_confidence_loss_weight > 0.0 if "atom_confidence_loss_weight" in args else False,
                            sidechain_pred=(hasattr(args, 'sidechain_loss_weight') and args.sidechain_loss_weight > 0) or
                                           (hasattr(args, 'backbone_loss_weight') and args.backbone_loss_weight > 0),
                            depthwise_convolution=args.depthwise_convolution if hasattr(args, 'depthwise_convolution') else False)

    if device.type == 'cuda' and not no_parallel and ('dataset' not in args or not args.dataset == 'torsional'):
        model = DataParallel(model)
    model.to(device)
    return model

import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]


def crop_beyond(complex_graph, cutoff, all_atoms):
    ligand_pos = complex_graph['ligand'].pos
    receptor_pos = complex_graph['receptor'].pos
    residues_to_keep = torch.any(torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1) < cutoff ** 2, dim=1)

    if all_atoms:
        #print(complex_graph['atom'].x.shape, complex_graph['atom'].pos.shape, complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index.shape)
        atom_to_res_mapping = complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index[1]
        atoms_to_keep = residues_to_keep[atom_to_res_mapping]
        rec_remapper = (torch.cumsum(residues_to_keep.long(), dim=0) - 1)
        atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
        atom_res_edge_index = torch.stack([torch.arange(len(atom_to_res_new_mapping), device=atom_to_res_new_mapping.device), atom_to_res_new_mapping])

    complex_graph['receptor'].pos = complex_graph['receptor'].pos[residues_to_keep]
    complex_graph['receptor'].x = complex_graph['receptor'].x[residues_to_keep]
    complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[residues_to_keep]
    # Also crop ring features/vectors if present (for DiffDock-EM)
    if hasattr(complex_graph['receptor'], 'ring_features'):
        complex_graph['receptor'].ring_features = complex_graph['receptor'].ring_features[residues_to_keep]
    if hasattr(complex_graph['receptor'], 'ring_vectors'):
        complex_graph['receptor'].ring_vectors = complex_graph['receptor'].ring_vectors[residues_to_keep]
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = \
        subgraph(residues_to_keep, complex_graph['receptor', 'rec_contact', 'receptor'].edge_index, relabel_nodes=True)[0]

    if all_atoms:
        complex_graph['atom'].x = complex_graph['atom'].x[atoms_to_keep]
        complex_graph['atom'].pos = complex_graph['atom'].pos[atoms_to_keep]
        if hasattr(complex_graph['atom'], 'ring_features'):
            complex_graph['atom'].ring_features = complex_graph['atom'].ring_features[atoms_to_keep]
        if hasattr(complex_graph['atom'], 'ring_vectors'):
            complex_graph['atom'].ring_vectors = complex_graph['atom'].ring_vectors[atoms_to_keep]
        complex_graph['atom', 'atom_contact', 'atom'].edge_index = subgraph(atoms_to_keep, complex_graph['atom', 'atom_contact', 'atom'].edge_index, relabel_nodes=True)[0]
        complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index


def get_em_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, ring_dim=4, ring_vec_dim=4,
                 ring_vec_scale=1.0):
    """Create a DiffDock model with ring feature support for EM finetuning.

    This is similar to get_model() but adds the ring_dim parameter for
    cryo-EM density feature injection.

    Args:
        args: Model configuration arguments
        device: PyTorch device
        t_to_sigma: Time to sigma function
        no_parallel: Disable data parallelism
        confidence_mode: Whether to use confidence prediction mode
        ring_dim: Number of ring feature channels (default: 4 for radii [4,5,6,7] Å)

    Returns:
        Model with ring feature embedding support
    """
    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type if 'embedding_type' in args else 'sinusoidal',
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale if 'embedding_type' in args else 10000)

    if 'all_atoms' in args and args.all_atoms:
        model_class = AAModel
        # AAModel already rescales ring_vectors via normalize_ring_per_complex.
        extra_model_kwargs = {}
    else:
        model_class = CGModel
        # CGModel injects ring_vectors raw, so it needs an explicit input scale.
        extra_model_kwargs = {'ring_vec_scale': ring_vec_scale}

    lm_embedding_type = None
    if ('moad_esm_embeddings_path' in args and args.moad_esm_embeddings_path is not None) or \
        ('pdbbind_esm_embeddings_path' in args and args.pdbbind_esm_embeddings_path is not None) or \
        ('pdbsidechain_esm_embeddings_path' in args and args.pdbsidechain_esm_embeddings_path is not None) or \
        ('esm_embeddings_path' in args and args.esm_embeddings_path is not None):
        lm_embedding_type = 'precomputed'
    if 'esm_embeddings_model' in args and args.esm_embeddings_model is not None:
        lm_embedding_type = args.esm_embeddings_model

    model = model_class(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_emb_func,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
        ns=args.ns, nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=args.dynamic_max_cross,
        smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
        odd_parity=args.odd_parity if "odd_parity" in args else False,
        lm_embedding_type=lm_embedding_type,
        confidence_mode=confidence_mode,
        affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
        parallel=args.parallel if "parallel" in args else 1,
        num_confidence_outputs=len(
            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
            args.rmsd_classification_cutoff, list) else 1,
        atom_num_confidence_outputs=len(
            args.atom_rmsd_classification_cutoff) + 1 if 'atom_rmsd_classification_cutoff' in args and isinstance(
            args.atom_rmsd_classification_cutoff, list) else 1,
        parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
        fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
        no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
        include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
        sh_lmax=args.sh_lmax if 'sh_lmax' in args else 2,
        differentiate_convolutions=not args.no_differentiate_convolutions if "no_differentiate_convolutions" in args else True,
        tp_weights_layers=args.tp_weights_layers if "tp_weights_layers" in args else 2,
        num_prot_emb_layers=args.num_prot_emb_layers if "num_prot_emb_layers" in args else 0,
        reduce_pseudoscalars=args.reduce_pseudoscalars if "reduce_pseudoscalars" in args else False,
        embed_also_ligand=args.embed_also_ligand if "embed_also_ligand" in args else False,
        atom_confidence=args.atom_confidence_loss_weight > 0.0 if "atom_confidence_loss_weight" in args else False,
        sidechain_pred=(hasattr(args, 'sidechain_loss_weight') and args.sidechain_loss_weight > 0) or
                       (hasattr(args, 'backbone_loss_weight') and args.backbone_loss_weight > 0),
        depthwise_convolution=args.depthwise_convolution if hasattr(args, 'depthwise_convolution') else False,
        ring_dim=ring_dim,
        ring_vec_dim=ring_vec_dim,
        confidence_no_batchnorm=args.confidence_no_batchnorm if hasattr(args, 'confidence_no_batchnorm') else False,
        confidence_dropout=args.confidence_dropout if hasattr(args, 'confidence_dropout') else 0,
        **extra_model_kwargs,
    )

    if device.type == 'cuda' and not no_parallel and ('dataset' not in args or not args.dataset == 'torsional'):
        model = DataParallel(model)
    model.to(device)
    return model


def load_pretrained_weights(model, checkpoint_path, strict=False, verbose=True):  # noqa: ARG001 (strict always False; size mismatches are filtered)
    """Load pretrained weights into a model, skipping missing, unexpected, and size-mismatched keys.

    Size-mismatched keys are filtered out before loading so that architectures
    with different hyperparameters (e.g. distance_embed_dim, nv) can still load
    the compatible subset of weights without raising RuntimeError.

    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file (.pt)
        strict: Ignored — always loads with strict=False after filtering size mismatches.
        verbose: Print information about loaded/missing/skipped keys

    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle DataParallel wrapped models
    if hasattr(model, 'module'):
        target_model = model.module
    else:
        target_model = model

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Handle LoRA <-> plain model mismatch.
    # Case: checkpoint was saved from a LoRA model (has 'original_layer' keys),
    # but current model is plain (no LoRA applied yet).
    # Remap X.original_layer.weight/bias -> X.weight/bias and drop lora_A/lora_B.
    checkpoint_has_lora = any('original_layer' in k for k in state_dict.keys())
    model_has_lora = any('original_layer' in k for k in target_model.state_dict().keys())
    if checkpoint_has_lora and not model_has_lora:
        # Case: LoRA checkpoint → plain model
        # Remap X.original_layer.weight/bias -> X.weight/bias, drop lora_A/lora_B
        remapped = {}
        for k, v in state_dict.items():
            if '.original_layer.weight' in k:
                remapped[k.replace('.original_layer.weight', '.weight')] = v
            elif '.original_layer.bias' in k:
                remapped[k.replace('.original_layer.bias', '.bias')] = v
            elif 'lora_A' in k or 'lora_B' in k:
                pass  # drop LoRA adapter weights; will be re-initialized
            else:
                remapped[k] = v
        state_dict = remapped
        if verbose:
            print("Detected LoRA checkpoint loaded into plain model: remapped original_layer weights.")
    elif not checkpoint_has_lora and model_has_lora:
        # Case: plain checkpoint (e.g. original DiffDock-L) → LoRA-wrapped model
        # Remap X.weight/bias -> X.original_layer.weight/bias
        model_keys = set(target_model.state_dict().keys())
        remapped = {}
        for k, v in state_dict.items():
            lora_key_w = k.replace('.weight', '.original_layer.weight')
            lora_key_b = k.replace('.bias', '.original_layer.bias')
            if lora_key_w in model_keys:
                remapped[lora_key_w] = v
            elif lora_key_b in model_keys:
                remapped[lora_key_b] = v
            else:
                remapped[k] = v
        state_dict = remapped
        if verbose:
            print("Detected plain checkpoint loaded into LoRA model: remapped weight keys to original_layer.")

    # Filter out keys with shape mismatches (strict=False doesn't handle those)
    model_state = target_model.state_dict()
    size_mismatch_keys = []
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            size_mismatch_keys.append(k)
            if verbose:
                print(f"  Skipping size mismatch: {k} checkpoint={tuple(v.shape)} model={tuple(model_state[k].shape)}")
        else:
            filtered_state_dict[k] = v

    if size_mismatch_keys and verbose:
        print(f"Skipped {len(size_mismatch_keys)} size-mismatched keys (pretrained arch differs from current model).")

    # Load with strict=False to allow missing keys (new ring embedding params)
    result = target_model.load_state_dict(filtered_state_dict, strict=False)

    if verbose:
        if result.missing_keys:
            print(f"Missing keys (new parameters): {result.missing_keys}")
        if result.unexpected_keys:
            print(f"Unexpected keys (removed parameters): {result.unexpected_keys}")
        loaded_count = len(filtered_state_dict) - len(result.unexpected_keys)
        print(f"Loaded {loaded_count} parameter groups from {checkpoint_path}")

    return result.missing_keys, result.unexpected_keys
