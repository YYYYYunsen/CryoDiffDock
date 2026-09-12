#!/usr/bin/env python3
"""Recompute official DiffDock-Pocket confidence for discrete-search poses.

The discrete generator initially inherits a parent's confidence for every
local variant.  This script builds one reference graph per complex, replaces
the ligand coordinates for each shortlisted pose, restores that pose's paired
flexible-protein coordinates, and evaluates the official confidence model at
t=0.  It never reads a reference ligand or RMSD.
"""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace
from functools import partial
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml
from rdkit import Chem
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.confidence_rescoring import (
    calibrate_recomputed_confidence,
    select_rescoring_shortlist,
)
from datasets.pdbbind import PDBBind, load_protein_ligand_df
from datasets.process_mols import parse_pdb_from_path
from utils.diffusion_utils import set_time, t_to_sigma as t_to_sigma_compl
from utils.utils import get_default_device, get_model


def load_expanded_rankings(search_root: Path) -> pd.DataFrame:
    files = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not files:
        raise FileNotFoundError(f"No expanded ranking files under {search_root}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def read_pose_coordinates(path: Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read candidate SDF: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)


def protein_atoms(path: Path, remove_hs: bool) -> tuple[list[tuple], np.ndarray]:
    structure_or_model = parse_pdb_from_path(str(path))
    level = structure_or_model.get_level()
    if level == "S":
        models = list(structure_or_model)
    elif level == "M":
        models = [structure_or_model]
    else:
        raise ValueError(
            f"Expected a Biopython Structure or Model for {path}, got level {level}"
        )
    keys, coordinates = [], []
    for model in models:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if remove_hs and atom.element == "H":
                        continue
                    key = (
                        str(model.id), str(chain.id), str(residue.id[0]),
                        int(residue.id[1]), str(residue.id[2]), str(atom.name),
                    )
                    keys.append(key)
                    coordinates.append(np.asarray(atom.coord, dtype=np.float32))
    return keys, np.asarray(coordinates, dtype=np.float32)


def graph_atom_keys(graph, reference_protein: Path, remove_hs: bool) -> list[tuple]:
    keys, coordinates = protein_atoms(reference_protein, remove_hs)
    absolute = (
        graph["atom"].pos.detach().cpu().numpy()
        + graph.original_center.detach().cpu().numpy()
    )
    distances, indices = cKDTree(coordinates).query(absolute, k=1)
    if len(distances) and float(np.max(distances)) > 1e-3:
        raise ValueError(
            f"Could not map graph atoms to {reference_protein}; "
            f"maximum coordinate mismatch is {float(np.max(distances)):.6f} A"
        )
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Graph-to-PDB atom mapping is not one-to-one")
    return [keys[int(index)] for index in indices]


def paired_atom_coordinates(
    path: Path,
    graph_keys: list[tuple],
    remove_hs: bool,
) -> np.ndarray:
    keys, coordinates = protein_atoms(path, remove_hs)
    lookup = {key: xyz for key, xyz in zip(keys, coordinates)}
    missing = [key for key in graph_keys if key not in lookup]
    if missing:
        raise ValueError(
            f"Paired protein {path} is missing {len(missing)} graph atoms; "
            f"first missing key: {missing[0]}"
        )
    return np.asarray([lookup[key] for key in graph_keys], dtype=np.float32)


def build_dataset(manifest_path: Path, model_args: Namespace, cache_root: Path, seed: int):
    frame = load_protein_ligand_df(str(manifest_path), strict=False)
    if "computational_protein" in frame.columns:
        frame = frame.drop(columns="computational_protein")
    return PDBBind(
        transform=None, protein_ligand_df=frame, chain_cutoff=np.inf,
        receptor_radius=model_args.receptor_radius, cache_path=str(cache_root),
        remove_hs=model_args.remove_hs, max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors, matching=False,
        keep_original=False, conformer_match_sidechains=False,
        use_original_conformer_fallback=True, popsize=model_args.matching_popsize,
        maxiter=model_args.matching_maxiter, all_atoms=model_args.all_atoms,
        # Match the oracle-density ablation exactly.  That inference path
        # intentionally regenerates a seeded RDKit conformer; using the SDF's
        # deposited coordinates here changes pocket reduction and therefore
        # the confidence graph even after candidate coordinates are replaced.
        require_ligand=True, num_workers=1, keep_local_structures=False,
        pocket_reduction=model_args.pocket_reduction,
        pocket_buffer=model_args.pocket_buffer,
        pocket_cutoff=model_args.pocket_cutoff,
        pocket_reduction_mode=model_args.pocket_reduction_mode,
        flexible_sidechains=model_args.flexible_sidechains,
        flexdist=model_args.flexdist,
        flexdist_distance_metric=model_args.flexdist_distance_metric,
        fixed_knn_radius_graph=not model_args.not_fixed_knn_radius_graph,
        knn_only_graph=not model_args.not_knn_only_graph,
        include_miscellaneous_atoms=getattr(model_args, "include_miscellaneous_atoms", False),
        use_old_wrong_embedding_order=model_args.use_old_wrong_embedding_order,
        conformer_seed=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(".cache/model/v1.0.0/score_model"))
    parser.add_argument("--confidence-model-dir", type=Path, default=Path(".cache/model/v1.0.0/confidence_model"))
    parser.add_argument("--confidence-ckpt", default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--minimum-ncc-gain", type=float, default=0.005)
    parser.add_argument("--maximum-anchor-rmsd", type=float, default=2.0)
    parser.add_argument("--maximum-centroid-shift", type=float, default=1.5)
    parser.add_argument("--minimum-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--torsion-top-k-per-parent", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir.expanduser().resolve()
    confidence_dir = args.confidence_model_dir.expanduser().resolve()

    with open(model_dir / "model_parameters.yml") as handle:
        score_args = Namespace(**yaml.full_load(handle))
    with open(confidence_dir / "model_parameters.yml") as handle:
        confidence_args = Namespace(**yaml.full_load(handle))

    rankings = load_expanded_rankings(search_root)
    shortlist = select_rescoring_shortlist(
        rankings,
        minimum_ncc_gain=args.minimum_ncc_gain,
        maximum_anchor_rmsd_A=args.maximum_anchor_rmsd,
        maximum_centroid_shift_A=args.maximum_centroid_shift,
        minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
        torsion_top_k_per_parent=args.torsion_top_k_per_parent,
    )
    if args.limit is not None:
        allowed = sorted(shortlist["row_index"].astype(int).unique())[: args.limit]
        shortlist = shortlist[shortlist["row_index"].astype(int).isin(allowed)].copy()

    manifest = pd.read_csv(manifest_path)
    device = get_default_device()
    t_to_sigma = partial(t_to_sigma_compl, args=score_args)
    model = get_model(
        confidence_args, device, t_to_sigma=t_to_sigma,
        no_parallel=True, confidence_mode=True,
    )
    state = torch.load(confidence_dir / args.confidence_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    failures, scored_rows = [], []
    with tempfile.TemporaryDirectory(prefix="s1_graphs_", dir=str(output)) as temporary:
        dataset = build_dataset(
            manifest_path, score_args, Path(temporary) / "testset", args.seed
        )
        graph_rows = dataset.protein_ligand_df
        for position, (row_index, candidates) in enumerate(
            shortlist.groupby("row_index", sort=True), start=1
        ):
            try:
                row_index = int(row_index)
                manifest_row = manifest.iloc[row_index]
                complex_name = str(manifest_row["complex_name"])
                graph_match = graph_rows[
                    graph_rows["complex_name"].astype(str).eq(complex_name)
                ]
                if len(graph_match) != 1:
                    raise ValueError(
                        f"Expected one graph for {complex_name}, found {len(graph_match)}"
                    )
                graph = copy.deepcopy(graph_match.iloc[0]["complex_graph"])
                reference_protein = Path(str(manifest_row["experimental_protein"])).expanduser().resolve()
                atom_keys = graph_atom_keys(graph, reference_protein, score_args.remove_hs)
                centre = graph.original_center.detach().cpu().numpy()
                protein_cache: dict[str, np.ndarray] = {}

                prepared_graphs, prepared_rows = [], []
                for _, candidate in candidates.iterrows():
                    pose_path = Path(str(candidate["materialized_path"])).expanduser().resolve()
                    protein_path = Path(str(candidate["protein_path"])).expanduser().resolve()
                    coordinates = read_pose_coordinates(pose_path)
                    if coordinates.shape != tuple(graph["ligand"].pos.shape):
                        raise ValueError(
                            f"Ligand atom mismatch for {pose_path}: {coordinates.shape} "
                            f"!= {tuple(graph['ligand'].pos.shape)}"
                        )
                    protein_key = str(protein_path)
                    if protein_key not in protein_cache:
                        protein_cache[protein_key] = paired_atom_coordinates(
                            protein_path, atom_keys, score_args.remove_hs
                        )
                    candidate_graph = copy.deepcopy(graph)
                    candidate_graph["ligand"].pos = torch.from_numpy(
                        coordinates - centre
                    ).float()
                    candidate_graph["atom"].pos = torch.from_numpy(
                        protein_cache[protein_key] - centre
                    ).float()
                    prepared_graphs.append(candidate_graph)
                    prepared_rows.append(candidate.to_dict())

                    if len(prepared_graphs) >= args.batch_size:
                        batch = next(iter(DataLoader(prepared_graphs, batch_size=len(prepared_graphs)))).to(device)
                        set_time(batch, 0, 0, 0, 0, 0, len(prepared_graphs), confidence_args.all_atoms,
                                 confidence_args.asyncronous_noise_schedule, device,
                                 include_miscellaneous_atoms=getattr(confidence_args, "include_miscellaneous_atoms", False))
                        with torch.no_grad():
                            values = model(batch).detach().cpu().reshape(-1).numpy()
                        for record, value in zip(prepared_rows, values):
                            record["confidence_recomputed"] = float(value)
                            scored_rows.append(record)
                        prepared_graphs, prepared_rows = [], []

                if prepared_graphs:
                    batch = next(iter(DataLoader(prepared_graphs, batch_size=len(prepared_graphs)))).to(device)
                    set_time(batch, 0, 0, 0, 0, 0, len(prepared_graphs), confidence_args.all_atoms,
                             confidence_args.asyncronous_noise_schedule, device,
                             include_miscellaneous_atoms=getattr(confidence_args, "include_miscellaneous_atoms", False))
                    with torch.no_grad():
                        values = model(batch).detach().cpu().reshape(-1).numpy()
                    for record, value in zip(prepared_rows, values):
                        record["confidence_recomputed"] = float(value)
                        scored_rows.append(record)
                print(f"[{position}/{shortlist['row_index'].nunique()}] row {row_index}: {len(candidates)} poses scored")
            except Exception as exc:
                failures.append({"row_index": int(row_index), "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED row {row_index}: {type(exc).__name__}: {exc}")
                if args.fail_on_error:
                    raise

    if not scored_rows:
        raise RuntimeError("No candidates were rescored")
    rescored = calibrate_recomputed_confidence(pd.DataFrame(scored_rows))
    rescored.to_csv(output / "rescored_candidates.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    originals = rescored[rescored["is_original"].astype(str).str.lower().isin({"true", "1"})].copy()
    originals["confidence_delta"] = originals["confidence_recomputed"] - originals["confidence"].astype(float)
    originals["calibrated_z_delta"] = (
        originals["confidence_recomputed_z"] - originals["confidence_z"]
    )
    old_confidence = originals["confidence"].to_numpy(dtype=float)
    new_confidence = originals["confidence_recomputed"].to_numpy(dtype=float)
    pearson = float(np.corrcoef(old_confidence, new_confidence)[0, 1])
    spearman = float(spearmanr(old_confidence, new_confidence).statistic)
    per_complex_spearman = []
    top1_agreement = []
    for _, group in originals.groupby("row_index", sort=False):
        if group["confidence"].nunique() > 1 and group["confidence_recomputed"].nunique() > 1:
            per_complex_spearman.append(float(spearmanr(
                group["confidence"].astype(float),
                group["confidence_recomputed"].astype(float),
            ).statistic))
        old_top = str(group.loc[group["confidence"].astype(float).idxmax(), "candidate_id"])
        new_top = str(group.loc[group["confidence_recomputed"].astype(float).idxmax(), "candidate_id"])
        top1_agreement.append(old_top == new_top)
    summary = {
        "num_complexes": int(rescored["row_index"].nunique()),
        "num_candidates": int(len(rescored)),
        "num_originals": int(len(originals)),
        "num_failures": int(len(failures)),
        "median_original_confidence_abs_delta": float(originals["confidence_delta"].abs().median()),
        "p95_original_confidence_abs_delta": float(originals["confidence_delta"].abs().quantile(0.95)),
        "max_original_confidence_abs_delta": float(originals["confidence_delta"].abs().max()),
        "original_confidence_pearson": pearson,
        "original_confidence_spearman": spearman,
        "median_per_complex_original_confidence_spearman": float(np.median(per_complex_spearman)),
        "original_confidence_top1_agreement_pct": float(100 * np.mean(top1_agreement)),
        "original_confidence_abs_delta_gt_0p02_pct": float(
            100 * (originals["confidence_delta"].abs() > 0.02).mean()
        ),
        "max_calibrated_original_confidence_z_abs_delta": float(
            originals["calibrated_z_delta"].abs().max()
        ),
        "torsion_top_k_per_parent": args.torsion_top_k_per_parent,
        "uses_ground_truth": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
