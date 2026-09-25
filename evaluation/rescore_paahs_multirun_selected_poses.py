#!/usr/bin/env python3
"""Recompute multiscale density agreement for final post-PAAHS multirun poses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.adaptive_search import MultiScaleNCCEvaluator
from cryodiffdock_pocket.density_ranking import load_cartesian_density_crop
from evaluation.evaluate_ligand_accuracy import read_heavy_molecule
from scripts.run_adaptive_density_search import molecule_coordinates


def choose_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-table", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--expected-cases", type=int, default=173)
    parser.add_argument("--expected-runs", type=int, default=8)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected_table, dtype={"complex_name": str})
    selected["complex_name"] = selected["complex_name"].astype(str).str.upper()
    counts = selected.groupby("complex_name").size()
    if len(counts) != args.expected_cases or not counts.eq(args.expected_runs).all():
        raise ValueError("Selected table does not contain the expected case/run grid")

    first_manifest = pd.read_csv(
        args.input_root / "replicate01" / "baseline_manifest.csv",
        dtype={"complex_name": str},
    )
    first_manifest["complex_name"] = first_manifest["complex_name"].astype(str).str.upper()
    density_map = dict(zip(first_manifest["complex_name"], first_manifest["density_crop"]))
    device = choose_device(args.device)

    score_rows = []
    for case_index, (name, group) in enumerate(selected.groupby("complex_name", sort=False), start=1):
        group = group.sort_values("replicate_order", kind="stable")
        molecules = [read_heavy_molecule(Path(path)) for path in group["selected_pose_path"]]
        atom_numbers = [atom.GetAtomicNum() for atom in molecules[0].GetAtoms()]
        if any([atom.GetAtomicNum() for atom in molecule.GetAtoms()] != atom_numbers for molecule in molecules[1:]):
            raise ValueError(f"Atom identity/order differs across runs for {name}")
        density_path = Path(density_map[name]).expanduser().resolve()
        evaluator = MultiScaleNCCEvaluator(
            load_cartesian_density_crop(density_path),
            atom_numbers,
            sigmas_A=(1.5, 2.25),
            mask_radius_A=10.0,
            min_mask_voxels=64,
            device=device,
            voxel_chunk_size=512,
            pose_batch_size=args.expected_runs,
        )
        scores = evaluator.score([molecule_coordinates(molecule) for molecule in molecules])
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores, dtype=float)
        for row, values in zip(group.itertuples(index=False), scores):
            score_rows.append({
                "complex_name": name,
                "replicate": row.replicate,
                "ncc_sigma_1p5": float(values[0]),
                "ncc_sigma_2p25": float(values[1]),
                "ncc": float(values[0]),
                "multiscale_ncc": float(values.mean()),
            })
        if case_index % 20 == 0 or case_index == args.expected_cases:
            print(f"[{case_index}/{args.expected_cases}]", flush=True)

    score_table = pd.DataFrame(score_rows)
    output = selected.drop(
        columns=[column for column in ("ncc_sigma_1p5", "ncc_sigma_2p25", "ncc", "multiscale_ncc") if column in selected],
    ).merge(score_table, on=["complex_name", "replicate"], validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"PAAHS_MULTIRUN_NCC_OK: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
