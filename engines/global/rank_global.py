#!/usr/bin/env python3
"""Apply the frozen Global confidence+density ranking to custom inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import rerank_by_density as density


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--ring-features", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    parser.add_argument("--confidence-model", required=True, type=Path)
    parser.add_argument("--confidence-ckpt", default="best_model.pt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "confidence_input.csv"
    split = output / "ids.txt"
    pd.DataFrame([{
        "complex_name": args.name,
        "protein_path": str(args.protein.resolve()),
        "ligand_description": str(args.ligand.resolve()),
        "ring_features_path": str(args.ring_features.resolve()),
    }]).to_csv(manifest, index=False)
    split.write_text(args.name + "\n")

    scored_root = output / "confidence_scored"
    command = [
        sys.executable, "compare_confidence_rerank.py",
        "--protein_ligand_csv", str(manifest), "--test_split", str(split),
        "--data_dir", str(output), "--results_dir", str(scored_root),
        "--pretrained_poses", str(args.poses), "--ablation_poses", str(args.poses),
        "--finetuned_poses", str(args.poses), "--orig_conf_dir", str(args.confidence_model),
        "--orig_conf_ckpt", args.confidence_ckpt, "--em_conf_dir", str(args.confidence_model),
        "--pose_set", "finetuned", "--conf_model", "orig",
        "--max_poses", str(args.top_k), "--batch_size", str(args.batch_size),
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)

    pose_dir = scored_root / "finetuned_orig_conf" / args.name
    confidence_table = pd.read_csv(pose_dir / "confidence_ranking.csv")
    entries = []
    for row in confidence_table.head(args.top_k).itertuples(index=False):
        rank = int(row.new_rank)
        matches = sorted(pose_dir.glob(f"rank{rank}_confidence*.sdf"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one confidence pose for rank {rank}, found {len(matches)}")
        entries.append((int(row.original_rank), float(row.confidence), matches[0].resolve()))
    if len(entries) != args.top_k:
        raise RuntimeError(f"expected {args.top_k} confidence-ranked poses, found {len(entries)}")

    protein_xyz = density.read_protein_heavy_coords(args.protein)
    handle, grid = density.open_map(str(args.map))
    try:
        features = []
        for _, _, path in entries:
            xyz = density.read_pose_heavy_coords(path)
            features.append(density.compute_density_features(
                grid, xyz, protein_coords=protein_xyz, enable_rscc=False
            ))
    finally:
        if hasattr(handle, "close"):
            handle.close()

    confidence = np.asarray([entry[1] for entry in entries], dtype=float)
    mean = np.asarray([item["mean"] for item in features], dtype=float)
    residual = np.asarray([item["residual_shape_corr"] for item in features], dtype=float)
    normalise = lambda values: np.asarray(density.normalise_per_complex({args.name: values})[args.name])
    confidence_z = np.asarray(density.normalise_confidence_per_complex({args.name: confidence})[args.name])
    density_z = normalise(mean)
    residual_z = normalise(residual)
    rows = []
    for index, (original_rank, raw_confidence, path) in enumerate(entries):
        rows.append({
            "complex": args.name,
            "original_rank": original_rank,
            "confidence_rank": index + 1,
            "sdf_path": str(path),
            "confidence_raw": raw_confidence,
            "confidence_z": float(confidence_z[index]),
            "density_mean_z": float(density_z[index]),
            "residual_shape_corr_z": float(residual_z[index]),
            "combined_at_best_weights": float(confidence_z[index] + 0.5 * density_z[index] + residual_z[index]),
        })
    result = output / "global_scores.csv"
    ranked = pd.DataFrame(rows).sort_values(
        ["combined_at_best_weights", "confidence_rank"],
        ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    ranked.insert(1, "frozen_rank", np.arange(1, len(ranked) + 1))
    ranked.to_csv(result, index=False)
    print(f"FROZEN_GLOBAL_RANKING_OK: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
