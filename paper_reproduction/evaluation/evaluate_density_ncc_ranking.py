#!/usr/bin/env python3
"""Validation-only evaluation and hyperparameter sweep for Gaussian NCC ranking.

Unlike ``scripts/rerank_ligand_by_density_ncc.py``, this script deliberately
reads deposited reference ligands.  Never use it as a deployment ranking stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import zscore
from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd


NCC_COLUMN = re.compile(r"^ncc_sigma_(.+)$")


def resolve_path(value, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def available_ncc_columns(scores: pd.DataFrame):
    return [column for column in scores.columns if NCC_COLUMN.match(column)]


def choose_pose(group: pd.DataFrame, density_column: str, weight: float):
    group = group.copy()
    confidence = group["confidence"].to_numpy(dtype=float)
    if not np.isfinite(confidence).all():
        confidence = np.where(np.isfinite(confidence), confidence, -group["source_rank"])
    group["eval_confidence_z"] = zscore(confidence)
    group["eval_density_z"] = zscore(group[density_column])
    group["eval_combined_score"] = (
        group["eval_confidence_z"] + float(weight) * group["eval_density_z"]
    )
    return group.sort_values(
        ["eval_combined_score", "confidence", "source", "source_rank"],
        ascending=[False, False, True, True], kind="stable",
    ).iloc[0]


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {
        "num_complexes": int(len(array)),
        "median_top1_rmsd_A": float(np.median(array)),
        "mean_top1_rmsd_A": float(np.mean(array)),
        "top1_lt2A_pct": float(100.0 * np.mean(array < 2.0)),
        "top1_lt5A_pct": float(100.0 * np.mean(array < 5.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weight-grid", default="0,0.1,0.25,0.5,1,2")
    parser.add_argument("--pools", default="baseline,guided,union",
                        help="Source names and/or union. Names must match pose_scores.csv")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    scores = pd.read_csv(args.scores.expanduser().resolve())
    ncc_columns = available_ncc_columns(scores)
    if not ncc_columns:
        raise ValueError("Scores CSV has no ncc_sigma_* columns")
    weights = [float(value) for value in args.weight_grid.split(",") if value.strip()]
    pools = [value.strip() for value in args.pools.split(",") if value.strip()]
    known_sources = set(scores["source"].astype(str))
    unknown = sorted(set(pools) - known_sources - {"union"})
    if unknown:
        raise ValueError(f"Unknown pools {unknown}; sources are {sorted(known_sources)}")

    reference_by_index = {}
    for row_index, row in table.iterrows():
        reference_by_index[int(row_index)] = read_heavy_molecule(
            resolve_path(row["reference_ligand"], manifest.parent)
        )

    rmsds, failures = [], []
    for pose_index, pose in scores.iterrows():
        try:
            prediction = read_heavy_molecule(Path(pose["prediction_path"]))
            rmsds.append(symmetry_rmsd(reference_by_index[int(pose["row_index"])], prediction))
        except Exception as exc:
            rmsds.append(float("nan"))
            failures.append({
                "pose_index": pose_index,
                "prediction_path": pose["prediction_path"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            if args.fail_on_error:
                raise
    scores = scores.copy()
    scores["ligand_rmsd_A"] = rmsds
    valid = scores[np.isfinite(scores["ligand_rmsd_A"])].copy()
    if valid.empty:
        raise RuntimeError("No candidate RMSDs could be evaluated")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    valid.to_csv(output / "per_pose_with_rmsd.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)

    correlation_rows = []
    for pool in pools:
        pool_scores = valid if pool == "union" else valid[valid["source"] == pool]
        for density_column in ncc_columns:
            correlations = []
            for (row_index, complex_name), group in pool_scores.groupby(
                ["row_index", "complex_name"], sort=False
            ):
                if group[density_column].nunique() < 2 or group["ligand_rmsd_A"].nunique() < 2:
                    correlation = float("nan")
                else:
                    correlation = float(spearmanr(
                        group[density_column], -group["ligand_rmsd_A"]
                    ).statistic)
                correlations.append(correlation)
                correlation_rows.append({
                    "pool": pool, "density_column": density_column,
                    "row_index": row_index, "complex_name": complex_name,
                    "spearman_density_vs_negative_rmsd": correlation,
                })
    correlation_table = pd.DataFrame(correlation_rows)
    correlation_table.to_csv(output / "per_complex_spearman.csv", index=False)

    sweep_rows, selected_rows = [], []
    for pool in pools:
        pool_scores = valid if pool == "union" else valid[valid["source"] == pool]
        oracle = pool_scores.groupby(["row_index", "complex_name"])["ligand_rmsd_A"].min()
        for density_column in ncc_columns:
            correlation_subset = correlation_table[
                (correlation_table["pool"] == pool)
                & (correlation_table["density_column"] == density_column)
            ]["spearman_density_vs_negative_rmsd"].dropna()
            for weight in weights:
                chosen = []
                for (row_index, complex_name), group in pool_scores.groupby(
                    ["row_index", "complex_name"], sort=False
                ):
                    pose = choose_pose(group, density_column, weight)
                    chosen.append({
                        "pool": pool, "density_column": density_column,
                        "density_weight": weight, "row_index": row_index,
                        "complex_name": complex_name,
                        "selected_rmsd_A": float(pose["ligand_rmsd_A"]),
                        "selected_source": pose["source"],
                        "selected_source_rank": int(pose["source_rank"]),
                        "prediction_path": pose["prediction_path"],
                    })
                selected_rows.extend(chosen)
                metrics = summarize([row["selected_rmsd_A"] for row in chosen])
                sweep_rows.append({
                    "pool": pool, "density_column": density_column,
                    "density_weight": weight, **metrics,
                    "oracle_median_rmsd_A": float(oracle.median()),
                    "oracle_lt2A_pct": float(100.0 * (oracle < 2.0).mean()),
                    "median_per_complex_spearman": (
                        float(correlation_subset.median()) if len(correlation_subset) else float("nan")
                    ),
                    "positive_spearman_pct": (
                        float(100.0 * (correlation_subset > 0).mean())
                        if len(correlation_subset) else float("nan")
                    ),
                })

    sweep = pd.DataFrame(sweep_rows).sort_values(
        ["top1_lt2A_pct", "median_top1_rmsd_A", "top1_lt5A_pct"],
        ascending=[False, True, False], kind="stable",
    )
    sweep.to_csv(output / "sweep.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output / "selected_poses.csv", index=False)
    # Round-trip through pandas JSON to convert NumPy scalar values into plain
    # Python JSON values for the human/machine-readable summary.
    best = json.loads(sweep.iloc[0].to_json())
    summary = {
        "num_poses": int(len(valid)),
        "num_failures": len(failures),
        "pools": pools,
        "ncc_columns": ncc_columns,
        "weight_grid": weights,
        "best_validation_configuration": best,
        "warning": "Hyperparameters were selected on validation references; freeze them before test evaluation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nTop validation configurations:")
    print(sweep.head(15).to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
