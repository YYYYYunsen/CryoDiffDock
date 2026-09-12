#!/usr/bin/env python3
"""Compare old/new density masks on exactly the same docking candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import zscore


KEYS = ["row_index", "complex_name", "source", "source_rank"]


def paired_wilcoxon(delta) -> float:
    values = np.asarray(delta, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0.0):
        return float("nan")
    return float(wilcoxon(values).pvalue)


def select(group: pd.DataFrame, ncc_column: str, weight: float):
    group = group.copy()
    group["audit_score"] = (
        zscore(group["confidence"]) + weight * zscore(group[ncc_column])
    )
    return group.sort_values(
        ["audit_score", "confidence", "source", "source_rank"],
        ascending=[False, False, True, True], kind="stable",
    ).iloc[0]


def condition_summary(table, ncc_column, weight, pools):
    summaries, selected_rows, correlation_rows = {}, [], []
    for pool in pools:
        subset = table if pool == "union" else table[table["source"] == pool]
        selected = []
        correlations = []
        for (row_index, complex_name), group in subset.groupby(
            ["row_index", "complex_name"], sort=False
        ):
            chosen = select(group, ncc_column, weight)
            selected.append(float(chosen["ligand_rmsd_A"]))
            selected_rows.append({
                "pool": pool,
                "row_index": row_index,
                "complex_name": complex_name,
                "selected_source": chosen["source"],
                "selected_source_rank": int(chosen["source_rank"]),
                "selected_rmsd_A": float(chosen["ligand_rmsd_A"]),
                "selected_ncc": float(chosen[ncc_column]),
                "prediction_path": chosen["prediction_path"],
            })
            if group[ncc_column].nunique() > 1 and group["ligand_rmsd_A"].nunique() > 1:
                correlation = float(spearmanr(
                    group[ncc_column], -group["ligand_rmsd_A"]
                ).statistic)
                correlations.append(correlation)
                correlation_rows.append({
                    "pool": pool, "row_index": row_index,
                    "complex_name": complex_name, "spearman": correlation,
                })
        values = np.asarray(selected)
        correlations = np.asarray(correlations)
        summaries[pool] = {
            "num_complexes": int(len(values)),
            "median_selected_rmsd_A": float(np.median(values)),
            "selected_lt2A_pct": float(100 * np.mean(values < 2.0)),
            "selected_lt5A_pct": float(100 * np.mean(values < 5.0)),
            "median_spearman_density_vs_negative_rmsd": (
                float(np.median(correlations)) if len(correlations) else float("nan")
            ),
            "positive_spearman_pct": (
                float(100 * np.mean(correlations > 0)) if len(correlations) else float("nan")
            ),
        }
    return summaries, pd.DataFrame(selected_rows), pd.DataFrame(correlation_rows)


def guided_baseline_contrast(table: pd.DataFrame, ncc_column: str):
    """Summarize whether guidance shifts each candidate pool toward density."""
    available = set(table["source"].astype(str))
    if not {"baseline", "guided"}.issubset(available):
        return {}
    rows = []
    for (row_index, complex_name, source), group in table.groupby(
        ["row_index", "complex_name", "source"], sort=False
    ):
        top_confidence = group.sort_values(
            ["confidence", "source_rank"], ascending=[False, True], kind="stable"
        ).iloc[0]
        rows.append({
            "row_index": row_index,
            "complex_name": complex_name,
            "source": source,
            "top_confidence_ncc": float(top_confidence[ncc_column]),
            "median_candidate_ncc": float(group[ncc_column].median()),
            "max_candidate_ncc": float(group[ncc_column].max()),
        })
    wide = pd.DataFrame(rows).pivot(
        index=["row_index", "complex_name"], columns="source"
    )
    result = {}
    for metric in ("top_confidence_ncc", "median_candidate_ncc", "max_candidate_ncc"):
        delta = wide[(metric, "guided")] - wide[(metric, "baseline")]
        result[metric] = {
            "num_paired_complexes": int(len(delta)),
            "median_guided_minus_baseline": float(delta.median()),
            "guided_better_pct": float(100 * (delta > 0).mean()),
            "guided_worse_pct": float(100 * (delta < 0).mean()),
            "wilcoxon_pvalue": paired_wilcoxon(delta),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-scores", required=True, type=Path)
    parser.add_argument("--new-scores", required=True, type=Path)
    parser.add_argument(
        "--evaluated-poses", required=True, type=Path,
        help="per_pose_with_rmsd.csv from either mask evaluation",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--density-column", default="ncc_sigma_1p5")
    parser.add_argument("--density-weight", type=float, default=2.0)
    parser.add_argument("--pools", default="baseline,guided,union")
    args = parser.parse_args()

    old = pd.read_csv(args.old_scores.expanduser().resolve())
    new = pd.read_csv(args.new_scores.expanduser().resolve())
    evaluated = pd.read_csv(args.evaluated_poses.expanduser().resolve())
    required = set(KEYS + [args.density_column])
    for name, table in (("old", old), ("new", new)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} scores missing columns: {missing}")

    rmsd = evaluated[KEYS + ["ligand_rmsd_A"]].drop_duplicates(KEYS)
    old = old.merge(rmsd, on=KEYS, how="inner", validate="one_to_one")
    new = new.merge(rmsd, on=KEYS, how="inner", validate="one_to_one")
    paired = old[KEYS + [args.density_column]].merge(
        new[KEYS + [args.density_column]], on=KEYS, suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if len(paired) != len(old) or len(paired) != len(new):
        raise ValueError(
            f"Candidate sets differ: paired={len(paired)}, old={len(old)}, new={len(new)}"
        )
    delta_column = f"{args.density_column}_delta_new_minus_old"
    paired[delta_column] = (
        paired[f"{args.density_column}_new"] - paired[f"{args.density_column}_old"]
    )

    pools = [item.strip() for item in args.pools.split(",") if item.strip()]
    old_summary, old_selected, old_corr = condition_summary(
        old, args.density_column, args.density_weight, pools
    )
    new_summary, new_selected, new_corr = condition_summary(
        new, args.density_column, args.density_weight, pools
    )
    selected_pair = old_selected.merge(
        new_selected, on=["pool", "row_index", "complex_name"],
        suffixes=("_old", "_new"), validate="one_to_one",
    )
    selected_pair["rmsd_delta_new_minus_old_A"] = (
        selected_pair["selected_rmsd_A_new"] - selected_pair["selected_rmsd_A_old"]
    )
    selected_pair["selection_changed"] = (
        selected_pair["prediction_path_old"] != selected_pair["prediction_path_new"]
    )

    comparison = {}
    for pool, group in selected_pair.groupby("pool"):
        delta = group["rmsd_delta_new_minus_old_A"]
        old_success = group["selected_rmsd_A_old"] < 2.0
        new_success = group["selected_rmsd_A_new"] < 2.0
        comparison[pool] = {
            "selection_changed_pct": float(100 * group["selection_changed"].mean()),
            "median_rmsd_delta_new_minus_old_A": float(delta.median()),
            "new_better_pct": float(100 * (delta < 0).mean()),
            "new_worse_pct": float(100 * (delta > 0).mean()),
            "lt2A_gained_complexes": int((~old_success & new_success).sum()),
            "lt2A_lost_complexes": int((old_success & ~new_success).sum()),
            "wilcoxon_rmsd_pvalue": paired_wilcoxon(delta),
        }

    summary = {
        "num_paired_poses": int(len(paired)),
        "density_column": args.density_column,
        "density_weight": args.density_weight,
        "per_pose_ncc_change": {
            "median_delta_new_minus_old": float(paired[delta_column].median()),
            "new_higher_pct": float(100 * (paired[delta_column] > 0).mean()),
            "wilcoxon_pvalue": paired_wilcoxon(paired[delta_column]),
        },
        "old_mask": old_summary,
        "new_mask": new_summary,
        "old_mask_guided_vs_baseline_ncc": guided_baseline_contrast(
            old, args.density_column
        ),
        "new_mask_guided_vs_baseline_ncc": guided_baseline_contrast(
            new, args.density_column
        ),
        "paired_selection_change": comparison,
        "uses_ground_truth": True,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "per_pose_ncc_change.csv", index=False)
    selected_pair.to_csv(output / "paired_selected_poses.csv", index=False)
    old_corr.assign(mask="old").to_csv(output / "old_per_complex_spearman.csv", index=False)
    new_corr.assign(mask="new").to_csv(output / "new_per_complex_spearman.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
