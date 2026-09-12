#!/usr/bin/env python3
"""Audit whether selected-pose scores retrieve the true ligand across chemotypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_SCORES = (
    "final_selection_score", "multiscale_ncc", "ncc", "confidence",
    "basin_q25_ncc",
)
DEFAULT_DESCRIPTORS = (
    "heavy_atoms", "molecular_weight", "rotatable_bonds", "formal_charge",
    "rings", "aromatic_rings", "logp", "tpsa",
)


def bootstrap_interval(values: np.ndarray, rng, statistic, draws: int) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    sampled = np.asarray([
        statistic(values[rng.integers(0, len(values), size=len(values))])
        for _ in range(draws)
    ])
    return tuple(np.quantile(sampled, [0.025, 0.975]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", required=True, type=Path,
                        help="CryoDiffDock-Local selected.csv; one row per target-candidate task")
    parser.add_argument("--answer-key", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-columns", default=",".join(DEFAULT_SCORES))
    parser.add_argument("--descriptor-columns", default=",".join(DEFAULT_DESCRIPTORS))
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    selected = pd.read_csv(args.selected, keep_default_na=False)
    answers = pd.read_csv(args.answer_key, keep_default_na=False)
    required = {"complex_name"}
    if not required.issubset(selected) or not {"complex_name", "target_id", "is_true"}.issubset(answers):
        raise ValueError("selected/answer-key files lack required identity columns")
    if selected["complex_name"].duplicated().any():
        raise ValueError("selected.csv must contain exactly one row per screening task")
    merged = answers.merge(selected, on="complex_name", how="left", validate="one_to_one", indicator=True)
    merged["is_true"] = merged["is_true"].astype(str).str.lower().isin({"1", "true", "yes"})
    score_columns = [x.strip() for x in args.score_columns.split(",") if x.strip() and x.strip() in merged]
    if not score_columns:
        raise ValueError("None of the requested score columns exists in selected.csv")
    descriptors = [x.strip() for x in args.descriptor_columns.split(",") if x.strip() and x.strip() in merged]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rank_rows = []
    summary_rows = []
    for score_column in score_columns:
        usable = merged.copy()
        usable[score_column] = pd.to_numeric(usable[score_column], errors="coerce")
        for target_id, group in usable.groupby("target_id", sort=True):
            finite = group[group[score_column].notna()].copy()
            true = finite[finite["is_true"]]
            if len(true) != 1:
                rank_rows.append({
                    "score": score_column, "target_id": target_id,
                    "num_candidates": len(finite), "true_rank": np.nan,
                    "top1": False, "top5": False, "reciprocal_rank": 0.0,
                })
                continue
            finite["_rank"] = finite[score_column].rank(method="average", ascending=False)
            rank = float(finite.loc[true.index[0], "_rank"])
            rank_rows.append({
                "score": score_column, "target_id": target_id,
                "num_candidates": len(finite), "true_rank": rank,
                "top1": rank <= 1, "top5": rank <= 5,
                "reciprocal_rank": 1.0 / rank,
                "true_score": float(true.iloc[0][score_column]),
                "best_decoy_score": float(finite.loc[~finite["is_true"], score_column].max()),
            })
        ranks = pd.DataFrame([row for row in rank_rows if row["score"] == score_column])
        complete = ranks[ranks["true_rank"].notna()]
        metrics = {
            "top1_fraction": complete["top1"].to_numpy(float),
            "top5_fraction": complete["top5"].to_numpy(float),
            "mean_reciprocal_rank": complete["reciprocal_rank"].to_numpy(float),
        }
        row = {
            "score": score_column,
            "num_targets": len(complete),
            "num_failed_targets": int(len(ranks) - len(complete)),
            "median_true_rank": float(complete["true_rank"].median()) if len(complete) else np.nan,
        }
        chance_top1 = float(np.mean(1.0 / complete["num_candidates"])) if len(complete) else np.nan
        chance_top5 = float(np.mean(
            np.minimum(5, complete["num_candidates"]) / complete["num_candidates"]
        )) if len(complete) else np.nan
        row["chance_top1_fraction"] = chance_top1
        row["chance_top5_fraction"] = chance_top5
        for name, values in metrics.items():
            estimate = float(np.mean(values)) if len(values) else np.nan
            low, high = bootstrap_interval(values, rng, np.mean, args.bootstrap_draws)
            row[name] = estimate
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        row["top1_enrichment_over_random"] = (
            row["top1_fraction"] / chance_top1 if chance_top1 > 0 else np.nan
        )
        row["top5_enrichment_over_random"] = (
            row["top5_fraction"] / chance_top5 if chance_top5 > 0 else np.nan
        )
        summary_rows.append(row)

    bias_rows = []
    decoys = merged[~merged["is_true"]].copy()
    for score_column in score_columns:
        for descriptor in descriptors:
            correlations = []
            for target_id, group in decoys.groupby("target_id", sort=True):
                x = pd.to_numeric(group[descriptor], errors="coerce")
                y = pd.to_numeric(group[score_column], errors="coerce")
                keep = x.notna() & y.notna()
                if keep.sum() >= 5 and x[keep].nunique() > 1 and y[keep].nunique() > 1:
                    correlations.append(float(spearmanr(x[keep], y[keep]).statistic))
            values = np.asarray(correlations, dtype=float)
            low, high = bootstrap_interval(values, rng, np.nanmedian, args.bootstrap_draws)
            bias_rows.append({
                "score": score_column, "descriptor": descriptor,
                "num_targets": len(values),
                "median_within_target_decoy_spearman": float(np.nanmedian(values)) if len(values) else np.nan,
                "ci_low": low, "ci_high": high,
            })

    ranks = pd.DataFrame(rank_rows)
    summaries = pd.DataFrame(summary_rows)
    biases = pd.DataFrame(bias_rows)
    merged.to_csv(output / "candidate_scores.csv", index=False)
    ranks.to_csv(output / "true_ligand_ranks.csv", index=False)
    summaries.to_csv(output / "retrieval_summary.csv", index=False)
    biases.to_csv(output / "descriptor_bias.csv", index=False)
    run_summary = {
        "num_expected_tasks": len(answers),
        "num_scored_tasks": int((merged["_merge"] == "both").sum()),
        "num_missing_tasks": int((merged["_merge"] != "both").sum()),
        "score_columns": score_columns,
        "bootstrap_draws": args.bootstrap_draws,
        "ranking_direction": "higher is better",
        "unit_of_resampling": "target",
    }
    (output / "audit_manifest.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(summaries.to_string(index=False))
    print(f"Wrote audit to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
