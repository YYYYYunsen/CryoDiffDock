#!/usr/bin/env python3
"""Tune parent-relative GNINA ranking under an S0-relative safety gate.

Coarse variants are generated from an explicit parent pose.  This script uses
the GNINA change from that parent to help choose among density-supported coarse
variants, while retaining the GNINA change from frozen S0 as the final
replacement gate.  GNINA never modifies coordinates and RMSD is used only for
validation-time hyperparameter selection and reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_coarse_basin_ranking import paired_summary
from evaluation.sweep_gnina_relative_coarse_gate import materialize


def quiet_paired_summary(paired: pd.DataFrame) -> dict:
    """Run paired statistics without flooding a grid sweep with SciPy warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Exact p-value calculation does not work if there are zeros.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Sample size too small for normal approximation.*",
            category=UserWarning,
        )
        return paired_summary(paired)


def comma_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return result


def comma_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated values")
    return result


def successful_gnina(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path.expanduser().resolve())
    required = {
        "row_index", "candidate_id", "gnina_status",
        "gnina_affinity_kcal_mol", "gnina_cnnscore", "gnina_cnnaffinity",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing GNINA columns: {missing}")
    frame = frame[frame["gnina_status"].astype(str).eq("ok")].copy()
    if frame.duplicated(["row_index", "candidate_id"]).any():
        raise ValueError(f"{path} contains duplicate successful candidates")
    return frame


def expanded_pose_metrics(search_root: Path) -> pd.DataFrame:
    ranking_paths = sorted(
        (search_root / "expanded_pool").glob("index*___*/ranking.csv")
    )
    if not ranking_paths:
        raise FileNotFoundError(f"No expanded rankings under {search_root}")
    rankings = pd.concat(
        [pd.read_csv(path) for path in ranking_paths], ignore_index=True
    )
    required = {
        "row_index", "complex_name", "candidate_id", "materialized_rank",
        "materialized_path", "protein_path", "parent_id", "move_type",
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")
    metrics = pd.read_csv(search_root / "metrics_expanded_pool" / "per_pose.csv")
    result = rankings.merge(
        metrics[["row_index", "rank", "ligand_rmsd_A"]],
        left_on=["row_index", "materialized_rank"],
        right_on=["row_index", "rank"],
        how="inner", validate="one_to_one",
    )
    if len(result) != len(rankings):
        raise ValueError(
            f"Expanded ranking/RMSD pairing failed: {len(result)} / {len(rankings)}"
        )
    return result


def prepare_scores(
    search_root: Path,
    basin_path: Path,
    candidate_gnina_path: Path,
    parent_gnina_path: Path,
) -> pd.DataFrame:
    poses = expanded_pose_metrics(search_root)
    basin = pd.read_csv(basin_path.expanduser().resolve())
    candidate_gnina = successful_gnina(candidate_gnina_path)
    parent_gnina = successful_gnina(parent_gnina_path)
    gnina_columns = [
        "row_index", "candidate_id", "gnina_affinity_kcal_mol",
        "gnina_cnnscore", "gnina_cnnaffinity",
    ]

    scores = basin.merge(
        candidate_gnina[gnina_columns],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    )
    if len(scores) != len(basin):
        raise ValueError(
            f"Basin/candidate-GNINA pairing failed: {len(scores)} / {len(basin)}"
        )
    scores = scores.merge(
        poses[["row_index", "candidate_id", "ligand_rmsd_A"]],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    )

    # Anchor every candidate to the one frozen S0 pose for the final safety gate.
    s0_controls = scores[
        scores["candidate_role"].astype(str).eq("s0")
    ][[
        "row_index", "gnina_affinity_kcal_mol", "gnina_cnnscore",
        "gnina_cnnaffinity",
    ]].rename(columns={
        "gnina_affinity_kcal_mol": "s0_gnina_affinity_kcal_mol",
        "gnina_cnnscore": "s0_gnina_cnnscore",
        "gnina_cnnaffinity": "s0_gnina_cnnaffinity",
    })
    if len(s0_controls) != scores["row_index"].nunique():
        raise ValueError("Expected exactly one S0 GNINA control per complex")
    scores = scores.merge(
        s0_controls, on="row_index", how="left", validate="many_to_one"
    )
    scores["delta_cnnscore_vs_s0"] = (
        scores["gnina_cnnscore"] - scores["s0_gnina_cnnscore"]
    )
    scores["delta_cnnaffinity_vs_s0"] = (
        scores["gnina_cnnaffinity"] - scores["s0_gnina_cnnaffinity"]
    )
    scores["delta_affinity_improvement_vs_s0"] = (
        scores["s0_gnina_affinity_kcal_mol"]
        - scores["gnina_affinity_kcal_mol"]
    )

    coarse = scores[scores["candidate_role"].astype(str).eq("coarse")].copy()
    parent_columns = {
        "candidate_id": "parent_id",
        "gnina_affinity_kcal_mol": "parent_gnina_affinity_kcal_mol",
        "gnina_cnnscore": "parent_gnina_cnnscore",
        "gnina_cnnaffinity": "parent_gnina_cnnaffinity",
    }
    parents = parent_gnina[gnina_columns].rename(columns=parent_columns)
    coarse = coarse.merge(
        parents,
        on=["row_index", "parent_id"], how="inner", validate="many_to_one",
    )
    if len(coarse) != int((scores["candidate_role"].astype(str) == "coarse").sum()):
        raise ValueError("Some coarse candidates are missing parent GNINA controls")

    parent_rmsd = poses[["row_index", "candidate_id", "ligand_rmsd_A"]].rename(
        columns={"candidate_id": "parent_id", "ligand_rmsd_A": "parent_rmsd_A"}
    )
    coarse = coarse.merge(
        parent_rmsd,
        on=["row_index", "parent_id"], how="inner", validate="many_to_one",
    )
    coarse["delta_parent_cnnscore"] = (
        coarse["gnina_cnnscore"] - coarse["parent_gnina_cnnscore"]
    )
    coarse["delta_parent_cnnaffinity"] = (
        coarse["gnina_cnnaffinity"] - coarse["parent_gnina_cnnaffinity"]
    )
    coarse["delta_parent_affinity_improvement"] = (
        coarse["parent_gnina_affinity_kcal_mol"]
        - coarse["gnina_affinity_kcal_mol"]
    )
    coarse["improves_parent"] = coarse["ligand_rmsd_A"] < coarse["parent_rmsd_A"]

    fallback = scores[scores["candidate_role"].astype(str).eq("s0")].copy()
    fallback["parent_rmsd_A"] = fallback["ligand_rmsd_A"]
    fallback["delta_parent_cnnscore"] = 0.0
    fallback["delta_parent_cnnaffinity"] = 0.0
    fallback["delta_parent_affinity_improvement"] = 0.0
    fallback["improves_parent"] = False
    return pd.concat([fallback, coarse], ignore_index=True, sort=False)


def auc(values: pd.Series, labels: pd.Series) -> float:
    positive = values[labels].dropna().to_numpy(float)
    negative = values[~labels].dropna().to_numpy(float)
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    return float(
        mannwhitneyu(positive, negative, alternative="two-sided").statistic
        / (len(positive) * len(negative))
    )


def signal_summary(coarse: pd.DataFrame) -> dict[str, object]:
    labels = coarse["improves_parent"].astype(bool)
    correlations = []
    for _, group in coarse.groupby("row_index", sort=True):
        if group["delta_parent_cnnscore"].nunique() < 2:
            continue
        rho = group["delta_parent_cnnscore"].corr(
            -group["ligand_rmsd_A"], method="spearman"
        )
        if np.isfinite(rho):
            correlations.append(float(rho))
    array = np.asarray(correlations, dtype=float)
    return {
        "num_complexes": int(coarse["row_index"].nunique()),
        "num_candidates": int(len(coarse)),
        "improving_parent_candidates": int(labels.sum()),
        "parent_delta_cnnscore_auc": auc(
            coarse["delta_parent_cnnscore"], labels
        ),
        "parent_delta_cnnaffinity_auc": auc(
            coarse["delta_parent_cnnaffinity"], labels
        ),
        "parent_delta_affinity_auc": auc(
            coarse["delta_parent_affinity_improvement"], labels
        ),
        "median_per_complex_parent_delta_cnnscore_spearman": (
            float(np.median(array)) if len(array) else float("nan")
        ),
        "positive_per_complex_spearman_pct": (
            float(100.0 * np.mean(array > 0)) if len(array) else float("nan")
        ),
    }


def select_rescues(
    scores: pd.DataFrame,
    *,
    robust_column: str,
    density_weight: float,
    parent_cnnscore_weight: float,
    minimum_score_advantage: float,
    minimum_ncc_gain: float,
    minimum_support_fraction: float,
    minimum_cnnscore_delta_vs_s0: float,
    minimum_parent_cnnscore_delta: float,
    minimum_in_bounds_fraction: float,
    maximum_centroid_shift_A: float,
) -> pd.DataFrame:
    selected_rows = []
    for row_index, group in scores.groupby("row_index", sort=True):
        fallback = group[group["candidate_role"].astype(str).eq("s0")]
        if len(fallback) != 1:
            raise ValueError(f"row {row_index} has {len(fallback)} S0 candidates")
        fallback = fallback.iloc[0].copy()
        fallback_score = float(fallback["s0_selection_score"]) + float(
            density_weight
        ) * (
            float(fallback[robust_column])
            - float(fallback["expanded_density_z"])
        )

        coarse = group[group["candidate_role"].astype(str).eq("coarse")].copy()
        if not coarse.empty:
            coarse["parent_relative_selection_score"] = (
                coarse["confidence_z"].astype(float)
                + float(density_weight) * coarse[robust_column].astype(float)
                + float(parent_cnnscore_weight)
                * coarse["delta_parent_cnnscore"].astype(float)
            )
            eligible = (
                (coarse["ncc_gain"].fillna(-np.inf).astype(float)
                 >= float(minimum_ncc_gain))
                & (coarse["ncc_support_fraction"].fillna(0).astype(float)
                   >= float(minimum_support_fraction))
                & (coarse["neighbor_in_bounds_fraction"].fillna(0).astype(float)
                   >= float(minimum_in_bounds_fraction))
                & (coarse["centroid_shift_A"].fillna(np.inf).astype(float)
                   <= float(maximum_centroid_shift_A))
                & (coarse["delta_cnnscore_vs_s0"].astype(float)
                   >= float(minimum_cnnscore_delta_vs_s0))
                & (coarse["delta_parent_cnnscore"].astype(float)
                   >= float(minimum_parent_cnnscore_delta))
            )
            coarse = coarse[eligible]

        if not coarse.empty:
            best = coarse.sort_values(
                ["parent_relative_selection_score", "ncc_gain", "candidate_id"],
                ascending=[False, False, True], kind="stable",
            ).iloc[0].copy()
            if float(best["parent_relative_selection_score"]) >= (
                fallback_score + float(minimum_score_advantage)
            ):
                best["selected_role"] = "coarse"
                best["final_selection_score"] = float(
                    best["parent_relative_selection_score"]
                )
                selected_rows.append(best)
                continue

        fallback["selected_role"] = "s0"
        fallback["parent_relative_selection_score"] = fallback_score
        fallback["final_selection_score"] = fallback_score
        selected_rows.append(fallback)
    return pd.DataFrame(selected_rows).sort_values("row_index").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--basin-scores", required=True, type=Path)
    parser.add_argument("--candidate-gnina-scores", required=True, type=Path)
    parser.add_argument("--parent-gnina-scores", required=True, type=Path)
    parser.add_argument("--s0-selected", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--selection-holdout-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--robust-columns",
        default="neighbor_ncc_z_q25,neighbor_ncc_z_median",
    )
    parser.add_argument("--parent-cnnscore-weights", default="0.5,1,2,4")
    parser.add_argument("--minimum-score-advantages", default="-1,-0.5,0")
    parser.add_argument("--minimum-ncc-gains", default="0.005,0.02,0.05")
    parser.add_argument("--minimum-support-fractions", default="0.4,0.5,0.6")
    parser.add_argument("--minimum-cnnscore-deltas-vs-s0", default="0,0.1")
    parser.add_argument("--minimum-parent-cnnscore-deltas", default="-0.1,0,0.1")
    parser.add_argument("--minimum-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-centroid-shift", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores = prepare_scores(
        search_root,
        args.basin_scores,
        args.candidate_gnina_scores,
        args.parent_gnina_scores,
    )
    scores.to_csv(output / "parent_relative_candidates.csv", index=False)
    s0 = pd.read_csv(args.s0_selected.expanduser().resolve())
    config = json.loads(args.s0_config.expanduser().resolve().read_text())
    density_weight = float(config["density_weight"])
    s0_rmsd = s0[["row_index", "complex_name", "ligand_rmsd_A"]].rename(
        columns={"ligand_rmsd_A": "s0_rmsd_A"}
    )

    available = set(s0_rmsd["complex_name"].astype(str))
    holdout_names = {
        line.strip()
        for line in args.selection_holdout_file.expanduser().resolve().read_text().splitlines()
        if line.strip()
    }
    absent = sorted(holdout_names - available)
    if absent:
        raise ValueError(f"Holdout complexes are absent: {absent}")
    tuning_names = available - holdout_names
    coarse = scores[scores["candidate_role"].astype(str).eq("coarse")].copy()
    tuning_coarse = coarse[coarse["complex_name"].astype(str).isin(tuning_names)]
    holdout_coarse = coarse[coarse["complex_name"].astype(str).isin(holdout_names)]
    signal = {
        "tuning": signal_summary(tuning_coarse),
        "holdout": signal_summary(holdout_coarse),
        "all_validation": signal_summary(coarse),
        "uses_ground_truth": True,
    }
    (output / "signal_summary.json").write_text(json.dumps(signal, indent=2) + "\n")

    robust_columns = comma_strings(args.robust_columns)
    for column in robust_columns:
        if column not in scores.columns:
            raise ValueError(f"Unknown robust column: {column}")
    rows = []
    selected_by_config: dict[int, pd.DataFrame] = {}
    configuration_id = 0
    for robust_column in robust_columns:
        for parent_weight in comma_floats(args.parent_cnnscore_weights):
            for score_advantage in comma_floats(args.minimum_score_advantages):
                for ncc_gain in comma_floats(args.minimum_ncc_gains):
                    for support in comma_floats(args.minimum_support_fractions):
                        for s0_delta in comma_floats(args.minimum_cnnscore_deltas_vs_s0):
                            for parent_delta in comma_floats(
                                args.minimum_parent_cnnscore_deltas
                            ):
                                configuration_id += 1
                                selected = select_rescues(
                                    scores,
                                    robust_column=robust_column,
                                    density_weight=density_weight,
                                    parent_cnnscore_weight=parent_weight,
                                    minimum_score_advantage=score_advantage,
                                    minimum_ncc_gain=ncc_gain,
                                    minimum_support_fraction=support,
                                    minimum_cnnscore_delta_vs_s0=s0_delta,
                                    minimum_parent_cnnscore_delta=parent_delta,
                                    minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
                                    maximum_centroid_shift_A=args.maximum_centroid_shift,
                                )
                                paired = s0_rmsd.merge(
                                    selected[[
                                        "row_index", "candidate_id", "candidate_role",
                                        "ligand_rmsd_A", "move_type", "materialized_path",
                                        "protein_path", "final_selection_score",
                                    ]].rename(columns={
                                        "ligand_rmsd_A": "selected_rmsd_A",
                                        "candidate_role": "selected_role",
                                    }),
                                    on="row_index", how="inner", validate="one_to_one",
                                )
                                tuning = paired[
                                    paired["complex_name"].astype(str).isin(tuning_names)
                                ]
                                summary = quiet_paired_summary(tuning)
                                rows.append({
                                    "configuration_id": configuration_id,
                                    "robust_column": robust_column,
                                    "parent_cnnscore_weight": parent_weight,
                                    "minimum_score_advantage": score_advantage,
                                    "minimum_ncc_gain": ncc_gain,
                                    "minimum_support_fraction": support,
                                    "minimum_cnnscore_delta_vs_s0": s0_delta,
                                    "minimum_parent_cnnscore_delta": parent_delta,
                                    **summary,
                                })
                                selected_by_config[configuration_id] = paired

    sweep = pd.DataFrame(rows).sort_values(
        [
            "catastrophic_worsening_gt2A", "net_gained_lt2A", "lost_lt2A",
            "top1_lt2A_pct", "net_gained_lt5A", "mean_top1_rmsd_A",
            "selected_coarse_count", "configuration_id",
        ],
        ascending=[True, False, True, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    sweep.to_csv(output / "sweep.csv", index=False)
    sweep.head(args.top_k).to_csv(output / "top_configurations.csv", index=False)
    best = sweep.iloc[0].to_dict()
    paired = selected_by_config[int(best["configuration_id"])].copy()
    tuning = paired[paired["complex_name"].astype(str).isin(tuning_names)].copy()
    holdout = paired[paired["complex_name"].astype(str).isin(holdout_names)].copy()
    paired.to_csv(output / "best_paired_complexes.csv", index=False)
    tuning.to_csv(output / "best_paired_tuning.csv", index=False)
    holdout.to_csv(output / "best_paired_holdout.csv", index=False)

    selected_ids = set(zip(paired["row_index"].astype(int), paired["candidate_id"].astype(str)))
    selected_rows = scores[
        [(int(row.row_index), str(row.candidate_id)) in selected_ids for row in scores.itertuples()]
    ].copy()
    selected_rows = selected_rows.merge(
        paired[["row_index", "candidate_id", "selected_role", "final_selection_score"]],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    ).sort_values("row_index")
    selected_rows.to_csv(output / "best_selected_candidates.csv", index=False)
    materialize(output, selected_rows)

    best_config = {
        key: best[key]
        for key in [
            "robust_column", "parent_cnnscore_weight", "minimum_score_advantage",
            "minimum_ncc_gain", "minimum_support_fraction",
            "minimum_cnnscore_delta_vs_s0", "minimum_parent_cnnscore_delta",
        ]
    }
    best_config.update({
        "density_weight": density_weight,
        "minimum_in_bounds_fraction": args.minimum_in_bounds_fraction,
        "maximum_centroid_shift_A": args.maximum_centroid_shift,
        "signal_summary": signal,
        "tuning_metrics": quiet_paired_summary(tuning),
        "holdout_metrics": quiet_paired_summary(holdout),
        "all_validation_metrics": quiet_paired_summary(paired),
        "uses_ground_truth_for_hyperparameter_selection": True,
        "selection_baseline": "frozen_s0",
        "gnina_usage": "parent_relative_ranking_plus_s0_relative_gate",
        "warning": "Freeze this configuration before independent test evaluation.",
    })
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n")
    print("PARENT-RELATIVE GNINA SIGNAL")
    print(json.dumps(signal, indent=2))
    print("\nTop tuning configurations:")
    print(sweep.head(min(10, len(sweep))).to_string(index=False))
    print("\n" + json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
