#!/usr/bin/env python3
"""Tune an S0-anchored GNINA safety gate for coarse-orientation rescues.

GNINA is not used to rank the coarse pool.  Density/confidence chooses one
coarse proposal, while GNINA only decides whether that proposal is sufficiently
compatible with the *same fixed receptor* relative to the frozen S0 pose.
RMSD is read only to select hyperparameters on the tuning cohort and to report
the untouched holdout cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_coarse_basin_ranking import paired_summary


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


def add_relative_gnina(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_index, group in scores.groupby("row_index", sort=True):
        fallback = group[group["candidate_role"].astype(str).eq("s0")]
        if len(fallback) != 1:
            raise ValueError(f"row {row_index} has {len(fallback)} S0 candidates")
        fallback = fallback.iloc[0]
        group = group.copy()
        group["delta_cnnscore_vs_s0"] = (
            group["gnina_cnnscore"].astype(float)
            - float(fallback["gnina_cnnscore"])
        )
        group["delta_cnnaffinity_vs_s0"] = (
            group["gnina_cnnaffinity"].astype(float)
            - float(fallback["gnina_cnnaffinity"])
        )
        # Lower Vina-style affinity is better.  Positive means an improvement.
        group["delta_affinity_improvement_vs_s0"] = (
            float(fallback["gnina_affinity_kcal_mol"])
            - group["gnina_affinity_kcal_mol"].astype(float)
        )
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def select_rescues(
    scores: pd.DataFrame,
    *,
    robust_column: str,
    density_weight: float,
    minimum_score_advantage: float,
    minimum_ncc_gain: float,
    minimum_support_fraction: float,
    minimum_in_bounds_fraction: float,
    maximum_centroid_shift_A: float,
    gate_mode: str,
    minimum_cnnscore_delta: float,
    minimum_cnnaffinity_delta: float,
    minimum_affinity_improvement: float,
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
            coarse["relative_gate_selection_score"] = (
                coarse["confidence_z"].astype(float)
                + float(density_weight) * coarse[robust_column].astype(float)
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
                   >= float(minimum_cnnscore_delta))
            )
            if gate_mode in {"cnnscore_cnnaffinity", "all"}:
                eligible &= (
                    coarse["delta_cnnaffinity_vs_s0"].astype(float)
                    >= float(minimum_cnnaffinity_delta)
                )
            if gate_mode == "all":
                eligible &= (
                    coarse["delta_affinity_improvement_vs_s0"].astype(float)
                    >= float(minimum_affinity_improvement)
                )
            coarse = coarse[eligible]

        if not coarse.empty:
            # GNINA is deliberately absent from this ordering.  It is a gate,
            # not a second pose ranker.
            best = coarse.sort_values(
                ["relative_gate_selection_score", "ncc_gain", "candidate_id"],
                ascending=[False, False, True], kind="stable",
            ).iloc[0].copy()
            if float(best["relative_gate_selection_score"]) >= (
                fallback_score + float(minimum_score_advantage)
            ):
                best["selected_role"] = "coarse"
                best["final_selection_score"] = float(
                    best["relative_gate_selection_score"]
                )
                selected_rows.append(best)
                continue

        fallback["selected_role"] = "s0"
        fallback["relative_gate_selection_score"] = fallback_score
        fallback["final_selection_score"] = fallback_score
        selected_rows.append(fallback)
    return pd.DataFrame(selected_rows).sort_values("row_index").reset_index(drop=True)


def materialize(output: Path, selected: pd.DataFrame) -> None:
    root = output / "best_selected"
    root.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in str(row["complex_name"])
        )
        directory = root / f"index{int(row['row_index'])}___{name}"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(str(row["materialized_path"]))
        score = float(row["final_selection_score"])
        destination = directory / f"rank1_confidence{score:.8f}.sdf"
        shutil.copy2(source, destination)
        protein = Path(str(row["protein_path"]))
        if protein.is_file():
            shutil.copy2(
                protein,
                destination.with_name(f"{destination.stem}_protein.pdb"),
            )
        pd.DataFrame([row.to_dict()]).to_csv(directory / "ranking.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--basin-scores", required=True, type=Path)
    parser.add_argument("--gnina-scores", required=True, type=Path)
    parser.add_argument("--s0-selected", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--selection-holdout-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--robust-columns",
        default="neighbor_ncc_z_q25,neighbor_ncc_z_median",
    )
    parser.add_argument(
        "--gate-modes", default="cnnscore,cnnscore_cnnaffinity,all"
    )
    parser.add_argument("--minimum-score-advantages", default="-0.5,0,0.5")
    parser.add_argument("--minimum-ncc-gains", default="0.005,0.02,0.05")
    parser.add_argument("--minimum-support-fractions", default="0.4,0.5,0.6")
    parser.add_argument("--minimum-cnnscore-deltas", default="-0.1,0,0.1")
    parser.add_argument("--minimum-cnnaffinity-deltas", default="-0.5,0,0.25")
    parser.add_argument("--minimum-affinity-improvements", default="-20,-10,0")
    parser.add_argument("--minimum-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-centroid-shift", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.basin_scores.expanduser().resolve())
    gnina = pd.read_csv(args.gnina_scores.expanduser().resolve())
    gnina = gnina[gnina["gnina_status"].astype(str).eq("ok")].copy()
    gnina_columns = [
        "row_index", "candidate_id", "gnina_affinity_kcal_mol",
        "gnina_cnnscore", "gnina_cnnaffinity",
    ]
    missing_gnina = sorted(set(gnina_columns) - set(gnina.columns))
    if missing_gnina:
        raise ValueError(f"GNINA scores are missing columns: {missing_gnina}")
    scores = scores.merge(
        gnina[gnina_columns],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    )
    if len(scores) != len(gnina):
        raise ValueError(
            f"Basin/GNINA pairing failed: {len(scores)} / {len(gnina)}"
        )
    scores = add_relative_gnina(scores)

    s0 = pd.read_csv(args.s0_selected.expanduser().resolve())
    config = json.loads(args.s0_config.expanduser().resolve().read_text())
    density_weight = float(config["density_weight"])
    metrics = pd.read_csv(search_root / "metrics_expanded_pool" / "per_pose.csv")
    scores = scores.merge(
        metrics[["row_index", "rank", "ligand_rmsd_A"]],
        left_on=["row_index", "materialized_rank"],
        right_on=["row_index", "rank"],
        how="inner", validate="one_to_one",
    )
    if scores.empty:
        raise ValueError("No candidates matched expanded-pool RMSD metrics")

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

    robust_columns = comma_strings(args.robust_columns)
    for column in robust_columns:
        if column not in scores.columns:
            raise ValueError(f"Unknown robust column: {column}")
    gate_modes = comma_strings(args.gate_modes)
    unknown_modes = sorted(set(gate_modes) - {"cnnscore", "cnnscore_cnnaffinity", "all"})
    if unknown_modes:
        raise ValueError(f"Unknown gate modes: {unknown_modes}")

    rows = []
    selected_by_config: dict[int, pd.DataFrame] = {}
    configuration_id = 0
    for robust_column in robust_columns:
        for gate_mode in gate_modes:
            cnn_affinity_grid = (
                comma_floats(args.minimum_cnnaffinity_deltas)
                if gate_mode in {"cnnscore_cnnaffinity", "all"} else [float("-inf")]
            )
            affinity_grid = (
                comma_floats(args.minimum_affinity_improvements)
                if gate_mode == "all" else [float("-inf")]
            )
            for score_advantage in comma_floats(args.minimum_score_advantages):
                for ncc_gain in comma_floats(args.minimum_ncc_gains):
                    for support in comma_floats(args.minimum_support_fractions):
                        for cnn_delta in comma_floats(args.minimum_cnnscore_deltas):
                            for cnn_affinity_delta in cnn_affinity_grid:
                                for affinity_improvement in affinity_grid:
                                    configuration_id += 1
                                    selected = select_rescues(
                                        scores,
                                        robust_column=robust_column,
                                        density_weight=density_weight,
                                        minimum_score_advantage=score_advantage,
                                        minimum_ncc_gain=ncc_gain,
                                        minimum_support_fraction=support,
                                        minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
                                        maximum_centroid_shift_A=args.maximum_centroid_shift,
                                        gate_mode=gate_mode,
                                        minimum_cnnscore_delta=cnn_delta,
                                        minimum_cnnaffinity_delta=cnn_affinity_delta,
                                        minimum_affinity_improvement=affinity_improvement,
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
                                    summary = paired_summary(tuning)
                                    rows.append({
                                        "configuration_id": configuration_id,
                                        "robust_column": robust_column,
                                        "gate_mode": gate_mode,
                                        "minimum_score_advantage": score_advantage,
                                        "minimum_ncc_gain": ncc_gain,
                                        "minimum_support_fraction": support,
                                        "minimum_cnnscore_delta": cnn_delta,
                                        "minimum_cnnaffinity_delta": cnn_affinity_delta,
                                        "minimum_affinity_improvement": affinity_improvement,
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

    best_config = {}
    for key in [
            "robust_column", "gate_mode", "minimum_score_advantage",
            "minimum_ncc_gain", "minimum_support_fraction",
            "minimum_cnnscore_delta", "minimum_cnnaffinity_delta",
            "minimum_affinity_improvement",
    ]:
        value = best[key]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            value = None
        best_config[key] = value
    best_config.update({
        "density_weight": density_weight,
        "minimum_in_bounds_fraction": args.minimum_in_bounds_fraction,
        "maximum_centroid_shift_A": args.maximum_centroid_shift,
        "tuning_metrics": paired_summary(tuning),
        "holdout_metrics": paired_summary(holdout),
        "all_validation_metrics": paired_summary(paired),
        "uses_ground_truth_for_hyperparameter_selection": True,
        "selection_baseline": "frozen_s0",
        "gnina_usage": "relative_gate_only_not_candidate_ranking",
        "warning": "Freeze this configuration before independent test evaluation.",
    })
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n")
    print("Top tuning configurations:")
    print(sweep.head(min(10, len(sweep))).to_string(index=False))
    print("\n" + json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
