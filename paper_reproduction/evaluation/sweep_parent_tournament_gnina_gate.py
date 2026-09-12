#!/usr/bin/env python3
"""Attempt 2: parent-wise density tournament with binary GNINA gates.

GNINA is deliberately not used as a continuous within-complex ranking score.
For every direct parent pose, density-supported variants first pass binary
candidate-vs-parent GNINA gates and then compete by robust NCC.  The surviving
winner from each parent competes using the frozen confidence+density score and
must also pass candidate-vs-S0 GNINA gates.  S0 is the exact fallback.

RMSD is used only to select hyperparameters on the tuning cohort and to report
the untouched holdout cohort.  It is never available to deployment selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_gnina_relative_coarse_gate import materialize
from evaluation.sweep_parent_relative_gnina_gate import (
    comma_floats,
    comma_strings,
    prepare_scores,
    quiet_paired_summary,
    signal_summary,
)


def canonical_complex_name(value: object) -> str:
    """Undo pandas scientific-notation coercion of PDB IDs such as 8E94."""
    text = str(value).strip()
    match = re.fullmatch(r"([0-9])e\+?([0-9]{2})", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}E{match.group(2)}"
    return text.upper()


def select_parent_tournament(
    scores: pd.DataFrame,
    *,
    robust_column: str,
    density_weight: float,
    minimum_score_advantage: float,
    minimum_ncc_gain: float,
    minimum_support_fraction: float,
    minimum_parent_cnnscore_delta: float,
    minimum_parent_cnnaffinity_delta: float,
    minimum_cnnscore_delta_vs_s0: float,
    minimum_cnnaffinity_delta_vs_s0: float,
    minimum_in_bounds_fraction: float,
    maximum_centroid_shift_A: float,
) -> pd.DataFrame:
    """Select one pose per complex while keeping S0 as an exact fallback."""
    selected_rows = []
    for row_index, group in scores.groupby("row_index", sort=True):
        fallback_rows = group[group["candidate_role"].astype(str).eq("s0")]
        if len(fallback_rows) != 1:
            raise ValueError(
                f"row {row_index} has {len(fallback_rows)} S0 candidates"
            )
        fallback = fallback_rows.iloc[0].copy()
        fallback_score = float(fallback["s0_selection_score"]) + float(
            density_weight
        ) * (
            float(fallback[robust_column])
            - float(fallback["expanded_density_z"])
        )

        coarse = group[group["candidate_role"].astype(str).eq("coarse")].copy()
        parent_winners = coarse.iloc[0:0].copy()
        if not coarse.empty:
            eligible = (
                (coarse["ncc_gain"].fillna(-np.inf).astype(float)
                 >= float(minimum_ncc_gain))
                & (coarse["ncc_support_fraction"].fillna(0).astype(float)
                   >= float(minimum_support_fraction))
                & (coarse["neighbor_in_bounds_fraction"].fillna(0).astype(float)
                   >= float(minimum_in_bounds_fraction))
                & (coarse["centroid_shift_A"].fillna(np.inf).astype(float)
                   <= float(maximum_centroid_shift_A))
                & (coarse["delta_parent_cnnscore"].fillna(-np.inf).astype(float)
                   >= float(minimum_parent_cnnscore_delta))
                & (coarse["delta_parent_cnnaffinity"].fillna(-np.inf).astype(float)
                   >= float(minimum_parent_cnnaffinity_delta))
            )
            eligible_coarse = coarse[eligible].copy()
            if not eligible_coarse.empty:
                eligible_coarse["parent_eligible_count"] = (
                    eligible_coarse.groupby("parent_id")["candidate_id"]
                    .transform("size")
                )
                # GNINA has candidate-level gate signal but near-zero
                # within-complex Spearman.  It therefore cannot break ties or
                # rank variants here; robust NCC exclusively chooses the local
                # winner for each direct parent.
                parent_winners = (
                    eligible_coarse.sort_values(
                        [robust_column, "ncc_gain", "candidate_id"],
                        ascending=[False, False, True],
                        kind="stable",
                    )
                    .drop_duplicates("parent_id", keep="first")
                    .copy()
                )
                parent_winners["tournament_local_score"] = parent_winners[
                    robust_column
                ].astype(float)

                # The parent winners now compete across binding modes using
                # the same frozen confidence+density form as S0.
                parent_winners["tournament_final_score"] = (
                    parent_winners["confidence_z"].astype(float)
                    + float(density_weight)
                    * parent_winners[robust_column].astype(float)
                )
                final_gate = (
                    (parent_winners["delta_cnnscore_vs_s0"].fillna(-np.inf)
                     .astype(float) >= float(minimum_cnnscore_delta_vs_s0))
                    & (parent_winners["delta_cnnaffinity_vs_s0"].fillna(-np.inf)
                       .astype(float)
                       >= float(minimum_cnnaffinity_delta_vs_s0))
                )
                parent_winners = parent_winners[final_gate].copy()

        if not parent_winners.empty:
            best = parent_winners.sort_values(
                ["tournament_final_score", robust_column, "ncc_gain", "candidate_id"],
                ascending=[False, False, False, True],
                kind="stable",
            ).iloc[0].copy()
            if float(best["tournament_final_score"]) >= (
                fallback_score + float(minimum_score_advantage)
            ):
                best["selected_role"] = "coarse"
                best["final_selection_score"] = float(
                    best["tournament_final_score"]
                )
                best["num_parent_winners"] = int(len(parent_winners))
                selected_rows.append(best)
                continue

        fallback["selected_role"] = "s0"
        fallback["tournament_local_score"] = float(fallback[robust_column])
        fallback["tournament_final_score"] = fallback_score
        fallback["final_selection_score"] = fallback_score
        fallback["num_parent_winners"] = int(len(parent_winners))
        selected_rows.append(fallback)

    return pd.DataFrame(selected_rows).sort_values("row_index").reset_index(drop=True)


def tournament_diagnostic(scores: pd.DataFrame, robust_column: str) -> dict:
    """Measure density-only local-winner regret; diagnostic only, never deployed."""
    coarse = scores[scores["candidate_role"].astype(str).eq("coarse")].copy()
    rows = []
    for (_, _), group in coarse.groupby(["row_index", "parent_id"], sort=True):
        density_winner = group.sort_values(
            [robust_column, "ncc_gain", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        ).iloc[0]
        oracle_rmsd = float(group["ligand_rmsd_A"].min())
        winner_rmsd = float(density_winner["ligand_rmsd_A"])
        rows.append({
            "winner_rmsd_A": winner_rmsd,
            "oracle_rmsd_A": oracle_rmsd,
            "regret_A": winner_rmsd - oracle_rmsd,
            "winner_improves_parent": (
                winner_rmsd < float(density_winner["parent_rmsd_A"])
            ),
        })
    frame = pd.DataFrame(rows)
    return {
        "robust_column": robust_column,
        "num_parent_groups": int(len(frame)),
        "density_winner_is_oracle_pct": float(
            100.0 * np.mean(frame["regret_A"].to_numpy(float) <= 1e-8)
        ),
        "density_winner_improves_parent_pct": float(
            100.0 * frame["winner_improves_parent"].mean()
        ),
        "median_density_winner_regret_A": float(frame["regret_A"].median()),
    }


def pair_selection(s0_rmsd: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    return s0_rmsd.merge(
        selected[[
            "row_index", "candidate_id", "candidate_role", "ligand_rmsd_A",
            "move_type", "materialized_path", "protein_path",
            "final_selection_score", "num_parent_winners",
        ]].rename(columns={
            "ligand_rmsd_A": "selected_rmsd_A",
            "candidate_role": "selected_role",
        }),
        on="row_index", how="inner", validate="one_to_one",
    )


def json_scalar(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


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
    parser.add_argument("--minimum-score-advantages", default="-1,-0.5,0")
    parser.add_argument("--minimum-ncc-gains", default="0.005,0.02,0.05")
    parser.add_argument("--minimum-support-fractions", default="0.4,0.5,0.6")
    parser.add_argument(
        "--minimum-parent-cnnscore-deltas", default="-0.2,-0.1,0"
    )
    parser.add_argument(
        "--minimum-parent-cnnaffinity-deltas", default="-1,-0.5,0"
    )
    parser.add_argument(
        "--minimum-cnnscore-delta-vs-s0", type=float, default=0.0
    )
    parser.add_argument(
        "--minimum-cnnaffinity-delta-vs-s0", type=float, default=-0.5
    )
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
    scores["complex_name"] = scores["complex_name"].map(canonical_complex_name)
    scores.to_csv(output / "tournament_candidates.csv", index=False)

    s0 = pd.read_csv(args.s0_selected.expanduser().resolve(), dtype={"complex_name": str})
    s0["complex_name"] = s0["complex_name"].map(canonical_complex_name)
    config = json.loads(args.s0_config.expanduser().resolve().read_text())
    density_weight = float(config["density_weight"])
    s0_rmsd = s0[["row_index", "complex_name", "ligand_rmsd_A"]].rename(
        columns={"ligand_rmsd_A": "s0_rmsd_A"}
    )

    available = set(s0_rmsd["complex_name"].astype(str))
    holdout_names = {
        canonical_complex_name(line)
        for line in args.selection_holdout_file.expanduser().resolve()
        .read_text().splitlines()
        if line.strip()
    }
    absent = sorted(holdout_names - available)
    if absent:
        raise ValueError(f"Holdout complexes are absent: {absent}")
    tuning_names = available - holdout_names

    coarse = scores[scores["candidate_role"].astype(str).eq("coarse")].copy()
    signal = {
        "tuning": signal_summary(
            coarse[coarse["complex_name"].astype(str).isin(tuning_names)]
        ),
        "holdout": signal_summary(
            coarse[coarse["complex_name"].astype(str).isin(holdout_names)]
        ),
        "all_validation": signal_summary(coarse),
        "uses_ground_truth": True,
    }
    diagnostics = {
        column: tournament_diagnostic(scores, column)
        for column in comma_strings(args.robust_columns)
    }
    (output / "signal_summary.json").write_text(
        json.dumps(signal, indent=2) + "\n"
    )
    (output / "tournament_diagnostic.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )

    robust_columns = comma_strings(args.robust_columns)
    for column in robust_columns:
        if column not in scores.columns:
            raise ValueError(f"Unknown robust column: {column}")

    configurations = []
    configuration_id = 0
    for robust_column in robust_columns:
        for score_advantage in comma_floats(args.minimum_score_advantages):
            for ncc_gain in comma_floats(args.minimum_ncc_gains):
                for support in comma_floats(args.minimum_support_fractions):
                    for parent_cnnscore in comma_floats(
                        args.minimum_parent_cnnscore_deltas
                    ):
                        for parent_cnnaffinity in comma_floats(
                            args.minimum_parent_cnnaffinity_deltas
                        ):
                            configuration_id += 1
                            selected = select_parent_tournament(
                                scores,
                                robust_column=robust_column,
                                density_weight=density_weight,
                                minimum_score_advantage=score_advantage,
                                minimum_ncc_gain=ncc_gain,
                                minimum_support_fraction=support,
                                minimum_parent_cnnscore_delta=parent_cnnscore,
                                minimum_parent_cnnaffinity_delta=parent_cnnaffinity,
                                minimum_cnnscore_delta_vs_s0=(
                                    args.minimum_cnnscore_delta_vs_s0
                                ),
                                minimum_cnnaffinity_delta_vs_s0=(
                                    args.minimum_cnnaffinity_delta_vs_s0
                                ),
                                minimum_in_bounds_fraction=(
                                    args.minimum_in_bounds_fraction
                                ),
                                maximum_centroid_shift_A=args.maximum_centroid_shift,
                            )
                            paired = pair_selection(s0_rmsd, selected)
                            tuning = paired[
                                paired["complex_name"].astype(str).isin(tuning_names)
                            ]
                            summary = quiet_paired_summary(tuning)
                            configurations.append({
                                "configuration_id": configuration_id,
                                "robust_column": robust_column,
                                "minimum_score_advantage": score_advantage,
                                "minimum_ncc_gain": ncc_gain,
                                "minimum_support_fraction": support,
                                "minimum_parent_cnnscore_delta": parent_cnnscore,
                                "minimum_parent_cnnaffinity_delta": (
                                    parent_cnnaffinity
                                ),
                                **summary,
                            })

    sweep = pd.DataFrame(configurations).sort_values(
        [
            "catastrophic_worsening_gt2A", "lost_lt2A", "net_gained_lt2A",
            "top1_lt2A_pct", "lost_lt5A", "net_gained_lt5A",
            "mean_top1_rmsd_A", "selected_coarse_count", "configuration_id",
        ],
        ascending=[True, True, False, False, True, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    sweep.to_csv(output / "sweep.csv", index=False)
    sweep.head(args.top_k).to_csv(output / "top_configurations.csv", index=False)

    best = sweep.iloc[0].to_dict()
    selected = select_parent_tournament(
        scores,
        robust_column=str(best["robust_column"]),
        density_weight=density_weight,
        minimum_score_advantage=float(best["minimum_score_advantage"]),
        minimum_ncc_gain=float(best["minimum_ncc_gain"]),
        minimum_support_fraction=float(best["minimum_support_fraction"]),
        minimum_parent_cnnscore_delta=float(
            best["minimum_parent_cnnscore_delta"]
        ),
        minimum_parent_cnnaffinity_delta=float(
            best["minimum_parent_cnnaffinity_delta"]
        ),
        minimum_cnnscore_delta_vs_s0=args.minimum_cnnscore_delta_vs_s0,
        minimum_cnnaffinity_delta_vs_s0=args.minimum_cnnaffinity_delta_vs_s0,
        minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
        maximum_centroid_shift_A=args.maximum_centroid_shift,
    )
    paired = pair_selection(s0_rmsd, selected)
    tuning = paired[paired["complex_name"].astype(str).isin(tuning_names)].copy()
    holdout = paired[paired["complex_name"].astype(str).isin(holdout_names)].copy()
    paired.to_csv(output / "best_paired_complexes.csv", index=False)
    tuning.to_csv(output / "best_paired_tuning.csv", index=False)
    holdout.to_csv(output / "best_paired_holdout.csv", index=False)

    selected.to_csv(output / "best_selected_candidates.csv", index=False)
    materialize(output, selected)

    parameter_keys = [
        "robust_column", "minimum_score_advantage", "minimum_ncc_gain",
        "minimum_support_fraction", "minimum_parent_cnnscore_delta",
        "minimum_parent_cnnaffinity_delta",
    ]
    best_config = {key: json_scalar(best[key]) for key in parameter_keys}
    best_config.update({
        "density_weight": density_weight,
        "minimum_cnnscore_delta_vs_s0": args.minimum_cnnscore_delta_vs_s0,
        "minimum_cnnaffinity_delta_vs_s0": (
            args.minimum_cnnaffinity_delta_vs_s0
        ),
        "minimum_in_bounds_fraction": args.minimum_in_bounds_fraction,
        "maximum_centroid_shift_A": args.maximum_centroid_shift,
        "tuning_metrics": quiet_paired_summary(tuning),
        "holdout_metrics": quiet_paired_summary(holdout),
        "all_validation_metrics": quiet_paired_summary(paired),
        "uses_ground_truth_for_hyperparameter_selection": True,
        "selection_baseline": "frozen_s0",
        "gnina_usage": "binary_parent_and_s0_gates_only",
        "within_parent_ranker": "robust_ncc_only",
        "warning": "Freeze this configuration before independent test evaluation.",
    })
    (output / "best_config.json").write_text(
        json.dumps(best_config, indent=2) + "\n"
    )
    print("PARENT TOURNAMENT DIAGNOSTIC")
    print(json.dumps(diagnostics, indent=2))
    print("\nTop tuning configurations:")
    print(sweep.head(min(10, len(sweep))).to_string(index=False))
    print("\n" + json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
