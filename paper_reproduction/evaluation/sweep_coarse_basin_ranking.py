#!/usr/bin/env python3
"""Tune a conservative coarse-basin rescue on top of a frozen S0 selector.

Hyperparameters are chosen only on the tuning split.  The held-out complexes
are evaluated once after the winning configuration is frozen.  RMSD is used
only here for validation; deployment selection uses confidence, robust NCC,
relative clashes, crop coverage and basin-support statistics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def comma_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated numbers")
    return result


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def paired_summary(paired: pd.DataFrame) -> dict[str, float | int]:
    delta = paired["selected_rmsd_A"] - paired["s0_rmsd_A"]
    s0_success2 = paired["s0_rmsd_A"] < 2.0
    selected_success2 = paired["selected_rmsd_A"] < 2.0
    s0_success5 = paired["s0_rmsd_A"] < 5.0
    selected_success5 = paired["selected_rmsd_A"] < 5.0
    gained2 = int((~s0_success2 & selected_success2).sum())
    lost2 = int((s0_success2 & ~selected_success2).sum())
    gained5 = int((~s0_success5 & selected_success5).sum())
    lost5 = int((s0_success5 & ~selected_success5).sum())
    try:
        pvalue = float(wilcoxon(delta).pvalue)
    except ValueError:
        pvalue = 1.0
    discordant = gained2 + lost2
    return {
        "num_complexes": int(len(paired)),
        "median_top1_rmsd_A": float(paired["selected_rmsd_A"].median()),
        "mean_top1_rmsd_A": float(paired["selected_rmsd_A"].mean()),
        "top1_lt2A_pct": float(100.0 * selected_success2.mean()),
        "top1_lt5A_pct": float(100.0 * selected_success5.mean()),
        "median_delta_A": float(delta.median()),
        "mean_delta_A": float(delta.mean()),
        "better_pct": float(100.0 * (delta < -1e-8).mean()),
        "worse_pct": float(100.0 * (delta > 1e-8).mean()),
        "gained_lt2A": gained2,
        "lost_lt2A": lost2,
        "net_gained_lt2A": gained2 - lost2,
        "gained_lt5A": gained5,
        "lost_lt5A": lost5,
        "net_gained_lt5A": gained5 - lost5,
        "catastrophic_worsening_gt2A": int((delta > 2.0).sum()),
        "selected_coarse_count": int((paired["selected_role"] == "coarse").sum()),
        "wilcoxon_pvalue": pvalue,
        "mcnemar_pvalue": (
            float(binomtest(gained2, discordant, 0.5).pvalue)
            if discordant else 1.0
        ),
    }


def select_rescues(
    scores: pd.DataFrame,
    *,
    robust_column: str,
    density_weight: float,
    coarse_penalty: float,
    replacement_margin: float,
    minimum_ncc_gain: float,
    minimum_support_fraction: float,
    minimum_physics_fraction: float,
    minimum_in_bounds_fraction: float,
    maximum_centroid_shift_A: float,
    clash_penalty: float,
) -> pd.DataFrame:
    selected_rows = []
    for row_index, group in scores.groupby("row_index", sort=True):
        fallback = group[group["candidate_role"].astype(str).eq("s0")]
        if len(fallback) != 1:
            raise ValueError(f"row {row_index} has {len(fallback)} S0 candidates")
        fallback = fallback.iloc[0].copy()
        # Preserve the frozen S0 score, replacing its point NCC contribution
        # with the same robust-neighbour statistic used for coarse candidates.
        fallback_robust_score = float(fallback["s0_selection_score"]) + float(
            density_weight
        ) * (
            float(fallback[robust_column])
            - float(fallback["expanded_density_z"])
        )

        coarse = group[group["candidate_role"].astype(str).eq("coarse")].copy()
        if not coarse.empty:
            severe = coarse["severe_clash_count"].fillna(0).to_numpy(float)
            parent = coarse["parent_severe_clash_count"].fillna(0).to_numpy(float)
            clash_excess = np.maximum(severe - parent, 0.0)
            coarse["robust_selection_score"] = (
                coarse["confidence_z"].to_numpy(float)
                + float(density_weight) * coarse[robust_column].to_numpy(float)
                - float(coarse_penalty)
                - float(clash_penalty) * np.log1p(clash_excess)
            )
            eligible = (
                (coarse["ncc_gain"].fillna(-np.inf).to_numpy(float)
                 >= float(minimum_ncc_gain))
                & (coarse["ncc_support_fraction"].to_numpy(float)
                   >= float(minimum_support_fraction))
                & (coarse["neighbor_physics_pass_fraction"].to_numpy(float)
                   >= float(minimum_physics_fraction))
                & (coarse["neighbor_in_bounds_fraction"].to_numpy(float)
                   >= float(minimum_in_bounds_fraction))
                & (coarse["centroid_shift_A"].to_numpy(float)
                   <= float(maximum_centroid_shift_A))
                & as_bool(coarse["relative_physics_pass"])
            )
            coarse = coarse[eligible]
        if not coarse.empty:
            best = coarse.sort_values(
                ["robust_selection_score", "candidate_id"],
                ascending=[False, True], kind="stable",
            ).iloc[0].copy()
            if float(best["robust_selection_score"]) >= (
                fallback_robust_score + float(replacement_margin)
            ):
                best["selected_role"] = "coarse"
                best["final_selection_score"] = float(best["robust_selection_score"])
                selected_rows.append(best)
                continue
        fallback["selected_role"] = "s0"
        fallback["robust_selection_score"] = fallback_robust_score
        fallback["final_selection_score"] = fallback_robust_score
        selected_rows.append(fallback)
    return pd.DataFrame(selected_rows).sort_values("row_index").reset_index(drop=True)


def materialize(output: Path, selected: pd.DataFrame) -> None:
    root = output / "best_selected"
    root.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in str(row["complex_name"])
        )
        directory = root / f"index{int(row['row_index'])}___{name}"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(str(row["materialized_path"]))
        score = float(row["final_selection_score"])
        destination = directory / f"rank1_confidence{score:.8f}.sdf"
        shutil.copy2(source, destination)
        protein = Path(str(row["protein_path"]))
        if protein.is_file():
            shutil.copy2(protein, destination.with_name(f"{destination.stem}_protein.pdb"))
        pd.DataFrame([row.to_dict()]).to_csv(directory / "ranking.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--basin-scores", required=True, type=Path)
    parser.add_argument("--s0-selected", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection-holdout-file", required=True, type=Path)
    parser.add_argument("--robust-columns", default="neighbor_ncc_z_q25,neighbor_ncc_z_median")
    parser.add_argument("--coarse-penalties", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--replacement-margins", default="0,0.25,0.5,0.75")
    parser.add_argument("--minimum-ncc-gains", default="0.005,0.01")
    parser.add_argument("--minimum-support-fractions", default="0.5,0.7,0.85")
    parser.add_argument("--minimum-physics-fractions", default="0.7,0.85,1.0")
    parser.add_argument("--minimum-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-centroid-shift", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.basin_scores.expanduser().resolve())
    s0 = pd.read_csv(args.s0_selected.expanduser().resolve())
    config = json.loads(args.s0_config.expanduser().resolve().read_text())
    density_weight = float(config["density_weight"])
    clash_penalty = float(config.get("clash_penalty", 0.0))

    metrics = pd.read_csv(search_root / "metrics_expanded_pool" / "per_pose.csv")
    metrics = metrics[["row_index", "rank", "ligand_rmsd_A"]]
    scores = scores.merge(
        metrics,
        left_on=["row_index", "materialized_rank"],
        right_on=["row_index", "rank"],
        how="inner", validate="one_to_one",
    )
    if scores.empty:
        raise ValueError("No basin candidates matched expanded-pool RMSD metrics")
    s0_rmsd = s0[["row_index", "complex_name", "ligand_rmsd_A"]].rename(
        columns={"ligand_rmsd_A": "s0_rmsd_A"}
    )
    basin_s0_names = scores[
        scores["candidate_role"].astype(str).eq("s0")
    ][["row_index", "complex_name"]].drop_duplicates()
    name_check = s0_rmsd.merge(
        basin_s0_names,
        on="row_index",
        how="outer",
        suffixes=("_s0", "_basin"),
        indicator=True,
    )
    mismatched = name_check[
        (name_check["_merge"] != "both")
        | (
            name_check["complex_name_s0"].astype(str)
            != name_check["complex_name_basin"].astype(str)
        )
    ]
    if not mismatched.empty:
        preview = mismatched.head(10).to_dict(orient="records")
        raise ValueError(
            "S0/basin row-to-complex mapping differs. Recompute basin scores "
            f"with the matching S0_ROOT. First mismatches: {preview}"
        )
    holdout_names = {
        line.strip() for line in args.selection_holdout_file.expanduser().resolve().read_text().splitlines()
        if line.strip()
    }
    if not holdout_names:
        raise ValueError("holdout file is empty")
    available = set(s0_rmsd["complex_name"].astype(str))
    absent = sorted(holdout_names - available)
    if absent:
        raise ValueError(f"Holdout complexes are absent: {absent}")
    tuning_names = available - holdout_names

    robust_columns = [item.strip() for item in args.robust_columns.split(",") if item.strip()]
    for column in robust_columns:
        if column not in scores.columns:
            raise ValueError(f"Unknown robust column: {column}")

    rows = []
    selected_by_config: dict[int, pd.DataFrame] = {}
    configuration_id = 0
    for robust_column in robust_columns:
        for coarse_penalty in comma_floats(args.coarse_penalties):
            for margin in comma_floats(args.replacement_margins):
                for minimum_gain in comma_floats(args.minimum_ncc_gains):
                    for support in comma_floats(args.minimum_support_fractions):
                        for physics in comma_floats(args.minimum_physics_fractions):
                            configuration_id += 1
                            selected = select_rescues(
                                scores,
                                robust_column=robust_column,
                                density_weight=density_weight,
                                coarse_penalty=coarse_penalty,
                                replacement_margin=margin,
                                minimum_ncc_gain=minimum_gain,
                                minimum_support_fraction=support,
                                minimum_physics_fraction=physics,
                                minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
                                maximum_centroid_shift_A=args.maximum_centroid_shift,
                                clash_penalty=clash_penalty,
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
                            tuning = paired[paired["complex_name"].astype(str).isin(tuning_names)]
                            summary = paired_summary(tuning)
                            rows.append({
                                "configuration_id": configuration_id,
                                "robust_column": robust_column,
                                "coarse_penalty": coarse_penalty,
                                "replacement_margin": margin,
                                "minimum_ncc_gain": minimum_gain,
                                "minimum_support_fraction": support,
                                "minimum_physics_fraction": physics,
                                **summary,
                            })
                            selected_by_config[configuration_id] = paired

    sweep = pd.DataFrame(rows).sort_values(
        [
            "lost_lt2A", "catastrophic_worsening_gt2A", "lost_lt5A",
            "net_gained_lt2A", "top1_lt2A_pct", "net_gained_lt5A",
            "mean_top1_rmsd_A", "selected_coarse_count", "configuration_id",
        ],
        ascending=[True, True, True, False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    sweep.to_csv(output / "sweep.csv", index=False)
    sweep.head(args.top_k).to_csv(output / "top_configurations.csv", index=False)
    best = sweep.iloc[0].to_dict()
    paired = selected_by_config[int(best["configuration_id"])].copy()
    tuning = paired[paired["complex_name"].astype(str).isin(tuning_names)].copy()
    holdout = paired[paired["complex_name"].astype(str).isin(holdout_names)].copy()
    full = paired.copy()
    tuning.to_csv(output / "best_paired_tuning.csv", index=False)
    holdout.to_csv(output / "best_paired_holdout.csv", index=False)
    full.to_csv(output / "best_paired_complexes.csv", index=False)

    selected_ids = set(zip(full["row_index"].astype(int), full["candidate_id"].astype(str)))
    selected_rows = scores[
        [(int(row.row_index), str(row.candidate_id)) in selected_ids for row in scores.itertuples()]
    ].copy()
    selected_rows = selected_rows.merge(
        full[["row_index", "candidate_id", "selected_role", "final_selection_score"]],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    ).sort_values("row_index")
    selected_rows.to_csv(output / "best_selected_candidates.csv", index=False)
    materialize(output, selected_rows)

    best_config = {
        "robust_column": str(best["robust_column"]),
        "density_weight": density_weight,
        "coarse_penalty": float(best["coarse_penalty"]),
        "replacement_margin": float(best["replacement_margin"]),
        "minimum_ncc_gain": float(best["minimum_ncc_gain"]),
        "minimum_support_fraction": float(best["minimum_support_fraction"]),
        "minimum_physics_fraction": float(best["minimum_physics_fraction"]),
        "minimum_in_bounds_fraction": args.minimum_in_bounds_fraction,
        "maximum_centroid_shift_A": args.maximum_centroid_shift,
        "tuning_metrics": paired_summary(tuning),
        "holdout_metrics": paired_summary(holdout),
        "all_validation_metrics": paired_summary(full),
        "uses_ground_truth_for_hyperparameter_selection": True,
        "selection_baseline": "frozen_s0",
        "warning": "Freeze this configuration before independent test evaluation.",
    }
    (output / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n")
    print("Top tuning configurations:")
    print(sweep.head(min(10, len(sweep))).to_string(index=False))
    print(json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
