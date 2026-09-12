#!/usr/bin/env python3
"""Tune deployment-safe ranking rules on an expanded local-search pool.

This is validation/evaluation code: RMSD is used to compare hyperparameter
configurations and must never be called during deployment.  Candidate scores
themselves use only confidence, density NCC, move size, crop coverage and an
optional soft clash penalty.  The selected hyperparameters must be frozen
before running on an independent test split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def comma_floats(value: str, *, allow_inf: bool = False) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if allow_inf and item.lower() in {"inf", "infinity", "none"}:
            values.append(float("inf"))
        else:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one comma-separated number")
    return values


def boolean_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def load_expanded_candidates(
    search_root: Path,
    rescored_confidence: Path | None = None,
) -> pd.DataFrame:
    ranking_files = sorted(
        (search_root / "expanded_pool").glob("index*___*/ranking.csv")
    )
    if not ranking_files:
        raise FileNotFoundError(
            f"No expanded_pool ranking.csv files under {search_root}"
        )
    rankings = pd.concat(
        [pd.read_csv(path) for path in ranking_files], ignore_index=True
    )
    required_ranking = {
        "row_index", "complex_name", "candidate_id", "materialized_rank",
        "materialized_path", "is_original", "confidence_z",
        "expanded_density_z", "anchor_rmsd_A", "centroid_shift_A",
        "ncc_gain", "in_bounds_fraction", "move_type", "protein_path",
    }
    missing = sorted(required_ranking - set(rankings.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")

    pose_metrics_path = search_root / "metrics_expanded_pool" / "per_pose.csv"
    if not pose_metrics_path.is_file():
        raise FileNotFoundError(f"Missing expanded-pool metrics: {pose_metrics_path}")
    pose_metrics = pd.read_csv(pose_metrics_path)
    metric_columns = [
        "row_index", "rank", "ligand_rmsd_A", "centroid_distance_A",
        "centered_rmsd_A", "rigid_aligned_rmsd_A",
    ]
    missing_metrics = sorted(set(metric_columns) - set(pose_metrics.columns))
    if missing_metrics:
        raise ValueError(f"Expanded per-pose metrics are missing: {missing_metrics}")

    candidates = rankings.merge(
        pose_metrics[metric_columns],
        left_on=["row_index", "materialized_rank"],
        right_on=["row_index", "rank"],
        how="inner",
        validate="one_to_one",
    )
    if len(candidates) != len(rankings):
        raise ValueError(
            f"Only {len(candidates)} / {len(rankings)} candidates matched RMSD metrics"
        )
    candidates["is_original_bool"] = boolean_array(candidates["is_original"])
    if rescored_confidence is not None:
        rescored = pd.read_csv(rescored_confidence)
        required_rescored = {
            "row_index", "candidate_id", "confidence_recomputed",
            "confidence_recomputed_z",
        }
        missing_rescored = sorted(required_rescored - set(rescored.columns))
        if missing_rescored:
            raise ValueError(
                f"Rescored confidence table is missing: {missing_rescored}"
            )
        candidates = candidates.merge(
            rescored[list(required_rescored)],
            on=["row_index", "candidate_id"],
            how="inner",
            validate="one_to_one",
        )
        if candidates.empty:
            raise ValueError("No expanded candidates matched rescored confidence")
        candidates["confidence_z_inherited"] = candidates["confidence_z"]
        candidates["confidence_z"] = candidates["confidence_recomputed_z"]
    candidates = candidates.sort_values(
        ["row_index", "candidate_id"], kind="stable"
    ).reset_index(drop=True)
    return candidates


def load_baseline(search_root: Path) -> pd.DataFrame:
    path = search_root / "metrics_original_density" / "per_complex.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing original-density metrics: {path}")
    baseline = pd.read_csv(path)
    required = {"row_index", "complex_name", "top1_rmsd_A"}
    missing = sorted(required - set(baseline.columns))
    if missing:
        raise ValueError(f"Baseline metrics are missing columns: {missing}")
    return baseline.sort_values("row_index").reset_index(drop=True)


def group_slices(row_indices: np.ndarray) -> list[tuple[int, int, int]]:
    if len(row_indices) == 0:
        return []
    boundaries = np.flatnonzero(np.r_[True, row_indices[1:] != row_indices[:-1], True])
    return [
        (int(row_indices[start]), int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def select_indices(
    candidates: pd.DataFrame,
    slices: list[tuple[int, int, int]],
    *,
    density_weight: float,
    anchor_penalty: float,
    minimum_ncc_gain: float,
    maximum_anchor_rmsd_A: float,
    maximum_centroid_shift_A: float,
    minimum_in_bounds_fraction: float,
    coarse_orientation_penalty: float,
    clash_penalty: float,
    coarse_rescue: bool = False,
    coarse_minimum_ncc_gain: float = 0.005,
    coarse_maximum_centroid_shift_A: float = 0.25,
) -> np.ndarray:
    confidence = candidates["confidence_z"].to_numpy(dtype=float)
    density = candidates["expanded_density_z"].to_numpy(dtype=float)
    anchor = candidates["anchor_rmsd_A"].to_numpy(dtype=float)
    ncc_gain = candidates["ncc_gain"].fillna(0.0).to_numpy(dtype=float)
    centroid = candidates["centroid_shift_A"].to_numpy(dtype=float)
    in_bounds = candidates["in_bounds_fraction"].to_numpy(dtype=float)
    original = candidates["is_original_bool"].to_numpy(dtype=bool)
    coarse = candidates["move_type"].astype(str).eq("coarse_orientation").to_numpy()

    clash_count = candidates.get(
        "severe_clash_count", pd.Series(np.zeros(len(candidates)))
    ).fillna(0.0).to_numpy(dtype=float)
    parent_clash_count = candidates.get(
        "parent_severe_clash_count", pd.Series(np.zeros(len(candidates)))
    ).to_numpy(dtype=float)
    parent_clash_count = np.where(
        np.isfinite(parent_clash_count), parent_clash_count, clash_count
    )
    clash_excess = np.maximum(clash_count - parent_clash_count, 0.0)

    normal_eligible_variant = (
        (ncc_gain >= float(minimum_ncc_gain))
        & (anchor <= float(maximum_anchor_rmsd_A))
        & (centroid <= float(maximum_centroid_shift_A))
        & (in_bounds >= float(minimum_in_bounds_fraction))
    )
    if coarse_rescue:
        # A coarse orientation is an intentional large rigid-body rotation
        # around the ligand centroid.  Raw RMSD to its parent is therefore a
        # move magnitude, not a safety diagnostic.  Use a stricter density
        # gate and near-zero centroid shift instead of the generic anchor gate.
        coarse_eligible_variant = (
            coarse
            & (ncc_gain >= float(coarse_minimum_ncc_gain))
            & (centroid <= float(coarse_maximum_centroid_shift_A))
            & (in_bounds >= float(minimum_in_bounds_fraction))
        )
        eligible = (
            original
            | (normal_eligible_variant & ~coarse)
            | coarse_eligible_variant
        )
        # Do not charge a per-Angstrom anchor penalty for a move whose purpose
        # is to make a large orientation jump.  The fixed coarse penalty is the
        # conservative prior against selecting this rescue branch.
        anchor_cost = float(anchor_penalty) * anchor * (~coarse).astype(float)
    else:
        eligible = original | normal_eligible_variant
        anchor_cost = float(anchor_penalty) * anchor
    score = (
        confidence
        + float(density_weight) * density
        - anchor_cost
        - float(coarse_orientation_penalty) * coarse.astype(float)
        - float(clash_penalty) * np.log1p(clash_excess)
    )
    score[~eligible | ~np.isfinite(score)] = -np.inf

    selected = []
    for _, start, end in slices:
        local = score[start:end]
        if not np.isfinite(local).any():
            raise RuntimeError("A complex has no eligible candidates")
        selected.append(start + int(np.argmax(local)))
    return np.asarray(selected, dtype=int)


def paired_statistics(
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    selected_table = selected[[
        "row_index", "complex_name", "ligand_rmsd_A", "candidate_id",
        "move_type", "is_original_bool", "materialized_path", "protein_path",
        "ncc_gain", "anchor_rmsd_A", "centroid_shift_A",
    ]].rename(columns={"ligand_rmsd_A": "selected_rmsd_A"})
    paired = baseline[[
        "row_index", "complex_name", "top1_rmsd_A"
    ]].rename(columns={"complex_name": "baseline_complex_name"}).merge(
        selected_table,
        on=["row_index"],
        how="inner",
        validate="one_to_one",
    ).rename(columns={
        "top1_rmsd_A": "baseline_rmsd_A",
        "complex_name": "candidate_complex_name",
    })
    if len(paired) != len(baseline):
        raise ValueError(
            f"Baseline/selection pairing failed: {len(paired)} / {len(baseline)}"
        )
    paired["complex_name_matches"] = (
        paired["baseline_complex_name"].astype(str)
        == paired["candidate_complex_name"].astype(str)
    )
    paired["complex_name"] = paired["baseline_complex_name"]
    delta = paired["selected_rmsd_A"] - paired["baseline_rmsd_A"]
    paired["delta_A"] = delta
    baseline_success = paired["baseline_rmsd_A"] < 2.0
    selected_success = paired["selected_rmsd_A"] < 2.0
    gained = int((~baseline_success & selected_success).sum())
    lost = int((baseline_success & ~selected_success).sum())
    discordant = gained + lost
    try:
        wilcoxon_p = float(wilcoxon(
            paired["selected_rmsd_A"], paired["baseline_rmsd_A"]
        ).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    mcnemar_p = (
        float(binomtest(gained, discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    summary = {
        "num_complexes": int(len(paired)),
        "median_top1_rmsd_A": float(paired["selected_rmsd_A"].median()),
        "mean_top1_rmsd_A": float(paired["selected_rmsd_A"].mean()),
        "top1_lt2A_pct": float(100.0 * selected_success.mean()),
        "top1_lt5A_pct": float(100.0 * (paired["selected_rmsd_A"] < 5.0).mean()),
        "median_delta_A": float(delta.median()),
        "mean_delta_A": float(delta.mean()),
        "better_pct": float(100.0 * (delta < 0).mean()),
        "worse_pct": float(100.0 * (delta > 0).mean()),
        "gained_lt2A": gained,
        "lost_lt2A": lost,
        "net_gained_lt2A": gained - lost,
        "catastrophic_worsening_gt2A": int((delta > 2.0).sum()),
        "selected_new_variant_pct": float(
            100.0 * (~paired["is_original_bool"]).mean()
        ),
        "wilcoxon_pvalue": wilcoxon_p,
        "mcnemar_pvalue": mcnemar_p,
    }
    return summary, paired


def fold_metrics(paired: pd.DataFrame, folds: np.ndarray) -> dict[str, float]:
    success_rates = []
    deltas = []
    for fold in sorted(set(folds.tolist())):
        subset = paired.iloc[np.flatnonzero(folds == fold)]
        success_rates.append(float(100.0 * (subset["selected_rmsd_A"] < 2).mean()))
        deltas.append(float((subset["selected_rmsd_A"] - subset["baseline_rmsd_A"]).median()))
    return {
        "min_fold_top1_lt2A_pct": float(min(success_rates)),
        "max_fold_top1_lt2A_pct": float(max(success_rates)),
        "std_fold_top1_lt2A_pct": float(np.std(success_rates)),
        "max_fold_median_delta_A": float(max(deltas)),
    }


def materialize_best(
    output_root: Path,
    selected: pd.DataFrame,
    score_values: np.ndarray,
) -> None:
    root = output_root / "best_selected"
    root.mkdir(parents=True, exist_ok=True)
    for (_, row), score in zip(selected.iterrows(), score_values):
        row_index = int(row["row_index"])
        complex_name = str(row["complex_name"])
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in complex_name
        )
        directory = root / f"index{row_index}___{safe_name}"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(str(row["materialized_path"]))
        destination = directory / f"rank1_confidence{float(score):.8f}.sdf"
        shutil.copy2(source, destination)
        protein = Path(str(row["protein_path"]))
        if protein.is_file():
            shutil.copy2(
                protein,
                destination.with_name(f"{destination.stem}_protein.pdb"),
            )
        pd.DataFrame([row.to_dict()]).assign(selection_score=float(score)).to_csv(
            directory / "ranking.csv", index=False
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rescored-confidence", type=Path, default=None,
        help=(
            "Optional rescored_candidates.csv from S1. Only candidates in "
            "this GT-free shortlist participate, and recomputed confidence "
            "replaces inherited parent confidence."
        ),
    )
    parser.add_argument("--density-weights", default="1,1.5,2,2.5,3")
    parser.add_argument("--anchor-penalties", default="0,0.025,0.05,0.1,0.2")
    parser.add_argument("--minimum-ncc-gains", default="0,0.002,0.005,0.01")
    parser.add_argument("--maximum-anchor-rmsds", default="1,2,4,inf")
    parser.add_argument("--coarse-orientation-penalties", default="0,0.1,0.25")
    parser.add_argument(
        "--enable-coarse-rescue", action="store_true",
        help=(
            "Give coarse orientations a move-specific NCC/centroid gate and "
            "exempt them from the generic anchor-RMSD gate and penalty."
        ),
    )
    parser.add_argument(
        "--coarse-minimum-ncc-gains", default="0.005,0.01,0.02,0.03",
        help="Validation grid used only with --enable-coarse-rescue.",
    )
    parser.add_argument(
        "--coarse-maximum-centroid-shift", type=float, default=0.25,
    )
    parser.add_argument("--clash-penalties", default="0,0.02,0.05")
    parser.add_argument("--maximum-centroid-shift", type=float, default=1.5)
    parser.add_argument("--minimum-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260822)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--selection-holdout-file", type=Path, default=None,
        help=(
            "Optional one-complex-name-per-line holdout. Hyperparameters are "
            "chosen on the remaining complexes, then applied once to all and "
            "reported separately on this holdout."
        ),
    )
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rescored_path = (
        None if args.rescored_confidence is None
        else args.rescored_confidence.expanduser().resolve()
    )
    all_candidates = load_expanded_candidates(search_root, rescored_path)
    all_baseline = load_baseline(search_root)
    holdout_names: set[str] = set()
    if args.selection_holdout_file is not None:
        holdout_path = args.selection_holdout_file.expanduser().resolve()
        holdout_names = {
            line.strip() for line in holdout_path.read_text().splitlines()
            if line.strip()
        }
        available_names = set(all_baseline["complex_name"].astype(str))
        absent = sorted(holdout_names - available_names)
        if absent:
            raise ValueError(f"Holdout complexes are absent from search root: {absent}")
        if not holdout_names or len(holdout_names) >= len(all_baseline):
            raise ValueError("Holdout must contain between 1 and N-1 complexes")

    tuning_rows = ~all_baseline["complex_name"].astype(str).isin(holdout_names)
    baseline = all_baseline[tuning_rows].copy().reset_index(drop=True)
    tuning_indices = set(baseline["row_index"].astype(int))
    candidates = all_candidates[
        all_candidates["row_index"].astype(int).isin(tuning_indices)
    ].copy().reset_index(drop=True)
    slices = group_slices(candidates["row_index"].to_numpy(dtype=int))
    if len(slices) != len(baseline):
        raise ValueError(
            f"Found {len(slices)} candidate groups but {len(baseline)} baselines"
        )

    density_weights = comma_floats(args.density_weights)
    anchor_penalties = comma_floats(args.anchor_penalties)
    minimum_ncc_gains = comma_floats(args.minimum_ncc_gains)
    maximum_anchor_rmsds = comma_floats(
        args.maximum_anchor_rmsds, allow_inf=True
    )
    coarse_penalties = comma_floats(args.coarse_orientation_penalties)
    coarse_minimum_ncc_gains = (
        comma_floats(args.coarse_minimum_ncc_gains)
        if args.enable_coarse_rescue else [float("nan")]
    )
    clash_penalties = comma_floats(args.clash_penalties)

    if args.num_folds < 2 or args.num_folds > len(baseline):
        raise ValueError("num-folds must be between 2 and number of complexes")
    rng = np.random.default_rng(args.fold_seed)
    fold_assignment = np.empty(len(baseline), dtype=int)
    permutation = rng.permutation(len(baseline))
    fold_assignment[permutation] = np.arange(len(baseline)) % args.num_folds

    rows = []
    configuration_id = 0
    for density_weight in density_weights:
        for anchor_penalty in anchor_penalties:
            for minimum_ncc_gain in minimum_ncc_gains:
                for maximum_anchor_rmsd in maximum_anchor_rmsds:
                    for coarse_minimum_ncc_gain in coarse_minimum_ncc_gains:
                        for coarse_penalty in coarse_penalties:
                            for clash_penalty in clash_penalties:
                                configuration_id += 1
                                selected_indices = select_indices(
                                    candidates, slices,
                                    density_weight=density_weight,
                                    anchor_penalty=anchor_penalty,
                                    minimum_ncc_gain=minimum_ncc_gain,
                                    maximum_anchor_rmsd_A=maximum_anchor_rmsd,
                                    maximum_centroid_shift_A=args.maximum_centroid_shift,
                                    minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
                                    coarse_orientation_penalty=coarse_penalty,
                                    clash_penalty=clash_penalty,
                                    coarse_rescue=args.enable_coarse_rescue,
                                    coarse_minimum_ncc_gain=(
                                        0.005 if not args.enable_coarse_rescue
                                        else coarse_minimum_ncc_gain
                                    ),
                                    coarse_maximum_centroid_shift_A=(
                                        args.coarse_maximum_centroid_shift
                                    ),
                                )
                                selected = candidates.iloc[selected_indices]
                                summary, paired = paired_statistics(baseline, selected)
                                stable = fold_metrics(paired, fold_assignment)
                                rows.append({
                                    "configuration_id": configuration_id,
                                    "density_weight": density_weight,
                                    "anchor_penalty": anchor_penalty,
                                    "minimum_ncc_gain": minimum_ncc_gain,
                                    "maximum_anchor_rmsd_A": maximum_anchor_rmsd,
                                    "coarse_rescue": args.enable_coarse_rescue,
                                    "coarse_minimum_ncc_gain": (
                                        coarse_minimum_ncc_gain
                                        if args.enable_coarse_rescue else None
                                    ),
                                    "coarse_orientation_penalty": coarse_penalty,
                                    "clash_penalty": clash_penalty,
                                    **summary,
                                    **stable,
                                })

    sweep = pd.DataFrame(rows).sort_values(
        [
            "top1_lt2A_pct", "net_gained_lt2A", "lost_lt2A",
            "median_top1_rmsd_A", "mean_top1_rmsd_A",
            "std_fold_top1_lt2A_pct", "configuration_id",
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    sweep.to_csv(output / "sweep.csv", index=False)
    sweep.head(args.top_k).to_csv(output / "top_configurations.csv", index=False)

    best = sweep.iloc[0].to_dict()
    # Apply the frozen best validation configuration to every complex.  When
    # a holdout is supplied this is the first point at which those complexes
    # participate in selection/evaluation.
    all_slices = group_slices(all_candidates["row_index"].to_numpy(dtype=int))
    selected_indices = select_indices(
        all_candidates, all_slices,
        density_weight=float(best["density_weight"]),
        anchor_penalty=float(best["anchor_penalty"]),
        minimum_ncc_gain=float(best["minimum_ncc_gain"]),
        maximum_anchor_rmsd_A=float(best["maximum_anchor_rmsd_A"]),
        maximum_centroid_shift_A=args.maximum_centroid_shift,
        minimum_in_bounds_fraction=args.minimum_in_bounds_fraction,
        coarse_orientation_penalty=float(best["coarse_orientation_penalty"]),
        clash_penalty=float(best["clash_penalty"]),
        coarse_rescue=args.enable_coarse_rescue,
        coarse_minimum_ncc_gain=(
            float(best["coarse_minimum_ncc_gain"])
            if args.enable_coarse_rescue else 0.005
        ),
        coarse_maximum_centroid_shift_A=args.coarse_maximum_centroid_shift,
    )
    selected = all_candidates.iloc[selected_indices].copy()
    full_summary, paired = paired_statistics(all_baseline, selected)
    tuning_selected = selected[
        selected["row_index"].astype(int).isin(tuning_indices)
    ]
    tuning_summary, tuning_paired = paired_statistics(baseline, tuning_selected)
    holdout_summary = None
    holdout_paired = None
    if holdout_names:
        holdout_baseline = all_baseline[
            all_baseline["complex_name"].astype(str).isin(holdout_names)
        ].copy().reset_index(drop=True)
        holdout_indices = set(holdout_baseline["row_index"].astype(int))
        holdout_selected = selected[
            selected["row_index"].astype(int).isin(holdout_indices)
        ]
        holdout_summary, holdout_paired = paired_statistics(
            holdout_baseline, holdout_selected
        )
    confidence = selected["confidence_z"].to_numpy(dtype=float)
    density = selected["expanded_density_z"].to_numpy(dtype=float)
    anchor = selected["anchor_rmsd_A"].to_numpy(dtype=float)
    coarse = selected["move_type"].astype(str).eq("coarse_orientation").to_numpy(dtype=float)
    clash_count = selected.get("severe_clash_count", pd.Series(np.zeros(len(selected)))).fillna(0).to_numpy(float)
    parent_clash = selected.get(
        "parent_severe_clash_count", pd.Series(np.zeros(len(selected)))
    ).to_numpy(float)
    parent_clash = np.where(np.isfinite(parent_clash), parent_clash, clash_count)
    anchor_cost = float(best["anchor_penalty"]) * anchor
    if args.enable_coarse_rescue:
        anchor_cost = anchor_cost * (1.0 - coarse)
    score_values = (
        confidence
        + float(best["density_weight"]) * density
        - anchor_cost
        - float(best["coarse_orientation_penalty"]) * coarse
        - float(best["clash_penalty"]) * np.log1p(np.maximum(clash_count - parent_clash, 0))
    )
    selected["selection_score"] = score_values
    selected.to_csv(output / "best_selected_candidates.csv", index=False)
    paired.to_csv(output / "best_paired_complexes.csv", index=False)
    tuning_paired.to_csv(output / "best_paired_tuning.csv", index=False)
    if holdout_paired is not None:
        holdout_paired.to_csv(output / "best_paired_holdout.csv", index=False)
    materialize_best(output, selected, score_values)

    best_config = {
        "density_weight": float(best["density_weight"]),
        "anchor_penalty": float(best["anchor_penalty"]),
        "minimum_ncc_gain": float(best["minimum_ncc_gain"]),
        "maximum_anchor_rmsd_A": (
            None if math.isinf(float(best["maximum_anchor_rmsd_A"]))
            else float(best["maximum_anchor_rmsd_A"])
        ),
        "coarse_orientation_penalty": float(best["coarse_orientation_penalty"]),
        "coarse_rescue": bool(args.enable_coarse_rescue),
        "coarse_minimum_ncc_gain": (
            float(best["coarse_minimum_ncc_gain"])
            if args.enable_coarse_rescue else None
        ),
        "coarse_maximum_centroid_shift_A": (
            args.coarse_maximum_centroid_shift
        ),
        "clash_penalty": float(best["clash_penalty"]),
        "maximum_centroid_shift_A": args.maximum_centroid_shift,
        "minimum_in_bounds_fraction": args.minimum_in_bounds_fraction,
        "num_tuning_complexes": int(len(baseline)),
        "num_holdout_complexes": int(len(holdout_names)),
        "tuning_metrics": tuning_summary,
        "holdout_metrics": holdout_summary,
        "all_validation_metrics": full_summary,
        "uses_ground_truth_for_hyperparameter_selection": True,
        "confidence_mode": (
            "recomputed_s1" if rescored_path is not None else "inherited_s0"
        ),
        "rescored_confidence": (
            None if rescored_path is None else str(rescored_path)
        ),
        "warning": "Freeze this configuration before independent test evaluation.",
    }
    (output / "best_config.json").write_text(
        json.dumps(best_config, indent=2) + "\n"
    )
    print("Top validation configurations:")
    print(sweep.head(min(10, len(sweep))).to_string(index=False))
    print(json.dumps(best_config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
