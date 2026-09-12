"""Helpers for deployment-safe confidence rescoring of discrete candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def boolean_series(series: pd.Series) -> pd.Series:
    """Parse booleans written by pandas CSV without Python truthiness traps."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def select_rescoring_shortlist(
    candidates: pd.DataFrame,
    *,
    minimum_ncc_gain: float = 0.005,
    maximum_anchor_rmsd_A: float = 2.0,
    maximum_centroid_shift_A: float = 1.5,
    minimum_in_bounds_fraction: float = 0.95,
    torsion_top_k_per_parent: int = 0,
) -> pd.DataFrame:
    """Keep every original and every variant eligible for the frozen S0 rule.

    ``torsion_top_k_per_parent`` optionally adds near-miss torsion variants for
    the S1 torsion-coverage experiment.  It is deliberately off by default so
    the first S1 comparison changes only the confidence calculation.
    """

    required = {
        "row_index", "candidate_id", "parent_id", "is_original", "move_type",
        "ncc_gain", "anchor_rmsd_A", "centroid_shift_A", "in_bounds_fraction",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    if torsion_top_k_per_parent < 0:
        raise ValueError("torsion_top_k_per_parent must be non-negative")

    original = boolean_series(candidates["is_original"])
    eligible = (
        (candidates["ncc_gain"].fillna(0).astype(float) >= minimum_ncc_gain)
        & (candidates["anchor_rmsd_A"].astype(float) <= maximum_anchor_rmsd_A)
        & (candidates["centroid_shift_A"].astype(float) <= maximum_centroid_shift_A)
        & (candidates["in_bounds_fraction"].astype(float) >= minimum_in_bounds_fraction)
    )
    keep = original | eligible

    if torsion_top_k_per_parent:
        torsions = candidates.loc[
            (~original) & candidates["move_type"].astype(str).eq("torsion")
        ].copy()
        torsions = torsions.sort_values(
            ["row_index", "parent_id", "ncc_gain", "anchor_rmsd_A", "candidate_id"],
            ascending=[True, True, False, True, True],
            kind="stable",
        )
        extra_indices = torsions.groupby(
            ["row_index", "parent_id"], sort=False
        ).head(torsion_top_k_per_parent).index
        keep.loc[extra_indices] = True

    result = candidates.loc[keep].copy()
    result["s1_s0_eligible"] = eligible.loc[result.index].to_numpy(dtype=bool)
    result["s1_extra_torsion"] = (
        ~original.loc[result.index].to_numpy(dtype=bool)
        & ~result["s1_s0_eligible"].to_numpy(dtype=bool)
        & result["move_type"].astype(str).eq("torsion").to_numpy(dtype=bool)
    )
    return result.reset_index(drop=True)


def calibrate_recomputed_confidence(candidates: pd.DataFrame) -> pd.DataFrame:
    """Map recomputed logits onto the stored original-confidence scale.

    The paired original poses provide a deployment-available, GT-free
    calibration set.  A per-complex affine map removes graph-reconstruction
    shifts/scales.  Original candidates then retain their exact inherited S0
    z-scores; only new variants receive calibrated recomputed confidence.
    Consequently an S0->S1 comparison cannot be driven by reshuffling the
    original pool.
    """

    required = {
        "row_index", "is_original", "confidence", "confidence_z",
        "confidence_recomputed",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Rescored table is missing columns: {missing}")

    table = candidates.copy()
    table["_original"] = boolean_series(table["is_original"])
    output = np.empty(len(table), dtype=np.float64)
    calibrated_raw = np.empty(len(table), dtype=np.float64)
    slopes = np.empty(len(table), dtype=np.float64)
    intercepts = np.empty(len(table), dtype=np.float64)
    for _, indices in table.groupby("row_index", sort=False).groups.items():
        group = table.loc[indices]
        originals = group.loc[group["_original"]]
        old_values = originals["confidence"].to_numpy(dtype=float)
        new_values = originals["confidence_recomputed"].to_numpy(dtype=float)
        finite = np.isfinite(old_values) & np.isfinite(new_values)
        old_values, new_values = old_values[finite], new_values[finite]
        if not len(old_values):
            raise ValueError("Each complex requires at least one rescored original")
        old_mean = float(old_values.mean())
        old_std = float(old_values.std())
        new_mean = float(new_values.mean())
        new_variance = float(np.mean((new_values - new_mean) ** 2))
        if new_variance > 1e-12:
            slope = float(np.mean(
                (new_values - new_mean) * (old_values - old_mean)
            ) / new_variance)
            # Confidence ordering must not be inverted by a noisy fit.
            if not np.isfinite(slope) or slope <= 0:
                slope = 1.0
        else:
            slope = 1.0
        intercept = old_mean - slope * new_mean
        positions = table.index.get_indexer(indices)
        mapped = (
            slope * group["confidence_recomputed"].to_numpy(dtype=float)
            + intercept
        )
        calibrated_raw[positions] = mapped
        if old_std > 1e-12:
            group_z = (mapped - old_mean) / old_std
        else:
            group_z = np.zeros_like(mapped)
        # Exact S0 preservation for originals is intentional.  It isolates
        # the scientific effect of assigning real confidence to variants.
        original_mask = group["_original"].to_numpy(dtype=bool)
        group_z[original_mask] = group.loc[
            group["_original"], "confidence_z"
        ].to_numpy(dtype=float)
        output[positions] = group_z
        slopes[positions] = slope
        intercepts[positions] = intercept
    table["confidence_recomputed_calibrated"] = calibrated_raw
    table["confidence_recomputed_z"] = output
    table["confidence_calibration_slope"] = slopes
    table["confidence_calibration_intercept"] = intercepts
    return table.drop(columns="_original")
