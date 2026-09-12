#!/usr/bin/env python3
"""Apply a frozen discrete-search selector without reading reference poses.

Unlike ``evaluation/sweep_discrete_search_ranking.py``, this deployment stage
does not load ``metrics_expanded_pool`` and cannot inspect ligand RMSD.  It
only reads the expanded candidate metadata and a previously frozen JSON
configuration, then materializes one selected pose per complex.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_discrete_search_ranking import (  # noqa: E402
    boolean_array,
    group_slices,
    materialize_best,
    select_indices,
)


REQUIRED_COLUMNS = {
    "row_index", "complex_name", "candidate_id", "materialized_path",
    "protein_path", "is_original", "confidence_z", "expanded_density_z",
    "anchor_rmsd_A", "centroid_shift_A", "ncc_gain",
    "in_bounds_fraction", "move_type",
}


def load_candidates(search_root: Path) -> pd.DataFrame:
    paths = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not paths:
        raise FileNotFoundError(f"No expanded_pool rankings under {search_root}")
    candidates = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    missing = sorted(REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")
    if candidates.duplicated(["row_index", "candidate_id"]).any():
        raise ValueError("Expanded rankings contain duplicate candidate identifiers")
    candidates["is_original_bool"] = boolean_array(candidates["is_original"])
    return candidates.sort_values(
        ["row_index", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def selection_scores(selected: pd.DataFrame, config: dict) -> np.ndarray:
    confidence = selected["confidence_z"].to_numpy(float)
    density = selected["expanded_density_z"].to_numpy(float)
    anchor = selected["anchor_rmsd_A"].to_numpy(float)
    coarse = selected["move_type"].astype(str).eq("coarse_orientation").to_numpy(float)
    clashes = selected.get(
        "severe_clash_count", pd.Series(np.zeros(len(selected)))
    ).fillna(0).to_numpy(float)
    parent_clashes = selected.get(
        "parent_severe_clash_count", pd.Series(np.zeros(len(selected)))
    ).to_numpy(float)
    parent_clashes = np.where(np.isfinite(parent_clashes), parent_clashes, clashes)
    anchor_cost = float(config["anchor_penalty"]) * anchor
    if bool(config.get("coarse_rescue", False)):
        anchor_cost *= 1.0 - coarse
    return (
        confidence
        + float(config["density_weight"]) * density
        - anchor_cost
        - float(config["coarse_orientation_penalty"]) * coarse
        - float(config.get("clash_penalty", 0.0))
        * np.log1p(np.maximum(clashes - parent_clashes, 0.0))
    )


def apply_frozen(candidates: pd.DataFrame, config: dict) -> pd.DataFrame:
    maximum_anchor = config.get("maximum_anchor_rmsd_A")
    if maximum_anchor is None:
        maximum_anchor = math.inf
    slices = group_slices(candidates["row_index"].to_numpy(int))
    indices = select_indices(
        candidates,
        slices,
        density_weight=float(config["density_weight"]),
        anchor_penalty=float(config["anchor_penalty"]),
        minimum_ncc_gain=float(config["minimum_ncc_gain"]),
        maximum_anchor_rmsd_A=float(maximum_anchor),
        maximum_centroid_shift_A=float(config["maximum_centroid_shift_A"]),
        minimum_in_bounds_fraction=float(config["minimum_in_bounds_fraction"]),
        coarse_orientation_penalty=float(config["coarse_orientation_penalty"]),
        clash_penalty=float(config.get("clash_penalty", 0.0)),
        coarse_rescue=bool(config.get("coarse_rescue", False)),
        coarse_minimum_ncc_gain=float(
            config.get("coarse_minimum_ncc_gain") or 0.005
        ),
        coarse_maximum_centroid_shift_A=float(
            config.get("coarse_maximum_centroid_shift_A", 0.25)
        ),
    )
    selected = candidates.iloc[indices].copy().reset_index(drop=True)
    selected["selection_score"] = selection_scores(selected, config)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    config_path = args.frozen_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    candidates = load_candidates(search_root)
    selected = apply_frozen(candidates, config)
    selected.to_csv(output / "best_selected_candidates.csv", index=False)
    materialize_best(output, selected, selected["selection_score"].to_numpy(float))
    summary = {
        "num_complexes": int(selected["row_index"].nunique()),
        "num_candidates": int(len(candidates)),
        "num_selected_variants": int((~selected["is_original_bool"]).sum()),
        "selected_move_type_counts": {
            str(key): int(value)
            for key, value in selected["move_type"].value_counts().items()
        },
        "search_root": str(search_root),
        "frozen_config": str(config_path),
        "uses_ground_truth": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
