#!/usr/bin/env python3
"""Apply the frozen parent-relative GNINA gate without reference/RMSD data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_gnina_relative_coarse_gate import materialize  # noqa: E402
from evaluation.sweep_parent_relative_gnina_gate import (  # noqa: E402
    select_rescues,
    successful_gnina,
)


GNINA_COLUMNS = [
    "row_index", "candidate_id", "gnina_affinity_kcal_mol",
    "gnina_cnnscore", "gnina_cnnaffinity",
]


def prepare_deployment_scores(
    basin_path: Path,
    candidate_gnina_path: Path,
    parent_gnina_path: Path,
) -> pd.DataFrame:
    basin = pd.read_csv(basin_path.expanduser().resolve())
    candidate_gnina = successful_gnina(candidate_gnina_path)
    parent_gnina = successful_gnina(parent_gnina_path)
    scores = basin.merge(
        candidate_gnina[GNINA_COLUMNS],
        on=["row_index", "candidate_id"], how="inner", validate="one_to_one",
    )
    if len(scores) != len(basin):
        raise ValueError(
            f"Basin/candidate-GNINA pairing failed: {len(scores)} / {len(basin)}"
        )

    s0 = scores[scores["candidate_role"].astype(str).eq("s0")][
        GNINA_COLUMNS
    ].rename(columns={
        "candidate_id": "s0_candidate_id",
        "gnina_affinity_kcal_mol": "s0_gnina_affinity_kcal_mol",
        "gnina_cnnscore": "s0_gnina_cnnscore",
        "gnina_cnnaffinity": "s0_gnina_cnnaffinity",
    })
    if len(s0) != scores["row_index"].nunique():
        raise ValueError("Expected exactly one S0 GNINA control per complex")
    scores = scores.merge(s0, on="row_index", how="left", validate="many_to_one")
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
    parents = parent_gnina[GNINA_COLUMNS].rename(columns={
        "candidate_id": "parent_id",
        "gnina_affinity_kcal_mol": "parent_gnina_affinity_kcal_mol",
        "gnina_cnnscore": "parent_gnina_cnnscore",
        "gnina_cnnaffinity": "parent_gnina_cnnaffinity",
    })
    coarse = coarse.merge(
        parents, on=["row_index", "parent_id"], how="inner", validate="many_to_one"
    )
    expected_coarse = int(scores["candidate_role"].astype(str).eq("coarse").sum())
    if len(coarse) != expected_coarse:
        raise ValueError(
            f"Coarse/parent-GNINA pairing failed: {len(coarse)} / {expected_coarse}"
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
    fallback = scores[scores["candidate_role"].astype(str).eq("s0")].copy()
    fallback["delta_parent_cnnscore"] = 0.0
    fallback["delta_parent_cnnaffinity"] = 0.0
    fallback["delta_parent_affinity_improvement"] = 0.0
    return pd.concat([fallback, coarse], ignore_index=True, sort=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-scores", required=True, type=Path)
    parser.add_argument("--candidate-gnina-scores", required=True, type=Path)
    parser.add_argument("--parent-gnina-scores", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    s0_config = json.loads(args.s0_config.expanduser().resolve().read_text())
    config_path = args.frozen_config.expanduser().resolve()
    config = json.loads(config_path.read_text())
    scores = prepare_deployment_scores(
        args.basin_scores, args.candidate_gnina_scores, args.parent_gnina_scores
    )
    selected = select_rescues(
        scores,
        robust_column=str(config["robust_column"]),
        density_weight=float(s0_config["density_weight"]),
        parent_cnnscore_weight=float(config["parent_cnnscore_weight"]),
        minimum_score_advantage=float(config["minimum_score_advantage"]),
        minimum_ncc_gain=float(config["minimum_ncc_gain"]),
        minimum_support_fraction=float(config["minimum_support_fraction"]),
        minimum_cnnscore_delta_vs_s0=float(
            config["minimum_cnnscore_delta_vs_s0"]
        ),
        minimum_parent_cnnscore_delta=float(
            config["minimum_parent_cnnscore_delta"]
        ),
        minimum_in_bounds_fraction=float(config["minimum_in_bounds_fraction"]),
        maximum_centroid_shift_A=float(config["maximum_centroid_shift_A"]),
    )
    selected.to_csv(output / "best_selected_candidates.csv", index=False)
    materialize(output, selected)
    summary = {
        "num_complexes": int(selected["row_index"].nunique()),
        "num_candidates_considered": int(len(scores)),
        "selected_coarse_count": int(
            selected["selected_role"].astype(str).eq("coarse").sum()
        ),
        "selected_s0_count": int(
            selected["selected_role"].astype(str).eq("s0").sum()
        ),
        "frozen_config": str(config_path),
        "uses_ground_truth": False,
        "coordinates_modified_by_gnina": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
