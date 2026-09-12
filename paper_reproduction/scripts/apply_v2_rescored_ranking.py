#!/usr/bin/env python3
"""Apply recomputed candidate confidence to the frozen V2 finalist set.

The generation pass uses inherited root confidence because candidate graphs are
expensive.  ``rescore_discrete_candidates.py`` can then evaluate generated
poses in batches.  This script replaces only the confidence term in V2's final
score, preserving density, basin, anchor, and clash terms exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def load_materialized(search_root: Path) -> pd.DataFrame:
    paths = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not paths:
        raise FileNotFoundError(f"No expanded-pool rankings below {search_root}")
    table = pd.concat(
        [pd.read_csv(path, usecols=["row_index", "candidate_id", "materialized_path"])
         for path in paths],
        ignore_index=True,
    )
    return table.drop_duplicates(["row_index", "candidate_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--rescored-confidence", required=True, type=Path,
                        help="rescored_candidates.csv from rescore_discrete_candidates.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--require-relative-physics", action="store_true")
    args = parser.parse_args()

    search_root = args.search_root.expanduser().resolve()
    output = (
        search_root / "v2_rescored" if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(search_root / "candidates.csv", low_memory=False)
    rescored = pd.read_csv(args.rescored_confidence.expanduser().resolve())
    required_candidates = {
        "row_index", "complex_name", "candidate_id", "v2_finalist",
        "final_selection_score", "confidence_z", "protein_path",
        "relative_physics_pass", "is_original",
    }
    required_rescored = {
        "row_index", "candidate_id", "confidence_recomputed_z",
    }
    missing = sorted(required_candidates - set(candidates.columns))
    missing_rescored = sorted(required_rescored - set(rescored.columns))
    if missing or missing_rescored:
        raise ValueError(
            f"Missing candidate columns {missing}; rescored columns {missing_rescored}"
        )
    finalists = candidates[boolean_series(candidates["v2_finalist"])].copy()
    expected_finalists = len(finalists)
    finalists = finalists.merge(
        rescored[list(required_rescored)],
        on=["row_index", "candidate_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(finalists) != expected_finalists:
        raise ValueError(
            f"Only {len(finalists)} / {expected_finalists} V2 finalists have "
            "recomputed confidence. Run rescore_discrete_candidates.py with "
            "--minimum-ncc-gain 0 --maximum-anchor-rmsd inf and a sufficiently "
            "large centroid threshold."
        )
    if args.require_relative_physics:
        keep = boolean_series(finalists["relative_physics_pass"]) | boolean_series(
            finalists["is_original"]
        )
        finalists = finalists[keep].copy()
    finalists["v2_rescored_selection_score"] = (
        finalists["final_selection_score"].astype(float)
        - finalists["confidence_z"].astype(float)
        + finalists["confidence_recomputed_z"].astype(float)
    )
    materialized = load_materialized(search_root)
    finalists = finalists.merge(
        materialized, on=["row_index", "candidate_id"], how="inner",
        validate="one_to_one",
    )
    selected_rows = []
    for row_index, group in finalists.groupby("row_index", sort=True):
        ranked = group.sort_values(
            ["v2_rescored_selection_score", "candidate_id"],
            ascending=[False, True], kind="stable",
        )
        selected = ranked.iloc[0].to_dict()
        complex_name = str(selected["complex_name"])
        safe_name = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in complex_name
        )
        directory = output / f"index{int(row_index)}___{safe_name}"
        directory.mkdir(parents=True, exist_ok=True)
        score = float(selected["v2_rescored_selection_score"])
        destination = directory / f"rank1_confidence{score:.8f}.sdf"
        shutil.copy2(Path(str(selected["materialized_path"])), destination)
        protein = Path(str(selected["protein_path"]))
        if protein.is_file():
            shutil.copy2(
                protein,
                destination.with_name(f"{destination.stem}_protein.pdb"),
            )
        selected["materialized_rank"] = 1
        selected["materialized_path"] = str(destination)
        pd.DataFrame([selected]).to_csv(directory / "ranking.csv", index=False)
        selected_rows.append(selected)

    selected_table = pd.DataFrame(selected_rows)
    selected_table.to_csv(output / "selected.csv", index=False)
    summary = {
        "search_root": str(search_root),
        "rescored_confidence": str(args.rescored_confidence.expanduser().resolve()),
        "num_complexes": int(selected_table["row_index"].nunique()),
        "num_rescored_finalists": int(len(finalists)),
        "num_selected_generated": int((~boolean_series(selected_table["is_original"])).sum()),
        "require_relative_physics": bool(args.require_relative_physics),
        "score_update": (
            "final_selection_score - inherited_confidence_z "
            "+ recomputed_confidence_z"
        ),
        "uses_ground_truth": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
