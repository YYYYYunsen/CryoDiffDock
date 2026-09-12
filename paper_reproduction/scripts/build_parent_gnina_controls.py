#!/usr/bin/env python3
"""Build the unique direct-parent pose table needed by Attempt 1 GNINA scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_rankings(search_root: Path) -> pd.DataFrame:
    paths = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not paths:
        raise FileNotFoundError(f"No expanded_pool rankings under {search_root}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def build_parent_controls(
    basin: pd.DataFrame, rankings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_basin = {"row_index", "candidate_role", "parent_id"}
    required_rankings = {"row_index", "complex_name", "candidate_id", "move_type"}
    missing_basin = sorted(required_basin - set(basin.columns))
    missing_rankings = sorted(required_rankings - set(rankings.columns))
    if missing_basin:
        raise ValueError(f"Basin scores are missing columns: {missing_basin}")
    if missing_rankings:
        raise ValueError(f"Expanded rankings are missing columns: {missing_rankings}")

    parents = (
        basin[basin["candidate_role"].astype(str).eq("coarse")]
        [["row_index", "parent_id"]]
        .dropna()
        .rename(columns={"parent_id": "candidate_id"})
        .drop_duplicates(["row_index", "candidate_id"])
    )
    parents["row_index"] = parents["row_index"].astype(int)
    parents["candidate_id"] = parents["candidate_id"].astype(str)

    lookup = rankings[
        ["row_index", "complex_name", "candidate_id", "move_type"]
    ].drop_duplicates(["row_index", "candidate_id"])
    controls = parents.merge(
        lookup,
        on=["row_index", "candidate_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    failures = controls[controls["_merge"].ne("both")][
        ["row_index", "candidate_id"]
    ].copy()
    controls = controls[controls["_merge"].eq("both")].drop(columns="_merge")
    return controls.sort_values(
        ["row_index", "candidate_id"], kind="stable"
    ).reset_index(drop=True), failures.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin-scores", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    basin = pd.read_csv(args.basin_scores.expanduser().resolve())
    rankings = load_rankings(args.search_root.expanduser().resolve())
    controls, failures = build_parent_controls(basin, rankings)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    controls.to_csv(output, index=False)
    failure_path = output.with_suffix(".failures.csv")
    failures.to_csv(failure_path, index=False)
    print(
        f"PARENT_GNINA_CONTROLS: {len(controls)} parents across "
        f"{controls['row_index'].nunique()} complexes -> {output}"
    )
    if len(failures):
        print(f"Missing parents: {len(failures)} -> {failure_path}")
        if args.fail_on_error:
            raise RuntimeError(f"Missing {len(failures)} parent poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
