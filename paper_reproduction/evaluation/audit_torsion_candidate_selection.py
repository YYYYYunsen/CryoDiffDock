#!/usr/bin/env python3
"""Measure whether torsion candidates add coverage and whether scores find it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.sweep_discrete_search_ranking import load_expanded_candidates


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    valid = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rescored-confidence", type=Path, default=None)
    parser.add_argument(
        "--selected-candidates", type=Path, default=None,
        help="Optional best_selected_candidates.csv from an S0/S1 ranking run",
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rescored = None if args.rescored_confidence is None else args.rescored_confidence.expanduser().resolve()
    candidates = load_expanded_candidates(args.search_root.expanduser().resolve(), rescored)
    candidates["is_original_bool"] = candidates["is_original_bool"].astype(bool)

    rows = []
    for row_index, group in candidates.groupby("row_index", sort=True):
        originals = group[group["is_original_bool"]]
        torsions = group[group["move_type"].astype(str).eq("torsion")]
        all_variants = group[~group["is_original_bool"]]
        rows.append({
            "row_index": int(row_index),
            "complex_name": str(group.iloc[0]["complex_name"]),
            "best_original_rmsd_A": float(originals["ligand_rmsd_A"].min()),
            "best_torsion_rmsd_A": (
                float(torsions["ligand_rmsd_A"].min()) if len(torsions) else np.nan
            ),
            "best_expanded_rmsd_A": float(group["ligand_rmsd_A"].min()),
            "num_torsion_candidates": int(len(torsions)),
            "torsion_confidence_rmsd_spearman": (
                safe_spearman(torsions["confidence_z"], -torsions["ligand_rmsd_A"])
                if len(torsions) else np.nan
            ),
            "torsion_ncc_rmsd_spearman": (
                safe_spearman(torsions["expanded_density_z"], -torsions["ligand_rmsd_A"])
                if len(torsions) else np.nan
            ),
            "best_variant_move_type": str(
                all_variants.loc[all_variants["ligand_rmsd_A"].idxmin(), "move_type"]
            ) if len(all_variants) else "none",
        })
    table = pd.DataFrame(rows)
    table["torsion_rescues_original_lt2"] = (
        (table["best_original_rmsd_A"] >= 2.0)
        & (table["best_torsion_rmsd_A"] < 2.0)
    )

    if args.selected_candidates is not None:
        selected = pd.read_csv(args.selected_candidates.expanduser().resolve())
        selected = selected[["row_index", "candidate_id", "move_type", "ligand_rmsd_A"]].rename(columns={
            "candidate_id": "selected_candidate_id",
            "move_type": "selected_move_type",
            "ligand_rmsd_A": "selected_rmsd_A",
        })
        table = table.merge(selected, on="row_index", how="left", validate="one_to_one")

    table.to_csv(output / "per_complex.csv", index=False)
    summary = {
        "num_complexes": int(len(table)),
        "complexes_with_torsion_candidates": int(table["best_torsion_rmsd_A"].notna().sum()),
        "torsion_rescue_count": int(table["torsion_rescues_original_lt2"].sum()),
        "torsion_rescue_pct": float(100 * table["torsion_rescues_original_lt2"].mean()),
        "median_best_original_rmsd_A": float(table["best_original_rmsd_A"].median()),
        "median_best_torsion_rmsd_A": float(table["best_torsion_rmsd_A"].median()),
        "median_torsion_confidence_rmsd_spearman": float(table["torsion_confidence_rmsd_spearman"].median()),
        "median_torsion_ncc_rmsd_spearman": float(table["torsion_ncc_rmsd_spearman"].median()),
        "best_variant_move_type_counts": table["best_variant_move_type"].value_counts().to_dict(),
        "uses_ground_truth": True,
    }
    if "selected_move_type" in table:
        summary["selected_move_type_counts"] = table["selected_move_type"].value_counts().to_dict()
        summary["unselected_torsion_rescues"] = int((
            table["torsion_rescues_original_lt2"]
            & ~table["selected_move_type"].astype(str).eq("torsion")
        ).sum())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
