#!/usr/bin/env python3
"""Combine frozen CAAHS-A results with CAAHS-A+SC rescue outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-final-analysis", required=True, type=Path)
    parser.add_argument("--sc-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    baseline_path = args.a_final_analysis / "per_case_continuous_refinement.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Missing A-final per-case table: {baseline_path}. "
            "The side-chain run must complete before analysis."
        )
    baseline = pd.read_csv(baseline_path)
    if len(baseline) != args.expected_cases:
        raise ValueError(f"Expected {args.expected_cases}, found {len(baseline)}")
    rows = []
    for row in baseline.itertuples(index=False):
        name = str(row.complex_name)
        before = int(row.final_post_severe_clash_count)
        summary_path = args.sc_root / "cases" / name / "summary.json"
        after = before
        method = "caahs_a_retained"
        protein_path = None
        sc_status = "not_run_already_resolved"
        interaction_change = None
        sidechain_displacement = 0.0
        runtime = 0.0
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            proposed = int(summary["selected_severe_clash_count"])
            if proposed > before:
                raise AssertionError(f"{name}: side-chain rescue worsened clash count")
            after = proposed
            sc_status = str(summary["status"])
            interaction_change = summary.get("interaction_geometry_change")
            sidechain_displacement = float(summary["sidechain_displacement_A"])
            runtime = float(summary["runtime_seconds"])
            if after < before:
                method = "caahs_a_sc"
                protein_path = str(summary_path.parent / "selected_protein.pdb")
        rows.append({
            "complex_name": name,
            "a_final_post_severe_clash_count": before,
            "sc_final_post_severe_clash_count": after,
            "additional_clash_reduction": before - after,
            "newly_rescued_by_sc": before > 0 and after == 0,
            "selected_method": method,
            "sc_status": sc_status,
            "sidechain_displacement_A": sidechain_displacement,
            "interaction_geometry_change": interaction_change,
            "sc_runtime_seconds": runtime,
            "selected_pose_path": str(row.selected_pose_path),
            "selected_protein_path": protein_path,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "cases_n": len(frame),
        "a_clash_free_n": int(frame["a_final_post_severe_clash_count"].eq(0).sum()),
        "a_sc_clash_free_n": int(frame["sc_final_post_severe_clash_count"].eq(0).sum()),
        "newly_rescued_by_sc_n": int(frame["newly_rescued_by_sc"].sum()),
        "a_residual_clashes": int(frame["a_final_post_severe_clash_count"].sum()),
        "a_sc_residual_clashes": int(frame["sc_final_post_severe_clash_count"].sum()),
        "successes_lost_n": 0,
        "sidechain_selected_n": int(frame["selected_method"].eq("caahs_a_sc").sum()),
        "runtime_hours": float(frame["sc_runtime_seconds"].sum() / 3600.0),
        "ligand_coordinates_changed": False,
        "reference_or_native_rmsd_used": False,
    }
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "per_case_caahs_a_sidechain_rescue.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CAAHS_A_SIDECHAIN_RESCUE_ANALYSIS_OK: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
