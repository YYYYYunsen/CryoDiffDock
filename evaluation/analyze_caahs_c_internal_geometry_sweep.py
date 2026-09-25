#!/usr/bin/env python3
"""Analyze restrained CAAHS-C internal-geometry refinement variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads(args.config_json.read_text())
    variants = (
        config["variants"] if config.get("status") == "development_sweep"
        else [config["frozen_configuration"]]
    )
    rows = []
    for variant in variants:
        name = variant["name"]
        paths = sorted((args.sweep_root / name / "cases").glob("*/summary.json"))
        if len(paths) != args.expected_cases:
            raise ValueError(f"{name}: expected {args.expected_cases}, found {len(paths)}")
        for path in paths:
            item = json.loads(path.read_text())
            before = int(item["baseline_internal_clash_count"])
            after = int(item["selected_internal_clash_count"])
            rows.append({
                "variant": name,
                "complex_name": path.parent.name,
                "baseline_internal_clash_count": before,
                "selected_internal_clash_count": after,
                "baseline_internal_pass": before == 0,
                "selected_internal_pass": after == 0,
                "new_internal_rescue": before > 0 and after == 0,
                "baseline_pass_lost": before == 0 and after > 0,
                "selected_method": item["selected_method"],
                "ligand_internal_rmsd_A": item["ligand_internal_rmsd_A"],
                "maximum_atom_displacement_A": item["maximum_atom_displacement_A"],
                "protein_relative_vdw_clash_count": item[
                    "protein_relative_vdw_clash_count"
                ],
                "baseline_severe_clash_count": item.get(
                    "baseline_severe_clash_count"
                ),
                "selected_severe_clash_count": item.get(
                    "selected_severe_clash_count"
                ),
                "protein_minimum_relative_vdw_distance": item[
                    "protein_minimum_relative_vdw_distance"
                ],
                "selected_bond_outlier_count": item["selected_bond_outlier_count"],
                "selected_angle_outlier_count": item["selected_angle_outlier_count"],
                "delta_ncc": item["delta_ncc"],
                "interaction_geometry_change": item["interaction_geometry_change"],
                "runtime_seconds": item["runtime_seconds"],
                "selected_pose_path": str(path.parent / "selected_pose.sdf"),
                "selected_protein_path": str(path.parent / "selected_protein.pdb"),
            })
    frame = pd.DataFrame(rows)
    summaries = []
    for name, group in frame.groupby("variant", sort=False):
        summaries.append({
            "variant": name,
            "cases_n": len(group),
            "baseline_internal_pass_n": int(group.baseline_internal_pass.sum()),
            "selected_internal_pass_n": int(group.selected_internal_pass.sum()),
            "new_internal_rescues_n": int(group.new_internal_rescue.sum()),
            "baseline_passes_lost_n": int(group.baseline_pass_lost.sum()),
            "baseline_internal_clashes": int(group.baseline_internal_clash_count.sum()),
            "selected_internal_clashes": int(group.selected_internal_clash_count.sum()),
            "selected_cases_n": int(
                group.selected_method.eq("restrained_internal_geometry").sum()
            ),
            "maximum_internal_rmsd_A": float(group.ligand_internal_rmsd_A.max()),
            "maximum_atom_displacement_A": float(
                group.maximum_atom_displacement_A.max()
            ),
            "worst_delta_ncc": float(group.delta_ncc.min()),
            "worst_interaction_geometry_change": float(
                group.interaction_geometry_change.min()
            ),
            "runtime_hours": float(group.runtime_seconds.sum() / 3600.0),
            "uses_reference_or_native_rmsd": False,
        })
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "per_case_internal_geometry_refinement.csv", index=False)
    summary.to_csv(args.output_dir / "internal_geometry_sweep_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"method": config["method"], "variants": summaries}, indent=2)
        + "\n"
    )
    print(summary.to_string(index=False))
    print(f"CAAHS_C_INTERNAL_GEOMETRY_ANALYSIS_OK: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
