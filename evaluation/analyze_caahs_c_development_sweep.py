#!/usr/bin/env python3
"""Analyze CAAHS-C development variants without native ligand outcomes."""

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
        config["variants"]
        if config.get("status") == "development_sweep"
        else [config["frozen_configuration"]]
    )
    rows = []
    for variant in variants:
        variant_name = variant["name"]
        case_root = args.sweep_root / variant_name / "cases"
        paths = sorted(case_root.glob("*/summary.json"))
        if len(paths) != args.expected_cases:
            raise ValueError(f"{variant_name}: expected {args.expected_cases}, found {len(paths)}")
        for path in paths:
            item = json.loads(path.read_text())
            before = int(item["baseline_relative_vdw_clash_count"])
            after = int(item["selected_relative_vdw_clash_count"])
            rows.append({
                "variant": variant_name,
                "complex_name": path.parent.name,
                "baseline_relative_vdw_clash_count": before,
                "selected_relative_vdw_clash_count": after,
                "baseline_minimum_relative_vdw_distance": item["baseline_minimum_relative_vdw_distance"],
                "selected_minimum_relative_vdw_distance": item["selected_minimum_relative_vdw_distance"],
                "baseline_severe_clash_count": item["baseline_severe_clash_count"],
                "selected_severe_clash_count": item["selected_severe_clash_count"],
                "baseline_pass": before == 0,
                "selected_pass": after == 0,
                "new_rescue": before > 0 and after == 0,
                "baseline_pass_lost": before == 0 and after > 0,
                "selected_method": item["selected_method"],
                "ligand_displacement_A": item["ligand_displacement_A"],
                "sidechain_displacement_A": item["sidechain_displacement_A"],
                "delta_ncc": item["delta_ncc"],
                "interaction_geometry_change": item["interaction_geometry_change"],
                "runtime_seconds": item["runtime_seconds"],
                "selected_pose_path": str(path.parent / "selected_pose.sdf"),
                "selected_protein_path": str(path.parent / "selected_protein.pdb"),
            })
    frame = pd.DataFrame(rows)
    summaries = []
    for variant_name, group in frame.groupby("variant", sort=False):
        summaries.append({
            "variant": variant_name,
            "cases_n": len(group),
            "baseline_pass_n": int(group.baseline_pass.sum()),
            "selected_pass_n": int(group.selected_pass.sum()),
            "new_rescues_n": int(group.new_rescue.sum()),
            "baseline_passes_lost_n": int(group.baseline_pass_lost.sum()),
            "baseline_residual_relative_clashes": int(group.baseline_relative_vdw_clash_count.sum()),
            "selected_residual_relative_clashes": int(group.selected_relative_vdw_clash_count.sum()),
            "selected_cases_n": int(group.selected_method.ne("baseline_retained").sum()),
            "maximum_ligand_displacement_A": float(group.ligand_displacement_A.max()),
            "worst_delta_ncc": float(group.delta_ncc.min()),
            "worst_interaction_geometry_change": float(group.interaction_geometry_change.min()),
            "runtime_hours": float(group.runtime_seconds.sum() / 3600.0),
            "uses_reference_or_native_rmsd": False,
        })
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "per_case_caahs_c_development_sweep.csv", index=False)
    summary.to_csv(args.output_dir / "caahs_c_development_sweep_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"method": "CAAHS-C", "variants": summaries}, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(f"CAAHS_C_DEVELOPMENT_SWEEP_ANALYSIS_OK: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
