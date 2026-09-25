#!/usr/bin/env python3
"""Summarize validation clash-repair sweeps and diagnose search bottlenecks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def failure_class(summary: dict) -> str:
    original = summary["original"]
    selected = summary["selected"]
    if int(original["severe_clash_count"]) == 0:
        return "no_action_clash_free"
    if int(selected["severe_clash_count"]) == 0:
        return "resolved_clash_free"
    oracle = summary.get("candidate_oracle", {})
    if not oracle.get("any_clash_free", False):
        return "search_space_failure"
    if not oracle.get("clash_free_with_displacement_gate", False):
        return "displacement_gate_limited"
    if not oracle.get("clash_free_with_both_final_gates", False):
        return "density_gate_limited"
    return "selector_failure"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=62)
    args = parser.parse_args()

    sweep_root = args.sweep_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config_dirs = sorted(
        path for path in sweep_root.iterdir()
        if path.is_dir() and (path / "cases").is_dir()
    )
    if not config_dirs:
        raise RuntimeError(f"No configuration directories under {sweep_root}")

    rows = []
    for config_dir in config_dirs:
        summaries = sorted((config_dir / "cases").glob("*/summary.json"))
        if len(summaries) != args.expected_cases:
            raise RuntimeError(
                f"{config_dir.name}: expected {args.expected_cases} summaries, "
                f"found {len(summaries)}"
            )
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            original, selected = summary["original"], summary["selected"]
            rows.append({
                "config": config_dir.name,
                "complex_name": summary_path.parent.name,
                "status": summary["status"],
                "failure_class": failure_class(summary),
                "pre_severe_clash_count": int(original["severe_clash_count"]),
                "post_severe_clash_count": int(selected["severe_clash_count"]),
                "severe_clash_reduction": int(original["severe_clash_count"])
                - int(selected["severe_clash_count"]),
                "displacement_A": float(selected["displacement_A"]),
                "delta_ncc": float(selected["ncc_change"]),
                "smooth_clash_energy_before": float(original["smooth_clash_energy"]),
                "smooth_clash_energy_after": float(selected["smooth_clash_energy"]),
                "candidate_count": int(summary["candidate_count"]),
                "density_scored_candidate_count": int(
                    summary.get("density_scored_candidate_count", summary["candidate_count"])
                ),
                "runtime_seconds": float(summary.get("runtime_seconds", np.nan)),
                **{
                    key: bool(value)
                    for key, value in summary.get("candidate_oracle", {}).items()
                },
                "selected_pose_path": str(summary_path.parent / "selected_pose.sdf"),
                "summary_path": str(summary_path),
            })
    cases = pd.DataFrame(rows)
    summary_rows = []
    for config, group in cases.groupby("config", sort=False):
        targets = group[group["pre_severe_clash_count"] > 0]
        before_total = int(targets["pre_severe_clash_count"].sum())
        after_total = int(targets["post_severe_clash_count"].sum())
        failure_counts = targets["failure_class"].value_counts()
        summary_rows.append({
            "config": config,
            "cases_n": len(group),
            "clashing_targets_n": len(targets),
            "overall_clash_free_n": int((group["post_severe_clash_count"] == 0).sum()),
            "overall_clash_free_pct": float(
                100.0 * (group["post_severe_clash_count"] == 0).mean()
            ),
            "targets_resolved_n": int((targets["post_severe_clash_count"] == 0).sum()),
            "target_resolution_pct": float(
                100.0 * (targets["post_severe_clash_count"] == 0).mean()
            ) if len(targets) else float("nan"),
            "total_clash_reduction_pct": float(
                100.0 * (1.0 - after_total / before_total)
            ) if before_total else 0.0,
            "search_space_failure_n": int(failure_counts.get("search_space_failure", 0)),
            "displacement_gate_limited_n": int(
                failure_counts.get("displacement_gate_limited", 0)
            ),
            "density_gate_limited_n": int(failure_counts.get("density_gate_limited", 0)),
            "selector_failure_n": int(failure_counts.get("selector_failure", 0)),
            "median_displacement_A": float(targets["displacement_A"].median()),
            "worst_delta_ncc": float(targets["delta_ncc"].min()),
            "median_runtime_seconds": float(group["runtime_seconds"].median()),
            "total_runtime_hours": float(group["runtime_seconds"].sum() / 3600.0),
            "median_candidate_count": float(targets["candidate_count"].median()),
        })
    configs = pd.DataFrame(summary_rows).sort_values(
        ["targets_resolved_n", "total_clash_reduction_pct", "median_runtime_seconds"],
        ascending=[False, False, True], kind="stable",
    )
    output.mkdir(parents=True)
    cases.to_csv(output / "per_case_validation_audit.csv", index=False)
    configs.to_csv(output / "validation_config_summary.csv", index=False)
    print(configs.to_string(index=False))
    print(f"VALIDATION_REPAIR_SWEEP_ANALYSIS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
