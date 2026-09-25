#!/usr/bin/env python3
"""Combine frozen V2 results with V3 continuous-refinement fallbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def selected_case_ids(path: Path, split: str | None) -> set[str]:
    table = pd.read_csv(path)
    id_column = next(
        (column for column in ("complex_name", "pdb_id", "complex_id") if column in table),
        None,
    )
    if id_column is None:
        raise ValueError(f"No case ID column in {path}")
    if split is not None:
        if "split" not in table:
            raise ValueError(f"--split requires a split column in {path}")
        table = table[table["split"].astype(str).eq(split)]
    values = table[id_column].dropna().astype(str)
    if values.duplicated().any():
        raise ValueError(f"Duplicate case IDs in {path}")
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-case-root", required=True, type=Path)
    parser.add_argument("--v3-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--split")
    args = parser.parse_args()
    v2_root = args.v2_case_root.expanduser().resolve()
    v3_root = args.v3_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    v2_summaries = sorted(v2_root.glob("*/summary.json"))
    if args.split and not args.case_manifest:
        raise ValueError("--split requires --case-manifest")
    if args.case_manifest:
        requested = selected_case_ids(args.case_manifest.expanduser().resolve(), args.split)
        available = {path.parent.name for path in v2_summaries}
        if missing := sorted(requested - available):
            raise ValueError(f"Manifest cases absent from V2 root: {missing[:20]}")
        v2_summaries = [path for path in v2_summaries if path.parent.name in requested]
    if len(v2_summaries) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} V2 summaries, found {len(v2_summaries)}"
        )
    rows = []
    for v2_path in v2_summaries:
        name = v2_path.parent.name
        v2 = json.loads(v2_path.read_text())
        original = v2["original"]
        v2_selected = v2["selected"]
        v3_path = v3_root / "cases" / name / "summary.json"
        if v3_path.is_file():
            v3 = json.loads(v3_path.read_text())
            selected = v3["selected"]
            role = str(v3.get("analysis_role", ""))
            baseline_ids = {"v2_selected", "v4_selected"}
            if selected["candidate_id"] in baseline_ids:
                selected_method = "input_retained"
            elif "iterative_local" in role:
                selected_method = "v5_iterative_local"
            elif "original_restart" in role:
                selected_method = "v6_original_restart"
            else:
                selected_method = "v3_continuous"
            runtime_v3 = float(v3["runtime_seconds"])
            selected_pose = v3_path.parent / "selected_pose.sdf"
        else:
            selected = v2_selected
            selected_method = "v2_already_resolved"
            runtime_v3 = 0.0
            selected_pose = v2_path.parent / "selected_pose.sdf"
        pre = int(original["severe_clash_count"])
        v2_post = int(v2_selected["severe_clash_count"])
        final_post = int(selected["severe_clash_count"])
        if final_post > v2_post:
            raise AssertionError(f"{name}: V3 worsened clash count")
        rows.append({
            "complex_name": name,
            "pre_severe_clash_count": pre,
            "v2_post_severe_clash_count": v2_post,
            "final_post_severe_clash_count": final_post,
            "additional_clash_reduction": v2_post - final_post,
            "v2_resolved": v2_post == 0,
            "final_resolved": final_post == 0,
            "newly_rescued_by_v3": v2_post > 0 and final_post == 0,
            "selected_method": selected_method,
            "final_displacement_A": float(selected["displacement_A"]),
            "final_delta_ncc": float(selected["ncc_change"]),
            "v3_runtime_seconds": runtime_v3,
            "selected_pose_path": str(selected_pose),
        })
    frame = pd.DataFrame(rows)
    targets = frame[frame["pre_severe_clash_count"] > 0]
    pre_total = int(targets["pre_severe_clash_count"].sum())
    post_total = int(targets["final_post_severe_clash_count"].sum())
    summary = {
        "cases_n": len(frame),
        "clashing_targets_n": len(targets),
        "v2_targets_resolved_n": int(targets["v2_resolved"].sum()),
        "v3_combined_targets_resolved_n": int(targets["final_resolved"].sum()),
        "newly_rescued_by_v3_n": int(targets["newly_rescued_by_v3"].sum()),
        "cases_with_additional_clash_reduction_n": int(
            (targets["additional_clash_reduction"] > 0).sum()
        ),
        "overall_clash_free_n": int((frame["final_post_severe_clash_count"] == 0).sum()),
        "overall_clash_free_pct": float(
            100.0 * (frame["final_post_severe_clash_count"] == 0).mean()
        ),
        "target_resolution_pct": float(100.0 * targets["final_resolved"].mean()),
        "total_clash_reduction_pct": float(100.0 * (1.0 - post_total / pre_total)),
        "v2_residual_clashes": int(targets["v2_post_severe_clash_count"].sum()),
        "v3_combined_residual_clashes": post_total,
        "additional_runtime_hours": float(frame["v3_runtime_seconds"].sum() / 3600.0),
        "reference_or_rmsd_used": False,
    }
    output.mkdir(parents=True)
    frame.to_csv(output / "per_case_continuous_refinement.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CONTINUOUS_REFINEMENT_ANALYSIS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
