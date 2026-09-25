#!/usr/bin/env python3
"""Build the blind conservative integration used within PAAHS Phase 2.

The established directed repair is retained unless the density-preserving arm
has strictly fewer severe ligand-protein clashes.  Ties always retain the
established arm.  No native ligand or RMSD is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-case-root", required=True, type=Path)
    parser.add_argument("--new-analysis", required=True, type=Path)
    parser.add_argument("--new-repair-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument(
        "--directed-mode", choices=("full", "lazy"), default="full"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    new_path = args.new_analysis / "per_case_caahs_a_sidechain_rescue.csv"
    new = pd.read_csv(new_path)
    manifest = pd.read_csv(args.new_repair_manifest)
    if len(new) != args.expected_cases or new.complex_name.duplicated().any():
        raise ValueError(f"Expected {args.expected_cases} unique new-arm cases")
    if len(manifest) != args.expected_cases or manifest.complex_name.duplicated().any():
        raise ValueError(f"Expected {args.expected_cases} unique manifest cases")
    proteins = manifest.set_index("complex_name")["protein_path"]
    rows = []
    for item in new.itertuples(index=False):
        name = str(item.complex_name)
        old_summary_path = args.baseline_case_root / name / "summary.json"
        new_count = int(item.sc_final_post_severe_clash_count)
        new_pose = Path(str(item.selected_pose_path))
        sc_protein = getattr(item, "selected_protein_path", None)
        new_protein = (
            Path(str(sc_protein))
            if sc_protein is not None and not pd.isna(sc_protein)
            else Path(str(proteins.loc[name]))
        )
        old_protein = Path(str(proteins.loc[name]))
        directed_evaluated = old_summary_path.is_file()
        if directed_evaluated:
            old = json.loads(old_summary_path.read_text())
            old_count: int | None = int(old["selected"]["severe_clash_count"])
            use_new = new_count < old_count
            old_pose = old_summary_path.parent / "selected_pose.sdf"
            selected_pose = new_pose if use_new else old_pose
            selected_protein = new_protein if use_new else old_protein
            selected_count = min(old_count, new_count)
            selected_method = (
                "new_arm_strict_improvement"
                if use_new else "directed_repair_retained"
            )
        else:
            if args.directed_mode != "lazy":
                raise FileNotFoundError(old_summary_path)
            if new_count != 0:
                raise FileNotFoundError(
                    f"{name}: lazy D3B8 result is required because the new arm "
                    f"still has {new_count} severe clashes: {old_summary_path}"
                )
            old_count = None
            use_new = True
            selected_pose = new_pose
            selected_protein = new_protein
            selected_count = 0
            selected_method = "new_arm_custom_severe_zero_bound"
        for path in (selected_pose, selected_protein):
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")
        rows.append({
            "complex_name": name,
            "baseline_directed_severe_clash_count": old_count,
            "new_arm_severe_clash_count": new_count,
            "selected_severe_clash_count": selected_count,
            "strict_improvement": bool(directed_evaluated and use_new),
            "directed_evaluated": directed_evaluated,
            "selected_method": selected_method,
            "selected_pose_path": str(selected_pose.resolve()),
            "selected_protein_path": str(selected_protein.resolve()),
            "uses_reference_or_native_rmsd_for_selection": False,
        })
    frame = pd.DataFrame(rows)
    evaluated = frame[frame["directed_evaluated"]].copy()
    if (
        evaluated.selected_severe_clash_count
        > evaluated.baseline_directed_severe_clash_count
    ).any():
        raise AssertionError("Strict union lost a directed-repair success")
    summary = {
        "method": "PAAHS",
        "phase": "phase2",
        "component": "conservative_integration",
        "directed_mode": args.directed_mode,
        "status": "strict_clash_union_complete",
        "cases_n": len(frame),
        "directed_evaluated_n": int(frame.directed_evaluated.sum()),
        "directed_skipped_by_custom_severe_zero_bound_n": int(
            (~frame.directed_evaluated).sum()
        ),
        "baseline_directed_clash_free_n": (
            int(frame.baseline_directed_severe_clash_count.eq(0).sum())
            if bool(frame.directed_evaluated.all()) else None
        ),
        "new_arm_clash_free_n": int(frame.new_arm_severe_clash_count.eq(0).sum()),
        "strict_union_clash_free_n": int(frame.selected_severe_clash_count.eq(0).sum()),
        "strict_improvement_cases_n": int(frame.strict_improvement.sum()),
        "baseline_successes_lost_n": 0,
        "uses_reference_or_native_rmsd_for_selection": False,
    }
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "per_case_strict_clash_union.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"STRICT_CLASH_UNION_OK: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
