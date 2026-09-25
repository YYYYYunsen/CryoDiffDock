#!/usr/bin/env python3
"""Build compatibility replicate roots and a selected-pose table from PAAHS runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def final_audit(root: Path) -> Path:
    for relative in ("final/per_case_pipeline_audit.csv", "step7/per_case_pipeline_audit.csv"):
        path = root / relative
        if path.is_file():
            return path
    raise FileNotFoundError(f"No final PAAHS audit below {root}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=173)
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    selected_rows = []
    for order in range(1, 9):
        number = f"{order:02d}"
        input_dir = input_root / f"replicate{number}"
        paahs_dir = run_root / f"replicate{number}"
        audit_path = final_audit(paahs_dir)
        audit = pd.read_csv(audit_path, dtype={"complex_name": str})
        rankings = pd.read_csv(input_dir / "original_rankings.csv", dtype=str)
        for frame in (audit, rankings):
            frame["complex_name"] = frame["complex_name"].astype(str).str.upper()
        if len(audit) != args.expected_cases or audit["complex_name"].duplicated().any():
            raise ValueError(f"Invalid PAAHS audit: {audit_path}")
        table = rankings.merge(
            audit[["complex_name", "selected_pose_path", "selected_protein_path"]],
            on="complex_name", validate="one_to_one",
        )
        if len(table) != args.expected_cases:
            raise ValueError(f"Replicate {number} case mismatch")

        compat_root = output_root / "compat_replicates" / f"run{order}"
        selected_root = compat_root / "03_v2_top1" / "v2_selected"
        for case_order, row in enumerate(table.itertuples(index=False)):
            original_ranking = Path(row.ranking_path).expanduser().resolve()
            ranking = pd.read_csv(original_ranking, dtype={"complex_name": str})
            if "materialized_rank" in ranking:
                ranking = ranking[pd.to_numeric(ranking["materialized_rank"], errors="coerce").eq(1)]
            if len(ranking) != 1:
                raise ValueError(f"Expected one Top-1 row: {original_ranking}")
            ranking = ranking.copy()
            ranking.loc[:, "materialized_path"] = str(Path(row.selected_pose_path).resolve())
            ranking.loc[:, "protein_path"] = str(Path(row.selected_protein_path).resolve())
            if "final_severe_clash_count" in audit.columns:
                final_row = audit[audit["complex_name"].eq(row.complex_name)].iloc[0]
                ranking.loc[:, "severe_clash_count"] = int(
                    final_row["final_severe_clash_count"]
                )
            if "phase3_target_gates_pass" in audit.columns:
                target_pass = bool(
                    audit[audit["complex_name"].eq(row.complex_name)].iloc[0][
                        "phase3_target_gates_pass"
                    ]
                )
                ranking.loc[:, "relative_physics_pass"] = target_pass
                ranking.loc[:, "density_physics_gate_pass"] = target_pass
            case_dir = selected_root / f"index{case_order}___{row.complex_name}"
            case_dir.mkdir(parents=True)
            compat_ranking = case_dir / "ranking.csv"
            ranking.to_csv(compat_ranking, index=False)

            record = ranking.iloc[0].to_dict()
            selected_rows.append({
                "complex_name": row.complex_name,
                "replicate": f"run{order}",
                "replicate_order": order,
                "cohort_order": case_order,
                "selected_pose_path": str(Path(row.selected_pose_path).resolve()),
                "protein_path": str(Path(row.selected_protein_path).resolve()),
                "ranking_path": str(compat_ranking.resolve()),
                "confidence": record.get("confidence"),
                "final_selection_score": record.get("final_selection_score"),
                "multiscale_ncc": record.get("multiscale_ncc"),
                "expanded_density_z": record.get("expanded_density_z"),
            })
        (compat_root / "replicate_summary.json").write_text(json.dumps({
            "analysis_role": "compatibility_index_for_post_paahs_multirun_analysis",
            "replicate_order": order,
            "case_count": args.expected_cases,
            "paahs_root": str(paahs_dir),
            "uses_reference_coordinates": False,
        }, indent=2) + "\n")

    selected = pd.DataFrame(selected_rows).sort_values(
        ["cohort_order", "replicate_order"], kind="stable"
    )
    if len(selected) != 8 * args.expected_cases:
        raise ValueError("Post-PAAHS selected table is incomplete")
    selected.to_csv(output_root / "selected_pose_replicates.csv", index=False)
    print(f"PAAHS_MULTIRUN_INDEX_OK: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
