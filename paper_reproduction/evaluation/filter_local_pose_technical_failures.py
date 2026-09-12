#!/usr/bin/env python3
"""Audit and filter technically invalid Local density-scoring poses.

Validity is determined without reference coordinates or RMSD. A pose is valid
only when its density score is finite and at least one ligand atom lies inside
the local density crop. Complexes with fewer than the required number of valid
parent poses are marked as technical failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--density-column", default="ncc_sigma_1p5")
    parser.add_argument(
        "--in-bounds-column", default="ligand_in_bounds_fraction"
    )
    parser.add_argument("--minimum-valid-poses", type=int, default=6)
    parser.add_argument("--expected-complexes", type=int)
    args = parser.parse_args()

    if args.minimum_valid_poses <= 0:
        raise ValueError("--minimum-valid-poses must be positive")
    scores_path = args.scores.expanduser().resolve()
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    scores = pd.read_csv(scores_path, dtype={"complex_name": str})
    required = {
        "row_index", "complex_name", args.density_column,
        args.in_bounds_column,
    }
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{scores_path} is missing columns: {missing}")
    scores = scores.copy()
    scores["complex_name"] = scores["complex_name"].str.strip().str.upper()
    density = pd.to_numeric(scores[args.density_column], errors="coerce")
    in_bounds = pd.to_numeric(scores[args.in_bounds_column], errors="coerce")
    finite_density = np.isfinite(density.to_numpy(float))
    positive_overlap = np.isfinite(in_bounds.to_numpy(float)) & in_bounds.gt(0).to_numpy()
    valid = finite_density & positive_overlap

    scores["technical_pose_valid"] = valid
    scores["technical_failure_reason"] = ""
    scores.loc[~finite_density, "technical_failure_reason"] = (
        "nonfinite_density_score"
    )
    scores.loc[
        finite_density & ~positive_overlap, "technical_failure_reason"
    ] = "no_ligand_atoms_inside_density_crop"

    complex_order = scores["complex_name"].drop_duplicates().tolist()
    if (
        args.expected_complexes is not None
        and len(complex_order) != args.expected_complexes
    ):
        raise ValueError(
            f"Expected {args.expected_complexes} complexes, found {len(complex_order)}"
        )

    case_rows = []
    for complex_name in complex_order:
        group = scores[scores["complex_name"].eq(complex_name)]
        valid_count = int(group["technical_pose_valid"].sum())
        total_count = len(group)
        usable = valid_count >= args.minimum_valid_poses
        case_rows.append({
            "complex_name": complex_name,
            "total_pose_count": total_count,
            "valid_pose_count": valid_count,
            "technical_failure_pose_count": total_count - valid_count,
            "minimum_valid_poses_required": args.minimum_valid_poses,
            "case_status": (
                "technically_evaluable"
                if usable else "technical_failure_insufficient_valid_poses"
            ),
        })
    cases = pd.DataFrame(case_rows)
    usable_ids = cases.loc[
        cases["case_status"].eq("technically_evaluable"), "complex_name"
    ].tolist()
    failed_ids = cases.loc[
        ~cases["case_status"].eq("technically_evaluable"), "complex_name"
    ].tolist()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "pose_score_technical_audit.csv"
    valid_path = output / "pose_scores_valid.csv"
    cases_path = output / "case_technical_status.csv"
    usable_path = output / "usable_case_ids.txt"
    failed_path = output / "technical_failure_case_ids.txt"
    summary_path = output / "summary.json"
    scores.to_csv(audit_path, index=False)
    scores.loc[valid].drop(
        columns=["technical_pose_valid", "technical_failure_reason"]
    ).to_csv(valid_path, index=False)
    cases.to_csv(cases_path, index=False)
    usable_path.write_text("".join(f"{value}\n" for value in usable_ids))
    failed_path.write_text("".join(f"{value}\n" for value in failed_ids))

    summary = {
        "input_complexes": len(complex_order),
        "input_poses": len(scores),
        "valid_poses": int(valid.sum()),
        "technical_failure_poses": int((~valid).sum()),
        "minimum_valid_poses_per_complex": args.minimum_valid_poses,
        "technically_evaluable_complexes": len(usable_ids),
        "technical_failure_complexes": len(failed_ids),
        "technical_failure_complex_ids": failed_ids,
        "validity_rule": (
            f"finite {args.density_column} and {args.in_bounds_column} > 0"
        ),
        "uses_reference_coordinates_or_rmsd": False,
        "input": str(scores_path),
        "input_sha256": sha256(scores_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(cases.to_string(index=False))
    print(f"LOCAL_POSE_TECHNICAL_FILTER_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
