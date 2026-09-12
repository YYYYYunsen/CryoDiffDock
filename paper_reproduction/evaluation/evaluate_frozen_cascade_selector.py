#!/usr/bin/env python3
"""Reveal RMSD for an already sealed validation-frozen test selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.compare_global_cascade import summarize
from evaluation.tune_validation_cascade_selector import read_ids


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--selected-poses", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--candidate-features", required=True, type=Path)
    parser.add_argument("--global-audit", required=True, type=Path)
    parser.add_argument("--branch-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    cohort = read_ids(args.cohort)
    if len(cohort) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} parents, found {len(cohort)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")
    seal = json.loads(args.selection_manifest.read_text())
    if seal.get("status") != "test_selection_sealed_before_this_evaluation":
        raise ValueError("Test selection is not sealed")
    if seal.get("selected_poses_sha256") != sha256(args.selected_poses):
        raise ValueError("Sealed test selection changed")
    if seal.get("candidate_features_sha256") != sha256(args.candidate_features):
        raise ValueError("Candidate features changed after selection")
    selected = pd.read_csv(args.selected_poses, dtype=str).fillna("")
    if (
        len(selected) != len(cohort)
        or selected.pdb_id.str.upper().duplicated().any()
        or set(selected.pdb_id.str.upper()) != set(cohort)
    ):
        raise ValueError("Selected pose table does not match fixed test denominator")
    selected["pdb_id"] = selected.pdb_id.str.upper()
    selected = selected.set_index("pdb_id").reindex(cohort)

    audit = pd.read_csv(args.global_audit)
    audit["pdb_id"] = audit.pdb_id.astype(str).str.upper()
    audit["global_top1_rmsd_A"] = pd.to_numeric(audit.global_top1_rmsd_A, errors="coerce")
    if audit.pdb_id.duplicated().any() or set(cohort) - set(audit.pdb_id):
        raise ValueError("Global audit does not cover test cohort")
    global_rmsd = audit.set_index("pdb_id").global_top1_rmsd_A.reindex(cohort)
    branch = pd.read_csv(args.branch_table, dtype={"candidate_id": str})
    branch["complex_name"] = branch.complex_name.astype(str)
    branch["candidate_id"] = branch.candidate_id.astype(str)
    branch["parent_complex_name"] = branch.parent_complex_name.astype(str).str.upper()
    branch["global_rank"] = pd.to_numeric(branch.global_rank, errors="raise").astype(int)
    branch["selected_pose_rmsd_A"] = pd.to_numeric(branch.selected_pose_rmsd_A, errors="coerce")
    if branch.complex_name.duplicated().any():
        raise ValueError("Duplicate branch metrics")
    branch = branch.set_index("complex_name")

    features = pd.read_csv(args.candidate_features, dtype={"candidate_id": str})
    features["pdb_id"] = features.pdb_id.astype(str).str.upper()
    features["candidate"] = features.candidate.astype(str)
    features["candidate_id"] = features.candidate_id.astype(str)
    features["global_rank"] = pd.to_numeric(features.global_rank, errors="raise").astype(int)
    selected_rmsd, oracle = [], []
    for pdb_id in cohort:
        choice = selected.loc[pdb_id]
        candidate = str(choice.selected_candidate)
        feature_match = features[
            features.pdb_id.eq(pdb_id) & features.candidate.eq(candidate)
        ]
        if len(feature_match) != 1:
            raise ValueError(f"Selected candidate does not uniquely match sealed features: {pdb_id}/{candidate}")
        feature_row = feature_match.iloc[0]
        if (
            str(feature_row.candidate_id) != str(choice.selected_candidate_id)
            or int(feature_row.global_rank) != int(choice.selected_global_rank)
            or Path(str(feature_row.pose_path)).expanduser().resolve()
            != Path(str(choice.selected_pose_path)).expanduser().resolve()
        ):
            raise ValueError(f"Selected candidate identity changed: {pdb_id}/{candidate}")
        if candidate == "global":
            value = global_rmsd.loc[pdb_id]
        else:
            if candidate not in branch.index:
                raise ValueError(f"Missing selected branch metric: {candidate}")
            metric = branch.loc[candidate]
            if (
                str(metric.candidate_id) != str(choice.selected_candidate_id)
                or str(metric.parent_complex_name) != pdb_id
                or int(metric.global_rank) != int(choice.selected_global_rank)
            ):
                raise ValueError(f"Selected branch identity mismatch: {pdb_id}/{candidate}")
            value = metric.selected_pose_rmsd_A
        selected_rmsd.append(float(value))
        group = features[features.pdb_id.eq(pdb_id)]
        complete = str(choice.complete_top5).lower() in {"true", "1", "yes"}
        values = [float(global_rmsd.loc[pdb_id])]
        if complete:
            for name in group.loc[group.candidate.ne("global"), "candidate"].astype(str):
                if name not in branch.index or not np.isfinite(branch.loc[name, "selected_pose_rmsd_A"]):
                    raise ValueError(f"Incomplete oracle candidate metrics: {pdb_id}/{name}")
                values.append(float(branch.loc[name, "selected_pose_rmsd_A"]))
        oracle.append(min(values))
    if not np.isfinite(global_rmsd).all() or not np.isfinite(selected_rmsd).all():
        raise ValueError("Nonfinite test RMSD in fixed denominator")
    comparison = pd.DataFrame({
        "global": global_rmsd.to_numpy(float),
        "validation_frozen_selector": selected_rmsd,
        "oracle_union6": oracle,
    }, index=pd.Index(cohort, name="pdb_id"))
    summary, paired, transitions = summarize(
        comparison, args.bootstrap_repeats, args.seed,
        oracle_methods=("oracle_union6",), reference_methods=("global",),
    )
    args.output_dir.mkdir(parents=True)
    comparison.to_csv(args.output_dir / "test_per_parent_rmsd.csv")
    summary.to_csv(args.output_dir / "test_success_summary.csv", index=False)
    paired.to_csv(args.output_dir / "test_paired_comparisons.csv", index=False)
    transitions.to_csv(args.output_dir / "test_transitions.csv", index=False)
    metadata = {
        "status": "test_evaluated_once_from_sealed_selection",
        "selector_status": "validation_frozen",
        "cohort_n": len(cohort),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "selected_poses_sha256": sha256(args.selected_poses),
        "evaluation_input_hashes": {
            str(path.resolve()): sha256(path)
            for path in (args.global_audit, args.branch_table)
        },
        "evaluator_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "evaluation_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(paired.to_string(index=False))
    print(f"FROZEN_SELECTOR_TEST_EVALUATION_OK: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
