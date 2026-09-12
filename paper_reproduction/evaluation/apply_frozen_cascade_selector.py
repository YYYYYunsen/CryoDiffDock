#!/usr/bin/env python3
"""Apply a validation-frozen cascade selector without reading test outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.tune_validation_cascade_selector import choose_parent, read_ids


FORBIDDEN = ("rmsd", "reference", "native", "ground_truth", "success")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--candidate-features", required=True, type=Path)
    parser.add_argument("--feature-seal", required=True, type=Path)
    parser.add_argument("--frozen-selector", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--materialize-selected", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate every sealed input and pose path without writing output.",
    )
    args = parser.parse_args(argv)
    cohort = read_ids(args.cohort)
    if len(cohort) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} test parents, found {len(cohort)}")
    if not args.preflight_only and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")

    frozen = json.loads(args.frozen_selector.read_text())
    if frozen.get("status") != "validation_frozen" or frozen.get("test_outcomes_used") is not False:
        raise ValueError("Selector is not explicitly validation-frozen")
    config = frozen.get("selected_config", {})
    required_config = {"guard_margin", "site_weight", "basin_weight", "shared_weight"}
    if required_config - set(config):
        raise ValueError("Frozen selector lacks required parameters")
    feature_seal = json.loads(args.feature_seal.read_text())
    if feature_seal.get("status") not in {"selection_sealed", "evaluated"}:
        raise ValueError("Test candidate features were not sealed")
    if feature_seal.get("features_sha256") != sha256(args.candidate_features):
        raise ValueError("Test candidate feature hash differs from its seal")

    # Recover and verify the validation feature definition recorded at freeze.
    validation_seals = [
        (Path(path), checksum) for path, checksum in frozen.get("input_hashes", {}).items()
        if path.endswith("experiment_manifest.json")
    ]
    if len(validation_seals) != 1 or not validation_seals[0][0].is_file():
        raise ValueError("Cannot locate the sealed validation feature definition")
    validation_seal_path, validation_seal_sha256 = validation_seals[0]
    if sha256(validation_seal_path) != validation_seal_sha256:
        raise ValueError("Validation feature seal changed after tuning")
    validation_seal = json.loads(validation_seal_path.read_text())
    for key in ("sigmas_A", "mask_radius_A", "shared_target"):
        if feature_seal.get(key) != validation_seal.get(key):
            raise ValueError(f"Test/validation feature definition differs for {key}")

    features = pd.read_csv(args.candidate_features)
    forbidden = [name for name in features if any(token in name.lower() for token in FORBIDDEN)]
    if forbidden:
        raise ValueError(f"Blind candidate feature table contains forbidden outcome columns: {forbidden}")
    required = {"pdb_id", "candidate", "candidate_id", "global_rank", "basin_q25_ncc", "pose_path"}
    shared = sorted(
        (name for name in features if name.startswith("shared_ncc_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if required - set(features) or len(shared) < 2:
        raise ValueError("Test candidate feature schema is incomplete")
    features["pdb_id"] = features.pdb_id.astype(str).str.upper()
    features["candidate"] = features.candidate.astype(str)
    features["candidate_id"] = features.candidate_id.astype(str)
    features["global_rank"] = pd.to_numeric(features.global_rank, errors="raise").astype(int)
    for column in ["basin_q25_ncc", *shared]:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    if set(features.pdb_id) != set(cohort) or features.duplicated(["pdb_id", "candidate"]).any():
        raise ValueError("Test feature parent/candidate identities differ from cohort")

    margin = float(config["guard_margin"])
    weights = tuple(int(config[name]) for name in ("site_weight", "basin_weight", "shared_weight"))
    rows = []
    for pdb_id in cohort:
        group = features[features.pdb_id.eq(pdb_id)].copy()
        global_rows = group[group.candidate.eq("global")]
        if len(global_rows) != 1:
            raise ValueError(f"{pdb_id}: expected exactly one Global anchor")
        complete = len(group) == 6 and set(group.loc[group.candidate.ne("global"), "global_rank"]) == set(range(1, 6))
        if complete:
            chosen = choose_parent(group, margin, weights, shared)
            reason = "frozen_guard_and_rank_fusion"
        else:
            chosen = global_rows.iloc[0]
            reason = "incomplete_top5_global_fallback"
        pose_path = Path(str(chosen.pose_path)).expanduser().resolve()
        if not pose_path.is_file():
            raise FileNotFoundError(pose_path)
        rows.append({
            "pdb_id": pdb_id,
            "selected_candidate": str(chosen.candidate),
            "selected_candidate_id": str(chosen.candidate_id),
            "selected_pose_path": str(pose_path),
            "selected_global_rank": int(chosen.global_rank),
            "selected_global": bool(chosen.candidate == "global"),
            "complete_top5": bool(complete),
            "reason": reason,
        })
    choices = pd.DataFrame(rows)
    if args.preflight_only:
        print(
            "FROZEN_SELECTOR_TEST_PREFLIGHT_OK: "
            f"cohort={len(choices)}, selected_global={int(choices.selected_global.sum())}, "
            f"selected_local={int((~choices.selected_global).sum())}, "
            f"incomplete_fallback={int((~choices.complete_top5).sum())}"
        )
        return 0
    args.output_dir.mkdir(parents=True)
    selected_path = args.output_dir / "selected_poses_blind.csv"
    choices.to_csv(selected_path, index=False)
    if args.materialize_selected:
        for row in choices.itertuples(index=False):
            destination = args.output_dir / "selected_results" / row.pdb_id
            destination.mkdir(parents=True)
            shutil.copy2(row.selected_pose_path, destination / "rank1.sdf")
    metadata = {
        "status": "test_selection_sealed_before_this_evaluation",
        "selector_status": "validation_frozen",
        "selection_uses_test_rmsd": False,
        "selection_uses_reference": False,
        "test_outcomes_previously_explored_in_project": True,
        "cohort_n": len(cohort),
        "selected_global_n": int(choices.selected_global.sum()),
        "selected_local_n": int((~choices.selected_global).sum()),
        "incomplete_top5_global_fallback_n": int((~choices.complete_top5).sum()),
        "selected_config": config,
        "frozen_selector_sha256": sha256(args.frozen_selector),
        "candidate_features_sha256": sha256(args.candidate_features),
        "feature_seal_sha256": sha256(args.feature_seal),
        "cohort_sha256": sha256(args.cohort),
        "selected_poses_sha256": sha256(selected_path),
        "applier_sha256": sha256(Path(__file__)),
    }
    (args.output_dir / "selection_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"FROZEN_SELECTOR_TEST_SELECTION_SEALED: {selected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
