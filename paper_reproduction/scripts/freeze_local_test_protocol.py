#!/usr/bin/env python3
"""Create or verify an immutable CryoDiffDock-Pocket test protocol bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


IMPLEMENTATION_FILES = (
    "inference.py",
    "cryodiffdock_pocket/density_guidance.py",
    "cryodiffdock_pocket/density_ranking.py",
    "cryodiffdock_pocket/discrete_search.py",
    "cryodiffdock_pocket/physics.py",
    "scripts/run_discrete_density_search.py",
    "scripts/rerank_ligand_by_density_ncc.py",
    "scripts/score_coarse_basin_support.py",
    "scripts/score_discrete_candidates_gnina.py",
    "scripts/build_parent_gnina_controls.py",
    "scripts/apply_frozen_discrete_ranking.py",
    "scripts/apply_frozen_parent_relative_gnina_gate.py",
    "evaluation/evaluate_ligand_accuracy.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(item for item in path.rglob("*") if item.is_file())
    else:
        raise FileNotFoundError(path)


def git_value(project: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=project, text=True, capture_output=True, check=True
        )
        return completed.stdout.strip()
    except Exception:
        return None


def protocol(s0_config: dict, final_config: dict) -> dict:
    return {
        "benchmark": {
            "mode": "oracle-pocket deposited-holo redocking",
            "test_policy": "all valid cases from the pre-existing test split",
            "technical_failures": "reported and excluded identically from all paired arms",
            "reference_use_before_unblinding": "crop centre and QC only",
        },
        "density_preparation": {
            "box_size_A": 28.0,
            "spacing_A": 0.75,
            "mask_policy": "full_polymer",
            "mask_style": "soft",
            "mask_inner_A": 1.5,
            "mask_outer_A": 2.5,
        },
        "inference": {
            "seed": 20260813,
            "samples_per_complex_per_arm": 40,
            "batch_size": 10,
            "inference_steps": 30,
            "arms": {
                "baseline": {"density_guidance": False},
                "guided": {
                    "density_guidance": True,
                    "objective": "atom_density",
                    "lambda_translation": 0.5,
                    "lambda_rotation": 0.5,
                    "start_fraction": 0.6,
                    "clip_ratio": 1.0,
                },
            },
        },
        "ncc": {
            "gaussian_sigma_A": 1.5,
            "mask_radius_A": 10.0,
            "confidence_plus_ncc_weight": 2.0,
            "candidate_pool": "baseline40+guided40",
        },
        "discrete_expansion": {
            "parent_score_column": "confidence",
            "parent_count": 6,
            "top_score_parents": 3,
            "parent_diversity_rmsd_A": 1.5,
            "rotation_degrees": [15.0, 30.0, 60.0],
            "translation_distances_A": [0.5, 1.0],
            "torsion_degrees": [60.0, 120.0, 180.0],
            "max_torsion_bonds": 8,
            "coarse_orientation_library": True,
            "minimum_ncc_gain": 0.0,
            "anchor_penalty": 0.05,
            "max_centroid_shift_A": 1.5,
            "min_in_bounds_fraction": 0.95,
            "severe_overlap_A": 0.8,
            "materialize_all_candidates": True,
        },
        "basin_support": {
            "coarse_top_k": 32,
            "rotation_degrees": [10.0, 20.0],
            "translation_distances_A": [0.25, 0.5],
            "ncc_tolerance": 0.01,
        },
        "s0_selector": s0_config,
        "final_parent_relative_gnina_selector": final_config,
        "test_rule": (
            "No hyperparameter search, threshold changes, case removal, or "
            "inspection of RMSD is permitted before all selected poses are fixed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-split", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--final-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--artifact", action="append", default=[], type=Path,
        help="Checkpoint, model directory, GNINA image, or other immutable input; repeatable.",
    )
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    output = args.output_dir.expanduser().resolve()
    split = args.test_split.expanduser().resolve()
    s0_path = args.s0_config.expanduser().resolve()
    final_path = args.final_config.expanduser().resolve()
    for path in (split, s0_path, final_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tracked = [split, s0_path, final_path]
    tracked.extend(project / name for name in IMPLEMENTATION_FILES)
    for artifact in args.artifact:
        tracked.extend(files_under(artifact.expanduser().resolve()))
    tracked = sorted(set(path.resolve() for path in tracked), key=str)
    hashes = {str(path): sha256(path) for path in tracked}

    lock_path = output / "lock.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text())
        if existing.get("sha256") != hashes:
            raise RuntimeError(
                "Frozen bundle differs from current files. Use a new output directory; "
                "never overwrite a test lock."
            )
        print(f"FROZEN_TEST_PROTOCOL_VERIFIED: {output}")
        return 0

    output.mkdir(parents=True, exist_ok=False)
    s0_config = json.loads(s0_path.read_text())
    final_config = json.loads(final_path.read_text())
    shutil.copy2(s0_path, output / "s0_config.json")
    shutil.copy2(final_path, output / "final_config.json")
    (output / "protocol.json").write_text(
        json.dumps(protocol(s0_config, final_config), indent=2) + "\n"
    )
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split": str(split),
        "test_split_sha256": hashes[str(split)],
        "git_commit": git_value(project, "rev-parse", "HEAD"),
        "git_status_porcelain": git_value(project, "status", "--porcelain"),
        "sha256": hashes,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in hashes.items())
    )
    print(f"FROZEN_TEST_PROTOCOL_CREATED: {output}")
    print(f"test split sha256: {hashes[str(split)]}")
    print(f"files hashed: {len(hashes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
