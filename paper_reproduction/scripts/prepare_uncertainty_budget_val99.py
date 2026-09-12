#!/usr/bin/env python3
"""Rebuild the frozen blind val99 uncertainty-budget input package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--pose-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-source-n", type=int, default=100)
    parser.add_argument("--expected-selected-n", type=int, default=99)
    parser.add_argument("--expected-excluded", default="7Z0T")
    parser.add_argument("--poses-per-source", type=int, default=40)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-pose-scores-sha256")
    args = parser.parse_args()

    source_path = args.source_manifest.expanduser().resolve()
    scores_path = args.pose_scores.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not scores_path.is_file():
        raise FileNotFoundError(scores_path)
    if args.expected_source_n <= 0 or args.expected_selected_n <= 0:
        raise ValueError("Expected cohort sizes must be positive")
    if args.poses_per_source <= 0:
        raise ValueError("poses-per-source must be positive")

    source_hash = sha256(source_path)
    scores_hash = sha256(scores_path)
    if (
        args.expected_source_sha256
        and source_hash != args.expected_source_sha256.lower()
    ):
        raise ValueError(
            "Source manifest SHA256 mismatch: "
            f"expected {args.expected_source_sha256}, found {source_hash}"
        )
    if (
        args.expected_pose_scores_sha256
        and scores_hash != args.expected_pose_scores_sha256.lower()
    ):
        raise ValueError(
            "Pose-score SHA256 mismatch: "
            f"expected {args.expected_pose_scores_sha256}, found {scores_hash}"
        )

    source = pd.read_csv(source_path, dtype=str)
    scores = pd.read_csv(scores_path, dtype=str)
    required_source = {"complex_name", "reference_ligand"}
    required_scores = {"complex_name", "source"}
    if missing := sorted(required_source - set(source.columns)):
        raise ValueError(f"Source manifest lacks columns: {missing}")
    if missing := sorted(required_scores - set(scores.columns)):
        raise ValueError(f"Pose-score table lacks columns: {missing}")

    source = source.copy()
    scores = scores.copy()
    source["complex_name"] = normalize_id(source["complex_name"])
    scores["complex_name"] = normalize_id(scores["complex_name"])
    scores["source"] = scores["source"].astype(str).str.strip().str.lower()
    if len(source) != args.expected_source_n:
        raise ValueError(
            f"Expected {args.expected_source_n} source rows, found {len(source)}"
        )
    if source["complex_name"].eq("").any() or source["complex_name"].duplicated().any():
        raise ValueError("Source manifest has empty or duplicate complex IDs")

    counts = scores.groupby(["complex_name", "source"]).size().unstack(fill_value=0)
    total = scores.groupby("complex_name").size()
    expected_pool = 2 * args.poses_per_source
    complete_ids = {
        complex_name
        for complex_name in source["complex_name"]
        if complex_name in counts.index
        and int(counts.at[complex_name, "baseline"] if "baseline" in counts else 0)
        == args.poses_per_source
        and int(counts.at[complex_name, "guided"] if "guided" in counts else 0)
        == args.poses_per_source
        and int(total.get(complex_name, 0)) == expected_pool
    }
    selected = source[source["complex_name"].isin(complete_ids)].copy()
    excluded = sorted(set(source["complex_name"]) - set(selected["complex_name"]))
    expected_excluded = sorted(
        value.strip().upper()
        for value in args.expected_excluded.split(",")
        if value.strip()
    )
    if len(selected) != args.expected_selected_n:
        raise ValueError(
            f"Expected {args.expected_selected_n} selected cases, found {len(selected)}"
        )
    if excluded != expected_excluded:
        raise ValueError(
            f"Excluded-case mismatch: expected {expected_excluded}, found {excluded}"
        )

    blind = selected.drop(columns=["reference_ligand"])
    if "reference_ligand" in blind.columns:
        raise AssertionError("Blind manifest still contains reference_ligand")

    blind_path = output / "inference_blind_val99.csv"
    reveal_path = output / "evaluation_reveal_val99.csv"
    cohort_path = output / "validation99_ids.txt"
    summary_path = output / "summary.json"
    targets = (blind_path, reveal_path, cohort_path, summary_path)
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    blind.to_csv(blind_path, index=False)
    selected.to_csv(reveal_path, index=False)
    cohort_path.write_text("\n".join(selected["complex_name"]) + "\n")
    summary = {
        "source_manifest": str(source_path),
        "source_manifest_n": len(source),
        "selection_source": str(scores_path),
        "selection_rule": (
            f"Retain the {len(selected)} validation complexes with an existing "
            f"complete baseline{args.poses_per_source}+guided{args.poses_per_source} "
            "NCC pose pool; no RMSD or success label was read."
        ),
        "selected_n": len(selected),
        "excluded_n": len(excluded),
        "excluded_ids": excluded,
        "blind_manifest": str(blind_path),
        "reveal_manifest": str(reveal_path),
        "cohort": str(cohort_path),
        "blind_contains_reference_ligand": False,
        "sha256": {
            "source_manifest": source_hash,
            "pose_scores": scores_hash,
            "blind_manifest": sha256(blind_path),
            "reveal_manifest": sha256(reveal_path),
            "cohort": sha256(cohort_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"VALIDATION99_BLIND_INPUT_OK: {blind_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
