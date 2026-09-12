#!/usr/bin/env python3
"""Reveal RMSD for the fixed V2 Top-1 pose from the first replicate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return values


def locate_selected_root(root: Path) -> Path:
    candidates = (root / "03_v2_top1" / "v2_selected", root / "v2_selected", root)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("index*___*/ranking.csv")):
            return candidate
    raise FileNotFoundError(f"No V2 Top-1 rankings below {root}")


def evaluate(task: tuple[int, str, str, str]) -> dict[str, object]:
    cohort_order, complex_name, pose_text, reference_text = task
    pose = Path(pose_text)
    reference = Path(reference_text)
    row: dict[str, object] = {
        "cohort_order": cohort_order,
        "complex_name": complex_name,
        "v2_scored": False,
        "v2_rmsd_A": np.nan,
        "reported_pose_path": str(pose),
        "reference_ligand": str(reference),
        "evaluation_error": "",
    }
    try:
        if not pose.is_file():
            raise FileNotFoundError(f"Missing reported pose: {pose}")
        if not reference.is_file():
            raise FileNotFoundError(f"Missing reference ligand: {reference}")
        value = symmetry_rmsd(read_heavy_molecule(reference), read_heavy_molecule(pose))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid RMSD: {value}")
        row["v2_scored"] = True
        row["v2_rmsd_A"] = float(value)
    except Exception as exc:
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-root", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--reveal-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-n", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.expected_n <= 0 or args.workers <= 0:
        raise ValueError("expected-n and workers must be positive")

    replicate_root = args.replicate_root.expanduser().resolve()
    cohort_path = args.cohort.expanduser().resolve()
    reveal_path = args.reveal_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite endpoint table: {output}")
    cohort = read_ids(cohort_path)
    if len(cohort) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cohort IDs, found {len(cohort)}")

    reveal = pd.read_csv(reveal_path, dtype=str)
    required = {"complex_name", "reference_ligand"}
    if missing := sorted(required - set(reveal.columns)):
        raise ValueError(f"Reveal manifest lacks columns: {missing}")
    reveal = reveal.copy()
    reveal["complex_name"] = reveal["complex_name"].map(normalize_id)
    if reveal["complex_name"].duplicated().any():
        raise ValueError("Reveal manifest contains duplicate complex IDs")
    reveal = reveal.set_index("complex_name")
    missing_references = sorted(set(cohort) - set(reveal.index))
    if missing_references:
        raise ValueError(f"Reveal manifest lacks cases: {missing_references[:20]}")

    selected_root = locate_selected_root(replicate_root)
    pose_map: dict[str, Path] = {}
    for ranking_path in sorted(selected_root.glob("index*___*/ranking.csv")):
        # PDB IDs such as 9E39 must remain identifiers rather than being
        # inferred as scientific-notation numbers by pandas.
        ranking = pd.read_csv(ranking_path, dtype={"complex_name": str})
        if "materialized_rank" in ranking.columns:
            ranking = ranking[
                pd.to_numeric(ranking["materialized_rank"], errors="coerce").eq(1)
            ]
        if len(ranking) != 1:
            raise ValueError(f"Expected one reported pose: {ranking_path}")
        row = ranking.iloc[0]
        complex_name = normalize_id(row["complex_name"])
        if complex_name in pose_map:
            raise ValueError(f"Duplicate reported pose for {complex_name}")
        pose_map[complex_name] = resolve_path(row["materialized_path"], ranking_path.parent)
    missing_poses = sorted(set(cohort) - set(pose_map))
    extra_poses = sorted(set(pose_map) - set(cohort))
    if missing_poses or extra_poses:
        raise ValueError(
            f"Replicate/cohort mismatch: missing={missing_poses[:20]}, "
            f"extra={extra_poses[:20]}"
        )

    tasks = []
    for order, complex_name in enumerate(cohort):
        reference = resolve_path(
            reveal.at[complex_name, "reference_ligand"], reveal_path.parent
        )
        tasks.append((order, complex_name, str(pose_map[complex_name]), str(reference)))
    if args.workers == 1:
        rows = [evaluate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(evaluate, tasks))
    result = pd.DataFrame(rows).sort_values("cohort_order", kind="stable")
    failures = result[~result["v2_scored"].astype(bool)]
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "analysis_role": "post_inference_reveal_of_fixed_first_replicate_endpoint",
        "reported_pose_policy": "V2 selected Top-1 from the first replicate",
        "rmsd_definition": "symmetry-corrected heavy-atom RMSD; no alignment",
        "requested_n": len(cohort),
        "scored_n": int(result["v2_scored"].astype(bool).sum()),
        "failures_n": len(failures),
        "top1_lt2A_n": int(result["v2_rmsd_A"].lt(2.0).sum()),
        "top1_lt5A_n": int(result["v2_rmsd_A"].lt(5.0).sum()),
        "inputs": {
            "replicate_root": str(replicate_root),
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
            "reveal_manifest": {
                "path": str(reveal_path),
                "sha256": sha256(reveal_path),
            },
        },
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if len(failures):
        print(failures[["complex_name", "evaluation_error"]].to_string(index=False))
        raise RuntimeError(f"Endpoint evaluation failed for {len(failures)} cases")
    print(f"MULTIRUN_REPORTED_POSE_ENDPOINT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
