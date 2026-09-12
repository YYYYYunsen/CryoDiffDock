#!/usr/bin/env python3
"""Evaluate one confidence-ranked pose per cohort member against references."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(task: tuple[str, str, str]) -> dict[str, object]:
    pdb_id, pose_root_text, reference_text = task
    pose_root, reference = Path(pose_root_text), Path(reference_text)
    matches = sorted((pose_root / pdb_id).glob("rank1_confidence*.sdf"))
    row: dict[str, object] = {
        "pdb_id": pdb_id, "top1_rmsd_A": np.nan, "status": "",
        "prediction_path": str(matches[0]) if len(matches) == 1 else "",
        "reference_ligand": str(reference), "evaluation_error": "",
    }
    if not matches:
        row["status"] = "missing_prediction_counted_as_failure"
        return row
    if len(matches) != 1:
        row["status"] = "evaluation_error"
        row["evaluation_error"] = f"Expected one rank1 pose, found {len(matches)}"
        return row
    if not reference.is_file():
        row["status"] = "evaluation_error"
        row["evaluation_error"] = "Missing reference ligand"
        return row
    try:
        from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd
        value = symmetry_rmsd(read_heavy_molecule(reference), read_heavy_molecule(matches[0]))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid RMSD: {value}")
        row["top1_rmsd_A"] = float(value)
        row["status"] = "scored"
    except Exception as exc:
        row["status"] = "evaluation_error"
        row["evaluation_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path)
    references = parser.add_mutually_exclusive_group(required=True)
    references.add_argument(
        "--reference-manifest", type=Path,
        help="CSV/TSV with parent ID and reference_ligand path; preferred.",
    )
    references.add_argument(
        "--reference-root", type=Path,
        help="Directory containing direct {PDB}_ligand_deposited.sdf files.",
    )
    parser.add_argument("--reference-column", default="reference_ligand")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    ids = [
        line.split()[0].strip().upper() for line in args.cohort.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != args.expected_n or len(ids) != len(set(ids)):
        raise ValueError(f"Expected {args.expected_n} unique cohort IDs; found {len(ids)}")
    if args.workers < 1 or args.output.exists():
        raise ValueError("Use positive --workers and a new --output path")
    if args.reference_manifest is not None:
        manifest_path = args.reference_manifest.expanduser().resolve()
        manifest = pd.read_csv(manifest_path, sep=None, engine="python")
        id_column = next(
            (column for column in ("pdb_id", "parent_complex_name", "complex_name") if column in manifest),
            None,
        )
        if id_column is None or args.reference_column not in manifest:
            raise ValueError(
                f"Reference manifest requires parent ID and {args.reference_column}: "
                f"{manifest.columns.tolist()}"
            )
        manifest["_pdb_id"] = manifest[id_column].astype(str).str.strip().str.upper()
        if id_column == "complex_name":
            manifest["_pdb_id"] = manifest._pdb_id.str.replace(
                r"__GLOBAL_RANK\d+$", "", regex=True
            )
        manifest = manifest[manifest._pdb_id.isin(ids)].copy()
        manifest["_reference"] = manifest[args.reference_column].map(
            lambda value: Path(str(value)).expanduser()
        )
        manifest["_reference"] = manifest._reference.map(
            lambda path: (path if path.is_absolute() else manifest_path.parent / path).resolve()
        )
        conflicts = manifest.groupby("_pdb_id")._reference.nunique()
        if conflicts.gt(1).any():
            raise ValueError(
                f"Conflicting references: {conflicts[conflicts.gt(1)].index.tolist()}"
            )
        reference_map = (
            manifest.drop_duplicates("_pdb_id").set_index("_pdb_id")._reference.to_dict()
        )
        missing_references = [pdb_id for pdb_id in ids if pdb_id not in reference_map]
        if missing_references:
            raise ValueError(f"Reference manifest lacks cohort IDs: {missing_references}")
    else:
        reference_root = args.reference_root.expanduser().resolve()
        reference_map = {
            pdb_id: reference_root / f"{pdb_id}_ligand_deposited.sdf" for pdb_id in ids
        }
    tasks = [
        (pdb_id, str(args.poses.resolve()), str(reference_map[pdb_id])) for pdb_id in ids
    ]
    if args.workers == 1:
        rows = [evaluate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(evaluate, tasks))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    status = frame.status.value_counts().to_dict()
    summary = {
        "cohort_n": len(frame),
        "scored_n": int(frame.top1_rmsd_A.notna().sum()),
        "missing_n": int(frame.top1_rmsd_A.isna().sum()),
        "status_counts": status,
        "success_lt2_n": int(frame.top1_rmsd_A.lt(2.0).sum()),
        "success_lt5_n": int(frame.top1_rmsd_A.lt(5.0).sum()),
        "denominator": len(frame),
        "missing_counts_as_failure": True,
        "rmsd_definition": "symmetry-corrected heavy-atom RMSD; center=False; minimize=False",
        "reference_source": str(
            args.reference_manifest.resolve()
            if args.reference_manifest is not None else args.reference_root.resolve()
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    failures = frame[frame.status.eq("evaluation_error")]
    if len(failures):
        print(failures[["pdb_id", "prediction_path", "evaluation_error"]].to_string(index=False))
        raise RuntimeError(f"RMSD evaluation errors: {len(failures)}")
    print(f"RANK1_POSE_ROOT_EVALUATION_OK: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
