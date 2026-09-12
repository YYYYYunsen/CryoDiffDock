#!/usr/bin/env python3
"""Score and rerank DiffDock-Pocket candidates with Gaussian density NCC.

This is a deployment-safe stage: it reads candidate poses and experimental
density crops, but never reads ``reference_ligand`` or any RMSD/ground truth.
Multiple candidate roots can be pooled, e.g. baseline + density-guided poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import (
    fixed_crop_mask,
    load_cartesian_density_crop,
    score_gaussian_density,
    sigma_column,
    zscore,
)


RANKED_SDF = re.compile(r"^rank(?P<rank>\d+)_confidence(?P<confidence>[-+0-9.eE]+)\.sdf$")


def parse_root(specification: str):
    if "=" not in specification:
        raise ValueError("--candidate-root must be NAME=/path/to/results")
    name, value = specification.split("=", 1)
    name, path = name.strip(), Path(value).expanduser().resolve()
    if not name or not path.is_dir():
        raise ValueError(f"Invalid candidate root: {specification}")
    return name, path


def locate_complex(root: Path, row_index: int) -> Path:
    matches = sorted(path for path in root.glob(f"index{row_index}___*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one directory matching index{row_index}___* under {root}, found {len(matches)}"
        )
    return matches[0]


def candidates(directory: Path, limit: int):
    rows = []
    for path in directory.iterdir():
        match = RANKED_SDF.match(path.name)
        if match and int(match.group("rank")) <= limit:
            rows.append((int(match.group("rank")), float(match.group("confidence")), path))
    return sorted(rows, key=lambda item: item[0])


def read_pose(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read pose: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=np.float32)
    return coordinates, atomic_numbers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-root", action="append", required=True,
                        help="Repeatable NAME=/path, e.g. baseline=... and guided=...")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument("--sigmas", default="1.0,1.5,2.0")
    parser.add_argument("--mask-radius", type=float, default=10.0)
    parser.add_argument("--target-floor", type=float, default=None)
    parser.add_argument("--min-mask-voxels", type=int, default=64)
    parser.add_argument("--cutoff-sigma", type=float, default=4.0)
    parser.add_argument("--ranking-sigma", type=float, default=1.5)
    parser.add_argument("--density-weight", type=float, default=0.5)
    parser.add_argument("--materialize-top-k", type=int, default=0)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    sigmas = [float(value) for value in args.sigmas.split(",") if value.strip()]
    if not sigmas or args.ranking_sigma not in sigmas:
        raise ValueError("--ranking-sigma must be present in --sigmas")
    roots = [parse_root(specification) for specification in args.candidate_root]
    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    required = {"complex_name", "ligand_density_crop"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_rows, failures = [], []
    for row_index, row in table.iterrows():
        name = str(row["complex_name"])
        try:
            crop_path = Path(str(row["ligand_density_crop"])).expanduser()
            if not crop_path.is_absolute():
                crop_path = manifest.parent / crop_path
            crop = load_cartesian_density_crop(crop_path)
            mask = fixed_crop_mask(
                crop, radius_A=args.mask_radius, target_floor=args.target_floor,
                min_voxels=args.min_mask_voxels,
            )
            count = 0
            for source, root in roots:
                directory = locate_complex(root, row_index)
                for source_rank, confidence, pose_path in candidates(directory, args.num_predictions):
                    coordinates, atomic_numbers = read_pose(pose_path)
                    record = {
                        "row_index": int(row_index),
                        "complex_name": name,
                        "source": source,
                        "source_rank": int(source_rank),
                        "confidence": float(confidence),
                        "prediction_path": str(pose_path.resolve()),
                        "crop_path": crop.path,
                        "mask_voxels": int(mask.sum()),
                    }
                    for sigma in sigmas:
                        result = score_gaussian_density(
                            crop, coordinates, atomic_numbers, sigma, mask,
                            cutoff_sigma=args.cutoff_sigma,
                        )
                        record[sigma_column("ncc", sigma)] = result.ncc
                        record[sigma_column("cosine", sigma)] = result.cosine_overlap
                        record["ligand_in_bounds_fraction"] = result.ligand_in_bounds_fraction
                    score_rows.append(record)
                    count += 1
            print(f"[{row_index + 1}/{len(table)}] {name}: scored {count} poses")
        except Exception as exc:
            failures.append({
                "row_index": row_index, "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{row_index + 1}/{len(table)}] FAILED {name}: {exc}")
            if args.fail_on_error:
                raise

    if not score_rows:
        raise RuntimeError("No poses were scored")
    scores = pd.DataFrame(score_rows)
    density_column = sigma_column("ncc", args.ranking_sigma)
    ranked_groups = []
    for _, group in scores.groupby(["row_index", "complex_name"], sort=False):
        group = group.copy()
        group["confidence_z"] = zscore(group["confidence"])
        group["density_z"] = zscore(group[density_column])
        group["combined_score"] = (
            group["confidence_z"] + float(args.density_weight) * group["density_z"]
        )
        group = group.sort_values(
            ["combined_score", "confidence", "source", "source_rank"],
            ascending=[False, False, True, True], kind="stable",
        )
        group["combined_rank"] = np.arange(1, len(group) + 1)
        ranked_groups.append(group)
    ranked = pd.concat(ranked_groups, ignore_index=True)
    ranked.to_csv(output / "pose_scores.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)

    if args.materialize_top_k > 0:
        materialized = output / "ranked"
        for (row_index, complex_name), group in ranked.groupby(
            ["row_index", "complex_name"], sort=False
        ):
            destination = materialized / f"index{row_index}___{complex_name}"
            destination.mkdir(parents=True, exist_ok=True)
            selected = group.head(args.materialize_top_k)
            for _, pose in selected.iterrows():
                rank = int(pose["combined_rank"])
                target = destination / f"rank{rank}_confidence{pose['combined_score']:.6f}.sdf"
                shutil.copy2(pose["prediction_path"], target)
            selected.to_csv(destination / "ranking.csv", index=False)

    summary = {
        "manifest": str(manifest),
        "candidate_roots": {name: str(path) for name, path in roots},
        "num_complexes": int(ranked["row_index"].nunique()),
        "num_poses": int(len(ranked)),
        "num_failures": len(failures),
        "sigmas_A": sigmas,
        "mask_radius_A": args.mask_radius,
        "target_floor": args.target_floor,
        "ranking_sigma_A": args.ranking_sigma,
        "density_weight": args.density_weight,
        "uses_ground_truth": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
