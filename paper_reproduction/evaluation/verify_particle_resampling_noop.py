from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two seeded sampling runs particle-by-particle using "
            "particle_ranking.json, independent of confidence-bearing filenames."
        )
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-4)
    parser.add_argument("--confidence-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def ranking_by_particle(directory: Path):
    path = directory / "particle_ranking.json"
    rows = json.loads(path.read_text())
    result = {}
    for row in rows:
        particle = int(row["particle_index"])
        if particle in result:
            raise ValueError(f"duplicate particle index {particle} in {path}")
        result[particle] = {
            "rank": int(row["rank"]),
            "confidence": float(row["confidence"]),
        }
    return result


def ranked_pose_path(directory: Path, rank: int) -> Path:
    paths = list(directory.glob(f"rank{rank}_confidence*.sdf"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one confidence SDF for rank {rank} in {directory}, "
            f"found {len(paths)}"
        )
    return paths[0]


def coordinates(path: Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"could not read coordinates from {path}")
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)


def compare_complex(baseline_dir: Path, control_dir: Path):
    baseline_ranking = ranking_by_particle(baseline_dir)
    control_ranking = ranking_by_particle(control_dir)
    if set(baseline_ranking) != set(control_ranking):
        raise ValueError("particle index sets differ")
    coordinate_rmsds = []
    confidence_deltas = []
    for particle in sorted(baseline_ranking):
        baseline_row = baseline_ranking[particle]
        control_row = control_ranking[particle]
        baseline_coordinates = coordinates(
            ranked_pose_path(baseline_dir, baseline_row["rank"])
        )
        control_coordinates = coordinates(
            ranked_pose_path(control_dir, control_row["rank"])
        )
        if baseline_coordinates.shape != control_coordinates.shape:
            raise ValueError(f"particle {particle} atom counts differ")
        difference = baseline_coordinates - control_coordinates
        coordinate_rmsds.append(float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1)))))
        confidence_deltas.append(abs(
            baseline_row["confidence"] - control_row["confidence"]
        ))
    return {
        "complex_name": baseline_dir.name.split("___", 1)[-1],
        "num_particles": len(coordinate_rmsds),
        "median_particle_raw_rmsd_A": float(np.median(coordinate_rmsds)),
        "mean_particle_raw_rmsd_A": float(np.mean(coordinate_rmsds)),
        "max_particle_raw_rmsd_A": float(np.max(coordinate_rmsds)),
        "num_coordinate_identical_particles": int(
            np.count_nonzero(np.asarray(coordinate_rmsds) <= 1e-12)
        ),
        "max_confidence_abs_delta": float(np.max(confidence_deltas)),
    }


def main() -> int:
    args = parse_args()
    if args.coordinate_tolerance < 0 or args.confidence_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    baseline = args.baseline.expanduser().resolve()
    control = args.control.expanduser().resolve()
    failures = []
    rows = []
    directories = sorted(path for path in baseline.glob("index*___*") if path.is_dir())
    for baseline_dir in directories:
        control_dir = control / baseline_dir.name
        if not control_dir.is_dir():
            failures.append(f"missing control directory: {control_dir}")
            continue
        try:
            row = compare_complex(baseline_dir, control_dir)
            row["coordinates_within_tolerance"] = bool(
                row["max_particle_raw_rmsd_A"] <= args.coordinate_tolerance
            )
            row["confidence_within_tolerance"] = bool(
                row["max_confidence_abs_delta"] <= args.confidence_tolerance
            )
            rows.append(row)
            if not row["coordinates_within_tolerance"]:
                failures.append(
                    f"coordinate mismatch for {baseline_dir.name}: "
                    f"max raw RMSD {row['max_particle_raw_rmsd_A']:.6g} A"
                )
            if not row["confidence_within_tolerance"]:
                failures.append(
                    f"confidence mismatch for {baseline_dir.name}: "
                    f"max abs delta {row['max_confidence_abs_delta']:.6g}"
                )
        except Exception as exc:
            failures.append(f"{baseline_dir.name}: {type(exc).__name__}: {exc}")
    summary = {
        "num_complexes": len(directories),
        "num_compared_complexes": len(rows),
        "num_compared_particles": int(sum(row["num_particles"] for row in rows)),
        "coordinate_tolerance_A": args.coordinate_tolerance,
        "confidence_tolerance": args.confidence_tolerance,
        "max_particle_raw_rmsd_A": max(
            (row["max_particle_raw_rmsd_A"] for row in rows), default=None
        ),
        "max_confidence_abs_delta": max(
            (row["max_confidence_abs_delta"] for row in rows), default=None
        ),
        "num_failures": len(failures),
        "exact_noop": bool(directories) and not failures,
        "per_complex": rows,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["exact_noop"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
