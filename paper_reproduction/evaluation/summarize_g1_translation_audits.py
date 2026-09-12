#!/usr/bin/env python3
"""Summarize G1 correction audits and optionally pair final particles to baseline.

No ground truth is read. Pairing uses ``particle_ranking.json`` emitted by
inference so baseline and G1 particles generated with the same seed/noise can
be compared after confidence ranking changes their output rank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_candidate_density_gradient import (
    atomic_numbers,
    molecule_coordinates,
)
from evaluation.evaluate_ligand_accuracy import ranked_predictions, read_heavy_molecule


INDEX = re.compile(r"^index(?P<index>\d+)___(?P<name>.*)$")


def result_directories(root: Path) -> dict[int, Path]:
    directories = {}
    for path in root.glob("index*___*"):
        match = INDEX.match(path.name)
        if path.is_dir() and match:
            index = int(match.group("index"))
            if index in directories:
                raise ValueError(f"Duplicate result index {index} under {root}")
            directories[index] = path
    return directories


def particle_pose_paths(directory: Path) -> dict[int, dict[str, object]]:
    metadata_path = directory / "particle_ranking.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; rerun this condition with the G1 code version"
        )
    ranks = {
        rank: path
        for rank, _confidence, path in ranked_predictions(directory, limit=100000)
    }
    particles = {}
    for row in json.loads(metadata_path.read_text()):
        rank = int(row["rank"])
        if rank not in ranks:
            raise FileNotFoundError(f"No ranked SDF for rank {rank} in {directory}")
        particles[int(row["particle_index"])] = {
            "rank": rank,
            "confidence": float(row["confidence"]),
            "path": ranks[rank],
        }
    return particles


def paired_pose_differences(baseline_path: Path, g1_path: Path) -> dict[str, float]:
    baseline = read_heavy_molecule(baseline_path)
    g1 = read_heavy_molecule(g1_path)
    if not np.array_equal(atomic_numbers(baseline), atomic_numbers(g1)):
        raise ValueError(f"Particle atom ordering differs: {baseline_path} vs {g1_path}")
    baseline_coordinates = molecule_coordinates(baseline).astype(np.float64)
    g1_coordinates = molecule_coordinates(g1).astype(np.float64)
    if baseline_coordinates.shape != g1_coordinates.shape:
        raise ValueError("Paired particle coordinate shapes differ")
    baseline_center = baseline_coordinates.mean(axis=0)
    g1_center = g1_coordinates.mean(axis=0)
    centered_baseline = baseline_coordinates - baseline_center
    centered_g1 = g1_coordinates - g1_center
    return {
        "paired_final_centroid_distance_A": float(
            np.linalg.norm(g1_center - baseline_center)
        ),
        "paired_final_atom_rmsd_A": float(
            np.sqrt(np.mean(np.sum((g1_coordinates - baseline_coordinates) ** 2, axis=1)))
        ),
        "paired_final_centered_rmsd_A": float(
            np.sqrt(np.mean(np.sum((centered_g1 - centered_baseline) ** 2, axis=1)))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-results", required=True, type=Path)
    parser.add_argument("--baseline-results", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    g1_root = args.g1_results.expanduser().resolve()
    baseline_root = (
        args.baseline_results.expanduser().resolve()
        if args.baseline_results is not None else None
    )
    g1_directories = result_directories(g1_root)
    baseline_directories = result_directories(baseline_root) if baseline_root else {}
    if not g1_directories:
        raise FileNotFoundError(f"No G1 result directories under {g1_root}")

    particle_rows = []
    complex_rows = []
    failures = []
    for row_index, g1_directory in sorted(g1_directories.items()):
        complex_name = INDEX.match(g1_directory.name).group("name")
        try:
            audit_path = g1_directory / "density_translation_correction_audit.json"
            if not audit_path.is_file():
                raise FileNotFoundError(audit_path)
            audit = json.loads(audit_path.read_text())
            g1_particles = particle_pose_paths(g1_directory)
            baseline_particles = None
            if baseline_root is not None:
                if row_index not in baseline_directories:
                    raise FileNotFoundError(
                        f"No baseline result directory for row {row_index}"
                    )
                baseline_particles = particle_pose_paths(baseline_directories[row_index])

            current_rows = []
            for particle in audit["particles"]:
                particle_index = int(particle["particle_index"])
                g1_pose = g1_particles[particle_index]
                row = {
                    "row_index": row_index,
                    "complex_name": complex_name,
                    **particle,
                    "g1_final_rank": int(g1_pose["rank"]),
                    "g1_final_confidence": float(g1_pose["confidence"]),
                    "g1_prediction_path": str(g1_pose["path"]),
                }
                if baseline_particles is not None:
                    baseline_pose = baseline_particles[particle_index]
                    row.update({
                        "baseline_final_rank": int(baseline_pose["rank"]),
                        "baseline_final_confidence": float(baseline_pose["confidence"]),
                        "baseline_prediction_path": str(baseline_pose["path"]),
                        **paired_pose_differences(
                            baseline_pose["path"], g1_pose["path"]
                        ),
                    })
                current_rows.append(row)
                particle_rows.append(row)

            particles = pd.DataFrame(current_rows)
            steps = pd.DataFrame(audit["steps"])
            active = particles.loc[~particles["protected"]]
            record = {
                "row_index": row_index,
                "complex_name": complex_name,
                "num_particles": int(len(particles)),
                "num_protected": int(particles["protected"].sum()),
                "num_active_diffusion_steps": int(audit["num_active_diffusion_steps"]),
                "mean_accepted_steps": float(active["accepted_steps"].mean()),
                "median_cumulative_path_A": float(active["cumulative_path_A"].median()),
                "median_net_translation_A": float(active["net_translation_A"].median()),
                "median_cumulative_ncc_gain": float(active["cumulative_ncc_gain"].median()),
                "mean_step_acceptance_pct": float(steps["acceptance_pct_of_active"].mean()),
                "mean_model_translation_perturb_norm_A": float(
                    steps["mean_model_translation_perturb_norm_A"].mean()
                ),
                "mean_correction_to_model_ratio": float(
                    steps["mean_correction_to_model_ratio"].mean()
                ),
            }
            if baseline_particles is not None:
                for column in (
                    "paired_final_centroid_distance_A",
                    "paired_final_atom_rmsd_A",
                    "paired_final_centered_rmsd_A",
                ):
                    record[f"median_{column}"] = float(particles[column].median())
                    record[f"q95_{column}"] = float(particles[column].quantile(0.95))
            complex_rows.append(record)
            print(f"[{row_index}] {complex_name}: {len(particles)} particles OK")
        except Exception as exc:
            failures.append({
                "row_index": row_index, "complex_name": complex_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{row_index}] FAILED {complex_name}: {exc}")
            if args.fail_on_error:
                raise

    if not particle_rows:
        raise RuntimeError("No G1 audits were summarized")
    particles = pd.DataFrame(particle_rows)
    complexes = pd.DataFrame(complex_rows)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    particles.to_csv(output / "per_particle.csv", index=False)
    complexes.to_csv(output / "per_complex.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)

    numeric = [
        column for column in complexes.columns
        if column not in {"row_index", "complex_name"}
    ]
    overall = {
        "g1_results": str(g1_root),
        "baseline_results": str(baseline_root) if baseline_root else None,
        "num_complexes": int(len(complexes)),
        "num_particles": int(len(particles)),
        "num_failures": int(len(failures)),
        "uses_ground_truth": False,
        "metrics": {
            column: {
                "median_across_complexes": float(complexes[column].median()),
                "mean_across_complexes": float(complexes[column].mean()),
            }
            for column in numeric
        },
    }
    (output / "summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    print("\nPer-complex audit")
    print(complexes.to_string(index=False))
    print(f"\nOutputs: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
