#!/usr/bin/env python3
"""Score candidate poses with deployment-safe physical diagnostics.

This script never reads reference ligands or RMSD.  It consumes the
``pose_scores.csv`` produced by ``rerank_ligand_by_density_ncc.py`` and scores
every pose against the input protein from the inference manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.physics import (  # noqa: E402
    paired_protein_for_pose,
    read_protein_heavy_atoms,
    score_ligand_pose_physics,
)


KEY_COLUMNS = ["row_index", "complex_name", "source", "source_rank", "prediction_path"]


def _resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _score_complex(task):
    (
        row_index, complex_name, protein_path, pose_rows, severe_overlap_A,
        calculate_strain, strain_variant, strain_max_iterations,
        score_paired_protein,
    ) = task
    protein_coordinates, protein_elements = read_protein_heavy_atoms(protein_path)
    scored, failures = [], []
    for pose in pose_rows:
        try:
            ligand_path = Path(pose["prediction_path"])
            metrics = score_ligand_pose_physics(
                protein_coordinates,
                protein_elements,
                ligand_path,
                severe_overlap_A=severe_overlap_A,
                calculate_strain=calculate_strain,
                strain_variant=strain_variant,
                strain_max_iterations=strain_max_iterations,
            )
            record = {key: pose[key] for key in KEY_COLUMNS}
            record.update({
                "input_protein": str(protein_path),
                "ligand_in_bounds_fraction": pose.get(
                    "ligand_in_bounds_fraction", float("nan")
                ),
                **{f"input_{key}": value for key, value in metrics.items()},
            })
            paired_path = paired_protein_for_pose(ligand_path)
            record["paired_protein"] = str(paired_path) if paired_path else ""
            record["paired_protein_available"] = bool(paired_path)
            if score_paired_protein and paired_path is not None:
                paired_coordinates, paired_elements = read_protein_heavy_atoms(paired_path)
                paired = score_ligand_pose_physics(
                    paired_coordinates,
                    paired_elements,
                    ligand_path,
                    severe_overlap_A=severe_overlap_A,
                    calculate_strain=False,
                )
                for key in (
                    "severe_clash_count", "num_clashing_ligand_atoms",
                    "clashing_ligand_atom_fraction", "max_vdw_overlap_A",
                    "min_protein_ligand_distance_A",
                ):
                    record[f"paired_{key}"] = paired[key]
            scored.append(record)
        except Exception as exc:
            failures.append({
                "row_index": row_index,
                "complex_name": complex_name,
                "source": pose.get("source", ""),
                "source_rank": pose.get("source_rank", ""),
                "prediction_path": pose.get("prediction_path", ""),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return scored, failures


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--severe-overlap", type=float, default=0.8)
    parser.add_argument("--skip-strain", action="store_true")
    parser.add_argument("--strain-variant", default="MMFF94s")
    parser.add_argument("--strain-max-iterations", type=int, default=500)
    parser.add_argument("--score-paired-protein", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.severe_overlap < 0 or args.strain_max_iterations <= 0:
        raise ValueError("workers/iterations must be positive and overlap non-negative")

    manifest = args.manifest.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    table = pd.read_csv(manifest)
    scores = pd.read_csv(scores_path)
    required_manifest = {"complex_name", "experimental_protein"}
    required_scores = set(KEY_COLUMNS)
    missing = sorted(required_manifest - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    missing = sorted(required_scores - set(scores.columns))
    if missing:
        raise ValueError(f"Scores are missing columns: {missing}")

    manifest_by_index = table.to_dict(orient="index")
    tasks = []
    for (row_index, complex_name), group in scores.groupby(
        ["row_index", "complex_name"], sort=False
    ):
        row_index = int(row_index)
        if row_index not in manifest_by_index:
            raise KeyError(f"No manifest row {row_index}")
        protein_path = _resolve(
            manifest_by_index[row_index]["experimental_protein"], manifest.parent
        )
        pose_rows = []
        for pose in group.to_dict(orient="records"):
            pose["prediction_path"] = str(_resolve(
                pose["prediction_path"], scores_path.parent
            ))
            pose_rows.append(pose)
        tasks.append((
            row_index, str(complex_name), protein_path, pose_rows,
            float(args.severe_overlap), not args.skip_strain,
            args.strain_variant, int(args.strain_max_iterations),
            bool(args.score_paired_protein),
        ))

    scored_rows, failures = [], []
    if args.workers == 1:
        for index, task in enumerate(tasks, start=1):
            scored, failed = _score_complex(task)
            scored_rows.extend(scored)
            failures.extend(failed)
            print(f"[{index}/{len(tasks)}] {task[1]}: {len(scored)} poses OK")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {executor.submit(_score_complex, task): task for task in tasks}
            completed = 0
            for future in as_completed(pending):
                task = pending[future]
                completed += 1
                try:
                    scored, failed = future.result()
                except Exception as exc:
                    scored, failed = [], [{
                        "row_index": task[0], "complex_name": task[1],
                        "source": "", "source_rank": "", "prediction_path": "",
                        "error": f"{type(exc).__name__}: {exc}",
                    }]
                scored_rows.extend(scored)
                failures.extend(failed)
                print(f"[{completed}/{len(tasks)}] {task[1]}: {len(scored)} poses OK")

    if not scored_rows:
        raise RuntimeError("No candidate poses were scored")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(scored_rows).sort_values(
        ["row_index", "source", "source_rank"], kind="stable"
    )
    frame.to_csv(output / "per_pose_physics.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    by_source = {}
    for source, group in frame.groupby("source"):
        strain_available = group["input_strain_available"].astype(bool)
        by_source[str(source)] = {
            "num_poses": int(len(group)),
            "median_severe_clash_count": float(group["input_severe_clash_count"].median()),
            "clash_free_pct": float(100.0 * (group["input_severe_clash_count"] == 0).mean()),
            "median_clashing_ligand_atom_fraction": float(
                group["input_clashing_ligand_atom_fraction"].median()
            ),
            "strain_available_pct": float(100.0 * strain_available.mean()),
            "median_strain_energy_kcal_mol": (
                float(group.loc[strain_available, "input_strain_energy_kcal_mol"].median())
                if strain_available.any() else None
            ),
        }
    summary = {
        "manifest": str(manifest),
        "scores": str(scores_path),
        "num_complexes": int(frame["row_index"].nunique()),
        "num_poses": int(len(frame)),
        "num_failures": len(failures),
        "severe_overlap_A": args.severe_overlap,
        "strain_enabled": not args.skip_strain,
        "strain_variant": args.strain_variant,
        "score_paired_protein": args.score_paired_protein,
        "uses_ground_truth": False,
        "by_source": by_source,
    }
    text = json.dumps(summary, indent=2, default=_json_value) + "\n"
    (output / "summary.json").write_text(text)
    print(text, end="")
    if failures and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
