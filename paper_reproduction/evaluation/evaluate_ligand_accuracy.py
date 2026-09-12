#!/usr/bin/env python3
"""Evaluate ranked DiffDock-Pocket SDFs against reference ligands by manifest."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


RANKED_SDF = re.compile(r"^rank(?P<rank>\d+)_confidence(?P<confidence>[-+0-9.eE]+)\.sdf$")


def resolve_path(value, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def locate_result_directory(results_root: Path, row_index: int) -> Path:
    matches = sorted(path for path in results_root.glob(f"index{row_index}___*") if path.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one result directory matching index{row_index}___* under "
            f"{results_root}, found {len(matches)}"
        )
    return matches[0]


def ranked_predictions(result_directory: Path, limit: int):
    predictions = []
    for path in result_directory.iterdir():
        match = RANKED_SDF.match(path.name)
        if match:
            rank = int(match.group("rank"))
            if rank <= limit:
                predictions.append((rank, float(match.group("confidence")), path))
    predictions.sort(key=lambda item: item[0])
    if not predictions:
        alias = result_directory / "rank1.sdf"
        if alias.is_file():
            predictions = [(1, float("nan"), alias)]
    return predictions


def read_heavy_molecule(path: Path):
    from rdkit import Chem

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"Unsupported ligand format: {path}")
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand: {path}")
    return heavy_atom_copy(mol)


def heavy_atom_copy(mol):
    """Induced non-hydrogen graph, preserving heavy-atom order/coordinates.

    RemoveHs defaults can retain special hydrogens (e.g. isotopic/stereo H).
    RMSD here is explicitly heavy-atom RMSD, so remove every atomic-number-1
    node in memory, irrespective of its annotations. Do not sanitize, align,
    optimize, or write back to the deposited structure.
    """
    from rdkit import Chem

    editable = Chem.RWMol(mol)
    for index in reversed(range(editable.GetNumAtoms())):
        if editable.GetAtomWithIdx(index).GetAtomicNum() == 1:
            editable.RemoveAtom(index)
    result = editable.GetMol()
    if result.GetNumAtoms() == 0:
        raise ValueError("Ligand contains no heavy atoms")
    return result


def symmetry_rmsd(reference, prediction, *, center: bool = False, minimize: bool = False) -> float:
    """Return symmetry-corrected RMSD with optional rigid-body removal.

    ``center=True`` removes translation.  ``minimize=True`` additionally
    superposes the molecules by the optimal rigid rotation, leaving primarily
    internal-conformation (torsion) differences.
    """
    from spyrmsd import molecule, rmsd

    reference = heavy_atom_copy(reference)
    prediction = heavy_atom_copy(prediction)
    reference_elements = sorted(atom.GetAtomicNum() for atom in reference.GetAtoms())
    prediction_elements = sorted(atom.GetAtomicNum() for atom in prediction.GetAtoms())
    if reference_elements != prediction_elements:
        raise ValueError(
            "Heavy-atom composition mismatch after excluding all hydrogens: "
            f"reference={reference_elements}, prediction={prediction_elements}"
        )
    reference_spy = molecule.Molecule.from_rdkit(reference)
    prediction_spy = molecule.Molecule.from_rdkit(prediction)
    value = rmsd.symmrmsd(
        np.asarray(reference.GetConformer().GetPositions()),
        np.asarray(prediction.GetConformer().GetPositions()),
        reference_spy.atomicnums,
        prediction_spy.atomicnums,
        reference_spy.adjacency_matrix,
        prediction_spy.adjacency_matrix,
        center=center,
        minimize=minimize,
    )
    return float(np.asarray(value).reshape(-1)[0])


def centroid_distance(reference, prediction) -> float:
    reference_center = np.asarray(reference.GetConformer().GetPositions()).mean(axis=0)
    prediction_center = np.asarray(prediction.GetConformer().GetPositions()).mean(axis=0)
    return float(np.linalg.norm(reference_center - prediction_center))


def evaluate_complex(task):
    """Evaluate one manifest row.

    This top-level worker is intentionally process-picklable.  A complete
    complex is handled by one process so RDKit molecules never need to cross
    process boundaries and result-directory indices retain their original
    manifest meaning.
    """
    (
        row_index,
        row,
        manifest_parent,
        results_root,
        condition,
        num_predictions,
        total_complexes,
    ) = task
    complex_name = str(row["complex_name"])
    try:
        manifest_parent = Path(manifest_parent)
        results_root = Path(results_root)
        reference_path = resolve_path(row["reference_ligand"], manifest_parent)
        reference = read_heavy_molecule(reference_path)
        result_directory = locate_result_directory(results_root, row_index)
        predictions = ranked_predictions(result_directory, num_predictions)
        if not predictions:
            raise FileNotFoundError(f"No ranked SDF predictions in {result_directory}")

        pose_rows = []
        pose_metrics = []
        for rank, confidence, prediction_path in predictions:
            prediction = read_heavy_molecule(prediction_path)
            pose_rmsd = symmetry_rmsd(reference, prediction)
            center_distance = centroid_distance(reference, prediction)
            centered_rmsd = symmetry_rmsd(reference, prediction, center=True)
            aligned_rmsd = symmetry_rmsd(
                reference, prediction, center=True, minimize=True
            )
            metrics = {
                "ligand_rmsd_A": pose_rmsd,
                "centroid_distance_A": center_distance,
                "centered_rmsd_A": centered_rmsd,
                "rigid_aligned_rmsd_A": aligned_rmsd,
            }
            pose_metrics.append(metrics)
            pose_rows.append({
                "condition": condition,
                "row_index": row_index,
                "complex_name": complex_name,
                "rank": rank,
                "confidence": confidence,
                **metrics,
                "prediction_path": str(prediction_path),
                "reference_ligand": str(reference_path),
            })

        top1_metrics = pose_metrics[0]
        best_index = int(np.argmin([item["ligand_rmsd_A"] for item in pose_metrics]))
        best_metrics = pose_metrics[best_index]
        top1 = float(top1_metrics["ligand_rmsd_A"])
        best = float(best_metrics["ligand_rmsd_A"])
        complex_row = {
            "condition": condition,
            "row_index": row_index,
            "complex_name": complex_name,
            "num_predictions": len(pose_metrics),
            "top1_rmsd_A": top1,
            "top1_centroid_distance_A": float(top1_metrics["centroid_distance_A"]),
            "top1_centered_rmsd_A": float(top1_metrics["centered_rmsd_A"]),
            "top1_rigid_aligned_rmsd_A": float(top1_metrics["rigid_aligned_rmsd_A"]),
            "best_of_n_rmsd_A": best,
            "best_of_n_rank": int(predictions[best_index][0]),
            "best_rmsd_centroid_distance_A": float(best_metrics["centroid_distance_A"]),
            "best_rmsd_centered_rmsd_A": float(best_metrics["centered_rmsd_A"]),
            "best_rmsd_rigid_aligned_rmsd_A": float(best_metrics["rigid_aligned_rmsd_A"]),
            "best_of_n_centroid_distance_A": float(min(
                item["centroid_distance_A"] for item in pose_metrics
            )),
            "top1_lt2A": top1 < 2.0,
            "top1_lt5A": top1 < 5.0,
            "best_of_n_lt2A": best < 2.0,
            "best_of_n_lt5A": best < 5.0,
        }
        message = (
            f"[{row_index + 1}/{total_complexes}] {complex_name} "
            f"top1={top1:.3f} "
            f"(centroid={top1_metrics['centroid_distance_A']:.3f}, "
            f"centered={top1_metrics['centered_rmsd_A']:.3f}, "
            f"aligned={top1_metrics['rigid_aligned_rmsd_A']:.3f}) "
            f"best={best:.3f} "
            f"(centroid={best_metrics['centroid_distance_A']:.3f}, "
            f"centered={best_metrics['centered_rmsd_A']:.3f}, "
            f"aligned={best_metrics['rigid_aligned_rmsd_A']:.3f})"
        )
        return pose_rows, complex_row, None, message
    except Exception as exc:
        failure = {
            "row_index": row_index,
            "complex_name": complex_name,
            "error": f"{type(exc).__name__}: {exc}",
        }
        message = f"[{row_index + 1}/{total_complexes}] FAILED {complex_name}: {exc}"
        return [], None, failure, message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Complex-level worker processes; 1 preserves serial execution.",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.num_predictions <= 0:
        raise ValueError("num-predictions must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    manifest = args.manifest.expanduser().resolve()
    results_root = args.results.expanduser().resolve()
    table = pd.read_csv(manifest)
    if "reference_ligand" not in table.columns:
        raise ValueError("Manifest requires a reference_ligand column")

    tasks = [
        (
            int(row_index),
            row.to_dict(),
            str(manifest.parent),
            str(results_root),
            args.condition,
            args.num_predictions,
            len(table),
        )
        for row_index, row in table.iterrows()
    ]
    if args.workers == 1:
        results = map(evaluate_complex, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(evaluate_complex, tasks, chunksize=1)

    pose_rows = []
    complex_rows = []
    failures = []
    try:
        for task_pose_rows, complex_row, failure, message in results:
            print(message, flush=True)
            pose_rows.extend(task_pose_rows)
            if complex_row is not None:
                complex_rows.append(complex_row)
            if failure is not None:
                failures.append(failure)
                if args.fail_on_error:
                    raise RuntimeError(failure["error"])
    finally:
        if args.workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)

    if not complex_rows:
        raise RuntimeError("No complexes were evaluated")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pose_frame = pd.DataFrame(pose_rows).sort_values(["row_index", "rank"])
    pose_frame.to_csv(output_dir / "per_pose.csv", index=False)
    per_complex = pd.DataFrame(complex_rows).sort_values("row_index")
    per_complex.to_csv(output_dir / "per_complex.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "failures.csv", index=False)
    summary = {
        "condition": args.condition,
        "num_complexes": len(per_complex),
        "num_failures": len(failures),
        "median_top1_rmsd_A": float(per_complex["top1_rmsd_A"].median()),
        "median_top1_centroid_distance_A": float(per_complex["top1_centroid_distance_A"].median()),
        "median_top1_centered_rmsd_A": float(per_complex["top1_centered_rmsd_A"].median()),
        "median_top1_rigid_aligned_rmsd_A": float(per_complex["top1_rigid_aligned_rmsd_A"].median()),
        "median_best_of_n_rmsd_A": float(per_complex["best_of_n_rmsd_A"].median()),
        "median_best_rmsd_centroid_distance_A": float(per_complex["best_rmsd_centroid_distance_A"].median()),
        "median_best_rmsd_centered_rmsd_A": float(per_complex["best_rmsd_centered_rmsd_A"].median()),
        "median_best_rmsd_rigid_aligned_rmsd_A": float(per_complex["best_rmsd_rigid_aligned_rmsd_A"].median()),
        "median_best_of_n_centroid_distance_A": float(per_complex["best_of_n_centroid_distance_A"].median()),
        "top1_lt2A_pct": float(100.0 * per_complex["top1_lt2A"].mean()),
        "top1_lt5A_pct": float(100.0 * per_complex["top1_lt5A"].mean()),
        "best_of_n_lt2A_pct": float(100.0 * per_complex["best_of_n_lt2A"].mean()),
        "best_of_n_lt5A_pct": float(100.0 * per_complex["best_of_n_lt5A"].mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
