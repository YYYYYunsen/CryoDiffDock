#!/usr/bin/env python3
"""Score local density/physics support around S0 and coarse pose candidates.

This is a deployment-safe scoring stage: it never reads a reference ligand or
an RMSD.  Existing coarse-orientation candidates are perturbed by small rigid
moves.  A broad density basin should retain NCC and avoid new severe clashes
under those perturbations, whereas an isolated NCC spike should not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
)
from cryodiffdock_pocket.discrete_search import (
    principal_axes,
    rotate_about_point,
)
from cryodiffdock_pocket.physics import clash_metrics, read_protein_heavy_atoms


def comma_floats(value: str) -> list[float]:
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive values")
    return result


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_pose(path: Path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read pose: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    try:
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
    except Exception:
        pass
    return molecule


def perturbations(
    coordinates: np.ndarray,
    rotation_degrees: list[float],
    translation_distances: list[float],
) -> list[tuple[str, float, np.ndarray]]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    centre = coordinates.mean(axis=0)
    result: list[tuple[str, float, np.ndarray]] = []
    for axis_index, axis in enumerate(principal_axes(coordinates)):
        for magnitude in rotation_degrees:
            for sign in (-1.0, 1.0):
                angle = sign * magnitude
                result.append((
                    f"rot_a{axis_index}_{angle:+g}",
                    float(angle),
                    rotate_about_point(coordinates, axis, angle, centre),
                ))
    for axis_index, axis in enumerate(np.eye(3, dtype=np.float64)):
        for magnitude in translation_distances:
            for sign in (-1.0, 1.0):
                distance = sign * magnitude
                result.append((
                    f"tr_a{axis_index}_{distance:+g}",
                    float(distance),
                    coordinates + distance * axis.reshape(1, 3),
                ))
    return result


def shortlist_coarse(
    group: pd.DataFrame,
    top_k: int,
    density_weight: float,
    clash_penalty: float,
) -> pd.DataFrame:
    coarse = group[group["move_type"].astype(str).eq("coarse_orientation")].copy()
    if coarse.empty:
        return coarse
    coarse["_preliminary_score"] = (
        coarse["confidence_z"].astype(float)
        + float(density_weight) * coarse["expanded_density_z"].astype(float)
        - float(clash_penalty) * np.log1p(
            np.maximum(
                coarse["severe_clash_count"].fillna(0).astype(float)
                - coarse["parent_severe_clash_count"].fillna(0).astype(float),
                0.0,
            )
        )
    )
    selected = set(
        coarse.nlargest(top_k, "_preliminary_score").index.astype(int).tolist()
    )
    # Preserve at least the best density-supported orientation from every
    # independent parent.  This costs little and avoids an NCC-rich parent
    # crowding all other binding-mode hypotheses out of the shortlist.
    for _, parent_group in coarse.groupby("parent_id", sort=False):
        selected.add(int(parent_group["_preliminary_score"].idxmax()))
    return coarse.loc[sorted(selected)].drop(columns="_preliminary_score")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--s0-selected", required=True, type=Path)
    parser.add_argument("--s0-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--coarse-top-k", type=int, default=32)
    parser.add_argument("--rotation-degrees", type=comma_floats,
                        default=comma_floats("10,20"))
    parser.add_argument("--translation-distances", type=comma_floats,
                        default=comma_floats("0.25,0.5"))
    parser.add_argument("--density-sigma", type=float, default=1.5)
    parser.add_argument("--mask-radius", type=float, default=10.0)
    parser.add_argument("--target-floor", type=float, default=None)
    parser.add_argument("--min-mask-voxels", type=int, default=64)
    parser.add_argument("--cutoff-sigma", type=float, default=4.0)
    parser.add_argument("--ncc-tolerance", type=float, default=0.01)
    parser.add_argument("--severe-overlap", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.coarse_top_k <= 0:
        raise ValueError("coarse-top-k must be positive")
    search_root = args.search_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        (search_root / "evaluation_manifest.csv")
        if args.manifest is None else args.manifest.expanduser().resolve()
    )
    manifest = pd.read_csv(manifest_path)
    s0 = pd.read_csv(args.s0_selected.expanduser().resolve())
    s0_config = json.loads(args.s0_config.expanduser().resolve().read_text())
    density_weight = float(s0_config["density_weight"])
    clash_penalty = float(s0_config.get("clash_penalty", 0.0))

    ranking_files = sorted((search_root / "expanded_pool").glob("index*___*/ranking.csv"))
    if not ranking_files:
        raise FileNotFoundError(f"No expanded rankings under {search_root}")
    rankings = pd.concat([pd.read_csv(path) for path in ranking_files], ignore_index=True)
    required = {
        "row_index", "complex_name", "candidate_id", "materialized_rank",
        "materialized_path", "move_type", "parent_id", "confidence_z",
        "expanded_density_z", "ncc", "ncc_gain", "in_bounds_fraction",
        "centroid_shift_A", "severe_clash_count",
        "parent_severe_clash_count", "relative_physics_pass", "protein_path",
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError(f"Expanded rankings are missing columns: {missing}")
    if "selection_score" not in s0.columns:
        raise ValueError("S0 selected table must contain selection_score")

    # The source manifest may contain a preprocessing/inference failure (the
    # current val100 run has 99 completed cases).  Iterate the paired result
    # indices rather than failing on a manifest row that has no pose pool.
    available_rows = sorted(
        set(rankings["row_index"].astype(int))
        & set(s0["row_index"].astype(int))
    )
    if args.limit is not None:
        available_rows = available_rows[: args.limit]

    summaries: list[dict] = []
    neighbor_rows: list[dict] = []
    failures: list[dict] = []
    protein_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    for local_position, row_index in enumerate(available_rows, start=1):
        if row_index < 0 or row_index >= len(manifest):
            raise IndexError(f"Result row index {row_index} is outside the manifest")
        manifest_row = manifest.iloc[row_index]
        try:
            group = rankings[rankings["row_index"].astype(int).eq(int(row_index))].copy()
            s0_group = s0[s0["row_index"].astype(int).eq(int(row_index))]
            if group.empty or len(s0_group) != 1:
                raise ValueError(
                    f"Expected candidates and one S0 row for row {row_index}; "
                    f"found {len(group)} and {len(s0_group)}"
                )
            s0_row = s0_group.iloc[0]
            expected_name = str(manifest_row["complex_name"])
            ranking_names = set(group["complex_name"].astype(str))
            s0_name = str(s0_row["complex_name"])
            if ranking_names != {expected_name} or s0_name != expected_name:
                raise ValueError(
                    "Complex-name mismatch at row "
                    f"{row_index}: manifest={expected_name!r}, "
                    f"rankings={sorted(ranking_names)!r}, s0={s0_name!r}. "
                    "SEARCH_ROOT and S0_ROOT are not the same evaluated cohort."
                )
            fallback = group[group["candidate_id"].astype(str).eq(str(s0_row["candidate_id"]))]
            if len(fallback) != 1:
                raise ValueError(f"Could not locate S0 candidate {s0_row['candidate_id']}")
            centers = pd.concat(
                [fallback.assign(candidate_role="s0"),
                 shortlist_coarse(
                     group, args.coarse_top_k, density_weight, clash_penalty
                 ).assign(candidate_role="coarse")],
                ignore_index=True,
            ).drop_duplicates("candidate_id", keep="first")

            original_ncc = group[as_bool(group.get("is_original", pd.Series(False, index=group.index)))]["ncc"].astype(float)
            if original_ncc.empty:
                raise ValueError("No original candidates available for NCC calibration")
            ncc_mean = float(original_ncc.mean())
            ncc_std = float(original_ncc.std(ddof=0))
            if not np.isfinite(ncc_std) or ncc_std <= 1e-12:
                ncc_std = 1.0

            crop_column = "ligand_density_crop"
            crop_path = resolve_path(manifest_row[crop_column], manifest_path.parent)
            crop = load_cartesian_density_crop(crop_path)
            mask = fixed_crop_mask(
                crop, radius_A=args.mask_radius, target_floor=args.target_floor,
                min_voxels=args.min_mask_voxels,
            )

            for _, center_row in centers.iterrows():
                pose_path = resolve_path(center_row["materialized_path"], search_root)
                molecule = read_pose(pose_path)
                xyz = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float64)
                atomic_numbers = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
                ligand_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
                protein_path = resolve_path(center_row["protein_path"], search_root)
                protein_key = str(protein_path)
                if protein_key not in protein_cache:
                    protein_cache[protein_key] = read_protein_heavy_atoms(protein_path)
                protein_xyz, protein_elements = protein_cache[protein_key]
                center_clashes = float(center_row["severe_clash_count"])
                if not np.isfinite(center_clashes):
                    center_clashes = float(clash_metrics(
                        protein_xyz, protein_elements, xyz, ligand_elements,
                        severe_overlap_A=args.severe_overlap,
                    )["severe_clash_count"])

                local_ncc = []
                local_clashes = []
                for perturbation_id, magnitude, perturbed in perturbations(
                    xyz, args.rotation_degrees, args.translation_distances
                ):
                    density = score_gaussian_density(
                        crop, perturbed, atomic_numbers, args.density_sigma, mask,
                        cutoff_sigma=args.cutoff_sigma,
                    )
                    physics = clash_metrics(
                        protein_xyz, protein_elements, perturbed, ligand_elements,
                        severe_overlap_A=args.severe_overlap,
                    )
                    value = float(density.ncc)
                    clashes = float(physics["severe_clash_count"])
                    local_ncc.append(value)
                    local_clashes.append(clashes)
                    neighbor_rows.append({
                        "row_index": int(row_index),
                        "complex_name": str(center_row["complex_name"]),
                        "candidate_id": str(center_row["candidate_id"]),
                        "candidate_role": str(center_row["candidate_role"]),
                        "perturbation_id": perturbation_id,
                        "magnitude": magnitude,
                        "ncc": value,
                        "ncc_z": (value - ncc_mean) / ncc_std,
                        "severe_clash_count": clashes,
                        "in_bounds_fraction": float(density.ligand_in_bounds_fraction),
                    })

                ncc_values = np.asarray(local_ncc, dtype=float)
                ncc_z = (ncc_values - ncc_mean) / ncc_std
                clash_values = np.asarray(local_clashes, dtype=float)
                center_ncc = float(center_row["ncc"])
                summary = center_row.to_dict()
                summary.update({
                    "candidate_role": str(center_row["candidate_role"]),
                    "s0_selection_score": float(s0_row["selection_score"]),
                    "ncc_calibration_mean": ncc_mean,
                    "ncc_calibration_std": ncc_std,
                    "num_neighbors": int(len(ncc_values)),
                    "neighbor_ncc_mean": float(np.mean(ncc_values)),
                    "neighbor_ncc_median": float(np.median(ncc_values)),
                    "neighbor_ncc_q25": float(np.quantile(ncc_values, 0.25)),
                    "neighbor_ncc_min": float(np.min(ncc_values)),
                    "neighbor_ncc_max": float(np.max(ncc_values)),
                    "neighbor_ncc_z_mean": float(np.mean(ncc_z)),
                    "neighbor_ncc_z_median": float(np.median(ncc_z)),
                    "neighbor_ncc_z_q25": float(np.quantile(ncc_z, 0.25)),
                    "neighbor_ncc_z_min": float(np.min(ncc_z)),
                    "ncc_support_fraction": float(np.mean(
                        ncc_values >= center_ncc - args.ncc_tolerance
                    )),
                    "median_ncc_drop": float(center_ncc - np.median(ncc_values)),
                    "q25_ncc_drop": float(center_ncc - np.quantile(ncc_values, 0.25)),
                    "neighbor_physics_pass_fraction": float(np.mean(
                        clash_values <= center_clashes
                    )),
                    "neighbor_in_bounds_fraction": float(np.mean([
                        row["in_bounds_fraction"] for row in neighbor_rows[-len(ncc_values):]
                    ])),
                })
                summaries.append(summary)
            print(
                f"[{local_position}/{len(available_rows)}] {manifest_row['complex_name']}: "
                f"{len(centers)} basin centres OK"
            )
        except Exception as exc:
            failures.append({
                "row_index": int(row_index),
                "complex_name": str(manifest_row.get("complex_name", row_index)),
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{local_position}/{len(available_rows)}] FAILED: {exc}")
            if args.fail_on_error:
                raise

    pd.DataFrame(summaries).to_csv(output / "basin_scores.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(output / "neighbor_scores.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    summary = {
        "num_complexes": int(pd.DataFrame(summaries)["row_index"].nunique()) if summaries else 0,
        "num_basin_centres": len(summaries),
        "num_neighbor_scores": len(neighbor_rows),
        "num_failures": len(failures),
        "coarse_top_k": args.coarse_top_k,
        "shortlist_density_weight": density_weight,
        "shortlist_clash_penalty": clash_penalty,
        "rotation_degrees": args.rotation_degrees,
        "translation_distances_A": args.translation_distances,
        "ncc_tolerance": args.ncc_tolerance,
        "uses_ground_truth": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
