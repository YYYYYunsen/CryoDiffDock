#!/usr/bin/env python3
"""Prepare fixed-size experimental-density crops centered on native ligands.

This is an oracle-pocket/local-docking preparation tool. Reference ligand
coordinates are used only to compute the crop center and QC coverage. Density
values always come from the experimental map. Protein masking uses only
polymer protein heavy atoms and never the reference ligand shape.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_preparation import (
    apply_protein_mask,
    read_modelled_heavy_atoms,
    subtract_model_density,
    cartesian_axes,
    fraction_in_crop,
    hard_protein_mask,
    normalization,
    read_ligand_heavy_coordinates,
    read_protein_heavy_coordinates,
    read_full_polymer_heavy_coordinates,
    sample_map,
    soft_protein_mask,
)


REQUIRED_COLUMNS = {
    "complex_name", "experimental_protein", "ligand", "reference_ligand", "map_path",
    "flexible_sidechains",
}


def resolve_input(value, manifest_directory: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_directory / path
    return path.resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "complex"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--box-size", type=float, default=28.0)
    parser.add_argument("--spacing", type=float, default=0.75)
    parser.add_argument("--mask-inner", type=float, default=1.5)
    parser.add_argument("--mask-outer", type=float, default=2.5)
    parser.add_argument(
        "--mask-policy", choices=("legacy_cropped_protein", "full_polymer"),
        default="legacy_cropped_protein",
    )
    parser.add_argument("--mask-structure-column", default="mask_structure")
    parser.add_argument("--mask-style", choices=("soft", "hard"), default="soft")
    parser.add_argument("--mask-radius", type=float, default=3.5)
    parser.add_argument("--no-protein-mask", action="store_true")
    parser.add_argument(
        "--protein-subtraction", choices=("none", "all_modelled", "backbone_cb"),
        default="none",
        help="Remove simulated density for everything modelled except the docking "
             "target, instead of masking it. 'backbone_cb' subtracts only backbone "
             "and C-beta atoms, which is robust to side-chain modelling error.",
    )
    parser.add_argument(
        "--subtraction-ligand-exclusion-radius", type=float, default=6.0,
        help="Non-polymer residues whose centroid lies within this radius of the "
             "crop centre are never subtracted, so the target ligand survives.",
    )
    parser.add_argument(
        "--subtraction-fit-exclusion-radius", type=float, default=8.0,
        help="Scale and width are fitted only outside this radius of the crop "
             "centre, so ligand density cannot bias the fit.",
    )
    parser.add_argument(
        "--subtraction-sigma-candidates", default="0.8,1.0,1.2,1.5,1.8,2.1,2.5",
        help="Candidate Gaussian widths in Angstrom. The best-correlating value is "
             "kept and recorded as a per-map resolution proxy.",
    )
    parser.add_argument(
        "--subtraction-exclude-resnames", default="",
        help="Comma-separated residue names never to subtract.",
    )
    parser.add_argument("--keep-negative-density", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.box_size <= 0 or args.spacing <= 0 or args.mask_radius <= 0:
        raise ValueError("box-size and spacing must be positive")
    subtraction_sigmas = tuple(
        float(value) for value in args.subtraction_sigma_candidates.split(",") if value.strip()
    )
    if args.protein_subtraction != "none":
        if not args.no_protein_mask:
            raise ValueError(
                "--protein-subtraction replaces protein masking; pass "
                "--no-protein-mask to make that explicit"
            )
        if not subtraction_sigmas or any(value <= 0 for value in subtraction_sigmas):
            raise ValueError("--subtraction-sigma-candidates must be positive")

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if args.mask_policy == "full_polymer" and args.mask_structure_column not in table.columns:
        raise ValueError(
            f"full_polymer requires manifest column {args.mask_structure_column!r}; "
            "add complete deposited PDB/mmCIF paths before preparation"
        )

    output_dir = args.output_dir.expanduser().resolve()
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = []
    report_rows = []

    for row_index, row in table.iterrows():
        complex_name = str(row["complex_name"])
        try:
            flexible_sidechains = str(row["flexible_sidechains"]).strip()
            if not flexible_sidechains or flexible_sidechains.lower() in {"nan", "none"}:
                raise ValueError(
                    "flexible_sidechains must be explicit (for example A:10-A:14); "
                    "automatic selection may read the ligand input coordinates"
                )
            protein_path = resolve_input(row["experimental_protein"], manifest.parent)
            ligand_path = resolve_input(row["ligand"], manifest.parent)
            reference_path = resolve_input(row["reference_ligand"], manifest.parent)
            map_path = resolve_input(row["map_path"], manifest.parent)
            mask_structure_path = protein_path
            if args.mask_policy == "full_polymer":
                value = str(row[args.mask_structure_column]).strip()
                if not value or value.lower() in {"nan", "none"}:
                    raise ValueError(
                        f"Empty {args.mask_structure_column!r} for {complex_name}"
                    )
                mask_structure_path = resolve_input(value, manifest.parent)
            for path in (
                protein_path, ligand_path, reference_path, map_path,
                mask_structure_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(path)

            reference_coordinates = read_ligand_heavy_coordinates(reference_path)
            center = reference_coordinates.mean(axis=0).astype(np.float32)
            origin, axes = cartesian_axes(center, args.box_size, args.spacing)
            sampled = sample_map(map_path, axes)
            finite_fraction = float(np.isfinite(sampled).mean())

            mask_mean = 1.0
            mask_category_counts = {}
            subtraction = None
            if args.protein_subtraction != "none":
                subtraction_coordinates, subtraction_numbers, dropped = (
                    read_modelled_heavy_atoms(
                        mask_structure_path,
                        centre=center,
                        ligand_exclusion_radius_A=args.subtraction_ligand_exclusion_radius,
                        backbone_only=args.protein_subtraction == "backbone_cb",
                        exclude_resnames=[
                            name for name in args.subtraction_exclude_resnames.split(",")
                        ],
                    )
                )
                subtraction = subtract_model_density(
                    sampled,
                    axes,
                    subtraction_coordinates,
                    subtraction_numbers,
                    centre=center,
                    excluded_residues=dropped,
                    sigma_candidates=subtraction_sigmas,
                    fit_exclusion_radius_A=args.subtraction_fit_exclusion_radius,
                )
                prepared_density = subtraction.residual
            elif args.no_protein_mask:
                prepared_density = sampled
            else:
                if args.mask_policy == "full_polymer":
                    protein_coordinates, mask_category_counts = (
                        read_full_polymer_heavy_coordinates(mask_structure_path)
                    )
                else:
                    protein_coordinates = read_protein_heavy_coordinates(protein_path)
                    mask_category_counts = {
                        "protein_polymer": int(len(protein_coordinates))
                    }
                crop_lower = np.asarray([axis[0] for axis in axes], dtype=np.float32)
                crop_upper = np.asarray([axis[-1] for axis in axes], dtype=np.float32)
                margin = args.mask_radius if args.mask_style == "hard" else args.mask_outer
                near_crop = (
                    (protein_coordinates >= crop_lower - margin)
                    & (protein_coordinates <= crop_upper + margin)
                ).all(axis=1)
                mask_atom_count_near_crop = int(near_crop.sum())
                if args.mask_style == "hard":
                    mask = hard_protein_mask(
                        axes, protein_coordinates, radius=args.mask_radius
                    )
                else:
                    mask = soft_protein_mask(
                        axes,
                        protein_coordinates,
                        inner_radius=args.mask_inner,
                        outer_radius=args.mask_outer,
                    )
                mask_mean = float(mask.mean())
                prepared_density, _ = apply_protein_mask(
                    sampled,
                    mask,
                    positive_only=not args.keep_negative_density,
                )

            density, norm_center, norm_scale = normalization(prepared_density)
            ligand_coverage = fraction_in_crop(
                reference_coordinates, origin, density.shape, args.spacing
            )
            crop_path = crop_dir / f"{row_index:04d}_{safe_name(complex_name)}.npz"
            if crop_path.exists() and not args.overwrite:
                raise FileExistsError(f"Crop exists; pass --overwrite: {crop_path}")
            np.savez_compressed(
                crop_path,
                density=density,
                origin=origin,
                spacing=np.repeat(np.float32(args.spacing), 3),
                norm_center=np.float32(norm_center),
                norm_scale=np.float32(norm_scale),
                center=center,
                box_size=np.float32(args.box_size),
                source_map=np.asarray(str(map_path)),
                source_protein=np.asarray(str(protein_path)),
                source_mask_structure=np.asarray(str(mask_structure_path)),
                mask_policy=np.asarray(args.mask_policy),
                mask_style=np.asarray(args.mask_style),
                mask_polymer_heavy_atom_count=np.int32(len(protein_coordinates)) if not args.no_protein_mask else np.int32(0),
                mask_polymer_heavy_atom_count_near_crop=np.int32(mask_atom_count_near_crop) if not args.no_protein_mask else np.int32(0),
                center_source=np.asarray("ground_truth_ligand_heavy_atom_centroid"),
                reference_ligand_coordinates_used=np.asarray(True),
                reference_ligand_use=np.asarray("crop_center_and_qc_only"),
                protein_mask_applied=np.asarray(not args.no_protein_mask),
                protein_mask_inner_A=np.float32(args.mask_inner),
                protein_mask_outer_A=np.float32(args.mask_outer),
                protein_mask_radius_A=np.float32(args.mask_radius),
                protein_subtraction=np.asarray(args.protein_subtraction),
                subtraction_sigma_A=np.float32(
                    subtraction.sigma_A if subtraction is not None else 0.0
                ),
                subtraction_scale=np.float32(
                    subtraction.scale if subtraction is not None else 0.0
                ),
                subtraction_correlation=np.float32(
                    subtraction.correlation if subtraction is not None else 0.0
                ),
            )
            metadata = {
                "complex_name": complex_name,
                "crop_path": str(crop_path),
                "center_source": "ground_truth_ligand_heavy_atom_centroid",
                "reference_ligand_use": "crop_center_and_qc_only",
                "density_source": "experimental_map",
                "protein_mask_uses_reference_ligand": False,
                "mask_policy": args.mask_policy,
                "mask_style": args.mask_style,
                "mask_structure_path": str(mask_structure_path),
                "mask_polymer_heavy_atom_count": (
                    int(len(protein_coordinates)) if not args.no_protein_mask else 0
                ),
                "mask_polymer_heavy_atom_count_near_crop": (
                    mask_atom_count_near_crop if not args.no_protein_mask else 0
                ),
                "mask_category_heavy_atom_counts": mask_category_counts,
                "box_size_A": args.box_size,
                "spacing_A": args.spacing,
                "native_ligand_atom_coverage": ligand_coverage,
                "finite_map_fraction": finite_fraction,
                "protein_subtraction": args.protein_subtraction,
                "subtraction": (
                    {
                        "sigma_A": subtraction.sigma_A,
                        "scale": subtraction.scale,
                        "offset": subtraction.offset,
                        "correlation": subtraction.correlation,
                        "atom_count": subtraction.atom_count,
                        "excluded_residues": list(subtraction.excluded_residues),
                        "fit_voxels": subtraction.fit_voxels,
                    }
                    if subtraction is not None
                    else None
                ),
            }
            crop_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

            prepared = row.to_dict()
            prepared.update({
                "experimental_protein": str(protein_path),
                args.mask_structure_column: str(mask_structure_path),
                "ligand": str(ligand_path),
                "reference_ligand": str(reference_path),
                "map_path": str(map_path),
                "pocket_center_x": float(center[0]),
                "pocket_center_y": float(center[1]),
                "pocket_center_z": float(center[2]),
                "ligand_density_crop": str(crop_path),
            })
            prepared_rows.append(prepared)
            report_rows.append({
                "row_index": row_index,
                "complex_name": complex_name,
                "status": "ok",
                "error": "",
                "crop_path": str(crop_path),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "center_z": float(center[2]),
                "native_ligand_atom_coverage": ligand_coverage,
                "finite_map_fraction": finite_fraction,
                "mean_protein_mask_weight": mask_mean,
                "mask_policy": args.mask_policy,
                "mask_style": args.mask_style,
                "mask_structure_path": str(mask_structure_path),
                "mask_polymer_heavy_atom_count": (
                    int(len(protein_coordinates)) if not args.no_protein_mask else 0
                ),
                "mask_polymer_heavy_atom_count_near_crop": (
                    mask_atom_count_near_crop if not args.no_protein_mask else 0
                ),
            })
            print(f"[{row_index + 1}/{len(table)}] OK {complex_name} coverage={ligand_coverage:.3f}")
        except Exception as exc:
            report_rows.append({
                "row_index": row_index,
                "complex_name": complex_name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{row_index + 1}/{len(table)}] FAILED {complex_name}: {exc}")
            if args.fail_on_error:
                raise

    inference_csv = output_dir / "inference.csv"
    report_csv = output_dir / "preparation_report.csv"
    pd.DataFrame(prepared_rows).to_csv(inference_csv, index=False)
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)
    print(f"PREPARED_ORACLE_DENSITY_DATASET: {len(prepared_rows)} / {len(table)}")
    print(f"Inference CSV: {inference_csv}")
    print(f"Preparation report: {report_csv}")
    return 0 if len(prepared_rows) == len(table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
