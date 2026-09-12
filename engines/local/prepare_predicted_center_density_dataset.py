#!/usr/bin/env python3
"""Prepare experimental-density crops at externally predicted pocket centres."""

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

from cryodiffdock_pocket.density_preparation import (  # noqa: E402
    apply_protein_mask,
    cartesian_axes,
    normalization,
    read_full_polymer_heavy_coordinates,
    sample_map,
    soft_protein_mask,
)


REQUIRED = {
    "complex_name", "experimental_protein", "ligand", "map_path",
    "mask_structure", "flexible_sidechains", "pocket_center_x",
    "pocket_center_y", "pocket_center_z", "parent_complex_name", "global_rank",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "complex"


def resolve(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--box-size", type=float, default=28.0)
    parser.add_argument("--spacing", type=float, default=0.75)
    parser.add_argument("--mask-inner", type=float, default=1.5)
    parser.add_argument("--mask-outer", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.box_size <= 0 or args.spacing <= 0:
        raise ValueError("box-size and spacing must be positive")

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest, dtype={"complex_name": str})
    missing = sorted(REQUIRED - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if "reference_ligand" in table.columns:
        raise ValueError("Predicted-centre preparation refuses a reference_ligand column")

    output = args.output_dir.expanduser().resolve()
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    prepared_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for row_index, row in table.iterrows():
        name = str(row["complex_name"])
        try:
            center = np.asarray(
                [row["pocket_center_x"], row["pocket_center_y"], row["pocket_center_z"]],
                dtype=np.float32,
            )
            if not np.isfinite(center).all():
                raise ValueError("non-finite predicted centre")
            protein = resolve(row["experimental_protein"], manifest.parent)
            ligand = resolve(row["ligand"], manifest.parent)
            map_path = resolve(row["map_path"], manifest.parent)
            mask_structure = resolve(row["mask_structure"], manifest.parent)
            for path in (protein, ligand, map_path, mask_structure):
                if not path.is_file():
                    raise FileNotFoundError(path)

            origin, axes = cartesian_axes(center, args.box_size, args.spacing)
            sampled = sample_map(map_path, axes)
            protein_coordinates, category_counts = read_full_polymer_heavy_coordinates(
                mask_structure
            )
            lower = np.asarray([axis[0] for axis in axes], dtype=np.float32)
            upper = np.asarray([axis[-1] for axis in axes], dtype=np.float32)
            near = (
                (protein_coordinates >= lower - args.mask_outer)
                & (protein_coordinates <= upper + args.mask_outer)
            ).all(axis=1)
            mask = soft_protein_mask(
                axes, protein_coordinates,
                inner_radius=args.mask_inner, outer_radius=args.mask_outer,
            )
            prepared_density, _ = apply_protein_mask(sampled, mask, positive_only=True)
            density, norm_center, norm_scale = normalization(prepared_density)
            crop_path = crops / f"{row_index:04d}_{safe_name(name)}.npz"
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
                source_protein=np.asarray(str(protein)),
                source_mask_structure=np.asarray(str(mask_structure)),
                mask_policy=np.asarray("full_polymer"),
                mask_style=np.asarray("soft"),
                mask_polymer_heavy_atom_count=np.int32(len(protein_coordinates)),
                mask_polymer_heavy_atom_count_near_crop=np.int32(near.sum()),
                center_source=np.asarray("global_predicted_ligand_heavy_atom_centroid"),
                reference_ligand_coordinates_used=np.asarray(False),
                protein_mask_applied=np.asarray(True),
                protein_mask_inner_A=np.float32(args.mask_inner),
                protein_mask_outer_A=np.float32(args.mask_outer),
            )
            crop_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "complex_name": name,
                        "parent_complex_name": str(row["parent_complex_name"]),
                        "global_rank": int(row["global_rank"]),
                        "crop_path": str(crop_path),
                        "center": center.tolist(),
                        "center_source": "global_predicted_ligand_heavy_atom_centroid",
                        "reference_ligand_coordinates_used": False,
                        "density_source": "experimental_map",
                        "mask_policy": "full_polymer",
                        "mask_style": "soft",
                        "mask_category_heavy_atom_counts": category_counts,
                    },
                    indent=2,
                ) + "\n"
            )
            prepared = row.to_dict()
            prepared.update(
                {
                    "experimental_protein": str(protein),
                    "ligand": str(ligand),
                    "map_path": str(map_path),
                    "mask_structure": str(mask_structure),
                    "ligand_density_crop": str(crop_path),
                }
            )
            prepared_rows.append(prepared)
            report_rows.append(
                {
                    "row_index": row_index,
                    "complex_name": name,
                    "parent_complex_name": row["parent_complex_name"],
                    "global_rank": row["global_rank"],
                    "status": "ok",
                    "finite_map_fraction": float(np.isfinite(sampled).mean()),
                    "mask_polymer_heavy_atom_count_near_crop": int(near.sum()),
                }
            )
            print(f"[{row_index + 1}/{len(table)}] OK {name}", flush=True)
        except Exception as exc:
            failure = {
                "row_index": row_index,
                "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[{row_index + 1}/{len(table)}] FAILED {name}: {exc}", flush=True)
            if args.fail_on_error:
                raise

    pd.DataFrame(prepared_rows).to_csv(output / "inference_blind.csv", index=False)
    pd.DataFrame(report_rows).to_csv(output / "preparation_report.csv", index=False)
    pd.DataFrame(failures, columns=["row_index", "complex_name", "error"]).to_csv(
        output / "failures.csv", index=False
    )
    summary = {
        "requested_branch_rows": len(table),
        "prepared_branch_rows": len(prepared_rows),
        "failures": len(failures),
        "box_size_A": args.box_size,
        "spacing_A": args.spacing,
        "mask_policy": "full_polymer",
        "mask_style": "soft",
        "center_source": "global_predicted_ligand_heavy_atom_centroid",
        "uses_reference_coordinates": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_PREPARATION_OK: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
