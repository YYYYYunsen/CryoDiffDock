#!/usr/bin/env python3
"""Create pocket-centred component-segmented crops from prepared density NPZs."""

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

from cryodiffdock_pocket.density_preparation import (
    normalization,
    segment_pocket_density_component,
)


def resolve_path(value, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "complex"


def scalar(archive, name, default=None):
    if name not in archive.files:
        return default
    return np.asarray(archive[name]).reshape(()).item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--low-z", type=float, default=0.5)
    parser.add_argument("--high-z", type=float, default=1.5)
    parser.add_argument("--seed-radius", type=float, default=4.0)
    parser.add_argument("--dilation", type=float, default=2.0)
    parser.add_argument("--max-seed-distance", type=float, default=6.0)
    parser.add_argument("--min-component-voxels", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    table = pd.read_csv(manifest)
    required = {"complex_name", "ligand_density_crop"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    output = args.output_dir.expanduser().resolve()
    crop_dir = output / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows, report_rows = [], []
    for row_index, row in table.iterrows():
        complex_name = str(row["complex_name"])
        try:
            source = resolve_path(row["ligand_density_crop"], manifest.parent)
            if not source.is_file():
                raise FileNotFoundError(source)
            with np.load(source, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]) for name in archive.files}
            density = np.asarray(payload["density"], dtype=np.float32)
            origin = np.asarray(payload["origin"], dtype=np.float32).reshape(3)
            spacing = np.asarray(payload["spacing"], dtype=np.float32)
            if spacing.ndim == 0:
                spacing = np.repeat(float(spacing), 3)
            spacing = spacing.reshape(3)
            center = np.asarray(payload.get("center"), dtype=np.float32).reshape(3)

            result = segment_pocket_density_component(
                density=density, origin=origin, spacing=spacing, center=center,
                low_z=args.low_z, high_z=args.high_z,
                seed_radius_A=args.seed_radius, dilation_A=args.dilation,
                max_seed_distance_A=args.max_seed_distance,
                min_component_voxels=args.min_component_voxels,
            )
            segmented, norm_center, norm_scale = normalization(result.density)
            destination = crop_dir / f"{row_index:04d}_{safe_name(complex_name)}.npz"
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Crop exists; pass --overwrite: {destination}")
            payload.update({
                "density": segmented,
                "norm_center": np.float32(norm_center),
                "norm_scale": np.float32(norm_scale),
                "density_component_support": result.support.astype(np.uint8),
                "density_component_segmentation": np.asarray(True),
                "density_component_source_crop": np.asarray(str(source)),
                "density_component_status": np.asarray(result.status),
                "density_component_low_z": np.float32(args.low_z),
                "density_component_high_z": np.float32(args.high_z),
                "density_component_seed_radius_A": np.float32(args.seed_radius),
                "density_component_dilation_A": np.float32(args.dilation),
                "density_component_max_seed_distance_A": np.float32(args.max_seed_distance),
                "density_component_min_voxels": np.int32(args.min_component_voxels),
            })
            np.savez_compressed(destination, **payload)

            prepared = row.to_dict()
            prepared["ligand_density_crop"] = str(destination)
            prepared_rows.append(prepared)
            report_rows.append({
                "row_index": row_index,
                "complex_name": complex_name,
                "status": result.status,
                "error": "",
                "source_crop": str(source),
                "crop_path": str(destination),
                "num_components": result.num_components,
                "selected_label": result.selected_label,
                "component_voxels": result.component_voxels,
                "support_voxels": result.support_voxels,
                "seed_distance_A": result.seed_distance_A,
                "retained_positive_signal_fraction": result.retained_positive_signal_fraction,
                "fallback": result.status.startswith("fallback_"),
            })
            print(
                f"[{row_index + 1}/{len(table)}] {complex_name}: {result.status} "
                f"retained={result.retained_positive_signal_fraction:.3f}"
            )
        except Exception as exc:
            report_rows.append({
                "row_index": row_index, "complex_name": complex_name,
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{row_index + 1}/{len(table)}] FAILED {complex_name}: {exc}")
            if args.fail_on_error:
                raise

    pd.DataFrame(prepared_rows).to_csv(output / "inference.csv", index=False)
    report = pd.DataFrame(report_rows)
    report.to_csv(output / "segmentation_report.csv", index=False)
    summary = {
        "manifest": str(manifest),
        "num_complexes": len(prepared_rows),
        "num_failures": int((report["status"] == "failed").sum()),
        "num_fallbacks": int(report.get("fallback", pd.Series(dtype=bool)).fillna(False).sum()),
        "parameters": {
            "low_z": args.low_z, "high_z": args.high_z,
            "seed_radius_A": args.seed_radius, "dilation_A": args.dilation,
            "max_seed_distance_A": args.max_seed_distance,
            "min_component_voxels": args.min_component_voxels,
        },
        "uses_reference_ligand_atoms": False,
        "uses_ground_truth_beyond_manifest_pocket_center": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(prepared_rows) == len(table) else 1


if __name__ == "__main__":
    raise SystemExit(main())
