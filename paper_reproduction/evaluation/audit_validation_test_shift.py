#!/usr/bin/env python3
"""Audit validation/test workflow parity and local-docking difficulty shift.

The audit is deliberately read-only.  It compares three performance stages
(original density selection, frozen S0, and expanded-pool oracle), verifies
the density-crop schema and scalar preparation parameters, inventories the
actual inference outputs, and compares ligand/crop covariates between two
cohorts.  Reference ligands are read only for retrospective audit metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


SCIENTIFIC_PDB = re.compile(r"^(?P<a>[0-9])[eE]\+(?P<b>[0-9]{2})$")
RANKED_SDF = re.compile(r"^rank(?P<rank>[0-9]+)_confidence.*[.]sdf$")

CROP_CONSTANT_KEYS = (
    "box_size",
    "spacing",
    "mask_policy",
    "mask_style",
    "center_source",
    "reference_ligand_coordinates_used",
    "reference_ligand_use",
    "protein_mask_applied",
    "protein_mask_inner_A",
    "protein_mask_outer_A",
    "protein_mask_radius_A",
    "protein_subtraction",
)

SEARCH_KEYS = (
    "parent_score_column",
    "parent_count",
    "rotation_degrees",
    "translation_distances_A",
    "torsion_degrees",
    "max_torsion_bonds",
    "coarse_orientations",
    "density_sigma_A",
    "density_weight",
    "minimum_ncc_gain",
    "relative_physics_gate",
)

GUIDANCE_KEYS = (
    "objective",
    "lambda_translation",
    "lambda_rotation",
    "lambda_torsion",
    "start_fraction",
    "clip_ratio",
    "use_x0_estimate",
    "use_torsion_projection",
    "ncc_sigma_A",
    "ncc_sigma_noise_scale",
    "ncc_sigma_max_A",
    "ncc_mask_radius_A",
    "ncc_chunk_size",
)


def normalise_name(value: Any) -> str:
    text = str(value).strip()
    match = SCIENTIFIC_PDB.fullmatch(text)
    if match:
        return f"{match.group('a')}E{match.group('b')}"
    return text.upper()


def resolve(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        result = array.item()
        if isinstance(result, bytes):
            return result.decode()
        if isinstance(result, np.generic):
            return result.item()
        return result
    return tuple(scalar(item) for item in array.reshape(-1))


def canonical(value: Any) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps([canonical(item) for item in value])
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.9g}"
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = [normalise_name(line) for line in path.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate IDs in {path}")
    return set(values)


def load_metric(path: Path, ids: set[str] | None, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"complex_name": str})
    required = {
        "complex_name", "top1_rmsd_A", "best_of_n_rmsd_A",
        "top1_centroid_distance_A", "top1_centered_rmsd_A",
        "top1_rigid_aligned_rmsd_A", "best_rmsd_centroid_distance_A",
        "best_rmsd_centered_rmsd_A", "best_rmsd_rigid_aligned_rmsd_A",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: {path} lacks columns {missing}")
    frame = frame.copy()
    frame["complex_name"] = frame["complex_name"].map(normalise_name)
    if ids is not None:
        frame = frame[frame["complex_name"].isin(ids)].copy()
    if frame["complex_name"].duplicated().any():
        raise ValueError(f"{label}: duplicate complex names in {path}")
    return frame


def load_manifest(path: Path, names: set[str], label: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"complex_name": str})
    if "complex_name" not in frame:
        raise ValueError(f"{label}: manifest lacks complex_name")
    frame = frame.copy()
    frame["complex_name"] = frame["complex_name"].map(normalise_name)
    frame = frame[frame["complex_name"].isin(names)].copy()
    if frame["complex_name"].duplicated().any():
        raise ValueError(f"{label}: duplicate complex names in manifest")
    absent = sorted(names - set(frame["complex_name"]))
    if absent:
        raise ValueError(f"{label}: {len(absent)} metric complexes absent from manifest: {absent[:10]}")
    return frame


def molecule_covariates(path: Path) -> dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    suffix = path.suffix.lower()
    if suffix in {".sdf", ".mol"}:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        molecule = supplier[0] if supplier and len(supplier) else None
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    else:
        raise ValueError(f"unsupported ligand format {suffix}")
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"could not read ligand {path}")
    molecule.UpdatePropertyCache(strict=False)
    rings = [len(ring) for ring in Chem.GetSymmSSSR(molecule)]
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)[heavy]
    centre = coordinates.mean(axis=0)
    return {
        "ligand_heavy_atoms": len(heavy),
        "ligand_rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "ligand_max_ring_size": max(rings, default=0),
        "ligand_radius_A": float(np.linalg.norm(coordinates - centre, axis=1).max()),
        "reference_center_x": float(centre[0]),
        "reference_center_y": float(centre[1]),
        "reference_center_z": float(centre[2]),
    }


def crop_covariates(path: Path, reference_center: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(CROP_CONSTANT_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"crop {path} lacks keys {missing}")
        constants = {key: scalar(archive[key]) for key in CROP_CONSTANT_KEYS}
        density = np.asarray(archive["density"], dtype=float)
        centre = np.asarray(archive["center"], dtype=float)
        spacing = np.asarray(archive["spacing"], dtype=float)
        stats = {
            "crop_shape": "x".join(str(value) for value in density.shape),
            "crop_finite_fraction": float(np.isfinite(density).mean()),
            "crop_density_mean": float(np.nanmean(density)),
            "crop_density_std": float(np.nanstd(density)),
            "crop_density_p95": float(np.nanpercentile(density, 95)),
            "crop_density_max": float(np.nanmax(density)),
            "crop_norm_center": float(scalar(archive["norm_center"])),
            "crop_norm_scale": float(scalar(archive["norm_scale"])),
            "crop_center_reference_delta_A": float(np.linalg.norm(centre - reference_center)),
            "crop_spacing_mean_A": float(spacing.mean()),
            "crop_mask_atoms": int(scalar(archive["mask_polymer_heavy_atom_count"])),
            "crop_mask_atoms_near": int(scalar(archive["mask_polymer_heavy_atom_count_near_crop"])),
            "crop_source_map_name": Path(str(scalar(archive["source_map"]))).name,
            "crop_source_protein_name": Path(str(scalar(archive["source_protein"]))).name,
        }
    metadata_path = path.with_suffix(".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        stats["native_ligand_atom_coverage"] = metadata.get(
            "native_ligand_atom_coverage", np.nan
        )
        stats["finite_map_fraction_preparation"] = metadata.get(
            "finite_map_fraction", np.nan
        )
    return constants, stats


def inventory_inference(root: Path | None, names: set[str], cohort: str, arm: str) -> list[dict]:
    if root is None:
        return []
    rows = []
    found = set()
    for directory in sorted(root.glob("index*___*")):
        if not directory.is_dir():
            continue
        name = normalise_name(directory.name.split("___", 1)[1].split("_protein", 1)[0])
        if name not in names:
            continue
        ranks = []
        for path in directory.iterdir():
            match = RANKED_SDF.match(path.name)
            if match:
                ranks.append(int(match.group("rank")))
        found.add(name)
        rows.append({
            "cohort": cohort,
            "arm": arm,
            "complex_name": name,
            "num_ranked_poses": len(ranks),
            "maximum_rank": max(ranks, default=0),
            "result_directory": str(directory),
        })
    for name in sorted(names - found):
        rows.append({
            "cohort": cohort, "arm": arm, "complex_name": name,
            "num_ranked_poses": 0, "maximum_rank": 0, "result_directory": "",
        })
    return rows


def guidance_inventory(root: Path | None, names: set[str], cohort: str) -> tuple[list[dict], dict]:
    if root is None:
        return [], {"cohort": cohort, "num_audits": 0, "unique_configs": []}
    rows = []
    for path in sorted(root.glob("index*___*/density_guidance_audit.json")):
        name = normalise_name(path.parent.name.split("___", 1)[1].split("_protein", 1)[0])
        if name not in names:
            continue
        payload = json.loads(path.read_text())
        row = {"cohort": cohort, "complex_name": name, "audit_path": str(path)}
        row.update({key: payload.get(key) for key in GUIDANCE_KEYS})
        row["num_audited_active_steps"] = len(payload.get("steps", []))
        rows.append(row)
    unique = sorted({
        json.dumps({key: row.get(key) for key in GUIDANCE_KEYS}, sort_keys=True)
        for row in rows
    })
    return rows, {"cohort": cohort, "num_audits": len(rows), "unique_configs": unique}


def performance_rows(cohort: str, original: pd.DataFrame, s0: pd.DataFrame,
                     oracle: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    names = set(original["complex_name"]) & set(s0["complex_name"]) & set(oracle["complex_name"])
    if not names:
        raise ValueError(f"{cohort}: metric tables have no common complexes")
    if any(set(frame["complex_name"]) != names for frame in (original, s0, oracle)):
        raise ValueError(f"{cohort}: original/S0/oracle metrics are not exactly paired")
    order = sorted(names)
    aligned = {
        "original_ncc": original.set_index("complex_name").loc[order],
        "s0": s0.set_index("complex_name").loc[order],
        "expanded_oracle": oracle.set_index("complex_name").loc[order],
    }
    rows = []
    for condition, frame in aligned.items():
        values = (
            frame["best_of_n_rmsd_A"].to_numpy(float)
            if condition == "expanded_oracle"
            else frame["top1_rmsd_A"].to_numpy(float)
        )
        rows.append({
            "cohort": cohort,
            "condition": condition,
            "num_complexes": len(values),
            "success_lt2_count": int((values < 2).sum()),
            "success_lt2_pct": float(100 * np.mean(values < 2)),
            "success_lt5_pct": float(100 * np.mean(values < 5)),
            "median_rmsd_A": float(np.median(values)),
        })
    original_values = aligned["original_ncc"]["top1_rmsd_A"].to_numpy(float)
    s0_values = aligned["s0"]["top1_rmsd_A"].to_numpy(float)
    oracle_values = aligned["expanded_oracle"]["best_of_n_rmsd_A"].to_numpy(float)
    original_success = original_values < 2
    s0_success = s0_values < 2
    oracle_success = oracle_values < 2
    rows.append({
        "cohort": cohort,
        "condition": "selector_diagnostics",
        "num_complexes": len(order),
        "success_lt2_count": int(s0_success.sum()),
        "success_lt2_pct": float(100 * s0_success.mean()),
        "success_lt5_pct": np.nan,
        "median_rmsd_A": float(np.median(s0_values)),
        "s0_gain_over_original_pp": float(100 * (s0_success.mean() - original_success.mean())),
        "s0_gained_count": int((~original_success & s0_success).sum()),
        "s0_lost_count": int((original_success & ~s0_success).sum()),
        "selector_capture_of_oracle_pct": float(100 * s0_success.sum() / max(oracle_success.sum(), 1)),
        "unselected_oracle_successes": int((oracle_success & ~s0_success).sum()),
    })
    per_complex = pd.DataFrame({
        "cohort": cohort,
        "complex_name": order,
        "original_ncc_rmsd_A": original_values,
        "s0_rmsd_A": s0_values,
        "expanded_oracle_rmsd_A": oracle_values,
        "original_ncc_success": original_success,
        "s0_success": s0_success,
        "expanded_oracle_success": oracle_success,
        "s0_delta_vs_original_A": s0_values - original_values,
    })
    for prefix, frame in aligned.items():
        if prefix == "expanded_oracle":
            columns = {
                "best_rmsd_centroid_distance_A": "oracle_centroid_distance_A",
                "best_rmsd_centered_rmsd_A": "oracle_centered_rmsd_A",
                "best_rmsd_rigid_aligned_rmsd_A": "oracle_rigid_aligned_rmsd_A",
            }
        else:
            columns = {
                "top1_centroid_distance_A": f"{prefix}_centroid_distance_A",
                "top1_centered_rmsd_A": f"{prefix}_centered_rmsd_A",
                "top1_rigid_aligned_rmsd_A": f"{prefix}_rigid_aligned_rmsd_A",
            }
        for source, destination in columns.items():
            per_complex[destination] = frame[source].to_numpy(float)
    return rows, per_complex


def distribution_rows(frame: pd.DataFrame) -> list[dict]:
    cohorts = sorted(frame["cohort"].unique())
    if len(cohorts) != 2:
        return []
    left, right = cohorts
    ignored = {"cohort", "complex_name"}
    rows = []
    for column in frame.columns:
        if column in ignored or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        # Pandas considers bool a numeric dtype, but NumPy cannot interpolate
        # boolean values when computing quantiles.  Cast every audited numeric
        # series to float so binary indicators are summarized as 0/1 rates.
        a = (
            pd.to_numeric(
                frame.loc[frame["cohort"] == left, column], errors="coerce"
            )
            .dropna()
            .astype(float)
        )
        b = (
            pd.to_numeric(
                frame.loc[frame["cohort"] == right, column], errors="coerce"
            )
            .dropna()
            .astype(float)
        )
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            pvalue = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        except ValueError:
            pvalue = np.nan
        rows.append({
            "metric": column,
            "left_cohort": left,
            "right_cohort": right,
            "left_n": len(a),
            "right_n": len(b),
            "left_median": float(a.median()),
            "right_median": float(b.median()),
            "right_minus_left_median": float(b.median() - a.median()),
            "left_q25": float(a.quantile(0.25)),
            "left_q75": float(a.quantile(0.75)),
            "right_q25": float(b.quantile(0.25)),
            "right_q75": float(b.quantile(0.75)),
            "mannwhitney_pvalue": pvalue,
        })
    return rows


def cohort(args, prefix: str, label: str) -> dict:
    ids = load_ids(getattr(args, f"{prefix}_ids"))
    metrics = {
        name: load_metric(getattr(args, f"{prefix}_{name}"), ids, f"{label}/{name}")
        for name in ("original", "s0", "oracle")
    }
    common = set.intersection(*(set(frame["complex_name"]) for frame in metrics.values()))
    if any(set(frame["complex_name"]) != common for frame in metrics.values()):
        sizes = {name: len(frame) for name, frame in metrics.items()}
        raise ValueError(f"{label}: metric cohorts differ: {sizes}")
    manifest_path = getattr(args, f"{prefix}_manifest").resolve()
    manifest = load_manifest(manifest_path, common, label)
    reference_ncc_path = getattr(args, f"{prefix}_reference_ncc")
    reference_ncc = {}
    if reference_ncc_path is not None:
        reference_frame = pd.read_csv(reference_ncc_path, dtype={"complex_name": str})
        reference_frame["complex_name"] = reference_frame["complex_name"].map(normalise_name)
        ncc_columns = [column for column in reference_frame if column.startswith("ncc_sigma_")]
        if len(ncc_columns) != 1:
            raise ValueError(
                f"{label}: expected exactly one ncc_sigma_* column in "
                f"{reference_ncc_path}, found {ncc_columns}"
            )
        reference_ncc = dict(zip(
            reference_frame["complex_name"],
            pd.to_numeric(reference_frame[ncc_columns[0]], errors="coerce"),
        ))
    return {
        "label": label,
        "ids": ids,
        "names": common,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "metrics": metrics,
        "baseline_root": getattr(args, f"{prefix}_baseline_root"),
        "guided_root": getattr(args, f"{prefix}_guided_root"),
        "search_summary": getattr(args, f"{prefix}_search_summary"),
        "s0_config": getattr(args, f"{prefix}_s0_config"),
        "reference_ncc": reference_ncc,
    }


def add_cohort_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-manifest", required=True, type=Path)
    parser.add_argument(f"--{prefix}-ids", type=Path)
    parser.add_argument(f"--{prefix}-original", required=True, type=Path)
    parser.add_argument(f"--{prefix}-s0", required=True, type=Path)
    parser.add_argument(f"--{prefix}-oracle", required=True, type=Path)
    parser.add_argument(f"--{prefix}-baseline-root", type=Path)
    parser.add_argument(f"--{prefix}-guided-root", type=Path)
    parser.add_argument(f"--{prefix}-search-summary", type=Path)
    parser.add_argument(f"--{prefix}-s0-config", type=Path)
    parser.add_argument(f"--{prefix}-reference-ncc", type=Path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cohort_arguments(parser, "val")
    add_cohort_arguments(parser, "test")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    cohorts = [cohort(args, "val", "validation"), cohort(args, "test", "test")]
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    performance = []
    per_complex_performance = []
    covariates = []
    crop_constants = []
    failures = []
    inference_rows = []
    guidance_rows = []
    guidance_summary = []
    parameter_rows = []

    for item in cohorts:
        rows, per_complex = performance_rows(
            item["label"], item["metrics"]["original"],
            item["metrics"]["s0"], item["metrics"]["oracle"],
        )
        performance.extend(rows)
        per_complex_performance.append(per_complex)
        manifest = item["manifest"].set_index("complex_name")
        for name in sorted(item["names"]):
            row = manifest.loc[name]
            try:
                reference_path = resolve(row["reference_ligand"], item["manifest_path"].parent)
                molecule = molecule_covariates(reference_path)
                reference_center = np.asarray([
                    molecule.pop("reference_center_x"), molecule.pop("reference_center_y"),
                    molecule.pop("reference_center_z"),
                ])
                crop_path = resolve(row["ligand_density_crop"], item["manifest_path"].parent)
                constants, crop_stats = crop_covariates(crop_path, reference_center)
                flexible = str(row.get("flexible_sidechains", ""))
                record = {
                    "cohort": item["label"], "complex_name": name,
                    "num_flexible_sidechains": 0 if flexible.lower() in {"", "nan", "none"}
                    else len([value for value in flexible.split("-") if value]),
                    **molecule, **crop_stats,
                }
                if item["reference_ncc"]:
                    record["reference_ncc_sigma_1p5"] = item["reference_ncc"].get(
                        name, np.nan
                    )
                for column in manifest.columns:
                    if "resolution" in column.lower():
                        try:
                            record[f"manifest_{column}"] = float(row[column])
                        except (TypeError, ValueError):
                            pass
                covariates.append(record)
                crop_constants.append({
                    "cohort": item["label"], "complex_name": name,
                    **{key: canonical(value) for key, value in constants.items()},
                })
            except Exception as exc:
                failures.append({
                    "cohort": item["label"], "complex_name": name,
                    "stage": "ligand_or_crop", "error": f"{type(exc).__name__}: {exc}",
                })

        inference_rows.extend(inventory_inference(
            item["baseline_root"], item["names"], item["label"], "baseline"
        ))
        inference_rows.extend(inventory_inference(
            item["guided_root"], item["names"], item["label"], "guided"
        ))
        guidance, guidance_info = guidance_inventory(
            item["guided_root"], item["names"], item["label"]
        )
        guidance_rows.extend(guidance)
        guidance_summary.append(guidance_info)

        if item["search_summary"] is not None:
            payload = json.loads(item["search_summary"].read_text())
            for key in SEARCH_KEYS:
                parameter_rows.append({
                    "stage": "discrete_search", "parameter": key,
                    "cohort": item["label"], "value": canonical(payload.get(key)),
                    "source": str(item["search_summary"]),
                })
        if item["s0_config"] is not None:
            parameter_rows.append({
                "stage": "s0", "parameter": "config_sha256",
                "cohort": item["label"], "value": sha256(item["s0_config"]),
                "source": str(item["s0_config"]),
            })

    performance_frame = pd.DataFrame(performance)
    per_complex_frame = pd.concat(per_complex_performance, ignore_index=True)
    covariate_frame = pd.DataFrame(covariates)
    crop_frame = pd.DataFrame(crop_constants)
    inference_frame = pd.DataFrame(inference_rows)
    guidance_frame = pd.DataFrame(guidance_rows)
    parameter_frame = pd.DataFrame(parameter_rows)

    if not guidance_frame.empty:
        for _, row in guidance_frame.iterrows():
            for key in GUIDANCE_KEYS:
                parameter_rows.append({
                    "stage": "guidance", "parameter": key,
                    "cohort": row["cohort"], "value": canonical(row[key]),
                    "source": row["audit_path"],
                })
        parameter_frame = pd.DataFrame(parameter_rows)

    crop_parameter_rows = []
    for cohort_name, group in crop_frame.groupby("cohort"):
        for key in CROP_CONSTANT_KEYS:
            values = sorted(set(group[key].dropna().astype(str)))
            crop_parameter_rows.append({
                "stage": "density_crop", "parameter": key, "cohort": cohort_name,
                "value": " | ".join(values), "source": "all cohort crop npz files",
            })
    parameter_frame = pd.concat(
        [parameter_frame, pd.DataFrame(crop_parameter_rows)], ignore_index=True
    )

    parity_rows = []
    for (stage, parameter), group in parameter_frame.groupby(["stage", "parameter"]):
        values = {
            cohort_name: sorted(set(rows["value"].astype(str)))
            for cohort_name, rows in group.groupby("cohort")
        }
        parity_rows.append({
            "stage": stage, "parameter": parameter,
            "validation_values": " | ".join(values.get("validation", [])),
            "test_values": " | ".join(values.get("test", [])),
            "exact_match": values.get("validation", []) == values.get("test", []),
        })
    parity_frame = pd.DataFrame(parity_rows)

    combined = per_complex_frame.merge(
        covariate_frame, on=["cohort", "complex_name"], how="left", validate="one_to_one"
    )
    distributions = pd.DataFrame(distribution_rows(combined))

    performance_frame.to_csv(output / "performance_summary.csv", index=False)
    combined.to_csv(output / "per_complex_audit.csv", index=False)
    covariate_frame.to_csv(output / "cohort_covariates.csv", index=False)
    crop_frame.to_csv(output / "crop_parameters_per_complex.csv", index=False)
    inference_frame.to_csv(output / "inference_inventory.csv", index=False)
    guidance_frame.to_csv(output / "guidance_inventory.csv", index=False)
    parameter_frame.to_csv(output / "parameter_inventory.csv", index=False)
    parity_frame.to_csv(output / "workflow_parity.csv", index=False)
    distributions.to_csv(output / "distribution_comparison.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "audit_failures.csv", index=False)

    inference_ok = True
    if not inference_frame.empty:
        inference_ok = bool(
            (inference_frame["num_ranked_poses"] == 40).all()
            and set(inference_frame["cohort"]) == {"validation", "test"}
        )
    parity_ok = bool(parity_frame["exact_match"].all()) if len(parity_frame) else False
    centre_ok = bool(
        len(covariate_frame)
        and (covariate_frame["crop_center_reference_delta_A"] < 1e-3).all()
    )
    finite_ok = bool(
        len(covariate_frame)
        and (covariate_frame["crop_finite_fraction"] == 1.0).all()
    )
    summary = {
        "validation_complexes": len(cohorts[0]["names"]),
        "test_complexes": len(cohorts[1]["names"]),
        "workflow_parameter_parity": parity_ok,
        "inference_exactly_40_poses_per_requested_arm": inference_ok,
        "crop_centres_match_reference_centroids_within_0p001A": centre_ok,
        "all_crop_voxels_finite": finite_ok,
        "num_audit_failures": len(failures),
        "guidance": guidance_summary,
        "warning": (
            "Passing workflow parity does not prove identical model checkpoints or base "
            "inference seed. Preserve and compare checkpoint, model YAML, code, manifest, "
            "and launch-record SHA-256 hashes separately."
        ),
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nPERFORMANCE")
    print(performance_frame.to_string(index=False))
    print("\nWORKFLOW PARITY MISMATCHES")
    mismatches = parity_frame[~parity_frame["exact_match"]]
    print(mismatches.to_string(index=False) if len(mismatches) else "none")
    print("\nLARGEST DISTRIBUTION SHIFTS (smallest p-values)")
    if len(distributions):
        print(distributions.sort_values("mannwhitney_pvalue").head(20).to_string(index=False))
    else:
        print("none")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
