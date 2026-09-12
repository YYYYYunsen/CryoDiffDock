#!/usr/bin/env python3
"""Validation-only evaluation of NCC ranking with hard physical gates.

Reference ligand coordinates are used only here for RMSD and optional
validation calibration of gate thresholds.  Deployment scoring belongs in
``scripts/score_candidate_physics.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_ranking import zscore  # noqa: E402
from cryodiffdock_pocket.physics import (  # noqa: E402
    read_protein_heavy_atoms,
    score_ligand_pose_physics,
)
from evaluation.evaluate_ligand_accuracy import (  # noqa: E402
    read_heavy_molecule,
    resolve_path,
    symmetry_rmsd,
)


MERGE_KEYS = ["row_index", "complex_name", "source", "source_rank", "prediction_path"]


def _normalise_paths(frame: pd.DataFrame, column: str, base: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame[column] = [
        str(resolve_path(value, base)) for value in frame[column]
    ]
    return frame


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def physical_gate_mask(
    frame: pd.DataFrame,
    *,
    max_severe_clashes: float,
    max_clashing_ligand_fraction: float,
    min_in_bounds_fraction: float,
    max_strain_kcal_mol: Optional[float] = None,
    missing_strain_policy: str = "allow",
) -> pd.Series:
    gate = (
        (frame["input_severe_clash_count"] <= max_severe_clashes)
        & (
            frame["input_clashing_ligand_atom_fraction"]
            <= max_clashing_ligand_fraction
        )
        & (frame["ligand_in_bounds_fraction"] >= min_in_bounds_fraction)
    )
    if max_strain_kcal_mol is not None:
        available = _bool_series(frame["input_strain_available"])
        strain_ok = frame["input_strain_energy_kcal_mol"] <= max_strain_kcal_mol
        if missing_strain_policy == "allow":
            strain_ok = strain_ok | ~available
        elif missing_strain_policy == "reject":
            strain_ok = strain_ok & available
        else:
            raise ValueError("missing_strain_policy must be allow or reject")
        gate &= strain_ok
    return gate.fillna(False)


def select_pool(
    group: pd.DataFrame,
    *,
    density_column: str,
    density_weight: float,
    fallback_policy: str,
) -> dict[str, object]:
    group = group.copy()
    confidence = group["confidence"].to_numpy(dtype=float)
    if not np.isfinite(confidence).all():
        confidence = np.where(
            np.isfinite(confidence), confidence, -group["source_rank"].to_numpy()
        )
    group["selection_confidence_z"] = zscore(confidence)
    group["selection_density_z"] = zscore(group[density_column])
    group["selection_score"] = (
        group["selection_confidence_z"]
        + float(density_weight) * group["selection_density_z"]
    )
    ordered = group.sort_values(
        ["selection_score", "confidence", "source", "source_rank"],
        ascending=[False, False, True, True],
        kind="stable",
    )
    ungated = ordered.iloc[0]
    eligible = ordered[ordered["physical_gate_pass"].astype(bool)]
    fallback = eligible.empty
    if fallback and fallback_policy == "fail":
        selected = None
    elif fallback:
        selected = ungated
    else:
        selected = eligible.iloc[0]
    return {
        "ordered": ordered,
        "ungated": ungated,
        "selected": selected,
        "fallback": bool(fallback),
        "num_candidates": int(len(group)),
        "num_physical_pass": int(group["physical_gate_pass"].sum()),
    }


def _pose_fields(prefix: str, pose: Optional[pd.Series], density_column: str):
    if pose is None:
        return {
            f"{prefix}_available": False,
            f"{prefix}_source": None,
            f"{prefix}_source_rank": None,
            f"{prefix}_rmsd_A": None,
            f"{prefix}_ncc": None,
            f"{prefix}_physical_gate_pass": False,
            f"{prefix}_severe_clash_count": None,
            f"{prefix}_clashing_ligand_atom_fraction": None,
            f"{prefix}_strain_energy_kcal_mol": None,
            f"{prefix}_prediction_path": None,
        }
    return {
        f"{prefix}_available": True,
        f"{prefix}_source": str(pose["source"]),
        f"{prefix}_source_rank": int(pose["source_rank"]),
        f"{prefix}_rmsd_A": float(pose["ligand_rmsd_A"]),
        f"{prefix}_ncc": float(pose[density_column]),
        f"{prefix}_physical_gate_pass": bool(pose["physical_gate_pass"]),
        f"{prefix}_severe_clash_count": int(pose["input_severe_clash_count"]),
        f"{prefix}_clashing_ligand_atom_fraction": float(
            pose["input_clashing_ligand_atom_fraction"]
        ),
        f"{prefix}_strain_energy_kcal_mol": (
            float(pose["input_strain_energy_kcal_mol"])
            if pd.notna(pose["input_strain_energy_kcal_mol"]) else None
        ),
        f"{prefix}_prediction_path": str(pose["prediction_path"]),
    }


def _summary_for_pool(group: pd.DataFrame) -> dict[str, object]:
    selected = group[group["selected_available"].astype(bool)]
    if selected.empty:
        return {"num_complexes": 0}
    return {
        "num_complexes": int(len(group)),
        "candidate_physical_pass_pct": float(
            100.0 * group["num_physical_pass"].sum() / group["num_candidates"].sum()
        ),
        "complexes_without_physical_candidate_pct": float(
            100.0 * group["gate_fallback"].mean()
        ),
        "gate_changed_selection_pct": float(
            100.0 * (
                group["selected_prediction_path"]
                != group["ungated_prediction_path"]
            ).mean()
        ),
        "median_ungated_rmsd_A": float(group["ungated_rmsd_A"].median()),
        "ungated_lt2A_pct": float(100.0 * (group["ungated_rmsd_A"] < 2.0).mean()),
        "ungated_lt5A_pct": float(100.0 * (group["ungated_rmsd_A"] < 5.0).mean()),
        "median_selected_rmsd_A": float(selected["selected_rmsd_A"].median()),
        "selected_lt2A_pct": float(100.0 * (selected["selected_rmsd_A"] < 2.0).mean()),
        "selected_lt5A_pct": float(100.0 * (selected["selected_rmsd_A"] < 5.0).mean()),
        "median_selected_ncc": float(selected["selected_ncc"].median()),
        "median_selected_severe_clash_count": float(
            selected["selected_severe_clash_count"].median()
        ),
        "selected_strain_available_pct": float(
            100.0 * selected["selected_strain_energy_kcal_mol"].notna().mean()
        ),
        "median_selected_strain_energy_kcal_mol": (
            float(selected["selected_strain_energy_kcal_mol"].dropna().median())
            if selected["selected_strain_energy_kcal_mol"].notna().any() else None
        ),
    }


def _paired_summary(selected: pd.DataFrame, left: str, right: str):
    subset = selected[selected["pool"].isin([left, right])]
    rmsd = subset.pivot(
        index=["row_index", "complex_name"], columns="pool", values="selected_rmsd_A"
    ).dropna()
    ncc = subset.pivot(
        index=["row_index", "complex_name"], columns="pool", values="selected_ncc"
    ).dropna()
    if left not in rmsd or right not in rmsd:
        return None
    rmsd_delta = rmsd[right] - rmsd[left]
    ncc_delta = ncc[right] - ncc[left]
    try:
        pvalue = float(wilcoxon(rmsd[right], rmsd[left]).pvalue)
    except ValueError:
        pvalue = 1.0
    return {
        "left": left,
        "right": right,
        "num_paired": int(len(rmsd)),
        "median_rmsd_delta_right_minus_left_A": float(rmsd_delta.median()),
        "right_rmsd_better_pct": float(100.0 * (rmsd_delta < 0).mean()),
        "wilcoxon_rmsd_pvalue": pvalue,
        "median_ncc_delta_right_minus_left": float(ncc_delta.median()),
        "right_ncc_better_pct": float(100.0 * (ncc_delta > 0).mean()),
        "right_map_and_rmsd_improved_pct": float(
            100.0 * ((ncc_delta > 0) & (rmsd_delta < 0)).mean()
        ),
        "right_map_improved_rmsd_not_improved_pct": float(
            100.0 * ((ncc_delta > 0) & (rmsd_delta >= 0)).mean()
        ),
        "warning": (
            "Map-improved/non-RMSD-improved poses are hypotheses only; null "
            "calibration is required before calling them map-supported."
        ),
    }


def _reference_physics(
    manifest: Path,
    table: pd.DataFrame,
    *,
    severe_overlap_A: float,
    calculate_strain: bool,
    strain_variant: str,
    strain_max_iterations: int,
):
    rows, failures = [], []
    for row_index, row in table.iterrows():
        name = str(row["complex_name"])
        try:
            protein_path = resolve_path(row["experimental_protein"], manifest.parent)
            ligand_path = resolve_path(row["reference_ligand"], manifest.parent)
            protein_coordinates, protein_elements = read_protein_heavy_atoms(protein_path)
            metrics = score_ligand_pose_physics(
                protein_coordinates,
                protein_elements,
                ligand_path,
                severe_overlap_A=severe_overlap_A,
                calculate_strain=calculate_strain,
                strain_variant=strain_variant,
                strain_max_iterations=strain_max_iterations,
            )
            rows.append({
                "row_index": int(row_index), "complex_name": name,
                "experimental_protein": str(protein_path),
                "reference_ligand": str(ligand_path), **metrics,
            })
        except Exception as exc:
            failures.append({
                "row_index": int(row_index), "complex_name": name,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows), failures


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if pd.isna(value):
        return None
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--physics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pools", default="baseline,guided")
    parser.add_argument("--density-column", default="ncc_sigma_1p5")
    parser.add_argument("--density-weight", type=float, default=2.0)
    parser.add_argument("--reference-quantile", type=float, default=0.95)
    parser.add_argument("--max-severe-clashes", type=float, default=None)
    parser.add_argument("--max-clashing-ligand-fraction", type=float, default=None)
    parser.add_argument("--min-in-bounds-fraction", type=float, default=0.95)
    parser.add_argument("--max-strain-kcal-mol", type=float, default=None)
    parser.add_argument("--strain-reference-quantile", type=float, default=None)
    parser.add_argument("--missing-strain-policy", choices=["allow", "reject"], default="allow")
    parser.add_argument("--fallback-policy", choices=["best_score", "fail"], default="best_score")
    parser.add_argument("--severe-overlap", type=float, default=0.8)
    parser.add_argument("--strain-variant", default="MMFF94s")
    parser.add_argument("--strain-max-iterations", type=int, default=500)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    for quantile in (args.reference_quantile, args.strain_reference_quantile):
        if quantile is not None and not 0 <= quantile <= 1:
            raise ValueError("quantiles must be between zero and one")

    manifest = args.manifest.expanduser().resolve()
    scores_path = args.scores.expanduser().resolve()
    physics_path = args.physics.expanduser().resolve()
    table = pd.read_csv(manifest)
    scores = _normalise_paths(pd.read_csv(scores_path), "prediction_path", scores_path.parent)
    physics = _normalise_paths(pd.read_csv(physics_path), "prediction_path", physics_path.parent)
    if args.density_column not in scores:
        raise ValueError(f"Missing density column {args.density_column}")
    required_manifest = {"complex_name", "experimental_protein", "reference_ligand"}
    missing = sorted(required_manifest - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    physics_columns = MERGE_KEYS + [
        column for column in physics.columns
        if column not in MERGE_KEYS and column not in scores.columns
    ]
    merged = scores.merge(
        physics[physics_columns], on=MERGE_KEYS, how="inner", validate="one_to_one"
    )
    if len(merged) != len(scores):
        raise ValueError(
            f"Physics merge retained {len(merged)} of {len(scores)} score rows"
        )

    reference_by_index = {
        int(row_index): read_heavy_molecule(
            resolve_path(row["reference_ligand"], manifest.parent)
        )
        for row_index, row in table.iterrows()
    }
    rmsds, failures = [], []
    for pose_index, pose in merged.iterrows():
        try:
            prediction = read_heavy_molecule(Path(pose["prediction_path"]))
            rmsds.append(symmetry_rmsd(
                reference_by_index[int(pose["row_index"])], prediction
            ))
        except Exception as exc:
            rmsds.append(float("nan"))
            failures.append({
                "pose_index": int(pose_index),
                "prediction_path": pose["prediction_path"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            if args.fail_on_error:
                raise
    merged["ligand_rmsd_A"] = rmsds
    merged = merged[np.isfinite(merged["ligand_rmsd_A"])].copy()

    calculate_reference_strain = (
        args.max_strain_kcal_mol is not None
        or args.strain_reference_quantile is not None
    )
    reference, reference_failures = _reference_physics(
        manifest, table,
        severe_overlap_A=args.severe_overlap,
        calculate_strain=calculate_reference_strain,
        strain_variant=args.strain_variant,
        strain_max_iterations=args.strain_max_iterations,
    )
    if reference.empty:
        raise RuntimeError("No reference physics metrics could be calculated")
    max_clashes = args.max_severe_clashes
    if max_clashes is None:
        max_clashes = float(np.ceil(reference["severe_clash_count"].quantile(
            args.reference_quantile
        )))
    max_fraction = args.max_clashing_ligand_fraction
    if max_fraction is None:
        max_fraction = float(reference["clashing_ligand_atom_fraction"].quantile(
            args.reference_quantile
        ))
    max_strain = args.max_strain_kcal_mol
    if max_strain is None and args.strain_reference_quantile is not None:
        available = reference[_bool_series(reference["strain_available"])]
        if not available.empty:
            max_strain = float(available["strain_energy_kcal_mol"].quantile(
                args.strain_reference_quantile
            ))

    merged["physical_gate_pass"] = physical_gate_mask(
        merged,
        max_severe_clashes=max_clashes,
        max_clashing_ligand_fraction=max_fraction,
        min_in_bounds_fraction=args.min_in_bounds_fraction,
        max_strain_kcal_mol=max_strain,
        missing_strain_policy=args.missing_strain_policy,
    )

    pools = [value.strip() for value in args.pools.split(",") if value.strip()]
    known_sources = set(merged["source"].astype(str))
    unknown = sorted(set(pools) - known_sources - {"union"})
    if unknown:
        raise ValueError(f"Unknown pools {unknown}; sources are {sorted(known_sources)}")
    selected_rows = []
    for pool in pools:
        pool_frame = merged if pool == "union" else merged[merged["source"] == pool]
        for (row_index, complex_name), group in pool_frame.groupby(
            ["row_index", "complex_name"], sort=False
        ):
            result = select_pool(
                group,
                density_column=args.density_column,
                density_weight=args.density_weight,
                fallback_policy=args.fallback_policy,
            )
            record = {
                "pool": pool,
                "row_index": int(row_index),
                "complex_name": str(complex_name),
                "num_candidates": result["num_candidates"],
                "num_physical_pass": result["num_physical_pass"],
                "gate_fallback": result["fallback"],
            }
            record.update(_pose_fields("ungated", result["ungated"], args.density_column))
            record.update(_pose_fields("selected", result["selected"], args.density_column))
            selected_rows.append(record)
    selected = pd.DataFrame(selected_rows)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "per_pose_with_physics_and_rmsd.csv", index=False)
    selected.to_csv(output / "selected_poses.csv", index=False)
    reference.to_csv(output / "reference_physics.csv", index=False)
    pd.DataFrame(failures + reference_failures).to_csv(
        output / "failures.csv", index=False
    )
    pool_summary = {
        pool: _summary_for_pool(group)
        for pool, group in selected.groupby("pool", sort=False)
    }
    paired = None
    if "baseline" in pools and "guided" in pools:
        paired = _paired_summary(selected, "baseline", "guided")
    gate_config = {
        "max_severe_clashes": max_clashes,
        "max_clashing_ligand_fraction": max_fraction,
        "min_in_bounds_fraction": args.min_in_bounds_fraction,
        "max_strain_kcal_mol": max_strain,
        "missing_strain_policy": args.missing_strain_policy,
        "severe_overlap_A": args.severe_overlap,
        "calibration_reference_quantile": args.reference_quantile,
        "strain_reference_quantile": args.strain_reference_quantile,
    }
    (output / "frozen_gate.json").write_text(
        json.dumps(gate_config, indent=2, default=_json_default) + "\n"
    )
    summary = {
        "num_poses": int(len(merged)),
        "num_complexes": int(merged["row_index"].nunique()),
        "num_failures": len(failures),
        "num_reference_failures": len(reference_failures),
        "pools": pools,
        "density_column": args.density_column,
        "density_weight": args.density_weight,
        "fallback_policy": args.fallback_policy,
        "gate": gate_config,
        "pool_summary": pool_summary,
        "paired_baseline_guided": paired,
        "uses_ground_truth": True,
        "warning": (
            "Gate thresholds calibrated here are validation-only. Freeze "
            "frozen_gate.json before evaluating test data. Null calibration "
            "is still required for non-reference pose claims."
        ),
    }
    text = json.dumps(summary, indent=2, default=_json_default) + "\n"
    (output / "summary.json").write_text(text)
    print(text, end="")
    if (failures or reference_failures) and args.fail_on_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
