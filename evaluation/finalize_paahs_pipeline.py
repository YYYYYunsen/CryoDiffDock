#!/usr/bin/env python3
"""Seal and materialize the three-phase production PAAHS pipeline output.

This module deliberately does not read a reference ligand, native pose, RMSD,
or outcome label.  It joins the reference-free stage audits, verifies that each
optional component preserves the preceding physical success, and writes one stable
ligand/protein pair per case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import pandas as pd


FORBIDDEN_TOKENS = ("reference", "native", "rmsd", "outcome", "success")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEVERE_OVERLAP_A = 0.8
SERIALIZATION_BOUNDARY_TOLERANCE_A = 0.001


def phase2_serialized_clash_count(
    name: str, union_row: pd.Series, phase3_baseline_count: int,
    *, directed_fallback_mode: str,
) -> tuple[int, bool, float | None, float | None]:
    """Audit a V2000 coordinate-rounding flip at the severe-clash boundary.

    Normally Phase 2's recorded count must equal the Phase 3 baseline.  The
    only exception is a lazy zero-clash selection whose serialized SDF can be
    independently re-read and shown to cross the frozen 0.8-A threshold.
    """

    recorded_count = int(union_row["selected_severe_clash_count"])
    if phase3_baseline_count == recorded_count:
        return recorded_count, False, None, None
    if not (
        directed_fallback_mode == "lazy"
        and recorded_count == 0
        and phase3_baseline_count == 1
        and not bool(union_row["directed_evaluated"])
        and str(union_row["selected_method"]) == "new_arm_custom_severe_zero_bound"
    ):
        raise AssertionError(f"{name}: Phase 3 baseline does not match Phase 2 integration")

    pose = Path(str(union_row["selected_pose_path"])).expanduser().resolve()
    protein = Path(str(union_row["selected_protein_path"])).expanduser().resolve()
    summary_path = pose.parent / "summary.json"
    if not (pose.is_file() and protein.is_file() and summary_path.is_file()):
        raise AssertionError(f"{name}: missing artifacts for Phase 2 boundary recount")
    summary = json.loads(summary_path.read_text())
    selected = summary.get("selected", {})
    source_count = int(selected.get("severe_clash_count", -1))
    source_overlap = float(selected.get("max_vdw_overlap_A", float("nan")))
    if not (
        source_count == recorded_count
        and math.isfinite(source_overlap)
        and SEVERE_OVERLAP_A - SERIALIZATION_BOUNDARY_TOLERANCE_A
        <= source_overlap <= SEVERE_OVERLAP_A
    ):
        raise AssertionError(f"{name}: Phase 2 source is not at the severe-clash boundary")

    # Import only for the exceptional case; most seals need no structure read.
    # Slurm launches this file by absolute path, so its sibling package is not
    # automatically on sys.path even when the process cwd is the project root.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from cryodiffdock_pocket.clash_repair import clash_field
    from cryodiffdock_pocket.physics import read_protein_heavy_atoms
    from evaluation.evaluate_ligand_accuracy import read_heavy_molecule

    molecule = read_heavy_molecule(pose)
    protein_xyz, protein_elements = read_protein_heavy_atoms(protein)
    ligand_xyz = molecule.GetConformer().GetPositions()
    ligand_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    serialized = clash_field(
        protein_xyz, protein_elements, ligand_xyz, ligand_elements,
        severe_overlap_A=SEVERE_OVERLAP_A,
    )
    serialized_overlap = float(serialized.max_overlap_A)
    if not (
        serialized.severe_count == phase3_baseline_count
        and SEVERE_OVERLAP_A < serialized_overlap
        <= SEVERE_OVERLAP_A + SERIALIZATION_BOUNDARY_TOLERANCE_A
    ):
        raise AssertionError(f"{name}: serialized pose does not explain Phase 3 baseline")
    return serialized.severe_count, True, source_overlap, serialized_overlap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_unique(path: Path, expected: int, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    frame = pd.read_csv(path)
    if "complex_name" not in frame:
        raise ValueError(f"{label} lacks complex_name")
    frame["complex_name"] = frame["complex_name"].astype(str).str.upper()
    if len(frame) != expected or frame["complex_name"].duplicated().any():
        raise ValueError(
            f"{label}: expected {expected} unique cases, found {len(frame)} rows "
            f"and {frame['complex_name'].nunique()} unique names"
        )
    return frame


def require_same_names(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    expected = set(reference["complex_name"])
    observed = set(other["complex_name"])
    if expected != observed:
        raise ValueError(
            f"{label} case mismatch; missing={sorted(expected - observed)[:10]}, "
            f"extra={sorted(observed - expected)[:10]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument(
        "--phase1-selection", "--step1-selection",
        dest="phase1_selection", required=True, type=Path,
    )
    parser.add_argument("--repair-manifest", required=True, type=Path)
    parser.add_argument(
        "--phase2-density-repair-analysis", "--continuous-analysis",
        dest="phase2_density_repair_analysis", required=True, type=Path,
    )
    parser.add_argument(
        "--phase2-local-environment-analysis", "--sidechain-analysis",
        dest="phase2_local_environment_analysis", required=True, type=Path,
    )
    parser.add_argument(
        "--phase2-integration", "--strict-union",
        dest="phase2_integration", required=True, type=Path,
    )
    parser.add_argument(
        "--phase3-distance-analysis", "--step5-analysis",
        dest="phase3_distance_analysis", required=True, type=Path,
    )
    parser.add_argument(
        "--phase3-geometry-analysis", "--internal-geometry-analysis",
        dest="phase3_geometry_analysis", required=True, type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument(
        "--directed-fallback-mode",
        choices=("full", "lazy"),
        default="full",
    )
    parser.add_argument("--config-json", action="append", type=Path, default=[])
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    # The baseline seal must remain blind.  Reject rather than silently ignore
    # any outcome-bearing column accidentally introduced upstream.
    header = pd.read_csv(args.baseline_manifest, nrows=0).columns.tolist()
    forbidden = [
        column for column in header
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Baseline manifest contains forbidden columns: {forbidden}")

    baseline = read_unique(
        args.baseline_manifest, args.expected_cases, "baseline manifest"
    )
    required_baseline = {
        "case_order", "complex_name", "baseline_pose_path", "protein_path",
        "density_crop",
    }
    if missing := sorted(required_baseline - set(baseline.columns)):
        raise ValueError(f"Baseline manifest lacks columns: {missing}")
    baseline["case_order"] = pd.to_numeric(
        baseline["case_order"], errors="raise"
    ).astype(int)
    if baseline["case_order"].duplicated().any():
        raise ValueError("Baseline manifest has duplicate case_order values")

    tables = {
        "phase1": read_unique(
            args.phase1_selection, args.expected_cases, "phase1 selection"
        ),
        "repair": read_unique(args.repair_manifest, args.expected_cases, "repair manifest"),
        "continuous": read_unique(
            args.phase2_density_repair_analysis / "per_case_continuous_refinement.csv",
            args.expected_cases,
            "V4/continuous analysis",
        ),
        "sc": read_unique(
            args.phase2_local_environment_analysis
            / "per_case_caahs_a_sidechain_rescue.csv",
            args.expected_cases,
            "side-chain analysis",
        ),
        "union": read_unique(
            args.phase2_integration / "per_case_strict_clash_union.csv",
            args.expected_cases,
            "strict clash union",
        ),
        "c": read_unique(
            args.phase3_distance_analysis
            / "per_case_caahs_c_development_sweep.csv",
            args.expected_cases,
            "phase3 minimum-distance analysis",
        ),
        "ig": read_unique(
            args.phase3_geometry_analysis
            / "per_case_internal_geometry_refinement.csv",
            args.expected_cases,
            "internal-geometry analysis",
        ),
    }
    for label, frame in tables.items():
        require_same_names(baseline, frame, label)

    phase1 = tables["phase1"].set_index("complex_name")
    continuous = tables["continuous"].set_index("complex_name")
    sc = tables["sc"].set_index("complex_name")
    union = tables["union"].set_index("complex_name")
    c = tables["c"].set_index("complex_name")
    ig = tables["ig"].set_index("complex_name")

    rows: list[dict[str, object]] = []
    cases_root = output / "cases"
    for record in baseline.sort_values("case_order", kind="stable").itertuples(index=False):
        name = str(record.complex_name)
        ar, vr, sr, ur, cr, gr = (
            phase1.loc[name], continuous.loc[name], sc.loc[name], union.loc[name],
            c.loc[name], ig.loc[name]
        )

        a_count = int(ar["selected_severe_clash_count"])
        v4_count = int(vr["v2_post_severe_clash_count"])
        fast_count = int(vr["final_post_severe_clash_count"])
        sc_count = int(sr["sc_final_post_severe_clash_count"])
        union_count = int(ur["selected_severe_clash_count"])
        c_baseline_count = int(cr["baseline_severe_clash_count"])
        c_count = int(cr["selected_severe_clash_count"])
        ig_baseline_count = gr.get("baseline_severe_clash_count")
        ig_selected_count = gr.get("selected_severe_clash_count")
        ig_baseline_count = c_count if pd.isna(ig_baseline_count) else int(ig_baseline_count)
        ig_selected_count = c_count if pd.isna(ig_selected_count) else int(ig_selected_count)

        if not (v4_count <= a_count and fast_count <= v4_count):
            raise AssertionError(f"{name}: V4/fast worsened severe clashes")
        if sc_count > fast_count:
            raise AssertionError(f"{name}: SC worsened severe clashes")
        directed_evaluated = bool(ur.get("directed_evaluated", True))
        directed_count = ur.get("baseline_directed_severe_clash_count")
        if directed_evaluated:
            if pd.isna(directed_count):
                raise AssertionError(f"{name}: evaluated directed result lacks a count")
            if union_count > int(directed_count):
                raise AssertionError(f"{name}: strict union lost baseline repair")
        elif not (
            args.directed_fallback_mode == "lazy"
            and union_count == 0
            and str(ur["selected_method"])
            == "new_arm_custom_severe_zero_bound"
        ):
            raise AssertionError(f"{name}: invalid lazy directed omission")
        (
            serialized_union_count, boundary_recount,
            source_overlap, serialized_overlap,
        ) = phase2_serialized_clash_count(
            name, ur, c_baseline_count,
            directed_fallback_mode=args.directed_fallback_mode,
        )
        if c_count > serialized_union_count:
            raise AssertionError(f"{name}: Phase 3 minimum-distance refinement worsened severe clashes")
        if ig_baseline_count != c_count or ig_selected_count > ig_baseline_count:
            raise AssertionError(f"{name}: internal-geometry stage broke rollback")
        if bool(cr["baseline_pass"]) and not bool(cr["selected_pass"]):
            raise AssertionError(f"{name}: Phase 3 lost a minimum-distance pass")
        if bool(gr["baseline_internal_pass"]) and not bool(gr["selected_internal_pass"]):
            raise AssertionError(f"{name}: internal-geometry stage lost a pass")

        source_pose = Path(str(gr["selected_pose_path"])).expanduser().resolve()
        source_protein = Path(str(gr["selected_protein_path"])).expanduser().resolve()
        for path in (source_pose, source_protein):
            if not path.is_file():
                raise FileNotFoundError(f"{name}: missing final artifact {path}")
        case_dir = cases_root / name
        case_dir.mkdir(parents=True, exist_ok=False)
        final_pose = case_dir / "rank1.sdf"
        final_protein = case_dir / "rank1_protein.pdb"
        shutil.copy2(source_pose, final_pose)
        shutil.copy2(source_protein, final_protein)

        rows.append({
            "case_order": int(record.case_order),
            "complex_name": name,
            "selected_pose_path": str(final_pose.resolve()),
            "selected_protein_path": str(final_protein.resolve()),
            "density_crop": str(Path(str(record.density_crop)).expanduser().resolve()),
            "baseline_pose_path": str(Path(str(record.baseline_pose_path)).expanduser().resolve()),
            "baseline_protein_path": str(Path(str(record.protein_path)).expanduser().resolve()),
            "phase1_selected": bool(ar["caahs_accepted"]),
            "baseline_severe_clash_count": int(ar["old_severe_clash_count"]),
            "phase1_severe_clash_count": a_count,
            "phase2_v4_severe_clash_count": v4_count,
            "phase2_density_repair_severe_clash_count": fast_count,
            "phase2_local_environment_severe_clash_count": sc_count,
            "phase2_final_severe_clash_count": union_count,
            "phase2_serialized_severe_clash_count": serialized_union_count,
            "phase2_serialization_boundary_recount": boundary_recount,
            "phase2_source_max_overlap_A": source_overlap,
            "phase2_serialized_max_overlap_A": serialized_overlap,
            "phase2_integration_method": str(ur["selected_method"]),
            "phase2_directed_evaluated": directed_evaluated,
            "phase3_severe_clash_count": c_count,
            "final_severe_clash_count": ig_selected_count,
            "phase3_minimum_distance_pass": bool(cr["selected_pass"]),
            "phase3_minimum_relative_vdw_distance": float(
                cr["selected_minimum_relative_vdw_distance"]
            ),
            "phase3_internal_geometry_pass": bool(gr["selected_internal_pass"]),
            "final_internal_clash_count": int(gr["selected_internal_clash_count"]),
            "phase3_target_gates_pass": bool(
                cr["selected_pass"] and gr["selected_internal_pass"]
            ),
            "phase2_density_repair_method": str(vr["selected_method"]),
            "phase2_local_environment_method": (
                "selected" if str(sr["selected_method"]) == "caahs_a_sc" else "retained"
            ),
            "phase3_minimum_distance_method": (
                "retained" if str(cr["selected_method"]) == "baseline_retained"
                else "selected"
            ),
            "phase3_internal_geometry_method": (
                "selected"
                if str(gr["selected_method"]) == "restrained_internal_geometry"
                else "retained"
            ),
            "uses_reference_or_native_rmsd_for_selection": False,
        })

    frame = pd.DataFrame(rows).sort_values("case_order", kind="stable")
    output.mkdir(parents=True, exist_ok=True)
    final_csv = output / "selected_poses.csv"
    frame.to_csv(final_csv, index=False)
    frame.to_csv(output / "per_case_pipeline_audit.csv", index=False)
    frame[[
        "case_order", "complex_name", "selected_pose_path",
        "selected_protein_path", "density_crop",
    ]].to_csv(output / "posebusters_input_manifest.csv", index=False)

    config_hashes = {}
    for path in args.config_json:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        config_hashes[str(resolved)] = sha256(resolved)
    summary = {
        "status": "production_pipeline_complete",
        "method": "PAAHS",
        "description": "Physics-Aware Adaptive Hierarchical Search",
        "directed_fallback_mode": args.directed_fallback_mode,
        "directed_evaluated_n": int(frame["phase2_directed_evaluated"].sum()),
        "phase2_serialization_boundary_recount_n": int(
            frame["phase2_serialization_boundary_recount"].sum()
        ),
        "phases": ["phase1", "phase2", "phase3"],
        "phase_descriptions": {
            "phase1": "density--physics guided hierarchical search",
            "phase2": "density-preserving steric repair and conservative integration",
            "phase3": "minimum-distance and restrained internal-geometry refinement",
        },
        "cases_n": len(frame),
        "baseline_clash_free_n": int(frame["baseline_severe_clash_count"].eq(0).sum()),
        "final_clash_free_n": int(frame["final_severe_clash_count"].eq(0).sum()),
        "final_clash_free_pct": float(100.0 * frame["final_severe_clash_count"].eq(0).mean()),
        "phase3_minimum_distance_pass_n": int(frame["phase3_minimum_distance_pass"].sum()),
        "phase3_internal_geometry_pass_n": int(frame["phase3_internal_geometry_pass"].sum()),
        "phase3_target_gates_pass_n": int(frame["phase3_target_gates_pass"].sum()),
        "phase1_selected_n": int(frame["phase1_selected"].sum()),
        "phase2_local_environment_selected_n": int(
            frame["phase2_local_environment_method"].eq("selected").sum()
        ),
        "phase3_minimum_distance_selected_n": int(
            frame["phase3_minimum_distance_method"].eq("selected").sum()
        ),
        "phase3_internal_geometry_selected_n": int(
            frame["phase3_internal_geometry_method"].eq("selected").sum()
        ),
        "uses_reference_or_native_rmsd_for_selection": False,
        "selected_poses_sha256": sha256(final_csv),
        "config_sha256": config_hashes,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"PAAHS_PIPELINE_FINALIZED: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
