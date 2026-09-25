#!/usr/bin/env python3
"""Run the frozen three-phase production PAAHS pipeline.

The workflow is reference-free and resumable at phase/component boundaries:

Phase 1 performs ensemble-level density--physics guided hierarchical search.
Phase 2 performs conditional density-preserving steric repair and conservative
integration.  Phase 3 performs minimum-distance and restrained internal-
geometry refinement.  A final blind seal materializes one pose per case.

An existing Phase 1 search can be supplied with --phase1-run-root. Otherwise
the runner creates it from --search-manifest and --scores.  The former
``--step*`` command-line spellings remain accepted as deprecated aliases so
that frozen historical commands can still be reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
def require_config_subset(
    observed: object, expected: object, label: str, path: str = ""
) -> None:
    """Require behavior-defining config values while allowing provenance metadata."""
    location = f"{label}{path}"
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            raise ValueError(f"Frozen {location} must be an object")
        for key, value in expected.items():
            if key not in observed:
                raise ValueError(f"Frozen {location} lacks required field {key!r}")
            require_config_subset(observed[key], value, label, f"{path}.{key}")
        return
    if observed != expected:
        raise ValueError(
            f"Frozen {location} mismatch: expected {expected!r}, observed {observed!r}"
        )


def validate_frozen_configs(configs: list[Path]) -> None:
    payloads = [json.loads(path.read_text()) for path in configs]
    required = [
        {
            "phase": "A",
            "steric_parent_slots": 2,
            "steric_parent_max_ncc_drop": 0.02,
            "steric_beam_fraction": 0.25,
            "steric_rescue_max_ncc_drop": 0.02,
            "steric_rescue_max_anchor_rmsd_A": 0.75,
            "clash_aware_final_max_ncc_drop": 0.02,
            "clash_aware_final_max_anchor_rmsd_A": 0.75,
            "severe_overlap_A": 0.8,
            "conservative_fallback": {
                "status": "development_frozen",
                "require_old_clashing": True,
                "require_new_clash_free": True,
                "maximum_old_to_new_displacement_A": 1.5,
                "maximum_ncc_drop": 0.02,
                "accept_partial_clash_reduction": False,
                "uses_reference_or_rmsd_to_native_for_selection": False,
            },
        },
        {"configs": {"directed_b8_d3_frozen": {
            "search_depth": 3, "beam_size": 8, "generic_proposals": True,
        }}},
        {"configs": {"v4_frozen": {
            "search_depth": 3, "beam_size": 8,
            "adaptive_early_stop": True,
            "density_prescreen_multiplier": 2,
            "protein_neighbor_radius_A": 6.0,
        }}, "continuous_fallback": {
            "trigger_max_v2_residual_clashes": 1,
            "optimization_mode": "fast", "num_starts": 2,
            "max_iterations": 30, "max_torsion_bonds": 8,
        }},
        {"status": "development_frozen", "frozen_configuration": {
            "selection_mode": "interaction", "beam_width": 16,
            "coordinate_rounds": 2, "maximum_target_residues": 3,
            "maximum_sidechain_displacement_A": 1.5,
            "maximum_interaction_geometry_drop": 0.05,
        }},
        {"status": "development_frozen", "frozen_configuration": {
            "name": "c_joint050", "mode": "joint",
            "maximum_ligand_displacement_A": 0.5,
            "target_relative_vdw_distance": 0.77,
            "optimization_relative_vdw_distance": 0.78,
            "acceptance_relative_vdw_distance": 0.75,
            "maximum_ncc_drop": 0.02,
            "maximum_interaction_drop": 0.05,
        }},
        {"status": "development_frozen", "frozen_configuration": {
            "name": "ig_d035", "maximum_ligand_rmsd_A": 0.35,
            "maximum_iterations": 300,
        }, "shared_hard_gates": {
            "minimum_relative_vdw_distance": 0.77,
            "maximum_ncc_drop": 0.02,
            "maximum_interaction_drop": 0.05,
            "fallback_to_baseline": True,
        }},
    ]
    labels = (
        "phase1_search",
        "phase2_directed_repair",
        "phase2_density_repair",
        "phase2_local_environment",
        "phase3_minimum_distance",
        "phase3_internal_geometry",
    )
    for payload, expected, label in zip(payloads, required, labels):
        require_config_subset(payload, expected, label)


def absolute(path: Path) -> Path:
    return path.expanduser().resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sentinel(name: str, sentinel: Path) -> None:
    if sentinel.suffix == ".json":
        payload = json.loads(sentinel.read_text())
        failure_keys = ("failures", "failures_n", "num_failures")
        failures = {
            key: int(payload[key]) for key in failure_keys
            if key in payload and payload[key] is not None
        }
        if any(value != 0 for value in failures.values()):
            raise RuntimeError(
                f"{name} has a completion file but contains failures {failures}; "
                f"inspect the stage logs under {sentinel.parent}"
            )
    elif sentinel.name.startswith("run_status"):
        status = pd.read_csv(sentinel)
        if "status" in status:
            bad = status[
                ~status["status"].isin(["completed", "skipped_complete"])
            ]
            if len(bad):
                raise RuntimeError(
                    f"{name} has {len(bad)} failed/incomplete tasks; inspect {sentinel}"
                )


def run_stage(name: str, command: list[str], sentinel: Path) -> None:
    if sentinel.is_file():
        validate_sentinel(name, sentinel)
        print(f"[{name}] complete; reusing {sentinel}", flush=True)
        return
    print(f"[{name}] {' '.join(command)}", flush=True)
    process = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {process.returncode}")
    if not sentinel.is_file():
        raise RuntimeError(f"{name} returned successfully but did not write {sentinel}")
    validate_sentinel(name, sentinel)


def blind_names(path: Path, expected: int) -> list[str]:
    if not path.is_file():
        kind = "directory" if path.is_dir() else "missing path"
        raise FileNotFoundError(
            f"--baseline-manifest must be a CSV file, but received {kind}: {path}. "
            "If a shell variable was used, print it before running PAAHS."
        )
    frame = pd.read_csv(path)
    required = {
        "case_order", "complex_name", "baseline_pose_path", "protein_path",
        "density_crop",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Baseline manifest lacks columns: {missing}")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("reference", "native", "rmsd", "outcome"))
    ]
    if forbidden:
        raise ValueError(f"Baseline manifest contains forbidden columns: {forbidden}")
    frame["complex_name"] = frame["complex_name"].astype(str).str.upper()
    frame["case_order"] = pd.to_numeric(frame["case_order"], errors="raise").astype(int)
    frame = frame.sort_values("case_order", kind="stable")
    if len(frame) != expected or frame["complex_name"].duplicated().any():
        raise ValueError(f"Expected {expected} unique baseline cases")
    return frame["complex_name"].tolist()


def build_repair_manifest(selection: Path, output: Path, expected: int) -> None:
    if output.is_file():
        frame = pd.read_csv(output)
        if len(frame) != expected:
            raise ValueError(f"Existing repair manifest has {len(frame)} cases")
        return
    source = pd.read_csv(selection)
    if len(source) != expected or source["complex_name"].duplicated().any():
        raise ValueError(f"Expected {expected} unique Phase 1 selections")
    frame = source[[
        "case_order", "complex_name", "selected_pose_path",
        "selected_protein_path", "density_crop",
    ]].rename(columns={"selected_protein_path": "protein_path"})
    for column in ("selected_pose_path", "protein_path", "density_crop"):
        missing = [value for value in frame[column] if not Path(str(value)).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {column} inputs: {missing[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"[repair manifest] wrote {output}", flush=True)


def build_baseline_repair_manifest(
    baseline: Path,
    output: Path,
    expected: int,
    include_names: set[str] | None = None,
) -> int:
    """Write the directed-repair worklist and return its case count.

    In full mode ``include_names`` is omitted and all cases are written.  In
    lazy mode it contains only cases whose Phase 2 local-environment result still has at least one
    severe clash.  Filtering is deliberately based only on the blind physical
    endpoint, never on a reference structure or RMSD.
    """
    if output.is_file():
        frame = pd.read_csv(output)
        observed = set(frame["complex_name"].astype(str).str.upper())
        wanted = include_names
        if wanted is None:
            if len(frame) != expected:
                raise ValueError(
                    f"Existing baseline repair manifest has {len(frame)} cases"
                )
        elif observed != wanted:
            raise ValueError(
                "Existing lazy directed-repair manifest does not match the "
                "current Phase 2 unresolved set"
            )
        return len(frame)
    source = pd.read_csv(baseline)
    if len(source) != expected or source["complex_name"].duplicated().any():
        raise ValueError(f"Expected {expected} unique baseline cases")
    source["complex_name"] = source["complex_name"].astype(str).str.upper()
    if include_names is not None:
        unknown = include_names - set(source["complex_name"])
        if unknown:
            raise ValueError(f"Lazy directed worklist contains unknown cases: {sorted(unknown)}")
        source = source[source["complex_name"].isin(include_names)].copy()
    frame = source[[
        "case_order", "complex_name", "baseline_pose_path", "protein_path",
        "density_crop",
    ]].rename(columns={"baseline_pose_path": "selected_pose_path"})
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(
        f"[baseline repair manifest] wrote {len(frame)} cases to {output}",
        flush=True,
    )
    return len(frame)


def unresolved_after_phase2_local_environment(path: Path, expected: int) -> set[str]:
    frame = pd.read_csv(path)
    required = {"complex_name", "sc_final_post_severe_clash_count"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Phase 2 local-environment analysis lacks columns: {missing}")
    frame["complex_name"] = frame["complex_name"].astype(str).str.upper()
    if len(frame) != expected or frame["complex_name"].duplicated().any():
        raise ValueError(f"Expected {expected} unique Phase 2 cases")
    counts = pd.to_numeric(
        frame["sc_final_post_severe_clash_count"], errors="raise"
    ).astype(int)
    if (counts < 0).any():
        raise ValueError("Phase 2 severe-clash counts must be non-negative")
    return set(frame.loc[counts > 0, "complex_name"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--search-manifest", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument(
        "--phase1-run-root", "--step1-run-root", dest="phase1_run_root", type=Path,
        help="Reuse an existing complete Phase 1 search instead of generating it",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--directed-fallback-mode",
        choices=("full", "lazy"),
        default="full",
        help=(
            "full reproduces the frozen tie-retaining union; lazy runs D3B8 only "
            "when Phase 2 retains a historical custom severe clash and uses the "
            "custom-severe zero lower bound otherwise; PoseBusters-aligned "
            "minimum distance is handled later by Phase 3"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Validate blind inputs, cohort alignment, and frozen hashes without running",
    )
    parser.add_argument(
        "--phase1-config", "--step1-config", dest="phase1_config", type=Path,
        default=PROJECT_ROOT / "configs/clash_aware_adaptive_hierarchical_search_a.json",
    )
    parser.add_argument(
        "--phase2-density-config", "--step2-v4-config", dest="phase2_density_config", type=Path,
        default=PROJECT_ROOT / "configs/density_repair_v4_frozen.json",
    )
    parser.add_argument(
        "--phase2-directed-config", "--step2-directed-config", dest="phase2_directed_config", type=Path,
        default=PROJECT_ROOT / "configs/density_repair_validation62_v2_frozen.json",
    )
    parser.add_argument(
        "--phase2-local-environment-config", "--step3-config", dest="phase2_local_environment_config", type=Path,
        default=PROJECT_ROOT / "configs/caahs_a_sidechain_rescue_frozen.json",
    )
    parser.add_argument(
        "--phase3-distance-config", "--step5-config", dest="phase3_distance_config", type=Path,
        default=PROJECT_ROOT / "configs/caahs_c_frozen_joint050.json",
    )
    parser.add_argument(
        "--phase3-geometry-config", "--step6-config", dest="phase3_geometry_config", type=Path,
        default=PROJECT_ROOT / "configs/caahs_c_internal_geometry_frozen_d035.json",
    )
    args = parser.parse_args()
    if args.expected_cases < 1 or args.workers < 1:
        raise ValueError("expected-cases and workers must be positive")
    if args.phase1_run_root is None and (args.search_manifest is None or args.scores is None):
        parser.error("provide --search-manifest and --scores, or --phase1-run-root")

    baseline = absolute(args.baseline_manifest)
    output = absolute(args.output_root)
    if output == PROJECT_ROOT:
        raise ValueError(
            "--output-root resolved to the project directory. The shell variable "
            "is probably empty; use a dedicated new output directory."
        )
    names = blind_names(baseline, args.expected_cases)
    configs = [
        absolute(args.phase1_config), absolute(args.phase2_directed_config),
        absolute(args.phase2_density_config),
        absolute(args.phase2_local_environment_config),
        absolute(args.phase3_distance_config),
        absolute(args.phase3_geometry_config),
    ]
    for path in [baseline, *configs]:
        if not path.is_file():
            raise FileNotFoundError(path)
    config_labels = (
        "phase1_search",
        "phase2_directed_repair",
        "phase2_density_repair",
        "phase2_local_environment",
        "phase3_minimum_distance",
        "phase3_internal_geometry",
    )
    validate_frozen_configs(configs)
    observed_config_hashes = {
        label: sha256(path) for label, path in zip(config_labels, configs)
    }
    phase1_config = json.loads(configs[0].read_text())
    if args.phase1_run_root is None:
        search_manifest = absolute(args.search_manifest)
        scores = absolute(args.scores)
        for path in (search_manifest, scores):
            if not path.is_file():
                raise FileNotFoundError(path)
        search = pd.read_csv(search_manifest)
        search_names = search["complex_name"].astype(str).str.upper().tolist()
        if search_names != names:
            raise ValueError(
                "Search manifest identities/order must exactly match baseline manifest"
            )
        phase1_root = output / "phase1" / "search"
        search_command = [
            args.python,
            "-u",
            str(PROJECT_ROOT / "scripts/run_clash_aware_adaptive_hierarchical_search.py"),
            "--manifest", str(search_manifest),
            "--scores", str(scores),
            "--output-dir", str(phase1_root),
            "--experiment-id", "paahs_production",
            "--device", args.device,
            "--materialization-mode", "top1",
            "--steric-parent-slots", str(phase1_config["steric_parent_slots"]),
            "--steric-parent-max-ncc-drop", str(phase1_config["steric_parent_max_ncc_drop"]),
            "--steric-beam-fraction", str(phase1_config["steric_beam_fraction"]),
            "--steric-rescue-max-ncc-drop", str(phase1_config["steric_rescue_max_ncc_drop"]),
            "--steric-rescue-max-anchor-rmsd-A", str(phase1_config["steric_rescue_max_anchor_rmsd_A"]),
            "--clash-aware-final-max-ncc-drop", str(phase1_config["clash_aware_final_max_ncc_drop"]),
            "--clash-aware-final-max-anchor-rmsd-A", str(phase1_config["clash_aware_final_max_anchor_rmsd_A"]),
            "--severe-overlap", str(phase1_config["severe_overlap_A"]),
            "--fail-on-error",
        ]
        if not args.preflight_only:
            output.mkdir(parents=True, exist_ok=True)
            run_stage("phase1 search", search_command, phase1_root / "summary.json")
    else:
        phase1_root = absolute(args.phase1_run_root)
        if not (phase1_root / "summary.json").is_file():
            raise FileNotFoundError(phase1_root / "summary.json")
        validate_sentinel("external phase1", phase1_root / "summary.json")
        run_manifest = pd.read_csv(phase1_root / "evaluation_manifest.csv")
        run_names = run_manifest["complex_name"].astype(str).str.upper().tolist()
        if run_names != names:
            raise ValueError(
                "Existing Phase 1 identities/order do not match baseline manifest"
            )
        print(f"[phase1] reusing external search {phase1_root}", flush=True)

    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_ok",
            "cases_n": len(names),
            "phase1_mode": "reuse" if args.phase1_run_root else "generate",
            "directed_fallback_mode": args.directed_fallback_mode,
            "config_sha256": observed_config_hashes,
            "config_validation": "behavior_defining_fields_match_frozen_policy",
            "uses_reference_or_native_rmsd_for_selection": False,
        }, indent=2))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    (output / "paahs_plan.json").write_text(json.dumps({
        "method": "PAAHS",
        "directed_fallback_mode": args.directed_fallback_mode,
        "phases": [
            {"phase": "phase1", "purpose": "density--physics guided hierarchical search"},
            {"phase": "phase2", "purpose": "density-preserving steric repair and conservative integration"},
            {"phase": "phase3", "purpose": "minimum-distance and restrained internal-geometry refinement"},
        ],
        "finalization": "blind seal and materialization",
        "uses_reference_or_native_rmsd_for_selection": False,
    }, indent=2) + "\n")

    selection_dir = output / "phase1" / "selection"
    run_stage(
        "phase1 selection",
        [
            args.python, str(PROJECT_ROOT / "scripts/select_caahs_standalone_fallback.py"),
            "--baseline-manifest", str(baseline),
            "--caahs-root", str(phase1_root),
            "--config-json", str(configs[0]),
            "--output-dir", str(selection_dir),
            "--expected-cases", str(args.expected_cases),
            "--device", args.device,
        ],
        selection_dir / "selection_seal.json",
    )
    selection_csv = selection_dir / "selected_poses_blind.csv"
    repair_manifest = output / "phase2" / "input" / "repair_manifest.csv"
    build_repair_manifest(selection_csv, repair_manifest, args.expected_cases)

    v4_root = output / "phase2" / "density_preserving_repair" / "v4"
    run_stage(
        "phase2 density-preserving V4 repair",
        [
            args.python, str(PROJECT_ROOT / "scripts/run_density_preserving_repair_validation.py"),
            "--manifest", str(repair_manifest),
            "--config-json", str(configs[2]),
            "--output-root", str(v4_root),
            "--expected-cases", str(args.expected_cases),
            "--workers", str(args.workers),
        ],
        v4_root / "run_status_task00.csv",
    )
    v4_analysis = output / "phase2" / "density_preserving_repair" / "v4_analysis"
    run_stage(
        "phase2 V4 analysis",
        [
            args.python, str(PROJECT_ROOT / "evaluation/analyze_density_repair_validation_sweep.py"),
            "--sweep-root", str(v4_root),
            "--output-dir", str(v4_analysis),
            "--expected-cases", str(args.expected_cases),
        ],
        v4_analysis / "per_case_validation_audit.csv",
    )

    v4_config = json.loads(configs[2].read_text())
    if len(v4_config["configs"]) != 1:
        raise ValueError("Production V4 config must contain exactly one variant")
    v4_name = next(iter(v4_config["configs"]))
    fallback = v4_config["continuous_fallback"]
    v4_cases = v4_root / v4_name / "cases"
    fast_root = output / "phase2" / "density_preserving_repair" / "fast_fallback"
    run_stage(
        "phase2 fast fallback",
        [
            args.python, str(PROJECT_ROOT / "scripts/run_continuous_clash_refinement_validation.py"),
            "--v2-case-root", str(v4_cases),
            "--output-root", str(fast_root),
            "--expected-cases", str(args.expected_cases),
            "--workers", str(args.workers),
            "--num-starts", str(fallback["num_starts"]),
            "--max-iterations", str(fallback["max_iterations"]),
            "--max-torsion-bonds", str(fallback["max_torsion_bonds"]),
            "--optimization-mode", str(fallback["optimization_mode"]),
            "--max-v2-residual-clashes", str(fallback["trigger_max_v2_residual_clashes"]),
        ],
        fast_root / "run_summary.json",
    )
    fast_analysis = output / "phase2" / "density_preserving_repair" / "analysis"
    run_stage(
        "phase2 density-preserving repair analysis",
        [
            args.python, str(PROJECT_ROOT / "evaluation/analyze_continuous_clash_refinement.py"),
            "--v2-case-root", str(v4_cases),
            "--v3-root", str(fast_root),
            "--output-dir", str(fast_analysis),
            "--expected-cases", str(args.expected_cases),
        ],
        fast_analysis / "per_case_continuous_refinement.csv",
    )

    sc_root = output / "phase2" / "local_environment_refinement" / "run"
    run_stage(
        "phase2 optional local-environment refinement",
        [
            args.python, str(PROJECT_ROOT / "scripts/run_caahs_a_sidechain_rescue_validation.py"),
            "--a-final-analysis", str(fast_analysis),
            "--a-repair-manifest", str(repair_manifest),
            "--output-root", str(sc_root),
            "--expected-cases", str(args.expected_cases),
            "--workers", str(args.workers),
            "--config-json", str(configs[3]),
        ],
        sc_root / "run_summary.json",
    )
    sc_analysis = output / "phase2" / "local_environment_refinement" / "analysis"
    run_stage(
        "phase2 local-environment analysis",
        [
            args.python, str(PROJECT_ROOT / "evaluation/analyze_caahs_a_sidechain_rescue.py"),
            "--a-final-analysis", str(fast_analysis),
            "--sc-root", str(sc_root),
            "--output-dir", str(sc_analysis),
            "--expected-cases", str(args.expected_cases),
        ],
        sc_analysis / "per_case_caahs_a_sidechain_rescue.csv",
    )

    baseline_repair_manifest = output / "phase2" / "input" / "baseline_repair_manifest.csv"
    lazy_names = None
    if args.directed_fallback_mode == "lazy":
        lazy_names = unresolved_after_phase2_local_environment(
            sc_analysis / "per_case_caahs_a_sidechain_rescue.csv",
            args.expected_cases,
        )
    directed_cases_n = build_baseline_repair_manifest(
        baseline,
        baseline_repair_manifest,
        args.expected_cases,
        include_names=lazy_names,
    )
    directed_root = output / "phase2" / "density_preserving_repair" / "directed_fallback"
    if directed_cases_n:
        run_stage(
            "phase2 lazy directed fallback",
            [
                args.python, str(PROJECT_ROOT / "scripts/run_density_preserving_repair_validation.py"),
                "--manifest", str(baseline_repair_manifest),
                "--config-json", str(configs[1]),
                "--output-root", str(directed_root),
                "--expected-cases", str(directed_cases_n),
                "--workers", str(args.workers),
            ],
            directed_root / "run_status_task00.csv",
        )
    else:
        print(
            "[phase2 directed fallback] lazy custom-severe worklist is empty; "
            "skipping D3B8",
            flush=True,
        )
    directed_config = json.loads(configs[1].read_text())
    if len(directed_config["configs"]) != 1:
        raise ValueError("Directed repair config must contain exactly one variant")
    directed_name = next(iter(directed_config["configs"]))
    strict_union = output / "phase2" / "conservative_integration"
    run_stage(
        "phase2 conservative integration",
        [
            args.python, str(PROJECT_ROOT / "evaluation/build_paahs_phase2_integration.py"),
            "--baseline-case-root", str(directed_root / directed_name / "cases"),
            "--new-analysis", str(sc_analysis),
            "--new-repair-manifest", str(repair_manifest),
            "--output-dir", str(strict_union),
            "--expected-cases", str(args.expected_cases),
            "--directed-mode", args.directed_fallback_mode,
        ],
        strict_union / "per_case_strict_clash_union.csv",
    )

    c_root = output / "phase3" / "minimum_distance_refinement" / "run"
    run_stage(
        "phase3 minimum-distance refinement",
        [
            args.python, str(PROJECT_ROOT / "scripts/run_caahs_c_development_sweep.py"),
            "--baseline-cases", str(strict_union / "per_case_strict_clash_union.csv"),
            "--repair-manifest", str(repair_manifest),
            "--config-json", str(configs[4]),
            "--output-root", str(c_root),
            "--expected-cases", str(args.expected_cases),
            "--workers", str(args.workers),
        ],
        c_root / "run_summary.json",
    )
    c_analysis = output / "phase3" / "minimum_distance_refinement" / "analysis"
    run_stage(
        "phase3 minimum-distance analysis",
        [
            args.python, str(PROJECT_ROOT / "evaluation/analyze_caahs_c_development_sweep.py"),
            "--sweep-root", str(c_root),
            "--config-json", str(configs[4]),
            "--output-dir", str(c_analysis),
            "--expected-cases", str(args.expected_cases),
        ],
        c_analysis / "per_case_caahs_c_development_sweep.csv",
    )

    ig_root = output / "phase3" / "internal_geometry_refinement" / "run"
    run_stage(
        "phase3 restrained internal-geometry refinement",
        [
            args.python, str(PROJECT_ROOT / "scripts/run_caahs_c_internal_geometry_sweep.py"),
            "--baseline-cases", str(c_analysis / "per_case_caahs_c_development_sweep.csv"),
            "--repair-manifest", str(repair_manifest),
            "--config-json", str(configs[5]),
            "--output-root", str(ig_root),
            "--expected-cases", str(args.expected_cases),
            "--workers", str(args.workers),
        ],
        ig_root / "run_summary.json",
    )
    ig_analysis = output / "phase3" / "internal_geometry_refinement" / "analysis"
    run_stage(
        "phase3 internal-geometry analysis",
        [
            args.python, str(PROJECT_ROOT / "evaluation/analyze_caahs_c_internal_geometry_sweep.py"),
            "--sweep-root", str(ig_root),
            "--config-json", str(configs[5]),
            "--output-dir", str(ig_analysis),
            "--expected-cases", str(args.expected_cases),
        ],
        ig_analysis / "per_case_internal_geometry_refinement.csv",
    )

    final = output / "final"
    finalize_command = [
        args.python, str(PROJECT_ROOT / "evaluation/finalize_paahs_pipeline.py"),
        "--baseline-manifest", str(baseline),
        "--phase1-selection", str(selection_csv),
        "--repair-manifest", str(repair_manifest),
        "--phase2-density-repair-analysis", str(fast_analysis),
        "--phase2-local-environment-analysis", str(sc_analysis),
        "--phase2-integration", str(strict_union),
        "--phase3-distance-analysis", str(c_analysis),
        "--phase3-geometry-analysis", str(ig_analysis),
        "--output-dir", str(final),
        "--expected-cases", str(args.expected_cases),
        "--directed-fallback-mode", args.directed_fallback_mode,
    ]
    for config in configs:
        finalize_command.extend(["--config-json", str(config)])
    run_stage("final seal", finalize_command, final / "summary.json")
    print(f"PAAHS_PIPELINE_OK: {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
