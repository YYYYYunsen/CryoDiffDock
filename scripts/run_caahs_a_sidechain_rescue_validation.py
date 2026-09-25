#!/usr/bin/env python3
"""Run CAAHS-A+SC on unresolved validation cases without native information."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def run_one(task: dict[str, object]) -> dict[str, object]:
    output = Path(task["output"])
    summary = output / "summary.json"
    if summary.is_file() and (output / "selected_protein.pdb").is_file():
        return {**task, "status": "skipped_complete", "returncode": 0}
    if output.exists():
        return {**task, "status": "incomplete_existing_output", "returncode": -1}
    command = [
        str(task["python"]), str(task["script"]),
        "--ligand", str(task["ligand"]),
        "--protein", str(task["protein"]),
        "--output-dir", str(output),
        "--selection-mode", str(task["selection_mode"]),
        f"--chi-offsets={task['chi_offsets']}",
        "--beam-width", str(task["beam_width"]),
        "--coordinate-rounds", str(task["coordinate_rounds"]),
        "--max-residues", str(task["max_residues"]),
        "--max-sidechain-displacement-A", str(task["max_sidechain_displacement_A"]),
        "--maximum-interaction-drop", str(task["maximum_interaction_drop"]),
    ]
    log = Path(task["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False
        )
    ok = process.returncode == 0 and summary.is_file()
    return {
        **task,
        "status": "completed" if ok else "failed",
        "returncode": int(process.returncode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-final-analysis", required=True, type=Path)
    parser.add_argument("--a-repair-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--selection-mode", choices=("steric", "interaction"))
    parser.add_argument("--chi-offsets", default="-60,-30,30,60")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--coordinate-rounds", type=int, default=2)
    parser.add_argument("--max-residues", type=int, default=3)
    parser.add_argument("--max-sidechain-displacement-A", type=float, default=1.5)
    parser.add_argument("--maximum-interaction-drop", type=float, default=0.05)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    config_sha256 = None
    if args.config_json is not None:
        config_bytes = args.config_json.read_bytes()
        config = json.loads(config_bytes)
        if config.get("status") != "development_frozen":
            raise ValueError("CAAHS-A+SC configuration is not development-frozen")
        frozen = config["frozen_configuration"]
        args.selection_mode = frozen["selection_mode"]
        args.chi_offsets = ",".join(str(value) for value in frozen["chi_offsets_degrees"])
        args.beam_width = int(frozen["beam_width"])
        args.coordinate_rounds = int(frozen["coordinate_rounds"])
        args.max_residues = int(frozen["maximum_target_residues"])
        args.max_sidechain_displacement_A = float(
            frozen["maximum_sidechain_displacement_A"]
        )
        args.maximum_interaction_drop = float(
            frozen["maximum_interaction_geometry_drop"]
        )
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    elif args.selection_mode is None:
        parser.error("provide --selection-mode or --config-json")

    case_path = args.a_final_analysis / "per_case_continuous_refinement.csv"
    if not case_path.is_file():
        raise FileNotFoundError(
            f"Missing A-final per-case table: {case_path}. "
            "Set --a-final-analysis to the completed CAAHS-A+V4+fast "
            "analysis directory, not an empty variable or selection directory."
        )
    if not args.a_repair_manifest.is_file():
        raise FileNotFoundError(
            f"Missing A repair manifest: {args.a_repair_manifest}. "
            "Set --a-repair-manifest to the matching cohort manifest."
        )
    cases = pd.read_csv(case_path)
    proteins = pd.read_csv(args.a_repair_manifest, usecols=["complex_name", "protein_path"])
    if len(cases) != args.expected_cases or cases["complex_name"].duplicated().any():
        raise ValueError(f"Expected {args.expected_cases} unique A-final cases")
    table = cases.merge(proteins, on="complex_name", validate="one_to_one")
    unresolved = table[table["final_post_severe_clash_count"].gt(0)].copy()

    script = Path(__file__).resolve().with_name("refine_caahs_a_sidechain_rotamers.py")
    tasks = []
    for row in unresolved.itertuples(index=False):
        tasks.append({
            "complex_name": str(row.complex_name),
            "ligand": str(Path(row.selected_pose_path).resolve()),
            "protein": str(Path(row.protein_path).resolve()),
            "output": str(args.output_root / "cases" / str(row.complex_name)),
            "log": str(args.output_root / "logs" / f"{row.complex_name}.log"),
            "selection_mode": args.selection_mode,
            "chi_offsets": args.chi_offsets,
            "beam_width": args.beam_width,
            "coordinate_rounds": args.coordinate_rounds,
            "max_residues": args.max_residues,
            "max_sidechain_displacement_A": args.max_sidechain_displacement_A,
            "maximum_interaction_drop": args.maximum_interaction_drop,
            "python": args.python,
            "script": str(script),
        })

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(pending), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {result['complex_name']}: {result['status']}",
                flush=True,
            )
    failures = sum(row["status"] not in {"completed", "skipped_complete"} for row in results)
    pd.DataFrame(results).to_csv(args.output_root / "run_status.csv", index=False)
    summary = {
        "method": "CAAHS-A+SC",
        "expected_cases": args.expected_cases,
        "submitted_unresolved_cases": len(tasks),
        "selection_mode": args.selection_mode,
        "chi_offsets": args.chi_offsets,
        "beam_width": args.beam_width,
        "coordinate_rounds": args.coordinate_rounds,
        "max_residues": args.max_residues,
        "max_sidechain_displacement_A": args.max_sidechain_displacement_A,
        "maximum_interaction_drop": args.maximum_interaction_drop,
        "config_json": str(args.config_json.resolve()) if args.config_json else None,
        "config_sha256": config_sha256,
        "failures": failures,
        "uses_reference_or_native_rmsd": False,
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
