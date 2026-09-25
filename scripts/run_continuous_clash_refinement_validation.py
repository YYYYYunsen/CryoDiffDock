#!/usr/bin/env python3
"""Run V3 continuous refinement only for unresolved V2 validation cases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def selected_case_ids(path: Path, split: str | None) -> set[str]:
    table = pd.read_csv(path)
    id_column = next(
        (column for column in ("complex_name", "pdb_id", "complex_id") if column in table),
        None,
    )
    if id_column is None:
        raise ValueError(f"No case ID column in {path}")
    if split is not None:
        if "split" not in table:
            raise ValueError(f"--split requires a split column in {path}")
        table = table[table["split"].astype(str).eq(split)]
    values = table[id_column].dropna().astype(str)
    if values.duplicated().any():
        raise ValueError(f"Duplicate case IDs in {path}")
    return set(values)


def run_case(task: dict[str, object]) -> dict[str, object]:
    output = Path(task["output_dir"])
    summary = output / "summary.json"
    pose = output / "selected_pose.sdf"
    if summary.is_file() and pose.is_file():
        return {**task, "status": "skipped_complete", "returncode": 0, "error": ""}
    if output.exists():
        return {**task, "status": "incomplete_existing_output", "returncode": -1,
                "error": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(task["python"]), str(task["script"]),
        "--v2-case-dir", str(task["v2_case_dir"]),
        "--output-dir", str(output),
        "--num-starts", str(task["num_starts"]),
        "--max-iterations", str(task["max_iterations"]),
        "--max-torsion-bonds", str(task["max_torsion_bonds"]),
    ]
    if task["refinement_mode"] in {"v3_continuous", "v6_original_restart"}:
        command.extend(["--optimization-mode", str(task["optimization_mode"])])
        if task["max_displacement_A"] is not None:
            command.extend([
                "--max-displacement-A", str(task["max_displacement_A"]),
            ])
        if task["max_ncc_drop"] is not None:
            command.extend(["--max-ncc-drop", str(task["max_ncc_drop"])])
        if task["refinement_mode"] == "v6_original_restart":
            command.append("--restart-from-original")
    else:
        command.extend([
            "--max-rounds", str(task["max_rounds"]),
            "--local-topological-radius", str(task["local_topological_radius"]),
        ])
    log = Path(task["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as handle:
            process = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        ok = process.returncode == 0 and summary.is_file() and pose.is_file()
        return {**task, "status": "completed" if ok else "failed",
                "returncode": int(process.returncode),
                "error": "" if ok else f"See {log}"}
    except Exception as exc:
        return {**task, "status": "failed", "returncode": -1,
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-case-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument(
        "--case-manifest", type=Path,
        help="Optionally restrict the run to case IDs in this CSV",
    )
    parser.add_argument(
        "--split", help="With --case-manifest, retain only this split value",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--num-starts", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-torsion-bonds", type=int, default=8)
    parser.add_argument(
        "--refinement-mode",
        choices=["v3_continuous", "v5_iterative_local", "v6_original_restart"],
        default="v3_continuous",
    )
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--local-topological-radius", type=int, default=3)
    parser.add_argument(
        "--max-displacement-A", type=float,
        help="Override the per-case displacement gate for V3/V6 development sweeps",
    )
    parser.add_argument(
        "--max-ncc-drop", type=float,
        help="Override the per-case NCC-drop gate for V3/V6 development sweeps",
    )
    parser.add_argument("--optimization-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument(
        "--max-v2-residual-clashes", type=int, default=None,
        help="Only refine unresolved cases at or below this V2 clash count",
    )
    parser.add_argument(
        "--min-v2-residual-clashes", type=int, default=1,
        help="Only refine unresolved cases at or above this V2 clash count",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.workers < 1 or args.min_v2_residual_clashes < 1:
        raise ValueError("workers and minimum residual clash count must be positive")
    if (args.max_v2_residual_clashes is not None
            and args.max_v2_residual_clashes < args.min_v2_residual_clashes):
        raise ValueError("maximum residual clash count is below the minimum")

    project_root = Path(__file__).resolve().parents[1]
    script_name = (
        "refine_density_clash_iterative_local.py"
        if args.refinement_mode == "v5_iterative_local"
        else "refine_density_clash_continuous.py"
    )
    script = project_root / "scripts" / script_name
    v2_root = args.v2_case_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    case_dirs = sorted(path.parent for path in v2_root.glob("*/summary.json"))
    if args.split and not args.case_manifest:
        raise ValueError("--split requires --case-manifest")
    if args.case_manifest:
        requested = selected_case_ids(args.case_manifest.expanduser().resolve(), args.split)
        available = {path.name for path in case_dirs}
        if missing := sorted(requested - available):
            raise ValueError(f"Manifest cases absent from V2 root: {missing[:20]}")
        case_dirs = [path for path in case_dirs if path.name in requested]
    if len(case_dirs) != args.expected_cases:
        raise RuntimeError(
            f"Expected {args.expected_cases} V2 cases, found {len(case_dirs)}"
        )
    tasks = []
    already_resolved = []
    skipped_above_residual_limit = []
    skipped_below_residual_limit = []
    for case_dir in case_dirs:
        summary = json.loads((case_dir / "summary.json").read_text())
        residual_clashes = int(summary["selected"]["severe_clash_count"])
        if residual_clashes == 0:
            already_resolved.append(case_dir.name)
            continue
        if residual_clashes < args.min_v2_residual_clashes:
            skipped_below_residual_limit.append(case_dir.name)
            continue
        if (args.max_v2_residual_clashes is not None
                and residual_clashes > args.max_v2_residual_clashes):
            skipped_above_residual_limit.append(case_dir.name)
            continue
        tasks.append({
            "complex_name": case_dir.name,
            "v2_case_dir": str(case_dir),
            "output_dir": str(output_root / "cases" / case_dir.name),
            "log": str(output_root / "logs" / f"{case_dir.name}.log"),
            "num_starts": args.num_starts,
            "max_iterations": args.max_iterations,
            "max_torsion_bonds": args.max_torsion_bonds,
            "optimization_mode": args.optimization_mode,
            "refinement_mode": args.refinement_mode,
            "max_rounds": args.max_rounds,
            "local_topological_radius": args.local_topological_radius,
            "max_displacement_A": args.max_displacement_A,
            "max_ncc_drop": args.max_ncc_drop,
            "python": args.python,
            "script": str(script),
        })
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {result['complex_name']}: {result['status']}",
                flush=True,
            )
    columns = [
        "complex_name", "status", "returncode", "error", "v2_case_dir",
        "output_dir", "log",
    ]
    pd.DataFrame(results, columns=columns).to_csv(
        output_root / "run_status.csv", index=False
    )
    (output_root / "already_resolved_v2.txt").write_text(
        "\n".join(already_resolved) + ("\n" if already_resolved else "")
    )
    (output_root / "skipped_above_residual_limit.txt").write_text(
        "\n".join(skipped_above_residual_limit)
        + ("\n" if skipped_above_residual_limit else "")
    )
    (output_root / "skipped_below_residual_limit.txt").write_text(
        "\n".join(skipped_below_residual_limit)
        + ("\n" if skipped_below_residual_limit else "")
    )
    failures = sum(result["status"] not in {"completed", "skipped_complete"}
                   for result in results)
    run_summary = {
        "expected_v2_cases": args.expected_cases,
        "case_manifest": str(args.case_manifest.expanduser().resolve())
        if args.case_manifest else None,
        "split": args.split,
        "already_resolved_v2_n": len(already_resolved),
        "submitted_unresolved_n": len(tasks),
        "skipped_above_residual_limit_n": len(skipped_above_residual_limit),
        "skipped_below_residual_limit_n": len(skipped_below_residual_limit),
        "min_v2_residual_clashes": args.min_v2_residual_clashes,
        "max_v2_residual_clashes": args.max_v2_residual_clashes,
        "failures_n": failures,
        "num_starts": args.num_starts,
        "max_iterations": args.max_iterations,
        "max_torsion_bonds": args.max_torsion_bonds,
        "optimization_mode": args.optimization_mode,
        "refinement_mode": args.refinement_mode,
        "max_rounds": args.max_rounds if args.refinement_mode == "v5_iterative_local" else None,
        "local_topological_radius": (
            args.local_topological_radius
            if args.refinement_mode == "v5_iterative_local" else None
        ),
        "max_displacement_A": args.max_displacement_A,
        "max_ncc_drop": args.max_ncc_drop,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n"
    )
    print(json.dumps(run_summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
