#!/usr/bin/env python3
"""Run a reproducible clash-repair configuration sweep on a validation cohort."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


COLUMN_ALIASES = {
    "complex_name": ("complex_name", "name", "pdb_id", "complex_id"),
    "ligand": (
        "selected_pose_path", "prediction_path", "ligand_path", "ligand",
        "medoid_pose_path",
    ),
    "protein": (
        "paired_flexible_protein_path", "experimental_protein",
        "protein_path", "protein",
    ),
    "density_crop": (
        "ligand_density_crop", "density_crop", "density_crop_path", "crop_path",
    ),
}


def resolve_column(frame: pd.DataFrame, role: str, explicit: str | None) -> str:
    if explicit:
        if explicit not in frame:
            raise ValueError(f"Requested {role} column is absent: {explicit}")
        return explicit
    matches = [column for column in COLUMN_ALIASES[role] if column in frame]
    if not matches:
        raise ValueError(
            f"Could not infer {role} column; tried {COLUMN_ALIASES[role]}. "
            f"Available columns: {list(frame.columns)}"
        )
    return matches[0]


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def config_arguments(parameters: dict[str, object]) -> list[str]:
    result: list[str] = []
    for key, value in parameters.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            result.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            result.extend([flag, ",".join(str(item) for item in value)])
        else:
            result.extend([flag, str(value)])
    return result


def run_one(task: dict[str, object]) -> dict[str, object]:
    case_root = Path(task["case_root"])
    summary_path = case_root / "summary.json"
    pose_path = case_root / "selected_pose.sdf"
    if summary_path.is_file() and pose_path.is_file():
        return {**task, "status": "skipped_complete", "returncode": 0, "error": ""}
    if case_root.exists():
        return {
            **task, "status": "incomplete_existing_output", "returncode": -1,
            "error": str(case_root),
        }
    case_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(task["python"]), str(task["repair_script"]),
        "--ligand", str(task["ligand"]),
        "--protein", str(task["protein"]),
        "--density-crop", str(task["density_crop"]),
        "--output-dir", str(case_root),
        *task["config_arguments"],
    ]
    log_path = Path(task["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w") as handle:
            process = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        ok = process.returncode == 0 and summary_path.is_file() and pose_path.is_file()
        return {
            **task, "status": "completed" if ok else "failed",
            "returncode": int(process.returncode),
            "error": "" if ok else f"See {log_path}",
        }
    except Exception as exc:
        return {
            **task, "status": "failed", "returncode": -1,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=62)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--complex-column")
    parser.add_argument("--ligand-column")
    parser.add_argument("--protein-column")
    parser.add_argument("--density-column")
    args = parser.parse_args()
    if args.workers < 1 or args.task_count < 1 or not 0 <= args.task_index < args.task_count:
        raise ValueError("Invalid workers/task-count/task-index")

    project_root = Path(__file__).resolve().parents[1]
    repair_script = project_root / "scripts" / "repair_pose_density_clash_v2.py"
    manifest_path = args.manifest.expanduser().resolve()
    config_path = args.config_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    table = pd.read_csv(manifest_path, dtype=str)
    columns = {
        "complex_name": resolve_column(table, "complex_name", args.complex_column),
        "ligand": resolve_column(table, "ligand", args.ligand_column),
        "protein": resolve_column(table, "protein", args.protein_column),
        "density_crop": resolve_column(table, "density_crop", args.density_column),
    }
    if len(table) != args.expected_cases or table[columns["complex_name"]].nunique() != len(table):
        raise ValueError(
            f"Expected {args.expected_cases} unique cases, found {len(table)} rows and "
            f"{table[columns['complex_name']].nunique()} unique names"
        )
    configuration = json.loads(config_path.read_text())
    configs = configuration.get("configs", configuration)
    if not isinstance(configs, dict) or not configs:
        raise ValueError("Config JSON must contain a non-empty 'configs' object")

    tasks = []
    for row_index, row in table.iterrows():
        if row_index % args.task_count != args.task_index:
            continue
        name = str(row[columns["complex_name"]])
        paths = {
            role: resolve_path(row[column], manifest_path.parent)
            for role, column in columns.items() if role != "complex_name"
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: missing inputs: {missing}")
        for config_name, parameters in configs.items():
            if not isinstance(parameters, dict):
                raise ValueError(f"Config {config_name} must be an object")
            case_root = output_root / str(config_name) / "cases" / name
            tasks.append({
                "row_index": int(row_index), "complex_name": name,
                "config": str(config_name), "case_root": str(case_root),
                "ligand": str(paths["ligand"]), "protein": str(paths["protein"]),
                "density_crop": str(paths["density_crop"]),
                "config_arguments": config_arguments(parameters),
                "repair_script": str(repair_script), "python": args.python,
                "log_path": str(output_root / str(config_name) / "logs" / f"{name}.log"),
            })

    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = {executor.submit(run_one, task): task for task in tasks}
        for completed, future in enumerate(as_completed(pending), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{completed}/{len(tasks)}] {result['config']}/{result['complex_name']}: "
                f"{result['status']}", flush=True,
            )
    keep = [
        "row_index", "complex_name", "config", "status", "returncode", "error",
        "case_root", "ligand", "protein", "density_crop", "log_path",
    ]
    status_path = output_root / f"run_status_task{args.task_index:02d}.csv"
    pd.DataFrame(results)[keep].sort_values(
        ["config", "row_index"], kind="stable"
    ).to_csv(status_path, index=False)
    failures = sum(result["status"] not in {"completed", "skipped_complete"}
                   for result in results)
    print(f"VALIDATION_REPAIR_SWEEP_DONE tasks={len(tasks)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
