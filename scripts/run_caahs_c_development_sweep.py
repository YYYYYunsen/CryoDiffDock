#!/usr/bin/env python3
"""Run the predefined CAAHS-C constrained-refinement variants on validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


def resolve_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of the required columns is available: {names}")


def run_one(task: dict[str, object]) -> dict[str, object]:
    output = Path(task["output"])
    if (output / "summary.json").is_file():
        return {**task, "status": "skipped_complete", "returncode": 0}
    if output.exists():
        return {**task, "status": "incomplete_existing_output", "returncode": -1}
    variant = task["variant"]
    command = [
        str(task["python"]), str(task["script"]),
        "--ligand", str(task["ligand"]),
        "--protein", str(task["protein"]),
        "--density-crop", str(task["density_crop"]),
        "--output-dir", str(output),
        "--mode", str(variant["mode"]),
        "--maximum-ligand-displacement-A", str(variant["maximum_ligand_displacement_A"]),
        "--target-relative-vdw-distance", str(variant["target_relative_vdw_distance"]),
        "--optimization-relative-vdw-distance", str(
            variant["optimization_relative_vdw_distance"]
        ),
        "--acceptance-relative-vdw-distance", str(variant["acceptance_relative_vdw_distance"]),
        "--maximum-ncc-drop", str(variant["maximum_ncc_drop"]),
        "--maximum-interaction-drop", str(variant["maximum_interaction_drop"]),
        "--severe-overlap-A", str(variant["severe_overlap_A"]),
        "--rotation-bound-degrees", str(variant["rotation_bound_degrees"]),
        "--maximum-iterations", str(variant["maximum_iterations"]),
        "--number-of-starts", str(variant["number_of_starts"]),
        "--joint-top-k", str(variant["joint_top_k"]),
        "--sidechain-beam-width", str(variant["sidechain_beam_width"]),
        "--sidechain-coordinate-rounds", str(variant["sidechain_coordinate_rounds"]),
        "--maximum-target-residues", str(variant["maximum_target_residues"]),
        "--maximum-sidechain-displacement-A", str(variant["maximum_sidechain_displacement_A"]),
    ]
    log = Path(task["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                 text=True, check=False)
    ok = process.returncode == 0 and (output / "summary.json").is_file()
    return {**task, "status": "completed" if ok else "failed",
            "returncode": int(process.returncode)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cases", required=True, type=Path)
    parser.add_argument("--repair-manifest", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant", action="append", dest="selected_variants")
    parser.add_argument("--complex-name", action="append", dest="selected_complexes")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config_bytes = args.config_json.read_bytes()
    config = json.loads(config_bytes)
    status = config.get("status")
    if config.get("method") != "CAAHS-C" or status not in {
        "development_sweep", "development_frozen"
    }:
        raise ValueError(
            "Expected a CAAHS-C development_sweep or development_frozen configuration"
        )
    variants = (
        config["variants"]
        if status == "development_sweep"
        else [config["frozen_configuration"]]
    )
    if args.selected_variants:
        requested = set(args.selected_variants)
        variants = [item for item in variants if item["name"] in requested]
        found = {item["name"] for item in variants}
        if found != requested:
            raise ValueError(f"Unknown variants: {sorted(requested - found)}")
    cases = pd.read_csv(args.baseline_cases)
    manifest = pd.read_csv(args.repair_manifest)
    if len(cases) != args.expected_cases or cases.complex_name.duplicated().any():
        raise ValueError(f"Expected {args.expected_cases} unique baseline cases")
    density_column = resolve_column(
        manifest, ("density_crop", "ligand_density_crop", "density_path")
    )
    pose_column = resolve_column(
        cases, ("selected_pose_path_final", "selected_pose_path")
    )
    selected_protein_column = next(
        (
            name for name in (
                "selected_protein_path_final",
                "selected_protein_path",
            )
            if name in cases.columns
        ),
        None,
    )
    required = manifest[["complex_name", "protein_path", density_column]].copy()
    required = required.rename(columns={density_column: "resolved_density_crop"})
    required = required.drop_duplicates("complex_name")
    table = cases.merge(required, on="complex_name", how="left", validate="one_to_one")
    if table[["protein_path", "resolved_density_crop"]].isna().any().any():
        raise ValueError("Missing protein or density path after manifest merge")
    if args.selected_complexes:
        requested_complexes = set(args.selected_complexes)
        table = table[table["complex_name"].astype(str).isin(requested_complexes)].copy()
        found_complexes = set(table["complex_name"].astype(str))
        if found_complexes != requested_complexes:
            raise ValueError(
                f"Unknown complexes: {sorted(requested_complexes - found_complexes)}"
            )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        table = table.head(args.limit).copy()

    script = Path(__file__).resolve().with_name("refine_caahs_c_joint.py")
    tasks = []
    for variant in variants:
        for row in table.itertuples(index=False):
            selected_protein = (
                getattr(row, selected_protein_column)
                if selected_protein_column is not None else None
            )
            protein = (
                Path(str(selected_protein))
                if selected_protein is not None and not pd.isna(selected_protein)
                else Path(str(row.protein_path))
            )
            name = str(row.complex_name)
            variant_name = str(variant["name"])
            tasks.append({
                "complex_name": name,
                "variant_name": variant_name,
                "ligand": str(Path(str(getattr(row, pose_column))).resolve()),
                "protein": str(protein.resolve()),
                "density_crop": str(Path(str(row.resolved_density_crop)).resolve()),
                "output": str(args.output_root / variant_name / "cases" / name),
                "log": str(args.output_root / variant_name / "logs" / f"{name}.log"),
                "variant": variant,
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
            print(f"[{index}/{len(tasks)}] {result['variant_name']}/{result['complex_name']}: "
                  f"{result['status']}", flush=True)
    failures = sum(x["status"] not in {"completed", "skipped_complete"} for x in results)
    pd.DataFrame(results).to_csv(args.output_root / "run_status.csv", index=False)
    summary = {
        "method": "CAAHS-C",
        "status": (
            "frozen_validation_run_complete"
            if status == "development_frozen"
            else "development_sweep_complete"
        ),
        "expected_cases": args.expected_cases,
        "processed_cases": len(table),
        "variants": [item["name"] for item in variants],
        "tasks": len(tasks),
        "failures": failures,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "uses_reference_or_native_rmsd": False,
    }
    (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
