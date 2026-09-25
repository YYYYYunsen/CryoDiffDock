#!/usr/bin/env python3
"""Run restrained CAAHS-C ligand internal-geometry variants on validation."""

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
    output = Path(str(task["output"]))
    if (output / "summary.json").is_file():
        return {**task, "status": "skipped_complete", "returncode": 0}
    if output.exists():
        return {**task, "status": "incomplete_existing_output", "returncode": -1}
    variant = task["variant"]
    shared = task["shared"]
    command = [
        str(task["python"]), str(task["script"]),
        "--ligand", str(task["ligand"]),
        "--protein", str(task["protein"]),
        "--density-crop", str(task["density_crop"]),
        "--output-dir", str(output),
        "--maximum-atom-displacements-A",
        str(variant["maximum_atom_displacements_A"]),
        "--force-constants", str(variant["force_constants"]),
        "--maximum-iterations", str(variant["maximum_iterations"]),
        "--maximum-ligand-rmsd-A", str(variant["maximum_ligand_rmsd_A"]),
        "--minimum-relative-vdw-distance",
        str(shared["minimum_relative_vdw_distance"]),
        "--maximum-ncc-drop", str(shared["maximum_ncc_drop"]),
        "--maximum-interaction-drop", str(shared["maximum_interaction_drop"]),
        "--severe-overlap-A", str(shared["severe_overlap_A"]),
    ]
    log = Path(str(task["log"]))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        process = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False
        )
    complete = process.returncode == 0 and (output / "summary.json").is_file()
    return {
        **task,
        "status": "completed" if complete else "failed",
        "returncode": int(process.returncode),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-cases", required=True, type=Path)
    parser.add_argument(
        "--baseline-variant",
        help=(
            "Optional value from the baseline table's variant/config column. "
            "Required when the table contains more than one baseline variant."
        ),
    )
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
    if config.get("method") != "CAAHS-C restrained ligand internal-geometry refinement":
        raise ValueError("Unexpected configuration method")
    status = config.get("status")
    if status not in {"development_sweep", "development_frozen"}:
        raise ValueError("Configuration must be development_sweep or development_frozen")
    variants = (
        config["variants"] if status == "development_sweep"
        else [config["frozen_configuration"]]
    )
    if args.selected_variants:
        requested = set(args.selected_variants)
        variants = [item for item in variants if item["name"] in requested]
        missing = requested - {item["name"] for item in variants}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    cases = pd.read_csv(args.baseline_cases)
    manifest = pd.read_csv(args.repair_manifest)
    baseline_variant_column = next(
        (name for name in ("variant", "config") if name in cases.columns),
        None,
    )
    if args.baseline_variant is not None:
        if baseline_variant_column is None:
            raise ValueError(
                "--baseline-variant was supplied but the baseline table has no "
                "variant or config column"
            )
        available = set(cases[baseline_variant_column].astype(str))
        if args.baseline_variant not in available:
            raise ValueError(
                f"Unknown baseline variant {args.baseline_variant!r}; "
                f"available values: {sorted(available)}"
            )
        cases = cases[
            cases[baseline_variant_column].astype(str).eq(args.baseline_variant)
        ].copy()
    elif baseline_variant_column is not None and len(cases) != args.expected_cases:
        available = sorted(set(cases[baseline_variant_column].astype(str)))
        raise ValueError(
            "Baseline table contains multiple variants; pass --baseline-variant. "
            f"Available values: {available}"
        )
    if len(cases) != args.expected_cases or cases.complex_name.duplicated().any():
        raise ValueError(
            f"Expected {args.expected_cases} unique baseline cases after filtering; "
            f"found {len(cases)} rows and "
            f"{cases.complex_name.nunique()} unique complexes"
        )
    pose_column = resolve_column(
        cases, ("selected_pose_path", "selected_pose_path_final")
    )
    protein_column = resolve_column(
        cases, ("selected_protein_path", "selected_protein_path_final", "protein_path")
    )
    density_column = resolve_column(
        manifest, ("density_crop", "ligand_density_crop", "density_path")
    )
    density = manifest[["complex_name", density_column]].drop_duplicates("complex_name")
    density = density.rename(columns={density_column: "resolved_density_crop"})
    table = cases.merge(density, on="complex_name", how="left", validate="one_to_one")
    if table["resolved_density_crop"].isna().any():
        raise ValueError("Missing density crop after manifest merge")
    if args.selected_complexes:
        requested = set(args.selected_complexes)
        table = table[table.complex_name.astype(str).isin(requested)].copy()
        missing = requested - set(table.complex_name.astype(str))
        if missing:
            raise ValueError(f"Unknown complexes: {sorted(missing)}")
    if args.limit is not None:
        table = table.head(args.limit).copy()

    script = Path(__file__).resolve().with_name(
        "refine_caahs_c_internal_geometry.py"
    )
    shared = config["shared_hard_gates"]
    tasks = []
    for variant in variants:
        for row in table.itertuples(index=False):
            name = str(row.complex_name)
            tasks.append({
                "complex_name": name,
                "variant_name": str(variant["name"]),
                "ligand": str(Path(str(getattr(row, pose_column))).resolve()),
                "protein": str(Path(str(getattr(row, protein_column))).resolve()),
                "density_crop": str(Path(str(row.resolved_density_crop)).resolve()),
                "output": str(
                    args.output_root / str(variant["name"]) / "cases" / name
                ),
                "log": str(
                    args.output_root / str(variant["name"]) / "logs" / f"{name}.log"
                ),
                "variant": variant,
                "shared": shared,
                "python": args.python,
                "script": str(script),
            })

    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {result['variant_name']}/"
                f"{result['complex_name']}: {result['status']}",
                flush=True,
            )
    failures = sum(
        item["status"] not in {"completed", "skipped_complete"}
        for item in results
    )
    pd.DataFrame(results).to_csv(args.output_root / "run_status.csv", index=False)
    summary = {
        "method": config["method"],
        "status": (
            "frozen_validation_run_complete"
            if status == "development_frozen" else "development_sweep_complete"
        ),
        "expected_cases": args.expected_cases,
        "processed_cases": len(table),
        "baseline_variant": args.baseline_variant,
        "variants": [item["name"] for item in variants],
        "tasks": len(tasks),
        "failures": failures,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "uses_reference_or_native_rmsd": False,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
