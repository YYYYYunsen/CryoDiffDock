#!/usr/bin/env python3
"""Apply the frozen CAAHS-A fallback to standalone (non-cascade) pose pools.

The input baseline manifest is deliberately blind: it contains only case order,
complex identity, the frozen Top-1 pose, its protein, and the density crop.  No
reference ligand or native RMSD is read or used by this selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.adaptive_search import MultiScaleNCCEvaluator, ProteinClashIndex
from cryodiffdock_pocket.density_ranking import load_cartesian_density_crop
from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd
from scripts.run_adaptive_density_search import molecule_coordinates, read_molecule
from scripts.select_caahs_conservative_fallback import (
    choose_device,
    clash_score,
    sha256,
)


REQUIRED_COLUMNS = {
    "case_order",
    "complex_name",
    "baseline_pose_path",
    "protein_path",
    "density_crop",
}
FORBIDDEN_TOKENS = ("reference", "native", "rmsd", "outcome", "success")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--caahs-root", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cases", required=True, type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    started = time.perf_counter()
    for path in (args.baseline_manifest, args.config_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.caahs_root.is_dir():
        raise FileNotFoundError(args.caahs_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {args.output_dir}")

    config = json.loads(args.config_json.read_text())
    frozen = config.get("conservative_fallback", {})
    if frozen.get("status") != "development_frozen":
        raise ValueError("CAAHS conservative fallback is not development-frozen")
    if frozen.get("require_old_clashing") is not True:
        raise ValueError("Frozen selector must require a clashing baseline")
    if frozen.get("require_new_clash_free") is not True:
        raise ValueError("Frozen selector must require a clash-free CAAHS pose")
    if frozen.get("accept_partial_clash_reduction") is not False:
        raise ValueError("Frozen selector must reject partial clash reductions")
    if frozen.get("uses_reference_or_rmsd_to_native_for_selection") is not False:
        raise ValueError("Frozen selector must prohibit native/reference selection")

    maximum_displacement = float(frozen["maximum_old_to_new_displacement_A"])
    maximum_ncc_drop = float(frozen["maximum_ncc_drop"])
    severe_overlap = float(config["severe_overlap_A"])
    device = choose_device(args.device)

    # Inspect the header before selecting columns, so accidental reveal columns
    # make the blind seal fail loudly rather than merely being ignored.
    header = pd.read_csv(args.baseline_manifest, nrows=0).columns.tolist()
    forbidden = [
        column for column in header
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Blind baseline manifest contains forbidden columns: {forbidden}")
    missing = sorted(REQUIRED_COLUMNS - set(header))
    if missing:
        raise ValueError(f"Baseline manifest lacks columns: {missing}")
    baseline = pd.read_csv(args.baseline_manifest, usecols=sorted(REQUIRED_COLUMNS))
    if len(baseline) != args.expected_cases:
        raise ValueError(f"Expected {args.expected_cases} cases, found {len(baseline)}")
    baseline["complex_name"] = baseline["complex_name"].astype(str).str.upper()
    baseline["case_order"] = pd.to_numeric(baseline["case_order"], errors="raise").astype(int)
    baseline = baseline.sort_values("case_order", kind="stable").reset_index(drop=True)
    if baseline["complex_name"].duplicated().any():
        raise ValueError("Baseline manifest has duplicate complex_name values")
    if baseline["case_order"].duplicated().any():
        raise ValueError("Baseline manifest has duplicate case_order values")

    run_manifest_path = args.caahs_root / "evaluation_manifest.csv"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = pd.read_csv(run_manifest_path, usecols=["complex_name"])
    run_names = run_manifest["complex_name"].astype(str).str.upper().tolist()
    baseline_names = baseline["complex_name"].tolist()
    if run_names != baseline_names:
        raise ValueError(
            "CAAHS evaluation_manifest order/identities do not exactly match "
            "the blind baseline manifest"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_root = args.output_dir / "selected_results"
    protein_cache: dict[str, ProteinClashIndex] = {}
    rows: list[dict] = []

    for row_index, row in baseline.iterrows():
        name = str(row["complex_name"])
        old_pose = Path(str(row["baseline_pose_path"])).expanduser().resolve()
        old_protein = Path(str(row["protein_path"])).expanduser().resolve()
        density_crop = Path(str(row["density_crop"])).expanduser().resolve()
        for path in (old_pose, old_protein, density_crop):
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path}")

        ranking_path = (
            args.caahs_root / "v2_selected" / f"index{row_index}___{name}" / "ranking.csv"
        )
        if not ranking_path.is_file():
            matches = list((args.caahs_root / "v2_selected").glob(f"index*___{name}/ranking.csv"))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{name}: expected one CAAHS ranking, found {len(matches)}"
                )
            ranking_path = matches[0]
        ranking = pd.read_csv(ranking_path)
        if len(ranking) != 1:
            raise ValueError(f"{name}: expected exactly one materialized CAAHS Top-1")
        new_pose = Path(str(ranking.iloc[0]["materialized_path"])).expanduser().resolve()
        new_protein = Path(str(ranking.iloc[0]["protein_path"])).expanduser().resolve()

        old_count, old_fraction = clash_score(
            old_pose, old_protein, protein_cache, severe_overlap
        )
        new_count, new_fraction = clash_score(
            new_pose, new_protein, protein_cache, severe_overlap
        )
        displacement = float(
            symmetry_rmsd(read_heavy_molecule(old_pose), read_heavy_molecule(new_pose))
        )

        old_molecule = read_molecule(old_pose)
        new_molecule = read_molecule(new_pose)
        old_atomic_numbers = [atom.GetAtomicNum() for atom in old_molecule.GetAtoms()]
        new_atomic_numbers = [atom.GetAtomicNum() for atom in new_molecule.GetAtoms()]
        if old_atomic_numbers != new_atomic_numbers:
            raise ValueError(f"{name}: old/new ligand atom ordering differs")
        evaluator = MultiScaleNCCEvaluator(
            load_cartesian_density_crop(density_crop),
            old_atomic_numbers,
            sigmas_A=tuple(float(value) for value in config.get("density_sigmas_A", (1.5, 2.25))),
            mask_radius_A=float(config.get("mask_radius_A", 10.0)),
            min_mask_voxels=int(config.get("minimum_mask_voxels", 64)),
            device=device,
            voxel_chunk_size=512,
            pose_batch_size=2,
        )
        ncc = evaluator.score([
            molecule_coordinates(old_molecule),
            molecule_coordinates(new_molecule),
        ])
        old_ncc = float(ncc[0].mean())
        new_ncc = float(ncc[1].mean())
        delta_ncc = new_ncc - old_ncc

        accepted = bool(
            old_count > 0
            and new_count == 0
            and displacement <= maximum_displacement
            and delta_ncc >= -maximum_ncc_drop
        )
        if accepted:
            reason = "accepted_complete_clash_rescue_within_frozen_gates"
        elif old_count == 0:
            reason = "retain_old_already_clash_free"
        elif new_count != 0:
            reason = "retain_old_new_not_clash_free"
        elif displacement > maximum_displacement:
            reason = "retain_old_displacement_gate"
        else:
            reason = "retain_old_ncc_gate"

        final_pose = new_pose if accepted else old_pose
        final_protein = new_protein if accepted else old_protein
        final_count = new_count if accepted else old_count
        final_fraction = new_fraction if accepted else old_fraction
        destination = selected_root / name
        destination.mkdir(parents=True, exist_ok=True)
        selected_pose = destination / "rank1_confidence0.00000000.sdf"
        shutil.copy2(final_pose, selected_pose)
        shutil.copy2(final_pose, destination / "rank1.sdf")
        shutil.copy2(final_protein, destination / "rank1_protein.pdb")

        rows.append({
            "case_order": int(row["case_order"]),
            "complex_name": name,
            "caahs_attempted": True,
            "caahs_accepted": accepted,
            "selection_reason": reason,
            "old_pose_path": str(old_pose),
            "caahs_pose_path": str(new_pose),
            "selected_pose_path": str(selected_pose.resolve()),
            "selected_protein_path": str((destination / "rank1_protein.pdb").resolve()),
            "density_crop": str(density_crop),
            "old_severe_clash_count": old_count,
            "caahs_severe_clash_count": new_count,
            "selected_severe_clash_count": final_count,
            "old_clashing_atom_fraction": old_fraction,
            "caahs_clashing_atom_fraction": new_fraction,
            "selected_clashing_atom_fraction": final_fraction,
            "old_to_new_displacement_A": displacement,
            "old_multiscale_ncc": old_ncc,
            "caahs_multiscale_ncc": new_ncc,
            "delta_ncc": delta_ncc,
        })

    table = pd.DataFrame(rows)
    selection_path = args.output_dir / "selected_poses_blind.csv"
    table.to_csv(selection_path, index=False)
    summary = {
        "status": "selection_sealed_before_outcome_evaluation",
        "method": "CAAHS-A standalone conservative fallback",
        "cases_n": len(table),
        "caahs_attempted_n": len(table),
        "caahs_accepted_n": int(table["caahs_accepted"].sum()),
        "old_clash_free_n": int(table["old_severe_clash_count"].eq(0).sum()),
        "selected_clash_free_n": int(table["selected_severe_clash_count"].eq(0).sum()),
        "maximum_old_to_new_displacement_A": maximum_displacement,
        "maximum_ncc_drop": maximum_ncc_drop,
        "uses_reference_or_native_rmsd_for_selection": False,
        "selection_sha256": sha256(selection_path),
        "config_sha256": sha256(args.config_json),
        "input_hashes": {
            str(args.baseline_manifest.resolve()): sha256(args.baseline_manifest),
            str(run_manifest_path.resolve()): sha256(run_manifest_path),
        },
        "caahs_summary_sha256": sha256(args.caahs_root / "summary.json"),
        "runtime_seconds": time.perf_counter() - started,
        "hardware": {"platform": platform.platform(), "device": str(device)},
    }
    (args.output_dir / "selection_seal.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print(f"CAAHS_STANDALONE_SELECTION_SEALED: {selection_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
