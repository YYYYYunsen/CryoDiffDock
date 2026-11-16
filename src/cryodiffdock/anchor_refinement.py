"""Run exit-torsion scanning and density refinement on anchored samples."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def prepare_dihedrals(ligand, indices, specifications):
    from rdkit import Chem

    anchors = set(indices)
    dihedrals = []
    for text in specifications:
        atoms = [int(value) - 1 for value in text.split(",")]
        if len(atoms) != 4 or len(set(atoms)) != 4 or min(atoms) < 0 or max(atoms) >= ligand.GetNumAtoms():
            raise ValueError("Each exit dihedral needs four distinct one-based ligand heavy-atom indices.")
        for a, b in zip(atoms, atoms[1:]):
            if ligand.GetBondBetweenAtoms(a, b) is None:
                raise ValueError("Exit dihedral atoms must be sequentially bonded.")
        bond = ligand.GetBondBetweenAtoms(atoms[1], atoms[2])
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            raise ValueError("The central bond of each exit dihedral must be a non-ring single bond.")
        if not dihedrals and (atoms[1] not in anchors or atoms[2] in anchors):
            raise ValueError("The first central bond must point from the anchored fragment towards the free fragment.")
        cut = Chem.RWMol(ligand)
        cut.RemoveBond(atoms[1], atoms[2])
        moving = next(fragment for fragment in Chem.GetMolFrags(cut.GetMol()) if atoms[2] in fragment)
        if anchors.intersection(moving):
            raise ValueError("An exit dihedral would rotate anchored atoms. Check its direction.")
        dihedrals.append(",".join(map(str, atoms)))
    if dihedrals[0].split(",")[1:3] == dihedrals[1].split(",")[1:3]:
        raise ValueError("Provide two different central bonds for the exit scan.")
    return dihedrals


def refine(output, engine, manifest, config, candidates, scores, dihedrals, tolerance, threads):
    import numpy as np
    from rdkit import Chem

    root = output / "refinement"
    seed_dir = root / "sampling_seeds" / "index0"
    seed_dir.mkdir(parents=True)
    reference = json.loads(config.read_text())
    reference["ligand_indices"] = reference["indices"]
    reference_path = root / "config.json"
    reference_path.write_text(json.dumps(reference, indent=2) + "\n")
    source = Path(candidates).parent
    compatible = [row for row in scores if row["stereochemistry_matches"] == "True"
                  and float(row["free_atoms_in_map_fraction"]) >= 0.99]
    compatible.sort(key=lambda row: (int(row["severe_receptor_clashes"]),
                                     -float(row["free_fragment_ncc"])))
    if not compatible:
        return {"refinement_status": "No stereochemically consistent seeds within the density crop."}, None
    for rank, row in enumerate(compatible[:6], 1):
        shutil.copy2(source / row["file"], seed_dir / f"rank{rank:02d}_confidence0.sdf")

    def run(stage, script, *arguments):
        log_path = root / f"{stage}.log"
        print(f"{stage.replace('_', ' ').capitalize()}. Log: {log_path}", flush=True)
        command = [sys.executable, str(engine / "anchor_refinement" / script),
                   "--config", str(reference_path), *map(str, arguments)]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with log_path.open("w") as log:
            code = subprocess.run(command, cwd=engine, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if code:
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 3000))
                details = log.read().decode(errors="replace")
            raise RuntimeError(f"{stage} failed (exit {code}). See {log_path}.\n{details}")

    repaired = root / "repaired"
    run("initial_repair", "repair_guided.py", "--inference", seed_dir.parent,
        "--out", repaired / "index0", "--iterations", 2000)
    scan = root / "scan"
    run("exit_torsion_scan", "scan_head_exit.py", "--manifest", manifest,
        "--seeds", repaired, "--out", scan, "--dihedral1", dihedrals[0],
        "--dihedral2", dihedrals[1], "--nseeds", 3, "--angle-step", 30,
        "--iterations", 700, "--threads", threads)
    with (scan / "candidates.csv").open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["stereo_matches"] == "True"
                and row["head_gate_pass"] == "True" and float(row["crop_in_bounds_fraction"]) >= 0.99]
    rows.sort(key=lambda row: (int(row["severe_clash_pairs"]), -float(row["density_ncc"])))
    seeds = sorted(repaired.glob("index*/rank*_confidence*.sdf")) + [Path(row["pose"]) for row in rows[:6]]
    seed_manifest = root / "density_seeds.csv"
    with seed_manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pose"])
        writer.writerows([seed] for seed in seeds)
    optimized = root / "density"
    run("local_density_refinement", "optimize_masked_density.py", "--manifest", manifest,
        "--seeds", seed_manifest, "--out", optimized, "--max-seeds", len(seeds),
        "--iterations", 4000, "--weights=0,1,2,4")

    target = Chem.MolFromMolFile(reference["reference_path"]).GetConformer().GetPositions()
    with (optimized / "candidates.csv").open() as handle:
        final = list(csv.DictReader(handle))
    eligible = []
    for row in final:
        ligand = Chem.MolFromMolFile(row["pose"])
        xyz = ligand.GetConformer().GetPositions()[reference["indices"]]
        row["max_anchor_deviation_A"] = float(np.linalg.norm(xyz - target, axis=1).max())
        row["eligible"] = (
            np.isfinite(float(row["density_ncc"])) and row["max_anchor_deviation_A"] <= tolerance
            and row["optimizer_converged"] == "True" and row["stereo_matches"] == "True"
            and int(row["severe_clash_pairs"]) == 0 and float(row["crop_in_bounds_fraction"]) >= 0.99
        )
        if row["eligible"]:
            eligible.append(row)
    eligible.sort(key=lambda row: -float(row["density_ncc"]))
    with (output / "refined_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final[0])
        writer.writeheader()
        writer.writerows(final)
    metrics = {"scanned_seeds": min(3, len(list(repaired.glob("index*/rank*_confidence*.sdf")))),
               "refined_candidates": len(final), "refined_eligible": len(eligible),
               "minimum_refined_severe_clashes": min(int(row["severe_clash_pairs"]) for row in final)}
    return metrics, eligible[0] if eligible else None
