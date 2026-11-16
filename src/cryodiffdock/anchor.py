"""Input validation and reporting for fragment-anchored Local docking."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def validate_mapping(ligand, reference, indices):
    import numpy as np

    if ligand is None or reference is None or reference.GetNumConformers() != 1:
        raise ValueError("Provide a readable full-ligand SDF and a placed 3D reference fragment SDF.")
    if len(indices) != reference.GetNumAtoms() or len(set(indices)) != len(indices):
        raise ValueError("Map each heavy atom in the reference SDF to one unique ligand heavy atom.")
    if not 3 <= len(indices) < ligand.GetNumAtoms() or min(indices) < 0 or max(indices) >= ligand.GetNumAtoms():
        raise ValueError("Select at least three anchor atoms and leave at least one ligand atom free.")
    if any(reference.GetAtomWithIdx(i).GetAtomicNum() != ligand.GetAtomWithIdx(j).GetAtomicNum()
           for i, j in enumerate(indices)):
        raise ValueError("Mapped reference and ligand atoms have different elements.")
    for i, first in enumerate(indices):
        for j in range(i):
            reference_bond = reference.GetBondBetweenAtoms(i, j)
            ligand_bond = ligand.GetBondBetweenAtoms(first, indices[j])
            if ((reference_bond is None) != (ligand_bond is None)
                    or reference_bond is not None and reference_bond.GetBondType() != ligand_bond.GetBondType()):
                raise ValueError("Mapped reference and ligand atoms have different bonds or bond orders.")
    xyz = reference.GetConformer().GetPositions()
    if not np.isfinite(xyz).all() or np.linalg.matrix_rank(xyz - xyz.mean(axis=0)) < 2:
        raise ValueError("The placed anchor needs at least three non-collinear atoms with finite coordinates.")
    return xyz


def add_anchor_parser(subparsers):
    parser = subparsers.add_parser("anchor", help="sample the free part of a ligand around a placed fragment")
    for name, help_text in (
        ("protein", "receptor PDB"),
        ("ligand", "full ligand SDF"),
        ("reference", "placed anchor fragment SDF in the receptor/map frame"),
        ("map", "segmented/residual MRC or prepared Cartesian NPZ density crop"),
        ("output", "new or empty output directory"),
    ):
        parser.add_argument(f"--{name}", type=Path, required=True, help=help_text)
    parser.add_argument("--clash-protein", type=Path,
                        help="full receptor PDB for clash checks when --protein is cropped")
    parser.add_argument("--anchor-atoms", nargs="+", type=int, required=True,
                        metavar="INDEX", help="one-based ligand heavy-atom indices in reference SDF order")
    parser.add_argument("--anchor-tolerance", type=float, default=0.5, metavar="ANGSTROM",
                        help="maximum allowed deviation of any anchored atom (default 0.5 Å)")
    parser.add_argument("--box-size", type=float, default=36.0,
                        help="map crop width in Å for MRC input (default 36)")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--exit-dihedrals", nargs=2, metavar=("EXIT", "NEXT"),
                        help="two comma-separated dihedrals in one-based ligand heavy-atom order")
    parser.add_argument("--refinement-threads", type=int, default=1,
                        help="CPU workers for the exit scan")
    parser.add_argument("--reuse-sampling", type=Path, help="reuse a prior anchor output for refinement")
    parser.add_argument("--check", action="store_true", help="validate inputs and CUDA without sampling")
    parser.set_defaults(func=anchor_command)


def anchor_command(args):
    from .anchor_refinement import prepare_dihedrals, refine
    from .cli import CHECKPOINTS, LOCAL, prepare_crop, write_manifest
    from .demo import check_runtime, create_output, write_density_map, write_report

    try:
        output = create_output(args.output)
    except OSError as error:
        print(f"Could not create output directory: {error}", file=sys.stderr)
        return 2
    if output is None:
        return 2

    stage = "Input validation"
    log_path = output / "run.log"
    result = {"mode": "Fragment-anchored Local", "anchor_atoms_1_based": args.anchor_atoms,
              "anchor_tolerance_A": args.anchor_tolerance}
    try:
        import math
        from rdkit import Chem

        for field in ("protein", "ligand", "reference", "map"):
            path = getattr(args, field).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {field} input: {path}")
            setattr(args, field, path)
        clash_protein = args.clash_protein or args.protein
        clash_protein = clash_protein.expanduser().resolve()
        if not clash_protein.is_file():
            raise FileNotFoundError(f"Missing full receptor for clash checks: {clash_protein}")
        if min(args.samples, args.steps, args.batch_size) <= 0 or any(
            not math.isfinite(value) or value <= 0 for value in (args.anchor_tolerance, args.box_size)
        ):
            raise ValueError("Sample, step, batch, crop size and tolerance values must be positive.")
        if args.refinement_threads < 1:
            raise ValueError("Refinement threads must be positive.")
        if args.reuse_sampling and not args.exit_dihedrals:
            raise ValueError("--reuse-sampling requires --exit-dihedrals.")
        indices = [index - 1 for index in args.anchor_atoms]
        ligand = Chem.MolFromMolFile(str(args.ligand))
        xyz = validate_mapping(ligand,
                               Chem.MolFromMolFile(str(args.reference)), indices)
        dihedrals = prepare_dihedrals(ligand, indices, args.exit_dihedrals) if args.exit_dihedrals else []
        args.name, args.center, args.spacing = "anchored", xyz.mean(axis=0), 0.75
        args.flexible_sidechains = None
        stage = "Preparing density"
        with log_path.open("a") as log:
            crop = prepare_crop(args, output, log=log)
        import numpy as np
        with np.load(crop, allow_pickle=False) as data:
            density, origin, spacing = (data[key] for key in ("density", "origin", "spacing"))
        origin = np.asarray(origin).reshape(3)
        spacing = np.broadcast_to(spacing, (3,))
        if (density.ndim != 3 or min(density.shape) < 2
                or not np.isfinite(density).all() or not np.isfinite(origin).all()
                or not np.isfinite(spacing).all() or np.any(spacing <= 0)
                or np.any(xyz < origin)
                or np.any(xyz > origin + spacing * (np.asarray(density.shape) - 1))):
            raise ValueError("The density crop is invalid or the reference lies outside it. Check the map coordinate frame.")
        write_density_map(output / "density.mrc", density, origin, spacing)
        manifest = output / "input.csv"
        write_manifest(args, crop, manifest)
        config = output / "anchor.json"
        config.write_text(json.dumps({
            "ligand_path": str(args.ligand), "reference_path": str(args.reference),
            "protein_path": str(clash_protein), "crop_path": str(crop),
            "indices": indices, "anchor_tolerance": args.anchor_tolerance,
        }, indent=2) + "\n")
        if args.reuse_sampling:
            import filecmp

            stage = "Checking reused samples"
            reuse = args.reuse_sampling.expanduser().resolve()
            result["sampling_reused"] = True
            previous = json.loads((reuse / "anchor.json").read_text())
            current = json.loads(config.read_text())
            fields = ("indices", "ligand_path", "reference_path", "protein_path")
            if (any(current[key] != previous[key] for key in fields)
                    or not filecmp.cmp(crop, reuse / "density_crop.npz", shallow=False)):
                raise ValueError("Reused samples must have the same ligand, reference, anchor mapping, clash receptor and density crop.")
            prior_manifest = next(csv.DictReader((reuse / "input.csv").open()))
            if prior_manifest["experimental_protein"] != str(args.protein):
                raise ValueError("Reused samples must have the same inference receptor.")
            if not args.check:
                shutil.copytree(reuse / "candidates", output / "candidates")
                shutil.copy2(reuse / "candidate_scores.csv", output / "candidate_scores.csv")
        else:
            stage = "Checking CUDA and Local weights"
            check_runtime()
        result["status"] = "Ready"
        if not args.check and not args.reuse_sampling:
            stage = "Anchored sampling and candidate ranking"
            print(f"Sampling {args.samples} anchored poses. See {log_path} for progress.", flush=True)
            command = [
                sys.executable, "-u", "anchored_inference.py", "--anchor-config", str(config),
                "--protein_ligand_csv", str(manifest), "--out_dir", str(output / "candidates"),
                "--model_dir", str(CHECKPOINTS / "local_score"),
                "--filtering_model_dir", str(CHECKPOINTS / "local_confidence"),
                "--samples_per_complex", str(args.samples), "--batch_size", str(args.batch_size),
                "--inference_steps", str(args.steps), "--seed", str(args.seed),
                "--rigid", "--density_guidance", "--density_guidance_torsion",
                "--density_guidance_start", "0.3",
                "--density_guidance_lambda_tr", "0", "--density_guidance_lambda_rot", "0",
            ]
            with log_path.open("a") as log:
                code = subprocess.run(command, cwd=LOCAL, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 6000))
                    detail = log.read().decode(errors="replace")
                raise RuntimeError(f"Sampling or ranking exited with status {code}.\n{detail}")
        if not args.check:
            with (output / "candidate_scores.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            selected = (output / "final_pose.sdf").is_file()
            result.update(status="Completed" if selected else "Needs review",
                          sampled_candidates=len(rows),
                          sampled_eligible=sum(not row["review_reason"] for row in rows))
            if dihedrals:
                stage = "Exit torsion scanning and local density refinement"
                metrics, top = refine(output, LOCAL, manifest, config, output / "candidates",
                                      rows, dihedrals, args.anchor_tolerance, args.refinement_threads)
                result.update(metrics)
                if top:
                    shutil.copy2(top["pose"], output / "final_pose.sdf")
                    result.update(status="Completed", selected_candidate=top,
                                  selected_stage="Local density refinement")
                    selected = True
                elif not selected:
                    result["status"] = "Needs review"
            if selected and "selected_candidate" not in result:
                result["selected_candidate"] = rows[0]
                result["selected_stage"] = "Anchored sampling"
            if not selected:
                result["next_step"] = "Inspect the candidates and failed checks. The viewer opens the highest-density candidate for review."
            shutil.copy2(clash_protein, output / "receptor.pdb")
            shutil.copy2(args.reference, output / "anchor_reference.sdf")
            shown = "final_pose.sdf" if selected else rows[0]["file"]
            (output / "view.cxc").write_text(
                f"open receptor.pdb\nopen {json.dumps(shown, ensure_ascii=False)}\n"
                "open anchor_reference.sdf\nopen density.mrc\ncolor #2 orange\n"
                "color #3 cyan\nvolume #4 style mesh\nview #2,3\n"
            )
    except (Exception, KeyboardInterrupt) as error:
        with log_path.open("a") as log:
            traceback.print_exc(file=log)
        reason = str(error) or type(error).__name__
        hint = "Check the inputs and see run.log for details."
        if "out of memory" in reason.lower():
            hint = "Retry with a smaller --batch-size or a GPU with more free memory."
        elif stage == "Checking CUDA and Local weights":
            hint = "Run on an NVIDIA GPU node with the installed CryoDiffDock environment active."
        result.update(status="Failed", stage=stage, reason=reason, next_step=hint)
        print(reason.splitlines()[0], file=sys.stderr)
    write_report(output, result, title="CryoDiffDock · Fragment-anchored docking")
    print(f"{result['status']}. Open {output / 'report.html'}")
    return {"Failed": 1, "Needs review": 2}.get(result["status"], 0)
