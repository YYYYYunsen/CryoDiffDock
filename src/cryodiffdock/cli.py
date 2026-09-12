from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "engines" / "local"
GLOBAL = ROOT / "engines" / "global"
PREPROCESS = ROOT / "preprocessing"
CHECKPOINTS = ROOT / "checkpoints"


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def existing(path: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {result}")
    return result


def write_manifest(args, crop: Path, destination: Path) -> None:
    import pandas as pd

    row = {
        "complex_name": args.name,
        "experimental_protein": str(args.protein),
        "ligand": str(args.ligand),
        "ligand_density_crop": str(crop),
        "pocket_center_x": args.center[0],
        "pocket_center_y": args.center[1],
        "pocket_center_z": args.center[2],
    }
    if args.flexible_sidechains:
        row["flexible_sidechains"] = args.flexible_sidechains
    pd.DataFrame([row]).to_csv(destination, index=False)


def prepare_crop(args, output: Path) -> Path:
    crop = output / "density_crop.npz"
    run([
        sys.executable, str(LOCAL / "prepare_density_guidance_crop.py"),
        "--map", str(args.map), "--output", str(crop),
        "--center-x", str(args.center[0]), "--center-y", str(args.center[1]),
        "--center-z", str(args.center[2]), "--box-size", str(args.box_size),
        "--spacing", str(args.spacing),
    ], LOCAL)
    return crop


def local_command(args) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    crop = prepare_crop(args, output)
    manifest = output / "input.csv"
    write_manifest(args, crop, manifest)
    common = [
        sys.executable, "inference.py", "--protein_ligand_csv", str(manifest),
        "--model_dir", str(args.local_score_model),
        "--filtering_model_dir", str(args.local_confidence_model),
        "--samples_per_complex", str(args.samples), "--batch_size", str(args.batch_size),
        "--inference_steps", str(getattr(args, "local_steps", args.steps)), "--seed", str(args.seed),
    ]
    run(common + ["--out_dir", str(output / "baseline")], LOCAL)
    run(common + [
        "--density_guidance", "--density_guidance_objective", "atom_density",
        "--density_guidance_lambda_tr", "0.5", "--density_guidance_lambda_rot", "0.5",
        "--density_guidance_start", "0.6", "--density_guidance_clip_ratio", "1.0",
        "--out_dir", str(output / "guided"),
    ], LOCAL)
    scores = output / "ncc"
    run([
        sys.executable, "rerank_ligand_by_density_ncc.py", "--manifest", str(manifest),
        "--candidate-root", f"baseline={output / 'baseline'}",
        "--candidate-root", f"guided={output / 'guided'}",
        "--output-dir", str(scores), "--num-predictions", str(args.samples),
        "--sigmas", "1.5", "--ranking-sigma", "1.5", "--density-weight", "2.0",
        "--materialize-top-k", "1", "--fail-on-error",
    ], LOCAL)
    run([
        sys.executable, "run_adaptive_density_search.py", "--manifest", str(manifest),
        "--scores", str(scores / "pose_scores.csv"), "--output-dir", str(output / "search"),
        "--device", args.device, "--density-sigmas", "1.5,2.25", "--parent-count", "6",
        "--search-depth", "3", "--beam-width", "48", "--beam-width-per-parent", "10",
        "--coarse-beam-fraction", "0.25", "--translation-steps", "0.25,0.5",
        "--rotation-steps", "10,20", "--torsion-steps", "30,60",
        "--top-torsion-gradients", "2", "--materialization-mode", "top1", "--fail-on-error",
    ], LOCAL)
    print(f"CryoDiffDock-Local complete: {output / 'search' / 'v2_selected'}")
    return 0


def run_global_pipeline(args, output: Path) -> Path:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    ring = args.ring_features or output / "ring_features.npz"
    if args.ring_features is None:
        run([
            sys.executable, str(PREPROCESS / "build_ring_features.py"),
            "--structure", str(args.protein), "--map", str(args.map), "--out", str(ring),
            "--pdb-id", args.name, "--mask-structure", str(args.mask_structure or args.protein),
            "--mask-policy", "full_polymer", "--normalize", "percentile",
        ], PREPROCESS)
    command = [
        sys.executable, "inference_em.py", "--protein_path", str(args.protein),
        "--ligand_description", str(args.ligand), "--ring_features_path", str(ring),
        "--complex_name", args.name, "--out_dir", str(output / "poses"),
        "--model_dir", str(args.global_model), "--ckpt", args.global_ckpt,
        "--samples_per_complex", str(args.samples), "--batch_size", str(args.batch_size),
        "--inference_steps", str(getattr(args, "global_steps", args.steps)), "--seed", str(args.seed),
    ]
    if args.pretrained_score_model:
        command += ["--pretrained_score_model", str(args.pretrained_score_model)]
    run(command, GLOBAL)
    ranking = output / "ranking"
    run([
        sys.executable, "rank_global.py", "--name", args.name,
        "--protein", str(args.protein), "--ligand", str(args.ligand),
        "--map", str(args.map), "--ring-features", str(ring),
        "--poses", str(output / "poses"),
        "--confidence-model", str(args.global_confidence_model),
        "--confidence-ckpt", args.global_confidence_ckpt,
        "--output-dir", str(ranking),
        "--top-k", str(min(args.ranking_top_k, args.samples)),
        "--batch-size", str(args.batch_size),
    ], GLOBAL)
    return ranking / "global_scores.csv"


def global_command(args) -> int:
    output = args.output.resolve()
    scores = run_global_pipeline(args, output)
    print(f"CryoDiffDock-Global complete: {scores}")
    return 0


def cascade_command(args) -> int:
    """Run Global generation/ranking followed by Local Top-K refinement."""
    output = args.output.resolve()
    if args.top_k > min(args.ranking_top_k, args.samples):
        raise ValueError("--top-k cannot exceed the number of ranked Global samples")
    global_output = output / "global"
    global_scores = run_global_pipeline(args, global_output)
    input_dir = output / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = input_dir / "source_manifest.csv"
    cohort = input_dir / "ids.txt"
    import pandas as pd
    pd.DataFrame([{
        "complex_name": args.name,
        "experimental_protein": str(args.protein),
        "ligand": str(args.ligand),
        "map_path": str(args.map),
        "mask_structure": str(args.mask_structure or args.protein),
    }]).to_csv(source_manifest, index=False)
    cohort.write_text(args.name + "\n")
    source = output / "source"
    prepared = output / "prepared"
    local_out = output / "local"
    source.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "build_cascade_predicted_center_manifest.py",
        "--global-scores", str(global_scores), "--source-manifest", str(source_manifest),
        "--cohort", str(cohort), "--output-dir", str(source), "--top-k", str(args.top_k),
        "--score-column", args.score_column, "--seed", str(args.seed),
        "--flexible-distance", "3.5", "--no-flexible-policy", "rigid", "--fail-on-error",
    ], LOCAL)
    run([
        sys.executable, "prepare_predicted_center_density_dataset.py",
        "--manifest", str(source / "cascade_source_blind.csv"), "--output-dir", str(prepared),
        "--box-size", "28", "--spacing", "0.75", "--mask-inner", "1.5",
        "--mask-outer", "2.5", "--fail-on-error",
    ], LOCAL)
    manifest = prepared / "inference_blind.csv"
    common = [
        sys.executable, "inference.py", "--protein_ligand_csv", str(manifest),
        "--model_dir", str(args.local_score_model),
        "--filtering_model_dir", str(args.local_confidence_model),
        "--samples_per_complex", str(args.samples), "--batch_size", str(args.batch_size),
        "--inference_steps", str(args.local_steps), "--seed", str(args.seed),
    ]
    run(common + ["--out_dir", str(local_out / "baseline")], LOCAL)
    run(common + [
        "--density_guidance", "--density_guidance_objective", "atom_density",
        "--density_guidance_lambda_tr", "0.5", "--density_guidance_lambda_rot", "0.5",
        "--density_guidance_start", "0.6", "--density_guidance_clip_ratio", "1.0",
        "--out_dir", str(local_out / "guided"),
    ], LOCAL)
    scores = output / "ncc"
    run([
        sys.executable, "rerank_ligand_by_density_ncc.py", "--manifest", str(manifest),
        "--candidate-root", f"baseline={local_out / 'baseline'}",
        "--candidate-root", f"guided={local_out / 'guided'}", "--output-dir", str(scores),
        "--num-predictions", str(args.samples), "--sigmas", "1.5",
        "--ranking-sigma", "1.5", "--density-weight", "2.0", "--fail-on-error",
    ], LOCAL)
    run([
        sys.executable, "run_adaptive_density_search.py", "--manifest", str(manifest),
        "--scores", str(scores / "pose_scores.csv"), "--output-dir", str(output / "search"),
        "--device", args.device, "--density-sigmas", "1.5,2.25", "--parent-count", "6",
        "--search-depth", "3", "--beam-width", "48", "--beam-width-per-parent", "10",
        "--coarse-beam-fraction", "0.25", "--translation-steps", "0.25,0.5",
        "--rotation-steps", "10,20", "--torsion-steps", "30,60",
        "--top-torsion-gradients", "2", "--materialization-mode", "top1", "--fail-on-error",
    ], LOCAL)
    selected_root = output / "search" / "v2_selected"
    rank1_dirs = sorted(selected_root.glob("index0___*"))
    rank1_poses = sorted(rank1_dirs[0].glob("rank1_confidence*.sdf")) if len(rank1_dirs) == 1 else []
    if len(rank1_poses) != 1:
        raise RuntimeError("could not identify the refined Global-rank-1 Cascade pose")
    final_pose = output / "final_pose.sdf"
    shutil.copy2(rank1_poses[0], final_pose)
    print(f"CryoDiffDock-Cascade complete: {final_pose}")
    print(f"All {args.top_k} refined branches: {selected_root}")
    return 0


def doctor_command(args) -> int:
    modules = ["torch", "torch_geometric", "torch_scatter", "e3nn", "rdkit", "gemmi", "prody", "Bio"]
    report = {name: bool(importlib.util.find_spec(name)) for name in modules}
    report["cuda_available"] = False
    if report["torch"]:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
    manifest = json.loads((ROOT / "checkpoints" / "manifest.json").read_text())
    report["checkpoints"] = {}
    for name, item in manifest["files"].items():
        path = ROOT / item["path"]
        record = {"exists": path.is_file(), "path": str(path)}
        if path.is_file() and item.get("sha256"):
            record["sha256_ok"] = hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        report["checkpoints"][name] = record
    print(json.dumps(report, indent=2))
    dependencies_ok = all(report[name] for name in modules)
    checkpoints_ok = all(
        item["exists"] and item.get("sha256_ok", True)
        for item in report["checkpoints"].values()
    )
    return 0 if dependencies_ok and checkpoints_ok else 1


def add_common(parser, *, steps: int = 30):
    parser.add_argument("--name", default="complex")
    parser.add_argument("--protein", required=True, type=existing)
    parser.add_argument("--ligand", required=True, type=existing)
    parser.add_argument("--map", required=True, type=existing)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--seed", type=int, default=20260813)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="cryodiffdock")
    sub = result.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="check dependencies and model files")
    doctor.set_defaults(func=doctor_command)
    local = sub.add_parser("local", help="known-pocket Local inference and hierarchical search")
    add_common(local)
    local.add_argument("--center", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    local.add_argument("--flexible-sidechains")
    local.add_argument("--local-score-model", type=existing, default=CHECKPOINTS / "local_score")
    local.add_argument("--local-confidence-model", type=existing, default=CHECKPOINTS / "local_confidence")
    local.add_argument("--box-size", type=float, default=28.0)
    local.add_argument("--spacing", type=float, default=0.75)
    local.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    local.set_defaults(func=local_command)
    glob = sub.add_parser("global", help="site-agnostic Global pose generation")
    add_common(glob, steps=20)
    glob.add_argument("--global-model", type=existing, default=CHECKPOINTS / "global_score")
    glob.add_argument("--global-ckpt", default="best_ema_model.pt")
    glob.add_argument("--pretrained-score-model", type=existing, default=CHECKPOINTS / "global_pretrained" / "best_ema_inference_epoch_model.pt")
    glob.add_argument("--global-confidence-model", type=existing, default=CHECKPOINTS / "global_confidence")
    glob.add_argument("--global-confidence-ckpt", default="best_model.pt")
    glob.add_argument("--ranking-top-k", type=int, default=20)
    glob.add_argument("--ring-features", type=existing)
    glob.add_argument("--mask-structure", type=existing)
    glob.set_defaults(func=global_command)
    cascade = sub.add_parser("cascade", help="run Global ranking and Local Top-K refinement")
    add_common(cascade, steps=20)
    cascade.add_argument("--global-model", type=existing, default=CHECKPOINTS / "global_score")
    cascade.add_argument("--global-ckpt", default="best_ema_model.pt")
    cascade.add_argument("--pretrained-score-model", type=existing, default=CHECKPOINTS / "global_pretrained" / "best_ema_inference_epoch_model.pt")
    cascade.add_argument("--global-confidence-model", type=existing, default=CHECKPOINTS / "global_confidence")
    cascade.add_argument("--global-confidence-ckpt", default="best_model.pt")
    cascade.add_argument("--ranking-top-k", type=int, default=20)
    cascade.add_argument("--local-steps", type=int, default=30)
    cascade.add_argument("--ring-features", type=existing)
    cascade.add_argument("--mask-structure", type=existing)
    cascade.add_argument("--local-score-model", type=existing, default=CHECKPOINTS / "local_score")
    cascade.add_argument("--local-confidence-model", type=existing, default=CHECKPOINTS / "local_confidence")
    cascade.add_argument("--score-column", default="combined_at_best_weights")
    cascade.add_argument("--top-k", type=int, default=5)
    cascade.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cascade.set_defaults(func=cascade_command)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
