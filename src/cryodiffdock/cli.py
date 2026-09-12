from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "engines" / "local"
GLOBAL = ROOT / "engines" / "global"
PREPROCESS = ROOT / "preprocessing"


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
        "--inference_steps", str(args.steps), "--seed", str(args.seed),
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


def global_command(args) -> int:
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
        "--inference_steps", str(args.steps), "--seed", str(args.seed),
    ]
    if args.pretrained_score_model:
        command += ["--pretrained_score_model", str(args.pretrained_score_model)]
    run(command, GLOBAL)
    print(f"CryoDiffDock-Global generation complete: {output / 'poses' / args.name}")
    print("The generated files are an unranked pose ensemble; use the released confidence/ranking workflow for paper-equivalent Top-1 selection.")
    return 0


def cascade_command(args) -> int:
    """Run the released Global-Top-K to Local refinement stages.

    Global scores must already contain the frozen confidence+density score. This
    keeps the public command from silently substituting a different ranker.
    """
    output = args.output.resolve()
    source = output / "source"
    prepared = output / "prepared"
    local_out = output / "local"
    source.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "build_cascade_predicted_center_manifest.py",
        "--global-scores", str(args.global_scores), "--source-manifest", str(args.source_manifest),
        "--cohort", str(args.cohort), "--output-dir", str(source), "--top-k", str(args.top_k),
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
        "--inference_steps", str(args.steps), "--seed", str(args.seed),
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
    print(f"Cascade branch refinement complete: {output / 'search' / 'v2_selected'}")
    print("Apply the validation-frozen parent selector in paper_reproduction/evaluation for paper-equivalent final Top-1 selection.")
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
    return 0 if all(report[name] for name in modules) else 1


def add_common(parser):
    parser.add_argument("--name", default="complex")
    parser.add_argument("--protein", required=True, type=existing)
    parser.add_argument("--ligand", required=True, type=existing)
    parser.add_argument("--map", required=True, type=existing)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
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
    local.add_argument("--local-score-model", required=True, type=existing)
    local.add_argument("--local-confidence-model", required=True, type=existing)
    local.add_argument("--box-size", type=float, default=28.0)
    local.add_argument("--spacing", type=float, default=0.75)
    local.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    local.set_defaults(func=local_command)
    glob = sub.add_parser("global", help="site-agnostic Global pose generation")
    add_common(glob)
    glob.add_argument("--global-model", required=True, type=existing)
    glob.add_argument("--global-ckpt", default="best_ema_model.pt")
    glob.add_argument("--pretrained-score-model", type=existing)
    glob.add_argument("--ring-features", type=existing)
    glob.add_argument("--mask-structure", type=existing)
    glob.set_defaults(func=global_command)
    cascade = sub.add_parser("cascade", help="refine frozen-ranked Global Top-K poses with Local")
    cascade.add_argument("--global-scores", required=True, type=existing)
    cascade.add_argument("--source-manifest", required=True, type=existing)
    cascade.add_argument("--cohort", required=True, type=existing)
    cascade.add_argument("--output", required=True, type=Path)
    cascade.add_argument("--local-score-model", required=True, type=existing)
    cascade.add_argument("--local-confidence-model", required=True, type=existing)
    cascade.add_argument("--score-column", default="combined_at_best_weights")
    cascade.add_argument("--top-k", type=int, default=5)
    cascade.add_argument("--samples", type=int, default=40)
    cascade.add_argument("--batch-size", type=int, default=10)
    cascade.add_argument("--steps", type=int, default=30)
    cascade.add_argument("--seed", type=int, default=20260813)
    cascade.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cascade.set_defaults(func=cascade_command)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
