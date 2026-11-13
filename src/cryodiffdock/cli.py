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


def run(command: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def paahs_environment() -> dict[str, str]:
    """Expose the Local engine to the frozen, repository-root paper scripts."""
    env = os.environ.copy()
    paths = [str(LOCAL), str(ROOT)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_paahs(
    baseline: Path, search_manifest: Path, scores: Path, output: Path,
    expected_cases: int, workers: int, device: str, *, preflight_only: bool = False,
) -> None:
    command = [
        sys.executable, str(ROOT / "scripts" / "run_paahs_pipeline.py"),
        "--baseline-manifest", str(baseline),
        "--search-manifest", str(search_manifest),
        "--scores", str(scores),
        "--output-root", str(output),
        "--expected-cases", str(expected_cases),
        "--workers", str(workers),
        "--device", device,
        "--directed-fallback-mode", "lazy",
    ]
    if preflight_only:
        command.append("--preflight-only")
    run(command, ROOT, env=paahs_environment())


def paahs_command(args) -> int:
    run_paahs(
        args.baseline_manifest, args.search_manifest, args.scores,
        args.output.resolve(), args.expected_cases, args.workers, args.device,
        preflight_only=args.preflight_only,
    )
    return 0


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
    if not args.flexible_sidechains:
        common.append("--rigid")
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
    if args.no_paahs:
        print(f"CryoDiffDock-Local V2-only complete: {output / 'search' / 'v2_selected'}")
        return 0
    import pandas as pd

    rankings = sorted((output / "search" / "v2_selected").glob("index*___*/ranking.csv"))
    if len(rankings) != 1:
        raise RuntimeError(f"Expected one baseline V2 ranking, found {len(rankings)}")
    ranking = pd.read_csv(rankings[0])
    if len(ranking) != 1 or "materialized_path" not in ranking:
        raise RuntimeError(f"Invalid V2 Top-1 ranking: {rankings[0]}")
    baseline_pose = Path(str(ranking.iloc[0]["materialized_path"])).resolve()
    if not baseline_pose.is_file():
        raise FileNotFoundError(baseline_pose)
    baseline = output / "paahs_baseline_manifest.csv"
    pd.DataFrame([{
        "case_order": 0,
        "complex_name": args.name,
        "baseline_pose_path": str(baseline_pose),
        "protein_path": str(args.protein),
        "density_crop": str(crop),
    }]).to_csv(baseline, index=False)
    run_paahs(
        baseline, manifest, scores / "pose_scores.csv", output / "paahs",
        1, args.paahs_workers, args.device,
    )
    final_pose = output / "paahs" / "final" / "cases" / args.name.upper() / "rank1.sdf"
    if not final_pose.is_file():
        raise FileNotFoundError(final_pose)
    shutil.copy2(final_pose, output / "final_pose.sdf")
    print(f"CryoDiffDock-Local PAAHS complete: {output / 'final_pose.sdf'}")
    return 0


def run_global_pipeline(args, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    ring = args.ring_features or output / "ring_features.npz"
    if args.ring_features is None:
        run([
            sys.executable, str(PREPROCESS / "build_ring_features.py"),
            "--structure", str(args.protein), "--map", str(args.map), "--out", str(ring),
            "--pdb-id", args.name, "--mask-structure", str(args.mask_structure or args.protein),
            "--mask-policy", "full_polymer", "--normalize", "percentile",
            "--radii", "4", "5", "6", "7", "8", "9",
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
    if not args.rank1_only and args.top_k != 5:
        raise ValueError("the frozen cross-branch selector requires --top-k 5")
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
        "--rigid",
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
    final_pose = output / "final_pose.sdf"
    if args.rank1_only:
        rank1_dirs = sorted(selected_root.glob("index0___*"))
        rank1_poses = sorted(rank1_dirs[0].glob("rank1_confidence*.sdf")) if len(rank1_dirs) == 1 else []
        if len(rank1_poses) != 1:
            raise RuntimeError("could not identify the refined Global-rank-1 Cascade pose")
        shutil.copy2(rank1_poses[0], final_pose)
    else:
        selector = output / "selector"
        run([
            sys.executable, str(ROOT / "evaluation" / "upgrade_cascade_selection.py"),
            "--cohort", str(cohort),
            "--blind-manifest", str(manifest),
            "--selected-root", str(selected_root),
            "--output-dir", str(selector),
            "--expected-n", "1",
            "--selection-only", "--materialize-selected",
        ], ROOT, env=paahs_environment())
        selected_pose = (
            selector / "selected_results" / "union_rank_fusion_guard"
            / args.name.upper() / "rank1.sdf"
        )
        if not selected_pose.is_file():
            raise FileNotFoundError(selected_pose)
        shutil.copy2(selected_pose, final_pose)
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
    local.add_argument(
        "--flexible-sidechains", metavar="RESIDUES",
        help="Enable flexible sampling for specified residues (e.g. A:10-A:14). "
             "The receptor is fixed during sampling by default.",
    )
    local.add_argument("--local-score-model", type=existing, default=CHECKPOINTS / "local_score")
    local.add_argument("--local-confidence-model", type=existing, default=CHECKPOINTS / "local_confidence")
    local.add_argument("--box-size", type=float, default=28.0)
    local.add_argument("--spacing", type=float, default=0.75)
    local.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    local.add_argument("--paahs-workers", type=int, default=4)
    local.add_argument("--no-paahs", action="store_true", help="stop at legacy V2 Top-1")
    local.set_defaults(func=local_command)
    paahs = sub.add_parser("paahs", help="run frozen lazy PAAHS on a blind manifest")
    paahs.add_argument("--baseline-manifest", required=True, type=existing)
    paahs.add_argument("--search-manifest", required=True, type=existing)
    paahs.add_argument("--scores", required=True, type=existing)
    paahs.add_argument("--output", required=True, type=Path)
    paahs.add_argument("--expected-cases", required=True, type=int)
    paahs.add_argument("--workers", type=int, default=4)
    paahs.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    paahs.add_argument("--preflight-only", action="store_true")
    paahs.set_defaults(func=paahs_command)
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
    cascade.add_argument("--rank1-only", action="store_true", help="retain the older Global-rank-1 output")
    cascade.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cascade.set_defaults(func=cascade_command)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
