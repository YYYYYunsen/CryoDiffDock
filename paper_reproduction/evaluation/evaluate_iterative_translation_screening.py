#!/usr/bin/env python3
"""Screen cumulative fixed-step NCC translation on existing docking poses.

This is a validation-only, post-hoc diagnostic. It never reruns diffusion and
never uses ground truth to choose or accept a correction. For each generated
pose it performs repeated translation-only Gaussian-NCC ascent, records the
cumulative path/net displacement, and then uses deposited ligands solely to
measure complex-level top-1, best-of-N, rescue, and break outcomes.

The stored DiffDock confidence is not recomputed after correction. Therefore
``fixed_confidence_top1`` is a screening proxy. A complete reverse-diffusion
implementation must score its final corrected poses with the confidence model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_guidance import gaussian_ncc_batch, load_density_crop
from cryodiffdock_pocket.density_ranking import zscore
from evaluation.evaluate_candidate_density_gradient import (
    atomic_numbers,
    geometry_metrics,
    molecule_coordinates,
    molecule_with_coordinates,
)
from evaluation.evaluate_density_gradient_informativeness import Objective, resolve_path
from evaluation.evaluate_ligand_accuracy import (
    locate_result_directory,
    ranked_predictions,
    read_heavy_molecule,
)


def parse_positive_csv(value: str, label: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label} must be a comma-separated list of positive values")
    return values


def step_label(step_A: float) -> str:
    return f"translation_{float(step_A):g}A".replace(".", "p")


def in_bounds_fraction(
    coordinates: torch.Tensor,
    origin: torch.Tensor,
    spacing: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    upper = origin + (
        torch.as_tensor(shape, dtype=coordinates.dtype, device=coordinates.device) - 1.0
    ) * spacing
    return (
        ((coordinates >= origin) & (coordinates <= upper)).all(dim=-1).float().mean(dim=1)
    )


def iterative_translation(
    original: torch.Tensor,
    evaluator: Objective,
    step_cap_A: float,
    iterations: int,
    min_ncc_gain: float,
    batch_chunk_size: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]], dict[str, np.ndarray]]:
    """Run gated, translation-only NCC ascent for a batch of ligand poses."""
    current = original.detach().clone()
    original_centers = original.mean(dim=1)
    batch_size = original.shape[0]
    cumulative_path = torch.zeros(batch_size, dtype=current.dtype, device=current.device)
    accepted_steps = torch.zeros(batch_size, dtype=torch.long, device=current.device)
    audit_rows: list[dict[str, float | int]] = []

    with torch.no_grad():
        initial_ncc = gaussian_ncc_batch(
            evaluator.sample_points, current, evaluator.atomic_weights,
            evaluator.target_density, evaluator.sigma_A, batch_chunk_size,
        )

    for iteration in range(1, int(iterations) + 1):
        coordinates = current.detach().clone().requires_grad_(True)
        before = gaussian_ncc_batch(
            evaluator.sample_points, coordinates, evaluator.atomic_weights,
            evaluator.target_density, evaluator.sigma_A, batch_chunk_size,
        )
        coordinate_gradient = torch.autograd.grad(before.sum(), coordinates)[0]
        translation_gradient = coordinate_gradient.sum(dim=1)
        gradient_norm = torch.linalg.vector_norm(translation_gradient, dim=1)
        usable = gradient_norm > 1e-12
        direction = translation_gradient / gradient_norm.clamp_min(1e-12)[:, None]
        proposal = current + float(step_cap_A) * direction[:, None, :]
        with torch.no_grad():
            after = gaussian_ncc_batch(
                evaluator.sample_points, proposal, evaluator.atomic_weights,
                evaluator.target_density, evaluator.sigma_A, batch_chunk_size,
            )
            gain = after - before.detach()
            accepted = usable & (gain > float(min_ncc_gain))
            current = torch.where(accepted[:, None, None], proposal, current)
            cumulative_path = cumulative_path + accepted.to(current.dtype) * float(step_cap_A)
            accepted_steps = accepted_steps + accepted.long()
            net = torch.linalg.vector_norm(current.mean(dim=1) - original_centers, dim=1)
            bounds = in_bounds_fraction(
                current, evaluator.origin, evaluator.spacing, evaluator.density.shape
            )
        for pose_index in range(batch_size):
            audit_rows.append({
                "pose_index": int(pose_index),
                "iteration": int(iteration),
                "ncc_before": float(before[pose_index].detach().cpu()),
                "ncc_proposal": float(after[pose_index].detach().cpu()),
                "delta_ncc_proposal": float(gain[pose_index].detach().cpu()),
                "translation_gradient_norm": float(gradient_norm[pose_index].detach().cpu()),
                "accepted": int(accepted[pose_index].detach().cpu()),
                "cumulative_path_A": float(cumulative_path[pose_index].detach().cpu()),
                "net_translation_A": float(net[pose_index].detach().cpu()),
                "in_bounds_fraction": float(bounds[pose_index].detach().cpu()),
            })

    with torch.no_grad():
        final_ncc = gaussian_ncc_batch(
            evaluator.sample_points, current, evaluator.atomic_weights,
            evaluator.target_density, evaluator.sigma_A, batch_chunk_size,
        )
        final_net = torch.linalg.vector_norm(current.mean(dim=1) - original_centers, dim=1)
        final_bounds = in_bounds_fraction(
            current, evaluator.origin, evaluator.spacing, evaluator.density.shape
        )
    summary = {
        "initial_ncc": initial_ncc.detach().cpu().numpy(),
        "final_ncc": final_ncc.detach().cpu().numpy(),
        "accepted_steps": accepted_steps.detach().cpu().numpy(),
        "cumulative_path_A": cumulative_path.detach().cpu().numpy(),
        "net_translation_A": final_net.detach().cpu().numpy(),
        "in_bounds_fraction": final_bounds.detach().cpu().numpy(),
    }
    return current.detach(), audit_rows, summary


def build_complex_metrics(
    pose_table: pd.DataFrame,
    density_weight: float,
) -> pd.DataFrame:
    rows = []
    for (condition, row_index, complex_name), group in pose_table.groupby(
        ["condition", "row_index", "complex_name"], sort=False
    ):
        group = group.copy()
        group["confidence_z"] = zscore(group["confidence"])
        group["density_z"] = zscore(group["ncc_final"])
        group["combined_score"] = (
            group["confidence_z"] + float(density_weight) * group["density_z"]
        )
        fixed_top = group.sort_values(
            ["rank", "confidence"], ascending=[True, False], kind="stable"
        ).iloc[0]
        combined_top = group.sort_values(
            ["combined_score", "confidence", "rank"],
            ascending=[False, False, True], kind="stable",
        ).iloc[0]
        best = group.sort_values(
            ["final_ligand_rmsd_A", "rank"], ascending=[True, True], kind="stable"
        ).iloc[0]
        rows.append({
            "condition": condition,
            "step_cap_A": float(group["step_cap_A"].iloc[0]),
            "row_index": int(row_index),
            "complex_name": complex_name,
            "num_predictions": int(len(group)),
            "fixed_confidence_top1_rmsd_A": float(fixed_top["final_ligand_rmsd_A"]),
            "fixed_confidence_top1_source_rank": int(fixed_top["rank"]),
            "combined_top1_rmsd_A": float(combined_top["final_ligand_rmsd_A"]),
            "combined_top1_source_rank": int(combined_top["rank"]),
            "best_of_n_rmsd_A": float(best["final_ligand_rmsd_A"]),
            "best_of_n_source_rank": int(best["rank"]),
            "median_pose_rmsd_A": float(group["final_ligand_rmsd_A"].median()),
            "median_cumulative_path_A": float(group["cumulative_path_A"].median()),
            "median_net_translation_A": float(group["net_translation_A"].median()),
            "median_delta_ncc": float(group["delta_ncc"].median()),
            "mean_accepted_steps": float(group["accepted_steps"].mean()),
        })
    return pd.DataFrame(rows)


def safe_wilcoxon(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if not len(differences) or np.all(np.abs(differences) <= 1e-12):
        return 1.0
    return float(wilcoxon(differences).pvalue)


def paired_comparisons(complex_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = complex_metrics[complex_metrics["condition"] == "baseline"].copy()
    metrics = [
        "fixed_confidence_top1_rmsd_A",
        "combined_top1_rmsd_A",
        "best_of_n_rmsd_A",
    ]
    rows = []
    for condition in complex_metrics["condition"].drop_duplicates():
        if condition == "baseline":
            continue
        guided = complex_metrics[complex_metrics["condition"] == condition].copy()
        paired = baseline.merge(
            guided, on=["row_index", "complex_name"], suffixes=("_baseline", "_screened"),
            validate="one_to_one",
        )
        for metric in metrics:
            before = paired[f"{metric}_baseline"].to_numpy(dtype=float)
            after = paired[f"{metric}_screened"].to_numpy(dtype=float)
            delta = after - before
            record = {
                "condition": condition,
                "step_cap_A": float(paired["step_cap_A_screened"].iloc[0]),
                "metric": metric,
                "num_complexes": int(len(paired)),
                "median_baseline_A": float(np.median(before)),
                "median_screened_A": float(np.median(after)),
                "median_delta_A": float(np.median(delta)),
                "mean_delta_A": float(np.mean(delta)),
                "screened_better_pct": float((delta < -1e-7).mean() * 100.0),
                "screened_worse_pct": float((delta > 1e-7).mean() * 100.0),
                "wilcoxon_pvalue": safe_wilcoxon(delta),
            }
            for threshold in (2.0, 5.0):
                label = str(int(threshold))
                before_success = before < threshold
                after_success = after < threshold
                gained = (~before_success) & after_success
                lost = before_success & (~after_success)
                record[f"baseline_lt{label}_pct"] = float(before_success.mean() * 100.0)
                record[f"screened_lt{label}_pct"] = float(after_success.mean() * 100.0)
                record[f"gained_lt{label}"] = int(gained.sum())
                record[f"lost_lt{label}"] = int(lost.sum())
                record[f"net_rescue_lt{label}"] = int(gained.sum() - lost.sum())
            rows.append(record)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-predictions", type=int, default=40)
    parser.add_argument("--limit-complexes", type=int, default=50)
    parser.add_argument("--step-caps", default="0.02,0.05,0.10")
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--min-ncc-gain", type=float, default=0.0)
    parser.add_argument("--ncc-sigma", type=float, default=1.5)
    parser.add_argument("--ncc-mask-radius", type=float, default=10.0)
    parser.add_argument("--ncc-chunk-size", type=int, default=512,
                        help="Voxel chunk size for batched GPU NCC")
    parser.add_argument("--density-weight", type=float, default=2.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    step_caps = parse_positive_csv(args.step_caps, "--step-caps")
    if args.num_predictions <= 0 or args.iterations <= 0 or args.ncc_chunk_size <= 0:
        raise ValueError("Prediction count, iterations, and chunk size must be positive")
    if args.ncc_sigma <= 0 or args.ncc_mask_radius <= 0:
        raise ValueError("NCC sigma and mask radius must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    manifest = args.manifest.expanduser().resolve()
    candidate_root = args.candidate_root.expanduser().resolve()
    table = pd.read_csv(manifest)
    required = {"complex_name", "reference_ligand", "ligand_density_crop"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")
    if args.limit_complexes > 0:
        table = table.head(args.limit_complexes)
    if not candidate_root.is_dir():
        raise FileNotFoundError(candidate_root)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for position, (row_index, row) in enumerate(table.iterrows(), start=1):
        complex_name = str(row["complex_name"])
        try:
            reference_path = resolve_path(row["reference_ligand"], manifest.parent)
            crop_path = resolve_path(row["ligand_density_crop"], manifest.parent)
            reference = read_heavy_molecule(reference_path)
            crop = load_density_crop(crop_path)
            result_directory = locate_result_directory(candidate_root, int(row_index))
            predictions = ranked_predictions(result_directory, args.num_predictions)
            if not predictions:
                raise FileNotFoundError(f"No ranked poses under {result_directory}")

            molecules = []
            metadata = []
            expected_atomic_numbers = None
            for rank, confidence, prediction_path in predictions:
                molecule = read_heavy_molecule(prediction_path)
                numbers = atomic_numbers(molecule)
                if expected_atomic_numbers is None:
                    expected_atomic_numbers = numbers
                elif not np.array_equal(numbers, expected_atomic_numbers):
                    raise ValueError(f"Candidate atom ordering differs: {prediction_path}")
                molecules.append(molecule)
                metadata.append({
                    "rank": int(rank), "confidence": float(confidence),
                    "prediction_path": str(prediction_path.resolve()),
                })

            original = torch.as_tensor(
                np.stack([molecule_coordinates(molecule) for molecule in molecules]),
                dtype=torch.float32, device=device,
            )
            evaluator = Objective(
                "map_ncc", crop, expected_atomic_numbers, device,
                args.ncc_sigma, args.ncc_mask_radius, args.ncc_chunk_size,
            )
            with torch.no_grad():
                initial_ncc = gaussian_ncc_batch(
                    evaluator.sample_points, original, evaluator.atomic_weights,
                    evaluator.target_density, evaluator.sigma_A, args.ncc_chunk_size,
                ).detach().cpu().numpy()

            before_metrics = [geometry_metrics(reference, molecule) for molecule in molecules]
            for pose_index, (item, metrics) in enumerate(zip(metadata, before_metrics)):
                pose_rows.append({
                    "condition": "baseline", "step_cap_A": 0.0,
                    "row_index": int(row_index), "complex_name": complex_name,
                    **item, "ncc_initial": float(initial_ncc[pose_index]),
                    "ncc_final": float(initial_ncc[pose_index]), "delta_ncc": 0.0,
                    "accepted_steps": 0, "cumulative_path_A": 0.0,
                    "net_translation_A": 0.0, "in_bounds_fraction": float("nan"),
                    **{f"initial_{key}": value for key, value in metrics.items()},
                    **{f"final_{key}": value for key, value in metrics.items()},
                    **{f"delta_{key}": 0.0 for key in metrics},
                })

            for step_cap in step_caps:
                condition = step_label(step_cap)
                corrected, local_audit, correction_summary = iterative_translation(
                    original, evaluator, step_cap, args.iterations,
                    args.min_ncc_gain, args.ncc_chunk_size,
                )
                for audit in local_audit:
                    pose_index = int(audit.pop("pose_index"))
                    audit_rows.append({
                        "condition": condition, "step_cap_A": float(step_cap),
                        "row_index": int(row_index), "complex_name": complex_name,
                        **metadata[pose_index], **audit,
                    })
                for pose_index, (item, metrics) in enumerate(zip(metadata, before_metrics)):
                    corrected_molecule = molecule_with_coordinates(
                        molecules[pose_index], corrected[pose_index].cpu().numpy()
                    )
                    final_metrics = geometry_metrics(reference, corrected_molecule)
                    pose_rows.append({
                        "condition": condition, "step_cap_A": float(step_cap),
                        "row_index": int(row_index), "complex_name": complex_name,
                        **item,
                        "ncc_initial": float(correction_summary["initial_ncc"][pose_index]),
                        "ncc_final": float(correction_summary["final_ncc"][pose_index]),
                        "delta_ncc": float(
                            correction_summary["final_ncc"][pose_index]
                            - correction_summary["initial_ncc"][pose_index]
                        ),
                        "accepted_steps": int(correction_summary["accepted_steps"][pose_index]),
                        "cumulative_path_A": float(
                            correction_summary["cumulative_path_A"][pose_index]
                        ),
                        "net_translation_A": float(
                            correction_summary["net_translation_A"][pose_index]
                        ),
                        "in_bounds_fraction": float(
                            correction_summary["in_bounds_fraction"][pose_index]
                        ),
                        **{f"initial_{key}": value for key, value in metrics.items()},
                        **{f"final_{key}": value for key, value in final_metrics.items()},
                        **{
                            f"delta_{key}": final_metrics[key] - value
                            for key, value in metrics.items()
                        },
                    })
            print(
                f"[{position}/{len(table)}] {complex_name}: {len(predictions)} poses, "
                f"{len(step_caps)} caps OK"
            )
        except Exception as exc:
            failures.append({
                "row_index": int(row_index), "complex_name": complex_name,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{position}/{len(table)}] FAILED {complex_name}: {exc}")
            if args.fail_on_error:
                raise

    if not pose_rows:
        raise RuntimeError("No iterative screening trials completed")
    poses = pd.DataFrame(pose_rows)
    audits = pd.DataFrame(audit_rows)
    complex_metrics = build_complex_metrics(poses, args.density_weight)
    comparisons = paired_comparisons(complex_metrics)
    audit_summary = poses[poses["condition"] != "baseline"].groupby(
        ["condition", "step_cap_A"], sort=False
    ).agg(
        num_poses=("rank", "size"),
        mean_accepted_steps=("accepted_steps", "mean"),
        median_accepted_steps=("accepted_steps", "median"),
        median_cumulative_path_A=("cumulative_path_A", "median"),
        q95_cumulative_path_A=("cumulative_path_A", lambda values: values.quantile(0.95)),
        median_net_translation_A=("net_translation_A", "median"),
        q95_net_translation_A=("net_translation_A", lambda values: values.quantile(0.95)),
        median_delta_ncc=("delta_ncc", "median"),
        median_final_in_bounds_fraction=("in_bounds_fraction", "median"),
    ).reset_index()

    poses.to_csv(output / "pose_metrics.csv", index=False)
    audits.to_csv(output / "step_audit.csv", index=False)
    audit_summary.to_csv(output / "cumulative_audit_summary.csv", index=False)
    complex_metrics.to_csv(output / "complex_metrics.csv", index=False)
    comparisons.to_csv(output / "paired_comparison.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    report = {
        "manifest": str(manifest),
        "candidate_root": str(candidate_root),
        "device": str(device),
        "num_complexes": int(poses["row_index"].nunique()),
        "num_input_poses": int(
            poses[poses["condition"] == "baseline"][["row_index", "rank"]]
            .drop_duplicates().shape[0]
        ),
        "step_caps_A": step_caps,
        "iterations": int(args.iterations),
        "theoretical_max_paths_A": {
            step_label(step): float(step * args.iterations) for step in step_caps
        },
        "min_ncc_gain": float(args.min_ncc_gain),
        "ncc_sigma_A": float(args.ncc_sigma),
        "ncc_mask_radius_A": float(args.ncc_mask_radius),
        "density_weight": float(args.density_weight),
        "num_failures": int(len(failures)),
        "uses_ground_truth": True,
        "uses_ground_truth_for_acceptance": False,
        "reruns_diffusion": False,
        "recomputes_confidence": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nCumulative displacement audit")
    print(audit_summary.to_string(index=False))
    print("\nComplex-level paired outcomes")
    print(comparisons.to_string(index=False))
    print("\n" + json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
