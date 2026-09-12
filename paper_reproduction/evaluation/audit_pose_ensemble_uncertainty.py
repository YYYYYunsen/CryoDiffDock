#!/usr/bin/env python3
"""Audit whether pose-ensemble disagreement predicts Top-1 docking failure.

All ensemble features are computed without reading a reference ligand.  The
reference is read only after the operational Top-1 pose has been selected, and
is used solely to assign the retrospective RMSD/success label.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def choose_column(columns, candidates, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; found {list(columns)}")


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def molecule_coordinates(molecule) -> tuple[np.ndarray, tuple[int, ...]]:
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
    atomic_numbers = tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    if not np.isfinite(coordinates).all():
        raise ValueError("Pose contains non-finite coordinates")
    return coordinates, atomic_numbers


def ordered_pairwise_rmsd(molecules) -> np.ndarray:
    """Pairwise unaligned RMSD when generated poses preserve atom order."""
    unpacked = [molecule_coordinates(molecule) for molecule in molecules]
    atomic_numbers = unpacked[0][1]
    if any(item[1] != atomic_numbers for item in unpacked[1:]):
        raise ValueError(
            "Heavy-atom order differs among poses; use --pairwise-rmsd-mode symmetry"
        )
    coordinates = np.stack([item[0] for item in unpacked])
    differences = coordinates[:, None, :, :] - coordinates[None, :, :, :]
    return np.sqrt(np.mean(np.sum(differences * differences, axis=-1), axis=-1))


def pairwise_rmsd_from_isomorphisms(
    coordinates: np.ndarray,
    isomorphisms,
) -> np.ndarray:
    """Compute an exact all-pairs symmetry RMSD from cached atom mappings."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must have shape (poses, atoms, 3)")
    pose_count, atom_count, _ = coordinates.shape
    if pose_count == 0 or atom_count == 0:
        raise ValueError("coordinates must contain poses and atoms")
    minimum_squared = np.full((pose_count, pose_count), np.inf, dtype=float)
    mapping_count = 0
    for left_indices, right_indices in isomorphisms:
        left_indices = np.asarray(left_indices, dtype=int)
        right_indices = np.asarray(right_indices, dtype=int)
        if len(left_indices) != atom_count or len(right_indices) != atom_count:
            raise ValueError("isomorphism does not cover every atom")
        left = coordinates[:, left_indices, :]
        right = coordinates[:, right_indices, :]
        left_norm = np.einsum("pia,pia->p", left, left)
        right_norm = np.einsum("pia,pia->p", right, right)
        cross = np.einsum("pia,qia->pq", left, right, optimize=True)
        squared = (
            left_norm[:, None] + right_norm[None, :] - 2.0 * cross
        ) / atom_count
        minimum_squared = np.minimum(minimum_squared, np.maximum(squared, 0.0))
        mapping_count += 1
    if mapping_count == 0 or not np.isfinite(minimum_squared).all():
        raise ValueError("no usable graph isomorphisms were generated")
    distances = np.sqrt(minimum_squared)
    # The complete automorphism set contains inverse mappings, so the matrix is
    # symmetric up to floating-point noise.  Symmetrize only that numerical noise.
    distances = np.minimum(distances, distances.T)
    np.fill_diagonal(distances, 0.0)
    return distances


def symmetry_pairwise_rmsd(molecules) -> np.ndarray:
    """All-pairs symmetry RMSD, enumerating graph automorphisms only once.

    V2 finalists originate from the same input molecule and normally preserve
    both atom order and connectivity.  In that case the self-isomorphisms of the
    shared molecular graph are cached once and reused for every pose pair.  The
    legacy pairwise path remains as a correctness fallback for heterogeneous
    atom ordering or connectivity.
    """
    from spyrmsd import graph, molecule

    count = len(molecules)
    if count == 0:
        raise ValueError("At least one molecule is required")
    converted = [molecule.Molecule.from_rdkit(value) for value in molecules]
    reference_atomicnums = np.asarray(converted[0].atomicnums)
    reference_adjacency = np.asarray(converted[0].adjacency_matrix)
    homogeneous_graph = all(
        np.array_equal(np.asarray(value.atomicnums), reference_atomicnums)
        and np.array_equal(np.asarray(value.adjacency_matrix), reference_adjacency)
        for value in converted[1:]
    )
    if homogeneous_graph:
        molecular_graph = graph.graph_from_adjacency_matrix(
            reference_adjacency,
            reference_atomicnums,
        )
        isomorphisms = graph.match_graphs(molecular_graph, molecular_graph)
        coordinates = np.stack([
            np.asarray(value.GetConformer().GetPositions(), dtype=float)
            for value in molecules
        ])
        return pairwise_rmsd_from_isomorphisms(coordinates, isomorphisms)

    distances = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            value = symmetry_rmsd(molecules[left], molecules[right])
            distances[left, right] = distances[right, left] = value
    return distances


def cluster_features(
    distances: np.ndarray,
    cutoffs: list[float],
    selected_index: int,
) -> dict[str, float | int]:
    count = len(distances)
    if distances.shape != (count, count) or count == 0:
        raise ValueError("Pairwise distance matrix must be non-empty and square")
    if count == 1:
        condensed = np.array([], dtype=float)
        hierarchy = None
    else:
        condensed = squareform(distances, checks=False)
        hierarchy = linkage(condensed, method="complete")

    upper = distances[np.triu_indices(count, k=1)]
    result: dict[str, float | int] = {
        "pairwise_rmsd_median_A": float(np.median(upper)) if len(upper) else 0.0,
        "pairwise_rmsd_p90_A": float(np.quantile(upper, 0.9)) if len(upper) else 0.0,
    }
    for cutoff in cutoffs:
        token = str(cutoff).replace(".", "p")
        labels = (
            np.ones(1, dtype=int)
            if hierarchy is None
            else fcluster(hierarchy, t=cutoff, criterion="distance")
        )
        sizes = np.bincount(labels)[1:]
        probabilities = sizes.astype(float) / count
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        normalized_entropy = entropy / math.log(count) if count > 1 else 0.0
        selected_cluster_size = int(np.sum(labels == labels[selected_index]))
        result.update({
            f"cluster_count_at_{token}A": int(len(sizes)),
            f"cluster_entropy_at_{token}A": normalized_entropy,
            f"largest_cluster_fraction_at_{token}A": float(sizes.max() / count),
            f"selected_cluster_fraction_at_{token}A": float(selected_cluster_size / count),
        })
    return result


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        value = binary_auc(labels[indices], scores[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def bootstrap_selective_success(
    labels: np.ndarray,
    scores: np.ndarray,
    coverages: tuple[float, ...],
    repeats: int,
    seed: int,
) -> dict[float, dict[str, float]]:
    """Bootstrap selective success and its gain over full valid coverage.

    Cases are resampled first, and the retained most-certain subset is selected
    again within every bootstrap replicate.  This preserves uncertainty-ranking
    variation instead of treating the originally retained subset as fixed.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if not len(labels):
        raise ValueError("No finite uncertainty scores for selective bootstrap")
    if any(coverage <= 0 or coverage > 1 for coverage in coverages):
        raise ValueError("coverages must be in (0, 1]")

    rng = np.random.default_rng(seed)
    success_samples = {coverage: [] for coverage in coverages}
    gain_samples = {coverage: [] for coverage in coverages}
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        sampled_scores = scores[indices]
        full_success_pct = 100.0 * float((~sampled_labels).mean())
        ordered = np.argsort(sampled_scores, kind="stable")
        for coverage in coverages:
            retained_n = max(1, int(math.ceil(coverage * len(ordered))))
            retained = ordered[:retained_n]
            success_pct = 100.0 * float((~sampled_labels[retained]).mean())
            success_samples[coverage].append(success_pct)
            gain_samples[coverage].append(success_pct - full_success_pct)

    result = {}
    for coverage in coverages:
        success_low, success_high = np.quantile(
            success_samples[coverage], [0.025, 0.975]
        )
        gain_low, gain_high = np.quantile(
            gain_samples[coverage], [0.025, 0.975]
        )
        result[coverage] = {
            "success_95ci_low_pct": float(success_low),
            "success_95ci_high_pct": float(success_high),
            "gain_95ci_low_pp": float(gain_low),
            "gain_95ci_high_pp": float(gain_high),
        }
    return result


def uncertainty_columns(table: pd.DataFrame, cutoffs: list[float]) -> dict[str, np.ndarray]:
    columns = {
        "negative_standardized_score_gap": -table["standardized_score_gap"].to_numpy(float),
        "pairwise_rmsd_median_A": table["pairwise_rmsd_median_A"].to_numpy(float),
        "pairwise_rmsd_p90_A": table["pairwise_rmsd_p90_A"].to_numpy(float),
    }
    for cutoff in cutoffs:
        token = str(cutoff).replace(".", "p")
        columns[f"cluster_entropy_at_{token}A"] = table[
            f"cluster_entropy_at_{token}A"
        ].to_numpy(float)
        columns[f"one_minus_largest_cluster_fraction_at_{token}A"] = 1.0 - table[
            f"largest_cluster_fraction_at_{token}A"
        ].to_numpy(float)
        columns[f"one_minus_selected_cluster_fraction_at_{token}A"] = 1.0 - table[
            f"selected_cluster_fraction_at_{token}A"
        ].to_numpy(float)
    return columns


def evaluate_complex(task):
    (
        complex_name,
        rows,
        reference_path,
        score_column,
        auxiliary_columns,
        cutoffs,
        pairwise_mode,
        pose_scores_parent,
        success_cutoff,
        provided_rmsd_column,
    ) = task
    try:
        feature_started = time.perf_counter()
        group = pd.DataFrame(rows)
        numeric_scores = pd.to_numeric(group[score_column], errors="coerce")
        if not np.isfinite(numeric_scores).any():
            raise ValueError(f"No finite values in score column {score_column!r}")
        group = group.assign(_selection_score=numeric_scores).sort_values(
            ["_selection_score", "source_rank", "prediction_path"],
            ascending=[False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        paths = [resolve_path(value, Path(pose_scores_parent)) for value in group["prediction_path"]]
        pose_read_started = time.perf_counter()
        molecules = [read_heavy_molecule(path) for path in paths]
        pose_read_seconds = time.perf_counter() - pose_read_started
        pairwise_started = time.perf_counter()
        distances = (
            ordered_pairwise_rmsd(molecules)
            if pairwise_mode == "ordered"
            else symmetry_pairwise_rmsd(molecules)
        )
        pairwise_rmsd_seconds = time.perf_counter() - pairwise_started
        result = {
            "complex_name": complex_name,
            "pose_count": len(group),
            "selected_source": str(group.iloc[0]["source"]) if "source" in group else "",
            "selected_source_rank": int(group.iloc[0]["source_rank"]),
            "selected_prediction_path": str(paths[0]),
            "selected_score": float(group.iloc[0]["_selection_score"]),
        }
        finite = np.sort(numeric_scores[np.isfinite(numeric_scores)].to_numpy())[::-1]
        raw_gap = float(finite[0] - finite[1]) if len(finite) > 1 else float("nan")
        score_std = float(np.std(finite))
        result["score_gap"] = raw_gap
        result["standardized_score_gap"] = (
            raw_gap / score_std if score_std > 0 and np.isfinite(raw_gap) else float("nan")
        )
        for column in auxiliary_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            finite_values = values[np.isfinite(values)]
            result[f"{column}_std"] = (
                float(np.std(finite_values)) if len(finite_values) else float("nan")
            )
            result[f"selected_{column}"] = (
                float(values[0]) if np.isfinite(values[0]) else float("nan")
            )
        clustering_started = time.perf_counter()
        result.update(cluster_features(distances, cutoffs, selected_index=0))
        result["clustering_seconds"] = time.perf_counter() - clustering_started
        result["pose_read_seconds"] = pose_read_seconds
        result["pairwise_rmsd_seconds"] = pairwise_rmsd_seconds
        result["ensemble_feature_seconds"] = time.perf_counter() - feature_started

        if provided_rmsd_column is not None:
            selected_rmsd = float(group.iloc[0][provided_rmsd_column])
            if not np.isfinite(selected_rmsd):
                raise ValueError("Selected pose has no finite supplied ligand RMSD")
        else:
            reference = read_heavy_molecule(Path(reference_path))
            selected_rmsd = symmetry_rmsd(reference, molecules[0])
        result["selected_top1_rmsd_A"] = selected_rmsd
        result["top1_success"] = bool(selected_rmsd < success_cutoff)
        result["top1_failure"] = not result["top1_success"]
        return result, None
    except Exception as exc:
        return None, {
            "complex_name": complex_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    audit_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-scores", required=True, type=Path)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument(
        "--evaluation-manifest", type=Path,
        help="Manifest with one reference_ligand per complex.",
    )
    labels.add_argument(
        "--per-pose-rmsd", type=Path,
        help=(
            "Existing same-run per-pose table containing prediction_path and "
            "ligand_rmsd_A; avoids reopening deposited references."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pool", default="guided", help="Source to audit, or 'all'.")
    parser.add_argument("--score-column", default="combined_score")
    parser.add_argument("--aux-score-column", action="append", default=[])
    parser.add_argument("--cluster-cutoffs", default="1.5,2.0")
    parser.add_argument("--pairwise-rmsd-mode", choices=("ordered", "symmetry"), default="ordered")
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument("--expected-poses", type=int, default=40)
    parser.add_argument(
        "--allowed-pose-counts",
        default=None,
        help=(
            "Comma-separated accepted pose counts per complex. When supplied, "
            "this overrides --expected-poses for pose-count auditing (for example "
            "48,49 for V2 finalists, where the frozen best original may be appended)."
        ),
    )
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.workers <= 0 or args.bootstrap_repeats <= 0 or args.expected_poses <= 0:
        raise ValueError("workers, bootstrap-repeats, and expected-poses must be positive")
    allowed_pose_counts = (
        {int(value) for value in args.allowed_pose_counts.split(",") if value.strip()}
        if args.allowed_pose_counts is not None else
        {args.expected_poses}
    )
    if not allowed_pose_counts or any(value <= 0 for value in allowed_pose_counts):
        raise ValueError("allowed-pose-counts must contain positive integers")
    cutoffs = [float(value) for value in args.cluster_cutoffs.split(",") if value.strip()]
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("cluster-cutoffs must contain positive values")

    score_path = args.pose_scores.expanduser().resolve()
    manifest_path = (
        args.evaluation_manifest.expanduser().resolve()
        if args.evaluation_manifest is not None else None
    )
    per_pose_rmsd_path = (
        args.per_pose_rmsd.expanduser().resolve()
        if args.per_pose_rmsd is not None else None
    )
    scores = pd.read_csv(score_path)
    required_scores = {"prediction_path", "source_rank", args.score_column}
    missing_scores = sorted(required_scores - set(scores.columns))
    if missing_scores:
        raise ValueError(f"Pose scores missing columns: {missing_scores}")
    missing_aux = sorted(set(args.aux_score_column) - set(scores.columns))
    if missing_aux:
        raise ValueError(f"Pose scores missing auxiliary columns: {missing_aux}")
    score_id = choose_column(scores.columns, ID_COLUMNS, "pose-score complex ID")
    scores = scores.copy()
    scores["complex_name"] = scores[score_id].map(normalize_id)
    references = None
    provided_rmsd_column = None
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path, dtype=str)
        if "reference_ligand" not in manifest.columns:
            raise ValueError("Evaluation manifest requires a reference_ligand column")
        manifest_id = choose_column(manifest.columns, ID_COLUMNS, "manifest complex ID")
        manifest["complex_name"] = manifest[manifest_id].map(normalize_id)
        reference_counts = manifest.groupby("complex_name")["reference_ligand"].nunique()
        conflicts = reference_counts[reference_counts > 1]
        if len(conflicts):
            raise ValueError(f"Evaluation manifest has conflicting references: {list(conflicts.index)[:20]}")
        manifest = manifest.drop_duplicates("complex_name")
        references = {
            row["complex_name"]: resolve_path(row["reference_ligand"], manifest_path.parent)
            for _, row in manifest.iterrows()
        }
    else:
        supplied = pd.read_csv(per_pose_rmsd_path)
        required_supplied = {"prediction_path", "ligand_rmsd_A"}
        missing_supplied = sorted(required_supplied - set(supplied.columns))
        if missing_supplied:
            raise ValueError(f"Per-pose RMSD table missing columns: {missing_supplied}")
        supplied = supplied.copy()
        supplied["_resolved_prediction_path"] = supplied["prediction_path"].map(
            lambda value: str(resolve_path(value, per_pose_rmsd_path.parent))
        )
        supplied["_provided_ligand_rmsd_A"] = pd.to_numeric(
            supplied["ligand_rmsd_A"], errors="coerce"
        )
        rmsd_counts = supplied.groupby("_resolved_prediction_path")[
            "_provided_ligand_rmsd_A"
        ].nunique(dropna=True)
        conflicts = rmsd_counts[rmsd_counts > 1]
        if len(conflicts):
            raise ValueError(f"Per-pose table has conflicting RMSDs for {len(conflicts)} paths")
        supplied = supplied.drop_duplicates("_resolved_prediction_path")
        scores["_resolved_prediction_path"] = scores["prediction_path"].map(
            lambda value: str(resolve_path(value, score_path.parent))
        )
        scores = scores.merge(
            supplied[["_resolved_prediction_path", "_provided_ligand_rmsd_A"]],
            on="_resolved_prediction_path", how="left", validate="many_to_one",
        )
        provided_rmsd_column = "_provided_ligand_rmsd_A"
    if args.pool != "all":
        if "source" not in scores.columns:
            raise ValueError("--pool requires a source column in pose scores")
        scores = scores[scores["source"].astype(str) == args.pool].copy()
    if args.cohort:
        cohort = {
            normalize_id(line) for line in args.cohort.expanduser().resolve().read_text().splitlines()
            if line.strip()
        }
        scores = scores[scores["complex_name"].isin(cohort)].copy()
    pose_counts = scores.groupby("complex_name").size()
    wrong_counts = pose_counts[~pose_counts.isin(allowed_pose_counts)]
    if references is not None:
        missing_references = sorted(set(pose_counts.index) - set(references))
        if missing_references:
            raise ValueError(f"Missing reference rows for: {missing_references[:20]}")

    tasks = []
    for complex_name, group in scores.groupby("complex_name", sort=True):
        tasks.append((
            complex_name,
            group.to_dict("records"),
            str(references[complex_name]) if references is not None else None,
            args.score_column,
            args.aux_score_column,
            cutoffs,
            args.pairwise_rmsd_mode,
            str(score_path.parent),
            args.success_cutoff,
            provided_rmsd_column,
        ))
    feature_evaluation_started = time.perf_counter()
    if args.workers == 1:
        evaluated = map(evaluate_complex, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        evaluated = executor.map(evaluate_complex, tasks, chunksize=1)

    rows, failures = [], []
    try:
        for index, (row, failure) in enumerate(evaluated, start=1):
            if failure:
                failures.append(failure)
                print(f"[{index}/{len(tasks)}] FAILED {failure['complex_name']}: {failure['error']}", flush=True)
                if args.fail_on_error:
                    raise RuntimeError(failure["error"])
            else:
                rows.append(row)
                print(
                    f"[{index}/{len(tasks)}] {row['complex_name']}: "
                    f"poses={row['pose_count']} top1={row['selected_top1_rmsd_A']:.3f} A",
                    flush=True,
                )
    finally:
        if args.workers != 1:
            executor.shutdown()
    feature_evaluation_wall_seconds = time.perf_counter() - feature_evaluation_started
    if not rows:
        raise RuntimeError("No complexes were evaluated")

    per_complex = pd.DataFrame(rows).sort_values("complex_name")
    labels = per_complex["top1_failure"].to_numpy(bool)
    uncertainty = uncertainty_columns(per_complex, cutoffs)
    metric_rows = []
    risk_rows = []
    quartile_rows = []
    coverages = (0.25, 0.5, 0.75, 1.0)
    statistical_summary_started = time.perf_counter()
    for feature_index, (name, values) in enumerate(uncertainty.items()):
        auc = binary_auc(labels, values)
        low, high = bootstrap_auc(
            labels, values, args.bootstrap_repeats, args.seed + feature_index
        )
        correlation = spearmanr(
            values, per_complex["selected_top1_rmsd_A"].to_numpy(float), nan_policy="omit"
        ).statistic
        metric_rows.append({
            "uncertainty_feature": name,
            "failure_auroc": auc,
            "bootstrap_95ci_low": low,
            "bootstrap_95ci_high": high,
            "spearman_vs_top1_rmsd": float(correlation),
        })

        valid_indices = np.flatnonzero(np.isfinite(values))
        ordered = valid_indices[np.argsort(values[valid_indices], kind="stable")]
        feature_full_success_pct = 100.0 * float((~labels[valid_indices]).mean())
        selective_bootstrap = bootstrap_selective_success(
            labels,
            values,
            coverages,
            args.bootstrap_repeats,
            args.seed + 1000 + feature_index,
        )
        for coverage in coverages:
            retained = ordered[:max(1, int(math.ceil(coverage * len(ordered))))]
            successes = ~labels[retained]
            success_pct = 100.0 * float(successes.mean())
            risk_rows.append({
                "uncertainty_feature": name,
                "target_coverage": coverage,
                "retained_n": len(retained),
                "success_count": int(successes.sum()),
                "success_pct": success_pct,
                "failure_pct": 100.0 * float(labels[retained].mean()),
                "success_95ci_low_pct": selective_bootstrap[coverage]["success_95ci_low_pct"],
                "success_95ci_high_pct": selective_bootstrap[coverage]["success_95ci_high_pct"],
                "gain_vs_full_percentage_points": success_pct - feature_full_success_pct,
                "gain_95ci_low_pp": selective_bootstrap[coverage]["gain_95ci_low_pp"],
                "gain_95ci_high_pp": selective_bootstrap[coverage]["gain_95ci_high_pp"],
            })
        for quartile, indices in enumerate(np.array_split(ordered, 4), start=1):
            quartile_rows.append({
                "uncertainty_feature": name,
                "uncertainty_quartile": quartile,
                "n": len(indices),
                "median_top1_rmsd_A": float(per_complex.iloc[indices]["selected_top1_rmsd_A"].median()),
                "failure_pct": 100.0 * float(labels[indices].mean()),
            })

    metric_table = pd.DataFrame(metric_rows)
    risk_table = pd.DataFrame(risk_rows)
    quartile_table = pd.DataFrame(quartile_rows)
    primary_cutoff = min(cutoffs, key=lambda value: abs(value - 2.0))
    primary_token = str(primary_cutoff).replace(".", "p")
    primary_name = f"one_minus_largest_cluster_fraction_at_{primary_token}A"
    primary_metric = next(
        row for row in metric_rows if row["uncertainty_feature"] == primary_name
    )
    primary_risk = risk_table[risk_table["uncertainty_feature"] == primary_name]
    certain_half_success = float(
        primary_risk.loc[primary_risk["target_coverage"] == 0.5, "success_pct"].iloc[0]
    )
    full_success = 100.0 * float(per_complex["top1_success"].mean())
    certain_half_gain = certain_half_success - full_success
    certain_half_row = primary_risk.loc[
        primary_risk["target_coverage"] == 0.5
    ].iloc[0]
    full_coverage_row = primary_risk.loc[
        primary_risk["target_coverage"] == 1.0
    ].iloc[0]
    primary_quartiles = quartile_table[
        quartile_table["uncertainty_feature"] == primary_name
    ].sort_values("uncertainty_quartile")
    quartile_failure = primary_quartiles["failure_pct"].to_numpy(float)
    monotonic_quartiles = bool(np.all(np.diff(quartile_failure) >= -1e-12))
    primary_auc = float(primary_metric["failure_auroc"])
    if primary_auc >= 0.65 and certain_half_gain >= 10.0 and monotonic_quartiles:
        screening_verdict = "PROMISING"
    elif primary_auc >= 0.58 or certain_half_gain >= 5.0:
        screening_verdict = "BORDERLINE"
    else:
        screening_verdict = "WEAK"
    decision_support = {
        "heuristic_not_hypothesis_test": True,
        "primary_uncertainty_feature": primary_name,
        "primary_cluster_cutoff_A": primary_cutoff,
        "failure_auroc": primary_auc,
        "failure_auroc_95ci": [
            float(primary_metric["bootstrap_95ci_low"]),
            float(primary_metric["bootstrap_95ci_high"]),
        ],
        "full_success_pct": full_success,
        "full_success_95ci_pct": [
            float(full_coverage_row["success_95ci_low_pct"]),
            float(full_coverage_row["success_95ci_high_pct"]),
        ],
        "most_certain_50pct_success_pct": certain_half_success,
        "most_certain_50pct_success_95ci_pct": [
            float(certain_half_row["success_95ci_low_pct"]),
            float(certain_half_row["success_95ci_high_pct"]),
        ],
        "most_certain_50pct_gain_pp": certain_half_gain,
        "most_certain_50pct_gain_95ci_pp": [
            float(certain_half_row["gain_95ci_low_pp"]),
            float(certain_half_row["gain_95ci_high_pp"]),
        ],
        "failure_rate_monotonic_across_uncertainty_quartiles": monotonic_quartiles,
        "screening_verdict": screening_verdict,
        "verdict_rule": (
            "PROMISING requires AUROC >= 0.65, certain-half gain >= 10 pp, and "
            "monotonic quartile risk; BORDERLINE requires AUROC >= 0.58 or gain >= 5 pp."
        ),
    }
    statistical_summary_seconds = time.perf_counter() - statistical_summary_started

    timing_columns = (
        "pose_read_seconds",
        "pairwise_rmsd_seconds",
        "clustering_seconds",
        "ensemble_feature_seconds",
    )
    timing_summary = {
        "timing_scope": (
            "Per-case ensemble feature timing includes pose loading, pairwise pose RMSD, "
            "and clustering, and excludes reference-label evaluation and bootstrap statistics."
        ),
        "host": platform.node(),
        "workers": args.workers,
        "cases_n": len(per_complex),
        "feature_evaluation_batch_wall_seconds": feature_evaluation_wall_seconds,
        "statistical_summary_seconds": statistical_summary_seconds,
    }
    for column in timing_columns:
        values = pd.to_numeric(per_complex[column], errors="coerce").dropna()
        timing_summary[column] = {
            "n": len(values),
            "median_seconds": float(values.median()),
            "q1_seconds": float(values.quantile(0.25)),
            "q3_seconds": float(values.quantile(0.75)),
            "mean_seconds": float(values.mean()),
            "sum_seconds": float(values.sum()),
        }
    timing_summary["audit_total_wall_seconds_before_output_write"] = (
        time.perf_counter() - audit_started
    )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_complex.to_csv(output / "per_complex_uncertainty.csv", index=False)
    metric_table.sort_values("failure_auroc", ascending=False).to_csv(
        output / "uncertainty_metrics.csv", index=False
    )
    risk_table.to_csv(output / "risk_coverage.csv", index=False)
    quartile_table.to_csv(output / "uncertainty_quartiles.csv", index=False)
    pd.DataFrame(failures).to_csv(output / "failures.csv", index=False)
    (output / "timing_summary.json").write_text(
        json.dumps(timing_summary, indent=2) + "\n"
    )
    wrong_counts.rename("observed_pose_count").rename_axis("complex_name").reset_index().to_csv(
        output / "unexpected_pose_counts.csv", index=False
    )

    best_metric = max(metric_rows, key=lambda row: row["failure_auroc"] if np.isfinite(row["failure_auroc"]) else -1)
    summary = {
        "pose_scores": str(score_path),
        "evaluation_manifest": str(manifest_path) if manifest_path is not None else None,
        "per_pose_rmsd": str(per_pose_rmsd_path) if per_pose_rmsd_path is not None else None,
        "pool": args.pool,
        "selection_score_column": args.score_column,
        "pairwise_rmsd_mode": args.pairwise_rmsd_mode,
        "cluster_cutoffs_A": cutoffs,
        "expected_poses_per_complex": args.expected_poses,
        "allowed_pose_counts_per_complex": sorted(allowed_pose_counts),
        "requested_complexes_n": len(tasks),
        "evaluated_complexes_n": len(per_complex),
        "failures_n": len(failures),
        "unexpected_pose_count_complexes_n": len(wrong_counts),
        "top1_success_cutoff_A": args.success_cutoff,
        "top1_success_count": int(per_complex["top1_success"].sum()),
        "top1_success_pct": 100.0 * float(per_complex["top1_success"].mean()),
        "best_exploratory_uncertainty_metric": best_metric,
        "prespecified_decision_support": decision_support,
        "timing": timing_summary,
        "uses_reference_for_features": False,
        "reference_usage": (
            "Only the already selected Top-1 pose is RMSD-labelled after feature calculation; "
            "the label is read from the supplied same-run per-pose table."
            if per_pose_rmsd_path is not None else
            "Only the already selected Top-1 pose is RMSD-labelled after feature calculation."
        ),
        "interpretation_warning": "Treat test-set results as exploratory; freeze the uncertainty feature on validation before confirmatory test reporting.",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nUNCERTAINTY METRICS")
    print(metric_table.sort_values("failure_auroc", ascending=False).to_string(index=False))
    print("\nPRESPECIFIED GO/NO-GO SCREEN")
    print(json.dumps(decision_support, indent=2))
    print("\nENSEMBLE FEATURE TIMING")
    print(json.dumps(timing_summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"POSE_ENSEMBLE_UNCERTAINTY_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
