#!/usr/bin/env python3
"""Evaluate a prespecified cross-seed uncertainty score without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "ok"}


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def choose_column(columns, candidates, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; found {list(columns)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ordered = labels[order]
    if not ordered.sum():
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def aurc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    failures = np.asarray(labels, dtype=float)[order]
    return float(np.mean(np.cumsum(failures) / np.arange(1, len(failures) + 1)))


def success_at_coverage(labels: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    retained_n = int(np.ceil(fraction * len(labels)))
    retained = np.argsort(np.asarray(scores, dtype=float), kind="stable")[:retained_n]
    return 100.0 * float((~np.asarray(labels, dtype=bool)[retained]).mean())


def wilson_interval(successes: int, denominator: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson interval on the percentage scale."""
    if denominator <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / denominator
    scale = 1.0 + z * z / denominator
    center = (p + z * z / (2.0 * denominator)) / scale
    half = z * math.sqrt(
        p * (1.0 - p) / denominator + z * z / (4.0 * denominator**2)
    ) / scale
    return 100.0 * (center - half), 100.0 * (center + half)


def tie_aware_risk_coverage(
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Retain complete uncertainty ties and report the resulting risk curve."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    rows = []
    for threshold in np.sort(np.unique(scores)):
        retained = scores <= threshold
        retained_n = int(retained.sum())
        failure_n = int(labels[retained].sum())
        success_n = retained_n - failure_n
        success_low, success_high = wilson_interval(success_n, retained_n)
        failure_low, failure_high = wilson_interval(failure_n, retained_n)
        rows.append({
            "uncertainty_threshold": float(threshold),
            "retained_n": retained_n,
            "coverage_pct": 100.0 * retained_n / len(labels),
            "failure_count": failure_n,
            "failure_pct": 100.0 * failure_n / retained_n,
            "failure_wilson_95ci_low_pct": failure_low,
            "failure_wilson_95ci_high_pct": failure_high,
            "success_count": success_n,
            "success_pct": 100.0 * success_n / retained_n,
            "success_wilson_95ci_low_pct": success_low,
            "success_wilson_95ci_high_pct": success_high,
        })
    return pd.DataFrame(rows)


def metrics(labels: np.ndarray, scores: np.ndarray, rmsd: np.ndarray) -> dict[str, float]:
    correlation_result = spearmanr(scores, rmsd, nan_policy="omit")
    correlation = (
        correlation_result.statistic
        if hasattr(correlation_result, "statistic")
        else correlation_result.correlation
    )
    return {
        "failure_auroc": binary_auc(labels, scores),
        "failure_auprc": average_precision(labels, scores),
        "success_at_50pct_coverage_pct": success_at_coverage(labels, scores, 0.5),
        "area_under_risk_coverage": aurc(labels, scores),
        "spearman_vs_top1_rmsd": float(correlation),
    }


def bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    rmsd: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {}
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        values = metrics(labels[indices], scores[indices], rmsd[indices])
        for name, value in values.items():
            if np.isfinite(value):
                samples.setdefault(name, []).append(value)
    return {
        name: tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))
        for name, values in samples.items()
    }


def paired_bootstrap_auc_delta(
    labels: np.ndarray,
    selected_scores: np.ndarray,
    reference_scores: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Return selected-minus-reference AUROC and its paired case-bootstrap CI."""
    labels = np.asarray(labels, dtype=bool)
    selected_scores = np.asarray(selected_scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    if not (
        len(labels) == len(selected_scores) == len(reference_scores)
        and np.isfinite(selected_scores).all()
        and np.isfinite(reference_scores).all()
    ):
        raise ValueError("Invalid paired AUROC inputs")
    estimate = binary_auc(labels, selected_scores) - binary_auc(labels, reference_scores)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), len(labels))
        selected_auc = binary_auc(labels[indices], selected_scores[indices])
        reference_auc = binary_auc(labels[indices], reference_scores[indices])
        delta = selected_auc - reference_auc
        if np.isfinite(delta):
            deltas.append(float(delta))
    if not deltas:
        raise RuntimeError("No valid paired bootstrap AUROC samples")
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(estimate), float(low), float(high), len(deltas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--endpoints", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--endpoint-rmsd-column", default="v2_rmsd_A")
    parser.add_argument("--endpoint-status-column", default="v2_scored")
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    feature_path = args.features.expanduser().resolve()
    endpoint_path = args.endpoints.expanduser().resolve()
    # Preserve PDB IDs such as 9E39, then convert numerical fields explicitly.
    features = pd.read_csv(feature_path, dtype=str)
    endpoints = pd.read_csv(endpoint_path, dtype=str)
    feature_id = choose_column(features.columns, ID_COLUMNS, "feature ID")
    endpoint_id = choose_column(endpoints.columns, ID_COLUMNS, "endpoint ID")
    features["complex_name"] = features[feature_id].map(normalize_id)
    endpoints["complex_name"] = endpoints[endpoint_id].map(normalize_id)
    if endpoints["complex_name"].duplicated().any():
        raise ValueError("Endpoint table contains duplicate cases")
    for column in (args.endpoint_rmsd_column, args.endpoint_status_column):
        if column not in endpoints:
            raise ValueError(f"Endpoint table lacks {column!r}")
    endpoints["endpoint_rmsd_A"] = pd.to_numeric(
        endpoints[args.endpoint_rmsd_column], errors="coerce"
    )
    endpoints["endpoint_scored"] = (
        endpoints[args.endpoint_status_column]
        .astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        & np.isfinite(endpoints["endpoint_rmsd_A"])
    )
    endpoints["endpoint_failure"] = ~(
        endpoints["endpoint_scored"]
        & endpoints["endpoint_rmsd_A"].lt(args.success_cutoff)
    )
    analysis = features.merge(
        endpoints[["complex_name", "endpoint_rmsd_A", "endpoint_scored", "endpoint_failure"]],
        on="complex_name", how="left", validate="many_to_one",
    )
    if analysis["endpoint_failure"].isna().any():
        raise ValueError("One or more feature cases lack endpoint rows")

    specifications = [(1, "single_pool_disagreement", "single_pool_uncertainty")]
    specifications.extend(
        (budget, "cross_seed_reported_pose_disagreement",
         "one_minus_reported_pose_cluster_fraction_at_2p0A")
        for budget in (2, 4, 8)
    )
    rows, quartiles, level_rows, risk_rows, per_case_rows = [], [], [], [], []
    paired_inputs: dict[int, dict[str, object]] = {}
    for budget, method, score_column in specifications:
        selected = analysis[
            pd.to_numeric(analysis["budget_runs"], errors="coerce").eq(budget)
        ].sort_values("complex_name", kind="stable")
        if selected.empty:
            raise ValueError(f"No rows for budget {budget}")
        scores = pd.to_numeric(selected[score_column], errors="coerce").to_numpy(float)
        rmsd = selected["endpoint_rmsd_A"].to_numpy(float)
        labels = selected["endpoint_failure"].to_numpy(bool)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite scores for budget {budget}: {score_column}")
        values = metrics(labels, scores, rmsd)
        intervals = bootstrap(labels, scores, rmsd, args.bootstrap_repeats, args.seed + budget)
        row = {
            "budget_runs": budget, "method": method, "score_column": score_column,
            "n": len(selected), "failure_count": int(labels.sum()),
            "success_count": int((~labels).sum()),
        }
        for name, value in values.items():
            row[name] = value
            row[f"{name}_95ci_low"], row[f"{name}_95ci_high"] = intervals[name]
        rows.append(row)
        paired_inputs[budget] = {
            "cases": selected["complex_name"].tolist(),
            "labels": labels,
            "scores": scores,
            "method": method,
        }

        for case, score, label, case_rmsd, scored in zip(
            selected["complex_name"],
            scores,
            labels,
            rmsd,
            selected["endpoint_scored"].to_numpy(bool),
        ):
            per_case_rows.append({
                "complex_name": case,
                "budget_runs": budget,
                "method": method,
                "score_column": score_column,
                "uncertainty": float(score),
                "endpoint_failure": bool(label),
                "endpoint_scored": bool(scored),
                "endpoint_rmsd_A": float(case_rmsd),
            })

        for level in np.sort(np.unique(scores)):
            at_level = scores == level
            level_n = int(at_level.sum())
            failure_n = int(labels[at_level].sum())
            low, high = wilson_interval(failure_n, level_n)
            level_rows.append({
                "budget_runs": budget,
                "method": method,
                "uncertainty_level": float(level),
                "n": level_n,
                "failure_count": failure_n,
                "failure_pct": 100.0 * failure_n / level_n,
                "failure_wilson_95ci_low_pct": low,
                "failure_wilson_95ci_high_pct": high,
                "median_top1_rmsd_A": float(np.median(rmsd[at_level])),
            })

        budget_risk = tie_aware_risk_coverage(labels, scores)
        budget_risk.insert(0, "method", method)
        budget_risk.insert(0, "budget_runs", budget)
        risk_rows.extend(budget_risk.to_dict("records"))

        ranked = selected.assign(_score=scores).sort_values("_score", kind="stable")
        ranked["uncertainty_quartile"] = pd.qcut(
            np.arange(len(ranked)), 4, labels=["Q1_most_certain", "Q2", "Q3", "Q4_most_uncertain"]
        )
        for quartile, group in ranked.groupby("uncertainty_quartile", observed=True):
            quartiles.append({
                "budget_runs": budget, "method": method,
                "uncertainty_quartile": str(quartile), "n": len(group),
                "failure_count": int(group["endpoint_failure"].sum()),
                "failure_pct": 100.0 * float(group["endpoint_failure"].mean()),
                "median_uncertainty": float(group["_score"].median()),
                "median_top1_rmsd_A": float(group["endpoint_rmsd_A"].median()),
            })

    comparison_specs = (
        (2, 1),
        (4, 1),
        (8, 1),
        (4, 2),
        (8, 4),
    )
    delta_rows = []
    for comparison_index, (selected_budget, reference_budget) in enumerate(
        comparison_specs
    ):
        selected_input = paired_inputs[selected_budget]
        reference_input = paired_inputs[reference_budget]
        if selected_input["cases"] != reference_input["cases"]:
            raise ValueError(
                f"Case order differs for budgets {selected_budget} and {reference_budget}"
            )
        selected_labels = np.asarray(selected_input["labels"], dtype=bool)
        reference_labels = np.asarray(reference_input["labels"], dtype=bool)
        if not np.array_equal(selected_labels, reference_labels):
            raise ValueError(
                f"Endpoint labels differ for budgets {selected_budget} and {reference_budget}"
            )
        selected_scores = np.asarray(selected_input["scores"], dtype=float)
        reference_scores = np.asarray(reference_input["scores"], dtype=float)
        estimate, low, high, valid_repeats = paired_bootstrap_auc_delta(
            selected_labels,
            selected_scores,
            reference_scores,
            args.bootstrap_repeats,
            args.seed + 1000 + comparison_index,
        )
        delta_rows.append({
            "selected_budget_runs": selected_budget,
            "selected_method": selected_input["method"],
            "reference_budget_runs": reference_budget,
            "reference_method": reference_input["method"],
            "n": len(selected_labels),
            "selected_failure_auroc": binary_auc(selected_labels, selected_scores),
            "reference_failure_auroc": binary_auc(selected_labels, reference_scores),
            "delta_selected_minus_reference": estimate,
            "paired_bootstrap_95ci_low": low,
            "paired_bootstrap_95ci_high": high,
            "bootstrap_repeats_requested": args.bootstrap_repeats,
            "bootstrap_repeats_valid": valid_repeats,
        })

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "direct_uncertainty_budget_metrics.csv", index=False)
    pd.DataFrame(quartiles).to_csv(output / "uncertainty_quartiles.csv", index=False)
    pd.DataFrame(level_rows).to_csv(output / "uncertainty_levels.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "risk_coverage_by_budget.csv", index=False)
    pd.DataFrame(per_case_rows).to_csv(output / "per_case_scores.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(output / "paired_auroc_deltas.csv", index=False)
    summary = {
        "analysis_role": "prespecified_direct_test_without_model_fitting",
        "reported_pose": "first-replicate V2 selected Top-1",
        "primary_score": "1 - fraction of runs in the 2 A cluster containing the reported pose",
        "primary_budget_runs": 8,
        "positive_class": f"reported Top-1 RMSD >= {args.success_cutoff:g} A",
        "calibration_metrics_omitted": True,
        "calibration_reason": "The prespecified disagreement is a ranking score, not a fitted probability.",
        "features": {"path": str(feature_path), "sha256": sha256(feature_path)},
        "endpoints": {"path": str(endpoint_path), "sha256": sha256(endpoint_path)},
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nUNCERTAINTY QUARTILES")
    print(pd.DataFrame(quartiles).to_string(index=False))
    print("\nPAIRED AUROC DELTAS")
    print(pd.DataFrame(delta_rows).to_string(index=False))
    print(f"MULTIRUN_UNCERTAINTY_DIRECT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
