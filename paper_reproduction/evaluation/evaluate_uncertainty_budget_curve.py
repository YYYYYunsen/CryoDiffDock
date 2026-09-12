#!/usr/bin/env python3
"""Fit uncertainty models on validation and evaluate a frozen budget curve.

The docking endpoint is fixed to the first replicate's reported Top-1 pose.
Additional replicates may change only the estimated probability that this pose
fails; they do not change the reported pose in the primary analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "ok"}
METHOD_FEATURES = {
    "top1_confidence": ("replicate1_confidence",),
    "single_pool_disagreement": ("single_pool_uncertainty",),
    "multirun_consistency": (
        "one_minus_reported_pose_cluster_fraction_at_2p0A",
    ),
    "consistency_plus_density_stability": (
        "one_minus_reported_pose_cluster_fraction_at_2p0A",
        "selected_density_z_mean",
        "selected_density_z_sd",
    ),
    "integrated_risk": (
        "replicate1_confidence",
        "single_pool_uncertainty",
        "one_minus_reported_pose_cluster_fraction_at_2p0A",
        "selected_density_z_mean",
        "selected_density_z_sd",
    ),
}
MULTIRUN_METHODS = {
    "multirun_consistency",
    "consistency_plus_density_stability",
    "integrated_risk",
}
SINGLE_RUN_METHODS = {"top1_confidence", "single_pool_disagreement"}


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


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(
    features: np.ndarray,
    outcome: np.ndarray,
    l2_penalty: float,
) -> dict:
    features = np.asarray(features, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    mean = features.mean(axis=0)
    scale = features.std(axis=0, ddof=0)
    scale[scale <= 1e-12] = 1.0
    standardized = (features - mean) / scale
    design = np.column_stack([np.ones(len(features)), standardized])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = sigmoid(design @ beta)
        eps = np.finfo(float).eps
        loss = -np.sum(
            outcome * np.log(np.clip(probabilities, eps, 1.0))
            + (1.0 - outcome) * np.log(np.clip(1.0 - probabilities, eps, 1.0))
        )
        loss += 0.5 * l2_penalty * float(np.dot(beta[1:], beta[1:]))
        gradient = design.T @ (probabilities - outcome)
        gradient[1:] += l2_penalty * beta[1:]
        return float(loss), gradient

    prevalence = np.clip(outcome.mean(), 1e-6, 1.0 - 1e-6)
    initial = np.zeros(design.shape[1], dtype=float)
    initial[0] = math.log(prevalence / (1.0 - prevalence))
    fitted = minimize(
        lambda beta: objective(beta)[0],
        initial,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
    )
    if not fitted.success:
        raise RuntimeError(f"Logistic fit failed: {fitted.message}")
    return {
        "mean": mean,
        "scale": scale,
        "beta": np.asarray(fitted.x, dtype=float),
        "l2_penalty": float(l2_penalty),
    }


def predict_logistic(model: dict, features: np.ndarray) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=float) - np.asarray(model["mean"], dtype=float)
    ) / np.asarray(model["scale"], dtype=float)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    return sigmoid(design @ np.asarray(model["beta"], dtype=float))


def stratified_folds(outcome: np.ndarray, folds: int, seed: int) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=bool)
    if folds < 2 or folds > min(int(outcome.sum()), int((~outcome).sum())):
        raise ValueError("cross-validation folds exceed the smallest outcome class")
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(outcome), dtype=int)
    for label in (False, True):
        indices = np.flatnonzero(outcome == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignments[index] = position % folds
    return assignments


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ordered = labels[order]
    positives = int(ordered.sum())
    if not positives:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].mean())


def area_under_risk_coverage(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(np.asarray(scores, dtype=float), kind="stable")
    ordered_failures = np.asarray(labels, dtype=float)[order]
    risk = np.cumsum(ordered_failures) / np.arange(1, len(ordered_failures) + 1)
    return float(risk.mean())


def success_at_half_coverage(labels: np.ndarray, scores: np.ndarray) -> float:
    retained_n = int(math.ceil(0.5 * len(labels)))
    retained = np.argsort(np.asarray(scores, dtype=float), kind="stable")[:retained_n]
    return 100.0 * float((~np.asarray(labels, dtype=bool)[retained]).mean())


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    order = np.argsort(np.asarray(probabilities, dtype=float), kind="stable")
    total = len(order)
    error = 0.0
    for indices in np.array_split(order, min(bins, total)):
        if not len(indices):
            continue
        observed = np.asarray(labels, dtype=float)[indices].mean()
        predicted = np.asarray(probabilities, dtype=float)[indices].mean()
        error += len(indices) / total * abs(float(observed - predicted))
    return float(error)


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, calibration_bins: int
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "failure_auroc": binary_auc(labels, probabilities),
        "failure_auprc": average_precision(labels, probabilities),
        "success_at_50pct_coverage_pct": success_at_half_coverage(labels, probabilities),
        "area_under_risk_coverage": area_under_risk_coverage(labels, probabilities),
        "expected_calibration_error": expected_calibration_error(
            labels, probabilities, calibration_bins
        ),
        "brier_score": float(np.mean((probabilities - labels.astype(float)) ** 2)),
    }


def bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    repeats: int,
    seed: int,
    calibration_bins: int,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled = calculate_metrics(
            labels[indices], probabilities[indices], calibration_bins
        )
        for name, value in sampled.items():
            if np.isfinite(value):
                values.setdefault(name, []).append(value)
    return {
        name: tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))
        for name, samples in values.items()
    }


def bootstrap_metric_deltas(
    labels: np.ndarray,
    selected_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    repeats: int,
    seed: int,
    calibration_bins: int,
) -> dict[str, tuple[float, float]]:
    """Paired bootstrap CIs for selected-minus-baseline metric differences."""
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(repeats):
        indices = rng.integers(0, len(labels), size=len(labels))
        selected = calculate_metrics(
            labels[indices], selected_probabilities[indices], calibration_bins
        )
        baseline = calculate_metrics(
            labels[indices], baseline_probabilities[indices], calibration_bins
        )
        for name in selected:
            difference = selected[name] - baseline[name]
            if np.isfinite(difference):
                values.setdefault(name, []).append(float(difference))
    return {
        name: tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))
        for name, samples in values.items()
    }


def load_analysis_table(
    feature_path: Path,
    endpoint_path: Path,
    rmsd_column: str,
    status_column: str | None,
    success_cutoff: float,
) -> pd.DataFrame:
    # Preserve PDB IDs such as 9E39, then convert numerical fields explicitly.
    features = pd.read_csv(feature_path, dtype=str)
    endpoint = pd.read_csv(endpoint_path, dtype=str)
    feature_id = choose_column(features.columns, ID_COLUMNS, "feature ID")
    endpoint_id = choose_column(endpoint.columns, ID_COLUMNS, "endpoint ID")
    features = features.copy()
    endpoint = endpoint.copy()
    features["complex_name"] = features[feature_id].map(normalize_id)
    endpoint["complex_name"] = endpoint[endpoint_id].map(normalize_id)
    if endpoint["complex_name"].duplicated().any():
        raise ValueError(f"Endpoint table contains duplicate cases: {endpoint_path}")
    if rmsd_column not in endpoint.columns:
        raise ValueError(f"Endpoint table lacks {rmsd_column!r}")
    endpoint["endpoint_rmsd_A"] = pd.to_numeric(endpoint[rmsd_column], errors="coerce")
    if status_column:
        if status_column not in endpoint.columns:
            raise ValueError(f"Endpoint table lacks {status_column!r}")
        scored = (
            endpoint[status_column].astype(str).str.strip().str.lower().isin(TRUE_VALUES)
        )
    else:
        scored = np.isfinite(endpoint["endpoint_rmsd_A"])
    endpoint["endpoint_scored"] = scored & np.isfinite(endpoint["endpoint_rmsd_A"])
    endpoint["endpoint_failure"] = ~(
        endpoint["endpoint_scored"] & (endpoint["endpoint_rmsd_A"] < success_cutoff)
    )
    merged = features.merge(
        endpoint[[
            "complex_name", "endpoint_rmsd_A", "endpoint_scored", "endpoint_failure"
        ]],
        on="complex_name", how="left", validate="many_to_one",
    )
    if merged["endpoint_failure"].isna().any():
        missing = sorted(
            merged.loc[merged["endpoint_failure"].isna(), "complex_name"].unique()
        )
        raise ValueError(f"Endpoint rows are missing for: {missing[:20]}")
    return merged


def feature_matrix(table: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"Feature table lacks columns: {missing}")
    matrix = table[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Features contain missing/non-finite values: {columns}")
    return matrix


def serialize_model(model: dict, columns: tuple[str, ...]) -> dict:
    return {
        "feature_columns": list(columns),
        "mean": np.asarray(model["mean"]).tolist(),
        "scale": np.asarray(model["scale"]).tolist(),
        "beta": np.asarray(model["beta"]).tolist(),
        "l2_penalty": model["l2_penalty"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-features", required=True, type=Path)
    parser.add_argument("--validation-endpoints", required=True, type=Path)
    parser.add_argument("--test-features", type=Path)
    parser.add_argument("--test-endpoints", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--endpoint-rmsd-column", default="v2_rmsd_A")
    parser.add_argument("--endpoint-status-column", default=None)
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2-penalty", type=float, default=1.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if (args.test_features is None) != (args.test_endpoints is None):
        raise ValueError("Supply both --test-features and --test-endpoints, or neither")
    if args.success_cutoff <= 0 or args.l2_penalty < 0:
        raise ValueError("success-cutoff must be positive and l2-penalty non-negative")
    if args.bootstrap_repeats <= 0 or args.calibration_bins <= 0:
        raise ValueError("bootstrap-repeats and calibration-bins must be positive")

    validation_features = args.validation_features.expanduser().resolve()
    validation_endpoints = args.validation_endpoints.expanduser().resolve()
    validation = load_analysis_table(
        validation_features,
        validation_endpoints,
        args.endpoint_rmsd_column,
        args.endpoint_status_column,
        args.success_cutoff,
    )
    test = None
    if args.test_features is not None:
        test = load_analysis_table(
            args.test_features.expanduser().resolve(),
            args.test_endpoints.expanduser().resolve(),
            args.endpoint_rmsd_column,
            args.endpoint_status_column,
            args.success_cutoff,
        )

    budgets = [
        int(value) for value in sorted(
            pd.to_numeric(validation["budget_runs"], errors="raise").astype(int).unique()
        )
    ]
    if test is not None:
        test_budgets = [
            int(value) for value in sorted(
                pd.to_numeric(test["budget_runs"], errors="raise").astype(int).unique()
            )
        ]
        if budgets != test_budgets:
            raise ValueError(f"Validation/test budget mismatch: {budgets} vs {test_budgets}")

    metric_rows = []
    prediction_rows = []
    frozen_models: dict[str, dict] = {}
    for budget in budgets:
        validation_budget = validation[
            pd.to_numeric(validation["budget_runs"]).astype(int).eq(budget)
        ].sort_values("complex_name", kind="stable").reset_index(drop=True)
        validation_labels = validation_budget["endpoint_failure"].to_numpy(bool)
        assignments = stratified_folds(validation_labels, args.folds, args.seed + budget)
        for method, columns in METHOD_FEATURES.items():
            if budget == 1 and method in MULTIRUN_METHODS:
                continue
            if budget > 1 and method in SINGLE_RUN_METHODS:
                continue
            validation_x = feature_matrix(validation_budget, columns)
            oof = np.full(len(validation_budget), np.nan, dtype=float)
            for fold in range(args.folds):
                held_out = assignments == fold
                model = fit_logistic(
                    validation_x[~held_out], validation_labels[~held_out], args.l2_penalty
                )
                oof[held_out] = predict_logistic(model, validation_x[held_out])
            final_model = fit_logistic(validation_x, validation_labels, args.l2_penalty)
            frozen_models[f"budget{budget}:{method}"] = serialize_model(final_model, columns)

            datasets = [("validation_oof", validation_budget, validation_labels, oof)]
            if test is not None:
                test_budget = test[
                    pd.to_numeric(test["budget_runs"]).astype(int).eq(budget)
                ].sort_values("complex_name", kind="stable").reset_index(drop=True)
                test_labels = test_budget["endpoint_failure"].to_numpy(bool)
                test_x = feature_matrix(test_budget, columns)
                datasets.append((
                    "test_frozen", test_budget, test_labels,
                    predict_logistic(final_model, test_x),
                ))

            for dataset_name, dataset_table, labels, probabilities in datasets:
                metrics = calculate_metrics(labels, probabilities, args.calibration_bins)
                intervals = bootstrap_metrics(
                    labels, probabilities, args.bootstrap_repeats,
                    args.seed + budget * 100 + len(metric_rows), args.calibration_bins,
                )
                row = {
                    "dataset": dataset_name,
                    "budget_runs": budget,
                    "method": method,
                    "n": len(labels),
                    "failure_count": int(labels.sum()),
                    "failure_prevalence": float(labels.mean()),
                }
                for metric_name, value in metrics.items():
                    row[metric_name] = value
                    low, high = intervals[metric_name]
                    row[f"{metric_name}_95ci_low"] = low
                    row[f"{metric_name}_95ci_high"] = high
                metric_rows.append(row)
                for case, label, probability in zip(
                    dataset_table["complex_name"], labels, probabilities
                ):
                    prediction_rows.append({
                        "dataset": dataset_name,
                        "budget_runs": budget,
                        "method": method,
                        "complex_name": case,
                        "endpoint_failure": bool(label),
                        "predicted_failure_probability": float(probability),
                    })

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_table = pd.DataFrame(metric_rows)
    prediction_table = pd.DataFrame(prediction_rows)
    metric_table.to_csv(output / "uncertainty_budget_metrics.csv", index=False)
    prediction_table.to_csv(output / "per_case_predictions.csv", index=False)

    delta_rows = []
    for dataset_name in sorted(prediction_table["dataset"].unique()):
        baseline = prediction_table[
            prediction_table["dataset"].eq(dataset_name)
            & prediction_table["method"].eq("single_pool_disagreement")
            & prediction_table["budget_runs"].eq(1)
        ].sort_values("complex_name", kind="stable")
        if baseline.empty:
            continue
        baseline_cases = baseline["complex_name"].tolist()
        baseline_labels = baseline["endpoint_failure"].to_numpy(bool)
        baseline_probabilities = baseline["predicted_failure_probability"].to_numpy(float)
        for (budget, method), selected in prediction_table[
            prediction_table["dataset"].eq(dataset_name)
            & ~prediction_table["method"].eq("single_pool_disagreement")
        ].groupby(["budget_runs", "method"], sort=True):
            selected = selected.sort_values("complex_name", kind="stable")
            if selected["complex_name"].tolist() != baseline_cases:
                raise ValueError(
                    f"Cannot pair {dataset_name} budget={budget} method={method} "
                    "to the single-pool baseline"
                )
            labels = selected["endpoint_failure"].to_numpy(bool)
            if not np.array_equal(labels, baseline_labels):
                raise ValueError("Paired comparison endpoint labels differ")
            probabilities = selected["predicted_failure_probability"].to_numpy(float)
            selected_metrics = calculate_metrics(labels, probabilities, args.calibration_bins)
            baseline_metrics = calculate_metrics(
                labels, baseline_probabilities, args.calibration_bins
            )
            intervals = bootstrap_metric_deltas(
                labels, probabilities, baseline_probabilities,
                args.bootstrap_repeats,
                args.seed + int(budget) * 1000 + len(delta_rows),
                args.calibration_bins,
            )
            for metric_name in selected_metrics:
                low, high = intervals[metric_name]
                delta_rows.append({
                    "dataset": dataset_name,
                    "budget_runs": int(budget),
                    "method": method,
                    "reference_method": "single_pool_disagreement",
                    "metric": metric_name,
                    "selected_value": selected_metrics[metric_name],
                    "reference_value": baseline_metrics[metric_name],
                    "delta_selected_minus_reference": (
                        selected_metrics[metric_name] - baseline_metrics[metric_name]
                    ),
                    "paired_bootstrap_95ci_low": low,
                    "paired_bootstrap_95ci_high": high,
                })
    pd.DataFrame(delta_rows).to_csv(
        output / "paired_deltas_vs_single_pool.csv", index=False
    )
    model_output = {
        "analysis_role": "validation_fitted_test_frozen_uncertainty_budget_models",
        "reported_pose_policy": "first-replicate V2 selected Top-1 is fixed at every budget",
        "positive_class": f"first-replicate Top-1 RMSD >= {args.success_cutoff:g} A",
        "nested_budgets": budgets,
        "models": frozen_models,
        "validation": {
            "features": str(validation_features),
            "features_sha256": sha256(validation_features),
            "endpoints": str(validation_endpoints),
            "endpoints_sha256": sha256(validation_endpoints),
            "performance_estimate": f"{args.folds}-fold stratified out-of-fold",
        },
        "test": (
            {
                "features": str(args.test_features.expanduser().resolve()),
                "features_sha256": sha256(args.test_features.expanduser().resolve()),
                "endpoints": str(args.test_endpoints.expanduser().resolve()),
                "endpoints_sha256": sha256(args.test_endpoints.expanduser().resolve()),
                "model_selection_or_refitting_on_test": False,
            }
            if test is not None else None
        ),
        "l2_penalty": args.l2_penalty,
        "calibration": "validation-fitted penalized logistic probabilities",
        "calibration_bins": args.calibration_bins,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "uses_reference_for_uncertainty_features": False,
    }
    (output / "frozen_models_and_protocol.json").write_text(
        json.dumps(model_output, indent=2) + "\n"
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(f"UNCERTAINTY_BUDGET_CURVE_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
