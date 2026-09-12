#!/usr/bin/env python3
"""Test whether pose-ensemble uncertainty adds signal beyond case covariates.

The primary model treats Top-1 failure as the binary outcome and includes
continuous, standardized map resolution, ligand heavy-atom count, and ligand
rotatable-bond count.  The full model adds the prespecified pose-ensemble
uncertainty feature.  This script uses NumPy/SciPy only; statsmodels is not
required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, norm, rankdata


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex_id", "id")


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


def unique_id_table(path: Path, label: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    id_column = choose_column(table.columns, ID_COLUMNS, f"{label} ID")
    table = table.copy()
    table["complex_name"] = table[id_column].map(normalize_id)
    duplicates = sorted(
        table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate IDs: {duplicates[:20]}")
    return table


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
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def standardize(values: pd.Series, label: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} contains missing or non-finite values")
    scale = float(np.std(numeric, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label} has no usable variation")
    return (numeric - float(np.mean(numeric))) / scale


def fit_logistic(design: np.ndarray, outcome: np.ndarray) -> dict[str, object]:
    design = np.asarray(design, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if design.ndim != 2 or len(design) != len(outcome):
        raise ValueError("Invalid logistic-regression design")
    if not np.isfinite(design).all() or not np.isfinite(outcome).all():
        raise ValueError("Logistic-regression inputs must be finite")

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        return float(np.sum(np.logaddexp(0.0, linear) - outcome * linear))

    def gradient(beta: np.ndarray) -> np.ndarray:
        return design.T @ (expit(design @ beta) - outcome)

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    beta = np.asarray(result.x, dtype=float)
    probabilities = expit(design @ beta)
    weights = probabilities * (1.0 - probabilities)
    information = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "beta": beta,
        "standard_errors": standard_errors,
        "probabilities": probabilities,
        "log_likelihood": -objective(beta),
        "converged": bool(result.success or np.linalg.norm(gradient(beta)) < 1e-6),
        "optimizer_message": str(result.message),
    }


def make_tertiles(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ordered_rank = numeric.rank(method="first")
    return pd.qcut(ordered_rank, 3, labels=("low", "middle", "high"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-table", required=True, type=Path)
    parser.add_argument("--chemistry-table", required=True, type=Path)
    parser.add_argument("--resolution-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=174)
    parser.add_argument(
        "--largest-cluster-column",
        default="largest_cluster_fraction_at_2p0A",
    )
    parser.add_argument("--failure-column", default="top1_failure")
    parser.add_argument("--resolution-column", default="resolution_A")
    parser.add_argument("--heavy-atom-column", default="n_heavy_atoms")
    parser.add_argument("--rotatable-bond-column", default="n_rotatable_bonds")
    args = parser.parse_args()

    paths = {
        "uncertainty_table": args.uncertainty_table.expanduser().resolve(),
        "chemistry_table": args.chemistry_table.expanduser().resolve(),
        "resolution_table": args.resolution_table.expanduser().resolve(),
    }
    uncertainty = unique_id_table(paths["uncertainty_table"], "uncertainty table")
    chemistry = unique_id_table(paths["chemistry_table"], "chemistry table")
    resolution = unique_id_table(paths["resolution_table"], "resolution table")

    required_uncertainty = {args.largest_cluster_column, args.failure_column}
    missing = sorted(required_uncertainty - set(uncertainty.columns))
    if missing:
        raise ValueError(f"Uncertainty table missing columns: {missing}")
    for table, column, label in (
        (chemistry, args.heavy_atom_column, "heavy-atom"),
        (chemistry, args.rotatable_bond_column, "rotatable-bond"),
        (resolution, args.resolution_column, "resolution"),
    ):
        if column not in table.columns:
            raise ValueError(f"{label} table lacks {column!r}")

    uncertainty_ids = set(uncertainty["complex_name"])
    audit = {
        "uncertainty_cases_n": len(uncertainty_ids),
        "missing_from_chemistry": sorted(
            uncertainty_ids - set(chemistry["complex_name"])
        ),
        "missing_from_resolution": sorted(
            uncertainty_ids - set(resolution["complex_name"])
        ),
        "extra_chemistry_rows": sorted(
            set(chemistry["complex_name"]) - uncertainty_ids
        ),
        "extra_resolution_rows": sorted(
            set(resolution["complex_name"]) - uncertainty_ids
        ),
    }
    if len(uncertainty_ids) != args.expected_n:
        raise ValueError(
            f"Expected {args.expected_n} uncertainty cases, found {len(uncertainty_ids)}"
        )
    if audit["missing_from_chemistry"] or audit["missing_from_resolution"]:
        raise ValueError(f"Covariate coverage is incomplete: {audit}")

    selected = uncertainty[[
        "complex_name", args.failure_column, args.largest_cluster_column
    ]].copy()
    selected["uncertainty"] = 1.0 - pd.to_numeric(
        selected[args.largest_cluster_column], errors="coerce"
    )
    selected["top1_failure"] = selected[args.failure_column].astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    if selected["top1_failure"].isna().any():
        raise ValueError(f"Cannot parse {args.failure_column!r} as Boolean")

    chemistry_selected = chemistry[[
        "complex_name", args.heavy_atom_column, args.rotatable_bond_column
    ]].rename(columns={
        args.heavy_atom_column: "heavy_atoms",
        args.rotatable_bond_column: "rotatable_bonds",
    })
    resolution_selected = resolution[[
        "complex_name", args.resolution_column
    ]].rename(columns={args.resolution_column: "resolution_A"})
    analysis = selected.merge(
        chemistry_selected, on="complex_name", how="left", validate="one_to_one"
    ).merge(
        resolution_selected, on="complex_name", how="left", validate="one_to_one"
    ).sort_values("complex_name").reset_index(drop=True)

    continuous = ("uncertainty", "resolution_A", "heavy_atoms", "rotatable_bonds")
    for column in continuous:
        analysis[column] = pd.to_numeric(analysis[column], errors="coerce")
        analysis[f"{column}_z"] = standardize(analysis[column], column)

    outcome = analysis["top1_failure"].to_numpy(float)
    intercept = np.ones((len(analysis), 1), dtype=float)
    base_columns = ("resolution_A_z", "heavy_atoms_z", "rotatable_bonds_z")
    base_design = np.column_stack(
        [intercept, *(analysis[column].to_numpy(float) for column in base_columns)]
    )
    full_design = np.column_stack(
        [base_design, analysis["uncertainty_z"].to_numpy(float)]
    )
    base = fit_logistic(base_design, outcome)
    full = fit_logistic(full_design, outcome)

    uncertainty_index = full_design.shape[1] - 1
    coefficient = float(full["beta"][uncertainty_index])
    standard_error = float(full["standard_errors"][uncertainty_index])
    z_value = coefficient / standard_error
    p_value = float(2.0 * norm.sf(abs(z_value)))
    likelihood_ratio = max(
        0.0, 2.0 * (float(full["log_likelihood"]) - float(base["log_likelihood"]))
    )

    coefficient_rows = []
    names = ("intercept", "resolution_A_z", "heavy_atoms_z", "rotatable_bonds_z", "uncertainty_z")
    for index, name in enumerate(names):
        beta = float(full["beta"][index])
        se = float(full["standard_errors"][index])
        z = beta / se
        coefficient_rows.append({
            "term": name,
            "log_odds_coefficient": beta,
            "standard_error": se,
            "odds_ratio": float(np.exp(beta)),
            "odds_ratio_95ci_low": float(np.exp(beta - 1.96 * se)),
            "odds_ratio_95ci_high": float(np.exp(beta + 1.96 * se)),
            "wald_z": z,
            "wald_pvalue": float(2.0 * norm.sf(abs(z))),
        })

    strata_rows = []
    labels = analysis["top1_failure"].to_numpy(bool)
    for covariate in ("resolution_A", "heavy_atoms", "rotatable_bonds"):
        strata = make_tertiles(analysis[covariate])
        analysis[f"{covariate}_tertile"] = strata.astype(str)
        for stratum in ("low", "middle", "high"):
            mask = strata.astype(str).eq(stratum).to_numpy()
            strata_rows.append({
                "stratifying_covariate": covariate,
                "stratum": stratum,
                "n": int(mask.sum()),
                "failure_count": int(labels[mask].sum()),
                "failure_pct": 100.0 * float(labels[mask].mean()),
                "uncertainty_failure_auroc": binary_auc(
                    labels[mask], analysis.loc[mask, "uncertainty"].to_numpy(float)
                ),
                "median_covariate": float(analysis.loc[mask, covariate].median()),
                "median_uncertainty": float(analysis.loc[mask, "uncertainty"].median()),
            })

    summary = {
        "analysis_role": "secondary_post_confirmatory_covariate_audit",
        "n": len(analysis),
        "failure_count": int(outcome.sum()),
        "outcome": "Top-1 RMSD >= 2 A",
        "primary_uncertainty": "1 - largest cluster fraction at 2 A",
        "covariates": [
            "continuous map resolution in A",
            "ligand heavy-atom count",
            "ligand rotatable-bond count",
        ],
        "standardization": "continuous variables standardized to one population SD",
        "uncertainty_adjusted_log_odds_per_sd": coefficient,
        "uncertainty_adjusted_odds_ratio_per_sd": float(np.exp(coefficient)),
        "uncertainty_adjusted_or_95ci": [
            float(np.exp(coefficient - 1.96 * standard_error)),
            float(np.exp(coefficient + 1.96 * standard_error)),
        ],
        "uncertainty_wald_z": z_value,
        "uncertainty_wald_pvalue": p_value,
        "nested_model_likelihood_ratio": likelihood_ratio,
        "nested_model_lr_pvalue": float(chi2.sf(likelihood_ratio, df=1)),
        "covariate_only_auc_descriptive": binary_auc(
            labels, np.asarray(base["probabilities"], dtype=float)
        ),
        "covariate_plus_uncertainty_auc_descriptive": binary_auc(
            labels, np.asarray(full["probabilities"], dtype=float)
        ),
        "base_model_converged": bool(base["converged"]),
        "full_model_converged": bool(full["converged"]),
        "warning": (
            "AUC values are in-sample and descriptive. The adjusted odds ratio and "
            "nested likelihood-ratio test address conditional association; this "
            "secondary audit must not be used to retune the uncertainty feature."
        ),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(output / "merged_complete_cases.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output / "logistic_coefficients.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(
        output / "uncertainty_within_covariate_strata.csv", index=False
    )
    (output / "coverage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("COVARIATE COVERAGE")
    print(json.dumps(audit, indent=2))
    print("\nADJUSTED UNCERTAINTY RESULT")
    print(json.dumps(summary, indent=2))
    print("\nFULL MODEL COEFFICIENTS")
    print(pd.DataFrame(coefficient_rows).to_string(index=False))
    print("\nSTRATIFIED RESULTS")
    print(pd.DataFrame(strata_rows).to_string(index=False))
    print(f"\nUNCERTAINTY_COVARIATE_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
