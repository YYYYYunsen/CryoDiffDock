#!/usr/bin/env python3
"""Build reference-free uncertainty features across independent Local V2 runs.

The first replicate is the reported prediction.  At nested budgets 1, 2, 4,
and 8, later replicates provide evidence about whether that reported pose is
reproducible.  This script never reads an RMSD or reference ligand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.audit_pose_ensemble_uncertainty import (
    cluster_features,
    ordered_pairwise_rmsd,
    symmetry_pairwise_rmsd,
)
from evaluation.evaluate_ligand_accuracy import read_heavy_molecule


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
NUMERIC_COLUMNS = (
    "confidence",
    "final_selection_score",
    "multiscale_ncc",
    "expanded_density_z",
)


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


def parse_replicate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("replicate must be NAME=/path/to/replicate")
    return name.strip(), Path(path).expanduser().resolve()


def read_ids(path: Path) -> list[str]:
    ids = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return ids


def locate_selected_root(root: Path) -> Path:
    candidates = (
        root / "03_v2_top1" / "v2_selected",
        root / "v2_selected",
        root,
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("index*___*/ranking.csv")):
            return candidate
    raise FileNotFoundError(f"No V2 Top-1 ranking directories below {root}")


def load_replicate(name: str, root: Path, cohort: set[str]) -> pd.DataFrame:
    selected_root = locate_selected_root(root)
    rows = []
    for ranking_path in sorted(selected_root.glob("index*___*/ranking.csv")):
        # PDB IDs such as 9E39 are valid identifiers but pandas otherwise
        # interprets them as scientific notation (9E+39).  Preserve the ID
        # column as text while allowing score columns to be converted below.
        ranking = pd.read_csv(ranking_path, dtype={"complex_name": str})
        if "materialized_rank" in ranking.columns:
            ranking = ranking[
                pd.to_numeric(ranking["materialized_rank"], errors="coerce").eq(1)
            ]
        if len(ranking) != 1:
            raise ValueError(f"Expected exactly one selected row: {ranking_path}")
        row = ranking.iloc[0]
        complex_name = normalize_id(row["complex_name"])
        pose_path = Path(str(row["materialized_path"])).expanduser().resolve()
        if not pose_path.is_file():
            raise FileNotFoundError(f"Selected pose does not exist: {pose_path}")
        record = {
            "complex_name": complex_name,
            "replicate": name,
            "selected_pose_path": str(pose_path),
            "ranking_path": str(ranking_path.resolve()),
        }
        for column in NUMERIC_COLUMNS:
            record[column] = pd.to_numeric(row.get(column, np.nan), errors="coerce")
        rows.append(record)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"No selected poses found for replicate {name}")
    duplicates = sorted(
        table.loc[table["complex_name"].duplicated(False), "complex_name"].unique()
    )
    if duplicates:
        raise ValueError(f"Replicate {name} has duplicate cases: {duplicates[:20]}")
    found = set(table["complex_name"])
    missing = sorted(cohort - found)
    extra = sorted(found - cohort)
    if missing or extra:
        raise ValueError(
            f"Replicate {name} does not match the frozen cohort: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    return table


def summarize_numeric(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_mean": float("nan"), f"{prefix}_sd": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_sd": float(np.std(finite, ddof=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate", action="append", required=True, type=parse_replicate)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--cluster-cutoff", type=float, default=2.0)
    parser.add_argument(
        "--pairwise-rmsd-mode", choices=("ordered", "symmetry"), default="symmetry"
    )
    parser.add_argument("--single-pool-uncertainty", type=Path)
    parser.add_argument(
        "--single-pool-largest-cluster-column",
        default="largest_cluster_fraction_at_2p0A",
    )
    parser.add_argument("--expected-n", type=int)
    args = parser.parse_args()

    if args.cluster_cutoff <= 0:
        raise ValueError("cluster-cutoff must be positive")
    replicate_names = [name for name, _path in args.replicate]
    if len(replicate_names) != len(set(replicate_names)):
        raise ValueError("Replicate names must be unique")
    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    if not budgets or budgets != sorted(set(budgets)) or budgets[0] != 1:
        raise ValueError("budgets must be sorted unique integers beginning with 1")
    if budgets[-1] > len(args.replicate):
        raise ValueError(
            f"Largest budget is {budgets[-1]}, but only {len(args.replicate)} replicates exist"
        )

    cohort_path = args.cohort.expanduser().resolve()
    cohort_ids = read_ids(cohort_path)
    if args.expected_n is not None and len(cohort_ids) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cases, found {len(cohort_ids)}")
    cohort_set = set(cohort_ids)
    replicate_tables = [
        load_replicate(name, root, cohort_set) for name, root in args.replicate
    ]
    selected = pd.concat(replicate_tables, ignore_index=True)
    replicate_order = {name: index + 1 for index, name in enumerate(replicate_names)}
    cohort_order = {name: index for index, name in enumerate(cohort_ids)}
    selected["replicate_order"] = selected["replicate"].map(replicate_order)
    selected["cohort_order"] = selected["complex_name"].map(cohort_order)
    selected = selected.sort_values(
        ["cohort_order", "replicate_order"], kind="stable"
    ).reset_index(drop=True)

    single_pool = None
    if args.single_pool_uncertainty is not None:
        single_pool_path = args.single_pool_uncertainty.expanduser().resolve()
        single_pool = pd.read_csv(single_pool_path, dtype=str)
        id_column = choose_column(single_pool.columns, ID_COLUMNS, "single-pool ID")
        if args.single_pool_largest_cluster_column not in single_pool.columns:
            raise ValueError(
                "Single-pool table lacks "
                f"{args.single_pool_largest_cluster_column!r}"
            )
        single_pool = single_pool[[
            id_column, args.single_pool_largest_cluster_column
        ]].copy()
        single_pool["complex_name"] = single_pool[id_column].map(normalize_id)
        single_pool["single_pool_uncertainty"] = 1.0 - pd.to_numeric(
            single_pool[args.single_pool_largest_cluster_column], errors="coerce"
        )
        if single_pool["complex_name"].duplicated().any():
            raise ValueError("Single-pool uncertainty table contains duplicate cases")
        single_pool = single_pool[["complex_name", "single_pool_uncertainty"]]
        missing = sorted(cohort_set - set(single_pool["complex_name"]))
        if missing:
            raise ValueError(f"Single-pool uncertainty is missing cases: {missing[:20]}")

    feature_rows = []
    failures = []
    for cohort_index, complex_name in enumerate(cohort_ids):
        group = selected[selected["complex_name"].eq(complex_name)].sort_values(
            "replicate_order", kind="stable"
        )
        try:
            molecules = [
                read_heavy_molecule(Path(path)) for path in group["selected_pose_path"]
            ]
            distances = (
                symmetry_pairwise_rmsd(molecules)
                if args.pairwise_rmsd_mode == "symmetry"
                else ordered_pairwise_rmsd(molecules)
            )
            for budget in budgets:
                subset = group.iloc[:budget]
                features = cluster_features(
                    distances[:budget, :budget], [args.cluster_cutoff], selected_index=0
                )
                token = str(args.cluster_cutoff).replace(".", "p")
                row = {
                    "cohort_order": cohort_index,
                    "complex_name": complex_name,
                    "budget_runs": budget,
                    "replicate_names": "|".join(subset["replicate"].astype(str)),
                    "reported_pose_path": str(group.iloc[0]["selected_pose_path"]),
                    "replicate1_confidence": float(group.iloc[0]["confidence"]),
                    "replicate1_final_selection_score": float(
                        group.iloc[0]["final_selection_score"]
                    ),
                    "one_minus_largest_cluster_fraction_at_2p0A": (
                        1.0 - float(features[f"largest_cluster_fraction_at_{token}A"])
                    ),
                    "one_minus_reported_pose_cluster_fraction_at_2p0A": (
                        1.0 - float(features[f"selected_cluster_fraction_at_{token}A"])
                    ),
                    "cluster_entropy_at_2p0A": float(
                        features[f"cluster_entropy_at_{token}A"]
                    ),
                    "pairwise_rmsd_median_A": float(features["pairwise_rmsd_median_A"]),
                    "pairwise_rmsd_p90_A": float(features["pairwise_rmsd_p90_A"]),
                }
                for source_column, prefix in (
                    ("final_selection_score", "final_selection_score"),
                    ("multiscale_ncc", "selected_multiscale_ncc"),
                    ("expanded_density_z", "selected_density_z"),
                ):
                    row.update(summarize_numeric(subset[source_column], prefix))
                feature_rows.append(row)
        except Exception as exc:
            failures.append({
                "complex_name": complex_name,
                "error": f"{type(exc).__name__}: {exc}",
            })

    features = pd.DataFrame(feature_rows)
    if single_pool is not None and len(features):
        features = features.merge(
            single_pool, on="complex_name", how="left", validate="many_to_one"
        )
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_pose_replicates.csv", index=False)
    features.to_csv(output / "multirun_uncertainty_features.csv", index=False)
    pd.DataFrame(failures, columns=["complex_name", "error"]).to_csv(
        output / "failures.csv", index=False
    )
    summary = {
        "analysis_role": "reference_free_multirun_uncertainty_feature_generation",
        "cohort": str(cohort_path),
        "cohort_sha256": sha256(cohort_path),
        "requested_cases_n": len(cohort_ids),
        "evaluated_cases_n": int(features["complex_name"].nunique()) if len(features) else 0,
        "failures_n": len(failures),
        "replicates": [
            {"name": name, "root": str(root)} for name, root in args.replicate
        ],
        "nested_budgets": budgets,
        "reported_prediction_policy": "V2 selected Top-1 from the first replicate",
        "cross_run_cluster_anchor": "cluster containing the first-replicate reported pose",
        "cluster_cutoff_A": args.cluster_cutoff,
        "pairwise_rmsd_mode": args.pairwise_rmsd_mode,
        "uses_reference_coordinates": False,
        "single_pool_uncertainty": (
            str(args.single_pool_uncertainty.expanduser().resolve())
            if args.single_pool_uncertainty is not None else None
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"Feature construction failed for {len(failures)} cases")
    print(f"MULTIRUN_UNCERTAINTY_FEATURES_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
