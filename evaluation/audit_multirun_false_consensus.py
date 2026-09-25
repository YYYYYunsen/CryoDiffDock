#!/usr/bin/env python3
"""Audit complete cross-run agreement, density support, and RMSD failure modes.

This is a post-hoc mechanism audit of a *frozen* multirun uncertainty result.
It does not tune the 2 A clustering threshold or use reference coordinates to
construct the uncertainty score.  Reference coordinates are used only after
the reported pose and agreement groups have already been fixed.

Expected inputs are produced by ``build_multirun_uncertainty_features.py``:

* ``multirun_uncertainty_features.csv`` (one row per case and run budget)
* ``selected_pose_replicates.csv`` (one frozen V2 Top-1 per run and case)

The script re-reads each selected pose's ``ranking.csv`` to recover NCC,
in-bounds, and physics fields.  It optionally reads a reveal manifest to
calculate translation-, orientation-, and conformation-sensitive RMSD
components when those columns are absent from the endpoint table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

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
from evaluation.evaluate_ligand_accuracy import (
    centroid_distance,
    read_heavy_molecule,
    symmetry_rmsd,
)


ID_COLUMNS = ("complex_name", "pdb_id", "complex", "name", "id")
RMSD_COLUMNS = (
    "endpoint_rmsd_A",
    "top1_rmsd_A",
    "selected_rmsd_A",
    "ligand_rmsd_A",
    "v2_rmsd_A",
)
DECOMPOSITION_COLUMNS = {
    "top1_centroid_distance_A": (
        "top1_centroid_distance_A",
        "centroid_distance_A",
        "selected_centroid_distance_A",
    ),
    "top1_centered_rmsd_A": (
        "top1_centered_rmsd_A",
        "centered_rmsd_A",
        "selected_centered_rmsd_A",
    ),
    "top1_rigid_aligned_rmsd_A": (
        "top1_rigid_aligned_rmsd_A",
        "rigid_aligned_rmsd_A",
        "selected_rigid_aligned_rmsd_A",
    ),
}
RANKING_NUMERIC_COLUMNS = (
    "ncc",
    "multiscale_ncc",
    "expanded_density_z",
    "in_bounds_fraction",
    "severe_clash_count",
    "clashing_ligand_atom_fraction",
    "basin_q25_ncc",
    "basin_support_fraction",
    "final_selection_score",
    "confidence",
)
RANKING_BOOLEAN_COLUMNS = (
    "relative_physics_pass",
    "density_gate_pass",
    "density_physics_gate_pass",
)
INTERACTION_NUMERIC_COLUMNS = (
    "hbond_geometry_quality",
    "hbond_geometry_count",
    "aromatic_geometry_quality",
    "aromatic_contact_count",
    "salt_bridge_geometry_quality",
    "salt_bridge_geometry_count",
    "interaction_geometry_quality",
)


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_column(columns: Iterable[str], candidates: Iterable[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; found {list(columns)}")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def as_bool(value: object) -> bool | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return np.nan


def parse_cutoffs(value: str) -> list[float]:
    cutoffs = [float(item) for item in value.split(",") if item.strip()]
    if not cutoffs or any(item <= 0 for item in cutoffs):
        raise argparse.ArgumentTypeError("cluster cutoffs must be positive")
    if len(cutoffs) != len(set(cutoffs)):
        raise argparse.ArgumentTypeError("cluster cutoffs must be unique")
    return cutoffs


def read_cohort(path: Path) -> list[str]:
    ids = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return ids


def select_ranking_row(path: Path) -> pd.Series:
    ranking = pd.read_csv(path, dtype={"complex_name": str})
    if "materialized_rank" in ranking.columns:
        chosen = ranking[
            pd.to_numeric(ranking["materialized_rank"], errors="coerce").eq(1)
        ]
    elif "final_rank" in ranking.columns:
        chosen = ranking[pd.to_numeric(ranking["final_rank"], errors="coerce").eq(1)]
    else:
        chosen = ranking.iloc[:1]
    if len(chosen) != 1:
        raise ValueError(f"Expected exactly one selected ranking row: {path}")
    return chosen.iloc[0]


def augment_selected_pose_rows(
    selected: pd.DataFrame, score_interactions: bool
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    records = []
    ranking_failures = []
    interaction_failures = []
    protein_cache: dict[str, object] = {}
    if score_interactions:
        from cryodiffdock_pocket.interactions import (
            read_protein_interaction_sites,
            score_pose_interactions,
        )
    for raw in selected.to_dict("records"):
        record = dict(raw)
        ranking_path = Path(str(raw["ranking_path"])).expanduser().resolve()
        try:
            row = select_ranking_row(ranking_path)
            ranking_id = normalize_id(row.get("complex_name", raw["complex_name"]))
            if ranking_id != raw["complex_name"]:
                raise ValueError(
                    f"ranking case {ranking_id} != selected case {raw['complex_name']}"
                )
            for column in RANKING_NUMERIC_COLUMNS:
                record[column] = pd.to_numeric(row.get(column, np.nan), errors="coerce")
            for column in RANKING_BOOLEAN_COLUMNS:
                record[column] = as_bool(row.get(column, np.nan))
            protein_path = Path(str(row.get("protein_path", ""))).expanduser().resolve()
            record["protein_path"] = str(protein_path)
            if score_interactions:
                try:
                    cache_key = str(protein_path)
                    if cache_key not in protein_cache:
                        protein_cache[cache_key] = read_protein_interaction_sites(
                            protein_path
                        )
                    interaction = score_pose_interactions(
                        protein_cache[cache_key], raw["selected_pose_path"]
                    )
                    for column in INTERACTION_NUMERIC_COLUMNS:
                        record[column] = pd.to_numeric(
                            interaction.get(column, np.nan), errors="coerce"
                        )
                except Exception as exc:
                    interaction_failures.append(
                        {
                            "complex_name": raw["complex_name"],
                            "replicate": raw.get("replicate", ""),
                            "selected_pose_path": raw["selected_pose_path"],
                            "protein_path": str(protein_path),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    for column in INTERACTION_NUMERIC_COLUMNS:
                        record[column] = np.nan
            records.append(record)
        except Exception as exc:
            ranking_failures.append(
                {
                    "complex_name": raw["complex_name"],
                    "replicate": raw.get("replicate", ""),
                    "ranking_path": str(ranking_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(records), ranking_failures, interaction_failures


def aggregate_selected_pose_fields(group: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for column in (*RANKING_NUMERIC_COLUMNS, *INTERACTION_NUMERIC_COLUMNS):
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        prefix = f"selected_{column}"
        result[f"{prefix}_n"] = int(len(finite))
        result[f"{prefix}_median"] = float(np.median(finite)) if len(finite) else np.nan
        result[f"{prefix}_sd"] = float(np.std(finite, ddof=0)) if len(finite) else np.nan
        result[f"{prefix}_min"] = float(np.min(finite)) if len(finite) else np.nan
        result[f"{prefix}_max"] = float(np.max(finite)) if len(finite) else np.nan
        result[f"{prefix}_range"] = (
            float(np.max(finite) - np.min(finite)) if len(finite) else np.nan
        )
    for column in RANKING_BOOLEAN_COLUMNS:
        values = group[column].dropna().astype(bool)
        result[f"selected_{column}_n"] = int(len(values))
        result[f"selected_{column}_fraction"] = (
            float(values.mean()) if len(values) else np.nan
        )
    return result


def build_pairwise_features(
    group: pd.DataFrame,
    cutoffs: list[float],
    mode: str,
) -> tuple[dict[str, object], list[dict]]:
    group = group.sort_values("replicate_order", kind="stable")
    molecules = [read_heavy_molecule(Path(path)) for path in group["selected_pose_path"]]
    distances = (
        symmetry_pairwise_rmsd(molecules)
        if mode == "symmetry"
        else ordered_pairwise_rmsd(molecules)
    )
    features = cluster_features(distances, cutoffs, selected_index=0)
    pair_rows = []
    names = group["replicate"].astype(str).tolist()
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_rows.append(
                {
                    "complex_name": group.iloc[0]["complex_name"],
                    "replicate_left": names[left],
                    "replicate_right": names[right],
                    "pairwise_rmsd_A": float(distances[left, right]),
                }
            )
    return features, pair_rows


def standardize_endpoint(endpoint: pd.DataFrame, rmsd_column: str | None) -> pd.DataFrame:
    id_column = choose_column(endpoint.columns, ID_COLUMNS, "endpoint ID")
    if rmsd_column is None:
        rmsd_column = choose_column(endpoint.columns, RMSD_COLUMNS, "Top-1 RMSD")
    if rmsd_column not in endpoint.columns:
        raise ValueError(f"Endpoint RMSD column not found: {rmsd_column}")
    result = pd.DataFrame(
        {
            "complex_name": endpoint[id_column].map(normalize_id),
            "top1_rmsd_A": pd.to_numeric(endpoint[rmsd_column], errors="coerce"),
        }
    )
    for output_column, candidates in DECOMPOSITION_COLUMNS.items():
        found = next((item for item in candidates if item in endpoint.columns), None)
        result[output_column] = (
            pd.to_numeric(endpoint[found], errors="coerce") if found else np.nan
        )
    if result["complex_name"].duplicated().any():
        duplicates = sorted(
            result.loc[result["complex_name"].duplicated(False), "complex_name"].unique()
        )
        raise ValueError(f"Endpoint table contains duplicate cases: {duplicates[:20]}")
    return result


def reference_paths(manifest_path: Path) -> dict[str, Path]:
    manifest = read_table(manifest_path)
    id_column = choose_column(manifest.columns, ID_COLUMNS, "reference-manifest ID")
    reference_column = choose_column(
        manifest.columns,
        ("reference_ligand", "reference_ligand_path", "ground_truth_ligand"),
        "reference ligand",
    )
    result = {}
    for _, row in manifest.iterrows():
        complex_name = normalize_id(row[id_column])
        path = Path(str(row[reference_column])).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        result[complex_name] = path.resolve()
    return result


def fill_reference_decomposition(
    per_case: pd.DataFrame,
    run1_poses: dict[str, Path],
    references: dict[str, Path],
) -> tuple[pd.DataFrame, list[dict]]:
    failures = []
    for index, row in per_case.iterrows():
        complex_name = row["complex_name"]
        try:
            prediction = read_heavy_molecule(run1_poses[complex_name])
            reference = read_heavy_molecule(references[complex_name])
            per_case.at[index, "recomputed_top1_rmsd_A"] = symmetry_rmsd(
                reference, prediction
            )
            per_case.at[index, "top1_centroid_distance_A"] = centroid_distance(
                reference, prediction
            )
            per_case.at[index, "top1_centered_rmsd_A"] = symmetry_rmsd(
                reference, prediction, center=True
            )
            per_case.at[index, "top1_rigid_aligned_rmsd_A"] = symmetry_rmsd(
                reference, prediction, center=True, minimize=True
            )
        except Exception as exc:
            failures.append(
                {
                    "complex_name": complex_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return per_case, failures


def group_summary(
    group: pd.DataFrame, label: str, success_cutoff: float
) -> dict[str, object]:
    rmsd = pd.to_numeric(group["top1_rmsd_A"], errors="coerce")
    result: dict[str, object] = {
        "group": label,
        "n": int(len(group)),
        "success_count": int(rmsd.lt(success_cutoff).sum()),
        "failure_count": int(rmsd.ge(success_cutoff).sum()),
        "success_pct": (
            float(100.0 * rmsd.lt(success_cutoff).mean()) if len(group) else np.nan
        ),
        "median_top1_rmsd_A": float(rmsd.median()) if len(group) else np.nan,
    }
    for column in (
        "selected_multiscale_ncc_median",
        "selected_multiscale_ncc_sd",
        "selected_in_bounds_fraction_min",
        "selected_severe_clash_count_median",
        "selected_clashing_ligand_atom_fraction_median",
        "selected_hbond_geometry_quality_median",
        "selected_aromatic_geometry_quality_median",
        "selected_salt_bridge_geometry_quality_median",
        "selected_interaction_geometry_quality_median",
        "top1_centroid_distance_A",
        "top1_centered_rmsd_A",
        "top1_rigid_aligned_rmsd_A",
    ):
        if column in group.columns:
            values = pd.to_numeric(group[column], errors="coerce")
            result[f"median_{column}"] = (
                float(values.median()) if values.notna().any() else np.nan
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multirun-features", required=True, type=Path)
    parser.add_argument("--selected-pose-replicates", required=True, type=Path)
    parser.add_argument("--endpoint-table", required=True, type=Path)
    parser.add_argument("--endpoint-rmsd-column")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument(
        "--score-interactions",
        action="store_true",
        help=(
            "Compute deployment-safe protein-ligand interaction geometry for "
            "each selected pose. This is descriptive and does not rerank poses."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--success-cutoff", type=float, default=2.0)
    parser.add_argument("--rmsd-consistency-tolerance", type=float, default=1e-3)
    parser.add_argument("--cluster-cutoffs", type=parse_cutoffs, default=[1.0, 2.0])
    parser.add_argument(
        "--pairwise-rmsd-mode", choices=("ordered", "symmetry"), default="symmetry"
    )
    parser.add_argument("--expected-n", type=int)
    args = parser.parse_args()

    if (
        args.budget <= 0
        or args.success_cutoff <= 0
        or args.rmsd_consistency_tolerance < 0
    ):
        raise ValueError("budget/cutoff must be positive and tolerance nonnegative")
    agreement_cutoff = 2.0
    if not any(np.isclose(value, agreement_cutoff) for value in args.cluster_cutoffs):
        raise ValueError("cluster-cutoffs must contain the frozen 2.0 A cutoff")
    inputs = {
        name: path.expanduser().resolve()
        for name, path in {
            "multirun_features": args.multirun_features,
            "selected_pose_replicates": args.selected_pose_replicates,
            "endpoint_table": args.endpoint_table,
        }.items()
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")

    features = read_table(inputs["multirun_features"])
    features["complex_name"] = features["complex_name"].map(normalize_id)
    features["budget_runs"] = pd.to_numeric(features["budget_runs"], errors="raise")
    features = features[features["budget_runs"].eq(args.budget)].copy()
    if features.empty:
        raise ValueError(f"No multirun feature rows for budget {args.budget}")
    if features["complex_name"].duplicated().any():
        raise ValueError("Multirun feature table contains duplicate cases at this budget")
    frozen_disagreement_column = (
        "one_minus_reported_pose_cluster_fraction_at_2p0A"
    )
    if frozen_disagreement_column not in features.columns:
        raise ValueError(
            f"Multirun feature table lacks {frozen_disagreement_column!r}"
        )
    features = features[["complex_name", frozen_disagreement_column]].copy()
    features["frozen_reported_pose_disagreement"] = pd.to_numeric(
        features.pop(frozen_disagreement_column), errors="raise"
    )

    selected = read_table(inputs["selected_pose_replicates"])
    required_selected = {
        "complex_name", "replicate", "selected_pose_path", "ranking_path"
    }
    missing_columns = sorted(required_selected - set(selected.columns))
    if missing_columns:
        raise ValueError(f"Selected-pose table lacks columns: {missing_columns}")
    selected["complex_name"] = selected["complex_name"].map(normalize_id)
    if "replicate_order" not in selected.columns:
        replicate_order = {
            name: index + 1
            for index, name in enumerate(selected["replicate"].drop_duplicates())
        }
        selected["replicate_order"] = selected["replicate"].map(replicate_order)
    selected["replicate_order"] = pd.to_numeric(
        selected["replicate_order"], errors="raise"
    )
    selected = selected[selected["replicate_order"].le(args.budget)].copy()

    cohort_ids = None
    cohort_path = None
    if args.cohort is not None:
        cohort_path = args.cohort.expanduser().resolve()
        cohort_ids = read_cohort(cohort_path)
        feature_ids = set(features["complex_name"])
        if feature_ids != set(cohort_ids):
            raise ValueError(
                "Feature table does not match cohort: "
                f"missing={sorted(set(cohort_ids)-feature_ids)[:20]}, "
                f"extra={sorted(feature_ids-set(cohort_ids))[:20]}"
            )
    if args.expected_n is not None and len(features) != args.expected_n:
        raise ValueError(f"Expected {args.expected_n} cases, found {len(features)}")

    counts = selected.groupby("complex_name").size()
    bad_counts = counts[counts.ne(args.budget)]
    if len(bad_counts):
        raise ValueError(
            "Selected-pose replicate count mismatch: "
            + ", ".join(f"{name}={count}" for name, count in bad_counts.head(20).items())
        )
    if set(counts.index) != set(features["complex_name"]):
        raise ValueError("Selected-pose and multirun-feature case sets differ")

    (
        selected_augmented,
        ranking_failures,
        interaction_failures,
    ) = augment_selected_pose_rows(selected, args.score_interactions)
    pair_rows = []
    case_rows = []
    pose_failures = []
    for complex_name, group in selected_augmented.groupby("complex_name", sort=False):
        try:
            pair_features, case_pairs = build_pairwise_features(
                group, args.cluster_cutoffs, args.pairwise_rmsd_mode
            )
            pair_rows.extend(case_pairs)
            case_rows.append(
                {
                    "complex_name": complex_name,
                    **pair_features,
                    **aggregate_selected_pose_fields(group),
                }
            )
        except Exception as exc:
            pose_failures.append(
                {"complex_name": complex_name, "error": f"{type(exc).__name__}: {exc}"}
            )
    if ranking_failures or pose_failures:
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ranking_failures).to_csv(output / "ranking_read_failures.csv", index=False)
        pd.DataFrame(interaction_failures).to_csv(
            output / "interaction_scoring_failures.csv", index=False
        )
        pd.DataFrame(pose_failures).to_csv(output / "pose_analysis_failures.csv", index=False)
        raise RuntimeError(
            f"Audit input failures: ranking={len(ranking_failures)}, pose={len(pose_failures)}"
        )

    audit = pd.DataFrame(case_rows)
    endpoint_raw = read_table(inputs["endpoint_table"])
    # Direct multirun evaluation tables contain one row per nested budget.
    # Select the requested frozen budget before enforcing one row per case.
    if "budget_runs" in endpoint_raw.columns:
        endpoint_budget = pd.to_numeric(endpoint_raw["budget_runs"], errors="coerce")
        endpoint_raw = endpoint_raw[endpoint_budget.eq(args.budget)].copy()
        if endpoint_raw.empty:
            raise ValueError(
                f"Endpoint table has no rows for budget_runs={args.budget}"
            )
    endpoint = standardize_endpoint(endpoint_raw, args.endpoint_rmsd_column)
    audit = audit.merge(endpoint, on="complex_name", how="left", validate="one_to_one")
    audit = audit.merge(features, on="complex_name", how="left", validate="one_to_one")
    if audit["top1_rmsd_A"].isna().any():
        missing = audit.loc[audit["top1_rmsd_A"].isna(), "complex_name"].tolist()
        raise ValueError(f"Endpoint table is missing scored cases: {missing[:20]}")

    reference_failures = []
    reference_manifest_path = None
    if args.reference_manifest is not None:
        reference_manifest_path = args.reference_manifest.expanduser().resolve()
        references = reference_paths(reference_manifest_path)
        run1 = selected_augmented.sort_values("replicate_order").groupby(
            "complex_name", sort=False
        ).first()
        run1_poses = {
            name: Path(path).expanduser().resolve()
            for name, path in run1["selected_pose_path"].items()
        }
        audit, reference_failures = fill_reference_decomposition(
            audit, run1_poses, references
        )
        audit["endpoint_vs_recomputed_rmsd_absdiff_A"] = np.abs(
            pd.to_numeric(audit["top1_rmsd_A"], errors="coerce")
            - pd.to_numeric(audit["recomputed_top1_rmsd_A"], errors="coerce")
        )
        inconsistent = audit[
            audit["endpoint_vs_recomputed_rmsd_absdiff_A"].gt(
                args.rmsd_consistency_tolerance
            )
        ]
        if len(inconsistent):
            example = inconsistent[[
                "complex_name",
                "top1_rmsd_A",
                "recomputed_top1_rmsd_A",
                "endpoint_vs_recomputed_rmsd_absdiff_A",
            ]].head(20)
            raise ValueError(
                "Endpoint RMSD does not match the first-replicate reported pose:\n"
                + example.to_string(index=False)
            )

    cutoff_token = str(float(agreement_cutoff)).replace(".", "p")
    selected_fraction_column = f"selected_cluster_fraction_at_{cutoff_token}A"
    if selected_fraction_column not in audit.columns:
        raise ValueError(f"Missing clustering field: {selected_fraction_column}")
    audit["reported_pose_cluster_fraction"] = pd.to_numeric(
        audit[selected_fraction_column], errors="raise"
    )
    audit["reported_pose_cluster_size"] = np.rint(
        audit["reported_pose_cluster_fraction"] * args.budget
    ).astype(int)
    audit["reported_pose_disagreement"] = 1.0 - audit[
        "reported_pose_cluster_fraction"
    ]
    disagreement_delta = np.abs(
        audit["reported_pose_disagreement"]
        - audit["frozen_reported_pose_disagreement"]
    )
    if disagreement_delta.gt(1e-9).any():
        bad = audit.loc[
            disagreement_delta.gt(1e-9),
            [
                "complex_name",
                "reported_pose_disagreement",
                "frozen_reported_pose_disagreement",
            ],
        ].head(20)
        raise ValueError(
            "Recomputed 2 A disagreement differs from the frozen feature:\n"
            + bad.to_string(index=False)
        )
    audit["top1_success"] = audit["top1_rmsd_A"].lt(args.success_cutoff)
    audit["top1_failure"] = ~audit["top1_success"]
    audit["complete_agreement"] = audit["reported_pose_cluster_size"].eq(args.budget)
    audit["false_consensus"] = audit["complete_agreement"] & audit["top1_failure"]
    audit["agreement_label"] = (
        audit["reported_pose_cluster_size"].astype(str) + f"/{args.budget}"
    )

    # These are descriptive, nonexclusive error components.  They do not
    # redefine success and are intentionally tied to the fixed 2 A benchmark.
    for output_column, source_column in (
        ("centroid_mismatch_ge2A", "top1_centroid_distance_A"),
        ("orientation_or_internal_mismatch_ge2A", "top1_centered_rmsd_A"),
        ("internal_mismatch_ge2A", "top1_rigid_aligned_rmsd_A"),
    ):
        values = pd.to_numeric(audit[source_column], errors="coerce")
        audit[output_column] = values.ge(2.0).where(values.notna()).astype("boolean")

    agreement_rows = []
    for cluster_size, group in audit.groupby("reported_pose_cluster_size", sort=True):
        row = group_summary(
            group, f"{int(cluster_size)}/{args.budget}", args.success_cutoff
        )
        row["reported_pose_cluster_size"] = int(cluster_size)
        row["reported_pose_disagreement"] = 1.0 - cluster_size / args.budget
        agreement_rows.append(row)
    agreement_summary = pd.DataFrame(agreement_rows).sort_values(
        "reported_pose_cluster_size", ascending=False
    )

    complete = audit[audit["complete_agreement"]]
    comparison = pd.DataFrame(
        [
            group_summary(
                complete[complete["top1_success"]],
                f"{args.budget}/{args.budget} success",
                args.success_cutoff,
            ),
            group_summary(
                complete[complete["top1_failure"]],
                f"{args.budget}/{args.budget} RMSD failure",
                args.success_cutoff,
            ),
            group_summary(
                audit[~audit["complete_agreement"]],
                f"less than {args.budget}/{args.budget}",
                args.success_cutoff,
            ),
        ]
    )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    order = {name: index for index, name in enumerate(cohort_ids or features["complex_name"])}
    audit["cohort_order"] = audit["complex_name"].map(order)
    audit = audit.sort_values("cohort_order", kind="stable")
    audit.to_csv(output / "per_case_false_consensus_audit.csv", index=False)
    audit[audit["false_consensus"]].to_csv(
        output / "complete_agreement_rmsd_failures.csv", index=False
    )
    agreement_summary.to_csv(output / "agreement_level_summary.csv", index=False)
    comparison.to_csv(output / "complete_agreement_group_comparison.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output / "pairwise_selected_pose_rmsd.csv", index=False)
    selected_augmented.to_csv(output / "selected_pose_replicates_augmented.csv", index=False)
    pd.DataFrame(reference_failures, columns=["complex_name", "error"]).to_csv(
        output / "reference_decomposition_failures.csv", index=False
    )
    pd.DataFrame(
        interaction_failures,
        columns=[
            "complex_name", "replicate", "selected_pose_path", "protein_path", "error"
        ],
    ).to_csv(output / "interaction_scoring_failures.csv", index=False)

    complete_n = int(audit["complete_agreement"].sum())
    complete_success_n = int((audit["complete_agreement"] & audit["top1_success"]).sum())
    false_consensus_n = int(audit["false_consensus"].sum())
    summary = {
        "analysis_role": "post_frozen_false_consensus_mechanism_audit",
        "requested_cases_n": int(len(audit)),
        "budget_runs": args.budget,
        "reported_prediction": "first-replicate frozen V2 selected Top-1",
        "agreement_definition": (
            f"size of the complete-linkage cluster containing the reported pose; "
            f"symmetry-aware heavy-atom RMSD cutoff {args.cluster_cutoffs[-1]:g} A"
        ),
        "complete_agreement_n": complete_n,
        "complete_agreement_success_n": complete_success_n,
        "complete_agreement_rmsd_failure_n": false_consensus_n,
        "complete_agreement_success_pct": (
            100.0 * complete_success_n / complete_n if complete_n else np.nan
        ),
        "success_definition": f"reported Top-1 RMSD < {args.success_cutoff:g} A",
        "interpretation": (
            "Agreement is a reproducibility and empirical risk signal, not a "
            "correctness certificate.  NCC support does not by itself establish "
            "atom-level equivalence to the deposited reference."
        ),
        "reference_usage": (
            "Reference-derived RMSD labels and error decomposition are joined only "
            "after the reported pose and agreement clusters are fixed."
        ),
        "pairwise_rmsd_mode": args.pairwise_rmsd_mode,
        "cluster_cutoffs_A": args.cluster_cutoffs,
        "rmsd_consistency_tolerance_A": args.rmsd_consistency_tolerance,
        "ranking_read_failures_n": len(ranking_failures),
        "pose_analysis_failures_n": len(pose_failures),
        "reference_decomposition_failures_n": len(reference_failures),
        "interaction_geometry_scored": args.score_interactions,
        "interaction_scoring_failures_n": len(interaction_failures),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
    }
    if cohort_path is not None:
        summary["inputs"]["cohort"] = {
            "path": str(cohort_path), "sha256": sha256(cohort_path)
        }
    if reference_manifest_path is not None:
        summary["inputs"]["reference_manifest"] = {
            "path": str(reference_manifest_path),
            "sha256": sha256(reference_manifest_path),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("AGREEMENT LEVEL SUMMARY")
    print(agreement_summary.to_string(index=False))
    print("\nCOMPLETE AGREEMENT GROUP COMPARISON")
    print(comparison.to_string(index=False))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nMULTIRUN_FALSE_CONSENSUS_AUDIT_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
