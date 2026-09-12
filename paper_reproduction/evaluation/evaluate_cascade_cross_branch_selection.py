#!/usr/bin/env python3
"""Evaluate GT-free selectors across the first K Global-to-Local branches.

This is a post-inference analysis.  Each selector sees only deployment-time
metadata from the Global branch and the frozen V2-selected pose.  RMSD is read
only after branch selection to evaluate the chosen output.  The RMSD-minimum
rule is emitted separately and explicitly labelled as an oracle ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError("Cohort must contain unique nonempty IDs")
    return values


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_selected_rows(root: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, manifest_row in manifest.iterrows():
        matches = sorted(path for path in root.glob(f"index{row_index}___*") if path.is_dir())
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one V2-selected directory for row {row_index}, found {len(matches)}"
            )
        ranking_path = matches[0] / "ranking.csv"
        ranking = pd.read_csv(ranking_path)
        if len(ranking) != 1:
            raise ValueError(f"{ranking_path} contains {len(ranking)} rows; expected 1")
        record = ranking.iloc[0].to_dict()
        expected_name = str(manifest_row["complex_name"])
        if str(record.get("complex_name", "")) != expected_name:
            raise ValueError(
                f"V2 ranking name mismatch at row {row_index}: "
                f"{record.get('complex_name')!r} != {expected_name!r}"
            )
        record["row_index"] = int(row_index)
        rows.append(record)
    return pd.DataFrame(rows)


def choose_max(group: pd.DataFrame, column: str) -> pd.Series | None:
    finite = group[np.isfinite(group[column])].copy()
    if finite.empty:
        return None
    return finite.sort_values(
        [column, "global_rank", "complex_name"],
        ascending=[False, True, True],
        kind="stable",
    ).iloc[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-manifest", required=True, type=Path)
    parser.add_argument("--branch-metrics", required=True, type=Path)
    parser.add_argument("--v2-selected-root", required=True, type=Path)
    parser.add_argument("--requested-cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--allow-incomplete-top-k", action="store_true",
        help="Select from available branches instead of counting incomplete Top-K as failure.",
    )
    args = parser.parse_args()
    if args.top_k <= 0 or args.bootstrap_repeats <= 0:
        raise ValueError("top-k and bootstrap-repeats must be positive")

    manifest_path = args.branch_manifest.expanduser().resolve()
    metrics_path = args.branch_metrics.expanduser().resolve()
    selected_root = args.v2_selected_root.expanduser().resolve()
    manifest = pd.read_csv(manifest_path, dtype=str)
    metrics = pd.read_csv(metrics_path, dtype=str)
    required_manifest = {"complex_name", "parent_complex_name", "global_rank", "global_score"}
    missing = sorted(required_manifest - set(manifest.columns))
    if missing:
        raise ValueError(f"Branch manifest lacks columns: {missing}")
    if manifest["complex_name"].isna().any() or manifest["complex_name"].duplicated().any():
        raise ValueError("Branch manifest complex names must be nonempty and unique")
    if not {"row_index", "top1_rmsd_A"} <= set(metrics.columns):
        raise ValueError("Branch metrics requires row_index and top1_rmsd_A")

    manifest = manifest.reset_index().rename(columns={"index": "row_index"})
    manifest["row_index"] = manifest["row_index"].astype(int)
    manifest["parent_complex_name"] = manifest["parent_complex_name"].map(normalize)
    manifest["global_rank"] = pd.to_numeric(manifest["global_rank"], errors="raise").astype(int)
    manifest["global_score"] = pd.to_numeric(manifest["global_score"], errors="coerce")
    metrics["row_index"] = pd.to_numeric(metrics["row_index"], errors="raise").astype(int)
    metrics["top1_rmsd_A"] = pd.to_numeric(metrics["top1_rmsd_A"], errors="coerce")
    if metrics["row_index"].duplicated().any():
        raise ValueError("Branch metrics contains duplicate row indices")

    selected = read_selected_rows(selected_root, manifest)
    feature_columns = (
        "multiscale_ncc", "basin_q25_ncc", "basin_support_fraction",
        "final_selection_score", "selection_score", "relative_physics_pass",
        "is_original", "severe_clash_count", "clashing_ligand_atom_fraction",
    )
    missing = sorted(set(feature_columns) - set(selected.columns))
    if missing:
        raise ValueError(f"V2 ranking rows lack selector features: {missing}")
    selected = selected[["row_index", "candidate_id", *feature_columns]].copy()
    numeric_features = [
        column for column in feature_columns
        if column not in {"relative_physics_pass", "is_original"}
    ]
    for column in numeric_features:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["relative_physics_pass"] = as_bool(selected["relative_physics_pass"])
    selected["is_original"] = as_bool(selected["is_original"])

    branch = manifest.merge(selected, on="row_index", how="left", validate="one_to_one")
    branch = branch.merge(
        metrics[["row_index", "top1_rmsd_A"]],
        on="row_index", how="left", validate="one_to_one",
    ).rename(columns={"top1_rmsd_A": "selected_pose_rmsd_A"})
    branch = branch[branch["global_rank"].le(args.top_k)].copy()
    requested = read_ids(args.requested_cohort.expanduser().resolve())
    requested_set = set(requested)
    branch = branch[branch["parent_complex_name"].isin(requested_set)].copy()
    expected_ranks = set(range(1, args.top_k + 1))

    rule_metadata = {
        "global_rank1": {
            "uses_ground_truth": False,
            "description": "Frozen Global rank-1 branch (current operational cascade)",
        },
        "max_selected_multiscale_ncc": {
            "uses_ground_truth": False,
            "description": "Highest absolute multiscale NCC among K frozen V2 outputs",
        },
        "max_basin_q25_ncc": {
            "uses_ground_truth": False,
            "description": "Highest lower-quartile local-basin NCC among K frozen V2 outputs",
        },
        "max_final_selection_score": {
            "uses_ground_truth": False,
            "description": "Highest V2 final score; exploratory because components are standardized within branch",
        },
        "max_selected_ncc_physics_gate": {
            "uses_ground_truth": False,
            "description": "Highest multiscale NCC among relative-physics-pass/original outputs",
        },
        "max_basin_support_fraction": {
            "uses_ground_truth": False,
            "description": "Highest V2 basin-support fraction among K frozen V2 outputs",
        },
        f"oracle_best_of_{args.top_k}": {
            "uses_ground_truth": True,
            "description": "Minimum RMSD among K outputs; retrospective ceiling only",
        },
    }

    per_parent_rows: list[dict[str, object]] = []
    for pdb_id in requested:
        group = branch[branch["parent_complex_name"].eq(pdb_id)].copy()
        ranks = set(group["global_rank"].tolist())
        complete = ranks >= expected_ranks and group["selected_pose_rmsd_A"].notna().sum() >= args.top_k
        strict_failure = not complete and not args.allow_incomplete_top_k
        selections: dict[str, pd.Series | None] = {}
        rank1 = group[group["global_rank"].eq(1)]
        selections["global_rank1"] = rank1.iloc[0] if len(rank1) == 1 else None
        if strict_failure:
            for rule in rule_metadata:
                if rule != "global_rank1":
                    selections[rule] = None
        else:
            selections["max_selected_multiscale_ncc"] = choose_max(group, "multiscale_ncc")
            selections["max_basin_q25_ncc"] = choose_max(group, "basin_q25_ncc")
            selections["max_final_selection_score"] = choose_max(group, "final_selection_score")
            physics = group[group["relative_physics_pass"] | group["is_original"]]
            selections["max_selected_ncc_physics_gate"] = choose_max(
                physics, "multiscale_ncc"
            )
            selections["max_basin_support_fraction"] = choose_max(
                group, "basin_support_fraction"
            )
            finite_rmsd = group[np.isfinite(group["selected_pose_rmsd_A"])]
            selections[f"oracle_best_of_{args.top_k}"] = (
                finite_rmsd.sort_values(
                    ["selected_pose_rmsd_A", "global_rank"],
                    ascending=[True, True], kind="stable",
                ).iloc[0]
                if len(finite_rmsd) else None
            )
        record: dict[str, object] = {
            "pdb_id": pdb_id,
            "available_branch_count": int(group["selected_pose_rmsd_A"].notna().sum()),
            "complete_top_k": bool(complete),
        }
        for rule, chosen in selections.items():
            record[f"{rule}_selected_global_rank"] = (
                int(chosen["global_rank"]) if chosen is not None else np.nan
            )
            record[f"{rule}_selected_rmsd_A"] = (
                float(chosen["selected_pose_rmsd_A"])
                if chosen is not None and np.isfinite(chosen["selected_pose_rmsd_A"])
                else np.nan
            )
        per_parent_rows.append(record)

    parents = pd.DataFrame(per_parent_rows)
    denominator = len(parents)
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(
        0, denominator, size=(args.bootstrap_repeats, denominator)
    )
    metric_rows: list[dict[str, object]] = []
    success_arrays: dict[tuple[str, float], np.ndarray] = {}
    for rule, metadata in rule_metadata.items():
        values = parents[f"{rule}_selected_rmsd_A"].to_numpy(float)
        finite = values[np.isfinite(values)]
        for cutoff in (1.0, 2.0, 5.0):
            success = np.isfinite(values) & (values < cutoff)
            success_arrays[(rule, cutoff)] = success
            sampled = 100.0 * success[sample_indices].mean(axis=1)
            low, high = np.quantile(sampled, [0.025, 0.975])
            metric_rows.append({
                "selector": rule,
                "uses_ground_truth": metadata["uses_ground_truth"],
                "cutoff_A": cutoff,
                "success_count": int(success.sum()),
                "denominator": denominator,
                "success_pct": 100.0 * float(success.mean()),
                "bootstrap_95ci_low_pct": float(low),
                "bootstrap_95ci_high_pct": float(high),
                "scored_n": int(np.isfinite(values).sum()),
                "median_rmsd_A_among_scored": float(np.median(finite)) if len(finite) else np.nan,
            })

    transition_rows: list[dict[str, object]] = []
    for rule, metadata in rule_metadata.items():
        if rule == "global_rank1":
            continue
        for cutoff in (1.0, 2.0, 5.0):
            baseline = success_arrays[("global_rank1", cutoff)]
            selected_success = success_arrays[(rule, cutoff)]
            delta = selected_success.astype(float) - baseline.astype(float)
            sampled = 100.0 * delta[sample_indices].mean(axis=1)
            low, high = np.quantile(sampled, [0.025, 0.975])
            gained = int((~baseline & selected_success).sum())
            lost = int((baseline & ~selected_success).sum())
            try:
                from scipy.stats import binomtest

                mcnemar = float(
                    binomtest(gained, gained + lost, 0.5).pvalue
                    if gained + lost else 1.0
                )
            except ImportError:
                mcnemar = np.nan
            transition_rows.append({
                "selector": rule,
                "uses_ground_truth": metadata["uses_ground_truth"],
                "reference_selector": "global_rank1",
                "cutoff_A": cutoff,
                "delta_percentage_points": 100.0 * float(delta.mean()),
                "paired_bootstrap_95ci_low_pp": float(low),
                "paired_bootstrap_95ci_high_pp": float(high),
                "gained": gained,
                "lost": lost,
                "exact_mcnemar_pvalue": mcnemar,
            })

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    branch.to_csv(output / "cross_branch_input_table.csv", index=False)
    parents.to_csv(output / "cross_branch_per_parent.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output / "cross_branch_metrics.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(output / "cross_branch_transitions.csv", index=False)
    summary = {
        "requested_cohort_n": denominator,
        "available_branch_n": len(branch),
        "complete_top_k_n": int(parents["complete_top_k"].sum()),
        "top_k": args.top_k,
        "incomplete_top_k_policy": (
            "select available branches" if args.allow_incomplete_top_k
            else "count non-rank1 selectors as failure"
        ),
        "selectors": rule_metadata,
        "selection_uses_rmsd": False,
        "oracle_uses_rmsd": True,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nCROSS-BRANCH SELECTOR METRICS")
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print("\nPAIRED CHANGES FROM GLOBAL RANK-1")
    print(pd.DataFrame(transition_rows).to_string(index=False))
    print(f"CASCADE_CROSS_BRANCH_SELECTION_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
