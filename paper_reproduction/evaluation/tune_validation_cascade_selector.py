#!/usr/bin/env python3
"""Tune and freeze a compact Global-preserving cascade selector on validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.compare_global_cascade import summarize


MARGINS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1)
WEIGHTS = (
    (1, 1, 1),
    (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (0, 1, 1), (1, 0, 1), (1, 1, 0),
    (0, 1, 2), (1, 0, 2), (2, 1, 0), (2, 0, 1), (0, 2, 1),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [
        line.split()[0].strip().upper()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Empty or duplicate IDs: {path}")
    return values


def attach_validation_rmsd(
    features: pd.DataFrame,
    cohort: list[str],
    global_audit: pd.DataFrame,
    branch_table: pd.DataFrame,
) -> pd.DataFrame:
    required = {"pdb_id", "candidate", "candidate_id", "global_rank", "basin_q25_ncc"}
    shared = sorted(
        (name for name in features.columns if name.startswith("shared_ncc_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if required - set(features) or len(shared) < 2:
        raise ValueError(f"Candidate features lack required columns/shared scales: {features.columns.tolist()}")
    work = features.copy()
    work["pdb_id"] = work["pdb_id"].astype(str).str.upper()
    work["candidate"] = work["candidate"].astype(str)
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["global_rank"] = pd.to_numeric(work["global_rank"], errors="raise").astype(int)
    for column in ["basin_q25_ncc", *shared]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if set(work.pdb_id) != set(cohort) or work.duplicated(["pdb_id", "candidate"]).any():
        raise ValueError("Candidate-feature parent/candidate identities are incomplete or duplicated")

    audit = global_audit.copy()
    key = next((name for name in ("pdb_id", "complex_name") if name in audit), None)
    if key is None or "global_top1_rmsd_A" not in audit:
        raise ValueError("Global audit lacks parent ID or global_top1_rmsd_A")
    audit["pdb_id"] = audit[key].astype(str).str.upper()
    audit["global_top1_rmsd_A"] = pd.to_numeric(audit["global_top1_rmsd_A"], errors="coerce")
    if audit.pdb_id.duplicated().any() or set(cohort) - set(audit.pdb_id):
        raise ValueError("Global audit coverage differs from cohort")
    global_rmsd = audit.set_index("pdb_id")["global_top1_rmsd_A"]

    branch = branch_table.copy()
    branch_required = {"complex_name", "candidate_id", "selected_pose_rmsd_A", "parent_complex_name", "global_rank"}
    if branch_required - set(branch) or branch.complex_name.astype(str).duplicated().any():
        raise ValueError("Invalid cross-branch input table")
    branch["complex_name"] = branch.complex_name.astype(str)
    branch["candidate_id"] = branch.candidate_id.astype(str)
    branch["parent_complex_name"] = branch.parent_complex_name.astype(str).str.upper()
    branch["global_rank"] = pd.to_numeric(branch.global_rank, errors="raise").astype(int)
    branch["selected_pose_rmsd_A"] = pd.to_numeric(branch.selected_pose_rmsd_A, errors="coerce")
    branch = branch.set_index("complex_name")

    rmsd: list[float] = []
    for row in work.itertuples(index=False):
        if row.candidate == "global":
            if row.global_rank != 0:
                raise ValueError(f"{row.pdb_id}: Global anchor must have global_rank=0")
            value = global_rmsd.get(row.pdb_id, np.nan)
        else:
            if row.candidate not in branch.index:
                raise ValueError(f"Missing branch RMSD: {row.candidate}")
            metric = branch.loc[row.candidate]
            if (
                str(metric.candidate_id) != row.candidate_id
                or str(metric.parent_complex_name) != row.pdb_id
                or int(metric.global_rank) != row.global_rank
            ):
                raise ValueError(f"Branch identity mismatch: {row.candidate}")
            value = metric.selected_pose_rmsd_A
        rmsd.append(float(value) if pd.notna(value) else np.nan)
    work["rmsd_A"] = rmsd
    if not np.isfinite(work.rmsd_A).all():
        failed = work.loc[~np.isfinite(work.rmsd_A), ["pdb_id", "candidate"]].to_dict("records")
        raise ValueError(f"Validation RMSD is incomplete: {failed[:10]}")
    for pdb_id in cohort:
        group = work[work.pdb_id.eq(pdb_id)]
        if len(group) != 6 or group.candidate.eq("global").sum() != 1 or set(group.loc[group.candidate.ne("global"), "global_rank"]) != set(range(1, 6)):
            raise ValueError(f"{pdb_id}: expected one Global plus five Local candidates")
    return work


def choose_parent(group: pd.DataFrame, margin: float, weights: tuple[int, int, int], shared: list[str]) -> pd.Series:
    global_row = group[group.candidate.eq("global")].iloc[0]
    local = group[group.candidate.ne("global")].copy()
    global_values = global_row[shared].to_numpy(float)
    eligible = local[(local[shared].to_numpy(float) > global_values + margin + 1e-12).all(axis=1)].copy()
    if eligible.empty:
        return global_row
    local["shared_mean"] = local[shared].mean(axis=1)
    local["site_rr"] = 1.0 / local.global_rank
    local["basin_rr"] = 1.0 / local.basin_q25_ncc.rank(ascending=False, method="min")
    local["shared_rr"] = 1.0 / local.shared_mean.rank(ascending=False, method="min")
    site_weight, basin_weight, shared_weight = weights
    local["fusion"] = (
        site_weight * local.site_rr
        + basin_weight * local.basin_rr
        + shared_weight * local.shared_rr
    )
    eligible = eligible.merge(
        local[["candidate", "shared_mean", "fusion"]], on="candidate", validate="one_to_one"
    )
    return eligible.sort_values(
        ["fusion", "shared_mean", "global_rank", "candidate"],
        ascending=[False, False, True, True],
        kind="stable",
    ).iloc[0]


def apply_config(features: pd.DataFrame, cohort: list[str], margin: float, weights: tuple[int, int, int]) -> pd.DataFrame:
    shared = sorted(
        (name for name in features.columns if name.startswith("shared_ncc_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    rows = []
    for pdb_id in cohort:
        chosen = choose_parent(features[features.pdb_id.eq(pdb_id)], margin, weights, shared)
        rows.append({
            "pdb_id": pdb_id,
            "selected_candidate": chosen.candidate,
            "selected_pose_path": chosen.pose_path,
            "selected_global_rank": int(chosen.global_rank),
            "selected_global": bool(chosen.candidate == "global"),
            "selected_rmsd_A": float(chosen.rmsd_A),
        })
    return pd.DataFrame(rows)


def config_distance(margin: float, weights: tuple[int, int, int]) -> float:
    values = np.asarray(weights, dtype=float)
    values /= values.sum()
    return float(np.abs(values - 1 / 3).sum() + margin / max(MARGINS))


def exact_mcnemar(gained: int, lost: int) -> float:
    total = gained + lost
    if total == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(total, k) for k in range(min(gained, lost) + 1)) / 2**total)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--candidate-features", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--global-audit", required=True, type=Path)
    parser.add_argument("--branch-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-n", type=int, default=187)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args(argv)
    cohort = read_ids(args.cohort)
    if len(cohort) != args.expected_n or args.bootstrap_repeats < 1:
        raise ValueError(f"Expected {args.expected_n} validation parents, found {len(cohort)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Use an empty tuning output directory: {args.output_dir}")
    seal = json.loads(args.selection_manifest.read_text())
    if seal.get("status") not in {"selection_sealed", "evaluated"}:
        raise ValueError("Candidate features were not sealed before RMSD evaluation")
    if seal.get("features_sha256") != sha256(args.candidate_features):
        raise ValueError("Sealed candidate feature hash mismatch")
    features = attach_validation_rmsd(
        pd.read_csv(args.candidate_features), cohort,
        pd.read_csv(args.global_audit), pd.read_csv(args.branch_table),
    )

    global_by_parent = features[features.candidate.eq("global")].set_index("pdb_id").reindex(cohort)
    global_success2 = global_by_parent.rmsd_A.lt(2.0).to_numpy()
    grid_rows: list[dict[str, object]] = []
    selections: dict[str, pd.DataFrame] = {}
    for margin in MARGINS:
        for weights in WEIGHTS:
            config_id = f"m{margin:g}_w{weights[0]}-{weights[1]}-{weights[2]}"
            selected = apply_config(features, cohort, margin, weights)
            success2 = selected.selected_rmsd_A.lt(2.0).to_numpy()
            success5 = selected.selected_rmsd_A.lt(5.0).to_numpy()
            gained = int((~global_success2 & success2).sum())
            lost = int((global_success2 & ~success2).sum())
            grid_rows.append({
                "config_id": config_id,
                "guard_margin": margin,
                "site_weight": weights[0],
                "basin_weight": weights[1],
                "shared_weight": weights[2],
                "lt2A_count": int(success2.sum()),
                "lt2A_pct": float(100 * success2.mean()),
                "lt5A_count": int(success5.sum()),
                "lt5A_pct": float(100 * success5.mean()),
                "median_rmsd_A": float(selected.selected_rmsd_A.median()),
                "selected_local_n": int((~selected.selected_global).sum()),
                "gained_vs_global_lt2A": gained,
                "lost_vs_global_lt2A": lost,
                "delta_vs_global_lt2A_pp": float(100 * (success2.astype(int) - global_success2.astype(int)).mean()),
                "mcnemar_p_vs_global_lt2A": exact_mcnemar(gained, lost),
                "conservative_distance_from_equal_zero_margin": config_distance(margin, weights),
            })
            selections[config_id] = selected
    grid = pd.DataFrame(grid_rows).sort_values(
        ["lt2A_count", "lt5A_count", "selected_local_n", "conservative_distance_from_equal_zero_margin", "config_id"],
        ascending=[False, False, True, True, True], kind="stable",
    ).reset_index(drop=True)
    winner = grid.iloc[0]
    winner_id = str(winner.config_id)
    selected = selections[winner_id]
    oracle = features.groupby("pdb_id", sort=False).rmsd_A.min().reindex(cohort)
    comparison = pd.DataFrame({
        "global": global_by_parent.rmsd_A.to_numpy(float),
        "validation_frozen_selector": selected.selected_rmsd_A.to_numpy(float),
        "oracle_union6": oracle.to_numpy(float),
    }, index=pd.Index(cohort, name="pdb_id"))
    summary, paired, transitions = summarize(
        comparison, args.bootstrap_repeats, args.seed,
        oracle_methods=("oracle_union6",), reference_methods=("global",),
    )

    args.output_dir.mkdir(parents=True)
    grid.to_csv(args.output_dir / "parameter_grid.csv", index=False)
    selected.to_csv(args.output_dir / "validation_selected_poses.csv", index=False)
    comparison.to_csv(args.output_dir / "validation_per_parent_rmsd.csv")
    summary.to_csv(args.output_dir / "validation_success_summary.csv", index=False)
    paired.to_csv(args.output_dir / "validation_paired_comparisons.csv", index=False)
    transitions.to_csv(args.output_dir / "validation_transitions.csv", index=False)
    frozen = {
        "status": "validation_frozen",
        "selector": "global_preserving_multiscale_guard_rank_fusion",
        "primary_objective": "maximize fixed-denominator validation Top-1 RMSD <2A count",
        "tie_breaking": [
            "maximize <5A count",
            "minimize number of Local selections",
            "prefer parameters closest to equal rank-fusion weights and zero guard margin",
            "lexicographic config_id",
        ],
        "cohort_n": len(cohort),
        "guard": "Local candidate must exceed Global shared NCC at every sealed sigma by strictly more than guard_margin",
        "rank_fusion": "site_weight/global_rank + basin_weight/basin_rank + shared_weight/shared_mean_rank",
        "selected_config": {
            "config_id": winner_id,
            "guard_margin": float(winner.guard_margin),
            "site_weight": int(winner.site_weight),
            "basin_weight": int(winner.basin_weight),
            "shared_weight": int(winner.shared_weight),
        },
        "validation_metrics": {
            "lt2A_count": int(winner.lt2A_count),
            "lt2A_pct": float(winner.lt2A_pct),
            "lt5A_count": int(winner.lt5A_count),
            "lt5A_pct": float(winner.lt5A_pct),
            "selected_local_n": int(winner.selected_local_n),
        },
        "grid": {"guard_margins": list(MARGINS), "rank_fusion_weights": [list(item) for item in WEIGHTS]},
        "input_hashes": {
            str(path.resolve()): sha256(path)
            for path in (args.cohort, args.candidate_features, args.selection_manifest, args.global_audit, args.branch_table)
        },
        "code_sha256": sha256(Path(__file__)),
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.seed,
        "test_outcomes_used": False,
    }
    (args.output_dir / "frozen_selector.json").write_text(json.dumps(frozen, indent=2) + "\n")
    report = [
        "# Validation-frozen Global-to-Local selector",
        "",
        f"Selected {winner_id} on N={len(cohort)} validation parents.",
        "",
        "The primary objective was fixed-denominator <2 A success; <5 A and conservative switching were declared tie-breakers.",
        "",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "The selected validation score is tuning performance. The frozen JSON must be applied once to test candidate features without reading test RMSD.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(grid.head(15).to_string(index=False))
    print(summary.to_string(index=False))
    print(f"VALIDATION_SELECTOR_FROZEN: {args.output_dir / 'frozen_selector.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
