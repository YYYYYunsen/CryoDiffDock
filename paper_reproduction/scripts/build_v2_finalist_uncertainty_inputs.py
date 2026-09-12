#!/usr/bin/env python3
"""Assemble materialized Local V2 finalists for a frozen uncertainty audit.

Expanded-pool roots are read in priority order.  For each cohort member, the
first root containing a complete, internally consistent finalist set is used.
The selected finalist is checked against the sibling ``v2_selected`` ranking.
No reference RMSD is used to construct or choose the finalist ensemble.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex_id", "id")
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "ok"}


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(TRUE_VALUES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize_id(line.split("#", 1)[0])
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return values


def id_column(table: pd.DataFrame, label: str) -> str:
    column = next((value for value in ID_COLUMNS if value in table.columns), None)
    if column is None:
        raise ValueError(f"{label} has no recognized ID column: {list(table.columns)}")
    return column


def endpoint_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    column = id_column(table, "endpoint table")
    required = {"v2_scored", "v2_rmsd_A"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Endpoint table missing columns: {missing}")
    table = table.copy()
    table["complex_name"] = table[column].map(normalize_id)
    if table["complex_name"].duplicated().any():
        raise ValueError("Endpoint table contains duplicate complex IDs")
    table["v2_scored"] = truthy(table["v2_scored"])
    table["v2_rmsd_A"] = pd.to_numeric(table["v2_rmsd_A"], errors="coerce")
    return table[["complex_name", "v2_scored", "v2_rmsd_A"]]


def resolve_materialized_path(value: object, ranking_path: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ranking_path.parent / path).resolve()


def selected_candidate_id(expanded_root: Path, directory_name: str) -> str:
    path = expanded_root.parent / "v2_selected" / directory_name / "ranking.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing V2-selected ranking: {path}")
    table = pd.read_csv(path)
    if len(table) != 1 or "candidate_id" not in table.columns:
        raise ValueError(f"Expected one V2-selected candidate in {path}")
    return str(table.iloc[0]["candidate_id"])


def read_finalists(ranking_path: Path) -> tuple[str, pd.DataFrame]:
    table = pd.read_csv(ranking_path)
    required = {
        "complex_name", "candidate_id", "final_selection_score",
        "materialized_path", "v2_finalist",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{ranking_path} missing columns: {missing}")
    names = table["complex_name"].map(normalize_id).unique()
    if len(names) != 1:
        raise ValueError(f"Expected one complex in {ranking_path}, found {list(names)}")
    finalists = table[truthy(table["v2_finalist"])].copy()
    finalists["final_selection_score"] = pd.to_numeric(
        finalists["final_selection_score"], errors="coerce"
    )
    finalists = finalists[np.isfinite(finalists["final_selection_score"])].copy()
    if finalists.empty:
        raise ValueError(f"No finite V2 finalists in {ranking_path}")
    if finalists["candidate_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate finalist candidate IDs in {ranking_path}")
    finalists["prediction_path"] = finalists["materialized_path"].map(
        lambda value: str(resolve_materialized_path(value, ranking_path))
    )
    missing_paths = [value for value in finalists["prediction_path"] if not Path(value).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} materialized finalist files are missing; first: {missing_paths[0]}"
        )
    finalists = finalists.sort_values(
        ["final_selection_score", "candidate_id"],
        ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    finalists["source_rank"] = np.arange(1, len(finalists) + 1)
    return str(names[0]), finalists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--endpoint-table", required=True, type=Path)
    parser.add_argument("--expanded-pool-root", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-cohort-n", type=int, default=174)
    args = parser.parse_args()

    cohort_path = args.cohort.expanduser().resolve()
    endpoint_path = args.endpoint_table.expanduser().resolve()
    roots = [path.expanduser().resolve() for path in args.expanded_pool_root]
    cohort = read_ids(cohort_path)
    if len(cohort) != args.expected_cohort_n:
        raise ValueError(f"Expected {args.expected_cohort_n} cohort IDs, found {len(cohort)}")
    cohort_set = set(cohort)
    endpoints = endpoint_table(endpoint_path).set_index("complex_name")
    missing_endpoint = sorted(cohort_set - set(endpoints.index))
    if missing_endpoint:
        raise ValueError(f"Endpoint table lacks cohort IDs: {missing_endpoint}")

    candidates: dict[str, list[tuple[int, Path, pd.DataFrame]]] = {}
    rejected_rows = []
    for priority, root in enumerate(roots):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for ranking_path in sorted(root.glob("index*___*/ranking.csv")):
            try:
                name, finalists = read_finalists(ranking_path)
                if name not in cohort_set:
                    continue
                expected_selected = selected_candidate_id(root, ranking_path.parent.name)
                observed_selected = str(finalists.iloc[0]["candidate_id"])
                if observed_selected != expected_selected:
                    raise ValueError(
                        f"Final-score rank 1 {observed_selected!r} differs from "
                        f"V2-selected {expected_selected!r}"
                    )
                candidates.setdefault(name, []).append(
                    (priority, ranking_path, finalists)
                )
            except Exception as exc:
                rejected_rows.append({
                    "expanded_pool_root": str(root),
                    "ranking_path": str(ranking_path),
                    "error": f"{type(exc).__name__}: {exc}",
                })

    pose_rows = []
    label_rows = []
    case_rows = []
    for cohort_order, name in enumerate(cohort):
        choices = sorted(candidates.get(name, []), key=lambda item: item[0])
        if not choices:
            case_rows.append({
                "cohort_order": cohort_order,
                "complex_name": name,
                "status": "MISSING_FINALIST_ENSEMBLE",
                "source_priority": np.nan,
                "source_ranking": "",
                "finalist_count": 0,
                "v2_scored": bool(endpoints.loc[name, "v2_scored"]),
                "v2_rmsd_A": endpoints.loc[name, "v2_rmsd_A"],
            })
            continue
        priority, ranking_path, finalists = choices[0]
        endpoint_scored = bool(endpoints.loc[name, "v2_scored"])
        endpoint_rmsd = float(endpoints.loc[name, "v2_rmsd_A"])
        if not endpoint_scored or not np.isfinite(endpoint_rmsd):
            case_rows.append({
                "cohort_order": cohort_order,
                "complex_name": name,
                "status": "FINALISTS_EXIST_BUT_ENDPOINT_UNSCORED",
                "source_priority": priority,
                "source_ranking": str(ranking_path),
                "finalist_count": len(finalists),
                "v2_scored": endpoint_scored,
                "v2_rmsd_A": endpoint_rmsd,
            })
            continue
        case_rows.append({
            "cohort_order": cohort_order,
            "complex_name": name,
            "status": "READY",
            "source_priority": priority,
            "source_ranking": str(ranking_path),
            "finalist_count": len(finalists),
            "v2_scored": endpoint_scored,
            "v2_rmsd_A": endpoint_rmsd,
        })
        for _, row in finalists.iterrows():
            rank = int(row["source_rank"])
            pose_rows.append({
                "complex_name": name,
                "source": "v2_finalist",
                "source_rank": rank,
                "combined_score": float(row["final_selection_score"]),
                "candidate_id": str(row["candidate_id"]),
                "prediction_path": str(row["prediction_path"]),
                "source_ranking": str(ranking_path),
            })
            label_rows.append({
                "complex_name": name,
                "prediction_path": str(row["prediction_path"]),
                "ligand_rmsd_A": endpoint_rmsd if rank == 1 else np.nan,
                "label_role": "v2_selected_top1" if rank == 1 else "not_labelled",
            })

    cases = pd.DataFrame(case_rows)
    poses = pd.DataFrame(pose_rows)
    labels = pd.DataFrame(label_rows)
    ready = cases[cases["status"].eq("READY")]
    count_distribution = {
        str(int(count)): int(number)
        for count, number in ready["finalist_count"].value_counts().sort_index().items()
    }
    summary = {
        "cohort_n": len(cohort),
        "ready_case_n": len(ready),
        "missing_or_unscored_case_n": len(cases) - len(ready),
        "status_counts": {
            str(key): int(value) for key, value in cases["status"].value_counts().items()
        },
        "finalist_count_distribution": count_distribution,
        "pose_rows_n": len(poses),
        "source_priority_policy": "first complete root in command-line order",
        "source_roots": [str(root) for root in roots],
        "duplicate_source_case_n": int(sum(len(value) > 1 for value in candidates.values())),
        "rejected_ranking_n": len(rejected_rows),
        "uses_reference_for_finalist_selection": False,
        "label_policy": (
            "Only final-selection rank 1 receives the already frozen V2 Top-1 RMSD; "
            "all other finalists remain unlabelled."
        ),
        "inputs": {
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
            "endpoint_table": {"path": str(endpoint_path), "sha256": sha256(endpoint_path)},
        },
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    poses.to_csv(output / "v2_finalist_pose_scores.csv", index=False)
    labels.to_csv(output / "v2_finalist_top1_labels.csv", index=False)
    cases.to_csv(output / "v2_finalist_case_audit.csv", index=False)
    pd.DataFrame(rejected_rows).to_csv(output / "rejected_rankings.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nINCOMPLETE CASES")
    print(cases[~cases["status"].eq("READY")].to_string(index=False))
    print(f"\nV2_FINALIST_UNCERTAINTY_INPUTS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
