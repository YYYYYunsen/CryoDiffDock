#!/usr/bin/env python3
"""Build frozen ChemEM Stage-1 pose ensembles for a reliability audit.

The builder never reads per-pose RMSD when selecting an ensemble.  It takes
the first K entries from each ChemEM ``fitting/results.txt`` file, requires all
K corresponding SDF files to exist, and only then attaches the already frozen
ChemEM Top-1 RMSD as a retrospective outcome label.  Missing or invalid poses
within the first K are not replaced by lower-ranked poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLUMNS = ("complex_name", "pdb_id", "pdb_id_upper", "complex", "name", "id")
RESULT_LINE = re.compile(
    r"^\s*(?P<iteration>\d+)\s*:\s*(?P<score>[-+0-9.eE]+)\s*$"
)
TRUE_VALUES = {"1", "true", "yes", "y", "complete", "completed", "success", "ok"}
TOP1_RMSD_COLUMNS = ("chemem_top1_rmsd_A", "top1_rmsd_A")
TOP1_ITERATION_COLUMNS = ("chemem_top1_iteration", "top1_iteration")
RESULTS_COLUMNS = ("chemem_results_path", "results_path")
STAGE1_COLUMNS = ("chemem_stage1_output", "stage1_output")


def normalize_id(value: object) -> str:
    return str(value).strip().upper()


def choose_column(columns, candidates, label: str) -> str:
    column = next((value for value in candidates if value in columns), None)
    if column is None:
        raise ValueError(f"No {label} column; found {list(columns)}")
    return column


def finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cohort(path: Path) -> list[str]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=str)
        column = choose_column(table.columns, ID_COLUMNS, "cohort ID")
        values = [normalize_id(value) for value in table[column] if str(value).strip()]
    else:
        values = [
            normalize_id(line.split("#", 1)[0])
            for line in path.read_text().splitlines()
            if line.split("#", 1)[0].strip()
        ]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"Cohort is empty or contains duplicate IDs: {path}")
    return values


def read_endpoint_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str)
    id_column = choose_column(table.columns, ID_COLUMNS, "endpoint ID")
    rmsd_column = choose_column(table.columns, TOP1_RMSD_COLUMNS, "ChemEM Top-1 RMSD")
    iteration_column = next(
        (value for value in TOP1_ITERATION_COLUMNS if value in table.columns), None
    )
    table = table.copy()
    table["complex_name"] = table[id_column].map(normalize_id)
    if table["complex_name"].duplicated().any():
        duplicates = table.loc[
            table["complex_name"].duplicated(False), "complex_name"
        ].unique().tolist()
        raise ValueError(f"Endpoint table contains duplicate IDs: {duplicates[:20]}")
    table["_top1_rmsd_A"] = pd.to_numeric(table[rmsd_column], errors="coerce")
    table["_top1_iteration"] = (
        pd.to_numeric(table[iteration_column], errors="coerce")
        if iteration_column is not None else np.nan
    )
    complete_column = next(
        (value for value in ("chemem_complete", "complete", "status") if value in table.columns),
        None,
    )
    table["_complete"] = (
        table[complete_column].map(truthy)
        if complete_column is not None else table["_top1_rmsd_A"].notna()
    )
    table["_table_path"] = str(path)
    return table.set_index("complex_name", drop=False)


def resolve_record_path(value: object, table_path: Path) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (table_path.parent / path).resolve()


def result_candidates(record: pd.Series, roots: list[Path], complex_name: str) -> list[Path]:
    candidates: list[Path] = []
    table_path = Path(str(record["_table_path"]))
    for column in RESULTS_COLUMNS:
        if column in record:
            value = resolve_record_path(record.get(column), table_path)
            if value is not None:
                candidates.append(value)
    for column in STAGE1_COLUMNS:
        if column in record:
            value = resolve_record_path(record.get(column), table_path)
            if value is not None:
                candidates.append(value / "fitting" / "results.txt")
    for root in roots:
        candidates.extend([
            root / "runs" / complex_name / "stage1" / "fitting" / "results.txt",
            root / "01_stage1" / "runs" / complex_name / "stage1" / "fitting" / "results.txt",
            root / "01_stage1" / "outputs" / complex_name / "stage1" / "fitting" / "results.txt",
        ])
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def recursive_result_candidates(roots: list[Path], complex_name: str) -> list[Path]:
    """Fallback for historical roots whose directory layout predates ChemEM100."""
    matches = []
    seen = set()
    for root in roots:
        for path in sorted(root.glob(f"**/{complex_name}/stage1/fitting/results.txt")):
            resolved = path.resolve()
            if resolved not in seen:
                matches.append(resolved)
                seen.add(resolved)
    return matches


def parse_ranked_results(path: Path) -> list[tuple[int, int, float]]:
    ranked = []
    for rank, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        match = RESULT_LINE.match(line)
        if match is None:
            continue
        ranked.append((rank, int(match.group("iteration")), float(match.group("score"))))
    if not ranked:
        raise ValueError(f"No ranked records in {path}")
    return ranked


def pose_for_iteration(fitting: Path, iteration: int) -> Path:
    exact = fitting / f"Ligand_{iteration}_refined.sdf"
    if exact.is_file():
        return exact.resolve()
    matches = sorted(fitting.glob(f"Ligand_{iteration}_*_refined.sdf"))
    matches = [path.resolve() for path in matches if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one refined pose for iteration {iteration}; found {len(matches)}"
        )
    return matches[0]


def validate_sdf(path: Path) -> None:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read SDF pose: {path}")
    heavy = Chem.RemoveHs(molecule, sanitize=False)
    coordinates = np.asarray(heavy.GetConformer().GetPositions(), dtype=float)
    if heavy.GetNumAtoms() == 0 or not np.isfinite(coordinates).all():
        raise ValueError(f"Pose has no finite heavy-atom coordinates: {path}")


def inspect_results_path(
    path: Path,
    top_k: int,
    expected_top1_iteration: float,
    validate_pose_files: bool = False,
) -> tuple[list[dict[str, object]], str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], "results.txt missing or empty"
    try:
        ranked = parse_ranked_results(path)
        if len(ranked) < top_k:
            raise ValueError(f"only {len(ranked)} ranked results, requires {top_k}")
        selected = ranked[:top_k]
        if np.isfinite(expected_top1_iteration) and selected[0][1] != int(expected_top1_iteration):
            raise ValueError(
                f"rank-1 iteration {selected[0][1]} differs from frozen endpoint "
                f"iteration {int(expected_top1_iteration)}"
            )
        rows = []
        for rank, iteration, chemem_score in selected:
            pose = pose_for_iteration(path.parent, iteration)
            if validate_pose_files:
                validate_sdf(pose)
            rows.append({
                "source_rank": rank,
                "iteration": iteration,
                "chemem_score": chemem_score,
                "prediction_path": str(pose),
            })
        return rows, None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--endpoint-table", required=True, type=Path)
    parser.add_argument("--chemem-root", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expected-cohort-n", type=int)
    parser.add_argument("--expected-ready-n", type=int)
    parser.add_argument(
        "--validate-sdf",
        action="store_true",
        help="Require every frozen Top-K SDF to contain finite heavy-atom coordinates.",
    )
    args = parser.parse_args()
    if args.top_k < 2:
        raise ValueError("top-k must be at least 2")

    cohort_path = args.cohort.expanduser().resolve()
    endpoint_path = args.endpoint_table.expanduser().resolve()
    roots = [path.expanduser().resolve() for path in args.chemem_root]
    cohort = read_cohort(cohort_path)
    if args.expected_cohort_n is not None and len(cohort) != args.expected_cohort_n:
        raise ValueError(f"Expected {args.expected_cohort_n} cohort cases, found {len(cohort)}")
    endpoints = read_endpoint_table(endpoint_path)

    case_rows: list[dict[str, object]] = []
    pose_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    for cohort_order, complex_name in enumerate(cohort):
        if complex_name not in endpoints.index:
            case_rows.append({
                "cohort_order": cohort_order,
                "complex_name": complex_name,
                "status": "MISSING_ENDPOINT_ROW",
            })
            continue
        record = endpoints.loc[complex_name]
        top1_rmsd = finite_number(record["_top1_rmsd_A"])
        if not bool(record["_complete"]) or not np.isfinite(top1_rmsd):
            case_rows.append({
                "cohort_order": cohort_order,
                "complex_name": complex_name,
                "status": "CHEMEM_INCOMPLETE_OR_UNSCORED",
                "top1_rmsd_A": top1_rmsd,
            })
            continue

        choices = result_candidates(record, roots, complex_name)
        selected_path = None
        selected_rows = None
        for priority, path in enumerate(choices):
            rows, error = inspect_results_path(
                path, args.top_k, finite_number(record["_top1_iteration"]), args.validate_sdf
            )
            if error is None:
                selected_path, selected_rows = path, rows
                selected_priority = priority
                break
            if path.exists():
                rejected_rows.append({
                    "complex_name": complex_name,
                    "candidate_priority": priority,
                    "results_path": str(path),
                    "error": error,
                })
        if selected_rows is None:
            direct_set = set(choices)
            recursive = [
                path for path in recursive_result_candidates(roots, complex_name)
                if path not in direct_set
            ]
            for offset, path in enumerate(recursive, start=len(choices)):
                rows, error = inspect_results_path(
                    path, args.top_k, finite_number(record["_top1_iteration"]), args.validate_sdf
                )
                if error is None:
                    selected_path, selected_rows = path, rows
                    selected_priority = offset
                    break
                rejected_rows.append({
                    "complex_name": complex_name,
                    "candidate_priority": offset,
                    "results_path": str(path),
                    "error": error,
                })
        if selected_rows is None:
            case_rows.append({
                "cohort_order": cohort_order,
                "complex_name": complex_name,
                "status": "NO_COMPLETE_TOP_K_ENSEMBLE",
                "top1_rmsd_A": top1_rmsd,
                "candidate_results_paths_n": len(choices),
            })
            continue

        case_rows.append({
            "cohort_order": cohort_order,
            "complex_name": complex_name,
            "status": "READY",
            "top1_rmsd_A": top1_rmsd,
            "top_k": args.top_k,
            "results_path": str(selected_path),
            "source_priority": selected_priority,
        })
        for row in selected_rows:
            rank = int(row["source_rank"])
            pose_rows.append({
                "complex_name": complex_name,
                "source": "chemem_stage1",
                "source_rank": rank,
                # Preserve the authoritative results.txt ordering without
                # assuming whether raw ChemEM scores are maximized or minimized.
                "selection_score": -float(rank),
                "chemem_score": row["chemem_score"],
                "iteration": row["iteration"],
                "prediction_path": row["prediction_path"],
                "results_path": str(selected_path),
            })
            label_rows.append({
                "complex_name": complex_name,
                "prediction_path": row["prediction_path"],
                "ligand_rmsd_A": top1_rmsd if rank == 1 else np.nan,
                "label_role": "chemem_stage1_top1" if rank == 1 else "not_labelled",
            })

    cases = pd.DataFrame(case_rows)
    poses = pd.DataFrame(pose_rows)
    labels = pd.DataFrame(label_rows)
    ready = cases[cases["status"].eq("READY")]
    if args.expected_ready_n is not None and len(ready) != args.expected_ready_n:
        raise ValueError(f"Expected {args.expected_ready_n} ready cases, found {len(ready)}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_output = output / "chemem_stage1_pose_scores.csv"
    label_output = output / "chemem_stage1_top1_labels.csv"
    case_output = output / "chemem_stage1_case_audit.csv"
    rejected_output = output / "rejected_results_paths.csv"
    ready_output = output / "ready_ids.txt"
    poses.to_csv(pose_output, index=False)
    labels.to_csv(label_output, index=False)
    cases.to_csv(case_output, index=False)
    pd.DataFrame(rejected_rows).to_csv(rejected_output, index=False)
    ready_output.write_text(
        "\n".join(ready.sort_values("cohort_order")["complex_name"]) + ("\n" if len(ready) else "")
    )
    summary = {
        "cohort_n": len(cohort),
        "ready_case_n": len(ready),
        "incomplete_or_unavailable_case_n": len(cases) - len(ready),
        "status_counts": {
            str(key): int(value) for key, value in cases["status"].value_counts().items()
        },
        "top_k": args.top_k,
        "sdf_validation": args.validate_sdf,
        "pose_rows_n": len(poses),
        "ensemble_policy": (
            "First K entries in ChemEM fitting/results.txt; missing/invalid entries "
            "within Top-K are not replaced by lower-ranked poses."
        ),
        "selection_score_policy": (
            "Negative ordinal rank preserves the authoritative results.txt order; "
            "raw ChemEM scores are retained for audit but not reinterpreted."
        ),
        "label_policy": (
            "The frozen ChemEM Top-1 RMSD is attached only after the Top-K ensemble "
            "has been selected without per-pose RMSD."
        ),
        "analysis_role": "supplementary_complete_case_external_control",
        "inputs": {
            "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path)},
            "endpoint_table": {"path": str(endpoint_path), "sha256": sha256(endpoint_path)},
            "chemem_roots": [str(path) for path in roots],
        },
        "output_sha256": {
            "pose_scores": sha256(pose_output),
            "top1_labels": sha256(label_output),
            "case_audit": sha256(case_output),
            "ready_ids": sha256(ready_output),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nINCOMPLETE CASES")
    print(cases[~cases["status"].eq("READY")].to_string(index=False))
    print(f"\nCHEMEM_STAGE1_UNCERTAINTY_INPUTS_OK: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
