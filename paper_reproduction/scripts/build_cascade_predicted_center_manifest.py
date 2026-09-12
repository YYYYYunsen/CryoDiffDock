#!/usr/bin/env python3
"""Build GT-free Local input rows from ranked CryoDiffDock-Global poses.

Each ranked Global pose supplies both the local crop centre and the ligand-shaped
neighbourhood used to select flexible protein side chains.  Reference ligand
coordinates are copied only to a separate reveal manifest and are never read to
construct a Local input row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cryodiffdock_pocket.density_preparation import (  # noqa: E402
    STANDARD_PROTEIN_RESIDUES,
    read_ligand_heavy_coordinates,
)


ID_COLUMNS = ("complex", "complex_name", "pdb_id", "complex_key")
POSE_COLUMNS = ("sdf_path", "prediction_path", "pose_path")


def choose(columns, candidates, label):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {label} column; available columns: {list(columns)}")


def normalize(value: object) -> str:
    return str(value).strip().upper()


def read_ids(path: Path) -> list[str]:
    values = [
        normalize(line.split()[0])
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError("Cohort contains duplicate IDs")
    return values


def flexible_residues(
    protein_path: Path, ligand_coordinates: np.ndarray, cutoff: float
) -> str:
    import gemmi

    structure = gemmi.read_structure(str(protein_path))
    selected: list[str] = []
    seen: set[str] = set()
    cutoff_squared = cutoff**2
    backbone = {"N", "CA", "C", "O", "OXT"}
    no_flexible_chi = {"ALA", "GLY", "PRO"}
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_name = residue.name.upper()
                if (
                    residue_name not in STANDARD_PROTEIN_RESIDUES
                    or residue_name in no_flexible_chi
                ):
                    continue
                coordinates = np.asarray(
                    [
                        (atom.pos.x, atom.pos.y, atom.pos.z)
                        for atom in residue
                        if atom.element.name not in {"H", "D"}
                        and atom.name.strip().upper() not in backbone
                    ],
                    dtype=np.float32,
                )
                if coordinates.size == 0:
                    continue
                delta = coordinates[:, None, :] - ligand_coordinates[None, :, :]
                if float(np.square(delta).sum(axis=-1).min()) > cutoff_squared:
                    continue
                key = f"{chain.name}:{residue.seqid.num}"
                if key not in seen:
                    seen.add(key)
                    selected.append(key)
        break
    if not selected:
        raise ValueError(
            f"No torsion-bearing protein side chain within {cutoff:.1f} A "
            "of the Global pose"
        )
    return "-".join(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-scores", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--score-column", default="combined_at_best_weights")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--flexible-distance", type=float, default=3.5)
    parser.add_argument(
        "--no-flexible-policy", choices=("rigid", "error"), default="rigid",
        help="Use an explicit zero-flexible-residue branch, or fail the parent.",
    )
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0 or args.flexible_distance <= 0:
        raise ValueError("top-k and flexible-distance must be positive")

    global_path = args.global_scores.expanduser().resolve()
    source_path = args.source_manifest.expanduser().resolve()
    global_scores = pd.read_csv(global_path)
    source = pd.read_csv(source_path, dtype={"complex_name": str})
    global_id = choose(global_scores.columns, ID_COLUMNS, "Global complex ID")
    source_id = choose(source.columns, ("complex_name", "pdb_id"), "source complex ID")
    pose_column = choose(global_scores.columns, POSE_COLUMNS, "Global pose path")
    if args.score_column not in global_scores.columns:
        raise ValueError(f"Missing Global ranking score {args.score_column!r}")
    # Outcome columns such as RMSD may coexist in historical analysis tables.
    # Restrict the in-memory table to deployment-time fields before ranking.
    deployment_columns = [global_id, args.score_column, pose_column]
    if "original_rank" in global_scores.columns:
        deployment_columns.append("original_rank")
    global_scores = global_scores[deployment_columns].copy()
    required_source = {
        "experimental_protein", "ligand", "map_path", "mask_structure"
    }
    missing = sorted(required_source - set(source.columns))
    if missing:
        raise ValueError(f"Source manifest is missing columns: {missing}")

    global_scores["_pdb_id"] = global_scores[global_id].map(normalize)
    global_scores["_score"] = pd.to_numeric(
        global_scores[args.score_column], errors="coerce"
    )
    source["_pdb_id"] = source[source_id].map(normalize)
    if source["_pdb_id"].duplicated().any():
        raise ValueError("Source manifest contains duplicate complex IDs")
    source = source.set_index("_pdb_id")
    cohort = read_ids(args.cohort.expanduser().resolve())

    blind_rows: list[dict[str, object]] = []
    reveal_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    forbidden = {"reference_ligand", "flexible_sidechains", "_pdb_id"}

    for cohort_index, pdb_id in enumerate(cohort):
        try:
            if pdb_id not in source.index:
                raise KeyError("missing source-manifest row")
            ranked = global_scores[
                global_scores["_pdb_id"].eq(pdb_id)
                & np.isfinite(global_scores["_score"])
            ].copy()
            tie_columns = ["_score"]
            ascending = [False]
            if "original_rank" in ranked.columns:
                ranked["_original_rank"] = pd.to_numeric(
                    ranked["original_rank"], errors="coerce"
                )
                tie_columns.append("_original_rank")
                ascending.append(True)
            ranked = ranked.sort_values(tie_columns, ascending=ascending, kind="stable")
            if len(ranked) < args.top_k:
                raise ValueError(
                    f"only {len(ranked)} finite Global poses; need {args.top_k}"
                )
            source_row = source.loc[pdb_id]
            protein_path = Path(str(source_row["experimental_protein"])).expanduser()
            if not protein_path.is_file():
                raise FileNotFoundError(protein_path)

            parent_blind_rows: list[dict[str, object]] = []
            parent_reveal_rows: list[dict[str, object]] = []
            parent_audit_rows: list[dict[str, object]] = []

            for global_rank, (_, pose_row) in enumerate(
                ranked.head(args.top_k).iterrows(), start=1
            ):
                pose_path = Path(str(pose_row[pose_column])).expanduser()
                if not pose_path.is_file():
                    raise FileNotFoundError(pose_path)
                coordinates = read_ligand_heavy_coordinates(pose_path)
                center = coordinates.mean(axis=0)
                try:
                    flexible = flexible_residues(
                        protein_path, coordinates, args.flexible_distance
                    )
                    sidechain_policy = "global_pose_atoms"
                    num_flexible = len(flexible.split("-"))
                except ValueError:
                    if args.no_flexible_policy == "error":
                        raise
                    # A nonmatching sentinel keeps predefined selection active
                    # while producing a zero-node flexible-residue store.  The
                    # model already has an explicit zero-flexible-bond path.
                    flexible = "__NONE__:0"
                    sidechain_policy = "rigid_receptor_fallback"
                    num_flexible = 0
                branch_name = f"{pdb_id}__global_rank{global_rank:02d}"
                record = {
                    key: value
                    for key, value in source_row.to_dict().items()
                    if key not in forbidden
                }
                record.update(
                    {
                        "complex_name": branch_name,
                        "parent_complex_name": pdb_id,
                        "global_rank": global_rank,
                        "global_score": float(pose_row["_score"]),
                        "global_pose": str(pose_path.resolve()),
                        "global_original_rank": pose_row.get("original_rank", np.nan),
                        "inference_seed": int(
                            args.seed + cohort_index * args.top_k + global_rank - 1
                        ),
                        "pocket_center_x": float(center[0]),
                        "pocket_center_y": float(center[1]),
                        "pocket_center_z": float(center[2]),
                        "flexible_sidechains": flexible,
                        "center_source": "global_predicted_ligand_heavy_atom_centroid",
                        "sidechain_source": "global_predicted_ligand_atoms",
                        "sidechain_policy": sidechain_policy,
                    }
                )
                parent_blind_rows.append(record)
                reveal = dict(record)
                if "reference_ligand" in source_row.index:
                    reveal["reference_ligand"] = source_row["reference_ligand"]
                parent_reveal_rows.append(reveal)
                parent_audit_rows.append(
                    {
                        "parent_complex_name": pdb_id,
                        "complex_name": branch_name,
                        "global_rank": global_rank,
                        "global_score": float(pose_row["_score"]),
                        "global_pose": str(pose_path.resolve()),
                        "num_flexible_sidechains": num_flexible,
                        "sidechain_policy": sidechain_policy,
                        "status": "ok",
                    }
                )
            # Commit a parent only after every requested Global branch passed.
            # This prevents a late branch failure from creating an unequal-K case.
            blind_rows.extend(parent_blind_rows)
            reveal_rows.extend(parent_reveal_rows)
            audit_rows.extend(parent_audit_rows)
        except Exception as exc:
            failure = {
                "parent_complex_name": pdb_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            if args.fail_on_error:
                raise

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(blind_rows).to_csv(output / "cascade_source_blind.csv", index=False)
    pd.DataFrame(reveal_rows).to_csv(output / "cascade_source_reveal.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output / "cascade_branch_audit.csv", index=False)
    pd.DataFrame(failures, columns=["parent_complex_name", "error"]).to_csv(
        output / "failures.csv", index=False
    )
    completed_ids = {str(row["parent_complex_name"]) for row in blind_rows}
    summary = {
        "requested_cohort_n": len(cohort),
        "complete_parent_complexes_n": len(completed_ids),
        "failed_parent_complexes_n": len(failures),
        "top_k": args.top_k,
        "base_seed": args.seed,
        "branch_seed_rule": "base_seed + cohort_index * top_k + global_rank - 1",
        "branch_rows_n": len(blind_rows),
        "global_ranking_score": args.score_column,
        "global_ranking_direction": "descending",
        "center_definition": "Global predicted ligand heavy-atom centroid",
        "flexible_sidechain_definition": (
            f"torsion-bearing side chains within {args.flexible_distance:g} A "
            "of Global predicted ligand heavy atoms"
        ),
        "no_flexible_sidechain_policy": args.no_flexible_policy,
        "uses_reference_coordinates_for_generation": False,
        "reference_manifest_is_separate": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"CASCADE_SOURCE_OK: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
