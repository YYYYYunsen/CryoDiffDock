#!/usr/bin/env python3
"""Test whether Q-score helps choose poses within one frozen inference run.

Prepare a fixed candidate pool, score it without reference structures,
then reveal reference RMSD only in the separate evaluate command. This is a
validation diagnostic; it does not replace the frozen PAAHS Top-1 selector.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.paahs_figure5_mode_qscore import map_lookup
from evaluation.qscore_core import (
    QScoreParameters, atom_qscores, read_density_map,
    read_protein_heavy_atoms, read_sdf_heavy_atoms,
)

FORBIDDEN = ("reference", "native", "rmsd", "outcome", "success")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def blind_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"complex_name": str})
    if any(any(token in column.lower() for token in FORBIDDEN) for column in frame):
        raise ValueError(f"Outcome-bearing column in blind input: {path}")
    frame["complex_name"] = frame.complex_name.str.strip().str.upper()
    return frame


def parent_map_name(case: str) -> str:
    """Cascade branches share the experimental map of their parent complex."""
    return re.sub(r"__GLOBAL_RANK\d+$", "", case)


def select_top_k(scores: pd.DataFrame, manifest: pd.DataFrame, maps: dict,
                 ncc_columns: list[str], top_k: int) -> pd.DataFrame:
    if top_k < 0 or top_k == 1:
        raise ValueError("top-k must be 0 (all poses) or at least 2")
    needed = {"complex_name", "prediction_path", "source", "source_rank", *ncc_columns}
    if missing := needed - set(scores):
        raise ValueError(f"Pose scores lack {sorted(missing)}")
    if manifest.complex_name.duplicated().any():
        raise ValueError("Duplicate cases in blind manifest")
    protein_col = next((c for c in ("protein_path", "experimental_protein") if c in manifest), None)
    if protein_col is None:
        raise ValueError("Blind manifest needs protein_path or experimental_protein")
    cohort = set(manifest.complex_name)
    if missing := cohort - set(scores.complex_name):
        raise ValueError(f"Pose scores lack blind-manifest cases: {sorted(missing)[:10]}")
    # The frozen val99 manifest can be a subset of its 100-case source pool.
    # Filter by the predetermined blind cohort, never by RMSD or score success.
    scores = scores[scores.complex_name.isin(cohort)].copy()
    if missing := {case for case in cohort if case not in maps and parent_map_name(case) not in maps}:
        raise ValueError(f"No raw map for {sorted(missing)[:10]}")
    scores = scores.copy()
    for column in ncc_columns:
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    finite_ncc = np.isfinite(scores[ncc_columns].to_numpy(float)).all(axis=1)
    invalid_counts = (scores.loc[~finite_ncc].groupby("complex_name").size().to_dict())
    if invalid_counts:
        print(
            f"Found {sum(invalid_counts.values())} candidates with missing/nonfinite "
            f"NCC across {len(invalid_counts)} cases; first cases: "
            f"{dict(list(sorted(invalid_counts.items()))[:10])}",
            flush=True,
        )
    if top_k:
        scores = scores.loc[finite_ncc].copy()
    required_count = top_k if top_k else 2
    candidate_counts = scores.groupby("complex_name").size().reindex(sorted(cohort), fill_value=0)
    short = candidate_counts[candidate_counts.lt(required_count)]
    if len(short):
        raise ValueError(
            f"Cases with fewer than {required_count} candidates: "
            f"{short.head(20).to_dict()}"
        )
    scores["multiscale_ncc"] = scores[ncc_columns].mean(axis=1, skipna=False)
    scores["candidate_id"] = scores.source.astype(str) + "__" + scores.source_rank.astype(str)
    if scores.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Candidate IDs are not unique within a case")
    proteins = manifest.set_index("complex_name")[protein_col].to_dict()
    rows = []
    for case, group in scores.groupby("complex_name", sort=True):
        chosen = group.sort_values(["multiscale_ncc", "candidate_id"],
                                   ascending=[False, True], kind="stable", na_position="last")
        if top_k:
            chosen = chosen.head(top_k)
        for rank, record in enumerate(chosen.to_dict("records"), 1):
            pose = Path(str(record["prediction_path"])).expanduser().resolve()
            protein = Path(str(proteins[case])).expanduser().resolve()
            map_path = Path(maps[case if case in maps else parent_map_name(case)]["map_path"])
            for path in (pose, protein, map_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            rows.append({"complex_name": case, "candidate_id": record["candidate_id"],
                         "ncc_rank": rank, "pose_path": str(pose),
                         "protein_path": str(protein), "map_path": str(map_path),
                         "multiscale_ncc": float(record["multiscale_ncc"]),
                         "source": record["source"], "source_rank": record["source_rank"]})
    result = pd.DataFrame(rows)
    result.attrs["nonfinite_ncc_found"] = int(sum(invalid_counts.values()))
    result.attrs["nonfinite_ncc_excluded"] = int(sum(invalid_counts.values())) if top_k else 0
    result.attrs["nonfinite_ncc_cases"] = len(invalid_counts)
    result.attrs["nonfinite_ncc_retained"] = int(sum(invalid_counts.values())) if top_k == 0 else 0
    result.attrs["candidate_count_min"] = int(result.groupby("complex_name").size().min())
    result.attrs["candidate_count_max"] = int(result.groupby("complex_name").size().max())
    return result


def prepare(args: argparse.Namespace) -> None:
    scores = blind_csv(args.pose_scores)
    manifest = blind_csv(args.blind_manifest)
    if len(manifest) != args.expected_cases:
        parent_count = (
            manifest.parent_complex_name.astype(str).str.strip().str.upper().nunique()
            if "parent_complex_name" in manifest else None
        )
        raise ValueError(
            f"Blind manifest has {len(manifest)} branch rows"
            + (f" from {parent_count} parent complexes" if parent_count is not None else "")
            + f"; --expected-cases={args.expected_cases} counts candidate groups, "
              "not parent complexes"
        )
    columns = [item.strip() for item in args.ncc_columns.split(",") if item.strip()]
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Give one or more distinct NCC columns")
    result = select_top_k(scores, manifest, map_lookup(args.map_csv), columns, args.top_k)
    ncc_audit = dict(result.attrs)
    if result.complex_name.nunique() != args.expected_cases or (
        args.top_k and len(result) != args.expected_cases * args.top_k
    ):
        raise ValueError("Selected candidate grid has wrong dimensions")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(json.dumps({
        "stage": "single_run_blind_prepare", "pose_scores_sha256": sha256(args.pose_scores),
        "blind_manifest_sha256": sha256(args.blind_manifest),
        "map_csv_sha256": sha256(args.map_csv), "ncc_columns": columns,
        "top_k": args.top_k, "cases": args.expected_cases,
        **ncc_audit,
    }, indent=2) + "\n")


def score_case(case: str, rows: list[dict]) -> list[dict]:
    params = QScoreParameters()
    records = []
    protein_cache = {}
    if len({row["map_path"] for row in rows}) != 1:
        raise ValueError(f"{case}: candidates use different maps")
    density = read_density_map(Path(rows[0]["map_path"]))
    for item in rows:
        record = {"complex_name": case, "candidate_id": item["candidate_id"],
                  "qscore_mean": np.nan, "qscore_heavy_atoms": 0, "qscore_error": ""}
        try:
            protein_path = item["protein_path"]
            if protein_path not in protein_cache:
                protein_cache[protein_path] = read_protein_heavy_atoms(Path(protein_path))
            protein = protein_cache[protein_path]
            ligand = read_sdf_heavy_atoms(Path(item["pose_path"]))
            protein_xyz = protein[["x", "y", "z"]].to_numpy(float)
            ligand_xyz = ligand[["x", "y", "z"]].to_numpy(float)
            from scipy.spatial import cKDTree
            if (cKDTree(protein_xyz).query(ligand_xyz)[0] < 0.2).any():
                raise ValueError("receptor duplicates ligand atom")
            context = np.vstack((protein_xyz, ligand_xyz))
            atoms = atom_qscores(density, context, range(len(protein_xyz), len(context)), params)
            values = atoms.qscore.to_numpy(float)
            if not np.isfinite(values).all() or int(atoms.n_points_outside_map.sum()):
                raise ValueError("incomplete Q-score")
            record["qscore_mean"] = float(values.mean())
            record["qscore_heavy_atoms"] = len(values)
        except (ValueError, OSError, RuntimeError) as exc:
            record["qscore_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def score(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("workers must be positive")
    frame = blind_csv(args.candidates)
    needed = {"complex_name", "candidate_id", "pose_path", "protein_path",
              "map_path", "multiscale_ncc"}
    if missing := needed - set(frame):
        raise ValueError(f"Candidate table lacks {sorted(missing)}")
    if frame.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Duplicate candidate keys")
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = [(case, group.to_dict("records"))
            for case, group in frame.groupby("complex_name", sort=True)]
    checkpoint_dir = args.output.parent / f"{args.output.stem}.cases"
    checkpoint_meta = {"candidate_sha256": sha256(args.candidates),
                       "qscore_core_sha256": sha256(ROOT / "evaluation/qscore_core.py"),
                       "qscore_parameters": QScoreParameters().__dict__}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = checkpoint_dir / "provenance.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != checkpoint_meta:
            raise ValueError(f"Checkpoint inputs or Q-score implementation changed: {checkpoint_dir}")
    elif any(checkpoint_dir.iterdir()):
        raise ValueError(f"Checkpoint directory lacks provenance: {checkpoint_dir}")
    else:
        metadata_path.write_text(json.dumps(checkpoint_meta, indent=2) + "\n")
    def checkpoint_path(case: str) -> Path:
        return checkpoint_dir / f"{hashlib.sha256(case.encode()).hexdigest()[:16]}.csv"
    def save_case(case: str, rows: list[dict]) -> None:
        path = checkpoint_path(case)
        temp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(temp, index=False)
        temp.replace(path)
    records = []
    pending = []
    for case, rows in jobs:
        path = checkpoint_path(case)
        if path.exists():
            saved = pd.read_csv(path).to_dict("records")
            if {(item["complex_name"], item["candidate_id"]) for item in saved} != {
                (case, item["candidate_id"]) for item in rows} or len(saved) != len(rows):
                raise ValueError(f"Incomplete checkpoint for {case}: {path}")
            records.extend(saved)
        else:
            pending.append((case, rows))
    if len(pending) != len(jobs):
        print(f"[Q-score] resumed {len(jobs) - len(pending)}/{len(jobs)} cases", flush=True)
    if args.workers == 1:
        for index, (case, rows) in enumerate(pending, 1):
            scored = score_case(case, rows)
            save_case(case, scored)
            records.extend(scored)
            print(f"[Q-score] {index}/{len(pending)} {case}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_case, case, rows): case for case, rows in pending}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                scored = future.result()
                save_case(case, scored)
                records.extend(scored)
                print(f"[Q-score] {index}/{len(pending)} {case}", flush=True)
    result = frame.merge(pd.DataFrame(records), on=["complex_name", "candidate_id"],
                         validate="one_to_one")
    if len(result) != len(frame):
        raise ValueError("Q-score results do not cover every input candidate")
    failed = result.qscore_mean.isna()
    if failed.any():
        print(f"Q-score unavailable for {int(failed.sum())} candidates across "
              f"{result.loc[failed, 'complex_name'].nunique()} cases; "
              "details are retained in qscore_error", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(json.dumps({
        "stage": "single_run_blind_qscore", "candidate_sha256": sha256(args.candidates),
        "qscore_core_sha256": sha256(ROOT / "evaluation/qscore_core.py"),
        "qscore_parameters": QScoreParameters().__dict__, "uses_reference": False,
        "qscore_unavailable_n": int(failed.sum()),
    }, indent=2) + "\n")


def evaluate(args: argparse.Namespace) -> None:
    from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd
    from scipy.stats import binomtest

    frame = blind_csv(args.scored)
    refs = pd.read_csv(args.reference_manifest, dtype={"complex_name": str})
    if not {"complex_name", "reference_ligand"}.issubset(refs):
        raise ValueError("Reference manifest needs complex_name, reference_ligand")
    refs.complex_name = refs.complex_name.str.strip().str.upper()
    if refs.complex_name.duplicated().any() or set(refs.complex_name) != set(frame.complex_name):
        raise ValueError("Reference and scored cohorts differ")
    paths = {
        row.complex_name: (Path(row.reference_ligand) if Path(row.reference_ligand).is_absolute()
                           else args.reference_manifest.parent / row.reference_ligand).resolve()
        for row in refs.itertuples(index=False)
    }
    frame["ligand_rmsd_A"] = [
        symmetry_rmsd(read_heavy_molecule(Path(paths[row.complex_name])),
                      read_heavy_molecule(Path(row.pose_path)))
        for row in frame.itertuples(index=False)
    ]
    cases = frame.complex_name.nunique()
    if frame.groupby("complex_name").size().lt(2).any():
        raise ValueError("Every case needs at least two poses")
    for metric in ("multiscale_ncc", "qscore_mean"):
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        valid = frame.groupby("complex_name")[metric].apply(lambda x: np.isfinite(x).any())
        if not valid.all():
            raise ValueError(f"No finite {metric} for cases: {valid.index[~valid].tolist()}")
    results = []
    selected_tables = {}
    for metric in ("multiscale_ncc", "qscore_mean"):
        selected = frame.sort_values(["complex_name", metric, "candidate_id"],
                                     ascending=[True, False, True], kind="stable",
                                     na_position="last")
        selected = selected.drop_duplicates("complex_name")
        selected_tables[metric] = selected[["complex_name", "candidate_id", "ligand_rmsd_A"]].rename(
            columns={"candidate_id": f"{metric}_candidate_id",
                     "ligand_rmsd_A": f"{metric}_rmsd_A"})
        results.append({"selector": metric, "cases": cases,
                        "top1_lt2A_n": int(selected.ligand_rmsd_A.lt(2).sum()),
                        "top1_lt2A_fraction": float(selected.ligand_rmsd_A.lt(2).mean()),
                        "median_rmsd_A": float(selected.ligand_rmsd_A.median())})
    oracle = frame.groupby("complex_name").ligand_rmsd_A.min()
    results.append({"selector": "oracle_within_scored_pool", "cases": cases,
                    "top1_lt2A_n": int(oracle.lt(2).sum()),
                    "top1_lt2A_fraction": float(oracle.lt(2).mean()),
                    "median_rmsd_A": float(oracle.median())})
    paired = selected_tables["multiscale_ncc"].merge(
        selected_tables["qscore_mean"], on="complex_name", validate="one_to_one")
    ncc_ok = paired.multiscale_ncc_rmsd_A.lt(2)
    q_ok = paired.qscore_mean_rmsd_A.lt(2)
    q_wins = int((q_ok & ~ncc_ok).sum())
    ncc_wins = int((ncc_ok & ~q_ok).sum())
    paired["qscore_rescues_ncc"] = q_ok & ~ncc_ok
    paired["qscore_loses_ncc"] = ncc_ok & ~q_ok
    paired_summary = {"qscore_rescues_ncc_n": q_wins,
                      "qscore_loses_ncc_n": ncc_wins,
                      "paired_exact_p": float(binomtest(q_wins, q_wins + ncc_wins).pvalue)
                      if q_wins + ncc_wins else 1.0}
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    frame.to_csv(args.output_dir / "per_pose_with_rmsd.csv", index=False)
    pd.DataFrame(results).to_csv(args.output_dir / "selector_comparison.csv", index=False)
    paired.to_csv(args.output_dir / "per_case_paired_selection.csv", index=False)
    (args.output_dir / "paired_summary.json").write_text(json.dumps(paired_summary, indent=2) + "\n")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "stage": "single_run_validation_reveal", "scored_sha256": sha256(args.scored),
        "reference_manifest_sha256": sha256(args.reference_manifest),
        "uses_reference_only_after_blind_scoring": True,
    }, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False))
    print(json.dumps(paired_summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--pose-scores", required=True, type=Path)
    a.add_argument("--blind-manifest", required=True, type=Path)
    a.add_argument("--map-csv", required=True, type=Path)
    a.add_argument("--ncc-columns", default="multiscale_ncc")
    a.add_argument("--top-k", type=int, default=0,
                   help="0 scores every pose; positive K selects the K highest finite-NCC poses")
    a.add_argument("--expected-cases", required=True, type=int)
    a.add_argument("--output", required=True, type=Path)
    a.set_defaults(func=prepare)
    a = sub.add_parser("score")
    a.add_argument("--candidates", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--workers", type=int, default=4)
    a.set_defaults(func=score)
    a = sub.add_parser("evaluate")
    a.add_argument("--scored", required=True, type=Path)
    a.add_argument("--reference-manifest", required=True, type=Path)
    a.add_argument("--output-dir", required=True, type=Path)
    a.set_defaults(func=evaluate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
