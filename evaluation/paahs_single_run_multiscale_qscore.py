#!/usr/bin/env python3
"""Blind two-scale NCC rescoring, then development-only NCC/Q-score weight sweep.

Use the same fixed crop/mask Gaussian NCC as rerank_ligand_by_density_ncc.py.
No reference ligand is read during `rescore`; `sweep` uses an existing reveal
table and must only be run on the development cohort.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cryodiffdock_pocket.density_ranking import (
    fixed_crop_mask, load_cartesian_density_crop, score_gaussian_density,
)
from evaluation.paahs_single_run_qscore_pilot import blind_csv, sha256

SIGMAS = (1.5, 2.25)


def read_pose(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Match rerank_ligand_by_density_ncc.py's heavy-atom SDF handling."""
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = supplier[0] if supplier and len(supplier) else None
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Could not read pose: {path}")
    molecule = Chem.RemoveHs(molecule, sanitize=False)
    coordinates = np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)
    atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                                dtype=np.float32)
    return coordinates, atomic_numbers


def _resolve_path(value: str, base: Path) -> str:
    path = Path(str(value)).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _rescore_case(case: str, rows: list[dict]) -> list[dict]:
    crops = {row["crop_path"] for row in rows}
    if len(crops) != 1:
        raise ValueError(f"{case}: candidates use different density crops")
    crop = load_cartesian_density_crop(Path(next(iter(crops))))
    mask = fixed_crop_mask(crop, radius_A=10.0, min_voxels=64)
    records = []
    for row in rows:
        record = {"complex_name": case, "candidate_id": row["candidate_id"],
                  "ncc_sigma_1p5_rescored": np.nan,
                  "ncc_sigma_2p25": np.nan, "multiscale_ncc": np.nan,
                  "ncc_rescore_error": ""}
        try:
            coords, atomic_numbers = read_pose(Path(row["pose_path"]))
            values = [score_gaussian_density(crop, coords, atomic_numbers, sigma, mask).ncc
                      for sigma in SIGMAS]
            if not np.isfinite(values).all():
                raise ValueError("nonfinite NCC at one or both scales")
            record["ncc_sigma_1p5_rescored"] = float(values[0])
            record["ncc_sigma_2p25"] = float(values[1])
            record["multiscale_ncc"] = float(np.mean(values))
        except (ValueError, OSError, RuntimeError, IndexError) as exc:
            record["ncc_rescore_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
    return records


def rescore(args: argparse.Namespace) -> None:
    if args.workers < 1 or args.output.exists():
        raise ValueError("workers must be positive and output must not already exist")
    scored = blind_csv(args.scored)
    original = blind_csv(args.pose_scores)
    required_scored = {"complex_name", "candidate_id", "source", "source_rank",
                       "pose_path", "qscore_mean", "multiscale_ncc"}
    required_original = {"complex_name", "source", "source_rank", "prediction_path",
                         "crop_path", "ncc_sigma_1p5"}
    if required_scored - set(scored) or required_original - set(original):
        raise ValueError(f"Missing columns: scored={sorted(required_scored - set(scored))}, "
                         f"pose_scores={sorted(required_original - set(original))}")
    if scored.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Duplicate candidate keys in scored table")
    if original.duplicated(["complex_name", "source", "source_rank"]).any():
        raise ValueError("Duplicate source/rank keys in original pose scores")
    original = original[["complex_name", "source", "source_rank", "prediction_path",
                         "crop_path", "ncc_sigma_1p5"]]
    frame = scored.merge(original, on=["complex_name", "source", "source_rank"],
                         how="left", validate="one_to_one", indicator=True)
    if len(frame) != len(scored) or frame._merge.ne("both").any():
        raise ValueError("Some Q-score candidates do not match original pose scores")
    original_base = args.pose_scores.parent.resolve()
    if any(_resolve_path(row.prediction_path, original_base) !=
           str(Path(row.pose_path).expanduser().resolve())
           for row in frame.itertuples(index=False)):
        raise ValueError("Original pose path differs from Q-score candidate pose path")
    frame["crop_path"] = frame.crop_path.map(lambda p: _resolve_path(p, original_base))
    if not frame.crop_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("At least one original density crop is missing")
    if not frame.pose_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("At least one candidate pose is missing")
    frame = frame.drop(columns=["_merge"])

    checkpoint_dir = args.output.parent / f"{args.output.stem}.cases"
    checkpoint_meta = {"scored_sha256": sha256(args.scored),
                       "pose_scores_sha256": sha256(args.pose_scores),
                       "density_ranking_sha256": sha256(ROOT / "cryodiffdock_pocket/density_ranking.py"),
                       "sigmas_A": SIGMAS, "mask_radius_A": 10.0,
                       "min_mask_voxels": 64, "cutoff_sigma": 4.0}
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_dir / "provenance.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != json.loads(json.dumps(checkpoint_meta)):
            raise ValueError(f"Rescore checkpoint provenance differs: {checkpoint_dir}")
    elif any(checkpoint_dir.iterdir()):
        raise ValueError(f"Rescore checkpoint lacks provenance: {checkpoint_dir}")
    else:
        meta_path.write_text(json.dumps(checkpoint_meta, indent=2) + "\n")

    jobs = [(case, group.to_dict("records"))
            for case, group in frame.groupby("complex_name", sort=True)]
    def case_path(case: str) -> Path:
        return checkpoint_dir / f"{hashlib.sha256(case.encode()).hexdigest()[:16]}.csv"
    records, pending = [], []
    for case, rows in jobs:
        path = case_path(case)
        if path.exists():
            saved = pd.read_csv(path).to_dict("records")
            if len(saved) != len(rows) or {
                (r["complex_name"], r["candidate_id"]) for r in saved
            } != {(case, r["candidate_id"]) for r in rows}:
                raise ValueError(f"Incomplete rescore checkpoint: {path}")
            records.extend(saved)
        else:
            pending.append((case, rows))
    if len(pending) != len(jobs):
        print(f"[NCC] resumed {len(jobs) - len(pending)}/{len(jobs)} cases", flush=True)
    if args.workers == 1:
        completed = ((case, _rescore_case(case, rows)) for case, rows in pending)
        for index, (case, result) in enumerate(completed, 1):
            path = case_path(case)
            temp = path.with_suffix(".tmp")
            pd.DataFrame(result).to_csv(temp, index=False)
            temp.replace(path)
            records.extend(result)
            print(f"[NCC] {index}/{len(pending)} {case}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_rescore_case, case, rows): case for case, rows in pending}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                result = future.result()
                path = case_path(case)
                temp = path.with_suffix(".tmp")
                pd.DataFrame(result).to_csv(temp, index=False)
                temp.replace(path)
                records.extend(result)
                print(f"[NCC] {index}/{len(pending)} {case}", flush=True)

    output = frame.rename(columns={"multiscale_ncc": "original_selection_ncc"}).merge(
        pd.DataFrame(records), on=["complex_name", "candidate_id"],
        how="left", validate="one_to_one", indicator=True)
    if len(records) != len(frame) or len(output) != len(frame) or output._merge.ne("both").any():
        raise ValueError("NCC results do not cover every candidate")
    output = output.drop(columns=["_merge"])
    # Empty strings become NaN when checkpoints are read back.
    output["ncc_rescore_error"] = output.ncc_rescore_error.fillna("")
    old = pd.to_numeric(output.ncc_sigma_1p5, errors="coerce")
    new = pd.to_numeric(output.ncc_sigma_1p5_rescored, errors="coerce")
    both = np.isfinite(old) & np.isfinite(new)
    discrepancy = (old[both] - new[both]).abs()
    audit = {"stage": "single_run_blind_multiscale_rescore", **checkpoint_meta,
             "cases": int(output.complex_name.nunique()), "poses": len(output),
             "original_single_scale_finite": int(np.isfinite(old).sum()),
             "rescored_single_scale_finite": int(np.isfinite(new).sum()),
             "multiscale_finite": int(np.isfinite(output.multiscale_ncc).sum()),
             "both_single_scale_finite": int(both.sum()),
             "max_single_scale_abs_difference": float(discrepancy.max()) if len(discrepancy) else None,
             "single_scale_disagreements_gt_1e_4": int(discrepancy.gt(1e-4).sum()),
             "ncc_rescore_failures": int(output.ncc_rescore_error.astype(bool).sum()),
             "uses_reference": False}
    if audit["single_scale_disagreements_gt_1e_4"]:
        raise ValueError(f"Rescored 1.5 A NCC disagrees with original; audit: {audit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def _case_z(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").astype(float)
    finite = np.isfinite(values)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if finite.any():
        mean = float(values[finite].mean())
        std = float(values[finite].std(ddof=0))
        result.loc[finite] = (values[finite] - mean) / std if std > 1e-12 else 0.0
    return result


def select_weighted(frame: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Weight is the Q-score fraction; intermediate weights require both scores."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be in [0,1]")
    work = frame.copy()
    work["ncc_z"] = work.groupby("complex_name", group_keys=False)["multiscale_ncc"].transform(_case_z)
    work["qscore_z"] = work.groupby("complex_name", group_keys=False)["qscore_mean"].transform(_case_z)
    if weight == 0:
        work["selection_score"] = work.ncc_z
    elif weight == 1:
        work["selection_score"] = work.qscore_z
    else:
        work["selection_score"] = (1 - weight) * work.ncc_z + weight * work.qscore_z
    if not work.groupby("complex_name").selection_score.apply(lambda s: np.isfinite(s).any()).all():
        raise ValueError(f"No eligible pose for weight={weight}")
    work = work.sort_values(["complex_name", "selection_score", "candidate_id"],
                            ascending=[True, False, True], na_position="last", kind="stable")
    return work.drop_duplicates("complex_name")


def sweep(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    scored = blind_csv(args.multiscale_scored)
    revealed = pd.read_csv(args.revealed_poses, dtype={"complex_name": str})
    if not {"complex_name", "candidate_id", "ligand_rmsd_A"}.issubset(revealed):
        raise ValueError("Reveal table needs complex_name, candidate_id, ligand_rmsd_A")
    revealed.complex_name = revealed.complex_name.str.strip().str.upper()
    if scored.duplicated(["complex_name", "candidate_id"]).any() or \
            revealed.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Duplicate candidate keys")
    if set(map(tuple, scored[["complex_name", "candidate_id"]].to_numpy())) != \
            set(map(tuple, revealed[["complex_name", "candidate_id"]].to_numpy())):
        raise ValueError("Multiscale scores and revealed poses have different candidate pools")
    frame = scored.drop(columns=["ligand_rmsd_A"], errors="ignore").merge(
        revealed[["complex_name", "candidate_id", "ligand_rmsd_A"]],
        on=["complex_name", "candidate_id"], validate="one_to_one")
    for name in ("ncc_sigma_1p5_rescored", "multiscale_ncc", "qscore_mean", "ligand_rmsd_A"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    weights = [float(item) for item in args.q_weights.split(",") if item.strip()]
    if len(weights) != len(set(weights)) or 0 not in weights or 1 not in weights:
        raise ValueError("Weights must be distinct and include 0 and 1")
    if any(not 0 <= w <= 1 for w in weights):
        raise ValueError("Weights must lie in [0,1]")
    baseline = frame.sort_values(["complex_name", "ncc_sigma_1p5_rescored", "candidate_id"],
                                 ascending=[True, False, True], na_position="last",
                                 kind="stable").drop_duplicates("complex_name")
    selections = [baseline.assign(selector="single_scale_ncc", q_weight=np.nan)]
    for weight in sorted(weights):
        selections.append(select_weighted(frame, weight).assign(
            selector=f"multiscale_plus_q_w{weight:g}", q_weight=weight))
    selected = pd.concat(selections, ignore_index=True)
    cases = frame.complex_name.nunique()
    if selected.groupby("selector").size().ne(cases).any():
        raise ValueError("A selector did not choose exactly one pose per case")
    summary = selected.groupby("selector", sort=False).agg(
        cases=("complex_name", "size"),
        top1_lt2A_n=("ligand_rmsd_A", lambda s: int(s.lt(2).sum())),
        median_rmsd_A=("ligand_rmsd_A", "median"),
        q_weight=("q_weight", "first"),
    ).reset_index()
    summary["top1_lt2A_fraction"] = summary.top1_lt2A_n / cases
    baseline_ok = selected.loc[selected.selector.eq("multiscale_plus_q_w0"),
                               ["complex_name", "ligand_rmsd_A"]].set_index("complex_name").ligand_rmsd_A.lt(2)
    paired = []
    for name, group in selected.groupby("selector", sort=False):
        if name == "single_scale_ncc":
            continue
        current = group.set_index("complex_name").ligand_rmsd_A.lt(2)
        paired.append({"selector": name,
                       "rescues_vs_multiscale_ncc": int((current & ~baseline_ok).sum()),
                       "losses_vs_multiscale_ncc": int((~current & baseline_ok).sum())})
    summary = summary.merge(pd.DataFrame(paired), on="selector", how="left")
    oracle = frame.groupby("complex_name").ligand_rmsd_A.min()
    summary = pd.concat([summary, pd.DataFrame([{
        "selector": "oracle_within_pool", "cases": cases,
        "top1_lt2A_n": int(oracle.lt(2).sum()),
        "top1_lt2A_fraction": float(oracle.lt(2).mean()),
        "median_rmsd_A": float(oracle.median()),
    }])], ignore_index=True)
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "weight_sweep_summary.csv", index=False)
    selected[["selector", "q_weight", "complex_name", "candidate_id", "pose_path",
              "ncc_sigma_1p5_rescored", "ncc_sigma_2p25", "multiscale_ncc",
              "qscore_mean", "selection_score", "ligand_rmsd_A"]].to_csv(
                  args.output_dir / "weight_sweep_selected_poses.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "stage": "single_run_development_weight_sweep",
        "multiscale_scored_sha256": sha256(args.multiscale_scored),
        "revealed_poses_sha256": sha256(args.revealed_poses),
        "weights": sorted(weights), "cases": cases,
        "normalization": "within-case population z-score over finite candidate values",
        "missing_score_rule": "intermediate weights require both finite scores; endpoints require only their own score",
        "uses_reference": True,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("rescore")
    p.add_argument("--scored", required=True, type=Path)
    p.add_argument("--pose-scores", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=rescore)
    p = commands.add_parser("sweep")
    p.add_argument("--multiscale-scored", required=True, type=Path)
    p.add_argument("--revealed-poses", required=True, type=Path)
    p.add_argument("--q-weights", default="0,0.1,0.25,0.5,0.75,1")
    p.add_argument("--output-dir", required=True, type=Path)
    p.set_defaults(func=sweep)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
