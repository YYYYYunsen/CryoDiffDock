#!/usr/bin/env python3
"""Q-score of the two reproducible PAAHS mode medoids in each multimodal case.

Supplementary check for Figure 5: does an independent atom-level map-fit
measure (Q-score) separate the two modes better than the multiscale NCC that
PAAHS optimises?  Poses are frozen; nothing is fitted or moved.

Subcommands (run in order on the server):

``prepare``          72-case / 144-medoid worklist, map lookup and input checks
``run``              Q-score per pose (resumable; ``--limit-cases`` for a pilot)
``summarize``        per-atom / per-pose / per-case tables, statistics, validation
``compare-chimerax`` agreement with the ChimeraX QScore bundle on pilot cases

Deposited ligands are scored in the same map and protein context as a
positive control only; mode choice never uses them.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.qscore_core import (  # noqa: E402
    QScoreParameters,
    WATER_RESIDUES,
    atom_qscores,
    drop_target_ligand_copies,
    read_density_map,
    read_protein_heavy_atoms,
    read_sdf_heavy_atoms,
)

MEDOID_COLUMNS = {
    "complex_name", "mode_rank", "mode_id", "mode_size", "mode_multiscale_ncc_median",
    "medoid_replicate", "medoid_pose_path", "medoid_protein_path",
    "medoid_final_multiscale_ncc", "reference_ligand_path",
}
MODE_METRIC_COLUMNS = ("rmsd", "pb_valid", "near_best_density", "joint_density_pb_supported")
ROLES = ("mode1", "mode2", "deposited")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def read_table(path: Path, required: set) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"complex_name": str})
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame["complex_name"] = frame["complex_name"].str.strip().str.upper()
    return frame


def map_lookup(path: Path) -> dict:
    """Same rules as scripts/build_oracle_density_source_manifest.py."""
    lookup = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        resolution_column = next(
            (name for name in (reader.fieldnames or []) if "resol" in name.lower()), None
        )
        for row in reader:
            pdb_id = str(row.get("pdb_id") or row.get("complex_name") or "").strip().upper()
            value = str(row.get("map_path") or "").strip()
            status = str(row.get("status") or "ok").strip().lower()
            if not pdb_id or not value or status not in {"ok", "success", "downloaded"}:
                continue
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = Path(path).parent / candidate
            lookup[pdb_id] = {
                "map_path": str(candidate.resolve()),
                "map_resolution": row.get(resolution_column, "") if resolution_column else "",
            }
    return lookup


def heavy_coords(frame: pd.DataFrame) -> np.ndarray:
    return frame[["x", "y", "z"]].to_numpy(dtype=np.float64)


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    medoids = read_table(args.medoids, MEDOID_COLUMNS)
    medoids["mode_rank"] = pd.to_numeric(medoids.mode_rank, errors="raise").astype(int)
    metrics = read_table(args.mode_metrics, {"complex_name", "mode_rank", *MODE_METRIC_COLUMNS})
    metrics["mode_rank"] = pd.to_numeric(metrics.mode_rank, errors="raise").astype(int)
    metrics = metrics[["complex_name", "mode_rank", *MODE_METRIC_COLUMNS]]

    if medoids.duplicated(["complex_name", "mode_rank"]).any():
        raise ValueError("Duplicate case/mode_rank in medoid table")
    ranks = medoids.groupby("complex_name").mode_rank.apply(lambda s: tuple(sorted(s)))
    bad = ranks[ranks != (1, 2)]
    if len(bad):
        raise ValueError(f"Cases without exactly mode ranks 1 and 2: {bad.index.tolist()[:10]}")
    if args.expected_cases and medoids.complex_name.nunique() != args.expected_cases:
        raise ValueError(
            f"{medoids.complex_name.nunique()} cases, expected {args.expected_cases}"
        )
    merged = medoids.merge(metrics, on=["complex_name", "mode_rank"], how="left", validate="1:1")
    if merged.rmsd.isna().any():
        raise ValueError("Mode metrics table lacks RMSD for some medoids")

    maps = map_lookup(args.map_csv)
    missing_maps = sorted(set(merged.complex_name) - set(maps))
    if missing_maps:
        raise FileNotFoundError(f"No map for {len(missing_maps)} cases: {missing_maps[:10]}")

    work_rows, case_rows, problems = [], [], []
    for case, group in merged.groupby("complex_name", sort=True):
        group = group.sort_values("mode_rank")
        m1, m2 = group.iloc[0], group.iloc[1]
        map_path = Path(maps[case]["map_path"])
        paths = [map_path, Path(m1.medoid_pose_path), Path(m2.medoid_pose_path),
                 Path(m1.medoid_protein_path), Path(m2.medoid_protein_path),
                 Path(m1.reference_ligand_path)]
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            problems.append({"complex_name": case, "problem": "missing_file", "detail": ";".join(absent)})
            continue
        lig1 = read_sdf_heavy_atoms(Path(m1.medoid_pose_path))
        lig2 = read_sdf_heavy_atoms(Path(m2.medoid_pose_path))
        deposited = read_sdf_heavy_atoms(Path(m1.reference_ligand_path))
        same_atoms = len(lig1) == len(lig2) and list(lig1.element) == list(lig2.element)
        if not same_atoms:
            problems.append({"complex_name": case, "problem": "mode_ligands_differ",
                             "detail": f"{len(lig1)} vs {len(lig2)} heavy atoms"})
        prot1 = read_protein_heavy_atoms(Path(m1.medoid_protein_path))
        prot2 = read_protein_heavy_atoms(Path(m2.medoid_protein_path))
        if len(prot1) == len(prot2):
            protein_deviation = float(np.abs(heavy_coords(prot1) - heavy_coords(prot2)).max())
        else:
            protein_deviation = float("inf")
        protein_identical = protein_deviation <= args.protein_tolerance
        # Identical receptors: score both modes (and the control) against the
        # mode-1 receptor so the paired difference reflects only the ligand.
        protein2 = m1.medoid_protein_path if protein_identical else m2.medoid_protein_path
        if not protein_identical:
            problems.append({"complex_name": case, "problem": "mode_receptors_differ",
                             "detail": f"max heavy-atom deviation {protein_deviation:.3f} A"})
        case_rows.append({
            "complex_name": case,
            "map_path": str(map_path),
            "map_resolution": maps[case]["map_resolution"],
            "reference_ligand_path": m1.reference_ligand_path,
            "mode1_protein_path": m1.medoid_protein_path,
            "mode2_protein_path": m2.medoid_protein_path,
            "protein_identical": protein_identical,
            "protein_max_heavy_atom_deviation": protein_deviation,
            "mode_ligand_heavy_atoms": len(lig1),
            "mode_ligands_same_atoms": same_atoms,
            "deposited_heavy_atoms": len(deposited),
            "deposited_matches_mode_atom_count": len(deposited) == len(lig1),
        })
        for role, row, protein in (("mode1", m1, m1.medoid_protein_path), ("mode2", m2, protein2)):
            work_rows.append({
                "complex_name": case, "role": role, "mode_rank": int(row.mode_rank),
                "mode_id": row.mode_id, "mode_size": row.mode_size,
                "medoid_replicate": row.medoid_replicate,
                "ligand_path": row.medoid_pose_path, "protein_path": protein,
                "map_path": str(map_path), "reference_ligand_path": row.reference_ligand_path,
                "medoid_final_multiscale_ncc": row.medoid_final_multiscale_ncc,
                "mode_multiscale_ncc_median": row.mode_multiscale_ncc_median,
                **{column: row[column] for column in MODE_METRIC_COLUMNS},
            })
        if not args.no_deposited_control:
            work_rows.append({
                "complex_name": case, "role": "deposited", "mode_rank": 0,
                "ligand_path": m1.reference_ligand_path, "protein_path": m1.medoid_protein_path,
                "map_path": str(map_path), "reference_ligand_path": m1.reference_ligand_path,
                "rmsd": 0.0,
            })

    output.mkdir(parents=True)
    pd.DataFrame(work_rows).to_csv(output / "worklist.csv", index=False)
    pd.DataFrame(case_rows).to_csv(output / "cases.csv", index=False)
    pd.DataFrame(problems, columns=["complex_name", "problem", "detail"]).to_csv(
        output / "prepare_problems.csv", index=False
    )
    (output / "provenance.json").write_text(json.dumps({
        "created": now(),
        "stage": "prepare",
        "inputs": {
            "medoids": {"path": str(args.medoids.resolve()), "sha256": sha256(args.medoids)},
            "mode_metrics": {"path": str(args.mode_metrics.resolve()), "sha256": sha256(args.mode_metrics)},
            "map_csv": {"path": str(args.map_csv.resolve()), "sha256": sha256(args.map_csv)},
        },
        "protein_tolerance_A": args.protein_tolerance,
        "deposited_control": not args.no_deposited_control,
        "cases": len(case_rows),
        "poses": len(work_rows),
        "problems": len(problems),
    }, indent=2) + "\n")
    print(f"cases={len(case_rows)} poses={len(work_rows)} problems={len(problems)} -> {output}")
    if problems:
        print(pd.DataFrame(problems).to_string(index=False))
        missing = [p for p in problems if p["problem"] == "missing_file"]
        if missing and not args.allow_missing:
            raise SystemExit("Missing inputs; fix paths or pass --allow-missing")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run_parameters(args: argparse.Namespace) -> dict:
    parameters = QScoreParameters(
        reference_sigma=args.sigma,
        max_shell_radius=args.max_shell_radius,
        shell_step=args.shell_step,
        points_per_shell=args.points_per_shell,
        trial_points=args.trial_points,
        random_seed=args.random_seed,
    )
    return {
        "qscore": parameters.__dict__,
        "origin_mode": args.origin_mode,
        "drop_waters": args.drop_waters,
        "target_copy_distance_A": args.target_copy_distance,
        "implementation": "evaluation/qscore_core.py",
        "implementation_sha256": sha256(PROJECT_ROOT / "evaluation" / "qscore_core.py"),
    }


def score_case(case: str, rows: list, settings: dict, output: str) -> dict:
    """Score every pose of one case; one map load, one receptor per path."""
    output = Path(output)
    parameters = QScoreParameters(**settings["qscore"])
    started = time.perf_counter()
    status = {"complex_name": case, "ok": 0, "failed": 0}
    try:
        density = read_density_map(Path(rows[0]["map_path"]), settings["origin_mode"])
    except Exception as exc:  # noqa: BLE001 - recorded per case
        for row in rows:
            write_failure(output, row, f"map: {exc!r}")
        status["failed"] = len(rows)
        return status
    map_seconds = time.perf_counter() - started
    min_d, max_d = density.reference_bounds()
    (output / "per_case").mkdir(exist_ok=True)
    (output / "per_case" / f"{case}.map.json").write_text(json.dumps({
        "complex_name": case, "map_path": rows[0]["map_path"],
        "shape_xyz": list(density.data.shape), "voxel_size": density.voxel_size.tolist(),
        "origin": density.origin.tolist(), "frame_rule": density.frame_rule,
        "header_nstart_xyz": density.header_nstart_xyz, "header_origin_xyz": density.header_origin_xyz,
        "mean": density.mean, "sd": density.sd, "min": density.minimum, "max": density.maximum,
        "reference_min_d": min_d, "reference_max_d": max_d, "load_seconds": map_seconds,
    }, indent=2) + "\n")

    reference = heavy_coords(read_sdf_heavy_atoms(Path(rows[0]["reference_ligand_path"])))
    receptors = {}
    for row in rows:
        t0 = time.perf_counter()
        try:
            if row["protein_path"] not in receptors:
                protein = read_protein_heavy_atoms(Path(row["protein_path"]))
                if settings["drop_waters"]:
                    protein = protein[~protein.resname.isin(WATER_RESIDUES)].reset_index(drop=True)
                protein, removed = drop_target_ligand_copies(
                    protein, reference, settings["target_copy_distance_A"]
                )
                receptors[row["protein_path"]] = (protein, removed)
            protein, removed = receptors[row["protein_path"]]
            ligand = read_sdf_heavy_atoms(Path(row["ligand_path"]))
            context = np.vstack([heavy_coords(protein), heavy_coords(ligand)])
            query = np.arange(len(protein), len(context))
            atoms = atom_qscores(density, context, query, parameters)
            atoms.insert(0, "complex_name", case)
            atoms.insert(1, "role", row["role"])
            atoms.insert(2, "ligand_atom_index", ligand.atom_index.to_numpy())
            atoms.insert(3, "element", ligand.element.to_numpy())
            for axis in "xyz":
                atoms[axis] = ligand[axis].to_numpy()
            # Nearest receptor atom distance flags ligand atoms buried in protein
            from scipy.spatial import cKDTree
            nearest, _ = cKDTree(heavy_coords(protein)).query(heavy_coords(ligand))
            atoms["nearest_receptor_atom_A"] = nearest
            atoms = atoms.drop(columns=["context_index"])
            stem = f"{case}__{row['role']}"
            atoms.to_csv(output / "per_pose" / f"{stem}.atoms.csv", index=False)
            q = atoms.qscore.to_numpy()
            summary = {
                "complex_name": case, "role": row["role"], "status": "ok",
                "ligand_path": row["ligand_path"], "protein_path": row["protein_path"],
                "map_path": row["map_path"],
                "n_heavy_atoms": int(len(atoms)),
                "n_receptor_atoms": int(len(protein)),
                "removed_target_copies": ";".join(removed),
                "qscore_mean": float(np.nanmean(q)) if np.isfinite(q).any() else None,
                "qscore_median": float(np.nanmedian(q)) if np.isfinite(q).any() else None,
                "qscore_min": float(np.nanmin(q)) if np.isfinite(q).any() else None,
                "qscore_fraction_below_0p5": float(np.mean(q < 0.5)),
                "n_atoms_nan": int(np.isnan(q).sum()),
                "n_points_outside_map": int(atoms.n_points_outside_map.sum()),
                "n_short_shells": int(atoms.n_short_shells.sum()),
                "seconds": time.perf_counter() - t0,
            }
            (output / "per_pose" / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
            error_file = output / "per_pose" / f"{stem}.error.json"
            if error_file.exists():
                error_file.unlink()
            status["ok"] += 1
        except Exception as exc:  # noqa: BLE001 - recorded per pose
            write_failure(output, row, f"{exc!r}\n{traceback.format_exc()}")
            status["failed"] += 1
    status["seconds"] = time.perf_counter() - started
    return status


def write_failure(output: Path, row: dict, message: str) -> None:
    stem = f"{row['complex_name']}__{row['role']}"
    (output / "per_pose" / f"{stem}.error.json").write_text(json.dumps({
        "complex_name": row["complex_name"], "role": row["role"], "status": "failed",
        "error": message,
    }, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    prepared = args.prepared_dir.resolve()
    output = (args.output_dir or prepared / "qscore_run").resolve()
    work = pd.read_csv(prepared / "worklist.csv", dtype={"complex_name": str})
    settings = run_parameters(args)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_pose").mkdir(exist_ok=True)
    params_file = output / "run_params.json"
    if params_file.exists():
        previous = json.loads(params_file.read_text())
        if previous != settings:
            raise ValueError(f"{params_file} differs from current settings; use a new --output-dir")
    else:
        params_file.write_text(json.dumps(settings, indent=2) + "\n")

    cases = sorted(work.complex_name.unique())
    if args.cases:
        wanted = [value.strip().upper() for value in args.cases.split(",") if value.strip()]
        unknown = sorted(set(wanted) - set(cases))
        if unknown:
            raise ValueError(f"Unknown cases: {unknown}")
        cases = wanted
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    todo = []
    for case in cases:
        rows = work[work.complex_name == case].to_dict("records")
        done = all((output / "per_pose" / f"{case}__{row['role']}.json").exists() for row in rows)
        if done and not args.overwrite:
            continue
        todo.append((case, rows))
    print(f"{len(cases)} selected, {len(cases) - len(todo)} already done, {len(todo)} to score")

    started = time.perf_counter()
    log = []
    if args.workers <= 1:
        for index, (case, rows) in enumerate(todo, start=1):
            status = score_case(case, rows, settings, str(output))
            log.append(status)
            print(f"[{index}/{len(todo)}] {case} ok={status['ok']} failed={status['failed']} "
                  f"{status.get('seconds', 0):.1f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(score_case, case, rows, settings, str(output)): case
                       for case, rows in todo}
            for index, future in enumerate(as_completed(futures), start=1):
                status = future.result()
                log.append(status)
                print(f"[{index}/{len(todo)}] {status['complex_name']} ok={status['ok']} "
                      f"failed={status['failed']} {status.get('seconds', 0):.1f}s", flush=True)
    wall = time.perf_counter() - started
    with (output / "run_log.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "finished": now(), "cases": [s["complex_name"] for s in log],
            "wall_seconds": wall, "workers": args.workers,
            "failed_poses": int(sum(s["failed"] for s in log)),
            "host": platform.node(), "python": sys.version.split()[0],
        }) + "\n")
    failed = sum(s["failed"] for s in log)
    print(f"wall {wall:.1f}s; failed poses {failed}; output {output}")
    if failed:
        raise SystemExit("Some poses failed; see per_pose/*.error.json")


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def pick_accuracy(frame: pd.DataFrame, score1: str, score2: str) -> dict:
    """Among exactly-one-correct cases: does the higher score pick it?"""
    higher1 = frame[score1] > frame[score2]
    ties = frame[score1] == frame[score2]
    correct = np.where(frame.mode1_correct, higher1, ~higher1 & ~ties)
    return {"n": int(len(frame)), "correct": int(correct.sum()), "ties": int(ties.sum()),
            "fraction": float(correct.mean()) if len(frame) else None}, correct


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    try:
        from scipy.stats import binomtest
    except ImportError:  # scipy < 1.7
        from scipy.stats import binom_test

        def binomtest(k, n, p):
            return argparse.Namespace(pvalue=binom_test(k, n, p))

    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = float(binomtest(only_a, n, 0.5).pvalue) if n else 1.0
    return {"only_first_correct": only_a, "only_second_correct": only_b, "exact_p": p}


def summarize(args: argparse.Namespace) -> None:
    from scipy.stats import spearmanr, wilcoxon

    prepared = args.prepared_dir.resolve()
    run_dir = (args.run_dir or prepared / "qscore_run").resolve()
    output = (args.output_dir or run_dir / "summary").resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    work = pd.read_csv(prepared / "worklist.csv", dtype={"complex_name": str})
    cases = pd.read_csv(prepared / "cases.csv", dtype={"complex_name": str})

    pose_rows, atom_frames, failures = [], [], []
    for row in work.itertuples(index=False):
        stem = f"{row.complex_name}__{row.role}"
        summary_file = run_dir / "per_pose" / f"{stem}.json"
        if summary_file.exists():
            pose_rows.append(json.loads(summary_file.read_text()))
            atom_frames.append(pd.read_csv(run_dir / "per_pose" / f"{stem}.atoms.csv",
                                           dtype={"complex_name": str}))
        else:
            error_file = run_dir / "per_pose" / f"{stem}.error.json"
            reason = json.loads(error_file.read_text())["error"] if error_file.exists() else "not run"
            failures.append({"complex_name": row.complex_name, "role": row.role,
                             "error": reason.splitlines()[0]})
    if not pose_rows:
        raise SystemExit(f"No scored poses under {run_dir}")
    poses = pd.DataFrame(pose_rows)
    poses = work.merge(poses.drop(columns=["ligand_path", "protein_path", "map_path"]),
                       on=["complex_name", "role"], how="inner", validate="1:1")
    atoms = pd.concat(atom_frames, ignore_index=True)
    poses.to_csv(output / "per_pose_qscore.csv", index=False)
    atoms.to_csv(output / "per_atom_qscore.csv", index=False)

    # One row per case: mode 1 vs mode 2, deposited control alongside.
    wide = []
    for case, group in poses.groupby("complex_name", sort=True):
        by_role = {row.role: row for row in group.itertuples(index=False)}
        if "mode1" not in by_role or "mode2" not in by_role:
            continue
        m1, m2 = by_role["mode1"], by_role["mode2"]
        dep = by_role.get("deposited")
        record = {"complex_name": case}
        for label, row in (("mode1", m1), ("mode2", m2)):
            record.update({
                f"{label}_qscore_mean": row.qscore_mean,
                f"{label}_qscore_min": row.qscore_min,
                f"{label}_medoid_final_ncc": row.medoid_final_multiscale_ncc,
                f"{label}_mode_median_ncc": row.mode_multiscale_ncc_median,
                f"{label}_rmsd": row.rmsd,
                f"{label}_pb_valid": row.pb_valid,
                f"{label}_joint_density_pb_supported": row.joint_density_pb_supported,
                f"{label}_mode_size": row.mode_size,
            })
        record["deposited_qscore_mean"] = dep.qscore_mean if dep is not None else np.nan
        wide.append(record)
    paired = pd.DataFrame(wide).merge(
        cases[["complex_name", "map_resolution", "protein_identical"]], on="complex_name", how="left"
    )
    threshold = args.rmsd_threshold
    paired["mode1_correct"] = paired.mode1_rmsd < threshold
    paired["mode2_correct"] = paired.mode2_rmsd < threshold
    paired["outcome"] = np.select(
        [paired.mode1_correct & paired.mode2_correct,
         paired.mode1_correct & ~paired.mode2_correct,
         ~paired.mode1_correct & paired.mode2_correct],
        ["both_correct", "only_mode1_correct", "only_mode2_correct"],
        default="neither_correct",
    )
    paired["delta_qscore_mean"] = paired.mode1_qscore_mean - paired.mode2_qscore_mean
    paired["delta_qscore_min"] = paired.mode1_qscore_min - paired.mode2_qscore_min
    paired["delta_medoid_final_ncc"] = paired.mode1_medoid_final_ncc - paired.mode2_medoid_final_ncc
    paired["delta_mode_median_ncc"] = paired.mode1_mode_median_ncc - paired.mode2_mode_median_ncc
    paired["qscore_and_ncc_agree"] = (
        np.sign(paired.delta_qscore_mean) == np.sign(paired.delta_medoid_final_ncc)
    )
    paired["best_mode_qscore_minus_deposited"] = (
        paired[["mode1_qscore_mean", "mode2_qscore_mean"]].max(axis=1) - paired.deposited_qscore_mean
    )
    paired.to_csv(output / "per_case_mode_comparison.csv", index=False)

    one = paired[paired.mode1_correct != paired.mode2_correct].copy()
    stats = {"rmsd_threshold_A": threshold, "cases_paired": int(len(paired)),
             "outcome_counts": paired.outcome.value_counts().to_dict()}
    if len(paired) >= 3:
        rho, p = spearmanr(paired.delta_qscore_mean, paired.delta_medoid_final_ncc)
        stats["spearman_delta_q_vs_delta_medoid_ncc"] = {"rho": float(rho), "p": float(p)}
    stats["sign_agreement_q_vs_medoid_ncc"] = {
        "agree": int(paired.qscore_and_ncc_agree.sum()), "n": int(len(paired)),
    }
    stats["abs_delta_qscore_mean_quantiles"] = {
        str(q): float(paired.delta_qscore_mean.abs().quantile(q)) for q in (0.25, 0.5, 0.75)
    }
    stats["deposited_qscore_mean_quantiles"] = {
        str(q): float(paired.deposited_qscore_mean.quantile(q)) for q in (0.25, 0.5, 0.75)
    }
    selection = {}
    correct_masks = {}
    for name, (s1, s2) in {
        "qscore_mean": ("mode1_qscore_mean", "mode2_qscore_mean"),
        "qscore_min_atom": ("mode1_qscore_min", "mode2_qscore_min"),
        "medoid_final_ncc": ("mode1_medoid_final_ncc", "mode2_medoid_final_ncc"),
        "mode_median_ncc": ("mode1_mode_median_ncc", "mode2_mode_median_ncc"),
    }.items():
        selection[name], correct_masks[name] = pick_accuracy(one, s1, s2)
    stats["exactly_one_correct_selection"] = selection
    if len(one):
        stats["mcnemar_qscore_mean_vs_medoid_ncc"] = mcnemar_exact(
            correct_masks["qscore_mean"], correct_masks["medoid_final_ncc"]
        )
        sign = np.where(one.mode1_correct, 1.0, -1.0)
        oriented_q = sign * one.delta_qscore_mean.to_numpy()
        oriented_ncc = sign * one.delta_medoid_final_ncc.to_numpy()
        for label, values in (("qscore_mean", oriented_q), ("medoid_final_ncc", oriented_ncc)):
            entry = {"median_correct_minus_incorrect": float(np.median(values))}
            if len(values) >= 6 and np.any(values != 0):
                entry["wilcoxon_p"] = float(wilcoxon(values).pvalue)
            stats[f"correct_minus_incorrect_{label}"] = entry
    stats["failures"] = len(failures)

    # Validation: hard checks fail the command unless --allow-incomplete.
    expected_roles = set(work.role.unique())
    per_case_roles = poses.groupby("complex_name").role.apply(set)
    checks = {
        "all_poses_scored": len(failures) == 0,
        "every_case_has_all_roles": bool((per_case_roles == expected_roles).all())
        and len(per_case_roles) == work.complex_name.nunique(),
        "all_mean_qscores_finite": bool(np.isfinite(poses.qscore_mean.astype(float)).all()),
        "no_probe_points_outside_map": int(poses.n_points_outside_map.sum()) == 0,
        "mode_heavy_atom_counts_equal": bool(cases.mode_ligands_same_atoms.astype(str).str.lower()
                                             .eq("true").all()),
    }
    informational = {
        "cases_with_identical_receptors": int(as_bool(cases.protein_identical).sum()),
        "cases": int(len(cases)),
        "poses_with_removed_target_copies": int((poses.removed_target_copies.fillna("") != "").sum()),
        "atoms_with_short_shells": int((atoms.n_short_shells > 0).sum()),
        "deposited_atom_count_mismatch_cases": int((~as_bool(cases.deposited_matches_mode_atom_count)).sum()),
    }
    validation = {"checks": checks, "informational": informational, "failures": failures[:50]}
    (output / "validation.json").write_text(json.dumps(validation, indent=2, default=str) + "\n")
    (output / "summary.json").write_text(json.dumps(stats, indent=2, default=str) + "\n")
    (output / "provenance.json").write_text(json.dumps({
        "created": now(), "stage": "summarize", "prepared_dir": str(prepared),
        "run_dir": str(run_dir),
        "run_params": json.loads((run_dir / "run_params.json").read_text()),
        "note": "Q-score is an independent atom-level map-fit measure; it is not a pose-correctness label.",
    }, indent=2) + "\n")
    with (output / "SHA256SUMS").open("w") as handle:
        for path in sorted(output.glob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{sha256(path)}  {path.name}\n")

    print(json.dumps(stats, indent=2, default=str))
    print(json.dumps(checks, indent=2))
    if not all(checks.values()) and not args.allow_incomplete:
        raise SystemExit("Validation failed; see validation.json (or pass --allow-incomplete for a pilot)")


# --------------------------------------------------------------------------
# compare-chimerax
# --------------------------------------------------------------------------

def compare_chimerax(args: argparse.Namespace) -> None:
    from scipy.spatial import cKDTree

    ours = pd.read_csv(args.per_atom, dtype={"complex_name": str})
    theirs = pd.read_csv(args.chimerax_csv, dtype={"complex_name": str})
    joined = []
    for (case, role), group in theirs.groupby(["complex_name", "role"]):
        mine = ours[(ours.complex_name == case) & (ours.role == role)]
        if mine.empty:
            continue
        distance, index = cKDTree(mine[["x", "y", "z"]].to_numpy()).query(
            group[["x", "y", "z"]].to_numpy()
        )
        matched = group.assign(
            qscore_ours=mine.qscore.to_numpy()[index], match_distance=distance
        )
        joined.append(matched[matched.match_distance < 0.01])
    if not joined:
        raise SystemExit("No overlapping poses between the two tables")
    joined = pd.concat(joined, ignore_index=True)
    joined.to_csv(args.output, index=False)
    per_pose = joined.groupby(["complex_name", "role"]).agg(
        chimerax=("qscore", "mean"), ours=("qscore_ours", "mean")
    )
    report = {
        "atoms_matched": int(len(joined)),
        "poses_matched": int(len(per_pose)),
        "atom_pearson": float(np.corrcoef(joined.qscore, joined.qscore_ours)[0, 1]),
        "atom_mean_abs_diff": float((joined.qscore - joined.qscore_ours).abs().mean()),
        "pose_mean_abs_diff": float((per_pose.chimerax - per_pose.ours).abs().mean()),
        "pose_max_abs_diff": float((per_pose.chimerax - per_pose.ours).abs().max()),
    }
    if len(per_pose) >= 3:
        report["pose_pearson"] = float(np.corrcoef(per_pose.chimerax, per_pose.ours)[0, 1])
    print(per_pose.round(3).to_string())
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--medoids", required=True, type=Path,
                   help="05_figure5_paahs_mode_posebusters_v1/top2_reproducible_mode_medoids.csv")
    p.add_argument("--mode-metrics", required=True, type=Path,
                   help="08_figure5_paahs_joint_density_pb_multimodal72_v2/per_mode_joint_density_pb.csv")
    p.add_argument("--map-csv", required=True, type=Path,
                   help="CSV with pdb_id/complex_name and map_path of the raw experimental map")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--expected-cases", type=int, default=72)
    p.add_argument("--protein-tolerance", type=float, default=1e-3)
    p.add_argument("--no-deposited-control", action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    p.set_defaults(func=prepare)

    r = sub.add_parser("run")
    r.add_argument("--prepared-dir", required=True, type=Path)
    r.add_argument("--output-dir", type=Path, help="default: <prepared-dir>/qscore_run")
    r.add_argument("--cases", default="", help="comma-separated subset, e.g. a pilot")
    r.add_argument("--limit-cases", type=int, default=0)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--overwrite", action="store_true", help="rescore cases that are already done")
    r.add_argument("--origin-mode", choices=("auto", "nstart", "origin", "sum"), default="auto")
    r.add_argument("--drop-waters", action="store_true")
    r.add_argument("--target-copy-distance", type=float, default=1.5)
    r.add_argument("--sigma", type=float, default=0.6)
    r.add_argument("--max-shell-radius", type=float, default=2.0)
    r.add_argument("--shell-step", type=float, default=0.1)
    r.add_argument("--points-per-shell", type=int, default=8)
    r.add_argument("--trial-points", type=int, default=128)
    r.add_argument("--random-seed", type=int, default=1985)
    r.set_defaults(func=run)

    s = sub.add_parser("summarize")
    s.add_argument("--prepared-dir", required=True, type=Path)
    s.add_argument("--run-dir", type=Path, help="default: <prepared-dir>/qscore_run")
    s.add_argument("--output-dir", type=Path, help="default: <run-dir>/summary")
    s.add_argument("--rmsd-threshold", type=float, default=2.0)
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--allow-incomplete", action="store_true")
    s.set_defaults(func=summarize)

    c = sub.add_parser("compare-chimerax")
    c.add_argument("--per-atom", required=True, type=Path, help="summary/per_atom_qscore.csv")
    c.add_argument("--chimerax-csv", required=True, type=Path)
    c.add_argument("--output", required=True, type=Path)
    c.set_defaults(func=compare_chimerax)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
