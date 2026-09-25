#!/usr/bin/env python3
"""Fixed Test173 experiment: lower-tail atom Q-score for eight final PAAHS poses.

This intentionally leaves ``paahs_crossrun_qscore.py`` and its v1 lock alone.
``score`` reuses the frozen blind pose pool, recalculates per-atom Q-scores,
checks their mean against the completed v1 scoring, and records the linear
25th percentile. ``develop`` compares exactly two new rules to fixed v1
baselines. Neither command reads Local200 outcomes.
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

from evaluation import paahs_crossrun_qscore as v1  # noqa: E402
from evaluation.qscore_core import (  # noqa: E402
    QScoreParameters, atom_qscores, read_density_map,
    read_protein_heavy_atoms, read_sdf_heavy_atoms,
)
from evaluation.paahs_single_run_qscore_pilot import sha256  # noqa: E402

NEW_RULES = ("q25_top", "blend_ncc75_q25_25")
ALL_RULES = ("run1", "ncc_top", "q_top", "blend_q25", *NEW_RULES)
MEAN_TOLERANCE = 1e-5


def _score_case(case: str, rows: list[dict]) -> list[dict]:
    if len({row["map_path"] for row in rows}) != 1:
        raise ValueError(f"{case}: inconsistent maps")
    from scipy.spatial import cKDTree

    density = read_density_map(Path(rows[0]["map_path"]))
    parameters = QScoreParameters()
    proteins: dict[str, pd.DataFrame] = {}
    output = []
    for row in rows:
        protein_path = row["protein_path"]
        if protein_path not in proteins:
            proteins[protein_path] = read_protein_heavy_atoms(Path(protein_path))
        protein = proteins[protein_path]
        ligand = read_sdf_heavy_atoms(Path(row["pose_path"]))
        protein_xyz = protein[["x", "y", "z"]].to_numpy(float)
        ligand_xyz = ligand[["x", "y", "z"]].to_numpy(float)
        if (cKDTree(protein_xyz).query(ligand_xyz)[0] < 0.2).any():
            raise ValueError(f"{case} {row['candidate_id']}: receptor duplicates ligand")
        context = np.vstack((protein_xyz, ligand_xyz))
        atoms = atom_qscores(density, context,
                             range(len(protein_xyz), len(context)), parameters)
        values = atoms.qscore.to_numpy(float)
        if not np.isfinite(values).all() or int(atoms.n_points_outside_map.sum()):
            raise ValueError(f"{case} {row['candidate_id']}: incomplete per-atom Q-score")
        recomputed = float(values.mean())
        original = float(row["qscore_mean"])
        if abs(recomputed - original) > MEAN_TOLERANCE:
            raise ValueError(f"{case} {row['candidate_id']}: mean Q-score changed: "
                             f"original={original:.8f}, recomputed={recomputed:.8f}")
        output.append({
            "complex_name": case, "candidate_id": row["candidate_id"],
            "replicate_order": int(row["replicate_order"]),
            "qscore_mean_recomputed": recomputed,
            "qscore_q25": float(np.quantile(values, 0.25, interpolation="linear")),
            "qscore_heavy_atoms": len(values),
        })
    return output


def score(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    v1._verify_scoring_provenance(args.mean_scored)
    frame = v1._read_blind(args.mean_scored, args.expected_cases)
    if "qscore_mean" not in frame:
        raise ValueError("Mean Q-score column is absent")
    coverage = v1._check_qscore_coverage(frame)
    if coverage["finite_poses"] != coverage["total_poses"]:
        raise ValueError("Q25 comparison requires complete v1 mean Q-score coverage")
    jobs = [(case, group.to_dict("records"))
            for case, group in frame.groupby("complex_name", sort=True)]
    checkpoints = args.output.parent / f"{args.output.stem}.cases"
    meta = {
        "stage": "blind_atom_q25", "mean_scored_sha256": sha256(args.mean_scored),
        "qscore_core_sha256": sha256(ROOT / "evaluation/qscore_core.py"),
        "q25_code_sha256": sha256(Path(__file__)),
        "qscore_parameters": QScoreParameters().__dict__,
        "q25_definition": "numpy quantile, 0.25, linear interpolation, ligand heavy atoms",
        "mean_tolerance": MEAN_TOLERANCE, "uses_reference": False,
    }
    checkpoints.mkdir(parents=True, exist_ok=True)
    metadata = checkpoints / "provenance.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != meta:
            raise ValueError(f"Q25 checkpoint provenance differs: {checkpoints}")
    elif any(checkpoints.iterdir()):
        raise ValueError(f"Q25 checkpoints exist without provenance: {checkpoints}")
    else:
        metadata.write_text(json.dumps(meta, indent=2) + "\n")

    def case_path(case: str) -> Path:
        return checkpoints / f"{hashlib.sha256(case.encode()).hexdigest()[:16]}.csv"

    rows, pending = [], []
    for case, inputs in jobs:
        path = case_path(case)
        if path.exists():
            saved = pd.read_csv(path)
            if len(saved) != 8 or set(saved.candidate_id) != {
                    item["candidate_id"] for item in inputs}:
                raise ValueError(f"Incomplete Q25 checkpoint: {path}")
            rows.extend(saved.to_dict("records"))
        else:
            pending.append((case, inputs))
    if pending:
        print(f"[Q25] resumed {len(jobs) - len(pending)}/{len(jobs)} cases", flush=True)
    if args.workers == 1:
        completed = ((case, _score_case(case, inputs)) for case, inputs in pending)
        for index, (case, result) in enumerate(completed, 1):
            path = case_path(case)
            temporary = path.with_suffix(".tmp")
            pd.DataFrame(result).to_csv(temporary, index=False)
            temporary.replace(path)
            rows.extend(result)
            print(f"[Q25] {index}/{len(pending)} {case}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_score_case, case, inputs): case
                       for case, inputs in pending}
            for index, future in enumerate(as_completed(futures), 1):
                case = futures[future]
                result = future.result()
                path = case_path(case)
                temporary = path.with_suffix(".tmp")
                pd.DataFrame(result).to_csv(temporary, index=False)
                temporary.replace(path)
                rows.extend(result)
                print(f"[Q25] {index}/{len(pending)} {case}", flush=True)
    output = pd.DataFrame(rows).sort_values(["complex_name", "replicate_order"])
    if len(output) != args.expected_cases * 8 or output.duplicated(
            ["complex_name", "candidate_id"]).any():
        raise ValueError("Q25 output is not a complete eight-run grid")
    output.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"Q25_COMPLETE: {len(output)} poses -> {args.output}")


def choose_q25(group: pd.DataFrame, rule: str) -> int:
    group = group.sort_values("replicate_order", kind="stable")
    q = pd.to_numeric(group.qscore_q25, errors="coerce").to_numpy(float)
    if rule == "q25_top":
        values = q
    elif rule == "blend_ncc75_q25_25":
        ncc = pd.to_numeric(group.multiscale_ncc, errors="coerce").to_numpy(float)
        zn, zq = v1._z(ncc), v1._z(q)
        values = 0.75 * zn + 0.25 * zq
        values[~(np.isfinite(zn) & np.isfinite(zq))] = -np.inf
    else:
        raise ValueError(rule)
    values = np.where(np.isfinite(values), values, -np.inf)
    if not np.isfinite(values).any():
        return 1
    return int(group.replicate_order.iloc[int(np.argmax(values))])


def develop(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    v1._verify_scoring_provenance(args.mean_scored)
    mean = v1._read_blind(args.mean_scored, args.expected_cases)
    q25_path = args.q25_scored.parent / f"{args.q25_scored.stem}.provenance.json"
    provenance = json.loads(q25_path.read_text())
    if provenance.get("stage") != "blind_atom_q25" or \
            provenance.get("mean_scored_sha256") != sha256(args.mean_scored) or \
            provenance.get("qscore_core_sha256") != sha256(ROOT / "evaluation/qscore_core.py") or \
            provenance.get("q25_code_sha256") != sha256(Path(__file__)) or \
            provenance.get("uses_reference") is not False:
        raise ValueError("Q25 score provenance does not match v1 mean scoring")
    q25 = pd.read_csv(args.q25_scored, dtype={"complex_name": str})
    required = {"complex_name", "candidate_id", "qscore_q25", "qscore_mean_recomputed"}
    if required - set(q25) or q25.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Invalid Q25 table")
    frame = mean.merge(q25[list(required)], on=["complex_name", "candidate_id"],
                       validate="one_to_one", how="left")
    if len(frame) != len(mean) or frame.qscore_q25.isna().any():
        raise ValueError("Q25 scores do not cover the frozen v1 pose pool")
    selected = []
    for case, group in frame.groupby("complex_name", sort=True):
        for rule in ALL_RULES:
            run = (choose_q25(group, rule) if rule in NEW_RULES else v1.choose(group, rule))
            selected.append({"complex_name": case, "rule": rule,
                             "replicate_order": run})
    joined = v1._join_outcomes(pd.DataFrame(selected), args.outcomes, args.expected_cases)
    summary = v1._summary(joined)
    wide = joined.pivot(index="complex_name", columns="rule", values="success")
    comparisons = []
    from scipy.stats import binomtest
    for rule in NEW_RULES:
        for baseline in ("run1", "ncc_top", "blend_q25"):
            gain = int((wide[rule] & ~wide[baseline]).sum())
            loss = int((~wide[rule] & wide[baseline]).sum())
            comparisons.append({
                "rule": rule, "baseline": baseline,
                "gained_n": gain, "lost_n": loss, "net_n": gain - loss,
                "paired_exact_p": float(binomtest(gain, gain + loss).pvalue)
                if gain + loss else 1.0,
            })
    joined["fold"] = joined.complex_name.map(v1._fold)
    folds = joined.groupby(["fold", "rule"]).success.agg(["sum", "size"]).reset_index()
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "rule_comparison.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "paired_comparison.csv", index=False)
    folds.to_csv(args.output_dir / "five_fold_descriptive.csv", index=False)
    joined.to_csv(args.output_dir / "per_case_choices.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "stage": "exploratory_q25_development_only", "cases": args.expected_cases,
        "new_rules": list(NEW_RULES), "baselines": list(ALL_RULES[:4]),
        "mean_scored_sha256": sha256(args.mean_scored),
        "q25_scored_sha256": sha256(args.q25_scored),
        "outcomes_sha256": sha256(args.outcomes),
        "qscore_core_sha256": sha256(ROOT / "evaluation/qscore_core.py"),
        "v1_selector_sha256": sha256(ROOT / "evaluation/paahs_crossrun_qscore.py"),
        "q25_code_sha256": sha256(Path(__file__)),
        "note": "No new rule is locked; do not apply to Local200 before an explicit v2 freeze.",
    }, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score")
    p.add_argument("--mean-scored", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, default=173)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=score)
    p = sub.add_parser("develop")
    p.add_argument("--mean-scored", type=Path, required=True)
    p.add_argument("--q25-scored", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, default=173)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=develop)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
