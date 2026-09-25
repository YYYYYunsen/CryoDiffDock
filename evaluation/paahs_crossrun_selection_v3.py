#!/usr/bin/env python3
"""Exploratory v3 reranking of eight final PAAHS poses.

``features`` and ``apply`` are reference-blind. ``develop`` alone reads RMSD.
The already-locked v1 Q-score protocol is neither edited nor reselected here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import paahs_crossrun_qscore as v1  # noqa: E402
from evaluation.paahs_single_run_qscore_pilot import blind_csv, sha256  # noqa: E402

PROTOCOL = "paahs_crossrun_selection_v3"
RULES = (
    "v1_blend_q25",
    "v3_interaction",
    "v3_interaction_support",
    "v3_interaction_contact_steric",
)
FEATURES = (
    "interaction_geometry_quality", "contact_fraction_4A",
    "cluster_support_fraction", "relative_vdw_violation_count",
)
CLUSTER_CUTOFF_A = 2.0
SUCCESS_A = 2.0
FOLDS = 5
CODE_DEPENDENCIES = (
    "evaluation/paahs_crossrun_selection_v3.py",
    "evaluation/paahs_crossrun_qscore.py",
    "evaluation/audit_pose_ensemble_uncertainty.py",
    "evaluation/audit_multirun_pose_modes.py",
    "cryodiffdock_pocket/interactions.py",
    "cryodiffdock_pocket/physics.py",
)


def code_hashes() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in CODE_DEPENDENCIES}


def read_grid(path: Path, expected_cases: int, *, require_features: bool) -> pd.DataFrame:
    frame = v1._read_blind(path, expected_cases)
    required = {"qscore_mean", "multiscale_ncc"}
    if require_features:
        required.update(FEATURES)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing score/features: {sorted(missing)}")
    for column in required:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(float)).all():
            raise ValueError(f"Nonfinite {column} in blind grid")
        frame[column] = values
    return frame


def _score_case(rows: list[dict]) -> list[dict]:
    # Chemistry imports are kept inside this worker: development/apply can run
    # on a machine without RDKit or spyrmsd after blind features are generated.
    from scipy.spatial import cKDTree
    from cryodiffdock_pocket.interactions import (
        read_protein_interaction_sites, score_pose_interactions,
    )
    from cryodiffdock_pocket.physics import (
        read_protein_heavy_atoms, relative_vdw_clashes,
    )
    from evaluation.evaluate_ligand_accuracy import read_heavy_molecule
    from evaluation.audit_pose_ensemble_uncertainty import symmetry_pairwise_rmsd
    from evaluation.audit_multirun_pose_modes import stable_complete_linkage_labels

    rows = sorted(rows, key=lambda row: int(row["replicate_order"]))
    molecules = [read_heavy_molecule(Path(row["pose_path"])) for row in rows]
    distances = symmetry_pairwise_rmsd(molecules)
    labels = stable_complete_linkage_labels(distances, CLUSTER_CUTOFF_A)
    support = np.asarray([(labels == label).sum() / len(labels) for label in labels])
    protein_cache = {}
    results = []
    for row, molecule, support_fraction in zip(rows, molecules, support):
        protein_path = str(row["protein_path"])
        if protein_path not in protein_cache:
            xyz, elements = read_protein_heavy_atoms(protein_path)
            protein_cache[protein_path] = (
                xyz, elements, cKDTree(xyz),
                read_protein_interaction_sites(protein_path),
            )
        protein_xyz, protein_elements, tree, sites = protein_cache[protein_path]
        ligand_xyz = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
        ligand_elements = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        neighbours = tree.query_ball_point(ligand_xyz, 6.0)
        local_ids = sorted({index for atom_ids in neighbours for index in atom_ids})
        if local_ids:
            local_xyz = protein_xyz[local_ids]
            local_elements = [protein_elements[index] for index in local_ids]
            violations, _, _ = relative_vdw_clashes(
                local_xyz, local_elements, ligand_xyz, ligand_elements,
            )
        else:
            violations = 0
        contact_fraction = float(np.mean([
            bool(tree.query_ball_point(atom, 4.0)) for atom in ligand_xyz
        ]))
        interactions = score_pose_interactions(sites, row["pose_path"])
        results.append({
            "complex_name": row["complex_name"],
            "replicate_order": int(row["replicate_order"]),
            "interaction_geometry_quality": float(interactions["interaction_geometry_quality"]),
            "contact_fraction_4A": contact_fraction,
            "cluster_support_fraction": float(support_fraction),
            "relative_vdw_violation_count": int(violations),
        })
    return results


def features(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    v1._verify_scoring_provenance(args.qscored)
    scored = read_grid(args.qscored, args.expected_cases, require_features=False)
    for path in scored[["pose_path", "protein_path"]].stack().unique():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    jobs = [group.to_dict("records") for _, group in scored.groupby("complex_name", sort=True)]
    measured = []
    if args.workers == 1:
        for index, job in enumerate(jobs, 1):
            measured.extend(_score_case(job))
            if index % 20 == 0 or index == len(jobs):
                print(f"[v3 features] {index}/{len(jobs)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(_score_case, job): job[0]["complex_name"] for job in jobs}
            for index, future in enumerate(as_completed(pending), 1):
                try:
                    measured.extend(future.result())
                except Exception as error:
                    raise RuntimeError(f"Feature scoring failed for {pending[future]}") from error
                if index % 20 == 0 or index == len(jobs):
                    print(f"[v3 features] {index}/{len(jobs)}", flush=True)
    feature_frame = pd.DataFrame(measured)
    merged = scored.merge(feature_frame, on=["complex_name", "replicate_order"],
                          how="left", validate="one_to_one")
    merged = read_grid_frame(merged, args.expected_cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(
        json.dumps({"stage": "blind_v3_features", "uses_reference": False,
                    "qscored_sha256": sha256(args.qscored),
                    "code_sha256": code_hashes(), "cases": args.expected_cases,
                    "cluster_cutoff_A": CLUSTER_CUTOFF_A}, indent=2) + "\n"
    )
    print(f"V3_BLIND_FEATURES_OK: {args.output}")


def read_grid_frame(frame: pd.DataFrame, expected_cases: int) -> pd.DataFrame:
    # Use the same forbidden-outcome and 8-run checks as v1 without writing a file.
    if any(any(token in column.lower() for token in
               ("reference", "native", "rmsd", "outcome", "success")) for column in frame):
        raise ValueError("Outcome-bearing column in blind feature table")
    if len(frame) != expected_cases * 8 or frame.duplicated(["complex_name", "replicate_order"]).any():
        raise ValueError("Blind feature table is not a complete 8-run grid")
    if frame["complex_name"].nunique() != expected_cases:
        raise ValueError("Wrong number of cases")
    for column in (*FEATURES, "qscore_mean", "multiscale_ncc"):
        if column not in frame or not np.isfinite(pd.to_numeric(frame[column], errors="coerce")).all():
            raise ValueError(f"Missing/nonfinite {column}")
    return frame.sort_values(["complex_name", "replicate_order"], kind="stable")


def feature_provenance(path: Path) -> dict:
    provenance_path = path.parent / f"{path.stem}.provenance.json"
    payload = json.loads(provenance_path.read_text())
    if payload.get("stage") != "blind_v3_features" or payload.get("uses_reference") is not False:
        raise ValueError("Not a blind v3 feature table")
    if payload.get("code_sha256") != code_hashes():
        raise ValueError("v3 feature code differs from provenance")
    return payload


def _z(values: np.ndarray) -> np.ndarray:
    scale = float(values.std(ddof=0))
    return (values - values.mean()) / scale if scale > 1e-12 else np.zeros(len(values))


def choose(group: pd.DataFrame, rule: str) -> int:
    group = group.sort_values("replicate_order", kind="stable")
    if rule == "v1_blend_q25":
        return v1.choose(group, "blend_q25")
    if rule not in RULES:
        raise ValueError(rule)
    z = {column: _z(group[column].to_numpy(float))
         for column in ("multiscale_ncc", "qscore_mean", *FEATURES)}
    score = 0.75 * z["multiscale_ncc"] + 0.25 * z["qscore_mean"]
    score += 0.25 * z["interaction_geometry_quality"]
    if rule == "v3_interaction_support":
        score += 0.15 * z["cluster_support_fraction"]
    elif rule == "v3_interaction_contact_steric":
        score += 0.15 * z["contact_fraction_4A"]
        score -= 0.20 * z["relative_vdw_violation_count"]
    return int(group.replicate_order.iloc[int(np.argmax(score))])


def selections(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"complex_name": case, "rule": rule, "replicate_order": choose(group, rule)}
        for case, group in frame.groupby("complex_name", sort=True)
        for rule in RULES
    ])


def comparison(joined: pd.DataFrame, rule: str) -> dict:
    baseline = joined[joined.rule.eq("v1_blend_q25")].set_index("complex_name").success
    candidate = joined[joined.rule.eq(rule)].set_index("complex_name").success
    gained = int((~baseline & candidate).sum())
    lost = int((baseline & ~candidate).sum())
    return {"rule": rule, "baseline": "v1_blend_q25", "gained_n": gained,
            "lost_n": lost, "net_n": gained - lost,
            "paired_exact_p": float(binomtest(gained, gained + lost, .5).pvalue)
            if gained + lost else 1.0}


def develop(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    provenance = feature_provenance(args.features)
    frame = read_grid(args.features, args.expected_cases, require_features=True)
    selected = selections(frame)
    joined = v1._join_outcomes(selected, args.outcomes, args.expected_cases)
    summary = v1._summary(joined)
    order = {rule: index for index, rule in enumerate(RULES)}
    def best(block: pd.DataFrame) -> str:
        counts = block.groupby("rule").success.sum().to_dict()
        return sorted(RULES, key=lambda rule: (-counts[rule], order[rule]))[0]
    joined["fold"] = joined.complex_name.map(v1._fold)
    folds = []
    for fold in range(FOLDS):
        train = joined[joined.fold.ne(fold)]
        held = joined[joined.fold.eq(fold)]
        chosen = best(train)
        folds.append({"fold": fold, "chosen_rule": chosen,
                      "held_cases": held.complex_name.nunique(),
                      "chosen_success_n": int(held.loc[held.rule.eq(chosen), "success"].sum()),
                      "v1_success_n": int(held.loc[held.rule.eq("v1_blend_q25"), "success"].sum())})
    cv = pd.DataFrame(folds)
    full_best = best(joined)
    cv_gain = int((cv.chosen_success_n - cv.v1_success_n).sum())
    paired = pd.DataFrame([comparison(joined, rule) for rule in RULES[1:]])
    full_gain = int(paired.set_index("rule").loc[full_best, "net_n"]) if full_best != RULES[0] else 0
    locked = full_best if full_gain > 0 and cv_gain > 0 else RULES[0]
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "rule_summary.csv", index=False)
    cv.to_csv(args.output_dir / "five_fold_cv.csv", index=False)
    paired.to_csv(args.output_dir / "paired_vs_v1.csv", index=False)
    joined.to_csv(args.output_dir / "development_selections.csv", index=False)
    lock = {"protocol": PROTOCOL, "development_cases": args.expected_cases,
            "selection_rule": locked, "best_development_rule": full_best,
            "candidate_rules": list(RULES), "rule_weights_fixed_before_outcomes": True,
            "success_threshold_A": SUCCESS_A, "cluster_cutoff_A": CLUSTER_CUTOFF_A,
            "comparison": "locked_v1_blend_q25", "full_gain_vs_v1_n": full_gain,
            "cv_gain_vs_v1_n": cv_gain,
            "lock_gate": "positive full-cohort and rule-selection CV gains over v1",
            "development_features_sha256": sha256(args.features),
            "development_outcomes_sha256": sha256(args.outcomes),
            "feature_provenance": provenance, "code_sha256": code_hashes(),
            "uses_reference_for_development_only": True}
    (args.output_dir / "locked_rule.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"best": full_best, "locked": locked,
                      "full_gain_vs_v1_n": full_gain, "cv_gain_vs_v1_n": cv_gain}, indent=2))


def apply(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    provenance = feature_provenance(args.features)
    lock = json.loads(args.lock.read_text())
    if lock.get("protocol") != PROTOCOL or lock.get("code_sha256") != code_hashes():
        raise ValueError("v3 code differs from locked development version")
    if lock.get("selection_rule") not in RULES:
        raise ValueError("Unknown locked selection rule")
    frame = read_grid(args.features, args.expected_cases, require_features=True)
    chosen = pd.DataFrame([
        {"complex_name": case, "replicate_order": choose(group, lock["selection_rule"])}
        for case, group in frame.groupby("complex_name", sort=True)
    ])
    chosen["rule"] = lock["selection_rule"]
    chosen = chosen.merge(frame, on=["complex_name", "replicate_order"],
                          how="left", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(
        json.dumps({"stage": "blind_locked_v3_application", "uses_reference": False,
                    "lock_sha256": sha256(args.lock),
                    "features_sha256": sha256(args.features),
                    "feature_provenance": provenance}, indent=2) + "\n"
    )
    print(f"V3_BLIND_APPLICATION_OK: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("features")
    p.add_argument("--qscored", required=True, type=Path)
    p.add_argument("--expected-cases", required=True, type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=features)
    p = sub.add_parser("develop")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--outcomes", required=True, type=Path)
    p.add_argument("--expected-cases", type=int, default=173)
    p.add_argument("--output-dir", required=True, type=Path)
    p.set_defaults(func=develop)
    p = sub.add_parser("apply")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--lock", required=True, type=Path)
    p.add_argument("--expected-cases", type=int, default=200)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
