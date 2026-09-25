#!/usr/bin/env python3
"""Develop and lock a Q-score rule for choosing one of eight final PAAHS poses.

``prepare`` and ``score`` are blind: they never read reference coordinates or
RMSD. ``develop`` alone reads the Test173 outcome table. ``apply`` uses the
locked rule on a separate cohort without opening an outcome table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.paahs_figure5_mode_qscore import map_lookup  # noqa: E402
from evaluation.paahs_single_run_qscore_pilot import (  # noqa: E402
    blind_csv, score as score_blind, sha256,
)
from evaluation.qscore_core import QScoreParameters  # noqa: E402

RUNS = tuple(range(1, 9))
FOLDS = 5
SUCCESS_A = 2.0
RULES = (
    "run1", "ncc_top", "q_top", "blend_q25", "blend_q50", "blend_q75",
    "q_gate_0p02", "q_gate_0p05",
)


def _read_blind(path: Path, expected_cases: int) -> pd.DataFrame:
    frame = blind_csv(path)
    required = {"complex_name", "candidate_id", "replicate_order", "pose_path",
                "protein_path", "map_path", "multiscale_ncc"}
    if missing := required - set(frame):
        raise ValueError(f"Missing blind columns: {sorted(missing)}")
    frame["replicate_order"] = pd.to_numeric(frame.replicate_order, errors="raise").astype(int)
    if frame.duplicated(["complex_name", "replicate_order"]).any():
        raise ValueError("Duplicate case/replicate")
    if frame.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Duplicate case/candidate_id")
    counts = frame.groupby("complex_name").replicate_order.apply(lambda s: tuple(sorted(s)))
    if len(counts) != expected_cases or any(value != RUNS for value in counts):
        raise ValueError(f"Expected {expected_cases} cases with exactly runs 1..8; got {len(counts)}")
    return frame.sort_values(["complex_name", "replicate_order"], kind="stable").reset_index(drop=True)


def _check_qscore_coverage(frame: pd.DataFrame) -> dict:
    values = pd.to_numeric(frame.qscore_mean, errors="coerce")
    available = frame.assign(q_finite=np.isfinite(values)).groupby("complex_name").q_finite.sum()
    if available.eq(0).any():
        raise ValueError(f"No finite Q-score in cases: {available.index[available.eq(0)].tolist()[:10]}")
    return {"finite_poses": int(np.isfinite(values).sum()),
            "total_poses": len(values),
            "cases_with_partial_qscore": int(available.lt(len(RUNS)).sum())}


def _verify_scoring_provenance(scored_path: Path) -> dict:
    path = scored_path.parent / f"{scored_path.stem}.provenance.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing blind Q-score provenance: {path}")
    provenance = json.loads(path.read_text())
    if provenance.get("stage") != "single_run_blind_qscore" or \
            provenance.get("uses_reference") is not False or \
            provenance.get("qscore_core_sha256") != sha256(ROOT / "evaluation/qscore_core.py") or \
            provenance.get("qscore_parameters") != QScoreParameters().__dict__:
        raise ValueError(f"Blind Q-score provenance differs from current implementation: {path}")
    return provenance


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    source = blind_csv(args.selected_index)
    required = {"complex_name", "replicate_order", "selected_pose_path",
                "protein_path", "multiscale_ncc"}
    if missing := required - set(source):
        raise ValueError(f"Selected index lacks {sorted(missing)}")
    maps = map_lookup(args.map_csv)
    output = source[["complex_name", "replicate_order", "selected_pose_path",
                     "protein_path", "multiscale_ncc"]].copy()
    output = output.rename(columns={"selected_pose_path": "pose_path"})
    output["candidate_id"] = output.replicate_order.map(lambda n: f"run{int(n)}")
    output["map_path"] = output.complex_name.map(
        lambda name: maps.get(name, {}).get("map_path", ""))
    output = _read_blind_frame(output, args.expected_cases)
    missing = [str(path) for path in output[["pose_path", "protein_path", "map_path"]]
               .stack().unique() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} pose/protein/map paths; examples: {missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(
        json.dumps({"stage": "blind_crossrun_prepare", "cases": args.expected_cases,
                    "selected_index_sha256": sha256(args.selected_index),
                    "map_csv_sha256": sha256(args.map_csv), "uses_reference": False}, indent=2) + "\n")
    print(f"Prepared {len(output)} poses from {args.expected_cases} cases: {args.output}")


def _read_blind_frame(frame: pd.DataFrame, expected_cases: int) -> pd.DataFrame:
    if frame.map_path.eq("").any():
        raise ValueError("At least one case has no raw experimental map")
    frame["replicate_order"] = pd.to_numeric(frame.replicate_order, errors="raise").astype(int)
    counts = frame.groupby("complex_name").replicate_order.apply(lambda s: tuple(sorted(s)))
    if len(counts) != expected_cases or any(value != RUNS for value in counts):
        raise ValueError("Cases/replicates must form a complete 8-run grid")
    if frame.duplicated(["complex_name", "candidate_id"]).any():
        raise ValueError("Duplicate candidate IDs")
    return frame.sort_values(["complex_name", "replicate_order"], kind="stable")


def score(args: argparse.Namespace) -> None:
    _read_blind(args.candidates, args.expected_cases)
    score_blind(argparse.Namespace(candidates=args.candidates, output=args.output,
                                   workers=args.workers))


def _z(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), -np.inf)
    finite = np.isfinite(values)
    if finite.any():
        scale = float(values[finite].std(ddof=0))
        result[finite] = ((values[finite] - values[finite].mean()) / scale
                          if scale > 1e-12 else 0.0)
    return result


def choose(group: pd.DataFrame, rule: str) -> int:
    """Return replicate_order; stable ties prefer the earliest run."""
    group = group.sort_values("replicate_order", kind="stable")
    if rule == "run1":
        return 1
    ncc = pd.to_numeric(group.multiscale_ncc, errors="coerce").to_numpy(float)
    q = pd.to_numeric(group.qscore_mean, errors="coerce").to_numpy(float)
    if rule == "ncc_top":
        values = ncc
    elif rule == "q_top":
        values = q
    elif rule.startswith("blend_q"):
        weight = {"blend_q25": .25, "blend_q50": .5, "blend_q75": .75}[rule]
        zn, zq = _z(ncc), _z(q)
        values = (1 - weight) * zn + weight * zq
        values[~(np.isfinite(zn) & np.isfinite(zq))] = -np.inf
    elif rule.startswith("q_gate_"):
        threshold = {"q_gate_0p02": .02, "q_gate_0p05": .05}[rule]
        if not np.isfinite(q[0]) or not np.isfinite(q).any():
            return 1
        best = int(np.nanargmax(q))
        return int(group.replicate_order.iloc[best]) if q[best] - q[0] >= threshold else 1
    else:
        raise ValueError(f"Unknown rule: {rule}")
    values = np.where(np.isfinite(values), values, -np.inf)
    if not np.isfinite(values).any():
        return 1  # fixed missing-score fallback
    return int(group.replicate_order.iloc[int(np.argmax(values))])


def _selected(scored: pd.DataFrame, rules: tuple[str, ...] = RULES) -> pd.DataFrame:
    rows = []
    for case, group in scored.groupby("complex_name", sort=True):
        for rule in rules:
            rows.append({"complex_name": case, "rule": rule,
                         "replicate_order": choose(group, rule)})
    return pd.DataFrame(rows)


def _join_outcomes(selected: pd.DataFrame, outcome_path: Path, expected_cases: int) -> pd.DataFrame:
    outcomes = pd.read_csv(outcome_path, dtype={"complex_name": str})
    required = {"complex_name", "replicate_order", "reference_rmsd_A"}
    if missing := required - set(outcomes):
        raise ValueError(f"Outcome table lacks {sorted(missing)}")
    outcomes = outcomes[list(required)].copy()
    outcomes.complex_name = outcomes.complex_name.str.strip().str.upper()
    outcomes.replicate_order = pd.to_numeric(outcomes.replicate_order, errors="raise").astype(int)
    if outcomes.duplicated(["complex_name", "replicate_order"]).any():
        raise ValueError("Duplicate outcomes")
    if set(outcomes.complex_name) != set(selected.complex_name):
        raise ValueError("Scored and outcome cohorts differ")
    if len(outcomes) != expected_cases * len(RUNS):
        raise ValueError("Outcome table is not a complete 8-run grid")
    joined = selected.merge(outcomes, on=["complex_name", "replicate_order"],
                            validate="many_to_one", how="left")
    if joined.reference_rmsd_A.isna().any():
        raise ValueError("At least one selected outcome is absent")
    joined["success"] = joined.reference_rmsd_A.lt(SUCCESS_A)
    return joined


def _fold(case: str) -> int:
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16) % FOLDS


def _summary(joined: pd.DataFrame) -> pd.DataFrame:
    return joined.groupby("rule", sort=False).agg(
        cases_n=("complex_name", "size"), success_n=("success", "sum"),
        median_rmsd_A=("reference_rmsd_A", "median")
    ).assign(success_pct=lambda f: 100 * f.success_n / f.cases_n).reset_index()


def _best(summary: pd.DataFrame) -> str:
    order = {name: index for index, name in enumerate(RULES)}
    return sorted(summary.itertuples(index=False),
                  key=lambda row: (-row.success_n, order[row.rule]))[0].rule


def develop(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    scoring_provenance = _verify_scoring_provenance(args.scored)
    scored = _read_blind(args.scored, args.expected_cases)
    if "qscore_mean" not in scored:
        raise ValueError("Blind score table lacks qscore_mean")
    coverage = _check_qscore_coverage(scored)
    selected = _selected(scored)
    joined = _join_outcomes(selected, args.outcomes, args.expected_cases)
    summary = _summary(joined)
    joined["fold"] = joined.complex_name.map(_fold)
    cv = []
    for fold in range(FOLDS):
        train = joined[joined.fold.ne(fold)]
        held = joined[joined.fold.eq(fold)]
        choice = _best(_summary(train))
        cv.append({"fold": fold, "train_cases": train.complex_name.nunique(),
                   "held_cases": held.complex_name.nunique(), "chosen_rule": choice,
                   "chosen_success_n": int(held.loc[held.rule.eq(choice), "success"].sum()),
                   "run1_success_n": int(held.loc[held.rule.eq("run1"), "success"].sum())})
    cv_frame = pd.DataFrame(cv)
    best_full = _best(summary)
    baseline_n = int(summary.loc[summary.rule.eq("run1"), "success_n"].iloc[0])
    best_n = int(summary.loc[summary.rule.eq(best_full), "success_n"].iloc[0])
    cv_gain = int((cv_frame.chosen_success_n - cv_frame.run1_success_n).sum())
    # Predeclared conservative gate: a Q-score rule is locked only if it beats
    # run 1 on both all development cases and out-of-fold evaluation.
    locked = best_full if best_n > baseline_n and cv_gain > 0 else "run1"
    args.output_dir.mkdir(parents=True)
    summary.to_csv(args.output_dir / "rule_sweep.csv", index=False)
    cv_frame.to_csv(args.output_dir / "five_fold_cv.csv", index=False)
    joined.to_csv(args.output_dir / "development_selections.csv", index=False)
    lock = {
        "protocol": "paahs_crossrun_qscore_v1", "development_cases": args.expected_cases,
        "candidate_runs": list(RUNS), "success_threshold_A": SUCCESS_A,
        "candidate_rules": list(RULES), "folds": FOLDS,
        "fold_rule": "sha256(uppercase_complex_name)[:8] integer modulo 5",
        "primary_metric": "symmetry-aware top1 RMSD < 2 A",
        "selection_rule": locked, "best_development_rule": best_full,
        "lock_gate": "full-cohort and five-fold held-out net gains must both exceed zero",
        "missing_score_fallback": "run1",
        "tie_break": "earliest replicate_order; rule tie follows candidate_rules order",
        "development_results": {"run1_success_n": baseline_n,
                                "best_rule_success_n": best_n, "cv_net_gain_n": cv_gain,
                                "qscore_coverage": coverage},
        "scored_sha256": sha256(args.scored),
        "blind_candidate_sha256": scoring_provenance["candidate_sha256"],
        "outcomes_sha256": sha256(args.outcomes),
        "qscore_core_sha256": sha256(ROOT / "evaluation/qscore_core.py"),
        "qscore_scoring_sha256": sha256(ROOT / "evaluation/paahs_single_run_qscore_pilot.py"),
        "map_lookup_sha256": sha256(ROOT / "evaluation/paahs_figure5_mode_qscore.py"),
        "selection_code_sha256": sha256(Path(__file__)),
        "qscore_parameters": QScoreParameters().__dict__,
        "uses_reference_for_development_only": True,
    }
    (args.output_dir / "locked_rule.json").write_text(json.dumps(lock, indent=2) + "\n")
    print(json.dumps({"best_development_rule": best_full, "locked_rule": locked,
                      "run1_success_n": baseline_n, "best_rule_success_n": best_n,
                      "cv_net_gain_n": cv_gain}, indent=2))


def apply(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    lock = json.loads(args.lock.read_text())
    if lock.get("protocol") != "paahs_crossrun_qscore_v1":
        raise ValueError("Unknown locked protocol")
    _verify_scoring_provenance(args.scored)
    if lock["selection_code_sha256"] != sha256(Path(__file__)) or \
            lock["qscore_core_sha256"] != sha256(ROOT / "evaluation/qscore_core.py") or \
            lock["qscore_scoring_sha256"] != sha256(
                ROOT / "evaluation/paahs_single_run_qscore_pilot.py") or \
            lock["map_lookup_sha256"] != sha256(
                ROOT / "evaluation/paahs_figure5_mode_qscore.py"):
        raise ValueError("Code differs from the locked development version")
    scored = _read_blind(args.scored, args.expected_cases)
    if "qscore_mean" not in scored:
        raise ValueError("Blind score table lacks qscore_mean")
    coverage = _check_qscore_coverage(scored)
    chosen = _selected(scored, (lock["selection_rule"],))
    chosen = chosen.merge(scored, on=["complex_name", "replicate_order"],
                          validate="one_to_one", how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}.provenance.json").write_text(json.dumps({
        "stage": "blind_locked_application", "lock_sha256": sha256(args.lock),
        "scored_sha256": sha256(args.scored), "qscore_coverage": coverage,
        "uses_reference": False,
    }, indent=2) + "\n")
    print(f"Applied {lock['selection_rule']} to {args.expected_cases} blinded cases")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selected-index", type=Path, required=True)
    p.add_argument("--map-csv", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("score")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=score)
    p = sub.add_parser("develop")
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, default=173)
    p.add_argument("--output-dir", type=Path, required=True)
    p.set_defaults(func=develop)
    p = sub.add_parser("apply")
    p.add_argument("--scored", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--expected-cases", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
