#!/usr/bin/env python3
"""Compare canonical Global Top-1 with existing cascade endpoints, by parent.

No docking or selector fitting. Default Global evaluation uses the same
symmetry-corrected, non-superposed RMSD implementation as Local evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def read_table(path):
    return pd.read_csv(path, sep=None, engine="python")


def numeric(series):
    values = pd.to_numeric(series, errors="raise")
    if ((values.dropna() < 0) | ~np.isfinite(values.dropna())).any():
        raise ValueError("Invalid negative or infinite RMSD")
    return values


def parents(path, ids):
    table = read_table(path)
    key = next((c for c in ("pdb_id", "complex_name") if c in table), None)
    if key is None or table[key].isna().any():
        raise ValueError(f"Missing parent IDs: {path}")
    table["pdb_id"] = table[key].astype(str).str.strip().str.upper()
    if table.pdb_id.duplicated().any():
        raise ValueError(f"Duplicate parent IDs: {path}")
    return table.set_index("pdb_id").reindex(ids)


def resolved(value, base):
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def reference_paths(path, ids, column="reference_ligand"):
    """Join references by identity, never by manifest row number.

    Multiple branch rows are allowed only if they identify the same reference
    file for their parent. Relative paths belong to this manifest's directory.
    """
    table = read_table(path)
    if column not in table:
        raise ValueError(
            f"Reference manifest {path} lacks {column!r}. Columns: {table.columns.tolist()}. "
            "Pass --reference-manifest pointing to the evaluation manifest containing native "
            "reference paths (not the blind input); use --reference-column for an explicit column name."
        )
    key = next((c for c in ("parent_complex_name", "pdb_id", "complex_name") if c in table), None)
    if key is None or table[key].isna().any():
        raise ValueError(f"Reference manifest {path} has missing parent/branch IDs")
    table["_parent"] = table[key].astype(str).str.strip().str.upper()
    if key == "complex_name":
        table["_parent"] = table._parent.str.replace(r"__GLOBAL_RANK\d+$", "", regex=True)
    table = table[table._parent.isin(ids)].copy()
    if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any():
        raise ValueError(f"Empty reference paths in {path}")
    table["_reference"] = table[column].map(lambda value: str(resolved(value, path.parent)))
    conflicts = table.groupby("_parent")._reference.nunique()
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting reference paths for parents: {conflicts[conflicts.gt(1)].index.tolist()}")
    refs = table.drop_duplicates("_parent").set_index("_parent")._reference.reindex(ids)
    if refs.isna().any():
        raise ValueError(f"Missing reference parents in {path}: {refs.index[refs.isna()].tolist()}")
    missing = [(pdb, value) for pdb, value in refs.items() if not Path(value).is_file()]
    if missing:
        raise ValueError(f"Reference files unavailable; fix manifest paths before evaluation: {missing[:10]}")
    return refs


def global_top1(args, ids):
    refs = None
    if not args.global_rmsd_column:
        refs = reference_paths(args.reference_manifest or args.source_reveal, ids, args.reference_column)
    scores = read_table(args.global_scores)
    key = next((c for c in ("complex", "complex_name", "pdb_id", "complex_key") if c in scores), None)
    pose = next((c for c in ("sdf_path", "prediction_path", "pose_path") if c in scores), None)
    if key is None or pose is None or args.score_column not in scores:
        raise ValueError(f"Global columns: {scores.columns.tolist()}; need ID, pose path, ranking score")
    scores["_parent"] = scores[key].astype(str).str.strip().str.upper()
    scores = scores[scores._parent.isin(ids)].copy()
    scores["_score"] = pd.to_numeric(scores[args.score_column], errors="raise")
    if not np.isfinite(scores._score).all() or scores[pose].isna().any():
        raise ValueError("Global candidate has missing/nonfinite score or pose path")
    if scores.duplicated(["_parent", pose]).any():
        raise ValueError("Duplicate Global candidate pose paths")
    counts = scores.groupby("_parent").size().reindex(ids, fill_value=0)
    if not counts.eq(args.pool_size).all():
        raise ValueError(f"Expected {args.pool_size} candidates per parent: {counts[counts.ne(args.pool_size)].to_dict()}")
    order, ascending = ["_score"], [False]
    if "original_rank" in scores:
        scores["_original_rank"] = pd.to_numeric(scores.original_rank, errors="raise")
        order.append("_original_rank")
        ascending.append(True)
    # Match build_cascade_predicted_center_manifest.py: stable input order
    # breaks residual ties; never rank by reference-dependent metrics.
    selected = scores.sort_values(order, ascending=ascending, kind="stable").groupby("_parent", sort=False).head(1).set_index("_parent").reindex(ids)
    reveal = read_table(args.source_reveal)
    reveal["_parent"] = reveal.parent_complex_name.astype(str).str.strip().str.upper()
    reveal["global_rank"] = pd.to_numeric(reveal.global_rank, errors="raise")
    reveal = reveal[reveal.global_rank.eq(1) & reveal._parent.isin(ids)]
    if reveal._parent.duplicated().any():
        raise ValueError("Duplicate rank-1 source branches")
    reveal = reveal.set_index("_parent").reindex(ids)
    if reveal.global_pose.isna().any():
        raise ValueError("Missing source rank1: cannot verify canonical Global/cascade pairing")
    if args.global_rmsd_column and args.global_rmsd_column not in selected:
        raise ValueError(f"Requested RMSD column missing; columns: {selected.columns.tolist()}")
    rows = []
    for pdb in ids:
        row, source = selected.loc[pdb], reveal.loc[pdb]
        prediction = resolved(row[pose], args.global_scores.parent)
        source_pose = resolved(source.global_pose, args.source_reveal.parent)
        if prediction != source_pose:
            raise ValueError(f"{pdb}: canonical rank1 pose does not match cascade source: {prediction} != {source_pose}")
        error, rmsd = "", np.nan
        if args.global_rmsd_column:
            rmsd = numeric(pd.Series([row[args.global_rmsd_column]])).iloc[0]
            if pd.isna(rmsd):
                error = "missing_precomputed_rmsd"
        else:
            from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd
            # Import dependencies outside the per-case exception: missing
            # software must stop the run, not turn the whole cohort into failures.
            import rdkit  # noqa: F401
            import spyrmsd  # noqa: F401
            try:
                reference = Path(refs.loc[pdb])
                rmsd = symmetry_rmsd(read_heavy_molecule(reference), read_heavy_molecule(prediction))
                numeric(pd.Series([rmsd]))
                if not np.isfinite(rmsd):
                    raise ValueError("Nonfinite computed RMSD")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rmsd = np.nan
        rows.append(dict(pdb_id=pdb, global_top1_rmsd_A=rmsd, global_pose=str(prediction), reference_ligand=refs.loc[pdb] if refs is not None else "", global_score=float(row._score), candidate_count=int(counts[pdb]), evaluation_error=error))
    return pd.DataFrame(rows).set_index("pdb_id")


def summarize(frame, repeats=10000, seed=20260908, *, oracle_methods=("oracle_best5",), reference_methods=("global", "rank1")):
    n = len(frame)
    draws = np.random.default_rng(seed).integers(0, n, size=(repeats, n))
    summaries, changes, transitions = [], [], []
    contrasts = []
    for index, reference in enumerate(reference_methods):
        if reference in frame:
            contrasts += [(reference, m) for m in frame if m not in reference_methods[:index+1]]
    for cutoff in (2.0, 5.0):
        success = frame.lt(cutoff) & frame.notna()
        for method in frame:
            hits = success[method].to_numpy()
            low, high = np.quantile(100 * hits[draws].mean(axis=1), [.025, .975])
            summaries.append(dict(method=method, cutoff_A=cutoff, success_count=int(hits.sum()), denominator=n, success_pct=100*hits.mean(), ci_low_pct=low, ci_high_pct=high, scored_n=int(frame[method].notna().sum()), median_rmsd_A_among_scored=frame[method].median(), oracle=method in oracle_methods))
        for left, right in contrasts:
            a, b = success[left].to_numpy(), success[right].to_numpy()
            gained, lost = ~a & b, a & ~b
            delta = b.astype(float)-a.astype(float)
            low, high = np.quantile(100*delta[draws].mean(axis=1), [.025,.975])
            oracle = right in oracle_methods
            g, l = int(gained.sum()), int(lost.sum())
            # Exact two-sided binomial McNemar test, p0=1/2.
            p = np.nan if oracle else (min(1., 2*sum(math.comb(g+l,k) for k in range(min(g,l)+1))/2**(g+l)) if g+l else 1.)
            changes.append(dict(reference=left, method=right, cutoff_A=cutoff, denominator=n, gained=g, lost=l, delta_pp=100*delta.mean(), ci_low_pp=low, ci_high_pp=high, mcnemar_p=p, oracle_descriptive_only=oracle))
            for i, pdb in enumerate(frame.index):
                transitions.append(dict(pdb_id=pdb, reference=left, method=right, cutoff_A=cutoff, reference_rmsd_A=frame[left].iloc[i], method_rmsd_A=frame[right].iloc[i], outcome="gained" if gained[i] else "lost" if lost[i] else "both_success" if a[i] else "both_failure"))
    changes = pd.DataFrame(changes)
    changes["holm_p"] = np.nan
    # One disclosed family: all non-oracle contrasts across both thresholds.
    ix = changes.mcnemar_p.dropna().sort_values().index
    adjusted = np.maximum.accumulate(changes.loc[ix,"mcnemar_p"].to_numpy()*np.arange(len(ix),0,-1))
    changes.loc[ix,"holm_p"] = np.minimum(1,adjusted)
    return pd.DataFrame(summaries), changes, pd.DataFrame(transitions)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "global-scores", "source-reveal", "cascade-per-parent", "cross-per-parent", "cross-summary", "output-dir"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--score-column", default="combined_at_best_weights")
    p.add_argument("--reference-manifest", type=Path, help="Independent evaluation manifest with native reference paths; defaults to source reveal")
    p.add_argument("--reference-column", default="reference_ligand")
    p.add_argument("--global-rmsd-column", help="Explicit trusted per-pose RMSD column; default evaluates selected SDFs")
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--expected-n", type=int, default=174)
    p.add_argument("--bootstrap-repeats", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--selector-status", choices=["exploratory","validation_frozen"], default="exploratory")
    p.add_argument("--allow-global-evaluation-failures", action="store_true", help="Explicitly retain evaluation failures in fixed denominator; default stops for review")
    args = p.parse_args(argv)
    ids = [line.split()[0].upper() for line in args.cohort.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(ids)!=args.expected_n or len(set(ids))!=len(ids) or not ids or args.bootstrap_repeats < 1:
        raise ValueError("Invalid cohort or bootstrap count")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; use a new output path")
    global_rows = global_top1(args,ids)
    global_rows.to_csv(args.output_dir/'global_top1_audit.csv')
    if global_rows.global_top1_rmsd_A.isna().any() and not args.allow_global_evaluation_failures:
        failures = global_rows.loc[global_rows.global_top1_rmsd_A.isna(), 'evaluation_error']
        raise ValueError(
            f"Global evaluation failed for {len(failures)}/{len(ids)} parents. "
            f"Error counts: {failures.value_counts().to_dict()}. "
            f"Audit: {args.output_dir / 'global_top1_audit.csv'}. "
            "Fix the reported input/evaluation error before rerunning; do not automatically count it as a docking failure."
        )
    cascade = parents(args.cascade_per_parent,ids)
    cross = parents(args.cross_per_parent,ids)
    cross_meta = json.loads(args.cross_summary.read_text())
    if cross_meta.get("requested_cohort_n") != len(ids) or "incomplete_top_k_policy" not in cross_meta:
        raise ValueError("Cross-selector summary cohort/policy missing or inconsistent")
    frame = pd.DataFrame(index=pd.Index(ids,name='pdb_id'))
    frame['global'] = global_rows.global_top1_rmsd_A
    frame['rank1'] = numeric(cascade.top1_cascade_rmsd_A)
    frame['multiscale_ncc'] = numeric(cross.max_selected_multiscale_ncc_selected_rmsd_A)
    frame['basin_q25_ncc'] = numeric(cross.max_basin_q25_ncc_selected_rmsd_A)
    frame['oracle_best5'] = numeric(cross.oracle_best_of_5_selected_rmsd_A)
    other_rank1 = numeric(cross.global_rank1_selected_rmsd_A)
    if not np.allclose(frame['rank1'],other_rank1,equal_nan=True):
        raise ValueError("Cascade and cross-selector rank1 results disagree")
    if 'best_of_5_cascade_rmsd_A' in cascade and not np.allclose(numeric(cascade.best_of_5_cascade_rmsd_A),frame.oracle_best5,equal_nan=True):
        raise ValueError("Cascade/cross oracle disagreement; check incomplete-branch policies")
    for method in ['multiscale_ncc','basin_q25_ncc']:
        if (frame[method].notna() & (frame.oracle_best5.isna() | (frame[method]+1e-7 < frame.oracle_best5))).any():
            raise ValueError(f"Oracle candidate-set inconsistency for {method}")
    summary,changes,transitions = summarize(frame,args.bootstrap_repeats,args.seed)
    frame.to_csv(args.output_dir/'per_parent_rmsd.csv')
    summary.to_csv(args.output_dir/'success_summary.csv',index=False)
    changes.to_csv(args.output_dir/'paired_comparisons.csv',index=False)
    transitions.to_csv(args.output_dir/'per_parent_transitions.csv',index=False)
    sources = {}
    for name in ['cohort','global_scores','source_reveal','cascade_per_parent','cross_per_parent','cross_summary','reference_manifest']:
        path = getattr(args,name)
        if path is None:
            continue
        sources[name] = dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    meta = dict(cohort_n=len(ids),sources=sources,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},cross_selector_metadata=cross_meta,missing_counts_as_failure=True,ci='parent percentile bootstrap',holm_family='all non-oracle contrasts across 2A and 5A',oracle='descriptive; no significance test',global_rmsd='explicit input column' if args.global_rmsd_column else 'symmetry corrected; center=False; minimize=False')
    evaluator_path = Path(__file__).with_name('evaluate_ligand_accuracy.py')
    meta['evaluation_code_sha256'] = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    meta['hydrogen_policy'] = 'exclude every atomic-number-1 node in memory; preserve remaining coordinates'
    (args.output_dir/'analysis_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    report = ['# Canonical Global versus cascade', '', f'N={len(ids)}; missing predictions count as failures. Selector status: {args.selector_status}.', '', '## Success rates', '', summary.to_string(index=False), '', '## Paired changes', '', changes.to_string(index=False), '', 'Oracle contrasts describe coverage, not deployable improvement. Median RMSD uses scored cases only. See analysis_manifest.json for source hashes and completeness policy.']
    (args.output_dir/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(summary.to_string(index=False))
    print(changes.to_string(index=False))
    print(f'GLOBAL_CASCADE_COMPARISON_OK: {args.output_dir}')
    return 0


if __name__ == '__main__':
    main()
