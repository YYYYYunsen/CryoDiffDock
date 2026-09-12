#!/usr/bin/env python3
"""Benchmark outputs from density_aware_confidence.py against a reference ligand.

This script is intentionally separate from deployment ranking: only this file
reads deposited coordinates or computes RMSD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_em import compute_rmsds, load_test_ids


DEFAULT_CONDITIONS = ('confidence_only', 'density_aware', 'density_aware_refine')


def summarize(rmsds, thresholds):
    rows = []
    for pdb_id, values in sorted(rmsds.items()):
        finite = [float(value) for value in values if np.isfinite(value)]
        if not finite:
            continue
        top1 = float(values[0]) if values and np.isfinite(values[0]) else float('inf')
        top5_values = [float(value) for value in values[:5] if np.isfinite(value)]
        rows.append({
            'pdb_id': pdb_id,
            'top1_rmsd_A': top1,
            'top5_best_rmsd_A': min(top5_values) if top5_values else float('inf'),
            'oracle_rmsd_A': min(finite),
            'num_valid_poses': len(finite),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {'n': 0}
    top1 = frame['top1_rmsd_A'].to_numpy(float)
    metrics = {
        'n': int(len(frame)),
        'mean_top1_rmsd_A': float(np.mean(top1)),
        'median_top1_rmsd_A': float(np.median(top1)),
        'top5_lt_2A_pct': float(np.mean(frame['top5_best_rmsd_A'] < 2.0) * 100),
        'oracle_lt_2A_pct': float(np.mean(frame['oracle_rmsd_A'] < 2.0) * 100),
        'threshold_pcts': {
            str(threshold): float(np.mean(top1 < threshold) * 100)
            for threshold in thresholds
        },
    }
    return frame, metrics


def print_table(results, thresholds):
    labels = list(results)
    width = max(28, max(map(len, labels), default=0))
    headers = [f'<{value:g}A' for value in thresholds]
    print('\n' + '=' * 105)
    print('  DENSITY-AWARE CONFIDENCE — FINAL COMPARISON')
    print('=' * 105)
    print(
        f'  {"Condition":<{width}} {"N":>4} {"Mean":>8} {"Median":>8} '
        + ' '.join(f'{header:>7}' for header in headers)
        + f' {"Top5<2A":>9} {"Oracle<2A":>11}'
    )
    print(
        f'  {"-" * width} {"-" * 4} {"-" * 8} {"-" * 8} '
        + ' '.join('-' * 7 for _ in headers)
        + f' {"-" * 9} {"-" * 11}'
    )
    for condition, metrics in results.items():
        if not metrics.get('n'):
            continue
        threshold_text = ' '.join(
            f'{metrics["threshold_pcts"][str(value)]:>6.1f}%'
            for value in thresholds
        )
        print(
            f'  {condition:<{width}} {metrics["n"]:>4d} '
            f'{metrics["mean_top1_rmsd_A"]:>7.2f}A '
            f'{metrics["median_top1_rmsd_A"]:>7.2f}A '
            f'{threshold_text} '
            f'{metrics["top5_lt_2A_pct"]:>8.1f}% '
            f'{metrics["oracle_lt_2A_pct"]:>10.1f}%'
        )
    print('=' * 105)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking_root', required=True,
                        help='Output directory from density_aware_confidence.py.')
    parser.add_argument('--test_split', required=True)
    parser.add_argument('--gt_data_dir', required=True)
    parser.add_argument('--gt_ligand_suffix', default='ligand_deposited')
    parser.add_argument('--conditions', nargs='*', default=None,
                        help='Condition subdirectories; defaults to all available standard ones.')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[1.0, 2.0, 3.5, 5.0])
    parser.add_argument('--out_dir', default=None)
    args = parser.parse_args()

    root = Path(args.ranking_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / 'evaluation'
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions or [
        name for name in DEFAULT_CONDITIONS if (root / name).is_dir()
    ]
    if not conditions:
        parser.error(f'no condition directories found below {root}')

    pdb_ids = load_test_ids(args.test_split)
    all_metrics = {}
    comparison_rows = []
    for condition in conditions:
        condition_dir = root / condition
        if not condition_dir.is_dir():
            print(f'[WARNING] condition does not exist: {condition_dir}')
            continue
        print(f'[INFO] Evaluating {condition}: {condition_dir}')
        rmsds = compute_rmsds(
            pdb_ids,
            args.gt_data_dir,
            str(condition_dir),
            ligand_suffix=args.gt_ligand_suffix,
        )
        per_complex, metrics = summarize(rmsds, args.thresholds)
        all_metrics[condition] = metrics
        per_complex.insert(0, 'condition', condition)
        comparison_rows.append(per_complex)
        per_complex.to_csv(out_dir / f'{condition}_rmsds.csv', index=False)

    print_table(all_metrics, args.thresholds)
    if comparison_rows:
        pd.concat(comparison_rows, ignore_index=True).to_csv(
            out_dir / 'per_complex_rmsds.csv', index=False
        )
    summary_path = out_dir / 'comparison_summary.json'
    summary_path.write_text(json.dumps(all_metrics, indent=2) + '\n')
    print(f'[INFO] Evaluation outputs: {out_dir}')


if __name__ == '__main__':
    main()
