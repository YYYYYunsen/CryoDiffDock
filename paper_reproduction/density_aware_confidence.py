#!/usr/bin/env python3
"""Unified DiffDock confidence + cryo-EM density-aware pose ranking.

This is the deployment/inference entry point.  It deliberately does not read a
reference ligand and never computes RMSD.  Benchmark evaluation belongs in
``evaluate_density_aware_ranking.py``.

Pipeline
--------
1. Score raw score-model poses with the original DiffDock confidence model, or
   reuse an existing confidence-scored directory.
2. Compute pose-conditioned map-density mean and protein-subtracted residual
   shape correlation.
3. Rank with fixed, predeclared weights.
4. Optionally append density-refined poses that pass inference-time movement
   gates. Refined poses inherit their parent's confidence score.
5. Write independent ``confidence_only``, ``density_aware`` and
   ``density_aware_refine`` result directories plus a complete scores CSV.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from rerank_by_density import (
    append_gated_refinement_candidates,
    compute_density_features,
    load_map_mapping,
    parse_rerank_dir,
    read_pose_heavy_coords,
    read_protein_heavy_coords,
)
from utils.maps import open_map


MODES = ('confidence', 'density_aware', 'density_aware_refine')


def _has_confidence_files(root: Path) -> bool:
    for complex_dir in root.iterdir() if root.is_dir() else ():
        if complex_dir.is_dir() and next(complex_dir.glob('rank*_confidence*.sdf'), None):
            return True
    return False


def _write_generated_split(protein_ligand_csv: Path, output: Path) -> Path:
    frame = pd.read_csv(protein_ligand_csv)
    if 'complex_name' not in frame.columns:
        raise ValueError(f'{protein_ligand_csv} has no complex_name column')
    ids = [str(value).strip() for value in frame['complex_name'] if str(value).strip()]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(ids) + '\n')
    return output


def run_original_confidence(args) -> Path:
    """Return a directory containing confidence-ranked pose files."""
    if args.confidence_scored_dir:
        scored = Path(args.confidence_scored_dir).resolve()
        if not scored.is_dir():
            raise FileNotFoundError(f'confidence-scored directory not found: {scored}')
        return scored

    poses = Path(args.poses_dir).resolve()
    if _has_confidence_files(poses):
        print(f'[INFO] Input already contains confidence scores: {poses}')
        return poses

    required = {
        '--protein_ligand_csv': args.protein_ligand_csv,
        '--orig_conf_dir': args.orig_conf_dir,
        '--esm_embeddings_path': args.esm_embeddings_path,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            'Raw poses require the original confidence stage; missing ' + ', '.join(missing)
        )

    protein_csv = Path(args.protein_ligand_csv).resolve()
    if not protein_csv.is_file():
        raise FileNotFoundError(f'protein/ligand CSV not found: {protein_csv}')
    test_split = (
        Path(args.test_split).resolve()
        if args.test_split
        else _write_generated_split(
            protein_csv, Path(args.output_dir) / '_confidence_stage' / 'input_ids.txt'
        )
    )
    stage_root = Path(args.output_dir).resolve() / '_confidence_stage'
    stage_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().with_name('compare_confidence_rerank.py')
    command = [
        sys.executable,
        str(script),
        '--pose_set', 'finetuned',
        '--conf_model', 'orig',
        '--finetuned_poses', str(poses),
        '--pretrained_poses', str(poses),
        '--ablation_poses', str(poses),
        '--orig_conf_dir', str(Path(args.orig_conf_dir).resolve()),
        '--orig_conf_ckpt', args.orig_conf_ckpt,
        '--em_conf_dir', str(Path(args.orig_conf_dir).resolve()),
        '--protein_ligand_csv', str(protein_csv),
        '--test_split', str(test_split),
        '--data_dir', str(Path(args.data_dir).resolve()),
        '--esm_embeddings_path', str(Path(args.esm_embeddings_path).resolve()),
        '--batch_size', str(args.confidence_batch_size),
        '--results_dir', str(stage_root),
    ]
    if args.top_k is not None:
        command.extend(['--max_poses', str(args.top_k)])
    print('[INFO] Running original DiffDock confidence model:')
    print('  ' + ' '.join(command))
    subprocess.run(command, check=True)
    scored = stage_root / 'finetuned_orig_conf'
    if not scored.is_dir():
        raise RuntimeError(f'confidence stage completed but output is missing: {scored}')
    return scored


def _zscore(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    result = np.zeros(array.shape, dtype=np.float64)
    if not finite.any():
        return result
    mean = float(np.mean(array[finite]))
    std = float(np.std(array[finite]))
    if std < 1e-8:
        return result
    result[finite] = (array[finite] - mean) / std
    return result


def _candidate_is_refined(name, rank, refined_keys) -> bool:
    return (str(name).upper(), int(rank)) in refined_keys


def compute_rankings(args, confidence_dir: Path):
    entries = parse_rerank_dir(confidence_dir, top_k=args.top_k)
    if args.test_split:
        requested = {
            line.strip().upper()
            for line in Path(args.test_split).read_text().splitlines()
            if line.strip()
        }
        entries = {name: poses for name, poses in entries.items() if name.upper() in requested}
        missing = sorted(requested - {name.upper() for name in entries})
        print(
            f'[INFO] Test filter: {len(entries)}/{len(requested)} complexes have confidence poses'
        )
        if missing:
            print(f'[WARNING] Missing predictions ({len(missing)}): {" ".join(missing[:20])}')
    if not entries:
        raise ValueError(f'no ranked poses found below {confidence_dir}')

    gate_rows = []
    if args.mode == 'density_aware_refine':
        required = {
            '--refinement_metrics_csv': args.refinement_metrics_csv,
            '--refine_csv': args.refine_csv,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError('refinement mode is missing ' + ', '.join(missing))
        entries, gate_rows = append_gated_refinement_candidates(
            per_complex_entries=entries,
            metrics_csv=args.refinement_metrics_csv,
            refine_csv=args.refine_csv,
            parent_scores_csv=args.refinement_parent_scores_csv,
            parent_score_column=args.refinement_parent_score_column,
            centroid_max=args.refinement_centroid_max,
            pose_change_max=args.refinement_pose_change_max,
            audit_csv=args.refinement_gate_audit_csv,
        )

    refined_keys = {
        (str(row['pdb_id']).upper(), int(row['candidate_rank']))
        for row in gate_rows
        if row.get('accepted') and row.get('candidate_rank') not in ('', None)
    }
    map_mapping = load_map_mapping(args.map_csv) if args.mode != 'confidence' else {}
    all_rankings = {}
    score_rows = []
    missing_maps = 0

    for name, candidates in tqdm(entries.items(), desc='Density-aware ranking'):
        records = []
        map_path = map_mapping.get(name.upper())
        use_density = (
            args.mode != 'confidence'
            and map_path is not None
            and Path(map_path).is_file()
        )
        grid = None
        protein_coords = np.zeros((0, 3), dtype=np.float32)
        if use_density:
            try:
                _map, grid = open_map(map_path)
                protein_path = (
                    Path(args.data_dir) / name / f'{name}_protein_processed.pdb'
                )
                protein_coords = read_protein_heavy_coords(str(protein_path))
            except Exception as exc:
                print(f'[WARNING] {name}: density unavailable ({exc}); confidence fallback')
                use_density = False
        elif args.mode != 'confidence':
            missing_maps += 1

        for rank, confidence, pose_path in candidates:
            mean_density = residual_shape = float('nan')
            if use_density:
                coords = read_pose_heavy_coords(pose_path)
                if coords is not None and len(coords):
                    features = compute_density_features(
                        grid,
                        coords,
                        trim_pct=0.0,
                        protein_coords=protein_coords,
                        diff_sigma=args.diff_sigma,
                        diff_cutoff=args.diff_cutoff,
                        shape_sigma=args.shape_sigma,
                        shape_shell_radii=tuple(args.shape_shell_radii),
                        residual_shell_radii=tuple(args.residual_shell_radii),
                        residual_fit_exclusion=args.residual_fit_exclusion,
                        residual_mask_fraction=args.residual_mask_fraction,
                        enable_rscc=False,
                        rscc_resolution=3.5,
                    )
                    mean_density = float(features['mean'])
                    residual_shape = float(features['residual_shape_corr'])
            records.append({
                'pdb_id': name,
                'candidate_rank': int(rank),
                'confidence_raw': float(confidence),
                'density_mean_raw': mean_density,
                'residual_shape_corr_raw': residual_shape,
                'is_refined': _candidate_is_refined(name, rank, refined_keys),
                'sdf_path': str(pose_path),
                'density_available': bool(use_density),
            })

        original = [record for record in records if not record['is_refined']]
        confidence_order = sorted(original, key=lambda record: record['candidate_rank'])

        def score_pool(pool):
            conf_z = _zscore([record['confidence_raw'] for record in pool])
            mean_z = _zscore([record['density_mean_raw'] for record in pool])
            residual_z = _zscore([record['residual_shape_corr_raw'] for record in pool])
            scored = []
            for index, record in enumerate(pool):
                item = dict(record)
                item['confidence_z'] = float(conf_z[index])
                item['density_mean_z'] = float(mean_z[index])
                item['residual_shape_corr_z'] = float(residual_z[index])
                item['combined_score'] = float(
                    conf_z[index]
                    + args.lambda_mean * mean_z[index]
                    + args.lambda_residual_shape_corr * residual_z[index]
                )
                scored.append(item)
            return sorted(scored, key=lambda record: record['combined_score'], reverse=True)

        density_original = score_pool(original)
        density_refined = score_pool(records)
        all_rankings[name] = {
            'confidence_only': confidence_order,
            'density_aware': density_original,
            'density_aware_refine': density_refined,
        }
        for condition, ranking in all_rankings[name].items():
            if condition == 'density_aware_refine' and args.mode != 'density_aware_refine':
                continue
            if condition == 'density_aware' and args.mode == 'confidence':
                continue
            for final_rank, record in enumerate(ranking, start=1):
                row = dict(record)
                row['condition'] = condition
                row['final_rank'] = final_rank
                if condition == 'confidence_only':
                    row.setdefault('confidence_z', float('nan'))
                    row.setdefault('density_mean_z', float('nan'))
                    row.setdefault('residual_shape_corr_z', float('nan'))
                    row['combined_score'] = row['confidence_raw']
                score_rows.append(row)

    if args.mode != 'confidence':
        print(f'[INFO] Confidence-only fallback due to missing/unreadable maps: {missing_maps}')
    return all_rankings, pd.DataFrame(score_rows), gate_rows


def materialize_rankings(args, rankings, scores: pd.DataFrame):
    output_root = Path(args.output_dir).resolve()
    conditions = ['confidence_only']
    if args.mode in ('density_aware', 'density_aware_refine'):
        conditions.append('density_aware')
    if args.mode == 'density_aware_refine':
        conditions.append('density_aware_refine')

    for condition in conditions:
        condition_root = output_root / condition
        for name, condition_map in rankings.items():
            destination = condition_root / name
            destination.mkdir(parents=True, exist_ok=True)
            for old in destination.glob('rank*.sdf'):
                old.unlink()
            ranking = condition_map[condition]
            for final_rank, record in enumerate(ranking[:args.num_output], start=1):
                source = Path(record['sdf_path']).resolve()
                target = destination / f'rank{final_rank}.sdf'
                if args.symlink:
                    target.symlink_to(source)
                else:
                    shutil.copy2(source, target)

    output_root.mkdir(parents=True, exist_ok=True)
    scores_path = output_root / 'density_aware_scores.csv'
    scores.to_csv(scores_path, index=False)
    final_condition = conditions[-1]
    (output_root / 'final_condition.txt').write_text(final_condition + '\n')
    config = {
        'mode': args.mode,
        'poses_dir': str(Path(args.poses_dir).resolve()),
        'confidence_scored_dir': args.confidence_scored_dir,
        'map_csv': args.map_csv,
        'data_dir': str(Path(args.data_dir).resolve()),
        'top_k': args.top_k,
        'num_output': args.num_output,
        'lambda_mean': args.lambda_mean,
        'lambda_residual_shape_corr': args.lambda_residual_shape_corr,
        'refinement_centroid_max': args.refinement_centroid_max,
        'refinement_pose_change_max': args.refinement_pose_change_max,
        'final_condition': final_condition,
        'conditions_written': conditions,
    }
    (output_root / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    summary = {
        condition: {
            'complexes': len(rankings),
            'poses_written': int(min(args.num_output, args.top_k)),
        }
        for condition in conditions
    }
    (output_root / 'ranking_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'[INFO] Final condition: {final_condition}')
    print(f'[INFO] Ranked poses: {output_root / final_condition}')
    print(f'[INFO] Scores: {scores_path}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses_dir', required=True,
                        help='Raw score-model poses, or already confidence-scored poses.')
    parser.add_argument('--data_dir', required=True,
                        help='Processed receptor directory containing protein_processed.pdb files.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mode', choices=MODES, default='density_aware')
    parser.add_argument('--map_csv', default=None,
                        help='Required for density-aware modes; contains pdb_id,map_path,status.')
    parser.add_argument('--test_split', default=None,
                        help='Optional cohort filter. Omit for a user-submitted batch.')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of original confidence-ranked candidates to consider.')
    parser.add_argument('--num_output', type=int, default=10)
    parser.add_argument('--lambda_mean', type=float, default=0.5)
    parser.add_argument('--lambda_residual_shape_corr', type=float, default=1.0)
    parser.add_argument('--confidence_scored_dir', default=None,
                        help='Reuse an existing original-DiffDock-confidence output directory.')
    parser.add_argument('--protein_ligand_csv', default=None,
                        help='Inference CSV required when --poses_dir has no confidence scores.')
    parser.add_argument('--orig_conf_dir', default=None)
    parser.add_argument('--orig_conf_ckpt', default='best_model.pt')
    parser.add_argument('--esm_embeddings_path', default=None)
    parser.add_argument('--confidence_batch_size', type=int, default=16)
    parser.add_argument('--refinement_metrics_csv', default=None)
    parser.add_argument('--refine_csv', default=None)
    parser.add_argument('--refinement_parent_scores_csv', default=None)
    parser.add_argument('--refinement_parent_score_column',
                        default='combined_at_best_weights')
    parser.add_argument('--refinement_centroid_max', type=float, default=2.0)
    parser.add_argument('--refinement_pose_change_max', type=float, default=2.5)
    parser.add_argument('--refinement_gate_audit_csv', default=None)
    parser.add_argument('--diff_sigma', type=float, default=1.2)
    parser.add_argument('--diff_cutoff', type=float, default=4.0)
    parser.add_argument('--shape_sigma', type=float, default=1.0)
    parser.add_argument('--shape_shell_radii', type=float, nargs='+', default=[0.0, 0.8, 1.6])
    parser.add_argument('--residual_shell_radii', type=float, nargs='+',
                        default=[0.0, 0.8, 1.6, 2.4, 3.2])
    parser.add_argument('--residual_fit_exclusion', type=float, default=2.0)
    parser.add_argument('--residual_mask_fraction', type=float, default=0.05)
    parser.add_argument('--symlink', action='store_true',
                        help='Symlink ranked poses instead of copying them.')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.top_k <= 0 or args.num_output <= 0:
        parser.error('--top_k and --num_output must be positive')
    if args.num_output > args.top_k and args.mode != 'density_aware_refine':
        parser.error('--num_output cannot exceed --top_k without refinement candidates')
    if args.mode != 'confidence' and not args.map_csv:
        parser.error('--map_csv is required for density-aware modes')
    if args.mode == 'density_aware_refine' and not args.refinement_gate_audit_csv:
        args.refinement_gate_audit_csv = str(
            Path(args.output_dir).resolve() / 'refinement_gate_audit.csv'
        )

    confidence_dir = run_original_confidence(args)
    print(f'[INFO] Confidence candidates: {confidence_dir}')
    rankings, scores, _gate_rows = compute_rankings(args, confidence_dir)
    materialize_rankings(args, rankings, scores)


if __name__ == '__main__':
    main()
