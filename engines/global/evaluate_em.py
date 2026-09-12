"""Evaluation utilities for DiffDock-EM on cryo-EM test set.

Subcommands:
  build-csv   Build inference CSV files for different evaluation conditions
  metrics     Compute symmetry-aware RMSD metrics and print comparison table

Usage:
  python evaluate_em.py build-csv --format diffdock
  python evaluate_em.py build-csv --format em --with-ring
  python evaluate_em.py build-csv --format em
  python evaluate_em.py metrics
  python evaluate_em.py metrics results_test/pretrained results_test/finetuned
"""

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---- Defaults ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TEST_SPLIT = PROJECT_ROOT / 'EM-dataset-codes' / 'data' / 'q1_2023' / 'splits' / 'em_test.txt'
DEFAULT_DATA_DIR = PROJECT_ROOT / 'EM-dataset-codes' / 'data' / 'q1_2023' / 'processed'
DEFAULT_RESULTS_DIR = SCRIPT_DIR / 'results_test'

# Known bad entries (no CA atoms — nucleic acid only structures)
SKIP_IDS = {'7YGC', '7YG9'}


def load_test_ids(test_split_path):
    """Load PDB IDs from test split file, skipping known bad entries."""
    with open(test_split_path) as f:
        ids = [line.strip() for line in f if line.strip()]
    valid = [pdb_id for pdb_id in ids if pdb_id not in SKIP_IDS]
    skipped = [pdb_id for pdb_id in ids if pdb_id in SKIP_IDS]
    if skipped:
        logger.info(f"Skipped {len(skipped)} entries (no CA atoms): {skipped}")
    return valid


def get_protein_sequence(pdb_path):
    """Extract colon-separated chain sequences from a PDB file via prody."""
    import prody
    prody.confProDy(verbosity='none')
    pdb = prody.parsePDB(pdb_path)
    if pdb.ca is None:
        return None
    seq = pdb.ca.getSequence()
    chain_ids = pdb.ca.getChids()
    seg_ids = pdb.ca.getSegnames()
    combined_ids = [s + c for s, c in zip(seg_ids, chain_ids)]

    unique_ids = []
    for cid in combined_ids:
        if cid not in unique_ids:
            unique_ids.append(cid)

    sequences = []
    for uid in unique_ids:
        mask = [s + c == uid for s, c in zip(seg_ids, chain_ids)]
        chain_seq = ''.join(seq[i] for i, m in enumerate(mask) if m)
        sequences.append(chain_seq)
    return ':'.join(sequences)


# ---------------------------------------------------------------------------
# build-csv subcommand
# ---------------------------------------------------------------------------

def cmd_build_csv(args):
    """Build inference CSV for a given evaluation format."""
    test_split = Path(args.test_split)
    data_dir = Path(args.data_dir)
    ligand_data_dir = Path(args.ligand_data_dir or args.data_dir)
    results_dir = Path(args.results_dir)

    pdb_ids = load_test_ids(test_split)
    logger.info(f"Loaded {len(pdb_ids)} test complexes from {test_split}")

    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine CSV path and columns
    if args.format == 'diffdock':
        suffix = 'diffdock'
        columns = ['complex_name', 'protein_path', 'ligand_description', 'protein_sequence']
    elif args.with_ring:
        suffix = 'em_ring'
        columns = ['complex_name', 'protein_path', 'ligand_description', 'ring_features_path']
    else:
        suffix = 'em'
        columns = ['complex_name', 'protein_path', 'ligand_description']

    csv_path = results_dir / f'test_input_{suffix}.csv'

    skipped = []
    rows = []
    for pdb_id in pdb_ids:
        pdb_dir = data_dir / pdb_id
        ligand_dir = ligand_data_dir / pdb_id
        protein_path = pdb_dir / f'{pdb_id}_protein_processed.pdb'
        ligand_path = ligand_dir / f'{pdb_id}_ligand.sdf'
        ring_path = pdb_dir / f'{pdb_id}_ring_features.npz'

        if not protein_path.exists():
            logger.warning(f"Missing protein: {protein_path}")
            skipped.append(pdb_id)
            continue
        if not ligand_path.exists():
            logger.warning(f"Missing ligand: {ligand_path}")
            skipped.append(pdb_id)
            continue

        row = [pdb_id, str(protein_path), str(ligand_path)]

        if args.format == 'diffdock':
            seq = get_protein_sequence(str(protein_path))
            if seq is None:
                logger.warning(f"No CA atoms in {pdb_id}, skipping")
                skipped.append(pdb_id)
                continue
            row.append(seq)
        elif args.with_ring:
            if ring_path.exists():
                row.append(str(ring_path))
            else:
                logger.warning(f"Missing ring features: {ring_path}, using empty")
                row.append('')

        rows.append(row)

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    if skipped:
        logger.warning(f"Skipped {len(skipped)} complexes: {skipped}")
    logger.info(f"Wrote {len(rows)} complexes to {csv_path}")
    print(str(csv_path))


# ---------------------------------------------------------------------------
# metrics subcommand
# ---------------------------------------------------------------------------

def find_pose_sdfs(complex_dir, max_rank=100):
    """Find pose SDF files in a complex directory.

    Handles both naming conventions:
      - rank{N}.sdf              (from inference_em.py)
      - rank{N}_confidence*.sdf  (from DiffDock/inference.py with confidence model)

    Returns list of (rank, path) sorted by rank.
    """
    poses = {}
    for f in os.listdir(complex_dir):
        if not f.endswith('.sdf'):
            continue
        # Match rank{N}.sdf or rank{N}_confidence{score}.sdf
        m = re.match(r'rank(\d+)(?:_confidence[-\d.]+)?\.sdf$', f)
        if m:
            rank = int(m.group(1))
            if rank <= max_rank and rank not in poses:
                poses[rank] = os.path.join(complex_dir, f)
    return sorted(poses.items())


def compute_rmsds(pdb_ids, data_dir, results_dir, ligand_suffix="ligand"):
    """Compute symmetry-aware RMSD for each complex and pose.

    Returns dict: {pdb_id: [rmsd_rank1, rmsd_rank2, ...]}

    ``ligand_suffix`` is the filename portion after ``<PDB_ID>_`` and before
    ``.sdf``.  The default reads the normal processed-dataset layout, while
    ``ligand_deposited`` reads non-destructive deposited-reference roots.
    """
    from rdkit import Chem
    from rdkit.Chem import RemoveAllHs
    from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule

    rmsds = {}

    for pdb_id in pdb_ids:
        gt_path = os.path.join(
            data_dir, pdb_id, f'{pdb_id}_{ligand_suffix}.sdf'
        )
        if not os.path.exists(gt_path):
            logger.warning(f"No ground truth for {pdb_id}")
            continue

        gt_mol = Chem.MolFromMolFile(gt_path, sanitize=True, removeHs=False)
        if gt_mol is None:
            logger.warning(f"Could not read ground truth mol: {gt_path}")
            continue
        gt_mol = RemoveAllHs(gt_mol)
        gt_coords = gt_mol.GetConformer().GetPositions()

        complex_dir = os.path.join(results_dir, pdb_id)
        if not os.path.isdir(complex_dir):
            logger.warning(f"No results directory for {pdb_id}")
            continue

        poses = find_pose_sdfs(complex_dir)
        if not poses:
            logger.warning(f"No pose SDFs for {pdb_id}")
            continue

        pose_rmsds = []
        for rank, pose_path in poses:
            pred_mol = Chem.MolFromMolFile(pose_path, sanitize=True, removeHs=False)
            if pred_mol is None:
                logger.warning(f"Could not read pose: {pose_path}")
                pose_rmsds.append(float('inf'))
                continue
            pred_mol = RemoveAllHs(pred_mol)
            pred_coords = pred_mol.GetConformer().GetPositions()

            if gt_mol.GetNumAtoms() != pred_mol.GetNumAtoms():
                logger.warning(f"Atom count mismatch for {pdb_id} rank{rank}: "
                               f"gt={gt_mol.GetNumAtoms()}, pred={pred_mol.GetNumAtoms()}")
                pose_rmsds.append(float('inf'))
                continue

            try:
                mol_spy = spyrmsd_molecule.Molecule.from_rdkit(gt_mol)
                val = spyrmsd_rmsd.symmrmsd(
                    gt_coords, pred_coords,
                    mol_spy.atomicnums, mol_spy.atomicnums,
                    mol_spy.adjacency_matrix, mol_spy.adjacency_matrix,
                )
                pose_rmsds.append(float(val))
            except Exception as e:
                logger.warning(f"RMSD computation failed for {pdb_id} rank{rank}: {e}")
                pose_rmsds.append(float('inf'))

        if pose_rmsds:
            rmsds[pdb_id] = pose_rmsds

    return rmsds


def report_metrics(rmsds, condition_name):
    """Compute and return metrics dict from RMSD results. Also prints summary."""
    if not rmsds:
        logger.warning(f"No RMSD data for {condition_name}")
        return {}

    min_rmsds = []
    for pdb_id, pose_rmsds in sorted(rmsds.items()):
        valid = [r for r in pose_rmsds if r < float('inf')]
        if valid:
            min_rmsds.append(min(valid))

    if not min_rmsds:
        logger.warning(f"No valid RMSDs for {condition_name}")
        return {}

    min_rmsds = np.array(min_rmsds)

    metrics = {
        'n_complexes': len(min_rmsds),
        'mean_min_rmsd': float(np.mean(min_rmsds)),
        'median_min_rmsd': float(np.median(min_rmsds)),
        'pct_lt_2A': float(np.mean(min_rmsds < 2.0) * 100),
        'pct_lt_3_5A': float(np.mean(min_rmsds < 3.5) * 100),
        'pct_lt_5A': float(np.mean(min_rmsds < 5.0) * 100),
    }

    print(f"\n{'='*60}")
    print(f"  {condition_name}")
    print(f"{'='*60}")
    print(f"  Complexes evaluated:     {metrics['n_complexes']}")
    print(f"  Mean min-RMSD (oracle):  {metrics['mean_min_rmsd']:.2f} A")
    print(f"  Median min-RMSD:         {metrics['median_min_rmsd']:.2f} A")
    print(f"  % < 2 A:                 {metrics['pct_lt_2A']:.1f}%")
    print(f"  % < 3.5 A:               {metrics['pct_lt_3_5A']:.1f}%")
    print(f"  % < 5 A:                 {metrics['pct_lt_5A']:.1f}%")
    print(f"{'='*60}")

    print(f"\n  Per-complex min-RMSD:")
    for pdb_id, pose_rmsds in sorted(rmsds.items()):
        valid = [r for r in pose_rmsds if r < float('inf')]
        if valid:
            best = min(valid)
            tag = "<2A" if best < 2.0 else ("<5A" if best < 5.0 else ">5A")
            print(f"    {pdb_id}: {best:6.2f} A  ({tag})")

    return metrics


def save_rmsds_csv(rmsds, csv_path):
    """Save per-complex RMSDs to CSV."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Determine max number of poses
        max_poses = max(len(v) for v in rmsds.values()) if rmsds else 0
        writer.writerow(['pdb_id', 'min_rmsd'] + [f'rank{i+1}_rmsd' for i in range(max_poses)])
        for pdb_id in sorted(rmsds.keys()):
            valid = [r for r in rmsds[pdb_id] if r < float('inf')]
            min_rmsd = min(valid) if valid else float('inf')
            writer.writerow([pdb_id, f'{min_rmsd:.4f}'] + [f'{r:.4f}' for r in rmsds[pdb_id]])
    logger.info(f"Saved RMSDs to {csv_path}")


def cmd_metrics(args):
    """Compute metrics for one or more result directories."""
    data_dir = str(args.data_dir)
    gt_data_dir = str(args.gt_data_dir or args.data_dir)
    results_base = Path(args.results_dir)

    # Load test IDs
    pdb_ids = load_test_ids(args.test_split)

    # Determine which conditions to evaluate
    if args.dirs:
        # Explicit directories given as positional args
        conditions = {}
        for d in args.dirs:
            p = Path(d)
            if p.is_dir():
                conditions[p.name] = str(p)
            else:
                logger.warning(f"Not a directory: {d}")
    else:
        # Auto-discover from results_test/{pretrained,finetuned,ablation}
        conditions = {}
        for name in ['pretrained', 'finetuned', 'ablation']:
            d = results_base / name
            if d.is_dir():
                conditions[name] = str(d)
        if not conditions:
            logger.error(f"No result directories found in {results_base}")
            sys.exit(1)

    logger.info(f"Evaluating conditions: {list(conditions.keys())}")
    logger.info(f"RMSD ground-truth root: {gt_data_dir}")
    logger.info(f"RMSD ground-truth ligand suffix: {args.gt_ligand_suffix}")

    all_metrics = {}
    for cond_name, cond_dir in conditions.items():
        logger.info(f"Computing RMSDs for: {cond_name}")
        rmsds = compute_rmsds(
            pdb_ids,
            gt_data_dir,
            cond_dir,
            ligand_suffix=args.gt_ligand_suffix,
        )
        metrics = report_metrics(rmsds, cond_name)
        all_metrics[cond_name] = metrics

        # Save per-complex CSV
        rmsd_csv = results_base / f'{cond_name}_rmsds.csv'
        save_rmsds_csv(rmsds, rmsd_csv)

    # Comparison table
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Condition':<20} {'N':>4} {'Mean':>8} {'Median':>8} {'<2A':>7} {'<3.5A':>7} {'<5A':>7}")
        print(f"  {'-'*20} {'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
        for cond_name in conditions:
            m = all_metrics.get(cond_name, {})
            if m:
                print(f"  {cond_name:<20} {m['n_complexes']:>4} "
                      f"{m['mean_min_rmsd']:>7.2f}A {m['median_min_rmsd']:>7.2f}A "
                      f"{m['pct_lt_2A']:>6.1f}% {m['pct_lt_3_5A']:>6.1f}% {m['pct_lt_5A']:>6.1f}%")
        print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DiffDock-EM evaluation utilities')
    parser.add_argument('--test_split', type=str, default=str(DEFAULT_TEST_SPLIT),
                        help='Path to em_test.txt')
    parser.add_argument('--data_dir', type=str, default=str(DEFAULT_DATA_DIR),
                        help='Directory with processed complexes')
    parser.add_argument(
        '--ligand_data_dir',
        type=str,
        default=None,
        help='Optional separate ligand-input root for build-csv; protein and '
             'ring features continue to come from --data_dir',
    )
    parser.add_argument(
        '--gt_data_dir',
        type=str,
        default=None,
        help='Optional separate RMSD-reference root; defaults to --data_dir',
    )
    parser.add_argument(
        '--gt_ligand_suffix',
        type=str,
        default='ligand',
        help='Ground-truth filename suffix after <PDB_ID>_ and before .sdf '
             '(default: ligand; deposited reference roots use ligand_deposited)',
    )
    parser.add_argument('--results_dir', type=str, default=str(DEFAULT_RESULTS_DIR),
                        help='Base directory for results (default: results_test/)')

    subparsers = parser.add_subparsers(dest='command', help='Subcommand')

    # build-csv
    p_csv = subparsers.add_parser('build-csv', help='Build inference CSV')
    p_csv.add_argument('--format', choices=['diffdock', 'em'], required=True,
                       help='diffdock: includes protein_sequence; em: for inference_em.py')
    p_csv.add_argument('--with-ring', action='store_true',
                       help='Include ring_features_path column (em format only)')
    p_csv.set_defaults(func=cmd_build_csv)

    # metrics
    p_met = subparsers.add_parser('metrics', help='Compute RMSD metrics')
    p_met.add_argument('dirs', nargs='*', help='Result directories (default: auto-discover)')
    p_met.set_defaults(func=cmd_metrics)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
