"""Second-stage reranking of top-K confidence poses using per-pose EM density
and optional pose-consensus.

Motivation:
    Confidence models on this dataset cap out at ~37% top-1 hit rate, but
    top-5 hits ~51%. The gap means the model correctly identifies the "good
    basin" (top 5 contains a <2Å pose for 51% of complexes) but fails to
    pick the best pose *within* that basin. Simple receptor-side global ring
    summaries do not discriminate within top-K, but pose-conditioned local
    ring summaries can: evaluate the receptor ring environment *around the
    current pose* and use that as a complementary signal to ligand density.

    Per-pose density probing gives a pose-dependent signal: for each pose,
    evaluate the EM density directly at its ligand heavy-atom positions.
    Good poses → atoms in high-density regions → high density score.

Algorithm:
    1. Parse confidence-reranked SDFs under ``--rerank_dir``; keep top K per complex.
    2. For each pose, trilinear-interpolate the EM map at every heavy-atom coord.
    3. density_score = mean (or trimmed_mean) of per-atom densities.
    4. (Optional) consensus_score = 1 / (1 + mean pairwise RMSD to other top-K poses).
    5. (Optional) smina_score = -Affinity from `smina --score_only` on each pose.
    6. new_score = confidence + lambda_density * density_z
                   + lambda_consensus * consensus_z
                   + lambda_smina * smina_z
    7. Re-rank the top-K by new_score; report top-1 hit rate (<2Å).

Usage (on server):
    python rerank_by_density.py \\
        --rerank_dir  results_test_confidence_rerank/finetuned_em_conf \\
        --map_csv     /Scr/yunsenz2/data/EM-dataset-all/04_maps.csv \\
        --data_dir    /Scr/yunsenz2/data/EM-dataset-all/data/processed_6radii \\
        --top_k       5 \\
        --lambda_grid 0.0,0.5,1.0,2.0,5.0 \\
        --consensus_lambda_grid 0.0,0.5,1.0 \\
        --smina_lambda_grid 0.0,0.5,1.0

The --map_csv must have columns: pdb_id, map_path, status  (only status=='ok' rows used).
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import gemmi
from rdkit import Chem
from rdkit.Chem import AllChem, RemoveAllHs, rdMolDescriptors
from tqdm import tqdm
try:
    from utils.ligand_density_features import compute_rscc
except ImportError:
    _RSCC_IMPORT_WARNING_SHOWN = False

    def compute_rscc(*args, **kwargs):
        """Graceful fallback when the RSCC helper is unavailable."""
        global _RSCC_IMPORT_WARNING_SHOWN
        if not _RSCC_IMPORT_WARNING_SHOWN:
            print(
                'Warning: utils.ligand_density_features.compute_rscc is unavailable; '
                'disabling RSCC feature for this run.'
            )
            _RSCC_IMPORT_WARNING_SHOWN = True
        return float('nan')

# Reuse the EM-dataset-codes helpers (maps.py lives alongside this repo).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'EM-dataset-codes'))
try:
    from utils.maps import open_map, interp_points, compute_ncc
except ImportError as e:
    raise ImportError(
        'Could not import utils.maps from EM-dataset-codes. '
        'Make sure EM-dataset-codes is a sibling of this DiffDock-EM directory.'
    ) from e

from evaluate_em import load_test_ids


RANK_RE = re.compile(r'rank(\d+)(?:_confidence([-\d.]+))?\.sdf$')


def load_reviewed_chemistry_manifest(path):
    """Load per-PDB reviewed rcsb_class labels for rerank stratification."""
    df = pd.read_csv(path)
    required = {'pdb_id', 'rcsb_class'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f'chemistry manifest is missing columns: {sorted(missing)}'
        )
    allowed = {'cofactor', 'drug_like', 'exclude', 'review'}
    result = {}
    for _, row in df.iterrows():
        pdb_id = str(row['pdb_id']).strip().upper()
        label = str(row['rcsb_class']).strip().lower()
        if not pdb_id or pdb_id == 'NAN':
            continue
        if label not in allowed:
            raise ValueError(
                f'Unsupported rcsb_class={label!r} for {pdb_id}; '
                f'expected one of {sorted(allowed)}'
            )
        result[pdb_id] = label
    return result


# ─── Ligand chemistry classifier (copied verbatim from scan_dataset_chemistry.py)
# so per-class breakdown here matches numbers elsewhere in the codebase. ────────

_LONG_ALIPHATIC = Chem.MolFromSmarts('[C;R0]~[C;R0]~[C;R0]~[C;R0]~[C;R0]~[C;R0]~[C;R0]~[C;R0]')
_AMINO_ACID_BACKBONE = Chem.MolFromSmarts('[NX3,NX4][CX4][CX3](=O)[OX2,OX1-]')


def classify_mol(mol):
    """Return 'amino_acid' | 'phospholipid' | 'cofactor' | 'drug_like' | 'unknown'."""
    if mol is None:
        return 'unknown'
    n_heavy = mol.GetNumHeavyAtoms()
    formula = rdMolDescriptors.CalcMolFormula(mol)

    def count(elem):
        m = re.search(rf'{elem}(\d+)', formula)
        return int(m.group(1)) if m else (1 if elem in formula else 0)

    C = count('C'); N = count('N'); O = count('O'); P = count('P')

    if n_heavy <= 16 and mol.HasSubstructMatch(_AMINO_ACID_BACKBONE):
        return 'amino_acid'
    if n_heavy <= 14 and N >= 1 and O >= 2 and C <= 10 and P == 0:
        return 'amino_acid'
    if mol.HasSubstructMatch(_LONG_ALIPHATIC):
        return 'phospholipid'
    if P >= 1 and C >= 14 and O >= 5 and N <= 1:
        return 'phospholipid'
    if P >= 2 and N >= 4:
        return 'cofactor'
    return 'drug_like'


def classify_complex(data_dir, pdb_id, ligand_suffix='ligand'):
    """Classify a complex by its crystal ligand SDF.

    Tries sanitized parse first; falls back to non-sanitised parse (for
    modified residues / unusual valence) and manually runs ring perception
    so classify_mol's HasSubstructMatch doesn't hit "RingInfo not initialized".
    """
    gt_path = os.path.join(
        data_dir, pdb_id, f'{pdb_id}_{ligand_suffix}.sdf'
    )
    if not os.path.exists(gt_path):
        return 'unknown'
    mol = Chem.MolFromMolFile(gt_path, sanitize=True, removeHs=False)
    if mol is None:
        # Non-sanitised fallback: skips valence/aromaticity but leaves RingInfo
        # uninitialised. Call FastFindRings so substructure matching works.
        mol = Chem.MolFromMolFile(gt_path, sanitize=False, removeHs=False)
        if mol is not None:
            try:
                Chem.FastFindRings(mol)
            except Exception:
                pass
    if mol is None:
        return 'unknown'
    try:
        mol = RemoveAllHs(mol, sanitize=False)
    except Exception:
        pass
    # Ensure rings are perceived on the final mol (RemoveAllHs can reset info).
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    return classify_mol(mol)


def load_map_mapping(csv_path):
    """Load pdb_id → map_path from the 04_maps.csv schema."""
    df = pd.read_csv(csv_path)
    if 'status' in df.columns:
        df = df[df['status'] == 'ok'].copy()
    mapping = {}
    for _, row in df.iterrows():
        pdb = str(row['pdb_id']).strip().upper()
        mpath = str(row['map_path']).strip()
        if pdb and mpath and mpath != 'nan':
            mapping[pdb] = mpath
    print(f'Loaded {len(mapping)} PDB → map_path entries from {csv_path}')
    return mapping


def parse_rerank_dir(rerank_dir, top_k=5, ignore_confidence=False):
    """Return dict: complex_name → list of (rank, score, sdf_path), length ≤ top_k.

    Accepts BOTH confidence-scored files (``rank{N}_confidence{score}.sdf``) and
    plain pose files (``rank{N}.sdf``). For a given rank the confidence-scored
    file is preferred; a plain file is used only when no scored twin exists, with
    a fallback score of 0.0 so that at lambda=0 the original rank order is kept
    (combined == confidence_z == constant → argsort is stable → rank order).
    """
    rerank_dir = Path(rerank_dir)
    out = {}
    for complex_dir in sorted(rerank_dir.iterdir()):
        if not complex_dir.is_dir():
            continue
        by_rank = {}   # rank -> (score, path, has_score)
        for f in complex_dir.iterdir():
            m = RANK_RE.match(f.name)
            if m is None:
                continue
            rank = int(m.group(1))
            score_s = m.group(2)
            has_score = score_s is not None
            score = (
                0.0 if ignore_confidence
                else (float(score_s) if has_score else 0.0)
            )
            prev = by_rank.get(rank)
            # Keep the confidence-scored file in preference to a plain twin.
            if prev is None or (has_score and not prev[2]):
                by_rank[rank] = (score, f, has_score)
        if not by_rank:
            continue
        entries = [(rank, sc, path) for rank, (sc, path, _) in by_rank.items()]
        entries.sort(key=lambda x: x[0])    # by rank
        out[complex_dir.name] = entries[:top_k]
    return out


def read_pose_heavy_coords(sdf_path):
    """Read heavy-atom coords from an SDF. Returns (N_heavy, 3) float32 or None."""
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
    if mol is None:
        return None
    mol = RemoveAllHs(mol, sanitize=False)
    conf = mol.GetConformer()
    return np.asarray(conf.GetPositions(), dtype=np.float32)


def direct_pose_change_rmsd(before_path, after_path):
    """Direct heavy-atom coordinate RMSD between a pose and its refinement.

    This intentionally performs no alignment and no symmetry remapping.  The
    refinement preserves atom order, so the value measures the actual movement
    applied by the optimizer and is available at inference time without ground
    truth coordinates.
    """
    before = read_pose_heavy_coords(before_path)
    after = read_pose_heavy_coords(after_path)
    if before is None or after is None:
        raise ValueError('could not read heavy-atom coordinates')
    if before.shape != after.shape:
        raise ValueError(
            f'heavy-atom shape mismatch: before={before.shape}, after={after.shape}'
        )
    return float(np.sqrt(np.mean(np.sum((after - before) ** 2, axis=1))))


def apply_refinement_gates(
    per_complex_entries,
    rerank_dir,
    metrics_csv,
    refine_csv=None,
    centroid_max=None,
    pose_change_max=None,
    audit_csv=None,
):
    """Remove unsafe *refined* candidates while retaining every original pose.

    ``density_gradient_refine.py --merge_original_dir`` writes, for each PDB,
    all original poses first and then one refined pose per successful metrics
    row sorted by ``orig_rank``.  This function reconstructs that mapping.  It
    never removes an original pose, and it never uses deposited coordinates or
    RMSD-to-ground-truth in a gate.

    Parameters are optional so legacy reranking behavior remains unchanged.
    ``pose_change_max`` requires ``refine_csv`` because that CSV records the
    original pose path corresponding to each refined ``complex_name``.
    """
    if centroid_max is None and pose_change_max is None:
        raise ValueError(
            'refinement gating requested but neither centroid_max nor '
            'pose_change_max was supplied'
        )
    if centroid_max is not None and centroid_max < 0:
        raise ValueError('centroid_max must be non-negative')
    if pose_change_max is not None and pose_change_max < 0:
        raise ValueError('pose_change_max must be non-negative')
    if pose_change_max is not None and not refine_csv:
        raise ValueError('pose_change_max requires --refinement_gate_refine_csv')

    metrics_path = Path(metrics_csv)
    if not metrics_path.is_file():
        raise FileNotFoundError(f'refinement metrics CSV not found: {metrics_path}')
    metrics = pd.read_csv(metrics_path)
    required = {'complex_name', 'pdb_id', 'orig_rank', 'snap_path'}
    missing = required - set(metrics.columns)
    if centroid_max is not None:
        missing |= {'centroid_shift_A'} - set(metrics.columns)
    if missing:
        raise ValueError(
            f'refinement metrics CSV is missing columns: {sorted(missing)}'
        )

    original_path_by_name = {}
    if refine_csv:
        refine_path = Path(refine_csv)
        if not refine_path.is_file():
            raise FileNotFoundError(f'refine CSV not found: {refine_path}')
        refine_df = pd.read_csv(refine_path)
        refine_required = {'complex_name', 'ligand_description'}
        refine_missing = refine_required - set(refine_df.columns)
        if refine_missing:
            raise ValueError(
                f'refine CSV is missing columns: {sorted(refine_missing)}'
            )
        original_path_by_name = {
            str(row['complex_name']): str(row['ligand_description'])
            for _, row in refine_df.iterrows()
        }

    metrics = metrics.copy()
    metrics['pdb_key'] = metrics['pdb_id'].astype(str).str.upper()
    metrics['orig_rank'] = pd.to_numeric(metrics['orig_rank'], errors='raise').astype(int)
    metrics = metrics.sort_values(['pdb_key', 'orig_rank', 'complex_name'])
    rows_by_pdb = {
        pdb_key: group.to_dict(orient='records')
        for pdb_key, group in metrics.groupby('pdb_key', sort=False)
    }

    filtered = {}
    audit_rows = []
    total_refined = accepted_refined = rejected_refined = 0
    missing_from_topk = 0

    for name, entries in per_complex_entries.items():
        pdb_key = str(name).upper()
        metric_rows = rows_by_pdb.get(pdb_key, [])
        if not metric_rows:
            filtered[name] = entries
            continue

        # Count the complete union on disk rather than the already top-K-limited
        # entries.  This makes a too-small --top_k detectable instead of silently
        # assigning a refined row to the wrong rank.
        union_dir = Path(rerank_dir) / name
        full_by_rank = {}
        if union_dir.is_dir():
            for pose_path in union_dir.iterdir():
                match = RANK_RE.match(pose_path.name)
                if match:
                    full_by_rank[int(match.group(1))] = pose_path
        n_union = len(full_by_rank)
        n_refined = len(metric_rows)
        n_original = n_union - n_refined
        if n_original < 0:
            raise ValueError(
                f'{name}: metrics has {n_refined} refined rows but union contains '
                f'only {n_union} rank files'
            )

        entry_by_rank = {rank: (rank, score, path) for rank, score, path in entries}
        keep_ranks = set(range(1, n_original + 1))

        for offset, metric_row in enumerate(metric_rows, start=1):
            total_refined += 1
            union_rank = n_original + offset
            complex_name = str(metric_row['complex_name'])
            snap_path = str(metric_row['snap_path'])
            centroid_shift = float(metric_row['centroid_shift_A']) \
                if centroid_max is not None else float('nan')
            before_path = original_path_by_name.get(complex_name, '')
            pose_change = float('nan')
            errors = []

            if centroid_max is not None:
                if not np.isfinite(centroid_shift):
                    errors.append('centroid_shift_unavailable')
                elif centroid_shift > centroid_max:
                    errors.append('centroid_shift_exceeds_limit')

            if pose_change_max is not None:
                if not before_path:
                    errors.append('original_pose_path_unavailable')
                else:
                    try:
                        pose_change = direct_pose_change_rmsd(before_path, snap_path)
                    except Exception as exc:
                        errors.append(f'pose_change_unavailable:{exc}')
                    else:
                        if pose_change > pose_change_max:
                            errors.append('pose_change_exceeds_limit')

            present_in_considered_topk = union_rank in entry_by_rank
            if not present_in_considered_topk:
                missing_from_topk += 1
                errors.append('not_considered_due_to_top_k')

            accepted = not errors
            if accepted:
                keep_ranks.add(union_rank)
                accepted_refined += 1
            else:
                rejected_refined += 1

            audit_rows.append({
                'pdb_id': name,
                'complex_name': complex_name,
                'source_orig_rank': int(metric_row['orig_rank']),
                'union_rank': union_rank,
                'is_refined': True,
                'accepted': accepted,
                'rejection_reason': ';'.join(errors),
                'centroid_shift_A': centroid_shift,
                'pose_change_rmsd_A': pose_change,
                'centroid_max_A': centroid_max,
                'pose_change_max_A': pose_change_max,
                'original_pose_path': before_path,
                'refined_snap_path': snap_path,
                'union_pose_path': str(full_by_rank.get(union_rank, '')),
            })

        # Original ranks always survive. Refined ranks survive only if accepted.
        filtered[name] = [entry for entry in entries if entry[0] in keep_ranks]

    if audit_csv:
        audit_path = Path(audit_csv)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f'Refinement-gate audit CSV: {audit_path}')

    print('\n--- Refinement safety gate ---')
    print(f'  refined candidates audited: {total_refined}')
    print(f'  accepted:                  {accepted_refined}')
    print(f'  rejected:                  {rejected_refined}')
    if centroid_max is not None:
        print(f'  centroid-shift limit:      {centroid_max:g} A')
    if pose_change_max is not None:
        print(f'  pose-change RMSD limit:    {pose_change_max:g} A')
    if missing_from_topk:
        print(
            f'  WARNING: {missing_from_topk} refined candidates were outside '
            '--top_k; increase --top_k to include the complete union'
        )
    print('  Original candidates removed: 0')
    return filtered, audit_rows


def append_gated_refinement_candidates(
    per_complex_entries,
    metrics_csv,
    refine_csv,
    parent_scores_csv=None,
    parent_score_column='combined_at_best_weights',
    centroid_max=None,
    pose_change_max=None,
    audit_csv=None,
):
    """Append accepted refined poses to the original confidence candidate pool.

    The refined pose inherits its parent pose's raw confidence score. Density
    features are then recomputed for both poses, allowing the density reranker
    to choose between the parent and the refinement without discarding the
    confidence model or restricting evaluation to the refined subset.
    """
    if centroid_max is None and pose_change_max is None:
        raise ValueError('append mode requires at least one refinement gate')
    if not refine_csv:
        raise ValueError('append mode requires --refinement_gate_refine_csv')

    metrics_path = Path(metrics_csv)
    refine_path = Path(refine_csv)
    if not metrics_path.is_file():
        raise FileNotFoundError(f'refinement metrics CSV not found: {metrics_path}')
    if not refine_path.is_file():
        raise FileNotFoundError(f'refine CSV not found: {refine_path}')

    metrics = pd.read_csv(metrics_path)
    refine_df = pd.read_csv(refine_path)
    metric_required = {'complex_name', 'pdb_id', 'orig_rank', 'snap_path'}
    if centroid_max is not None:
        metric_required.add('centroid_shift_A')
    refine_required = {'complex_name', 'ligand_description'}
    missing_metrics = metric_required - set(metrics.columns)
    missing_refine = refine_required - set(refine_df.columns)
    if missing_metrics:
        raise ValueError(f'refinement metrics CSV is missing columns: {sorted(missing_metrics)}')
    if missing_refine:
        raise ValueError(f'refine CSV is missing columns: {sorted(missing_refine)}')

    refine_by_name = {
        str(row['complex_name']): str(row['ligand_description'])
        for _, row in refine_df.iterrows()
    }
    selected_parent_path = {}
    if parent_scores_csv:
        parent_scores_path = Path(parent_scores_csv)
        if not parent_scores_path.is_file():
            raise FileNotFoundError(
                f'parent rerank scores CSV not found: {parent_scores_path}'
            )
        parent_scores = pd.read_csv(parent_scores_path)
        parent_required = {'complex', 'sdf_path', parent_score_column}
        parent_missing = parent_required - set(parent_scores.columns)
        if parent_missing:
            raise ValueError(
                f'parent scores CSV is missing columns: {sorted(parent_missing)}'
            )
        for pdb_id, group in parent_scores.groupby('complex'):
            selected = group.sort_values(
                parent_score_column, ascending=False, kind='stable'
            )
            for selected_rank, (_, selected_row) in enumerate(selected.iterrows(), start=1):
                selected_parent_path[(str(pdb_id).upper(), selected_rank)] = str(
                    selected_row['sdf_path']
                )
    appended = {name: list(entries) for name, entries in per_complex_entries.items()}
    next_rank = {
        str(name).upper(): max((entry[0] for entry in entries), default=0) + 1
        for name, entries in appended.items()
    }
    actual_name = {str(name).upper(): name for name in appended}
    resolved_parent_entries = {}
    for name, entries in appended.items():
        lookup = {}
        for rank, score, path in entries:
            try:
                lookup[str(Path(path).resolve())] = (rank, score, path)
            except OSError:
                lookup[str(Path(path).absolute())] = (rank, score, path)
        resolved_parent_entries[str(name).upper()] = lookup

    audit_rows = []
    accepted_count = rejected_count = parent_missing_count = 0
    metrics = metrics.sort_values(['pdb_id', 'orig_rank', 'complex_name'])
    for _, row in metrics.iterrows():
        pdb_key = str(row['pdb_id']).upper()
        complex_name = str(row['complex_name'])
        snap_path = str(row['snap_path'])
        before_path = refine_by_name.get(complex_name, '')
        centroid_shift = (
            float(row['centroid_shift_A']) if centroid_max is not None else float('nan')
        )
        pose_change = float('nan')
        reasons = []

        if pdb_key not in actual_name:
            reasons.append('complex_not_in_rerank_candidates')
        if centroid_max is not None:
            if not np.isfinite(centroid_shift):
                reasons.append('centroid_shift_unavailable')
            elif centroid_shift > centroid_max:
                reasons.append('centroid_shift_exceeds_limit')
        if pose_change_max is not None:
            if not before_path:
                reasons.append('original_pose_path_unavailable')
            else:
                try:
                    pose_change = direct_pose_change_rmsd(before_path, snap_path)
                except Exception as exc:
                    reasons.append(f'pose_change_unavailable:{exc}')
                else:
                    if pose_change > pose_change_max:
                        reasons.append('pose_change_exceeds_limit')

        parent_entry = None
        parent_resolution = ''
        if before_path and pdb_key in resolved_parent_entries:
            try:
                parent_key = str(Path(before_path).resolve())
            except OSError:
                parent_key = str(Path(before_path).absolute())
            parent_entry = resolved_parent_entries[pdb_key].get(parent_key)
            if parent_entry is not None:
                parent_resolution = 'refine_path'
        if parent_entry is None and pdb_key in resolved_parent_entries:
            selected_path = selected_parent_path.get(
                (pdb_key, int(row['orig_rank']))
            )
            if selected_path:
                try:
                    selected_key = str(Path(selected_path).resolve())
                except OSError:
                    selected_key = str(Path(selected_path).absolute())
                parent_entry = resolved_parent_entries[pdb_key].get(selected_key)
                if parent_entry is not None:
                    parent_resolution = 'parent_scores_csv'
        if parent_entry is None:
            reasons.append('parent_not_in_confidence_top_k')
            parent_missing_count += 1

        candidate_rank = None
        accepted = not reasons
        if accepted:
            name = actual_name[pdb_key]
            candidate_rank = next_rank[pdb_key]
            next_rank[pdb_key] += 1
            # Inherit confidence from the unrefined parent. The refined pose
            # gains or loses rank only through its recomputed auxiliary scores.
            appended[name].append((candidate_rank, parent_entry[1], Path(snap_path)))
            accepted_count += 1
        else:
            rejected_count += 1

        audit_rows.append({
            'pdb_id': actual_name.get(pdb_key, str(row['pdb_id'])),
            'complex_name': complex_name,
            'source_orig_rank': int(row['orig_rank']),
            'parent_confidence_rank': parent_entry[0] if parent_entry else '',
            'parent_resolution': parent_resolution,
            'candidate_rank': candidate_rank if candidate_rank is not None else '',
            'is_refined': True,
            'accepted': accepted,
            'rejection_reason': ';'.join(reasons),
            'centroid_shift_A': centroid_shift,
            'pose_change_rmsd_A': pose_change,
            'centroid_max_A': centroid_max,
            'pose_change_max_A': pose_change_max,
            'original_pose_path': before_path,
            'refined_snap_path': snap_path,
            'inherited_confidence_raw': parent_entry[1] if parent_entry else '',
        })

    if audit_csv:
        audit_path = Path(audit_csv)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
        print(f'Refinement-gate audit CSV: {audit_path}')

    print('\n--- Refinement safety gate (append to confidence candidates) ---')
    print(f'  refined candidates audited: {len(audit_rows)}')
    print(f'  accepted and appended:      {accepted_count}')
    print(f'  rejected:                   {rejected_count}')
    print(f'  parent outside top-K:       {parent_missing_count}')
    if centroid_max is not None:
        print(f'  centroid-shift limit:       {centroid_max:g} A')
    if pose_change_max is not None:
        print(f'  pose-change RMSD limit:     {pose_change_max:g} A')
    print('  Original confidence candidates removed: 0')
    return appended, audit_rows


def compute_entry_rmsds(per_complex_entries, data_dir, ligand_suffix='ligand'):
    """Compute symmetry-aware RMSD for explicit candidate paths.

    Unlike ``evaluate_em.compute_rmsds``, this supports in-memory appended
    refinement candidates whose files do not live below ``--rerank_dir``.
    Returned arrays remain addressable by ``candidate_rank - 1``.
    """
    from spyrmsd import rmsd as spyrmsd_rmsd, molecule as spyrmsd_molecule

    result = {}
    for pdb_id, entries in per_complex_entries.items():
        gt_path = Path(data_dir) / pdb_id / f'{pdb_id}_{ligand_suffix}.sdf'
        if not gt_path.is_file():
            print(f'Warning: no ground truth for {pdb_id}: {gt_path}')
            continue
        gt_mol = Chem.MolFromMolFile(str(gt_path), sanitize=True, removeHs=False)
        if gt_mol is None:
            print(f'Warning: could not read ground truth: {gt_path}')
            continue
        gt_mol = RemoveAllHs(gt_mol)
        gt_coords = gt_mol.GetConformer().GetPositions()
        gt_spy = spyrmsd_molecule.Molecule.from_rdkit(gt_mol)
        values = [float('inf')] * max((rank for rank, _score, _path in entries), default=0)
        for rank, _score, pose_path in entries:
            try:
                pose_mol = Chem.MolFromMolFile(
                    str(pose_path), sanitize=True, removeHs=False
                )
                if pose_mol is None:
                    raise ValueError('RDKit returned None')
                pose_mol = RemoveAllHs(pose_mol)
                if pose_mol.GetNumAtoms() != gt_mol.GetNumAtoms():
                    raise ValueError(
                        f'atom count mismatch: gt={gt_mol.GetNumAtoms()}, '
                        f'pose={pose_mol.GetNumAtoms()}'
                    )
                pose_coords = pose_mol.GetConformer().GetPositions()
                value = spyrmsd_rmsd.symmrmsd(
                    gt_coords,
                    pose_coords,
                    gt_spy.atomicnums,
                    gt_spy.atomicnums,
                    gt_spy.adjacency_matrix,
                    gt_spy.adjacency_matrix,
                )
                values[rank - 1] = float(value)
            except Exception as exc:
                print(f'Warning: RMSD failed for {pdb_id} rank{rank}: {exc}')
        if values:
            result[pdb_id] = values
    return result


def read_pose_heavy_mol(sdf_path):
    """Read a pose SDF and return a heavy-atom-only RDKit mol (or None)."""
    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False, removeHs=False)
    if mol is None:
        return None
    try:
        mol = RemoveAllHs(mol, sanitize=False)
    except Exception:
        pass
    if mol.GetNumConformers() == 0:
        return None
    return mol


def read_protein_heavy_coords(pdb_path):
    """Read heavy-atom receptor coordinates from processed protein PDB."""
    st = gemmi.read_structure(pdb_path)
    st.remove_alternative_conformations()
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.element.name == 'H':
                        continue
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.asarray(coords, dtype=np.float32)


def compute_contact_features(pose_coords, protein_coords,
                             contact_cutoff=4.5, clash_cutoff=2.0):
    """Simple receptor-ligand local chemistry proxies from pairwise distances."""
    out = dict.fromkeys(CHEM_FEATURE_NAMES, float('nan'))
    pose_coords = np.asarray(pose_coords, dtype=np.float32)
    protein_coords = np.asarray(protein_coords, dtype=np.float32)
    if pose_coords.size == 0 or protein_coords.size == 0:
        return out

    diff = pose_coords[:, None, :] - protein_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    out['contact_count'] = float(np.sum(dist2 <= float(contact_cutoff) ** 2))
    clash_count = float(np.sum(dist2 <= float(clash_cutoff) ** 2))
    out['neg_clash_count'] = -clash_count
    return out


def compute_ligand_strain_features(sdf_path, self_clash_cutoff=1.8):
    """Lightweight ligand internal quality proxies."""
    out = dict.fromkeys(STRAIN_FEATURE_NAMES, float('nan'))
    mol = read_pose_heavy_mol(sdf_path)
    if mol is None:
        return out

    try:
        mol_ff = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(mol_ff)
        except Exception:
            pass
        # Docking poses often omit explicit hydrogens. Add them transiently for
        # force-field evaluation so MMFF/UFF can assign parameters more reliably
        # without requiring regenerated SDFs from upstream tools.
        try:
            mol_ff = Chem.AddHs(mol_ff, addCoords=True)
        except Exception:
            pass
        energy = float('nan')
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_ff, mmffVariant='MMFF94s')
            if mmff_props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_ff, mmff_props)
                if ff is not None:
                    energy = float(ff.CalcEnergy())
        except Exception:
            pass
        if not np.isfinite(energy):
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol_ff)
                if ff is not None:
                    energy = float(ff.CalcEnergy())
            except Exception:
                pass
        if np.isfinite(energy):
            out['neg_strain_energy'] = -energy
    except Exception:
        pass

    try:
        conf = mol.GetConformer()
        coords = np.asarray(conf.GetPositions(), dtype=np.float32)
        if coords.shape[0] >= 2:
            bond_pairs = {
                tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                for b in mol.GetBonds()
            }
            cutoff2 = float(self_clash_cutoff) ** 2
            clash_count = 0
            for i in range(coords.shape[0]):
                for j in range(i + 1, coords.shape[0]):
                    if (i, j) in bond_pairs:
                        continue
                    dist2 = float(np.sum((coords[i] - coords[j]) ** 2))
                    if dist2 <= cutoff2:
                        clash_count += 1
            out['neg_self_clash_count'] = -float(clash_count)
    except Exception:
        pass
    return out


def gaussian_density_from_atoms(sample_points, atom_coords, sigma=1.2, chunk_size=2048,
                                cutoff=None):
    """Evaluate a Gaussian atom-density model at sample points.

    This is an approximation to simulated protein/ligand density that is fast
    enough for reranking and only needs coordinates. Values are arbitrary units;
    we only use them for within-complex ordering / NCC.
    """
    sample_points = np.asarray(sample_points, dtype=np.float32)
    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    if sample_points.size == 0:
        return np.zeros(0, dtype=np.float32)
    if atom_coords.size == 0:
        return np.zeros(len(sample_points), dtype=np.float32)

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    cutoff2 = None if cutoff is None else float(cutoff) ** 2
    out = np.zeros(len(sample_points), dtype=np.float32)
    for start in range(0, len(sample_points), chunk_size):
        pts = sample_points[start:start + chunk_size]
        diff = pts[:, None, :] - atom_coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if cutoff2 is not None:
            weights = np.where(dist2 <= cutoff2, np.exp(-dist2 * inv_two_sigma2), 0.0)
        else:
            weights = np.exp(-dist2 * inv_two_sigma2)
        out[start:start + len(pts)] = weights.sum(axis=1, dtype=np.float64)
    return out


def make_shape_sample_points(pose_coords, shell_radii=(0.0, 0.8, 1.6)):
    """Build a compact local point cloud around the ligand for envelope matching."""
    pose_coords = np.asarray(pose_coords, dtype=np.float32)
    if pose_coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    directions = np.asarray([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.float32)
    pts = [pose_coords]
    for radius in shell_radii:
        if radius <= 1e-8:
            continue
        pts.append(pose_coords[:, None, :] + radius * directions[None, :, :])
    return np.concatenate([p.reshape(-1, 3) for p in pts], axis=0)


def robust_nonnegative_affine_fit(x, y, mask=None, max_iter=4, z_cutoff=2.5):
    """Robustly fit ``y ~= scale*x + offset`` with non-negative scale."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if valid.sum() < 8:
        return float('nan'), float('nan'), valid

    keep = valid.copy()
    scale = 0.0
    offset = float(np.median(y[keep]))
    for _ in range(max_iter):
        xv = x[keep]
        yv = y[keep]
        if xv.size < 8:
            break
        design = np.column_stack((xv, np.ones_like(xv)))
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        scale = max(0.0, float(coef[0]))
        offset = float(np.median(yv - scale * xv))

        residual = y - (scale * x + offset)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        robust_sd = 1.4826 * mad
        if not np.isfinite(robust_sd) or robust_sd < 1e-8:
            break
        new_keep = valid & (np.abs(residual - center) <= z_cutoff * robust_sd)
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return scale, offset, keep


def residual_shape_corr_from_samples(observed, protein_sim, ligand_sim,
                                     fit_mask, ligand_mask_fraction=0.05):
    """Protein-aware ligand-envelope NCC on already sampled local values."""
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    protein_sim = np.asarray(protein_sim, dtype=np.float64).reshape(-1)
    ligand_sim = np.asarray(ligand_sim, dtype=np.float64).reshape(-1)
    finite = np.isfinite(observed) & np.isfinite(protein_sim) & np.isfinite(ligand_sim)
    if finite.sum() < 8:
        return float('nan')

    scale, offset, _ = robust_nonnegative_affine_fit(
        protein_sim, observed, mask=finite & np.asarray(fit_mask, dtype=bool)
    )
    if not np.isfinite(scale) or not np.isfinite(offset):
        return float('nan')

    residual = observed - (scale * protein_sim + offset)
    ligand_max = float(np.nanmax(ligand_sim[finite]))
    if not np.isfinite(ligand_max) or ligand_max <= 1e-12:
        return float('nan')
    eval_mask = finite & (ligand_sim >= ligand_mask_fraction * ligand_max)
    if eval_mask.sum() < 10:
        return float('nan')
    return float(compute_ncc(residual[eval_mask], ligand_sim[eval_mask]))


def compute_residual_shape_corr(grid, pose_coords, protein_coords,
                                protein_sigma=1.2, protein_cutoff=4.0,
                                ligand_sigma=1.0,
                                shell_radii=(0.0, 0.8, 1.6, 2.4, 3.2),
                                fit_exclusion=2.0,
                                ligand_mask_fraction=0.05):
    """Correlate ligand shape with density left after local protein fitting."""
    pose_coords = np.asarray(pose_coords, dtype=np.float32)
    protein_coords = np.asarray(protein_coords, dtype=np.float32)
    if pose_coords.size == 0 or protein_coords.size == 0:
        return float('nan')

    sample_pts = make_shape_sample_points(pose_coords, shell_radii=shell_radii)
    observed = interp_points(grid, sample_pts)

    # Render only receptor atoms near this pose to control memory and runtime.
    margin = float(protein_cutoff) + max((float(r) for r in shell_radii), default=0.0)
    lo = pose_coords.min(axis=0) - margin
    hi = pose_coords.max(axis=0) + margin
    local_mask = np.all((protein_coords >= lo) & (protein_coords <= hi), axis=1)
    local_protein = protein_coords[local_mask]
    protein_sim = gaussian_density_from_atoms(
        sample_pts, local_protein, sigma=protein_sigma, cutoff=protein_cutoff
    )
    ligand_sim = gaussian_density_from_atoms(
        sample_pts, pose_coords, sigma=ligand_sigma, cutoff=3.0 * ligand_sigma
    )

    # Fit protein scale/offset on shell samples sufficiently far from ligand.
    nearest2 = np.full(len(sample_pts), np.inf, dtype=np.float32)
    for start in range(0, len(sample_pts), 2048):
        pts = sample_pts[start:start + 2048]
        d2 = np.sum((pts[:, None, :] - pose_coords[None, :, :]) ** 2, axis=2)
        nearest2[start:start + len(pts)] = d2.min(axis=1)
    fit_mask = nearest2 >= float(fit_exclusion) ** 2

    return residual_shape_corr_from_samples(
        observed,
        protein_sim,
        ligand_sim,
        fit_mask=fit_mask,
        ligand_mask_fraction=ligand_mask_fraction,
    )


def compute_density_score(grid, pose_coords, trim_pct=0.0):
    """Interpolate EM density at each heavy-atom position and aggregate.

    trim_pct: fraction (0-1) of lowest and highest values to drop before mean
              (reduces sensitivity to single outlier atoms in e.g. clash regions).
              0 → plain mean.
    """
    vals = interp_points(grid, pose_coords)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float('nan')
    if trim_pct > 0 and vals.size >= 4:
        lo = int(np.ceil(trim_pct * vals.size))
        hi = vals.size - lo
        vals = np.sort(vals)[lo:hi]
    return float(vals.mean())


# Features we compute per pose. Add new names here + in compute_density_features()
# to expose them as a CLI knob. Every feature is z-normalised per-complex before
# being combined, so units/scales don't matter — only within-complex ordering.
DENSITY_FEATURE_NAMES = ('mean', 'std', 'min', 'p25', 'high_frac',
                         'neg_std', 'neg_abs_grad', 'diff_mean', 'shape_corr',
                         'residual_shape_corr', 'rscc')
RING_FEATURE_NAMES = ('ring_mean', 'ring_p25', 'ring_closest')
CHEM_FEATURE_NAMES = ('contact_count', 'neg_clash_count')
STRAIN_FEATURE_NAMES = ('neg_strain_energy', 'neg_self_clash_count')


def compute_density_features(grid, pose_coords, trim_pct=0.0, grad_step=0.5,
                             protein_coords=None, diff_sigma=1.2, diff_cutoff=4.0,
                             shape_sigma=1.0, shape_shell_radii=(0.0, 0.8, 1.6),
                             residual_shell_radii=(0.0, 0.8, 1.6, 2.4, 3.2),
                             residual_fit_exclusion=2.0,
                             residual_mask_fraction=0.05,
                             enable_rscc=False,
                             rscc_resolution=3.5):
    """Return a dict of density-derived per-pose features.

    Rationale (for catching drug-like failure modes that plain mean misses):
        - mean       : overall fit to density (existing baseline)
        - std        : uniformity of fit; good poses have all atoms in density
                       evenly → smaller std. Returned as-is (higher is NOT better).
        - neg_std    : -std, so "higher is better" (easier to combine additively).
        - min        : worst-atom density; one atom hanging in vacuum tanks this.
                       Good poses have high min.
        - p25        : 25th percentile; softer version of min (robust to a single
                       noisy atom interpolation).
        - high_frac  : fraction of atoms above per-pose median density; stable
                       proxy for "most atoms are in well-supported regions".
        - neg_abs_grad: -|mean density gradient at atom positions|. Good poses
                       sit on local density peaks (low |grad|); bad poses drift
                       onto slopes (high |grad|). Negated so higher is better.
        - diff_mean  : observed density minus simulated protein density at ligand
                       atom positions. Tries to suppress receptor/background signal.
        - shape_corr : NCC between observed local density and a ligand Gaussian
                       envelope over a local point cloud around the pose.
        - residual_shape_corr: ligand-envelope NCC after robust local fitting
                       and subtraction of simulated receptor density.
        - rscc       : real-space cross-correlation between observed map values
                       and a simple ligand Gaussian density model over the
                       ligand bounding box.

    All returned values assume "higher is better" so downstream combining is
    just additive with positive λ.
    """
    vals = interp_points(grid, pose_coords)
    finite_mask = np.isfinite(vals)
    vals = vals[finite_mask]
    out = dict.fromkeys(DENSITY_FEATURE_NAMES, float('nan'))
    if vals.size == 0:
        return out
    if trim_pct > 0 and vals.size >= 4:
        lo = int(np.ceil(trim_pct * vals.size))
        hi = vals.size - lo
        vals_trim = np.sort(vals)[lo:hi]
    else:
        vals_trim = vals

    out['mean']      = float(vals_trim.mean())
    out['std']       = float(vals_trim.std())
    out['neg_std']   = float(-vals_trim.std())
    out['min']       = float(vals_trim.min())
    out['p25']       = float(np.percentile(vals_trim, 25))
    med = float(np.median(vals_trim))
    out['high_frac'] = float((vals_trim > med).mean())

    # Gradient magnitude via central differences. We only compute this on the
    # finite-value atoms to avoid NaN poisoning from out-of-box atoms.
    try:
        coords_ok = pose_coords[finite_mask]
        if coords_ok.shape[0] >= 1:
            grads = []
            for axis in range(3):
                step = np.zeros(3, dtype=np.float32)
                step[axis] = grad_step
                v_plus  = interp_points(grid, coords_ok + step)
                v_minus = interp_points(grid, coords_ok - step)
                grads.append((v_plus - v_minus) / (2 * grad_step))
            grad_mag = np.sqrt(sum(g ** 2 for g in grads))
            grad_mag = grad_mag[np.isfinite(grad_mag)]
            if grad_mag.size > 0:
                out['neg_abs_grad'] = float(-grad_mag.mean())
    except Exception:
        pass

    # Approximate difference density: subtract a receptor-only Gaussian density
    # model from the observed map at ligand atom positions.
    if protein_coords is not None and len(protein_coords) > 0:
        try:
            protein_sim = gaussian_density_from_atoms(
                pose_coords, protein_coords, sigma=diff_sigma, cutoff=diff_cutoff
            )
            protein_sim = protein_sim[finite_mask]
            if protein_sim.size == vals.size:
                diff_vals = vals - protein_sim
                if trim_pct > 0 and diff_vals.size >= 4:
                    lo = int(np.ceil(trim_pct * diff_vals.size))
                    hi = diff_vals.size - lo
                    diff_vals = np.sort(diff_vals)[lo:hi]
                if diff_vals.size > 0:
                    out['diff_mean'] = float(diff_vals.mean())
        except Exception:
            pass

    # Ligand shape correlation against the observed local density.
    try:
        sample_pts = make_shape_sample_points(pose_coords, shell_radii=shape_shell_radii)
        obs_local = interp_points(grid, sample_pts)
        env_local = gaussian_density_from_atoms(
            sample_pts, pose_coords, sigma=shape_sigma, cutoff=3.0 * shape_sigma
        )
        out['shape_corr'] = compute_ncc(obs_local, env_local)
    except Exception:
        pass
    if protein_coords is not None and len(protein_coords) > 0:
        try:
            out['residual_shape_corr'] = compute_residual_shape_corr(
                grid,
                pose_coords,
                protein_coords,
                protein_sigma=diff_sigma,
                protein_cutoff=diff_cutoff,
                ligand_sigma=shape_sigma,
                shell_radii=residual_shell_radii,
                fit_exclusion=residual_fit_exclusion,
                ligand_mask_fraction=residual_mask_fraction,
            )
        except Exception:
            pass
    if enable_rscc:
        try:
            out['rscc'] = compute_rscc(
                grid, pose_coords, interp_points, resolution=rscc_resolution
            )
        except Exception:
            pass
    return out


def get_ring_npz_path(data_dir, pdb_id):
    return os.path.join(data_dir, pdb_id, f'{pdb_id}_ring_features.npz')


def load_ring_pose_context(npz_path, scalar_mode='mean'):
    """Load receptor-side ring features and collapse them to one scalar/atom.

    The ring npz stores per-receptor-atom channels across multiple radii/stats.
    For reranking we just need a simple receptor-local scalar field that can be
    queried around each ligand pose. Collapsing by mean is the most conservative
    option and avoids overfitting to any one radius/stat channel.
    """
    data = np.load(npz_path, allow_pickle=True)
    coords = np.asarray(data['coords'], dtype=np.float32)
    rings = np.asarray(data['rings'], dtype=np.float32)
    if rings.ndim != 2 or coords.ndim != 2 or coords.shape[0] != rings.shape[0]:
        raise ValueError(f'Unexpected ring feature shapes in {npz_path}: '
                         f'coords={coords.shape}, rings={rings.shape}')
    if scalar_mode == 'mean':
        scalar = np.nanmean(rings, axis=1)
    elif scalar_mode == 'max':
        scalar = np.nanmax(rings, axis=1)
    else:
        raise ValueError(f'Unknown ring scalar mode: {scalar_mode}')
    return coords, np.asarray(scalar, dtype=np.float32)


def compute_ring_pose_features(
    pose_coords,
    ring_coords,
    ring_scalar,
    contact_radius=5.0,
):
    """Build pose-dependent receptor ring summaries.

    For each ligand atom, gather receptor ring-feature atoms within a local
    radius and average their collapsed ring intensity. This turns a receptor-
    only signal into a pose-conditioned one: a pose sitting in a locally
    well-supported receptor environment gets higher ring_mean / ring_p25.
    If no receptor atoms fall within the radius, fall back to the nearest atom.
    """
    out = dict.fromkeys(RING_FEATURE_NAMES, float('nan'))
    pose_coords = np.asarray(pose_coords, dtype=np.float32)
    ring_coords = np.asarray(ring_coords, dtype=np.float32)
    ring_scalar = np.asarray(ring_scalar, dtype=np.float32)
    if (
        pose_coords.size == 0 or ring_coords.size == 0 or ring_scalar.size == 0
        or ring_coords.shape[0] != ring_scalar.shape[0]
    ):
        return out

    diff = pose_coords[:, None, :] - ring_coords[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    radius2 = float(contact_radius) ** 2
    atom_local = []
    nearest_vals = []
    for row in dist2:
        within = row <= radius2
        if np.any(within):
            atom_local.append(float(np.nanmean(ring_scalar[within])))
        else:
            atom_local.append(float('nan'))
        nearest_vals.append(float(ring_scalar[int(np.argmin(row))]))

    atom_local = np.asarray(atom_local, dtype=np.float32)
    nearest_vals = np.asarray(nearest_vals, dtype=np.float32)
    finite_local = atom_local[np.isfinite(atom_local)]
    if finite_local.size > 0:
        out['ring_mean'] = float(finite_local.mean())
        out['ring_p25'] = float(np.percentile(finite_local, 25))
    if nearest_vals.size > 0:
        out['ring_closest'] = float(nearest_vals.mean())
    return out


def compute_pose_consensus_scores(pose_coords_list):
    """Score each pose by how close it is to the rest of the top-K set.

    consensus_i = 1 / (1 + mean_j RMSD(i, j))

    We use direct Cartesian RMSD in the receptor frame because all poses are
    already expressed in the same pocket coordinate system.
    """
    n = len(pose_coords_list)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    mean_rmsds = np.full(n, np.nan, dtype=np.float64)
    for i, coords_i in enumerate(pose_coords_list):
        if coords_i is None or coords_i.shape[0] == 0:
            continue
        pairwise = []
        for j, coords_j in enumerate(pose_coords_list):
            if i == j or coords_j is None or coords_j.shape != coords_i.shape:
                continue
            diff = coords_i - coords_j
            rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
            if np.isfinite(rmsd):
                pairwise.append(rmsd)
        if pairwise:
            mean_rmsds[i] = float(np.mean(pairwise))

    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(mean_rmsds)
    out[valid] = 1.0 / (1.0 + mean_rmsds[valid])
    return out.tolist()


def get_receptor_pdb_path(data_dir, pdb_id):
    return os.path.join(data_dir, pdb_id, f'{pdb_id}_protein_processed.pdb')


def ensure_receptor_pdbqt(pdb_path, pdbqt_path, obabel_path='obabel'):
    """Convert receptor PDB -> PDBQT once and reuse it across poses."""
    if os.path.exists(pdbqt_path) and os.path.getsize(pdbqt_path) > 0:
        return pdbqt_path

    os.makedirs(os.path.dirname(pdbqt_path), exist_ok=True)
    cmd = [obabel_path, pdb_path, '-O', pdbqt_path]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0 or not os.path.exists(pdbqt_path):
        raise RuntimeError(
            f'Open Babel receptor conversion failed for {pdb_path}\n'
            f'cmd: {" ".join(cmd)}\n'
            f'stdout:\n{res.stdout}\n'
            f'stderr:\n{res.stderr}'
        )
    return pdbqt_path


def parse_smina_affinity(output_text):
    """Extract affinity from smina stdout/stderr."""
    patterns = [
        r'Affinity:\s*([-\d.]+)',
        r'^\s*1\s+([-\d.]+)\s+',
    ]
    for line in output_text.splitlines():
        for pattern in patterns:
            m = re.search(pattern, line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f'Could not parse smina affinity from output:\n{output_text}')


def run_smina_score(smina_path, receptor_pdbqt, ligand_sdf, scoring='vina'):
    """Return smina affinity; lower is better."""
    cmd = [smina_path, '--score_only', '-r', receptor_pdbqt, '-l', ligand_sdf]
    if scoring and scoring != 'vina':
        cmd.extend(['--scoring', scoring])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f'smina failed for ligand {ligand_sdf}\n'
            f'cmd: {" ".join(cmd)}\n'
            f'stdout:\n{res.stdout}\n'
            f'stderr:\n{res.stderr}'
        )
    return parse_smina_affinity(res.stdout + '\n' + res.stderr)


def normalise_per_complex(density_scores):
    """Z-normalise density scores within each complex. Returns dict with same shape."""
    out = {}
    for name, scores in density_scores.items():
        arr = np.asarray(scores, dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            out[name] = np.zeros_like(arr)
            continue
        mu = np.nanmean(arr)
        sd = np.nanstd(arr)
        if not np.isfinite(sd) or sd < 1e-8:
            out[name] = np.zeros_like(arr)
        else:
            # A missing optional feature must be neutral, including when its
            # lambda is zero (0 * NaN is still NaN in NumPy).
            out[name] = np.nan_to_num((arr - mu) / sd, nan=0.0)
    return out


def normalise_confidence_per_complex(confidence_scores):
    """Same z-norm on confidence logits so the combination isn't scale-dominated."""
    return normalise_per_complex(confidence_scores)


def evaluate_topk_hit_rate(per_complex_ranking, rmsds_dict, cutoff=2.0,
                           class_map=None):
    """per_complex_ranking: name → list of (orig_rank, combined_score, sdf_path), sorted DESC.
    rmsds_dict: name → list of per-rank RMSDs from evaluate_em.compute_rmsds.
    class_map: optional dict name → ligand class label (drug_like / cofactor / ...).
               When given, the returned dict also contains per-class sub-metrics
               keyed by ``per_class`` → {class_label: metrics_dict}.
    """
    hits_top1 = hits_top5 = hits_oracle = total = 0
    all_top1_rmsd = []
    # Per-class accumulators (only used if class_map provided)
    from collections import defaultdict
    pc_total  = defaultdict(int)
    pc_top1   = defaultdict(int)
    pc_top5   = defaultdict(int)
    pc_oracle = defaultdict(int)
    pc_rmsds  = defaultdict(list)

    for name, ranking in per_complex_ranking.items():
        rmsds = rmsds_dict.get(name)
        if rmsds is None or len(rmsds) == 0:
            continue
        # orig_rank in our file is 1-indexed matching rank{N}.sdf; rmsds is list aligned
        # to sorted rank 1..N. Index into rmsds by orig_rank-1.
        pose_rmsds = []
        for orig_rank, _score, _path in ranking:
            if orig_rank - 1 < len(rmsds):
                pose_rmsds.append(rmsds[orig_rank - 1])
        if not pose_rmsds:
            continue
        total += 1
        is_top1 = pose_rmsds[0] < cutoff
        is_top5 = any(r < cutoff for r in pose_rmsds[:5])
        is_oracle = any(r < cutoff for r in pose_rmsds)
        if is_top1:   hits_top1 += 1
        if is_top5:   hits_top5 += 1
        if is_oracle: hits_oracle += 1
        all_top1_rmsd.append(pose_rmsds[0])

        if class_map is not None:
            cls = class_map.get(name, 'unknown')
            pc_total[cls]  += 1
            pc_top1[cls]   += int(is_top1)
            pc_top5[cls]   += int(is_top5)
            pc_oracle[cls] += int(is_oracle)
            pc_rmsds[cls].append(pose_rmsds[0])

    out = {
        'n': total,
        'top1_hit': hits_top1,
        'top1_pct': 100.0 * hits_top1 / max(total, 1),
        'top5_pct': 100.0 * hits_top5 / max(total, 1),
        'oracle_pct': 100.0 * hits_oracle / max(total, 1),
        'top1_rmsds': all_top1_rmsd,
        'top1_mean_rmsd': float(np.mean(all_top1_rmsd)) if all_top1_rmsd else float('inf'),
        'top1_median_rmsd': float(np.median(all_top1_rmsd)) if all_top1_rmsd else float('inf'),
    }
    if class_map is not None:
        per_class = {}
        for cls in pc_total:
            rmsds_cls = pc_rmsds[cls]
            per_class[cls] = {
                'n': pc_total[cls],
                'top1_hit': pc_top1[cls],
                'top1_pct': 100.0 * pc_top1[cls] / max(pc_total[cls], 1),
                'top5_pct': 100.0 * pc_top5[cls] / max(pc_total[cls], 1),
                'oracle_pct': 100.0 * pc_oracle[cls] / max(pc_total[cls], 1),
                'top1_median_rmsd': float(np.median(rmsds_cls)) if rmsds_cls else float('inf'),
            }
        out['per_class'] = per_class
    return out


def format_optional_seconds(seconds):
    if seconds is None or not np.isfinite(seconds):
        return '-'
    return f'{seconds:.1f}s'


def build_comparison_row(condition, metrics, thresholds, time_s=None):
    top1_rmsds = np.asarray(metrics.get('top1_rmsds', []), dtype=np.float64)
    row = {
        'condition': condition,
        'n': int(metrics.get('n', 0)),
        'mean_rmsd': float(np.mean(top1_rmsds)) if top1_rmsds.size else float('inf'),
        'median_rmsd': float(np.median(top1_rmsds)) if top1_rmsds.size else float('inf'),
        'time_s': time_s,
        'threshold_pcts': [],
    }
    for thr in thresholds:
        pct = 100.0 * float(np.mean(top1_rmsds < thr)) if top1_rmsds.size else 0.0
        row['threshold_pcts'].append((thr, pct))
    return row


def print_comparison_table(rows, thresholds):
    if not rows:
        return

    threshold_headers = [f'<{thr:g}A' for thr in thresholds]
    cond_w = max(20, max(len(r['condition']) for r in rows))
    n_w = max(4, len('N'))
    mean_w = max(8, len('Mean'))
    median_w = max(8, len('Median'))
    thr_w = max(7, max(len(h) for h in threshold_headers))
    time_w = max(7, len('Time'))

    print('\n' + '=' * 70)
    print('  COMPARISON')
    print('=' * 70)
    header = (
        f'  {"Condition":<{cond_w}} {"N":>{n_w}} '
        f'{"Mean":>{mean_w}} {"Median":>{median_w}} '
        + ' '.join(f'{h:>{thr_w}}' for h in threshold_headers)
        + f' {"Time":>{time_w}}'
    )
    print(header)
    print(
        f'  {"-" * cond_w} {"-" * n_w} '
        f'{"-" * mean_w} {"-" * median_w} '
        + ' '.join(f'{"-" * thr_w}' for _ in threshold_headers)
        + f' {"-" * time_w}'
    )
    for row in rows:
        mean_s = 'n/a' if not np.isfinite(row['mean_rmsd']) else f'{row["mean_rmsd"]:.2f}A'
        median_s = 'n/a' if not np.isfinite(row['median_rmsd']) else f'{row["median_rmsd"]:.2f}A'
        thr_s = ' '.join(f'{pct:>{thr_w - 1}.1f}%' for _thr, pct in row['threshold_pcts'])
        print(
            f'  {row["condition"]:<{cond_w}} {row["n"]:>{n_w}d} '
            f'{mean_s:>{mean_w}} {median_s:>{median_w}} '
            f'{thr_s} {format_optional_seconds(row["time_s"]):>{time_w}}'
        )
    print('=' * 70)


def main():
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--rerank_dir', required=True,
                    help='Dir of confidence-reranked SDFs (e.g. results_test_confidence_rerank/finetuned_em_conf)')
    ap.add_argument('--map_csv', required=True,
                    help='CSV with pdb_id,map_path,status (rows where status==ok are used)')
    ap.add_argument('--data_dir', required=True,
                    help='Processed data dir supplying receptor PDB and ring features; also supplies RMSD ground truth unless --gt_data_dir is set')
    ap.add_argument('--gt_data_dir', default=None,
                    help='Optional separate RMSD-reference root; useful for deposited-reference directories that do not contain receptor/ring files')
    ap.add_argument('--gt_ligand_suffix', default='ligand',
                    help='Ground-truth filename suffix after <PDB_ID>_ and before .sdf (default: ligand; deposited reference roots use ligand_deposited)')
    ap.add_argument('--test_split', default=None,
                    help='Optional split file to filter which complexes to evaluate')
    ap.add_argument('--chemistry_manifest', default=None,
                    help='Reviewed chemistry_manifest.csv; replaces the legacy ligand heuristic for class-stratified outputs')
    ap.add_argument('--top_k', type=int, default=5,
                    help='Consider only the top-K ranked poses per complex (default: 5)')
    ap.add_argument('--ignore_confidence', action='store_true',
                    help='Force every confidence score to zero, even when pose filenames '
                         'contain _confidence*. Use with top_k equal to the complete pose '
                         'count for a true density-only ranking.')
    ap.add_argument('--refinement_gate_metrics_csv', default=None,
                    help='Optional refinement_metrics.csv. When set, apply inference-time '
                         'safety gates only to appended refined poses; originals are retained.')
    ap.add_argument('--refinement_gate_mode', choices=('filter_union', 'append'),
                    default='filter_union',
                    help='filter_union filters refined ranks already present below --rerank_dir; '
                         'append keeps --rerank_dir as the full confidence candidate pool and '
                         'appends accepted snap_path poses from the metrics CSV.')
    ap.add_argument('--refinement_gate_refine_csv', default=None,
                    help='Refine input CSV containing complex_name and ligand_description; '
                         'required for the direct pose-change RMSD gate.')
    ap.add_argument('--refinement_parent_scores_csv', default=None,
                    help='Original rerank_by_density.py --out_csv used to select poses for '
                         'refinement. In append mode this recovers the confidence parent when '
                         'the selected top-K poses were copied rather than symlinked.')
    ap.add_argument('--refinement_parent_score_column',
                    default='combined_at_best_weights',
                    help='Score column used to order --refinement_parent_scores_csv.')
    ap.add_argument('--refinement_gate_centroid_max', type=float, default=None,
                    help='Accept a refined pose only when centroid_shift_A is at most this value.')
    ap.add_argument('--refinement_gate_pose_change_max', type=float, default=None,
                    help='Accept a refined pose only when direct original-to-refined heavy-atom '
                         'coordinate RMSD is at most this value (no alignment, no ground truth).')
    ap.add_argument('--refinement_gate_out_csv', default=None,
                    help='Optional per-refined-pose gate audit CSV.')
    ap.add_argument('--trim_pct', type=float, default=0.0,
                    help='Trim fraction of atoms from each tail before mean density (default: 0)')
    ap.add_argument('--lambda_grid', type=str, default='0.0,0.5,1.0,2.0,5.0',
                    help='Comma-separated lambda weights for density mean (back-compat alias for --lambda_mean)')
    ap.add_argument('--consensus_lambda_grid', type=str, default='0.0',
                    help='Comma-separated consensus weights to sweep: new += λ_consensus*consensus_z')
    ap.add_argument('--smina_lambda_grid', type=str, default='0.0',
                    help='Comma-separated smina weights to sweep: new += λ_smina*smina_z')

    # Per-feature λ grids. Defaults to 0 (disabled) for all extras so existing
    # invocations keep behaving the same. Sweep only the feature(s) you want to
    # investigate to avoid cartesian-product blow-up.
    ap.add_argument('--lambda_mean', type=str, default=None,
                    help='Override --lambda_grid for density mean')
    ap.add_argument('--lambda_std', type=str, default='0.0',
                    help='Comma-separated λ for density std (higher std = less uniform fit)')
    ap.add_argument('--lambda_neg_std', type=str, default='0.0',
                    help='Comma-separated λ for -std (higher neg_std = more uniform = better)')
    ap.add_argument('--lambda_min', type=str, default='0.0',
                    help='Comma-separated λ for min density (worst-atom fit)')
    ap.add_argument('--lambda_p25', type=str, default='0.0',
                    help='Comma-separated λ for 25th percentile density (robust min)')
    ap.add_argument('--lambda_high_frac', type=str, default='0.0',
                    help='Comma-separated λ for fraction-of-atoms-above-median-density')
    ap.add_argument('--lambda_neg_abs_grad', type=str, default='0.0',
                    help='Comma-separated λ for -|density gradient| (higher = atom on density peak)')
    ap.add_argument('--lambda_diff_mean', type=str, default='0.0',
                    help='Comma-separated λ for difference-density mean (observed minus simulated protein density)')
    ap.add_argument('--lambda_shape_corr', type=str, default='0.0',
                    help='Comma-separated λ for ligand-envelope/local-density NCC')
    ap.add_argument('--lambda_residual_shape_corr', type=str, default='0.0',
                    help='Comma-separated λ for protein-subtracted ligand-envelope NCC')
    ap.add_argument('--lambda_rscc', type=str, default='0.0',
                    help='Comma-separated λ for ligand real-space cross-correlation coefficient')
    ap.add_argument('--lambda_ring_mean', type=str, default='0.0',
                    help='Comma-separated λ for local receptor ring summary mean around the pose')
    ap.add_argument('--lambda_ring_p25', type=str, default='0.0',
                    help='Comma-separated λ for 25th percentile of local receptor ring support')
    ap.add_argument('--lambda_ring_closest', type=str, default='0.0',
                    help='Comma-separated λ for mean nearest-receptor ring support per ligand atom')
    ap.add_argument('--lambda_contact_count', type=str, default='0.0',
                    help='Comma-separated λ for receptor-ligand heavy-atom contact count')
    ap.add_argument('--lambda_neg_clash_count', type=str, default='0.0',
                    help='Comma-separated λ for negative receptor-ligand clash count')
    ap.add_argument('--lambda_neg_strain_energy', type=str, default='0.0',
                    help='Comma-separated λ for negative ligand MMFF/UFF single-point energy')
    ap.add_argument('--lambda_neg_self_clash_count', type=str, default='0.0',
                    help='Comma-separated λ for negative ligand internal self-clash count')
    ap.add_argument('--contact_cutoff', type=float, default=4.5,
                    help='Contact cutoff (A) for receptor-ligand heavy-atom contacts')
    ap.add_argument('--clash_cutoff', type=float, default=2.0,
                    help='Clash cutoff (A) for receptor-ligand heavy-atom clashes')
    ap.add_argument('--self_clash_cutoff', type=float, default=1.8,
                    help='Internal ligand self-clash cutoff (A) for non-bonded heavy atoms')
    ap.add_argument('--ring_contact_radius', type=float, default=5.0,
                    help='Radius (A) for aggregating receptor ring values around each ligand atom')
    ap.add_argument('--ring_scalar_mode', choices=('mean', 'max'), default='mean',
                    help='How to collapse multi-channel receptor ring features to one scalar per receptor atom')
    ap.add_argument('--diff_sigma', type=float, default=1.2,
                    help='Gaussian sigma (A) for simulated protein density in difference-density feature')
    ap.add_argument('--diff_cutoff', type=float, default=4.0,
                    help='Cutoff radius (A) for protein Gaussian density contribution')
    ap.add_argument('--shape_sigma', type=float, default=1.0,
                    help='Gaussian sigma (A) for ligand envelope in shape correlation')
    ap.add_argument('--shape_shell_radii', type=str, default='0.0,0.8,1.6',
                    help='Comma-separated shell radii (A) around each ligand atom for shape correlation')
    ap.add_argument('--residual_shell_radii', type=str, default='0.0,0.8,1.6,2.4,3.2',
                    help='Shell radii (A) used for local protein fitting and residual-shape NCC')
    ap.add_argument('--residual_fit_exclusion', type=float, default=2.0,
                    help='Minimum distance (A) from ligand atoms for protein scale/offset fitting')
    ap.add_argument('--residual_mask_fraction', type=float, default=0.05,
                    help='Fraction of maximum ligand envelope defining residual-NCC support')
    ap.add_argument('--rscc_resolution', type=float, default=3.5,
                    help='Resolution (A) used to set the Gaussian width in RSCC computation')
    ap.add_argument('--smina_path', default='smina',
                    help='Path to smina executable')
    ap.add_argument('--obabel_path', default='obabel',
                    help='Path to Open Babel executable used for receptor PDB->PDBQT conversion')
    ap.add_argument('--smina_scoring', default='vina',
                    help='Smina scoring function name, e.g. vina or vinardo')
    ap.add_argument('--smina_cache_dir', default=None,
                    help='Optional cache dir for receptor PDBQT files; default: <rerank_dir>/_smina_cache')
    ap.add_argument('--out_csv', default=None,
                    help='Optional path to write per-complex scored results')
    ap.add_argument('--comparison_thresholds', type=str, default='2.0,3.5,5.0',
                    help='Comma-separated RMSD thresholds (A) for the COMPARISON table')
    args = ap.parse_args()
    shape_shell_radii = tuple(float(x) for x in args.shape_shell_radii.split(',') if x.strip())
    residual_shell_radii = tuple(
        float(x) for x in args.residual_shell_radii.split(',') if x.strip()
    )
    if not residual_shell_radii or any(r < 0 for r in residual_shell_radii):
        ap.error('--residual_shell_radii must contain one or more non-negative values')
    if args.residual_fit_exclusion <= 0:
        ap.error('--residual_fit_exclusion must be positive')
    if not 0.0 < args.residual_mask_fraction <= 1.0:
        ap.error('--residual_mask_fraction must be in (0, 1]')
    comparison_thresholds = [float(x) for x in args.comparison_thresholds.split(',') if x.strip()]

    def _grid_has_nonzero(s):
        return any(abs(float(x)) > 1e-12 for x in s.split(',') if x.strip())

    use_smina = _grid_has_nonzero(args.smina_lambda_grid)
    use_diff_mean = _grid_has_nonzero(args.lambda_diff_mean)
    use_residual_shape = _grid_has_nonzero(args.lambda_residual_shape_corr)
    use_rscc = _grid_has_nonzero(args.lambda_rscc)
    use_ring = any(
        _grid_has_nonzero(s)
        for s in (args.lambda_ring_mean, args.lambda_ring_p25, args.lambda_ring_closest)
    )
    use_chem = any(
        _grid_has_nonzero(s)
        for s in (args.lambda_contact_count, args.lambda_neg_clash_count)
    )
    use_strain = any(
        _grid_has_nonzero(s)
        for s in (args.lambda_neg_strain_energy, args.lambda_neg_self_clash_count)
    )

    # 1. Parse per-complex top-K confidence rankings
    per_complex_entries = parse_rerank_dir(
        args.rerank_dir,
        top_k=args.top_k,
        ignore_confidence=args.ignore_confidence,
    )
    print(f'Parsed {len(per_complex_entries)} complexes from {args.rerank_dir}')
    if args.ignore_confidence:
        print('Confidence disabled: all pose confidence scores forced to zero')

    gate_audit_rows = []
    if args.refinement_gate_metrics_csv:
        try:
            if args.refinement_gate_mode == 'append':
                per_complex_entries, gate_audit_rows = append_gated_refinement_candidates(
                    per_complex_entries=per_complex_entries,
                    metrics_csv=args.refinement_gate_metrics_csv,
                    refine_csv=args.refinement_gate_refine_csv,
                    parent_scores_csv=args.refinement_parent_scores_csv,
                    parent_score_column=args.refinement_parent_score_column,
                    centroid_max=args.refinement_gate_centroid_max,
                    pose_change_max=args.refinement_gate_pose_change_max,
                    audit_csv=args.refinement_gate_out_csv,
                )
            else:
                per_complex_entries, gate_audit_rows = apply_refinement_gates(
                    per_complex_entries=per_complex_entries,
                    rerank_dir=args.rerank_dir,
                    metrics_csv=args.refinement_gate_metrics_csv,
                    refine_csv=args.refinement_gate_refine_csv,
                    centroid_max=args.refinement_gate_centroid_max,
                    pose_change_max=args.refinement_gate_pose_change_max,
                    audit_csv=args.refinement_gate_out_csv,
                )
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))
    elif any(value is not None for value in (
        args.refinement_gate_refine_csv,
        args.refinement_parent_scores_csv,
        args.refinement_gate_centroid_max,
        args.refinement_gate_pose_change_max,
        args.refinement_gate_out_csv,
    )):
        ap.error(
            '--refinement_gate_metrics_csv is required when any refinement gate '
            'option is supplied'
        )

    # Apply the requested evaluation cohort before any density work.  Older
    # versions only filtered the RMSD lookup, so stale pose directories could
    # still be probed and written to --out_csv even though they were outside
    # the current test split.
    if args.test_split:
        requested_ids = load_test_ids(args.test_split)
        allowed = {name.upper() for name in requested_ids}
        before = len(per_complex_entries)
        per_complex_entries = {
            name: entries for name, entries in per_complex_entries.items()
            if name.upper() in allowed
        }
        present = {name.upper() for name in per_complex_entries}
        missing_requested = [
            name for name in requested_ids if name.upper() not in present
        ]
        print(
            f'Test-split filter: retained {len(per_complex_entries)}/{before} '
            f'pose complexes ({len(requested_ids)} requested IDs)'
        )
        if missing_requested:
            print(
                f'  Missing predictions for {len(missing_requested)}/{len(requested_ids)} '
                f'requested complexes: {" ".join(missing_requested[:20])}'
            )

    # 2. Load map mapping & RMSD ground truth
    map_mapping = load_map_mapping(args.map_csv)
    pdb_ids = list(per_complex_entries.keys())
    gt_data_dir = args.gt_data_dir or args.data_dir
    # RMSD ground truth may live separately from receptor/ring feature data.
    print(f'Computing ground-truth RMSDs for {len(pdb_ids)} complexes...')
    print(f'  Ground-truth root: {gt_data_dir}')
    print(f'  Ground-truth suffix: {args.gt_ligand_suffix}')
    rmsds_dict = compute_entry_rmsds(
        per_complex_entries,
        gt_data_dir,
        ligand_suffix=args.gt_ligand_suffix,
    )

    # 2b. Classify each complex's ligand once (drug_like / cofactor / ...)
    print('Classifying ligands by chemistry...')
    class_map = {}
    reviewed_classes = (
        load_reviewed_chemistry_manifest(args.chemistry_manifest)
        if args.chemistry_manifest else None
    )
    missing_reviewed = []
    for pdb_id in tqdm(per_complex_entries.keys(), desc='Classify'):
        key = str(pdb_id).strip().upper()
        if reviewed_classes is not None:
            class_map[pdb_id] = reviewed_classes.get(key, 'unknown')
            if key not in reviewed_classes:
                missing_reviewed.append(key)
        else:
            class_map[pdb_id] = classify_complex(
                gt_data_dir, pdb_id, ligand_suffix=args.gt_ligand_suffix
            )
    if reviewed_classes is not None:
        print(f'Using reviewed chemistry labels from {args.chemistry_manifest}')
        if missing_reviewed:
            print(
                f'WARNING: {len(missing_reviewed)} pose complexes are missing '
                f'from the reviewed manifest; using unknown: {" ".join(missing_reviewed[:20])}'
            )
    from collections import Counter
    cls_counter = Counter(class_map.values())
    print(f'  Ligand-class breakdown: {dict(cls_counter)}')

    # 3. Compute density scores per pose, caching open_map per complex
    conf_scores = {}   # name → list of floats (aligned with per_complex_entries[name])
    dens_scores = {}   # name → list of floats (back-compat: still holds 'mean')
    # Extended: per-feature {feature_name: {complex_name: [floats]}}
    dens_feat_scores = {fname: {} for fname in DENSITY_FEATURE_NAMES}
    consensus_scores = {}
    smina_scores = {}
    ring_feat_scores = {fname: {} for fname in RING_FEATURE_NAMES}
    chem_feat_scores = {fname: {} for fname in CHEM_FEATURE_NAMES}
    strain_feat_scores = {fname: {} for fname in STRAIN_FEATURE_NAMES}
    protein_coord_cache = {}
    ring_context_cache = {}
    missing_map = missing_coords = 0
    missing_ring = 0
    missing_receptor = failed_pdbqt = failed_smina = 0
    smina_cache_dir = args.smina_cache_dir or os.path.join(args.rerank_dir, '_smina_cache')
    for name, entries in tqdm(per_complex_entries.items(), desc='Density probe'):
        pose_coords_list = []
        c_list = []
        for _rank, score, sdf_path in entries:
            coords = read_pose_heavy_coords(sdf_path)
            pose_coords_list.append(coords)
            c_list.append(score)
            if coords is None or coords.shape[0] == 0:
                missing_coords += 1
        conf_scores[name] = c_list
        consensus_scores[name] = compute_pose_consensus_scores(pose_coords_list)

        receptor_pdb = get_receptor_pdb_path(args.data_dir, name)
        receptor_pdbqt = os.path.join(smina_cache_dir, name, f'{name}_protein_processed.pdbqt')
        if (use_diff_mean or use_residual_shape or use_chem) and name not in protein_coord_cache:
            try:
                protein_coord_cache[name] = read_protein_heavy_coords(receptor_pdb)
            except Exception:
                protein_coord_cache[name] = np.zeros((0, 3), dtype=np.float32)
        if use_ring and name not in ring_context_cache:
            ring_npz = get_ring_npz_path(args.data_dir, name)
            if os.path.exists(ring_npz):
                try:
                    ring_context_cache[name] = load_ring_pose_context(
                        ring_npz, scalar_mode=args.ring_scalar_mode
                    )
                except Exception:
                    ring_context_cache[name] = None
                    missing_ring += 1
            else:
                ring_context_cache[name] = None
                missing_ring += 1
        if not use_smina:
            smina_scores[name] = [float('nan')] * len(entries)
        elif not os.path.exists(receptor_pdb):
            missing_receptor += 1
            smina_scores[name] = [float('nan')] * len(entries)
        else:
            try:
                ensure_receptor_pdbqt(receptor_pdb, receptor_pdbqt, obabel_path=args.obabel_path)
                s_list = []
                for _rank, _score, sdf_path in entries:
                    try:
                        affinity = run_smina_score(
                            args.smina_path,
                            receptor_pdbqt,
                            str(sdf_path),
                            scoring=args.smina_scoring,
                        )
                        # Lower (more negative) affinity is better, so flip sign.
                        s_list.append(-affinity)
                    except Exception:
                        failed_smina += 1
                        s_list.append(float('nan'))
                smina_scores[name] = s_list
            except Exception:
                failed_pdbqt += 1
                smina_scores[name] = [float('nan')] * len(entries)

        pdb_key = name.upper()
        map_path = map_mapping.get(pdb_key)
        if map_path is None or not os.path.exists(map_path):
            missing_map += 1
            dens_scores[name] = [float('nan')] * len(entries)
            for fname in DENSITY_FEATURE_NAMES:
                dens_feat_scores[fname][name] = [float('nan')] * len(entries)
            for fname in CHEM_FEATURE_NAMES:
                chem_feat_scores[fname][name] = [float('nan')] * len(entries)
            for fname in STRAIN_FEATURE_NAMES:
                strain_feat_scores[fname][name] = [float('nan')] * len(entries)
            continue
        try:
            _m, grid = open_map(map_path)
        except Exception as e:
            print(f'  {name}: open_map failed — {e}')
            missing_map += 1
            dens_scores[name] = [float('nan')] * len(entries)
            for fname in DENSITY_FEATURE_NAMES:
                dens_feat_scores[fname][name] = [float('nan')] * len(entries)
            for fname in CHEM_FEATURE_NAMES:
                chem_feat_scores[fname][name] = [float('nan')] * len(entries)
            for fname in STRAIN_FEATURE_NAMES:
                strain_feat_scores[fname][name] = [float('nan')] * len(entries)
            continue

        # Collect all density features per pose in one pass (reuses the interp).
        feat_lists = {fname: [] for fname in DENSITY_FEATURE_NAMES}
        ring_lists = {fname: [] for fname in RING_FEATURE_NAMES}
        chem_lists = {fname: [] for fname in CHEM_FEATURE_NAMES}
        strain_lists = {fname: [] for fname in STRAIN_FEATURE_NAMES}
        for (_rank, _score, sdf_path), coords in zip(entries, pose_coords_list):
            if coords is None or coords.shape[0] == 0:
                for fname in DENSITY_FEATURE_NAMES:
                    feat_lists[fname].append(float('nan'))
                for fname in RING_FEATURE_NAMES:
                    ring_lists[fname].append(float('nan'))
                for fname in CHEM_FEATURE_NAMES:
                    chem_lists[fname].append(float('nan'))
                for fname in STRAIN_FEATURE_NAMES:
                    strain_lists[fname].append(float('nan'))
                continue
            feats = compute_density_features(
                grid,
                coords,
                trim_pct=args.trim_pct,
                protein_coords=protein_coord_cache.get(name),
                diff_sigma=args.diff_sigma,
                diff_cutoff=args.diff_cutoff,
                shape_sigma=args.shape_sigma,
                shape_shell_radii=shape_shell_radii,
                residual_shell_radii=residual_shell_radii,
                residual_fit_exclusion=args.residual_fit_exclusion,
                residual_mask_fraction=args.residual_mask_fraction,
                enable_rscc=use_rscc,
                rscc_resolution=args.rscc_resolution,
            )
            for fname in DENSITY_FEATURE_NAMES:
                feat_lists[fname].append(feats[fname])
            if use_ring:
                ring_ctx = ring_context_cache.get(name)
                if ring_ctx is None:
                    ring_feats = dict.fromkeys(RING_FEATURE_NAMES, float('nan'))
                else:
                    ring_feats = compute_ring_pose_features(
                        coords,
                        ring_ctx[0],
                        ring_ctx[1],
                        contact_radius=args.ring_contact_radius,
                    )
            else:
                ring_feats = dict.fromkeys(RING_FEATURE_NAMES, float('nan'))
            for fname in RING_FEATURE_NAMES:
                ring_lists[fname].append(ring_feats[fname])
            if use_chem:
                chem_feats = compute_contact_features(
                    coords,
                    protein_coord_cache.get(name, np.zeros((0, 3), dtype=np.float32)),
                    contact_cutoff=args.contact_cutoff,
                    clash_cutoff=args.clash_cutoff,
                )
            else:
                chem_feats = dict.fromkeys(CHEM_FEATURE_NAMES, float('nan'))
            for fname in CHEM_FEATURE_NAMES:
                chem_lists[fname].append(chem_feats[fname])
            if use_strain:
                strain_feats = compute_ligand_strain_features(
                    sdf_path,
                    self_clash_cutoff=args.self_clash_cutoff,
                )
            else:
                strain_feats = dict.fromkeys(STRAIN_FEATURE_NAMES, float('nan'))
            for fname in STRAIN_FEATURE_NAMES:
                strain_lists[fname].append(strain_feats[fname])

        # Back-compat: dens_scores keeps the mean (used by existing output/CSV paths).
        dens_scores[name] = feat_lists['mean']
        for fname in DENSITY_FEATURE_NAMES:
            dens_feat_scores[fname][name] = feat_lists[fname]
        for fname in RING_FEATURE_NAMES:
            ring_feat_scores[fname][name] = ring_lists[fname]
        for fname in CHEM_FEATURE_NAMES:
            chem_feat_scores[fname][name] = chem_lists[fname]
        for fname in STRAIN_FEATURE_NAMES:
            strain_feat_scores[fname][name] = strain_lists[fname]

    print(f'  Density probed for {len(conf_scores)} complexes  '
          f'(missing map: {missing_map}, missing coords: {missing_coords})')
    if use_residual_shape:
        residual_values = [
            value
            for values in dens_feat_scores['residual_shape_corr'].values()
            for value in values
        ]
        residual_finite = sum(np.isfinite(value) for value in residual_values)
        residual_complexes = sum(
            any(np.isfinite(value) for value in values)
            for values in dens_feat_scores['residual_shape_corr'].values()
        )
        print(
            f'  Residual-shape NCC available for {residual_finite}/{len(residual_values)} poses '
            f'across {residual_complexes}/{len(per_complex_entries)} complexes'
        )
    if use_ring:
        print(f'  Ring context loaded for {len(per_complex_entries) - missing_ring} complexes  '
              f'(missing/failed ring npz: {missing_ring})')
    if use_smina:
        print(f'  Smina scored for {len(smina_scores)} complexes  '
              f'(missing receptor: {missing_receptor}, pdbqt failures: {failed_pdbqt}, '
              f'pose score failures: {failed_smina})')
    else:
        print('  Smina skipped (all λ_smina are zero)')

    # 4. Sweep lambda and report hit rates
    conf_z = normalise_confidence_per_complex(conf_scores)
    # Per-feature z-scores. dens_feat_z['mean'] == the original density signal.
    dens_feat_z = {
        fname: normalise_per_complex(dens_feat_scores[fname])
        for fname in DENSITY_FEATURE_NAMES
    }
    ring_feat_z = {
        fname: normalise_per_complex(ring_feat_scores[fname])
        for fname in RING_FEATURE_NAMES
    }
    chem_feat_z = {
        fname: normalise_per_complex(chem_feat_scores[fname])
        for fname in CHEM_FEATURE_NAMES
    }
    strain_feat_z = {
        fname: normalise_per_complex(strain_feat_scores[fname])
        for fname in STRAIN_FEATURE_NAMES
    }
    consensus_z = normalise_per_complex(consensus_scores)
    smina_z = normalise_per_complex(smina_scores)

    # In append mode, preserve an independently normalised original-only arm.
    # This makes the final table a fair three-way comparison:
    # confidence only; confidence+density on originals; and the same scorer
    # after adding gated refinements. Refined candidates must not change the
    # z-normalisation used by the original-only control arm.
    accepted_refined_keys = {
        (str(row['pdb_id']).upper(), int(row['candidate_rank']))
        for row in gate_audit_rows
        if row.get('accepted') and row.get('candidate_rank') not in ('', None)
    }
    append_mode_active = (
        args.refinement_gate_metrics_csv
        and args.refinement_gate_mode == 'append'
    )
    original_entries = {
        name: [
            entry for entry in entries
            if (str(name).upper(), int(entry[0])) not in accepted_refined_keys
        ]
        for name, entries in per_complex_entries.items()
    }
    original_indices = {
        name: [
            idx for idx, entry in enumerate(entries)
            if (str(name).upper(), int(entry[0])) not in accepted_refined_keys
        ]
        for name, entries in per_complex_entries.items()
    }

    def _original_subset(score_dict):
        return {
            name: [score_dict[name][idx] for idx in original_indices[name]]
            for name in original_entries
            if name in score_dict
        }

    original_conf_z = normalise_confidence_per_complex(_original_subset(conf_scores))
    original_dens_feat_z = {
        fname: normalise_per_complex(_original_subset(dens_feat_scores[fname]))
        for fname in DENSITY_FEATURE_NAMES
    }
    original_ring_feat_z = {
        fname: normalise_per_complex(_original_subset(ring_feat_scores[fname]))
        for fname in RING_FEATURE_NAMES
    }
    original_chem_feat_z = {
        fname: normalise_per_complex(_original_subset(chem_feat_scores[fname]))
        for fname in CHEM_FEATURE_NAMES
    }
    original_strain_feat_z = {
        fname: normalise_per_complex(_original_subset(strain_feat_scores[fname]))
        for fname in STRAIN_FEATURE_NAMES
    }
    original_consensus_z = normalise_per_complex(_original_subset(consensus_scores))
    original_smina_z = normalise_per_complex(_original_subset(smina_scores))

    # Parse each λ grid. `lambda_mean` overrides `lambda_grid` if supplied so
    # legacy invocations keep working (mean=primary density feature).
    def _parse_grid(s):
        return [float(x) for x in s.split(',')]

    mean_lambdas = _parse_grid(args.lambda_mean if args.lambda_mean else args.lambda_grid)
    std_lambdas  = _parse_grid(args.lambda_std)
    neg_std_lambdas = _parse_grid(args.lambda_neg_std)
    min_lambdas  = _parse_grid(args.lambda_min)
    p25_lambdas  = _parse_grid(args.lambda_p25)
    high_frac_lambdas = _parse_grid(args.lambda_high_frac)
    neg_abs_grad_lambdas = _parse_grid(args.lambda_neg_abs_grad)
    diff_mean_lambdas = _parse_grid(args.lambda_diff_mean)
    shape_corr_lambdas = _parse_grid(args.lambda_shape_corr)
    residual_shape_corr_lambdas = _parse_grid(args.lambda_residual_shape_corr)
    rscc_lambdas = _parse_grid(args.lambda_rscc)
    ring_mean_lambdas = _parse_grid(args.lambda_ring_mean)
    ring_p25_lambdas = _parse_grid(args.lambda_ring_p25)
    ring_closest_lambdas = _parse_grid(args.lambda_ring_closest)
    contact_lambdas = _parse_grid(args.lambda_contact_count)
    neg_clash_lambdas = _parse_grid(args.lambda_neg_clash_count)
    neg_strain_lambdas = _parse_grid(args.lambda_neg_strain_energy)
    neg_self_clash_lambdas = _parse_grid(args.lambda_neg_self_clash_count)
    consensus_lambdas = _parse_grid(args.consensus_lambda_grid)
    smina_lambdas = _parse_grid(args.smina_lambda_grid)

    # Each entry in the sweep is a dict of the non-zero feature λs + metrics.
    sweep_results = []
    class_order = ['drug_like', 'cofactor', 'amino_acid', 'phospholipid', 'unknown']

    # Enumerate all grid combinations. Users should keep most grids at [0.0] and
    # sweep just one or two features at a time to avoid explosion.
    from itertools import product
    all_combos = list(product(
        mean_lambdas, std_lambdas, neg_std_lambdas, min_lambdas,
        p25_lambdas, high_frac_lambdas, neg_abs_grad_lambdas,
        diff_mean_lambdas, shape_corr_lambdas, residual_shape_corr_lambdas,
        rscc_lambdas,
        ring_mean_lambdas, ring_p25_lambdas, ring_closest_lambdas,
        contact_lambdas, neg_clash_lambdas,
        neg_strain_lambdas, neg_self_clash_lambdas,
        consensus_lambdas, smina_lambdas,
    ))
    print(f'\n--- Results (top-K rerank on CONFIDENCE TOP {args.top_k}) ---')
    print(f'Sweeping {len(all_combos)} λ combinations  (λ names: mean,std,neg_std,min,p25,high_frac,neg_abs_grad,diff_mean,shape_corr,residual_shape_corr,rscc,ring_mean,ring_p25,ring_closest,contact,neg_clash,neg_strain,neg_self_clash,cons,smina)')
    print(f'{"λ_mean":>7}  {"λ_std":>6}  {"λ_nst":>6}  {"λ_min":>6}  {"λ_p25":>6}  {"λ_hf":>6}  {"λ_ng":>6}  '
          f'{"λ_dif":>6}  {"λ_shp":>6}  {"λ_rshp":>7}  {"λ_rsc":>6}  {"λ_rmn":>6}  {"λ_rp25":>7}  {"λ_rcl":>6}  {"λ_cnt":>6}  {"λ_ncl":>6}  '
          f'{"λ_nse":>6}  {"λ_nsc":>6}  {"λ_cons":>7}  {"λ_smina":>8}  {"top1%":>6}  {"top5%":>6}  {"oracle%":>7}')

    def _lambda_label(lambdas_dict):
        """Compact label for a λ-combo: only show features with non-zero λ."""
        nz = [(k, v) for k, v in lambdas_dict.items() if abs(v) > 1e-12]
        if not nz:
            return '(all zero)'
        return ','.join(f'{k}={v:g}' for k, v in nz)

    for (lm, ls, lns, lmin, lp25, lhf, lng, ldiff, lshape, lrshape, lrscc, lrm, lrp25, lrcl,
         lcnt, lncl, lnse, lnsc, lc, lsm) in all_combos:
        per_complex_ranking = {}
        for name, entries in per_complex_entries.items():
            if name not in conf_scores:
                continue
            n = len(entries)
            c = conf_z[name]
            f_mean = dens_feat_z['mean'].get(name, np.zeros(n))
            f_std  = dens_feat_z['std'].get(name, np.zeros(n))
            f_nstd = dens_feat_z['neg_std'].get(name, np.zeros(n))
            f_min  = dens_feat_z['min'].get(name, np.zeros(n))
            f_p25  = dens_feat_z['p25'].get(name, np.zeros(n))
            f_hf   = dens_feat_z['high_frac'].get(name, np.zeros(n))
            f_ng   = dens_feat_z['neg_abs_grad'].get(name, np.zeros(n))
            f_diff = dens_feat_z['diff_mean'].get(name, np.zeros(n))
            f_shp  = dens_feat_z['shape_corr'].get(name, np.zeros(n))
            f_rshp = dens_feat_z['residual_shape_corr'].get(name, np.zeros(n))
            f_rscc = dens_feat_z['rscc'].get(name, np.zeros(n))
            f_rmn  = ring_feat_z['ring_mean'].get(name, np.zeros(n))
            f_rp25 = ring_feat_z['ring_p25'].get(name, np.zeros(n))
            f_rcl  = ring_feat_z['ring_closest'].get(name, np.zeros(n))
            f_cnt  = chem_feat_z['contact_count'].get(name, np.zeros(n))
            f_ncl  = chem_feat_z['neg_clash_count'].get(name, np.zeros(n))
            f_nse  = strain_feat_z['neg_strain_energy'].get(name, np.zeros(n))
            f_nsc  = strain_feat_z['neg_self_clash_count'].get(name, np.zeros(n))
            s      = consensus_z[name]
            sm     = smina_z[name]
            combined = (
                c
                + lm  * f_mean + ls * f_std + lns * f_nstd
                + lmin * f_min + lp25 * f_p25 + lhf * f_hf + lng * f_ng
                + ldiff * f_diff + lshape * f_shp + lrshape * f_rshp + lrscc * f_rscc
                + lrm * f_rmn + lrp25 * f_rp25 + lrcl * f_rcl
                + lcnt * f_cnt + lncl * f_ncl + lnse * f_nse + lnsc * f_nsc
                + lc * s + lsm * sm
            )
            # Stable ordering is required for the all-zero baseline used in
            # density-only experiments: it must preserve the incoming rank
            # order instead of arbitrarily permuting tied poses.
            order = np.argsort(-np.asarray(combined), kind='stable')
            ranking = [entries[i] for i in order]
            ranking = [(e[0], float(combined[i]), e[2]) for i, e in zip(order, ranking)]
            per_complex_ranking[name] = ranking

        metrics = evaluate_topk_hit_rate(
            per_complex_ranking, rmsds_dict, class_map=class_map
        )
        original_metrics = None
        if append_mode_active:
            original_ranking = {}
            for name, entries in original_entries.items():
                n = len(entries)
                if not n or name not in original_conf_z:
                    continue
                combined = (
                    original_conf_z[name]
                    + lm * original_dens_feat_z['mean'].get(name, np.zeros(n))
                    + ls * original_dens_feat_z['std'].get(name, np.zeros(n))
                    + lns * original_dens_feat_z['neg_std'].get(name, np.zeros(n))
                    + lmin * original_dens_feat_z['min'].get(name, np.zeros(n))
                    + lp25 * original_dens_feat_z['p25'].get(name, np.zeros(n))
                    + lhf * original_dens_feat_z['high_frac'].get(name, np.zeros(n))
                    + lng * original_dens_feat_z['neg_abs_grad'].get(name, np.zeros(n))
                    + ldiff * original_dens_feat_z['diff_mean'].get(name, np.zeros(n))
                    + lshape * original_dens_feat_z['shape_corr'].get(name, np.zeros(n))
                    + lrshape * original_dens_feat_z['residual_shape_corr'].get(name, np.zeros(n))
                    + lrscc * original_dens_feat_z['rscc'].get(name, np.zeros(n))
                    + lrm * original_ring_feat_z['ring_mean'].get(name, np.zeros(n))
                    + lrp25 * original_ring_feat_z['ring_p25'].get(name, np.zeros(n))
                    + lrcl * original_ring_feat_z['ring_closest'].get(name, np.zeros(n))
                    + lcnt * original_chem_feat_z['contact_count'].get(name, np.zeros(n))
                    + lncl * original_chem_feat_z['neg_clash_count'].get(name, np.zeros(n))
                    + lnse * original_strain_feat_z['neg_strain_energy'].get(name, np.zeros(n))
                    + lnsc * original_strain_feat_z['neg_self_clash_count'].get(name, np.zeros(n))
                    + lc * original_consensus_z.get(name, np.zeros(n))
                    + lsm * original_smina_z.get(name, np.zeros(n))
                )
                order = np.argsort(-np.asarray(combined), kind='stable')
                original_ranking[name] = [
                    (entries[idx][0], float(combined[idx]), entries[idx][2])
                    for idx in order
                ]
            original_metrics = evaluate_topk_hit_rate(
                original_ranking, rmsds_dict, class_map=class_map
            )
        sweep_results.append({
            'lambdas': dict(mean=lm, std=ls, neg_std=lns, min=lmin, p25=lp25,
                            high_frac=lhf, neg_abs_grad=lng, diff_mean=ldiff,
                            shape_corr=lshape, residual_shape_corr=lrshape,
                            rscc=lrscc, ring_mean=lrm, ring_p25=lrp25,
                            ring_closest=lrcl, contact_count=lcnt, neg_clash_count=lncl,
                            neg_strain_energy=lnse, neg_self_clash_count=lnsc,
                            cons=lc, smina=lsm),
            'metrics': metrics,
            'original_metrics': original_metrics,
        })
        print(f'{lm:>7.2f}  {ls:>6.2f}  {lns:>6.2f}  {lmin:>6.2f}  {lp25:>6.2f}  {lhf:>6.2f}  {lng:>6.2f}  {ldiff:>6.2f}  {lshape:>6.2f}  {lrshape:>7.2f}  {lrscc:>6.2f}  '
              f'{lrm:>6.2f}  {lrp25:>7.2f}  {lrcl:>6.2f}  {lcnt:>6.2f}  {lncl:>6.2f}  {lnse:>6.2f}  {lnsc:>6.2f}  {lc:>7.2f}  {lsm:>8.2f}  '
              f'{metrics["top1_pct"]:>5.2f}%  {metrics["top5_pct"]:>5.2f}%  {metrics["oracle_pct"]:>6.2f}%')

    comparison_rows = []
    if append_mode_active:
        confidence_only_metrics = evaluate_topk_hit_rate(
            original_entries, rmsds_dict, class_map=class_map
        )
        comparison_rows.append(build_comparison_row(
            condition='confidence only',
            metrics=confidence_only_metrics,
            thresholds=comparison_thresholds,
            time_s=None,
        ))
    for result in sweep_results:
        label = _lambda_label(result['lambdas'])
        if append_mode_active:
            comparison_rows.append(build_comparison_row(
                condition=f'{label} [original poses]',
                metrics=result['original_metrics'],
                thresholds=comparison_thresholds,
                time_s=None,
            ))
            comparison_rows.append(build_comparison_row(
                condition=f'{label} [+ gated refine]',
                metrics=result['metrics'],
                thresholds=comparison_thresholds,
                time_s=None,
            ))
        else:
            comparison_rows.append(build_comparison_row(
                condition=label,
                metrics=result['metrics'],
                thresholds=comparison_thresholds,
                time_s=None,
            ))
    print_comparison_table(comparison_rows, comparison_thresholds)

    # ── Per-class breakdown across the λ sweep ────────────────────────────────
    # For each class × λ-combo, show top1%  (and n for context). Makes it obvious
    # which ligand class benefits most from a given feature mix.
    print('\n--- Per-class top1% by λ-combo ---')
    classes_present = [c for c in class_order if any(
        c in r['metrics'].get('per_class', {}) for r in sweep_results
    )]
    label_width = max(max(len(_lambda_label(r['lambdas'])) for r in sweep_results) + 2, 14)
    class_width = max(max(len(c) + 10 for c in classes_present), 12)
    header_cells = ['λ-combo'] + [
        f'{c}(n={sweep_results[0]["metrics"]["per_class"].get(c, {}).get("n", 0)})'
        for c in classes_present
    ]
    header = (
        f'{header_cells[0]:<{label_width}}'
        + ''.join(f'{h:>{class_width}}' for h in header_cells[1:])
    )
    print(header)
    for r in sweep_results:
        pc = r['metrics'].get('per_class', {})
        row_cells = [_lambda_label(r['lambdas'])]
        for c in classes_present:
            cm = pc.get(c)
            if cm and cm['n'] > 0:
                row_cells.append(f'{cm["top1_pct"]:.2f}%')
            else:
                row_cells.append('-')
        print(
            f'{row_cells[0]:<{label_width}}'
            + ''.join(f'{cell:>{class_width}}' for cell in row_cells[1:])
        )

    # Also report improvement delta (best λ vs baseline all-zero) per class
    if len(sweep_results) >= 2:
        baseline_metrics = None
        for r in sweep_results:
            if all(abs(v) < 1e-12 for v in r['lambdas'].values()):
                baseline_metrics = r['metrics']
                break
        best = max(sweep_results, key=lambda r: r['metrics']['top1_pct'])
        if baseline_metrics is not None and any(abs(v) >= 1e-12 for v in best['lambdas'].values()):
            print(f'\n--- Improvement per class: all-zero → best [{_lambda_label(best["lambdas"])}] ---')
            print(f'{"class":>16}  {"n":>5}  {"base top1%":>10}  {"best top1%":>12}  {"Δ":>8}')
            for c in classes_present:
                a = baseline_metrics.get('per_class', {}).get(c)
                b = best['metrics'].get('per_class', {}).get(c)
                if a and b and a['n'] > 0:
                    delta = b['top1_pct'] - a['top1_pct']
                    print(f'{c:>16}  {a["n"]:>5}  {a["top1_pct"]:>9.2f}%  '
                          f'{b["top1_pct"]:>11.2f}%  {delta:>+7.2f}%')

    # 5. Optional CSV
    if args.out_csv:
        best = max(sweep_results, key=lambda r: r['metrics']['top1_pct'])
        blam = best['lambdas']
        gate_lookup = {
            (
                str(row['pdb_id']).upper(),
                int(row.get('candidate_rank', row.get('union_rank'))),
            ): row
            for row in gate_audit_rows
            if row.get('accepted')
        }
        print(f'\nBest λ combo = [{_lambda_label(blam)}]  '
              f'top1={best["metrics"]["top1_pct"]:.2f}%')

        def _zv(zdict, nm, idx):
            """Safe z-score lookup: 0.0 when a feature was never computed
            (e.g. ring/chem/strain/smina disabled because their λ are all 0)."""
            arr = zdict.get(nm)
            return float(arr[idx]) if arr is not None and idx < len(arr) else 0.0

        rows = []
        for name, entries in per_complex_entries.items():
            if name not in conf_scores:
                continue
            for i, (rank, _score, path) in enumerate(entries):
                gate_row = gate_lookup.get((str(name).upper(), int(rank)))
                combined = (
                    _zv(conf_z, name, i)
                    + blam['mean']         * _zv(dens_feat_z['mean'], name, i)
                    + blam['std']          * _zv(dens_feat_z['std'], name, i)
                    + blam['neg_std']      * _zv(dens_feat_z['neg_std'], name, i)
                    + blam['min']          * _zv(dens_feat_z['min'], name, i)
                    + blam['p25']          * _zv(dens_feat_z['p25'], name, i)
                    + blam['high_frac']    * _zv(dens_feat_z['high_frac'], name, i)
                    + blam['neg_abs_grad'] * _zv(dens_feat_z['neg_abs_grad'], name, i)
                    + blam['diff_mean']    * _zv(dens_feat_z['diff_mean'], name, i)
                    + blam['shape_corr']   * _zv(dens_feat_z['shape_corr'], name, i)
                    + blam['residual_shape_corr'] * _zv(
                        dens_feat_z['residual_shape_corr'], name, i
                    )
                    + blam['ring_mean']    * _zv(ring_feat_z['ring_mean'], name, i)
                    + blam['ring_p25']     * _zv(ring_feat_z['ring_p25'], name, i)
                    + blam['ring_closest'] * _zv(ring_feat_z['ring_closest'], name, i)
                    + blam['contact_count'] * _zv(chem_feat_z['contact_count'], name, i)
                    + blam['neg_clash_count'] * _zv(chem_feat_z['neg_clash_count'], name, i)
                    + blam['neg_strain_energy'] * _zv(strain_feat_z['neg_strain_energy'], name, i)
                    + blam['neg_self_clash_count'] * _zv(strain_feat_z['neg_self_clash_count'], name, i)
                    + blam['cons']         * _zv(consensus_z, name, i)
                    + blam['smina']        * _zv(smina_z, name, i)
                )
                rows.append({
                    'complex': name,
                    'ligand_class': class_map.get(name, 'unknown'),
                    'original_rank': rank,
                    'rmsd': rmsds_dict.get(name, [float('nan')] * len(entries))[rank - 1]
                            if rank - 1 < len(rmsds_dict.get(name, [])) else float('nan'),
                    'is_hit_2A': int(
                        rank - 1 < len(rmsds_dict.get(name, []))
                        and rmsds_dict[name][rank - 1] < 2.0
                    ),
                    'confidence_raw': conf_scores[name][i],
                    'confidence_z': _zv(conf_z, name, i),
                    **{f'dens_{fname}_z': _zv(dens_feat_z[fname], name, i)
                       for fname in DENSITY_FEATURE_NAMES},
                    **{f'{fname}_z': _zv(ring_feat_z[fname], name, i)
                       for fname in RING_FEATURE_NAMES},
                    **{f'{fname}_z': _zv(chem_feat_z[fname], name, i)
                       for fname in CHEM_FEATURE_NAMES},
                    **{f'{fname}_z': _zv(strain_feat_z[fname], name, i)
                       for fname in STRAIN_FEATURE_NAMES},
                    'consensus_z': _zv(consensus_z, name, i),
                    'smina_z': _zv(smina_z, name, i),
                    'combined_at_best_weights': combined,
                    'sdf_path': str(path),
                    'is_refined_candidate': int(gate_row is not None),
                    'refinement_source_rank': (
                        gate_row['source_orig_rank'] if gate_row is not None else ''
                    ),
                    'refinement_centroid_shift_A': (
                        gate_row['centroid_shift_A'] if gate_row is not None else ''
                    ),
                    'refinement_pose_change_rmsd_A': (
                        gate_row['pose_change_rmsd_A'] if gate_row is not None else ''
                    ),
                })
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f'Wrote per-pose scores to {args.out_csv}')
    print(f'Total wall time: {time.perf_counter() - t_start:.1f}s')


if __name__ == '__main__':
    main()
