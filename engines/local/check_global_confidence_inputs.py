#!/usr/bin/env python3
"""Fail early for missing ESM coverage or incomplete Global/confidence poses."""
import argparse
from pathlib import Path
import re


def check_poses(root, ids, confidence=False):
    pattern = re.compile(r'^rank(\d+)_confidence([-+\d.eE]+)\.sdf$' if confidence else r'^rank(\d+)\.sdf$')
    for name in ids:
        ranks = []
        for path in (Path(root)/name).glob('*.sdf'):
            match = pattern.fullmatch(path.name)
            if match:
                if not path.stat().st_size:
                    raise ValueError(f'Empty pose: {path}')
                ranks.append(int(match[1]))
        if sorted(ranks) != list(range(1, 41)):
            raise ValueError(f'{name}: expected exactly ranks 1..40 in {root}; found {sorted(ranks)}')


def check_embeddings(raw, ids):
    import math
    groups = {}
    for key, emb in raw.items():
        name = key.split('_chain_')[0].upper()
        if name in ids:
            if getattr(emb, 'ndim', None) != 2 or emb.shape[0] < 1 or emb.shape[1] != 1280:
                raise ValueError(f'{key}: expected nonempty [residues,1280] ESM tensor')
            if not math.isfinite(float(emb.sum())):
                raise ValueError(f'{key}: nonfinite ESM values')
            groups.setdefault(name, []).append(key)
    missing = sorted(set(ids)-set(groups))
    if missing:
        raise ValueError(f'ESM file missing requested cases: {missing}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--ids', help='Space-separated subset for preflight only')
    p.add_argument('--embeddings', type=Path)
    p.add_argument('--poses', type=Path)
    p.add_argument('--confidence', action='store_true')
    a = p.parse_args()
    ids = (a.ids if a.ids else a.cohort.read_text()).split()
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Empty or duplicate cohort')
    if a.embeddings:
        import torch
        check_embeddings(torch.load(a.embeddings, map_location='cpu'), ids)
        print(f'ESM_COVERAGE_OK: {len(ids)} cases; {a.embeddings}')
    if a.poses:
        check_poses(a.poses, ids, a.confidence)
        print(f'POSES_OK: {len(ids)} cases x 40 poses; {a.poses}')


if __name__ == '__main__':
    main()
