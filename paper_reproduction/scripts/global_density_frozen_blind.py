#!/usr/bin/env python3
"""Frozen Global formula using existing feature helpers, never RMSD main().

Uses feature defaults of the installed Global rerank_by_density.py; record
its hash. Confidence top20 is normalized within each complex exactly via
that module's normalization helpers. Does not call its GT-dependent sweep.
"""
import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['global-root','cohort','poses','map-csv','data-dir','output']:
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite score output')
    sys.path.insert(0,str(a.global_root.resolve()))
    m=importlib.import_module('rerank_by_density')
    ids=[s.split()[0].upper() for s in a.cohort.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if len(ids)!=len(set(ids)) or not ids:
        raise ValueError('Invalid cohort')
    maps=m.load_map_mapping(str(a.map_csv))
    rows=[]
    for i,name in enumerate(ids):
        entries=[]
        for path in (a.poses/name).glob('rank*.sdf'):
            match=m.RANK_RE.fullmatch(path.name)
            if match and int(match.group(1))<=20:
                if match.group(2) is None:
                    raise ValueError(f'Missing original confidence: {path}')
                entries.append((int(match.group(1)),float(match.group(2)),path.resolve()))
        entries.sort(key=lambda entry:entry[0])
        if [e[0] for e in entries]!=list(range(1,21)):
            raise ValueError(f'{name}: expected exactly confidence ranks 1..20; do not drop this parent')
        protein=m.read_protein_heavy_coords(m.get_receptor_pdb_path(str(a.data_dir),name))
        if not len(protein):
            raise ValueError(f'{name}: missing receptor coordinates')
        handle,grid=m.open_map(maps[name])
        try:
            features=[]
            for rank,conf,path in entries:
                xyz=m.read_pose_heavy_coords(path)
                if xyz is None or not len(xyz) or not np.isfinite(xyz).all():
                    raise ValueError(f'Invalid pose: {path}')
                features.append(m.compute_density_features(grid,xyz,protein_coords=protein,enable_rscc=False))
            conf=np.array([e[1] for e in entries])
            mean=np.array([f['mean'] for f in features])
            residual=np.array([f['residual_shape_corr'] for f in features])
            if not np.isfinite(np.concatenate([conf,mean,residual])).all():
                raise ValueError(f'{name}: nonfinite score features')
            cz=np.asarray(m.normalise_confidence_per_complex({name:conf})[name])
            dz=np.asarray(m.normalise_per_complex({name:mean})[name])
            rz=np.asarray(m.normalise_per_complex({name:residual})[name])
            for j,(rank,confidence,path) in enumerate(entries):
                rows.append(dict(complex=name,original_rank=rank,sdf_path=str(path),confidence_raw=confidence,
                    confidence_z=float(cz[j]),density_mean_z=float(dz[j]),residual_shape_corr_z=float(rz[j]),
                    combined_at_best_weights=float(cz[j]+.5*dz[j]+rz[j])))
        finally:
            if hasattr(handle,'close'):
                handle.close()
        print(f'FROZEN_DENSITY {i+1}/{len(ids)} {name}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output,index=False)
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    a.output.with_suffix('.summary.json').write_text(json.dumps(dict(
        cohort_n=len(ids),rows=len(rows),poses_per_complex=20,uses_reference=False,uses_rmsd=False,
        selector='original confidence_z + 0.5*density_mean_z + 1.0*residual_shape_corr_z',
        feature_signature=str(inspect.signature(m.compute_density_features)),
        global_module_sha256=sha(m.__file__),cohort_sha256=sha(a.cohort),score_sha256=sha(a.output)),indent=2)+'\n')
    print(f'FROZEN_GLOBAL_SCORES_OK: {a.output}')


if __name__=='__main__':
    main()
