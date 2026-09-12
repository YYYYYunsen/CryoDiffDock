#!/usr/bin/env python3
"""Replay only failed RMSD evaluations preserved before directory cleanup.

Does not change structures, choose new poses, or produce benchmark scores.
"""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def saved_audit(report, run_name):
    import pandas as pd
    matches = [run for run in report['failed_runs'] if Path(run['path']).name == run_name]
    if len(matches) != 1 or not matches[0].get('audit_csv'):
        raise ValueError(f'No unique preserved audit for {run_name}')
    return pd.read_csv(io.StringIO(matches[0]['audit_csv']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-report', required=True, type=Path)
    parser.add_argument('--run-name', default='global_vs_cascade174_reference_fix_v1')
    parser.add_argument('--reference-manifest', type=Path, default=Path('/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1/01_prepared/evaluation_manifest.csv'))
    args = parser.parse_args()
    import numpy as np
    from rdkit import Chem
    import spyrmsd
    from spyrmsd import molecule
    from evaluation.compare_global_cascade import reference_paths
    from evaluation.evaluate_ligand_accuracy import read_heavy_molecule, symmetry_rmsd

    audit = saved_audit(json.loads(args.diagnostic_report.read_text()), args.run_name)
    failed = audit[audit.global_top1_rmsd_A.isna()]
    print(f'Preserved audit: {len(audit)} cases; {len(failed)} failures; no full-cohort rerun.', flush=True)
    print(f'RDKit={Chem.rdBase.rdkitVersion}; spyrmsd={getattr(spyrmsd,"__version__","unknown")}', flush=True)
    if failed.empty:
        return
    refs = reference_paths(args.reference_manifest, failed.pdb_id.str.upper().tolist())
    failure_count = 0
    for _, row in failed.iterrows():
        pdb = str(row.pdb_id).upper()
        print(f'\nCASE {pdb}\nprevious_error={row.evaluation_error}\nreference={refs.loc[pdb]}\nprediction={row.global_pose}', flush=True)
        try:
            ref, pred = read_heavy_molecule(Path(refs.loc[pdb])), read_heavy_molecule(Path(row.global_pose))
            for label, mol in [('reference', ref), ('prediction', pred)]:
                spy = molecule.Molecule.from_rdkit(mol)
                xyz = np.asarray(mol.GetConformer().GetPositions())
                details = dict(atoms=mol.GetNumAtoms(), bonds=mol.GetNumBonds(),
                               elements=dict(Counter(atom.GetSymbol() for atom in mol.GetAtoms())),
                               remaining_hydrogens=sum(atom.GetAtomicNum()==1 for atom in mol.GetAtoms()),
                               fragments=len(Chem.GetMolFrags(mol)), coordinate_shape=list(xyz.shape),
                               coordinates_finite=bool(np.isfinite(xyz).all()),
                               spy_atomicnums_shape=list(np.asarray(spy.atomicnums).shape),
                               spy_adjacency_shape=list(np.asarray(spy.adjacency_matrix).shape))
                print(f'{label}: {json.dumps(details)}', flush=True)
            value = symmetry_rmsd(ref, pred)
            print(f'replayed_RMSD_A={value}', flush=True)
        except Exception:
            failure_count += 1
            # Full stack identifies the failed assertion; an empty
            # AssertionError message alone cannot diagnose the cause.
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    print(f'REPLAY_RESULT: {len(failed)-failure_count}/{len(failed)} succeeded', flush=True)
    return 1 if failure_count else 0


if __name__ == '__main__':
    raise SystemExit(main())
