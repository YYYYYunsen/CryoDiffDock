#!/usr/bin/env python3
"""Exploratory Global-preserving cascade selectors on existing poses.

Blind selection is sealed before importing per-parent/branch RMSD tables.
Every candidate sees the SAME five predicted-centre crop/mask blocks.
This is a new score, not the native V2 NCC; overlapping crop regions are
repeated observations, equally repeated for every candidate. No native pose
coordinates, centre, RMSD, or branch-local standardized scores select poses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cryodiffdock_pocket.density_ranking import (
    load_cartesian_density_crop, fixed_crop_mask, normalized_target,
    render_gaussian_ligand_density, normalized_cross_correlation,
)
from evaluation.compare_global_cascade import summarize, parents, numeric, resolved

RULES = ("union_shared_ncc", "union_multiscale_guard", "union_rank_fusion_guard")
BLIND_COLUMNS = ("complex_name", "parent_complex_name", "global_rank", "global_pose", "ligand_density_crop")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pose(path):
    from evaluation.evaluate_ligand_accuracy import read_heavy_molecule
    mol = read_heavy_molecule(path)
    coords = np.asarray(mol.GetConformer().GetPositions())
    elements = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    if not np.isfinite(coords).all():
        raise ValueError(f'Nonfinite coordinates: {path}')
    return coords, elements


def choose(features, sigma_columns, complete=True):
    """Input allowlist deliberately excludes all RMSD and reference fields."""
    f = features[["candidate", "global_rank", "basin_q25_ncc", *sigma_columns]].copy()
    g = f[f.candidate.eq('global')]
    if len(g) != 1 or not np.isfinite(f[sigma_columns].to_numpy(float)).all():
        raise ValueError('Exactly one Global anchor and finite shared scores required')
    if not complete:
        return {rule: ('global', 'incomplete_top5_global_fallback') for rule in RULES}
    f['shared_mean'] = f[sigma_columns].mean(axis=1)
    ranked = f.sort_values(['shared_mean','global_rank','candidate'], ascending=[False,True,True],kind='stable')
    output = {'union_shared_ncc': (ranked.iloc[0].candidate, 'highest_shared_mean_global_wins_ties')}
    locals_ = f[f.candidate.ne('global')].copy()
    # Strictly positive gain at EACH scale. No threshold fitted to this test set.
    eligible = locals_[(locals_[sigma_columns] > g.iloc[0][sigma_columns].to_numpy(float) + 1e-12).all(axis=1)].copy()
    if eligible.empty:
        output.update({r: ('global','no_local_improves_every_scale') for r in RULES[1:]})
        return output
    best = eligible.sort_values(['shared_mean','global_rank','candidate'],ascending=[False,True,True],kind='stable').iloc[0]
    output['union_multiscale_guard'] = (best.candidate, 'positive_gain_at_each_scale')
    # Equal reciprocal-rank fusion: Global site rank, original branch-basin
    # evidence, and newly comparable common-target NCC. No z-score mixing.
    if not np.isfinite(locals_.basin_q25_ncc.to_numpy(float)).all():
        raise ValueError('Missing basin evidence for rank fusion')
    locals_['fusion'] = (1/locals_.global_rank +
                         1/locals_.basin_q25_ncc.rank(ascending=False, method='min') +
                         1/locals_.shared_mean.rank(ascending=False, method='min'))
    eligible = eligible.merge(locals_[['candidate','fusion']],on='candidate',validate='one_to_one')
    best = eligible.sort_values(['fusion','global_rank','candidate'],ascending=[False,True,True],kind='stable').iloc[0]
    output['union_rank_fusion_guard'] = (best.candidate, 'equal_rank_fusion_after_multiscale_guard')
    return output


def collect(args, ids):
    blind = pd.read_csv(args.blind_manifest, usecols=lambda c: c in BLIND_COLUMNS)
    if set(BLIND_COLUMNS)-set(blind):
        raise ValueError(f'Blind manifest missing {set(BLIND_COLUMNS)-set(blind)}')
    blind['parent_complex_name'] = blind.parent_complex_name.astype(str).str.strip().str.upper()
    blind = blind[blind.parent_complex_name.isin(ids)].copy()
    blind.global_rank = pd.to_numeric(blind.global_rank,errors='raise')
    if blind.complex_name.duplicated().any() or blind.duplicated(['parent_complex_name','global_rank']).any():
        raise ValueError('Duplicate branch identities')
    expected = {(p,k) for p in ids for k in range(1,6)}
    if set(zip(blind.parent_complex_name,blind.global_rank)) != expected:
        raise ValueError('Need exactly five prepared branches for every authoritative parent')
    directories = {}
    for directory in args.selected_root.iterdir():
        if '___' in directory.name:
            branch = directory.name.split('___',1)[1]
            if branch in directories:
                raise ValueError(f'Duplicate selected directory for {branch}')
            directories[branch] = directory
    plans, inventory = {}, []
    for pdb in ids:
        group = blind[blind.parent_complex_name.eq(pdb)].sort_values('global_rank')
        crops, candidates = [], []
        for _, row in group.iterrows():
            crop = resolved(row.ligand_density_crop,args.blind_manifest.parent)
            if not crop.is_file():
                raise FileNotFoundError(crop)
            crops.append(crop)
            if int(row.global_rank)==1:
                global_pose = resolved(row.global_pose,args.blind_manifest.parent)
                if not global_pose.is_file():
                    raise FileNotFoundError(global_pose)
                candidates.append(dict(candidate='global',global_rank=0,branch_name='',candidate_id='global',pose_path=str(global_pose),basin_q25_ncc=np.nan))
            name = str(row.complex_name)
            directory = directories.get(name)
            if directory is None:
                continue
            if not directory.is_dir():
                raise ValueError(f'Broken selected directory link: {directory}')
            ranking = pd.read_csv(directory/'ranking.csv',usecols=lambda c: c in {'complex_name','candidate_id','basin_q25_ncc'})
            if len(ranking)!=1 or set(ranking)!={'complex_name','candidate_id','basin_q25_ncc'} or str(ranking.iloc[0].complex_name)!=name:
                raise ValueError(f'Invalid selected ranking: {directory}')
            poses = [p for p in directory.glob('rank1*.sdf') if re.fullmatch(r'rank1_confidence[-+0-9.eE]+\.sdf',p.name)]
            if len(poses)!=1 or not poses[0].is_file():
                raise ValueError(f'Expected one materialized Top1 SDF: {directory}')
            selected = ranking.iloc[0]
            candidates.append(dict(candidate=name,branch_name=name,candidate_id=str(selected.candidate_id),global_rank=int(row.global_rank),pose_path=str(poses[0].resolve()),basin_q25_ncc=float(selected.basin_q25_ncc)))
        # Fully read all inputs before making the output directory or scoring.
        for crop_path in crops:
            crop = load_cartesian_density_crop(crop_path)
            mask = fixed_crop_mask(crop,radius_A=args.mask_radius)
            if np.std(normalized_target(crop)[mask]) <= 1e-12:
                raise ValueError(f'Constant target crop: {crop_path}')
            inventory.append(dict(path=str(crop_path),sha256=sha(crop_path),role='density_crop',pdb_id=pdb))
        elements = None
        for candidate in candidates:
            _, z = read_pose(Path(candidate['pose_path']))
            if elements is None:
                elements = sorted(z)
            elif elements != sorted(z):
                raise ValueError(f'{pdb}: candidate heavy-atom compositions disagree')
            inventory.append(dict(path=candidate['pose_path'],sha256=sha(candidate['pose_path']),role=candidate['candidate'],pdb_id=pdb))
        plans[pdb] = (crops,candidates)
    return plans, inventory


def score_parent(crops, candidates, sigmas, radius):
    grids = [load_cartesian_density_crop(p) for p in crops]
    masks = [fixed_crop_mask(c,radius_A=radius) for c in grids]
    target = np.concatenate([normalized_target(c)[m] for c,m in zip(grids,masks)])
    rows = []
    for candidate in candidates:
        xyz, z = read_pose(Path(candidate['pose_path']))
        row = dict(candidate)
        for i,sigma in enumerate(sigmas):
            model = np.concatenate([render_gaussian_ligand_density(c,xyz,z,sigma)[m] for c,m in zip(grids,masks)])
            row[f'shared_ncc_{i}'] = normalized_cross_correlation(model,target)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(args,ids,features,selections):
    # Invoked only AFTER writing and hashing the deployment-time selections.
    baseline = parents(args.baseline_dir/'per_parent_rmsd.csv',ids)
    audit = parents(args.baseline_dir/'global_top1_audit.csv',ids)
    if not np.allclose(numeric(baseline['global']),numeric(audit.global_top1_rmsd_A),equal_nan=True):
        raise ValueError('Baseline Global audit disagrees with baseline parent metrics')
    branch = pd.read_csv(args.branch_table)
    required = {'complex_name','candidate_id','selected_pose_rmsd_A','parent_complex_name','global_rank'}
    if required-set(branch) or branch.complex_name.duplicated().any():
        raise ValueError('Invalid cross_branch_input_table schema/branch duplicates')
    branch = branch.set_index('complex_name')
    frame = baseline[['global','rank1','basin_q25_ncc','oracle_best5']].copy()
    for name in frame:
        frame[name]=numeric(frame[name])
    for rule in RULES:
        frame[rule]=np.nan
    frame['oracle_union6']=np.nan
    for pdb in ids:
        group=features[features.pdb_id.eq(pdb)]
        global_row=group[group.candidate.eq('global')].iloc[0]
        if Path(global_row.pose_path).resolve()!=Path(audit.loc[pdb,'global_pose']).resolve():
            raise ValueError(f'{pdb}: baseline and selected Global poses differ')
        rmsds={'global':frame.loc[pdb,'global']}
        for _,row in group[group.candidate.ne('global')].iterrows():
            if row.candidate not in branch.index:
                raise ValueError(f'Missing evaluation for selected candidate {row.candidate}')
            metric=branch.loc[row.candidate]
            if str(metric.candidate_id)!=str(row.candidate_id) or str(metric.parent_complex_name).upper()!=pdb or int(metric.global_rank)!=int(row.global_rank):
                raise ValueError(f'Branch evaluation identity mismatch: {row.candidate}')
            rmsds[row.candidate]=numeric(pd.Series([metric.selected_pose_rmsd_A])).iloc[0]
        rank1=group[group.global_rank.eq(1)]
        r1=rmsds[rank1.iloc[0].candidate] if len(rank1) else np.nan
        if not np.allclose(r1,frame.loc[pdb,'rank1'],equal_nan=True):
            raise ValueError(f'{pdb}: rank1 branch results differ from baseline')
        complete=len(group)==6
        local_values=[rmsds[row.candidate] for _,row in group[group.candidate.ne('global')].iterrows()]
        old_complete=complete and all(np.isfinite(value) for value in local_values)
        expected_oracle=min(local_values) if old_complete else np.nan
        if not np.allclose(expected_oracle,frame.loc[pdb,'oracle_best5'],equal_nan=True):
            raise ValueError(f'{pdb}: Local candidate oracle differs from baseline; inspect metric version/completeness')
        old_basin=np.nan
        if old_complete:
            b=group[group.candidate.ne('global')].sort_values(['basin_q25_ncc','global_rank','candidate'],ascending=[False,True,True],kind='stable').iloc[0]
            old_basin=rmsds[b.candidate]
        if not np.allclose(old_basin,frame.loc[pdb,'basin_q25_ncc'],equal_nan=True):
            raise ValueError(f'{pdb}: Reproduced basin selector disagrees with baseline')
        available_values=[v for v in rmsds.values() if np.isfinite(v)]
        # Match primary selectors: incomplete parents retain Global only.
        frame.loc[pdb,'oracle_union6']=min(available_values) if complete and available_values else rmsds['global']
        for _,selection in selections[selections.pdb_id.eq(pdb)].iterrows():
            frame.loc[pdb,selection.rule]=rmsds[selection.selected_candidate]
    summary,paired,transitions=summarize(frame,args.bootstrap_repeats,args.seed,
        oracle_methods=('oracle_best5','oracle_union6'),reference_methods=('global','rank1','basin_q25_ncc'))
    frame.to_csv(args.output_dir/'per_parent_rmsd.csv')
    summary.to_csv(args.output_dir/'success_summary.csv',index=False)
    paired.to_csv(args.output_dir/'paired_comparisons.csv',index=False)
    transitions.to_csv(args.output_dir/'per_parent_transitions.csv',index=False)
    policy_audit=[]
    for rule in RULES:
        choice=selections[selections.rule.eq(rule)].set_index('pdb_id').reindex(ids)
        complete=choice.complete_top5.astype(str).str.lower().eq('true')
        for cutoff in (2.,5.):
            for label,mask in [('complete_top5',complete),('incomplete_global_fallback',~complete)]:
                delta=frame[rule].lt(cutoff).astype(int)-frame.basin_q25_ncc.lt(cutoff).astype(int)
                policy_audit.append(dict(rule=rule,cutoff_A=cutoff,availability_group=label,group_n=int(mask.sum()),
                    selected_global_n=int((choice.selected_candidate.eq('global') & mask).sum()),
                    success_count=int((frame[rule].lt(cutoff)&mask).sum()),net_success_delta_vs_old_basin=int(delta[mask].sum()),
                    contribution_pp_fixed_denominator=float(100*delta[mask].sum()/len(ids))))
    pd.DataFrame(policy_audit).to_csv(args.output_dir/'availability_effects.csv',index=False)
    (args.output_dir/'REPORT.md').write_text(
        '# Global-preserving cascade experiment\n\nExploratory; fixed cohort denominator; no test-set parameter fitting.\n\n'
        '```text\n'+summary.to_string(index=False)+'\n```\n\n'
        'Paired changes from the existing basin selector:\n\n```text\n'+
        paired[(paired.reference=='basin_q25_ncc') & paired.method.isin(RULES)].to_string(index=False)+'\n```\n\n'
        'See availability_effects.csv to separate incomplete-case fallback from complete-case ranking changes.\n'
        'Oracle endpoints are retrospective coverage only. All variants must be reported, not only the best test result.\n')
    print(summary.to_string(index=False),flush=True)
    print(paired[paired.method.isin(RULES)].to_string(index=False),flush=True)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort','blind-manifest','selected-root','baseline-dir','branch-table','output-dir'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--selection-only',action='store_true')
    p.add_argument('--evaluate-only',action='store_true',help='Evaluate previously sealed selections without rescoring')
    p.add_argument('--materialize-selected',action='store_true',help='Copy selected SDFs into separate per-rule result directories')
    p.add_argument('--expected-n',type=int,default=174)
    p.add_argument('--sigmas',default='1.0,1.5')
    p.add_argument('--mask-radius',type=float,default=10.)
    p.add_argument('--bootstrap-repeats',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20260908)
    args=p.parse_args(argv)
    if sum([args.preflight_only,args.selection_only,args.evaluate_only])>1:
        raise ValueError('Choose only one stage-only mode')
    ids=[s.split()[0].upper() for s in args.cohort.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    sigmas=[float(s) for s in args.sigmas.split(',')]
    if len(ids)!=args.expected_n or len(set(ids))!=len(ids) or not sigmas or not all(np.isfinite(s) and s>0 for s in sigmas) or not np.isfinite(args.mask_radius) or args.mask_radius<=0 or args.bootstrap_repeats<1:
        raise ValueError('Invalid cohort/scoring parameters')
    if args.evaluate_only:
        metadata=args.output_dir/'experiment_manifest.json'
        meta=json.loads(metadata.read_text())
        if meta.get('status') not in ('selection_sealed','evaluated'):
            raise ValueError('No sealed selection stage to evaluate')
        for file,key in [('selected_poses_blind.csv','selection_sha256'),('candidate_features_blind.csv','features_sha256')]:
            if sha(args.output_dir/file)!=meta[key]:
                raise ValueError(f'Sealed file changed: {file}')
        if meta['inputs'].get(str(args.cohort.resolve()))!=sha(args.cohort):
            raise ValueError('Cohort differs from sealed selection')
        evaluate(args,ids,pd.read_csv(args.output_dir/'candidate_features_blind.csv'),pd.read_csv(args.output_dir/'selected_poses_blind.csv'))
        meta.update(status='evaluated',evaluation_inputs={str(path.resolve()):sha(path) for path in (args.branch_table,args.baseline_dir/'per_parent_rmsd.csv',args.baseline_dir/'global_top1_audit.csv')})
        metadata.write_text(json.dumps(meta,indent=2)+'\n')
        print(f'CASCADE_UPGRADE_EVALUATION_OK: {args.output_dir}',flush=True)
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use an empty experiment output directory; originals are never overwritten')
    if not args.selection_only:
        for path in (args.branch_table,args.baseline_dir/'per_parent_rmsd.csv',args.baseline_dir/'global_top1_audit.csv'):
            if not path.is_file():
                raise FileNotFoundError(path)
    print('Preflight: checking all crop/pose paths, schemas, molecules and identities; no RMSD read.',flush=True)
    plans,inventory=collect(args,ids)
    complete=sum(len(candidates)==6 for crops,candidates in plans.values())
    print(f'PREFLIGHT_OK: parents={len(ids)}, complete_top5={complete}, global_fallback={len(ids)-complete}',flush=True)
    if args.preflight_only:
        return 0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    meta=dict(status='selection_running',protocol='exploratory_not_validation_frozen',
        selection_uses_reference=False,selection_uses_rmsd=False,cohort_n=len(ids),sigmas_A=sigmas,mask_radius_A=args.mask_radius,
        shared_target='Concatenated five predicted-centre crop masks; repeated overlap retained identically for all candidates',
        incomplete_policy='Global Top1 fallback for all new selectors; no available-branch substitution',
        rules=list(RULES),seed=args.seed,bootstrap_repeats=args.bootstrap_repeats,
        inputs={str(path.resolve()):sha(path) for path in (args.cohort,args.blind_manifest)},code_sha256=sha(__file__))
    metadata=args.output_dir/'experiment_manifest.json'
    metadata.write_text(json.dumps(meta,indent=2)+'\n')
    pd.DataFrame(inventory).to_csv(args.output_dir/'input_file_hashes.csv',index=False)
    all_features,selected=[],[]
    columns=[f'shared_ncc_{i}' for i in range(len(sigmas))]
    started=time.monotonic()
    for i,pdb in enumerate(ids):
        crops,candidates=plans[pdb]
        scored=score_parent(crops,candidates,sigmas,args.mask_radius)
        scored['pdb_id']=pdb
        all_features.append(scored)
        for rule,(candidate,reason) in choose(scored,columns,len(candidates)==6).items():
            pose=scored[scored.candidate.eq(candidate)].iloc[0]
            selected.append(dict(pdb_id=pdb,rule=rule,selected_candidate=candidate,pose_path=pose.pose_path,reason=reason,complete_top5=len(candidates)==6))
        if i%5==0 or i+1==len(ids):
            print(f'SCORED {i+1}/{len(ids)} {pdb}',flush=True)
    features=pd.concat(all_features,ignore_index=True)
    choices=pd.DataFrame(selected)
    features.to_csv(args.output_dir/'candidate_features_blind.csv',index=False)
    choices.to_csv(args.output_dir/'selected_poses_blind.csv',index=False)
    meta.update(status='selection_sealed',selection_seconds=time.monotonic()-started,
        selection_sha256=sha(args.output_dir/'selected_poses_blind.csv'),features_sha256=sha(args.output_dir/'candidate_features_blind.csv'))
    metadata.write_text(json.dumps(meta,indent=2)+'\n')
    if args.materialize_selected:
        for _,row in choices.iterrows():
            destination=args.output_dir/'selected_results'/row.rule/row.pdb_id
            destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(row.pose_path,destination/'rank1.sdf')
    if not args.selection_only:
        evaluate(args,ids,features,choices)
        meta.update(status='evaluated',evaluation_inputs={str(path.resolve()):sha(path) for path in (args.branch_table,args.baseline_dir/'per_parent_rmsd.csv',args.baseline_dir/'global_top1_audit.csv')})
        metadata.write_text(json.dumps(meta,indent=2)+'\n')
    print(f'CASCADE_UPGRADE_OK: {args.output_dir}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
