#!/usr/bin/env python3
"""Staged validation cascade runner. No test174 defaults or silent exclusions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def ids(path):
    values=[s.split()[0].upper() for s in Path(path).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if not values or len(values)!=len(set(values)):
        raise ValueError(f'Empty or duplicate IDs: {path}')
    return values


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_ids(config):
    cohort=ids(config['cohort'])
    if not set(cohort)<=set(ids(config['val_split'])):
        raise ValueError('Cohort is not a subset of validation')
    for key in ('train_split','test_split'):
        if set(cohort)&set(ids(config[key])):
            raise ValueError(f'Validation cohort overlaps {key}')
    return cohort


def run(script,*args,env=None):
    path=PROJECT/script
    command=([sys.executable] if path.suffix=='.py' else ['bash'])+[str(path),*map(str,args)]
    print('RUN:', ' '.join(command),flush=True)
    subprocess.run(command,cwd=PROJECT,env={**os.environ,**(env or {})},check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['draft-cohort','lock','global','prepare','local','merge','ncc','v2','select','evaluate'])
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    c=json.loads(args.config.read_text())
    root=Path(c['output_root'])
    if args.stage=='draft-cohort':
        val=ids(c['val_split'])
        for split in ('train_split','test_split'):
            if set(val)&set(ids(c[split])):
                raise ValueError(f'Val overlaps {split}')
        chemistry=pd.read_csv(c['chemistry_manifest'],dtype=str)
        chemistry['pdb_id']=chemistry.pdb_id.str.strip().str.upper()
        if chemistry.pdb_id.duplicated().any():
            raise ValueError('Duplicate chemistry IDs')
        table=chemistry.set_index('pdb_id').reindex(val)
        if table.rcsb_class.isna().any():
            raise ValueError('Missing validation chemistry labels')
        if not table.split_membership.eq('em_val').all():
            raise ValueError('Chemistry split and em_val.txt disagree')
        draft=table[table.rcsb_class.eq('drug_like')].copy()
        exclusions=pd.read_csv(c['test_chemistry_decisions'],dtype=str)
        excluded_ccd=set(exclusions.reviewed_ccd_id.str.upper())
        draft['matches_previous_excluded_ccd']=draft.rcsb_matched_ccd_id.fillna('').str.upper().isin(excluded_ccd)
        draft['review_decision']=''
        draft['review_reason']=''
        out=root/'00_cohort_review'
        out.mkdir(parents=True,exist_ok=True)
        if (out/'druglike_review.csv').exists():
            raise ValueError('Review already exists; do not overwrite decisions')
        draft.to_csv(out/'druglike_review.csv')
        (out/'source_hashes.json').write_text(json.dumps({k:digest(c[k]) for k in ('chemistry_manifest','val_split','train_split','test_split','test_chemistry_decisions')},indent=2))
        print(f'DRAFT_ONLY: {len(val)} val, {len(draft)} initially drug_like; review {out}/druglike_review.csv')
        return
    if args.stage=='lock':
        review=pd.read_csv(root/'00_cohort_review/druglike_review.csv',dtype=str).fillna('')
        if review.pdb_id.duplicated().any() or not review.review_decision.isin(['include','exclude']).all():
            raise ValueError('Review each row: review_decision must be include or exclude')
        if review.loc[review.review_decision.eq('exclude'),'review_reason'].str.strip().eq('').any():
            raise ValueError('Chemical exclusions require reasons, never performance-based exclusions')
        selected=review.loc[review.review_decision.eq('include'),'pdb_id'].str.upper().tolist()
        if not selected:
            raise ValueError('Empty reviewed cohort')
        source_hashes=json.loads((root/'00_cohort_review/source_hashes.json').read_text())
        if any(digest(c[k])!=v for k,v in source_hashes.items()):
            raise ValueError('Cohort review source files changed')
        chemistry=pd.read_csv(c['chemistry_manifest'],dtype=str)
        expected=set(chemistry.loc[chemistry.split_membership.eq('em_val') & chemistry.rcsb_class.eq('drug_like'),'pdb_id'].str.upper())
        if set(review.pdb_id.str.upper())!=expected:
            raise ValueError('Review must preserve all initial drug-like rows')
        out=Path(c['cohort']); out.parent.mkdir(parents=True,exist_ok=True)
        if not Path(c['global_adapter']).is_file():
            raise ValueError('Before lock, provide the verified frozen Global adapter')
        if out.exists() or (root/'protocol_lock.json').exists():
            raise ValueError('Already frozen; use a new version instead of overwriting')
        out.write_text(''.join(s+'\n' for s in selected))
        checked_ids(c)
        code=['scripts/run_validation_cascade.py','scripts/global_density_frozen_blind.py','scripts/run_cascade_4gpu.sh','scripts/run_cascade_v2_4gpu.sh','scripts/run_frozen_v2_full.sh','evaluation/upgrade_cascade_selection.py','cryodiffdock_pocket/adaptive_search.py']
        lock={'config':c,'config_sha256':digest(args.config),'cohort_n':len(selected),'cohort_sha256':digest(out),
              'primary_comparison':'union_rank_fusion_guard vs global, RMSD <2A',
              'validation_role':'cascade replication; not assumed unused by Global/Local tuning',
              'code_hashes':{s:digest(PROJECT/s) for s in code},'chemistry_sources':source_hashes,
              'global_adapter_sha256':digest(c['global_adapter'])}
        (root/'protocol_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
        print(f'COHORT_FROZEN: N={len(selected)}, branches={5*len(selected)}, raw poses={400*len(selected)}')
        return
    cohort=checked_ids(c); n=len(cohort)
    lock=json.loads((root/'protocol_lock.json').read_text())
    if lock['config_sha256']!=digest(args.config) or lock['cohort_sha256']!=digest(c['cohort']):
        raise ValueError('Frozen config/cohort changed')
    if digest(c['global_adapter'])!=lock['global_adapter_sha256']:
        raise ValueError('Frozen Global adapter changed')
    for script,checksum in lock['code_hashes'].items():
        if digest(PROJECT/script)!=checksum:
            raise ValueError(f'Frozen code changed: {script}')
    stage=root/f'.stage_{args.stage}_complete'
    if stage.exists():
        raise ValueError(f'Stage already completed: {args.stage}')
    source=root/'00_source'; prepared=root/'01_prepared'; inference=root/'02_inference'
    available=inference/'available'; ncc=root/'03_ncc'; v2=root/'04_v2'; metrics=root/'05_metrics'
    env={'PROJECT_ROOT':str(PROJECT),'NUM_GPUS':str(c.get('num_gpus',4)), 'ENV_BIN':str(Path(sys.executable).parent)}
    if args.stage=='global':
        # Adapter must be the exact test-equivalent generation/confidence/
        # frozen residual-shape ranking protocol, adapted to VAL_COHORT.
        adapter=Path(c['global_adapter'])
        if not adapter.is_file():
            raise ValueError(f'Configure verified Global generation+ranking adapter first: {adapter}')
        subprocess.run(['bash',str(adapter)],check=True,env={**os.environ,'VAL_COHORT':str(c['cohort']),
            'VAL_GLOBAL_OUTPUT':str(root/'global'),'VAL_GLOBAL_SCORES':str(c['global_scores'])})
        if not Path(c['global_scores']).is_file():
            raise ValueError('Global adapter did not produce global_scores')
    elif args.stage=='prepare':
        scores=pd.read_csv(c['global_scores'])
        key=next(k for k in ('complex','complex_name','pdb_id','complex_key') if k in scores)
        counts=scores.groupby(scores[key].astype(str).str.upper()).size().reindex(cohort,fill_value=0)
        if not counts.eq(20).all():
            raise ValueError(f'Canonical Top20 incomplete: {counts[counts.ne(20)].to_dict()}; keep cohort fixed, audit source failures')
        inputs=pd.read_csv(c['source_manifest'])
        key=next(k for k in ('complex_name','pdb_id') if k in inputs)
        inputs['_id']=inputs[key].astype(str).str.upper()
        if inputs._id.duplicated().any() or not set(cohort)<=set(inputs._id):
            raise ValueError('Full source manifest lacks unique inputs for cohort')
        for column in ('reference_ligand','experimental_protein','ligand','map_path','mask_structure'):
            if column not in inputs or inputs.set_index('_id').loc[cohort,column].isna().any():
                raise ValueError(f'Missing source column/values: {column}')
        run('scripts/build_cascade_predicted_center_manifest.py','--global-scores',c['global_scores'],'--source-manifest',c['source_manifest'],
            '--cohort',c['cohort'],'--output-dir',source,'--top-k',5,'--score-column','combined_at_best_weights','--seed',20260813,'--flexible-distance',3.5,'--no-flexible-policy','rigid','--fail-on-error')
        run('scripts/prepare_predicted_center_density_dataset.py','--manifest',source/'cascade_source_blind.csv','--output-dir',prepared,'--box-size',28,'--spacing',.75,'--mask-inner',1.5,'--mask-outer',2.5,'--fail-on-error')
        blind=pd.read_csv(prepared/'inference_blind.csv')
        reveal=pd.read_csv(source/'cascade_source_reveal.csv')
        if 'reference_ligand' in blind or len(blind)!=5*n:
            raise ValueError('Blind schema/count failed')
        evaluated=blind.merge(reveal[['complex_name','reference_ligand']],on='complex_name',validate='one_to_one')
        if evaluated.reference_ligand.isna().any():
            raise ValueError('Missing evaluation references')
        evaluated.to_csv(prepared/'evaluation_manifest.csv',index=False)
    elif args.stage=='local':
        run('scripts/run_cascade_4gpu.sh',env={**env,'MANIFEST':str(prepared/'inference_blind.csv'),'OUTPUT_ROOT':str(inference)})
    elif args.stage=='merge':
        run('scripts/merge_cascade_sharded_inference_available.py','--full-manifest',prepared/'inference_blind.csv','--evaluation-manifest',prepared/'evaluation_manifest.csv',
            '--shard-root',inference/'shards','--parent-cohort',c['cohort'],'--output-root',available,'--num-shards',c.get('num_gpus',4),'--expected-parents',n,'--validate-sdf')
    elif args.stage=='ncc':
        run('scripts/rerank_ligand_by_density_ncc.py','--manifest',available/'inference_blind_available.csv','--candidate-root',f'baseline={available}/baseline','--candidate-root',f'guided={available}/guided',
            '--output-dir',ncc,'--num-predictions',40,'--sigmas',1.5,'--mask-radius',10,'--ranking-sigma',1.5,'--density-weight',2,'--materialize-top-k',1,'--fail-on-error')
    elif args.stage=='v2':
        run('scripts/run_cascade_v2_4gpu.sh',env={**env,'MANIFEST':str(available/'inference_blind_available.csv'),'POSE_SCORES':str(ncc/'pose_scores.csv'),'OUTPUT_ROOT':str(v2)})
    elif args.stage=='select':
        run('evaluation/upgrade_cascade_selection.py','--cohort',c['cohort'],'--expected-n',n,'--blind-manifest',prepared/'inference_blind.csv','--selected-root',v2/'merged/v2_selected',
            '--baseline-dir',metrics/'global_comparison','--branch-table',metrics/'cross/cross_branch_input_table.csv','--output-dir',root/'06_selection','--selection-only','--materialize-selected')
    elif args.stage=='evaluate':
        if not (root/'06_selection/selected_poses_blind.csv').exists():
            raise ValueError('Run select before revealing RMSD')
        run('evaluation/evaluate_ligand_accuracy.py','--manifest',available/'evaluation_manifest_available.csv','--results',v2/'merged/v2_selected','--condition','cascade_v2','--num-predictions',1,'--output-dir',metrics/'branch_v2')
        run('evaluation/summarize_cascade_branches.py','--branch-manifest',available/'inference_blind_available.csv','--branch-metrics',metrics/'branch_v2/per_complex.csv','--requested-cohort',c['cohort'],'--output-dir',metrics/'cascade_v2')
        run('evaluation/evaluate_cascade_cross_branch_selection.py','--branch-manifest',available/'inference_blind_available.csv','--branch-metrics',metrics/'branch_v2/per_complex.csv','--v2-selected-root',v2/'merged/v2_selected','--requested-cohort',c['cohort'],'--output-dir',metrics/'cross')
        run('evaluation/compare_global_cascade.py','--cohort',c['cohort'],'--expected-n',n,'--global-scores',c['global_scores'],'--source-reveal',source/'cascade_source_reveal.csv','--reference-manifest',prepared/'evaluation_manifest.csv',
            '--cascade-per-parent',metrics/'cascade_v2/cascade_per_complex.csv','--cross-per-parent',metrics/'cross/cross_branch_per_parent.csv','--cross-summary',metrics/'cross/summary.json','--output-dir',metrics/'global_comparison')
        run('evaluation/upgrade_cascade_selection.py','--cohort',c['cohort'],'--expected-n',n,'--blind-manifest',prepared/'inference_blind.csv','--selected-root',v2/'merged/v2_selected','--baseline-dir',metrics/'global_comparison','--branch-table',metrics/'cross/cross_branch_input_table.csv','--output-dir',root/'06_selection','--evaluate-only')
    stage.write_text('complete\n')


if __name__=='__main__':
    main()
