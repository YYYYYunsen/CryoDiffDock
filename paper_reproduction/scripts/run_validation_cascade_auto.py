#!/usr/bin/env python3
"""One-case smoke or full validation automation through V2; no RMSD evaluation."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_validation_cascade import checked_ids, digest, PROJECT
from scripts.check_global_confidence_inputs import check_poses


def validate_reuse(previous, selected, source):
    previous = previous.resolve()
    old_ids = (previous/'inputs/ids.txt').read_text().split()
    if old_ids != selected:
        raise ValueError('Reuse requires the identical ordered cohort')
    old_source = pd.read_csv(previous/'inputs/source_full_polymer.csv').fillna('')
    pd.testing.assert_frame_equal(old_source, pd.DataFrame(source), check_dtype=False)
    table = pd.read_csv(previous/'global/test_input_em_ring.csv')
    if table.complex_name.tolist() != selected:
        raise ValueError('Reused Global input CSV does not match requested cohort')
    check_poses(previous/'global/finetuned', selected)
    files = [previous/'protocol_lock.json', previous/'global.log', previous/'global/test_input_em_ring.csv']
    files += [previous/'global/finetuned'/name/f'rank{rank}.sdf' for name in selected for rank in range(1,41)]
    return {'source_run':str(previous), 'file_sha256':{str(f.relative_to(previous)):digest(f) for f in files}}


def shard_ids(values, count, index):
    if count < 1 or index < 0 or index >= count:
        raise ValueError(f'Invalid job shard {index}/{count}')
    if count > len(values):
        raise ValueError(f'Job shard count {count} exceeds cohort size {len(values)}')
    return values[index::count]


def gpu_count(mode, job_shards, requested, configured):
    if requested is not None:
        if requested < 1:
            raise ValueError('--gpus-per-job must be positive')
        return requested
    if mode == 'smoke' or job_shards > 1:
        return 1
    return int(configured)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--case',help='Optional explicit reviewed validation ID; default alphabetically first')
    p.add_argument('--global-env',default='diffdock',help='Conda environment for Global generation AND confidence/density')
    p.add_argument('--raw-root',type=Path,help='Complete receptor CIF root; otherwise infer from existing full-polymer audit')
    p.add_argument('--data-root',type=Path,default=Path('/Scr/yunsenz2/data/EM-dataset-final/data'))
    p.add_argument('--output-root',type=Path)
    p.add_argument('--reuse-global-from',type=Path,help='Previous auto run root; copy its completed Global poses into a NEW output root and restart confidence')
    p.add_argument('--smoke-result',type=Path,help='Successful smoke AUTO_SUCCESS.json required for full mode')
    p.add_argument('--job-shards',type=int,default=1,help='Split the reviewed full cohort across this many independent jobs')
    p.add_argument('--job-shard-index',type=int,default=0,help='Zero-based parent-level job shard index')
    p.add_argument('--gpus-per-job',type=int,help='GPUs visible inside this job; defaults to one for smoke or parent-sharded runs')
    p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    c=json.loads(a.config.read_text())
    parent_root=Path(c['output_root'])
    if a.mode=='full':
        smoke_path=a.smoke_result or parent_root/'auto_smoke_v1/AUTO_SUCCESS.json'
        smoke=json.loads(smoke_path.read_text())
        if smoke.get('mode')!='smoke' or smoke.get('status')!='complete_through_v2' or smoke.get('selected_v2_branches')!=5:
            raise ValueError('Full run requires a successful five-branch smoke first')
    review_path=parent_root/'00_cohort_review/druglike_review.csv'
    review=pd.read_csv(review_path).fillna('')
    if review.pdb_id.duplicated().any() or not review.review_decision.isin(['include','exclude']).all():
        raise ValueError('Apply test chemistry rules first')
    all_ids=sorted(review.loc[review.review_decision.eq('include'),'pdb_id'].str.upper())
    if not all_ids:
        raise ValueError('Empty reviewed validation cohort')
    if a.case and a.case.upper() not in all_ids:
        raise ValueError('Smoke case must belong to reviewed validation drug-like cohort')
    if a.mode == 'smoke' and (a.job_shards != 1 or a.job_shard_index != 0):
        raise ValueError('Parent-level job sharding is full-mode only')
    selected=[a.case.upper() if a.case else all_ids[0]] if a.mode=='smoke' else shard_ids(all_ids,a.job_shards,a.job_shard_index)
    worker_gpus=gpu_count(a.mode,a.job_shards,a.gpus_per_job,c.get('num_gpus',4))
    if a.case and a.mode!='smoke':
        raise ValueError('--case is smoke-only')
    suffix = f'_shard{a.job_shard_index:02d}of{a.job_shards:02d}' if a.job_shards > 1 else ''
    root=a.output_root or parent_root/f'auto_{a.mode}{suffix}_v1'
    if root.exists():
        raise ValueError(f'Output exists: {root}; preserve it and use a new --output-root after diagnosing failures')
    data=a.data_root.resolve()
    processed=data/'processed_6radii_deposited'
    maps=pd.read_csv(data/'04_maps_fixed.csv').fillna('')
    valid=maps[maps.status.str.lower().isin(['ok','success','downloaded'])].copy()
    valid.pdb_id=valid.pdb_id.str.upper()
    if valid.pdb_id.duplicated().any():
        # Refuse ambiguous mapping instead of silently choosing a map.
        if valid[valid.pdb_id.isin(selected)].pdb_id.duplicated().any():
            raise ValueError('Multiple successful map rows for requested case')
    valid=valid.set_index('pdb_id')
    raw=a.raw_root
    if raw is None:
        audit=PROJECT/'artifacts/oracle_local/full_polymer_mask_audit100/source_full_polymer100.csv'
        prior=pd.read_csv(audit,usecols=['mask_structure']).dropna()
        roots={Path(value).resolve().parent.parent for value in prior.mask_structure}
        if len(roots)!=1:
            raise ValueError('Cannot infer a unique full-polymer root; set --raw-root')
        raw=roots.pop()
    source=[]
    for pdb in selected:
        # Do not fall back to another case when a file is missing.
        choices=[raw/pdb/f'{pdb}.cif',raw/pdb/f'{pdb}.mmcif',raw/pdb.lower()/f'{pdb.lower()}.cif']
        structures=[]
        for path in choices:
            if path.is_file() and not any(path.samefile(other) for other in structures):
                structures.append(path.resolve())
        if len(structures)!=1:
            raise ValueError(f'{pdb}: expected one explicit deposited CIF, found {structures}; --raw-root={raw}')
        record=dict(complex_name=pdb,experimental_protein=str(processed/pdb/f'{pdb}_protein_processed.pdb'),
                    ligand=str(processed/pdb/f'{pdb}_ligand.sdf'),mask_structure=str(structures.pop()),
                    reference_ligand=str(data/'deposited_ligand_references_complete_global_v1'/pdb/f'{pdb}_ligand_deposited.sdf'))
        if pdb not in valid.index:
            raise ValueError(f'{pdb}: no usable map row')
        map_path=Path(valid.loc[pdb,'map_path'])
        record['map_path']=str(map_path if map_path.is_absolute() else data/map_path)
        for key,value in record.items():
            if key!='complex_name' and not Path(value).is_file():
                raise FileNotFoundError(f'{pdb} {key}: {value}')
        source.append(record)
    reuse = validate_reuse(a.reuse_global_from, selected, source) if a.reuse_global_from else None
    # Preflight split identities without modifying the frozen parent cohort.
    val=set(Path(c['val_split']).read_text().split())
    if not set(selected)<=val:
        raise ValueError('Reviewed IDs are not validation IDs')
    for key in ['test_split','train_split']:
        if set(selected)&set(Path(c[key]).read_text().split()):
            raise ValueError(f'Overlap with {key}')
    adapter=PROJECT/'scripts/global_validation_frozen_adapter.sh'
    cohort_path=parent_root/'00_cohort_review/validation_druglike_ids.txt'
    if set(cohort_path.read_text().split())!=set(all_ids):
        raise ValueError('Reviewed decisions and exported ID list disagree')
    global_env={**os.environ,'VAL_COHORT':str(cohort_path),'VAL_GLOBAL_OUTPUT':str(root/'global'),
                'VAL_GLOBAL_SCORES':str(root/'global/global_top20_scores.csv'),'NUM_GPUS':str(worker_gpus),
                'CONDA_ENV':a.global_env,'DATA_ROOT':str(data), 'VAL_PREFLIGHT_IDS':' '.join(selected),
                'REUSE_GLOBAL_INFERENCE':'1' if reuse else '0'}
    if reuse:
        global_env['REUSE_GLOBAL_LOG']=str(Path(reuse['source_run'])/'global.log')
    subprocess.run(['conda','run','--no-capture-output','-n',a.global_env,'bash',str(adapter),'--preflight-only'],env=global_env,check=True)
    print(f'AUTO_PREFLIGHT_OK: mode={a.mode}, IDs={selected}, parents={len(selected)}, raw_root={raw}',flush=True)
    if a.preflight_only:
        return 0
    root.mkdir(parents=True)
    (root/'inputs').mkdir()
    if reuse:
        (root/'global').mkdir()
        prior = Path(reuse['source_run'])
        shutil.copy2(prior/'global/test_input_em_ring.csv',root/'global/test_input_em_ring.csv')
        for name in selected:
            destination = root/'global/finetuned'/name
            destination.mkdir(parents=True)
            for rank in range(1,41):
                filename = f'rank{rank}.sdf'
                shutil.copy2(prior/'global/finetuned'/name/filename,destination/filename)
                if digest(destination/filename) != reuse['file_sha256'][f'global/finetuned/{name}/{filename}']:
                    raise ValueError(f'Reused pose changed while copying: {name}/{filename}')
        (root/'global_reuse_provenance.json').write_text(json.dumps(reuse,indent=2)+'\n')
    cohort=root/'inputs/ids.txt'; cohort.write_text(''.join(s+'\n' for s in selected))
    source_file=root/'inputs/source_full_polymer.csv'; pd.DataFrame(source).to_csv(source_file,index=False)
    c.update(output_root=str(root),cohort=str(cohort),source_manifest=str(source_file),
             global_scores=str(root/'global/global_top20_scores.csv'),global_adapter=str(adapter),num_gpus=worker_gpus)
    config=root/'run_config.json'; config.write_text(json.dumps(c,indent=2)+'\n')
    checked_ids(c)
    code=['scripts/run_validation_cascade.py','scripts/run_validation_cascade_auto.py','scripts/check_global_confidence_inputs.py','scripts/global_density_frozen_blind.py','scripts/run_cascade_4gpu.sh',
          'scripts/run_cascade_v2_4gpu.sh','scripts/run_frozen_v2_full.sh','evaluation/upgrade_cascade_selection.py','cryodiffdock_pocket/adaptive_search.py']
    lock=dict(config_sha256=digest(config),cohort_sha256=digest(cohort),code_hashes={s:digest(PROJECT/s) for s in code},
              global_adapter_sha256=digest(adapter),mode=a.mode,selection='explicit ID or alphabetical first, no outcome selection',
              review_sha256=digest(review_path),source_sha256=digest(source_file),global_environment=a.global_env,
              evaluates_rmsd=False,expected_parents=len(selected),reviewed_parent_n=len(all_ids),
              job_shards=a.job_shards,job_shard_index=a.job_shard_index,
              gpus_per_job=worker_gpus,
              shard_assignment='sorted reviewed IDs, zero-based round-robin values[index::count]')
    (root/'protocol_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    global_env['VAL_COHORT']=str(cohort)
    print(f'AUTO_STAGE global; log={root / "global.log"}',flush=True)
    with (root/'global.log').open('w') as log:
        result=subprocess.run(['conda','run','--no-capture-output','-n',a.global_env,'bash',str(adapter)],env=global_env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join((root/'global.log').read_text(errors='replace').splitlines()[-80:]),file=sys.stderr)
        raise RuntimeError(f'Global/confidence/density failed; full log: {root / "global.log"}')
    (root/'.stage_global_complete').write_text('complete\n')
    for stage in ['prepare','local','merge','ncc','v2']:
        print(f'AUTO_STAGE {stage}; log={root / (stage+".log")}',flush=True)
        with (root/(stage+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(PROJECT/'scripts/run_validation_cascade.py'),stage,'--config',str(config)],
                           cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,check=True)
        if stage=='merge' and a.mode=='smoke':
            available=pd.read_csv(root/'02_inference/available/inference_blind_available.csv')
            if len(available)!=5:
                raise ValueError('Smoke requires all five Local branches; inspect merge audit')
    selected_root=root/'04_v2/merged/v2_selected'
    directories=[path for path in selected_root.iterdir() if path.is_dir()]
    available=pd.read_csv(root/'02_inference/available/inference_blind_available.csv')
    if len(directories)!=len(available):
        raise ValueError('V2 selected output count mismatch')
    for directory in directories:
        if len(list(directory.glob('rank1_confidence*.sdf')))!=1:
            raise ValueError(f'Invalid final V2 pose: {directory}')
    report=dict(mode=a.mode,parents=len(selected),requested_branches=5*len(selected),available_branches=len(available),
                selected_v2_branches=len(directories),rmsd_evaluated=False,status='complete_through_v2',
                job_shards=a.job_shards,job_shard_index=a.job_shard_index,gpus_per_job=worker_gpus,parent_ids=selected)
    (root/'AUTO_SUCCESS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'AUTO_CASCADE_OK: {root}\n{json.dumps(report)}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
