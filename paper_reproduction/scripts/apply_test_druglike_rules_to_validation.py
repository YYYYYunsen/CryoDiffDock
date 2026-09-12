#!/usr/bin/env python3
"""Transfer the frozen test CCD exclusion set to val, with no outcome inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(chemistry, decisions):
    chemistry=chemistry.copy().fillna('')
    chemistry['pdb_id']=chemistry.pdb_id.str.strip().str.upper()
    if chemistry.pdb_id.duplicated().any():
        raise ValueError('Duplicate chemistry IDs')
    if not decisions.decision.eq('exclude').all():
        raise ValueError('Expected exclusion-only frozen test decisions')
    if decisions.reviewed_ccd_id.isna().any():
        raise ValueError('Missing reviewed CCD')
    excluded=set(decisions.reviewed_ccd_id.str.strip().str.upper())
    test=chemistry[chemistry.split_membership.eq('em_test') & chemistry.rcsb_class.eq('drug_like')]
    removed=test[test.rcsb_matched_ccd_id.str.upper().isin(excluded)]
    if set(removed.pdb_id)!=set(decisions.pdb_id.str.upper()):
        raise ValueError('Frozen CCD rules do not reproduce test exclusion membership')
    table=chemistry[chemistry.split_membership.eq('em_val') & chemistry.rcsb_class.eq('drug_like')].copy()
    if table.rcsb_matched_ccd_id.str.strip().eq('').any():
        raise ValueError('Validation ligand lacks matched CCD identity')
    table['review_decision']='include'
    table['review_reason']='Original reviewed drug_like label; CCD not in frozen test exclusion set'
    table['matches_previous_excluded_ccd']=table.rcsb_matched_ccd_id.str.upper().isin(excluded)
    by_ccd=decisions.drop_duplicates('reviewed_ccd_id').set_index('reviewed_ccd_id')
    for index,row in table[table.matches_previous_excluded_ccd].iterrows():
        decision=by_ccd.loc[row.rcsb_matched_ccd_id.upper()]
        table.loc[index,'review_decision']='exclude'
        table.loc[index,'review_reason']=decision.rationale
    return table.sort_values('pdb_id'), test[~test.pdb_id.isin(removed.pdb_id)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--test-ids',required=True,type=Path)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    root=Path(config['output_root'])
    if (root/'protocol_lock.json').exists():
        raise ValueError('Protocol already locked; do not change membership')
    chemistry=pd.read_csv(config['chemistry_manifest'],dtype=str)
    decisions=pd.read_csv(config['test_chemistry_decisions'],dtype=str)
    table,test=select(chemistry,decisions)
    test_ids=set(args.test_ids.read_text().split())
    if set(test.pdb_id)!=test_ids or len(test_ids)!=174:
        raise ValueError('Rules do not reproduce authoritative174 test IDs exactly')
    val_ids=set(Path(config['val_split']).read_text().split())
    if set(chemistry.loc[chemistry.split_membership.eq('em_val'),'pdb_id'])!=val_ids:
        raise ValueError('Chemistry val membership differs from val split')
    for key in ['train_split','test_split']:
        if val_ids & set(Path(config[key]).read_text().split()):
            raise ValueError(f'Val overlaps {key}')
    out=root/'00_cohort_review'; out.mkdir(parents=True,exist_ok=True)
    review=out/'druglike_review.csv'
    if review.exists():
        old=pd.read_csv(review,dtype=str).fillna('')
        if set(old.pdb_id)!=set(table.pdb_id):
            raise ValueError('Existing draft membership differs')
        if old.review_decision.str.strip().ne('').any():
            same=old.set_index('pdb_id').review_decision.sort_index().equals(table.set_index('pdb_id').review_decision.sort_index())
            if not same:
                raise ValueError('Existing decisions conflict; preserving them')
        backup=out/'druglike_review_before_rule_transfer.csv'
        if not backup.exists():
            backup.write_bytes(review.read_bytes())
    table.to_csv(review,index=False)
    keep=table[table.review_decision.eq('include')]
    excluded=table[table.review_decision.eq('exclude')]
    keep.to_csv(out/'validation_druglike_cohort.csv',index=False)
    excluded.to_csv(out/'validation_chemistry_exclusions.csv',index=False)
    (out/'validation_druglike_ids.txt').write_text(''.join(s+'\n' for s in keep.pdb_id))
    hashes={k:sha(config[k]) for k in ['chemistry_manifest','val_split','train_split','test_split','test_chemistry_decisions']}
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    summary=dict(initial_druglike_n=len(table),excluded_n=len(excluded),retained_n=len(keep),
        rule='Reviewed rcsb_class=drug_like minus exactly the frozen test CCD exclusion set',
        excluded_ccds=sorted(set(decisions.reviewed_ccd_id)),test174_reproduced=True,
        uses_outcome_data=False,scope='CCD-rule transfer; no new category-level curation or exclusions',
        input_hashes=hashes,test_ids_sha256=sha(args.test_ids),builder_sha256=sha(__file__))
    (out/'rule_transfer_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(excluded[['pdb_id','rcsb_matched_ccd_id','review_reason']].to_string(index=False))
    print(f'VAL_CHEMISTRY_RULES_OK: {len(table)} - {len(excluded)} = {len(keep)}; test174 reproduced')
    print(f'REVIEW: {review}\nIDS: {out / "validation_druglike_ids.txt"}')


if __name__=='__main__':
    main()
