#!/usr/bin/env python3
"""Read-only validation cascade inventory; never runs inference or reads RMSD."""
import csv
import json
import os
from pathlib import Path


def describe(path):
    record={'path':str(path),'exists':path.exists()}
    if path.is_file() and path.suffix=='.csv':
        with path.open(newline='') as handle:
            reader=csv.DictReader(handle)
            record['columns']=reader.fieldnames
            split_counts={}
            count=0
            for row in reader:
                count+=1
                if 'split_membership' in row:
                    key=(row['split_membership'],row.get('rcsb_class',''))
                    split_counts[str(key)]=split_counts.get(str(key),0)+1
            record['row_count']=count
            if split_counts:
                record['split_class_counts']=split_counts
    print(json.dumps(record),flush=True)


def main():
    project=Path('/Scr/yunsenz2/data/DiffDock-Pocket-EM')
    global_root=Path('/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM')
    data=Path('/Scr/yunsenz2/data/EM-dataset-final/data')
    split=data/'splits_deposited_noncovalent_cofactor35_contact_entity_emdb_v1'
    sets={}
    for name in ('em_val.txt','em_test.txt','em_train.txt'):
        path=split/name
        describe(path)
        if path.is_file():
            ids=[s.split()[0].upper() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
            sets[name]=set(ids)
            print(json.dumps({'split':name,'n':len(ids),'unique_n':len(set(ids))}))
    for other in ('em_test.txt','em_train.txt'):
        if 'em_val.txt' in sets and other in sets:
            print(json.dumps({'overlap':['em_val.txt',other],'ids':sorted(sets['em_val.txt'] & sets[other])}))
    for path in (project/'visualizations/figure2_global/input/raw/chemistry/chemistry_manifest.csv',
                 project/'artifacts/oracle_local/full_polymer_mask_audit100/source_full_polymer100.csv',
                 data/'04_maps_fixed.csv',data/'processed_6radii_deposited',
                 data/'deposited_ligand_references_complete_global_v1'):
        describe(path)
    # Bounded inventory: don't descend into large per-pose result trees.
    prune={'shards','conf_scored','v2_selected','expanded_pool','crops','baseline','guided','.git','node_modules','__pycache__'}
    for base in (global_root/'scripts',global_root/'tools',global_root/'configs',global_root/'artifacts',project/'artifacts/oracle_local'):
        print(f'INVENTORY_ROOT {base}',flush=True)
        if not base.is_dir():
            continue
        found=0
        for current,dirs,files in os.walk(base):
            depth=len(Path(current).relative_to(base).parts)
            dirs[:]=sorted(d for d in dirs if d not in prune and not d.startswith('index')) if depth<4 else []
            for name in sorted(files):
                path=Path(current)/name
                lower=str(path.relative_to(base)).lower()
                if path.suffix in ('.sh','.py','.yaml','.yml','.json','.csv','.txt') and any(token in lower for token in ('val','residual_shape','top20','chemistry')):
                    # Print paths only: avoid looking at validation outcomes.
                    print(f'CANDIDATE {path}',flush=True)
                    found+=1
                    if found>=100:
                        break
            if found>=100:
                print('Inventory capped at 100 paths for this root.',flush=True)
                break


if __name__=='__main__':
    main()
