#!/usr/bin/env python3
"""Inspect existing input files and failures without running RMSD or docking.

Optional cleanup is confined to two explicitly named failed analysis folders.
Their contents and audit errors are saved to a temporary report before removal.
Uses only the Python standard library.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter


def rows(path):
    with path.open(newline='') as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t')
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return reader.fieldnames or [], list(reader)


def inspect_file(path):
    record = {'path': str(path), 'exists': path.is_file()}
    if not record['exists']:
        return record
    if path.suffix == '.csv':
        columns, data = rows(path)
        record.update(columns=columns, row_count=len(data))
        # Check explicit file fields only; do not infer alternate reference
        # columns or substitute prediction files for native references.
        checks = {}
        for column in ('reference_ligand', 'global_pose', 'sdf_path', 'prediction_path', 'pose_path'):
            if column not in columns:
                continue
            values = {row.get(column, '').strip() for row in data}
            missing = []
            unreadable = []
            for value in sorted(values):
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                if not value or not candidate.is_file():
                    missing.append(value)
                elif not os.access(candidate, os.R_OK):
                    unreadable.append(value)
            checks[column] = {'unique_paths': len(values), 'missing_paths': missing, 'unreadable_paths': unreadable}
        record['path_checks'] = checks
    elif path.suffix == '.json':
        record['content'] = json.loads(path.read_text())
    return record


def cleanup_failed(path):
    if not path.exists():
        return 'absent'
    if path.is_symlink() or not path.is_dir():
        return 'refused: not a regular directory'
    children = list(path.iterdir())
    if any(item.name != 'global_top1_audit.csv' or item.is_symlink() or not item.is_file() for item in children):
        return 'refused: contains files other than the failed audit'
    if children:
        columns, data = rows(children[0])
        if 'evaluation_error' not in columns or not any(row.get('evaluation_error', '').strip() for row in data):
            return 'refused: audit does not document evaluation errors'
    shutil.rmtree(path)
    return 'removed failed analysis directory'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('/Scr/yunsenz2/data/DiffDock-Pocket-EM'))
    parser.add_argument('--global-scores', type=Path, default=Path('/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM/artifacts/paper_test/global_top20_residual_shape_canonical192_v1/global_top20_residual_shape_scores_192.csv'))
    parser.add_argument('--clean-failed', action='store_true')
    args = parser.parse_args()
    root = args.project_root / 'artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1'
    metrics = root / '05_metrics_druglike174_v1'
    paths = [args.global_scores,
             root/'00_source/cascade_source_reveal.csv',
             root/'01_prepared/evaluation_manifest.csv',
             root/'02_inference/available_druglike174_v1_complete/evaluation_manifest_available.csv',
             root/'04_v2_search_druglike174_v1/merged/evaluation_manifest.csv',
             metrics/'cascade_v2/cascade_per_complex.csv',
             metrics/'cross_branch_selector_v2/cross_branch_per_parent.csv',
             metrics/'cross_branch_selector_v2/summary.json']
    report = {'mode': 'inspection_only_no_rmsd', 'inputs': [], 'failed_runs': []}
    for path in paths:
        try:
            entry = inspect_file(path)
        except Exception as exc:
            entry = {'path': str(path), 'inspection_error': f'{type(exc).__name__}: {exc}'}
        report['inputs'].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    failed_paths = [metrics/name for name in ('global_vs_cascade174_v1', 'global_vs_cascade174_reference_fix_v1')]
    for path in failed_paths:
        audit = path/'global_top1_audit.csv'
        entry = {'path': str(path), 'exists': path.exists()}
        if audit.is_file():
            # Preserve the entire audit before any optional cleanup.
            entry['audit_csv'] = audit.read_text()
            _, data = rows(audit)
            errors = Counter(row.get('evaluation_error', '') for row in data if row.get('evaluation_error', '').strip())
            entry['error_counts'] = dict(errors)
            print(f'FAILURES {path.name}: {json.dumps(dict(errors), ensure_ascii=False)}', flush=True)
        report['failed_runs'].append(entry)
    fd, report_name = tempfile.mkstemp(prefix='global_cascade_diagnostic_', suffix='.json')
    with os.fdopen(fd, 'w') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    print(f'DIAGNOSTIC_SAVED: {report_name}', flush=True)
    if args.clean_failed:
        for path, entry in zip(failed_paths, report['failed_runs']):
            entry['cleanup'] = cleanup_failed(path)
            print(f'CLEANUP {path}: {entry["cleanup"]}', flush=True)
        Path(report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
