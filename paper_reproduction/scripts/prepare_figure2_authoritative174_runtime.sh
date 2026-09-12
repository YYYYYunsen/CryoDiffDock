#!/usr/bin/env bash

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
PYTHON=${PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}

FIG2=$POCKET_ROOT/artifacts/figure2_authoritative174_v1
AUTH_IDS=$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt
LOCAL=$FIG2/03_existing_local174
PREVIEW=$FIG2/04_current_preview
OUTPUT=$FIG2/06_runtime

INFERENCE_ROOT=$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1
V2_TOP1_ROOT=$FIG2/07_v2_top1_runtime_rerun/01_search
V2_TOP1_AUDIT=$V2_TOP1_ROOT/per_complex_audit.csv
V2_TOP1_SUMMARY=$V2_TOP1_ROOT/summary.json
COMPLETION_ROOT=$INFERENCE_ROOT/12_formal190_completion_v1
COMPLETION_MANIFEST=$COMPLETION_ROOT/02_prepared/combined12/inference_blind.csv
COMPLETION_LOG=$COMPLETION_ROOT/03_inference_numericfix_v3000_20260904/logs/baseline_guided.log
COMPLETION_POSE_SCORES=$COMPLETION_ROOT/04_ncc_technical_filter_v1/pose_scores_valid.csv
CHEMEM_OLD=$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1
CHEMEM_HISTORICAL=$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike60_v1
CHEMEM_NEW=$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_expansion79_v1
CHEMEM_RECOVERY=$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_pro_h2_recovery_v1
EMERALD_OLD=$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1
EMERALD_NEW=$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_v1
EMERALD_RDKIT=$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_rdkit_recovery4_v1
EMERALD_SITECENTER=$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_sitecenter_recovery_v1

PRIMARY_IDS=$OUTPUT/cryodiffdock_runtime_primary165_ids.txt
COMPLETION_IDS=$OUTPUT/cryodiffdock_runtime_completion9_ids.txt
PRIMARY_RAW=$OUTPUT/cryodiffdock_primary165_runtime_raw.csv
COMPLETION_RAW=$OUTPUT/cryodiffdock_completion9_runtime_raw.csv
V2_RAW=$OUTPUT/cryodiffdock_authoritative174_runtime_raw.csv

required=(
  "$PYTHON"
  "$AUTH_IDS"
  "$INFERENCE_ROOT/02_prepared/inference_blind.csv"
  "$INFERENCE_ROOT/03_inference/logs/baseline.log"
  "$INFERENCE_ROOT/03_inference/logs/guided.log"
  "$INFERENCE_ROOT/04_ncc/pose_scores.csv"
  "$COMPLETION_MANIFEST"
  "$COMPLETION_LOG"
  "$COMPLETION_POSE_SCORES"
  "$LOCAL/local174_v2_audit_exact.csv"
  "$LOCAL/local174_v2_metrics_exact.csv"
  "$V2_TOP1_AUDIT"
  "$V2_TOP1_SUMMARY"
  "$PREVIEW/chemem_analysis/per_complex_benchmark.csv"
  "$PREVIEW/assembled/emerald_current_174.tsv"
  "$POCKET_ROOT/scripts/extract_figure2_frozen_inference_runtime.py"
  "$POCKET_ROOT/scripts/build_figure2_authoritative174_runtime.py"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || fail "missing $path"
done

run mkdir -p "$OUTPUT"

run "$PYTHON" - "$V2_TOP1_SUMMARY" "$V2_TOP1_AUDIT" <<'PY'
import json
import pandas as pd
import sys

summary_path, audit_path = sys.argv[1:]
summary = json.load(open(summary_path))
audit = pd.read_csv(audit_path)
if summary.get("materialization_mode") != "top1":
    raise ValueError("V2 runtime rerun did not use materialization_mode=top1")
if len(audit) != 173 or audit["complex_name"].nunique() != 173:
    raise ValueError(f"Expected 173 unique V2 timing rows, found {len(audit)}")
if not audit["materialization_mode"].eq("top1").all():
    raise ValueError("V2 audit contains non-top1 materialization rows")
if pd.to_numeric(audit["deployment_runtime_seconds"], errors="coerce").isna().any():
    raise ValueError("V2 audit has missing deployment runtime")
print("V2_TOP1_EXACT_RUNTIME_INPUT_OK: 173 cases")
PY

# The authoritative cohort spans two actual CryoDiffDock inference runs:
# 165 cases in the original 289-case run and 9 repaired/prepared cases in the
# later 12-case numericfix-v3000 completion run.  Derive this split from the
# manifests and selected pose-score table, then enforce the expected 165+9=174
# coverage before extracting any runtime.
run "$PYTHON" - "$AUTH_IDS" \
  "$INFERENCE_ROOT/02_prepared/inference_blind.csv" \
  "$COMPLETION_MANIFEST" \
  "$COMPLETION_POSE_SCORES" \
  "$PRIMARY_IDS" "$COMPLETION_IDS" <<'PY'
import csv
import pathlib
import sys

auth_path, primary_manifest_path, completion_manifest_path, pose_path, primary_out, completion_out = map(pathlib.Path, sys.argv[1:])

def ids_from_text(path):
    values = [line.split("#", 1)[0].strip().upper() for line in path.read_text().splitlines()]
    return [value for value in values if value]

def ids_from_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty table: {path}")
    column = next((name for name in ("complex_name", "pdb_id", "complex_id", "id") if name in rows[0]), None)
    if column is None:
        raise ValueError(f"no ID column in {path}")
    return [str(row[column]).strip().upper() for row in rows]

auth = ids_from_text(auth_path)
primary_manifest = set(ids_from_csv(primary_manifest_path))
completion_manifest = set(ids_from_csv(completion_manifest_path))
completion_pose = set(ids_from_csv(pose_path))

if len(auth) != 174 or len(set(auth)) != 174:
    raise ValueError(f"expected 174 unique authoritative IDs, found {len(auth)}/{len(set(auth))}")

primary = [case_id for case_id in auth if case_id in primary_manifest]
completion = [case_id for case_id in auth if case_id not in primary_manifest]
expected_completion = {
    "8IO3", "8WGD", "8Y2G", "8Y90", "8YNZ",
    "9J6U", "9JEL", "9MFM", "9VSP",
}
missing_completion_manifest = [case_id for case_id in completion if case_id not in completion_manifest]
missing_completion_pose = [case_id for case_id in completion if case_id not in completion_pose]
if len(primary) != 165 or len(completion) != 9:
    raise ValueError(f"expected primary/completion split 165/9, found {len(primary)}/{len(completion)}")
if set(completion) != expected_completion:
    raise ValueError(
        "unexpected completion-nine IDs: "
        f"observed={sorted(completion)}, expected={sorted(expected_completion)}"
    )
if missing_completion_manifest or missing_completion_pose:
    raise ValueError(
        "completion coverage mismatch: "
        f"manifest_missing={missing_completion_manifest}, pose_score_missing={missing_completion_pose}"
    )

primary_out.write_text("".join(f"{case_id}\n" for case_id in primary))
completion_out.write_text("".join(f"{case_id}\n" for case_id in completion))
print(f"CryoDiffDock runtime split: original={len(primary)}, completion={len(completion)}")
print("Completion IDs: " + ",".join(completion))
PY

run "$PYTHON" "$POCKET_ROOT/scripts/extract_figure2_frozen_inference_runtime.py" \
  --manifest "$INFERENCE_ROOT/02_prepared/inference_blind.csv" \
  --frozen-ids "$PRIMARY_IDS" \
  --baseline-log "$INFERENCE_ROOT/03_inference/logs/baseline.log" \
  --guided-log "$INFERENCE_ROOT/03_inference/logs/guided.log" \
  --pose-scores "$INFERENCE_ROOT/04_ncc/pose_scores.csv" \
  --v2-audit "$V2_TOP1_AUDIT" \
  --v2-metrics "$LOCAL/local174_v2_metrics_exact.csv" \
  --run-label original_289 \
  --output "$PRIMARY_RAW"

run "$PYTHON" "$POCKET_ROOT/scripts/extract_figure2_frozen_inference_runtime.py" \
  --manifest "$COMPLETION_MANIFEST" \
  --frozen-ids "$COMPLETION_IDS" \
  --combined-baseline-guided-log "$COMPLETION_LOG" \
  --pose-scores "$COMPLETION_POSE_SCORES" \
  --v2-audit "$V2_TOP1_AUDIT" \
  --v2-metrics "$LOCAL/local174_v2_metrics_exact.csv" \
  --run-label completion_numericfix_v3000_12 \
  --output "$COMPLETION_RAW"

run "$PYTHON" - "$AUTH_IDS" "$PRIMARY_RAW" "$COMPLETION_RAW" "$V2_RAW" <<'PY'
import csv
import math
import pathlib
import sys

auth_path, primary_path, completion_path, output_path = map(pathlib.Path, sys.argv[1:])
auth = [line.split("#", 1)[0].strip().upper() for line in auth_path.read_text().splitlines()]
auth = [value for value in auth if value]

def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

combined = rows(primary_path) + rows(completion_path)
by_id = {}
for row in combined:
    case_id = row["complex_name"].strip().upper()
    if case_id in by_id:
        raise ValueError(f"duplicate CryoDiffDock runtime row: {case_id}")
    by_id[case_id] = row

missing = [case_id for case_id in auth if case_id not in by_id]
extra = sorted(set(by_id) - set(auth))
if len(combined) != 174 or missing or extra:
    raise ValueError(f"runtime merge is not exact: rows={len(combined)}, missing={missing}, extra={extra}")

output_rows = []
pose_score_missing = []
upstream_failure_ids = []
for authoritative_index, case_id in enumerate(auth):
    row = dict(by_id[case_id])
    row["run_subset_index"] = row.pop("frozen_index")
    row = {
        "frozen_index": authoritative_index,
        **row,
    }
    pose_count = int(float(row["pose_score_rows"]))
    if pose_count not in (0, 80):
        raise ValueError(
            f"expected 0 or 80 pose-score rows for {case_id}, found {row['pose_score_rows']}"
        )
    if pose_count == 0:
        pose_score_missing.append(case_id)
    if row["v2_expansion_status"] == "not_attempted_upstream_failure":
        upstream_failure_ids.append(case_id)
    elif row["v2_expansion_status"] != "completed":
        raise ValueError(
            f"missing V2 expansion/selection runtime for {case_id}; "
            "refusing to substitute inference-only time"
        )
    runtime = float(row["runtime_seconds"])
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError(f"invalid runtime for {case_id}: {row['runtime_seconds']}")
    output_rows.append(row)

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)

counts = {}
for row in output_rows:
    label = row["inference_run_label"]
    counts[label] = counts.get(label, 0) + 1
print(f"Merged CryoDiffDock runtime rows: {len(output_rows)}")
print("Runtime source counts: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
print("Pose-score-missing technical cases: " + (",".join(pose_score_missing) if pose_score_missing else "none"))
print("Upstream technical failures with inference-only observed runtime: " + (",".join(upstream_failure_ids) if upstream_failure_ids else "none"))
print(f"CRYODIFFDOCK_AUTHORITATIVE174_RUNTIME_RAW_OK: {output_path}")
PY

run "$PYTHON" "$POCKET_ROOT/scripts/build_figure2_authoritative174_runtime.py" \
  --cohort "$AUTH_IDS" \
  --chemem-results "$PREVIEW/chemem_analysis/per_complex_benchmark.csv" \
  --chemem-root "$CHEMEM_OLD" \
  --chemem-root "$CHEMEM_HISTORICAL" \
  --chemem-root "$CHEMEM_NEW" \
  --chemem-root "$CHEMEM_RECOVERY" \
  --v2-runtime "$V2_RAW" \
  --emerald-results "$PREVIEW/assembled/emerald_current_174.tsv" \
  --emerald-root "$EMERALD_OLD" \
  --emerald-root "$EMERALD_NEW" \
  --emerald-root "$EMERALD_RDKIT" \
  --emerald-root "$EMERALD_SITECENTER" \
  --output-dir "$OUTPUT" \
  --expected-n 174 \
  --timeout-cap-s 43200

echo "FIGURE2_AUTHORITATIVE174_RUNTIME_PREPARED: $OUTPUT"
