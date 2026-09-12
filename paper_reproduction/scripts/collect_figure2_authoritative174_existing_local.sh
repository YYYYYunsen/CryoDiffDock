#!/usr/bin/env bash
# Collect completed Local V2 outputs; this script never runs inference/search.

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
RUN_ROOT=${RUN_ROOT:-$POCKET_ROOT/artifacts/figure2_authoritative174_v1}
V1_ROOT=${V1_ROOT:-$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
AUTH_IDS=$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt
OUTPUT=$RUN_ROOT/03_existing_local174
PYTHON_BIN=$ENV_BIN/python

# Colon-separated lists, highest-priority source first. Override these if the
# completed 174 run lives under another root.
LOCAL_V2_AUDITS=${LOCAL_V2_AUDITS:-$POCKET_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1/per_complex_audit.csv:$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1/12_formal190_completion_v1/07_v2_search_valid11_v1/per_complex_audit.csv:$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1/13_formal190_v2_completion9_v1/v2_search/per_complex_audit.csv}
LOCAL_V2_METRICS=${LOCAL_V2_METRICS:-$POCKET_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1/metrics_v2_selected/per_complex.csv:$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1/12_formal190_completion_v1/07_v2_search_valid11_v1/metrics_v2_selected/per_complex.csv:$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1/13_formal190_v2_completion9_v1/v2_search/metrics_v2_selected/per_complex.csv}
LOCAL_MANIFESTS=${LOCAL_MANIFESTS:-$V1_ROOT/02_prepared/inference_blind.csv}
LOCAL_POSE_SCORES=${LOCAL_POSE_SCORES:-$V1_ROOT/04_ncc/pose_scores.csv}

args=(
  --cohort "$AUTH_IDS"
  --output-dir "$OUTPUT"
  --expected-n 174
)
IFS=: read -r -a manifest_paths <<< "$LOCAL_MANIFESTS"
IFS=: read -r -a score_paths <<< "$LOCAL_POSE_SCORES"
IFS=: read -r -a audit_paths <<< "$LOCAL_V2_AUDITS"
IFS=: read -r -a metric_paths <<< "$LOCAL_V2_METRICS"
for path in "${manifest_paths[@]}"; do args+=(--manifest "$path"); done
for path in "${score_paths[@]}"; do args+=(--pose-scores "$path"); done
for path in "${audit_paths[@]}"; do args+=(--v2-audit "$path"); done
for path in "${metric_paths[@]}"; do args+=(--v2-metrics "$path"); done

cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"
run "$PYTHON_BIN" scripts/collect_figure2_existing_local.py "${args[@]}"
