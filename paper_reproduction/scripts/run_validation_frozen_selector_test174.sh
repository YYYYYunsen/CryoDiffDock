#!/usr/bin/env bash
set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }

STAGE=${1:-status}
PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
CASCADE_ROOT=${TEST174_CASCADE_ROOT:-$PROJECT_ROOT/artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1}
METRICS_ROOT=${TEST174_METRICS_ROOT:-$CASCADE_ROOT/05_metrics_druglike174_v1}
COHORT=${TEST174_COHORT:-$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
FEATURE_ROOT=${TEST174_FEATURE_ROOT:-$METRICS_ROOT/global_preserving_selection_exploratory_v1}
FROZEN_SELECTOR=${TEST174_FROZEN_SELECTOR:-$PROJECT_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_full_10xA6000_v3/analysis_val187_v1/07_tuning/frozen_selector.json}
GLOBAL_BASELINE=${TEST174_GLOBAL_BASELINE:-$METRICS_ROOT/global_vs_cascade174_hfix_v1}
BRANCH_TABLE=${TEST174_BRANCH_TABLE:-$METRICS_ROOT/cross_branch_selector_v2/cross_branch_input_table.csv}
OUTPUT_ROOT=${TEST174_FROZEN_OUTPUT:-$METRICS_ROOT/validation_frozen_selector_test174_v1}
PYTHON_BIN=${TEST174_PYTHON:-python}

cd "$PROJECT_ROOT"
case "$STAGE" in
  preflight)
    "$PYTHON_BIN" -u evaluation/apply_frozen_cascade_selector.py \
      --cohort "$COHORT" --expected-n 174 \
      --candidate-features "$FEATURE_ROOT/candidate_features_blind.csv" \
      --feature-seal "$FEATURE_ROOT/experiment_manifest.json" \
      --frozen-selector "$FROZEN_SELECTOR" \
      --output-dir "$OUTPUT_ROOT" --preflight-only
    ;;
  seal)
    [[ ! -e $OUTPUT_ROOT ]] || fail "test frozen-selector output already exists: $OUTPUT_ROOT"
    "$PYTHON_BIN" -u evaluation/apply_frozen_cascade_selector.py \
      --cohort "$COHORT" --expected-n 174 \
      --candidate-features "$FEATURE_ROOT/candidate_features_blind.csv" \
      --feature-seal "$FEATURE_ROOT/experiment_manifest.json" \
      --frozen-selector "$FROZEN_SELECTOR" \
      --output-dir "$OUTPUT_ROOT" --materialize-selected
    touch "$OUTPUT_ROOT/.stage_test_selection_sealed"
    ;;
  evaluate)
    [[ -f $OUTPUT_ROOT/.stage_test_selection_sealed ]] || fail "run seal first"
    [[ ! -e $OUTPUT_ROOT/evaluation ]] || fail "test evaluation already exists"
    "$PYTHON_BIN" -u evaluation/evaluate_frozen_cascade_selector.py \
      --cohort "$COHORT" --expected-n 174 \
      --selected-poses "$OUTPUT_ROOT/selected_poses_blind.csv" \
      --selection-manifest "$OUTPUT_ROOT/selection_manifest.json" \
      --candidate-features "$FEATURE_ROOT/candidate_features_blind.csv" \
      --global-audit "$GLOBAL_BASELINE/global_top1_audit.csv" \
      --branch-table "$BRANCH_TABLE" \
      --output-dir "$OUTPUT_ROOT/evaluation"
    touch "$OUTPUT_ROOT/.stage_test_evaluation_complete"
    ;;
  status)
    echo "FROZEN_SELECTOR=$FROZEN_SELECTOR"
    echo "OUTPUT_ROOT=$OUTPUT_ROOT"
    [[ -f $OUTPUT_ROOT/.stage_test_selection_sealed ]] && echo "seal=complete" || echo "seal=pending"
    [[ -f $OUTPUT_ROOT/.stage_test_evaluation_complete ]] && echo "evaluate=complete" || echo "evaluate=pending"
    ;;
  *) fail "usage: $0 {preflight|seal|evaluate|status}" ;;
esac
