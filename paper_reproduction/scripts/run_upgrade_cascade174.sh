#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
CASCADE_ROOT=${CASCADE_ROOT:-$PROJECT_ROOT/artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1}
METRICS_ROOT=${METRICS_ROOT:-$CASCADE_ROOT/05_metrics_druglike174_v1}
BASELINE_DIR=${BASELINE_DIR:-$METRICS_ROOT/global_vs_cascade174_hfix_v1}
OUTPUT_DIR=${OUTPUT_DIR:-$METRICS_ROOT/global_preserving_selection_exploratory_v1}
PYTHON_BIN=${PYTHON_BIN:-python}

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" evaluation/upgrade_cascade_selection.py \
  --cohort "$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt" \
  --blind-manifest "$CASCADE_ROOT/01_prepared/inference_blind.csv" \
  --selected-root "$CASCADE_ROOT/04_v2_search_druglike174_v1/merged/v2_selected" \
  --baseline-dir "$BASELINE_DIR" \
  --branch-table "$METRICS_ROOT/cross_branch_selector_v2/cross_branch_input_table.csv" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
