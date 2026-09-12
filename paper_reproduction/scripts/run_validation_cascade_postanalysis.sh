#!/usr/bin/env bash
# Assemble, evaluate, and tune the completed ten-shard validation cascade.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 2; }

STAGE=${1:-status}
PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
RUN_ROOT=${VALCAS_ANALYSIS_RUN_ROOT:-$PROJECT_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_full_10xA6000_v3}
COHORT=${VALCAS_ANALYSIS_COHORT:-$PROJECT_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/00_cohort_review/validation_druglike_ids.txt}
ANALYSIS_ROOT=${VALCAS_ANALYSIS_ROOT:-$RUN_ROOT/analysis_val187_v1}
PYTHON_BIN=${VALCAS_ANALYSIS_PYTHON:-python}
RMSD_WORKERS=${VALCAS_RMSD_WORKERS:-8}
EXPECTED_N=${VALCAS_EXPECTED_N:-187}
NUM_SHARDS=${VALCAS_NUM_SHARDS:-10}

[[ $EXPECTED_N =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_EXPECTED_N must be positive"
[[ $NUM_SHARDS =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_NUM_SHARDS must be positive"
[[ $RMSD_WORKERS =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_RMSD_WORKERS must be positive"
for required in "$RUN_ROOT" "$COHORT"; do
  [[ -e $required ]] || fail "missing required input: $required"
done
if [[ $PYTHON_BIN == */* ]]; then
  [[ -x $PYTHON_BIN ]] || fail "Python is not executable: $PYTHON_BIN"
else
  command -v "$PYTHON_BIN" >/dev/null || fail "Python command is unavailable: $PYTHON_BIN"
fi
cd "$PROJECT_ROOT"

assemble_args=(
  --run-root "$RUN_ROOT"
  --cohort "$COHORT"
  --output-root "$ANALYSIS_ROOT"
  --num-shards "$NUM_SHARDS"
  --expected-parents "$EXPECTED_N"
)

case "$STAGE" in
  audit)
    exec "$PYTHON_BIN" -u scripts/assemble_validation_cascade_shards.py "${assemble_args[@]}" --preflight-only
    ;;
  assemble)
    exec "$PYTHON_BIN" -u scripts/assemble_validation_cascade_shards.py "${assemble_args[@]}"
    ;;
  seal)
    [[ -f $ANALYSIS_ROOT/.stage_assemble_complete ]] || fail "run assemble first"
    [[ ! -e $ANALYSIS_ROOT/06_selection ]] || fail "selection output already exists"
    "$PYTHON_BIN" -u evaluation/upgrade_cascade_selection.py \
      --cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --expected-n "$EXPECTED_N" \
      --blind-manifest "$ANALYSIS_ROOT/01_prepared/inference_blind.csv" \
      --selected-root "$ANALYSIS_ROOT/04_v2/merged/v2_selected" \
      --baseline-dir "$ANALYSIS_ROOT/05_metrics/global_comparison" \
      --branch-table "$ANALYSIS_ROOT/05_metrics/cross/cross_branch_input_table.csv" \
      --output-dir "$ANALYSIS_ROOT/06_selection" \
      --selection-only
    touch "$ANALYSIS_ROOT/.stage_seal_complete"
    echo "VALIDATION_SELECTION_SEALED: $ANALYSIS_ROOT/06_selection/candidate_features_blind.csv"
    ;;
  evaluate)
    [[ -f $ANALYSIS_ROOT/.stage_seal_complete ]] || fail "run seal before revealing RMSD"
    [[ ! -e $ANALYSIS_ROOT/05_metrics ]] || fail "metrics output already exists"
    METRICS=$ANALYSIS_ROOT/05_metrics
    mkdir -p "$METRICS"
    "$PYTHON_BIN" -u evaluation/evaluate_cascade_branch_rmsd_cached.py \
      --manifest "$ANALYSIS_ROOT/02_inference/available/evaluation_manifest_available.csv" \
      --results "$ANALYSIS_ROOT/04_v2/merged/v2_selected" \
      --condition validation_cascade_v2 \
      --workers "$RMSD_WORKERS" \
      --expected-parents "$EXPECTED_N" \
      --branches-per-parent 5 \
      --legacy-check-parents 3 \
      --output-dir "$METRICS/branch_v2"
    "$PYTHON_BIN" - "$METRICS/branch_v2/summary.json" "$((5 * EXPECTED_N))" <<'PY'
import json,sys
summary=json.load(open(sys.argv[1]))
expected=int(sys.argv[2])
if summary.get('num_complexes') != expected or summary.get('num_failures') != 0:
    raise SystemExit(f"Branch RMSD incomplete: {summary}")
print(f"BRANCH_RMSD_COMPLETE: {expected}/{expected}")
PY
    "$PYTHON_BIN" -u evaluation/summarize_cascade_branches.py \
      --branch-manifest "$ANALYSIS_ROOT/02_inference/available/inference_blind_available.csv" \
      --branch-metrics "$METRICS/branch_v2/per_complex.csv" \
      --requested-cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --output-dir "$METRICS/cascade_v2"
    "$PYTHON_BIN" -u evaluation/evaluate_cascade_cross_branch_selection.py \
      --branch-manifest "$ANALYSIS_ROOT/02_inference/available/inference_blind_available.csv" \
      --branch-metrics "$METRICS/branch_v2/per_complex.csv" \
      --v2-selected-root "$ANALYSIS_ROOT/04_v2/merged/v2_selected" \
      --requested-cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --output-dir "$METRICS/cross"
    GLOBAL_TMP=$ANALYSIS_ROOT/.global_comparison_tmp
    [[ ! -e $GLOBAL_TMP ]] || fail "remove or preserve failed temporary comparison before retrying: $GLOBAL_TMP"
    "$PYTHON_BIN" -u evaluation/compare_global_cascade.py \
      --cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --expected-n "$EXPECTED_N" \
      --global-scores "$ANALYSIS_ROOT/global/global_top20_scores.csv" \
      --source-reveal "$ANALYSIS_ROOT/00_source/cascade_source_reveal.csv" \
      --reference-manifest "$ANALYSIS_ROOT/01_prepared/evaluation_manifest.csv" \
      --cascade-per-parent "$METRICS/cascade_v2/cascade_per_complex.csv" \
      --cross-per-parent "$METRICS/cross/cross_branch_per_parent.csv" \
      --cross-summary "$METRICS/cross/summary.json" \
      --output-dir "$GLOBAL_TMP"
    mv "$GLOBAL_TMP" "$METRICS/global_comparison"
    "$PYTHON_BIN" -u evaluation/upgrade_cascade_selection.py \
      --cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --expected-n "$EXPECTED_N" \
      --blind-manifest "$ANALYSIS_ROOT/01_prepared/inference_blind.csv" \
      --selected-root "$ANALYSIS_ROOT/04_v2/merged/v2_selected" \
      --baseline-dir "$METRICS/global_comparison" \
      --branch-table "$METRICS/cross/cross_branch_input_table.csv" \
      --output-dir "$ANALYSIS_ROOT/06_selection" \
      --evaluate-only
    touch "$ANALYSIS_ROOT/.stage_evaluate_complete"
    echo "VALIDATION_CASCADE_EVALUATED: $METRICS/global_comparison/success_summary.csv"
    ;;
  tune)
    [[ -f $ANALYSIS_ROOT/.stage_evaluate_complete ]] || fail "run evaluate first"
    [[ ! -e $ANALYSIS_ROOT/07_tuning ]] || fail "tuning output already exists"
    "$PYTHON_BIN" -u evaluation/tune_validation_cascade_selector.py \
      --cohort "$ANALYSIS_ROOT/inputs/cohort.txt" \
      --expected-n "$EXPECTED_N" \
      --candidate-features "$ANALYSIS_ROOT/06_selection/candidate_features_blind.csv" \
      --selection-manifest "$ANALYSIS_ROOT/06_selection/experiment_manifest.json" \
      --global-audit "$ANALYSIS_ROOT/05_metrics/global_comparison/global_top1_audit.csv" \
      --branch-table "$ANALYSIS_ROOT/05_metrics/cross/cross_branch_input_table.csv" \
      --output-dir "$ANALYSIS_ROOT/07_tuning"
    touch "$ANALYSIS_ROOT/.stage_tune_complete"
    echo "VALIDATION_TUNING_COMPLETE: $ANALYSIS_ROOT/07_tuning/frozen_selector.json"
    ;;
  status)
    echo "RUN_ROOT=$RUN_ROOT"
    echo "ANALYSIS_ROOT=$ANALYSIS_ROOT"
    for name in assemble seal evaluate tune; do
      if [[ -f $ANALYSIS_ROOT/.stage_${name}_complete ]]; then
        echo "$name=complete"
      else
        echo "$name=pending"
      fi
    done
    ;;
  *)
    fail "usage: $0 {audit|assemble|seal|evaluate|tune|status}"
    ;;
esac
