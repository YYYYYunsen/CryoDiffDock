#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
CASCADE_ROOT=${CASCADE_ROOT:-$PROJECT_ROOT/artifacts/paper_test/global_to_local_cascade_top20_canonical192_v1}
METRICS_ROOT=${METRICS_ROOT:-$CASCADE_ROOT/05_metrics_druglike174_v1}
GLOBAL_SCORES=${GLOBAL_SCORES:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM/artifacts/paper_test/global_top20_residual_shape_canonical192_v1/global_top20_residual_shape_scores_192.csv}
COHORT=${COHORT:-$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$METRICS_ROOT/global_vs_cascade174_v1}
PYTHON_BIN=${PYTHON_BIN:-python}
REFERENCE_MANIFEST=${REFERENCE_MANIFEST:-$CASCADE_ROOT/01_prepared/evaluation_manifest.csv}

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" evaluation/compare_global_cascade.py \
  --cohort "$COHORT" \
  --global-scores "$GLOBAL_SCORES" \
  --source-reveal "$CASCADE_ROOT/00_source/cascade_source_reveal.csv" \
  --reference-manifest "$REFERENCE_MANIFEST" \
  --cascade-per-parent "$METRICS_ROOT/cascade_v2/cascade_per_complex.csv" \
  --cross-per-parent "$METRICS_ROOT/cross_branch_selector_v2/cross_branch_per_parent.csv" \
  --cross-summary "$METRICS_ROOT/cross_branch_selector_v2/summary.json" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
