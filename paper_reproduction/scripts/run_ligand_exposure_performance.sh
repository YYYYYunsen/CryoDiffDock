#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
EXPOSURE=${EXPOSURE:-$PROJECT_ROOT/visualizations/figure4_cryoliganddb/data/pdbbind_comparison/cryoliganddb_druglike_exposure.csv}
PERFORMANCE=${PERFORMANCE:-$PROJECT_ROOT/artifacts/figure2_authoritative174_v1/03_existing_local174/local174_v2_metrics_exact.csv}
AUTHORITATIVE_IDS=${AUTHORITATIVE_IDS:-$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/visualizations/figure4_cryoliganddb/data/ligand_exposure_performance}

for required_file in "$EXPOSURE" "$PERFORMANCE" "$AUTHORITATIVE_IDS"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required input file does not exist: $required_file" >&2
    exit 2
  fi
done

cd "$PROJECT_ROOT"
python -u evaluation/analyze_ligand_exposure_performance.py \
  --exposure "$EXPOSURE" \
  --performance "$PERFORMANCE" \
  --authoritative-ids "$AUTHORITATIVE_IDS" \
  --output-dir "$OUTPUT_DIR" \
  --method-name "CryoDiffDock-Local V2" \
  --metric-column top1_rmsd_A \
  --strict \
  "$@"

echo "LIGAND_EXPOSURE_PERFORMANCE_RUN_COMPLETE: $OUTPUT_DIR"
