#!/usr/bin/env bash
# Assemble and compare all currently available authoritative-174 results.
# This script performs evaluation/statistics only. It runs no docking/inference.

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}

AUTH_ROOT=$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1
WORKLIST_ROOT=$POCKET_ROOT/artifacts/figure2_authoritative174_v1/02_worklists
LOCAL_ROOT=$POCKET_ROOT/artifacts/figure2_authoritative174_v1/03_existing_local174
OUTPUT_ROOT=${OUTPUT_ROOT:-$POCKET_ROOT/artifacts/figure2_authoritative174_v1/04_current_preview}

OLD_CHEMEM_ROOT=${OLD_CHEMEM_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1}
NEW_CHEMEM_ROOT=${NEW_CHEMEM_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_expansion79_v1}
CHEMEM_RECOVERY_ROOT=${CHEMEM_RECOVERY_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_pro_h2_recovery_v1}

OLD_EMERALD_ROOT=${OLD_EMERALD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1}
NEW_EMERALD_ROOT=${NEW_EMERALD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_v1}
EMERALD_RECOVERY_ROOT=${EMERALD_RECOVERY_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_rdkit_recovery4_v1}
EMERALD_SITECENTER_RECOVERY_ROOT=${EMERALD_SITECENTER_RECOVERY_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_sitecenter_recovery_v1}

PYTHON_BIN=$ENV_BIN/python
AUTH_IDS=$AUTH_ROOT/authoritative_druglike174_ids.txt
MANIFEST=$WORKLIST_ROOT/authoritative174_manifest.csv
V2_METRICS=$LOCAL_ROOT/local174_v2_metrics_exact.csv
OLD_CHEMEM=$OLD_CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv
NEW_CHEMEM_CASES=$NEW_CHEMEM_ROOT/01_stage1/cases_to_run.csv
CHEMEM_RECOVERY_CASES=$CHEMEM_RECOVERY_ROOT/recovery_cases.csv
OLD_EMERALD=$OLD_EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_final.tsv
NEW_EMERALD=$NEW_EMERALD_ROOT/emerald_paper_summary.tsv

ASSEMBLED=$OUTPUT_ROOT/assembled
CHEMEM_ANALYSIS=$OUTPUT_ROOT/chemem_analysis
PREVIEW=$OUTPUT_ROOT/comparison
STATISTICS=$OUTPUT_ROOT/statistics
RUNTIME_ROOT=${RUNTIME_ROOT:-$POCKET_ROOT/artifacts/figure2_authoritative174_v1/06_runtime}
CHEMEM_RUNTIME=$RUNTIME_ROOT/chemem_authoritative174_runtime.csv
V2_RUNTIME=$RUNTIME_ROOT/cryodiffdock_authoritative174_runtime_raw.csv
EMERALD_RUNTIME=$RUNTIME_ROOT/emerald_authoritative174_runtime.csv
RUNTIME_LONG=$RUNTIME_ROOT/runtime_per_complex_long.csv

required_files=(
  "$PYTHON_BIN"
  "$AUTH_IDS"
  "$MANIFEST"
  "$V2_METRICS"
  "$OLD_CHEMEM"
  "$NEW_CHEMEM_CASES"
  "$OLD_EMERALD"
  "$NEW_EMERALD"
  "$POCKET_ROOT/scripts/assemble_figure2_authoritative174_current.py"
  "$POCKET_ROOT/evaluation/analyze_chemem_pocket_benchmark.py"
  "$POCKET_ROOT/scripts/preview_figure2_results.py"
  "$POCKET_ROOT/scripts/summarize_figure2_external.py"
)
for required in "${required_files[@]}"; do
  [[ -f "$required" ]] || fail "Missing required input: $required"
done

run mkdir -p "$ASSEMBLED" "$CHEMEM_ANALYSIS" "$PREVIEW"
cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"

assemble_args=(
  "$PYTHON_BIN" scripts/assemble_figure2_authoritative174_current.py
  --cohort-ids "$AUTH_IDS"
  --manifest "$MANIFEST"
  --old-chemem-benchmark "$OLD_CHEMEM"
  --new-chemem-cases "$NEW_CHEMEM_CASES"
  --old-emerald-summary "$OLD_EMERALD"
  --new-emerald-summary "$NEW_EMERALD"
  --emerald-timeout-id 9C6R
  --emerald-timeout-cap-s 43200
  --output-dir "$ASSEMBLED"
  --expected-n 174
)
if [[ -f "$CHEMEM_RECOVERY_CASES" ]]; then
  assemble_args+=(--chemem-recovery-cases "$CHEMEM_RECOVERY_CASES")
fi
if [[ -d "$EMERALD_RECOVERY_ROOT" ]]; then
  assemble_args+=(--emerald-recovery-root "$EMERALD_RECOVERY_ROOT")
fi
if [[ -d "$EMERALD_SITECENTER_RECOVERY_ROOT" ]]; then
  assemble_args+=(--emerald-recovery-root "$EMERALD_SITECENTER_RECOVERY_ROOT")
fi
run "${assemble_args[@]}"

echo "[2/4] Evaluate all currently available ChemEM ranked poses"
run "$PYTHON_BIN" evaluation/analyze_chemem_pocket_benchmark.py \
  --cohort "$ASSEMBLED/preview_cohort.csv" \
  --manifest "$MANIFEST" \
  --old-chemem-results "$ASSEMBLED/chemem_historical_for_evaluator.csv" \
  --new-cases "$ASSEMBLED/chemem_current_cases.csv" \
  --pocket-s0 "$V2_METRICS" \
  --deployment-only \
  --output-dir "$CHEMEM_ANALYSIS" \
  --workers "${CHEMEM_EVAL_WORKERS:-8}" \
  --expected-cohort-n 174

echo "[3/4] Build fixed-174 and strict paired-complete preview"
preview_args=(
  "$PYTHON_BIN" scripts/preview_figure2_results.py
  --cohort "$AUTH_IDS" \
  --chemem "$CHEMEM_ANALYSIS/per_complex_benchmark.csv" \
  --v2 "$V2_METRICS" \
  --emerald "$ASSEMBLED/emerald_current_174.tsv" \
  --output-dir "$PREVIEW" \
  --expected-cohort-n 174 \
  --allow-v2-incomplete
)
if [[ -f "$CHEMEM_RUNTIME" ]]; then
  preview_args+=(--chemem-runtime "$CHEMEM_RUNTIME")
fi
if [[ -f "$V2_RUNTIME" ]]; then
  preview_args+=(--v2-runtime "$V2_RUNTIME")
fi
if [[ -f "$EMERALD_RUNTIME" ]]; then
  preview_args+=(--emerald-runtime "$EMERALD_RUNTIME")
fi
run "${preview_args[@]}"

echo "[4/4] Run current three-software paired statistics"
run "$PYTHON_BIN" - "$PREVIEW/preview_per_complex.csv" "$ASSEMBLED/accuracy_current_long.csv" <<'PY'
import numpy as np
import pandas as pd
import sys

input_path, output_path = sys.argv[1:]
table = pd.read_csv(input_path, dtype={"complex_name": str})
specs = (
    ("ChemEM Stage 1", "chemem_scored", "chemem_rmsd_A"),
    ("CryoDiffDock-Local V2 selected", "v2_scored", "v2_rmsd_A"),
    ("EMERALD", "emerald_scored", "emerald_rmsd_A"),
)
true_values = {"1", "true", "yes", "y", "complete", "completed", "success", "succeeded", "ok"}
rows = []
for method, completed_column, rmsd_column in specs:
    for _, row in table.iterrows():
        completed = str(row[completed_column]).strip().lower() in true_values
        rmsd = pd.to_numeric(pd.Series([row[rmsd_column]]), errors="coerce").iloc[0]
        rows.append({
            "complex_id": str(row["complex_name"]).upper(),
            "method": method,
            "completed": completed and np.isfinite(rmsd),
            "top1_symmetry_corrected_rmsd_A": rmsd if completed else np.nan,
        })
pd.DataFrame(rows).to_csv(output_path, index=False)
PY

statistics_args=(
  "$PYTHON_BIN" scripts/summarize_figure2_external.py
  --accuracy-long "$ASSEMBLED/accuracy_current_long.csv" \
  --output-dir "$STATISTICS" \
  --expected-frozen-n 174 \
  --bootstrap-repeats 10000 \
  --seed 20260906 \
  --overwrite-output
)
if [[ -f "$RUNTIME_LONG" ]]; then
  statistics_args+=(--runtime-long "$RUNTIME_LONG")
fi
run "${statistics_args[@]}"

echo
echo "========== CURRENT FIXED-174 ACCURACY =========="
run "$PYTHON_BIN" - "$STATISTICS/accuracy_success_summary.csv" <<'PY'
import pandas as pd
import sys

table = pd.read_csv(sys.argv[1])
print(table[table["analysis_set"].eq("frozen_174_missing_as_failure")].to_string(index=False))
PY

echo
echo "========== CURRENT STRICT PAIRED ACCURACY =========="
run "$PYTHON_BIN" - "$STATISTICS/accuracy_success_summary.csv" <<'PY'
import pandas as pd
import sys

table = pd.read_csv(sys.argv[1])
print(table[table["analysis_set"].eq("three_software_paired_complete")].to_string(index=False))
PY

echo
echo "========== CURRENT PAIRED TESTS =========="
run "$PYTHON_BIN" - "$STATISTICS/paired_success_tests.csv" <<'PY'
import pandas as pd
import sys

table = pd.read_csv(sys.argv[1])
columns = [
    "cutoff_A", "left_method", "right_method", "paired_n",
    "left_success_count", "right_success_count",
    "delta_right_minus_left_percentage_points",
    "exact_mcnemar_pvalue", "exact_mcnemar_holm_pvalue",
]
print(table[columns].to_string(index=False))
PY

if [[ -s "$STATISTICS/runtime_summary.csv" ]]; then
  echo
  echo "========== METHOD-NATIVE RUNTIME =========="
  run "$PYTHON_BIN" - "$STATISTICS/runtime_summary.csv" <<'PY'
import pandas as pd
import sys

table = pd.read_csv(sys.argv[1])
table = table[table["analysis_set"].isin([
    "frozen_174_all_observed_attempts",
    "frozen_174_successful_outputs_only",
])].copy()
for source, target in (
    ("median_wall_s", "median_runtime_min"),
    ("q1_wall_s", "q1_runtime_min"),
    ("q3_wall_s", "q3_runtime_min"),
    ("mean_wall_s", "mean_runtime_min"),
):
    table[target] = pd.to_numeric(table[source], errors="coerce") / 60.0
columns = [
    "analysis_set", "method", "eligible_cohort_n", "runtime_n",
    "runtime_missing_n", "successful_output_with_runtime_n",
    "failed_or_incomplete_output_with_runtime_n", "median_runtime_min",
    "q1_runtime_min", "q3_runtime_min", "mean_runtime_min",
]
print(table[columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
PY
fi

echo
echo "FIGURE2_AUTHORITATIVE174_CURRENT_PREVIEW_OK: $OUTPUT_ROOT"

echo
echo "========== RESOLUTION-STRATIFIED CURRENT RESULTS =========="
run "$PYTHON_BIN" scripts/summarize_figure2_resolution.py \
  --preview-per-complex "$PREVIEW/preview_per_complex.csv" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/resolution" \
  --expected-n 174
