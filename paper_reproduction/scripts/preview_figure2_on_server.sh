#!/usr/bin/env bash
# Build derived preview tables and print frozen-99 Figure 2 sanity checks.
# No docking, inference, or local search is run by this script.

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

run() {
  "$@" || fail "Command failed: $*"
}

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}

CHEMEM_ROOT=${CHEMEM_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1}
EMERALD_ROOT=${EMERALD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1}
EMERALD_IDS=${EMERALD_IDS:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_v1/frozen_99_ids.txt}
V1_ROOT=${V1_ROOT:-$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
OLD_V2_ROOT=${OLD_V2_ROOT:-$POCKET_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1}
FIG2_WORK=${FIG2_WORK:-$POCKET_ROOT/artifacts/figure2_prepare_v1}
V2_TOP1_ROOT=${V2_TOP1_ROOT:-$FIG2_WORK/v2_frozen99_top1_runtime_v1}
PREVIEW_DIR=${PREVIEW_DIR:-$FIG2_WORK/preview_final}

PYTHON_BIN="$ENV_BIN/python"
EMERALD_FINAL_DIR="$EMERALD_ROOT/final_analysis"
EMERALD_FINAL="$EMERALD_FINAL_DIR/emerald_paper_summary_recovery_aware_final.tsv"
V2_RUNTIME="$FIG2_WORK/cryodiffdock_v2_frozen99_top1_runtime.csv"

for required in \
  "$PYTHON_BIN" \
  "$EMERALD_IDS" \
  "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  "$V1_ROOT/02_prepared/inference.csv" \
  "$V1_ROOT/02_prepared/inference_blind.csv" \
  "$V1_ROOT/03_inference/logs/baseline.log" \
  "$V1_ROOT/03_inference/logs/guided.log" \
  "$V1_ROOT/04_ncc/pose_scores.csv" \
  "$V2_TOP1_ROOT/evaluation_manifest.csv" \
  "$V2_TOP1_ROOT/per_complex_audit.csv" \
  "$V2_TOP1_ROOT/summary.json" \
  "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv" \
  "$FIG2_WORK/chemem_iteration_runtime.csv" \
  "$POCKET_ROOT/scripts/build_emerald99_recovery_summary.py" \
  "$POCKET_ROOT/scripts/extract_figure2_frozen_inference_runtime.py" \
  "$POCKET_ROOT/scripts/preview_figure2_results.py"
do
  [[ -e "$required" ]] || fail "Missing required input: $required"
done

run mkdir -p "$PREVIEW_DIR"
cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"

echo "[1/4] Verify Top-1-only timing mode and build reveal manifest"
run "$PYTHON_BIN" - \
  "$V2_TOP1_ROOT/summary.json" \
  "$V2_TOP1_ROOT/evaluation_manifest.csv" \
  "$V1_ROOT/02_prepared/inference.csv" \
  "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" <<'PY'
import json
import pandas as pd
import sys

summary_path, blind_path, reference_path, output_path = sys.argv[1:]
summary = json.load(open(summary_path))
if summary.get("materialization_mode") != "top1":
    raise ValueError("V2 timing run was not executed with --materialization-mode top1")
blind = pd.read_csv(blind_path, dtype={"complex_name": str})
reference = pd.read_csv(reference_path, dtype={"complex_name": str})
merged = blind.merge(
    reference[["complex_name", "reference_ligand"]],
    on="complex_name", how="left", validate="one_to_one", sort=False,
)
if len(merged) != 99 or merged["complex_name"].nunique() != 99:
    raise ValueError(f"Expected 99 unique V2 cases, found {len(merged)}")
if merged["reference_ligand"].isna().any():
    raise ValueError("Reference ligand missing after reveal merge")
merged.to_csv(output_path, index=False)
print(f"V2_REVEAL_99_OK: {output_path}")
PY

echo "[2/4] Evaluate saved V2 Top-1 poses outside the runtime boundary"
run "$PYTHON_BIN" evaluation/evaluate_ligand_accuracy.py \
  --manifest "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" \
  --results "$V2_TOP1_ROOT/v2_selected" \
  --condition v2_selected \
  --num-predictions 1 \
  --output-dir "$V2_TOP1_ROOT/metrics_v2_selected" \
  --fail-on-error

echo "[3/4] Refresh EMERALD and V2 runtime tables"
run "$PYTHON_BIN" scripts/build_emerald99_recovery_summary.py \
  --recovery-audit "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv" \
  --frozen-ids "$EMERALD_IDS" \
  --expected-completed 97 \
  --output "$EMERALD_FINAL"

run "$PYTHON_BIN" scripts/extract_figure2_frozen_inference_runtime.py \
  --manifest "$V1_ROOT/02_prepared/inference_blind.csv" \
  --frozen-ids "$EMERALD_IDS" \
  --baseline-log "$V1_ROOT/03_inference/logs/baseline.log" \
  --guided-log "$V1_ROOT/03_inference/logs/guided.log" \
  --pose-scores "$V1_ROOT/04_ncc/pose_scores.csv" \
  --v2-audit "$V2_TOP1_ROOT/per_complex_audit.csv" \
  --output "$V2_RUNTIME"

echo "[4/4] Print comparison and save preview CSVs"
preview_args=(
  "$PYTHON_BIN" scripts/preview_figure2_results.py
  --cohort "$EMERALD_IDS"
  --chemem "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv"
  --v2 "$V2_TOP1_ROOT/metrics_v2_selected/per_complex.csv"
  --emerald "$EMERALD_FINAL"
  --chemem-runtime "$FIG2_WORK/chemem_iteration_runtime.csv"
  --v2-runtime "$V2_RUNTIME"
  --output-dir "$PREVIEW_DIR"
)

if [[ -f "$OLD_V2_ROOT/metrics_v2_selected/per_complex.csv" ]]; then
  preview_args+=(--old-v2 "$OLD_V2_ROOT/metrics_v2_selected/per_complex.csv")
fi

run "${preview_args[@]}"
echo "FIGURE2_PREVIEW_OK: $PREVIEW_DIR"
