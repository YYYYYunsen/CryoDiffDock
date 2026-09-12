#!/usr/bin/env bash
# Finalize Figure 2 after every external method and the V2 Top-1 timing run.
# This script performs evaluation, table construction, statistics, and archive
# creation only. It does not run docking, inference, or local search.

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
CHEMEM_OLD_ROOT=${CHEMEM_OLD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike60_v1}
EMERALD_ROOT=${EMERALD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1}
EMERALD_IDS=${EMERALD_IDS:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_v1/frozen_99_ids.txt}
V1_ROOT=${V1_ROOT:-$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
FIG2_WORK=${FIG2_WORK:-$POCKET_ROOT/artifacts/figure2_prepare_v1}
V2_TOP1_ROOT=${V2_TOP1_ROOT:-$FIG2_WORK/v2_frozen99_top1_runtime_v1}
FIG2_BUNDLE=${FIG2_BUNDLE:-$POCKET_ROOT/artifacts/figure2_external_local_bundle_final_v1}

PYTHON_BIN="$ENV_BIN/python"
EMERALD_FINAL_DIR="$EMERALD_ROOT/final_analysis"
EMERALD_FINAL="$EMERALD_FINAL_DIR/emerald_paper_summary_recovery_aware_final.tsv"
V2_RUNTIME="$FIG2_WORK/cryodiffdock_v2_frozen99_top1_runtime.csv"

required_files=(
  "$PYTHON_BIN"
  "$EMERALD_IDS"
  "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv"
  "$CHEMEM_ROOT/02_live_analysis/new_chemem_per_pose.csv"
  "$CHEMEM_ROOT/00_cohort/chemem100_cohort.csv"
  "$V1_ROOT/02_prepared/inference.csv"
  "$V1_ROOT/02_prepared/inference_blind.csv"
  "$V1_ROOT/03_inference/logs/baseline.log"
  "$V1_ROOT/03_inference/logs/guided.log"
  "$V1_ROOT/04_ncc/pose_scores.csv"
  "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv"
  "$V1_ROOT/11_metrics_reveal/frozen_s0/per_pose.csv"
  "$V2_TOP1_ROOT/evaluation_manifest.csv"
  "$V2_TOP1_ROOT/per_complex_audit.csv"
  "$V2_TOP1_ROOT/summary.json"
  "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv"
  "$FIG2_WORK/chemem_iteration_runtime.csv"
  "$POCKET_ROOT/scripts/build_figure2_chemem_top1_paths.py"
  "$POCKET_ROOT/scripts/build_emerald99_recovery_summary.py"
  "$POCKET_ROOT/scripts/extract_figure2_frozen_inference_runtime.py"
  "$POCKET_ROOT/scripts/collect_figure2_external_bundle.py"
  "$POCKET_ROOT/scripts/summarize_figure2_external.py"
)

for required in "${required_files[@]}"; do
  [[ -e "$required" ]] || fail "Missing required input: $required"
done

[[ ! -e "$FIG2_BUNDLE" ]] || fail "Bundle path already exists: $FIG2_BUNDLE"
[[ ! -e "$FIG2_BUNDLE.tar.gz" ]] || fail "Archive already exists: $FIG2_BUNDLE.tar.gz"

run mkdir -p "$FIG2_WORK" "$EMERALD_FINAL_DIR"
cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"

echo "[1/7] Build ChemEM Top-1 pose index"
run "$PYTHON_BIN" scripts/build_figure2_chemem_top1_paths.py \
  --benchmark "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  --new-per-pose "$CHEMEM_ROOT/02_live_analysis/new_chemem_per_pose.csv" \
  --historical-root "$CHEMEM_OLD_ROOT" \
  --output "$FIG2_WORK/chemem_top1_paths.csv"

echo "[2/7] Build V2 frozen-99 reveal manifest"
run "$PYTHON_BIN" - \
  "$V2_TOP1_ROOT/evaluation_manifest.csv" \
  "$V1_ROOT/02_prepared/inference.csv" \
  "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" <<'PY'
import pandas as pd
import sys

blind_path, reference_path, output_path = sys.argv[1:]
blind = pd.read_csv(blind_path, dtype={"complex_name": str})
reference = pd.read_csv(reference_path, dtype={"complex_name": str})
if blind["complex_name"].duplicated().any() or reference["complex_name"].duplicated().any():
    raise ValueError("V2 blind/reveal manifests must have unique complex_name")
merged = blind.merge(
    reference[["complex_name", "reference_ligand"]],
    on="complex_name", how="left", validate="one_to_one", sort=False,
)
if len(merged) != 99 or merged["complex_name"].nunique() != 99:
    raise ValueError(f"Expected 99 unique V2 cases, found {len(merged)}")
if merged["reference_ligand"].isna().any():
    raise ValueError("Reference ligand missing after V2 reveal merge")
merged.to_csv(output_path, index=False)
print(f"V2_REVEAL_99_OK: {output_path}")
PY

echo "[3/7] Evaluate only the saved V2 Top-1 pose (outside runtime boundary)"
run "$PYTHON_BIN" evaluation/evaluate_ligand_accuracy.py \
  --manifest "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" \
  --results "$V2_TOP1_ROOT/v2_selected" \
  --condition v2_selected \
  --num-predictions 1 \
  --output-dir "$V2_TOP1_ROOT/metrics_v2_selected" \
  --fail-on-error

echo "[4/7] Rebuild failure-aware EMERALD summary"
run "$PYTHON_BIN" scripts/build_emerald99_recovery_summary.py \
  --recovery-audit "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv" \
  --frozen-ids "$EMERALD_IDS" \
  --expected-completed 97 \
  --output "$EMERALD_FINAL"

echo "[5/7] Build frozen-99 V2 deployment runtime"
run "$PYTHON_BIN" scripts/extract_figure2_frozen_inference_runtime.py \
  --manifest "$V1_ROOT/02_prepared/inference_blind.csv" \
  --frozen-ids "$EMERALD_IDS" \
  --baseline-log "$V1_ROOT/03_inference/logs/baseline.log" \
  --guided-log "$V1_ROOT/03_inference/logs/guided.log" \
  --pose-scores "$V1_ROOT/04_ncc/pose_scores.csv" \
  --v2-audit "$V2_TOP1_ROOT/per_complex_audit.csv" \
  --matched-manifest-output "$FIG2_WORK/cryodiffdock_frozen99_manifest.csv" \
  --pose-score-output "$FIG2_WORK/cryodiffdock_frozen99_pose_scores.csv" \
  --output "$V2_RUNTIME"

echo "[6/7] Preflight frozen-99 inputs"
run "$PYTHON_BIN" - \
  "$EMERALD_IDS" \
  "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv" \
  "$V2_TOP1_ROOT/metrics_v2_selected/per_complex.csv" \
  "$EMERALD_FINAL" \
  "$V2_RUNTIME" \
  "$V2_TOP1_ROOT/summary.json" <<'PY'
from pathlib import Path
import json
import pandas as pd
import sys

ids_path, chemem_path, v1_path, v2_path, emerald_path, runtime_path, v2_summary_path = sys.argv[1:]
ids = [line.split("#", 1)[0].strip().upper() for line in open(ids_path)]
ids = [value for value in ids if value]
if len(ids) != 99 or len(set(ids)) != 99:
    raise ValueError("Frozen cohort is not 99 unique IDs")

for label, path, sep in (
    ("ChemEM", chemem_path, ","),
    ("V1", v1_path, ","),
    ("V2", v2_path, ","),
    ("EMERALD", emerald_path, "\t"),
    ("V2 runtime", runtime_path, ","),
):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=sep)
    id_column = next(column for column in ("complex_name", "pdb_id") if column in frame.columns)
    frame[id_column] = frame[id_column].astype(str).str.upper()
    matched = frame[frame[id_column].isin(ids)]
    print(label, "frozen rows:", len(matched), "unique IDs:", matched[id_column].nunique())

v2 = pd.read_csv(v2_path)
runtime = pd.read_csv(runtime_path)
if len(v2) != 99 or v2["complex_name"].astype(str).str.upper().nunique() != 99:
    raise ValueError("V2 Top-1 metrics are not complete frozen-99")
if len(runtime) != 99 or runtime["complex_name"].astype(str).str.upper().nunique() != 99:
    raise ValueError("V2 deployment runtime is not complete frozen-99")
if not runtime["status"].astype(str).str.lower().eq("completed").all():
    raise ValueError("V2 deployment runtime contains incomplete cases")
if not runtime["pose_score_rows"].eq(80).all():
    raise ValueError("At least one frozen case does not have 80 inference poses")
if not runtime["v2_expansion_runtime_source_column"].eq("deployment_runtime_seconds").all():
    raise ValueError("V2 runtime did not use deployment_runtime_seconds")
v2_summary = json.load(open(v2_summary_path))
if v2_summary.get("materialization_mode") != "top1":
    raise ValueError("V2 timing run was not executed with --materialization-mode top1")
print("FIGURE2_FINAL_INPUTS_OK")
PY

echo "[7/7] Build bundle, statistics, and archive"
collector_args=(
  "$PYTHON_BIN" scripts/collect_figure2_external_bundle.py
  --cohort "$EMERALD_IDS"
  --input-manifest "$V1_ROOT/02_prepared/inference.csv"
  --metadata-manifest "$CHEMEM_ROOT/00_cohort/chemem100_cohort.csv"
  --chemem-benchmark "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv"
  --v1-metrics "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv"
  --v2-metrics "$V2_TOP1_ROOT/metrics_v2_selected/per_complex.csv"
  --emerald-summary "$EMERALD_FINAL"
  --chemem-per-pose "$FIG2_WORK/chemem_top1_paths.csv"
  --v1-per-pose "$V1_ROOT/11_metrics_reveal/frozen_s0/per_pose.csv"
  --v2-per-pose "$V2_TOP1_ROOT/metrics_v2_selected/per_pose.csv"
  --chemem-runtime "$FIG2_WORK/chemem_iteration_runtime.csv"
  --v2-runtime "$V2_RUNTIME"
  --protocol-file "$V2_TOP1_ROOT/summary.json"
  --protocol-file "$EMERALD_FINAL_DIR/emerald_paper_summary_recovery_aware_final_audit.json"
  --extra-file "$FIG2_WORK/cryodiffdock_frozen99_manifest.csv"
  --extra-file "$FIG2_WORK/cryodiffdock_frozen99_pose_scores.csv"
  --output-dir "$FIG2_BUNDLE"
)

if [[ -f "$FIG2_WORK/chemem_iteration_runtime.unpaired.csv" ]]; then
  collector_args+=(--protocol-file "$FIG2_WORK/chemem_iteration_runtime.unpaired.csv")
fi
if [[ -f "$EMERALD_FINAL_DIR/emerald99_recovery_summary.json" ]]; then
  collector_args+=(--protocol-file "$EMERALD_FINAL_DIR/emerald99_recovery_summary.json")
fi

run "${collector_args[@]}"

run "$PYTHON_BIN" scripts/summarize_figure2_external.py \
  --accuracy-long "$FIG2_BUNDLE/tables/accuracy_per_complex_long.csv" \
  --runtime-long "$FIG2_BUNDLE/tables/runtime_per_complex_long.csv" \
  --output-dir "$FIG2_BUNDLE/statistics" \
  --expected-frozen-n 99 \
  --bootstrap-repeats 10000 \
  --seed 20260903

run "$PYTHON_BIN" -m json.tool "$FIG2_BUNDLE/bundle_summary.json"
run "$PYTHON_BIN" -m json.tool "$FIG2_BUNDLE/statistics/figure2_statistics_summary.json"
run tar -czf "$FIG2_BUNDLE.tar.gz" -C "$(dirname "$FIG2_BUNDLE")" "$(basename "$FIG2_BUNDLE")"
run sha256sum "$FIG2_BUNDLE.tar.gz"
sha256sum "$FIG2_BUNDLE.tar.gz" >"$FIG2_BUNDLE.tar.gz.sha256" \
  || fail "Could not write archive checksum"

echo "FIGURE2_FINAL_BUNDLE_OK: $FIG2_BUNDLE"
echo "FIGURE2_FINAL_ARCHIVE_OK: $FIG2_BUNDLE.tar.gz"
