#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [[ -z "${INPUT_CSV:-}" ]]; then
  echo "Set INPUT_CSV to a prepared inference.csv" >&2
  exit 2
fi
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV does not exist: $INPUT_CSV" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-artifacts/g1_translation_ablation}
SAMPLES=${SAMPLES:-40}
BATCH_SIZE=${BATCH_SIZE:-10}
INFERENCE_STEPS=${INFERENCE_STEPS:-30}
SEED=${SEED:-20260813}
CONDITION=${CONDITION:-both}

COMMON_ARGS=(
  --protein_ligand_csv "$INPUT_CSV"
  --seed "$SEED"
  --samples_per_complex "$SAMPLES"
  --batch_size "$BATCH_SIZE"
  --inference_steps "$INFERENCE_STEPS"
)

case "$CONDITION" in
  baseline|g1|both) ;;
  *) echo "CONDITION must be baseline, g1, or both" >&2; exit 2 ;;
esac

if [[ "$CONDITION" == "baseline" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --out_dir "$OUT_ROOT/baseline"
fi

if [[ "$CONDITION" == "g1" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --density_translation_correction \
    --density_translation_step_cap "${G1_STEP_CAP:-0.02}" \
    --density_translation_start "${G1_START:-0.7}" \
    --density_translation_min_ncc_gain "${G1_MIN_NCC_GAIN:-0.0}" \
    --density_translation_min_in_bounds "${G1_MIN_IN_BOUNDS:-0.0}" \
    --density_translation_protected_fraction "${G1_PROTECTED_FRACTION:-0.0}" \
    --density_translation_ncc_sigma "${G1_NCC_SIGMA:-1.5}" \
    --density_translation_ncc_mask_radius "${G1_NCC_MASK_RADIUS:-10.0}" \
    --density_translation_ncc_chunk_size "${G1_NCC_CHUNK_SIZE:-512}" \
    --out_dir "$OUT_ROOT/g1"
fi

echo "G1_TRANSLATION_ABLATION_OK: $OUT_ROOT"
