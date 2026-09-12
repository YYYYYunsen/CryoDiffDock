#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [[ -z "${INPUT_CSV:-}" ]]; then
  echo "Set INPUT_CSV to the prepared inference.csv" >&2
  exit 2
fi
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV does not exist: $INPUT_CSV" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-artifacts/oracle_density_ablation}
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
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  COMMON_ARGS+=(--skip_existing)
fi

case "$CONDITION" in
  baseline|guided|both) ;;
  *) echo "CONDITION must be baseline, guided, or both" >&2; exit 2 ;;
esac

if [[ "$CONDITION" == "baseline" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --out_dir "$OUT_ROOT/baseline"
fi

if [[ "$CONDITION" == "guided" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --density_guidance \
    --density_guidance_lambda_tr "${DENSITY_LAMBDA_TR:-0.5}" \
    --density_guidance_lambda_rot "${DENSITY_LAMBDA_ROT:-0.5}" \
    --density_guidance_start "${DENSITY_GUIDANCE_START:-0.6}" \
    --density_guidance_clip_ratio "${DENSITY_CLIP_RATIO:-1.0}" \
    --density_guidance_objective "${DENSITY_GUIDANCE_OBJECTIVE:-atom_density}" \
    --density_guidance_ncc_sigma "${DENSITY_NCC_SIGMA:-1.5}" \
    --density_guidance_ncc_mask_radius "${DENSITY_NCC_MASK_RADIUS:-10.0}" \
    --density_guidance_ncc_chunk_size "${DENSITY_NCC_CHUNK_SIZE:-8192}" \
    --out_dir "$OUT_ROOT/guided"
fi

echo "ORACLE_DENSITY_ABLATION_OK: $OUT_ROOT"
