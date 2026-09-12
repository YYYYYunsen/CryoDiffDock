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

OUT_ROOT=${OUT_ROOT:-artifacts/density_particle_resampling_ablation}
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
  baseline|resampled|both) ;;
  *) echo "CONDITION must be baseline, resampled, or both" >&2; exit 2 ;;
esac

if [[ "$CONDITION" == "baseline" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --out_dir "$OUT_ROOT/baseline"
fi

if [[ "$CONDITION" == "resampled" || "$CONDITION" == "both" ]]; then
  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --density_particle_resampling \
    --density_particle_checkpoints "${PARTICLE_CHECKPOINTS:-0.8,0.9}" \
    --density_particle_protected_fraction "${PARTICLE_PROTECTED_FRACTION:-0.25}" \
    --density_particle_target_ess_fraction "${PARTICLE_TARGET_ESS_FRACTION:-0.7}" \
    --density_particle_resample_ess_threshold "${PARTICLE_RESAMPLE_ESS_THRESHOLD:-0.8}" \
    --density_particle_max_beta "${PARTICLE_MAX_BETA:-10.0}" \
    --density_particle_ncc_sigma "${PARTICLE_NCC_SIGMA:-1.5}" \
    --density_particle_ncc_mask_radius "${PARTICLE_NCC_MASK_RADIUS:-10.0}" \
    --density_particle_ncc_chunk_size "${PARTICLE_NCC_CHUNK_SIZE:-512}" \
    --out_dir "$OUT_ROOT/resampled"
fi

echo "DENSITY_PARTICLE_RESAMPLING_ABLATION_OK: $OUT_ROOT"
