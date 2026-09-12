#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [[ -z "${INPUT_CSV:-}" || ! -f "$INPUT_CSV" ]]; then
  echo "Set INPUT_CSV to the prepared inference.csv" >&2
  exit 2
fi

OUT_ROOT=${OUT_ROOT:-artifacts/density_guidance_c1_c2}
SAMPLES=${SAMPLES:-40}
BATCH_SIZE=${BATCH_SIZE:-10}
INFERENCE_STEPS=${INFERENCE_STEPS:-30}
SEEDS=${SEEDS:-20260813,20260814,20260815,20260816,20260817}
DENSITY_OBJECTIVE=${DENSITY_OBJECTIVE:-map_ncc}
DENSITY_LAMBDA_TR=${DENSITY_LAMBDA_TR:-0.5}
DENSITY_LAMBDA_ROT=${DENSITY_LAMBDA_ROT:-0.5}
DENSITY_LAMBDA_TOR=${DENSITY_LAMBDA_TOR:-0.5}
DENSITY_START=${DENSITY_START:-0.6}
DENSITY_CLIP=${DENSITY_CLIP:-1.0}
DENSITY_SIGMA=${DENSITY_SIGMA:-1.5}
DENSITY_MASK_RADIUS=${DENSITY_MASK_RADIUS:-10.0}
DENSITY_CHUNK_SIZE=${DENSITY_CHUNK_SIZE:-8192}

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

COMMON_ARGS=(
  --protein_ligand_csv "$INPUT_CSV"
  --samples_per_complex "$SAMPLES"
  --batch_size "$BATCH_SIZE"
  --inference_steps "$INFERENCE_STEPS"
  --density_guidance_objective "$DENSITY_OBJECTIVE"
  --density_guidance_lambda_tr "$DENSITY_LAMBDA_TR"
  --density_guidance_lambda_rot "$DENSITY_LAMBDA_ROT"
  --density_guidance_lambda_tor "$DENSITY_LAMBDA_TOR"
  --density_guidance_start "$DENSITY_START"
  --density_guidance_clip_ratio "$DENSITY_CLIP"
  --density_guidance_ncc_sigma "$DENSITY_SIGMA"
  --density_guidance_ncc_mask_radius "$DENSITY_MASK_RADIUS"
  --density_guidance_ncc_chunk_size "$DENSITY_CHUNK_SIZE"
)

for seed in "${SEED_LIST[@]}"; do
  seed=$(echo "$seed" | xargs)
  [[ -n "$seed" ]] || continue

  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --out_dir "$OUT_ROOT/seed_${seed}/baseline"

  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --density_guidance \
    --density_guidance_x0 \
    --out_dir "$OUT_ROOT/seed_${seed}/c1_x0"

  python -u inference.py \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --density_guidance \
    --density_guidance_x0 \
    --density_guidance_torsion \
    --out_dir "$OUT_ROOT/seed_${seed}/c2_x0_torsion"
done

echo "DENSITY_GUIDANCE_C1_C2_ABLATION_OK: $OUT_ROOT"
