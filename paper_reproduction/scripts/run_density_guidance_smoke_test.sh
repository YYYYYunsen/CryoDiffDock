#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

OUT_DIR=${OUT_DIR:-artifacts/density_guidance_smoke}
SAMPLES=${SAMPLES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
INFERENCE_STEPS=${INFERENCE_STEPS:-5}
SEED=${SEED:-20260813}

if [[ -z "${DENSITY_CROP:-}" ]]; then
  DENSITY_CROP="$OUT_DIR/synthetic_3dpf_density_crop.npz"
  python scripts/make_synthetic_density_crop.py \
    --ligand example_data/3dpf_ligand.sdf \
    --output "$DENSITY_CROP"
  echo "WARNING: using a GT-derived synthetic density for engineering smoke only." >&2
fi
if [[ ! -f "$DENSITY_CROP" ]]; then
  echo "Density crop does not exist: $DENSITY_CROP" >&2
  exit 2
fi

python -u inference.py \
  --complex_name 3dpf_density_guided_smoke \
  --protein_path example_data/3dpf_protein.pdb \
  --ligand example_data/3dpf_ligand.sdf \
  --pocket_center_x 9.7742 \
  --pocket_center_y 27.2863 \
  --pocket_center_z 14.6573 \
  --flexible_sidechains A:160-A:193-A:197-A:198-A:222-A:224-A:227 \
  --ligand_density_crop "$DENSITY_CROP" \
  --density_guidance \
  --density_guidance_lambda_tr "${DENSITY_LAMBDA_TR:-0.5}" \
  --density_guidance_lambda_rot "${DENSITY_LAMBDA_ROT:-0.5}" \
  --density_guidance_start "${DENSITY_GUIDANCE_START:-0.6}" \
  --density_guidance_clip_ratio "${DENSITY_CLIP_RATIO:-1.0}" \
  --density_guidance_objective "${DENSITY_GUIDANCE_OBJECTIVE:-atom_density}" \
  --density_guidance_ncc_sigma "${DENSITY_NCC_SIGMA:-1.5}" \
  --density_guidance_ncc_mask_radius "${DENSITY_NCC_MASK_RADIUS:-10.0}" \
  --density_guidance_ncc_chunk_size "${DENSITY_NCC_CHUNK_SIZE:-8192}" \
  --seed "$SEED" \
  --samples_per_complex "$SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --inference_steps "$INFERENCE_STEPS" \
  --out_dir "$OUT_DIR/results"

mapfile -t audits < <(find "$OUT_DIR/results" -mindepth 2 -maxdepth 2 -type f -name density_guidance_audit.json -print)
if [[ ${#audits[@]} -ne 1 ]]; then
  echo "Expected exactly one density_guidance_audit.json; found ${#audits[@]}" >&2
  exit 1
fi
result_dir=$(dirname "${audits[0]}")
test -s "$result_dir/rank1.sdf"
test -s "$result_dir/rank1_protein.pdb"
test -s "$result_dir/density_guidance_audit.json"
python -m json.tool "$result_dir/density_guidance_audit.json" >/dev/null
echo "DENSITY_GUIDANCE_SMOKE_OK: $result_dir"
