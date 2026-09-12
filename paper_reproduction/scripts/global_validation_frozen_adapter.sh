#!/usr/bin/env bash
# Generation and confidence ranking reuse Global's existing interfaces.
set -euo pipefail
: "${VAL_COHORT:?Set VAL_COHORT}"
: "${VAL_GLOBAL_OUTPUT:?Set VAL_GLOBAL_OUTPUT}"
: "${VAL_GLOBAL_SCORES:?Set VAL_GLOBAL_SCORES}"
POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}
FINETUNED_MODEL_DIR=${FINETUNED_MODEL_DIR:-$GLOBAL_ROOT/workdir_em/em_deposited_cofactor35_contact_entity_emdb_v1_diffdock_init}
ORIG_CONF_DIR=${ORIG_CONF_DIR:-/Scr/yunsenz2/data/software/DiffDock/workdir/v1.1/confidence_model}
ORIG_CONF_CKPT=${ORIG_CONF_CKPT:-best_model.pt}
CKPT_NAME=${CKPT_NAME:-best_ema_model.pt}
NUM_GPUS=${NUM_GPUS:-1}
ESM_EMBEDDINGS_PATH=${ESM_EMBEDDINGS_PATH:-${ESM_EMBEDDINGS:-$GLOBAL_ROOT/data/esm_embeddings.pt}}
for path in "$VAL_COHORT" "$ESM_EMBEDDINGS_PATH" "$POCKET_ROOT/scripts/check_global_confidence_inputs.py" "$FINETUNED_MODEL_DIR/$CKPT_NAME" "$ORIG_CONF_DIR/$ORIG_CONF_CKPT" "$GLOBAL_ROOT/run_evaluation_single_slurm.sh" "$GLOBAL_ROOT/compare_confidence_rerank.py" "$GLOBAL_ROOT/rerank_by_density.py"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
echo "Diffusion checkpoint: $FINETUNED_MODEL_DIR/$CKPT_NAME"
echo "Confidence checkpoint: $ORIG_CONF_DIR/$ORIG_CONF_CKPT"
sha256sum "$FINETUNED_MODEL_DIR/$CKPT_NAME" "$ORIG_CONF_DIR/$ORIG_CONF_CKPT"
if [[ ${REUSE_GLOBAL_INFERENCE:-0} == 1 ]]; then
  : "${REUSE_GLOBAL_LOG:?Reuse requires original Global log with checkpoint hashes}"
  for checkpoint in "$FINETUNED_MODEL_DIR/$CKPT_NAME" "$ORIG_CONF_DIR/$ORIG_CONF_CKPT"; do
    fingerprint=$(sha256sum "$checkpoint")
    grep -Fqx -- "$fingerprint" "$REUSE_GLOBAL_LOG" || { echo "Reuse checkpoint provenance mismatch: $checkpoint" >&2; exit 2; }
  done
fi
python "$POCKET_ROOT/scripts/check_global_confidence_inputs.py" \
  --cohort "$VAL_COHORT" --ids "${VAL_PREFLIGHT_IDS:-}" --embeddings "$ESM_EMBEDDINGS_PATH"
if [[ ${1:-} == --preflight-only ]]; then
  echo "GLOBAL_PATH_PREFLIGHT_OK (checkpoint defaults must match test provenance)"
  exit 0
fi
VISIBLE_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
[[ $NUM_GPUS =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be positive" >&2; exit 2; }
if (( NUM_GPUS > VISIBLE_GPUS )); then
  echo "Requested NUM_GPUS=$NUM_GPUS but confidence environment sees only $VISIBLE_GPUS GPU(s)" >&2
  exit 2
fi
[[ ! -e "$VAL_GLOBAL_SCORES" ]] || { echo "Refusing to overwrite Global scores" >&2; exit 2; }
mkdir -p "$VAL_GLOBAL_OUTPUT"
cd "$GLOBAL_ROOT"
# RUN_LOCAL=1 is synchronous: run only within an existing GPU allocation/session.
if [[ ${REUSE_GLOBAL_INFERENCE:-0} != 1 ]]; then
WORK_DIR="$GLOBAL_ROOT" RUN_LOCAL=1 NUM_GPUS="$NUM_GPUS" \
VAL_SPLIT="$VAL_COHORT" VAL_RESULTS_DIR="$VAL_GLOBAL_OUTPUT" \
RESULTS_DIR="$VAL_GLOBAL_OUTPUT/unused_test_output" LOG_DIR="$VAL_GLOBAL_OUTPUT/logs" \
FINETUNED_MODEL_DIR="$FINETUNED_MODEL_DIR" CKPT_NAME="$CKPT_NAME" \
SAMPLES=40 STEPS=20 BATCH_SIZE=10 RING_DIM=24 RING_VEC_DIM=6 \
bash run_evaluation_single_slurm.sh val_inference
else
  echo "Reusing validated Global poses; starting at confidence"
fi
python "$POCKET_ROOT/scripts/check_global_confidence_inputs.py" --cohort "$VAL_COHORT" --poses "$VAL_GLOBAL_OUTPUT/finetuned"

python compare_confidence_rerank.py \
  --protein_ligand_csv "$VAL_GLOBAL_OUTPUT/test_input_em_ring.csv" \
  --test_split "$VAL_COHORT" --data_dir "$DATA_ROOT/processed_6radii_deposited" \
  --results_dir "$VAL_GLOBAL_OUTPUT/conf_scored" \
  --pretrained_poses "$VAL_GLOBAL_OUTPUT/finetuned" \
  --ablation_poses "$VAL_GLOBAL_OUTPUT/finetuned" \
  --finetuned_poses "$VAL_GLOBAL_OUTPUT/finetuned" \
  --orig_conf_dir "$ORIG_CONF_DIR" --orig_conf_ckpt "$ORIG_CONF_CKPT" \
  --esm_embeddings_path "$ESM_EMBEDDINGS_PATH" \
  --em_conf_dir "$ORIG_CONF_DIR" --pose_set finetuned --conf_model orig --max_poses 40

python "$POCKET_ROOT/scripts/check_global_confidence_inputs.py" --cohort "$VAL_COHORT" \
  --poses "$VAL_GLOBAL_OUTPUT/conf_scored/finetuned_orig_conf" --confidence

python "$POCKET_ROOT/scripts/global_density_frozen_blind.py" \
  --global-root "$GLOBAL_ROOT" --cohort "$VAL_COHORT" \
  --poses "$VAL_GLOBAL_OUTPUT/conf_scored/finetuned_orig_conf" \
  --map-csv "$DATA_ROOT/04_maps_fixed.csv" \
  --data-dir "$DATA_ROOT/processed_6radii_deposited" --output "$VAL_GLOBAL_SCORES"
