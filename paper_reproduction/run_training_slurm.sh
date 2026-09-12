#!/usr/bin/env bash
# Train CryoDiffDock on the deposited-coordinate, noncovalent,
# contact-entity-aware, EMDB-disjoint v1 split.
# Submission target: the current 4xA100 SLURM partition.
#
# Usage:
#   bash run_training_slurm.sh          # fresh training from native DiffDock score model
#   bash run_training_slurm.sh resume   # resume this exact run from last_model.pt

set -euo pipefail

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
WORK_DIR=${WORK_DIR:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}

DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed_6radii_deposited}
SPLITS_DIR=${SPLITS_DIR:-${DATA_ROOT}/splits_deposited_noncovalent_cofactor35_contact_entity_emdb_v1}
TRAIN_SPLIT=${TRAIN_SPLIT:-${SPLITS_DIR}/em_train.txt}
VAL_SPLIT=${VAL_SPLIT:-${SPLITS_DIR}/em_val.txt}

# Native DiffDock v1.1 score model. This is intentionally not an older EM-finetuned checkpoint.
PRETRAINED=${PRETRAINED:-/Scr/yunsenz2/data/software/DiffDock/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt}
ESM_EMBEDDINGS=${ESM_EMBEDDINGS:-${WORK_DIR}/data/esm_embeddings.pt}

RUN_NAME=${RUN_NAME:-em_deposited_cofactor35_contact_entity_emdb_v1_diffdock_init}
LOG_DIR=${LOG_DIR:-workdir_em}
# This must remain distinct from every cache made with the old ligand coordinates.
CACHE_PATH=${CACHE_PATH:-data/cache_em_deposited_cofactor35_contact_entity_emdb_v1}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-16}
N_EPOCHS=${N_EPOCHS:-100}
BASE_LR=${BASE_LR:-1e-4}
TOR_WEIGHT=${TOR_WEIGHT:-1.0}
FREEZE_PRETRAINED_EPOCHS=${FREEZE_PRETRAINED_EPOCHS:-10}
RING_LR_MULTIPLIER=${RING_LR_MULTIPLIER:-10.0}
LORA_LR_MULTIPLIER=${LORA_LR_MULTIPLIER:-1.0}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.10}

PARTITION=${PARTITION:-A100}
NUM_GPUS=${NUM_GPUS:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-64}
MEMORY=${MEMORY:-128G}
WALLTIME=${WALLTIME:-48:00:00}

MODE=${1:-fresh}
if [[ "${MODE}" != "fresh" && "${MODE}" != "resume" ]]; then
    echo "Usage: bash $0 [fresh|resume]" >&2
    exit 2
fi

RUN_DIR=${WORK_DIR}/${LOG_DIR}/${RUN_NAME}
RESUME_FLAG=""
if [[ "${MODE}" == "resume" ]]; then
    if [[ ! -f "${RUN_DIR}/last_model.pt" ]]; then
        echo "ERROR: resume requested but checkpoint is missing:" >&2
        echo "  ${RUN_DIR}/last_model.pt" >&2
        exit 2
    fi
    RESUME_FLAG="--resume ${RUN_DIR}"
else
    if [[ -e "${RUN_DIR}/last_model.pt" ]]; then
        echo "ERROR: refusing fresh training over an existing checkpoint:" >&2
        echo "  ${RUN_DIR}/last_model.pt" >&2
        echo "Use 'bash $0 resume' or choose a new RUN_NAME." >&2
        exit 2
    fi
fi

for required_file in \
    "${TRAIN_SPLIT}" \
    "${VAL_SPLIT}" \
    "${PRETRAINED}" \
    "${ESM_EMBEDDINGS}"
do
    if [[ ! -f "${required_file}" ]]; then
        echo "ERROR: required file not found: ${required_file}" >&2
        exit 2
    fi
done
if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory not found: ${DATA_DIR}" >&2
    exit 2
fi

mkdir -p "${WORK_DIR}/slurm_logs"

echo "Submitting ${MODE} training"
echo "  data:       ${DATA_DIR}"
echo "  train:      ${TRAIN_SPLIT} ($(wc -l < "${TRAIN_SPLIT}") complexes)"
echo "  val:        ${VAL_SPLIT} ($(wc -l < "${VAL_SPLIT}") complexes)"
echo "  pretrained: ${PRETRAINED}"
echo "  cache:      ${CACHE_PATH}"
echo "  run:        ${RUN_DIR}"

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=em_contact_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gpus-per-node=${NUM_GPUS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${WORK_DIR}/slurm_logs/${RUN_NAME}_%j.log
#SBATCH --error=${WORK_DIR}/slurm_logs/${RUN_NAME}_%j.log

set -euo pipefail

source /Scr/yunsenz2/data/software/miniconda3/etc/profile.d/conda.sh
conda activate diffdock
export LD_LIBRARY_PATH=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock/lib:\${LD_LIBRARY_PATH:-}

cd "${WORK_DIR}"

echo "============================================================"
echo "SLURM Job ID: \${SLURM_JOB_ID:-N/A}"
echo "Host: \$(hostname)"
echo "Mode: ${MODE}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Data: ${DATA_DIR}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Val split: ${VAL_SPLIT}"
echo "Pretrained: ${PRETRAINED}"
echo "Cache: ${CACHE_PATH}"
echo "Run: ${RUN_DIR}"
echo "============================================================"

python - <<'PY'
from pathlib import Path
import torch

data = Path("${DATA_DIR}")
split_paths = {
    "train": Path("${TRAIN_SPLIT}"),
    "val": Path("${VAL_SPLIT}"),
}

split_ids = {}
for name, path in split_paths.items():
    ids = [line.strip().split()[0].upper() for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"ERROR: duplicate IDs in {name} split")
    split_ids[name] = ids

overlap = sorted(set(split_ids["train"]) & set(split_ids["val"]))
if overlap:
    raise SystemExit(f"ERROR: train/val overlap: {overlap[:20]}")

missing = []
for split_name, ids in split_ids.items():
    for pdb_id in ids:
        complex_dir = data / pdb_id
        for suffix in (
            "ligand.sdf",
            "protein_processed.pdb",
            "ring_features.npz",
            "ring_features.npz.json",
        ):
            path = complex_dir / f"{pdb_id}_{suffix}"
            if not path.is_file():
                missing.append((split_name, pdb_id, str(path)))

if missing:
    for record in missing[:20]:
        print("MISSING", *record)
    raise SystemExit(f"ERROR: {len(missing)} required training assets are missing")

print(f"Preflight passed: train={len(split_ids['train'])}, val={len(split_ids['val'])}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"  GPU {index}: {torch.cuda.get_device_name(index)}")
if torch.cuda.device_count() != ${NUM_GPUS}:
    raise SystemExit(
        f"ERROR: requested ${NUM_GPUS} GPUs but PyTorch sees {torch.cuda.device_count()}"
    )
PY

python -u train_em.py \
  --train_split "${TRAIN_SPLIT}" \
  --val_split "${VAL_SPLIT}" \
  --data_dir "${DATA_DIR}" \
  --ring_features_dir "${DATA_DIR}" \
  --pretrained_model "${PRETRAINED}" \
  --esm_embeddings_path "${ESM_EMBEDDINGS}" \
  --cache_path "${CACHE_PATH}" \
  --ring_dim 24 \
  --ring_vec_dim 6 \
  --freeze_pretrained_epochs "${FREEZE_PRETRAINED_EPOCHS}" \
  --ring_lr_multiplier "${RING_LR_MULTIPLIER}" \
  --lora_lr_multiplier "${LORA_LR_MULTIPLIER}" \
  --use_lora \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --n_epochs "${N_EPOCHS}" \
  --lr "${BASE_LR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --scheduler_patience 10 \
  --save_model_freq 5 \
  --weight_decay 1e-4 \
  --tor_weight "${TOR_WEIGHT}" \
  --run_name "${RUN_NAME}" \
  --log_dir "${LOG_DIR}" ${RESUME_FLAG}

echo "Training complete: ${RUN_DIR}"
EOF
