#!/usr/bin/env bash
# Train CryoDiffDock on the deposited-coordinate, noncovalent,
# contact-entity-aware, EMDB-disjoint v1 split.
#
# This is the baseline script with exactly one experimental variable changed:
# ring features come from full_polymer masking (the complete deposited model
# masks the map) instead of feature_structure masking (the cropped receptor
# masked the map, which put 72.65% of the strongest ring signal on protein the
# crop had removed). Every hyperparameter is identical to
# run_training_slurm.sh so the two runs stay comparable.
#
# Submission target: the current 4xA100 SLURM partition.
#
# Usage:
#   bash run_training_full_polymer_slurm.sh          # fresh training from native DiffDock score model
#   bash run_training_full_polymer_slurm.sh resume   # resume this exact run from last_model.pt

set -euo pipefail

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
WORK_DIR=${WORK_DIR:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}

DATA_DIR=${DATA_DIR:-${DATA_ROOT}/processed_6radii_deposited}
# The one experimental variable. Protein and ligand files still come from
# DATA_DIR; only the density features change.
RING_DIR=${RING_DIR:-${DATA_ROOT}/processed_6radii_full_polymer}
SPLITS_DIR=${SPLITS_DIR:-${DATA_ROOT}/splits_deposited_noncovalent_cofactor35_contact_entity_emdb_v1}
TRAIN_SPLIT=${TRAIN_SPLIT:-${SPLITS_DIR}/em_train.txt}
VAL_SPLIT=${VAL_SPLIT:-${SPLITS_DIR}/em_val.txt}

# Native DiffDock v1.1 score model. This is intentionally not an older EM-finetuned checkpoint.
PRETRAINED=${PRETRAINED:-/Scr/yunsenz2/data/software/DiffDock/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt}
ESM_EMBEDDINGS=${ESM_EMBEDDINGS:-${WORK_DIR}/data/esm_embeddings.pt}

RUN_NAME=${RUN_NAME:-em_full_polymer_cofactor35_contact_entity_emdb_v1_diffdock_init}
LOG_DIR=${LOG_DIR:-workdir_em}
# The EMDataset cache directory name does not encode ring_features_dir, so a
# cache built with the old features would be reused silently and the run would
# train on them. This must stay distinct from every earlier cache.
CACHE_PATH=${CACHE_PATH:-data/cache_em_full_polymer_cofactor35_contact_entity_emdb_v1}

BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-16}
N_EPOCHS=${N_EPOCHS:-100}
BASE_LR=${BASE_LR:-1e-4}
SCHEDULER_PATIENCE=${SCHEDULER_PATIENCE:-3}
TOR_WEIGHT=${TOR_WEIGHT:-1.0}
FREEZE_PRETRAINED_EPOCHS=${FREEZE_PRETRAINED_EPOCHS:-10}
RING_LR_MULTIPLIER=${RING_LR_MULTIPLIER:-10.0}
# Model-side only: no feature regeneration needed to change this. Default 1.0
# matches every earlier run here; a run that changes it must pick a new
# RUN_NAME/CACHE_PATH pair like any other experimental variable, but the cache
# itself is unaffected since ring_vec_scale is applied at forward time, not
# baked into the cached graphs.
RING_VEC_SCALE=${RING_VEC_SCALE:-1.0}
LORA_LR_MULTIPLIER=${LORA_LR_MULTIPLIER:-1.0}
LORA_R=${LORA_R:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.10}

# Sampling-based validation. This run uses the docking metric for checkpoint
# selection, so it is enabled every five epochs by default. It does not replace
# the differentiable score-matching loss used for gradient updates.
INFERENCE_EPOCH_FREQ=${INFERENCE_EPOCH_FREQ:-5}
INFERENCE_COMPLEXES=${INFERENCE_COMPLEXES:-100}
INFERENCE_SAMPLES=${INFERENCE_SAMPLES:-10}
INFERENCE_STEPS=${INFERENCE_STEPS:-20}
SELECTION_METRIC=${SELECTION_METRIC:-median_min_rmsd}

PARTITION=${PARTITION:-A100}
NUM_GPUS=${NUM_GPUS:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-64}
MEMORY=${MEMORY:-128G}
WALLTIME=${WALLTIME:-48:00:00}
# Override these on systems where the local conda installation and GPU Slurm
# directive differ, e.g. Stampede3: CONDA_INIT=$HOME/.../conda.sh,
# CONDA_ENV=$WORK/conda_envs/cryodiffdock, GPU_DIRECTIVE="", ACCOUNT=...
CONDA_INIT=${CONDA_INIT:-/Scr/yunsenz2/data/software/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-diffdock}
GPU_DIRECTIVE=${GPU_DIRECTIVE:---gpus-per-node=${NUM_GPUS}}
ACCOUNT=${ACCOUNT:-}
LD_LIBRARY_PATH_PREFIX=${LD_LIBRARY_PATH_PREFIX:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock/lib}

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

INFERENCE_FLAGS=""
if [[ "${INFERENCE_EPOCH_FREQ}" -gt 0 ]]; then
    for positive_value in "${INFERENCE_COMPLEXES}" "${INFERENCE_SAMPLES}" "${INFERENCE_STEPS}"; do
        if [[ "${positive_value}" -le 0 ]]; then
            echo "ERROR: inference epoch settings must be positive when enabled:" >&2
            echo "  complexes=${INFERENCE_COMPLEXES} samples=${INFERENCE_SAMPLES} steps=${INFERENCE_STEPS}" >&2
            exit 2
        fi
    done
    INFERENCE_FLAGS="--inference_epoch_freq ${INFERENCE_EPOCH_FREQ}"
    INFERENCE_FLAGS="${INFERENCE_FLAGS} --inference_complexes ${INFERENCE_COMPLEXES}"
    INFERENCE_FLAGS="${INFERENCE_FLAGS} --inference_samples ${INFERENCE_SAMPLES}"
    INFERENCE_FLAGS="${INFERENCE_FLAGS} --inference_steps ${INFERENCE_STEPS}"
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
if [[ ! -d "${RING_DIR}" ]]; then
    echo "ERROR: ring feature directory not found: ${RING_DIR}" >&2
    exit 2
fi

mkdir -p "${WORK_DIR}/slurm_logs"

GPU_SBATCH_LINE=""
if [[ -n "${GPU_DIRECTIVE}" ]]; then
    GPU_SBATCH_LINE="#SBATCH ${GPU_DIRECTIVE}"
fi
ACCOUNT_SBATCH_LINE=""
if [[ -n "${ACCOUNT}" ]]; then
    ACCOUNT_SBATCH_LINE="#SBATCH --account=${ACCOUNT}"
fi

echo "Submitting ${MODE} training"
echo "  data:       ${DATA_DIR}"
echo "  ring:       ${RING_DIR}"
echo "  vec_scale:  ${RING_VEC_SCALE}"
echo "  selection:  ${SELECTION_METRIC}"
echo "  scheduler patience: ${SCHEDULER_PATIENCE} RMSD evaluations"
echo "  train:      ${TRAIN_SPLIT} ($(wc -l < "${TRAIN_SPLIT}") complexes)"
echo "  val:        ${VAL_SPLIT} ($(wc -l < "${VAL_SPLIT}") complexes)"
echo "  pretrained: ${PRETRAINED}"
echo "  cache:      ${CACHE_PATH}"
echo "  run:        ${RUN_DIR}"
if [[ -n "${INFERENCE_FLAGS}" ]]; then
    echo "  inference:  every ${INFERENCE_EPOCH_FREQ} epoch(s), ${INFERENCE_COMPLEXES} complexes, ${INFERENCE_SAMPLES}x${INFERENCE_STEPS}"
else
    echo "  inference:  disabled (checkpoint selection by val loss only)"
fi

sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --partition=${PARTITION}
#SBATCH --job-name=em_fullpoly_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
${GPU_SBATCH_LINE}
#SBATCH --ntasks-per-node=1
${ACCOUNT_SBATCH_LINE}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${WALLTIME}
#SBATCH --output=${WORK_DIR}/slurm_logs/${RUN_NAME}_%j.log
#SBATCH --error=${WORK_DIR}/slurm_logs/${RUN_NAME}_%j.log

set -euo pipefail

source "${CONDA_INIT}"
conda activate "${CONDA_ENV}"
if [[ -n "${LD_LIBRARY_PATH_PREFIX}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH_PREFIX}:\${LD_LIBRARY_PATH:-}"
fi

cd "${WORK_DIR}"

echo "============================================================"
echo "SLURM Job ID: \${SLURM_JOB_ID:-N/A}"
echo "Host: \$(hostname)"
echo "Mode: ${MODE}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Data: ${DATA_DIR}"
echo "Ring: ${RING_DIR}"
echo "Ring vector scale: ${RING_VEC_SCALE}"
echo "Selection metric: ${SELECTION_METRIC}"
echo "Train split: ${TRAIN_SPLIT}"
echo "Val split: ${VAL_SPLIT}"
echo "Pretrained: ${PRETRAINED}"
echo "Cache: ${CACHE_PATH}"
echo "Run: ${RUN_DIR}"
echo "============================================================"

python - <<'PY'
from pathlib import Path

import numpy as np
import torch

data = Path("${DATA_DIR}")
ring = Path("${RING_DIR}")
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

# Protein and ligand come from the processed dataset; density features come from
# the regenerated directory. Checking both against DATA_DIR would pass on the old
# ring features and never look at the new ones.
missing = []
for split_name, ids in split_ids.items():
    for pdb_id in ids:
        for root, suffix in (
            (data, "ligand.sdf"),
            (data, "protein_processed.pdb"),
            (ring, "ring_features.npz"),
            (ring, "ring_features.npz.json"),
        ):
            path = root / pdb_id / f"{pdb_id}_{suffix}"
            if not path.is_file():
                missing.append((split_name, pdb_id, str(path)))

if missing:
    for record in missing[:20]:
        print("MISSING", *record)
    raise SystemExit(f"ERROR: {len(missing)} required training assets are missing")

print(f"Preflight passed: train={len(split_ids['train'])}, val={len(split_ids['val'])}")

# Guard against training 48 hours on the wrong features: the bundles must say
# they were generated with full_polymer masking.
sample = split_ids["val"][:20]
policies = set()
for pdb_id in sample:
    with np.load(ring / pdb_id / f"{pdb_id}_ring_features.npz", allow_pickle=True) as bundle:
        policies.add(str(bundle["mask_policy"]) if "mask_policy" in bundle.files else "(missing)")
if policies != {"full_polymer"}:
    raise SystemExit(f"ERROR: ring features are not full_polymer: {sorted(policies)}")
print(f"Ring feature mask_policy verified on {len(sample)} complexes: {sorted(policies)}")

# The cache name does not encode ring_features_dir, so a cache built against
# other features would be loaded instead of rebuilt and the run would train on
# them. Refusing every existing cache is wrong though: a run interrupted after
# preprocessing leaves a complete, correct cache that should be reused. Decide
# by age instead — a cache older than the features it must contain cannot have
# been built from them.
cache = Path("${WORK_DIR}") / "${CACHE_PATH}"
done_markers = sorted(cache.glob("*/done")) if cache.exists() else []
if not done_markers:
    print(f"Cache path clear, will be built: {cache}")
else:
    feature_mtimes = [path.stat().st_mtime for path in ring.glob("*/*_ring_features.npz")]
    if not feature_mtimes:
        raise SystemExit(f"ERROR: no ring feature bundles under {ring}")
    newest_feature = max(feature_mtimes)
    oldest_done = min(path.stat().st_mtime for path in done_markers)
    if oldest_done < newest_feature:
        raise SystemExit(
            f"ERROR: cache at {cache} predates the ring features in {ring} "
            f"(cache {oldest_done:.0f} < features {newest_feature:.0f}); reusing it would "
            "train on different features. Delete the cache or set CACHE_PATH."
        )
    splits = ", ".join(path.parent.name for path in done_markers)
    print(f"Cache reuse verified: built after the ring features -> {splits}")

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
  --ring_features_dir "${RING_DIR}" \
  --pretrained_model "${PRETRAINED}" \
  --esm_embeddings_path "${ESM_EMBEDDINGS}" \
  --cache_path "${CACHE_PATH}" \
  --ring_dim 24 \
  --ring_vec_dim 6 \
  --freeze_pretrained_epochs "${FREEZE_PRETRAINED_EPOCHS}" \
  --ring_lr_multiplier "${RING_LR_MULTIPLIER}" \
  --ring_vec_scale "${RING_VEC_SCALE}" \
  --selection_metric "${SELECTION_METRIC}" \
  --lora_lr_multiplier "${LORA_LR_MULTIPLIER}" \
  --use_lora \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --n_epochs "${N_EPOCHS}" \
  --lr "${BASE_LR}" \
  --scheduler_patience "${SCHEDULER_PATIENCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --save_model_freq 5 \
  --weight_decay 1e-4 \
  --tor_weight "${TOR_WEIGHT}" \
  --run_name "${RUN_NAME}" \
  --log_dir "${LOG_DIR}" ${INFERENCE_FLAGS} ${RESUME_FLAG}

echo "Training complete: ${RUN_DIR}"
EOF
