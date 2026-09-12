#!/bin/bash
# DiffDock-EM contact-entity-aware v1 evaluation (SLURM / KS cluster hgx4)
#
# SLURM port of run_evaluation_single.sh (which used qsub/PBS). Submits via sbatch,
# activates conda (no apptainer). Multi-GPU inference modes request NUM_GPUS,
# split the input CSV, and launch one pinned worker per GPU.
#
# Usage (from a login node, sbatch available):
#   NUM_GPUS=4 bash run_evaluation_single_slurm.sh pretrained
#   NUM_GPUS=4 bash run_evaluation_single_slurm.sh ablation
#   NUM_GPUS=4 bash run_evaluation_single_slurm.sh test_inference   # test set, 4 GPUs
#   bash run_evaluation_single_slurm.sh test_inference_metrics
#   NUM_GPUS=4 bash run_evaluation_single_slurm.sh train_rest
#   NUM_GPUS=4 bash run_evaluation_single_slurm.sh val_inference
#   bash run_evaluation_single_slurm.sh metrics
#
# Run directly in an interactive allocation (no sbatch):
#   RUN_LOCAL=1 NUM_GPUS=4 bash run_evaluation_single_slurm.sh test_inference
# Resume is automatic. Set RESUME=0 to intentionally rerun every complex.
#
# Monitor:  squeue -u $USER

set -euo pipefail

MODE="${1:-}"
VALID="pretrained|finetuned|external|external_pretrained|external_ablation|external_finetuned|external_local|external_pretrained_local|external_ablation_local|external_finetuned_local|ablation|metrics|external_metrics|external_metrics_local|test_inference|test_inference_metrics|train_check|train_check_metrics|train_rest|train_rest_metrics|val_inference|val_inference_metrics"
if [[ -z "${MODE}" ]]; then
  echo "Usage: bash run_evaluation_single_slurm.sh {${VALID}}"
  exit 1
fi
case "|${VALID}|" in
  *"|${MODE}|"*) ;;
  *) echo "Invalid mode: ${MODE}"; echo "Valid: ${VALID}"; exit 1 ;;
esac

# ============================================================
# Global Configuration (KS cluster paths)
# ============================================================
WORK_DIR="${WORK_DIR:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}"
DATA_ROOT="${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}"
DATA_DIR="${DATA_DIR:-${DATA_ROOT}/processed_6radii_deposited}"
RING_DIR="${RING_DIR:-${DATA_DIR}}"
SPLITS_DIR="${SPLITS_DIR:-${DATA_ROOT}/splits_deposited_noncovalent_cofactor35_contact_entity_emdb_v1}"
RESULTS_DIR="${RESULTS_DIR:-${WORK_DIR}/results_test_deposited_cofactor35_contact_entity_emdb_v1}"
DIFFDOCK_DIR="${DIFFDOCK_DIR:-/Scr/yunsenz2/data/software/DiffDock}"
GT_DATA_DIR="${GT_DATA_DIR:-${DATA_ROOT}/deposited_ligand_references_complete_global_v1}"
GT_LIGAND_SUFFIX="${GT_LIGAND_SUFFIX:-ligand_deposited}"

EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-external/data}"
EXTERNAL_DATA_DIR="${EXTERNAL_DATA_DIR:-${EXTERNAL_DATA_ROOT}/processed_6radii_deposited}"
EXTERNAL_SPLITS_DIR="${EXTERNAL_SPLITS_DIR:-${EXTERNAL_DATA_ROOT}/splits_deposited_noncovalent_cofactor35_entity_emdb_v4}"
EXTERNAL_RESULTS_DIR="${EXTERNAL_RESULTS_DIR:-${WORK_DIR}/results_external_deposited}"

FINETUNED_MODEL_DIR="${FINETUNED_MODEL_DIR:-${WORK_DIR}/workdir_em/em_deposited_cofactor35_contact_entity_emdb_v1_diffdock_init}"
CKPT_NAME="${CKPT_NAME:-best_ema_model.pt}"
TEST_SPLIT="${TEST_SPLIT:-${SPLITS_DIR}/em_test.txt}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${SPLITS_DIR}/em_train.txt}"
VAL_SPLIT="${VAL_SPLIT:-${SPLITS_DIR}/em_val.txt}"
EXTERNAL_TEST_SPLIT="${EXTERNAL_TEST_SPLIT:-${EXTERNAL_SPLITS_DIR}/em_external.txt}"

SAMPLES="${SAMPLES:-40}"
STEPS="${STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-10}"
RING_DIM="${RING_DIM:-24}"
RING_VEC_DIM="${RING_VEC_DIM:-6}"

TRAIN_SAMPLE="${TRAIN_SAMPLE:-300}"
TRAIN_CHECK_RESULTS_DIR="${TRAIN_CHECK_RESULTS_DIR:-${WORK_DIR}/results_train_check_contact_entity_v1}"
TRAIN_REST_RESULTS_DIR="${TRAIN_REST_RESULTS_DIR:-${WORK_DIR}/results_train_rest_contact_entity_v1}"
VAL_RESULTS_DIR="${VAL_RESULTS_DIR:-${WORK_DIR}/results_val_contact_entity_v1}"

NUM_GPUS="${NUM_GPUS:-4}"
# 1: skip complexes with a complete set of SAMPLES output poses; 0: rerun all.
RESUME="${RESUME:-1}"

# SLURM settings
PARTITION="${PARTITION:-A100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-14}"
MEM="${MEM:-256G}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
CONDA_ENV="${CONDA_ENV:-diffdock}"
CONDA_SH="${CONDA_SH:-/Scr/yunsenz2/data/software/miniconda3/etc/profile.d/conda.sh}"

RUN_LOCAL="${RUN_LOCAL:-0}"
GPU_ID="${GPU_ID:-0}"
# Optional comma-separated physical GPU mapping for local parallel inference,
# e.g. GPU_IDS=1,2 with NUM_GPUS=2.  When unset, existing 0..N-1 behaviour is
# retained (and single-GPU local mode continues to honour GPU_ID).
GPU_IDS="${GPU_IDS:-}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/eval_logs_contact_entity_v1}"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${VAL_RESULTS_DIR}" "${EXTERNAL_RESULTS_DIR}"

JOB_NAME="eval_${MODE}"
LOG_FILE="${LOG_DIR}/${MODE}.log"
LOCAL_MODE=0
BASE_MODE="${MODE}"

case "${MODE}" in
  external_local)            LOCAL_MODE=1; BASE_MODE="external_finetuned" ;;
  external_pretrained_local) LOCAL_MODE=1; BASE_MODE="external_pretrained" ;;
  external_ablation_local)   LOCAL_MODE=1; BASE_MODE="external_ablation" ;;
  external_finetuned_local)  LOCAL_MODE=1; BASE_MODE="external_finetuned" ;;
  external_metrics_local)    LOCAL_MODE=1; BASE_MODE="external_metrics" ;;
  external)                  BASE_MODE="external_finetuned" ;;
esac

# GPUs to request: inference uses GPUs; RMSD/statistics modes are CPU-only.
case "${MODE}" in
  pretrained|ablation|external_pretrained|external_ablation|test_inference|train_rest|val_inference)
    SLURM_GPUS="${NUM_GPUS}"
    ;;
  metrics|external_metrics|external_metrics_local|test_inference_metrics|train_check_metrics|train_rest_metrics|val_inference_metrics)
    SLURM_GPUS=0
    ;;
  *) SLURM_GPUS=1 ;;
esac

emit_job_env() {
  local var
  for var in \
    MODE BASE_MODE WORK_DIR DATA_DIR RING_DIR SPLITS_DIR RESULTS_DIR DIFFDOCK_DIR \
    GT_DATA_DIR GT_LIGAND_SUFFIX \
    EXTERNAL_DATA_DIR EXTERNAL_SPLITS_DIR EXTERNAL_RESULTS_DIR EXTERNAL_TEST_SPLIT \
    FINETUNED_MODEL_DIR CKPT_NAME TEST_SPLIT TRAIN_SPLIT VAL_SPLIT \
    SAMPLES STEPS BATCH_SIZE RING_DIM RING_VEC_DIM \
    TRAIN_SAMPLE TRAIN_CHECK_RESULTS_DIR TRAIN_REST_RESULTS_DIR VAL_RESULTS_DIR \
    NUM_GPUS GPU_ID GPU_IDS RESUME
  do
    printf 'export %s=%q\n' "${var}" "${!var}"
  done
}

run_mode() {
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}"
echo "Running on: $(hostname)"
echo "Mode: ${MODE} (base=${BASE_MODE})"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-unset}}"
echo "Data: ${DATA_DIR}"
echo "Test split: ${TEST_SPLIT}"
echo "Ground truth: ${GT_DATA_DIR} (*_${GT_LIGAND_SUFFIX}.sdf)"
echo "Model: ${FINETUNED_MODEL_DIR}/${CKPT_NAME}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

mkdir -p ${RESULTS_DIR} ${TRAIN_CHECK_RESULTS_DIR} ${TRAIN_REST_RESULTS_DIR} ${EXTERNAL_RESULTS_DIR}

# Split one input CSV and launch either native DiffDock or DiffDock-EM workers.
# runner: diffdock | em
# ring_dir: empty for ablation, populated for density-guided finetuned inference.
run_parallel_inference() {
  local runner="$1"
  local input_csv="$2"
  local out_dir="$3"
  local chunk_dir="$4"
  local ring_dir="${5:-}"

  unset CUDA_VISIBLE_DEVICES
  mkdir -p "${out_dir}" "${chunk_dir}"
  rm -f "${chunk_dir}"/chunk_*.csv "${chunk_dir}"/gpu_*.log

  python - \
    "${input_csv}" "${chunk_dir}" "${NUM_GPUS}" \
    "${runner}" "${out_dir}" "${SAMPLES}" "${RESUME}" <<'PY'
import csv
import math
import sys
from pathlib import Path

input_csv = Path(sys.argv[1])
chunk_dir = Path(sys.argv[2])
n_gpus = int(sys.argv[3])
runner = sys.argv[4]
out_dir = Path(sys.argv[5])
samples = int(sys.argv[6])
resume = bool(int(sys.argv[7]))


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def is_complete(complex_name):
    complex_dir = out_dir / complex_name
    if runner == "em":
        return all(nonempty(complex_dir / f"rank{rank}.sdf") for rank in range(1, samples + 1))
    if runner == "diffdock":
        if not nonempty(complex_dir / "rank1.sdf"):
            return False
        return all(
            any(nonempty(path) for path in complex_dir.glob(f"rank{rank}_confidence*.sdf"))
            for rank in range(1, samples + 1)
        )
    raise ValueError(f"unknown runner: {runner}")

with input_csv.open(newline="") as handle:
    reader = csv.reader(handle)
    header = next(reader)
    rows = list(reader)

if resume:
    pending = [row for row in rows if row and not is_complete(row[0].strip())]
    print(f"Resume check: {len(rows) - len(pending)} complete, {len(pending)} remaining")
    rows = pending
else:
    print(f"Resume disabled: rerunning all {len(rows)} complexes")

chunk_size = math.ceil(len(rows) / n_gpus) if rows else 0
written = 0
for gpu_id in range(n_gpus):
    chunk = rows[gpu_id * chunk_size:(gpu_id + 1) * chunk_size] if chunk_size else []
    if not chunk:
        continue
    path = chunk_dir / f"chunk_{gpu_id}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(chunk)
    written += len(chunk)
    print(f"  chunk {gpu_id}: {len(chunk)} complexes")
print(f"Total: {written} complexes across {n_gpus} requested workers")
PY

  echo "Launching ${runner} inference on ${NUM_GPUS} GPUs..."
  local pids=()
  local worker_gpus=()
  local requested_gpu_ids=()
  local gpu_id worker_cuda_id chunk_csv worker_pid log_file

  if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a requested_gpu_ids <<< "${GPU_IDS}"
    if [[ "${#requested_gpu_ids[@]}" -ne "${NUM_GPUS}" ]]; then
      echo "ERROR: GPU_IDS=${GPU_IDS} contains ${#requested_gpu_ids[@]} GPU(s), but NUM_GPUS=${NUM_GPUS}" >&2
      exit 2
    fi
  fi

  for gpu_id in $(seq 0 $((${NUM_GPUS} - 1))); do
    chunk_csv="${chunk_dir}/chunk_${gpu_id}.csv"
    if [[ ! -f "${chunk_csv}" ]]; then
      echo "  skip GPU ${gpu_id} (no chunk)"
      continue
    fi
    log_file="${chunk_dir}/gpu_${gpu_id}.log"

    # In local single-GPU mode, GPU_ID selects the physical GPU requested by
    # the caller (for example GPU_ID=2).  The chunk index remains zero, but it
    # must not silently be interpreted as physical GPU 0.  Multi-GPU and
    # submitted Slurm jobs retain the established contiguous 0..N-1 mapping.
    worker_cuda_id="${gpu_id}"
    if [[ -n "${GPU_IDS}" ]]; then
      worker_cuda_id="${requested_gpu_ids[$gpu_id]}"
    elif [[ "${RUN_LOCAL}" == "1" && "${NUM_GPUS}" == "1" ]]; then
      worker_cuda_id="${GPU_ID}"
    fi

    case "${runner}" in
      diffdock)
        (
          cd "${DIFFDOCK_DIR}"
          CUDA_VISIBLE_DEVICES="${worker_cuda_id}" python inference.py \
            --config default_inference_args.yaml \
            --protein_ligand_csv "${chunk_csv}" \
            --out_dir "${out_dir}" \
            --samples_per_complex "${SAMPLES}" \
            --inference_steps "${STEPS}" \
            --batch_size "${BATCH_SIZE}"
        ) >"${log_file}" 2>&1 &
        ;;
      em)
        local ring_args=()
        if [[ -n "${ring_dir}" ]]; then
          ring_args=(--ring_features_dir "${ring_dir}")
        fi
        CUDA_VISIBLE_DEVICES="${worker_cuda_id}" python "${WORK_DIR}/inference_em.py" \
          --model_dir "${FINETUNED_MODEL_DIR}" \
          --ckpt "${CKPT_NAME}" \
          --protein_ligand_csv "${chunk_csv}" \
          --out_dir "${out_dir}" \
          --ring_dim "${RING_DIM}" \
          --ring_vec_dim "${RING_VEC_DIM}" \
          --samples_per_complex "${SAMPLES}" \
          --inference_steps "${STEPS}" \
          --batch_size "${BATCH_SIZE}" \
          "${ring_args[@]}" >"${log_file}" 2>&1 &
        ;;
      *)
        echo "ERROR: unknown parallel runner: ${runner}" >&2
        exit 2
        ;;
    esac

    worker_pid=$!
    pids+=("${worker_pid}")
    worker_gpus+=("${worker_cuda_id}")
    echo "  GPU ${worker_cuda_id} -> PID ${worker_pid} ($(tail -n +2 "${chunk_csv}" | wc -l) complexes)"
  done

  local failed=0 index
  for index in "${!pids[@]}"; do
    gpu_id="${worker_gpus[$index]}"
    if wait "${pids[$index]}"; then
      echo "  GPU ${gpu_id} completed"
    else
      echo "ERROR: GPU ${gpu_id} worker PID ${pids[$index]} failed; see ${chunk_dir}/gpu_${gpu_id}.log" >&2
      failed=1
    fi
  done
  [[ ${failed} -eq 0 ]] || exit 1
  echo "All parallel ${runner} workers completed."
}

case "${BASE_MODE}" in
  pretrained)
    cd "${WORK_DIR}"
    python evaluate_em.py --test_split "${TEST_SPLIT}" --data_dir "${DATA_DIR}" --results_dir "${RESULTS_DIR}" build-csv --format diffdock
    run_parallel_inference diffdock \
      "${RESULTS_DIR}/test_input_diffdock.csv" \
      "${RESULTS_DIR}/pretrained" \
      "${RESULTS_DIR}/chunks_pretrained"
    ;;

  finetuned)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${TEST_SPLIT} --data_dir ${DATA_DIR} --results_dir ${RESULTS_DIR} build-csv --format em --with-ring
    python inference_em.py --model_dir ${FINETUNED_MODEL_DIR} --ckpt ${CKPT_NAME} \
      --protein_ligand_csv ${RESULTS_DIR}/test_input_em_ring.csv --ring_features_dir ${RING_DIR} \
      --out_dir ${RESULTS_DIR}/finetuned --ring_dim ${RING_DIM} --ring_vec_dim ${RING_VEC_DIM} \
      --samples_per_complex ${SAMPLES} --inference_steps ${STEPS} --batch_size ${BATCH_SIZE}
    ;;

  test_inference)
    # Multi-GPU finetuned on the TEST set -> ${RESULTS_DIR}/finetuned
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${TEST_SPLIT} --data_dir ${DATA_DIR} --results_dir ${RESULTS_DIR} build-csv --format em --with-ring
    run_parallel_inference em \
      "${RESULTS_DIR}/test_input_em_ring.csv" \
      "${RESULTS_DIR}/finetuned" \
      "${RESULTS_DIR}/chunks_finetuned" \
      "${RING_DIR}"
    ;;

  test_inference_metrics)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${TEST_SPLIT} --data_dir ${DATA_DIR} \
      --gt_data_dir ${GT_DATA_DIR} --gt_ligand_suffix ${GT_LIGAND_SUFFIX} \
      --results_dir ${RESULTS_DIR} \
      metrics ${RESULTS_DIR}/finetuned | tee ${RESULTS_DIR}/test_inference_results.txt
    ;;

  external_pretrained)
    cd "${WORK_DIR}"
    python evaluate_em.py --test_split "${EXTERNAL_TEST_SPLIT}" --data_dir "${EXTERNAL_DATA_DIR}" --results_dir "${EXTERNAL_RESULTS_DIR}" build-csv --format diffdock
    run_parallel_inference diffdock \
      "${EXTERNAL_RESULTS_DIR}/test_input_diffdock.csv" \
      "${EXTERNAL_RESULTS_DIR}/pretrained" \
      "${EXTERNAL_RESULTS_DIR}/chunks_pretrained"
    ;;

  external_ablation)
    cd "${WORK_DIR}"
    python evaluate_em.py --test_split "${EXTERNAL_TEST_SPLIT}" --data_dir "${EXTERNAL_DATA_DIR}" --results_dir "${EXTERNAL_RESULTS_DIR}" build-csv --format em
    run_parallel_inference em \
      "${EXTERNAL_RESULTS_DIR}/test_input_em.csv" \
      "${EXTERNAL_RESULTS_DIR}/ablation" \
      "${EXTERNAL_RESULTS_DIR}/chunks_ablation"
    ;;

  external_finetuned)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${EXTERNAL_TEST_SPLIT} --data_dir ${EXTERNAL_DATA_DIR} --results_dir ${EXTERNAL_RESULTS_DIR} build-csv --format em --with-ring
    python inference_em.py --model_dir ${FINETUNED_MODEL_DIR} --ckpt ${CKPT_NAME} \
      --protein_ligand_csv ${EXTERNAL_RESULTS_DIR}/test_input_em_ring.csv --ring_features_dir ${EXTERNAL_DATA_DIR} \
      --out_dir ${EXTERNAL_RESULTS_DIR}/finetuned --ring_dim ${RING_DIM} --ring_vec_dim ${RING_VEC_DIM} \
      --samples_per_complex ${SAMPLES} --inference_steps ${STEPS} --batch_size ${BATCH_SIZE}
    ;;

  ablation)
    cd "${WORK_DIR}"
    python evaluate_em.py --test_split "${TEST_SPLIT}" --data_dir "${DATA_DIR}" --results_dir "${RESULTS_DIR}" build-csv --format em
    run_parallel_inference em \
      "${RESULTS_DIR}/test_input_em.csv" \
      "${RESULTS_DIR}/ablation" \
      "${RESULTS_DIR}/chunks_ablation"
    ;;

  metrics)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${TEST_SPLIT} --data_dir ${DATA_DIR} \
      --gt_data_dir ${GT_DATA_DIR} --gt_ligand_suffix ${GT_LIGAND_SUFFIX} \
      --results_dir ${RESULTS_DIR} \
      metrics ${RESULTS_DIR}/pretrained ${RESULTS_DIR}/ablation ${RESULTS_DIR}/finetuned | tee ${RESULTS_DIR}/ablation_study_results.txt
    ;;

  external_metrics)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${EXTERNAL_TEST_SPLIT} --data_dir ${EXTERNAL_DATA_DIR} --results_dir ${EXTERNAL_RESULTS_DIR} \
      metrics ${EXTERNAL_RESULTS_DIR}/pretrained ${EXTERNAL_RESULTS_DIR}/ablation ${EXTERNAL_RESULTS_DIR}/finetuned | tee ${EXTERNAL_RESULTS_DIR}/external_results.txt
    ;;

  train_check)
    cd ${WORK_DIR}
    TRAIN_SAMPLE_SPLIT=${TRAIN_CHECK_RESULTS_DIR}/em_train_sample${TRAIN_SAMPLE}.txt
    python -c "
import random
random.seed(42)
lines=[l for l in open('${TRAIN_SPLIT}').read().splitlines() if l.strip()]
print('\n'.join(random.sample(lines, min(${TRAIN_SAMPLE}, len(lines)))))
" > ${TRAIN_SAMPLE_SPLIT}
    echo "Sampled $(wc -l < ${TRAIN_SAMPLE_SPLIT}) complexes -> ${TRAIN_SAMPLE_SPLIT}"
    python evaluate_em.py --test_split ${TRAIN_SAMPLE_SPLIT} --data_dir ${DATA_DIR} --results_dir ${TRAIN_CHECK_RESULTS_DIR} build-csv --format em --with-ring
    python inference_em.py --model_dir ${FINETUNED_MODEL_DIR} --ckpt ${CKPT_NAME} \
      --protein_ligand_csv ${TRAIN_CHECK_RESULTS_DIR}/test_input_em_ring.csv --ring_features_dir ${DATA_DIR} \
      --out_dir ${TRAIN_CHECK_RESULTS_DIR}/finetuned --ring_dim ${RING_DIM} --ring_vec_dim ${RING_VEC_DIM} \
      --samples_per_complex ${SAMPLES} --inference_steps ${STEPS} --batch_size ${BATCH_SIZE}
    ;;

  train_check_metrics)
    cd ${WORK_DIR}
    TRAIN_SAMPLE_SPLIT=${TRAIN_CHECK_RESULTS_DIR}/em_train_sample${TRAIN_SAMPLE}.txt
    [[ -f "${TRAIN_SAMPLE_SPLIT}" ]] || { echo "Run train_check first."; exit 1; }
    python evaluate_em.py --test_split ${TRAIN_SAMPLE_SPLIT} --data_dir ${DATA_DIR} \
      --gt_data_dir ${GT_DATA_DIR} --gt_ligand_suffix ${GT_LIGAND_SUFFIX} \
      --results_dir ${TRAIN_CHECK_RESULTS_DIR} \
      metrics ${TRAIN_CHECK_RESULTS_DIR}/finetuned | tee ${TRAIN_CHECK_RESULTS_DIR}/train_check_results.txt
    ;;

  train_rest)
    cd ${WORK_DIR}
    SAMPLE_SPLIT=${TRAIN_CHECK_RESULTS_DIR}/em_train_sample${TRAIN_SAMPLE}.txt
    [[ -f "${SAMPLE_SPLIT}" ]] || { echo "Run train_check first."; exit 1; }
    REST_SPLIT=${TRAIN_REST_RESULTS_DIR}/em_train_rest.txt
    mkdir -p ${TRAIN_REST_RESULTS_DIR}
    python -c "
full=set(l.strip() for l in open('${TRAIN_SPLIT}') if l.strip())
sampled=set(l.strip() for l in open('${SAMPLE_SPLIT}') if l.strip())
rest=sorted(full-sampled)
print(f'Full {len(full)}, sampled {len(sampled)}, rest {len(rest)}')
open('${REST_SPLIT}','w').write('\n'.join(rest)+'\n')
"
    python evaluate_em.py --test_split ${REST_SPLIT} --data_dir ${DATA_DIR} --results_dir ${TRAIN_REST_RESULTS_DIR} build-csv --format em --with-ring
    run_parallel_inference em \
      "${TRAIN_REST_RESULTS_DIR}/test_input_em_ring.csv" \
      "${TRAIN_REST_RESULTS_DIR}/finetuned" \
      "${TRAIN_REST_RESULTS_DIR}/chunks_finetuned" \
      "${RING_DIR}"
    ;;

  train_rest_metrics)
    cd ${WORK_DIR}
    REST_SPLIT=${TRAIN_REST_RESULTS_DIR}/em_train_rest.txt
    [[ -f "${REST_SPLIT}" ]] || { echo "Run train_rest first."; exit 1; }
    python evaluate_em.py --test_split ${REST_SPLIT} --data_dir ${DATA_DIR} \
      --gt_data_dir ${GT_DATA_DIR} --gt_ligand_suffix ${GT_LIGAND_SUFFIX} \
      --results_dir ${TRAIN_REST_RESULTS_DIR} \
      metrics ${TRAIN_REST_RESULTS_DIR}/finetuned | tee ${TRAIN_REST_RESULTS_DIR}/train_rest_results.txt
    ;;

  val_inference)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${VAL_SPLIT} --data_dir ${DATA_DIR} --results_dir ${VAL_RESULTS_DIR} build-csv --format em --with-ring
    run_parallel_inference em \
      "${VAL_RESULTS_DIR}/test_input_em_ring.csv" \
      "${VAL_RESULTS_DIR}/finetuned" \
      "${VAL_RESULTS_DIR}/chunks_finetuned" \
      "${RING_DIR}"
    ;;

  val_inference_metrics)
    cd ${WORK_DIR}
    python evaluate_em.py --test_split ${VAL_SPLIT} --data_dir ${DATA_DIR} \
      --gt_data_dir ${GT_DATA_DIR} --gt_ligand_suffix ${GT_LIGAND_SUFFIX} \
      --results_dir ${VAL_RESULTS_DIR} \
      metrics ${VAL_RESULTS_DIR}/finetuned | tee ${VAL_RESULTS_DIR}/val_inference_results.txt
    ;;
esac

echo "============================================================"
echo "Done: ${MODE}"
echo "============================================================"
}

# ============================================================
# Dispatch: run directly (interactive) or submit to SLURM
# ============================================================
if [[ "${RUN_LOCAL}" == "1" || "${LOCAL_MODE}" == "1" ]]; then
  echo "Running mode=${MODE} directly (RUN_LOCAL)..."
  [[ -f "${CONDA_SH}" ]] && { source "${CONDA_SH}"; conda activate "${CONDA_ENV}" || true; }
  export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
  if [[ "${SLURM_GPUS}" == "0" ]]; then
    unset CUDA_VISIBLE_DEVICES
  elif [[ "${SLURM_GPUS}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
  run_mode 2>&1 | tee "${LOG_FILE}"
else
  GPU_REQUEST_ARGS=()
  if [[ "${SLURM_GPUS}" -gt 0 ]]; then
    GPU_REQUEST_ARGS=(--gres="gpu:${SLURM_GPUS}")
    echo "Submitting mode=${MODE} to ${PARTITION} (${SLURM_GPUS} GPU)..."
  else
    echo "Submitting mode=${MODE} to ${PARTITION} (CPU-only; no GPU requested)..."
  fi
  sbatch \
    --partition="${PARTITION}" --job-name="${JOB_NAME}" \
    --nodes=1 --ntasks=1 --cpus-per-task="${CPUS_PER_TASK}" \
    "${GPU_REQUEST_ARGS[@]}" \
    --mem="${MEM}" --time="${SLURM_TIME}" \
    --output="${LOG_FILE}" --error="${LOG_FILE}" <<EOF
#!/bin/bash
set -euo pipefail
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
$(emit_job_env)
$(declare -f run_mode)
run_mode
EOF
  echo "Submitted. Log: ${LOG_FILE}"
fi
