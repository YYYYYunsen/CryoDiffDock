#!/usr/bin/env bash
#SBATCH --job-name=valcas
#SBATCH --partition=A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }

SHARD_INDEX=${1:-}
SHARD_COUNT=${VALCAS_SHARD_COUNT:?Set VALCAS_SHARD_COUNT from the submission-time idle-node snapshot}
[[ -n ${SLURM_JOB_ID:-} ]] || fail "submit with sbatch $0 <zero-based-shard-index>"
[[ $SHARD_INDEX =~ ^[0-9]+$ ]] || fail "shard index must be a non-negative integer"
[[ $SHARD_COUNT =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_SHARD_COUNT must be positive"
(( SHARD_INDEX < SHARD_COUNT )) || fail "shard index $SHARD_INDEX is outside 0..$((SHARD_COUNT-1))"

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
CONFIG=${VALCAS_CONFIG:-$POCKET_ROOT/configs/validation_cascade.json}
SMOKE_RESULT=${VALCAS_SMOKE_RESULT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_esmfix_v1/AUTO_SUCCESS.json}
RUN_ROOT=${VALCAS_RUN_ROOT:?Set VALCAS_RUN_ROOT}
OUTPUT_ROOT=$(printf '%s/shard%02dof%02d' "$RUN_ROOT" "$SHARD_INDEX" "$SHARD_COUNT")
PYTHON_BIN=${VALCAS_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}

for required in "$PYTHON_BIN" "$CONFIG" "$SMOKE_RESULT" "$POCKET_ROOT/scripts/run_validation_cascade_auto.py"; do
  [[ -e $required ]] || fail "missing required input: $required"
done
[[ ! -e $OUTPUT_ROOT ]] || fail "refusing to overwrite or resume existing shard: $OUTPUT_ROOT"

cd "$POCKET_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
echo "VALIDATION_CASCADE_SHARD_START index=$SHARD_INDEX count=$SHARD_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=${SLURMD_NODENAME:-$(hostname)} JOB=$SLURM_JOB_ID OUTPUT=$OUTPUT_ROOT"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
VISIBLE_GPUS=$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')
[[ $VISIBLE_GPUS == 1 ]] || fail "single-GPU shard requires exactly one visible GPU; torch sees $VISIBLE_GPUS"
export NUM_GPUS=1

"$PYTHON_BIN" -u scripts/run_validation_cascade_auto.py \
  --config "$CONFIG" \
  --mode full \
  --job-shards "$SHARD_COUNT" \
  --job-shard-index "$SHARD_INDEX" \
  --gpus-per-job 1 \
  --smoke-result "$SMOKE_RESULT" \
  --output-root "$OUTPUT_ROOT"

[[ -f $OUTPUT_ROOT/AUTO_SUCCESS.json ]] || fail "runner returned without AUTO_SUCCESS.json"
echo "VALIDATION_CASCADE_SHARD_OK index=$SHARD_INDEX utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
