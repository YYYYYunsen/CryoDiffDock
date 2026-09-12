#!/usr/bin/env bash
set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
CONFIG=${VALCAS_CONFIG:-$POCKET_ROOT/configs/validation_cascade.json}
SMOKE_RESULT=${VALCAS_SMOKE_RESULT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_esmfix_v1/AUTO_SUCCESS.json}
JOB_SCRIPT=${VALCAS_JOB_SCRIPT:-$POCKET_ROOT/scripts/run_validation_cascade_a6000_shard_slurm.sh}
PYTHON_BIN=${VALCAS_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}
WALLTIME=${VALCAS_WALLTIME:-48:00:00}
PARTITION=${VALCAS_PARTITION:-A6000}
GRES=${VALCAS_GRES:-gpu:1}
MAX_SHARDS=${VALCAS_MAX_SHARDS:-187}
FIXED_SHARDS=${VALCAS_SHARD_COUNT:-}
DRY_RUN=0
[[ ${1:-} == --dry-run ]] && DRY_RUN=1
[[ $# -le 1 ]] || fail "usage: $0 [--dry-run]"
[[ $MAX_SHARDS =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_MAX_SHARDS must be positive"
if [[ -n $FIXED_SHARDS && ! $FIXED_SHARDS =~ ^[1-9][0-9]*$ ]]; then
  fail "VALCAS_SHARD_COUNT must be positive when set"
fi

command -v sbatch >/dev/null || fail "sbatch is unavailable"
command -v sinfo >/dev/null || fail "sinfo is unavailable"
for required in "$CONFIG" "$SMOKE_RESULT" "$JOB_SCRIPT" "$PYTHON_BIN"; do
  [[ -e $required ]] || fail "missing required input: $required"
done
mapfile -t IDLE_NODES < <(sinfo -N -h -p "$PARTITION" -t idle -o '%N' | awk 'NF' | sort -u)
AVAILABLE_COUNT=${#IDLE_NODES[@]}
if (( AVAILABLE_COUNT == 0 )) && [[ -z $FIXED_SHARDS ]]; then
  fail "no idle nodes currently reported; set VALCAS_SHARD_COUNT to submit a fixed number into the queue"
fi
if [[ -n $FIXED_SHARDS ]]; then
  SHARD_COUNT=$FIXED_SHARDS
else
  SHARD_COUNT=$AVAILABLE_COUNT
  if (( SHARD_COUNT > MAX_SHARDS )); then
    SHARD_COUNT=$MAX_SHARDS
  fi
fi
(( SHARD_COUNT <= 187 )) || fail "shard count exceeds the 187-parent cohort"
RUN_ROOT=${VALCAS_RUN_ROOT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_full_${SHARD_COUNT}xA6000_v1}

echo "Partition $PARTITION currently reports $AVAILABLE_COUNT idle nodes: ${IDLE_NODES[*]}"
echo "Frozen submission plan: $SHARD_COUNT parent shards under $RUN_ROOT"
if (( SHARD_COUNT > AVAILABLE_COUNT )); then
  echo "Submitting beyond the idle snapshot: $AVAILABLE_COUNT may start now and the remainder will queue."
fi
"$PYTHON_BIN" -u "$POCKET_ROOT/scripts/run_validation_cascade_auto.py" \
  --config "$CONFIG" --mode full --job-shards 1 --job-shard-index 0 \
  --gpus-per-job 1 \
  --smoke-result "$SMOKE_RESULT" --output-root "$RUN_ROOT/preflight_sentinel" --preflight-only

for ((index=0; index<SHARD_COUNT; index++)); do
  shard=$(printf '%s/shard%02dof%02d' "$RUN_ROOT" "$index" "$SHARD_COUNT")
  [[ ! -e $shard ]] || fail "output already exists before submission: $shard"
done

mkdir -p "$RUN_ROOT/slurm_logs"
STATE=$RUN_ROOT/submitted_jobs.tsv
[[ ! -e $STATE ]] || fail "submission record already exists: $STATE"
printf '%s\n' "${IDLE_NODES[@]}" > "$RUN_ROOT/idle_nodes_at_submission.txt"
{
  echo "partition=$PARTITION"
  echo "idle_node_count=$AVAILABLE_COUNT"
  echo "shard_count=$SHARD_COUNT"
  echo "shard_count_source=$([[ -n $FIXED_SHARDS ]] && echo explicit_VALCAS_SHARD_COUNT || echo idle_node_snapshot)"
  echo "walltime=$WALLTIME"
  echo "gres=$GRES"
  echo "cohort_assignment=sorted reviewed IDs, values[index::count]"
} > "$RUN_ROOT/submission_plan.txt"
if (( DRY_RUN )); then
  echo "DRY_RUN_OK: preflight passed; no jobs submitted"
  exit 0
fi
printf 'shard_index\tjob_id\n' > "$STATE"

echo "Submitting $SHARD_COUNT independent one-GPU jobs to $PARTITION."
for ((index=0; index<SHARD_COUNT; index++)); do
  label=$(printf '%02d' "$index")
  job_id=$(sbatch --parsable \
    --job-name="valcas${label}" \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time="$WALLTIME" \
    --output="$RUN_ROOT/slurm_logs/shard${label}-%j.out" \
    --error="$RUN_ROOT/slurm_logs/shard${label}-%j.err" \
    --export="ALL,POCKET_ROOT=$POCKET_ROOT,VALCAS_CONFIG=$CONFIG,VALCAS_SMOKE_RESULT=$SMOKE_RESULT,VALCAS_RUN_ROOT=$RUN_ROOT,VALCAS_SHARD_COUNT=$SHARD_COUNT,VALCAS_PYTHON=$PYTHON_BIN,VALCAS_GPU_CLASS=$PARTITION" \
    "$JOB_SCRIPT" "$index")
  [[ $job_id =~ ^[0-9]+([_;].*)?$ ]] || fail "unexpected sbatch response for shard $index: $job_id"
  printf '%d\t%s\n' "$index" "$job_id" >> "$STATE"
  echo "submitted shard $index -> job $job_id"
done
echo "SUBMITTED_VALIDATION_CASCADE_DYNAMIC: shards=$SHARD_COUNT jobs=$STATE"
