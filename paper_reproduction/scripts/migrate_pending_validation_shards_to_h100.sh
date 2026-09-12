#!/usr/bin/env bash
# Replace specific pending A6000 validation shards with held H100 jobs, then
# atomically cancel the old jobs and release the replacements.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 2; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
# Use migration-specific overrides so stale VALCAS_RUN_ROOT/VALCAS_SHARD_COUNT
# values from an earlier submission cannot redirect an active production run.
RUN_ROOT=${VALCAS_MIGRATION_RUN_ROOT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_full_10xA6000_v3}
SHARD_COUNT=${VALCAS_MIGRATION_SHARD_COUNT:-10}
CONFIG=${VALCAS_MIGRATION_CONFIG:-$POCKET_ROOT/configs/validation_cascade.json}
PYTHON_BIN=${VALCAS_H100_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em-h100/bin/python}
SMOKE_RESULT=${VALCAS_MIGRATION_H100_SMOKE_RESULT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_h100_env_v2/AUTO_SUCCESS.json}
JOB_SCRIPT=${VALCAS_H100_JOB_SCRIPT:-$POCKET_ROOT/scripts/run_validation_cascade_h100_shard_slurm.sh}
WALLTIME=${VALCAS_H100_WALLTIME:-48:00:00}

(( $# > 0 )) || fail "usage: $0 OLD_JOB_ID:SHARD_INDEX [OLD_JOB_ID:SHARD_INDEX ...]"
[[ $SHARD_COUNT =~ ^[1-9][0-9]*$ ]] || fail "VALCAS_MIGRATION_SHARD_COUNT must be positive"
echo "Migration target: $RUN_ROOT"
echo "H100 smoke result: $SMOKE_RESULT"
for required in "$RUN_ROOT" "$CONFIG" "$PYTHON_BIN" "$SMOKE_RESULT" "$JOB_SCRIPT"; do
  [[ -e $required ]] || fail "missing required input: $required"
done

declare -a old_jobs=() shard_indices=() new_jobs=()
for replacement in "$@"; do
  [[ $replacement =~ ^([0-9]+):([0-9]+)$ ]] || fail "invalid OLD_JOB_ID:SHARD_INDEX pair: $replacement"
  old_job=${BASH_REMATCH[1]}
  shard_index=${BASH_REMATCH[2]}
  (( shard_index < SHARD_COUNT )) || fail "shard $shard_index is outside 0..$((SHARD_COUNT - 1))"
  output_root=$(printf '%s/shard%02dof%02d' "$RUN_ROOT" "$shard_index" "$SHARD_COUNT")
  [[ ! -e $output_root ]] || fail "shard output already exists; refusing duplicate work: $output_root"
  state=$(squeue -h -j "$old_job" -o '%T')
  [[ $state == PENDING ]] || fail "source job $old_job must still be PENDING; current state: ${state:-absent}"
  old_jobs+=("$old_job")
  shard_indices+=("$shard_index")
done

mkdir -p "$RUN_ROOT/slurm_logs_h100"

cleanup_held_jobs() {
  if (( ${#new_jobs[@]} > 0 )); then
    scancel "${new_jobs[@]}" 2>/dev/null || true
  fi
}
trap cleanup_held_jobs ERR INT TERM

# Submit replacements on hold first. No H100 computation can start before all
# old jobs have been rechecked and cancelled.
for shard_index in "${shard_indices[@]}"; do
  label=$(printf '%02d' "$shard_index")
  new_job=$(sbatch --parsable --hold \
    --job-name="valh${label}" \
    --partition=h100 \
    --gres=gpu:h100:1 \
    --time="$WALLTIME" \
    --output="$RUN_ROOT/slurm_logs_h100/shard${label}-%j.out" \
    --error="$RUN_ROOT/slurm_logs_h100/shard${label}-%j.err" \
    --export="ALL,POCKET_ROOT=$POCKET_ROOT,VALCAS_CONFIG=$CONFIG,VALCAS_SMOKE_RESULT=$SMOKE_RESULT,VALCAS_RUN_ROOT=$RUN_ROOT,VALCAS_SHARD_COUNT=$SHARD_COUNT,VALCAS_PYTHON=$PYTHON_BIN,VALCAS_GPU_CLASS=h100" \
    "$JOB_SCRIPT" "$shard_index")
  new_job=${new_job%%;*}
  [[ $new_job =~ ^[0-9]+$ ]] || fail "unexpected sbatch response: $new_job"
  new_jobs+=("$new_job")
  echo "held H100 replacement: shard $shard_index -> job $new_job"
done

# Close the race window: none of the source jobs may have started while the
# held replacements were being created.
for old_job in "${old_jobs[@]}"; do
  state=$(squeue -h -j "$old_job" -o '%T')
  [[ $state == PENDING ]] || fail "source job $old_job changed state to ${state:-absent}; held replacements will be cancelled"
done

scancel "${old_jobs[@]}"
for old_job in "${old_jobs[@]}"; do
  for _ in {1..20}; do
    [[ -z $(squeue -h -j "$old_job" -o '%T') ]] && break
    sleep 1
  done
  [[ -z $(squeue -h -j "$old_job" -o '%T') ]] || fail "source job $old_job did not leave the queue"
done

scontrol release "${new_jobs[@]}"
trap - ERR INT TERM

record=$RUN_ROOT/migrated_to_h100.tsv
if [[ ! -e $record ]]; then
  printf 'utc\tshard_index\told_a6000_job\tnew_h100_job\tpython\tsmoke_result\n' > "$record"
fi
utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
for ((i=0; i<${#old_jobs[@]}; i++)); do
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$utc" "${shard_indices[$i]}" "${old_jobs[$i]}" "${new_jobs[$i]}" "$PYTHON_BIN" "$SMOKE_RESULT" >> "$record"
done

echo "MIGRATED_PENDING_SHARDS_TO_H100: ${new_jobs[*]}"
echo "Provenance: $record"
