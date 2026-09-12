#!/usr/bin/env bash
# Submit the fixed ten-way validation cohort entirely to the H100 partition.
set -euo pipefail
POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
export POCKET_ROOT
export VALCAS_PARTITION=h100
export VALCAS_GRES=gpu:h100:1
export VALCAS_SHARD_COUNT=${VALCAS_H100_SHARD_COUNT:-10}
export VALCAS_JOB_SCRIPT=$POCKET_ROOT/scripts/run_validation_cascade_h100_shard_slurm.sh
export VALCAS_PYTHON=${VALCAS_H100_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em-h100/bin/python}
export VALCAS_SMOKE_RESULT=${VALCAS_H100_SMOKE_RESULT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_h100_env_v1/AUTO_SUCCESS.json}
export VALCAS_RUN_ROOT=${VALCAS_H100_RUN_ROOT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_full_10xH100_env_v1}
exec bash "$POCKET_ROOT/scripts/submit_validation_cascade_a6000_7.sh" "$@"
