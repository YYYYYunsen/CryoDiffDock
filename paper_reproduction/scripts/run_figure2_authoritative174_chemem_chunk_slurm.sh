#!/usr/bin/env bash
#SBATCH --job-name=chem174x
#SBATCH --partition=A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=chem174x-%j.out
#SBATCH --error=chem174x-%j.err

START_INDEX=${1:-}
END_INDEX=${2:-}
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit with sbatch $0 <start> <end>" >&2
  exit 2
fi
if [[ ! "$START_INDEX" =~ ^[0-9]+$ || ! "$END_INDEX" =~ ^[0-9]+$ ]]; then
  echo "ERROR: start/end must be non-negative integers" >&2
  exit 2
fi
if (( START_INDEX < 0 || END_INDEX <= START_INDEX || END_INDEX > 79 )); then
  echo "ERROR: require 0 <= start < end <= 79" >&2
  exit 2
fi

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
CHEMEM_ROOT=${CHEMEM_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_expansion79_v1}
CASES=$CHEMEM_ROOT/01_stage1/cases_to_run.csv
RUNNER=$POCKET_ROOT/scripts/run_chemem100_cases.py

[[ -f "$CASES" ]] || { echo "ERROR: missing $CASES" >&2; exit 1; }
[[ -f "$RUNNER" ]] || { echo "ERROR: missing $RUNNER" >&2; exit 1; }

CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN=/Scr/yunsenz2/data/software/miniconda3/bin/conda
fi
[[ -x "$CONDA_BIN" ]] || { echo "ERROR: conda not found" >&2; exit 1; }
CONDA_BASE=$($CONDA_BIN info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${CHEMEM_CONDA_ENV:-chemem}"

row_count=$(python -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$CASES")
if [[ "$row_count" != "79" ]]; then
  echo "ERROR: expected 79 preflighted rows, found $row_count" >&2
  exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
echo "CHEMEM_AUTHORITATIVE174_EXPANSION [$START_INDEX,$END_INDEX) START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python -u "$RUNNER" \
  --cases "$CASES" \
  --start "$START_INDEX" \
  --end "$END_INDEX" \
  --chemem-bin chemem \
  --timeout-hours "${CHEMEM_CASE_TIMEOUT_HOURS:-12}"
status=$?
echo "CHEMEM_AUTHORITATIVE174_EXPANSION [$START_INDEX,$END_INDEX) END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
