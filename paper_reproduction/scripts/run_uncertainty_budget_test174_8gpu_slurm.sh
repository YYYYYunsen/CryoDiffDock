#!/usr/bin/env bash
#SBATCH --job-name=unc174x8
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=unc174x8-%j.out
#SBATCH --error=unc174x8-%j.err

# Resource directives match the verified interactive allocation command:
# srun -p A100 --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=8 --mem=128G ...
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 2; }

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
PYTHON_BIN=${PYTHON_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}
INPUT_CSV=${INPUT_CSV:-$PROJECT_ROOT/artifacts/figure2_authoritative174_v1/02_worklists/authoritative174_manifest_blind.csv}
COHORT=${COHORT:-$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
BUDGET_ROOT=${BUDGET_ROOT:-$PROJECT_ROOT/artifacts/paper_test/local_v2_uncertainty_budget_test174_a100_v1}
LAUNCHER=$PROJECT_ROOT/scripts/launch_uncertainty_budget_8gpu.sh
EXPECTED_COHORT_SHA256=96ae9f3bf3cca1a56b5e5a9451f036fcbe12d35c94140485b709eec875a40e5e

[[ -n ${SLURM_JOB_ID:-} ]] || fail "submit this script with sbatch"
for required in "$PYTHON_BIN" "$INPUT_CSV" "$COHORT" "$LAUNCHER"; do
  [[ -e $required ]] || fail "missing required input: $required"
done
[[ $(sha256sum "$COHORT" | awk '{print $1}') == "$EXPECTED_COHORT_SHA256" ]] || \
  fail "authoritative174 cohort SHA256 mismatch"

cd "$PROJECT_ROOT" || fail "cannot enter $PROJECT_ROOT"
mkdir -p "$BUDGET_ROOT"
if find "$BUDGET_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'replicate*' -print -quit | grep -q .; then
  fail "refusing to overwrite existing replicate directories below $BUDGET_ROOT"
fi

echo "UNCERTAINTY_BUDGET_TEST174_SLURM_START"
echo "UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=${SLURMD_NODENAME:-$(hostname)}"
echo "JOB=$SLURM_JOB_ID PARTITION=${SLURM_JOB_PARTITION:-unknown}"
echo "INPUT=$INPUT_CSV"
echo "COHORT=$COHORT"
echo "OUTPUT=$BUDGET_ROOT"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

"$PYTHON_BIN" - "$INPUT_CSV" "$COHORT" <<'PY'
import sys
from pathlib import Path

import pandas as pd
import torch

manifest_path, cohort_path = map(Path, sys.argv[1:])
manifest = pd.read_csv(manifest_path, dtype=str)
cohort = [
    line.split("#", 1)[0].strip().upper()
    for line in cohort_path.read_text().splitlines()
    if line.split("#", 1)[0].strip()
]
if "complex_name" not in manifest:
    raise SystemExit("blind manifest lacks complex_name")
if "reference_ligand" in manifest:
    raise SystemExit("blind manifest contains reference_ligand")
ids = manifest["complex_name"].astype(str).str.strip().str.upper().tolist()
if len(ids) != 174 or len(ids) != len(set(ids)):
    raise SystemExit(f"expected 174 unique manifest cases, found {len(ids)}")
if set(ids) != set(cohort) or len(cohort) != 174:
    raise SystemExit("blind manifest and authoritative174 cohort differ")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 visible GPUs, torch sees {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(index) for index in range(8)]
if not all("A100" in name.upper() for name in names):
    raise SystemExit(f"expected eight A100 GPUs, found {names}")
print("manifest cases:", len(ids))
print("visible GPUs:", names)
print("TEST174_8GPU_PREFLIGHT_OK")
PY

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

INPUT_CSV="$INPUT_CSV" \
BUDGET_ROOT="$BUDGET_ROOT" \
EXPECTED_CASES=174 \
PROJECT_ROOT="$PROJECT_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
GPU_LIST=0,1,2,3,4,5,6,7 \
SEEDS=20260813,20260814,20260815,20260816,20260817,20260818,20260819,20260820 \
CPU_THREADS_PER_REPLICATE=4 \
bash "$LAUNCHER"

echo "UNCERTAINTY_BUDGET_TEST174_SLURM_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "UNCERTAINTY_BUDGET_TEST174_8GPU_OK: $BUDGET_ROOT"
