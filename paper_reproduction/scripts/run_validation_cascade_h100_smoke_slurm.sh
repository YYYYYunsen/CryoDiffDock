#!/usr/bin/env bash
#SBATCH --job-name=valhsmoke
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
#SBATCH --output=valhsmoke-%j.out
#SBATCH --error=valhsmoke-%j.err

set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }
POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
PYTHON_BIN=${VALCAS_H100_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em-h100/bin/python}
CONFIG=${VALCAS_CONFIG:-$POCKET_ROOT/configs/validation_cascade.json}
REUSE=${VALCAS_H100_SMOKE_REUSE:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_esmfix_v1}
OUTPUT=${VALCAS_H100_SMOKE_OUTPUT:-$POCKET_ROOT/artifacts/paper_validation/global_to_local_druglike_v1/auto_smoke_h100_env_v1}

[[ -n ${SLURM_JOB_ID:-} ]] || fail "submit this script with sbatch"
for required in "$PYTHON_BIN" "$CONFIG" "$REUSE/AUTO_SUCCESS.json"; do
  [[ -e $required ]] || fail "missing required input: $required"
done
[[ ! -e $OUTPUT ]] || fail "refusing to overwrite H100 smoke: $OUTPUT"
cd "$POCKET_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUM_GPUS=1 PYTHONUNBUFFERED=1

"$PYTHON_BIN" - <<'PY'
import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA runtime:', torch.version.cuda)
print('compiled arches:', torch.cuda.get_arch_list())
assert 'sm_90' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x=torch.randn(16,3,device='cuda')
edge=radius_graph(x,r=2.0,max_num_neighbors=16)
out=scatter(x,torch.arange(16,device='cuda'),dim=0)
torch.cuda.synchronize()
print('H100_TORCH_PYG_KERNEL_OK',tuple(edge.shape),tuple(out.shape))
PY

"$PYTHON_BIN" -u scripts/run_validation_cascade_auto.py \
  --config "$CONFIG" --mode smoke --case 6DZW --gpus-per-job 1 \
  --reuse-global-from "$REUSE" --output-root "$OUTPUT"
[[ -f $OUTPUT/AUTO_SUCCESS.json ]] || fail "H100 end-to-end smoke did not finish"
echo "H100_LOCAL_CASCADE_SMOKE_OK: $OUTPUT"
