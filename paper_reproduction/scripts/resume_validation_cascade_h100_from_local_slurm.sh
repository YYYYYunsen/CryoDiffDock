#!/usr/bin/env bash
#SBATCH --job-name=valhresume
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }
SHARD_INDEX=${1:-}
SHARD_COUNT=${VALCAS_SHARD_COUNT:-10}
POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
RUN_ROOT=${VALCAS_RECOVERY_ROOT:?Set VALCAS_RECOVERY_ROOT to the existing failed full-run root}
PYTHON_BIN=${VALCAS_H100_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em-h100/bin/python}
SHARD_ROOT=$(printf '%s/shard%02dof%02d' "$RUN_ROOT" "$SHARD_INDEX" "$SHARD_COUNT")
CONFIG=$SHARD_ROOT/run_config.json

[[ -n ${SLURM_JOB_ID:-} ]] || fail "submit this script with sbatch"
[[ $SHARD_INDEX =~ ^[0-9]+$ ]] || fail "invalid shard index"
for required in "$PYTHON_BIN" "$CONFIG" "$SHARD_ROOT/.stage_global_complete" "$SHARD_ROOT/.stage_prepare_complete"; do
  [[ -e $required ]] || fail "cannot resume; missing $required"
done
[[ ! -f $SHARD_ROOT/AUTO_SUCCESS.json ]] || fail "shard is already complete"

# The sm_90-incompatible run died on row zero. Refuse automatic recovery if it
# managed to materialize any result directory, because mixing partial attempts
# would make pose provenance ambiguous.
if find "$SHARD_ROOT/02_inference/shards" -type d -name 'index*___*' -print -quit 2>/dev/null | grep -q .; then
  fail "partial Local result directories exist; audit them before recovery"
fi

cd "$POCKET_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUM_GPUS=1 PYTHONUNBUFFERED=1
"$PYTHON_BIN" - <<'PY'
import json, torch
from torch_cluster import radius_graph
from torch_scatter import scatter
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert 'sm_90' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x=torch.randn(16,3,device='cuda')
edge=radius_graph(x,r=2.0,max_num_neighbors=16)
scatter(x,torch.arange(16,device='cuda'),dim=0)
torch.cuda.synchronize()
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0),'arches':torch.cuda.get_arch_list(),
                  'kernel_edge_shape':list(edge.shape)}))
PY

for stage in local merge ncc v2; do
  if [[ -f $SHARD_ROOT/.stage_${stage}_complete ]]; then
    echo "RECOVERY_SKIP completed stage=$stage"
    continue
  fi
  echo "RECOVERY_STAGE stage=$stage shard=$SHARD_INDEX"
  "$PYTHON_BIN" -u scripts/run_validation_cascade.py "$stage" --config "$CONFIG" \
    2>&1 | tee "$SHARD_ROOT/${stage}_h100_recovery.log"
done

"$PYTHON_BIN" - "$SHARD_ROOT" "$SHARD_INDEX" "$SHARD_COUNT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
root=Path(sys.argv[1]); index=int(sys.argv[2]); count=int(sys.argv[3])
cohort=root.joinpath('inputs/ids.txt').read_text().split()
available=pd.read_csv(root/'02_inference/available/inference_blind_available.csv')
selected=root/'04_v2/merged/v2_selected'
dirs=[p for p in selected.iterdir() if p.is_dir()]
if len(dirs)!=len(available):
    raise ValueError('V2 selected output count mismatch')
for directory in dirs:
    if len(list(directory.glob('rank1_confidence*.sdf')))!=1:
        raise ValueError(f'Invalid final V2 pose: {directory}')
report={'mode':'full','parents':len(cohort),'requested_branches':5*len(cohort),
        'available_branches':len(available),'selected_v2_branches':len(dirs),
        'rmsd_evaluated':False,'status':'complete_through_v2',
        'job_shards':count,'job_shard_index':index,'gpus_per_job':1,
        'parent_ids':cohort,'recovery':'global_and_prepare_reused; local_through_v2_rerun_in_h100_env'}
(root/'AUTO_SUCCESS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
PY
echo "H100_LOCAL_RECOVERY_OK: $SHARD_ROOT"
