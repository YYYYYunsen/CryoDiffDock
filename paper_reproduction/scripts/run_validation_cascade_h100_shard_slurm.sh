#!/usr/bin/env bash
#SBATCH --job-name=valh100
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --export=ALL

# Resource directives above are H100-specific. The shared worker enforces one
# visible GPU and runs one parent shard through Global -> V2.
set -euo pipefail
POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
export VALCAS_PYTHON=${VALCAS_H100_PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em-h100/bin/python}
exec bash "$POCKET_ROOT/scripts/run_validation_cascade_a6000_shard_slurm.sh" "$@"
