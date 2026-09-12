#!/usr/bin/env bash
#SBATCH --job-name=emersc
#SBATCH --partition=titanv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=emersc-%j.out
#SBATCH --error=emersc-%j.err

CASE_ID=${1:-}
CASE_ID=$(printf '%s' "$CASE_ID" | tr '[:lower:]' '[:upper:]')
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit with sbatch $0 <CASE_ID>" >&2
  exit 2
fi
if [[ ! "$CASE_ID" =~ ^[A-Z0-9]+$ ]]; then
  echo "ERROR: invalid case ID: $CASE_ID" >&2
  exit 2
fi

# Every protocol-defining value is pinned here so inherited shell variables
# cannot mix native-perturbed and site-center runs.
POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
TOOLS_ROOT=/Scr/yunsenz2/data/software/benchmark/tools/local_docking
RUN_ROOT=${EMERALD_SITECENTER_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_sitecenter_recovery_v1}
MASTER_IDS=${EMERALD_SITECENTER_IDS:-$RUN_ROOT/sitecenter_standard_ids.txt}
INPUT_MANIFEST=${EMERALD_SITECENTER_MANIFEST:-$RUN_ROOT/sitecenter_recovery_manifest.csv}
BASE_RUNNER=$TOOLS_ROOT/run_emerald_paper_parallel.sh
REAL_DRIVER=$TOOLS_ROOT/emerald_paper_driver.sh
REAL_LIGAND_UTIL=$TOOLS_ROOT/ligand_pose_utils.py
COMPAT_DRIVER=$POCKET_ROOT/scripts/emerald_paper_driver_compat.sh
COMPAT_LIGAND_UTIL=$POCKET_ROOT/scripts/emerald_ligand_pose_utils_compat.py
RDKIT_FALLBACK=$POCKET_ROOT/scripts/emerald_rdkit_conformer_fallback.py
RDKIT_PYTHON=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python

for required in "$MASTER_IDS" "$INPUT_MANIFEST" "$BASE_RUNNER" "$REAL_DRIVER" "$REAL_LIGAND_UTIL" "$COMPAT_DRIVER" "$COMPAT_LIGAND_UTIL" "$RDKIT_FALLBACK" "$RDKIT_PYTHON"; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

WORK_INDEX=$(awk -v requested="$CASE_ID" '
  NF { if (toupper($1) == requested) { print n; found++ } n++ }
  END { if (found != 1) exit 3 }
' "$MASTER_IDS")
index_status=$?
if [[ "$index_status" != "0" || -z "$WORK_INDEX" ]]; then
  echo "ERROR: expected exactly one $CASE_ID row in $MASTER_IDS" >&2
  exit 2
fi

CHUNK_LABEL=$(printf '%03d_%s' "$WORK_INDEX" "$CASE_ID")
CHUNK_DIR=$RUN_ROOT/chunks/$CHUNK_LABEL
CHUNK_IDS=$CHUNK_DIR/ids.txt
RUNNER_BUNDLE=$CHUNK_DIR/runner_bundle
mkdir -p "$RUNNER_BUNDLE"
printf '%s\n' "$CASE_ID" > "$CHUNK_IDS"
cp "$BASE_RUNNER" "$RUNNER_BUNDLE/run_emerald_paper_parallel.sh"
cp "$COMPAT_DRIVER" "$RUNNER_BUNDLE/emerald_paper_driver.sh"
RUNNER=$RUNNER_BUNDLE/run_emerald_paper_parallel.sh

export TEST_IDS_FILE=$CHUNK_IDS
export WORKDIR=$CHUNK_DIR
export INPUT_MANIFEST
export N_COMPLEX_JOBS=1
export EMERALD_PROTEIN_COLUMN=mask_structure
export EMERALD_TRAJECTORIES=3
export EMERALD_TRAJECTORY_JOBS=3
export EMERALD_TOPK=5
export EMERALD_RANDOM_SEED=2026
export EMERALD_INPUT_CONFORMER_MODE=smiles_3d
export EMERALD_START_POSITION_MODE=site_center
export EMERALD_INITIAL_TRANSLATION_MAX_A=2.0
export EMERALD_AM1BCC=0
export EMERALD_RESUME=1
export EMERALD_CASE_TIMEOUT_HOURS=12

export ROSETTA_HOME=/Scr/yunsenz2/data/software/benchmark/EMERALD/rosetta.binary.ubuntu.release-408
export ROSETTA_BIN=$ROSETTA_HOME/main/source/bin/rosetta_scripts.static.linuxgccrelease
export ROSETTA3_DB=$ROSETTA_HOME/main/database
export MOL2GENPARAMS_PY=$ROSETTA_HOME/main/source/scripts/python/public/generic_potential/mol2genparams.py
export MINICONDA_ROOT=/Scr/yunsenz2/data/software/miniconda3
export CONDA_SH=$MINICONDA_ROOT/etc/profile.d/conda.sh
export HELPER_CONDA_ENV=emring
export EMERALD_REAL_DRIVER=$REAL_DRIVER
export EMERALD_REAL_LIGAND_UTIL_PY=$REAL_LIGAND_UTIL
export LIGAND_UTIL_PY=$COMPAT_LIGAND_UTIL
export EMERALD_ENABLE_RDKIT_FALLBACK=1
export EMERALD_RDKIT_FALLBACK_SCRIPT=$RDKIT_FALLBACK
export EMERALD_RDKIT_PYTHON=$RDKIT_PYTHON
export EMERALD_RDKIT_FALLBACK_SEED=2026
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "EMERALD_SITECENTER case=$CASE_ID work_index=$WORK_INDEX START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PROTOCOL conformer=smiles_3d start=site_center AM1BCC=0 trajectories=3 trajectory_jobs=3 seed=2026"
echo "RESOURCES complex_jobs=1 cpus=${SLURM_CPUS_PER_TASK:-24} memory_request=60G"
bash -lc "source \"$CONDA_SH\" && conda activate \"$HELPER_CONDA_ENV\" && echo \"ACTIVE_PREFIX=\$CONDA_PREFIX\" && command -v python3 && command -v obabel"
env_status=$?
if [[ "$env_status" != "0" ]]; then
  echo "ERROR: failed to activate helper environment" >&2
  exit "$env_status"
fi
sha256sum "$MASTER_IDS" "$INPUT_MANIFEST" "$RUNNER" "$REAL_DRIVER" "$REAL_LIGAND_UTIL" "$COMPAT_DRIVER" "$COMPAT_LIGAND_UTIL" "$RDKIT_FALLBACK" "$MOL2GENPARAMS_PY"

bash "$RUNNER"
status=$?
echo "EMERALD_SITECENTER case=$CASE_ID END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ) runner_status=$status"
if [[ "$status" == "2" ]]; then
  echo "WARNING: runner recorded a case failure" >&2
  exit 0
fi
exit "$status"
