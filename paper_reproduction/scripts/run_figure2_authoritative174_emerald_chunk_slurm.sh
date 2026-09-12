#!/usr/bin/env bash
#SBATCH --job-name=emer174x
#SBATCH --partition=titanv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --export=ALL
#SBATCH --output=emer174x-%j.out
#SBATCH --error=emer174x-%j.err

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
if (( START_INDEX < 0 || END_INDEX <= START_INDEX || END_INDEX > 80 )); then
  echo "ERROR: require 0 <= start < end <= 80" >&2
  exit 2
fi

# Pin authoritative software/data roots.  Slurm exports the submission shell's
# environment, which may contain stale variables from earlier recovery runs.
POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
TOOLS_ROOT=/Scr/yunsenz2/data/software/benchmark/tools/local_docking
RUN_ROOT=${EMERALD_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_v1}
MASTER_IDS=$POCKET_ROOT/artifacts/figure2_authoritative174_v1/02_worklists/run_emerald_new80_ids.txt
INPUT_MANIFEST=$POCKET_ROOT/artifacts/figure2_authoritative174_v1/02_worklists/authoritative174_manifest.csv
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

CHUNK_LABEL=$(printf '%03d_%03d' "$START_INDEX" "$END_INDEX")
CHUNK_DIR=$RUN_ROOT/chunks/$CHUNK_LABEL
CHUNK_IDS=$CHUNK_DIR/ids.txt
mkdir -p "$CHUNK_DIR"

# The runner locates emerald_paper_driver.sh beside itself.  Pair it with a
# wrapper which preserves Rosetta's MOL2 layout and, only for an invalid
# all-zero conformer, replaces coordinates with a deterministic SMILES-only
# RDKit conformer. Atom order, bonds, types, and input charges remain unchanged.
RUNNER_BUNDLE=$CHUNK_DIR/runner_bundle
mkdir -p "$RUNNER_BUNDLE"
cp "$BASE_RUNNER" "$RUNNER_BUNDLE/run_emerald_paper_parallel.sh"
cp "$COMPAT_DRIVER" "$RUNNER_BUNDLE/emerald_paper_driver.sh"
RUNNER=$RUNNER_BUNDLE/run_emerald_paper_parallel.sh

awk -v start="$START_INDEX" -v end="$END_INDEX" '
  NF { if (n >= start && n < end) print toupper($1); n++ }
' "$MASTER_IDS" > "$CHUNK_IDS"
selected_count=$(wc -l < "$CHUNK_IDS" | tr -d ' ')
if (( selected_count != END_INDEX - START_INDEX )); then
  echo "ERROR: selected $selected_count cases, expected $((END_INDEX - START_INDEX))" >&2
  exit 1
fi

cpus=${SLURM_CPUS_PER_TASK:-24}
trajectory_jobs=3
capacity=$((cpus / trajectory_jobs))
complex_jobs=$selected_count
if (( complex_jobs > capacity )); then complex_jobs=$capacity; fi
if [[ -n "${EMERALD_COMPLEX_JOBS_PER_NODE:-}" ]]; then
  if [[ ! "$EMERALD_COMPLEX_JOBS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: EMERALD_COMPLEX_JOBS_PER_NODE must be a positive integer" >&2
    exit 2
  fi
  if (( EMERALD_COMPLEX_JOBS_PER_NODE > capacity )); then
    echo "ERROR: EMERALD_COMPLEX_JOBS_PER_NODE exceeds CPU capacity $capacity" >&2
    exit 2
  fi
  complex_jobs=$EMERALD_COMPLEX_JOBS_PER_NODE
fi

export TEST_IDS_FILE=$CHUNK_IDS
export WORKDIR=$CHUNK_DIR
export INPUT_MANIFEST
export N_COMPLEX_JOBS=$complex_jobs
export EMERALD_PROTEIN_COLUMN=mask_structure
export EMERALD_TRAJECTORIES=3
export EMERALD_TRAJECTORY_JOBS=$trajectory_jobs
export EMERALD_TOPK=5
export EMERALD_RANDOM_SEED=2026
export EMERALD_INPUT_CONFORMER_MODE=smiles_3d
export EMERALD_START_POSITION_MODE=site_center
export EMERALD_INITIAL_TRANSLATION_MAX_A=2.0
# Match the actual protocol used by the old frozen-99 runs: retain the partial
# charges already present in the OpenBabel-generated input MOL2.  Do not allow
# an inherited submission-shell variable to silently mix charge protocols.
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "EMERALD_AUTHORITATIVE174_EXPANSION [$START_INDEX,$END_INDEX) START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "EMERALD helper environment audit"
echo "  conda.sh:          $CONDA_SH"
echo "  requested env:     $HELPER_CONDA_ENV"
echo "  mol2genparams:     $MOL2GENPARAMS_PY"
echo "  ligand utility:    $LIGAND_UTIL_PY"
echo "  charge protocol:   input MOL2 partial charges (EMERALD_AM1BCC=0)"
echo "  conformer fallback: deterministic SMILES-only RDKit ETKDGv3 for invalid all-zero MOL2"
bash -lc "source \"$CONDA_SH\" && conda activate \"$HELPER_CONDA_ENV\" && echo \"  active prefix:     \$CONDA_PREFIX\" && echo \"  python:            \$(command -v python3)\" && python3 -c 'import sys, numpy, scipy; print(\"  Python version:    \" + sys.version.replace(chr(10), \" \")); print(\"  NumPy version:     \" + numpy.__version__); print(\"  SciPy version:     \" + scipy.__version__); import numba; print(\"  Numba version:     \" + numba.__version__)' && echo \"  OpenBabel:         \$(command -v obabel)\" && obabel -V"
env_status=$?
if [[ "$env_status" != "0" ]]; then
  echo "ERROR: failed to activate or audit helper environment" >&2
  exit "$env_status"
fi
"$RDKIT_PYTHON" -c 'import rdkit; print("  fallback RDKit:    " + rdkit.__version__)'
rdkit_status=$?
if [[ "$rdkit_status" != "0" ]]; then
  echo "ERROR: RDKit fallback environment is unavailable" >&2
  exit "$rdkit_status"
fi
sha256sum "$RUNNER" "$REAL_DRIVER" "$REAL_LIGAND_UTIL" "$COMPAT_DRIVER" "$COMPAT_LIGAND_UTIL" "$RDKIT_FALLBACK" "$MOL2GENPARAMS_PY"
bash "$RUNNER"
status=$?
echo "EMERALD_AUTHORITATIVE174_EXPANSION [$START_INDEX,$END_INDEX) END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$status" == "2" ]]; then
  echo "WARNING: runner recorded one or more case failures"
  exit 0
fi
exit "$status"
