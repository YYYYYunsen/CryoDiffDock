#!/usr/bin/env bash
#SBATCH --job-name=chemproh2
#SBATCH --partition=A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:30:00
#SBATCH --export=ALL
#SBATCH --output=chemproh2-%j.out
#SBATCH --error=chemproh2-%j.err

CASE_ID=${1:-}
CASE_ID=$(printf '%s' "$CASE_ID" | tr '[:lower:]' '[:upper:]')
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: submit with sbatch $0 <CASE_ID>" >&2
  exit 2
fi

POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
RECOVERY_ROOT=${CHEMEM_PRO_H2_RECOVERY_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_pro_h2_recovery_v1}
CASES=$RECOVERY_ROOT/recovery_cases.csv
COMPAT_PYTHON=$RECOVERY_ROOT/compat_python
COMPAT_PRO_JSON=$COMPAT_PYTHON/ChemEM/tools/amino_acids/PRO_hydrogens.json
RUNNER=$POCKET_ROOT/scripts/run_chemem100_cases.py

for required in "$CASES" "$COMPAT_PRO_JSON" "$RUNNER"; do
  [[ -f "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

START_INDEX=$(
  python3 - "$CASES" "$CASE_ID" <<'PY'
import csv
import sys

path, requested = sys.argv[1:3]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle))
matches = [row for row in rows if row.get("complex_name", "").strip().upper() == requested]
if len(matches) != 1:
    raise SystemExit(
        f"ERROR: expected exactly one {requested} row in {path}, found {len(matches)}"
    )
print(int(matches[0]["work_index"]))
PY
)
index_status=$?
if [[ "$index_status" != "0" ]]; then
  exit "$index_status"
fi
END_INDEX=$((START_INDEX + 1))

CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN=/Scr/yunsenz2/data/software/miniconda3/bin/conda
fi
[[ -x "$CONDA_BIN" ]] || { echo "ERROR: conda not found" >&2; exit 1; }
CONDA_BASE=$($CONDA_BIN info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate chemem

# The private package copy wins over site-packages. Existing ChemEM jobs keep
# using the unmodified installed package and are unaffected.
export PYTHONPATH=$COMPAT_PYTHON
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

python - "$COMPAT_PYTHON" <<'PY'
import json
import sys
from pathlib import Path
import ChemEM.tools.openff_tools as openff_tools

expected = (Path(sys.argv[1]) / "ChemEM" / "tools" / "openff_tools.py").resolve()
observed = Path(openff_tools.__file__).resolve()
if observed != expected:
    raise RuntimeError(f"compat import mismatch: expected {expected}, observed {observed}")
mapping_path = observed.parent / "amino_acids" / "PRO_hydrogens.json"
mapping = json.loads(mapping_path.read_text())
if mapping.get("H2") != "N" or mapping.get("H3") != "N":
    raise RuntimeError(f"PRO compatibility mapping is missing: {mapping}")
print("compat openff_tools:", observed)
print("compat PRO mapping:", {key: mapping[key] for key in ("H2", "H3")})
PY
compat_status=$?
if [[ "$compat_status" != "0" ]]; then
  echo "ERROR: isolated ChemEM compatibility preflight failed" >&2
  exit "$compat_status"
fi

echo "CHEMEM_PRO_H2_RECOVERY case=$CASE_ID START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
sha256sum "$COMPAT_PRO_JSON" "$CASES" "$RUNNER"
python -u "$RUNNER" \
  --cases "$CASES" \
  --start "$START_INDEX" \
  --end "$END_INDEX" \
  --chemem-bin chemem \
  --timeout-hours 12
status=$?
echo "CHEMEM_PRO_H2_RECOVERY case=$CASE_ID END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
