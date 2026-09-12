#!/usr/bin/env bash
# Preflight 79 newly required ChemEM cases. EMERALD has a separate 80-case list;
# 9D6D is a reusable historical ChemEM timeout but is new for EMERALD.

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
CHEMEM_ENV_BIN=${CHEMEM_ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/chemem/bin}
RUN_ROOT=${RUN_ROOT:-$POCKET_ROOT/artifacts/figure2_authoritative174_v1}
CHEMEM_ROOT=${CHEMEM_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike174_expansion79_v1}
WORKLIST_DIR=$RUN_ROOT/02_worklists
PYTHON_BIN=$CHEMEM_ENV_BIN/python

for required in \
  "$PYTHON_BIN" \
  "$WORKLIST_DIR/run_chemem_new79_ids.txt" \
  "$WORKLIST_DIR/authoritative174_manifest.csv" \
  "$POCKET_ROOT/scripts/prepare_chemem100_stage1.py"; do
  [[ -e "$required" ]] || fail "Missing required input: $required"
done

run mkdir -p "$CHEMEM_ROOT"
cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"

run "$PYTHON_BIN" -u scripts/prepare_chemem100_stage1.py \
  --cohort "$WORKLIST_DIR/authoritative174_manifest.csv" \
  --new-case-ids "$WORKLIST_DIR/run_chemem_new79_ids.txt" \
  --local-manifest "$WORKLIST_DIR/authoritative174_manifest.csv" \
  --output-root "$CHEMEM_ROOT/01_stage1" \
  --platform OpenCL \
  --n-cpu 8 \
  --ph 7.0 \
  --no-default-repairs \
  --expected-new-count 79 \
  --expected-repair-count 0 \
  --new-case-group authoritative174_expansion79

echo "FIGURE2_AUTHORITATIVE174_EXTERNAL_PREFLIGHT_OK"
echo "ChemEM cases: $CHEMEM_ROOT/01_stage1/cases_to_run.csv"
echo "EMERALD IDs: $WORKLIST_DIR/run_emerald_new80_ids.txt"
echo "EMERALD manifest: $WORKLIST_DIR/authoritative174_manifest.csv"
