#!/usr/bin/env bash
# Prepare the outcome-independent 174-case Figure 2 manifests and work lists.
# This stage performs no docking and is safe to run on a login/CPU node.

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}
RAW_ROOT=${RAW_ROOT:-/Scr/yunsenz2/data/EM-dataset-all/data/raw}
RUN_ROOT=${RUN_ROOT:-$POCKET_ROOT/artifacts/figure2_authoritative174_v1}

AUTH_ROOT=$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1
AUTH_IDS=$AUTH_ROOT/authoritative_druglike174_ids.txt
AUTH_COHORT=$AUTH_ROOT/authoritative_druglike174_cohort.csv
OLD_IDS=${OLD_IDS:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_v1/frozen_99_ids.txt}
OLD_CHEMEM_COHORT=${OLD_CHEMEM_COHORT:-$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1/00_cohort/chemem100_cohort.csv}
SOURCE_DIR=$RUN_ROOT/00_source
PREPARED_DIR=$RUN_ROOT/01_prepared
WORKLIST_DIR=$RUN_ROOT/02_worklists
PYTHON_BIN=$ENV_BIN/python

for required in \
  "$PYTHON_BIN" "$AUTH_IDS" "$AUTH_COHORT" "$OLD_IDS" "$OLD_CHEMEM_COHORT" \
  "$DATA_ROOT/04_maps_fixed.csv" \
  "$POCKET_ROOT/visualizations/figure4_cryoliganddb/data/complex_manifest.csv" \
  "$POCKET_ROOT/scripts/build_oracle_density_source_manifest.py" \
  "$POCKET_ROOT/scripts/add_full_polymer_mask_paths.py" \
  "$POCKET_ROOT/scripts/prepare_oracle_density_dataset.py" \
  "$POCKET_ROOT/scripts/build_figure2_authoritative_worklists.py"; do
  [[ -e "$required" ]] || fail "Missing required input: $required"
done

run mkdir -p "$SOURCE_DIR" "$PREPARED_DIR" "$WORKLIST_DIR"
cd "$POCKET_ROOT" || fail "Cannot enter $POCKET_ROOT"

echo "[1/4] Build the exact authoritative-174 source manifest"
run "$PYTHON_BIN" scripts/build_oracle_density_source_manifest.py \
  --split "$AUTH_IDS" \
  --data-dir "$DATA_ROOT/processed_6radii_deposited" \
  --reference-dir "$DATA_ROOT/deposited_ligand_references_complete_global_v1" \
  --map-csv "$DATA_ROOT/04_maps_fixed.csv" \
  --no-flexible-policy rigid \
  --output "$SOURCE_DIR/source.csv" \
  --limit 0 \
  --flexible-distance 3.5 \
  --fail-on-error

echo "[2/4] Attach complete-polymer structures for masking and EMERALD"
run "$PYTHON_BIN" scripts/add_full_polymer_mask_paths.py \
  --manifest "$SOURCE_DIR/source.csv" \
  --raw-root "$RAW_ROOT" \
  --output "$SOURCE_DIR/source_fullpoly.csv" \
  --fail-on-error

echo "[3/4] Prepare full-polymer-masked density crops"
prepare_args=(
  --manifest "$SOURCE_DIR/source_fullpoly.csv"
  --output-dir "$PREPARED_DIR"
  --box-size 28
  --spacing 0.75
  --mask-policy full_polymer
  --mask-style soft
  --mask-inner 1.5
  --mask-outer 2.5
  --fail-on-error
)
if [[ "${PREPARE_OVERWRITE:-0}" == "1" ]]; then
  prepare_args+=(--overwrite)
fi
run "$PYTHON_BIN" scripts/prepare_oracle_density_dataset.py "${prepare_args[@]}"

echo "[4/4] Freeze 174/94/80/5 work lists and blind/reveal manifests"
run "$PYTHON_BIN" scripts/build_figure2_authoritative_worklists.py \
  --authoritative-ids "$AUTH_IDS" \
  --authoritative-cohort "$AUTH_COHORT" \
  --previous-ids "$OLD_IDS" \
  --previous-chemem-cohort "$OLD_CHEMEM_COHORT" \
  --prepared-manifest "$PREPARED_DIR/inference.csv" \
  --resolution-table "$POCKET_ROOT/visualizations/figure4_cryoliganddb/data/complex_manifest.csv" \
  --output-dir "$WORKLIST_DIR"

echo "FIGURE2_AUTHORITATIVE174_PREPARED: $RUN_ROOT"
