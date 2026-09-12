#!/usr/bin/env bash
# Resume confidence ranking only for Test174 ablation cases that have raw poses.
set -euo pipefail

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}
RESULT_ROOT=${RESULT_ROOT:-$GLOBAL_ROOT/results_test_deposited_cofactor35_contact_entity_emdb_v1}
COHORT=${COHORT:-$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
AUDIT_DIR=${AUDIT_DIR:-$RESULT_ROOT/confidence_analysis/ablation_orig_conf_audit_test174_v1}
TARGET=${TARGET:-$RESULT_ROOT/confidence_analysis/conf_scored/ablation_orig_conf}
RESUME_ROOT=${RESUME_ROOT:-$RESULT_ROOT/confidence_analysis/ablation_orig_conf_resume_test174_v2}
ORIG_CONF_DIR=${ORIG_CONF_DIR:-/Scr/yunsenz2/data/software/DiffDock/workdir/v1.1/confidence_model}
ORIG_CONF_CKPT=${ORIG_CONF_CKPT:-best_model.pt}
ESM_EMBEDDINGS_PATH=${ESM_EMBEDDINGS_PATH:-$GLOBAL_ROOT/data/esm_embeddings.pt}
CONFIDENCE_ENV=${CONFIDENCE_ENV:-diffdock}

required=(
  "$COHORT"
  "$AUDIT_DIR/ablation_confidence_audit.csv"
  "$RESULT_ROOT/test_input_em_ring.csv"
  "$GLOBAL_ROOT/compare_confidence_rerank.py"
  "$ORIG_CONF_DIR/$ORIG_CONF_CKPT"
  "$ESM_EMBEDDINGS_PATH"
)
for path in "${required[@]}"; do
  [[ -f $path ]] || { echo "ERROR: missing input: $path" >&2; exit 2; }
done
[[ -d $RESULT_ROOT/ablation ]] || { echo "ERROR: missing raw ablation root" >&2; exit 2; }
[[ ! -e $RESUME_ROOT ]] || { echo "ERROR: resume output exists; use a new RESUME_ROOT: $RESUME_ROOT" >&2; exit 2; }

mkdir -p "$RESUME_ROOT/inputs"
python - "$AUDIT_DIR/ablation_confidence_audit.csv" "$RESUME_ROOT/inputs/scoreable_ids.txt" <<'PY'
import csv, sys
source, output = sys.argv[1:]
with open(source, newline="") as handle:
    rows = list(csv.DictReader(handle))
ids = [
    row["pdb_id"] for row in rows
    if row["status"] != "complete" and int(row["raw_pose_n"]) > 0
]
if not ids:
    raise SystemExit("No incomplete cases with raw poses; nothing to resume")
with open(output, "w") as handle:
    handle.writelines(f"{pdb_id}\n" for pdb_id in ids)
print(f"Scoreable confidence resumes ({len(ids)}): {' '.join(ids)}")
PY

# compare_confidence_rerank builds its dataset from protein_ligand_csv.  The
# split file alone does not reduce that dataset, so both inputs must be sealed
# to the resume IDs.
python - "$RESULT_ROOT/test_input_em_ring.csv" \
  "$RESUME_ROOT/inputs/scoreable_ids.txt" \
  "$RESUME_ROOT/inputs/test_input_em_ring.csv" <<'PY'
import csv, sys
source, ids_path, output = sys.argv[1:]
ids = [line.strip().upper() for line in open(ids_path) if line.strip()]
with open(source, newline="") as handle:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames or []
    id_column = next(
        (name for name in ("complex_name", "pdb_id", "complex", "name", "TEST_PDB_ID") if name in fields),
        None,
    )
    if id_column is None:
        raise SystemExit(f"Cannot identify complex ID column in {source}: {fields}")
    indexed = {}
    for row in reader:
        key = str(row[id_column]).strip().upper()
        if key in ids:
            if key in indexed:
                raise SystemExit(f"Duplicate input row for {key} in {source}")
            indexed[key] = row
missing = [key for key in ids if key not in indexed]
if missing:
    raise SystemExit(f"Resume IDs absent from {source}: {missing}")
with open(output, "w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(indexed[key] for key in ids)
print(f"SEALED_RESUME_INPUT_OK: {len(ids)} rows -> {output}")
PY

while read -r pdb_id; do
  [[ ! -e $TARGET/$pdb_id ]] || {
    echo "ERROR: target already exists for $pdb_id; audit or move it explicitly before resuming" >&2
    exit 2
  }
done < "$RESUME_ROOT/inputs/scoreable_ids.txt"

python "$POCKET_ROOT/scripts/check_global_confidence_inputs.py" \
  --cohort "$RESUME_ROOT/inputs/scoreable_ids.txt" \
  --poses "$RESULT_ROOT/ablation"

echo "Running original DiffDock confidence model for the missing ablation cases"
conda run --no-capture-output -n "$CONFIDENCE_ENV" python "$GLOBAL_ROOT/compare_confidence_rerank.py" \
  --protein_ligand_csv "$RESUME_ROOT/inputs/test_input_em_ring.csv" \
  --test_split "$RESUME_ROOT/inputs/scoreable_ids.txt" \
  --data_dir "$DATA_ROOT/processed_6radii_deposited" \
  --results_dir "$RESUME_ROOT/conf_scored" \
  --pretrained_poses "$RESULT_ROOT/pretrained" \
  --ablation_poses "$RESULT_ROOT/ablation" \
  --finetuned_poses "$RESULT_ROOT/finetuned" \
  --orig_conf_dir "$ORIG_CONF_DIR" \
  --orig_conf_ckpt "$ORIG_CONF_CKPT" \
  --esm_embeddings_path "$ESM_EMBEDDINGS_PATH" \
  --em_conf_dir "$ORIG_CONF_DIR" \
  --pose_set ablation \
  --conf_model orig \
  --max_poses 40

STAGED=$RESUME_ROOT/conf_scored/ablation_orig_conf
python "$POCKET_ROOT/scripts/audit_diffdock_ablation_confidence.py" \
  --cohort "$RESUME_ROOT/inputs/scoreable_ids.txt" \
  --raw-poses "$RESULT_ROOT/ablation" \
  --confidence-poses "$STAGED" \
  --output-dir "$RESUME_ROOT/staged_audit"

while read -r pdb_id; do
  [[ -d $STAGED/$pdb_id ]] || { echo "ERROR: staged result missing: $pdb_id" >&2; exit 2; }
  cp -a "$STAGED/$pdb_id" "$TARGET/$pdb_id"
done < "$RESUME_ROOT/inputs/scoreable_ids.txt"

python "$POCKET_ROOT/scripts/audit_diffdock_ablation_confidence.py" \
  --cohort "$COHORT" \
  --raw-poses "$RESULT_ROOT/ablation" \
  --confidence-poses "$TARGET" \
  --output-dir "$RESUME_ROOT/final_audit" || status=$?
status=${status:-0}
if [[ $status -ne 1 ]]; then
  exit "$status"
fi
python - "$RESUME_ROOT/final_audit/summary.json" <<'PY'
import json, sys
summary=json.load(open(sys.argv[1]))
if summary["complete_n"] != 173 or summary["incomplete_n"] != 1:
    raise SystemExit(f"Unexpected final coverage: {summary}")
print("ABLATION_CONFIDENCE_RESUME_OK: 173/174 ranked; the one raw-inference failure remains fixed-denominator failure")
PY
