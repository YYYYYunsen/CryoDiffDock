#!/usr/bin/env bash
set -euo pipefail

# Audit one pretrained DiffDock model against the frozen CryoLigandDB 174-case
# drug-like test cohort. Override the paths below to match the server layout.

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
DATA_ROOT=${DATA_ROOT:-/Scr/yunsenz2/data/EM-dataset-final/data}

MODEL_NAME=${MODEL_NAME:?Set MODEL_NAME to DiffDock-L or DiffDock-Pocket}
TRAIN_RECEPTOR_ROOT=${TRAIN_RECEPTOR_ROOT:?Set TRAIN_RECEPTOR_ROOT to the PDBBind directory used for model training}
TRAIN_PROTEIN_PATTERN=${TRAIN_PROTEIN_PATTERN:-}
CHECKPOINT=${CHECKPOINT:-}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/artifacts/paper_test/pretraining_exposure/${MODEL_NAME}}

COHORT=${COHORT:-$PROJECT_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt}
TRAIN_SPLIT=${TRAIN_SPLIT:-$PROJECT_ROOT/data/splits/timesplit_no_lig_overlap_train}
TEST_RAW_ROOT=${TEST_RAW_ROOT:-/Scr/yunsenz2/data/EM-dataset-all/data/raw}
TEST_LIGAND_ROOT=${TEST_LIGAND_ROOT:-$DATA_ROOT/deposited_ligand_references_complete_global_v1}
WORKERS=${WORKERS:-16}
REQUIRE_TRAINING_LIGAND=${REQUIRE_TRAINING_LIGAND:-1}

for required_file in "$COHORT" "$TRAIN_SPLIT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required input file does not exist: $required_file" >&2
    exit 2
  fi
done

for required_directory in "$TRAIN_RECEPTOR_ROOT" "$TEST_RAW_ROOT" "$TEST_LIGAND_ROOT"; do
  if [[ ! -d "$required_directory" ]]; then
    echo "Required input directory does not exist: $required_directory" >&2
    exit 2
  fi
done

checkpoint_args=()
if [[ -n "$CHECKPOINT" ]]; then
  checkpoint_args=(--checkpoint "$CHECKPOINT")
fi

protein_args=()
if [[ -n "$TRAIN_PROTEIN_PATTERN" ]]; then
  protein_args+=(--train-protein-pattern "$TRAIN_PROTEIN_PATTERN")
fi
protein_args+=(
  --train-protein-pattern '{pdb_id_lower}/{pdb_id_lower}_protein_esmfold_aligned_tr_fix.pdb'
  --train-protein-pattern '{pdb_id}/{pdb_id}_protein_esmfold_aligned_tr_fix.pdb'
  --train-protein-pattern '{pdb_id_lower}/{pdb_id_lower}_protein_processed_fix.pdb'
  --train-protein-pattern '{pdb_id}/{pdb_id}_protein_processed_fix.pdb'
  --train-protein-pattern '{pdb_id_lower}/{pdb_id_lower}_protein_processed.pdb'
  --train-protein-pattern '{pdb_id}/{pdb_id}_protein_processed.pdb'
  --train-protein-pattern '{pdb_id_lower}/{pdb_id_lower}_protein_obabel_reduce.pdb'
  --train-protein-pattern '{pdb_id}/{pdb_id}_protein_obabel_reduce.pdb'
  --train-protein-pattern '{pdb_id_lower}/{pdb_id_lower}_protein.pdb'
  --train-protein-pattern '{pdb_id}/{pdb_id}_protein.pdb'
)

training_ligand_args=()
if [[ "$REQUIRE_TRAINING_LIGAND" == "1" ]]; then
  training_ligand_args=(--require-training-ligand)
fi

cd "$PROJECT_ROOT"
python -u evaluation/audit_diffdock_pretraining_exposure.py \
  --stage all \
  --model-name "$MODEL_NAME" \
  --cohort "$COHORT" \
  --train-split "$TRAIN_SPLIT" \
  --train-receptor-root "$TRAIN_RECEPTOR_ROOT" \
  "${protein_args[@]}" \
  --train-ligand-pattern '{pdb_id}/{pdb_id}_ligand.sdf' \
  --train-ligand-pattern '{pdb_id_lower}/{pdb_id_lower}_ligand.sdf' \
  --train-ligand-pattern '{pdb_id}/{pdb_id}_ligand.mol2' \
  --train-ligand-pattern '{pdb_id_lower}/{pdb_id_lower}_ligand.mol2' \
  --test-raw-root "$TEST_RAW_ROOT" \
  --test-ligand-root "$TEST_LIGAND_ROOT" \
  --contact-cutoff 6 \
  --minimum-query-coverage 80 \
  --minimum-target-coverage 80 \
  --minimum-alignment-length 30 \
  "${training_ligand_args[@]}" \
  --workers "$WORKERS" \
  --mmseqs-threads "$WORKERS" \
  --require-complete \
  --output-dir "$OUTPUT_ROOT" \
  "${checkpoint_args[@]}"

echo "PRETRAINING_EXPOSURE_RUN_COMPLETE: $OUTPUT_ROOT"
