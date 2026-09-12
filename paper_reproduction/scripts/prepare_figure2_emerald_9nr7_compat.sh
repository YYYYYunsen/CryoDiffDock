#!/usr/bin/env bash

# Build the isolated, audited 9NR7 receptor input used only for the EMERALD
# site-centred compatibility recovery.  No native ligand pose is read.

POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
PYTHON=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python
SITECENTER_ROOT=${EMERALD_SITECENTER_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_sitecenter_recovery_v1}
INPUT_MANIFEST=${EMERALD_SITECENTER_MANIFEST:-$SITECENTER_ROOT/sitecenter_recovery_manifest.csv}
SOURCE_RECEPTOR=${EMERALD_9NR7_SOURCE_RECEPTOR:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_expansion80_fullpolymer_inputcharges_v1/chunks/060_065/9NR7/preparation/protein_receptor.pdb}
OUTPUT_DIR=$SITECENTER_ROOT/input_compat/9NR7
OUTPUT_RECEPTOR=$OUTPUT_DIR/9NR7_protein_receptor_without_remote_UNK_EG.pdb
OUTPUT_MANIFEST=$SITECENTER_ROOT/sitecenter_recovery_manifest_9nr7_compat.csv
AUDIT=$OUTPUT_DIR/9NR7_remote_UNK_chain_removal_audit.json

for required in "$PYTHON" "$INPUT_MANIFEST" "$SOURCE_RECEPTOR" "$POCKET_ROOT/scripts/prepare_figure2_emerald_9nr7_compat.py"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

"$PYTHON" "$POCKET_ROOT/scripts/prepare_figure2_emerald_9nr7_compat.py" \
  --input-manifest "$INPUT_MANIFEST" \
  --source-receptor "$SOURCE_RECEPTOR" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --output-receptor "$OUTPUT_RECEPTOR" \
  --audit "$AUDIT" \
  --case-id 9NR7 \
  --remove-chain E \
  --remove-chain G \
  --minimum-distance-A 20
status=$?
if [[ "$status" != "0" ]]; then
  echo "ERROR: 9NR7 compatibility preparation failed" >&2
  exit "$status"
fi

echo "EMERALD_9NR7_COMPAT_MANIFEST=$OUTPUT_MANIFEST"
echo "EMERALD_9NR7_COMPAT_AUDIT=$AUDIT"
