#!/usr/bin/env bash

POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
EXTERNAL=$GLOBAL_ROOT/artifacts/paper_external
PYTHON=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python

OLD99=$EXTERNAL/emerald_druglike99_fullpolymer_v1
PILOT3=$EXTERNAL/emerald_paper_start_pilot_local
SITECENTER=$EXTERNAL/emerald_druglike174_sitecenter_recovery_v1
EXPANSION=$EXTERNAL/emerald_druglike174_expansion80_fullpolymer_inputcharges_v1
OUTPUT=$POCKET_ROOT/artifacts/figure2_authoritative174_v1/05_emerald_initialization_supplement
ANALYZER=$POCKET_ROOT/scripts/analyze_emerald_initialization_contrast.py

OLD_SUMMARY=$OLD99/final_analysis/emerald_paper_summary_recovery_aware_final.tsv

for required in "$PYTHON" "$ANALYZER" "$OLD_SUMMARY" "$OLD99/chunks" "$PILOT3" "$SITECENTER" "$EXPANSION/emerald_paper_summary.tsv"; do
  if [[ ! -e "$required" ]]; then
    echo "ERROR: missing $required" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT"

extra_args=()
if [[ "${ALLOW_INCOMPLETE_SITECENTER:-0}" == "1" ]]; then
  extra_args+=(--allow-incomplete-sitecenter)
fi

"$PYTHON" "$ANALYZER" \
  --native-summary "historical12=$OLD_SUMMARY" \
  --native-case-root "pilot3=$PILOT3" \
  --sitecenter-source "historical99_original_chunks=$OLD99/chunks" \
  --sitecenter-source "expansion80=$EXPANSION/emerald_paper_summary.tsv" \
  --sitecenter-source "sitecenter_recovery=$SITECENTER" \
  --output-dir "$OUTPUT" \
  --expected-native 15 \
  --expected-historical 12 \
  --expected-pilot 3 \
  --pilot-id 7LEP \
  --pilot-id 7OCE \
  --pilot-id 9HLG \
  --bootstrap-repeats 10000 \
  --seed 20260907 \
  "${extra_args[@]}"
status=$?
if [[ "$status" != "0" ]]; then
  echo "ERROR: EMERALD initialization supplement failed" >&2
  exit "$status"
fi

echo
echo "Per-case table:"
echo "$OUTPUT/supplementary_emerald_initialization_per_case.csv"
echo "Native-perturbed descriptive tables:"
echo "$OUTPUT/supplementary_emerald_native_perturbed_rmsd_summary.csv"
echo "$OUTPUT/supplementary_emerald_native_perturbed_success_summary.csv"
echo "Summary tables:"
echo "$OUTPUT/supplementary_emerald_initialization_rmsd_summary.csv"
echo "$OUTPUT/supplementary_emerald_initialization_success_summary.csv"
