#!/usr/bin/env bash

fail() { echo "ERROR: $*" >&2; exit 2; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
GLOBAL_ROOT=${GLOBAL_ROOT:-/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM}
PYTHON_BIN=${PYTHON_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}
AUTHORITATIVE_ROOT=$POCKET_ROOT/artifacts/figure2_authoritative174_v1
OLD_EMERALD_ROOT=$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1
OLD_CHEMEM_ANALYSIS=$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1/02_live_analysis
RECOVERY_ROOT=${EMERALD_SITECENTER_ROOT:-$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike174_sitecenter_recovery_v1}

AUTHORITATIVE_IDS=$POCKET_ROOT/paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt
AUTHORITATIVE_MANIFEST=$AUTHORITATIVE_ROOT/02_worklists/authoritative174_manifest.csv
HISTORICAL_SUMMARY=$OLD_EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_final.tsv
HISTORICAL_MANIFEST=$OLD_CHEMEM_ANALYSIS/per_complex_benchmark.csv
PREPARER=$POCKET_ROOT/scripts/prepare_figure2_emerald_sitecenter_recovery.py

for required in "$PYTHON_BIN" "$AUTHORITATIVE_IDS" "$AUTHORITATIVE_MANIFEST" "$HISTORICAL_MANIFEST" "$HISTORICAL_SUMMARY" "$PREPARER"; do
  [[ -f "$required" ]] || fail "missing $required"
done

mkdir -p "$RECOVERY_ROOT"
cd "$POCKET_ROOT" || fail "cannot enter $POCKET_ROOT"

"$PYTHON_BIN" "$PREPARER" \
  --authoritative-ids "$AUTHORITATIVE_IDS" \
  --authoritative-manifest "$AUTHORITATIVE_MANIFEST" \
  --historical-manifest "$HISTORICAL_MANIFEST" \
  --historical-summary "$HISTORICAL_SUMMARY" \
  --output-dir "$RECOVERY_ROOT" \
  --expected-n 174 \
  --expected-near-native 15
status=$?
if [[ "$status" != "0" ]]; then
  fail "site-center worklist preparation failed with status $status"
fi

echo "Standard site-center IDs (safe to submit now):"
nl -ba "$RECOVERY_ROOT/sitecenter_standard_ids.txt"
echo
echo "Input-compatibility gate (do not submit with the standard runner yet):"
cat "$RECOVERY_ROOT/sitecenter_input_compat_ids.txt"
echo
echo "Retained technical failure:"
cat "$RECOVERY_ROOT/sitecenter_retained_failure_ids.txt"
echo
echo "EMERALD_SITECENTER_WORKLIST_OK: $RECOVERY_ROOT"
