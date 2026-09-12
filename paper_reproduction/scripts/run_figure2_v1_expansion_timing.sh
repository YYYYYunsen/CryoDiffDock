#!/usr/bin/env bash
set -euo pipefail

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
V1_ROOT=${V1_ROOT:-$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
COHORT_IDS=${COHORT_IDS:?Set COHORT_IDS to the frozen external ID list}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new V1 timing directory}

PYTHON_BIN="$ENV_BIN/python"
MANIFEST="$V1_ROOT/02_prepared/inference_blind.csv"
SCORES="$V1_ROOT/04_ncc/pose_scores.csv"
FROZEN_SUMMARY="$V1_ROOT/06_s0_frozen/summary.json"
SEARCH_ROOT="$OUTPUT_ROOT/search"
SELECTED_ROOT="$OUTPUT_ROOT/selected"
TIME_FILE="$OUTPUT_ROOT/v1_expansion_selection.time"

for required in "$PYTHON_BIN" "$MANIFEST" "$SCORES" "$FROZEN_SUMMARY" "$COHORT_IDS"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input does not exist: $required" >&2
    exit 2
  fi
done
if [[ -e "$SEARCH_ROOT" || -e "$SELECTED_ROOT" || -e "$TIME_FILE" ]]; then
  echo "Refusing to overwrite an existing V1 timing run under $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT"

FROZEN_CONFIG=$(
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["frozen_config"])' "$FROZEN_SUMMARY"
)
if [[ ! -f "$FROZEN_CONFIG" ]]; then
  echo "Frozen V1 config does not exist: $FROZEN_CONFIG" >&2
  exit 2
fi
COMPLEXES=$(awk '{sub(/#.*/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); if(length($0)) printf "%s%s", sep, toupper($0); sep=","} END{print ""}' "$COHORT_IDS")
if [[ -z "$COMPLEXES" ]]; then
  echo "Cohort is empty: $COHORT_IDS" >&2
  exit 2
fi

export PYTHON_BIN POCKET_ROOT MANIFEST SCORES SEARCH_ROOT SELECTED_ROOT FROZEN_CONFIG COMPLEXES
cd "$POCKET_ROOT"
/usr/bin/time -p -o "$TIME_FILE" bash -c '
  set -euo pipefail
  "$PYTHON_BIN" -u "$POCKET_ROOT/scripts/run_discrete_density_search.py" \
    --manifest "$MANIFEST" \
    --scores "$SCORES" \
    --output-dir "$SEARCH_ROOT" \
    --complexes "$COMPLEXES" \
    --density-column ncc_sigma_1p5 \
    --density-sigma 1.5 \
    --density-weight 2.0 \
    --parent-score-column confidence \
    --parent-count 6 \
    --top-score-parents 3 \
    --parent-diversity-rmsd 1.5 \
    --rotation-degrees 15,30,60 \
    --translation-distances 0.5,1.0 \
    --torsion-degrees 60,120,180 \
    --max-torsion-bonds 8 \
    --minimum-ncc-gain 0 \
    --anchor-penalty 0.05 \
    --max-centroid-shift 1.5 \
    --min-in-bounds-fraction 0.95 \
    --severe-overlap 0.8 \
    --mask-radius 10 \
    --materialize-expanded-limit 0 \
    --fail-on-error
  "$PYTHON_BIN" "$POCKET_ROOT/scripts/apply_frozen_discrete_ranking.py" \
    --search-root "$SEARCH_ROOT" \
    --frozen-config "$FROZEN_CONFIG" \
    --output-dir "$SELECTED_ROOT"
'

echo "V1_FIGURE2_TIMING_OK: $TIME_FILE"
