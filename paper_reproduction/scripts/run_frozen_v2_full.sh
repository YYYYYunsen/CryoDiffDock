#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
V1_ROOT=${V1_ROOT:-$PROJECT_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
V2_ROOT=${V2_ROOT:-$PROJECT_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
COHORT_FILE=${COHORT_FILE:-$PROJECT_ROOT/visualizations/figure3_local/data/figure3_frozen_drug_like_cohort.txt}

MANIFEST=$V1_ROOT/02_prepared/inference_blind.csv
REFERENCE_MANIFEST=$V1_ROOT/02_prepared/inference.csv
POSE_SCORES=$V1_ROOT/04_ncc/pose_scores.csv
PYTHON_BIN=$ENV_BIN/python

cd "$PROJECT_ROOT"

for required in "$MANIFEST" "$REFERENCE_MANIFEST" "$POSE_SCORES" "$COHORT_FILE" "$PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input does not exist: $required" >&2
    exit 2
  fi
done
if [[ -e "$V2_ROOT" ]]; then
  echo "Refusing to overwrite existing V2_ROOT: $V2_ROOT" >&2
  exit 2
fi

COHORT_AUDIT=$(
  "$PYTHON_BIN" -c '
from pathlib import Path
import pandas as pd
import sys
cohort_path, manifest_path, scores_path = map(Path, sys.argv[1:])
cohort = [line.strip().upper() for line in cohort_path.read_text().splitlines() if line.strip()]
manifest = set(pd.read_csv(manifest_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
scores = set(pd.read_csv(scores_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
if len(cohort) != len(set(cohort)):
    raise SystemExit("Frozen cohort contains duplicate IDs")
missing_manifest = sorted(set(cohort) - manifest)
missing_scores = sorted(set(cohort) - scores)
if missing_manifest or missing_scores:
    raise SystemExit(f"Frozen cohort mismatch: missing_manifest={missing_manifest}, missing_scores={missing_scores}")
print(str(len(cohort)) + "|" + ",".join(cohort))
' "$COHORT_FILE" "$MANIFEST" "$POSE_SCORES"
)
NUM_COMPLEXES=${COHORT_AUDIT%%|*}
COMPLEXES=${COHORT_AUDIT#*|}
if [[ "$NUM_COMPLEXES" != "171" ]]; then
  echo "Expected the frozen 171-complex drug-like cohort, found $NUM_COMPLEXES" >&2
  exit 2
fi

echo "FROZEN_V2_DRUGLIKE171_START: $NUM_COMPLEXES complexes"
echo "V1_ROOT: $V1_ROOT"
echo "V2_ROOT: $V2_ROOT"
echo "COHORT_FILE: $COHORT_FILE"
echo "Started UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "${AUDIT_ONLY:-0}" == "1" ]]; then
  echo "FROZEN_V2_DRUGLIKE171_AUDIT_OK"
  exit 0
fi

"$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
  --manifest "$MANIFEST" \
  --scores "$POSE_SCORES" \
  --output-dir "$V2_ROOT" \
  --complexes "$COMPLEXES" \
  --device cuda \
  --density-sigmas 1.5,2.25 \
  --parent-count 6 \
  --search-depth 3 \
  --beam-width 48 \
  --beam-width-per-parent 10 \
  --coarse-beam-fraction 0.25 \
  --translation-steps 0.25,0.5 \
  --rotation-steps 10,20 \
  --torsion-steps 30,60 \
  --top-torsion-gradients 2 \
  --materialize-expanded-limit 0 \
  --fail-on-error

PATH="$ENV_BIN:$PATH" \
SEARCH_ROOT="$V2_ROOT" \
V1_ROOT="$V1_ROOT" \
REFERENCE_MANIFEST="$REFERENCE_MANIFEST" \
bash scripts/evaluate_adaptive_density_search.sh

echo "FROZEN_V2_DRUGLIKE171_COMPLETE: $V2_ROOT"
echo "Completed UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
