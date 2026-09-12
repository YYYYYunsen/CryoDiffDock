#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${SEARCH_ROOT:?Set SEARCH_ROOT to validation search output containing interaction_pool/ or expanded_pool/ plus matching metrics}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SEARCH_ROOT}/interaction_candidate_diagnostic_v1}"
WORKERS="${WORKERS:-8}"

arguments=(
  --search-root "${SEARCH_ROOT}"
  --output-dir "${OUTPUT_ROOT}"
  --workers "${WORKERS}"
  --strict
)
if [[ -n "${VAL_COHORT:-}" ]]; then
  arguments+=(--cohort "${VAL_COHORT}")
fi

python evaluation/diagnose_candidate_interactions.py "${arguments[@]}"

echo "VALIDATION_INTERACTION_DIAGNOSTIC_COMPLETE: ${OUTPUT_ROOT}"
