#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${MANIFEST:?Set MANIFEST to the frozen validation blind manifest}"
: "${REFERENCE_MANIFEST:?Set REFERENCE_MANIFEST to its matching reveal/evaluation manifest}"
: "${POSE_SCORES:?Set POSE_SCORES to the frozen validation V1 pose_scores.csv}"
: "${SEARCH_ROOT:?Set SEARCH_ROOT to a new output directory}"

WORKERS="${WORKERS:-8}"
DIAGNOSTIC_ROOT="${DIAGNOSTIC_ROOT:-${SEARCH_ROOT}/interaction_candidate_diagnostic_v1}"

for required in "${MANIFEST}" "${REFERENCE_MANIFEST}" "${POSE_SCORES}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Required input does not exist: ${required}" >&2
    exit 2
  fi
done
if [[ -e "${SEARCH_ROOT}" ]]; then
  echo "Refusing to overwrite SEARCH_ROOT: ${SEARCH_ROOT}" >&2
  exit 2
fi

python -u scripts/run_adaptive_density_search.py \
  --manifest "${MANIFEST}" \
  --scores "${POSE_SCORES}" \
  --output-dir "${SEARCH_ROOT}" \
  --device cuda \
  --density-sigmas 1.5,2.25 \
  --density-weight 3.0 \
  --basin-weight 0.25 \
  --anchor-penalty 0.10 \
  --coarse-anchor-penalty 0.0 \
  --coarse-penalty 0.10 \
  --clash-penalty 0.02 \
  --parent-count 6 \
  --top-score-parents 3 \
  --parent-score-column confidence \
  --parent-diversity-rmsd 1.5 \
  --search-depth 3 \
  --beam-width 48 \
  --beam-width-per-parent 10 \
  --coarse-beam-fraction 0.25 \
  --translation-steps 0.25,0.5 \
  --rotation-steps 10,20 \
  --torsion-steps 30,60 \
  --max-torsion-bonds 8 \
  --top-torsion-gradients 2 \
  --minimum-step-ncc-gain 0.0 \
  --coarse-minimum-ncc-gain=-0.01 \
  --max-centroid-shift 1.5 \
  --min-in-bounds-fraction 0.95 \
  --clash-tolerance-count 0 \
  --clash-tolerance-fraction 0.0 \
  --severe-overlap 0.8 \
  --mask-radius 10.0 \
  --finalist-count 48 \
  --basin-translation-step 0.20 \
  --basin-rotation-step 7.5 \
  --basin-support-tolerance 0.01 \
  --materialization-mode interaction_diagnostic \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "${REFERENCE_MANIFEST}" \
  --results "${SEARCH_ROOT}/interaction_pool" \
  --condition interaction_pool \
  --num-predictions 100000 \
  --output-dir "${SEARCH_ROOT}/metrics_interaction_pool"

python evaluation/diagnose_candidate_interactions.py \
  --search-root "${SEARCH_ROOT}" \
  --pool-condition interaction_pool \
  --output-dir "${DIAGNOSTIC_ROOT}" \
  --workers "${WORKERS}" \
  --strict

echo "VALIDATION_INTERACTION_EXPERIMENT_COMPLETE: ${DIAGNOSTIC_ROOT}"
