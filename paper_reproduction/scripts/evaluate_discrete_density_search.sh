#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [[ -z "${SEARCH_ROOT:-}" || ! -d "$SEARCH_ROOT" ]]; then
  echo "Set SEARCH_ROOT to run_discrete_density_search.py output" >&2
  exit 2
fi
MANIFEST=${MANIFEST:-$SEARCH_ROOT/evaluation_manifest.csv}
if [[ ! -f "$MANIFEST" ]]; then
  echo "Evaluation manifest does not exist: $MANIFEST" >&2
  exit 2
fi

evaluate_condition() {
  local condition=$1
  local limit=$2
  if [[ -s "$SEARCH_ROOT/metrics_$condition/summary.json" ]]; then
    echo "[resume] $condition already complete; skipping"
    return 0
  fi
  python evaluation/evaluate_ligand_accuracy.py \
    --manifest "$MANIFEST" \
    --results "$SEARCH_ROOT/$condition" \
    --condition "$condition" \
    --num-predictions "$limit" \
    --output-dir "$SEARCH_ROOT/metrics_$condition"
}

# The complete original and expanded pools provide matched candidate-coverage
# controls.  The other directories intentionally contain only their selected
# top pose.
evaluate_condition original_confidence 100000
evaluate_condition original_density 1
evaluate_condition expanded_confidence 1
evaluate_condition expanded_density 1
evaluate_condition expanded_density_physics 1
evaluate_condition expanded_pool 100000

python evaluation/compare_ligand_conditions.py \
  --baseline "$SEARCH_ROOT/metrics_original_density/per_complex.csv" \
  --guided "$SEARCH_ROOT/metrics_expanded_density/per_complex.csv" \
  --output-dir "$SEARCH_ROOT/paired_original_vs_expanded_density"

python evaluation/compare_ligand_conditions.py \
  --baseline "$SEARCH_ROOT/metrics_expanded_density/per_complex.csv" \
  --guided "$SEARCH_ROOT/metrics_expanded_density_physics/per_complex.csv" \
  --output-dir "$SEARCH_ROOT/paired_density_vs_density_physics"

python - "$SEARCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
conditions = (
    "original_confidence",
    "original_density",
    "expanded_confidence",
    "expanded_density",
    "expanded_density_physics",
    "expanded_pool",
)

print("\nDISCRETE SEARCH SUMMARY")
print(
    f"{'condition':34s} {'n':>4s} {'median_top1':>12s} "
    f"{'top1<2':>9s} {'top1<5':>9s} {'best<2':>9s} {'best_median':>12s}"
)
for condition in conditions:
    summary = json.load(open(root / f"metrics_{condition}" / "summary.json"))
    print(
        f"{condition:34s} {summary['num_complexes']:4d} "
        f"{summary['median_top1_rmsd_A']:12.3f} "
        f"{summary['top1_lt2A_pct']:9.2f} "
        f"{summary['top1_lt5A_pct']:9.2f} "
        f"{summary['best_of_n_lt2A_pct']:9.2f} "
        f"{summary['median_best_of_n_rmsd_A']:12.3f}"
    )

audit = root / "per_complex_audit.csv"
audit_table = pd.read_csv(audit)
print("\nGENERATION AUDIT")
print(
    "median generated candidates:",
    float(audit_table["num_generated"].median()),
)
print(
    "median NCC-improving variants %:",
    float(audit_table["generated_ncc_improved_pct"].median()),
)
print(
    "expanded density selected a new variant %:",
    100.0 * float(audit_table["expanded_density_selected_new"].mean()),
)
print(
    "density+physics selected a new variant %:",
    100.0 * float(
        audit_table["expanded_density_physics_selected_new"].mean()
    ),
)
print(f"\nCandidate-generation audit: {audit}")
print("Interpretation:")
print("  original_confidence best-of-N = original proposal ceiling")
print("  expanded_pool best-of-N       = local-search proposal ceiling")
print("  original_density vs expanded_density isolates local expansion")
print("  expanded_density_physics tests the relative clash safety gate")
PY

echo "DISCRETE_DENSITY_SEARCH_EVALUATION_OK: $SEARCH_ROOT"
