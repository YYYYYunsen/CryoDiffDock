#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

if [[ -z "${SEARCH_ROOT:-}" || ! -d "$SEARCH_ROOT" ]]; then
  echo "Set SEARCH_ROOT to run_adaptive_density_search.py output" >&2
  exit 2
fi
MANIFEST=${MANIFEST:-$SEARCH_ROOT/evaluation_manifest.csv}
if ! head -n 1 "$MANIFEST" | tr ',' '\n' | grep -qx 'reference_ligand'; then
  if [[ -z "${REFERENCE_MANIFEST:-}" || ! -f "$REFERENCE_MANIFEST" ]]; then
    echo "Evaluation manifest is blind (no reference_ligand): $MANIFEST" >&2
    echo "Set REFERENCE_MANIFEST to the matching reveal manifest, e.g. 02_prepared/inference.csv" >&2
    exit 2
  fi
  REVEAL_MANIFEST=$SEARCH_ROOT/evaluation_manifest_reveal.csv
  python - "$MANIFEST" "$REFERENCE_MANIFEST" "$REVEAL_MANIFEST" <<'PY'
import sys
import pandas as pd

blind_path, reference_path, output_path = sys.argv[1:]
blind = pd.read_csv(blind_path, dtype={"complex_name": str})
reference = pd.read_csv(reference_path, dtype={"complex_name": str})
if "reference_ligand" not in reference.columns:
    raise ValueError(f"Reference manifest has no reference_ligand column: {reference_path}")
if blind["complex_name"].duplicated().any() or reference["complex_name"].duplicated().any():
    raise ValueError("complex_name must be unique in blind and reference manifests")
merged = blind.merge(
    reference[["complex_name", "reference_ligand"]],
    on="complex_name", how="left", sort=False, validate="one_to_one",
)
if len(merged) != len(blind) or merged["reference_ligand"].isna().any():
    missing = merged.loc[merged["reference_ligand"].isna(), "complex_name"].tolist()
    raise ValueError(f"Reference-manifest pairing failed; missing complexes: {missing[:10]}")
if merged["complex_name"].astype(str).tolist() != blind["complex_name"].astype(str).tolist():
    raise ValueError("Reference merge changed manifest row order")
merged.to_csv(output_path, index=False)
print(f"V2_REVEAL_MANIFEST_OK: {output_path} ({len(merged)} complexes)")
PY
  MANIFEST=$REVEAL_MANIFEST
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

evaluate_condition original_confidence 100000
evaluate_condition original_density 1
evaluate_condition expanded_confidence 1
evaluate_condition expanded_density 1
evaluate_condition expanded_density_physics 1
evaluate_condition expanded_pool 100000
evaluate_condition v2_selected 1

if [[ -d "$SEARCH_ROOT/v2_rescored" ]]; then
  evaluate_condition v2_rescored 1
fi

python evaluation/compare_ligand_conditions.py \
  --baseline "$SEARCH_ROOT/metrics_original_density/per_complex.csv" \
  --guided "$SEARCH_ROOT/metrics_v2_selected/per_complex.csv" \
  --output-dir "$SEARCH_ROOT/paired_original_vs_v2"

if [[ -n "${V1_ROOT:-}" ]]; then
  if [[ -n "${V1_S0_METRICS:-}" ]]; then
    V1_SELECTED=$V1_S0_METRICS
  elif [[ -f "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv" ]]; then
    V1_SELECTED=$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv
  elif [[ -f "$V1_ROOT/metrics_frozen_s0/per_complex.csv" ]]; then
    V1_SELECTED=$V1_ROOT/metrics_frozen_s0/per_complex.csv
  else
    echo "Could not locate frozen-S0 metrics below V1_ROOT=$V1_ROOT" >&2
    echo "Set V1_S0_METRICS to frozen_s0/per_complex.csv explicitly" >&2
    exit 2
  fi
  if [[ -n "${V1_ORACLE_METRICS:-}" ]]; then
    V1_ORACLE=$V1_ORACLE_METRICS
  elif [[ -f "$V1_ROOT/11_metrics_reveal/expanded_pool/per_complex.csv" ]]; then
    V1_ORACLE=$V1_ROOT/11_metrics_reveal/expanded_pool/per_complex.csv
  elif [[ -f "$V1_ROOT/metrics_expanded_pool/per_complex.csv" ]]; then
    V1_ORACLE=$V1_ROOT/metrics_expanded_pool/per_complex.csv
  else
    echo "Could not locate V1 expanded-pool oracle metrics below V1_ROOT=$V1_ROOT" >&2
    echo "Set V1_ORACLE_METRICS to expanded_pool/per_complex.csv explicitly" >&2
    exit 2
  fi
  echo "V1 frozen-S0 metrics: $V1_SELECTED"
  echo "V1 expanded-pool oracle metrics: $V1_ORACLE"
  python evaluation/compare_ligand_conditions.py \
    --baseline "$V1_SELECTED" \
    --guided "$SEARCH_ROOT/metrics_v2_selected/per_complex.csv" \
    --allow-intersection \
    --output-dir "$SEARCH_ROOT/paired_v1_frozen_s0_vs_v2_selected"
  python evaluation/compare_ligand_conditions.py \
    --baseline "$V1_ORACLE" \
    --guided "$SEARCH_ROOT/metrics_expanded_pool/per_complex.csv" \
    --allow-intersection \
    --output-dir "$SEARCH_ROOT/paired_v1_vs_v2_oracle"
fi

python - "$SEARCH_ROOT" <<'PY'
import json
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
conditions = (
    "original_confidence", "original_density", "expanded_confidence",
    "expanded_density", "expanded_density_physics", "v2_selected",
    "expanded_pool",
)
if (root / "metrics_v2_rescored" / "summary.json").is_file():
    conditions = (*conditions[:-1], "v2_rescored", conditions[-1])
print("\nCRYODIFFDOCK-LOCAL V2 SUMMARY")
print(f"{'condition':30s} {'n':>5s} {'median':>9s} {'top1<2':>9s} {'top1<5':>9s} {'best<2':>9s}")
for condition in conditions:
    summary = json.loads((root / f"metrics_{condition}" / "summary.json").read_text())
    print(
        f"{condition:30s} {summary['num_complexes']:5d} "
        f"{summary['median_top1_rmsd_A']:9.3f} "
        f"{summary['top1_lt2A_pct']:9.2f} "
        f"{summary['top1_lt5A_pct']:9.2f} "
        f"{summary['best_of_n_lt2A_pct']:9.2f}"
    )
audit = pd.read_csv(root / "per_complex_audit.csv")
print("\nSEARCH AUDIT")
print("median generated candidates:", float(audit["num_generated"].median()))
print("median eligible candidates:", float(audit["num_eligible_generated"].median()))
print("selected composed move (depth > 1) %:", 100.0 * float((audit["selected_depth"] > 1).mean()))
print("selected coarse branch %:", 100.0 * float((audit["selected_branch_type"] == "coarse").mean()))
runtime = json.loads((root / "summary.json").read_text())
print("total V2 search runtime (s):", runtime["runtime_seconds"])
print("timing breakdown:", runtime["timing"])
PY

echo "ADAPTIVE_DENSITY_SEARCH_EVALUATION_OK: $SEARCH_ROOT"
