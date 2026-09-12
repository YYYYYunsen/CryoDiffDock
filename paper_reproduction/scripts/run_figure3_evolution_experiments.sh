#!/usr/bin/env bash
set -euo pipefail

# Build native-compute Figure 3 A--E. The frozen Original, V1, and V2 main
# results are reused; only missing V2 component ablations are searched.

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
V1_ROOT=${V1_ROOT:-$PROJECT_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
V2_MAIN=${V2_MAIN:-$PROJECT_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1}
FIG3_WORK=${FIG3_WORK:-$PROJECT_ROOT/artifacts/paper_test/figure3_local_evolution_ablation60_v1}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
COHORT_FILE=${COHORT_FILE:-$PROJECT_ROOT/visualizations/figure3_local/data/figure3_frozen_drug_like_cohort.txt}
ABLATION_COHORT_FILE=${ABLATION_COHORT_FILE:-$PROJECT_ROOT/visualizations/figure3_local/data/figure3_ablation_cohort60.txt}
HARDWARE_ID=${HARDWARE_ID:-$(hostname)-A100}
PROCESS_COUNT=${PROCESS_COUNT:-1}

PYTHON_BIN=$ENV_BIN/python
BLIND_MANIFEST=$V1_ROOT/02_prepared/inference_blind.csv
REFERENCE_MANIFEST=$V1_ROOT/02_prepared/inference.csv
POSE_SCORES=$V1_ROOT/04_ncc/pose_scores.csv
ABLATIONS=$FIG3_WORK/ablations
EXPERIMENT_MANIFEST=$FIG3_WORK/figure3_native_experiments.csv
FIGURE_DATA=$FIG3_WORK/figure3_data
FIGURE_PLOTS=$FIG3_WORK/figure3_plots

cd "$PROJECT_ROOT"
for required in \
  "$PYTHON_BIN" "$BLIND_MANIFEST" "$REFERENCE_MANIFEST" "$POSE_SCORES" \
  "$COHORT_FILE" "$ABLATION_COHORT_FILE" "$V2_MAIN/summary.json" "$V2_MAIN/per_complex_audit.csv" \
  "$V2_MAIN/candidates.csv" "$V2_MAIN/selected.csv"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input does not exist: $required" >&2
    exit 2
  fi
done
mkdir -p "$FIG3_WORK"

COMPLEXES=$(
  "$PYTHON_BIN" -c '
from pathlib import Path
import json, pandas as pd, sys
cohort_path, ablation_path, manifest_path, scores_path, v2_root = map(Path, sys.argv[1:])
cohort = [x.strip().upper() for x in cohort_path.read_text().splitlines() if x.strip()]
if len(cohort) != 171 or len(cohort) != len(set(cohort)):
    raise SystemExit(f"Expected 171 unique frozen IDs, got {len(cohort)}")
ablation = [x.strip().upper() for x in ablation_path.read_text().splitlines() if x.strip()]
if len(ablation) != 60 or len(ablation) != len(set(ablation)):
    raise SystemExit(f"Expected 60 unique ablation IDs, got {len(ablation)}")
if not set(ablation) <= set(cohort):
    raise SystemExit("Ablation cohort is not a subset of the frozen 171-complex cohort")
manifest = set(pd.read_csv(manifest_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
scores = set(pd.read_csv(scores_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
audit = pd.read_csv(v2_root / "per_complex_audit.csv", dtype={"complex_name": str})
v2_names = audit["complex_name"].astype(str).str.upper().tolist()
summary = json.loads((v2_root / "summary.json").read_text())
if int(summary.get("num_complexes", -1)) != 171 or len(v2_names) != 171 or set(v2_names) != set(cohort):
    raise SystemExit("Existing V2 main does not match the frozen 171-complex cohort")
missing = sorted(set(cohort) - manifest) + sorted(set(cohort) - scores)
if missing:
    raise SystemExit(f"Frozen cohort is absent from inputs: {missing[:10]}")
print(",".join(ablation))
' "$COHORT_FILE" "$ABLATION_COHORT_FILE" "$BLIND_MANIFEST" "$POSE_SCORES" "$V2_MAIN"
)

V2_COMMON=(
  --manifest "$BLIND_MANIFEST"
  --scores "$POSE_SCORES"
  --complexes "$COMPLEXES"
  --device cuda
  --density-sigmas 1.5,2.25
  --parent-count 6
  --search-depth 3
  --beam-width 48
  --beam-width-per-parent 10
  --coarse-beam-fraction 0.25
  --translation-steps 0.25,0.5
  --rotation-steps 10,20
  --torsion-steps 30,60
  --top-torsion-gradients 2
  --materialize-expanded-limit 0
  --fail-on-error
)

evaluate_v2() {
  local root=$1
  PATH="$ENV_BIN:$PATH" \
  SEARCH_ROOT="$root" \
  V1_ROOT="$V1_ROOT" \
  REFERENCE_MANIFEST="$REFERENCE_MANIFEST" \
    bash scripts/evaluate_adaptive_density_search.sh
}

run_ablation() {
  local experiment_id=$1
  shift
  local root=$ABLATIONS/$experiment_id
  echo "FIGURE3_ABLATION_START: $experiment_id"
  if [[ ! -s "$root/summary.json" ]]; then
    "$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
      "${V2_COMMON[@]}" \
      --output-dir "$root" \
      --experiment-id "$experiment_id" \
      "$@"
  else
    "$PYTHON_BIN" -c '
import json, sys
observed = json.load(open(sys.argv[1])).get("experiment_id")
if observed != sys.argv[2]:
    raise SystemExit(f"Refusing incompatible resume: expected {sys.argv[2]!r}, found {observed!r}")
' "$root/summary.json" "$experiment_id"
    echo "[resume] search already complete: $root"
  fi
  evaluate_v2 "$root"
  echo "FIGURE3_ABLATION_COMPLETE: $experiment_id"
}

echo "FIGURE3_NATIVE_EVOLUTION_START"
echo "Core cohort: 171 complexes"
echo "Prespecified ablation cohort: 60 complexes ($ABLATION_COHORT_FILE)"
echo "Hardware label: $HARDWARE_ID"
echo "REUSING_FROZEN_V2_MAIN: $V2_MAIN"
evaluate_v2 "$V2_MAIN"

run_ablation no_density_gradients --no-density-gradients
run_ablation no_hierarchical_composition --search-depth 1
run_ablation no_combined_rigid --no-combined-rigid
run_ablation no_torsion --max-torsion-bonds 0
run_ablation no_coarse_orientations --no-coarse-orientations
run_ablation no_basin_robustness --basin-weight 0
run_ablation no_clash_penalty --clash-penalty 0

"$PYTHON_BIN" scripts/write_figure3_experiment_manifest.py \
  --v2-main-root "$V2_MAIN" \
  --ablations-root "$ABLATIONS" \
  --hardware-id "$HARDWARE_ID" \
  --process-count "$PROCESS_COUNT" \
  --output "$EXPERIMENT_MANIFEST"

"$PYTHON_BIN" visualizations/figure3_local/build_figure3_evolution_data.py \
  --cohort "$COHORT_FILE" \
  --ablation-cohort "$ABLATION_COHORT_FILE" \
  --v1-root "$V1_ROOT" \
  --v2-root "$V2_MAIN" \
  --experiments-manifest "$EXPERIMENT_MANIFEST" \
  --output-dir "$FIGURE_DATA"

"$PYTHON_BIN" visualizations/figure3_local/plot_figure3.py \
  --data-dir "$FIGURE_DATA" \
  --output-dir "$FIGURE_PLOTS"

echo "FIGURE3_NATIVE_EVOLUTION_COMPLETE: $FIG3_WORK"
