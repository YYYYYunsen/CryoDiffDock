#!/usr/bin/env bash
set -euo pipefail

# Run all seven V2 ablations for one disjoint cohort shard on one allocated GPU.

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
V1_ROOT=${V1_ROOT:-$PROJECT_ROOT/artifacts/paper_test/oracle_local_frozen_v1}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
SHARD_FILE=${SHARD_FILE:?Set SHARD_FILE to one frozen shard ID list}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a unique directory for this shard}
SHARD_ID=${SHARD_ID:?Set SHARD_ID, for example shard_b60}
HARDWARE_ID=${HARDWARE_ID:-$(hostname)-A100}
MAIN_REPAIRED_FILE=${MAIN_REPAIRED_FILE:-}
MAIN_REPAIRED_OUTPUT=${MAIN_REPAIRED_OUTPUT:-}

PYTHON_BIN=$ENV_BIN/python
BLIND_MANIFEST=$V1_ROOT/02_prepared/inference_blind.csv
REFERENCE_MANIFEST=$V1_ROOT/02_prepared/inference.csv
POSE_SCORES=$V1_ROOT/04_ncc/pose_scores.csv

cd "$PROJECT_ROOT"
for required in \
  "$PYTHON_BIN" "$SHARD_FILE" "$BLIND_MANIFEST" "$REFERENCE_MANIFEST" "$POSE_SCORES"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input does not exist: $required" >&2
    exit 2
  fi
done
mkdir -p "$OUTPUT_ROOT"

COMPLEXES=$(
  "$PYTHON_BIN" -c '
from pathlib import Path
import pandas as pd, sys
shard_path, manifest_path, scores_path = map(Path, sys.argv[1:])
ids = [x.strip().upper() for x in shard_path.read_text().splitlines() if x.strip()]
if not ids or len(ids) != len(set(ids)):
    raise SystemExit("Shard must contain non-empty unique IDs")
manifest = set(pd.read_csv(manifest_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
scores = set(pd.read_csv(scores_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
missing = sorted(set(ids) - manifest) + sorted(set(ids) - scores)
if missing:
    raise SystemExit(f"Shard IDs are absent from inputs: {missing[:20]}")
print(",".join(ids))
' "$SHARD_FILE" "$BLIND_MANIFEST" "$POSE_SCORES"
)
SHARD_N=$(awk 'NF {n++} END {print n+0}' "$SHARD_FILE")

V2_COMMON=(
  --manifest "$BLIND_MANIFEST"
  --scores "$POSE_SCORES"
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
  --fail-on-error
)

evaluate_selected() {
  local root=$1
  local reveal=$root/evaluation_manifest_reveal.csv
  "$PYTHON_BIN" scripts/attach_reference_manifest.py \
    --blind "$root/evaluation_manifest.csv" \
    --reference "$REFERENCE_MANIFEST" \
    --output "$reveal"
  if [[ ! -s "$root/metrics_v2_selected/summary.json" ]]; then
    "$PYTHON_BIN" evaluation/evaluate_ligand_accuracy.py \
      --manifest "$reveal" \
      --results "$root/v2_selected" \
      --condition v2_selected \
      --num-predictions 1 \
      --output-dir "$root/metrics_v2_selected" \
      --fail-on-error
  else
    echo "[resume] selected metrics already complete: $root"
  fi
}

run_ablation() {
  local key=$1
  shift
  local root=$OUTPUT_ROOT/$key
  local experiment_id=${key}__${SHARD_ID}
  echo "FIGURE3_SHARD_ABLATION_START: $SHARD_ID $key ($SHARD_N cases)"
  if [[ ! -s "$root/summary.json" ]]; then
    "$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
      "${V2_COMMON[@]}" \
      --complexes "$COMPLEXES" \
      --output-dir "$root" \
      --experiment-id "$experiment_id" \
      --materialization-mode top1 \
      "$@"
  else
    "$PYTHON_BIN" -c '
import json, sys
from pathlib import Path
import pandas as pd
summary_path, expected_id, shard_path = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
summary = json.loads(summary_path.read_text())
expected = [x.strip().upper() for x in shard_path.read_text().splitlines() if x.strip()]
audit = pd.read_csv(summary_path.parent / "per_complex_audit.csv", dtype={"complex_name": str})
observed = audit["complex_name"].astype(str).str.upper().tolist()
if summary.get("experiment_id") != expected_id or len(observed) != len(expected) or set(observed) != set(expected):
    raise SystemExit("Refusing incompatible shard resume")
' "$root/summary.json" "$experiment_id" "$SHARD_FILE"
    echo "[resume] search already complete: $root"
  fi
  evaluate_selected "$root"
  echo "FIGURE3_SHARD_ABLATION_COMPLETE: $SHARD_ID $key"
}

echo "FIGURE3_ABLATION_SHARD_START: $SHARD_ID"
echo "Cases: $SHARD_N"
echo "GPU label: $HARDWARE_ID"
echo "Output: $OUTPUT_ROOT"

run_ablation no_density_gradients --no-density-gradients
run_ablation no_hierarchical_composition --search-depth 1
run_ablation no_combined_rigid --no-combined-rigid
run_ablation no_torsion --max-torsion-bonds 0
run_ablation no_coarse_orientations --no-coarse-orientations
run_ablation no_basin_robustness --basin-weight 0
run_ablation no_clash_penalty --clash-penalty 0

if [[ -n "$MAIN_REPAIRED_FILE" || -n "$MAIN_REPAIRED_OUTPUT" ]]; then
  if [[ -z "$MAIN_REPAIRED_FILE" || -z "$MAIN_REPAIRED_OUTPUT" ]]; then
    echo "Set both MAIN_REPAIRED_FILE and MAIN_REPAIRED_OUTPUT, or neither" >&2
    exit 2
  fi
  REPAIRED_COMPLEXES=$(
    "$PYTHON_BIN" -c '
from pathlib import Path
import pandas as pd, sys
cohort_path, manifest_path, scores_path = map(Path, sys.argv[1:])
ids = [x.strip().upper() for x in cohort_path.read_text().splitlines() if x.strip()]
if len(ids) != 9 or len(ids) != len(set(ids)):
    raise SystemExit(f"Expected 9 unique repaired IDs, got {len(ids)}")
manifest = set(pd.read_csv(manifest_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
scores = set(pd.read_csv(scores_path, usecols=["complex_name"])["complex_name"].astype(str).str.upper())
missing = sorted(set(ids) - manifest) + sorted(set(ids) - scores)
if missing:
    raise SystemExit(f"Repaired IDs are absent from inputs: {missing[:20]}")
print(",".join(ids))
' "$MAIN_REPAIRED_FILE" "$BLIND_MANIFEST" "$POSE_SCORES"
  )
  main_id="v2_main_repaired__${SHARD_ID}"
  echo "FIGURE3_REPAIRED_MAIN_START: $SHARD_ID (9 cases)"
  if [[ ! -s "$MAIN_REPAIRED_OUTPUT/summary.json" ]]; then
    "$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
      "${V2_COMMON[@]}" \
      --complexes "$REPAIRED_COMPLEXES" \
      --output-dir "$MAIN_REPAIRED_OUTPUT" \
      --experiment-id "$main_id" \
      --materialization-mode benchmark \
      --materialize-expanded-limit 0
  fi
  PATH="$ENV_BIN:$PATH" \
  SEARCH_ROOT="$MAIN_REPAIRED_OUTPUT" \
  V1_ROOT="$V1_ROOT" \
  REFERENCE_MANIFEST="$REFERENCE_MANIFEST" \
    bash scripts/evaluate_adaptive_density_search.sh
  echo "FIGURE3_REPAIRED_MAIN_COMPLETE: $SHARD_ID"
fi

"$PYTHON_BIN" -c '
import hashlib, json, sys
from pathlib import Path
root, shard, shard_id, hardware = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
ids = [x.strip().upper() for x in shard.read_text().splitlines() if x.strip()]
record = {
    "shard_id": shard_id,
    "num_complexes": len(ids),
    "hardware_id": hardware,
    "cohort_file": str(shard.resolve()),
    "cohort_sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    "materialization_mode": "top1",
    "ablations": [
        "no_density_gradients", "no_hierarchical_composition",
        "no_combined_rigid", "no_torsion", "no_coarse_orientations",
        "no_basin_robustness", "no_clash_penalty",
    ],
}
(root / "shard_complete.json").write_text(json.dumps(record, indent=2) + "\n")
' "$OUTPUT_ROOT" "$SHARD_FILE" "$SHARD_ID" "$HARDWARE_ID"

echo "FIGURE3_ABLATION_SHARD_COMPLETE: $SHARD_ID $OUTPUT_ROOT"
