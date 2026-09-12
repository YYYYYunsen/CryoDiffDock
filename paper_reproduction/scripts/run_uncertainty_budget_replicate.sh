#!/usr/bin/env bash
# Run one blind, independent CryoDiffDock-Local replicate for a budget curve.

# This is intentionally a fail-fast child script.  Invoke it with `bash`; do not
# source it into a valuable interactive GPU allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-python}
INPUT_CSV=${INPUT_CSV:-}
REPLICATE_ROOT=${REPLICATE_ROOT:-}
SEED=${SEED:-}
EXPECTED_CASES=${EXPECTED_CASES:-}
MATERIALIZATION_MODE=${MATERIALIZATION_MODE:-top1}

if [[ -z "$INPUT_CSV" || -z "$REPLICATE_ROOT" || -z "$SEED" ]]; then
  echo "Set INPUT_CSV, REPLICATE_ROOT, and SEED." >&2
  exit 2
fi
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "INPUT_CSV does not exist: $INPUT_CSV" >&2
  exit 2
fi
if [[ -e "$REPLICATE_ROOT" ]]; then
  echo "Refusing to overwrite REPLICATE_ROOT: $REPLICATE_ROOT" >&2
  exit 2
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "SEED must be a non-negative integer: $SEED" >&2
  exit 2
fi
if [[ "$MATERIALIZATION_MODE" != "top1" && "$MATERIALIZATION_MODE" != "interaction_diagnostic" ]]; then
  echo "MATERIALIZATION_MODE must be top1 or interaction_diagnostic" >&2
  exit 2
fi

PYTHON_EXE=$(command -v "$PYTHON_BIN" || true)
if [[ -z "$PYTHON_EXE" || ! -x "$PYTHON_EXE" ]]; then
  echo "Python executable is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
PYTHON_BIN="$PYTHON_EXE"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

cd "$PROJECT_ROOT"

CASE_COUNT=$(
  "$PYTHON_BIN" - "$INPUT_CSV" "$EXPECTED_CASES" <<'PY'
import sys
import pandas as pd

path, expected = sys.argv[1:]
table = pd.read_csv(path, dtype=str)
if "complex_name" not in table.columns:
    raise SystemExit("INPUT_CSV lacks complex_name")
if "reference_ligand" in table.columns:
    raise SystemExit(
        "INPUT_CSV contains reference_ligand; use the blind inference manifest"
    )
ids = table["complex_name"].astype(str).str.strip().str.upper()
if ids.eq("").any() or ids.duplicated().any():
    raise SystemExit("INPUT_CSV has empty or duplicate complex_name values")
if expected and len(table) != int(expected):
    raise SystemExit(f"Expected {expected} cases, found {len(table)}")
print(len(table))
PY
)

mkdir -p "$REPLICATE_ROOT"
INFERENCE_ROOT="$REPLICATE_ROOT/01_inference"
NCC_ROOT="$REPLICATE_ROOT/02_ncc"
V2_ROOT="$REPLICATE_ROOT/03_v2_top1"
LOG_ROOT="$REPLICATE_ROOT/logs"
mkdir -p "$LOG_ROOT"

STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL_STARTED=$(date +%s)

echo "UNCERTAINTY_BUDGET_REPLICATE_START"
echo "cases=$CASE_COUNT seed=$SEED root=$REPLICATE_ROOT"
echo "started_utc=$STARTED_UTC"

STEP_STARTED=$(date +%s)
PYTHONUNBUFFERED=1 \
INPUT_CSV="$INPUT_CSV" \
OUT_ROOT="$INFERENCE_ROOT" \
CONDITION=both \
SEED="$SEED" \
SAMPLES=40 \
BATCH_SIZE=10 \
INFERENCE_STEPS=30 \
bash scripts/run_oracle_density_ablation.sh \
  2>&1 | tee "$LOG_ROOT/01_inference.log"
INFERENCE_SECONDS=$(( $(date +%s) - STEP_STARTED ))

STEP_STARTED=$(date +%s)
"$PYTHON_BIN" -u scripts/rerank_ligand_by_density_ncc.py \
  --manifest "$INPUT_CSV" \
  --candidate-root "baseline=$INFERENCE_ROOT/baseline" \
  --candidate-root "guided=$INFERENCE_ROOT/guided" \
  --output-dir "$NCC_ROOT" \
  --num-predictions 40 \
  --sigmas 1.5 \
  --mask-radius 10 \
  --ranking-sigma 1.5 \
  --density-weight 2.0 \
  --fail-on-error \
  2>&1 | tee "$LOG_ROOT/02_ncc.log"
NCC_SECONDS=$(( $(date +%s) - STEP_STARTED ))

STEP_STARTED=$(date +%s)
"$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
  --manifest "$INPUT_CSV" \
  --scores "$NCC_ROOT/pose_scores.csv" \
  --output-dir "$V2_ROOT" \
  --experiment-id "uncertainty_budget_seed_${SEED}" \
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
  --materialization-mode "$MATERIALIZATION_MODE" \
  --fail-on-error \
  2>&1 | tee "$LOG_ROOT/03_v2_top1.log"
V2_SECONDS=$(( $(date +%s) - STEP_STARTED ))

TOTAL_SECONDS=$(( $(date +%s) - TOTAL_STARTED ))
COMPLETED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

"$PYTHON_BIN" - \
  "$INPUT_CSV" "$NCC_ROOT/pose_scores.csv" "$V2_ROOT" \
  "$REPLICATE_ROOT/replicate_summary.json" \
  "$SEED" "$CASE_COUNT" "$STARTED_UTC" "$COMPLETED_UTC" \
  "$INFERENCE_SECONDS" "$NCC_SECONDS" "$V2_SECONDS" "$TOTAL_SECONDS" \
  "$MATERIALIZATION_MODE" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import torch

(
    manifest_name, scores_name, v2_name, output_name, seed, expected,
    started, completed, inference_seconds, ncc_seconds, v2_seconds, total_seconds,
    materialization_mode,
) = sys.argv[1:]
manifest = Path(manifest_name).resolve()
scores = Path(scores_name).resolve()
v2 = Path(v2_name).resolve()
output = Path(output_name).resolve()
expected = int(expected)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

score_table = pd.read_csv(scores)
score_counts = score_table.groupby("complex_name").size()
ranking_files = sorted((v2 / "v2_selected").glob("index*___*/ranking.csv"))
if len(score_table) != expected * 80 or len(score_counts) != expected:
    raise SystemExit(
        f"NCC completeness failed: rows={len(score_table)}, cases={len(score_counts)}"
    )
if not score_counts.eq(80).all():
    raise SystemExit("At least one case does not contain exactly 80 NCC-scored poses")
if len(ranking_files) != expected:
    raise SystemExit(
        f"V2 Top-1 completeness failed: expected {expected}, found {len(ranking_files)}"
    )
for ranking in ranking_files:
    table = pd.read_csv(ranking)
    if len(table) != 1 or not Path(str(table.iloc[0]["materialized_path"])).is_file():
        raise SystemExit(f"Invalid V2 Top-1 ranking: {ranking}")

summary = {
    "analysis_role": "blind_independent_replicate_for_uncertainty_budget_curve",
    "seed": int(seed),
    "case_count": expected,
    "started_utc": started,
    "completed_utc": completed,
    "input_manifest": str(manifest),
    "input_manifest_sha256": sha256(manifest),
    "ncc_pose_scores": str(scores),
    "ncc_pose_scores_sha256": sha256(scores),
    "v2_root": str(v2),
    "v2_top1_count": len(ranking_files),
    "uses_reference_coordinates": False,
    "inference": {
        "conditions": ["baseline", "guided"],
        "poses_per_condition": 40,
        "steps": 30,
    },
    "v2_materialization_mode": materialization_mode,
    "timing_seconds": {
        "baseline_plus_guided_inference": int(inference_seconds),
        "ncc": int(ncc_seconds),
        "v2_search_and_top1_materialization": int(v2_seconds),
        "end_to_end": int(total_seconds),
        "end_to_end_per_case": float(total_seconds) / expected,
    },
    "hardware": {
        "platform": platform.platform(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    },
}
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

touch "$REPLICATE_ROOT/REPLICATE_COMPLETE"
echo "UNCERTAINTY_BUDGET_REPLICATE_OK: $REPLICATE_ROOT"
