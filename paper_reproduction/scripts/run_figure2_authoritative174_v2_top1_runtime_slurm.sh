#!/usr/bin/env bash
#SBATCH --job-name=fig2-v2top1
#SBATCH --partition=A6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --output=fig2-v2top1-%j.out
#SBATCH --error=fig2-v2top1-%j.err

fail() { echo "ERROR: $*" >&2; exit 2; }
run() { "$@" || fail "Command failed: $*"; }

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
PYTHON=${PYTHON:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin/python}
FIG2=$POCKET_ROOT/artifacts/figure2_authoritative174_v1
INPUT_ROOT=$FIG2/07_v2_top1_runtime_rerun/00_inputs
OUTPUT_ROOT=$FIG2/07_v2_top1_runtime_rerun/01_search
REFERENCE_AUDIT=$FIG2/03_existing_local174/local174_v2_audit_exact.csv

MANIFEST=$INPUT_ROOT/v2_top1_runtime_manifest.csv
POSE_SCORES=$INPUT_ROOT/v2_top1_runtime_pose_scores.csv
IDS=$INPUT_ROOT/v2_top1_runtime_ids.txt

[[ -n "${SLURM_JOB_ID:-}" ]] || fail "Submit this script with sbatch"
for required in "$PYTHON" "$MANIFEST" "$POSE_SCORES" "$IDS" "$REFERENCE_AUDIT"; do
  [[ -f "$required" ]] || fail "missing $required"
done
[[ ! -e "$OUTPUT_ROOT" ]] || fail "refusing to overwrite $OUTPUT_ROOT"

cd "$POCKET_ROOT" || fail "cannot enter $POCKET_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo "FIGURE2_V2_TOP1_RUNTIME_START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "HOST=${SLURMD_NODENAME:-$(hostname)}"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CASES=$(awk 'NF {n++} END {print n+0}' "$IDS")"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run "$PYTHON" -u scripts/run_adaptive_density_search.py \
  --manifest "$MANIFEST" \
  --scores "$POSE_SCORES" \
  --output-dir "$OUTPUT_ROOT" \
  --experiment-id figure2_authoritative174_v2_top1_runtime_v1 \
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
  --materialization-mode top1 \
  --materialize-expanded-limit 0

run "$PYTHON" - "$OUTPUT_ROOT" "$REFERENCE_AUDIT" "$IDS" <<'PY'
import json
import math
import pathlib
import sys

import pandas as pd

output, reference_path, ids_path = map(pathlib.Path, sys.argv[1:])
expected = [line.strip().upper() for line in ids_path.read_text().splitlines() if line.strip()]
summary = json.loads((output / "summary.json").read_text())
audit = pd.read_csv(output / "per_complex_audit.csv")
failure_path = output / "failures.csv"
failures = (
    pd.read_csv(failure_path)
    if failure_path.is_file() and failure_path.stat().st_size > 1
    else pd.DataFrame()
)
reference = pd.read_csv(reference_path)

for frame in (audit, reference):
    frame["complex_name"] = frame["complex_name"].astype(str).str.strip().str.upper()

if summary.get("materialization_mode") != "top1":
    raise ValueError("materialization_mode is not top1")
if len(audit) != len(expected) or set(audit["complex_name"]) != set(expected):
    raise ValueError(f"search coverage mismatch: audit={len(audit)}, expected={len(expected)}")
if len(failures):
    raise ValueError(f"V2 Top-1 timing failures:\n{failures.to_string(index=False)}")
required = ["selection_ready_seconds", "top1_materialization_seconds", "deployment_runtime_seconds"]
for column in required:
    values = pd.to_numeric(audit[column], errors="coerce")
    if not values.map(math.isfinite).all() or values.lt(0).any():
        raise ValueError(f"invalid timing column: {column}")
if not audit["materialization_mode"].eq("top1").all():
    raise ValueError("one or more audit rows are not top1-only")

comparison_columns = [
    "complex_name", "selected_new", "selected_depth", "selected_branch_type",
    "selected_move_type", "selected_move_sequence", "selected_multiscale_ncc",
]
old = reference[comparison_columns].dropna(subset=["selected_multiscale_ncc"])
new = audit[comparison_columns]
paired = old.merge(new, on="complex_name", suffixes=("_old", "_new"), validate="one_to_one")
for column in ("selected_new", "selected_branch_type", "selected_move_type", "selected_move_sequence"):
    left = paired[f"{column}_old"].astype(str)
    right = paired[f"{column}_new"].astype(str)
    if not left.eq(right).all():
        bad = paired.loc[~left.eq(right), "complex_name"].tolist()
        raise ValueError(f"selection parity failed for {column}: {bad}")
depth_delta = (
    pd.to_numeric(paired["selected_depth_old"], errors="raise")
    - pd.to_numeric(paired["selected_depth_new"], errors="raise")
).abs()
if len(depth_delta) and depth_delta.max() != 0:
    raise ValueError("selection parity failed for selected_depth")
delta = (
    pd.to_numeric(paired["selected_multiscale_ncc_old"], errors="raise")
    - pd.to_numeric(paired["selected_multiscale_ncc_new"], errors="raise")
).abs()
if len(delta) and delta.max() > 1e-5:
    raise ValueError(f"selected NCC parity failed; max delta={delta.max()}")

print("V2 Top-1 timing rows:", len(audit))
print("Selection parity rows:", len(paired))
print("Maximum selected NCC delta:", float(delta.max()) if len(delta) else 0.0)
print("Deployment runtime seconds:")
print(pd.to_numeric(audit["deployment_runtime_seconds"]).describe().to_string())
print("FIGURE2_V2_TOP1_RUNTIME_AUDIT_OK")
PY

echo "FIGURE2_V2_TOP1_RUNTIME_END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FIGURE2_V2_TOP1_RUNTIME_OK=$OUTPUT_ROOT"
