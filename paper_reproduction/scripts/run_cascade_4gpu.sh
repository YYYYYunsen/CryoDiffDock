#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
MANIFEST=${MANIFEST:?Set MANIFEST to cascade inference_blind.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to the cascade inference directory}
NUM_GPUS=${NUM_GPUS:-4}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
PYTHON_BIN=$ENV_BIN/python

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT/shards" "$OUTPUT_ROOT/logs"

"$PYTHON_BIN" - "$MANIFEST" "$OUTPUT_ROOT/shards" "$NUM_GPUS" <<'PY'
import sys
from pathlib import Path
import pandas as pd

manifest, output, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
table = pd.read_csv(manifest)
for shard in range(n):
    directory = output / f"shard{shard}"
    directory.mkdir(parents=True, exist_ok=True)
    table.iloc[shard::n].reset_index(drop=True).to_csv(directory / "manifest.csv", index=False)
    print(f"shard{shard}: {len(table.iloc[shard::n])} rows")
PY

pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  shard_root="$OUTPUT_ROOT/shards/shard${gpu}"
  (
    export PATH="$ENV_BIN:$PATH"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export INPUT_CSV="$shard_root/manifest.csv"
    export OUT_ROOT="$shard_root/results"
    export CONDITION=both
    export SAMPLES=40
    export BATCH_SIZE=10
    export INFERENCE_STEPS=30
    export SEED=20260813
    export DENSITY_GUIDANCE_OBJECTIVE=atom_density
    export DENSITY_LAMBDA_TR=0.5
    export DENSITY_LAMBDA_ROT=0.5
    export DENSITY_GUIDANCE_START=0.6
    export DENSITY_CLIP_RATIO=1.0
    bash scripts/run_oracle_density_ablation.sh
  ) > >(tee "$OUTPUT_ROOT/logs/gpu${gpu}.log") 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" != 0 ]]; then
  echo "At least one GPU shard failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/merge_cascade_sharded_inference.py \
  --full-manifest "$MANIFEST" \
  --shard-root "$OUTPUT_ROOT/shards" \
  --output-root "$OUTPUT_ROOT/merged" \
  --num-shards "$NUM_GPUS"

echo "CASCADE_4GPU_INFERENCE_OK: $OUTPUT_ROOT/merged"
