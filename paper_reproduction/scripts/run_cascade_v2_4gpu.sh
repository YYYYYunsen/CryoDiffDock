#!/usr/bin/env bash
set -euo pipefail

# Run the validation-frozen CryoDiffDock-Local V2 selector on cascade branches.
#
# The manifest and pose-score table retain the full branch row indices.  Each
# GPU receives a disjoint list of complex_name values while reading those same
# full, row-aligned inputs.  This lets run_adaptive_density_search.py preserve
# source_row_index and allows an exact merge back to the original branch order.

PROJECT_ROOT=${PROJECT_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
MANIFEST=${MANIFEST:?Set MANIFEST to the prepared cascade inference_blind.csv}
POSE_SCORES=${POSE_SCORES:?Set POSE_SCORES to the cascade NCC pose_scores.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new cascade V2 search directory}
NUM_GPUS=${NUM_GPUS:-4}
ENV_BIN=${ENV_BIN:-/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin}
PYTHON_BIN=$ENV_BIN/python

cd "$PROJECT_ROOT"

for required in "$MANIFEST" "$POSE_SCORES" "$PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input does not exist: $required" >&2
    exit 2
  fi
done
if [[ "$NUM_GPUS" -le 0 ]]; then
  echo "NUM_GPUS must be positive" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT/merged" ]]; then
  echo "Refusing to overwrite an existing merged V2 result: $OUTPUT_ROOT/merged" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/shards" "$OUTPUT_ROOT/logs"

"$PYTHON_BIN" - "$MANIFEST" "$POSE_SCORES" "$OUTPUT_ROOT/shards" "$NUM_GPUS" <<'PY'
import sys
from pathlib import Path

import pandas as pd

manifest_path = Path(sys.argv[1]).expanduser().resolve()
scores_path = Path(sys.argv[2]).expanduser().resolve()
shard_root = Path(sys.argv[3]).expanduser().resolve()
num_shards = int(sys.argv[4])

manifest = pd.read_csv(manifest_path, dtype={"complex_name": str})
scores = pd.read_csv(scores_path, usecols=["row_index", "complex_name"], dtype={"complex_name": str})
if "complex_name" not in manifest.columns:
    raise RuntimeError("Cascade manifest lacks complex_name")
if manifest["complex_name"].isna().any():
    raise RuntimeError("Cascade manifest contains an empty complex_name")
if manifest["complex_name"].duplicated().any():
    duplicated = manifest.loc[
        manifest["complex_name"].duplicated(False), "complex_name"
    ].tolist()
    raise RuntimeError(f"Cascade manifest has duplicate names: {duplicated[:20]}")

score_rows = set(pd.to_numeric(scores["row_index"], errors="raise").astype(int))
scores["row_index"] = pd.to_numeric(scores["row_index"], errors="raise").astype(int)
expected_rows = set(range(len(manifest)))
missing_scores = sorted(expected_rows - score_rows)
extra_scores = sorted(score_rows - expected_rows)
if missing_scores or extra_scores:
    raise RuntimeError(
        "NCC pose-score row mismatch: "
        f"missing={missing_scores[:20]}, extra={extra_scores[:20]}"
    )
counts = scores.groupby("row_index").size().reindex(range(len(manifest)), fill_value=0)
bad_counts = counts[counts.ne(80)]
if len(bad_counts):
    raise RuntimeError(
        "Expected exactly 80 NCC-scored baseline+guided poses per branch; "
        f"bad counts={bad_counts.head(20).to_dict()}"
    )
score_names = scores.groupby("row_index")["complex_name"].agg(
    lambda values: set(values.astype(str))
)
bad_names = {
    row_index: sorted(names)
    for row_index, names in score_names.items()
    if names != {str(manifest.iloc[row_index]["complex_name"])}
}
if bad_names:
    raise RuntimeError(
        f"NCC score complex names do not match manifest rows: {dict(list(bad_names.items())[:20])}"
    )

for shard_index in range(num_shards):
    names = manifest.iloc[shard_index::num_shards]["complex_name"].astype(str)
    directory = shard_root / f"shard{shard_index}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "complex_names.txt"
    path.write_text("".join(f"{name}\n" for name in names))
    print(f"shard{shard_index}: {len(names)} V2 branches -> {path}")

print(f"total V2 branches: {len(manifest)}")
PY

started_epoch=$(date +%s)
pids=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
  shard_root="$OUTPUT_ROOT/shards/shard${gpu}"
  shard_output="$shard_root/search"
  if [[ -e "$shard_output" ]]; then
    echo "Refusing to overwrite existing shard output: $shard_output" >&2
    exit 2
  fi
  complexes=$(paste -sd, "$shard_root/complex_names.txt")
  if [[ -z "$complexes" ]]; then
    echo "Shard $gpu is empty; reduce NUM_GPUS" >&2
    exit 2
  fi
  (
    export PATH="$ENV_BIN:$PATH"
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON_BIN" -u scripts/run_adaptive_density_search.py \
      --manifest "$MANIFEST" \
      --scores "$POSE_SCORES" \
      --output-dir "$shard_output" \
      --complexes "$complexes" \
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
      --materialization-mode top1 \
      --fail-on-error
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
  echo "At least one V2 GPU shard failed; inspect $OUTPUT_ROOT/logs" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/merge_cascade_sharded_v2.py \
  --full-manifest "$MANIFEST" \
  --shard-root "$OUTPUT_ROOT/shards" \
  --output-root "$OUTPUT_ROOT/merged" \
  --num-shards "$NUM_GPUS"

completed_epoch=$(date +%s)
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$started_epoch" "$completed_epoch" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1]).expanduser().resolve()
started = int(sys.argv[2])
completed = int(sys.argv[3])
(output / "driver_runtime.json").write_text(
    json.dumps(
        {
            "started_epoch_seconds": started,
            "completed_epoch_seconds": completed,
            "parallel_wall_seconds": completed - started,
        },
        indent=2,
    )
    + "\n"
)
PY

echo "CASCADE_V2_4GPU_OK: $OUTPUT_ROOT/merged/v2_selected"
