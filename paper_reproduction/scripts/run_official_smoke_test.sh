#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

OUT_DIR=${OUT_DIR:-artifacts/smoke_test}
SAMPLES=${SAMPLES:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
INFERENCE_STEPS=${INFERENCE_STEPS:-5}
ENVIRONMENT_ARGS=()
if [[ "${REQUIRE_CUDA:-0}" == "1" ]]; then
  ENVIRONMENT_ARGS+=(--require-cuda)
fi

mkdir -p "$OUT_DIR"
python scripts/check_environment.py "${ENVIRONMENT_ARGS[@]}" \
  --json "$OUT_DIR/environment.json"

python -u inference.py \
  --complex_name 3dpf_smoke \
  --protein_path example_data/3dpf_protein.pdb \
  --ligand example_data/3dpf_ligand.sdf \
  --keep_local_structures \
  --samples_per_complex "$SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --inference_steps "$INFERENCE_STEPS" \
  --out_dir "$OUT_DIR/results"

mapfile -t rank1_files < <(find "$OUT_DIR/results" -mindepth 2 -maxdepth 2 -type f -name rank1.sdf -print)
if [[ ${#rank1_files[@]} -ne 1 ]]; then
  echo "Expected exactly one rank1.sdf below $OUT_DIR/results; found ${#rank1_files[@]}" >&2
  printf '  %s\n' "${rank1_files[@]}" >&2
  exit 1
fi
result_dir=$(dirname "${rank1_files[0]}")
test -s "$result_dir/rank1.sdf"
test -s "$result_dir/rank1_protein.pdb"
find "$result_dir" -maxdepth 1 -type f -print | sort
echo "SMOKE_TEST_OK: $result_dir"
