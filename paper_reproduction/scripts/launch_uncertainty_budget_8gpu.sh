#!/usr/bin/env bash
# Launch eight independent uncertainty-budget replicates on one 8-GPU node.
# Invoke this as a child script with `bash`; never source it into the allocation.

set -u
set -o pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
INPUT_CSV=${INPUT_CSV:-}
BUDGET_ROOT=${BUDGET_ROOT:-}
EXPECTED_CASES=${EXPECTED_CASES:-}
PYTHON_BIN=${PYTHON_BIN:-python}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
SEEDS=${SEEDS:-20260813,20260814,20260815,20260816,20260817,20260818,20260819,20260820}
CPU_THREADS_PER_REPLICATE=${CPU_THREADS_PER_REPLICATE:-4}

if [[ -z "$INPUT_CSV" || -z "$BUDGET_ROOT" || -z "$EXPECTED_CASES" ]]; then
  echo "Set INPUT_CSV, BUDGET_ROOT, and EXPECTED_CASES." >&2
  exit 2
fi
if [[ ! -f "$INPUT_CSV" ]]; then
  echo "INPUT_CSV does not exist: $INPUT_CSV" >&2
  exit 2
fi
if [[ ! "$EXPECTED_CASES" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_CASES must be a positive integer." >&2
  exit 2
fi
if [[ ! "$CPU_THREADS_PER_REPLICATE" =~ ^[1-9][0-9]*$ ]]; then
  echo "CPU_THREADS_PER_REPLICATE must be a positive integer." >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a SEED_VALUES <<< "$SEEDS"
if [[ ${#GPUS[@]} -ne 8 || ${#SEED_VALUES[@]} -ne 8 ]]; then
  echo "GPU_LIST and SEEDS must each contain exactly eight comma-separated values." >&2
  exit 2
fi

cd "$PROJECT_ROOT" || exit 2
RUNNER="$PROJECT_ROOT/scripts/run_uncertainty_budget_replicate.sh"
if [[ ! -f "$RUNNER" ]]; then
  echo "Replicate runner does not exist: $RUNNER" >&2
  exit 2
fi

# Read-only GPU preflight.  It runs inside this child launcher, not the valuable
# parent allocation shell.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable." >&2
  exit 2
fi
VISIBLE_GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$VISIBLE_GPU_COUNT" -lt 8 ]]; then
  echo "Expected at least 8 visible GPUs, found $VISIBLE_GPU_COUNT." >&2
  exit 2
fi

mkdir -p "$BUDGET_ROOT/launcher_logs"
for index in "${!GPUS[@]}"; do
  replicate_number=$((index + 1))
  replicate_label=$(printf 'replicate%02d_seed%s' "$replicate_number" "${SEED_VALUES[$index]}")
  if [[ -e "$BUDGET_ROOT/$replicate_label" ]]; then
    echo "Refusing to overwrite existing replicate: $BUDGET_ROOT/$replicate_label" >&2
    exit 2
  fi
done

STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$BUDGET_ROOT/launcher_plan.tsv" <<'EOF'
replicate	gpu	seed	materialization_mode	output_root
EOF

PIDS=()
LABELS=()
for index in "${!GPUS[@]}"; do
  replicate_number=$((index + 1))
  gpu=${GPUS[$index]}
  seed=${SEED_VALUES[$index]}
  replicate_label=$(printf 'replicate%02d_seed%s' "$replicate_number" "$seed")
  replicate_root="$BUDGET_ROOT/$replicate_label"
  log="$BUDGET_ROOT/launcher_logs/$replicate_label.log"
  # Keep all eight replicates computationally identical.  The budget curve
  # uses only the selected V2 Top-1 pose from each run; materializing the full
  # finalist pool in one replicate would inflate its runtime and make the
  # 1/2/4/8-run wall-clock budgets incomparable.
  materialization_mode=top1
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$replicate_number" "$gpu" "$seed" "$materialization_mode" "$replicate_root" \
    >> "$BUDGET_ROOT/launcher_plan.tsv"

  echo "Launching $replicate_label on physical GPU $gpu -> $log"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS="$CPU_THREADS_PER_REPLICATE"
    export MKL_NUM_THREADS="$CPU_THREADS_PER_REPLICATE"
    export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_REPLICATE"
    export PYTHONUNBUFFERED=1
    INPUT_CSV="$INPUT_CSV" \
    REPLICATE_ROOT="$replicate_root" \
    SEED="$seed" \
    EXPECTED_CASES="$EXPECTED_CASES" \
    MATERIALIZATION_MODE="$materialization_mode" \
    PROJECT_ROOT="$PROJECT_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$RUNNER"
  ) > "$log" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$replicate_label")
done

{
  echo "started_utc=$STARTED_UTC"
  for index in "${!PIDS[@]}"; do
    echo -e "${LABELS[$index]}\t${PIDS[$index]}"
  done
} > "$BUDGET_ROOT/launcher_pids.tsv"

echo
echo "All eight replicates launched."
echo "PID table: $BUDGET_ROOT/launcher_pids.tsv"
echo "Monitor: tail -f $BUDGET_ROOT/launcher_logs/replicate01_seed${SEED_VALUES[0]}.log"
echo "GPU view: watch -n 5 nvidia-smi"
echo

overall_status=0
for index in "${!PIDS[@]}"; do
  pid=${PIDS[$index]}
  label=${LABELS[$index]}
  if wait "$pid"; then
    echo "COMPLETE  $label"
  else
    returncode=$?
    echo "FAILED    $label returncode=$returncode" >&2
    overall_status=1
  fi
done

COMPLETED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
complete_count=$(find "$BUDGET_ROOT" -mindepth 2 -maxdepth 2 -name REPLICATE_COMPLETE -type f | wc -l | tr -d ' ')
{
  echo "started_utc=$STARTED_UTC"
  echo "completed_utc=$COMPLETED_UTC"
  echo "expected_replicates=8"
  echo "complete_replicates=$complete_count"
  echo "overall_status=$overall_status"
} > "$BUDGET_ROOT/launcher_summary.txt"

if [[ "$overall_status" -ne 0 || "$complete_count" -ne 8 ]]; then
  echo "UNCERTAINTY_BUDGET_8GPU_INCOMPLETE: $complete_count/8" >&2
  exit 1
fi

echo "UNCERTAINTY_BUDGET_8GPU_OK: $BUDGET_ROOT"
