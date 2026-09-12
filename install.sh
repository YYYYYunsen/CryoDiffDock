#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_NAME=${CRYODIFFDOCK_ENV_NAME:-cryodiffdock}
WEIGHTS=""
SKIP_ENV=0

usage() {
  echo "Usage: ./install.sh [--weights /path/to/weight_bundle] [--skip-env]"
}

while (($#)); do
  case "$1" in
    --weights) WEIGHTS=${2:?missing directory after --weights}; shift 2 ;;
    --skip-env) SKIP_ENV=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $SKIP_ENV -eq 0 ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    MANAGER=micromamba
  elif command -v mamba >/dev/null 2>&1; then
    MANAGER=mamba
  elif command -v conda >/dev/null 2>&1; then
    MANAGER=conda
  else
    echo "Install Miniconda, Mamba, or Micromamba first." >&2
    exit 1
  fi
  if ! "$MANAGER" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    "$MANAGER" env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
  fi
  "$MANAGER" run -n "$ENV_NAME" python -m pip install --no-deps -e "$ROOT"
  PYTHON=("$MANAGER" run -n "$ENV_NAME" python)
else
  python -m pip install --no-deps -e "$ROOT"
  PYTHON=(python)
fi

if [[ -n $WEIGHTS ]]; then
  "${PYTHON[@]}" "$ROOT/scripts/install_weights.py" --from "$WEIGHTS"
elif "${PYTHON[@]}" "$ROOT/scripts/fetch_checkpoints.py"; then
  :
else
  echo "Automatic download is not configured yet. Re-run with --weights /path/to/bundle." >&2
  exit 1
fi

"${PYTHON[@]}" "$ROOT/scripts/verify_release.py"
echo "CryoDiffDock installed. Activate '$ENV_NAME' and run: cryodiffdock --help"
