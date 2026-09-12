#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SSH_HOST REMOTE_DIRECTORY" >&2
  echo "Example: $0 gpu-login /path/to/DiffDock-Pocket-EM" >&2
  exit 2
fi

SSH_HOST=$1
REMOTE_DIRECTORY=$2
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ -z "$REMOTE_DIRECTORY" || "$REMOTE_DIRECTORY" == "/" || "$REMOTE_DIRECTORY" == "~" ]]; then
  echo "Refusing unsafe remote directory: $REMOTE_DIRECTORY" >&2
  exit 2
fi

# Never use --delete here: remote datasets, weights, caches and user results may
# coexist with the checkout and must not be removed by a code deployment.
ssh "$SSH_HOST" "mkdir -p '$REMOTE_DIRECTORY' && test -d '$REMOTE_DIRECTORY'"
rsync -az --info=stats2,progress2 \
  --exclude='.git/' \
  --exclude='.cache/' \
  --exclude='artifacts/' \
  --exclude='results/' \
  --exclude='workdir/' \
  --exclude='data/PDBBIND_atomCorrected/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "$PROJECT_DIR/" "$SSH_HOST:$REMOTE_DIRECTORY/"

ssh "$SSH_HOST" "cd '$REMOTE_DIRECTORY' && python -m compileall -q cryodiffdock_pocket evaluation scripts"
echo "SYNC_OK: $SSH_HOST:$REMOTE_DIRECTORY"
