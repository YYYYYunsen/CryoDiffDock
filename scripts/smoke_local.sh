#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cryodiffdock local \
  --name 6PEK \
  --protein "$ROOT/examples/6PEK/receptor.pdb" \
  --ligand "$ROOT/examples/6PEK/ligand.sdf" \
  --map "$ROOT/examples/6PEK/local_map.mrc" \
  --center 99.5225 129.031 141.351 \
  --local-score-model "$ROOT/checkpoints/local_score" \
  --local-confidence-model "$ROOT/checkpoints/local_confidence" \
  --samples 4 --batch-size 4 \
  --output "$ROOT/outputs/smoke_local"

