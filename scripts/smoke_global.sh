#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cryodiffdock global \
  --name 6PEK \
  --protein "$ROOT/examples/6PEK/receptor.pdb" \
  --ligand "$ROOT/examples/6PEK/ligand.sdf" \
  --map "$ROOT/examples/6PEK/local_map.mrc" \
  --ring-features "$ROOT/examples/6PEK/ring_features.npz" \
  --samples 4 --ranking-top-k 4 --batch-size 4 \
  --output "$ROOT/outputs/smoke_global"
