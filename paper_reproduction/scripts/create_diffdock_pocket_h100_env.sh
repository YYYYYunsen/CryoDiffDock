#!/usr/bin/env bash
# Recreate the validated conda package set without its malformed pypi_0 clone
# records, then install the required pip packages and H100 CUDA stack.
set -euo pipefail
fail() { echo "ERROR: $*" >&2; exit 2; }

SOURCE_ENV=${SOURCE_ENV:-diffdock-pocket-em}
TARGET_ENV=${TARGET_ENV:-diffdock-pocket-em-h100}
CONDA_BIN=${CONDA_EXE:-$(command -v conda || true)}
[[ -n $CONDA_BIN && -x $CONDA_BIN ]] || fail "conda executable not found"

if "$CONDA_BIN" env list | awk '{print $1}' | grep -Fqx "$TARGET_ENV"; then
  echo "H100_ENV_RESUME: continuing the existing partially upgraded $TARGET_ENV"
else
  TMP_ROOT=$(mktemp -d)
  trap 'rm -rf "$TMP_ROOT"' EXIT
  CONDA_EXPLICIT=$TMP_ROOT/conda-explicit-no-pip.txt
  # --no-pip is essential: the source metadata contains entries such as
  # e3nn==0.5.1=pypi_0 which this conda version incorrectly tries to clone.
  "$CONDA_BIN" list -n "$SOURCE_ENV" --explicit --no-pip > "$CONDA_EXPLICIT"
  grep -Fqx '@EXPLICIT' "$CONDA_EXPLICIT" || fail "invalid conda explicit export"
  "$CONDA_BIN" create -y -n "$TARGET_ENV" --file "$CONDA_EXPLICIT"
fi

# An explicit recreation marks CUDA 11.7 leaf libraries as requested specs.
# Removing only pytorch-cuda leaves libcublas=11.10 pinned and makes the CUDA
# 12.1 solve impossible. Remove every installed member of the old GPU stack.
mapfile -t OLD_GPU_PACKAGES < <(
  "$CONDA_BIN" list -n "$TARGET_ENV" |
  awk 'NR>2 {print $1}' |
  grep -E '^(pytorch($|-)|torchvision$|torchaudio$|pyg$|cuda($|-)|cudnn$|nccl$|libcublas($|-)|libcufft($|-)|libcurand($|-)|libcufile($|-)|libcusolver($|-)|libcusparse($|-)|libnpp($|-)|libnvjpeg($|-))' || true
)
if (( ${#OLD_GPU_PACKAGES[@]} )); then
  echo "Removing incompatible GPU packages: ${OLD_GPU_PACKAGES[*]}"
  "$CONDA_BIN" remove -y -n "$TARGET_ENV" "${OLD_GPU_PACKAGES[@]}"
fi
"$CONDA_BIN" install -y -n "$TARGET_ENV" \
  pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1 \
  -c pytorch -c nvidia
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install --no-cache-dir \
  e3nn==0.5.1 PyYAML==6.0.1 rdkit==2023.3.3 pandas==2.1.1 \
  biopython==1.81 networkx==3.2 spyrmsd==0.6.0 scipy==1.11.3 \
  gemmi==0.7.3 fair-esm==2.0.0
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install \
  --no-cache-dir --force-reinstall --no-deps \
  torch_geometric==2.4.0 \
  torch_scatter==2.1.2 torch_sparse==0.6.18 \
  torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

"$CONDA_BIN" run -n "$TARGET_ENV" python - <<'PY'
import torch
import torch_cluster, torch_geometric, torch_scatter, torch_sparse
print('python/torch import preflight OK')
print('torch:', torch.__version__)
print('torch CUDA runtime:', torch.version.cuda)
print('PyG:', torch_geometric.__version__)
assert torch.__version__.startswith('2.1.'), torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
PY
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
"$CONDA_BIN" run -n "$TARGET_ENV" python "$PROJECT_DIR/scripts/check_environment.py"
echo "H100_ENV_CREATED: $TARGET_ENV"
