#!/usr/bin/env bash
# Force the fixed-column MOL2 compatibility utility into each EMERALD case.

POCKET_ROOT=${POCKET_ROOT:-/Scr/yunsenz2/data/DiffDock-Pocket-EM}
UPSTREAM_TOOLS_ROOT=${EMERALD_UPSTREAM_TOOLS_ROOT:-/Scr/yunsenz2/data/software/benchmark/tools/local_docking}
EMERALD_REAL_DRIVER=${EMERALD_REAL_DRIVER:-$UPSTREAM_TOOLS_ROOT/emerald_paper_driver.sh}
EMERALD_REAL_LIGAND_UTIL_PY=${EMERALD_REAL_LIGAND_UTIL_PY:-$UPSTREAM_TOOLS_ROOT/ligand_pose_utils.py}
LIGAND_UTIL_PY=${LIGAND_UTIL_PY:-$POCKET_ROOT/scripts/emerald_ligand_pose_utils_compat.py}

for required in "$EMERALD_REAL_DRIVER" "$EMERALD_REAL_LIGAND_UTIL_PY" "$LIGAND_UTIL_PY"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing EMERALD compatibility dependency: $required" >&2
    exit 2
  fi
done

export EMERALD_REAL_LIGAND_UTIL_PY
export LIGAND_UTIL_PY

echo "[compat] real driver: $EMERALD_REAL_DRIVER"
echo "[compat] ligand utility: $LIGAND_UTIL_PY"
echo "[compat] delegated ligand utility: $EMERALD_REAL_LIGAND_UTIL_PY"
echo "[compat] mol2genparams: ${MOL2GENPARAMS_PY:-driver default}"
exec bash "$EMERALD_REAL_DRIVER"
