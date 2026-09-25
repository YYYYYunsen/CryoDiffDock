#!/usr/bin/env python3
"""Verify repository contents, checkpoint layout, and runtime dependencies."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "pyproject.toml", "environment.yml", "src/cryodiffdock/cli.py",
    "engines/local/inference.py", "engines/global/inference_em.py",
    "engines/global/rank_global.py", "preprocessing/build_ring_features.py",
    "configs/paper_inference.json", "examples/6PEK/receptor.pdb",
    "examples/6PEK/ligand.sdf", "examples/6PEK/local_map.mrc",
    "examples/6PEK/ring_features.npz",
    "scripts/run_paahs_pipeline.py", "scripts/run_clash_aware_adaptive_hierarchical_search.py",
    "evaluation/finalize_paahs_pipeline.py",
    "engines/local/cryodiffdock_pocket/clash_repair.py",
    "configs/clash_aware_adaptive_hierarchical_search_a.json",
    "configs/density_repair_validation62_v2_frozen.json",
    "configs/density_repair_v4_frozen.json",
    "configs/caahs_a_sidechain_rescue_frozen.json",
    "configs/caahs_c_frozen_joint050.json",
    "configs/caahs_c_internal_geometry_frozen_d035.json",
)
MODULES = ("torch", "torch_geometric", "torch_scatter", "e3nn", "rdkit", "gemmi", "prody", "Bio")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true",
                        help="skip dependencies and checkpoints")
    args = parser.parse_args()
    failures: list[str] = []
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            failures.append(f"missing source file: {relative}")

    if not args.source_only:
        for module in MODULES:
            if importlib.util.find_spec(module) is None:
                failures.append(f"missing Python dependency: {module}")
        manifest = json.loads((ROOT / "checkpoints" / "manifest.json").read_text())
        for role, item in manifest["files"].items():
            path = ROOT / item["path"]
            if not path.is_file():
                failures.append(f"missing checkpoint: {role} ({item['path']})")
                continue
            expected = item.get("sha256", "")
            if expected and digest(path) != expected:
                failures.append(f"checkpoint SHA-256 mismatch: {role}")
            config_dir = path.parent
            if not any((config_dir / name).is_file() for name in ("model_parameters.yml", "config.yml")):
                failures.append(f"missing model configuration: {role}")

    if failures:
        print("VERIFY_FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
