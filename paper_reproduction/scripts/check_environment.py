#!/usr/bin/env python3
"""Audit the runtime needed for a CryoDiffDock-Pocket smoke test."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "torch", "torch_geometric", "torch_scatter", "torch_sparse", "e3nn",
    "rdkit", "Bio", "openmm", "pandas", "scipy", "yaml", "gemmi",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "modules": {},
    }
    missing = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            missing.append(name)
            report["modules"][name] = f"ERROR: {type(exc).__name__}: {exc}"
    cuda_available = False
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        report["torch_cuda_available"] = cuda_available
        report["torch_cuda_version"] = torch.version.cuda
        report["gpu_count"] = int(torch.cuda.device_count())
        report["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception:
        report["torch_cuda_available"] = False
    report["nvidia_smi"] = shutil.which("nvidia-smi")
    report["slurm"] = shutil.which("srun") is not None
    report["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
        report["nvidia_smi_query"] = result.stdout.strip() or result.stderr.strip()
    report["ok"] = not missing and (cuda_available or not args.require_cuda)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.json_path:
        output = Path(args.json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
