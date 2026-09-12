#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify released CryoDiffDock checkpoints")
    parser.add_argument("--manifest", type=Path, default=ROOT / "checkpoints" / "manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    missing_metadata = []
    for name, item in manifest["files"].items():
        if not item.get("source", "").startswith(("https://", "http://")) or not item.get("sha256"):
            missing_metadata.append(name)
    if missing_metadata:
        raise SystemExit(
            "Checkpoint release metadata is incomplete for: " + ", ".join(missing_metadata)
            + ". Publish the frozen files and update checkpoints/manifest.json first."
        )
    for name, item in manifest["files"].items():
        target = ROOT / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            print(f"Downloading {name} -> {target}")
            urllib.request.urlretrieve(item["source"], target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {name}: {digest}")
        print(f"OK {name}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

