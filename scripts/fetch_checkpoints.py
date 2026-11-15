#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
from tempfile import TemporaryDirectory

from install_weights import install_bundle, sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify released CryoDiffDock checkpoints")
    parser.add_argument("--manifest", type=Path, default=ROOT / "checkpoints" / "manifest.json")
    parser.add_argument("--bundle-url", help="direct download URL for the pinned weight ZIP")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    bundle = manifest.get("bundle", {})
    source = args.bundle_url or bundle.get("source", "")
    if source:
        if not source.startswith(("https://", "http://")) or not bundle.get("sha256"):
            raise SystemExit("A weight bundle URL and its SHA-256 are required.")
        with TemporaryDirectory(prefix="cryodiffdock-download-") as directory:
            archive = Path(directory) / "weights.zip"
            print(f"Downloading CryoDiffDock weights from {source}", flush=True)
            urllib.request.urlretrieve(source, archive)
            install_bundle(archive, manifest)
        return 0

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
        digest = sha256(target)
        if digest != item["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {name}: {digest}")
        print(f"OK {name}: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

