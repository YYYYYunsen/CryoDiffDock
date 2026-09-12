#!/usr/bin/env python3
"""Install a user-supplied CryoDiffDock weight bundle and verify its layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ROLES = {
    "local_score": ("best_ema_inference_epoch_model.pt",),
    "local_confidence": ("best_model.pt",),
    "global_score": ("best_ema_model.pt",),
    "global_pretrained": ("best_ema_inference_epoch_model.pt",),
    "global_confidence": ("best_model.pt",),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_config(directory: Path) -> Path | None:
    for name in ("model_parameters.yml", "config.yml"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", required=True, type=Path,
                        help="directory containing the five checkpoint directories")
    parser.add_argument("--force", action="store_true", help="replace installed files")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    manifest_path = ROOT / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    plan: list[tuple[Path, Path]] = []

    for role, filenames in ROLES.items():
        source_dir = source / role
        target_dir = ROOT / "checkpoints" / role
        weight = next((source_dir / name for name in filenames if (source_dir / name).is_file()), None)
        config = find_config(source_dir)
        if weight is None:
            errors.append(f"{role}: missing {filenames[0]}")
        if config is None:
            errors.append(f"{role}: missing model_parameters.yml or config.yml")
        if weight is not None:
            plan.append((weight, target_dir / weight.name))
        if config is not None:
            plan.append((config, target_dir / config.name))

    if errors:
        raise SystemExit("Invalid weight bundle:\n  - " + "\n  - ".join(errors))

    for source_path, target_path in plan:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() and not args.force:
            if sha256(source_path) != sha256(target_path):
                raise SystemExit(f"Refusing to replace {target_path}; pass --force")
            print(f"Already installed: {target_path.relative_to(ROOT)}")
            continue
        shutil.copy2(source_path, target_path)
        print(f"Installed: {target_path.relative_to(ROOT)}")

    expected = {item["path"]: item for item in manifest["files"].values()}
    for relative, item in expected.items():
        path = ROOT / relative
        digest = sha256(path)
        expected_digest = item.get("sha256", "")
        if expected_digest and digest != expected_digest:
            raise SystemExit(f"SHA-256 mismatch for {relative}: {digest}")
        print(f"Verified: {relative} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
