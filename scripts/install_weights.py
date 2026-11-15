#!/usr/bin/env python3
"""Install a user-supplied CryoDiffDock weight bundle and verify its layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile


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


def install(source: Path, manifest: dict, force: bool = False) -> None:
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

    expected = {item["path"]: item.get("sha256") for item in manifest["files"].values()}
    pending = []
    for source_path, target_path in plan:
        relative = target_path.relative_to(ROOT).as_posix()
        digest = sha256(source_path)
        if expected.get(relative) and digest != expected[relative]:
            raise SystemExit(f"SHA-256 mismatch for {relative}: {digest}")
        if target_path.exists():
            if sha256(target_path) == digest:
                continue
            if not force:
                raise SystemExit(f"Refusing to replace {target_path}; pass --force")
        pending.append((source_path, target_path))

    for source_path, target_path in pending:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target_path.parent) as staging:
            staged = Path(staging) / target_path.name
            shutil.copy2(source_path, staged)
            staged.replace(target_path)
        print(f"Installed: {target_path.relative_to(ROOT)}")
    print("Verified all checkpoint files.")


def install_bundle(archive: Path, manifest: dict, force: bool = False) -> None:
    bundle = manifest["bundle"]
    if sha256(archive) != bundle["sha256"]:
        raise SystemExit(f"SHA-256 mismatch for weight bundle: {archive}")
    with TemporaryDirectory(prefix="cryodiffdock-weights-") as directory:
        with ZipFile(archive) as packaged:
            packaged.extractall(directory)
        install(Path(directory) / bundle["directory"], manifest, force)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", required=True, type=Path,
                        help="weight bundle ZIP or extracted directory")
    parser.add_argument("--force", action="store_true", help="replace installed files")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    manifest = json.loads((ROOT / "checkpoints" / "manifest.json").read_text())
    if source.is_file():
        install_bundle(source, manifest, args.force)
    else:
        install(source, manifest, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
