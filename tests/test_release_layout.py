import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_release_paths_exist():
    required = [
        "engines/local/inference.py",
        "engines/local/run_adaptive_density_search.py",
        "engines/global/inference_em.py",
        "preprocessing/build_ring_features.py",
        "configs/paper_inference.json",
        "checkpoints/manifest.json",
    ]
    assert all((ROOT / path).is_file() for path in required)


def test_checkpoint_manifest_has_all_model_roles():
    files = json.loads((ROOT / "checkpoints/manifest.json").read_text())["files"]
    assert set(files) == {
        "local_score", "local_confidence", "global_score",
        "global_pretrained_backbone", "global_confidence",
    }

