import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_paahs_configs_validate():
    sys.path.insert(0, str(ROOT))
    from scripts.run_paahs_pipeline import validate_frozen_configs

    names = (
        "clash_aware_adaptive_hierarchical_search_a.json",
        "density_repair_validation62_v2_frozen.json",
        "density_repair_v4_frozen.json",
        "caahs_a_sidechain_rescue_frozen.json",
        "caahs_c_frozen_joint050.json",
        "caahs_c_internal_geometry_frozen_d035.json",
    )
    validate_frozen_configs([ROOT / "configs" / name for name in names])


def test_paahs_preflight_is_blind_and_does_not_run_gpu(tmp_path):
    baseline = tmp_path / "baseline.csv"
    search = tmp_path / "search.csv"
    scores = tmp_path / "scores.csv"
    baseline.write_text(
        "case_order,complex_name,baseline_pose_path,protein_path,density_crop\n"
        "0,TEST,pose.sdf,protein.pdb,crop.npz\n"
    )
    search.write_text("complex_name\nTEST\n")
    scores.write_text("complex_name\nTEST\n")
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "run_paahs_pipeline.py"),
            "--baseline-manifest", str(baseline),
            "--search-manifest", str(search),
            "--scores", str(scores),
            "--output-root", str(tmp_path / "out"),
            "--expected-cases", "1",
            "--directed-fallback-mode", "lazy",
            "--preflight-only",
        ],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "preflight_ok"
    assert payload["directed_fallback_mode"] == "lazy"
    assert payload["uses_reference_or_native_rmsd_for_selection"] is False
    assert not (tmp_path / "out").exists()


def test_cascade_frozen_rule_uses_guard_then_equal_rank_fusion():
    import pandas as pd

    sys.path.insert(0, str(ROOT / "engines" / "local"))
    sys.path.insert(0, str(ROOT))
    from evaluation.upgrade_cascade_selection import choose

    features = pd.DataFrame([
        {"candidate": "global", "global_rank": 0, "basin_q25_ncc": float("nan"),
         "shared_ncc_0": 0.4, "shared_ncc_1": 0.4},
        {"candidate": "rank1", "global_rank": 1, "basin_q25_ncc": 0.8,
         "shared_ncc_0": 0.45, "shared_ncc_1": 0.46},
        {"candidate": "rank2", "global_rank": 2, "basin_q25_ncc": 0.9,
         "shared_ncc_0": 0.5, "shared_ncc_1": 0.52},
    ])
    cols = ["shared_ncc_0", "shared_ncc_1"]
    assert choose(features, cols)["union_rank_fusion_guard"][0] == "rank2"
    features.loc[features.candidate.eq("rank2"), "shared_ncc_0"] = 0.39
    assert choose(features, cols)["union_rank_fusion_guard"][0] == "rank1"


def test_cascade_cli_materializes_frozen_selector_output(tmp_path, monkeypatch):
    from types import SimpleNamespace

    sys.path.insert(0, str(ROOT / "src"))
    from cryodiffdock import cli

    output = tmp_path / "cascade"
    commands = []

    def fake_run(command, cwd, *, env=None):
        commands.append(command)
        if command[1].endswith("upgrade_cascade_selection.py"):
            pose = (
                output / "selector" / "selected_results"
                / "union_rank_fusion_guard" / "TEST" / "rank1.sdf"
            )
            pose.parent.mkdir(parents=True)
            pose.write_text("SELECTED\n")

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(
        cli, "run_global_pipeline", lambda args, destination: tmp_path / "global_scores.csv"
    )
    args = SimpleNamespace(
        output=output, top_k=5, ranking_top_k=20, samples=40,
        name="TEST", protein=tmp_path / "protein.pdb",
        ligand=tmp_path / "ligand.sdf", map=tmp_path / "map.mrc",
        mask_structure=None, score_column="combined_at_best_weights",
        seed=1, local_score_model=tmp_path / "score",
        local_confidence_model=tmp_path / "confidence",
        batch_size=10, local_steps=30, device="cpu", rank1_only=False,
    )
    assert cli.cascade_command(args) == 0
    assert (output / "final_pose.sdf").read_text() == "SELECTED\n"
    sampling = [command for command in commands if command[1] == "inference.py"]
    assert len(sampling) == 2
    assert all("--rigid" in command for command in sampling)
    selector = next(c for c in commands if c[1].endswith("upgrade_cascade_selection.py"))
    assert "--selection-only" in selector
    assert "--materialize-selected" in selector


def test_local_cli_defaults_to_rigid_sampling_and_allows_explicit_flexibility(tmp_path, monkeypatch):
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from cryodiffdock import cli

    protein = tmp_path / "protein.pdb"
    ligand = tmp_path / "ligand.sdf"
    density_map = tmp_path / "map.mrc"
    for path in (protein, ligand, density_map):
        path.touch()
    commands = []
    monkeypatch.setattr(cli, "run", lambda command, cwd: commands.append(command))
    base_args = [
        "local", "--protein", str(protein), "--ligand", str(ligand),
        "--map", str(density_map), "--center", "1", "2", "3", "--no-paahs",
    ]
    for name, residues in (("rigid", []), ("flexible", ["--flexible-sidechains", "A:10"])):
        commands.clear()
        output = tmp_path / name
        args = cli.parser().parse_args(base_args + ["--output", str(output)] + residues)
        assert args.func(args) == 0
        sampling = [command for command in commands if command[1] == "inference.py"]
        assert len(sampling) == 2
        assert "--density_guidance" not in sampling[0]
        assert "--density_guidance" in sampling[1]
        assert all(("--rigid" in command) == (not residues) for command in sampling)
        manifest = pd.read_csv(output / "input.csv")
        if residues:
            assert manifest.loc[0, "flexible_sidechains"] == "A:10"
        else:
            assert "flexible_sidechains" not in manifest.columns
