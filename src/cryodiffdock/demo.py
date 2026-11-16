"""Run and report the bundled 9CF5 known-pocket example."""
from __future__ import annotations

import hashlib
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from collections import deque
from tempfile import mkdtemp

from .cli import CHECKPOINTS, ROOT

EXAMPLE = ROOT / "examples" / "9CF5"


def load_inputs(output):
    import gemmi
    import numpy as np
    from rdkit import Chem

    protein = EXAMPLE / "receptor.pdb"
    reference = EXAMPLE / "reference.sdf"
    crop = EXAMPLE / "density_crop.npz"
    if not gemmi.read_structure(str(protein))[0]:
        raise ValueError("The receptor PDB has no chains.")
    ligand = Chem.MolFromMolFile(str(reference))
    if ligand is None or ligand.GetNumConformers() != 1:
        raise ValueError("The deposited ligand SDF could not be read.")
    # Provide chemistry only. The Local sampler builds its own conformers.
    chemistry = Chem.MolFromSmiles(Chem.MolToSmiles(ligand, isomericSmiles=True))
    ligand_input = output / "ligand_input.sdf"
    with Chem.SDWriter(str(ligand_input)) as writer:
        writer.write(chemistry)
    with np.load(crop, allow_pickle=False) as data:
        density = data["density"]
        origin, spacing, center = (data[key] for key in ("origin", "spacing", "center"))
    if (density.ndim != 3 or min(density.shape) < 2
            or any(a.shape != (3,) or not np.isfinite(a).all() for a in (origin, spacing, center))
            or not np.isfinite(density).all() or np.any(spacing <= 0)
            or np.any(center < origin)
            or np.any(center > origin + spacing * (np.asarray(density.shape) - 1))):
        raise ValueError("The density crop has invalid grid coordinates or values.")
    write_density_map(output / "density.mrc", density, origin, spacing)
    return protein, reference, crop, ligand_input, center


def write_density_map(path, density, origin, spacing):
    import gemmi
    import numpy as np

    grid = gemmi.FloatGrid(*density.shape)
    grid.unit_cell = gemmi.UnitCell(*(spacing * density.shape), 90, 90, 90)
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    np.asarray(grid)[:] = density
    density_map = gemmi.Ccp4Map()
    density_map.grid = grid
    density_map.update_ccp4_header()
    for word, value in enumerate(origin, start=50):
        density_map.set_header_float(word, float(value))
    density_map.write_ccp4_map(str(path))


def check_runtime():
    import torch
    import torch_geometric
    import torch_scatter
    import e3nn
    import prody
    import Bio
    import openmm
    import esm

    manifest = json.loads((CHECKPOINTS / "manifest.json").read_text())
    for role in ("local_score", "local_confidence"):
        entry = manifest["files"][role]
        weight = ROOT / entry["path"]
        if not weight.is_file() or not (weight.parent / "model_parameters.yml").is_file():
            raise FileNotFoundError(f"Missing Local weights or YAML under {weight.parent}.")
        digest = hashlib.sha256()
        with weight.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {weight.name}.")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see a CUDA GPU. Run on a GPU node with the CryoDiffDock environment active.")


def run_local(protein, crop, ligand, center, output, log):
    command = [
        sys.executable, "-m", "cryodiffdock.cli", "local", "--name", "9CF5",
        "--protein", str(protein), "--ligand", str(ligand), "--map", str(crop),
        "--center", *map(str, center), "--device", "cuda", "--output", str(output),
    ]
    recent = deque(maxlen=22)
    exception_lines = deque(maxlen=8)
    stage = "Local startup"
    stages = {
        "inference.py": "Ligand sampling",
        "rerank_ligand_by_density_ncc.py": "Density scoring",
        "run_adaptive_density_search.py": "Hierarchical search",
        "run_paahs_pipeline.py": "Physical refinement",
    }
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env={**os.environ, "PYTHONUNBUFFERED": "1"},
                          text=True, errors="replace") as child:
        for line in child.stdout:
            log.write(line)
            log.flush()
            recent.append(line.rstrip()[:1500])
            heading = line.partition(":")[0]
            if not heading.startswith((" ", "RuntimeError", "subprocess.")) and heading.endswith(("Error", "Exception")):
                exception_lines.append(line.rstrip()[:1500])
            if line.startswith("+ "):
                for script, label in stages.items():
                    if script in line:
                        stage = label + (" (density-guided)" if "--density_guidance " in line else "")
                        print(stage, flush=True)
                        break
        code = child.wait()
    if code:
        detail = "\n".join(dict.fromkeys([*exception_lines, *recent]))
        raise RuntimeError(f"{stage} exited with status {code}.\n{detail}")
    return output / "final_pose.sdf"


def evaluate_pose(protein, reference, pose, final_protein, output):
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    predicted = Chem.MolFromMolFile(str(pose))
    deposited = Chem.MolFromMolFile(str(reference))
    if predicted is None or deposited is None:
        raise ValueError("The predicted or deposited ligand SDF could not be read.")
    rmsd = rdMolAlign.CalcRMS(predicted, deposited)
    models = (final_protein if final_protein.is_file() else protein,
              pose, reference, output / "density.mrc")
    lines = [f"open {json.dumps(str(path), ensure_ascii=False)}" for path in models]
    lines += ["color #2 orange", "color #3 cyan", "volume #4 style mesh", "view #2,3"]
    (output / "view.cxc").write_text("\n".join(lines) + "\n")
    return {"run_rmsd_A": round(rmsd, 4), "within_2_A": rmsd < 2.0}


def write_report(output, result, title="CryoDiffDock · 9CF5"):
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows = "".join(f"<dt>{escape(key.replace('_', ' ').capitalize())}</dt>"
                   f"<dd>{escape(str(value))}</dd>" for key, value in result.items())
    links = {"run.log": "Run log", "refinement-recovery.log": "Refinement recovery log",
             "summary.json": "JSON summary", "final_pose.sdf": "Predicted pose",
             "candidate_scores.csv": "Candidate scores", "refined_scores.csv": "Refinement scores",
             "refinement/exit_torsion_scan.log": "Torsion scan log",
             "refinement/local_density_refinement.log": "Density refinement log",
             "density.mrc": "Density map", "view.cxc": "ChimeraX overlay"}
    navigation = " · ".join(f'<a href="{file}">{label}</a>' for file, label in links.items()
                            if (output / file).is_file())
    (output / "report.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title>'
        '<style>body{font:16px system-ui;max-width:52rem;margin:3rem auto;padding:0 1rem}'
        'dt{font-weight:600;margin-top:1rem}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        f'<h1>{escape(title)}</h1><nav>{navigation}</nav><dl>{rows}</dl></html>\n'
    )


def demo_command(args):
    try:
        output = create_output(args.output)
    except OSError as error:
        print(f"Could not create the demo output directory: {error}", file=sys.stderr)
        return 2
    if output is None:
        return 2

    return execute_demo(args, output)


def create_output(requested):
    if requested is None:
        parent = Path("outputs").resolve()
        parent.mkdir(exist_ok=True)
        return Path(mkdtemp(prefix="demo-9CF5-", dir=parent))
    output = requested.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if next(output.iterdir(), None) is not None:
        print(f"Choose a new or empty output directory. Existing files are in {output}.", file=sys.stderr)
        return None
    return output


def execute_demo(args, output):
    result = {"case": "9CF5", "mode": "Known-pocket Local",
              "pocket_definition": "Deposited ligand heavy-atom centroid",
              "archived_benchmark_rmsd_A": 1.2437}
    stage = "Checking example inputs"
    print(f"9CF5 demo output: {output}", flush=True)
    with (output / "run.log").open("w") as log:
        try:
            protein, reference, crop, ligand, center = load_inputs(output)
            stage = "Checking the runtime and weights"
            check_runtime()
            result["status"] = "Ready" if args.check else "Completed"
            if not args.check:
                stage = "Local inference"
                pose = run_local(protein, crop, ligand, center, output, log)
                stage = "Evaluating the final pose"
                final_protein = output / "paahs" / "final" / "cases" / "9CF5" / "rank1_protein.pdb"
                result.update(evaluate_pose(protein, reference, pose, final_protein, output))
        except (Exception, KeyboardInterrupt) as error:
            traceback.print_exc(file=log)
            detail = str(error) or type(error).__name__
            lowered = detail.lower()
            hint = "See run.log for the failing command and its output."
            if "out of memory" in lowered:
                hint = "The GPU ran out of memory. Retry on a GPU with more free memory."
            elif "sha-256 mismatch" in lowered:
                hint = "The checkpoint differs from the published bundle. Reinstall in a fresh checkout or restore the verified checkpoint."
            elif "no space left" in lowered:
                hint = "Free disk space or choose an output directory on another volume with --output."
            elif stage == "Local inference" and any(error in lowered for error in ("urlerror", "httperror", "connectionerror", "certificate verify")):
                hint = "A model download failed. Check network access from the GPU node and the ESM model cache."
            elif stage == "Checking the runtime and weights":
                hint = "Activate the CryoDiffDock environment and run ./install.sh to verify the installation. Run the demo on an NVIDIA GPU node."
            elif stage == "Checking example inputs":
                hint = "Check that the checkout contains examples/9CF5 and that the runtime dependencies are installed."
            result.update(status="Failed", stage=stage, reason=detail, next_step=hint)
    write_report(output, result)
    if result["status"] == "Failed":
        print(result["reason"].splitlines()[0], file=sys.stderr)
    print(f"{result['status']}. Open {output / 'report.html'}")
    return int(result["status"] == "Failed")
