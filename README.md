# CryoDiffDock

![Graphical abstract of cryo-EM density-guided ligand docking](docs/assets/graphical_abstract.png)

CryoDiffDock combines generative molecular docking with experimental cryo-EM
density.

## Choose a docking mode

![Three docking modes. Global searches for candidate binding sites across the receptor. Local fits the whole ligand in a known pocket. Anchor restrains a placed fragment and fits the rest.](docs/assets/docking-modes-structural.png)

*Conceptual illustration. The turquoise fragment in Anchor marks the supplied restraint.*

- [**Global**](#global) starts with an unknown binding site and ranks candidate sites and poses from the full map.
- [**Local**](#local) starts with a known pocket and uses local density to search and physically refine the full ligand.
- [**Anchor**](#anchor-a-known-ligand-fragment) starts with a placed fragment and uses density to sample the rest of the ligand. Exit torsion scanning and local density refinement can be enabled for long flexible fragments. This mode is experimental.
- [**Cascade**](#cascade) starts with an unknown site, proposes up to five pockets with Global, and runs Local refinement at those pockets.

## Release status

The five inference checkpoints and their configurations are available through
GitHub Releases. Checkpoint provenance against the archived paper runs and
end-to-end numerical regression tests are pending. See `RELEASE_CHECKLIST.md`.

## Installation

CUDA 11.7 and a CUDA-capable GPU are recommended. Inference on CPU is possible
for some Local stages but is substantially slower.

Run the installer to download and verify the model weights automatically:

```bash
./install.sh
conda activate cryodiffdock
cryodiffdock doctor
```

The installer downloads the [model-weight ZIP (523 MiB)](https://github.com/YYYYYunsen/CryoDiffDock/releases/tag/weights-v1.0.0)
from GitHub Releases and installs the five checkpoints and their YAML
configurations under `checkpoints/`. If you already have the ZIP, use
`./install.sh --weights /path/to/cryodiffdock-weights-v1.0.0.zip`.

`environment-local.yml` and `environment-global.yml` retain the original engine
environments for debugging version-sensitive behavior.

## Run the 9CF5 demo

After installation and environment activation, run this on a Linux GPU node:

```bash
cryodiffdock demo
```

Open the printed `report.html` path in a browser. The report links to the
predicted SDF, run log, and a `view.cxc` script to open in ChimeraX. The overlay
shows the prediction in orange and deposited reference in cyan. The report
compares the new run's symmetry-corrected RMSD with the archived 1.24 Å Local
result. This example uses a known pocket centered on the deposited ligand.

Use `cryodiffdock demo --check` to check example inputs, weights, and CUDA
before running inference. Failures are recorded in `report.html`,
`summary.json`, and `run.log`. Each run gets its own output directory,
or pass `--output PATH` to specify a new or empty directory.

## Inputs

All modes require a receptor structure in PDB coordinates, ligand chemistry as
an SDF, and an experimental MRC/CCP4 map in the same coordinate frame. Local
also requires a pocket center. The map supplied to Local should be segmented
ligand density or a protein-subtracted residual map; using an unprocessed full
map changes the method's input condition.

## Local

```bash
cryodiffdock local \
  --name example \
  --protein receptor.pdb \
  --ligand ligand.sdf \
  --map ligand_residual.mrc \
  --center 12.4 -3.1 28.7 \
  --output outputs/example_local
```

The command samples matched baseline and density-guided ensembles with a fixed
receptor, scores their union with Gaussian NCC, performs the V2 hierarchical
search, and then runs the frozen three-phase lazy PAAHS pipeline. Conditional
physical repair can adjust nearby side chains. The delivered pose is
`outputs/example_local/final_pose.sdf`; stage audits are under
`outputs/example_local/paahs/`. Use `--no-paahs` only to stop at the earlier V2
Top-1 output. Use `--flexible-sidechains A:10-A:14` to enable flexible sampling
for specified residues.

To run the same reference-free PAAHS pipeline on already generated ensembles,
use `cryodiffdock paahs` with the blind baseline/search manifests and per-pose
NCC table. The exact column contract and frozen configuration files are
documented in [the PAAHS workflow](docs/PAAHS_PUBLIC_WORKFLOW.md).

## Anchor a known ligand fragment

Use `anchor` when a fragment already has a trusted pose and the rest of the
ligand needs to be fitted to density.

```bash
cryodiffdock anchor \
  --protein receptor.pdb --ligand full_ligand.sdf \
  --reference placed_fragment.sdf --anchor-atoms 6 8 9 10 11 12 \
  --map ligand_residual.mrc --output outputs/anchored
```

Replace the example indices with the **one-based heavy-atom indices** of the
full ligand, listed in the order of the heavy atoms in the reference fragment.
Supply at least three non-collinear reference atoms and leave at least one atom
free. The receptor, reference, and map must use the same coordinate frame.
For a long ligand, increase `--box-size` or use a larger prepared NPZ crop that
covers the free fragment. If you use a cropped receptor for sampling, pass
the full receptor as `--clash-protein` for steric guidance and screening.
Add `--check` to validate setup before sampling.

For a ligand with a long free fragment, pass `--exit-dihedrals EXIT NEXT`
to run restrained repair, a two-bond torsion scan, and local masked-density
refinement after sampling. Each dihedral is a comma-separated list of four
one-based heavy-atom indices. Point the first central bond from the anchor
toward the free fragment. Choose the second in the same direction on the next
rotatable bond. For example, the VK3-to-MK9 mapping uses
`--exit-dihedrals 6,8,14,15 8,14,15,16`. The indices depend on the input ligand.

To refine samples you already generated, supply the same inputs, the two
dihedrals, `--reuse-sampling PATH_TO_PREVIOUS_OUTPUT` and a new `--output`.
Refinement uses the CPU. Increase `--refinement-threads` to parallelize the exit
scan. The report links to the original `candidate_scores.csv` and the new
`refined_scores.csv` so you can inspect both stages.

During reverse diffusion, the anchor is aligned to the reference after each
update and torsions that deform it are suppressed. Map gradients act on the
free atoms, with a soft receptor-clash penalty. Candidate selection ranks eligible poses by free-fragment NCC after
checking anchor deviation, receptor clashes, stereochemistry, and map bounds.
The default maximum per-atom anchor deviation is 0.5 Å. If no candidate passes,
the run reports **Needs review** and preserves the candidates and their
diagnostics in `candidate_scores.csv`. Open `report.html` for results and
`view.cxc` in ChimeraX to inspect a selected pose against the map.

This experimental mode uses the supplied fragment as a geometric restraint.
Anchor agreement measures restraint compliance, and the free-fragment density
score uses the same map as sampling. The mode has no explicit tunnel prior.

## Global

```bash
cryodiffdock global \
  --name example \
  --protein receptor.pdb \
  --mask-structure full_model.cif \
  --ligand ligand.sdf \
  --map full_map.mrc \
  --output outputs/example_global
```

This command builds the 24 scalar and 6 vector ring channels, generates the
Global pose ensemble, runs the frozen confidence model, and applies
`confidence_z + 0.5*density_mean_z + residual_shape_corr_z`. Ranked results are
written to `outputs/example_global/ranking/global_scores.csv`.

## Cascade

```bash
cryodiffdock cascade \
  --name example \
  --protein receptor.pdb \
  --mask-structure full_model.cif \
  --ligand ligand.sdf \
  --map full_map.mrc \
  --output outputs/example_cascade
```

Cascade performs Global generation and frozen ranking, takes the Global Top-5,
and derives each Local pocket centre from the predicted pose. Each Local branch
samples poses with a fixed receptor and runs V2 Local refinement. Cascade then
compares all five branch poses with the original Global pose under the same five
density crops, requires a strictly positive multiscale-NCC gain over Global, and
applies equal-weight reciprocal-rank fusion of Global site rank, Local basin-NCC
rank, and shared-map NCC rank. `final_pose.sdf` is the selected output; the blind candidate scores
and choices are under `selector/`. Use `--rank1-only` to recover the older
Global-rank-1 output. The released selector is reference-free and has no
cohort-fitted parameters. A full numerical regression of the one-command
implementation against the paper's Figure 7 archive remains on the release
checklist.

## Paper reproduction

`configs/paper_inference.json` records the main frozen inference parameters.
The final PAAHS scripts and frozen JSON configurations are under `scripts/`,
`evaluation/`, and `configs/`; Figure 5 repeated-inference, Q-score, and
cross-run selection analysis code is under `evaluation/`. The optional Q-score
selector is not part of the default Local/Cascade commands; its original
173-case lock file is not yet included. Historical cohort
construction, leakage checks, and sharding scripts remain under
`paper_reproduction/`. Server paths in historical launchers are provenance
examples and must be supplied through their documented environment variables
on another system. The new scripts do not contain server datasets or generated
results.

Run `python -m compileall src engines preprocessing scripts` as a lightweight
source check. Full smoke and numerical regression tests require the released
checkpoints and example map.

## Citation

Add the final CryoDiffDock bibliographic record here when the manuscript has a
persistent identifier. Please also retain the DiffDock and DiffDock-Pocket
citations and notices described in `THIRD_PARTY.md`.
