# CryoDiffDock

CryoDiffDock combines generative molecular docking with experimental cryo-EM
density. This release tree contains the code paths for:

- **CryoDiffDock-Local**: known-pocket diffusion, late-stage density guidance,
  density/ confidence reranking, and hierarchical local search.
- **CryoDiffDock-Global**: site-agnostic diffusion conditioned on scalar and
  vector cryo-EM ring features.
- **CryoDiffDock-Cascade**: Global Top-5 site proposals followed by the frozen
  Local refinement protocol.

## Release status

The source code has been assembled into a standalone repository layout. It is
not yet a complete archival release: the frozen paper checkpoints, their
remaining SHA-256 values, and permanent download URLs must still be added. A
compact 6PEK/EMD-20327 smoke input is included, while its redistribution record
still needs final confirmation. See `RELEASE_CHECKLIST.md`. Do not advertise exact
paper reproducibility until `cryodiffdock doctor` reports every checkpoint.

## Installation

CUDA 11.7 and a CUDA-capable GPU are recommended. Inference on CPU is possible
for some Local stages but is substantially slower.

Prepare a weight bundle with the layout documented in `checkpoints/README.md`,
then run the automated installer:

```bash
./install.sh --weights /path/to/cryodiffdock-weights
conda activate cryodiffdock
cryodiffdock doctor
```

After permanent checkpoint URLs are added to `checkpoints/manifest.json`,
`./install.sh` without `--weights` will download and verify them automatically.

`environment-local.yml` and `environment-global.yml` retain the original engine
environments for debugging version-sensitive behavior.

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

The command runs matched baseline and density-guided ensembles, scores their
union with Gaussian NCC, and performs the released three-depth hierarchical
search. The final materialized pose is under `outputs/example_local/search`.
Use `--flexible-sidechains A:10-A:14` to specify flexible residues explicitly.

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
derives each Local pocket centre and flexible residues from the predicted pose,
and runs the released Local refinement. `final_pose.sdf` is the operational
Global-rank-1 Cascade output; all five refined branches are retained under
`search/v2_selected`. The validation selector experiments and feature-sealing
code are retained under `paper_reproduction/evaluation` and are not silently
applied to a new target.

## Paper reproduction

`configs/paper_inference.json` records the main frozen inference parameters.
The original cohort construction, leakage checks, evaluation, sharding, and
selector scripts are preserved under `paper_reproduction/`. Server paths in
those historical launchers are provenance examples and must be supplied through
their documented environment variables on another system.

Run `python -m compileall src engines preprocessing scripts` as a lightweight
source check. Full smoke and numerical regression tests require the released
checkpoints and example map.

## Citation

Add the final CryoDiffDock bibliographic record here when the manuscript has a
persistent identifier. Please also retain the DiffDock and DiffDock-Pocket
citations and notices described in `THIRD_PARTY.md`.
