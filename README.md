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
SHA-256 values, permanent download URLs, and a redistributable cryo-EM smoke
test map must still be added. See `RELEASE_CHECKLIST.md`. Do not advertise exact
paper reproducibility until `cryodiffdock doctor` reports every checkpoint.

## Installation

CUDA 11.7 and a CUDA-capable GPU are recommended. Inference on CPU is possible
for some Local stages but is substantially slower.

```bash
conda env create -f environment.yml
conda activate cryodiffdock
python -m pip install --no-deps -e .
python scripts/fetch_checkpoints.py
cryodiffdock doctor
```

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
  --local-score-model checkpoints/local_score \
  --local-confidence-model checkpoints/local_confidence \
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
  --global-model checkpoints/global_score \
  --pretrained-score-model checkpoints/global_pretrained/best_ema_inference_epoch_model.pt \
  --output outputs/example_global
```

This command builds the 24 scalar and 6 vector ring channels and generates the
Global pose ensemble. The checkpoint configuration remains authoritative for
LoRA and architecture settings. The raw Global files are an ensemble; the
paper's Top-1 result additionally uses the frozen confidence+density formula in
`configs/paper_inference.json`.

## Cascade

The Cascade command starts from a Global score table produced by the frozen
ranker. This explicit boundary prevents accidental replacement of the paper
ranker with file order or density alone.

```bash
cryodiffdock cascade \
  --global-scores global_top20_scores.csv \
  --source-manifest source_manifest.csv \
  --cohort ids.txt \
  --local-score-model checkpoints/local_score \
  --local-confidence-model checkpoints/local_confidence \
  --output outputs/cascade
```

The source manifest must contain `complex_name`, `experimental_protein`,
`ligand`, `map_path`, and `mask_structure`. Cascade derives pocket centers and
flexible residues only from predicted Global poses. The validation-frozen final
parent selector and feature sealing code are retained under
`paper_reproduction/evaluation`.

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

