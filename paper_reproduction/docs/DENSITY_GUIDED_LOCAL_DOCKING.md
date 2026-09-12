# Training-free density-guided local docking

This mode composes the pretrained DiffDock-Pocket ligand translation and
rotation scores with a differentiable cryo-EM ligand-density likelihood score
during reverse diffusion. It does not retrain or modify either checkpoint.
By default side-chain and ligand-torsion scores remain the original model
outputs; the optional C2 flags additionally project the density gradient onto
ligand torsions.

Two likelihood objectives are available:

- `atom_density` (default) samples map values at ligand atoms; this preserves
  the original implementation and completed ablations.
- `map_ncc` renders a differentiable, atomic-number-weighted Gaussian ligand
  density and maximizes NCC against the experimental crop on one fixed,
  crop-centred support. This is intended to better distinguish ligand
  orientation and shape.

## Density input

Each complex requires an absolute path to a Cartesian `.npz` crop containing:

```text
density       float32 [X,Y,Z]
origin        float32 [3]       absolute Angstrom coordinate of voxel [0,0,0]
spacing       float32 scalar or [3]
norm_center   float32 scalar
norm_scale    positive float32 scalar
```

For an end-to-end deployment result, the crop must be obtainable at deployment
time. For the controlled predefined-pocket benchmark, the deposited ligand
heavy-atom centroid may define a fixed-size crop if the condition is labeled
``oracle-pocket local docking``. Do not use the deposited ligand's atom shape
to synthesize or mask density. Density values must still come from the
experimental map, and the reference ligand must not enter sampling or ranking.

Convert such an MRC/CCP4 map into the Cartesian crop using the same predefined
pocket center supplied to DiffDock-Pocket:

```bash
python scripts/prepare_density_guidance_crop.py \
  --map /absolute/path/segmented_ligand_or_residual_density.mrc \
  --center-x 9.7742 --center-y 27.2863 --center-z 14.6573 \
  --box-size 24 --spacing 0.75 \
  --output /absolute/path/3dpf_density_guidance_crop.npz
```

The converter resamples through Gemmi, so map axis order and unit-cell
coordinates are respected. It does not inspect a ligand file. A raw complex
map is accepted technically, but protein density can then attract the ligand;
use segmented or protein-subtracted density for the scientific comparison.

## Single-complex inference

```bash
python -u inference.py \
  --complex_name 3dpf_guided \
  --protein_path example_data/3dpf_protein.pdb \
  --ligand example_data/3dpf_ligand.sdf \
  --pocket_center_x 9.7742 \
  --pocket_center_y 27.2863 \
  --pocket_center_z 14.6573 \
  --flexible_sidechains A:160-A:193-A:197-A:198-A:222-A:224-A:227 \
  --ligand_density_crop /absolute/path/3DPF_ligand_density_crop.npz \
  --density_guidance \
  --density_guidance_lambda_tr 0.5 \
  --density_guidance_lambda_rot 0.5 \
  --density_guidance_start 0.6 \
  --density_guidance_clip_ratio 1.0 \
  --density_guidance_objective map_ncc \
  --density_guidance_ncc_sigma 1.5 \
  --density_guidance_ncc_mask_radius 10 \
  --seed 20260813 \
  --samples_per_complex 40 \
  --batch_size 10 \
  --inference_steps 30 \
  --out_dir artifacts/3dpf_guided/results
```

For batch inference, add a `ligand_density_crop` column containing an absolute
path for every row of the normal DiffDock-Pocket input CSV, then use
`--protein_ligand_csv` and `--density_guidance`. Do not also pass the
single-complex `--ligand_density_crop` override.

Each result directory contains `density_guidance_audit.json` with the active
steps, ramp weights, in-box atom fraction, density objective, and actual
translation/rotation score-addition norms.

`map_ncc` still performs training-free posterior guidance: it improves the
effective conditional score used by the reverse process but does not change
the pretrained network parameters. Demonstrate a generation improvement with
best-of-N metrics before investing in a density-conditioned fine-tune.

For a self-contained engineering check, run:

```bash
bash scripts/run_density_guidance_smoke_test.sh
```

The script generates a GT-derived synthetic 3dpf density when `DENSITY_CROP`
is unset. That synthetic map verifies code execution only and is explicitly
invalid for scientific evaluation. Set `DENSITY_CROP=/absolute/real_crop.npz`
to exercise the same path with a real deployment-valid crop.

## Paired ablation

Use the same input CSV, checkpoint, sample count, step count, and seed:

```bash
# Exact pretrained baseline.
python -u inference.py \
  --protein_ligand_csv validation_pilot.csv \
  --seed 20260813 --samples_per_complex 40 --inference_steps 30 \
  --out_dir artifacts/local_ablation/baseline

# Likelihood-guided generation.
python -u inference.py \
  --protein_ligand_csv validation_pilot.csv \
  --density_guidance \
  --density_guidance_lambda_tr 0.5 \
  --density_guidance_lambda_rot 0.5 \
  --density_guidance_start 0.6 \
  --density_guidance_clip_ratio 1.0 \
  --seed 20260813 --samples_per_complex 40 --inference_steps 30 \
  --out_dir artifacts/local_ablation/guided
```

Tune only on the CryoDiffDock validation split. Freeze the guidance start,
weights, clip ratio, samples, and steps before running the test split. The
primary evidence that guidance improves generation is a paired improvement in
best-of-40 ligand RMSD/success; top-1 additionally measures selection.

Recommended conditions are:

1. DiffDock-Pocket baseline.
2. Baseline candidates plus density reranking.
3. Density-guided sampling.
4. Density-guided sampling plus the same density reranking.

Also run a wrong-map control and report ligand clashes, strain, and candidate
diversity so density attraction cannot hide mode collapse or bad chemistry.

## G0.5 raw-gradient informativeness check

Before interpreting norm-matched reverse-diffusion guidance, test the raw
density derivative on controlled perturbations of validation reference poses:

```bash
python evaluation/evaluate_density_gradient_informativeness.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --output-dir artifacts/oracle_local/g05_gradient_val5 \
  --limit 5 \
  --objectives atom_density,map_ncc \
  --device cuda \
  --fail-on-error
```

This validation-only diagnostic reads ground truth. It applies translations of
0.1/0.5/1.0 A and rotations of 2/10/30 degrees in both directions about each
Cartesian axis, then
takes a fixed 0.1 A or 2 degree step along the unscaled gradient. The primary
outputs are `summary.csv`, `per_complex_summary.csv`, and
`gradient_trials.csv`. A useful gradient should increase the density objective
and reduce the known perturbation error; its raw norm alone is not evidence of
usefulness.

## G0.5B generated-candidate gradient check

After G0.5 establishes a local signal around deposited poses, test that signal
on the actual validation candidates without rerunning diffusion:

```bash
python evaluation/evaluate_candidate_density_gradient.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --candidate-root artifacts/oracle_density_ablation/baseline \
  --output-dir artifacts/oracle_local/g05b_candidate_val5 \
  --limit-complexes 5 \
  --num-predictions 40 \
  --translation-step 0.02 \
  --rotation-step-degrees 0.5 \
  --ncc-sigma 1.5 \
  --ncc-mask-radius 10 \
  --device cuda \
  --fail-on-error
```

The script separately tests fixed translation, rotation, and joint corrections
and writes overall, per-complex, and initial-RMSD-stratified summaries. It reads
ground truth only for evaluation and never uses it to choose or accept a
correction.

## Iterative translation screening before G1

Use existing baseline poses to measure whether a small safe translation step
accumulates into a useful correction before editing reverse diffusion:

```bash
python evaluation/evaluate_iterative_translation_screening.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --candidate-root artifacts/oracle_density_ablation/baseline \
  --output-dir artifacts/oracle_local/iterative_translation_val50 \
  --limit-complexes 50 \
  --num-predictions 40 \
  --step-caps 0.02,0.05,0.10 \
  --iterations 9 \
  --ncc-sigma 1.5 \
  --ncc-mask-radius 10 \
  --density-weight 2.0 \
  --device cuda \
  --fail-on-error
```

The script batches all poses for one complex, accepts a step only when NCC
increases, and writes per-step path/net displacement plus complex-level paired
top-1, best-of-N, rescue, and break metrics. `fixed_confidence_top1` retains the
original confidence ordering because this post-hoc diagnostic does not rerun
the confidence model. The complete G1 pipeline must rescore corrected poses.

## G1 late fixed-step translation correction

G1 runs after each complete DiffDock-Pocket tr/rot/torsion reverse step. During
the last 30% of steps it proposes at most 0.02 A of translation along the raw
Gaussian-NCC gradient. It accepts only an NCC-increasing, in-bounds proposal;
otherwise the model-only state is retained exactly. G1 does not norm-match the
density gradient and does not modify rotation or torsion.

```bash
INPUT_CSV=artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
OUT_ROOT=artifacts/oracle_local/g1_translation_val50 \
CONDITION=both \
SAMPLES=40 BATCH_SIZE=10 INFERENCE_STEPS=30 SEED=20260813 \
G1_STEP_CAP=0.02 G1_START=0.7 G1_PROTECTED_FRACTION=0.0 \
bash scripts/run_g1_translation_ablation.sh
```

Each G1 result directory contains
`density_translation_correction_audit.json`; every baseline and G1 directory
contains `particle_ranking.json`, which preserves the pre-confidence particle
identity for paired trajectory analysis. Summarize those audits without GT:

```bash
python evaluation/summarize_g1_translation_audits.py \
  --g1-results artifacts/oracle_local/g1_translation_val50/g1 \
  --baseline-results artifacts/oracle_local/g1_translation_val50/baseline \
  --output-dir artifacts/oracle_local/g1_translation_val50/audit_summary \
  --fail-on-error
```

Evaluate baseline and G1 with the same ligand-accuracy and density-reranking
commands. The inference confidence model is run after G1, so unlike cumulative
post-hoc screening, G1 confidence scores correspond to the final corrected
poses.

## C1/C2 x0 and torsion guidance ablation

The existing rigid guidance path remains unchanged unless the new flags are
passed. `--density_guidance_x0` evaluates the map objective on the first-order
Tweedie estimate `x0 ~= xt + sigma^2 * score`; `--density_guidance_torsion`
projects the Cartesian density gradient onto ligand rotatable bonds. C2 enables
both options. The torsion projection uses the same positive-angle convention
as the native sampler and is applied to the model torsion score before the
reverse-SDE update.

Run paired baseline/C1/C2 experiments with five seeds:

```bash
INPUT_CSV=artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
OUT_ROOT=artifacts/oracle_local/density_guidance_c1_c2_val50 \
SAMPLES=40 BATCH_SIZE=10 INFERENCE_STEPS=30 \
SEEDS=20260813,20260814,20260815,20260816,20260817 \
bash scripts/run_density_guidance_c1_c2_ablation.sh
```

Keep the same seed for baseline, C1, and C2. Evaluate both top-1 and
best-of-40 per complex; do not interpret a single seed as evidence of a
generation improvement.
