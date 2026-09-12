# CryoDiffDock-Pocket development baseline

The current quantitative results, frozen cohort identity, negative ablations,
and selector decisions are recorded in
[`CRYODIFFDOCK_POCKET_VALIDATION_MEMORY.md`](CRYODIFFDOCK_POCKET_VALIDATION_MEMORY.md).
Update that file whenever a validation gate is completed; keep this file
focused on architecture and forward plans.

## Repository boundary

This checkout is the development base for flexible cryo-EM docking. Preserve
the upstream DiffDock-Pocket implementation and put new project code under
`cryodiffdock_pocket/`, with thin CLI entry points in `evaluation/` and
`scripts/`. Density utilities may be ported selectively from DiffDock-EM, but
the two repositories must not be merged wholesale because their sampling and
all-atom graph implementations are incompatible.

Each generated candidate is a paired object:

```text
(ligand pose, flexible protein conformer, confidence, density features)
```

Never rank or refine the ligand SDF independently of its corresponding protein
PDB.

## Stage gates

1. **Runtime gate** — official `3dpf` inference produces both `rank1.sdf` and
   `rank1_protein.pdb` on the target GPU server.
2. **Candidate-coverage gate** — on a 20–50 complex pilot with distinct
   apo/input and holo/reference proteins, test whether best-of-N side-chain
   RMSD improves over the input. If the deposited protein is both input and
   reference, improvement is mathematically unavailable; report deposited
   conformation retention and joint ligand<2 A/side-chain<1 A coverage instead.
   Do not claim side-chain improvement from that oracle-input condition.
3. **Density-selection gate** — fixed density features improve top-1 selection
   toward the best-of-N oracle on validation data.
4. **Joint-refinement gate** — only after selection works, add differentiable
   ligand plus side-chain chi refinement with movement, clash, rotamer and
   density-improvement acceptance checks.
5. **Fine-tuning gate** — only after the inference-only ablations establish an
   incremental density signal.

## Evaluation conditions

Materialize independent paired ligand/protein outputs for:

- `input`: starting apo/predicted protein conformation;
- `confidence_only`: native DiffDock-Pocket ranking;
- `density_aware`: native candidates reranked by cryo-EM density;
- `density_aware_refine`: accepted joint refinements appended to the original
  candidate pool.

Ground-truth deposited coordinates are read only by evaluators. Deployment and
candidate-generation code must never use RMSD or deposited ligand/protein
coordinates.

## Baseline metrics

Run `evaluation/evaluate_sidechain_accuracy.py` on the exact flexible residue
set. It reports backbone-aligned side-chain heavy-atom RMSD, per-residue RMSD,
chi-angle error, chi accuracy and symmetry-corrected terminal atom matching.
Also retain ligand symmetry-aware RMSD and protein–ligand / side-chain clash
metrics from the parent projects.

Example batch manifest:

```csv
complex_name,condition,rank,reference_protein,input_protein,prediction_protein,flexible_residues
3dpf,confidence_only,1,/data/3dpf_holo.pdb,/data/3dpf_apo.pdb,/results/3dpf/rank1_protein.pdb,A:160-A:193-A:197
```

```bash
python evaluation/evaluate_sidechain_accuracy.py batch \
  --manifest pilot_manifest.csv \
  --output-dir artifacts/pilot_sidechain_metrics \
  --fail-on-error
```

For an existing DiffDock-Pocket candidate pool, evaluate all paired ligand and
protein outputs together:

```bash
python evaluation/evaluate_sidechain_candidate_coverage.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --results artifacts/oracle_density_ablation/baseline \
  --num-predictions 40 \
  --sidechain-threshold 1.0 \
  --ligand-threshold 2.0 \
  --output-dir artifacts/oracle_local/sidechain_coverage_val50 \
  --fail-on-error
```

## Density guidance physical gate

Keep density guidance and physical validity as separate stages.  Score final
baseline/guided candidates against the input protein without reading GT:

```bash
python scripts/score_candidate_physics.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --scores artifacts/oracle_local/baseline_guided_map_physics/ncc_scores/pose_scores.csv \
  --output-dir artifacts/oracle_local/baseline_guided_map_physics/physics_scores \
  --workers 8 \
  --fail-on-error
```

Then use references only in the validation evaluator to calibrate clash gates
and quantify RMSD/NCC/physics tradeoffs:

```bash
python evaluation/evaluate_density_physics_gate.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --scores artifacts/oracle_local/baseline_guided_map_physics/ncc_scores/pose_scores.csv \
  --physics artifacts/oracle_local/baseline_guided_map_physics/physics_scores/per_pose_physics.csv \
  --output-dir artifacts/oracle_local/baseline_guided_map_physics/physical_gate \
  --pools baseline,guided,union \
  --density-column ncc_sigma_1p5 \
  --density-weight 2.0 \
  --reference-quantile 0.95 \
  --min-in-bounds-fraction 0.95 \
  --fallback-policy best_score \
  --fail-on-error
```

The first pass leaves strain as an audit field because MMFF coverage may be
poor for cofactors.  After inspecting coverage, enable a validation-calibrated
strain gate with `--strain-reference-quantile 0.95`; keep
`--missing-strain-policy allow` unless a separately validated policy is used.

## Server workflow

The sync helper updates repository code without deleting server-only datasets,
weights, caches, or results:

```bash
bash scripts/sync_code_to_server.sh SSH_HOST /remote/path/DiffDock-Pocket-EM
ssh SSH_HOST
cd /remote/path/DiffDock-Pocket-EM
conda activate diffdock-pocket-em
python -m unittest \
  tests.test_density_guidance \
  tests.test_prepare_density_guidance_crop \
  tests.test_inference_cli_source \
  tests.test_sidechain_accuracy
REQUIRE_CUDA=1 bash scripts/run_official_smoke_test.sh
SAMPLES=2 BATCH_SIZE=1 INFERENCE_STEPS=5 \
  bash scripts/run_density_guidance_smoke_test.sh
```

Model weights are downloaded automatically by upstream `inference.py` when the
server can access the GitHub release. Keep weights and datasets outside version
control.
