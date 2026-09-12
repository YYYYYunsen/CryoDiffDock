# Cross-ligand score audit

This is a separate validation-only feasibility experiment.  The frozen
CryoDiffDock-Local score continues to select poses **within one ligand**.  Raw
absolute features from each ligand's selected pose are then audited for their
ability to rank different ligand identities for the same target.

## 1. Build a small blinded panel

Start with 5 validation targets and 20 candidates per target.  The target
manifest should be the already prepared, reference-free Local validation
manifest containing `experimental_protein`, `ligand_density_crop`, and the
frozen flexible-sidechain definition.  The chemistry manifest is Figure 4's
canonical one-row-per-complex table.

```bash
python scripts/build_cross_ligand_screening_panel.py \
  --target-manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --chemistry-manifest visualizations/figure4_cryoliganddb/data/complex_manifest.csv \
  --output-dir artifacts/ligand_screening/val_5x20/panel \
  --target-split val \
  --pool-splits train \
  --ligand-class drug_like \
  --num-targets 5 \
  --decoys-per-target 19 \
  --seed 20260906
```

`inference_blind.csv` contains no native ligand path or true/decoy label.
Every ligand SDF is regenerated from canonical SMILES with a fixed seed.
`answer_key.csv` is used only after all candidate poses and scores exist.

The existing ligand-centred crop makes this an **oracle-pocket ligand
retrieval** experiment.  It must not be described as blind binding-site
discovery.

## 2. Run the frozen pose pipeline unchanged

Use exactly the same sample count, inference settings, NCC settings, and Local
V2 parameters as the validated method.  Treat each row in the screening
manifest as an independent complex.  Do not pass `--keep_local_structures`.

```bash
PANEL=artifacts/ligand_screening/val_5x20/panel/inference_blind.csv
ROOT=artifacts/ligand_screening/val_5x20

INPUT_CSV="$PANEL" OUT_ROOT="$ROOT/v1" CONDITION=both \
  SAMPLES=40 BATCH_SIZE=10 INFERENCE_STEPS=30 SEED=20260813 \
  bash scripts/run_oracle_density_ablation.sh

python scripts/rerank_ligand_by_density_ncc.py \
  --manifest "$PANEL" \
  --candidate-root baseline="$ROOT/v1/baseline" \
  --candidate-root guided="$ROOT/v1/guided" \
  --output-dir "$ROOT/ncc" \
  --num-predictions 40 \
  --sigmas 1.0,1.5,2.0 \
  --ranking-sigma 1.5 \
  --fail-on-error

python scripts/run_adaptive_density_search.py \
  --manifest "$PANEL" \
  --scores "$ROOT/ncc/pose_scores.csv" \
  --output-dir "$ROOT/local_v2" \
  --device cuda \
  --density-sigmas 1.5,2.25 \
  --parent-count 6 \
  --search-depth 3 \
  --beam-width 48 \
  --beam-width-per-parent 10 \
  --coarse-beam-fraction 0.25 \
  --translation-steps 0.25,0.5 \
  --rotation-steps 10,20 \
  --torsion-steps 30,60 \
  --top-torsion-gradients 2 \
  --materialization-mode top1 \
  --fail-on-error
```

If the final paper method uses different frozen arguments, replace this block
with those exact arguments.  Do not tune them in this experiment.

## 3. Reveal labels and audit absolute scores

```bash
python evaluation/audit_cross_ligand_scores.py \
  --selected artifacts/ligand_screening/val_5x20/local_v2/selected.csv \
  --answer-key artifacts/ligand_screening/val_5x20/panel/answer_key.csv \
  --output-dir artifacts/ligand_screening/val_5x20/audit
```

Compare `multiscale_ncc`, `ncc`, and raw `confidence` against the current
within-ligand `final_selection_score`.  The primary outputs are:

- `retrieval_summary.csv`: Top-1, Top-5, MRR and target-bootstrap intervals;
- `true_ligand_ranks.csv`: the true ligand's rank for every target;
- `descriptor_bias.csv`: score correlations with molecular size and chemistry
  among decoys only;
- `candidate_scores.csv`: the complete merged audit table.

## 4. Decision sequence

1. Smoke test 5 targets x 20 candidates.
2. If an absolute score is better than random without strong size bias, repeat
   on 20 validation targets x 50 candidates.
3. Define any size correction or composite ligand score on validation only.
4. Freeze the formula, feature list, candidate count, and all hashes.
5. Evaluate once on a new retrieval holdout or a temporal cohort.

All candidates must receive the same pose count and adaptive-search budget.
Otherwise a maximum-score/extreme-value advantage can be mistaken for ligand
recognition.
