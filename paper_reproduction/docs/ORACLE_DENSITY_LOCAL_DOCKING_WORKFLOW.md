# Executable oracle-pocket density workflow

This workflow asks a controlled question: when the binding-site center is
known, does experimental cryo-EM density improve DiffDock-Pocket generation?
The native ligand supplies only its heavy-atom centroid and final RMSD target.

## 1. Create the source manifest

For the existing CryoDiffDock QC-core validation data, automatically select
the first five complete cases:

```bash
DATA_ROOT=/Scr/yunsenz2/data/EM-dataset-final/data

python scripts/build_oracle_density_source_manifest.py \
  --split "$DATA_ROOT/splits/em_val_qc_core_v1.txt" \
  --data-dir "$DATA_ROOT/processed_6radii_deposited_qc_extended_v1" \
  --reference-dir "$DATA_ROOT/deposited_ligand_references_complete_global_v1" \
  --map-csv "$DATA_ROOT/04_maps.csv" \
  --output artifacts/oracle_local/val_pilot_source.csv \
  --limit 5 \
  --flexible-distance 3.5
```

The builder skips incomplete entries, records them in
`val_pilot_source.failures.csv`, and derives the shared flexible-sidechain list
from side-chain heavy atoms within 3.5 Å of the deposited ligand, matching the
DiffDock-Pocket flex-distance convention. This is also oracle pocket information
and is held identical for baseline and guided conditions.

To build the complete validation manifest after the pilot, use `--limit 0`.
The resulting CSV has one row per complex:

```csv
complex_name,experimental_protein,ligand,reference_ligand,map_path,flexible_sidechains
case001,/data/case001/input_protein.pdb,/data/case001/ligand.sdf,/data/case001/native_ligand.sdf,/data/case001/emd_map.mrc,A:10-A:14-A:18
```

`experimental_protein` must already be in the map/reference coordinate frame.
`ligand` provides the chemical graph; do not use `--keep_local_structures` in
the ablation. `flexible_sidechains` must be an explicit `CHAIN:RESID-...` list
shared by both conditions; automatic selection could accidentally depend on
the ligand input coordinates. Relative paths are resolved relative to the
source manifest.

## 2. Prepare GT-center experimental-density crops

The docking graph and density mask have different structure inputs.  The
cropped `experimental_protein` remains the DiffDock-Pocket receptor, whereas
`mask_structure` must be the complete deposited/input PDB or mmCIF.  Add that
column to an existing manifest without changing its row order:

```bash
python scripts/add_full_polymer_mask_paths.py \
  --manifest artifacts/oracle_local/source.csv \
  --raw-root /data/raw_per_pdb \
  --output artifacts/oracle_local/source_full_polymer.csv \
  --fail-on-error
```

The recommended corrected crop reproduces the validated full-polymer policy:

```bash
python scripts/prepare_oracle_density_dataset.py \
  --manifest artifacts/oracle_local/source_full_polymer.csv \
  --output-dir artifacts/oracle_local/prepared_full_polymer_hard35 \
  --box-size 28 \
  --spacing 0.75 \
  --mask-policy full_polymer \
  --mask-style hard \
  --mask-radius 3.5 \
  --fail-on-error
```

This masks protein, nucleic-acid and other-polymer heavy atoms from the full
model while leaving every non-polymer unmasked.  The command refuses to run if
`mask_structure` is absent, preventing silent fallback to the cropped receptor.
For a source-only ablation, hold the old mask geometry fixed and use
`--mask-style soft --mask-inner 1.5 --mask-outer 2.5`.

The following command is retained only to reproduce the legacy cropped-mask
condition:

```bash
python scripts/prepare_oracle_density_dataset.py \
  --manifest /data/oracle_local/val_pilot_source.csv \
  --output-dir artifacts/oracle_local/prepared_val_pilot \
  --box-size 28 \
  --spacing 0.75 \
  --mask-policy legacy_cropped_protein \
  --mask-style soft \
  --mask-inner 1.5 \
  --mask-outer 2.5 \
  --fail-on-error
```

The default soft mask suppresses map signal near polymer protein heavy atoms.
It does not use ligand atom shape. Outputs are:

```text
prepared_val_pilot/inference.csv
prepared_val_pilot/preparation_report.csv
prepared_val_pilot/crops/*.npz
prepared_val_pilot/crops/*.json
```

Every row should have `native_ligand_atom_coverage=1.0`. Inspect failures and
coordinate-frame problems before inference:

```bash
python - <<'PY'
import pandas as pd
p = "artifacts/oracle_local/prepared_val_pilot/preparation_report.csv"
d = pd.read_csv(p)
print(d[["complex_name", "status", "native_ligand_atom_coverage", "finite_map_fraction"]])
assert (d["status"] == "ok").all()
assert (d["native_ligand_atom_coverage"] == 1.0).all()
PY
```

Use `--no-protein-mask` only for an explicit raw-map control. Use
`--overwrite` when intentionally regenerating existing crops.

## 3. Run paired baseline and guided sampling

Start with 2–5 complexes to verify paths, then use 20 validation complexes.

```bash
INPUT_CSV=artifacts/oracle_local/prepared_val_pilot/inference.csv \
OUT_ROOT=artifacts/oracle_local/val_pilot \
SAMPLES=40 \
BATCH_SIZE=10 \
INFERENCE_STEPS=30 \
SEED=20260813 \
CONDITION=both \
bash scripts/run_oracle_density_ablation.sh
```

This runs identical inputs and seeds for:

```text
artifacts/oracle_local/val_pilot/baseline
artifacts/oracle_local/val_pilot/guided
```

The guided defaults are `lambda_tr=0.5`, `lambda_rot=0.5`, `start=0.6`,
and `clip_ratio=1.0`. Override them only on validation, for example:

```bash
DENSITY_LAMBDA_TR=0.25 DENSITY_LAMBDA_ROT=0.25 \
DENSITY_GUIDANCE_START=0.5 DENSITY_CLIP_RATIO=1.0 \
INPUT_CSV=artifacts/oracle_local/prepared_val_pilot/inference.csv \
OUT_ROOT=artifacts/oracle_local/val_pilot_lambda025 \
CONDITION=guided \
bash scripts/run_oracle_density_ablation.sh
```

## 4. Evaluate ligand RMSD

```bash
MANIFEST=artifacts/oracle_local/prepared_val_pilot/inference.csv

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$MANIFEST" \
  --results artifacts/oracle_local/val_pilot/baseline \
  --condition baseline \
  --num-predictions 40 \
  --output-dir artifacts/oracle_local/val_pilot/metrics_baseline \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$MANIFEST" \
  --results artifacts/oracle_local/val_pilot/guided \
  --condition guided \
  --num-predictions 40 \
  --output-dir artifacts/oracle_local/val_pilot/metrics_guided \
  --fail-on-error
```

Each condition writes per-pose RMSD, per-complex top-1/best-of-N metrics, and a
summary JSON. Compare the paired complexes with:

```bash
python evaluation/compare_ligand_conditions.py \
  --baseline artifacts/oracle_local/val_pilot/metrics_baseline/per_complex.csv \
  --guided artifacts/oracle_local/val_pilot/metrics_guided/per_complex.csv \
  --output-dir artifacts/oracle_local/val_pilot/paired_comparison
```

Generation improves only when guided `best_of_n_rmsd_A` improves. A top-1-only
gain is a ranking effect. Freeze all parameters after validation before using
the test split.

## 5. Gaussian ligand-density NCC reranking

Score an existing baseline/guided candidate union without reading deposited
ligand coordinates:

```bash
python scripts/rerank_ligand_by_density_ncc.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --candidate-root baseline=artifacts/oracle_density_ablation/baseline \
  --candidate-root guided=artifacts/oracle_density_ablation/guided \
  --output-dir artifacts/oracle_density_ablation/ncc_ranking \
  --num-predictions 40 \
  --sigmas 1.0,1.5,2.0 \
  --mask-radius 10 \
  --ranking-sigma 1.5 \
  --density-weight 0.5 \
  --materialize-top-k 40 \
  --fail-on-error
```

The scoring stage uses only candidate SDFs, their confidence values, and the
experimental density crop. It writes `pose_scores.csv` and an optional
materialized ranking under `ncc_ranking/ranked/`. Supplying both roots retains
every baseline candidate as a fallback.

Select sigma and density weight on validation references only:

```bash
python evaluation/evaluate_density_ncc_ranking.py \
  --manifest artifacts/oracle_local/prepared_contact_entity_val50/inference.csv \
  --scores artifacts/oracle_density_ablation/ncc_ranking/pose_scores.csv \
  --output-dir artifacts/oracle_density_ablation/ncc_validation \
  --weight-grid 0,0.1,0.25,0.5,1,2 \
  --pools baseline,guided,union \
  --fail-on-error
```

This second command is explicitly evaluation-only: it reads deposited ligands
to compute RMSD, per-complex Spearman correlations, and the validation sweep.
Freeze the selected sigma/weight before evaluating a held-out test split.
