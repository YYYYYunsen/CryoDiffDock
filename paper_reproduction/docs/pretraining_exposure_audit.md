# DiffDock pretraining sequence-exposure audit

This workflow tests the frozen 174-complex CryoLigandDB drug-like cohort against
the PDBBind receptors used to train DiffDock-L and DiffDock-Pocket. It reports
exact PDB overlap, contact-to-all-receptor sequence exposure (primary), and
contact-to-contact exposure (secondary). It never reads docking outcomes.

## Requirements

- The project conda environment (`gemmi`, RDKit, pandas and NumPy).
- MMseqs2 available as `mmseqs`.
- The exact PDBBind directory and protein-file variant used by each checkpoint.

Both checked-in model repositories currently contain the same 16,379-complex
training split. Their split SHA256 is
`b943f37a45c68dabdc31a5beba86997b1820446346a618a7ab1132662423562a`.
Do not assume that their receptor files or additional checkpoint provenance are
identical; record those separately.

Before interpreting the audit, inspect the frozen checkpoint configuration for
self-distillation, van-der-Mer or other receptor-only augmentation sources. If
such sequences were used, include each source with a repeated
`--additional-training-fasta` argument when calling the Python audit directly.
Otherwise the result covers PDBBind exposure only, not total model exposure.

## 1. Run the DiffDock-L audit

Adjust the checkpoint and PDBBind path if the server uses a different layout.

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em

MODEL_NAME=DiffDock-L \
TRAIN_RECEPTOR_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM/data/PDBBind_processed \
TRAIN_PROTEIN_PATTERN='{pdb_id_lower}/{pdb_id_lower}_protein_processed.pdb' \
CHECKPOINT=/Scr/yunsenz2/data/software/DiffDock/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt \
OUTPUT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/diffdock_l \
bash scripts/run_diffdock_pretraining_exposure_audit.sh
```

## 2. Run the DiffDock-Pocket audit

Use the protein file actually configured for the frozen model. The command
below follows the training command documented in this repository.

```bash
MODEL_NAME=DiffDock-Pocket \
TRAIN_RECEPTOR_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/data/PDBBIND_atomCorrected \
TRAIN_PROTEIN_PATTERN='{pdb_id_lower}/{pdb_id_lower}_protein_esmfold_aligned_tr_fix.pdb' \
CHECKPOINT=/path/to/frozen/diffdock-pocket/checkpoint.pt \
OUTPUT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/diffdock_pocket \
bash scripts/run_diffdock_pretraining_exposure_audit.sh
```

If the actual checkpoint used another protein-file variant, replace
`TRAIN_PROTEIN_PATTERN`; the audit must reflect the model's real training input.

The extraction runs over all 16,379 training complexes and can take time. To
separate preparation and MMseqs execution, call
`evaluation/audit_diffdock_pretraining_exposure.py` with `--stage prepare`,
`--stage search`, and `--stage summarize` in sequence, keeping all other
arguments identical.

The wrapper mirrors PDBBind preprocessing by requiring a readable SDF or MOL2
ligand before counting a split entry as an effective training example. Entries
without a usable ligand are reported in `training_exclusions.csv`; they are not
treated as unresolved exposure. Protein lookup prioritizes the configured model
input and then checks the released experimental-structure fallbacks. Preserve
the ordered patterns and checkpoint configuration in `provenance.json`.

## 3. Inspect completeness before interpreting novelty

```bash
AUDIT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/diffdock_pocket
cat "$AUDIT_ROOT/audit_summary.json"
column -s, -t < "$AUDIT_ROOT/overlap_summary.csv" | less -S
column -s, -t < "$AUDIT_ROOT/extraction_failures.csv" | less -S
```

The primary claim is valid only when `primary_audit_complete` is true. The
contact-to-contact result additionally requires
`contact_to_training_contact_audit_ready` in `provenance.json` to be true.

## 4. Stratify Local performance

Replace the two metric paths with the authoritative 174-case per-complex files.
Missing predictions remain in the fixed denominator and count as failures.

```bash
AUDIT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/diffdock_pocket
LOCAL_OUT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/local_performance

python -u evaluation/analyze_pretraining_exposure_performance.py \
  --exposure "$AUDIT_ROOT/per_complex_best_hits.csv" \
  --scope all_receptor \
  --baseline-table /path/to/diffdock_pocket_174_per_complex.csv \
  --baseline-name 'DiffDock-Pocket' \
  --baseline-metric top1_rmsd_A \
  --guided-table /path/to/cryodiffdock_local_v2_174_per_complex.csv \
  --guided-name 'CryoDiffDock-Local V2' \
  --guided-metric top1_rmsd_A \
  --output-dir "$LOCAL_OUT"
```

## 5. Stratify Global performance

The existing Global table contains both columns, so the same file can be
passed twice with different metric names.

```bash
AUDIT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/diffdock_l
GLOBAL_TABLE=/Scr/yunsenz2/data/DiffDock-Pocket-EM/visualizations/figure2_global/figure2_panel_a_final/panel_a_per_complex.csv
GLOBAL_OUT=/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/paper_test/pretraining_exposure/global_performance

python -u evaluation/analyze_pretraining_exposure_performance.py \
  --exposure "$AUDIT_ROOT/per_complex_best_hits.csv" \
  --scope all_receptor \
  --baseline-table "$GLOBAL_TABLE" \
  --baseline-name 'DiffDock-L' \
  --baseline-metric diffdock_top1_rmsd_A \
  --guided-table "$GLOBAL_TABLE" \
  --guided-name 'CryoDiffDock-Global' \
  --guided-metric cryodiffdock_global_top1_rmsd_A \
  --output-dir "$GLOBAL_OUT"
```

The main output is `performance_by_exposure_summary.csv`, containing the fixed
cohort and the `<90%`, `<50%`, and `<30%` identity subsets, paired bootstrap
confidence intervals, gained/lost cases, and exact McNemar p-values.
