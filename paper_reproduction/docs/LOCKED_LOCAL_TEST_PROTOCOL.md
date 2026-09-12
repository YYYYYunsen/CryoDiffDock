# Locked independent-test protocol for CryoDiffDock-Pocket

This protocol evaluates the frozen local-docking method once on the existing
independent test split.  The benchmark is oracle-pocket redocking into the
deposited holo cryo-EM model: the deposited ligand is used to define the local
crop centre, but it is not available to candidate generation or selection.

Do not run any `sweep_*` program on the test set.  All commands through Section
8 are reference/RMSD-free.  Section 9 is the single reveal.

## 1. Sync and verify the new test-only code

Run locally:

```bash
cd /Users/yunsenzhang/Documents/2_mycode/1_diffdock/DiffDock-Pocket-EM

rsync -avz --relative \
  ./scripts/apply_frozen_discrete_ranking.py \
  ./scripts/apply_frozen_parent_relative_gnina_gate.py \
  ./scripts/freeze_local_test_protocol.py \
  ./evaluation/summarize_locked_test_results.py \
  ./tests/test_frozen_test_application.py \
  ./docs/CRYODIFFDOCK_POCKET_VALIDATION_MEMORY.md \
  ./docs/LOCKED_LOCAL_TEST_PROTOCOL.md \
  yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/
```

Run on the server/GPU allocation:

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em

python -m unittest -v \
  tests.test_frozen_test_application \
  tests.test_discrete_ranking_sweep \
  tests.test_parent_relative_gnina_gate \
  tests.test_parent_tournament_gnina_gate
```

Expected: `Ran 10 tests ... OK`.

## 2. Set paths and verify split identity

```bash
DATA_ROOT=/Scr/yunsenz2/data/EM-dataset-final/data
SPLIT_ROOT="$DATA_ROOT/splits_deposited_noncovalent_cofactor35_contact_entity_emdb_v1"
VAL_SPLIT="$SPLIT_ROOT/em_val.txt"
TEST_SPLIT="$SPLIT_ROOT/em_test.txt"

TEST_ROOT=artifacts/paper_test/oracle_local_frozen_v1
LOCK_ROOT="$TEST_ROOT/00_lock"
SOURCE_ROOT="$TEST_ROOT/01_source"
PREPARED_ROOT="$TEST_ROOT/02_prepared"
INFERENCE_ROOT="$TEST_ROOT/03_inference"
NCC_ROOT="$TEST_ROOT/04_ncc"
SEARCH_ROOT="$TEST_ROOT/05_discrete_search"
S0_TEST_ROOT="$TEST_ROOT/06_s0_frozen"
BASIN_ROOT="$TEST_ROOT/07_basin_support"
GNINA_CAND_ROOT="$TEST_ROOT/08_gnina_candidates"
GNINA_PARENT_ROOT="$TEST_ROOT/09_gnina_parents"
FINAL_TEST_ROOT="$TEST_ROOT/10_final_frozen"
METRICS_ROOT="$TEST_ROOT/11_metrics_reveal"

S0_VAL_ROOT=artifacts/oracle_local/discrete_density_ranking_sweep_s0_v1
FINAL_VAL_ROOT=artifacts/oracle_local/parent_relative_gnina_attempt1_v1

GNINA_ROOT=/Scr/yunsenz2/data/software/benchmark/GNINA
APPTAINER_BIN=$(command -v apptainer)
GNINA_SIF=$(find "$GNINA_ROOT" -maxdepth 3 -type f -name '*.sif' | head -n 1)

mkdir -p "$TEST_ROOT" "$SOURCE_ROOT" "$INFERENCE_ROOT/logs"

printf 'test split: %s\n' "$TEST_SPLIT"
wc -l "$VAL_SPLIT" "$TEST_SPLIT"
sha256sum "$VAL_SPLIT" "$TEST_SPLIT"
printf 'apptainer: %s\nGNINA image: %s\n' "$APPTAINER_BIN" "$GNINA_SIF"
```

Confirm that `em_test.txt` exists and `GNINA_SIF` is non-empty.  Then verify
that validation and test are disjoint:

```bash
python - "$VAL_SPLIT" "$TEST_SPLIT" <<'PY'
import sys

def ids(path):
    return {
        line.strip().split()[0].upper()
        for line in open(path)
        if line.strip() and not line.lstrip().startswith('#')
    }

validation = ids(sys.argv[1])
test = ids(sys.argv[2])
overlap = sorted(validation & test)
print('validation:', len(validation))
print('test:', len(test))
print('overlap:', len(overlap), overlap[:20])
print('SPLITS_DISJOINT:', len(overlap) == 0)
PY
```

Do not continue unless `SPLITS_DISJOINT: True`.

## 3. Freeze the method before touching test results

The model cache below is the default used by `inference.py`.  If the actual
validated run used explicit model directories, pass those exact directories
as additional `--artifact` arguments instead.

```bash
python scripts/freeze_local_test_protocol.py \
  --test-split "$TEST_SPLIT" \
  --s0-config "$S0_VAL_ROOT/best_config.json" \
  --final-config "$FINAL_VAL_ROOT/best_config.json" \
  --artifact .cache/model/v1.0.0 \
  --artifact "$GNINA_SIF" \
  --output-dir "$LOCK_ROOT"

cat "$LOCK_ROOT/protocol.json"
cat "$LOCK_ROOT/lock.json"
```

The lock directory is intentionally non-overwritable.  Re-running the same
command verifies hashes; a changed input requires a new versioned test root.

## 4. Build and prepare the complete test cohort

```bash
SOURCE_MANIFEST="$SOURCE_ROOT/test_source.csv"
SOURCE_FULLPOLY="$SOURCE_ROOT/test_source_full_polymer.csv"

python scripts/build_oracle_density_source_manifest.py \
  --split "$TEST_SPLIT" \
  --data-dir "$DATA_ROOT/processed_6radii_deposited" \
  --reference-dir "$DATA_ROOT/deposited_ligand_references_complete_global_v1" \
  --map-csv "$DATA_ROOT/04_maps_fixed.csv" \
  --output "$SOURCE_MANIFEST" \
  --limit 0 \
  --flexible-distance 3.5 \
  --fail-on-error
```

Recover the same complete-structure root that passed the val100 full-polymer
audit, then add those paths:

```bash
VAL_FULLPOLY=artifacts/oracle_local/full_polymer_mask_audit100/source_full_polymer100.csv

RAW_ROOT=$(python - "$VAL_FULLPOLY" <<'PY'
import sys
from pathlib import Path
import pandas as pd

table = pd.read_csv(sys.argv[1], dtype=str)
path = Path(table['mask_structure'].dropna().iloc[0]).resolve()
print(path.parent.parent)
PY
)

printf 'full-polymer raw root: %s\n' "$RAW_ROOT"

python scripts/add_full_polymer_mask_paths.py \
  --manifest "$SOURCE_MANIFEST" \
  --raw-root "$RAW_ROOT" \
  --output "$SOURCE_FULLPOLY" \
  --fail-on-error

python scripts/prepare_oracle_density_dataset.py \
  --manifest "$SOURCE_FULLPOLY" \
  --output-dir "$PREPARED_ROOT" \
  --box-size 28 \
  --spacing 0.75 \
  --mask-policy full_polymer \
  --mask-style soft \
  --mask-inner 1.5 \
  --mask-outer 2.5 \
  --fail-on-error
```

Seal the reference-containing manifest and make an operational copy with no
reference-ligand path:

```bash
EVAL_MANIFEST="$PREPARED_ROOT/inference.csv"
BLIND_MANIFEST="$PREPARED_ROOT/inference_blind.csv"

python - "$EVAL_MANIFEST" "$BLIND_MANIFEST" <<'PY'
import sys
import pandas as pd

table = pd.read_csv(sys.argv[1], dtype={'complex_name': str})
blind = table.drop(columns=['reference_ligand'])
blind.to_csv(sys.argv[2], index=False)
print('prepared cases:', len(table))
print('blind columns contain reference_ligand:', 'reference_ligand' in blind.columns)
print('BLIND_MANIFEST_OK:', len(table) == len(blind))
PY
```

## 5. Generate the two frozen 40-pose proposal arms

These are GPU jobs.  Run baseline and guided in the same environment and with
the same seed.

```bash
INPUT_CSV="$BLIND_MANIFEST" \
OUT_ROOT="$INFERENCE_ROOT" \
CONDITION=baseline \
SAMPLES=40 \
BATCH_SIZE=10 \
INFERENCE_STEPS=30 \
SEED=20260813 \
bash scripts/run_oracle_density_ablation.sh \
  2>&1 | tee "$INFERENCE_ROOT/logs/baseline.log"

INPUT_CSV="$BLIND_MANIFEST" \
OUT_ROOT="$INFERENCE_ROOT" \
CONDITION=guided \
SAMPLES=40 \
BATCH_SIZE=10 \
INFERENCE_STEPS=30 \
SEED=20260813 \
DENSITY_GUIDANCE_OBJECTIVE=atom_density \
DENSITY_LAMBDA_TR=0.5 \
DENSITY_LAMBDA_ROT=0.5 \
DENSITY_GUIDANCE_START=0.6 \
DENSITY_CLIP_RATIO=1.0 \
bash scripts/run_oracle_density_ablation.sh \
  2>&1 | tee "$INFERENCE_ROOT/logs/guided.log"
```

Check only completion and file counts, not RMSD:

```bash
python - "$BLIND_MANIFEST" "$INFERENCE_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

n = len(pd.read_csv(sys.argv[1], dtype={'complex_name': str}))
root = Path(sys.argv[2])
for arm in ('baseline', 'guided'):
    directories = list((root / arm).glob('index*___*'))
    poses = list((root / arm).glob('index*___*/rank*_confidence*.sdf'))
    print(arm, 'complexes=', len(directories), 'poses=', len(poses))
print('expected complexes per arm:', n)
print('expected poses per arm:', 40 * n)
PY
```

Resolve technical failures before continuing, without inspecting RMSD.  The
same case set must be used for every condition.

## 6. NCC scoring, deterministic expansion, and frozen S0

```bash
python scripts/rerank_ligand_by_density_ncc.py \
  --manifest "$BLIND_MANIFEST" \
  --candidate-root baseline="$INFERENCE_ROOT/baseline" \
  --candidate-root guided="$INFERENCE_ROOT/guided" \
  --output-dir "$NCC_ROOT" \
  --num-predictions 40 \
  --sigmas 1.5 \
  --mask-radius 10 \
  --ranking-sigma 1.5 \
  --density-weight 2.0 \
  --materialize-top-k 1 \
  --fail-on-error

python -u scripts/run_discrete_density_search.py \
  --manifest "$BLIND_MANIFEST" \
  --scores "$NCC_ROOT/pose_scores.csv" \
  --output-dir "$SEARCH_ROOT" \
  --density-column ncc_sigma_1p5 \
  --density-sigma 1.5 \
  --density-weight 2.0 \
  --parent-score-column confidence \
  --parent-count 6 \
  --top-score-parents 3 \
  --parent-diversity-rmsd 1.5 \
  --rotation-degrees 15,30,60 \
  --translation-distances 0.5,1.0 \
  --torsion-degrees 60,120,180 \
  --max-torsion-bonds 8 \
  --minimum-ncc-gain 0 \
  --anchor-penalty 0.05 \
  --max-centroid-shift 1.5 \
  --min-in-bounds-fraction 0.95 \
  --severe-overlap 0.8 \
  --mask-radius 10 \
  --materialize-expanded-limit 0 \
  --fail-on-error \
  2>&1 | tee "$SEARCH_ROOT.log"

python scripts/apply_frozen_discrete_ranking.py \
  --search-root "$SEARCH_ROOT" \
  --frozen-config "$LOCK_ROOT/s0_config.json" \
  --output-dir "$S0_TEST_ROOT"
```

## 7. Frozen basin support and GNINA gate

```bash
python scripts/score_coarse_basin_support.py \
  --search-root "$SEARCH_ROOT" \
  --s0-selected "$S0_TEST_ROOT/best_selected_candidates.csv" \
  --s0-config "$LOCK_ROOT/s0_config.json" \
  --manifest "$BLIND_MANIFEST" \
  --output-dir "$BASIN_ROOT" \
  --coarse-top-k 32 \
  --rotation-degrees 10,20 \
  --translation-distances 0.25,0.5 \
  --density-sigma 1.5 \
  --mask-radius 10 \
  --ncc-tolerance 0.01 \
  --severe-overlap 0.8 \
  --fail-on-error \
  2>&1 | tee "$BASIN_ROOT.log"

python scripts/score_discrete_candidates_gnina.py \
  --manifest "$BLIND_MANIFEST" \
  --search-root "$SEARCH_ROOT" \
  --candidate-table "$BASIN_ROOT/basin_scores.csv" \
  --s0-selected "$S0_TEST_ROOT/best_selected_candidates.csv" \
  --output-dir "$GNINA_CAND_ROOT" \
  --apptainer "$APPTAINER_BIN" \
  --gnina-sif "$GNINA_SIF" \
  --bind-root /Scr \
  --workers 8 \
  --cpu-per-task 1 \
  --timeout-seconds 300 \
  --fail-on-error \
  2>&1 | tee "$GNINA_CAND_ROOT.log"

PARENT_TABLE="$GNINA_PARENT_ROOT/parent_controls.csv"

python scripts/build_parent_gnina_controls.py \
  --basin-scores "$BASIN_ROOT/basin_scores.csv" \
  --search-root "$SEARCH_ROOT" \
  --output "$PARENT_TABLE" \
  --fail-on-error

python scripts/score_discrete_candidates_gnina.py \
  --manifest "$BLIND_MANIFEST" \
  --search-root "$SEARCH_ROOT" \
  --candidate-table "$PARENT_TABLE" \
  --output-dir "$GNINA_PARENT_ROOT/scores" \
  --apptainer "$APPTAINER_BIN" \
  --gnina-sif "$GNINA_SIF" \
  --bind-root /Scr \
  --workers 8 \
  --cpu-per-task 1 \
  --timeout-seconds 300 \
  --fail-on-error \
  2>&1 | tee "$GNINA_PARENT_ROOT/score.log"

python scripts/apply_frozen_parent_relative_gnina_gate.py \
  --basin-scores "$BASIN_ROOT/basin_scores.csv" \
  --candidate-gnina-scores "$GNINA_CAND_ROOT/gnina_scores.csv" \
  --parent-gnina-scores "$GNINA_PARENT_ROOT/scores/gnina_scores.csv" \
  --s0-config "$LOCK_ROOT/s0_config.json" \
  --frozen-config "$LOCK_ROOT/final_config.json" \
  --output-dir "$FINAL_TEST_ROOT"
```

## 8. Pre-reveal audit and output hashes

```bash
python - \
  "$NCC_ROOT/summary.json" \
  "$SEARCH_ROOT/summary.json" \
  "$S0_TEST_ROOT/summary.json" \
  "$BASIN_ROOT/summary.json" \
  "$GNINA_CAND_ROOT/summary.json" \
  "$GNINA_PARENT_ROOT/scores/summary.json" \
  "$FINAL_TEST_ROOT/summary.json" <<'PY'
import json
import sys

for path in sys.argv[1:]:
    data = json.load(open(path))
    print(path)
    print('  complexes:', data.get('num_complexes'))
    print('  failures:', data.get('num_failures', data.get('num_failed_candidates', 0)))
    print('  uses_ground_truth:', data.get('uses_ground_truth'))
PY

find "$TEST_ROOT" -type f \
  ! -path "$METRICS_ROOT/*" \
  -print0 | sort -z | xargs -0 sha256sum \
  > "$TEST_ROOT/pre_reveal_SHA256SUMS"

sha256sum "$TEST_ROOT/pre_reveal_SHA256SUMS"
```

Verify equal complex counts, zero failures, and `uses_ground_truth: False` for
selection stages.  Save the terminal output.  This is the last checkpoint at
which technical problems may be fixed without seeing accuracy.

## 9. Single RMSD reveal

Run this block once.  Do not alter the lock or rerun a different selector after
seeing the output.

```bash
mkdir -p "$METRICS_ROOT"

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$INFERENCE_ROOT/baseline" \
  --condition raw_baseline \
  --num-predictions 40 \
  --output-dir "$METRICS_ROOT/raw_baseline" \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$INFERENCE_ROOT/guided" \
  --condition density_mean_guided \
  --num-predictions 40 \
  --output-dir "$METRICS_ROOT/density_mean_guided" \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$NCC_ROOT/ranked" \
  --condition union_ncc \
  --num-predictions 1 \
  --output-dir "$METRICS_ROOT/union_ncc" \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$S0_TEST_ROOT/best_selected" \
  --condition frozen_s0 \
  --num-predictions 1 \
  --output-dir "$METRICS_ROOT/frozen_s0" \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$FINAL_TEST_ROOT/best_selected" \
  --condition frozen_final \
  --num-predictions 1 \
  --output-dir "$METRICS_ROOT/frozen_final" \
  --fail-on-error

python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$EVAL_MANIFEST" \
  --results "$SEARCH_ROOT/expanded_pool" \
  --condition expanded_pool \
  --num-predictions 100000 \
  --output-dir "$METRICS_ROOT/expanded_pool" \
  --fail-on-error

python evaluation/summarize_locked_test_results.py \
  --condition raw_baseline="$METRICS_ROOT/raw_baseline/per_complex.csv" \
  --condition density_mean_guided="$METRICS_ROOT/density_mean_guided/per_complex.csv" \
  --condition union_ncc="$METRICS_ROOT/union_ncc/per_complex.csv" \
  --condition frozen_s0="$METRICS_ROOT/frozen_s0/per_complex.csv" \
  --condition frozen_final="$METRICS_ROOT/frozen_final/per_complex.csv" \
  --condition expanded_pool="$METRICS_ROOT/expanded_pool/per_complex.csv" \
  --baseline raw_baseline \
  --bootstrap-repeats 10000 \
  --seed 20260821 \
  --output-dir "$METRICS_ROOT/paper_summary" \
  2>&1 | tee "$METRICS_ROOT/paper_summary.log"

python evaluation/compare_ligand_conditions.py \
  --baseline "$METRICS_ROOT/frozen_s0/per_complex.csv" \
  --guided "$METRICS_ROOT/frozen_final/per_complex.csv" \
  --output-dir "$METRICS_ROOT/paired_s0_vs_final"
```

The primary paper values are in
`11_metrics_reveal/paper_summary/locked_test_summary.json`.  Report the final
test result even if it is worse than validation.  The expanded-pool
`best_of_n` values are an oracle coverage ceiling and must not be described as
top-1 deployable accuracy.
