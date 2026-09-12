# Frozen Global-to-Local cascade protocol

This protocol replaces the centre-coverage proxy with actual Local V2 docking
at CryoDiffDock-Global predicted centres.  It uses the authoritative 192-case
drug-like test cohort as the reporting denominator.  Global outputs exist for
183 cases; 174 of those also have the frozen Local source inputs required to
launch a cascade.  All unavailable or failed cases remain failures in the
192-case final summary and are never replaced.

The predefined outputs are:

- operational Top-1: validation-frozen Local V2 launched from the Global rank-1
  centre;
- Top-5 multi-start coverage: best RMSD among one frozen Local V2 output from
  each of the first five Global centres.  This is a branch-coverage ceiling and
  is not labelled Top-1.

No reference coordinates are used to build a crop, choose a side chain, run
inference, score NCC, run the V2 hierarchical search, or select the final V2
pose.

## 1. Sync the implementation

Run locally:

```bash
cd /Users/yunsenzhang/Documents/2_mycode/1_diffdock/DiffDock-Pocket-EM

rsync -avz --relative \
  ./inference.py \
  ./scripts/build_cascade_predicted_center_manifest.py \
  ./scripts/prepare_predicted_center_density_dataset.py \
  ./scripts/merge_cascade_sharded_inference.py \
  ./scripts/run_cascade_4gpu.sh \
  ./cryodiffdock_pocket/adaptive_search.py \
  ./scripts/run_adaptive_density_search.py \
  ./scripts/run_cascade_v2_4gpu.sh \
  ./scripts/merge_cascade_sharded_v2.py \
  ./evaluation/summarize_cascade_branches.py \
  ./docs/GLOBAL_TO_LOCAL_CASCADE_PROTOCOL.md \
  yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/
```

## 2. Set frozen inputs

On the four-GPU server:

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
conda activate diffdock-pocket-em

PROJECT_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM/results_test_deposited_cofactor35_contact_entity_emdb_v1
GLOBAL_SCORES="$GLOBAL_ROOT/density_rerank_finetuned_confidence_top40.csv"

LOCAL_V1_ROOT="$PROJECT_ROOT/artifacts/paper_test/oracle_local_frozen_v1"
SOURCE_MANIFEST="$LOCAL_V1_ROOT/01_source/test_source_full_polymer.csv"

REPORT_COHORT="$PROJECT_ROOT/visualizations/figure2_global/figure2_drug_like_cohort.txt"
RUN_COHORT="$PROJECT_ROOT/visualizations/figure2_global/figure2_global_available_drug_like_cohort.txt"

CASCADE_ROOT="$PROJECT_ROOT/artifacts/paper_test/global_to_local_cascade_top5_frozen_v1"
SOURCE_ROOT="$CASCADE_ROOT/00_source"
PREPARED_ROOT="$CASCADE_ROOT/01_prepared"
INFERENCE_ROOT="$CASCADE_ROOT/02_inference"
NCC_ROOT="$CASCADE_ROOT/03_ncc"
V2_ROOT="$CASCADE_ROOT/04_v2_search"
METRICS_ROOT="$CASCADE_ROOT/05_metrics"

mkdir -p "$SOURCE_ROOT" "$METRICS_ROOT"

wc -l "$REPORT_COHORT" "$RUN_COHORT"
```

Expected cohort sizes are 192 and 183.  Do not substitute the legacy 163-case
cohort: that number came from an older ligand classification plus intersection
with oracle-Local evaluability.  The computational run cohort is built below as
the prespecified intersection with source availability; it is not selected from
any cascade outcome.

## 3. Build predicted-centre branches and density crops

First freeze the source-available run intersection.  The nine missing source
rows remain documented failures in the 192-case result.

```bash
CASCADE_RUN_COHORT="$SOURCE_ROOT/cascade_runnable_drug_like_cohort.txt"
MISSING_SOURCE_AUDIT="$SOURCE_ROOT/global_available_missing_source.csv"

python - \
  "$SOURCE_MANIFEST" \
  "$RUN_COHORT" \
  "$CASCADE_RUN_COHORT" \
  "$MISSING_SOURCE_AUDIT" <<'PY'
import sys
from pathlib import Path
import pandas as pd

source = pd.read_csv(sys.argv[1], dtype=str)
source_id = "complex_name" if "complex_name" in source else "pdb_id"
source_ids = set(source[source_id].astype(str).str.strip().str.upper())
requested = [
    line.split()[0].strip().upper()
    for line in Path(sys.argv[2]).read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(requested) != len(set(requested)):
    raise RuntimeError("Global-available cohort contains duplicate IDs")
runnable = [pdb_id for pdb_id in requested if pdb_id in source_ids]
missing = [pdb_id for pdb_id in requested if pdb_id not in source_ids]
Path(sys.argv[3]).write_text("".join(f"{pdb_id}\n" for pdb_id in runnable))
pd.DataFrame({
    "pdb_id": missing,
    "reason": "missing_source_manifest_row",
}).to_csv(sys.argv[4], index=False)
print("Global-available:", len(requested))
print("Cascade runnable:", len(runnable))
print("Missing source rows:", missing)
PY

python scripts/build_cascade_predicted_center_manifest.py \
  --global-scores "$GLOBAL_SCORES" \
  --source-manifest "$SOURCE_MANIFEST" \
  --cohort "$CASCADE_RUN_COHORT" \
  --output-dir "$SOURCE_ROOT" \
  --score-column combined_at_best_weights \
  --top-k 5 \
  --seed 20260813 \
  --flexible-distance 3.5 \
  --no-flexible-policy rigid \
  --fail-on-error \
  2>&1 | tee "$SOURCE_ROOT/build.log"

python scripts/prepare_predicted_center_density_dataset.py \
  --manifest "$SOURCE_ROOT/cascade_source_blind.csv" \
  --output-dir "$PREPARED_ROOT" \
  --box-size 28 \
  --spacing 0.75 \
  --mask-inner 1.5 \
  --mask-outer 2.5 \
  --fail-on-error \
  2>&1 | tee "$PREPARED_ROOT.log"
```

Create the row-aligned reveal manifest only after preparation:

```bash
python - \
  "$PREPARED_ROOT/inference_blind.csv" \
  "$SOURCE_ROOT/cascade_source_reveal.csv" \
  "$PREPARED_ROOT/evaluation_manifest.csv" <<'PY'
import sys
import pandas as pd

blind = pd.read_csv(sys.argv[1], dtype={"complex_name": str})
reveal = pd.read_csv(sys.argv[2], dtype={"complex_name": str})
evaluation = blind.merge(
    reveal[["complex_name", "reference_ligand"]],
    on="complex_name",
    how="left",
    validate="one_to_one",
)
if evaluation["reference_ligand"].isna().any():
    raise RuntimeError("Some prepared branches lack a reveal reference")
evaluation.to_csv(sys.argv[3], index=False)
print("branch rows:", len(evaluation))
print("parent complexes:", evaluation["parent_complex_name"].nunique())
print("references in blind manifest:", "reference_ligand" in blind.columns)
PY
```

Expected: 870 branch rows, 174 parents, and `references in blind manifest:
False`.

Before the full launch, smoke-test one ordinary branch and, when present, one
explicit rigid-fallback branch.  This tests the zero-flexible-residue model path
without reading RMSD:

```bash
python - \
  "$PREPARED_ROOT/inference_blind.csv" \
  "$PREPARED_ROOT/smoke.csv" <<'PY'
import sys
import pandas as pd

table = pd.read_csv(sys.argv[1])
ordinary = table[table["sidechain_policy"].eq("global_pose_atoms")].head(1)
rigid = table[table["sidechain_policy"].eq("rigid_receptor_fallback")].head(1)
smoke = pd.concat([ordinary, rigid], ignore_index=True).drop_duplicates("complex_name")
smoke.to_csv(sys.argv[2], index=False)
print(smoke[["complex_name", "sidechain_policy"]].to_string(index=False))
PY

CUDA_VISIBLE_DEVICES=0 \
INPUT_CSV="$PREPARED_ROOT/smoke.csv" \
OUT_ROOT="$CASCADE_ROOT/smoke_inference" \
CONDITION=both \
SAMPLES=2 \
BATCH_SIZE=2 \
INFERENCE_STEPS=30 \
SEED=20260813 \
bash scripts/run_oracle_density_ablation.sh
```

Do not inspect RMSD from this smoke test.  Continue only if both arms create two
poses for every smoke row.

## 4. Run baseline40 + guided40 on four GPUs

Use a persistent `tmux` session.  The command runs four independent processes,
waits for all of them, and then creates a row-aligned merged result tree.

```bash
mkdir -p "$INFERENCE_ROOT"

PROJECT_ROOT="$PROJECT_ROOT" \
MANIFEST="$PREPARED_ROOT/inference_blind.csv" \
OUTPUT_ROOT="$INFERENCE_ROOT" \
NUM_GPUS=4 \
bash scripts/run_cascade_4gpu.sh \
  2>&1 | tee "$INFERENCE_ROOT/driver.log"
```

With the measured 76 seconds per branch per arm, 870 branches and two arms
require approximately 9.2 GPU-wall-clock hours on four equally fast A6000s,
before NCC and expansion.  Request/use about eight CPU cores per GPU and at
least 128 GB aggregate RAM if available.

Completion audit:

```bash
python - "$PREPARED_ROOT/inference_blind.csv" "$INFERENCE_ROOT/merged" <<'PY'
import sys
from pathlib import Path
import pandas as pd

n = len(pd.read_csv(sys.argv[1]))
root = Path(sys.argv[2])
for arm in ("baseline", "guided"):
    directories = list((root / arm).glob("index*___*"))
    poses = list((root / arm).glob("index*___*/rank*_confidence*.sdf"))
    print(arm, "directories=", len(directories), "poses=", len(poses))
    if len(directories) != n or len(poses) != 40 * n:
        raise RuntimeError(f"Incomplete {arm} output")
PY
```

## 5. Apply the validation-frozen Local V2 method

```bash
python scripts/rerank_ligand_by_density_ncc.py \
  --manifest "$PREPARED_ROOT/inference_blind.csv" \
  --candidate-root baseline="$INFERENCE_ROOT/merged/baseline" \
  --candidate-root guided="$INFERENCE_ROOT/merged/guided" \
  --output-dir "$NCC_ROOT" \
  --num-predictions 40 \
  --sigmas 1.5 \
  --mask-radius 10 \
  --ranking-sigma 1.5 \
  --density-weight 2.0 \
  --materialize-top-k 1 \
  --fail-on-error

mkdir -p "$V2_ROOT"

PROJECT_ROOT="$PROJECT_ROOT" \
MANIFEST="$PREPARED_ROOT/inference_blind.csv" \
POSE_SCORES="$NCC_ROOT/pose_scores.csv" \
OUTPUT_ROOT="$V2_ROOT" \
NUM_GPUS=4 \
bash scripts/run_cascade_v2_4gpu.sh \
  2>&1 | tee "$V2_ROOT/driver.log"

cat "$V2_ROOT/merged/summary.json"
find "$V2_ROOT/merged/v2_selected" \
  -mindepth 1 -maxdepth 1 -type l | wc -l
```

The V2 call above exactly reuses the validation-frozen search parameters from
`scripts/run_frozen_v2_full.sh`: two NCC scales, depth 3, beam width 48, at most
10 active poses per root, a 25% coarse reserve, and the frozen translation,
rotation, torsion, basin, anchor, and clash settings.  `top1` materialization
writes only the deployable V2 selection for each branch; the five-branch
cascade coverage is still computed after reveal from five independently
selected V2 poses.

Seal the non-reference artifacts before RMSD evaluation:

```bash
find "$CASCADE_ROOT" -type f \
  ! -path "$METRICS_ROOT/*" \
  ! -name 'cascade_source_reveal.csv' \
  ! -name 'evaluation_manifest.csv' \
  -print0 | sort -z | xargs -0 sha256sum \
  > "$CASCADE_ROOT/pre_reveal_SHA256SUMS"
```

## 6. Evaluate actual cascade performance

```bash
python evaluation/evaluate_ligand_accuracy.py \
  --manifest "$PREPARED_ROOT/evaluation_manifest.csv" \
  --results "$V2_ROOT/merged/v2_selected" \
  --condition cascade_branch_frozen_v2 \
  --num-predictions 1 \
  --workers 16 \
  --output-dir "$METRICS_ROOT/branch_v2"

python evaluation/summarize_cascade_branches.py \
  --branch-manifest "$PREPARED_ROOT/inference_blind.csv" \
  --branch-metrics "$METRICS_ROOT/branch_v2/per_complex.csv" \
  --requested-cohort "$REPORT_COHORT" \
  --top-k 5 \
  --output-dir "$METRICS_ROOT/cascade_summary"

cat "$METRICS_ROOT/cascade_summary/summary.json"
column -s, -t "$METRICS_ROOT/cascade_summary/cascade_success_rates.csv"
```

The primary paper number is `top1_lt2A_pct`.  `best_of_5_lt2A_pct` is reported
as multi-start coverage.  Also report the same statistics on the 174
source-available cases as a sensitivity analysis, but retain 192 as the
end-to-end denominator.  V1 may be evaluated later as a prespecified internal
ablation, but it is not the default or headline cascade method.
