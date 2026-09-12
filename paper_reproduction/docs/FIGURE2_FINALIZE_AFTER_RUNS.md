# Finalize the Figure 2 frozen-99 data bundle

Use this workflow after ChemEM, EMERALD, and the CryoDiffDock-Local V2
Top-1-only timing run have finished. It performs no docking or search.

The primary external comparison contains ChemEM Stage 1,
CryoDiffDock-Local V2 selected, and EMERALD. CryoDiffDock-Local V1 is retained
as an internal accuracy reference, but its legacy oracle-materialization timing
must not be included in the primary runtime panel.

## 1. Paths

```bash
POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
ENV_BIN=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin

CHEMEM_ROOT="$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1"
CHEMEM_OLD_ROOT="$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike60_v1"
EMERALD_ROOT="$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1"
EMERALD_IDS="$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_v1/frozen_99_ids.txt"
V1_ROOT="$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1"

FIG2_WORK="$POCKET_ROOT/artifacts/figure2_prepare_v1"
V2_TOP1_ROOT="$FIG2_WORK/v2_frozen99_top1_runtime_v1"
FIG2_BUNDLE="$POCKET_ROOT/artifacts/figure2_external_local_bundle_final_v1"

cd "$POCKET_ROOT"
```

## 2. Evaluate the saved V2 Top-1 poses outside the runtime boundary

```bash
"$ENV_BIN/python" - \
  "$V2_TOP1_ROOT/evaluation_manifest.csv" \
  "$V1_ROOT/02_prepared/inference.csv" \
  "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" <<'PY'
import pandas as pd
import sys

blind_path, reference_path, output_path = sys.argv[1:]
blind = pd.read_csv(blind_path, dtype={"complex_name": str})
reference = pd.read_csv(reference_path, dtype={"complex_name": str})
reference = reference[["complex_name", "reference_ligand"]]
merged = blind.merge(reference, on="complex_name", how="left", validate="one_to_one", sort=False)
if len(merged) != 99 or merged["complex_name"].nunique() != 99:
    raise ValueError(f"Expected 99 unique V2 cases, found {len(merged)}")
if merged["reference_ligand"].isna().any():
    raise ValueError("Reference ligand missing after V2 reveal merge")
merged.to_csv(output_path, index=False)
print(f"V2_REVEAL_99_OK: {output_path}")
PY

"$ENV_BIN/python" evaluation/evaluate_ligand_accuracy.py \
  --manifest "$V2_TOP1_ROOT/evaluation_manifest_reveal.csv" \
  --results "$V2_TOP1_ROOT/v2_selected" \
  --condition v2_selected \
  --num-predictions 1 \
  --output-dir "$V2_TOP1_ROOT/metrics_v2_selected"
```

This evaluation occurs after `case_runtime_seconds` is recorded and therefore
does not add RMSD or best-of-N evaluation time to the deployment runtime.

## 3. Rebuild the failure-aware EMERALD summary

```bash
EMERALD_FINAL_DIR="$EMERALD_ROOT/final_analysis"

"$ENV_BIN/python" scripts/build_emerald99_recovery_summary.py \
  --recovery-audit "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv" \
  --frozen-ids "$EMERALD_IDS" \
  --expected-completed 97 \
  --output "$EMERALD_FINAL_DIR/emerald_paper_summary_recovery_aware_final.tsv"
```

Completed attempts use `runtime.tsv`. Failed or interrupted attempts use the
auditable interval from the first timestamp in `driver.log` to the last file
written inside that attempt directory.

## 4. Build the V2 frozen-99 deployment runtime table

```bash
"$ENV_BIN/python" scripts/extract_figure2_frozen_inference_runtime.py \
  --manifest "$V1_ROOT/02_prepared/inference_blind.csv" \
  --frozen-ids "$EMERALD_IDS" \
  --baseline-log "$V1_ROOT/03_inference/logs/baseline.log" \
  --guided-log "$V1_ROOT/03_inference/logs/guided.log" \
  --pose-scores "$V1_ROOT/04_ncc/pose_scores.csv" \
  --v2-audit "$V2_TOP1_ROOT/per_complex_audit.csv" \
  --matched-manifest-output "$FIG2_WORK/cryodiffdock_frozen99_manifest.csv" \
  --pose-score-output "$FIG2_WORK/cryodiffdock_frozen99_pose_scores.csv" \
  --output "$FIG2_WORK/cryodiffdock_v2_frozen99_top1_runtime.csv"
```

The output must report 99 matched cases, no missing pose scores, and no missing
V2 expansion-audit IDs.

## 5. Preflight the final inputs

```bash
"$ENV_BIN/python" - \
  "$EMERALD_IDS" \
  "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv" \
  "$V2_TOP1_ROOT/metrics_v2_selected/per_complex.csv" \
  "$EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_final.tsv" \
  "$FIG2_WORK/cryodiffdock_v2_frozen99_top1_runtime.csv" <<'PY'
from pathlib import Path
import pandas as pd
import sys

ids_path, chemem_path, v1_path, v2_path, emerald_path, runtime_path = sys.argv[1:]
ids = [line.strip().upper() for line in open(ids_path) if line.strip()]
if len(ids) != 99 or len(set(ids)) != 99:
    raise ValueError("Frozen cohort is not 99 unique IDs")

specs = (
    ("ChemEM", chemem_path, ","),
    ("V1", v1_path, ","),
    ("V2", v2_path, ","),
    ("EMERALD", emerald_path, "\t"),
    ("V2 runtime", runtime_path, ","),
)
for label, path, sep in specs:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep=sep)
    id_column = next(column for column in ("complex_name", "pdb_id") if column in frame.columns)
    frame[id_column] = frame[id_column].astype(str).str.upper()
    matched = frame[frame[id_column].isin(ids)]
    print(label, "frozen rows:", len(matched), "unique IDs:", matched[id_column].nunique())

v2 = pd.read_csv(v2_path)
runtime = pd.read_csv(runtime_path)
if len(v2) != 99 or v2["complex_name"].nunique() != 99:
    raise ValueError("V2 Top-1 metrics are not complete frozen-99")
if len(runtime) != 99 or runtime["complex_name"].nunique() != 99:
    raise ValueError("V2 deployment runtime is not complete frozen-99")
if not runtime["status"].astype(str).str.lower().eq("completed").all():
    raise ValueError("V2 deployment runtime contains incomplete cases")
print("FIGURE2_FINAL_INPUTS_OK")
PY
```

## 6. Create the bundle

Do not pass a hard-coded paired count on the discovery build. The final count
must be calculated from the completed tables.

```bash
"$ENV_BIN/python" scripts/collect_figure2_external_bundle.py \
  --cohort "$EMERALD_IDS" \
  --input-manifest "$V1_ROOT/02_prepared/inference.csv" \
  --metadata-manifest "$CHEMEM_ROOT/00_cohort/chemem100_cohort.csv" \
  --chemem-benchmark "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  --v1-metrics "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv" \
  --v2-metrics "$V2_TOP1_ROOT/metrics_v2_selected/per_complex.csv" \
  --emerald-summary "$EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_final.tsv" \
  --chemem-per-pose "$FIG2_WORK/chemem_top1_paths.csv" \
  --v1-per-pose "$V1_ROOT/11_metrics_reveal/frozen_s0/per_pose.csv" \
  --v2-per-pose "$V2_TOP1_ROOT/metrics_v2_selected/per_pose.csv" \
  --chemem-runtime "$FIG2_WORK/chemem_iteration_runtime.csv" \
  --v2-runtime "$FIG2_WORK/cryodiffdock_v2_frozen99_top1_runtime.csv" \
  --protocol-file "$V2_TOP1_ROOT/summary.json" \
  --protocol-file "$EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_final_audit.json" \
  --protocol-file "$EMERALD_ROOT/final_analysis/emerald99_recovery_summary.json" \
  --extra-file "$FIG2_WORK/cryodiffdock_frozen99_manifest.csv" \
  --extra-file "$FIG2_WORK/cryodiffdock_frozen99_pose_scores.csv" \
  --output-dir "$FIG2_BUNDLE"
```

The legacy V1 runtime is intentionally omitted because its timing includes
complete oracle-pool materialization. V1 accuracy remains in the bundle.

## 7. Statistics and archive

```bash
"$ENV_BIN/python" scripts/summarize_figure2_external.py \
  --accuracy-long "$FIG2_BUNDLE/tables/accuracy_per_complex_long.csv" \
  --runtime-long "$FIG2_BUNDLE/tables/runtime_per_complex_long.csv" \
  --output-dir "$FIG2_BUNDLE/statistics" \
  --expected-frozen-n 99 \
  --bootstrap-repeats 10000 \
  --seed 20260903

"$ENV_BIN/python" -m json.tool "$FIG2_BUNDLE/bundle_summary.json"
"$ENV_BIN/python" -m json.tool "$FIG2_BUNDLE/statistics/figure2_statistics_summary.json"

tar -czf "$FIG2_BUNDLE.tar.gz" \
  -C "$(dirname "$FIG2_BUNDLE")" "$(basename "$FIG2_BUNDLE")"
sha256sum "$FIG2_BUNDLE.tar.gz" >"$FIG2_BUNDLE.tar.gz.sha256"
```

The statistics output reports both the strict paired-complete accuracy set and
the full frozen-99 missing-as-failure result. Runtime summaries retain every
finite observed attempt, including failed or interrupted attempts, and provide
a completed-only secondary summary.
