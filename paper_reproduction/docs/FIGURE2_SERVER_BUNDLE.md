# Figure 2 server-material bundle (victoria.ks.uiuc.edu)

The primary external comparison is the strict paired-complete intersection of
ChemEM Stage 1, CryoDiffDock-Local V2 selected, and EMERALD. Keep V1 in the
bundle as an internal method-development reference, but do not use V1 to define
the main three-software denominator. The external eligible cohort is the
pre-specified 99-case list; also report those 99 with missing/failed cases
counted as failures.

## 0. Sync the preparation code to victoria.ks.uiuc.edu

Run this on the local Mac before the server commands below:

```bash
LOCAL_ROOT=/Users/yunsenzhang/Documents/2_mycode/1_diffdock/DiffDock-Pocket-EM
REMOTE_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM

mkdir -p /Users/yunsenzhang/.ssh/sockets
chmod 700 /Users/yunsenzhang/.ssh/sockets
ssh yunsenz2@victoria.ks.uiuc.edu 'hostname; test -d /Scr/yunsenz2/data/DiffDock-Pocket-EM'

rsync -av \
  "$LOCAL_ROOT/scripts/collect_figure2_external_bundle.py" \
  "$LOCAL_ROOT/scripts/build_figure2_chemem_top1_paths.py" \
  "$LOCAL_ROOT/scripts/build_figure2_cryodiffdock_runtime.py" \
  "$LOCAL_ROOT/scripts/build_emerald99_recovery_summary.py" \
  "$LOCAL_ROOT/scripts/summarize_figure2_external.py" \
  "$LOCAL_ROOT/scripts/run_figure2_v1_expansion_timing.sh" \
  "$LOCAL_ROOT/scripts/subset_table_by_ids.py" \
  yunsenz2@victoria.ks.uiuc.edu:"$REMOTE_ROOT/scripts/"

rsync -av \
  "$LOCAL_ROOT/visualizations/figure5_chemem/extract_chemem_slurm_runtime.py" \
  yunsenz2@victoria.ks.uiuc.edu:"$REMOTE_ROOT/visualizations/figure5_chemem/"
```

## 1. Define the frozen paths

Run from the Pocket repository on victoria.ks.uiuc.edu:

```bash
POCKET_ROOT=/Scr/yunsenz2/data/DiffDock-Pocket-EM
GLOBAL_ROOT=/Scr/yunsenz2/data/DiffDock-EM/DiffDock-EM
ENV_BIN=/Scr/yunsenz2/data/software/miniconda3/envs/diffdock-pocket-em/bin

CHEMEM_ROOT="$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike100_v1"
CHEMEM_OLD_ROOT="$GLOBAL_ROOT/artifacts/paper_external/chemem_druglike60_v1"
EMERALD_ROOT="$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_fullpolymer_v1"
EMERALD_IDS="$GLOBAL_ROOT/artifacts/paper_external/emerald_druglike99_v1/frozen_99_ids.txt"
V1_ROOT="$POCKET_ROOT/artifacts/paper_test/oracle_local_frozen_v1"
V2_ROOT="$POCKET_ROOT/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1"

FIG2_WORK="$POCKET_ROOT/artifacts/figure2_prepare_v1"
FIG2_BUNDLE="$POCKET_ROOT/artifacts/figure2_external_local_bundle_v1"
mkdir -p "$FIG2_WORK"
cd "$POCKET_ROOT"

comm -23 \
  <(sort "$CHEMEM_ROOT/00_cohort/chemem100_ids.txt") \
  <(sort "$EMERALD_IDS") \
  >"$FIG2_WORK/excluded_from_external99.txt"
test "$(wc -l <"$FIG2_WORK/excluded_from_external99.txt")" -eq 1
cat "$FIG2_WORK/excluded_from_external99.txt"
```

Record the displayed ID and its result-independent, pre-run ineligibility
reason in the protocol audit. If the exclusion was chosen after observing a
run failure, the 99-case cohort is not defensible and the case must remain in
the attempted-cohort analysis as a failure.

## 2. Refresh the ChemEM100 accuracy tables

```bash
GLOBAL_ROOT="$GLOBAL_ROOT" POCKET_ROOT="$POCKET_ROOT" \
  bash visualizations/figure5_chemem/cohort_design/analyze_chemem100_on_server.sh

"$ENV_BIN/python" scripts/build_figure2_chemem_top1_paths.py \
  --benchmark "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  --new-per-pose "$CHEMEM_ROOT/02_live_analysis/new_chemem_per_pose.csv" \
  --historical-root "$CHEMEM_OLD_ROOT" \
  --output "$FIG2_WORK/chemem_top1_paths.csv"
```

The second command must print `completed cases missing rank-1 file: 0`.

## 3. Build the recovery-aware EMERALD 99-case summary

```bash
EMERALD_FINAL_DIR="$EMERALD_ROOT/final_analysis"

"$ENV_BIN/python" "$GLOBAL_ROOT/tools/local_docking/audit_emerald99_recovery.py" \
  --run-root "$EMERALD_ROOT" \
  --frozen-ids "$EMERALD_IDS" \
  --output-dir "$EMERALD_FINAL_DIR"

"$ENV_BIN/python" scripts/build_emerald99_recovery_summary.py \
  --recovery-audit "$EMERALD_FINAL_DIR/emerald99_recovery_audit.tsv" \
  --frozen-ids "$EMERALD_IDS" \
  --expected-completed 97 \
  --output "$EMERALD_FINAL_DIR/emerald_paper_summary_recovery_aware.tsv"
```

The original `merge_emerald99_chunks.py` sees only original chunk summaries
and omits recovery-round attempts; do not use its 55-completed summary. The
recovery-aware table must contain 99 rows: 97 completed, one failed, and one
interrupted. Use its `end_to_end_wall_s` for EMERALD runtime.

## 4. Extract ChemEM iteration runtimes

Combine old and new logs in deterministic order so a completed retry replaces
an older record for the same PDB ID:

```bash
rm -rf "$FIG2_WORK/chemem_log_links"
mkdir -p "$FIG2_WORK/chemem_log_links"

i=0
while IFS= read -r -d '' log; do
  i=$((i + 1))
  ln -s "$log" "$FIG2_WORK/chemem_log_links/$(printf '%05d' "$i")_$(basename "$log")"
done < <(find "$CHEMEM_OLD_ROOT" -type f -name '*.log' -print0 | sort -z)

while IFS= read -r -d '' log; do
  i=$((i + 1))
  ln -s "$log" "$FIG2_WORK/chemem_log_links/$(printf '%05d' "$i")_$(basename "$log")"
done < <(find "$CHEMEM_ROOT/01_stage1/logs" -type f -name '*.log' -print0 | sort -z)

"$ENV_BIN/python" visualizations/figure5_chemem/extract_chemem_slurm_runtime.py \
  --log-dir "$FIG2_WORK/chemem_log_links" \
  --glob '*.log' \
  --prefer-iteration-runtime \
  --output "$FIG2_WORK/chemem_iteration_runtime.csv"
```

Retain both the runtime CSV and its `.unpaired.csv` audit.

## 5. Build CryoDiffDock-Local V2 runtime

The V2 audit already has per-case expansion, selection, and materialization
time in `case_runtime_seconds`. The old inference run did not record a reliable
wall clock. If the exact baseline+guided SLURM `ElapsedRaw` is available, sum
it and pass the total to the runtime builder. Otherwise run this frozen 99-case
timing replicate:

```bash
"$ENV_BIN/python" scripts/subset_table_by_ids.py \
  --input "$V1_ROOT/02_prepared/inference_blind.csv" \
  --ids "$EMERALD_IDS" \
  --output "$FIG2_WORK/inference_blind_emerald99.csv"

TIMING_JOB=$(sbatch --parsable \
  --job-name=fig2-cryo-time \
  --partition=A6000 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=64G --time=48:00:00 \
  --output="$FIG2_WORK/cryodiffdock_inference_emerald99.slurm.out" \
  --error="$FIG2_WORK/cryodiffdock_inference_emerald99.slurm.err" \
  --export=ALL,POCKET_ROOT="$POCKET_ROOT",ENV_BIN="$ENV_BIN",FIG2_WORK="$FIG2_WORK" \
  --wrap='set -euo pipefail; cd "$POCKET_ROOT"; PATH="$ENV_BIN:$PATH" /usr/bin/time -p -o "$FIG2_WORK/cryodiffdock_inference_emerald99.time" env INPUT_CSV="$FIG2_WORK/inference_blind_emerald99.csv" OUT_ROOT="$FIG2_WORK/cryodiffdock_inference_timing_output" CONDITION=both SAMPLES=40 BATCH_SIZE=10 INFERENCE_STEPS=30 SEED=20260813 bash scripts/run_oracle_density_ablation.sh')
echo "TIMING_JOB=$TIMING_JOB"
printf '%s\n' "$TIMING_JOB" >"$FIG2_WORK/cryodiffdock_inference_timing_job_id.txt"
```

After that job reaches `COMPLETED`, record its allocation:

```bash
TIMING_JOB=$(cat "$FIG2_WORK/cryodiffdock_inference_timing_job_id.txt")
sacct -j "$TIMING_JOB" \
  --format=JobIDRaw,JobName%30,State,ElapsedRaw,AllocCPUS,ReqTRES%80 -P \
  >"$FIG2_WORK/cryodiffdock_inference_emerald99.sacct.txt"
```

The legacy V1 expansion did not record timing. To include the V1 runtime point,
run the frozen expansion and final selector once under a cohort-level timer:

```bash
V1_TIMING_ROOT="$FIG2_WORK/v1_expansion_selection_timing"
V1_JOB=$(sbatch --parsable \
  --job-name=fig2-v1-time \
  --partition=A6000 --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:1 --mem=64G --time=48:00:00 \
  --output="$FIG2_WORK/v1_expansion_selection.slurm.out" \
  --error="$FIG2_WORK/v1_expansion_selection.slurm.err" \
  --export=ALL,POCKET_ROOT="$POCKET_ROOT",V1_ROOT="$V1_ROOT",ENV_BIN="$ENV_BIN",COHORT_IDS="$EMERALD_IDS",OUTPUT_ROOT="$V1_TIMING_ROOT" \
  --wrap='set -euo pipefail; cd "$POCKET_ROOT"; bash scripts/run_figure2_v1_expansion_timing.sh')
echo "V1_JOB=$V1_JOB"
printf '%s\n' "$V1_JOB" >"$FIG2_WORK/v1_expansion_selection_job_id.txt"
```

After V1 reaches `COMPLETED`, record it and build both V1 and V2 runtime tables:

```bash
V1_JOB=$(cat "$FIG2_WORK/v1_expansion_selection_job_id.txt")
V1_TIMING_ROOT="$FIG2_WORK/v1_expansion_selection_timing"
sacct -j "$V1_JOB" \
  --format=JobIDRaw,JobName%30,State,ElapsedRaw,AllocCPUS,ReqTRES%80 -P \
  >"$FIG2_WORK/v1_expansion_selection.sacct.txt"

"$ENV_BIN/python" scripts/build_figure2_cryodiffdock_runtime.py \
  --cohort "$EMERALD_IDS" \
  --v2-audit "$V2_ROOT/per_complex_audit.csv" \
  --inference-time-file "$FIG2_WORK/cryodiffdock_inference_emerald99.time" \
  --inference-case-count 99 \
  --output "$FIG2_WORK/cryodiffdock_v2_method_native_runtime.csv" \
  --v1-expansion-time-file "$V1_TIMING_ROOT/v1_expansion_selection.time" \
  --v1-output "$FIG2_WORK/cryodiffdock_v1_method_native_runtime.csv"
```

If using SLURM accounting instead, run the builder with
`--inference-total-seconds BASELINE_ELAPSEDRAW_PLUS_GUIDED_ELAPSEDRAW` and save
the `sacct` output in the bundle. Never infer runtime from file modification
times.

V1 remains an internal V1-to-V2 reference; it is not one of the three methods
that define the primary external paired denominator.

## 6. Create the transfer bundle

```bash
if [[ -e "$FIG2_BUNDLE" || -e "$FIG2_BUNDLE.tar.gz" ]]; then
  echo "Choose a new FIG2_BUNDLE path; refusing to overwrite existing output" >&2
  exit 2
fi

"$ENV_BIN/python" scripts/collect_figure2_external_bundle.py \
  --cohort "$EMERALD_IDS" \
  --input-manifest "$V1_ROOT/02_prepared/inference.csv" \
  --metadata-manifest "$CHEMEM_ROOT/00_cohort/chemem100_cohort.csv" \
  --chemem-benchmark "$CHEMEM_ROOT/02_live_analysis/per_complex_benchmark.csv" \
  --v1-metrics "$V1_ROOT/11_metrics_reveal/frozen_s0/per_complex.csv" \
  --v2-metrics "$V2_ROOT/metrics_v2_selected/per_complex.csv" \
  --emerald-summary "$EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware.tsv" \
  --chemem-per-pose "$FIG2_WORK/chemem_top1_paths.csv" \
  --v1-per-pose "$V1_ROOT/11_metrics_reveal/frozen_s0/per_pose.csv" \
  --v2-per-pose "$V2_ROOT/metrics_v2_selected/per_pose.csv" \
  --chemem-runtime "$FIG2_WORK/chemem_iteration_runtime.csv" \
  --v1-runtime "$FIG2_WORK/cryodiffdock_v1_method_native_runtime.csv" \
  --v2-runtime "$FIG2_WORK/cryodiffdock_v2_method_native_runtime.csv" \
  --protocol-file "$V2_ROOT/summary.json" \
  --protocol-file "$EMERALD_ROOT/final_analysis/emerald_paper_summary_recovery_aware_audit.json" \
  --protocol-file "$EMERALD_ROOT/final_analysis/emerald99_recovery_summary.json" \
  --protocol-file "$FIG2_WORK/chemem_iteration_runtime.unpaired.csv" \
  --protocol-file "$FIG2_WORK/cryodiffdock_inference_emerald99.time" \
  --extra-file "$FIG2_WORK/excluded_from_external99.txt" \
  --extra-file "$CHEMEM_ROOT/00_cohort/chemem100_ids.txt" \
  --extra-file "$FIG2_WORK/cryodiffdock_inference_timing_job_id.txt" \
  --extra-file "$FIG2_WORK/cryodiffdock_inference_emerald99.sacct.txt" \
  --extra-file "$FIG2_WORK/cryodiffdock_inference_emerald99.slurm.out" \
  --extra-file "$FIG2_WORK/cryodiffdock_inference_emerald99.slurm.err" \
  --extra-file "$FIG2_WORK/v1_expansion_selection_job_id.txt" \
  --extra-file "$FIG2_WORK/v1_expansion_selection.sacct.txt" \
  --extra-file "$FIG2_WORK/v1_expansion_selection.slurm.out" \
  --extra-file "$FIG2_WORK/v1_expansion_selection.slurm.err" \
  --expected-three-software-n 87 \
  --output-dir "$FIG2_BUNDLE"
```

Do not set `--expected-three-software-n` on the first run. Discover it from
the completed tables and `bundle_summary.json`; then rerun to a new output path
with that count supplied as an assertion.

## 7. Generate statistics, archive, and transfer

```bash
"$ENV_BIN/python" scripts/summarize_figure2_external.py \
  --accuracy-long "$FIG2_BUNDLE/tables/accuracy_per_complex_long.csv" \
  --runtime-long "$FIG2_BUNDLE/tables/runtime_per_complex_long.csv" \
  --output-dir "$FIG2_BUNDLE/statistics" \
  --expected-frozen-n 99 \
  --expected-three-software-n 87 \
  --bootstrap-repeats 10000 \
  --seed 20260903

"$ENV_BIN/python" -m json.tool "$FIG2_BUNDLE/bundle_summary.json"
"$ENV_BIN/python" -m json.tool "$FIG2_BUNDLE/statistics/figure2_statistics_summary.json"
wc -l \
  "$FIG2_BUNDLE/tables/three_software_common_ids.txt" \
  "$FIG2_BUNDLE/tables/chemem_v1_v2_common_ids.txt"
find "$FIG2_BUNDLE/representative_cases" -name case_metadata.csv -print -exec sed -n '1,2p' {} \;
tar -czf "$FIG2_BUNDLE.tar.gz" -C "$(dirname "$FIG2_BUNDLE")" "$(basename "$FIG2_BUNDLE")"
sha256sum "$FIG2_BUNDLE.tar.gz" >"$FIG2_BUNDLE.tar.gz.sha256"
```

The summary must show frozen external cohort size 99 and the actual
ChemEM/V1/V2 and strict three-software paired sizes. Every selected case must have a
receptor, reference ligand, map, and rank-1 poses for all three external
methods; no required asset may say `MISSING`.

Download from the local machine:

```bash
scp yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/figure2_external_local_bundle_v1.tar.gz .
scp yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/artifacts/figure2_external_local_bundle_v1.tar.gz.sha256 .
shasum -a 256 -c figure2_external_local_bundle_v1.tar.gz.sha256
```

Bundle-to-panel mapping:

- panel a: `protocol/`;
- panel b: `tables/cohort_ledger.csv`;
- panels c-d: `tables/accuracy_per_complex_long.csv`, using
  `in_three_software_common` for the paired result, plus `statistics/`;
- panel e: `tables/runtime_per_complex_long.csv`;
- panel f: `representative_cases/` and the candidate table.

Label panel e as method-native runtime. Do not call these identical end-to-end
wall times unless all methods are rerun with one shared boundary and matched
hardware policy.
