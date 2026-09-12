# Figure 2 authoritative-174 run

The frozen primary cohort is
`paper_design/cohort_definitions/authoritative_druglike174_v1/authoritative_druglike174_ids.txt`.
It is outcome-independent. Relative to the previous frozen 99 cohort, 94 cases
are reusable, 80 cases require new EMERALD runs, and 5 previous cases are
excluded. ChemEM reuses 95 historical attempts, including the recorded 9D6D
timeout, and therefore requires 79 new runs. CryoDiffDock-Local is collected
from existing completed outputs; the commands below do not rerun it.

## 1. Upload the changed code to Victoria

Run on the Mac:

```bash
rsync -avz \
  /Users/yunsenzhang/Documents/2_mycode/1_diffdock/DiffDock-Pocket-EM/scripts/ \
  yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/scripts/

rsync -avz \
  /Users/yunsenzhang/Documents/2_mycode/1_diffdock/DiffDock-Pocket-EM/paper_design/cohort_definitions/authoritative_druglike174_v1/ \
  yunsenz2@victoria.ks.uiuc.edu:/Scr/yunsenz2/data/DiffDock-Pocket-EM/paper_design/cohort_definitions/authoritative_druglike174_v1/
```

## 2. Prepare manifests and exact work lists

Run on Victoria in `diffdock-pocket-em`:

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM
bash scripts/prepare_figure2_authoritative174.sh
```

This writes:

- `artifacts/figure2_authoritative174_v1/02_worklists/authoritative174_ids.txt`
- `.../reusable_from_frozen99_94_ids.txt`
- `.../run_emerald_new80_ids.txt`
- `.../run_chemem_new79_ids.txt`
- `.../excluded_from_frozen99_5_ids.txt`
- `.../authoritative174_manifest.csv` and its blind counterpart
- `.../worklist_audit.json`

The preparation uses the rigid-receptor sentinel for the nine cases with no
torsion-bearing side chain inside 3.5 Å; it does not discard those cases.

## 3. Collect the already completed CryoDiffDock-Local run

The defaults merge the completed `frozen_full_v1` and `druglike171_v1` roots,
preferring the former. The collector matches `complex_name` exactly and requires
174 Top-1 RMSDs, 174 deployment runtimes, and 80 input poses per case.

```bash
bash scripts/collect_figure2_authoritative174_existing_local.sh
```

If the completed 174 run has a different location, point the collector at it:

```bash
LOCAL_MANIFESTS=/path/to/completed_run/inference_blind.csv \
LOCAL_POSE_SCORES=/path/to/completed_run/pose_scores.csv \
LOCAL_V2_AUDITS=/path/to/completed_run/per_complex_audit.csv \
LOCAL_V2_METRICS=/path/to/completed_run/metrics_v2_selected/per_complex.csv \
bash scripts/collect_figure2_authoritative174_existing_local.sh
```

The reported V2 time is `selection_ready_seconds + top1_materialization_seconds`.
Oracle/best-of-N materialization is explicitly excluded even if the original run
used benchmark materialization mode.

## 4. Preflight the 79 new ChemEM cases

```bash
bash scripts/prepare_figure2_authoritative174_external.sh
```

This performs input validation only; it does not run ChemEM or EMERALD.

## 5. Submit the 79 ChemEM cases

Submit sixteen independent five-case jobs. With four A6000 nodes currently
available, four jobs run first and the remaining jobs stay queued:

```bash
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 0 5
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 5 10
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 10 15
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 15 20
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 20 25
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 25 30
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 30 35
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 35 40
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 40 45
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 45 50
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 50 55
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 55 60
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 60 65
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 65 70
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 70 75
sbatch scripts/run_figure2_authoritative174_chemem_chunk_slurm.sh 75 79
```

Each job processes its five cases sequentially. Each case log records wrapper
start/end timestamps and elapsed seconds. If a 48-hour allocation ends before
the range finishes, resubmitting the same range safely skips completed cases.
Successful ChemEM runtime should still use the summed iteration runtime; the
wrapper elapsed time preserves time spent by failed or timed-out attempts.

## 6. Submit the 80 EMERALD cases on TitanV

```bash
for start in 0 13 26 39 52 65 78; do
  end=$((start + 13))
  if (( end > 80 )); then end=80; fi
  sbatch scripts/run_figure2_authoritative174_emerald_chunk_slurm.sh "$start" "$end"
done
```

Monitor all jobs with:

```bash
squeue -u yunsenz2
```

The primary accuracy analysis should use the strict completed intersection of
all three methods. The sensitivity analysis should retain all 174 cases and
count missing/failed outputs as failures. Runtime summaries should use all
observed attempts, including failures, with completed-only runtime secondary.
