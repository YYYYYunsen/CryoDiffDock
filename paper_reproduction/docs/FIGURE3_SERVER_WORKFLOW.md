# Figure 3 native-evolution workflow

This workflow builds the revised Local-evolution Figure 3 (panels A--E). Core
Original/V1/V2 results use the frozen 171-complex drug-like cohort. Component
ablations use a prespecified result-blind 60-complex subset. Candidate count
and runtime are descriptive audit fields, not matched constraints.

Definitions:

- Original DiffDock-Pocket: confidence Top-1 and best-of-40 from the original
  40-pose pool.
- V1: validation-frozen S0 Top-1 and complete V1 expanded-pool oracle.
- V2: frozen native V2 Top-1 and complete V2 expanded-pool oracle.
- References are attached only after every blind operational ranking is fixed.

## Required existing results

```text
artifacts/paper_test/oracle_local_frozen_v1/
artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1/
```

The workflow reuses both roots. It only searches missing V2 component
ablations below `$FIG3_WORK/ablations`.

The frozen ablation list is
`visualizations/figure3_local/data/figure3_ablation_cohort60.txt`. It was
selected without RMSD or success labels by proportional 3×3×3 stratification
over ligand heavy atoms, ligand rotatable bonds, and flexible side-chain count,
with deterministic hash tie-breaking (seed 20260904). The CSV/JSON audit files
beside the list record membership, stratum allocation, hashes, and covariate
balance.

## Run on one allocated A100

```bash
cd /Scr/yunsenz2/data/DiffDock-Pocket-EM

export PROJECT_ROOT="$PWD"
export V1_ROOT="$PWD/artifacts/paper_test/oracle_local_frozen_v1"
export V2_MAIN="$PWD/artifacts/paper_test/adaptive_local_v2_frozen_druglike171_v1"
export FIG3_WORK="$PWD/artifacts/paper_test/figure3_local_evolution_ablation60_v1"
export HARDWARE_ID="$(hostname)-A100"
export PROCESS_COUNT=1

mkdir -p "$FIG3_WORK"
set -o pipefail
bash scripts/run_figure3_evolution_experiments.sh 2>&1 \
  | tee "$FIG3_WORK/run_figure3.log"
```

No `WALL_CLOCK_BUDGET_SECONDS` or `CANDIDATE_BUDGET` is required.

## Tables produced

```text
$FIG3_WORK/figure3_data/
├── panel_a_method_progression.csv
├── panel_b_performance.csv
├── panel_b_per_complex.csv
├── panel_c_paired_changes.csv
├── panel_c_statistics.csv
├── panel_d_ablations.csv
├── panel_d_ablation_statistics.csv
├── panel_d_search_audit.csv
├── panel_e_oracle_gap.csv
├── figure3_per_complex_core.csv
├── figure3_experiment_per_complex.csv
└── figure3_evolution_data_audit.json
```

`panel_c_statistics.csv` contains paired Wilcoxon and exact McNemar tests for
Original→V1, V1→V2, and Original→V2. `panel_d_ablation_statistics.csv` contains
the corresponding paired tests for each ablation versus main V2, with Holm
correction.

## Figures produced

```text
$FIG3_WORK/figure3_plots/
├── figure3_panel_a.{pdf,svg,png}
├── figure3_panel_b.{pdf,svg,png}
├── figure3_panel_c.{pdf,svg,png}
├── figure3_panel_d.{pdf,svg,png}
├── figure3_panel_e.{pdf,svg,png}
└── figure3_abcde_combined.{pdf,svg,png}
```

The seven searched ablations are no density gradients, depth 1, no combined
rigid move, no torsion, no coarse orientations, no basin robustness, and no
clash penalty. Main soft-clash and hard relative-physics rows are derived from
the existing main V2 outputs.
