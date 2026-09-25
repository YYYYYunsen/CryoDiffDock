# Frozen PAAHS workflow

The public Local command runs two 40-pose inference conditions, V2 Top-1
selection, then the manuscript's three-phase **lazy PAAHS** pipeline. It uses
no deposited ligand, native RMSD, or success label for pose selection. The
final SDF and paired receptor are in `paahs/final/cases/<CASE>/`.

For precomputed predictions, run:

```bash
cryodiffdock paahs \
  --baseline-manifest baseline_manifest.csv \
  --search-manifest search_manifest.csv \
  --scores pose_scores.csv \
  --expected-cases 173 \
  --workers 4 --device cuda \
  --output outputs/paahs
```

Use `--preflight-only` first to validate the cohort, blind input headers,
and behavior-defining frozen configurations without starting GPU work. The
runner is resumable at complete stage boundaries. Do not reuse an output
directory from a different cohort or parameterization.

`baseline_manifest.csv` must contain `case_order`, `complex_name`,
`baseline_pose_path`, `protein_path`, and `density_crop`. It contains one row
per case, ordered exactly like `search_manifest.csv`. The search manifest is
the Local inference manifest and supplies the protein and ligand-density crop
for each case. `pose_scores.csv` is the NCC reranker's **per-pose** output;
the paper protocol has 40 original plus 40 density-guided poses per case.
All paths should resolve to existing local files. Do not include reference,
native, RMSD, or outcome fields in the blind baseline manifest.

The frozen phase configs are the six JSON files in `configs/` named
`clash_aware_adaptive_hierarchical_search_a.json`,
`density_repair_validation62_v2_frozen.json`,
`density_repair_v4_frozen.json`, `caahs_a_sidechain_rescue_frozen.json`,
`caahs_c_frozen_joint050.json`, and
`caahs_c_internal_geometry_frozen_d035.json`. The pipeline validates their
behavior-defining fields. It includes density-preserving repair, conservative
integration, minimum-distance refinement, internal-geometry refinement, and
final blind sealing. `--workers` controls per-stage CPU workers; a CUDA device
is used by the search stage.

The original eight-run and Q-score research analyses are published in
`evaluation/` but are **not** automatically applied to a new Local result.
The manuscript's Figure 5 Run-1 prediction uses the first frozen run; any
cross-run Q-score selector requires eight independently generated, scored
PAAHS outputs and its locked rule. The 200-case external cohort is not bundled
in this repository.

The complete numerical benchmark cannot be regenerated without the original
cohort inputs and released checkpoints. Check `RELEASE_CHECKLIST.md` before
claiming exact paper reproducibility.
