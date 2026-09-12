# CryoDiffDock-Local V2: density-conditioned hierarchical search

V2 is an independent search path for a controlled V1/V2 comparison. It does
not modify `run_discrete_density_search.py`, the frozen V1 selector, or any V1
artifact. Both methods start from the same `baseline40+guided40` pose union and
use no deposited-ligand coordinates during generation or selection.

## Method

V1 applies a fixed, one-move expansion to six confidence/diversity-selected
parents. V2 instead searches a bounded tree on the local pose space

\[
SE(3) \times \mathbb{T}^{K},
\]

where `SE(3)` contains ligand translation and rotation and
`T^K` contains the ligand's rotatable-bond angles.

For each active pose, V2 differentiates the mean Gaussian NCC over one or more
ligand-envelope scales. The Cartesian gradient is projected into:

- translation gradient;
- rigid-body rotational torque;
- individual torsion derivatives.

Line-search proposals are generated only along these density-conditioned
directions. Rotation and translation may be applied together, and accepted
poses remain in a bounded beam so later iterations can compose rigid and
torsional moves. A separate 23-member cube-rotation branch retains large-basin
coverage; it has an independent anchor penalty and a reserved beam quota.

The final shortlist is reranked using local NCC-basin stability. Slight rigid
perturbations measure whether a pose lies in a broad density-supported basin
rather than at an isolated NCC spike. The score contains no RMSD or reference
pose:

\[
S(x)=C(x)+\lambda Z_{NCC}(x)-\alpha d(x,x_0)
-\beta\log(1+\Delta clash)-\rho I_{coarse},
\]

followed by replacement of a configurable fraction of the central NCC term by
the neighborhood lower-quartile NCC. The first pass inherits root confidence;
the optional second pass recomputes confidence for every finalist and replaces
only that score term.

## Main implementation

- `cryodiffdock_pocket/adaptive_search.py`: batched multi-scale NCC, gradients,
  density-conditioned moves, coordinate deduplication, spatial clash index, and
  basin robustness.
- `scripts/run_adaptive_density_search.py`: complete V2 beam search and V1-
  compatible result materialization.
- `scripts/apply_v2_rescored_ranking.py`: final generated-pose confidence
  replacement.
- `scripts/evaluate_adaptive_density_search.sh`: V2 evaluation and paired V1/V2
  comparison.

## Run the V2 generation/search pass

Use the same manifest and `pose_scores.csv` that were supplied to V1:

```bash
python -u scripts/run_adaptive_density_search.py \
  --manifest "$MANIFEST" \
  --scores "$POSE_SCORES" \
  --output-dir "$V2_ROOT" \
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
  --materialize-expanded-limit 0 \
  --fail-on-error
```

`--materialize-expanded-limit 0` is required to measure the complete V2 oracle
ceiling. For tuning or runtime profiling, set a finite limit to reduce SDF I/O.

Run a smoke test first:

```bash
python -u scripts/run_adaptive_density_search.py \
  --manifest "$MANIFEST" \
  --scores "$POSE_SCORES" \
  --output-dir "$V2_SMOKE_ROOT" \
  --limit 1 \
  --device cuda \
  --search-depth 2 \
  --beam-width 16 \
  --materialize-expanded-limit 50 \
  --fail-on-error
```

## Optional generated-pose confidence rescoring

The complete V2 pool is compatible with the existing batched confidence
rescoring script. The permissive geometric thresholds below ensure that every
V2 finalist, including the large-orientation branch, is rescored:

```bash
python -u scripts/rescore_discrete_candidates.py \
  --search-root "$V2_ROOT" \
  --manifest "$V2_ROOT/evaluation_manifest.csv" \
  --output-dir "$V2_ROOT/confidence_rescoring" \
  --minimum-ncc-gain 0 \
  --maximum-anchor-rmsd inf \
  --maximum-centroid-shift 1.5 \
  --minimum-in-bounds-fraction 0.95 \
  --batch-size 32 \
  --fail-on-error

python scripts/apply_v2_rescored_ranking.py \
  --search-root "$V2_ROOT" \
  --rescored-confidence "$V2_ROOT/confidence_rescoring/rescored_candidates.csv"
```

The second command writes the selected poses to `$V2_ROOT/v2_rescored`.

## Evaluate and compare with V1

```bash
SEARCH_ROOT="$V2_ROOT" \
V1_ROOT="$V1_ROOT" \
REFERENCE_MANIFEST="$REFERENCE_MANIFEST" \
bash scripts/evaluate_adaptive_density_search.sh
```

`REFERENCE_MANIFEST` must point to the row-compatible reveal manifest containing
`reference_ligand` (for the locked local benchmark this is
`02_prepared/inference.csv`). The search itself continues to use
`inference_blind.csv`; the evaluation script joins references only after poses
have been fixed.

The script evaluates selected performance and the full proposal ceiling, then
writes paired comparisons:

- `paired_original_vs_v2`: gain over the common unexpanded starting pool;
- `paired_v1_vs_v2_selected`: operational Top-1 V1 versus V2;
- `paired_v1_vs_v2_oracle`: proposal-ceiling V1 versus V2.

The V1 and V2 manifests must contain the same complexes in the same row order.
Do not compare roots with different inference arms, seeds, failures, or density
preparation.

The full V2 comparison intentionally changes both proposal generation and the
density selector. To isolate proposal generation, also run V2 with
`--density-sigmas 1.5 --basin-weight 0` and the same density/anchor weights used
by V1. Report both the complete-pool oracle and a compute-matched oracle using
the same number of scored candidates per complex; otherwise a coverage gain
could be attributed only to a larger evaluation budget.

## Recommended paper ablations

Freeze all hyperparameters on validation before independent test evaluation.
At minimum report:

1. V1 fixed one-move expansion.
2. V2 gradient proposals without composed moves (`--search-depth 1`).
3. V2 composed beam without the coarse branch (`--no-coarse-orientations`).
4. V2 complete central-NCC selector (`--basin-weight 0`).
5. V2 complete robust selector.

For the formal V1 selected-pose comparison, V1 means the deterministic
discrete expansion pool followed by the validation-frozen S0 selector
(`11_metrics_reveal/frozen_s0/per_complex.csv`).  The raw
`expanded_density` condition is only a diagnostic selector and must not be
reported as frozen S0.  V1 proposal coverage is compared separately using
`11_metrics_reveal/expanded_pool/per_complex.csv`.
6. V2 plus recomputed candidate confidence.

For each arm report Top-1 `<2 A`, Top-1 `<5 A`, median Top-1 RMSD, full-pool
oracle `<2 A`, candidate count, peak GPU memory, and paired wall-clock runtime.
Generation-only runtime and SDF materialization time are separately recorded in
`summary.json`; the latter should not be conflated with search compute.
