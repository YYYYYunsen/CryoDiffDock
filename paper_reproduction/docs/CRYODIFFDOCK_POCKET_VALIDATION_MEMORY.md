# CryoDiffDock-Pocket validation memory

Last updated: 2026-08-21

This file is the quantitative memory for the local-docking branch.  It records
what was actually run, the cohort used, the strongest positive and negative
results, and which conclusions are safe to carry into the paper.  Design ideas
that have not yet produced results belong in `CRYODIFFDOCK_POCKET_PLAN.md` or
`DENSITY_GENERATION_SIDE_FRAMEWORK.md`, not here.

## 1. Benchmark identity and non-negotiable caveats

- The source validation manifest contains 100 complexes.
- `7Z0T` failed ligand preparation and has no valid result.  Every paired
  comparison therefore uses the same 99-complex cohort and excludes `7Z0T`.
- The result directories were reindexed after removing `7Z0T`.  Evaluating them
  against the original 100-row manifest shifts every case after index 25 and
  can produce atom-count assertion failures.  The aligned evaluation manifest
  is:
  `artifacts/oracle_local/parent_relative_gnina_attempt1_v1/evaluation_manifest99.csv`.
- `8E94` can be parsed by pandas as scientific notation (`8e+94`) when a
  one-complex CSV is read without a string dtype.  Identifier readers and the
  Attempt-2 selector canonicalize this value back to `8E94`.
- The frozen 20-complex selection holdout is:
  `artifacts/oracle_local/frozen_gnina_relative_holdout20.txt`.
  SHA-256:
  `a121cd4ab2f42f5f4cd8b7703286dc07fba7f85d73c6e959737feb4c0cac7655`.
- This is **oracle-pocket redocking into the deposited holo cryo-EM model**.
  The deposited ligand centroid defines the local crop/box.  It is a valid
  local-docking upper-bound benchmark, but it is not a claim that deployment
  knows the true ligand position.
- RMSD and deposited coordinates are used only by validation evaluators and
  hyperparameter selection.  Candidate generation, NCC ranking, GNINA scoring,
  and physical gates do not receive RMSD.

## 2. Raw 99-complex DiffDock-Pocket results

Forty candidates were generated per complex for both the native baseline and
the original density-mean-guided condition.

| Metric | Baseline | Density-mean guided |
|---|---:|---:|
| Median confidence top-1 RMSD | 2.764 A | 2.742 A |
| Confidence top-1 `<2 A` | 37.37% | 31.31% |
| Confidence top-1 `<5 A` | 73.74% | 71.72% |
| Median best-of-40 RMSD | 1.859 A | 1.792 A |
| Best-of-40 `<2 A` | 52.53% | 54.55% |
| Best-of-40 `<5 A` | 92.93% | 91.92% |

Interpretation:

- Density-mean guidance does not improve native confidence top-1 selection.
- It changes the generated distribution slightly and raises the `<2 A`
  candidate ceiling by two complexes, but the paired success difference is
  not significant: baseline 52 successes, guided 54, gained/lost 4/2,
  exact McNemar `p=0.6875`, bootstrap 95% CI for the success-rate delta
  `[-3.03, 7.07]` percentage points.
- The condition is therefore retained as a complementary proposal source and
  generation ablation, not as a stand-alone accuracy improvement.

## 3. What density-mean guidance reliably changes

Gaussian ligand-map NCC was computed for all 7,920 baseline/guided poses using
`sigma=1.5 A`.

| Pool statistic | Baseline median | Guided median | Median paired delta | Guided better | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| Top-confidence pose NCC | 0.57786 | 0.58570 | +0.00865 | 75.76% | 2.56e-5 |
| Median candidate NCC | 0.54919 | 0.56847 | +0.01185 | 93.94% | 8.23e-17 |
| Maximum candidate NCC | 0.62162 | 0.63465 | +0.00796 | 84.85% | 1.79e-11 |

This is the defensible positive claim for the old generation guidance:
it shifts the generated pool toward better map agreement, even though native
confidence does not turn that shift into better top-1 RMSD.

NCC remains informative inside the local pocket, not only for gross pocket
localization.  Per-complex NCC/RMSD Spearman medians were:

| Pool | All poses | RMSD `<5 A` | RMSD `<3 A` |
|---|---:|---:|---:|
| Baseline | 0.552 | 0.677 | 0.685 |
| Guided | 0.562 | 0.687 | 0.617 |
| Union | 0.559 | 0.694 | 0.686 |

For the union pool restricted to `<3 A`, 95.7% of complexes had positive
Spearman correlation.  Density is therefore a useful discrete selector in the
near-native regime even though its continuous gradient is a weak generator.

## 4. Gaussian-NCC ranking and physical-gate result

Using the baseline+guided union and the validation-selected
`confidence + 2 * NCC(sigma=1.5 A)` ranking:

| Metric | Result |
|---|---:|
| Median top-1 RMSD | 2.022 A |
| Top-1 `<2 A` | 49.49% |
| Top-1 `<5 A` | 77.78% |
| Union oracle median RMSD | 1.769 A |
| Union oracle `<2 A` | 56.57% |

This raises top-1 `<2 A` from the raw baseline's 37.37% to 49.49% without
changing coordinates.

The reference-calibrated hard physical gate is useful for chemical auditing
but not as a primary ranker in its present form:

- 96/100 deposited references passed the joint clash thresholds.
- On the 99-case union pool, 45.54% of candidates passed; 6.06% of complexes
  required fallback.
- The gate reduced selected severe clashes to a median of zero, but top-1
  `<2 A` fell from 49.49% to 45.45% and top-1 `<5 A` from 77.78% to 75.76%.
- MMFF strain is retained as an audit field, not a hard deployment threshold.

Therefore physics enters the main method as a conservative rejection/safety
gate, never as an unrestricted replacement ranker.

## 5. Density preparation and local-map ablations

- Fixing the global ring-feature mask from cropped protein to full polymer was
  essential for the global model: target-ligand peak attribution rose from
  10.52% to 37.40%, target-ligand complex hits from 67.5% to 89.2%, and
  protein-polymer peak attribution fell from 72.65% to 21.10%.
- For the local ligand NCC crop, `full_polymer + soft 1.5--2.5 A` was
  numerically equivalent to the legacy soft crop: both median deposited-pose
  NCC values were 0.69014; median paired delta was zero (`p=0.632`).
- A hard 3.5 A protein mask was harmful: median deposited-pose NCC fell to
  0.48640 (`p=4.27e-18` versus full-polymer soft masking).
- Pocket-centred connected-component segmentation retained 49.5% top-1
  `<2 A` for thresholds z=0.25, 0.50, and 0.75.  It slightly increased native
  NCC and `<5 A` at the higher thresholds, but did not improve the primary
  `<2 A` endpoint.
- A fitted protein-subtraction/difference-map smoke test on five cases produced
  weaker real-map NCC gradients than the soft crop.  This does not invalidate
  subtraction in general, but it did not justify replacing the current local
  crop.

The frozen local-density representation is therefore the soft local crop with
Gaussian NCC; hard masking and connected-component segmentation are negative
ablations.

## 6. Continuous guidance and SMC conclusion

- Gradient probes showed NCC gradients are directionally restorative for
  moderate translation and rotation perturbations, but the single-step RMSD
  changes are small.
- Multi-seed generation tests of atom-density at `x_t`, NCC at `x_t`, NCC at
  an `x0` estimate, and NCC `x0` plus ligand torsion projection did not produce
  a meaningful top-1 improvement over baseline.
- Direct post-model coordinate injection destabilized subsequent diffusion
  steps despite small nominal translations.
- SMC/particle-resampling experiments were not interpretable across separate
  processes because nominal beta-zero reruns were not particle-wise identical.
  Within the resampling code path, beta-zero audits did confirm the intended
  exact no-op branch.  SMC is not part of the current main method.

The main paper should distinguish clearly between a useful discrete density
likelihood and an ineffective continuous local force.  Density is retained for
candidate expansion and selection, not claimed to improve the pretrained score
model itself.

## 7. Side-chain and rigid/flexible conclusions

- Input protein and reference protein are the same deposited holo structure,
  so side-chain improvement is mathematically not evaluable in this benchmark.
- Starting from the correct deposited side chains, flexible sampling yielded a
  median top-1 side-chain RMSD of about 1.36 A and a best-of-40 median of about
  1.04 A.  Only 42% of complexes had any candidate below 1 A, and the median
  number of joint ligand `<2 A` / side-chain `<1 A` candidates was zero.
- The manuscript must not claim flexible side-chain improvement from this
  oracle-input dataset.  Apo/AlphaFold input would be a separate project.
- Rigid and flexible ligand docking were statistically indistinguishable on
  val50: median top-1 RMSD 2.411 A rigid versus 2.395 A flexible,
  rigid better in 56%, Wilcoxon `p=0.323`; `<2 A` was 42% rigid versus 46%
  flexible.

Flexible output can remain a proposal mechanism, but side-chain accuracy is
not a headline endpoint for the present paper.

## 8. Discrete density expansion: strongest positive result

The successful generation-side intervention is deterministic local expansion
of existing poses by translation, rotation, torsion, and coarse orientation,
followed by density selection.  Unlike per-step guidance, this searches
discrete basins without destabilizing the diffusion trajectory.

Full 99-complex expanded pool:

- 7,920 original baseline/guided poses plus 4,132 generated variants = 12,052
  evaluated candidates.
- Expanded-pool oracle median RMSD: 1.424 A.
- Expanded-pool oracle `<2 A`: 70.71%.
- Expanded-pool oracle `<5 A`: 98.99%.
- No complex lost a previously available `<2 A` candidate when the expansion
  pool was added.
- Oracle winners were distributed across all move types: original 24,
  coarse orientation 22, torsion 20, translation 19, rotation 14.

The full-pool oracle is a coverage ceiling, not a deployable top-1 result.

### S0 conservative selector

The frozen safe discrete selector (`discrete_s0_safe`) achieved:

| Metric | S0 |
|---|---:|
| Median top-1 RMSD | 1.850 A |
| Top-1 `<2 A` | 56.57% |
| Top-1 `<5 A` | 81.82% |

Relative to the original union NCC selector (49.49%), S0 gains seven `<2 A`
complexes and loses none.  Relative to raw DiffDock-Pocket confidence
(37.37%), this is a +19.20 percentage-point gain.

## 9. GNINA diagnostics

GNINA `--score_only` was run on 3,268 coarse/S0 candidates with the fixed
manifest receptor; all 3,268 succeeded.  Median runtime was 10.23 seconds per
candidate.  GNINA never modified coordinates.

Absolute GNINA scores cannot directly rank the coarse candidate pool:

| Signal | Median per-complex Spearman | Direct selected `<2 A` | Net `<2 A` versus S0 |
|---|---:|---:|---:|
| CNNscore | -0.022 | 44.44% | -12 |
| CNNaffinity | -0.020 | 50.51% | -6 |
| Affinity | -0.105 | 54.55% | -2 |

Relative GNINA changes are useful binary safety signals for whether a coarse
candidate improves its control:

| Relative signal | Candidate-improvement AUC |
|---|---:|
| Delta CNNscore | 0.756 |
| Delta CNNaffinity | 0.744 |
| Delta affinity improvement | 0.679 |

This justifies GNINA as an acceptance gate, not as a continuous ranking term.

## 10. Attempt 1: parent-relative continuous GNINA term

Attempt 1 added candidate-minus-parent CNNscore as a continuous term while
retaining candidate-minus-S0 as a safety gate.

Signal generalization was real at the candidate-classification level:

| Cohort | Delta-parent CNNscore AUC | Delta-parent CNNaffinity AUC | Median within-complex Spearman |
|---|---:|---:|---:|
| Tuning79 | 0.742 | 0.745 | 0.023 |
| Holdout20 | 0.707 | 0.730 | 0.056 |
| Full99 | 0.735 | 0.746 | 0.025 |

The near-zero within-complex Spearman is the decisive limitation: GNINA can
classify whether a local move is broadly plausible but cannot order all poses
inside a complex.

Best tuning-selected configuration:

```text
robust NCC               = neighbor_ncc_z_q25
parent CNNscore weight   = 0.5
minimum score advantage  = -1.0
minimum NCC gain         = 0.05
minimum support          = 0.4
minimum CNNscore vs S0   = 0.0
minimum CNNscore vs parent = -0.1
density weight           = 3.0 (inherited from S0)
```

Results versus S0:

| Cohort | S0 `<2 A` | Attempt 1 `<2 A` | Gained/lost | Selected coarse |
|---|---:|---:|---:|---:|
| Tuning79 | 58.23% | 59.49% | 1 / 0 | 3 |
| Holdout20 | 50.00% | 50.00% | 0 / 0 | 0 |
| Full99 | 56.57% | 57.58% | 1 / 0 | 3 |

Full99 median RMSD changed from 1.850 A to 1.842 A; paired Wilcoxon
`p=0.109`.  Attempt 1 did **not** pass the preregistered holdout gate because
it made no holdout selections.  It is retained as a mechanistic ablation, not
as independent evidence for a better selector.

The provisional highest observed validation top-1 is therefore 57.58%, a
+20.20 percentage-point increase over raw DiffDock-Pocket confidence, but the
extra +1.01 point over S0 is tuning-only and must not be overstated.

## 11. Attempt 2: parent-wise tournament (completed; No-Go)

Attempt 2 implements a parent-wise tournament:

1. variants must pass binary candidate-vs-parent CNNscore and CNNaffinity gates;
2. robust NCC, not GNINA, chooses one local winner per direct parent;
3. parent winners compete using the frozen confidence+density score;
4. the final winner must pass candidate-vs-S0 GNINA gates, otherwise S0 is
   returned exactly.

Implementation:

- `evaluation/sweep_parent_tournament_gnina_gate.py`
- `tests/test_parent_tournament_gnina_gate.py`
- 11 related local unit tests passed.

The sweep selected parameters on tuning79 only.  The fixed holdout was read
after configuration selection.  The predeclared Go criterion was:

```text
holdout20: net gained <2 A >= 1, lost <2 A = 0,
           catastrophic worsening = 0
full99:    top-1 <2 A >= 59/99 = 59.60%
```

### Parent-tournament diagnostic

| Within-parent density statistic | Parent groups | Density winner is oracle | Density winner improves parent | Median winner regret |
|---|---:|---:|---:|---:|
| Neighbour NCC q25 | 594 | 23.40% | 26.43% | 3.118 A |
| Neighbour NCC median | 594 | 22.73% | 25.59% | 3.220 A |

This is the decisive negative result.  Robust neighbourhood NCC is useful as a
global candidate likelihood, but forcing it to choose one variant inside each
direct-parent family usually selects neither the oracle variant nor an
improvement over the parent.

Best tuning-selected configuration:

```text
robust NCC                         = neighbor_ncc_z_q25
minimum score advantage            = -1.0
minimum NCC gain                   = 0.005
minimum support                    = 0.4
minimum parent CNNscore delta      = -0.2
minimum parent CNNaffinity delta   = -1.0
minimum CNNscore delta vs S0       = 0.0
minimum CNNaffinity delta vs S0    = -0.5
density weight                     = 3.0
minimum in-bounds fraction         = 0.95
maximum centroid shift             = 0.25 A
```

Results versus S0:

| Cohort | S0 `<2 A` | Attempt 2 `<2 A` | Gained/lost | Selected coarse |
|---|---:|---:|---:|---:|
| Tuning79 | 58.23% | 59.49% | 1 / 0 | 3 |
| Holdout20 | 50.00% | 50.00% | 0 / 0 | 0 |
| Full99 | 56.57% | 57.58% | 1 / 0 | 3 |

Attempt 2 was exactly identical to Attempt 1 on every reported 99-complex
metric: zero RMSD differences, zero success changes, and the same three coarse
selections.  The paired Wilcoxon value is undefined (`NaN`) because every
Attempt-2-minus-Attempt-1 difference is exactly zero.

Attempt 2 therefore failed both preregistered gates: the holdout net gain was
zero and full99 remained 57/99 (57.58%), below the required 59/99 (59.60%).
Local hand-designed selector development is now stopped.  Freeze the existing
57.58% conservative density-expansion/GNINA-gated validation configuration and
evaluate it once on the independent test set.  The extra one-complex gain over
S0 remains tuning-only until the independent test supports it.

## 12. Claims safe for the manuscript

Safe:

- DiffDock-Pocket is a fast proposal generator for local cryo-EM docking.
- Explicit Gaussian density likelihood substantially improves selection.
- Deterministic density-driven local expansion increases candidate coverage
  from a 56.57% union ceiling to a 70.71% expanded-pool ceiling.
- The deployable conservative selector improves top-1 from 37.37% raw
  confidence to at least 56.57%; 57.58% is the frozen validation incumbent
  taken forward to the independent test.
- GNINA/physics are useful as relative safety gates, not absolute rankers.
- Learned/continuous density guidance is not required for the positive result.

Not safe:

- claiming the 70.71% oracle as deployable performance;
- claiming flexible side-chain improvement on deposited-holo input;
- claiming Attempt 1 independently validated a gain over S0;
- claiming Attempt 2 improves Attempt 1 (their selected results are identical);
- claiming high-NCC, high-RMSD poses are true alternative conformations without
  null calibration or independent structural evidence;
- claiming the local proposal score model itself is density-conditioned.

## 13. Frozen independent-test protocol

Validation-time selector development stopped after Attempt 2.  Independent
test evaluation must use the existing test split in full, subject only to
pre-specified technical/data-availability exclusions that are reported for
every arm.  It must not use another random validation subset as a test set.

The locked validation inputs are:

- S0 configuration:
  `artifacts/oracle_local/discrete_density_ranking_sweep_s0_v1/best_config.json`
- final parent-relative GNINA configuration:
  `artifacts/oracle_local/parent_relative_gnina_attempt1_v1/best_config.json`
- validation holdout hash (provenance only):
  `a121cd4ab2f42f5f4cd8b7703286dc07fba7f85d73c6e959737feb4c0cac7655`

Test-only execution support added on 2026-08-21:

- `scripts/freeze_local_test_protocol.py` records the test-split, model,
  container, configuration, and implementation SHA-256 values in an
  immutable lock directory.
- `scripts/apply_frozen_discrete_ranking.py` applies S0 without loading an
  RMSD table.
- `scripts/apply_frozen_parent_relative_gnina_gate.py` applies the final gate
  without loading an RMSD table.
- `evaluation/summarize_locked_test_results.py` performs the single final
  reveal and reports Wilson confidence intervals, paired bootstrap intervals,
  exact McNemar tests, and paired Wilcoxon tests.

The test order is non-negotiable:

1. hash/freeze the existing test split, two validation-selected JSON files,
   score/confidence checkpoints, GNINA image, and implementation files;
2. prepare the oracle-pocket crop, then remove the reference-ligand column
   from the operational manifest;
3. generate baseline/guided candidates, compute NCC, expand candidates, apply
   S0, compute basin/GNINA scores, and apply the final gate without RMSD;
4. verify expected case counts and zero scoring failures without looking at
   reference accuracy;
5. run all RMSD evaluations and the locked statistical summary once.

Primary test comparison: raw DiffDock-Pocket confidence versus the frozen
final selector.  Predeclared secondary results are union NCC selection, S0,
the expanded-pool oracle ceiling, top-1 `<5 A`, median RMSD, centroid/centred/
rigid-aligned decomposition, and runtime.  The final selector must be reported
even if it loses on test; no thresholds may be changed after the reveal.
