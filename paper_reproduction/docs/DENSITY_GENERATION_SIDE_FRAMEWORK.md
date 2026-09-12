# Generation-side density framework

Scope: changing the *distribution of generated poses* using density, as opposed
to re-ranking a fixed candidate pool.  Ranking-side work (NCC ranking, physics
gate, union) is unaffected by anything here and remains the paper's safe core.

## 0. Current implementation state (audited)

| Capability | Status | Location |
|---|---|---|
| Differentiable Gaussian ligand-map NCC | implemented | `density_guidance.py:170` |
| Rigid (tr/rot) score projection | implemented | `density_guidance.py:300` |
| Ligand torsion projection | implemented, **default off** | `density_guidance.py:324`, flag `--density_guidance_torsion` |
| x0 / Tweedie evaluation | implemented, **default off** | `density_guidance.py:392`, flag `--density_guidance_x0` |
| Score composition (norm-matched) | implemented | `density_guidance.py:327` |
| t-dependent blur schedule | implemented, **default off** | `density_guidance.py` `ncc_sigma_noise_scale`, flag `--density_guidance_ncc_sigma_noise_scale` |
| Side-chain torsion guidance | **missing** | no `sc_tor` path in `density_guidance.py` |
| SMC particle resampling | implemented | `particle_resampling.py`, checkpoints `(0.8, 0.9)` |
| FFT orientation seeding | **missing** | hook point is `sampling.py:16` `randomize_position` |

The released score model runs with `flexible_sidechains: true`, `all_atoms: true`,
`flexdist: 3.5`.  Side-chain torsions are already being diffused in every run,
baseline included.

## 1. Prerequisites that gate every experiment below

### P1. Error bars (blocking)

Identically seeded runs were observed not to be particle-wise reproducible
across processes.  Until the run-to-run standard deviation of top-1 `<2 A` and
best-of-40 `<2 A` is known, no generation-side A/B is interpretable.

- Run baseline on the validation split with >= 5 seeds.
- Report sigma for both metrics.
- Every generation-side result below is reported as mean +/- sigma over >= 3 seeds.
- Any delta inside 1 sigma is written up as "no significant difference", not as
  a positive or negative finding.

This is GPU-bound and runs in parallel with all CPU-bound preparation work.

### P2. No-op determinism, settled within-process

Do not attempt cross-process bitwise matching.  Hash the particle tensors
before and after each `apply()` call inside a single run; a disabled guidance or
resampling path must leave the hash unchanged.  This is independent of GPU
non-determinism.

Separately, before attributing divergence to GPU atomics, rule out incomplete
seeding (torch / numpy / python `random` / DataLoader workers / RDKit ETKDG) and
batch-ordering differences.

### P3. Leakage tiers (blocking for A and B)

Any experiment that uses density to move protein atoms must be reported in two
tiers:

- **Tier O (oracle map)**: map simulated from or derived using the ground-truth
  structure.  Upper bound only.  Never a headline number.
- **Tier E (experimental map)**: the claimable result.

Side-chain guidance under Tier O leaks ground-truth protein conformation into
the ligand metric.  This must be stated explicitly wherever Tier O appears.

### P4. Acyclicity between difference map and side-chain guidance

If a residual (difference) map is used, the protein model that is subtracted
must not have been guided by the same map.  Fix the order:

1. subtract using un-guided baseline side chains, or backbone + C-beta only;
2. then guide / score.

Never subtract density-guided side chains and then score the ligand against that
residual.

## 1b. Synthetic probe result (2026-08-19) — reorders everything below

`evaluation/probe_synthetic_gradient_conditions.py` measures the *direction* of
the Gaussian-NCC gradient against a known correction, varying only what the
target map contains.  Direction is the right quantity because `compose_score`
norm-matches the magnitude to the learned score and throws it away.

30 trials, ~1.0 A translation error and 25 degree rotation error, mean cosine to
the true correction (fraction pointing the right way):

| target map | envelope | translation | rotation |
|---|---|---|---|
| clean ligand only | 1.5 A | +0.98 (1.00) | +0.96 (1.00) |
| ligand + protein, no mask | 1.5 A | +0.79 (0.97) | **+0.56 (0.93)** |
| ligand + protein, **soft mask 1.5-2.5 A (current default)** | 1.5 A | +0.82 (1.00) | **+0.55 (0.87)** |
| ligand + protein, hard mask r=3.5 A | 1.5 A | +0.63 (0.90) | **+0.05 (0.53)** |
| difference map, exact protein | 1.5 A | +0.98 (1.00) | **+0.96 (1.00)** |
| difference map, 30% of side chains displaced 1.5 A | 1.5 A | +0.92 (1.00) | **+0.89 (1.00)** |
| ligand + protein, no mask | 3.0 A | +0.87 (1.00) | +0.26 (0.67) |
| ligand + protein, soft mask (current default) | 3.0 A | +0.95 (1.00) | +0.59 (0.90) |

Masking is modelled the way `apply_protein_mask` actually applies it: the target
map is down-weighted voxel-wise, every voxel is retained, and the rendered
ligand model is never masked.

Four conclusions, all of which change the plan:

1. **On a clean map the analytic gradient is already almost perfect** — cosine
   ~0.96 in rotation, 100% correct sign.  Nothing is wrong with the projection,
   the objective, or the NCC formulation per se.
2. **Protein density in the crop is the dominant corruption.**  Rotation
   direction collapses from 0.96 to ~0.55 purely from the protein being present.
   Translation is barely affected (0.98 -> 0.79), which is why translation-only
   experiments looked healthy while nothing improved.
3. **The current soft 1.5-2.5 A mask does essentially nothing about it**
   (0.56 unmasked -> 0.55 soft-masked).  Ligand heavy atoms sit >=3.3 A from
   protein heavy atoms, so the mask removes little ligand density — but it also
   leaves the entire protein density *shoulder* beyond 2.5 A inside the
   correlation.  This is the most likely explanation for why every crop,
   protein-cutoff, mask-radius and segmentation variant tested so far showed no
   effect: none of them removes the protein shoulder, and only subtraction does.
4. **Do not switch to a hard mask.**  At r=3.5 A the rotation gradient becomes
   essentially random (0.05, 53% correct sign) because what survives is the
   solvent-facing lobe, which carries almost no orientational information.

The difference map recovers nearly all of the loss (0.89) even when 30% of side
chains are displaced by 1.5 A, which is much more robust to side-chain error
than section 3 assumed.

Caveats: Gaussian protein shells, uniform occupancy, no resolution model, 5%
Gaussian noise.  This predicts a direction and a rough effect size, not a
real-data gain.  Validate on real crops with the existing
`evaluate_density_gradient_informativeness.py`, sweeping `--ncc-sigma` and
comparing masked vs unmasked vs difference-map crops.

## 1c. Unmodelled density, and the subtraction implementation

`evaluation/probe_unmodelled_density.py` answers the obvious objection: in a
membrane-protein map the residual still contains lipid, detergent and
disordered protein that no model accounts for.  Rotation gradient cosine,
1.5 A envelope, 25 degree rotation error:

| residual map | rotation |
|---|---|
| nothing unmodelled | +0.964 |
| + unmodelled lipid 8 A away | +0.964 |
| + unmodelled lipid 5 A away | +0.964 |
| + unmodelled lipid touching the ligand (2.5 A) | +0.964 |
| + four unmodelled lipid chains alongside | +0.964 |
| + unmodelled lipid **shell surrounding** the ligand (worst case) | **+0.700** |
| (reference) soft-masked map, nothing subtracted | +0.586 |

**Position matters, quantity does not.**  A blob beside the ligand — even
touching it, even four of them — costs nothing, because it does not vary as the
ligand rotates and NCC is mean-centred and variance-normalised; it dilutes the
coefficient, not the gradient direction.  Only density that *surrounds* the
ligand produces a spurious rotational gradient, which is exactly what the
modelled protein does.

Consequences: unmodelled lipid belt, detergent and disordered loops are mostly
outside the pocket and are effectively harmless.  The case to watch is a
genuinely lipid-lined pocket, where the worst case still beats the status quo
(0.700 vs 0.586).  Segmentation of the residual does not help (0.961 vs 0.964),
which independently confirms the earlier segmentation negative rather than
overturning it.

### Implementation

`--protein-subtraction {none,all_modelled,backbone_cb}` on
`scripts/prepare_oracle_density_dataset.py`, default `none` (exact no-op).  It
requires `--no-protein-mask`, so masking and subtraction cannot be applied
silently together.

Design points that matter:

- **Everything modelled except the target is subtracted** — protein, nucleic
  acid, modelled lipid, cholesterol, ions, waters.  The previous choice to
  retain non-polymers as "hard negatives" is correct under masking and wrong
  under subtraction.
- **The target ligand is protected by construction.**  Any non-polymer residue
  whose centroid lies within `--subtraction-ligand-exclusion-radius` (6 A) of
  the crop centre is never subtracted, and the excluded residue names are
  recorded.  The bias is deliberate: leaving a nearby lipid in place costs
  almost nothing per the table above, whereas subtracting the target is fatal.
  Polymer residues near the centre line the pocket and are always subtracted.
- **The Gaussian width is fitted, not assumed.**  Each candidate sigma is
  rendered and the best-correlating one kept, which doubles as a per-map
  resolution proxy — the covariate that section 1 noted was never tracked.
- **Scale and offset are fitted outside a sphere around the crop centre**
  (`--subtraction-fit-exclusion-radius`, 8 A), so ligand density cannot bias the
  fit and then be subtracted away.
- `backbone_cb` subtracts only backbone and C-beta, for use when predicted
  side chains are not trusted.

Every fitted quantity (sigma, scale, offset, correlation, atom count, excluded
residues, fit voxel count) is written to the crop `.npz` and its sidecar JSON.

## 2. Track C — finish and settle the existing guidance (cheapest)

Almost entirely a matter of running what is already coded.

### C1. Blur schedule — DONE, but demoted

Implemented as a noise-matched envelope rather than an ad-hoc interpolation.
At noise level t the pose is known only to within `tr_sigma` in translation and
`rot_sigma * Rg` in rotation, so the density it is *expected* to explain is its
envelope convolved with that spread.  Gaussians convolve by adding variances:

    sigma_eff**2 = ncc_sigma_A**2 + (noise_scale * sqrt(tr_sigma**2 + (rot_sigma*Rg)**2))**2

capped by `ncc_sigma_max_A`.  `--density_guidance_ncc_sigma_noise_scale 0`
(the default) is an exact identity.  Only valid with `--density_guidance_objective
map_ncc`; the constructor rejects the combination with `atom_density`, which
reads the map by trilinear interpolation and has no envelope width.

**Do not run this arm before the map is cleaned.**  Per section 1b, widening the
envelope on a crop that still contains protein density degrades the rotation
gradient badly (0.56 -> 0.26).  It is retained as an ablation arm for use on a
difference map, where the probe shows it should be neutral-to-slightly-positive.

### C2. Run the ablation grid

Factors, on the validation split, >= 3 seeds each:

- `use_x0_estimate` in {off, on}
- `use_torsion_projection` in {off, on}
- `objective` in {`atom_density`, `map_ncc`}

Hold `ncc_sigma_noise_scale` at 0 for this grid.  Per section 1b it cannot help
on the current crops, and it would confound the two factors worth measuring.

Do not run the full cross product.  Run the four single-factor arms against the
current default first; promote only factors that clear 1 sigma.

### C3. Decision gate

- Any arm clears 1 sigma on top-1 `<2 A` -> guidance is alive; carry the winning
  configuration into A and D.
- No arm clears 1 sigma -> training-free guidance is closed, with a properly
  controlled negative result covering objective, evaluation point (x_t vs x0),
  degrees of freedom (rigid vs +torsion), and resolution matching (blur).

C3's negative outcome is a publishable result, not a dead end.  It is the
version of "we tried density guidance and it did not help" that survives review.

## 3. Track A — side-chain density guidance

Rationale: at typical map resolution the ligand is a weak, often partially
occupied blob, whereas side chains sit in high-occupancy protein density that
the map was solved for.  Signal-to-noise strongly favours guiding side chains.
The causal claim is indirect but each link is stronger than direct ligand
guidance: density fixes the pocket, the pocket constrains the ligand.

### A1. Prerequisite: an unmasked map

`apply_protein_mask` suppresses protein-proximal voxels, which removes exactly
the side-chain signal this track depends on.  Side-chain guidance requires the
unmasked crop.  This is the same design decision that costs the ligand NCC its
contact-region signal.

### A2. Mechanism

Project the Cartesian density gradient onto side-chain chi angles, reusing the
`project_torsion_scores` construction already written for ligand torsions
(`density_guidance.py:324`), applied to `complex_graph['flexResidues'].edge_idx`
and the `modify_sidechains` rotation convention.

Guide side chains only; leave the ligand score untouched in the first arm, so
that any ligand improvement is unambiguously mediated by the pocket.

### A3. Evaluation

Primary (already built):
- `evaluate_sidechain_accuracy.py` — did side chains actually get better?
- `evaluate_ligand_accuracy.py` — did the ligand follow?

Report both tiers from P3.  The interesting cell is Tier E: side-chain accuracy
up *and* ligand top-1 up.  Side-chain accuracy up with ligand flat is a real but
weaker result and should be reported as such rather than buried.

### A4. Risks

- Guided side chains may over-fit to map noise and close the pocket, degrading
  ligand placement.  Mitigate with a guidance weight ramp and a clash gate on
  the protein itself.
- Tier O will look much better than Tier E.  Expect this; do not report Tier O
  as the result.

## 4. Track D — FFT orientation seeding

The pocket is given, so translation is largely solved and **orientation is not**.
Current initialisation (`randomize_position`, `sampling.py:16`) draws rotation
uniformly from SO(3) and torsions uniformly.  Density is never consulted at the
one point where it has the most leverage and the sampler has the most freedom.

### D1. Mechanism

1. Generate a conformer library for the ligand (RDKit ETKDG, ~20-50 conformers).
2. For each conformer, exhaustive rigid rotational search over the pocket crop
   by FFT cross-correlation against the map; keep top-N orientation modes.
3. Seed `randomize_position` from those modes instead of uniform SO(3),
   optionally with a small jitter, entering the reverse SDE at t slightly below 1.

CPU-bound, minutes per complex, no gradients, no scale matching, no training.

### D2. Controls

- **D-null**: uniform SO(3) seeding (current behaviour), same seed count.
- **D-shuffle**: seed from FFT modes computed against a *different complex's*
  map.  If D beats D-null but not D-shuffle, the gain came from conformer-library
  seeding, not from density.

D-shuffle is the control that makes this track defensible; do not skip it.

### D3. Interaction with C

If C3 promoted a guidance configuration, run D on top of it.  Seeding and drift
act at opposite ends of the trajectory and should compose.

## 5. Track B — learned guidance

The only theoretically correct classifier guidance available here: the required
quantity is grad log p(rho | x_t) at *noisy* poses, which no analytic NCC can
supply.

### B0. Why this is not a repeat of the failed density-aware confidence model

The earlier confidence model lost to hand-tuned `rerank_ligand_by_density_ncc`,
most plausibly because (i) the training pool was small relative to model
capacity while the hand-tuned combiner had ~1-2 free parameters, and (ii) the
network could reach low loss using geometry and confidence features while
effectively ignoring the density channel.

Two structural differences here:

- **Training data is effectively unlimited.**  Samples are (complex, t, noise
  draw) triples produced by forward-noising ground-truth poses.  No inference is
  required to make them.  The sample-size bottleneck that favoured the
  hand-tuned combiner largely disappears.
- **There is no hand-tuned competitor in the t > 0 regime.**  The comparison is
  not against `rerank_by_density.py`; it is against having no signal at all.

Failure mode (ii) persists and must be designed against.

### B1. Kill test, before any full training run

Train only the pretext task: given (pose, map), is this the matching map?
Negatives are maps from other complexes.  Evaluate at t = 0.4.

- AUC not clearly above 0.5 -> stop.  The model cannot see density at the noise
  levels where guidance would operate, and nothing downstream can work.
- AUC clearly above 0.5 -> proceed to B2.

Cost is a fraction of the full training run and it is decisive.

### B2. Training

- Small model.  Deliberately under-capacity relative to the confidence model.
- Time-conditioned.
- Map-swap hard negatives retained in the main objective, not just the pretext
  task, to prevent the density channel from being ignored.
- Target: binary consistency of the pose with the map at noise level t, not
  RMSD regression.

Guidance term is the gradient of the model's logit with respect to
translation, rotation, and torsions.  No Tweedie correction, no scale matching,
no blur schedule needed — those are all approximations this replaces.

### B3. Dual use

The same trained artifact serves as a ranker at t = 0 and as guidance at t > 0.
The t = 0 use is directly comparable against hand-tuned NCC ranking, which
provides an independent check on whether the model learned anything real.

### B4. Fallback: learn only the scale

If B2 underperforms, retreat to learning a scalar lambda(t, local features)
while keeping the analytic Gaussian-NCC gradient *direction*.  Very low
capacity, low over-fitting risk, interpretable, and consistent with the prior
evidence that low-degree-of-freedom parameterisations win at this data scale.

## 6. Sequencing (revised after section 1b)

The probe moves the difference map from a Track-A prerequisite to the single
highest-priority change, because it is now the predicted cause of the guidance
failure rather than one candidate among several.

Parallel from day one:

- **GPU lane**: P1 error bars, then C2 ablation grid (blur held at 0).
- **CPU lane**: difference-map preparation, then re-run
  `evaluate_density_gradient_informativeness.py` on masked vs unmasked vs
  difference-map crops to confirm section 1b on real data.

Then, gated:

1. **Difference map** (days, CPU) — highest expected value, validates or kills
   the probe's prediction on real crops before any GPU time is spent on it.
2. **C** (days) — settles the existing x0 / torsion code into a result either
   way.  Re-run the winning arm on difference-map crops, now including the blur.
3. **D** (1-2 weeks) — cheap, no training, orthogonal to C, has a real control.
4. **A** (2-3 weeks) — the probe's side-chain-error robustness (0.89 with 30% of
   side chains 1.5 A off) makes P4's acyclicity concern much less severe than
   originally assumed.
5. **B** (3+ weeks) — gated on B1; B4 is the retreat.

Track A and Track D both benefit from C's outcome but do not depend on it.
Track B is the only one that cannot be completed on a short submission timeline
and should not be started until C and D have reported.

## 7. What is claimable in each outcome

- All tracks negative: "density is a strong coarse validator and re-ranking
  signal for cryo-EM ligand placement; it does not improve generation" —
  supported by controlled ablations over objective, evaluation point, degrees of
  freedom, resolution matching, seeding, and in-trajectory selection, with
  seed-level error bars.  This is a defensible paper.
- C or D positive: a training-free density-guided sampler with an ablation
  showing which component carries the gain.
- A positive: the stronger and more novel claim — density improves the *pocket*,
  and the ligand follows.
- B positive: learned density guidance, with the ranker as a built-in control.
