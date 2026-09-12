# Density-weighted particle resampling

This experiment lets ligand density influence reverse diffusion without
directly editing ligand coordinates. At late checkpoints, Gaussian ligand-map
NCC supplies weights for systematic resampling of an active particle pool.
Protected particles remain model-only.

This implementation is intentionally described as particle selection, not as
an exact Bayesian SMC sampler. It uses standardized NCC and an ESS target as a
controlled, derivative-free selection heuristic.

## Default protocol

- 40 particles
- first 25% (10 particles) protected
- active-pool checkpoints after 80% and 90% of reverse steps
- target active-pool ESS: 70%
- resampling skipped if calibrated ESS remains above 80%
- Gaussian NCC sigma: 1.5 A
- no coordinate correction and no manually added rejuvenation noise
- systematic-resampling RNG isolated from the reverse-diffusion RNG

After a particle is cloned, the normal stochastic DiffDock-Pocket reverse SDE
provides independent translation, rotation, torsion, and side-chain noise to
the descendant slots. For this reason particle resampling rejects `--ode` and
`--no_random`.

## Exact no-op control

Set `PARTICLE_MAX_BETA=0`. At checkpoints the selector returns before NCC or
any other GPU work, does not resample, does not consume the reverse-diffusion
RNG, and does not replace any graph object. With the same seed, ligand outputs
must match the baseline run up to the reproducibility of the upstream sampler.

```bash
INPUT_CSV=artifacts/oracle_local/prepared_contact_entity_val5/inference.csv \
OUT_ROOT=artifacts/oracle_local/particle_resampling_noop5 \
CONDITION=both \
PARTICLE_MAX_BETA=0 \
bash scripts/run_density_particle_resampling_ablation.sh

python evaluation/verify_particle_resampling_noop.py \
  --baseline artifacts/oracle_local/particle_resampling_noop5/baseline \
  --control artifacts/oracle_local/particle_resampling_noop5/resampled
```

## Active pilot

```bash
INPUT_CSV=artifacts/oracle_local/prepared_contact_entity_val5/inference.csv \
OUT_ROOT=artifacts/oracle_local/particle_resampling_val5 \
CONDITION=both \
bash scripts/run_density_particle_resampling_ablation.sh
```

Each resampled result directory contains
`density_particle_resampling_audit.json`. Aggregate those files with:

```bash
python evaluation/summarize_particle_resampling_audits.py \
  --results artifacts/oracle_local/particle_resampling_val5/resampled \
  --output-dir artifacts/oracle_local/particle_resampling_val5/audit_summary \
  --fail-on-error
```

The audit records checkpoint ESS, selected-parent count, maximum offspring,
initial ancestry of every final particle, unique surviving ancestry, and the
largest family fraction. `particle_ranking.json` also includes ancestry for
each confidence-ranked output pose.

## Evaluation gates

Use the existing ligand evaluator and both confidence and NCC ranking. Do not
accept the method based only on pose-level NCC improvement.

Primary gates:

1. The beta-zero pilot is exactly identical to baseline.
2. Best-of-40 `<2 A` has no net loss.
3. Top-1 `<2 A` improves after the predeclared ranking method.
4. The active pool does not collapse to one lineage. Report unique active
   ancestors and maximum active-family fraction.
5. Confirm any chosen configuration on a held-out split before test reporting.
