# Public release checklist

- [x] Separate Local and Global inference engines from paper figures/results.
- [x] Provide one command-line entry point for Local, Global, and Cascade.
- [x] Preserve frozen paper scripts and inference parameters.
- [x] Add final lazy PAAHS Local pipeline, frozen configurations, and blind
      evaluation/selection code.
- [x] Integrate the paper's reference-free cross-branch Cascade selector into
      the public `cryodiffdock cascade` command.
- [ ] Run a numerical regression of the one-command Cascade workflow against
      the archived Figure 7 outputs.
- [ ] Add the exact signed/hashed 173-case Q-score `locked_rule.json` if the
      optional eight-run selector is to be advertised as reproducible.
- [x] Retain upstream licenses and attribution.
- [x] Stage the exact Local score and confidence checkpoints locally and verify
      their recorded SHA-256 values (large files remain excluded from Git).
- [ ] Copy the exact frozen Global score, backbone, and confidence checkpoints.
- [ ] Record SHA-256 values and permanent GitHub Release or Zenodo URLs in
      `checkpoints/manifest.json`.
- [x] Add a compact 6PEK/EMD-20327 smoke-test input bundle.
- [ ] Confirm and document the redistribution terms for the 6PEK/EMD-20327 bundle.
- [ ] Run Local, Global, and Cascade smoke tests from a fresh environment.
- [ ] Publish benchmark cohort manifests or permanent dataset instructions.
- [ ] Replace the provisional citation with the manuscript DOI/preprint URL.
- [x] Create a clean standalone Git repository and inspect `git status`.
- [ ] Tag the exact paper release, for example `v1.0.0`.
