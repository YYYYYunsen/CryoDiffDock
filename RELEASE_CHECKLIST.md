# Public release checklist

- [x] Separate Local and Global inference engines from paper figures/results.
- [x] Provide one command-line entry point for Local, Global, and Cascade.
- [x] Preserve frozen paper scripts and inference parameters.
- [x] Retain upstream licenses and attribution.
- [ ] Copy the exact frozen Local score and confidence checkpoints.
- [ ] Copy the exact frozen Global score, backbone, and confidence checkpoints.
- [ ] Record SHA-256 values and permanent GitHub Release or Zenodo URLs in
      `checkpoints/manifest.json`.
- [x] Add a compact 6PEK/EMD-20327 smoke-test input bundle.
- [ ] Confirm and document the redistribution terms for the 6PEK/EMD-20327 bundle.
- [ ] Run Local, Global, and Cascade smoke tests from a fresh environment.
- [ ] Publish benchmark cohort manifests or permanent dataset instructions.
- [ ] Replace the provisional citation with the manuscript DOI/preprint URL.
- [ ] Create a clean Git repository and inspect `git status` before pushing.
- [ ] Tag the exact paper release, for example `v1.0.0`.
