# Checkpoints

Large model files are intentionally excluded from Git. Before making a public
release, place the five frozen checkpoints at the paths listed in
`manifest.json`, calculate SHA-256 values, upload the files to a GitHub Release
or Zenodo, and replace every `TODO` source and any empty checksum.

Do not call the paper results reproducible until the Global score checkpoint,
its pretrained backbone, and the Global confidence checkpoint are available.
