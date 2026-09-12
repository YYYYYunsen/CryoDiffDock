# Checkpoints

Large model files are intentionally excluded from Git. Before making a public
release, place the five frozen checkpoints at the paths listed in
`manifest.json`, calculate SHA-256 values, upload the files to a GitHub Release
or Zenodo, and replace every `TODO` source and any empty checksum.

Do not call the paper results reproducible until the Global score checkpoint,
its pretrained backbone, and the Global confidence checkpoint are available.

The automated installer accepts this directory layout:

```text
cryodiffdock-weights/
├── local_score/
│   ├── best_ema_inference_epoch_model.pt
│   └── model_parameters.yml
├── local_confidence/
│   ├── best_model.pt
│   └── model_parameters.yml
├── global_score/
│   ├── best_ema_model.pt
│   └── config.yml                 # model_parameters.yml is also accepted
├── global_pretrained/
│   ├── best_ema_inference_epoch_model.pt
│   └── model_parameters.yml
└── global_confidence/
    ├── best_model.pt
    └── model_parameters.yml
```

Install it with `./install.sh --weights /path/to/cryodiffdock-weights`.
Model configuration files are required because a `.pt` state dictionary alone
does not define the network architecture.
