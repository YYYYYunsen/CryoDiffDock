# Checkpoints

The five checkpoint files and YAML configurations are available in the
[model weights release](https://github.com/YYYYYunsen/CryoDiffDock/releases/tag/weights-v1.0.0).
Run `./install.sh` to download the pinned archive, verify its SHA-256 checksum,
and install the files.

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

Install the ZIP with `./install.sh --weights /path/to/cryodiffdock-weights-v1.0.0.zip`
or pass the extracted directory to `--weights`. For automatic download, set
`bundle.source` in `manifest.json` to the direct URL of the published ZIP.
Model configuration files are required because a `.pt` state dictionary alone
does not define the network architecture.
