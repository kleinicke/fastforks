# Changelog

## 0.1.0

First release.

- 174 models in eight tasks: stereo, depth, classification, segmentation, features, promptable segmentation,
  detection and multi-view depth.
- Two backends per model: `portable` (plain PyTorch, runs on CPU, any CUDA GPU and Apple MPS) and `kernels` (custom CUDA kernels
  for NVIDIA GPUs of compute capability 8.0 or newer; Metal kernels for some families on Apple silicon). `mode="auto"` picks kernels where they run and the recorded measurements prefer them.
- Typical speed-up of kernels over portable: median 1.9x across the models and sizes measured, a quarter at 2.5x or more, best 10x (small classifiers at batch 1). Speed and memory per model are in the
  [benchmarks](docs/benchmarks.md) and the [overview](docs/overview.md).
- Weights are downloaded from the authors' hosts at pinned revisions and checked by SHA-256; every model carries its
  code licence, weights licence and commercial-use status.
- `pip install fastforks` installs everything (every model, timm, torchvision, transformers, ONNX export).
  `pip install fastforks-core` is the lean install with every model whose code is permissive; `[nc]` adds the models
  with non-commercial or GPL code, `[libs]` the optional libraries. Non-commercial weights are downloaded only with
  `FASTFORKS_NONCOMMERCIAL=1`, in every install.
- ONNX and TensorRT export of the portable backend, stereo geometry (rectification, point clouds), command line
  (`list`, `info`, `bench`, `export`, `download`, `doctor`, `env`, `stereo run`).
