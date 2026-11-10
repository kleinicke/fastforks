# Changelog

## 0.1.1

Fixes and faster kernels. No new models.

- `kernels` is faster on GPUs that run fp16 tensor-core math with fp32 accumulation at full rate (NVIDIA L4, A10,
  A10G, L40S, A100, RTX PRO 6000): the DPT kernels (new `dpt_fused` build) use larger tiles for the deep layers and
  a tanh GELU there. Depth Pro and UniK3D keep the exact GELU. GeForce cards run as before.
- Marigold on `kernels` runs GroupNorm and SiLU on the channels-last map, without layout copies. The MoGe-3 refiner
  runs in fp16 on CUDA. RT-MonSter++ and CFNet on `kernels` no longer fall into slow paths on datacenter cards.
- TensorRT: the FoundationStereo cost volumes export without per-disparity loops. `c-fast-foundationstereo` at
  994 x 1482 on the RTX 5060 Ti takes 156 ms instead of 449 ms; the output is within fp16 rounding of `portable`.
- The fp16 overflow guard of the attention kernels syncs with the host once per model call instead of once per
  launch, which shortens the first calls of every guarded model. It also checks the warm-up before a CUDA graph is
  captured, so a graph never records a variant that overflows.
- Fix: in models with several attention launchers (`vggt-1b`, `depth-pro`), a launcher could run out of checked
  launches after reruns and stay unchecked.
- Fix: `Model.compile()` on segmentation models compiled the resize into the class maximum, three times slower than
  eager (SegFormer). The label step now runs eagerly inside the compiled model.
- Fix on Apple MPS: `unproject` and the stereo cameras move points to the CPU before float64 maths, and `save()` /
  `save_scene()` apply the `max_depth` and mask filters on the points' device.
- SAM 3, DINOv3, DINOv3 ConvNeXt and VGGT-Omega are validated on Apple MPS in fp32 and fp16 (`portable`).
- `fastforks download` no longer stops at the first model or file it cannot fetch (a gated repo without access, a
  host that is down). It fetches everything else, lists the failures as `MISSING` lines and exits with code 1.
- Changed: the weight files of Depth Anything 3 and MoGe have unique names (`DA3-LARGE-1.1/model.safetensors`,
  `moge-3-vitl/model.pt`) instead of `model.safetensors` / `model.pt` for every variant. A local weights folder
  (`FASTFORKS_WEIGHTS_PATH`) can now hold all variants; files placed there by hand under the old names must be moved.
- Docs: speed surveys per device (Apple M1 Pro, NVIDIA L4, A10, A10G, L40S, A100, RTX PRO 6000) and accuracy
  against the original code for depth, stereo, segmentation, detection, promptable segmentation, features,
  classification and multi-view depth.

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
