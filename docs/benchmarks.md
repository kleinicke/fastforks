# Benchmarks

`kernels` (our CUDA kernels) against `portable` (plain PyTorch, each model's tested precision) and the original authors'
code. One RTX 5060 Ti 16 GB, torch 2.14.1+cu130, CUDA 12.8 kernels (13.0 for four stereo families), cuDNN 9.24. Stereo and depth rows use real images
(KITTI 2015 `000000_10`, Middlebury `Motorcycle`) as uint8 CUDA tensors with default options; other tasks use a synthetic
input at the task's default size. Speeds are medians of warm runs (5 for most rows; each row's protocol is in `docs/data/speed.jsonl`); peak memory is torch's peak allocated / reserved GPU MiB (CUDA, weights included) over the first and the timed calls. `fastforks bench MODEL --size HxW` measures your GPU (it uses 10 runs).

Sizes are written height x width everywhere (`375x1242`, as in `fastforks bench --size`).

To choose a model, use the [overview](overview.md): one standard size per task, generated from the same data. The tables
below give one table per task and scene (each model at the standard size, or its own input where it has no other;
the model pages list every measured size), the startup times and, at the end, the same models at fixed sizes up to 12 MP.

Accuracy matches the original code. On ETH3D and Middlebury (the "Measured" tables on the model pages) both modes
stay within 0.15 bad-% / 0.025 EPE of the original with the same checkpoint for AANet, AANet+, CoEx, IGEV,
Selective-IGEV, LiteAnyStereo V2 M/H, DEFOM, FoundationStereo-L and CREStereo. S2M2 is better (up to 1 bad-% lower)
because our port uses the cost-volume lookup convention the released weights were trained with.

The accuracy columns on the model pages check parity (original code vs portable vs kernels) on public images; they are not a
ranking. Variants trained or fine-tuned on ETH3D / Middlebury are marked `†` in those columns (`?` where the training data
is not stated); to rank models use the official Middlebury and ETH3D test-set leaderboards.

## Modes, states and devices

These terms apply to every device:

- Modes: `original` = the authors' code with the same checkpoint (fp32 and the authors' own fp16 option, else
  `torch.autocast`), `portable` = fastforks in plain PyTorch, `kernels` = fastforks with its own kernels (CUDA, or Metal on
  Apple silicon). Every time column names its dtype or its original precision.
- Timing: median of warm, synchronised calls after warm-up, one model at a time. Startup (load, first call) is
  listed separately.
- Memory is recorded as one of two kinds, never compared with each other: peak torch memory (allocated / reserved,
  CUDA and ROCm) or the peak process footprint (Apple silicon, where weights and activations share the system memory). A TensorRT engine allocates outside torch: its peak (allocated / reserved) is the engine's device
  memory plus torch's own peak (allocated / reserved) of the same run.
- Speed-ups name their baseline (`vs original`, `vs portable`, `vs fastest portable`). When a baseline is missing the
  cell stays empty; another baseline is not used in its place.
- States: `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised,
  `OOM` = out of memory, `–` = not tested.
- The reference device is the RTX 5060 Ti below: its rows are the regression floor (`tests/test_speed_floor.py`).
  Every other device is a survey with its own page; no floor is kept for it. Adding a device:
  "Devices" in `docs/data/README.md`.

<!-- devices:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed*) -->

## Other devices

- [Apple M1 Pro 16 GB](devices/apple-m1-pro-16gb.md) (mps): 573 models, 2241 rows, measured 2026-11-06 to 2026-11-18.
- [Apple M1 Pro 16 GB (MLX)](devices/apple-m1-pro-mlx-16gb.md) (mlx): 81 models, 264 rows, measured 2026-11-12 to 2026-11-18.
- [NVIDIA L4](devices/l4-22gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA L40S](devices/l40s-44gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA A10](devices/a10-22gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA A100-SXM4-40GB](devices/a100-sxm4-40gb.md) (cuda): 117 models, 234 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA RTX PRO 6000](devices/rtx-pro-6000-96gb.md) (cuda): 174 models, 348 rows, measured 2026-11-08 to 2026-11-08.

### Across devices

Each task at its standard input (the overview's). A cell is the fastest validated fastforks mode on that device (`portable` or `kernels`, time and dtype), or why there is none; a row needs a survey device measured at that input. Cells compare only rows of the same input and protocol (`warm-median`: median of warm, synchronised calls after warm-up (at least 5; 2 for calls over 2 s on surveys)); no speed-up is derived across devices, as the modes and dtypes differ. – = not measured at this input.

#### Stereo, 994x1482

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](models/stereo.md#aanet) | 88.9 ms · kernels fp16 | 818 ms · kernels fp16 | – | 111 ms · kernels fp16 | 46.0 ms · kernels fp16 | 97.6 ms · kernels fp16 | 50.8 ms · kernels fp16 | 22.1 ms · kernels fp16 |
| [`aanet-sceneflow`](models/stereo.md#aanet) | 101 ms · kernels fp16 | 631 ms · kernels fp16 | – | 140 ms · kernels fp16 | 44.7 ms · kernels fp16 | 105 ms · kernels fp16 | 49.6 ms · kernels fp16 | 20.9 ms · kernels fp16 |
| [`bridgedepth-rvc-pretrain`](models/stereo.md#bridgedepth) | 354 ms · kernels fp16 | 3756 ms · kernels fp16 | – | – | – | – | 143 ms · kernels fp16 | 68.8 ms · kernels fp16 |
| [`bridgedepth-sf`](models/stereo.md#bridgedepth) | 348 ms · kernels fp16 | 3693 ms · kernels fp16 | – | 430 ms · kernels fp16 | 122 ms · kernels fp16 | 333 ms · kernels fp16 | 141 ms · kernels fp16 | 68.8 ms · kernels fp16 |
| [`c-fast-foundationstereo`](models/stereo.md#c_foundationstereo) | 135 ms · kernels fp16 | 4881 ms · portable fp32 | – | – | – | – | 85.1 ms · kernels fp16 | 37.3 ms · kernels fp16 |
| [`c-foundationstereo-s`](models/stereo.md#c_foundationstereo) | – | OOM | – | 1908 ms · kernels fp16 | 551 ms · kernels fp16 | 1537 ms · kernels fp16 | 668 ms · kernels fp16 | 319 ms · kernels fp16 |
| [`cas-gwcnet-c-sceneflow`](models/stereo.md#cas_gwcnet) | 225 ms · kernels fp16 | 2489 ms · portable fp16 | – | – | – | – | 87.9 ms · kernels fp16 | 46.7 ms · kernels fp16 |
| [`cas-gwcnet-gc-sceneflow`](models/stereo.md#cas_gwcnet) | 358 ms · kernels fp16 | 2995 ms · portable fp16 | – | 500 ms · kernels fp16 | 159 ms · kernels fp16 | 345 ms · kernels fp16 | 136 ms · kernels fp16 | 78.8 ms · kernels fp16 |
| [`cfnet-sceneflow`](models/stereo.md#cfnet) | – | 2206 ms · kernels fp16 | – | 484 ms · kernels fp16 | 152 ms · kernels fp16 | 432 ms · kernels fp16 | 145 ms · kernels fp16 | 76.4 ms · kernels fp16 |
| [`coex-sceneflow`](models/stereo.md#coex) | – | 411 ms · portable fp16 | – | 32.4 ms · kernels fp16 | 11.9 ms · kernels fp16 | 27.3 ms · kernels fp16 | 16.3 ms · kernels fp16 | 6.43 ms · kernels fp16 |
| [`crestereo-eth3d`](models/stereo.md#crestereo) | 229 ms · kernels fp16 | 4127 ms · kernels fp16 | 3058 ms · kernels fp16 | 318 ms · kernels fp16 | 102 ms · kernels fp16 | 255 ms · kernels fp16 | 108 ms · kernels fp16 | 60.9 ms · kernels fp16 |
| [`defom-stereo-vitl-sceneflow`](models/stereo.md#defom_stereo) | 659 ms · kernels fp16 | 8224 ms · kernels fp16 | – | 921 ms · kernels fp16 | 271 ms · kernels fp16 | 689 ms · kernels fp16 | 248 ms · kernels fp16 | 146 ms · kernels fp16 |
| [`dlnr-middlebury`](models/stereo.md#dlnr) | 214 ms · kernels fp16 | 2143 ms · portable fp16 | – | – | – | – | – | 48.6 ms · kernels fp16 |
| [`dlnr-sceneflow`](models/stereo.md#dlnr) | 214 ms · kernels fp16 | 2143 ms · portable fp16 | – | 293 ms · kernels fp16 | 94.6 ms · kernels fp16 | 224 ms · kernels fp16 | 95.4 ms · kernels fp16 | 48.6 ms · kernels fp16 |
| [`fast-acvnet-generalization`](models/stereo.md#fast_acvnet) | – | 478 ms · portable fp16 | – | 51.7 ms · kernels fp16 | 19.2 ms · kernels fp16 | 43.6 ms · kernels fp16 | 21.5 ms · kernels fp16 | 9.33 ms · kernels fp16 |
| [`fast-acvnet-kitti12`](models/stereo.md#fast_acvnet) | – | 464 ms · portable fp16 | – | – | – | – | – | 9.58 ms · kernels fp16 |
| [`fast-acvnet-kitti15`](models/stereo.md#fast_acvnet) | – | 463 ms · portable fp16 | – | – | – | – | – | 9.34 ms · kernels fp16 |
| [`fast-acvnet-plus-generalization`](models/stereo.md#fast_acvnet) | – | 625 ms · portable fp16 | – | – | – | – | – | 11.7 ms · kernels fp16 |
| [`fast-acvnet-plus-kitti12`](models/stereo.md#fast_acvnet) | – | 624 ms · portable fp16 | – | – | – | – | – | 11.5 ms · kernels fp16 |
| [`fast-acvnet-plus-kitti15`](models/stereo.md#fast_acvnet) | – | 624 ms · portable fp16 | – | – | – | – | – | 11.6 ms · kernels fp16 |
| [`fast-acvnet-plus-sceneflow`](models/stereo.md#fast_acvnet) | – | 624 ms · portable fp16 | – | 62.8 ms · kernels fp16 | 20.3 ms · kernels fp16 | 51.2 ms · kernels fp16 | 28.7 ms · kernels fp16 | 11.8 ms · kernels fp16 |
| [`fast-acvnet-sceneflow`](models/stereo.md#fast_acvnet) | – | 463 ms · portable fp16 | – | – | – | – | – | 9.55 ms · kernels fp16 |
| [`fast-foundationstereo`](models/stereo.md#c_foundationstereo) | 135 ms · kernels fp16 | 5083 ms · portable fp16 | – | 201 ms · kernels fp16 | 70.2 ms · kernels fp16 | 170 ms · kernels fp16 | 85.3 ms · kernels fp16 | 36.8 ms · kernels fp16 |
| [`foundationstereo-vit-l`](models/stereo.md#foundationstereo) | 1513 ms · kernels fp16 | 24480 ms · kernels fp16 | – | 2035 ms · kernels fp16 | 605 ms · kernels fp16 | 1674 ms · kernels fp16 | 680 ms · kernels fp16 | 337 ms · kernels fp16 |
| [`foundationstereo-vit-s`](models/stereo.md#foundationstereo) | 1120 ms · kernels fp16 | 19496 ms · kernels fp16 | – | 1713 ms · kernels fp16 | 487 ms · kernels fp16 | 1369 ms · kernels fp16 | 561 ms · kernels fp16 | 270 ms · kernels fp16 |
| [`great-igev-eth3d`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9636 ms · kernels fp16 | – | – | – | – | – | 97.2 ms · kernels fp16 |
| [`great-igev-kitti`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9634 ms · kernels fp16 | – | – | – | – | – | 97.4 ms · kernels fp16 |
| [`great-igev-middlebury`](models/stereo.md#great_igev) | 579 ms · kernels fp16 | 25368 ms · kernels fp16 | – | – | – | – | – | 143 ms · kernels fp16 |
| [`great-igev-sceneflow`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9861 ms · kernels fp16 | – | 579 ms · kernels fp16 | 176 ms · kernels fp16 | 477 ms · kernels fp16 | 196 ms · kernels fp16 | 98.3 ms · kernels fp16 |
| [`gwcnet-g-sceneflow`](models/stereo.md#gwcnet) | 238 ms · kernels fp16 | 2378 ms · portable fp16 | – | – | – | – | – | 50.5 ms · kernels fp16 |
| [`gwcnet-gc-sceneflow`](models/stereo.md#gwcnet) | 257 ms · kernels fp16 | 2455 ms · portable fp16 | – | 342 ms · kernels fp16 | 109 ms · kernels fp16 | 244 ms · kernels fp16 | 97.7 ms · kernels fp16 | 56.7 ms · kernels fp16 |
| [`hitnet-eth3d`](models/stereo.md#hitnet) | – | 136 ms · kernels fp16 | – | 31.4 ms · kernels fp16 | 11.0 ms · kernels fp16 | 28.3 ms · kernels fp16 | 16.9 ms · kernels fp16 | 6.22 ms · kernels fp16 |
| [`hitnet-middlebury-d400`](models/stereo.md#hitnet) | – | 440 ms · kernels fp16 | – | – | – | – | – | 14.3 ms · kernels fp16 |
| [`igev-plusplus-eth3d`](models/stereo.md#igev_plusplus) | – | 6139 ms · kernels fp16 | – | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-kitti12`](models/stereo.md#igev_plusplus) | – | 6124 ms · kernels fp16 | – | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-kitti15`](models/stereo.md#igev_plusplus) | – | 6125 ms · kernels fp16 | – | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-middlebury`](models/stereo.md#igev_plusplus) | – | 6130 ms · kernels fp16 | – | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-sceneflow`](models/stereo.md#igev_plusplus) | – | 6135 ms · kernels fp16 | – | 547 ms · kernels fp16 | 177 ms · kernels fp16 | 463 ms · kernels fp16 | 186 ms · kernels fp16 | 104 ms · kernels fp16 |
| [`igev-stereo-eth3d`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 3998 ms · kernels fp16 | – | – | – | – | – | 72.6 ms · kernels fp16 |
| [`igev-stereo-kitti12`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4006 ms · kernels fp16 | – | – | – | – | – | 72.2 ms · kernels fp16 |
| [`igev-stereo-kitti15`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4008 ms · kernels fp16 | – | – | – | – | – | 72.3 ms · kernels fp16 |
| [`igev-stereo-middlebury`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4007 ms · kernels fp16 | – | – | – | – | – | 72.6 ms · kernels fp16 |
| [`igev-stereo-sceneflow`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4001 ms · kernels fp16 | – | 430 ms · kernels fp16 | 134 ms · kernels fp16 | 349 ms · kernels fp16 | 137 ms · kernels fp16 | 72.6 ms · kernels fp16 |
| [`las2-h`](models/stereo.md#las2) | – | 734 ms · kernels fp16 | – | 77.0 ms · kernels fp16 | 36.3 ms · kernels fp16 | 65.1 ms · kernels fp16 | 38.3 ms · kernels fp16 | 15.5 ms · kernels fp16 |
| [`las2-l`](models/stereo.md#las2) | 29.9 ms · kernels fp16 | 400 ms · portable fp16 | – | – | – | – | – | 8.96 ms · kernels fp16 |
| [`las2-m`](models/stereo.md#las2) | 18.3 ms · kernels fp16 | 237 ms · portable fp16 | – | – | – | – | – | 6.91 ms · kernels fp16 |
| [`las2-s`](models/stereo.md#las2) | – | 192 ms · portable fp16 | – | 23.0 ms · kernels fp16 | 15.3 ms · kernels fp16 | 16.9 ms · kernels fp16 | 15.7 ms · kernels fp16 | 4.99 ms · kernels fp16 |
| [`mocha-v1-sceneflow`](models/stereo.md#mocha_stereo) | 455 ms · kernels fp16 | 6693 ms · kernels fp16 | – | 631 ms · kernels fp16 | 224 ms · kernels fp16 | 527 ms · kernels fp16 | 227 ms · kernels fp16 | 116 ms · kernels fp16 |
| [`mocha-v2-sceneflow`](models/stereo.md#mocha_stereo) | 448 ms · kernels fp16 | 6501 ms · kernels fp16 | – | – | – | – | – | 115 ms · kernels fp16 |
| [`monster-mix-all`](models/stereo.md#monster) | – | 9646 ms · kernels fp16 | – | – | – | – | – | 164 ms · kernels fp16 |
| [`monster-plusplus-mix-all`](models/stereo.md#monster_plusplus) | 698 ms · kernels fp16 | 10202 ms · kernels fp16 | – | 949 ms · kernels fp16 | 288 ms · kernels fp16 | 792 ms · kernels fp16 | 289 ms · kernels fp16 | 167 ms · kernels fp16 |
| [`monster-sceneflow`](models/stereo.md#monster) | – | 9642 ms · kernels fp16 | – | 951 ms · kernels fp16 | 287 ms · kernels fp16 | 783 ms · kernels fp16 | 286 ms · kernels fp16 | 164 ms · kernels fp16 |
| [`nmrf-sceneflow`](models/stereo.md#nmrf) | 196 ms · kernels fp16 | 1204 ms · kernels fp16 | – | 239 ms · kernels fp16 | 64.6 ms · kernels fp16 | 173 ms · kernels fp16 | 84.4 ms · kernels fp16 | 37.7 ms · kernels fp16 |
| [`psmnet-sceneflow`](models/stereo.md#psmnet) | – | 2579 ms · portable fp16 | – | 290 ms · kernels fp16 | 90.6 ms · kernels fp16 | 218 ms · kernels fp16 | 91.3 ms · kernels fp16 | 48.9 ms · kernels fp16 |
| [`raft-stereo-eth3d`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3638 ms · kernels fp16 | – | – | – | – | – | 63.6 ms · kernels fp16 |
| [`raft-stereo-middlebury`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3634 ms · kernels fp16 | – | 416 ms · kernels fp16 | 121 ms · kernels fp16 | 313 ms · kernels fp16 | 110 ms · kernels fp16 | 63.6 ms · kernels fp16 |
| [`raft-stereo-sceneflow`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3637 ms · kernels fp16 | – | – | – | – | – | 63.6 ms · kernels fp16 |
| [`rt-monster-plusplus-mix-all`](models/stereo.md#rt_monster_plusplus) | – | 663 ms · portable fp32 | – | 125 ms · kernels fp16 | 48.3 ms · portable fp16 | 105 ms · kernels fp16 | 49.4 ms · portable fp16 | 23.6 ms · kernels fp16 |
| [`s2m2-l`](models/stereo.md#s2m2) | 376 ms · kernels fp16 | 6590 ms · portable fp16 | – | – | – | – | – | 91.7 ms · kernels fp16 |
| [`s2m2-m`](models/stereo.md#s2m2) | 248 ms · kernels fp16 | 3620 ms · portable fp16 | – | – | – | – | – | 61.1 ms · kernels fp16 |
| [`s2m2-s`](models/stereo.md#s2m2) | 113 ms · kernels fp16 | 1607 ms · portable fp16 | – | 163 ms · kernels fp16 | 56.8 ms · kernels fp16 | 137 ms · kernels fp16 | 62.4 ms · kernels fp16 | 29.0 ms · kernels fp16 |
| [`s2m2-xl`](models/stereo.md#s2m2) | 682 ms · kernels fp16 | 13398 ms · portable fp16 | – | 1152 ms · kernels fp16 | 310 ms · kernels fp16 | 822 ms · kernels fp16 | 312 ms · kernels fp16 | 168 ms · kernels fp16 |
| [`selective-igev-eth3d-finetune`](models/stereo.md#selective_igev) | – | 5708 ms · kernels fp16 | – | – | – | – | – | 89.3 ms · kernels fp16 |
| [`selective-igev-eth3d-train`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | – | – | – | – | – | 89.3 ms · kernels fp16 |
| [`selective-igev-kitti12`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5709 ms · kernels fp16 | – | – | – | – | – | 89.2 ms · kernels fp16 |
| [`selective-igev-kitti15`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | – | – | – | – | – | 89.1 ms · kernels fp16 |
| [`selective-igev-middlebury-finetune`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5778 ms · kernels fp16 | – | – | – | – | – | 89.4 ms · kernels fp16 |
| [`selective-igev-middlebury-train`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5738 ms · kernels fp16 | – | – | – | – | – | 89.2 ms · kernels fp16 |
| [`selective-igev-sceneflow`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | – | 515 ms · kernels fp16 | 163 ms · kernels fp16 | 410 ms · kernels fp16 | 168 ms · kernels fp16 | 89.2 ms · kernels fp16 |
| [`waft-stereo-b`](models/stereo.md#waft_stereo) | 393 ms · kernels fp16 | 11022 ms · portable fp32 | – | 574 ms · kernels fp16 | 146 ms · kernels fp16 | 422 ms · kernels fp16 | 173 ms · kernels fp16 | 85.7 ms · kernels fp16 |
| [`waft-stereo-s`](models/stereo.md#waft_stereo) | 284 ms · kernels fp16 | 3360 ms · portable fp32 | – | 358 ms · kernels fp16 | 100 ms · kernels fp16 | 283 ms · kernels fp16 | 125 ms · kernels fp16 | 60.1 ms · kernels fp16 |

#### Depth, 994x1482

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|---|
| [`any2full-b`](models/depth.md#any2full) | – | 268 ms · portable fp16 | – | – | – | – | – | – |
| [`any2full-l`](models/depth.md#any2full) | – | 684 ms · portable fp16 | – | – | – | – | – | – |
| [`any2full-s`](models/depth.md#any2full) | – | 118 ms · portable fp16 | – | – | – | – | – | – |
| [`da3-giant-1.1`](models/depth.md#depth_anything_3) | 44.2 ms · kernels fp16 | 759 ms · portable fp16 | – | 63.2 ms · kernels fp16 | 22.5 ms · kernels fp16 | 60.1 ms · kernels fp16 | 25.0 ms · kernels fp16 | 14.2 ms · kernels fp16 |
| [`da3-large-1.1`](models/depth.md#depth_anything_3) | 17.1 ms · kernels fp16 | 318 ms · portable fp16 | – | 22.4 ms · kernels fp16 | 8.49 ms · kernels fp16 | 21.1 ms · kernels fp16 | – | 6.59 ms · kernels fp16 |
| [`da3metric-large`](models/depth.md#depth_anything_3) | 14.7 ms · kernels fp16 | 280 ms · portable fp16 | – | 19.5 ms · kernels fp16 | 7.78 ms · kernels fp16 | 18.6 ms · kernels fp16 | – | 6.04 ms · kernels fp16 |
| [`da3mono-large`](models/depth.md#depth_anything_3) | 14.6 ms · kernels fp16 | 280 ms · portable fp16 | – | 19.2 ms · kernels fp16 | 7.66 ms · kernels fp16 | 18.5 ms · kernels fp16 | 12.8 ms · kernels fp16 | 5.81 ms · kernels fp16 |
| [`da3nested-giant-large-1.1`](models/depth.md#depth_anything_3) | 58.2 ms · kernels fp16 | 1184 ms · portable fp16 | – | 81.9 ms · kernels fp16 | 30.0 ms · kernels fp16 | 78.1 ms · kernels fp16 | – | 19.7 ms · kernels fp16 |
| [`depth-anything-v2-b`](models/depth.md#depth_anything_v2) | 10.5 ms · kernels fp16 | 199 ms · portable fp16 | 192 ms · kernels fp16 | 14.6 ms · kernels fp16 | 4.67 ms · kernels fp16 | 13.7 ms · kernels fp16 | – | 4.00 ms · kernels fp16 |
| [`depth-anything-v2-l`](models/depth.md#depth_anything_v2) | 31.1 ms · kernels fp16 | 579 ms · portable fp16 | 548 ms · kernels fp16 | 42.7 ms · kernels fp16 | 13.6 ms · kernels fp16 | 40.1 ms · kernels fp16 | 16.1 ms · kernels fp16 | 8.61 ms · kernels fp16 |
| [`depth-anything-v2-metric-hypersim-b`](models/depth.md#depth_anything_v2) | – | 199 ms · portable fp16 | 189 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-metric-hypersim-l`](models/depth.md#depth_anything_v2) | – | 579 ms · portable fp16 | 534 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-metric-hypersim-s`](models/depth.md#depth_anything_v2) | – | 79.5 ms · portable fp16 | 73.2 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-b`](models/depth.md#depth_anything_v2) | – | 199 ms · portable fp16 | 185 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-l`](models/depth.md#depth_anything_v2) | – | 580 ms · portable fp16 | 535 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-s`](models/depth.md#depth_anything_v2) | – | 79.5 ms · portable fp16 | 75.0 ms · kernels fp16 | – | – | – | – | – |
| [`depth-anything-v2-s`](models/depth.md#depth_anything_v2) | 4.51 ms · kernels fp16 | 80.7 ms · portable fp16 | 79.1 ms · kernels fp16 | 5.92 ms · kernels fp16 | 2.24 ms · kernels fp16 | 5.43 ms · kernels fp16 | 4.63 ms · kernels fp16 | 2.02 ms · kernels fp16 |
| [`depth-pro`](models/depth.md#depth_pro) | – | 5011 ms · portable fp16 | – | 472 ms · kernels fp16 | 133 ms · kernels fp16 | 359 ms · kernels fp16 | 119 ms · kernels fp16 | 68.9 ms · kernels fp16 |
| [`depthfm-v1`](models/depth.md#depthfm) | – | OOM | – | – | – | – | – | – |
| [`distill-any-depth-b`](models/depth.md#depth_anything_v2) | – | 199 ms · portable fp16 | 203 ms · kernels fp16 | – | – | – | – | – |
| [`distill-any-depth-l`](models/depth.md#depth_anything_v2) | – | 580 ms · portable fp16 | 568 ms · kernels fp16 | – | – | – | – | – |
| [`distill-any-depth-s`](models/depth.md#depth_anything_v2) | – | 79.3 ms · portable fp16 | 76.7 ms · kernels fp16 | – | – | – | – | – |
| [`lingbot-depth`](models/depth.md#lingbot_depth) | – | 3363 ms · portable fp32 | – | – | – | – | – | – |
| [`lingbot-depth-dc`](models/depth.md#lingbot_depth) | – | 3363 ms · portable fp32 | – | – | – | – | – | – |
| [`lotus-normal-d-v1-1`](models/depth.md#lotus) | – | 1459 ms · portable fp16 | – | – | – | – | – | – |
| [`lotus-normal-g-v1-1`](models/depth.md#lotus) | – | 1553 ms · portable fp16 | – | – | – | – | – | – |
| [`marigold-depth-v1-1`](models/depth.md#marigold) | – | 2498 ms · portable fp16 | – | 346 ms · kernels fp16 | 128 ms · kernels fp16 | 260 ms · kernels fp16 | 177 ms · portable fp16 | 74.8 ms · kernels fp16 |
| [`marigold-normals-lcm-v0-1`](models/depth.md#marigold_normals) | – | 1655 ms · portable fp16 | – | – | – | – | – | – |
| [`marigold-v2-depth`](models/depth.md#marigold_v2) | – | unsupported | – | – | – | – | – | – |
| [`metric3d-vit-large`](models/depth.md#metric3d) | – | 1822 ms · portable fp32 | – | 193 ms · kernels fp16 | 56.9 ms · kernels fp16 | 166 ms · kernels fp16 | 61.5 ms · kernels fp16 | 38.9 ms · kernels fp16 |
| [`midas-dpt-beit-large-512`](models/depth.md#midas) | 63.9 ms · kernels fp16 | 635 ms · portable fp16 | – | 91.5 ms · kernels fp16 | 26.8 ms · kernels fp16 | 73.1 ms · portable fp16 | 28.5 ms · kernels fp16 | 17.8 ms · portable fp16 |
| [`moge-2-vitb-normal`](models/depth.md#moge) | – | 585 ms · portable fp32 | – | – | – | – | – | – |
| [`moge-2-vitl-normal`](models/depth.md#moge) | 73.4 ms · kernels fp16 | 1343 ms · portable fp32 | – | 103 ms · kernels fp16 | 35.2 ms · kernels fp16 | 95.2 ms · kernels fp16 | – | 24.1 ms · kernels fp16 |
| [`moge-2-vits-normal`](models/depth.md#moge) | – | 361 ms · portable fp32 | – | 44.0 ms · kernels fp16 | 18.0 ms · kernels fp16 | 35.6 ms · kernels fp16 | 22.1 ms · kernels fp16 | 11.0 ms · kernels fp16 |
| [`moge-3-vitl`](models/depth.md#moge) | – | 2415 ms · kernels fp32 | – | 284 ms · kernels fp16 | 93.5 ms · kernels fp16 | 276 ms · kernels fp16 | 114 ms · kernels fp16 | 61.5 ms · kernels fp16 |
| [`omni-dc`](models/depth.md#omni_dc) | – | 2349 ms · portable fp32 | – | – | – | – | – | – |
| [`pointdit-b-512`](models/depth.md#pointdit) | – | 337 ms · portable fp16 | – | – | – | – | – | – |
| [`pointdit-l-512`](models/depth.md#pointdit) | – | 1013 ms · portable fp16 | – | – | – | – | – | – |
| [`stablenormal-turbo`](models/depth.md#stablenormal) | – | 1690 ms · portable fp16 | – | – | – | – | – | – |
| [`unidepth-v2-vitl14`](models/depth.md#unidepth_v2) | 66.4 ms · kernels fp16 | 1252 ms · portable fp32 | – | 85.8 ms · kernels fp16 | 25.4 ms · kernels fp16 | 77.0 ms · kernels fp16 | 26.6 ms · kernels fp16 | 18.4 ms · portable fp16 |
| [`unik3d-vitl`](models/depth.md#unik3d) | 62.9 ms · kernels fp16 | 1217 ms · portable fp32 | – | 88.9 ms · kernels fp16 | 27.7 ms · kernels fp16 | 78.4 ms · kernels fp16 | 29.8 ms · kernels fp16 | 19.1 ms · kernels fp16 |
| [`zoedepth-nk`](models/depth.md#zoedepth) | 88.9 ms · kernels fp16 | 778 ms · portable fp32 | – | 116 ms · kernels fp16 | 32.1 ms · kernels fp16 | 92.9 ms · kernels fp16 | 41.2 ms · kernels fp16 | 19.2 ms · kernels fp16 |

#### Classification, 224x224

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](models/classification.md#alexnet) | 0.660 ms · kernels fp16 | 3.04 ms · portable fp16 | – | 1.03 ms · kernels fp16 | 0.544 ms · kernels fp16 | 0.616 ms · kernels fp16 | 0.591 ms · kernels fp16 | 0.240 ms · kernels fp16 |
| [`coatnet_0_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 20.1 ms · portable fp32 | – | – | – | – | – | – |
| [`coatnet_1_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 32.4 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_2_rw_224.sw_in12k_ft_in1k`](models/classification.md#coatnet) | – | 42.4 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_nano_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 19.6 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`](models/classification.md#coatnet) | – | 34.8 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_rmlp_1_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 34.6 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`](models/classification.md#coatnet) | – | 45.7 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_rmlp_2_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 45.4 ms · portable fp16 | – | – | – | – | – | – |
| [`coatnet_rmlp_nano_rw_224.sw_in1k`](models/classification.md#coatnet) | – | 21.2 ms · portable fp32 | – | – | – | – | – | – |
| [`convnext_base.fb_in1k`](models/classification.md#convnext) | – | 24.8 ms · portable fp16 | 16.1 ms · portable fp16 | – | – | – | – | – |
| [`convnext_base.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 24.9 ms · portable fp16 | 16.2 ms · portable fp16 | – | – | – | – | – |
| [`convnext_base.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 45.2 ms · portable fp16 | 33.7 ms · portable fp16 | – | – | – | – | – |
| [`convnext_large.fb_in1k`](models/classification.md#convnext) | – | 38.0 ms · portable fp16 | 29.2 ms · portable fp16 | – | – | – | – | – |
| [`convnext_large.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 39.0 ms · portable fp16 | 29.1 ms · kernels fp16 | – | – | – | – | – |
| [`convnext_large.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 79.6 ms · portable fp16 | 63.8 ms · portable fp16 | – | – | – | – | – |
| [`convnext_small.fb_in1k`](models/classification.md#convnext) | – | 18.2 ms · portable fp16 | 12.3 ms · portable fp16 | – | – | – | – | – |
| [`convnext_small.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 18.7 ms · portable fp16 | 12.2 ms · portable fp16 | – | – | – | – | – |
| [`convnext_small.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 31.0 ms · portable fp16 | 25.0 ms · kernels fp16 | – | – | – | – | – |
| [`convnext_tiny.fb_in1k`](models/classification.md#convnext) | – | 10.2 ms · portable fp16 | 7.61 ms · portable fp16 | – | – | – | – | – |
| [`convnext_tiny.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 10.8 ms · portable fp16 | 7.57 ms · portable fp16 | – | – | – | – | – |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 18.1 ms · portable fp16 | 15.2 ms · kernels fp16 | – | – | – | – | – |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](models/classification.md#convnextv2) | 3.07 ms · kernels fp16 | 35.8 ms · portable fp16 | 20.0 ms · kernels fp16 | 3.68 ms · kernels fp16 | 2.64 ms · kernels fp16 | 3.79 ms · kernels fp16 | 5.03 ms · kernels fp16 | 2.83 ms · kernels fp16 |
| [`davit_base.msft_in1k`](models/classification.md#davit) | – | 31.8 ms · portable fp16 | – | – | – | – | – | – |
| [`davit_small.msft_in1k`](models/classification.md#davit) | – | 25.4 ms · portable fp32 | – | – | – | – | – | – |
| [`davit_tiny.msft_in1k`](models/classification.md#davit) | – | 14.2 ms · portable fp16 | – | – | – | – | – | – |
| [`deit_base_patch16_224.fb_in1k`](models/classification.md#deit) | 1.79 ms · kernels fp16 | 16.4 ms · portable fp16 | – | 1.90 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.77 ms · kernels fp16 | 2.21 ms · kernels fp16 | 1.02 ms · kernels fp16 |
| [`densenet121.tv_in1k`](models/classification.md#densenet) | – | 25.4 ms · portable fp32 | – | 1.80 ms · kernels fp16 | 1.40 ms · kernels fp16 | 1.91 ms · kernels fp16 | 2.69 ms · kernels fp16 | 1.31 ms · kernels fp16 |
| [`dinov2_vitl14_lc`](models/classification.md#dinov2_linear_cls) | 5.24 ms · kernels fp16 | 59.3 ms · portable fp16 | 63.0 ms · portable fp16 | 6.65 ms · kernels fp16 | 2.76 ms · kernels fp16 | 5.94 ms · portable fp16 | 4.71 ms · kernels fp16 | 2.22 ms · portable fp16 |
| [`efficientformer_l1.snap_dist_in1k`](models/classification.md#efficientformer) | – | 8.57 ms · portable fp32 | – | – | – | – | – | – |
| [`efficientformer_l3.snap_dist_in1k`](models/classification.md#efficientformer) | – | 15.5 ms · portable fp16 | – | – | – | – | – | – |
| [`efficientformer_l7.snap_dist_in1k`](models/classification.md#efficientformer) | – | 27.4 ms · portable fp16 | – | – | – | – | – | – |
| [`efficientformerv2_l.snap_dist_in1k`](models/classification.md#efficientformerv2) | – | 34.7 ms · portable fp32 | – | – | – | – | – | – |
| [`efficientformerv2_s0.snap_dist_in1k`](models/classification.md#efficientformerv2) | – | 13.9 ms · portable fp32 | – | – | – | – | – | – |
| [`efficientformerv2_s1.snap_dist_in1k`](models/classification.md#efficientformerv2) | – | 17.4 ms · portable fp32 | – | – | – | – | – | – |
| [`efficientformerv2_s2.snap_dist_in1k`](models/classification.md#efficientformerv2) | – | 25.4 ms · portable fp32 | – | – | – | – | – | – |
| [`efficientnet_b0.tv_in1k`](models/classification.md#efficientnet) | 0.870 ms · kernels fp16 | 13.0 ms · portable fp32 | 4.49 ms · kernels fp16 | 1.14 ms · kernels fp16 | 0.844 ms · kernels fp16 | 1.09 ms · kernels fp16 | 1.51 ms · kernels fp16 | 0.736 ms · kernels fp16 |
| [`efficientnet_b4.tv_in1k`](models/classification.md#efficientnet) | – | 38.3 ms · portable fp32 | 16.0 ms · kernels fp16 | 2.45 ms · kernels fp16 | 1.70 ms · kernels fp16 | 2.55 ms · kernels fp16 | 3.30 ms · kernels fp16 | 1.61 ms · kernels fp16 |
| [`efficientnet_v2_s.tv_in1k`](models/classification.md#efficientnetv2) | – | 34.0 ms · portable fp32 | 16.4 ms · kernels fp16 | 2.43 ms · kernels fp16 | 1.74 ms · kernels fp16 | 2.47 ms · kernels fp16 | 3.29 ms · kernels fp16 | 1.62 ms · kernels fp16 |
| [`efficientvit_b2.r288_in1k`](models/classification.md#efficientvit) | – | 18.6 ms · portable fp32 | – | 1.55 ms · kernels fp16 | 1.01 ms · kernels fp16 | 1.40 ms · kernels fp16 | 1.89 ms · kernels fp16 | 0.926 ms · kernels fp16 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](models/classification.md#eva02) | – | 235 ms · portable fp16 | – | 19.6 ms · kernels fp16 | 6.82 ms · kernels fp16 | 16.9 ms · kernels fp16 | 10.3 ms · kernels fp16 | 5.46 ms · kernels fp16 |
| [`falconsai-nsfw-vit`](models/classification.md#nsfw_vit) | – | 15.9 ms · portable fp16 | – | – | – | – | – | – |
| [`fastvit_sa24.apple_dist_in1k`](models/classification.md#fastvit) | – | – | – | 1.76 ms · kernels fp16 | 1.27 ms · kernels fp16 | 1.73 ms · kernels fp16 | 2.49 ms · kernels fp16 | 1.31 ms · kernels fp16 |
| [`focalnet_base_lrf.ms_in1k`](models/classification.md#focalnet) | – | 43.2 ms · portable fp16 | – | – | – | – | – | – |
| [`focalnet_base_srf.ms_in1k`](models/classification.md#focalnet) | – | 40.1 ms · portable fp16 | – | – | – | – | – | – |
| [`focalnet_small_lrf.ms_in1k`](models/classification.md#focalnet) | – | 33.7 ms · portable fp16 | – | – | – | – | – | – |
| [`focalnet_small_srf.ms_in1k`](models/classification.md#focalnet) | – | 31.1 ms · portable fp16 | – | – | – | – | – | – |
| [`focalnet_tiny_lrf.ms_in1k`](models/classification.md#focalnet) | – | 19.0 ms · portable fp16 | – | – | – | – | – | – |
| [`focalnet_tiny_srf.ms_in1k`](models/classification.md#focalnet) | – | 18.0 ms · portable fp32 | – | – | – | – | – | – |
| [`hgnetv2_b0.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 6.34 ms · portable fp32 | – | – | – | – | – | – |
| [`hgnetv2_b1.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 6.58 ms · portable fp16 | – | – | – | – | – | – |
| [`hgnetv2_b2.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 9.18 ms · portable fp16 | – | – | – | – | – | – |
| [`hgnetv2_b3.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 10.7 ms · portable fp16 | – | – | – | – | – | – |
| [`hgnetv2_b4.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 11.8 ms · portable fp16 | – | – | – | – | – | – |
| [`hgnetv2_b5.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 18.3 ms · portable fp16 | – | – | – | – | – | – |
| [`hgnetv2_b6.ssld_stage2_ft_in1k`](models/classification.md#hgnetv2) | – | 31.0 ms · portable fp16 | – | – | – | – | – | – |
| [`inception_v3.tv_in1k`](models/classification.md#inception_v3) | – | 17.4 ms · portable fp32 | – | 1.76 ms · kernels fp16 | 1.31 ms · kernels fp16 | 1.74 ms · kernels fp16 | 2.40 ms · kernels fp16 | 1.26 ms · kernels fp16 |
| [`mobilenet_v2.tv2_in1k`](models/classification.md#mobilenetv2) | 0.520 ms · kernels fp16 | 6.48 ms · portable fp32 | 2.72 ms · kernels fp16 | 0.695 ms · kernels fp16 | 0.506 ms · kernels fp16 | 0.580 ms · kernels fp16 | 0.768 ms · kernels fp16 | 0.346 ms · kernels fp16 |
| [`mobilenet_v3_large.tv2_in1k`](models/classification.md#mobilenetv3) | 0.630 ms · kernels fp16 | 8.54 ms · portable fp32 | 3.16 ms · kernels fp16 | 0.853 ms · kernels fp16 | 0.632 ms · kernels fp16 | 0.777 ms · kernels fp16 | 1.05 ms · kernels fp16 | 0.489 ms · kernels fp16 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](models/classification.md#mobilenetv1) | – | 4.29 ms · portable fp32 | 2.38 ms · portable fp16 | 0.593 ms · kernels fp16 | 0.413 ms · kernels fp16 | 0.448 ms · kernels fp16 | 0.574 ms · kernels fp16 | 0.253 ms · kernels fp16 |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](models/classification.md#mobilenetv4) | 0.750 ms · kernels fp16 | 9.58 ms · portable fp32 | – | 0.955 ms · kernels fp16 | 0.722 ms · kernels fp16 | 0.900 ms · kernels fp16 | 1.19 ms · kernels fp16 | 0.592 ms · kernels fp16 |
| [`regnety_160.tv2_in1k`](models/classification.md#regnet) | – | 33.1 ms · portable fp16 | 18.3 ms · kernels fp16 | 2.37 ms · kernels fp16 | 1.58 ms · kernels fp16 | 2.27 ms · kernels fp16 | 2.70 ms · kernels fp16 | 1.46 ms · kernels fp16 |
| [`repvit_m0_9.dist_300e_in1k`](models/classification.md#repvit) | – | 11.6 ms · portable fp32 | 5.62 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m0_9.dist_450e_in1k`](models/classification.md#repvit) | – | 11.5 ms · portable fp32 | 4.45 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_0.dist_300e_in1k`](models/classification.md#repvit) | – | 12.0 ms · portable fp32 | 5.41 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_0.dist_450e_in1k`](models/classification.md#repvit) | – | 12.0 ms · portable fp32 | 5.47 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_1.dist_300e_in1k`](models/classification.md#repvit) | – | 11.5 ms · portable fp32 | 4.57 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_1.dist_450e_in1k`](models/classification.md#repvit) | – | 11.6 ms · portable fp32 | 4.75 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_5.dist_300e_in1k`](models/classification.md#repvit) | – | 19.3 ms · portable fp32 | 7.08 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m1_5.dist_450e_in1k`](models/classification.md#repvit) | – | 19.1 ms · portable fp32 | 7.92 ms · portable fp32 | – | – | – | – | – |
| [`repvit_m2_3.dist_300e_in1k`](models/classification.md#repvit) | – | 27.3 ms · portable fp32 | 10.7 ms · portable fp16 | – | – | – | – | – |
| [`repvit_m2_3.dist_450e_in1k`](models/classification.md#repvit) | – | 27.2 ms · portable fp32 | 11.0 ms · portable fp16 | – | – | – | – | – |
| [`resnet152.tv2_in1k`](models/classification.md#resnet) | – | 24.3 ms · portable fp16 | 16.8 ms · kernels fp16 | 2.32 ms · kernels fp16 | 1.84 ms · kernels fp16 | 2.48 ms · kernels fp16 | 3.42 ms · kernels fp16 | 1.82 ms · kernels fp16 |
| [`resnet18.tv_in1k`](models/classification.md#resnet) | – | 4.67 ms · portable fp32 | 3.74 ms · portable fp16 | 0.718 ms · kernels fp16 | 0.519 ms · kernels fp16 | 0.609 ms · kernels fp16 | 0.797 ms · kernels fp16 | 0.370 ms · kernels fp16 |
| [`resnet50.tv2_in1k`](models/classification.md#resnet) | 0.990 ms · kernels fp16 | 10.2 ms · portable fp16 | 6.68 ms · portable fp16 | – | – | – | – | 0.666 ms · kernels fp16 |
| [`shufflenet_v2_x1_0.tv_in1k`](models/classification.md#shufflenetv2) | – | 6.33 ms · portable fp32 | – | 1.13 ms · kernels fp16 | 0.874 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.63 ms · kernels fp16 | 0.811 ms · kernels fp16 |
| [`shvit_s1.in1k`](models/classification.md#shvit) | – | 10.3 ms · portable fp32 | – | – | – | – | – | – |
| [`shvit_s2.in1k`](models/classification.md#shvit) | – | 10.9 ms · portable fp32 | – | – | – | – | – | – |
| [`shvit_s3.in1k`](models/classification.md#shvit) | – | 11.8 ms · portable fp32 | – | – | – | – | – | – |
| [`squeezenet1_1.tv_in1k`](models/classification.md#squeezenet) | – | 3.72 ms · portable fp16 | – | 0.590 ms · kernels fp16 | 0.437 ms · kernels fp16 | 0.461 ms · kernels fp16 | 0.592 ms · kernels fp16 | 0.269 ms · kernels fp16 |
| [`starnet_s1.in1k`](models/classification.md#starnet) | – | 8.33 ms · portable fp32 | 3.28 ms · portable fp16 | – | – | – | – | – |
| [`starnet_s2.in1k`](models/classification.md#starnet) | – | 7.21 ms · portable fp16 | 5.64 ms · portable fp16 | – | – | – | – | – |
| [`starnet_s3.in1k`](models/classification.md#starnet) | – | 9.55 ms · portable fp16 | 5.01 ms · portable fp32 | – | – | – | – | – |
| [`starnet_s4.in1k`](models/classification.md#starnet) | – | 12.7 ms · portable fp32 | 5.88 ms · portable fp16 | – | – | – | – | – |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](models/classification.md#swin) | 2.45 ms · kernels fp16 | 28.2 ms · portable fp32 | – | 2.92 ms · kernels fp16 | 2.02 ms · kernels fp16 | 2.80 ms · kernels fp16 | 3.79 ms · kernels fp16 | 1.91 ms · kernels fp16 |
| [`tiny_vit_11m_224.dist_in22k_ft_in1k`](models/classification.md#tinyvit) | – | 9.49 ms · portable fp32 | – | – | – | – | – | – |
| [`tiny_vit_21m_224.dist_in22k_ft_in1k`](models/classification.md#tinyvit) | – | 11.6 ms · portable fp32 | – | – | – | – | – | – |
| [`tiny_vit_5m_224.dist_in22k_ft_in1k`](models/classification.md#tinyvit) | – | 8.61 ms · portable fp32 | – | – | – | – | – | – |
| [`vgg16.tv_in1k`](models/classification.md#vgg) | 1.56 ms · kernels fp16 | 11.7 ms · portable fp16 | – | 2.10 ms · kernels fp16 | 1.03 ms · kernels fp16 | 1.47 ms · kernels fp16 | 1.15 ms · kernels fp16 | 0.622 ms · kernels fp16 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](models/classification.md#vit) | 1.78 ms · kernels fp16 | 16.9 ms · portable fp32 | – | 1.92 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.77 ms · kernels fp16 | 2.21 ms · kernels fp16 | 1.02 ms · kernels fp16 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](models/classification.md#siglip) | 12.9 ms · kernels fp16 | 197 ms · portable fp16 | – | 15.8 ms · kernels fp16 | 6.99 ms · kernels fp16 | 16.0 ms · kernels fp16 | 7.96 ms · portable fp16 | 4.80 ms · portable fp16 |

#### Segmentation, 1024x2048

| model | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](models/segmentation.md#mmseg) | 528 ms · portable fp16 | – | – | – | – | – | 6.03 ms · kernels fp16 |
| [`deeplabv3plus-r101-d8-ade20k`](models/segmentation.md#mmseg) | 430 ms · portable fp16 | – | 24.9 ms · kernels fp16 | 7.55 ms · kernels fp16 | 20.8 ms · kernels fp16 | 13.0 ms · kernels fp16 | 5.21 ms · kernels fp16 |
| [`dinov2-b-linear-ade20k`](models/segmentation.md#dinov2_linear) | 361 ms · portable fp16 | 929 ms · portable fp32 | – | – | – | – | 6.86 ms · kernels fp16 |
| [`dinov2-l-linear-ade20k`](models/segmentation.md#dinov2_linear) | 927 ms · portable fp16 | 1348 ms · portable fp16 | 81.8 ms · kernels fp16 | 22.8 ms · kernels fp16 | 67.1 ms · kernels fp16 | 22.3 ms · kernels fp16 | 13.8 ms · kernels fp16 |
| [`dinov2-s-linear-ade20k`](models/segmentation.md#dinov2_linear) | 195 ms · portable fp16 | 708 ms · portable fp32 | 16.6 ms · kernels fp16 | 6.60 ms · kernels fp16 | 13.1 ms · kernels fp16 | 7.64 ms · kernels fp16 | 3.90 ms · kernels fp16 |
| [`eomt-l-dinov2-ade20k`](models/segmentation.md#eomt) | 540 ms · portable fp16 | – | 47.3 ms · kernels fp16 | 13.9 ms · kernels fp16 | 38.3 ms · kernels fp16 | 15.0 ms · kernels fp16 | 8.51 ms · kernels fp16 |
| [`eomt-l-dinov3-ade20k`](models/segmentation.md#eomt) | 575 ms · portable fp16 | – | 54.3 ms · kernels fp16 | 15.4 ms · kernels fp16 | 44.6 ms · kernels fp16 | 17.8 ms · kernels fp16 | 9.54 ms · kernels fp16 |
| [`fcn-r101-d8-ade20k`](models/segmentation.md#mmseg) | 405 ms · portable fp16 | – | 21.8 ms · kernels fp16 | 6.92 ms · kernels fp16 | 19.5 ms · kernels fp16 | 11.8 ms · kernels fp16 | 5.21 ms · kernels fp16 |
| [`hrnet-w48-fcn-ade20k`](models/segmentation.md#mmseg) | 228 ms · portable fp16 | – | – | – | – | – | 4.64 ms · kernels fp16 |
| [`mask2former-swinl-ade20k`](models/segmentation.md#mask2former) | 909 ms · portable fp32 | – | 77.8 ms · kernels fp16 | 32.8 ms · kernels fp16 | 67.4 ms · kernels fp16 | 35.3 ms · kernels fp16 | 16.9 ms · kernels fp16 |
| [`oneformer-swinl-ade20k`](models/segmentation.md#oneformer) | 1127 ms · portable fp32 | – | 134 ms · kernels fp16 | 47.2 ms · kernels fp16 | 109 ms · kernels fp16 | 58.9 ms · kernels fp16 | 31.4 ms · kernels fp16 |
| [`pspnet-r101-d8-ade20k`](models/segmentation.md#mmseg) | 411 ms · portable fp16 | – | – | – | – | – | 10.3 ms · kernels fp16 |
| [`sapiens-seg-0.3b-goliath`](models/segmentation.md#sapiens) | 1517 ms · portable fp16 | – | 135 ms · kernels fp16 | 39.7 ms · kernels fp16 | 97.7 ms · kernels fp16 | 34.4 ms · kernels fp16 | 24.4 ms · portable fp16 |
| [`sapiens-seg-1b-goliath`](models/segmentation.md#sapiens) | 3113 ms · portable fp16 | – | 305 ms · kernels fp16 | 81.4 ms · kernels fp16 | 200 ms · kernels fp16 | 67.8 ms · kernels fp16 | 45.5 ms · portable fp16 |
| [`segformer-b0-ade20k`](models/segmentation.md#segformer) | 106 ms · portable fp32 | – | 3.91 ms · kernels fp16 | 1.76 ms · kernels fp16 | 3.79 ms · kernels fp16 | 3.43 ms · kernels fp16 | 1.24 ms · kernels fp16 |
| [`segformer-b5-ade20k`](models/segmentation.md#segformer) | 375 ms · portable fp16 | – | 34.5 ms · kernels fp16 | 9.82 ms · kernels fp16 | 28.5 ms · kernels fp16 | 18.3 ms · kernels fp16 | 8.22 ms · kernels fp16 |
| [`segnext-mscan-b-ade20k`](models/segmentation.md#segnext) | 184 ms · portable fp16 | – | – | – | – | – | – |
| [`segnext-mscan-l-ade20k`](models/segmentation.md#segnext) | 268 ms · portable fp16 | – | – | – | – | – | – |
| [`segnext-mscan-s-ade20k`](models/segmentation.md#segnext) | 134 ms · portable fp16 | – | – | – | – | – | – |
| [`segnext-mscan-t-ade20k`](models/segmentation.md#segnext) | 118 ms · portable fp16 | – | – | – | – | – | – |

#### Matting, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`ben2`](models/matting.md#ben2) | 1148 ms · portable fp32 |
| [`birefnet`](models/matting.md#birefnet) | 1152 ms · portable fp16 |
| [`birefnet-lite`](models/matting.md#birefnet) | 474 ms · portable fp32 |
| [`inspyrenet`](models/matting.md#inspyrenet) | 702 ms · portable fp16 |
| [`inspyrenet-fast`](models/matting.md#inspyrenet) | 145 ms · portable fp16 |
| [`isnet`](models/matting.md#isnet) | 129 ms · portable fp16 |
| [`isnet-general-use`](models/matting.md#isnet) | 130 ms · portable fp16 |
| [`matanyone`](models/matting.md#matanyone) | 5907 ms · portable fp16 |
| [`rmbg-2.0`](models/matting.md#birefnet) | 1008 ms · portable fp16 |
| [`u2net`](models/matting.md#u2net) | 79.4 ms · portable fp16 |
| [`u2net-human-seg`](models/matting.md#u2net) | 75.1 ms · portable fp16 |
| [`u2netp`](models/matting.md#u2net) | 50.2 ms · portable fp16 |
| [`vitmatte-b`](models/matting.md#vitmatte) | 2740 ms · portable fp16 |
| [`vitmatte-s`](models/matting.md#vitmatte) | 1360 ms · portable fp16 |

#### Promptable segmentation, 1024x1024

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`clipseg`](models/promptable_segmentation.md#clipseg) | – | 37.7 ms · portable fp16 | – | – | – | – | – |
| [`efficientvit-sam-l0`](models/promptable_segmentation.md#efficientvit_sam) | – | 50.6 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-l1`](models/promptable_segmentation.md#efficientvit_sam) | – | 75.2 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-l2`](models/promptable_segmentation.md#efficientvit_sam) | – | 91.2 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-xl0`](models/promptable_segmentation.md#efficientvit_sam) | – | 167 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-xl1`](models/promptable_segmentation.md#efficientvit_sam) | – | 259 ms · portable fp32 | – | – | – | – | – |
| [`insid3-vitl16`](models/promptable_segmentation.md#insid3) | – | 2510 ms · portable fp16 | – | – | – | – | – |
| [`mobile-sam`](models/promptable_segmentation.md#mobile_sam) | – | 69.9 ms · portable fp16 | – | – | – | – | – |
| [`sam-hq-vit-b`](models/promptable_segmentation.md#sam_hq) | – | 645 ms · portable fp32 | – | – | – | – | – |
| [`sam-hq-vit-h`](models/promptable_segmentation.md#sam_hq) | – | 2347 ms · portable fp16 | – | – | – | – | – |
| [`sam-hq-vit-l`](models/promptable_segmentation.md#sam_hq) | – | 1292 ms · portable fp16 | – | – | – | – | – |
| [`sam-vit-b`](models/promptable_segmentation.md#sam) | 37.0 ms · kernels fp16 | 600 ms · portable fp16 | 40.7 ms · kernels fp16 | 15.7 ms · kernels fp16 | 35.1 ms · kernels fp16 | 19.2 ms · kernels fp16 | 10.3 ms · kernels fp16 |
| [`sam-vit-h`](models/promptable_segmentation.md#sam) | 143 ms · kernels fp16 | 2333 ms · portable fp16 | 189 ms · kernels fp16 | 59.1 ms · kernels fp16 | 168 ms · kernels fp16 | 61.3 ms · kernels fp16 | 35.1 ms · kernels fp16 |
| [`sam-vit-l`](models/promptable_segmentation.md#sam) | 81.5 ms · kernels fp16 | 1279 ms · portable fp16 | 102 ms · kernels fp16 | 32.5 ms · kernels fp16 | 91.4 ms · kernels fp16 | 37.5 ms · kernels fp16 | 19.6 ms · kernels fp16 |
| [`sam2.1-hiera-base-plus`](models/promptable_segmentation.md#sam2) | – | 394 ms · portable fp16 | 31.7 ms · kernels fp16 | 10.3 ms · kernels fp16 | 29.1 ms · kernels fp16 | 15.8 ms · portable fp16 | 7.78 ms · portable fp16 |
| [`sam2.1-hiera-large`](models/promptable_segmentation.md#sam2) | – | 544 ms · portable fp16 | 65.8 ms · kernels fp16 | 21.4 ms · kernels fp16 | 59.3 ms · kernels fp16 | 27.1 ms · portable fp16 | 13.8 ms · portable fp16 |
| [`sam2.1-hiera-small`](models/promptable_segmentation.md#sam2) | 15.6 ms · kernels fp16 | 148 ms · portable fp16 | – | – | – | – | 5.43 ms · portable fp16 |
| [`sam2.1-hiera-tiny`](models/promptable_segmentation.md#sam2) | 13.9 ms · kernels fp16 | 129 ms · portable fp16 | 17.6 ms · kernels fp16 | 8.14 ms · kernels fp16 | 15.6 ms · kernels fp16 | 12.0 ms · portable fp16 | 5.11 ms · kernels fp16 |
| [`sam3`](models/promptable_segmentation.md#sam3) | – | 3620 ms · portable fp32 | 161 ms · kernels fp16 | 48.6 ms · kernels fp16 | 136 ms · kernels fp16 | 50.8 ms · kernels fp16 | 29.5 ms · kernels fp16 |

#### Detection, 800x1344

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`deim-l-coco`](models/detection.md#dfine) | – | 71.0 ms · portable fp16 | – | – | – | – | – |
| [`deim-m-coco`](models/detection.md#dfine) | – | 57.3 ms · portable fp16 | – | – | – | – | – |
| [`deim-n-coco`](models/detection.md#dfine) | – | 22.9 ms · portable fp16 | – | – | – | – | – |
| [`deim-s-coco`](models/detection.md#dfine) | – | 40.9 ms · portable fp16 | – | – | – | – | – |
| [`deim-x-coco`](models/detection.md#dfine) | – | 97.7 ms · portable fp16 | – | – | – | – | – |
| [`detr-r101-coco`](models/detection.md#detr) | 13.7 ms · kernels fp16 | 153 ms · portable fp16 | 12.8 ms · kernels fp16 | 9.91 ms · kernels fp16 | 11.6 ms · kernels fp16 | 10.6 ms · kernels fp16 | 3.52 ms · kernels fp16 |
| [`detr-r50-coco`](models/detection.md#detr) | 9.93 ms · kernels fp16 | 105 ms · portable fp16 | 11.8 ms · kernels fp16 | 8.21 ms · kernels fp16 | 8.64 ms · kernels fp16 | 8.82 ms · kernels fp16 | 3.14 ms · kernels fp16 |
| [`dfine-l-coco`](models/detection.md#dfine) | – | 71.0 ms · portable fp16 | 9.47 ms · kernels fp16 | 5.16 ms · kernels fp16 | 9.48 ms · kernels fp16 | 9.36 ms · kernels fp16 | 4.23 ms · kernels fp16 |
| [`dfine-m-coco`](models/detection.md#dfine) | – | 56.6 ms · portable fp16 | – | – | – | – | 3.30 ms · kernels fp16 |
| [`dfine-n-coco`](models/detection.md#dfine) | – | 25.2 ms · portable fp16 | 3.45 ms · kernels fp16 | 2.41 ms · kernels fp16 | 3.68 ms · kernels fp16 | 4.06 ms · kernels fp16 | 1.89 ms · kernels fp16 |
| [`dfine-s-coco`](models/detection.md#dfine) | – | 41.0 ms · portable fp16 | – | – | – | – | 2.37 ms · kernels fp16 |
| [`dfine-x-coco`](models/detection.md#dfine) | – | 101 ms · portable fp16 | 14.8 ms · kernels fp16 | 6.81 ms · kernels fp16 | 14.4 ms · kernels fp16 | 13.1 ms · kernels fp16 | 6.05 ms · kernels fp16 |
| [`faster-rcnn-r50-fpn-3x-coco`](models/detection.md#rcnn) | 19.5 ms · kernels fp16 | 192 ms · portable fp16 | 19.6 ms · kernels fp16 | 10.6 ms · kernels fp16 | 18.1 ms · kernels fp16 | 13.0 ms · kernels fp16 | 5.86 ms · kernels fp16 |
| [`grounding-dino-base`](models/detection.md#grounding_dino) | – | 1087 ms · portable fp32 | – | – | – | – | – |
| [`grounding-dino-tiny`](models/detection.md#grounding_dino) | – | 846 ms · portable fp32 | – | – | – | – | – |
| [`lw-detr-large-coco`](models/detection.md#lwdetr) | – | 72.7 ms · portable fp16 | – | – | – | – | – |
| [`lw-detr-medium-coco`](models/detection.md#lwdetr) | – | 47.2 ms · portable fp16 | – | – | – | – | – |
| [`lw-detr-small-coco`](models/detection.md#lwdetr) | – | 54.7 ms · portable fp16 | – | – | – | – | – |
| [`lw-detr-tiny-coco`](models/detection.md#lwdetr) | – | 41.2 ms · portable fp16 | – | – | – | – | – |
| [`lw-detr-xlarge-coco`](models/detection.md#lwdetr) | – | 130 ms · portable fp16 | – | – | – | – | – |
| [`mask-rcnn-r50-fpn-3x-coco`](models/detection.md#rcnn) | 21.0 ms · kernels fp16 | 203 ms · portable fp16 | 22.1 ms · kernels fp16 | 11.8 ms · kernels fp16 | 20.1 ms · kernels fp16 | 14.7 ms · kernels fp16 | 6.55 ms · kernels fp16 |
| [`mm-grounding-dino-base`](models/detection.md#grounding_dino) | – | 1273 ms · portable fp32 | – | – | – | – | – |
| [`mm-grounding-dino-tiny`](models/detection.md#grounding_dino) | – | 1104 ms · portable fp32 | – | – | – | – | – |
| [`owlv2-b16`](models/detection.md#owlv2) | – | 394 ms · portable fp32 | – | – | – | – | – |
| [`owlv2-l14`](models/detection.md#owlv2) | – | 2047 ms · portable fp32 | – | – | – | – | – |
| [`rf-detr-l-coco`](models/detection.md#rfdetr) | – | 75.5 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-m-coco`](models/detection.md#rfdetr) | – | 55.5 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-n-coco`](models/detection.md#rfdetr) | – | 30.3 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-s-coco`](models/detection.md#rfdetr) | – | 44.9 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-seg-2xl-coco`](models/detection.md#rfdetr) | – | 184 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-seg-l-coco`](models/detection.md#rfdetr) | – | 89.2 ms · portable fp32 | – | – | – | – | – |
| [`rf-detr-seg-m-coco`](models/detection.md#rfdetr) | – | 64.7 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-seg-n-coco`](models/detection.md#rfdetr) | – | 41.8 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-seg-s-coco`](models/detection.md#rfdetr) | – | 50.1 ms · portable fp16 | – | – | – | – | – |
| [`rf-detr-seg-xl-coco`](models/detection.md#rfdetr) | – | 124 ms · portable fp16 | – | – | – | – | – |
| [`rtdetrv2-l-coco`](models/detection.md#rtdetrv2) | – | 77.3 ms · portable fp16 | 7.20 ms · kernels fp16 | 3.82 ms · kernels fp16 | 7.63 ms · kernels fp16 | 6.65 ms · kernels fp16 | 3.31 ms · kernels fp16 |
| [`rtdetrv2-m-coco`](models/detection.md#rtdetrv2) | – | 55.0 ms · portable fp16 | – | – | – | – | 2.68 ms · kernels fp16 |
| [`rtdetrv2-ms-coco`](models/detection.md#rtdetrv2) | – | 62.7 ms · portable fp16 | – | – | – | – | 2.57 ms · kernels fp16 |
| [`rtdetrv2-s-coco`](models/detection.md#rtdetrv2) | – | 44.0 ms · portable fp32 | 4.56 ms · kernels fp16 | 2.76 ms · kernels fp16 | 4.88 ms · kernels fp16 | 4.74 ms · kernels fp16 | 2.21 ms · kernels fp16 |
| [`rtdetrv2-x-coco`](models/detection.md#rtdetrv2) | – | 109 ms · portable fp16 | 10.1 ms · kernels fp16 | 4.89 ms · kernels fp16 | 10.4 ms · kernels fp16 | 8.11 ms · kernels fp16 | 4.07 ms · kernels fp16 |
| [`rtmdet-l-coco`](models/detection.md#rtmdet) | – | 75.8 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-m-coco`](models/detection.md#rtmdet) | – | 51.4 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-m-person`](models/detection.md#rtmdet) | – | 48.8 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-nano-person`](models/detection.md#rtmdet) | – | 17.8 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-s-coco`](models/detection.md#rtmdet) | – | 31.0 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-tiny-coco`](models/detection.md#rtmdet) | – | 27.0 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-x-coco`](models/detection.md#rtmdet) | – | 116 ms · portable fp16 | – | – | – | – | – |
| [`yolox-l`](models/detection.md#yolox) | – | 79.4 ms · portable fp16 | – | – | – | – | – |
| [`yolox-m`](models/detection.md#yolox) | – | 50.0 ms · portable fp16 | – | – | – | – | – |
| [`yolox-s`](models/detection.md#yolox) | – | 33.4 ms · portable fp32 | – | – | – | – | – |
| [`yolox-tiny`](models/detection.md#yolox) | – | 20.1 ms · portable fp16 | – | – | – | – | – |
| [`yolox-x`](models/detection.md#yolox) | – | 123 ms · portable fp16 | – | – | – | – | – |

#### Ocr, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`ppocrv5-mobile`](models/ocr.md#ppocrv5) | 736 ms · portable fp16 |
| [`ppocrv5-server`](models/ocr.md#ppocrv5) | 1607 ms · portable fp16 |

#### Pose, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`dwpose-l`](models/pose.md#rtmpose) | 76.3 ms · portable fp16 |
| [`dwpose-l-384`](models/pose.md#rtmpose) | 155 ms · kernels fp32 |
| [`dwpose-m`](models/pose.md#rtmpose) | 50.9 ms · kernels fp32 |
| [`dwpose-s`](models/pose.md#rtmpose) | 29.8 ms · kernels fp32 |
| [`dwpose-t`](models/pose.md#rtmpose) | 23.7 ms · kernels fp32 |
| [`hrnet-w32`](models/pose.md#hrnet) | 122 ms · portable fp16 |
| [`hrnet-w32-384`](models/pose.md#hrnet) | 217 ms · portable fp16 |
| [`hrnet-w48`](models/pose.md#hrnet) | 198 ms · portable fp16 |
| [`hrnet-w48-384`](models/pose.md#hrnet) | 369 ms · portable fp16 |
| [`retinaface-mobilenet0.25`](models/pose.md#retinaface) | 15.3 ms · portable fp16 |
| [`retinaface-resnet50`](models/pose.md#retinaface) | 150 ms · portable fp16 |
| [`rf-detr-keypoint-preview`](models/pose.md#rfdetr_keypoint) | 148 ms · portable fp32 |
| [`rtmo-l`](models/pose.md#rtmo) | 61.9 ms · portable fp16 |
| [`rtmo-m`](models/pose.md#rtmo) | 43.9 ms · portable fp16 |
| [`rtmo-s`](models/pose.md#rtmo) | 31.0 ms · portable fp16 |
| [`rtmo-t`](models/pose.md#rtmo) | 19.9 ms · portable fp16 |
| [`rtmpose-l`](models/pose.md#rtmpose) | 70.0 ms · portable fp16 |
| [`rtmpose-l-384`](models/pose.md#rtmpose) | 136 ms · portable fp16 |
| [`rtmpose-l-halpe26`](models/pose.md#rtmpose) | 83.7 ms · portable fp16 |
| [`rtmpose-l-halpe26-384`](models/pose.md#rtmpose) | 160 ms · portable fp16 |
| [`rtmpose-m`](models/pose.md#rtmpose) | 45.1 ms · kernels fp32 |
| [`rtmpose-m-384`](models/pose.md#rtmpose) | 83.9 ms · kernels fp32 |
| [`rtmpose-m-ap10k`](models/pose.md#rtmpose) | 65.8 ms · portable fp16 |
| [`rtmpose-m-face6`](models/pose.md#rtmpose) | 67.3 ms · kernels fp32 |
| [`rtmpose-m-halpe26`](models/pose.md#rtmpose) | 54.4 ms · kernels fp32 |
| [`rtmpose-m-halpe26-384`](models/pose.md#rtmpose) | 100 ms · kernels fp32 |
| [`rtmpose-m-hand5`](models/pose.md#rtmpose) | 50.0 ms · kernels fp32 |
| [`rtmpose-s`](models/pose.md#rtmpose) | 24.9 ms · kernels fp32 |
| [`rtmpose-s-face6`](models/pose.md#rtmpose) | 31.9 ms · kernels fp32 |
| [`rtmpose-s-halpe26`](models/pose.md#rtmpose) | 24.9 ms · kernels fp32 |
| [`rtmpose-t`](models/pose.md#rtmpose) | 20.1 ms · kernels fp32 |
| [`rtmpose-t-face6`](models/pose.md#rtmpose) | 25.2 ms · kernels fp32 |
| [`rtmpose-t-halpe26`](models/pose.md#rtmpose) | 20.0 ms · kernels fp32 |
| [`rtmpose-x-halpe26-384`](models/pose.md#rtmpose) | 213 ms · portable fp16 |
| [`rtmw-l`](models/pose.md#rtmpose) | 125 ms · portable fp16 |
| [`rtmw-l-384`](models/pose.md#rtmpose) | 248 ms · portable fp16 |
| [`rtmw-m`](models/pose.md#rtmpose) | 90.0 ms · portable fp16 |
| [`rtmw-x`](models/pose.md#rtmpose) | 184 ms · portable fp16 |
| [`rtmw-x-384`](models/pose.md#rtmpose) | 363 ms · portable fp16 |
| [`vitpose-base`](models/pose.md#vitpose) | 172 ms · kernels fp16 |
| [`vitpose-base-simple`](models/pose.md#vitpose) | 160 ms · kernels fp16 |
| [`vitpose-plus-base`](models/pose.md#vitpose) | 174 ms · kernels fp16 |
| [`vitpose-plus-huge`](models/pose.md#vitpose) | 1158 ms · portable fp16 |
| [`vitpose-plus-large`](models/pose.md#vitpose) | 526 ms · kernels fp16 |
| [`vitpose-plus-small`](models/pose.md#vitpose) | 66.5 ms · kernels fp16 |
| [`yunet-n`](models/pose.md#yunet) | 21.6 ms · portable fp16 |
| [`yunet-s`](models/pose.md#yunet) | 19.4 ms · portable fp16 |

#### Features, 224x224

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](models/features.md#c_radio) | – | 16.5 ms · portable fp16 | 16.4 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v3-g`](models/features.md#c_radio) | – | 145 ms · portable fp16 | 159 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v3-h`](models/features.md#c_radio) | – | 85.0 ms · portable fp16 | 96.4 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v3-l`](models/features.md#c_radio) | – | 47.9 ms · portable fp16 | 47.4 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v4-h`](models/features.md#c_radio) | – | 86.6 ms · portable fp16 | – | – | – | – | – | – |
| [`c-radio-v4-so400m`](models/features.md#c_radio) | – | 58.5 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-b16-laion2b`](models/features.md#clip) | – | 15.3 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-b16-openai`](models/features.md#clip) | – | 16.1 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-h14-378-dfn5b`](models/features.md#clip) | – | 108 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-h14-dfn5b`](models/features.md#clip) | – | 109 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-h14-laion2b`](models/features.md#clip) | – | 102 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-l14-336-openai`](models/features.md#clip) | – | 58.3 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-l14-datacompxl`](models/features.md#clip) | – | 56.7 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-l14-dfn2b`](models/features.md#clip) | – | 60.1 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-l14-laion2b`](models/features.md#clip) | – | 56.6 ms · portable fp16 | – | – | – | – | – | – |
| [`clip-l14-openai`](models/features.md#clip) | – | 58.3 ms · portable fp16 | – | – | – | – | – | – |
| [`convnext_base.fb_in22k`](models/features.md#convnext_features) | – | 24.5 ms · portable fp16 | 16.0 ms · portable fp16 | – | – | – | – | – |
| [`convnext_large.fb_in22k`](models/features.md#convnext_features) | – | 37.5 ms · portable fp16 | 29.1 ms · portable fp16 | – | – | – | – | – |
| [`convnext_small.fb_in22k`](models/features.md#convnext_features) | – | 17.9 ms · portable fp16 | 12.0 ms · portable fp16 | – | – | – | – | – |
| [`convnext_tiny.fb_in22k`](models/features.md#convnext_features) | – | 9.96 ms · portable fp16 | 7.43 ms · portable fp16 | – | – | – | – | – |
| [`dinov2-vitb14`](models/features.md#dinov2) | – | 24.7 ms · portable fp16 | 18.5 ms · portable fp16 | – | – | – | – | 0.985 ms · kernels fp16 |
| [`dinov2-vitb14-reg`](models/features.md#dinov2) | – | 46.4 ms · portable fp16 | 18.9 ms · portable fp16 | – | – | – | – | 0.987 ms · kernels fp16 |
| [`dinov2-vitg14`](models/features.md#dinov2) | – | 218 ms · portable fp16 | 193 ms · portable fp16 | 22.4 ms · kernels fp16 | 6.66 ms · kernels fp16 | 18.0 ms · kernels fp16 | 9.51 ms · kernels fp16 | 4.95 ms · kernels fp16 |
| [`dinov2-vitg14-reg`](models/features.md#dinov2) | 13.0 ms · kernels fp16 | 261 ms · portable fp16 | 206 ms · portable fp16 | – | – | – | – | 4.94 ms · kernels fp16 |
| [`dinov2-vitl14`](models/features.md#dinov2) | – | 65.2 ms · portable fp16 | 59.7 ms · portable fp16 | 6.34 ms · kernels fp16 | 2.39 ms · kernels fp16 | 5.34 ms · portable fp16 | 3.77 ms · kernels fp16 | 2.00 ms · kernels fp16 |
| [`dinov2-vitl14-reg`](models/features.md#dinov2) | – | 95.7 ms · portable fp16 | 58.7 ms · portable fp16 | – | – | – | – | 2.00 ms · kernels fp16 |
| [`dinov2-vits14`](models/features.md#dinov2) | – | 12.7 ms · portable fp16 | 8.60 ms · portable fp16 | 1.05 ms · kernels fp16 | 0.721 ms · kernels fp16 | 0.996 ms · kernels fp16 | 1.40 ms · kernels fp16 | 0.645 ms · kernels fp16 |
| [`dinov2-vits14-reg`](models/features.md#dinov2) | – | 23.7 ms · portable fp16 | 8.14 ms · portable fp16 | – | – | – | – | 0.639 ms · kernels fp16 |
| [`dinov3-convnext-base`](models/features.md#dinov3_convnext) | – | 23.1 ms · portable fp16 | 16.2 ms · portable fp16 | – | – | – | – | 1.78 ms · portable fp16 |
| [`dinov3-convnext-large`](models/features.md#dinov3_convnext) | – | 36.9 ms · portable fp16 | 29.0 ms · kernels fp16 | 4.42 ms · kernels fp16 | 2.36 ms · kernels fp16 | 3.73 ms · kernels fp16 | 4.10 ms · kernels fp16 | 1.88 ms · portable fp16 |
| [`dinov3-convnext-small`](models/features.md#dinov3_convnext) | – | 18.5 ms · portable fp16 | 11.9 ms · kernels fp16 | – | – | – | – | 1.78 ms · kernels fp16 |
| [`dinov3-convnext-tiny`](models/features.md#dinov3_convnext) | – | 10.6 ms · portable fp16 | 8.00 ms · kernels fp16 | 1.46 ms · kernels fp16 | 0.965 ms · kernels fp16 | 1.34 ms · kernels fp16 | 1.79 ms · kernels fp16 | 0.973 ms · kernels fp16 |
| [`dinov3-vitb16`](models/features.md#dinov3) | – | 22.5 ms · portable fp16 | 16.1 ms · portable fp16 | – | – | – | – | 0.865 ms · kernels fp16 |
| [`dinov3-vith16plus`](models/features.md#dinov3) | – | 143 ms · portable fp16 | 133 ms · portable fp16 | 14.2 ms · kernels fp16 | 4.46 ms · kernels fp16 | 11.4 ms · kernels fp16 | 6.54 ms · kernels fp16 | 3.56 ms · kernels fp16 |
| [`dinov3-vitl16`](models/features.md#dinov3) | – | 61.5 ms · portable fp16 | 54.8 ms · portable fp16 | 5.22 ms · kernels fp16 | 2.27 ms · kernels fp16 | 3.76 ms · kernels fp16 | 3.64 ms · kernels fp16 | 1.99 ms · kernels fp16 |
| [`dinov3-vits16`](models/features.md#dinov3) | – | 13.5 ms · portable fp16 | 8.32 ms · portable fp32 | 1.02 ms · kernels fp16 | 0.727 ms · kernels fp16 | 1.02 ms · kernels fp16 | 1.47 ms · kernels fp16 | 0.654 ms · kernels fp16 |
| [`pe-core-b16-224`](models/features.md#perception_encoder) | – | 16.0 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-core-g14-448`](models/features.md#perception_encoder) | – | 259 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-core-l14-336`](models/features.md#perception_encoder) | – | 58.3 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-core-s16-384`](models/features.md#perception_encoder) | – | 10.4 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-core-t16-384`](models/features.md#perception_encoder) | – | 7.25 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-spatial-b16-512`](models/features.md#perception_encoder) | – | 16.1 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-spatial-g14-448`](models/features.md#perception_encoder) | – | 258 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-spatial-l14-448`](models/features.md#perception_encoder) | – | 57.6 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-spatial-s16-512`](models/features.md#perception_encoder) | – | 8.31 ms · kernels fp16 | – | – | – | – | – | – |
| [`pe-spatial-t16-512`](models/features.md#perception_encoder) | – | 5.94 ms · kernels fp16 | – | – | – | – | – | – |
| [`siglip2-b16-256`](models/features.md#siglip2) | – | 15.8 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-b16-512`](models/features.md#siglip2) | – | 15.8 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-g16-256`](models/features.md#siglip2) | – | 143 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-g16-384`](models/features.md#siglip2) | – | 142 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-l16-256`](models/features.md#siglip2) | – | 45.9 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-l16-512`](models/features.md#siglip2) | – | 45.5 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-so400m14-224`](models/features.md#siglip2) | – | 70.2 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-so400m14-378`](models/features.md#siglip2) | – | 68.6 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-so400m16-256`](models/features.md#siglip2) | – | 57.5 ms · portable fp16 | – | – | – | – | – | – |
| [`siglip2-so400m16-512`](models/features.md#siglip2) | – | 56.8 ms · portable fp16 | – | – | – | – | – | – |
| [`tipsv2-b14`](models/features.md#tipsv2) | – | 24.0 ms · portable fp16 | 19.4 ms · portable fp16 | – | – | – | – | – |
| [`tipsv2-l14`](models/features.md#tipsv2) | – | 61.7 ms · portable fp16 | 56.9 ms · portable fp16 | – | – | – | – | – |

#### Retrieval, 480x640

| model | Apple M1 Pro 16 GB |
|---|---|
| [`boq-dinov2`](models/retrieval.md#boq) | 48.2 ms · portable fp16 |
| [`megaloc`](models/retrieval.md#megaloc) | 45.2 ms · portable fp16 |

#### Calibration, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`anycalib-pinhole`](models/calibration.md#anycalib) | 266 ms · portable fp32 |
| [`geocalib-pinhole`](models/calibration.md#geocalib) | 427 ms · portable fp16 |

#### Flow, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`flowformer-small-chairs`](models/flow.md#flowformer) | OOM |
| [`flowformer-small-sintel`](models/flow.md#flowformer) | OOM |
| [`flowformer-small-things`](models/flow.md#flowformer) | OOM |
| [`flowseek-t`](models/flow.md#flowseek) | OOM |
| [`flowseek-t-things`](models/flow.md#flowseek) | OOM |
| [`flowseek-t-things-notartan`](models/flow.md#flowseek) | 3268 ms · portable fp16 |
| [`gmflow-scale1-mixdata`](models/flow.md#gmflow) | 3269 ms · portable fp32 |
| [`gmflow-scale1-things`](models/flow.md#gmflow) | 3269 ms · portable fp32 |
| [`gmflow-scale2-mixdata`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-regrefine6-kitti`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-regrefine6-mixdata`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-regrefine6-sintelft`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-regrefine6-things`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-sintel`](models/flow.md#gmflow) | OOM |
| [`gmflow-scale2-things`](models/flow.md#gmflow) | OOM |
| [`megaflow`](models/flow.md#megaflow) | 11196 ms · portable fp16 |
| [`memflow`](models/flow.md#memflow) | 10016 ms · portable fp32 |
| [`memflow-kitti`](models/flow.md#memflow) | 9826 ms · portable fp32 |
| [`memflow-sintel`](models/flow.md#memflow) | 10075 ms · portable fp32 |
| [`memfof`](models/flow.md#memfof) | 3054 ms · portable fp32 |
| [`memfof-sintel`](models/flow.md#memfof) | OOM |
| [`memfof-spring`](models/flow.md#memfof) | 3060 ms · portable fp32 |
| [`neuflow-v2`](models/flow.md#neuflow_v2) | 1039 ms · portable fp32 |
| [`neuflow-v2-sintel`](models/flow.md#neuflow_v2) | 1076 ms · portable fp32 |
| [`neuflow-v2-things`](models/flow.md#neuflow_v2) | 945 ms · portable fp32 |
| [`raft-large`](models/flow.md#raft) | 1351 ms · kernels fp16 |
| [`raft-large-kitti`](models/flow.md#raft) | 1349 ms · kernels fp16 |
| [`raft-large-things`](models/flow.md#raft) | 1348 ms · kernels fp16 |
| [`raft-small`](models/flow.md#raft) | 386 ms · kernels fp16 |
| [`recover-cx`](models/flow.md#recover) | 853 ms · portable fp16 |
| [`recover-mn`](models/flow.md#recover) | 347 ms · portable fp16 |
| [`recover-rn`](models/flow.md#recover) | 405 ms · portable fp16 |
| [`sea-raft-m`](models/flow.md#sea_raft) | 1220 ms · kernels fp16 |
| [`sea-raft-m-kitti`](models/flow.md#sea_raft) | 1222 ms · kernels fp16 |
| [`sea-raft-m-spring`](models/flow.md#sea_raft) | 340 ms · kernels fp16 |
| [`sea-raft-m-things`](models/flow.md#sea_raft) | 1221 ms · kernels fp16 |
| [`sea-raft-s`](models/flow.md#sea_raft) | 833 ms · kernels fp16 |
| [`sea-raft-s-kitti`](models/flow.md#sea_raft) | 831 ms · kernels fp16 |
| [`sea-raft-s-spring`](models/flow.md#sea_raft) | 233 ms · kernels fp16 |
| [`waft-a1`](models/flow.md#waft) | 7168 ms · portable fp16 |
| [`waft-a1-kitti`](models/flow.md#waft) | 7384 ms · portable fp16 |
| [`waft-a1-sintel`](models/flow.md#waft) | 7430 ms · portable fp16 |

#### Matching, 768x1024

| model | Apple M1 Pro 16 GB | Apple M1 Pro 16 GB (MLX) |
|---|---|---|
| [`aliked-flexglue`](models/matching.md#flexglue) | 1069 ms · portable fp16 | – |
| [`aliked-lightglue`](models/matching.md#lightglue) | 206 ms · portable fp32 | – |
| [`dedode-v2-b`](models/matching.md#dedode) | 1464 ms · portable fp32 | – |
| [`dedode-v2-g`](models/matching.md#dedode) | 3767 ms · portable fp32 | – |
| [`disk-lightglue`](models/matching.md#lightglue) | 1994 ms · portable fp32 | – |
| [`loftr-indoor`](models/matching.md#loftr) | 701 ms · portable fp16 | – |
| [`loftr-outdoor`](models/matching.md#loftr) | 723 ms · portable fp16 | – |
| [`loma-b`](models/matching.md#loma) | 3037 ms · portable fp32 | – |
| [`romav1`](models/matching.md#romav1) | 3050 ms · portable fp32 | – |
| [`romav2`](models/matching.md#romav2) | 3981 ms · kernels fp32 | 3931 ms · kernels fp32 |
| [`xfeat`](models/matching.md#xfeat) | 41.7 ms · portable fp32 | – |
| [`xfeat-lighterglue`](models/matching.md#xfeat) | 179 ms · portable fp32 | – |

#### Tracking, 480x854

| model | Apple M1 Pro 16 GB |
|---|---|
| [`alltracker`](models/tracking.md#alltracker) | 3200 ms · portable fp16 |
| [`alltracker-tiny`](models/tracking.md#alltracker) | 2743 ms · portable fp16 |
| [`bootstapir`](models/tracking.md#tapir) | 1721 ms · portable fp32 |
| [`cotracker3-offline`](models/tracking.md#cotracker) | 1430 ms · portable fp32 |
| [`cotracker3-online`](models/tracking.md#cotracker) | 1707 ms · portable fp32 |
| [`locotrack-b`](models/tracking.md#locotrack) | 2104 ms · portable fp32 |
| [`locotrack-s`](models/tracking.md#locotrack) | 1781 ms · portable fp32 |
| [`tapir`](models/tracking.md#tapir) | 1382 ms · portable fp32 |
| [`tapnext`](models/tracking.md#tapnext) | 5855 ms · portable fp32 |
| [`tapnext-pp`](models/tracking.md#tapnext) | 5680 ms · portable fp32 |
| [`track-on-r`](models/tracking.md#track_on) | 8274 ms · portable fp32 |
| [`track-on2`](models/tracking.md#track_on) | 8354 ms · portable fp32 |

#### Interpolation, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`ema-vfi`](models/interpolation.md#ema_vfi) | 1981 ms · portable fp16 |
| [`ema-vfi-small`](models/interpolation.md#ema_vfi) | 556 ms · portable fp16 |
| [`ema-vfi-small-t`](models/interpolation.md#ema_vfi) | 513 ms · portable fp16 |
| [`ema-vfi-t`](models/interpolation.md#ema_vfi) | 2272 ms · portable fp16 |
| [`film`](models/interpolation.md#film) | OOM |
| [`gimm-vfi-r`](models/interpolation.md#gimm_vfi) | OOM |
| [`gimm-vfi-r-p`](models/interpolation.md#gimm_vfi) | OOM |
| [`rife-4.25`](models/interpolation.md#rife) | 106 ms · portable fp16 |
| [`rife-4.25-lite`](models/interpolation.md#rife) | 80.0 ms · portable fp16 |
| [`rife-4.26`](models/interpolation.md#rife) | 103 ms · portable fp16 |

#### Restoration, 512x512

| model | Apple M1 Pro 16 GB |
|---|---|
| [`big-lama`](models/restoration.md#lama) | 214 ms · portable fp16 |
| [`codeformer`](models/restoration.md#codeformer) | 196 ms · portable fp16 |
| [`dat-2-x4`](models/restoration.md#dat) | 9404 ms · portable fp16 |
| [`dat-light-x4`](models/restoration.md#dat) | 2562 ms · portable fp16 |
| [`dat-x4`](models/restoration.md#dat) | 10512 ms · portable fp16 |
| [`drct-l-x4`](models/restoration.md#drct) | OOM |
| [`drct-x4`](models/restoration.md#drct) | OOM |
| [`gfpgan-v1.4`](models/restoration.md#gfpgan) | 138 ms · portable fp16 |
| [`hat-s-x4`](models/restoration.md#hat) | OOM |
| [`hat-x4`](models/restoration.md#hat) | OOM |
| [`nafnet-gopro`](models/restoration.md#nafnet) | 368 ms · portable fp16 |
| [`nafnet-sidd`](models/restoration.md#nafnet) | 423 ms · portable fp32 |
| [`real-hat-gan-x4`](models/restoration.md#hat) | OOM |
| [`realesr-general-x4v3`](models/restoration.md#compact) | 132 ms · portable fp16 |
| [`realesrgan-x2plus`](models/restoration.md#realesrgan) | 549 ms · portable fp16 |
| [`realesrgan-x4plus`](models/restoration.md#realesrgan) | 2066 ms · portable fp16 |
| [`realesrgan-x4plus-anime-6b`](models/restoration.md#realesrgan) | 664 ms · portable fp16 |
| [`realplksr-nomoswebphoto-x4`](models/restoration.md#realplksr) | 1092 ms · portable fp16 |
| [`restormer-deblur`](models/restoration.md#restormer) | 1376 ms · portable fp32 |
| [`restormer-derain`](models/restoration.md#restormer) | 1678 ms · portable fp32 |
| [`restormer-sidd`](models/restoration.md#restormer) | 2218 ms · portable fp32 |
| [`seedvr2-3b`](models/restoration.md#seedvr2) | unsupported |
| [`seedvr2-7b`](models/restoration.md#seedvr2) | unsupported |
| [`span-x2`](models/restoration.md#span) | 74.0 ms · portable fp16 |
| [`span-x4`](models/restoration.md#span) | 89.0 ms · portable fp16 |
| [`swin2sr-x2`](models/restoration.md#swin2sr) | 9258 ms · portable fp16 |
| [`swin2sr-x4`](models/restoration.md#swin2sr) | 9410 ms · portable fp16 |
| [`swin2sr-x4-real`](models/restoration.md#swin2sr) | 9531 ms · portable fp16 |
| [`swinir-l-x4-real`](models/restoration.md#swinir) | 9874 ms · portable fp16 |
| [`swinir-m-x2`](models/restoration.md#swinir) | 5797 ms · portable fp16 |
| [`swinir-m-x4`](models/restoration.md#swinir) | 5893 ms · portable fp16 |
| [`swinir-m-x4-real`](models/restoration.md#swinir) | 5851 ms · portable fp16 |

#### Volume, 200x200

| model | Apple M1 Pro 16 GB |
|---|---|
| [`nnfoundation-cnn`](models/volume.md#nnfoundation) | 2935 ms · portable fp16 |
| [`nninteractive`](models/volume.md#nninteractive) | 3525 ms · portable fp32 |
| [`totalsegmentator-total`](models/volume.md#totalsegmentator) | 23070 ms · portable fp16 |
| [`totalsegmentator-total-3mm`](models/volume.md#totalsegmentator) | 522 ms · portable fp16 |
<!-- devices:end -->

## RTX 5060 Ti 16 GB (reference)

<!-- tables:begin (tools/gen_docs.py, from docs/data/speed.jsonl) -->

### Stereo, KITTI 2015, 375x1242

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 65.7 | 55.4 | 55.9 | 32.6 | 1.7x | 1.7x | 1139 / 1210 | 469 / 518 | 287 / 380 |
| aanet-sceneflow | 75.4 | 47.9 | 45.9 | 30.6 | 1.6x | 1.5x | 363 / 428 | 310 / 366 | 208 / 264 |
| bridgedepth-rvc-pretrain | 169 | 121 | 114 | 94.7 | 1.3x | 1.2x | 1891 / 2006 | 1910 / 2006 | 1323 / 1446 |
| bridgedepth-sf | 164 | 118 | 111 | 92.1 | 1.3x | 1.2x | 1840 / 1970 | 1859 / 1954 | 1272 / 1394 |
| c-fast-foundationstereo | 203 | 143 | 97.7 | 50.9 | 2.8x | 1.9x | 1006 / 1138 | 733 / 914 | 438 / 554 |
| c-foundationstereo-s | 2499 | 1121 | – | – | – | – | 2781 / 3204 | – | – |
| cas-gwcnet-c-sceneflow | 195 | 115 | 96.7 | 65.8 | 1.8x | 1.5x | 917 / 990 | 668 / 834 | 581 / 670 |
| cas-gwcnet-gc-sceneflow | 250 | 161 | 138 | 103 | 1.6x | 1.3x | 2634 / 2772 | 772 / 954 | 751 / 834 |
| cfnet-sceneflow | 194 | 155 | – | – | – | – | 2211 / 2584 | – | – |
| coex-sceneflow | 26.0 | 16.6 | – | – | – | – | 671 / 698 | – | – |
| crestereo-eth3d | 636 | 480 | 209 | 83.1 | 5.8x | 2.5x | 516 / 642 | 315 / 370 | 245 / 292 |
| defom-stereo-vitl-sceneflow | 947 | 509 | 404 | 184 | 2.8x | 2.2x | 2458 / 2610 | 2394 / 2538 | 1604 / 1846 |
| dlnr-middlebury | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| dlnr-sceneflow | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| fast-acvnet-generalization | 28.1 | 24.1 | – | – | – | – | 461 / 486 | – | – |
| fast-acvnet-kitti12 | 28.1 | 24.1 | – | – | – | – | 461 / 486 | – | – |
| fast-acvnet-kitti15 | 28.1 | 24.1 | – | – | – | – | 461 / 486 | – | – |
| fast-acvnet-plus-generalization | 35.4 | 26.4 | – | – | – | – | 468 / 506 | – | – |
| fast-acvnet-plus-kitti12 | 35.4 | 26.4 | – | – | – | – | 468 / 506 | – | – |
| fast-acvnet-plus-kitti15 | 35.4 | 26.4 | – | – | – | – | 468 / 506 | – | – |
| fast-acvnet-plus-sceneflow | 35.4 | 26.4 | – | – | – | – | 468 / 506 | – | – |
| fast-acvnet-sceneflow | 28.1 | 24.1 | – | – | – | – | 461 / 486 | – | – |
| fast-foundationstereo | 203 | 142 | 97.8 | 51.0 | 2.8x | 1.9x | 1052 / 1108 | 733 / 774 | 438 / 554 |
| foundationstereo-vit-l | 2801 | 1272 | 905 | 491 | 2.6x | 1.8x | 3614 / 3754 | 2154 / 2294 | 1588 / 1658 |
| foundationstereo-vit-s | 2139 | 1052 | 737 | 358 | 2.9x | 2.1x | 2388 / 2484 | 1542 / 1744 | 977 / 1090 |
| great-igev-eth3d | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-kitti | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-middlebury | 1055 | 705 | 492 | 175 | 4.0x | 2.8x | 10728 / 10974 | 1165 / 1298 | 1093 / 1184 |
| great-igev-sceneflow | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| gwcnet-g-sceneflow | 175 | 104 | 104 | 72.7 | 1.4x | 1.4x | 1622 / 1734 | 686 / 882 | 598 / 782 |
| gwcnet-gc-sceneflow | 183 | 107 | 107 | 78.3 | 1.4x | 1.4x | 1867 / 1976 | 687 / 982 | 567 / 642 |
| hitnet-eth3d | 63.1 | 46.0 | – | – | – | – | 746 / – | – | – |
| hitnet-middlebury-d400 | 177 | 130 | – | – | – | – | 2124 / – | – | – |
| igev-plusplus-eth3d | 575 | 383 | – | – | – | – | 976 / 1054 | – | – |
| igev-plusplus-kitti12 | 575 | 383 | – | – | – | – | 976 / 1054 | – | – |
| igev-plusplus-kitti15 | 575 | 383 | – | – | – | – | 976 / 1054 | – | – |
| igev-plusplus-middlebury | 575 | 383 | – | – | – | – | 976 / 1054 | – | – |
| igev-plusplus-sceneflow | 575 | 383 | – | – | – | – | 976 / 1054 | – | – |
| igev-stereo-eth3d | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti12 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti15 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-middlebury | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-sceneflow | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| las2-h | 51.1 | 36.4 | – | – | – | – | 352 / 386 | – | – |
| las2-l | 46.9 | 30.1 | 30.9 | 8.96 | 3.4x | 3.4x | 422 / 488 | 424 / 458 | 169 / 380 |
| las2-m | 23.9 | 20.0 | 23.2 | 5.28 | 3.8x | 4.4x | 360 / 404 | 360 / 386 | 136 / 332 |
| las2-s | 19.3 | 17.3 | – | – | – | – | 345 / 382 | – | – |
| mocha-v1-sceneflow | 648 | 451 | 321 | 165 | 2.7x | 1.9x | 1430 / 1608 | 668 / 750 | 657 / 750 |
| mocha-v2-sceneflow | 615 | 398 | 315 | 164 | 2.4x | 1.9x | 1411 / 1588 | 668 / 750 | 656 / 750 |
| monster-mix-all | 993 | 995 | – | – | – | – | 2733 / 2958 | – | – |
| monster-plusplus-mix-all | 1012 | 1009 | 432 | 195 | 5.2x | 2.2x | 2810 / 3018 | 1339 / 1618 | 1308 / 1660 |
| monster-sceneflow | 993 | 995 | – | – | – | – | 2733 / 2958 | – | – |
| nmrf-sceneflow | 122 | 78.1 | 58.1 | 51.5 | 1.5x | 1.1x | 587 / 634 | 548 / 586 | 364 / 406 |
| psmnet-sceneflow | 248 | 204 | – | – | – | – | 1950 / 1990 | – | – |
| raft-stereo-eth3d | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-middlebury | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-sceneflow | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| rt-monster-plusplus-mix-all | 91.8 | 91.3 | – | – | – | – | 1051 / 1106 | – | – |
| s2m2-l | 597 | 283 | – | 120 | 2.4x | – | 1747 / 1998 | – | 831 / 988 |
| s2m2-m | 356 | 184 | – | 78.2 | 2.4x | – | 1080 / 1284 | – | 491 / 542 |
| s2m2-s | 163 | 99.1 | 91.5 | 38.3 | 2.6x | 2.4x | 676 / 748 | 633 / 750 | 287 / 400 |
| s2m2-xl | 1063 | 529 | – | 212 | 2.5x | – | 3315 / 3484 | – | 1502 / 1614 |
| selective-igev-eth3d-finetune | 637 | 422 | – | – | – | – | 698 / 740 | – | – |
| selective-igev-eth3d-train | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-kitti12 | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-kitti15 | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-middlebury-finetune | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-middlebury-train | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| selective-igev-sceneflow | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
| waft-stereo-b | 380 | 208 | 208 | 93.6 | 2.2x | 2.2x | 1674 / 1860 | 1610 / 1732 | 851 / 936 |
| waft-stereo-s | 234 | 128 | 129 | 65.2 | 2.0x | 2.0x | 1326 / 1480 | 1281 / 1356 | 489 / 566 |

### Stereo, Middlebury 2014 half, 994x1482

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 263 | 194 | 159 | 88.9 | 2.2x | 1.8x | 3652 / 3784 | 1142 / 1278 | 813 / 920 |
| aanet-sceneflow | 289 | 178 | 140 | 101 | 1.8x | 1.4x | 1000 / 1126 | 845 / 986 | 536 / 644 |
| bridgedepth-rvc-pretrain | 629 | 470 | 442 | 354 | 1.3x | 1.2x | 2903 / 3200 | 2968 / 3246 | 2361 / 2686 |
| bridgedepth-sf | 616 | 460 | 437 | 348 | 1.3x | 1.3x | 2852 / 3144 | 2916 / 3214 | 2310 / 2634 |
| c-fast-foundationstereo | 741 | 527 | 330 | 135 | 3.9x | 2.4x | 2892 / 2996 | 2109 / 2336 | 1196 / 1516 |
| c-foundationstereo-s | OOM | 3874 | – | – | – | – | 8338 / 12870 | – | – |
| cas-gwcnet-c-sceneflow | 656 | 389 | 327 | 225 | 1.7x | 1.5x | 2791 / 2970 | 2006 / 2474 | 1725 / 1930 |
| cas-gwcnet-gc-sceneflow | 862 | 567 | 472 | 358 | 1.6x | 1.3x | 8317 / 8730 | 2338 / 2794 | 2272 / 2514 |
| cfnet-sceneflow | 659 | 545 | – | – | – | – | 6765 / 7880 | – | – |
| coex-sceneflow | 94.9 | 60.0 | – | – | – | – | 2098 / 2160 | – | – |
| crestereo-eth3d | 1980 | 1597 | 724 | 229 | 7.0x | 3.2x | 1541 / 1752 | 810 / 946 | 644 / 732 |
| defom-stereo-vitl-sceneflow | 4175 | 2816 | 1531 | 659 | 4.3x | 2.3x | 8639 / 8766 | 4362 / 4838 | 3154 / 3566 |
| dlnr-middlebury | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| dlnr-sceneflow | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| fast-acvnet-generalization | 101 | 60.6 | – | – | – | – | 1421 / 1506 | – | – |
| fast-acvnet-kitti12 | 101 | 60.6 | – | – | – | – | 1421 / 1506 | – | – |
| fast-acvnet-kitti15 | 101 | 60.6 | – | – | – | – | 1421 / 1506 | – | – |
| fast-acvnet-plus-generalization | 138 | 80.9 | – | – | – | – | 1441 / 1522 | – | – |
| fast-acvnet-plus-kitti12 | 138 | 80.9 | – | – | – | – | 1441 / 1522 | – | – |
| fast-acvnet-plus-kitti15 | 138 | 80.9 | – | – | – | – | 1441 / 1522 | – | – |
| fast-acvnet-plus-sceneflow | 138 | 80.9 | – | – | – | – | 1441 / 1522 | – | – |
| fast-acvnet-sceneflow | 101 | 60.6 | – | – | – | – | 1421 / 1506 | – | – |
| fast-foundationstereo | 741 | 527 | 330 | 135 | 3.9x | 2.4x | 2939 / 3306 | 2109 / 2256 | 1196 / 1576 |
| foundationstereo-vit-l | 9228 | 4237 | 2989 | 1440 | 2.9x | 2.1x | 8286 / 8750 | 2415 / 2656 | 2319 / 2760 |
| foundationstereo-vit-s | 7129 | 3609 | 2514 | 1120 | 3.2x | 2.2x | 6987 / 7280 | 2493 / 2606 | 2169 / 2470 |
| great-igev-eth3d | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-kitti | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-middlebury | OOM | OOM | 1785 | 579 | – | 3.1x | – | 1875 / 2078 | 1553 / 1744 |
| great-igev-sceneflow | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| gwcnet-g-sceneflow | 604 | 352 | 344 | 238 | 1.5x | 1.4x | 5105 / 5414 | 2081 / 2402 | 1799 / 2102 |
| gwcnet-gc-sceneflow | 632 | 365 | 353 | 257 | 1.4x | 1.4x | 5887 / 6214 | 2082 / 2402 | 1768 / 1962 |
| hitnet-eth3d | 234 | 182 | – | – | – | – | 2433 / – | – | – |
| hitnet-middlebury-d400 | 637 | 483 | – | – | – | – | 6831 / – | – | – |
| igev-plusplus-eth3d | 2012 | 1278 | – | – | – | – | 2891 / 3108 | – | – |
| igev-plusplus-kitti12 | 2012 | 1278 | – | – | – | – | 2891 / 3108 | – | – |
| igev-plusplus-kitti15 | 2012 | 1278 | – | – | – | – | 2891 / 3108 | – | – |
| igev-plusplus-middlebury | 2012 | 1278 | – | – | – | – | 2891 / 3108 | – | – |
| igev-plusplus-sceneflow | 2012 | 1278 | – | – | – | – | 2891 / 3108 | – | – |
| igev-stereo-eth3d | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti12 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti15 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-middlebury | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-sceneflow | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| las2-h | 205 | 135 | – | – | – | – | 1022 / 1084 | – | – |
| las2-l | 154 | 90.8 | 80.3 | 29.9 | 3.0x | 2.7x | 1010 / 1082 | 1036 / 1098 | 397 / 600 |
| las2-m | 79.1 | 52.4 | 54.1 | 18.3 | 2.9x | 3.0x | 948 / 1000 | 972 / 1026 | 329 / 572 |
| las2-s | 62.0 | 43.5 | – | – | – | – | 933 / 978 | – | – |
| mocha-v1-sceneflow | 2148 | 1456 | 1057 | 455 | 3.2x | 2.3x | 3637 / 3982 | 1632 / 1770 | 1588 / 1730 |
| mocha-v2-sceneflow | 2054 | 1295 | 1033 | 448 | 2.9x | 2.3x | 3576 / 3982 | 1632 / 1770 | 1587 / 1790 |
| monster-mix-all | 4275 | 4277 | – | – | – | – | 6382 / 6430 | – | – |
| monster-plusplus-mix-all | 4339 | 4334 | 1633 | 698 | 6.2x | 2.3x | 6382 / 6430 | 2583 / 2918 | 2438 / 2760 |
| monster-sceneflow | 4275 | 4277 | – | – | – | – | 6382 / 6430 | – | – |
| nmrf-sceneflow | 445 | 313 | 216 | 196 | 1.6x | 1.1x | 1656 / 2054 | 1510 / 1552 | 1022 / 1112 |
| psmnet-sceneflow | 760 | 608 | – | – | – | – | 6015 / 6090 | – | – |
| raft-stereo-eth3d | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-middlebury | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-sceneflow | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| rt-monster-plusplus-mix-all | 252 | 251 | – | – | – | – | 2922 / 3044 | – | – |
| s2m2-l | 1934 | 1005 | – | 376 | 2.7x | – | 3248 / 4090 | – | 1249 / 1568 |
| s2m2-m | 1242 | 666 | – | 248 | 2.7x | – | 2371 / 2920 | – | 876 / 1162 |
| s2m2-s | 582 | 350 | 348 | 113 | 3.1x | 3.1x | 1759 / 1986 | 1733 / 1930 | 686 / 820 |
| s2m2-xl | 3510 | 1796 | – | 682 | 2.6x | – | 5606 / 6080 | – | 2092 / 2274 |
| selective-igev-eth3d-finetune | 2201 | 1455 | – | – | – | – | 2013 / 2152 | – | – |
| selective-igev-eth3d-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti12 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti15 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-finetune | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-sceneflow | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| waft-stereo-b | 1898 | 1140 | 1137 | 393 | 2.9x | 2.9x | 5743 / 5876 | 5775 / 5852 | 1907 / 2316 |
| waft-stereo-s | 1252 | 689 | 689 | 284 | 2.4x | 2.4x | 3314 / 3398 | 3293 / 3456 | 1113 / 1286 |

### Stereo, synthetic, 994x1482

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved): portable | kernels |
|---|---|---|---|---|---|---|---|
| bridgedepth-rvc-pretrain | – | – | 444 | 355 | 1.2x | 2836 / 3144 | 2226 / 2522 |
| c-fast-foundationstereo | – | – | – | 135 | – | – | 1120 / 1236 |
| c-foundationstereo-s | OOM | – | 3631 | 1366 | 2.7x | 4357 / 4950 | 3058 / 4408 |
| cas-gwcnet-gc-sceneflow | – | – | 473 | 359 | 1.3x | 2032 / 2228 | 2035 / 2230 |
| cfnet-sceneflow | – | – | 409 | 225 | 1.8x | 1822 / 2578 | 1073 / 1604 |
| coex-sceneflow | – | – | 49.9 | 23.5 | 2.1x | 1093 / 1174 | 368 / 514 |
| defom-stereo-vitl-sceneflow | – | – | 1439 | 645 | 2.2x | 4102 / 4252 | 2781 / 4234 |
| dlnr-middlebury | – | – | 399 | 215 | 1.9x | 1502 / 1976 | 851 / 982 |
| fast-acvnet-generalization | – | – | 54.7 | 38.1 | 1.4x | 653 / 756 | 654 / 716 |
| fast-acvnet-kitti12 | – | – | 54.7 | 38.1 | 1.4x | 653 / 756 | 654 / 716 |
| fast-acvnet-kitti15 | – | – | 54.7 | 38.1 | 1.4x | 653 / 756 | 654 / 716 |
| fast-acvnet-plus-generalization | – | – | 70.5 | 46.5 | 1.5x | 651 / 756 | 651 / 716 |
| fast-acvnet-plus-kitti12 | – | – | 70.6 | 46.5 | 1.5x | 651 / 756 | 651 / 716 |
| fast-acvnet-plus-kitti15 | – | – | 70.5 | 46.5 | 1.5x | 651 / 756 | 651 / 716 |
| fast-acvnet-plus-sceneflow | – | – | 70.5 | 46.5 | 1.5x | 651 / 756 | 651 / 716 |
| fast-acvnet-sceneflow | – | – | 54.7 | 38.1 | 1.4x | 653 / 756 | 654 / 716 |
| fast-foundationstereo | – | – | – | 135 | – | – | 1120 / 1236 |
| foundationstereo-vit-l | – | – | 2992 | 1432 | 2.1x | 2923 / 3136 | 2597 / 2920 |
| great-igev-eth3d | – | – | – | 397 | – | – | 1134 / 1302 |
| great-igev-kitti | – | – | – | 396 | – | – | 1134 / 1302 |
| great-igev-middlebury | OOM | OOM | 1789 | 580 | 3.1x | 1602 / 1878 | 1472 / 1642 |
| great-igev-sceneflow | – | – | – | 397 | – | – | 1134 / 1302 |
| gwcnet-gc-sceneflow | – | – | 353 | 259 | 1.4x | 1792 / 2102 | 1510 / 1782 |
| hitnet-eth3d | – | – | 67.6 | 23.2 | 2.9x | 848 / 1068 | 623 / 770 |
| hitnet-middlebury-d400 | – | – | 176 | 53.1 | 3.3x | 1701 / 2154 | 1207 / 1514 |
| igev-plusplus-eth3d | – | – | 1079 | 427 | 2.5x | 1583 / 1696 | 1272 / 1514 |
| igev-plusplus-kitti12 | – | – | 1080 | 394 | 2.7x | 1583 / 1696 | 1272 / 1514 |
| igev-plusplus-kitti15 | – | – | 1080 | 394 | 2.7x | 1583 / 1696 | 1272 / 1494 |
| igev-plusplus-middlebury | – | – | 1080 | 393 | 2.7x | 1583 / 1696 | 1272 / 1494 |
| igev-plusplus-sceneflow | – | – | 1079 | 395 | 2.7x | 1583 / 1696 | 1272 / 1514 |
| las2-h | – | – | 113 | 53.6 | 2.1x | 1002 / 1046 | 872 / 960 |
| las2-l | – | – | 75.7 | 30.1 | 2.5x | 648 / 716 | 268 / 362 |
| las2-m | – | – | 37.1 | 18.1 | 2.1x | 584 / 664 | 235 / 336 |
| las2-s | – | – | 27.9 | 14.3 | 1.9x | 557 / 608 | 226 / 320 |
| monster-mix-all | – | – | 1585 | 656 | 2.4x | 1926 / 2092 | 1829 / 1954 |
| monster-plusplus-mix-all | – | – | – | 657 | – | – | 2438 / 2754 |
| monster-sceneflow | – | – | 1585 | 655 | 2.4x | 2325 / 2574 | 2216 / 2514 |
| psmnet-sceneflow | – | – | 340 | 231 | 1.5x | 1881 / 2402 | 1469 / 1782 |
| raft-stereo-eth3d | – | – | – | 275 | – | – | 722 / 738 |
| raft-stereo-middlebury | – | – | – | 274 | – | – | 722 / 758 |
| raft-stereo-sceneflow | – | – | – | 274 | – | – | 722 / 758 |
| rt-monster-plusplus-mix-all | – | – | 109 | 93.8 | 1.2x | 1291 / 1378 | 1277 / 1420 |
| s2m2-l | – | – | 1017 | – | – | 3234 / 3816 | – |
| s2m2-m | – | – | 670 | – | – | 2289 / 2766 | – |
| s2m2-xl | – | – | 1824 | – | – | 4811 / 5234 | – |
| selective-igev-eth3d-finetune | – | – | 996 | 351 | 2.8x | 1227 / 1448 | 1051 / 1190 |
| waft-stereo-b | – | – | 1094 | 388 | 2.8x | 2999 / 3070 | 1664 / 1932 |
| waft-stereo-s | – | – | – | 284 | – | – | 954 / 1102 |

### Depth, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| any2full-b | 994x1482 | – | 108 | 26.8 | 17.2 | 6.3x | 1.6x | 1109 / 1160 | 538 / 596 | 481 / 536 |
| any2full-l | 994x1482 | – | 269 | 67.7 | 42.3 | 6.4x | 1.6x | 2436 / 2788 | 1178 / 1250 | 1115 / 1210 |
| any2full-s | 994x1482 | – | 47.5 | 12.4 | 9.47 | 5.0x | 1.3x | 646 / 670 | 274 / 312 | 218 / 252 |
| da3-giant-1.1 | 994x1482 | – | – | 71.5 | 45.2 | – | 1.6x | – | 2634 / 2728 | 2535 / 2590 |
| da3-large-1.1 | 994x1482 | – | – | 30.0 | 17.3 | – | 1.7x | – | 988 / 1088 | 891 / 950 |
| da3metric-large | 994x1482 | – | – | 24.3 | 15.0 | – | 1.6x | – | 811 / 868 | 802 / 848 |
| da3mono-large | 994x1482 | – | – | 24.1 | 14.9 | – | 1.6x | – | 814 / 868 | 805 / 868 |
| da3nested-giant-large-1.1 | 994x1482 | – | – | 92.5 | 59.5 | – | 1.6x | – | 3118 / 3194 | 3120 / 3198 |
| depth-anything-v2-b | 518x770 | 88.8 | 53.8 | 18.8 | 10.6 | 5.1x | 1.8x | 928 / 1026 | 433 / 516 | 370 / 436 |
| depth-anything-v2-l | 518x770 | 240 | 136 | 55.0 | 31.6 | 4.3x | 1.7x | 2560 / 2654 | 1009 / 1352 | 904 / 1034 |
| depth-anything-v2-metric-hypersim-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| depth-anything-v2-metric-hypersim-l | 994x1482 | – | – | 54.5 | 31.7 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| depth-anything-v2-metric-hypersim-s | 994x1482 | – | – | 7.66 | 4.67 | – | 1.6x | – | 244 / 288 | 152 / 208 |
| depth-anything-v2-metric-vkitti-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| depth-anything-v2-metric-vkitti-l | 994x1482 | – | – | 54.5 | 31.9 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| depth-anything-v2-metric-vkitti-s | 994x1482 | – | – | 7.66 | 4.67 | – | 1.6x | – | 244 / 288 | 152 / 208 |
| depth-anything-v2-s | 518x770 | 45.5 | 38.1 | 7.73 | 4.45 | 8.6x | 1.7x | 373 / 418 | 232 / 326 | 173 / 248 |
| depth-pro | 994x1482 | – | – | 521 | 277 | – | 1.9x | – | 3036 / 3298 | 3050 / 3198 |
| depthfm-v1 | 994x1482 | – | OOM | 4854 | 3555 | – | 1.4x | – | 5422 / 7404 | 3768 / 5942 |
| distill-any-depth-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| distill-any-depth-l | 994x1482 | – | – | 54.4 | 31.8 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| distill-any-depth-s | 994x1482 | – | – | 7.65 | 4.63 | – | 1.7x | – | 244 / 288 | 152 / 208 |
| lingbot-depth | 994x1482 | – | 297 | 280 | 161 | 1.8x | 1.7x | 2267 / 2338 | 1559 / 1872 | 1453 / 1590 |
| lingbot-depth-dc | 994x1482 | – | 297 | 281 | 161 | 1.8x | 1.7x | 2267 / 2338 | 1559 / 1872 | 1453 / 1590 |
| lotus-normal-d-v1-1 | 994x1482 | – | 344 | 268 | 164 | 2.1x | 1.6x | 3835 / 3898 | 2777 / 2940 | 2351 / 2500 |
| lotus-normal-g-v1-1 | 994x1482 | – | 344 | 267 | 164 | 2.1x | 1.6x | 3836 / 3898 | 2777 / 2940 | 2351 / 2500 |
| marigold-depth-v1-1 | 994x1482 | – | – | 410 | 261 | – | 1.6x | – | 2776 / 2938 | 2343 / 2500 |
| marigold-normals-lcm-v0-1 | 994x1482 | – | 280 | 272 | 167 | 1.7x | 1.6x | 3441 / 3640 | 2792 / 2960 | 2361 / 2500 |
| marigold-v2-depth | 994x1482 | – | OOM | 4763 | 3805 | – | 1.3x | – | 1408 / 1838 | 1408 / 1838 |
| metric3d-vit-large | 994x1482 | – | – | 187 | 151 | – | 1.2x | – | 1239 / 1436 | 1464 / 1568 |
| midas-dpt-beit-large-512 | 518x770 | 399 | 320 | 75.6 | 63.9 | 5.0x | 1.2x | 2759 / 2858 | 975 / 1418 | 986 / 1436 |
| moge-2-vitb-normal | 994x1482 | – | – | 83.9 | 40.7 | – | 2.1x | – | 1163 / 1312 | 1093 / 1208 |
| moge-2-vitl-normal | 518x770 | 500 | 169 | 145 | 73.9 | 2.3x | 2.0x | 2326 / 2418 | 1645 / 1836 | 1481 / 1610 |
| moge-2-vits-normal | 994x1482 | – | – | 61.6 | 33.3 | – | 1.9x | – | 1007 / 1172 | 932 / 1048 |
| moge-3-vitl | 994x1482 | – | – | 328 | 149 | – | 2.2x | – | 1622 / 1706 | 1626 / 1762 |
| omni-dc | 994x1482 | – | 875 | 571 | 510 | 1.7x | 1.1x | 5966 / 6732 | 4130 / 6144 | 4154 / 6528 |
| pointdit-b-512 | 994x1482 | – | – | 31.1 | 20.5 | – | 1.5x | – | 446 / 504 | 462 / 630 |
| pointdit-l-512 | 994x1482 | – | – | 94.6 | 61.3 | – | 1.5x | – | 1297 / 1430 | 1310 / 1676 |
| prompt-depth-anything-l | 756x1008 | – | – | 125 | 69.3 | – | 1.8x | – | 1318 / 1488 | 1179 / 1348 |
| prompt-depth-anything-s | 756x1008 | – | – | 18.4 | 11.4 | – | 1.6x | – | 363 / 442 | 297 / 384 |
| stablenormal-turbo | 994x1482 | – | 299 | 291 | 180 | 1.7x | 1.6x | 4156 / 4374 | 3316 / 3626 | 3037 / 3172 |
| unidepth-v2-vitl14 | 994x1482 | – | – | – | 63.0 | – | – | – | – | 1153 / 1252 |
| unik3d-vitl | 518x770 | 123 | 123 | 99.2 | 58.5 | 2.1x | 1.7x | 3220 / 3530 | 1172 / 1492 | 1070 / 1106 |
| video-depth-anything-metric-s | 1080x1920 | – | – | 202 | 200 | – | 1.0x | – | 1962 / 2224 | 1967 / 2226 |
| video-depth-anything-s | 1080x1920 | – | – | 202 | 200 | – | 1.0x | – | 1962 / 2204 | 1967 / 2226 |
| zoedepth-nk | 994x1482 | – | – | 124 | – | – | – | – | 1843 / 1944 | – |

### Depth, Middlebury 2014 half, 994x1482

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| da3-giant-1.1 | 410 | 227 | 81.8 | 44.2 | 5.1x | 1.9x | 8590 / 8718 | 2638 / 4608 | 2628 / 2710 |
| da3-large-1.1 | 208 | 166 | 37.4 | 17.1 | 9.7x | 2.2x | 3374 / 3472 | 992 / 1350 | 985 / 1090 |
| da3metric-large | 128 | 85.1 | 24.1 | 14.7 | 5.8x | 1.6x | 2996 / 3050 | 815 / 1328 | 811 / 868 |
| da3mono-large | 127 | 85.2 | 23.9 | 14.6 | 5.8x | 1.6x | 2997 / 3052 | 818 / 1328 | 814 / 868 |
| da3nested-giant-large-1.1 | 533 | 308 | 108 | 58.2 | 5.3x | 1.9x | 10332 / 10402 | 3275 / 5914 | 3271 / 3378 |
| depth-anything-v2-b | 93.6 | 63.2 | 18.8 | 10.5 | 6.0x | 1.8x | 928 / 1026 | 450 / 516 | 387 / 436 |
| depth-anything-v2-l | 245 | 140 | 55.1 | 31.1 | 4.5x | 1.8x | 2560 / 2654 | 1026 / 1352 | 921 / 1034 |
| depth-anything-v2-metric-hypersim-b | 91.4 | 60.7 | – | – | – | – | 928 / 1026 | – | – |
| depth-anything-v2-metric-hypersim-l | 246 | 141 | – | – | – | – | 2560 / 2654 | – | – |
| depth-anything-v2-metric-hypersim-s | 53.0 | 43.6 | – | – | – | – | 373 / 418 | – | – |
| depth-anything-v2-metric-vkitti-b | 90.8 | 60.6 | – | – | – | – | 928 / 1026 | – | – |
| depth-anything-v2-metric-vkitti-l | 245 | 141 | – | – | – | – | 2560 / 2654 | – | – |
| depth-anything-v2-metric-vkitti-s | 46.1 | 36.0 | – | – | – | – | 373 / 418 | – | – |
| depth-anything-v2-s | 53.6 | 41.0 | 7.82 | 4.51 | 9.1x | 1.7x | 373 / 418 | 249 / 326 | 190 / 248 |
| depth-pro | 1776 | 569 | – | – | – | – | 3828 / 4294 | – | – |
| distill-any-depth-b | 90.5 | 59.8 | – | – | – | – | 928 / 1026 | – | – |
| distill-any-depth-l | 244 | 139 | – | – | – | – | 2560 / 2654 | – | – |
| distill-any-depth-s | 45.3 | 35.2 | – | – | – | – | 373 / 418 | – | – |
| marigold-depth-v1-1 | 963 | 439 | – | – | – | – | 3414 / 3580 | – | – |
| metric3d-vit-large | 629 | 438 | – | – | – | – | 3275 / 3350 | – | – |
| midas-dpt-beit-large-512 | 386 | 333 | 78.8 | 63.9 | 5.2x | 1.2x | 2759 / 2898 | 992 / 1418 | 1003 / 1436 |
| moge-2-vitb-normal | 237 | 102 | – | – | – | – | 1469 / 1596 | – | – |
| moge-2-vitl-normal | 501 | 171 | 144 | 73.4 | 2.3x | 2.0x | 2339 / 2438 | 1661 / 1836 | 1498 / 1610 |
| moge-2-vits-normal | 162 | 82.6 | – | – | – | – | 1193 / 1268 | – | – |
| moge-3-vitl | 843 | 486 | – | – | – | – | 2505 / 2630 | – | – |
| pointdit-b-512 | 105 | 48.7 | – | – | – | – | 1214 / 1246 | – | – |
| pointdit-l-512 | 335 | 112 | – | – | – | – | 3953 / 4010 | – | – |
| unidepth-v2-vitl14 | 132 | 132 | 99.8 | 66.4 | 2.0x | 1.5x | 3230 / 3508 | 1208 / 1472 | 1211 / 1472 |
| unik3d-vitl | 134 | 134 | 108 | 62.9 | 2.1x | 1.7x | 3307 / 3570 | 1219 / 1492 | 1166 / 1266 |
| zoedepth-nk | 408 | 271 | 124 | 88.9 | 3.1x | 1.4x | 2040 / 2146 | 2037 / 2124 | 1771 / 1960 |

### Classification, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet.tv_in1k | 224x224 | 1.17 | 1.82 | 0.970 | 0.604 | 1.9x | 1.6x | 383 / 404 | 151 / 224 | 184 / 228 |
| coatnet_0_rw_224.sw_in1k | 224x224 | 7.64 | 9.59 | 6.49 | 1.74 | 4.4x | 3.7x | 194 / 210 | 102 / 124 | 100 / 144 |
| coatnet_1_rw_224.sw_in1k | 224x224 | 12.3 | 15.7 | 10.7 | 2.89 | 4.3x | 3.7x | 277 / 294 | 132 / 158 | 130 / 196 |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 224x224 | 14.6 | 15.7 | 11.9 | 4.01 | 3.6x | 3.0x | 466 / 470 | 207 / 240 | 204 / 240 |
| coatnet_nano_rw_224.sw_in1k | 224x224 | 7.88 | 10.6 | 7.23 | 1.62 | 4.8x | 4.4x | 124 / 146 | 74 / 86 | 72 / 106 |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 224x224 | 13.9 | 18.3 | 12.3 | 3.01 | 4.6x | 4.1x | 278 / 296 | 132 / 158 | 130 / 178 |
| coatnet_rmlp_1_rw_224.sw_in1k | 224x224 | 14.0 | 18.4 | 12.4 | 3.04 | 4.6x | 4.1x | 278 / 296 | 131 / 156 | 129 / 178 |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 224x224 | 15.3 | 18.6 | 12.8 | 4.24 | 3.6x | 3.0x | 466 / 470 | 207 / 240 | 204 / 240 |
| coatnet_rmlp_2_rw_224.sw_in1k | 224x224 | 15.3 | 18.6 | 15.1 | 4.24 | 3.6x | 3.6x | 466 / 472 | 207 / 240 | 204 / 240 |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 384x384 | 27.9 | 17.9 | 13.9 | 12.8 | 1.4x | 1.1x | 516 / 528 | 296 / 330 | 288 / 356 |
| coatnet_rmlp_nano_rw_224.sw_in1k | 224x224 | 8.95 | 12.1 | 8.09 | 1.75 | 5.1x | 4.6x | 124 / 146 | 78 / 106 | 77 / 106 |
| convnext_base.fb_in1k | 224x224 | 7.77 | 11.2 | 5.84 | 2.00 | 3.9x | 2.9x | 541 / 566 | 211 / 234 | 208 / 236 |
| convnext_base.fb_in22k_ft_in1k | 224x224 | 7.78 | 11.3 | 5.84 | 2.01 | 3.9x | 2.9x | 541 / 566 | 211 / 234 | 208 / 236 |
| convnext_base.fb_in22k_ft_in1k_384 | 224x224 | 11.6 | 11.2 | 5.79 | 3.05 | 3.7x | 1.9x | 549 / 562 | 229 / 252 | 213 / 232 |
| convnext_large.fb_in1k | 224x224 | 10.7 | 11.5 | 5.81 | 3.28 | 3.2x | 1.8x | 1166 / 1180 | 423 / 432 | 419 / 434 |
| convnext_large.fb_in22k_ft_in1k | 224x224 | 10.7 | 11.5 | 5.88 | 3.28 | 3.2x | 1.8x | 1166 / 1180 | 423 / 432 | 419 / 434 |
| convnext_large.fb_in22k_ft_in1k_384 | 224x224 | 22.9 | 11.5 | 6.11 | 5.21 | 2.2x | 1.2x | 1177 / 1260 | 449 / 492 | 426 / 454 |
| convnext_small.fb_in1k | 224x224 | 7.44 | 11.3 | 5.97 | 1.84 | 4.1x | 3.2x | 321 / 338 | 136 / 150 | 134 / 152 |
| convnext_small.fb_in22k_ft_in1k | 224x224 | 7.51 | 11.3 | 5.84 | 1.84 | 4.1x | 3.2x | 321 / 338 | 136 / 150 | 134 / 152 |
| convnext_small.fb_in22k_ft_in1k_384 | 224x224 | 8.04 | 11.2 | 5.82 | 2.53 | 3.2x | 2.3x | 328 / 338 | 150 / 172 | 138 / 152 |
| convnext_tiny.fb_in1k | 224x224 | 4.05 | 6.14 | 3.31 | 1.15 | 3.5x | 2.9x | 198 / 216 | 94 / 110 | 92 / 110 |
| convnext_tiny.fb_in22k_ft_in1k | 224x224 | 4.11 | 6.13 | 3.34 | 1.15 | 3.6x | 2.9x | 198 / 216 | 94 / 110 | 92 / 110 |
| convnext_tiny.fb_in22k_ft_in1k_384 | 224x224 | 4.59 | 6.05 | 3.30 | 1.56 | 2.9x | 2.1x | 204 / 216 | 108 / 132 | 96 / 110 |
| convnextv2_base.fcmae_ft_in22k_in1k | 224x224 | 10.7 | 14.6 | 10.2 | 2.96 | 3.6x | 3.4x | 542 / 566 | 220 / 234 | 206 / 238 |
| davit_base.msft_in1k | 224x224 | 13.4 | 20.0 | 11.7 | 2.09 | 6.4x | 5.6x | 537 / 548 | 212 / 224 | 209 / 244 |
| davit_small.msft_in1k | 224x224 | 13.0 | 18.8 | 12.8 | 1.88 | 6.9x | 6.8x | 319 / 324 | 136 / 150 | 134 / 168 |
| davit_tiny.msft_in1k | 224x224 | 6.93 | 9.88 | 6.33 | 1.17 | 5.9x | 5.4x | 196 / 204 | 96 / 116 | 93 / 134 |
| deit_base_patch16_224.fb_in1k | 224x224 | 6.39 | 5.46 | 2.89 | 1.66 | 3.3x | 1.7x | 531 / 548 | 203 / 226 | 200 / 248 |
| densenet121.tv_in1k | 224x224 | 10.7 | 13.6 | 9.68 | 1.29 | 8.3x | 7.5x | 80 / 102 | 53 / 70 | 53 / 92 |
| dinov2_vitl14_lc | 224x224 | 22.4 | 10.8 | 5.43 | 4.95 | 2.2x | 1.1x | 1782 / 1806 | 623 / 1228 | 623 / 672 |
| efficientformer_l1.snap_dist_in1k | 224x224 | 4.87 | 6.40 | 3.72 | 0.704 | 6.9x | 5.3x | 104 / 124 | 93 / 104 | 91 / 126 |
| efficientformer_l3.snap_dist_in1k | 224x224 | 8.39 | 11.0 | 6.37 | 1.21 | 6.9x | 5.3x | 214 / 236 | 161 / 216 | 160 / 218 |
| efficientformer_l7.snap_dist_in1k | 224x224 | 12.3 | 15.9 | 9.65 | 2.04 | 6.0x | 4.7x | 505 / 522 | 300 / 324 | 298 / 346 |
| efficientformerv2_l.snap_dist_in1k | 224x224 | 19.9 | 29.0 | 17.1 | 3.05 | 6.5x | 5.6x | 186 / 198 | 93 / 144 | 94 / 188 |
| efficientformerv2_s0.snap_dist_in1k | 224x224 | 8.16 | 11.8 | 7.62 | 1.20 | 6.8x | 6.4x | 55 / 64 | 44 / 64 | 44 / 66 |
| efficientformerv2_s1.snap_dist_in1k | 224x224 | 10.1 | 14.9 | 8.91 | 1.39 | 7.3x | 6.4x | 70 / 80 | 50 / 88 | 50 / 88 |
| efficientformerv2_s2.snap_dist_in1k | 224x224 | 15.2 | 21.8 | 13.9 | 2.17 | 7.0x | 6.4x | 108 / 116 | 64 / 104 | 64 / 104 |
| efficientnet_b0.tv_in1k | 224x224 | 5.64 | 7.91 | 4.98 | 0.769 | 7.3x | 6.5x | 64 / 84 | 48 / 96 | 78 / 118 |
| efficientnet_b4.tv_in1k | 224x224 | 11.1 | 15.7 | 9.93 | 2.02 | 5.5x | 4.9x | 147 / 174 | 89 / 114 | 90 / 118 |
| efficientnet_v2_s.tv_in1k | 224x224 | 12.1 | 16.6 | 10.2 | 1.91 | 6.4x | 5.3x | 159 / 176 | 81 / 126 | 81 / 128 |
| efficientvit_b2.r288_in1k | 224x224 | 10.7 | 14.7 | 12.9 | 1.12 | 9.5x | 11.5x | 173 / 194 | 85 / 144 | 88 / 170 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 224x224 | 83.7 | 32.1 | 18.2 | 14.1 | 2.3x | 1.3x | 1819 / 1864 | 648 / 664 | 652 / 786 |
| falconsai-nsfw-vit | 224x224 | 6.76 | 5.74 | 2.83 | 1.61 | 3.6x | 1.8x | 200 / 226 | 201 / 226 | 198 / 228 |
| fastvit_sa24.apple_dist_in1k | 256x256 | 7.10 | 10.9 | 6.03 | 1.41 | 5.0x | 4.3x | 157 / 176 | 81 / 110 | 80 / 132 |
| fastvit_t12.apple_dist_in1k | 256x256 | 3.61 | 5.66 | 3.27 | 0.781 | 4.6x | 4.2x | 77 / 84 | 54 / 76 | 52 / 76 |
| fastvit_t8.apple_dist_in1k | 256x256 | 3.07 | 4.91 | 3.25 | 0.709 | 4.3x | 4.6x | 57 / 76 | 45 / 72 | 45 / 76 |
| focalnet_base_lrf.ms_in1k | 224x224 | 18.0 | 22.1 | 15.5 | 5.44 | 3.3x | 2.9x | 546 / 560 | 212 / 234 | 212 / 258 |
| focalnet_base_srf.ms_in1k | 224x224 | 17.1 | 20.0 | 14.2 | 4.99 | 3.4x | 2.8x | 542 / 558 | 211 / 234 | 210 / 256 |
| focalnet_small_lrf.ms_in1k | 224x224 | 16.8 | 22.0 | 15.8 | 4.18 | 4.0x | 3.8x | 323 / 338 | 136 / 152 | 136 / 154 |
| focalnet_small_srf.ms_in1k | 224x224 | 15.9 | 20.1 | 14.0 | 3.73 | 4.3x | 3.7x | 320 / 336 | 136 / 152 | 134 / 154 |
| focalnet_tiny_lrf.ms_in1k | 224x224 | 8.92 | 11.4 | 8.07 | 2.39 | 3.7x | 3.4x | 199 / 208 | 95 / 118 | 93 / 138 |
| focalnet_tiny_srf.ms_in1k | 224x224 | 8.39 | 10.4 | 7.36 | 2.19 | 3.8x | 3.4x | 197 / 206 | 95 / 118 | 93 / 138 |
| hgnetv2_b0.ssld_stage2_ft_in1k | 224x224 | 4.45 | 5.89 | 3.46 | 0.568 | 7.8x | 6.1x | 68 / 72 | 48 / 70 | 46 / 72 |
| hgnetv2_b1.ssld_stage2_ft_in1k | 224x224 | 4.57 | 6.17 | 3.47 | 0.603 | 7.6x | 5.7x | 70 / 92 | 50 / 68 | 48 / 70 |
| hgnetv2_b2.ssld_stage2_ft_in1k | 224x224 | 6.25 | 7.98 | 4.55 | 0.765 | 8.2x | 5.9x | 98 / 118 | 59 / 94 | 57 / 94 |
| hgnetv2_b3.ssld_stage2_ft_in1k | 224x224 | 7.16 | 9.30 | 5.22 | 0.909 | 7.9x | 5.7x | 127 / 140 | 76 / 114 | 67 / 116 |
| hgnetv2_b4.ssld_stage2_ft_in1k | 224x224 | 7.06 | 8.41 | 4.53 | 0.864 | 8.2x | 5.2x | 147 / 166 | 82 / 116 | 75 / 138 |
| hgnetv2_b5.ssld_stage2_ft_in1k | 224x224 | 11.4 | 13.8 | 7.04 | 1.29 | 8.9x | 5.5x | 260 / 280 | 114 / 150 | 115 / 156 |
| hgnetv2_b6.ssld_stage2_ft_in1k | 224x224 | 16.5 | 17.9 | 9.44 | 2.10 | 7.9x | 4.5x | 465 / 490 | 198 / 224 | 188 / 232 |
| inception_v3.tv_in1k | 224x224 | 7.88 | 10.3 | 5.13 | 1.31 | 6.0x | 3.9x | 185 / 206 | 120 / 164 | 118 / 208 |
| mobilenet_v2.tv2_in1k | 224x224 | 3.60 | 5.14 | 2.95 | 0.439 | 8.2x | 6.7x | 54 / 72 | 43 / 70 | 75 / 92 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 4.33 | 6.20 | 3.80 | 0.545 | 7.9x | 7.0x | 65 / 80 | 46 / 76 | 78 / 114 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 2.19 | 2.96 | 1.77 | 0.382 | 5.7x | 4.6x | 58 / 74 | 44 / 70 | 43 / 72 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 5.78 | 7.77 | 3.91 | 0.641 | 9.0x | 6.1x | 90 / 106 | 54 / 86 | 87 / 124 |
| regnety_160.tv2_in1k | 224x224 | 12.1 | 12.2 | 8.08 | 1.87 | 6.5x | 4.3x | 512 / 524 | 208 / 250 | 202 / 276 |
| repvit_m0_9.dist_300e_in1k | 224x224 | 8.45 | 12.7 | 5.35 | 0.776 | 10.9x | 6.9x | 65 / 84 | 45 / 76 | 45 / 76 |
| repvit_m0_9.dist_450e_in1k | 224x224 | 8.55 | 12.6 | 5.41 | 0.767 | 11.1x | 7.1x | 65 / 84 | 45 / 76 | 45 / 76 |
| repvit_m1_0.dist_300e_in1k | 224x224 | 8.79 | 12.6 | 5.29 | 0.786 | 11.2x | 6.7x | 75 / 92 | 48 / 78 | 48 / 78 |
| repvit_m1_0.dist_450e_in1k | 224x224 | 8.80 | 12.6 | 5.44 | 0.785 | 11.2x | 6.9x | 75 / 92 | 48 / 78 | 48 / 78 |
| repvit_m1_1.dist_300e_in1k | 224x224 | 8.14 | 11.7 | 5.03 | 0.765 | 10.6x | 6.6x | 84 / 98 | 51 / 62 | 51 / 64 |
| repvit_m1_1.dist_450e_in1k | 224x224 | 8.07 | 11.7 | 4.99 | 0.758 | 10.6x | 6.6x | 84 / 98 | 51 / 62 | 51 / 64 |
| repvit_m1_5.dist_300e_in1k | 224x224 | 13.9 | 20.4 | 8.55 | 1.19 | 11.8x | 7.2x | 118 / 124 | 62 / 94 | 62 / 94 |
| repvit_m1_5.dist_450e_in1k | 224x224 | 13.9 | 20.5 | 8.71 | 1.19 | 11.7x | 7.3x | 118 / 124 | 62 / 94 | 62 / 94 |
| repvit_m2_3.dist_300e_in1k | 224x224 | 17.9 | 26.1 | 11.1 | 1.66 | 10.8x | 6.7x | 170 / 184 | 80 / 128 | 79 / 130 |
| repvit_m2_3.dist_450e_in1k | 224x224 | 18.2 | 26.1 | 10.9 | 1.70 | 10.8x | 6.5x | 170 / 184 | 80 / 128 | 79 / 130 |
| resnet152.tv2_in1k | 224x224 | 12.8 | 16.3 | 8.51 | 1.85 | 6.9x | 4.6x | 378 / 390 | 153 / 200 | 152 / 224 |
| resnet18.tv_in1k | 224x224 | 1.91 | 2.44 | 1.44 | 0.472 | 4.0x | 3.0x | 105 / 130 | 59 / 88 | 58 / 110 |
| resnet50.tv2_in1k | 224x224 | 4.74 | 5.84 | 3.15 | 0.826 | 5.7x | 3.8x | 180 / 192 | 89 / 142 | 116 / 162 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 4.79 | 6.53 | 3.72 | 0.804 | 6.0x | 4.6x | 47 / 48 | 39 / 48 | 55 / 76 |
| shvit_s1.in1k | 224x224 | 8.41 | 11.1 | 6.75 | 0.805 | 10.4x | 8.4x | 70 / 76 | 47 / 58 | 47 / 62 |
| shvit_s2.in1k | 224x224 | 8.37 | 11.1 | 6.76 | 0.949 | 8.8x | 7.1x | 99 / 120 | 56 / 88 | 57 / 88 |
| shvit_s3.in1k | 224x224 | 9.40 | 12.5 | 8.11 | 1.01 | 9.3x | 8.0x | 115 / 134 | 62 / 112 | 62 / 112 |
| shvit_s4.in1k | 256x256 | 11.5 | 14.6 | 9.31 | 1.78 | 6.4x | 5.2x | 129 / 146 | 67 / 118 | 68 / 102 |
| squeezenet1_1.tv_in1k | 224x224 | 1.72 | 2.63 | 1.78 | 0.376 | 4.6x | 4.7x | 10 / 30 | 7 / 28 | 8 / 12 |
| starnet_s1.in1k | 224x224 | 4.57 | 7.29 | 3.88 | 0.683 | 6.7x | 5.7x | 51 / 60 | 41 / 52 | 41 / 56 |
| starnet_s2.in1k | 224x224 | 3.64 | 5.63 | 3.02 | 0.578 | 6.3x | 5.2x | 55 / 64 | 43 / 54 | 43 / 58 |
| starnet_s3.in1k | 224x224 | 4.96 | 7.67 | 4.19 | 0.715 | 6.9x | 5.9x | 67 / 76 | 47 / 60 | 47 / 64 |
| starnet_s4.in1k | 224x224 | 6.84 | 11.3 | 5.63 | 0.912 | 7.5x | 6.2x | 77 / 86 | 50 / 66 | 51 / 72 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 224x224 | 15.0 | 19.1 | 10.8 | 2.16 | 6.9x | 5.0x | 538 / 550 | 211 / 230 | 218 / 238 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 224x224 | 5.95 | 8.27 | 4.75 | 0.934 | 6.4x | 5.1x | 106 / 122 | 64 / 86 | 60 / 84 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 224x224 | 6.05 | 8.37 | 4.83 | 1.19 | 5.1x | 4.1x | 168 / 188 | 87 / 106 | 83 / 106 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 384x384 | 5.84 | 7.95 | 4.54 | 1.80 | 3.2x | 2.5x | 270 / 290 | 157 / 184 | 132 / 178 |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 512x512 | 11.2 | 8.13 | 4.46 | 3.13 | 2.6x | 1.4x | 525 / 570 | 310 / 344 | 237 / 258 |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 224x224 | 5.92 | 8.54 | 4.73 | 0.858 | 6.9x | 5.5x | 71 / 90 | 52 / 62 | 48 / 78 |
| vgg16.tv_in1k | 224x224 | 3.62 | 4.09 | 1.99 | 1.50 | 2.4x | 1.3x | 825 / 846 | 313 / 372 | 332 / 388 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 224x224 | 6.33 | 5.48 | 3.02 | 1.66 | 3.3x | 1.8x | 531 / 548 | 205 / 390 | 200 / 248 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 224x224 | 76.9 | 25.8 | 13.1 | 12.9 | 2.0x | 1.0x | 2495 / 2564 | 879 / 1708 | 876 / 926 |
| wd-convnext-tagger-v3 | 448x448 | 15.3 | 9.60 | 5.59 | 4.46 | 2.2x | 1.3x | 259 / 292 | 260 / 292 | 238 / 276 |
| wd-eva02-large-tagger-v3 | 448x448 | 84.4 | 26.6 | 18.3 | 14.1 | 1.9x | 1.3x | 668 / 684 | 669 / 684 | 673 / 806 |
| wd-vit-large-tagger-v3 | 448x448 | 76.8 | 19.9 | 12.9 | 12.6 | 1.6x | 1.0x | 662 / 682 | 669 / 682 | 667 / 724 |
| wd-vit-tagger-v3 | 448x448 | 19.0 | 5.78 | 4.01 | 4.06 | 1.4x | 1.0x | 231 / 242 | 235 / 242 | 234 / 284 |

### Segmentation, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| bisenet-face-parsing | 512x512 | – | 5.94 | 2.41 | 1.59 | 3.7x | 1.5x | 69 / 94 | 51 / 72 | 61 / 112 |
| deeplabv3-r101-d8-ade20k | 1024x2048 | 108 | 54.8 | 48.0 | 20.5 | 2.7x | 2.3x | 2031 / 2162 | 450 / 492 | 356 / 414 |
| deeplabv3plus-r101-d8-ade20k | 1024x2048 | 104 | 53.4 | 47.1 | 17.9 | 3.0x | 2.6x | 1891 / 2024 | 410 / 452 | 355 / 436 |
| dinov2-b-linear-ade20k | 1024x2048 | 115 | 40.1 | 33.3 | 20.1 | 2.0x | 1.7x | 2063 / 2426 | 765 / 884 | 771 / 884 |
| dinov2-l-linear-ade20k | 1024x2048 | 332 | 95.7 | 83.9 | 52.8 | 1.8x | 1.6x | 3308 / 3726 | 1181 / 1244 | 1189 / 1244 |
| dinov2-s-linear-ade20k | 1024x2048 | 55.8 | 24.7 | 18.3 | 11.8 | 2.1x | 1.6x | 1694 / 2046 | 642 / 696 | 645 / 676 |
| eomt-l-dinov2-ade20k | 1024x2048 | 178 | 63.7 | 51.2 | 35.1 | 1.8x | 1.5x | 3827 / 4086 | 1357 / 1504 | 1344 / 1504 |
| eomt-l-dinov3-ade20k | 1024x2048 | 186 | 72.1 | 55.0 | 38.9 | 1.9x | 1.4x | 3822 / 4086 | 1355 / 1484 | 1342 / 1484 |
| fcn-r101-d8-ade20k | 1024x2048 | 94.2 | 47.0 | 40.7 | 16.5 | 2.8x | 2.5x | 1925 / 2102 | 414 / 472 | 288 / 374 |
| hrnet-w48-fcn-ade20k | 1024x2048 | 48.1 | 54.8 | 27.3 | 11.2 | 4.3x | 2.4x | 1915 / 2010 | 421 / 470 | 265 / 344 |
| mask2former-swinl-ade20k | 1024x2048 | 244 | 145 | 104 | 60.2 | 2.4x | 1.7x | 3572 / 3878 | 1052 / 1218 | 999 / 1238 |
| oneformer-swinl-ade20k | 1024x2048 | 307 | 205 | 145 | 106 | 1.9x | 1.4x | 7395 / 7792 | 1768 / 1994 | 1768 / 1914 |
| pspnet-r101-d8-ade20k | 1024x2048 | 92.5 | 47.2 | 48.7 | 24.9 | 1.9x | 2.0x | 1922 / 2222 | 414 / 472 | 335 / 394 |
| sapiens-seg-0.3b-goliath | 1024x2048 | 466 | 173 | 132 | 92.6 | 1.9x | 1.4x | 2520 / 2724 | 1643 / 1762 | 1643 / 1762 |
| sapiens-seg-1b-goliath | 1024x2048 | 1189 | 354 | 276 | 180 | 2.0x | 1.5x | 5721 / 5906 | 2894 / 3024 | 2894 / 3022 |
| segformer-b0-ade20k | 1024x2048 | 17.2 | 17.4 | 11.4 | 3.49 | 4.9x | 3.3x | 1590 / 1766 | 339 / 376 | 105 / 178 |
| segformer-b2-clothes | 512x512 | 18.9 | 14.1 | 8.13 | 3.61 | 3.9x | 2.3x | 425 / 488 | 156 / 168 | 133 / 170 |
| segformer-b5-ade20k | 1024x2048 | 132 | 72.0 | 38.9 | 25.4 | 2.8x | 1.5x | 2531 / 2620 | 444 / 490 | 294 / 432 |
| segformer-b5-face-parsing | 512x512 | 38.8 | 34.2 | 21.9 | 9.21 | 3.7x | 2.4x | 647 / 758 | 278 / 292 | 257 / 294 |
| segnext-mscan-b-ade20k | 1024x2048 | 44.8 | 33.9 | 28.3 | 12.1 | 2.8x | 2.3x | 1727 / 1838 | 376 / 424 | 192 / 226 |
| segnext-mscan-l-ade20k | 1024x2048 | 69.7 | 52.2 | 43.1 | 20.7 | 2.5x | 2.1x | 1849 / 1894 | 417 / 458 | 277 / 320 |
| segnext-mscan-s-ade20k | 1024x2048 | 28.6 | 23.3 | 18.5 | 8.04 | 2.9x | 2.3x | 1649 / 1700 | 350 / 402 | 164 / 202 |
| segnext-mscan-t-ade20k | 1024x2048 | 21.1 | 23.8 | 15.9 | 9.02 | 2.3x | 1.8x | 1593 / 1648 | 332 / 378 | 110 / 158 |

### Matting, synthetic, 1080x1920

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| ben2 | 201 | 115 | 105 | 1.9x | 1.1x | 2466 / 2772 | 1296 / 1376 | 1046 / 1118 |
| birefnet | 234 | 133 | 121 | 1.9x | 1.1x | 1666 / 1930 | 1165 / 1288 | 1145 / 1256 |
| birefnet-lite | 75.1 | 58.1 | 47.6 | 1.6x | 1.2x | 913 / 974 | 748 / 792 | 594 / 660 |
| inspyrenet | 241 | 107 | 107 | 2.3x | 1.0x | 3240 / 3262 | 1153 / 1468 | 1153 / 1494 |
| inspyrenet-fast | 71.1 | 45.4 | 15.8 | 4.5x | 2.9x | 810 / 836 | 363 / 448 | 363 / 454 |
| isnet | 38.5 | 21.0 | 12.1 | 3.2x | 1.7x | 993 / 1042 | 512 / 562 | 487 / 522 |
| isnet-general-use | 38.5 | 21.0 | 12.1 | 3.2x | 1.7x | 993 / 1042 | 512 / 562 | 487 / 522 |
| matanyone | 1220 | 966 | 911 | 1.3x | 1.1x | 2978 / 3208 | 2268 / 2726 | 2268 / 2706 |
| rmbg-2.0 | 149 | 133 | 121 | 1.2x | 1.1x | 1672 / 1990 | 1165 / 1288 | 1145 / 1296 |
| u2net | 15.9 | 9.77 | 5.18 | 3.1x | 1.9x | 476 / 502 | 240 / 278 | 161 / 236 |
| u2net-human-seg | 16.0 | 10.0 | 5.14 | 3.1x | 2.0x | 476 / 502 | 240 / 278 | 161 / 236 |
| u2netp | 14.8 | 15.1 | 2.92 | 5.1x | 5.2x | 280 / 314 | 139 / 180 | 72 / 124 |
| vitmatte-b | 257 | 180 | 146 | 1.8x | 1.2x | 5070 / 5128 | 891 / 1036 | 747 / 832 |
| vitmatte-s | 140 | 102 | 84.1 | 1.7x | 1.2x | 2536 / 2582 | 1401 / 1524 | 1305 / 1404 |

### Edges, synthetic, 1080x1920

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| anyline | 585 | 47.4 | 36.9 | 15.9x | 1.3x | 909 / 1002 | 654 / 722 | 654 / 702 |
| hed | 38.5 | 11.3 | 8.25 | 4.7x | 1.4x | 300 / 324 | 187 / 224 | 193 / 264 |
| lineart | 29.8 | 18.3 | 15.5 | 1.9x | 1.2x | 221 / 264 | 222 / 284 | 217 / 284 |
| lineart-anime | 25.2 | 8.36 | 6.50 | 3.9x | 1.3x | 407 / 448 | 241 / 284 | 217 / 244 |
| lineart-coarse | 30.4 | 18.3 | 15.4 | 2.0x | 1.2x | 221 / 264 | 222 / 284 | 217 / 284 |
| mlsd | 35.6 | 9.33 | 7.73 | 4.6x | 1.2x | 185 / 212 | 127 / 170 | 127 / 148 |
| pidinet | 35.9 | 13.1 | 12.3 | 2.9x | 1.1x | 339 / 368 | 328 / 372 | 328 / 392 |
| teed | 29.6 | 4.09 | 3.83 | 7.7x | 1.1x | 149 / 184 | 122 / 146 | 122 / 146 |

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| clipseg | – | 15.3 | 4.46 | 2.76 | 5.6x | 1.6x | 346 / 404 | 328 / 356 | 335 / 362 |
| efficientvit-sam-l0 | 26.9 | 26.3 | 11.1 | 4.55 | 5.8x | 2.4x | 211 / 232 | 158 / 190 | 164 / 236 |
| efficientvit-sam-l1 | 30.1 | 29.9 | 14.4 | 5.47 | 5.5x | 2.6x | 260 / 278 | 183 / 234 | 191 / 264 |
| efficientvit-sam-l2 | 34.4 | 34.4 | 17.8 | 6.54 | 5.3x | 2.7x | 313 / 344 | 209 / 240 | 219 / 292 |
| efficientvit-sam-xl0 | 46.9 | 28.2 | 19.0 | 9.75 | 2.9x | 1.9x | 654 / 670 | 433 / 506 | 401 / 492 |
| efficientvit-sam-xl1 | 72.6 | 40.9 | 28.9 | 15.3 | 2.7x | 1.9x | 984 / 1022 | 586 / 676 | 569 / 686 |
| insid3-vitl16 | 1649 | 713 | 255 | 157 | 4.5x | 1.6x | 2148 / 2226 | 986 / 1228 | 987 / 1228 |
| mobile-sam | 30.3 | 23.5 | 10.5 | 7.22 | 3.3x | 1.5x | 207 / 286 | 171 / 190 | 203 / 296 |
| sam-hq-vit-b | 172 | 103 | 48.3 | 35.3 | 2.9x | 1.4x | 1704 / 1756 | 375 / 422 | 420 / 544 |
| sam-hq-vit-h | 738 | 311 | 213 | 140 | 2.2x | 1.5x | 4266 / 4318 | 1488 / 1526 | 1528 / 1648 |
| sam-hq-vit-l | 412 | 202 | 114 | 79.0 | 2.6x | 1.4x | 2969 / 3018 | 834 / 862 | 877 / 1006 |
| sam-vit-b | 168 | 98.2 | 46.6 | 35.5 | 2.8x | 1.3x | 1769 / 1802 | 365 / 400 | 365 / 400 |
| sam-vit-h | 734 | 306 | 211 | 140 | 2.2x | 1.5x | 5295 / 5404 | 1472 / 1504 | 1467 / 1484 |
| sam-vit-l | 406 | 197 | 113 | 79.0 | 2.5x | 1.4x | 3411 / 3464 | 821 / 882 | 818 / 862 |
| sam2.1-hiera-base-plus | 107 | 47.2 | 31.9 | 25.4 | 1.9x | 1.3x | 823 / 870 | 400 / 444 | 414 / 464 |
| sam2.1-hiera-large | 242 | 86.4 | 68.0 | 49.1 | 1.8x | 1.4x | 1656 / 1724 | 730 / 768 | 748 / 808 |
| sam2.1-hiera-small | 74.8 | 41.8 | 18.8 | 15.6 | 2.7x | 1.2x | 618 / 662 | 318 / 366 | 330 / 386 |
| sam2.1-hiera-tiny | 66.3 | 38.0 | 16.6 | 13.9 | 2.7x | 1.2x | 577 / 636 | 304 / 382 | 316 / 362 |
| sam3 | 695 | 254 | 190 | 107 | 2.4x | 1.8x | 2052 / 2124 | 1910 / 1984 | 1900 / 2044 |

### Detection, synthetic, 800x1344

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| deim-l-coco | – | 36.1 | 16.3 | 6.50 | 5.6x | 2.5x | 264 / 322 | 155 / 172 | 151 / 216 |
| deim-m-coco | – | 31.4 | 13.8 | 5.08 | 6.2x | 2.7x | 210 / 250 | 129 / 142 | 129 / 184 |
| deim-n-coco | – | 24.6 | 10.4 | 2.46 | 10.0x | 4.2x | 93 / 124 | 92 / 114 | 94 / 118 |
| deim-s-coco | – | 26.4 | 11.3 | 3.73 | 7.1x | 3.0x | 154 / 180 | 108 / 128 | 108 / 170 |
| deim-x-coco | – | 41.3 | 18.7 | 9.53 | 4.3x | 2.0x | 462 / 502 | 230 / 258 | 231 / 298 |
| detr-r101-coco | 78.9 | 80.8 | 18.7 | 13.7 | 5.8x | 1.4x | 531 / 650 | 316 / 368 | 254 / 328 |
| detr-r50-coco | 55.5 | 62.0 | 12.4 | 9.93 | 5.6x | 1.2x | 459 / 558 | 280 / 332 | 218 / 272 |
| dfine-l-coco | 37.1 | 44.9 | 16.6 | 6.53 | 5.7x | 2.5x | 278 / 350 | 155 / 172 | 151 / 216 |
| dfine-m-coco | 30.8 | 38.2 | 13.7 | 5.07 | 6.1x | 2.7x | 209 / 256 | 129 / 142 | 129 / 184 |
| dfine-n-coco | 24.3 | 29.1 | 10.4 | 2.48 | 9.8x | 4.2x | 92 / 126 | 92 / 114 | 94 / 118 |
| dfine-s-coco | 25.7 | 31.9 | 11.4 | 3.75 | 6.8x | 3.0x | 152 / 184 | 108 / 128 | 108 / 170 |
| dfine-x-coco | 44.9 | 54.6 | 18.6 | 9.59 | 4.7x | 1.9x | 476 / 536 | 230 / 258 | 231 / 298 |
| faster-rcnn-r50-fpn-3x-coco | 83.6 | 41.4 | 22.8 | 19.5 | 2.1x | 1.2x | 626 / 708 | 390 / 490 | 391 / 450 |
| grounding-dino-base | – | 230 | 160 | 109 | 2.1x | 1.5x | 2895 / 3094 | 1525 / 1624 | 1156 / 1364 |
| grounding-dino-tiny | – | 199 | 138 | 86.9 | 2.3x | 1.6x | 2541 / 2716 | 1404 / 1468 | 1034 / 1188 |
| lw-detr-large-coco | – | 22.8 | 10.2 | 5.64 | 4.0x | 1.8x | 355 / 382 | 204 / 230 | 192 / 254 |
| lw-detr-medium-coco | – | 23.3 | 8.75 | 3.78 | 6.2x | 2.3x | 228 / 274 | 131 / 156 | 131 / 178 |
| lw-detr-small-coco | – | 19.9 | 8.82 | 2.80 | 7.1x | 3.1x | 129 / 166 | 97 / 116 | 99 / 140 |
| lw-detr-tiny-coco | – | 16.4 | 7.45 | 2.20 | 7.5x | 3.4x | 114 / 132 | 92 / 112 | 94 / 134 |
| lw-detr-xlarge-coco | – | 25.5 | 12.1 | 9.39 | 2.7x | 1.3x | 762 / 850 | 348 / 360 | 336 / 382 |
| mask-rcnn-r50-fpn-3x-coco | 85.4 | 43.5 | 24.6 | 21.0 | 2.1x | 1.2x | 604 / 650 | 352 / 440 | 353 / 400 |
| mm-grounding-dino-base | – | 232 | 160 | 109 | 2.1x | 1.5x | 2895 / 3094 | 1525 / 1624 | 1157 / 1366 |
| mm-grounding-dino-tiny | – | 202 | 138 | 87.2 | 2.3x | 1.6x | 2541 / 2716 | 1405 / 1470 | 1035 / 1188 |
| owlv2-b16 | – | 92.3 | 31.8 | 23.7 | 3.9x | 1.3x | 921 / 960 | 460 / 510 | 461 / 612 |
| owlv2-l14 | – | 212 | 154 | 110 | 1.9x | 1.4x | 2550 / 2606 | 1097 / 1144 | 1027 / 1184 |
| rf-detr-l-coco | – | 11.8 | 10.6 | 5.30 | 2.2x | 2.0x | 219 / 240 | 149 / 184 | 149 / 206 |
| rf-detr-m-coco | – | 11.8 | 10.6 | 3.86 | 3.0x | 2.8x | 140 / 174 | 136 / 156 | 136 / 178 |
| rf-detr-n-coco | – | 10.5 | 9.39 | 2.39 | 4.4x | 3.9x | 118 / 160 | 117 / 140 | 117 / 142 |
| rf-detr-s-coco | – | 11.2 | 10.0 | 3.33 | 3.3x | 3.0x | 189 / 238 | 128 / 148 | 128 / 172 |
| rf-detr-seg-2xl-coco | – | 25.6 | 26.0 | 16.0 | 1.6x | 1.6x | 287 / 352 | 207 / 228 | 209 / 312 |
| rf-detr-seg-l-coco | – | 15.3 | 14.1 | 6.60 | 2.3x | 2.1x | 459 / 492 | 150 / 168 | 150 / 208 |
| rf-detr-seg-m-coco | – | 14.4 | 13.3 | 5.31 | 2.7x | 2.5x | 381 / 424 | 141 / 180 | 141 / 202 |
| rf-detr-seg-n-coco | – | 13.0 | 11.7 | 3.68 | 3.5x | 3.2x | 255 / 298 | 125 / 148 | 125 / 170 |
| rf-detr-seg-s-coco | – | 13.0 | 12.0 | 4.28 | 3.0x | 2.8x | 327 / 382 | 132 / 154 | 132 / 178 |
| rf-detr-seg-xl-coco | – | 18.7 | 18.1 | 10.4 | 1.8x | 1.7x | 182 / 248 | 177 / 206 | 178 / 268 |
| rtdetrv2-l-coco | 30.3 | 38.4 | 12.4 | 6.53 | 4.6x | 1.9x | 354 / 412 | 180 / 200 | 180 / 262 |
| rtdetrv2-m-coco | 17.1 | 20.9 | 9.96 | 5.19 | 3.3x | 1.9x | 254 / 270 | 159 / 190 | 159 / 212 |
| rtdetrv2-ms-coco | 24.7 | 22.5 | 11.1 | 5.40 | 4.2x | 2.0x | 309 / 358 | 169 / 202 | 169 / 242 |
| rtdetrv2-s-coco | 14.1 | 16.8 | 9.60 | 4.10 | 3.4x | 2.3x | 192 / 220 | 137 / 166 | 137 / 208 |
| rtdetrv2-x-coco | 36.3 | 46.8 | 14.9 | 9.17 | 4.0x | 1.6x | 589 / 622 | 247 / 278 | 247 / 320 |
| rtmdet-l-coco | 19.5 | 20.4 | 9.59 | 5.52 | 3.5x | 1.7x | 332 / 356 | 170 / 210 | 149 / 286 |
| rtmdet-m-coco | 14.7 | 17.8 | 7.50 | 4.27 | 3.4x | 1.8x | 168 / 198 | 92 / 112 | 75 / 182 |
| rtmdet-m-person | 14.1 | 16.3 | 7.18 | 3.89 | 3.6x | 1.8x | 168 / 198 | 92 / 112 | 75 / 182 |
| rtmdet-nano-person | 12.9 | 16.0 | 7.39 | 1.82 | 7.1x | 4.0x | 12 / 30 | 9 / 28 | 9 / 52 |
| rtmdet-s-coco | 12.1 | 14.5 | 6.13 | 2.75 | 4.4x | 2.2x | 73 / 92 | 49 / 82 | 38 / 94 |
| rtmdet-tiny-coco | 10.7 | 13.8 | 5.74 | 2.46 | 4.3x | 2.3x | 48 / 64 | 36 / 56 | 29 / 66 |
| rtmdet-x-coco | 32.2 | 24.2 | 13.1 | 8.30 | 2.9x | 1.6x | 585 / 604 | 254 / 298 | 229 / 378 |
| yolox-l | – | 15.1 | 7.66 | 5.02 | 3.0x | 1.5x | 265 / 298 | 151 / 170 | 139 / 250 |
| yolox-m | – | 12.4 | 6.03 | 3.44 | 3.6x | 1.8x | 142 / 160 | 87 / 112 | 77 / 184 |
| yolox-s | – | 10.4 | 5.27 | 2.43 | 4.3x | 2.2x | 67 / 90 | 46 / 60 | 40 / 94 |
| yolox-tiny | – | 9.79 | 5.25 | 1.79 | 5.5x | 2.9x | 32 / 58 | 22 / 56 | 22 / 68 |
| yolox-x | – | 17.2 | 11.6 | 7.56 | 2.3x | 1.5x | 450 / 504 | 246 / 280 | 230 / 378 |

### Ocr, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| parseq | 32x128 | 15.9 | 19.6 | 3.62 | 3.62 | 4.4x | 1.0x | 126 / 144 | 79 / 100 | 79 / 100 |
| parseq-tiny | 32x128 | 14.6 | 18.0 | 2.10 | 2.10 | 6.9x | 1.0x | 56 / 58 | 44 / 54 | 44 / 54 |
| ppocrv5-mobile | 1080x1920 | 619 | 418 | 153 | 154 | 2.7x | 1.0x | 393 / 1005 | 289 / 448 | 500 / 1016 |
| ppocrv5-server | 1080x1920 | – | 762 (fp32) | 279 | 228 | 3.3x | 1.2x | 2355 / 2901 | 615 / 752 | 730 / 1648 |

### Pose, synthetic, 1080x1920

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| dwpose-l | 178 | 184 | 10.3 | 6.20 | 28.7x | 1.7x | 393 / 428 | 211 / 260 | 163 / 266 |
| dwpose-l-384 | 194 | 203 | 21.4 | 12.7 | 15.2x | 1.7x | 773 / 816 | 325 / 418 | 229 / 404 |
| dwpose-m | 169 | 164 | 7.31 | 4.86 | 33.8x | 1.5x | 756 / 792 | 153 / 196 | 133 / 200 |
| dwpose-s | 173 | 180 | 5.16 | 3.35 | 51.6x | 1.5x | 806 / 848 | 124 / 154 | 115 / 178 |
| dwpose-t | 160 | 174 | 7.84 | 2.52 | 63.6x | 3.1x | 832 / 858 | 104 / 132 | 106 / 152 |
| hrnet-w32 | 176 | 190 | 20.3 | 9.75 | 18.0x | 2.1x | 336 / 398 | 195 / 238 | 155 / 194 |
| hrnet-w32-384 | 216 | 190 | 39.2 | 19.7 | 9.7x | 2.0x | 653 / 760 | 341 / 414 | 250 / 294 |
| hrnet-w48 | 200 | 206 | 32.0 | 15.6 | 12.8x | 2.1x | 440 / 462 | 262 / 302 | 222 / 262 |
| hrnet-w48-384 | 276 | 245 | 70.7 | 34.2 | 7.1x | 2.1x | 1014 / 1082 | 408 / 458 | 317 / 360 |
| retinaface-mobilenet0.25 | 75.9 | 77.4 | 3.44 | 2.74 | 27.7x | 1.3x | 87 / 110 | 86 / 108 | 105 / 128 |
| retinaface-resnet50 | 81.0 | 79.4 | 24.3 | 11.7 | 6.8x | 2.1x | 371 / 490 | 405 / 504 | 379 / 464 |
| rf-detr-keypoint-preview | 29.4 | 19.3 | 17.2 | 8.98 | 2.1x | 1.9x | 186 / 244 | 176 / 202 | 176 / 224 |
| rtmo-l | 17.1 | 21.4 | 10.0 | 8.30 | 2.1x | 1.2x | 800 / 824 | 211 / 242 | 208 / 228 |
| rtmo-m | 14.3 | 17.0 | 7.44 | 6.85 | 2.1x | 1.1x | 422 / 462 | 168 / 192 | 166 / 212 |
| rtmo-s | 11.4 | 15.5 | 6.72 | 3.93 | 2.9x | 1.7x | 228 / 252 | 142 / 176 | 105 / 166 |
| rtmo-t | 11.3 | 13.3 | 7.82 | 2.96 | 3.8x | 2.6x | 118 / 122 | 122 / 140 | 82 / 140 |
| rtmpose-l | 161 | 176 | 9.04 | 5.70 | 28.3x | 1.6x | 1105 / 1154 | 200 / 256 | 152 / 244 |
| rtmpose-l-384 | 192 | 171 | 19.4 | 11.4 | 14.9x | 1.7x | 366 / 410 | 314 / 416 | 217 / 390 |
| rtmpose-l-halpe26 | 180 | 164 | 9.10 | 5.69 | 28.9x | 1.6x | 529 / 572 | 201 / 256 | 152 / 244 |
| rtmpose-l-halpe26-384 | 206 | 206 | 19.5 | 11.5 | 18.0x | 1.7x | 975 / 1038 | 315 / 416 | 218 / 410 |
| rtmpose-m | 179 | 180 | 6.36 | 4.50 | 39.7x | 1.4x | 316 / 364 | 145 / 192 | 125 / 200 |
| rtmpose-m-384 | 174 | 167 | 13.3 | 9.23 | 18.1x | 1.4x | 505 / 582 | 224 / 332 | 178 / 320 |
| rtmpose-m-ap10k | 196 | 186 | 8.02 | 5.42 | 34.4x | 1.5x | 757 / 814 | 166 / 214 | 139 / 200 |
| rtmpose-m-face6 | 198 | 207 | 8.75 | 5.77 | 34.4x | 1.5x | 652 / 698 | 173 / 216 | 145 / 222 |
| rtmpose-m-halpe26 | 179 | 181 | 6.42 | 4.53 | 39.7x | 1.4x | 283 / 318 | 146 / 192 | 125 / 180 |
| rtmpose-m-halpe26-384 | 192 | 193 | 13.4 | 9.28 | 20.7x | 1.4x | 690 / 764 | 225 / 332 | 179 / 320 |
| rtmpose-m-hand5 | 192 | 195 | 8.05 | 5.46 | 35.2x | 1.5x | 425 / 472 | 166 / 214 | 139 / 200 |
| rtmpose-s | 161 | 165 | 4.95 | 3.08 | 52.1x | 1.6x | 449 / 480 | 118 / 152 | 109 / 158 |
| rtmpose-s-face6 | 258 | 237 | 5.04 | 3.74 | 63.4x | 1.3x | 504 / 540 | 140 / 174 | 128 / 202 |
| rtmpose-s-halpe26 | 172 | 173 | 4.99 | 3.05 | 56.3x | 1.6x | 162 / 200 | 119 / 152 | 110 / 178 |
| rtmpose-t | 165 | 151 | 7.79 | 2.25 | 67.2x | 3.5x | 459 / 490 | 100 / 128 | 91 / 156 |
| rtmpose-t-face6 | 188 | 204 | 4.53 | 3.61 | 52.1x | 1.3x | 424 / 464 | 116 / 152 | 123 / 180 |
| rtmpose-t-halpe26 | 181 | 172 | 4.61 | 2.90 | 59.2x | 1.6x | 103 / 144 | 100 / 130 | 105 / 156 |
| rtmpose-x-halpe26-384 | 247 | 217 | 33.8 | 19.1 | 11.4x | 1.8x | 470 / 556 | 385 / 478 | 280 / 448 |
| rtmw-l | 213 | 203 | 15.3 | 9.99 | 20.3x | 1.5x | 865 / 908 | 256 / 308 | 217 / 312 |
| rtmw-l-384 | 274 | 275 | 31.8 | 20.6 | 13.3x | 1.5x | 1094 / 1144 | 371 / 468 | 276 / 458 |
| rtmw-m | 169 | 205 | 10.9 | 7.54 | 22.5x | 1.4x | 379 / 410 | 181 / 244 | 164 / 266 |
| rtmw-x | 218 | 207 | 24.0 | 14.4 | 14.3x | 1.7x | 604 / 636 | 335 / 390 | 289 / 392 |
| rtmw-x-384 | 306 | 291 | 51.7 | 49.1 | 5.9x | 1.1x | 1874 / 1940 | 466 / 552 | 496 / 612 |
| vitpose-base | 139 | 105 | 17.8 | 12.0 | 8.8x | 1.5x | 713 / 762 | 299 / 322 | 274 / 364 |
| vitpose-base-simple | 145 | 109 | 18.1 | 12.1 | 9.0x | 1.5x | 828 / 962 | 312 / 342 | 295 / 344 |
| vitpose-plus-base | 162 | 113 | 17.9 | 11.9 | 9.5x | 1.5x | 918 / 1002 | 299 / 322 | 274 / 344 |
| vitpose-plus-huge | 610 | 242 | 102 | 62.5 | 3.9x | 1.6x | 5732 / 5846 | 1382 / 1424 | 1351 / 1406 |
| vitpose-plus-large | 331 | 165 | 53.7 | 33.2 | 5.0x | 1.6x | 2873 / 3004 | 1014 / 1062 | 706 / 764 |
| vitpose-plus-small | 120 | 104 | 7.83 | 6.05 | 17.2x | 1.3x | 312 / 376 | 164 / 196 | 145 / 198 |
| yunet-n | – | 6.04 | 2.95 | 2.22 | 2.7x | 1.3x | 134 / 168 | 96 / 124 | 84 / 104 |
| yunet-s | – | 5.55 | 2.67 | 2.00 | 2.8x | 1.3x | 118 / 148 | 96 / 124 | 84 / 104 |

### Features, synthetic, 224x224

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| c-radio-v3-b | 6.31 | 5.75 | 3.04 | 1.36 | 4.2x | 2.2x | 589 / 626 | 217 / 246 | 214 / 248 |
| c-radio-v3-g | 58.0 | 29.5 | 14.6 | 10.7 | 2.8x | 1.4x | 6672 / 6804 | 2266 / 2308 | 2263 / 2310 |
| c-radio-v3-h | 34.5 | 17.0 | 8.90 | 7.43 | 2.3x | 1.2x | 3810 / 4110 | 1289 / 1308 | 1286 / 1308 |
| c-radio-v3-l | 16.8 | 10.7 | 5.53 | 3.28 | 3.3x | 1.7x | 1838 / 1878 | 648 / 686 | 645 / 706 |
| c-radio-v4-h | – | 17.1 | 8.91 | 7.40 | 2.3x | 1.2x | 6214 / 6250 | 1286 / 1308 | 1286 / 1308 |
| c-radio-v4-so400m | – | 11.5 | 6.22 | 4.14 | 2.8x | 1.5x | 4115 / 4150 | 861 / 906 | 861 / 908 |
| clip-b16-laion2b | 6.22 | 5.29 | 3.09 | 1.36 | 3.9x | 2.3x | 544 / 580 | 202 / 226 | 203 / 230 |
| clip-b16-openai | 6.28 | 5.65 | 3.35 | 1.36 | 4.2x | 2.5x | 544 / 580 | 204 / 226 | 203 / 250 |
| clip-h14-378-dfn5b | 47.9 | 19.5 | 11.4 | 8.24 | 2.4x | 1.4x | 3722 / 4038 | 1254 / 1288 | 1251 / 1312 |
| clip-h14-dfn5b | 47.5 | 19.4 | 11.4 | 8.54 | 2.3x | 1.3x | 3720 / 4038 | 1252 / 1268 | 1250 / 1292 |
| clip-h14-laion2b | 47.2 | 19.3 | 11.2 | 8.27 | 2.3x | 1.4x | 3720 / 4032 | 1249 / 1268 | 1250 / 1272 |
| clip-l14-336-openai | 22.1 | 10.8 | 6.16 | 4.76 | 2.3x | 1.3x | 1779 / 1814 | 625 / 646 | 622 / 668 |
| clip-l14-datacompxl | 21.6 | 10.1 | 5.99 | 4.76 | 2.1x | 1.3x | 1779 / 1812 | 621 / 666 | 622 / 670 |
| clip-l14-dfn2b | 22.0 | 10.6 | 6.16 | 4.76 | 2.2x | 1.3x | 1779 / 1812 | 623 / 666 | 622 / 690 |
| clip-l14-laion2b | 21.6 | 10.1 | 5.98 | 4.76 | 2.1x | 1.3x | 1779 / 1812 | 621 / 666 | 622 / 670 |
| clip-l14-openai | 22.0 | 10.8 | 6.16 | 4.76 | 2.3x | 1.3x | 1779 / 1812 | 623 / 666 | 622 / 690 |
| convnext_base.fb_in22k | 7.68 | 11.2 | 5.80 | 1.96 | 3.9x | 3.0x | 537 / 544 | 209 / 214 | 204 / 216 |
| convnext_large.fb_in22k | 10.5 | 11.5 | 5.82 | 3.23 | 3.3x | 1.8x | 1160 / 1180 | 420 / 430 | 413 / 434 |
| convnext_small.fb_in22k | 7.51 | 11.2 | 5.92 | 1.79 | 4.2x | 3.3x | 318 / 336 | 134 / 150 | 130 / 150 |
| convnext_tiny.fb_in22k | 3.89 | 5.91 | 3.26 | 1.10 | 3.5x | 3.0x | 194 / 214 | 93 / 108 | 88 / 110 |
| dinov2-vitb14 | 6.90 | 4.50 | 2.92 | 1.50 | 3.0x | 1.9x | 530 / 548 | 205 / 226 | 208 / 228 |
| dinov2-vitb14-reg | 7.00 | 4.30 | 2.94 | 1.52 | 2.8x | 1.9x | 530 / 548 | 207 / 226 | 208 / 228 |
| dinov2-vitg14 | 86.4 | 33.9 | 18.9 | 12.8 | 2.7x | 1.5x | 6542 / 6548 | 2216 / 2230 | 2222 / 2250 |
| dinov2-vitg14-reg | 86.6 | 34.2 | 18.7 | 12.7 | 2.7x | 1.5x | 6542 / 6568 | 2217 / 2228 | 2222 / 2250 |
| dinov2-vitl14 | 22.8 | 9.80 | 6.13 | 4.62 | 2.1x | 1.3x | 1777 / 1784 | 622 / 644 | 627 / 648 |
| dinov2-vitl14-reg | 22.5 | 10.7 | 6.38 | 4.60 | 2.3x | 1.4x | 1777 / 1786 | 624 / 646 | 627 / 648 |
| dinov2-vits14 | 3.30 | 4.40 | 2.95 | 0.827 | 4.0x | 3.6x | 160 / 166 | 79 / 98 | 80 / 122 |
| dinov2-vits14-reg | 4.11 | 5.59 | 2.98 | 0.832 | 4.9x | 3.6x | 160 / 168 | 80 / 98 | 80 / 102 |
| dinov3-convnext-base | 7.38 | 6.62 | 5.71 | 2.19 | 3.0x | 2.6x | 207 / 232 | 208 / 232 | 174 / 214 |
| dinov3-convnext-large | 10.5 | 6.72 | 5.45 | 3.61 | 1.9x | 1.5x | 418 / 430 | 419 / 430 | 409 / 452 |
| dinov3-convnext-small | 5.97 | 6.62 | 5.42 | 1.98 | 3.0x | 2.7x | 133 / 168 | 134 / 150 | 100 / 132 |
| dinov3-convnext-tiny | 2.96 | 3.60 | 3.04 | 1.22 | 2.4x | 2.5x | 91 / 128 | 93 / 108 | 59 / 110 |
| dinov3-vitb16 | 6.95 | 6.72 | 4.76 | 1.36 | 4.9x | 3.5x | 200 / 206 | 200 / 206 | 167 / 210 |
| dinov3-vith16plus | 48.0 | 16.0 | 14.1 | 8.85 | 1.8x | 1.6x | 1645 / 1816 | 1646 / 1666 | 1643 / 1670 |
| dinov3-vitl16 | 19.2 | 11.2 | 10.2 | 3.26 | 3.4x | 3.1x | 617 / 640 | 616 / 626 | 615 / 628 |
| dinov3-vits16 | 6.47 | 6.45 | 5.19 | 0.799 | 8.1x | 6.5x | 76 / 118 | 76 / 84 | 44 / 68 |
| pe-core-b16-224 | 7.14 | 9.35 | 4.97 | 1.47 | 4.9x | 3.4x | 584 / 620 | 218 / 266 | 218 / 268 |
| pe-core-g14-448 | 97.7 | 52.4 | 25.7 | 17.0 | 3.1x | 1.5x | 10848 / 11330 | 3640 / 3670 | 3640 / 3696 |
| pe-core-l14-336 | 23.3 | 17.7 | 9.09 | 4.97 | 3.6x | 1.8x | 1848 / 1888 | 650 / 668 | 648 / 672 |
| pe-core-s16-384 | 7.02 | 9.28 | 4.80 | 0.957 | 7.3x | 5.0x | 171 / 180 | 83 / 102 | 82 / 104 |
| pe-core-t16-384 | 6.74 | 9.04 | 4.68 | 0.683 | 9.9x | 6.9x | 68 / 70 | 48 / 56 | 53 / 82 |
| pe-spatial-b16-512 | 7.02 | 8.79 | 4.74 | 1.40 | 5.0x | 3.4x | 544 / 580 | 207 / 226 | 204 / 248 |
| pe-spatial-g14-448 | 97.9 | 52.7 | 25.7 | 16.9 | 3.1x | 1.5x | 10688 / 11178 | 3582 / 3592 | 3583 / 3676 |
| pe-spatial-l14-448 | 23.0 | 17.2 | 8.84 | 4.87 | 3.5x | 1.8x | 1780 / 1816 | 626 / 668 | 623 / 672 |
| pe-spatial-s16-512 | 6.70 | 8.79 | 4.51 | 0.922 | 7.3x | 4.9x | 161 / 178 | 81 / 100 | 78 / 104 |
| pe-spatial-t16-512 | 6.56 | 8.73 | 4.48 | 0.666 | 9.8x | 6.7x | 66 / 68 | 49 / 56 | 53 / 82 |
| siglip2-b16-256 | 6.29 | 6.11 | 3.28 | 1.43 | 4.3x | 2.3x | 583 / 620 | 216 / 248 | 216 / 248 |
| siglip2-b16-512 | 6.32 | 5.99 | 3.31 | 1.44 | 4.2x | 2.3x | 583 / 618 | 219 / 266 | 217 / 268 |
| siglip2-g16-256 | 59.8 | 30.9 | 14.7 | 10.7 | 2.9x | 1.4x | 6733 / 6854 | 2261 / 2288 | 2262 / 2312 |
| siglip2-g16-384 | 59.9 | 31.0 | 14.7 | 10.7 | 2.9x | 1.4x | 6735 / 6854 | 2262 / 2288 | 2263 / 2312 |
| siglip2-l16-256 | 17.4 | 11.1 | 5.71 | 3.36 | 3.3x | 1.7x | 1843 / 1890 | 643 / 668 | 643 / 670 |
| siglip2-l16-512 | 17.5 | 11.1 | 5.70 | 3.35 | 3.3x | 1.7x | 1846 / 1888 | 647 / 666 | 644 / 670 |
| siglip2-so400m14-224 | 23.9 | 12.4 | 6.38 | 4.32 | 2.9x | 1.5x | 2515 / 2632 | 858 / 868 | 858 / 892 |
| siglip2-so400m14-378 | 23.9 | 12.5 | 6.38 | 4.33 | 2.9x | 1.5x | 2516 / 2632 | 861 / 908 | 860 / 912 |
| siglip2-so400m16-256 | 22.8 | 12.6 | 6.42 | 4.21 | 3.0x | 1.5x | 2515 / 2636 | 857 / 868 | 857 / 890 |
| siglip2-so400m16-512 | 22.8 | 12.4 | 6.40 | 4.21 | 2.9x | 1.5x | 2518 / 2636 | 861 / 906 | 858 / 910 |
| tipsv2-b14 | 7.03 | 7.21 | 2.93 | 1.49 | 4.7x | 2.0x | 529 / 548 | 204 / 224 | 206 / 226 |
| tipsv2-l14 | 21.9 | 14.2 | 6.15 | 4.58 | 3.1x | 1.3x | 1776 / 1786 | 621 / 624 | 624 / 648 |

### Retrieval, synthetic, 480x640

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| boq-dinov2 | 7.44 | 5.73 | 2.84 | 2.6x | 2.0x | 231 / 256 | 231 / 254 | 227 / 256 |
| megaloc | 9.68 | 5.29 | 4.26 | 2.3x | 1.2x | 1347 / 1368 | 483 / 508 | 485 / 508 |

### Calibration, synthetic, 1080x1920

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| anycalib-pinhole | 38.4 | 29.4 | 21.5 | 1.8x | 1.4x | 1343 / 1352 | 705 / 742 | 739 / 812 |
| geocalib-pinhole | 106 | 81.2 | 81.8 | 1.3x | 1.0x | 502 / 558 | 320 / 378 | 277 / 338 |

### Multiview, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| anysplat | 336x518 | – | 741 | 508 | 369 | 2.0x | 1.4x | 8931 / 9394 | 4164 / 5384 | 4173 / 5548 |
| depthsplat-small-re10k | 336x518 | – | 126 | 122 | 95.1 | 1.3x | 1.3x | 5694 / 5892 | 956 / 1134 | 965 / 1314 |
| lingbot-map | 480x640 | – | 993 | 733 | 520 | 1.9x | 1.4x | 7141 / 7672 | 3910 / 5734 | 3919 / 5932 |
| mapanything-apache | 336x518 | 2875 | 1967 | 637 | 525 | 3.7x | 1.2x | 10227 / 10952 | 3252 / 3652 | 3252 / 3652 |
| mvsanywhere | 448x672 | 273 | 219 | 93.8 | 35.2 | 6.2x | 2.7x | 1504 / 1576 | 712 / 868 | 702 / 760 |
| mvsplat-re10k | 336x518 | – | 279 | 181 | 172 | 1.6x | 1.0x | 2889 / 4160 | 1239 / 1532 | 1239 / 1532 |
| patchmatchnet | 1536x1024 | 248 | 220 | 201 | 29.5 | 7.4x | 6.8x | 1228 / 1272 | 1039 / 1212 | 648 / 772 |
| pi3 | 336x518 | – | 1057 | 822 | 560 | 1.9x | 1.5x | 5926 / 6016 | 2209 / 2254 | 2217 / 2296 |
| resplat-base-re10k | 480x640 | – | 2521 | 6868 | 1169 | 2.2x | 5.9x | 10139 / 12684 | 4336 / 5596 | 4348 / 5546 |
| resplat-small-dl3dv | 480x640 | – | 493 | 1680 | 313 | 1.6x | 5.4x | 5148 / 5376 | 1103 / 1240 | 1089 / 1208 |
| vggt-1b | 336x518 | 2502 | 965 | 653 | 446 | 2.2x | 1.5x | 8688 / 9158 | 3573 / 3794 | 3580 / 3854 |
| vggt-omega-1b-512 | 336x518 | 2650 | 906 | 727 | 469 | 1.9x | 1.5x | 7495 / 7654 | 3269 / 3352 | 3270 / 3330 |

### Flow, synthetic, 1080x1920

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| flowformer-small-chairs | OOM | OOM | 4006 | 3951 | – | 1.0x | – | 8780 / 8960 | 8696 / 8822 |
| flowformer-small-sintel | OOM | OOM | 4003 | 3958 | – | 1.0x | – | 8780 / 8960 | 8696 / 8822 |
| flowformer-small-things | OOM | OOM | 4003 | 3950 | – | 1.0x | – | 8780 / 8960 | 8696 / 8822 |
| flowseek-t | 650 | 457 | 396 | 255 | 1.8x | 1.6x | 8693 / 9880 | 6123 / 6252 | 1100 / 1154 |
| flowseek-t-things | 650 | 458 | 395 | 254 | 1.8x | 1.6x | 8693 / 9820 | 6123 / 6312 | 1100 / 1154 |
| flowseek-t-things-notartan | 651 | 458 | 395 | 254 | 1.8x | 1.6x | 8693 / 9820 | 6123 / 6312 | 1100 / 1154 |
| gmflow-scale1-mixdata | 1246 | 750 | 448 | 446 | 1.7x | 1.0x | 8360 / 8448 | 1988 / 2142 | 1988 / 2142 |
| gmflow-scale1-things | 1245 | 750 | 448 | 446 | 1.7x | 1.0x | 8360 / 8468 | 1988 / 2142 | 1988 / 2142 |
| gmflow-scale2-mixdata | 2544 | 1608 | 915 | 915 | 1.8x | 1.0x | 8425 / 8572 | 2995 / 3184 | 2995 / 3204 |
| gmflow-scale2-regrefine6-kitti | 3255 | 2188 | 1383 | 1389 | 1.6x | 1.0x | 8537 / 8822 | 3005 / 3192 | 3005 / 3212 |
| gmflow-scale2-regrefine6-mixdata | 3254 | 2188 | 1382 | 1383 | 1.6x | 1.0x | 8537 / 8822 | 3005 / 3192 | 3005 / 3212 |
| gmflow-scale2-regrefine6-sintelft | 3255 | 2184 | 1381 | 1384 | 1.6x | 1.0x | 8537 / 8822 | 3005 / 3192 | 3005 / 3212 |
| gmflow-scale2-regrefine6-things | 3230 | 2175 | 1478 | 1481 | 1.5x | 1.0x | 8537 / 8822 | 4036 / 4696 | 4036 / 4696 |
| gmflow-scale2-sintel | 2541 | 1607 | 915 | 916 | 1.8x | 1.0x | 8425 / 8552 | 2995 / 3184 | 2995 / 3204 |
| gmflow-scale2-things | 2537 | 1599 | 914 | 916 | 1.7x | 1.0x | 8425 / 8552 | 2995 / 3184 | 2995 / 3204 |
| megaflow | – | 1588 | 1112 | 735 | 2.2x | 1.5x | 8230 / 8786 | 2713 / 3518 | 2724 / 3530 |
| memflow | – | 3116 | 965 | 865 | 3.6x | 1.1x | 15061 / 15594 | 6001 / 6294 | 968 / 1022 |
| memflow-kitti | – | 3247 | 964 | 865 | 3.8x | 1.1x | 15061 / 15594 | 5985 / 6102 | 968 / 1022 |
| memflow-sintel | – | 3214 | 964 | 865 | 3.7x | 1.1x | 15061 / 15634 | 5985 / 6102 | 968 / 1022 |
| memfof | – | 531 | 433 | 349 | 1.5x | 1.2x | 2003 / 2144 | 1911 / 2196 | 917 / 1016 |
| memfof-kitti | – | – | 2202 | 1565 | – | 1.4x | – | 14515 / 14918 | 3805 / 4038 |
| memfof-sintel | – | OOM | 2203 | 1565 | – | 1.4x | – | 14515 / 14918 | 3805 / 4038 |
| memfof-spring | – | 1035 | 433 | 349 | 3.0x | 1.2x | 2340 / 2676 | 1900 / 2196 | 917 / 1016 |
| neuflow-v2 | 171 | 75.6 | 72.0 | 59.8 | 1.3x | 1.2x | 4367 / 4556 | 2327 / 2398 | 490 / 636 |
| neuflow-v2-sintel | 171 | 75.7 | 71.3 | 58.9 | 1.3x | 1.2x | 4367 / 4576 | 2366 / 2398 | 790 / 896 |
| neuflow-v2-things | 172 | 75.8 | 71.4 | 58.9 | 1.3x | 1.2x | 4374 / 4720 | 2366 / 2398 | 779 / 896 |
| raft-large | 480 | 567 | 283 | 83.4 | 5.7x | 3.4x | 7164 / 10928 | 5721 / 5758 | 332 / 352 |
| raft-large-kitti | 480 | 567 | 284 | 84.2 | 5.7x | 3.4x | 7164 / 10928 | 5721 / 5758 | 332 / 352 |
| raft-large-things | 481 | 566 | 283 | 83.5 | 5.8x | 3.4x | 7164 / 10928 | 5721 / 5758 | 332 / 352 |
| raft-small | 196 | 395 | 141 | 36.5 | 5.4x | 3.9x | 7032 / 10888 | 5601 / 5646 | 196 / 204 |
| recover-cx | – | 179 | 107 | 89.8 | 2.0x | 1.2x | 1066 / 1168 | 733 / 892 | 733 / 892 |
| recover-mn | – | 103 | 45.5 | 41.8 | 2.5x | 1.1x | 921 / 1036 | 580 / 676 | 580 / 676 |
| recover-rn | – | 130 | 64.1 | 51.3 | 2.5x | 1.2x | 950 / 1046 | 590 / 672 | 590 / 672 |
| sea-raft-m | 504 | 367 | 269 | 73.4 | 5.0x | 3.7x | 7385 / 8570 | 5847 / 6570 | 425 / 454 |
| sea-raft-m-kitti | 504 | 367 | 270 | 73.3 | 5.0x | 3.7x | 7385 / 8570 | 6060 / 6180 | 425 / 454 |
| sea-raft-m-spring | 96.1 | 56.7 | 46.8 | 23.5 | 2.4x | 2.0x | 761 / 812 | 652 / 700 | 170 / 194 |
| sea-raft-m-things | 504 | 367 | 270 | 73.0 | 5.0x | 3.7x | 7385 / 8570 | 6060 / 6180 | 425 / 454 |
| sea-raft-s | 365 | 300 | 211 | 50.6 | 5.9x | 4.2x | 7323 / 8502 | 6019 / 6136 | 404 / 452 |
| sea-raft-s-kitti | 365 | 299 | 210 | 50.7 | 5.9x | 4.2x | 7323 / 8502 | 6019 / 6136 | 404 / 432 |
| sea-raft-s-spring | 67.1 | 42.4 | 32.8 | 16.2 | 2.6x | 2.0x | 699 / 764 | 611 / 656 | 150 / 172 |
| waft-a1 | – | 1533 | 741 | 519 | 3.0x | 1.4x | 5537 / 5916 | 1832 / 2196 | 1851 / 2264 |
| waft-a1-kitti | – | 1533 | 791 | 519 | 3.0x | 1.5x | 5537 / 5936 | 3553 / 4196 | 1851 / 2264 |
| waft-a1-sintel | – | 1533 | 791 | 519 | 3.0x | 1.5x | 5537 / 5916 | 3553 / 4076 | 1851 / 2264 |

### Flow, KITTI 2015, 375x1242

| model | time (ms): portable | kernels | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved): portable | kernels |
|---|---|---|---|---|---|
| raft-large | 45.8 | 21.0 | 2.2x | 511 / 738 | 85 / 112 |
| raft-large-kitti | 45.8 | 21.0 | 2.2x | 511 / 738 | 85 / 112 |
| raft-large-things | 45.8 | 21.1 | 2.2x | 511 / 738 | 85 / 112 |
| raft-small | 23.2 | 12.1 | 1.9x | 483 / 508 | 48 / 66 |

### Matching, synthetic, 768x1024

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| aliked-flexglue | – | 348 | 129 | 129 | 2.7x | 1.0x | 1801 / 1892 | 1775 / 1886 | 1814 / 2178 |
| aliked-lightglue | – | 41.8 | 40.6 | 35.1 | 1.2x | 1.2x | 1305 / 1364 | 1300 / 1404 | 1238 / 1324 |
| dedode-v2-b | – | 318 | 236 | 189 | 1.7x | 1.2x | 1734 / 1992 | 1160 / 1292 | 1197 / 1246 |
| dedode-v2-g | – | 666 | 379 | 332 | 2.0x | 1.1x | 2335 / 2472 | 1761 / 2114 | 1787 / 1984 |
| disk-lightglue | – | 117 | 116 | 49.2 | 2.4x | 2.4x | 1955 / 2412 | 1950 / 2400 | 736 / 778 |
| loftr-indoor | – | 242 | 110 | 86.0 | 2.8x | 1.3x | 1751 / 1910 | 740 / 944 | 740 / 806 |
| loftr-outdoor | – | 241 | 110 | 85.8 | 2.8x | 1.3x | 1739 / 1870 | 733 / 944 | 734 / 806 |
| loma-b | – | 1036 | 309 | 263 | 3.9x | 1.2x | 4176 / 4390 | 1767 / 1878 | 1217 / 1328 |
| loma-b128 | – | – | 150 | 116 | – | 1.3x | – | 878 / 984 | 625 / 684 |
| loma-b64 | – | – | 131 | 107 | – | 1.2x | – | 854 / 962 | 613 / 720 |
| loma-g | – | – | 359 | 310 | – | 1.2x | – | 2444 / 2548 | 1555 / 1668 |
| loma-l | – | – | 320 | 273 | – | 1.2x | – | 1903 / 2012 | 1285 / 1406 |
| loma-r | – | – | 309 | 262 | – | 1.2x | – | 1767 / 1878 | 1217 / 1328 |
| romav1 | – | 386 | 377 | 218 | 1.8x | 1.7x | 3887 / 4094 | 1597 / 1738 | 1597 / 1738 |
| romav2 | 1412 | 741 | 550 | 350 | 2.1x | 1.6x | 9881 / 10124 | 1989 / 2476 | 1890 / 2460 |
| xfeat | – | 8.89 | 7.17 | 15.2 | 0.6x | 0.5x | 128 / 174 | 122 / 154 | 313 / 352 |
| xfeat-lighterglue | – | 23.6 | 21.3 | 29.2 | 0.8x | 0.7x | 448 / 478 | 445 / 476 | 445 / 478 |

### Tracking, synthetic, 480x854

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| alltracker | 485 | 385 | 213 | 2.3x | 1.8x | 2100 / 2318 | 885 / 998 | 967 / 1068 |
| alltracker-tiny | 424 | 330 | 217 | 2.0x | 1.5x | 1995 / 2200 | 919 / 1178 | 978 / 1166 |
| bootstapir | 243 | 233 | 191 | 1.3x | 1.2x | 639 / 774 | 499 / 568 | 499 / 568 |
| cotracker3-offline | 244 | 162 | 154 | 1.6x | 1.0x | 2221 / 2480 | 755 / 882 | 757 / 882 |
| cotracker3-online | 328 | 184 | 176 | 1.9x | 1.0x | 1659 / 1886 | 755 / 906 | 757 / 888 |
| locotrack-b | 203 | 189 | 109 | 1.9x | 1.7x | 719 / 784 | 535 / 596 | 507 / 600 |
| locotrack-s | 193 | 180 | 151 | 1.3x | 1.2x | 706 / 812 | 759 / 842 | 596 / 642 |
| tapir | 187 | 182 | 141 | 1.3x | 1.3x | 498 / 574 | 463 / 548 | 463 / 548 |
| tapnext | 909 | 528 | 381 | 2.4x | 1.4x | 1721 / 1902 | 1220 / 1294 | 1205 / 1254 |
| tapnext-pp | 1493 | 529 | 382 | 3.9x | 1.4x | 1527 / 1718 | 1028 / 1094 | 1013 / 1074 |
| track-on-r | 2022 | 1150 | 1064 | 1.9x | 1.1x | 1801 / 1928 | 1030 / 1190 | 975 / 1156 |
| track-on2 | 1953 | 1159 | 1073 | 1.8x | 1.1x | 1403 / 1558 | 1030 / 1190 | 975 / 1158 |

### Interpolation, synthetic, 1080x1920

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| ema-vfi | 296 | 233 | 185 | 1.6x | 1.3x | 3940 / 4290 | 1659 / 2082 | 1659 / 2102 |
| ema-vfi-small | 121 | 89.4 | 84.4 | 1.4x | 1.1x | 2272 / 2384 | 1614 / 1792 | 271 / 1914 |
| ema-vfi-small-t | 114 | 89.4 | 84.3 | 1.4x | 1.1x | 2272 / 2384 | 1614 / 1792 | 271 / 1914 |
| ema-vfi-t | 287 | 233 | 207 | 1.4x | 1.1x | 3940 / 4290 | 2827 / 3184 | 368 / 3566 |
| film | 464 | 412 | 328 | 1.4x | 1.3x | 2953 / 3556 | 2397 / 2702 | 2397 / 2942 |
| gimm-vfi-r | OOM | 2528 | 1521 | – | 1.7x | – | 5872 / 6518 | 4511 / 5282 |
| gimm-vfi-r-p | OOM | 2527 | 1520 | – | 1.7x | – | 5872 / 6518 | 4511 / 5282 |
| rife-4.25 | 24.5 | 25.0 | 21.4 | 1.1x | 1.2x | 987 / 1154 | 755 / 894 | 848 / 916 |
| rife-4.25-lite | 21.3 | 21.1 | 18.3 | 1.2x | 1.2x | 1012 / 1214 | 754 / 854 | 834 / 896 |
| rife-4.26 | 24.2 | 25.0 | 21.4 | 1.1x | 1.2x | 1004 / 1154 | 570 / 754 | 663 / 776 |

### Restoration, synthetic, 512x512

| model | time (ms): original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| big-lama | 38.0 | 26.0 | 21.2 | 1.8x | 1.2x | 239 / 286 | 219 / 242 | 219 / 284 |
| codeformer | 85.7 | 48.6 | 40.4 | 2.1x | 1.2x | 959 / 1090 | 512 / 638 | 432 / 496 |
| dat-2-x4 | 4880 | 1156 | 1037 | 4.7x | 1.1x | 2436 / 2518 | 1420 / 2220 | 1399 / 2220 |
| dat-light-x4 | 2204 | 234 | 230 | 9.6x | 1.0x | 1606 / 1656 | 521 / 532 | 521 / 532 |
| dat-x4 | 9564 | 1386 | 1209 | 7.9x | 1.1x | 2443 / 2528 | 1427 / 2246 | 1406 / 2226 |
| drct-l-x4 | 14651 | 1567 | 1314 | 11.2x | 1.2x | 5574 / 6926 | 1675 / 1950 | 1603 / 1790 |
| drct-x4 | 7492 | 810 | 674 | 11.1x | 1.2x | 3359 / 3944 | 1608 / 1888 | 1560 / 1748 |
| gfpgan-v1.4 | 21.9 | 21.2 | 18.0 | 1.2x | 1.2x | 563 / 670 | 505 / 528 | 473 / 530 |
| hat-s-x4 | 2402 | 1072 | 956 | 2.5x | 1.1x | 5018 / 5644 | 1984 / 2148 | 1856 / 1888 |
| hat-x4 | 3038 | 1596 | 1427 | 2.1x | 1.1x | 5318 / 6106 | 2426 / 2590 | 2298 / 2470 |
| nafnet-gopro | 63.7 | 60.4 | 31.3 | 2.0x | 1.9x | 484 / 612 | 439 / 472 | 376 / 408 |
| nafnet-sidd | 71.6 | 61.1 | 22.5 | 3.2x | 2.7x | 608 / 730 | 607 / 650 | 502 / 536 |
| real-hat-gan-x4 | 3038 | 1428 | 1234 | 2.5x | 1.2x | 5318 / 6106 | 1430 / 2070 | 1312 / 1510 |
| realesr-general-x4v3 | 42.1 | 30.1 | 24.0 | 1.8x | 1.3x | 157 / 206 | 125 / 164 | 125 / 164 |
| realesrgan-x2plus | 128 | 97.0 | 80.2 | 1.6x | 1.2x | 592 / 656 | 333 / 394 | 333 / 394 |
| realesrgan-x4plus | 631 | 473 | 403 | 1.6x | 1.2x | 2266 / 2496 | 292 / 354 | 1220 / 1434 |
| realesrgan-x4plus-anime-6b | 205 | 155 | 132 | 1.6x | 1.2x | 2243 / 2472 | 1197 / 1430 | 1197 / 1410 |
| realplksr-nomoswebphoto-x4 | 267 | 182 | 130 | 2.1x | 1.4x | 244 / 320 | 179 / 216 | 179 / 216 |
| restormer-deblur | 234 | 183 | 132 | 1.8x | 1.4x | 1221 / 1418 | 724 / 740 | 718 / 740 |
| restormer-derain | 234 | 228 | 133 | 1.8x | 1.7x | 1221 / 1418 | 850 / 1000 | 892 / 980 |
| restormer-sidd | 218 | 181 | 133 | 1.6x | 1.4x | 1221 / 1416 | 850 / 920 | 892 / 980 |
| seedvr2-3b | 13150 | 2332 | 2356 | 5.6x | 1.0x | 7598 / 15268 | 4233 / 7262 | 3857 / 7778 |
| seedvr2-7b | OOM | 2748 | 2748 | – | 1.0x | – | 4495 / 8260 | 4495 / 8260 |
| span-x2 | 24.6 | 23.5 | 14.9 | 1.6x | 1.6x | 611 / 648 | 607 / 662 | 487 / 522 |
| span-x4 | 26.9 | 24.2 | 14.6 | 1.8x | 1.7x | 611 / 648 | 607 / 662 | 487 / 522 |
| swin2sr-x2 | 2565 | 909 | 721 | 3.6x | 1.3x | 1488 / 1578 | 1459 / 1554 | 1459 / 1636 |
| swin2sr-x4 | 2922 | 932 | 745 | 3.9x | 1.3x | 1488 / 2160 | 1463 / 2276 | 1459 / 1736 |
| swin2sr-x4-real | 2527 | 960 | 751 | 3.4x | 1.3x | 2321 / 2660 | 1459 / 1556 | 1459 / 1734 |
| swinir-l-x4-real | 3709 | 1218 | 896 | 4.1x | 1.4x | 2357 / 2746 | 1600 / 1794 | 1600 / 1794 |
| swinir-m-x2 | 3252 | 739 | 553 | 5.9x | 1.3x | 1500 / 1818 | 1353 / 1454 | 1353 / 1454 |
| swinir-m-x4 | 3497 | 767 | 576 | 6.1x | 1.3x | 1500 / 2058 | 1499 / 2054 | 1354 / 1456 |
| swinir-m-x4-real | 2462 | 1242 | 584 | 4.2x | 2.1x | 2321 / 2978 | 2387 / 3034 | 1353 / 1814 |

### Volume, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| nnfoundation-cnn | 200x200 | 501 | 274 | 248 | 232 | 1.2x | 1.1x | 3818 / 4628 | 2368 / 3330 | 2325 / 3330 |
| nnfoundation-vit | 192x192 | OOM | 1930 | 1848 | 1732 | 1.1x | 1.1x | 3930 / 4298 | 1669 / 1884 | 1669 / 1884 |
| nninteractive | 200x200 | 705 | 460 | 209 | 193 | 2.4x | 1.1x | 4076 / 4636 | 2423 / 2772 | 2423 / 2772 |
| totalsegmentator-total | 200x200 | 47779 | 45630 | 1977 | 1647 | 27.7x | 1.2x | 2640 / 2874 | 2017 / 2244 | 1219 / 1638 |
| totalsegmentator-total-3mm | 200x200 | 7304 | 7270 | 70.3 | 71.8 | 101.2x | 1.0x | 2010 / 2684 | 962 / 1050 | 913 / 1376 |

### Startup and first call

load: from the load call until the model is on the GPU; first call: the whole first inference call, including one normal inference (see the speed table for the warm time). Seconds, at each task's standard size (another size is shown in brackets).

| model | load (s): original | portable | kernels | first call (s): original | portable | kernels |
|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 0.15 (375x1242) | 0.16 | 0.20† | 0.70 (375x1242) | 0.17† | 0.09† |
| aanet-sceneflow | 0.12 (375x1242) | 0.19 | 0.27† | 0.07 (375x1242) | 0.14† | 0.10† |
| bridgedepth-rvc-pretrain | 5.87 (375x1242) | 1.17 | 1.12 | 1.36 (375x1242) | 1.06 | 1.04 |
| bridgedepth-sf | 4.86 (375x1242) | 1.53 | 0.72† | 0.12 (375x1242) | 0.44† | 0.35† |
| c-fast-foundationstereo | 0.17 (375x1242) | 0.24 | 0.72 (375x1242) | 0.17 (375x1242) | 0.33† | 0.55 (375x1242) |
| c-foundationstereo-s | 1.67 (375x1242) | 0.96 (375x1242) | 0.81 (375x1242) | 2.73 (375x1242) | 2.09 (375x1242) | 1.38 (375x1242) |
| cas-gwcnet-c-sceneflow | 0.48 (375x1242) | 1.29 | 0.18† | 0.41 (375x1242) | 0.33† | 0.23† |
| cas-gwcnet-gc-sceneflow | 0.39 | 1.10 | 0.81 | 0.64 | 0.64 | 0.40 |
| cfnet-sceneflow | 0.55 | 0.96 (375x1242) | 1.14 (375x1242) | 0.56 | 0.36 (375x1242) | 0.27 (375x1242) |
| coex-sceneflow | – | 0.27 (375x1242) | 0.57 (375x1242) | 0.63 (375x1242) | 0.70 (375x1242) | 0.48 (375x1242) |
| crestereo-eth3d | 0.05 | 0.16 | 0.03† | 1.50 | 1.21 | 0.23† |
| defom-stereo-vitl-sceneflow | 4.24 | 1.20 | 2.74 (375x1242) | 2.83 | 1.73 | 0.30 (375x1242) |
| dlnr-middlebury | 0.83 (375x1242) | 0.55 | 0.79 | 1.71 (375x1242) | 1.05 | 0.57 |
| dlnr-sceneflow | 0.83 (375x1242) | 0.63 | 0.26† | 1.71 (375x1242) | 0.40† | 0.21† |
| fast-acvnet-generalization | 0.24 (375x1242) | 0.27 (375x1242) | 0.59 (375x1242) | 1.73 (375x1242) | 0.79 (375x1242) | 0.62 (375x1242) |
| fast-acvnet-kitti12 | 0.24 (375x1242) | 0.27 (375x1242) | 0.68 (375x1242) | 1.73 (375x1242) | 0.76 (375x1242) | 0.62 (375x1242) |
| fast-acvnet-kitti15 | 0.24 (375x1242) | 0.18 (375x1242) | 0.57 (375x1242) | 1.73 (375x1242) | 0.83 (375x1242) | 0.62 (375x1242) |
| fast-acvnet-plus-generalization | 0.23 (375x1242) | 0.27 (375x1242) | 0.56 (375x1242) | 0.04 (375x1242) | 0.71 (375x1242) | 0.55 (375x1242) |
| fast-acvnet-plus-kitti12 | 0.23 (375x1242) | 0.27 (375x1242) | 0.57 (375x1242) | 0.04 (375x1242) | 0.70 (375x1242) | 0.55 (375x1242) |
| fast-acvnet-plus-kitti15 | 0.23 (375x1242) | 0.27 (375x1242) | 0.66 (375x1242) | 0.04 (375x1242) | 0.69 (375x1242) | 0.55 (375x1242) |
| fast-acvnet-plus-sceneflow | 0.23 (375x1242) | 0.27 (375x1242) | 0.57 (375x1242) | 0.04 (375x1242) | 0.71 (375x1242) | 0.55 (375x1242) |
| fast-acvnet-sceneflow | 0.24 (375x1242) | 0.27 (375x1242) | 0.57 (375x1242) | 1.73 (375x1242) | 0.76 (375x1242) | 0.63 (375x1242) |
| fast-foundationstereo | 0.16 (375x1242) | 0.25 | 0.60 (375x1242) | 0.14 (375x1242) | 0.33† | 0.49 (375x1242) |
| foundationstereo-vit-l | 10.86 (375x1242) | 1.46 | 1.40 | 2.56 (375x1242) | 3.78 | 2.21 |
| foundationstereo-vit-s | 1.44 (375x1242) | 1.03 | 0.33† | 1.06 (375x1242) | 2.51† | 1.12† |
| great-igev-eth3d | 0.44 (375x1242) | 0.32 | 0.70 (375x1242) | 4.60 (375x1242) | 1.10† | 1.14 (375x1242) |
| great-igev-kitti | 0.44 (375x1242) | 0.32 | 0.65 (375x1242) | 4.60 (375x1242) | 1.10† | 0.70 (375x1242) |
| great-igev-middlebury | 0.39 (375x1242) | – | 1.02 (375x1242) | 4.59 (375x1242) | 1.79 | 0.73 (375x1242) |
| great-igev-sceneflow | 0.44 (375x1242) | 0.32 | 0.64 (375x1242) | 4.60 (375x1242) | 1.10† | 0.70 (375x1242) |
| gwcnet-g-sceneflow | 0.26 (375x1242) | 0.40 | 0.14† | 0.46 (375x1242) | 0.34† | 0.24† |
| gwcnet-gc-sceneflow | 0.29 (375x1242) | 0.53 | 0.80 | 0.12 (375x1242) | 0.47 | 0.29 |
| hitnet-eth3d | 0.31 | 0.18 (375x1242) | 0.27 (375x1242) | 2.06 | 0.45 (375x1242) | 0.28 (375x1242) |
| hitnet-middlebury-d400 | 0.57 | 0.18 (375x1242) | 0.36 (375x1242) | 3.51 | 0.48 (375x1242) | 0.23 (375x1242) |
| igev-plusplus-eth3d | 0.42 (375x1242) | 0.69 (375x1242) | 0.68 (375x1242) | 2.33 (375x1242) | 0.84 (375x1242) | 0.57 (375x1242) |
| igev-plusplus-kitti12 | 0.42 (375x1242) | 0.99 (375x1242) | 0.59 (375x1242) | 2.33 (375x1242) | 0.82 (375x1242) | 0.53 (375x1242) |
| igev-plusplus-kitti15 | 0.42 (375x1242) | 0.67 (375x1242) | 0.60 (375x1242) | 2.33 (375x1242) | 0.82 (375x1242) | 0.53 (375x1242) |
| igev-plusplus-middlebury | 0.42 (375x1242) | 0.69 (375x1242) | 0.59 (375x1242) | 2.33 (375x1242) | 0.82 (375x1242) | 0.54 (375x1242) |
| igev-plusplus-sceneflow | 0.42 (375x1242) | 0.90 (375x1242) | 0.56 (375x1242) | 2.33 (375x1242) | 0.82 (375x1242) | 0.54 (375x1242) |
| igev-stereo-eth3d | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti12 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti15 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-middlebury | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-sceneflow | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| las2-h | 0.26 (375x1242) | 0.38 (375x1242) | 0.60 (375x1242) | 1.21 (375x1242) | 0.41 (375x1242) | 0.24 (375x1242) |
| las2-l | 0.39 (375x1242) | 0.53 (375x1242) | 0.66 (375x1242) | 0.04 (375x1242) | 0.33 (375x1242) | 0.18 (375x1242) |
| las2-m | 0.17 (375x1242) | 0.25 (375x1242) | 0.77 (375x1242) | 0.02 (375x1242) | 0.40 (375x1242) | 0.18 (375x1242) |
| las2-s | 0.11 (375x1242) | 0.66 (375x1242) | 0.60 (375x1242) | 0.02 (375x1242) | 0.34 (375x1242) | 0.17 (375x1242) |
| mocha-v1-sceneflow | 1.95 (375x1242) | 0.79 | 0.48† | 1.85 (375x1242) | 1.06† | 0.45† |
| mocha-v2-sceneflow | 2.37 (375x1242) | 1.30 | 0.45† | 0.40 (375x1242) | 1.03† | 0.45† |
| monster-mix-all | 5.85 (375x1242) | 2.02 (375x1242) | 2.69 (375x1242) | 1.26 (375x1242) | 0.70 (375x1242) | 0.55 (375x1242) |
| monster-plusplus-mix-all | 8.37 (375x1242) | 2.45 | 3.12 (375x1242) | 1.23 (375x1242) | 1.63† | 0.42 (375x1242) |
| monster-sceneflow | 5.85 (375x1242) | 2.61 (375x1242) | 2.31 (375x1242) | 1.26 (375x1242) | 0.74 (375x1242) | 0.47 (375x1242) |
| nmrf-sceneflow | 0.21 (375x1242) | 0.10 | 0.07† | 0.77 (375x1242) | 0.22† | 0.20† |
| psmnet-sceneflow | 0.20 (375x1242) | 0.30 (375x1242) | 0.59 (375x1242) | 0.53 (375x1242) | 0.32 (375x1242) | 0.16 (375x1242) |
| raft-stereo-eth3d | 0.27 (375x1242) | 0.25 | 0.55 (375x1242) | 1.45 (375x1242) | 0.75† | 0.10 (375x1242) |
| raft-stereo-middlebury | 0.27 (375x1242) | 0.25 | 0.73 (375x1242) | 1.45 (375x1242) | 0.75† | 0.10 (375x1242) |
| raft-stereo-sceneflow | 0.27 (375x1242) | 0.25 | 0.46 (375x1242) | 1.45 (375x1242) | 0.75† | 0.10 (375x1242) |
| rt-monster-plusplus-mix-all | 0.99 (375x1242) | 0.63 (375x1242) | 0.93 (375x1242) | 0.66 (375x1242) | 0.39 (375x1242) | 0.35 (375x1242) |
| s2m2-l | 1.38 | 2.61 (375x1242) | 0.58† | 1.01 | 0.76 (375x1242) | 0.37† |
| s2m2-m | 3.95 (375x1242) | 1.21 (375x1242) | 0.31† | 0.27 (375x1242) | 0.70 (375x1242) | 0.25† |
| s2m2-s | 0.22 (375x1242) | 0.22 | 0.20† | 0.14 (375x1242) | 0.35† | 0.11† |
| s2m2-xl | 4.50 (375x1242) | 2.64 (375x1242) | 0.90† | 0.53 (375x1242) | 0.89 (375x1242) | 0.68† |
| selective-igev-eth3d-finetune | 0.36 (375x1242) | 0.68 (375x1242) | 0.60 (375x1242) | 2.25 (375x1242) | 1.02 (375x1242) | 0.57 (375x1242) |
| selective-igev-eth3d-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti12 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti15 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-finetune | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-sceneflow | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| waft-stereo-b | 3.07 (375x1242) | 0.55 | 1.77 (375x1242) | 0.44 (375x1242) | 1.36 | 0.46 (375x1242) |
| waft-stereo-s | 1.60 (375x1242) | 0.58 | 1.47 (375x1242) | 0.14 (375x1242) | 0.69† | 0.32 (375x1242) |
| any2full-b | 3.73 | 6.38 | 0.74 | 0.34 | 3.34 | 0.26 |
| any2full-l | 9.41 | 1.45 | 1.50 | 0.47 | 0.30 | 0.28 |
| any2full-s | 2.55 | 0.31 | 0.37 | 0.74 | 0.36 | 0.30 |
| da3-giant-1.1 | 12.39 | 7.21 (518x770) | 5.38 (518x770) | 0.68 | 0.36 (518x770) | 0.37 (518x770) |
| da3-large-1.1 | 3.46 | 1.59 (518x770) | 2.16 (518x770) | 0.45 | 0.41 (518x770) | 0.26 (518x770) |
| da3metric-large | 2.70 | 4.84 (518x770) | 2.34 (518x770) | 0.28 | 0.24 (518x770) | 0.21 (518x770) |
| da3mono-large | 2.96 | 2.40 (504x504) | 2.23 (504x504) | 0.28 | 0.21 (504x504) | 0.17 (504x504) |
| da3nested-giant-large-1.1 | 28.67 | 18.24 (518x770) | 7.35 (518x770) | 1.63 | 0.46 (518x770) | 0.18 (518x770) |
| depth-anything-v2-b | 3.94 | 0.85 | 0.20† | 0.08 | 0.02 | 0.01 |
| depth-anything-v2-l | 5.66 | 2.39 | 0.91† | 0.15 | 0.08 | 0.06 |
| depth-anything-v2-metric-hypersim-b | 2.13 | 0.33 (518x770) | 1.13 (518x770) | 0.07 | 0.27 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-hypersim-l | 4.87 | 0.84 (518x770) | 1.73 (518x770) | 0.15 | 0.15 (518x770) | 0.07 (518x770) |
| depth-anything-v2-metric-hypersim-s | 1.54 | 0.19 (518x770) | 0.95 (518x770) | 0.17 | 0.23 (518x770) | 0.06 (518x770) |
| depth-anything-v2-metric-vkitti-b | 2.06 | 0.30 (518x770) | 1.13 (518x770) | 0.06 | 0.28 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-vkitti-l | 4.87 | 0.84 (518x770) | 1.65 (518x770) | 0.14 | 0.15 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-vkitti-s | 1.46 | 0.13 (518x770) | 0.96 (518x770) | 0.04 | 0.16 (518x770) | 0.05 (518x770) |
| depth-anything-v2-s | 0.71 | 0.11 | 0.08† | 0.38 | 0.03 | 0.01 |
| depth-pro | 17.34 | 6.20 | 4.11 | 0.74 | 0.52 | 0.27 |
| depthfm-v1 | – | 3.58 | 4.59 | – | 6.00 | 3.73 |
| distill-any-depth-b | 0.84 | 0.29 (518x770) | 1.13 (518x770) | 0.06 | 0.27 (518x770) | 0.04 (518x770) |
| distill-any-depth-l | 3.03 | 0.82 (518x770) | 1.68 (518x770) | 0.14 | 0.14 (518x770) | 0.06 (518x770) |
| distill-any-depth-s | 0.25 | 0.13 (518x770) | 0.96 (518x770) | 0.04 | 0.16 (518x770) | 0.04 (518x770) |
| lingbot-depth | 4.81 | 1.38 | 1.36 | 1.38 | 0.53 | 0.33 |
| lingbot-depth-dc | 7.54 | 1.29 | 1.33 | 1.36 | 0.53 | 0.34 |
| lotus-normal-d-v1-1 | 2.10 | 5.48 | 2.70 | 0.56 | 0.69 | 0.32 |
| lotus-normal-g-v1-1 | 10.43 | 4.37 | 3.49 | 1.15 | 0.59 | 0.38 |
| marigold-depth-v1-1 | 1.32 | 11.04 (518x770) | 2.49 (518x770) | 0.60 | 0.89 (518x770) | 0.50 (518x770) |
| marigold-normals-lcm-v0-1 | 6.02 | 4.11 | 8.22 | 2.86 | 0.62 | 0.47 |
| marigold-v2-depth | 174.65 (518x770) | 7.80 | 6.20 | 1.76 (518x770) | 5.09 | 3.89 |
| metric3d-vit-large | 9.96 | 2.39 (518x770) | 0.71 (518x770) | 0.88 | 0.42 (518x770) | 0.50 (518x770) |
| midas-dpt-beit-large-512 | 5.16 | 2.49 | 0.28† | 1.14 | 0.09 | 0.07 |
| moge-2-vitb-normal | 1.26 | 0.49 (518x770) | 1.24 (518x770) | 0.24 | 0.40 (518x770) | 0.28 (518x770) |
| moge-2-vitl-normal | 7.24 | 1.37 | 0.63† | 0.17 | 0.14 | 0.13 |
| moge-2-vits-normal | 0.88 | 1.18 (518x770) | 0.21 (518x770) | 1.50 | 0.30 (518x770) | 0.44 (518x770) |
| moge-3-vitl | 10.60 | 2.31 (518x770) | 1.16 (518x770) | 2.92 | 0.69 (518x770) | 0.74 (518x770) |
| omni-dc | 9.52 | 2.47 | 2.44 | 2.04 | 1.06 | 0.94 |
| pointdit-b-512 | 2.57 | 1.67 (518x770) | 1.74 (518x770) | 0.11 | 0.18 (518x770) | 0.15 (518x770) |
| pointdit-l-512 | 9.47 | 3.96 (518x770) | 3.91 (518x770) | 0.12 | 0.25 (518x770) | 0.23 (518x770) |
| prompt-depth-anything-l | 4.80 (756x1008) | 1.50 (756x1008) | 2.78 (756x1008) | 0.77 (756x1008) | 0.30 (756x1008) | 0.22 (756x1008) |
| prompt-depth-anything-s | 0.42 (756x1008) | 0.37 (756x1008) | 1.20 (756x1008) | 0.79 (756x1008) | 0.24 (756x1008) | 0.18 (756x1008) |
| stablenormal-turbo | 6.36 | 2.94 | 3.25 | 1.13 | 0.54 | 0.35 |
| unidepth-v2-vitl14 | 3.80 | 4.16 | 1.90 (518x770) | 0.38 | 0.15 | 0.30 (518x770) |
| unik3d-vitl | 7.28 | 4.16 | 3.77 (518x770) | 0.57 | 0.14 | 0.30 (518x770) |
| video-depth-anything-metric-s | 3.85 (1080x1920) | 0.19 (480x640) | 0.19 (480x640) | 17.09 (1080x1920) | 0.44 (480x640) | 0.50 (480x640) |
| video-depth-anything-s | 0.26 (1080x1920) | 0.98 (480x640) | 0.38 (480x640) | 14.04 (1080x1920) | 0.54 (480x640) | 0.37 (480x640) |
| zoedepth-nk | 8.35 | 0.46 | 0.33† | 1.53 | 0.49 | 0.12 |
| alexnet.tv_in1k | 2.51 | 1.30 | 1.40 | 0.81 | 0.02 | 0.06 |
| coatnet_0_rw_224.sw_in1k | 0.38 | 1.37 | 1.94 | 0.31 | 0.38 | 0.28 |
| coatnet_1_rw_224.sw_in1k | 0.47 | 1.44 | 2.45 | 0.15 | 0.37 | 0.28 |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 0.74 | 1.52 | 2.88 | 0.31 | 0.51 | 0.44 |
| coatnet_nano_rw_224.sw_in1k | 0.28 | 1.37 | 2.25 | 0.54 | 0.42 | 0.45 |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 0.47 | 1.49 | 2.82 | 0.02 | 0.37 | 0.29 |
| coatnet_rmlp_1_rw_224.sw_in1k | 0.47 | 1.47 | 2.46 | 0.02 | 0.37 | 0.29 |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 0.71 | 1.80 | 2.61 | 0.02 | 0.50 | 0.43 |
| coatnet_rmlp_2_rw_224.sw_in1k | 0.71 | 1.54 | 2.58 | 0.02 | 0.51 | 0.43 |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 0.74 (384x384) | 1.52 (384x384) | 2.41 (384x384) | 0.05 (384x384) | 0.43 (384x384) | 0.40 (384x384) |
| coatnet_rmlp_nano_rw_224.sw_in1k | 0.26 | 1.36 | 2.04 | 0.19 | 0.39 | 0.31 |
| convnext_base.fb_in1k | 0.95 | 1.91 | 1.99 | 0.01 | 0.25 | 0.27 |
| convnext_base.fb_in22k_ft_in1k | 0.89 | 2.07 | 1.97 | 0.01 | 0.25 | 0.26 |
| convnext_base.fb_in22k_ft_in1k_384 | 0.87 | 2.54 | 1.91 | 0.01 | 0.25 | 0.26 |
| convnext_large.fb_in1k | 1.89 | 2.79 | 2.58 | 0.02 | 0.25 | 0.24 |
| convnext_large.fb_in22k_ft_in1k | 1.91 | 2.55 | 2.62 | 0.01 | 0.25 | 0.28 |
| convnext_large.fb_in22k_ft_in1k_384 | 1.75 | 2.58 | 2.51 | 0.02 | 0.25 | 0.25 |
| convnext_small.fb_in1k | 0.65 | 1.81 | 1.59 | 0.01 | 0.22 | 0.19 |
| convnext_small.fb_in22k_ft_in1k | 0.52 | 1.67 | 1.93 | 0.01 | 0.22 | 0.21 |
| convnext_small.fb_in22k_ft_in1k_384 | 0.52 | 1.70 | 1.81 | 0.01 | 0.22 | 0.36 |
| convnext_tiny.fb_in1k | 0.41 | 1.48 | 1.50 | 0.49 | 0.24 | 0.24 |
| convnext_tiny.fb_in22k_ft_in1k | 0.35 | 1.49 | 1.45 | 0.01 | 0.24 | 0.24 |
| convnext_tiny.fb_in22k_ft_in1k_384 | 0.35 | 1.57 | 1.48 | 0.01 | 0.24 | 0.24 |
| convnextv2_base.fcmae_ft_in22k_in1k | 2.70 | 2.09 | 2.12 | 0.63 | 0.33 | 0.16 |
| davit_base.msft_in1k | 0.99 | 1.84 | 3.74 | 0.56 | 0.27 | 0.08 |
| davit_small.msft_in1k | 0.53 | 1.60 | 2.91 | 0.02 | 0.23 | 0.08 |
| davit_tiny.msft_in1k | 0.39 | 1.47 | 2.35 | 0.85 | 0.25 | 0.12 |
| deit_base_patch16_224.fb_in1k | 2.52 | 1.97 | 1.96 | 0.31 | 0.10 | 0.07 |
| densenet121.tv_in1k | 1.22 | 1.71 | 2.01 | 0.56 | 0.17 | 0.04 |
| dinov2_vitl14_lc | 5.80 | 3.34 | 3.94 | 0.32 | 0.02 | 0.10 |
| efficientformer_l1.snap_dist_in1k | 0.22 | 1.71 | 1.99 | 0.22 | 0.07 | 0.07 |
| efficientformer_l3.snap_dist_in1k | 0.37 | 1.86 | 2.28 | 0.03 | 0.08 | 0.06 |
| efficientformer_l7.snap_dist_in1k | 0.93 | 2.27 | 2.87 | 0.03 | 0.07 | 0.06 |
| efficientformerv2_l.snap_dist_in1k | 0.32 | 2.74 | 2.91 | 0.73 | 0.43 | 0.09 |
| efficientformerv2_s0.snap_dist_in1k | 0.17 | 1.88 | 2.03 | 1.76 | 0.50 | 0.08 |
| efficientformerv2_s1.snap_dist_in1k | 0.19 | 1.96 | 2.14 | 0.59 | 0.43 | 0.09 |
| efficientformerv2_s2.snap_dist_in1k | 0.24 | 2.68 | 2.53 | 0.86 | 0.44 | 0.09 |
| efficientnet_b0.tv_in1k | 1.25 | 0.17 | 1.53 | 2.86 | 0.38 | 0.06 |
| efficientnet_b4.tv_in1k | 0.41 | 1.83 | 2.11 | 1.72 | 0.95 | 0.09 |
| efficientnet_v2_s.tv_in1k | 1.48 | 1.78 | 2.11 | 0.97 | 0.42 | 0.06 |
| efficientvit_b2.r288_in1k | 1.84 | 2.31 | 2.43 | 1.89 | 0.73 | 0.07 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 6.98 | 4.54 | 5.89 | 0.39 | 0.17 | 0.17 |
| falconsai-nsfw-vit | 2.95 | 2.63 | 2.42 | 0.27 | 0.09 | 0.09 |
| fastvit_sa24.apple_dist_in1k | 0.31 (256x256) | 2.45 (256x256) | 2.77 (256x256) | 0.03 (256x256) | 0.82 (256x256) | 0.05 (256x256) |
| fastvit_t12.apple_dist_in1k | 0.14 (256x256) | 1.81 (256x256) | 2.01 (256x256) | 0.71 (256x256) | 0.41 (256x256) | 0.04 (256x256) |
| fastvit_t8.apple_dist_in1k | 0.21 (256x256) | 2.10 (256x256) | 1.83 (256x256) | 0.98 (256x256) | 0.85 (256x256) | 0.05 (256x256) |
| focalnet_base_lrf.ms_in1k | 0.91 | 1.89 | 3.85 | 0.03 | 0.62 | 0.14 |
| focalnet_base_srf.ms_in1k | 0.89 | 1.92 | 3.27 | 0.05 | 0.49 | 0.14 |
| focalnet_small_lrf.ms_in1k | 0.54 | 1.86 | 3.29 | 0.03 | 0.69 | 0.12 |
| focalnet_small_srf.ms_in1k | 0.54 | 1.73 | 3.08 | 0.02 | 0.49 | 0.12 |
| focalnet_tiny_lrf.ms_in1k | 0.36 | 2.56 | 2.86 | 0.15 | 0.77 | 0.14 |
| focalnet_tiny_srf.ms_in1k | 0.40 | 1.49 | 2.61 | 0.45 | 0.49 | 0.14 |
| hgnetv2_b0.ssld_stage2_ft_in1k | 0.19 | 1.67 | 1.74 | 1.36 | 0.73 | 0.05 |
| hgnetv2_b1.ssld_stage2_ft_in1k | 0.31 | 1.87 | 1.73 | 0.31 | 0.40 | 0.05 |
| hgnetv2_b2.ssld_stage2_ft_in1k | 0.23 | 1.79 | 1.89 | 0.59 | 0.39 | 0.05 |
| hgnetv2_b3.ssld_stage2_ft_in1k | 0.27 | 1.81 | 1.96 | 0.17 | 0.40 | 0.05 |
| hgnetv2_b4.ssld_stage2_ft_in1k | 0.32 | 1.74 | 2.11 | 0.30 | 0.39 | 0.05 |
| hgnetv2_b5.ssld_stage2_ft_in1k | 0.49 | 1.95 | 2.17 | 0.30 | 0.40 | 0.05 |
| hgnetv2_b6.ssld_stage2_ft_in1k | 0.86 | 2.21 | 2.65 | 0.45 | 0.46 | 0.05 |
| inception_v3.tv_in1k | 1.62 | 1.70 | 2.10 | 0.40 | 0.10 | 0.05 |
| mobilenet_v2.tv2_in1k | 1.23 | 0.11 | 1.51 | 1.65 | 0.03 | 0.05 |
| mobilenet_v3_large.tv2_in1k | 1.25 | 0.34 | 1.46 | 1.97 | 0.03 | 0.05 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 1.34 | 1.80 | 2.22 | 1.50 | 0.35 | 0.05 |
| mobilenetv4_conv_medium.e500_r256_in1k | 1.56 | 0.36 | 1.89 | 2.29 | 0.04 | 0.05 |
| regnety_160.tv2_in1k | 3.07 | 2.94 | 2.74 | 0.48 | 0.24 | 0.05 |
| repvit_m0_9.dist_300e_in1k | 0.20 | 1.65 | 2.01 | 2.43 | 0.28 | 0.08 |
| repvit_m0_9.dist_450e_in1k | 0.19 | 1.65 | 1.96 | 0.01 | 0.27 | 0.05 |
| repvit_m1_0.dist_300e_in1k | 0.20 | 1.64 | 2.00 | 1.44 | 0.28 | 0.05 |
| repvit_m1_0.dist_450e_in1k | 0.20 | 1.74 | 1.96 | 0.01 | 0.27 | 0.05 |
| repvit_m1_1.dist_300e_in1k | 0.20 | 1.86 | 1.97 | 1.36 | 0.28 | 0.06 |
| repvit_m1_1.dist_450e_in1k | 0.21 | 1.89 | 2.35 | 0.01 | 0.27 | 0.05 |
| repvit_m1_5.dist_300e_in1k | 0.26 | 1.84 | 2.63 | 0.02 | 0.27 | 0.05 |
| repvit_m1_5.dist_450e_in1k | 0.25 | 1.86 | 2.21 | 0.02 | 0.28 | 0.05 |
| repvit_m2_3.dist_300e_in1k | 0.33 | 2.00 | 2.41 | 1.54 | 0.28 | 0.05 |
| repvit_m2_3.dist_450e_in1k | 0.34 | 2.01 | 2.64 | 0.03 | 0.27 | 0.06 |
| resnet152.tv2_in1k | 2.34 | 2.01 | 2.33 | 0.24 | 0.09 | 0.04 |
| resnet18.tv_in1k | 0.27 | 1.25 | 1.40 | 0.00 | 0.06 | 0.04 |
| resnet50.tv2_in1k | 0.51 | 0.16 | 1.48 | 0.01 | 0.00 | 0.04 |
| shufflenet_v2_x1_0.tv_in1k | 1.13 | 1.44 | 1.30 | 1.15 | 0.36 | 0.11 |
| shvit_s1.in1k | 0.31 | 2.16 | 2.91 | 0.92 | 0.41 | 0.40 |
| shvit_s2.in1k | 0.20 | 2.32 | 2.31 | 0.30 | 0.48 | 0.45 |
| shvit_s3.in1k | 0.21 | 2.80 | 2.22 | 0.57 | 0.54 | 0.45 |
| shvit_s4.in1k | 0.23 (256x256) | 2.49 (256x256) | 2.38 (256x256) | 0.31 (256x256) | 0.44 (256x256) | 0.40 (256x256) |
| squeezenet1_1.tv_in1k | 1.09 | 1.19 | 1.33 | 0.37 | 0.06 | 0.08 |
| starnet_s1.in1k | 0.18 | 1.60 | 2.20 | 0.26 | 0.24 | 0.21 |
| starnet_s2.in1k | 0.17 | 1.62 | 1.86 | 0.02 | 0.24 | 0.06 |
| starnet_s3.in1k | 0.30 | 1.60 | 1.82 | 0.01 | 0.24 | 0.05 |
| starnet_s4.in1k | 0.21 | 1.66 | 2.00 | 0.01 | 0.24 | 0.05 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 2.71 | 2.03 | 2.28 | 0.41 | 0.14 | 0.09 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 0.56 | 1.83 | 1.44 | 0.60 | 0.41 | 0.11 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 0.37 | 1.98 | 1.45 | 0.99 | 0.40 | 0.10 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 0.69 (384x384) | 2.26 (384x384) | 1.81 (384x384) | 0.80 (384x384) | 0.38 (384x384) | 0.07 (384x384) |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 1.46 (512x512) | 3.05 (512x512) | 2.72 (512x512) | 0.16 (512x512) | 0.38 (512x512) | 0.05 (512x512) |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 0.26 | 2.02 | 1.37 | 1.28 | 0.26 | 0.12 |
| vgg16.tv_in1k | 3.22 | 1.74 | 1.90 | 0.30 | 0.00 | 0.06 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 2.49 | 0.73 | 2.05 | 0.28 | 0.00 | 0.07 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 7.52 | 7.29 | 5.82 | 0.35 | 0.07 | 0.09 |
| wd-convnext-tagger-v3 | 2.92 (448x448) | 2.34 (448x448) | 2.50 (448x448) | 0.48 (448x448) | 0.24 (448x448) | 0.07 (448x448) |
| wd-eva02-large-tagger-v3 | 4.82 (448x448) | 4.58 (448x448) | 4.22 (448x448) | 0.35 (448x448) | 0.15 (448x448) | 0.08 (448x448) |
| wd-vit-large-tagger-v3 | 4.74 (448x448) | 5.56 (448x448) | 4.50 (448x448) | 0.44 (448x448) | 0.11 (448x448) | 0.07 (448x448) |
| wd-vit-tagger-v3 | 2.33 (448x448) | 2.25 (448x448) | 2.47 (448x448) | 0.32 (448x448) | 0.09 (448x448) | 0.06 (448x448) |
| bisenet-face-parsing | 0.31 (512x512) | 0.20 (512x512) | 0.61 (512x512) | 0.41 (512x512) | 0.28 (512x512) | 0.08 (512x512) |
| deeplabv3-r101-d8-ade20k | 1.64 (512x512) | 2.28 (512x512) | 1.92 (512x512) | 0.03 (512x512) | 0.11 (512x512) | 0.03 (512x512) |
| deeplabv3plus-r101-d8-ade20k | 4.72 (512x512) | 1.11 (512x512) | 1.15 (512x512) | 0.66 (512x512) | 0.11 (512x512) | 0.02 (512x512) |
| dinov2-b-linear-ade20k | 12.01 (512x512) | 0.27 | 1.11 | 0.02 (512x512) | 0.10 | 0.07 |
| dinov2-l-linear-ade20k | 7.86 (512x512) | 0.81 | 1.83 | 0.38 (512x512) | 0.16 | 0.10 |
| dinov2-s-linear-ade20k | 4.02 (512x512) | 0.12 | 1.00 | 0.30 (512x512) | 0.09 | 0.06 |
| eomt-l-dinov2-ade20k | 4.80 (512x512) | 3.13 | 2.83 | 0.54 (512x512) | 0.05 | 0.25 |
| eomt-l-dinov3-ade20k | 1.83 (512x512) | 3.20 | 2.92 | 0.07 (512x512) | 0.08 | 0.23 |
| fcn-r101-d8-ade20k | 4.72 (512x512) | 1.42 (512x512) | 4.35 (512x512) | 0.39 (512x512) | 0.10 (512x512) | 0.02 (512x512) |
| hrnet-w48-fcn-ade20k | 1.48 (512x512) | 1.18 (512x512) | 1.52 (512x512) | 0.08 (512x512) | 0.11 (512x512) | 0.03 (512x512) |
| mask2former-swinl-ade20k | 4.86 (512x512) | 3.03 | 2.82 | 0.76 (512x512) | 0.13 | 0.32 |
| oneformer-swinl-ade20k | 5.08 (512x512) | 6.95 | 4.46 | 0.81 (512x512) | 0.50 | 0.49 |
| pspnet-r101-d8-ade20k | 1.35 (512x512) | 1.42 (512x512) | 1.59 | 0.09 (512x512) | 0.30 (512x512) | 0.03 |
| sapiens-seg-0.3b-goliath | 4.92 (512x512) | 1.22 | 0.96† | 1.02 (512x512) | 0.13† | 0.09† |
| sapiens-seg-1b-goliath | 11.33 (512x512) | 0.80 | 1.03 | 0.57 (512x512) | 0.47 | 0.29 |
| segformer-b0-ade20k | 4.83 (512x512) | 2.55 | 2.92 | 2.23 (512x512) | 0.08 | 0.01 |
| segformer-b2-clothes | – | 3.72 (512x512) | 3.65 (512x512) | – | 0.24 (512x512) | 0.06 (512x512) |
| segformer-b5-ade20k | 1.24 (512x512) | 3.09 | 3.43 | 0.50 (512x512) | 0.08 | 0.03 |
| segformer-b5-face-parsing | – | 4.16 (512x512) | 4.87 (512x512) | – | 0.23 (512x512) | 0.09 (512x512) |
| segnext-mscan-b-ade20k | 0.37 | 0.26 (512x512) | 1.34 (512x512) | 0.19 | 0.58 (512x512) | 0.18 (512x512) |
| segnext-mscan-l-ade20k | 0.57 | 0.57 (512x512) | 1.51 (512x512) | 0.06 | 0.41 (512x512) | 0.10 (512x512) |
| segnext-mscan-s-ade20k | 0.23 | 0.15 (512x512) | 1.29 (512x512) | 0.32 | 0.72 (512x512) | 0.19 (512x512) |
| segnext-mscan-t-ade20k | 0.19 | 0.13 (512x512) | 1.26 (512x512) | 0.34 | 0.65 (512x512) | 0.20 (512x512) |
| ben2 | 2.29 | 13.95 | 1.12 | 0.93 | 0.92 | 0.43 |
| birefnet | 3.60 | 0.46 | 13.58 | 0.39 | 0.56 | 0.37 |
| birefnet-lite | 8.04 | 0.67 | 0.20 | 1.10 | 0.42 | 0.47 |
| inspyrenet | 6.68 | 0.78 | 1.14 | 1.49 | 0.62 | 0.46 |
| inspyrenet-fast | 5.93 | 11.07 | 0.75 | 1.33 | 0.79 | 0.86 |
| isnet | 1.58 | 0.23 | 1.54 | 0.41 | 0.34 | 0.12 |
| isnet-general-use | 1.58 | 0.24 | 1.32 | 0.41 | 0.29 | 0.09 |
| matanyone | 3.39 | 0.40 | 0.35 | 2.37 | 1.87 | 1.56 |
| rmbg-2.0 | 6.42 | 0.94 | 0.64 | 0.94 | 0.49 | 0.47 |
| u2net | 0.64 | 0.49 | 1.37 | 0.40 | 0.21 | 0.11 |
| u2net-human-seg | 0.54 | 0.32 | 1.37 | 0.38 | 0.30 | 0.11 |
| u2netp | 0.18 | 0.11 | 1.28 | 0.47 | 0.26 | 0.11 |
| vitmatte-b | 0.21 | 0.52 | 1.32 | 0.56 | 0.35 | 0.25 |
| vitmatte-s | 2.85 | 0.16 | 1.30 | 0.72 | 0.41 | 0.20 |
| anyline | 0.11 | 0.17 | 0.15 (1024x1024) | 0.62 | 0.05 | 0.52 (1024x1024) |
| hed | 0.24 | 0.18 | 0.31 (1024x1024) | 0.06 | 0.01 | 0.12 (1024x1024) |
| lineart | 0.65 | 0.16 | 0.28 (1024x1024) | 0.05 | 0.02 | 0.10 (1024x1024) |
| lineart-anime | 0.74 | 0.82 | 0.54 (1024x1024) | 0.04 | 0.01 | 0.13 (1024x1024) |
| lineart-coarse | 0.16 | 0.16 | 0.52 (1024x1024) | 0.05 | 0.02 | 0.09 (1024x1024) |
| mlsd | 0.21 | 0.17 | 0.57 (1024x1024) | 0.06 | 0.01 | 0.49 (1024x1024) |
| pidinet | 0.28 | 0.16 | 0.17 (1024x1024) | 0.06 | 0.01 | 0.62 (1024x1024) |
| teed | 0.13 | 0.16 | 0.16 (1024x1024) | 0.08 | 0.01 | 0.44 (1024x1024) |
| clipseg | 3.05 | 4.93 | 4.54 | 0.02 | 0.29 | 0.24 |
| efficientvit-sam-l0 | 0.35 | 1.36 | 0.81 | 0.82 | 0.74 | 0.24 |
| efficientvit-sam-l1 | 0.31 | 0.86 | 0.90 | 0.03 | 0.67 | 0.23 |
| efficientvit-sam-l2 | 0.39 | 0.93 | 1.00 | 0.04 | 0.67 | 0.24 |
| efficientvit-sam-xl0 | 0.62 | 2.67 | 2.14 | 0.59 | 0.52 | 0.26 |
| efficientvit-sam-xl1 | 1.06 | 3.24 | 3.21 | 0.04 | 0.53 | 0.24 |
| insid3-vitl16 | 1.06 | 2.69 | 1.39 | 1.11 | 0.44 | 0.46 |
| mobile-sam | 0.21 | 0.71 | 0.82 | 1.01 | 0.42 | 0.31 |
| sam-hq-vit-b | 0.77 | 0.69 | 0.97 | 0.19 | 0.39 | 0.38 |
| sam-hq-vit-h | 3.92 | 2.27 | 2.01 | 0.32 | 0.52 | 0.48 |
| sam-hq-vit-l | 1.98 | 1.35 | 1.25 | 0.20 | 0.43 | 0.42 |
| sam-vit-b | 0.75 | 0.32 | 0.33 | 0.18 | 0.56 | 0.54 |
| sam-vit-h | 4.61 | 1.38 | 1.27 | 0.38 | 0.53 | 0.44 |
| sam-vit-l | 1.80 | 0.81 | 0.68 | 0.27 | 0.42 | 0.41 |
| sam2.1-hiera-base-plus | 0.74 | 0.80 (840x1297) | 1.15 (840x1297) | 0.14 | 0.03 (840x1297) | 0.03 (840x1297) |
| sam2.1-hiera-large | 1.95 | 1.46 (840x1297) | 2.07 (840x1297) | 0.18 | 0.07 (840x1297) | 0.05 (840x1297) |
| sam2.1-hiera-small | 0.48 | 0.36 | 0.13† | 0.14 | 0.02† | 0.02† |
| sam2.1-hiera-tiny | 0.41 | 0.25 | 0.11† | 0.15 | 0.02† | 0.01† |
| sam3 | 1.35 | 3.85 (840x1297) | 4.03 (840x1297) | 0.26 | 0.18 (840x1297) | 0.12 (840x1297) |
| deim-l-coco | 2.34 | 1.51 | 1.61 | 0.52 | 0.67 | 0.32 |
| deim-m-coco | 2.24 | 1.38 | 1.52 | 0.58 | 0.68 | 0.35 |
| deim-n-coco | 2.39 | 1.61 | 1.25 | 0.62 | 1.35 | 0.58 |
| deim-s-coco | 2.08 | 1.29 | 1.43 | 0.53 | 0.69 | 0.35 |
| deim-x-coco | 2.63 | 1.72 | 1.84 | 0.48 | 0.64 | 0.29 |
| detr-r101-coco | 0.95 | 0.81 | 0.22† | 0.06 | 0.02 | 0.01 |
| detr-r50-coco | 2.55 | 0.70 | 0.28† | 0.83 | 0.44 | 0.03 |
| dfine-l-coco | 3.68 | 1.72 | 1.56 | 1.41 | 0.67 | 0.32 |
| dfine-m-coco | 3.63 | 1.36 | 1.49 | 1.55 | 0.70 | 0.34 |
| dfine-n-coco | 5.24 | 1.25 | 1.21 | 4.07 | 0.85 | 0.62 |
| dfine-s-coco | 2.99 | 1.24 | 1.39 | 1.44 | 0.70 | 0.36 |
| dfine-x-coco | 3.88 | 1.65 | 1.83 | 1.45 | 0.64 | 0.29 |
| faster-rcnn-r50-fpn-3x-coco | 0.73 | 0.53 | 0.14† | 0.04 | 0.25 | 0.04 |
| grounding-dino-base | 4.16 | 3.29 | 4.22 | 0.99 | 0.53 | 0.45 |
| grounding-dino-tiny | 4.06 | 3.01 | 4.14 (320x320) | 1.18 | 0.51 | 0.42 (320x320) |
| lw-detr-large-coco | 3.82 | 3.20 | 3.04 | 0.72 | 0.36 | 0.34 |
| lw-detr-medium-coco | 3.91 | 3.64 | 3.65 | 0.64 | 0.34 | 0.33 |
| lw-detr-small-coco | 4.14 | 2.72 | 2.91 | 0.63 | 0.35 | 0.33 |
| lw-detr-tiny-coco | 3.80 | 2.69 | 2.91 | 0.64 | 0.35 | 0.33 |
| lw-detr-xlarge-coco | 4.72 | 3.32 | 4.25 | 0.71 | 0.34 | 0.38 |
| mask-rcnn-r50-fpn-3x-coco | 0.97 | 1.50 | 0.22† | 0.84 | 0.42 | 0.04 |
| mm-grounding-dino-base | 4.10 | 3.33 | 4.67 | 1.03 | 0.53 | 0.45 |
| mm-grounding-dino-tiny | 4.17 | 3.26 | 3.42 (320x320) | 0.99 | 0.52 | 0.43 (320x320) |
| owlv2-b16 | 8.22 | 3.41 | 3.32 | 0.74 | 0.26 | 0.24 |
| owlv2-l14 | 9.55 | 4.44 | 4.10 | 1.18 | 0.39 | 0.35 |
| rf-detr-l-coco | 1.10 | 6.42 | 3.36 | 0.02 | 0.47 | 0.47 |
| rf-detr-m-coco | 2.77 | 3.59 | 2.96 | 0.02 | 0.42 | 0.35 |
| rf-detr-n-coco | 5.93 | 2.59 | 2.67 | 0.56 | 0.35 | 0.35 |
| rf-detr-s-coco | 2.74 | 2.61 | 2.65 | 0.02 | 0.35 | 0.34 |
| rf-detr-seg-2xl-coco | 0.47 | 3.37 | 5.77 | 0.06 | 0.34 | 0.36 |
| rf-detr-seg-l-coco | 0.44 | 3.12 | 2.88 | 0.02 | 0.34 | 0.33 |
| rf-detr-seg-m-coco | 0.44 | 3.95 | 2.74 | 0.02 | 0.40 | 0.34 |
| rf-detr-seg-n-coco | 0.43 | 2.92 | 2.84 | 0.03 | 0.36 | 0.34 |
| rf-detr-seg-s-coco | 0.42 | 3.18 | 2.79 | 0.02 | 0.35 | 0.34 |
| rf-detr-seg-xl-coco | 0.61 | 3.20 | 2.92 | 0.04 | 0.33 | 0.33 |
| rtdetrv2-l-coco | 0.66 | 1.54 | 1.89 | 0.04 | 0.36 | 0.30 |
| rtdetrv2-m-coco | 0.45 | 1.45 | 1.64 | 0.02 | 0.37 | 0.33 |
| rtdetrv2-ms-coco | 0.52 | 1.51 | 1.80 | 0.06 | 0.34 | 0.31 |
| rtdetrv2-s-coco | 1.32 | 1.37 | 1.51 | 0.42 | 0.39 | 0.36 |
| rtdetrv2-x-coco | 1.13 | 1.79 | 2.14 | 0.08 | 0.32 | 0.28 |
| rtmdet-l-coco | 0.68 | 0.99 | 2.43 | 0.32 | 1.29 | 1.17 |
| rtmdet-m-coco | 0.46 | 0.72 | 1.66 | 0.31 | 1.30 | 1.20 |
| rtmdet-m-person | 0.51 | 0.31 | 1.00 | 0.03 | 0.39 | 0.25 |
| rtmdet-nano-person | 0.14 | 0.19 | 4.13 | 2.69 | 2.20 | 0.60 |
| rtmdet-s-coco | 0.27 | 0.28 | 1.81 | 0.58 | 1.58 | 0.21 |
| rtmdet-tiny-coco | 0.22 | 0.31 | 2.12 | 1.24 | 1.54 | 0.26 |
| rtmdet-x-coco | 1.00 | 1.05 | 2.39 | 0.61 | 1.26 | 1.16 |
| yolox-l | 0.65 | 0.72 | 1.05 | 0.03 | 1.11 | 1.17 |
| yolox-m | 0.32 | 0.41 | 0.84 | 0.03 | 0.23 | 0.18 |
| yolox-s | 0.22 | 0.15 | 1.55 | 0.03 | 1.35 | 0.30 |
| yolox-tiny | 1.55 | 0.18 | 1.58 | 0.52 | 1.31 | 0.31 |
| yolox-x | 0.76 | 0.92 | 1.33 | 0.08 | 0.14 | 1.14 |
| parseq | 0.53 (32x128) | 1.44 (32x128) | 0.91 (32x128) | 1.04 (32x128) | 0.26 (32x128) | 0.20 (32x128) |
| parseq-tiny | 0.34 (32x128) | 0.74 (32x128) | 0.76 (32x128) | 0.95 (32x128) | 0.20 (32x128) | 0.19 (32x128) |
| ppocrv5-mobile | 1.51 | 0.27 | 0.41 | 1.82 | 2.82 | 3.00 |
| ppocrv5-server | 2.07 | 0.56 | 1.07 | 2.03 | 1.87 | 1.11 |
| dwpose-l | 0.50 | 0.88 | 1.64 | 1.14 | 1.83 | 0.28 |
| dwpose-l-384 | 0.42 | 0.58 | 1.67 | 0.42 | 1.10 | 0.28 |
| dwpose-m | 0.26 | 0.24 | 1.51 | 0.60 | 1.10 | 0.29 |
| dwpose-s | 0.19 | 0.12 | 1.48 | 0.47 | 1.02 | 0.29 |
| dwpose-t | 0.16 | 0.17 | 0.69 | 0.33 | 1.24 | 0.29 |
| hrnet-w32 | 0.45 | 0.25 | 2.53 | 0.38 | 0.56 | 0.28 |
| hrnet-w32-384 | 0.42 | 0.25 | 2.04 | 0.23 | 0.57 | 0.25 |
| hrnet-w48 | 0.72 | 0.72 | 1.76 | 0.23 | 0.35 | 0.20 |
| hrnet-w48-384 | 0.53 | 0.39 | 2.16 | 0.24 | 0.35 | 0.21 |
| retinaface-mobilenet0.25 | 0.13 | 0.71 | 1.49 | 1.37 | 0.68 | 0.06 |
| retinaface-resnet50 | 0.43 | 0.71 | 1.55 | 0.31 | 0.11 | 0.10 |
| rf-detr-keypoint-preview | 1.49 | 2.66 | 2.70 | 0.08 | 0.39 | 0.38 |
| rtmo-l | 0.55 | 0.36 | 1.52 | 0.06 | 0.24 | 0.16 |
| rtmo-m | 0.37 | 0.22 | 1.43 | 0.08 | 0.43 | 0.24 |
| rtmo-s | 0.27 | 0.12 | 1.37 | 0.08 | 0.41 | 0.26 |
| rtmo-t | 0.26 | 0.11 | 1.38 | 0.32 | 0.41 | 0.28 |
| rtmpose-l | 0.38 | 0.94 | 1.60 | 0.18 | 0.86 | 0.29 |
| rtmpose-l-384 | 0.58 | 0.41 | 1.56 | 0.17 | 1.00 | 0.29 |
| rtmpose-l-halpe26 | 0.37 | 0.25 | 1.55 | 0.45 | 1.13 | 0.29 |
| rtmpose-l-halpe26-384 | 0.41 | 0.26 | 1.57 | 0.38 | 1.24 | 0.29 |
| rtmpose-m | 0.23 | 0.22 | 1.50 | 0.16 | 1.10 | 0.29 |
| rtmpose-m-384 | 0.23 | 0.20 | 1.48 | 0.35 | 1.25 | 0.29 |
| rtmpose-m-ap10k | 0.23 | 0.13 | 1.48 | 0.19 | 1.14 | 0.29 |
| rtmpose-m-face6 | 0.29 | 0.15 | 1.49 | 0.20 | 1.15 | 0.29 |
| rtmpose-m-halpe26 | 0.25 | 0.14 | 1.51 | 0.45 | 1.04 | 0.30 |
| rtmpose-m-halpe26-384 | 0.26 | 0.13 | 1.48 | 0.34 | 1.15 | 0.29 |
| rtmpose-m-hand5 | 0.23 | 0.13 | 1.49 | 0.22 | 1.13 | 0.29 |
| rtmpose-s | 0.16 | 0.10 | 1.40 | 0.16 | 1.03 | 0.29 |
| rtmpose-s-face6 | 0.31 | 0.17 | 0.69 | 0.35 | 1.11 | 0.30 |
| rtmpose-s-halpe26 | 0.19 | 0.07 | 1.46 | 0.61 | 0.94 | 0.30 |
| rtmpose-t | 0.13 | 0.10 | 1.42 | 0.15 | 0.88 | 0.29 |
| rtmpose-t-face6 | 0.18 | 0.16 | 0.71 | 0.34 | 1.29 | 0.29 |
| rtmpose-t-halpe26 | 0.18 | 0.16 | 0.73 | 0.71 | 0.87 | 0.29 |
| rtmpose-x-halpe26-384 | 0.89 | 0.41 | 1.76 | 0.94 | 1.17 | 0.28 |
| rtmw-l | 0.76 | 0.39 | 1.72 | 0.63 | 1.03 | 0.31 |
| rtmw-l-384 | 0.75 | 0.34 | 1.69 | 0.56 | 1.20 | 0.28 |
| rtmw-m | 0.54 | 0.32 | 1.02 | 1.21 | 0.90 | 0.30 |
| rtmw-x | 1.23 | 1.35 | 2.10 | 0.64 | 0.83 | 0.28 |
| rtmw-x-384 | 1.05 | 0.92 | 1.36 | 0.42 | 0.97 | 0.33 |
| vitpose-base | 0.16 | 0.73 | 0.71 | 0.33 | 0.37 | 0.35 |
| vitpose-base-simple | 0.15 | 0.40 | 1.04 | 0.17 | 0.41 | 0.41 |
| vitpose-plus-base | 0.19 | 0.65 | 1.10 | 0.12 | 0.34 | 0.34 |
| vitpose-plus-huge | 0.65 | 2.69 | 2.26 | 0.25 | 0.44 | 0.40 |
| vitpose-plus-large | 0.41 | 1.50 | 1.63 | 0.17 | 0.41 | 0.41 |
| vitpose-plus-small | 0.15 | 0.81 | 1.09 | 0.12 | 0.46 | 0.50 |
| yunet-n | 0.49 | 0.61 | 1.91 | 0.48 | 0.23 | 0.07 |
| yunet-s | 0.19 | 0.57 | 0.59 | 0.01 | 0.28 | 0.06 |
| c-radio-v3-b | – | – | – | 0.14 | 0.08 | 0.06 |
| c-radio-v3-g | – | – | – | 0.17 | 0.08 | 0.07 |
| c-radio-v3-h | – | – | – | 0.16 | 0.07 | 0.07 |
| c-radio-v3-l | – | – | – | 0.15 | 0.07 | 0.04 |
| c-radio-v4-h | 5.55 | 6.43 | 2.77 | 0.20 | 0.07 | 0.07 |
| c-radio-v4-so400m | 3.42 | 2.43 | 2.52 | 0.20 | 0.07 | 0.07 |
| clip-b16-laion2b | 13.69 | – | – | 0.16 | 0.06 | 0.05 |
| clip-b16-openai | 16.66 | – | – | 0.09 | 0.07 | 0.06 |
| clip-h14-378-dfn5b | 78.39 | – | – | 0.18 | 0.08 | 0.06 |
| clip-h14-dfn5b | 79.37 | – | – | 0.18 | 0.09 | 0.06 |
| clip-h14-laion2b | 78.69 | – | – | 0.18 | 0.07 | 0.07 |
| clip-l14-336-openai | 30.98 | – | – | 0.17 | 0.06 | 0.03 |
| clip-l14-datacompxl | 39.60 | – | – | 0.17 | 0.04 | 0.03 |
| clip-l14-dfn2b | 39.79 | – | – | 0.17 | 0.06 | 0.04 |
| clip-l14-laion2b | 39.40 | – | – | 0.17 | 0.04 | 0.03 |
| clip-l14-openai | 41.72 | – | – | 0.07 | 0.06 | 0.03 |
| convnext_base.fb_in22k | 0.90 | 2.18 | 2.05 | 0.02 | 0.24 | 0.24 |
| convnext_large.fb_in22k | 2.17 | 2.71 | 2.78 | 0.02 | 0.24 | 0.23 |
| convnext_small.fb_in22k | 0.65 | 1.93 | 1.73 | 0.01 | 0.21 | 0.18 |
| convnext_tiny.fb_in22k | 0.40 | 1.66 | 1.57 | 0.28 | 0.23 | 0.22 |
| dinov2-vitb14 | 1.51 | 0.27 | 0.19 | 0.01 | 0.20 | 0.19 |
| dinov2-vitb14-reg | 1.71 | 0.59 | 0.21 | 0.04 | 0.04 | 0.20 |
| dinov2-vitg14 | 19.29 | 4.37 | 2.22 | 0.05 | 0.10 | 0.16 |
| dinov2-vitg14-reg | 14.67 | 4.32 | 15.39 | 0.09 | 0.03 | 0.15 |
| dinov2-vitl14 | 7.06 | 2.30 | 2.31 | 0.01 | 0.09 | 0.12 |
| dinov2-vitl14-reg | 4.60 | 2.08 | 2.11 | 0.07 | 0.09 | 0.12 |
| dinov2-vits14 | 0.41 | 0.07 | 0.07 | 0.17 | 0.12 | 0.15 |
| dinov2-vits14-reg | 1.32 | 0.21 | 0.08 | 0.11 | 0.13 | 0.17 |
| dinov3-convnext-base | 0.42 | 0.43 | 0.19 | 0.01 | 0.27 | 0.10 |
| dinov3-convnext-large | 1.64 | 0.85 | 0.38 | 0.02 | 0.24 | 0.20 |
| dinov3-convnext-small | 0.27 | 0.25 | 0.11 | 0.01 | 0.33 | 0.07 |
| dinov3-convnext-tiny | 0.43 | 0.48 | 0.08 | 1.03 | 0.42 | 0.12 |
| dinov3-vitb16 | 0.68 | 0.38 | 0.16 | 0.01 | 0.20 | 0.29 |
| dinov3-vith16plus | 5.95 | 2.02 | 1.40 | 0.05 | 0.10 | 0.20 |
| dinov3-vitl16 | 2.37 | 0.54 | 0.52 | 0.02 | 0.18 | 0.25 |
| dinov3-vits16 | 0.48 | 0.07 | 0.06 | 0.95 | 0.27 | 0.26 |
| pe-core-b16-224 | 14.55 | – | – | 0.19 | 0.29 | 0.07 |
| pe-core-g14-448 | 232.98 | 27.16 | 15.98 | 0.28 | 0.56 | 0.12 |
| pe-core-l14-336 | 41.39 | – | – | 0.20 | 0.11 | 0.05 |
| pe-core-s16-384 | 5.70 | – | – | 0.18 | 0.10 | 0.04 |
| pe-core-t16-384 | 3.20 | – | – | 0.19 | 0.12 | 0.07 |
| pe-spatial-b16-512 | 13.65 | – | – | 0.16 | 0.10 | 0.07 |
| pe-spatial-g14-448 | 227.90 | 21.43 | 18.93 | 0.36 | 0.23 | 0.17 |
| pe-spatial-l14-448 | 44.48 | – | – | 0.17 | 0.09 | 0.06 |
| pe-spatial-s16-512 | 5.40 | – | – | 0.16 | 0.08 | 0.06 |
| pe-spatial-t16-512 | 3.09 | – | – | 0.16 | 0.10 | 0.08 |
| siglip2-b16-256 | 14.20 | – | – | 0.18 | 0.07 | 0.05 |
| siglip2-b16-512 | 14.17 | – | – | 0.18 | 0.07 | 0.05 |
| siglip2-g16-256 | 140.56 | 13.75 | 10.31 | 0.22 | 0.11 | 0.13 |
| siglip2-g16-384 | 139.99 | 9.63 | 9.84 | 0.22 | 0.09 | 0.08 |
| siglip2-l16-256 | 40.91 | – | – | 0.19 | 0.05 | 0.03 |
| siglip2-l16-512 | 41.54 | – | – | 0.18 | 0.05 | 0.03 |
| siglip2-so400m14-224 | 54.40 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m14-378 | 54.79 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m16-256 | 54.53 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m16-512 | 54.67 | – | – | 0.20 | 0.08 | 0.07 |
| tipsv2-b14 | 0.90 | 0.20 | 1.42 | 0.07 | 0.18 | 0.09 |
| tipsv2-l14 | 3.10 | 2.91 | 1.44 | 0.02 | 0.08 | 0.11 |
| boq-dinov2 | 1.22 | 0.24 | 1.09 | 0.04 | 0.23 | 0.13 |
| megaloc | 2.49 | 0.50 | 1.36 | 0.56 | 0.37 | 0.22 |
| anycalib-pinhole | 4.09 | 1.73 | 1.53 | 0.58 | 0.69 | 0.37 |
| geocalib-pinhole | 1.80 | 0.27 | 0.44 | 0.84 | 1.84 | 0.74 |
| anysplat | 8.32 (336x518) | 4.71 (336x518) | 4.04 (336x518) | 3.07 (336x518) | 0.89 (336x518) | 0.71 (336x518) |
| depthsplat-small-re10k | 1.83 (336x518) | 0.22 (336x518) | 1.47 (336x518) | 2.97 (336x518) | 0.97 (336x518) | 0.71 (336x518) |
| lingbot-map | 131.28 (480x640) | 6.58 (480x640) | 5.69 (480x640) | 4.64 (480x640) | 1.08 (480x640) | 0.84 (480x640) |
| mapanything-apache | 6.12 (336x518) | 1.53 (336x518) | 0.92 (336x518) | 2.62 (336x518) | 1.03 (336x518) | 0.88 (336x518) |
| mvsanywhere | 3.21 (448x672) | 0.93 (448x672) | 1.76 (448x672) | 1.04 (448x672) | 0.72 (448x672) | 0.48 (448x672) |
| mvsplat-re10k | 1.37 (336x518) | 0.15 (336x518) | 1.27 (336x518) | 2.82 (336x518) | 1.02 (336x518) | 0.69 (336x518) |
| patchmatchnet | 0.04 (1536x1024) | 0.16 (1536x1024) | 0.17 (1536x1024) | 0.92 (1536x1024) | 0.86 (1536x1024) | 0.56 (1536x1024) |
| pi3 | 10.64 (336x518) | 3.46 (336x518) | 3.78 (336x518) | 4.06 (336x518) | 1.28 (336x518) | 0.70 (336x518) |
| resplat-base-re10k | 2.26 (480x640) | 0.90 (480x640) | 3.30 (480x640) | 2.74 (480x640) | 7.76 (480x640) | 1.98 (480x640) |
| resplat-small-dl3dv | 16.80 (480x640) | 0.72 (480x640) | 3.88 (480x640) | 10.69 (480x640) | 2.51 (480x640) | 2.84 (480x640) |
| vggt-1b | 19.37 (336x518) | 6.78 (336x518) | 7.03 (336x518) | 1.29 (336x518) | 1.02 (336x518) | 0.75 (336x518) |
| vggt-omega-1b-512 | 16.31 (336x518) | 2.18 (336x518) | 5.24 (336x518) | 1.10 (336x518) | 0.95 (336x518) | 0.75 (336x518) |
| flowformer-small-chairs | 0.16 (375x1242) | 1.59 (375x1242) | 1.19 (375x1242) | 0.37 (375x1242) | 0.56 (375x1242) | 0.55 (375x1242) |
| flowformer-small-sintel | 0.15 (375x1242) | 1.56 (375x1242) | 1.29 (375x1242) | 0.30 (375x1242) | 0.58 (375x1242) | 0.55 (375x1242) |
| flowformer-small-things | 0.15 (375x1242) | 1.42 (375x1242) | 1.18 (375x1242) | 0.30 (375x1242) | 0.56 (375x1242) | 0.55 (375x1242) |
| flowseek-t | 0.87 | 0.18 (375x1242) | 0.28 (375x1242) | 0.63 | 0.46 (375x1242) | 0.38 (375x1242) |
| flowseek-t-things | 0.78 | 0.18 (375x1242) | 0.67 (375x1242) | 0.46 | 0.47 (375x1242) | 0.40 (375x1242) |
| flowseek-t-things-notartan | 0.81 | 0.18 (375x1242) | 0.37 (375x1242) | 0.46 | 0.46 (375x1242) | 0.38 (375x1242) |
| gmflow-scale1-mixdata | 0.13 | 0.16 | 0.16 (375x1242) | 0.76 | 0.56 | 0.57 (375x1242) |
| gmflow-scale1-things | 0.14 | 0.16 | 0.16 | 0.94 | 0.56 | 0.88 |
| gmflow-scale2-mixdata | 0.13 | 0.16 | 0.17 (375x1242) | 1.62 | 1.14 | 0.71 (375x1242) |
| gmflow-scale2-regrefine6-kitti | 0.15 | 0.46 | 0.17 (375x1242) | 2.20 | 1.68 | 0.89 (375x1242) |
| gmflow-scale2-regrefine6-mixdata | 0.14 | 0.25 | 0.17 (375x1242) | 2.20 | 1.68 | 0.89 (375x1242) |
| gmflow-scale2-regrefine6-sintelft | 0.15 | 0.26 | 0.36 (375x1242) | 2.37 | 1.68 | 0.72 (375x1242) |
| gmflow-scale2-regrefine6-things | 0.15 | 0.16 | – | 2.19 | 1.67 | – |
| gmflow-scale2-sintel | 0.13 | 0.16 | 0.16 (375x1242) | 1.62 | 1.14 | 0.72 (375x1242) |
| gmflow-scale2-things | 0.14 | 0.17 | 0.17 (375x1242) | 1.61 | 1.14 | 0.70 (375x1242) |
| megaflow | 9.56 | 1.96 (375x1242) | 7.10 (375x1242) | 6.77 | 0.76 (375x1242) | 0.67 (375x1242) |
| memflow | 0.16 | 0.06 | 1.18 | 3.82 | 0.97 | 0.88 |
| memflow-kitti | 0.14 | 0.14 | 0.97 | 3.06 | 0.98 | 0.88 |
| memflow-sintel | 0.14 | 0.10 | 0.95 | 3.06 | 0.98 | 0.88 |
| memfof | 2.06 | 0.22 (375x1242) | 1.84 (375x1242) | 1.29 | 0.41 (375x1242) | 0.29 (375x1242) |
| memfof-kitti | 2.59 (375x1242) | 0.21 (375x1242) | 1.07 (375x1242) | 0.96 (375x1242) | 0.65 (375x1242) | 0.56 (375x1242) |
| memfof-sintel | 1.30 (375x1242) | 0.69 (375x1242) | 2.49 | 0.90 (375x1242) | 0.57 (375x1242) | 1.99 |
| memfof-spring | 1.12 | 0.71 | 0.51 | 1.04 | 0.43 | 0.63 |
| neuflow-v2 | 0.20 | 0.26 | 0.96 (375x1242) | 0.22 | 0.07 | 0.18 (375x1242) |
| neuflow-v2-sintel | 0.17 | 1.29 (375x1242) | 0.95 (375x1242) | 0.08 | 0.81 (375x1242) | 0.18 (375x1242) |
| neuflow-v2-things | 0.17 | 0.38 | 0.26 | 0.08 | 0.07 | 0.06 |
| raft-large | 0.14 | 0.17 | 1.07 | 0.76 | 0.30 | 0.11 |
| raft-large-kitti | 0.14 | 0.17 | 1.06 | 0.58 | 0.30 | 0.11 |
| raft-large-things | 0.13 | 0.25 | 1.07 | 0.58 | 0.30 | 0.11 |
| raft-small | 0.09 | 0.16 | 1.06 | 0.43 | 0.17 | 0.07 |
| recover-cx | 3.28 | 0.19 (375x1242) | 0.47 (375x1242) | 0.74 | 0.46 (375x1242) | 0.35 (375x1242) |
| recover-mn | 2.03 | 0.26 (375x1242) | 0.48 (375x1242) | 0.53 | 1.08 (375x1242) | 1.05 (375x1242) |
| recover-rn | 2.44 | 0.18 (375x1242) | 0.60 (375x1242) | 0.51 | 0.49 (375x1242) | 0.36 (375x1242) |
| sea-raft-m | 0.53 | 0.29 | 1.15 | 0.37 | 0.27 | 0.07 |
| sea-raft-m-kitti | 0.67 | 1.02 | 1.14 | 0.37 | 0.27 | 0.07 |
| sea-raft-m-spring | 0.57 | 0.74 | 1.14 | 0.06 | 0.05 | 0.02 |
| sea-raft-m-things | 0.55 | 0.29 | 1.13 | 0.37 | 0.27 | 0.07 |
| sea-raft-s | 0.30 | 0.18 | 1.10 | 0.30 | 0.21 | 0.05 |
| sea-raft-s-kitti | 0.30 | 0.27 | 1.10 | 0.30 | 0.21 | 0.05 |
| sea-raft-s-spring | 0.31 | 0.34 | 1.11 | 0.04 | 0.04 | 0.02 |
| waft-a1 | 5.87 | 0.51 | 0.57 (375x1242) | 3.41 | 0.79 | 0.21 (375x1242) |
| waft-a1-kitti | 4.06 | 0.54 | 1.40 (375x1242) | 2.87 | 0.79 | 0.28 (375x1242) |
| waft-a1-sintel | 2.93 | 0.51 | 0.69 (375x1242) | 2.87 | 0.79 | 0.21 (375x1242) |
| aliked-flexglue | 4.50 | 0.92 | 0.97 | 1.66 | 1.20 | 1.06 |
| aliked-lightglue | 1.49 | 0.17 | 1.46 | 0.73 | 0.79 | 0.90 |
| dedode-v2-b | 3.73 | 0.67 | 0.55 | 1.10 | 0.75 | 0.84 |
| dedode-v2-g | 6.36 | 1.12 | 3.01 | 1.51 | 1.04 | 0.68 |
| disk-lightglue | 1.43 | 0.20 | 0.57 | 0.64 | 0.63 | 0.30 |
| loftr-indoor | 1.49 | 0.32 | 0.54 | 0.69 | 0.53 | 0.43 |
| loftr-outdoor | 0.35 | 0.24 | 0.67 | 0.24 | 0.58 | 0.37 |
| loma-b | 8.52 | 2.53 | 1.61 | 1.37 | 1.27 | 0.76 |
| loma-b128 | – | 0.52 | 0.85 | – | 0.64 | 0.66 |
| loma-b64 | – | 0.28 | 0.91 | – | 0.76 | 0.65 |
| loma-g | – | 2.36 | 2.19 | – | 0.80 | 0.68 |
| loma-l | – | 1.49 | 3.06 | – | 0.69 | 0.76 |
| loma-r | – | 2.80 | 1.93 | – | 0.73 | 0.81 |
| romav1 | 6.28 | 2.34 | 3.17 | 2.57 | 0.92 | 1.03 |
| romav2 | 5.45 | 0.85 | 0.96 | 2.09 | 0.91 | 0.62 |
| xfeat | 1.51 | 0.16 | 0.17 | 0.63 | 0.74 | 0.64 |
| xfeat-lighterglue | 1.46 | 0.25 | 0.16 | 0.69 | 0.68 | 0.73 |
| alltracker | 1.68 | 0.89 | 0.26 | 1.31 | 2.08 | 0.49 |
| alltracker-tiny | 1.22 | 0.06 | 0.96 | 1.40 | 0.73 | 0.54 |
| bootstapir | 0.60 | 0.32 | 1.14 | 0.90 | 0.80 | 0.66 |
| cotracker3-offline | 0.65 | 0.28 | 0.19 | 0.87 | 0.42 | 0.52 |
| cotracker3-online | 0.54 | 0.27 | 0.18 | 0.54 | 0.47 | 0.52 |
| locotrack-b | 0.79 | 0.35 | 0.18 | 1.06 | 0.72 | 0.87 |
| locotrack-s | 0.29 | 0.06 | 1.13 | 0.23 | 0.65 | 0.66 |
| tapir | 0.35 | 0.19 | 1.13 | 0.83 | 0.84 | 0.69 |
| tapnext | 16.92 | 2.37 | 3.01 | 2.17 | 0.67 | 0.54 |
| tapnext-pp | 4.83 | 1.34 | 1.28 | 1.80 | 0.77 | 0.67 |
| track-on-r | 0.88 | 0.24 | 1.11 | 2.91 | 1.99 | 1.72 |
| track-on2 | 3.77 | 1.16 | 0.63 | 3.63 | 1.96 | 1.76 |
| ema-vfi | 2.69 | 0.83 | 0.81 | 0.50 | 0.23 | 0.21 |
| ema-vfi-small | 0.36 | 0.27 | 0.56 | 0.18 | 0.09 | 0.09 |
| ema-vfi-small-t | 1.05 | 0.28 (480x854) | 0.56 (480x854) | 0.17 | 0.44 (480x854) | 0.35 (480x854) |
| ema-vfi-t | 2.77 | 0.73 (480x854) | 1.18 (480x854) | 0.49 | 0.40 (480x854) | 0.36 (480x854) |
| film | 0.29 | 0.43 | 0.65 | 0.95 | 0.42 | 0.33 |
| gimm-vfi-r | 2.05 (480x854) | 0.27 | 0.59 | 1.34 (480x854) | 2.62 | 1.63 |
| gimm-vfi-r-p | 2.96 (480x854) | 0.27 (480x854) | 0.58 (480x854) | 1.44 (480x854) | 1.00 (480x854) | 0.68 (480x854) |
| rife-4.25 | 1.42 | 0.16 | 1.50 | 0.27 | 0.45 | 0.20 |
| rife-4.25-lite | 0.18 | 0.24 | 0.60 | 0.12 | 0.27 | 0.20 |
| rife-4.26 | 0.18 | 0.16 | 0.29 | 0.05 | 0.33 | 0.21 |
| big-lama | 0.52 | 0.81 | 0.72 | 0.33 | 0.25 | 0.15 |
| codeformer | 1.24 | 0.69 | 0.65 | 0.41 | 0.23 | 0.23 |
| dat-2-x4 | 2.04 | 0.71 | 0.72 | 5.52 | 2.14 | 2.01 |
| dat-light-x4 | 3.48 | 0.21 | 1.25 | 2.72 | 1.30 | 1.15 |
| dat-x4 | 2.20 | 2.99 | 0.71 | 6.13 | 2.31 | 2.15 |
| drct-l-x4 | 2.09 | 0.79 | 1.84 | 15.38 | 1.70 | 1.45 |
| drct-x4 | 1.88 | 0.39 | 1.32 | 8.66 | 1.02 | 0.77 |
| gfpgan-v1.4 | 0.86 | 1.04 | 0.78 | 0.33 | 0.25 | 0.07 |
| hat-s-x4 | 3.14 | 0.35 | 1.58 | 4.41 | 1.27 | 1.07 |
| hat-x4 | 1.75 | 0.47 | 1.36 | 5.42 | 1.81 | 1.57 |
| nafnet-gopro | 1.50 | 0.35 | 0.64 | 0.39 | 0.28 | 0.23 |
| nafnet-sidd | 1.16 | 0.79 | 0.70 | 0.29 | 0.37 | 0.24 |
| real-hat-gan-x4 | 1.96 | 1.24 | 1.49 | 5.41 | 1.77 | 1.56 |
| realesr-general-x4v3 | 0.16 | 0.40 | 0.31 | 0.20 | 0.29 | 0.05 |
| realesrgan-x2plus | 1.54 | 0.25 | 1.26 | 0.42 | 0.26 | 0.11 |
| realesrgan-x4plus | 1.62 | 0.20 | 2.43 | 1.64 | 0.62 | 0.43 |
| realesrgan-x4plus-anime-6b | 1.30 | 0.10 | 1.30 | 0.50 | 0.23 | 0.15 |
| realplksr-nomoswebphoto-x4 | 0.21 | 0.18 | 0.56 | 0.76 | 0.41 | 0.25 |
| restormer-deblur | 0.36 | 0.28 | 0.60 | 0.90 | 0.71 | 0.53 |
| restormer-derain | 0.34 | 0.29 | 0.58 | 0.90 | 0.72 | 0.53 |
| restormer-sidd | 0.34 | 0.29 | 0.59 | 0.89 | 0.89 | 0.53 |
| seedvr2-3b | 96.20 | 2.65 | 3.27 | 13.89 | 2.45 | 2.46 |
| seedvr2-7b | – | 222.77 | 118.43 | – | 3.17 | 3.18 |
| span-x2 | 0.23 | 0.10 | 2.88 | 0.32 | 0.26 | 0.10 |
| span-x4 | 0.45 | 0.10 | 1.11 | 0.58 | 0.26 | 0.05 |
| swin2sr-x2 | 1.81 | 0.49 | 0.62 | 4.12 | 1.15 | 0.86 |
| swin2sr-x4 | 1.63 | 0.48 | 0.57 | 3.58 | 1.16 | 1.02 |
| swin2sr-x4-real | 1.89 | 0.47 | 0.57 | 3.75 | 1.18 | 0.88 |
| swinir-l-x4-real | 2.15 | 0.70 | 1.46 | 5.95 | 2.19 | 1.10 |
| swinir-m-x2 | 2.95 | 0.67 | 0.43 | 4.49 | 0.85 | 0.63 |
| swinir-m-x4 | 2.07 | 0.44 | 1.28 | 3.27 | 1.36 | 0.65 |
| swinir-m-x4-real | 2.06 | 0.40 | 1.31 | 3.33 | 1.39 | 0.70 |
| nnfoundation-cnn | 1.18 | 0.56 | 0.39 | 0.69 | 0.27 | 0.29 |
| nnfoundation-vit | 9.54 (192x192) | 1.24 (192x192) | 1.27 (192x192) | 2.37 (192x192) | 1.87 (192x192) | 1.76 (192x192) |
| nninteractive | 3.23 | 1.09 | 0.65 | 1.04 | 0.48 | 0.40 |
| totalsegmentator-total | 0.07 | 0.77 | 0.77 | 54.51 | 2.10 | 1.88 |
| totalsegmentator-total-3mm | 0.06 | 1.18 | 1.19 | 12.48 | 0.26 | 0.21 |

† measured in the speed sweep after the portable load (or an earlier size) in the same process: file cache and kernels are warm, so the time is lower than a fresh start.
<!-- tables:end -->

## torch.compile and TensorRT

Portable with `model.compile()` (Inductor, default mode) and the fp16 TensorRT engine of the portable model
(`export_tensorrt`), one variant per family at the task's standard size, next to portable and `kernels`
([export](export.md) has the methods). The compile time is the cold first call (empty Inductor cache); the engine
runs `module()` on synthetic inputs of the shape the timed call gives the network (classifier crop, all views of a
scene; "only" marks engines of part of the call) and its memory is the engine's device memory plus torch's peak (the engine allocates outside torch; `unknown` = the engine's memory was not recorded). Rows measured by
`tools/speed_check.py run --modes compile tensorrt`; a failure is shown as `fails: reason`.

<!-- compile:begin (tools/gen_docs.py, from docs/data/speed_compile.jsonl) -->
| model | input | time: portable | compile | TensorRT (fp16 unless shown) | kernels | preparation: compile (cold first call) | ONNX + engine build | peak MiB (alloc / reserved): portable | compile | kernels | TensorRT (engine + torch) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 994x1482 | 159 ms | 143 ms | 202 ms | 88.9 ms | 89.3 s | 145 s | 1142 / 1278 | 3020 / 3716 | 813 / 920 | 1786 / 1834 |
| aanet-sceneflow | 994x1482 | 140 ms | 91.8 ms | 91.2 ms | 101 ms | 79.9 s | 211 s | 845 / 986 | 754 / 928 | 536 / 644 | 602 / 620 |
| bridgedepth-rvc-pretrain | 994x1482 | 444 ms | 337 ms | public API unavailable | 355 ms | 118 s | – | 2836 / 3144 | 2329 / 2360 | 2226 / 2522 | – |
| bridgedepth-rvc-pretrain | 375x1242 | 114 ms | – | 284 ms, fp32 engine (DINOv2 and DPT monocular branch with eager stereo inference) | 94.7 ms | – | 50.0 s | 1910 / 2006 | – | 1323 / 1446 | 2410 / 2588 |
| bridgedepth-sf | 994x1482 | 437 ms | 325 ms | public API unavailable | 348 ms | 111 s | – | 2916 / 3214 | 2280 / 2308 | 2310 / 2634 | – |
| bridgedepth-sf | 375x1242 | 111 ms | – | 284 ms, fp32 engine (DINOv2 and DPT monocular branch with eager stereo inference) | 92.1 ms | – | 54.1 s | 1859 / 1954 | – | 1272 / 1394 | 2358 / 2508 |
| c-fast-foundationstereo | 994x1482 | 330 ms | 339 ms | 451 ms | 135 ms | 75.4 s | 240 s | 2109 / 2336 | 1649 / 1792 | 1120 / 1236 | 1425 / 1462 |
| c-foundationstereo-s | 994x1482 | 3631 ms | 2773 ms | 2229 ms | 1366 ms | 626 s | 444 s | 4357 / 4950 | 7298 / 7624 | 3058 / 4408 | 10027 / 10108 |
| cas-gwcnet-c-sceneflow | 994x1482 | 327 ms | 295 ms | 220 ms | 225 ms | 61.7 s | 131 s | 2006 / 2474 | 2098 / 2448 | 1725 / 1930 | 1533 / 1548 |
| cas-gwcnet-gc-sceneflow | 994x1482 | 473 ms | 383 ms | 307 ms | 359 ms | 63.7 s | 98.6 s | 2032 / 2228 | 6592 / 6670 | 2035 / 2230 | 4142 / 4194 |
| cfnet-sceneflow | 994x1482 | 409 ms | 249 ms | 171 ms | 225 ms | 53.0 s | 139 s | 1822 / 2578 | 1026 / 1398 | 1073 / 1604 | 1357 / 1410 |
| coex-sceneflow | 994x1482 | 49.9 ms | 29.4 ms | 20.6 ms | 23.5 ms | 22.4 s | 56.6 s | 1093 / 1174 | 351 / 452 | 368 / 514 | 438 / 452 |
| crestereo-eth3d | 994x1482 | 724 ms | 398 ms | 298 ms | 229 ms | 120 s | 224 s | 810 / 946 | 695 / 748 | 644 / 732 | 703 / 740 |
| defom-stereo-vitl-sceneflow | 994x1482 | 1439 ms | 1002 ms | public API unavailable | 645 ms | 122 s | – | 4102 / 4252 | 3021 / 3292 | 2781 / 4234 | – |
| defom-stereo-vitl-sceneflow | 375x1242 | 404 ms | – | 1151 ms, bf16 engine (whole graph at fixed 375 x 1242) | 180 ms | – | 168 s | 2394 / 2538 | – | 1627 / 1786 | 1782 / 1790 |
| dlnr-middlebury | 994x1482 | 399 ms | 303 ms | 241 ms | 215 ms | 67.9 s | 156 s | 1502 / 1976 | 1343 / 1692 | 851 / 982 | 1455 / 1546 |
| dlnr-sceneflow | 994x1482 | 398 ms | 301 ms | 241 ms | 214 ms | 69.0 s | 161 s | 2635 / 3142 | 1343 / 1672 | 1236 / 1606 | 1711 / 1784 |
| fast-acvnet-generalization | 994x1482 | 54.7 ms | 36.4 ms | 36.1 ms | 38.1 ms | 41.9 s | 68.4 s | 653 / 756 | 549 / 678 | 654 / 716 | 573 / 624 |
| fast-acvnet-kitti12 | 994x1482 | 54.7 ms | 36.2 ms | 36.2 ms | 38.1 ms | 42.0 s | 68.7 s | 653 / 756 | 549 / 678 | 654 / 716 | 573 / 624 |
| fast-acvnet-kitti15 | 994x1482 | 54.7 ms | 36.5 ms | 36.2 ms | 38.1 ms | 42.2 s | 70.3 s | 653 / 756 | 549 / 678 | 654 / 716 | 573 / 624 |
| fast-acvnet-plus-generalization | 994x1482 | 70.5 ms | 42.0 ms | 40.8 ms | 46.5 ms | 40.9 s | 74.1 s | 651 / 756 | 549 / 830 | 651 / 716 | 1127 / 1178 |
| fast-acvnet-plus-kitti12 | 994x1482 | 70.6 ms | 42.1 ms | 40.6 ms | 46.5 ms | 41.9 s | 71.9 s | 651 / 756 | 549 / 830 | 651 / 716 | 1127 / 1178 |
| fast-acvnet-plus-kitti15 | 994x1482 | 70.5 ms | 42.1 ms | 40.7 ms | 46.5 ms | 39.5 s | 73.9 s | 651 / 756 | 549 / 830 | 651 / 716 | 1127 / 1178 |
| fast-acvnet-plus-sceneflow | 994x1482 | 70.5 ms | 41.9 ms | 40.9 ms | 46.5 ms | 40.6 s | 72.8 s | 651 / 756 | 549 / 830 | 651 / 716 | 1127 / 1178 |
| fast-acvnet-sceneflow | 994x1482 | 54.7 ms | 36.2 ms | 36.7 ms | 38.1 ms | 41.8 s | 71.3 s | 653 / 756 | 584 / 698 | 654 / 716 | 563 / 576 |
| fast-foundationstereo | 994x1482 | 330 ms | 339 ms | 156 ms | 135 ms | 78.4 s | 157 s | 2109 / 2256 | 1613 / 1752 | 1120 / 1236 | 1342 / 1398 |
| foundationstereo-vit-l | 994x1482 | 2989 ms | 2501 ms | 1513 ms | 1440 ms | 226 s | 611 s | 2415 / 2656 | 3453 / 3954 | 2319 / 2760 | 4253 / 4322 |
| foundationstereo-vit-s | 994x1482 | 2514 ms | 2044 ms | 1196 ms | 1120 ms | 212 s | 545 s | 2493 / 2606 | 2822 / 3646 | 2169 / 2470 | 2794 / 2872 |
| great-igev-eth3d | 994x1482 | 1085 ms | 829 ms | 940 ms | 397 ms | 224 s | 685 s | 1402 / 1678 | 2831 / 3362 | 1134 / 1302 | 6151 / 6172 |
| great-igev-kitti | 994x1482 | 1085 ms | 830 ms | 942 ms | 396 ms | 225 s | 687 s | 1402 / 1678 | 2831 / 3362 | 1134 / 1302 | 6155 / 6234 |
| great-igev-middlebury | 994x1482 | 1789 ms | 1676 ms | fails: AttributeError: 'NoneType' object has no attribute 'get_tens | 580 ms | 326 s | – | 1602 / 1878 | 5228 / 5720 | 1472 / 1642 | – |
| great-igev-middlebury | 375x1242 | 492 ms | – | 3697 ms, fp32 engine (MobileNetV2 feature pyramid with eager stereo inference) | 176 ms | – | 13.1 s | 1165 / 1298 | – | 1060 / 1144 | 2388 / 4022 |
| great-igev-sceneflow | 994x1482 | 1085 ms | 830 ms | 944 ms | 397 ms | 224 s | 675 s | 1402 / 1678 | 2831 / 3362 | 1134 / 1302 | 6155 / 6234 |
| gwcnet-g-sceneflow | 994x1482 | 344 ms | 417 ms | 275 ms | 238 ms | 52.6 s | 66.0 s | 2081 / 2402 | 1527 / 1864 | 1799 / 2102 | 3634 / 3648 |
| gwcnet-gc-sceneflow | 994x1482 | 353 ms | 420 ms | 669 ms | 259 ms | 83.8 s | 117 s | 1792 / 2102 | 1774 / 2246 | 1510 / 1782 | 2732 / 2786 |
| hitnet-eth3d | 994x1482 | 67.6 ms | 27.7 ms | 31.1 ms | 23.2 ms | 39.1 s | 136 s | 848 / 1068 | 719 / 728 | 623 / 770 | 912 / 920 |
| hitnet-middlebury-d400 | 994x1482 | 176 ms | 71.5 ms | 74.9 ms | 53.1 ms | 55.4 s | 166 s | 1701 / 2154 | 1296 / 1332 | 1207 / 1514 | 1985 / 2014 |
| igev-plusplus-eth3d | 994x1482 | 1079 ms | 716 ms | 494 ms | 427 ms | 159 s | 425 s | 1583 / 1696 | 952 / 1074 | 1272 / 1514 | 3765 / 3786 |
| igev-plusplus-kitti12 | 994x1482 | 1080 ms | 718 ms | 494 ms | 394 ms | 158 s | 422 s | 1583 / 1696 | 917 / 1054 | 1272 / 1514 | 3821 / 3900 |
| igev-plusplus-kitti15 | 994x1482 | 1080 ms | 718 ms | 494 ms | 394 ms | 159 s | 425 s | 1583 / 1696 | 917 / 1052 | 1272 / 1494 | 3821 / 3900 |
| igev-plusplus-middlebury | 994x1482 | 1080 ms | 718 ms | 494 ms | 393 ms | 157 s | 436 s | 1583 / 1696 | 917 / 1054 | 1272 / 1494 | 3821 / 3900 |
| igev-plusplus-sceneflow | 994x1482 | 1079 ms | 718 ms | 495 ms | 395 ms | 158 s | 426 s | 1583 / 1696 | 917 / 1054 | 1272 / 1514 | 3821 / 3900 |
| igev-stereo-eth3d | 994x1482 | 763 ms | 527 ms | 339 ms | 288 ms | 111 s | 213 s | 1237 / 1492 | 807 / 1052 | 1113 / 1374 | 1350 / 1370 |
| igev-stereo-kitti12 | 994x1482 | 763 ms | 527 ms | 340 ms | 288 ms | 112 s | 215 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1313 / 1372 |
| igev-stereo-kitti15 | 994x1482 | 763 ms | 527 ms | 340 ms | 288 ms | 110 s | 214 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1315 / 1374 |
| igev-stereo-middlebury | 994x1482 | 763 ms | 527 ms | 338 ms | 288 ms | 110 s | 214 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1317 / 1376 |
| igev-stereo-sceneflow | 994x1482 | 763 ms | 527 ms | 340 ms | 288 ms | 111 s | 216 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1317 / 1376 |
| las2-h | 994x1482 | 113 ms | 75.0 ms | public API unavailable | 53.6 ms | 37.5 s | – | 1002 / 1046 | 579 / 724 | 872 / 960 | – |
| las2-h | 375x1242 | 35.5 ms | – | 17.5 ms (whole graph at fixed 375 x 1242) | 15.0 ms | – | 100 s | 365 / 390 | – | 298 / 368 | 300 / 308 |
| las2-l | 994x1482 | 75.7 ms | 46.6 ms | 23.4 ms | 30.1 ms | 30.5 s | 97.5 s | 648 / 716 | 385 / 432 | 268 / 362 | 393 / 424 |
| las2-m | 994x1482 | 37.1 ms | 28.6 ms | 13.2 ms | 18.1 ms | 24.0 s | 98.3 s | 584 / 664 | 252 / 300 | 235 / 336 | 343 / 386 |
| las2-s | 994x1482 | 27.9 ms | 23.7 ms | 10.3 ms | 14.3 ms | 22.4 s | 86.9 s | 557 / 608 | 238 / 264 | 226 / 320 | 258 / 304 |
| mocha-v1-sceneflow | 994x1482 | 1057 ms | 735 ms | 447 ms | 455 ms | 154 s | 371 s | 1632 / 1770 | 1129 / 1268 | 1588 / 1730 | 1466 / 1590 |
| mocha-v2-sceneflow | 994x1482 | 1033 ms | 713 ms | 437 ms | 448 ms | 148 s | 361 s | 1632 / 1770 | 1122 / 1308 | 1587 / 1790 | 1400 / 1486 |
| monster-mix-all | 994x1482 | 1585 ms | 1098 ms | 674 ms | 656 ms | 105 s | 391 s | 1926 / 2092 | 2081 / 2272 | 1829 / 1954 | 2898 / 2960 |
| monster-plusplus-mix-all | 994x1482 | 1633 ms | 1126 ms | 697 ms | 657 ms | 109 s | 416 s | 2583 / 2918 | 2328 / 2532 | 2438 / 2754 | 3489 / 3552 |
| monster-sceneflow | 994x1482 | 1585 ms | 1096 ms | 671 ms | 655 ms | 103 s | 384 s | 2325 / 2574 | 2081 / 2272 | 2216 / 2514 | 2865 / 2928 |
| nmrf-sceneflow | 994x1482 | 216 ms | 161 ms | 184 ms | 196 ms | 84.8 s | 142 s | 1510 / 1552 | 854 / 1104 | 1022 / 1112 | 1027 / 1134 |
| psmnet-sceneflow | 994x1482 | 340 ms | 440 ms | 669 ms | 231 ms | 60.0 s | 88.9 s | 1881 / 2402 | 1760 / 2082 | 1469 / 1782 | 1611 / 1630 |
| raft-stereo-eth3d | 994x1482 | 746 ms | 506 ms | 333 ms | 275 ms | 99.9 s | 132 s | 1388 / 1492 | 815 / 988 | 722 / 738 | 952 / 972 |
| raft-stereo-middlebury | 994x1482 | 746 ms | 506 ms | 334 ms | 274 ms | 100 s | 132 s | 1388 / 1492 | 780 / 988 | 722 / 758 | 1012 / 1090 |
| raft-stereo-sceneflow | 994x1482 | 746 ms | 507 ms | 333 ms | 274 ms | 100 s | 132 s | 1388 / 1492 | 780 / 988 | 722 / 758 | 1014 / 1092 |
| rt-monster-plusplus-mix-all | 994x1482 | 109 ms | 77.0 ms | 44.1 ms | 93.8 ms | 69.4 s | 150 s | 1291 / 1378 | 2750 / 2942 | 1277 / 1420 | 1139 / 1210 |
| s2m2-l | 994x1482 | 1017 ms | 650 ms | 497 ms | 376 ms | 144 s | 374 s | 3234 / 3816 | 3169 / 3514 | 1249 / 1568 | 2910 / 2982 |
| s2m2-m | 994x1482 | 670 ms | 420 ms | 285 ms | 248 ms | 126 s | 305 s | 2289 / 2766 | 2197 / 2572 | 876 / 1162 | 1958 / 2002 |
| s2m2-s | 994x1482 | 348 ms | 186 ms | 137 ms | 113 ms | 95.9 s | 363 s | 1733 / 1930 | 1579 / 1710 | 686 / 820 | 1186 / 1240 |
| s2m2-xl | 994x1482 | 1824 ms | 1298 ms | 753 ms | 682 ms | 153 s | 408 s | 4811 / 5234 | 5594 / 6014 | 2092 / 2274 | 5283 / 5338 |
| selective-igev-eth3d-finetune | 994x1482 | 996 ms | 691 ms | 461 ms | 351 ms | 126 s | 256 s | 1227 / 1448 | 919 / 1088 | 1051 / 1190 | 1279 / 1316 |
| selective-igev-eth3d-train | 994x1482 | 998 ms | 691 ms | 462 ms | 349 ms | 127 s | 260 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1278 / 1334 |
| selective-igev-kitti12 | 994x1482 | 998 ms | 691 ms | 462 ms | 349 ms | 125 s | 255 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1278 / 1334 |
| selective-igev-kitti15 | 994x1482 | 998 ms | 691 ms | 462 ms | 349 ms | 127 s | 256 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1278 / 1334 |
| selective-igev-middlebury-finetune | 994x1482 | 998 ms | 692 ms | 462 ms | 349 ms | 127 s | 258 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1280 / 1316 |
| selective-igev-middlebury-train | 994x1482 | 998 ms | 692 ms | 462 ms | 349 ms | 128 s | 256 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1278 / 1334 |
| selective-igev-sceneflow | 994x1482 | 998 ms | 692 ms | 463 ms | 349 ms | 127 s | 255 s | 1238 / 1488 | 915 / 1028 | 1109 / 1390 | 1278 / 1334 |
| waft-stereo-b | 994x1482 | 1094 ms | 571 ms | public API unavailable | 388 ms | 104 s | – | 2999 / 3070 | 3908 / 3936 | 1664 / 1932 | – |
| waft-stereo-b | 384x1248 | – | – | 1209 ms, fp32 engine (standalone build) | – | – | 107 s | – | – | – | 1447 / 1454 |
| waft-stereo-s | 994x1482 | 689 ms | 357 ms | 170 ms | 284 ms | 95.1 s | 152 s | 3293 / 3456 | 2024 / 2058 | 954 / 1102 | 1061 / 1114 |
| any2full-b | 994x1482 | 26.8 ms | 24.2 ms | 15.8 ms | 17.2 ms | 97.0 s | 64.4 s | 538 / 596 | 442 / 496 | 481 / 536 | 731 / 794 |
| any2full-l | 994x1482 | 67.7 ms | 63.8 ms | 40.2 ms | 42.3 ms | 33.9 s | 79.0 s | 1178 / 1250 | 987 / 1092 | 1115 / 1210 | 1860 / 1928 |
| any2full-s | 994x1482 | 12.4 ms | 9.88 ms | 6.72 ms | 9.47 ms | 26.4 s | 59.0 s | 274 / 312 | 223 / 294 | 218 / 252 | 366 / 398 |
| da3-giant-1.1 | 994x1482 | 71.5 ms | 64.3 ms | 37.2 ms | 45.2 ms | 47.2 s | 84.1 s | 2634 / 2728 | 2487 / 2528 | 2535 / 2590 | 4795 / 4842 |
| da3-large-1.1 | 994x1482 | 30.0 ms | 26.5 ms | 13.8 ms | 17.3 ms | 45.4 s | 62.6 s | 988 / 1088 | 841 / 868 | 891 / 950 | 1562 / 1622 |
| da3metric-large | 994x1482 | 24.3 ms | 21.2 ms | 14.4 ms | 15.0 ms | 26.6 s | 58.2 s | 811 / 868 | 792 / 828 | 802 / 848 | 1465 / 1502 |
| da3mono-large | 994x1482 | 24.1 ms | 21.2 ms | 14.4 ms | 14.9 ms | 25.0 s | 52.3 s | 814 / 868 | 792 / 828 | 805 / 868 | 1463 / 1520 |
| da3nested-giant-large-1.1 | 994x1482 | 92.5 ms | 85.4 ms | 51.9 ms | 59.5 ms | 60.0 s | 154 s | 3118 / 3194 | 3124 / 3194 | 3120 / 3198 | 6152 / 6226 |
| depth-anything-v2-b | 994x1482 | 18.8 ms | 17.0 ms | 10.9 ms | 10.5 ms | 14.2 s | 32.7 s | 450 / 516 | 340 / 394 | 387 / 436 | 539 / 596 |
| depth-anything-v2-l | 994x1482 | 55.1 ms | 51.7 ms | 32.4 ms | 31.1 ms | 20.6 s | 43.5 s | 1026 / 1352 | 850 / 952 | 921 / 1034 | 1579 / 1634 |
| depth-anything-v2-metric-hypersim-b | 994x1482 | 18.6 ms | 17.0 ms | 10.8 ms | 10.8 ms | 16.0 s | 32.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 539 / 614 |
| depth-anything-v2-metric-hypersim-l | 994x1482 | 54.5 ms | 51.3 ms | 32.3 ms | 31.7 ms | 20.2 s | 43.5 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 1579 / 1634 |
| depth-anything-v2-metric-hypersim-s | 994x1482 | 7.66 ms | 6.67 ms | 4.47 ms | 4.67 ms | 15.8 s | 29.8 s | 244 / 288 | 201 / 224 | 152 / 208 | 305 / 348 |
| depth-anything-v2-metric-vkitti-b | 994x1482 | 18.6 ms | 17.0 ms | 10.9 ms | 10.8 ms | 15.9 s | 32.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 539 / 614 |
| depth-anything-v2-metric-vkitti-l | 994x1482 | 54.5 ms | 51.3 ms | 32.4 ms | 31.9 ms | 20.1 s | 43.4 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 1579 / 1634 |
| depth-anything-v2-metric-vkitti-s | 994x1482 | 7.66 ms | 6.68 ms | 4.47 ms | 4.67 ms | 15.8 s | 29.9 s | 244 / 288 | 201 / 224 | 152 / 208 | 305 / 348 |
| depth-anything-v2-s | 994x1482 | 7.82 ms | 6.65 ms | 4.46 ms | 4.51 ms | 15.7 s | 29.9 s | 249 / 326 | 201 / 224 | 190 / 248 | 273 / 308 |
| depth-pro | 994x1482 | 521 ms | 488 ms | 299 ms | 277 ms | 39.2 s | 163 s | 3036 / 3298 | 3053 / 3198 | 3050 / 3198 | 4816 / 4882 |
| depthfm-v1 | 994x1482 | 4854 ms | 3947 ms | fails: ONNX export: conversion failed | 3555 ms | 102 s | – | 5422 / 7404 | 9240 / 10292 | 3768 / 5942 | – |
| depthfm-v1 | 480x640 | – | – | 353 ms (fixed-shape exported depth network) | – | – | – | – | – | – | 3479 / 3494 |
| distill-any-depth-b | 994x1482 | 18.6 ms | 17.0 ms | 10.8 ms | 10.8 ms | 16.2 s | 33.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 539 / 614 |
| distill-any-depth-l | 994x1482 | 54.4 ms | 51.6 ms | 32.4 ms | 31.8 ms | 20.1 s | 42.5 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 1579 / 1634 |
| distill-any-depth-s | 994x1482 | 7.65 ms | 6.67 ms | 4.46 ms | 4.63 ms | 15.9 s | 29.9 s | 244 / 288 | 201 / 224 | 152 / 208 | 305 / 348 |
| lingbot-depth | 994x1482 | 280 ms | 251 ms | 188 ms | 161 ms | 28.0 s | 65.5 s | 1559 / 1872 | 1381 / 1572 | 1453 / 1590 | 1821 / 1876 |
| lingbot-depth-dc | 994x1482 | 281 ms | 252 ms | 188 ms | 161 ms | 32.4 s | 67.3 s | 1559 / 1872 | 1381 / 1572 | 1453 / 1590 | 1821 / 1876 |
| lotus-normal-d-v1-1 | 994x1482 | 268 ms | 222 ms | 107 ms | 164 ms | 63.9 s | 183 s | 2777 / 2940 | 2344 / 2416 | 2351 / 2500 | 4215 / 4242 |
| lotus-normal-g-v1-1 | 994x1482 | 267 ms | 222 ms | 107 ms | 164 ms | 63.4 s | 183 s | 2777 / 2940 | 2344 / 2416 | 2351 / 2500 | 4213 / 4240 |
| marigold-depth-v1-1 | 994x1482 | 410 ms | 342 ms | 172 ms | 261 ms | 127 s | 285 s | 2776 / 2938 | 2341 / 2418 | 2343 / 2500 | 4263 / 4318 |
| marigold-normals-lcm-v0-1 | 994x1482 | 272 ms | 223 ms | 109 ms | 167 ms | 64.5 s | 182 s | 2792 / 2960 | 2348 / 2436 | 2361 / 2500 | 4220 / 4250 |
| marigold-v2-depth | 994x1482 | 4763 ms | 4599 ms | public API unavailable | 3805 ms | 141 s | – | 1408 / 1838 | 1959 / 2120 | 1408 / 1838 | – |
| marigold-v2-depth | 480x640 | – | – | 1156 ms, bf16 engine (VAE encoder and decoder with eager streamed NF4 transformer) | – | – | 137 s | – | – | – | 2042 / 2198 |
| metric3d-vit-large | 994x1482 | 187 ms | 149 ms | 91.4 ms | 151 ms | 54.7 s | 96.6 s | 1239 / 1436 | 1126 / 1148 | 1464 / 1568 | 1867 / 1952 |
| midas-dpt-beit-large-512 | 994x1482 | 78.8 ms | 58.8 ms | 52.2 ms | 63.9 ms | 32.6 s | 53.5 s | 992 / 1418 | 1088 / 1116 | 1003 / 1436 | 1514 / 1592 |
| moge-2-vitb-normal | 994x1482 | 83.9 ms | 60.9 ms | 35.5 ms | 40.7 ms | 133 s | 127 s | 1163 / 1312 | 972 / 1172 | 1093 / 1208 | 1044 / 1092 |
| moge-2-vitl-normal | 994x1482 | 144 ms | 124 ms | 75.6 ms | 73.4 ms | 152 s | 173 s | 1661 / 1836 | 1411 / 1634 | 1498 / 1610 | 1882 / 1936 |
| moge-2-vits-normal | 994x1482 | 61.6 ms | 43.9 ms | 24.5 ms | 33.3 ms | 124 s | 131 s | 1007 / 1172 | 836 / 992 | 932 / 1048 | 768 / 852 |
| moge-3-vitl | 994x1482 | 328 ms | 311 ms | public API unavailable | 149 ms | 136 s | – | 1622 / 1706 | 1500 / 1724 | 1626 / 1762 | – |
| moge-3-vitl | 480x640 | – | – | 1069 ms, fp32 engine (DINOv2 encoder and projections with eager decoder and sparse refinement) | – | – | – | – | – | – | 4568 / 5382 |
| omni-dc | 994x1482 | 571 ms | 411 ms | public API unavailable | 510 ms | 137 s | – | 4130 / 6144 | 4201 / 5932 | 4154 / 6528 | – |
| omni-dc | 480x640 | – | – | 253 ms, fp32 engine (Depth Anything V2 with eager prompt fusion and solver) | – | – | 52.0 s | – | – | – | 2930 / 3298 |
| pointdit-b-512 | 994x1482 | 31.1 ms | 23.7 ms | 15.5 ms | 20.5 ms | 33.6 s | 86.5 s | 446 / 504 | 465 / 524 | 462 / 630 | 795 / 898 |
| pointdit-l-512 | 994x1482 | 94.6 ms | 76.0 ms | 49.1 ms | 61.3 ms | 57.3 s | 150 s | 1297 / 1430 | 1316 / 1430 | 1310 / 1676 | 2474 / 2626 |
| prompt-depth-anything-l | 756x1008 | 125 ms | 117 ms | 75.3 ms | 69.3 ms | 21.2 s | 52.5 s | 1318 / 1488 | 968 / 1148 | 1179 / 1348 | 1649 / 1714 |
| prompt-depth-anything-s | 756x1008 | 18.4 ms | 16.3 ms | 10.5 ms | 11.4 ms | 14.7 s | 46.6 s | 363 / 442 | 260 / 322 | 297 / 384 | 368 / 410 |
| stablenormal-turbo | 994x1482 | 291 ms | 238 ms | 118 ms | 180 ms | 76.3 s | 198 s | 3316 / 3626 | 3019 / 3106 | 3037 / 3172 | 5587 / 5626 |
| unidepth-v2-vitl14 | 994x1482 | 99.8 ms | 96.4 ms | 64.3 ms | 63.0 ms | 31.1 s | 119 s | 1208 / 1472 | 1304 / 1392 | 1153 / 1252 | 1815 / 1842 |
| unik3d-vitl | 994x1482 | 108 ms | 94.4 ms | 58.0 ms | 62.9 ms | 32.8 s | 105 s | 1219 / 1492 | 1258 / 1312 | 1166 / 1266 | 1609 / 1678 |
| video-depth-anything-metric-s | 480x640 | 148 ms | 101 ms | 60.2 ms | 59.1 ms | 32.5 s | 82.1 s | 1461 / 1666 | 1143 / 1464 | 1270 / 1408 | 886 / 914 |
| video-depth-anything-s | 1080x1920 | 202 ms | 1314 ms | – | 200 ms | 82.7 s | – | 1962 / 2204 | 7762 / 8530 | 1967 / 2226 | – |
| video-depth-anything-s | 480x640 | 147 ms | 100 ms | 60.3 ms | 58.5 ms | 34.8 s | 103 s | 1070 / 1266 | 1143 / 1344 | 1101 / 1288 | 886 / 914 |
| zoedepth-nk | 994x1482 | 124 ms | 71.7 ms | 73.3 ms | 88.9 ms | 93.8 s | 138 s | 1843 / 1944 | 1813 / 1880 | 1771 / 1960 | 2246 / 2306 |
| alexnet.tv_in1k | 224x224 | 0.970 ms | 0.824 ms | 0.401 ms | 0.604 ms | 4.33 s | 13.4 s | 151 / 224 | 183 / 204 | 184 / 228 | 278 / 312 |
| coatnet_0_rw_224.sw_in1k | 224x224 | 6.49 ms | 3.59 ms | 0.849 ms | 1.74 ms | 12.1 s | 38.2 s | 102 / 124 | 120 / 142 | 100 / 144 | 158 / 170 |
| coatnet_1_rw_224.sw_in1k | 224x224 | 10.7 ms | 5.58 ms | 1.38 ms | 2.89 ms | 16.2 s | 42.0 s | 132 / 158 | 149 / 176 | 130 / 196 | 231 / 250 |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 224x224 | 11.9 ms | 5.53 ms | 1.73 ms | 4.01 ms | 16.2 s | 36.0 s | 207 / 240 | 211 / 240 | 204 / 240 | 374 / 416 |
| coatnet_nano_rw_224.sw_in1k | 224x224 | 7.23 ms | 4.71 ms | 0.709 ms | 1.62 ms | 13.9 s | 31.8 s | 74 / 86 | 96 / 126 | 72 / 106 | 102 / 124 |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 224x224 | 12.3 ms | 6.63 ms | 1.55 ms | 3.01 ms | 17.9 s | 42.6 s | 132 / 158 | 162 / 176 | 130 / 178 | 257 / 276 |
| coatnet_rmlp_1_rw_224.sw_in1k | 224x224 | 12.4 ms | 6.54 ms | 1.47 ms | 3.04 ms | 18.6 s | 43.8 s | 131 / 156 | 155 / 176 | 129 / 178 | 243 / 262 |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 224x224 | 12.8 ms | 6.52 ms | 1.87 ms | 4.24 ms | 18.4 s | 41.0 s | 207 / 240 | 238 / 260 | 204 / 240 | 402 / 444 |
| coatnet_rmlp_2_rw_224.sw_in1k | 224x224 | 15.1 ms | 6.34 ms | 1.92 ms | 4.24 ms | 18.4 s | 45.9 s | 207 / 240 | 238 / 260 | 204 / 240 | 404 / 446 |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 384x384 | 13.9 ms | 6.92 ms | 3.69 ms | 12.8 ms | 18.6 s | 42.3 s | 296 / 330 | 326 / 370 | 288 / 356 | 464 / 500 |
| coatnet_rmlp_nano_rw_224.sw_in1k | 224x224 | 8.09 ms | 9.89 ms | 0.868 ms | 1.75 ms | 13.9 s | 37.1 s | 78 / 106 | 110 / 126 | 77 / 106 | 122 / 144 |
| convnext_base.fb_in1k | 224x224 | 5.84 ms | 3.37 ms | 1.45 ms | 2.00 ms | 10.9 s | 32.5 s | 211 / 234 | 323 / 334 | 208 / 236 | 482 / 494 |
| convnext_base.fb_in22k_ft_in1k | 224x224 | 5.84 ms | 3.37 ms | 1.45 ms | 2.01 ms | 11.4 s | 31.3 s | 211 / 234 | 323 / 334 | 208 / 236 | 482 / 494 |
| convnext_base.fb_in22k_ft_in1k_384 | 224x224 | 5.79 ms | 3.41 ms | 2.31 ms | 3.05 ms | 10.8 s | 31.2 s | 229 / 252 | 336 / 354 | 213 / 232 | 493 / 522 |
| convnext_large.fb_in1k | 224x224 | 5.81 ms | 3.45 ms | 2.50 ms | 3.28 ms | 10.8 s | 33.9 s | 423 / 432 | 642 / 652 | 419 / 434 | 986 / 1006 |
| convnext_large.fb_in22k_ft_in1k | 224x224 | 5.88 ms | 3.44 ms | 2.50 ms | 3.28 ms | 10.8 s | 35.0 s | 423 / 432 | 642 / 652 | 419 / 434 | 986 / 1006 |
| convnext_large.fb_in22k_ft_in1k_384 | 224x224 | 6.11 ms | 5.80 ms | 4.23 ms | 5.21 ms | 10.7 s | 33.4 s | 449 / 492 | 654 / 672 | 426 / 454 | 1001 / 1040 |
| convnext_small.fb_in1k | 224x224 | 5.97 ms | 3.47 ms | 1.16 ms | 1.84 ms | 10.6 s | 30.5 s | 136 / 150 | 211 / 230 | 134 / 152 | 278 / 294 |
| convnext_small.fb_in22k_ft_in1k | 224x224 | 5.84 ms | 3.50 ms | 1.17 ms | 1.84 ms | 10.5 s | 30.9 s | 136 / 150 | 211 / 230 | 134 / 152 | 278 / 294 |
| convnext_small.fb_in22k_ft_in1k_384 | 224x224 | 5.82 ms | 3.36 ms | 1.68 ms | 2.53 ms | 10.7 s | 31.8 s | 150 / 172 | 223 / 232 | 138 / 152 | 317 / 350 |
| convnext_tiny.fb_in1k | 224x224 | 3.31 ms | 2.09 ms | 0.682 ms | 1.15 ms | 8.22 s | 27.4 s | 94 / 110 | 150 / 170 | 92 / 110 | 184 / 200 |
| convnext_tiny.fb_in22k_ft_in1k | 224x224 | 3.34 ms | 2.08 ms | 0.687 ms | 1.15 ms | 8.09 s | 28.3 s | 94 / 110 | 150 / 170 | 92 / 110 | 184 / 200 |
| convnext_tiny.fb_in22k_ft_in1k_384 | 224x224 | 3.30 ms | 2.05 ms | 0.970 ms | 1.56 ms | 7.90 s | 28.2 s | 108 / 132 | 162 / 170 | 96 / 110 | 192 / 226 |
| convnextv2_base.fcmae_ft_in22k_in1k | 224x224 | 10.2 ms | 3.85 ms | 2.13 ms | 2.96 ms | 14.4 s | 35.1 s | 220 / 234 | 324 / 356 | 206 / 238 | 486 / 518 |
| davit_base.msft_in1k | 224x224 | 11.7 ms | 5.48 ms | 2.28 ms | 2.09 ms | 21.8 s | 71.1 s | 212 / 224 | 242 / 262 | 209 / 244 | 403 / 422 |
| davit_small.msft_in1k | 224x224 | 12.8 ms | 5.54 ms | 1.92 ms | 1.88 ms | 21.8 s | 66.4 s | 136 / 150 | 166 / 190 | 134 / 168 | 234 / 254 |
| davit_tiny.msft_in1k | 224x224 | 6.33 ms | 3.10 ms | 1.03 ms | 1.17 ms | 15.5 s | 53.8 s | 96 / 116 | 125 / 136 | 93 / 134 | 159 / 166 |
| deit_base_patch16_224.fb_in1k | 224x224 | 2.89 ms | 2.09 ms | 1.12 ms | 1.66 ms | 7.27 s | 28.3 s | 203 / 226 | 238 / 266 | 200 / 248 | 394 / 420 |
| densenet121.tv_in1k | 224x224 | 9.68 ms | 5.49 ms | 0.686 ms | 1.29 ms | 20.4 s | 33.6 s | 53 / 70 | 89 / 112 | 53 / 92 | 85 / 106 |
| dinov2_vitl14_lc | 224x224 | 5.43 ms | 5.09 ms | 4.22 ms | 4.95 ms | 12.1 s | 38.2 s | 623 / 1228 | 660 / 688 | 623 / 672 | 1233 / 1262 |
| efficientformer_l1.snap_dist_in1k | 224x224 | 3.72 ms | 1.69 ms | 0.335 ms | 0.704 ms | 9.10 s | 35.4 s | 93 / 104 | 125 / 144 | 91 / 126 | 128 / 142 |
| efficientformer_l3.snap_dist_in1k | 224x224 | 6.37 ms | 2.57 ms | 0.750 ms | 1.21 ms | 11.8 s | 39.4 s | 161 / 216 | 193 / 216 | 160 / 218 | 227 / 286 |
| efficientformer_l7.snap_dist_in1k | 224x224 | 9.65 ms | 3.69 ms | 1.40 ms | 2.04 ms | 14.4 s | 40.0 s | 300 / 324 | 332 / 364 | 298 / 346 | 460 / 490 |
| efficientformerv2_l.snap_dist_in1k | 224x224 | 17.1 ms | 7.32 ms | 1.20 ms | 3.05 ms | 20.7 s | 45.1 s | 93 / 144 | 125 / 184 | 94 / 188 | 161 / 214 |
| efficientformerv2_s0.snap_dist_in1k | 224x224 | 7.62 ms | 3.34 ms | 0.550 ms | 1.20 ms | 11.3 s | 36.3 s | 44 / 64 | 76 / 104 | 44 / 66 | 78 / 100 |
| efficientformerv2_s1.snap_dist_in1k | 224x224 | 8.91 ms | 3.91 ms | 0.632 ms | 1.39 ms | 12.7 s | 41.2 s | 50 / 88 | 82 / 108 | 50 / 88 | 86 / 126 |
| efficientformerv2_s2.snap_dist_in1k | 224x224 | 13.9 ms | 5.55 ms | 0.919 ms | 2.17 ms | 16.4 s | 38.9 s | 64 / 104 | 96 / 124 | 64 / 104 | 100 / 142 |
| efficientnet_b0.tv_in1k | 224x224 | 4.98 ms | 2.88 ms | 0.536 ms | 0.769 ms | 9.12 s | 25.8 s | 48 / 96 | 80 / 96 | 78 / 118 | 81 / 92 |
| efficientnet_b4.tv_in1k | 224x224 | 9.93 ms | 5.39 ms | 1.63 ms | 2.02 ms | 13.8 s | 34.9 s | 89 / 114 | 124 / 154 | 90 / 118 | 157 / 200 |
| efficientnet_v2_s.tv_in1k | 224x224 | 10.2 ms | 5.39 ms | 1.35 ms | 1.91 ms | 12.1 s | 32.0 s | 81 / 126 | 116 / 146 | 81 / 128 | 149 / 198 |
| efficientvit_b2.r288_in1k | 224x224 | 12.9 ms | 6.91 ms | 0.790 ms | 1.12 ms | 11.0 s | 35.3 s | 85 / 144 | 120 / 164 | 88 / 170 | 154 / 216 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 224x224 | 18.2 ms | 14.9 ms | 12.4 ms | 14.1 ms | 21.0 s | 43.5 s | 648 / 664 | 674 / 684 | 652 / 786 | 1259 / 1286 |
| falconsai-nsfw-vit | 224x224 | 2.83 ms | 2.02 ms | 1.12 ms | 1.61 ms | 8.29 s | 27.4 s | 201 / 226 | 236 / 266 | 198 / 228 | 393 / 420 |
| fastvit_sa24.apple_dist_in1k | 256x256 | 6.03 ms | 3.97 ms | 0.818 ms | 1.41 ms | 8.44 s | 54.5 s | 81 / 110 | 113 / 148 | 80 / 132 | 148 / 200 |
| fastvit_t12.apple_dist_in1k | 256x256 | 3.27 ms | 2.32 ms | 0.431 ms | 0.781 ms | 7.39 s | 39.7 s | 54 / 76 | 86 / 94 | 52 / 76 | 89 / 114 |
| fastvit_t8.apple_dist_in1k | 256x256 | 3.25 ms | 2.55 ms | 0.368 ms | 0.709 ms | 6.81 s | 30.8 s | 45 / 72 | 77 / 92 | 45 / 76 | 78 / 108 |
| focalnet_base_lrf.ms_in1k | 224x224 | 15.5 ms | 7.47 ms | 2.08 ms | 5.44 ms | 19.7 s | 63.9 s | 212 / 234 | 235 / 274 | 212 / 258 | 404 / 432 |
| focalnet_base_srf.ms_in1k | 224x224 | 14.2 ms | 6.94 ms | 1.74 ms | 4.99 ms | 17.8 s | 63.8 s | 211 / 234 | 234 / 272 | 210 / 256 | 401 / 430 |
| focalnet_small_lrf.ms_in1k | 224x224 | 15.8 ms | 7.28 ms | 1.70 ms | 4.18 ms | 19.5 s | 62.5 s | 136 / 152 | 162 / 192 | 136 / 154 | 265 / 286 |
| focalnet_small_srf.ms_in1k | 224x224 | 14.0 ms | 6.90 ms | 1.49 ms | 3.73 ms | 17.8 s | 65.3 s | 136 / 152 | 161 / 190 | 134 / 154 | 234 / 256 |
| focalnet_tiny_lrf.ms_in1k | 224x224 | 8.07 ms | 3.94 ms | 0.952 ms | 2.39 ms | 13.3 s | 49.9 s | 95 / 118 | 120 / 138 | 93 / 138 | 158 / 186 |
| focalnet_tiny_srf.ms_in1k | 224x224 | 7.36 ms | 3.72 ms | 0.864 ms | 2.19 ms | 12.9 s | 52.0 s | 95 / 118 | 120 / 136 | 93 / 138 | 159 / 188 |
| hgnetv2_b0.ssld_stage2_ft_in1k | 224x224 | 3.46 ms | 1.78 ms | 0.334 ms | 0.568 ms | 6.74 s | 21.2 s | 48 / 70 | 78 / 110 | 46 / 72 | 77 / 100 |
| hgnetv2_b1.ssld_stage2_ft_in1k | 224x224 | 3.47 ms | 1.84 ms | 0.307 ms | 0.603 ms | 7.07 s | 20.1 s | 50 / 68 | 80 / 108 | 48 / 70 | 81 / 104 |
| hgnetv2_b2.ssld_stage2_ft_in1k | 224x224 | 4.55 ms | 2.33 ms | 0.383 ms | 0.765 ms | 7.65 s | 23.4 s | 59 / 94 | 89 / 134 | 57 / 94 | 91 / 128 |
| hgnetv2_b3.ssld_stage2_ft_in1k | 224x224 | 5.22 ms | 2.55 ms | 0.436 ms | 0.909 ms | 7.99 s | 24.6 s | 76 / 114 | 99 / 152 | 67 / 116 | 100 / 150 |
| hgnetv2_b4.ssld_stage2_ft_in1k | 224x224 | 4.53 ms | 2.66 ms | 0.463 ms | 0.864 ms | 7.84 s | 19.8 s | 82 / 116 | 107 / 134 | 75 / 138 | 139 / 182 |
| hgnetv2_b5.ssld_stage2_ft_in1k | 224x224 | 7.04 ms | 4.13 ms | 0.758 ms | 1.29 ms | 9.97 s | 23.8 s | 114 / 150 | 146 / 190 | 115 / 156 | 209 / 250 |
| hgnetv2_b6.ssld_stage2_ft_in1k | 224x224 | 9.44 ms | 5.44 ms | 1.42 ms | 2.10 ms | 11.2 s | 26.9 s | 198 / 224 | 217 / 264 | 188 / 232 | 347 / 394 |
| inception_v3.tv_in1k | 224x224 | 5.13 ms | 3.03 ms | 0.685 ms | 1.31 ms | 8.61 s | 22.4 s | 120 / 164 | 152 / 184 | 118 / 208 | 184 / 234 |
| mobilenet_v2.tv2_in1k | 224x224 | 2.95 ms | 1.82 ms | 0.261 ms | 0.439 ms | 6.56 s | 16.3 s | 43 / 70 | 77 / 92 | 75 / 92 | 76 / 84 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 3.80 ms | 2.19 ms | 0.393 ms | 0.545 ms | 10.2 s | 36.5 s | 46 / 76 | 79 / 90 | 78 / 114 | 80 / 106 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 1.77 ms | 1.21 ms | 0.176 ms | 0.382 ms | 4.59 s | 14.1 s | 44 / 70 | 77 / 90 | 43 / 72 | 77 / 106 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 3.91 ms | 2.48 ms | 0.345 ms | 0.641 ms | 6.98 s | 19.6 s | 54 / 86 | 87 / 102 | 87 / 124 | 88 / 116 |
| regnety_160.tv2_in1k | 224x224 | 8.08 ms | 5.42 ms | 1.78 ms | 1.87 ms | 8.01 s | 28.1 s | 208 / 250 | 236 / 290 | 202 / 276 | 364 / 420 |
| repvit_m0_9.dist_300e_in1k | 224x224 | 5.35 ms | 3.30 ms | 0.509 ms | 0.776 ms | 8.62 s | 33.7 s | 45 / 76 | 77 / 96 | 45 / 76 | 75 / 108 |
| repvit_m0_9.dist_450e_in1k | 224x224 | 5.41 ms | 3.34 ms | 0.489 ms | 0.767 ms | 8.11 s | 30.5 s | 45 / 76 | 77 / 96 | 45 / 76 | 77 / 110 |
| repvit_m1_0.dist_300e_in1k | 224x224 | 5.29 ms | 3.33 ms | 0.503 ms | 0.786 ms | 8.20 s | 35.2 s | 48 / 78 | 80 / 118 | 48 / 78 | 78 / 110 |
| repvit_m1_0.dist_450e_in1k | 224x224 | 5.44 ms | 3.21 ms | 0.477 ms | 0.785 ms | 8.29 s | 40.2 s | 48 / 78 | 80 / 118 | 48 / 78 | 80 / 112 |
| repvit_m1_1.dist_300e_in1k | 224x224 | 5.03 ms | 6.12 ms | 0.470 ms | 0.765 ms | 7.72 s | 33.0 s | 51 / 62 | 83 / 102 | 51 / 64 | 83 / 96 |
| repvit_m1_1.dist_450e_in1k | 224x224 | 4.99 ms | 6.07 ms | 0.478 ms | 0.758 ms | 7.57 s | 33.4 s | 51 / 62 | 83 / 102 | 51 / 64 | 83 / 96 |
| repvit_m1_5.dist_300e_in1k | 224x224 | 8.55 ms | 4.89 ms | 0.766 ms | 1.19 ms | 10.2 s | 38.9 s | 62 / 94 | 94 / 114 | 62 / 94 | 96 / 130 |
| repvit_m1_5.dist_450e_in1k | 224x224 | 8.71 ms | 4.93 ms | 0.760 ms | 1.19 ms | 10.1 s | 33.8 s | 62 / 94 | 94 / 114 | 62 / 94 | 94 / 128 |
| repvit_m2_3.dist_300e_in1k | 224x224 | 11.1 ms | 6.51 ms | 1.03 ms | 1.66 ms | 11.9 s | 41.2 s | 80 / 128 | 112 / 148 | 79 / 130 | 149 / 200 |
| repvit_m2_3.dist_450e_in1k | 224x224 | 10.9 ms | 6.49 ms | 1.03 ms | 1.70 ms | 11.8 s | 40.8 s | 80 / 128 | 112 / 148 | 79 / 130 | 149 / 200 |
| resnet152.tv2_in1k | 224x224 | 8.51 ms | 4.55 ms | 1.09 ms | 1.85 ms | 8.82 s | 22.7 s | 153 / 200 | 185 / 240 | 152 / 224 | 282 / 334 |
| resnet18.tv_in1k | 224x224 | 1.44 ms | 1.04 ms | 0.220 ms | 0.472 ms | 4.31 s | 14.0 s | 59 / 88 | 91 / 108 | 58 / 110 | 89 / 120 |
| resnet50.tv2_in1k | 224x224 | 3.15 ms | 2.10 ms | 0.466 ms | 0.826 ms | 8.31 s | 24.0 s | 89 / 142 | 119 / 138 | 116 / 162 | 152 / 166 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 3.72 ms | 1.98 ms | 0.329 ms | 0.804 ms | 7.73 s | 16.9 s | 39 / 48 | 71 / 88 | 55 / 76 | 70 / 80 |
| shvit_s1.in1k | 224x224 | 6.75 ms | 3.11 ms | 0.510 ms | 0.805 ms | 10.2 s | 23.3 s | 47 / 58 | 78 / 98 | 47 / 62 | 77 / 90 |
| shvit_s2.in1k | 224x224 | 6.76 ms | 3.90 ms | 0.533 ms | 0.949 ms | 10.1 s | 24.0 s | 56 / 88 | 88 / 128 | 57 / 88 | 87 / 120 |
| shvit_s3.in1k | 224x224 | 8.11 ms | 3.64 ms | 0.575 ms | 1.01 ms | 10.9 s | 24.6 s | 62 / 112 | 94 / 132 | 62 / 112 | 92 / 144 |
| shvit_s4.in1k | 256x256 | 9.31 ms | 4.11 ms | 0.658 ms | 1.78 ms | 11.3 s | 26.3 s | 67 / 118 | 99 / 138 | 68 / 102 | 101 / 154 |
| squeezenet1_1.tv_in1k | 224x224 | 1.78 ms | 1.13 ms | 0.214 ms | 0.376 ms | 4.49 s | 9.10 s | 7 / 28 | 39 / 48 | 8 / 12 | 38 / 40 |
| starnet_s1.in1k | 224x224 | 3.88 ms | 2.41 ms | 0.774 ms | 0.683 ms | 6.23 s | 24.4 s | 41 / 52 | 73 / 92 | 41 / 56 | 73 / 86 |
| starnet_s2.in1k | 224x224 | 3.02 ms | 1.96 ms | 0.622 ms | 0.578 ms | 5.71 s | 24.7 s | 43 / 54 | 75 / 94 | 43 / 58 | 74 / 88 |
| starnet_s3.in1k | 224x224 | 4.19 ms | 2.60 ms | 0.878 ms | 0.715 ms | 6.03 s | 27.5 s | 47 / 60 | 79 / 100 | 47 / 64 | 80 / 96 |
| starnet_s4.in1k | 224x224 | 5.63 ms | 3.37 ms | 1.18 ms | 0.912 ms | 6.84 s | 28.0 s | 50 / 66 | 82 / 106 | 51 / 72 | 85 / 104 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 224x224 | 10.8 ms | 4.19 ms | 1.48 ms | 2.16 ms | 26.9 s | 64.2 s | 211 / 230 | 243 / 270 | 218 / 238 | 404 / 432 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 224x224 | 4.75 ms | 2.92 ms | 0.513 ms | 0.934 ms | 13.4 s | 60.7 s | 64 / 86 | 93 / 130 | 60 / 84 | 98 / 128 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 224x224 | 4.83 ms | 2.95 ms | 0.714 ms | 1.19 ms | 12.8 s | 50.3 s | 87 / 106 | 114 / 166 | 83 / 106 | 153 / 216 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 384x384 | 4.54 ms | 2.79 ms | 1.31 ms | 1.80 ms | 11.9 s | 65.7 s | 157 / 184 | 174 / 204 | 132 / 178 | 255 / 298 |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 512x512 | 4.46 ms | 3.14 ms | 2.13 ms | 3.13 ms | 11.9 s | 79.3 s | 310 / 344 | 314 / 362 | 237 / 258 | 499 / 518 |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 224x224 | 4.73 ms | 2.92 ms | 0.514 ms | 0.858 ms | 13.8 s | 54.0 s | 52 / 62 | 81 / 102 | 48 / 78 | 86 / 118 |
| vgg16.tv_in1k | 224x224 | 1.99 ms | 1.86 ms | 1.17 ms | 1.50 ms | 4.08 s | 15.9 s | 313 / 372 | 342 / 368 | 332 / 388 | 599 / 628 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 224x224 | 3.02 ms | 2.10 ms | 1.14 ms | 1.66 ms | 7.26 s | 28.6 s | 205 / 390 | 238 / 266 | 200 / 248 | 394 / 420 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 224x224 | 13.1 ms | 13.3 ms | 11.4 ms | 12.9 ms | 14.2 s | 60.7 s | 879 / 1708 | 909 / 944 | 876 / 926 | 1701 / 1752 |
| wd-convnext-tagger-v3 | 448x448 | 5.59 ms | 4.45 ms | 3.10 ms | 4.46 ms | 12.2 s | 39.7 s | 260 / 292 | 380 / 392 | 238 / 276 | 521 / 548 |
| wd-eva02-large-tagger-v3 | 448x448 | 18.3 ms | 15.7 ms | 12.4 ms | 14.1 ms | 22.4 s | 39.5 s | 669 / 684 | 714 / 724 | 673 / 806 | 1280 / 1306 |
| wd-vit-large-tagger-v3 | 448x448 | 12.9 ms | 13.3 ms | 11.2 ms | 12.6 ms | 13.5 s | 36.2 s | 669 / 682 | 713 / 742 | 667 / 724 | 1266 / 1290 |
| wd-vit-tagger-v3 | 448x448 | 4.01 ms | 5.33 ms | 3.32 ms | 4.06 ms | 9.55 s | 18.8 s | 235 / 242 | 282 / 302 | 234 / 284 | 421 / 446 |
| bisenet-face-parsing | 512x512 | 2.41 ms | 1.34 ms | 0.678 ms | 1.59 ms | 5.79 s | 12.9 s | 51 / 72 | 76 / 110 | 61 / 112 | 85 / 110 |
| deeplabv3-r101-d8-ade20k | 1024x2048 | 48.0 ms | 42.1 ms | 30.6 ms | 20.5 ms | 9.60 s | 25.7 s | 450 / 492 | 450 / 512 | 356 / 414 | 1764 / 1826 |
| deeplabv3plus-r101-d8-ade20k | 1024x2048 | 47.1 ms | 44.6 ms | 30.9 ms | 17.9 ms | 10.9 s | 23.3 s | 410 / 452 | 410 / 432 | 355 / 436 | 1683 / 1732 |
| dinov2-b-linear-ade20k | 1024x2048 | 33.3 ms | 35.8 ms | 49.5 ms | 20.1 ms | 10.8 s | 40.6 s | 765 / 884 | 1455 / 1504 | 771 / 884 | 3367 / 3424 |
| dinov2-l-linear-ade20k | 1024x2048 | 83.9 ms | 84.5 ms | 82.1 ms | 52.8 ms | 15.5 s | 43.8 s | 1181 / 1244 | 1870 / 1904 | 1189 / 1244 | 4257 / 4318 |
| dinov2-s-linear-ade20k | 1024x2048 | 18.3 ms | 20.7 ms | 40.3 ms | 11.8 ms | 10.6 s | 36.2 s | 642 / 696 | 1331 / 1376 | 645 / 676 | 3084 / 3130 |
| eomt-l-dinov2-ade20k | 1024x2048 | 51.2 ms | 49.4 ms | 54.3 ms | 35.1 ms | 16.5 s | 52.9 s | 1357 / 1504 | 1585 / 1624 | 1344 / 1504 | 3475 / 3500 |
| eomt-l-dinov3-ade20k | 1024x2048 | 55.0 ms | 50.0 ms | 61.9 ms | 38.9 ms | 22.1 s | 57.5 s | 1355 / 1484 | 1584 / 1624 | 1342 / 1484 | 3475 / 3522 |
| fcn-r101-d8-ade20k | 1024x2048 | 40.7 ms | 35.6 ms | 27.4 ms | 16.5 ms | 8.53 s | 22.2 s | 414 / 472 | 414 / 472 | 288 / 374 | 1656 / 1734 |
| hrnet-w48-fcn-ade20k | 1024x2048 | 27.3 ms | 23.9 ms | 20.9 ms | 11.2 ms | 35.7 s | 29.6 s | 421 / 470 | 421 / 448 | 265 / 344 | 1662 / 1712 |
| mask2former-swinl-ade20k | 1024x2048 | 104 ms | 94.2 ms | 55.6 ms | 60.2 ms | 101 s | 129 s | 1052 / 1218 | 1485 / 1514 | 999 / 1238 | 3051 / 3090 |
| oneformer-swinl-ade20k | 1024x2048 | 145 ms | 182 ms | 83.3 ms | 106 ms | 85.8 s | 140 s | 1768 / 1994 | 2372 / 2426 | 1768 / 1914 | 4524 / 4576 |
| pspnet-r101-d8-ade20k | 1024x2048 | 48.7 ms | 41.3 ms | 29.8 ms | 24.9 ms | 11.0 s | 44.1 s | 414 / 472 | 414 / 472 | 335 / 394 | 1859 / 1898 |
| sapiens-seg-0.3b-goliath | 1024x2048 | 132 ms | 118 ms | 156 ms | 92.6 ms | 12.5 s | 52.9 s | 1643 / 1762 | 1373 / 1462 | 1643 / 1762 | 2465 / 2526 |
| sapiens-seg-1b-goliath | 1024x2048 | 276 ms | 263 ms | 244 ms | 180 ms | 18.8 s | 84.9 s | 2894 / 3024 | 2986 / 3084 | 2894 / 3022 | 5694 / 5742 |
| segformer-b0-ade20k | 1024x2048 | 11.4 ms | 9.27 ms | 5.51 ms | 3.49 ms | 12.3 s | 60.9 s | 339 / 376 | 336 / 356 | 105 / 178 | 1480 / 1522 |
| segformer-b2-clothes | 512x512 | 8.13 ms | 4.38 ms | 2.68 ms | 3.61 ms | 17.6 s | 63.1 s | 156 / 168 | 182 / 228 | 133 / 170 | 312 / 328 |
| segformer-b5-ade20k | 1024x2048 | 38.9 ms | 31.4 ms | 22.8 ms | 25.4 ms | 33.8 s | 83.9 s | 444 / 490 | 543 / 630 | 294 / 432 | 2069 / 2110 |
| segformer-b5-face-parsing | 512x512 | 21.9 ms | 14.1 ms | 6.33 ms | 9.21 ms | 34.5 s | 84.7 s | 278 / 292 | 305 / 352 | 257 / 294 | 538 / 552 |
| segnext-mscan-b-ade20k | 1024x2048 | 28.3 ms | 40.7 ms | 23.3 ms | 12.1 ms | 24.1 s | 66.3 s | 376 / 424 | 491 / 526 | 192 / 226 | 1553 / 1622 |
| segnext-mscan-l-ade20k | 1024x2048 | 43.1 ms | 63.6 ms | 32.3 ms | 20.7 ms | 34.9 s | 82.2 s | 417 / 458 | 532 / 578 | 277 / 320 | 1634 / 1694 |
| segnext-mscan-s-ade20k | 1024x2048 | 18.5 ms | 24.9 ms | 18.0 ms | 8.04 ms | 17.2 s | 56.6 s | 350 / 402 | 465 / 482 | 164 / 202 | 1493 / 1544 |
| segnext-mscan-t-ade20k | 1024x2048 | 15.9 ms | 16.6 ms | 16.0 ms | 9.02 ms | 19.1 s | 59.6 s | 332 / 378 | 446 / 480 | 110 / 158 | 1476 / 1498 |
| ben2 | 1080x1920 | 115 ms | 103 ms | 58.2 ms | 105 ms | 70.8 s | 140 s | 1296 / 1376 | 813 / 916 | 1046 / 1118 | 1280 / 1326 |
| birefnet | 1080x1920 | 133 ms | 138 ms | 87.8 ms | 121 ms | 93.8 s | 201 s | 1165 / 1288 | 1874 / 1932 | 1145 / 1256 | 1753 / 1814 |
| birefnet-lite | 1080x1920 | 58.1 ms | 50.2 ms | 49.0 ms | 47.6 ms | 75.1 s | 156 s | 748 / 792 | 1469 / 1552 | 594 / 660 | 874 / 934 |
| inspyrenet | 1080x1920 | 107 ms | 85.8 ms | 51.1 ms | 107 ms | 70.8 s | 133 s | 1153 / 1468 | 913 / 1018 | 1153 / 1494 | 1181 / 1252 |
| inspyrenet-fast | 1080x1920 | 45.4 ms | 17.7 ms | 6.99 ms | 15.8 ms | 76.6 s | 138 s | 363 / 448 | 345 / 376 | 363 / 454 | 561 / 600 |
| isnet | 1080x1920 | 21.0 ms | 13.1 ms | 6.70 ms | 12.1 ms | 33.8 s | 34.4 s | 512 / 562 | 350 / 400 | 487 / 522 | 404 / 436 |
| isnet-general-use | 1080x1920 | 21.0 ms | 13.1 ms | 6.66 ms | 12.1 ms | 33.9 s | 32.3 s | 512 / 562 | 350 / 400 | 487 / 522 | 404 / 436 |
| matanyone | 1080x1920 | 966 ms | 635 ms | fails: BackendUnavailableError: matanyone@1 (portable) has no expor | 911 ms | 35.9 s | – | 2268 / 2726 | 1363 / 3452 | 2268 / 2706 | – |
| matanyone | 288x512 | – | – | 1476 ms, fp32 engine (frame and key encoder with eager video memory and alpha decoder, 100 frames) | – | – | 15.8 s | – | – | – | 474 / 542 |
| rmbg-2.0 | 1080x1920 | 133 ms | 110 ms | 88.1 ms | 121 ms | 82.9 s | 219 s | 1165 / 1288 | 1874 / 1932 | 1145 / 1296 | 1723 / 1784 |
| u2net | 1080x1920 | 9.77 ms | 9.09 ms | 3.07 ms | 5.18 ms | – | 36.0 s | 240 / 278 | 221 / 258 | 161 / 236 | 335 / 390 |
| u2net-human-seg | 1080x1920 | 10.0 ms | 6.18 ms | 3.08 ms | 5.14 ms | – | 34.7 s | 240 / 278 | 193 / 234 | 161 / 236 | 335 / 390 |
| u2netp | 1080x1920 | 15.1 ms | 5.73 ms | 1.87 ms | 2.92 ms | – | 26.1 s | 139 / 180 | 129 / 144 | 72 / 124 | 206 / 236 |
| vitmatte-b | 1080x1920 | 180 ms | 157 ms | 108 ms | 146 ms | 32.6 s | 48.8 s | 891 / 1036 | 1166 / 1472 | 747 / 832 | 1233 / 1262 |
| vitmatte-s | 1080x1920 | 102 ms | 81.9 ms | 54.6 ms | 84.1 ms | 31.5 s | 51.3 s | 1401 / 1524 | 1025 / 1284 | 1305 / 1404 | 876 / 920 |
| anyline | 1024x1024 | 21.0 ms | 11.6 ms | 12.9 ms (full line map only; memory estimated) | 18.8 ms | 40.0 s | 29.6 s | 362 / 384 | 306 / 346 | 362 / 384 | 528 / 542 |
| hed | 1024x1024 | 6.02 ms | 4.81 ms | 2.57 ms | 4.44 ms | 9.37 s | 11.6 s | 117 / 146 | 137 / 166 | 121 / 146 | 170 / 178 |
| lineart | 1024x1024 | 9.93 ms | 6.49 ms | 4.80 ms | 8.49 ms | 9.92 s | 10.3 s | 129 / 164 | 117 / 164 | 126 / 186 | 150 / 176 |
| lineart-anime | 1024x1024 | 3.86 ms | 2.80 ms | 1.54 ms | 3.62 ms | 10.3 s | 13.4 s | 161 / 206 | 180 / 206 | 161 / 186 | 304 / 314 |
| lineart-coarse | 1024x1024 | 9.92 ms | 6.49 ms | 4.80 ms | 8.51 ms | 9.66 s | 10.5 s | 129 / 164 | 117 / 164 | 126 / 166 | 150 / 176 |
| mlsd | 1024x1024 | 7.65 ms | 4.61 ms | 5.75 ms, fp32 engine (standalone CNN with eager resize and line decoding) | 7.03 ms | 11.4 s | – | 72 / 90 | 101 / 130 | 72 / 90 | 295 / 310 |
| pidinet | 1024x1024 | 6.59 ms | 5.66 ms | 4.50 ms | 6.37 ms | 12.4 s | 20.4 s | 186 / 228 | 166 / 190 | 186 / 228 | 187 / 200 |
| teed | 1024x1024 | 2.54 ms | 1.10 ms | 0.799 ms | 2.43 ms | 9.06 s | 12.5 s | 68 / 88 | 71 / 104 | 68 / 88 | 106 / 120 |
| clipseg | 1024x1024 | 4.46 ms | 3.19 ms | 1.75 ms (segmentation network on a normalized 352x352 image and precomputed text embedding only) | 2.76 ms | 9.92 s | 39.9 s | 328 / 356 | 348 / 374 | 335 / 362 | 464 / 480 |
| efficientvit-sam-l0 | 1024x1024 | 11.1 ms | 6.44 ms | 5.56 ms, fp32 engine (image encoder only) | 4.55 ms | 25.2 s | 19.9 s | 158 / 190 | 188 / 228 | 164 / 236 | 404 / 428 |
| efficientvit-sam-l1 | 1024x1024 | 14.4 ms | 11.3 ms | 7.30 ms, fp32 engine (image encoder only) | 5.47 ms | 24.8 s | 22.0 s | 183 / 234 | 212 / 252 | 191 / 264 | 517 / 578 |
| efficientvit-sam-l2 | 1024x1024 | 17.8 ms | 10.8 ms | 9.65 ms, fp32 engine (image encoder only) | 6.54 ms | 26.1 s | 24.1 s | 209 / 240 | 238 / 278 | 219 / 292 | 633 / 690 |
| efficientvit-sam-xl0 | 1024x1024 | 19.0 ms | 14.2 ms | 22.7 ms, fp32 engine (image encoder only) | 9.75 ms | 24.0 s | 29.8 s | 433 / 506 | 397 / 424 | 401 / 492 | 1288 / 1328 |
| efficientvit-sam-xl1 | 1024x1024 | 28.9 ms | 22.9 ms | 38.4 ms, fp32 engine (image encoder only) | 15.3 ms | 26.7 s | 36.6 s | 586 / 676 | 562 / 616 | 569 / 686 | 1894 / 1960 |
| insid3-vitl16 | 1024x1024 | 255 ms | 235 ms | 64.1 ms (image encoder only) | 157 ms | 17.2 s | 42.1 s | 986 / 1228 | 985 / 1206 | 987 / 1228 | 1339 / 1508 |
| mobile-sam | 1024x1024 | 10.5 ms | 6.22 ms | 4.99 ms (image encoder only) | 7.22 ms | 25.5 s | 60.9 s | 171 / 190 | 203 / 232 | 203 / 296 | 219 / 252 |
| sam-hq-vit-b | 1024x1024 | 48.3 ms | 39.4 ms | 28.0 ms (image encoder only) | 35.3 ms | 32.8 s | 39.0 s | 375 / 422 | 372 / 422 | 420 / 544 | 609 / 642 |
| sam-hq-vit-h | 1024x1024 | 213 ms | 187 ms | 125 ms (image encoder only) | 140 ms | 42.3 s | 69.1 s | 1488 / 1526 | 1476 / 1526 | 1528 / 1648 | 2790 / 2818 |
| sam-hq-vit-l | 1024x1024 | 114 ms | 97.0 ms | 68.3 ms (image encoder only) | 79.0 ms | 32.9 s | 53.3 s | 834 / 862 | 822 / 902 | 877 / 1006 | 1482 / 1538 |
| sam-vit-b | 1024x1024 | 46.6 ms | 40.9 ms | 27.3 ms (image encoder only) | 35.5 ms | 14.5 s | 36.4 s | 365 / 400 | 362 / 400 | 365 / 400 | 545 / 582 |
| sam-vit-h | 1024x1024 | 211 ms | 188 ms | 124 ms (image encoder only) | 140 ms | 42.6 s | 69.7 s | 1472 / 1504 | 1436 / 1484 | 1467 / 1484 | 2744 / 2784 |
| sam-vit-l | 1024x1024 | 113 ms | 97.2 ms | 67.5 ms (image encoder only) | 79.0 ms | 32.9 s | 53.4 s | 821 / 882 | 793 / 862 | 818 / 862 | 1453 / 1520 |
| sam2.1-hiera-base-plus | 1024x1024 | 31.9 ms | 26.4 ms | 16.7 ms (image encoder only) | 25.4 ms | 28.3 s | 43.7 s | 400 / 444 | 365 / 404 | 414 / 464 | 588 / 620 |
| sam2.1-hiera-large | 1024x1024 | 68.0 ms | 59.5 ms | 38.1 ms (image encoder only) | 49.1 ms | 38.9 s | 55.3 s | 730 / 768 | 678 / 748 | 748 / 808 | 1212 / 1232 |
| sam2.1-hiera-small | 1024x1024 | 18.8 ms | 14.9 ms | 9.93 ms (image encoder only) | 15.6 ms | 24.5 s | 40.2 s | 318 / 366 | 283 / 326 | 330 / 386 | 429 / 452 |
| sam2.1-hiera-tiny | 1024x1024 | 16.6 ms | 14.2 ms | 8.56 ms (image encoder only) | 13.9 ms | 13.4 s | 34.1 s | 304 / 382 | 270 / 322 | 316 / 362 | 414 / 446 |
| sam3 | 1024x1024 | 190 ms | 167 ms | 89.5 ms (image encoder only) | 107 ms | 66.6 s | 54.2 s | 1910 / 1984 | 1863 / 1944 | 1900 / 2044 | 2926 / 2944 |
| deim-l-coco | 800x1344 | 16.3 ms | 7.99 ms | 3.39 ms | 6.50 ms | 49.8 s | 58.1 s | 155 / 172 | 175 / 198 | 151 / 216 | 229 / 266 |
| deim-m-coco | 800x1344 | 13.8 ms | 6.73 ms | 2.54 ms | 5.08 ms | 39.1 s | 58.2 s | 129 / 142 | 149 / 166 | 129 / 184 | 204 / 232 |
| deim-n-coco | 800x1344 | 10.4 ms | 5.44 ms | 1.10 ms | 2.46 ms | 28.0 s | 65.6 s | 92 / 114 | 101 / 118 | 94 / 118 | 104 / 128 |
| deim-s-coco | 800x1344 | 11.3 ms | 5.63 ms | 1.68 ms | 3.73 ms | 31.3 s | 57.5 s | 108 / 128 | 131 / 152 | 108 / 170 | 163 / 176 |
| deim-x-coco | 800x1344 | 18.7 ms | 11.0 ms | 5.10 ms | 9.53 ms | 48.8 s | 66.5 s | 230 / 258 | 245 / 284 | 231 / 298 | 352 / 394 |
| detr-r101-coco | 800x1344 | 18.7 ms | 14.7 ms | 6.60 ms | 13.7 ms | 17.7 s | 35.0 s | 316 / 368 | 272 / 330 | 254 / 328 | 377 / 440 |
| detr-r50-coco | 800x1344 | 12.4 ms | 9.27 ms | 4.40 ms | 9.93 ms | 21.7 s | 33.4 s | 280 / 332 | 233 / 292 | 218 / 272 | 318 / 350 |
| dfine-l-coco | 800x1344 | 16.6 ms | 7.91 ms | 3.19 ms | 6.53 ms | 45.7 s | 58.7 s | 155 / 172 | 175 / 198 | 151 / 216 | 219 / 256 |
| dfine-m-coco | 800x1344 | 13.7 ms | 6.79 ms | 2.49 ms | 5.07 ms | 36.4 s | 54.6 s | 129 / 142 | 149 / 166 | 129 / 184 | 204 / 232 |
| dfine-n-coco | 800x1344 | 10.4 ms | 5.43 ms | 1.08 ms | 2.48 ms | 26.7 s | 64.4 s | 92 / 114 | 101 / 118 | 94 / 118 | 104 / 128 |
| dfine-s-coco | 800x1344 | 11.4 ms | 5.63 ms | 1.65 ms | 3.75 ms | 31.2 s | 51.4 s | 108 / 128 | 131 / 152 | 108 / 170 | 147 / 160 |
| dfine-x-coco | 800x1344 | 18.6 ms | 11.0 ms | 5.64 ms | 9.59 ms | 47.7 s | 107 s | 230 / 258 | 245 / 284 | 231 / 298 | 342 / 374 |
| faster-rcnn-r50-fpn-3x-coco | 800x1344 | 22.8 ms | 18.9 ms | 5.35 ms (backbone + FPN only) | 19.5 ms | 26.0 s | 15.6 s | 390 / 490 | 413 / 438 | 391 / 450 | 439 / 494 |
| grounding-dino-base | 800x1344 | 160 ms | 131 ms | 159 ms, fp32 engine (fixed 80-class text prompt) | 109 ms | 98.4 s | 130 s | 1525 / 1624 | 2669 / 2800 | 1156 / 1364 | 1153 / 1160 |
| grounding-dino-base | 320x320 | – | – | 17.6 ms, fp32 engine (fixed 80-class text prompt) | – | – | 119 s | – | – | – | 609 / 628 |
| grounding-dino-tiny | 800x1344 | 138 ms | 111 ms | 116 ms, fp32 engine (fixed 80-class text prompt) | 86.9 ms | 72.5 s | 104 s | 1404 / 1468 | 2550 / 2664 | 1034 / 1188 | 883 / 888 |
| grounding-dino-tiny | 320x320 | 69.0 ms | – | 13.6 ms, fp32 engine (network with fixed COCO prompts, standalone build) | 55.4 ms | – | – | 841 / 948 | – | 841 / 948 | 387 / 406 |
| lw-detr-large-coco | 800x1344 | 10.2 ms | 5.62 ms | 3.45 ms | 5.64 ms | 29.6 s | 56.0 s | 204 / 230 | 195 / 208 | 192 / 254 | 270 / 296 |
| lw-detr-medium-coco | 800x1344 | 8.75 ms | 3.74 ms | 2.30 ms | 3.78 ms | 22.4 s | 63.3 s | 131 / 156 | 147 / 174 | 131 / 178 | 185 / 212 |
| lw-detr-small-coco | 800x1344 | 8.82 ms | 3.63 ms | 1.36 ms | 2.80 ms | 23.0 s | 63.1 s | 97 / 116 | 116 / 134 | 99 / 140 | 121 / 130 |
| lw-detr-tiny-coco | 800x1344 | 7.45 ms | 2.98 ms | 0.999 ms | 2.20 ms | 21.6 s | 60.3 s | 92 / 112 | 112 / 130 | 94 / 134 | 116 / 128 |
| lw-detr-xlarge-coco | 800x1344 | 12.1 ms | 10.7 ms | 6.58 ms | 9.39 ms | 30.5 s | 57.5 s | 348 / 360 | 339 / 358 | 336 / 382 | 544 / 566 |
| mask-rcnn-r50-fpn-3x-coco | 800x1344 | 24.6 ms | 21.5 ms | 5.34 ms (backbone + FPN only) | 21.0 ms | 29.3 s | 18.3 s | 352 / 440 | 333 / 358 | 353 / 400 | 444 / 496 |
| mm-grounding-dino-base | 800x1344 | 160 ms | 132 ms | 158 ms, fp32 engine (fixed 80-class text prompt) | 109 ms | 105 s | 125 s | 1525 / 1624 | 2669 / 2800 | 1157 / 1366 | 1153 / 1160 |
| mm-grounding-dino-base | 320x320 | – | – | 17.7 ms, fp32 engine (fixed 80-class text prompt) | – | – | 134 s | – | – | – | 641 / 662 |
| mm-grounding-dino-tiny | 800x1344 | 138 ms | 111 ms | 116 ms, fp32 engine (fixed 80-class text prompt) | 87.2 ms | 71.2 s | 104 s | 1405 / 1470 | 2555 / 2666 | 1035 / 1188 | 883 / 890 |
| mm-grounding-dino-tiny | 320x320 | 69.2 ms | – | 13.8 ms, fp32 engine (network with fixed COCO prompts, standalone build) | 55.5 ms | – | – | 842 / 950 | – | 842 / 950 | 387 / 408 |
| owlv2-b16 | 800x1344 | 31.8 ms | 28.5 ms | 18.4 ms | 23.7 ms | 10.6 s | 25.9 s | 460 / 510 | 487 / 530 | 461 / 612 | 655 / 726 |
| owlv2-l14 | 800x1344 | 154 ms | 141 ms | 95.9 ms | 110 ms | 12.4 s | 33.8 s | 1097 / 1144 | 1190 / 1242 | 1027 / 1184 | 1786 / 1820 |
| rf-detr-l-coco | 800x1344 | 10.6 ms | 5.32 ms | 2.87 ms | 5.30 ms | 28.7 s | 65.7 s | 149 / 184 | 164 / 182 | 149 / 206 | 203 / 222 |
| rf-detr-m-coco | 800x1344 | 10.6 ms | 4.92 ms | 2.10 ms | 3.86 ms | 27.8 s | 55.3 s | 136 / 156 | 156 / 166 | 136 / 178 | 194 / 216 |
| rf-detr-n-coco | 800x1344 | 9.39 ms | 4.58 ms | 1.15 ms | 2.39 ms | 25.8 s | 60.1 s | 117 / 140 | 141 / 156 | 117 / 142 | 179 / 208 |
| rf-detr-s-coco | 800x1344 | 10.0 ms | 4.56 ms | 1.83 ms | 3.33 ms | 24.4 s | 52.5 s | 128 / 148 | 150 / 162 | 128 / 172 | 188 / 214 |
| rf-detr-seg-2xl-coco | 800x1344 | 26.0 ms | 16.9 ms | 8.62 ms | 16.0 ms | 44.6 s | 69.2 s | 207 / 228 | 245 / 308 | 209 / 312 | 334 / 366 |
| rf-detr-seg-l-coco | 800x1344 | 14.1 ms | 7.08 ms | 3.30 ms | 6.60 ms | 31.5 s | 68.5 s | 150 / 168 | 190 / 208 | 150 / 208 | 218 / 234 |
| rf-detr-seg-m-coco | 800x1344 | 13.3 ms | 6.86 ms | 2.50 ms | 5.31 ms | 32.1 s | 72.4 s | 141 / 180 | 172 / 194 | 141 / 202 | 208 / 228 |
| rf-detr-seg-n-coco | 800x1344 | 11.7 ms | 6.59 ms | 1.52 ms | 3.68 ms | 31.2 s | 54.0 s | 125 / 148 | 158 / 166 | 125 / 170 | 189 / 218 |
| rf-detr-seg-s-coco | 800x1344 | 12.0 ms | 6.55 ms | 2.00 ms | 4.28 ms | 29.8 s | 63.6 s | 132 / 154 | 162 / 190 | 132 / 178 | 197 / 224 |
| rf-detr-seg-xl-coco | 800x1344 | 18.1 ms | 10.8 ms | 5.20 ms | 10.4 ms | 35.3 s | 92.5 s | 177 / 206 | 212 / 246 | 178 / 268 | 302 / 326 |
| rtdetrv2-l-coco | 800x1344 | 12.4 ms | 7.29 ms | 3.32 ms | 6.53 ms | 41.6 s | 48.3 s | 180 / 200 | 204 / 238 | 180 / 262 | 255 / 286 |
| rtdetrv2-m-coco | 800x1344 | 9.96 ms | 5.35 ms | 2.42 ms | 5.19 ms | 31.5 s | 43.7 s | 159 / 190 | 177 / 206 | 159 / 212 | 199 / 222 |
| rtdetrv2-ms-coco | 800x1344 | 11.1 ms | 5.82 ms | 2.62 ms | 5.40 ms | 28.3 s | 44.2 s | 169 / 202 | 193 / 220 | 169 / 242 | 244 / 288 |
| rtdetrv2-s-coco | 800x1344 | 9.60 ms | 4.22 ms | 1.81 ms | 4.10 ms | 37.0 s | 42.2 s | 137 / 166 | 152 / 182 | 137 / 208 | 187 / 218 |
| rtdetrv2-x-coco | 800x1344 | 14.9 ms | 10.9 ms | 5.23 ms | 9.17 ms | 44.2 s | 54.6 s | 247 / 278 | 269 / 296 | 247 / 320 | 381 / 430 |
| rtmdet-l-coco | 800x1344 | 9.59 ms | 10.6 ms | 3.59 ms | 5.52 ms | 17.2 s | 25.0 s | 170 / 210 | 178 / 226 | 149 / 286 | 283 / 318 |
| rtmdet-m-coco | 800x1344 | 7.50 ms | 4.78 ms | 2.39 ms | 4.27 ms | 15.7 s | 22.4 s | 92 / 112 | 115 / 130 | 75 / 182 | 156 / 212 |
| rtmdet-m-person | 800x1344 | 7.18 ms | 4.44 ms | 2.38 ms | 3.89 ms | 15.5 s | 24.8 s | 92 / 112 | 115 / 128 | 75 / 182 | 153 / 190 |
| rtmdet-nano-person | 800x1344 | 7.39 ms | 4.23 ms | 0.769 ms | 1.82 ms | 16.3 s | 22.6 s | 9 / 28 | 48 / 64 | 9 / 52 | 49 / 58 |
| rtmdet-s-coco | 800x1344 | 6.13 ms | 3.56 ms | 1.19 ms | 2.75 ms | 14.3 s | 20.1 s | 49 / 82 | 76 / 98 | 38 / 94 | 93 / 130 |
| rtmdet-tiny-coco | 800x1344 | 5.74 ms | 3.43 ms | 0.962 ms | 2.46 ms | 13.7 s | 19.3 s | 36 / 56 | 64 / 94 | 29 / 66 | 80 / 122 |
| rtmdet-x-coco | 800x1344 | 13.1 ms | 11.8 ms | 6.17 ms | 8.30 ms | 18.6 s | 27.8 s | 254 / 298 | 271 / 292 | 229 / 378 | 471 / 504 |
| yolox-l | 800x1344 | 7.66 ms | 6.15 ms | 3.10 ms | 5.02 ms | 15.3 s | 21.2 s | 151 / 170 | 173 / 208 | 139 / 250 | 279 / 324 |
| yolox-m | 800x1344 | 6.03 ms | 3.70 ms | 2.03 ms | 3.44 ms | 13.4 s | 20.1 s | 87 / 112 | 111 / 132 | 77 / 184 | 152 / 194 |
| yolox-s | 800x1344 | 5.27 ms | 2.95 ms | 1.02 ms | 2.43 ms | 12.7 s | 17.2 s | 46 / 60 | 74 / 96 | 40 / 94 | 86 / 126 |
| yolox-tiny | 800x1344 | 5.25 ms | 2.87 ms | 0.561 ms | 1.79 ms | 13.1 s | 23.7 s | 22 / 56 | 58 / 94 | 22 / 68 | 62 / 90 |
| yolox-x | 800x1344 | 11.6 ms | 11.3 ms | 5.28 ms | 7.56 ms | 15.8 s | 24.1 s | 246 / 280 | 265 / 296 | 230 / 378 | 432 / 478 |
| parseq | 32x128 | 3.62 ms | 8.80 ms | 1.67 ms | 3.62 ms | 19.9 s | 114 s | 79 / 100 | 111 / 118 | 79 / 100 | 148 / 168 |
| parseq-tiny | 32x128 | 2.10 ms | 1.54 ms | 1.67 ms | 2.10 ms | 18.0 s | 102 s | 44 / 54 | 76 / 94 | 44 / 54 | 82 / 90 |
| ppocrv5-mobile | 1080x1920 | 153 ms | 115 ms | 2.68 ms | 154 ms | 75.0 s | 156 s | 289 / 448 | 292 / 448 | 500 / 1016 | 241 / 268 |
| ppocrv5-server | 1080x1920 | 279 ms | 277 ms | 25.1 ms | 228 ms | 105 s | 40.6 s | 615 / 752 | 958 / 1132 | 730 / 1648 | 1131 / 1146 |
| dwpose-l | 1080x1920 | 10.3 ms | 5.75 ms | 3.33 ms (network on person crops only) | 6.20 ms | 14.8 s | 20.0 s | 211 / 260 | 197 / 216 | 163 / 266 | 277 / 294 |
| dwpose-l-384 | 1080x1920 | 21.4 ms | 11.9 ms | 7.46 ms (network on person crops only) | 12.7 ms | 14.9 s | 20.6 s | 325 / 418 | 253 / 296 | 229 / 404 | 347 / 370 |
| dwpose-m | 1080x1920 | 7.31 ms | 3.80 ms | 2.13 ms (network on person crops only) | 4.86 ms | 12.9 s | 18.9 s | 153 / 196 | 156 / 174 | 133 / 200 | 203 / 228 |
| dwpose-s | 1080x1920 | 5.16 ms | 2.27 ms | 1.17 ms (network on person crops only) | 3.35 ms | 11.8 s | 16.7 s | 124 / 154 | 135 / 150 | 115 / 178 | 149 / 170 |
| dwpose-t | 1080x1920 | 7.84 ms | 3.63 ms | 0.940 ms (network on person crops only) | 2.52 ms | 17.4 s | 16.1 s | 104 / 132 | 129 / 148 | 106 / 152 | 136 / 160 |
| hrnet-w32 | 1080x1920 | 20.3 ms | 9.94 ms | 5.18 ms (network on person crops only) | 9.75 ms | 21.1 s | 25.5 s | 195 / 238 | 178 / 214 | 155 / 194 | 216 / 254 |
| hrnet-w32-384 | 1080x1920 | 39.2 ms | 19.6 ms | 10.7 ms (network on person crops only) | 19.7 ms | 21.5 s | 26.3 s | 341 / 414 | 211 / 234 | 250 / 294 | 296 / 320 |
| hrnet-w48 | 1080x1920 | 32.0 ms | 21.2 ms | 9.46 ms (network on person crops only) | 15.6 ms | 21.3 s | 26.9 s | 262 / 302 | 245 / 282 | 222 / 262 | 346 / 388 |
| hrnet-w48-384 | 1080x1920 | 70.7 ms | 46.0 ms | 22.3 ms (network on person crops only) | 34.2 ms | 22.6 s | 28.3 s | 408 / 458 | 278 / 300 | 317 / 360 | 427 / 454 |
| retinaface-mobilenet0.25 | 1080x1920 | 3.44 ms | 2.16 ms | 0.982 ms (network only) | 2.74 ms | 8.87 s | 53.5 s | 86 / 108 | 125 / 148 | 105 / 128 | 133 / 144 |
| retinaface-resnet50 | 1080x1920 | 24.3 ms | 16.5 ms | 9.10 ms (network only) | 11.7 ms | 9.56 s | 20.4 s | 405 / 504 | 374 / 444 | 379 / 464 | 451 / 462 |
| rf-detr-keypoint-preview | 1080x1920 | 17.2 ms | 7.82 ms | 4.28 ms (network on person crops only) | 8.98 ms | – | 67.2 s | 176 / 202 | 160 / 180 | 176 / 224 | 264 / 294 |
| rtmo-l | 1080x1920 | 10.0 ms | 7.07 ms | 2.90 ms (network on person crops only) | 8.30 ms | 18.9 s | 39.6 s | 211 / 242 | 192 / 224 | 208 / 228 | 277 / 306 |
| rtmo-m | 1080x1920 | 7.44 ms | 4.99 ms | 2.03 ms (network on person crops only) | 6.85 ms | 18.0 s | 39.4 s | 168 / 192 | 146 / 170 | 166 / 212 | 194 / 232 |
| rtmo-s | 1080x1920 | 6.72 ms | 6.74 ms | 1.21 ms (network on person crops only) | 3.93 ms | 21.1 s | 36.7 s | 142 / 176 | 121 / 136 | 105 / 166 | 133 / 164 |
| rtmo-t | 1080x1920 | 7.82 ms | 6.68 ms | 0.886 ms (network on person crops only) | 2.96 ms | 20.7 s | 36.8 s | 122 / 140 | 111 / 134 | 82 / 140 | 112 / 150 |
| rtmpose-l | 1080x1920 | 9.04 ms | 5.32 ms | 3.20 ms (network on person crops only) | 5.70 ms | 14.2 s | 19.4 s | 200 / 256 | 185 / 194 | 152 / 244 | 230 / 266 |
| rtmpose-l-384 | 1080x1920 | 19.4 ms | 10.7 ms | 7.08 ms (network on person crops only) | 11.4 ms | 15.0 s | 20.3 s | 314 / 416 | 242 / 274 | 217 / 390 | 297 / 322 |
| rtmpose-l-halpe26 | 1080x1920 | 9.10 ms | 5.34 ms | 3.20 ms (network on person crops only) | 5.69 ms | 14.3 s | 19.6 s | 201 / 256 | 186 / 194 | 152 / 244 | 231 / 266 |
| rtmpose-l-halpe26-384 | 1080x1920 | 19.5 ms | 10.6 ms | 7.08 ms (network on person crops only) | 11.5 ms | 15.1 s | 20.6 s | 315 / 416 | 243 / 276 | 218 / 410 | 299 / 322 |
| rtmpose-m | 1080x1920 | 6.36 ms | 3.42 ms | 2.02 ms (network on person crops only) | 4.50 ms | 12.7 s | 17.8 s | 145 / 192 | 147 / 172 | 125 / 200 | 157 / 176 |
| rtmpose-m-384 | 1080x1920 | 13.3 ms | 7.09 ms | 4.33 ms (network on person crops only) | 9.23 ms | 13.6 s | 18.6 s | 224 / 332 | 190 / 212 | 178 / 320 | 212 / 242 |
| rtmpose-m-ap10k | 1080x1920 | 8.02 ms | 4.54 ms | 2.57 ms (network on person crops only) | 5.42 ms | 13.0 s | 18.2 s | 166 / 214 | 176 / 192 | 139 / 200 | 172 / 188 |
| rtmpose-m-face6 | 1080x1920 | 8.75 ms | 4.79 ms | 2.67 ms (network on person crops only) | 5.77 ms | 12.8 s | 18.4 s | 173 / 216 | 182 / 192 | 145 / 222 | 216 / 240 |
| rtmpose-m-halpe26 | 1080x1920 | 6.42 ms | 3.44 ms | 2.02 ms (network on person crops only) | 4.53 ms | 12.9 s | 18.0 s | 146 / 192 | 148 / 172 | 125 / 180 | 160 / 180 |
| rtmpose-m-halpe26-384 | 1080x1920 | 13.4 ms | 7.12 ms | 4.34 ms (network on person crops only) | 9.28 ms | 13.7 s | 18.9 s | 225 / 332 | 190 / 212 | 179 / 320 | 212 / 238 |
| rtmpose-m-hand5 | 1080x1920 | 8.05 ms | 4.51 ms | 2.58 ms (network on person crops only) | 5.46 ms | 12.9 s | 18.3 s | 166 / 214 | 176 / 192 | 139 / 200 | 174 / 192 |
| rtmpose-s | 1080x1920 | 4.95 ms | 2.07 ms | 1.06 ms (network on person crops only) | 3.08 ms | 11.5 s | 16.5 s | 118 / 152 | 129 / 150 | 109 / 158 | 137 / 170 |
| rtmpose-s-face6 | 1080x1920 | 5.04 ms | 2.69 ms | 1.28 ms (network on person crops only) | 3.74 ms | 11.9 s | 16.9 s | 140 / 174 | 164 / 190 | 128 / 202 | 158 / 178 |
| rtmpose-s-halpe26 | 1080x1920 | 4.99 ms | 2.11 ms | 1.07 ms (network on person crops only) | 3.05 ms | 11.7 s | 16.6 s | 119 / 152 | 130 / 150 | 110 / 178 | 138 / 170 |
| rtmpose-t | 1080x1920 | 7.79 ms | 4.07 ms | 0.866 ms (network on person crops only) | 2.25 ms | 16.4 s | 15.9 s | 100 / 128 | 125 / 148 | 91 / 156 | 127 / 164 |
| rtmpose-t-face6 | 1080x1920 | 4.53 ms | 2.23 ms | 1.04 ms (network on person crops only) | 3.61 ms | 11.5 s | – | 116 / 152 | 159 / 168 | 123 / 180 | 83 / 106 |
| rtmpose-t-halpe26 | 1080x1920 | 4.61 ms | 1.94 ms | 0.871 ms (network on person crops only) | 2.90 ms | 13.5 s | 17.2 s | 100 / 130 | 125 / 128 | 105 / 156 | 128 / 164 |
| rtmpose-x-halpe26-384 | 1080x1920 | 33.8 ms | 21.1 ms | 13.3 ms (network on person crops only) | 19.1 ms | 17.4 s | 23.0 s | 385 / 478 | 262 / 298 | 280 / 448 | 430 / 456 |
| rtmw-l | 1080x1920 | 15.3 ms | 9.34 ms | 5.17 ms (network on person crops only) | 9.99 ms | 18.8 s | 24.9 s | 256 / 308 | 242 / 264 | 217 / 312 | 354 / 396 |
| rtmw-l-384 | 1080x1920 | 31.8 ms | 19.5 ms | 11.2 ms (network on person crops only) | 20.6 ms | 19.8 s | 26.8 s | 371 / 468 | 298 / 344 | 276 / 458 | 424 / 452 |
| rtmw-m | 1080x1920 | 10.9 ms | 6.23 ms | 3.34 ms (network on person crops only) | 7.54 ms | 18.7 s | 23.1 s | 181 / 244 | 183 / 202 | 164 / 266 | 231 / 256 |
| rtmw-x | 1080x1920 | 24.0 ms | 15.3 ms | 8.47 ms (network on person crops only) | 14.4 ms | 20.4 s | 27.0 s | 335 / 390 | 320 / 328 | 289 / 392 | 497 / 514 |
| rtmw-x-384 | 1080x1920 | 51.7 ms | 35.8 ms | 19.3 ms (network on person crops only) | 49.1 ms | 31.1 s | 29.8 s | 466 / 552 | 361 / 388 | 496 / 612 | 581 / 624 |
| vitpose-base | 1080x1920 | 17.8 ms | 16.4 ms | 8.88 ms (network on person crops only) | 12.0 ms | 10.5 s | 35.8 s | 299 / 322 | 597 / 622 | 274 / 364 | 473 / 516 |
| vitpose-base-simple | 1080x1920 | 18.1 ms | 16.0 ms | 9.36 ms (network on person crops only) | 12.1 ms | 11.8 s | 35.0 s | 312 / 342 | 590 / 622 | 295 / 344 | 499 / 550 |
| vitpose-plus-base | 1080x1920 | 17.9 ms | 16.4 ms | 8.87 ms (network on person crops only) | 11.9 ms | 10.6 s | 35.0 s | 299 / 322 | 597 / 622 | 274 / 344 | 473 / 516 |
| vitpose-plus-huge | 1080x1920 | 102 ms | 92.9 ms | 58.7 ms (network on person crops only) | 62.5 ms | 21.2 s | 47.3 s | 1382 / 1424 | 1641 / 1682 | 1351 / 1406 | 2595 / 2654 |
| vitpose-plus-large | 1080x1920 | 53.7 ms | 46.9 ms | 29.3 ms (network on person crops only) | 33.2 ms | 17.9 s | 39.0 s | 1014 / 1062 | 1014 / 1062 | 706 / 764 | 1312 / 1338 |
| vitpose-plus-small | 1080x1920 | 7.83 ms | 7.54 ms | 3.39 ms (network on person crops only) | 6.05 ms | 10.6 s | 37.5 s | 164 / 196 | 472 / 496 | 145 / 198 | 214 / 256 |
| yunet-n | 1080x1920 | 2.95 ms | 1.79 ms | 0.794 ms (network only) | 2.22 ms | 8.40 s | 12.1 s | 96 / 124 | 124 / 144 | 84 / 104 | 119 / 150 |
| yunet-s | 1080x1920 | 2.67 ms | 1.65 ms | 0.680 ms (network only) | 2.00 ms | 7.92 s | 13.4 s | 96 / 124 | 124 / 144 | 84 / 104 | 114 / 146 |
| c-radio-v3-b | 224x224 | 3.04 ms | 7.33 ms | 1.19 ms | 1.36 ms | 8.29 s | 37.4 s | 217 / 246 | 278 / 306 | 214 / 248 | 427 / 454 |
| c-radio-v3-g | 224x224 | 14.6 ms | 25.5 ms | 10.3 ms | 10.7 ms | 16.5 s | 57.1 s | 2266 / 2308 | 2329 / 2368 | 2263 / 2310 | 4483 / 4530 |
| c-radio-v3-h | 224x224 | 8.90 ms | 18.1 ms | 6.62 ms | 7.43 ms | 14.7 s | 44.7 s | 1289 / 1308 | 1350 / 1386 | 1286 / 1308 | 2558 / 2586 |
| c-radio-v3-l | 224x224 | 5.53 ms | 11.8 ms | 3.06 ms | 3.28 ms | 11.3 s | 41.8 s | 648 / 686 | 710 / 746 | 645 / 706 | 1297 / 1350 |
| c-radio-v4-h | 224x224 | 8.91 ms | 17.9 ms | 6.64 ms | 7.40 ms | 12.9 s | 51.2 s | 1286 / 1308 | 1351 / 1388 | 1286 / 1308 | 2558 / 2586 |
| c-radio-v4-so400m | 224x224 | 6.22 ms | 14.2 ms | 3.75 ms | 4.14 ms | 11.5 s | 49.0 s | 861 / 906 | 927 / 966 | 861 / 908 | 1695 / 1740 |
| clip-b16-laion2b | 224x224 | 3.09 ms | 2.41 ms | 1.12 ms | 1.36 ms | 6.49 s | 24.3 s | 202 / 226 | 237 / 266 | 203 / 230 | 394 / 422 |
| clip-b16-openai | 224x224 | 3.35 ms | 2.39 ms | 1.11 ms | 1.36 ms | 6.37 s | 14.6 s | 204 / 226 | 237 / 266 | 203 / 250 | 394 / 422 |
| clip-h14-378-dfn5b | 224x224 | 11.4 ms | 11.8 ms | 7.85 ms | 8.24 ms | 13.6 s | 58.9 s | 1254 / 1288 | 1290 / 1328 | 1251 / 1312 | 2468 / 2510 |
| clip-h14-dfn5b | 224x224 | 11.4 ms | 11.5 ms | 7.76 ms | 8.54 ms | 13.0 s | 36.3 s | 1252 / 1268 | 1289 / 1310 | 1250 / 1292 | 2465 / 2490 |
| clip-h14-laion2b | 224x224 | 11.2 ms | 11.6 ms | 7.84 ms | 8.27 ms | 14.3 s | 40.6 s | 1249 / 1268 | 1289 / 1310 | 1250 / 1272 | 2467 / 2492 |
| clip-l14-336-openai | 224x224 | 6.16 ms | 6.36 ms | 4.25 ms | 4.76 ms | 10.7 s | 33.1 s | 625 / 646 | 658 / 686 | 622 / 668 | 1235 / 1264 |
| clip-l14-datacompxl | 224x224 | 5.99 ms | 6.24 ms | 4.19 ms | 4.76 ms | 10.5 s | 32.1 s | 621 / 666 | 658 / 686 | 622 / 670 | 1232 / 1282 |
| clip-l14-dfn2b | 224x224 | 6.16 ms | 6.16 ms | 4.14 ms | 4.76 ms | 11.4 s | 26.6 s | 623 / 666 | 658 / 686 | 622 / 690 | 1230 / 1280 |
| clip-l14-laion2b | 224x224 | 5.98 ms | 6.19 ms | 4.19 ms | 4.76 ms | 10.4 s | 32.0 s | 621 / 666 | 658 / 686 | 622 / 670 | 1232 / 1282 |
| clip-l14-openai | 224x224 | 6.16 ms | 6.17 ms | 4.14 ms | 4.76 ms | 10.2 s | 23.0 s | 623 / 666 | 658 / 686 | 622 / 690 | 1230 / 1280 |
| convnext_base.fb_in22k | 224x224 | 5.80 ms | 3.21 ms | 1.51 ms | 1.96 ms | 10.2 s | 29.9 s | 209 / 214 | 321 / 334 | 204 / 216 | 480 / 494 |
| convnext_large.fb_in22k | 224x224 | 5.82 ms | 3.34 ms | 2.50 ms | 3.23 ms | 9.87 s | 46.7 s | 420 / 430 | 639 / 650 | 413 / 434 | 983 / 1006 |
| convnext_small.fb_in22k | 224x224 | 5.92 ms | 3.36 ms | 1.16 ms | 1.79 ms | 9.51 s | 35.1 s | 134 / 150 | 210 / 230 | 130 / 150 | 277 / 292 |
| convnext_tiny.fb_in22k | 224x224 | 3.26 ms | 2.08 ms | 0.726 ms | 1.10 ms | 12.0 s | 33.9 s | 93 / 108 | 148 / 170 | 88 / 110 | 183 / 200 |
| dinov2-vitb14 | 224x224 | 2.92 ms | 2.20 ms | 1.30 ms | 1.50 ms | 9.52 s | 25.2 s | 205 / 226 | 240 / 266 | 208 / 228 | 367 / 384 |
| dinov2-vitb14-reg | 224x224 | 2.94 ms | 2.30 ms | 1.33 ms | 1.52 ms | 6.12 s | 29.2 s | 207 / 226 | 240 / 246 | 208 / 228 | 399 / 402 |
| dinov2-vitg14 | 224x224 | 18.9 ms | 18.3 ms | 12.0 ms | 12.8 ms | 19.0 s | 46.5 s | 2216 / 2230 | 2267 / 2290 | 2222 / 2250 | 4392 / 4410 |
| dinov2-vitg14-reg | 224x224 | 18.7 ms | 18.3 ms | 12.0 ms | 12.7 ms | 15.4 s | 56.9 s | 2217 / 2228 | 2267 / 2290 | 2222 / 2250 | 4400 / 4418 |
| dinov2-vitl14 | 224x224 | 6.13 ms | 6.03 ms | 4.25 ms | 4.62 ms | 13.2 s | 33.5 s | 622 / 644 | 661 / 666 | 627 / 648 | 1235 / 1262 |
| dinov2-vitl14-reg | 224x224 | 6.38 ms | 6.17 ms | 4.27 ms | 4.60 ms | 9.55 s | 42.0 s | 624 / 646 | 661 / 666 | 627 / 648 | 1237 / 1242 |
| dinov2-vits14 | 224x224 | 2.95 ms | 2.02 ms | 0.601 ms | 0.827 ms | 9.99 s | 22.9 s | 79 / 98 | 111 / 140 | 80 / 122 | 141 / 160 |
| dinov2-vits14-reg | 224x224 | 2.98 ms | 2.19 ms | 0.614 ms | 0.832 ms | 6.11 s | 32.1 s | 80 / 98 | 111 / 118 | 80 / 102 | 145 / 164 |
| dinov3-convnext-base | 224x224 | 5.71 ms | 3.41 ms | 1.45 ms | 2.19 ms | 11.4 s | 29.6 s | 208 / 232 | 241 / 252 | 174 / 214 | 400 / 430 |
| dinov3-convnext-large | 224x224 | 5.45 ms | 4.30 ms | 2.50 ms | 3.61 ms | 11.3 s | 30.0 s | 419 / 430 | 459 / 470 | 409 / 452 | 804 / 824 |
| dinov3-convnext-small | 224x224 | 5.42 ms | 3.41 ms | 1.17 ms | 1.98 ms | 10.6 s | 25.9 s | 134 / 150 | 165 / 188 | 100 / 132 | 232 / 252 |
| dinov3-convnext-tiny | 224x224 | 3.04 ms | 1.96 ms | 0.681 ms | 1.22 ms | 7.76 s | 22.8 s | 93 / 108 | 123 / 148 | 59 / 110 | 126 / 158 |
| dinov3-vitb16 | 224x224 | 4.76 ms | 2.85 ms | 1.14 ms | 1.36 ms | 16.4 s | 25.0 s | 200 / 206 | 236 / 244 | 167 / 210 | 393 / 480 |
| dinov3-vith16plus | 224x224 | 14.1 ms | 13.6 ms | 7.86 ms | 8.85 ms | 29.7 s | 99.2 s | 1646 / 1666 | 1699 / 1726 | 1643 / 1670 | 3280 / 3628 |
| dinov3-vitl16 | 224x224 | 10.2 ms | 6.88 ms | 2.98 ms | 3.26 ms | 25.9 s | 34.4 s | 616 / 626 | 655 / 666 | 615 / 628 | 1226 / 1418 |
| dinov3-vits16 | 224x224 | 5.19 ms | 3.24 ms | 0.564 ms | 0.799 ms | 13.2 s | 25.0 s | 76 / 84 | 109 / 124 | 44 / 68 | 107 / 120 |
| pe-core-b16-224 | 224x224 | 4.97 ms | 2.50 ms | 1.17 ms | 1.47 ms | 8.71 s | 25.8 s | 218 / 266 | 251 / 286 | 218 / 268 | 409 / 460 |
| pe-core-g14-448 | 224x224 | 25.7 ms | 24.5 ms | 15.0 ms | 17.0 ms | 29.9 s | 106 s | 3640 / 3670 | 3667 / 3690 | 3640 / 3696 | 7189 / 7230 |
| pe-core-l14-336 | 224x224 | 9.09 ms | 6.53 ms | 4.34 ms | 4.97 ms | 13.9 s | 50.1 s | 650 / 668 | 683 / 706 | 648 / 672 | 1260 / 1286 |
| pe-core-s16-384 | 224x224 | 4.80 ms | 2.39 ms | 0.581 ms | 0.957 ms | 8.75 s | 36.7 s | 83 / 102 | 113 / 122 | 82 / 104 | 145 / 166 |
| pe-core-t16-384 | 224x224 | 4.68 ms | 2.29 ms | 0.432 ms | 0.683 ms | 8.89 s | 36.1 s | 48 / 56 | 78 / 96 | 53 / 82 | 79 / 88 |
| pe-spatial-b16-512 | 224x224 | 4.74 ms | 2.67 ms | 1.20 ms | 1.40 ms | 8.67 s | 35.1 s | 207 / 226 | 237 / 266 | 204 / 248 | 397 / 422 |
| pe-spatial-g14-448 | 224x224 | 25.7 ms | 24.3 ms | 14.9 ms | 16.9 ms | 28.3 s | 126 s | 3582 / 3592 | 3608 / 3630 | 3583 / 3676 | 7136 / 7156 |
| pe-spatial-l14-448 | 224x224 | 8.84 ms | 6.59 ms | 4.35 ms | 4.87 ms | 13.7 s | 53.4 s | 626 / 668 | 658 / 686 | 623 / 672 | 1235 / 1284 |
| pe-spatial-s16-512 | 224x224 | 4.51 ms | 2.67 ms | 0.580 ms | 0.922 ms | 8.67 s | 35.3 s | 81 / 100 | 110 / 120 | 78 / 104 | 142 / 144 |
| pe-spatial-t16-512 | 224x224 | 4.48 ms | 2.12 ms | 0.426 ms | 0.666 ms | 8.45 s | 33.1 s | 49 / 56 | 77 / 96 | 53 / 82 | 78 / 88 |
| siglip2-b16-256 | 224x224 | 3.28 ms | 2.53 ms | 1.13 ms | 1.43 ms | 6.89 s | 24.3 s | 216 / 248 | 250 / 266 | 216 / 248 | 407 / 440 |
| siglip2-b16-512 | 224x224 | 3.31 ms | 2.75 ms | 1.14 ms | 1.44 ms | 6.85 s | 24.3 s | 219 / 266 | 251 / 286 | 217 / 268 | 408 / 460 |
| siglip2-g16-256 | 224x224 | 14.7 ms | 15.3 ms | 9.99 ms | 10.7 ms | 17.9 s | 51.2 s | 2261 / 2288 | 2307 / 2348 | 2262 / 2312 | 4441 / 4474 |
| siglip2-g16-384 | 224x224 | 14.7 ms | 16.9 ms | 10.2 ms | 10.7 ms | 19.3 s | 54.1 s | 2262 / 2288 | 2308 / 2348 | 2263 / 2312 | 4442 / 4474 |
| siglip2-l16-256 | 224x224 | 5.71 ms | 5.15 ms | 2.93 ms | 3.36 ms | 10.8 s | 31.7 s | 643 / 668 | 679 / 708 | 643 / 670 | 1253 / 1282 |
| siglip2-l16-512 | 224x224 | 5.70 ms | 5.38 ms | 2.91 ms | 3.35 ms | 11.1 s | 28.2 s | 647 / 666 | 681 / 706 | 644 / 670 | 1252 / 1278 |
| siglip2-so400m14-224 | 224x224 | 6.38 ms | 6.71 ms | 3.77 ms | 4.32 ms | 12.3 s | 34.7 s | 858 / 868 | 887 / 908 | 858 / 892 | 1659 / 1674 |
| siglip2-so400m14-378 | 224x224 | 6.38 ms | 6.98 ms | 3.79 ms | 4.33 ms | 12.2 s | 32.4 s | 861 / 908 | 888 / 926 | 860 / 912 | 1660 / 1712 |
| siglip2-so400m16-256 | 224x224 | 6.42 ms | 6.62 ms | 3.60 ms | 4.21 ms | 12.9 s | 30.0 s | 857 / 868 | 894 / 908 | 857 / 890 | 1657 / 1672 |
| siglip2-so400m16-512 | 224x224 | 6.40 ms | 7.02 ms | 3.61 ms | 4.21 ms | 13.2 s | 29.8 s | 861 / 906 | 896 / 928 | 858 / 910 | 1658 / 1712 |
| tipsv2-b14 | 224x224 | 2.93 ms | 2.18 ms | 1.33 ms | 1.49 ms | 9.18 s | 41.5 s | 204 / 224 | 239 / 266 | 206 / 226 | 398 / 422 |
| tipsv2-l14 | 224x224 | 6.15 ms | 6.04 ms | 4.28 ms | 4.58 ms | 12.4 s | 47.9 s | 621 / 624 | 659 / 666 | 624 / 648 | 1236 / 1264 |
| boq-dinov2 | 480x640 | 5.73 ms | 3.89 ms | 2.73 ms | 2.84 ms | 11.8 s | 46.4 s | 231 / 254 | 269 / 274 | 227 / 256 | 426 / 434 |
| megaloc | 480x640 | 5.29 ms | 4.27 ms | 3.21 ms | 4.26 ms | 13.2 s | 51.1 s | 483 / 508 | 518 / 530 | 485 / 508 | 937 / 964 |
| anycalib-pinhole | 1080x1920 | 29.4 ms | 25.5 ms | 12.0 ms | 21.5 ms | 16.5 s | 116 s | 705 / 742 | 719 / 756 | 739 / 812 | 778 / 780 |
| anycalib-pinhole | 480x640 | 29.9 ms | – | 12.4 ms | – | – | 118 s | 693 / 704 | – | – | 750 / 768 |
| geocalib-pinhole | 1080x1920 | 81.2 ms | 73.1 ms | public API unavailable | 81.8 ms | 28.4 s | – | 320 / 378 | 246 / 300 | 277 / 338 | – |
| geocalib-pinhole | 1920x1440 | 123 ms | – | 109 ms, fp32 engine (field predictor only, standalone build) | – | – | 42.9 s | 433 / 560 | – | – | 540 / 558 |
| anysplat | 336x518 | 508 ms | 427 ms | 314 ms | 369 ms | 80.3 s | 215 s | 4164 / 5384 | 4137 / 4694 | 4173 / 5548 | 7629 / 7674 |
| depthsplat-small-re10k | 336x518 | 122 ms | 82.2 ms | 86.4 ms | 95.1 ms | 80.0 s | 224 s | 956 / 1134 | 5432 / 5526 | 965 / 1314 | 3148 / 3194 |
| lingbot-map | 480x640 | 733 ms | 657 ms | 489 ms | 520 ms | 71.0 s | 181 s | 3910 / 5734 | 3339 / 3468 | 3919 / 5932 | 6280 / 6338 |
| mapanything-apache | 336x518 | 637 ms | 596 ms | 367 ms | 525 ms | 46.2 s | 137 s | 3252 / 3652 | 3347 / 3792 | 3252 / 3652 | 6220 / 6248 |
| mvsanywhere | 448x672 | 93.8 ms | 70.8 ms | 75.4 ms (standalone build) | 35.2 ms | 153 s | 217 s | 712 / 868 | 818 / 860 | 702 / 760 | 1581 / 1658 |
| mvsplat-re10k | 336x518 | 181 ms | 115 ms | 106 ms | 172 ms | 102 s | 241 s | 1239 / 1532 | 2762 / 2810 | 1239 / 1532 | 4953 / 4972 |
| patchmatchnet | 1536x1024 | 201 ms | 175 ms | 51.0 ms | 29.5 ms | 29.4 s | 82.1 s | 1039 / 1212 | 1798 / 1888 | 648 / 772 | 1073 / 1106 |
| pi3 | 336x518 | 822 ms | 752 ms | public API unavailable | 560 ms | 55.6 s | – | 2209 / 2254 | 2205 / 2334 | 2217 / 2296 | – |
| pi3 | 8x512x640 | – | – | 735 ms (DINOv2 encoder with eager scene decoding, 8 views) | – | – | 53.8 s | – | – | – | 2586 / 2976 |
| pi3 | 8x336x518 | – | – | 691 ms (34 decoder blocks with eager encoder and output heads, 8 views) | – | – | 58.0 s | – | – | – | 2600 / 2952 |
| resplat-base-re10k | 480x640 | 6868 ms | 7098 ms (standalone Gaussian initialization and recurrent refinement heads, eager rendering and geometry) | fails: TypeError: _Base.forward() missing 1 required positional arg | 1169 ms | 12.6 s | – | 4336 / 5596 | 4656 / 7630 | 4348 / 5546 | – |
| resplat-base-re10k | 256x256 | – | – | 1028 ms (Gaussian initialization and refinement heads with eager depth and rendering) | – | – | – | – | – | – | 2173 / 2988 |
| resplat-small-dl3dv | 480x640 | 1680 ms | 1538 ms | public API unavailable | 313 ms | 243 s | – | 1103 / 1240 | 4995 / 5138 | 1089 / 1208 | – |
| resplat-small-dl3dv | 256x448 | – | – | 1362 ms (Gaussian initialization and refinement heads with eager depth and rendering) | – | – | – | – | – | – | 1367 / 1800 |
| vggt-1b | 336x518 | 653 ms | 557 ms | 424 ms | 446 ms | 74.3 s | 199 s | 3573 / 3794 | 3611 / 3856 | 3580 / 3854 | 6421 / 6472 |
| vggt-omega-1b-512 | 336x518 | 727 ms | 589 ms | 377 ms | 469 ms | 65.7 s | 187 s | 3269 / 3352 | 3082 / 3154 | 3270 / 3330 | 5657 / 5766 |
| flowformer-small-chairs | 1080x1920 | 4006 ms | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | 3951 ms | – | – | 8780 / 8960 | – | 8696 / 8822 | – |
| flowformer-small-chairs | 375x1242 | 338 ms | 250 ms | 202 ms | 326 ms | 98.7 s | 118 s | 744 / 838 | 2438 / 3196 | 740 / 838 | 2087 / 2138 |
| flowformer-small-sintel | 1080x1920 | 4003 ms | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | 3958 ms | – | – | 8780 / 8960 | – | 8696 / 8822 | – |
| flowformer-small-sintel | 375x1242 | 336 ms | 250 ms | 202 ms | 322 ms | 98.9 s | 117 s | 1404 / 1698 | 2438 / 3196 | 1401 / 1698 | 2087 / 2138 |
| flowformer-small-things | 1080x1920 | 4003 ms | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 15.7 | 3950 ms | – | – | 8780 / 8960 | – | 8696 / 8822 | – |
| flowformer-small-things | 375x1242 | 337 ms | 250 ms | 202 ms | 322 ms | 99.3 s | 118 s | 1404 / 1698 | 2438 / 3196 | 1401 / 1698 | 2087 / 2138 |
| flowseek-t | 1080x1920 | 396 ms | 258 ms | 168 ms | 255 ms | 54.2 s | 107 s | 6123 / 6252 | 6126 / 6212 | 1100 / 1154 | 11328 / 11350 |
| flowseek-t-things | 1080x1920 | 395 ms | 258 ms | 168 ms | 254 ms | 54.6 s | 102 s | 6123 / 6312 | 6126 / 6212 | 1100 / 1154 | 11330 / 11352 |
| flowseek-t-things-notartan | 1080x1920 | 395 ms | 258 ms | 168 ms | 254 ms | 54.3 s | 97.5 s | 6123 / 6312 | 6126 / 6212 | 1100 / 1154 | 11328 / 11350 |
| gmflow-scale1-mixdata | 1080x1920 | 448 ms | 382 ms | 187 ms | 446 ms | 29.1 s | 67.3 s | 1988 / 2142 | 4293 / 4558 | 1988 / 2142 | 4333 / 4390 |
| gmflow-scale1-things | 1080x1920 | 448 ms | 382 ms | 187 ms | 446 ms | 35.3 s | 69.3 s | 1988 / 2142 | 4293 / 4558 | 1988 / 2142 | 4333 / 4390 |
| gmflow-scale2-mixdata | 1080x1920 | 915 ms | 765 ms | public API unavailable | 915 ms | 46.6 s | – | 2995 / 3184 | 10865 / 10904 | 2995 / 3204 | – |
| gmflow-scale2-mixdata | 384x1248 | – | – | 212 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 13.6 s | – | – | – | 1616 / 2560 |
| gmflow-scale2-regrefine6-kitti | 1080x1920 | 1383 ms | 1116 ms | public API unavailable | 1389 ms | 58.4 s | – | 3005 / 3192 | 10875 / 10972 | 3005 / 3212 | – |
| gmflow-scale2-regrefine6-kitti | 384x1248 | – | – | 351 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 13.1 s | – | – | – | 1660 / 2526 |
| gmflow-scale2-regrefine6-mixdata | 1080x1920 | 1382 ms | 1116 ms | public API unavailable | 1383 ms | 58.9 s | – | 3005 / 3192 | 10875 / 10972 | 3005 / 3212 | – |
| gmflow-scale2-regrefine6-mixdata | 384x1248 | – | – | 349 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 13.6 s | – | – | – | 1660 / 2526 |
| gmflow-scale2-regrefine6-sintelft | 1080x1920 | 1381 ms | 1116 ms | public API unavailable | 1384 ms | 58.5 s | – | 3005 / 3192 | 10875 / 10972 | 3005 / 3212 | – |
| gmflow-scale2-regrefine6-sintelft | 384x1248 | – | – | 350 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 14.5 s | – | – | – | 1660 / 2526 |
| gmflow-scale2-regrefine6-things | 1080x1920 | 1478 ms | 1116 ms | public API unavailable | 1481 ms | 58.2 s | – | 4036 / 4696 | 10875 / 10972 | 4036 / 4696 | – |
| gmflow-scale2-regrefine6-things | 384x1248 | – | – | 350 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 13.7 s | – | – | – | 1660 / 2526 |
| gmflow-scale2-sintel | 1080x1920 | 915 ms | 765 ms | public API unavailable | 916 ms | 46.6 s | – | 2995 / 3184 | 10865 / 10904 | 2995 / 3204 | – |
| gmflow-scale2-sintel | 384x1248 | – | – | 212 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 12.7 s | – | – | – | 1616 / 2560 |
| gmflow-scale2-things | 1080x1920 | 914 ms | 765 ms | public API unavailable | 916 ms | 46.7 s | – | 2995 / 3184 | 10865 / 10904 | 2995 / 3204 | – |
| gmflow-scale2-things | 384x1248 | – | – | 212 ms, fp32 engine (standalone feature CNN, eager transformer and flow update) | – | – | 13.6 s | – | – | – | 1616 / 2560 |
| megaflow | 1080x1920 | 1112 ms | 917 ms | 677 ms | 735 ms | 102 s | 214 s | 2713 / 3518 | 2611 / 2886 | 2724 / 3530 | 9265 / 9300 |
| memflow | 1080x1920 | 965 ms | 1642 ms | fails: TensorRT engine build failed | 865 ms | 78.9 s | – | 6001 / 6294 | 6813 / 6946 | 968 / 1022 | – |
| memflow | 384x1248 | – | – | 178 ms, fp32 engine (full fixed-shape flow network, standalone trace export and build) | – | – | 99.3 s | – | – | – | 1171 / 1176 |
| memflow-kitti | 1080x1920 | 964 ms | 1644 ms | fails: TensorRT engine build failed | 865 ms | 72.6 s | – | 5985 / 6102 | 6813 / 6946 | 968 / 1022 | – |
| memflow-kitti | 384x1248 | – | – | 135 ms (full fixed-shape flow network, standalone engine build) | – | – | 330 s | – | – | – | 859 / 864 |
| memflow-sintel | 1080x1920 | 964 ms | 1640 ms | fails: TensorRT engine build failed | 865 ms | 76.0 s | – | 5985 / 6102 | 6813 / 6946 | 968 / 1022 | – |
| memflow-sintel | 384x1248 | 143 ms | – | 137 ms (full fixed-shape flow network, standalone trace export and build) | – | – | 140 s | 528 / 738 | – | – | 887 / 892 |
| memfof | 1080x1920 | 433 ms | 291 ms | 160 ms | 349 ms | 50.5 s | 110 s | 1911 / 2196 | 2452 / 2666 | 917 / 1016 | 2070 / 2132 |
| memfof-kitti | 375x1242 | 386 ms | 258 ms | fails: unsupported: engine build needs more than the 16 GB of this  | 309 ms | 48.7 s | – | 1619 / 1856 | 2041 / 2226 | 786 / 916 | – |
| memfof-kitti | 384x1248 | – | – | 1335 ms, fp32 engine (ResNet encoders with eager correlation, flow update and upsampling) | – | – | 32.0 s | – | – | – | 4252 / 5692 |
| memfof-sintel | 1080x1920 | 2203 ms | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 | fails: OutOfMemoryError: CUDA out of memory. Tried to allocate 3.91 | 1565 ms | – | – | 14515 / 14918 | – | 3805 / 4038 | – |
| memfof-sintel | 375x1242 | 385 ms | 258 ms | – | 309 ms | 45.0 s | – | 1619 / 1836 | 2041 / 2226 | 786 / 876 | – |
| memfof-sintel | 384x1248 | – | – | 1334 ms, fp32 engine (ResNet encoders with eager correlation, flow update and upsampling) | – | – | 31.6 s | – | – | – | 4252 / 5692 |
| memfof-spring | 1080x1920 | 433 ms | 291 ms | 160 ms | 349 ms | 45.4 s | 95.8 s | 1900 / 2196 | 2452 / 2666 | 917 / 1016 | 2070 / 2112 |
| neuflow-v2 | 1080x1920 | 72.0 ms | 91.4 ms | 65.0 ms | 59.8 ms | 28.5 s | 104 s | 2327 / 2398 | 4529 / 4558 | 490 / 636 | 8710 / 8748 |
| neuflow-v2-sintel | 1080x1920 | 71.3 ms | 91.4 ms | 64.9 ms | 58.9 ms | 28.5 s | 102 s | 2366 / 2398 | 4529 / 4558 | 790 / 896 | 8710 / 8748 |
| neuflow-v2-things | 1080x1920 | 71.4 ms | 91.5 ms | 65.0 ms | 58.9 ms | 28.8 s | 104 s | 2366 / 2398 | 4529 / 4558 | 779 / 896 | 8710 / 8748 |
| raft-large | 1080x1920 | 283 ms | 195 ms | fails: TensorRT engine build failed | 83.4 ms | 41.3 s | – | 5721 / 5758 | 5886 / 5924 | 332 / 352 | – |
| raft-large | 384x1248 | – | – | 58.9 ms, fp32 engine | – | – | 147 s | – | – | – | 873 / 918 |
| raft-large-kitti | 1080x1920 | 284 ms | 195 ms | fails: TensorRT engine build failed | 84.2 ms | 39.3 s | – | 5721 / 5758 | 5886 / 5924 | 332 / 352 | – |
| raft-large-kitti | 384x1248 | – | – | 19.8 ms | – | – | 173 s | – | – | – | 845 / 856 |
| raft-large-things | 1080x1920 | 283 ms | 195 ms | fails: TensorRT engine build failed | 83.5 ms | 39.4 s | – | 5721 / 5758 | 5886 / 5924 | 332 / 352 | – |
| raft-large-things | 384x1248 | – | – | 59.0 ms, fp32 engine | – | – | 161 s | – | – | – | 873 / 918 |
| raft-small | 1080x1920 | 141 ms | 110 ms | fails: unsupported: engine build needs more than the 16 GB of this  | 36.5 ms | 37.0 s | – | 5601 / 5646 | 5704 / 5728 | 196 / 204 | – |
| raft-small | 384x1248 | – | – | 20.7 ms, fp32 engine | – | – | 178 s | – | – | – | 803 / 820 |
| recover-cx | 1080x1920 | 107 ms | 89.9 ms | 47.5 ms | 89.8 ms | 13.9 s | 41.6 s | 733 / 892 | 504 / 572 | 733 / 892 | 603 / 632 |
| recover-mn | 1080x1920 | 45.5 ms | 32.2 ms | 21.2 ms | 41.8 ms | 14.4 s | 56.9 s | 580 / 676 | 365 / 416 | 580 / 676 | 425 / 466 |
| recover-rn | 1080x1920 | 64.1 ms | 48.8 ms | 28.9 ms | 51.3 ms | 9.34 s | 34.1 s | 590 / 672 | 375 / 432 | 590 / 672 | 419 / 466 |
| sea-raft-m | 1080x1920 | 269 ms | 191 ms | 132 ms | 73.4 ms | 34.3 s | 70.0 s | 5847 / 6570 | 5869 / 5920 | 425 / 454 | 11110 / 11130 |
| sea-raft-m-kitti | 1080x1920 | 270 ms | 191 ms | 132 ms | 73.3 ms | 34.9 s | 69.7 s | 6060 / 6180 | 5869 / 5920 | 425 / 454 | 11110 / 11130 |
| sea-raft-m-spring | 1080x1920 | 46.8 ms | 34.6 ms | 4.43 ms | 23.5 ms | 23.2 s | 47.1 s | 652 / 700 | 624 / 640 | 170 / 194 | 291 / 318 |
| sea-raft-m-things | 1080x1920 | 270 ms | 192 ms | 132 ms | 73.0 ms | 34.9 s | 73.6 s | 6060 / 6180 | 5869 / 5920 | 425 / 454 | 11110 / 11130 |
| sea-raft-s | 1080x1920 | 211 ms | 150 ms | 108 ms | 50.6 ms | 35.4 s | 68.0 s | 6019 / 6136 | 5828 / 5876 | 404 / 452 | 11043 / 11078 |
| sea-raft-s-kitti | 1080x1920 | 210 ms | 150 ms | 108 ms | 50.7 ms | 31.8 s | 67.3 s | 6019 / 6136 | 5828 / 5876 | 404 / 432 | 11043 / 11078 |
| sea-raft-s-spring | 1080x1920 | 32.8 ms | 23.8 ms | 3.01 ms | 16.2 ms | 21.9 s | 38.8 s | 611 / 656 | 582 / 596 | 150 / 172 | 215 / 278 |
| waft-a1 | 1080x1920 | 741 ms | 610 ms | 343 ms | 519 ms | 84.7 s | 166 s | 1832 / 2196 | 3479 / 3638 | 1851 / 2264 | 1625 / 1672 |
| waft-a1-kitti | 1080x1920 | 791 ms | 610 ms | 343 ms | 519 ms | 84.1 s | 168 s | 3553 / 4196 | 3479 / 3758 | 1851 / 2264 | 1625 / 1672 |
| waft-a1-sintel | 1080x1920 | 791 ms | 610 ms | 343 ms | 519 ms | 84.7 s | 166 s | 3553 / 4076 | 3479 / 3578 | 1851 / 2264 | 1625 / 1672 |
| aliked-flexglue | 768x1024 | 129 ms | 96.1 ms | 328 ms, fp32 engine (dense descriptor with eager ALIKED and matching) | 129 ms | 78.8 s | 22.3 s | 1775 / 1886 | 1762 / 1826 | 1814 / 2178 | 2225 / 2410 |
| aliked-lightglue | 768x1024 | 40.6 ms | 29.5 ms | 65.2 ms, fp32 engine (standalone ALIKED encoder with eager keypoint selection and LightGlue) | 35.1 ms | 30.8 s | 26.4 s | 1300 / 1404 | 1653 / 1680 | 1238 / 1324 | 1520 / 1652 |
| dedode-v2-b | 768x1024 | 236 ms | 164 ms | 494 ms, fp32 engine (FP32 public builder probe) | 189 ms | 50.5 s | 67.8 s | 1160 / 1292 | 1893 / 2434 | 1197 / 1246 | 3354 / 3378 |
| dedode-v2-g | 768x1024 | 379 ms | 330 ms | 1197 ms, fp32 engine (FP32 descriptor and bilinear sampling with eager detector, keypoint order and matching) | 332 ms | 86.4 s | 71.7 s | 1761 / 2114 | 2881 / 3316 | 1787 / 1984 | 5282 / 6458 |
| disk-lightglue | 768x1024 | 116 ms | 113 ms | public API unavailable | 49.2 ms | 21.4 s | – | 1950 / 2400 | 1375 / 1502 | 736 / 778 | – |
| loftr-indoor | 768x1024 | 110 ms | 71.0 ms | 199 ms, fp32 engine (ResNetFPN with eager coarse and fine matching) | 86.0 ms | 30.2 s | 26.4 s | 740 / 944 | 634 / 684 | 740 / 806 | 2218 / 2962 |
| loftr-outdoor | 768x1024 | 110 ms | 176 ms | 199 ms, fp32 engine (ResNetFPN with eager coarse and fine matching) | 85.8 ms | 33.3 s | 26.2 s | 733 / 944 | 1149 / 1232 | 734 / 806 | 2206 / 2958 |
| loma-b | 768x1024 | 309 ms | 287 ms | 326 ms, bf16 engine (standalone build) | 263 ms | 143 s | 176 s | 1767 / 1878 | 2870 / 3258 | 1217 / 1328 | 1986 / 2010 |
| loma-b128 | 768x1024 | 150 ms | 124 ms | 144 ms, bf16 engine | 116 ms | 86.5 s | 103 s | 878 / 984 | 1398 / 1800 | 625 / 684 | 1062 / 1066 |
| loma-b64 | 768x1024 | 131 ms | 102 ms | 109 ms, bf16 engine | 107 ms | 86.6 s | 102 s | 854 / 962 | 1230 / 1356 | 613 / 720 | 1010 / 1034 |
| loma-g | 768x1024 | 359 ms | 331 ms | 366 ms, bf16 engine | 310 ms | 131 s | 171 s | 2444 / 2548 | 3547 / 3890 | 1555 / 1668 | 2298 / 2322 |
| loma-l | 768x1024 | 320 ms | 298 ms | 336 ms, bf16 engine (standalone build) | 273 ms | 134 s | 157 s | 1903 / 2012 | 3006 / 3194 | 1285 / 1406 | 2042 / 2066 |
| loma-r | 768x1024 | 309 ms | 287 ms | 326 ms, bf16 engine | 262 ms | 142 s | 175 s | 1767 / 1878 | 2870 / 3256 | 1217 / 1328 | 1986 / 2010 |
| romav1 | 768x1024 | 377 ms | 247 ms | public API unavailable | 218 ms | 698 s | – | 1597 / 1738 | 1348 / 1476 | 1597 / 1738 | – |
| romav2 | 768x1024 | 550 ms | 723 ms | 353 ms | 350 ms | 360 s | 185 s | 1989 / 2476 | 3030 / 3144 | 1890 / 2460 | 6694 / 6740 |
| xfeat | 768x1024 | 7.17 ms | 5.65 ms | 5.50 ms, fp32 engine (standalone image CNN, eager keypoint selection and matching) | 15.2 ms | 30.8 s | 11.8 s | 122 / 154 | 186 / 212 | 313 / 352 | 268 / 416 |
| xfeat-lighterglue | 768x1024 | 21.3 ms | 16.2 ms | public API unavailable | 29.2 ms | 39.7 s | – | 445 / 476 | 262 / 278 | 445 / 478 | – |
| alltracker | 480x854 | 385 ms | 284 ms | public API unavailable | 213 ms | 105 s | – | 885 / 998 | 1273 / 1358 | 967 / 1068 | – |
| alltracker | 256x256 | – | – | 330 ms, fp32 engine (frame CNN with eager tracking) | – | – | 20.3 s | – | – | – | 2262 / 3372 |
| alltracker-tiny | 480x854 | 330 ms | 229 ms | public API unavailable | 217 ms | 103 s | – | 919 / 1178 | 1198 / 1338 | 978 / 1166 | – |
| alltracker-tiny | 256x256 | – | – | 270 ms, fp32 engine (frame CNN with eager tracking) | – | – | 19.3 s | – | – | – | 2010 / 3220 |
| bootstapir | 480x854 | 233 ms | 183 ms | 13.2 ms | 191 ms | 207 s | 65.0 s | 499 / 568 | 758 / 832 | 499 / 568 | 492 / 532 |
| cotracker3-offline | 480x854 | 162 ms | 135 ms | 25.9 ms | 154 ms | 36.5 s | 157 s | 755 / 882 | 755 / 902 | 757 / 882 | 519 / 586 |
| cotracker3-online | 480x854 | 184 ms | 159 ms | public API unavailable | 176 ms | 37.7 s | – | 755 / 906 | 701 / 846 | 757 / 888 | – |
| cotracker3-online | 256x256 | – | – | 50.3 ms, fp32 engine (fixed eight-frame graph with 16 queries) | – | – | 117 s | – | – | – | 456 / 472 |
| cotracker3-online | 24x256x256 | 280 ms | – | 141 ms, fp32 engine (fixed 24-frame clip with 16 queries) | – | – | – | 2381 / 2694 | – | – | 850 / 856 |
| locotrack-b | 480x854 | 189 ms | 82.6 ms | 6.12 ms (8 frames with 16 queries only) | 109 ms | 146 s | – | 535 / 596 | 1117 / 1140 | 507 / 600 | 233 / 242 |
| locotrack-s | 480x854 | 180 ms | 74.1 ms | 5.66 ms (8 frames with 16 queries only) | 151 ms | 140 s | 51.4 s | 759 / 842 | 1000 / 1034 | 596 / 642 | 251 / 256 |
| tapir | 480x854 | 182 ms | 131 ms | 6.47 ms | 141 ms | 169 s | 55.2 s | 463 / 548 | 630 / 752 | 463 / 548 | 361 / 406 |
| tapnext | 480x854 | 528 ms | 398 ms | 69.4 ms | 381 ms | 40.5 s | 366 s | 1220 / 1294 | 1164 / 1272 | 1205 / 1254 | 1498 / 1660 |
| tapnext-pp | 480x854 | 529 ms | 395 ms | 69.5 ms | 382 ms | 34.1 s | 362 s | 1028 / 1094 | 1164 / 1272 | 1013 / 1074 | 1498 / 1660 |
| track-on-r | 480x854 | 1150 ms | 1066 ms | public API unavailable | 1064 ms | 66.0 s | – | 1030 / 1190 | 1340 / 1454 | 975 / 1156 | – |
| track-on-r | 256x256 | – | – | 958 ms, fp32 engine (standalone eight-frame encoder, eager recurrent decoder) | – | – | 42.1 s | – | – | – | 1979 / 2286 |
| track-on2 | 480x854 | 1159 ms | 1075 ms | public API unavailable | 1073 ms | 60.0 s | – | 1030 / 1190 | 1340 / 1454 | 975 / 1158 | – |
| track-on2 | 256x256 | – | – | 961 ms, fp32 engine (standalone eight-frame encoder, eager recurrent decoder) | – | – | 46.3 s | – | – | – | 1979 / 2288 |
| ema-vfi | 1080x1920 | 233 ms | 165 ms | 124 ms | 185 ms | 90.6 s | 90.9 s | 1659 / 2082 | 3325 / 3542 | 1659 / 2102 | 1719 / 1786 |
| ema-vfi-small | 1080x1920 | 89.4 ms | 59.6 ms | 43.7 ms | 84.4 ms | 78.5 s | 67.6 s | 1614 / 1792 | 1809 / 1890 | 271 / 1914 | 1018 / 1110 |
| ema-vfi-small-t | 1080x1920 | 89.4 ms | 59.8 ms | 43.4 ms | 84.3 ms | 77.9 s | 68.8 s | 1614 / 1792 | 1809 / 1890 | 271 / 1914 | 1018 / 1110 |
| ema-vfi-t | 1080x1920 | 233 ms | 165 ms | 125 ms | 207 ms | 84.6 s | 91.5 s | 2827 / 3184 | 3325 / 3702 | 368 / 3566 | 1719 / 1766 |
| film | 1080x1920 | 412 ms | 332 ms | 189 ms | 328 ms | 41.1 s | 203 s | 2397 / 2702 | 2666 / 3192 | 2397 / 2942 | 2658 / 2658 |
| gimm-vfi-r | 1080x1920 | 2528 ms | 1498 ms | public API unavailable | 1521 ms | 307 s | – | 5872 / 6518 | 12051 / 12880 | 4511 / 5282 | – |
| gimm-vfi-r | 480x854 | 468 ms | – | 1210 ms, fp32 engine (RAFT feature encoder with eager context, flow update and synthesis) | 289 ms | – | 12.9 s | 1574 / 1674 | – | 1388 / 1618 | 3522 / 6116 |
| gimm-vfi-r-p | 1080x1920 | 2527 ms | 1498 ms | fails: TensorRT parser: Sc node | 1520 ms | 307 s | – | 5872 / 6518 | 12051 / 12880 | 4511 / 5282 | – |
| gimm-vfi-r-p | 480x854 | 467 ms | – | 1190 ms, fp32 engine (RAFT feature and context encoders with eager flow update and synthesis) | 289 ms | – | 16.9 s | 1574 / 1674 | – | 1388 / 1598 | 3592 / 6000 |
| rife-4.25 | 1080x1920 | 25.0 ms | 44.9 ms | 17.6 ms | 21.4 ms | 36.1 s | 56.3 s | 755 / 894 | 1094 / 1234 | 848 / 916 | 696 / 714 |
| rife-4.25-lite | 1080x1920 | 21.1 ms | 42.3 ms | 14.1 ms | 18.3 ms | 34.9 s | 49.5 s | 754 / 854 | 1061 / 1374 | 834 / 896 | 696 / 714 |
| rife-4.26 | 1080x1920 | 25.0 ms | 44.5 ms | 17.6 ms | 21.4 ms | 34.9 s | 54.9 s | 570 / 754 | 1094 / 1234 | 663 / 776 | 696 / 714 |
| big-lama | 512x512 | 26.0 ms | 23.4 ms | 33.9 ms | 21.2 ms | 25.9 s | 110 s | 219 / 242 | 312 / 342 | 219 / 284 | 386 / 406 |
| codeformer | 512x512 | 48.6 ms | 30.8 ms | 16.3 ms | 40.4 ms | 34.1 s | 64.8 s | 512 / 638 | 407 / 458 | 432 / 496 | 593 / 642 |
| dat-2-x4 | 512x512 | 1156 ms | 964 ms | 921 ms | 1037 ms | 44.3 s | 206 s | 1420 / 2220 | 2546 / 2598 | 1399 / 2220 | 9001 / 9048 |
| dat-light-x4 | 512x512 | 234 ms | 158 ms | 136 ms | 230 ms | 36.7 s | 74.8 s | 521 / 532 | 1385 / 1412 | 521 / 532 | 4144 / 4192 |
| dat-x4 | 512x512 | 1386 ms | 1063 ms | 1001 ms | 1209 ms | 43.1 s | 191 s | 1427 / 2246 | 2552 / 2606 | 1406 / 2226 | 9010 / 9056 |
| drct-l-x4 | 512x512 | 1567 ms | 1210 ms | fails: TensorRT engine build failed | 1314 ms | 67.2 s | – | 1675 / 1950 | 2005 / 2410 | 1603 / 1790 | – |
| drct-l-x4 | 256x256 | 384 ms | – | 289 ms | – | – | 222 s | 766 / 1204 | – | – | 559 / 560 |
| drct-x4 | 512x512 | 810 ms | 618 ms | 731 ms | 674 ms | 40.5 s | 187 s | 1608 / 1888 | 1938 / 2348 | 1560 / 1748 | 9684 / 9728 |
| gfpgan-v1.4 | 512x512 | 21.2 ms | 12.5 ms | 15.8 ms | 18.0 ms | 27.3 s | 115 s | 505 / 528 | 360 / 428 | 473 / 530 | 609 / 644 |
| hat-s-x4 | 512x512 | 1072 ms | 750 ms | fails: TensorRT engine build failed | 956 ms | 41.9 s | – | 1984 / 2148 | 2352 / 3068 | 1856 / 1888 | – |
| hat-s-x4 | 256x256 | 233 ms | – | 141 ms | – | – | 89.1 s | 531 / 782 | – | – | 533 / 534 |
| hat-x4 | 512x512 | 1596 ms | 1185 ms | fails: TensorRT engine build failed | 1427 ms | 53.4 s | – | 2426 / 2590 | 1811 / 2750 | 2298 / 2470 | – |
| hat-x4 | 256x256 | 346 ms | – | 217 ms | – | – | 92.0 s | 658 / 1002 | – | – | 681 / 682 |
| nafnet-gopro | 512x512 | 60.4 ms | 37.3 ms | 52.6 ms | 31.3 ms | 46.2 s | 66.5 s | 439 / 472 | 422 / 470 | 376 / 408 | 722 / 734 |
| nafnet-sidd | 512x512 | 61.1 ms | 32.3 ms | 24.5 ms | 22.5 ms | 36.9 s | 33.8 s | 607 / 650 | 450 / 488 | 502 / 536 | 650 / 668 |
| real-hat-gan-x4 | 512x512 | 1428 ms | 1185 ms | fails: TensorRT engine build failed | 1234 ms | 54.3 s | – | 1430 / 2070 | 1811 / 2750 | 1312 / 1510 | – |
| real-hat-gan-x4 | 256x256 | 346 ms | – | 217 ms | – | – | 98.7 s | 658 / 1002 | – | – | 681 / 682 |
| realesr-general-x4v3 | 512x512 | 30.1 ms | 22.6 ms | 11.9 ms | 24.0 ms | 3.89 s | 15.3 s | 125 / 164 | 125 / 144 | 125 / 164 | 220 / 228 |
| realesrgan-x2plus | 512x512 | 97.0 ms | 85.2 ms | 59.7 ms | 80.2 ms | 33.2 s | 40.3 s | 333 / 394 | 323 / 394 | 333 / 394 | 476 / 500 |
| realesrgan-x4plus | 512x512 | 473 ms | 354 ms | 273 ms | 403 ms | 29.9 s | 55.2 s | 292 / 354 | 1091 / 1394 | 1220 / 1434 | 1546 / 1594 |
| realesrgan-x4plus-anime-6b | 512x512 | 155 ms | 114 ms | 83.1 ms | 132 ms | 10.0 s | 36.7 s | 1197 / 1430 | 1068 / 1370 | 1197 / 1410 | 1505 / 1552 |
| realplksr-nomoswebphoto-x4 | 512x512 | 182 ms | 154 ms | 247 ms | 130 ms | 15.2 s | 26.2 s | 179 / 216 | 179 / 196 | 179 / 216 | 358 / 366 |
| restormer-deblur | 512x512 | 183 ms | 143 ms | 162 ms | 132 ms | 44.1 s | 58.3 s | 724 / 740 | 660 / 778 | 718 / 740 | 757 / 774 |
| restormer-derain | 512x512 | 228 ms | 144 ms | 162 ms | 133 ms | 41.5 s | 55.3 s | 850 / 1000 | 660 / 778 | 892 / 980 | 757 / 774 |
| restormer-sidd | 512x512 | 181 ms | 142 ms | 162 ms | 133 ms | 35.5 s | 59.7 s | 850 / 920 | 660 / 778 | 892 / 980 | 757 / 774 |
| seedvr2-3b | 512x512 | 2332 ms | 1941 ms | 2280 ms, bf16 engine (BF16 VAE encoder and decoder with the host-streamed transformer in PyTorch) | 2356 ms | 238 s | – | 4233 / 7262 | 5237 / 8144 | 3857 / 7778 | 9068 / 9284 |
| seedvr2-3b | 64x64 | – | – | 46.8 ms, bf16 engine | – | – | 324 s | – | – | – | 7662 / 7664 |
| seedvr2-7b | 512x512 | 2748 ms | 2471 ms (VAE only, standalone compile) | 2696 ms, bf16 engine (BF16 VAE encoder and decoder with the host-streamed transformer in PyTorch) | 2748 ms | 52.2 s | 271 s | 4495 / 8260 | 5522 / 9042 | 4495 / 8260 | 9369 / 9860 |
| span-x2 | 512x512 | 23.5 ms | 13.2 ms | 7.98 ms | 14.9 ms | 7.26 s | 9.60 s | 607 / 662 | 204 / 222 | 487 / 522 | 243 / 266 |
| span-x4 | 512x512 | 24.2 ms | 13.2 ms | 8.28 ms | 14.6 ms | 4.57 s | 9.80 s | 607 / 662 | 204 / 222 | 487 / 522 | 319 / 346 |
| swin2sr-x2 | 512x512 | 909 ms | 482 ms | 547 ms | 721 ms | 43.7 s | 112 s | 1459 / 1554 | 1308 / 1516 | 1459 / 1636 | 4941 / 4997 |
| swin2sr-x4 | 512x512 | 932 ms | 496 ms | 556 ms | 745 ms | 45.8 s | 113 s | 1463 / 2276 | 1308 / 1556 | 1459 / 1736 | 5410 / 5470 |
| swin2sr-x4-real | 512x512 | 960 ms | 508 ms | 562 ms | 751 ms | 41.7 s | 128 s | 1459 / 1556 | 1308 / 1556 | 1459 / 1734 | 5409 / 5468 |
| swinir-l-x4-real | 512x512 | 1218 ms | 1426 ms | 1118 ms | 896 ms | 68.1 s | 193 s | 1600 / 1794 | 2153 / 2876 | 1600 / 1794 | 9182 / 9282 |
| swinir-m-x2 | 512x512 | 739 ms | 781 ms | 719 ms | 553 ms | 60.3 s | 102 s | 1353 / 1454 | 1683 / 1796 | 1353 / 1454 | 5088 / 5144 |
| swinir-m-x4 | 512x512 | 767 ms | 795 ms | 727 ms | 576 ms | 61.8 s | 104 s | 1499 / 2054 | 1684 / 1796 | 1354 / 1456 | 5525 / 5604 |
| swinir-m-x4-real | 512x512 | 1242 ms | 807 ms | 733 ms | 584 ms | 59.6 s | 109 s | 2387 / 3034 | 1683 / 1796 | 1353 / 1814 | 5427 / 5506 |
| nnfoundation-cnn | 200x200 | 248 ms | 170 ms | 255 ms (encoder on a normalized, padded volume only; memory estimated) | 232 ms | 19.8 s | 49.0 s | 2368 / 3330 | 5067 / 5110 | 2325 / 3330 | 3374 / 3398 |
| nnfoundation-vit | 192x192 | 1848 ms | 1440 ms | 7024 ms, fp32 engine (standalone encoder, fixed 192³, TF32 disabled) | 1732 ms | 69.5 s | 127 s | 1669 / 1884 | 1406 / 1484 | 1669 / 1884 | 8452 / 8452 |
| nninteractive | 200x200 | 209 ms | 162 ms | fails: The TensorRT export accepts prepared 192³ network patches. | 193 ms | 18.0 s | – | 2423 / 2772 | 3643 / 3832 | 2423 / 2772 | – |
| nninteractive | 192x192 | – | – | 262 ms (U-Net on one prepared 8-channel, 192³ patch only; memory estimated) | – | – | 58.0 s | – | – | – | 2738 / 2740 |
| totalsegmentator-total | 200x200 | 1977 ms | 1595 ms | 1582 ms (five nnU-Net patch networks with eager volume processing) | 1647 ms | 37.0 s | 152 s | 2017 / 2244 | 1697 / 1984 | 1219 / 1638 | 6646 / 7136 |
| totalsegmentator-total-3mm | 200x200 | 70.3 ms | 63.1 ms | 74.4 ms (patch network, standalone build) | 71.8 ms | 11.9 s | 31.3 s | 962 / 1050 | 907 / 1030 | 913 / 1376 | 2909 / 2970 |
<!-- compile:end -->

## Original code

The original columns are the authors' code with the same checkpoint and settings, in fp16 where the code supports it and
the output stays finite, else fp32 (marked `(fp32)`). Original fp32 is the same code at its default fp32 precision,
where that differs. The kernels speed-up is against the faster of original fp16 and fp32. Notes per row are on the model
pages. A dash means not measured or no runnable original. Where kernels was not timed at the original's size, the
overview shows the speed-up against portable instead.

## Fixed-size speed and peak memory, all models

Every model, `portable` and `kernels`, on a synthetic input of the size in the column header, median of 10 warm runs
(3 for calls over 2 s). Cells: median ms / peak allocated MiB / peak reserved MiB, weights included. OOM = out of memory
on the 16 GB GPU. Fixed-input models (classification, DINO features) run only at their native size: see their speed table above. The 12 MP column probes the largest size
that fits. Depth-anything-v2, midas-dpt-beit-large-512 and zoedepth-nk resize to their working resolution: time depends on the aspect ratio, not the input pixels (12 MP is 4:3, so it is faster than 1080p at 16:9). Measured 2026-10/11 on main at the time; the fixed-input table and the GREAT-IGEV and IGEV++ rows are from 2026-11-03, and the other columns have not been re-measured since.

<!-- sizes:begin (tools/bench_sizes.py docs) -->
Cells: median ms / peak allocated MiB / peak reserved MiB (incl. weights). OOM = out of memory; SKIPPED =
a smaller size already failed; TIMEOUT = the call did not finish in a 170 s GPU-lock step; - = not run (the
size is outside the model's range).

| model | mode | 480x640 | 720x1280 | 1080x1920 | 12 MP 3000x4000 |
|---|---|---|---|---|---|
| aanet-plus-sceneflow | portable | 58.4 ms / 1109 / 1292 | 170.2 ms / 3400 / 3858 | 389.0 ms / 7214 / 8226 | SKIPPED |
| aanet-plus-sceneflow | kernels | 28.8 ms / 178 / 244 | 53.9 ms / 501 / 696 | 128.5 ms / 1049 / 1430 | 793.2 ms / 5752 / 8230 |
| aanet-sceneflow | portable | 40.1 ms / 300 / 340 | 94.2 ms / 781 / 830 | 231.2 ms / 1711 / 1820 | 1591.3 ms / 9603 / 12836 |
| aanet-sceneflow | kernels | 25.9 ms / 148 / 196 | 59.2 ms / 344 / 602 | 152.4 ms / 728 / 1252 | 967.9 ms / 3967 / 6854 |
| bridgedepth-rvc-pretrain | portable | 76.7 ms / 1727 / 1798 | 247.2 ms / 2359 / 2628 | 676.5 ms / 3562 / 4102 | 7918.7 ms (n=3) / 13620 / 14590 |
| bridgedepth-rvc-pretrain | kernels | 62.2 ms / 1147 / 1266 | 200.9 ms / 1764 / 2096 | 534.7 ms / 2939 / 3514 | 5573.1 ms (n=3) / 12950 / 13592 |
| bridgedepth-sf | portable | 73.2 ms / 1676 / 1748 | 242.8 ms / 2308 / 2560 | 668.4 ms / 3511 / 4032 | 7849.6 ms (n=3) / 13569 / 14540 |
| bridgedepth-sf | kernels | 59.1 ms / 1096 / 1214 | 196.3 ms / 1714 / 2044 | 526.9 ms / 2888 / 3462 | 5498.6 ms (n=3) / 12898 / 13540 |
| c-fast-foundationstereo | portable | 64.6 ms / 600 / 784 | 193.1 ms / 1626 / 2144 | 464.9 ms / 3485 / 4540 | OOM |
| c-fast-foundationstereo | kernels | 38.5 ms / 306 / 386 | 85.9 ms / 762 / 960 | 182.0 ms / 1582 / 2060 | 1072.8 ms / 8699 / 11372 |
| c-foundationstereo-s | portable | 681.8 ms / 2075 / 3130 | 2218.0 ms (n=3) / 4355 / 6278 | 4704.5 ms (n=3) / 8457 / 12000 | SKIPPED |
| c-foundationstereo-s | kernels | 291.3 ms / 1262 / 1430 | 838.6 ms / 2380 / 3312 | 1826.4 ms / 4137 / 6164 | SKIPPED |
| fast-foundationstereo | portable | 64.6 ms / 600 / 784 | 193.1 ms / 1626 / 2144 | 464.8 ms / 3485 / 4540 | OOM |
| fast-foundationstereo | kernels | 38.7 ms / 306 / 386 | 85.8 ms / 762 / 960 | 181.9 ms / 1582 / 2060 | 1073.0 ms / 8699 / 11372 |
| cas-gwcnet-c-sceneflow | portable | 61.3 ms / 465 / 738 | 195.5 ms / 1386 / 2160 | 452.8 ms / 3046 / 4720 | OOM |
| cas-gwcnet-c-sceneflow | kernels | 44.8 ms / 442 / 568 | 135.5 ms / 1242 / 1586 | 316.1 ms / 2692 / 3408 | OOM |
| cas-gwcnet-gc-sceneflow | portable | 86.7 ms / 533 / 1166 | 281.6 ms / 1588 / 3308 | 647.4 ms / 3494 / 7262 | OOM |
| cas-gwcnet-gc-sceneflow | kernels | 66.9 ms / 510 / 956 | 213.7 ms / 1449 / 2606 | 493.8 ms / 3147 / 5634 | OOM |
| cfnet-sceneflow | portable | 73.6 ms / 1070 / 1588 | 284.0 ms / 2640 / 4342 | 758.8 ms / 4508 / 5956 | OOM |
| cfnet-sceneflow | kernels | 59.3 ms / 1118 / 1618 | 229.9 ms / 2778 / 4362 | 634.1 ms / 4801 / 5976 | OOM |
| coex-sceneflow | portable | 11.2 ms / 225 / 312 | 28.5 ms / 671 / 890 | 70.0 ms / 1485 / 1916 | 554.7 ms / 8660 / 10964 |
| coex-sceneflow | kernels | 4.3 ms / 103 / 322 | 13.7 ms / 256 / 400 | 32.9 ms / 559 / 804 | 211.1 ms / 3190 / 4410 |
| crestereo-eth3d | portable | 142.5 ms / 214 / 284 | 427.8 ms / 556 / 740 | 1022.9 ms / 1172 / 1568 | 6466.7 ms (n=3) / 6508 / 8990 |
| crestereo-eth3d | kernels | 61.8 ms / 216 / 292 | 145.2 ms / 552 / 792 | 311.2 ms / 1162 / 1638 | 1767.7 ms / 6427 / 9362 |
| defom-stereo-vitl-sceneflow | portable | 263.9 ms / 2070 / 2368 | 860.4 ms / 3248 / 4044 | 2169.2 ms (n=3) / 5378 / 7082 | OOM |
| defom-stereo-vitl-sceneflow | kernels | 120.5 ms / 1357 / 1702 | 376.5 ms / 2275 / 3190 | 936.5 ms / 2991 / 3578 | 5950.0 ms (n=3) / 11923 / 15526 |
| dlnr-middlebury | portable | 81.8 ms / 623 / 774 | 235.4 ms / 1651 / 2134 | 548.9 ms / 3617 / 4726 | OOM |
| dlnr-middlebury | kernels | 52.8 ms / 372 / 470 | 126.7 ms / 815 / 1070 | 297.2 ms / 1617 / 1958 | 1775.1 ms / 8572 / 12080 |
| dlnr-sceneflow | portable | 81.8 ms / 623 / 774 | 235.2 ms / 1651 / 2134 | 549.1 ms / 3617 / 4726 | OOM |
| dlnr-sceneflow | kernels | 52.9 ms / 372 / 470 | 126.8 ms / 815 / 1070 | 297.4 ms / 1617 / 1958 | 1775.1 ms / 8572 / 12080 |
| fast-acvnet-generalization | portable | 13.9 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.3 ms / 890 / 1418 | 492.1 ms / 5077 / 8074 |
| fast-acvnet-generalization | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.7 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-kitti12 | portable | 14.0 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-kitti12 | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-kitti15 | portable | 13.9 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-kitti15 | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.3 ms / 5445 / 7618 |
| fast-acvnet-plus-generalization | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.6 ms / 5054 / 8076 |
| fast-acvnet-plus-generalization | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.5 ms / 5743 / 7342 |
| fast-acvnet-plus-kitti12 | portable | 14.4 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.3 ms / 5054 / 8076 |
| fast-acvnet-plus-kitti12 | kernels | 7.7 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.4 ms / 5743 / 7342 |
| foundationstereo-vit-l | portable | 577.7 ms / 1654 / 2216 | 1846.4 ms / 2676 / 3370 | 3776.7 ms (n=3) / 3570 / 5048 | OOM |
| foundationstereo-vit-l | kernels | 320.0 ms / 1296 / 1642 | 961.0 ms / 2232 / 2800 | 1788.5 ms / 3254 / 4506 | 9521.9 ms (n=3) / 12653 / 14398 |
| foundationstereo-vit-s | portable | 481.4 ms / 1042 / 1352 | 1508.2 ms / 2047 / 2678 | 3424.1 ms (n=3) / 2900 / 4592 | OOM |
| foundationstereo-vit-s | kernels | 245.6 ms / 685 / 862 | 687.1 ms / 1604 / 2008 | 1499.8 ms / 2585 / 3440 | 8960.9 ms (n=3) / 12058 / 13800 |
| gwcnet-g-sceneflow | portable | 69.2 ms / 418 / 586 | 219.1 ms / 1254 / 1776 | 502.0 ms / 2764 / 3874 | 2911.8 ms (n=3) / 13647 / 15104 |
| gwcnet-g-sceneflow | kernels | 50.9 ms / 393 / 526 | 156.0 ms / 1112 / 1590 | 358.5 ms / 2412 / 3464 | OOM |
| gwcnet-gc-sceneflow | portable | 71.2 ms / 418 / 442 | 224.5 ms / 1255 / 1688 | 514.1 ms / 2766 / 3680 | 2981.7 ms (n=3) / 13924 / 14002 |
| gwcnet-gc-sceneflow | kernels | 54.9 ms / 394 / 468 | 167.5 ms / 1114 / 1534 | 385.9 ms / 2415 / 3300 | OOM |
| hitnet-eth3d | portable | 18.1 ms / 221 / 336 | 40.2 ms / 659 / 1004 | 92.6 ms / 1134 / 2438 | 579.4 ms / 6558 / 11110 |
| hitnet-eth3d | kernels | 5.2 ms / 140 / 470 | 14.5 ms / 392 / 482 | 30.8 ms / 836 / 1004 | 181.7 ms / 4814 / 5758 |
| hitnet-middlebury-d400 | portable | 35.8 ms / 602 / 838 | 108.8 ms / 1067 / 2222 | 251.7 ms / 2392 / 3814 | OOM |
| hitnet-middlebury-d400 | kernels | 11.7 ms / 266 / 966 | 33.1 ms / 759 / 978 | 74.2 ms / 1695 / 2248 | 426.2 ms / 9604 / 12664 |
| igev-stereo-eth3d | portable | 148.4 ms / 280 / 352 | 442.8 ms / 768 / 1034 | 1089.2 ms / 1689 / 2272 | 6940.4 ms (n=3) / 10547 / 14300 |
| igev-stereo-eth3d | kernels | 70.2 ms / 263 / 372 | 177.6 ms / 698 / 1000 | 391.1 ms / 1506 / 2044 | 2328.3 ms (n=3) / 6315 / 8190 |
| igev-stereo-kitti12 | portable | 148.6 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.5 ms / 1689 / 2272 | 6944.6 ms (n=3) / 10547 / 14300 |
| igev-stereo-kitti12 | kernels | 70.1 ms / 263 / 372 | 177.7 ms / 698 / 1000 | 391.5 ms / 1506 / 2044 | 2330.4 ms (n=3) / 6315 / 8190 |
| igev-stereo-kitti15 | portable | 148.3 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.7 ms / 1689 / 2272 | 6944.1 ms (n=3) / 10547 / 14300 |
| igev-stereo-kitti15 | kernels | 70.3 ms / 263 / 372 | 177.8 ms / 698 / 1000 | 391.3 ms / 1506 / 2044 | 2330.4 ms (n=3) / 6315 / 8190 |
| igev-stereo-middlebury | portable | 148.4 ms / 280 / 352 | 442.4 ms / 768 / 1034 | 1089.3 ms / 1689 / 2272 | 6939.3 ms (n=3) / 10547 / 14300 |
| igev-stereo-middlebury | kernels | 70.2 ms / 263 / 372 | 177.9 ms / 698 / 1000 | 391.5 ms / 1506 / 2044 | 2328.7 ms (n=3) / 6315 / 8190 |
| igev-stereo-sceneflow | portable | 148.6 ms / 280 / 352 | 442.9 ms / 768 / 1034 | 1089.8 ms / 1689 / 2272 | 6943.4 ms (n=3) / 10547 / 14300 |
| igev-stereo-sceneflow | kernels | 70.2 ms / 263 / 372 | 178.2 ms / 698 / 1000 | 392.0 ms / 1506 / 2044 | 2334.4 ms (n=3) / 6315 / 8190 |
| las2-h | portable | 29.8 ms / 269 / 328 | 69.1 ms / 668 / 904 | 174.8 ms / 1386 / 2168 | 1408.7 ms / 8708 / 12018 |
| las2-h | kernels | 10.2 ms / 218 / 596 | 28.5 ms / 604 / 1636 | 73.6 ms / 1344 / 1648 | 512.5 ms / 9695 / 12606 |
| las2-l | portable | 25.5 ms / 286 / 328 | 51.1 ms / 653 / 700 | 110.1 ms / 1317 / 1404 | 824.0 ms / 7201 / 7598 |
| las2-l | kernels | 6.6 ms / 129 / 270 | 16.8 ms / 241 / 676 | 38.6 ms / 422 / 588 | 254.6 ms / 2174 / 2896 |
| las2-m | portable | 18.9 ms / 219 / 252 | 35.4 ms / 586 / 648 | 73.9 ms / 1249 / 1358 | 602.8 ms / 7135 / 7584 |
| las2-m | kernels | 4.0 ms / 96 / 258 | 9.2 ms / 187 / 618 | 21.5 ms / 346 / 516 | 146.5 ms / 1870 / 2684 |
| las2-s | portable | 16.1 ms / 202 / 234 | 31.0 ms / 569 / 632 | 64.4 ms / 1232 / 1342 | 532.1 ms / 7119 / 7566 |
| las2-s | kernels | 3.0 ms / 88 / 246 | 6.9 ms / 179 / 602 | 16.6 ms / 338 / 500 | 109.3 ms / 1859 / 2598 |
| mocha-v1-sceneflow | portable | 231.1 ms / 500 / 662 | 628.3 ms / 1078 / 1438 | 1484.8 ms / 2164 / 2688 | 10361.4 ms (n=3) / 12466 / 15520 |
| mocha-v1-sceneflow | kernels | 135.4 ms / 495 / 694 | 317.0 ms / 1055 / 1338 | 718.5 ms / 2092 / 2484 | 4396.8 ms (n=3) / 11664 / 14958 |
| mocha-v2-sceneflow | portable | 227.0 ms / 500 / 662 | 613.9 ms / 1078 / 1438 | 1451.9 ms / 2164 / 2688 | 10135.5 ms (n=3) / 12464 / 15164 |
| mocha-v2-sceneflow | kernels | 133.6 ms / 494 / 654 | 312.9 ms / 1054 / 1456 | 709.3 ms / 2091 / 2482 | 4316.6 ms (n=3) / 11664 / 14404 |
| monster-mix-all | portable | 284.1 ms / 1089 / 1184 | 894.5 ms / 1732 / 2154 | 2291.3 ms (n=3) / 2935 / 3668 | OOM |
| monster-mix-all | kernels | 127.2 ms / 1112 / 1250 | 379.5 ms / 1752 / 2288 | 956.8 ms / 2939 / 3460 | OOM |
| monster-sceneflow | portable | 284.1 ms / 1089 / 1184 | 894.5 ms / 1732 / 2154 | 2290.9 ms (n=3) / 2935 / 3668 | OOM |
| monster-sceneflow | kernels | 127.1 ms / 1112 / 1250 | 379.4 ms / 1752 / 2288 | 956.9 ms / 2939 / 3460 | OOM |
| monster-plusplus-mix-all | portable | 294.3 ms / 1138 / 1246 | 928.8 ms / 1880 / 2318 | 2370.0 ms (n=3) / 3269 / 3812 | OOM |
| monster-plusplus-mix-all | kernels | 132.9 ms / 1130 / 1270 | 396.3 ms / 1804 / 2344 | 998.1 ms / 3051 / 3724 | OOM |
| nmrf-sceneflow | portable | 35.9 ms / 390 / 454 | 124.3 ms / 953 / 1094 | 315.1 ms / 2040 / 2402 | 2021.2 ms (n=3) / 11263 / 13666 |
| nmrf-sceneflow | kernels | 31.7 ms / 247 / 298 | 112.1 ms / 626 / 800 | 286.8 ms / 1356 / 1814 | 1834.3 ms / 7543 / 10866 |
| psmnet-sceneflow | portable | 69.0 ms / 426 / 606 | 206.2 ms / 1256 / 1780 | 479.2 ms / 2834 / 3966 | OOM |
| psmnet-sceneflow | kernels | 49.6 ms / 415 / 518 | 140.3 ms / 1108 / 1414 | 329.2 ms / 2459 / 3104 | OOM |
| raft-stereo-eth3d | portable | 143.6 ms / 320 / 376 | 433.7 ms / 864 / 1058 | 1036.0 ms / 1852 / 2258 | 6289.1 ms (n=3) / 10406 / 12814 |
| raft-stereo-eth3d | kernels | 57.8 ms / 201 / 214 | 166.5 ms / 569 / 590 | 370.4 ms / 1233 / 1264 | 2138.5 ms (n=3) / 7295 / 7436 |
| raft-stereo-middlebury | portable | 143.7 ms / 320 / 376 | 433.7 ms / 864 / 1058 | 1035.8 ms / 1852 / 2258 | 6287.7 ms (n=3) / 10406 / 12814 |
| raft-stereo-middlebury | kernels | 57.8 ms / 201 / 214 | 165.9 ms / 569 / 590 | 370.5 ms / 1233 / 1264 | 2139.9 ms (n=3) / 7295 / 7436 |
| raft-stereo-sceneflow | portable | 143.7 ms / 320 / 376 | 433.6 ms / 864 / 1058 | 1035.8 ms / 1852 / 2258 | 6286.9 ms (n=3) / 10406 / 12814 |
| raft-stereo-sceneflow | kernels | 57.6 ms / 201 / 214 | 166.0 ms / 569 / 590 | 370.4 ms / 1233 / 1264 | 2138.4 ms (n=3) / 7295 / 7436 |
| rt-monster-plusplus-mix-all | portable | 31.2 ms / 609 / 720 | 67.1 ms / 1625 / 1968 | 168.0 ms / 3478 / 4148 | OOM |
| rt-monster-plusplus-mix-all | kernels | 18.3 ms / 630 / 1934 | 57.5 ms / 1660 / 1998 | 140.1 ms / 3545 / 4188 | OOM |
| s2m2-l | portable | 182.7 ms / 1304 / 1460 | 618.6 ms / 2499 / 2956 | 1459.4 ms / 4660 / 5680 | OOM |
| s2m2-l | kernels | 83.3 ms / 719 / 782 | 228.8 ms / 988 / 1224 | 534.8 ms / 1530 / 2250 | 4189.4 ms (n=3) / 6355 / 9246 |
| s2m2-m | portable | 114.5 ms / 780 / 904 | 402.5 ms / 1690 / 1954 | 973.5 ms / 3330 / 3896 | OOM |
| s2m2-m | kernels | 55.3 ms / 395 / 492 | 153.2 ms / 655 / 940 | 355.5 ms / 1119 / 1760 | 2793.3 ms (n=3) / 5151 / 7876 |
| s2m2-s | portable | 56.0 ms / 456 / 536 | 202.3 ms / 1141 / 1450 | 511.1 ms / 2427 / 3054 | OOM |
| s2m2-s | kernels | 32.7 ms / 212 / 294 | 70.2 ms / 452 / 692 | 161.9 ms / 887 / 1534 | 1203.3 ms / 4637 / 7166 |
| s2m2-xl | portable | 338.0 ms / 2476 / 2710 | 1112.0 ms / 4248 / 4842 | 2594.0 ms (n=3) / 7447 / 8684 | SKIPPED |
| s2m2-xl | kernels | 140.8 ms / 1441 / 1644 | 414.8 ms / 1748 / 2040 | 956.5 ms / 2493 / 3446 | 6933.0 ms (n=3) / 9200 / 11578 |
| selective-igev-eth3d-finetune | portable | 190.5 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.2 ms / 1690 / 2030 | 8918.7 ms (n=3) / 10549 / 14320 |
| selective-igev-eth3d-finetune | kernels | 82.5 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 472.5 ms / 1502 / 2046 | 2806.7 ms (n=3) / 6928 / 8800 |
| selective-igev-eth3d-train | portable | 190.5 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.4 ms / 1690 / 2030 | 8919.8 ms (n=3) / 10549 / 14320 |
| selective-igev-eth3d-train | kernels | 82.4 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 472.6 ms / 1502 / 2046 | 2807.0 ms (n=3) / 6928 / 8800 |
| selective-igev-kitti12 | portable | 190.6 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.5 ms / 1690 / 2030 | 8917.7 ms (n=3) / 10549 / 14320 |
| selective-igev-kitti12 | kernels | 82.3 ms / 260 / 368 | 212.9 ms / 693 / 1010 | 471.9 ms / 1502 / 2046 | 2803.4 ms (n=3) / 6928 / 8800 |
| selective-igev-kitti15 | portable | 190.6 ms / 282 / 362 | 575.3 ms / 769 / 1036 | 1420.8 ms / 1690 / 2030 | 8918.8 ms (n=3) / 10549 / 14320 |
| selective-igev-kitti15 | kernels | 82.5 ms / 260 / 368 | 212.8 ms / 693 / 1010 | 471.9 ms / 1502 / 2046 | 2803.9 ms (n=3) / 6928 / 8800 |
| selective-igev-middlebury-finetune | portable | 190.6 ms / 282 / 362 | 575.6 ms / 769 / 1036 | 1420.8 ms / 1690 / 2030 | 8923.1 ms (n=3) / 10549 / 14320 |
| selective-igev-middlebury-finetune | kernels | 82.3 ms / 260 / 368 | 212.6 ms / 693 / 1010 | 472.1 ms / 1502 / 2046 | 2804.4 ms (n=3) / 6928 / 8800 |
| waft-stereo-b | portable | SKIPPED | SKIPPED | SKIPPED | SKIPPED |
| waft-stereo-b | kernels | 62.9 ms / 676 / 908 | 214.9 ms / 1310 / 1880 | 607.1 ms / 2523 / 3712 | OOM |
| waft-stereo-s | portable | 83.4 ms / 941 / 1146 | 357.7 ms / 2165 / 3356 | 1234.1 ms / 5368 / 8448 | SKIPPED |
| waft-stereo-s | kernels | 54.4 ms / 381 / 516 | 148.0 ms / 760 / 1120 | 441.0 ms / 1473 / 2150 | 6423.1 ms (n=3) / 7556 / 11364 |
| da3-giant-1.1 | portable | 112.6 ms / 5032 / 5332 | 90.1 ms / 4930 / 5150 | 90.3 ms / 4943 / 5150 | 118.5 ms / 5206 / 5524 |
| da3-giant-1.1 | kernels | 49.5 ms / 2630 / 3032 | 36.0 ms / 2562 / 2854 | 36.7 ms / 2575 / 2886 | 54.5 ms / 3011 / 3386 |
| da3-large-1.1 | portable | 45.6 ms / 1733 / 1992 | 37.8 ms / 1639 / 1814 | 38.8 ms / 1651 / 1826 | 51.5 ms / 1938 / 2214 |
| da3-large-1.1 | kernels | 18.3 ms / 967 / 1218 | 14.2 ms / 901 / 1068 | 14.8 ms / 914 / 1108 | 24.1 ms / 1354 / 1608 |
| da3metric-large | portable | 32.3 ms / 1542 / 1652 | 25.4 ms / 1488 / 1576 | 26.0 ms / 1502 / 1600 | 38.1 ms / 1896 / 2184 |
| da3metric-large | kernels | 15.7 ms / 805 / 880 | 12.1 ms / 782 / 838 | 12.8 ms / 795 / 858 | 21.5 ms / 1266 / 1570 |
| da3mono-large | portable | 32.2 ms / 1542 / 1652 | 25.2 ms / 1488 / 1576 | 25.9 ms / 1502 / 1600 | 38.0 ms / 1896 / 2184 |
| da3mono-large | kernels | 15.6 ms / 805 / 880 | 12.0 ms / 782 / 838 | 12.6 ms / 795 / 858 | 21.3 ms / 1266 / 1570 |
| da3nested-giant-large-1.1 | portable | 145.3 ms / 6309 / 6624 | 114.9 ms / 6206 / 6432 | 115.5 ms / 6218 / 6440 | 151.5 ms / 6482 / 6810 |
| da3nested-giant-large-1.1 | kernels | 65.6 ms / 3276 / 3658 | 48.0 ms / 3206 / 3490 | 48.7 ms / 3220 / 3528 | 71.2 ms / 3658 / 4168 |
| depth-anything-v2-b | portable | 18.8 ms / 643 / 776 | 26.2 ms / 733 / 876 | 26.3 ms / 745 / 908 | 20.0 ms / 787 / 924 |
| depth-anything-v2-b | kernels | 9.5 ms / 325 / 408 | 13.6 ms / 376 / 466 | 13.7 ms / 390 / 486 | 10.7 ms / 606 / 622 |
| depth-anything-v2-l | portable | 55.9 ms / 1752 / 1946 | 78.6 ms / 1915 / 2170 | 78.8 ms / 1928 / 2180 | 57.1 ms / 1885 / 2216 |
| depth-anything-v2-l | kernels | 28.3 ms / 875 / 1000 | 38.7 ms / 965 / 1104 | 38.8 ms / 977 / 1112 | 29.3 ms / 1058 / 1174 |
| depth-anything-v2-s | portable | 8.1 ms / 262 / 326 | 10.9 ms / 314 / 376 | 11.0 ms / 329 / 422 | 9.5 ms / 509 / 536 |
| depth-anything-v2-s | kernels | 4.0 ms / 129 / 166 | 5.7 ms / 164 / 218 | 5.8 ms / 177 / 214 | 5.1 ms / 462 / 480 |
| depth-pro | portable | 558.6 ms / 3317 / 4044 | 558.7 ms / 3323 / 4080 | 559.0 ms / 3337 / 4084 | 562.3 ms / 3451 / 4120 |
| depth-pro | kernels | 272.9 ms / 3621 / 4372 | 272.1 ms / 3628 / 4388 | 271.8 ms / 3641 / 4386 | 274.9 ms / 3754 / 4500 |
| marigold-depth-v1-1 | portable | 455.3 ms / 2898 / 3650 | 351.7 ms / 2648 / 3306 | 351.8 ms / 2661 / 3378 | 458.1 ms / 3033 / 3844 |
| marigold-depth-v1-1 | kernels | 433.0 ms / 2636 / 3464 | 324.3 ms / 2452 / 3070 | 324.9 ms / 2465 / 3142 | 435.6 ms / 2770 / 3552 |
| metric3d-vit-large | portable | 207.0 ms / 2219 / 2422 | 207.0 ms / 2228 / 2440 | 207.3 ms / 2242 / 2428 | 211.9 ms / 2401 / 2926 |
| metric3d-vit-large | kernels | 158.4 ms / 1471 / 1796 | 158.4 ms / 1478 / 1812 | 158.8 ms / 1492 / 1822 | 162.9 ms / 1616 / 2022 |
| midas-dpt-beit-large-512 | portable | 79.3 ms / 1729 / 1880 | 141.1 ms / 1870 / 2052 | 141.2 ms / 1883 / 2080 | 81.6 ms / 1863 / 1990 |
| midas-dpt-beit-large-512 | kernels | 52.1 ms / 922 / 1048 | 96.0 ms / 1012 / 1220 | 96.0 ms / 1025 / 1280 | 53.1 ms / 1077 / 1178 |
| moge-2-vitl-normal | portable | 175.5 ms / 2275 / 2508 | 178.8 ms / 2286 / 2518 | 179.2 ms / 2299 / 2528 | 191.5 ms / 2410 / 2562 |
| moge-2-vitl-normal | kernels | 74.3 ms / 1435 / 1572 | 77.5 ms / 1444 / 1588 | 76.0 ms / 1459 / 1644 | 90.5 ms / 1755 / 2174 |
| moge-2-vits-normal | portable | 88.6 ms / 1123 / 1418 | 85.4 ms / 1132 / 1392 | 86.7 ms / 1146 / 1446 | 98.6 ms / 1257 / 1464 |
| moge-2-vits-normal | kernels | 33.9 ms / 818 / 970 | 33.1 ms / 826 / 976 | 31.0 ms / 839 / 1010 | 46.9 ms / 1144 / 1566 |
| moge-3-vitl | portable | 657.4 ms / 2580 / 3070 | 633.4 ms / 2577 / 2792 | 620.0 ms / 2588 / 2804 | 655.1 ms / 2701 / 3066 |
| moge-3-vitl | kernels | 386.2 ms / 1570 / 1836 | 374.2 ms / 1557 / 1840 | 362.3 ms / 1566 / 1890 | 390.6 ms / 1838 / 2300 |
| unidepth-v2-vitl14 | portable | 61.2 ms / 1993 / 2266 | 130.9 ms / 2561 / 3154 | 131.1 ms / 2575 / 3150 | 134.4 ms / 2674 / 3266 |
| unidepth-v2-vitl14 | kernels | 32.5 ms / 1011 / 1110 | 69.1 ms / 1214 / 1378 | 69.4 ms / 1227 / 1430 | 72.6 ms / 1334 / 1706 |
| unik3d-vitl | portable | 115.1 ms / 2101 / 2474 | 124.2 ms / 2153 / 2570 | 124.8 ms / 2166 / 2582 | 128.7 ms / 2271 / 2486 |
| unik3d-vitl | kernels | 57.9 ms / 1076 / 1202 | 62.6 ms / 1101 / 1272 | 63.2 ms / 1115 / 1240 | 65.9 ms / 1226 / 1512 |
| zoedepth-nk | portable | 107.5 ms / 1857 / 1966 | 147.2 ms / 2046 / 2184 | 147.6 ms / 2107 / 2220 | 113.0 ms / 2451 / 2658 |
| zoedepth-nk | kernels | 74.9 ms / 1548 / 1802 | 105.6 ms / 1847 / 2138 | 106.2 ms / 1907 / 2230 | 80.3 ms / 2142 / 2486 |
| deeplabv3-r101-d8-ade20k | portable | 33.5 ms / 550 / 784 | 45.5 ms / 978 / 1282 | 52.4 ms / 1650 / 1966 | 103.8 ms / 7442 / 7698 |
| deeplabv3-r101-d8-ade20k | kernels | 20.0 ms / 583 / 718 | 28.5 ms / 1012 / 1196 | 35.5 ms / 1684 / 1880 | 90.5 ms / 7476 / 7708 |
| deeplabv3plus-r101-d8-ade20k | portable | 35.4 ms / 513 / 766 | 48.8 ms / 944 / 1272 | 55.8 ms / 1616 / 1944 | 105.7 ms / 7396 / 7644 |
| deeplabv3plus-r101-d8-ade20k | kernels | 23.8 ms / 546 / 716 | 34.0 ms / 977 / 1222 | 41.0 ms / 1649 / 1902 | 94.4 ms / 7429 / 7668 |
| fcn-r101-d8-ade20k | portable | 28.6 ms / 515 / 720 | 38.9 ms / 943 / 1230 | 46.0 ms / 1614 / 1894 | 99.2 ms / 7407 / 7662 |
| fcn-r101-d8-ade20k | kernels | 17.6 ms / 549 / 692 | 25.5 ms / 977 / 1154 | 32.5 ms / 1649 / 1854 | 88.2 ms / 7441 / 7682 |
| hrnet-w48-fcn-ade20k | portable | 31.0 ms / 523 / 696 | 36.1 ms / 954 / 1194 | 42.8 ms / 1626 / 1836 | 101.5 ms / 7406 / 7632 |
| hrnet-w48-fcn-ade20k | kernels | 23.0 ms / 523 / 652 | 27.7 ms / 955 / 1114 | 35.1 ms / 1627 / 1824 | 93.3 ms / 7406 / 7694 |
| pspnet-r101-d8-ade20k | portable | 31.8 ms / 513 / 810 | 46.8 ms / 941 / 1296 | 53.8 ms / 1613 / 1976 | 102.4 ms / 7406 / 7632 |
| pspnet-r101-d8-ade20k | kernels | 20.5 ms / 548 / 722 | 32.6 ms / 975 / 1178 | 39.5 ms / 1647 / 1838 | 91.2 ms / 7439 / 7668 |
| dinov2-b-linear-ade20k | portable | 22.5 ms / 922 / 1040 | 35.5 ms / 1217 / 1398 | 40.0 ms / 1230 / 1416 | 151.7 ms / 1289 / 1960 |
| dinov2-b-linear-ade20k | kernels | 18.3 ms / 929 / 1080 | 28.4 ms / 1224 / 1438 | 34.1 ms / 1237 / 1456 | 147.9 ms / 1296 / 2000 |
| dinov2-l-linear-ade20k | portable | 56.5 ms / 1330 / 1524 | 86.3 ms / 1624 / 1876 | 91.1 ms / 1638 / 1890 | 184.9 ms / 1697 / 2266 |
| dinov2-l-linear-ade20k | kernels | 39.6 ms / 1338 / 1526 | 62.8 ms / 1632 / 1884 | 67.7 ms / 1645 / 1894 | 168.5 ms / 1704 / 2264 |
| dinov2-s-linear-ade20k | portable | 12.6 ms / 792 / 834 | 20.1 ms / 1088 / 1154 | 25.1 ms / 1101 / 1168 | 141.7 ms / 1160 / 1522 |
| dinov2-s-linear-ade20k | kernels | 11.5 ms / 796 / 854 | 18.5 ms / 1092 / 1168 | 23.5 ms / 1105 / 1174 | 140.4 ms / 1163 / 1774 |
| eomt-l-dinov2-ade20k | portable | 50.3 ms / 1917 / 2484 | 55.4 ms / 2412 / 3506 | 63.2 ms / 3417 / 4834 | 129.2 ms / 14890 / 15016 |
| eomt-l-dinov2-ade20k | kernels | 38.5 ms / 1917 / 2484 | 43.8 ms / 2412 / 3506 | 51.5 ms / 3417 / 4834 | 117.4 ms / 14890 / 15016 |
| eomt-l-dinov3-ade20k | portable | 54.2 ms / 1916 / 2486 | 59.2 ms / 2411 / 3502 | 67.0 ms / 3415 / 4836 | 133.1 ms / 14889 / 15016 |
| eomt-l-dinov3-ade20k | kernels | 42.5 ms / 1916 / 2486 | 47.7 ms / 2411 / 3502 | 55.3 ms / 3415 / 4836 | 121.2 ms / 14889 / 15016 |
| mask2former-swinl-ade20k | portable | 71.9 ms / 1194 / 1912 | 96.1 ms / 1698 / 3182 | 104.1 ms / 2985 / 4500 | 152.8 ms / 14457 / 15480 |
| mask2former-swinl-ade20k | kernels | 45.3 ms / 1194 / 1816 | 60.5 ms / 1699 / 3056 | 68.5 ms / 2986 / 4374 | 126.0 ms / 14457 / 15344 |
| oneformer-swinl-ade20k | portable | 86.2 ms / 1295 / 1940 | 120.8 ms / 2220 / 3960 | 138.9 ms / 3772 / 6702 | OOM |
| oneformer-swinl-ade20k | kernels | 59.0 ms / 1295 / 1858 | 85.4 ms / 2222 / 3854 | 103.3 ms / 3772 / 6642 | OOM |
| sapiens-seg-0.3b-goliath | portable | 131.3 ms / 1621 / 1960 | 132.2 ms / 1627 / 1964 | 133.7 ms / 1641 / 1950 | 146.0 ms / 2254 / 3302 |
| sapiens-seg-0.3b-goliath | kernels | 92.6 ms / 1621 / 1960 | 93.5 ms / 1627 / 1964 | 95.2 ms / 1641 / 1950 | 107.5 ms / 2254 / 3302 |
| sapiens-seg-1b-goliath | portable | 274.9 ms / 3238 / 3718 | 275.7 ms / 3245 / 3734 | 277.0 ms / 3259 / 3730 | 289.7 ms / 3869 / 5076 |
| sapiens-seg-1b-goliath | kernels | 185.7 ms / 3238 / 3718 | 186.6 ms / 3245 / 3734 | 187.6 ms / 3259 / 3730 | 199.5 ms / 3869 / 5076 |
| segformer-b0-ade20k | portable | 7.5 ms / 237 / 290 | 11.9 ms / 454 / 518 | 18.5 ms / 798 / 864 | 74.7 ms / 3888 / 4048 |
| segformer-b0-ade20k | kernels | 7.3 ms / 237 / 290 | 11.6 ms / 454 / 518 | 18.2 ms / 798 / 864 | 74.5 ms / 3888 / 4048 |
| segformer-b5-ade20k | portable | 28.3 ms / 469 / 668 | 36.7 ms / 707 / 958 | 43.4 ms / 1050 / 1304 | 96.1 ms / 4063 / 4380 |
| segformer-b5-ade20k | kernels | 26.6 ms / 470 / 668 | 33.7 ms / 708 / 958 | 40.4 ms / 1051 / 1304 | 94.4 ms / 4064 / 4380 |
| sam-vit-b | portable | 48.6 ms / 367 / 482 | 48.6 ms / 374 / 482 | 49.0 ms / 387 / 484 | - |
| sam-vit-b | kernels | 37.3 ms / 364 / 424 | 37.4 ms / 370 / 404 | 38.1 ms / 384 / 432 | - |
| sam-vit-h | portable | 216.1 ms / 1490 / 1882 | 215.4 ms / 1496 / 1902 | 215.7 ms / 1510 / 1916 | - |
| sam-vit-h | kernels | 143.6 ms / 1459 / 1712 | 144.4 ms / 1466 / 1704 | 145.1 ms / 1479 / 1714 | - |
| sam-vit-l | portable | 117.2 ms / 814 / 1052 | 116.5 ms / 820 / 1060 | 117.5 ms / 834 / 1048 | - |
| sam-vit-l | kernels | 82.0 ms / 812 / 930 | 81.6 ms / 818 / 946 | 83.2 ms / 833 / 922 | - |
| sam2.1-hiera-base-plus | portable | 32.0 ms / 395 / 540 | 31.9 ms / 402 / 558 | 32.1 ms / 415 / 568 | - |
| sam2.1-hiera-base-plus | kernels | 25.1 ms / 408 / 472 | 25.1 ms / 415 / 490 | 25.3 ms / 428 / 472 | - |
| sam2.1-hiera-large | portable | 68.2 ms / 724 / 936 | 68.6 ms / 732 / 948 | 68.3 ms / 745 / 954 | - |
| sam2.1-hiera-large | kernels | 48.7 ms / 742 / 824 | 48.7 ms / 750 / 836 | 49.0 ms / 763 / 842 | - |
| sam2.1-hiera-small | portable | 18.7 ms / 298 / 420 | 19.1 ms / 305 / 422 | 18.9 ms / 319 / 426 | - |
| sam2.1-hiera-small | kernels | 15.4 ms / 310 / 356 | 15.5 ms / 317 / 358 | 15.6 ms / 331 / 346 | - |
| sam2.1-hiera-tiny | portable | 16.7 ms / 285 / 410 | 16.6 ms / 292 / 410 | 16.8 ms / 305 / 426 | - |
| sam2.1-hiera-tiny | kernels | 13.8 ms / 296 / 346 | 13.8 ms / 304 / 346 | 14.0 ms / 317 / 346 | - |
| sam3 | portable | 179.8 ms / 1928 / 2196 | 181.5 ms / 1935 / 2208 | 182.6 ms / 1948 / 2192 | - |
| sam3 | kernels | 116.3 ms / 2052 / 2460 | 116.5 ms / 2059 / 2420 | 116.9 ms / 2072 / 2490 | - |
| detr-r101-coco | portable | 14.4 ms / 278 / 332 | 17.6 ms / 307 / 356 | 18.3 ms / 320 / 416 | 23.8 ms / 636 / 676 |
| detr-r101-coco | kernels | 10.5 ms / 228 / 278 | 13.0 ms / 248 / 294 | 13.3 ms / 262 / 354 | 19.8 ms / 637 / 678 |
| detr-r50-coco | portable | 9.5 ms / 241 / 296 | 11.6 ms / 270 / 320 | 12.2 ms / 283 / 380 | 18.7 ms / 599 / 640 |
| detr-r50-coco | kernels | 7.7 ms / 191 / 240 | 8.9 ms / 211 / 256 | 9.8 ms / 225 / 316 | 16.7 ms / 600 / 640 |
| faster-rcnn-r50-fpn-3x-coco | portable | 18.5 ms / 309 / 420 | 21.5 ms / 338 / 426 | 22.2 ms / 352 / 454 | 27.2 ms / 602 / 734 |
| faster-rcnn-r50-fpn-3x-coco | kernels | 16.1 ms / 310 / 422 | 18.7 ms / 339 / 460 | 18.4 ms / 353 / 488 | 24.6 ms / 602 / 736 |
| mask-rcnn-r50-fpn-3x-coco | portable | 18.1 ms / 314 / 422 | 23.1 ms / 348 / 440 | 25.1 ms / 413 / 536 | 27.1 ms / 607 / 736 |
| mask-rcnn-r50-fpn-3x-coco | kernels | 15.6 ms / 315 / 422 | 19.6 ms / 349 / 472 | 22.3 ms / 414 / 568 | 24.6 ms / 607 / 736 |
| rtdetrv2-l-coco | portable | 24.1 ms / 171 / 202 | 15.4 ms / 176 / 204 | 13.2 ms / 204 / 226 | 18.2 ms / 574 / 698 |
| rtdetrv2-l-coco | kernels | 6.4 ms / 205 / 328 | 6.5 ms / 213 / 338 | 7.1 ms / 240 / 338 | 16.4 ms / 610 / 758 |
| rtdetrv2-m-coco | portable | 9.4 ms / 147 / 178 | 11.9 ms / 155 / 202 | 12.8 ms / 181 / 216 | 16.1 ms / 552 / 688 |
| rtdetrv2-m-coco | kernels | 5.1 ms / 182 / 312 | 5.3 ms / 189 / 324 | 5.8 ms / 216 / 336 | 15.1 ms / 586 / 752 |
| rtdetrv2-ms-coco | portable | 9.9 ms / 160 / 190 | 9.5 ms / 166 / 200 | 9.7 ms / 193 / 228 | 16.6 ms / 563 / 700 |
| rtdetrv2-ms-coco | kernels | 5.3 ms / 195 / 314 | 5.4 ms / 202 / 326 | 6.0 ms / 229 / 338 | 15.2 ms / 600 / 760 |
| rtdetrv2-s-coco | portable | 7.7 ms / 128 / 154 | 17.8 ms / 135 / 164 | 10.2 ms / 160 / 192 | 14.8 ms / 530 / 664 |
| rtdetrv2-s-coco | kernels | 4.0 ms / 161 / 288 | 4.1 ms / 168 / 300 | 4.7 ms / 194 / 312 | 14.0 ms / 564 / 728 |
| rtdetrv2-x-coco | portable | 15.4 ms / 237 / 264 | 23.5 ms / 244 / 274 | 18.4 ms / 266 / 302 | 22.8 ms / 636 / 774 |
| rtdetrv2-x-coco | kernels | 9.0 ms / 272 / 390 | 9.2 ms / 279 / 402 | 9.8 ms / 302 / 414 | 19.0 ms / 673 / 836 |
| dinov2-vitb14 | portable | 10.3 ms / 231 / 242 | 39.8 ms / 295 / 342 | 137.6 ms / 417 / 442 | 3232.4 ms (n=3) / 1460 / 1562 |
| dinov2-vitb14 | kernels | 7.5 ms / 243 / 262 | 28.0 ms / 324 / 362 | 94.7 ms / 480 / 542 | 2147.8 ms (n=3) / 1821 / 2002 |
| dinov2-vitb14-reg | portable | 10.1 ms / 238 / 254 | 39.9 ms / 303 / 368 | 140.2 ms / 423 / 544 | 3244.0 ms (n=3) / 1467 / 2244 |
| dinov2-vitb14-reg | kernels | 7.6 ms / 254 / 374 | 28.1 ms / 337 / 452 | 94.7 ms / 492 / 704 | 2147.6 ms (n=3) / 1833 / 2404 |
| dinov2-vitg14 | portable | 101.5 ms / 2273 / 2304 | 375.9 ms / 2410 / 2464 | 1169.8 ms / 2673 / 2764 | 23035.0 ms (n=3) / 4929 / 5404 |
| dinov2-vitg14 | kernels | 69.7 ms / 2296 / 2322 | 257.4 ms / 2469 / 2522 | 797.8 ms / 2801 / 2924 | 15248.1 ms (n=3) / 5650 / 6264 |
| dinov2-vitg14-reg | portable | 101.8 ms / 2294 / 2454 | 377.1 ms / 2432 / 2656 | 1184.7 ms / 2694 / 2984 | 23131.1 ms (n=3) / 4951 / 6020 |
| dinov2-vitg14-reg | kernels | 70.3 ms / 2298 / 2634 | 257.6 ms / 2472 / 2864 | 802.9 ms / 2803 / 3352 | 15244.2 ms (n=3) / 5654 / 6720 |
| dinov2-vitl14 | portable | 29.8 ms / 656 / 682 | 118.1 ms / 736 / 782 | 395.5 ms / 889 / 902 | 8770.0 ms (n=3) / 2205 / 2482 |
| dinov2-vitl14 | kernels | 19.6 ms / 671 / 702 | 77.1 ms / 775 / 802 | 252.1 ms / 974 / 1042 | 5331.0 ms (n=3) / 2686 / 2862 |
| dinov3-convnext-base | portable | 8.3 ms / 253 / 290 | 22.8 ms / 357 / 410 | 55.2 ms / 565 / 650 | 323.2 ms / 2310 / 2670 |
| dinov3-convnext-base | kernels | 6.5 ms / 279 / 360 | 15.1 ms / 372 / 572 | 33.5 ms / 502 / 548 | 190.9 ms / 1914 / 2048 |
| dinov3-convnext-large | portable | 14.8 ms / 484 / 510 | 42.5 ms / 633 / 766 | 99.9 ms / 932 / 1048 | 587.2 ms / 3435 / 4008 |
| dinov3-convnext-large | kernels | 11.0 ms / 505 / 642 | 26.7 ms / 635 / 950 | 61.0 ms / 837 / 926 | 357.8 ms / 2887 / 2946 |
| dinov3-convnext-small | portable | 5.9 ms / 169 / 188 | 15.6 ms / 250 / 266 | 37.9 ms / 413 / 486 | 220.9 ms / 1778 / 2106 |
| dinov3-convnext-small | kernels | 4.5 ms / 197 / 276 | 10.6 ms / 272 / 410 | 23.3 ms / 366 / 406 | 133.7 ms / 1474 / 1526 |
| dinov3-convnext-tiny | portable | 3.6 ms / 127 / 148 | 9.8 ms / 209 / 226 | 23.9 ms / 372 / 446 | 136.6 ms / 1737 / 2066 |
| dinov3-convnext-tiny | kernels | 2.7 ms / 156 / 236 | 6.5 ms / 231 / 368 | 14.6 ms / 324 / 366 | 82.1 ms / 1433 / 1484 |
| dinov3-vitb16 | portable | 8.8 ms / 224 / 242 | 31.6 ms / 281 / 304 | 102.4 ms / 389 / 464 | 2039.3 ms (n=3) / 1308 / 1502 |
| dinov3-vitb16 | kernels | 5.3 ms / 233 / 326 | 19.4 ms / 297 / 324 | 64.4 ms / 424 / 462 | 1322.5 ms / 1515 / 1702 |
| dinov3-vith16plus | portable | 60.7 ms / 1689 / 1724 | 218.8 ms / 1795 / 1846 | 636.2 ms / 1996 / 2106 | 10005.0 ms (n=3) / 3712 / 3944 |
| dinov3-vith16plus | kernels | 38.6 ms / 1690 / 1786 | 138.3 ms / 1789 / 1806 | 393.4 ms / 1984 / 2024 | 6021.9 ms (n=3) / 3643 / 3804 |
| dinov3-vitl16 | portable | 25.9 ms / 645 / 662 | 95.3 ms / 715 / 764 | 298.3 ms / 847 / 924 | 5549.3 ms (n=3) / 1975 / 2142 |
| dinov3-vitl16 | kernels | 15.6 ms / 656 / 726 | 54.9 ms / 738 / 764 | 174.5 ms / 899 / 922 | 3262.4 ms (n=3) / 2274 / 2402 |
| dinov3-vits16 | portable | 4.8 ms / 92 / 104 | 12.9 ms / 130 / 158 | 44.4 ms / 201 / 238 | 990.9 ms / 807 / 1036 |
| dinov3-vits16 | kernels | 2.3 ms / 97 / 168 | 8.6 ms / 145 / 242 | 29.7 ms / 213 / 236 | 644.8 ms / 876 / 1076 |
| fast-acvnet-plus-kitti15 | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.7 ms / 5054 / 8076 |
| fast-acvnet-plus-kitti15 | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.2 ms / 1006 / 1226 | 422.8 ms / 5743 / 7342 |
| fast-acvnet-plus-sceneflow | portable | 14.3 ms / 136 / 208 | 38.0 ms / 401 / 664 | 99.4 ms / 887 / 1418 | 623.7 ms / 5054 / 8076 |
| fast-acvnet-plus-sceneflow | kernels | 7.8 ms / 169 / 546 | 25.3 ms / 457 / 598 | 66.1 ms / 1006 / 1226 | 422.7 ms / 5743 / 7342 |
| fast-acvnet-sceneflow | portable | 14.0 ms / 137 / 208 | 31.1 ms / 402 / 664 | 78.2 ms / 890 / 1418 | 492.4 ms / 5077 / 8074 |
| fast-acvnet-sceneflow | kernels | 6.8 ms / 169 / 582 | 22.0 ms / 435 / 608 | 55.6 ms / 954 / 1372 | 357.6 ms / 5445 / 7618 |
| selective-igev-middlebury-train | portable | 190.7 ms / 282 / 362 | 575.4 ms / 769 / 1036 | 1420.5 ms / 1690 / 2030 | 8921.4 ms (n=3) / 10549 / 14320 |
| selective-igev-middlebury-train | kernels | 82.5 ms / 260 / 368 | 213.1 ms / 693 / 1010 | 472.4 ms / 1502 / 2046 | 2804.8 ms (n=3) / 6928 / 8800 |
| selective-igev-sceneflow | portable | 190.6 ms / 282 / 362 | 575.4 ms / 769 / 1036 | 1420.3 ms / 1690 / 2030 | 8917.6 ms (n=3) / 10549 / 14320 |
| selective-igev-sceneflow | kernels | 82.5 ms / 260 / 368 | 213.5 ms / 693 / 1010 | 473.3 ms / 1502 / 2046 | 2808.4 ms (n=3) / 6928 / 8800 |
| dinov2-vitl14-reg | portable | 30.1 ms / 656 / 682 | 118.2 ms / 736 / 782 | 396.1 ms / 889 / 982 | 8778.1 ms (n=3) / 2205 / 2622 |
| dinov2-vitl14-reg | kernels | 19.6 ms / 671 / 702 | 77.1 ms / 775 / 802 | 252.5 ms / 974 / 1042 | 5326.8 ms (n=3) / 2686 / 2982 |
| dinov2-vits14 | portable | 4.3 ms / 95 / 116 | 16.8 ms / 133 / 176 | 61.4 ms / 207 / 256 | 1574.5 ms / 843 / 976 |
| dinov2-vits14 | kernels | 3.3 ms / 100 / 116 | 12.0 ms / 148 / 176 | 39.0 ms / 239 / 276 | 957.1 ms / 1023 / 1116 |
| dinov2-vits14-reg | portable | 4.3 ms / 96 / 108 | 16.9 ms / 135 / 164 | 61.3 ms / 210 / 248 | 1582.1 ms / 846 / 1240 |
| dinov2-vits14-reg | kernels | 3.3 ms / 106 / 148 | 12.0 ms / 153 / 176 | 39.0 ms / 244 / 308 | 960.9 ms / 1029 / 1344 |
| vggt-1b | portable | 793.4 ms / 3868 / 5308 | 561.3 ms / 3710 / 4930 | 566.5 ms / 3920 / 5178 | 849.5 ms / 8219 / 10306 |
| vggt-1b | kernels | 559.0 ms / 3880 / 5474 | 398.6 ms / 3721 / 5156 | 404.0 ms / 3931 / 5390 | 615.0 ms / 8232 / 10518 |
| vggt-omega-1b-512 | portable | 722.8 ms / 3449 / 3866 | 723.2 ms / 3559 / 4038 | 725.7 ms / 3771 / 4366 | 757.0 ms / 5594 / 6894 |
| vggt-omega-1b-512 | kernels | 461.8 ms / 3449 / 3796 | 461.1 ms / 3560 / 3968 | 463.9 ms / 3771 / 4332 | 493.0 ms / 5594 / 6894 |
| mapanything-apache | portable | 680.9 ms / 3914 / 5848 | 493.1 ms / 3635 / 5196 | 498.1 ms / 3846 / 5442 | 734.6 ms / 7240 / 9442 |
| mapanything-apache | kernels | 569.7 ms / 3914 / 5846 | 415.3 ms / 3635 / 5196 | 419.9 ms / 3846 / 5442 | 622.3 ms / 7240 / 9440 |
| mvsanywhere | portable | 107.5 ms / 969 / 1500 | 75.8 ms / 954 / 1398 | 77.4 ms / 1166 / 1756 | 124.0 ms / 3110 / 4342 |
| mvsanywhere | kernels | 40.6 ms / 769 / 1006 | 31.0 ms / 838 / 1068 | 32.5 ms / 1049 / 1414 | 57.1 ms / 2908 / 4006 |
| patchmatchnet | portable | 43.1 ms / 439 / 522 | 154.4 ms / 1242 / 1534 | 408.6 ms / 2753 / 3346 | OOM |
| patchmatchnet | kernels | 9.2 ms / 253 / 328 | 24.4 ms / 693 / 942 | 56.7 ms / 1520 / 2016 | 378.6 ms / 8628 / 11332 |
| great-igev-eth3d | portable | 206.7 ms / 681 / 1030 | 627.7 ms / 1216 / 2262 | 1532.6 ms / 1688 / 2780 | 13067.4 ms (n=3) / 11364 / 15106 |
| great-igev-eth3d | kernels | 99.0 ms / 385 / 476 | 241.2 ms / 1041 / 1338 | 544.2 ms / 1506 / 2354 | 3753.5 ms (n=3) / 11736 / 15500 |
| great-igev-kitti | portable | 217.1 ms / 681 / 1030 | 628.2 ms / 1216 / 2262 | 1532.7 ms / 1688 / 2780 | 13065.3 ms (n=3) / 11364 / 15106 |
| great-igev-kitti | kernels | 93.2 ms / 385 / 476 | 240.9 ms / 1041 / 1338 | 544.6 ms / 1506 / 2356 | 3755.4 ms (n=3) / 11736 / 15500 |
| great-igev-middlebury | portable | 328.5 ms / 1082 / 1888 | 1025.5 ms / 1344 / 2262 | 2554.0 ms (n=3) / 2549 / 4266 | OOM |
| great-igev-middlebury | kernels | 131.4 ms / 942 / 1258 | 356.0 ms / 1279 / 1906 | 804.4 ms / 1825 / 2656 | 6324.4 ms (n=3) / 11920 / 15500 |
| great-igev-sceneflow | portable | 200.9 ms / 681 / 1030 | 624.1 ms / 1216 / 2262 | 1531.8 ms / 1688 / 2780 | 13071.3 ms (n=3) / 11364 / 15106 |
| great-igev-sceneflow | kernels | 93.5 ms / 385 / 476 | 241.1 ms / 1041 / 1338 | 545.3 ms / 1506 / 2354 | 3759.3 ms (n=3) / 11736 / 15500 |
| igev-plusplus-eth3d | portable | 201.5 ms / 354 / 614 | 606.5 ms / 987 / 1662 | 1534.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-eth3d | kernels | 88.7 ms / 357 / 1092 | 241.2 ms / 830 / 1210 | 534.3 ms / 1799 / 2464 | 3135.8 ms (n=3) / 8419 / 9752 |
| igev-plusplus-kitti12 | portable | 201.4 ms / 354 / 614 | 607.2 ms / 987 / 1662 | 1535.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-kitti12 | kernels | 89.1 ms / 357 / 1092 | 244.8 ms / 830 / 1210 | 535.9 ms / 1799 / 2464 | 3143.3 ms (n=3) / 8419 / 9750 |
| igev-plusplus-kitti15 | portable | 202.7 ms / 354 / 614 | 606.9 ms / 987 / 1662 | 1535.8 ms / 2172 / 3496 | OOM |
| igev-plusplus-kitti15 | kernels | 88.7 ms / 357 / 1092 | 240.2 ms / 830 / 1210 | 534.9 ms / 1799 / 2464 | 3141.8 ms (n=3) / 8419 / 9752 |
| igev-plusplus-middlebury | portable | 201.2 ms / 354 / 612 | 606.8 ms / 987 / 1662 | 1535.6 ms / 2172 / 3496 | OOM |
| igev-plusplus-middlebury | kernels | 88.7 ms / 357 / 1092 | 240.5 ms / 830 / 1210 | 535.0 ms / 1799 / 2464 | 3140.6 ms (n=3) / 8419 / 9752 |
| igev-plusplus-sceneflow | portable | 201.0 ms / 354 / 612 | 606.2 ms / 987 / 1662 | 1534.9 ms / 2172 / 3496 | OOM |
| igev-plusplus-sceneflow | kernels | 88.6 ms / 357 / 1092 | 240.6 ms / 830 / 1210 | 535.4 ms / 1799 / 2464 | 3143.9 ms (n=3) / 8419 / 9752 |

<!-- sizes:end -->
