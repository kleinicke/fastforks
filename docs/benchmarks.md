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
  CUDA and ROCm) or the peak process footprint (Apple silicon, where weights and activations share the system memory).
- Speed-ups name their baseline (`vs original`, `vs portable`, `vs fastest portable`). When a baseline is missing the
  cell stays empty; another baseline is not used in its place.
- States: `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised,
  `OOM` = out of memory, `–` = not tested.
- The reference device is the RTX 5060 Ti below: its rows are the regression floor (`tests/test_speed_floor.py`).
  Every other device is a survey with its own page; no floor is kept for it. Adding a device:
  "Devices" in `docs/data/README.md`.

<!-- devices:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed*) -->

## Other devices

- [Apple M1 Pro 16 GB](devices/apple-m1-pro-16gb.md) (mps): 261 models, 660 rows, measured 2026-11-06 to 2026-11-10.
- [NVIDIA L4](devices/l4-22gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA L40S](devices/l40s-44gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA A10](devices/a10-22gb.md) (cuda): 119 models, 238 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA A100-SXM4-40GB](devices/a100-sxm4-40gb.md) (cuda): 117 models, 234 rows, measured 2026-11-07 to 2026-11-08.
- [NVIDIA RTX PRO 6000](devices/rtx-pro-6000-96gb.md) (cuda): 174 models, 348 rows, measured 2026-11-08 to 2026-11-08.

### Across devices

Each task at its standard input (the overview's). A cell is the fastest validated fastforks mode on that device (`portable` or `kernels`, time and dtype), or why there is none; a row needs a survey device measured at that input. Cells compare only rows of the same input and protocol (`warm-median`: median of warm, synchronised calls after warm-up (at least 5; 2 for calls over 2 s on surveys)); no speed-up is derived across devices, as the modes and dtypes differ. – = not measured at this input.

#### Stereo, 994x1482

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](models/stereo.md#aanet) | 88.9 ms · kernels fp16 | 818 ms · kernels fp16 | 111 ms · kernels fp16 | 46.0 ms · kernels fp16 | 97.6 ms · kernels fp16 | 50.8 ms · kernels fp16 | 22.1 ms · kernels fp16 |
| [`aanet-sceneflow`](models/stereo.md#aanet) | 101 ms · kernels fp16 | 631 ms · kernels fp16 | 140 ms · kernels fp16 | 44.7 ms · kernels fp16 | 105 ms · kernels fp16 | 49.6 ms · kernels fp16 | 20.9 ms · kernels fp16 |
| [`bridgedepth-rvc-pretrain`](models/stereo.md#bridgedepth) | 354 ms · kernels fp16 | 3756 ms · kernels fp16 | – | – | – | 143 ms · kernels fp16 | 68.8 ms · kernels fp16 |
| [`bridgedepth-sf`](models/stereo.md#bridgedepth) | 348 ms · kernels fp16 | 3693 ms · kernels fp16 | 430 ms · kernels fp16 | 122 ms · kernels fp16 | 333 ms · kernels fp16 | 141 ms · kernels fp16 | 68.8 ms · kernels fp16 |
| [`c-fast-foundationstereo`](models/stereo.md#c_foundationstereo) | 135 ms · kernels fp16 | 4881 ms · portable fp32 | – | – | – | 85.1 ms · kernels fp16 | 37.3 ms · kernels fp16 |
| [`c-foundationstereo-s`](models/stereo.md#c_foundationstereo) | 1362 ms · kernels fp16 | OOM | 1908 ms · kernels fp16 | 551 ms · kernels fp16 | 1537 ms · kernels fp16 | 668 ms · kernels fp16 | 319 ms · kernels fp16 |
| [`cas-gwcnet-c-sceneflow`](models/stereo.md#cas_gwcnet) | 225 ms · kernels fp16 | 2489 ms · portable fp16 | – | – | – | 87.9 ms · kernels fp16 | 46.7 ms · kernels fp16 |
| [`cas-gwcnet-gc-sceneflow`](models/stereo.md#cas_gwcnet) | 358 ms · kernels fp16 | 2995 ms · portable fp16 | 500 ms · kernels fp16 | 159 ms · kernels fp16 | 345 ms · kernels fp16 | 136 ms · kernels fp16 | 78.8 ms · kernels fp16 |
| [`cfnet-sceneflow`](models/stereo.md#cfnet) | 401 ms · kernels fp16 | 2206 ms · kernels fp16 | 484 ms · kernels fp16 | 152 ms · kernels fp16 | 432 ms · kernels fp16 | 145 ms · kernels fp16 | 76.4 ms · kernels fp16 |
| [`coex-sceneflow`](models/stereo.md#coex) | 23.5 ms · kernels fp16 | 411 ms · portable fp16 | 32.4 ms · kernels fp16 | 11.9 ms · kernels fp16 | 27.3 ms · kernels fp16 | 16.3 ms · kernels fp16 | 6.43 ms · kernels fp16 |
| [`crestereo-eth3d`](models/stereo.md#crestereo) | 229 ms · kernels fp16 | 3342 ms · kernels fp16 | 318 ms · kernels fp16 | 102 ms · kernels fp16 | 255 ms · kernels fp16 | 108 ms · kernels fp16 | 60.9 ms · kernels fp16 |
| [`defom-stereo-vitl-sceneflow`](models/stereo.md#defom_stereo) | 659 ms · kernels fp16 | 8224 ms · kernels fp16 | 921 ms · kernels fp16 | 271 ms · kernels fp16 | 689 ms · kernels fp16 | 248 ms · kernels fp16 | 146 ms · kernels fp16 |
| [`dlnr-middlebury`](models/stereo.md#dlnr) | 214 ms · kernels fp16 | 2143 ms · portable fp16 | – | – | – | – | 48.6 ms · kernels fp16 |
| [`dlnr-sceneflow`](models/stereo.md#dlnr) | 214 ms · kernels fp16 | 2143 ms · portable fp16 | 293 ms · kernels fp16 | 94.6 ms · kernels fp16 | 224 ms · kernels fp16 | 95.4 ms · kernels fp16 | 48.6 ms · kernels fp16 |
| [`fast-acvnet-generalization`](models/stereo.md#fast_acvnet) | 38.2 ms · kernels fp16 | 944 ms · portable fp32 | 51.7 ms · kernels fp16 | 19.2 ms · kernels fp16 | 43.6 ms · kernels fp16 | 21.5 ms · kernels fp16 | 9.33 ms · kernels fp16 |
| [`fast-acvnet-kitti12`](models/stereo.md#fast_acvnet) | 38.2 ms · kernels fp16 | 942 ms · portable fp32 | – | – | – | – | 9.58 ms · kernels fp16 |
| [`fast-acvnet-kitti15`](models/stereo.md#fast_acvnet) | 38.2 ms · kernels fp16 | 943 ms · portable fp32 | – | – | – | – | 9.34 ms · kernels fp16 |
| [`fast-acvnet-plus-generalization`](models/stereo.md#fast_acvnet) | 46.4 ms · kernels fp16 | 1322 ms · portable fp32 | – | – | – | – | 11.7 ms · kernels fp16 |
| [`fast-acvnet-plus-kitti12`](models/stereo.md#fast_acvnet) | 46.4 ms · kernels fp16 | 1321 ms · portable fp32 | – | – | – | – | 11.5 ms · kernels fp16 |
| [`fast-acvnet-plus-kitti15`](models/stereo.md#fast_acvnet) | 46.4 ms · kernels fp16 | 1322 ms · portable fp32 | – | – | – | – | 11.6 ms · kernels fp16 |
| [`fast-acvnet-plus-sceneflow`](models/stereo.md#fast_acvnet) | 46.4 ms · kernels fp16 | 1322 ms · portable fp32 | 62.8 ms · kernels fp16 | 20.3 ms · kernels fp16 | 51.2 ms · kernels fp16 | 28.7 ms · kernels fp16 | 11.8 ms · kernels fp16 |
| [`fast-acvnet-sceneflow`](models/stereo.md#fast_acvnet) | 38.2 ms · kernels fp16 | 942 ms · portable fp32 | – | – | – | – | 9.55 ms · kernels fp16 |
| [`fast-foundationstereo`](models/stereo.md#c_foundationstereo) | 135 ms · kernels fp16 | 5195 ms · portable fp32 | 201 ms · kernels fp16 | 70.2 ms · kernels fp16 | 170 ms · kernels fp16 | 85.3 ms · kernels fp16 | 36.8 ms · kernels fp16 |
| [`foundationstereo-vit-l`](models/stereo.md#foundationstereo) | 1513 ms · kernels fp16 | 24480 ms · kernels fp16 | 2035 ms · kernels fp16 | 605 ms · kernels fp16 | 1674 ms · kernels fp16 | 680 ms · kernels fp16 | 337 ms · kernels fp16 |
| [`foundationstereo-vit-s`](models/stereo.md#foundationstereo) | 1120 ms · kernels fp16 | 19496 ms · kernels fp16 | 1713 ms · kernels fp16 | 487 ms · kernels fp16 | 1369 ms · kernels fp16 | 561 ms · kernels fp16 | 270 ms · kernels fp16 |
| [`great-igev-eth3d`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9636 ms · kernels fp16 | – | – | – | – | 97.2 ms · kernels fp16 |
| [`great-igev-kitti`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9634 ms · kernels fp16 | – | – | – | – | 97.4 ms · kernels fp16 |
| [`great-igev-middlebury`](models/stereo.md#great_igev) | 579 ms · kernels fp16 | 25368 ms · kernels fp16 | – | – | – | – | 143 ms · kernels fp16 |
| [`great-igev-sceneflow`](models/stereo.md#great_igev) | 393 ms · kernels fp16 | 9861 ms · kernels fp16 | 579 ms · kernels fp16 | 176 ms · kernels fp16 | 477 ms · kernels fp16 | 196 ms · kernels fp16 | 98.3 ms · kernels fp16 |
| [`gwcnet-g-sceneflow`](models/stereo.md#gwcnet) | 238 ms · kernels fp16 | 4340 ms · portable fp32 | – | – | – | – | 50.5 ms · kernels fp16 |
| [`gwcnet-gc-sceneflow`](models/stereo.md#gwcnet) | 257 ms · kernels fp16 | 4461 ms · portable fp32 | 342 ms · kernels fp16 | 109 ms · kernels fp16 | 244 ms · kernels fp16 | 97.7 ms · kernels fp16 | 56.7 ms · kernels fp16 |
| [`hitnet-eth3d`](models/stereo.md#hitnet) | 23.2 ms · kernels fp16 | 136 ms · kernels fp16 | 31.4 ms · kernels fp16 | 11.0 ms · kernels fp16 | 28.3 ms · kernels fp16 | 16.9 ms · kernels fp16 | 6.22 ms · kernels fp16 |
| [`hitnet-middlebury-d400`](models/stereo.md#hitnet) | 52.7 ms · kernels fp16 | 440 ms · kernels fp16 | – | – | – | – | 14.3 ms · kernels fp16 |
| [`igev-plusplus-eth3d`](models/stereo.md#igev_plusplus) | 393 ms · kernels fp16 | 6139 ms · kernels fp16 | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-kitti12`](models/stereo.md#igev_plusplus) | 393 ms · kernels fp16 | 6124 ms · kernels fp16 | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-kitti15`](models/stereo.md#igev_plusplus) | 393 ms · kernels fp16 | 6125 ms · kernels fp16 | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-middlebury`](models/stereo.md#igev_plusplus) | 393 ms · kernels fp16 | 6130 ms · kernels fp16 | – | – | – | – | 104 ms · kernels fp16 |
| [`igev-plusplus-sceneflow`](models/stereo.md#igev_plusplus) | 393 ms · kernels fp16 | 6135 ms · kernels fp16 | 547 ms · kernels fp16 | 177 ms · kernels fp16 | 463 ms · kernels fp16 | 186 ms · kernels fp16 | 104 ms · kernels fp16 |
| [`igev-stereo-eth3d`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 3998 ms · kernels fp16 | – | – | – | – | 72.6 ms · kernels fp16 |
| [`igev-stereo-kitti12`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4006 ms · kernels fp16 | – | – | – | – | 72.2 ms · kernels fp16 |
| [`igev-stereo-kitti15`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4008 ms · kernels fp16 | – | – | – | – | 72.3 ms · kernels fp16 |
| [`igev-stereo-middlebury`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4007 ms · kernels fp16 | – | – | – | – | 72.6 ms · kernels fp16 |
| [`igev-stereo-sceneflow`](models/stereo.md#igev_stereo) | 288 ms · kernels fp16 | 4001 ms · kernels fp16 | 430 ms · kernels fp16 | 134 ms · kernels fp16 | 349 ms · kernels fp16 | 137 ms · kernels fp16 | 72.6 ms · kernels fp16 |
| [`las2-h`](models/stereo.md#las2) | 53.3 ms · kernels fp16 | 734 ms · kernels fp16 | 77.0 ms · kernels fp16 | 36.3 ms · kernels fp16 | 65.1 ms · kernels fp16 | 38.3 ms · kernels fp16 | 15.5 ms · kernels fp16 |
| [`las2-l`](models/stereo.md#las2) | 29.9 ms · kernels fp16 | 519 ms · kernels fp16 | – | – | – | – | 8.96 ms · kernels fp16 |
| [`las2-m`](models/stereo.md#las2) | 18.3 ms · kernels fp16 | 281 ms · kernels fp16 | – | – | – | – | 6.91 ms · kernels fp16 |
| [`las2-s`](models/stereo.md#las2) | 14.3 ms · kernels fp16 | 198 ms · kernels fp16 | 23.0 ms · kernels fp16 | 15.3 ms · kernels fp16 | 16.9 ms · kernels fp16 | 15.7 ms · kernels fp16 | 4.99 ms · kernels fp16 |
| [`mocha-v1-sceneflow`](models/stereo.md#mocha_stereo) | 455 ms · kernels fp16 | 6693 ms · kernels fp16 | 631 ms · kernels fp16 | 224 ms · kernels fp16 | 527 ms · kernels fp16 | 227 ms · kernels fp16 | 116 ms · kernels fp16 |
| [`mocha-v2-sceneflow`](models/stereo.md#mocha_stereo) | 448 ms · kernels fp16 | 6501 ms · kernels fp16 | – | – | – | – | 115 ms · kernels fp16 |
| [`monster-mix-all`](models/stereo.md#monster) | 668 ms · kernels fp16 | 9646 ms · kernels fp16 | – | – | – | – | 164 ms · kernels fp16 |
| [`monster-plusplus-mix-all`](models/stereo.md#monster_plusplus) | 698 ms · kernels fp16 | 10202 ms · kernels fp16 | 949 ms · kernels fp16 | 288 ms · kernels fp16 | 792 ms · kernels fp16 | 289 ms · kernels fp16 | 167 ms · kernels fp16 |
| [`monster-sceneflow`](models/stereo.md#monster) | 668 ms · kernels fp16 | 9642 ms · kernels fp16 | 951 ms · kernels fp16 | 287 ms · kernels fp16 | 783 ms · kernels fp16 | 286 ms · kernels fp16 | 164 ms · kernels fp16 |
| [`nmrf-sceneflow`](models/stereo.md#nmrf) | 196 ms · kernels fp16 | 1204 ms · kernels fp16 | 239 ms · kernels fp16 | 64.6 ms · kernels fp16 | 173 ms · kernels fp16 | 84.4 ms · kernels fp16 | 37.7 ms · kernels fp16 |
| [`psmnet-sceneflow`](models/stereo.md#psmnet) | 232 ms · kernels fp16 | 4615 ms · portable fp32 | 290 ms · kernels fp16 | 90.6 ms · kernels fp16 | 218 ms · kernels fp16 | 91.3 ms · kernels fp16 | 48.9 ms · kernels fp16 |
| [`raft-stereo-eth3d`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3638 ms · kernels fp16 | – | – | – | – | 63.6 ms · kernels fp16 |
| [`raft-stereo-middlebury`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3634 ms · kernels fp16 | 416 ms · kernels fp16 | 121 ms · kernels fp16 | 313 ms · kernels fp16 | 110 ms · kernels fp16 | 63.6 ms · kernels fp16 |
| [`raft-stereo-sceneflow`](models/stereo.md#raft_stereo) | 273 ms · kernels fp16 | 3637 ms · kernels fp16 | – | – | – | – | 63.6 ms · kernels fp16 |
| [`rt-monster-plusplus-mix-all`](models/stereo.md#rt_monster_plusplus) | 94.1 ms · kernels fp16 | 663 ms · portable fp32 | 125 ms · kernels fp16 | 48.3 ms · portable fp16 | 105 ms · kernels fp16 | 49.4 ms · portable fp16 | 23.6 ms · kernels fp16 |
| [`s2m2-l`](models/stereo.md#s2m2) | 376 ms · kernels fp16 | 6590 ms · portable fp16 | – | – | – | – | 91.7 ms · kernels fp16 |
| [`s2m2-m`](models/stereo.md#s2m2) | 248 ms · kernels fp16 | 3620 ms · portable fp16 | – | – | – | – | 61.1 ms · kernels fp16 |
| [`s2m2-s`](models/stereo.md#s2m2) | 113 ms · kernels fp16 | 1607 ms · portable fp16 | 163 ms · kernels fp16 | 56.8 ms · kernels fp16 | 137 ms · kernels fp16 | 62.4 ms · kernels fp16 | 29.0 ms · kernels fp16 |
| [`s2m2-xl`](models/stereo.md#s2m2) | 682 ms · kernels fp16 | 13398 ms · portable fp16 | 1152 ms · kernels fp16 | 310 ms · kernels fp16 | 822 ms · kernels fp16 | 312 ms · kernels fp16 | 168 ms · kernels fp16 |
| [`selective-igev-eth3d-finetune`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5708 ms · kernels fp16 | – | – | – | – | 89.3 ms · kernels fp16 |
| [`selective-igev-eth3d-train`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | – | – | – | – | 89.3 ms · kernels fp16 |
| [`selective-igev-kitti12`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5709 ms · kernels fp16 | – | – | – | – | 89.2 ms · kernels fp16 |
| [`selective-igev-kitti15`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | – | – | – | – | 89.1 ms · kernels fp16 |
| [`selective-igev-middlebury-finetune`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5778 ms · kernels fp16 | – | – | – | – | 89.4 ms · kernels fp16 |
| [`selective-igev-middlebury-train`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5738 ms · kernels fp16 | – | – | – | – | 89.2 ms · kernels fp16 |
| [`selective-igev-sceneflow`](models/stereo.md#selective_igev) | 349 ms · kernels fp16 | 5710 ms · kernels fp16 | 515 ms · kernels fp16 | 163 ms · kernels fp16 | 410 ms · kernels fp16 | 168 ms · kernels fp16 | 89.2 ms · kernels fp16 |
| [`waft-stereo-b`](models/stereo.md#waft_stereo) | 393 ms · kernels fp16 | 11022 ms · portable fp32 | 574 ms · kernels fp16 | 146 ms · kernels fp16 | 422 ms · kernels fp16 | 173 ms · kernels fp16 | 85.7 ms · kernels fp16 |
| [`waft-stereo-s`](models/stereo.md#waft_stereo) | 284 ms · kernels fp16 | 3360 ms · portable fp32 | 358 ms · kernels fp16 | 100 ms · kernels fp16 | 283 ms · kernels fp16 | 125 ms · kernels fp16 | 60.1 ms · kernels fp16 |

#### Depth, 994x1482

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`da3-giant-1.1`](models/depth.md#depth_anything_3) | 44.2 ms · kernels fp16 | 759 ms · portable fp16 | 63.2 ms · kernels fp16 | 22.5 ms · kernels fp16 | 60.1 ms · kernels fp16 | 25.0 ms · kernels fp16 | 14.2 ms · kernels fp16 |
| [`da3-large-1.1`](models/depth.md#depth_anything_3) | 17.1 ms · kernels fp16 | 327 ms · portable fp16 | 22.4 ms · kernels fp16 | 8.49 ms · kernels fp16 | 21.1 ms · kernels fp16 | – | 6.59 ms · kernels fp16 |
| [`da3metric-large`](models/depth.md#depth_anything_3) | 14.7 ms · kernels fp16 | 318 ms · portable fp16 | 19.5 ms · kernels fp16 | 7.78 ms · kernels fp16 | 18.6 ms · kernels fp16 | – | 6.04 ms · kernels fp16 |
| [`da3mono-large`](models/depth.md#depth_anything_3) | 14.6 ms · kernels fp16 | 272 ms · portable fp16 | 19.2 ms · kernels fp16 | 7.66 ms · kernels fp16 | 18.5 ms · kernels fp16 | 12.8 ms · kernels fp16 | 5.81 ms · kernels fp16 |
| [`da3nested-giant-large-1.1`](models/depth.md#depth_anything_3) | 58.2 ms · kernels fp16 | 1184 ms · portable fp16 | 81.9 ms · kernels fp16 | 30.0 ms · kernels fp16 | 78.1 ms · kernels fp16 | – | 19.7 ms · kernels fp16 |
| [`depth-anything-v2-b`](models/depth.md#depth_anything_v2) | 10.5 ms · kernels fp16 | 212 ms · portable fp32 | 14.6 ms · kernels fp16 | 4.67 ms · kernels fp16 | 13.7 ms · kernels fp16 | – | 4.00 ms · kernels fp16 |
| [`depth-anything-v2-l`](models/depth.md#depth_anything_v2) | 31.1 ms · kernels fp16 | 622 ms · portable fp32 | 42.7 ms · kernels fp16 | 13.6 ms · kernels fp16 | 40.1 ms · kernels fp16 | 16.1 ms · kernels fp16 | 8.61 ms · kernels fp16 |
| [`depth-anything-v2-metric-hypersim-b`](models/depth.md#depth_anything_v2) | – | 211 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-metric-hypersim-l`](models/depth.md#depth_anything_v2) | – | 628 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-metric-hypersim-s`](models/depth.md#depth_anything_v2) | – | 87.8 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-b`](models/depth.md#depth_anything_v2) | – | 211 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-l`](models/depth.md#depth_anything_v2) | – | 623 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-metric-vkitti-s`](models/depth.md#depth_anything_v2) | – | 90.0 ms · portable fp32 | – | – | – | – | – |
| [`depth-anything-v2-s`](models/depth.md#depth_anything_v2) | 4.51 ms · kernels fp16 | 87.3 ms · portable fp32 | 5.92 ms · kernels fp16 | 2.24 ms · kernels fp16 | 5.43 ms · kernels fp16 | 4.63 ms · kernels fp16 | 2.02 ms · kernels fp16 |
| [`depth-pro`](models/depth.md#depth_pro) | 271 ms · kernels fp16 | 5026 ms · portable fp16 | 472 ms · kernels fp16 | 133 ms · kernels fp16 | 359 ms · kernels fp16 | 119 ms · kernels fp16 | 68.9 ms · kernels fp16 |
| [`distill-any-depth-b`](models/depth.md#depth_anything_v2) | – | 212 ms · portable fp32 | – | – | – | – | – |
| [`distill-any-depth-l`](models/depth.md#depth_anything_v2) | – | 622 ms · portable fp32 | – | – | – | – | – |
| [`distill-any-depth-s`](models/depth.md#depth_anything_v2) | – | 87.3 ms · portable fp32 | – | – | – | – | – |
| [`marigold-depth-v1-1`](models/depth.md#marigold) | 383 ms · kernels fp16 | 2498 ms · portable fp16 | 346 ms · kernels fp16 | 128 ms · kernels fp16 | 260 ms · kernels fp16 | 177 ms · portable fp16 | 74.8 ms · kernels fp16 |
| [`metric3d-vit-large`](models/depth.md#metric3d) | 158 ms · kernels fp16 | 1822 ms · portable fp32 | 193 ms · kernels fp16 | 56.9 ms · kernels fp16 | 166 ms · kernels fp16 | 61.5 ms · kernels fp16 | 38.9 ms · kernels fp16 |
| [`midas-dpt-beit-large-512`](models/depth.md#midas) | 63.9 ms · kernels fp16 | 635 ms · portable fp16 | 91.5 ms · kernels fp16 | 26.8 ms · kernels fp16 | 73.1 ms · portable fp16 | 28.5 ms · kernels fp16 | 17.8 ms · portable fp16 |
| [`moge-2-vitl-normal`](models/depth.md#moge) | 73.4 ms · kernels fp16 | 1454 ms · kernels fp32 | 103 ms · kernels fp16 | 35.2 ms · kernels fp16 | 95.2 ms · kernels fp16 | – | 24.1 ms · kernels fp16 |
| [`moge-2-vits-normal`](models/depth.md#moge) | 29.7 ms · kernels fp16 | 363 ms · portable fp32 | 44.0 ms · kernels fp16 | 18.0 ms · kernels fp16 | 35.6 ms · kernels fp16 | 22.1 ms · kernels fp16 | 11.0 ms · kernels fp16 |
| [`moge-3-vitl`](models/depth.md#moge) | – | 2764 ms · kernels fp32 | 284 ms · kernels fp16 | 93.5 ms · kernels fp16 | 276 ms · kernels fp16 | 114 ms · kernels fp16 | 61.5 ms · kernels fp16 |
| [`unidepth-v2-vitl14`](models/depth.md#unidepth_v2) | 66.4 ms · kernels fp16 | 1260 ms · portable fp32 | 85.8 ms · kernels fp16 | 25.4 ms · kernels fp16 | 77.0 ms · kernels fp16 | 26.6 ms · kernels fp16 | 18.4 ms · portable fp16 |
| [`unik3d-vitl`](models/depth.md#unik3d) | 62.9 ms · kernels fp16 | 1217 ms · portable fp32 | 88.9 ms · kernels fp16 | 27.7 ms · kernels fp16 | 78.4 ms · kernels fp16 | 29.8 ms · kernels fp16 | 19.1 ms · kernels fp16 |
| [`zoedepth-nk`](models/depth.md#zoedepth) | 88.9 ms · kernels fp16 | 778 ms · portable fp32 | 116 ms · kernels fp16 | 32.1 ms · kernels fp16 | 92.9 ms · kernels fp16 | 41.2 ms · kernels fp16 | 19.2 ms · kernels fp16 |

#### Classification, 224x224

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](models/classification.md#alexnet) | 0.660 ms · kernels fp16 | 3.04 ms · portable fp16 | 1.03 ms · kernels fp16 | 0.544 ms · kernels fp16 | 0.616 ms · kernels fp16 | 0.591 ms · kernels fp16 | 0.240 ms · kernels fp16 |
| [`convnext_base.fb_in1k`](models/classification.md#convnext) | – | 27.7 ms · portable fp32 | – | – | – | – | – |
| [`convnext_base.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 27.4 ms · portable fp32 | – | – | – | – | – |
| [`convnext_base.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 50.8 ms · portable fp32 | – | – | – | – | – |
| [`convnext_large.fb_in1k`](models/classification.md#convnext) | – | 41.3 ms · portable fp32 | – | – | – | – | – |
| [`convnext_large.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 41.3 ms · portable fp32 | – | – | – | – | – |
| [`convnext_large.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 88.7 ms · portable fp32 | – | – | – | – | – |
| [`convnext_small.fb_in1k`](models/classification.md#convnext) | – | 19.1 ms · portable fp32 | – | – | – | – | – |
| [`convnext_small.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 19.4 ms · portable fp32 | – | – | – | – | – |
| [`convnext_small.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 33.5 ms · portable fp32 | – | – | – | – | – |
| [`convnext_tiny.fb_in22k_ft_in1k`](models/classification.md#convnext) | – | 11.4 ms · portable fp32 | – | – | – | – | – |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](models/classification.md#convnext) | – | 20.4 ms · portable fp32 | – | – | – | – | – |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](models/classification.md#convnextv2) | 3.07 ms · kernels fp16 | 36.1 ms · portable fp32 | 3.68 ms · kernels fp16 | 2.64 ms · kernels fp16 | 3.79 ms · kernels fp16 | 5.03 ms · kernels fp16 | 2.83 ms · kernels fp16 |
| [`deit_base_patch16_224.fb_in1k`](models/classification.md#deit) | 1.79 ms · kernels fp16 | 16.9 ms · portable fp32 | 1.90 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.77 ms · kernels fp16 | 2.21 ms · kernels fp16 | 1.02 ms · kernels fp16 |
| [`densenet121.tv_in1k`](models/classification.md#densenet) | 1.57 ms · kernels fp16 | 25.4 ms · portable fp32 | 1.80 ms · kernels fp16 | 1.40 ms · kernels fp16 | 1.91 ms · kernels fp16 | 2.69 ms · kernels fp16 | 1.31 ms · kernels fp16 |
| [`dinov2_vitl14_lc`](models/classification.md#dinov2_linear_cls) | 5.24 ms · kernels fp16 | 64.5 ms · portable fp32 | 6.65 ms · kernels fp16 | 2.76 ms · kernels fp16 | 5.94 ms · portable fp16 | 4.71 ms · kernels fp16 | 2.22 ms · portable fp16 |
| [`efficientnet_b0.tv_in1k`](models/classification.md#efficientnet) | 0.870 ms · kernels fp16 | 13.0 ms · portable fp32 | 1.14 ms · kernels fp16 | 0.844 ms · kernels fp16 | 1.09 ms · kernels fp16 | 1.51 ms · kernels fp16 | 0.736 ms · kernels fp16 |
| [`efficientnet_b4.tv_in1k`](models/classification.md#efficientnet) | 2.19 ms · kernels fp16 | 38.3 ms · portable fp32 | 2.45 ms · kernels fp16 | 1.70 ms · kernels fp16 | 2.55 ms · kernels fp16 | 3.30 ms · kernels fp16 | 1.61 ms · kernels fp16 |
| [`efficientnet_v2_s.tv_in1k`](models/classification.md#efficientnetv2) | 2.11 ms · kernels fp16 | 34.0 ms · portable fp32 | 2.43 ms · kernels fp16 | 1.74 ms · kernels fp16 | 2.47 ms · kernels fp16 | 3.29 ms · kernels fp16 | 1.62 ms · kernels fp16 |
| [`efficientvit_b2.r288_in1k`](models/classification.md#efficientvit) | 1.20 ms · kernels fp16 | 18.6 ms · portable fp32 | 1.55 ms · kernels fp16 | 1.01 ms · kernels fp16 | 1.40 ms · kernels fp16 | 1.89 ms · kernels fp16 | 0.926 ms · kernels fp16 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](models/classification.md#eva02) | 14.2 ms · kernels fp16 | 235 ms · portable fp16 | 19.6 ms · kernels fp16 | 6.82 ms · kernels fp16 | 16.9 ms · kernels fp16 | 10.3 ms · kernels fp16 | 5.46 ms · kernels fp16 |
| [`fastvit_sa24.apple_dist_in1k`](models/classification.md#fastvit) | – | 16.8 ms · portable fp32 | 1.76 ms · kernels fp16 | 1.27 ms · kernels fp16 | 1.73 ms · kernels fp16 | 2.49 ms · kernels fp16 | 1.31 ms · kernels fp16 |
| [`inception_v3.tv_in1k`](models/classification.md#inception_v3) | 1.56 ms · kernels fp16 | 17.4 ms · portable fp32 | 1.76 ms · kernels fp16 | 1.31 ms · kernels fp16 | 1.74 ms · kernels fp16 | 2.40 ms · kernels fp16 | 1.26 ms · kernels fp16 |
| [`mobilenet_v2.tv2_in1k`](models/classification.md#mobilenetv2) | 0.520 ms · kernels fp16 | 6.48 ms · portable fp32 | 0.695 ms · kernels fp16 | 0.506 ms · kernels fp16 | 0.580 ms · kernels fp16 | 0.768 ms · kernels fp16 | 0.346 ms · kernels fp16 |
| [`mobilenet_v3_large.tv2_in1k`](models/classification.md#mobilenetv3) | 0.630 ms · kernels fp16 | 8.54 ms · portable fp32 | 0.853 ms · kernels fp16 | 0.632 ms · kernels fp16 | 0.777 ms · kernels fp16 | 1.05 ms · kernels fp16 | 0.489 ms · kernels fp16 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](models/classification.md#mobilenetv1) | 0.430 ms · kernels fp16 | 4.29 ms · portable fp32 | 0.593 ms · kernels fp16 | 0.413 ms · kernels fp16 | 0.448 ms · kernels fp16 | 0.574 ms · kernels fp16 | 0.253 ms · kernels fp16 |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](models/classification.md#mobilenetv4) | 0.750 ms · kernels fp16 | 9.58 ms · portable fp32 | 0.955 ms · kernels fp16 | 0.722 ms · kernels fp16 | 0.900 ms · kernels fp16 | 1.19 ms · kernels fp16 | 0.592 ms · kernels fp16 |
| [`regnety_160.tv2_in1k`](models/classification.md#regnet) | 1.96 ms · kernels fp16 | 33.1 ms · portable fp16 | 2.37 ms · kernels fp16 | 1.58 ms · kernels fp16 | 2.27 ms · kernels fp16 | 2.70 ms · kernels fp16 | 1.46 ms · kernels fp16 |
| [`resnet152.tv2_in1k`](models/classification.md#resnet) | 2.17 ms · kernels fp16 | 25.7 ms · portable fp32 | 2.32 ms · kernels fp16 | 1.84 ms · kernels fp16 | 2.48 ms · kernels fp16 | 3.42 ms · kernels fp16 | 1.82 ms · kernels fp16 |
| [`resnet18.tv_in1k`](models/classification.md#resnet) | 0.550 ms · kernels fp16 | 4.67 ms · portable fp32 | 0.718 ms · kernels fp16 | 0.519 ms · kernels fp16 | 0.609 ms · kernels fp16 | 0.797 ms · kernels fp16 | 0.370 ms · kernels fp16 |
| [`resnet50.tv2_in1k`](models/classification.md#resnet) | 0.990 ms · kernels fp16 | 11.2 ms · portable fp32 | – | – | – | – | 0.666 ms · kernels fp16 |
| [`shufflenet_v2_x1_0.tv_in1k`](models/classification.md#shufflenetv2) | 0.840 ms · kernels fp16 | 6.33 ms · portable fp32 | 1.13 ms · kernels fp16 | 0.874 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.63 ms · kernels fp16 | 0.811 ms · kernels fp16 |
| [`squeezenet1_1.tv_in1k`](models/classification.md#squeezenet) | 0.420 ms · kernels fp16 | 3.72 ms · portable fp16 | 0.590 ms · kernels fp16 | 0.437 ms · kernels fp16 | 0.461 ms · kernels fp16 | 0.592 ms · kernels fp16 | 0.269 ms · kernels fp16 |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](models/classification.md#swin) | 2.45 ms · kernels fp16 | 28.2 ms · portable fp32 | 2.92 ms · kernels fp16 | 2.02 ms · kernels fp16 | 2.80 ms · kernels fp16 | 3.79 ms · kernels fp16 | 1.91 ms · kernels fp16 |
| [`vgg16.tv_in1k`](models/classification.md#vgg) | 1.56 ms · kernels fp16 | 11.7 ms · portable fp16 | 2.10 ms · kernels fp16 | 1.03 ms · kernels fp16 | 1.47 ms · kernels fp16 | 1.15 ms · kernels fp16 | 0.622 ms · kernels fp16 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](models/classification.md#vit) | 1.78 ms · kernels fp16 | 16.9 ms · portable fp32 | 1.92 ms · kernels fp16 | 1.22 ms · kernels fp16 | 1.77 ms · kernels fp16 | 2.21 ms · kernels fp16 | 1.02 ms · kernels fp16 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](models/classification.md#siglip) | 12.9 ms · kernels fp16 | 197 ms · portable fp16 | 15.8 ms · kernels fp16 | 6.99 ms · kernels fp16 | 16.0 ms · kernels fp16 | 7.96 ms · portable fp16 | 4.80 ms · portable fp16 |

#### Segmentation, 1024x2048

| model | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](models/segmentation.md#mmseg) | 528 ms · portable fp16 | – | – | – | – | 6.03 ms · kernels fp16 |
| [`deeplabv3plus-r101-d8-ade20k`](models/segmentation.md#mmseg) | 430 ms · portable fp16 | 24.9 ms · kernels fp16 | 7.55 ms · kernels fp16 | 20.8 ms · kernels fp16 | 13.0 ms · kernels fp16 | 5.21 ms · kernels fp16 |
| [`dinov2-b-linear-ade20k`](models/segmentation.md#dinov2_linear) | 409 ms · portable fp32 | – | – | – | – | 6.86 ms · kernels fp16 |
| [`dinov2-l-linear-ade20k`](models/segmentation.md#dinov2_linear) | 1032 ms · portable fp32 | 81.8 ms · kernels fp16 | 22.8 ms · kernels fp16 | 67.1 ms · kernels fp16 | 22.3 ms · kernels fp16 | 13.8 ms · kernels fp16 |
| [`dinov2-s-linear-ade20k`](models/segmentation.md#dinov2_linear) | 224 ms · portable fp32 | 16.6 ms · kernels fp16 | 6.60 ms · kernels fp16 | 13.1 ms · kernels fp16 | 7.64 ms · kernels fp16 | 3.90 ms · kernels fp16 |
| [`eomt-l-dinov2-ade20k`](models/segmentation.md#eomt) | 540 ms · portable fp16 | 47.3 ms · kernels fp16 | 13.9 ms · kernels fp16 | 38.3 ms · kernels fp16 | 15.0 ms · kernels fp16 | 8.51 ms · kernels fp16 |
| [`eomt-l-dinov3-ade20k`](models/segmentation.md#eomt) | 575 ms · portable fp16 | 54.3 ms · kernels fp16 | 15.4 ms · kernels fp16 | 44.6 ms · kernels fp16 | 17.8 ms · kernels fp16 | 9.54 ms · kernels fp16 |
| [`fcn-r101-d8-ade20k`](models/segmentation.md#mmseg) | 405 ms · portable fp16 | 21.8 ms · kernels fp16 | 6.92 ms · kernels fp16 | 19.5 ms · kernels fp16 | 11.8 ms · kernels fp16 | 5.21 ms · kernels fp16 |
| [`hrnet-w48-fcn-ade20k`](models/segmentation.md#mmseg) | 228 ms · portable fp16 | – | – | – | – | 4.64 ms · kernels fp16 |
| [`mask2former-swinl-ade20k`](models/segmentation.md#mask2former) | 909 ms · portable fp32 | 77.8 ms · kernels fp16 | 32.8 ms · kernels fp16 | 67.4 ms · kernels fp16 | 35.3 ms · kernels fp16 | 16.9 ms · kernels fp16 |
| [`oneformer-swinl-ade20k`](models/segmentation.md#oneformer) | 1127 ms · portable fp32 | 134 ms · kernels fp16 | 47.2 ms · kernels fp16 | 109 ms · kernels fp16 | 58.9 ms · kernels fp16 | 31.4 ms · kernels fp16 |
| [`pspnet-r101-d8-ade20k`](models/segmentation.md#mmseg) | 411 ms · portable fp16 | – | – | – | – | 10.3 ms · kernels fp16 |
| [`sapiens-seg-0.3b-goliath`](models/segmentation.md#sapiens) | 1517 ms · portable fp16 | 135 ms · kernels fp16 | 39.7 ms · kernels fp16 | 97.7 ms · kernels fp16 | 34.4 ms · kernels fp16 | 24.4 ms · portable fp16 |
| [`sapiens-seg-1b-goliath`](models/segmentation.md#sapiens) | 3113 ms · portable fp16 | 305 ms · kernels fp16 | 81.4 ms · kernels fp16 | 200 ms · kernels fp16 | 67.8 ms · kernels fp16 | 45.5 ms · portable fp16 |
| [`segformer-b0-ade20k`](models/segmentation.md#segformer) | 106 ms · portable fp32 | 3.91 ms · kernels fp16 | 1.76 ms · kernels fp16 | 3.79 ms · kernels fp16 | 3.43 ms · kernels fp16 | 1.24 ms · kernels fp16 |
| [`segformer-b5-ade20k`](models/segmentation.md#segformer) | 428 ms · portable fp32 | 34.5 ms · kernels fp16 | 9.82 ms · kernels fp16 | 28.5 ms · kernels fp16 | 18.3 ms · kernels fp16 | 8.22 ms · kernels fp16 |

#### Promptable segmentation, 1024x1024

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`efficientvit-sam-l0`](models/promptable_segmentation.md#efficientvit_sam) | – | 50.6 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-l1`](models/promptable_segmentation.md#efficientvit_sam) | – | 75.2 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-l2`](models/promptable_segmentation.md#efficientvit_sam) | – | 91.2 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-xl0`](models/promptable_segmentation.md#efficientvit_sam) | – | 167 ms · portable fp32 | – | – | – | – | – |
| [`efficientvit-sam-xl1`](models/promptable_segmentation.md#efficientvit_sam) | – | 259 ms · portable fp32 | – | – | – | – | – |
| [`sam-hq-vit-b`](models/promptable_segmentation.md#sam_hq) | – | 645 ms · portable fp32 | – | – | – | – | – |
| [`sam-hq-vit-h`](models/promptable_segmentation.md#sam_hq) | – | 2633 ms · portable fp32 | – | – | – | – | – |
| [`sam-hq-vit-l`](models/promptable_segmentation.md#sam_hq) | – | 1420 ms · portable fp32 | – | – | – | – | – |
| [`sam-vit-b`](models/promptable_segmentation.md#sam) | 37.0 ms · kernels fp16 | 631 ms · portable fp32 | 40.7 ms · kernels fp16 | 15.7 ms · kernels fp16 | 35.1 ms · kernels fp16 | 19.2 ms · kernels fp16 | 10.3 ms · kernels fp16 |
| [`sam-vit-h`](models/promptable_segmentation.md#sam) | 143 ms · kernels fp16 | 2518 ms · portable fp32 | 189 ms · kernels fp16 | 59.1 ms · kernels fp16 | 168 ms · kernels fp16 | 61.3 ms · kernels fp16 | 35.1 ms · kernels fp16 |
| [`sam-vit-l`](models/promptable_segmentation.md#sam) | 81.5 ms · kernels fp16 | 1373 ms · portable fp32 | 102 ms · kernels fp16 | 32.5 ms · kernels fp16 | 91.4 ms · kernels fp16 | 37.5 ms · kernels fp16 | 19.6 ms · kernels fp16 |
| [`sam2.1-hiera-base-plus`](models/promptable_segmentation.md#sam2) | 25.1 ms · kernels fp16 | 424 ms · portable fp32 | 31.7 ms · kernels fp16 | 10.3 ms · kernels fp16 | 29.1 ms · kernels fp16 | 15.8 ms · portable fp16 | 7.78 ms · portable fp16 |
| [`sam2.1-hiera-large`](models/promptable_segmentation.md#sam2) | 48.8 ms · kernels fp16 | 613 ms · portable fp32 | 65.8 ms · kernels fp16 | 21.4 ms · kernels fp16 | 59.3 ms · kernels fp16 | 27.1 ms · portable fp16 | 13.8 ms · portable fp16 |
| [`sam2.1-hiera-small`](models/promptable_segmentation.md#sam2) | 15.6 ms · kernels fp16 | 177 ms · portable fp32 | – | – | – | – | 5.43 ms · portable fp16 |
| [`sam2.1-hiera-tiny`](models/promptable_segmentation.md#sam2) | 13.9 ms · kernels fp16 | 151 ms · portable fp32 | 17.6 ms · kernels fp16 | 8.14 ms · kernels fp16 | 15.6 ms · kernels fp16 | 12.0 ms · portable fp16 | 5.11 ms · kernels fp16 |
| [`sam3`](models/promptable_segmentation.md#sam3) | 116 ms · kernels fp16 | 3620 ms · portable fp32 | 161 ms · kernels fp16 | 48.6 ms · kernels fp16 | 136 ms · kernels fp16 | 50.8 ms · kernels fp16 | 29.5 ms · kernels fp16 |

#### Detection, 800x1344

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`detr-r101-coco`](models/detection.md#detr) | 13.7 ms · kernels fp16 | 183 ms · portable fp32 | 12.8 ms · kernels fp16 | 9.91 ms · kernels fp16 | 11.6 ms · kernels fp16 | 10.6 ms · kernels fp16 | 3.52 ms · kernels fp16 |
| [`detr-r50-coco`](models/detection.md#detr) | 9.93 ms · kernels fp16 | 125 ms · portable fp32 | 11.8 ms · kernels fp16 | 8.21 ms · kernels fp16 | 8.64 ms · kernels fp16 | 8.82 ms · kernels fp16 | 3.14 ms · kernels fp16 |
| [`dfine-l-coco`](models/detection.md#dfine) | – | – | 9.47 ms · kernels fp16 | 5.16 ms · kernels fp16 | 9.48 ms · kernels fp16 | 9.36 ms · kernels fp16 | 4.23 ms · kernels fp16 |
| [`dfine-m-coco`](models/detection.md#dfine) | – | – | – | – | – | – | 3.30 ms · kernels fp16 |
| [`dfine-n-coco`](models/detection.md#dfine) | – | – | 3.45 ms · kernels fp16 | 2.41 ms · kernels fp16 | 3.68 ms · kernels fp16 | 4.06 ms · kernels fp16 | 1.89 ms · kernels fp16 |
| [`dfine-s-coco`](models/detection.md#dfine) | – | – | – | – | – | – | 2.37 ms · kernels fp16 |
| [`dfine-x-coco`](models/detection.md#dfine) | – | – | 14.8 ms · kernels fp16 | 6.81 ms · kernels fp16 | 14.4 ms · kernels fp16 | 13.1 ms · kernels fp16 | 6.05 ms · kernels fp16 |
| [`faster-rcnn-r50-fpn-3x-coco`](models/detection.md#rcnn) | 19.5 ms · kernels fp16 | 214 ms · portable fp32 | 19.6 ms · kernels fp16 | 10.6 ms · kernels fp16 | 18.1 ms · kernels fp16 | 13.0 ms · kernels fp16 | 5.86 ms · kernels fp16 |
| [`mask-rcnn-r50-fpn-3x-coco`](models/detection.md#rcnn) | 21.0 ms · kernels fp16 | 222 ms · portable fp32 | 22.1 ms · kernels fp16 | 11.8 ms · kernels fp16 | 20.1 ms · kernels fp16 | 14.7 ms · kernels fp16 | 6.55 ms · kernels fp16 |
| [`rtdetrv2-l-coco`](models/detection.md#rtdetrv2) | 6.61 ms · kernels fp16 | 82.6 ms · portable fp32 | 7.20 ms · kernels fp16 | 3.82 ms · kernels fp16 | 7.63 ms · kernels fp16 | 6.65 ms · kernels fp16 | 3.31 ms · kernels fp16 |
| [`rtdetrv2-m-coco`](models/detection.md#rtdetrv2) | 5.27 ms · kernels fp16 | 55.3 ms · portable fp32 | – | – | – | – | 2.68 ms · kernels fp16 |
| [`rtdetrv2-ms-coco`](models/detection.md#rtdetrv2) | 5.46 ms · kernels fp16 | 67.9 ms · portable fp32 | – | – | – | – | 2.57 ms · kernels fp16 |
| [`rtdetrv2-s-coco`](models/detection.md#rtdetrv2) | 4.19 ms · kernels fp16 | 44.0 ms · portable fp32 | 4.56 ms · kernels fp16 | 2.76 ms · kernels fp16 | 4.88 ms · kernels fp16 | 4.74 ms · kernels fp16 | 2.21 ms · kernels fp16 |
| [`rtdetrv2-x-coco`](models/detection.md#rtdetrv2) | 9.24 ms · kernels fp16 | 114 ms · portable fp32 | 10.1 ms · kernels fp16 | 4.89 ms · kernels fp16 | 10.4 ms · kernels fp16 | 8.11 ms · kernels fp16 | 4.07 ms · kernels fp16 |
| [`rtmdet-l-coco`](models/detection.md#rtmdet) | – | 75.2 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-m-coco`](models/detection.md#rtmdet) | – | 50.9 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-m-person`](models/detection.md#rtmdet) | – | 48.5 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-nano-person`](models/detection.md#rtmdet) | – | 22.0 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-s-coco`](models/detection.md#rtmdet) | – | 30.7 ms · portable fp16 | – | – | – | – | – |
| [`rtmdet-x-coco`](models/detection.md#rtmdet) | – | 115 ms · portable fp16 | – | – | – | – | – |

#### Pose, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`dwpose-l`](models/pose.md#rtmpose) | 87.6 ms · portable fp16 |
| [`dwpose-l-384`](models/pose.md#rtmpose) | 173 ms · portable fp16 |
| [`dwpose-m`](models/pose.md#rtmpose) | 57.7 ms · kernels fp32 |
| [`dwpose-s`](models/pose.md#rtmpose) | 33.2 ms · kernels fp32 |
| [`dwpose-t`](models/pose.md#rtmpose) | 26.4 ms · kernels fp32 |
| [`rtmpose-l`](models/pose.md#rtmpose) | 81.3 ms · portable fp16 |
| [`rtmpose-l-384`](models/pose.md#rtmpose) | 158 ms · portable fp16 |
| [`rtmpose-m`](models/pose.md#rtmpose) | 52.4 ms · kernels fp32 |
| [`rtmpose-m-384`](models/pose.md#rtmpose) | 98.3 ms · kernels fp32 |
| [`rtmpose-s`](models/pose.md#rtmpose) | 29.0 ms · kernels fp32 |
| [`rtmpose-t`](models/pose.md#rtmpose) | 20.5 ms · kernels fp32 |
| [`vitpose-base`](models/pose.md#vitpose) | 172 ms · kernels fp16 |
| [`vitpose-base-simple`](models/pose.md#vitpose) | 160 ms · kernels fp16 |
| [`vitpose-plus-base`](models/pose.md#vitpose) | 174 ms · kernels fp16 |
| [`vitpose-plus-huge`](models/pose.md#vitpose) | 1158 ms · portable fp16 |
| [`vitpose-plus-large`](models/pose.md#vitpose) | 526 ms · kernels fp16 |
| [`vitpose-plus-small`](models/pose.md#vitpose) | 66.5 ms · kernels fp16 |

#### Features, 224x224

| model | RTX 5060 Ti 16 GB | Apple M1 Pro 16 GB | NVIDIA L4 | NVIDIA L40S | NVIDIA A10 | NVIDIA A100-SXM4-40GB | NVIDIA RTX PRO 6000 |
|---|---|---|---|---|---|---|---|
| [`c-radio-v3-g`](models/features.md#c_radio) | – | 145 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v3-h`](models/features.md#c_radio) | – | 85.0 ms · portable fp16 | – | – | – | – | – |
| [`c-radio-v3-l`](models/features.md#c_radio) | – | 47.9 ms · portable fp16 | – | – | – | – | – |
| [`clip-b16-laion2b`](models/features.md#clip) | – | 15.3 ms · portable fp16 | – | – | – | – | – |
| [`clip-h14-378-dfn5b`](models/features.md#clip) | – | 108 ms · portable fp16 | – | – | – | – | – |
| [`clip-h14-dfn5b`](models/features.md#clip) | – | 109 ms · portable fp16 | – | – | – | – | – |
| [`clip-h14-laion2b`](models/features.md#clip) | – | 102 ms · portable fp16 | – | – | – | – | – |
| [`clip-l14-336-openai`](models/features.md#clip) | – | 58.3 ms · portable fp16 | – | – | – | – | – |
| [`clip-l14-datacompxl`](models/features.md#clip) | – | 56.7 ms · portable fp16 | – | – | – | – | – |
| [`clip-l14-dfn2b`](models/features.md#clip) | – | 60.1 ms · portable fp16 | – | – | – | – | – |
| [`clip-l14-laion2b`](models/features.md#clip) | – | 56.6 ms · portable fp16 | – | – | – | – | – |
| [`clip-l14-openai`](models/features.md#clip) | – | 58.3 ms · portable fp16 | – | – | – | – | – |
| [`dinov2-vitb14`](models/features.md#dinov2) | – | 21.7 ms · portable fp16 | – | – | – | – | 0.985 ms · kernels fp16 |
| [`dinov2-vitb14-reg`](models/features.md#dinov2) | – | 42.8 ms · portable fp16 | – | – | – | – | 0.987 ms · kernels fp16 |
| [`dinov2-vitg14`](models/features.md#dinov2) | – | 193 ms · portable fp16 | 22.4 ms · kernels fp16 | 6.66 ms · kernels fp16 | 18.0 ms · kernels fp16 | 9.51 ms · kernels fp16 | 4.95 ms · kernels fp16 |
| [`dinov2-vitg14-reg`](models/features.md#dinov2) | 13.0 ms · kernels fp16 | 235 ms · portable fp16 | – | – | – | – | 4.94 ms · kernels fp16 |
| [`dinov2-vitl14`](models/features.md#dinov2) | – | 59.4 ms · portable fp16 | 6.34 ms · kernels fp16 | 2.39 ms · kernels fp16 | 5.34 ms · portable fp16 | 3.77 ms · kernels fp16 | 2.00 ms · kernels fp16 |
| [`dinov2-vitl14-reg`](models/features.md#dinov2) | – | 86.3 ms · portable fp16 | – | – | – | – | 2.00 ms · kernels fp16 |
| [`dinov2-vits14`](models/features.md#dinov2) | – | 11.2 ms · portable fp16 | 1.05 ms · kernels fp16 | 0.721 ms · kernels fp16 | 0.996 ms · kernels fp16 | 1.40 ms · kernels fp16 | 0.645 ms · kernels fp16 |
| [`dinov2-vits14-reg`](models/features.md#dinov2) | – | 22.1 ms · portable fp16 | – | – | – | – | 0.639 ms · kernels fp16 |
| [`dinov3-convnext-base`](models/features.md#dinov3_convnext) | – | failed | – | – | – | – | 1.78 ms · portable fp16 |
| [`dinov3-convnext-large`](models/features.md#dinov3_convnext) | – | failed | 4.42 ms · kernels fp16 | 2.36 ms · kernels fp16 | 3.73 ms · kernels fp16 | 4.10 ms · kernels fp16 | 1.88 ms · portable fp16 |
| [`dinov3-convnext-small`](models/features.md#dinov3_convnext) | – | 21.7 ms · portable fp16 | – | – | – | – | 1.78 ms · kernels fp16 |
| [`dinov3-convnext-tiny`](models/features.md#dinov3_convnext) | – | failed | 1.46 ms · kernels fp16 | 0.965 ms · kernels fp16 | 1.34 ms · kernels fp16 | 1.79 ms · kernels fp16 | 0.973 ms · kernels fp16 |
| [`dinov3-vitb16`](models/features.md#dinov3) | – | 22.7 ms · portable fp16 | – | – | – | – | 0.865 ms · kernels fp16 |
| [`dinov3-vith16plus`](models/features.md#dinov3) | – | failed | 14.2 ms · kernels fp16 | 4.46 ms · kernels fp16 | 11.4 ms · kernels fp16 | 6.54 ms · kernels fp16 | 3.56 ms · kernels fp16 |
| [`dinov3-vitl16`](models/features.md#dinov3) | – | failed | 5.22 ms · kernels fp16 | 2.27 ms · kernels fp16 | 3.76 ms · kernels fp16 | 3.64 ms · kernels fp16 | 1.99 ms · kernels fp16 |
| [`dinov3-vits16`](models/features.md#dinov3) | – | failed | 1.02 ms · kernels fp16 | 0.727 ms · kernels fp16 | 1.02 ms · kernels fp16 | 1.47 ms · kernels fp16 | 0.654 ms · kernels fp16 |
| [`pe-core-b16-224`](models/features.md#perception_encoder) | – | 16.0 ms · kernels fp16 | – | – | – | – | – |
| [`pe-core-g14-448`](models/features.md#perception_encoder) | – | 259 ms · kernels fp16 | – | – | – | – | – |
| [`pe-core-l14-336`](models/features.md#perception_encoder) | – | 58.3 ms · kernels fp16 | – | – | – | – | – |
| [`pe-core-s16-384`](models/features.md#perception_encoder) | – | 10.4 ms · kernels fp16 | – | – | – | – | – |
| [`pe-spatial-b16-512`](models/features.md#perception_encoder) | – | 16.1 ms · kernels fp16 | – | – | – | – | – |
| [`pe-spatial-g14-448`](models/features.md#perception_encoder) | – | 258 ms · kernels fp16 | – | – | – | – | – |
| [`pe-spatial-l14-448`](models/features.md#perception_encoder) | – | 57.6 ms · kernels fp16 | – | – | – | – | – |
| [`pe-spatial-s16-512`](models/features.md#perception_encoder) | – | 8.31 ms · kernels fp16 | – | – | – | – | – |
| [`pe-spatial-t16-512`](models/features.md#perception_encoder) | – | 5.94 ms · kernels fp16 | – | – | – | – | – |
| [`siglip2-b16-256`](models/features.md#siglip2) | – | 15.8 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-b16-512`](models/features.md#siglip2) | – | 15.8 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-g16-256`](models/features.md#siglip2) | – | 143 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-g16-384`](models/features.md#siglip2) | – | 142 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-l16-256`](models/features.md#siglip2) | – | 45.9 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-l16-512`](models/features.md#siglip2) | – | 45.5 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-so400m14-224`](models/features.md#siglip2) | – | 70.2 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-so400m14-378`](models/features.md#siglip2) | – | 68.6 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-so400m16-256`](models/features.md#siglip2) | – | 57.5 ms · portable fp16 | – | – | – | – | – |
| [`siglip2-so400m16-512`](models/features.md#siglip2) | – | 56.8 ms · portable fp16 | – | – | – | – | – |

#### Flow, 1080x1920

| model | Apple M1 Pro 16 GB |
|---|---|
| [`raft-large-kitti`](models/flow.md#raft) | 10206 ms · portable fp32 |
| [`raft-large-things`](models/flow.md#raft) | 3992 ms · portable fp32 |
| [`raft-small`](models/flow.md#raft) | 4513 ms · portable fp32 |
| [`sea-raft-m`](models/flow.md#sea_raft) | 1820 ms · portable fp32 |
| [`sea-raft-m-kitti`](models/flow.md#sea_raft) | 5026 ms · portable fp32 |
| [`sea-raft-m-spring`](models/flow.md#sea_raft) | 390 ms · portable fp32 |
| [`sea-raft-m-things`](models/flow.md#sea_raft) | 4162 ms · portable fp32 |
| [`sea-raft-s`](models/flow.md#sea_raft) | 2115 ms · portable fp32 |
| [`sea-raft-s-kitti`](models/flow.md#sea_raft) | 2000 ms · portable fp32 |
| [`sea-raft-s-spring`](models/flow.md#sea_raft) | 268 ms · portable fp32 |
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
| c-foundationstereo-s | 2499 | 1121 | 1074 | 431 | 2.6x | 2.5x | 2781 / 3204 | 3085 / 3450 | 1785 / 2036 |
| cas-gwcnet-c-sceneflow | 195 | 115 | 96.7 | 65.8 | 1.8x | 1.5x | 917 / 990 | 668 / 834 | 581 / 670 |
| cas-gwcnet-gc-sceneflow | 250 | 161 | 138 | 103 | 1.6x | 1.3x | 2634 / 2772 | 772 / 954 | 751 / 834 |
| cfnet-sceneflow | 194 | 155 | 112 | 89.8 | 1.7x | 1.3x | 2211 / 2584 | 1622 / 2016 | 1692 / 2116 |
| coex-sceneflow | 26.0 | 16.6 | 14.8 | 6.60 | 2.5x | 2.2x | 671 / 698 | 387 / 434 | 163 / 486 |
| crestereo-eth3d | 636 | 480 | 209 | 83.1 | 5.8x | 2.5x | 516 / 642 | 315 / 370 | 245 / 292 |
| defom-stereo-vitl-sceneflow | 947 | 509 | 404 | 184 | 2.8x | 2.2x | 2458 / 2610 | 2394 / 2538 | 1604 / 1846 |
| dlnr-middlebury | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| dlnr-sceneflow | 292 | 168 | 117 | 71.5 | 2.4x | 1.6x | 1173 / 1334 | 915 / 1082 | 495 / 606 |
| fast-acvnet-generalization | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-kitti12 | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-kitti15 | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-acvnet-plus-generalization | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-kitti12 | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-kitti15 | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-plus-sceneflow | 35.4 | 26.4 | 19.5 | 12.0 | 2.2x | 1.6x | 468 / 506 | 218 / 256 | 246 / 358 |
| fast-acvnet-sceneflow | 28.1 | 24.1 | 17.4 | 10.5 | 2.3x | 1.7x | 461 / 486 | 219 / 256 | 225 / 338 |
| fast-foundationstereo | 203 | 142 | 97.8 | 51.0 | 2.8x | 1.9x | 1052 / 1108 | 733 / 774 | 438 / 554 |
| foundationstereo-vit-l | 2801 | 1272 | 905 | 491 | 2.6x | 1.8x | 3614 / 3754 | 2154 / 2294 | 1588 / 1658 |
| foundationstereo-vit-s | 2139 | 1052 | 737 | 358 | 2.9x | 2.1x | 2388 / 2484 | 1542 / 1744 | 977 / 1090 |
| great-igev-eth3d | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-kitti | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| great-igev-middlebury | 1055 | 705 | 492 | 175 | 4.0x | 2.8x | 10728 / 10974 | 1165 / 1298 | 1093 / 1184 |
| great-igev-sceneflow | 659 | 425 | 301 | 126 | 3.4x | 2.4x | 2831 / 3074 | 1039 / 1258 | 572 / 704 |
| gwcnet-g-sceneflow | 175 | 104 | 104 | 72.7 | 1.4x | 1.4x | 1622 / 1734 | 686 / 882 | 598 / 782 |
| gwcnet-gc-sceneflow | 183 | 107 | 107 | 78.3 | 1.4x | 1.4x | 1867 / 1976 | 687 / 982 | 567 / 642 |
| hitnet-eth3d | 63.1 | 46.0 | 22.4 | 7.41 | 6.2x | 3.0x | 746 / – | 341 / 510 | 208 / 728 |
| hitnet-middlebury-d400 | 177 | 130 | 52.6 | 17.0 | 7.7x | 3.1x | 2124 / – | 698 / 1074 | 425 / 1428 |
| igev-plusplus-eth3d | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-kitti12 | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-kitti15 | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-middlebury | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-plusplus-sceneflow | 575 | 383 | 297 | 126 | 3.0x | 2.4x | 976 / 1054 | 540 / 720 | 453 / 720 |
| igev-stereo-eth3d | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti12 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-kitti15 | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-middlebury | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| igev-stereo-sceneflow | 449 | 279 | 213 | 96.4 | 2.9x | 2.2x | 626 / 702 | 426 / 532 | 393 / 494 |
| las2-h | 51.1 | 36.4 | 37.5 | 14.9 | 2.4x | 2.5x | 352 / 386 | 387 / 430 | 311 / 528 |
| las2-l | 46.9 | 30.1 | 30.9 | 8.96 | 3.4x | 3.4x | 422 / 488 | 424 / 458 | 169 / 380 |
| las2-m | 23.9 | 20.0 | 23.2 | 5.28 | 3.8x | 4.4x | 360 / 404 | 360 / 386 | 136 / 332 |
| las2-s | 19.3 | 17.3 | 20.5 | 3.90 | 4.4x | 5.3x | 345 / 382 | 343 / 368 | 128 / 338 |
| mocha-v1-sceneflow | 648 | 451 | 321 | 165 | 2.7x | 1.9x | 1430 / 1608 | 668 / 750 | 657 / 750 |
| mocha-v2-sceneflow | 615 | 398 | 315 | 164 | 2.4x | 1.9x | 1411 / 1588 | 668 / 750 | 656 / 750 |
| monster-mix-all | 993 | 995 | 419 | 187 | 5.3x | 2.2x | 2733 / 2958 | 1262 / 1594 | 1282 / 1634 |
| monster-plusplus-mix-all | 1012 | 1009 | 432 | 195 | 5.2x | 2.2x | 2810 / 3018 | 1339 / 1618 | 1308 / 1660 |
| monster-sceneflow | 993 | 995 | 419 | 187 | 5.3x | 2.2x | 2733 / 2958 | 1262 / 1594 | 1282 / 1634 |
| nmrf-sceneflow | 122 | 78.1 | 58.1 | 51.5 | 1.5x | 1.1x | 587 / 634 | 548 / 586 | 364 / 406 |
| psmnet-sceneflow | 248 | 204 | 107 | 73.5 | 2.8x | 1.5x | 1950 / 1990 | 700 / 822 | 681 / 862 |
| raft-stereo-eth3d | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-middlebury | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| raft-stereo-sceneflow | 543 | 291 | 215 | 87.7 | 3.3x | 2.4x | 816 / 872 | 476 / 552 | 282 / 362 |
| rt-monster-plusplus-mix-all | 91.8 | 91.3 | 37.4 | 31.4 | 2.9x | 1.2x | 1051 / 1106 | 892 / 926 | 899 / 1116 |
| s2m2-l | 597 | 283 | 288 | 120 | 2.4x | 2.4x | 1747 / 1998 | 1510 / 1718 | 831 / 988 |
| s2m2-m | 356 | 184 | 184 | 78.2 | 2.4x | 2.4x | 1080 / 1284 | 952 / 1126 | 491 / 542 |
| s2m2-s | 163 | 99.1 | 91.5 | 38.3 | 2.6x | 2.4x | 676 / 748 | 633 / 750 | 287 / 400 |
| s2m2-xl | 1063 | 529 | 536 | 212 | 2.5x | 2.5x | 3315 / 3484 | 2722 / 2976 | 1502 / 1614 |
| selective-igev-eth3d-finetune | 637 | 422 | 279 | 114 | 3.7x | 2.4x | 698 / 740 | 427 / 548 | 389 / 490 |
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
| c-foundationstereo-s | OOM | 3874 | 3646 | 1362 | 2.8x | 2.7x | 8338 / 12870 | 6312 / 6590 | 3107 / 4376 |
| cas-gwcnet-c-sceneflow | 656 | 389 | 327 | 225 | 1.7x | 1.5x | 2791 / 2970 | 2006 / 2474 | 1725 / 1930 |
| cas-gwcnet-gc-sceneflow | 862 | 567 | 472 | 358 | 1.6x | 1.3x | 8317 / 8730 | 2338 / 2794 | 2272 / 2514 |
| cfnet-sceneflow | 659 | 545 | 484 | 401 | 1.4x | 1.2x | 6765 / 7880 | 3441 / 4376 | 3662 / 4718 |
| coex-sceneflow | 94.9 | 60.0 | 49.9 | 23.5 | 2.6x | 2.1x | 2098 / 2160 | 1136 / 1254 | 495 / 886 |
| crestereo-eth3d | 1980 | 1597 | 732 | 229 | 7.0x | 3.2x | 1541 / 1752 | 884 / 1030 | 644 / 732 |
| defom-stereo-vitl-sceneflow | 4175 | 2816 | 1531 | 659 | 4.3x | 2.3x | 8639 / 8766 | 4362 / 4838 | 3154 / 3566 |
| dlnr-middlebury | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| dlnr-sceneflow | 978 | 598 | 398 | 214 | 2.8x | 1.9x | 3004 / 3552 | 2635 / 3142 | 1236 / 1606 |
| fast-acvnet-generalization | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-kitti12 | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-kitti15 | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-acvnet-plus-generalization | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-kitti12 | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-kitti15 | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-plus-sceneflow | 138 | 80.9 | 70.6 | 46.4 | 1.7x | 1.5x | 1441 / 1522 | 662 / 756 | 790 / 1120 |
| fast-acvnet-sceneflow | 101 | 60.6 | 54.7 | 38.2 | 1.6x | 1.4x | 1421 / 1506 | 664 / 736 | 739 / 1042 |
| fast-foundationstereo | 741 | 527 | 330 | 135 | 3.9x | 2.4x | 2939 / 3306 | 2109 / 2256 | 1196 / 1576 |
| foundationstereo-vit-l | 9228 | 4237 | 2990 | 1513 | 2.8x | 2.0x | 8286 / 8750 | 3141 / 3436 | 2818 / 3300 |
| foundationstereo-vit-s | 7129 | 3609 | 2514 | 1120 | 3.2x | 2.2x | 6987 / 7280 | 2493 / 2606 | 2169 / 2470 |
| great-igev-eth3d | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-kitti | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| great-igev-middlebury | OOM | OOM | 1785 | 579 | – | 3.1x | – | 1875 / 2078 | 1553 / 1744 |
| great-igev-sceneflow | 2252 | 1458 | 1085 | 393 | 3.7x | 2.8x | 8861 / 9206 | 1402 / 1678 | 1192 / 1564 |
| gwcnet-g-sceneflow | 604 | 352 | 344 | 238 | 1.5x | 1.4x | 5105 / 5414 | 2081 / 2402 | 1799 / 2102 |
| gwcnet-gc-sceneflow | 632 | 365 | 353 | 257 | 1.4x | 1.4x | 5887 / 6214 | 2082 / 2402 | 1768 / 1962 |
| hitnet-eth3d | 234 | 182 | 67.8 | 23.2 | 7.9x | 2.9x | 2433 / – | 860 / 1090 | 647 / 1248 |
| hitnet-middlebury-d400 | 637 | 483 | 176 | 52.7 | 9.2x | 3.3x | 6831 / – | 1744 / 2414 | 1263 / 2368 |
| igev-plusplus-eth3d | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-kitti12 | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-kitti15 | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-middlebury | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-plusplus-sceneflow | 2012 | 1278 | 1081 | 393 | 3.3x | 2.7x | 2891 / 3108 | 1594 / 1840 | 1389 / 2302 |
| igev-stereo-eth3d | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti12 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-kitti15 | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-middlebury | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| igev-stereo-sceneflow | 1518 | 941 | 763 | 288 | 3.3x | 2.6x | 1789 / 1936 | 1237 / 1492 | 1113 / 1374 |
| las2-h | 205 | 135 | 123 | 53.3 | 2.5x | 2.3x | 1022 / 1084 | 1050 / 1110 | 1021 / 1388 |
| las2-l | 154 | 90.8 | 80.3 | 29.9 | 3.0x | 2.7x | 1010 / 1082 | 1036 / 1098 | 397 / 600 |
| las2-m | 79.1 | 52.4 | 54.1 | 18.3 | 2.9x | 3.0x | 948 / 1000 | 972 / 1026 | 329 / 572 |
| las2-s | 62.0 | 43.5 | 46.9 | 14.3 | 3.0x | 3.3x | 933 / 978 | 955 / 988 | 320 / 558 |
| mocha-v1-sceneflow | 2148 | 1456 | 1057 | 455 | 3.2x | 2.3x | 3637 / 3982 | 1632 / 1770 | 1588 / 1730 |
| mocha-v2-sceneflow | 2054 | 1295 | 1033 | 448 | 2.9x | 2.3x | 3576 / 3982 | 1632 / 1770 | 1587 / 1790 |
| monster-mix-all | 4275 | 4277 | 1585 | 668 | 6.4x | 2.4x | 6382 / 6430 | 2336 / 2494 | 2356 / 2714 |
| monster-plusplus-mix-all | 4339 | 4334 | 1633 | 698 | 6.2x | 2.3x | 6382 / 6430 | 2583 / 2918 | 2438 / 2760 |
| monster-sceneflow | 4275 | 4277 | 1585 | 668 | 6.4x | 2.4x | 6382 / 6430 | 2336 / 2494 | 2356 / 2714 |
| nmrf-sceneflow | 445 | 313 | 216 | 196 | 1.6x | 1.1x | 1656 / 2054 | 1510 / 1552 | 1022 / 1112 |
| psmnet-sceneflow | 760 | 608 | 340 | 232 | 2.6x | 1.5x | 6015 / 6090 | 2078 / 2442 | 1786 / 2122 |
| raft-stereo-eth3d | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-middlebury | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| raft-stereo-sceneflow | 1775 | 1037 | 746 | 273 | 3.8x | 2.7x | 2516 / 2628 | 1388 / 1492 | 765 / 802 |
| rt-monster-plusplus-mix-all | 252 | 251 | 113 | 94.1 | 2.7x | 1.2x | 2922 / 3044 | 2592 / 2646 | 2704 / 3976 |
| s2m2-l | 1934 | 1005 | 1015 | 376 | 2.7x | 2.7x | 3248 / 4090 | 3245 / 3838 | 1249 / 1568 |
| s2m2-m | 1242 | 666 | 670 | 248 | 2.7x | 2.7x | 2371 / 2920 | 2300 / 2786 | 876 / 1162 |
| s2m2-s | 582 | 350 | 348 | 113 | 3.1x | 3.1x | 1759 / 1986 | 1733 / 1930 | 686 / 820 |
| s2m2-xl | 3510 | 1796 | 1815 | 682 | 2.6x | 2.7x | 5606 / 6080 | 5228 / 6016 | 2092 / 2274 |
| selective-igev-eth3d-finetune | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-eth3d-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti12 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-kitti15 | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-finetune | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-middlebury-train | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| selective-igev-sceneflow | 2201 | 1455 | 998 | 349 | 4.2x | 2.9x | 2013 / 2152 | 1238 / 1488 | 1109 / 1390 |
| waft-stereo-b | 1898 | 1140 | 1137 | 393 | 2.9x | 2.9x | 5743 / 5876 | 5775 / 5852 | 1907 / 2316 |
| waft-stereo-s | 1252 | 689 | 689 | 284 | 2.4x | 2.4x | 3314 / 3398 | 3293 / 3456 | 1113 / 1286 |

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
| depth-pro | 1776 | 569 | 518 | 271 | 2.1x | 1.9x | 3828 / 4294 | 3306 / 3498 | 3667 / 3778 |
| distill-any-depth-b | 90.5 | 59.8 | – | – | – | – | 928 / 1026 | – | – |
| distill-any-depth-l | 244 | 139 | – | – | – | – | 2560 / 2654 | – | – |
| distill-any-depth-s | 45.3 | 35.2 | – | – | – | – | 373 / 418 | – | – |
| marigold-depth-v1-1 | 963 | 439 | 408 | 383 | 1.1x | 1.1x | 3414 / 3580 | 2817 / 3760 | 2583 / 3762 |
| metric3d-vit-large | 629 | 438 | 200 | 158 | 2.8x | 1.3x | 3275 / 3350 | 1484 / 1696 | 1516 / 1688 |
| midas-dpt-beit-large-512 | 386 | 333 | 78.8 | 63.9 | 5.2x | 1.2x | 2759 / 2898 | 992 / 1418 | 1003 / 1436 |
| moge-2-vitb-normal | 237 | 102 | – | – | – | – | 1469 / 1596 | – | – |
| moge-2-vitl-normal | 501 | 171 | 144 | 73.4 | 2.3x | 2.0x | 2339 / 2438 | 1661 / 1836 | 1498 / 1610 |
| moge-2-vits-normal | 162 | 82.6 | 64.2 | 29.7 | 2.8x | 2.2x | 1193 / 1268 | 1066 / 1336 | 884 / 1008 |
| moge-3-vitl | 843 | 486 | 514 | 215 | 2.3x | 2.4x | 2505 / 2630 | 1686 / 1924 | 1673 / 1884 |
| unidepth-v2-vitl14 | 132 | 132 | 99.8 | 66.4 | 2.0x | 1.5x | 3230 / 3508 | 1208 / 1472 | 1211 / 1472 |
| unik3d-vitl | 134 | 134 | 108 | 62.9 | 2.1x | 1.7x | 3307 / 3570 | 1219 / 1492 | 1166 / 1266 |
| zoedepth-nk | 408 | 271 | 124 | 88.9 | 3.1x | 1.4x | 2040 / 2146 | 2037 / 2124 | 1771 / 1960 |

### Depth, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| da3-giant-1.1 | 518x770 | 406 | 224 | 81.5 | 44.2 | 5.1x | 1.8x | 8590 / 8718 | 2621 / 4608 | 2527 / 2710 |
| da3-large-1.1 | 518x770 | 205 | 163 | 36.9 | 16.7 | 9.7x | 2.2x | 3374 / 3472 | 975 / 1350 | 883 / 1090 |
| da3metric-large | 518x770 | 125 | 82.7 | 23.7 | 14.3 | 5.8x | 1.7x | 2996 / 3050 | 798 / 1328 | 794 / 868 |
| da3mono-large | 518x770 | 125 | 82.7 | 23.5 | 14.2 | 5.8x | 1.7x | 2997 / 3052 | 801 / 1328 | 797 / 868 |
| da3nested-giant-large-1.1 | 518x770 | 529 | 307 | 105 | 58.3 | 5.3x | 1.8x | 10332 / 10402 | 3259 / 5914 | 3171 / 3378 |
| depth-anything-v2-b | 518x770 | 88.8 | 53.8 | 18.8 | 10.6 | 5.1x | 1.8x | 928 / 1026 | 433 / 516 | 370 / 436 |
| depth-anything-v2-l | 518x770 | 240 | 136 | 55.0 | 31.6 | 4.3x | 1.7x | 2560 / 2654 | 1009 / 1352 | 904 / 1034 |
| depth-anything-v2-metric-hypersim-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| depth-anything-v2-metric-hypersim-l | 994x1482 | – | – | 54.5 | 31.7 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| depth-anything-v2-metric-hypersim-s | 994x1482 | – | – | 7.66 | 4.67 | – | 1.6x | – | 244 / 288 | 152 / 208 |
| depth-anything-v2-metric-vkitti-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| depth-anything-v2-metric-vkitti-l | 994x1482 | – | – | 54.5 | 31.9 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| depth-anything-v2-metric-vkitti-s | 994x1482 | – | – | 7.66 | 4.67 | – | 1.6x | – | 244 / 288 | 152 / 208 |
| depth-anything-v2-s | 518x770 | 45.5 | 38.1 | 7.73 | 4.45 | 8.6x | 1.7x | 373 / 418 | 232 / 326 | 173 / 248 |
| depth-pro | 518x770 | 1769 | 566 | 519 | 271 | 2.1x | 1.9x | 3822 / 4354 | 3289 / 3498 | 3650 / 3778 |
| distill-any-depth-b | 994x1482 | – | – | 18.6 | 10.8 | – | 1.7x | – | 445 / 514 | 349 / 396 |
| distill-any-depth-l | 994x1482 | – | – | 54.4 | 31.8 | – | 1.7x | – | 1022 / 1092 | 917 / 1034 |
| distill-any-depth-s | 994x1482 | – | – | 7.65 | 4.63 | – | 1.7x | – | 244 / 288 | 152 / 208 |
| marigold-depth-v1-1 | 518x770 | 962 | 429 | 408 | 384 | 1.1x | 1.1x | 3414 / 3580 | 2800 / 3760 | 2566 / 3762 |
| metric3d-vit-large | 518x770 | 630 | 439 | 200 | 158 | 2.8x | 1.3x | 3275 / 3370 | 1467 / 1696 | 1499 / 1688 |
| midas-dpt-beit-large-512 | 518x770 | 399 | 320 | 75.6 | 63.9 | 5.0x | 1.2x | 2759 / 2858 | 975 / 1418 | 986 / 1436 |
| moge-2-vitb-normal | 994x1482 | – | – | 83.9 | 40.7 | – | 2.1x | – | 1163 / 1312 | 1093 / 1208 |
| moge-2-vitl-normal | 518x770 | 500 | 169 | 145 | 73.9 | 2.3x | 2.0x | 2326 / 2418 | 1645 / 1836 | 1481 / 1610 |
| moge-2-vits-normal | 518x770 | 160 | 81.2 | 61.3 | 32.9 | 2.5x | 1.9x | 1180 / 1288 | 1049 / 1336 | 867 / 1008 |
| moge-3-vitl | 518x770 | 734 | 324 | 354 | 150 | 2.2x | 2.4x | 2493 / 2594 | 1669 / 1804 | 1506 / 1662 |
| unidepth-v2-vitl14 | 518x770 | 78.5 | 78.6 | 59.6 | 40.7 | 1.9x | 1.5x | 2820 / 3030 | 1053 / 1472 | 1061 / 1472 |
| unik3d-vitl | 518x770 | 123 | 123 | 99.2 | 58.2 | 2.1x | 1.7x | 3220 / 3530 | 1172 / 1492 | 1130 / 1266 |
| zoedepth-nk | 518x770 | 295 | 250 | 114 | 80.0 | 3.1x | 1.4x | 1965 / 2124 | 1950 / 2124 | 1658 / 1960 |

### Classification, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| alexnet.tv_in1k | 224x224 | 1.17 | 1.82 | 0.970 | 0.604 | 1.9x | 1.6x | 383 / 404 | 151 / 224 | 184 / 228 |
| coatnet_0_rw_224.sw_in1k | 224x224 | 7.64 | 9.59 | 6.49 | – | – | – | 194 / 210 | 102 / 124 | – |
| coatnet_1_rw_224.sw_in1k | 224x224 | 12.3 | 15.7 | 10.7 | – | – | – | 277 / 294 | 132 / 158 | – |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 224x224 | 14.6 | 15.7 | 11.9 | – | – | – | 466 / 470 | 207 / 240 | – |
| coatnet_nano_rw_224.sw_in1k | 224x224 | 7.88 | 10.6 | 7.23 | – | – | – | 124 / 146 | 74 / 86 | – |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 224x224 | 13.9 | 18.3 | 12.3 | – | – | – | 278 / 296 | 132 / 158 | – |
| coatnet_rmlp_1_rw_224.sw_in1k | 224x224 | 14.0 | 18.4 | 12.4 | – | – | – | 278 / 296 | 131 / 156 | – |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 224x224 | 15.3 | 18.6 | 12.8 | – | – | – | 466 / 470 | 207 / 240 | – |
| coatnet_rmlp_2_rw_224.sw_in1k | 224x224 | 15.3 | 18.6 | 15.1 | – | – | – | 466 / 472 | 207 / 240 | – |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 384x384 | 27.9 | 17.9 | 13.9 | – | – | – | 516 / 528 | 296 / 330 | – |
| coatnet_rmlp_nano_rw_224.sw_in1k | 224x224 | 8.95 | 12.1 | 8.09 | – | – | – | 124 / 146 | 78 / 106 | – |
| convnext_base.fb_in1k | 224x224 | 7.77 | 11.2 | 5.57 | 2.01 | 3.9x | 2.8x | 541 / 566 | 291 / 296 | 208 / 236 |
| convnext_base.fb_in22k_ft_in1k | 224x224 | 7.78 | 11.3 | 5.66 | 2.00 | 3.9x | 2.8x | 541 / 566 | 291 / 296 | 208 / 236 |
| convnext_base.fb_in22k_ft_in1k_384 | 224x224 | 11.6 | 11.2 | 5.48 | 3.00 | 3.7x | 1.8x | 549 / 562 | 309 / 334 | 213 / 232 |
| convnext_large.fb_in1k | 224x224 | 10.7 | 11.5 | 5.59 | 3.29 | 3.2x | 1.7x | 1166 / 1180 | 603 / 632 | 419 / 434 |
| convnext_large.fb_in22k_ft_in1k | 224x224 | 10.7 | 11.5 | 5.57 | 3.28 | 3.2x | 1.7x | 1166 / 1180 | 603 / 632 | 419 / 434 |
| convnext_large.fb_in22k_ft_in1k_384 | 224x224 | 22.9 | 11.5 | 5.99 | 5.20 | 2.2x | 1.2x | 1177 / 1260 | 629 / 654 | 426 / 454 |
| convnext_small.fb_in1k | 224x224 | 7.44 | 11.3 | 5.69 | 1.84 | 4.0x | 3.1x | 321 / 338 | 181 / 190 | 134 / 152 |
| convnext_small.fb_in22k_ft_in1k | 224x224 | 7.51 | 11.3 | 5.73 | 1.84 | 4.1x | 3.1x | 321 / 338 | 181 / 190 | 134 / 152 |
| convnext_small.fb_in22k_ft_in1k_384 | 224x224 | 8.04 | 11.2 | 5.51 | 2.52 | 3.2x | 2.2x | 328 / 338 | 194 / 212 | 138 / 152 |
| convnext_tiny.fb_in1k | 224x224 | 4.05 | 6.14 | 3.20 | 1.15 | 3.5x | 2.8x | 198 / 216 | 119 / 130 | 92 / 110 |
| convnext_tiny.fb_in22k_ft_in1k | 224x224 | 4.11 | 6.13 | 3.21 | 1.15 | 3.6x | 2.8x | 198 / 216 | 119 / 130 | 92 / 110 |
| convnext_tiny.fb_in22k_ft_in1k_384 | 224x224 | 4.59 | 6.05 | 3.15 | 1.56 | 2.9x | 2.0x | 204 / 216 | 133 / 152 | 96 / 110 |
| convnextv2_base.fcmae_ft_in22k_in1k | 224x224 | 10.7 | 14.6 | 10.2 | 2.94 | 3.7x | 3.5x | 542 / 566 | 299 / 376 | 318 / 380 |
| davit_base.msft_in1k | 224x224 | 13.4 | 20.0 | 11.7 | – | – | – | 537 / 548 | 212 / 224 | – |
| davit_small.msft_in1k | 224x224 | 13.0 | 18.8 | 12.8 | – | – | – | 319 / 324 | 136 / 150 | – |
| davit_tiny.msft_in1k | 224x224 | 6.93 | 9.88 | 6.33 | – | – | – | 196 / 204 | 96 / 116 | – |
| deit_base_patch16_224.fb_in1k | 224x224 | 6.39 | 5.46 | 2.98 | 1.64 | 3.3x | 1.8x | 531 / 548 | 203 / 386 | 300 / 370 |
| densenet121.tv_in1k | 224x224 | 10.7 | 13.6 | 9.41 | 1.28 | 8.4x | 7.4x | 80 / 102 | 53 / 90 | 83 / 132 |
| dinov2_vitl14_lc | 224x224 | 22.4 | 10.8 | 5.43 | 4.93 | 2.2x | 1.1x | 1782 / 1806 | 623 / 1228 | 890 / 950 |
| efficientformer_l1.snap_dist_in1k | 224x224 | 4.87 | 6.40 | 3.72 | – | – | – | 104 / 124 | 93 / 104 | – |
| efficientformer_l3.snap_dist_in1k | 224x224 | 8.39 | 11.0 | 6.37 | – | – | – | 214 / 236 | 161 / 216 | – |
| efficientformer_l7.snap_dist_in1k | 224x224 | 12.3 | 15.9 | 9.65 | – | – | – | 505 / 522 | 300 / 324 | – |
| efficientformerv2_l.snap_dist_in1k | 224x224 | 19.9 | 29.0 | 17.1 | – | – | – | 186 / 198 | 93 / 144 | – |
| efficientformerv2_s0.snap_dist_in1k | 224x224 | 8.16 | 11.8 | 7.62 | – | – | – | 55 / 64 | 44 / 64 | – |
| efficientformerv2_s1.snap_dist_in1k | 224x224 | 10.1 | 14.9 | 8.91 | – | – | – | 70 / 80 | 50 / 88 | – |
| efficientformerv2_s2.snap_dist_in1k | 224x224 | 15.2 | 21.8 | 13.9 | – | – | – | 108 / 116 | 64 / 104 | – |
| efficientnet_b0.tv_in1k | 224x224 | 5.64 | 7.91 | 4.98 | 0.769 | 7.3x | 6.5x | 64 / 84 | 48 / 96 | 78 / 118 |
| efficientnet_b4.tv_in1k | 224x224 | 11.1 | 15.7 | 10.0 | 2.00 | 5.5x | 5.0x | 147 / 174 | 89 / 114 | 111 / 180 |
| efficientnet_v2_s.tv_in1k | 224x224 | 12.1 | 16.6 | 10.2 | 1.89 | 6.4x | 5.4x | 159 / 176 | 81 / 106 | 114 / 188 |
| efficientvit_b2.r288_in1k | 224x224 | 10.7 | 14.7 | 11.9 | 1.10 | 9.7x | 10.8x | 173 / 194 | 85 / 124 | 120 / 212 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 224x224 | 83.7 | 32.1 | 18.0 | 14.1 | 2.3x | 1.3x | 1819 / 1864 | 648 / 1206 | 1236 / 1386 |
| fastvit_sa24.apple_dist_in1k | 256x256 | 7.10 | 10.9 | 6.05 | 1.40 | 5.1x | 4.3x | 157 / 176 | 81 / 110 | 80 / 150 |
| fastvit_t12.apple_dist_in1k | 256x256 | 3.61 | 5.66 | 3.27 | 0.781 | 4.6x | 4.2x | 77 / 84 | 54 / 76 | 52 / 76 |
| fastvit_t8.apple_dist_in1k | 256x256 | 3.07 | 4.91 | 3.30 | 0.703 | 4.4x | 4.7x | 57 / 76 | 45 / 72 | 45 / 76 |
| focalnet_base_lrf.ms_in1k | 224x224 | 18.0 | 22.1 | 15.5 | – | – | – | 546 / 560 | 212 / 234 | – |
| focalnet_base_srf.ms_in1k | 224x224 | 17.1 | 20.0 | 14.2 | – | – | – | 542 / 558 | 211 / 234 | – |
| focalnet_small_lrf.ms_in1k | 224x224 | 16.8 | 22.0 | 15.8 | – | – | – | 323 / 338 | 136 / 152 | – |
| focalnet_small_srf.ms_in1k | 224x224 | 15.9 | 20.1 | 14.0 | – | – | – | 320 / 336 | 136 / 152 | – |
| focalnet_tiny_lrf.ms_in1k | 224x224 | 8.92 | 11.4 | 8.07 | – | – | – | 199 / 208 | 95 / 118 | – |
| focalnet_tiny_srf.ms_in1k | 224x224 | 8.39 | 10.4 | 7.36 | – | – | – | 197 / 206 | 95 / 118 | – |
| hgnetv2_b0.ssld_stage2_ft_in1k | 224x224 | 4.45 | 5.89 | 3.46 | – | – | – | 68 / 72 | 48 / 70 | – |
| hgnetv2_b1.ssld_stage2_ft_in1k | 224x224 | 4.57 | 6.17 | 3.47 | – | – | – | 70 / 92 | 50 / 68 | – |
| hgnetv2_b2.ssld_stage2_ft_in1k | 224x224 | 6.25 | 7.98 | 4.55 | – | – | – | 98 / 118 | 59 / 94 | – |
| hgnetv2_b3.ssld_stage2_ft_in1k | 224x224 | 7.16 | 9.30 | 5.22 | – | – | – | 127 / 140 | 76 / 114 | – |
| hgnetv2_b4.ssld_stage2_ft_in1k | 224x224 | 7.06 | 8.41 | 4.53 | – | – | – | 147 / 166 | 82 / 116 | – |
| hgnetv2_b5.ssld_stage2_ft_in1k | 224x224 | 11.4 | 13.8 | 7.04 | – | – | – | 260 / 280 | 114 / 150 | – |
| hgnetv2_b6.ssld_stage2_ft_in1k | 224x224 | 16.5 | 17.9 | 9.44 | – | – | – | 465 / 490 | 198 / 224 | – |
| inception_v3.tv_in1k | 224x224 | 7.88 | 10.3 | 5.21 | 1.30 | 6.1x | 4.0x | 185 / 206 | 120 / 166 | 150 / 228 |
| mobilenet_v2.tv2_in1k | 224x224 | 3.60 | 5.14 | 2.95 | 0.439 | 8.2x | 6.7x | 54 / 72 | 43 / 70 | 75 / 92 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 4.33 | 6.20 | 3.80 | 0.545 | 7.9x | 7.0x | 65 / 80 | 46 / 76 | 78 / 114 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 2.19 | 2.96 | 1.77 | 0.370 | 5.9x | 4.8x | 58 / 74 | 44 / 72 | 76 / 114 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 5.78 | 7.77 | 3.91 | 0.641 | 9.0x | 6.1x | 90 / 106 | 54 / 86 | 87 / 124 |
| regnety_160.tv2_in1k | 224x224 | 12.1 | 12.2 | 8.34 | 1.86 | 6.5x | 4.5x | 512 / 524 | 208 / 250 | 229 / 316 |
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
| resnet152.tv2_in1k | 224x224 | 12.8 | 16.3 | 8.36 | 1.83 | 7.0x | 4.6x | 378 / 390 | 155 / 186 | 182 / 266 |
| resnet18.tv_in1k | 224x224 | 1.91 | 2.44 | 1.48 | 0.459 | 4.2x | 3.2x | 105 / 130 | 61 / 90 | 90 / 150 |
| resnet50.tv2_in1k | 224x224 | 4.74 | 5.84 | 3.15 | 0.826 | 5.7x | 3.8x | 180 / 192 | 89 / 142 | 116 / 162 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 4.79 | 6.53 | 3.71 | 0.787 | 6.1x | 4.7x | 47 / 48 | 41 / 72 | 87 / 116 |
| shvit_s1.in1k | 224x224 | 8.41 | 11.1 | 6.75 | – | – | – | 70 / 76 | 47 / 58 | – |
| shvit_s2.in1k | 224x224 | 8.37 | 11.1 | 6.76 | – | – | – | 99 / 120 | 56 / 88 | – |
| shvit_s3.in1k | 224x224 | 9.40 | 12.5 | 8.11 | – | – | – | 115 / 134 | 62 / 112 | – |
| shvit_s4.in1k | 256x256 | 11.5 | 14.6 | 9.31 | – | – | – | 129 / 146 | 67 / 118 | – |
| squeezenet1_1.tv_in1k | 224x224 | 1.72 | 2.63 | 1.82 | 0.359 | 4.8x | 5.1x | 10 / 30 | 41 / 72 | 8 / 12 |
| starnet_s1.in1k | 224x224 | 4.57 | 7.29 | 3.88 | 0.683 | 6.7x | 5.7x | 51 / 60 | 41 / 52 | 41 / 56 |
| starnet_s2.in1k | 224x224 | 3.64 | 5.63 | 3.02 | 0.578 | 6.3x | 5.2x | 55 / 64 | 43 / 54 | 43 / 58 |
| starnet_s3.in1k | 224x224 | 4.96 | 7.67 | 4.19 | 0.715 | 6.9x | 5.9x | 67 / 76 | 47 / 60 | 47 / 64 |
| starnet_s4.in1k | 224x224 | 6.84 | 11.3 | 5.63 | 0.912 | 7.5x | 6.2x | 77 / 86 | 50 / 66 | 51 / 72 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 224x224 | 15.0 | 19.1 | 10.8 | 2.15 | 7.0x | 5.0x | 538 / 550 | 213 / 394 | 246 / 300 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 224x224 | 5.95 | 8.27 | 5.19 | 0.930 | 6.4x | 5.6x | 106 / 122 | 64 / 90 | 60 / 84 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 224x224 | 6.05 | 8.37 | 5.26 | 1.18 | 5.1x | 4.4x | 168 / 188 | 87 / 146 | 83 / 106 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 384x384 | 5.84 | 7.95 | 4.97 | 1.79 | 3.3x | 2.8x | 270 / 290 | 157 / 186 | 132 / 178 |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 512x512 | 11.2 | 8.13 | 4.92 | 3.13 | 2.6x | 1.6x | 525 / 570 | 310 / 342 | 237 / 258 |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 224x224 | 5.92 | 8.54 | 5.19 | 0.846 | 7.0x | 6.1x | 71 / 90 | 52 / 82 | 48 / 78 |
| vgg16.tv_in1k | 224x224 | 3.62 | 4.09 | 1.99 | 1.50 | 2.4x | 1.3x | 825 / 846 | 313 / 372 | 332 / 388 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 224x224 | 6.33 | 5.48 | 3.02 | 1.65 | 3.3x | 1.8x | 531 / 548 | 205 / 390 | 300 / 370 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 224x224 | 76.9 | 25.8 | 13.1 | 12.8 | 2.0x | 1.0x | 2495 / 2564 | 879 / 1708 | 1216 / 1346 |

### Segmentation, synthetic, 1024x2048

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| deeplabv3-r101-d8-ade20k | 108 | 54.8 | 47.9 | 20.1 | 2.7x | 2.4x | 2031 / 2162 | 453 / 532 | 362 / 518 |
| deeplabv3plus-r101-d8-ade20k | 104 | 53.4 | 47.1 | 18.0 | 3.0x | 2.6x | 1891 / 2024 | 413 / 492 | 365 / 560 |
| dinov2-b-linear-ade20k | 115 | 40.1 | 33.3 | 22.6 | 1.8x | 1.5x | 2063 / 2426 | 765 / 884 | 771 / 904 |
| dinov2-l-linear-ade20k | 332 | 95.7 | 83.9 | 56.8 | 1.7x | 1.5x | 3308 / 3726 | 1181 / 1244 | 1189 / 1244 |
| dinov2-s-linear-ade20k | 55.8 | 24.7 | 18.3 | 12.4 | 2.0x | 1.5x | 1694 / 2046 | 642 / 696 | 645 / 676 |
| eomt-l-dinov2-ade20k | 178 | 63.7 | 50.8 | 35.0 | 1.8x | 1.5x | 3827 / 4086 | 1515 / 1584 | 1515 / 1584 |
| eomt-l-dinov3-ade20k | 186 | 72.1 | 54.6 | 38.8 | 1.9x | 1.4x | 3822 / 4086 | 1514 / 1584 | 1514 / 1584 |
| fcn-r101-d8-ade20k | 94.2 | 47.0 | 40.7 | 16.6 | 2.8x | 2.4x | 1925 / 2102 | 417 / 492 | 294 / 478 |
| hrnet-w48-fcn-ade20k | 48.1 | 54.8 | 26.9 | 11.9 | 4.1x | 2.3x | 1915 / 2010 | 424 / 490 | 274 / 430 |
| mask2former-swinl-ade20k | 244 | 145 | 103 | 60.4 | 2.4x | 1.7x | 3572 / 3878 | 1164 / 1400 | 1164 / 1400 |
| oneformer-swinl-ade20k | 307 | 205 | 139 | 100.0 | 2.1x | 1.4x | 7395 / 7792 | 2054 / 2476 | 2054 / 2476 |
| pspnet-r101-d8-ade20k | 92.5 | 47.2 | 48.8 | 25.0 | 1.9x | 2.0x | 1922 / 2222 | 417 / 492 | 341 / 494 |
| sapiens-seg-0.3b-goliath | 466 | 173 | 132 | 92.6 | 1.9x | 1.4x | 2520 / 2724 | 1643 / 1762 | 1643 / 1762 |
| sapiens-seg-1b-goliath | 1189 | 354 | 275 | 185 | 1.9x | 1.5x | 5721 / 5906 | 3259 / 3384 | 3259 / 3382 |
| segformer-b0-ade20k | 17.2 | 17.4 | 11.4 | 3.48 | 5.0x | 3.3x | 1590 / 1766 | 339 / 376 | 148 / 238 |
| segformer-b5-ade20k | 132 | 72.0 | 38.7 | 25.3 | 2.8x | 1.5x | 2531 / 2620 | 546 / 590 | 407 / 534 |
| segnext-mscan-b-ade20k | 44.8 | 33.9 | 28.3 | 12.2 | 2.8x | 2.3x | 1727 / 1838 | 376 / 424 | 229 / 332 |
| segnext-mscan-l-ade20k | 69.7 | 52.2 | 43.1 | 20.6 | 2.5x | 2.1x | 1849 / 1894 | 417 / 458 | 314 / 428 |
| segnext-mscan-s-ade20k | 28.6 | 23.3 | 18.5 | 8.04 | 2.9x | 2.3x | 1649 / 1700 | 350 / 402 | 202 / 290 |
| segnext-mscan-t-ade20k | 21.1 | 23.8 | 15.9 | 8.99 | 2.3x | 1.8x | 1593 / 1648 | 332 / 378 | 147 / 206 |

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| efficientvit-sam-l0 | 26.9 | 26.3 | 12.7 | 4.45 | 5.9x | 2.9x | 211 / 232 | 171 / 210 | 176 / 256 |
| efficientvit-sam-l1 | 30.1 | 29.9 | 14.4 | 5.79 | 5.2x | 2.5x | 260 / 278 | 195 / 234 | 203 / 284 |
| efficientvit-sam-l2 | 34.4 | 34.4 | 17.8 | 6.83 | 5.0x | 2.6x | 313 / 344 | 221 / 260 | 232 / 312 |
| efficientvit-sam-xl0 | 46.9 | 28.2 | 19.0 | 9.75 | 2.9x | 1.9x | 654 / 670 | 433 / 506 | 401 / 492 |
| efficientvit-sam-xl1 | 72.6 | 40.9 | 28.9 | 15.3 | 2.7x | 1.9x | 984 / 1022 | 598 / 676 | 582 / 728 |
| mobile-sam | 30.3 | 23.5 | 10.5 | 6.86 | 3.4x | 1.5x | 207 / 286 | 183 / 210 | 215 / 296 |
| sam-hq-vit-b | 172 | 103 | 49.7 | 36.8 | 2.8x | 1.4x | 1704 / 1756 | 388 / 422 | 433 / 564 |
| sam-hq-vit-h | 738 | 311 | 217 | 144 | 2.2x | 1.5x | 4266 / 4318 | 1500 / 1566 | 1541 / 1708 |
| sam-hq-vit-l | 412 | 202 | 118 | 81.4 | 2.5x | 1.4x | 2969 / 3018 | 846 / 902 | 889 / 1046 |
| sam-vit-b | 168 | 98.2 | 48.3 | 37.0 | 2.7x | 1.3x | 1769 / 1802 | 378 / 420 | 377 / 420 |
| sam-vit-h | 734 | 306 | 214 | 143 | 2.1x | 1.5x | 5295 / 5404 | 1484 / 2524 | 1480 / 1524 |
| sam-vit-l | 406 | 197 | 116 | 81.5 | 2.4x | 1.4x | 3411 / 3464 | 833 / 1262 | 831 / 862 |
| sam2.1-hiera-base-plus | 107 | 47.2 | 32.0 | 25.1 | 1.9x | 1.3x | 823 / 870 | 412 / 464 | 426 / 524 |
| sam2.1-hiera-large | 242 | 86.4 | 68.0 | 48.8 | 1.8x | 1.4x | 1656 / 1724 | 742 / 768 | 760 / 828 |
| sam2.1-hiera-small | 74.8 | 41.8 | 18.8 | 15.6 | 2.7x | 1.2x | 618 / 662 | 318 / 366 | 330 / 386 |
| sam2.1-hiera-tiny | 66.3 | 38.0 | 16.6 | 13.9 | 2.7x | 1.2x | 577 / 636 | 304 / 382 | 316 / 362 |
| sam3 | 695 | 281 | 179 | 116 | 2.4x | 1.5x | 5255 / 5334 | 1922 / 2004 | 2019 / 2264 |

### Detection, synthetic, 800x1344

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| detr-r101-coco | 78.9 | 80.8 | 18.7 | 13.7 | 5.8x | 1.4x | 531 / 650 | 316 / 368 | 254 / 328 |
| detr-r50-coco | 55.5 | 62.0 | 12.4 | 9.93 | 5.6x | 1.2x | 459 / 558 | 280 / 332 | 218 / 272 |
| dfine-l-coco | 37.1 | 44.9 | 16.2 | 8.70 | 4.3x | 1.9x | 278 / 350 | 152 / 172 | 179 / 300 |
| dfine-m-coco | 30.8 | 38.2 | 13.5 | 5.97 | 5.2x | 2.3x | 209 / 256 | 126 / 142 | 156 / 252 |
| dfine-n-coco | 24.3 | 29.1 | 10.3 | 2.72 | 8.9x | 3.8x | 92 / 126 | 88 / 116 | 124 / 202 |
| dfine-s-coco | 25.7 | 31.9 | 11.1 | 4.04 | 6.4x | 2.7x | 152 / 184 | 105 / 128 | 138 / 212 |
| dfine-x-coco | 44.9 | 54.6 | 18.5 | 14.1 | 3.2x | 1.3x | 476 / 536 | 227 / 258 | 237 / 364 |
| faster-rcnn-r50-fpn-3x-coco | 83.6 | 41.4 | 22.8 | 19.5 | 2.1x | 1.2x | 626 / 708 | 390 / 490 | 391 / 450 |
| mask-rcnn-r50-fpn-3x-coco | 85.4 | 43.5 | 24.6 | 21.0 | 2.1x | 1.2x | 604 / 650 | 352 / 440 | 353 / 400 |
| rtdetrv2-l-coco | 30.3 | 38.4 | 12.1 | 6.61 | 4.6x | 1.8x | 354 / 412 | 177 / 200 | 200 / 306 |
| rtdetrv2-m-coco | 17.1 | 20.9 | 10.4 | 5.27 | 3.2x | 2.0x | 254 / 270 | 156 / 170 | 177 / 274 |
| rtdetrv2-ms-coco | 24.7 | 22.5 | 9.60 | 5.46 | 4.1x | 1.8x | 309 / 358 | 166 / 182 | 189 / 306 |
| rtdetrv2-s-coco | 14.1 | 16.8 | 8.04 | 4.19 | 3.4x | 1.9x | 192 / 220 | 134 / 166 | 155 / 250 |
| rtdetrv2-x-coco | 36.3 | 46.8 | 14.6 | 9.24 | 3.9x | 1.6x | 589 / 622 | 244 / 278 | 262 / 384 |
| rtmdet-l-coco | 19.5 | 20.4 | 9.60 | 5.74 | 3.4x | 1.7x | 332 / 356 | 183 / 230 | 178 / 284 |
| rtmdet-m-coco | 14.7 | 17.8 | 7.69 | 4.49 | 3.3x | 1.7x | 168 / 198 | 114 / 148 | 122 / 200 |
| rtmdet-m-person | 14.1 | 16.3 | 7.39 | 4.12 | 3.4x | 1.8x | 168 / 198 | 114 / 146 | 119 / 202 |
| rtmdet-nano-person | 12.9 | 16.0 | 7.56 | 2.10 | 6.1x | 3.6x | 12 / 30 | 48 / 84 | 49 / 92 |
| rtmdet-s-coco | 12.1 | 14.5 | 6.25 | 2.98 | 4.1x | 2.1x | 73 / 92 | 81 / 102 | 88 / 136 |
| rtmdet-tiny-coco | 10.7 | 13.8 | 6.08 | 2.69 | 4.0x | 2.3x | 48 / 64 | 72 / 96 | 80 / 124 |
| rtmdet-x-coco | 32.2 | 24.2 | 13.3 | 8.55 | 2.8x | 1.6x | 585 / 604 | 267 / 316 | 267 / 378 |

### Pose, synthetic, 1080x1920

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| dwpose-l | 178 | 184 | 10.2 | 6.20 | 28.7x | 1.7x | 393 / 428 | 222 / 280 | 186 / 306 |
| dwpose-l-384 | 194 | 203 | 21.2 | 12.5 | 15.5x | 1.7x | 773 / 816 | 351 / 460 | 240 / 424 |
| dwpose-m | 169 | 164 | 7.25 | 4.86 | 33.8x | 1.5x | 756 / 792 | 162 / 196 | 157 / 220 |
| dwpose-s | 173 | 180 | 5.03 | 3.30 | 52.3x | 1.5x | 806 / 848 | 129 / 174 | 138 / 198 |
| dwpose-t | 160 | 174 | 4.47 | 3.15 | 50.8x | 1.4x | 832 / 858 | 124 / 152 | 133 / 192 |
| hrnet-w32 | 176 | 190 | 20.3 | 9.79 | 18.0x | 2.1x | 336 / 398 | 195 / 258 | 210 / 262 |
| hrnet-w32-384 | 216 | 190 | 38.8 | 19.4 | 9.8x | 2.0x | 653 / 760 | 341 / 434 | 305 / 380 |
| hrnet-w48 | 200 | 206 | 32.0 | 15.5 | 12.9x | 2.1x | 440 / 462 | 262 / 322 | 343 / 398 |
| hrnet-w48-384 | 276 | 245 | 70.5 | 33.9 | 7.2x | 2.1x | 1014 / 1082 | 408 / 498 | 438 / 498 |
| rf-detr-keypoint-preview | 29.4 | 19.3 | 18.3 | 9.48 | 2.0x | 1.9x | 186 / 244 | 179 / 204 | 176 / 224 |
| rtmo-l | 17.1 | 21.4 | 10.8 | 8.30 | 2.1x | 1.3x | 800 / 824 | 208 / 226 | 208 / 228 |
| rtmo-m | 14.3 | 17.0 | 8.71 | 6.85 | 2.1x | 1.3x | 422 / 462 | 166 / 190 | 166 / 212 |
| rtmo-s | 11.4 | 15.5 | 7.48 | 6.27 | 1.8x | 1.2x | 228 / 252 | 142 / 160 | 142 / 180 |
| rtmo-t | 11.3 | 13.3 | 7.62 | 6.31 | 1.8x | 1.2x | 118 / 122 | 121 / 136 | 121 / 156 |
| rtmpose-l | 161 | 176 | 8.94 | 5.63 | 28.6x | 1.6x | 1105 / 1154 | 211 / 236 | 175 / 264 |
| rtmpose-l-384 | 192 | 171 | 19.3 | 11.2 | 15.3x | 1.7x | 366 / 410 | 340 / 436 | 229 / 406 |
| rtmpose-l-halpe26 | 180 | 164 | 9.06 | 5.65 | 29.1x | 1.6x | 529 / 572 | 212 / 276 | 176 / 264 |
| rtmpose-l-halpe26-384 | 206 | 206 | 19.3 | 11.2 | 18.4x | 1.7x | 975 / 1038 | 341 / 436 | 230 / 406 |
| rtmpose-m | 179 | 180 | 6.28 | 4.42 | 40.4x | 1.4x | 316 / 364 | 153 / 192 | 148 / 200 |
| rtmpose-m-384 | 174 | 167 | 13.1 | 8.97 | 18.6x | 1.5x | 505 / 582 | 243 / 352 | 202 / 340 |
| rtmpose-m-ap10k | 196 | 186 | 7.98 | 5.41 | 34.5x | 1.5x | 757 / 814 | 177 / 234 | 163 / 220 |
| rtmpose-m-face6 | 198 | 207 | 8.69 | 5.78 | 34.3x | 1.5x | 652 / 698 | 184 / 256 | 169 / 242 |
| rtmpose-m-halpe26 | 179 | 181 | 6.35 | 4.46 | 40.2x | 1.4x | 283 / 318 | 154 / 192 | 149 / 220 |
| rtmpose-m-halpe26-384 | 192 | 193 | 13.2 | 9.01 | 21.3x | 1.5x | 690 / 764 | 244 / 352 | 203 / 340 |
| rtmpose-m-hand5 | 192 | 195 | 8.04 | 5.44 | 35.3x | 1.5x | 425 / 472 | 178 / 234 | 163 / 240 |
| rtmpose-s | 161 | 165 | 4.92 | 3.01 | 53.4x | 1.6x | 449 / 480 | 124 / 172 | 133 / 178 |
| rtmpose-s-face6 | 258 | 237 | 5.03 | 3.75 | 63.2x | 1.3x | 504 / 540 | 147 / 174 | 151 / 222 |
| rtmpose-s-halpe26 | 172 | 173 | 4.98 | 3.04 | 56.5x | 1.6x | 162 / 200 | 124 / 172 | 133 / 198 |
| rtmpose-t | 165 | 151 | 4.49 | 2.90 | 52.0x | 1.5x | 459 / 490 | 120 / 150 | 129 / 176 |
| rtmpose-t-face6 | 188 | 204 | 4.58 | 3.63 | 51.8x | 1.3x | 424 / 464 | 134 / 152 | 146 / 220 |
| rtmpose-t-halpe26 | 181 | 172 | 4.48 | 2.90 | 59.1x | 1.5x | 103 / 144 | 120 / 150 | 129 / 196 |
| rtmpose-x-halpe26-384 | 247 | 217 | 33.5 | 18.7 | 11.6x | 1.8x | 470 / 556 | 401 / 498 | 280 / 470 |
| rtmw-l | 213 | 203 | 15.1 | 10.2 | 19.8x | 1.5x | 865 / 908 | 267 / 308 | 232 / 336 |
| rtmw-l-384 | 274 | 275 | 31.5 | 20.2 | 13.6x | 1.6x | 1094 / 1144 | 396 / 508 | 286 / 476 |
| rtmw-m | 169 | 205 | 11.2 | 7.54 | 22.5x | 1.5x | 379 / 410 | 189 / 224 | 184 / 268 |
| rtmw-x | 218 | 207 | 23.9 | 14.3 | 14.5x | 1.7x | 604 / 636 | 342 / 392 | 298 / 394 |
| rtmw-x-384 | 306 | 291 | 50.6 | 50.8 | 5.7x | 1.0x | 1874 / 1940 | 482 / 592 | 496 / 674 |
| vitpose-base | 139 | 105 | 19.9 | 14.0 | 7.5x | 1.4x | 713 / 762 | 597 / 642 | 574 / 704 |
| vitpose-base-simple | 145 | 109 | 20.1 | 14.1 | 7.7x | 1.4x | 828 / 962 | 590 / 642 | 567 / 684 |
| vitpose-plus-base | 162 | 113 | 19.9 | 13.9 | 8.1x | 1.4x | 918 / 1002 | 597 / 642 | 574 / 704 |
| vitpose-plus-huge | 610 | 242 | 104 | 64.1 | 3.8x | 1.6x | 5732 / 5846 | 1641 / 1704 | 1618 / 1766 |
| vitpose-plus-large | 331 | 165 | 53.7 | 34.9 | 4.7x | 1.5x | 2873 / 3004 | 1014 / 1062 | 991 / 1124 |
| vitpose-plus-small | 120 | 104 | 9.96 | 8.07 | 12.9x | 1.2x | 312 / 376 | 472 / 496 | 449 / 558 |

### Features, synthetic, 224x224

| model | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| c-radio-v3-b | 6.31 | 5.75 | 3.04 | 1.35 | 4.2x | 2.2x | 589 / 626 | 217 / 246 | 284 / 326 |
| c-radio-v3-g | 58.0 | 29.5 | 14.6 | 10.6 | 2.8x | 1.4x | 6672 / 6804 | 2266 / 2308 | 3166 / 3230 |
| c-radio-v3-h | 34.5 | 17.0 | 8.90 | 7.42 | 2.3x | 1.2x | 3810 / 4110 | 1289 / 1308 | 1789 / 1828 |
| c-radio-v3-l | 16.8 | 10.7 | 5.53 | 3.26 | 3.3x | 1.7x | 1838 / 1878 | 648 / 686 | 888 / 946 |
| clip-b16-laion2b | 6.22 | 5.29 | 3.09 | 1.36 | 3.9x | 2.3x | 544 / 580 | 202 / 226 | 270 / 310 |
| clip-b16-openai | 6.28 | 5.65 | 3.35 | 1.35 | 4.2x | 2.5x | 544 / 580 | 204 / 226 | 324 / 370 |
| clip-h14-378-dfn5b | 47.9 | 19.5 | 11.4 | 8.22 | 2.4x | 1.4x | 3722 / 4038 | 1254 / 1288 | 2153 / 2212 |
| clip-h14-dfn5b | 47.5 | 19.4 | 11.4 | 8.22 | 2.4x | 1.4x | 3720 / 4038 | 1252 / 1268 | 2150 / 2192 |
| clip-h14-laion2b | 47.2 | 19.3 | 11.2 | 8.26 | 2.3x | 1.4x | 3720 / 4032 | 1249 / 1268 | 1750 / 1792 |
| clip-l14-336-openai | 22.1 | 10.8 | 6.16 | 4.75 | 2.3x | 1.3x | 1779 / 1814 | 625 / 646 | 1056 / 1090 |
| clip-l14-datacompxl | 21.6 | 10.1 | 5.99 | 4.77 | 2.1x | 1.3x | 1779 / 1812 | 621 / 666 | 862 / 930 |
| clip-l14-dfn2b | 22.0 | 10.6 | 6.16 | 4.75 | 2.2x | 1.3x | 1779 / 1812 | 623 / 666 | 1054 / 1110 |
| clip-l14-laion2b | 21.6 | 10.1 | 5.98 | 4.77 | 2.1x | 1.3x | 1779 / 1812 | 621 / 666 | 862 / 930 |
| clip-l14-openai | 22.0 | 10.8 | 6.16 | 4.74 | 2.3x | 1.3x | 1779 / 1812 | 623 / 666 | 1054 / 1110 |
| convnext_base.fb_in22k | 7.68 | 11.2 | 5.57 | 1.96 | 3.9x | 2.8x | 537 / 544 | 289 / 296 | 204 / 216 |
| convnext_large.fb_in22k | 10.5 | 11.5 | 5.53 | 3.23 | 3.3x | 1.7x | 1160 / 1180 | 600 / 612 | 413 / 434 |
| convnext_small.fb_in22k | 7.51 | 11.2 | 5.64 | 1.79 | 4.2x | 3.2x | 318 / 336 | 179 / 190 | 130 / 150 |
| convnext_tiny.fb_in22k | 3.89 | 5.91 | 3.18 | 1.09 | 3.6x | 2.9x | 194 / 214 | 117 / 130 | 88 / 110 |
| dinov2-vitb14 | 6.90 | 4.50 | 2.92 | 1.48 | 3.0x | 2.0x | 530 / 548 | 205 / 226 | 237 / 290 |
| dinov2-vitb14-reg | 7.00 | 4.30 | 2.94 | 1.50 | 2.9x | 2.0x | 530 / 548 | 207 / 226 | 237 / 288 |
| dinov2-vitg14 | 86.4 | 33.9 | 18.9 | 13.2 | 2.6x | 1.4x | 6542 / 6548 | 2216 / 2230 | 2246 / 2318 |
| dinov2-vitg14-reg | 86.6 | 34.2 | 18.7 | 13.0 | 2.6x | 1.4x | 6542 / 6568 | 2217 / 2228 | 2222 / 4430 |
| dinov2-vitl14 | 22.8 | 9.80 | 6.12 | 4.59 | 2.1x | 1.3x | 1777 / 1784 | 622 / 646 | 655 / 714 |
| dinov2-vitl14-reg | 22.5 | 10.7 | 6.17 | 4.60 | 2.3x | 1.3x | 1777 / 1786 | 625 / 646 | 655 / 714 |
| dinov2-vits14 | 3.30 | 4.40 | 2.95 | 0.848 | 3.9x | 3.5x | 160 / 166 | 79 / 98 | 112 / 148 |
| dinov2-vits14-reg | 4.11 | 5.59 | 2.98 | 0.814 | 5.0x | 3.7x | 160 / 168 | 80 / 98 | 112 / 142 |
| dinov3-convnext-base | 7.38 | 8.24 | 5.71 | 2.18 | 3.4x | 2.6x | 535 / 552 | 208 / 232 | 200 / 214 |
| dinov3-convnext-large | 10.5 | 9.59 | 5.45 | 3.59 | 2.7x | 1.5x | 1163 / 1252 | 419 / 430 | 441 / 494 |
| dinov3-convnext-small | 5.97 | 8.09 | 5.42 | 1.99 | 3.0x | 2.7x | 328 / 348 | 134 / 150 | 127 / 152 |
| dinov3-convnext-tiny | 2.96 | 4.47 | 3.04 | 1.18 | 2.5x | 2.6x | 198 / 206 | 93 / 108 | 86 / 112 |
| dinov3-vitb16 | 6.95 | 9.33 | 4.86 | 1.40 | 5.0x | 3.5x | 527 / 550 | 201 / 206 | 201 / 252 |
| dinov3-vith16plus | 48.0 | 25.9 | 13.3 | 9.30 | 2.8x | 1.4x | 4849 / 4876 | 1645 / 1668 | 1644 / 1694 |
| dinov3-vitl16 | 19.2 | 17.7 | 9.13 | 3.28 | 5.4x | 2.8x | 1772 / 1812 | 617 / 646 | 617 / 674 |
| dinov3-vits16 | 6.47 | 9.16 | 4.67 | 0.903 | 7.2x | 5.2x | 158 / 166 | 77 / 80 | 77 / 106 |
| pe-core-b16-224 | 7.14 | 9.35 | 4.97 | 1.47 | 4.9x | 3.4x | 584 / 620 | 218 / 266 | 285 / 348 |
| pe-core-g14-448 | 97.7 | 52.4 | 25.4 | 17.0 | 3.1x | 1.5x | 10848 / 11330 | 3642 / 3670 | 5180 / 5256 |
| pe-core-l14-336 | 23.3 | 17.7 | 9.09 | 4.96 | 3.6x | 1.8x | 1848 / 1888 | 650 / 668 | 889 / 932 |
| pe-core-s16-384 | 7.02 | 9.28 | 4.80 | 0.951 | 7.4x | 5.0x | 171 / 180 | 83 / 102 | 100 / 128 |
| pe-core-t16-384 | 6.74 | 9.04 | 4.68 | 0.683 | 9.9x | 6.9x | 68 / 70 | 48 / 56 | 53 / 82 |
| pe-spatial-b16-512 | 7.02 | 8.79 | 4.74 | 1.39 | 5.1x | 3.4x | 544 / 580 | 207 / 226 | 274 / 328 |
| pe-spatial-g14-448 | 97.9 | 52.7 | 25.5 | 16.9 | 3.1x | 1.5x | 10688 / 11178 | 3584 / 3612 | 5123 / 5156 |
| pe-spatial-l14-448 | 23.0 | 17.2 | 8.84 | 4.86 | 3.5x | 1.8x | 1780 / 1816 | 626 / 668 | 865 / 932 |
| pe-spatial-s16-512 | 6.70 | 8.79 | 4.51 | 0.916 | 7.3x | 4.9x | 161 / 178 | 81 / 100 | 98 / 126 |
| pe-spatial-t16-512 | 6.56 | 8.73 | 4.48 | 0.666 | 9.8x | 6.7x | 66 / 68 | 49 / 56 | 53 / 82 |
| siglip2-b16-256 | 6.29 | 6.11 | 3.28 | 1.43 | 4.3x | 2.3x | 583 / 620 | 216 / 248 | 284 / 350 |
| siglip2-b16-512 | 6.32 | 5.99 | 3.31 | 1.43 | 4.2x | 2.3x | 583 / 618 | 219 / 266 | 287 / 368 |
| siglip2-g16-256 | 59.8 | 30.9 | 14.7 | 10.6 | 2.9x | 1.4x | 6733 / 6854 | 2262 / 2290 | 3163 / 3232 |
| siglip2-g16-384 | 59.9 | 31.0 | 14.7 | 10.6 | 2.9x | 1.4x | 6735 / 6854 | 2264 / 2288 | 3165 / 3230 |
| siglip2-l16-256 | 17.4 | 11.1 | 5.71 | 3.31 | 3.3x | 1.7x | 1843 / 1890 | 643 / 668 | 883 / 930 |
| siglip2-l16-512 | 17.5 | 11.1 | 5.70 | 3.31 | 3.4x | 1.7x | 1846 / 1888 | 647 / 666 | 887 / 930 |
| siglip2-so400m14-224 | 23.9 | 12.4 | 6.38 | 4.30 | 2.9x | 1.5x | 2515 / 2632 | 858 / 868 | 1182 / 1230 |
| siglip2-so400m14-378 | 23.9 | 12.5 | 6.38 | 4.29 | 2.9x | 1.5x | 2516 / 2632 | 861 / 908 | 1185 / 1250 |
| siglip2-so400m16-256 | 22.8 | 12.6 | 6.42 | 4.19 | 3.0x | 1.5x | 2515 / 2636 | 857 / 868 | 1181 / 1232 |
| siglip2-so400m16-512 | 22.8 | 12.4 | 6.40 | 4.18 | 3.0x | 1.5x | 2518 / 2636 | 861 / 906 | 1185 / 1250 |
| tipsv2-b14 | 7.03 | 7.21 | 2.93 | 1.49 | 4.7x | 2.0x | 529 / 548 | 204 / 224 | 206 / 226 |
| tipsv2-l14 | 21.9 | 14.2 | 6.15 | 4.58 | 3.1x | 1.3x | 1776 / 1786 | 621 / 624 | 624 / 648 |

### Multiview, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| mapanything-apache | 336x518 | 2875 | 1967 | 636 | 544 | 3.6x | 1.2x | 10227 / 10952 | 3697 / 4092 | 3697 / 4072 |
| mvsanywhere | 448x672 | 273 | 219 | 93.2 | 35.6 | 6.1x | 2.6x | 1504 / 1576 | 931 / 1226 | 761 / 884 |
| patchmatchnet | 1536x1024 | 248 | 220 | 202 | 29.6 | 7.4x | 6.8x | 1228 / 1272 | 1806 / 2096 | 740 / 856 |
| vggt-1b | 336x518 | 2502 | 965 | 651 | 458 | 2.1x | 1.4x | 8688 / 9158 | 3691 / 4674 | 3702 / 4654 |
| vggt-omega-1b-512 | 336x518 | 2650 | 906 | 726 | 464 | 2.0x | 1.6x | 7495 / 7654 | 3421 / 3518 | 3422 / 3518 |

### Flow, synthetic

| model | input | time (ms): original fp32 | original | portable | kernels | kernels speed-up: vs original | vs portable | peak memory (MiB, allocated / reserved): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| flowformer-small-chairs | 375x1242 | 352 | 303 | 351 | – | – | – | 2382 / 2970 | 3165 / 3638 | – |
| flowformer-small-sintel | 375x1242 | 352 | 303 | 351 | – | – | – | 2382 / 2990 | 3165 / 3638 | – |
| flowformer-small-things | 375x1242 | 352 | 303 | 351 | – | – | – | 2382 / 3010 | 3165 / 3638 | – |
| flowseek-t | 1080x1920 | 650 | 457 | 395 | – | – | – | 8693 / 9880 | 8671 / 8712 | – |
| flowseek-t-things | 1080x1920 | 650 | 458 | 395 | – | – | – | 8693 / 9820 | 8671 / 8712 | – |
| flowseek-t-things-notartan | 1080x1920 | 651 | 458 | 396 | – | – | – | 8693 / 9820 | 8671 / 8712 | – |
| gmflow-scale1-mixdata | 1080x1920 | 1246 | 750 | 800 | – | – | – | 8360 / 8448 | 5523 / 5660 | – |
| gmflow-scale1-things | 1080x1920 | 1245 | 750 | 801 | – | – | – | 8360 / 8468 | 5523 / 5660 | – |
| gmflow-scale2-mixdata | 1080x1920 | 2544 | 1608 | 1645 | – | – | – | 8425 / 8572 | 6433 / 7244 | – |
| gmflow-scale2-regrefine6-kitti | 1080x1920 | 3255 | 2188 | 2194 | – | – | – | 8537 / 8822 | 6443 / 7250 | – |
| gmflow-scale2-regrefine6-mixdata | 1080x1920 | 3254 | 2188 | 2192 | – | – | – | 8537 / 8822 | 6443 / 7250 | – |
| gmflow-scale2-regrefine6-sintelft | 1080x1920 | 3255 | 2184 | 2189 | – | – | – | 8537 / 8822 | 6443 / 7250 | – |
| gmflow-scale2-regrefine6-things | 1080x1920 | 3230 | 2175 | 2183 | – | – | – | 8537 / 8822 | 6443 / 7250 | – |
| gmflow-scale2-sintel | 1080x1920 | 2541 | 1607 | 1644 | – | – | – | 8425 / 8552 | 6433 / 7244 | – |
| gmflow-scale2-things | 1080x1920 | 2537 | 1599 | 1636 | – | – | – | 8425 / 8552 | 6433 / 7244 | – |
| neuflow-v2 | 1080x1920 | 171 | 75.6 | 103 | 59.0 | 1.3x | 1.7x | 4367 / 4556 | 8392 / 8416 | 790 / 896 |
| neuflow-v2-sintel | 1080x1920 | 171 | 75.7 | 102 | 58.9 | 1.3x | 1.7x | 4367 / 4576 | 8392 / 8416 | 790 / 896 |
| neuflow-v2-things | 1080x1920 | 172 | 75.8 | 103 | 58.9 | 1.3x | 1.7x | 4374 / 4720 | 8392 / 8416 | 790 / 896 |
| raft-large | 1080x1920 | 480 | 567 | 290 | 83.3 | 5.8x | 3.5x | 7164 / 10928 | 8268 / 8298 | 333 / 352 |
| raft-large-kitti | 1080x1920 | 480 | 567 | 290 | 83.6 | 5.7x | 3.5x | 7164 / 10928 | 8268 / 8298 | 333 / 352 |
| raft-large-things | 1080x1920 | 481 | 566 | 290 | 83.1 | 5.8x | 3.5x | 7164 / 10928 | 8268 / 8298 | 333 / 352 |
| raft-small | 1080x1920 | 196 | 395 | 144 | 36.3 | 5.4x | 4.0x | 7032 / 10888 | 8198 / 8226 | 197 / 224 |
| sea-raft-m | 1080x1920 | 504 | 367 | 299 | 73.1 | 5.0x | 4.1x | 7385 / 8570 | 8333 / 8420 | 435 / 514 |
| sea-raft-m-kitti | 1080x1920 | 504 | 367 | 299 | 73.1 | 5.0x | 4.1x | 7385 / 8570 | 8333 / 8420 | 435 / 514 |
| sea-raft-m-spring | 1080x1920 | 96.1 | 56.7 | 52.3 | 23.2 | 2.4x | 2.3x | 761 / 812 | 713 / 740 | 181 / 214 |
| sea-raft-m-things | 1080x1920 | 504 | 367 | 299 | 73.1 | 5.0x | 4.1x | 7385 / 8570 | 8333 / 8420 | 435 / 514 |
| sea-raft-s | 1080x1920 | 365 | 300 | 226 | 50.4 | 6.0x | 4.5x | 7323 / 8502 | 8292 / 8354 | 415 / 492 |
| sea-raft-s-kitti | 1080x1920 | 365 | 299 | 226 | 50.5 | 5.9x | 4.5x | 7323 / 8502 | 8292 / 8354 | 415 / 492 |
| sea-raft-s-spring | 1080x1920 | 67.1 | 42.4 | 37.1 | 16.0 | 2.6x | 2.3x | 699 / 764 | 672 / 714 | 160 / 192 |

### Flow, KITTI 2015, 375x1242

| model | time (ms): portable | kernels | kernels speed-up: vs portable | peak memory (MiB, allocated / reserved): portable | kernels |
|---|---|---|---|---|---|
| raft-large | 45.8 | 21.0 | 2.2x | 511 / 738 | 85 / 112 |
| raft-large-kitti | 45.8 | 21.0 | 2.2x | 511 / 738 | 85 / 112 |
| raft-large-things | 45.8 | 21.1 | 2.2x | 511 / 738 | 85 / 112 |
| raft-small | 23.2 | 12.1 | 1.9x | 483 / 508 | 48 / 66 |

### Startup and first call

load: from the load call until the model is on the GPU; first call: the whole first inference call, including one normal inference (see the speed table for the warm time). Seconds, at each task's standard size (another size is shown in brackets).

| model | load (s): original | portable | kernels | first call (s): original | portable | kernels |
|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | 0.15 (375x1242) | 0.16 | 0.20† | 0.70 (375x1242) | 0.17† | 0.09† |
| aanet-sceneflow | 0.12 (375x1242) | 0.19 | 0.27† | 0.07 (375x1242) | 0.14† | 0.10† |
| bridgedepth-rvc-pretrain | 5.87 (375x1242) | 1.84 | 0.76† | 1.36 (375x1242) | 0.45† | 0.36† |
| bridgedepth-sf | 4.86 (375x1242) | 1.53 | 0.72† | 0.12 (375x1242) | 0.44† | 0.35† |
| c-fast-foundationstereo | 0.17 (375x1242) | 0.24 | 0.16† | 0.17 (375x1242) | 0.33† | 0.14† |
| c-foundationstereo-s | 1.67 (375x1242) | 0.62 | 0.40† | 2.73 (375x1242) | 3.65† | 1.36† |
| cas-gwcnet-c-sceneflow | 0.48 (375x1242) | 1.29 | 0.18† | 0.41 (375x1242) | 0.33† | 0.23† |
| cas-gwcnet-gc-sceneflow | 0.39 | 0.97 | 0.17† | 0.64 | 0.47† | 0.36† |
| cfnet-sceneflow | 0.55 | 0.94 | 0.21† | 0.56 | 0.48† | 0.40† |
| coex-sceneflow | – | 0.16 | 0.09† | 0.63 (375x1242) | 0.06† | 0.04† |
| crestereo-eth3d | 0.05 | 0.03 | 0.03† | 1.50 | 0.75† | 0.23† |
| defom-stereo-vitl-sceneflow | 4.24 | 2.31 | 1.18† | 2.83 | 1.53† | 0.66† |
| dlnr-middlebury | 0.83 (375x1242) | 0.63 | 0.26† | 1.71 (375x1242) | 0.40† | 0.21† |
| dlnr-sceneflow | 0.83 (375x1242) | 0.63 | 0.26† | 1.71 (375x1242) | 0.40† | 0.21† |
| fast-acvnet-generalization | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-kitti12 | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-kitti15 | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-acvnet-plus-generalization | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-kitti12 | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-kitti15 | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-plus-sceneflow | 0.23 (375x1242) | 0.10 | 0.10† | 0.04 (375x1242) | 0.07† | 0.05† |
| fast-acvnet-sceneflow | 0.24 (375x1242) | 0.18 | 0.20† | 1.73 (375x1242) | 0.08† | 0.04† |
| fast-foundationstereo | 0.16 (375x1242) | 0.25 | 0.14† | 0.14 (375x1242) | 0.33† | 0.14† |
| foundationstereo-vit-l | 10.86 (375x1242) | 1.98 | 1.17† | 2.56 (375x1242) | 2.99† | 1.52† |
| foundationstereo-vit-s | 1.44 (375x1242) | 1.03 | 0.33† | 1.06 (375x1242) | 2.51† | 1.12† |
| great-igev-eth3d | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| great-igev-kitti | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| great-igev-middlebury | 0.39 (375x1242) | 3.91 | 0.61† | 4.59 (375x1242) | 1.81† | 0.59† |
| great-igev-sceneflow | 0.44 (375x1242) | 0.32 | 0.23† | 4.60 (375x1242) | 1.10† | 0.40† |
| gwcnet-g-sceneflow | 0.26 (375x1242) | 0.40 | 0.14† | 0.46 (375x1242) | 0.34† | 0.24† |
| gwcnet-gc-sceneflow | 0.29 (375x1242) | 0.39 | 0.44† | 0.12 (375x1242) | 0.35† | 0.26† |
| hitnet-eth3d | 0.31 | 0.09 | 0.07† | 2.06 | 0.08† | 0.03† |
| hitnet-middlebury-d400 | 0.57 | 0.13 | 0.10† | 3.51 | 0.18† | 0.05† |
| igev-plusplus-eth3d | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-kitti12 | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-kitti15 | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-middlebury | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-plusplus-sceneflow | 0.42 (375x1242) | 0.20 | 0.16† | 2.33 (375x1242) | 1.10† | 0.40† |
| igev-stereo-eth3d | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti12 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-kitti15 | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-middlebury | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| igev-stereo-sceneflow | 0.32 (375x1242) | 0.47 | 0.15† | 2.56 (375x1242) | 0.76† | 0.29† |
| las2-h | 0.26 (375x1242) | 0.18 | 0.13† | 1.21 (375x1242) | 0.14† | 0.06† |
| las2-l | 0.39 (375x1242) | 0.23 | 0.18† | 0.04 (375x1242) | 0.08† | 0.03† |
| las2-m | 0.17 (375x1242) | 0.12 | 0.10† | 0.02 (375x1242) | 0.05† | 0.02† |
| las2-s | 0.11 (375x1242) | 0.09 | 0.07† | 0.02 (375x1242) | 0.05† | 0.01† |
| mocha-v1-sceneflow | 1.95 (375x1242) | 0.79 | 0.48† | 1.85 (375x1242) | 1.06† | 0.45† |
| mocha-v2-sceneflow | 2.37 (375x1242) | 1.30 | 0.45† | 0.40 (375x1242) | 1.03† | 0.45† |
| monster-mix-all | 5.85 (375x1242) | 4.83 | 0.95† | 1.26 (375x1242) | 1.59† | 0.67† |
| monster-plusplus-mix-all | 8.37 (375x1242) | 2.45 | 0.79† | 1.23 (375x1242) | 1.63† | 0.70† |
| monster-sceneflow | 5.85 (375x1242) | 4.83 | 0.95† | 1.26 (375x1242) | 1.59† | 0.67† |
| nmrf-sceneflow | 0.21 (375x1242) | 0.10 | 0.07† | 0.77 (375x1242) | 0.22† | 0.20† |
| psmnet-sceneflow | 0.20 (375x1242) | 0.10 | 0.08† | 0.53 (375x1242) | 0.35† | 0.23† |
| raft-stereo-eth3d | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| raft-stereo-middlebury | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| raft-stereo-sceneflow | 0.27 (375x1242) | 0.25 | 0.05† | 1.45 (375x1242) | 0.75† | 0.27† |
| rt-monster-plusplus-mix-all | 0.99 (375x1242) | 0.26 | 0.23† | 0.66 (375x1242) | 0.13† | 0.10† |
| s2m2-l | 1.38 | 0.99 | 0.58† | 1.01 | 1.01† | 0.37† |
| s2m2-m | 3.95 (375x1242) | 0.50 | 0.31† | 0.27 (375x1242) | 0.68† | 0.25† |
| s2m2-s | 0.22 (375x1242) | 0.22 | 0.20† | 0.14 (375x1242) | 0.35† | 0.11† |
| s2m2-xl | 4.50 (375x1242) | 2.73 | 0.90† | 0.53 (375x1242) | 1.81† | 0.68† |
| selective-igev-eth3d-finetune | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-eth3d-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti12 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-kitti15 | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-finetune | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-middlebury-train | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| selective-igev-sceneflow | 0.36 (375x1242) | 0.50 | 0.14† | 2.25 (375x1242) | 1.02† | 0.35† |
| waft-stereo-b | 3.07 (375x1242) | 1.65 | 0.42† | 0.44 (375x1242) | 1.15† | 0.39† |
| waft-stereo-s | 1.60 (375x1242) | 0.58 | 0.30† | 0.14 (375x1242) | 0.69† | 0.28† |
| da3-giant-1.1 | 12.39 | 6.29 | 0.90† | 0.68 | 0.16 | 0.06 |
| da3-large-1.1 | 3.46 | 3.65 | 0.31† | 0.45 | 0.25 | 0.03 |
| da3metric-large | 2.70 | 0.35 | 0.30† | 0.28 | 0.05 | 0.02 |
| da3mono-large | 2.96 | 4.25 | 1.44† | 0.28 | 0.09 | 0.02 |
| da3nested-giant-large-1.1 | 28.67 | 13.25 | 2.73† | 1.63 | 0.55 | 0.07 |
| depth-anything-v2-b | 3.94 | 0.85 | 0.20† | 0.08 | 0.02 | 0.01 |
| depth-anything-v2-l | 5.66 | 2.39 | 0.91† | 0.15 | 0.08 | 0.06 |
| depth-anything-v2-metric-hypersim-b | 2.13 | 0.33 (518x770) | 1.13 (518x770) | 0.07 | 0.27 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-hypersim-l | 4.87 | 0.84 (518x770) | 1.73 (518x770) | 0.15 | 0.15 (518x770) | 0.07 (518x770) |
| depth-anything-v2-metric-hypersim-s | 1.54 | 0.19 (518x770) | 0.95 (518x770) | 0.17 | 0.23 (518x770) | 0.06 (518x770) |
| depth-anything-v2-metric-vkitti-b | 2.06 | 0.30 (518x770) | 1.13 (518x770) | 0.06 | 0.28 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-vkitti-l | 4.87 | 0.84 (518x770) | 1.65 (518x770) | 0.14 | 0.15 (518x770) | 0.05 (518x770) |
| depth-anything-v2-metric-vkitti-s | 1.46 | 0.13 (518x770) | 0.96 (518x770) | 0.04 | 0.16 (518x770) | 0.05 (518x770) |
| depth-anything-v2-s | 0.71 | 0.11 | 0.08† | 0.38 | 0.03 | 0.01 |
| depth-pro | 17.34 | 4.67 | 1.11† | 0.74 | 0.58 | 0.37 |
| distill-any-depth-b | 0.84 | 0.29 (518x770) | 1.13 (518x770) | 0.06 | 0.27 (518x770) | 0.04 (518x770) |
| distill-any-depth-l | 3.03 | 0.82 (518x770) | 1.68 (518x770) | 0.14 | 0.14 (518x770) | 0.06 (518x770) |
| distill-any-depth-s | 0.25 | 0.13 (518x770) | 0.96 (518x770) | 0.04 | 0.16 (518x770) | 0.04 (518x770) |
| marigold-depth-v1-1 | 1.32 | 8.20 | 1.10† | 0.60 | 0.56 | 0.52 |
| metric3d-vit-large | 9.96 | 3.45 | 0.31† | 0.88 | 0.23 | 0.16 |
| midas-dpt-beit-large-512 | 5.16 | 2.49 | 0.28† | 1.14 | 0.09 | 0.07 |
| moge-2-vitb-normal | 1.26 | 0.49 (518x770) | 1.24 (518x770) | 0.24 | 0.40 (518x770) | 0.28 (518x770) |
| moge-2-vitl-normal | 7.24 | 1.37 | 0.63† | 0.17 | 0.14 | 0.13 |
| moge-2-vits-normal | 0.88 | 0.35 | 0.11† | 1.50 | 0.08 | 0.10 |
| moge-3-vitl | 10.60 | 1.02 | 1.66 | 2.92 | 0.56 | 0.30 |
| unidepth-v2-vitl14 | 3.80 | 4.16 | 0.90† | 0.38 | 0.15 | 0.07 |
| unik3d-vitl | 7.28 | 4.16 | 0.84† | 0.57 | 0.14 | 0.07 |
| zoedepth-nk | 8.35 | 1.92 | 0.33† | 1.53 | 0.28 | 0.12 |
| alexnet.tv_in1k | 2.51 | 1.30 | 1.40 | 0.81 | 0.02 | 0.06 |
| coatnet_0_rw_224.sw_in1k | 0.38 | 1.37 | – | 0.31 | 0.38 | – |
| coatnet_1_rw_224.sw_in1k | 0.47 | 1.44 | – | 0.15 | 0.37 | – |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 0.74 | 1.52 | – | 0.31 | 0.51 | – |
| coatnet_nano_rw_224.sw_in1k | 0.28 | 1.37 | – | 0.54 | 0.42 | – |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 0.47 | 1.49 | – | 0.02 | 0.37 | – |
| coatnet_rmlp_1_rw_224.sw_in1k | 0.47 | 1.47 | – | 0.02 | 0.37 | – |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 0.71 | 1.80 | – | 0.02 | 0.50 | – |
| coatnet_rmlp_2_rw_224.sw_in1k | 0.71 | 1.54 | – | 0.02 | 0.51 | – |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 0.74 (384x384) | 1.52 (384x384) | – | 0.05 (384x384) | 0.43 (384x384) | – |
| coatnet_rmlp_nano_rw_224.sw_in1k | 0.26 | 1.36 | – | 0.19 | 0.39 | – |
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
| convnextv2_base.fcmae_ft_in22k_in1k | 2.70 | 1.23 | 2.12 | 0.63 | 0.03 | 0.16 |
| davit_base.msft_in1k | 0.99 | 1.84 | – | 0.56 | 0.27 | – |
| davit_small.msft_in1k | 0.53 | 1.60 | – | 0.02 | 0.23 | – |
| davit_tiny.msft_in1k | 0.39 | 1.47 | – | 0.85 | 0.25 | – |
| deit_base_patch16_224.fb_in1k | 2.52 | 0.72 | 1.96 | 0.31 | 0.00 | 0.07 |
| densenet121.tv_in1k | 1.22 | 0.41 | 1.89 | 0.56 | 0.09 | 0.04 |
| dinov2_vitl14_lc | 5.80 | 3.34 | 3.94 | 0.32 | 0.02 | 0.10 |
| efficientformer_l1.snap_dist_in1k | 0.22 | 1.71 | – | 0.22 | 0.07 | – |
| efficientformer_l3.snap_dist_in1k | 0.37 | 1.86 | – | 0.03 | 0.08 | – |
| efficientformer_l7.snap_dist_in1k | 0.93 | 2.27 | – | 0.03 | 0.07 | – |
| efficientformerv2_l.snap_dist_in1k | 0.32 | 2.74 | – | 0.73 | 0.43 | – |
| efficientformerv2_s0.snap_dist_in1k | 0.17 | 1.88 | – | 1.76 | 0.50 | – |
| efficientformerv2_s1.snap_dist_in1k | 0.19 | 1.96 | – | 0.59 | 0.43 | – |
| efficientformerv2_s2.snap_dist_in1k | 0.24 | 2.68 | – | 0.86 | 0.44 | – |
| efficientnet_b0.tv_in1k | 1.25 | 0.17 | 1.53 | 2.86 | 0.38 | 0.06 |
| efficientnet_b4.tv_in1k | 0.41 | 0.37 | 1.94 | 1.72 | 0.06 | 0.07 |
| efficientnet_v2_s.tv_in1k | 1.48 | 0.35 | 1.90 | 0.97 | 0.04 | 0.05 |
| efficientvit_b2.r288_in1k | 1.84 | 0.40 | 1.92 | 1.89 | 0.04 | 0.05 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 6.98 | 2.80 | 3.61 | 0.39 | 0.02 | 0.10 |
| fastvit_sa24.apple_dist_in1k | 0.31 (256x256) | 3.40 (256x256) | 2.30 (256x256) | 0.03 (256x256) | 0.58 (256x256) | 0.00 (256x256) |
| fastvit_t12.apple_dist_in1k | 0.14 (256x256) | 1.81 (256x256) | 2.01 (256x256) | 0.71 (256x256) | 0.41 (256x256) | 0.04 (256x256) |
| fastvit_t8.apple_dist_in1k | 0.21 (256x256) | 1.76 (256x256) | 1.95 (256x256) | 0.98 (256x256) | 0.55 (256x256) | 0.04 (256x256) |
| focalnet_base_lrf.ms_in1k | 0.91 | 1.89 | – | 0.03 | 0.62 | – |
| focalnet_base_srf.ms_in1k | 0.89 | 1.92 | – | 0.05 | 0.49 | – |
| focalnet_small_lrf.ms_in1k | 0.54 | 1.86 | – | 0.03 | 0.69 | – |
| focalnet_small_srf.ms_in1k | 0.54 | 1.73 | – | 0.02 | 0.49 | – |
| focalnet_tiny_lrf.ms_in1k | 0.36 | 2.56 | – | 0.15 | 0.77 | – |
| focalnet_tiny_srf.ms_in1k | 0.40 | 1.49 | – | 0.45 | 0.49 | – |
| hgnetv2_b0.ssld_stage2_ft_in1k | 0.19 | 1.67 | – | 1.36 | 0.73 | – |
| hgnetv2_b1.ssld_stage2_ft_in1k | 0.31 | 1.87 | – | 0.31 | 0.40 | – |
| hgnetv2_b2.ssld_stage2_ft_in1k | 0.23 | 1.79 | – | 0.59 | 0.39 | – |
| hgnetv2_b3.ssld_stage2_ft_in1k | 0.27 | 1.81 | – | 0.17 | 0.40 | – |
| hgnetv2_b4.ssld_stage2_ft_in1k | 0.32 | 1.74 | – | 0.30 | 0.39 | – |
| hgnetv2_b5.ssld_stage2_ft_in1k | 0.49 | 1.95 | – | 0.30 | 0.40 | – |
| hgnetv2_b6.ssld_stage2_ft_in1k | 0.86 | 2.21 | – | 0.45 | 0.46 | – |
| inception_v3.tv_in1k | 1.62 | 0.28 | 1.86 | 0.40 | 0.04 | 0.05 |
| mobilenet_v2.tv2_in1k | 1.23 | 0.11 | 1.51 | 1.65 | 0.03 | 0.05 |
| mobilenet_v3_large.tv2_in1k | 1.25 | 0.34 | 1.46 | 1.97 | 0.03 | 0.05 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 1.34 | 0.06 | 1.70 | 1.50 | 0.01 | 0.05 |
| mobilenetv4_conv_medium.e500_r256_in1k | 1.56 | 0.36 | 1.89 | 2.29 | 0.04 | 0.05 |
| regnety_160.tv2_in1k | 3.07 | 1.08 | 2.22 | 0.48 | 0.04 | 0.06 |
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
| resnet152.tv2_in1k | 2.34 | 0.43 | 1.88 | 0.24 | 0.02 | 0.04 |
| resnet18.tv_in1k | 0.27 | 0.07 | 1.48 | 0.00 | 0.00 | 0.04 |
| resnet50.tv2_in1k | 0.51 | 0.16 | 1.48 | 0.01 | 0.00 | 0.04 |
| shufflenet_v2_x1_0.tv_in1k | 1.13 | 0.14 | 1.55 | 1.15 | 0.02 | 0.08 |
| shvit_s1.in1k | 0.31 | 2.16 | – | 0.92 | 0.41 | – |
| shvit_s2.in1k | 0.20 | 2.32 | – | 0.30 | 0.48 | – |
| shvit_s3.in1k | 0.21 | 2.80 | – | 0.57 | 0.54 | – |
| shvit_s4.in1k | 0.23 (256x256) | 2.49 (256x256) | – | 0.31 (256x256) | 0.44 (256x256) | – |
| squeezenet1_1.tv_in1k | 1.09 | 0.05 | 1.44 | 0.37 | 0.02 | 0.03 |
| starnet_s1.in1k | 0.18 | 1.60 | 2.20 | 0.26 | 0.24 | 0.21 |
| starnet_s2.in1k | 0.17 | 1.62 | 1.86 | 0.02 | 0.24 | 0.06 |
| starnet_s3.in1k | 0.30 | 1.60 | 1.82 | 0.01 | 0.24 | 0.05 |
| starnet_s4.in1k | 0.21 | 1.66 | 2.00 | 0.01 | 0.24 | 0.05 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 2.71 | 0.94 | 2.28 | 0.41 | 0.03 | 0.09 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 0.56 | 1.83 | 1.44 | 0.60 | 0.41 | 0.11 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 0.37 | 1.98 | 1.45 | 0.99 | 0.40 | 0.10 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 0.69 (384x384) | 2.26 (384x384) | 1.81 (384x384) | 0.80 (384x384) | 0.38 (384x384) | 0.07 (384x384) |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 1.46 (512x512) | 3.05 (512x512) | 2.72 (512x512) | 0.16 (512x512) | 0.38 (512x512) | 0.05 (512x512) |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 0.26 | 2.02 | 1.37 | 1.28 | 0.26 | 0.12 |
| vgg16.tv_in1k | 3.22 | 1.74 | 1.90 | 0.30 | 0.00 | 0.06 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 2.49 | 0.73 | 2.05 | 0.28 | 0.00 | 0.07 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 7.52 | 7.29 | 5.82 | 0.35 | 0.07 | 0.09 |
| deeplabv3-r101-d8-ade20k | 1.64 (512x512) | 0.73 | 0.91† | 0.03 (512x512) | 0.08 | 0.02† |
| deeplabv3plus-r101-d8-ade20k | 4.72 (512x512) | 0.62 | 0.75† | 0.66 (512x512) | 0.10 | 0.02† |
| dinov2-b-linear-ade20k | 12.01 (512x512) | 0.27 | 0.40† | 0.02 (512x512) | 0.10 | 0.07† |
| dinov2-l-linear-ade20k | 7.86 (512x512) | 0.81 | 0.92† | 0.38 (512x512) | 0.16 | 0.10† |
| dinov2-s-linear-ade20k | 4.02 (512x512) | 0.12 | 0.23† | 0.30 (512x512) | 0.09 | 0.06† |
| eomt-l-dinov2-ade20k | 4.80 (512x512) | 3.13 | 3.27† | 0.54 (512x512) | 0.05 | 0.04† |
| eomt-l-dinov3-ade20k | 1.83 (512x512) | 3.20 | 3.24† | 0.07 (512x512) | 0.08 | 0.04† |
| fcn-r101-d8-ade20k | 4.72 (512x512) | 0.53 | 0.91† | 0.39 (512x512) | 0.07 | 0.02† |
| hrnet-w48-fcn-ade20k | 1.48 (512x512) | 0.62 | 0.85† | 0.08 (512x512) | 0.07 | 0.01† |
| mask2former-swinl-ade20k | 4.86 (512x512) | 3.03 | 3.17† | 0.76 (512x512) | 0.13 | 0.07† |
| oneformer-swinl-ade20k | 5.08 (512x512) | 4.10 | 3.27† | 0.81 (512x512) | 0.15† | 0.11† |
| pspnet-r101-d8-ade20k | 1.35 (512x512) | 0.47 | 0.77† | 0.09 (512x512) | 0.08 | 0.03† |
| sapiens-seg-0.3b-goliath | 4.92 (512x512) | 1.22 | 0.96† | 1.02 (512x512) | 0.13† | 0.09† |
| sapiens-seg-1b-goliath | 11.33 (512x512) | 3.54 | 2.27† | 0.57 (512x512) | 0.27† | 0.18† |
| segformer-b0-ade20k | 4.83 (512x512) | 2.55 | 2.92 | 2.23 (512x512) | 0.08 | 0.01 |
| segformer-b5-ade20k | 1.24 (512x512) | 3.09 | 3.43 | 0.50 (512x512) | 0.08 | 0.03 |
| segnext-mscan-b-ade20k | 0.37 | 0.26 (512x512) | 1.34 (512x512) | 0.19 | 0.58 (512x512) | 0.18 (512x512) |
| segnext-mscan-l-ade20k | 0.57 | 0.57 (512x512) | 1.51 (512x512) | 0.06 | 0.41 (512x512) | 0.10 (512x512) |
| segnext-mscan-s-ade20k | 0.23 | 0.15 (512x512) | 1.29 (512x512) | 0.32 | 0.72 (512x512) | 0.19 (512x512) |
| segnext-mscan-t-ade20k | 0.19 | 0.13 (512x512) | 1.26 (512x512) | 0.34 | 0.65 (512x512) | 0.20 (512x512) |
| efficientvit-sam-l0 | 0.35 | 0.83 | 1.63 | 0.82 | 0.72 | 0.25 |
| efficientvit-sam-l1 | 0.31 | 0.76 | 1.64 | 0.03 | 0.69 | 0.25 |
| efficientvit-sam-l2 | 0.39 | 0.93 | 1.78 | 0.04 | 0.66 | 0.25 |
| efficientvit-sam-xl0 | 0.62 | 2.67 | 2.14 | 0.59 | 0.52 | 0.26 |
| efficientvit-sam-xl1 | 1.06 | 2.32 | 2.51 | 0.04 | 0.53 | 0.24 |
| mobile-sam | 0.21 | 0.71 | 1.60 | 1.01 | 0.47 | 0.29 |
| sam-hq-vit-b | 0.77 | 0.27 | 1.06 | 0.19 | 0.41 | 0.35 |
| sam-hq-vit-h | 3.92 | 1.34 | 2.29 | 0.32 | 0.52 | 0.47 |
| sam-hq-vit-l | 1.98 | 0.76 | 1.64 | 0.20 | 0.41 | 0.40 |
| sam-vit-b | 0.75 | 0.46 | 0.17† | 0.18 | 0.05† | 0.04† |
| sam-vit-h | 4.61 | 2.45 | 0.89† | 0.38 | 0.21† | 0.14† |
| sam-vit-l | 1.80 | 1.21 | 0.43† | 0.27 | 0.12† | 0.08† |
| sam2.1-hiera-base-plus | 0.74 | 0.43 | 0.19† | 0.14 | 0.03† | 0.03† |
| sam2.1-hiera-large | 1.95 | 1.10 | 0.39† | 0.18 | 0.07† | 0.05† |
| sam2.1-hiera-small | 0.48 | 0.36 | 0.13† | 0.14 | 0.02† | 0.02† |
| sam2.1-hiera-tiny | 0.41 | 0.25 | 0.11† | 0.15 | 0.02† | 0.01† |
| sam3 | 1.12 (840x1297) | 2.95 | 0.70† | 0.96 (840x1297) | 0.18† | 0.12† |
| detr-r101-coco | 0.95 | 0.81 | 0.22† | 0.06 | 0.02 | 0.01 |
| detr-r50-coco | 2.55 | 0.70 | 0.28† | 0.83 | 0.44 | 0.03 |
| dfine-l-coco | 3.68 | 1.09 | 1.46 | 1.41 | 0.67 | 0.47 |
| dfine-m-coco | 3.63 | 1.04 | 1.20 | 1.55 | 0.69 | 0.56 |
| dfine-n-coco | 5.24 | 1.25 | 1.21 | 4.07 | 0.85 | 0.62 |
| dfine-s-coco | 2.99 | 1.05 | 1.16 | 1.44 | 0.71 | 0.52 |
| dfine-x-coco | 3.88 | 1.39 | 1.41 | 1.45 | 0.65 | 0.45 |
| faster-rcnn-r50-fpn-3x-coco | 0.73 | 0.53 | 0.14† | 0.04 | 0.25 | 0.04 |
| mask-rcnn-r50-fpn-3x-coco | 0.97 | 1.50 | 0.22† | 0.84 | 0.42 | 0.04 |
| rtdetrv2-l-coco | 0.66 | 0.50 | 0.25† | 0.04 | 0.01 | 0.02 |
| rtdetrv2-m-coco | 0.45 | 0.33 | 0.16† | 0.02 | 0.01 | 0.01 |
| rtdetrv2-ms-coco | 0.52 | 0.45 | 0.17† | 0.06 | 0.03 | 0.09 |
| rtdetrv2-s-coco | 1.32 | 0.31 | 0.46† | 0.42 | 0.10 | 0.53 |
| rtdetrv2-x-coco | 1.13 | 1.09 | 0.29† | 0.08 | 0.05 | 0.02 |
| rtmdet-l-coco | 0.68 | 0.63 | 2.07 | 0.32 | 1.27 | 0.36 |
| rtmdet-m-coco | 0.46 | 0.34 | 1.81 | 0.31 | 1.34 | 0.34 |
| rtmdet-m-person | 0.51 | 0.23 | 1.72 | 0.03 | 0.52 | 0.21 |
| rtmdet-nano-person | 0.14 | 0.10 | 1.92 | 2.69 | 1.64 | 0.35 |
| rtmdet-s-coco | 0.27 | 0.18 | 1.68 | 0.58 | 1.39 | 0.23 |
| rtmdet-tiny-coco | 0.22 | 0.13 | 1.66 | 1.24 | 1.32 | 0.25 |
| rtmdet-x-coco | 1.00 | 0.77 | 2.38 | 0.61 | 1.25 | 0.29 |
| dwpose-l | 0.50 | 0.88 | 1.64 | 1.14 | 1.83 | 0.28 |
| dwpose-l-384 | 0.42 | 0.58 | 1.67 | 0.42 | 1.10 | 0.28 |
| dwpose-m | 0.26 | 0.24 | 1.51 | 0.60 | 1.10 | 0.29 |
| dwpose-s | 0.19 | 0.12 | 1.48 | 0.47 | 1.02 | 0.29 |
| dwpose-t | 0.16 | 0.08 | 1.42 | 0.33 | 0.89 | 0.29 |
| hrnet-w32 | 0.45 | 0.25 | 2.53 | 0.38 | 0.56 | 0.28 |
| hrnet-w32-384 | 0.42 | 0.25 | 2.04 | 0.23 | 0.57 | 0.25 |
| hrnet-w48 | 0.72 | 0.40 | 2.10 | 0.23 | 0.34 | 0.15 |
| hrnet-w48-384 | 0.53 | 0.39 | 2.16 | 0.24 | 0.35 | 0.21 |
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
| rtmpose-s-face6 | 0.31 | 0.08 | 1.43 | 0.35 | 1.18 | 0.29 |
| rtmpose-s-halpe26 | 0.19 | 0.07 | 1.46 | 0.61 | 0.94 | 0.30 |
| rtmpose-t | 0.13 | 0.10 | 1.42 | 0.15 | 0.88 | 0.29 |
| rtmpose-t-face6 | 0.18 | 0.06 | 1.45 | 0.34 | 1.29 | 0.29 |
| rtmpose-t-halpe26 | 0.18 | 0.05 | 1.48 | 0.71 | 0.81 | 0.29 |
| rtmpose-x-halpe26-384 | 0.89 | 0.41 | 1.76 | 0.94 | 1.17 | 0.28 |
| rtmw-l | 0.76 | 0.39 | 1.72 | 0.63 | 1.03 | 0.31 |
| rtmw-l-384 | 0.75 | 0.34 | 1.69 | 0.56 | 1.20 | 0.28 |
| rtmw-m | 0.54 | 0.35 | 1.64 | 1.21 | 1.20 | 0.29 |
| rtmw-x | 1.23 | 1.35 | 2.10 | 0.64 | 0.83 | 0.28 |
| rtmw-x-384 | 1.05 | 0.81 | 2.20 | 0.42 | 0.96 | 0.34 |
| vitpose-base | 0.16 | 3.48 | 1.06 | 0.33 | 0.40 | 0.34 |
| vitpose-base-simple | 0.15 | 0.40 | 1.04 | 0.17 | 0.41 | 0.41 |
| vitpose-plus-base | 0.19 | 0.65 | 1.10 | 0.12 | 0.34 | 0.34 |
| vitpose-plus-huge | 0.65 | 2.69 | 2.26 | 0.25 | 0.44 | 0.40 |
| vitpose-plus-large | 0.41 | 1.50 | 1.63 | 0.17 | 0.41 | 0.41 |
| vitpose-plus-small | 0.15 | 0.25 | 0.95 | 0.12 | 0.53 | 0.33 |
| c-radio-v3-b | – | – | – | 0.14 | 0.08 | 0.06 |
| c-radio-v3-g | – | – | – | 0.17 | 0.08 | 0.07 |
| c-radio-v3-h | – | – | – | 0.16 | 0.07 | 0.07 |
| c-radio-v3-l | – | – | – | 0.15 | 0.07 | 0.04 |
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
| dinov2-vitg14-reg | 14.67 | 4.32 | 2.91† | 0.09 | 0.03 | 0.02 |
| dinov2-vitl14 | 7.06 | 0.57 | 0.57 | 0.01 | 0.15 | 0.18 |
| dinov2-vitl14-reg | 4.60 | 0.66 | 0.56 | 0.07 | 0.11 | 0.20 |
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
| pe-core-g14-448 | 232.98 | – | – | 0.28 | 0.20 | 0.13 |
| pe-core-l14-336 | 41.39 | – | – | 0.20 | 0.11 | 0.05 |
| pe-core-s16-384 | 5.70 | – | – | 0.18 | 0.10 | 0.04 |
| pe-core-t16-384 | 3.20 | – | – | 0.19 | 0.12 | 0.07 |
| pe-spatial-b16-512 | 13.65 | – | – | 0.16 | 0.10 | 0.07 |
| pe-spatial-g14-448 | 227.90 | – | – | 0.36 | 0.22 | 0.32 |
| pe-spatial-l14-448 | 44.48 | – | – | 0.17 | 0.09 | 0.06 |
| pe-spatial-s16-512 | 5.40 | – | – | 0.16 | 0.08 | 0.06 |
| pe-spatial-t16-512 | 3.09 | – | – | 0.16 | 0.10 | 0.08 |
| siglip2-b16-256 | 14.20 | – | – | 0.18 | 0.07 | 0.05 |
| siglip2-b16-512 | 14.17 | – | – | 0.18 | 0.07 | 0.05 |
| siglip2-g16-256 | 140.56 | – | – | 0.22 | 0.10 | 0.08 |
| siglip2-g16-384 | 139.99 | – | – | 0.22 | 0.08 | 0.07 |
| siglip2-l16-256 | 40.91 | – | – | 0.19 | 0.05 | 0.03 |
| siglip2-l16-512 | 41.54 | – | – | 0.18 | 0.05 | 0.03 |
| siglip2-so400m14-224 | 54.40 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m14-378 | 54.79 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m16-256 | 54.53 | – | – | 0.20 | 0.08 | 0.07 |
| siglip2-so400m16-512 | 54.67 | – | – | 0.20 | 0.08 | 0.07 |
| tipsv2-b14 | 0.90 | 0.20 | 1.42 | 0.07 | 0.18 | 0.09 |
| tipsv2-l14 | 3.10 | 2.91 | 1.44 | 0.02 | 0.08 | 0.11 |
| mapanything-apache | 6.12 (336x518) | 2.89 (336x518) | 0.57† (336x518) | 2.62 (336x518) | 0.66 (336x518) | 0.54 (336x518) |
| mvsanywhere | 3.21 (448x672) | 1.88 (448x672) | 0.52† (448x672) | 1.04 (448x672) | 0.22 (448x672) | 0.05 (448x672) |
| patchmatchnet | 0.04 (1536x1024) | 0.06 (1536x1024) | 0.05† (1536x1024) | 0.92 (1536x1024) | 0.25 (1536x1024) | 0.04 (1536x1024) |
| vggt-1b | 19.37 (336x518) | 4.98 (336x518) | 5.99† (336x518) | 1.29 (336x518) | 0.69 (336x518) | 0.46 (336x518) |
| vggt-omega-1b-512 | 16.31 (336x518) | 3.31 (336x518) | 2.27† (336x518) | 1.10 (336x518) | 0.73 (336x518) | 0.46 (336x518) |
| flowformer-small-chairs | 0.16 (375x1242) | 1.31 (375x1242) | – | 0.37 (375x1242) | 0.57 (375x1242) | – |
| flowformer-small-sintel | 0.15 (375x1242) | 1.30 (375x1242) | – | 0.30 (375x1242) | 0.57 (375x1242) | – |
| flowformer-small-things | 0.15 (375x1242) | 1.31 (375x1242) | – | 0.30 (375x1242) | 0.57 (375x1242) | – |
| flowseek-t | 0.87 | 0.18 (375x1242) | – | 0.63 | 0.46 (375x1242) | – |
| flowseek-t-things | 0.78 | 0.18 (375x1242) | – | 0.46 | 0.47 (375x1242) | – |
| flowseek-t-things-notartan | 0.81 | 0.18 (375x1242) | – | 0.46 | 0.46 (375x1242) | – |
| gmflow-scale1-mixdata | 0.13 | 0.14 (375x1242) | – | 0.76 | 0.48 (375x1242) | – |
| gmflow-scale1-things | 0.14 | 0.18 (375x1242) | – | 0.94 | 0.56 (375x1242) | – |
| gmflow-scale2-mixdata | 0.13 | 0.08 (375x1242) | – | 1.62 | 0.61 (375x1242) | – |
| gmflow-scale2-regrefine6-kitti | 0.15 | 0.11 (375x1242) | – | 2.20 | 0.75 (375x1242) | – |
| gmflow-scale2-regrefine6-mixdata | 0.14 | 0.19 (375x1242) | – | 2.20 | 0.72 (375x1242) | – |
| gmflow-scale2-regrefine6-sintelft | 0.15 | 0.12 (375x1242) | – | 2.37 | 0.76 (375x1242) | – |
| gmflow-scale2-regrefine6-things | 0.15 | 0.16 (375x1242) | – | 2.19 | 0.75 (375x1242) | – |
| gmflow-scale2-sintel | 0.13 | 0.09 (375x1242) | – | 1.62 | 0.61 (375x1242) | – |
| gmflow-scale2-things | 0.14 | 0.11 (375x1242) | – | 1.61 | 0.63 (375x1242) | – |
| neuflow-v2 | 0.20 | 0.05 (375x1242) | 0.96 (375x1242) | 0.22 | 0.30 (375x1242) | 0.18 (375x1242) |
| neuflow-v2-sintel | 0.17 | 0.07 (375x1242) | 0.95 (375x1242) | 0.08 | 0.28 (375x1242) | 0.18 (375x1242) |
| neuflow-v2-things | 0.17 | 0.06 (375x1242) | 0.92 (375x1242) | 0.08 | 0.30 (375x1242) | 0.18 (375x1242) |
| raft-large | 0.14 | 0.05 | 1.07 | 0.76 | 0.61 | 0.11 |
| raft-large-kitti | 0.14 | 0.05 | 1.06 | 0.58 | 0.60 | 0.11 |
| raft-large-things | 0.13 | 0.05 | 1.07 | 0.58 | 0.61 | 0.11 |
| raft-small | 0.09 | 0.03 | 1.06 | 0.43 | 0.48 | 0.07 |
| sea-raft-m | 0.53 | 0.09 | 1.15 | 0.37 | 0.30 | 0.07 |
| sea-raft-m-kitti | 0.67 | 0.09 | 1.14 | 0.37 | 0.30 | 0.07 |
| sea-raft-m-spring | 0.57 | 0.09 | 1.14 | 0.06 | 0.06 | 0.02 |
| sea-raft-m-things | 0.55 | 0.09 | 1.13 | 0.37 | 0.30 | 0.07 |
| sea-raft-s | 0.30 | 0.05 | 1.10 | 0.30 | 0.23 | 0.05 |
| sea-raft-s-kitti | 0.30 | 0.05 | 1.10 | 0.30 | 0.23 | 0.05 |
| sea-raft-s-spring | 0.31 | 0.05 | 1.11 | 0.04 | 0.04 | 0.02 |

† measured in the speed sweep after the portable load (or an earlier size) in the same process: file cache and kernels are warm, so the time is lower than a fresh start.
<!-- tables:end -->

## torch.compile and TensorRT

Portable with `model.compile()` (Inductor, default mode) and the fp16 TensorRT engine of the portable model
(`export_tensorrt`), one variant per family at the task's standard size, next to portable and `kernels`
([export](export.md) has the methods). The compile time is the cold first call (empty Inductor cache); the engine
runs `module()` on synthetic inputs of the shape the timed call gives the network (classifier crop, all views of a
scene; "only" marks engines of part of the call) and its memory is the device memory it holds. Rows measured by
`tools/speed_check.py run --modes compile tensorrt`; a failure is shown as `fails: reason`.

<!-- compile:begin (tools/gen_docs.py, from docs/data/speed_compile.jsonl) -->
| model | input | time: portable | compile | TensorRT fp16 | kernels | preparation: compile (cold first call) | ONNX + engine build | peak MiB (alloc / reserved): portable | compile | kernels | TensorRT (device) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-sceneflow | 994x1482 | 140 ms | 91.8 ms | 91.2 ms | 101 ms | 79.9 s | 211 s | 845 / 986 | 754 / 928 | 536 / 644 | 482 |
| bridgedepth-sf | 994x1482 | 437 ms | 325 ms | fails: TensorRT bug: Myelin internal error | 348 ms | 111 s | – | 2916 / 3214 | 2280 / 2308 | 2310 / 2634 | – |
| c-fast-foundationstereo | 994x1482 | 330 ms | 339 ms | 451 ms | 135 ms | 75.4 s | 240 s | 2109 / 2336 | 1649 / 1792 | 1196 / 1516 | 1252 |
| c-foundationstereo-s | 994x1482 | 3646 ms | fails: Dynamo guard error (torch bug) | 2229 ms | 1362 ms | – | 444 s | 6312 / 6590 | – | 3107 / 4376 | 9682 |
| cas-gwcnet-c-sceneflow | 994x1482 | 327 ms | 295 ms | 220 ms | 225 ms | 61.7 s | 131 s | 2006 / 2474 | 2098 / 2448 | 1725 / 1930 | 1440 |
| cas-gwcnet-gc-sceneflow | 994x1482 | 472 ms | 383 ms | 307 ms | 358 ms | 63.7 s | 98.6 s | 2338 / 2794 | 6592 / 6670 | 2272 / 2514 | 4046 |
| cfnet-sceneflow | 994x1482 | 484 ms | 249 ms | 171 ms | 401 ms | 53.0 s | 139 s | 3441 / 4376 | 1026 / 1398 | 3662 / 4718 | 1238 |
| coex-sceneflow | 994x1482 | 49.9 ms | 29.4 ms | 20.6 ms | 23.5 ms | 22.4 s | 56.6 s | 1136 / 1254 | 351 / 452 | 495 / 886 | 360 |
| crestereo-eth3d | 994x1482 | 732 ms | 398 ms | 298 ms | 229 ms | 120 s | 224 s | 884 / 1030 | 695 / 748 | 644 / 732 | 584 |
| defom-stereo-vitl-sceneflow | 994x1482 | 1531 ms | 1002 ms | fails: TensorRT bug: Myelin internal error | 659 ms | 122 s | – | 4362 / 4838 | 3021 / 3292 | 3154 / 3566 | – |
| dlnr-sceneflow | 994x1482 | 398 ms | 301 ms | 241 ms | 214 ms | 69.0 s | 161 s | 2635 / 3142 | 1343 / 1672 | 1236 / 1606 | 1492 |
| fast-acvnet-generalization | 994x1482 | 54.7 ms | 36.4 ms | 36.1 ms | 38.2 ms | 41.9 s | 68.4 s | 664 / 736 | 549 / 678 | 739 / 1042 | 492 |
| fast-acvnet-sceneflow | 994x1482 | 54.7 ms | 36.2 ms | 36.7 ms | 38.2 ms | 41.8 s | 71.3 s | 664 / 736 | 584 / 698 | 739 / 1042 | 484 |
| fast-foundationstereo | 994x1482 | 330 ms | 339 ms | 156 ms | 135 ms | 78.4 s | 157 s | 2109 / 2256 | 1613 / 1752 | 1196 / 1576 | 1168 |
| foundationstereo-vit-s | 994x1482 | 2514 ms | 2044 ms | 1196 ms | 1120 ms | 212 s | 545 s | 2493 / 2606 | 2822 / 3646 | 2169 / 2470 | 2570 |
| great-igev-eth3d | 994x1482 | 1085 ms | 829 ms | 940 ms | 393 ms | 224 s | 685 s | 1402 / 1678 | 2831 / 3362 | 1192 / 1564 | 6018 |
| great-igev-kitti | 994x1482 | 1085 ms | 830 ms | 942 ms | 393 ms | 225 s | 687 s | 1402 / 1678 | 2831 / 3362 | 1192 / 1564 | 6020 |
| great-igev-middlebury | 994x1482 | 1785 ms | 1676 ms | – | 579 ms | 326 s | – | 1875 / 2078 | 5228 / 5720 | 1553 / 1744 | – |
| great-igev-sceneflow | 994x1482 | 1085 ms | 830 ms | 944 ms | 393 ms | 224 s | 675 s | 1402 / 1678 | 2831 / 3362 | 1192 / 1564 | 6020 |
| gwcnet-g-sceneflow | 994x1482 | 344 ms | 417 ms | 275 ms | 238 ms | 52.6 s | 66.0 s | 2081 / 2402 | 1527 / 1864 | 1799 / 2102 | 3548 |
| hitnet-eth3d | 994x1482 | 67.8 ms | 27.7 ms | 31.1 ms | 23.2 ms | 39.1 s | 136 s | 860 / 1090 | 719 / 728 | 647 / 1248 | 836 |
| hitnet-middlebury-d400 | 994x1482 | 176 ms | 71.5 ms | 74.9 ms | 52.7 ms | 55.4 s | 166 s | 1744 / 2414 | 1296 / 1332 | 1263 / 2368 | 1906 |
| igev-plusplus-eth3d | 994x1482 | 1081 ms | 716 ms | 494 ms | 393 ms | 159 s | 425 s | 1594 / 1840 | 952 / 1074 | 1389 / 2302 | 3632 |
| igev-plusplus-kitti12 | 994x1482 | 1081 ms | 718 ms | 494 ms | 393 ms | 158 s | 422 s | 1594 / 1840 | 917 / 1054 | 1389 / 2302 | 3686 |
| igev-plusplus-kitti15 | 994x1482 | 1081 ms | 718 ms | 494 ms | 393 ms | 159 s | 425 s | 1594 / 1840 | 917 / 1052 | 1389 / 2302 | 3686 |
| igev-plusplus-middlebury | 994x1482 | 1081 ms | 718 ms | 494 ms | 393 ms | 157 s | 436 s | 1594 / 1840 | 917 / 1054 | 1389 / 2302 | 3686 |
| igev-plusplus-sceneflow | 994x1482 | 1081 ms | 718 ms | 495 ms | 393 ms | 158 s | 426 s | 1594 / 1840 | 917 / 1054 | 1389 / 2302 | 3686 |
| igev-stereo-eth3d | 994x1482 | 763 ms | 527 ms | 339 ms | 288 ms | 111 s | 213 s | 1237 / 1492 | 807 / 1052 | 1113 / 1374 | 1220 |
| igev-stereo-middlebury | 994x1482 | 763 ms | 527 ms | 338 ms | 288 ms | 110 s | 214 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1186 |
| igev-stereo-sceneflow | 994x1482 | 763 ms | 527 ms | 340 ms | 288 ms | 111 s | 216 s | 1237 / 1492 | 771 / 1012 | 1113 / 1374 | 1186 |
| las2-h | 994x1482 | 123 ms | 75.0 ms | fails: TensorRT bug: Myelin internal error | 53.3 ms | 37.5 s | – | 1050 / 1110 | 579 / 724 | 1021 / 1388 | – |
| las2-l | 994x1482 | 80.3 ms | 46.6 ms | 23.4 ms | 29.9 ms | 30.5 s | 97.5 s | 1036 / 1098 | 385 / 432 | 397 / 600 | 214 |
| las2-m | 994x1482 | 54.1 ms | 28.6 ms | 13.2 ms | 18.3 ms | 24.0 s | 98.3 s | 972 / 1026 | 252 / 300 | 329 / 572 | 228 |
| las2-s | 994x1482 | 46.9 ms | 23.7 ms | 10.3 ms | 14.3 ms | 22.4 s | 86.9 s | 955 / 988 | 238 / 264 | 320 / 558 | 162 |
| mocha-v2-sceneflow | 994x1482 | 1033 ms | 713 ms | 437 ms | 448 ms | 148 s | 361 s | 1632 / 1770 | 1122 / 1308 | 1587 / 1790 | 1098 |
| monster-mix-all | 994x1482 | 1585 ms | 1098 ms | 674 ms | 668 ms | 105 s | 391 s | 2336 / 2494 | 2081 / 2272 | 2356 / 2714 | 2050 |
| monster-plusplus-mix-all | 994x1482 | 1633 ms | 1126 ms | 697 ms | 698 ms | 109 s | 416 s | 2583 / 2918 | 2328 / 2532 | 2438 / 2760 | 2642 |
| monster-sceneflow | 994x1482 | 1585 ms | 1096 ms | 671 ms | 668 ms | 103 s | 384 s | 2336 / 2494 | 2081 / 2272 | 2356 / 2714 | 2018 |
| nmrf-sceneflow | 994x1482 | 216 ms | 161 ms | 184 ms | 196 ms | 84.8 s | 142 s | 1510 / 1552 | 854 / 1104 | 1022 / 1112 | 826 |
| psmnet-sceneflow | 994x1482 | 340 ms | 440 ms | 669 ms | 232 ms | 60.0 s | 88.9 s | 2078 / 2442 | 1760 / 2082 | 1786 / 2122 | 1528 |
| raft-stereo-eth3d | 994x1482 | 746 ms | 506 ms | 333 ms | 273 ms | 99.9 s | 132 s | 1388 / 1492 | 815 / 988 | 765 / 802 | 826 |
| raft-stereo-middlebury | 994x1482 | 746 ms | 506 ms | 334 ms | 273 ms | 100 s | 132 s | 1388 / 1492 | 780 / 988 | 765 / 802 | 884 |
| raft-stereo-sceneflow | 994x1482 | 746 ms | 507 ms | 333 ms | 273 ms | 100 s | 132 s | 1388 / 1492 | 780 / 988 | 765 / 802 | 886 |
| rt-monster-plusplus-mix-all | 994x1482 | 113 ms | 77.0 ms | 44.1 ms | 94.1 ms | 69.4 s | 150 s | 2592 / 2646 | 2750 / 2942 | 2704 / 3976 | 952 |
| s2m2-l | 994x1482 | 1015 ms | 650 ms | 497 ms | 376 ms | 144 s | 374 s | 3245 / 3838 | 3169 / 3514 | 1249 / 1568 | 2114 |
| s2m2-s | 994x1482 | 348 ms | 186 ms | 137 ms | 113 ms | 95.9 s | 363 s | 1733 / 1930 | 1579 / 1710 | 686 / 820 | 980 |
| s2m2-xl | 994x1482 | 1815 ms | 1298 ms | 753 ms | 682 ms | 153 s | 408 s | 5228 / 6016 | 5594 / 6014 | 2092 / 2274 | 3628 |
| selective-igev-eth3d-finetune | 994x1482 | 998 ms | 691 ms | 461 ms | 349 ms | 126 s | 256 s | 1238 / 1488 | 919 / 1088 | 1109 / 1390 | 1148 |
| waft-stereo-s | 994x1482 | 689 ms | 357 ms | 170 ms | 284 ms | 95.1 s | 152 s | 3293 / 3456 | 2024 / 2058 | 1113 / 1286 | 662 |
| da3-giant-1.1 | 994x1482 | 81.8 ms | 64.3 ms | 37.2 ms | 44.2 ms | 47.2 s | 84.1 s | 2638 / 4608 | 2487 / 2528 | 2628 / 2710 | 2416 |
| da3-large-1.1 | 994x1482 | 37.4 ms | 26.5 ms | 13.8 ms | 17.1 ms | 45.4 s | 62.6 s | 992 / 1350 | 841 / 868 | 985 / 1090 | 816 |
| da3metric-large | 994x1482 | 24.1 ms | 21.2 ms | 14.4 ms | 14.7 ms | 26.6 s | 58.2 s | 815 / 1328 | 792 / 828 | 811 / 868 | 756 |
| da3mono-large | 994x1482 | 23.9 ms | 21.2 ms | 14.4 ms | 14.6 ms | 25.0 s | 52.3 s | 818 / 1328 | 792 / 828 | 814 / 868 | 754 |
| da3nested-giant-large-1.1 | 994x1482 | 108 ms | 85.4 ms | 51.9 ms | 58.2 ms | 60.0 s | 154 s | 3275 / 5914 | 3124 / 3194 | 3271 / 3378 | 3136 |
| depth-anything-v2-l | 994x1482 | 55.1 ms | 51.7 ms | 32.4 ms | 31.1 ms | 20.6 s | 43.5 s | 1026 / 1352 | 850 / 952 | 921 / 1034 | 868 |
| depth-anything-v2-metric-hypersim-b | 994x1482 | 18.6 ms | 17.0 ms | 10.8 ms | 10.8 ms | 16.0 s | 32.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 282 |
| depth-anything-v2-metric-hypersim-l | 994x1482 | 54.5 ms | 51.3 ms | 32.3 ms | 31.7 ms | 20.2 s | 43.5 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 868 |
| depth-anything-v2-metric-hypersim-s | 994x1482 | 7.66 ms | 6.67 ms | 4.47 ms | 4.67 ms | 15.8 s | 29.8 s | 244 / 288 | 201 / 224 | 152 / 208 | 186 |
| depth-anything-v2-metric-vkitti-b | 994x1482 | 18.6 ms | 17.0 ms | 10.9 ms | 10.8 ms | 15.9 s | 32.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 282 |
| depth-anything-v2-metric-vkitti-l | 994x1482 | 54.5 ms | 51.3 ms | 32.4 ms | 31.9 ms | 20.1 s | 43.4 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 868 |
| depth-anything-v2-metric-vkitti-s | 994x1482 | 7.66 ms | 6.68 ms | 4.47 ms | 4.67 ms | 15.8 s | 29.9 s | 244 / 288 | 201 / 224 | 152 / 208 | 186 |
| depth-anything-v2-s | 994x1482 | 7.82 ms | 6.65 ms | 4.46 ms | 4.51 ms | 15.7 s | 29.9 s | 249 / 326 | 201 / 224 | 190 / 248 | 186 |
| depth-pro | 994x1482 | 518 ms | 488 ms | 299 ms | 271 ms | 39.2 s | 163 s | 3306 / 3498 | 3053 / 3198 | 3667 / 3778 | 2926 |
| distill-any-depth-b | 994x1482 | 18.6 ms | 17.0 ms | 10.8 ms | 10.8 ms | 16.2 s | 33.0 s | 445 / 514 | 340 / 394 | 349 / 396 | 282 |
| distill-any-depth-l | 994x1482 | 54.4 ms | 51.6 ms | 32.4 ms | 31.8 ms | 20.1 s | 42.5 s | 1022 / 1092 | 850 / 910 | 917 / 1034 | 868 |
| distill-any-depth-s | 994x1482 | 7.65 ms | 6.67 ms | 4.46 ms | 4.63 ms | 15.9 s | 29.9 s | 244 / 288 | 201 / 224 | 152 / 208 | 186 |
| marigold-depth-v1-1 | 994x1482 | 408 ms | 342 ms | 172 ms | 383 ms | 127 s | 285 s | 2817 / 3760 | 2341 / 2418 | 2583 / 3762 | 2368 |
| metric3d-vit-large | 994x1482 | 200 ms | 149 ms | 91.4 ms | 158 ms | 54.7 s | 96.6 s | 1484 / 1696 | 1126 / 1148 | 1516 / 1688 | 1010 |
| midas-dpt-beit-large-512 | 994x1482 | 78.8 ms | 58.8 ms | 52.2 ms | 63.9 ms | 32.6 s | 53.5 s | 992 / 1418 | 1088 / 1116 | 1003 / 1436 | 800 |
| moge-2-vitb-normal | 994x1482 | 83.9 ms | 60.9 ms | 35.5 ms | 40.7 ms | 133 s | 127 s | 1163 / 1312 | 972 / 1172 | 1093 / 1208 | 760 |
| moge-2-vitl-normal | 994x1482 | 144 ms | 124 ms | 75.6 ms | 73.4 ms | 152 s | 173 s | 1661 / 1836 | 1411 / 1634 | 1498 / 1610 | 1164 |
| moge-2-vits-normal | 994x1482 | 64.2 ms | 43.9 ms | 24.5 ms | 29.7 ms | 124 s | 131 s | 1066 / 1336 | 836 / 992 | 884 / 1008 | 620 |
| unidepth-v2-vitl14 | 994x1482 | 99.8 ms | 96.4 ms | 64.3 ms | 66.4 ms | 31.1 s | 119 s | 1208 / 1472 | 1304 / 1392 | 1211 / 1472 | 1050 |
| unik3d-vitl | 994x1482 | 108 ms | 94.4 ms | 58.0 ms | 62.9 ms | 32.8 s | 105 s | 1219 / 1492 | 1258 / 1312 | 1166 / 1266 | 846 |
| zoedepth-nk | 994x1482 | 124 ms | 71.7 ms | 73.3 ms | 88.9 ms | 93.8 s | 138 s | 2037 / 2124 | 1813 / 1880 | 1771 / 1960 | 852 |
| alexnet.tv_in1k | 224x224 | 0.970 ms | 0.824 ms | 0.401 ms | 0.604 ms | 4.33 s | 13.4 s | 151 / 224 | 183 / 204 | 184 / 228 | 128 |
| coatnet_0_rw_224.sw_in1k | 224x224 | 6.49 ms | 3.59 ms | 0.849 ms | – | 12.1 s | 38.2 s | 102 / 124 | 120 / 142 | – | 70 |
| coatnet_1_rw_224.sw_in1k | 224x224 | 10.7 ms | 5.58 ms | 1.38 ms | – | 16.2 s | 42.0 s | 132 / 158 | 149 / 176 | – | 114 |
| coatnet_2_rw_224.sw_in12k_ft_in1k | 224x224 | 11.9 ms | 5.53 ms | 1.73 ms | – | 16.2 s | 36.0 s | 207 / 240 | 211 / 240 | – | 196 |
| coatnet_nano_rw_224.sw_in1k | 224x224 | 7.23 ms | 4.71 ms | 0.709 ms | – | 13.9 s | 31.8 s | 74 / 86 | 96 / 126 | – | 38 |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | 224x224 | 12.3 ms | 6.63 ms | 1.55 ms | – | 17.9 s | 42.6 s | 132 / 158 | 162 / 176 | – | 140 |
| coatnet_rmlp_1_rw_224.sw_in1k | 224x224 | 12.4 ms | 6.54 ms | 1.47 ms | – | 18.6 s | 43.8 s | 131 / 156 | 155 / 176 | – | 126 |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | 224x224 | 12.8 ms | 6.52 ms | 1.87 ms | – | 18.4 s | 41.0 s | 207 / 240 | 238 / 260 | – | 224 |
| coatnet_rmlp_2_rw_224.sw_in1k | 224x224 | 15.1 ms | 6.34 ms | 1.92 ms | – | 18.4 s | 45.9 s | 207 / 240 | 238 / 260 | – | 226 |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | 384x384 | 13.9 ms | 6.92 ms | 3.69 ms | – | 18.6 s | 42.3 s | 296 / 330 | 326 / 370 | – | 252 |
| coatnet_rmlp_nano_rw_224.sw_in1k | 224x224 | 8.09 ms | 9.89 ms | 0.868 ms | – | 13.9 s | 37.1 s | 78 / 106 | 110 / 126 | – | 58 |
| convnext_base.fb_in1k | 224x224 | 5.57 ms | 3.37 ms | 1.45 ms | 2.01 ms | 10.9 s | 32.5 s | 291 / 296 | 323 / 334 | 208 / 236 | 200 |
| convnext_base.fb_in22k_ft_in1k | 224x224 | 5.66 ms | 3.37 ms | 1.45 ms | 2.00 ms | 11.4 s | 31.3 s | 291 / 296 | 323 / 334 | 208 / 236 | 200 |
| convnext_base.fb_in22k_ft_in1k_384 | 224x224 | 5.48 ms | 3.41 ms | 2.31 ms | 3.00 ms | 10.8 s | 31.2 s | 309 / 334 | 336 / 354 | 213 / 232 | 210 |
| convnext_large.fb_in1k | 224x224 | 5.59 ms | 3.45 ms | 2.50 ms | 3.29 ms | 10.8 s | 33.9 s | 603 / 632 | 642 / 652 | 419 / 434 | 396 |
| convnext_large.fb_in22k_ft_in1k | 224x224 | 5.57 ms | 3.44 ms | 2.50 ms | 3.28 ms | 10.8 s | 35.0 s | 603 / 632 | 642 / 652 | 419 / 434 | 396 |
| convnext_large.fb_in22k_ft_in1k_384 | 224x224 | 5.99 ms | 5.80 ms | 4.23 ms | 5.20 ms | 10.7 s | 33.4 s | 629 / 654 | 654 / 672 | 426 / 454 | 410 |
| convnext_small.fb_in1k | 224x224 | 5.69 ms | 3.47 ms | 1.16 ms | 1.84 ms | 10.6 s | 30.5 s | 181 / 190 | 211 / 230 | 134 / 152 | 104 |
| convnext_small.fb_in22k_ft_in1k | 224x224 | 5.73 ms | 3.50 ms | 1.17 ms | 1.84 ms | 10.5 s | 30.9 s | 181 / 190 | 211 / 230 | 134 / 152 | 104 |
| convnext_small.fb_in22k_ft_in1k_384 | 224x224 | 5.51 ms | 3.36 ms | 1.68 ms | 2.52 ms | 10.7 s | 31.8 s | 194 / 212 | 223 / 232 | 138 / 152 | 142 |
| convnext_tiny.fb_in1k | 224x224 | 3.20 ms | 2.09 ms | 0.682 ms | 1.15 ms | 8.22 s | 27.4 s | 119 / 130 | 150 / 170 | 92 / 110 | 72 |
| convnext_tiny.fb_in22k_ft_in1k | 224x224 | 3.21 ms | 2.08 ms | 0.687 ms | 1.15 ms | 8.09 s | 28.3 s | 119 / 130 | 150 / 170 | 92 / 110 | 72 |
| convnext_tiny.fb_in22k_ft_in1k_384 | 224x224 | 3.15 ms | 2.05 ms | 0.970 ms | 1.56 ms | 7.90 s | 28.2 s | 133 / 152 | 162 / 170 | 96 / 110 | 78 |
| convnextv2_base.fcmae_ft_in22k_in1k | 224x224 | 10.2 ms | 3.85 ms | 2.13 ms | 2.94 ms | 14.4 s | 35.1 s | 299 / 376 | 324 / 356 | 318 / 380 | 204 |
| davit_base.msft_in1k | 224x224 | 11.7 ms | 5.48 ms | 2.28 ms | – | 21.8 s | 71.1 s | 212 / 224 | 242 / 262 | – | 202 |
| davit_small.msft_in1k | 224x224 | 12.8 ms | 5.54 ms | 1.92 ms | – | 21.8 s | 66.4 s | 136 / 150 | 166 / 190 | – | 106 |
| davit_tiny.msft_in1k | 224x224 | 6.33 ms | 3.10 ms | 1.03 ms | – | 15.5 s | 53.8 s | 96 / 116 | 125 / 136 | – | 72 |
| deit_base_patch16_224.fb_in1k | 224x224 | 2.98 ms | 2.09 ms | 1.12 ms | 1.64 ms | 7.27 s | 28.3 s | 203 / 386 | 238 / 266 | 300 / 370 | 196 |
| densenet121.tv_in1k | 224x224 | 9.41 ms | 5.49 ms | 0.686 ms | 1.28 ms | 20.4 s | 33.6 s | 53 / 90 | 89 / 112 | 83 / 132 | 36 |
| dinov2_vitl14_lc | 224x224 | 5.43 ms | 5.09 ms | 4.22 ms | 4.93 ms | 12.1 s | 38.2 s | 623 / 1228 | 660 / 688 | 890 / 950 | 618 |
| efficientformer_l1.snap_dist_in1k | 224x224 | 3.72 ms | 1.69 ms | 0.335 ms | – | 9.10 s | 35.4 s | 93 / 104 | 125 / 144 | – | 38 |
| efficientformer_l3.snap_dist_in1k | 224x224 | 6.37 ms | 2.57 ms | 0.750 ms | – | 11.8 s | 39.4 s | 161 / 216 | 193 / 216 | – | 70 |
| efficientformer_l7.snap_dist_in1k | 224x224 | 9.65 ms | 3.69 ms | 1.40 ms | – | 14.4 s | 40.0 s | 300 / 324 | 332 / 364 | – | 166 |
| efficientformerv2_l.snap_dist_in1k | 224x224 | 17.1 ms | 7.32 ms | 1.20 ms | – | 20.7 s | 45.1 s | 93 / 144 | 125 / 184 | – | 70 |
| efficientformerv2_s0.snap_dist_in1k | 224x224 | 7.62 ms | 3.34 ms | 0.550 ms | – | 11.3 s | 36.3 s | 44 / 64 | 76 / 104 | – | 36 |
| efficientformerv2_s1.snap_dist_in1k | 224x224 | 8.91 ms | 3.91 ms | 0.632 ms | – | 12.7 s | 41.2 s | 50 / 88 | 82 / 108 | – | 38 |
| efficientformerv2_s2.snap_dist_in1k | 224x224 | 13.9 ms | 5.55 ms | 0.919 ms | – | 16.4 s | 38.9 s | 64 / 104 | 96 / 124 | – | 38 |
| efficientnet_b0.tv_in1k | 224x224 | 4.98 ms | 2.88 ms | 0.536 ms | 0.769 ms | 9.12 s | 25.8 s | 48 / 96 | 80 / 96 | 78 / 118 | 38 |
| efficientnet_b4.tv_in1k | 224x224 | 10.0 ms | 5.39 ms | 1.63 ms | 2.00 ms | 13.8 s | 34.9 s | 89 / 114 | 124 / 154 | 111 / 180 | 86 |
| efficientnet_v2_s.tv_in1k | 224x224 | 10.2 ms | 5.39 ms | 1.35 ms | 1.89 ms | 12.1 s | 32.0 s | 81 / 106 | 116 / 146 | 114 / 188 | 74 |
| efficientvit_b2.r288_in1k | 224x224 | 11.9 ms | 6.91 ms | 0.790 ms | 1.10 ms | 11.0 s | 35.3 s | 85 / 124 | 120 / 164 | 120 / 212 | 74 |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | 224x224 | 18.0 ms | 14.9 ms | 12.4 ms | 14.1 ms | 21.0 s | 43.5 s | 648 / 1206 | 674 / 684 | 1236 / 1386 | 642 |
| fastvit_sa24.apple_dist_in1k | 256x256 | 6.05 ms | 3.97 ms | 0.818 ms | 1.40 ms | 8.44 s | 54.5 s | 81 / 110 | 113 / 148 | 80 / 150 | 72 |
| fastvit_t12.apple_dist_in1k | 256x256 | 3.27 ms | 2.32 ms | 0.431 ms | 0.781 ms | 7.39 s | 39.7 s | 54 / 76 | 86 / 94 | 52 / 76 | 40 |
| fastvit_t8.apple_dist_in1k | 256x256 | 3.30 ms | 2.55 ms | 0.368 ms | 0.703 ms | 6.81 s | 30.8 s | 45 / 72 | 77 / 92 | 45 / 76 | 36 |
| focalnet_base_lrf.ms_in1k | 224x224 | 15.5 ms | 7.47 ms | 2.08 ms | – | 19.7 s | 63.9 s | 212 / 234 | 235 / 274 | – | 202 |
| focalnet_base_srf.ms_in1k | 224x224 | 14.2 ms | 6.94 ms | 1.74 ms | – | 17.8 s | 63.8 s | 211 / 234 | 234 / 272 | – | 200 |
| focalnet_small_lrf.ms_in1k | 224x224 | 15.8 ms | 7.28 ms | 1.70 ms | – | 19.5 s | 62.5 s | 136 / 152 | 162 / 192 | – | 136 |
| focalnet_small_srf.ms_in1k | 224x224 | 14.0 ms | 6.90 ms | 1.49 ms | – | 17.8 s | 65.3 s | 136 / 152 | 161 / 190 | – | 106 |
| focalnet_tiny_lrf.ms_in1k | 224x224 | 8.07 ms | 3.94 ms | 0.952 ms | – | 13.3 s | 49.9 s | 95 / 118 | 120 / 138 | – | 70 |
| focalnet_tiny_srf.ms_in1k | 224x224 | 7.36 ms | 3.72 ms | 0.864 ms | – | 12.9 s | 52.0 s | 95 / 118 | 120 / 136 | – | 72 |
| hgnetv2_b0.ssld_stage2_ft_in1k | 224x224 | 3.46 ms | 1.78 ms | 0.334 ms | – | 6.74 s | 21.2 s | 48 / 70 | 78 / 110 | – | 32 |
| hgnetv2_b1.ssld_stage2_ft_in1k | 224x224 | 3.47 ms | 1.84 ms | 0.307 ms | – | 7.07 s | 20.1 s | 50 / 68 | 80 / 108 | – | 36 |
| hgnetv2_b2.ssld_stage2_ft_in1k | 224x224 | 4.55 ms | 2.33 ms | 0.383 ms | – | 7.65 s | 23.4 s | 59 / 94 | 89 / 134 | – | 36 |
| hgnetv2_b3.ssld_stage2_ft_in1k | 224x224 | 5.22 ms | 2.55 ms | 0.436 ms | – | 7.99 s | 24.6 s | 76 / 114 | 99 / 152 | – | 36 |
| hgnetv2_b4.ssld_stage2_ft_in1k | 224x224 | 4.53 ms | 2.66 ms | 0.463 ms | – | 7.84 s | 19.8 s | 82 / 116 | 107 / 134 | – | 68 |
| hgnetv2_b5.ssld_stage2_ft_in1k | 224x224 | 7.04 ms | 4.13 ms | 0.758 ms | – | 9.97 s | 23.8 s | 114 / 150 | 146 / 190 | – | 100 |
| hgnetv2_b6.ssld_stage2_ft_in1k | 224x224 | 9.44 ms | 5.44 ms | 1.42 ms | – | 11.2 s | 26.9 s | 198 / 224 | 217 / 264 | – | 170 |
| inception_v3.tv_in1k | 224x224 | 5.21 ms | 3.03 ms | 0.685 ms | 1.30 ms | 8.61 s | 22.4 s | 120 / 166 | 152 / 184 | 150 / 228 | 70 |
| mobilenet_v2.tv2_in1k | 224x224 | 2.95 ms | 1.82 ms | 0.261 ms | 0.439 ms | 6.56 s | 16.3 s | 43 / 70 | 77 / 92 | 75 / 92 | 36 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 3.80 ms | 2.19 ms | 0.393 ms | 0.545 ms | 10.2 s | 36.5 s | 46 / 76 | 79 / 90 | 78 / 114 | 36 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 1.77 ms | 1.21 ms | 0.176 ms | 0.370 ms | 4.59 s | 14.1 s | 44 / 72 | 77 / 90 | 76 / 114 | 36 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 3.91 ms | 2.48 ms | 0.345 ms | 0.641 ms | 6.98 s | 19.6 s | 54 / 86 | 87 / 102 | 87 / 124 | 36 |
| regnety_160.tv2_in1k | 224x224 | 8.34 ms | 5.42 ms | 1.78 ms | 1.86 ms | 8.01 s | 28.1 s | 208 / 250 | 236 / 290 | 229 / 316 | 172 |
| repvit_m0_9.dist_300e_in1k | 224x224 | 5.35 ms | 3.30 ms | 0.509 ms | 0.776 ms | 8.62 s | 33.7 s | 45 / 76 | 77 / 96 | 45 / 76 | 32 |
| repvit_m0_9.dist_450e_in1k | 224x224 | 5.41 ms | 3.34 ms | 0.489 ms | 0.767 ms | 8.11 s | 30.5 s | 45 / 76 | 77 / 96 | 45 / 76 | 34 |
| repvit_m1_0.dist_300e_in1k | 224x224 | 5.29 ms | 3.33 ms | 0.503 ms | 0.786 ms | 8.20 s | 35.2 s | 48 / 78 | 80 / 118 | 48 / 78 | 32 |
| repvit_m1_0.dist_450e_in1k | 224x224 | 5.44 ms | 3.21 ms | 0.477 ms | 0.785 ms | 8.29 s | 40.2 s | 48 / 78 | 80 / 118 | 48 / 78 | 34 |
| repvit_m1_1.dist_300e_in1k | 224x224 | 5.03 ms | 6.12 ms | 0.470 ms | 0.765 ms | 7.72 s | 33.0 s | 51 / 62 | 83 / 102 | 51 / 64 | 34 |
| repvit_m1_1.dist_450e_in1k | 224x224 | 4.99 ms | 6.07 ms | 0.478 ms | 0.758 ms | 7.57 s | 33.4 s | 51 / 62 | 83 / 102 | 51 / 64 | 34 |
| repvit_m1_5.dist_300e_in1k | 224x224 | 8.55 ms | 4.89 ms | 0.766 ms | 1.19 ms | 10.2 s | 38.9 s | 62 / 94 | 94 / 114 | 62 / 94 | 36 |
| repvit_m1_5.dist_450e_in1k | 224x224 | 8.71 ms | 4.93 ms | 0.760 ms | 1.19 ms | 10.1 s | 33.8 s | 62 / 94 | 94 / 114 | 62 / 94 | 34 |
| repvit_m2_3.dist_300e_in1k | 224x224 | 11.1 ms | 6.51 ms | 1.03 ms | 1.66 ms | 11.9 s | 41.2 s | 80 / 128 | 112 / 148 | 79 / 130 | 72 |
| repvit_m2_3.dist_450e_in1k | 224x224 | 10.9 ms | 6.49 ms | 1.03 ms | 1.70 ms | 11.8 s | 40.8 s | 80 / 128 | 112 / 148 | 79 / 130 | 72 |
| resnet152.tv2_in1k | 224x224 | 8.36 ms | 4.55 ms | 1.09 ms | 1.83 ms | 8.82 s | 22.7 s | 155 / 186 | 185 / 240 | 182 / 266 | 134 |
| resnet18.tv_in1k | 224x224 | 1.48 ms | 1.04 ms | 0.220 ms | 0.459 ms | 4.31 s | 14.0 s | 61 / 90 | 91 / 108 | 90 / 150 | 34 |
| resnet50.tv2_in1k | 224x224 | 3.15 ms | 2.10 ms | 0.466 ms | 0.826 ms | 8.31 s | 24.0 s | 89 / 142 | 119 / 138 | 116 / 162 | 70 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 3.71 ms | 1.98 ms | 0.329 ms | 0.787 ms | 7.73 s | 16.9 s | 41 / 72 | 71 / 88 | 87 / 116 | 32 |
| shvit_s1.in1k | 224x224 | 6.75 ms | 3.11 ms | 0.510 ms | – | 10.2 s | 23.3 s | 47 / 58 | 78 / 98 | – | 32 |
| shvit_s2.in1k | 224x224 | 6.76 ms | 3.90 ms | 0.533 ms | – | 10.1 s | 24.0 s | 56 / 88 | 88 / 128 | – | 32 |
| shvit_s3.in1k | 224x224 | 8.11 ms | 3.64 ms | 0.575 ms | – | 10.9 s | 24.6 s | 62 / 112 | 94 / 132 | – | 32 |
| shvit_s4.in1k | 256x256 | 9.31 ms | 4.11 ms | 0.658 ms | – | 11.3 s | 26.3 s | 67 / 118 | 99 / 138 | – | 36 |
| squeezenet1_1.tv_in1k | 224x224 | 1.82 ms | 1.13 ms | 0.214 ms | 0.359 ms | 4.49 s | 9.10 s | 41 / 72 | 39 / 48 | 8 / 12 | 34 |
| starnet_s1.in1k | 224x224 | 3.88 ms | 2.41 ms | 0.774 ms | 0.683 ms | 6.23 s | 24.4 s | 41 / 52 | 73 / 92 | 41 / 56 | 34 |
| starnet_s2.in1k | 224x224 | 3.02 ms | 1.96 ms | 0.622 ms | 0.578 ms | 5.71 s | 24.7 s | 43 / 54 | 75 / 94 | 43 / 58 | 34 |
| starnet_s3.in1k | 224x224 | 4.19 ms | 2.60 ms | 0.878 ms | 0.715 ms | 6.03 s | 27.5 s | 47 / 60 | 79 / 100 | 47 / 64 | 36 |
| starnet_s4.in1k | 224x224 | 5.63 ms | 3.37 ms | 1.18 ms | 0.912 ms | 6.84 s | 28.0 s | 50 / 66 | 82 / 106 | 51 / 72 | 38 |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | 224x224 | 10.8 ms | 4.19 ms | 1.48 ms | 2.15 ms | 26.9 s | 64.2 s | 213 / 394 | 243 / 270 | 246 / 300 | 202 |
| tiny_vit_11m_224.dist_in22k_ft_in1k | 224x224 | 5.19 ms | 2.92 ms | 0.513 ms | 0.930 ms | 13.4 s | 60.7 s | 64 / 90 | 93 / 130 | 60 / 84 | 38 |
| tiny_vit_21m_224.dist_in22k_ft_in1k | 224x224 | 5.26 ms | 2.95 ms | 0.714 ms | 1.18 ms | 12.8 s | 50.3 s | 87 / 146 | 114 / 166 | 83 / 106 | 72 |
| tiny_vit_21m_384.dist_in22k_ft_in1k | 384x384 | 4.97 ms | 2.79 ms | 1.31 ms | 1.79 ms | 11.9 s | 65.7 s | 157 / 186 | 174 / 204 | 132 / 178 | 116 |
| tiny_vit_21m_512.dist_in22k_ft_in1k | 512x512 | 4.92 ms | 3.14 ms | 2.13 ms | 3.13 ms | 11.9 s | 79.3 s | 310 / 342 | 314 / 362 | 237 / 258 | 220 |
| tiny_vit_5m_224.dist_in22k_ft_in1k | 224x224 | 5.19 ms | 2.92 ms | 0.514 ms | 0.846 ms | 13.8 s | 54.0 s | 52 / 82 | 81 / 102 | 48 / 78 | 38 |
| vgg16.tv_in1k | 224x224 | 1.99 ms | 1.86 ms | 1.17 ms | 1.50 ms | 4.08 s | 15.9 s | 313 / 372 | 342 / 368 | 332 / 388 | 302 |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | 224x224 | 3.02 ms | 2.10 ms | 1.14 ms | 1.65 ms | 7.26 s | 28.6 s | 205 / 390 | 238 / 266 | 300 / 370 | 196 |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | 224x224 | 13.1 ms | 13.3 ms | 11.4 ms | 12.8 ms | 14.2 s | 60.7 s | 879 / 1708 | 909 / 944 | 1216 / 1346 | 848 |
| deeplabv3-r101-d8-ade20k | 1024x2048 | 47.9 ms | 42.1 ms | 30.6 ms | 20.1 ms | 9.60 s | 25.7 s | 453 / 532 | 450 / 512 | 362 / 518 | 954 |
| deeplabv3plus-r101-d8-ade20k | 1024x2048 | 47.1 ms | 44.6 ms | 30.9 ms | 18.0 ms | 10.9 s | 23.3 s | 413 / 492 | 410 / 432 | 365 / 560 | 920 |
| dinov2-b-linear-ade20k | 1024x2048 | 33.3 ms | 35.8 ms | 49.5 ms | 22.6 ms | 10.8 s | 40.6 s | 765 / 884 | 1455 / 1504 | 771 / 904 | 1922 |
| dinov2-l-linear-ade20k | 1024x2048 | 83.9 ms | 84.5 ms | 82.1 ms | 56.8 ms | 15.5 s | 43.8 s | 1181 / 1244 | 1870 / 1904 | 1189 / 1244 | 2396 |
| dinov2-s-linear-ade20k | 1024x2048 | 18.3 ms | 20.7 ms | 40.3 ms | 12.4 ms | 10.6 s | 36.2 s | 642 / 696 | 1331 / 1376 | 645 / 676 | 1794 |
| eomt-l-dinov2-ade20k | 1024x2048 | 50.8 ms | 49.4 ms | 54.3 ms | 35.0 ms | 16.5 s | 52.9 s | 1515 / 1584 | 1585 / 1624 | 1515 / 1584 | 1798 |
| eomt-l-dinov3-ade20k | 1024x2048 | 54.6 ms | 50.0 ms | 61.9 ms | 38.8 ms | 22.1 s | 57.5 s | 1514 / 1584 | 1584 / 1624 | 1514 / 1584 | 1800 |
| fcn-r101-d8-ade20k | 1024x2048 | 40.7 ms | 35.6 ms | 27.4 ms | 16.6 ms | 8.53 s | 22.2 s | 417 / 492 | 414 / 472 | 294 / 478 | 882 |
| hrnet-w48-fcn-ade20k | 1024x2048 | 26.9 ms | 23.9 ms | 20.9 ms | 11.9 ms | 35.7 s | 29.6 s | 424 / 490 | 421 / 448 | 274 / 430 | 888 |
| mask2former-swinl-ade20k | 1024x2048 | 103 ms | fails: Inductor: Triton kernel does not compile (torch bug) | 55.6 ms | 60.4 ms | – | 129 s | 1164 / 1400 | – | 1164 / 1400 | 1816 |
| oneformer-swinl-ade20k | 1024x2048 | 139 ms | fails: Inductor: Triton kernel does not compile (torch bug) | 83.3 ms | 100.0 ms | – | 140 s | 2054 / 2476 | – | 2054 / 2476 | 3258 |
| pspnet-r101-d8-ade20k | 1024x2048 | 48.8 ms | 41.3 ms | 29.8 ms | 25.0 ms | 11.0 s | 44.1 s | 417 / 492 | 414 / 472 | 341 / 494 | 1086 |
| sapiens-seg-0.3b-goliath | 1024x2048 | 132 ms | 118 ms | 156 ms | 92.6 ms | 12.5 s | 52.9 s | 1643 / 1762 | 1373 / 1462 | 1643 / 1762 | 1564 |
| sapiens-seg-1b-goliath | 1024x2048 | 275 ms | 263 ms | 244 ms | 185 ms | 18.8 s | 84.9 s | 3259 / 3384 | 2986 / 3084 | 3259 / 3382 | 3180 |
| segformer-b0-ade20k | 1024x2048 | 11.4 ms | 9.27 ms | 5.51 ms | 3.48 ms | 12.3 s | 60.9 s | 339 / 376 | 336 / 356 | 148 / 238 | 792 |
| segformer-b5-ade20k | 1024x2048 | 38.7 ms | 31.4 ms | 22.8 ms | 25.3 ms | 33.8 s | 83.9 s | 546 / 590 | 543 / 630 | 407 / 534 | 1206 |
| segnext-mscan-b-ade20k | 1024x2048 | 28.3 ms | 40.7 ms | 23.3 ms | 12.2 ms | 24.1 s | 66.3 s | 376 / 424 | 491 / 526 | 229 / 332 | 820 |
| segnext-mscan-l-ade20k | 1024x2048 | 43.1 ms | 63.6 ms | 32.3 ms | 20.6 ms | 34.9 s | 82.2 s | 417 / 458 | 532 / 578 | 314 / 428 | 860 |
| segnext-mscan-s-ade20k | 1024x2048 | 18.5 ms | 24.9 ms | 18.0 ms | 8.04 ms | 17.2 s | 56.6 s | 350 / 402 | 465 / 482 | 202 / 290 | 786 |
| segnext-mscan-t-ade20k | 1024x2048 | 15.9 ms | 16.6 ms | 16.0 ms | 8.99 ms | 19.1 s | 59.6 s | 332 / 378 | 446 / 480 | 147 / 206 | 788 |
| efficientvit-sam-l0 | 1024x1024 | 12.7 ms | 6.44 ms | 5.56 ms, fp32 engine (image encoder only) | 4.45 ms | 25.2 s | 19.9 s | 171 / 210 | 188 / 228 | 176 / 256 | 202 |
| efficientvit-sam-l1 | 1024x1024 | 14.4 ms | 11.3 ms | 7.30 ms, fp32 engine (image encoder only) | 5.79 ms | 24.8 s | 22.0 s | 195 / 234 | 212 / 252 | 203 / 284 | 266 |
| efficientvit-sam-l2 | 1024x1024 | 17.8 ms | 10.8 ms | 9.65 ms, fp32 engine (image encoder only) | 6.83 ms | 26.1 s | 24.1 s | 221 / 260 | 238 / 278 | 232 / 312 | 330 |
| efficientvit-sam-xl0 | 1024x1024 | 19.0 ms | 14.2 ms | 22.7 ms, fp32 engine (image encoder only) | 9.75 ms | 24.0 s | 29.8 s | 433 / 506 | 397 / 424 | 401 / 492 | 782 |
| efficientvit-sam-xl1 | 1024x1024 | 28.9 ms | 22.9 ms | 38.4 ms, fp32 engine (image encoder only) | 15.3 ms | 26.7 s | 36.6 s | 598 / 676 | 562 / 616 | 582 / 728 | 1058 |
| mobile-sam | 1024x1024 | 10.5 ms | 6.22 ms | 4.99 ms (image encoder only) | 6.86 ms | 25.5 s | 60.9 s | 183 / 210 | 203 / 232 | 215 / 296 | 122 |
| sam-hq-vit-b | 1024x1024 | 49.7 ms | 39.4 ms | 28.0 ms (image encoder only) | 36.8 ms | 32.8 s | 39.0 s | 388 / 422 | 372 / 422 | 433 / 564 | 342 |
| sam-hq-vit-h | 1024x1024 | 217 ms | 187 ms | 125 ms (image encoder only) | 144 ms | 42.3 s | 69.1 s | 1500 / 1566 | 1476 / 1526 | 1541 / 1708 | 1474 |
| sam-hq-vit-l | 1024x1024 | 118 ms | 97.0 ms | 68.3 ms (image encoder only) | 81.4 ms | 32.9 s | 53.3 s | 846 / 902 | 822 / 902 | 889 / 1046 | 796 |
| sam-vit-b | 1024x1024 | 48.3 ms | 40.9 ms | 27.3 ms (image encoder only) | 37.0 ms | 14.5 s | 36.4 s | 378 / 420 | 362 / 400 | 377 / 420 | 322 |
| sam-vit-h | 1024x1024 | 214 ms | 188 ms | 124 ms (image encoder only) | 143 ms | 42.6 s | 69.7 s | 1484 / 2524 | 1436 / 1484 | 1480 / 1524 | 1442 |
| sam-vit-l | 1024x1024 | 116 ms | 97.2 ms | 67.5 ms (image encoder only) | 81.5 ms | 32.9 s | 53.4 s | 833 / 1262 | 793 / 862 | 831 / 862 | 780 |
| sam2.1-hiera-base-plus | 1024x1024 | 32.0 ms | 26.4 ms | 16.7 ms (image encoder only) | 25.1 ms | 28.3 s | 43.7 s | 412 / 464 | 365 / 404 | 426 / 524 | 358 |
| sam2.1-hiera-large | 1024x1024 | 68.0 ms | 59.5 ms | 38.1 ms (image encoder only) | 48.8 ms | 38.9 s | 55.3 s | 742 / 768 | 678 / 748 | 760 / 828 | 704 |
| sam2.1-hiera-small | 1024x1024 | 18.8 ms | 14.9 ms | 9.93 ms (image encoder only) | 15.6 ms | 24.5 s | 40.2 s | 318 / 366 | 283 / 326 | 330 / 386 | 268 |
| sam2.1-hiera-tiny | 1024x1024 | 16.6 ms | 14.2 ms | 8.56 ms (image encoder only) | 13.9 ms | 13.4 s | 34.1 s | 304 / 382 | 270 / 322 | 316 / 362 | 266 |
| sam3 | 1024x1024 | 179 ms | 168 ms | 89.5 ms (image encoder only) | 116 ms | 20.0 s | 54.2 s | 1922 / 2004 | 1910 / 2004 | 2019 / 2264 | 1122 |
| detr-r50-coco | 800x1344 | 12.4 ms | 9.27 ms | 4.40 ms | 9.93 ms | 21.7 s | 33.4 s | 280 / 332 | 233 / 292 | 218 / 272 | 182 |
| dfine-l-coco | 800x1344 | 16.2 ms | 7.91 ms | 3.19 ms | 8.70 ms | 45.7 s | 58.7 s | 152 / 172 | 175 / 198 | 179 / 300 | 108 |
| dfine-m-coco | 800x1344 | 13.5 ms | 6.79 ms | 2.49 ms | 5.97 ms | 36.4 s | 54.6 s | 126 / 142 | 149 / 166 | 156 / 252 | 116 |
| dfine-n-coco | 800x1344 | 10.3 ms | 5.43 ms | 0.942 ms | 2.72 ms | 26.7 s | 51.5 s | 88 / 116 | 101 / 118 | 124 / 202 | 56 |
| dfine-s-coco | 800x1344 | 11.1 ms | 5.63 ms | 1.65 ms | 4.04 ms | 31.2 s | 51.4 s | 105 / 128 | 131 / 152 | 138 / 212 | 76 |
| dfine-x-coco | 800x1344 | 18.5 ms | 11.0 ms | fails: TensorRT: NaN output (onnxruntime on the same ONNX is right) | 14.1 ms | 47.7 s | – | 227 / 258 | 245 / 284 | 237 / 364 | – |
| faster-rcnn-r50-fpn-3x-coco | 800x1344 | 22.8 ms | 18.9 ms | 5.35 ms (backbone + FPN only) | 19.5 ms | 26.0 s | 15.6 s | 390 / 490 | 413 / 438 | 391 / 450 | 258 |
| mask-rcnn-r50-fpn-3x-coco | 800x1344 | 24.6 ms | 21.5 ms | 5.34 ms (backbone + FPN only) | 21.0 ms | 29.3 s | 18.3 s | 352 / 440 | 333 / 358 | 353 / 400 | 258 |
| rtdetrv2-s-coco | 800x1344 | 8.04 ms | 4.22 ms | 1.81 ms | 4.19 ms | 37.0 s | 42.2 s | 134 / 166 | 152 / 182 | 155 / 250 | 98 |
| rtmdet-l-coco | 800x1344 | 9.60 ms | 10.6 ms | 3.59 ms | 5.74 ms | 17.2 s | 25.0 s | 183 / 230 | 178 / 226 | 178 / 284 | 154 |
| rtmdet-m-coco | 800x1344 | 7.69 ms | 4.78 ms | 2.39 ms | 4.49 ms | 15.7 s | 22.4 s | 114 / 148 | 115 / 130 | 122 / 200 | 84 |
| rtmdet-m-person | 800x1344 | 7.39 ms | 4.44 ms | 2.38 ms | 4.12 ms | 15.5 s | 24.8 s | 114 / 146 | 115 / 128 | 119 / 202 | 84 |
| rtmdet-nano-person | 800x1344 | 7.56 ms | 4.23 ms | 0.769 ms | 2.10 ms | 16.3 s | 22.6 s | 48 / 84 | 48 / 64 | 49 / 92 | 34 |
| rtmdet-s-coco | 800x1344 | 6.25 ms | 3.56 ms | 1.19 ms | 2.98 ms | 14.3 s | 20.1 s | 81 / 102 | 76 / 98 | 88 / 136 | 54 |
| rtmdet-tiny-coco | 800x1344 | 6.08 ms | 3.43 ms | 0.962 ms | 2.69 ms | 13.7 s | 19.3 s | 72 / 96 | 64 / 94 | 80 / 124 | 50 |
| rtmdet-x-coco | 800x1344 | 13.3 ms | 11.8 ms | 6.17 ms | 8.55 ms | 18.6 s | 27.8 s | 267 / 316 | 271 / 292 | 267 / 378 | 256 |
| dwpose-l | 1080x1920 | 10.2 ms | 5.75 ms | 3.33 ms (network on person crops only) | 6.20 ms | 14.8 s | 20.0 s | 222 / 280 | 197 / 216 | 186 / 306 | 142 |
| dwpose-l-384 | 1080x1920 | 21.2 ms | 11.9 ms | 7.46 ms (network on person crops only) | 12.5 ms | 14.9 s | 20.6 s | 351 / 460 | 253 / 296 | 240 / 424 | 198 |
| dwpose-m | 1080x1920 | 7.25 ms | 3.80 ms | 2.13 ms (network on person crops only) | 4.86 ms | 12.9 s | 18.9 s | 162 / 196 | 156 / 174 | 157 / 220 | 98 |
| dwpose-s | 1080x1920 | 5.03 ms | 2.27 ms | 1.17 ms (network on person crops only) | 3.30 ms | 11.8 s | 16.7 s | 129 / 174 | 135 / 150 | 138 / 198 | 62 |
| dwpose-t | 1080x1920 | 4.47 ms | 1.96 ms | 0.940 ms (network on person crops only) | 3.15 ms | 11.2 s | 16.1 s | 124 / 152 | 129 / 148 | 133 / 192 | 54 |
| hrnet-w32 | 1080x1920 | 20.3 ms | 9.94 ms | 5.18 ms (network on person crops only) | 9.79 ms | 21.1 s | 25.5 s | 195 / 258 | 178 / 214 | 210 / 262 | 126 |
| hrnet-w32-384 | 1080x1920 | 38.8 ms | 19.6 ms | 10.7 ms (network on person crops only) | 19.4 ms | 21.5 s | 26.3 s | 341 / 434 | 211 / 234 | 305 / 380 | 192 |
| hrnet-w48 | 1080x1920 | 32.0 ms | 21.2 ms | 9.46 ms (network on person crops only) | 15.5 ms | 21.3 s | 26.9 s | 262 / 322 | 245 / 282 | 343 / 398 | 190 |
| hrnet-w48-384 | 1080x1920 | 70.5 ms | 46.0 ms | 22.3 ms (network on person crops only) | 33.9 ms | 22.6 s | 28.3 s | 408 / 498 | 278 / 300 | 438 / 498 | 256 |
| rf-detr-keypoint-preview | 1080x1920 | 18.3 ms | 8.00 ms | 4.28 ms (network on person crops only) | 9.48 ms | 47.9 s | 67.2 s | 179 / 204 | 191 / 220 | 176 / 224 | 124 |
| rtmo-l | 1080x1920 | 10.8 ms | 7.07 ms | 2.90 ms (network on person crops only) | 8.30 ms | 18.9 s | 39.6 s | 208 / 226 | 192 / 224 | 208 / 228 | 126 |
| rtmo-m | 1080x1920 | 8.71 ms | 4.99 ms | 2.03 ms (network on person crops only) | 6.85 ms | 18.0 s | 39.4 s | 166 / 190 | 146 / 170 | 166 / 212 | 86 |
| rtmo-s | 1080x1920 | 7.48 ms | 3.35 ms | 1.21 ms (network on person crops only) | 6.27 ms | 14.6 s | 36.7 s | 142 / 160 | 121 / 156 | 142 / 180 | 50 |
| rtmo-t | 1080x1920 | 7.62 ms | 3.30 ms | 0.886 ms (network on person crops only) | 6.31 ms | 15.6 s | 36.8 s | 121 / 136 | 111 / 134 | 121 / 156 | 40 |
| rtmpose-l | 1080x1920 | 8.94 ms | 5.32 ms | 3.20 ms (network on person crops only) | 5.63 ms | 14.2 s | 19.4 s | 211 / 236 | 185 / 194 | 175 / 264 | 112 |
| rtmpose-l-384 | 1080x1920 | 19.3 ms | 10.7 ms | 7.08 ms (network on person crops only) | 11.2 ms | 15.0 s | 20.3 s | 340 / 436 | 242 / 274 | 229 / 406 | 168 |
| rtmpose-l-halpe26 | 1080x1920 | 9.06 ms | 5.34 ms | 3.20 ms (network on person crops only) | 5.65 ms | 14.3 s | 19.6 s | 212 / 276 | 186 / 194 | 176 / 264 | 112 |
| rtmpose-l-halpe26-384 | 1080x1920 | 19.3 ms | 10.6 ms | 7.08 ms (network on person crops only) | 11.2 ms | 15.1 s | 20.6 s | 341 / 436 | 243 / 276 | 230 / 406 | 168 |
| rtmpose-m | 1080x1920 | 6.28 ms | 3.42 ms | 2.02 ms (network on person crops only) | 4.42 ms | 12.7 s | 17.8 s | 153 / 192 | 147 / 172 | 148 / 200 | 66 |
| rtmpose-m-384 | 1080x1920 | 13.1 ms | 7.09 ms | 4.33 ms (network on person crops only) | 8.97 ms | 13.6 s | 18.6 s | 243 / 352 | 190 / 212 | 202 / 340 | 110 |
| rtmpose-m-ap10k | 1080x1920 | 7.98 ms | 4.54 ms | 2.57 ms (network on person crops only) | 5.41 ms | 13.0 s | 18.2 s | 177 / 234 | 176 / 192 | 163 / 220 | 78 |
| rtmpose-m-face6 | 1080x1920 | 8.69 ms | 4.79 ms | 2.67 ms (network on person crops only) | 5.78 ms | 12.8 s | 18.4 s | 184 / 256 | 182 / 192 | 169 / 242 | 110 |
| rtmpose-m-halpe26 | 1080x1920 | 6.35 ms | 3.44 ms | 2.02 ms (network on person crops only) | 4.46 ms | 12.9 s | 18.0 s | 154 / 192 | 148 / 172 | 149 / 220 | 68 |
| rtmpose-m-halpe26-384 | 1080x1920 | 13.2 ms | 7.12 ms | 4.34 ms (network on person crops only) | 9.01 ms | 13.7 s | 18.9 s | 244 / 352 | 190 / 212 | 203 / 340 | 108 |
| rtmpose-m-hand5 | 1080x1920 | 8.04 ms | 4.51 ms | 2.58 ms (network on person crops only) | 5.44 ms | 12.9 s | 18.3 s | 178 / 234 | 176 / 192 | 163 / 240 | 80 |
| rtmpose-s | 1080x1920 | 4.92 ms | 2.07 ms | 1.06 ms (network on person crops only) | 3.01 ms | 11.5 s | 16.5 s | 124 / 172 | 129 / 150 | 133 / 178 | 62 |
| rtmpose-s-face6 | 1080x1920 | 5.03 ms | 2.69 ms | 1.28 ms (network on person crops only) | 3.75 ms | 11.9 s | 16.9 s | 147 / 174 | 164 / 190 | 151 / 222 | 70 |
| rtmpose-s-halpe26 | 1080x1920 | 4.98 ms | 2.11 ms | 1.07 ms (network on person crops only) | 3.04 ms | 11.7 s | 16.6 s | 124 / 172 | 130 / 150 | 133 / 198 | 62 |
| rtmpose-t | 1080x1920 | 4.49 ms | 1.91 ms | 0.866 ms (network on person crops only) | 2.90 ms | 10.8 s | 15.9 s | 120 / 150 | 125 / 128 | 129 / 176 | 56 |
| rtmpose-t-face6 | 1080x1920 | 4.58 ms | 2.23 ms | 1.00 ms (network on person crops only) | 3.63 ms | 11.5 s | 16.2 s | 134 / 152 | 159 / 168 | 146 / 220 | 62 |
| rtmpose-t-halpe26 | 1080x1920 | 4.48 ms | 1.94 ms | 0.871 ms (network on person crops only) | 2.90 ms | 13.5 s | 17.2 s | 120 / 150 | 125 / 128 | 129 / 196 | 56 |
| rtmpose-x-halpe26-384 | 1080x1920 | 33.5 ms | 21.1 ms | 13.3 ms (network on person crops only) | 18.7 ms | 17.4 s | 23.0 s | 401 / 498 | 262 / 298 | 280 / 470 | 258 |
| rtmw-l | 1080x1920 | 15.1 ms | 9.34 ms | 5.17 ms (network on person crops only) | 10.2 ms | 18.8 s | 24.9 s | 267 / 308 | 242 / 264 | 232 / 336 | 174 |
| rtmw-l-384 | 1080x1920 | 31.5 ms | 19.5 ms | 11.2 ms (network on person crops only) | 20.2 ms | 19.8 s | 26.8 s | 396 / 508 | 298 / 344 | 286 / 476 | 230 |
| rtmw-m | 1080x1920 | 11.2 ms | 6.23 ms | 3.34 ms (network on person crops only) | 7.54 ms | 18.7 s | 23.1 s | 189 / 224 | 183 / 202 | 184 / 268 | 98 |
| rtmw-x | 1080x1920 | 23.9 ms | 15.3 ms | 8.47 ms (network on person crops only) | 14.3 ms | 20.4 s | 27.0 s | 342 / 392 | 320 / 328 | 298 / 394 | 250 |
| rtmw-x-384 | 1080x1920 | 50.6 ms | 34.0 ms | 19.3 ms (network on person crops only) | 50.8 ms | 21.2 s | 29.8 s | 482 / 592 | 361 / 388 | 496 / 674 | 320 |
| vitpose-base | 1080x1920 | 19.9 ms | 16.4 ms | 8.88 ms (network on person crops only) | 14.0 ms | 10.5 s | 35.8 s | 597 / 642 | 597 / 622 | 574 / 704 | 234 |
| vitpose-base-simple | 1080x1920 | 20.1 ms | 16.0 ms | 9.36 ms (network on person crops only) | 14.1 ms | 11.8 s | 35.0 s | 590 / 642 | 590 / 622 | 567 / 684 | 268 |
| vitpose-plus-base | 1080x1920 | 19.9 ms | 16.4 ms | 8.87 ms (network on person crops only) | 13.9 ms | 10.6 s | 35.0 s | 597 / 642 | 597 / 622 | 574 / 704 | 234 |
| vitpose-plus-huge | 1080x1920 | 104 ms | 92.9 ms | 58.7 ms (network on person crops only) | 64.1 ms | 21.2 s | 47.3 s | 1641 / 1704 | 1641 / 1682 | 1618 / 1766 | 1312 |
| vitpose-plus-large | 1080x1920 | 53.7 ms | 46.9 ms | 29.3 ms (network on person crops only) | 34.9 ms | 17.9 s | 39.0 s | 1014 / 1062 | 1014 / 1062 | 991 / 1124 | 656 |
| vitpose-plus-small | 1080x1920 | 9.96 ms | 7.54 ms | 3.39 ms (network on person crops only) | 8.07 ms | 10.6 s | 37.5 s | 472 / 496 | 472 / 496 | 449 / 558 | 100 |
| c-radio-v3-b | 224x224 | 3.04 ms | 7.33 ms | 1.19 ms | 1.35 ms | 8.29 s | 37.4 s | 217 / 246 | 278 / 306 | 284 / 326 | 230 |
| c-radio-v3-g | 224x224 | 14.6 ms | 25.5 ms | 10.3 ms | 10.6 ms | 16.5 s | 57.1 s | 2266 / 2308 | 2329 / 2368 | 3166 / 3230 | 2284 |
| c-radio-v3-h | 224x224 | 8.90 ms | 18.1 ms | 6.62 ms | 7.42 ms | 14.7 s | 44.7 s | 1289 / 1308 | 1350 / 1386 | 1789 / 1828 | 1320 |
| c-radio-v3-l | 224x224 | 5.53 ms | 11.8 ms | 3.06 ms | 3.26 ms | 11.3 s | 41.8 s | 648 / 686 | 710 / 746 | 888 / 946 | 684 |
| clip-b16-laion2b | 224x224 | 3.09 ms | 2.41 ms | 1.12 ms | 1.36 ms | 6.49 s | 24.3 s | 202 / 226 | 237 / 266 | 270 / 310 | 196 |
| clip-b16-openai | 224x224 | 3.35 ms | 2.39 ms | 1.11 ms | 1.35 ms | 6.37 s | 14.6 s | 204 / 226 | 237 / 266 | 324 / 370 | 196 |
| clip-h14-laion2b | 224x224 | 11.2 ms | 11.6 ms | 7.84 ms | 8.26 ms | 14.3 s | 40.6 s | 1249 / 1268 | 1289 / 1310 | 1750 / 1792 | 1226 |
| clip-l14-336-openai | 224x224 | 6.16 ms | 6.36 ms | 4.25 ms | 4.75 ms | 10.7 s | 33.1 s | 625 / 646 | 658 / 686 | 1056 / 1090 | 620 |
| clip-l14-datacompxl | 224x224 | 5.99 ms | 6.24 ms | 4.19 ms | 4.77 ms | 10.5 s | 32.1 s | 621 / 666 | 658 / 686 | 862 / 930 | 618 |
| clip-l14-laion2b | 224x224 | 5.98 ms | 6.19 ms | 4.19 ms | 4.77 ms | 10.4 s | 32.0 s | 621 / 666 | 658 / 686 | 862 / 930 | 618 |
| clip-l14-openai | 224x224 | 6.16 ms | 6.17 ms | 4.14 ms | 4.74 ms | 10.2 s | 23.0 s | 623 / 666 | 658 / 686 | 1054 / 1110 | 616 |
| convnext_base.fb_in22k | 224x224 | 5.57 ms | 3.21 ms | 1.51 ms | 1.96 ms | 10.2 s | 29.9 s | 289 / 296 | 321 / 334 | 204 / 216 | 200 |
| convnext_large.fb_in22k | 224x224 | 5.53 ms | 3.34 ms | 2.50 ms | 3.23 ms | 9.87 s | 46.7 s | 600 / 612 | 639 / 650 | 413 / 434 | 396 |
| convnext_small.fb_in22k | 224x224 | 5.64 ms | 3.36 ms | 1.16 ms | 1.79 ms | 9.51 s | 35.1 s | 179 / 190 | 210 / 230 | 130 / 150 | 104 |
| convnext_tiny.fb_in22k | 224x224 | 3.18 ms | 2.08 ms | 0.726 ms | 1.09 ms | 12.0 s | 33.9 s | 117 / 130 | 148 / 170 | 88 / 110 | 72 |
| dinov2-vitb14 | 224x224 | 2.92 ms | 2.20 ms | 1.30 ms | 1.48 ms | 9.52 s | 25.2 s | 205 / 226 | 240 / 266 | 237 / 290 | 198 |
| dinov2-vitl14 | 224x224 | 6.12 ms | 6.03 ms | 4.25 ms | 4.59 ms | 13.2 s | 33.5 s | 622 / 646 | 661 / 666 | 655 / 714 | 618 |
| dinov2-vits14 | 224x224 | 2.95 ms | 2.02 ms | 0.601 ms | 0.848 ms | 9.99 s | 22.9 s | 79 / 98 | 111 / 140 | 112 / 148 | 64 |
| dinov3-convnext-large | 224x224 | 5.45 ms | 4.30 ms | 2.50 ms | 3.59 ms | 11.3 s | 30.0 s | 419 / 430 | 459 / 470 | 441 / 494 | 396 |
| dinov3-convnext-tiny | 224x224 | 3.04 ms | 1.96 ms | 0.681 ms | 1.18 ms | 7.76 s | 22.8 s | 93 / 108 | 123 / 148 | 86 / 112 | 72 |
| dinov3-vits16 | 224x224 | 4.67 ms | 2.18 ms | 0.564 ms | 0.903 ms | 8.81 s | 25.0 s | 77 / 80 | 109 / 138 | 77 / 106 | 64 |
| pe-core-b16-224 | 224x224 | 4.97 ms | 2.50 ms | 1.17 ms | 1.47 ms | 8.71 s | 25.8 s | 218 / 266 | 251 / 286 | 285 / 348 | 196 |
| pe-core-g14-448 | 224x224 | 25.4 ms | 24.5 ms | 15.0 ms | 17.0 ms | 29.9 s | 106 s | 3642 / 3670 | 3667 / 3690 | 5180 / 5256 | 3564 |
| pe-core-l14-336 | 224x224 | 9.09 ms | 6.53 ms | 4.34 ms | 4.96 ms | 13.9 s | 50.1 s | 650 / 668 | 683 / 706 | 889 / 932 | 620 |
| pe-core-s16-384 | 224x224 | 4.80 ms | 2.39 ms | 0.581 ms | 0.951 ms | 8.75 s | 36.7 s | 83 / 102 | 113 / 122 | 100 / 128 | 66 |
| pe-core-t16-384 | 224x224 | 4.68 ms | 2.29 ms | 0.432 ms | 0.683 ms | 8.89 s | 36.1 s | 48 / 56 | 78 / 96 | 53 / 82 | 34 |
| pe-spatial-b16-512 | 224x224 | 4.74 ms | 2.67 ms | 1.20 ms | 1.39 ms | 8.67 s | 35.1 s | 207 / 226 | 237 / 266 | 274 / 328 | 198 |
| pe-spatial-g14-448 | 224x224 | 25.5 ms | 24.3 ms | 14.9 ms | 16.9 ms | 28.3 s | 126 s | 3584 / 3612 | 3608 / 3630 | 5123 / 5156 | 3568 |
| pe-spatial-l14-448 | 224x224 | 8.84 ms | 6.59 ms | 4.35 ms | 4.86 ms | 13.7 s | 53.4 s | 626 / 668 | 658 / 686 | 865 / 932 | 620 |
| pe-spatial-s16-512 | 224x224 | 4.51 ms | 2.67 ms | 0.580 ms | 0.916 ms | 8.67 s | 35.3 s | 81 / 100 | 110 / 120 | 98 / 126 | 66 |
| pe-spatial-t16-512 | 224x224 | 4.48 ms | 2.12 ms | 0.426 ms | 0.666 ms | 8.45 s | 33.1 s | 49 / 56 | 77 / 96 | 53 / 82 | 34 |
| siglip2-b16-256 | 224x224 | 3.28 ms | 2.53 ms | 1.13 ms | 1.43 ms | 6.89 s | 24.3 s | 216 / 248 | 250 / 266 | 284 / 350 | 196 |
| siglip2-b16-512 | 224x224 | 3.31 ms | 2.75 ms | 1.14 ms | 1.43 ms | 6.85 s | 24.3 s | 219 / 266 | 251 / 286 | 287 / 368 | 196 |
| siglip2-g16-256 | 224x224 | 14.7 ms | 15.3 ms | 9.99 ms | 10.6 ms | 17.9 s | 51.2 s | 2262 / 2290 | 2307 / 2348 | 3163 / 3232 | 2188 |
| siglip2-g16-384 | 224x224 | 14.7 ms | 16.9 ms | 10.2 ms | 10.6 ms | 19.3 s | 54.1 s | 2264 / 2288 | 2308 / 2348 | 3165 / 3230 | 2188 |
| siglip2-l16-256 | 224x224 | 5.71 ms | 5.15 ms | 2.93 ms | 3.31 ms | 10.8 s | 31.7 s | 643 / 668 | 679 / 708 | 883 / 930 | 616 |
| siglip2-l16-512 | 224x224 | 5.70 ms | 5.38 ms | 2.91 ms | 3.31 ms | 11.1 s | 28.2 s | 647 / 666 | 681 / 706 | 887 / 930 | 614 |
| siglip2-so400m14-224 | 224x224 | 6.38 ms | 6.71 ms | 3.77 ms | 4.30 ms | 12.3 s | 34.7 s | 858 / 868 | 887 / 908 | 1182 / 1230 | 808 |
| siglip2-so400m14-378 | 224x224 | 6.38 ms | 6.98 ms | 3.79 ms | 4.29 ms | 12.2 s | 32.4 s | 861 / 908 | 888 / 926 | 1185 / 1250 | 808 |
| siglip2-so400m16-256 | 224x224 | 6.42 ms | 6.62 ms | 3.60 ms | 4.19 ms | 12.9 s | 30.0 s | 857 / 868 | 894 / 908 | 1181 / 1232 | 806 |
| siglip2-so400m16-512 | 224x224 | 6.40 ms | 7.02 ms | 3.61 ms | 4.18 ms | 13.2 s | 29.8 s | 861 / 906 | 896 / 928 | 1185 / 1250 | 806 |
| tipsv2-b14 | 224x224 | 2.93 ms | 2.18 ms | 1.33 ms | 1.49 ms | 9.18 s | 41.5 s | 204 / 224 | 239 / 266 | 206 / 226 | 198 |
| tipsv2-l14 | 224x224 | 6.15 ms | 6.04 ms | 4.28 ms | 4.58 ms | 12.4 s | 47.9 s | 621 / 624 | 659 / 666 | 624 / 648 | 620 |
| mapanything-apache | 336x518 | 636 ms | 596 ms | 367 ms | 544 ms | 46.2 s | 137 s | 3697 / 4092 | 3347 / 3792 | 3697 / 4072 | 3880 |
| mvsanywhere | 448x672 | 93.2 ms | 70.8 ms | fails: TensorRT: NaN output (onnxruntime on the same ONNX is right) | 35.6 ms | 153 s | – | 931 / 1226 | 818 / 860 | 761 / 884 | – |
| patchmatchnet | 1536x1024 | 202 ms | 175 ms | 51.0 ms | 29.6 ms | 29.4 s | 82.1 s | 1806 / 2096 | 1798 / 1888 | 740 / 856 | 844 |
| vggt-1b | 336x518 | 651 ms | 557 ms | 424 ms | 458 ms | 74.3 s | 199 s | 3691 / 4674 | 3611 / 3856 | 3702 / 4654 | 3642 |
| vggt-omega-1b-512 | 336x518 | 726 ms | 589 ms | 377 ms | 464 ms | 65.7 s | 187 s | 3421 / 3518 | 3082 / 3154 | 3422 / 3518 | 2994 |
| flowformer-small-chairs | 375x1242 | 351 ms | 250 ms | 202 ms | – | 98.7 s | 118 s | 3165 / 3638 | 2438 / 3196 | – | 2006 |
| flowformer-small-sintel | 375x1242 | 351 ms | 250 ms | 202 ms | – | 98.9 s | 117 s | 3165 / 3638 | 2438 / 3196 | – | 2006 |
| flowformer-small-things | 375x1242 | 351 ms | 250 ms | 202 ms | – | 99.3 s | 118 s | 3165 / 3638 | 2438 / 3196 | – | 2006 |
| flowseek-t | 1080x1920 | 395 ms | 258 ms | 168 ms | – | 54.2 s | 107 s | 8671 / 8712 | 6126 / 6212 | – | 11022 |
| flowseek-t-things | 1080x1920 | 395 ms | 258 ms | 168 ms | – | 54.6 s | 102 s | 8671 / 8712 | 6126 / 6212 | – | 11024 |
| flowseek-t-things-notartan | 1080x1920 | 396 ms | 258 ms | 168 ms | – | 54.3 s | 97.5 s | 8671 / 8712 | 6126 / 6212 | – | 11022 |
| gmflow-scale1-mixdata | 1080x1920 | 800 ms | 382 ms | 187 ms | – | 29.1 s | 67.3 s | 5523 / 5660 | 4293 / 4558 | – | 4172 |
| gmflow-scale1-things | 1080x1920 | 801 ms | 382 ms | 187 ms | – | 35.3 s | 69.3 s | 5523 / 5660 | 4293 / 4558 | – | 4172 |
| gmflow-scale2-mixdata | 1080x1920 | 1645 ms | 765 ms | fails: TensorRT bug: Myelin internal error | – | 46.6 s | – | 6433 / 7244 | 10865 / 10904 | – | – |
| gmflow-scale2-regrefine6-kitti | 1080x1920 | 2194 ms | 1116 ms | fails: TensorRT bug: Myelin internal error | – | 58.4 s | – | 6443 / 7250 | 10875 / 10972 | – | – |
| gmflow-scale2-regrefine6-mixdata | 1080x1920 | 2192 ms | 1116 ms | fails: TensorRT bug: Myelin internal error | – | 58.9 s | – | 6443 / 7250 | 10875 / 10972 | – | – |
| gmflow-scale2-regrefine6-sintelft | 1080x1920 | 2189 ms | 1116 ms | fails: TensorRT bug: Myelin internal error | – | 58.5 s | – | 6443 / 7250 | 10875 / 10972 | – | – |
| gmflow-scale2-regrefine6-things | 1080x1920 | 2183 ms | 1116 ms | fails: TensorRT bug: Myelin internal error | – | 58.2 s | – | 6443 / 7250 | 10875 / 10972 | – | – |
| gmflow-scale2-sintel | 1080x1920 | 1644 ms | 765 ms | fails: TensorRT bug: Myelin internal error | – | 46.6 s | – | 6433 / 7244 | 10865 / 10904 | – | – |
| gmflow-scale2-things | 1080x1920 | 1636 ms | 765 ms | fails: TensorRT bug: Myelin internal error | – | 46.7 s | – | 6433 / 7244 | 10865 / 10904 | – | – |
| neuflow-v2 | 1080x1920 | 103 ms | 91.4 ms | 65.0 ms | 59.0 ms | 28.5 s | 104 s | 8392 / 8416 | 4529 / 4558 | 790 / 896 | 8532 |
| neuflow-v2-sintel | 1080x1920 | 102 ms | 91.4 ms | 64.9 ms | 58.9 ms | 28.5 s | 102 s | 8392 / 8416 | 4529 / 4558 | 790 / 896 | 8532 |
| neuflow-v2-things | 1080x1920 | 103 ms | 91.5 ms | 65.0 ms | 58.9 ms | 28.8 s | 104 s | 8392 / 8416 | 4529 / 4558 | 790 / 896 | 8532 |
| raft-large | 1080x1920 | 290 ms | 195 ms | fails: TensorRT engine build failed | 83.3 ms | 41.3 s | – | 8268 / 8298 | 5886 / 5924 | 333 / 352 | – |
| raft-large-kitti | 1080x1920 | 290 ms | 195 ms | fails: TensorRT engine build failed | 83.6 ms | 39.3 s | – | 8268 / 8298 | 5886 / 5924 | 333 / 352 | – |
| raft-large-things | 1080x1920 | 290 ms | 195 ms | fails: TensorRT engine build failed | 83.1 ms | 39.4 s | – | 8268 / 8298 | 5886 / 5924 | 333 / 352 | – |
| raft-small | 1080x1920 | 144 ms | 110 ms | fails: TensorRT engine build failed | 36.3 ms | 37.0 s | – | 8198 / 8226 | 5704 / 5728 | 197 / 224 | – |
| sea-raft-m | 1080x1920 | 299 ms | 191 ms | 132 ms | 73.1 ms | 34.3 s | 70.0 s | 8333 / 8420 | 5869 / 5920 | 435 / 514 | 10892 |
| sea-raft-m-kitti | 1080x1920 | 299 ms | 191 ms | 132 ms | 73.1 ms | 34.9 s | 69.7 s | 8333 / 8420 | 5869 / 5920 | 435 / 514 | 10892 |
| sea-raft-m-spring | 1080x1920 | 52.3 ms | 34.6 ms | 4.43 ms | 23.2 ms | 23.2 s | 47.1 s | 713 / 740 | 624 / 640 | 181 / 214 | 120 |
| sea-raft-m-things | 1080x1920 | 299 ms | 192 ms | 132 ms | 73.1 ms | 34.9 s | 73.6 s | 8333 / 8420 | 5869 / 5920 | 435 / 514 | 10892 |
| sea-raft-s | 1080x1920 | 226 ms | 150 ms | 108 ms | 50.4 ms | 35.4 s | 68.0 s | 8292 / 8354 | 5828 / 5876 | 415 / 492 | 10866 |
| sea-raft-s-kitti | 1080x1920 | 226 ms | 150 ms | 108 ms | 50.5 ms | 31.8 s | 67.3 s | 8292 / 8354 | 5828 / 5876 | 415 / 492 | 10866 |
| sea-raft-s-spring | 1080x1920 | 37.1 ms | 23.8 ms | 3.01 ms | 16.0 ms | 21.9 s | 38.8 s | 672 / 714 | 582 / 596 | 160 / 192 | 86 |
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
