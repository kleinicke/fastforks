# Apple M1 Pro 16 GB

`portable` and Metal `kernels` on Apple silicon (MPS). The tables are generated from the data file named below;
the text around them is written by hand.

The 480x640 rows are `portable`, one variant per family, on the `fastforks bench` input (classification and features
at their native crop). float16 is measured only where the card offers it on MPS (`unsupported` = it does not). Timing:
median of 5 warm calls (2 for calls over 2 s). Memory: the process's peak footprint (Activity Monitor's "Memory":
model and activations, which share the system RAM on a Mac); it is not comparable with the CUDA tables' torch peak
memory. The first call at a new size takes 0.5-4 s longer. Not measured: the gated DINOv3, DINOv3-ConvNeXt, SAM 3 and
VGGT-Omega weights. Reproduce (`bench` prints the Metal driver's memory: the GPU buffers only, not the whole process):

    fastforks bench MODEL --device mps --size 480x640 --views 4 [--dtype float16]

These rows predate the cross-device protocol: they are at 480x640 and 384x1248, not at the tasks' standard inputs,
so the [across-devices tables](../benchmarks.md#across-devices) do not show them yet. New rows at the standard inputs:
`uv run python tools/speed_survey.py run --device apple-m1-pro-16gb` (docs/data/README.md, "Devices").

Accuracy against the CPU (float32, Middlebury ArtL for stereo): MPS float32 outputs match to 1e-3 px mean or better
(most families 1e-5), except MonSter++ (0.006 px) and NMRF (0.03 px on non-occluded pixels, about 0.7 px in
occlusions, where its local-maximum test flips on last-bit differences in flat cost curves; the error against ground
truth is unchanged). float16 moves stereo outputs by 0.003-0.07 px mean. Depth: float32 matches the CPU to 1e-4 of the
median depth, float16 moves it by 0.02-0.12 %.

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed/apple-m1-pro-16gb.jsonl) -->

| | |
|---|---|
| device | Apple M1 Pro 16 GB |
| backend | mps |
| memory | 16 GB unified |
| hardware | 16-core GPU |
| software | macOS 27, torch 2.14 |
| torch (rows) | 2.14, 2.14.1 |
| measured | 2026-11-03 to 2026-11-06 |
| rows | 602 in `docs/data/speed/apple-m1-pro-16gb.jsonl` |

A survey, not a regression floor: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest `portable` row of the same input (its dtype in brackets). Memory is measured the way the table says; it is not comparable with another device's. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

## Stereo

### Stereo, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 345 | unsupported |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 726 | unsupported |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 1169 | unsupported |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 1369 | 1335 |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 1829 | 1775 |
| [`coex-sceneflow`](../models/stereo.md#coex) | 196 | 187 |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 954 | 788 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 1863 | unsupported |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 591 | 537 |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 242 | unsupported |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | 8814 | 6265 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | 5446 | 2176 |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 1185 | unsupported |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 117 | 124 |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 2283 | 2154 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 1441 | unsupported |
| [`las2-h`](../models/stereo.md#las2) | 278 | unsupported |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 2066 | 1916 |
| [`monster-mix-all`](../models/stereo.md#monster) | 2512 | 2275 |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 2707 | 2467 |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 305 | unsupported |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 1092 | unsupported |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 953 | 866 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 148 | unsupported |
| [`s2m2-l`](../models/stereo.md#s2m2) | 1300 | 1198 |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 1845 | 1666 |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | 740 | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `aanet-plus-sceneflow` | 2623 | – |
| `bridgedepth-rvc-pretrain` | 4004 | – |
| `c-fast-foundationstereo` | 2696 | – |
| `cas-gwcnet-c-sceneflow` | 2579 | 1533 |
| `cfnet-sceneflow` | 5992 | 5589 |
| `coex-sceneflow` | 1387 | 1448 |
| `crestereo-eth3d` | 1629 | 1626 |
| `defom-stereo-vitl-sceneflow` | 5088 | – |
| `dlnr-middlebury` | 1981 | 1932 |
| `fast-acvnet-generalization` | 1432 | – |
| `foundationstereo-vit-l` | 8229 | 4485 |
| `great-igev-eth3d` | 3559 | 2845 |
| `gwcnet-g-sceneflow` | 2531 | – |
| `hitnet-eth3d` | 1408 | 1408 |
| `igev-plusplus-eth3d` | 2960 | 1790 |
| `igev-stereo-eth3d` | 1671 | – |
| `las2-h` | 1564 | – |
| `mocha-v1-sceneflow` | 2486 | 2399 |
| `monster-mix-all` | 4176 | 3494 |
| `monster-plusplus-mix-all` | 4212 | 3493 |
| `nmrf-sceneflow` | 2430 | – |
| `psmnet-sceneflow` | 2448 | – |
| `raft-stereo-eth3d` | 1548 | 1507 |
| `rt-monster-plusplus-mix-all` | 1916 | – |
| `s2m2-l` | 3959 | 3955 |
| `selective-igev-eth3d-finetune` | 1704 | 1703 |
| `waft-stereo-b` | 3362 | – |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).
Note: float16 is not validated on MPS for this family (card mps_dtypes).

### Stereo, synthetic, 384x1248

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp16 (ms) | kernels vs fastest portable |
|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 457 | – | 410 | 1.1x (vs fp32) |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 319 | – | 246 | 1.3x (vs fp32) |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 1065 | – | 1012 | 1.1x (vs fp32) |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | – | 2228 | 758 | 2.9x (vs fp16) |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | – | 1265 | 1104 | 1.1x (vs fp16) |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 3505 | – | 2627 | 1.3x (vs fp32) |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | – | 7825 | 7383 | 1.1x (vs fp16) |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | – | 3366 | 2869 | 1.2x (vs fp16) |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 206 | – | 74.0 | 2.8x (vs fp32) |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | 448 | – | 175 | 2.6x (vs fp32) |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | – | 2852 | 2054 | 1.4x (vs fp16) |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 2090 | – | 1354 | 1.5x (vs fp32) |
| [`las2-h`](../models/stereo.md#las2) | 418 | – | 266 | 1.6x (vs fp32) |
| [`las2-s`](../models/stereo.md#las2) | 94.0 | – | 85.0 | 1.1x (vs fp32) |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | – | 2734 | 2277 | 1.2x (vs fp16) |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | – | 3518 | 2928 | 1.2x (vs fp16) |
| [`monster-sceneflow`](../models/stereo.md#monster) | – | 3344 | 2756 | 1.2x (vs fp16) |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 455 | – | 365 | 1.2x (vs fp32) |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | – | 1273 | 1192 | 1.1x (vs fp16) |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | – | 2292 | 1878 | 1.2x (vs fp16) |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 | kernels fp16 |
|---|---|---|---|
| `aanet-plus-sceneflow` | 2650 | – | 1585 |
| `aanet-sceneflow` | 1484 | – | 1466 |
| `bridgedepth-sf` | 4145 | – | 4093 |
| `cfnet-sceneflow` | – | 5463 | 2931 |
| `crestereo-eth3d` | – | 1597 | 1642 |
| `defom-stereo-vitl-sceneflow` | 6120 | – | 4069 |
| `foundationstereo-vit-s` | – | 4304 | 4284 |
| `great-igev-sceneflow` | – | 3416 | 3463 |
| `hitnet-eth3d` | 2444 | – | 1409 |
| `hitnet-middlebury-d400` | 4539 | – | 1454 |
| `igev-plusplus-sceneflow` | – | 1783 | 1803 |
| `igev-stereo-sceneflow` | 2752 | – | 1665 |
| `las2-h` | 2625 | – | 1642 |
| `las2-s` | 1501 | – | 1542 |
| `mocha-v1-sceneflow` | – | 2403 | 2382 |
| `monster-plusplus-mix-all` | – | 3490 | 3492 |
| `monster-sceneflow` | – | 3491 | 4093 |
| `nmrf-sceneflow` | 2408 | – | 1425 |
| `raft-stereo-sceneflow` | – | 2604 | 2538 |
| `selective-igev-sceneflow` | – | 1727 | 1704 |

Protocol: fastforks bench --device mps --size 384x1248, each mode in its own process: median of 5 warm calls; peak = the process's peak footprint.
Note: portable dtype = the fastest validated one.

### Stereo, synthetic, 994x1482

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp16 (ms) | kernels vs fastest portable |
|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 990 | – | 818 | 1.2x (vs fp32) |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 912 | – | 631 | 1.4x (vs fp32) |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 4375 | – | 3756 | 1.2x (vs fp32) |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 4280 | – | 3693 | 1.2x (vs fp32) |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 4881 | – | unsupported | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | OOM | – | unsupported | – |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 4747 | 2489 | unsupported | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 5490 | 2995 | unsupported | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | OOM | OOM | 2206 | – |
| [`coex-sceneflow`](../models/stereo.md#coex) | 784 | 411 | unsupported | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 4771 | 3746 | 3342 | 1.1x (vs fp16) |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 11105 | – | 8224 | 1.4x (vs fp32) |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 2694 | 2143 | unsupported | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 2699 | 2143 | unsupported | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 944 | – | unsupported | – |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | 942 | – | unsupported | – |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | 943 | – | unsupported | – |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | 1322 | – | unsupported | – |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | 1321 | – | unsupported | – |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | 1322 | – | unsupported | – |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | 1322 | – | unsupported | – |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | 942 | – | unsupported | – |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 5195 | – | unsupported | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | OOM | 25647 | 24480 | 1.0x (vs fp16) |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | OOM | 20584 | 19496 | 1.1x (vs fp16) |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | OOM | 11459 | 9636 | 1.2x (vs fp16) |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | 31916 | 11076 | 9634 | 1.1x (vs fp16) |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | OOM | 26930 | 25368 | 1.1x (vs fp16) |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | 32187 | 11062 | 9861 | 1.1x (vs fp16) |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 4340 | – | unsupported | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | 4461 | – | unsupported | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 484 | 501 | 136 | 3.6x (vs fp32) |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | 1155 | 1227 | 440 | 2.6x (vs fp32) |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 10444 | 8613 | 6139 | 1.4x (vs fp16) |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 10398 | 8606 | 6124 | 1.4x (vs fp16) |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 10489 | 8597 | 6125 | 1.4x (vs fp16) |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 10399 | 8595 | 6130 | 1.4x (vs fp16) |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 10424 | 8602 | 6135 | 1.4x (vs fp16) |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 6407 | – | 3998 | 1.6x (vs fp32) |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 6411 | – | 4006 | 1.6x (vs fp32) |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 6407 | – | 4008 | 1.6x (vs fp32) |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 6408 | – | 4007 | 1.6x (vs fp32) |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 6423 | – | 4001 | 1.6x (vs fp32) |
| [`las2-h`](../models/stereo.md#las2) | 1271 | – | 734 | 1.7x (vs fp32) |
| [`las2-l`](../models/stereo.md#las2) | 576 | – | 519 | 1.1x (vs fp32) |
| [`las2-m`](../models/stereo.md#las2) | 300 | – | 281 | 1.1x (vs fp32) |
| [`las2-s`](../models/stereo.md#las2) | 245 | – | 198 | 1.2x (vs fp32) |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 9065 | 7967 | 6693 | 1.2x (vs fp16) |
| [`mocha-v2-sceneflow`](../models/stereo.md#mocha_stereo) | 8887 | 7766 | 6501 | 1.2x (vs fp16) |
| [`monster-mix-all`](../models/stereo.md#monster) | 13423 | 11377 | 9646 | 1.2x (vs fp16) |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 14519 | 11876 | 10202 | 1.2x (vs fp16) |
| [`monster-sceneflow`](../models/stereo.md#monster) | 13426 | 11368 | 9642 | 1.2x (vs fp16) |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 1560 | – | 1204 | 1.3x (vs fp32) |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 4615 | – | unsupported | – |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 4396 | 3937 | 3638 | 1.1x (vs fp16) |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 4395 | 3940 | 3634 | 1.1x (vs fp16) |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 4396 | 3938 | 3637 | 1.1x (vs fp16) |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 663 | – | unsupported | – |
| [`s2m2-l`](../models/stereo.md#s2m2) | 8709 | 6590 | unsupported | – |
| [`s2m2-m`](../models/stereo.md#s2m2) | 4714 | 3620 | unsupported | – |
| [`s2m2-s`](../models/stereo.md#s2m2) | 1707 | 1607 | unsupported | – |
| [`s2m2-xl`](../models/stereo.md#s2m2) | OOM | 13398 | unsupported | – |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 9066 | 6991 | 5708 | 1.2x (vs fp16) |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | 9010 | 6969 | 5710 | 1.2x (vs fp16) |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | 9019 | 6971 | 5709 | 1.2x (vs fp16) |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | 9016 | 6963 | 5710 | 1.2x (vs fp16) |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | 9026 | 6989 | 5778 | 1.2x (vs fp16) |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | 9100 | 7206 | 5738 | 1.3x (vs fp16) |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 9010 | 6968 | 5710 | 1.2x (vs fp16) |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | 11022 | – | unsupported | – |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | 3360 | – | unsupported | – |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 | kernels fp16 |
|---|---|---|---|
| `aanet-plus-sceneflow` | 4987 | – | 3754 |
| `aanet-sceneflow` | 3604 | – | 2606 |
| `bridgedepth-rvc-pretrain` | 8198 | – | 5100 |
| `bridgedepth-sf` | 8300 | – | 6270 |
| `c-fast-foundationstereo` | 12882 | – | – |
| `c-foundationstereo-s` | – | – | – |
| `cas-gwcnet-c-sceneflow` | 7854 | 5132 | – |
| `cas-gwcnet-gc-sceneflow` | 11920 | 7182 | – |
| `cfnet-sceneflow` | – | – | 5416 |
| `coex-sceneflow` | 5183 | 3576 | – |
| `crestereo-eth3d` | 3776 | 2745 | 2804 |
| `defom-stereo-vitl-sceneflow` | 13138 | – | 8328 |
| `dlnr-middlebury` | 6166 | 5148 | – |
| `dlnr-sceneflow` | 6162 | 4094 | – |
| `fast-acvnet-generalization` | 3646 | – | – |
| `fast-acvnet-kitti12` | 3647 | – | – |
| `fast-acvnet-kitti15` | 3646 | – | – |
| `fast-acvnet-plus-generalization` | 3940 | – | – |
| `fast-acvnet-plus-kitti12` | 3938 | – | – |
| `fast-acvnet-plus-kitti15` | 3940 | – | – |
| `fast-acvnet-plus-sceneflow` | 3941 | – | – |
| `fast-acvnet-sceneflow` | 3646 | – | – |
| `fast-foundationstereo` | 13014 | – | – |
| `foundationstereo-vit-l` | – | 13761 | 13277 |
| `foundationstereo-vit-s` | – | 13617 | 12980 |
| `great-igev-eth3d` | – | 10531 | 10133 |
| `great-igev-kitti` | 12669 | 10538 | 10130 |
| `great-igev-middlebury` | – | 11544 | 12420 |
| `great-igev-sceneflow` | 12709 | 9960 | 10126 |
| `gwcnet-g-sceneflow` | 10663 | – | – |
| `gwcnet-gc-sceneflow` | 12397 | – | – |
| `hitnet-eth3d` | 5503 | 5510 | 2486 |
| `hitnet-middlebury-d400` | 10563 | 9344 | 4591 |
| `igev-plusplus-eth3d` | 9127 | 6040 | 4971 |
| `igev-plusplus-kitti12` | 9130 | 6034 | 4967 |
| `igev-plusplus-kitti15` | 9139 | 6034 | 4967 |
| `igev-plusplus-middlebury` | 9138 | 6037 | 4966 |
| `igev-plusplus-sceneflow` | 9132 | 6037 | 4972 |
| `igev-stereo-eth3d` | 5874 | – | 3789 |
| `igev-stereo-kitti12` | 5870 | – | 3791 |
| `igev-stereo-kitti15` | 5871 | – | 3791 |
| `igev-stereo-middlebury` | 5873 | – | 3792 |
| `igev-stereo-sceneflow` | 5872 | – | 3793 |
| `las2-h` | 5100 | – | 3771 |
| `las2-l` | 3869 | – | 2818 |
| `las2-m` | 4031 | – | 2666 |
| `las2-s` | 3983 | – | 2620 |
| `mocha-v1-sceneflow` | 8748 | 5765 | 5695 |
| `mocha-v2-sceneflow` | 7682 | 5750 | 4643 |
| `monster-mix-all` | 8839 | 6320 | 6330 |
| `monster-plusplus-mix-all` | 11077 | 7369 | 7347 |
| `monster-sceneflow` | 8952 | 6320 | 6299 |
| `nmrf-sceneflow` | 5749 | – | 3806 |
| `psmnet-sceneflow` | 11257 | – | – |
| `raft-stereo-eth3d` | 4714 | 5720 | 4721 |
| `raft-stereo-middlebury` | 4712 | 4695 | 4722 |
| `raft-stereo-sceneflow` | 4713 | 4695 | 4720 |
| `rt-monster-plusplus-mix-all` | 6986 | – | – |
| `s2m2-l` | 12178 | 9213 | – |
| `s2m2-m` | 11572 | 7941 | – |
| `s2m2-s` | 8051 | 5991 | – |
| `s2m2-xl` | – | 12386 | – |
| `selective-igev-eth3d-finetune` | 6961 | 3854 | 3904 |
| `selective-igev-eth3d-train` | 6961 | 3862 | 3907 |
| `selective-igev-kitti12` | 6960 | 3853 | 3906 |
| `selective-igev-kitti15` | 6961 | 3853 | 3905 |
| `selective-igev-middlebury-finetune` | 6960 | 3852 | 3907 |
| `selective-igev-middlebury-train` | 6965 | 3854 | 3907 |
| `selective-igev-sceneflow` | 6961 | 3859 | 3907 |
| `waft-stereo-b` | 12854 | – | – |
| `waft-stereo-s` | 9608 | – | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

## Depth

### Depth, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 939 | 935 |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 368 | 373 |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 199 | unsupported |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 588 | unsupported |
| [`depth-pro`](../models/depth.md#depth_pro) | 5598 | 5117 |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 3451 | 2882 |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 1811 | unsupported |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 581 | 590 |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 1487 | unsupported |
| [`moge-3-vitl`](../models/depth.md#moge) | 3975 | unsupported |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 611 | unsupported |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1151 | unsupported |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 734 | 709 |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `da3-giant-1.1` | 6565 | 6567 |
| `da3-large-1.1` | 3480 | 2438 |
| `depth-anything-v2-b` | 2011 | – |
| `depth-anything-v2-l` | 3952 | – |
| `depth-pro` | 11224 | 6086 |
| `marigold-depth-v1-1` | 9757 | 6701 |
| `metric3d-vit-large` | 3445 | – |
| `midas-dpt-beit-large-512` | 3385 | 2351 |
| `moge-2-vitl-normal` | 3892 | – |
| `moge-3-vitl` | 6330 | – |
| `unidepth-v2-vitl14` | 4003 | – |
| `unik3d-vitl` | 3986 | – |
| `zoedepth-nk` | 3499 | 2454 |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).
Note: float16 is not validated on MPS for this family (card mps_dtypes).

### Depth, synthetic, 994x1482

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp32 (ms) | kernels fp16 (ms) | kernels vs fastest portable |
|---|---|---|---|---|---|
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 833 | 759 | – | unsupported | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 334 | 327 | – | unsupported | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 332 | 318 | – | unsupported | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 277 | 272 | – | unsupported | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 1248 | 1184 | – | unsupported | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 212 | – | – | unsupported | – |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 622 | – | – | unsupported | – |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 87.3 | – | – | unsupported | – |
| [`depth-pro`](../models/depth.md#depth_pro) | 5879 | 5026 | – | unsupported | – |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 2970 | 2498 | – | unsupported | – |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 1822 | – | – | unsupported | – |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 679 | 635 | – | unsupported | – |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 1535 | – | 1454 | – | 1.1x (vs fp32) |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 363 | – | 363 | – | 1.0x (vs fp32) |
| [`moge-3-vitl`](../models/depth.md#moge) | 3948 | – | 2764 | – | 1.4x (vs fp32) |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 1260 | – | – | unsupported | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1217 | – | – | unsupported | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 778 | 785 | – | unsupported | – |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 | kernels fp32 |
|---|---|---|---|
| `da3-giant-1.1` | 6613 | 7243 | – |
| `da3-large-1.1` | 3541 | 2474 | – |
| `da3metric-large` | 2493 | 2474 | – |
| `da3mono-large` | 2491 | 2472 | – |
| `da3nested-giant-large-1.1` | 7700 | 8136 | – |
| `depth-anything-v2-b` | 2054 | – | – |
| `depth-anything-v2-l` | 3975 | – | – |
| `depth-anything-v2-s` | 1522 | – | – |
| `depth-pro` | 11267 | 7172 | – |
| `marigold-depth-v1-1` | 9709 | 6640 | – |
| `metric3d-vit-large` | 4523 | – | – |
| `midas-dpt-beit-large-512` | 3443 | 2407 | – |
| `moge-2-vitl-normal` | 3915 | – | 3992 |
| `moge-2-vits-normal` | 2631 | – | 2702 |
| `moge-3-vitl` | 6235 | – | 6119 |
| `unidepth-v2-vitl14` | 6075 | – | – |
| `unik3d-vitl` | 4001 | – | – |
| `zoedepth-nk` | 3558 | 3550 | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

## Classification

### Classification, synthetic, 224x224

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp16 (ms) |
|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 4.91 | 3.04 | unsupported |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 36.1 | – | unsupported |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 16.9 | – | unsupported |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 25.4 | – | unsupported |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 64.5 | – | unsupported |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 13.0 | – | unsupported |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 38.3 | – | unsupported |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 34.0 | – | unsupported |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 18.6 | – | unsupported |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 267 | 235 | unsupported |
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 16.8 | – | unsupported |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 17.4 | – | unsupported |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 6.48 | – | unsupported |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 8.54 | – | unsupported |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 4.29 | – | unsupported |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 9.58 | – | unsupported |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 38.9 | 33.1 | unsupported |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 25.7 | – | unsupported |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 4.67 | – | unsupported |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 11.2 | – | unsupported |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 6.33 | – | unsupported |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 3.79 | 3.72 | unsupported |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 28.2 | – | unsupported |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 13.8 | 11.7 | unsupported |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 16.9 | – | unsupported |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 214 | 197 | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `alexnet.tv_in1k` | 1566 | 1567 |
| `convnextv2_base.fcmae_ft_in22k_in1k` | 2301 | – |
| `deit_base_patch16_224.fb_in1k` | 1314 | – |
| `densenet121.tv_in1k` | 708 | – |
| `dinov2_vitl14_lc` | 4419 | – |
| `efficientnet_b0.tv_in1k` | 463 | – |
| `efficientnet_b4.tv_in1k` | 1660 | – |
| `efficientnet_v2_s.tv_in1k` | 616 | – |
| `efficientvit_b2.r288_in1k` | 1698 | – |
| `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 4096 | 4094 |
| `fastvit_sa24.apple_dist_in1k` | 656 | – |
| `inception_v3.tv_in1k` | 746 | – |
| `mobilenet_v2.tv2_in1k` | 422 | – |
| `mobilenet_v3_large.tv2_in1k` | 462 | – |
| `mobilenetv1_100.ra4_e3600_r224_in1k` | 437 | – |
| `mobilenetv4_conv_medium.e500_r256_in1k` | 581 | – |
| `regnety_160.tv2_in1k` | 2005 | 2030 |
| `resnet152.tv2_in1k` | 1079 | – |
| `resnet18.tv_in1k` | 454 | – |
| `resnet50.tv2_in1k` | 643 | – |
| `shufflenet_v2_x1_0.tv_in1k` | 374 | – |
| `squeezenet1_1.tv_in1k` | 341 | 340 |
| `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 2273 | – |
| `vgg16.tv_in1k` | 2435 | 2446 |
| `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 1314 | – |
| `vit_so400m_patch14_siglip_378.webli_ft_in1k` | 5547 | 5547 |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

### Classification, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 5.30 | 4.50 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 36.6 | unsupported |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 18.1 | unsupported |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 26.5 | unsupported |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 65.8 | unsupported |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 12.9 | unsupported |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 34.7 | unsupported |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 20.6 | unsupported |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 266 | 236 |
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 18.0 | unsupported |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 18.2 | unsupported |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 7.10 | unsupported |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 10.5 | unsupported |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 5.10 | unsupported |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 10.2 | unsupported |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 40.0 | 33.8 |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 25.9 | unsupported |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 12.9 | unsupported |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 7.10 | unsupported |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 4.40 | 4.30 |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 29.0 | unsupported |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 14.5 | 12.5 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 18.1 | unsupported |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 216 | 197 |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `alexnet.tv_in1k` | 1590 | 1583 |
| `convnextv2_base.fcmae_ft_in22k_in1k` | 2317 | – |
| `deit_base_patch16_224.fb_in1k` | 1330 | – |
| `densenet121.tv_in1k` | 727 | – |
| `dinov2_vitl14_lc` | 4435 | – |
| `efficientnet_b0.tv_in1k` | 487 | – |
| `efficientnet_v2_s.tv_in1k` | 633 | – |
| `efficientvit_b2.r288_in1k` | 1710 | – |
| `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | 4108 | 4106 |
| `fastvit_sa24.apple_dist_in1k` | 669 | – |
| `inception_v3.tv_in1k` | 766 | – |
| `mobilenet_v2.tv2_in1k` | 444 | – |
| `mobilenet_v3_large.tv2_in1k` | 481 | – |
| `mobilenetv1_100.ra4_e3600_r224_in1k` | 456 | – |
| `mobilenetv4_conv_medium.e500_r256_in1k` | 601 | – |
| `regnety_160.tv2_in1k` | 2025 | 2038 |
| `resnet152.tv2_in1k` | 1101 | – |
| `resnet50.tv2_in1k` | 660 | – |
| `shufflenet_v2_x1_0.tv_in1k` | 386 | – |
| `squeezenet1_1.tv_in1k` | 359 | 356 |
| `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | 2283 | – |
| `vgg16.tv_in1k` | 2453 | 2459 |
| `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 1331 | – |
| `vit_so400m_patch14_siglip_378.webli_ft_in1k` | 5563 | 5560 |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).
Note: network at the model's native crop (the model resizes the input).
Note: network at the model's native crop (the model resizes the input).
Note: float16 is not validated on MPS for this family (card mps_dtypes).

## Segmentation

### Segmentation, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 415 | 354 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 267 | – |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 615 | 548 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 587 | 639 |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 661 | 697 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 4287 | 2791 |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 64.7 | – |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `deeplabv3-r101-d8-ade20k` | 2133 | 2035 |
| `dinov2-b-linear-ade20k` | 1974 | – |
| `eomt-l-dinov2-ade20k` | 6298 | 4257 |
| `mask2former-swinl-ade20k` | 5599 | 4473 |
| `oneformer-swinl-ade20k` | 6174 | 4512 |
| `sapiens-seg-0.3b-goliath` | 4522 | 2813 |
| `segformer-b0-ade20k` | 1522 | – |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).

### Segmentation, synthetic, 1024x2048

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp16 (ms) |
|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 721 | 528 | unsupported |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 516 | 430 | unsupported |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 409 | – | unsupported |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 1032 | – | unsupported |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 224 | – | unsupported |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 608 | 540 | unsupported |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 648 | 575 | unsupported |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 468 | 405 | unsupported |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 263 | 228 | unsupported |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 909 | 964 | unsupported |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 1127 | 1161 | unsupported |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 486 | 411 | unsupported |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 2988 | 1517 | unsupported |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 5099 | 3113 | unsupported |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 106 | – | unsupported |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 428 | – | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `deeplabv3-r101-d8-ade20k` | 3139 | 2010 |
| `deeplabv3plus-r101-d8-ade20k` | 2994 | 1891 |
| `dinov2-b-linear-ade20k` | 3025 | – |
| `dinov2-l-linear-ade20k` | 4120 | – |
| `dinov2-s-linear-ade20k` | 2529 | – |
| `eomt-l-dinov2-ade20k` | 6308 | 4262 |
| `eomt-l-dinov3-ade20k` | 6310 | 4259 |
| `fcn-r101-d8-ade20k` | 2978 | 1875 |
| `hrnet-w48-fcn-ade20k` | 3210 | 2044 |
| `mask2former-swinl-ade20k` | 6661 | 4503 |
| `oneformer-swinl-ade20k` | 9492 | 5559 |
| `pspnet-r101-d8-ade20k` | 3083 | 1945 |
| `sapiens-seg-0.3b-goliath` | 4766 | 3748 |
| `sapiens-seg-1b-goliath` | 9794 | 6340 |
| `segformer-b0-ade20k` | 1523 | – |
| `segformer-b5-ade20k` | 3113 | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

## Promptable segmentation

### Promptable segmentation, synthetic, 480x640

| model | portable fp32 (ms) |
|---|---|
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 635 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 428 |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `sam-vit-b` | 4350 |
| `sam2.1-hiera-base-plus` | 3951 |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).

### Promptable segmentation, synthetic, 1024x1024

| model | portable fp32 (ms) | kernels fp16 (ms) |
|---|---|---|
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 631 | unsupported |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 2518 | unsupported |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1373 | unsupported |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 424 | unsupported |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 613 | unsupported |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 177 | unsupported |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 151 | unsupported |
| [`sam3`](../models/promptable_segmentation.md#sam3) | unsupported | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `sam-vit-b` | 4362 |
| `sam-vit-h` | 8821 |
| `sam-vit-l` | 7227 |
| `sam2.1-hiera-base-plus` | 3973 |
| `sam2.1-hiera-large` | 2950 |
| `sam2.1-hiera-small` | 1597 |
| `sam2.1-hiera-tiny` | 1536 |
| `sam3` | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).
Note: the family is not validated on MPS (card mps_dtypes).

## Detection

### Detection, synthetic, 480x640

| model | portable fp32 (ms) |
|---|---|
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 189 |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `faster-rcnn-r50-fpn-3x-coco` | 1932 |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).

### Detection, synthetic, 800x1344

| model | portable fp32 (ms) | kernels fp16 (ms) |
|---|---|---|
| [`detr-r101-coco`](../models/detection.md#detr) | 183 | unsupported |
| [`detr-r50-coco`](../models/detection.md#detr) | 125 | unsupported |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 214 | unsupported |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 222 | unsupported |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 82.6 | unsupported |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 55.3 | unsupported |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 67.9 | unsupported |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 44.0 | unsupported |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 114 | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `detr-r101-coco` | 2181 |
| `detr-r50-coco` | 1995 |
| `faster-rcnn-r50-fpn-3x-coco` | 1930 |
| `mask-rcnn-r50-fpn-3x-coco` | 1930 |
| `rtdetrv2-l-coco` | 1927 |
| `rtdetrv2-m-coco` | 1776 |
| `rtdetrv2-ms-coco` | 1872 |
| `rtdetrv2-s-coco` | 1650 |
| `rtdetrv2-x-coco` | 2189 |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

## Features

### Features, synthetic, 224x224

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp16 (ms) |
|---|---|---|---|
| [`dinov2-vitb14`](../models/features.md#dinov2) | 23.3 | 21.7 | unsupported |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 44.3 | 42.8 | unsupported |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 221 | 193 | unsupported |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 263 | 235 | unsupported |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 68.0 | 59.4 | unsupported |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 96.1 | 86.3 | unsupported |
| [`dinov2-vits14`](../models/features.md#dinov2) | 12.2 | 11.2 | unsupported |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 22.2 | 22.1 | unsupported |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | unsupported | – | unsupported |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | unsupported | – | unsupported |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | unsupported | – | unsupported |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | unsupported | – | unsupported |
| [`dinov3-vitb16`](../models/features.md#dinov3) | unsupported | – | unsupported |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | unsupported | – | unsupported |
| [`dinov3-vitl16`](../models/features.md#dinov3) | unsupported | – | unsupported |
| [`dinov3-vits16`](../models/features.md#dinov3) | unsupported | – | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `dinov2-vitb14` | 880 | 774 |
| `dinov2-vitb14-reg` | 887 | 775 |
| `dinov2-vitg14` | 7151 | 4502 |
| `dinov2-vitg14-reg` | 7150 | 4502 |
| `dinov2-vitl14` | 3070 | 1562 |
| `dinov2-vitl14-reg` | 3071 | 1563 |
| `dinov2-vits14` | 381 | 341 |
| `dinov2-vits14-reg` | 382 | 344 |
| `dinov3-convnext-base` | – | – |
| `dinov3-convnext-large` | – | – |
| `dinov3-convnext-small` | – | – |
| `dinov3-convnext-tiny` | – | – |
| `dinov3-vitb16` | – | – |
| `dinov3-vith16plus` | – | – |
| `dinov3-vitl16` | – | – |
| `dinov3-vits16` | – | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).
Note: the family is not validated on MPS (card mps_dtypes).

### Features, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`dinov2-vitb14`](../models/features.md#dinov2) | 130 | 116 |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `dinov2-vitb14` | 1964 | 819 |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).
Note: network at the model's native crop (the model resizes the input).
Note: network at the model's native crop (the model resizes the input).

## Multiview

### Multiview, synthetic, 336x518

| model | portable fp32 (ms) | portable fp16 (ms) | kernels fp32 (ms) | kernels fp16 (ms) | kernels vs fastest portable |
|---|---|---|---|---|---|
| [`mapanything-apache`](../models/multiview.md#mapanything) | 7357 | 6369 | – | unsupported | – |
| [`vggt-1b`](../models/multiview.md#vggt) | 7209 | – | 7016 | – | 1.0x (vs fp32) |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | unsupported | – | – | unsupported | – |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 | kernels fp32 |
|---|---|---|---|
| `mapanything-apache` | 11638 | 7218 | – |
| `vggt-1b` | 10654 | – | 10692 |
| `vggt-omega-1b-512` | – | – | – |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).
Note: the family is not validated on MPS (card mps_dtypes).

### Multiview, synthetic, 448x672

| model | portable fp32 (ms) | kernels fp16 (ms) |
|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | 1051 | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `mvsanywhere` | 3534 |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).

### Multiview, synthetic, 480x640

| model | portable fp32 (ms) | portable fp16 (ms) |
|---|---|---|
| [`mapanything-apache`](../models/multiview.md#mapanything) | 3507 | 3138 |
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | 770 | – |
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 184 | – |
| [`vggt-1b`](../models/multiview.md#vggt) | 4072 | – |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 | portable fp16 |
|---|---|---|
| `mapanything-apache` | 8525 | 5482 |
| `mvsanywhere` | 2507 | – |
| `patchmatchnet` | 1379 | – |
| `vggt-1b` | 9555 | – |

Protocol: fastforks bench --device mps --size 480x640 --views 4: median of 5 warm calls (2 for calls over 2 s); peak = the process's peak footprint (Activity Monitor Memory).

### Multiview, synthetic, 1536x1024

| model | portable fp32 (ms) | kernels fp16 (ms) |
|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 1010 | unsupported |

Memory, peak process footprint (MiB, model + activations in unified memory):

| model | portable fp32 |
|---|---|
| `patchmatchnet` | 7593 |

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).
Note: no Metal kernels for this family (card mps_kernels).
<!-- device:end -->

## Notes on `kernels`

The 384x1248 stereo rows compare `kernels` (float16; Metal ConvGRUs, correlation / geometry lookups, cost volumes,
warp costs, deformable sampling, window attention) with the fastest validated `portable` dtype, each run in its own
process (median of 5, peak footprint).

Most of the IGEV family's gain is the lookup (the portable one gathers from each pyramid level separately); the fused
ConvGRUs add about 5 % (Selective-IGEV and MoCha get the lookup only: a selective GRU and an LSTM). IGEV-Stereo
`portable` is validated in float32 only on MPS, hence its larger factor (as DEFOM-Stereo's: fused lookup and ConvGRUs
in float16). CFNet's gain is the warped volumes of its refinement stages built by one kernel (and 2.5 GB less memory);
HITNet's is the warp costs of its tile hypotheses, which `portable` builds from many small ops. FoundationStereo's
Metal geometry lookup gains little: its hourglass and transformer dominate. NMRF and BridgeDepth get their window
attention with relative-position embeddings as one kernel (2.5-4x faster than the float16 module); BridgeDepth's time is
mostly its ViT and alignment layers.

RAFT-Stereo `kernels` (Metal ConvGRU and correlation lookup, float16) against float16 `portable`, interleaved runs in
separate processes: 1.08x at 480x640, 1.12x at 992x1440, about 30 MiB less. Its convolutions already run near the M1
Pro's peak, so fusing the elementwise steps gains little.

Both modes, in every dtype, replace the transposed convolutions MPS runs slowly: a stride-2 one as one convolution per
output parity (four `conv2d` or eight `conv3d` calls; MPS has no float16 3-D kernel at all), a 2-D one whose kernel
equals its stride (ViT necks, DPT heads) as a 1 x 1 convolution and `pixel_shuffle`. End to end at 384x1248 that makes
CAS-GwcNet float16 2.3x, CoEx float16 1.9x, GwcNet 1.3x and PSMNet 1.15x faster, and Sapiens 1.9x at 1024x768.
