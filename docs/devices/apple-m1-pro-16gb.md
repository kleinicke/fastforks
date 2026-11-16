# Apple M1 Pro 16 GB

The original code, `portable` and Metal `kernels` on Apple silicon (MPS), each model at its task's standard input
(the input of the [reference tables](../benchmarks.md); multiview models at their own), so a row compares with the
same model on the RTX 5060 Ti. The tables are generated from the data file named below; the text around them is
written by hand.

Memory is the process's peak footprint (Activity Monitor's "Memory": model and activations, which share the system
RAM on a Mac). It is not comparable with the CUDA tables' torch peak memory. Not run: DINOv3 ViT-S/16,
ViT-L/16, ViT-H+/16 and DINOv3-ConvNeXt Tiny / Base / Large (`failed`: their gated weights are not on this Mac; the
family's ViT-B/16 and ConvNeXt-Small rows pass the goldens on MPS) and D-FINE (no rows: weights not downloaded). Reproduce or extend:
`uv run python tools/speed_survey.py run --device apple-m1-pro-16gb` (docs/data/README.md, "Devices").

Accuracy against the CPU (float32, Middlebury ArtL for stereo): MPS float32 outputs match to 1e-3 px mean or better
(most families 1e-5), except MonSter++ (0.006 px) and NMRF (0.03 px on non-occluded pixels, about 0.7 px in
occlusions, where its local-maximum test flips on last-bit differences in flat cost curves; the error against ground
truth is unchanged). float16 moves stereo outputs by 0.003-0.07 px mean. Depth: float32 matches the CPU to 1e-4 of the
median depth, float16 moves it by 0.02-0.12 %.

Pose and Perception Encoder Metal `kernels` (added 2026-11-09, not yet in the tables below; median ms per call,
`kernels` vs `portable` in the same dtype): RTMPose-t / -m / DWPose-l fp32 with 16 boxes 29.7 / 59.3 / 102.9 ->
18.6 / 45.1 / 80.6 (keypoints unchanged); ViTPose-B / ViTPose+-S fp16 with 16 boxes 192 / 79 -> 161 / 58 (one box:
22.2 / 13.1 -> 16.1 / 8.6); PE-Core-B/16 fp16 batch 8 112 -> 84, PE-Spatial-B/16 at 512 fp16 batch 8 620 -> 467.

Optical flow Metal `kernels` (added 2026-11-11, `flow_metal`): RAFT and SEA-RAFT take their correlation windows on
the fly from the pooled fmap2 pyramid instead of building the all-pairs volume. At 1080 x 1920, fp16 `kernels` vs fp32
`portable` (the fastest validated MPS path before): RAFT-large 3992 -> 1348 ms with peak memory 11.6 -> 4.7 GB,
RAFT-small 4513 -> 386 ms, SEA-RAFT-M 1820 -> 1220 ms, SEA-RAFT-S 2115 -> 833 ms; goldens pass. RAFT's remaining time
is the update block (about 62 ms per iteration) and the encoders, SEA-RAFT's mostly its encoders (fp16 gains nothing
there on MPS). The spring variants are faster by design: they halve the input first.

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed/apple-m1-pro-16gb.jsonl) -->

| | |
|---|---|
| device | Apple M1 Pro 16 GB |
| backend | mps |
| memory | 16 GB unified |
| hardware | 16-core GPU |
| software | macOS 27, torch 2.14 |
| torch (rows) | 2.14.1 |
| measured | 2026-11-06 to 2026-11-12 |
| rows | 922 in `docs/data/speed/apple-m1-pro-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 870 | 990 | – | 818 | 1.1x | 1.2x | 6805 | 4987 | 3754 |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 813 | 912 | – | 631 | 1.3x | 1.4x | 3710 | 3604 | 2606 |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 4416 | 4375 | – | 3756 | 1.2x | 1.2x | 7028 | 8198 | 5100 |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 4279 | 4280 | – | 3693 | 1.2x | 1.2x | 7464 | 8300 | 6270 |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | failed | 4881 | – | unsupported | – | – | – | 12882 | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | – | OOM | – | unsupported | – | – | – | – | – |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 7909 | 4747 | 2489 | unsupported | – | – | 9174 | 5132 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | OOM | 5490 | 2995 | unsupported | – | – | – | 7182 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | – | OOM | OOM | 2206 | – | – | – | – | 5416 |
| [`coex-sceneflow`](../models/stereo.md#coex) | 1024 | 784 | 411 | unsupported | – | – | 5473 | 3576 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | failed | 4771 | 3746 | 3342 | – | 1.1x | – | 2745 | 2804 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | failed | 11105 | – | 8224 | – | 1.4x | – | 13138 | 8328 |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | failed | 2694 | 2143 | unsupported | – | – | – | 5148 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | failed | 2699 | 2143 | unsupported | – | – | – | 4094 | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | failed | 944 | – | unsupported | – | – | – | 3646 | – |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | failed | 942 | – | unsupported | – | – | – | 3647 | – |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | failed | 943 | – | unsupported | – | – | – | 3646 | – |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | failed | 1322 | – | unsupported | – | – | – | 3940 | – |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | failed | 1321 | – | unsupported | – | – | – | 3938 | – |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | failed | 1322 | – | unsupported | – | – | – | 3940 | – |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | failed | 1322 | – | unsupported | – | – | – | 3941 | – |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | failed | 942 | – | unsupported | – | – | – | 3646 | – |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | failed | 5195 | – | unsupported | – | – | – | 13014 | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | failed | OOM | 25647 | 24480 | – | 1.0x | – | 13761 | 13277 |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | failed | OOM | 20584 | 19496 | – | 1.1x | – | 13617 | 12980 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | failed | OOM | 11459 | 9636 | – | 1.2x | – | 10531 | 10133 |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | failed | 31916 | 11076 | 9634 | – | 1.1x | – | 10538 | 10130 |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | failed | OOM | 26930 | 25368 | – | 1.1x | – | 11544 | 12420 |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | failed | 32187 | 11062 | 9861 | – | 1.1x | – | 9960 | 10126 |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | failed | 4340 | – | unsupported | – | – | – | 10663 | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | failed | 4461 | – | unsupported | – | – | – | 12397 | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | failed | 484 | 501 | 136 | – | 3.6x | – | 5503 | 2486 |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | failed | 1155 | 1227 | 440 | – | 2.6x | – | 10563 | 4591 |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 9851 | 10444 | 8613 | 6139 | 1.6x | 1.4x | 7730 | 6040 | 4971 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 9779 | 10398 | 8606 | 6124 | 1.6x | 1.4x | 7841 | 6034 | 4967 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 11619 | 10489 | 8597 | 6125 | 1.9x | 1.4x | 8295 | 6034 | 4967 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | – | 10399 | 8595 | 6130 | – | 1.4x | – | 6037 | 4966 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | – | 10424 | 8602 | 6135 | – | 1.4x | – | 6037 | 4972 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | – | 6407 | – | 3998 | – | 1.6x | – | 5874 | 3789 |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | – | 6411 | – | 4006 | – | 1.6x | – | 5870 | 3791 |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | – | 6407 | – | 4008 | – | 1.6x | – | 5871 | 3791 |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | – | 6408 | – | 4007 | – | 1.6x | – | 5873 | 3792 |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | – | 6423 | – | 4001 | – | 1.6x | – | 5872 | 3793 |
| [`las2-h`](../models/stereo.md#las2) | – | 1271 | – | 734 | – | 1.7x | – | 5100 | 3771 |
| [`las2-l`](../models/stereo.md#las2) | – | 576 | – | 519 | – | 1.1x | – | 3869 | 2818 |
| [`las2-m`](../models/stereo.md#las2) | – | 300 | – | 281 | – | 1.1x | – | 4031 | 2666 |
| [`las2-s`](../models/stereo.md#las2) | – | 245 | – | 198 | – | 1.2x | – | 3983 | 2620 |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | – | 9065 | 7967 | 6693 | – | 1.2x | – | 5765 | 5695 |
| [`mocha-v2-sceneflow`](../models/stereo.md#mocha_stereo) | – | 8887 | 7766 | 6501 | – | 1.2x | – | 5750 | 4643 |
| [`monster-mix-all`](../models/stereo.md#monster) | – | 13423 | 11377 | 9646 | – | 1.2x | – | 6320 | 6330 |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | – | 14519 | 11876 | 10202 | – | 1.2x | – | 7369 | 7347 |
| [`monster-sceneflow`](../models/stereo.md#monster) | – | 13426 | 11368 | 9642 | – | 1.2x | – | 6320 | 6299 |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | – | 1560 | – | 1204 | – | 1.3x | – | 5749 | 3806 |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | – | 4615 | – | unsupported | – | – | – | 11257 | – |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | – | 4396 | 3937 | 3638 | – | 1.1x | – | 5720 | 4721 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 5574 | 4395 | 3940 | 3634 | 1.5x | 1.1x | 9545 | 4695 | 4722 |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | – | 4396 | 3938 | 3637 | – | 1.1x | – | 4695 | 4720 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | – | 663 | – | unsupported | – | – | – | 6986 | – |
| [`s2m2-l`](../models/stereo.md#s2m2) | – | 8709 | 6590 | unsupported | – | – | – | 9213 | – |
| [`s2m2-m`](../models/stereo.md#s2m2) | – | 4714 | 3620 | unsupported | – | – | – | 7941 | – |
| [`s2m2-s`](../models/stereo.md#s2m2) | – | 1707 | 1607 | unsupported | – | – | – | 5991 | – |
| [`s2m2-xl`](../models/stereo.md#s2m2) | – | OOM | 13398 | unsupported | – | – | – | 12386 | – |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | – | 9066 | 6991 | 5708 | – | 1.2x | – | 3854 | 3904 |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | – | 9010 | 6969 | 5710 | – | 1.2x | – | 3862 | 3907 |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | – | 9019 | 6971 | 5709 | – | 1.2x | – | 3853 | 3906 |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | – | 9016 | 6963 | 5710 | – | 1.2x | – | 3853 | 3905 |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | – | 9026 | 6989 | 5778 | – | 1.2x | – | 3852 | 3907 |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | – | 9100 | 7206 | 5738 | – | 1.3x | – | 3854 | 3907 |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | – | 9010 | 6968 | 5710 | – | 1.2x | – | 3859 | 3907 |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | – | 11022 | – | unsupported | – | – | – | 12854 | – |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | – | 3360 | – | unsupported | – | – | – | 9608 | – |

Note: no Metal kernels for this family (card mps_kernels).

## Depth

### Depth, synthetic, 994x1482

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | failed | 833 | 759 | – | unsupported | – | – | 7243 | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 412 | 334 | 327 | – | unsupported | – | 4019 | 2474 | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | failed | 277 | 268 | – | unsupported | – | – | 2473 | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | failed | 277 | 272 | – | unsupported | – | – | 2472 | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | failed | 1248 | 1184 | – | unsupported | – | – | 8136 | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 239 | 212 | – | – | unsupported | – | 2342 | 2054 | – |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 697 | 622 | – | – | unsupported | – | 4218 | 3975 | – |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | failed | 211 | – | – | unsupported | – | – | 2056 | – |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | failed | 628 | – | – | unsupported | – | – | 3977 | – |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | failed | 87.8 | – | – | unsupported | – | – | 1523 | – |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | failed | 211 | – | – | unsupported | – | – | 2056 | – |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | failed | 623 | – | – | unsupported | – | – | 3974 | – |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | failed | 90.0 | – | – | unsupported | – | – | 1523 | – |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 113 | 87.3 | – | – | unsupported | – | 1860 | 1522 | – |
| [`depth-pro`](../models/depth.md#depth_pro) | failed | 6115 | 5011 | – | unsupported | – | – | 7169 | – |
| [`distill-any-depth-b`](../models/depth.md#depth_anything_v2) | failed | 212 | – | – | unsupported | – | – | 2056 | – |
| [`distill-any-depth-l`](../models/depth.md#depth_anything_v2) | failed | 622 | – | – | unsupported | – | – | 3977 | – |
| [`distill-any-depth-s`](../models/depth.md#depth_anything_v2) | failed | 87.3 | – | – | unsupported | – | – | 1525 | – |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | failed | 2970 | 2498 | – | unsupported | – | – | 6640 | – |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | failed | 1822 | – | – | unsupported | – | – | 4523 | – |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | failed | 679 | 635 | – | unsupported | – | – | 2407 | – |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | failed | 1343 | – | 1344 | – | 1.0x | – | 3914 | 3994 |
| [`moge-2-vits-normal`](../models/depth.md#moge) | failed | 361 | – | 362 | – | 1.0x | – | 2634 | 2704 |
| [`moge-3-vitl`](../models/depth.md#moge) | failed | 3469 | – | 2415 | – | 1.4x | – | 6239 | 6121 |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | failed | 1252 | – | – | unsupported | – | – | 6077 | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | failed | 1217 | – | – | unsupported | – | – | 4001 | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 1268 | 778 | 785 | – | unsupported | – | 4223 | 3558 | – |

Note: no Metal kernels for this family (card mps_kernels).

## Classification

### Classification, synthetic, 224x224

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 9.48 | 4.91 | 3.04 | unsupported | 1819 | 1567 |
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | failed | 27.7 | – | unsupported | – | 2312 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | failed | 27.4 | – | unsupported | – | 2311 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | failed | 50.8 | – | unsupported | – | 2310 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | failed | 41.3 | – | unsupported | – | 3491 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | failed | 41.3 | – | unsupported | – | 3490 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | failed | 88.7 | – | unsupported | – | 3495 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | failed | 19.1 | – | unsupported | – | 1042 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | failed | 19.4 | – | unsupported | – | 1043 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | failed | 33.5 | – | unsupported | – | 2059 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | failed | 11.2 | – | unsupported | – | 740 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | failed | 11.4 | – | unsupported | – | 738 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | failed | 20.4 | – | unsupported | – | 1764 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | failed | 36.1 | – | unsupported | – | 2301 |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | failed | 16.9 | – | unsupported | – | 1314 |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 31.5 | 25.4 | – | unsupported | 554 | 708 |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | failed | 64.5 | – | unsupported | – | 4419 |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 17.9 | 13.0 | – | unsupported | 442 | 463 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 54.0 | 38.3 | – | unsupported | 1676 | 1660 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 45.9 | 34.0 | – | unsupported | 638 | 616 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | failed | 18.6 | – | unsupported | – | 1698 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | failed | 267 | 235 | unsupported | – | 4094 |
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | failed | 16.8 | – | unsupported | – | 656 |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 27.7 | 17.4 | – | unsupported | 730 | 746 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 12.4 | 6.48 | – | unsupported | 396 | 422 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 13.4 | 8.54 | – | unsupported | 440 | 462 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | failed | 4.29 | – | unsupported | – | 437 |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | failed | 9.58 | – | unsupported | – | 581 |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 46.1 | 38.9 | 33.1 | unsupported | 2209 | 2030 |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 41.1 | 25.7 | – | unsupported | 1007 | 1079 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 7.25 | 4.67 | – | unsupported | 466 | 454 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 16.8 | 11.2 | – | unsupported | 669 | 643 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 12.1 | 6.33 | – | unsupported | 364 | 374 |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 4.03 | 3.79 | 3.72 | unsupported | 347 | 340 |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | failed | 28.2 | – | unsupported | – | 2273 |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 13.6 | 13.8 | 11.7 | unsupported | 2438 | 2446 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | failed | 16.9 | – | unsupported | – | 1314 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | failed | 214 | 197 | unsupported | – | 5547 |

Note: no Metal kernels for this family (card mps_kernels).

## Segmentation

### Segmentation, synthetic, 1024x2048

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | failed | 721 | 528 | unsupported | – | 2010 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | failed | 516 | 430 | unsupported | – | 1891 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 491 | 409 | – | unsupported | 3325 | 3025 |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 1235 | 1032 | – | unsupported | 5570 | 4120 |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 264 | 224 | – | unsupported | 2830 | 2529 |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | failed | 608 | 540 | unsupported | – | 4262 |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | failed | 648 | 575 | unsupported | – | 4259 |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | failed | 468 | 405 | unsupported | – | 1875 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | failed | 263 | 228 | unsupported | – | 2044 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | failed | 909 | 964 | unsupported | – | 6661 |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | failed | 1127 | 1161 | unsupported | – | 9492 |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | failed | 486 | 411 | unsupported | – | 1945 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | failed | 2988 | 1517 | unsupported | – | 3748 |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | failed | 5099 | 3113 | unsupported | – | 6340 |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | failed | 106 | – | unsupported | – | 1523 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | failed | 428 | – | unsupported | – | 3113 |

Note: no Metal kernels for this family (card mps_kernels).

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`efficientvit-sam-l0`](../models/promptable_segmentation.md#efficientvit_sam) | failed | 50.6 | – | unsupported | – | 1779 |
| [`efficientvit-sam-l1`](../models/promptable_segmentation.md#efficientvit_sam) | failed | 75.2 | – | unsupported | – | 1783 |
| [`efficientvit-sam-l2`](../models/promptable_segmentation.md#efficientvit_sam) | failed | 91.2 | – | unsupported | – | 2004 |
| [`efficientvit-sam-xl0`](../models/promptable_segmentation.md#efficientvit_sam) | failed | 167 | – | unsupported | – | 3032 |
| [`efficientvit-sam-xl1`](../models/promptable_segmentation.md#efficientvit_sam) | failed | 259 | – | unsupported | – | 3452 |
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | failed | 645 | – | unsupported | – | 4374 |
| [`sam-hq-vit-h`](../models/promptable_segmentation.md#sam_hq) | failed | 2633 | – | unsupported | – | 8759 |
| [`sam-hq-vit-l`](../models/promptable_segmentation.md#sam_hq) | failed | 1420 | – | unsupported | – | 6205 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 841 (golden fail) | 631 | – | unsupported | – | 4362 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 3210 (golden fail) | 2518 | – | unsupported | – | 8821 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1844 (golden fail) | 1373 | – | unsupported | – | 7227 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 543 (golden fail) | 424 | – | unsupported | – | 3973 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 722 (golden fail) | 613 | – | unsupported | – | 2950 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 217 (golden fail) | 177 | – | unsupported | – | 1597 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 193 (golden fail) | 151 | – | unsupported | – | 1536 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 2203 | 3620 | 3688 | unsupported | 4842 | 5807 |

Note: no Metal kernels for this family (card mps_kernels).

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`detr-r101-coco`](../models/detection.md#detr) | failed | 183 | – | unsupported | 2181 |
| [`detr-r50-coco`](../models/detection.md#detr) | failed | 125 | – | unsupported | 1995 |
| [`dfine-l-coco`](../models/detection.md#dfine) | failed | 78.8 | – | unsupported | 1839 |
| [`dfine-m-coco`](../models/detection.md#dfine) | failed | 62.0 | – | unsupported | 1701 |
| [`dfine-n-coco`](../models/detection.md#dfine) | failed | 24.8 | – | unsupported | 1518 |
| [`dfine-s-coco`](../models/detection.md#dfine) | failed | 42.7 | – | unsupported | 1568 |
| [`dfine-x-coco`](../models/detection.md#dfine) | failed | 111 | – | unsupported | 2384 |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | failed | 214 | – | unsupported | 1930 |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | failed | 222 | – | unsupported | 1930 |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | failed | 82.6 | – | unsupported | 1927 |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | failed | 55.3 | – | unsupported | 1776 |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | failed | 67.9 | – | unsupported | 1872 |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | failed | 44.0 | – | unsupported | 1650 |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | failed | 114 | – | unsupported | 2189 |
| [`rtmdet-l-coco`](../models/detection.md#rtmdet) | failed | 83.7 | 75.2 | unsupported | 1927 |
| [`rtmdet-m-coco`](../models/detection.md#rtmdet) | failed | 54.8 | 50.9 | unsupported | 1862 |
| [`rtmdet-m-person`](../models/detection.md#rtmdet) | failed | 53.0 | 48.5 | unsupported | 1737 |
| [`rtmdet-nano-person`](../models/detection.md#rtmdet) | failed | 22.6 | 22.0 | unsupported | 1403 |
| [`rtmdet-s-coco`](../models/detection.md#rtmdet) | failed | 33.6 | 30.7 | unsupported | 1619 |
| [`rtmdet-tiny-coco`](../models/detection.md#rtmdet) | failed | 28.6 | 27.0 | unsupported | 1545 |
| [`rtmdet-x-coco`](../models/detection.md#rtmdet) | failed | 127 | 115 | unsupported | 2107 |

Note: no Metal kernels for this family (card mps_kernels).

## Pose

### Pose, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`dwpose-l`](../models/pose.md#rtmpose) | failed | 99.1 | 87.6 | 92.0 | – | 1.0x | 1862 | 1917 |
| [`dwpose-l-384`](../models/pose.md#rtmpose) | failed | 197 | 173 | 182 | – | 1.0x | 1864 | 1916 |
| [`dwpose-m`](../models/pose.md#rtmpose) | failed | 69.1 | 59.1 | 57.7 | – | 1.0x | 1682 | 1791 |
| [`dwpose-s`](../models/pose.md#rtmpose) | failed | 41.6 | 38.3 | 33.2 | – | 1.2x | 1527 | 1595 |
| [`dwpose-t`](../models/pose.md#rtmpose) | failed | 33.4 | 33.1 | 26.4 | – | 1.3x | 1465 | 1539 |
| [`rtmpose-l`](../models/pose.md#rtmpose) | failed | 91.7 | 81.3 | 85.0 | – | 1.0x | 1849 | 1893 |
| [`rtmpose-l-384`](../models/pose.md#rtmpose) | failed | 180 | 158 | 167 | – | 0.9x | 1851 | 1893 |
| [`rtmpose-m`](../models/pose.md#rtmpose) | failed | 60.3 | 57.8 | 52.4 | – | 1.1x | 1633 | 1740 |
| [`rtmpose-m-384`](../models/pose.md#rtmpose) | failed | 112 | 103 | 98.3 | – | 1.0x | 1634 | 1738 |
| [`rtmpose-s`](../models/pose.md#rtmpose) | failed | 35.7 | 33.2 | 29.0 | – | 1.1x | 1494 | 1559 |
| [`rtmpose-t`](../models/pose.md#rtmpose) | failed | 25.4 | 24.6 | 20.5 | – | 1.2x | 1444 | 1513 |
| [`vitpose-base`](../models/pose.md#vitpose) | failed | 228 | 212 | – | 172 | 1.2x | 1929 | 1995 |
| [`vitpose-base-simple`](../models/pose.md#vitpose) | failed | 196 | 186 | – | 160 | 1.2x | 1910 | 1977 |
| [`vitpose-plus-base`](../models/pose.md#vitpose) | failed | 229 | 212 | – | 174 | 1.2x | 1939 | 2006 |
| [`vitpose-plus-huge`](../models/pose.md#vitpose) | failed | 1247 | 1158 | – | 1182 | 1.0x | 4956 | 4957 |
| [`vitpose-plus-large`](../models/pose.md#vitpose) | failed | 646 | 603 | – | 526 | 1.1x | 2663 | 2741 |
| [`vitpose-plus-small`](../models/pose.md#vitpose) | failed | 101 | 92.3 | – | 66.5 | 1.4x | 1545 | 1633 |

## Features

### Features, synthetic, 224x224

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](../models/features.md#c_radio) | failed | 19.2 | 16.5 | unsupported | – | – | 2230 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | failed | 180 | 145 | unsupported | – | – | 9394 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | failed | 103 | 85.0 | unsupported | – | – | 5806 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | failed | 53.3 | 47.9 | unsupported | – | – | 3322 | – |
| [`clip-b16-laion2b`](../models/features.md#clip) | failed | 16.9 | 15.3 | unsupported | – | – | 1480 | – |
| [`clip-b16-openai`](../models/features.md#clip) | failed | 17.5 | 16.1 | unsupported | – | – | 1724 | – |
| [`clip-h14-378-dfn5b`](../models/features.md#clip) | failed | 132 | 108 | unsupported | – | – | 8090 | – |
| [`clip-h14-dfn5b`](../models/features.md#clip) | failed | 134 | 109 | unsupported | – | – | 8089 | – |
| [`clip-h14-laion2b`](../models/features.md#clip) | failed | 122 | 102 | unsupported | – | – | 8090 | – |
| [`clip-l14-336-openai`](../models/features.md#clip) | failed | 63.1 | 58.3 | unsupported | – | – | 4429 | – |
| [`clip-l14-datacompxl`](../models/features.md#clip) | failed | 60.8 | 56.7 | unsupported | – | – | 4426 | – |
| [`clip-l14-dfn2b`](../models/features.md#clip) | failed | 66.4 | 60.1 | unsupported | – | – | 4424 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | failed | 60.6 | 56.6 | unsupported | – | – | 4424 | – |
| [`clip-l14-openai`](../models/features.md#clip) | failed | 63.0 | 58.3 | unsupported | – | – | 4898 | – |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 28.5 | 23.3 | 21.7 | unsupported | – | 998 | 774 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 51.0 | 44.3 | 42.8 | unsupported | – | 1013 | 775 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 271 | 221 | 193 | unsupported | – | 8904 | 4502 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 294 | 263 | 235 | unsupported | – | 8897 | 4502 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 75.9 | 68.0 | 59.4 | unsupported | – | 3165 | 1562 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 107 | 96.1 | 86.3 | unsupported | – | 3168 | 1563 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 12.8 | 12.2 | 11.2 | unsupported | – | 476 | 341 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 24.8 | 22.2 | 22.1 | unsupported | – | 476 | 344 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | failed | 25.6 | 23.1 | unsupported | – | – | 880 | – |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | failed | 40.6 | 36.9 | unsupported | – | – | 2088 | – |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | failed | 22.4 | 18.5 | unsupported | – | – | 612 | – |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | failed | 11.5 | 10.6 | unsupported | – | – | 431 | – |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 24.9 | 22.6 | 20.0 | unsupported | – | 721 | 728 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 174 | 147 | 126 | unsupported | – | 4593 | 3466 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 71.2 | 62.0 | 52.7 | unsupported | – | 2465 | 1474 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 13.4 | 11.9 | 11.5 | unsupported | – | 446 | 354 | – |
| [`pe-core-b16-224`](../models/features.md#perception_encoder) | failed | 20.0 | 19.5 | 16.0 | 1.2x | – | 1569 | 1571 |
| [`pe-core-g14-448`](../models/features.md#perception_encoder) | failed | 368 | 286 | 259 | 1.1x | – | 17853 | 17853 |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | failed | 68.6 | 68.7 | 58.3 | 1.2x | – | 4527 | 4528 |
| [`pe-core-s16-384`](../models/features.md#perception_encoder) | failed | 10.9 | 11.8 | 10.4 | 1.1x | – | 604 | 646 |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | failed | 7.38 | 7.95 | 7.25 | 1.0x | – | 379 | 388 |
| [`pe-spatial-b16-512`](../models/features.md#perception_encoder) | failed | 19.2 | 19.0 | 16.1 | 1.2x | – | 1484 | 1486 |
| [`pe-spatial-g14-448`](../models/features.md#perception_encoder) | failed | 374 | 288 | 258 | 1.1x | – | 17726 | 17724 |
| [`pe-spatial-l14-448`](../models/features.md#perception_encoder) | failed | 68.1 | 68.7 | 57.6 | 1.2x | – | 4426 | 4429 |
| [`pe-spatial-s16-512`](../models/features.md#perception_encoder) | failed | 9.86 | 10.5 | 8.31 | 1.2x | – | 580 | 590 |
| [`pe-spatial-t16-512`](../models/features.md#perception_encoder) | failed | 6.90 | 7.47 | 5.94 | 1.2x | – | 375 | 383 |
| [`siglip2-b16-256`](../models/features.md#siglip2) | failed | 17.2 | 15.8 | unsupported | – | – | 1563 | – |
| [`siglip2-b16-512`](../models/features.md#siglip2) | failed | 17.2 | 15.8 | unsupported | – | – | 1574 | – |
| [`siglip2-g16-256`](../models/features.md#siglip2) | failed | 180 | 143 | unsupported | – | – | 13674 | – |
| [`siglip2-g16-384`](../models/features.md#siglip2) | failed | 180 | 142 | unsupported | – | – | 13640 | – |
| [`siglip2-l16-256`](../models/features.md#siglip2) | failed | 50.0 | 45.9 | unsupported | – | – | 4516 | – |
| [`siglip2-l16-512`](../models/features.md#siglip2) | failed | 50.3 | 45.5 | unsupported | – | – | 4524 | – |
| [`siglip2-so400m14-224`](../models/features.md#siglip2) | failed | 75.8 | 70.2 | unsupported | – | – | 5545 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | failed | 75.9 | 68.6 | unsupported | – | – | 5553 | – |
| [`siglip2-so400m16-256`](../models/features.md#siglip2) | failed | 65.7 | 57.5 | unsupported | – | – | 5546 | – |
| [`siglip2-so400m16-512`](../models/features.md#siglip2) | failed | 66.2 | 56.8 | unsupported | – | – | 5556 | – |

Note: no Metal kernels for this family (card mps_kernels).

## Multiview

### Multiview, synthetic, 336x518

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`mapanything-apache`](../models/multiview.md#mapanything) | failed | 7357 | 6369 | – | unsupported | – | 7218 | – |
| [`vggt-1b`](../models/multiview.md#vggt) | failed | 7209 | – | 7016 | – | 1.0x | 10654 | 10692 |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | failed | 7938 | 7242 | – | unsupported | – | 8602 | – |

Note: no Metal kernels for this family (card mps_kernels).

### Multiview, synthetic, 448x672

| model | time (ms): original fp32 | portable fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | failed | 1051 | unsupported | 3534 |

Note: no Metal kernels for this family (card mps_kernels).

### Multiview, synthetic, 1536x1024

| model | time (ms): original fp32 | portable fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | failed | 1010 | unsupported | 7593 |

Note: no Metal kernels for this family (card mps_kernels).

## Flow

### Flow, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`raft-large`](../models/flow.md#raft) | OOM | 2522 | 1351 | – | 1.9x | – | 12889 | 3647 |
| [`raft-large-kitti`](../models/flow.md#raft) | OOM | 10206 | 1349 | – | 7.6x | – | 11509 | 4671 |
| [`raft-large-things`](../models/flow.md#raft) | OOM | 3992 | 1348 | – | 3.0x | – | 11624 | 4670 |
| [`raft-small`](../models/flow.md#raft) | OOM | 4513 | 386 | – | 11.7x | – | 11385 | 2482 |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 2637 | 1820 | 1220 | 2.2x | 1.5x | 12444 | 12542 | 4752 |
| [`sea-raft-m-kitti`](../models/flow.md#sea_raft) | failed | 5026 | 1222 | – | 4.1x | – | 12688 | 4754 |
| [`sea-raft-m-spring`](../models/flow.md#sea_raft) | failed | 390 | 340 | – | 1.1x | – | 2574 | 1621 |
| [`sea-raft-m-things`](../models/flow.md#sea_raft) | failed | 4162 | 1221 | – | 3.4x | – | 11697 | 4753 |
| [`sea-raft-s`](../models/flow.md#sea_raft) | failed | 2115 | 833 | – | 2.5x | – | 11645 | 4667 |
| [`sea-raft-s-kitti`](../models/flow.md#sea_raft) | failed | 2000 | 831 | – | 2.4x | – | 11481 | 4664 |
| [`sea-raft-s-spring`](../models/flow.md#sea_raft) | failed | 268 | 233 | – | 1.1x | – | 2483 | 1527 |
<!-- device:end -->

## Notes on `kernels`

These notes come from an earlier comparison at 384x1248 (not in the tables): `kernels` (float16; Metal ConvGRUs,
correlation / geometry lookups, cost volumes, warp costs, deformable sampling, window attention) against the fastest
validated `portable` dtype, each run in its own process (median of 5, peak footprint).

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
