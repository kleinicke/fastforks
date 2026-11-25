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
| measured | 2026-11-06 to 2026-11-25 |
| rows | 2289 in `docs/data/speed/apple-m1-pro-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| [`aanet-plus-sceneflow`](../models/stereo.md#aanet) | 870 | 992 | 936 | 818 | 1.1x | 1.1x | 6805 | 3836 | 3754 |
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 813 | 880 | 723 | 631 | 1.3x | 1.1x | 3710 | 2594 | 2606 |
| [`bridgedepth-rvc-pretrain`](../models/stereo.md#bridgedepth) | 4416 | 4375 | 4232 | 3756 | 1.2x | 1.1x | 7028 | 6115 | 5100 |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 4279 | 4280 | 3782 | 3693 | 1.2x | 1.0x | 7464 | 6271 | 6270 |
| [`c-fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 7562 | 3326 | 2472 | unsupported | – | – | 13037 | 8460 | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | OOM | OOM | 23526 | unsupported | – | – | – | 12489 | – |
| [`cas-gwcnet-c-sceneflow`](../models/stereo.md#cas_gwcnet) | 7909 | 4747 | 2489 | unsupported | – | – | 9174 | 5132 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | OOM | 3361 | 2731 | unsupported | – | – | – | 8420 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | OOM | OOM | OOM | 2206 | – | – | – | – | 5416 |
| [`coex-sceneflow`](../models/stereo.md#coex) | 1024 | 323 | 241 | unsupported | – | – | 5473 | 3631 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 15862 | 5502 | 4484 | 4127 | 3.8x | 1.1x | 4683 | 2747 | 2801 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | OOM | 11105 | – | 8224 | – | 1.4x | – | 13138 | 8328 |
| [`dlnr-middlebury`](../models/stereo.md#dlnr) | 3257 | 2693 | 2143 | unsupported | – | – | 7472 | 4149 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 3258 | 2751 | 2195 | unsupported | – | – | 7469 | 4146 | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 1334 | 557 | 410 | unsupported | – | – | 4975 | 3819 | – |
| [`fast-acvnet-kitti12`](../models/stereo.md#fast_acvnet) | 1334 | 540 | 397 | unsupported | – | – | 4976 | 3817 | – |
| [`fast-acvnet-kitti15`](../models/stereo.md#fast_acvnet) | 1334 | 537 | 397 | unsupported | – | – | 4974 | 3822 | – |
| [`fast-acvnet-plus-generalization`](../models/stereo.md#fast_acvnet) | 1764 | 513 | 401 | unsupported | – | – | 4992 | 3843 | – |
| [`fast-acvnet-plus-kitti12`](../models/stereo.md#fast_acvnet) | 1761 | 512 | 401 | unsupported | – | – | 3963 | 3839 | – |
| [`fast-acvnet-plus-kitti15`](../models/stereo.md#fast_acvnet) | 1762 | 512 | 402 | unsupported | – | – | 4991 | 2814 | – |
| [`fast-acvnet-plus-sceneflow`](../models/stereo.md#fast_acvnet) | 1762 | 513 | 402 | unsupported | – | – | 3962 | 3844 | – |
| [`fast-acvnet-sceneflow`](../models/stereo.md#fast_acvnet) | 1333 | 537 | 396 | unsupported | – | – | 4980 | 3819 | – |
| [`fast-foundationstereo`](../models/stereo.md#c_foundationstereo) | 9528 | 3471 | 2477 | unsupported | – | – | 12528 | 8462 | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | OOM | 30919 | 25647 | 24480 | – | 1.0x | – | 13761 | 13277 |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | OOM | 22956 | 19059 | 19496 | – | 1.0x | – | 13094 | 12980 |
| [`great-igev-eth3d`](../models/stereo.md#great_igev) | OOM | 9113 | 11459 | 9636 | – | 0.9x | – | 11795 | 10133 |
| [`great-igev-kitti`](../models/stereo.md#great_igev) | OOM | 31916 | 11076 | 9634 | – | 1.1x | – | 10538 | 10130 |
| [`great-igev-middlebury`](../models/stereo.md#great_igev) | OOM | 15640 | 18185 | 25368 | – | 0.6x | – | 12460 | 12420 |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | OOM | 10367 | 9090 | 9861 | – | 0.9x | – | 11106 | 10126 |
| [`gwcnet-g-sceneflow`](../models/stereo.md#gwcnet) | 6074 | 4340 | 2378 | unsupported | – | – | 10858 | 6522 | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | OOM | 3559 | 2759 | unsupported | – | – | – | 7601 | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | unsupported | 307 | 313 | 136 | – | 2.3x | – | 4507 | 2486 |
| [`hitnet-middlebury-d400`](../models/stereo.md#hitnet) | unsupported | 1155 | 1227 | 440 | – | 2.6x | – | 10563 | 4591 |
| [`igev-plusplus-eth3d`](../models/stereo.md#igev_plusplus) | 9851 | 10444 | 8613 | 6139 | 1.6x | 1.4x | 7730 | 6040 | 4971 |
| [`igev-plusplus-kitti12`](../models/stereo.md#igev_plusplus) | 9779 | 10398 | 8606 | 6124 | 1.6x | 1.4x | 7841 | 6034 | 4967 |
| [`igev-plusplus-kitti15`](../models/stereo.md#igev_plusplus) | 11619 | 10489 | 8597 | 6125 | 1.9x | 1.4x | 8295 | 6034 | 4967 |
| [`igev-plusplus-middlebury`](../models/stereo.md#igev_plusplus) | 9785 | 10399 | 8595 | 6130 | 1.6x | 1.4x | 8179 | 6037 | 4966 |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 10992 | 9676 | 8611 | 6135 | 1.8x | 1.4x | 7860 | 6104 | 4972 |
| [`igev-stereo-eth3d`](../models/stereo.md#igev_stereo) | 6627 | 5849 | 5217 | 3998 | 1.7x | 1.3x | 7008 | 3862 | 3789 |
| [`igev-stereo-kitti12`](../models/stereo.md#igev_stereo) | 6805 | 5859 | 5210 | 4006 | 1.7x | 1.3x | 7022 | 3862 | 3791 |
| [`igev-stereo-kitti15`](../models/stereo.md#igev_stereo) | 6841 | 5849 | 5221 | 4008 | 1.7x | 1.3x | 5985 | 3862 | 3791 |
| [`igev-stereo-middlebury`](../models/stereo.md#igev_stereo) | 5660 | 5848 | 5210 | 4007 | 1.4x | 1.3x | 7012 | 3861 | 3792 |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 5949 | 6375 | 5347 | 4001 | 1.5x | 1.3x | 5984 | 3867 | 3793 |
| [`las2-h`](../models/stereo.md#las2) | 1306 | 664 | 588 | 734 | 1.8x | 0.8x | 5643 | 3674 | 3771 |
| [`las2-l`](../models/stereo.md#las2) | 604 | 500 | 426 | 519 | 1.2x | 0.8x | 4740 | 2818 | 2818 |
| [`las2-m`](../models/stereo.md#las2) | 327 | 241 | 232 | 281 | 1.2x | 0.8x | 4615 | 2684 | 2666 |
| [`las2-s`](../models/stereo.md#las2) | 273 | 182 | 167 | 198 | 1.4x | 0.8x | 4556 | 2564 | 2620 |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 8561 | 8362 | 7564 | 6693 | 1.3x | 1.1x | 8058 | 5829 | 5695 |
| [`mocha-v2-sceneflow`](../models/stereo.md#mocha_stereo) | 7685 | 8323 | 7560 | 6501 | 1.2x | 1.2x | 9037 | 5807 | 4643 |
| [`monster-mix-all`](../models/stereo.md#monster) | 20936 | 13423 | 11377 | 9646 | 2.2x | 1.2x | 12856 | 6320 | 6330 |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 17464 | 15001 | 11361 | 10202 | 1.7x | 1.1x | 12920 | 7425 | 7347 |
| [`monster-sceneflow`](../models/stereo.md#monster) | 18672 | 12916 | 11133 | 9642 | 1.9x | 1.2x | 13435 | 6364 | 6299 |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 1736 | 1560 | 1469 | 1204 | 1.4x | 1.2x | 7232 | 4747 | 3806 |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 6066 | 2631 | 2022 | unsupported | – | – | 10870 | 5537 | – |
| [`raft-stereo-eth3d`](../models/stereo.md#raft_stereo) | 5162 | 4396 | 3937 | 3638 | 1.4x | 1.1x | 9712 | 5720 | 4721 |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 5574 | 4395 | 3940 | 3634 | 1.5x | 1.1x | 9545 | 4695 | 4722 |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 5066 | 4396 | 3938 | 3637 | 1.4x | 1.1x | 9437 | 4695 | 4720 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 917 | 618 | 612 | unsupported | – | – | 6269 | 4076 | – |
| [`s2m2-l`](../models/stereo.md#s2m2) | unsupported | 8709 | 6590 | unsupported | – | – | – | 9213 | – |
| [`s2m2-m`](../models/stereo.md#s2m2) | unsupported | 4714 | 3620 | unsupported | – | – | – | 7941 | – |
| [`s2m2-s`](../models/stereo.md#s2m2) | unsupported | 1710 | 1606 | unsupported | – | – | – | 6017 | – |
| [`s2m2-xl`](../models/stereo.md#s2m2) | unsupported | 18956 | 13398 | unsupported | – | – | – | 12386 | – |
| [`selective-igev-eth3d-finetune`](../models/stereo.md#selective_igev) | 8184 | 9066 | 6991 | 5708 | 1.4x | 1.2x | 7003 | 3854 | 3904 |
| [`selective-igev-eth3d-train`](../models/stereo.md#selective_igev) | 8192 | 9010 | 6969 | 5710 | 1.4x | 1.2x | 7002 | 3862 | 3907 |
| [`selective-igev-kitti12`](../models/stereo.md#selective_igev) | 8197 | 9019 | 6971 | 5709 | 1.4x | 1.2x | 6999 | 3853 | 3906 |
| [`selective-igev-kitti15`](../models/stereo.md#selective_igev) | 8194 | 9016 | 6963 | 5710 | 1.4x | 1.2x | 7002 | 3853 | 3905 |
| [`selective-igev-middlebury-finetune`](../models/stereo.md#selective_igev) | 8198 | 9026 | 6989 | 5778 | 1.4x | 1.2x | 7003 | 3852 | 3907 |
| [`selective-igev-middlebury-train`](../models/stereo.md#selective_igev) | 8195 | 9100 | 7206 | 5738 | 1.4x | 1.3x | 7003 | 3854 | 3907 |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 8196 | 8458 | 6848 | 5710 | 1.4x | 1.2x | 7002 | 3946 | 3907 |
| [`waft-stereo-b`](../models/stereo.md#waft_stereo) | OOM | 11022 | OOM | unsupported | – | – | – | 12854 | – |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | 4098 | 3236 | 3323 | unsupported | – | – | 10164 | 7454 | – |

Note: no Metal kernels for this family (card mps_kernels).
Note: the authors' original is a TensorFlow graph (google-research/hitnet, released SavedModel checkpoints), not PyTorch; there is no MPS execution path for it and no original-code adapter in the bench.
Note: the original's adaptive pooling needs input sizes divisible by the output size on MPS.

## Depth

### Depth, synthetic, 756x1008

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`prompt-depth-anything-s`](../models/depth.md#prompt_depth_anything) | 251 | 216 | 195 | unsupported | 2741 | 1588 |

Note: no Metal kernels for this family (card mps_kernels).

### Depth, synthetic, 994x1482

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels bf16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`any2full-b`](../models/depth.md#any2full) | – | 292 | 268 | – | unsupported | – | – | – | – | 2198 | – |
| [`any2full-l`](../models/depth.md#any2full) | – | 742 | 684 | – | unsupported | – | – | – | – | 3956 | – |
| [`any2full-s`](../models/depth.md#any2full) | – | 130 | 118 | – | unsupported | – | – | – | – | 1641 | – |
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 898 | 833 | 759 | – | unsupported | – | – | – | 8269 | 7243 | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 412 | 336 | 318 | – | unsupported | – | – | – | 4019 | 2520 | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 271 | 291 | 280 | – | unsupported | – | – | – | 3393 | 2499 | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 271 | 293 | 280 | – | unsupported | – | – | – | 3393 | 2499 | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 1174 | 1248 | 1184 | – | unsupported | – | – | – | 9549 | 8136 | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 239 | 246 | 199 | – | unsupported | – | – | – | 2342 | 2061 | – |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 697 | 722 | 579 | – | unsupported | – | – | – | 4218 | 3282 | – |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | 239 | 242 | 199 | – | unsupported | – | – | – | 2351 | 2062 | – |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | 700 | 714 | 579 | – | unsupported | – | – | – | 4227 | 3283 | – |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | 110 | 99.5 | 79.5 | – | unsupported | – | – | – | 1862 | 1572 | – |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | 239 | 242 | 199 | – | unsupported | – | – | – | 2349 | 2061 | – |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | 698 | 714 | 580 | – | unsupported | – | – | – | 4226 | 3283 | – |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | 110 | 99.7 | 79.5 | – | unsupported | – | – | – | 1856 | 1572 | – |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 113 | 99.9 | 80.7 | – | unsupported | – | – | – | 1860 | 1575 | – |
| [`depth-pro`](../models/depth.md#depth_pro) | 5853 | 6115 | 5011 | – | unsupported | – | – | – | 13125 | 7169 | – |
| [`depthfm-v1`](../models/depth.md#depthfm) | – | OOM | OOM | – | unsupported | – | – | – | – | – | – |
| [`distill-any-depth-b`](../models/depth.md#depth_anything_v2) | 238 | 250 | 199 | – | unsupported | – | – | – | 2389 | 2061 | – |
| [`distill-any-depth-l`](../models/depth.md#depth_anything_v2) | 697 | 725 | 580 | – | unsupported | – | – | – | 4288 | 3281 | – |
| [`distill-any-depth-s`](../models/depth.md#depth_anything_v2) | 109 | 101 | 79.3 | – | unsupported | – | – | – | 1868 | 1570 | – |
| [`lingbot-depth`](../models/depth.md#lingbot_depth) | – | 3363 | 3376 | – | unsupported | – | – | – | – | 4011 | – |
| [`lingbot-depth-dc`](../models/depth.md#lingbot_depth) | – | 3363 | 3374 | – | unsupported | – | – | – | – | 4012 | – |
| [`lotus-normal-d-v1-1`](../models/depth.md#lotus) | 42030 | 2019 | 1459 | – | unsupported | – | – | – | 12351 | 5628 | – |
| [`lotus-normal-g-v1-1`](../models/depth.md#lotus) | 39522 | 1855 | 1461 | – | unsupported | – | – | – | 12354 | 5628 | – |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 3237 | 2982 | 2492 | – | unsupported | – | – | – | 10462 | 5590 | – |
| [`marigold-normals-lcm-v0-1`](../models/depth.md#marigold_normals) | 2639 | 1798 | 1514 | – | unsupported | – | – | – | 10318 | 5602 | – |
| [`marigold-v2-depth`](../models/depth.md#marigold_v2) | – | unsupported | – | – | – | unsupported | – | – | – | – | – |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 1946 | 1822 | 1861 | – | unsupported | – | – | – | 4973 | 4523 | – |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 1107 | 679 | 635 | – | unsupported | – | – | – | 4976 | 2407 | – |
| [`moge-2-vitb-normal`](../models/depth.md#moge) | 915 | 585 | 494 | 586 | – | – | 1.6x | 0.8x | 3223 | 2263 | 3160 |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 1553 | 1350 | 1176 | 1344 | – | – | 1.2x | 0.9x | 4036 | 3258 | 3994 |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 516 | 362 | 303 | 362 | – | – | 1.4x | 0.8x | 2849 | 1679 | 2704 |
| [`moge-3-vitl`](../models/depth.md#moge) | unsupported | 3692 | 3333 | 2415 | – | – | – | 1.4x | – | 6299 | 6121 |
| [`omni-dc`](../models/depth.md#omni_dc) | – | 2349 | 2408 | – | unsupported | – | – | – | – | 12317 | – |
| [`pointdit-b-512`](../models/depth.md#pointdit) | 468 | 367 | 337 | – | unsupported | – | – | – | 3167 | 1912 | – |
| [`pointdit-l-512`](../models/depth.md#pointdit) | 1356 | 1086 | 1013 | – | unsupported | – | – | – | 6994 | 4548 | – |
| [`prompt-depth-anything-l`](../models/depth.md#prompt_depth_anything) | – | 1278 | 1143 | – | unsupported | – | – | – | – | 3303 | – |
| [`stablenormal-turbo`](../models/depth.md#stablenormal) | 51987 | 1979 | 1652 | – | unsupported | – | – | – | 11428 | 6615 | – |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 1356 | 1252 | – | – | unsupported | – | – | – | 5260 | 6077 | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1333 | 1217 | 1129 | – | unsupported | – | – | – | 5316 | 4036 | – |
| [`video-depth-anything-metric-s`](../models/depth.md#video_depth_anything) | – | 836 | 780 | – | unsupported | – | – | – | – | 4581 | – |
| [`video-depth-anything-s`](../models/depth.md#video_depth_anything) | – | 839 | 779 | – | unsupported | – | – | – | – | 4578 | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 1268 | 778 | 785 | – | unsupported | – | – | – | 4223 | 3558 | – |

Note: no Metal kernels for this family (card mps_kernels).
Note: the family is not validated on MPS (card mps_dtypes).
Note: the original needs flex_gemm, a CUDA-only extension.

## Classification

### Classification, synthetic, 224x224

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`alexnet.tv_in1k`](../models/classification.md#alexnet) | 9.48 | 6.28 | 5.93 | – | unsupported | 1819 | 1577 |
| [`coatnet_0_rw_224.sw_in1k`](../models/classification.md#coatnet) | 21.6 | 21.3 | 20.5 | unsupported | unsupported | 763 | 620 |
| [`coatnet_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 33.9 | 35.9 | 36.0 | unsupported | unsupported | 1933 | 1759 |
| [`coatnet_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 47.9 | 56.1 | 47.5 | unsupported | unsupported | 2091 | 1896 |
| [`coatnet_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 19.8 | 23.4 | 22.1 | unsupported | unsupported | 596 | 508 |
| [`coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 37.1 | 39.2 | 36.5 | unsupported | unsupported | 1920 | 772 |
| [`coatnet_rmlp_1_rw_224.sw_in1k`](../models/classification.md#coatnet) | 35.2 | 37.8 | 37.2 | unsupported | unsupported | 940 | 766 |
| [`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 52.5 | 57.5 | 46.8 | unsupported | unsupported | 2183 | 1898 |
| [`coatnet_rmlp_2_rw_224.sw_in1k`](../models/classification.md#coatnet) | 51.9 | 57.8 | 47.9 | unsupported | unsupported | 2140 | 1899 |
| [`coatnet_rmlp_nano_rw_224.sw_in1k`](../models/classification.md#coatnet) | 21.9 | 23.4 | 23.9 | unsupported | unsupported | 1627 | 1534 |
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 26.7 | 28.6 | 27.6 | – | unsupported | 2061 | 1973 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 25.7 | 28.0 | 27.7 | – | unsupported | 2092 | 1973 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 51.9 | 56.5 | 49.1 | – | unsupported | 2065 | 1975 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 42.7 | 43.4 | 43.6 | – | unsupported | 2775 | 2756 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 40.8 | 44.2 | 40.2 | – | unsupported | 2770 | 2732 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 90.3 | 95.7 | 84.7 | – | unsupported | 2942 | 2732 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 20.3 | 22.7 | 20.8 | – | unsupported | 881 | 826 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 20.3 | 22.6 | 20.6 | – | unsupported | 825 | 823 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 34.8 | 36.8 | 34.2 | – | unsupported | 1890 | 824 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 12.0 | 13.3 | 12.6 | – | unsupported | 706 | 609 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 11.8 | 13.7 | 13.1 | – | unsupported | 697 | 609 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 20.7 | 23.2 | 20.2 | – | unsupported | 1736 | 612 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 41.0 | 41.9 | 38.5 | – | unsupported | 2066 | 1974 |
| [`davit_base.msft_in1k`](../models/classification.md#davit) | 34.5 | 37.5 | 34.5 | unsupported | unsupported | 2060 | 1967 |
| [`davit_small.msft_in1k`](../models/classification.md#davit) | 26.4 | 27.6 | 26.1 | unsupported | unsupported | 957 | 786 |
| [`davit_tiny.msft_in1k`](../models/classification.md#davit) | 15.0 | 16.2 | 16.7 | unsupported | unsupported | 738 | 593 |
| [`deit_base_patch16_224.fb_in1k`](../models/classification.md#deit) | 17.5 | 19.4 | 16.5 | – | unsupported | 1147 | 1159 |
| [`densenet121.tv_in1k`](../models/classification.md#densenet) | 31.5 | 28.7 | 27.7 | – | unsupported | 554 | 704 |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 64.1 | 68.7 | 63.2 | – | unsupported | 3291 | 3268 |
| [`efficientformer_l1.snap_dist_in1k`](../models/classification.md#efficientformer) | 12.5 | 9.36 | 9.34 | unsupported | unsupported | 550 | 581 |
| [`efficientformer_l3.snap_dist_in1k`](../models/classification.md#efficientformer) | 21.6 | 17.5 | 15.4 | unsupported | unsupported | 747 | 835 |
| [`efficientformer_l7.snap_dist_in1k`](../models/classification.md#efficientformer) | 34.9 | 27.4 | 26.6 | unsupported | unsupported | 1146 | 2245 |
| [`efficientformerv2_l.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 49.8 | 37.9 | 35.8 | unsupported | unsupported | 763 | 863 |
| [`efficientformerv2_s0.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 21.0 | 20.4 | 14.6 | unsupported | unsupported | 483 | 538 |
| [`efficientformerv2_s1.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 24.6 | 19.1 | 18.2 | unsupported | unsupported | 521 | 579 |
| [`efficientformerv2_s2.snap_dist_in1k`](../models/classification.md#efficientformerv2) | 36.4 | 27.5 | 25.9 | unsupported | unsupported | 575 | 662 |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 17.9 | 14.3 | 13.1 | – | unsupported | 442 | 493 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 54.0 | 41.1 | 43.0 | – | unsupported | 1676 | 1673 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 45.9 | 35.2 | 35.5 | – | unsupported | 638 | 626 |
| [`efficientvit_b2.r288_in1k`](../models/classification.md#efficientvit) | 25.7 | 20.4 | 19.5 | – | unsupported | 1695 | 1631 |
| [`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`](../models/classification.md#eva02) | 267 | 273 | 246 | – | unsupported | 3515 | 2937 |
| [`falconsai-nsfw-vit`](../models/classification.md#nsfw_vit) | 18.8 | 18.1 | 15.8 | – | unsupported | 1121 | 1236 |
| [`focalnet_base_lrf.ms_in1k`](../models/classification.md#focalnet) | 44.7 | 44.6 | 42.8 | unsupported | unsupported | 2225 | 2012 |
| [`focalnet_base_srf.ms_in1k`](../models/classification.md#focalnet) | 41.8 | 41.2 | 39.6 | unsupported | unsupported | 2222 | 2005 |
| [`focalnet_small_lrf.ms_in1k`](../models/classification.md#focalnet) | 35.2 | 35.5 | 33.1 | unsupported | unsupported | 921 | 825 |
| [`focalnet_small_srf.ms_in1k`](../models/classification.md#focalnet) | 32.5 | 32.3 | 30.6 | unsupported | unsupported | 1003 | 815 |
| [`focalnet_tiny_lrf.ms_in1k`](../models/classification.md#focalnet) | 20.5 | 19.9 | 18.7 | unsupported | unsupported | 762 | 616 |
| [`focalnet_tiny_srf.ms_in1k`](../models/classification.md#focalnet) | 18.7 | 18.2 | 17.2 | unsupported | unsupported | 755 | 612 |
| [`hgnetv2_b0.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 11.6 | 6.50 | 6.12 | unsupported | unsupported | 480 | 483 |
| [`hgnetv2_b1.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 11.0 | 6.73 | 6.39 | unsupported | unsupported | 492 | 520 |
| [`hgnetv2_b2.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 16.4 | 9.46 | 9.08 | unsupported | unsupported | 571 | 556 |
| [`hgnetv2_b3.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 19.5 | 11.9 | 10.5 | unsupported | unsupported | 1617 | 1580 |
| [`hgnetv2_b4.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 20.3 | 12.4 | 11.6 | unsupported | unsupported | 1658 | 1611 |
| [`hgnetv2_b5.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 32.8 | 19.5 | 18.0 | unsupported | unsupported | 948 | 1868 |
| [`hgnetv2_b6.ssld_stage2_ft_in1k`](../models/classification.md#hgnetv2) | 52.5 | 34.4 | 30.8 | unsupported | unsupported | 2269 | 2040 |
| [`inception_v3.tv_in1k`](../models/classification.md#inception_v3) | 27.7 | 17.4 | 16.5 | – | unsupported | 730 | 780 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 12.4 | 6.30 | 6.32 | – | unsupported | 396 | 434 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 13.4 | 8.59 | 8.20 | – | unsupported | 440 | 488 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 8.11 | 4.86 | 4.10 | – | unsupported | 446 | 432 |
| [`mobilenetv4_conv_medium.e500_r256_in1k`](../models/classification.md#mobilenetv4) | 18.1 | 13.8 | 9.82 | – | unsupported | 544 | 563 |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 46.1 | 41.2 | 34.0 | – | unsupported | 2209 | 2046 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 24.5 | 12.2 | 12.9 | – | unsupported | 478 | 413 |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 25.0 | 12.4 | 12.7 | – | unsupported | 479 | 411 |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 25.1 | 12.6 | 12.4 | – | unsupported | 505 | 427 |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 24.9 | 13.9 | 12.7 | – | unsupported | 502 | 431 |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 24.5 | 12.8 | 12.2 | – | unsupported | 511 | 439 |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 26.1 | 12.0 | 12.7 | – | unsupported | 511 | 434 |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 40.1 | 19.9 | 21.8 | – | unsupported | 575 | 473 |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 39.8 | 21.4 | 19.6 | – | unsupported | 575 | 491 |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 54.1 | 28.7 | 28.1 | – | unsupported | 685 | 603 |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 53.9 | 28.0 | 27.7 | – | unsupported | 692 | 604 |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 41.1 | 26.2 | 24.1 | – | unsupported | 1007 | 1094 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 7.25 | 5.15 | 4.86 | – | unsupported | 466 | 471 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 16.8 | 11.9 | 10.6 | – | unsupported | 669 | 659 |
| [`shufflenet_v2_x1_0.tv_in1k`](../models/classification.md#shufflenetv2) | 12.1 | 8.28 | 6.62 | – | unsupported | 364 | 391 |
| [`shvit_s1.in1k`](../models/classification.md#shvit) | 17.8 | 10.9 | 10.8 | unsupported | unsupported | 493 | 432 |
| [`shvit_s2.in1k`](../models/classification.md#shvit) | 18.7 | 12.1 | 13.5 | unsupported | unsupported | 536 | 481 |
| [`shvit_s3.in1k`](../models/classification.md#shvit) | 20.2 | 13.0 | 14.8 | unsupported | unsupported | 557 | 501 |
| [`squeezenet1_1.tv_in1k`](../models/classification.md#squeezenet) | 4.03 | 3.82 | 3.73 | – | unsupported | 347 | 339 |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 13.0 | 8.46 | 8.21 | – | unsupported | 436 | 381 |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 11.4 | 7.35 | 7.07 | – | unsupported | 440 | 418 |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 14.4 | 9.87 | 9.31 | – | unsupported | 464 | 438 |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 18.8 | 12.9 | 12.4 | – | unsupported | 486 | 457 |
| [`swin_base_patch4_window7_224.ms_in22k_ft_in1k`](../models/classification.md#swin) | 37.5 | 28.3 | 25.3 | – | unsupported | 2062 | 1946 |
| [`tiny_vit_11m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 13.2 | 9.54 | 9.73 | – | unsupported | 549 | 466 |
| [`tiny_vit_21m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 15.2 | 11.8 | 11.3 | – | unsupported | 679 | 537 |
| [`tiny_vit_5m_224.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 12.8 | 8.73 | 8.42 | – | unsupported | 465 | 407 |
| [`vgg16.tv_in1k`](../models/classification.md#vgg) | 13.6 | 13.5 | 11.4 | – | unsupported | 2438 | 2430 |
| [`vit_base_patch16_224.augreg2_in21k_ft_in1k`](../models/classification.md#vit) | 17.4 | 17.1 | 15.6 | – | unsupported | 1258 | 1158 |
| [`vit_so400m_patch14_siglip_378.webli_ft_in1k`](../models/classification.md#siglip) | 215 | 215 | 196 | – | unsupported | 3973 | 3916 |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.

### Classification, synthetic, 256x256

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`fastvit_sa24.apple_dist_in1k`](../models/classification.md#fastvit) | 24.8 | 17.2 | 16.3 | – | unsupported | 566 | 632 |
| [`fastvit_t12.apple_dist_in1k`](../models/classification.md#fastvit) | 12.1 | 9.25 | 9.06 | – | unsupported | 494 | 473 |
| [`fastvit_t8.apple_dist_in1k`](../models/classification.md#fastvit) | 9.94 | 8.66 | 7.55 | – | unsupported | 433 | 450 |
| [`shvit_s4.in1k`](../models/classification.md#shvit) | 26.6 | 16.7 | 15.5 | unsupported | unsupported | 627 | 514 |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.

### Classification, synthetic, 384x384

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`](../models/classification.md#coatnet) | 127 | 130 | 107 | unsupported | unsupported | 2111 | 2013 |
| [`tiny_vit_21m_384.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 24.6 | 21.5 | 18.8 | – | unsupported | 1724 | 613 |

Note: the card has no kernels mode.
Note: no Metal kernels for this family (card mps_kernels).

### Classification, synthetic, 448x448

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`wd-convnext-tagger-v3`](../models/classification.md#wd_tagger) | 68.5 | 66.7 | 56.0 | unsupported | 2379 | 2013 |
| [`wd-eva02-large-tagger-v3`](../models/classification.md#wd_tagger) | 268 | 266 | 235 | unsupported | 3754 | 2976 |
| [`wd-vit-large-tagger-v3`](../models/classification.md#wd_tagger) | 222 | 221 | 201 | unsupported | 3318 | 3304 |
| [`wd-vit-tagger-v3`](../models/classification.md#wd_tagger) | 59.6 | 58.6 | 52.3 | unsupported | 2463 | 2064 |

Note: no Metal kernels for this family (card mps_kernels).

### Classification, synthetic, 512x512

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`tiny_vit_21m_512.dist_in22k_ft_in1k`](../models/classification.md#tinyvit) | 38.7 | 34.3 | 29.3 | unsupported | 1818 | 1676 |

Note: no Metal kernels for this family (card mps_kernels).

## Segmentation

### Segmentation, synthetic, 512x512

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`bisenet-face-parsing`](../models/segmentation.md#bisenet) | 23.7 | 15.8 | 13.6 | unsupported | – | 1466 | 440 |
| [`segformer-b2-clothes`](../models/segmentation.md#segformer) | unsupported | 59.5 | 49.9 | – | unsupported | – | 1643 |
| [`segformer-b5-face-parsing`](../models/segmentation.md#segformer) | unsupported | 135 | 111 | – | unsupported | – | 1899 |

Note: the card has no kernels mode.
Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: no Metal kernels for this family (card mps_kernels).

### Segmentation, synthetic, 1024x2048

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | unsupported | 721 | 528 | unsupported | – | 2010 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | unsupported | 516 | 430 | unsupported | – | 1891 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 491 | 409 | 361 | unsupported | 3325 | 2174 |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 1235 | 1032 | 927 | unsupported | 5570 | 3167 |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 264 | 224 | 195 | unsupported | 2830 | 1532 |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 733 | 608 | 540 | unsupported | 7112 | 4262 |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 782 | 648 | 575 | unsupported | 6918 | 4259 |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | unsupported | 468 | 405 | unsupported | – | 1875 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | unsupported | 263 | 228 | unsupported | – | 2044 |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 933 | 909 | 964 | unsupported | 7342 | 6661 |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | OOM | 1127 | 1161 | unsupported | – | 9492 |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | unsupported | 486 | 411 | unsupported | – | 1945 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 4354 | 2988 | 1517 | unsupported | 4741 | 3748 |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 6308 | 5099 | 3113 | unsupported | 7360 | 6340 |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 147 | 106 | 113 | unsupported | 4085 | 1523 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 852 | 428 | 375 | unsupported | 5348 | 1974 |
| [`segnext-mscan-b-ade20k`](../models/segmentation.md#segnext) | unsupported | 206 | 184 | unsupported | – | 1659 |
| [`segnext-mscan-l-ade20k`](../models/segmentation.md#segnext) | unsupported | 299 | 268 | unsupported | – | 1840 |
| [`segnext-mscan-s-ade20k`](../models/segmentation.md#segnext) | unsupported | 152 | 134 | unsupported | – | 1556 |
| [`segnext-mscan-t-ade20k`](../models/segmentation.md#segnext) | unsupported | 128 | 118 | unsupported | – | 1445 |

Note: the original needs mmcv's compiled ops (mmcv._ext), which mmcv does not ship for macOS.
Note: no Metal kernels for this family (card mps_kernels).

## Matting

### Matting, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`ben2`](../models/matting.md#ben2) | 10380 | 1148 | – | – | unsupported | 6959 | 6252 |
| [`birefnet`](../models/matting.md#birefnet) | 2710 | 1333 | 1152 | – | unsupported | 9880 | 5433 |
| [`birefnet-lite`](../models/matting.md#birefnet) | 541 | 474 | 505 | – | unsupported | 6060 | 4930 |
| [`inspyrenet`](../models/matting.md#inspyrenet) | 864 | 1108 | 702 | unsupported | – | 7535 | 4172 |
| [`inspyrenet-fast`](../models/matting.md#inspyrenet) | 225 | 208 | 145 | unsupported | – | 2304 | 2071 |
| [`isnet`](../models/matting.md#isnet) | 156 | 157 | 129 | – | unsupported | 2032 | 1745 |
| [`isnet-general-use`](../models/matting.md#isnet) | 156 | 157 | 130 | – | unsupported | 2030 | 1746 |
| [`matanyone`](../models/matting.md#matanyone) | 6473 | 3444 | 3091 | unsupported | – | 6365 | 3838 |
| [`rmbg-2.0`](../models/matting.md#birefnet) | – | 1096 | 1008 | – | unsupported | – | 5434 |
| [`u2net`](../models/matting.md#u2net) | 80.7 | 81.6 | 79.4 | – | unsupported | 1910 | 1885 |
| [`u2net-human-seg`](../models/matting.md#u2net) | 81.7 | 81.7 | 75.1 | – | unsupported | 1911 | 1884 |
| [`u2netp`](../models/matting.md#u2net) | 53.1 | 55.1 | 50.2 | – | unsupported | 1404 | 1402 |
| [`vitmatte-b`](../models/matting.md#vitmatte) | OOM | 4680 | 2740 | – | unsupported | – | 2966 |
| [`vitmatte-s`](../models/matting.md#vitmatte) | 3605 | 2195 | 1254 | – | unsupported | 9787 | 1530 |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.

## Edges

### Edges, synthetic, 1024x1024

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`anyline`](../models/edges.md#anyline) | 259 | 130 | 117 | unsupported | 1557 | 1319 |
| [`hed`](../models/edges.md#hed) | 66.8 | 64.9 | 56.9 | unsupported | 1554 | 1381 |
| [`lineart`](../models/edges.md#lineart) | 114 | 95.8 | 66.5 | unsupported | 1481 | 1362 |
| [`lineart-anime`](../models/edges.md#lineart_anime) | 83.2 | 63.4 | 51.1 | unsupported | 1782 | 1774 |
| [`lineart-coarse`](../models/edges.md#lineart) | 117 | 93.0 | 68.2 | unsupported | 1481 | 1364 |
| [`mlsd`](../models/edges.md#mlsd) | 55.3 | 50.0 | 44.6 | unsupported | 1500 | 1373 |
| [`pidinet`](../models/edges.md#pidinet) | 109 | 90.8 | 73.2 | unsupported | 1480 | 1349 |
| [`teed`](../models/edges.md#teed) | 25.8 | 20.1 | 16.7 | unsupported | 1434 | 1313 |

Note: the card has no kernels mode.

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`clipseg`](../models/promptable_segmentation.md#clipseg) | 67.6 | 44.7 | 37.7 | unsupported | 1606 | 1561 |
| [`efficientvit-sam-l0`](../models/promptable_segmentation.md#efficientvit_sam) | 70.5 | 50.6 | – | unsupported | 1959 | 1779 |
| [`efficientvit-sam-l1`](../models/promptable_segmentation.md#efficientvit_sam) | 85.4 | 75.2 | – | unsupported | 2075 | 1783 |
| [`efficientvit-sam-l2`](../models/promptable_segmentation.md#efficientvit_sam) | 102 | 91.2 | – | unsupported | 2144 | 2004 |
| [`efficientvit-sam-xl0`](../models/promptable_segmentation.md#efficientvit_sam) | 157 | 167 | – | unsupported | 3588 | 3032 |
| [`efficientvit-sam-xl1`](../models/promptable_segmentation.md#efficientvit_sam) | 245 | 259 | – | unsupported | 3671 | 3452 |
| [`insid3-vitl16`](../models/promptable_segmentation.md#insid3) | 3121 | 2953 | 2510 | unsupported | 3003 | 3414 |
| [`mobile-sam`](../models/promptable_segmentation.md#mobile_sam) | 121 | 80.8 | 69.9 | unsupported | 1768 | 1504 |
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | 745 | 645 | 683 | unsupported | 4909 | 4374 |
| [`sam-hq-vit-h`](../models/promptable_segmentation.md#sam_hq) | 2758 | 2633 | 2347 | unsupported | 8858 | 7860 |
| [`sam-hq-vit-l`](../models/promptable_segmentation.md#sam_hq) | 1582 | 1420 | 1292 | unsupported | 6401 | 6214 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 841 | 631 | 600 | unsupported | 4783 | 3762 |
| [`sam-vit-h`](../models/promptable_segmentation.md#sam) | 3210 | 2518 | 2333 | unsupported | 8721 | 7784 |
| [`sam-vit-l`](../models/promptable_segmentation.md#sam) | 1844 | 1373 | 1279 | unsupported | 6292 | 5193 |
| [`sam2.1-hiera-base-plus`](../models/promptable_segmentation.md#sam2) | 543 | 704 | 394 | unsupported | 4874 | 3988 |
| [`sam2.1-hiera-large`](../models/promptable_segmentation.md#sam2) | 722 | 1038 | 544 | unsupported | 3860 | 2206 |
| [`sam2.1-hiera-small`](../models/promptable_segmentation.md#sam2) | 217 | 208 | 148 | unsupported | 3071 | 1550 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 193 | 241 | 129 | unsupported | 3044 | 1514 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 2203 | 3620 | 3688 | unsupported | 4842 | 5807 |

Note: no Metal kernels for this family (card mps_kernels).

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|
| [`deim-l-coco`](../models/detection.md#dfine) | 106 | 130 | 71.0 | – | unsupported | 1834 | 1806 |
| [`deim-m-coco`](../models/detection.md#dfine) | 82.3 | 101 | 57.3 | – | unsupported | 1737 | 1689 |
| [`deim-n-coco`](../models/detection.md#dfine) | 42.3 | 40.7 | 22.9 | – | unsupported | 1556 | 1508 |
| [`deim-s-coco`](../models/detection.md#dfine) | 56.2 | 67.8 | 40.9 | – | unsupported | 1640 | 1542 |
| [`deim-x-coco`](../models/detection.md#dfine) | 159 | 186 | 97.7 | – | unsupported | 2254 | 2207 |
| [`detr-r101-coco`](../models/detection.md#detr) | 342 | 183 | 153 | – | unsupported | 2192 | 2051 |
| [`detr-r50-coco`](../models/detection.md#detr) | 237 | 125 | 105 | – | unsupported | 1954 | 1954 |
| [`dfine-l-coco`](../models/detection.md#dfine) | 136 | 132 | 71.0 | – | unsupported | 1758 | 1810 |
| [`dfine-m-coco`](../models/detection.md#dfine) | 108 | 99.3 | 56.6 | – | unsupported | 1700 | 1694 |
| [`dfine-n-coco`](../models/detection.md#dfine) | 45.4 | 40.9 | 25.2 | – | unsupported | 1663 | 1510 |
| [`dfine-s-coco`](../models/detection.md#dfine) | 74.1 | 66.5 | 41.0 | – | unsupported | 1674 | 1545 |
| [`dfine-x-coco`](../models/detection.md#dfine) | 185 | 183 | 101 | – | unsupported | 1978 | 2211 |
| [`faster-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 240 | 214 | 192 | – | unsupported | 1938 | 1932 |
| [`grounding-dino-base`](../models/detection.md#grounding_dino) | 1132 | 1075 | – | unsupported | unsupported | 5081 | 3952 |
| [`grounding-dino-tiny`](../models/detection.md#grounding_dino) | 912 | 846 | – | unsupported | – | 4899 | 4787 |
| [`lw-detr-large-coco`](../models/detection.md#lwdetr) | 88.2 | 88.5 | 72.7 | – | unsupported | 1765 | 1622 |
| [`lw-detr-medium-coco`](../models/detection.md#lwdetr) | 56.8 | 158 | 47.2 | – | unsupported | 1660 | 1525 |
| [`lw-detr-small-coco`](../models/detection.md#lwdetr) | 68.5 | 69.1 | 54.7 | – | unsupported | 1886 | 1781 |
| [`lw-detr-tiny-coco`](../models/detection.md#lwdetr) | 52.0 | 47.3 | 41.2 | – | unsupported | 1864 | 1742 |
| [`lw-detr-xlarge-coco`](../models/detection.md#lwdetr) | 157 | 168 | 130 | – | unsupported | 1964 | 1765 |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 245 | 222 | 203 | – | unsupported | 1964 | 1931 |
| [`mm-grounding-dino-base`](../models/detection.md#grounding_dino) | 1084 | 1273 | – | unsupported | – | 5074 | 4952 |
| [`mm-grounding-dino-tiny`](../models/detection.md#grounding_dino) | 882 | 1104 | – | unsupported | – | 4903 | 4804 |
| [`owlv2-b16`](../models/detection.md#owlv2) | 468 | 394 | – | – | unsupported | 2079 | 1515 |
| [`owlv2-l14`](../models/detection.md#owlv2) | 2176 | 2047 | – | – | unsupported | 4155 | 3556 |
| [`rf-detr-l-coco`](../models/detection.md#rfdetr) | 98.3 | 84.7 | 75.5 | – | unsupported | 1938 | 1646 |
| [`rf-detr-m-coco`](../models/detection.md#rfdetr) | 71.8 | 63.2 | 55.5 | – | unsupported | 2160 | 1660 |
| [`rf-detr-n-coco`](../models/detection.md#rfdetr) | 39.6 | 31.8 | 30.3 | – | unsupported | 2117 | 1630 |
| [`rf-detr-s-coco`](../models/detection.md#rfdetr) | 59.2 | 50.8 | 44.9 | – | unsupported | 2129 | 1644 |
| [`rf-detr-seg-2xl-coco`](../models/detection.md#rfdetr) | 243 | 213 | 184 | – | unsupported | 2022 | 1698 |
| [`rf-detr-seg-l-coco`](../models/detection.md#rfdetr) | 103 | 89.2 | 79.1 (golden fail) | – | unsupported | 1966 | 1741 |
| [`rf-detr-seg-m-coco`](../models/detection.md#rfdetr) | 85.6 | 74.2 | 64.7 | – | unsupported | 1964 | 1668 |
| [`rf-detr-seg-n-coco`](../models/detection.md#rfdetr) | 54.3 | 44.8 | 41.8 | – | unsupported | 1958 | 1653 |
| [`rf-detr-seg-s-coco`](../models/detection.md#rfdetr) | 66.1 | 55.2 | 50.1 | – | unsupported | 1945 | 1651 |
| [`rf-detr-seg-xl-coco`](../models/detection.md#rfdetr) | 158 | 139 | 124 | – | unsupported | 1982 | 1669 |
| [`rtdetrv2-l-coco`](../models/detection.md#rtdetrv2) | 174 | 82.6 | 77.3 | – | unsupported | 2013 | 1823 |
| [`rtdetrv2-m-coco`](../models/detection.md#rtdetrv2) | 75.0 | 55.3 | 55.0 | – | unsupported | 1930 | 1747 |
| [`rtdetrv2-ms-coco`](../models/detection.md#rtdetrv2) | 122 | 67.9 | 62.7 | – | unsupported | 1996 | 1821 |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 65.2 | 44.0 | 44.4 | – | unsupported | 1768 | 1650 |
| [`rtdetrv2-x-coco`](../models/detection.md#rtdetrv2) | 238 | 114 | 109 | – | unsupported | 2356 | 2017 |
| [`rtmdet-l-coco`](../models/detection.md#rtmdet) | 109 | 85.1 | 75.8 | – | unsupported | 2230 | 1941 |
| [`rtmdet-m-coco`](../models/detection.md#rtmdet) | 71.9 | 56.0 | 51.4 | – | unsupported | 2155 | 1869 |
| [`rtmdet-m-person`](../models/detection.md#rtmdet) | 69.8 | 53.4 | 48.8 | – | unsupported | 2022 | 1753 |
| [`rtmdet-nano-person`](../models/detection.md#rtmdet) | 29.9 | 19.3 | 17.8 | – | unsupported | 1688 | 1426 |
| [`rtmdet-s-coco`](../models/detection.md#rtmdet) | 42.8 | 33.6 | 31.0 | – | unsupported | 1874 | 1636 |
| [`rtmdet-tiny-coco`](../models/detection.md#rtmdet) | 35.9 | 28.6 | 27.0 | – | unsupported | 1813 | 1545 |
| [`rtmdet-x-coco`](../models/detection.md#rtmdet) | 155 | 128 | 116 | – | unsupported | 2381 | 2124 |
| [`yolox-l`](../models/detection.md#yolox) | 92.9 | 88.7 | 79.4 | – | unsupported | 1738 | 1861 |
| [`yolox-m`](../models/detection.md#yolox) | 59.9 | 55.2 | 50.0 | – | unsupported | 1673 | 1640 |
| [`yolox-s`](../models/detection.md#yolox) | 36.7 | 33.4 | 30.7 (golden fail) | – | unsupported | 1537 | 1556 |
| [`yolox-tiny`](../models/detection.md#yolox) | 24.7 | 20.1 | 20.1 | – | unsupported | 1501 | 1492 |
| [`yolox-x`](../models/detection.md#yolox) | 133 | 135 | 123 | – | unsupported | 1925 | 2036 |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.

## Ocr

### Ocr, synthetic, 32x128

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`parseq`](../models/ocr.md#parseq) | unsupported | 23.1 | 23.6 | unsupported | 496 |
| [`parseq-tiny`](../models/ocr.md#parseq) | unsupported | 31.1 | 25.1 | unsupported | 329 |

Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: the card has no kernels mode.

### Ocr, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`ppocrv5-mobile`](../models/ocr.md#ppocrv5) | unsupported | 884 | 736 | unsupported | 2987 |
| [`ppocrv5-server`](../models/ocr.md#ppocrv5) | unsupported | 2067 | 1607 | unsupported | 6545 |

Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: the card has no kernels mode.

## Pose

### Pose, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| [`dwpose-l`](../models/pose.md#rtmpose) | 225 | 84.4 | 76.3 | 80.0 | – | 2.8x | 1.0x | 2059 | 1875 | 1936 |
| [`dwpose-l-384`](../models/pose.md#rtmpose) | 319 | 167 | 157 | 155 | – | 2.1x | 1.0x | 2034 | 1878 | 1938 |
| [`dwpose-m`](../models/pose.md#rtmpose) | 192 | 56.1 | 52.4 | 50.9 | – | 3.8x | 1.0x | 1991 | 1698 | 1813 |
| [`dwpose-s`](../models/pose.md#rtmpose) | 164 | 34.9 | 32.1 | 29.8 | – | 5.5x | 1.1x | 1885 | 1543 | 1614 |
| [`dwpose-t`](../models/pose.md#rtmpose) | 158 | 29.2 | 28.0 | 23.7 | – | 6.7x | 1.2x | 1817 | 1482 | 1557 |
| [`hrnet-w32`](../models/pose.md#hrnet) | 304 | 141 | 122 | – | unsupported | – | – | 2014 | 1675 | – |
| [`hrnet-w32-384`](../models/pose.md#hrnet) | 438 | 256 | 217 | – | unsupported | – | – | 2019 | 1690 | – |
| [`hrnet-w48`](../models/pose.md#hrnet) | 386 | 215 | 198 | – | unsupported | – | – | 2380 | 2065 | – |
| [`hrnet-w48-384`](../models/pose.md#hrnet) | 617 | 426 | 369 | – | unsupported | – | – | 2432 | 2067 | – |
| [`retinaface-mobilenet0.25`](../models/pose.md#retinaface) | 91.0 | 17.5 | 15.3 | – | unsupported | – | – | 1522 | 1423 | – |
| [`retinaface-resnet50`](../models/pose.md#retinaface) | 256 | 210 | 150 | – | unsupported | – | – | 1871 | 1913 | – |
| [`rf-detr-keypoint-preview`](../models/pose.md#rfdetr_keypoint) | 148 | 148 | 135 (golden fail) | – | unsupported | – | – | 2074 | 1830 | – |
| [`rtmo-l`](../models/pose.md#rtmo) | 88.0 | 68.8 | 61.9 | – | unsupported | – | – | 2101 | 1829 | – |
| [`rtmo-m`](../models/pose.md#rtmo) | 60.4 | 47.6 | 43.9 | – | unsupported | – | – | 2043 | 1802 | – |
| [`rtmo-s`](../models/pose.md#rtmo) | 37.4 | 32.1 | 31.0 | – | unsupported | – | – | 1870 | 1560 | – |
| [`rtmo-t`](../models/pose.md#rtmo) | 25.4 | 21.0 | 19.9 | – | unsupported | – | – | 1836 | 1506 | – |
| [`rtmpose-l`](../models/pose.md#rtmpose) | 216 | 77.6 | 70.0 | 71.8 | – | 3.0x | 1.0x | 2018 | 1866 | 1916 |
| [`rtmpose-l-384`](../models/pose.md#rtmpose) | 303 | 154 | 136 | 143 | – | 2.1x | 1.0x | 2039 | 1869 | 1913 |
| [`rtmpose-l-halpe26`](../models/pose.md#rtmpose) | 219 | 97.9 | 83.7 | 88.2 | – | 2.5x | 0.9x | 2014 | 1866 | 1920 |
| [`rtmpose-l-halpe26-384`](../models/pose.md#rtmpose) | 305 | 185 | 160 | 172 | – | 1.8x | 0.9x | 2039 | 1871 | 1919 |
| [`rtmpose-m`](../models/pose.md#rtmpose) | 185 | 50.2 | 47.5 | 45.1 | – | 4.1x | 1.1x | 1920 | 1645 | 1758 |
| [`rtmpose-m-384`](../models/pose.md#rtmpose) | 237 | 95.5 | 87.8 | 83.9 | – | 2.8x | 1.0x | 1936 | 1649 | 1761 |
| [`rtmpose-m-ap10k`](../models/pose.md#rtmpose) | 193 | 72.6 | 65.8 | 71.9 | – | 2.7x | 0.9x | 1935 | 1653 | 1759 |
| [`rtmpose-m-face6`](../models/pose.md#rtmpose) | 200 | 79.2 | 70.9 | 67.3 | – | 3.0x | 1.1x | 1958 | 1687 | 1800 |
| [`rtmpose-m-halpe26`](../models/pose.md#rtmpose) | 186 | 60.6 | 55.8 | 54.4 | – | 3.4x | 1.0x | 1918 | 1658 | 1761 |
| [`rtmpose-m-halpe26-384`](../models/pose.md#rtmpose) | 236 | 118 | 104 | 100 | – | 2.4x | 1.0x | 1935 | 1651 | 1761 |
| [`rtmpose-m-hand5`](../models/pose.md#rtmpose) | 192 | 57.5 | 55.0 | 50.0 | – | 3.8x | 1.1x | 1936 | 1639 | 1740 |
| [`rtmpose-s`](../models/pose.md#rtmpose) | 158 | 30.2 | 28.3 | 24.9 | – | 6.3x | 1.1x | 1839 | 1514 | 1578 |
| [`rtmpose-s-face6`](../models/pose.md#rtmpose) | 170 | 39.4 | 38.2 | 31.9 | – | 5.3x | 1.2x | 1861 | 1518 | 1587 |
| [`rtmpose-s-halpe26`](../models/pose.md#rtmpose) | 160 | 30.9 | 28.4 | 24.9 | – | 6.4x | 1.1x | 1831 | 1500 | 1567 |
| [`rtmpose-t`](../models/pose.md#rtmpose) | 157 | 25.7 | 25.6 | 20.1 | – | 7.8x | 1.3x | 1804 | 1460 | 1532 |
| [`rtmpose-t-face6`](../models/pose.md#rtmpose) | 162 | 32.6 | 31.3 | 25.2 | – | 6.4x | 1.2x | 1820 | 1461 | 1533 |
| [`rtmpose-t-halpe26`](../models/pose.md#rtmpose) | 156 | 25.5 | 25.6 | 20.0 | – | 7.8x | 1.3x | 1816 | 1454 | 1513 |
| [`rtmpose-x-halpe26-384`](../models/pose.md#rtmpose) | 395 | 239 | 213 | 225 | – | 1.8x | 1.0x | 3244 | 2017 | 2130 |
| [`rtmw-l`](../models/pose.md#rtmpose) | 294 | 142 | 125 | 137 | – | 2.1x | 0.9x | 2064 | 2079 | 2097 |
| [`rtmw-l-384`](../models/pose.md#rtmpose) | 455 | 278 | 248 | 268 | – | 1.7x | 0.9x | 2071 | 2076 | 2127 |
| [`rtmw-m`](../models/pose.md#rtmpose) | 243 | 96.6 | 90.0 | 95.8 | – | 2.5x | 0.9x | 2018 | 1784 | 1897 |
| [`rtmw-x`](../models/pose.md#rtmpose) | 365 | 208 | 184 | 201 | – | 1.8x | 0.9x | 2505 | 2266 | 2308 |
| [`rtmw-x-384`](../models/pose.md#rtmpose) | 585 | 406 | 363 | 393 | – | 1.5x | 0.9x | 3581 | 2268 | 3322 |
| [`vitpose-base`](../models/pose.md#vitpose) | 288 | 228 | 212 | – | 172 | 1.7x | 1.2x | 2309 | 1929 | 1995 |
| [`vitpose-base-simple`](../models/pose.md#vitpose) | 254 | 196 | 186 | – | 160 | 1.6x | 1.2x | 2270 | 1910 | 1977 |
| [`vitpose-plus-base`](../models/pose.md#vitpose) | 358 | 229 | 212 | – | 174 | 2.1x | 1.2x | 2557 | 1939 | 2006 |
| [`vitpose-plus-huge`](../models/pose.md#vitpose) | 1543 | 1247 | 1158 | – | 1182 | 1.3x | 1.0x | 6040 | 4956 | 4957 |
| [`vitpose-plus-large`](../models/pose.md#vitpose) | 842 | 646 | 603 | – | 526 | 1.6x | 1.1x | 5237 | 2663 | 2741 |
| [`vitpose-plus-small`](../models/pose.md#vitpose) | 213 | 101 | 92.3 | – | 66.5 | 3.2x | 1.4x | 1865 | 1545 | 1633 |
| [`yunet-n`](../models/pose.md#yunet) | 34.1 | 24.9 | 21.6 | – | unsupported | – | – | 1453 | 1377 | – |
| [`yunet-s`](../models/pose.md#yunet) | 30.5 | 21.1 | 19.4 | – | unsupported | – | – | 1456 | 1380 | – |

Note: no Metal kernels for this family (card mps_kernels).

## Features

### Features, synthetic, 224x224

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](../models/features.md#c_radio) | 21.7 | 19.9 | 18.5 | unsupported | – | – | 1841 | 1908 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | 297 | 196 | 155 | unsupported | – | – | 5681 | 7659 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | 121 | 109 | 92.9 | unsupported | – | – | 3596 | 4607 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | 62.3 | 56.7 | 50.1 | unsupported | – | – | 2561 | 2477 | – |
| [`c-radio-v4-h`](../models/features.md#c_radio) | 98.2 | 104 | 90.8 | unsupported | – | – | 3587 | 4600 | – |
| [`c-radio-v4-so400m`](../models/features.md#c_radio) | 66.7 | 69.7 | 61.8 | unsupported | – | – | 2561 | 2793 | – |
| [`clip-b16-laion2b`](../models/features.md#clip) | 19.8 | 19.1 | 17.1 | unsupported | – | – | 1348 | 1154 | – |
| [`clip-b16-openai`](../models/features.md#clip) | 18.5 | 18.4 | 17.9 | unsupported | – | – | 1592 | 1156 | – |
| [`clip-h14-378-dfn5b`](../models/features.md#clip) | 232 | 137 | 111 | unsupported | – | – | 7408 | 5676 | – |
| [`clip-h14-dfn5b`](../models/features.md#clip) | 199 | 130 | 110 | unsupported | – | – | 7403 | 5680 | – |
| [`clip-h14-laion2b`](../models/features.md#clip) | 186 | 129 | 105 | unsupported | – | – | 7403 | 5678 | – |
| [`clip-l14-336-openai`](../models/features.md#clip) | 86.1 | 66.8 | 61.6 | unsupported | – | – | 4454 | 3267 | – |
| [`clip-l14-datacompxl`](../models/features.md#clip) | 66.7 | 62.6 | 57.7 | unsupported | – | – | 4450 | 3263 | – |
| [`clip-l14-dfn2b`](../models/features.md#clip) | 73.5 | 67.3 | 58.7 | unsupported | – | – | 4452 | 3262 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | 69.9 | 63.4 | 57.8 | unsupported | – | – | 4449 | 3263 | – |
| [`clip-l14-openai`](../models/features.md#clip) | 73.5 | 64.3 | 59.4 | unsupported | – | – | 4925 | 3263 | – |
| [`convnext_base.fb_in22k`](../models/features.md#convnext_features) | 33.0 | 28.1 | 28.6 | unsupported | – | – | 3215 | 2056 | – |
| [`convnext_large.fb_in22k`](../models/features.md#convnext_features) | 61.6 | 43.3 | 41.4 | unsupported | – | – | 5446 | 2813 | – |
| [`convnext_small.fb_in22k`](../models/features.md#convnext_features) | 25.2 | 22.7 | 21.2 | unsupported | – | – | 2497 | 883 | – |
| [`convnext_tiny.fb_in22k`](../models/features.md#convnext_features) | 14.6 | 12.3 | 12.1 | unsupported | – | – | 2148 | 671 | – |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 28.5 | 26.1 | 24.7 | unsupported | – | – | 998 | 778 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 51.0 | 48.2 | 46.4 | unsupported | – | – | 1013 | 779 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 271 | 251 | 218 | unsupported | – | – | 8904 | 4504 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 294 | 299 | 261 | unsupported | – | – | 8897 | 4505 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 75.9 | 77.6 | 65.2 | unsupported | – | – | 3165 | 1564 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 107 | 106 | 95.7 | unsupported | – | – | 3168 | 1566 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 12.8 | 13.6 | 12.7 | unsupported | – | – | 476 | 346 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 24.8 | 25.0 | 23.7 | unsupported | – | – | 476 | 346 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 32.2 | 26.6 | 24.2 | unsupported | – | – | 1749 | 591 | – |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 49.1 | 44.2 | 38.4 | unsupported | – | – | 2107 | 1927 | – |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 24.1 | 21.4 | 19.3 | unsupported | – | – | 675 | 453 | – |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 13.2 | 12.1 | 11.3 | unsupported | – | – | 579 | 346 | – |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 24.9 | 25.0 | 22.5 | unsupported | – | – | 721 | 729 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 174 | 167 | 143 | unsupported | – | – | 4593 | 3467 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 71.2 | 71.7 | 61.5 | unsupported | – | – | 2465 | 1473 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 13.4 | 13.9 | 13.5 | unsupported | – | – | 446 | 352 | – |
| [`pe-core-b16-224`](../models/features.md#perception_encoder) | 25.2 | 19.8 | 20.1 | 16.7 | 1.5x | 1.2x | 1440 | 1059 | 1215 |
| [`pe-core-g14-448`](../models/features.md#perception_encoder) | 639 | 369 | 367 | 313 | 2.0x | 1.2x | 16765 | 10793 | 11495 |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | 102 | 69.3 | 69.4 | 58.5 | 1.7x | 1.2x | 4553 | 3314 | 3319 |
| [`pe-core-s16-384`](../models/features.md#perception_encoder) | 24.8 | 10.4 | 11.0 | 8.80 | 2.8x | 1.2x | 632 | 517 | 557 |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | 16.3 | 7.43 | 7.94 | 10.6 | 1.5x | 0.7x | 406 | 362 | 368 |
| [`pe-spatial-b16-512`](../models/features.md#perception_encoder) | 47.1 | 19.4 | 19.4 | 16.0 | 2.9x | 1.2x | 1353 | 1160 | 1158 |
| [`pe-spatial-g14-448`](../models/features.md#perception_encoder) | 645 | 376 | 1484 | 2966 | 0.2x | 0.1x | 16536 | 9688 | 11039 |
| [`pe-spatial-l14-448`](../models/features.md#perception_encoder) | 106 | 70.8 | 83.0 | 69.8 | 1.5x | 1.0x | 4453 | 3265 | 3273 |
| [`pe-spatial-s16-512`](../models/features.md#perception_encoder) | 24.2 | 12.8 | 12.8 | 8.97 | 2.7x | 1.4x | 608 | 500 | 509 |
| [`pe-spatial-t16-512`](../models/features.md#perception_encoder) | 16.4 | 7.39 | 8.41 | 6.21 | 2.6x | 1.2x | 402 | 360 | 365 |
| [`siglip2-b16-256`](../models/features.md#siglip2) | 34.1 | 20.7 | 16.9 | unsupported | – | – | 1432 | 1213 | – |
| [`siglip2-b16-512`](../models/features.md#siglip2) | 51.1 | 18.8 | 16.5 | unsupported | – | – | 1439 | 1218 | – |
| [`siglip2-g16-256`](../models/features.md#siglip2) | 227 | 201 | 148 | unsupported | – | – | 11991 | 8847 | – |
| [`siglip2-g16-384`](../models/features.md#siglip2) | 278 | 181 | 143 | unsupported | – | – | 11329 | 9238 | – |
| [`siglip2-l16-256`](../models/features.md#siglip2) | 90.5 | 50.9 | 45.9 | unsupported | – | – | 4541 | 3307 | – |
| [`siglip2-l16-512`](../models/features.md#siglip2) | 114 | 51.8 | 45.6 | unsupported | – | – | 4549 | 3314 | – |
| [`siglip2-so400m14-224`](../models/features.md#siglip2) | 92.5 | 75.8 | 70.2 | unsupported | – | – | 5572 | 3909 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | 117 | 76.0 | 68.9 | unsupported | – | – | 5579 | 3915 | – |
| [`siglip2-so400m16-256`](../models/features.md#siglip2) | 89.3 | 65.5 | 59.0 | unsupported | – | – | 5574 | 3912 | – |
| [`siglip2-so400m16-512`](../models/features.md#siglip2) | 118 | 65.5 | 57.2 | unsupported | – | – | 5580 | 3917 | – |
| [`tipsv2-b14`](../models/features.md#tipsv2) | 31.7 | 25.9 | 24.0 | unsupported | – | – | 930 | 777 | – |
| [`tipsv2-l14`](../models/features.md#tipsv2) | 78.5 | 72.3 | 61.7 | unsupported | – | – | 3149 | 1574 | – |

Note: no Metal kernels for this family (card mps_kernels).

## Retrieval

### Retrieval, synthetic, 480x640

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`boq-dinov2`](../models/retrieval.md#boq) | 63.7 | 57.2 | 48.2 | unsupported | 1448 | 849 |
| [`megaloc`](../models/retrieval.md#megaloc) | 58.5 | 53.8 | 45.2 | unsupported | 2092 | 2617 |

Note: no Metal kernels for this family (card mps_kernels).

## Calibration

### Calibration, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`anycalib-pinhole`](../models/calibration.md#anycalib) | 340 | 266 | – | unsupported | 3236 | 3023 |
| [`geocalib-pinhole`](../models/calibration.md#geocalib) | 529 | 517 | 427 | unsupported | 1908 | 1644 |

Note: no Metal kernels for this family (card mps_kernels).

## Multiview

### Multiview, synthetic, 336x518

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| [`anysplat`](../models/multiview.md#anysplat) | unsupported | 5566 | 5164 | – | unsupported | – | – | – | 8809 | – |
| [`depthsplat-small-re10k`](../models/multiview.md#depthsplat) | unsupported | 697 | 749 | – | unsupported | – | – | – | 3779 | – |
| [`mapanything-apache`](../models/multiview.md#mapanything) | 8492 | 7357 | 6369 | – | unsupported | – | – | 11136 | 7218 | – |
| [`mvsplat-re10k`](../models/multiview.md#mvsplat) | unsupported | 1040 | 1110 | 896 | unsupported | – | 1.2x | – | 4532 | 4293 |
| [`pi3`](../models/multiview.md#pi3) | 10379 | 10298 | 8341 | – | unsupported | – | – | 6238 | 6510 | – |
| [`vggt-1b`](../models/multiview.md#vggt) | 7987 | 7223 | 6193 | 7016 | – | 1.1x | 0.9x | 10337 | 8700 | 10692 |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | 7828 | 7938 | 7242 | – | unsupported | – | – | 9079 | 8602 | – |

Note: the authors' code builds its DPT position embedding in float64 (src/model/encoder/vggt/heads/utils.py:48, torch.arange(dtype=torch.double)), which MPS does not support; its vendored VGGT predates VGGT's own MPS fix.
Note: no Metal kernels for this family (card mps_kernels).
Note: the authors' spherical-harmonic rotation (src/misc/sh_rotation.py) calls e3nn's wigner_D, which builds complex tensors on the device; MPS cannot concatenate complex tensors (cat_int32_t_float_float2).
Note: upstream's Gaussian rotation (e3nn Wigner D) builds complex tensors, which MPS cannot concatenate (cat_int32_t_float_float2).

### Multiview, synthetic, 448x672

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | unsupported | 1050 | 1034 | unsupported | 3805 |

Note: the original's compiled (TorchScript) functions fail with variable arguments under torch 2.14.
Note: no Metal kernels for this family (card mps_kernels).

### Multiview, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`lingbot-map`](../models/multiview.md#lingbot_map) | 9234 | 7609 | unsupported | 5962 |
| [`resplat-base-re10k`](../models/multiview.md#resplat) | OOM | – | unsupported | – |
| [`resplat-small-dl3dv`](../models/multiview.md#resplat) | 14422 | – | unsupported | 12080 |

Note: no Metal kernels for this family (card mps_kernels).

### Multiview, synthetic, 1536x1024

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 875 | 1008 | 1019 | unsupported | 3724 | 3614 |

Note: no Metal kernels for this family (card mps_kernels).

## Flow

### Flow, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs original | vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| [`flowformer-small-chairs`](../models/flow.md#flowformer) | OOM | 25589 | 17074 | unsupported | – | – | – | – | 4797 | – |
| [`flowformer-small-sintel`](../models/flow.md#flowformer) | OOM | 25282 | 17045 | unsupported | – | – | – | – | 5821 | – |
| [`flowformer-small-things`](../models/flow.md#flowformer) | OOM | 25313 | 17023 | unsupported | – | – | – | – | 5822 | – |
| [`flowseek-t`](../models/flow.md#flowseek) | OOM | 2787 | 2756 | unsupported | – | – | – | – | 8660 | – |
| [`flowseek-t-things`](../models/flow.md#flowseek) | OOM | 2703 | 2757 | unsupported | – | – | – | – | 10686 | – |
| [`flowseek-t-things-notartan`](../models/flow.md#flowseek) | OOM | 2694 | 2750 | unsupported | – | – | – | – | 10687 | – |
| [`gmflow-scale1-mixdata`](../models/flow.md#gmflow) | 4893 | 3845 | 2904 | unsupported | – | – | – | 12995 | 7113 | – |
| [`gmflow-scale1-things`](../models/flow.md#gmflow) | 5143 | 3780 | 2756 | unsupported | – | – | – | 13106 | 7112 | – |
| [`gmflow-scale2-mixdata`](../models/flow.md#gmflow) | OOM | 8494 | 6039 | unsupported | – | – | – | – | 9667 | – |
| [`gmflow-scale2-regrefine6-kitti`](../models/flow.md#gmflow) | OOM | 14645 | 10998 | unsupported | – | – | – | – | 9696 | – |
| [`gmflow-scale2-regrefine6-mixdata`](../models/flow.md#gmflow) | OOM | 14901 | 11156 | unsupported | – | – | – | – | 9694 | – |
| [`gmflow-scale2-regrefine6-sintelft`](../models/flow.md#gmflow) | OOM | 14622 | 10867 | unsupported | – | – | – | – | 9689 | – |
| [`gmflow-scale2-regrefine6-things`](../models/flow.md#gmflow) | OOM | 14869 | 11300 | unsupported | – | – | – | – | 9692 | – |
| [`gmflow-scale2-sintel`](../models/flow.md#gmflow) | OOM | 8697 | 5725 | unsupported | – | – | – | – | 9669 | – |
| [`gmflow-scale2-things`](../models/flow.md#gmflow) | OOM | 8489 | 5982 | unsupported | – | – | – | – | 9577 | – |
| [`megaflow`](../models/flow.md#megaflow) | OOM | 16278 | 11141 | – | unsupported | – | – | – | 8832 | – |
| [`memflow`](../models/flow.md#memflow) | OOM | 10016 | – | – | unsupported | – | – | – | 12659 | – |
| [`memflow-kitti`](../models/flow.md#memflow) | OOM | 9826 | – | – | unsupported | – | – | – | 12656 | – |
| [`memflow-sintel`](../models/flow.md#memflow) | OOM | 10075 | – | – | unsupported | – | – | – | 12787 | – |
| [`memfof`](../models/flow.md#memfof) | 3625 | 3054 | – | – | unsupported | – | – | 5212 | 5038 | – |
| [`memfof-kitti`](../models/flow.md#memfof) | – | OOM | – | – | unsupported | – | – | – | – | – |
| [`memfof-sintel`](../models/flow.md#memfof) | – | OOM | – | – | unsupported | – | – | – | – | – |
| [`memfof-spring`](../models/flow.md#memfof) | – | 3060 | – | – | unsupported | – | – | – | 5037 | – |
| [`neuflow-v2`](../models/flow.md#neuflow_v2) | OOM | 761 | 702 | – | unsupported | – | – | – | 11081 | – |
| [`neuflow-v2-sintel`](../models/flow.md#neuflow_v2) | OOM | 739 | 721 | – | unsupported | – | – | – | 11374 | – |
| [`neuflow-v2-things`](../models/flow.md#neuflow_v2) | OOM | 762 | 739 | – | unsupported | – | – | – | 11717 | – |
| [`raft-large`](../models/flow.md#raft) | OOM | 2522 | 2710 | – | 1351 | – | 1.9x | – | 12889 | 3647 |
| [`raft-large-kitti`](../models/flow.md#raft) | OOM | 10206 | 2270 | – | 1349 | – | 1.7x | – | 12391 | 4671 |
| [`raft-large-things`](../models/flow.md#raft) | OOM | 3992 | 1641 | – | 1348 | – | 1.2x | – | 9662 | 4670 |
| [`raft-small`](../models/flow.md#raft) | OOM | 4513 | 602 | – | 386 | – | 1.6x | – | 7468 | 2482 |
| [`recover-cx`](../models/flow.md#recover) | 1244 | 1107 | 853 | – | unsupported | – | – | 5112 | 2659 | – |
| [`recover-mn`](../models/flow.md#recover) | 626 | 463 | 347 | – | unsupported | – | – | 4972 | 2562 | – |
| [`recover-rn`](../models/flow.md#recover) | 667 | 497 | 405 | – | unsupported | – | – | 3891 | 1519 | – |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 2637 | 1820 | 1635 | – | 1220 | 2.2x | 1.3x | 12444 | 9732 | 4752 |
| [`sea-raft-m-kitti`](../models/flow.md#sea_raft) | OOM | 1423 | 1507 | – | 1222 | – | 1.2x | – | 9778 | 4754 |
| [`sea-raft-m-spring`](../models/flow.md#sea_raft) | 354 | 390 | 350 | – | 340 | 1.0x | 1.0x | 2923 | 2632 | 1621 |
| [`sea-raft-m-things`](../models/flow.md#sea_raft) | 2005 | 1425 | 1491 | – | 1221 | 1.6x | 1.2x | 13022 | 9778 | 4753 |
| [`sea-raft-s`](../models/flow.md#sea_raft) | 1531 | 1049 | 1098 | – | 833 | 1.8x | 1.3x | 13174 | 9653 | 4667 |
| [`sea-raft-s-kitti`](../models/flow.md#sea_raft) | 1503 | 2000 | 1098 | – | 831 | 1.8x | 1.3x | 13130 | 9671 | 4664 |
| [`sea-raft-s-spring`](../models/flow.md#sea_raft) | 255 | 268 | 244 | – | 233 | 1.1x | 1.0x | 2757 | 2547 | 1527 |
| [`waft-a1`](../models/flow.md#waft) | 22117 | 8225 | 7168 | unsupported | unsupported | – | – | 12159 | 7185 | – |
| [`waft-a1-kitti`](../models/flow.md#waft) | OOM | 8502 | 7384 | unsupported | unsupported | – | – | – | 7187 | – |
| [`waft-a1-sintel`](../models/flow.md#waft) | OOM | 8835 | 7430 | unsupported | unsupported | – | – | – | 7187 | – |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.

## Matching

### Matching, synthetic, 768x1024

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels bf16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): original | portable | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| [`aliked-flexglue`](../models/matching.md#flexglue) | – | 1124 | 1069 | unsupported | – | – | – | – | 3088 | – |
| [`aliked-lightglue`](../models/matching.md#lightglue) | 214 | 206 | – | unsupported | unsupported | – | – | 2720 | 2586 | – |
| [`dedode-v2-b`](../models/matching.md#dedode) | 1711 | 1464 | – | unsupported | unsupported | – | – | 8024 | 8011 | – |
| [`dedode-v2-g`](../models/matching.md#dedode) | 4459 | 3767 | – | unsupported | unsupported | – | – | 9577 | 10008 | – |
| [`disk-lightglue`](../models/matching.md#lightglue) | 1780 | 1994 | – | unsupported | unsupported | – | – | 3679 | 3531 | – |
| [`loftr-indoor`](../models/matching.md#loftr) | 994 | 877 | 701 | unsupported | unsupported | – | – | 5973 | 3613 | – |
| [`loftr-outdoor`](../models/matching.md#loftr) | 997 | 875 | 723 | unsupported | unsupported | – | – | 6998 | 3614 | – |
| [`loma-b`](../models/matching.md#loma) | – | 3041 | – | unsupported | – | – | – | – | 8137 | – |
| [`loma-b128`](../models/matching.md#loma) | – | 1065 | – | – | unsupported | – | – | – | 4740 | – |
| [`loma-b64`](../models/matching.md#loma) | – | 917 | – | – | unsupported | – | – | – | 4674 | – |
| [`loma-g`](../models/matching.md#loma) | – | 3628 | – | – | unsupported | – | – | – | 8495 | – |
| [`loma-l`](../models/matching.md#loma) | – | 3178 | – | – | unsupported | – | – | – | 7941 | – |
| [`loma-r`](../models/matching.md#loma) | – | 3076 | – | – | unsupported | – | – | – | 8124 | – |
| [`romav1`](../models/matching.md#romav1) | 12208 | 3050 | – | – | unsupported | – | – | 12492 | 7979 | – |
| [`romav2`](../models/matching.md#romav2) | OOM | 5130 | – | 3981 | – | unsupported | 1.3x | – | 7278 | 6725 |
| [`xfeat`](../models/matching.md#xfeat) | 45.2 | 41.7 | – | unsupported | – | – | – | 1549 | 1471 | – |
| [`xfeat-lighterglue`](../models/matching.md#xfeat) | 192 | 179 | – | unsupported | – | – | – | 1570 | 1499 | – |

Note: the card has no kernels mode.
Note: no Metal kernels for this family (card mps_kernels).

## Tracking

### Tracking, synthetic, 480x854

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`alltracker`](../models/tracking.md#alltracker) | 3628 | 3218 | 3067 | unsupported | 6081 | 3782 |
| [`alltracker-tiny`](../models/tracking.md#alltracker) | 3057 | 3144 | 2743 | unsupported | 5785 | 3640 |
| [`bootstapir`](../models/tracking.md#tapir) | 1930 | 1704 | 1719 | unsupported | 2931 | 1743 |
| [`cotracker3-offline`](../models/tracking.md#cotracker) | – | 1528 | 1658 | unsupported | – | 2737 |
| [`cotracker3-online`](../models/tracking.md#cotracker) | – | 1829 | 1913 | unsupported | – | 2734 |
| [`locotrack-b`](../models/tracking.md#locotrack) | 1973 | 1786 | 2084 | unsupported | 2491 | 1470 |
| [`locotrack-s`](../models/tracking.md#locotrack) | 1821 | 1781 | 1951 | unsupported | 2433 | 2427 |
| [`tapir`](../models/tracking.md#tapir) | 1548 | 1356 | 1393 | unsupported | 2759 | 1604 |
| [`tapnext`](../models/tracking.md#tapnext) | 6104 | 5855 | 5998 | unsupported | 3210 | 3754 |
| [`tapnext-pp`](../models/tracking.md#tapnext) | – | 5610 | 5654 | unsupported | – | 3657 |
| [`track-on-r`](../models/tracking.md#track_on) | 11219 | 8274 | – | unsupported | 3052 | 6102 |
| [`track-on2`](../models/tracking.md#track_on) | 11323 | 8370 | – | unsupported | 3052 | 3823 |

Note: no Metal kernels for this family (card mps_kernels).

## Interpolation

### Interpolation, synthetic, 1080x1920

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|
| [`ema-vfi`](../models/interpolation.md#ema_vfi) | 36849 | 19632 | 1981 | unsupported | 10614 | 6883 |
| [`ema-vfi-small`](../models/interpolation.md#ema_vfi) | 1070 | 736 | 556 | unsupported | 5821 | 3587 |
| [`ema-vfi-small-t`](../models/interpolation.md#ema_vfi) | 1058 | 792 | 513 | unsupported | 5814 | 3583 |
| [`ema-vfi-t`](../models/interpolation.md#ema_vfi) | 5764 | 28800 | 2272 | unsupported | 11039 | 6882 |
| [`film`](../models/interpolation.md#film) | 10448 | 6339 | 4477 | unsupported | 12794 | 9219 |
| [`gimm-vfi-r`](../models/interpolation.md#gimm_vfi) | unsupported | OOM | 17906 | unsupported | – | 11886 |
| [`gimm-vfi-r-p`](../models/interpolation.md#gimm_vfi) | unsupported | OOM | 18002 | unsupported | – | 11881 |
| [`rife-4.25`](../models/interpolation.md#rife) | 169 | 119 | 106 | unsupported | 3588 | 2482 |
| [`rife-4.25-lite`](../models/interpolation.md#rife) | 145 | 91.0 | 80.0 | unsupported | 3576 | 2447 |
| [`rife-4.26`](../models/interpolation.md#rife) | 170 | 118 | 103 | unsupported | 3586 | 2479 |

Note: no Metal kernels for this family (card mps_kernels).
Note: original needs cupy (CUDA softsplat kernels), no CUDA on the Mac.

## Restoration

### Restoration, synthetic, 512x512

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels bf16 | peak memory (MiB, process footprint, fastest dtype): original | portable |
|---|---|---|---|---|---|---|---|---|
| [`big-lama`](../models/restoration.md#lama) | 334 | 241 | 214 | – | unsupported | – | 2096 | 1679 |
| [`codeformer`](../models/restoration.md#codeformer) | 274 | 260 | 196 | unsupported | unsupported | – | 3182 | 1866 |
| [`dat-2-x4`](../models/restoration.md#dat) | 89592 | 11525 | 9404 | – | unsupported | – | 13698 | 10600 |
| [`dat-light-x4`](../models/restoration.md#dat) | 3668 | 2992 | 2562 | – | unsupported | – | 6317 | 6851 |
| [`dat-x4`](../models/restoration.md#dat) | 18351 | 21047 | 10512 | – | unsupported | – | 13512 | 10620 |
| [`drct-l-x4`](../models/restoration.md#drct) | OOM | 16574 | 12431 | – | unsupported | – | – | 5367 |
| [`drct-x4`](../models/restoration.md#drct) | OOM | 8592 | 6277 | – | unsupported | – | – | 5688 |
| [`gfpgan-v1.4`](../models/restoration.md#gfpgan) | 172 | 153 | 138 | – | unsupported | – | 2871 | 1982 |
| [`hat-s-x4`](../models/restoration.md#hat) | OOM | 15070 | 13687 | – | unsupported | – | – | 13390 |
| [`hat-x4`](../models/restoration.md#hat) | OOM | 11505 | 8044 | – | unsupported | – | – | 6650 |
| [`nafnet-gopro`](../models/restoration.md#nafnet) | 645 | 409 | 368 | – | unsupported | – | 3634 | 1715 |
| [`nafnet-sidd`](../models/restoration.md#nafnet) | 802 | 423 | 442 | – | unsupported | – | 3427 | 3242 |
| [`real-hat-gan-x4`](../models/restoration.md#hat) | OOM | 10894 | 7775 | – | unsupported | – | – | 6652 |
| [`realesr-general-x4v3`](../models/restoration.md#compact) | 176 | 178 | 132 | – | unsupported | – | 1302 | 1289 |
| [`realesrgan-x2plus`](../models/restoration.md#realesrgan) | 675 | 709 | 549 | – | unsupported | – | 1696 | 1440 |
| [`realesrgan-x4plus`](../models/restoration.md#realesrgan) | 2991 | 3022 | 2066 | – | unsupported | – | 4713 | 2466 |
| [`realesrgan-x4plus-anime-6b`](../models/restoration.md#realesrgan) | 938 | 973 | 664 | – | unsupported | – | 4517 | 2351 |
| [`realplksr-nomoswebphoto-x4`](../models/restoration.md#realplksr) | 1555 | 1527 | 1092 | – | unsupported | – | 1407 | 1368 |
| [`restormer-deblur`](../models/restoration.md#restormer) | 2395 | 989 | 990 | – | unsupported | – | 6858 | 4726 |
| [`restormer-derain`](../models/restoration.md#restormer) | 2382 | 995 | 988 | – | unsupported | – | 5863 | 3653 |
| [`restormer-sidd`](../models/restoration.md#restormer) | 2183 | 1112 | 1132 | – | unsupported | – | 5863 | 5777 |
| [`seedvr2-3b`](../models/restoration.md#seedvr2) | unsupported | unsupported | – | – | – | unsupported | – | – |
| [`seedvr2-7b`](../models/restoration.md#seedvr2) | unsupported | unsupported | – | – | – | unsupported | – | – |
| [`span-x2`](../models/restoration.md#span) | 115 | 100 | 74.0 | – | unsupported | – | 2391 | 1296 |
| [`span-x4`](../models/restoration.md#span) | 117 | 103 | 89.0 | – | unsupported | – | 2388 | 1296 |
| [`swin2sr-x2`](../models/restoration.md#swin2sr) | 14608 | 5731 | 4607 | – | unsupported | – | 5529 | 3495 |
| [`swin2sr-x4`](../models/restoration.md#swin2sr) | 12965 | 5793 | 4690 | – | unsupported | – | 7727 | 4581 |
| [`swin2sr-x4-real`](../models/restoration.md#swin2sr) | 13527 | 5894 | 4751 | – | unsupported | – | 7724 | 4576 |
| [`swinir-l-x4-real`](../models/restoration.md#swinir) | 16883 | 10185 | 7252 | – | unsupported | – | 10504 | 5594 |
| [`swinir-m-x2`](../models/restoration.md#swinir) | 8690 | 4760 | 3674 | – | unsupported | – | 5520 | 3492 |
| [`swinir-m-x4`](../models/restoration.md#swinir) | 8782 | 4824 | 3750 | – | unsupported | – | 7699 | 5064 |
| [`swinir-m-x4-real`](../models/restoration.md#swinir) | 8901 | 5045 | 3829 | – | unsupported | – | 7694 | 5059 |

Note: no Metal kernels for this family (card mps_kernels).
Note: the card has no kernels mode.
Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: the family is not validated on MPS (card mps_dtypes).

## Volume

### Volume, synthetic, 192x192

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`nnfoundation-vit`](../models/volume.md#nnfoundation) | unsupported | 21342 | 17636 | unsupported | 3817 |

Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: no Metal kernels for this family (card mps_kernels).

### Volume, synthetic, 200x200

| model | time (ms): original fp32 | portable fp32 | portable fp16 | kernels fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|---|
| [`nnfoundation-cnn`](../models/volume.md#nnfoundation) | unsupported | 3736 | 3302 | unsupported | 9195 |
| [`nninteractive`](../models/volume.md#nninteractive) | – | 3525 | – | unsupported | 11439 |
| [`totalsegmentator-total`](../models/volume.md#totalsegmentator) | unsupported | 27804 | 23070 | unsupported | 6058 |
| [`totalsegmentator-total-3mm`](../models/volume.md#totalsegmentator) | unsupported | 593 | 522 | unsupported | 2588 |

Note: no local original adapter for this family in tools/_original.py (the authors' code was not run on this Mac).
Note: no Metal kernels for this family (card mps_kernels).
Note: no upstream adapter in tools/_original.py; original needs the nnU-Net/TotalSegmentator package (not in the Mac venv) and its 3D inference does not fit 16 GB here.
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
