# Apple M1 Pro 16 GB (MLX)

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs\data\speed\apple-m1-pro-mlx-16gb.jsonl) -->

| | |
|---|---|
| device | Apple M1 Pro 16 GB (MLX) |
| backend | mlx |
| memory | 16 GB unified |
| hardware | 16-core GPU; the families with an MLX port (device="mlx") |
| software | macOS 27, mlx 0.32.4, torch 2.14 |
| torch (rows) | 2.14.1 |
| measured | 2026-11-19 to 2026-11-21 |
| rows | 758 in `docs\data\speed\apple-m1-pro-mlx-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`aanet-sceneflow`](../models/stereo.md#aanet) | 1724 | 1695 | – | – | – | 2131 | – |
| [`bridgedepth-sf`](../models/stereo.md#bridgedepth) | 5211 | 5661 | – | – | – | 5075 | – |
| [`c-foundationstereo-s`](../models/stereo.md#c_foundationstereo) | 62607 | 27235 | – | – | – | 4625 | – |
| [`cas-gwcnet-gc-sceneflow`](../models/stereo.md#cas_gwcnet) | 4595 | 3841 | unsupported | – | – | 4754 | – |
| [`cfnet-sceneflow`](../models/stereo.md#cfnet) | 5355 | – | unsupported | – | – | 4405 | – |
| [`coex-sceneflow`](../models/stereo.md#coex) | 765 | 608 | – | – | – | 3187 | – |
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 6118 | 4297 | 5212 | 3558 | 1.2x | 1557 | 1863 |
| [`defom-stereo-vitl-sceneflow`](../models/stereo.md#defom_stereo) | 15944 | – | – | – | – | 6974 | – |
| [`dlnr-sceneflow`](../models/stereo.md#dlnr) | 4221 | 3762 | unsupported | – | – | 3937 | – |
| [`fast-acvnet-generalization`](../models/stereo.md#fast_acvnet) | 761 | 549 | unsupported | – | – | 1881 | – |
| [`foundationstereo-vit-l`](../models/stereo.md#foundationstereo) | – | 25847 | – | – | – | 4941 | – |
| [`foundationstereo-vit-s`](../models/stereo.md#foundationstereo) | 40293 | 20504 | – | 17814 | 1.2x | 4943 | 4613 |
| [`great-igev-sceneflow`](../models/stereo.md#great_igev) | 23111 | 14349 | – | – | – | 9356 | – |
| [`gwcnet-gc-sceneflow`](../models/stereo.md#gwcnet) | 4102 | 2933 | unsupported | – | – | 2876 | – |
| [`hitnet-eth3d`](../models/stereo.md#hitnet) | 723 | 572 | unsupported | – | – | 1697 | – |
| [`igev-plusplus-sceneflow`](../models/stereo.md#igev_plusplus) | 14770 | 10068 | 10986 | 5895 | 1.7x | 2993 | 3184 |
| [`igev-stereo-sceneflow`](../models/stereo.md#igev_stereo) | 6303 | 5678 | unsupported | – | – | 4566 | – |
| [`las2-s`](../models/stereo.md#las2) | 290 | 230 | unsupported | – | – | 988 | – |
| [`mocha-v1-sceneflow`](../models/stereo.md#mocha_stereo) | 11379 | 9657 | – | – | – | 3382 | – |
| [`monster-plusplus-mix-all`](../models/stereo.md#monster_plusplus) | 17687 | 11269 | unsupported | – | – | 3643 | – |
| [`monster-sceneflow`](../models/stereo.md#monster) | 16562 | 10863 | unsupported | – | – | 3633 | – |
| [`nmrf-sceneflow`](../models/stereo.md#nmrf) | 2580 | 2246 | unsupported | – | – | 3106 | – |
| [`psmnet-sceneflow`](../models/stereo.md#psmnet) | 3638 | 2597 | unsupported | – | – | 2941 | – |
| [`raft-stereo-middlebury`](../models/stereo.md#raft_stereo) | 7097 | 5704 | – | – | – | 4315 | – |
| [`raft-stereo-sceneflow`](../models/stereo.md#raft_stereo) | 6093 | 4212 | 5818 | 2986 | 1.4x | 3926 | 4650 |
| [`rt-monster-plusplus-mix-all`](../models/stereo.md#rt_monster_plusplus) | 1025 | 713 | 1020 | 622 | 1.1x | 2006 | 2046 |
| [`s2m2-s`](../models/stereo.md#s2m2) | 4036 | 1835 | unsupported | – | – | 2779 | – |
| [`selective-igev-sceneflow`](../models/stereo.md#selective_igev) | 11452 | 6659 | 10715 | 6296 | 1.1x | 2167 | 2735 |
| [`waft-stereo-s`](../models/stereo.md#waft_stereo) | 5474 | 3509 | unsupported | – | – | 1835 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Depth

### Depth, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`video-depth-anything-metric-s`](../models/depth.md#video_depth_anything) | 1246 | 872 | 2021 |
| [`video-depth-anything-s`](../models/depth.md#video_depth_anything) | 1300 | 896 | 2027 |

### Depth, synthetic, 756x1008

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`prompt-depth-anything-s`](../models/depth.md#prompt_depth_anything) | 250 | 244 | 1168 |

### Depth, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | portable bf16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`any2full-b`](../models/depth.md#any2full) | 379 | 379 | – | – | – | – | 1583 | – |
| [`any2full-l`](../models/depth.md#any2full) | 864 | 909 | – | – | – | – | 3827 | – |
| [`any2full-s`](../models/depth.md#any2full) | 169 | 160 | – | – | – | – | 1012 | – |
| [`da3-giant-1.1`](../models/depth.md#depth_anything_3) | 907 | 920 | – | – | – | – | 6114 | – |
| [`da3-large-1.1`](../models/depth.md#depth_anything_3) | 338 | 361 | – | – | – | – | 2773 | – |
| [`da3metric-large`](../models/depth.md#depth_anything_3) | 300 | 328 | – | – | – | – | 2710 | – |
| [`da3mono-large`](../models/depth.md#depth_anything_3) | 295 | 322 | – | – | – | – | 2710 | – |
| [`da3nested-giant-large-1.1`](../models/depth.md#depth_anything_3) | 1197 | 1269 | – | – | – | – | 7314 | – |
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 229 | 182 | – | 214 | 168 | 1.1x | 1026 | 1029 |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 669 | 522 | – | 628 | 501 | 1.0x | 1896 | 1768 |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | 237 | 184 | – | 216 | 170 | 1.1x | 1017 | 1027 |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | 700 | 527 | – | 641 | 581 | 0.9x | 1831 | 1737 |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | 96.4 | 79.8 | – | 106 | 68.5 | 1.2x | 737 | 710 |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | 291 | 188 | – | 232 | 198 | 0.9x | 1021 | 996 |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | 867 | 544 | – | 720 | 587 | 0.9x | 1728 | 1684 |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | 117 | 78.5 | – | 87.1 | 67.4 | 1.2x | 735 | 707 |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 93.0 | 73.4 | – | 88.2 | 66.3 | 1.1x | 727 | 698 |
| [`depth-pro`](../models/depth.md#depth_pro) | 7541 | 4653 | – | – | – | – | 5490 | – |
| [`depthfm-v1`](../models/depth.md#depthfm) | failed | failed | – | – | – | – | – | – |
| [`distill-any-depth-b`](../models/depth.md#depth_anything_v2) | 231 | 184 | – | 216 | 170 | 1.1x | 1027 | 1020 |
| [`distill-any-depth-l`](../models/depth.md#depth_anything_v2) | 685 | 529 | – | 640 | 544 | 1.0x | 1847 | 1769 |
| [`distill-any-depth-s`](../models/depth.md#depth_anything_v2) | 94.2 | 74.5 | – | 85.2 | 66.9 | 1.1x | 728 | 698 |
| [`lingbot-depth`](../models/depth.md#lingbot_depth) | 4367 | 4577 | – | unsupported | – | – | 3930 | – |
| [`lingbot-depth-dc`](../models/depth.md#lingbot_depth) | 4414 | 5338 | – | – | – | – | 3849 | – |
| [`lotus-normal-d-v1-1`](../models/depth.md#lotus) | 2872 | 1973 | – | unsupported | – | – | 5670 | – |
| [`lotus-normal-g-v1-1`](../models/depth.md#lotus) | 3063 | 2005 | – | – | – | – | 5674 | – |
| [`marigold-depth-v1-1`](../models/depth.md#marigold) | 5831 | 4702 | – | – | – | – | 4287 | – |
| [`marigold-normals-lcm-v0-1`](../models/depth.md#marigold_normals) | 5109 | 2363 | – | – | – | – | 4318 | – |
| [`marigold-v2-depth`](../models/depth.md#marigold_v2) | – | – | 51319 | – | – | – | 2251 | – |
| [`metric3d-vit-large`](../models/depth.md#metric3d) | 2273 | 2460 | – | – | – | – | 3902 | – |
| [`midas-dpt-beit-large-512`](../models/depth.md#midas) | 861 | 668 | – | – | – | – | 1994 | – |
| [`moge-2-vitb-normal`](../models/depth.md#moge) | 1214 | 1144 | – | unsupported | – | – | 2183 | – |
| [`moge-2-vitl-normal`](../models/depth.md#moge) | 2515 | 2081 | – | – | – | – | 3009 | – |
| [`moge-2-vits-normal`](../models/depth.md#moge) | 956 | 886 | – | unsupported | – | – | 2024 | – |
| [`moge-3-vitl`](../models/depth.md#moge) | 3508 | 3257 | – | – | – | – | 3170 | – |
| [`omni-dc`](../models/depth.md#omni_dc) | 3475 | 5978 | – | – | – | – | 6632 | – |
| [`pointdit-b-512`](../models/depth.md#pointdit) | 343 | 356 | – | – | – | – | 1695 | – |
| [`pointdit-l-512`](../models/depth.md#pointdit) | 1047 | 1231 | – | – | – | – | 4003 | – |
| [`prompt-depth-anything-l`](../models/depth.md#prompt_depth_anything) | 1404 | 1082 | – | – | – | – | 2757 | – |
| [`stablenormal-turbo`](../models/depth.md#stablenormal) | 3589 | 2299 | – | unsupported | – | – | 8683 | – |
| [`unidepth-v2-vitl14`](../models/depth.md#unidepth_v2) | 1691 | – | – | – | – | – | 3855 | – |
| [`unik3d-vitl`](../models/depth.md#unik3d) | 1396 | 1458 | – | – | – | – | 3000 | – |
| [`zoedepth-nk`](../models/depth.md#zoedepth) | 862 | – | – | – | – | – | 2895 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Classification

### Classification, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 19.2 | 16.2 | 19.6 | 16.5 | 1.0x | 1212 | 1212 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 19.3 | 16.9 | 19.7 | 16.2 | 1.0x | 1212 | 1212 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 39.0 | 34.1 | 39.2 | 35.2 | 1.0x | 1210 | 1212 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 33.8 | 29.2 | 34.9 | 29.0 | 1.0x | 2341 | 2340 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 34.1 | 29.1 | 34.0 | 29.7 | 1.0x | 2340 | 2340 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 73.3 | 63.4 | 73.3 | 63.5 | 1.0x | 2340 | 2341 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 14.1 | 12.4 | 14.2 | 12.3 | 1.0x | 812 | 813 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 14.2 | 12.4 | 14.2 | 12.2 | 1.0x | 813 | 812 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 27.9 | 25.5 | 28.0 | 25.0 | 1.0x | 813 | 811 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 8.45 | 7.26 | 8.45 | 7.46 | 1.0x | 642 | 646 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 8.41 | 7.51 | 8.48 | 7.52 | 1.0x | 644 | 646 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 17.4 | 15.2 | 17.3 | 15.3 | 1.0x | 714 | 707 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 22.2 | 19.7 | 22.1 | 19.7 | 1.0x | 1190 | 1192 |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 66.4 | 57.6 | unsupported | – | – | 3981 | – |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 4.84 | 4.39 | 4.77 | 4.52 | 1.0x | 345 | 344 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 18.1 | 16.8 | 17.8 | 16.0 | 1.1x | 645 | 649 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 19.0 | 17.1 | 19.1 | 16.7 | 1.0x | 553 | 552 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 3.47 | 2.78 | 3.25 | 2.80 | 1.0x | 319 | 318 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 3.20 | 3.31 | 3.31 | 4.51 | 1.0x | 339 | 338 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 2.85 | 2.29 | – | – | – | 367 | – |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 19.7 | 18.9 | 19.9 | 18.9 | 1.0x | 1047 | 1046 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 4.72 | 4.31 | – | – | – | 384 | – |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 4.65 | 4.30 | – | – | – | 384 | – |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 5.61 | 5.20 | – | – | – | 410 | – |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 5.78 | 5.29 | – | – | – | 414 | – |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 5.06 | 4.38 | – | – | – | 425 | – |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 5.11 | 4.41 | – | – | – | 425 | – |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 8.98 | 12.1 | – | – | – | 548 | – |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 8.86 | 6.88 | – | – | – | 506 | – |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 12.1 | 11.1 | – | – | – | 630 | – |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 12.1 | 10.7 | – | – | – | 628 | – |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 20.5 | 17.1 | 20.2 | 17.0 | 1.0x | 729 | 728 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 5.26 | 3.77 | 4.37 | 3.75 | 1.0x | 383 | 382 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 7.94 | 6.77 | 7.92 | 6.76 | 1.0x | 503 | 500 |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 5.95 | 3.15 | – | – | – | 342 | – |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 5.08 | 2.74 | – | – | – | 348 | – |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 5.28 | 3.31 | – | – | – | 375 | – |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 5.93 | 4.37 | – | – | – | 397 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Segmentation

### Segmentation, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`bisenet-face-parsing`](../models/segmentation.md#bisenet) | 21.0 | 17.6 | – | – | – | 607 | – |
| [`segformer-b2-clothes`](../models/segmentation.md#segformer) | 65.4 | 42.4 | 63.4 | 40.6 | 1.0x | 724 | 760 |
| [`segformer-b5-face-parsing`](../models/segmentation.md#segformer) | 137 | 101 | – | – | – | 963 | – |

### Segmentation, synthetic, 1024x2048

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`deeplabv3-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 654 | 457 | 645 | 452 | 1.0x | 1571 | 1588 |
| [`deeplabv3plus-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 613 | 444 | 511 | 365 | 1.2x | 1618 | 1638 |
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 796 | 761 | unsupported | – | – | 1611 | – |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 1459 | 1278 | unsupported | – | – | 2388 | – |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 607 | 600 | unsupported | – | – | 1232 | – |
| [`eomt-l-dinov2-ade20k`](../models/segmentation.md#eomt) | 676 | 572 | – | – | – | 3710 | – |
| [`eomt-l-dinov3-ade20k`](../models/segmentation.md#eomt) | 765 | 599 | – | – | – | 3724 | – |
| [`fcn-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 496 | 358 | 492 | 349 | 1.0x | 1567 | 1642 |
| [`hrnet-w48-fcn-ade20k`](../models/segmentation.md#mmseg) | 282 | 220 | – | – | – | 1112 | – |
| [`mask2former-swinl-ade20k`](../models/segmentation.md#mask2former) | 1084 | 1148 | – | – | – | 4323 | – |
| [`oneformer-swinl-ade20k`](../models/segmentation.md#oneformer) | 1758 | 2078 | – | – | – | 6587 | – |
| [`pspnet-r101-d8-ade20k`](../models/segmentation.md#mmseg) | 517 | 388 | 521 | 376 | 1.0x | 1582 | 1602 |
| [`sapiens-seg-0.3b-goliath`](../models/segmentation.md#sapiens) | 2664 | 2392 | – | – | – | 3921 | – |
| [`sapiens-seg-1b-goliath`](../models/segmentation.md#sapiens) | 5247 | 4110 | – | – | – | 6692 | – |
| [`segformer-b0-ade20k`](../models/segmentation.md#segformer) | 94.8 | 78.9 | 91.9 | 77.5 | 1.0x | 776 | 764 |
| [`segformer-b5-ade20k`](../models/segmentation.md#segformer) | 470 | 357 | – | – | – | 1382 | – |
| [`segnext-mscan-b-ade20k`](../models/segmentation.md#segnext) | 301 | 264 | – | – | – | 894 | – |
| [`segnext-mscan-l-ade20k`](../models/segmentation.md#segnext) | 489 | 424 | – | – | – | 974 | – |
| [`segnext-mscan-s-ade20k`](../models/segmentation.md#segnext) | 178 | 153 | 106 | 84.9 | 1.8x | 828 | 820 |
| [`segnext-mscan-t-ade20k`](../models/segmentation.md#segnext) | 133 | 117 | 85.1 | 67.7 | 1.7x | 769 | 754 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matting

### Matting, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ben2`](../models/matting.md#ben2) | 1706 | – | unsupported | – | – | 5012 | – |
| [`birefnet`](../models/matting.md#birefnet) | 1981 | 1203 | unsupported | – | – | 2705 | – |
| [`inspyrenet`](../models/matting.md#inspyrenet) | 1273 | – | unsupported | – | – | 2332 | – |
| [`isnet-general-use`](../models/matting.md#isnet) | 211 | 145 | 209 | 122 | 1.2x | 1217 | 1328 |
| [`u2net`](../models/matting.md#u2net) | 82.5 | 62.9 | unsupported | – | – | 1156 | – |
| [`vitmatte-b`](../models/matting.md#vitmatte) | 2502 | 2368 | – | – | – | 2474 | – |
| [`vitmatte-s`](../models/matting.md#vitmatte) | 1317 | 1280 | unsupported | – | – | 2283 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Edges

### Edges, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`anyline`](../models/edges.md#anyline) | 223 | 197 | 219 | 181 | 1.1x | 1776 | 1604 |
| [`hed`](../models/edges.md#hed) | 81.9 | – | unsupported | – | – | 1144 | – |
| [`lineart`](../models/edges.md#lineart) | 134 | 108 | unsupported | – | – | 1099 | – |
| [`lineart-anime`](../models/edges.md#lineart_anime) | 51.8 | 44.5 | unsupported | – | – | 982 | – |
| [`mlsd`](../models/edges.md#mlsd) | 46.3 | 35.8 | unsupported | – | – | 676 | – |
| [`pidinet`](../models/edges.md#pidinet) | 196 | 173 | 107 | 75.1 | 2.3x | 728 | 732 |
| [`teed`](../models/edges.md#teed) | 19.6 | 19.2 | 17.5 | 14.1 | 1.4x | 492 | 523 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Promptable segmentation

### Promptable segmentation, synthetic, 1024x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`clipseg`](../models/promptable_segmentation.md#clipseg) | 33.6 | 38.5 | – | 938 |
| [`efficientvit-sam-l0`](../models/promptable_segmentation.md#efficientvit_sam) | 113 | – | – | 1102 |
| [`insid3-vitl16`](../models/promptable_segmentation.md#insid3) | 3211 | – | – | 3003 |
| [`mobile-sam`](../models/promptable_segmentation.md#mobile_sam) | 102 | – | – | 947 |
| [`sam-hq-vit-b`](../models/promptable_segmentation.md#sam_hq) | 716 | 703 | – | 2588 |
| [`sam-vit-b`](../models/promptable_segmentation.md#sam) | 675 | 666 | – | 2571 |
| [`sam2.1-hiera-tiny`](../models/promptable_segmentation.md#sam2) | 184 | – | – | 1174 |
| [`sam3`](../models/promptable_segmentation.md#sam3) | 2062 | 1657 | unsupported | 2768 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Detection

### Detection, synthetic, 800x1344

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`detr-r50-coco`](../models/detection.md#detr) | 128 | 106 | unsupported | – | – | 1924 | – |
| [`dfine-n-coco`](../models/detection.md#dfine) | 25.7 | 22.3 | unsupported | – | – | 624 | – |
| [`grounding-dino-tiny`](../models/detection.md#grounding_dino) | 1043 | – | unsupported | – | – | 1645 | – |
| [`lw-detr-tiny-coco`](../models/detection.md#lwdetr) | 37.7 | 34.9 | unsupported | – | – | 935 | – |
| [`mask-rcnn-r50-fpn-3x-coco`](../models/detection.md#rcnn) | 462 | 411 | unsupported | – | – | 1533 | – |
| [`owlv2-b16`](../models/detection.md#owlv2) | 398 | 295 | unsupported | – | – | 1005 | – |
| [`rf-detr-n-coco`](../models/detection.md#rfdetr) | 22.1 | – | unsupported | – | – | 714 | – |
| [`rtdetrv2-s-coco`](../models/detection.md#rtdetrv2) | 48.1 | 39.5 | unsupported | – | – | 904 | – |
| [`rtmdet-tiny-coco`](../models/detection.md#rtmdet) | 32.4 | 24.4 | unsupported | – | – | 618 | – |
| [`yolox-tiny`](../models/detection.md#yolox) | 15.1 | 13.4 | 14.5 | 12.6 | 1.1x | 454 | 470 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Ocr

### Ocr, synthetic, 32x128

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`parseq`](../models/ocr.md#parseq) | 20.1 | 17.9 | 495 |

### Ocr, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`ppocrv5-mobile`](../models/ocr.md#ppocrv5) | 1251 | 869 | 1610 |
| [`ppocrv5-server`](../models/ocr.md#ppocrv5) | 2691 | 1904 | 2666 |

## Pose

### Pose, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`hrnet-w32`](../models/pose.md#hrnet) | 159 | 117 | unsupported | 911 |
| [`retinaface-resnet50`](../models/pose.md#retinaface) | 263 | 185 | unsupported | 1165 |
| [`rf-detr-keypoint-preview`](../models/pose.md#rfdetr_keypoint) | 92.6 | – | unsupported | 997 |
| [`rtmo-t`](../models/pose.md#rtmo) | 17.9 | – | unsupported | 667 |
| [`rtmpose-t`](../models/pose.md#rtmpose) | 29.6 | 25.6 | unsupported | 665 |
| [`vitpose-base-simple`](../models/pose.md#vitpose) | 202 | 169 | unsupported | 1080 |
| [`yunet-n`](../models/pose.md#yunet) | 30.2 | 22.3 | unsupported | 717 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Features

### Features, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](../models/features.md#c_radio) | 16.9 | 16.4 | – | – | – | 764 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | 190 | 158 | – | – | – | 4960 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | 109 | 93.5 | – | – | – | 2983 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | 55.1 | 47.9 | – | – | – | 1635 | – |
| [`clip-b16-openai`](../models/features.md#clip) | 18.8 | 15.3 | – | – | – | 1478 | – |
| [`clip-l14-laion2b`](../models/features.md#clip) | 69.3 | 70.8 | – | – | – | 3565 | – |
| [`convnext_base.fb_in22k`](../models/features.md#convnext_features) | 20.8 | 18.1 | 20.6 | 20.4 | 0.9x | 1368 | 1366 |
| [`convnext_large.fb_in22k`](../models/features.md#convnext_features) | 35.5 | 33.8 | 36.4 | 29.2 | 1.2x | 2575 | 2574 |
| [`convnext_small.fb_in22k`](../models/features.md#convnext_features) | 14.3 | 12.1 | 14.8 | 12.2 | 1.0x | 931 | 930 |
| [`convnext_tiny.fb_in22k`](../models/features.md#convnext_features) | 8.54 | 7.50 | 8.27 | 7.47 | 1.0x | 760 | 761 |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 20.5 | 18.6 | unsupported | – | – | 607 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 20.6 | 18.0 | unsupported | – | – | 601 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 227 | 193 | unsupported | – | – | 2634 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 234 | 203 | unsupported | – | – | 2630 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 64.3 | 55.5 | unsupported | – | – | 1011 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 75.2 | 58.2 | unsupported | – | – | 1029 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 8.24 | 7.21 | unsupported | – | – | 398 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 8.29 | 7.22 | unsupported | – | – | 398 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 19.6 | 17.1 | 21.4 | 16.0 | 1.1x | 512 | 512 |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 35.4 | 29.9 | 35.2 | 34.1 | 0.9x | 822 | 824 |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 14.6 | 11.9 | 14.0 | 12.1 | 1.0x | 388 | 387 |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 8.01 | 7.23 | 8.51 | 8.19 | 0.9x | 315 | 314 |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 17.1 | 15.0 | unsupported | – | – | 576 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 155 | 120 | unsupported | – | – | 2174 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 53.3 | 45.8 | unsupported | – | – | 1017 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 7.32 | 6.53 | unsupported | – | – | 385 | – |
| [`pe-core-l14-336`](../models/features.md#perception_encoder) | 75.1 | 95.7 | – | – | – | 3746 | – |
| [`pe-core-t16-384`](../models/features.md#perception_encoder) | 4.75 | 5.11 | – | – | – | 456 | – |
| [`siglip2-b16-256`](../models/features.md#siglip2) | 19.5 | 16.6 | – | – | – | 1306 | – |
| [`siglip2-so400m14-378`](../models/features.md#siglip2) | 79.6 | 70.8 | – | – | – | 3850 | – |
| [`tipsv2-b14`](../models/features.md#tipsv2) | 20.6 | 17.8 | unsupported | – | – | 598 | – |
| [`tipsv2-l14`](../models/features.md#tipsv2) | 68.5 | 72.9 | – | – | – | 1665 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Retrieval

### Retrieval, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`boq-dinov2`](../models/retrieval.md#boq) | 52.1 | 55.3 | 910 |
| [`megaloc`](../models/retrieval.md#megaloc) | 55.3 | 87.3 | 1420 |

## Calibration

### Calibration, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`anycalib-pinhole`](../models/calibration.md#anycalib) | 342 | 439 | 2089 |
| [`geocalib-pinhole`](../models/calibration.md#geocalib) | 416 | 352 | 1013 |

## Multiview

### Multiview, synthetic, 256x256

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`resplat-base-re10k`](../models/multiview.md#resplat) | 19441 | 14649 | 5639 |

### Multiview, synthetic, 336x518

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`anysplat`](../models/multiview.md#anysplat) | 8786 | 6974 | 8297 |
| [`depthsplat-small-re10k`](../models/multiview.md#depthsplat) | 1270 | – | 2671 |
| [`mapanything-apache`](../models/multiview.md#mapanything) | 11871 | 8876 | 8371 |
| [`mvsplat-re10k`](../models/multiview.md#mvsplat) | 2575 | 2418 | 2796 |
| [`pi3`](../models/multiview.md#pi3) | 11851 | 10309 | 3859 |
| [`vggt-1b`](../models/multiview.md#vggt) | 9359 | 8774 | 6853 |
| [`vggt-omega-1b-512`](../models/multiview.md#vggt_omega) | 16674 | 8564 | 5137 |

### Multiview, synthetic, 448x672

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`mvsanywhere`](../models/multiview.md#mvsanywhere) | 635 | 506 | 1425 |

### Multiview, synthetic, 480x640

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`lingbot-map`](../models/multiview.md#lingbot_map) | 11010 | 10046 | 8220 |

### Multiview, synthetic, 1536x1024

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`patchmatchnet`](../models/multiview.md#patchmatchnet) | 1612 | 920 | 2247 |

## Flow

### Flow, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`flowformer-small-chairs`](../models/flow.md#flowformer) | failed | failed | unsupported | – |
| [`flowseek-t`](../models/flow.md#flowseek) | 3246 | 2160 | unsupported | 5781 |
| [`gmflow-scale1-things`](../models/flow.md#gmflow) | 7587 | 6822 | unsupported | 10465 |
| [`memflow`](../models/flow.md#memflow) | 13101 | – | unsupported | 10550 |
| [`memfof`](../models/flow.md#memfof) | 5436 | – | unsupported | 2716 |
| [`neuflow-v2`](../models/flow.md#neuflow_v2) | 1087 | – | unsupported | 9252 |
| [`raft-large`](../models/flow.md#raft) | 2724 | 2134 | unsupported | 8987 |
| [`recover-cx`](../models/flow.md#recover) | 1755 | – | unsupported | 1933 |
| [`sea-raft-m`](../models/flow.md#sea_raft) | 2187 | 1427 | unsupported | 5714 |
| [`waft-a1`](../models/flow.md#waft) | 10168 | 9339 | unsupported | 3559 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matching

### Matching, synthetic, 768x1024

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|
| [`aliked-flexglue`](../models/matching.md#flexglue) | 1224 | 1100 | – | – | 1387 | – |
| [`aliked-lightglue`](../models/matching.md#lightglue) | 461 | – | – | – | 2416 | – |
| [`dedode-v2-b`](../models/matching.md#dedode) | 2534 | – | unsupported | – | 5327 | – |
| [`dedode-v2-g`](../models/matching.md#dedode) | 5544 | – | – | – | 6499 | – |
| [`disk-lightglue`](../models/matching.md#lightglue) | 1178 | – | – | – | 2333 | – |
| [`loftr-outdoor`](../models/matching.md#loftr) | 2017 | – | – | – | 3685 | – |
| [`loma-b`](../models/matching.md#loma) | 3860 | – | – | – | 6150 | – |
| [`romav1`](../models/matching.md#romav1) | 4440 | – | – | – | 4499 | – |
| [`romav2`](../models/matching.md#romav2) | 6426 | – | 4043 | 1.6x | 4489 | 3150 |
| [`xfeat`](../models/matching.md#xfeat) | 164 | – | – | – | 538 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Tracking

### Tracking, synthetic, 480x854

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`alltracker`](../models/tracking.md#alltracker) | 5397 | – | 2597 |
| [`cotracker3-offline`](../models/tracking.md#cotracker) | 3577 | – | 3438 |
| [`cotracker3-online`](../models/tracking.md#cotracker) | 3340 | – | 3971 |
| [`locotrack-b`](../models/tracking.md#locotrack) | 2039 | – | 1679 |
| [`tapir`](../models/tracking.md#tapir) | 1592 | – | 1717 |
| [`tapnext`](../models/tracking.md#tapnext) | 7477 | – | 3673 |
| [`track-on2`](../models/tracking.md#track_on) | 10755 | 7623 | 3056 |

## Interpolation

### Interpolation, synthetic, 1080x1920

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`ema-vfi-small-t`](../models/interpolation.md#ema_vfi) | 1019 | 876 | – | – | – | 2460 | – |
| [`film`](../models/interpolation.md#film) | 9227 | 6590 | – | – | – | 9495 | – |
| [`gimm-vfi-r`](../models/interpolation.md#gimm_vfi) | 58742 | 30451 | – | – | – | 12185 | – |
| [`rife-4.25`](../models/interpolation.md#rife) | 256 | 211 | 223 | 180 | 1.2x | 1680 | 1655 |
| [`rife-4.25-lite`](../models/interpolation.md#rife) | 200 | 202 | – | – | – | 1590 | – |

## Restoration

### Restoration, synthetic, 512x512

| model | time (ms): portable fp32 | portable fp16 | portable bf16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|---|
| [`big-lama`](../models/restoration.md#lama) | 343 | 321 | – | unsupported | – | – | 1094 | – |
| [`codeformer`](../models/restoration.md#codeformer) | 484 | 285 | – | 481 | 239 | 1.2x | 1597 | 1637 |
| [`dat-2-x4`](../models/restoration.md#dat) | 20494 | 15757 | – | – | – | – | 3512 | – |
| [`dat-light-x4`](../models/restoration.md#dat) | 5319 | 4092 | – | – | – | – | 2782 | – |
| [`dat-x4`](../models/restoration.md#dat) | 24476 | 16650 | – | – | – | – | 3779 | – |
| [`drct-l-x4`](../models/restoration.md#drct) | 29775 | 18644 | – | – | – | – | 5279 | – |
| [`drct-x4`](../models/restoration.md#drct) | 15300 | 10754 | – | – | – | – | 5003 | – |
| [`gfpgan-v1.4`](../models/restoration.md#gfpgan) | 318 | – | – | – | – | – | 1673 | – |
| [`hat-s-x4`](../models/restoration.md#hat) | 20203 | 12676 | – | – | – | – | 5361 | – |
| [`hat-x4`](../models/restoration.md#hat) | 24576 | 15497 | – | – | – | – | 5592 | – |
| [`nafnet-gopro`](../models/restoration.md#nafnet) | 649 | – | – | 694 | – | 0.9x | 1745 | 1746 |
| [`real-hat-gan-x4`](../models/restoration.md#hat) | 24682 | 15831 | – | – | – | – | 5687 | – |
| [`realesr-general-x4v3`](../models/restoration.md#compact) | 333 | 246 | – | 177 | 104 | 2.4x | 737 | 1061 |
| [`realesrgan-x2plus`](../models/restoration.md#realesrgan) | 1599 | 1199 | – | 1667 | 567 | 2.1x | 974 | 1222 |
| [`realesrgan-x4plus`](../models/restoration.md#realesrgan) | 9349 | 5321 | – | 7380 | 2184 | 2.4x | 1390 | 2452 |
| [`realesrgan-x4plus-anime-6b`](../models/restoration.md#realesrgan) | 2700 | 1617 | – | 2038 | 849 | 1.9x | 1319 | 2106 |
| [`realplksr-nomoswebphoto-x4`](../models/restoration.md#realplksr) | 2800 | 1991 | – | 3408 | 1910 (golden fail) | 0.6x | 953 | 1736 |
| [`restormer-deblur`](../models/restoration.md#restormer) | 2926 | 2532 | – | – | – | – | 2666 | – |
| [`restormer-derain`](../models/restoration.md#restormer) | 3040 | 2815 | – | – | – | – | 2668 | – |
| [`restormer-sidd`](../models/restoration.md#restormer) | 2793 | 2427 | – | – | – | – | 2672 | – |
| [`seedvr2-3b`](../models/restoration.md#seedvr2) | – | 14023 | 14307 | – | – | – | 5178 | – |
| [`seedvr2-7b`](../models/restoration.md#seedvr2) | – | 32426 | 25365 | – | – | – | 6547 | – |
| [`span-x2`](../models/restoration.md#span) | 180 | 134 | – | 171 | 76.5 | 1.7x | 753 | 1073 |
| [`span-x4`](../models/restoration.md#span) | 190 | 149 | – | 179 | 86.0 | 1.7x | 827 | 1210 |
| [`swin2sr-x2`](../models/restoration.md#swin2sr) | 14183 | 7619 | – | – | – | – | 2875 | – |
| [`swin2sr-x4`](../models/restoration.md#swin2sr) | 13955 | 7994 | – | – | – | – | 3266 | – |
| [`swin2sr-x4-real`](../models/restoration.md#swin2sr) | 14296 | 8555 | – | – | – | – | 2851 | – |
| [`swinir-l-x4-real`](../models/restoration.md#swinir) | 26151 | 14530 | – | – | – | – | 3161 | – |
| [`swinir-m-x2`](../models/restoration.md#swinir) | 13788 | 7248 | – | – | – | – | 2443 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Volume

### Volume, synthetic, 192x192

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`nnfoundation-vit`](../models/volume.md#nnfoundation) | 18038 | 13037 | 2469 |

### Volume, synthetic, 200x200

| model | time (ms): portable fp32 | portable fp16 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|
| [`nnfoundation-cnn`](../models/volume.md#nnfoundation) | 4271 | 3520 | 8041 |
| [`nninteractive`](../models/volume.md#nninteractive) | 4005 | 3797 | 5587 |
| [`totalsegmentator-total`](../models/volume.md#totalsegmentator) | 43030 | 31549 | 5163 |
| [`totalsegmentator-total-3mm`](../models/volume.md#totalsegmentator) | 1208 | 1111 | 4730 |
<!-- device:end -->
