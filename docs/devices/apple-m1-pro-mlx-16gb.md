# Apple M1 Pro 16 GB (MLX)

<!-- device:begin (tools/gen_docs.py, from docs/data/devices.toml and docs/data/speed/apple-m1-pro-mlx-16gb.jsonl) -->

| | |
|---|---|
| device | Apple M1 Pro 16 GB (MLX) |
| backend | mlx |
| memory | 16 GB unified |
| hardware | 16-core GPU; the families with an MLX port (device="mlx") |
| software | macOS 27, mlx 0.32.4, torch 2.14 |
| torch (rows) | 2.14.1 |
| measured | 2026-11-12 to 2026-11-18 |
| rows | 264 in `docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` |

Speed survey: the models and dtypes measured here, as recorded. Cells are median ms; every column names its dtype. `unsupported` = the card does not offer that mode or dtype on this backend, `failed` = it ran and raised, `OOM` = out of memory, – = not tested; `(golden fail)` = it ran but its result failed the golden check on this device (never counted as fastest). Speed-up = kernels against the fastest original and the fastest `portable` row of the same model (any dtype); memory = the peak of that mode's fastest row, measured the way the column says (not comparable with another device's). One input per model, the task's standard one. Each model links to its model page; [benchmarks](../benchmarks.md) has the method and the reference device.

Protocol: first call, 4 warm-up, median of 5 synchronised calls (tools/speed_survey.py).

## Stereo

### Stereo, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`crestereo-eth3d`](../models/stereo.md#crestereo) | 5763 | 4308 | 4439 | 3058 | 1.4x | 2543 | 2812 |

## Depth

### Depth, synthetic, 994x1482

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`depth-anything-v2-b`](../models/depth.md#depth_anything_v2) | 279 | 200 | 256 | 192 | 1.0x | 2004 | 1931 |
| [`depth-anything-v2-l`](../models/depth.md#depth_anything_v2) | 1098 | 582 | 780 | 548 | 1.1x | 2859 | 2788 |
| [`depth-anything-v2-metric-hypersim-b`](../models/depth.md#depth_anything_v2) | 270 | 191 | 242 | 189 | 1.0x | 2003 | 1929 |
| [`depth-anything-v2-metric-hypersim-l`](../models/depth.md#depth_anything_v2) | 805 | 596 | 756 | 534 | 1.1x | 2811 | 2785 |
| [`depth-anything-v2-metric-hypersim-s`](../models/depth.md#depth_anything_v2) | 102 | 79.9 | 95.2 | 73.2 | 1.1x | 1072 | 963 |
| [`depth-anything-v2-metric-vkitti-b`](../models/depth.md#depth_anything_v2) | 258 | 194 | 245 | 185 | 1.0x | 2007 | 1930 |
| [`depth-anything-v2-metric-vkitti-l`](../models/depth.md#depth_anything_v2) | 806 | 562 | 720 | 535 | 1.1x | 2813 | 2787 |
| [`depth-anything-v2-metric-vkitti-s`](../models/depth.md#depth_anything_v2) | 102 | 80.3 | 92.6 | 75.0 | 1.1x | 1070 | 915 |
| [`depth-anything-v2-s`](../models/depth.md#depth_anything_v2) | 99.1 | 82.4 | 96.8 | 79.1 | 1.0x | 1065 | 911 |
| [`distill-any-depth-b`](../models/depth.md#depth_anything_v2) | 284 | 211 | 261 | 203 | 1.0x | 2005 | 1931 |
| [`distill-any-depth-l`](../models/depth.md#depth_anything_v2) | 791 | 595 | 758 | 568 | 1.0x | 2862 | 2789 |
| [`distill-any-depth-s`](../models/depth.md#depth_anything_v2) | 105 | 83.3 | 97.5 | 76.7 | 1.1x | 1065 | 913 |

## Classification

### Classification, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`convnext_base.fb_in1k`](../models/classification.md#convnext) | 18.6 | 16.1 | 18.9 | 16.2 | 1.0x | 1201 | 1202 |
| [`convnext_base.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 18.7 | 16.2 | 18.6 | 16.4 | 1.0x | 1203 | 1204 |
| [`convnext_base.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 42.2 | 33.7 | 40.7 | 33.9 | 1.0x | 1203 | 1201 |
| [`convnext_large.fb_in1k`](../models/classification.md#convnext) | 35.2 | 29.2 | 34.8 | 29.3 | 1.0x | 2335 | 2334 |
| [`convnext_large.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 35.4 | 29.6 | 35.6 | 29.1 | 1.0x | 2334 | 2334 |
| [`convnext_large.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 81.5 | 63.8 | 82.4 | 64.0 | 1.0x | 2336 | 2334 |
| [`convnext_small.fb_in1k`](../models/classification.md#convnext) | 14.1 | 12.3 | 14.2 | 12.5 | 1.0x | 807 | 807 |
| [`convnext_small.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 14.1 | 12.2 | 14.2 | 12.5 | 1.0x | 808 | 807 |
| [`convnext_small.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 28.6 | 25.5 | 28.6 | 25.0 | 1.0x | 807 | 809 |
| [`convnext_tiny.fb_in1k`](../models/classification.md#convnext) | 8.40 | 7.61 | 8.42 | 8.01 | 1.0x | 637 | 636 |
| [`convnext_tiny.fb_in22k_ft_in1k`](../models/classification.md#convnext) | 8.20 | 7.57 | 8.89 | 7.58 | 1.0x | 639 | 637 |
| [`convnext_tiny.fb_in22k_ft_in1k_384`](../models/classification.md#convnext) | 17.5 | 15.4 | 17.5 | 15.2 | 1.0x | 700 | 702 |
| [`convnextv2_base.fcmae_ft_in22k_in1k`](../models/classification.md#convnextv2) | 22.7 | 21.6 | 23.1 | 20.0 | 1.1x | 1184 | 1183 |
| [`dinov2_vitl14_lc`](../models/classification.md#dinov2_linear_cls) | 78.3 | 63.0 | unsupported | – | – | 4218 | – |
| [`efficientnet_b0.tv_in1k`](../models/classification.md#efficientnet) | 4.79 | 5.32 | 4.88 | 4.49 | 1.1x | 346 | 336 |
| [`efficientnet_b4.tv_in1k`](../models/classification.md#efficientnet) | 19.0 | 17.9 | 17.6 | 16.0 | 1.1x | 639 | 639 |
| [`efficientnet_v2_s.tv_in1k`](../models/classification.md#efficientnetv2) | 19.1 | 17.2 | 18.5 | 16.4 | 1.1x | 541 | 541 |
| [`mobilenet_v2.tv2_in1k`](../models/classification.md#mobilenetv2) | 3.73 | 3.14 | 3.24 | 2.72 | 1.2x | 306 | 307 |
| [`mobilenet_v3_large.tv2_in1k`](../models/classification.md#mobilenetv3) | 3.56 | 3.74 | 3.47 | 3.16 | 1.1x | 334 | 325 |
| [`mobilenetv1_100.ra4_e3600_r224_in1k`](../models/classification.md#mobilenetv1) | 2.50 | 2.38 | – | – | – | 353 | – |
| [`regnety_160.tv2_in1k`](../models/classification.md#regnet) | 22.6 | 22.5 | 19.9 | 18.3 | 1.2x | 1038 | 1038 |
| [`repvit_m0_9.dist_300e_in1k`](../models/classification.md#repvit) | 6.43 | 5.62 | – | – | – | 370 | – |
| [`repvit_m0_9.dist_450e_in1k`](../models/classification.md#repvit) | 4.92 | 4.45 | – | – | – | 368 | – |
| [`repvit_m1_0.dist_300e_in1k`](../models/classification.md#repvit) | 5.72 | 5.41 | – | – | – | 397 | – |
| [`repvit_m1_0.dist_450e_in1k`](../models/classification.md#repvit) | 6.06 | 5.47 | – | – | – | 400 | – |
| [`repvit_m1_1.dist_300e_in1k`](../models/classification.md#repvit) | 5.25 | 4.57 | – | – | – | 411 | – |
| [`repvit_m1_1.dist_450e_in1k`](../models/classification.md#repvit) | 5.07 | 4.75 | – | – | – | 416 | – |
| [`repvit_m1_5.dist_300e_in1k`](../models/classification.md#repvit) | 8.03 | 7.08 | – | – | – | 488 | – |
| [`repvit_m1_5.dist_450e_in1k`](../models/classification.md#repvit) | 7.92 | 8.31 | – | – | – | 536 | – |
| [`repvit_m2_3.dist_300e_in1k`](../models/classification.md#repvit) | 12.2 | 10.7 | – | – | – | 609 | – |
| [`repvit_m2_3.dist_450e_in1k`](../models/classification.md#repvit) | 12.1 | 11.0 | – | – | – | 622 | – |
| [`resnet152.tv2_in1k`](../models/classification.md#resnet) | 19.9 | 17.0 | 20.2 | 16.8 | 1.0x | 724 | 722 |
| [`resnet18.tv_in1k`](../models/classification.md#resnet) | 4.31 | 3.74 | 4.25 | 4.74 | 0.9x | 373 | 388 |
| [`resnet50.tv2_in1k`](../models/classification.md#resnet) | 7.84 | 6.68 | 7.86 | 7.12 | 0.9x | 495 | 492 |
| [`starnet_s1.in1k`](../models/classification.md#starnet) | 4.40 | 3.28 | – | – | – | 328 | – |
| [`starnet_s2.in1k`](../models/classification.md#starnet) | 6.37 | 5.64 | – | – | – | 337 | – |
| [`starnet_s3.in1k`](../models/classification.md#starnet) | 5.01 | 5.38 | – | – | – | 381 | – |
| [`starnet_s4.in1k`](../models/classification.md#starnet) | 6.46 | 5.88 | – | – | – | 385 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Segmentation

### Segmentation, synthetic, 1024x2048

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | peak memory (MiB, process footprint, fastest dtype): portable |
|---|---|---|---|---|
| [`dinov2-b-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 929 | 965 | unsupported | 3844 |
| [`dinov2-l-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 1589 | 1348 | unsupported | 3270 |
| [`dinov2-s-linear-ade20k`](../models/segmentation.md#dinov2_linear) | 708 | 713 | unsupported | 2751 |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Features

### Features, synthetic, 224x224

| model | time (ms): portable fp32 | portable fp16 | kernels fp32 | kernels fp16 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|---|---|
| [`c-radio-v3-b`](../models/features.md#c_radio) | 18.7 | 16.4 | – | – | – | 934 | – |
| [`c-radio-v3-g`](../models/features.md#c_radio) | 210 | 159 | – | – | – | 5152 | – |
| [`c-radio-v3-h`](../models/features.md#c_radio) | 119 | 96.4 | – | – | – | 3178 | – |
| [`c-radio-v3-l`](../models/features.md#c_radio) | 58.2 | 47.4 | – | – | – | 1879 | – |
| [`convnext_base.fb_in22k`](../models/features.md#convnext_features) | 18.5 | 16.0 | 18.4 | 16.1 | 1.0x | 1360 | 1358 |
| [`convnext_large.fb_in22k`](../models/features.md#convnext_features) | 34.6 | 29.1 | 35.5 | 29.1 | 1.0x | 2569 | 2570 |
| [`convnext_small.fb_in22k`](../models/features.md#convnext_features) | 14.5 | 12.0 | 14.2 | 12.1 | 1.0x | 925 | 925 |
| [`convnext_tiny.fb_in22k`](../models/features.md#convnext_features) | 9.30 | 7.43 | 8.38 | 7.53 | 1.0x | 754 | 754 |
| [`dinov2-vitb14`](../models/features.md#dinov2) | 24.6 | 18.5 | unsupported | – | – | 807 | – |
| [`dinov2-vitb14-reg`](../models/features.md#dinov2) | 24.5 | 18.9 | unsupported | – | – | 808 | – |
| [`dinov2-vitg14`](../models/features.md#dinov2) | 258 | 193 | unsupported | – | – | 3223 | – |
| [`dinov2-vitg14-reg`](../models/features.md#dinov2) | 265 | 206 | unsupported | – | – | 3204 | – |
| [`dinov2-vitl14`](../models/features.md#dinov2) | 74.2 | 59.7 | unsupported | – | – | 1575 | – |
| [`dinov2-vitl14-reg`](../models/features.md#dinov2) | 86.4 | 58.7 | unsupported | – | – | 1595 | – |
| [`dinov2-vits14`](../models/features.md#dinov2) | 10.0 | 8.60 | unsupported | – | – | 382 | – |
| [`dinov2-vits14-reg`](../models/features.md#dinov2) | 10.3 | 8.14 | unsupported | – | – | 382 | – |
| [`dinov3-convnext-base`](../models/features.md#dinov3_convnext) | 18.4 | 16.2 | 18.6 | 16.5 | 1.0x | 501 | 503 |
| [`dinov3-convnext-large`](../models/features.md#dinov3_convnext) | 35.0 | 29.1 | 35.0 | 29.0 | 1.0x | 841 | 839 |
| [`dinov3-convnext-small`](../models/features.md#dinov3_convnext) | 13.9 | 12.0 | 13.7 | 11.9 | 1.0x | 377 | 378 |
| [`dinov3-convnext-tiny`](../models/features.md#dinov3_convnext) | 9.29 | 8.02 | 9.01 | 8.00 | 1.0x | 306 | 305 |
| [`dinov3-vitb16`](../models/features.md#dinov3) | 19.0 | 16.1 | unsupported | – | – | 784 | – |
| [`dinov3-vith16plus`](../models/features.md#dinov3) | 162 | 133 | unsupported | – | – | 2605 | – |
| [`dinov3-vitl16`](../models/features.md#dinov3) | 67.4 | 54.8 | unsupported | – | – | 1597 | – |
| [`dinov3-vits16`](../models/features.md#dinov3) | 8.32 | 8.69 | unsupported | – | – | 346 | – |
| [`tipsv2-b14`](../models/features.md#tipsv2) | 22.0 | 19.4 | – | – | – | 797 | – |
| [`tipsv2-l14`](../models/features.md#tipsv2) | 74.9 | 56.9 | – | – | – | 1542 | – |

Note: the MLX port has no kernels mode (card mlx_dtypes).

## Matching

### Matching, synthetic, 768x1024

| model | time (ms): portable fp32 | kernels fp32 | kernels speed-up: vs portable | peak memory (MiB, process footprint, fastest dtype): portable | kernels |
|---|---|---|---|---|---|
| [`romav2`](../models/matching.md#romav2) | 6235 | 3931 | 1.6x | 4896 | 4178 |
<!-- device:end -->
