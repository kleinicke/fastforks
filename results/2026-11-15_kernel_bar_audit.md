# Kernel bar audit, 2026-11-15

Bar (user, 2026-11-15): every network, in BOTH modes, faster than the original, lower peak memory than the original, same task accuracy; kernels additionally faster than portable. CPU only: `docs/data/speed.jsonl` at `6ec60fba` (origin/work/ports) + the gap matrix `results/raw/2026-11-15_gaps/gaps.csv` (452 refs; the card registry has 573 refs, 19 of them have no speed row yet: any2full-s/b/l, depthfm-v1, lingbot-depth(-dc), marigold-v2-depth, omni-dc, video-depth-anything-metric-s, memfof-kitti/sintel/spring, aliked-flexglue, loma-b, lingbot-map, resplat-small/base, seedvr2-7b, tapnext-pp; they are in the audit as "no row"). Script: `results/raw/2026-11-15_kernel_audit/audit.py` (rows: `audit_rows.csv` one per ref x size, `audit_families.csv`), ranking: `judge.py` -> `no_kernels_ranking.csv`.

Method: per ref and tested size, the latest row per mode. "Original" = the STRICTEST original row at that size (fastest ms over the upstream / upstream-default fp32 / fp16 rows; lowest peak_alloc over the same rows). Peak memory = `peak_alloc_mb`. Tolerance: portable "slower" if original < 0.99 x portable; memory "above" if ratio > 1.01 (noise). Accuracy is not part of this table (see the accuracy results per family).

## Counts

554 refs have rows (447 of the 452-gap-matrix refs); 750 ref x size rows.

| failure | refs | of them in the 452 |
|---|---|---|
| kernels_mem_above_orig | 30 | 16 |
| kernels_mem_above_portable | 150 | 130 |
| kernels_slower_than_orig | 2 | 1 |
| kernels_slower_than_portable | 5 | 3 |
| no_kernels_row | 84 | 64 |
| portable_mem_above | 38 | 22 |
| portable_slower | 23 | 21 |

`no_kernels_row` = 84 refs in 29 families (31 families carry `status = "none"` in their card; the user counted 29, the difference is the 2 experimental cards flexglue / loma that the 452 matrix omits). Kernels-mem-above-portable is mostly a small fixed overhead of the kernel workspace (+1..5 %), see the table.

## (a) Refs failing the bar

### Portable slower than the original (23 refs)
| ref | input | orig ms | portable ms | speedup | orig MB | portable MB | mem ratio |
|---|---|---|---|---|---|---|---|
| efficientvit_b2.r288_in1k | 224x224 | 10.67 | 11.9 | 0.897 | 144 | 85 | 0.59 |
| fastvit_t8.apple_dist_in1k | 256x256 | 3.07 | 3.302 | 0.93 | 56 | 45 | 0.804 |
| squeezenet1_1.tv_in1k | 224x224 | 1.72 | 1.82 | 0.945 | 10 | 41 | 4.1 |
| moge-3-vitl | 994x1482 | 485.6 | 513.676 | 0.945 | 2505 | 1686 | 0.673 |
| moge-3-vitl | 518x770 | 324.13 | 354.144 | 0.915 | 2493 | 1669 | 0.669 |
| dinov3-convnext-tiny | 224x224 | 2.96 | 3.039 | 0.974 | 151 | 93 | 0.616 |
| flowformer-small-chairs | 375x1242 | 302.65 | 351.288 | 0.862 | 2382 | 3165 | 1.329 |
| flowformer-small-things | 375x1242 | 303.13 | 350.814 | 0.864 | 2382 | 3165 | 1.329 |
| flowformer-small-sintel | 375x1242 | 302.97 | 350.805 | 0.864 | 2382 | 3165 | 1.329 |
| gmflow-scale1-things | 1080x1920 | 750.27 | 800.828 | 0.937 | 8360 | 5523 | 0.661 |
| gmflow-scale1-things | 375x1242 | 53.27 | 56.177 | 0.948 | 536 | 572 | 1.067 |
| gmflow-scale1-mixdata | 1080x1920 | 750.15 | 800.379 | 0.937 | 8360 | 5523 | 0.661 |
| gmflow-scale1-mixdata | 375x1242 | 53.46 | 56.282 | 0.95 | 536 | 572 | 1.067 |
| gmflow-scale2-things | 1080x1920 | 1598.6 | 1636.243 | 0.977 | 8425 | 6433 | 0.764 |
| gmflow-scale2-sintel | 1080x1920 | 1606.71 | 1644.25 | 0.977 | 8425 | 6433 | 0.764 |
| gmflow-scale2-mixdata | 1080x1920 | 1607.69 | 1644.946 | 0.977 | 8425 | 6433 | 0.764 |
| neuflow-v2 | 375x1242 | 12.15 | 14.403 | 0.844 | 310 | 340 | 1.097 |
| neuflow-v2-sintel | 375x1242 | 12.21 | 14.552 | 0.839 | 310 | 340 | 1.097 |
| neuflow-v2-things | 375x1242 | 12.18 | 14.425 | 0.844 | 318 | 340 | 1.069 |
| film | 1080x1920 | 463.81 | 527.602 | 0.879 | 2953 | 4747 | 1.608 |
| film | 480x854 | 103.88 | 112.882 | 0.92 | 711 | 1093 | 1.537 |
| rife-4.25 | 1080x1920 | 24.48 | 24.976 | 0.98 | 987 | 755 | 0.765 |
| rife-4.26 | 1080x1920 | 24.21 | 25.141 | 0.963 | 1004 | 755 | 0.752 |
| restormer-sidd | 512x512 | 217.95 | 227.062 | 0.96 | 1221 | 850 | 0.696 |
| pspnet-r101-d8-ade20k | 1024x2048 | 47.17 | 48.774 | 0.967 | 1797 | 417 | 0.232 |
| s2m2-l | 994x1482 | 1004.73 | 1015.34 | 0.99 | 3248 | 3245 | 0.999 |
| s2m2-l | 375x1242 | 282.67 | 288.48 | 0.98 | 1747 | 1510 | 0.864 |
| s2m2-xl | 994x1482 | 1796.1 | 1814.68 | 0.99 | 5606 | 5228 | 0.933 |
| s2m2-xl | 375x1242 | 529.1 | 535.96 | 0.987 | 3256 | 2722 | 0.836 |

### Portable above the original's peak memory (38 refs; speed column shows whether it is also slower)
| ref | input | orig ms | portable ms | speedup | orig MB | portable MB | mem ratio |
|---|---|---|---|---|---|---|---|
| efficientformer_l1.snap_dist_in1k | 224x224 | 4.87 | 3.724 | 1.308 | 87 | 93 | 1.069 |
| squeezenet1_1.tv_in1k | 224x224 | 1.72 | 1.82 | 0.945 | 10 | 41 | 4.1 |
| wd-vit-tagger-v3 | 448x448 | 5.78 | 3.985 | 1.45 | 231 | 235 | 1.017 |
| wd-convnext-tagger-v3 | 448x448 | 9.6 | 5.622 | 1.708 | 259 | 340 | 1.313 |
| wd-vit-large-tagger-v3 | 448x448 | 19.87 | 12.876 | 1.543 | 662 | 669 | 1.011 |
| yolox-tiny | 800x1344 | 7.75 | 5.583 | 1.388 | 32 | 60 | 1.875 |
| yolox-s | 800x1344 | 8.56 | 5.546 | 1.543 | 67 | 79 | 1.179 |
| flowformer-small-chairs | 375x1242 | 302.65 | 351.288 | 0.862 | 2382 | 3165 | 1.329 |
| flowformer-small-things | 375x1242 | 303.13 | 350.814 | 0.864 | 2382 | 3165 | 1.329 |
| flowformer-small-sintel | 375x1242 | 302.97 | 350.805 | 0.864 | 2382 | 3165 | 1.329 |
| gmflow-scale1-things | 375x1242 | 53.27 | 56.177 | 0.948 | 536 | 572 | 1.067 |
| gmflow-scale1-mixdata | 375x1242 | 53.46 | 56.282 | 0.95 | 536 | 572 | 1.067 |
| gmflow-scale2-things | 375x1242 | 145.21 | 145.649 | 0.997 | 1419 | 1466 | 1.033 |
| gmflow-scale2-sintel | 375x1242 | 145.43 | 145.328 | 1.001 | 1419 | 1466 | 1.033 |
| gmflow-scale2-mixdata | 375x1242 | 145.43 | 145.869 | 0.997 | 1419 | 1466 | 1.033 |
| neuflow-v2 | 375x1242 | 12.15 | 14.403 | 0.844 | 310 | 340 | 1.097 |
| neuflow-v2-sintel | 375x1242 | 12.21 | 14.552 | 0.839 | 310 | 340 | 1.097 |
| neuflow-v2-things | 375x1242 | 12.18 | 14.425 | 0.844 | 318 | 340 | 1.069 |
| sea-raft-m-spring | 375x1242 | 20.13 | 15.562 | 1.294 | 204 | 216 | 1.059 |
| sea-raft-s-spring | 375x1242 | 14.44 | 12.374 | 1.167 | 163 | 175 | 1.074 |
| film | 1080x1920 | 463.81 | 527.602 | 0.879 | 2953 | 4747 | 1.608 |
| film | 480x854 | 103.88 | 112.882 | 0.92 | 711 | 1093 | 1.537 |
| patchmatchnet | 1536x1024 | 219.74 | 201.66 | 1.09 | 1228 | 1806 | 1.471 |
| retinaface-resnet50 | 1080x1920 | 79.42 | 24.255 | 3.274 | 371 | 405 | 1.092 |
| rtmo-t | 1080x1920 | 11.3 | 7.621 | 1.483 | 90 | 121 | 1.344 |
| rtmpose-t-halpe26 | 1080x1920 | 171.73 | 4.479 | 38.341 | 103 | 120 | 1.165 |
| dat-x4 | 512x512 | 5633.19 | 1565.007 | 3.599 | 2443 | 2552 | 1.045 |
| dat-2-x4 | 512x512 | 4880.34 | 1337.965 | 3.648 | 2436 | 2546 | 1.045 |
| seedvr2-3b | 512x512 | 13150.0 | 2281.497 | 5.764 | 7598 | 11957 | 1.574 |
| swin2sr-x4-real | 512x512 | 2527.19 | 958.296 | 2.637 | 2321 | 2354 | 1.014 |
| swinir-m-x4-real | 512x512 | 2461.6 | 1241.651 | 1.983 | 2321 | 2387 | 1.028 |
| swinir-l-x4-real | 512x512 | 3709.3 | 2098.712 | 1.767 | 2357 | 2425 | 1.029 |
| bridgedepth-sf | 994x1482 | 459.9 | 436.86 | 1.053 | 2852 | 2916 | 1.022 |
| bridgedepth-rvc-pretrain | 994x1482 | 470.1 | 441.62 | 1.064 | 2903 | 2968 | 1.022 |
| c-foundationstereo-s | 375x1242 | 1120.69 | 1073.76 | 1.044 | 2781 | 3085 | 1.109 |
| las2-h | 375x1242 | 36.4 | 35.639 | 1.021 | 352 | 398 | 1.131 |
| locotrack-b | 480x854 | 203.35 | 190.125 | 1.07 | 719 | 771 | 1.072 |
| locotrack-s | 480x854 | 192.77 | 179.541 | 1.074 | 706 | 759 | 1.075 |
| track-on2 | 480x854 | 1952.78 | 1238.277 | 1.577 | 1236 | 1423 | 1.151 |

### Kernels slower than portable (5) and slower than the original (2)
| ref | input | portable ms | kernels ms | k vs port | k vs orig | orig MB | portable MB | kernels MB |
|---|---|---|---|---|---|---|---|---|
| geocalib-pinhole | 1080x1920 | 82.326 | 82.467 | 0.998 | 1.288 | 502 | 320 | 279 |
| wd-vit-tagger-v3 | 448x448 | 3.985 | 4.054 | 0.983 | 1.426 | 231 | 235 | 234 |
| depthsplat-small-re10k | 336x518 | 118.871 | 119.739 | 0.993 | 1.056 | 5539 | 5432 | 5479 |
| rtmw-x-384 | 1080x1920 | 50.632 | 50.766 | 0.997 | 5.725 | 1705 | 482 | 496 |
| dat-light-x4 | 512x512 | 267.844 | 268.219 | 0.999 | 8.215 | 1606 | 1535 | 1535 |

| ref | input | portable ms | kernels ms | k vs port | k vs orig | orig MB | portable MB | kernels MB |
|---|---|---|---|---|---|---|---|---|
| neuflow-v2-things | 375x1242 | 14.425 | 12.357 | 1.167 | 0.986 | 318 | 340 | 205 |
| restormer-sidd | 512x512 | 227.062 | 224.323 | 1.012 | 0.972 | 1221 | 850 | 850 |

### Kernels above the original's peak memory (30 refs)
| ref | input | portable ms | kernels ms | k vs port | k vs orig | orig MB | portable MB | kernels MB |
|---|---|---|---|---|---|---|---|---|
| densenet121.tv_in1k | 224x224 | 9.41 | 1.279 | 7.357 | 8.389 | 78 | 53 | 83 |
| efficientnet_b0.tv_in1k | 224x224 | 4.98 | 0.769 | 6.476 | 7.334 | 64 | 48 | 78 |
| mobilenetv1_100.ra4_e3600_r224_in1k | 224x224 | 1.77 | 0.37 | 4.784 | 5.919 | 58 | 44 | 76 |
| mobilenet_v2.tv2_in1k | 224x224 | 2.95 | 0.439 | 6.72 | 8.2 | 54 | 43 | 75 |
| mobilenet_v3_large.tv2_in1k | 224x224 | 3.8 | 0.545 | 6.972 | 7.945 | 61 | 46 | 78 |
| mobilenetv4_conv_medium.e500_r256_in1k | 224x224 | 3.91 | 0.641 | 6.1 | 9.017 | 81 | 54 | 87 |
| shufflenet_v2_x1_0.tv_in1k | 224x224 | 3.71 | 0.787 | 4.714 | 6.086 | 45 | 41 | 87 |
| wd-vit-tagger-v3 | 448x448 | 3.985 | 4.054 | 0.983 | 1.426 | 231 | 235 | 234 |
| wd-convnext-tagger-v3 | 448x448 | 5.622 | 4.451 | 1.263 | 2.157 | 259 | 340 | 318 |
| dfine-n-coco | 800x1344 | 10.344 | 2.515 | 4.113 | 9.65 | 91 | 88 | 93 |
| deim-n-coco | 800x1344 | 10.172 | 2.498 | 4.072 | 8.022 | 90 | 88 | 93 |
| rf-detr-seg-xl-coco | 800x1344 | 17.908 | 10.521 | 1.702 | 1.781 | 182 | 177 | 188 |
| yolox-tiny | 800x1344 | 5.583 | 2.018 | 2.767 | 3.84 | 32 | 60 | 62 |
| yolox-s | 800x1344 | 5.546 | 2.642 | 2.099 | 3.24 | 67 | 79 | 84 |
| pe-core-t16-384 | 384x384 | 4.895 | 0.906 | 5.403 | 7.351 | 63 | 53 | 90 |
| pe-core-s16-384 | 384x384 | 4.86 | 1.318 | 3.687 | 5.137 | 136 | 91 | 140 |
| pe-spatial-s16-512 | 512x512 | 4.517 | 2.378 | 1.899 | 3.654 | 138 | 98 | 145 |
| film | 1080x1920 | 527.602 | 444.334 | 1.187 | 1.044 | 2953 | 4747 | 4763 |
| film | 480x854 | 112.882 | 94.573 | 1.194 | 1.098 | 711 | 1093 | 1093 |
| retinaface-resnet50 | 1080x1920 | 24.255 | 11.7 | 2.073 | 6.788 | 371 | 405 | 379 |
| retinaface-mobilenet0.25 | 1080x1920 | 3.444 | 2.738 | 1.258 | 27.736 | 87 | 86 | 105 |
| rtmo-t | 1080x1920 | 7.621 | 6.313 | 1.207 | 1.79 | 90 | 121 | 121 |
| rtmpose-t-halpe26 | 1080x1920 | 4.479 | 2.904 | 1.542 | 59.136 | 103 | 120 | 129 |
| mobile-sam | 1024x1024 | 10.518 | 6.859 | 1.533 | 3.431 | 207 | 183 | 215 |
| dat-x4 | 512x512 | 1565.007 | 1421.149 | 1.101 | 3.964 | 2443 | 2552 | 2737 |
| dat-2-x4 | 512x512 | 1337.965 | 1246.667 | 1.073 | 3.915 | 2436 | 2546 | 2730 |
| seedvr2-3b | 512x512 | 2281.497 | 2281.23 | 1.0 | 5.764 | 7598 | 11957 | 11957 |
| swinir-m-x2 | 512x512 | 1192.312 | 557.755 | 2.138 | 5.831 | 1500 | 1386 | 1577 |
| locotrack-b | 480x854 | 190.125 | 177.607 | 1.07 | 1.145 | 719 | 771 | 782 |
| locotrack-s | 480x854 | 179.541 | 151.077 | 1.188 | 1.276 | 706 | 759 | 764 |
| track-on2 | 480x854 | 1238.277 | 1120.302 | 1.105 | 1.743 | 1236 | 1423 | 1368 |

Kernels peak memory above PORTABLE: 150 refs, full list in `audit_rows.csv` (flag `kernels_mem_above_portable`); worst: shufflenet_v2_x1_0.tv_in1k 2.122x (41 -> 87 MB), eva02_large_patch14_448.mim_m38m_ft_in22k_in1k 1.907x (648 -> 1236 MB), mobilenet_v2.tv2_in1k 1.744x (43 -> 75 MB), mobilenetv1_100.ra4_e3600_r224_in1k 1.727x (44 -> 76 MB), clip-l14-336-openai 1.725x (639 -> 1102 MB), clip-h14-378-dfn5b 1.721x (1280 -> 2203 MB), clip-h14-dfn5b 1.717x (1252 -> 2150 MB), clip-h14-378-dfn5b 1.717x (1254 -> 2153 MB), pe-core-t16-384 1.698x (53 -> 90 MB), mobilenet_v3_large.tv2_in1k 1.696x (46 -> 78 MB), clip-l14-openai 1.692x (623 -> 1054 MB), clip-l14-dfn2b 1.692x (623 -> 1054 MB).

## (b) Families without kernels: reason real or not, ranked

Score = users (1-5, how likely a user runs it) x (expected kernels gain vs portable - 1) x 10. Expected gain from the measured portable numbers and from what the same shared kernels gave in sibling families (fastbb + dpt_fused ViT path: 1.4-3x vs portable in depth_anything_v2; cudnn_hacc / cutlass_conv CNN runners 1.2-1.5x); an estimate until built. "REAL" = measured slower or tiny / already minimal; "PARTLY" = the claim holds for one part only.

| rank | family | task | refs | verdict | users | gain | effort | portable vs orig (speed / mem) | why |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lightglue | matching | 2 | NOT real | 5 | 1.6x | M-H | 1.015x / 0.997x | portable only 1.015x / 1.0x mem vs original (aliked 40.6 vs 41.2 ms, disk 116.4 vs 116.7); portable runs fp32 -> fp16 convs of DISK U-Net / ALIKED + fp16 attention are untouched; counts are data-dependent only in the post-processing, not in the dense nets |
| 2 | grounding_dino | detection | 4 | PARTLY real | 4 | 1.4x | H | 1.451x / 0.609x | measured: fp16-accumulate GEMMs alone gain 0-3 %; the remaining time is multi-scale deformable attention (grid_sample) + small ops, which needs a MSDA kernel + CUDA graphs; portable 1.45x / 0.60x mem already |
| 3 | codeformer | restoration | 1 | NOT real | 4 | 1.4x | M | 1.764x / 0.534x | portable 48.6 vs 85.7 ms (1.76x), 0.53x mem; VQGAN 3x3 convs + GroupNorm dominate -> cudnn_hacc / cutlass_conv + fused GroupNorm epilogue exist in the lib (used by other restoration families) |
| 4 | waft | flow | 3 | NOT real | 2 | 1.8x | L | 1.939x / 0.936x | portable 790 vs 1533 ms at 1080x1920; the DINOv2-S + DPT feature net uses the same fastbb + dpt_fused kernels the depth_anything_v2 family already has (the card says so); the 5 refinement ViT iterations reuse the timm ViT blocks |
| 5 | loftr | matching | 2 | NOT real | 4 | 1.3x | M | 2.198x / 0.458x | portable already 2.2x (110 vs 241 ms), 0.46x mem; "data-dependent match counts" only concerns the final mutual-NN step; the ResNet-FPN + 8 linear-attention blocks are dense and fusable |
| 6 | dedode | matching | 2 | NOT real | 3 | 1.3x | M-H | 1.292x / 0.917x | "not implemented yet"; portable 1.29x / 0.92x mem (G: 592 vs 666 ms); DeDoDe-G is a DINOv2-L ViT + VGG decoders: fastbb/dpt_fused ViT path + cudnn_hacc convs fit |
| 7 | flexglue | matching | 1 | NOT real | 2 | 1.4x | M | x / x | same dense net / data-dependent post-processing split as LightGlue; no speed rows yet |
| 8 | flowformer | flow | 3 | FIX PORTABLE FIRST | 2 | 1.4x | H | 0.864x / 1.329x | portable fails the bar (351 vs 303 ms, 3165 vs 2382 MB); the latent-cost-memory attention needs a dedicated kernel, real work |
| 9 | loma | matching | 1 | NOT real | 2 | 1.3x | H | x / x | "not implemented yet"; no speed rows at all yet (gap-fill queue first); DINOv2-G + decoder, large |
| 10 | matanyone | matting | 1 | NOT real | 2 | 1.3x | H | 1.012x / 0.913x | portable only 1.01x / 0.91x mem (1205 vs 1220 ms at 1080x1920); ResNet encoders + memory readout + object transformer: dense ops, nothing measured; large effort for 2-user family |
| 11 | gmflow | flow | 9 | NOT real | 3 | 1.2x | M | 0.997x / 0.997x | portable ties the original (0.94-0.998x speed on 8 of 9 refs; 1.07x mem at 375x1242); window attention + global softmax matching is plain attention (fastbb / cuBLAS fp16) and the matching softmax is fusable |
| 12 | parseq | ocr | 2 | PARTLY real | 2 | 1.3x | L | 1.268x / 0.786x | 16 ms; ViT on 128 tokens + 26-step AR decoder: launch-bound -> a CUDA-graph decode (shared cuda_graphs.py) is a cheap, measurable win; not custom kernels |
| 13 | coatnet | classification | 10 | NOT real | 2 | 1.3x | M | 1.153x / 0.587x | 10 refs, 14 ms at 224; MBConv + relative-position attention; classifier kernels runners exist (convnext/vit/repvit/...) and give 1.2-1.5x vs portable |
| 14 | focalnet | classification | 6 | NOT real | 2 | 1.3x | M | 1.14x / 0.569x | 6 refs, 17 ms; same classifier-runner argument |
| 15 | hgnetv2 | classification | 7 | NOT real | 2 | 1.3x | M | 1.373x / 0.663x | 7 refs, 7 ms; plain CNN: cudnn_hacc / cutlass_conv runner like the CNN classifier families |
| 16 | hed | edges | 1 | UNMEASURED | 3 | 1.2x | L | 3.397x / 0.623x | cuDNN VGG CNN; portable already 3.4x (11.3 vs 38.5 ms); the "no kernel measured to pay off" claim has no number: one cudnn_hacc A/B decides |
| 17 | lineart | edges | 2 | UNMEASURED | 3 | 1.2x | L | 1.624x / 1.005x | portable 1.6x, mem 1.005x of the original (fails the memory bar by 0.5 %); small conv generator; same A/B |
| 18 | ppocrv5 | ocr | 2 | PARTLY real | 4 | 1.1x | M | 2.757x / 0.735x | portable 2.76x already; cuDNN convs + CPU box geometry; only cudnn_hacc fp16 convs left to try (untried) |
| 19 | lineart_anime | edges | 1 | UNMEASURED | 2 | 1.2x | L | 2.892x / 0.626x | pix2pix U-Net, 24 ms; portable 2.9x; same A/B |
| 20 | xfeat | matching | 2 | PARTLY real | 3 | 1.1x | M | 1.139x / 0.993x | small CNN (8.2 ms extractor); portable 1.14x; the per-point steps are data dependent: low expected gain; lighterglue 22 ms is LightGlue-like (follows the lightglue work) |
| 21 | flowseek | flow | 3 | NOT real | 1 | 1.3x | M | 1.156x / 0.997x | portable 1.16x / 1.0x mem; the card says the SEA-RAFT engine + DA kernels fit (both exist) |
| 22 | davit | classification | 3 | NOT real | 1 | 1.3x | M | 1.095x / 0.569x | 3 refs; portable 1.095x only |
| 23 | shvit | classification | 4 | NOT real | 1 | 1.2x | M | 1.239x / 0.7x | 4 refs, 9 ms; portable 1.24x |
| 24 | efficientformer | classification | 3 | NOT real | 1 | 1.2x | M | 1.308x / 0.982x | portable mem 1.07x above the original (fails the bar: 93 vs 87 MB), 3 refs |
| 25 | efficientformerv2 | classification | 4 | NOT real | 1 | 1.2x | M | 1.134x / 0.769x | 4 refs, 15 ms |
| 26 | anyline | edges | 1 | REAL-ish (tiny CNN + GPU blend) | 2 | 1.1x | - | 12.456x / 0.719x | portable 12.5x already; no numbers on the kernel side |
| 27 | bisenet | segmentation | 1 | REAL | 2 | 1.0x | - | 2.164x / 0.739x | measured slower (cudnn_hacc 2.77 vs 2.41 ms portable at 512x512; 2.2 ms model) |
| 28 | inspyrenet | matting | 2 | REAL | 3 | 1.0x | - | 2.22x / 0.533x | measured slower (cudnn_hacc 189.9 vs 159.4 ms; fp16-accumulate GEMMs alone 3 %) |
| 29 | pidinet | edges | 1 | REAL | 2 | 1.0x | - | 2.805x / 0.968x | few ms plain convs, portable 2.8x / 0.97x mem |
| 30 | mlsd | edges | 1 | REAL | 2 | 1.0x | - | 3.797x / 0.686x | MobileNetV2 small CNN, portable 3.8x |
| 31 | teed | edges | 1 | REAL | 2 | 1.0x | - | 7.219x / 0.819x | 58 k parameters, few ms, portable 7.2x |

Summary: REAL = bisenet, inspyrenet (measured slower), pidinet, mlsd, teed, anyline (tiny / minimal); the three other edge families are UNMEASURED (one cudnn_hacc A/B each decides); everything else has no real reason. Portable also fails the bar for flowformer (slower, +33 % memory), gmflow (ties), neuflow-v2, film, moge-3, efficientvit-b2, squeezenet, the taggers / yolox / retinaface / rtmpose / swin-family memory (see (a)); these are portable fixes, not kernels work.

Step 2 picks (effort-adjusted from the top of the ranking, see the results files of each): see README log.
