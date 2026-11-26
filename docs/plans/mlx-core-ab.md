# MLX core A/B: hand-written networks against the graph translation (2026-11-26)

Every hand-ported ref (the families with `GRAPH_RUNNERS`, and the libnet classifiers written out in `mlx_cnn` /
`mlx_vit_cls`), portable mode, float16 and float32, on the network alone: `tools/mlx_core_ab.py` (how-to in
[mlx-coverage-gaps.md](mlx-coverage-gaps.md#how-to-ab-a-hand-written-mlx-network-against-its-graph-translation-2026-11-26)).
M1 Pro 16 GB, MLX 0.32.4, buffer cache 2048 MB in both arms, a throwaway warm-up process before each
arm (warm conversion cache), 4 warm-up calls then the median of 5. Rows: [mlx-core-ab.jsonl](mlx-core-ab.jsonl).
Diagnostic only: nothing here changes a runner; no survey rows were written.

**Bar**: the hand network at most 2 % slower (warm) and at most 2 % heavier (network peak) than the exported-graph
translation it replaced. 583 rows over 80 families; 579 measured in both arms.

- 325 rows flagged (139 slower, 238 heavier) in 65 families. Median hand / graph:
  warm 0.92, peak 0.98.
- Beyond 10 % (same number of network calls in both arms): 151 rows, listed first below.
- 44 flagged rows time a different number of network calls per model call in the two arms (the hand port
  splits or merges the network differently): their warm sums compare, their peaks (the largest single call) less so.
- Not fixed here: the brief is measurement only.

## Flagged beyond 10 %

| ref | dtype | net warm ms hand / graph | x | net peak MB hand / graph | x | flag |
|---|---|---|---|---|---|---|
| segformer-b2-clothes | fp16 | 43.7 / 24.4 | 1.79 | 517 / 158 | 3.27 | slower, heavier |
| segformer-b0-ade20k | fp16 | 32.9 / 17.3 | 1.91 | 346 / 114 | 3.04 | slower, heavier |
| segformer-b0-ade20k | fp32 | 33.6 / 23.1 | 1.46 | 373 / 149 | 2.50 | slower, heavier |
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 2072 / 855 | 2.42 | heavier |
| segformer-b2-clothes | fp32 | 46.1 / 32.2 | 1.43 | 589 / 252 | 2.34 | slower, heavier |
| segformer-b5-ade20k | fp16 | 278.2 / 206.5 | 1.35 | 1287 / 555 | 2.32 | slower, heavier |
| segformer-b5-face-parsing | fp16 | 83.9 / 62.6 | 1.34 | 627 / 274 | 2.29 | slower, heavier |
| segformer-b5-ade20k | fp32 | 311.6 / 262.6 | 1.19 | 1625 / 780 | 2.08 | slower, heavier |
| efficientformerv2_s2.snap_dist_in1k | fp16 | 7.1 / 6.2 | 1.14 | 115 / 56 | 2.06 | slower, heavier |
| hrnet-w48-fcn-ade20k | fp16 | 180.2 / 162.1 | 1.11 | 534 / 292 | 1.83 | slower, heavier |
| efficientformerv2_l.snap_dist_in1k | fp16 | 10.9 / 9.1 | 1.20 | 178 / 99 | 1.80 | slower, heavier |
| lw-detr-medium-coco | fp32 | 42.8 / 42.8 | 1.00 | 420 / 240 | 1.75 | heavier |
| efficientformerv2_s1.snap_dist_in1k | fp16 | 4.4 / 4.2 | 1.04 | 63 / 37 | 1.72 | slower, heavier |
| efficientformerv2_s0.snap_dist_in1k | fp16 | 3.5 / 3.2 | 1.11 | 44 / 26 | 1.72 | slower, heavier |
| selective-igev-kitti15 | fp32 | 4786.6 / 6315.6 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-finetune | fp32 | 4782.4 / 6335.6 | 0.75 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-train | fp32 | 4793.4 / 6300.8 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-kitti12 | fp32 | 4787.0 / 6291.4 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-finetune | fp32 | 4789.9 / 6299.3 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-train | fp32 | 4801.0 / 6367.1 | 0.75 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-sceneflow | fp32 | 4796.5 / 6299.9 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-finetune | fp16 | 3659.3 / 4639.0 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-eth3d-train | fp16 | 3650.8 / 4622.7 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-kitti12 | fp16 | 3642.7 / 4624.8 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-kitti15 | fp16 | 3648.9 / 4623.3 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-middlebury-finetune | fp16 | 3646.4 / 4626.4 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-middlebury-train | fp16 | 3642.6 / 4641.0 | 0.78 | 1873 / 1102 | 1.70 | heavier |
| selective-igev-sceneflow | fp16 | 3662.8 / 4645.5 | 0.79 | 1873 / 1102 | 1.70 | heavier |
| segnext-mscan-l-ade20k | fp16 | 143.2 / 127.0 | 1.13 | 519 / 305 | 1.70 | slower, heavier |
| rf-detr-keypoint-preview | fp32 | 72.3 / 74.0 | 0.98 | 428 / 256 | 1.67 | heavier |
| igev-stereo-middlebury | fp16 | 3187.7 / 3596.0 | 0.89 | 1686 / 1010 | 1.67 | heavier |
| igev-stereo-sceneflow | fp16 | 3187.4 / 3609.1 | 0.88 | 1686 / 1010 | 1.67 | heavier |
| segnext-mscan-t-ade20k | fp32 | 36.2 / 32.5 | 1.11 | 355 / 225 | 1.58 | slower, heavier |
| lw-detr-small-coco | fp32 | 28.9 / 29.2 | 0.99 | 341 / 216 | 1.58 | heavier |
| segnext-mscan-b-ade20k | fp16 | 80.6 / 70.3 | 1.15 | 396 / 251 | 1.58 | slower, heavier |
| raft-stereo-eth3d | fp32 | 4832.4 / 5011.0 | 0.96 | 2943 / 1872 | 1.57 | heavier |
| raft-stereo-middlebury | fp32 | 4808.4 / 5041.7 | 0.95 | 2943 / 1872 | 1.57 | heavier |
| raft-stereo-sceneflow | fp32 | 4812.2 / 5045.0 | 0.95 | 2943 / 1872 | 1.57 | heavier |
| lw-detr-xlarge-coco | fp16 | 96.9 / 108.3 | 0.90 | 698 / 445 | 1.57 | heavier |
| raft-stereo-sceneflow | fp16 | 3191.9 / 4043.8 | 0.79 | 1630 / 1042 | 1.56 | heavier |
| segformer-b5-face-parsing | fp32 | 92.6 / 79.0 | 1.17 | 808 / 517 | 1.56 | slower, heavier |
| igev-stereo-eth3d | fp16 | 3185.1 / 3594.5 | 0.89 | 1686 / 1080 | 1.56 | heavier |
| igev-stereo-kitti12 | fp16 | 3190.2 / 3603.2 | 0.89 | 1686 / 1080 | 1.56 | heavier |
| igev-stereo-kitti15 | fp16 | 3186.4 / 3601.5 | 0.88 | 1686 / 1080 | 1.56 | heavier |
| segnext-mscan-s-ade20k | fp16 | 44.8 / 37.4 | 1.20 | 310 / 201 | 1.54 | slower, heavier |
| lw-detr-large-coco | fp16 | 53.4 / 59.4 | 0.90 | 534 / 347 | 1.54 | heavier |
| igev-stereo-eth3d | fp32 | 4179.4 / 4676.3 | 0.89 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-kitti12 | fp32 | 4177.9 / 4578.8 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-kitti15 | fp32 | 4162.0 / 4564.7 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-middlebury | fp32 | 4175.4 / 4581.2 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-sceneflow | fp32 | 4171.9 / 4560.1 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 17.9 / 17.3 | 1.03 | 441 / 290 | 1.52 | slower, heavier |
| nafnet-sidd | fp32 | 254.2 / 268.5 | 0.95 | 1346 / 896 | 1.50 | heavier |
| segnext-mscan-t-ade20k | fp16 | 32.5 / 26.0 | 1.25 | 259 / 176 | 1.47 | slower, heavier |
| rf-detr-seg-2xl-coco | fp32 | 156.5 / 155.1 | 1.01 | 576 / 397 | 1.45 | heavier |
| raft-stereo-middlebury | fp16 | 3168.7 / 4054.7 | 0.78 | 1630 / 1128 | 1.45 | heavier |
| segnext-mscan-b-ade20k | fp32 | 87.4 / 87.8 | 1.00 | 528 / 366 | 1.44 | heavier |
| rf-detr-seg-2xl-coco | fp16 | 126.2 / 119.9 | 1.05 | 394 / 274 | 1.44 | slower, heavier |
| convnext_small.fb_in22k_ft_in1k_384 | fp32 | 26.4 / 24.8 | 1.07 | 443 / 308 | 1.44 | slower, heavier |
| raft-stereo-eth3d | fp16 | 3176.0 / 4061.6 | 0.78 | 1630 / 1166 | 1.40 | heavier |
| nafnet-gopro | fp32 | 257.5 / 291.6 | 0.88 | 1170 / 839 | 1.39 | heavier |
| davit_small.msft_in1k | fp32 | 17.1 / 16.3 | 1.05 | 389 / 279 | 1.39 | slower, heavier |
| shufflenet_v2_x1_0.tv_in1k | fp16 | 3.2 / 2.3 | 1.39 | 15 / 17 | 0.90 | slower |
| nafnet-sidd | fp16 | 219.4 / 207.7 | 1.06 | 804 / 578 | 1.39 | slower, heavier |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 20.0 / 21.2 | 0.95 | 583 / 422 | 1.38 | heavier |
| lineart | fp32 | 98.6 / 117.6 | 0.84 | 1245 / 903 | 1.38 | heavier |
| lineart-coarse | fp32 | 96.5 / 117.3 | 0.82 | 1245 / 903 | 1.38 | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 36.0 / 31.9 | 1.13 | 697 / 510 | 1.37 | slower, heavier |
| rf-detr-keypoint-preview | fp16 | 62.6 / 61.5 | 1.02 | 354 / 260 | 1.36 | heavier |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1647 / 1208 | 1.36 | heavier |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1647 / 1209 | 1.36 | heavier |
| efficientnet_b4.tv_in1k | fp32 | 13.8 / 18.0 | 0.77 | 349 / 259 | 1.35 | heavier |
| davit_base.msft_in1k | fp32 | 21.6 / 21.7 | 0.99 | 562 / 425 | 1.32 | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 2197 / 1668 | 1.32 | heavier |
| sam2.1-hiera-tiny | fp32 | 130.7 / 127.7 | 1.02 | 572 / 436 | 1.31 | slower, heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 1523 / 1166 | 1.31 | heavier |
| shvit_s1.in1k | fp16 | 3.9 / 3.0 | 1.30 | 20 / 20 | 0.98 | slower |
| segnext-mscan-s-ade20k | fp32 | 49.0 / 46.8 | 1.05 | 471 / 363 | 1.30 | slower, heavier |
| sam2.1-hiera-small | fp32 | 155.0 / 152.5 | 1.02 | 599 / 463 | 1.29 | heavier |
| wd-vit-tagger-v3 | fp32 | 53.7 / 53.8 | 1.00 | 572 / 444 | 1.29 | heavier |
| wd-convnext-tagger-v3 | fp32 | 52.4 / 52.3 | 1.00 | 657 / 512 | 1.28 | heavier |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2901 / 2261 | 1.28 | heavier |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2901 / 2261 | 1.28 | heavier |
| totalsegmentator-total-3mm | fp16 | 304.4 / 425.0 | 0.72 | 2058 / 1609 | 1.28 | heavier |
| convnext_base.fb_in22k_ft_in1k_384 | fp32 | 36.8 / 37.0 | 1.00 | 592 / 464 | 1.28 | heavier |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 1311 / 1026 | 1.28 | heavier |
| deeplabv3plus-r101-d8-ade20k | fp16 | 278.2 / 288.2 | 0.97 | 860 / 674 | 1.28 | heavier |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 15.8 / 14.1 | 1.12 | 279 / 221 | 1.26 | slower, heavier |
| repvit_m1_1.dist_450e_in1k | fp16 | 5.7 / 4.5 | 1.26 | 42 / 46 | 0.93 | slower |
| pspnet-r101-d8-ade20k | fp16 | 290.3 / 302.1 | 0.96 | 966 / 768 | 1.26 | heavier |
| davit_tiny.msft_in1k | fp16 | 9.3 / 7.5 | 1.24 | 127 / 129 | 0.98 | slower |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 488 / 393 | 1.24 | heavier |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 483 / 393 | 1.23 | heavier |
| davit_tiny.msft_in1k | fp32 | 10.0 / 9.2 | 1.09 | 229 / 187 | 1.23 | slower, heavier |
| segnext-mscan-l-ade20k | fp32 | 155.4 / 158.8 | 0.98 | 609 / 497 | 1.23 | heavier |
| convnextv2_base.fcmae_ft_in22k_in1k | fp32 | 20.1 / 19.9 | 1.01 | 558 / 456 | 1.22 | heavier |
| xfeat | fp32 | 16.9 / 31.8 | 0.53 | 199 / 165 | 1.21 | heavier |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.4 / 4.5 | 1.21 | 67 / 66 | 1.01 | slower |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp16 | 15.3 / 14.1 | 1.08 | 225 / 186 | 1.21 | slower, heavier |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | 2.3 / 1.9 | 1.20 | 23 / 23 | 0.97 | slower |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 14.2 / 12.0 | 1.19 | 151 / 161 | 0.93 | slower |
| hrnet-w48-fcn-ade20k | fp32 | 200.0 / 194.2 | 1.03 | 723 / 609 | 1.19 | slower, heavier |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 1062 / 900 | 1.18 | heavier |
| davit_base.msft_in1k | fp16 | 19.5 / 18.1 | 1.08 | 300 / 255 | 1.18 | slower, heavier |
| mobilenet_v2.tv2_in1k | fp16 | 2.4 / 2.0 | 1.18 | 23 / 24 | 0.95 | slower |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 27.6 / 25.2 | 1.10 | 349 / 297 | 1.17 | slower, heavier |
| deeplabv3-r101-d8-ade20k | fp16 | 365.2 / 378.3 | 0.97 | 682 / 582 | 1.17 | heavier |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp32 | 14.2 / 15.4 | 0.92 | 397 / 340 | 1.17 | heavier |
| coatnet_1_rw_224.sw_in1k | fp32 | 14.3 / 15.3 | 0.93 | 397 / 340 | 1.17 | heavier |
| sam2.1-hiera-large | fp16 | 555.8 / 562.2 | 0.99 | 849 / 734 | 1.16 | heavier |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 622 / 539 | 1.15 | heavier |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 613 / 531 | 1.15 | heavier |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.2 | 0.97 | 471 / 409 | 1.15 | heavier |
| deeplabv3plus-r101-d8-ade20k | fp32 | 354.0 / 379.1 | 0.93 | 1392 / 1209 | 1.15 | heavier |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 581 / 505 | 1.15 | heavier |
| totalsegmentator-total-3mm | fp32 | 414.7 / 535.0 | 0.78 | 2033 / 1769 | 1.15 | heavier |
| squeezenet1_1.tv_in1k | fp32 | 2.0 / 1.7 | 1.15 | 28 / 32 | 0.86 | slower |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp16 | 18.3 / 17.8 | 1.03 | 302 / 263 | 1.15 | slower, heavier |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.8 / 5.9 | 1.15 | 89 / 108 | 0.82 | slower |
| convnext_tiny.fb_in1k | fp16 | 6.8 / 5.9 | 1.15 | 89 / 108 | 0.82 | slower |
| repvit_m1_5.dist_300e_in1k | fp16 | 6.0 / 7.5 | 0.79 | 73 / 64 | 1.14 | heavier |
| repvit_m1_5.dist_450e_in1k | fp16 | 5.9 / 7.5 | 0.78 | 73 / 64 | 1.14 | heavier |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 1119 / 982 | 1.14 | heavier |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 1107 / 972 | 1.14 | heavier |
| pe-core-b16-224 | fp32 | 17.3 / 17.6 | 0.98 | 496 / 436 | 1.14 | heavier |
| sam2.1-hiera-large | fp32 | 600.5 / 608.8 | 0.99 | 1495 / 1317 | 1.14 | heavier |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 23.8 / 21.1 | 1.13 | 242 / 233 | 1.04 | slower, heavier |
| wd-vit-tagger-v3 | fp16 | 44.8 / 43.9 | 1.02 | 283 / 251 | 1.13 | heavier |
| convnext_tiny.fb_in1k | fp32 | 7.5 / 6.7 | 1.13 | 168 / 206 | 0.82 | slower |
| rf-detr-l-coco | fp32 | 52.6 / 51.0 | 1.03 | 268 / 238 | 1.13 | slower, heavier |
| efficientvit_b2.r288_in1k | fp32 | 7.7 / 10.1 | 0.76 | 218 / 194 | 1.13 | heavier |
| convnextv2_base.fcmae_ft_in22k_in1k | fp16 | 17.7 / 17.4 | 1.01 | 358 / 318 | 1.13 | heavier |
| igev-plusplus-eth3d | fp32 | 6887.7 / 7321.9 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti12 | fp32 | 6895.0 / 7199.4 | 0.96 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti15 | fp32 | 6871.0 / 7144.8 | 0.96 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-middlebury | fp32 | 6856.9 / 7342.5 | 0.93 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-sceneflow | fp32 | 6821.8 / 7224.3 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| davit_small.msft_in1k | fp16 | 15.6 / 13.8 | 1.13 | 201 / 185 | 1.09 | slower, heavier |
| efficientformerv2_s2.snap_dist_in1k | fp32 | 6.6 / 6.5 | 1.02 | 118 / 105 | 1.12 | heavier |
| focalnet_base_srf.ms_in1k | fp32 | 20.2 / 24.7 | 0.82 | 551 / 492 | 1.12 | heavier |
| sam2.1-hiera-base-plus | fp16 | 232.8 / 245.5 | 0.95 | 535 / 478 | 1.12 | heavier |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 69.3 / 70.8 | 0.98 | 991 / 892 | 1.11 | heavier |
| focalnet_base_lrf.ms_in1k | fp32 | 21.1 / 26.1 | 0.81 | 544 / 491 | 1.11 | heavier |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 7.4 / 6.7 | 1.11 | 168 / 206 | 0.82 | slower |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 81 / 74 | 1.11 | heavier |
| sapiens-seg-0.3b-goliath | fp32 | 1415.1 / 1781.7 | 0.79 | 3631 / 3288 | 1.10 | heavier |
| coatnet_rmlp_1_rw_224.sw_in1k | fp32 | 14.0 / 14.9 | 0.94 | 390 / 353 | 1.10 | heavier |
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 2112 / 1914 | 1.10 | heavier |
| igev-plusplus-kitti12 | fp16 | 5648.8 / 6263.4 | 0.90 | 2414 / 2189 | 1.10 | heavier |
| igev-plusplus-middlebury | fp16 | 5711.5 / 6166.5 | 0.93 | 2414 / 2189 | 1.10 | heavier |
| igev-plusplus-eth3d | fp16 | 5647.5 / 7058.8 | 0.80 | 2414 / 2189 | 1.10 | heavier |

## Per family

Flagged rows / rows, the worst hand / graph ratio of the family's rows (warm, peak); every flagged row is marked in
the full table below.

| family | flagged | slower | heavier | worst warm x | worst peak x |
|---|---|---|---|---|---|
| aanet | 3 / 4 | 0 | 3 | 0.54 | 2.42 |
| alexnet | 0 / 2 | 0 | 0 | 0.99 | 0.96 |
| anyline | 0 / 2 | 0 | 0 | 0.64 | 0.65 |
| clip | 13 / 20 | 13 | 0 | 1.04 | 0.98 |
| coatnet | 7 / 20 | 0 | 7 | 1.02 | 1.17 |
| coex | 0 / 2 | 0 | 0 | 0.50 | 0.53 |
| convnext | 20 / 24 | 14 | 9 | 1.19 | 1.44 |
| convnextv2 | 2 / 2 | 0 | 2 | 1.01 | 1.22 |
| davit | 6 / 6 | 5 | 5 | 1.24 | 1.39 |
| deit | 2 / 2 | 1 | 2 | 1.02 | 1.08 |
| densenet | 2 / 2 | 0 | 2 | 0.77 | 1.11 |
| depth_anything_3 | 8 / 10 | 8 | 0 | 1.06 | 1.00 |
| depth_pro | 0 / 2 | 0 | 0 | 0.96 | 0.98 |
| dfine | 4 / 20 | 0 | 4 | 0.92 | 1.24 |
| dlnr | 4 / 4 | 0 | 4 | 0.83 | 1.36 |
| efficientformer | 1 / 6 | 0 | 1 | 0.90 | 1.04 |
| efficientformerv2 | 8 / 8 | 5 | 7 | 1.21 | 2.06 |
| efficientnet | 1 / 4 | 0 | 1 | 0.85 | 1.35 |
| efficientnetv2 | 1 / 2 | 0 | 1 | 0.89 | 1.04 |
| efficientvit | 1 / 2 | 0 | 1 | 0.81 | 1.13 |
| efficientvit_sam | 2 / 5 | 0 | 2 | 1.02 | 1.32 |
| eva02 | 2 / 2 | 0 | 2 | 0.99 | 1.08 |
| fastvit | 3 / 6 | 3 | 0 | 1.09 | 0.88 |
| flexglue | 0 / 2 | 0 | 0 | 0.98 | 0.80 |
| focalnet | 9 / 12 | 3 | 6 | 1.03 | 1.12 |
| hgnetv2 | 1 / 14 | 1 | 0 | 1.06 | 0.95 |
| igev_plusplus | 10 / 10 | 0 | 10 | 0.96 | 1.13 |
| igev_stereo | 10 / 10 | 0 | 10 | 0.91 | 1.67 |
| inception_v3 | 0 / 2 | 0 | 0 | 0.79 | 0.59 |
| lightglue | 2 / 4 | 1 | 2 | 1.03 | 1.05 |
| lineart | 4 / 4 | 0 | 4 | 0.84 | 1.38 |
| loftr | 4 / 4 | 4 | 4 | 1.30 | 2.91 |
| loma | 2 / 12 | 1 | 2 | 1.03 | 1.18 |
| lwdetr | 6 / 10 | 0 | 6 | 1.00 | 1.75 |
| mask2former | 0 / 2 | 0 | 0 | 0.77 | 0.98 |
| mmseg | 8 / 10 | 3 | 7 | 1.11 | 1.83 |
| mobilenetv1 | 2 / 2 | 2 | 0 | 1.20 | 0.98 |
| mobilenetv2 | 1 / 2 | 1 | 0 | 1.18 | 0.96 |
| mobilenetv3 | 0 / 2 | 0 | 0 | 0.92 | 0.81 |
| mobilenetv4 | 1 / 2 | 1 | 0 | 1.07 | 0.97 |
| mvsanywhere | 2 / 2 | 2 | 0 | 1.11 | 1.00 |
| nafnet | 3 / 4 | 1 | 3 | 1.06 | 1.50 |
| nsfw_vit | 2 / 2 | 1 | 2 | 1.03 | 1.08 |
| oneformer | 2 / 2 | 0 | 2 | 0.77 | 1.74 |
| perception_encoder | 6 / 20 | 1 | 6 | 1.03 | 1.15 |
| raft_stereo | 6 / 6 | 0 | 6 | 0.96 | 1.57 |
| rcnn | 4 / 4 | 0 | 4 | 0.95 | 1.74 |
| regnet | 0 / 2 | 0 | 0 | 0.93 | 0.85 |
| repvit | 7 / 20 | 1 | 6 | 1.26 | 1.14 |
| resnet | 0 / 6 | 0 | 0 | 0.90 | 0.87 |
| resplat | no row measured in both arms | | | | |
| rfdetr | 19 / 20 | 18 | 5 | 1.09 | 1.45 |
| rfdetr_keypoint | 2 / 2 | 0 | 2 | 1.02 | 1.67 |
| rtdetrv2 | 3 / 10 | 0 | 3 | 0.92 | 1.08 |
| rtmdet | 0 / 14 | 0 | 0 | 0.92 | 0.97 |
| rtmo | 0 / 8 | 0 | 0 | 0.84 | 1.02 |
| rtmpose | 25 / 56 | 0 | 25 | 0.98 | 1.50 |
| sam | 3 / 6 | 0 | 3 | 0.90 | 1.15 |
| sam2 | 8 / 8 | 1 | 8 | 1.02 | 1.31 |
| sam3 | 2 / 2 | 2 | 2 | 1.09 | 1.56 |
| sam_hq | 3 / 6 | 0 | 3 | 0.90 | 1.15 |
| sapiens | 2 / 4 | 0 | 2 | 0.92 | 1.10 |
| segformer | 8 / 8 | 8 | 8 | 1.91 | 3.27 |
| segnext | 8 / 8 | 6 | 8 | 1.25 | 1.70 |
| selective_igev | 14 / 14 | 0 | 14 | 0.79 | 1.70 |
| shufflenetv2 | 1 / 2 | 1 | 0 | 1.39 | 0.90 |
| shvit | 5 / 8 | 5 | 0 | 1.30 | 0.99 |
| siglip | 1 / 2 | 1 | 0 | 1.02 | 1.02 |
| siglip2 | 11 / 20 | 11 | 0 | 1.05 | 1.01 |
| squeezenet | 1 / 2 | 1 | 0 | 1.15 | 0.86 |
| starnet | 0 / 8 | 0 | 0 | 0.87 | 0.83 |
| swin | 2 / 2 | 1 | 2 | 1.03 | 1.38 |
| tinyvit | 8 / 10 | 8 | 4 | 1.13 | 1.52 |
| totalsegmentator | 2 / 4 | 0 | 2 | 0.78 | 1.28 |
| unik3d | 1 / 2 | 0 | 1 | 1.02 | 1.31 |
| vgg | 0 / 2 | 0 | 0 | 0.97 | 0.91 |
| vit | 2 / 2 | 1 | 2 | 1.02 | 1.08 |
| wd_tagger | 7 / 8 | 2 | 6 | 1.04 | 1.29 |
| xfeat | 2 / 2 | 1 | 2 | 4.07 | 2.77 |
| yolox | 3 / 10 | 0 | 3 | 0.89 | 1.15 |

## SAM 3 after its parity pass (2026-11-26)

The two sam3 rows were measured again after the pass (linear weights stored `[in, out]`, the cached load path,
evaluation boundaries at the 288 x 288 level, the neck levels handed to the prompt without a copy); the counts at the
top are those of the first run. Both rows stay flagged, and by construction: the graph arm translates only `embed`
(trunk and neck, one network call) and runs the text encoder and the detector heads in PyTorch on the CPU, outside the
network timers; the hand arm runs them in MLX as a second network call (`detect`) and keeps their weights resident.
`net_call_ms` / `net_call_peak_mb` (new in the rows) split the hand arm's calls:

| dtype | embed warm ms hand / graph | x | embed peak MB hand / graph | resident MB | embed peak over resident MB | detect (hand only) ms / peak MB | e2e warm ms | e2e footprint MB |
|---|---|---|---|---|---|---|---|---|
| fp16 | 1371.9 / 1439.5 | 0.95 | 1970 / 1385 | 1633 / 892 | 336 / 493 | 202.1 / 2006 | 1590.7 / 2235.5 | 3496 / 4180 |
| fp32 | 1637.2 / 1741.0 | 0.94 | 3785 / 2551 | 3205 / 1776 | 580 / 775 | 260.2 / 3972 | 1909.2 / 2695.2 | 5625 / 5843 |

- On the call both arms make, the hand network is 5 to 6 % faster and its activations are a quarter to a third
  leaner; before the pass the same split was about 1320 / 1375 ms (fp16).
- The network peak is higher only by the text encoder's and the detector's weights (741 MB fp16, 1429 MB fp32), which
  the graph arm holds in PyTorch instead: its process footprint is the larger one. They buy the detector in MLX:
  29 % faster end to end in both dtypes.
- Hand peak before / after the pass: 2263 -> 2006 MB (fp16), 4202 -> 3972 MB (fp32).

## Families whose graph path could not be rebuilt

These have no `GRAPH_RUNNERS` (restoring the pre-port runner would change the torch path or need far more than a small
wrapper), so they are not in the A/B:

- fast_acvnet: the hand port removed the `exporting_for_mlx` / float32-island logic from the torch forward.
- gmflow: the export branches were removed from the torch path (large diff).
- raft: needs the deleted 148-line `raft/mlx.py`.
- sea_raft: needs the deleted `sea_raft/mlx.py` and its Winograd arguments.
- rife: the export branches were removed from `_interpolate`.
- ema_vfi: the torch-side `_loaded` / `_net` were rewritten.
- realplksr, restormer: the custom ops and their lowerings were stripped (over 60 lines, another file).
- moge: needs the deleted `_below_top`, `_rep_pad32`, the old `_trunk` and `to_float32` in the torch `_FoldedTop`.
- resplat: `GRAPH_RUNNERS` builds, but today's torch network calls the custom op `fastforks.resplat_knn`, which the
  graph translator has no mapping for (`UnsupportedOp`): no graph arm.

Hand-written from the start (never had a graph path): crestereo, seedvr2, dinov2, dinov3, c_radio, tipsv2, romav2,
depth_anything_v2, marigold_v2, dinov2_linear, dinov2_linear_cls.

## Rows with an arm that did not run

- resplat-base-re10k fp16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-base-re10k fp32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv fp16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv fp32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default

## Full table

hand / graph per cell; `x` = hand / graph; `net rel` = max |hand - graph| / max |graph| over the first network call's
outputs (only where both arms give the same number of arrays; it differs where the two ports return different
intermediate tensors), `result rel` the same over the model's results (a sanity check, not a golden).

| ref | dtype | net warm ms | x | net first s | net peak MB | x | resident MB | e2e warm ms | load s | e2e first s | net rel | result rel | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 0.71 / 6.15 | 2112 / 1914 | 1.10 | 33 / 33 | 656.3 / 1183.7 | 0.16 / 0.18 | 0.73 / 6.16 | 0.0057 | 0.012 (0.00014 % off) | heavier |
| aanet-plus-sceneflow | fp32 | 850.1 / 1572.5 | 0.54 | 0.99 / 6.25 | 3207 / 3767 | 0.85 | 33 / 49 | 860.7 / 1576.4 | 0.14 / 0.12 | 1.00 / 6.26 | 3.5e-05 | 7.2e-05 |  |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 0.48 / 5.17 | 1062 / 900 | 1.18 | 16 / 23 | 449.0 / 1090.5 | 0.13 / 0.11 | 0.50 / 5.18 | 0.019 (0.00027 % off) | 0.0079 | heavier |
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 0.66 / 5.16 | 2072 / 855 | 2.42 | 16 / 30 | 645.5 / 1183.5 | 0.12 / 0.11 | 0.67 / 5.17 | 0.0002 | 7.3e-05 | heavier |
| alexnet.tv_in1k | fp16 | 2.5 / 2.5 | 0.99 | 0.02 / 0.38 | 119 / 125 | 0.96 | 117 / 117 | 2.8 / 2.9 | 0.64 / 0.49 | 0.02 / 0.38 | 0 | 0 |  |
| alexnet.tv_in1k | fp32 | 3.3 / 3.4 | 0.97 | 0.03 / 0.38 | 238 / 248 | 0.96 | 233 / 233 | 3.6 / 3.8 | 0.62 / 0.49 | 0.03 / 0.38 | 0 | 0 |  |
| aliked-flexglue | fp16 | 932.4 / 954.6 | 0.98 | 4.20 / 4.84 | 800 / 1063 | 0.75 | 263 / 261 | 969.2 / 1000.3 | 0.10 / 0.10 | 4.26 / 4.90 | - | empty (4 keys differ) |  |
| aliked-flexglue | fp32 | 861.4 / 896.2 | 0.96 | 4.08 / 4.73 | 1057 / 1319 | 0.80 | 520 / 518 | 897.7 / 942.5 | 0.10 / 0.10 | 4.14 / 4.78 | - | 2.2e-06 |  |
| aliked-lightglue | fp16 | 89.6 / 164.7 | 0.54 | 0.14 / 1.63 | 714 / 1324 | 0.54 | 49 / 2 | 88.1 / 308.3 | 0.08 / 0.05 | 0.16 / 1.82 | - | 1.7e-05 |  |
| aliked-lightglue | fp32 | 88.3 / 167.4 | 0.53 | 0.13 / 1.63 | 714 / 1324 | 0.54 | 49 / 2 | 88.4 / 309.8 | 0.08 / 0.05 | 0.16 / 1.79 | - | 1.7e-05 |  |
| anyline | fp16 | 49.9 / 77.9 | 0.64 | 0.08 / 1.15 | 394 / 861 | 0.46 | 0 / 0 | 65.4 / 223.5 | 0.02 / 0.02 | 0.11 / 1.34 | - | 0.45 (0.27 % off) |  |
| anyline | fp32 | 66.4 / 122.9 | 0.54 | 0.10 / 1.10 | 743 / 1148 | 0.65 | 0 / 0 | 83.1 / 270.1 | 0.03 / 0.02 | 0.13 / 1.28 | - | 0.16 (0.00019 % off) |  |
| clip-b16-laion2b | fp16 | 14.2 / 14.0 | 1.02 | 0.04 / 0.67 | 186 / 231 | 0.80 | 164 / 164 | 14.5 / 14.2 | 0.85 / 0.71 | 0.04 / 0.68 | 0.0043 | 0.0043 |  |
| clip-b16-laion2b | fp32 | 16.3 / 16.1 | 1.01 | 0.06 / 0.66 | 362 / 417 | 0.87 | 327 / 327 | 16.6 / 16.5 | 0.78 / 0.71 | 0.06 / 0.66 | 6e-06 | 6e-06 |  |
| clip-b16-openai | fp16 | 14.2 / 13.9 | 1.03 | 0.03 / 0.67 | 185 / 231 | 0.80 | 164 / 164 | 14.4 / 14.2 | 0.86 / 0.71 | 0.04 / 0.68 | 0.019 (0.00066 % off) | 0.019 (0.00066 % off) | slower |
| clip-b16-openai | fp32 | 16.4 / 16.1 | 1.02 | 0.06 / 0.68 | 362 / 404 | 0.90 | 327 / 327 | 16.6 / 16.4 | 0.79 / 0.72 | 0.06 / 0.68 | 2.4e-05 | 2.4e-05 |  |
| clip-h14-378-dfn5b | fp16 | 107.4 / 104.8 | 1.03 | 0.13 / 1.59 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.4 / 104.9 | 2.21 / 0.73 | 0.13 / 1.59 | 0.064 (0.025 % off) | 0.064 (0.025 % off) | slower |
| clip-h14-378-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | 0.69 / 1.66 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.6 / 125.4 | 2.11 / 0.74 | 0.69 / 1.66 | 1.8e-05 | 1.8e-05 |  |
| clip-h14-dfn5b | fp16 | 107.3 / 104.7 | 1.03 | 0.12 / 1.57 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.4 / 105.0 | 3.29 / 0.73 | 0.12 / 1.57 | 0.032 (0.034 % off) | 0.032 (0.034 % off) | slower |
| clip-h14-dfn5b | fp32 | 127.4 / 125.0 | 1.02 | 0.25 / 1.66 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.8 / 125.3 | 1.59 / 0.73 | 0.25 / 1.66 | 3.6e-05 | 3.6e-05 |  |
| clip-h14-laion2b | fp16 | 107.5 / 104.9 | 1.02 | 0.12 / 1.55 | 1241 / 1277 | 0.97 | 1203 / 1203 | 107.8 / 105.2 | 1.82 / 0.74 | 0.12 / 1.56 | 0.017 (0.0021 % off) | 0.017 (0.0021 % off) | slower |
| clip-h14-laion2b | fp32 | 127.4 / 124.8 | 1.02 | 0.80 / 1.61 | 2469 / 2524 | 0.98 | 2407 / 2407 | 127.8 / 125.1 | 1.61 / 0.73 | 0.81 / 1.62 | 8.6e-06 | 8.6e-06 | slower |
| clip-l14-336-openai | fp16 | 55.0 / 53.9 | 1.02 | 0.07 / 1.15 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 1.34 / 0.71 | 0.07 / 1.15 | 0.022 (0.0023 % off) | 0.022 (0.0023 % off) | slower |
| clip-l14-336-openai | fp32 | 63.9 / 62.4 | 1.02 | 0.10 / 1.15 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.86 / 0.70 | 0.10 / 1.15 | 1.5e-05 | 1.5e-05 | slower |
| clip-l14-datacompxl | fp16 | 55.1 / 54.0 | 1.02 | 0.07 / 1.10 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 0.98 / 0.71 | 0.07 / 1.11 | 0.0063 | 0.0063 |  |
| clip-l14-datacompxl | fp32 | 64.0 / 62.3 | 1.03 | 0.10 / 1.14 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.7 | 0.86 / 0.71 | 0.10 / 1.14 | 1e-05 | 1e-05 | slower |
| clip-l14-dfn2b | fp16 | 56.1 / 54.1 | 1.04 | 0.07 / 1.12 | 618 / 658 | 0.94 | 578 / 578 | 56.3 / 54.2 | 1.00 / 0.72 | 0.08 / 1.12 | 0.05 (0.18 % off) | 0.05 (0.18 % off) | slower |
| clip-l14-dfn2b | fp32 | 63.9 / 62.3 | 1.03 | 0.10 / 1.14 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.86 / 0.72 | 0.10 / 1.14 | 7.5e-05 | 7.5e-05 | slower |
| clip-l14-laion2b | fp16 | 55.0 / 54.0 | 1.02 | 0.08 / 1.11 | 608 / 658 | 0.92 | 578 / 578 | 55.4 / 54.3 | 0.99 / 0.71 | 0.08 / 1.11 | 0.013 (0.00076 % off) | 0.013 (0.00076 % off) |  |
| clip-l14-laion2b | fp32 | 64.0 / 62.3 | 1.03 | 0.10 / 1.11 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.2 / 62.6 | 0.88 / 0.73 | 0.10 / 1.11 | 2.6e-05 | 2.6e-05 | slower |
| clip-l14-openai | fp16 | 55.1 / 53.9 | 1.02 | 0.08 / 1.13 | 608 / 658 | 0.92 | 578 / 578 | 55.3 / 54.2 | 0.99 / 0.72 | 0.08 / 1.13 | 0.0046 | 0.0046 | slower |
| clip-l14-openai | fp32 | 63.9 / 62.1 | 1.03 | 0.10 / 1.13 | 1205 / 1283 | 0.94 | 1156 / 1157 | 64.3 / 62.4 | 0.85 / 0.73 | 0.11 / 1.14 | 4.3e-06 | 4.3e-06 | slower |
| coatnet_0_rw_224.sw_in1k | fp16 | 7.6 / 7.7 | 0.98 | 0.03 / 0.92 | 142 / 168 | 0.85 | 58 / 58 | 8.2 / 8.0 | 0.82 / 0.74 | 0.04 / 0.93 | 0.0012 | 0.0012 |  |
| coatnet_0_rw_224.sw_in1k | fp32 | 8.4 / 8.7 | 0.97 | 0.04 / 0.84 | 245 / 284 | 0.86 | 116 / 117 | 9.0 / 9.5 | 0.81 / 0.73 | 0.04 / 0.84 | 5.8e-07 | 5.8e-07 |  |
| coatnet_1_rw_224.sw_in1k | fp16 | 12.8 / 12.7 | 1.01 | 0.04 / 1.30 | 226 / 253 | 0.89 | 92 / 92 | 13.3 / 13.4 | 0.88 / 0.71 | 0.04 / 1.30 | 0.00052 | 0.00052 |  |
| coatnet_1_rw_224.sw_in1k | fp32 | 14.3 / 15.3 | 0.93 | 0.04 / 1.24 | 397 / 340 | 1.17 | 183 / 184 | 14.7 / 15.9 | 0.80 / 0.72 | 0.04 / 1.24 | 3.8e-07 | 3.8e-07 | heavier |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.7 / 19.6 | 0.90 | 0.04 / 1.30 | 307 / 345 | 0.89 | 157 / 157 | 18.3 / 20.2 | 0.90 / 0.70 | 0.04 / 1.30 | 0.0027 | 0.0027 |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.7 / 24.3 | 0.81 | 0.04 / 1.24 | 552 / 504 | 1.09 | 314 / 314 | 20.2 / 24.8 | 0.84 / 0.72 | 0.04 / 1.24 | 2.9e-06 | 2.9e-06 | heavier |
| coatnet_nano_rw_224.sw_in1k | fp16 | 6.1 / 6.3 | 0.96 | 0.03 / 0.88 | 105 / 132 | 0.80 | 32 / 32 | 6.6 / 6.9 | 0.77 / 0.71 | 0.03 / 0.88 | 0.0044 | 0.0044 |  |
| coatnet_nano_rw_224.sw_in1k | fp32 | 6.5 / 7.3 | 0.88 | 0.03 / 0.82 | 172 / 213 | 0.81 | 65 / 65 | 7.0 / 7.8 | 0.75 / 0.70 | 0.03 / 0.82 | 8.3e-07 | 8.3e-07 |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp16 | 12.8 / 12.7 | 1.01 | 0.04 / 1.51 | 215 / 248 | 0.87 | 92 / 92 | 13.4 / 13.4 | 0.87 / 0.73 | 0.04 / 1.51 | 0.0051 | 0.0051 |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp32 | 14.2 / 15.4 | 0.92 | 0.04 / 1.46 | 397 / 340 | 1.17 | 183 / 184 | 14.7 / 16.0 | 0.82 / 0.71 | 0.04 / 1.46 | 4.7e-07 | 4.7e-07 | heavier |
| coatnet_rmlp_1_rw_224.sw_in1k | fp16 | 12.7 / 12.5 | 1.02 | 0.03 / 1.58 | 223 / 245 | 0.91 | 92 / 92 | 13.2 / 13.0 | 0.87 / 0.73 | 0.04 / 1.59 | 0.0012 | 0.0012 |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp32 | 14.0 / 14.9 | 0.94 | 0.04 / 1.44 | 390 / 353 | 1.10 | 183 / 184 | 14.7 / 15.5 | 0.81 / 0.72 | 0.04 / 1.45 | 3.9e-07 | 3.9e-07 | heavier |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.7 / 19.9 | 0.89 | 0.04 / 1.58 | 307 / 345 | 0.89 | 157 / 157 | 18.2 / 20.4 | 0.93 / 0.71 | 0.04 / 1.58 | 0.0041 | 0.0041 |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.4 / 24.5 | 0.79 | 0.04 / 1.52 | 552 / 504 | 1.10 | 314 / 314 | 20.3 / 25.2 | 0.90 / 0.71 | 0.04 / 1.53 | 6.7e-07 | 6.7e-07 | heavier |
| coatnet_rmlp_2_rw_224.sw_in1k | fp16 | 17.7 / 19.7 | 0.90 | 0.04 / 1.62 | 307 / 345 | 0.89 | 157 / 157 | 18.2 / 20.3 | 0.92 / 0.74 | 0.04 / 1.62 | 0.0015 | 0.0015 |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp32 | 19.7 / 24.6 | 0.80 | 0.04 / 1.50 | 552 / 504 | 1.10 | 314 / 314 | 20.3 / 25.3 | 0.88 / 0.72 | 0.04 / 1.50 | 3.5e-07 | 3.5e-07 | heavier |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp16 | 43.7 / 48.3 | 0.90 | 0.06 / 1.66 | 471 / 475 | 0.99 | 279 / 280 | 43.7 / 48.4 | 0.97 / 0.77 | 0.06 / 1.66 | 0.0014 | 0.0014 |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp32 | 51.0 / 60.6 | 0.84 | 0.07 / 1.57 | 891 / 826 | 1.08 | 558 / 559 | 51.3 / 60.9 | 0.93 / 0.72 | 0.07 / 1.57 | 5.3e-07 | 5.3e-07 | heavier |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp16 | 6.1 / 6.6 | 0.93 | 0.03 / 0.98 | 109 / 140 | 0.77 | 32 / 32 | 6.7 / 7.1 | 0.78 / 0.71 | 0.03 / 0.98 | 0.0017 | 0.0017 |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp32 | 6.6 / 7.7 | 0.86 | 0.03 / 0.97 | 179 / 235 | 0.76 | 65 / 65 | 7.1 / 8.5 | 0.76 / 0.70 | 0.03 / 0.98 | 6.1e-07 | 6.1e-07 |  |
| coex-sceneflow | fp16 | 197.2 / 409.9 | 0.48 | 0.24 / 2.21 | 550 / 1669 | 0.33 | 11 / 6 | 202.2 / 423.0 | 0.08 / 0.07 | 0.25 / 2.22 | 0.31 (0.1 % off) | 0.034 (0.0055 % off) |  |
| coex-sceneflow | fp32 | 258.0 / 518.0 | 0.50 | 0.29 / 2.27 | 1020 / 1943 | 0.53 | 11 / 11 | 264.1 / 515.7 | 0.08 / 0.06 | 0.30 / 2.28 | 2.8e-06 | 3e-06 |  |
| convnext_base.fb_in1k | fp16 | 15.0 / 14.9 | 1.00 | 0.04 / 1.05 | 252 / 278 | 0.91 | 170 / 169 | 15.6 / 15.5 | 0.95 / 0.71 | 0.04 / 1.05 | 0.00068 | 0.00068 |  |
| convnext_base.fb_in1k | fp32 | 17.3 / 17.2 | 1.01 | 0.05 / 1.01 | 475 / 448 | 1.06 | 338 / 338 | 17.9 / 17.8 | 0.86 / 0.71 | 0.05 / 1.02 | 2.9e-07 | 2.9e-07 | heavier |
| convnext_base.fb_in22k_ft_in1k | fp16 | 15.3 / 14.9 | 1.03 | 0.04 / 1.03 | 252 / 278 | 0.91 | 170 / 169 | 15.9 / 15.5 | 0.90 / 0.70 | 0.04 / 1.04 | 0.0012 | 0.0012 | slower |
| convnext_base.fb_in22k_ft_in1k | fp32 | 17.4 / 17.2 | 1.01 | 0.05 / 1.00 | 475 / 448 | 1.06 | 338 / 338 | 18.0 / 17.8 | 0.83 / 0.73 | 0.05 / 1.01 | 6.6e-07 | 6.6e-07 | heavier |
| convnext_base.fb_in22k_ft_in1k_384 | fp16 | 32.5 / 31.0 | 1.05 | 0.05 / 1.05 | 313 / 310 | 1.01 | 170 / 169 | 33.3 / 31.7 | 0.90 / 0.71 | 0.05 / 1.05 | 0.0028 | 0.0028 | slower |
| convnext_base.fb_in22k_ft_in1k_384 | fp32 | 36.8 / 37.0 | 1.00 | 0.07 / 1.02 | 592 / 464 | 1.28 | 338 / 338 | 37.5 / 37.8 | 0.84 / 0.70 | 0.07 / 1.02 | 4.2e-07 | 4.2e-07 | heavier |
| convnext_large.fb_in1k | fp16 | 27.7 / 27.8 | 1.00 | 0.05 / 1.05 | 465 / 478 | 0.97 | 378 / 378 | 28.3 / 28.5 | 1.11 / 0.70 | 0.05 / 1.06 | 0.0019 | 0.0019 |  |
| convnext_large.fb_in1k | fp32 | 31.9 / 33.2 | 0.96 | 0.07 / 1.05 | 902 / 866 | 1.04 | 755 / 755 | 32.4 / 33.7 | 1.04 / 0.70 | 0.07 / 1.05 | 2.3e-07 | 2.3e-07 | heavier |
| convnext_large.fb_in22k_ft_in1k | fp16 | 27.8 / 27.8 | 1.00 | 0.05 / 1.06 | 465 / 478 | 0.97 | 378 / 378 | 28.3 / 28.4 | 1.09 / 0.72 | 0.05 / 1.07 | 0.0017 | 0.0017 |  |
| convnext_large.fb_in22k_ft_in1k | fp32 | 31.9 / 33.1 | 0.96 | 0.06 / 1.05 | 902 / 866 | 1.04 | 755 / 755 | 32.5 / 33.7 | 1.03 / 0.71 | 0.06 / 1.06 | 4.8e-07 | 4.8e-07 | heavier |
| convnext_large.fb_in22k_ft_in1k_384 | fp16 | 61.0 / 60.6 | 1.01 | 0.08 / 1.10 | 501 / 495 | 1.01 | 378 / 378 | 61.8 / 61.8 | 1.08 / 0.72 | 0.08 / 1.10 | 0.0016 | 0.0016 |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 69.3 / 70.8 | 0.98 | 0.10 / 1.09 | 991 / 892 | 1.11 | 755 / 755 | 70.0 / 71.6 | 1.03 / 0.70 | 0.10 / 1.09 | 4.3e-07 | 4.3e-07 | heavier |
| convnext_small.fb_in1k | fp16 | 11.5 / 10.6 | 1.09 | 0.03 / 1.01 | 157 / 185 | 0.85 | 96 / 96 | 12.0 / 11.2 | 0.84 / 0.69 | 0.03 / 1.01 | 0.0013 | 0.0013 | slower |
| convnext_small.fb_in1k | fp32 | 12.9 / 12.5 | 1.03 | 0.05 / 0.97 | 293 / 302 | 0.97 | 192 / 192 | 13.4 / 13.0 | 0.81 / 0.69 | 0.05 / 0.97 | 3.9e-07 | 3.9e-07 | slower |
| convnext_small.fb_in22k_ft_in1k | fp16 | 11.4 / 10.6 | 1.08 | 0.03 / 1.01 | 157 / 185 | 0.85 | 96 / 96 | 12.1 / 11.2 | 0.86 / 0.71 | 0.04 / 1.02 | 0.0014 | 0.0014 | slower |
| convnext_small.fb_in22k_ft_in1k | fp32 | 12.9 / 12.4 | 1.04 | 0.04 / 0.97 | 293 / 302 | 0.97 | 192 / 192 | 13.4 / 12.9 | 0.79 / 0.70 | 0.04 / 0.97 | 5e-07 | 5e-07 | slower |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 23.8 / 21.1 | 1.13 | 0.04 / 1.09 | 242 / 233 | 1.04 | 96 / 96 | 24.6 / 21.9 | 0.87 / 0.69 | 0.05 / 1.10 | 0.0017 | 0.0017 | slower heavier |
| convnext_small.fb_in22k_ft_in1k_384 | fp32 | 26.4 / 24.8 | 1.07 | 0.06 / 0.98 | 443 / 308 | 1.44 | 192 / 192 | 27.1 / 25.4 | 0.79 / 0.69 | 0.06 / 0.99 | 2.7e-07 | 2.7e-07 | slower heavier |
| convnext_tiny.fb_in1k | fp16 | 6.8 / 5.9 | 1.15 | 0.02 / 0.67 | 89 / 108 | 0.82 | 55 / 55 | 7.3 / 6.6 | 0.79 / 0.70 | 0.03 / 0.67 | 0.0016 | 0.0016 | slower |
| convnext_tiny.fb_in1k | fp32 | 7.5 / 6.7 | 1.13 | 0.04 / 0.71 | 168 / 206 | 0.82 | 109 / 109 | 8.1 / 7.4 | 0.74 / 0.69 | 0.04 / 0.71 | 5.3e-07 | 5.3e-07 | slower |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.8 / 5.9 | 1.15 | 0.02 / 0.68 | 89 / 108 | 0.82 | 55 / 55 | 7.3 / 6.5 | 0.80 / 0.69 | 0.03 / 0.68 | 0.0012 | 0.0012 | slower |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 7.4 / 6.7 | 1.11 | 0.04 / 0.65 | 168 / 206 | 0.82 | 109 / 109 | 8.0 / 7.2 | 0.75 / 0.70 | 0.04 / 0.66 | 4.3e-07 | 4.3e-07 | slower |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 14.2 / 12.0 | 1.19 | 0.03 / 0.69 | 151 / 161 | 0.93 | 55 / 55 | 14.9 / 12.6 | 0.81 / 0.69 | 0.03 / 0.69 | 0.002 | 0.002 | slower |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 15.8 / 14.1 | 1.12 | 0.05 / 0.65 | 279 / 221 | 1.26 | 109 / 109 | 16.5 / 14.8 | 0.74 / 0.70 | 0.05 / 0.66 | 9.8e-07 | 9.8e-07 | slower heavier |
| convnextv2_base.fcmae_ft_in22k_in1k | fp16 | 17.7 / 17.4 | 1.01 | 0.04 / 1.40 | 358 / 318 | 1.13 | 170 / 170 | 18.3 / 18.1 | 0.94 / 0.76 | 0.04 / 1.40 | 0.0011 | 0.0011 | heavier |
| convnextv2_base.fcmae_ft_in22k_in1k | fp32 | 20.1 / 19.9 | 1.01 | 0.05 / 1.29 | 558 / 456 | 1.22 | 339 / 339 | 20.6 / 20.5 | 0.83 / 0.71 | 0.06 / 1.30 | 6.4e-07 | 6.4e-07 | heavier |
| da3-giant-1.1 | fp16 | 598.3 / 580.7 | 1.03 | 0.66 / 5.45 | 2729 / 2889 | 0.94 | 2321 / 2364 | 605.9 / 587.7 | 4.34 / 0.86 | 0.68 / 5.47 | 0.0057 | 0.0056 | slower |
| da3-giant-1.1 | fp32 | 732.9 / 721.4 | 1.02 | 0.93 / 11.83 | 5376 / 5546 | 0.97 | 4571 / 4638 | 743.8 / 732.9 | 5.76 / 0.76 | 0.96 / 11.86 | 4.6e-06 | 4.2e-06 |  |
| da3-large-1.1 | fp16 | 214.8 / 209.0 | 1.03 | 0.25 / 3.26 | 1096 / 1096 | 1.00 | 690 / 750 | 222.7 / 216.3 | 0.29 / 0.48 | 0.27 / 3.27 | 0.0036 | 0.0032 | slower |
| da3-large-1.1 | fp32 | 259.1 / 249.3 | 1.04 | 0.32 / 3.25 | 2148 / 2321 | 0.93 | 1346 / 1411 | 266.7 / 256.6 | 0.12 / 0.50 | 0.34 / 3.27 | 2.1e-06 | 2.1e-06 | slower |
| da3metric-large | fp16 | 207.0 / 197.0 | 1.05 | 0.23 / 1.91 | 1071 / 1102 | 0.97 | 636 / 639 | 224.1 / 208.5 | 0.29 / 0.50 | 0.26 / 1.93 | 0.018 (0.017 % off) | 0.017 (0.0054 % off) | slower |
| da3metric-large | fp32 | 248.9 / 235.7 | 1.06 | 0.29 / 1.92 | 2082 / 2195 | 0.95 | 1273 / 1278 | 266.3 / 254.2 | 0.12 / 0.48 | 0.31 / 1.95 | 2.7e-05 | 2.7e-05 | slower |
| da3mono-large | fp16 | 206.8 / 196.7 | 1.05 | 0.23 / 1.91 | 1071 / 1102 | 0.97 | 636 / 639 | 224.3 / 207.3 | 0.27 / 0.49 | 0.26 / 1.93 | 0.023 (0.0027 % off) | 1 (0.0013 % off) | slower |
| da3mono-large | fp32 | 248.8 / 235.7 | 1.06 | 0.29 / 1.92 | 2082 / 2195 | 0.95 | 1273 / 1278 | 265.6 / 252.3 | 0.12 / 0.48 | 0.31 / 1.95 | 1.3e-05 | 1.3e-05 | slower |
| da3nested-giant-large-1.1 | fp16 | 802.3 / 776.6 | 1.03 | 0.86 / 7.92 | 3392 / 3528 | 0.96 | 2958 / 3003 | 829.6 / 791.7 | 5.69 / 0.80 | 0.90 / 7.96 | 0.0029 | 0.017 (0.0041 % off) | slower |
| da3nested-giant-large-1.1 | fp32 | 982.5 / 974.7 | 1.01 | 1.41 / 11.21 | 6652 / 6833 | 0.97 | 5844 / 5916 | 1010.0 / 1003.7 | 7.42 / 0.79 | 1.48 / 11.26 | 3.7e-06 | 2.7e-05 |  |
| davit_base.msft_in1k | fp16 | 19.5 / 18.1 | 1.08 | 0.04 / 1.85 | 300 / 255 | 1.18 | 168 / 168 | 20.2 / 18.8 | 0.88 / 0.69 | 0.04 / 1.85 | 0.00054 | 0.00054 | slower heavier |
| davit_base.msft_in1k | fp32 | 21.6 / 21.7 | 0.99 | 0.05 / 1.81 | 562 / 425 | 1.32 | 336 / 336 | 22.1 / 22.3 | 0.82 / 0.70 | 0.06 / 1.81 | 3.9e-07 | 3.9e-07 | heavier |
| davit_small.msft_in1k | fp16 | 15.6 / 13.8 | 1.13 | 0.04 / 1.85 | 201 / 185 | 1.09 | 95 / 95 | 16.2 / 14.4 | 0.86 / 0.70 | 0.04 / 1.85 | 0.0019 | 0.0019 | slower heavier |
| davit_small.msft_in1k | fp32 | 17.1 / 16.3 | 1.05 | 0.05 / 1.81 | 389 / 279 | 1.39 | 190 / 190 | 17.7 / 16.9 | 0.77 / 0.71 | 0.06 / 1.81 | 5.6e-07 | 5.6e-07 | slower heavier |
| davit_tiny.msft_in1k | fp16 | 9.3 / 7.5 | 1.24 | 0.03 / 1.15 | 127 / 129 | 0.98 | 54 / 54 | 9.8 / 8.1 | 0.78 / 0.71 | 0.03 / 1.15 | 0.00093 | 0.00093 | slower |
| davit_tiny.msft_in1k | fp32 | 10.0 / 9.2 | 1.09 | 0.04 / 1.12 | 229 / 187 | 1.23 | 108 / 108 | 10.6 / 9.7 | 0.75 / 0.69 | 0.05 / 1.12 | 4.3e-07 | 4.3e-07 | slower heavier |
| deeplabv3-r101-d8-ade20k | fp16 | 365.2 / 378.3 | 0.97 | 0.41 / 1.66 | 682 / 582 | 1.17 | 162 / 162 | 399.0 / 410.3 | 0.21 / 0.14 | 0.45 / 1.70 | 0.0007 | 0 | heavier |
| deeplabv3-r101-d8-ade20k | fp32 | 543.4 / 529.3 | 1.03 | 0.56 / 1.78 | 1224 / 1312 | 0.93 | 323 / 323 | 570.9 / 561.6 | 0.21 / 0.14 | 0.60 / 1.83 | 3.4e-07 | 0 | slower |
| deeplabv3plus-r101-d8-ade20k | fp16 | 278.2 / 288.2 | 0.97 | 0.33 / 1.60 | 860 / 674 | 1.28 | 115 / 115 | 317.9 / 328.8 | 0.18 / 0.12 | 0.37 / 1.64 | 0.0011 | 0 | heavier |
| deeplabv3plus-r101-d8-ade20k | fp32 | 354.0 / 379.1 | 0.93 | 0.41 / 1.66 | 1392 / 1209 | 1.15 | 230 / 230 | 396.2 / 419.9 | 0.18 / 0.13 | 0.45 / 1.70 | 4.2e-07 | 0 | heavier |
| deim-l-coco | fp16 | 40.8 / 46.2 | 0.88 | 0.07 / 2.72 | 249 / 274 | 0.91 | 63 / 62 | 48.3 / 54.1 | 0.34 / 0.16 | 0.08 / 2.73 | 0.89 (48 % off) | 0.0013 |  |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 0.09 / 2.46 | 483 / 393 | 1.23 | 118 / 118 | 54.5 / 62.7 | 0.22 / 0.16 | 0.10 / 2.47 | 6.9e-06 | 1.1e-06 | heavier |
| deim-m-coco | fp16 | 27.4 / 33.6 | 0.82 | 0.09 / 2.44 | 197 / 232 | 0.85 | 40 / 40 | 35.5 / 42.5 | 0.27 / 0.13 | 0.10 / 2.45 | 0.89 (56 % off) | 0.0036 |  |
| deim-m-coco | fp32 | 31.7 / 40.2 | 0.79 | 0.09 / 2.26 | 381 / 376 | 1.01 | 74 / 74 | 39.4 / 48.3 | 0.18 / 0.13 | 0.10 / 2.28 | 0.49 (0.95 % off) | 2.6e-06 |  |
| deim-n-coco | fp16 | 9.1 / 11.9 | 0.77 | 0.06 / 1.94 | 58 / 139 | 0.41 | 8 / 8 | 17.2 / 19.9 | 0.18 / 0.09 | 0.07 / 1.96 | 0.96 (59 % off) | 0.71 (2.5 % off) |  |
| deim-n-coco | fp32 | 10.3 / 13.5 | 0.76 | 0.07 / 1.83 | 106 / 175 | 0.60 | 15 / 15 | 18.4 / 20.8 | 0.11 / 0.09 | 0.08 / 1.85 | 1.7e-05 | 1.5e-06 |  |
| deim-s-coco | fp16 | 16.3 / 20.2 | 0.80 | 0.07 / 2.10 | 129 / 193 | 0.67 | 22 / 22 | 24.1 / 28.1 | 0.21 / 0.10 | 0.09 / 2.11 | 0.97 (49 % off) | empty (3 keys differ) |  |
| deim-s-coco | fp32 | 18.4 / 23.9 | 0.77 | 0.08 / 2.02 | 191 / 206 | 0.93 | 40 / 40 | 26.1 / 31.9 | 0.14 / 0.10 | 0.09 / 2.04 | 4.2e-06 | 1.7e-06 |  |
| deim-x-coco | fp16 | 69.8 / 75.9 | 0.92 | 0.10 / 3.00 | 376 / 380 | 0.99 | 122 / 122 | 78.2 / 84.2 | 0.49 / 0.22 | 0.11 / 3.01 | 0.84 (42 % off) | 0.0068 |  |
| deim-x-coco | fp32 | 80.3 / 90.3 | 0.89 | 0.12 / 2.77 | 740 / 710 | 1.04 | 236 / 236 | 88.4 / 98.6 | 0.32 / 0.22 | 0.13 / 2.78 | 0.37 (0.59 % off) | 7.7e-07 | heavier |
| deit_base_patch16_224.fb_in1k | fp16 | 14.0 / 13.7 | 1.02 | 0.04 / 0.65 | 242 / 232 | 1.04 | 165 / 165 | 14.5 / 14.3 | 0.05 / 0.69 | 0.04 / 0.65 | 0.00093 | 0.00093 | slower heavier |
| deit_base_patch16_224.fb_in1k | fp32 | 16.1 / 16.1 | 1.00 | 0.05 / 0.63 | 454 / 420 | 1.08 | 330 / 330 | 16.7 / 16.7 | 0.06 / 0.68 | 0.05 / 0.63 | 6.8e-07 | 6.8e-07 | heavier |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 0.03 / 1.38 | 81 / 74 | 1.11 | 16 / 16 | 8.6 / 11.3 | 0.68 / 0.58 | 0.03 / 1.39 | 0.0017 | 0.0017 | heavier |
| densenet121.tv_in1k | fp32 | 9.6 / 12.4 | 0.77 | 0.03 / 1.31 | 148 / 142 | 1.04 | 30 / 31 | 10.1 / 12.9 | 0.64 / 0.58 | 0.03 / 1.31 | 5.4e-07 | 5.4e-07 | heavier |
| depth-pro | fp16 | 4168.0 / 4341.7 | 0.96 | 4.27 / 9.37 | 4508 / 4609 | 0.98 | 1809 / 1815 | 4170.1 / 4666.0 | 3.88 / 3.42 | 4.28 / 9.38 | 0.0017 | 0.0011 |  |
| depth-pro | fp32 | 5497.0 / 6831.3 | 0.80 | 5.96 / 10.71 | 9014 / 9191 | 0.98 | 3616 / 3630 | 5552.0 / 6573.6 | 4.63 / 3.41 | 5.99 / 10.74 | 1.5e-06 | 1.3e-06 |  |
| dfine-l-coco | fp16 | 40.9 / 46.5 | 0.88 | 0.07 / 2.62 | 249 / 274 | 0.91 | 63 / 62 | 48.1 / 54.1 | 0.29 / 0.10 | 0.08 / 2.63 | 0.97 (42 % off) | 0.97 (23 % off) |  |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 0.09 / 2.44 | 488 / 393 | 1.24 | 118 / 118 | 54.4 / 62.6 | 0.17 / 0.10 | 0.10 / 2.45 | 1e-05 | 3.4e-06 | heavier |
| dfine-m-coco | fp16 | 27.2 / 34.0 | 0.80 | 0.08 / 2.39 | 197 / 232 | 0.85 | 40 / 40 | 34.7 / 41.8 | 0.22 / 0.08 | 0.09 / 2.40 | 0.95 (61 % off) | 0.96 (46 % off) |  |
| dfine-m-coco | fp32 | 31.8 / 40.2 | 0.79 | 0.09 / 2.25 | 381 / 376 | 1.01 | 74 / 74 | 39.6 / 48.8 | 0.13 / 0.08 | 0.10 / 2.26 | 0.63 (0.6 % off) | 1.4e-05 |  |
| dfine-n-coco | fp16 | 9.0 / 11.8 | 0.76 | 0.06 / 1.93 | 58 / 139 | 0.41 | 8 / 8 | 16.9 / 19.6 | 0.14 / 0.06 | 0.08 / 1.95 | 0.95 (53 % off) | 0.97 (30 % off) |  |
| dfine-n-coco | fp32 | 10.3 / 13.7 | 0.75 | 0.07 / 1.85 | 106 / 175 | 0.60 | 15 / 15 | 17.6 / 22.1 | 0.08 / 0.06 | 0.08 / 1.86 | 6e-06 | 9e-07 |  |
| dfine-s-coco | fp16 | 16.2 / 20.2 | 0.80 | 0.07 / 2.10 | 129 / 193 | 0.67 | 22 / 22 | 24.1 / 28.4 | 0.17 / 0.07 | 0.09 / 2.12 | 0.96 (57 % off) | 1 (38 % off) |  |
| dfine-s-coco | fp32 | 18.4 / 23.7 | 0.77 | 0.08 / 1.99 | 191 / 206 | 0.93 | 40 / 40 | 26.4 / 31.7 | 0.10 / 0.06 | 0.09 / 2.00 | 8.8e-06 | 2e-05 |  |
| dfine-x-coco | fp16 | 69.8 / 76.0 | 0.92 | 0.10 / 2.92 | 376 / 380 | 0.99 | 122 / 122 | 77.8 / 83.7 | 0.39 / 0.15 | 0.11 / 2.94 | 1.1 (52 % off) | empty (3 keys differ) |  |
| dfine-x-coco | fp32 | 80.4 / 90.3 | 0.89 | 0.12 / 2.75 | 740 / 710 | 1.04 | 236 / 236 | 88.5 / 98.0 | 0.24 / 0.15 | 0.13 / 2.77 | 0.00025 | 4.9e-06 | heavier |
| disk-lightglue | fp16 | 478.8 / 466.5 | 1.03 | 1.51 / 1.57 | 990 / 965 | 1.03 | 50 / 5 | 591.3 / 618.7 | 0.07 / 0.05 | 1.68 / 1.75 | 0 | 7.4e-06 | slower heavier |
| disk-lightglue | fp32 | 558.2 / 549.9 | 1.02 | 1.62 / 1.68 | 967 / 920 | 1.05 | 50 / 4 | 675.9 / 702.0 | 0.07 / 0.05 | 1.82 / 1.86 | 0 | 7.7e-06 | heavier |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1.64 / 7.23 | 1647 / 1209 | 1.36 | 142 / 182 | 1658.0 / 1935.5 | 0.18 / 0.09 | 1.65 / 7.24 | 0.33 (0.15 % off) | 0.33 (0.027 % off) | heavier |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2.06 / 8.03 | 2901 / 2261 | 1.28 | 229 / 314 | 2053.7 / 2758.0 | 0.18 / 0.09 | 2.07 / 8.04 | 0.02 (0.00091 % off) | 0.016 (0.00054 % off) | heavier |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1.64 / 7.23 | 1647 / 1208 | 1.36 | 142 / 181 | 1667.1 / 1933.3 | 0.18 / 0.09 | 1.65 / 7.24 | 0.019 (0.0068 % off) | 0.017 (0.0013 % off) | heavier |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2.07 / 7.81 | 2901 / 2261 | 1.28 | 229 / 314 | 2054.8 / 2760.3 | 0.18 / 0.09 | 2.08 / 7.82 | 1.2e-05 | 1.2e-05 | heavier |
| dwpose-l | fp16 | 56.2 / 66.0 | 0.85 | 0.07 / 1.32 | 332 / 290 | 1.15 | 81 / 78 | 65.3 / 74.5 | 0.20 / 0.08 | 0.08 / 1.34 | - | 0.44 (0.099 % off) | heavier |
| dwpose-l | fp32 | 62.2 / 79.0 | 0.79 | 0.09 / 1.30 | 587 / 454 | 1.29 | 128 / 128 | 71.2 / 87.5 | 0.15 / 0.08 | 0.10 / 1.31 | - | 3.3e-06 | heavier |
| dwpose-l-384 | fp16 | 109.3 / 132.1 | 0.83 | 0.13 / 1.47 | 409 / 376 | 1.09 | 81 / 78 | 127.1 / 147.4 | 0.21 / 0.08 | 0.15 / 1.49 | - | 0.57 (0.21 % off) | heavier |
| dwpose-l-384 | fp32 | 122.6 / 163.0 | 0.75 | 0.15 / 1.42 | 765 / 518 | 1.48 | 128 / 128 | 140.4 / 178.8 | 0.15 / 0.09 | 0.17 / 1.44 | - | 4e-06 | heavier |
| dwpose-m | fp16 | 40.1 / 42.6 | 0.94 | 0.05 / 1.21 | 243 / 237 | 1.03 | 47 / 45 | 48.7 / 50.4 | 0.14 / 0.06 | 0.06 / 1.22 | - | 0.42 (0.18 % off) | heavier |
| dwpose-m | fp32 | 44.2 / 50.3 | 0.88 | 0.07 / 1.21 | 400 / 429 | 0.93 | 69 / 69 | 53.0 / 58.3 | 0.10 / 0.06 | 0.08 / 1.22 | - | 2.1e-06 |  |
| dwpose-s | fp16 | 17.8 / 21.2 | 0.84 | 0.03 / 1.11 | 200 / 201 | 0.99 | 24 / 24 | 26.6 / 28.9 | 0.10 / 0.04 | 0.04 / 1.12 | - | 0.38 (0.017 % off) |  |
| dwpose-s | fp32 | 18.8 / 24.2 | 0.78 | 0.04 / 1.10 | 286 / 256 | 1.12 | 32 / 32 | 27.3 / 32.2 | 0.07 / 0.04 | 0.06 / 1.11 | - | 2.5e-06 | heavier |
| dwpose-t | fp16 | 16.1 / 17.9 | 0.90 | 0.03 / 1.06 | 161 / 198 | 0.81 | 17 / 17 | 24.8 / 25.9 | 0.09 / 0.04 | 0.04 / 1.08 | - | 0.57 (0.23 % off) |  |
| dwpose-t | fp32 | 17.1 / 20.7 | 0.83 | 0.04 / 1.05 | 224 / 283 | 0.79 | 21 / 21 | 25.8 / 28.5 | 0.06 / 0.04 | 0.06 / 1.07 | - | 3.1e-06 |  |
| efficientformer_l1.snap_dist_in1k | fp16 | 3.4 / 3.8 | 0.89 | 0.02 / 0.72 | 45 / 59 | 0.76 | 23 / 24 | 3.9 / 4.4 | 0.76 / 0.68 | 0.02 / 0.72 | 0.0056 | 0.0056 |  |
| efficientformer_l1.snap_dist_in1k | fp32 | 3.7 / 4.1 | 0.90 | 0.03 / 0.69 | 83 / 96 | 0.87 | 45 / 47 | 4.1 / 4.8 | 0.73 / 0.69 | 0.04 / 0.69 | 1.3e-06 | 1.3e-06 |  |
| efficientformer_l3.snap_dist_in1k | fp16 | 6.1 / 7.4 | 0.83 | 0.03 / 1.17 | 104 / 129 | 0.81 | 59 / 60 | 6.6 / 8.1 | 0.82 / 0.71 | 0.03 / 1.17 | 0.0021 | 0.0021 |  |
| efficientformer_l3.snap_dist_in1k | fp32 | 7.0 / 8.5 | 0.82 | 0.04 / 1.15 | 201 / 218 | 0.92 | 118 / 120 | 7.6 / 9.8 | 0.76 / 0.69 | 0.04 / 1.15 | 7.9e-07 | 7.9e-07 |  |
| efficientformer_l7.snap_dist_in1k | fp16 | 12.6 / 14.0 | 0.90 | 0.04 / 1.65 | 238 / 288 | 0.83 | 156 / 157 | 13.2 / 14.9 | 0.94 / 0.72 | 0.04 / 1.65 | 0.006 | 0.006 |  |
| efficientformer_l7.snap_dist_in1k | fp32 | 14.1 / 16.1 | 0.87 | 0.05 / 1.66 | 461 / 442 | 1.04 | 311 / 314 | 14.7 / 17.1 | 0.85 / 0.71 | 0.05 / 1.66 | 6.9e-07 | 6.9e-07 | heavier |
| efficientformerv2_l.snap_dist_in1k | fp16 | 10.9 / 9.1 | 1.20 | 0.04 / 2.33 | 178 / 99 | 1.80 | 50 / 51 | 11.4 / 9.8 | 0.92 / 0.75 | 0.04 / 2.33 | 0.0028 | 0.0028 | slower heavier |
| efficientformerv2_l.snap_dist_in1k | fp32 | 10.0 / 10.2 | 0.98 | 0.05 / 2.30 | 209 / 193 | 1.08 | 100 / 102 | 10.5 / 10.8 | 0.83 / 0.75 | 0.05 / 2.30 | 1e-06 | 1e-06 | heavier |
| efficientformerv2_s0.snap_dist_in1k | fp16 | 3.5 / 3.2 | 1.11 | 0.03 / 1.24 | 44 / 26 | 1.72 | 7 / 8 | 3.9 / 3.6 | 0.78 / 0.71 | 0.03 / 1.24 | 0.0043 | 0.0043 | slower heavier |
| efficientformerv2_s0.snap_dist_in1k | fp32 | 3.4 / 3.5 | 0.98 | 0.03 / 1.19 | 46 / 45 | 1.03 | 14 / 14 | 3.9 / 4.0 | 0.73 / 0.72 | 0.04 / 1.19 | 1.3e-06 | 1.3e-06 | heavier |
| efficientformerv2_s1.snap_dist_in1k | fp16 | 4.4 / 4.2 | 1.04 | 0.03 / 1.42 | 63 / 37 | 1.72 | 12 / 13 | 4.8 / 4.5 | 0.81 / 0.71 | 0.03 / 1.42 | 0.0052 | 0.0052 | slower heavier |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.4 / 4.5 | 1.21 | 0.04 / 1.37 | 67 / 66 | 1.01 | 24 / 24 | 4.7 / 4.9 | 0.74 / 0.71 | 0.04 / 1.38 | 1.7e-06 | 1.7e-06 | slower |
| efficientformerv2_s2.snap_dist_in1k | fp16 | 7.1 / 6.2 | 1.14 | 0.04 / 1.76 | 115 / 56 | 2.06 | 25 / 25 | 7.5 / 7.8 | 0.84 / 0.72 | 0.04 / 1.76 | 0.0036 | 0.0036 | slower heavier |
| efficientformerv2_s2.snap_dist_in1k | fp32 | 6.6 / 6.5 | 1.02 | 0.04 / 1.73 | 118 / 105 | 1.12 | 48 / 50 | 7.1 / 7.4 | 0.78 / 0.71 | 0.04 / 1.73 | 1.1e-06 | 1.1e-06 | heavier |
| efficientnet_b0.tv_in1k | fp16 | 3.3 / 4.2 | 0.80 | 0.02 / 0.86 | 41 / 63 | 0.65 | 11 / 10 | 3.8 / 4.7 | 0.58 / 0.54 | 0.02 / 0.93 | 0.0013 | 0.0013 |  |
| efficientnet_b0.tv_in1k | fp32 | 3.4 / 4.4 | 0.78 | 0.03 / 0.84 | 58 / 94 | 0.62 | 21 / 21 | 4.0 / 4.9 | 0.53 / 0.52 | 0.04 / 0.90 | 9.5e-07 | 9.5e-07 |  |
| efficientnet_b4.tv_in1k | fp16 | 12.8 / 15.2 | 0.85 | 0.03 / 1.45 | 233 / 229 | 1.02 | 38 / 38 | 13.6 / 15.9 | 0.71 / 0.59 | 0.03 / 1.46 | 0.00084 | 0.00084 |  |
| efficientnet_b4.tv_in1k | fp32 | 13.8 / 18.0 | 0.77 | 0.04 / 1.40 | 349 / 259 | 1.35 | 75 / 75 | 14.6 / 19.1 | 0.62 / 0.60 | 0.05 / 1.41 | 5.1e-07 | 5.1e-07 | heavier |
| efficientnet_v2_s.tv_in1k | fp16 | 14.0 / 15.7 | 0.89 | 0.03 / 1.53 | 181 / 234 | 0.77 | 41 / 41 | 14.5 / 16.3 | 0.73 / 0.60 | 0.03 / 1.53 | 0.00064 | 0.00064 |  |
| efficientnet_v2_s.tv_in1k | fp32 | 14.9 / 17.8 | 0.84 | 0.04 / 1.47 | 270 / 260 | 1.04 | 82 / 82 | 15.7 / 18.4 | 0.66 / 0.61 | 0.04 / 1.48 | 3.9e-07 | 3.9e-07 | heavier |
| efficientvit-sam-l0 | fp32 | 28.7 / 52.3 | 0.55 | 0.06 / 1.23 | 458 / 472 | 0.97 | 120 / 117 | 57.8 / 81.1 | 0.15 / 1.17 | 0.10 / 1.27 | 4.4e-05 | 1 (0.00036 % off) |  |
| efficientvit-sam-l1 | fp32 | 38.7 / 62.4 | 0.62 | 0.07 / 1.38 | 615 / 624 | 0.99 | 169 / 166 | 67.8 / 91.6 | 0.18 / 1.40 | 0.10 / 1.42 | 0.00012 | 1 (0.0028 % off) |  |
| efficientvit-sam-l2 | fp32 | 51.8 / 76.7 | 0.68 | 0.08 / 1.50 | 732 / 778 | 0.94 | 222 / 219 | 80.7 / 105.8 | 0.23 / 1.66 | 0.12 / 1.54 | 0.0003 | 1 (0.005 % off) |  |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 0.22 / 1.50 | 1311 / 1026 | 1.28 | 443 / 431 | 214.6 / 222.6 | 0.46 / 1.41 | 0.26 / 1.53 | 3.5e-05 | 1 (0.0011 % off) | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 0.37 / 1.93 | 2197 / 1668 | 1.32 | 772 / 760 | 368.9 / 363.1 | 0.74 / 1.56 | 0.42 / 1.96 | 3.7e-05 | 1 (0.003 % off) | heavier |
| efficientvit_b2.r288_in1k | fp16 | 6.6 / 8.2 | 0.81 | 0.03 / 1.34 | 134 / 137 | 0.98 | 47 / 47 | 7.7 / 10.2 | 0.82 / 0.70 | 0.03 / 1.34 | 0.0015 | 0.0015 |  |
| efficientvit_b2.r288_in1k | fp32 | 7.7 / 10.1 | 0.76 | 0.04 / 1.36 | 218 / 194 | 1.13 | 93 / 93 | 8.3 / 10.7 | 0.78 / 0.70 | 0.04 / 1.36 | 2.1e-06 | 2.1e-06 | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp16 | 195.0 / 197.4 | 0.99 | 0.23 / 2.07 | 717 / 666 | 1.08 | 583 / 583 | 195.9 / 198.2 | 0.07 / 0.74 | 0.23 / 2.07 | 0.0012 | 0.0012 | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp32 | 240.7 / 242.1 | 0.99 | 0.27 / 2.07 | 1430 / 1396 | 1.02 | 1165 / 1165 | 241.4 / 243.0 | 0.09 / 0.73 | 0.27 / 2.07 | 6e-07 | 6e-07 | heavier |
| falconsai-nsfw-vit | fp16 | 14.0 / 13.6 | 1.03 | 0.04 / 0.66 | 240 / 231 | 1.04 | 164 / 164 | 14.3 / 13.9 | 0.05 / 0.72 | 0.04 / 0.67 | 0 | 0 | slower heavier |
| falconsai-nsfw-vit | fp32 | 16.0 / 16.0 | 1.00 | 0.05 / 0.65 | 452 / 417 | 1.08 | 327 / 327 | 16.3 / 16.4 | 0.06 / 0.73 | 0.05 / 0.66 | 1.1e-07 | 1.1e-07 | heavier |
| faster-rcnn-r50-fpn-3x-coco | fp16 | 95.5 / 104.1 | 0.92 | 0.11 / 0.68 | 745 / 432 | 1.72 | 80 / 52 | 116.2 / 339.5 | 0.54 / 0.57 | 0.16 / 0.93 | - | 0.00015 | heavier |
| faster-rcnn-r50-fpn-3x-coco | fp32 | 116.1 / 125.2 | 0.93 | 0.15 / 0.70 | 1488 / 857 | 1.74 | 159 / 105 | 136.8 / 359.3 | 0.56 / 0.56 | 0.19 / 0.98 | - | 1.6e-06 | heavier |
| fastvit_sa24.apple_dist_in1k | fp16 | 6.3 / 6.7 | 0.93 | 0.02 / 0.86 | 92 / 105 | 0.88 | 41 / 41 | 6.9 / 7.3 | 1.02 / 0.90 | 0.02 / 0.86 | 0.0033 | 0.0033 |  |
| fastvit_sa24.apple_dist_in1k | fp32 | 7.0 / 7.6 | 0.92 | 0.04 / 0.81 | 173 / 199 | 0.87 | 82 / 82 | 7.6 / 8.4 | 0.98 / 0.91 | 0.04 / 0.82 | 3.3e-07 | 3.3e-07 |  |
| fastvit_t12.apple_dist_in1k | fp16 | 3.4 / 3.3 | 1.04 | 0.02 / 0.51 | 39 / 47 | 0.83 | 14 / 14 | 3.9 / 3.9 | 0.92 / 0.85 | 0.02 / 0.52 | 0.0051 | 0.0051 | slower |
| fastvit_t12.apple_dist_in1k | fp32 | 3.7 / 3.6 | 1.03 | 0.03 / 0.50 | 71 / 87 | 0.82 | 29 / 29 | 4.3 / 4.2 | 0.87 / 0.84 | 0.04 / 0.50 | 7.1e-07 | 7.1e-07 | slower |
| fastvit_t8.apple_dist_in1k | fp16 | 2.7 / 2.7 | 0.99 | 0.02 / 0.49 | 24 / 30 | 0.81 | 8 / 8 | 3.2 / 3.2 | 0.89 / 0.86 | 0.02 / 0.50 | 0.015 (0.7 % off) | 0.015 (0.7 % off) |  |
| fastvit_t8.apple_dist_in1k | fp32 | 3.2 / 3.0 | 1.09 | 0.03 / 0.46 | 44 / 55 | 0.80 | 15 / 15 | 3.6 / 3.6 | 0.84 / 0.84 | 0.04 / 0.46 | 1.3e-06 | 1.3e-06 | slower |
| fcn-r101-d8-ade20k | fp16 | 276.8 / 290.8 | 0.95 | 0.34 / 1.54 | 638 / 670 | 0.95 | 126 / 126 | 311.9 / 325.4 | 0.19 / 0.13 | 0.37 / 1.58 | 0.001 | 0 |  |
| fcn-r101-d8-ade20k | fp32 | 365.0 / 392.2 | 0.93 | 0.41 / 1.59 | 1263 / 1236 | 1.02 | 252 / 252 | 402.9 / 425.6 | 0.19 / 0.13 | 0.45 / 1.63 | 0 | 0 | heavier |
| focalnet_base_lrf.ms_in1k | fp16 | 18.5 / 21.5 | 0.86 | 0.04 / 1.47 | 298 / 289 | 1.03 | 171 / 170 | 19.7 / 22.1 | 0.92 / 0.72 | 0.05 / 1.47 | 0.0006 | 0.0006 | heavier |
| focalnet_base_lrf.ms_in1k | fp32 | 21.1 / 26.1 | 0.81 | 0.04 / 1.38 | 544 / 491 | 1.11 | 340 / 340 | 22.1 / 26.7 | 0.87 / 0.72 | 0.05 / 1.38 | 2.9e-07 | 2.9e-07 | heavier |
| focalnet_base_srf.ms_in1k | fp16 | 18.1 / 20.1 | 0.90 | 0.04 / 1.36 | 297 / 295 | 1.01 | 169 / 169 | 18.7 / 20.6 | 0.91 / 0.72 | 0.04 / 1.36 | 0.0008 | 0.0008 |  |
| focalnet_base_srf.ms_in1k | fp32 | 20.2 / 24.7 | 0.82 | 0.04 / 1.28 | 551 / 492 | 1.12 | 337 / 337 | 20.8 / 25.2 | 0.86 / 0.70 | 0.04 / 1.28 | 3.4e-07 | 3.4e-07 | heavier |
| focalnet_small_lrf.ms_in1k | fp16 | 15.1 / 15.3 | 0.98 | 0.04 / 1.45 | 201 / 195 | 1.03 | 97 / 97 | 15.6 / 16.1 | 0.90 / 0.71 | 0.04 / 1.46 | 0.00077 | 0.00077 | heavier |
| focalnet_small_lrf.ms_in1k | fp32 | 16.5 / 17.5 | 0.94 | 0.04 / 1.36 | 371 / 350 | 1.06 | 193 / 193 | 17.1 / 18.0 | 0.83 / 0.72 | 0.04 / 1.37 | 3.3e-07 | 3.3e-07 | heavier |
| focalnet_small_srf.ms_in1k | fp16 | 14.2 / 13.8 | 1.03 | 0.04 / 1.34 | 204 / 201 | 1.01 | 96 / 96 | 14.8 / 14.5 | 0.87 / 0.71 | 0.04 / 1.34 | 0.00069 | 0.00069 | slower |
| focalnet_small_srf.ms_in1k | fp32 | 15.4 / 16.3 | 0.94 | 0.04 / 1.25 | 363 / 336 | 1.08 | 191 / 191 | 16.0 / 16.9 | 0.81 / 0.71 | 0.04 / 1.25 | 4e-07 | 4e-07 | heavier |
| focalnet_tiny_lrf.ms_in1k | fp16 | 8.6 / 8.4 | 1.02 | 0.03 / 0.83 | 121 / 143 | 0.85 | 55 / 55 | 9.2 / 9.0 | 0.81 / 0.69 | 0.03 / 0.83 | 0.00058 | 0.00058 | slower |
| focalnet_tiny_lrf.ms_in1k | fp32 | 9.3 / 9.7 | 0.95 | 0.03 / 0.79 | 214 / 257 | 0.83 | 110 / 110 | 10.1 / 10.3 | 0.76 / 0.71 | 0.03 / 0.80 | 2.5e-07 | 2.5e-07 |  |
| focalnet_tiny_srf.ms_in1k | fp16 | 8.0 / 7.8 | 1.03 | 0.03 / 0.79 | 118 / 138 | 0.85 | 55 / 55 | 8.6 / 8.5 | 0.79 / 0.69 | 0.03 / 0.80 | 0.00088 | 0.00088 | slower |
| focalnet_tiny_srf.ms_in1k | fp32 | 8.7 / 8.8 | 0.99 | 0.04 / 0.73 | 207 / 244 | 0.85 | 109 / 109 | 9.2 / 9.4 | 0.77 / 0.70 | 0.04 / 0.73 | 4.3e-07 | 4.3e-07 |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp16 | 1.8 / 2.3 | 0.78 | 0.03 / 0.73 | 21 / 28 | 0.78 | 12 / 12 | 2.1 / 2.7 | 0.75 / 0.70 | 0.03 / 0.73 | 0.00081 | 0.00081 |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp32 | 2.0 / 2.6 | 0.79 | 0.04 / 0.70 | 39 / 52 | 0.76 | 23 / 23 | 2.5 / 3.1 | 0.73 / 0.70 | 0.04 / 0.70 | 8.1e-07 | 8.1e-07 |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp16 | 2.1 / 2.7 | 0.80 | 0.03 / 0.73 | 25 / 35 | 0.71 | 12 / 12 | 2.6 / 3.1 | 0.75 / 0.72 | 0.03 / 0.73 | 0.00063 | 0.00063 |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp32 | 3.2 / 3.0 | 1.06 | 0.04 / 0.69 | 46 / 66 | 0.70 | 24 / 24 | 2.9 / 3.4 | 0.72 / 0.69 | 0.04 / 0.69 | 7.8e-07 | 7.8e-07 | slower |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp16 | 3.1 / 3.8 | 0.83 | 0.04 / 0.95 | 40 / 53 | 0.75 | 22 / 22 | 3.9 / 4.3 | 0.76 / 0.70 | 0.04 / 0.95 | 0.00053 | 0.00053 |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp32 | 3.5 / 4.2 | 0.84 | 0.05 / 0.99 | 74 / 100 | 0.74 | 43 / 43 | 3.9 / 5.0 | 0.72 / 0.77 | 0.05 / 0.99 | 9.1e-07 | 9.1e-07 |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp16 | 3.7 / 4.5 | 0.83 | 0.04 / 1.21 | 53 / 69 | 0.77 | 31 / 31 | 4.2 / 5.0 | 0.80 / 0.74 | 0.04 / 1.21 | 0.00052 | 0.00052 |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp32 | 4.3 / 5.2 | 0.82 | 0.05 / 1.12 | 99 / 131 | 0.76 | 62 / 62 | 4.7 / 5.8 | 0.77 / 0.72 | 0.05 / 1.12 | 5.1e-07 | 5.1e-07 |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp16 | 5.0 / 5.4 | 0.93 | 0.02 / 1.03 | 65 / 72 | 0.90 | 38 / 38 | 5.4 / 5.8 | 0.80 / 0.71 | 0.02 / 1.03 | 0.00064 | 0.00064 |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp32 | 5.7 / 6.2 | 0.92 | 0.03 / 1.01 | 124 / 137 | 0.90 | 76 / 76 | 6.1 / 6.7 | 0.73 / 0.69 | 0.04 / 1.01 | 5.4e-07 | 5.4e-07 |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp16 | 9.0 / 9.6 | 0.94 | 0.03 / 1.46 | 132 / 139 | 0.95 | 76 / 76 | 9.6 / 10.3 | 0.86 / 0.71 | 0.03 / 1.46 | 0.00064 | 0.00064 |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp32 | 10.5 / 11.3 | 0.93 | 0.04 / 1.42 | 242 / 256 | 0.95 | 151 / 151 | 11.0 / 11.9 | 0.78 / 0.74 | 0.04 / 1.43 | 3.9e-07 | 3.9e-07 |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp16 | 18.8 / 19.6 | 0.96 | 0.04 / 1.88 | 248 / 272 | 0.91 | 144 / 144 | 19.3 / 20.3 | 0.93 / 0.78 | 0.04 / 1.88 | 0.00088 | 0.00088 |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp32 | 21.8 / 23.2 | 0.94 | 0.05 / 1.74 | 460 / 493 | 0.93 | 288 / 288 | 22.2 / 23.9 | 0.83 / 0.72 | 0.06 / 1.75 | 5.9e-07 | 5.9e-07 |  |
| hrnet-w48-fcn-ade20k | fp16 | 180.2 / 162.1 | 1.11 | 0.21 / 2.27 | 534 / 292 | 1.83 | 129 / 128 | 221.3 / 200.3 | 0.26 / 0.20 | 0.26 / 2.32 | 0.00094 | 0.76 (0.054 % off) | slower heavier |
| hrnet-w48-fcn-ade20k | fp32 | 200.0 / 194.2 | 1.03 | 0.23 / 2.18 | 723 / 609 | 1.19 | 254 / 254 | 242.1 / 235.1 | 0.26 / 0.20 | 0.28 / 2.22 | 6.3e-07 | 0.56 (0.00019 % off) | slower heavier |
| igev-plusplus-eth3d | fp16 | 5647.5 / 7058.8 | 0.80 | 5.72 / 12.90 | 2414 / 2189 | 1.10 | 28 / 29 | 5657.6 / 6990.0 | 0.15 / 0.10 | 5.74 / 12.90 | 0.45 (0.33 % off) | 0.29 (0.034 % off) | heavier |
| igev-plusplus-eth3d | fp32 | 6887.7 / 7321.9 | 0.94 | 6.82 / 14.19 | 4260 / 3786 | 1.13 | 55 / 56 | 6834.3 / 7224.1 | 0.13 / 0.10 | 6.83 / 14.20 | 0.0043 | 0.0043 | heavier |
| igev-plusplus-kitti12 | fp16 | 5648.8 / 6263.4 | 0.90 | 5.72 / 13.24 | 2414 / 2189 | 1.10 | 28 / 29 | 5666.7 / 7159.4 | 0.15 / 0.10 | 5.73 / 13.25 | 0.023 (0.0032 % off) | 0.026 (0.00027 % off) | heavier |
| igev-plusplus-kitti12 | fp32 | 6895.0 / 7199.4 | 0.96 | 6.89 / 14.41 | 4260 / 3786 | 1.13 | 55 / 56 | 6898.8 / 7142.8 | 0.13 / 0.10 | 6.90 / 14.42 | 0.00037 | 0.00053 | heavier |
| igev-plusplus-kitti15 | fp16 | 5636.4 / 7098.7 | 0.79 | 5.69 / 14.05 | 2414 / 2199 | 1.10 | 28 / 29 | 5643.9 / 6982.9 | 0.15 / 0.10 | 5.70 / 14.05 | 0.053 (0.003 % off) | 0.0086 | heavier |
| igev-plusplus-kitti15 | fp32 | 6871.0 / 7144.8 | 0.96 | 6.89 / 14.60 | 4260 / 3786 | 1.13 | 55 / 56 | 6858.2 / 7146.3 | 0.13 / 0.10 | 6.91 / 14.60 | 1.9e-05 | 2.1e-05 | heavier |
| igev-plusplus-middlebury | fp16 | 5711.5 / 6166.5 | 0.93 | 5.79 / 14.20 | 2414 / 2189 | 1.10 | 28 / 29 | 5725.3 / 6161.5 | 0.15 / 0.10 | 5.80 / 14.21 | 0.39 (1.2 % off) | 0.41 (0.25 % off) | heavier |
| igev-plusplus-middlebury | fp32 | 6856.9 / 7342.5 | 0.93 | 6.88 / 14.28 | 4260 / 3786 | 1.13 | 55 / 56 | 6895.7 / 7289.7 | 0.13 / 0.10 | 6.89 / 14.29 | 0.00055 | 0.00022 | heavier |
| igev-plusplus-sceneflow | fp16 | 5617.9 / 7204.4 | 0.78 | 5.68 / 13.01 | 2414 / 2199 | 1.10 | 28 / 29 | 5629.0 / 7227.3 | 0.15 / 0.10 | 5.69 / 13.02 | 0.5 (0.023 % off) | 0.44 (0.017 % off) | heavier |
| igev-plusplus-sceneflow | fp32 | 6821.8 / 7224.3 | 0.94 | 6.87 / 14.21 | 4260 / 3786 | 1.13 | 55 / 56 | 6842.0 / 7390.8 | 0.13 / 0.10 | 6.88 / 14.21 | 0.00045 | 0.00045 | heavier |
| igev-stereo-eth3d | fp16 | 3185.1 / 3594.5 | 0.89 | 3.21 / 8.93 | 1686 / 1080 | 1.56 | 24 / 26 | 3190.7 / 3586.3 | 0.15 / 0.10 | 3.22 / 8.94 | 0.49 (0.38 % off) | 0.49 (0.067 % off) | heavier |
| igev-stereo-eth3d | fp32 | 4179.4 / 4676.3 | 0.89 | 4.22 / 9.75 | 3122 / 2055 | 1.52 | 48 / 49 | 4181.1 / 4581.5 | 0.12 / 0.09 | 4.23 / 9.76 | 0.00084 | 0.00084 | heavier |
| igev-stereo-kitti12 | fp16 | 3190.2 / 3603.2 | 0.89 | 3.20 / 8.78 | 1686 / 1080 | 1.56 | 24 / 26 | 3195.6 / 3596.9 | 0.14 / 0.10 | 3.21 / 8.78 | 0.052 (0.016 % off) | 0.0042 | heavier |
| igev-stereo-kitti12 | fp32 | 4177.9 / 4578.8 | 0.91 | 4.22 / 9.69 | 3122 / 2055 | 1.52 | 48 / 49 | 4175.4 / 4562.7 | 0.12 / 0.09 | 4.23 / 9.70 | 2.7e-05 | 4.9e-06 | heavier |
| igev-stereo-kitti15 | fp16 | 3186.4 / 3601.5 | 0.88 | 3.21 / 8.71 | 1686 / 1080 | 1.56 | 24 / 26 | 3191.0 / 3612.9 | 0.14 / 0.09 | 3.22 / 8.71 | 0.018 (0.00084 % off) | 0.004 | heavier |
| igev-stereo-kitti15 | fp32 | 4162.0 / 4564.7 | 0.91 | 4.19 / 9.64 | 3122 / 2055 | 1.52 | 48 / 49 | 4184.4 / 4609.8 | 0.12 / 0.09 | 4.21 / 9.65 | 9.1e-06 | 1.8e-06 | heavier |
| igev-stereo-middlebury | fp16 | 3187.7 / 3596.0 | 0.89 | 3.20 / 8.82 | 1686 / 1010 | 1.67 | 24 / 26 | 3189.5 / 3606.0 | 0.14 / 0.10 | 3.21 / 8.83 | 5.2 (1.7 % off) | 5.2 (0.24 % off) | heavier |
| igev-stereo-middlebury | fp32 | 4175.4 / 4581.2 | 0.91 | 4.21 / 9.72 | 3122 / 2055 | 1.52 | 48 / 49 | 4178.8 / 4567.6 | 0.12 / 0.09 | 4.23 / 9.73 | 0.0013 | 0.0013 | heavier |
| igev-stereo-sceneflow | fp16 | 3187.4 / 3609.1 | 0.88 | 3.20 / 8.73 | 1686 / 1010 | 1.67 | 24 / 26 | 3200.3 / 3589.2 | 0.14 / 0.09 | 3.21 / 8.73 | 0.36 (0.16 % off) | 0.2 (0.03 % off) | heavier |
| igev-stereo-sceneflow | fp32 | 4171.9 / 4560.1 | 0.91 | 4.17 / 9.65 | 3122 / 2055 | 1.52 | 48 / 49 | 4173.9 / 4573.6 | 0.12 / 0.10 | 4.18 / 9.66 | 3.7e-05 | 3.7e-05 | heavier |
| inception_v3.tv_in1k | fp16 | 7.8 / 9.9 | 0.79 | 0.03 / 1.07 | 86 / 152 | 0.56 | 46 / 46 | 8.4 / 10.6 | 0.69 / 0.60 | 0.03 / 1.07 | 0.00084 | 0.00084 |  |
| inception_v3.tv_in1k | fp32 | 8.7 / 11.7 | 0.74 | 0.04 / 1.03 | 159 / 272 | 0.59 | 91 / 91 | 9.2 / 12.0 | 0.64 / 0.59 | 0.04 / 1.04 | 1e-06 | 1e-06 |  |
| lineart | fp16 | 70.6 / 99.9 | 0.71 | 0.08 / 1.02 | 626 / 594 | 1.05 | 8 / 8 | 79.9 / 110.6 | 0.05 / 0.03 | 0.10 / 1.04 | - | 0.028 (0.047 % off) (1 keys differ) | heavier |
| lineart | fp32 | 98.6 / 117.6 | 0.84 | 0.11 / 1.04 | 1245 / 903 | 1.38 | 16 / 16 | 105.4 / 126.6 | 0.05 / 0.03 | 0.13 / 1.06 | - | 1.8e-05 (1 keys differ) | heavier |
| lineart-coarse | fp16 | 70.7 / 99.7 | 0.71 | 0.08 / 1.03 | 626 / 594 | 1.05 | 8 / 8 | 80.1 / 109.8 | 0.05 / 0.03 | 0.10 / 1.05 | - | 0.022 (0.014 % off) (1 keys differ) | heavier |
| lineart-coarse | fp32 | 96.5 / 117.3 | 0.82 | 0.12 / 1.03 | 1245 / 903 | 1.38 | 16 / 16 | 105.9 / 126.5 | 0.05 / 0.03 | 0.13 / 1.04 | - | 1.6e-05 (1 keys differ) | heavier |
| loftr-indoor | fp16 | 424.2 / 326.7 | 1.30 | 0.46 / 2.31 | 2391 / 1090 | 2.19 | 40 / 21 | 431.6 / 1446.9 | 0.09 / 0.04 | 0.46 / 3.55 | - | empty (4 keys differ) | slower heavier |
| loftr-indoor | fp32 | 595.1 / 533.8 | 1.11 | 0.63 / 2.51 | 3722 / 1280 | 2.91 | 64 / 43 | 597.7 / 1652.9 | 0.07 / 0.04 | 0.63 / 3.74 | - | 2.4e-05 | slower heavier |
| loftr-outdoor | fp16 | 421.7 / 330.3 | 1.28 | 0.45 / 2.32 | 2391 / 1090 | 2.19 | 40 / 21 | 431.0 / 1444.4 | 0.09 / 0.04 | 0.45 / 3.55 | - | 0.0018 | slower heavier |
| loftr-outdoor | fp32 | 591.4 / 533.1 | 1.11 | 0.63 / 2.51 | 3722 / 1314 | 2.83 | 64 / 43 | 592.2 / 1643.2 | 0.07 / 0.04 | 0.63 / 3.71 | - | 7.6e-06 | slower heavier |
| loma-b | fp16 | 2280.5 / 2547.8 | 0.90 | 2.30 / 7.88 | 1488 / 1856 | 0.80 | 712 / 654 | 2301.5 / 2674.6 | 1.87 / 0.23 | 2.33 / 7.94 | - | 0.98 (79 % off) (3 keys differ) |  |
| loma-b | fp32 | 3166.5 / 3379.9 | 0.94 | 3.16 / 9.51 | 2751 / 3714 | 0.74 | 1308 / 1308 | 3184.8 / 3464.9 | 1.44 / 0.40 | 3.19 / 9.60 | - | 0.64 (0.084 % off) (1 keys differ) |  |
| loma-b128 | fp16 | 800.0 / 925.8 | 0.86 | 0.80 / 4.54 | 886 / 994 | 0.89 | 112 / 60 | 801.5 / 983.5 | 0.51 / 0.10 | 0.83 / 4.60 | - | 0.98 (79 % off) (3 keys differ) |  |
| loma-b128 | fp32 | 1195.5 / 1312.4 | 0.91 | 1.24 / 4.74 | 1234 / 1064 | 1.16 | 119 / 120 | 1219.3 / 1359.8 | 0.17 / 0.10 | 1.26 / 4.80 | - | 0.64 (0.084 % off) (1 keys differ) | heavier |
| loma-b64 | fp16 | 639.4 / 639.7 | 1.00 | 0.67 / 4.10 | 817 / 907 | 0.90 | 79 / 48 | 647.4 / 691.1 | 0.40 / 0.11 | 0.70 / 4.16 | - | 0.98 (83 % off) (3 keys differ) |  |
| loma-b64 | fp32 | 1019.7 / 989.3 | 1.03 | 1.04 / 4.33 | 1164 / 988 | 1.18 | 96 / 96 | 1030.0 / 1027.2 | 0.17 / 0.11 | 1.06 / 4.38 | - | 0.74 (0.23 % off) (1 keys differ) | slower heavier |
| loma-g | fp16 | 2751.7 / 3019.5 | 0.91 | 2.77 / 8.54 | 1813 / 1916 | 0.95 | 1037 / 978 | 2770.2 / 3080.3 | 2.30 / 0.23 | 2.81 / 8.61 | - | 0.98 (79 % off) (3 keys differ) |  |
| loma-g | fp32 | 3791.5 / 4200.1 | 0.90 | 3.92 / 9.92 | 3398 / 4360 | 0.78 | 1954 / 1954 | 3744.8 / 4117.5 | 1.89 / 0.40 | 3.95 / 10.00 | - | 0.64 (0.084 % off) (1 keys differ) |  |
| loma-l | fp16 | 2390.2 / 2626.9 | 0.91 | 2.40 / 7.66 | 1553 / 1929 | 0.80 | 777 / 719 | 2407.0 / 2685.3 | 1.92 / 0.23 | 2.43 / 7.73 | - | 0.98 (79 % off) (3 keys differ) |  |
| loma-l | fp32 | 3299.5 / 3482.7 | 0.95 | 3.31 / 9.08 | 2881 / 3843 | 0.75 | 1437 / 1438 | 3321.7 / 3559.0 | 1.49 / 0.27 | 3.35 / 9.15 | - | 0.64 (0.084 % off) (1 keys differ) |  |
| loma-r | fp16 | 2272.8 / 2620.4 | 0.87 | 2.29 / 7.60 | 1488 / 1592 | 0.93 | 712 / 654 | 2290.7 / 2630.0 | 1.87 / 0.23 | 2.32 / 7.66 | - | 0.98 (79 % off) (3 keys differ) |  |
| loma-r | fp32 | 3145.3 / 3333.3 | 0.94 | 3.13 / 8.84 | 2751 / 3714 | 0.74 | 1308 / 1308 | 3148.5 / 3382.4 | 1.43 / 0.42 | 3.16 / 8.91 | - | 0.64 (0.084 % off) (1 keys differ) |  |
| lw-detr-large-coco | fp16 | 53.4 / 59.4 | 0.90 | 0.09 / 1.97 | 534 / 347 | 1.54 | 118 / 81 | 56.0 / 62.8 | 2.13 / 2.07 | 0.09 / 1.97 | 0.96 (67 % off) | 0.026 (11 % off) | heavier |
| lw-detr-large-coco | fp32 | 59.0 / 61.7 | 0.96 | 0.09 / 1.88 | 554 / 531 | 1.04 | 154 / 160 | 62.1 / 65.3 | 2.12 / 2.05 | 0.10 / 1.89 | 0.00012 | 2.2e-06 | heavier |
| lw-detr-medium-coco | fp16 | 37.0 / 43.2 | 0.86 | 0.07 / 1.77 | 224 / 317 | 0.71 | 60 / 51 | 40.0 / 46.3 | 2.11 / 2.00 | 0.07 / 1.78 | 0.97 (74 % off) | empty (3 keys differ) |  |
| lw-detr-medium-coco | fp32 | 42.8 / 42.8 | 1.00 | 0.08 / 1.75 | 420 / 240 | 1.75 | 96 / 100 | 45.6 / 45.9 | 2.07 / 2.01 | 0.09 / 1.76 | 0.00021 | 4.7e-06 | heavier |
| lw-detr-small-coco | fp16 | 23.7 / 28.9 | 0.82 | 0.06 / 1.72 | 185 / 346 | 0.54 | 34 / 24 | 26.6 / 32.0 | 2.04 / 2.03 | 0.07 / 1.73 | 0.95 (66 % off) | 0.72 (12 % off) |  |
| lw-detr-small-coco | fp32 | 28.9 / 29.2 | 0.99 | 0.08 / 1.67 | 341 / 216 | 1.58 | 43 / 47 | 31.8 / 32.6 | 2.03 / 2.02 | 0.08 / 1.67 | 0.00084 | 5.5e-06 | heavier |
| lw-detr-tiny-coco | fp16 | 16.8 / 20.4 | 0.82 | 0.05 / 1.46 | 147 / 276 | 0.53 | 30 / 19 | 19.5 / 23.4 | 2.02 / 1.99 | 0.06 / 1.46 | 1.1 (57 % off) | empty (3 keys differ) |  |
| lw-detr-tiny-coco | fp32 | 20.7 / 20.7 | 1.00 | 0.06 / 1.42 | 225 / 206 | 1.09 | 36 / 37 | 23.5 / 23.8 | 2.01 / 1.99 | 0.06 / 1.42 | 0.00021 | 1.6e-05 | heavier |
| lw-detr-xlarge-coco | fp16 | 96.9 / 108.3 | 0.90 | 0.12 / 2.03 | 698 / 445 | 1.57 | 288 / 219 | 100.1 / 111.6 | 2.27 / 2.04 | 0.13 / 2.04 | 0.97 (69 % off) | empty (3 keys differ) | heavier |
| lw-detr-xlarge-coco | fp32 | 105.1 / 111.2 | 0.95 | 0.13 / 2.00 | 837 / 884 | 0.95 | 427 / 434 | 108.2 / 114.2 | 2.25 / 2.08 | 0.14 / 2.01 | 0.61 (0.78 % off) | 0.00068 |  |
| mask-rcnn-r50-fpn-3x-coco | fp16 | 97.7 / 102.5 | 0.95 | 0.12 / 0.67 | 750 / 432 | 1.73 | 86 / 52 | 139.1 / 374.4 | 0.55 / 0.53 | 0.18 / 0.97 | - | 1 (0.0068 % off) | heavier |
| mask-rcnn-r50-fpn-3x-coco | fp32 | 119.4 / 125.0 | 0.95 | 0.15 / 0.67 | 1498 / 860 | 1.74 | 169 / 105 | 159.0 / 395.9 | 0.49 / 0.53 | 0.21 / 0.97 | - | 1 (7e-05 % off) | heavier |
| mask2former-swinl-ade20k | fp16 | 425.1 / 554.5 | 0.77 | 0.45 / 5.25 | 817 / 962 | 0.85 | 458 / 455 | 570.7 / 724.7 | 2.31 / 2.15 | 0.65 / 5.48 | - | 0 |  |
| mask2former-swinl-ade20k | fp32 | 473.1 / 723.5 | 0.65 | 0.48 / 5.31 | 1704 / 1737 | 0.98 | 915 / 907 | 662.1 / 881.7 | 2.33 / 2.15 | 0.72 / 5.54 | - | 0 |  |
| mobilenet_v2.tv2_in1k | fp16 | 2.4 / 2.0 | 1.18 | 0.01 / 0.78 | 23 / 24 | 0.95 | 7 / 7 | 2.4 / 2.4 | 0.56 / 0.51 | 0.02 / 0.79 | 0.0012 | 0.0012 | slower |
| mobilenet_v2.tv2_in1k | fp32 | 2.4 / 2.7 | 0.90 | 0.03 / 0.76 | 42 / 44 | 0.96 | 14 / 14 | 2.7 / 2.7 | 0.52 / 0.50 | 0.03 / 0.77 | 1.5e-06 | 1.5e-06 |  |
| mobilenet_v3_large.tv2_in1k | fp16 | 2.3 / 3.2 | 0.73 | 0.02 / 0.91 | 26 / 32 | 0.80 | 11 / 11 | 2.7 / 3.2 | 0.60 / 0.53 | 0.02 / 0.91 | 0.0025 | 0.0025 |  |
| mobilenet_v3_large.tv2_in1k | fp32 | 3.0 / 3.2 | 0.92 | 0.03 / 0.85 | 42 / 53 | 0.81 | 21 / 21 | 2.9 / 3.6 | 0.52 / 0.50 | 0.04 / 0.86 | 1.4e-06 | 1.4e-06 |  |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | 2.3 / 1.9 | 1.20 | 0.01 / 0.54 | 23 / 23 | 0.97 | 8 / 8 | 2.2 / 2.2 | 0.75 / 0.70 | 0.01 / 0.55 | 0.0015 | 0.0015 | slower |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp32 | 2.3 / 2.1 | 1.09 | 0.02 / 0.52 | 40 / 41 | 0.98 | 16 / 16 | 2.5 / 2.5 | 0.71 / 0.69 | 0.03 / 0.52 | 1.8e-06 | 1.8e-06 | slower |
| mobilenetv4_conv_medium.e500_r256_in1k | fp16 | 2.9 / 3.0 | 0.98 | 0.02 / 0.96 | 40 / 41 | 0.97 | 19 / 19 | 3.5 / 3.5 | 0.79 / 0.71 | 0.02 / 0.97 | 0.00074 | 0.00074 |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 3.7 / 3.5 | 1.07 | 0.03 / 0.92 | 72 / 74 | 0.97 | 37 / 37 | 3.8 / 3.9 | 0.77 / 0.74 | 0.03 / 0.92 | 5.4e-07 | 5.4e-07 | slower |
| mvsanywhere | fp16 | 414.7 / 372.0 | 1.11 | 3.50 / 3.45 | 767 / 767 | 1.00 | 386 / 386 | 424.6 / 3490.1 | 0.10 / 0.10 | 3.51 / 6.59 | 0 | 0.045 (0.49 % off) | slower |
| mvsanywhere | fp32 | 502.0 / 471.5 | 1.06 | 3.54 / 3.54 | 1407 / 1407 | 1.00 | 771 / 771 | 508.6 / 3574.8 | 0.10 / 0.10 | 3.56 / 6.70 | 0 | 0.00018 | slower |
| nafnet-gopro | fp16 | 214.5 / 249.6 | 0.86 | 0.23 / 3.09 | 769 / 772 | 1.00 | 130 / 255 | 215.1 / 250.1 | 0.28 / 0.16 | 0.23 / 3.10 | 0.0019 | 1 (0.052 % off) |  |
| nafnet-gopro | fp32 | 257.5 / 291.6 | 0.88 | 0.29 / 3.06 | 1170 / 839 | 1.39 | 259 / 384 | 257.9 / 292.4 | 0.26 / 0.16 | 0.30 / 3.07 | 2e-06 | 0.0085 | heavier |
| nafnet-sidd | fp16 | 219.4 / 207.7 | 1.06 | 0.23 / 2.25 | 804 / 578 | 1.39 | 221 / 221 | 220.0 / 207.7 | 0.41 / 0.21 | 0.24 / 2.26 | 0.0015 | 0.0034 | slower heavier |
| nafnet-sidd | fp32 | 254.2 / 268.5 | 0.95 | 0.29 / 2.21 | 1346 / 896 | 1.50 | 442 / 442 | 254.5 / 268.8 | 0.40 / 0.21 | 0.29 / 2.21 | 1.2e-06 | 2.6e-06 | heavier |
| oneformer-swinl-ade20k | fp16 | 504.3 / 656.9 | 0.77 | 0.54 / 5.96 | 1746 / 1002 | 1.74 | 523 / 536 | 794.7 / 893.9 | 2.47 / 2.24 | 0.91 / 6.20 | - | 0 | heavier |
| oneformer-swinl-ade20k | fp32 | 573.8 / 804.8 | 0.71 | 0.58 / 6.00 | 2270 / 1993 | 1.14 | 980 / 1070 | 915.6 / 1026.2 | 2.45 / 2.29 | 1.00 / 6.24 | - | 0 | heavier |
| pe-core-b16-224 | fp16 | 14.9 / 20.5 | 0.73 | 0.03 / 0.87 | 263 / 404 | 0.65 | 176 / 177 | 15.2 / 20.7 | 0.85 / 0.71 | 0.03 / 0.87 | 0.03 (0.024 % off) | 0.03 (0.024 % off) |  |
| pe-core-b16-224 | fp32 | 17.3 / 17.6 | 0.98 | 0.05 / 0.90 | 496 / 436 | 1.14 | 352 / 354 | 17.4 / 18.0 | 0.78 / 0.72 | 0.05 / 0.90 | 8.6e-05 | 8.6e-05 | heavier |
| pe-core-g14-448 | fp16 | 249.2 / 354.6 | 0.70 | 0.28 / 13.99 | 3664 / 3940 | 0.93 | 3580 / 3585 | 249.6 / 355.1 | 10.22 / 1.37 | 0.29 / 14.00 | 0.05 (0.014 % off) | 0.05 (0.014 % off) |  |
| pe-core-g14-448 | fp32 | 292.3 / 292.0 | 1.00 | 1.08 / 27.74 | 7303 / 7296 | 1.00 | 7160 / 7169 | 292.6 / 292.4 | 10.08 / 1.34 | 1.09 / 27.75 | 4e-05 | 4e-05 |  |
| pe-core-l14-336 | fp16 | 56.7 / 77.5 | 0.73 | 0.08 / 1.58 | 680 / 881 | 0.77 | 601 / 602 | 57.0 / 77.8 | 0.99 / 0.76 | 0.08 / 1.58 | 0.0091 | 0.0091 |  |
| pe-core-l14-336 | fp32 | 66.2 / 66.3 | 1.00 | 0.10 / 1.59 | 1334 / 1372 | 0.97 | 1201 / 1205 | 66.7 / 66.5 | 0.84 / 0.77 | 0.11 / 1.60 | 3e-05 | 3e-05 |  |
| pe-core-s16-384 | fp16 | 6.4 / 8.0 | 0.79 | 0.02 / 0.88 | 94 / 200 | 0.47 | 45 / 45 | 6.6 / 8.3 | 0.83 / 0.76 | 0.02 / 0.88 | 0.0056 | 0.0056 |  |
| pe-core-s16-384 | fp32 | 7.3 / 7.3 | 1.00 | 0.04 / 0.84 | 170 / 166 | 1.02 | 89 / 90 | 7.5 / 7.4 | 0.76 / 0.78 | 0.04 / 0.84 | 1.3e-05 | 1.3e-05 | heavier |
| pe-core-t16-384 | fp16 | 3.7 / 4.5 | 0.82 | 0.02 / 0.83 | 34 / 67 | 0.50 | 12 / 12 | 3.8 / 4.7 | 0.82 / 0.71 | 0.02 / 0.83 | 0.011 (0.0053 % off) | 0.011 (0.0053 % off) |  |
| pe-core-t16-384 | fp32 | 3.9 / 4.2 | 0.92 | 0.04 / 0.81 | 62 / 59 | 1.04 | 23 / 23 | 4.1 / 4.3 | 0.77 / 0.71 | 0.04 / 0.81 | 4.6e-05 | 4.6e-05 | heavier |
| pe-spatial-b16-512 | fp16 | 14.5 / 19.7 | 0.73 | 0.03 / 0.84 | 251 / 390 | 0.64 | 164 / 164 | 14.7 / 20.0 | 0.84 / 0.71 | 0.03 / 0.84 | 0.0021 | 0.0021 |  |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.2 | 0.97 | 0.05 / 0.82 | 471 / 409 | 1.15 | 327 / 327 | 17.0 / 17.3 | 0.79 / 0.72 | 0.05 / 0.83 | 7e-06 | 7e-06 | heavier |
| pe-spatial-g14-448 | fp16 | 248.2 / 353.0 | 0.70 | 0.31 / 6.58 | 3614 / 3886 | 0.93 | 3531 / 3531 | 248.7 / 353.2 | 9.38 / 1.35 | 0.31 / 6.58 | 0.0069 | 0.0069 |  |
| pe-spatial-g14-448 | fp32 | 290.7 / 290.5 | 1.00 | 0.87 / 25.75 | 7204 / 7188 | 1.00 | 7061 / 7061 | 291.4 / 291.1 | 9.94 / 1.39 | 0.88 / 25.76 | 5.8e-06 | 5.8e-06 |  |
| pe-spatial-l14-448 | fp16 | 56.2 / 76.4 | 0.74 | 0.08 / 1.55 | 658 / 857 | 0.77 | 578 / 578 | 56.4 / 76.6 | 1.02 / 0.76 | 0.08 / 1.55 | 0.0024 | 0.0024 |  |
| pe-spatial-l14-448 | fp32 | 65.4 / 65.5 | 1.00 | 0.10 / 1.51 | 1290 / 1324 | 0.97 | 1157 / 1157 | 65.7 / 65.8 | 0.85 / 0.75 | 0.10 / 1.51 | 1.7e-06 | 1.7e-06 |  |
| pe-spatial-s16-512 | fp16 | 6.2 / 7.6 | 0.82 | 0.03 / 0.81 | 91 / 197 | 0.46 | 42 / 41 | 6.4 / 7.8 | 0.83 / 0.70 | 0.03 / 0.81 | 0.0055 | 0.0055 |  |
| pe-spatial-s16-512 | fp32 | 7.0 / 6.9 | 1.03 | 0.04 / 0.80 | 163 / 159 | 1.02 | 83 / 83 | 7.3 / 7.2 | 0.75 / 0.73 | 0.04 / 0.81 | 1.4e-05 | 1.4e-05 | slower heavier |
| pe-spatial-t16-512 | fp16 | 3.6 / 4.2 | 0.85 | 0.02 / 0.79 | 33 / 66 | 0.50 | 11 / 11 | 3.9 / 4.4 | 0.80 / 0.70 | 0.02 / 0.79 | 0.0077 | 0.0077 |  |
| pe-spatial-t16-512 | fp32 | 3.8 / 3.7 | 1.01 | 0.04 / 0.76 | 59 / 58 | 1.03 | 21 / 21 | 4.0 / 4.1 | 0.77 / 0.70 | 0.04 / 0.76 | 1.3e-05 | 1.3e-05 | heavier |
| pspnet-r101-d8-ade20k | fp16 | 290.3 / 302.1 | 0.96 | 0.38 / 1.60 | 966 / 768 | 1.26 | 125 / 125 | 324.9 / 334.4 | 0.19 / 0.13 | 0.42 / 1.64 | 0.00078 | 0 | heavier |
| pspnet-r101-d8-ade20k | fp32 | 369.0 / 391.3 | 0.94 | 0.43 / 1.66 | 1664 / 1724 | 0.97 | 250 / 250 | 404.1 / 426.0 | 0.19 / 0.13 | 0.47 / 1.70 | 3.8e-07 | 0 |  |
| raft-stereo-eth3d | fp16 | 3176.0 / 4061.6 | 0.78 | 3.23 / 8.48 | 1630 / 1166 | 1.40 | 29 / 82 | 3178.6 / 4052.2 | 0.07 / 0.06 | 3.23 / 8.48 | 0.058 (0.06 % off) | 0.049 (0.015 % off) | heavier |
| raft-stereo-eth3d | fp32 | 4832.4 / 5011.0 | 0.96 | 4.87 / 9.57 | 2943 / 1872 | 1.57 | 42 / 116 | 4848.8 / 5022.1 | 0.07 / 0.06 | 4.88 / 9.57 | 0.00012 | 0.00012 | heavier |
| raft-stereo-middlebury | fp16 | 3168.7 / 4054.7 | 0.78 | 3.23 / 8.62 | 1630 / 1128 | 1.45 | 29 / 82 | 3185.3 / 4058.4 | 0.07 / 0.06 | 3.24 / 8.63 | 0.034 (0.15 % off) | 0.036 (0.13 % off) | heavier |
| raft-stereo-middlebury | fp32 | 4808.4 / 5041.7 | 0.95 | 4.84 / 9.51 | 2943 / 1872 | 1.57 | 42 / 116 | 4845.5 / 5028.2 | 0.07 / 0.06 | 4.85 / 9.52 | 0.00092 | 0.00098 | heavier |
| raft-stereo-sceneflow | fp16 | 3191.9 / 4043.8 | 0.79 | 3.23 / 8.52 | 1630 / 1042 | 1.56 | 29 / 82 | 3179.8 / 4049.5 | 0.07 / 0.06 | 3.24 / 8.52 | 0.27 (0.11 % off) | 0.28 (0.057 % off) | heavier |
| raft-stereo-sceneflow | fp32 | 4812.2 / 5045.0 | 0.95 | 4.86 / 9.46 | 2943 / 1872 | 1.57 | 42 / 116 | 4836.6 / 5021.1 | 0.07 / 0.06 | 4.87 / 9.47 | 0.00012 | 0.00012 | heavier |
| regnety_160.tv2_in1k | fp16 | 17.1 / 18.3 | 0.93 | 0.03 / 1.01 | 237 / 280 | 0.85 | 160 / 160 | 17.7 / 18.8 | 0.74 / 0.59 | 0.04 / 1.01 | 0.00029 | 0.00029 |  |
| regnety_160.tv2_in1k | fp32 | 18.3 / 23.5 | 0.78 | 0.05 / 1.00 | 430 / 574 | 0.75 | 319 / 319 | 19.0 / 24.0 | 0.68 / 0.61 | 0.05 / 1.00 | 3.5e-07 | 3.5e-07 |  |
| repvit_m0_9.dist_300e_in1k | fp16 | 3.6 / 4.4 | 0.80 | 0.02 / 1.32 | 32 / 33 | 0.95 | 10 / 11 | 4.0 / 5.1 | 0.79 / 0.71 | 0.02 / 1.32 | 0.0078 | 0.0078 |  |
| repvit_m0_9.dist_300e_in1k | fp32 | 3.8 / 4.8 | 0.80 | 0.04 / 1.27 | 52 / 62 | 0.84 | 20 / 21 | 4.3 / 5.4 | 0.72 / 0.70 | 0.04 / 1.27 | 1.4e-06 | 1.4e-06 |  |
| repvit_m0_9.dist_450e_in1k | fp16 | 3.5 / 4.4 | 0.80 | 0.02 / 1.32 | 32 / 34 | 0.94 | 10 / 11 | 4.1 / 5.0 | 0.79 / 0.69 | 0.02 / 1.32 | 0.012 (0.2 % off) | 0.012 (0.2 % off) |  |
| repvit_m0_9.dist_450e_in1k | fp32 | 3.9 / 4.8 | 0.81 | 0.04 / 1.27 | 52 / 65 | 0.80 | 20 / 21 | 4.3 / 5.3 | 0.74 / 0.71 | 0.04 / 1.27 | 2.3e-06 | 2.3e-06 |  |
| repvit_m1_0.dist_300e_in1k | fp16 | 3.9 / 4.9 | 0.79 | 0.02 / 1.31 | 39 / 42 | 0.92 | 14 / 15 | 4.4 / 6.0 | 0.78 / 0.72 | 0.02 / 1.32 | 0.01 (0.1 % off) | 0.01 (0.1 % off) |  |
| repvit_m1_0.dist_300e_in1k | fp32 | 4.2 / 5.2 | 0.82 | 0.04 / 1.28 | 64 / 80 | 0.79 | 27 / 28 | 4.7 / 5.8 | 0.72 / 0.70 | 0.04 / 1.28 | 1.9e-06 | 1.9e-06 |  |
| repvit_m1_0.dist_450e_in1k | fp16 | 4.0 / 4.7 | 0.86 | 0.02 / 1.37 | 39 / 45 | 0.86 | 14 / 15 | 4.4 / 5.1 | 0.79 / 0.70 | 0.02 / 1.37 | 0.0065 | 0.0065 |  |
| repvit_m1_0.dist_450e_in1k | fp32 | 4.3 / 5.2 | 0.82 | 0.04 / 1.26 | 64 / 77 | 0.82 | 27 / 28 | 4.8 / 5.7 | 0.73 / 0.71 | 0.04 / 1.26 | 1.3e-06 | 1.3e-06 |  |
| repvit_m1_1.dist_300e_in1k | fp16 | 3.7 / 4.5 | 0.82 | 0.02 / 1.26 | 42 / 46 | 0.92 | 16 / 17 | 4.1 / 5.1 | 0.80 / 0.71 | 0.02 / 1.26 | 0.0061 | 0.0061 |  |
| repvit_m1_1.dist_300e_in1k | fp32 | 4.0 / 5.0 | 0.79 | 0.04 / 1.21 | 71 / 85 | 0.84 | 32 / 34 | 4.5 / 5.5 | 0.74 / 0.70 | 0.04 / 1.21 | 8.9e-07 | 8.9e-07 |  |
| repvit_m1_1.dist_450e_in1k | fp16 | 5.7 / 4.5 | 1.26 | 0.02 / 1.25 | 42 / 46 | 0.93 | 16 / 17 | 4.1 / 5.0 | 0.79 / 0.71 | 0.02 / 1.25 | 0.021 (5.2 % off) | 0.021 (5.2 % off) | slower |
| repvit_m1_1.dist_450e_in1k | fp32 | 4.0 / 5.0 | 0.80 | 0.04 / 1.19 | 71 / 85 | 0.84 | 32 / 34 | 4.5 / 5.6 | 0.73 / 0.71 | 0.04 / 1.19 | 2.6e-06 | 2.6e-06 |  |
| repvit_m1_5.dist_300e_in1k | fp16 | 6.0 / 7.5 | 0.79 | 0.02 / 1.79 | 73 / 64 | 1.14 | 27 / 28 | 6.6 / 8.2 | 0.86 / 0.71 | 0.03 / 1.79 | 0.0073 | 0.0073 | heavier |
| repvit_m1_5.dist_300e_in1k | fp32 | 6.5 / 8.8 | 0.74 | 0.04 / 1.71 | 121 / 123 | 0.98 | 54 / 56 | 7.0 / 9.9 | 0.77 / 0.71 | 0.04 / 1.71 | 1.2e-06 | 1.2e-06 |  |
| repvit_m1_5.dist_450e_in1k | fp16 | 5.9 / 7.5 | 0.78 | 0.02 / 1.79 | 73 / 64 | 1.14 | 27 / 28 | 6.4 / 8.2 | 0.85 / 0.74 | 0.03 / 1.79 | 0.037 (12 % off) | 0.037 (12 % off) | heavier |
| repvit_m1_5.dist_450e_in1k | fp32 | 6.4 / 8.3 | 0.77 | 0.04 / 1.74 | 121 / 120 | 1.00 | 54 / 56 | 6.9 / 8.9 | 0.77 / 0.77 | 0.05 / 1.74 | 2e-06 | 2e-06 |  |
| repvit_m2_3.dist_300e_in1k | fp16 | 9.7 / 12.1 | 0.80 | 0.03 / 2.31 | 118 / 109 | 1.09 | 45 / 46 | 10.2 / 12.7 | 0.89 / 0.76 | 0.03 / 2.31 | 0.0098 | 0.0098 | heavier |
| repvit_m2_3.dist_300e_in1k | fp32 | 10.8 / 13.8 | 0.79 | 0.05 / 2.26 | 199 / 194 | 1.02 | 88 / 91 | 11.3 / 14.4 | 0.79 / 0.73 | 0.05 / 2.26 | 1.2e-06 | 1.2e-06 | heavier |
| repvit_m2_3.dist_450e_in1k | fp16 | 9.6 / 12.3 | 0.78 | 0.03 / 2.31 | 118 / 110 | 1.07 | 45 / 46 | 10.2 / 12.9 | 0.90 / 0.72 | 0.03 / 2.31 | 0.007 | 0.007 | heavier |
| repvit_m2_3.dist_450e_in1k | fp32 | 10.7 / 13.8 | 0.78 | 0.05 / 2.21 | 199 / 194 | 1.02 | 88 / 91 | 11.3 / 14.5 | 0.79 / 0.72 | 0.05 / 2.21 | 9.2e-07 | 9.2e-07 | heavier |
| resnet152.tv2_in1k | fp16 | 15.7 / 17.4 | 0.90 | 0.04 / 1.46 | 177 / 224 | 0.79 | 115 / 115 | 16.2 / 18.0 | 0.81 / 0.62 | 0.04 / 1.46 | 0.00062 | 0.00062 |  |
| resnet152.tv2_in1k | fp32 | 17.9 / 20.8 | 0.86 | 0.05 / 1.42 | 335 / 384 | 0.87 | 229 / 229 | 18.2 / 21.3 | 0.70 / 0.61 | 0.05 / 1.43 | 4.6e-07 | 4.6e-07 |  |
| resnet18.tv_in1k | fp16 | 3.4 / 3.8 | 0.89 | 0.02 / 0.48 | 30 / 52 | 0.58 | 22 / 22 | 3.8 / 4.3 | 0.53 / 0.49 | 0.02 / 0.48 | 0.00055 | 0.00055 |  |
| resnet18.tv_in1k | fp32 | 3.9 / 4.6 | 0.86 | 0.03 / 0.48 | 59 / 102 | 0.58 | 45 / 45 | 4.3 / 5.0 | 0.50 / 0.49 | 0.03 / 0.48 | 3.7e-07 | 3.7e-07 |  |
| resnet50.tv2_in1k | fp16 | 6.1 / 6.8 | 0.90 | 0.02 / 0.76 | 76 / 99 | 0.77 | 49 / 49 | 6.7 / 7.2 | 0.58 / 0.50 | 0.02 / 0.76 | 0.00082 | 0.00082 |  |
| resnet50.tv2_in1k | fp32 | 7.1 / 8.1 | 0.87 | 0.04 / 0.74 | 147 / 192 | 0.77 | 97 / 97 | 7.4 / 8.5 | 0.54 / 0.51 | 0.04 / 0.74 | 5.5e-07 | 5.5e-07 |  |
| rf-detr-keypoint-preview | fp16 | 62.6 / 61.5 | 1.02 | 0.10 / 3.27 | 354 / 260 | 1.36 | 95 / 71 | 63.6 / 63.2 | 2.08 / 2.01 | 0.10 / 3.28 | - | empty | heavier |
| rf-detr-keypoint-preview | fp32 | 72.3 / 74.0 | 0.98 | 0.11 / 3.16 | 428 / 256 | 1.67 | 138 / 142 | 73.1 / 75.9 | 2.10 / 1.96 | 0.11 / 3.17 | - | empty | heavier |
| rf-detr-l-coco | fp16 | 44.8 / 41.2 | 1.09 | 0.08 / 1.89 | 201 / 191 | 1.05 | 73 / 60 | 47.5 / 44.8 | 1.99 / 1.92 | 0.09 / 1.90 | 0.98 (69 % off) | empty | slower heavier |
| rf-detr-l-coco | fp32 | 52.6 / 51.0 | 1.03 | 0.09 / 1.86 | 268 / 238 | 1.13 | 116 / 120 | 55.2 / 54.6 | 1.99 / 1.90 | 0.09 / 1.86 | 0.42 (3.6 % off) | empty | slower heavier |
| rf-detr-m-coco | fp16 | 31.9 / 29.5 | 1.08 | 0.07 / 1.89 | 179 / 179 | 1.00 | 73 / 59 | 34.5 / 32.6 | 2.03 / 1.91 | 0.08 / 1.90 | 0.9 (70 % off) | empty | slower |
| rf-detr-m-coco | fp32 | 36.2 / 35.2 | 1.03 | 0.08 / 1.84 | 237 / 226 | 1.05 | 115 / 119 | 38.7 / 38.2 | 2.00 / 1.93 | 0.08 / 1.85 | 0.00048 | empty | slower heavier |
| rf-detr-n-coco | fp16 | 16.3 / 15.3 | 1.07 | 0.07 / 1.64 | 113 / 146 | 0.77 | 61 / 53 | 18.9 / 18.3 | 1.97 / 1.93 | 0.07 / 1.65 | 0.93 (51 % off) | empty | slower |
| rf-detr-n-coco | fp32 | 17.9 / 17.5 | 1.03 | 0.07 / 1.57 | 165 / 190 | 0.87 | 102 / 106 | 20.8 / 20.4 | 1.99 / 1.93 | 0.07 / 1.57 | 0.75 (0.56 % off) | empty | slower |
| rf-detr-s-coco | fp16 | 25.4 / 24.0 | 1.06 | 0.07 / 1.77 | 149 / 165 | 0.90 | 67 / 56 | 28.1 / 27.2 | 2.07 / 1.91 | 0.07 / 1.78 | 0.95 (71 % off) | empty | slower |
| rf-detr-s-coco | fp32 | 28.7 / 28.0 | 1.02 | 0.07 / 1.75 | 210 / 219 | 0.96 | 109 / 113 | 31.4 / 31.1 | 2.02 / 1.91 | 0.08 / 1.75 | 0.86 (1.4 % off) | empty | slower |
| rf-detr-seg-2xl-coco | fp16 | 126.2 / 119.9 | 1.05 | 0.17 / 2.32 | 394 / 274 | 1.44 | 88 / 69 | 128.8 / 123.6 | 2.02 / 1.96 | 0.17 / 2.33 | 1 (73 % off) | empty | slower heavier |
| rf-detr-seg-2xl-coco | fp32 | 156.5 / 155.1 | 1.01 | 0.20 / 2.25 | 576 / 397 | 1.45 | 134 / 138 | 159.3 / 159.3 | 2.10 / 1.93 | 0.20 / 2.26 | 0.00069 | empty | heavier |
| rf-detr-seg-l-coco | fp16 | 48.6 / 45.2 | 1.07 | 0.09 / 2.09 | 193 / 222 | 0.87 | 80 / 64 | 50.5 / 47.7 | 2.01 / 1.91 | 0.10 / 2.09 | 0.96 (84 % off) | empty | slower |
| rf-detr-seg-l-coco | fp32 | 57.6 / 56.3 | 1.02 | 0.10 / 2.07 | 289 / 301 | 0.96 | 124 / 127 | 59.7 / 58.8 | 1.99 / 1.93 | 0.10 / 2.07 | 0.00097 | empty | slower |
| rf-detr-seg-m-coco | fp16 | 38.1 / 35.3 | 1.08 | 0.08 / 2.08 | 172 / 212 | 0.81 | 80 / 63 | 40.0 / 37.6 | 1.98 / 1.89 | 0.08 / 2.09 | 0.93 (86 % off) | empty | slower |
| rf-detr-seg-m-coco | fp32 | 43.9 / 42.4 | 1.04 | 0.09 / 2.04 | 254 / 289 | 0.88 | 124 / 126 | 45.7 / 44.9 | 2.02 / 1.89 | 0.09 / 2.05 | 0.0041 | empty | slower |
| rf-detr-seg-n-coco | fp16 | 22.5 / 21.2 | 1.06 | 0.07 / 1.96 | 127 / 150 | 0.85 | 74 / 59 | 23.6 / 23.1 | 1.97 / 1.93 | 0.07 / 1.96 | 0.94 (77 % off) | empty | slower |
| rf-detr-seg-n-coco | fp32 | 25.9 / 24.6 | 1.05 | 0.07 / 1.92 | 190 / 250 | 0.76 | 117 / 118 | 26.7 / 25.8 | 1.99 / 1.96 | 0.08 / 1.93 | 0.00012 | empty | slower |
| rf-detr-seg-s-coco | fp16 | 29.2 / 27.1 | 1.08 | 0.07 / 1.98 | 149 / 177 | 0.84 | 74 / 59 | 30.5 / 28.8 | 1.96 / 1.89 | 0.08 / 1.99 | 0.93 (79 % off) | empty | slower |
| rf-detr-seg-s-coco | fp32 | 33.0 / 31.6 | 1.04 | 0.08 / 1.93 | 223 / 267 | 0.84 | 117 / 119 | 34.3 / 33.4 | 1.99 / 1.90 | 0.08 / 1.93 | 0.001 | empty | slower |
| rf-detr-seg-xl-coco | fp16 | 79.0 / 75.0 | 1.05 | 0.12 / 2.21 | 252 / 251 | 1.00 | 87 / 68 | 81.6 / 78.2 | 2.04 / 1.90 | 0.13 / 2.21 | 0.99 (79 % off) | empty | slower |
| rf-detr-seg-xl-coco | fp32 | 96.2 / 95.0 | 1.01 | 0.14 / 2.19 | 377 / 378 | 1.00 | 132 / 136 | 98.7 / 98.4 | 2.00 / 1.93 | 0.14 / 2.20 | 0.00081 | empty |  |
| rtdetrv2-l-coco | fp16 | 46.9 / 52.8 | 0.89 | 0.07 / 2.36 | 264 / 259 | 1.02 | 83 / 83 | 54.6 / 60.8 | 0.29 / 0.12 | 0.08 / 2.38 | 0.95 (61 % off) | 0.84 (3.3 % off) |  |
| rtdetrv2-l-coco | fp32 | 55.3 / 63.4 | 0.87 | 0.09 / 2.19 | 455 / 424 | 1.07 | 161 / 161 | 63.0 / 71.6 | 0.20 / 0.12 | 0.10 / 2.20 | 1.3e-05 | 1.5e-06 | heavier |
| rtdetrv2-m-coco | fp16 | 35.7 / 39.2 | 0.91 | 0.06 / 2.00 | 224 / 258 | 0.87 | 62 / 62 | 43.8 / 47.3 | 0.23 / 0.10 | 0.07 / 2.02 | 0.97 (45 % off) | empty (3 keys differ) |  |
| rtdetrv2-m-coco | fp32 | 42.0 / 47.0 | 0.89 | 0.08 / 1.89 | 418 / 453 | 0.92 | 119 / 119 | 49.8 / 54.8 | 0.16 / 0.10 | 0.09 / 1.90 | 4.1e-05 | 1.8e-06 |  |
| rtdetrv2-ms-coco | fp16 | 37.4 / 42.2 | 0.89 | 0.06 / 1.98 | 246 / 242 | 1.02 | 65 / 65 | 45.3 / 50.1 | 0.25 / 0.12 | 0.07 / 2.00 | 0.98 (52 % off) | 0.63 (2.5 % off) |  |
| rtdetrv2-ms-coco | fp32 | 44.5 / 51.5 | 0.86 | 0.08 / 1.91 | 412 / 391 | 1.05 | 126 / 126 | 51.8 / 59.2 | 0.17 / 0.11 | 0.09 / 1.92 | 1.1e-05 | 1.3e-06 | heavier |
| rtdetrv2-s-coco | fp16 | 24.8 / 27.4 | 0.90 | 0.05 / 1.88 | 150 / 244 | 0.61 | 40 / 40 | 33.1 / 35.1 | 0.19 / 0.08 | 0.06 / 1.90 | 0.97 (29 % off) | empty (3 keys differ) |  |
| rtdetrv2-s-coco | fp32 | 28.7 / 33.3 | 0.86 | 0.07 / 1.72 | 271 / 362 | 0.75 | 77 / 77 | 40.2 / 41.5 | 0.13 / 0.08 | 0.08 / 1.73 | 1.6e-05 | 1.1e-05 |  |
| rtdetrv2-x-coco | fp16 | 79.4 / 86.6 | 0.92 | 0.17 / 2.54 | 374 / 389 | 0.96 | 146 / 146 | 88.5 / 94.9 | 0.38 / 0.17 | 0.18 / 2.55 | 0.98 (37 % off) | 0.42 (1.6 % off) |  |
| rtdetrv2-x-coco | fp32 | 92.5 / 103.2 | 0.90 | 0.14 / 2.40 | 736 / 679 | 1.08 | 286 / 286 | 100.3 / 110.9 | 0.28 / 0.16 | 0.14 / 2.41 | 2.4e-05 | 4.5e-07 | heavier |
| rtmdet-l-coco | fp16 | 52.2 / 63.1 | 0.83 | 0.08 / 1.60 | 292 / 307 | 0.95 | 112 / 109 | 58.0 / 68.3 | 0.05 / 0.17 | 0.55 / 1.67 | 0.7 (68 % off) | empty (3 keys differ) |  |
| rtmdet-l-coco | fp32 | 61.2 / 77.5 | 0.79 | 0.09 / 1.54 | 554 / 573 | 0.97 | 218 / 218 | 67.2 / 82.8 | 0.06 / 0.17 | 0.56 / 1.61 | 0.7 (68 % off) | 4.3e-06 |  |
| rtmdet-m-coco | fp16 | 33.5 / 39.7 | 0.84 | 0.07 / 1.40 | 179 / 242 | 0.74 | 54 / 53 | 39.3 / 45.2 | 0.04 / 0.12 | 0.54 / 1.47 | 0.81 (62 % off) | 0.0018 |  |
| rtmdet-m-coco | fp32 | 37.1 / 46.2 | 0.80 | 0.07 / 1.37 | 326 / 379 | 0.86 | 105 / 105 | 43.0 / 51.7 | 0.04 / 0.12 | 0.54 / 1.45 | 0.81 (62 % off) | 1.1e-06 |  |
| rtmdet-m-person | fp16 | 33.4 / 39.3 | 0.85 | 0.07 / 1.41 | 179 / 240 | 0.75 | 54 / 53 | 37.9 / 43.8 | 0.04 / 0.08 | 0.08 / 1.42 | 0.046 (0.13 % off) | empty |  |
| rtmdet-m-person | fp32 | 37.0 / 46.2 | 0.80 | 0.08 / 1.39 | 318 / 379 | 0.84 | 104 / 104 | 41.8 / 50.6 | 0.04 / 0.09 | 0.09 / 1.39 | 2e-05 | empty |  |
| rtmdet-nano-person | fp16 | 4.0 / 6.4 | 0.63 | 0.04 / 1.37 | 14 / 32 | 0.46 | 2 / 2 | 6.7 / 8.5 | 0.04 / 0.04 | 0.50 / 1.43 | 0.062 (1.3 % off) | 0.4 (14 % off) |  |
| rtmdet-nano-person | fp32 | 4.2 / 6.9 | 0.62 | 0.05 / 1.28 | 23 / 54 | 0.43 | 4 / 4 | 6.1 / 9.1 | 0.04 / 0.04 | 0.50 / 1.35 | 6.3e-06 | 1e-06 |  |
| rtmdet-s-coco | fp16 | 14.3 / 19.0 | 0.75 | 0.05 / 1.25 | 96 / 216 | 0.44 | 20 / 19 | 20.0 / 24.1 | 0.04 / 0.08 | 0.53 / 1.31 | 0.75 (67 % off) | 0.022 (17 % off) |  |
| rtmdet-s-coco | fp32 | 16.0 / 22.8 | 0.70 | 0.06 / 1.22 | 181 / 243 | 0.74 | 38 / 38 | 21.8 / 27.9 | 0.04 / 0.08 | 0.52 / 1.29 | 0.76 (67 % off) | 2.6e-06 |  |
| rtmdet-tiny-coco | fp16 | 12.9 / 16.1 | 0.80 | 0.05 / 1.21 | 66 / 180 | 0.37 | 12 / 11 | 18.7 / 21.3 | 0.04 / 0.07 | 0.51 / 1.28 | 0.72 (67 % off) | 0.55 (1.8 % off) |  |
| rtmdet-tiny-coco | fp32 | 15.1 / 18.9 | 0.80 | 0.05 / 1.18 | 105 / 262 | 0.40 | 22 / 22 | 21.2 / 24.5 | 0.04 / 0.07 | 0.51 / 1.25 | 0.73 (66 % off) | 1.6e-06 |  |
| rtmdet-x-coco | fp16 | 102.2 / 111.6 | 0.92 | 0.13 / 1.79 | 421 / 470 | 0.90 | 200 / 196 | 107.8 / 116.9 | 0.06 / 0.20 | 0.60 / 1.86 | 0.79 (65 % off) | empty (3 keys differ) |  |
| rtmdet-x-coco | fp32 | 116.5 / 131.7 | 0.88 | 0.17 / 1.74 | 808 / 913 | 0.89 | 390 / 390 | 122.6 / 137.4 | 0.06 / 0.21 | 0.63 / 1.81 | 0.79 (65 % off) | 1.9e-06 |  |
| rtmo-l | fp16 | 47.0 / 55.7 | 0.84 | 0.07 / 1.56 | 401 / 414 | 0.97 | 82 / 82 | 52.9 / 61.4 | 0.27 / 0.14 | 0.07 / 1.57 | 0.011 (0.00028 % off) | empty |  |
| rtmo-l | fp32 | 52.4 / 64.7 | 0.81 | 0.08 / 1.51 | 537 / 527 | 1.02 | 163 / 163 | 58.5 / 70.4 | 0.20 / 0.14 | 0.09 / 1.52 | 1e-05 | empty |  |
| rtmo-m | fp16 | 27.7 / 34.7 | 0.80 | 0.06 / 1.43 | 165 / 206 | 0.80 | 41 / 41 | 33.6 / 40.0 | 0.20 / 0.09 | 0.06 / 1.44 | 0.0034 | empty |  |
| rtmo-m | fp32 | 30.1 / 39.1 | 0.77 | 0.07 / 1.39 | 278 / 313 | 0.89 | 82 / 82 | 35.8 / 44.2 | 0.13 / 0.09 | 0.08 / 1.40 | 1.2e-05 | empty |  |
| rtmo-s | fp16 | 13.3 / 17.3 | 0.77 | 0.05 / 1.30 | 82 / 199 | 0.41 | 18 / 18 | 18.8 / 22.5 | 0.14 / 0.06 | 0.05 / 1.31 | 0.017 (0.0018 % off) | empty |  |
| rtmo-s | fp32 | 14.7 / 20.2 | 0.73 | 0.06 / 1.28 | 128 / 226 | 0.56 | 35 / 35 | 20.3 / 25.3 | 0.09 / 0.06 | 0.07 / 1.29 | 1.4e-05 | empty |  |
| rtmo-t | fp16 | 7.2 / 9.6 | 0.75 | 0.04 / 1.29 | 35 / 88 | 0.40 | 12 / 12 | 11.6 / 14.0 | 0.13 / 0.06 | 0.05 / 1.30 | 0.015 (0.0011 % off) | empty |  |
| rtmo-t | fp32 | 7.7 / 10.6 | 0.73 | 0.05 / 1.26 | 65 / 155 | 0.42 | 24 / 24 | 12.3 / 14.8 | 0.08 / 0.06 | 0.06 / 1.26 | 1.9e-05 | empty |  |
| rtmpose-l | fp16 | 48.6 / 60.8 | 0.80 | 0.07 / 1.34 | 310 / 267 | 1.16 | 58 / 56 | 57.0 / 68.8 | 0.19 / 0.08 | 0.08 / 1.35 | - | 0.17 (0.36 % off) | heavier |
| rtmpose-l | fp32 | 54.5 / 73.8 | 0.74 | 0.08 / 1.30 | 565 / 431 | 1.31 | 106 / 106 | 63.4 / 81.9 | 0.14 / 0.08 | 0.09 / 1.31 | - | 7.2e-06 | heavier |
| rtmpose-l-384 | fp16 | 95.3 / 118.9 | 0.80 | 0.11 / 1.38 | 387 / 354 | 1.09 | 59 / 56 | 111.7 / 135.1 | 0.19 / 0.08 | 0.13 / 1.40 | - | 0.45 (12 % off) | heavier |
| rtmpose-l-384 | fp32 | 107.9 / 149.1 | 0.72 | 0.13 / 1.40 | 743 / 496 | 1.50 | 106 / 106 | 125.2 / 164.2 | 0.14 / 0.08 | 0.15 / 1.41 | - | 6.3e-06 | heavier |
| rtmpose-l-halpe26 | fp16 | 48.6 / 61.0 | 0.80 | 0.07 / 1.32 | 311 / 269 | 1.16 | 60 / 58 | 57.1 / 68.6 | 0.19 / 0.08 | 0.08 / 1.33 | - | 0.047 (0.08 % off) | heavier |
| rtmpose-l-halpe26 | fp32 | 54.6 / 73.9 | 0.74 | 0.09 / 1.34 | 566 / 433 | 1.31 | 107 / 107 | 63.6 / 81.8 | 0.14 / 0.09 | 0.10 / 1.35 | - | 6.4e-06 | heavier |
| rtmpose-l-halpe26-384 | fp16 | 95.6 / 118.8 | 0.80 | 0.11 / 1.39 | 389 / 356 | 1.09 | 61 / 58 | 112.1 / 134.5 | 0.20 / 0.08 | 0.13 / 1.41 | - | 0.28 (4.8 % off) | heavier |
| rtmpose-l-halpe26-384 | fp32 | 108.4 / 149.3 | 0.73 | 0.13 / 1.40 | 745 / 510 | 1.46 | 108 / 108 | 125.4 / 165.5 | 0.14 / 0.08 | 0.15 / 1.42 | - | 9.5e-06 | heavier |
| rtmpose-m | fp16 | 34.2 / 39.9 | 0.86 | 0.05 / 1.22 | 211 / 220 | 0.96 | 30 / 29 | 42.5 / 47.3 | 0.13 / 0.06 | 0.06 / 1.23 | - | 0.18 (2.5 % off) |  |
| rtmpose-m | fp32 | 38.2 / 47.5 | 0.80 | 0.07 / 1.19 | 383 / 412 | 0.93 | 52 / 52 | 46.5 / 55.3 | 0.09 / 0.06 | 0.08 / 1.20 | - | 9e-06 |  |
| rtmpose-m-384 | fp16 | 67.9 / 78.9 | 0.86 | 0.09 / 1.27 | 251 / 371 | 0.68 | 31 / 29 | 84.9 / 94.3 | 0.13 / 0.06 | 0.11 / 1.29 | - | 0.042 (1.5 % off) |  |
| rtmpose-m-384 | fp32 | 79.6 / 98.4 | 0.81 | 0.11 / 1.25 | 440 / 622 | 0.71 | 53 / 53 | 96.7 / 114.1 | 0.09 / 0.06 | 0.13 / 1.27 | - | 5.2e-06 |  |
| rtmpose-m-ap10k | fp16 | 42.6 / 49.9 | 0.85 | 0.06 / 1.22 | 228 / 232 | 0.98 | 30 / 29 | 52.9 / 59.4 | 0.13 / 0.06 | 0.07 / 1.23 | - | 0.019 (0.12 % off) |  |
| rtmpose-m-ap10k | fp32 | 48.8 / 61.2 | 0.80 | 0.07 / 1.20 | 402 / 435 | 0.92 | 52 / 52 | 59.2 / 70.9 | 0.09 / 0.06 | 0.09 / 1.21 | - | 2.3e-06 |  |
| rtmpose-m-face6 | fp16 | 48.5 / 52.2 | 0.93 | 0.06 / 1.23 | 241 / 244 | 0.98 | 43 / 42 | 58.9 / 61.8 | 0.14 / 0.06 | 0.08 / 1.24 | - | 0.2 (0.23 % off) |  |
| rtmpose-m-face6 | fp32 | 54.2 / 63.2 | 0.86 | 0.08 / 1.21 | 414 / 448 | 0.93 | 65 / 65 | 65.3 / 72.8 | 0.10 / 0.06 | 0.10 / 1.23 | - | 2.1e-06 |  |
| rtmpose-m-halpe26 | fp16 | 34.2 / 39.9 | 0.86 | 0.05 / 1.20 | 212 / 222 | 0.96 | 32 / 30 | 42.8 / 47.5 | 0.15 / 0.06 | 0.06 / 1.21 | - | 0.45 (1.8 % off) |  |
| rtmpose-m-halpe26 | fp32 | 38.4 / 47.7 | 0.81 | 0.07 / 1.18 | 384 / 414 | 0.93 | 53 / 53 | 46.7 / 55.3 | 0.09 / 0.06 | 0.08 / 1.19 | - | 9.1e-06 |  |
| rtmpose-m-halpe26-384 | fp16 | 68.1 / 78.6 | 0.87 | 0.09 / 1.26 | 252 / 372 | 0.68 | 32 / 30 | 85.5 / 94.1 | 0.13 / 0.06 | 0.10 / 1.28 | - | 0.43 (2.2 % off) |  |
| rtmpose-m-halpe26-384 | fp32 | 80.3 / 98.8 | 0.81 | 0.11 / 1.25 | 441 / 624 | 0.71 | 54 / 54 | 96.6 / 114.4 | 0.09 / 0.06 | 0.13 / 1.27 | - | 6e-06 |  |
| rtmpose-m-hand5 | fp16 | 42.8 / 50.3 | 0.85 | 0.06 / 1.23 | 228 / 232 | 0.98 | 31 / 29 | 53.1 / 59.6 | 0.14 / 0.06 | 0.07 / 1.24 | - | 0.0025 |  |
| rtmpose-m-hand5 | fp32 | 49.1 / 61.4 | 0.80 | 0.07 / 1.24 | 402 / 436 | 0.92 | 53 / 53 | 59.4 / 71.2 | 0.09 / 0.06 | 0.09 / 1.25 | - | 3.1e-06 |  |
| rtmpose-s | fp16 | 13.7 / 19.0 | 0.72 | 0.03 / 1.11 | 162 / 190 | 0.85 | 13 / 13 | 21.9 / 26.6 | 0.09 / 0.04 | 0.04 / 1.12 | - | 0.16 (1.1 % off) |  |
| rtmpose-s | fp32 | 14.5 / 22.0 | 0.66 | 0.04 / 1.09 | 248 / 245 | 1.01 | 21 / 21 | 22.9 / 30.3 | 0.06 / 0.04 | 0.06 / 1.10 | - | 7.3e-06 |  |
| rtmpose-s-face6 | fp16 | 20.6 / 24.8 | 0.83 | 0.04 / 1.09 | 204 / 179 | 1.14 | 22 / 21 | 30.9 / 34.1 | 0.10 / 0.05 | 0.05 / 1.11 | - | 0.47 (0.21 % off) | heavier |
| rtmpose-s-face6 | fp32 | 21.9 / 29.0 | 0.76 | 0.05 / 1.10 | 329 / 248 | 1.33 | 30 / 30 | 32.3 / 38.5 | 0.07 / 0.04 | 0.06 / 1.12 | - | 2.7e-06 | heavier |
| rtmpose-s-halpe26 | fp16 | 13.8 / 19.2 | 0.72 | 0.03 / 1.11 | 165 / 191 | 0.86 | 14 / 14 | 22.1 / 27.0 | 0.09 / 0.04 | 0.04 / 1.13 | - | 0.4 (6.3 % off) |  |
| rtmpose-s-halpe26 | fp32 | 14.6 / 22.1 | 0.66 | 0.04 / 1.08 | 251 / 246 | 1.02 | 22 / 22 | 22.9 / 29.6 | 0.06 / 0.04 | 0.06 / 1.09 | - | 0.001 | heavier |
| rtmpose-t | fp16 | 12.8 / 16.0 | 0.80 | 0.03 / 1.06 | 125 / 189 | 0.66 | 9 / 8 | 21.2 / 23.6 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.29 (2.7 % off) |  |
| rtmpose-t | fp32 | 13.8 / 18.8 | 0.73 | 0.04 / 1.07 | 188 / 275 | 0.68 | 13 / 13 | 22.0 / 26.4 | 0.05 / 0.04 | 0.05 / 1.08 | - | 8.9e-06 |  |
| rtmpose-t-face6 | fp16 | 18.8 / 21.0 | 0.90 | 0.04 / 1.06 | 188 / 195 | 0.96 | 15 / 15 | 29.2 / 30.3 | 0.09 / 0.04 | 0.05 / 1.07 | - | 0.24 (0.35 % off) |  |
| rtmpose-t-face6 | fp32 | 20.1 / 24.5 | 0.82 | 0.04 / 1.06 | 270 / 288 | 0.94 | 19 / 19 | 30.4 / 33.7 | 0.06 / 0.04 | 0.06 / 1.08 | - | 3.8e-06 |  |
| rtmpose-t-halpe26 | fp16 | 12.9 / 16.0 | 0.81 | 0.03 / 1.06 | 128 / 190 | 0.68 | 9 / 9 | 21.4 / 23.8 | 0.08 / 0.04 | 0.04 / 1.07 | - | 0.45 (2.9 % off) |  |
| rtmpose-t-halpe26 | fp32 | 13.8 / 18.8 | 0.73 | 0.04 / 1.07 | 191 / 276 | 0.69 | 14 / 14 | 22.2 / 26.7 | 0.05 / 0.04 | 0.05 / 1.08 | - | 6.9e-06 |  |
| rtmpose-x-halpe26-384 | fp16 | 198.3 / 214.5 | 0.92 | 0.21 / 1.59 | 553 / 658 | 0.84 | 105 / 100 | 216.4 / 231.0 | 0.26 / 0.12 | 0.23 / 1.61 | - | 0.19 (0.48 % off) |  |
| rtmpose-x-halpe26-384 | fp32 | 259.0 / 276.7 | 0.94 | 0.28 / 1.61 | 1006 / 932 | 1.08 | 191 / 191 | 277.8 / 292.3 | 0.19 / 0.11 | 0.31 / 1.63 | - | 0.00078 | heavier |
| rtmw-l | fp16 | 101.4 / 108.8 | 0.93 | 0.12 / 1.61 | 393 / 350 | 1.12 | 142 / 139 | 110.2 / 116.9 | 0.28 / 0.11 | 0.13 / 1.63 | - | 0.21 (0.13 % off) (2 keys differ) | heavier |
| rtmw-l | fp32 | 111.7 / 125.7 | 0.89 | 0.14 / 1.65 | 678 / 544 | 1.25 | 218 / 218 | 120.6 / 134.8 | 0.21 / 0.11 | 0.16 / 1.67 | - | 3.9e-06 (2 keys differ) | heavier |
| rtmw-l-384 | fp16 | 206.4 / 235.8 | 0.88 | 0.23 / 1.74 | 470 / 438 | 1.07 | 142 / 140 | 224.3 / 251.8 | 0.28 / 0.11 | 0.25 / 1.75 | - | 0.43 (0.51 % off) (2 keys differ) | heavier |
| rtmw-l-384 | fp32 | 224.6 / 272.5 | 0.82 | 0.25 / 1.74 | 856 / 634 | 1.35 | 219 / 219 | 242.3 / 288.4 | 0.21 / 0.11 | 0.27 / 1.76 | - | 0.00078 (2 keys differ) | heavier |
| rtmw-m | fp16 | 71.2 / 72.4 | 0.98 | 0.09 / 1.49 | 278 / 275 | 1.01 | 85 / 84 | 80.4 / 80.4 | 0.20 / 0.08 | 0.10 / 1.50 | - | 0.45 (0.13 % off) (2 keys differ) |  |
| rtmw-m | fp32 | 76.8 / 82.4 | 0.93 | 0.10 / 1.54 | 468 / 484 | 0.97 | 123 / 123 | 86.3 / 90.6 | 0.14 / 0.09 | 0.12 / 1.56 | - | 2.3e-06 (2 keys differ) |  |
| rtmw-x | fp16 | 176.2 / 206.2 | 0.85 | 0.20 / 1.81 | 560 / 474 | 1.18 | 219 / 215 | 185.8 / 216.2 | 0.41 / 0.18 | 0.20 / 1.83 | - | 0.28 (0.21 % off) (2 keys differ) | heavier |
| rtmw-x | fp32 | 203.7 / 240.3 | 0.85 | 0.23 / 1.82 | 968 / 830 | 1.17 | 351 / 351 | 213.3 / 247.0 | 0.33 / 0.18 | 0.24 / 1.83 | - | 5.9e-06 (2 keys differ) | heavier |
| rtmw-x-384 | fp16 | 350.0 / 414.6 | 0.84 | 0.37 / 2.04 | 668 / 773 | 0.86 | 220 / 216 | 368.7 / 425.3 | 0.41 / 0.18 | 0.39 / 2.06 | - | 0.2 (5.8 % off) (2 keys differ) |  |
| rtmw-x-384 | fp32 | 419.5 / 490.0 | 0.86 | 0.45 / 2.08 | 1174 / 1093 | 1.07 | 352 / 352 | 437.4 / 503.6 | 0.35 / 0.18 | 0.47 / 2.11 | - | 1.3e-05 (2 keys differ) | heavier |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 0.33 / 2.21 | 622 / 539 | 1.15 | 179 / 181 | 359.0 / 435.2 | 0.18 / 0.06 | 0.40 / 2.29 | 0.0047 | 1 (0.39 % off) | heavier |
| sam-hq-vit-b | fp32 | 370.9 / 484.5 | 0.77 | 0.40 / 2.29 | 879 / 1066 | 0.82 | 354 / 362 | 418.1 / 536.4 | 0.14 / 0.05 | 0.47 / 2.35 | 6.7e-06 | 1 (0.00081 % off) |  |
| sam-hq-vit-h | fp16 | 1538.9 / 1702.4 | 0.90 | 1.56 / 5.10 | 1762 / 1624 | 1.08 | 1227 / 1228 | 1592.5 / 1750.7 | 0.64 / 0.09 | 1.65 / 5.21 | 0.0049 | 1 (0.0077 % off) | heavier |
| sam-hq-vit-h | fp32 | 1881.1 / 2133.4 | 0.88 | 1.96 / 5.79 | 3059 / 3242 | 0.94 | 2449 / 2458 | 1947.3 / 2149.4 | 0.96 / 0.09 | 2.30 / 6.05 | 4.2e-06 | 2.8e-06 |  |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 0.84 / 3.70 | 1119 / 982 | 1.14 | 597 / 599 | 880.6 / 992.8 | 0.38 / 0.07 | 0.92 / 3.78 | 0.0029 | 1 (0.0052 % off) | heavier |
| sam-hq-vit-l | fp32 | 976.4 / 1120.5 | 0.87 | 1.01 / 3.87 | 1778 / 1953 | 0.91 | 1190 / 1199 | 1027.3 / 1167.4 | 0.32 / 0.07 | 1.09 / 3.95 | 3.5e-06 | 4.2e-06 |  |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 0.32 / 2.10 | 613 / 531 | 1.15 | 175 / 179 | 323.5 / 406.6 | 0.18 / 0.05 | 0.35 / 2.13 | 0.0047 | 1 (0.015 % off) | heavier |
| sam-vit-b | fp32 | 365.7 / 480.0 | 0.76 | 0.40 / 2.21 | 865 / 1050 | 0.82 | 351 / 359 | 389.8 / 505.0 | 0.14 / 0.05 | 0.43 / 2.23 | 6.7e-06 | 1 (0.00057 % off) |  |
| sam-vit-h | fp16 | 1528.6 / 1692.6 | 0.90 | 1.54 / 4.99 | 1745 / 1611 | 1.08 | 1222 / 1226 | 1553.0 / 1715.7 | 0.62 / 0.08 | 1.58 / 5.02 | 0.0049 | 1 (0.013 % off) | heavier |
| sam-vit-h | fp32 | 1873.2 / 2106.1 | 0.89 | 2.26 / 5.74 | 3034 / 3210 | 0.94 | 2443 / 2452 | 1905.9 / 2109.5 | 1.39 / 0.14 | 2.31 / 5.79 | 4.2e-06 | 1 (3e-05 % off) |  |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 0.83 / 3.62 | 1107 / 972 | 1.14 | 593 / 597 | 837.3 / 954.6 | 0.38 / 0.07 | 0.86 / 3.65 | 0.0029 | 1 (0.00033 % off) | heavier |
| sam-vit-l | fp32 | 972.1 / 1110.9 | 0.88 | 1.00 / 3.88 | 1758 / 1933 | 0.91 | 1186 / 1195 | 995.8 / 1131.8 | 0.32 / 0.07 | 1.04 / 3.91 | 3.5e-06 | 3.1e-06 |  |
| sam2.1-hiera-base-plus | fp16 | 232.8 / 245.5 | 0.95 | 0.24 / 2.21 | 535 / 478 | 1.12 | 153 / 155 | 250.9 / 264.3 | 0.30 / 0.08 | 0.27 / 2.24 | 0.0049 | 1 (0.002 % off) | heavier |
| sam2.1-hiera-base-plus | fp32 | 256.2 / 268.7 | 0.95 | 0.28 / 2.22 | 871 / 808 | 1.08 | 303 / 300 | 275.4 / 290.0 | 0.23 / 0.08 | 0.31 / 2.24 | 1.3e-05 | 8.9e-06 | heavier |
| sam2.1-hiera-large | fp16 | 555.8 / 562.2 | 0.99 | 0.57 / 3.55 | 849 / 734 | 1.16 | 426 / 434 | 574.5 / 583.0 | 0.60 / 0.11 | 0.59 / 3.58 | 0.026 (9.5e-05 % off) | 1 (0.0013 % off) | heavier |
| sam2.1-hiera-large | fp32 | 600.5 / 608.8 | 0.99 | 0.63 / 3.55 | 1495 / 1317 | 1.14 | 847 / 855 | 619.5 / 628.7 | 0.53 / 0.10 | 0.66 / 3.58 | 0.00018 | 5.9e-05 | heavier |
| sam2.1-hiera-small | fp16 | 133.8 / 138.1 | 0.97 | 0.14 / 1.77 | 390 / 362 | 1.08 | 78 / 86 | 153.6 / 157.4 | 0.20 / 0.07 | 0.17 / 1.80 | 0.01 (4.8e-05 % off) | 1 (0.012 % off) | heavier |
| sam2.1-hiera-small | fp32 | 155.0 / 152.5 | 1.02 | 0.18 / 1.78 | 599 / 463 | 1.29 | 155 / 163 | 174.3 / 171.4 | 0.15 / 0.07 | 0.21 / 1.81 | 4.7e-06 | 1 (3e-05 % off) | heavier |
| sam2.1-hiera-tiny | fp16 | 112.1 / 116.0 | 0.97 | 0.12 / 1.58 | 377 / 349 | 1.08 | 65 / 73 | 130.9 / 134.3 | 0.18 / 0.07 | 0.15 / 1.61 | 0.0077 | 1 (0.027 % off) | heavier |
| sam2.1-hiera-tiny | fp32 | 130.7 / 127.7 | 1.02 | 0.15 / 1.58 | 572 / 436 | 1.31 | 128 / 136 | 149.2 / 146.9 | 0.13 / 0.07 | 0.18 / 1.60 | 6.4e-06 | 1.1e-05 | slower heavier |
| sam3 | fp16 | 1574.1 / 1439.5 | 1.09 | 1.60 / 5.10 | 2006 / 1385 | 1.45 | 1633 / 892 | 1590.7 / 2235.5 | 0.18 / 1.01 | 1.65 / 7.30 | 0.21 (0.84 % off) | empty | slower heavier |
| sam3 | fp32 | 1896.0 / 1741.0 | 1.09 | 2.48 / 5.54 | 3972 / 2551 | 1.56 | 3205 / 1776 | 1909.2 / 2695.2 | 4.13 / 1.00 | 2.58 / 11.98 | 0.19 (0.7 % off) | empty | slower heavier |
| sapiens-seg-0.3b-goliath | fp16 | 1140.6 / 1508.4 | 0.76 | 1.20 / 3.15 | 1822 / 1808 | 1.01 | 648 / 648 | 1145.6 / 1833.1 | 0.31 / 0.07 | 1.22 / 3.48 | 0.003 | 0 |  |
| sapiens-seg-0.3b-goliath | fp32 | 1415.1 / 1781.7 | 0.79 | 1.78 / 3.57 | 3631 / 3288 | 1.10 | 1295 / 1295 | 1435.5 / 2108.3 | 5.03 / 0.07 | 1.81 / 3.91 | 2.6e-06 | 0 | heavier |
| sapiens-seg-1b-goliath | fp16 | 2648.0 / 3081.9 | 0.86 | 2.77 / 7.32 | 3426 / 3730 | 0.92 | 2248 / 2248 | 2652.4 / 3322.1 | 17.03 / 0.17 | 2.79 / 7.68 | 0.002 | 0 |  |
| sapiens-seg-1b-goliath | fp32 | 3399.9 / 3701.9 | 0.92 | 3.81 / 8.79 | 6839 / 6491 | 1.05 | 4497 / 4497 | 3196.7 / 4086.3 | 21.94 / 0.33 | 3.83 / 9.17 | 2.1e-06 | 0 | heavier |
| segformer-b0-ade20k | fp16 | 32.9 / 17.3 | 1.91 | 0.07 / 1.02 | 346 / 114 | 3.04 | 7 / 7 | 63.0 / 54.0 | 2.05 / 2.02 | 0.10 / 1.06 | 0.0011 | 0.5 (0.068 % off) | slower heavier |
| segformer-b0-ade20k | fp32 | 33.6 / 23.1 | 1.46 | 0.09 / 1.01 | 373 / 149 | 2.50 | 14 / 14 | 64.9 / 59.7 | 1.99 / 2.11 | 0.12 / 1.06 | 8.3e-07 | 0.5 (4.8e-05 % off) | slower heavier |
| segformer-b2-clothes | fp16 | 43.7 / 24.4 | 1.79 | 0.10 / 1.44 | 517 / 158 | 3.27 | 48 / 48 | 44.2 / 25.6 | 2.13 / 2.08 | 0.10 / 1.44 | 0.00082 | 0 | slower heavier |
| segformer-b2-clothes | fp32 | 46.1 / 32.2 | 1.43 | 0.17 / 1.48 | 589 / 252 | 2.34 | 96 / 96 | 46.8 / 32.6 | 2.21 / 2.20 | 0.18 / 1.49 | 6.4e-07 | 0 | slower heavier |
| segformer-b5-ade20k | fp16 | 278.2 / 206.5 | 1.35 | 0.32 / 3.47 | 1287 / 555 | 2.32 | 158 / 158 | 310.4 / 249.8 | 2.33 / 2.20 | 0.36 / 3.52 | 0.0011 | 1 (0.014 % off) | slower heavier |
| segformer-b5-ade20k | fp32 | 311.6 / 262.6 | 1.19 | 0.36 / 3.46 | 1625 / 780 | 2.08 | 314 / 314 | 344.3 / 305.4 | 2.29 / 2.27 | 0.41 / 3.51 | 7.1e-07 | 0 | slower heavier |
| segformer-b5-face-parsing | fp16 | 83.9 / 62.6 | 1.34 | 0.15 / 3.34 | 627 / 274 | 2.29 | 158 / 158 | 84.6 / 64.2 | 2.34 / 2.16 | 0.15 / 3.35 | 0.00083 | 0 | slower heavier |
| segformer-b5-face-parsing | fp32 | 92.6 / 79.0 | 1.17 | 0.32 / 3.31 | 808 / 517 | 1.56 | 314 / 314 | 93.6 / 80.5 | 2.54 / 2.27 | 0.32 / 3.31 | 5.4e-07 | 0 | slower heavier |
| segnext-mscan-b-ade20k | fp16 | 80.6 / 70.3 | 1.15 | 0.11 / 2.19 | 396 / 251 | 1.58 | 106 / 53 | 114.1 / 103.2 | 0.16 / 0.12 | 0.15 / 2.23 | 0.0076 | 0 | slower heavier |
| segnext-mscan-b-ade20k | fp32 | 87.4 / 87.8 | 1.00 | 0.12 / 2.12 | 528 / 366 | 1.44 | 106 / 106 | 121.4 / 120.7 | 0.15 / 0.12 | 0.16 / 2.16 | 4.2e-06 | 0 | heavier |
| segnext-mscan-l-ade20k | fp16 | 143.2 / 127.0 | 1.13 | 0.18 / 2.97 | 519 / 305 | 1.70 | 187 / 94 | 176.9 / 160.2 | 0.25 / 0.19 | 0.22 / 3.01 | 0.004 | 0.5 (0.0021 % off) | slower heavier |
| segnext-mscan-l-ade20k | fp32 | 155.4 / 158.8 | 0.98 | 0.19 / 2.81 | 609 / 497 | 1.23 | 187 / 187 | 189.2 / 191.5 | 0.25 / 0.19 | 0.23 / 2.85 | 5.3e-06 | 0 | heavier |
| segnext-mscan-s-ade20k | fp16 | 44.8 / 37.4 | 1.20 | 0.08 / 1.65 | 310 / 201 | 1.54 | 53 / 27 | 78.2 / 70.4 | 0.10 / 0.07 | 0.11 / 1.69 | 0.0022 | 0.12 (0.2 % off) | slower heavier |
| segnext-mscan-s-ade20k | fp32 | 49.0 / 46.8 | 1.05 | 0.08 / 1.57 | 471 / 363 | 1.30 | 53 / 53 | 82.3 / 80.4 | 0.10 / 0.07 | 0.12 / 1.61 | 3.1e-06 | 0 | slower heavier |
| segnext-mscan-t-ade20k | fp16 | 32.5 / 26.0 | 1.25 | 0.06 / 1.76 | 259 / 176 | 1.47 | 17 / 9 | 66.4 / 59.2 | 0.09 / 0.07 | 0.10 / 1.80 | 0.0018 | 0.92 (0.00052 % off) | slower heavier |
| segnext-mscan-t-ade20k | fp32 | 36.2 / 32.5 | 1.11 | 0.07 / 1.69 | 355 / 225 | 1.58 | 17 / 17 | 69.9 / 65.5 | 0.09 / 0.07 | 0.10 / 1.73 | 3.2e-06 | 0 | slower heavier |
| selective-igev-eth3d-finetune | fp16 | 3659.3 / 4639.0 | 0.79 | 3.66 / 10.92 | 1873 / 1102 | 1.70 | 25 / 27 | 3653.7 / 4643.5 | 0.14 / 0.10 | 3.67 / 10.92 | 0.5 (0.83 % off) | 0.5 (0.15 % off) | heavier |
| selective-igev-eth3d-finetune | fp32 | 4782.4 / 6335.6 | 0.75 | 4.74 / 12.49 | 3498 / 2057 | 1.70 | 50 / 51 | 4794.6 / 6320.7 | 0.14 / 0.10 | 4.75 / 12.49 | 0.00073 | 0.00074 | heavier |
| selective-igev-eth3d-train | fp16 | 3650.8 / 4622.7 | 0.79 | 3.67 / 10.89 | 1873 / 1102 | 1.70 | 25 / 27 | 3656.8 / 4620.7 | 0.14 / 0.10 | 3.68 / 10.89 | 0.99 (0.49 % off) | 0.99 (0.14 % off) | heavier |
| selective-igev-eth3d-train | fp32 | 4793.4 / 6300.8 | 0.76 | 4.72 / 12.53 | 3498 / 2057 | 1.70 | 50 / 51 | 4805.4 / 6316.0 | 0.12 / 0.10 | 4.73 / 12.54 | 0.00073 | 0.00073 | heavier |
| selective-igev-kitti12 | fp16 | 3642.7 / 4624.8 | 0.79 | 3.68 / 10.91 | 1873 / 1102 | 1.70 | 25 / 27 | 3660.5 / 4619.3 | 0.15 / 0.10 | 3.69 / 10.92 | 0.018 (0.00084 % off) | 0.0037 | heavier |
| selective-igev-kitti12 | fp32 | 4787.0 / 6291.4 | 0.76 | 4.77 / 12.48 | 3498 / 2057 | 1.70 | 50 / 51 | 4790.6 / 6327.5 | 0.12 / 0.10 | 4.78 / 12.49 | 6.3e-06 | 1.7e-06 | heavier |
| selective-igev-kitti15 | fp16 | 3648.9 / 4623.3 | 0.79 | 3.67 / 10.93 | 1873 / 1102 | 1.70 | 25 / 27 | 3661.2 / 4621.2 | 0.14 / 0.10 | 3.68 / 10.94 | 0.015 (0.00071 % off) | 0.00088 | heavier |
| selective-igev-kitti15 | fp32 | 4786.6 / 6315.6 | 0.76 | 4.82 / 12.48 | 3498 / 2057 | 1.70 | 50 / 51 | 4785.2 / 6364.2 | 0.12 / 0.10 | 4.83 / 12.49 | 9.8e-06 | 1.4e-06 | heavier |
| selective-igev-middlebury-finetune | fp16 | 3646.4 / 4626.4 | 0.79 | 3.67 / 10.90 | 1873 / 1102 | 1.70 | 25 / 27 | 3663.1 / 4631.1 | 0.15 / 0.10 | 3.68 / 10.91 | 0.84 (0.49 % off) | 0.84 (0.24 % off) | heavier |
| selective-igev-middlebury-finetune | fp32 | 4789.9 / 6299.3 | 0.76 | 4.74 / 12.50 | 3498 / 2057 | 1.70 | 50 / 51 | 4804.2 / 6309.2 | 0.12 / 0.10 | 4.75 / 12.51 | 0.001 | 0.001 | heavier |
| selective-igev-middlebury-train | fp16 | 3642.6 / 4641.0 | 0.78 | 3.68 / 10.89 | 1873 / 1102 | 1.70 | 25 / 27 | 3668.9 / 4647.3 | 0.14 / 0.10 | 3.69 / 10.90 | 0.61 (0.14 % off) | 0.61 (0.1 % off) | heavier |
| selective-igev-middlebury-train | fp32 | 4801.0 / 6367.1 | 0.75 | 4.72 / 12.66 | 3498 / 2057 | 1.70 | 50 / 51 | 4817.2 / 6324.1 | 0.12 / 0.10 | 4.74 / 12.66 | 0.0012 | 0.0012 | heavier |
| selective-igev-sceneflow | fp16 | 3662.8 / 4645.5 | 0.79 | 3.68 / 11.08 | 1873 / 1102 | 1.70 | 25 / 27 | 3651.9 / 4637.3 | 0.14 / 0.10 | 3.69 / 11.09 | 0.14 (0.099 % off) | 0.098 (0.064 % off) | heavier |
| selective-igev-sceneflow | fp32 | 4796.5 / 6299.9 | 0.76 | 4.81 / 12.51 | 3498 / 2057 | 1.70 | 50 / 51 | 4799.1 / 6291.6 | 0.13 / 0.10 | 4.82 / 12.52 | 0.0026 | 0.0018 | heavier |
| shufflenet_v2_x1_0.tv_in1k | fp16 | 3.2 / 2.3 | 1.39 | 0.02 / 0.77 | 15 / 17 | 0.90 | 4 / 4 | 3.1 / 2.6 | 0.57 / 0.51 | 0.02 / 0.77 | 0.0065 | 0.0065 | slower |
| shufflenet_v2_x1_0.tv_in1k | fp32 | 2.9 / 3.1 | 0.94 | 0.02 / 0.79 | 27 / 30 | 0.89 | 9 / 9 | 3.4 / 3.0 | 0.54 / 0.53 | 0.02 / 0.79 | 1.2e-06 | 1.2e-06 |  |
| shvit_s1.in1k | fp16 | 3.9 / 3.0 | 1.30 | 0.02 / 1.25 | 20 / 20 | 0.98 | 12 / 12 | 4.4 / 3.5 | 0.81 / 0.76 | 0.02 / 1.25 | 0.0016 | 0.0016 | slower |
| shvit_s1.in1k | fp32 | 3.3 / 3.3 | 1.01 | 0.03 / 1.19 | 34 / 35 | 0.96 | 24 / 24 | 3.9 / 3.7 | 0.73 / 0.70 | 0.04 / 1.19 | 1.3e-06 | 1.3e-06 |  |
| shvit_s2.in1k | fp16 | 3.3 / 3.4 | 0.98 | 0.02 / 1.20 | 30 / 31 | 0.99 | 22 / 22 | 3.8 / 3.8 | 0.80 / 0.70 | 0.02 / 1.21 | 0.0023 | 0.0023 |  |
| shvit_s2.in1k | fp32 | 3.7 / 3.5 | 1.06 | 0.04 / 1.20 | 55 / 56 | 0.98 | 44 / 44 | 4.3 / 4.2 | 0.74 / 0.71 | 0.04 / 1.20 | 7.7e-07 | 7.7e-07 | slower |
| shvit_s3.in1k | fp16 | 4.0 / 3.9 | 1.02 | 0.03 / 1.28 | 39 / 41 | 0.95 | 28 / 28 | 4.5 / 4.4 | 0.80 / 0.70 | 0.03 / 1.28 | 0.0014 | 0.0014 |  |
| shvit_s3.in1k | fp32 | 4.5 / 4.3 | 1.05 | 0.04 / 1.25 | 70 / 74 | 0.95 | 54 / 54 | 4.9 / 4.8 | 0.78 / 0.70 | 0.04 / 1.25 | 7.2e-07 | 7.2e-07 | slower |
| shvit_s4.in1k | fp16 | 4.9 / 4.8 | 1.03 | 0.02 / 1.44 | 49 / 52 | 0.95 | 32 / 32 | 5.5 / 5.4 | 0.82 / 0.70 | 0.03 / 1.45 | 0.002 | 0.002 | slower |
| shvit_s4.in1k | fp32 | 5.6 / 5.2 | 1.07 | 0.03 / 1.43 | 88 / 94 | 0.94 | 64 / 64 | 6.1 / 5.7 | 0.75 / 0.71 | 0.04 / 1.44 | 5e-07 | 5e-07 | slower |
| siglip2-b16-256 | fp16 | 14.5 / 14.4 | 1.01 | 0.04 / 0.70 | 197 / 244 | 0.81 | 176 / 177 | 14.8 / 14.6 | 0.84 / 0.70 | 0.05 / 0.70 | 0.0065 | 0.0065 |  |
| siglip2-b16-256 | fp32 | 16.8 / 16.7 | 1.00 | 0.06 / 0.70 | 387 / 444 | 0.87 | 352 / 354 | 17.0 / 17.0 | 0.78 / 0.71 | 0.06 / 0.70 | 4.8e-06 | 4.8e-06 |  |
| siglip2-b16-512 | fp16 | 14.5 / 14.3 | 1.02 | 0.04 / 0.72 | 197 / 244 | 0.81 | 176 / 177 | 14.8 / 14.5 | 0.85 / 0.71 | 0.04 / 0.72 | 0.0069 | 0.0069 |  |
| siglip2-b16-512 | fp32 | 16.8 / 16.7 | 1.01 | 0.07 / 0.70 | 387 / 444 | 0.87 | 352 / 354 | 17.0 / 17.1 | 0.77 / 0.70 | 0.07 / 0.70 | 3.6e-06 | 3.6e-06 |  |
| siglip2-g16-256 | fp16 | 154.2 / 148.5 | 1.04 | 0.18 / 2.02 | 2282 / 2266 | 1.01 | 2214 / 2218 | 154.5 / 149.0 | 4.44 / 0.74 | 0.18 / 2.02 | 0.076 (0.33 % off) | 0.076 (0.33 % off) | slower |
| siglip2-g16-256 | fp32 | 184.2 / 180.5 | 1.02 | 0.55 / 2.70 | 4497 / 4515 | 1.00 | 4429 / 4438 | 184.4 / 180.7 | 5.34 / 1.01 | 0.55 / 2.71 | 1.3e-05 | 1.3e-05 | slower |
| siglip2-g16-384 | fp16 | 154.2 / 148.7 | 1.04 | 0.18 / 2.00 | 2282 / 2266 | 1.01 | 2214 / 2218 | 154.4 / 148.9 | 6.37 / 0.74 | 0.18 / 2.00 | 0.029 (0.13 % off) | 0.029 (0.13 % off) | slower |
| siglip2-g16-384 | fp32 | 184.3 / 182.1 | 1.01 | 0.48 / 2.84 | 4497 / 4515 | 1.00 | 4429 / 4438 | 184.5 / 181.3 | 5.29 / 1.02 | 0.48 / 2.84 | 3.4e-05 | 3.4e-05 |  |
| siglip2-l16-256 | fp16 | 44.7 / 44.1 | 1.01 | 0.06 / 1.16 | 623 / 663 | 0.94 | 601 / 602 | 45.1 / 44.2 | 1.02 / 0.72 | 0.07 / 1.16 | 0.089 (0.57 % off) | 0.089 (0.57 % off) |  |
| siglip2-l16-256 | fp32 | 52.4 / 51.3 | 1.02 | 0.09 / 1.16 | 1238 / 1301 | 0.95 | 1201 / 1205 | 52.7 / 51.6 | 0.85 / 0.75 | 0.09 / 1.17 | 4.5e-05 | 4.5e-05 | slower |
| siglip2-l16-512 | fp16 | 46.3 / 44.2 | 1.05 | 0.06 / 1.17 | 646 / 663 | 0.97 | 601 / 602 | 46.6 / 44.5 | 1.00 / 0.72 | 0.07 / 1.17 | 0.011 (0.0005 % off) | 0.011 (0.0005 % off) | slower |
| siglip2-l16-512 | fp32 | 53.5 / 51.6 | 1.04 | 0.09 / 1.17 | 1247 / 1301 | 0.96 | 1201 / 1205 | 53.8 / 51.9 | 0.86 / 0.71 | 0.09 / 1.17 | 1.8e-05 | 1.8e-05 | slower |
| siglip2-so400m14-224 | fp16 | 63.9 / 63.2 | 1.01 | 0.08 / 1.36 | 846 / 883 | 0.96 | 814 / 816 | 64.2 / 63.4 | 1.63 / 0.70 | 0.08 / 1.37 | 0.046 (0.014 % off) | 0.046 (0.014 % off) |  |
| siglip2-so400m14-224 | fp32 | 73.5 / 72.0 | 1.02 | 0.11 / 1.38 | 1680 / 1740 | 0.97 | 1627 / 1632 | 73.8 / 72.1 | 0.89 / 0.71 | 0.12 / 1.38 | 2.8e-05 | 2.8e-05 | slower |
| siglip2-so400m14-378 | fp16 | 64.0 / 62.9 | 1.02 | 0.09 / 1.38 | 846 / 883 | 0.96 | 814 / 816 | 64.2 / 63.4 | 1.78 / 0.71 | 0.09 / 1.38 | 0.02 (0.001 % off) | 0.02 (0.001 % off) |  |
| siglip2-so400m14-378 | fp32 | 73.7 / 73.8 | 1.00 | 0.13 / 1.52 | 1680 / 1740 | 0.97 | 1627 / 1632 | 74.5 / 74.2 | 0.97 / 0.74 | 0.13 / 1.52 | 1.7e-05 | 1.7e-05 |  |
| siglip2-so400m16-256 | fp16 | 63.4 / 61.2 | 1.04 | 0.08 / 1.39 | 863 / 884 | 0.98 | 814 / 817 | 63.7 / 61.5 | 1.72 / 0.72 | 0.08 / 1.39 | 0.037 (0.0071 % off) | 0.037 (0.0071 % off) | slower |
| siglip2-so400m16-256 | fp32 | 72.8 / 70.3 | 1.04 | 0.11 / 1.44 | 1678 / 1740 | 0.96 | 1628 / 1633 | 73.1 / 70.6 | 0.88 / 0.81 | 0.12 / 1.44 | 4.6e-05 | 4.6e-05 | slower |
| siglip2-so400m16-512 | fp16 | 63.3 / 61.2 | 1.03 | 0.08 / 1.40 | 863 / 884 | 0.98 | 814 / 817 | 63.6 / 61.5 | 1.97 / 0.73 | 0.09 / 1.40 | 0.0061 | 0.0061 | slower |
| siglip2-so400m16-512 | fp32 | 72.0 / 69.6 | 1.03 | 0.11 / 1.45 | 1678 / 1740 | 0.96 | 1628 / 1633 | 72.3 / 69.9 | 0.88 / 0.77 | 0.11 / 1.45 | 9.6e-06 | 9.6e-06 | slower |
| squeezenet1_1.tv_in1k | fp16 | 1.7 / 1.9 | 0.88 | 0.01 / 0.44 | 16 / 18 | 0.86 | 2 / 2 | 2.1 / 2.2 | 0.53 / 0.52 | 0.02 / 0.44 | 0 | 0 |  |
| squeezenet1_1.tv_in1k | fp32 | 2.0 / 1.7 | 1.15 | 0.03 / 0.39 | 28 / 32 | 0.86 | 5 / 5 | 2.1 / 2.1 | 0.51 / 0.50 | 0.03 / 0.39 | 9.2e-08 | 9.2e-08 | slower |
| starnet_s1.in1k | fp16 | 2.2 / 2.6 | 0.83 | 0.01 / 0.66 | 19 / 24 | 0.80 | 6 / 6 | 2.6 / 3.1 | 0.75 / 0.69 | 0.02 / 0.67 | 0.0015 | 0.0015 |  |
| starnet_s1.in1k | fp32 | 2.5 / 3.0 | 0.83 | 0.03 / 0.64 | 33 / 43 | 0.77 | 11 / 11 | 3.0 / 3.6 | 0.70 / 0.69 | 0.03 / 0.64 | 1.6e-06 | 1.6e-06 |  |
| starnet_s2.in1k | fp16 | 2.0 / 2.4 | 0.85 | 0.01 / 0.56 | 19 / 24 | 0.82 | 7 / 7 | 2.5 / 2.8 | 0.74 / 0.70 | 0.02 / 0.57 | 0.00093 | 0.00093 |  |
| starnet_s2.in1k | fp32 | 2.2 / 3.0 | 0.75 | 0.03 / 0.54 | 35 / 44 | 0.81 | 14 / 14 | 2.8 / 3.1 | 0.70 / 0.69 | 0.03 / 0.54 | 6.2e-07 | 6.2e-07 |  |
| starnet_s3.in1k | fp16 | 2.6 / 3.0 | 0.87 | 0.01 / 0.69 | 28 / 34 | 0.83 | 11 / 11 | 3.1 / 3.5 | 0.78 / 0.70 | 0.02 / 0.69 | 0.00078 | 0.00078 |  |
| starnet_s3.in1k | fp32 | 2.9 / 3.3 | 0.87 | 0.03 / 0.67 | 52 / 63 | 0.82 | 22 / 22 | 3.4 / 3.9 | 0.71 / 0.70 | 0.03 / 0.67 | 3.8e-07 | 3.8e-07 |  |
| starnet_s4.in1k | fp16 | 3.3 / 3.9 | 0.85 | 0.01 / 0.86 | 38 / 46 | 0.83 | 14 / 14 | 3.8 / 4.4 | 0.78 / 0.69 | 0.02 / 0.86 | 0.00086 | 0.00086 |  |
| starnet_s4.in1k | fp32 | 3.8 / 4.5 | 0.86 | 0.03 / 0.82 | 71 / 87 | 0.82 | 29 / 29 | 4.3 / 4.9 | 0.71 / 0.70 | 0.03 / 0.82 | 4.2e-07 | 4.2e-07 |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp16 | 18.3 / 17.8 | 1.03 | 0.04 / 2.11 | 302 / 263 | 1.15 | 173 / 170 | 19.1 / 18.4 | 0.94 / 0.79 | 0.04 / 2.11 | 0.0029 | 0.0029 | slower heavier |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 20.0 / 21.2 | 0.95 | 0.06 / 2.15 | 583 / 422 | 1.38 | 346 / 340 | 20.9 / 22.1 | 0.91 / 0.79 | 0.06 / 2.15 | 1.6e-06 | 1.6e-06 | heavier |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp16 | 4.8 / 4.5 | 1.06 | 0.03 / 0.93 | 64 / 67 | 0.96 | 25 / 25 | 5.3 / 5.3 | 0.80 / 0.74 | 0.03 / 0.93 | 0.001 | 0.001 | slower |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp32 | 5.4 / 5.2 | 1.04 | 0.04 / 0.90 | 118 / 124 | 0.95 | 49 / 49 | 5.8 / 5.6 | 0.76 / 0.72 | 0.04 / 0.90 | 1.2e-06 | 1.2e-06 | slower |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp16 | 6.8 / 6.6 | 1.02 | 0.03 / 0.97 | 103 / 107 | 0.96 | 46 / 46 | 7.3 / 7.1 | 0.85 / 0.73 | 0.03 / 0.97 | 0.0012 | 0.0012 |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp32 | 7.6 / 7.9 | 0.96 | 0.04 / 0.94 | 192 / 198 | 0.97 | 92 / 92 | 8.2 / 8.4 | 0.77 / 0.73 | 0.04 / 0.94 | 9.6e-07 | 9.6e-07 |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp16 | 15.3 / 14.1 | 1.08 | 0.03 / 0.97 | 225 / 186 | 1.21 | 88 / 88 | 15.5 / 14.4 | 1.12 / 1.02 | 0.03 / 0.97 | 0.00069 | 0.00069 | slower heavier |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 17.9 / 17.3 | 1.03 | 0.05 / 0.96 | 441 / 290 | 1.52 | 176 / 176 | 18.2 / 17.7 | 1.08 / 1.01 | 0.05 / 0.96 | 6.8e-07 | 6.8e-07 | slower heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 27.6 / 25.2 | 1.10 | 0.04 / 1.00 | 349 / 297 | 1.17 | 191 / 191 | 28.0 / 25.4 | 1.83 / 1.69 | 0.04 / 1.00 | 0.0009 | 0.0009 | slower heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 36.0 / 31.9 | 1.13 | 0.07 / 0.98 | 697 / 510 | 1.37 | 381 / 381 | 36.3 / 32.4 | 1.78 / 1.68 | 0.07 / 0.99 | 1e-06 | 1e-06 | slower heavier |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp16 | 4.3 / 3.9 | 1.10 | 0.02 / 0.92 | 42 / 45 | 0.94 | 13 / 13 | 4.8 / 4.5 | 0.81 / 0.72 | 0.03 / 0.92 | 0.001 | 0.001 | slower |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp32 | 4.7 / 4.4 | 1.09 | 0.04 / 0.90 | 79 / 85 | 0.93 | 25 / 25 | 5.2 / 5.0 | 0.75 / 0.74 | 0.04 / 0.91 | 8.3e-07 | 8.3e-07 | slower |
| totalsegmentator-total | fp16 | 14612.3 / 26280.3 | 0.56 | 14.55 / 33.05 | 3464 / 3912 | 0.89 | 567 / 298 | 17992.2 / 30890.6 | 1.49 / 0.28 | 18.41 / 37.53 | - | 1 (0.0048 % off) (1 keys differ) |  |
| totalsegmentator-total | fp32 | 21167.3 / 38793.7 | 0.55 | 21.18 / 42.58 | 3456 / 3954 | 0.87 | 595 / 595 | 25054.6 / 42912.5 | 0.07 / 0.26 | 25.89 / 47.26 | - | 0 (1 keys differ) |  |
| totalsegmentator-total-3mm | fp16 | 304.4 / 425.0 | 0.72 | 0.33 / 1.53 | 2058 / 1609 | 1.28 | 45 / 32 | 478.9 / 672.2 | 0.14 / 0.76 | 0.69 / 1.85 | 0.045 (0.0023 % off) | 1 (0.29 % off) | heavier |
| totalsegmentator-total-3mm | fp32 | 414.7 / 535.0 | 0.78 | 0.44 / 1.58 | 2033 / 1769 | 1.15 | 63 / 63 | 626.9 / 761.6 | 0.04 / 0.74 | 0.80 / 1.92 | 0.00012 | 0.74 (0.002 % off) | heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 0.92 / 3.63 | 1523 / 1166 | 1.31 | 715 / 696 | 894.8 / 904.5 | 1.46 / 0.07 | 0.93 / 3.64 | 0.009 | 0.016 (1.1 % off) | heavier |
| unik3d-vitl | fp32 | 1214.4 / 1191.7 | 1.02 | 1.36 / 3.91 | 3060 / 3117 | 0.98 | 1376 / 1386 | 1221.6 / 1197.9 | 1.36 / 0.07 | 1.37 / 3.92 | 5.1e-06 | 1.2e-05 |  |
| vgg16.tv_in1k | fp16 | 13.2 / 13.6 | 0.97 | 0.03 / 0.47 | 322 / 353 | 0.91 | 264 / 264 | 13.6 / 14.1 | 0.84 / 0.55 | 0.03 / 0.47 | 0 | 0 |  |
| vgg16.tv_in1k | fp32 | 16.7 / 17.5 | 0.96 | 0.04 / 0.50 | 643 / 705 | 0.91 | 528 / 528 | 17.3 / 18.0 | 0.79 / 0.57 | 0.04 / 0.50 | 0 | 0 |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp16 | 14.1 / 13.8 | 1.02 | 0.04 / 0.65 | 242 / 232 | 1.04 | 165 / 165 | 14.7 / 14.3 | 0.05 / 0.72 | 0.04 / 0.66 | 0.0011 | 0.0011 | slower heavier |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp32 | 16.3 / 16.0 | 1.02 | 0.04 / 0.65 | 454 / 420 | 1.08 | 330 / 330 | 16.8 / 16.7 | 0.06 / 0.69 | 0.05 / 0.66 | 5.8e-07 | 5.8e-07 | heavier |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp16 | 180.7 / 178.6 | 1.01 | 0.21 / 1.47 | 912 / 895 | 1.02 | 817 / 820 | 181.6 / 179.5 | 0.08 / 0.73 | 0.21 / 1.47 | 0.00081 | 0.00081 |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp32 | 214.2 / 209.3 | 1.02 | 0.24 / 1.50 | 1823 / 1796 | 1.02 | 1634 / 1639 | 214.0 / 209.9 | 0.13 / 0.73 | 0.25 / 1.50 | 5.2e-07 | 5.2e-07 | slower |
| wd-convnext-tagger-v3 | fp16 | 45.8 / 43.9 | 1.04 | 0.06 / 1.07 | 359 / 328 | 1.09 | 189 / 189 | 46.2 / 44.2 | 0.94 / 0.69 | 0.06 / 1.08 | 0.00096 | 0.00096 | slower heavier |
| wd-convnext-tagger-v3 | fp32 | 52.4 / 52.3 | 1.00 | 0.08 / 1.04 | 657 / 512 | 1.28 | 377 / 377 | 52.7 / 52.7 | 0.88 / 0.71 | 0.08 / 1.04 | 1e-06 | 1e-06 | heavier |
| wd-eva02-large-tagger-v3 | fp16 | 195.5 / 197.7 | 0.99 | 0.23 / 2.07 | 736 / 686 | 1.07 | 602 / 602 | 195.9 / 198.3 | 0.08 / 0.75 | 0.23 / 2.07 | 0.001 | 0.001 | heavier |
| wd-eva02-large-tagger-v3 | fp32 | 241.3 / 241.6 | 1.00 | 0.27 / 2.11 | 1468 / 1435 | 1.02 | 1203 / 1203 | 241.7 / 242.4 | 0.11 / 0.75 | 0.28 / 2.11 | 6.1e-07 | 6.1e-07 | heavier |
| wd-vit-large-tagger-v3 | fp16 | 178.9 / 175.4 | 1.02 | 0.21 / 1.21 | 696 / 683 | 1.02 | 601 / 601 | 179.2 / 175.9 | 0.07 / 0.71 | 0.21 / 1.21 | 0.0009 | 0.0009 |  |
| wd-vit-large-tagger-v3 | fp32 | 202.5 / 198.1 | 1.02 | 0.23 / 1.21 | 1387 / 1401 | 0.99 | 1202 / 1202 | 202.9 / 198.6 | 0.10 / 0.70 | 0.24 / 1.21 | 3.9e-07 | 3.9e-07 | slower |
| wd-vit-tagger-v3 | fp16 | 44.8 / 43.9 | 1.02 | 0.07 / 0.68 | 283 / 251 | 1.13 | 180 / 180 | 45.1 / 44.3 | 0.05 / 0.69 | 0.08 / 0.68 | 0.00088 | 0.00088 | heavier |
| wd-vit-tagger-v3 | fp32 | 53.7 / 53.8 | 1.00 | 0.08 / 0.67 | 572 / 444 | 1.29 | 361 / 361 | 54.3 / 54.2 | 0.06 / 0.69 | 0.09 / 0.68 | 2.7e-07 | 2.7e-07 | heavier |
| xfeat | fp32 | 16.9 / 31.8 | 0.53 | 0.05 / 1.04 | 199 / 165 | 1.21 | 3 / 3 | 26.8 / 158.4 | 0.04 / 0.03 | 0.08 / 1.18 | - | 0.94 (0.33 % off) | heavier |
| xfeat-lighterglue | fp32 | 116.8 / 28.7 | 4.07 | 0.16 / 1.05 | 457 / 165 | 2.77 | 7 / 3 | 120.7 / 511.6 | 0.06 / 0.05 | 0.17 / 1.56 | - | 0.94 (0.36 % off) | slower heavier |
| yolox-l | fp16 | 51.0 / 61.3 | 0.83 | 0.07 / 1.48 | 293 / 280 | 1.05 | 103 / 103 | 56.0 / 66.1 | 0.05 / 0.14 | 0.54 / 1.55 | 0.7 (66 % off) | 0 | heavier |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 0.08 / 1.45 | 581 / 505 | 1.15 | 207 / 207 | 65.9 / 80.6 | 0.05 / 0.14 | 0.59 / 1.52 | 0.7 (66 % off) | 0 | heavier |
| yolox-m | fp16 | 30.0 / 37.5 | 0.80 | 0.06 / 1.35 | 170 / 207 | 0.82 | 49 / 49 | 34.8 / 41.8 | 0.04 / 0.11 | 0.07 / 1.35 | 0.74 (67 % off) | empty |  |
| yolox-m | fp32 | 33.1 / 42.5 | 0.78 | 0.06 / 1.38 | 368 / 441 | 0.83 | 97 / 97 | 38.3 / 46.9 | 0.04 / 0.15 | 0.07 / 1.39 | 0.74 (67 % off) | empty |  |
| yolox-s | fp16 | 12.4 / 17.0 | 0.73 | 0.05 / 1.25 | 83 / 178 | 0.47 | 17 / 17 | 18.0 / 21.5 | 0.04 / 0.08 | 0.54 / 1.32 | 0.77 (66 % off) | 0 |  |
| yolox-s | fp32 | 14.9 / 20.7 | 0.72 | 0.05 / 1.20 | 159 / 214 | 0.74 | 34 / 34 | 20.1 / 25.8 | 0.04 / 0.08 | 0.55 / 1.27 | 0.77 (66 % off) | 0 |  |
| yolox-tiny | fp16 | 8.8 / 12.4 | 0.71 | 0.04 / 1.23 | 37 / 65 | 0.57 | 10 / 10 | 13.4 / 17.3 | 0.04 / 0.07 | 0.53 / 1.30 | 0 | 0 |  |
| yolox-tiny | fp32 | 8.9 / 13.7 | 0.65 | 0.05 / 1.23 | 72 / 134 | 0.54 | 20 / 20 | 13.2 / 16.0 | 0.04 / 0.08 | 0.55 / 1.30 | 0 | 0 |  |
| yolox-x | fp16 | 94.4 / 106.6 | 0.89 | 0.12 / 1.73 | 391 / 441 | 0.89 | 190 / 190 | 100.1 / 111.8 | 0.06 / 0.21 | 0.64 / 1.80 | 0.58 (65 % off) | 0 |  |
| yolox-x | fp32 | 108.8 / 128.4 | 0.85 | 0.14 / 1.78 | 829 / 802 | 1.03 | 378 / 378 | 114.7 / 133.5 | 0.07 / 0.21 | 0.15 / 1.79 | 0.58 (65 % off) | empty | heavier |

Not measured (one arm failed):

- resplat-base-re10k float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-base-re10k float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
