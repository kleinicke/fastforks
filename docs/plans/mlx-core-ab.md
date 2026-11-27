# MLX core A/B: hand-written networks against the graph translation (2026-11-26)

Every hand-ported ref (the families with `GRAPH_RUNNERS`, and the libnet classifiers written out in `mlx_cnn` /
`mlx_vit_cls`), portable mode, float16 and float32, on the network alone: `tools/mlx_core_ab.py` (how-to in
[mlx-coverage-gaps.md](mlx-coverage-gaps.md#how-to-ab-a-hand-written-mlx-network-against-its-graph-translation-2026-11-26)).
M1 Pro 16 GB, MLX 0.32.4, buffer cache 2048 MB in both arms, a throwaway warm-up process before each
arm (warm conversion cache), 4 warm-up calls then the median of 5. Rows: [mlx-core-ab.jsonl](mlx-core-ab.jsonl).
First measured as a diagnostic sweep; the families touched by parity round 1 (below) re-measured after it.

**Bar**: the hand network at most 2 % slower (warm) and at most 2 % heavier (network peak) than the exported-graph
translation it replaced. 677 rows over 95 families; 673 measured in both arms (the restoration ports of 2026-11-27
included, [below](#restoration-ports-2026-11-27), and the detection / pose ports of the same day,
[below](#detection--pose-ports-2026-11-27)).

- 313 rows flagged (128 slower, 216 heavier) in 70 families (the sweep: 325).
  Median hand / graph: warm 0.91, peak 0.93.
- 85 of the flagged rows are accepted as a peak trade (clearly faster, see
  [Accepted: faster, peak trade](#accepted-faster-peak-trade)); 228 rows still miss the bar.
- Beyond 10 % (same number of network calls in both arms): 131 rows, listed first below.
- 46 flagged rows time a different number of network calls per model call in the two arms (the hand port
  splits or merges the network differently): their warm sums compare, their peaks (the largest single call) less so.

**Parity round 1 (2026-11-26).** Systematic peak causes in the shared helpers, each fixed once and re-measured over the
families that use it (those families' rows re-run after the fixes; the other families keep their sweep rows):

- *Unfused attention head sizes.* `mx.fast.scaled_dot_product_attention` fuses only head sizes 64, 72, 80, 96, 128;
  any other materialises the float32 scores (SegFormer b0 stage 1: 16384 x 256 per head). `_core.mlx.attention`,
  shared by the graph translation's `_sdpa` and the hand networks, zero-pads a smaller head to the next fused size
  where the scores outweigh the padded copies (`lq * lk > 2 * fused * (lq + lk)`: 12 x 12 Swin windows stay unpadded;
  padding them was heavier and in float32 slower) and runs a larger one in query blocks. Exact (the padded columns
  are zero); a no-op for head size 64 (DINOv2 / v3, SAM, C-RADIO in `mlx_vit`). `mlx_vit`'s float16 blocks keep
  MLX's own kernel: there the padded one is closer to float32 per call (LW-DETR-M's global blocks: max error 0.002
  against 0.004-0.02), yet LW-DETR-M float16 then lost its golden's 0.91 box. SegFormer b0 float16 346 -> 158 MB
  (32.9 -> 19.2 ms), OneFormer float16 1746 -> 1364 MB (float32 2270 -> 1894).
- *Float32 resize taps.* The segmentation heads' bilinear resize built float32 tap weights, promoting float16 maps to
  float32. `mlx_seg.resize` now runs the graph translation's one-pass NHWC kernel (`ops.bilinear_nhwc`) in the map's
  dtype: HRNet-W48 float16 534 -> 313 MB (180 -> 153 ms), DeepLabV3+ float16 860 -> 773 MB, part of the SegFormer /
  SegNeXt gains. SegNeXt also evaluates its weight casts once at load (a lazy `astype` kept the float32 source
  resident and recomputed it in every call).
- *One command buffer per network.* MLX returns a command buffer's transients only when it completes, so one compiled
  function over a whole classifier holds every block's intermediates to the end, where the graph translation
  evaluates every 256 MB. `mlx_cnn._segmented` compiles the top-level ops in 16 segments of about equal output bytes
  (at least 3 ops each) and evaluates them one after the other: the libnet classifiers' peaks fall 10-40 %
  (ConvNeXt-B float32 475 -> 378 MB, CoAtNet-2 float32 552 -> 424 MB, FocalNet-B float32 544 -> 419 MB,
  EfficientFormerV2-L float16 178 -> 69 MB) at equal warm time (1 against 16 segments on ShViT-S4, MobileNetV3-L,
  MobileNetV4-M, HGNetV2-B0: within noise).
- LoFTR: the dense pass staged and its dual softmax run in blocks.

**Measurement drift.** The machine ran 15-35 % slower in both arms than during the sweep on the large networks
(Mask2Former float16: today's tree and the round's base `07223ce4` both 585-595 ms in the same session, against 425 ms
in the sweep; FCN-R101, which this round did not touch: 277 -> 326 ms). The hand / graph ratios come from
back-to-back arms and hold; before / after absolute times of the large networks do not. FCN's float16 peak moved
638 -> 877 MB without a code change, so peak differences under about 15 % on the 300 ms+ networks are within noise.

**End to end** (the MLX survey of the 127 touched refs, run twice, the better of the two against the rows before the
round; 254 rows, every golden passing): median warm 0.99, peak 0.96, first call 1.05 (first calls of 0.02-0.1 s:
noise). No peak grew beyond 5 % (FocalNet-B 1222 -> 632 MB, HGNetV2-B6 1289 -> 791, LoFTR 2129 -> 1661,
OneFormer 2587 -> 2144). Warm times beyond 5 % slower (up to 1.1x, MobileNetV1 float32 2.5 -> 3.2 ms) are the drift
above: their network A/B rows are equal or faster than before, and untouched networks slowed alike.

**Still flagged, with the reason** (beyond the accepted peak trades):

- SegFormer, SegNeXt: still 1.1-1.8x the graph's peak and 5-10 % slower after the attention and resize fixes; the
  rest was not located this round.
- RF-DETR (float16 5-9 % slower, float32 1-5 %, as in the sweep) and rf-detr-keypoint (peak in a post-backbone stage): unchanged by the shared fixes.
- Selective-IGEV float32: 25 % faster, but the peak (the 3 x 3 conv workspace of the GRU stage, about 850 MB
  transient) puts the e2e peak at or just above the best MPS row, so not accepted. A graph fallback was rejected: the
  graph is slower warm and its first call takes 8.7 s.
- Sapiens float32, TotalSegmentator: faster, but the e2e peak is above the best MPS row.
- ConvNeXt tiny / small, DaViT, ShViT, DeiT / ViT, RF-DETR: 2-19 % slower, as in the sweep; the segments did not
  change it (ShViT measured unsegmented: the same).
- TinyViT 384 / 512, CoAtNet-RMLP-2 384 float32, ConvNeXt-L 384 float32, wd-taggers, Swin-B float32: heavier, not
  clearly faster; the ratio grows with the crop (TinyViT-21M float32 0.73 at 224, 1.12 at 384, 1.30 at 512).
- LoFTR (the two arms run different inputs), XFeat (the hand network includes LightGlue), the R-CNN heads (the box
  head counted in one arm only): A/B artifacts, not like-for-like.

Not reached this round: TinyViT's large crops, rcnn, raft_stereo, nafnet, rtmpose's float32 growth, the classic small
CNNs (shufflenetv2, squeezenet), mobilenetv1, lw-detr large / xlarge float16.

## Flagged beyond 10 %

| ref | dtype | net warm ms hand / graph | x | net peak MB hand / graph | x | flag |
|---|---|---|---|---|---|---|
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 2072 / 855 | 2.42 | heavier (accepted) |
| segformer-b5-ade20k | fp32 | 264.6 / 258.5 | 1.02 | 1410 / 780 | 1.81 | slower heavier |
| rf-detr-keypoint-preview | fp32 | 72.0 / 74.2 | 0.97 | 428 / 251 | 1.71 | heavier |
| segformer-b0-ade20k | fp32 | 23.0 / 22.5 | 1.02 | 254 / 149 | 1.70 | slower heavier |
| selective-igev-kitti15 | fp32 | 4786.6 / 6315.6 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-finetune | fp32 | 4782.4 / 6335.6 | 0.75 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-eth3d-train | fp32 | 4793.4 / 6300.8 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-kitti12 | fp32 | 4787.0 / 6291.4 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-finetune | fp32 | 4789.9 / 6299.3 | 0.76 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-middlebury-train | fp32 | 4801.0 / 6367.1 | 0.75 | 3498 / 2057 | 1.70 | heavier |
| selective-igev-sceneflow | fp32 | 4796.5 / 6299.9 | 0.76 | 3498 / 2057 | 1.70 | heavier (accepted) |
| selective-igev-eth3d-finetune | fp16 | 3659.3 / 4639.0 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-eth3d-train | fp16 | 3650.8 / 4622.7 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-kitti12 | fp16 | 3642.7 / 4624.8 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-kitti15 | fp16 | 3648.9 / 4623.3 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-middlebury-finetune | fp16 | 3646.4 / 4626.4 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-middlebury-train | fp16 | 3642.6 / 4641.0 | 0.78 | 1873 / 1102 | 1.70 | heavier (accepted) |
| selective-igev-sceneflow | fp16 | 3662.8 / 4645.5 | 0.79 | 1873 / 1102 | 1.70 | heavier (accepted) |
| lw-detr-large-coco | fp16 | 52.6 / 57.6 | 0.91 | 534 / 319 | 1.67 | heavier |
| igev-stereo-middlebury | fp16 | 3187.7 / 3596.0 | 0.89 | 1686 / 1010 | 1.67 | heavier (accepted) |
| igev-stereo-sceneflow | fp16 | 3187.4 / 3609.1 | 0.88 | 1686 / 1010 | 1.67 | heavier (accepted) |
| segnext-mscan-t-ade20k | fp32 | 35.2 / 32.5 | 1.08 | 362 / 225 | 1.61 | slower heavier |
| rf-detr-keypoint-preview | fp16 | 63.0 / 62.0 | 1.02 | 354 / 225 | 1.57 | heavier |
| raft-stereo-eth3d | fp32 | 4832.4 / 5011.0 | 0.96 | 2943 / 1872 | 1.57 | heavier |
| raft-stereo-middlebury | fp32 | 4808.4 / 5041.7 | 0.95 | 2943 / 1872 | 1.57 | heavier |
| raft-stereo-sceneflow | fp32 | 4812.2 / 5045.0 | 0.95 | 2943 / 1872 | 1.57 | heavier |
| lw-detr-xlarge-coco | fp16 | 96.9 / 108.3 | 0.90 | 698 / 445 | 1.57 | heavier (accepted) |
| raft-stereo-sceneflow | fp16 | 3191.9 / 4043.8 | 0.79 | 1630 / 1042 | 1.56 | heavier (accepted) |
| igev-stereo-eth3d | fp16 | 3185.1 / 3594.5 | 0.89 | 1686 / 1080 | 1.56 | heavier (accepted) |
| igev-stereo-kitti12 | fp16 | 3190.2 / 3603.2 | 0.89 | 1686 / 1080 | 1.56 | heavier (accepted) |
| igev-stereo-kitti15 | fp16 | 3186.4 / 3601.5 | 0.88 | 1686 / 1080 | 1.56 | heavier (accepted) |
| convnext_tiny.fb_in1k | fp16 | 9.2 / 6.0 | 1.53 | 70 / 108 | 0.65 | slower |
| segformer-b2-clothes | fp32 | 31.4 / 31.9 | 0.98 | 446 / 291 | 1.53 | heavier |
| igev-stereo-eth3d | fp32 | 4179.4 / 4676.3 | 0.89 | 3122 / 2055 | 1.52 | heavier (accepted) |
| igev-stereo-kitti12 | fp32 | 4177.9 / 4578.8 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-kitti15 | fp32 | 4162.0 / 4564.7 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-middlebury | fp32 | 4175.4 / 4581.2 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| igev-stereo-sceneflow | fp32 | 4171.9 / 4560.1 | 0.91 | 3122 / 2055 | 1.52 | heavier |
| nafnet-sidd | fp32 | 254.2 / 268.5 | 0.95 | 1346 / 896 | 1.50 | heavier |
| raft-stereo-middlebury | fp16 | 3168.7 / 4054.7 | 0.78 | 1630 / 1128 | 1.45 | heavier (accepted) |
| segformer-b2-clothes | fp16 | 26.6 / 24.6 | 1.08 | 225 / 158 | 1.42 | slower heavier |
| segnext-mscan-t-ade20k | fp16 | 31.1 / 25.7 | 1.21 | 249 / 176 | 1.41 | slower heavier |
| raft-stereo-eth3d | fp16 | 3176.0 / 4061.6 | 0.78 | 1630 / 1166 | 1.40 | heavier (accepted) |
| nafnet-gopro | fp32 | 257.5 / 291.6 | 0.88 | 1170 / 839 | 1.39 | heavier (accepted) |
| shufflenet_v2_x1_0.tv_in1k | fp16 | 3.2 / 2.3 | 1.39 | 15 / 17 | 0.90 | slower |
| nafnet-sidd | fp16 | 219.4 / 207.7 | 1.06 | 804 / 578 | 1.39 | slower heavier |
| segformer-b0-ade20k | fp16 | 19.2 / 17.3 | 1.11 | 158 / 114 | 1.39 | slower heavier |
| segnext-mscan-l-ade20k | fp16 | 137.8 / 126.8 | 1.09 | 423 / 305 | 1.38 | slower heavier |
| lineart | fp32 | 98.6 / 117.6 | 0.84 | 1245 / 903 | 1.38 | heavier (accepted) |
| lineart-coarse | fp32 | 96.5 / 117.3 | 0.82 | 1245 / 903 | 1.38 | heavier (accepted) |
| fcn-r101-d8-ade20k | fp16 | 326.3 / 354.0 | 0.92 | 877 / 638 | 1.38 | heavier |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1647 / 1208 | 1.36 | heavier (accepted) |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1647 / 1209 | 1.36 | heavier (accepted) |
| oneformer-swinl-ade20k | fp16 | 643.5 / 760.8 | 0.85 | 1364 / 1002 | 1.36 | heavier (accepted) |
| pspnet-r101-d8-ade20k | fp16 | 338.6 / 365.6 | 0.93 | 1042 / 768 | 1.36 | heavier |
| rf-detr-seg-2xl-coco | fp16 | 126.7 / 120.4 | 1.05 | 394 / 292 | 1.35 | slower heavier |
| yunet-n | fp32 | 12.0 / 14.7 | 0.81 | 367 / 271 | 1.35 | heavier (accepted) |
| segformer-b5-face-parsing | fp32 | 78.2 / 79.4 | 0.98 | 664 / 495 | 1.34 | heavier |
| rf-detr-seg-2xl-coco | fp32 | 156.9 / 155.4 | 1.01 | 576 / 433 | 1.33 | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 2197 / 1668 | 1.32 | heavier |
| sam2.1-hiera-tiny | fp32 | 130.7 / 127.7 | 1.02 | 572 / 436 | 1.31 | slower heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 1523 / 1166 | 1.31 | heavier |
| segnext-mscan-s-ade20k | fp16 | 42.4 / 37.5 | 1.13 | 256 / 197 | 1.30 | slower heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 31.1 / 31.9 | 0.97 | 666 / 513 | 1.30 | heavier |
| segnext-mscan-s-ade20k | fp32 | 47.0 / 46.8 | 1.00 | 471 / 363 | 1.30 | heavier |
| sam2.1-hiera-small | fp32 | 155.0 / 152.5 | 1.02 | 599 / 463 | 1.29 | heavier |
| wd-vit-tagger-v3 | fp32 | 53.8 / 53.9 | 1.00 | 572 / 444 | 1.29 | heavier |
| retinaface-resnet50 | fp32 | 151.1 / 171.8 | 0.88 | 902 / 697 | 1.29 | heavier (accepted) |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2901 / 2261 | 1.28 | heavier (accepted) |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2901 / 2261 | 1.28 | heavier (accepted) |
| segformer-b5-ade20k | fp16 | 223.7 / 205.4 | 1.09 | 710 / 555 | 1.28 | slower heavier |
| totalsegmentator-total-3mm | fp16 | 304.4 / 425.0 | 0.72 | 2058 / 1609 | 1.28 | heavier |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 1311 / 1026 | 1.28 | heavier |
| segnext-mscan-b-ade20k | fp32 | 85.5 / 87.1 | 0.98 | 528 / 416 | 1.27 | heavier |
| dat-light-x4 | fp32 | 1241.8 / 1651.2 | 0.75 | 2385 / 1899 | 1.26 | heavier (accepted) |
| retinaface-resnet50 | fp16 | 119.6 / 132.8 | 0.90 | 469 / 372 | 1.26 | heavier (accepted) |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 488 / 393 | 1.24 | heavier (accepted) |
| segnext-mscan-b-ade20k | fp16 | 77.9 / 70.2 | 1.11 | 316 / 255 | 1.24 | slower heavier |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 483 / 393 | 1.23 | heavier (accepted) |
| segnext-mscan-l-ade20k | fp32 | 153.5 / 158.2 | 0.97 | 610 / 497 | 1.23 | heavier |
| segformer-b5-face-parsing | fp16 | 67.2 / 62.9 | 1.07 | 335 / 274 | 1.22 | slower heavier |
| davit_tiny.msft_in1k | fp16 | 9.2 / 7.6 | 1.22 | 95 / 123 | 0.77 | slower |
| xfeat | fp32 | 16.9 / 31.8 | 0.53 | 199 / 165 | 1.21 | heavier (accepted) |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 14.3 / 12.0 | 1.20 | 91 / 161 | 0.56 | slower |
| realesrgan-x2plus | fp16 | 499.2 / 506.6 | 0.99 | 1018 / 847 | 1.20 | heavier |
| deeplabv3-r101-d8-ade20k | fp16 | 366.9 / 377.9 | 0.97 | 682 / 574 | 1.19 | heavier |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 1062 / 900 | 1.18 | heavier (accepted) |
| rf-detr-l-coco | fp32 | 52.7 / 51.2 | 1.03 | 268 / 228 | 1.18 | slower heavier |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 4.1 / 3.5 | 1.17 | 56 / 74 | 0.76 | slower |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.2 / 4.5 | 1.17 | 37 / 66 | 0.56 | slower |
| deeplabv3plus-r101-d8-ade20k | fp32 | 347.5 / 377.6 | 0.92 | 1404 / 1209 | 1.16 | heavier |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.7 / 5.8 | 1.16 | 69 / 108 | 0.64 | slower |
| sam2.1-hiera-large | fp16 | 555.8 / 562.2 | 0.99 | 849 / 734 | 1.16 | heavier |
| detr-r50-coco | fp32 | 85.2 / 95.1 | 0.90 | 524 / 450 | 1.16 | heavier (accepted) |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 622 / 539 | 1.15 | heavier (accepted) |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 613 / 531 | 1.15 | heavier (accepted) |
| pe-spatial-b16-512 | fp32 | 16.7 / 17.2 | 0.97 | 471 / 409 | 1.15 | heavier |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 581 / 505 | 1.15 | heavier (accepted) |
| totalsegmentator-total-3mm | fp32 | 414.7 / 535.0 | 0.78 | 2033 / 1769 | 1.15 | heavier |
| squeezenet1_1.tv_in1k | fp32 | 2.0 / 1.7 | 1.15 | 28 / 32 | 0.86 | slower |
| mobilenet_v3_large.tv2_in1k | fp32 | 3.7 / 3.2 | 1.15 | 34 / 53 | 0.64 | slower |
| deeplabv3plus-r101-d8-ade20k | fp16 | 268.6 / 290.9 | 0.92 | 773 / 674 | 1.15 | heavier |
| vitpose-base | fp32 | 160.6 / 164.4 | 0.98 | 540 / 469 | 1.15 | heavier |
| rf-detr-seg-xl-coco | fp32 | 96.5 / 95.4 | 1.01 | 377 / 331 | 1.14 | heavier |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 1119 / 982 | 1.14 | heavier (accepted) |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 1107 / 972 | 1.14 | heavier (accepted) |
| pe-core-b16-224 | fp32 | 17.3 / 17.6 | 0.98 | 496 / 436 | 1.14 | heavier |
| sam2.1-hiera-large | fp32 | 600.5 / 608.8 | 0.99 | 1495 / 1317 | 1.14 | heavier |
| vitpose-plus-base | fp32 | 160.2 / 165.6 | 0.97 | 535 / 469 | 1.14 | heavier |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 24.0 / 21.2 | 1.13 | 139 / 233 | 0.60 | slower |
| wd-vit-tagger-v3 | fp16 | 44.8 / 43.9 | 1.02 | 283 / 251 | 1.13 | heavier |
| igev-plusplus-eth3d | fp32 | 6887.7 / 7321.9 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti12 | fp32 | 6895.0 / 7199.4 | 0.96 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-kitti15 | fp32 | 6871.0 / 7144.8 | 0.96 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-middlebury | fp32 | 6856.9 / 7342.5 | 0.93 | 4260 / 3786 | 1.13 | heavier |
| igev-plusplus-sceneflow | fp32 | 6821.8 / 7224.3 | 0.94 | 4260 / 3786 | 1.13 | heavier |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 16.9 / 17.5 | 0.97 | 324 / 289 | 1.12 | heavier |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 15.9 / 14.2 | 1.12 | 195 / 224 | 0.87 | slower |
| sam2.1-hiera-base-plus | fp16 | 232.8 / 245.5 | 0.95 | 535 / 478 | 1.12 | heavier |
| yunet-s | fp32 | 10.5 / 13.1 | 0.80 | 303 / 271 | 1.12 | heavier (accepted) |
| shvit_s1.in1k | fp16 | 3.3 / 3.0 | 1.11 | 16 / 20 | 0.80 | slower |
| convnext_tiny.fb_in1k | fp32 | 7.5 / 6.8 | 1.11 | 135 / 206 | 0.66 | slower |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 25.2 / 25.1 | 1.00 | 333 / 300 | 1.11 | heavier |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 81 / 74 | 1.11 | heavier (accepted) |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 7.5 / 6.8 | 1.11 | 136 / 206 | 0.66 | slower |
| davit_small.msft_in1k | fp16 | 15.3 / 13.9 | 1.10 | 135 / 178 | 0.76 | slower |
| sapiens-seg-0.3b-goliath | fp32 | 1399.4 / 1783.6 | 0.78 | 3631 / 3288 | 1.10 | heavier |
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 2112 / 1914 | 1.10 | heavier (accepted) |
| igev-plusplus-kitti12 | fp16 | 5648.8 / 6263.4 | 0.90 | 2414 / 2189 | 1.10 | heavier |
| igev-plusplus-middlebury | fp16 | 5711.5 / 6166.5 | 0.93 | 2414 / 2189 | 1.10 | heavier |
| igev-plusplus-eth3d | fp16 | 5647.5 / 7058.8 | 0.80 | 2414 / 2189 | 1.10 | heavier (accepted) |

## Accepted: faster, peak trade

A hand network clearly faster than the graph translation (warm at most 0.90 of it) may peak higher on the network
call when that buys the speed (overlap, fused or float32 streams the goldens need), provided its end-to-end peak (the
MLX survey row) stays below the best MPS row and below its own MLX float32 row. These flagged rows meet that and are
accepted as they are: no effort spent shrinking them. Peak wins that cost nothing are still taken. The larger-input
condition (the peak must not grow out of proportion) is read off sibling refs at larger crops where the family has
them (below the table).

| ref | dtype | net warm ms hand / graph | x | net peak MB hand / graph | x | e2e peak MB | best MPS MB | MLX fp32 MB |
|---|---|---|---|---|---|---|---|---|
| aanet-plus-sceneflow | fp16 | 639.7 / 1224.1 | 0.52 | 2112 / 1914 | 1.10 | 2259 | 3836 | 3104 |
| aanet-sceneflow | fp16 | 443.1 / 1077.9 | 0.41 | 1062 / 900 | 1.18 | 1362 | 2594 | 2140 |
| aanet-sceneflow | fp32 | 636.4 / 1177.7 | 0.54 | 2072 / 855 | 2.42 | 2140 | 2594 | 2140 |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp32 | 47.0 / 60.3 | 0.78 | 864 / 823 | 1.05 | 1447 | 2013 | 1447 |
| dat-light-x4 (portable) | fp32 | 1241.8 / 1651.2 | 0.75 | 2385 / 1899 | 1.26 | 3919 | 6317 | 3919 |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 81 / 74 | 1.11 | 416 | 704 | 476 |
| densenet121.tv_in1k | fp32 | 9.6 / 12.4 | 0.77 | 148 / 142 | 1.04 | 476 | 704 | 476 |
| deim-l-coco | fp32 | 46.7 / 54.4 | 0.86 | 483 / 393 | 1.23 | 972 | 1806 | 972 |
| deim-x-coco | fp32 | 80.3 / 90.3 | 0.89 | 740 / 710 | 1.04 | 1288 | 2207 | 1288 |
| detr-r50-coco | fp32 | 85.2 / 95.1 | 0.90 | 524 / 450 | 1.16 | 1603 | 1954 | 1603 |
| dfine-l-coco | fp32 | 46.5 / 54.2 | 0.86 | 488 / 393 | 1.24 | 1024 | 1810 | 1024 |
| dfine-x-coco | fp32 | 80.4 / 90.3 | 0.89 | 740 / 710 | 1.04 | 1422 | 2211 | 1422 |
| dlnr-middlebury | fp16 | 1609.2 / 1927.6 | 0.83 | 1647 / 1209 | 1.36 | 2841 | 4149 | 3856 |
| dlnr-middlebury | fp32 | 2050.9 / 2774.3 | 0.74 | 2901 / 2261 | 1.28 | 3856 | 4149 | 3856 |
| dlnr-sceneflow | fp16 | 1614.9 / 1942.1 | 0.83 | 1647 / 1208 | 1.36 | 2844 | 4146 | 3849 |
| dlnr-sceneflow | fp32 | 2027.6 / 2769.5 | 0.73 | 2901 / 2261 | 1.28 | 3849 | 4146 | 3849 |
| hrnet-w32 | fp32 | 100.8 / 120.5 | 0.84 | 453 / 420 | 1.08 | 901 | 1675 | 901 |
| hrnet-w48 | fp16 | 182.1 / 205.9 | 0.88 | 285 / 268 | 1.06 | 746 | 2065 | 1150 |
| hrnet-w48 | fp32 | 209.2 / 245.7 | 0.85 | 731 / 710 | 1.03 | 1150 | 2065 | 1150 |
| igev-plusplus-eth3d | fp16 | 5647.5 / 7058.8 | 0.80 | 2414 / 2189 | 1.10 | 2953 | 6040 | 4217 |
| igev-plusplus-kitti15 | fp16 | 5636.4 / 7098.7 | 0.79 | 2414 / 2199 | 1.10 | 2962 | 6034 | 3930 |
| igev-plusplus-sceneflow | fp16 | 5617.9 / 7204.4 | 0.78 | 2414 / 2199 | 1.10 | 2949 | 6104 | 3981 |
| igev-stereo-eth3d | fp16 | 3185.1 / 3594.5 | 0.89 | 1686 / 1080 | 1.56 | 2546 | 3862 | 3447 |
| igev-stereo-eth3d | fp32 | 4179.4 / 4676.3 | 0.89 | 3122 / 2055 | 1.52 | 3447 | 3862 | 3447 |
| igev-stereo-kitti12 | fp16 | 3190.2 / 3603.2 | 0.89 | 1686 / 1080 | 1.56 | 2542 | 3862 | 3449 |
| igev-stereo-kitti15 | fp16 | 3186.4 / 3601.5 | 0.88 | 1686 / 1080 | 1.56 | 2575 | 3862 | 3461 |
| igev-stereo-middlebury | fp16 | 3187.7 / 3596.0 | 0.89 | 1686 / 1010 | 1.67 | 2548 | 3861 | 3418 |
| igev-stereo-sceneflow | fp16 | 3187.4 / 3609.1 | 0.88 | 1686 / 1010 | 1.67 | 2568 | 3867 | 3420 |
| lineart | fp32 | 98.6 / 117.6 | 0.84 | 1245 / 903 | 1.38 | 815 | 1361 | 815 |
| lineart-coarse | fp32 | 96.5 / 117.3 | 0.82 | 1245 / 903 | 1.38 | 782 | 1362 | 782 |
| lw-detr-xlarge-coco | fp16 | 96.9 / 108.3 | 0.90 | 698 / 445 | 1.57 | 1427 | 1765 | 1561 |
| nafnet-gopro | fp32 | 257.5 / 291.6 | 0.88 | 1170 / 839 | 1.39 | 1591 | 1715 | 1591 |
| oneformer-swinl-ade20k | fp16 | 643.5 / 760.8 | 0.85 | 1364 / 1002 | 1.36 | 2137 | 5559 | 2917 |
| raft-stereo-eth3d | fp16 | 3176.0 / 4061.6 | 0.78 | 1630 / 1166 | 1.40 | 3296 | 4714 | 4163 |
| raft-stereo-middlebury | fp16 | 3168.7 / 4054.7 | 0.78 | 1630 / 1128 | 1.45 | 3286 | 4695 | 4165 |
| raft-stereo-sceneflow | fp16 | 3191.9 / 4043.8 | 0.79 | 1630 / 1042 | 1.56 | 3284 | 4695 | 4168 |
| retinaface-resnet50 | fp16 | 119.6 / 132.8 | 0.90 | 469 / 372 | 1.26 | 936 | 1871 | 1291 |
| retinaface-resnet50 | fp32 | 151.1 / 171.8 | 0.88 | 902 / 697 | 1.29 | 1291 | 1871 | 1291 |
| rtdetrv2-l-coco | fp32 | 55.3 / 63.4 | 0.87 | 455 / 424 | 1.07 | 1027 | 1823 | 1027 |
| rtdetrv2-ms-coco | fp32 | 44.5 / 51.5 | 0.86 | 412 / 391 | 1.05 | 958 | 1821 | 958 |
| rtdetrv2-x-coco | fp32 | 92.5 / 103.2 | 0.90 | 736 / 679 | 1.08 | 1416 | 2017 | 1416 |
| dwpose-l | fp16 | 56.2 / 66.0 | 0.85 | 332 / 290 | 1.15 | 1144 | 1866 | 1397 |
| dwpose-l | fp32 | 62.2 / 79.0 | 0.79 | 587 / 454 | 1.29 | 1397 | 1866 | 1397 |
| dwpose-l-384 | fp16 | 109.3 / 132.1 | 0.83 | 409 / 376 | 1.09 | 1232 | 1872 | 1340 |
| dwpose-l-384 | fp32 | 122.6 / 163.0 | 0.75 | 765 / 518 | 1.48 | 1340 | 1872 | 1340 |
| dwpose-s | fp32 | 18.8 / 24.2 | 0.78 | 286 / 256 | 1.12 | 727 | 1543 | 727 |
| rtmpose-l | fp16 | 48.6 / 60.8 | 0.80 | 310 / 267 | 1.16 | 1054 | 1841 | 1325 |
| rtmpose-l | fp32 | 54.5 / 73.8 | 0.74 | 565 / 431 | 1.31 | 1325 | 1841 | 1325 |
| rtmpose-l-384 | fp16 | 95.3 / 118.9 | 0.80 | 387 / 354 | 1.09 | 1159 | 1844 | 1313 |
| rtmpose-l-384 | fp32 | 107.9 / 149.1 | 0.72 | 743 / 496 | 1.50 | 1313 | 1844 | 1313 |
| rtmpose-l-halpe26 | fp16 | 48.6 / 61.0 | 0.80 | 311 / 269 | 1.16 | 1077 | 1847 | 1326 |
| rtmpose-l-halpe26 | fp32 | 54.6 / 73.9 | 0.74 | 566 / 433 | 1.31 | 1326 | 1847 | 1326 |
| rtmpose-l-halpe26-384 | fp16 | 95.6 / 118.8 | 0.80 | 389 / 356 | 1.09 | 1164 | 1851 | 1353 |
| rtmpose-l-halpe26-384 | fp32 | 108.4 / 149.3 | 0.73 | 745 / 510 | 1.46 | 1353 | 1851 | 1353 |
| rtmpose-s-face6 | fp16 | 20.6 / 24.8 | 0.83 | 204 / 179 | 1.14 | 709 | 1518 | 823 |
| rtmpose-s-face6 | fp32 | 21.9 / 29.0 | 0.76 | 329 / 248 | 1.33 | 823 | 1518 | 823 |
| rtmpose-s-halpe26 | fp32 | 14.6 / 22.1 | 0.66 | 251 / 246 | 1.02 | 673 | 1500 | 673 |
| rtmw-l | fp32 | 111.7 / 125.7 | 0.89 | 678 / 544 | 1.25 | 1550 | 2040 | 1550 |
| rtmw-l-384 | fp16 | 206.4 / 235.8 | 0.88 | 470 / 438 | 1.07 | 1343 | 2076 | 1644 |
| rtmw-l-384 | fp32 | 224.6 / 272.5 | 0.82 | 856 / 634 | 1.35 | 1644 | 2076 | 1644 |
| rtmw-x | fp16 | 176.2 / 206.2 | 0.85 | 560 / 474 | 1.18 | 1605 | 2266 | 2033 |
| rtmw-x | fp32 | 203.7 / 240.3 | 0.85 | 968 / 830 | 1.17 | 2033 | 2266 | 2033 |
| rtmw-x-384 | fp32 | 419.5 / 490.0 | 0.86 | 1174 / 1093 | 1.07 | 2216 | 2268 | 2216 |
| sam-vit-b | fp16 | 299.6 / 383.8 | 0.78 | 613 / 531 | 1.15 | 1069 | 3762 | 1412 |
| sam-vit-l | fp16 | 815.8 / 931.0 | 0.88 | 1107 / 972 | 1.14 | 1639 | 5193 | 2342 |
| sam-hq-vit-b | fp16 | 303.6 / 393.4 | 0.77 | 622 / 539 | 1.15 | 1374 | 4374 | 1712 |
| sam-hq-vit-l | fp16 | 825.1 / 939.4 | 0.88 | 1119 / 982 | 1.14 | 1884 | 6205 | 2571 |
| selective-igev-eth3d-finetune | fp16 | 3659.3 / 4639.0 | 0.79 | 1873 / 1102 | 1.70 | 2774 | 3854 | 3935 |
| selective-igev-eth3d-train | fp16 | 3650.8 / 4622.7 | 0.79 | 1873 / 1102 | 1.70 | 2763 | 3862 | 3937 |
| selective-igev-kitti12 | fp16 | 3642.7 / 4624.8 | 0.79 | 1873 / 1102 | 1.70 | 2778 | 3853 | 3935 |
| selective-igev-kitti15 | fp16 | 3648.9 / 4623.3 | 0.79 | 1873 / 1102 | 1.70 | 2777 | 3853 | 3980 |
| selective-igev-middlebury-finetune | fp16 | 3646.4 / 4626.4 | 0.79 | 1873 / 1102 | 1.70 | 2779 | 3852 | 3938 |
| selective-igev-middlebury-train | fp16 | 3642.6 / 4641.0 | 0.78 | 1873 / 1102 | 1.70 | 2762 | 3854 | 3937 |
| selective-igev-sceneflow | fp16 | 3662.8 / 4645.5 | 0.79 | 1873 / 1102 | 1.70 | 2779 | 3946 | 3938 |
| selective-igev-sceneflow | fp32 | 4796.5 / 6299.9 | 0.76 | 3498 / 2057 | 1.70 | 3938 | 3946 | 3938 |
| span-x2 (kernels) | fp32 | 113.4 / 149.7 | 0.76 | 661 / 613 | 1.08 | 1024 | 1296 | 1024 |
| span-x2 (portable) | fp32 | 113.4 / 149.2 | 0.76 | 661 / 613 | 1.08 | 977 | 1296 | 977 |
| span-x4 (kernels) | fp32 | 116.7 / 156.6 | 0.75 | 661 / 613 | 1.08 | 1116 | 1296 | 1116 |
| span-x4 (portable) | fp32 | 116.6 / 156.3 | 0.75 | 661 / 613 | 1.08 | 1068 | 1296 | 1068 |
| xfeat | fp32 | 16.9 / 31.8 | 0.53 | 199 / 165 | 1.21 | 642 | 1471 | 642 |
| yolox-l | fp16 | 51.0 / 61.3 | 0.83 | 293 / 280 | 1.05 | 1021 | 1829 | 1307 |
| yolox-l | fp32 | 60.3 / 75.8 | 0.80 | 581 / 505 | 1.15 | 1307 | 1829 | 1307 |
| yolox-x | fp32 | 108.8 / 128.4 | 0.85 | 829 / 802 | 1.03 | 1675 | 2036 | 1675 |
| yunet-n | fp32 | 12.0 / 14.7 | 0.81 | 367 / 271 | 1.35 | 745 | 1377 | 745 |
| yunet-s | fp32 | 10.5 / 13.1 | 0.80 | 303 / 271 | 1.12 | 693 | 1380 | 693 |

Larger input, network peak hand / graph at the smaller and the larger crop of the same network (the other
accepted families have one crop in the survey):

- rtmpose-l 256x192 -> 384x288: fp16 1.16 -> 1.09, fp32 1.31 -> 1.50 (565 -> 743 MB for 2.25x the pixels, the graph
  431 -> 496); rtmpose-l-halpe26 and dwpose-l the same (fp32 1.29-1.31 -> 1.46-1.48). Sublinear in the pixels, so
  accepted, but the float32 ratio grows with the crop: the next pose round should look at it.
- rtmw-l: fp16 1.12 -> 1.07, fp32 1.25 -> 1.35; rtmw-x: fp16 1.18 -> 0.86, fp32 1.17 -> 1.07.
- lw-detr large -> xlarge (a larger network, 640 both): fp16 1.67 -> 1.57.
- hrnet-w32 256x192 -> 384x288: fp32 1.08 -> 0.96; hrnet-w48: fp16 1.06 -> 1.08 (the 384 row is not accepted:
  warm 0.92), fp32 1.03 -> 0.97. DETR, RetinaFace and YuNet have one input size each; YuNet float32's +96 MB
  network peak is at 640 x 640 (e2e 745 MB).

## Per family

Flagged rows / rows (accepted peak trades in brackets), the worst hand / graph ratio of the family's rows (warm,
peak); every flagged row is marked in the full table below.

| family | flagged (accepted) | slower | heavier | worst warm x | worst peak x |
|---|---|---|---|---|---|
| aanet | 3 (3) / 4 | 0 | 3 | 0.54 | 2.42 |
| alexnet | 0 (0) / 2 | 0 | 0 | 0.99 | 0.96 |
| anyline | 0 (0) / 2 | 0 | 0 | 0.64 | 0.65 |
| clip | 13 (0) / 20 | 13 | 0 | 1.04 | 0.98 |
| coatnet | 2 (1) / 20 | 1 | 1 | 1.07 | 1.05 |
| coex | 0 (0) / 2 | 0 | 0 | 0.50 | 0.53 |
| compact | 0 (0) / 4 | 0 | 0 | 0.93 | 1.00 |
| convnext | 15 (0) / 24 | 14 | 1 | 1.53 | 1.03 |
| convnextv2 | 1 (0) / 2 | 1 | 0 | 1.04 | 0.92 |
| dat | 1 (1) / 6 | 0 | 1 | 0.75 | 1.26 |
| davit | 6 (0) / 6 | 5 | 1 | 1.22 | 1.04 |
| deit | 2 (0) / 2 | 1 | 2 | 1.03 | 1.08 |
| densenet | 2 (2) / 2 | 0 | 2 | 0.77 | 1.11 |
| depth_anything_3 | 8 (0) / 10 | 8 | 0 | 1.06 | 1.00 |
| depth_pro | 0 (0) / 2 | 0 | 0 | 0.96 | 0.98 |
| detr | 2 (1) / 4 | 0 | 2 | 0.92 | 1.16 |
| dfine | 4 (4) / 20 | 0 | 4 | 0.92 | 1.24 |
| dlnr | 4 (4) / 4 | 0 | 4 | 0.83 | 1.36 |
| drct | 0 (0) / 4 | 0 | 0 | 0.72 | 0.43 |
| efficientformer | 0 (0) / 6 | 0 | 0 | 0.86 | 0.82 |
| efficientformerv2 | 1 (0) / 8 | 1 | 0 | 1.17 | 0.72 |
| efficientnet | 0 (0) / 4 | 0 | 0 | 0.83 | 0.77 |
| efficientnetv2 | 0 (0) / 2 | 0 | 0 | 0.87 | 0.60 |
| efficientvit | 0 (0) / 2 | 0 | 0 | 0.79 | 0.74 |
| efficientvit_sam | 2 (0) / 5 | 0 | 2 | 1.02 | 1.32 |
| eomt | 3 (0) / 4 | 1 | 2 | 1.02 | 1.03 |
| eva02 | 2 (0) / 2 | 0 | 2 | 0.99 | 1.08 |
| fastvit | 0 (0) / 6 | 0 | 0 | 0.99 | 0.62 |
| flexglue | 0 (0) / 2 | 0 | 0 | 0.98 | 0.80 |
| focalnet | 1 (0) / 12 | 1 | 0 | 1.04 | 0.85 |
| hat | 0 (0) / 6 | 0 | 0 | 0.78 | 1.00 |
| hgnetv2 | 0 (0) / 14 | 0 | 0 | 0.95 | 0.81 |
| hrnet | 4 (3) / 8 | 0 | 4 | 0.92 | 1.08 |
| igev_plusplus | 10 (3) / 10 | 0 | 10 | 0.96 | 1.13 |
| igev_stereo | 10 (6) / 10 | 0 | 10 | 0.91 | 1.67 |
| inception_v3 | 0 (0) / 2 | 0 | 0 | 0.76 | 0.46 |
| lightglue | 2 (0) / 4 | 1 | 2 | 1.03 | 1.05 |
| lineart | 4 (2) / 4 | 0 | 4 | 0.84 | 1.38 |
| loftr | 4 (0) / 4 | 4 | 4 | 1.31 | 2.30 |
| loma | 2 (0) / 12 | 1 | 2 | 1.03 | 1.18 |
| lwdetr | 3 (1) / 10 | 0 | 3 | 0.99 | 1.67 |
| mask2former | 0 (0) / 2 | 0 | 0 | 0.91 | 0.97 |
| mmseg | 8 (0) / 10 | 1 | 7 | 1.03 | 1.38 |
| mobilenetv1 | 1 (0) / 2 | 1 | 0 | 1.02 | 0.72 |
| mobilenetv2 | 0 (0) / 2 | 0 | 0 | 0.97 | 0.96 |
| mobilenetv3 | 1 (0) / 2 | 1 | 0 | 1.15 | 0.64 |
| mobilenetv4 | 1 (0) / 2 | 1 | 0 | 1.17 | 0.76 |
| mvsanywhere | 2 (0) / 2 | 2 | 0 | 1.11 | 1.00 |
| nafnet | 3 (1) / 4 | 1 | 3 | 1.06 | 1.50 |
| nsfw_vit | 2 (0) / 2 | 1 | 2 | 1.03 | 1.08 |
| oneformer | 1 (1) / 2 | 0 | 1 | 0.85 | 1.36 |
| owlv2 | 4 (0) / 4 | 0 | 4 | 1.01 | 1.58 |
| perception_encoder | 6 (0) / 20 | 1 | 6 | 1.03 | 1.15 |
| raft_stereo | 6 (3) / 6 | 0 | 6 | 0.96 | 1.57 |
| rcnn | 4 (0) / 4 | 0 | 4 | 0.95 | 1.74 |
| realesrgan | 3 (0) / 12 | 1 | 3 | 1.04 | 1.20 |
| regnet | 0 (0) / 2 | 0 | 0 | 0.94 | 0.71 |
| repvit | 0 (0) / 20 | 0 | 0 | 0.79 | 0.62 |
| resnet | 0 (0) / 6 | 0 | 0 | 0.91 | 0.87 |
| retinaface | 2 (2) / 4 | 0 | 2 | 0.90 | 1.29 |
| rfdetr | 20 (0) / 20 | 18 | 6 | 1.09 | 1.35 |
| rfdetr_keypoint | 2 (0) / 2 | 0 | 2 | 1.02 | 1.71 |
| rtdetrv2 | 3 (3) / 10 | 0 | 3 | 0.92 | 1.08 |
| rtmdet | 0 (0) / 14 | 0 | 0 | 0.92 | 0.97 |
| rtmo | 0 (0) / 8 | 0 | 0 | 0.84 | 1.02 |
| rtmpose | 25 (22) / 56 | 0 | 25 | 0.98 | 1.50 |
| sam | 3 (2) / 6 | 0 | 3 | 0.90 | 1.15 |
| sam2 | 8 (0) / 8 | 1 | 8 | 1.02 | 1.31 |
| sam3 | 2 (0) / 2 | 2 | 2 | 1.09 | 1.56 |
| sam_hq | 3 (2) / 6 | 0 | 3 | 0.90 | 1.15 |
| sapiens | 2 (0) / 4 | 0 | 2 | 0.88 | 1.10 |
| segformer | 8 (0) / 8 | 6 | 8 | 1.11 | 1.81 |
| segnext | 8 (0) / 8 | 5 | 8 | 1.21 | 1.61 |
| selective_igev | 14 (8) / 14 | 0 | 14 | 0.79 | 1.70 |
| shufflenetv2 | 1 (0) / 2 | 1 | 0 | 1.39 | 0.90 |
| shvit | 8 (0) / 8 | 8 | 0 | 1.11 | 0.86 |
| siglip | 1 (0) / 2 | 1 | 0 | 1.02 | 1.02 |
| siglip2 | 11 (0) / 20 | 11 | 0 | 1.05 | 1.01 |
| span | 4 (4) / 8 | 0 | 4 | 0.84 | 1.08 |
| squeezenet | 1 (0) / 2 | 1 | 0 | 1.15 | 0.86 |
| starnet | 0 (0) / 8 | 0 | 0 | 0.87 | 0.62 |
| swin | 1 (0) / 2 | 0 | 1 | 0.99 | 1.03 |
| swin2sr | 3 (0) / 6 | 3 | 0 | 1.07 | 0.76 |
| swinir | 2 (0) / 8 | 2 | 0 | 1.07 | 0.85 |
| tinyvit | 3 (0) / 10 | 0 | 3 | 1.01 | 1.30 |
| totalsegmentator | 2 (0) / 4 | 0 | 2 | 0.78 | 1.28 |
| unik3d | 1 (0) / 2 | 0 | 1 | 1.02 | 1.31 |
| vgg | 0 (0) / 2 | 0 | 0 | 0.97 | 0.91 |
| vit | 2 (0) / 2 | 1 | 2 | 1.03 | 1.08 |
| vitpose | 6 (0) / 12 | 3 | 3 | 1.03 | 1.15 |
| wd_tagger | 7 (0) / 8 | 2 | 5 | 1.05 | 1.29 |
| xfeat | 2 (1) / 2 | 1 | 2 | 4.07 | 2.77 |
| yolox | 3 (3) / 10 | 0 | 3 | 0.89 | 1.15 |
| yunet | 2 (2) / 4 | 0 | 2 | 0.84 | 1.35 |

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

## Restoration ports (2026-11-27)

SwinIR, Swin2SR, HAT, DRCT, DAT, Real-ESRGAN, SPAN, Compact and Restormer are hand-written (no graph replay):
`fastforks.shared.swin_sr.mlx_net` holds the Swin layers, the groups and the row-banded tails, `mlx_cnn` the 3 x 3
convs of the CNNs (Winograd `conv_mlx` in kernels mode), each family's `mlx_net` only its own blocks. All load
through `mlx_port.recorded` (the conversion cache, no checkpoint on a hit); `GRAPH_RUNNERS` rebuilds the graph arm
of every family but Restormer (below). 54 A/B rows (the CNNs in both modes, the Swin nets in portable mode, the only
one they have); the full table has no mode column: of a CNN ref's two rows per dtype, the kernels row comes first.

| family | net warm x fp16 | net peak x fp16 | net warm x fp32 | net peak x fp32 |
|---|---|---|---|---|
| swinir | 1.02-1.07 | 0.46-0.80 | 0.78-0.99 | 0.33-0.85 |
| swin2sr | 1.03-1.07 | 0.46-0.73 | 0.78-0.80 | 0.36-0.76 |
| hat | 0.66-0.78 | 0.73-0.91 | 0.42-0.48 | 0.82-1.00 |
| drct | 0.69-0.72 | 0.42-0.43 | 0.55-0.57 | 0.37-0.38 |
| dat | 0.31-0.53 | 0.62-0.74 | 0.48-0.75 | 0.95-1.26 |
| realesrgan kernels | 0.99-1.04 | 1.02-1.20 | 0.92-0.94 | 0.57-0.73 |
| realesrgan portable | 0.94-0.96 | 0.68-0.92 | 0.91-0.94 | 0.83-0.90 |
| span (both modes) | 0.81-0.84 | 0.78-1.00 | 0.75-0.76 | 1.08 |
| compact kernels / portable | 0.92 / 0.77 | 0.82 / 0.71 | 0.93 / 0.71 | 1.00 / 0.82 |

- *The buffer cache the mode asks for.* The graph ports' kernels runners asked `request_cache` for a larger MLX
  buffer cache (Real-ESRGAN x4plus 768 MB, anime 512, Compact and SPAN 512); the first hand runners did not, so the
  survey ran them at the 128 MB default: x4plus kernels float16 3907 ms against 2090 at 1024 MB (anime 1431 / 742,
  Compact float32 282 / 151). The hand runners now ask per variant (`CACHE_MB`: x4plus 1024, anime 512, x2plus 128,
  SPAN and Compact 512, Restormer 1024 as before).
- *Load.* A torch op on a meta tensor (the RDB's `0.2 * conv5` scale, DAT's BatchNorm fold, Swin2SR's bias MLP)
  imports `torch._dynamo` and `torch.distributed`, about 0.7 s: the scale is now applied in the conversion and the
  folds skip a replay's meta tensors. The Swin nets' skeletons still pay it once (the vendored modules' `torch.arange`
  on the meta device goes through the decomposition that imports it): their cached load is about 1.1-1.3 s in a
  fresh process against 0.2-0.4 s for the CNNs.
- *Restormer* was already hand-written; it now loads its folded, padded arrays from the conversion cache too
  (`mlx_port.tensor` by final shape): float32 load 0.77 -> 0.33 s. Its graph path stays gone (below), so it has no
  A/B rows; its e2e rows are below.

Still flagged, with the reason:

- SwinIR and Swin2SR float16: 2-7 % slower warm, 20-54 % leaner (float32 1-22 % faster). The slower part was not
  located this round.
- Real-ESRGAN x2plus kernels float16: peak 1.20 (1018 / 847 MB) at equal speed. One step per dense block against one
  per conv (cache 2048 MB): 479 / 485 ms at +885 / +885-970 MB, so the step granularity is not the cause; the rest
  was not located. x4plus and anime kernels float16: 1.00-1.04 slower and 1.02-1.03 heavier, within the drift above.
- DAT-light float32 (warm 0.75, peak 1.26) and SPAN float32 (0.75-0.76, 1.08): accepted peak trades (their e2e peaks
  are their own float32 rows, below the best MPS row). At 1024 x 1024 DAT-light float32 stays below the graph arm
  (MLX peak 6854 / 7428 MB, 12.2 / 17.5 s), SPAN-x4 float32 reaches 2382 MB (float16 1389).

End to end (`tools/speed_survey.py`, two runs, 512 x 512, the second run's rows kept), against the graph port's
MLX rows of 2026-11-19 to 25 (the old row) and the best MPS row:

| family | e2e warm new / old | e2e peak new / old | below the best MPS peak |
|---|---|---|---|
| swinir | 0.64-0.83 | 0.34-0.70 | all |
| swin2sr | 0.61-0.81 | 0.37-0.79 | all |
| hat | 0.46-0.66 | 0.60-0.94 | all |
| drct | 0.39-0.44 | 0.44-0.50 | all |
| dat | 0.33-0.70 | 0.88-1.17 | all |
| realesrgan | 0.66-1.00 | 0.57-0.95 | all |
| span | 0.68-0.82 | 0.76-0.92 | all |
| compact kernels / portable | 0.93 / 0.71-0.79 | 0.91-1.09 / 0.70-0.78 | all but compact kernels float32 |
| restormer (no graph arm) | 0.98-1.00 | 0.92-0.98 | all |

- Peaks above the old row: DAT float16 (dat-x4 3572 / 3305 MB, dat-2-x4 3643 / 3512, dat-light 2751 / 2425),
  DAT-light float32 (3919 / 3346) and Compact kernels float32 (1461 / 1335, best MPS 1289). The old rows do not
  reproduce: the graph arm (`FASTFORKS_MLX_GRAPH=1`) through the same survey worker on the same day peaks higher
  than the hand port in float16 (dat-light 3389 against 2784 MB, dat-x4 4265 / 3723, dat-2-x4 4332 / 3626) and in
  Compact kernels float32 (1477-1514 against 1444-1460, at 233-245 / 162-173 ms); only DAT-light float32 is
  heavier than it (3845 / 3324, the accepted trade). A smaller buffer cache does not lower Compact's float32 peak
  without the speed (512 / 256 / 128 MB: 159 / 183 / 268 ms at 1358 / 1346 / 1204 MB in one process).
- Load (`load_s`, torch already imported): the CNNs 0.16-0.35 s (old 0.10-0.27), the float16 kernels runners
  up to 0.54 (x2plus 0.51-0.54 against the graph arm's 0.30): their cache file holds the Winograd-transformed
  weights, 4x the plain ones (x2plus 167 / 33 MB), read from the cache disk. The Swin nets 0.76-1.15 s (old
  0.29-1.16): the skeleton's dynamo import above. Restormer float32 0.32-0.34 (old 0.76-0.78). The first call is
  far shorter everywhere but Restormer (no compile of a replayed graph: swinir-m-x4 float16 3.5 / 8.0 s,
  drct-l-x4 8.5 / 68.9 s).

## Detection / pose ports (2026-11-27)

ViTPose, HRNet, DETR, OWLv2, EoMT, RetinaFace and YuNet are hand-written (no graph replay). Each family's `mlx_net`
holds only its own blocks: the CNN backbones run through `mlx_cnn` and are stepped with `cspnext.mlx_net.stepped`
(HRNet, DETR's ResNet, RetinaFace, YuNet), and the ViTs use `mlx_vit`'s block maths (ViTPose, OWLv2, EoMT on the
DINOv2 / v3 backbones). They load through `mlx_port.recorded` / `tensor` (the conversion cache, no checkpoint on a
hit), do their pre- and post-processing in numpy / MLX, and ask `request_cache` for a cache sized per variant.
`GRAPH_RUNNERS` rebuilds the graph arm of every family. 40 A/B rows (portable, both dtypes). The goldens pass in
float16 and float32, and float16 is auto's pick.

| family | net warm x fp16 | net peak x fp16 | net warm x fp32 | net peak x fp32 |
|---|---|---|---|---|
| vitpose | 0.99-1.03 | 0.74-1.03 | 0.97-1.02 | 0.48-1.15 |
| hrnet | 0.87-0.92 | 0.86-1.08 | 0.83-0.87 | 0.96-1.08 |
| detr | 0.88-0.91 | 0.80-0.85 | 0.90-0.92 | 1.09-1.16 |
| owlv2 | 1.01 | 1.33-1.49 | 0.99-1.01 | 1.34-1.58 |
| eomt | 0.99-1.02 | 1.02-1.03 | 0.99-1.02 | 1.01-1.03 |
| retinaface | 0.89-0.90 | 0.52-1.26 | 0.88 | 0.76-1.29 |
| yunet | 0.83-0.84 | 0.83-0.93 | 0.80-0.81 | 1.12-1.35 |

- *Half blocks.* ViTPose steps its blocks in halves (attention, then MLP), and each half is `async_eval`'d while the
  previous one is awaited. That keeps the speed with about half the transients of `mlx_vit.blocks(bounded=True)`
  (ViTPose-B float32 141.0 ms at +262 MB -> 141.0 ms at +152 MB; one block per synchronous eval: 143.4 ms at +143 MB).
- *Post-processing outside the timed call.* EoMT's window averaging is a module function (`mlx_net.scores`), not a
  method, so the A/B times the same network in both arms.
- *Load.* No torch import on a cache hit. The A/B `load_s` is 0.04-0.11 s against the graph's 0.04-2.4 s. Three
  rows are higher: vitpose-plus-huge float32 1.17 / 0.69 s, owlv2-l14 float32 1.02 s, eomt-l-dinov3 float32 0.82 s.
  Those are large cache files read from the cache disk. The first network call takes 0.04-2.07 s against the
  graph's 0.95-4.2 s (no compile of a replayed graph).

Accepted peak trades (in the table above): detr-r50 float32, hrnet-w32 float32, hrnet-w48 both dtypes,
retinaface-resnet50 both dtypes, yunet-n and yunet-s float32.

Still flagged, with the reason:

- detr-r101 float32 (warm 0.92, peak 1.09) and hrnet-w48-384 float16 (0.92, 1.08) are faster but not by 10 %. The
  float32 peaks come from the stepped backbone's overlap (two stages in flight), which buys the speed.
- OWLv2, all four rows (peak 1.33-1.58 at equal speed): accounting. The hand port keeps the text tower in MLX
  (resident +121 / +242 MB at b16 float16 / float32, +236 / +472 MB at l14), while the graph arm ports only the image
  tower and leaves the text tower in torch. The hand arm also times three network calls per model call against the
  graph's one. Transients above resident are +45-74 MB. The e2e peak is lower than the old row everywhere but
  b16 float32 (1286 / 1253 MB).
- vitpose-base and vitpose-plus-base float32, peak 1.14-1.15 at 0.97-0.98 warm: the token preparation and the head
  on top of the half-block transients. Stepping the head (about -17 MB) and the tokens gave no net gain. Calling
  `mx.fast.scaled_dot_product_attention` directly gave nothing either, since head size 64 is already fused.
- ViTPose float16 base-simple, plus-small and plus-huge are 1.02-1.03 slower warm, and plus-large float16 is 1.03
  heavier. The cause was not located (GELU is already the fused one).
- EoMT, 1.01-1.03: drift level. Sharing the half-block stepper with `mlx_vit` would need an `mlx_vit` edit, which
  was left for another round (another port is editing it).
- grounding_dino was not ported: Swin + BERT + deformable attention + the fusion layers are too large for this
  round. It keeps its graph runner.

End to end (`tools/speed_survey.py`, two runs, the second run's rows kept; every row ok with the golden passing),
against the graph port's MLX row (the old row) and the best MPS row:

| family | e2e warm new / old | e2e peak new / old | below the best MPS peak |
|---|---|---|---|
| vitpose | 0.84-1.03 | 0.59-1.02 | all |
| hrnet | 0.77-0.93 | 0.80-0.90 | all |
| detr | 0.82-0.94 | 0.68-0.94 | all |
| owlv2 | 0.98-1.00 | 0.82-1.03 | all |
| eomt | 0.76-0.86 | 0.63-0.68 | all |
| retinaface | 0.66-0.95 | 0.76-0.83 | all |
| yunet | 0.62-0.79 | 0.78-0.96 | all |

- Above the old row: e2e warm for vitpose-plus-small float16 (1.03: 74.4 / 72.3 ms), plus-large and plus-huge
  float16 (1.02), base and plus-base float16 (1.01), and plus-huge and plus-large float32 (1.01). Peaks above it:
  owlv2-b16 float32 (1286 / 1253 MB) and vitpose-base float32 (1083 / 1066).
- MPS is faster warm on vitpose-plus-small (66.5 / 74.4 ms), hrnet-w48-384 (369 / 416 ms in float16) and
  retinaface-mobilenet0.25 float16 (15.3 / 15.8 ms). On vitpose-base and plus-base float32 the two are about equal
  (172-174 / 176 ms).
- Load (`load_s`, survey): the CNNs 0.15-0.51 s (old 0.16-0.92). EoMT 0.91-1.67 s (old 2.22-4.2). OWLv2 0.55-2.27 s
  (old 2.18-4.97). ViTPose takes longer than the old row: plus-huge 1.69 / 0.29 s in float16 and 3.21 / 2.66 in
  float32, plus-large 0.90 / 0.21 and 1.63 / 1.46. The old row loaded the replayed graph's weights lazily; the hand
  port reads its full cache file at load. The first call is 0.05-2.0 s against the old 1.3-5.9 s.

## SAM 2.1 prompt path and video, SAM 3 tracker (2026-11-27)

What still ran in PyTorch on the CPU in MLX mode is in `mx` now: SAM 2.1's prompt encoder and mask decoder
(`sam2/mlx_heads.py`, float32 as before), its video memory path (memory attention, memory encoder, object pointers:
`mlx_heads.py` in the run dtype, the loop in numpy / MLX in `sam2/mlx_video.py`) and SAM 3's tracker heads
(`sam3/net/mlx_tracker.py`). All weights come through `mlx_port.recorded`: a cache hit reads no checkpoint and builds
no float32 CPU twin. The graph arm is unchanged (`embed` alone, the prompt in PyTorch), so the SAM 2.1 rows now
count six network calls in the hand arm (embed, the prompt encoder's parts, the decoder: 5.5 ms and at most 98 MB
over the resident weights) against one, and its resident weights are 56-67 MB higher (the decoder and the dense
position embedding): that is most of the `heavier` flag. The embed call's peak over the resident weights is
236-348 MB against 276-320 MB in float16 (large alone is higher, by 48 MB) and 328-462 MB against 300-462 MB in
float32 (0-56 MB higher); the process footprint is 125-430 MB lower in every row. The sam3 rows are as in the
section above (image path untouched).

Image, end to end (`tools/speed_survey.py`, two runs, the second kept; before -> now):

| ref | dtype | warm ms | load s | peak footprint MB | load footprint MB |
|---|---|---|---|---|---|
| sam2.1-hiera-base-plus | fp16 | 233.8 -> 229.0 | 0.43 -> 0.38 | 1072 -> 911 | 624 -> 378 |
| sam2.1-hiera-base-plus | fp32 | 273.2 -> 271.8 | 0.97 -> 0.57 | 1311 -> 1153 | 713 -> 529 |
| sam2.1-hiera-large | fp16 | 522.8 -> 516.2 | 0.76 -> 0.77 | 1474 -> 1243 | 985 -> 654 |
| sam2.1-hiera-large | fp32 | 600.1 -> 597.1 | 2.28 -> 1.27 | 1986 -> 1737 | 1325 -> 1075 |
| sam2.1-hiera-small | fp16 | 136.6 -> 133.3 | 0.33 -> 0.29 | 931 -> 798 | 498 -> 304 |
| sam2.1-hiera-small | fp32 | 169.0 -> 168.5 | 0.67 -> 0.38 | 1075 -> 936 | 528 -> 380 |
| sam2.1-hiera-tiny | fp16 | 116.7 -> 114.7 | 0.31 -> 0.26 | 919 -> 785 | 486 -> 289 |
| sam2.1-hiera-tiny | fp32 | 146.2 -> 145.0 | 0.59 -> 0.34 | 1043 -> 905 | 501 -> 352 |
| sam3 | fp16 | 1543.8 -> 1546.6 | 2.24 -> 2.23 | 2590 -> 2582 | 1858 -> 1858 |
| sam3 | fp32 | 1901.6 -> 1893.9 | 4.21 -> 4.27 | 4501 -> 4472 | 3429 -> 3430 |

Per prompt on a cached embedding and per video frame (8 frames, one box prompt, warm second pass; a bench script, not
the survey; before -> now):

| ref | dtype | prompt ms (CPU ms) | 4 boxes ms | video ms / frame (CPU ms) | video MLX peak MB | video process footprint MB |
|---|---|---|---|---|---|---|
| sam2.1-hiera-tiny | fp16 | 10.8 (16.2) -> 8.1 (8.4) | 23.7 -> 18.8 | 348.6 (150) -> 330.4 (145) | 937 -> 1034 | 1527 -> 1384 |
| sam2.1-hiera-base-plus | fp16 | 10.1 (15.7) -> 8.4 (8.4) | 24.0 -> 19.0 | 467.8 (179) -> 446.5 (176) | 1025 -> 1136 | 1639 -> 1484 |
| sam2.1-hiera-base-plus | fp32 | 9.6 (15.3) -> 8.0 (8.3) | 22.1 -> 19.1 | 608.2 (337) -> 590.1 (336) | 1932 -> 2065 | 2339 -> 2242 |
| sam3 | fp16 | - | - | 2352.7 (4901) -> 1760.8 (316) | 2096 -> 2628 | 4205 -> 2977 |
| sam3 | fp32 | - | - | 2657.5 (5362) -> 2285.1 (666) | 4010 -> 4637 | 5418 -> 4904 |

- *Memory.* The MLX allocator's peak rises by the head weights that used to sit in PyTorch (about 100 MB for
  SAM 2.1, 530-630 MB for SAM 3); the process footprint, which counts both, falls in every row.
- *Load.* The replay skeleton of a cache hit ran torch ops that have no meta kernel (`torch.randn` in the prompt
  encoder's random position embedding, the neck fold's matmul, the position embedding's interpolate, `torch.arange`
  in the RoPE table, `scale * torch.ones` in the memory encoder's blocks): each goes through a Python decomposition
  that imports `torch._dynamo`, about 1 s once per process. The skeleton now gets the folded shapes directly
  (`_replay_shapes`) and builds the prompt encoder and the video modules unfilled on the CPU: cached load 0.63 ->
  0.23 s (tiny float16), 0.97 -> 0.53 s (base-plus float32).
- *Video is GPU-bound.* Of tiny's 330 ms / frame, 137 ms is the image encoder and about 200 ms the memory attention
  (four layers over up to seven frames of memory tokens, evaluated per layer); the host side (frame preparation in the
  prefetch thread, the tracker's numpy bookkeeping) is under 30 ms and not on the critical path.
- *Results.* Goldens pass in both dtypes (image for sam3, image and video for sam2.1); `test_mlx` has match tests
  against the PyTorch prompt path, the tracker and the motion model. Against the old CPU tracker the video mask
  logits move by at most 0.13 (float16) and 0.0023 (float32); at most 2.5e-6 of a frame's mask pixels flip.
- *PyTorch is still imported* in an MLX run: the task API takes and returns torch tensors, `model.py` / `video.py` /
  `net/*` import it at module level, the replay skeleton is a (weightless) torch module, and the image `prepare` /
  `upsample` are torch. No torch op runs per prompt or per frame in the heads or the tracker.

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
| coatnet_0_rw_224.sw_in1k | fp16 | 8.3 / 7.8 | 1.07 | 0.04 / 0.85 | 100 / 162 | 0.62 | 58 / 58 | 8.3 / 8.5 | 0.82 / 0.73 | 0.04 / 0.85 | 0.0012 | 0.0012 | slower |
| coatnet_0_rw_224.sw_in1k | fp32 | 8.6 / 8.8 | 0.98 | 0.05 / 0.81 | 177 / 272 | 0.65 | 116 / 117 | 9.1 / 9.4 | 0.81 / 0.72 | 0.05 / 0.81 | 7.9e-07 | 7.9e-07 |  |
| coatnet_1_rw_224.sw_in1k | fp16 | 12.7 / 12.9 | 0.99 | 0.04 / 1.30 | 138 / 248 | 0.56 | 92 / 92 | 13.2 / 13.4 | 0.88 / 0.71 | 0.04 / 1.30 | 0.00052 | 0.00052 |  |
| coatnet_1_rw_224.sw_in1k | fp32 | 14.2 / 15.3 | 0.93 | 0.05 / 1.23 | 251 / 340 | 0.74 | 183 / 184 | 14.6 / 16.0 | 0.80 / 0.71 | 0.05 / 1.23 | 3.8e-07 | 3.8e-07 |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.6 / 19.8 | 0.89 | 0.04 / 1.33 | 230 / 343 | 0.67 | 157 / 157 | 18.2 / 20.5 | 0.91 / 0.73 | 0.04 / 1.33 | 0.0027 | 0.0027 |  |
| coatnet_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.7 / 24.2 | 0.81 | 0.06 / 1.24 | 424 / 492 | 0.86 | 314 / 314 | 20.3 / 24.8 | 0.85 / 0.72 | 0.06 / 1.25 | 2.9e-06 | 2.9e-06 |  |
| coatnet_nano_rw_224.sw_in1k | fp16 | 6.0 / 6.4 | 0.93 | 0.03 / 0.87 | 71 / 128 | 0.56 | 32 / 32 | 6.5 / 7.1 | 0.81 / 0.70 | 0.04 / 0.88 | 0.0041 | 0.0041 |  |
| coatnet_nano_rw_224.sw_in1k | fp32 | 6.4 / 7.0 | 0.91 | 0.04 / 0.83 | 121 / 205 | 0.59 | 65 / 65 | 7.0 / 8.3 | 0.75 / 0.70 | 0.04 / 0.83 | 5.6e-07 | 5.6e-07 |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp16 | 12.7 / 12.7 | 1.00 | 0.05 / 1.51 | 138 / 245 | 0.56 | 92 / 92 | 13.2 / 13.4 | 0.88 / 0.73 | 0.05 / 1.51 | 0.0051 | 0.0051 |  |
| coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k | fp32 | 13.6 / 15.1 | 0.90 | 0.06 / 1.49 | 251 / 340 | 0.74 | 183 / 184 | 14.4 / 15.8 | 0.86 / 0.76 | 0.07 / 1.49 | 3.6e-07 | 3.6e-07 |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp16 | 12.1 / 12.2 | 1.00 | 0.07 / 1.56 | 136 / 238 | 0.57 | 92 / 92 | 12.9 / 13.0 | 0.90 / 0.73 | 0.07 / 1.56 | 0.001 | 0.001 |  |
| coatnet_rmlp_1_rw_224.sw_in1k | fp32 | 14.1 / 15.0 | 0.94 | 0.05 / 1.46 | 250 / 350 | 0.71 | 183 / 184 | 14.6 / 15.5 | 0.83 / 0.73 | 0.05 / 1.46 | 3.9e-07 | 3.9e-07 |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp16 | 17.4 / 19.9 | 0.88 | 0.04 / 1.65 | 230 / 343 | 0.67 | 157 / 157 | 17.9 / 20.6 | 0.91 / 0.72 | 0.04 / 1.65 | 0.0041 | 0.0041 |  |
| coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k | fp32 | 19.4 / 23.7 | 0.82 | 0.06 / 1.62 | 424 / 491 | 0.86 | 314 / 314 | 20.1 / 24.4 | 0.87 / 0.73 | 0.06 / 1.63 | 6.7e-07 | 6.7e-07 |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp16 | 17.0 / 19.7 | 0.86 | 0.07 / 1.80 | 230 / 343 | 0.67 | 157 / 157 | 17.6 / 20.2 | 0.97 / 0.82 | 0.07 / 1.81 | 0.0015 | 0.0015 |  |
| coatnet_rmlp_2_rw_224.sw_in1k | fp32 | 19.2 / 24.1 | 0.80 | 0.12 / 1.63 | 424 / 491 | 0.86 | 314 / 314 | 19.9 / 24.9 | 0.90 / 0.80 | 0.12 / 1.63 | 4e-07 | 4e-07 |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp16 | 40.7 / 48.5 | 0.84 | 0.08 / 1.67 | 459 / 475 | 0.97 | 279 / 280 | 41.1 / 49.2 | 1.00 / 0.74 | 0.08 / 1.67 | 0.0017 | 0.0017 |  |
| coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k | fp32 | 47.0 / 60.3 | 0.78 | 0.08 / 1.67 | 864 / 823 | 1.05 | 558 / 559 | 47.8 / 60.7 | 0.97 / 0.75 | 0.08 / 1.67 | 4.8e-07 | 4.8e-07 | heavier |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp16 | 5.9 / 6.6 | 0.90 | 0.03 / 1.02 | 73 / 137 | 0.53 | 32 / 32 | 6.5 / 7.2 | 0.81 / 0.73 | 0.03 / 1.02 | 0.0014 | 0.0014 |  |
| coatnet_rmlp_nano_rw_224.sw_in1k | fp32 | 6.4 / 7.6 | 0.85 | 0.04 / 0.95 | 129 / 227 | 0.57 | 65 / 65 | 7.0 / 7.8 | 0.75 / 0.71 | 0.04 / 0.95 | 2.7e-07 | 2.7e-07 |  |
| coex-sceneflow | fp16 | 197.2 / 409.9 | 0.48 | 0.24 / 2.21 | 550 / 1669 | 0.33 | 11 / 6 | 202.2 / 423.0 | 0.08 / 0.07 | 0.25 / 2.22 | 0.31 (0.1 % off) | 0.034 (0.0055 % off) |  |
| coex-sceneflow | fp32 | 258.0 / 518.0 | 0.50 | 0.29 / 2.27 | 1020 / 1943 | 0.53 | 11 / 11 | 264.1 / 515.7 | 0.08 / 0.06 | 0.30 / 2.28 | 2.8e-06 | 3e-06 |  |
| convnext_base.fb_in1k | fp16 | 15.2 / 15.0 | 1.01 | 0.05 / 1.05 | 192 / 278 | 0.69 | 170 / 169 | 16.1 / 15.6 | 0.91 / 0.71 | 0.05 / 1.05 | 0.00068 | 0.00068 |  |
| convnext_base.fb_in1k | fp32 | 17.4 / 17.4 | 1.00 | 0.06 / 1.00 | 378 / 448 | 0.84 | 338 / 338 | 18.0 / 18.0 | 0.84 / 0.70 | 0.06 / 1.01 | 2.9e-07 | 2.9e-07 |  |
| convnext_base.fb_in22k_ft_in1k | fp16 | 15.5 / 15.0 | 1.03 | 0.04 / 1.04 | 192 / 278 | 0.69 | 170 / 169 | 16.1 / 15.7 | 0.91 / 0.70 | 0.04 / 1.04 | 0.0012 | 0.0012 | slower |
| convnext_base.fb_in22k_ft_in1k | fp32 | 17.4 / 17.4 | 1.00 | 0.06 / 0.99 | 377 / 448 | 0.84 | 338 / 338 | 18.0 / 18.0 | 0.83 / 0.71 | 0.06 / 1.00 | 6.6e-07 | 6.6e-07 |  |
| convnext_base.fb_in22k_ft_in1k_384 | fp16 | 32.6 / 31.1 | 1.05 | 0.05 / 1.05 | 229 / 310 | 0.74 | 170 / 169 | 33.4 / 31.9 | 0.90 / 0.70 | 0.05 / 1.05 | 0.0028 | 0.0028 | slower |
| convnext_base.fb_in22k_ft_in1k_384 | fp32 | 37.0 / 37.3 | 0.99 | 0.07 / 1.03 | 442 / 464 | 0.95 | 338 / 338 | 37.9 / 38.1 | 0.83 / 0.70 | 0.08 / 1.03 | 4.2e-07 | 4.2e-07 |  |
| convnext_large.fb_in1k | fp16 | 27.5 / 27.9 | 0.98 | 0.05 / 1.09 | 412 / 478 | 0.86 | 378 / 378 | 28.3 / 28.6 | 1.16 / 0.71 | 0.06 / 1.09 | 0.0019 | 0.0019 |  |
| convnext_large.fb_in1k | fp32 | 31.9 / 33.3 | 0.96 | 0.07 / 1.05 | 815 / 866 | 0.94 | 755 / 755 | 32.5 / 33.9 | 1.05 / 0.71 | 0.08 / 1.05 | 2.3e-07 | 2.3e-07 |  |
| convnext_large.fb_in22k_ft_in1k | fp16 | 27.9 / 28.0 | 1.00 | 0.05 / 1.12 | 412 / 478 | 0.86 | 378 / 378 | 28.3 / 28.5 | 1.10 / 0.70 | 0.06 / 1.12 | 0.0017 | 0.0017 |  |
| convnext_large.fb_in22k_ft_in1k | fp32 | 32.0 / 33.2 | 0.96 | 0.07 / 1.05 | 814 / 866 | 0.94 | 755 / 755 | 32.6 / 33.8 | 1.05 / 0.71 | 0.07 / 1.06 | 4.8e-07 | 4.8e-07 |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp16 | 61.3 / 60.6 | 1.01 | 0.08 / 1.09 | 463 / 495 | 0.94 | 378 / 378 | 61.9 / 61.5 | 1.10 / 0.72 | 0.08 / 1.10 | 0.0016 | 0.0016 |  |
| convnext_large.fb_in22k_ft_in1k_384 | fp32 | 69.2 / 71.0 | 0.97 | 0.10 / 1.08 | 920 / 892 | 1.03 | 755 / 755 | 70.0 / 72.0 | 1.03 / 0.71 | 0.10 / 1.09 | 4.3e-07 | 4.3e-07 | heavier |
| convnext_small.fb_in1k | fp16 | 11.3 / 10.7 | 1.06 | 0.04 / 1.01 | 116 / 185 | 0.63 | 96 / 96 | 11.8 / 11.3 | 0.86 / 0.69 | 0.04 / 1.01 | 0.0013 | 0.0013 | slower |
| convnext_small.fb_in1k | fp32 | 12.9 / 12.5 | 1.03 | 0.06 / 0.97 | 223 / 302 | 0.74 | 192 / 192 | 13.3 / 13.1 | 0.81 / 0.72 | 0.06 / 0.97 | 3.9e-07 | 3.9e-07 | slower |
| convnext_small.fb_in22k_ft_in1k | fp16 | 11.2 / 10.7 | 1.05 | 0.04 / 1.02 | 118 / 185 | 0.64 | 96 / 96 | 11.8 / 11.3 | 0.88 / 0.71 | 0.04 / 1.02 | 0.0014 | 0.0014 | slower |
| convnext_small.fb_in22k_ft_in1k | fp32 | 12.9 / 12.4 | 1.04 | 0.06 / 0.97 | 222 / 302 | 0.73 | 192 / 192 | 13.4 / 13.0 | 0.80 / 0.70 | 0.06 / 0.97 | 5e-07 | 5e-07 | slower |
| convnext_small.fb_in22k_ft_in1k_384 | fp16 | 24.0 / 21.2 | 1.13 | 0.05 / 1.05 | 139 / 233 | 0.60 | 96 / 96 | 24.6 / 21.9 | 0.85 / 0.70 | 0.05 / 1.05 | 0.0017 | 0.0017 | slower |
| convnext_small.fb_in22k_ft_in1k_384 | fp32 | 26.2 / 24.9 | 1.05 | 0.06 / 1.02 | 280 / 308 | 0.91 | 192 / 192 | 27.3 / 26.1 | 0.79 / 0.71 | 0.07 / 1.02 | 2.7e-07 | 2.7e-07 | slower |
| convnext_tiny.fb_in1k | fp16 | 9.2 / 6.0 | 1.53 | 0.04 / 0.67 | 70 / 108 | 0.65 | 55 / 55 | 7.2 / 6.8 | 0.81 / 0.70 | 0.04 / 0.68 | 0.0016 | 0.0016 | slower |
| convnext_tiny.fb_in1k | fp32 | 7.5 / 6.8 | 1.11 | 0.05 / 0.65 | 135 / 206 | 0.66 | 109 / 109 | 8.0 / 7.4 | 0.75 / 0.76 | 0.06 / 0.65 | 5.3e-07 | 5.3e-07 | slower |
| convnext_tiny.fb_in22k_ft_in1k | fp16 | 6.7 / 5.8 | 1.16 | 0.03 / 0.68 | 69 / 108 | 0.64 | 55 / 55 | 7.2 / 6.5 | 0.81 / 0.71 | 0.04 / 0.68 | 0.0012 | 0.0012 | slower |
| convnext_tiny.fb_in22k_ft_in1k | fp32 | 7.5 / 6.8 | 1.11 | 0.05 / 0.67 | 136 / 206 | 0.66 | 109 / 109 | 8.1 / 7.3 | 0.75 / 0.69 | 0.06 / 0.67 | 4.3e-07 | 4.3e-07 | slower |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp16 | 14.3 / 12.0 | 1.20 | 0.04 / 0.68 | 91 / 161 | 0.56 | 55 / 55 | 15.0 / 12.9 | 0.79 / 0.69 | 0.04 / 0.68 | 0.002 | 0.002 | slower |
| convnext_tiny.fb_in22k_ft_in1k_384 | fp32 | 15.9 / 14.2 | 1.12 | 0.06 / 0.66 | 195 / 224 | 0.87 | 109 / 109 | 16.6 / 14.9 | 0.74 / 0.68 | 0.06 / 0.66 | 9.8e-07 | 9.8e-07 | slower |
| convnextv2_base.fcmae_ft_in22k_in1k | fp16 | 18.4 / 17.6 | 1.04 | 0.05 / 1.35 | 244 / 318 | 0.77 | 170 / 170 | 18.9 / 18.2 | 0.90 / 0.72 | 0.05 / 1.35 | 0.0011 | 0.0011 | slower |
| convnextv2_base.fcmae_ft_in22k_in1k | fp32 | 19.9 / 20.1 | 0.99 | 0.07 / 1.31 | 418 / 456 | 0.92 | 339 / 339 | 20.7 / 20.9 | 0.82 / 0.75 | 0.07 / 1.31 | 6.4e-07 | 6.4e-07 |  |
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
| dat-2-x4 | fp16 | 4832.4 / 9060.1 | 0.53 | 5.14 / 15.26 | 1974 / 2848 | 0.69 | 57 / 35 | 4790.4 / 8974.4 | 0.89 / 0.37 | 5.16 / 15.27 | 0.0089 | 0.0099 |  |
| dat-2-x4 | fp32 | 7227.0 / 15112.8 | 0.48 | 7.78 / 15.19 | 3943 / 4156 | 0.95 | 112 / 70 | 7499.0 / 15665.8 | 0.71 / 0.37 | 7.80 / 15.20 | 5.6e-06 | 6.3e-06 |  |
| dat-light-x4 | fp16 | 759.3 / 2449.4 | 0.31 | 1.11 / 6.19 | 1194 / 1921 | 0.62 | 28 / 8 | 763.3 / 2449.0 | 0.56 / 0.15 | 1.13 / 6.19 | 0.0031 | 0.0035 |  |
| dat-light-x4 | fp32 | 1241.8 / 1651.2 | 0.75 | 1.67 / 5.29 | 2385 / 1899 | 1.26 | 55 / 16 | 1251.0 / 1666.0 | 0.55 / 0.15 | 1.68 / 5.29 | 8.1e-06 | 9.2e-06 | heavier |
| dat-x4 | fp16 | 5330.8 / 11403.9 | 0.47 | 5.60 / 17.86 | 2137 / 2869 | 0.74 | 64 / 42 | 5323.7 / 11352.6 | 0.70 / 0.42 | 5.61 / 17.87 | 0.0063 | 0.0071 |  |
| dat-x4 | fp32 | 8481.3 / 17063.6 | 0.50 | 9.08 / 17.50 | 3999 / 4170 | 0.96 | 126 / 84 | 8699.1 / 17565.1 | 0.70 / 0.48 | 9.10 / 17.51 | 5.5e-06 | 6.2e-06 |  |
| davit_base.msft_in1k | fp16 | 19.3 / 18.2 | 1.06 | 0.05 / 1.90 | 224 / 248 | 0.90 | 168 / 168 | 19.8 / 18.8 | 0.90 / 0.71 | 0.05 / 1.90 | 0.00054 | 0.00054 | slower |
| davit_base.msft_in1k | fp32 | 21.5 / 21.5 | 1.00 | 0.07 / 1.80 | 434 / 417 | 1.04 | 336 / 336 | 22.1 / 22.1 | 0.82 / 0.74 | 0.07 / 1.80 | 3.3e-07 | 3.3e-07 | heavier |
| davit_small.msft_in1k | fp16 | 15.3 / 13.9 | 1.10 | 0.04 / 1.85 | 135 / 178 | 0.76 | 95 / 95 | 15.9 / 14.4 | 0.84 / 0.71 | 0.04 / 1.85 | 0.00076 | 0.00076 | slower |
| davit_small.msft_in1k | fp32 | 16.6 / 16.0 | 1.04 | 0.06 / 1.81 | 262 / 272 | 0.96 | 190 / 190 | 17.3 / 16.7 | 0.79 / 0.73 | 0.07 / 1.81 | 4.6e-07 | 4.6e-07 | slower |
| davit_tiny.msft_in1k | fp16 | 9.2 / 7.6 | 1.22 | 0.03 / 1.14 | 95 / 123 | 0.77 | 54 / 54 | 9.7 / 8.2 | 0.78 / 0.69 | 0.03 / 1.14 | 0.0013 | 0.0013 | slower |
| davit_tiny.msft_in1k | fp32 | 9.9 / 9.1 | 1.08 | 0.06 / 1.11 | 180 / 179 | 1.01 | 108 / 108 | 10.4 / 9.6 | 0.74 / 0.78 | 0.06 / 1.12 | 3.9e-07 | 3.9e-07 | slower |
| deeplabv3-r101-d8-ade20k | fp16 | 366.9 / 377.9 | 0.97 | 0.41 / 1.69 | 682 / 574 | 1.19 | 162 / 162 | 402.1 / 410.0 | 0.22 / 0.14 | 0.45 / 1.73 | 0.0007 | 0 | heavier |
| deeplabv3-r101-d8-ade20k | fp32 | 540.1 / 526.5 | 1.03 | 0.57 / 1.80 | 1224 / 1312 | 0.93 | 323 / 323 | 577.0 / 563.2 | 0.22 / 0.14 | 0.61 / 1.85 | 3.4e-07 | 0 | slower |
| deeplabv3plus-r101-d8-ade20k | fp16 | 268.6 / 290.9 | 0.92 | 0.32 / 1.62 | 773 / 674 | 1.15 | 115 / 115 | 307.3 / 327.2 | 0.19 / 0.13 | 0.37 / 1.67 | 0.0011 | 0 | heavier |
| deeplabv3plus-r101-d8-ade20k | fp32 | 347.5 / 377.6 | 0.92 | 0.40 / 1.67 | 1404 / 1209 | 1.16 | 230 / 230 | 387.4 / 422.0 | 0.19 / 0.13 | 0.45 / 1.72 | 4.2e-07 | 0 | heavier |
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
| deit_base_patch16_224.fb_in1k | fp16 | 14.1 / 13.7 | 1.03 | 0.04 / 0.64 | 242 / 232 | 1.04 | 165 / 165 | 14.6 / 14.3 | 0.05 / 0.70 | 0.05 / 0.65 | 0.00093 | 0.00093 | slower heavier |
| deit_base_patch16_224.fb_in1k | fp32 | 16.2 / 16.1 | 1.01 | 0.05 / 0.63 | 454 / 420 | 1.08 | 330 / 330 | 16.7 / 16.6 | 0.06 / 0.69 | 0.05 / 0.63 | 6.8e-07 | 6.8e-07 | heavier |
| densenet121.tv_in1k | fp16 | 8.2 / 10.7 | 0.77 | 0.03 / 1.38 | 81 / 74 | 1.11 | 16 / 16 | 8.6 / 11.3 | 0.68 / 0.58 | 0.03 / 1.39 | 0.0017 | 0.0017 | heavier |
| densenet121.tv_in1k | fp32 | 9.6 / 12.4 | 0.77 | 0.03 / 1.31 | 148 / 142 | 1.04 | 30 / 31 | 10.1 / 12.9 | 0.64 / 0.58 | 0.03 / 1.31 | 5.4e-07 | 5.4e-07 | heavier |
| depth-pro | fp16 | 4168.0 / 4341.7 | 0.96 | 4.27 / 9.37 | 4508 / 4609 | 0.98 | 1809 / 1815 | 4170.1 / 4666.0 | 3.88 / 3.42 | 4.28 / 9.38 | 0.0017 | 0.0011 |  |
| depth-pro | fp32 | 5497.0 / 6831.3 | 0.80 | 5.96 / 10.71 | 9014 / 9191 | 0.98 | 3616 / 3630 | 5552.0 / 6573.6 | 4.63 / 3.41 | 5.99 / 10.74 | 1.5e-06 | 1.3e-06 |  |
| detr-r101-coco | fp16 | 107.0 / 117.1 | 0.91 | 0.14 / 2.29 | 310 / 365 | 0.85 | 116 / 117 | 119.0 / 127.5 | 0.07 / 0.15 | 0.16 / 2.31 | 0.0044 | empty |  |
| detr-r101-coco | fp32 | 133.4 / 144.9 | 0.92 | 0.17 / 2.08 | 596 / 547 | 1.09 | 232 / 232 | 146.2 / 157.0 | 0.07 / 0.14 | 0.19 / 2.10 | 2.4e-06 | empty | heavier |
| detr-r50-coco | fp16 | 69.1 / 78.2 | 0.88 | 0.10 / 1.86 | 274 / 343 | 0.80 | 80 / 81 | 80.1 / 89.7 | 0.06 / 0.10 | 0.12 / 1.88 | 0.0078 | 0.00069 |  |
| detr-r50-coco | fp32 | 85.2 / 95.1 | 0.90 | 0.12 / 2.04 | 524 / 450 | 1.16 | 160 / 160 | 97.5 / 107.2 | 0.06 / 0.10 | 0.14 / 2.06 | 6.2e-06 | 3.7e-07 | heavier |
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
| drct-l-x4 | fp16 | 8173.1 / 11297.8 | 0.72 | 8.45 / 20.83 | 1083 / 2518 | 0.43 | 106 / 345 | 8295.1 / 11414.6 | 0.53 / 0.95 | 8.47 / 20.84 | 0.01 (1.6e-05 % off) | 0.011 (3.2e-05 % off) |  |
| drct-l-x4 | fp32 | 13143.6 / 23086.9 | 0.57 | 12.48 / 29.68 | 2130 / 5678 | 0.38 | 208 / 432 | 12237.2 / 23323.1 | 0.95 / 0.97 | 12.50 / 29.70 | 1e-05 | 1.1e-05 |  |
| drct-x4 | fp16 | 4250.7 / 6130.6 | 0.69 | 4.54 / 11.53 | 1040 / 2474 | 0.42 | 64 / 301 | 4168.5 / 6226.9 | 0.54 / 0.32 | 4.56 / 11.54 | 0.025 (0.003 % off) | 0.029 (0.0047 % off) |  |
| drct-x4 | fp32 | 6124.1 / 11220.6 | 0.55 | 6.79 / 15.07 | 2046 / 5592 | 0.37 | 123 / 345 | 6147.2 / 11288.8 | 0.75 / 0.37 | 6.82 / 15.09 | 8.1e-06 | 9.3e-06 |  |
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
| efficientformer_l1.snap_dist_in1k | fp16 | 3.2 / 3.8 | 0.85 | 0.04 / 0.71 | 32 / 59 | 0.54 | 23 / 24 | 4.3 / 4.3 | 0.76 / 0.69 | 0.04 / 0.72 | 0.0056 | 0.0056 |  |
| efficientformer_l1.snap_dist_in1k | fp32 | 3.5 / 4.1 | 0.85 | 0.05 / 0.69 | 66 / 96 | 0.69 | 45 / 47 | 3.9 / 4.8 | 0.73 / 0.69 | 0.05 / 0.70 | 1.3e-06 | 1.3e-06 |  |
| efficientformer_l3.snap_dist_in1k | fp16 | 5.9 / 7.4 | 0.79 | 0.04 / 1.18 | 73 / 131 | 0.55 | 59 / 60 | 6.4 / 7.9 | 0.82 / 0.70 | 0.04 / 1.18 | 0.0021 | 0.0021 |  |
| efficientformer_l3.snap_dist_in1k | fp32 | 6.8 / 8.4 | 0.81 | 0.06 / 1.21 | 146 / 218 | 0.67 | 118 / 120 | 7.3 / 8.8 | 0.76 / 0.71 | 0.06 / 1.21 | 7.9e-07 | 7.9e-07 |  |
| efficientformer_l7.snap_dist_in1k | fp16 | 12.3 / 14.2 | 0.86 | 0.06 / 1.58 | 184 / 290 | 0.63 | 156 / 157 | 12.8 / 14.7 | 0.93 / 0.72 | 0.06 / 1.58 | 0.006 | 0.006 |  |
| efficientformer_l7.snap_dist_in1k | fp32 | 13.8 / 16.3 | 0.84 | 0.07 / 1.51 | 361 / 442 | 0.82 | 311 / 314 | 14.1 / 17.2 | 0.84 / 0.71 | 0.07 / 1.52 | 6.9e-07 | 6.9e-07 |  |
| efficientformerv2_l.snap_dist_in1k | fp16 | 8.6 / 9.2 | 0.93 | 0.05 / 2.33 | 69 / 100 | 0.69 | 50 / 51 | 9.0 / 9.8 | 0.92 / 0.75 | 0.05 / 2.33 | 0.0027 | 0.0027 |  |
| efficientformerv2_l.snap_dist_in1k | fp32 | 9.0 / 10.2 | 0.88 | 0.06 / 2.27 | 130 / 189 | 0.69 | 100 / 102 | 9.7 / 10.8 | 0.83 / 0.77 | 0.07 / 2.27 | 9.3e-07 | 9.3e-07 |  |
| efficientformerv2_s0.snap_dist_in1k | fp16 | 2.9 / 3.0 | 0.96 | 0.04 / 1.25 | 16 / 25 | 0.63 | 7 / 8 | 3.4 / 3.5 | 0.78 / 0.73 | 0.04 / 1.25 | 0.0043 | 0.0043 |  |
| efficientformerv2_s0.snap_dist_in1k | fp32 | 3.2 / 4.4 | 0.74 | 0.05 / 1.20 | 27 / 45 | 0.61 | 14 / 14 | 3.7 / 4.3 | 0.73 / 0.71 | 0.05 / 1.20 | 1.5e-06 | 1.5e-06 |  |
| efficientformerv2_s1.snap_dist_in1k | fp16 | 3.5 / 5.1 | 0.69 | 0.04 / 1.41 | 20 / 37 | 0.56 | 12 / 13 | 4.0 / 5.8 | 0.82 / 0.71 | 0.04 / 1.41 | 0.0056 | 0.0056 |  |
| efficientformerv2_s1.snap_dist_in1k | fp32 | 5.2 / 4.5 | 1.17 | 0.05 / 1.35 | 37 / 66 | 0.56 | 24 / 24 | 5.5 / 4.9 | 0.73 / 0.71 | 0.06 / 1.36 | 1.8e-06 | 1.8e-06 | slower |
| efficientformerv2_s2.snap_dist_in1k | fp16 | 5.6 / 6.1 | 0.93 | 0.04 / 1.78 | 40 / 56 | 0.72 | 25 / 25 | 6.1 / 6.7 | 0.84 / 0.71 | 0.04 / 1.78 | 0.0034 | 0.0034 |  |
| efficientformerv2_s2.snap_dist_in1k | fp32 | 6.0 / 8.4 | 0.71 | 0.06 / 1.68 | 73 / 112 | 0.65 | 48 / 50 | 6.5 / 8.7 | 0.77 / 0.71 | 0.06 / 1.68 | 1.3e-06 | 1.3e-06 |  |
| efficientnet_b0.tv_in1k | fp16 | 3.2 / 4.2 | 0.75 | 0.02 / 0.85 | 29 / 63 | 0.46 | 11 / 10 | 3.7 / 4.8 | 0.58 / 0.50 | 0.02 / 0.91 | 0.0013 | 0.0013 |  |
| efficientnet_b0.tv_in1k | fp32 | 3.3 / 4.5 | 0.75 | 0.04 / 0.83 | 45 / 94 | 0.48 | 21 / 21 | 3.8 / 4.9 | 0.53 / 0.53 | 0.04 / 0.89 | 9.5e-07 | 9.5e-07 |  |
| efficientnet_b4.tv_in1k | fp16 | 12.7 / 15.4 | 0.83 | 0.05 / 1.45 | 126 / 229 | 0.55 | 38 / 38 | 13.4 / 16.2 | 0.72 / 0.59 | 0.05 / 1.46 | 0.00084 | 0.00084 |  |
| efficientnet_b4.tv_in1k | fp32 | 13.9 / 18.0 | 0.77 | 0.05 / 1.41 | 196 / 256 | 0.77 | 75 / 75 | 14.5 / 18.9 | 0.63 / 0.59 | 0.05 / 1.41 | 5.1e-07 | 5.1e-07 |  |
| efficientnet_v2_s.tv_in1k | fp16 | 13.9 / 16.0 | 0.87 | 0.03 / 1.51 | 103 / 234 | 0.44 | 41 / 41 | 14.5 / 16.4 | 0.75 / 0.59 | 0.04 / 1.51 | 0.00064 | 0.00064 |  |
| efficientnet_v2_s.tv_in1k | fp32 | 15.0 / 17.8 | 0.84 | 0.05 / 1.46 | 157 / 260 | 0.60 | 82 / 82 | 15.6 / 18.3 | 0.64 / 0.60 | 0.05 / 1.46 | 3.9e-07 | 3.9e-07 |  |
| efficientvit-sam-l0 | fp32 | 28.7 / 52.3 | 0.55 | 0.06 / 1.23 | 458 / 472 | 0.97 | 120 / 117 | 57.8 / 81.1 | 0.15 / 1.17 | 0.10 / 1.27 | 4.4e-05 | 1 (0.00036 % off) |  |
| efficientvit-sam-l1 | fp32 | 38.7 / 62.4 | 0.62 | 0.07 / 1.38 | 615 / 624 | 0.99 | 169 / 166 | 67.8 / 91.6 | 0.18 / 1.40 | 0.10 / 1.42 | 0.00012 | 1 (0.0028 % off) |  |
| efficientvit-sam-l2 | fp32 | 51.8 / 76.7 | 0.68 | 0.08 / 1.50 | 732 / 778 | 0.94 | 222 / 219 | 80.7 / 105.8 | 0.23 / 1.66 | 0.12 / 1.54 | 0.0003 | 1 (0.005 % off) |  |
| efficientvit-sam-xl0 | fp32 | 193.3 / 203.5 | 0.95 | 0.22 / 1.50 | 1311 / 1026 | 1.28 | 443 / 431 | 214.6 / 222.6 | 0.46 / 1.41 | 0.26 / 1.53 | 3.5e-05 | 1 (0.0011 % off) | heavier |
| efficientvit-sam-xl1 | fp32 | 348.8 / 342.0 | 1.02 | 0.37 / 1.93 | 2197 / 1668 | 1.32 | 772 / 760 | 368.9 / 363.1 | 0.74 / 1.56 | 0.42 / 1.96 | 3.7e-05 | 1 (0.003 % off) | heavier |
| efficientvit_b2.r288_in1k | fp16 | 6.5 / 8.2 | 0.79 | 0.04 / 1.35 | 90 / 137 | 0.66 | 47 / 47 | 7.1 / 8.9 | 0.82 / 0.69 | 0.04 / 1.36 | 0.0015 | 0.0015 |  |
| efficientvit_b2.r288_in1k | fp32 | 7.7 / 10.6 | 0.72 | 0.04 / 1.33 | 144 / 194 | 0.74 | 93 / 93 | 8.1 / 11.4 | 0.79 / 0.70 | 0.04 / 1.34 | 2.1e-06 | 2.1e-06 |  |
| eomt-l-dinov2-ade20k | fp16 | 393.6 / 386.1 | 1.02 | 0.42 / 2.02 | 909 / 878 | 1.03 | 603 / 603 | 443.6 / 525.1 | 0.07 / 2.14 | 0.55 / 2.30 | 0.0028 | 1 (0.06 % off) | heavier |
| eomt-l-dinov2-ade20k | fp32 | 478.6 / 467.6 | 1.02 | 0.50 / 2.08 | 1802 / 1785 | 1.01 | 1205 / 1206 | 533.4 / 608.3 | 0.10 / 2.31 | 0.69 / 2.83 | 4.1e-06 | 0 | slower |
| eomt-l-dinov3-ade20k | fp16 | 404.9 / 409.1 | 0.99 | 0.42 / 2.67 | 908 / 894 | 1.02 | 601 / 601 | 456.9 / 547.5 | 0.07 / 2.11 | 0.57 / 2.89 | 0.0039 | 1 (0.00033 % off) |  |
| eomt-l-dinov3-ade20k | fp32 | 503.5 / 510.8 | 0.99 | 0.53 / 2.75 | 1799 / 1749 | 1.03 | 1202 / 1202 | 560.5 / 651.1 | 0.82 / 2.17 | 0.76 / 3.10 | 3e-06 | 0 | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp16 | 195.0 / 197.4 | 0.99 | 0.23 / 2.07 | 717 / 666 | 1.08 | 583 / 583 | 195.9 / 198.2 | 0.07 / 0.74 | 0.23 / 2.07 | 0.0012 | 0.0012 | heavier |
| eva02_large_patch14_448.mim_m38m_ft_in22k_in1k | fp32 | 240.7 / 242.1 | 0.99 | 0.27 / 2.07 | 1430 / 1396 | 1.02 | 1165 / 1165 | 241.4 / 243.0 | 0.09 / 0.73 | 0.27 / 2.07 | 6e-07 | 6e-07 | heavier |
| falconsai-nsfw-vit | fp16 | 14.0 / 13.6 | 1.03 | 0.04 / 0.66 | 240 / 231 | 1.04 | 164 / 164 | 14.3 / 13.9 | 0.05 / 0.72 | 0.04 / 0.67 | 0 | 0 | slower heavier |
| falconsai-nsfw-vit | fp32 | 16.0 / 16.0 | 1.00 | 0.05 / 0.65 | 452 / 417 | 1.08 | 327 / 327 | 16.3 / 16.4 | 0.06 / 0.73 | 0.05 / 0.66 | 1.1e-07 | 1.1e-07 | heavier |
| faster-rcnn-r50-fpn-3x-coco | fp16 | 95.5 / 104.1 | 0.92 | 0.11 / 0.68 | 745 / 432 | 1.72 | 80 / 52 | 116.2 / 339.5 | 0.54 / 0.57 | 0.16 / 0.93 | - | 0.00015 | heavier |
| faster-rcnn-r50-fpn-3x-coco | fp32 | 116.1 / 125.2 | 0.93 | 0.15 / 0.70 | 1488 / 857 | 1.74 | 159 / 105 | 136.8 / 359.3 | 0.56 / 0.56 | 0.19 / 0.98 | - | 1.6e-06 | heavier |
| fastvit_sa24.apple_dist_in1k | fp16 | 6.1 / 6.8 | 0.90 | 0.04 / 0.87 | 60 / 104 | 0.57 | 41 / 41 | 6.6 / 7.3 | 1.02 / 0.90 | 0.04 / 0.87 | 0.003 | 0.003 |  |
| fastvit_sa24.apple_dist_in1k | fp32 | 6.8 / 7.7 | 0.88 | 0.06 / 0.81 | 122 / 197 | 0.62 | 82 / 82 | 7.4 / 8.3 | 0.95 / 0.90 | 0.06 / 0.81 | 3.7e-07 | 3.7e-07 |  |
| fastvit_t12.apple_dist_in1k | fp16 | 3.2 / 3.3 | 0.97 | 0.03 / 0.53 | 26 / 47 | 0.56 | 14 / 14 | 3.6 / 3.8 | 0.91 / 0.84 | 0.03 / 0.53 | 0.0051 | 0.0051 |  |
| fastvit_t12.apple_dist_in1k | fp32 | 3.6 / 3.6 | 0.99 | 0.05 / 0.49 | 54 / 87 | 0.62 | 29 / 29 | 4.1 / 4.2 | 0.86 / 0.86 | 0.05 / 0.50 | 7.1e-07 | 7.1e-07 |  |
| fastvit_t8.apple_dist_in1k | fp16 | 2.5 / 2.8 | 0.90 | 0.03 / 0.49 | 17 / 30 | 0.57 | 8 / 8 | 3.0 / 3.3 | 0.89 / 0.83 | 0.03 / 0.50 | 0.015 (0.7 % off) | 0.015 (0.7 % off) |  |
| fastvit_t8.apple_dist_in1k | fp32 | 2.9 / 3.5 | 0.81 | 0.05 / 0.47 | 33 / 55 | 0.60 | 15 / 15 | 3.4 / 3.6 | 0.85 / 0.83 | 0.05 / 0.47 | 1.3e-06 | 1.3e-06 |  |
| fcn-r101-d8-ade20k | fp16 | 326.3 / 354.0 | 0.92 | 0.40 / 1.82 | 877 / 638 | 1.38 | 126 / 126 | 364.1 / 380.5 | 0.23 / 0.15 | 0.45 / 1.87 | 0.001 | 0 | heavier |
| fcn-r101-d8-ade20k | fp32 | 439.6 / 472.5 | 0.93 | 0.49 / 1.91 | 1263 / 1236 | 1.02 | 252 / 252 | 472.3 / 510.6 | 0.21 / 0.15 | 0.53 / 1.96 | 0 | 0 | heavier |
| focalnet_base_lrf.ms_in1k | fp16 | 19.0 / 21.6 | 0.88 | 0.06 / 1.46 | 217 / 289 | 0.75 | 171 / 170 | 19.6 / 22.7 | 0.92 / 0.72 | 0.06 / 1.47 | 0.0006 | 0.0006 |  |
| focalnet_base_lrf.ms_in1k | fp32 | 21.3 / 25.7 | 0.83 | 0.05 / 1.41 | 419 / 491 | 0.85 | 340 / 340 | 22.0 / 26.5 | 0.95 / 0.70 | 0.06 / 1.41 | 2.9e-07 | 2.9e-07 |  |
| focalnet_base_srf.ms_in1k | fp16 | 17.9 / 20.3 | 0.88 | 0.04 / 1.36 | 214 / 285 | 0.75 | 169 / 169 | 18.4 / 20.9 | 0.93 / 0.70 | 0.04 / 1.36 | 0.0008 | 0.0008 |  |
| focalnet_base_srf.ms_in1k | fp32 | 20.0 / 24.6 | 0.81 | 0.05 / 1.29 | 412 / 492 | 0.84 | 337 / 337 | 20.6 / 25.4 | 0.88 / 0.74 | 0.05 / 1.29 | 3.4e-07 | 3.4e-07 |  |
| focalnet_small_lrf.ms_in1k | fp16 | 14.6 / 14.9 | 0.98 | 0.04 / 1.46 | 131 / 195 | 0.67 | 97 / 97 | 15.1 / 15.6 | 0.89 / 0.71 | 0.04 / 1.47 | 0.00077 | 0.00077 |  |
| focalnet_small_lrf.ms_in1k | fp32 | 16.3 / 17.5 | 0.93 | 0.05 / 1.35 | 251 / 347 | 0.72 | 193 / 193 | 16.8 / 18.1 | 0.81 / 0.71 | 0.05 / 1.36 | 3.3e-07 | 3.3e-07 |  |
| focalnet_small_srf.ms_in1k | fp16 | 13.9 / 13.9 | 1.00 | 0.04 / 1.40 | 128 / 201 | 0.64 | 96 / 96 | 14.4 / 14.7 | 0.89 / 0.73 | 0.04 / 1.40 | 0.00069 | 0.00069 |  |
| focalnet_small_srf.ms_in1k | fp32 | 15.0 / 16.3 | 0.92 | 0.05 / 1.25 | 246 / 341 | 0.72 | 191 / 191 | 15.7 / 16.8 | 0.81 / 0.70 | 0.05 / 1.25 | 4e-07 | 4e-07 |  |
| focalnet_tiny_lrf.ms_in1k | fp16 | 8.5 / 8.5 | 1.00 | 0.03 / 0.85 | 89 / 143 | 0.62 | 55 / 55 | 9.1 / 9.1 | 0.82 / 0.71 | 0.03 / 0.85 | 0.00058 | 0.00058 |  |
| focalnet_tiny_lrf.ms_in1k | fp32 | 9.3 / 9.7 | 0.95 | 0.04 / 0.79 | 168 / 257 | 0.65 | 110 / 110 | 9.8 / 10.2 | 0.77 / 0.71 | 0.04 / 0.79 | 2.5e-07 | 2.5e-07 |  |
| focalnet_tiny_srf.ms_in1k | fp16 | 8.2 / 7.9 | 1.04 | 0.04 / 0.78 | 86 / 138 | 0.63 | 55 / 55 | 8.6 / 8.5 | 0.80 / 0.69 | 0.04 / 0.79 | 0.00088 | 0.00088 | slower |
| focalnet_tiny_srf.ms_in1k | fp32 | 8.6 / 8.8 | 0.98 | 0.04 / 0.79 | 163 / 244 | 0.67 | 109 / 109 | 9.2 / 9.5 | 0.77 / 0.76 | 0.04 / 0.79 | 4.3e-07 | 4.3e-07 |  |
| hat-s-x4 | fp16 | 5957.5 / 9081.5 | 0.66 | 6.32 / 11.74 | 1958 / 2700 | 0.73 | 77 / 313 | 6019.4 / 9283.4 | 0.60 / 0.36 | 6.34 / 11.76 | 0.0089 | 0.01 (7.9e-06 % off) |  |
| hat-s-x4 | fp32 | 7460.5 / 17686.6 | 0.42 | 7.76 / 17.95 | 3907 / 4739 | 0.82 | 148 / 368 | 7705.2 / 17820.1 | 0.73 / 0.43 | 7.78 / 17.96 | 5.3e-06 | 6e-06 |  |
| hat-x4 | fp16 | 7017.5 / 8996.0 | 0.78 | 7.12 / 12.93 | 2276 / 2507 | 0.91 | 96 / 334 | 6905.1 / 9554.0 | 0.67 / 0.29 | 7.13 / 12.93 | 0.013 (0.00027 % off) | 0.014 (0.00058 % off) |  |
| hat-x4 | fp32 | 10861.1 / 23399.1 | 0.46 | 11.30 / 21.32 | 4514 / 4506 | 1.00 | 189 / 411 | 10852.4 / 22411.1 | 0.78 / 0.40 | 11.32 / 21.34 | 6.9e-06 | 7.5e-06 |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp16 | 2.0 / 2.2 | 0.87 | 0.03 / 0.74 | 19 / 28 | 0.70 | 12 / 12 | 2.0 / 2.7 | 0.76 / 0.74 | 0.04 / 0.74 | 0.00081 | 0.00081 |  |
| hgnetv2_b0.ssld_stage2_ft_in1k | fp32 | 2.5 / 3.5 | 0.72 | 0.04 / 0.69 | 36 / 52 | 0.70 | 23 / 23 | 2.4 / 3.2 | 0.72 / 0.69 | 0.05 / 0.69 | 8.1e-07 | 8.1e-07 |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp16 | 2.1 / 2.7 | 0.77 | 0.04 / 0.72 | 22 / 35 | 0.64 | 12 / 12 | 2.5 / 3.1 | 0.74 / 0.69 | 0.04 / 0.72 | 0.00063 | 0.00063 |  |
| hgnetv2_b1.ssld_stage2_ft_in1k | fp32 | 2.4 / 3.0 | 0.80 | 0.04 / 0.71 | 42 / 66 | 0.65 | 24 / 24 | 2.9 / 3.5 | 0.70 / 0.71 | 0.05 / 0.71 | 7.8e-07 | 7.8e-07 |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp16 | 3.0 / 3.8 | 0.80 | 0.04 / 0.95 | 34 / 53 | 0.64 | 22 / 22 | 3.4 / 4.3 | 0.78 / 0.72 | 0.04 / 0.95 | 0.00053 | 0.00053 |  |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp32 | 3.5 / 4.3 | 0.82 | 0.05 / 0.91 | 65 / 100 | 0.65 | 43 / 43 | 4.2 / 4.8 | 0.75 / 0.70 | 0.06 / 0.91 | 9.1e-07 | 9.1e-07 |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp16 | 3.6 / 4.6 | 0.79 | 0.04 / 1.13 | 46 / 69 | 0.66 | 31 / 31 | 4.1 / 5.1 | 0.80 / 0.70 | 0.04 / 1.13 | 0.00052 | 0.00052 |  |
| hgnetv2_b3.ssld_stage2_ft_in1k | fp32 | 4.2 / 5.1 | 0.83 | 0.06 / 1.08 | 88 / 131 | 0.67 | 62 / 62 | 4.8 / 5.8 | 0.74 / 0.70 | 0.06 / 1.08 | 5.1e-07 | 5.1e-07 |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp16 | 5.0 / 5.4 | 0.92 | 0.03 / 1.01 | 58 / 72 | 0.81 | 38 / 38 | 5.4 / 5.9 | 0.80 / 0.70 | 0.03 / 1.02 | 0.00064 | 0.00064 |  |
| hgnetv2_b4.ssld_stage2_ft_in1k | fp32 | 5.7 / 6.3 | 0.90 | 0.04 / 1.01 | 111 / 137 | 0.81 | 76 / 76 | 6.2 / 6.7 | 0.74 / 0.70 | 0.04 / 1.01 | 5.4e-07 | 5.4e-07 |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp16 | 9.0 / 9.7 | 0.93 | 0.04 / 1.46 | 104 / 139 | 0.75 | 76 / 76 | 9.5 / 10.4 | 0.86 / 0.71 | 0.04 / 1.46 | 0.00064 | 0.00064 |  |
| hgnetv2_b5.ssld_stage2_ft_in1k | fp32 | 10.4 / 11.3 | 0.92 | 0.04 / 1.43 | 196 / 256 | 0.76 | 151 / 151 | 10.9 / 11.9 | 0.80 / 0.71 | 0.05 / 1.43 | 3.9e-07 | 3.9e-07 |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp16 | 18.9 / 19.9 | 0.95 | 0.04 / 1.80 | 187 / 272 | 0.69 | 144 / 144 | 19.4 / 20.5 | 0.94 / 0.72 | 0.04 / 1.80 | 0.00088 | 0.00088 |  |
| hgnetv2_b6.ssld_stage2_ft_in1k | fp32 | 21.7 / 23.5 | 0.92 | 0.06 / 1.77 | 368 / 493 | 0.75 | 288 / 288 | 22.3 / 24.0 | 0.83 / 0.73 | 0.06 / 1.77 | 5.9e-07 | 5.9e-07 |  |
| hrnet-w32 | fp16 | 85.4 / 98.1 | 0.87 | 0.12 / 2.08 | 232 / 271 | 0.86 | 56 / 56 | 94.7 / 106.9 | 0.05 / 0.12 | 0.14 / 2.09 | 0.0036 | 0.29 (0.36 % off) |  |
| hrnet-w32 | fp32 | 100.8 / 120.5 | 0.84 | 0.14 / 1.95 | 453 / 420 | 1.08 | 110 / 110 | 109.7 / 129.6 | 0.06 / 0.12 | 0.15 / 1.97 | 0 | 0 | heavier |
| hrnet-w32-384 | fp16 | 177.4 / 201.0 | 0.88 | 0.21 / 2.15 | 439 / 471 | 0.93 | 56 / 56 | 193.7 / 217.2 | 0.05 / 0.12 | 0.23 / 2.17 | 0.011 (0.00023 % off) | 0.41 (0.83 % off) |  |
| hrnet-w32-384 | fp32 | 217.2 / 261.0 | 0.83 | 0.25 / 2.11 | 835 / 867 | 0.96 | 110 / 110 | 235.1 / 276.8 | 0.06 / 0.12 | 0.27 / 2.13 | 0 | 0 |  |
| hrnet-w48 | fp16 | 182.1 / 205.9 | 0.88 | 0.22 / 2.17 | 285 / 268 | 1.06 | 124 / 124 | 191.8 / 214.6 | 0.06 / 0.15 | 0.23 / 2.18 | 0.00028 | 0.0048 | heavier |
| hrnet-w48 | fp32 | 209.2 / 245.7 | 0.85 | 0.25 / 2.17 | 731 / 710 | 1.03 | 245 / 245 | 220.8 / 254.8 | 0.07 / 0.15 | 0.26 / 2.19 | 0 | 0 | heavier |
| hrnet-w48-384 | fp16 | 376.8 / 410.8 | 0.92 | 0.41 / 2.40 | 416 / 384 | 1.08 | 124 / 124 | 397.0 / 428.0 | 0.06 / 0.15 | 0.43 / 2.42 | 0.00065 | 0.00065 | heavier |
| hrnet-w48-384 | fp32 | 442.4 / 509.0 | 0.87 | 0.48 / 2.52 | 983 / 1012 | 0.97 | 245 / 245 | 455.7 / 525.2 | 0.07 / 0.16 | 0.50 / 2.54 | 0 | 0 |  |
| hrnet-w48-fcn-ade20k | fp16 | 152.8 / 162.0 | 0.94 | 0.18 / 2.29 | 313 / 292 | 1.07 | 129 / 128 | 194.1 / 200.5 | 0.27 / 0.20 | 0.23 / 2.33 | 0.00094 | 0.76 (0.056 % off) | heavier |
| hrnet-w48-fcn-ade20k | fp32 | 176.6 / 193.7 | 0.91 | 0.21 / 2.19 | 583 / 609 | 0.96 | 254 / 254 | 216.9 / 236.1 | 0.27 / 0.20 | 0.25 / 2.23 | 0 | 0 |  |
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
| inception_v3.tv_in1k | fp16 | 7.6 / 10.0 | 0.76 | 0.03 / 1.05 | 66 / 152 | 0.44 | 46 / 46 | 8.2 / 10.5 | 0.68 / 0.58 | 0.04 / 1.05 | 0.00084 | 0.00084 |  |
| inception_v3.tv_in1k | fp32 | 8.5 / 11.6 | 0.73 | 0.05 / 1.04 | 125 / 272 | 0.46 | 91 / 91 | 8.9 / 12.1 | 0.62 / 0.57 | 0.05 / 1.04 | 1e-06 | 1e-06 |  |
| lineart | fp16 | 70.6 / 99.9 | 0.71 | 0.08 / 1.02 | 626 / 594 | 1.05 | 8 / 8 | 79.9 / 110.6 | 0.05 / 0.03 | 0.10 / 1.04 | - | 0.028 (0.047 % off) (1 keys differ) | heavier |
| lineart | fp32 | 98.6 / 117.6 | 0.84 | 0.11 / 1.04 | 1245 / 903 | 1.38 | 16 / 16 | 105.4 / 126.6 | 0.05 / 0.03 | 0.13 / 1.06 | - | 1.8e-05 (1 keys differ) | heavier |
| lineart-coarse | fp16 | 70.7 / 99.7 | 0.71 | 0.08 / 1.03 | 626 / 594 | 1.05 | 8 / 8 | 80.1 / 109.8 | 0.05 / 0.03 | 0.10 / 1.05 | - | 0.022 (0.014 % off) (1 keys differ) | heavier |
| lineart-coarse | fp32 | 96.5 / 117.3 | 0.82 | 0.12 / 1.03 | 1245 / 903 | 1.38 | 16 / 16 | 105.9 / 126.5 | 0.05 / 0.03 | 0.13 / 1.04 | - | 1.6e-05 (1 keys differ) | heavier |
| loftr-indoor | fp16 | 434.5 / 332.6 | 1.31 | 0.47 / 2.35 | 1491 / 1237 | 1.21 | 40 / 21 | 457.6 / 1508.1 | 0.09 / 0.04 | 0.47 / 3.58 | - | empty (4 keys differ) | slower heavier |
| loftr-indoor | fp32 | 603.7 / 555.0 | 1.09 | 0.84 / 2.64 | 2947 / 1280 | 2.30 | 64 / 43 | 607.8 / 1651.1 | 0.07 / 0.04 | 0.85 / 3.90 | - | 2.4e-05 | slower heavier |
| loftr-outdoor | fp16 | 435.8 / 332.2 | 1.31 | 0.47 / 2.35 | 1491 / 1237 | 1.21 | 40 / 21 | 438.1 / 1430.9 | 0.09 / 0.04 | 0.48 / 3.54 | - | 0.0018 | slower heavier |
| loftr-outdoor | fp32 | 596.5 / 544.7 | 1.10 | 0.62 / 2.51 | 2947 / 1280 | 2.30 | 64 / 43 | 612.6 / 1647.0 | 0.07 / 0.04 | 0.63 / 3.74 | - | 7.6e-06 | slower heavier |
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
| lw-detr-large-coco | fp16 | 52.6 / 57.6 | 0.91 | 0.09 / 2.09 | 534 / 319 | 1.67 | 118 / 81 | 56.1 / 61.3 | 2.30 / 2.52 | 0.10 / 2.10 | 0.96 (66 % off) | 0.0072 | heavier |
| lw-detr-large-coco | fp32 | 56.7 / 60.5 | 0.94 | 0.10 / 2.07 | 554 / 522 | 1.06 | 154 / 160 | 60.1 / 64.2 | 2.34 / 2.22 | 0.10 / 2.08 | 0.0015 | 2.3e-06 | heavier |
| lw-detr-medium-coco | fp16 | 36.9 / 41.8 | 0.88 | 0.08 / 1.82 | 224 / 290 | 0.78 | 60 / 51 | 39.7 / 45.4 | 2.16 / 2.25 | 0.09 / 1.83 | 0.97 (76 % off) | empty (3 keys differ) |  |
| lw-detr-medium-coco | fp32 | 41.2 / 41.7 | 0.99 | 0.08 / 1.75 | 232 / 231 | 1.01 | 96 / 100 | 44.2 / 45.0 | 2.09 / 2.01 | 0.09 / 1.76 | 0.00041 | 5.4e-06 |  |
| lw-detr-small-coco | fp16 | 23.8 / 28.0 | 0.85 | 0.07 / 1.72 | 185 / 299 | 0.62 | 34 / 24 | 26.5 / 31.2 | 2.02 / 2.02 | 0.08 / 1.73 | 0.96 (66 % off) | empty (3 keys differ) |  |
| lw-detr-small-coco | fp32 | 27.3 / 28.3 | 0.96 | 0.07 / 1.67 | 143 / 185 | 0.77 | 43 / 47 | 30.2 / 31.5 | 2.04 / 2.03 | 0.08 / 1.68 | 0.00023 | 6.2e-06 |  |
| lw-detr-tiny-coco | fp16 | 16.8 / 19.6 | 0.86 | 0.06 / 1.46 | 147 / 258 | 0.57 | 30 / 19 | 19.5 / 22.7 | 2.01 / 2.03 | 0.06 / 1.46 | 0.96 (56 % off) | empty (3 keys differ) |  |
| lw-detr-tiny-coco | fp32 | 19.4 / 20.1 | 0.96 | 0.06 / 1.43 | 124 / 176 | 0.71 | 36 / 37 | 22.1 / 23.2 | 2.03 / 2.00 | 0.06 / 1.44 | 0.00023 | 6.3e-06 |  |
| lw-detr-xlarge-coco | fp16 | 96.9 / 108.3 | 0.90 | 0.12 / 2.05 | 698 / 445 | 1.57 | 288 / 219 | 99.9 / 111.7 | 2.26 / 2.04 | 0.13 / 2.06 | 0.97 (69 % off) | empty (3 keys differ) | heavier |
| lw-detr-xlarge-coco | fp32 | 105.2 / 110.5 | 0.95 | 0.14 / 2.01 | 837 / 884 | 0.95 | 427 / 434 | 108.2 / 114.1 | 2.25 / 2.08 | 0.14 / 2.02 | 0.61 (0.78 % off) | 0.00068 |  |
| mask-rcnn-r50-fpn-3x-coco | fp16 | 97.7 / 102.5 | 0.95 | 0.12 / 0.67 | 750 / 432 | 1.73 | 86 / 52 | 139.1 / 374.4 | 0.55 / 0.53 | 0.18 / 0.97 | - | 1 (0.0068 % off) | heavier |
| mask-rcnn-r50-fpn-3x-coco | fp32 | 119.4 / 125.0 | 0.95 | 0.15 / 0.67 | 1498 / 860 | 1.74 | 169 / 105 | 159.0 / 395.9 | 0.49 / 0.53 | 0.21 / 0.97 | - | 1 (7e-05 % off) | heavier |
| mask2former-swinl-ade20k | fp16 | 590.2 / 651.5 | 0.91 | 1.22 / 6.25 | 849 / 929 | 0.91 | 458 / 455 | 647.9 / 835.7 | 4.67 / 3.13 | 1.65 / 7.21 | - | 0 (2 keys differ) |  |
| mask2former-swinl-ade20k | fp32 | 686.5 / 1000.8 | 0.69 | 0.72 / 6.28 | 1704 / 1760 | 0.97 | 915 / 907 | 755.4 / 1011.8 | 2.85 / 3.15 | 0.84 / 7.73 | - | 0 (2 keys differ) |  |
| mobilenet_v2.tv2_in1k | fp16 | 2.1 / 2.6 | 0.81 | 0.02 / 0.77 | 23 / 24 | 0.95 | 7 / 7 | 2.4 / 2.4 | 0.56 / 0.50 | 0.02 / 0.78 | 0.0012 | 0.0012 |  |
| mobilenet_v2.tv2_in1k | fp32 | 2.3 / 2.4 | 0.97 | 0.03 / 0.75 | 42 / 44 | 0.96 | 14 / 14 | 2.7 / 2.7 | 0.52 / 0.51 | 0.03 / 0.75 | 1.5e-06 | 1.5e-06 |  |
| mobilenet_v3_large.tv2_in1k | fp16 | 2.2 / 3.3 | 0.68 | 0.02 / 0.87 | 18 / 32 | 0.57 | 11 / 11 | 2.5 / 3.3 | 0.57 / 0.50 | 0.02 / 0.88 | 0.0025 | 0.0025 |  |
| mobilenet_v3_large.tv2_in1k | fp32 | 3.7 / 3.2 | 1.15 | 0.04 / 0.86 | 34 / 53 | 0.64 | 21 / 21 | 2.9 / 3.6 | 0.51 / 0.51 | 0.05 / 0.86 | 1.4e-06 | 1.4e-06 | slower |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp16 | 1.8 / 1.7 | 1.02 | 0.02 / 0.52 | 15 / 23 | 0.65 | 8 / 8 | 2.3 / 2.2 | 0.73 / 0.72 | 0.02 / 0.52 | 0.0015 | 0.0015 | slower |
| mobilenetv1_100.ra4_e3600_r224_in1k | fp32 | 1.9 / 2.0 | 0.95 | 0.03 / 0.52 | 29 / 41 | 0.72 | 16 / 16 | 2.5 / 2.5 | 0.73 / 0.69 | 0.04 / 0.53 | 1.8e-06 | 1.8e-06 |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp16 | 3.0 / 3.3 | 0.91 | 0.02 / 0.91 | 30 / 41 | 0.73 | 19 / 19 | 3.5 / 3.7 | 0.79 / 0.70 | 0.03 / 0.92 | 0.00074 | 0.00074 |  |
| mobilenetv4_conv_medium.e500_r256_in1k | fp32 | 4.1 / 3.5 | 1.17 | 0.04 / 0.90 | 56 / 74 | 0.76 | 37 / 37 | 3.8 / 4.1 | 0.73 / 0.71 | 0.04 / 0.90 | 5.4e-07 | 5.4e-07 | slower |
| mvsanywhere | fp16 | 414.7 / 372.0 | 1.11 | 3.50 / 3.45 | 767 / 767 | 1.00 | 386 / 386 | 424.6 / 3490.1 | 0.10 / 0.10 | 3.51 / 6.59 | 0 | 0.045 (0.49 % off) | slower |
| mvsanywhere | fp32 | 502.0 / 471.5 | 1.06 | 3.54 / 3.54 | 1407 / 1407 | 1.00 | 771 / 771 | 508.6 / 3574.8 | 0.10 / 0.10 | 3.56 / 6.70 | 0 | 0.00018 | slower |
| nafnet-gopro | fp16 | 214.5 / 249.6 | 0.86 | 0.23 / 3.09 | 769 / 772 | 1.00 | 130 / 255 | 215.1 / 250.1 | 0.28 / 0.16 | 0.23 / 3.10 | 0.0019 | 1 (0.052 % off) |  |
| nafnet-gopro | fp32 | 257.5 / 291.6 | 0.88 | 0.29 / 3.06 | 1170 / 839 | 1.39 | 259 / 384 | 257.9 / 292.4 | 0.26 / 0.16 | 0.30 / 3.07 | 2e-06 | 0.0085 | heavier |
| nafnet-sidd | fp16 | 219.4 / 207.7 | 1.06 | 0.23 / 2.25 | 804 / 578 | 1.39 | 221 / 221 | 220.0 / 207.7 | 0.41 / 0.21 | 0.24 / 2.26 | 0.0015 | 0.0034 | slower heavier |
| nafnet-sidd | fp32 | 254.2 / 268.5 | 0.95 | 0.29 / 2.21 | 1346 / 896 | 1.50 | 442 / 442 | 254.5 / 268.8 | 0.40 / 0.21 | 0.29 / 2.21 | 1.2e-06 | 2.6e-06 | heavier |
| oneformer-swinl-ade20k | fp16 | 643.5 / 760.8 | 0.85 | 0.89 / 6.75 | 1364 / 1002 | 1.36 | 523 / 536 | 861.8 / 1017.3 | 2.92 / 2.80 | 1.13 / 7.07 | - | 0 (2 keys differ) | heavier |
| oneformer-swinl-ade20k | fp32 | 735.6 / 948.3 | 0.78 | 1.18 / 6.84 | 1894 / 2040 | 0.93 | 980 / 1070 | 990.3 / 1177.7 | 2.71 / 2.76 | 1.44 / 7.11 | - | 0 (2 keys differ) |  |
| owlv2-b16 | fp16 | 283.2 / 279.2 | 1.01 | 0.36 / 1.22 | 505 / 338 | 1.49 | 296 / 175 | 284.6 / 289.7 | 0.09 / 2.05 | 0.36 / 1.33 | - | empty | heavier |
| owlv2-b16 | fp32 | 365.0 / 367.5 | 0.99 | 0.45 / 1.31 | 860 / 544 | 1.58 | 585 / 343 | 366.1 / 378.0 | 0.10 / 2.08 | 0.46 / 1.42 | - | empty | heavier |
| owlv2-l14 | fp16 | 1419.7 / 1410.2 | 1.01 | 1.52 / 2.74 | 1055 / 793 | 1.33 | 835 / 599 | 1421.3 / 1420.3 | 0.11 / 2.20 | 1.53 / 2.92 | - | empty | heavier |
| owlv2-l14 | fp32 | 1869.0 / 1857.1 | 1.01 | 2.07 / 3.18 | 2087 / 1563 | 1.34 | 1659 / 1187 | 1870.5 / 1868.9 | 1.02 / 2.17 | 2.08 / 3.37 | - | empty | heavier |
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
| pspnet-r101-d8-ade20k | fp16 | 338.6 / 365.6 | 0.93 | 0.41 / 1.92 | 1042 / 768 | 1.36 | 125 / 125 | 377.3 / 405.3 | 0.21 / 0.14 | 0.45 / 1.97 | 0.00078 | 0 | heavier |
| pspnet-r101-d8-ade20k | fp32 | 434.8 / 486.7 | 0.89 | 0.50 / 1.99 | 1547 / 1724 | 0.90 | 250 / 250 | 470.7 / 548.4 | 0.21 / 0.15 | 0.54 / 2.04 | 2.9e-07 | 0 |  |
| raft-stereo-eth3d | fp16 | 3176.0 / 4061.6 | 0.78 | 3.23 / 8.48 | 1630 / 1166 | 1.40 | 29 / 82 | 3178.6 / 4052.2 | 0.07 / 0.06 | 3.23 / 8.48 | 0.058 (0.06 % off) | 0.049 (0.015 % off) | heavier |
| raft-stereo-eth3d | fp32 | 4832.4 / 5011.0 | 0.96 | 4.87 / 9.57 | 2943 / 1872 | 1.57 | 42 / 116 | 4848.8 / 5022.1 | 0.07 / 0.06 | 4.88 / 9.57 | 0.00012 | 0.00012 | heavier |
| raft-stereo-middlebury | fp16 | 3168.7 / 4054.7 | 0.78 | 3.23 / 8.62 | 1630 / 1128 | 1.45 | 29 / 82 | 3185.3 / 4058.4 | 0.07 / 0.06 | 3.24 / 8.63 | 0.034 (0.15 % off) | 0.036 (0.13 % off) | heavier |
| raft-stereo-middlebury | fp32 | 4808.4 / 5041.7 | 0.95 | 4.84 / 9.51 | 2943 / 1872 | 1.57 | 42 / 116 | 4845.5 / 5028.2 | 0.07 / 0.06 | 4.85 / 9.52 | 0.00092 | 0.00098 | heavier |
| raft-stereo-sceneflow | fp16 | 3191.9 / 4043.8 | 0.79 | 3.23 / 8.52 | 1630 / 1042 | 1.56 | 29 / 82 | 3179.8 / 4049.5 | 0.07 / 0.06 | 3.24 / 8.52 | 0.27 (0.11 % off) | 0.28 (0.057 % off) | heavier |
| raft-stereo-sceneflow | fp32 | 4812.2 / 5045.0 | 0.95 | 4.86 / 9.46 | 2943 / 1872 | 1.57 | 42 / 116 | 4836.6 / 5021.1 | 0.07 / 0.06 | 4.87 / 9.47 | 0.00012 | 0.00012 | heavier |
| real-hat-gan-x4 | fp16 | 7709.2 / 10941.2 | 0.70 | 7.76 / 14.16 | 2282 / 2938 | 0.78 | 96 / 335 | 7318.1 / 10434.9 | 0.76 / 0.60 | 7.78 / 14.17 | 0.032 (0.014 % off) | 0.033 (0.015 % off) |  |
| real-hat-gan-x4 | fp32 | 10545.0 / 22023.5 | 0.48 | 11.38 / 21.03 | 4514 / 4506 | 1.00 | 189 / 411 | 10530.7 / 21139.3 | 1.01 / 0.52 | 11.41 / 21.05 | 0.0001 | 0.0001 |  |
| realesr-general-x4v3 | fp16 | 89.4 / 97.5 | 0.92 | 0.13 / 1.33 | 472 / 576 | 0.82 | 12 / 12 | 92.1 / 98.3 | 0.04 / 0.03 | 0.15 / 1.35 | 0.015 (0.0043 % off) | 0.015 (0.0043 % off) |  |
| realesr-general-x4v3 | fp32 | 146.6 / 158.5 | 0.93 | 0.20 / 1.39 | 938 / 938 | 1.00 | 23 / 23 | 149.2 / 158.7 | 0.04 / 0.03 | 0.22 / 1.40 | 0 | 0 |  |
| realesr-general-x4v3 | fp16 | 186.3 / 241.2 | 0.77 | 0.21 / 1.41 | 199 / 279 | 0.71 | 3 / 3 | 188.8 / 242.4 | 0.04 / 0.03 | 0.23 / 1.42 | 0 | 0 |  |
| realesr-general-x4v3 | fp32 | 227.9 / 322.7 | 0.71 | 0.26 / 1.43 | 296 / 360 | 0.82 | 5 / 5 | 230.4 / 324.3 | 0.04 / 0.03 | 0.28 / 1.43 | 0 | 0 |  |
| realesrgan-x2plus | fp16 | 499.2 / 506.6 | 0.99 | 0.56 / 3.99 | 1018 / 847 | 1.20 | 163 / 165 | 500.5 / 507.0 | 0.52 / 0.10 | 0.56 / 4.00 | 0.11 (2.6 % off) | 0.13 (3.8 % off) | heavier |
| realesrgan-x2plus | fp32 | 1276.3 / 1386.6 | 0.92 | 1.32 / 3.79 | 469 / 822 | 0.57 | 65 / 65 | 1275.0 / 1387.6 | 0.52 / 0.10 | 1.32 / 3.79 | 9.6e-05 | 0.00012 |  |
| realesrgan-x2plus | fp16 | 1006.8 / 1057.8 | 0.95 | 1.05 / 3.75 | 284 / 421 | 0.68 | 35 / 37 | 1008.2 / 1056.9 | 0.52 / 0.10 | 1.05 / 3.75 | 0.14 (4 % off) | 0.18 (5.8 % off) |  |
| realesrgan-x2plus | fp32 | 1280.0 / 1366.0 | 0.94 | 1.32 / 3.83 | 469 / 566 | 0.83 | 65 / 65 | 1283.4 / 1363.2 | 0.62 / 0.10 | 1.33 / 3.83 | 9.6e-05 | 0.00012 |  |
| realesrgan-x4plus | fp16 | 1989.4 / 1987.8 | 1.00 | 2.02 / 5.38 | 1322 / 1292 | 1.02 | 163 / 165 | 1986.8 / 2004.0 | 0.53 / 0.10 | 2.04 / 5.39 | 0.014 (0.00023 % off) | 0.014 (0.00023 % off) | heavier |
| realesrgan-x4plus | fp32 | 5058.9 / 5356.9 | 0.94 | 5.18 / 7.80 | 1381 / 1883 | 0.73 | 65 / 65 | 5058.6 / 5572.1 | 0.62 / 0.10 | 5.21 / 7.81 | 1.9e-05 | 1.9e-05 |  |
| realesrgan-x4plus | fp16 | 4114.4 / 4302.8 | 0.96 | 4.14 / 7.24 | 801 / 867 | 0.92 | 35 / 37 | 4117.2 / 4304.2 | 0.61 / 0.10 | 4.16 / 7.25 | 0.017 (0.00052 % off) | 0.017 (0.00052 % off) |  |
| realesrgan-x4plus | fp32 | 5068.6 / 5405.7 | 0.94 | 5.22 / 7.88 | 1381 / 1531 | 0.90 | 65 / 65 | 5074.6 / 5579.2 | 0.64 / 0.10 | 5.24 / 7.89 | 1.9e-05 | 1.9e-05 |  |
| realesrgan-x4plus-anime-6b | fp16 | 685.8 / 658.9 | 1.04 | 0.76 / 2.22 | 1203 / 1171 | 1.03 | 44 / 44 | 665.0 / 668.5 | 0.46 / 0.05 | 0.78 / 2.23 | 0.18 (0.43 % off) | 0.21 (0.55 % off) | slower heavier |
| realesrgan-x4plus-anime-6b | fp32 | 1571.8 / 1717.7 | 0.92 | 1.60 / 2.99 | 1333 / 1836 | 0.73 | 18 / 18 | 1587.7 / 1726.8 | 0.48 / 0.05 | 1.62 / 3.00 | 0.00026 | 0.00031 |  |
| realesrgan-x4plus-anime-6b | fp16 | 1265.7 / 1349.1 | 0.94 | 1.29 / 2.72 | 775 / 840 | 0.92 | 9 / 10 | 1267.9 / 1351.3 | 0.60 / 0.05 | 1.31 / 2.74 | 0.23 (0.56 % off) | 0.27 (0.71 % off) |  |
| realesrgan-x4plus-anime-6b | fp32 | 1571.5 / 1733.4 | 0.91 | 1.62 / 3.13 | 1333 / 1483 | 0.90 | 18 / 18 | 1577.9 / 1731.8 | 0.59 / 0.04 | 1.64 / 3.14 | 0.00026 | 0.00031 |  |
| regnety_160.tv2_in1k | fp16 | 17.5 / 18.6 | 0.94 | 0.04 / 0.99 | 198 / 280 | 0.71 | 160 / 160 | 18.0 / 19.0 | 0.73 / 0.57 | 0.04 / 1.00 | 0.00029 | 0.00029 |  |
| regnety_160.tv2_in1k | fp32 | 18.4 / 23.6 | 0.78 | 0.05 / 0.96 | 377 / 574 | 0.66 | 319 / 319 | 18.9 / 24.1 | 0.66 / 0.56 | 0.05 / 0.96 | 3.5e-07 | 3.5e-07 |  |
| repvit_m0_9.dist_300e_in1k | fp16 | 3.5 / 4.5 | 0.77 | 0.03 / 1.31 | 18 / 35 | 0.51 | 10 / 11 | 4.0 / 5.0 | 0.79 / 0.70 | 0.03 / 1.32 | 0.0078 | 0.0078 |  |
| repvit_m0_9.dist_300e_in1k | fp32 | 3.9 / 4.9 | 0.79 | 0.05 / 1.26 | 31 / 62 | 0.50 | 20 / 21 | 4.2 / 6.3 | 0.75 / 0.70 | 0.05 / 1.26 | 1.4e-06 | 1.4e-06 |  |
| repvit_m0_9.dist_450e_in1k | fp16 | 3.5 / 4.5 | 0.78 | 0.03 / 1.36 | 17 / 33 | 0.51 | 10 / 11 | 3.9 / 5.0 | 0.78 / 0.71 | 0.03 / 1.36 | 0.012 (0.2 % off) | 0.012 (0.2 % off) |  |
| repvit_m0_9.dist_450e_in1k | fp32 | 3.7 / 4.8 | 0.77 | 0.05 / 1.32 | 32 / 61 | 0.53 | 20 / 21 | 4.3 / 5.4 | 0.76 / 0.74 | 0.05 / 1.32 | 2.3e-06 | 2.3e-06 |  |
| repvit_m1_0.dist_300e_in1k | fp16 | 3.8 / 5.0 | 0.76 | 0.03 / 1.31 | 21 / 41 | 0.52 | 14 / 15 | 4.3 / 5.5 | 0.79 / 0.69 | 0.04 / 1.31 | 0.01 (0.1 % off) | 0.01 (0.1 % off) |  |
| repvit_m1_0.dist_300e_in1k | fp32 | 4.1 / 5.3 | 0.77 | 0.05 / 1.26 | 40 / 78 | 0.51 | 27 / 28 | 4.6 / 5.9 | 0.73 / 0.70 | 0.05 / 1.27 | 1.9e-06 | 1.9e-06 |  |
| repvit_m1_0.dist_450e_in1k | fp16 | 3.7 / 4.9 | 0.76 | 0.03 / 1.32 | 21 / 45 | 0.48 | 14 / 15 | 4.2 / 5.5 | 0.79 / 0.71 | 0.03 / 1.32 | 0.0065 | 0.0065 |  |
| repvit_m1_0.dist_450e_in1k | fp32 | 4.1 / 5.3 | 0.78 | 0.05 / 1.28 | 40 / 77 | 0.51 | 27 / 28 | 4.5 / 5.8 | 0.73 / 0.69 | 0.05 / 1.28 | 1.3e-06 | 1.3e-06 |  |
| repvit_m1_1.dist_300e_in1k | fp16 | 3.6 / 4.6 | 0.78 | 0.03 / 1.25 | 24 / 45 | 0.54 | 16 / 17 | 4.0 / 5.2 | 0.80 / 0.72 | 0.03 / 1.25 | 0.0061 | 0.0061 |  |
| repvit_m1_1.dist_300e_in1k | fp32 | 3.9 / 5.2 | 0.76 | 0.05 / 1.20 | 47 / 83 | 0.57 | 32 / 34 | 4.4 / 5.7 | 0.73 / 0.71 | 0.05 / 1.21 | 8.9e-07 | 8.9e-07 |  |
| repvit_m1_1.dist_450e_in1k | fp16 | 3.5 / 4.6 | 0.77 | 0.03 / 1.24 | 24 / 46 | 0.53 | 16 / 17 | 4.0 / 5.1 | 0.78 / 0.70 | 0.03 / 1.24 | 0.021 (5.2 % off) | 0.021 (5.2 % off) |  |
| repvit_m1_1.dist_450e_in1k | fp32 | 3.8 / 5.0 | 0.76 | 0.05 / 1.25 | 46 / 83 | 0.56 | 32 / 34 | 4.4 / 5.5 | 0.73 / 0.73 | 0.05 / 1.25 | 2.6e-06 | 2.6e-06 |  |
| repvit_m1_5.dist_300e_in1k | fp16 | 5.4 / 7.3 | 0.74 | 0.04 / 1.87 | 39 / 64 | 0.61 | 27 / 28 | 6.0 / 8.4 | 0.86 / 0.75 | 0.04 / 1.87 | 0.0073 | 0.0073 |  |
| repvit_m1_5.dist_300e_in1k | fp32 | 6.1 / 8.6 | 0.71 | 0.06 / 1.75 | 73 / 120 | 0.61 | 54 / 56 | 6.5 / 9.0 | 0.80 / 0.75 | 0.06 / 1.75 | 1.2e-06 | 1.2e-06 |  |
| repvit_m1_5.dist_450e_in1k | fp16 | 5.5 / 7.6 | 0.73 | 0.04 / 1.79 | 39 / 66 | 0.59 | 27 / 28 | 6.2 / 8.2 | 0.86 / 0.73 | 0.04 / 1.79 | 0.037 (12 % off) | 0.037 (12 % off) |  |
| repvit_m1_5.dist_450e_in1k | fp32 | 6.2 / 8.4 | 0.73 | 0.06 / 1.71 | 74 / 126 | 0.58 | 54 / 56 | 6.7 / 9.0 | 0.76 / 0.74 | 0.06 / 1.71 | 2e-06 | 2e-06 |  |
| repvit_m2_3.dist_300e_in1k | fp16 | 9.4 / 12.0 | 0.78 | 0.04 / 2.41 | 63 / 108 | 0.59 | 45 / 46 | 9.9 / 13.0 | 0.93 / 0.72 | 0.04 / 2.41 | 0.0098 | 0.0098 |  |
| repvit_m2_3.dist_300e_in1k | fp32 | 10.4 / 13.9 | 0.75 | 0.06 / 2.18 | 120 / 194 | 0.61 | 88 / 91 | 11.1 / 14.8 | 0.80 / 0.73 | 0.07 / 2.18 | 1.2e-06 | 1.2e-06 |  |
| repvit_m2_3.dist_450e_in1k | fp16 | 9.1 / 12.3 | 0.74 | 0.04 / 2.33 | 63 / 109 | 0.58 | 45 / 46 | 9.7 / 13.0 | 0.91 / 0.73 | 0.04 / 2.33 | 0.007 | 0.007 |  |
| repvit_m2_3.dist_450e_in1k | fp32 | 10.6 / 13.9 | 0.76 | 0.07 / 2.22 | 120 / 194 | 0.62 | 88 / 91 | 11.2 / 14.6 | 0.85 / 0.73 | 0.07 / 2.22 | 9.2e-07 | 9.2e-07 |  |
| resnet152.tv2_in1k | fp16 | 15.7 / 17.2 | 0.91 | 0.03 / 1.44 | 177 / 224 | 0.79 | 115 / 115 | 16.1 / 17.9 | 0.78 / 0.65 | 0.04 / 1.44 | 0.00062 | 0.00062 |  |
| resnet152.tv2_in1k | fp32 | 17.8 / 20.7 | 0.86 | 0.05 / 1.41 | 335 / 384 | 0.87 | 229 / 229 | 18.2 / 21.2 | 0.71 / 0.61 | 0.05 / 1.41 | 4.6e-07 | 4.6e-07 |  |
| resnet18.tv_in1k | fp16 | 3.4 / 3.9 | 0.88 | 0.02 / 0.49 | 30 / 52 | 0.58 | 22 / 22 | 3.7 / 4.2 | 0.54 / 0.49 | 0.02 / 0.49 | 0.00055 | 0.00055 |  |
| resnet18.tv_in1k | fp32 | 3.9 / 4.5 | 0.86 | 0.03 / 0.48 | 59 / 102 | 0.58 | 45 / 45 | 4.3 / 4.9 | 0.51 / 0.51 | 0.03 / 0.48 | 3.7e-07 | 3.7e-07 |  |
| resnet50.tv2_in1k | fp16 | 6.2 / 6.8 | 0.91 | 0.02 / 0.92 | 76 / 99 | 0.77 | 49 / 49 | 6.5 / 7.2 | 0.59 / 0.50 | 0.02 / 0.92 | 0.00082 | 0.00082 |  |
| resnet50.tv2_in1k | fp32 | 7.0 / 8.1 | 0.87 | 0.04 / 0.76 | 147 / 192 | 0.77 | 97 / 97 | 7.4 / 8.5 | 0.55 / 0.51 | 0.04 / 0.76 | 5.5e-07 | 5.5e-07 |  |
| retinaface-mobilenet0.25 | fp16 | 10.6 / 11.9 | 0.89 | 0.04 / 1.07 | 92 / 178 | 0.52 | 1 / 1 | 15.5 / 17.2 | 0.04 / 0.12 | 0.06 / 1.07 | 0.0077 | empty |  |
| retinaface-mobilenet0.25 | fp32 | 12.9 / 14.7 | 0.88 | 0.05 / 1.05 | 136 / 180 | 0.76 | 2 / 2 | 18.1 / 20.1 | 0.04 / 0.11 | 0.06 / 1.05 | 3.7e-06 | empty |  |
| retinaface-resnet50 | fp16 | 119.6 / 132.8 | 0.90 | 0.15 / 1.27 | 469 / 372 | 1.26 | 52 / 52 | 125.1 / 138.9 | 0.05 / 0.17 | 0.16 / 1.27 | 0.014 (0.0017 % off) | empty | heavier |
| retinaface-resnet50 | fp32 | 151.1 / 171.8 | 0.88 | 0.19 / 1.27 | 902 / 697 | 1.29 | 104 / 104 | 157.6 / 177.3 | 0.05 / 0.17 | 0.20 / 1.28 | 1.2e-05 | empty | heavier |
| rf-detr-keypoint-preview | fp16 | 63.0 / 62.0 | 1.02 | 0.10 / 3.16 | 354 / 225 | 1.57 | 95 / 71 | 63.8 / 63.6 | 2.01 / 1.93 | 0.11 / 3.16 | - | empty | heavier |
| rf-detr-keypoint-preview | fp32 | 72.0 / 74.2 | 0.97 | 0.15 / 3.32 | 428 / 251 | 1.71 | 138 / 142 | 72.9 / 76.0 | 2.20 / 2.11 | 0.16 / 3.33 | - | empty | heavier |
| rf-detr-l-coco | fp16 | 45.1 / 41.5 | 1.09 | 0.08 / 1.92 | 201 / 191 | 1.05 | 73 / 60 | 47.6 / 45.1 | 2.08 / 1.93 | 0.08 / 1.93 | 0.98 (69 % off) | empty | slower heavier |
| rf-detr-l-coco | fp32 | 52.7 / 51.2 | 1.03 | 0.09 / 1.86 | 268 / 228 | 1.18 | 116 / 120 | 55.6 / 55.0 | 2.05 / 1.91 | 0.09 / 1.87 | 0.42 (3.6 % off) | empty | slower heavier |
| rf-detr-m-coco | fp16 | 32.2 / 29.6 | 1.09 | 0.07 / 1.88 | 179 / 179 | 1.00 | 73 / 59 | 34.7 / 33.0 | 2.03 / 1.95 | 0.08 / 1.88 | 0.9 (70 % off) | empty | slower |
| rf-detr-m-coco | fp32 | 36.4 / 35.3 | 1.03 | 0.07 / 1.83 | 237 / 226 | 1.05 | 115 / 119 | 39.0 / 38.5 | 2.03 / 1.94 | 0.08 / 1.84 | 0.00048 | empty | slower heavier |
| rf-detr-n-coco | fp16 | 16.5 / 15.5 | 1.06 | 0.06 / 1.64 | 113 / 146 | 0.77 | 61 / 53 | 19.0 / 18.6 | 2.00 / 2.00 | 0.07 / 1.64 | 0.93 (51 % off) | empty | slower |
| rf-detr-n-coco | fp32 | 18.1 / 17.6 | 1.03 | 0.07 / 1.58 | 164 / 190 | 0.86 | 102 / 106 | 20.6 / 20.8 | 2.00 / 1.93 | 0.07 / 1.59 | 0.75 (0.56 % off) | empty | slower |
| rf-detr-s-coco | fp16 | 25.7 / 24.2 | 1.06 | 0.07 / 1.79 | 149 / 165 | 0.90 | 67 / 56 | 28.3 / 27.6 | 2.03 / 1.91 | 0.07 / 1.79 | 0.95 (71 % off) | empty | slower |
| rf-detr-s-coco | fp32 | 29.0 / 28.1 | 1.03 | 0.07 / 1.76 | 209 / 219 | 0.95 | 109 / 113 | 31.7 / 31.4 | 2.04 / 1.95 | 0.07 / 1.76 | 0.86 (1.4 % off) | empty | slower |
| rf-detr-seg-2xl-coco | fp16 | 126.7 / 120.4 | 1.05 | 0.17 / 2.34 | 394 / 292 | 1.35 | 88 / 69 | 129.4 / 123.7 | 2.07 / 1.96 | 0.18 / 2.34 | 1 (73 % off) | empty | slower heavier |
| rf-detr-seg-2xl-coco | fp32 | 156.9 / 155.4 | 1.01 | 0.21 / 2.26 | 576 / 433 | 1.33 | 134 / 138 | 159.8 / 159.7 | 2.01 / 1.91 | 0.21 / 2.26 | 0.00069 | empty | heavier |
| rf-detr-seg-l-coco | fp16 | 49.0 / 45.4 | 1.08 | 0.09 / 2.10 | 193 / 221 | 0.87 | 80 / 64 | 50.8 / 47.9 | 2.03 / 1.90 | 0.10 / 2.10 | 0.96 (84 % off) | empty | slower |
| rf-detr-seg-l-coco | fp32 | 58.1 / 56.6 | 1.03 | 0.10 / 2.05 | 289 / 316 | 0.91 | 124 / 127 | 60.0 / 59.1 | 2.00 / 1.94 | 0.11 / 2.06 | 0.00097 | empty | slower |
| rf-detr-seg-m-coco | fp16 | 38.3 / 35.3 | 1.09 | 0.08 / 2.09 | 172 / 211 | 0.81 | 80 / 63 | 40.2 / 37.9 | 1.98 / 1.91 | 0.09 / 2.09 | 0.93 (86 % off) | empty | slower |
| rf-detr-seg-m-coco | fp32 | 44.3 / 42.6 | 1.04 | 0.09 / 2.03 | 254 / 289 | 0.88 | 124 / 126 | 46.0 / 45.0 | 2.07 / 1.91 | 0.10 / 2.04 | 0.0042 | empty | slower |
| rf-detr-seg-n-coco | fp16 | 23.1 / 21.4 | 1.08 | 0.07 / 1.96 | 127 / 150 | 0.85 | 74 / 59 | 24.2 / 23.2 | 1.98 / 1.90 | 0.08 / 1.97 | 0.94 (77 % off) | empty | slower |
| rf-detr-seg-n-coco | fp32 | 25.8 / 24.5 | 1.05 | 0.07 / 1.90 | 190 / 249 | 0.76 | 117 / 118 | 27.2 / 26.2 | 2.00 / 1.99 | 0.08 / 1.91 | 0.00012 | empty | slower |
| rf-detr-seg-s-coco | fp16 | 29.5 / 27.3 | 1.08 | 0.08 / 1.98 | 149 / 176 | 0.85 | 74 / 59 | 30.5 / 29.1 | 1.99 / 1.93 | 0.08 / 1.98 | 0.93 (79 % off) | empty | slower |
| rf-detr-seg-s-coco | fp32 | 33.3 / 31.8 | 1.05 | 0.08 / 1.95 | 223 / 266 | 0.84 | 117 / 119 | 34.8 / 33.6 | 1.99 / 1.90 | 0.08 / 1.95 | 0.001 | empty | slower |
| rf-detr-seg-xl-coco | fp16 | 79.3 / 75.3 | 1.05 | 0.12 / 2.24 | 252 / 251 | 1.00 | 87 / 68 | 82.0 / 78.7 | 2.03 / 1.91 | 0.12 / 2.25 | 0.99 (79 % off) | empty | slower |
| rf-detr-seg-xl-coco | fp32 | 96.5 / 95.4 | 1.01 | 0.14 / 2.19 | 377 / 331 | 1.14 | 132 / 136 | 99.1 / 99.0 | 2.00 / 1.94 | 0.14 / 2.20 | 0.00081 | empty | heavier |
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
| sam2.1-hiera-base-plus | fp16 | 213.5 / 244.4 | 0.87 | 0.26 / 2.27 | 506 / 475 | 1.06 | 217 / 155 | 216.9 / 262.8 | 0.09 / 0.08 | 0.26 / 2.30 | 0.008 | 1 (0.0028 % off) | heavier |
| sam2.1-hiera-base-plus | fp32 | 247.5 / 263.0 | 0.94 | 0.29 / 2.23 | 759 / 636 | 1.19 | 367 / 300 | 248.8 / 281.3 | 0.10 / 0.08 | 0.30 / 2.25 | 1.5e-05 | 1.6e-05 | heavier |
| sam2.1-hiera-large | fp16 | 496.8 / 562.6 | 0.88 | 0.54 / 3.60 | 838 / 734 | 1.14 | 490 / 434 | 498.9 / 581.9 | 0.11 / 0.11 | 0.55 / 3.63 | 0.08 (0.0035 % off) | 1 (0.0023 % off) | heavier |
| sam2.1-hiera-large | fp32 | 568.2 / 609.6 | 0.93 | 0.61 / 3.61 | 1373 / 1317 | 1.04 | 911 / 855 | 569.0 / 627.6 | 0.13 / 0.11 | 0.62 / 3.64 | 2.1e-05 | 9e-06 | heavier |
| sam2.1-hiera-small | fp16 | 123.1 / 138.2 | 0.89 | 0.16 / 1.80 | 378 / 362 | 1.04 | 142 / 86 | 124.5 / 157.4 | 0.08 / 0.07 | 0.17 / 1.82 | 0.012 (0.00014 % off) | 1 (0.012 % off) | heavier |
| sam2.1-hiera-small | fp32 | 148.2 / 152.3 | 0.97 | 0.19 / 1.82 | 547 / 463 | 1.18 | 219 / 163 | 151.1 / 171.5 | 0.09 / 0.07 | 0.20 / 1.84 | 5.1e-06 | 1 (3e-05 % off) | heavier |
| sam2.1-hiera-tiny | fp16 | 103.1 / 115.6 | 0.89 | 0.15 / 1.61 | 378 / 349 | 1.08 | 129 / 73 | 105.5 / 134.8 | 0.08 / 0.07 | 0.15 / 1.64 | 0.0077 | 1 (0.024 % off) | heavier |
| sam2.1-hiera-tiny | fp32 | 127.4 / 128.4 | 0.99 | 0.17 / 1.63 | 520 / 436 | 1.19 | 192 / 136 | 128.7 / 146.9 | 0.08 / 0.07 | 0.17 / 1.65 | 5.6e-06 | 1 (6e-05 % off) | heavier |
| sam3 | fp16 | 1504.2 / 1384.6 | 1.09 | 1.54 / 4.40 | 2037 / 1385 | 1.47 | 1633 / 892 | 1516.3 / 2094.0 | 0.16 / 1.00 | 1.58 / 5.80 | 0.21 (0.84 % off) | empty | slower heavier |
| sam3 | fp32 | 1776.2 / 1608.6 | 1.10 | 2.30 / 5.02 | 3972 / 2551 | 1.56 | 3205 / 1776 | 1793.7 / 2332.2 | 3.49 / 0.97 | 2.35 / 6.72 | 0.19 (0.7 % off) | empty | slower heavier |
| sapiens-seg-0.3b-goliath | fp16 | 1140.1 / 1509.1 | 0.76 | 1.18 / 3.15 | 1822 / 1808 | 1.01 | 648 / 648 | 1144.0 / 1829.0 | 0.32 / 0.10 | 1.20 / 3.47 | 0.003 | 0 |  |
| sapiens-seg-0.3b-goliath | fp32 | 1399.4 / 1783.6 | 0.78 | 1.82 / 3.42 | 3631 / 3288 | 1.10 | 1295 / 1295 | 1411.4 / 2102.4 | 4.20 / 0.14 | 1.89 / 3.74 | 2.6e-06 | 0 | heavier |
| sapiens-seg-1b-goliath | fp16 | 2650.0 / 3004.7 | 0.88 | 2.81 / 7.35 | 3426 / 3730 | 0.92 | 2248 / 2248 | 2655.2 / 3316.0 | 23.74 / 0.34 | 2.84 / 7.72 | 0.002 | 0 |  |
| sapiens-seg-1b-goliath | fp32 | 3438.1 / 4241.3 | 0.81 | 3.92 / 8.73 | 6839 / 6491 | 1.05 | 4497 / 4497 | 3239.8 / 4225.5 | 22.68 / 0.33 | 3.95 / 9.07 | 2.1e-06 | 0 | heavier |
| segformer-b0-ade20k | fp16 | 19.2 / 17.3 | 1.11 | 0.05 / 0.98 | 158 / 114 | 1.39 | 7 / 7 | 51.2 / 53.7 | 2.00 / 1.94 | 0.09 / 1.03 | 0.001 | 0.5 (0.067 % off) | slower heavier |
| segformer-b0-ade20k | fp32 | 23.0 / 22.5 | 1.02 | 0.07 / 0.99 | 254 / 149 | 1.70 | 14 / 14 | 54.8 / 59.4 | 1.97 / 1.94 | 0.11 / 1.03 | 8.9e-07 | 0 | slower heavier |
| segformer-b2-clothes | fp16 | 26.6 / 24.6 | 1.08 | 0.06 / 1.41 | 225 / 158 | 1.42 | 48 / 48 | 27.3 / 25.8 | 2.11 / 1.98 | 0.07 / 1.41 | 0.00081 | 0 | slower heavier |
| segformer-b2-clothes | fp32 | 31.4 / 31.9 | 0.98 | 0.09 / 1.38 | 446 / 291 | 1.53 | 96 / 96 | 32.1 / 33.3 | 2.02 / 2.01 | 0.09 / 1.38 | 6.9e-07 | 0 | heavier |
| segformer-b5-ade20k | fp16 | 223.7 / 205.4 | 1.09 | 0.25 / 3.44 | 710 / 555 | 1.28 | 158 / 158 | 256.1 / 249.6 | 2.32 / 2.21 | 0.29 / 3.49 | 0.00089 | 1 (0.015 % off) | slower heavier |
| segformer-b5-ade20k | fp32 | 264.6 / 258.5 | 1.02 | 0.32 / 3.41 | 1410 / 780 | 1.81 | 314 / 314 | 297.4 / 301.8 | 2.33 / 2.22 | 0.37 / 3.46 | 7.1e-07 | 0 | slower heavier |
| segformer-b5-face-parsing | fp16 | 67.2 / 62.9 | 1.07 | 0.10 / 3.21 | 335 / 274 | 1.22 | 158 / 158 | 68.2 / 64.4 | 2.25 / 2.11 | 0.10 / 3.21 | 0.00088 | 0 | slower heavier |
| segformer-b5-face-parsing | fp32 | 78.2 / 79.4 | 0.98 | 0.13 / 3.22 | 664 / 495 | 1.34 | 314 / 314 | 79.0 / 82.9 | 2.23 / 2.13 | 0.13 / 3.22 | 5.4e-07 | 0 | heavier |
| segnext-mscan-b-ade20k | fp16 | 77.9 / 70.2 | 1.11 | 0.10 / 2.28 | 316 / 255 | 1.24 | 54 / 53 | 111.2 / 103.2 | 0.18 / 0.12 | 0.14 / 2.33 | 0.0076 | 0 | slower heavier |
| segnext-mscan-b-ade20k | fp32 | 85.5 / 87.1 | 0.98 | 0.11 / 2.09 | 528 / 416 | 1.27 | 106 / 106 | 119.5 / 120.8 | 0.18 / 0.12 | 0.14 / 2.13 | 4.2e-06 | 0 | heavier |
| segnext-mscan-l-ade20k | fp16 | 137.8 / 126.8 | 1.09 | 0.16 / 3.01 | 423 / 305 | 1.38 | 95 / 94 | 171.9 / 160.0 | 0.28 / 0.18 | 0.20 / 3.05 | 0.004 | 0.5 (0.0021 % off) | slower heavier |
| segnext-mscan-l-ade20k | fp32 | 153.5 / 158.2 | 0.97 | 0.18 / 2.83 | 610 / 497 | 1.23 | 188 / 187 | 187.2 / 191.4 | 0.28 / 0.18 | 0.22 / 2.87 | 5.5e-06 | 0 | heavier |
| segnext-mscan-s-ade20k | fp16 | 42.4 / 37.5 | 1.13 | 0.06 / 1.62 | 256 / 197 | 1.30 | 27 / 27 | 75.9 / 70.6 | 0.12 / 0.07 | 0.10 / 1.66 | 0.0022 | 0.12 (0.2 % off) | slower heavier |
| segnext-mscan-s-ade20k | fp32 | 47.0 / 46.8 | 1.00 | 0.07 / 1.58 | 471 / 363 | 1.30 | 53 / 53 | 80.4 / 80.4 | 0.11 / 0.07 | 0.10 / 1.62 | 3e-06 | 0 | heavier |
| segnext-mscan-t-ade20k | fp16 | 31.1 / 25.7 | 1.21 | 0.05 / 1.76 | 249 / 176 | 1.41 | 8 / 9 | 64.7 / 59.2 | 0.11 / 0.07 | 0.09 / 1.80 | 0.0021 | 0.92 (0.00052 % off) | slower heavier |
| segnext-mscan-t-ade20k | fp32 | 35.2 / 32.5 | 1.08 | 0.06 / 1.71 | 362 / 225 | 1.61 | 17 / 17 | 68.5 / 65.5 | 0.11 / 0.07 | 0.10 / 1.75 | 3.3e-06 | 0 | slower heavier |
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
| shvit_s1.in1k | fp16 | 3.3 / 3.0 | 1.11 | 0.02 / 1.20 | 16 / 20 | 0.80 | 12 / 12 | 4.1 / 3.5 | 0.79 / 0.71 | 0.03 / 1.21 | 0.0016 | 0.0016 | slower |
| shvit_s1.in1k | fp32 | 3.5 / 3.3 | 1.06 | 0.04 / 1.18 | 29 / 35 | 0.84 | 24 / 24 | 4.0 / 3.8 | 0.73 / 0.70 | 0.04 / 1.18 | 1.3e-06 | 1.3e-06 | slower |
| shvit_s2.in1k | fp16 | 3.5 / 3.4 | 1.06 | 0.03 / 1.22 | 26 / 31 | 0.83 | 22 / 22 | 4.0 / 3.9 | 0.78 / 0.69 | 0.04 / 1.22 | 0.0023 | 0.0023 | slower |
| shvit_s2.in1k | fp32 | 4.0 / 3.7 | 1.07 | 0.05 / 1.20 | 48 / 56 | 0.86 | 44 / 44 | 4.4 / 4.2 | 0.73 / 0.70 | 0.05 / 1.20 | 7.7e-07 | 7.7e-07 | slower |
| shvit_s3.in1k | fp16 | 4.1 / 3.9 | 1.07 | 0.03 / 1.27 | 33 / 41 | 0.82 | 28 / 28 | 4.6 / 4.4 | 0.79 / 0.71 | 0.03 / 1.28 | 0.0014 | 0.0014 | slower |
| shvit_s3.in1k | fp32 | 4.6 / 4.2 | 1.08 | 0.05 / 1.25 | 62 / 74 | 0.84 | 54 / 54 | 5.1 / 4.9 | 0.76 / 0.71 | 0.05 / 1.25 | 7.2e-07 | 7.2e-07 | slower |
| shvit_s4.in1k | fp16 | 4.9 / 4.7 | 1.04 | 0.04 / 1.46 | 40 / 52 | 0.78 | 32 / 32 | 5.5 / 5.3 | 0.81 / 0.71 | 0.04 / 1.46 | 0.002 | 0.002 | slower |
| shvit_s4.in1k | fp32 | 5.5 / 5.2 | 1.05 | 0.05 / 1.43 | 75 / 94 | 0.80 | 64 / 64 | 6.1 / 5.8 | 0.78 / 0.71 | 0.06 / 1.43 | 5e-07 | 5e-07 | slower |
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
| span-x2 | fp16 | 59.3 / 72.1 | 0.82 | 0.11 / 1.08 | 391 / 499 | 0.78 | 4 / 4 | 60.2 / 72.6 | 0.09 / 0.13 | 0.12 / 1.08 | 1 (0.14 % off) | 1 (0.46 % off) |  |
| span-x2 | fp32 | 113.4 / 149.7 | 0.76 | 0.15 / 1.14 | 661 / 613 | 1.08 | 2 / 2 | 114.3 / 150.0 | 0.09 / 0.13 | 0.15 / 1.14 | 0 | 0 | heavier |
| span-x2 | fp16 | 99.8 / 118.3 | 0.84 | 0.13 / 1.09 | 332 / 345 | 0.96 | 1 / 1 | 100.7 / 118.6 | 0.09 / 0.13 | 0.14 / 1.10 | 0.11 (0.021 % off) | 1 (0.15 % off) |  |
| span-x2 | fp32 | 113.4 / 149.2 | 0.76 | 0.15 / 1.11 | 661 / 613 | 1.08 | 2 / 2 | 114.4 / 150.1 | 0.09 / 0.12 | 0.15 / 1.12 | 0 | 0 | heavier |
| span-x4 | fp16 | 60.2 / 74.0 | 0.81 | 0.11 / 1.11 | 481 / 535 | 0.90 | 4 / 4 | 62.9 / 75.4 | 0.09 / 0.13 | 0.13 / 1.11 | 0.03 (0.00061 % off) | 0.036 (0.001 % off) |  |
| span-x4 | fp32 | 116.7 / 156.6 | 0.75 | 0.15 / 1.14 | 661 / 613 | 1.08 | 2 / 2 | 119.2 / 154.7 | 0.09 / 0.13 | 0.17 / 1.14 | 0 | 0 | heavier |
| span-x4 | fp16 | 103.1 / 122.4 | 0.84 | 0.13 / 1.10 | 380 / 380 | 1.00 | 1 / 1 | 105.6 / 124.0 | 0.09 / 0.13 | 0.15 / 1.11 | 0.0074 | 0.0088 |  |
| span-x4 | fp32 | 116.6 / 156.3 | 0.75 | 0.15 / 1.13 | 661 / 613 | 1.08 | 2 / 2 | 119.6 / 154.6 | 0.09 / 0.13 | 0.17 / 1.14 | 0 | 0 | heavier |
| squeezenet1_1.tv_in1k | fp16 | 1.7 / 1.9 | 0.88 | 0.01 / 0.44 | 16 / 18 | 0.86 | 2 / 2 | 2.1 / 2.2 | 0.53 / 0.52 | 0.02 / 0.44 | 0 | 0 |  |
| squeezenet1_1.tv_in1k | fp32 | 2.0 / 1.7 | 1.15 | 0.03 / 0.39 | 28 / 32 | 0.86 | 5 / 5 | 2.1 / 2.1 | 0.51 / 0.50 | 0.03 / 0.39 | 9.2e-08 | 9.2e-08 | slower |
| starnet_s1.in1k | fp16 | 2.1 / 2.6 | 0.81 | 0.02 / 0.66 | 12 / 24 | 0.51 | 6 / 6 | 2.5 / 3.1 | 0.77 / 0.69 | 0.02 / 0.66 | 0.0015 | 0.0015 |  |
| starnet_s1.in1k | fp32 | 2.3 / 3.3 | 0.68 | 0.03 / 0.66 | 24 / 43 | 0.55 | 11 / 11 | 2.7 / 3.6 | 0.71 / 0.71 | 0.03 / 0.66 | 1.6e-06 | 1.6e-06 |  |
| starnet_s2.in1k | fp16 | 2.0 / 2.3 | 0.87 | 0.01 / 0.57 | 14 / 24 | 0.59 | 7 / 7 | 2.3 / 2.8 | 0.74 / 0.71 | 0.02 / 0.57 | 0.00093 | 0.00093 |  |
| starnet_s2.in1k | fp32 | 2.0 / 2.6 | 0.78 | 0.03 / 0.56 | 27 / 44 | 0.62 | 14 / 14 | 2.5 / 3.1 | 0.72 / 0.69 | 0.04 / 0.56 | 6.2e-07 | 6.2e-07 |  |
| starnet_s3.in1k | fp16 | 2.3 / 3.0 | 0.78 | 0.02 / 0.69 | 19 / 34 | 0.55 | 11 / 11 | 2.8 / 3.5 | 0.78 / 0.69 | 0.02 / 0.69 | 0.00078 | 0.00078 |  |
| starnet_s3.in1k | fp32 | 2.7 / 3.4 | 0.78 | 0.04 / 0.66 | 37 / 63 | 0.58 | 22 / 22 | 3.1 / 3.8 | 0.71 / 0.69 | 0.04 / 0.67 | 3.8e-07 | 3.8e-07 |  |
| starnet_s4.in1k | fp16 | 3.0 / 3.9 | 0.77 | 0.02 / 0.85 | 26 / 46 | 0.56 | 14 / 14 | 3.5 / 4.5 | 0.79 / 0.71 | 0.02 / 0.85 | 0.00086 | 0.00086 |  |
| starnet_s4.in1k | fp32 | 3.5 / 4.3 | 0.80 | 0.04 / 0.83 | 48 / 87 | 0.56 | 29 / 29 | 4.0 / 4.8 | 0.73 / 0.71 | 0.04 / 0.84 | 4.2e-07 | 4.2e-07 |  |
| swin2sr-x2 | fp16 | 3432.4 / 3215.8 | 1.07 | 3.59 / 7.58 | 932 / 1277 | 0.73 | 33 / 92 | 3428.7 / 3203.6 | 0.67 / 0.25 | 3.60 / 7.58 | 0.0059 | 0.0064 | slower |
| swin2sr-x2 | fp32 | 4362.9 / 5434.3 | 0.80 | 4.56 / 9.73 | 1575 / 2069 | 0.76 | 60 / 116 | 4400.9 / 5461.8 | 0.70 / 0.27 | 4.56 / 9.74 | 2.7e-06 | 2.7e-06 |  |
| swin2sr-x4 | fp16 | 3641.9 / 3550.9 | 1.03 | 3.84 / 8.00 | 1022 / 2206 | 0.46 | 33 / 92 | 3597.5 / 3531.1 | 0.73 / 0.26 | 3.87 / 8.01 | 0.012 (5.4e-05 % off) | 0.014 (0.00021 % off) | slower |
| swin2sr-x4 | fp32 | 4605.0 / 5796.5 | 0.79 | 4.83 / 10.19 | 1576 / 4341 | 0.36 | 60 / 117 | 4627.1 / 6034.6 | 0.70 / 0.26 | 4.85 / 10.20 | 5.5e-06 | 6.4e-06 |  |
| swin2sr-x4-real | fp16 | 3675.9 / 3560.4 | 1.03 | 3.79 / 7.88 | 792 / 1415 | 0.56 | 33 / 92 | 3685.9 / 3569.8 | 0.70 / 0.26 | 3.81 / 7.89 | 0.0033 | 0.0035 | slower |
| swin2sr-x4-real | fp32 | 4722.6 / 6065.2 | 0.78 | 4.80 / 9.83 | 1412 / 2496 | 0.57 | 60 / 116 | 4717.6 / 5897.9 | 0.68 / 0.26 | 4.82 / 9.84 | 2.1e-06 | 2.2e-06 |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp16 | 18.4 / 18.6 | 0.99 | 0.06 / 1.95 | 221 / 263 | 0.84 | 173 / 170 | 18.9 / 19.2 | 0.88 / 0.71 | 0.06 / 1.95 | 0.0029 | 0.0029 |  |
| swin_base_patch4_window7_224.ms_in22k_ft_in1k | fp32 | 20.8 / 22.2 | 0.93 | 0.07 / 1.91 | 435 / 422 | 1.03 | 346 / 340 | 21.4 / 22.2 | 0.84 / 0.71 | 0.08 / 1.92 | 1.6e-06 | 1.6e-06 | heavier |
| swinir-l-x4-real | fp16 | 6624.4 / 6269.0 | 1.06 | 6.54 / 11.79 | 951 / 1450 | 0.66 | 68 / 127 | 6529.2 / 6235.3 | 0.75 / 0.45 | 6.56 / 11.80 | 0.034 (0.02 % off) | 0.04 (0.042 % off) | slower |
| swinir-l-x4-real | fp32 | 8269.7 / 10549.6 | 0.78 | 8.35 / 16.65 | 1889 / 2564 | 0.74 | 127 / 184 | 8344.2 / 10716.7 | 0.68 / 0.64 | 8.37 / 16.66 | 1.8e-05 | 2.1e-05 |  |
| swinir-m-x2 | fp16 | 2942.8 / 2742.2 | 1.07 | 3.04 / 6.79 | 932 / 1169 | 0.80 | 33 / 92 | 2932.8 / 2749.9 | 0.52 / 0.23 | 3.05 / 6.80 | 0.0065 | 0.006 | slower |
| swinir-m-x2 | fp32 | 3905.8 / 3942.6 | 0.99 | 4.09 / 8.09 | 1389 / 1642 | 0.85 | 59 / 116 | 3912.5 / 3993.5 | 0.51 / 0.23 | 4.10 / 8.09 | 2.6e-06 | 3e-06 |  |
| swinir-m-x4 | fp16 | 3081.4 / 3027.1 | 1.02 | 3.24 / 7.18 | 1021 / 2206 | 0.46 | 33 / 92 | 3140.9 / 3108.6 | 0.62 / 0.23 | 3.26 / 7.19 | 0.01 (7.7e-06 % off) | 0.012 (1.6e-05 % off) |  |
| swinir-m-x4 | fp32 | 4146.5 / 4299.2 | 0.96 | 4.39 / 8.54 | 1417 / 4341 | 0.33 | 59 / 117 | 4154.9 / 4277.7 | 0.67 / 0.23 | 4.41 / 8.55 | 4.8e-06 | 5.5e-06 |  |
| swinir-m-x4-real | fp16 | 3196.2 / 3140.7 | 1.02 | 3.29 / 7.08 | 699 / 1415 | 0.49 | 33 / 92 | 3192.9 / 3114.2 | 0.67 / 0.22 | 3.31 / 7.09 | 0.044 (0.011 % off) | 0.06 (0.036 % off) |  |
| swinir-m-x4-real | fp32 | 4243.3 / 4495.1 | 0.94 | 4.38 / 9.16 | 1389 / 2496 | 0.56 | 59 / 116 | 4283.3 / 4365.5 | 0.62 / 0.24 | 4.41 / 9.17 | 1.3e-05 | 1.8e-05 |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp16 | 4.5 / 4.7 | 0.97 | 0.03 / 0.94 | 44 / 64 | 0.68 | 25 / 25 | 5.1 / 5.2 | 0.83 / 0.74 | 0.03 / 0.94 | 0.0015 | 0.0015 |  |
| tiny_vit_11m_224.dist_in22k_ft_in1k | fp32 | 5.0 / 5.2 | 0.95 | 0.05 / 0.91 | 82 / 120 | 0.69 | 49 / 49 | 5.4 / 5.7 | 0.78 / 0.73 | 0.05 / 0.91 | 1.2e-06 | 1.2e-06 |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp16 | 6.6 / 6.8 | 0.98 | 0.04 / 0.98 | 72 / 104 | 0.69 | 46 / 46 | 7.0 / 7.2 | 0.84 / 0.74 | 0.04 / 0.98 | 0.0016 | 0.0016 |  |
| tiny_vit_21m_224.dist_in22k_ft_in1k | fp32 | 7.5 / 7.9 | 0.95 | 0.05 / 0.98 | 141 / 192 | 0.73 | 92 / 92 | 8.0 / 8.7 | 0.81 / 0.74 | 0.05 / 0.99 | 1e-06 | 1e-06 |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp16 | 14.4 / 14.3 | 1.01 | 0.05 / 0.97 | 162 / 184 | 0.88 | 88 / 88 | 14.8 / 14.6 | 1.12 / 1.02 | 0.05 / 0.98 | 0.001 | 0.001 |  |
| tiny_vit_21m_384.dist_in22k_ft_in1k | fp32 | 16.9 / 17.5 | 0.97 | 0.07 / 0.96 | 324 / 289 | 1.12 | 176 / 176 | 17.2 / 17.7 | 1.07 / 1.01 | 0.07 / 0.96 | 6.8e-07 | 6.8e-07 | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp16 | 25.2 / 25.1 | 1.00 | 0.06 / 1.01 | 333 / 300 | 1.11 | 191 / 191 | 25.4 / 25.5 | 1.83 / 1.69 | 0.06 / 1.01 | 0.0009 | 0.0009 | heavier |
| tiny_vit_21m_512.dist_in22k_ft_in1k | fp32 | 31.1 / 31.9 | 0.97 | 0.07 / 0.98 | 666 / 513 | 1.30 | 381 / 381 | 31.5 / 32.3 | 1.78 / 1.69 | 0.07 / 0.99 | 6.3e-07 | 6.3e-07 | heavier |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp16 | 3.9 / 4.1 | 0.95 | 0.04 / 0.92 | 33 / 43 | 0.76 | 13 / 13 | 4.5 / 4.6 | 0.81 / 0.73 | 0.04 / 0.92 | 0.0014 | 0.0014 |  |
| tiny_vit_5m_224.dist_in22k_ft_in1k | fp32 | 4.5 / 4.6 | 0.98 | 0.05 / 0.97 | 57 / 81 | 0.70 | 25 / 25 | 4.9 / 5.0 | 0.75 / 0.75 | 0.05 / 0.97 | 8.5e-07 | 8.5e-07 |  |
| totalsegmentator-total | fp16 | 14612.3 / 26280.3 | 0.56 | 14.55 / 33.05 | 3464 / 3912 | 0.89 | 567 / 298 | 17992.2 / 30890.6 | 1.49 / 0.28 | 18.41 / 37.53 | - | 1 (0.0048 % off) (1 keys differ) |  |
| totalsegmentator-total | fp32 | 21167.3 / 38793.7 | 0.55 | 21.18 / 42.58 | 3456 / 3954 | 0.87 | 595 / 595 | 25054.6 / 42912.5 | 0.07 / 0.26 | 25.89 / 47.26 | - | 0 (1 keys differ) |  |
| totalsegmentator-total-3mm | fp16 | 304.4 / 425.0 | 0.72 | 0.33 / 1.53 | 2058 / 1609 | 1.28 | 45 / 32 | 478.9 / 672.2 | 0.14 / 0.76 | 0.69 / 1.85 | 0.045 (0.0023 % off) | 1 (0.29 % off) | heavier |
| totalsegmentator-total-3mm | fp32 | 414.7 / 535.0 | 0.78 | 0.44 / 1.58 | 2033 / 1769 | 1.15 | 63 / 63 | 626.9 / 761.6 | 0.04 / 0.74 | 0.80 / 1.92 | 0.00012 | 0.74 (0.002 % off) | heavier |
| unik3d-vitl | fp16 | 889.6 / 897.5 | 0.99 | 0.92 / 3.63 | 1523 / 1166 | 1.31 | 715 / 696 | 894.8 / 904.5 | 1.46 / 0.07 | 0.93 / 3.64 | 0.009 | 0.016 (1.1 % off) | heavier |
| unik3d-vitl | fp32 | 1214.4 / 1191.7 | 1.02 | 1.36 / 3.91 | 3060 / 3117 | 0.98 | 1376 / 1386 | 1221.6 / 1197.9 | 1.36 / 0.07 | 1.37 / 3.92 | 5.1e-06 | 1.2e-05 |  |
| vgg16.tv_in1k | fp16 | 13.2 / 13.6 | 0.97 | 0.03 / 0.47 | 322 / 353 | 0.91 | 264 / 264 | 13.6 / 14.1 | 0.84 / 0.55 | 0.03 / 0.47 | 0 | 0 |  |
| vgg16.tv_in1k | fp32 | 16.7 / 17.5 | 0.96 | 0.04 / 0.50 | 643 / 705 | 0.91 | 528 / 528 | 17.3 / 18.0 | 0.79 / 0.57 | 0.04 / 0.50 | 0 | 0 |  |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp16 | 14.1 / 13.7 | 1.03 | 0.04 / 0.64 | 242 / 232 | 1.04 | 165 / 165 | 14.7 / 14.3 | 0.05 / 0.69 | 0.05 / 0.64 | 0.0011 | 0.0011 | slower heavier |
| vit_base_patch16_224.augreg2_in21k_ft_in1k | fp32 | 16.2 / 16.1 | 1.01 | 0.05 / 0.63 | 454 / 420 | 1.08 | 330 / 330 | 16.8 / 16.6 | 0.06 / 0.69 | 0.05 / 0.64 | 5.8e-07 | 5.8e-07 | heavier |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp16 | 180.7 / 178.6 | 1.01 | 0.21 / 1.47 | 912 / 895 | 1.02 | 817 / 820 | 181.6 / 179.5 | 0.08 / 0.73 | 0.21 / 1.47 | 0.00081 | 0.00081 |  |
| vit_so400m_patch14_siglip_378.webli_ft_in1k | fp32 | 214.2 / 209.3 | 1.02 | 0.24 / 1.50 | 1823 / 1796 | 1.02 | 1634 / 1639 | 214.0 / 209.9 | 0.13 / 0.73 | 0.25 / 1.50 | 5.2e-07 | 5.2e-07 | slower |
| vitpose-base | fp16 | 144.5 / 145.3 | 0.99 | 0.17 / 1.40 | 272 / 277 | 0.98 | 172 / 172 | 157.1 / 158.0 | 0.05 / 0.04 | 0.19 / 1.42 | 0.0051 | 0.025 (0.12 % off) |  |
| vitpose-base | fp32 | 160.6 / 164.4 | 0.98 | 0.20 / 1.37 | 540 / 469 | 1.15 | 343 / 343 | 173.7 / 177.2 | 0.06 / 0.04 | 0.22 / 1.40 | 4.8e-06 | 4.8e-06 | heavier |
| vitpose-base-simple | fp16 | 142.0 / 137.4 | 1.03 | 0.16 / 1.39 | 286 / 346 | 0.83 | 164 / 164 | 154.6 / 149.3 | 0.05 / 0.04 | 0.18 / 1.41 | 0.0038 | 0.15 (0.24 % off) | slower |
| vitpose-base-simple | fp32 | 156.4 / 158.1 | 0.99 | 0.19 / 1.40 | 556 / 1162 | 0.48 | 328 / 328 | 168.9 / 171.6 | 0.06 / 0.04 | 0.21 / 1.42 | 3.9e-06 | 2.9e-06 |  |
| vitpose-plus-base | fp16 | 144.6 / 144.6 | 1.00 | 0.17 / 1.40 | 272 / 277 | 0.98 | 172 / 172 | 157.2 / 156.8 | 0.05 / 0.06 | 0.19 / 1.42 | 0.0042 | 0.0085 |  |
| vitpose-plus-base | fp32 | 160.2 / 165.6 | 0.97 | 0.31 / 1.42 | 535 / 469 | 1.14 | 343 / 343 | 174.4 / 178.0 | 0.06 / 0.06 | 0.33 / 1.44 | 3e-06 | 5.2e-06 | heavier |
| vitpose-plus-huge | fp16 | 879.9 / 861.6 | 1.02 | 0.91 / 3.07 | 1372 / 1393 | 0.98 | 1215 / 1216 | 891.8 / 873.3 | 0.12 / 0.21 | 0.93 / 3.10 | 0.0056 | 0.11 (0.36 % off) | slower |
| vitpose-plus-huge | fp32 | 1000.3 / 981.7 | 1.02 | 1.23 / 4.20 | 2693 / 2777 | 0.97 | 2431 / 2431 | 1012.5 / 994.6 | 1.17 / 0.69 | 1.26 / 4.23 | 3.2e-06 | 2.8e-06 |  |
| vitpose-plus-large | fp16 | 444.0 / 438.2 | 1.01 | 0.47 / 2.11 | 715 / 693 | 1.03 | 588 / 588 | 457.0 / 451.3 | 0.08 / 0.12 | 0.50 / 2.13 | 0.0026 | 0.21 (0.24 % off) | heavier |
| vitpose-plus-large | fp32 | 494.0 / 486.0 | 1.02 | 0.53 / 2.15 | 1391 / 1512 | 0.92 | 1177 / 1177 | 506.1 / 499.8 | 0.11 / 0.11 | 0.55 / 2.17 | 2.9e-06 | 2.7e-06 |  |
| vitpose-plus-small | fp16 | 61.9 / 59.9 | 1.03 | 0.09 / 1.30 | 129 / 176 | 0.74 | 46 / 46 | 73.8 / 71.4 | 0.04 / 0.04 | 0.11 / 1.32 | 0.0032 | 0.053 (0.48 % off) | slower |
| vitpose-plus-small | fp32 | 70.1 / 69.5 | 1.01 | 0.11 / 1.29 | 250 / 249 | 1.01 | 93 / 93 | 82.5 / 81.4 | 0.05 / 0.04 | 0.13 / 1.31 | 2.7e-06 | 2.5e-06 |  |
| wd-convnext-tagger-v3 | fp16 | 45.9 / 43.8 | 1.05 | 0.06 / 1.06 | 266 / 328 | 0.81 | 189 / 189 | 46.2 / 44.3 | 0.96 / 0.71 | 0.06 / 1.07 | 0.00096 | 0.00096 | slower |
| wd-convnext-tagger-v3 | fp32 | 52.2 / 52.4 | 1.00 | 0.08 / 1.04 | 524 / 512 | 1.02 | 377 / 377 | 52.6 / 53.3 | 0.89 / 0.70 | 0.08 / 1.04 | 1e-06 | 1e-06 | heavier |
| wd-eva02-large-tagger-v3 | fp16 | 194.7 / 197.9 | 0.98 | 0.23 / 2.08 | 736 / 686 | 1.07 | 602 / 602 | 195.8 / 198.6 | 0.08 / 0.84 | 0.23 / 2.09 | 0.001 | 0.001 | heavier |
| wd-eva02-large-tagger-v3 | fp32 | 241.3 / 241.9 | 1.00 | 0.28 / 2.12 | 1468 / 1435 | 1.02 | 1203 / 1203 | 241.6 / 242.3 | 0.11 / 0.74 | 0.28 / 2.12 | 6.1e-07 | 6.1e-07 | heavier |
| wd-vit-large-tagger-v3 | fp16 | 178.9 / 175.4 | 1.02 | 0.21 / 1.21 | 696 / 683 | 1.02 | 601 / 601 | 179.4 / 175.9 | 0.07 / 0.71 | 0.21 / 1.21 | 0.0009 | 0.0009 |  |
| wd-vit-large-tagger-v3 | fp32 | 202.5 / 198.2 | 1.02 | 0.24 / 1.25 | 1387 / 1401 | 0.99 | 1202 / 1202 | 202.8 / 198.6 | 0.11 / 0.74 | 0.24 / 1.25 | 3.9e-07 | 3.9e-07 | slower |
| wd-vit-tagger-v3 | fp16 | 44.8 / 43.9 | 1.02 | 0.08 / 0.68 | 283 / 251 | 1.13 | 180 / 180 | 45.3 / 44.5 | 0.05 / 0.70 | 0.08 / 0.68 | 0.00088 | 0.00088 | heavier |
| wd-vit-tagger-v3 | fp32 | 53.8 / 53.9 | 1.00 | 0.08 / 0.69 | 572 / 444 | 1.29 | 361 / 361 | 54.2 / 54.2 | 0.06 / 0.69 | 0.09 / 0.70 | 2.7e-07 | 2.7e-07 | heavier |
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
| yunet-n | fp16 | 9.3 / 11.1 | 0.84 | 0.04 / 1.00 | 207 / 223 | 0.93 | 0 / 0 | 14.0 / 15.1 | 0.04 / 0.10 | 0.05 / 1.01 | 0.0031 | empty |  |
| yunet-n | fp32 | 12.0 / 14.7 | 0.81 | 0.04 / 0.95 | 367 / 271 | 1.35 | 0 / 0 | 17.0 / 18.6 | 0.04 / 0.10 | 0.05 / 0.96 | 1.1e-06 | empty | heavier |
| yunet-s | fp16 | 8.5 / 10.1 | 0.83 | 0.04 / 1.00 | 175 / 211 | 0.83 | 0 / 0 | 13.2 / 14.2 | 0.05 / 0.09 | 0.05 / 1.01 | 0.0021 | empty |  |
| yunet-s | fp32 | 10.5 / 13.1 | 0.80 | 0.04 / 0.95 | 303 / 271 | 1.12 | 0 / 0 | 15.3 / 16.9 | 0.04 / 0.09 | 0.05 / 0.95 | 1.1e-06 | empty | heavier |

Not measured (one arm failed):

- resplat-base-re10k float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-base-re10k float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float16: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
- resplat-small-dl3dv float32: graph: UnsupportedOp: no MLX mapping for fastforks.resplat_knn.default
