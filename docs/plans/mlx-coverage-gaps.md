# MLX coverage gaps (M1 Pro 16 GB)

Every ref whose model card declares an MLX port (`portable` or `kernels` dtypes) and that had no `ok` row in
`docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` on 2026-11-23, before this pass. Built from the registry
(`registry.cards`) and the survey files. s2m2-m / l / xl were measured in the same session and are no longer gaps.

**Kinds**

- **no-family-row**: no ref of the family has an MLX row. One representative per family gives the first number.
- **size-sibling**: another ref of the family is measured, but this one differs in its parameters or weight size
  (another size, input resolution or head). It needs its own row before auto routing can use it.
- **checkpoint-sibling**: same parameters and weight size (within 2 %) as a measured ref, only different weights
  (dataset fine-tunes). Speed carries over from the named twin, so these were not measured.

**This pass** (column "this pass"): `speed_survey.py run --device apple-m1-pro-mlx-16gb --modes portable`, one job
at a time, offline (cached weights only), 150 s per job. Times are warm medians at the survey size, with goldens
checked. "open" means not measured in this pass; the reason is in the list below the table.

The auto routes from these rows are in `src/fastforks/_core/apple_mlx.json` (`tools/gen_auto_mode.py`). A row only
yields an `auto` entry where an MPS `ok` row exists to compare against.

| task | family | ref | kind | portable dtypes | weights MB | MPS ok rows | this pass |
|---|---|---|---|---|---|---|---|
| classification | AlexNet | `alexnet.tv_in1k` | no-family-row | float32/float16 | 233 | 3 | fp16 3 ms, fp32 4 ms |
| classification | CoAtNet | `coatnet_0_rw_224.sw_in1k` | no-family-row | float32 | 105 | 3 | fp32 11 ms |
| classification | CoAtNet | `coatnet_1_rw_224.sw_in1k` | no-family-row | float32 | 159 | 3 | fp32 20 ms |
| classification | CoAtNet | `coatnet_2_rw_224.sw_in12k_ft_in1k` | no-family-row | float32 | 282 | 3 | fp32 30 ms |
| classification | CoAtNet | `coatnet_nano_rw_224.sw_in1k` | no-family-row | float32 | 58 | 3 | fp32 9 ms |
| classification | CoAtNet | `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k` | no-family-row | float32 | 159 | 3 | fp32 20 ms |
| classification | CoAtNet | `coatnet_rmlp_1_rw_224.sw_in1k` | no-family-row | float32 | 159 | 3 | fp32 19 ms |
| classification | CoAtNet | `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k` | no-family-row | float32 | 282 | 3 | fp32 30 ms |
| classification | CoAtNet | `coatnet_rmlp_2_rw_224.sw_in1k` | no-family-row | float32 | 282 | 3 | fp32 30 ms |
| classification | CoAtNet | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` | no-family-row | float32 | 282 | 3 | fp32 85 ms |
| classification | CoAtNet | `coatnet_rmlp_nano_rw_224.sw_in1k` | no-family-row | float32 | 58 | 3 | fp32 10 ms |
| classification | DaViT | `davit_base.msft_in1k` | no-family-row | float32/float16 | 336 | 3 | fp16 19 ms, fp32 23 ms |
| classification | DaViT | `davit_small.msft_in1k` | no-family-row | float32/float16 | 190 | 3 | fp16 14 ms, fp32 17 ms |
| classification | DaViT | `davit_tiny.msft_in1k` | no-family-row | float32/float16 | 108 | 3 | fp16 8 ms, fp32 10 ms |
| classification | DeiT | `deit_base_patch16_224.fb_in1k` | no-family-row | float32/float16 | 330 | 3 | fp16 14 ms, fp32 17 ms |
| classification | DenseNet | `densenet121.tv_in1k` | no-family-row | float32/float16 | 31 | 3 | fp16 10 ms, fp32 12 ms |
| classification | EVA-02 | `eva02_large_patch14_448.mim_m38m_ft_in22k_in1k` | no-family-row | float32/float16 | 1164 | 3 | fp16 199 ms, fp32 245 ms |
| classification | EfficientFormer | `efficientformer_l1.snap_dist_in1k` | no-family-row | float32/float16 | 47 | 3 | fp16 5 ms, fp32 6 ms |
| classification | EfficientFormer | `efficientformer_l3.snap_dist_in1k` | no-family-row | float32/float16 | 120 | 3 | fp16 9 ms, fp32 10 ms |
| classification | EfficientFormer | `efficientformer_l7.snap_dist_in1k` | no-family-row | float32/float16 | 314 | 3 | fp16 17 ms, fp32 19 ms |
| classification | EfficientFormerV2 | `efficientformerv2_l.snap_dist_in1k` | no-family-row | float32/float16 | 101 | 3 | fp16 12 ms, fp32 14 ms |
| classification | EfficientFormerV2 | `efficientformerv2_s0.snap_dist_in1k` | no-family-row | float32/float16 | 14 | 3 | fp16 4 ms, fp32 5 ms |
| classification | EfficientFormerV2 | `efficientformerv2_s1.snap_dist_in1k` | no-family-row | float32/float16 | 24 | 3 | fp16 6 ms, fp32 6 ms |
| classification | EfficientFormerV2 | `efficientformerv2_s2.snap_dist_in1k` | no-family-row | float32/float16 | 49 | 3 | fp16 9 ms, fp32 9 ms |
| classification | EfficientViT (MIT) | `efficientvit_b2.r288_in1k` | no-family-row | float32/float16 | 93 | 3 | fp16 10 ms, fp32 14 ms |
| classification | Falconsai NSFW ViT | `falconsai-nsfw-vit` | no-family-row | float32/float16 | 327 | 3 | fp16 14 ms, fp32 16 ms |
| classification | FastViT | `fastvit_sa24.apple_dist_in1k` | no-family-row | float32 | 83 | 3 | fp32 13 ms |
| classification | FastViT | `fastvit_t12.apple_dist_in1k` | no-family-row | float32 | 29 | 3 | fp32 7 ms |
| classification | FastViT | `fastvit_t8.apple_dist_in1k` | no-family-row | float32 | 16 | 3 | fp32 5 ms |
| classification | FocalNet | `focalnet_base_lrf.ms_in1k` | no-family-row | float32/float16 | 339 | 3 | fp16 23 ms, fp32 31 ms |
| classification | FocalNet | `focalnet_base_srf.ms_in1k` | no-family-row | float32/float16 | 336 | 3 | fp16 22 ms, fp32 28 ms |
| classification | FocalNet | `focalnet_small_lrf.ms_in1k` | no-family-row | float32/float16 | 192 | 3 | fp16 18 ms, fp32 21 ms |
| classification | FocalNet | `focalnet_small_srf.ms_in1k` | no-family-row | float32/float16 | 190 | 3 | fp16 17 ms, fp32 20 ms |
| classification | FocalNet | `focalnet_tiny_lrf.ms_in1k` | no-family-row | float32/float16 | 109 | 3 | fp16 10 ms, fp32 12 ms |
| classification | FocalNet | `focalnet_tiny_srf.ms_in1k` | no-family-row | float32/float16 | 108 | 3 | fp16 10 ms, fp32 11 ms |
| classification | HGNetV2 | `hgnetv2_b0.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 23 | 3 | fp16 3 ms, fp32 3 ms |
| classification | HGNetV2 | `hgnetv2_b1.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 24 | 3 | fp16 3 ms, fp32 4 ms |
| classification | HGNetV2 | `hgnetv2_b2.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 43 | 3 | fp16 5 ms, fp32 5 ms |
| classification | HGNetV2 | `hgnetv2_b3.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 62 | 3 | fp16 8 ms, fp32 7 ms |
| classification | HGNetV2 | `hgnetv2_b4.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 76 | 3 | fp16 6 ms, fp32 8 ms |
| classification | HGNetV2 | `hgnetv2_b5.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 151 | 3 | fp16 11 ms, fp32 13 ms |
| classification | HGNetV2 | `hgnetv2_b6.ssld_stage2_ft_in1k` | no-family-row | float32/float16 | 288 | 3 | fp16 22 ms, fp32 27 ms |
| classification | Inception-v3 | `inception_v3.tv_in1k` | no-family-row | float32/float16 | 104 | 3 | fp16 11 ms, fp32 13 ms |
| classification | MobileNetV4 | `mobilenetv4_conv_medium.e500_r256_in1k` | no-family-row | float32/float16 | 37 | 3 | fp16 5 ms, fp32 6 ms |
| classification | SHViT | `shvit_s1.in1k` | no-family-row | float32/float16 | 24 | 3 | fp16 4 ms, fp32 5 ms |
| classification | SHViT | `shvit_s2.in1k` | no-family-row | float32/float16 | 44 | 3 | fp16 5 ms, fp32 6 ms |
| classification | SHViT | `shvit_s3.in1k` | no-family-row | float32/float16 | 55 | 3 | fp16 6 ms, fp32 7 ms |
| classification | SHViT | `shvit_s4.in1k` | no-family-row | float32/float16 | 64 | 3 | fp16 8 ms, fp32 8 ms |
| classification | ShuffleNetV2 | `shufflenet_v2_x1_0.tv_in1k` | no-family-row | float32/float16 | 9 | 3 | fp16 3 ms, fp32 3 ms |
| classification | SigLIP (fine-tuned) | `vit_so400m_patch14_siglip_378.webli_ft_in1k` | no-family-row | float32/float16 | 1638 | 3 | open: 1638 MB weights, skipped (download and memory cost on the shared Mac) |
| classification | SqueezeNet | `squeezenet1_1.tv_in1k` | no-family-row | float32/float16 | 5 | 3 | fp16 2 ms, fp32 2 ms |
| classification | Swin Transformer | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` | no-family-row | float32/float16 | 336 | 3 | fp16 19 ms, fp32 22 ms |
| classification | TinyViT | `tiny_vit_11m_224.dist_in22k_ft_in1k` | no-family-row | float32/float16 | 42 | 3 | fp16 5 ms, fp32 6 ms |
| classification | TinyViT | `tiny_vit_21m_224.dist_in22k_ft_in1k` | no-family-row | float32/float16 | 81 | 3 | fp16 7 ms, fp32 9 ms |
| classification | TinyViT | `tiny_vit_21m_384.dist_in22k_ft_in1k` | no-family-row | float32/float16 | 81 | 3 | fp16 15 ms, fp32 20 ms |
| classification | TinyViT | `tiny_vit_21m_512.dist_in22k_ft_in1k` | no-family-row | float32/float16 | 81 | 3 | fp16 27 ms, fp32 40 ms |
| classification | TinyViT | `tiny_vit_5m_224.dist_in22k_ft_in1k` | no-family-row | float32/float16 | 21 | 3 | fp16 5 ms, fp32 5 ms |
| classification | VGG | `vgg16.tv_in1k` | no-family-row | float32/float16 | 528 | 3 | fp16 14 ms, fp32 17 ms |
| classification | Vision Transformer | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | no-family-row | float32/float16 | 330 | 3 | fp16 14 ms, fp32 17 ms |
| classification | WD Tagger v3 | `wd-convnext-tagger-v3` | no-family-row | float32 | 377 | 3 | fp32 55 ms |
| classification | WD Tagger v3 | `wd-eva02-large-tagger-v3` | no-family-row | float32 | 1202 | 3 | fp32 244 ms |
| classification | WD Tagger v3 | `wd-vit-large-tagger-v3` | no-family-row | float32 | 1202 | 3 | fp32 200 ms |
| classification | WD Tagger v3 | `wd-vit-tagger-v3` | no-family-row | float32 | 361 | 3 | fp32 54 ms |
| depth | DepthFM | `depthfm-v1` | no-family-row | float32/float16 | 3623 | 0 | fp16 failed, fp32 failed; open: earlier MLX rows fail, a 71 GB buffer at the survey size (MPS OOM too) |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `deim-l-coco` | size-sibling | float32/float16 | 120 | 3 | fp16 60 ms, fp32 74 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `deim-m-coco` | size-sibling | float32/float16 | 75 | 3 | fp16 47 ms, fp32 58 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `deim-n-coco` | size-sibling | float32/float16 | 15 | 3 | fp16 21 ms, fp32 23 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `deim-s-coco` | size-sibling | float32/float16 | 40 | 3 | fp16 31 ms, fp32 36 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `deim-x-coco` | size-sibling | float32/float16 | 241 | 3 | fp16 95 ms, fp32 120 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `dfine-l-coco` | size-sibling | float32/float16 | 120 | 3 | fp16 60 ms, fp32 74 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `dfine-m-coco` | size-sibling | float32/float16 | 75 | 3 | fp16 47 ms, fp32 58 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `dfine-s-coco` | size-sibling | float32/float16 | 40 | 3 | fp16 30 ms, fp32 37 ms |
| detection | D-FINE (HGNetV2, N / S / M / L / X) | `dfine-x-coco` | size-sibling | float32/float16 | 240 | 3 | fp16 95 ms, fp32 120 ms |
| detection | DETR (ResNet-50 / ResNet-101) | `detr-r101-coco` | size-sibling | float32/float16 | 232 | 3 | fp16 144 ms, fp32 181 ms |
| detection | LW-DETR (ViT, tiny / small / medium / large / xlarge) | `lw-detr-large-coco` | size-sibling | float32/float16 | 179 | 3 | fp16 65 ms, fp32 71 ms |
| detection | LW-DETR (ViT, tiny / small / medium / large / xlarge) | `lw-detr-medium-coco` | size-sibling | float32/float16 | 108 | 3 | fp16 48 ms, fp32 47 ms |
| detection | LW-DETR (ViT, tiny / small / medium / large / xlarge) | `lw-detr-small-coco` | size-sibling | float32/float16 | 56 | 3 | fp16 34 ms, fp32 33 ms |
| detection | LW-DETR (ViT, tiny / small / medium / large / xlarge) | `lw-detr-xlarge-coco` | size-sibling | float32/float16 | 450 | 3 | fp16 117 ms, fp32 123 ms |
| detection | OWLv2 (open-vocabulary, CLIP B/16 and L/14) | `owlv2-l14` | size-sibling | float32/float16 | 1671 | 2 | open: 1671 MB weights, skipped (download and memory cost on the shared Mac) |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-l-coco` | size-sibling | float32 | 130 | 3 | fp32 56 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-m-coco` | size-sibling | float32 | 386 | 3 | fp32 40 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-s-coco` | size-sibling | float32 | 368 | 3 | fp32 33 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-2xl-coco` | size-sibling | float32 | 148 | 3 | fp32 186 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-l-coco` | size-sibling | float32 | 138 | 3 | fp32 65 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-m-coco` | size-sibling | float32 | 136 | 3 | fp32 52 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-n-coco` | size-sibling | float32 | 128 | 3 | fp32 28 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-s-coco` | size-sibling | float32 | 129 | 3 | fp32 39 ms |
| detection | RF-DETR (N / S / M / L detection, Seg N to 2XL) | `rf-detr-seg-xl-coco` | size-sibling | float32 | 146 | 3 | fp32 115 ms |
| detection | RT-DETRv2 (ResNet-vd, S / M / M* / L / X) | `rtdetrv2-l-coco` | size-sibling | float32/float16 | 165 | 3 | fp16 68 ms, fp32 89 ms |
| detection | RT-DETRv2 (ResNet-vd, S / M / M* / L / X) | `rtdetrv2-m-coco` | size-sibling | float32/float16 | 120 | 3 | fp16 50 ms, fp32 65 ms |
| detection | RT-DETRv2 (ResNet-vd, S / M / M* / L / X) | `rtdetrv2-ms-coco` | size-sibling | float32/float16 | 140 | 3 | fp16 57 ms, fp32 72 ms |
| detection | RT-DETRv2 (ResNet-vd, S / M / M* / L / X) | `rtdetrv2-x-coco` | size-sibling | float32/float16 | 293 | 3 | fp16 118 ms, fp32 139 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-l-coco` | size-sibling | float32/float16 | 433 | 3 | fp16 75 ms, fp32 99 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-m-coco` | size-sibling | float32/float16 | 214 | 3 | fp16 50 ms, fp32 60 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-m-person` | size-sibling | float32/float16 | 94 | 3 | fp16 48 ms, fp32 60 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-nano-person` | size-sibling | float32/float16 | 4 | 3 | fp16 9 ms, fp32 9 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-s-coco` | size-sibling | float32/float16 | 87 | 3 | fp16 26 ms, fp32 34 ms |
| detection | RTMDet (CSPNeXt, tiny / S / M / L / X; nano and M person detectors) | `rtmdet-x-coco` | size-sibling | float32/float16 | 377 | 3 | fp16 126 ms, fp32 163 ms |
| detection | YOLOX (tiny / S / M / L / X) | `yolox-l` | size-sibling | float32/float16 | 414 | 3 | fp16 70 ms, fp32 93 ms |
| detection | YOLOX (tiny / S / M / L / X) | `yolox-m` | size-sibling | float32/float16 | 194 | 3 | fp16 45 ms, fp32 52 ms |
| detection | YOLOX (tiny / S / M / L / X) | `yolox-s` | size-sibling | float32/float16 | 69 | 3 | fp16 23 ms, fp32 29 ms |
| detection | YOLOX (tiny / S / M / L / X) | `yolox-x` | size-sibling | float32/float16 | 757 | 3 | fp16 121 ms, fp32 152 ms |
| detection | detectron2 Mask / Faster R-CNN (R50-FPN) | `faster-rcnn-r50-fpn-3x-coco` | size-sibling | float32/float16 | 160 | 3 | fp16 367 ms, fp32 410 ms |
| edges | Line art (informative-drawings) | `lineart-coarse` | checkpoint-sibling | float32/float16 | 16 | 3 | skip: checkpoint of `lineart` |
| features | C-RADIOv3 / v4 | `c-radio-v4-h` | size-sibling | float32/float16 | 2486 | 3 | open: 2486 MB weights, skipped (download and memory cost on the shared Mac) |
| features | C-RADIOv3 / v4 | `c-radio-v4-so400m` | size-sibling | float32/float16 | 1645 | 3 | open: 1645 MB weights, skipped (download and memory cost on the shared Mac) |
| features | CLIP | `clip-b16-laion2b` | size-sibling | float32/float16 | 329 | 3 | fp16 14 ms, fp32 16 ms |
| features | CLIP | `clip-h14-378-dfn5b` | size-sibling | float32/float16 | 2414 | 3 | open: 2414 MB weights, skipped (download and memory cost on the shared Mac) |
| features | CLIP | `clip-h14-dfn5b` | size-sibling | float32/float16 | 2411 | 3 | open: 2411 MB weights, skipped (download and memory cost on the shared Mac) |
| features | CLIP | `clip-h14-laion2b` | size-sibling | float32/float16 | 2411 | 3 | open: 2411 MB weights, skipped (download and memory cost on the shared Mac) |
| features | CLIP | `clip-l14-336-openai` | size-sibling | float32/float16 | 1161 | 3 | fp16 54 ms, fp32 63 ms |
| features | CLIP | `clip-l14-datacompxl` | size-sibling | float32/float16 | 1160 | 3 | fp16 55 ms, fp32 63 ms |
| features | CLIP | `clip-l14-dfn2b` | size-sibling | float32/float16 | 1160 | 3 | fp16 54 ms, fp32 63 ms |
| features | CLIP | `clip-l14-openai` | size-sibling | float32/float16 | 1631 | 3 | open: 1631 MB weights, skipped (download and memory cost on the shared Mac) |
| features | Perception Encoder | `pe-core-b16-224` | size-sibling | float32/float16 | 357 | 4 | fp16 23 ms, fp32 18 ms |
| features | Perception Encoder | `pe-core-g14-448` | size-sibling | float32/float16 | 7179 | 4 | open: 7179 MB weights, too big for 16 GB |
| features | Perception Encoder | `pe-core-s16-384` | size-sibling | float32/float16 | 91 | 4 | fp16 10 ms, fp32 8 ms |
| features | Perception Encoder | `pe-spatial-b16-512` | size-sibling | float32/float16 | 330 | 4 | fp16 22 ms, fp32 18 ms |
| features | Perception Encoder | `pe-spatial-g14-448` | size-sibling | float32/float16 | 7064 | 4 | open: 7064 MB weights, too big for 16 GB |
| features | Perception Encoder | `pe-spatial-l14-448` | size-sibling | float32/float16 | 1160 | 4 | fp16 80 ms, fp32 66 ms |
| features | Perception Encoder | `pe-spatial-s16-512` | size-sibling | float32/float16 | 84 | 4 | fp16 8 ms, fp32 7 ms |
| features | Perception Encoder | `pe-spatial-t16-512` | size-sibling | float32/float16 | 22 | 4 | fp16 4 ms, fp32 4 ms |
| features | SigLIP 2 | `siglip2-b16-512` | size-sibling | float32/float16 | 357 | 3 | fp16 15 ms, fp32 17 ms |
| features | SigLIP 2 | `siglip2-g16-256` | size-sibling | float32/float16 | 4437 | 3 | open: 4437 MB weights, skipped (download and memory cost on the shared Mac) |
| features | SigLIP 2 | `siglip2-g16-384` | size-sibling | float32/float16 | 4439 | 3 | open: 4439 MB weights, skipped (download and memory cost on the shared Mac) |
| features | SigLIP 2 | `siglip2-l16-256` | size-sibling | float32/float16 | 1205 | 3 | fp16 44 ms, fp32 52 ms |
| features | SigLIP 2 | `siglip2-l16-512` | size-sibling | float32/float16 | 1208 | 3 | fp16 45 ms, fp32 52 ms |
| features | SigLIP 2 | `siglip2-so400m14-224` | size-sibling | float32/float16 | 1632 | 3 | open: 1632 MB weights, skipped (download and memory cost on the shared Mac) |
| features | SigLIP 2 | `siglip2-so400m16-256` | size-sibling | float32/float16 | 1632 | 3 | open: 1632 MB weights, skipped (download and memory cost on the shared Mac) |
| features | SigLIP 2 | `siglip2-so400m16-512` | size-sibling | float32/float16 | 1636 | 3 | open: 1636 MB weights, skipped (download and memory cost on the shared Mac) |
| flow | FlowFormer Small | `flowformer-small-chairs` | no-family-row | float32/float16 | 24 | 0 | fp16 13.0 s, fp32 19.8 s (re-run 2026-11-23 after the memory-lean MLX port, peak about 1.9 GB, goldens pass) |
| flow | FlowFormer Small | `flowformer-small-sintel` | no-family-row | float32/float16 | 24 | 0 | fp16 13.0 s, fp32 19.8 s (memory-lean MLX port, peak about 1.9 GB, goldens pass) |
| flow | FlowFormer Small | `flowformer-small-things` | no-family-row | float32/float16 | 24 | 0 | fp16 13.0 s, fp32 19.9 s (memory-lean MLX port, peak about 1.9 GB, goldens pass) |
| flow | FlowSeek | `flowseek-t-things` | checkpoint-sibling | float32/float16 | 163 | 0 | skip: checkpoint of `flowseek-t` |
| flow | FlowSeek | `flowseek-t-things-notartan` | checkpoint-sibling | float32/float16 | 163 | 1 | skip: checkpoint of `flowseek-t` |
| flow | GMFlow | `gmflow-scale1-mixdata` | checkpoint-sibling | float32/float16 | 18 | 3 | skip: checkpoint of `gmflow-scale1-things` |
| flow | GMFlow | `gmflow-scale2-mixdata` | size-sibling | float32/float16 | 18 | 0 | fp16 8276 ms, fp32 12373 ms |
| flow | GMFlow | `gmflow-scale2-regrefine6-kitti` | size-sibling | float32/float16 | 28 | 0 | fp16 11260 ms; fp32 re-run 2026-11-23: ~16.7 s, golden pass (MPS runs out of memory in both dtypes, so auto takes MLX) |
| flow | GMFlow | `gmflow-scale2-regrefine6-mixdata` | size-sibling | float32/float16 | 28 | 0 | fp16 11199 ms; fp32 re-run 2026-11-23: ~16.7 s, golden pass (MPS runs out of memory in both dtypes, so auto takes MLX) |
| flow | GMFlow | `gmflow-scale2-regrefine6-sintelft` | size-sibling | float32/float16 | 28 | 0 | fp16 11340 ms; fp32 re-run 2026-11-23: ~16.7 s, golden pass (MPS runs out of memory in both dtypes, so auto takes MLX) |
| flow | GMFlow | `gmflow-scale2-regrefine6-things` | size-sibling | float32/float16 | 28 | 0 | fp16 11121 ms; fp32 re-run 2026-11-23: ~16.7 s, golden pass (MPS runs out of memory in both dtypes, so auto takes MLX) |
| flow | GMFlow | `gmflow-scale2-sintel` | size-sibling | float32/float16 | 18 | 0 | fp16 8283 ms, fp32 13073 ms |
| flow | GMFlow | `gmflow-scale2-things` | size-sibling | float32/float16 | 18 | 0 | fp16 8261 ms, fp32 13356 ms |
| flow | MEMFOF | `memfof-kitti` | size-sibling | float32/float16 | 289 | 2 | fp16 16.1 s, fp32 19.7 s (memory-lean MLX port, peak 3.7 / 4.1 GB, goldens pass) |
| flow | MEMFOF | `memfof-sintel` | size-sibling | float32/float16 | 289 | 0 | fp16 16.0 s, fp32 19.9 s (memory-lean MLX port, peak 3.6 / 4.0 GB, goldens pass) |
| flow | MEMFOF | `memfof-spring` | checkpoint-sibling | float32/float16 | 289 | 1 | skip: checkpoint of `memfof` |
| flow | MemFlow | `memflow-kitti` | size-sibling | float32/float16 | 24 | 1 | fp16 6988 ms, fp32 9851 ms |
| flow | MemFlow | `memflow-sintel` | size-sibling | float32/float16 | 24 | 1 | fp16 6990 ms, fp32 9860 ms |
| flow | NeuFlow v2 | `neuflow-v2-sintel` | checkpoint-sibling | float32 | 35 | 2 | skip: checkpoint of `neuflow-v2` |
| flow | NeuFlow v2 | `neuflow-v2-things` | checkpoint-sibling | float32 | 35 | 2 | skip: checkpoint of `neuflow-v2` |
| flow | RAFT | `raft-large-kitti` | checkpoint-sibling | float32/float16 | 20 | 3 | skip: checkpoint of `raft-large` |
| flow | RAFT | `raft-large-things` | checkpoint-sibling | float32/float16 | 20 | 3 | skip: checkpoint of `raft-large` |
| flow | RAFT | `raft-small` | size-sibling | float32/float16 | 4 | 3 | fp16 927 ms, fp32 1020 ms |
| flow | ReCoVEr | `recover-mn` | size-sibling | float32 | 56 | 3 | fp32 467 ms |
| flow | ReCoVEr | `recover-rn` | size-sibling | float32 | 75 | 3 | fp32 637 ms |
| flow | SEA-RAFT | `sea-raft-m-kitti` | checkpoint-sibling | float32/float16 | 75 | 3 | skip: checkpoint of `sea-raft-m` |
| flow | SEA-RAFT | `sea-raft-m-spring` | size-sibling | float32/float16 | 75 | 4 | fp16 298 ms, fp32 369 ms |
| flow | SEA-RAFT | `sea-raft-m-things` | checkpoint-sibling | float32/float16 | 75 | 4 | skip: checkpoint of `sea-raft-m` |
| flow | SEA-RAFT | `sea-raft-s` | size-sibling | float32/float16 | 34 | 4 | fp16 1069 ms, fp32 1584 ms |
| flow | SEA-RAFT | `sea-raft-s-kitti` | size-sibling | float32/float16 | 34 | 4 | fp16 1083 ms, fp32 1585 ms |
| flow | SEA-RAFT | `sea-raft-s-spring` | size-sibling | float32/float16 | 34 | 4 | fp16 198 ms, fp32 251 ms |
| flow | WAFT | `waft-a1-kitti` | checkpoint-sibling | float32/float16 | 245 | 2 | skip: checkpoint of `waft-a1` |
| flow | WAFT | `waft-a1-sintel` | checkpoint-sibling | float32/float16 | 245 | 2 | skip: checkpoint of `waft-a1` |
| interpolation | EMA-VFI | `ema-vfi` | size-sibling | float32/float16 | 252 | 3 | fp16 2255 ms, fp32 3362 ms |
| interpolation | EMA-VFI | `ema-vfi-small` | checkpoint-sibling | float32/float16 | 56 | 3 | skip: checkpoint of `ema-vfi-small-t` |
| interpolation | EMA-VFI | `ema-vfi-t` | size-sibling | float32/float16 | 252 | 3 | fp16 2222 ms, fp32 3305 ms |
| interpolation | GIMM-VFI | `gimm-vfi-r-p` | checkpoint-sibling | float32/float16 | 76 | 0 | skip: checkpoint of `gimm-vfi-r` |
| interpolation | RIFE | `rife-4.26` | checkpoint-sibling | float32/float16 | 23 | 3 | skip: checkpoint of `rife-4.25` |
| matching | LoFTR | `loftr-indoor` | size-sibling | float32 | 44 | 3 | fp32 1792 ms |
| matching | LoMa | `loma-b128` | size-sibling | float32/float16 | 122 | 0 | fp16 1085 ms, fp32 1546 ms |
| matching | LoMa | `loma-b64` | size-sibling | float32/float16 | 49 | 0 | fp16 828 ms, fp32 1228 ms |
| matching | LoMa | `loma-g` | size-sibling | float32/float16 | 1399 | 0 | open: 1399 MB weights, skipped (download and memory cost on the shared Mac) |
| matching | LoMa | `loma-l` | size-sibling | float32/float16 | 858 | 0 | fp16 2810 ms, fp32 3706 ms |
| matching | LoMa | `loma-r` | checkpoint-sibling | float32/float16 | 723 | 0 | skip: checkpoint of `loma-b` |
| matching | XFeat | `xfeat-lighterglue` | size-sibling | float32 | 16 | 2 | fp32 524 ms |
| matting | BiRefNet | `birefnet-lite` | size-sibling | float32/float16 | 169 | 3 | fp16 392 ms, fp32 640 ms |
| matting | BiRefNet | `rmbg-2.0` | size-sibling | float32/float16 | 844 | 2 | fp16 1050 ms, fp32 1643 ms |
| matting | IS-Net | `isnet` | checkpoint-sibling | float32/float16 | 168 | 3 | skip: checkpoint of `isnet-general-use` |
| matting | U²-Net | `u2net-human-seg` | checkpoint-sibling | float32/float16 | 168 | 3 | skip: checkpoint of `u2net` |
| matting | U²-Net | `u2netp` | size-sibling | float32/float16 | 4 | 3 | fp16 28 ms, fp32 35 ms |
| multiview | ReSplat | `resplat-small-dl3dv` | size-sibling | float32/float16 | 293 | 1 | fp16 2113 ms, fp32 2804 ms |
| ocr | PARSeq (scene text recognition) | `parseq-tiny` | size-sibling | float32/float16 | 23 | 2 | fp16 7 ms, fp32 8 ms |
| pose | HRNet | `hrnet-w32-384` | size-sibling | float32/float16 | 116 | 3 | fp16 237 ms, fp32 327 ms |
| pose | HRNet | `hrnet-w48` | size-sibling | float32/float16 | 257 | 3 | fp16 222 ms, fp32 285 ms |
| pose | HRNet | `hrnet-w48-384` | size-sibling | float32/float16 | 255 | 3 | fp16 449 ms, fp32 597 ms |
| pose | RTMO | `rtmo-l` | size-sibling | float32 | 171 | 3 | fp32 84 ms |
| pose | RTMO | `rtmo-m` | size-sibling | float32 | 86 | 3 | fp32 51 ms |
| pose | RTMO | `rtmo-s` | size-sibling | float32 | 38 | 3 | fp32 30 ms |
| pose | RTMPose | `dwpose-l` | size-sibling | float32/float16 | 128 | 4 | fp16 81 ms, fp32 105 ms |
| pose | RTMPose | `dwpose-l-384` | size-sibling | float32/float16 | 128 | 4 | fp16 165 ms, fp32 217 ms |
| pose | RTMPose | `dwpose-m` | size-sibling | float32/float16 | 69 | 4 | fp16 54 ms, fp32 70 ms |
| pose | RTMPose | `dwpose-s` | size-sibling | float32/float16 | 32 | 4 | fp16 34 ms, fp32 40 ms |
| pose | RTMPose | `dwpose-t` | size-sibling | float32/float16 | 21 | 4 | fp16 29 ms, fp32 39 ms |
| pose | RTMPose | `rtmpose-l` | size-sibling | float32/float16 | 106 | 4 | fp16 75 ms, fp32 98 ms |
| pose | RTMPose | `rtmpose-l-384` | size-sibling | float32/float16 | 106 | 4 | fp16 150 ms, fp32 205 ms |
| pose | RTMPose | `rtmpose-l-halpe26` | size-sibling | float32/float16 | 108 | 4 | fp16 75 ms, fp32 99 ms |
| pose | RTMPose | `rtmpose-l-halpe26-384` | size-sibling | float32/float16 | 108 | 4 | fp16 151 ms, fp32 205 ms |
| pose | RTMPose | `rtmpose-m` | size-sibling | float32/float16 | 52 | 4 | fp16 51 ms, fp32 64 ms |
| pose | RTMPose | `rtmpose-m-384` | size-sibling | float32/float16 | 53 | 4 | fp16 108 ms, fp32 142 ms |
| pose | RTMPose | `rtmpose-m-ap10k` | size-sibling | float32/float16 | 52 | 4 | fp16 65 ms, fp32 85 ms |
| pose | RTMPose | `rtmpose-m-face6` | size-sibling | float32/float16 | 65 | 4 | fp16 68 ms, fp32 88 ms |
| pose | RTMPose | `rtmpose-m-halpe26` | size-sibling | float32/float16 | 53 | 4 | fp16 51 ms, fp32 65 ms |
| pose | RTMPose | `rtmpose-m-halpe26-384` | size-sibling | float32/float16 | 54 | 4 | fp16 106 ms, fp32 143 ms |
| pose | RTMPose | `rtmpose-m-hand5` | size-sibling | float32/float16 | 53 | 4 | fp16 65 ms, fp32 85 ms |
| pose | RTMPose | `rtmpose-s` | size-sibling | float32/float16 | 21 | 4 | fp16 30 ms, fp32 34 ms |
| pose | RTMPose | `rtmpose-s-face6` | size-sibling | float32/float16 | 30 | 4 | fp16 40 ms, fp32 45 ms |
| pose | RTMPose | `rtmpose-s-halpe26` | size-sibling | float32/float16 | 22 | 4 | fp16 31 ms, fp32 41 ms |
| pose | RTMPose | `rtmpose-t-face6` | size-sibling | float32/float16 | 19 | 4 | fp16 33 ms, fp32 38 ms |
| pose | RTMPose | `rtmpose-t-halpe26` | size-sibling | float32/float16 | 13 | 4 | fp16 25 ms, fp32 34 ms |
| pose | RTMPose | `rtmpose-x-halpe26-384` | size-sibling | float32/float16 | 191 | 4 | fp16 257 ms, fp32 367 ms |
| pose | RTMPose | `rtmw-l` | size-sibling | float32/float16 | 219 | 4 | fp16 128 ms, fp32 157 ms |
| pose | RTMPose | `rtmw-l-384` | size-sibling | float32/float16 | 220 | 4 | fp16 280 ms, fp32 348 ms |
| pose | RTMPose | `rtmw-m` | size-sibling | float32/float16 | 124 | 4 | fp16 87 ms, fp32 110 ms |
| pose | RTMPose | `rtmw-x` | size-sibling | float32/float16 | 352 | 4 | fp16 236 ms, fp32 293 ms |
| pose | RTMPose | `rtmw-x-384` | size-sibling | float32/float16 | 353 | 4 | fp16 472 ms, fp32 613 ms |
| pose | RetinaFace | `retinaface-mobilenet0.25` | size-sibling | float32/float16 | 2 | 3 | fp16 20 ms, fp32 24 ms |
| pose | ViTPose / ViTPose++ | `vitpose-base` | size-sibling | float32/float16 | 343 | 4 | fp16 158 ms, fp32 178 ms |
| pose | ViTPose / ViTPose++ | `vitpose-plus-base` | size-sibling | float32/float16 | 478 | 4 | fp16 158 ms, fp32 179 ms |
| pose | ViTPose / ViTPose++ | `vitpose-plus-huge` | size-sibling | float32/float16 | 3431 | 4 | open: 3431 MB weights, skipped (download and memory cost on the shared Mac) |
| pose | ViTPose / ViTPose++ | `vitpose-plus-large` | size-sibling | float32/float16 | 1657 | 4 | open: 1657 MB weights, skipped (download and memory cost on the shared Mac) |
| pose | ViTPose / ViTPose++ | `vitpose-plus-small` | size-sibling | float32/float16 | 126 | 4 | fp16 72 ms, fp32 85 ms |
| pose | YuNet | `yunet-s` | size-sibling | float32/float16 | 1 | 3 | fp16 18 ms, fp32 22 ms |
| promptable_segmentation | EfficientViT-SAM | `efficientvit-sam-l1` | size-sibling | float32 | 182 | 2 | fp32 101 ms |
| promptable_segmentation | EfficientViT-SAM | `efficientvit-sam-l2` | size-sibling | float32 | 234 | 2 | fp32 120 ms |
| promptable_segmentation | EfficientViT-SAM | `efficientvit-sam-xl0` | size-sibling | float32 | 447 | 2 | fp32 268 ms |
| promptable_segmentation | EfficientViT-SAM | `efficientvit-sam-xl1` | size-sibling | float32 | 776 | 2 | fp32 435 ms |
| promptable_segmentation | HQ-SAM | `sam-hq-vit-h` | size-sibling | float32/float16 | 2452 | 3 | open: 2452 MB weights, skipped (download and memory cost on the shared Mac) |
| promptable_segmentation | HQ-SAM | `sam-hq-vit-l` | size-sibling | float32/float16 | 1197 | 3 | fp16 1051 ms, fp32 1350 ms |
| promptable_segmentation | Segment Anything (SAM) | `sam-vit-h` | size-sibling | float32/float16 | 2446 | 3 | open: 2446 MB weights, skipped (download and memory cost on the shared Mac) |
| promptable_segmentation | Segment Anything (SAM) | `sam-vit-l` | size-sibling | float32/float16 | 1192 | 3 | fp16 1010 ms, fp32 1282 ms |
| restoration | NAFNet | `nafnet-sidd` | size-sibling | float32/float16 | 443 | 3 | fp16 277 ms, fp32 457 ms |
| restoration | SwinIR | `swinir-m-x4` | size-sibling | float32/float16 | 65 | 3 | fp16 4339 ms, fp32 6887 ms |
| restoration | SwinIR | `swinir-m-x4-real` | size-sibling | float32/float16 | 64 | 3 | fp16 4668 ms, fp32 7297 ms |
| stereo | AANet | `aanet-plus-sceneflow` | size-sibling | float32/float16 | 33 | 4 | fp16 1367 ms, fp32 1898 ms |
| stereo | BridgeDepth | `bridgedepth-rvc-pretrain` | size-sibling | float32/float16 | 1373 | 4 | open: 1373 MB weights, skipped (download and memory cost on the shared Mac) |
| stereo | C-FoundationStereo | `fast-foundationstereo` | checkpoint-sibling | float32/float16 | 68 | 3 | skip: checkpoint of `c-fast-foundationstereo` |
| stereo | Cas-GwcNet | `cas-gwcnet-c-sceneflow` | size-sibling | float32/float16 | 121 | 3 | fp16 2934 ms, fp32 6066 ms |
| stereo | DLNR | `dlnr-middlebury` | checkpoint-sibling | float32/float16 | 219 | 3 | skip: checkpoint of `dlnr-sceneflow` |
| stereo | Fast-ACVNet | `fast-acvnet-kitti12` | checkpoint-sibling | float32 | 12 | 3 | skip: checkpoint of `fast-acvnet-generalization` |
| stereo | Fast-ACVNet | `fast-acvnet-kitti15` | checkpoint-sibling | float32 | 12 | 3 | skip: checkpoint of `fast-acvnet-generalization` |
| stereo | Fast-ACVNet | `fast-acvnet-plus-generalization` | size-sibling | float32 | 13 | 3 | fp32 601 ms |
| stereo | Fast-ACVNet | `fast-acvnet-plus-kitti12` | size-sibling | float32 | 13 | 3 | fp32 580 ms |
| stereo | Fast-ACVNet | `fast-acvnet-plus-kitti15` | size-sibling | float32 | 13 | 3 | fp32 583 ms |
| stereo | Fast-ACVNet | `fast-acvnet-plus-sceneflow` | size-sibling | float32 | 13 | 3 | fp32 583 ms |
| stereo | Fast-ACVNet | `fast-acvnet-sceneflow` | checkpoint-sibling | float32 | 12 | 3 | skip: checkpoint of `fast-acvnet-generalization` |
| stereo | GREAT-IGEV | `great-igev-eth3d` | checkpoint-sibling | float32/float16 | 56 | 2 | skip: checkpoint of `great-igev-sceneflow` |
| stereo | GREAT-IGEV | `great-igev-kitti` | checkpoint-sibling | float32/float16 | 56 | 3 | skip: checkpoint of `great-igev-sceneflow` |
| stereo | GREAT-IGEV | `great-igev-middlebury` | size-sibling | float32/float16 | 56 | 2 | re-run 2026-11-23: fp32 23063 ms, fp16 14680 ms, golden pass (MPS fp16 26930 ms, fp32 out of memory) |
| stereo | GwcNet | `gwcnet-g-sceneflow` | size-sibling | float32/float16 | 75 | 3 | fp16 2450 ms, fp32 3676 ms |
| stereo | HITNet | `hitnet-middlebury-d400` | size-sibling | float32/float16 | 7 | 3 | fp16 1367 ms, fp32 2094 ms |
| stereo | IGEV++ | `igev-plusplus-eth3d` | checkpoint-sibling | float32/float16 | 56 | 4 | skip: checkpoint of `igev-plusplus-sceneflow` |
| stereo | IGEV++ | `igev-plusplus-kitti12` | checkpoint-sibling | float32/float16 | 56 | 4 | skip: checkpoint of `igev-plusplus-sceneflow` |
| stereo | IGEV++ | `igev-plusplus-kitti15` | checkpoint-sibling | float32/float16 | 56 | 4 | skip: checkpoint of `igev-plusplus-sceneflow` |
| stereo | IGEV++ | `igev-plusplus-middlebury` | checkpoint-sibling | float32/float16 | 56 | 4 | skip: checkpoint of `igev-plusplus-sceneflow` |
| stereo | IGEV-Stereo | `igev-stereo-eth3d` | checkpoint-sibling | float32/float16 | 48 | 4 | skip: checkpoint of `igev-stereo-sceneflow` |
| stereo | IGEV-Stereo | `igev-stereo-kitti12` | checkpoint-sibling | float32/float16 | 48 | 4 | skip: checkpoint of `igev-stereo-sceneflow` |
| stereo | IGEV-Stereo | `igev-stereo-kitti15` | checkpoint-sibling | float32/float16 | 48 | 4 | skip: checkpoint of `igev-stereo-sceneflow` |
| stereo | IGEV-Stereo | `igev-stereo-middlebury` | checkpoint-sibling | float32/float16 | 48 | 4 | skip: checkpoint of `igev-stereo-sceneflow` |
| stereo | LiteAnyStereo V2 | `las2-h` | size-sibling | float32/float16 | 45 | 4 | fp16 537 ms, fp32 888 ms |
| stereo | LiteAnyStereo V2 | `las2-l` | size-sibling | float32/float16 | 104 | 4 | fp16 355 ms, fp32 477 ms |
| stereo | LiteAnyStereo V2 | `las2-m` | size-sibling | float32/float16 | 40 | 4 | fp16 190 ms, fp32 252 ms |
| stereo | MonSter | `monster-mix-all` | checkpoint-sibling | float32/float16 | 1483 | 4 | skip: checkpoint of `monster-sceneflow` |
| stereo | RAFT-Stereo | `raft-stereo-eth3d` | checkpoint-sibling | float32/float16 | 43 | 4 | skip: checkpoint of `raft-stereo-middlebury` |
| stereo | Selective-IGEV | `selective-igev-eth3d-finetune` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | Selective-IGEV | `selective-igev-eth3d-train` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | Selective-IGEV | `selective-igev-kitti12` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | Selective-IGEV | `selective-igev-kitti15` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | Selective-IGEV | `selective-igev-middlebury-finetune` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | Selective-IGEV | `selective-igev-middlebury-train` | checkpoint-sibling | float32/float16 | 51 | 4 | skip: checkpoint of `selective-igev-sceneflow` |
| stereo | WAFT-Stereo | `waft-stereo-b` | size-sibling | float32/float16 | 1075 | 1 | fp16 5040 ms, fp32 7982 ms |
| tracking | TAPIR | `bootstapir` | size-sibling | float32 | 209 | 3 | fp32 1881 ms |
| tracking | Track-On2 | `track-on-r` | checkpoint-sibling | float32/float16 | 199 | 2 | skip: checkpoint of `track-on2` |

## Summary

| kind | gaps | measured now | open | skipped |
|---|---|---|---|---|
| no-family-row | 67 | 62 | 5 | 0 |
| size-sibling | 162 | 139 | 23 | 0 |
| checkpoint-sibling | 43 | 0 | 0 | 43 |

All measured rows have status `ok`, and every golden checked passed. 356 rows were added to the MLX survey file.

**Open, by reason**

- **Weights of 1.3 GB or more** (19 refs): the so400m, H/14 and g/14 encoders (CLIP, SigLIP 2, C-RADIO, PE), the H-size SAMs,
  vitpose-plus-large/huge, owlv2-l14, loma-g and bridgedepth-rvc-pretrain. The shared 16 GB Mac swaps heavily, so they were
  skipped. pe-*-g14 (7 GB) do not fit at all.
- **Out of memory on both backends at the survey size** (4 refs): depthfm-v1 (asks for a 71 GB buffer) and the three
  flowformer-small checkpoints (Metal out of memory).
- **Hit the 150 s cap of this pass** (3 refs and 4 fp32 rows): memfof-kitti, memfof-sintel and great-igev-middlebury in
  both dtypes, and gmflow-scale2-regrefine6-* in fp32. Re-run 2026-11-23 on a quiet Mac: great-igev-middlebury and the
  gmflow fp32 rows now pass (goldens checked); memfof-kitti / -sintel still time out at 900 s (2x upsampled frames). The Mac was swapping 4 GB of 5 GB, so the timeouts were dropped
  from the survey file. They need a re-run with the default timeout on a quiet machine.

## Auto routes from this pass

`tools/gen_auto_mode.py` now has 121 `auto` entries, up from 105; none were lost. The new MLX auto routes are alexnet,
densenet121, squeezenet1_1, shufflenet_v2, ema-vfi-t, memflow-kitti/sintel, rtmdet-nano-person/s/m/l, sam-vit-l,
waft-stereo-b and yolox-s/l/x. Most of the newly measured refs give no route because their family has no MPS `ok` row
to compare against, or because MPS is faster.

**Policy gap**: s2m2-xl fp32 runs on MLX (13.0 s warm) but runs out of memory on MPS. With no MPS row to compare against,
gen_auto_mode emits no route, so `auto` never picks MLX for it, even though MLX works. A rule of "MLX when MPS has
no `ok` row" would fix this and the same case for other models.

## Unmeasured refs surveyed (2026-11-23)

66 MLX-capable refs had no MLX row (mostly sibling checkpoints and larger sizes). All surveyed, portable + kernels:
151 runs ok, 147 golden pass. Open:

- flowformer-small-sintel / -things: MLX out of memory at 1080x1920 (fp32 and fp16); MPS also OOM. Fixed since: see
  the next section.
- loma-g / loma-r: run fine on MLX, but no golden and no MPS row yet.

24 refs gained an `auto` MLX route, all siblings of nets already routed (igev-stereo, igev-plusplus, selective-igev,
great-igev, raft-stereo, waft-a1) plus sam-vit-h, vitpose-plus-huge, owlv2-l14, track-on-r, flowseek-t-notartan.

## Memory-lean MLX ports of MEMFOF scale 1 and FlowFormer Small (2026-11-23)

memfof-kitti / -sintel (frames doubled to 2160x3840, 32400 tokens at 1/16) and flowformer-small-* (32400 tokens at
1/8) built O(N^2) tensors: the all-pairs volumes (4.2 GB float32 each), GMA's attention matrix (4.2 GB), FlowFormer's
patch embeddings of all cost maps (8.5 GB) and its sub-sampled cost-encoder attention scores (8 GB). The MLX runners
(`_Mlx` in each `model.py`) now avoid all of them:

- `fastforks::corr_windows` (`_core/volume_ops.py`, Metal kernel in `kernels/graph_mlx/ops.py`): RAFT's windows from
  the pooled f2 pyramid, one SIMD group per (pixel, level), fp32 accumulation; no volume.
- GMA: above 1 GiB of float32 matrix, (q, k) instead of the matrix; the aggregation is one SDPA (MEMFOF's head size
  512 runs in query blocks, FlowFormer's 128 fused). MEMFOF fp16 at 2160x3840: keeping the fp16 matrix was 15 s and
  4.6 GB peak, against 17 s and 2.9 GB recomputed.
- FlowFormer: deferred volume, cost maps made, embedded and projected 512 at a time; fused cost-encoder attention.

Survey at 1080x1920 (warm median, peak footprint): memfof-kitti fp16 16.1 s / 3652 MB, fp32 19.7 s / 4087 MB;
memfof-sintel 16.0 s / 3643 MB, 19.9 s / 3959 MB; flowformer-small-sintel / -things / -chairs fp16 13.0 s / about
1.9 GB, fp32 19.8 s / about 1.9 GB. All goldens pass. MPS has no `ok` row for these at 1080x1920 (out of memory), so
no `auto` route follows (the policy gap above). memfof / memfof-spring (scale 0) use the windows kernel too: peak
2380 -> 1986 MB fp16 and 2980 -> 2293 MB fp32, time +1 % fp16 and +4-6 % fp32, goldens pass.
