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
| stereo | HITNet | `hitnet-middlebury-d400` | size-sibling | float32/float16 | 7 | 3 | fp16 686 ms, fp32 1616 ms (cost volumes as Metal kernels) |
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

## Float16 MLX: families that stay float32 (2026-11-23)

- **efficientvit-sam (l0 .. xl1)**: not declared; no single op found that brings all five within limits (2026-11-24). Three causes of the
  gross float16 errors are fixed in the code: the LiteMLA linear attention's sums over H * W overflow float16 (now a
  `to_float32` island, back to the run dtype after it); `fast.tensor_eps` turns each 1e-15 epsilon into a tensor
  constant, which GraphNet casts to float16 = 0, so 0 / 0 wherever a ReLU'd query is all zero (the twin of an MLX
  runner keeps the Python float); and the final `LayerNorm2d` (mean and variance over 256 channels) ran in float16
  on the replay, where PyTorch runs it in float32 (xl0 pred_masks mean |diff| 0.022 of the golden's median, 0.0044
  with the norm in float32, limit 0.0081; xl1 0.037 -> 0.0051, limit 0.0052). With these, every output of every
  variant is within about 0.5-1.4x of its limit, and which single-scalar score crosses it changes with each further
  float32 island (worst error / limit per variant, `ok` = golden passes):

  | float16 replay plus | l0 | l1 | l2 | xl0 | xl1 |
  |---|---|---|---|---|---|
  | nothing | 1.04 iou | ok (0.99) | ok | ok | 1.38 iou |
  | 1x1 convs from >= 1024 channels in float32 (as `fast.wide_points` on CUDA) | ok | 1.13 box iou | ok | ok | 1.14 iou |
  | 1x1 convs from >= 256 channels in float32 | ok | 1.10 box iou | ok | ok | 1.03 iou |
  | all ungrouped 1x1 convs in float32 | ok | 1.00 box iou (over) | ok | ok | 1.03 iou |
  | Winograd 3x3 convs in float32 (`half_wino_f32`) | 1.27 iou p99 | 1.05 box iou | - | - | 1.15 iou |
  | Winograd 3x3 + 1x1 from >= 256 in float32 | ok | 1.20 box iou | - | - | 1.06 iou |

  No row passes all five, and the family has one card. The limits were set on CUDA float16 (float32 accumulation in
  cuDNN) with little headroom; what is left on MLX looks like the rounding of float16 activations through the whole
  trunk rather than one operation. Also without effect: the neck's bicubic upsampling in float32. Not isolated: the depthwise and the
  non-Winograd 3x3 convs (the switch tried for them did not reach them). Float16 would need either wider
  limits for this family or a float32 last stage, which gives up the speed. MLX float32 is unchanged (l0 pred_masks
  6e-06 of the median). MPS float16 is not declared either.
- **ben2**: fixed and declared (2026-11-24). MLX float32 failed its golden (`test_mlx[ben2@1]`: alpha mean |diff|
  0.1009; MPS passed) since the instance norm became a channels-last op (6ba44738): `mlx_band.band_tail` takes every
  such op for row-local, so the last block (nearest 2x, conv, InstanceNorm, GELU, output conv at 1024x1024, 512 MB a
  map) ran per row band with each band's own statistics. The 2026-11-19 row was a real pass (older code; its
  golden_mean 0.0 is rounding). An instance norm now ends the banded chain. Survey at 1080x1920: float32 1524 ms /
  4658 MB (golden 0.0), float16 850 ms / 2863 MB (golden mean 1e-05, limit 2.4e-05); float16 is the MLX default.

## MLX float16 for the geometry families (2026-11-23)

MLX float16 added (golden pass, no slower than MLX float32, peak footprint no higher beyond run-to-run noise), warm
median ms / peak MB on M1 Pro from the survey rows, fp16 vs fp32:

- recover-cx / -mn / -rn: 758 / 1297 vs 1064 / 1713, 316 / 1181 vs 471 / 1240, 429 / 1224 vs 624 / 1430. The delta
  flow, the accumulated flow and the upsampling mask stay float32.
- cfnet-sceneflow: 3178 / 3669 vs 4485 / 4664. The cost volumes and the disparity regression stay float32
  (`_core/mlx_cast.float32_call`: float32 inputs and weights in the MLX export).
- romav2, portable: 6078 / 2849 vs 6727 / 4449. Kernels: 3633 / 3120 vs 3669 / 3170. The DPT convs run in float16;
  the matcher and the refiners stay float32. This is an MLX-engine-only fix, so MPS stays float32.
- neuflow-v2 / -sintel / -things: 949 / 9200 vs 951 / 9257, 952 / 9201 vs 1021 / 9268, 946 / 9170 vs 975 / 9255.
  The backbone and the two s8 merge convs stay float32 (`float32_call`). With everything else in half, the golden
  failed at mean 0.0024 vs the 0.0014 tolerance. No single float32 island passed (backbone 0.0018, merges 0.0018,
  cross-attention / refinement / upsampling 0.0023-0.0024). Backbone plus merges together: mean 0.0007. The gain is
  small since the float32 backbone is most of the time.
- loftr-indoor / -outdoor: 1549 / 3675 vs 1806 / 4296, 1526 / 3703 vs 1788 / 4095. Plain float16 passed the golden
  but was slower (2354 vs 1792 ms): portable float16 keeps ragged 3x3 convs out of MLX's Winograd, so the ResNet-FPN's
  196-channel convs ran on the GEMM path (backbone 854 vs 509 ms). `half_wino_pad=True` (new `mlx_wrap.port` option)
  pads them into Winograd as float32 does. The golden error is the same either way (outdoor mean 0.0023 vs the
  0.0025 tolerance, indoor 0.0018). Fine matching runs in PyTorch on the CPU (about 1 s of the call).
- disk-lightglue: 786 / 2058 vs 886 / 2182 (float32 was 1178 / 2333 before the split head). The U-Net stays float32
  up to the heatmap; the last 5x5 conv (80 -> 129 at full resolution, most of the arithmetic) is split into a
  float32 1-channel heatmap conv and a 128-channel descriptor conv in the run dtype, which also drops the GEMM's
  padding of 129 outputs to 192 in float32. The whole U-Net in half failed (recall 0.9917, agree 0.9879, mean
  0.0092 vs 0.9971 / 0.9913 / 0.0052): it shifts integer-grid keypoint maxima, as CUDA's fp16 kernels path does.
- aliked-lightglue: 455 / 2407 vs 448 / 2407, the same arithmetic: all of ALIKED's convs are float32 islands (in
  half: recall 0.9963, agree 0.9903, also with only the score head in float32), so float16 is float32 here within
  run-to-run noise. Declared because the card's dtype list is per family and DISK gains.
- fast-acvnet (base, sceneflow / kitti12 / kitti15 / generalization): 459 / 1934 vs 523 / 1931, 457 / 1808 vs
  521 / 1939, 455 / 1831 vs 506 / 1933, 475 / 1918 vs 505 / 1935. Plus: 540 / 1813 vs 584 / 1928, 564 / 1814 vs
  582 / 1955, 562 / 1810 vs 580 / 1928, 541 / 1812 vs 587 / 1928. Everything before the top-k (features,
  correlation, attention hourglass) is a float32 island, the stages after it run in half. With those in half,
  fast-acvnet-plus-sceneflow failed (mean 0.0753, p99 0.155 vs 0.02 / 0.25: a quarter-resolution top-k flip) and
  float16 was slower (549 vs 487 ms).

Engine changes behind these (MLX only):

- `to_float32` of a parameter (`float32_call`) is the parameter itself, stored in float32: a float32 replay sees a
  plain parameter again (batch norms fold into their convs, siblings merge), a half replay folds the batch norms in
  float32 and keeps no half-rounded copy. Before, `float32_call` stages lost the batch-norm fold in both dtypes:
  float32 cfnet-sceneflow 4683 -> 4485 ms, neuflow-v2 1072 -> 951 ms, fast-acvnet with the islands 600-750 ->
  505-590 ms.
- A half replay with `float32_call` stages counts the activations downstream of them at 4 bytes when it paces its
  checkpoints. Counted at 2, the segments were twice as long: aliked-lightglue float16 393 ms / 2722 MB (float32
  446 / 2407), disk-lightglue 776 / 2492 (884 / 2188), fast-acvnet about 130 MB above float32.

No MLX float16, because the golden fails (tolerances unchanged):

- xfeat / xfeat-lighterglue: recall 0.9983 vs 0.999. It was also no faster (158 vs 164 ms, 521 vs 524 ms): the
  network is about a third of the call (64 of 190 ms profiled), the rest is CPU-side, and the float32 islands a fix
  needs would take the small gain away.

## Where MLX trails MPS: the next optimization queue (2026-11-24)

From the two M1 Pro survey files (best passing row per ref and backend, warm ms): 541 refs have both, geomean MLX/MPS
0.895, and 176 run more than 5% slower on MLX. `matting/matanyone` has had an MLX port since 2026-11-24; the six TotalSegmentator
refs have a port but no MLX row yet. Rows the running agents are re-measuring (ben2, EfficientViT-SAM, the Conv3d and
fp16 re-surveys) may already be better than below.

Work them one family at a time, at most two or three GPU agents at once (they share `gpu.sh`). Each: profile the
MLX graph (`FF_MLX_PROFILE=1`, the per-op timing of `mlx_graph`), fix the slowest ops in the translator or a
family `mlx.py`, re-survey the family's refs, regenerate `apple_mlx.json`.

| MLX/MPS | refs | lead |
|---|---|---|
| 2.1, 1.6 (were 4.2, 3.1) | hitnet-eth3d, hitnet-middlebury-d400 | cost volumes and warp SADs are Metal kernels now (289 / 686 ms fp16, faster than MPS portable, behind MPS kernels 136 / 440 ms). Left: 32-channel 3x3 convs are 1.4x slower than MPSGraph, and activations after the 1x1, 4x4 stride-2 and residual convs are not fused; `conv_mlx` gave no steady gain, so no kernels mode |
| 0.72, 0.76 (were 3.9, 2.9) | xfeat, xfeat-lighterglue | done, both auto-routed to MLX: hand-written network, keypoint selection and matchers in MLX operations ("Done: XFeat" below) |
| 0.75, 0.71, 0.75 (were 2.9, 2.2, 1.5) | restormer-derain/sidd/deblur | done, see "Restormer and RealPLKSR on MLX" below: 737 / 794 / 737 ms fp16, 3.7-4.0 s first call, peak below MPS |
| 2.7, 1.7 | aanet-sceneflow, aanet-plus | deformable conv: the MLX port gathers per sample point |
| 2.5 | coex-sceneflow | cost volume build and the guided aggregation |
| 2.5, 1.7 | rife-4.25-lite, rife-4.25 | `grid_sample` warps per IFBlock; small convs (launch bound) |
| 2.2 | loftr-indoor/outdoor | fine matching runs on the CPU (geometry agent) |
| 0.49 (was 2.2) | aliked-lightglue | done, auto-routed to MLX: ALIKED in MLX operations without the full-resolution descriptor map, `MlxLightGlue` ("Done: the LightGlue family" below) |
| 2.2, 1.8, 1.7 | efficientvit-sam-l0/xl1/xl0 | the ben2/EfficientViT-SAM agent is on it |
| 2.0, 1.9 | mask-rcnn, faster-rcnn r50-fpn | RoIAlign and the per-level proposal ops |
| 0.94 (was 1.8) | realplksr-nomoswebphoto-x4 | kernels fp16 1029 ms vs MPS 1092, but 1919 MB against 1368 MB; portable fp16 is the lean mode (1529 ms, 1198 MB). Left: the 862 MB active peak of the Winograd convs, group norm (3 ms x 28) |
| 1.7 | dlnr-sceneflow, ema-vfi-small-t, raft-stereo-middlebury | 29-37 s first calls: compile time first |
| 1.5-1.6 | lineart, anyline, segnext-mscan-l/b, sapiens-seg-0.3b, oneformer-swinl, gmflow-scale1 | not looked at yet |

Also: survey the six TotalSegmentator
refs on MLX (the 1.5 mm `total` runs 34 s warm vs 23 s on MPS; 3mm is 1.4x slower).

## Restormer and RealPLKSR on MLX (2026-11-24)

M1 Pro, 512 x 512, warm median, first call, peak process footprint; best row per backend. All MLX goldens pass.

| ref | MPS best | MLX before | MLX now | MLX/MPS |
|---|---|---|---|---|
| restormer-deblur | 989 ms, 1.10 s, 3651 MB | 1467 ms, 4.9 s, 1317 MB | fp16 737 ms, 3.66 s, 2046 MB | 0.75 |
| restormer-derain | 988 ms, 1.10 s, 3653 MB | 2815 ms, 17.1 s, 2668 MB | fp16 737 ms, 4.03 s, 2048 MB | 0.75 |
| restormer-sidd | 1112 ms, 1.24 s, 3675 MB | 2427 ms, 17.8 s, 2672 MB | fp16 794 ms, 3.66 s, 2137 MB | 0.71 |
| realplksr-nomoswebphoto-x4 | 1092 ms, 1.30 s, 1368 MB | portable fp16 1991 ms, 9.7 s, 953 MB | kernels fp16 1029 ms, 2.85 s, 1919 MB | 0.94 |

Other rows now: Restormer float32 1545 / 1548 / 1567 ms (2654 / 2658 / 2632 MB); RealPLKSR portable fp16 1529 ms,
3.00 s, 1198 MB; portable fp32 2230 ms, 1850 MB; kernels fp32 (runs as portable) 2131 ms, 2070 MB.

Restormer (`restoration/models/restormer/mlx_ops.py`):

- The transposed attention's `q @ k^T` over `H * W` tokens is one Metal kernel (`graph_mlx.ops.token_gram`), and the
  gated depthwise feed-forward (depthwise 3x3, chunk, GELU, multiply) another (`dw_gate`): both read the
  channels-last data, so nothing is copied.
- The MLX buffer cache limit is 1024 MB: the transients are larger than the default 128 MB, so every one was a fresh
  zero-filled allocation. Deblur fp16: 1062 ms at the default, 858 at 512 MB, 732 at 1024 MB, 699 at 2048 MB. The
  cache is resident memory, which is why the footprint went up (1317 to 2046 MB for deblur) while staying below MPS.
- First call 3.7-4.0 s, of which `torch.export` is 2.56 s and the translation 0.26 s: nothing cheap is left there.
- Left: the 1x1 convs are about 435 of 1001 ms in the eager profile. A 1x1 conv as `mx.matmul` on the channels-last
  data is 20-25% faster than `mx.conv2d` (96 to 512 channels at 512 x 512: 6.44 vs 8.43 ms; 96 to 288: 4.17 vs 5.32;
  256 to 96: 4.29 vs 5.11). Not done here: the regular conv path of the translator's `_conv2d` belongs to the
  ragged-padding work; it would help every 1x1-heavy family.

RealPLKSR (`restoration/models/realplksr/mlx_ops.py`):

- The 17 x 17 conv of 16 channels as row GEMMs: 15.2 ms against `mx.conv2d`'s 26.3 ms in float16 (20.5 vs 31.2 ms
  float32), 16.8 ms inside the network; 28 of them. Computed in bands of output rows (64 MB of row products, 3 bands
  at 512 x 512): same speed, 220 MB less peak.
- Not adopted: `conv_mlx.direct` on the 17 x 17 kernel, 26.5 ms at its best block size, no faster than `mx.conv2d`.
- Mish as one Metal kernel (`graph_mlx.ops.mish`, `tanh(log1p(e)) = (e^2 + 2e) / (e^2 + 2e + 2)`): 1.2 ms against
  3.1 ms for the compiled chain on float16 [1, 512, 512, 128], 28 times.
- A kernels mode on MLX (`conv_mlx`, the Winograd 3x3 convs, float16 only), with a 1024 MB buffer cache: 1582 ms at
  the default 128 MB, 1162 at 512 MB, 1126 at 768 MB, 1023 at 1024 MB, 945 at 2048 MB. Portable keeps the default
  cache (1526 ms, 1192 MB; 1395 ms and 1390 MB at 1024 MB).
- MPS's time and MPS's peak are not both reachable. Kernels fp16, PLK band MB / cache MB (/ chunk MB) to warm ms and
  footprint MB: 0 / 1024: 1022, 1896; 64 / 1024: 1030, 1917; 64 / 512: 1162, 1790; 32 / 512: 1317, 1711;
  32 / 384: 1307, 1577; 32 / 256: 1397, 1486; 16 / 256: 1591, 1356; 32 / 384 / 128: 1355, 1559;
  32 / 256 / 128: 1484, 1496; 32 / 512 / 512: 1235, 1751. The warm active MLX peak is 862 MB (789 MB with
  `FF_BAND_MB=48` and a 64 MB chunk, at 1078 ms), so the footprint is the base, that peak and the cache.
- Left: lower the active peak of the Winograd convs at 4x-upsampled width, group norm (about 3 ms x 28), and the
  1x1-as-matmul lead above.

## TAPIR / BootsTAPIR float16 on MLX (2026-11-23): not declared

The tracking fp16 pass (tapnext, cotracker, locotrack gained MLX float16) left the `tapir` card at MLX float32: one
card holds both variants, and tapir@1 misses its golden in every mixed-precision variant tried, while bootstapir@1
passes all of them (tracks mean ~0.0027-0.0034, p99 ~0.012-0.017, visibility mean <= 0.0001). Tolerances: tracks
mean 0.0075 / p99 0.027, visibility mean 0.0011 / p99 0.0063. MPS fp16 (autocast) passes for both (tapir mean 0.0039).

| tapir@1 MLX fp16 variant | tracks mean / p99 | visibility mean / p99 |
|---|---|---|
| fp32 queries, coordinates, logits, unit features, mixer residual stream, cost volume | 0.0086 / 0.061 | 0.0030 / 0.022 |
| + backbone input fp32, or + hid convs fp32 | unchanged | unchanged |
| + whole PIPs mixer fp32 | 0.0091 / 0.058 | 0.0031 / 0.023 |
| + antialiased resize in fp32 | 0.0066 / 0.032 | 0.0007 / 0.004 |
| + video entering as fp32 too | 0.0067 / 0.025 | 0.0022 / 0.011 |
| + MLX half Winograd 3x3 convs in fp32 (`half_wino_f32`) | 0.0038 / 0.022 | 0.0014 / 0.007 |
| + Winograd fp32 and fp32 resize | 0.0047 / 0.039 | 0.0020 / 0.013 |
| + Winograd fp32 and fp32 ResNet residual stream | 0.0051 / 0.032 | 0.0022 / 0.015 |
| + all three | 0.0034 / 0.017 | 0.0017 / 0.009 |

The error does not fall monotonically with more fp32: a few ambiguous tracks flip between candidate positions (and
their visibility flips with them), so each variant misses a different limit. Everything was reverted; tapir keeps
the fp32 port. Revisit with a tighter fp16 backbone (direct instead of Winograd convs throughout, fp32 instance-norm
outputs) or a golden with fewer ambiguous query points.

### Where the tapir@1 budget goes (2026-11-23, second pass): the fp16 weights

A second pass asked which op breaks the golden. No single op does. Two measurements on the M1 Pro (480 x 854 call
for time and peak; the golden clip for the errors; limits as above):

Weights rounded to fp16 with all arithmetic in fp32 (CPU, PyTorch; nothing else differs from the reference):

| tapir@1 weights rounded to fp16 | tracks mean / p99 | visibility mean / p99 |
|---|---|---|
| all | 0.00123 / 0.00848 | 0.00098 / 0.00466 |
| ResNet `initial_conv` only | | 0.00044 |
| ResNet stage 1 only | | 0.00041 |
| ResNet stage 2 / 3 / 4 only | | 0.00009 / 0.00005 / 0.00001 |
| whole ResNet | | 0.00100 |
| cost-volume convs only | | 0.00004 |
| PIPs mixer only | | 0.00003 |
| bootstapir@1, all | 0.00048 / 0.00191 | 0.00001 / 0.00011 |

So for tapir@1 the fp16 rounding of the first convolution and the first ResNet stage alone uses 0.9 of the
visibility-mean limit (0.0011) before any fp16 arithmetic happens. A float16 replay casts every constant to the run
dtype (also the weights of convolutions kept in fp32), so every MLX fp16 variant starts there. bootstapir@1 is not
sensitive to its weights, which is why it passes everything.

fp32 islands on top of the fp16 replay (W: half Winograd convs in fp32; V: video enters as fp32; Bk: ResNet stages
below k compute in fp32; H: cost-volume convs fp32; M: mixer fp32):

| tapir@1 MLX | tracks mean / p99 | visibility mean / p99 | warm ms | peak MB | golden |
|---|---|---|---|---|---|
| fp32 | 0.00001 / 0.00004 | 0 / 0 | 1299 | 1672 | pass |
| fp16 plain | 0.01195 / 0.05409 | 0.00324 / 0.02362 | 1079 | 1687 | fail |
| V, W | 0.00568 / 0.02217 | 0.00125 / 0.00570 | 1092 | 1680 | fail (visibility mean) |
| B1, V, W | 0.00811 / 0.03161 | 0.00073 / 0.00487 | 1157 | 1681 | fail (tracks) |
| B2, V, W | 0.00757 / 0.02325 | 0.00156 / 0.00927 | 1198 | 1689 | fail |
| B5, V | 0.00512 / 0.02368 | 0.00124 / 0.00570 | 1125 | 1737 | fail (visibility mean) |
| H, V, W | 0.00505 / 0.01592 | 0.00135 / 0.00672 | 1133 | 1680 | fail |
| M, V, W | 0.00548 / 0.02087 | 0.00122 / 0.00624 | 1398 | 1690 | fail |
| H, M, V, W | 0.00634 / 0.02865 | 0.00125 / 0.00642 | 1387 | 1681 | fail |
| B5, H, M, V (nearly all arithmetic fp32) | 0.00505 / 0.02417 | 0.00108 / 0.00510 | 1273 | 1695 | pass, at 98 % of the limit |

The one passing variant computes almost everything in fp32 on fp16-rounded weights: no faster than fp32 (1273 against
1299 ms) and its peak is above fp32's, so it fails the float16 gates (no slower, no more memory) by itself and passes
the golden with no margin. The golden has 36 track values, so a single track that moves between two candidate
positions shifts the mean by more than the remaining budget; that is the non-monotonic pattern in both tables.

A real fix needs fp32 weights for `initial_conv` and stage 1 inside an fp16 replay (a per-parameter fp32 mechanism
in `mlx_graph`'s constant and conv-fold paths, which does not exist) plus fp32 Winograd convs for the rest. The best
case is the V, W row's speed with stage 1 in fp32: about 1160-1200 ms against 1299 ms (8-10 %) with the peak a few
MB above fp32. Not built: the gain is small, the memory gate would likely fail, and it would be a mechanism for one
variant of one card. tapir stays fp32 on MLX. MPS fp16 (autocast, which also rounds the conv weights) passes with
tracks 0.0043 / 0.018: its arithmetic noise happens to land on the passing side of the same narrow margin.

### Done: DINOv2 linear segmentation and MoGe (2026-11-24)

Best passing row per backend, M1 Pro 16 GB, warm ms / first call s / peak MB (MPS best is fp16 portable, except
`moge-3-vitl`: kernels float32; MLX is fp16 portable):

| ref | MPS | MLX before | MLX now | MLX/MPS | auto |
|---|---|---|---|---|---|
| dinov2-s-linear-ade20k | 195 / 0.26 / 1532 | 600 / 0.76 / 1232 | 112 / 0.13 / 1002 | 0.57 | MLX |
| dinov2-b-linear-ade20k | 361 / 0.45 / 2174 | 761 / 0.80 / 1611 | 256 / 0.28 / 1372 | 0.71 | MLX |
| dinov2-l-linear-ade20k | 927 / 1.00 / 3167 | 1278 / 1.38 / 2388 | 761 / 0.78 / 2223 | 0.82 | MLX |
| moge-2-vits-normal | 303 / 0.40 / 1679 | 886 / 3.76 / 2024 | 271 / 1.96 / 1449 | 0.89 | MPS (first call) |
| moge-2-vitb-normal | 494 / 0.62 / 2263 | 1144 / 4.02 / 2183 | 449 / 2.14 / 1731 | 0.91 | MPS (first call) |
| moge-2-vitl-normal | 1176 / 1.25 / 3258 | 2081 / 6.36 / 3009 | 1027 / 3.21 / 2482 | 0.87 | MLX |
| moge-3-vitl | 2415 / 2.61 / 6121 | 3257 / 9.71 / 3170 | 2573 / 4.85 / 3084 | 1.07 | MPS |

**DINOv2 linear.** The leads in the queue were wrong: the ViT took 90 of dinov2-s's 660 ms. The hand-built `Mlx`
runner never set `mlx_wrap.POST_ON_MLX`, so the per-window bilinear resize, the window average (~40 ms) and the
image-size `upsample_argmax` (~525 ms) ran in PyTorch on the CPU. Now the patch logits stay on MLX
(`DinoLinearProbe.logits`), one Metal kernel resizes and averages the windows at the network size
(`mlx_seg.slide_scores`) and the existing argmax kernel takes that array (`mlx_seg.resize_argmax`). Labels agree
with MPS fp16 on 99.2% of the pixels of a smooth 1024x2048 test image (both pass the golden). What is left is the
ViT itself.

**MoGe.** The survey call computes no normals (`normals` is off by default), so the normal head was not the cause:
the cost was the full-resolution last level of the neck and of every head (bilinear 2x resize, replicate-padded
3x3 convs at 784x1168, the uv concat), and replicate pads that were gathered out of the channels-last layout.
Three changes, 886 -> 271 ms for vits fp16 (the first step is measured from this branch's base, ~420 ms: the fp16
work of 2026-11-23 had already landed after the 886 ms row):

- The last ConvStack level is linear from the level below to the output (resampler + uv 1x1 in the neck; resampler,
  1x1 in, sum, 1x1 out in the head). `_FoldedTop` folds it into two 3x3 convs to 4 x C_out phase channels at half
  resolution plus a pixel shuffle and a separable uv term, in float32 (weights folded in float64; equal to the
  plain stack to ~5e-7, `test_moge_folded_top_level_matches_the_conv_stack`). ~420 -> 331 ms. The same fold would
  help the MPS and CUDA portable runners (not done here: it needs their fp16 golden margins checked).
- Replicate and reflect padding run on the channels-last view (`mlx_graph` `CL_OPS`), so the conv after a pad no
  longer copies its input back: 331 -> 303 ms (the 64-channel convs 9.2 -> 5.3 ms each).
- `pipeline=True` on the port (the graph replays lazily in 256 MB checkpoints): 303 -> 272 ms.

Still open for MoGe:

- First call: done, see "MoGe trunk without an export" below.
- Warm, vits fp16 per call (eager per-op times): ViT ~135 ms (SDPA 64, linears 46), decoder 3x3 convs ~120,
  replicate pads ~47, relu/add ~29, antialiased input resize 14. A replicate pad (2.4 ms at 64 x 392 x 584) costs
  half its conv.
- `moge-3-vitl` (2573 ms vs 2415 ms for MPS kernels float32; MPS portable fp16 is 3333): the ViT-L encoder at 3578
  tokens dominates.

### MoGe trunk without an export (2026-11-24)

The first call of an exported port pays for `torch.export` itself: in a fresh process ~0.6 s of imports the first
time anything is exported, ~0.9 s to trace MoGe ViT-S's `_trunk` (more for B / L), 0.13-0.26 s to translate. Caching
the translated plan across processes would still leave the trace, and exporting only the decoder still pays the
0.6 s of imports, so the whole trunk is now hand-written on MLX (`moge/mlx_net.py`): the antialiased input resize
as two matrices, the shared DINOv2 encoder (`mlx_vit.dinov2_layers` / `dinov2_table`, which Depth Anything V2's
engine now uses too), and the decoder channels-last (3x3 replicate-padded convs, the stride-2 transposed convs as
one GEMM plus a depth-to-space, the uv 1x1 convs as a separable sum, the folded top level in float32 with the
neck's part of every head in one conv). Nothing is exported and the weights convert at load (as on MPS), so load
time rises by what the first call used to hide (survey loads are disk-bound and noisy: vits 0.2 -> 0.3 s, vitl
0.3 -> ~1.5 s, moge-3 0.3 -> 1.5-4 s; MPS loads 0.3 / 0.7 / 3.4 s).

MLX fp16 portable, 994x1482, survey rows (warm ms / first call s / peak MB), MPS best as in the table above:

| ref | MPS | MLX before | MLX now | first-call budget | auto |
|---|---|---|---|---|---|
| moge-2-vits-normal | 303 / 0.40 / 1679 | 271 / 1.96 / 1449 | 282 / 0.31 / 1250 | 1.00 | MLX (was MPS) |
| moge-2-vitb-normal | 494 / 0.62 / 2263 | 449 / 2.14 / 1731 | 443 / 0.46 / 1389 | 1.61 | MLX (was MPS) |
| moge-2-vitl-normal | 1176 / 1.25 / 3258 | 1027 / 3.21 / 2482 | 1033 / 1.06 / 1927 | 3.60 | MLX |
| moge-3-vitl | 2415 / 2.61 / 6121 | 2573 / 4.85 / 3084 | 2543 / 3.05 / 2566 | 7.44 | MPS (warm) |

Float32 rows: vits 372 -> 372 ms (first 2.20 -> 0.41 s, peak 1874 -> 1790 MB), vitb 595 -> 579 (2.33 -> 0.61,
2082 -> 2036), vitl 1350 -> 1349 (3.64 -> 1.44, 3520 -> 3004), moge-3 2867 -> 2879 (5.31 -> 4.71, 3873 -> 3207).
A second input shape costs one call now (vits 1.15 -> 0.28 s). The Depth Anything V2 rows did not move.

Warm speed is equal within noise except vits fp16, 271 -> 282 ms (+4%; in-process A/B 277-282 against 271-272).
What was tried on that gap, vits fp16:

- GELU as one compiled float32 kernel, as the translator has it (`mlx_vit.gelu`, every `mlx_vit` family): 282 ->
  277 ms. Kept.
- `row_layer_norm` for the 384-wide norms (the translator's choice below 512): 0-3 ms, inside the noise. Not taken
  (it would bring a kernels op into the shared portable encoder for one ViT size).
- `mx.pad(mode="edge")` instead of slice + concatenate: 282 -> 308 ms. Reverted.
- The translator's `_conv2d` channel padding rules on the decoder convs: no change (none of them applies: 256 / 128
  / 64 channel convs and 16 / 12 / 4 output float32 top convs).
- 1x1 convs as 2-D instead of 4-D matmuls: no change. Reverted.
- `mx.async_eval` checkpoints: none 281, synchronous 285, every 2 or 4 ViT blocks 279-281 against 282. No lead.
- Stage times: resize 8 ms, ViT 109-114 ms, whole trunk ~272 ms, call ~279 ms.

Still open:

- The same no-export trunk would cut the first call of the other DINOv2-backbone ports that still export
  (`da3-large-1.1`, `da3metric-large`, `da3mono-large`, `da3-giant`, `depth-pro`): `dinov2_layers` is the encoder,
  each needs its decoder written by hand.
- vits fp16's 4% warm gap against the replayed graph: cause not found (none of the single ops above).
- `moge-3-vitl` stays on MPS kernels float32 (2543 against 2415 ms warm).

## Ragged 3x3 convs and MLX's fast path: what the translator pads (2026-11-24)

From the RAFT and SEA-RAFT MLX ports: MLX's implicit GEMM is erratic on 3x3 output counts that are not a multiple of
64, and its Winograd (dense 3x3, stride 1, channels in multiples of 32, inputs + outputs >= 256, >= 4096 positions)
is the fast path. `_conv2d` in `shared/mlx_graph.py` already padded most ragged shapes; the question was a general
output-padding rule for the rest.

**Float32: added** (`FF_MLX_NO_WINO64=1` turns it off). The shapes no earlier rule caught are convs under the
Winograd width whose next multiple of 64 outputs is in it: 128 -> 80, 128 -> 96, 112 -> 96, 64 -> 144. They are now
padded to that multiple (zero weights, output sliced), within the existing 512 MB per-sample transient bound, in
portable mode only. `mx.conv2d` alone, M1 Pro, float32, ms as it is -> padded (peak MB of the bare conv):

| shape | 128 -> 80 | 128 -> 96 | 64 -> 144 |
|---|---|---|---|
| 64 x 64 | 0.64 -> 0.55 (5 -> 34) | 0.72 -> 0.56 (5 -> 34) | |
| 135 x 240 | 3.3 -> 2.2 (36 -> 204) | 5.4 -> 2.2 (40 -> 204) | 7.3 -> 1.9 (44 -> 193) |
| 8 x 96 x 128 | 63 -> 6.0 (108 -> 607) | 9.5 -> 6.0 (120 -> 607) | |
| 270 x 480 (over the bound: not padded) | 12.4 -> 7.7 (143 -> 776) | 21.5 -> 7.7 (159 -> 776) | 29.2 -> 6.5 (174 -> 742) |
| 540 x 960 (over the bound: not padded) | | 347 -> 28.5 (633 -> 3077) | |

Padding 128 -> 80 only to 96 (still the GEMM) is slower: 3.3 -> 14.2 ms. 64 -> 160 into 192 gains little
(2.7 -> 1.9 ms) and stays as it is (the existing rule leaves 160 / 224 / 288 outputs alone).

Families with such convs (scan of the cached checkpoints for 3x3 weights in the gap, then the `mx.conv2d` calls the
rule changes at the survey input) and the regression set, survey protocol, float32 portable, rule off -> on:

| ref | convs changed | warm ms | peak MB |
|---|---|---|---|
| igev-plusplus-sceneflow | 128 of 758 (GRU 128 -> 96, 256 x 376) | 15700 -> 11507 | 4721 -> 4675 |
| neuflow-v2 | 9 of 97 (128 -> 96 / 66) | 1016 -> 835 (noise, see below) | 9230 -> 9253 |
| rt-monster-plusplus-mix-all | 1 of 157 | 845 -> 843 | 2142 -> 2142 |
| dfine-l-coco | 1 of 108 | 73.1 -> 72.8 | 997 -> 974 |
| hgnetv2_b4, efficientformerv2_s2 | 0 (under 4096 positions) | 7.3 -> 7.5, 9.6 -> 9.4 | 757 -> 756, 642 -> 645 |
| hitnet-eth3d | 0 of 116 | 394 -> 389 | 1636 -> 1769 |
| loftr-outdoor | 0 | 1765 -> 1755 | 4103 -> 4107 |
| disk-lightglue | 0 | 892 -> 896 | 2199 -> 2189 |
| recover-cx | 0 | 1050 -> 1064 | 1576 -> 1551 |
| raft-stereo-middlebury | 0 | 5118 -> 5081 | 4278 -> 4232 |

The rows with no conv changed are the noise of the protocol: a few percent in time and up to ~130 MB in the peak
footprint (hitnet-eth3d runs the same graph both times). neuflow-v2 is noise too: its median swings between 830 and
1020 ms run to run (float16, where the rule cannot fire, gave 951 -> 828 ms in the same A/B), and the re-survey
below shows no gain. IGEV++ is the one family that gains.

Re-survey with goldens (float32 portable, all pass at mean 0.0; warm ms / first call s / peak MB, before -> after):

| ref | warm ms | first call s | peak MB |
|---|---|---|---|
| igev-plusplus-sceneflow | 14770 -> 11310 | 44.7 -> 18.3 | 4834 -> 4707 |
| igev-plusplus-eth3d | 14411 -> 11268 | 25.0 -> 18.3 | 4683 -> 4667 |
| igev-plusplus-kitti12 | 15055 -> 11383 | 22.5 -> 18.4 | 4456 -> 4744 |
| igev-plusplus-kitti15 | 15502 -> 11467 | 22.1 -> 18.5 | 4685 -> 4728 |
| igev-plusplus-middlebury | 15680 -> 11338 | 23.2 -> 18.4 | 4665 -> 4705 |
| neuflow-v2 | 951 -> 967 | 3.31 -> 2.77 | 9257 -> 9211 |
| neuflow-v2-sintel | 1021 -> 979 | 3.05 -> 2.75 | 9268 -> 9226 |
| neuflow-v2-things | 975 -> 941 | 2.99 -> 2.76 | 9255 -> 9267 |

The IGEV++ peaks are the same graph's footprint within the noise above (the five refs share one architecture; the
A/B in one process gave 4721 -> 4675 MB), kitti12's earlier 4456 MB being the outlier. The float32 rows stay behind
MLX kernels float16 (igev-plusplus-sceneflow 6039 ms / 2957 MB). neuflow-v2's two dtypes tie within that noise
(things: float32 941 ms against float16 951), which is all that moved its generated MLX default to float32.

**Kernels mode: not applied.** igev-plusplus-sceneflow kernels float32 10.35 -> 10.33 s (peak 4585 -> 4750 MB),
float16 5.38 -> 5.58 s; rt-monster-plusplus-mix-all kernels float16 512 -> 511 ms (peak 1522 -> 1596 MB).

**Float16 portable: no general rule.** Every padded float16 shape that is faster is faster because it lands in
MLX's Winograd (128 -> 80 3.6 -> 2.1 ms padded to 128; 242 -> 96 17.5 -> 5.6 ms; 256 -> 126 at 135 x 240
7.7 -> 3.0 ms), whose half-precision error is ~17x the direct conv's. That is the reason portable float16 stays out
of it by default (hat-x4 went over its golden), and SEA-RAFT confirmed it again: its encoders' 128 -> 128 convs in
half Winograd are 9 % faster (1029 -> 936 ms) but put sea-raft-s-spring at mean 0.0023 against a 0.002 tolerance.
Float16 padding that stays on the GEMM is never faster (128 -> 64 into 128: 1.6 -> 1.9 ms; 256 -> 6 into 16 / 32:
0.95 -> 1.7 / 2.2 ms; 64 -> 64 at 540 x 960 into 128 -> 128: 10.5 -> 26.5 ms). So it stays an opt-in per net
(`half_wino_pad=True` or a tuple of `(cin, cout)` pairs, checked against the family's golden): LoFTR, RAFT's local
padding, SEA-RAFT's 256 -> 126.

**Left, not built:** a 7x7 stride-2 stem on 6 input channels is ~9x slower than on 3 (float16 1080 x 1920: 46 ms
against 5.3 ms; float32 53 against 6.2; padding the input to 16 channels: 17 ms). SEA-RAFT's port splits it into
two 3-channel convs by hand. A translator rule (split a few-channel conv by input halves) would catch other
stacked-pair stems, but no other family was profiled for it.

### Done: XFeat (2026-11-24)

M1 Pro, 768x1024, float32, both goldens pass (`tools/speed_survey.py`):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| xfeat | MLX before | 164 | 2.35 | 538 | – |
| | MLX now | 29.9 | 0.07 | 642 | MLX |
| | MPS portable (best MPS row) | 41.7 | 0.81 | 1471 | |
| xfeat-lighterglue | MLX before | 524 | 2.47 | 1002 | – |
| | MLX now | 135 | 0.16 | 816 | MLX |
| | MPS portable (best MPS row) | 179 | 1.08 | 1499 | |

The old port replayed only the CNN from an exported graph and ran everything after it in PyTorch on the CPU (NMS,
top-k, two `grid_sample`s, the 4096 x 4096 nearest neighbours, all of LighterGlue): the network was a third of a
call, and the export ~1 s of the first one. Now nothing but index decisions runs on the host:

- `xfeat/mlx_net.py`: the network written out in `mlx.core` (26 convolutions, batch norm folded, channels last) with
  the resize, normalisation, heatmap and score map in one compiled function. No export, so the first call is the
  Metal shader builds only. 8.4 ms per image (conv1 at full resolution 1.4, block 1 to 3 3.7, fusion 1.5, keypoint
  head and heatmap 1.1, score map 1.0).
- `xfeat/mlx_post.py`: upstream's NMS, threshold and the two `grid_sample` reads at integer pixels are fixed gathers,
  so the score of every pixel is one map and the keypoints are its positive entries. One host read per image
  (`np.partition` + a stable sort of the candidates), bicubic descriptor sampling as a 16-tap gather on MLX: 3.9 ms
  per image. Both images' networks are queued (`mx.async_eval`) before the host ranks the first one.
- `shared/lightglue/mlx.py` (`MlxLightGlue`): LightGlue's forward, eager, with the CPU branch's point pruning; one
  host read per layer (the pruning masks) and one for the assignment. 6 layers at 4096 points: ~19 ms each before
  pruning shrinks them (self attention 4.0, cross attention 12.6, measured before the next point's change).
- MLX reduces along the last axis several times faster than along any other (4096 x 4096: softmax 0.95 against
  5.3 ms, argmax likewise). Every "other direction" (the cross attention's second softmax, the assignment's column
  normalisation and argmax, the mutual nearest neighbour) recomputes the transposed product instead:
  `softmax(b @ a.T, -1) @ v` 2.9 ms against `softmax(a @ b.T, -2).T @ v` 6.4 ms. xfeat-lighterglue 153 -> 135 ms,
  the nearest neighbours 6.8 -> 4.3 ms.

The results equal the CPU portable runner's on the survey pair (the same 4096 keypoints per image, the same 3715 /
3710 matches, scores within 7e-7 / 1.2e-5); keypoints of equal score can be ranked differently.

Tried, no gain: zero-padding the 24-channel layers to 32 (9.1 against 9.5 ms, kept at 16-aligned), `mx.compile` of
a LightGlue layer (18.0 against 18.8 ms), descriptors padded from 96 to 128 for `scaled_dot_product_attention`
(3.8 against 2.8 ms), one exp shared by both softmax directions (8.3 ms), both images as one batch of two.

Left:

- ~~`MlxLightGlue` is not used by the `lightglue` family~~ It is since "Done: the LightGlue family" below.
- float16: the attention is 2.0 against 2.8 ms in fp16 and the card is float32 only on MLX (the old port's fp16
  failed the golden at recall 0.9983 against 0.999; not re-tried, the network is no longer the larger part).
- xfeat's peak is 100 MB above the old port's (two 4096 x 4096 products for the nearest neighbours, both networks
  queued at once); still less than half of MPS.
- MPS portable was not looked at (41.7 / 179 ms); the score map as fixed gathers and the pruned LightGlue are
  plain tensor code and could be tried there.

### Done: the LightGlue family (2026-11-24)

M1 Pro, 768x1024, all goldens pass (`tools/speed_survey.py`; MPS rows of 2026-11-13 / 17):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| aliked-lightglue | MLX before (fp32) | 448 | 1.91 | 2407 | – |
| | MLX now (fp32) | 101 | 0.13 | 1105 | MLX |
| | MPS portable fp32 (best MPS row) | 206 | 0.73 | 2586 | |
| disk-lightglue | MLX before (fp16 / fp32) | 786 / 886 | 1.84 / 2.02 | 2058 / 2182 | MLX fp16 |
| | MLX now (fp16 / fp32) | 697 / 775 | 1.72 / 1.80 | 1729 / 1949 | MLX fp16 |
| | MPS portable fp32 (best MPS row) | 1994 | 2.22 | 3531 | |
| aliked-flexglue | MLX before (fp16 / fp32) | 1100 / 1224 | 8.50 / 8.79 | 1387 / 2054 | – |
| | MLX now (fp16 / fp32) | 992 / 938 | 4.22 / 4.23 | 1387 / 1644 | – |
| | MPS portable fp16 (best MPS row) | 1069 | 1.38 | 3088 | |

The card has no other refs: no superpoint / sift / doghardnet LightGlue and no standalone aliked / disk /
superpoint feature refs exist, so nothing was done for them (`MlxLightGlue` would also refuse the scale /
orientation inputs of sift and doghardnet).

The old port exported each extractor's dense network and ran the rest in PyTorch on the CPU. For ALIKED that
network ended in the full-resolution 128-channel descriptor map (four upsamplings, the concatenation, an L2 norm:
most of the 2 x 150 ms), which was then copied to the host for SDDH. Now:

- `shared/aliked/mlx.py` (`MlxAliked`): the encoder, the score head and DKD's threshold mode in `mlx.core`, batch
  norm folded, channels last, one compiled function, no export (first call 1.91 -> 0.13 s). The descriptor map is
  never formed, as in `shared/aliked/lazy.py`: the score head's first 1x1 conv is split per pyramid level and
  applied before the upsampling (8 channels instead of 128), and SDDH reads the four levels only at the pixels
  its sample points touch. The deformable convs are `graph_mlx.ops.grid_sample` plus a matmul, a few ms: no Metal
  kernel was needed. 28 ms per image for the maps, 1.2 ms for the keypoints (one host read), 9 ms for SDDH.
  Against PyTorch on the CPU at 768 x 1024 and 700 x 1001: the same keypoints (within 6e-5 px), descriptors
  within 4e-6. Top-k mode, a masked DCN and SDDH variants other than the released ones raise.
- `lightglue/mlx_post.py`: DISK's NMS (strict window maxima) on MLX, one host read of the candidates, upstream's
  top-n rule, the descriptors gathered from the dense map on MLX. The keypoints are the CPU's exactly.
- Both extractors feed `MlxLightGlue`; `describe` / `match` (the lazy stages) hand CPU tensors across, `match_many`
  stays `Portable`'s packed forward. aliked-lightglue on the survey pair: the CPU runner's 958 pairs, scores
  within 9e-6; disk-lightglue fp16: the same 1793 pairs, scores within 5e-4.
- ALIKED queues both images' networks (`mx.async_eval`) before the host selects the first one's keypoints. DISK
  does not: its U-Net is all of an image's time, and both sets of dense maps at once cost 2725 MB against 1729
  (fp16; 689 against 697 ms).
- aliked-flexglue: the keypoints come from `MlxAliked` (the old port exported ALIKED's maps for them): fp32
  1224 -> 938 ms, first call 8.8 -> 4.2 s, and float32 is now its faster MLX dtype.

Left:

- disk-lightglue is the U-Net: ~320 ms per image of 3x3 / 5x5 convs at full resolution (conv bound), and its
  export is the first call (1.7 s, already below MPS's 2.2 s). A hand-written U-Net would move the first call
  only; not built.
- aliked-flexglue is not routed to MLX: the first call (4.2 s, the export of FlexGlue's descriptor network and
  decoder) is above the budget of 1.38 + 2 x 1.07 = 3.5 s. Warm it is the descriptor network (not profiled here);
  a hand-written port of it would decide the routing.
- aliked-lightglue float16 equals float32 (ALIKED and `MlxLightGlue` are float32 in either dtype): 102 ms.
- The remaining 100 ms of aliked-lightglue is the two encoders (56 ms), SDDH (18 ms) and LightGlue (the rest,
  ~25 ms); MPS portable was not looked at.
