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
| detection | detectron2 Mask / Faster R-CNN (R50-FPN) | `faster-rcnn-r50-fpn-3x-coco` | size-sibling | float32/float16 | 160 | 3 | fp16 367 ms, fp32 410 ms (hand-written since: 126 / 156 ms, "Done: Mask / Faster R-CNN" below) |
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
| 0.78, 0.84 (were 2.7, 1.7) | aanet-sceneflow, aanet-plus | done, both auto-routed to MLX: hand-written network, grouped offset convolutions split per group ("Done: AANet and CoEx" below) |
| 0.89 (was 2.5) | coex-sceneflow | done, auto-routed to MLX: hand-written network, a quarter of the MPS peak ("Done: AANet and CoEx" below) |
| 0.79, 0.74 (were 2.5, 1.7) | rife-4.25-lite, rife-4.25 | done, all three RIFE refs auto-routed to MLX: hand-written network, the warps and resizes as Metal kernels ("Done: RIFE 4.25 and EMA-VFI" below) |
| 0.63, 0.67 (were 2.2) | loftr-indoor/outdoor | done, both auto-routed to MLX: hand-written matcher, fine matching on the GPU, window attention and layer norm as Metal kernels ("Done: LoFTR" below) |
| 0.49 (was 2.2) | aliked-lightglue | done, auto-routed to MLX: ALIKED in MLX operations without the full-resolution descriptor map, `MlxLightGlue` ("Done: the LightGlue family" below) |
| 2.2, 1.8, 1.7 | efficientvit-sam-l0/xl1/xl0 | the ben2/EfficientViT-SAM agent is on it |
| 0.71, 0.66 (were 2.0, 1.9) | mask-rcnn, faster-rcnn r50-fpn | done, both auto-routed to MLX: hand-written network, RoIAlign as one Metal kernel over all levels ("Done: Mask / Faster R-CNN" below) |
| 0.94 (was 1.8) | realplksr-nomoswebphoto-x4 | kernels fp16 1029 ms vs MPS 1092, but 1919 MB against 1368 MB; portable fp16 is the lean mode (1529 ms, 1198 MB). Left: the 862 MB active peak of the Winograd convs, group norm (3 ms x 28) |
| 0.76, 0.65 (were 1.7) | dlnr-sceneflow, raft-stereo-middlebury | done, all five refs of the two families auto-routed to MLX: hand-written networks, first calls 1.7 / 2.7 s ("Done: DLNR and RAFT-Stereo" below) |
| 0.83 (was 1.7) | ema-vfi-small-t | done, all four EMA-VFI refs auto-routed to MLX: hand-written network, 0.47 s first call ("Done: RIFE 4.25 and EMA-VFI" below) |
| 0.78, 0.34, 0.65, 0.67 (were 1.6, 1.6, 1.4, 1.6) | lineart, anyline, segnext-mscan-b/l | done, all seven refs of the three families auto-routed to MLX: hand-written networks, 0.07-0.22 s first calls ("Done: line art, AnyLine and SegNeXt" below) |
| 0.80, 0.70, 0.70 (were 1.5-1.6) | sapiens-seg-0.3b, oneformer-swinl, gmflow-scale1 | done: all auto-routed to MLX; oneformer and mask2former share a hand-written network (0.9 s first call), gmflow-scale2* and sapiens-seg-1b re-surveyed ("Done: Sapiens, OneFormer, GMFlow" below) |

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
  256 to 96: 4.29 vs 5.11). Done since as a translator rule: see "1x1 convs as a GEMM" below (deblur fp16 726 to
  599 ms).

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
- Left: lower the active peak of the Winograd convs at 4x-upsampled width, group norm (about 3 ms x 28). The
  1x1-as-GEMM rule is off for the kernels port (`gemm1x1=False`): its float16 golden sits on its limit (p99 0.00180
  of 0.0018) and the GEMM's differently rounded result lands at 0.00181, for 1034 to 991 ms.

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

- The other DINOv2-backbone ports that exported: done, see "Depth Anything 3 and Depth Pro without an export"
  below.
- vits fp16's 4% warm gap against the replayed graph: cause not found (none of the single ops above).
- `moge-3-vitl` stays on MPS kernels float32 (2543 against 2415 ms warm).

### Depth Anything 3 and Depth Pro without an export (2026-11-24)

Both families are hand-written on MLX now (`depth_anything_3/mlx_net.py`, `depth_pro/mlx_net.py`), on the shared ViT
blocks of `mlx_vit`; nothing is exported, the weights convert once into the conversion cache. MLX fp16 portable,
994x1482, survey rows (warm ms / first call s / peak MB), MPS = its fp16 portable row:

| ref | MPS | MLX before | MLX now | first-call budget | auto |
|---|---|---|---|---|---|
| da3mono-large | 280 / 0.45 / 2499 | 231 / 2.09 / 1557 | 227 / 0.26 / 1563 | 1.01 | MLX (was MPS) |
| da3metric-large | 280 / 0.44 / 2499 | 231 / 2.02 / 1555 | 227 / 0.26 / 1561 | 1.00 | MLX (was MPS) |
| da3-large-1.1 | 318 / 0.58 / 2520 | 268 / 3.44 / 1763 | 226 / 0.26 / 1594 | 1.22 | MLX (was MPS) |
| da3-giant-1.1 | 759 / 0.91 / 7243 | 647 / 31.7 / 3454 | 606 / 0.68 / 3231 | 2.43 | MLX (was MPS) |
| da3nested-giant-large-1.1 | 1184 / 2.03 / 8136 | 878 / 43.7 / 4123 | 840 / 0.91 / 3895 | 4.40 | MLX (was MPS) |
| depth-pro | 5011 / 5.16 / 7169 | 4653 / 30.9 / 5490 | 4271 / 4.31 / 4905 | 15.2 | MLX (was MPS) |

Float32 rows: mono 288 -> 275 ms (first 2.09 -> 0.31 s, peak 2730 -> 2425 MB), metric 286 -> 275 (2.02 -> 0.32,
2729 -> 2416), large 317 -> 273 (3.42 -> 0.33, 2720 -> 2503), giant 827 -> 751 (34.0 -> 0.95, 5876 -> 5770), nested
1138 -> 1037 (40.6 -> 1.52, 7272 -> 7076), depth-pro 7541 -> 5310 (22.3 -> 5.97, 10529 -> 8712). A second input
shape costs one warm call. Load time rises by the conversion's read (giant fp16 ~4 s, nested ~8 s, depth-pro ~5 s
from the cache on the external disk; the first load of a checkpoint converts and writes the float32 cache file:
nested peaked at 6.7 GB there, depth-pro at 6.9 GB).

Depth Anything 3 (one module for the mono, metric, any-view and nested variants):

- The DPT head is Depth Anything V2's (`_conv`, `_fuse`, `resize` of its `mlx_net`); the any-view backbone adds the
  camera token, the local / global alternation, RoPE and the q/k norm to `mlx_vit.attention` / `block`.
- LayerNorm eps: the blocks' norms are 1e-6, the final, q/k and head norms 1e-5 (`dinov2_layers(norm_eps=)`).
- The q/k norm (64 wide, on every token of every head) through `mx.fast.layer_norm` was the slowest op of the giant:
  written out in float32 (mean, variance, rsqrt) it is faster and the giant's warm time drops below the export's.
- Load peak of the giant: casting the float32 cache to float16 left the originals in MLX's buffer cache; emptying it
  every 64 MB of cast weights keeps the load under the run's peak (2555 MB against 3231 MB).

Depth Pro:

- The 35 windows of the patch encoder are one batch gathered from the three pyramid levels' patch embeddings
  (`net.tables`), the two other ViTs single images; every 1x1 convolution + 2x2 transposed convolution chain is one
  matmul and a depth-to-space. Encoders 3.3 s of the 4.3 s call (ViT-L on 35 x 577 tokens), decoder levels 4-1
  0.2 s, level 0 and head 0.75 s.
- Memory decides the structure of the last stage (768x768 maps, 300 MB each in float16). As one compiled function
  nothing is freed before the end: 7.5 GB over the weights, 7.5 GB footprint. Evaluated map by map with one queued
  ahead (`mx.async_eval`): 6.6 GB MLX peak, because MLX's Winograd holds ~2.4 GB while a 256 -> 256 conv runs and
  two are in flight. One at a time: 5.0 GB, but 4.6 s instead of 4.35 s. The 3x3 convs in bands of rows, two bands
  in flight (fp16, warm ms / footprint MB): 1 band 4340 / 7348, 2: 4250 / 6066, 3: 4200 / 5438, 4: 4240 / 5228,
  6: 4230 / 4994, 8: 4230 / 4940. Six taken (fp32: 4 bands 5310 ms / 9528 MB, 8 bands noisier at 8912 MB).
- The composed head has 136 channels between its two 3x3 convs (128, a constant one, 7 of zeros). At 768x768 fp16
  a 256 -> 136 conv takes 199 ms (144 outputs, what the translator padded to: 143 ms), 256 -> 160 73 ms (Winograd),
  136 -> 128 54 ms, 160 -> 128 44 ms: the weights are zero-padded to 160 at conversion. Goldens pass in both dtypes.

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

### Done: AANet and CoEx (2026-11-24)

M1 Pro, 994x1482, all six MLX goldens pass (`tools/speed_survey.py`; MLX/MPS above is against the best MPS row):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| aanet-sceneflow | MLX before, fp16 / fp32 | 1695 / 1724 | 19.8 / 18.3 | 2131 / 1777 | – |
| | MLX now, fp16 / fp32 | 492 / 692 | 0.51 / 0.71 | 1362 / 2140 | MLX |
| | MPS portable, fp16 / fp32 | 723 / 880 | 1.09 / 1.25 | 2594 / 3622 | |
| | MPS kernels fp16 (best MPS row) | 631 | 1.23 | 2606 | |
| aanet-plus-sceneflow | MLX before, fp16 / fp32 | 1367 / 1898 | 10.4 / 13.4 | 2989 / 5038 | – |
| | MLX now, fp16 / fp32 | 687 / 925 | 0.72 / 0.96 | 2259 / 3104 | MLX |
| | MPS portable, fp16 / fp32 | 936 / 992 | 1.47 / 1.50 | 3836 / 5109 | |
| | MPS kernels fp16 (best MPS row) | 818 | 1.64 | 3754 | |
| coex-sceneflow | MLX before, fp16 / fp32 | 608 / 765 | 5.69 / 5.87 | 3187 / 2682 | – |
| | MLX now, fp16 / fp32 | 215 / 282 | 0.25 / 0.30 | 947 / 1351 | MLX |
| | MPS portable, fp16 / fp32 (best MPS row) | 241 / 323 | 0.78 / 1.49 | 3631 / 6428 | |

Both families replayed an exported graph before: the export was 5-20 s of the first call, and the graph kept
upstream's layouts. Now each network is written out in `mlx.core`, channels last, batch norms folded, one compiled
function (`aanet/mlx_net.py`, `coex/mlx_net.py`), built from the shared kernels (`gwc_volume`, `grid_sample`,
`bilinear_nhwc`, `depthwise`, `mlx_graph`'s phased transposed 3-D convolution):

- both images go through the feature extractor as one batch of two; the correlation is `gwc_volume` with one group.
- AANet's disparities are the channels of the aggregation, so nothing is transposed between its 2-D convolutions. A
  modulated deformable convolution is `grid_sample` on a `[pixels, 9 taps]` grid per offset group, the mask, and one
  matrix product per group (no concatenated column matrix: at full resolution in AANet+'s refinement that copy was
  0.9 GB; 758 -> 687 ms, AANet 520 -> 492 ms).
- MLX's grouped convolution is several times slower than one convolution per group, and its first call costs
  seconds. The offset convolutions of the deformable layers have 2 groups (64 -> 54 channels at 336x496: 23 ms
  grouped, 2 x 4 ms split; 32 -> 54 at 1056x1536 in AANet+'s refinement: 115 ms against 2 x 15 ms). Splitting them
  took AANet from 576 to 520 ms and AANet+ from 930 to 758 ms, and the first calls from 3.5 / 5.4 s to 0.5 / 0.8 s.
- CoEx: the 3-D stem (1 -> 8 channels) is one 2-D convolution with its depth taps as channels, the top-2
  regression two arg-max passes over the last axis, the superpixel upsampling a broadcast over the 4 x 4 cells.
  MobileNetV2's depthwise layers as the `depthwise` kernel or as `mx.conv2d(groups=C)` measured the same (214
  against 218 ms).

Where AANet's 492 ms go (staged, uncompiled 680 ms): features 180, correlation 35, the six aggregation modules 205
(the three with deformable convolutions 55 each), the 1/2 refinement 54, the full-resolution refinement 204 (twelve
32 -> 32 3 x 3 convolutions at 1008x1488, 11 ms each). AANet+: features 135, aggregation 225, refinements 125 and
420 (the hourglass at full resolution).

Tried, no gain: without `mx.compile` (AANet 676 ms, AANet+ 986 ms, CoEx 249 ms; the compile itself costs nothing
measurable on the first call once the grouped convolutions are gone).

Left: both AANets are bound by plain 3 x 3 convolutions at full resolution (MLX's are at MPSGraph's speed there);
no kernels mode on MLX for either family.

### Done: DLNR and RAFT-Stereo (2026-11-24)

M1 Pro, 994x1482, all five MLX goldens pass in every mode and dtype (`tools/speed_survey.py`; "before" is the exported
graph, re-measured the same day):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| dlnr-sceneflow | MLX before, fp16 / fp32 | 2039 / 3341 | 18.7 / 25.1 | 2613 / 3752 | – |
| | MLX now, fp16 / fp32 | 1662 / 2276 | 1.75 / 2.33 | 2844 / 3849 | MLX fp16 |
| | MPS portable, fp16 (best MPS row) / fp32 | 2195 / 2751 | 2.65 / 2.88 | 4146 / 5170 | |
| dlnr-middlebury | MLX before, fp16 / fp32 | 2052 / 3213 | 16.8 / 21.2 | 2609 / 3766 | – |
| | MLX now, fp16 / fp32 | 1711 / 2234 | 1.74 / 2.31 | 2841 / 3856 | MLX fp16 |
| | MPS portable, fp16 (best MPS row) / fp32 | 2143 / 2693 | 2.55 / 2.87 | 4149 / 5169 | |
| raft-stereo-middlebury | MLX before, portable fp16 / fp32 | 4076 / 5040 | 8.59 / 9.71 | 3724 / 4269 | – |
| | MLX before, kernels fp16 / fp32 | 2579 / 4367 | 7.33 / 8.97 | 4242 / 5224 | |
| | MLX now, portable fp16 / fp32 | 3249 / 4944 | 3.28 / 4.96 | 3286 / 4165 | |
| | MLX now, kernels fp16 / fp32 | 2370 / 4193 | 2.72 / 4.65 | 3898 / 5361 | MLX kernels fp16 |
| | MPS portable, fp16 / fp32 | 3940 / 4396 | 3.98 / 4.43 | 4695 / 4712 | |
| | MPS kernels fp16 (best MPS row) | 3634 | 3.73 | 4720 | |
| raft-stereo-sceneflow | MLX before, kernels fp16 / fp32 | 2798 / 4662 | 8.01 / 11.7 | 4613 / 5837 | MLX kernels fp16 |
| | MLX now, portable fp16 / fp32 | 3250 / 4958 | 3.27 / 4.97 | 3284 / 4168 | |
| | MLX now, kernels fp16 / fp32 | 2361 / 4162 | 2.73 / 4.62 | 3874 / 5357 | MLX kernels fp16 |
| raft-stereo-eth3d | MLX before, kernels fp16 / fp32 | 2772 / 4521 | 8.94 / 9.74 | 4233 / 5318 | MLX kernels fp16 |
| | MLX now, portable fp16 / fp32 | 3239 / 4966 | 3.30 / 5.00 | 3296 / 4163 | |
| | MLX now, kernels fp16 / fp32 | 2361 / 4194 | 2.77 / 4.74 | 3875 / 5358 | MLX kernels fp16 |

The MPS rows of the two RAFT-Stereo siblings equal middlebury's (kernels fp16 3637 / 3638 ms). Both families replayed
an exported graph before; the export was most of the first call. Now each is written out in `mlx.core`, channels
last (`dlnr/mlx_net.py`, `raft_stereo/mlx_net.py`), built from the shared `graph_mlx` kernels, and the update step
is one compiled function:

- the 1-D correlation volume is one float32 matrix product per row, its pyramid lookup the `linear_sample` kernel.
- the gates of a recurrent cell are one convolution (DLNR's LSTM as `prepare` merges them; RAFT-Stereo's z and r),
  and the gate biases are added into the context once, before the loop. The motion encoder's last convolution has
  128 outputs instead of 126 (two zero filters, the flow added into their place), the flow head only its x output,
  and the upsampling mask is computed once, after the last step.
- DLNR's Restormer blocks: `row_layer_norm`, the 1x1 convolutions as matrix products, `depthwise`, `token_gram`
  for the channel attention, `dw_gate` for the feed-forward. The normalization refinement stays float32.
- RAFT-Stereo's feature encoder runs one image at a time with `channel_stats` instance norms (convolution biases
  before a norm dropped); the context encoder's batch norms are folded. In kernels mode the stride-1 3x3
  convolutions are `conv_mlx` with bias and ReLU fused.
- RAFT-Stereo memory: evaluating after every residual block and a 1.5 GB buffer cache instead of 2 GB took kernels
  fp16 from 4516 to 3873 MB at 2278 -> 2371 ms; a 1 GB cache gave 3483 MB but 2499 ms (portable fp32 5409 ms).

Where DLNR's 1679 ms go (fp16, staged): the extractor 486 (both images), the volume features 71, the ten update
steps 708, the upsampling 27, the refinement 333 (float32, full resolution). fp32: 665 / 111 / 1069 / 35 / 385.

Left:

- RAFT-Stereo kernels fp32 peaks at 5360 MB, above MPS (4712 MB); the Winograd buffers of the full-resolution
  64-channel layers. fp32 warm is behind MPS portable fp32 in portable mode (4944 against 4396 ms).
- DLNR has no kernels mode on MLX; its refinement could be tried in float16.
- RAFT-Stereo's stages were not profiled; padding the 96-channel layers to 128 for MLX's own Winograd in portable
  mode was not tried.
- `tests/test_policy.py --slow` (`test_card_kernels_match_imports[stereo]`) already fails on AANet's undeclared
  `graph_mlx` import; both new modules import `graph_mlx` the same way.

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

### Done: RIFE 4.25 and EMA-VFI (2026-11-24)

M1 Pro, 1080x1920, float16 / float32, all 14 goldens pass (`tools/speed_survey.py`; MPS rows are the portable ones,
the best MPS row of every ref):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| rife-4.25 | MLX before (kernels) | 180 / 223 | 3.45 / 4.54 | 1655 / 1743 | – |
| | MLX now | 78.5 / 91.8 | 0.09 / 0.11 | 921 / 1336 | MLX fp16 |
| | MPS | 106 / 119 | 0.61 / 0.39 | 2482 / 2481 | |
| rife-4.26 | MLX before (kernels) | 137 / 226 | 3.08 / 3.13 | 1379 / 1794 | – |
| | MLX now | 78.3 / 92.2 | 0.09 / 0.11 | 923 / 1337 | MLX fp16 |
| | MPS | 103 / 118 | 0.31 / 0.31 | 2479 / 2478 | |
| rife-4.25-lite | MLX before | 202 / 200 | 3.27 / 4.33 | 1636 / 1590 | – |
| | MLX now | 63.3 / 73.3 | 0.08 / 0.10 | 935 / 1294 | MLX fp16 |
| | MPS | 80.0 / 91.0 | 0.46 / 0.29 | 2447 / 2477 | |
| ema-vfi | MLX before | 2255 / 3362 | 7.74 / 9.28 | 2770 / 4128 | MLX for fp32 only |
| | MLX now | 1320 / 1741 | 4.24 / 1.86 | 3386 / 4648 | MLX fp16 |
| | MPS | 1981 / 19632 | 2.47 / 3.99 | 6883 / 11607 | |
| ema-vfi-t | MLX before | 2222 / 3305 | 5.48 / 8.61 | 2700 / 4129 | MLX fp16 |
| | MLX now | 1325 / 1713 | 1.37 / 1.83 | 3463 / 4922 | MLX fp16 |
| | MPS | 2272 / 28800 | 2.38 / 5.09 | 6882 / 10777 | |
| ema-vfi-small | MLX before | 666 / 980 | 8.32 / 6.36 | 1587 / 2034 | – |
| | MLX now | 426 / 538 | 0.47 / 0.57 | 1930 / 2881 | MLX fp16 |
| | MPS | 556 / 736 | 1.57 / 2.31 | 3587 / 6667 | |
| ema-vfi-small-t | MLX before | 876 / 1019 | 10.3 / 12.7 | 2460 / 2396 | – |
| | MLX now | 427 / 539 | 0.47 / 0.57 | 1932 / 3007 | MLX fp16 |
| | MPS | 513 / 792 | 1.02 / 2.03 | 3583 / 6499 | |

ema-vfi's 4.24 s is the first row of the survey run (it converted the weights into the cache); ema-vfi-t, the same
network, shows the first call with the cache filled.

Both families replayed an exported graph (`mlx_wrap`); RIFE also had a `kernels` mode on MLX (`conv_mlx`
convolutions in that graph). Both are hand-written now (`rife/mlx_net.py`, `ema_vfi/mlx_net.py`): no export, the
weights converted once into the conversion cache, one compiled function per shape. RIFE's MLX `kernels` mode is gone
(the card declares portable only, its four survey rows are dropped); the export-only branches of `rife/model.py`
(`EXPORTING_FOR_MLX`, the `cat` instead of in-place writes) went with it.

- A transposed convolution (4x4, stride 2, padding 1) is one 2x2 convolution with four times the outputs: phase `a`
  of the output is the 2x2 convolution's result at `j + a`. `mx.conv_transpose2d` is 3-4x slower (EMA-VFI's U-Net
  at 1088 x 1920 fp16: 76 against 106 to 156 ms each). The result is never interleaved by MLX operations: a Metal
  kernel reads the phase layout.
- RIFE (`_block_input`, `_blend`): one kernel writes an IFBlock's whole input at the block's resolution (both
  warped images and feature maps as the 2x2 means a bilinear 1/s downscale reads, the last block's mask and
  features, the flow). The flow at a pixel is the sum of the earlier blocks' low-resolution outputs, upsampled in
  the kernel in float32: no full-resolution flow field, warped frame or resized copy is stored between the blocks,
  which is where the memory went (921 MB against 1655). Ragged widths are zero-padded (15 / 28 input channels to
  16 / 32: a stride-2 convolution of 28 channels at 1152 x 1920 takes 7.8 ms, of 32 4.2 ms).
- EMA-VFI (`_stack`, `_up`): `_stack` writes a U-Net block's whole input (both frames' backbone features warped by
  the block mean of the flow, the block before it, zeros up to a width MLX's convolution is fast at: 81 -> 96
  channels 50 -> 31 ms for the first one) and warps the two images; `_up` interleaves the phases of a transposed
  convolution, adds the bias, applies the PReLU and appends the skip connection. The window attention is plain
  matmuls on `[windows, heads, 49, .]`, mask and position windows constants of the compiled function. `tta` runs
  its two passes one after the other.
- A Metal kernel thread per pixel that loops over the channels is 3-10x slower than a thread per four channels
  writing one `packed_vec<T, 4>` (grid z = channels / 4, thread group 8 x 8 x 4). The `_up` interleave of
  1088 x 1920 x 32 fp16: 30.7 ms with the scalar loop, 8.9 ms with a `vec4` loop per pixel, 2.8 ms with a thread
  per four channels. ema-vfi-small fp16: 666 ms exported, 554 ms hand-written with per-pixel kernels, 426 ms with
  these (ema-vfi 2255 -> 1578 -> 1320).
- Kernel inputs of fewer than 8 elements are bound in Metal's `constant` address space, so the size vector and the
  placeholders for absent inputs have at least 8.

Against the CPU portable runner at 1080x1920: float32 within 3e-5 to 9e-5 (max), float16 mean 2e-5 to 9e-5; `t`
other than 0.5 on the `-t` refs and rife-4.26, odd sizes (270x500) and `tta` (480x640) were compared too.

Where ema-vfi-small's time goes (fp16, the pieces timed one by one, 414 ms in all): the backbone's first
convolutions 48, its blocks 77, the first flow head 43, the U-Net's stride-2 convolutions ~86, its four 2x2 phase
convolutions 4 x 18.5, the `_stack` kernels ~31, the `_up` kernels ~8 (were 81), the last convolution 8.

Left:

- EMA-VFI's peak is above the exported graph's (ema-vfi fp16 3386 against 2770 MB, small 1930 against 1587; half
  of MPS). The phase layout of the transposed convolutions (four times the channels until `_up` reads it) and the
  padded `_stack` outputs are the candidates; not profiled.
- EMA-VFI is convolution bound now. The 2x2 phase convolutions and the stride-2 ones are not
  MLX's Winograd path; `conv_mlx` was not tried on them.
- RIFE's two kernels are still a thread per pixel (the work there is the flow accumulation, not the channel
  writes); the thread-per-four-channels layout was not tried on them.
- The MPS runners were not touched. The 2x2 phase form of the transposed convolutions and the zero-padded widths
  are plain tensor code and could be tried there; ema-vfi / ema-vfi-t fp32 on MPS (19.6 / 28.8 s, 11 GB) swap on
  16 GB.

### Done: Mask / Faster R-CNN (2026-11-24)

M1 Pro, 800x1344, float16 / float32, both goldens pass in both dtypes (`tools/speed_survey.py`, synthetic input; MPS
rows are the portable ones). `detection/rcnn` is the family this covers:

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| faster-rcnn-r50-fpn-3x-coco | MLX before | 367 / 410 | 0.94 / 0.96 | 1694 / 1919 | – |
| | MLX now | 126 / 156 | 0.14 / 0.19 | 1156 / 1576 | MLX fp16 |
| | MPS | 192 / 214 | 0.44 / 0.42 | 1932 / 1930 | |
| mask-rcnn-r50-fpn-3x-coco | MLX before | 411 / 462 | 1.50 / 1.83 | 1533 / 2012 | – |
| | MLX now | 144 / 181 | 0.16 / 0.20 | 1174 / 1600 | MLX fp16 |
| | MPS | 203 / 222 | 0.53 / 0.54 | 1931 / 1930 | |

The MLX runner replayed an exported graph of the dense half (`mlx_wrap`) and ran the proposals, both RoIAligns and
the heads in torch. It is hand-written now (`rcnn/mlx_net.py`, `MlxPortable` in `rcnn/model.py`): no export, the
weights converted once into the conversion cache, one compiled function per shape for backbone, FPN and RPN.

- RoIAlign (`roi_align`) is one Metal kernel over all four levels: a thread writes four channels of one bin of one
  box and reads the box's level in place (grid: channels / 4, bins, boxes), so there is no per-level gather,
  scatter or sort. torchvision's `aligned=True`, adaptive sampling ratio; the sum is accumulated in float32.
  Against `torchvision.ops.roi_align` on 500 boxes over the four levels: 2.4e-7 max in float32, 2e-3 in float16.
  The 1000 x 7 x 7 x 256 box pooling with both linear layers takes 11 ms, the mask pooling and head 6 ms.
- The dense half is MLX operations: 1x1 convolutions as `mx.addmm`, the max pool as a strided view, the mask
  head's 2x2 stride-2 transposed convolution as a GEMM to four phases; `fc1`'s columns are reordered at conversion
  so the pooled `[R, 7, 7, C]` is flattened as it is. The mask predictor runs for the predicted class only.
- Top-k per level, the box decoding and both NMS stay on the CPU with torchvision (`_Select`, shared with the torch
  runners so the selection is the same code): 9 ms. The compiled function returns the objectness and deltas of all
  levels as two float32 arrays, so that is two reads.
- Kernel inputs of fewer than 8 elements are bound in Metal's `constant` address space; the per-level size table
  is padded.

Against the CPU portable runner on the golden image (400x600, 4 detections): mask-rcnn float32 boxes 1e-4 px,
scores 2e-6, no mask pixel differs; float16 boxes 0.5 px, scores 4e-3, 95 of 284658 mask pixels; faster-rcnn
float16 boxes 3.0 px, scores 0.024; labels equal everywhere.

Where mask-rcnn's time goes (fp16, 800x1344, 16 detections, 149 ms in all): the dense function 90, top-k and RPN
NMS 6, box pooling and head 11, the class NMS 3, mask pooling and head 6, pasting the masks into the image on the
CPU 22. float32: 182 ms, 118 of them dense.

Left:

- The network is convolution bound (60% is ResNet-50 + FPN + RPN); `conv_mlx` and a `kernels` mode were not tried,
  the card declares portable only.
- Pasting the masks (`_output`, 22 ms for 16 detections, on the CPU) is the next piece; a Metal kernel that writes
  all masks would take it. It grows with the detection count.
- Top-k and NMS on the CPU cost 9 ms and two syncs; not ported.
- The channels per RoIAlign thread (4) and the thread group (8 x 8 x 1) were not tuned.
- The MPS runner was not touched.
- RoIAlign inside other families (sam3) was not examined.

### Done: line art, AnyLine and SegNeXt (2026-11-24)

M1 Pro, float16, 1024x1024 (edges) and 1024x2048 (SegNeXt), all 28 goldens pass (`tools/speed_survey.py`, 7 refs x
2 modes x 2 dtypes; "MLX now" is the routed row):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| lineart | MLX before | 108.2 | 2.27 | 1099 | – |
| | MLX now | 51.5 | 0.11 | 883 | MLX |
| | MPS portable (best MPS row) | 66.5 | 0.24 | 1362 | |
| lineart-coarse | MLX before | 78.9 | 1.93 | 933 | – |
| | MLX now | 51.8 | 0.07 | 867 | MLX |
| | MPS portable (best MPS row) | 68.2 | 0.30 | 1364 | |
| anyline | MLX before | 181.2 | 2.16 | 1604 | – |
| | MLX now | 39.8 | 0.07 | 731 | MLX |
| | MPS portable (best MPS row) | 116.8 | 0.34 | 1319 | |
| segnext-mscan-b-ade20k | MLX before | 263.7 | 3.95 | 894 | – |
| | MLX now | 118.9 | 0.15 | 940 | MLX |
| | MPS portable (best MPS row) | 184.1 | 0.42 | 1659 | |
| segnext-mscan-l-ade20k | MLX before | 424.1 | 5.05 | 974 | – |
| | MLX now | 180.8 | 0.22 | 1140 | MLX |
| | MPS portable (best MPS row) | 268.2 | 0.48 | 1840 | |
| segnext-mscan-s-ade20k | MLX before | 74.9 | 1.66 | 786 | – |
| | MLX now | 81.4 | 0.11 | 828 | MLX |
| | MPS portable (best MPS row) | 134.4 | 0.34 | 1556 | |
| segnext-mscan-t-ade20k | MLX before | 67.7 | 2.35 | 754 | – |
| | MLX now | 69.7 | 0.10 | 745 | MLX |
| | MPS portable (best MPS row) | 117.5 | 0.88 | 1445 | |

float32 now: lineart 77.5 ms / 952 MB (was 134.0 / 1244; MPS 95.8 / 1361), anyline 55.4 / 888 (was 218.5 / 1997;
MPS 129.7 / 2346), segnext-b 126.1 / 1087 (was 300.6 / 1048; MPS 206.5 / 1733), segnext-l 196.2 / 1248 (was
488.9 / 1203; MPS 298.9 / 1943), first calls 0.08 to 0.26 s.

The "before" rows of lineart and SegNeXt b / l were stale: re-surveyed at the start of this pass, the exported
graphs already ran at 71.7 (lineart), 174.8 (anyline), 111.5 (segnext-b) and 169.6 ms (segnext-l) with 836 to
1396 MB, and all that kept them off MLX was the first call (1.0 to 3.0 s against budgets of 0.38 to 1.0 s). So the
three families got hand-written networks: no export, and the first call is the Metal shader builds only.

- `segnext/mlx_net.py`: the folded network in `mlx.core`. Every depthwise convolution is a Metal kernel (the 5x5
  and the MLP's 3x3 with its GELU fused, the 1x7 / 7x1 to 1x21 / 21x1 strips as row and column passes), the 1x1
  convolutions GEMMs with the bias (`mlx_graph.conv1x1`), GELU, layer norm and the head's matrix factorisation in
  float32. float32 is 8 to 13% faster than the re-surveyed exported graph on b / l (126.1 against 136.5 ms, 196.2
  against 220.8); float16 is 3% behind it on t (69.7 against 67.7 ms) and 7 to 9% on s, b and l (see "Left").
- `lineart/mlx_net.py`: the two stride-2 transposed convolutions as four phase convolutions (1x1, 1x2, 2x1, 2x2)
  interleaved; instance norm statistics in one float32 pass (`graph_mlx.channel_stats`) with the normalisation, ReLU
  and cast in one expression; the biases in front of a norm dropped; the 7x7 head (64 -> 1 channel) as a 1x7
  convolution to seven channels plus the sum of seven shifted rows (8.2 -> 1.6 ms, the call 59.8 -> 51.5 ms).
- `shared/teed/mlx_net.py` + `anyline/mlx_net.py`: MTEED is 58 K weights and almost all element-wise work at
  1280 px. Smish is one exponential per element in a kernel; `DoubleFusion` (24 channels at full size) is two
  kernels that never store the 24-channel maps. The line art, which was 85 of 165 ms on the CPU twin, is three
  kernels: the separable blur, `min_c(blur - x)`, and threshold + speck removal + blend in one pass (every drawn
  pixel walks its component until it has 12 pixels), with the median as `np.partition` on the host. The network and
  line art take 25 ms of the 40; the rest is the cubic resize to the detect size on the CPU (9 ms) and transfers.
  The float32 result equals the CPU portable runner's map.
- lineart gained a kernels mode on MLX (the same runner; it uses `graph_mlx`), SegNeXt b / l lost their
  `kernels float16/float32` waivers: both modes of all seven refs have passing rows.

First calls depend on the system's Metal shader cache: the very first run of a new kernel source on a machine costs
about 1 s per family and dtype (anyline 1.1 s, lineart float32 1.1 s, segnext-b float16 2.5 s); every later
process, as surveyed above, starts in 0.07 to 0.26 s. The survey was run twice and the second run recorded.

Tried, no gain: an own Winograd F(4,3) kernel for the 64- to 256-channel 3x3s of lineart (within 0.4 ms per
convolution of `mx.conv2d`); a transposed convolution as one 2x2 convolution to 4x the channels plus depth-to-space
(slower than the four phase convolutions); the lineart head with its output zero-padded to 4 or 8 channels (8.3
against 8.2 ms; 16: 16.2 ms) or in float32 (9.2 ms).

Left:

- SegNeXt float16 on s / b / l is 7 to 9% slower and 40 to 230 MB higher than the exported graph was (81.4
  against 75.0 ms, 118.9 against 111.5, 180.8 against 169.6). The export paid 1.6 to 3.0 s of first call for it,
  which the auto budget does not allow; where the hand-written float16 path loses the time was not profiled.
- lineart: the eleven instance norms and the 256-channel 3x3s of the residual blocks are what is left in the
  network; `to_detect` is 5.5 ms on the CPU.
- anyline: the cubic resize to the detect size on the CPU (9 ms) is the largest single step now.
- sapiens-seg-0.3b, oneformer-swinl and gmflow-scale1 of the same queue row: see the next section.

### Done: Sapiens, OneFormer, GMFlow (2026-11-25)

M1 Pro, `tools/speed_survey.py` run twice and the second run recorded, all goldens pass. "MLX now" is the routed
row; best MPS is the fastest MPS row of either mode:

| ref | dtype | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|---|
| sapiens-seg-0.3b-goliath | float16 | MLX before | 2392 | 7.32 | 3921 | – |
| | | MLX now | 1214 | 1.29 | 2967 | MLX |
| | | best MPS | 1517 | 1.59 | 3748 | |
| | float32 | MLX before | 2664 | 13.26 | 5271 | – |
| | | MLX now | 1485 | 1.98 | 4545 | MLX |
| | | best MPS | 2988 | 3.06 | 4766 | |
| oneformer-swinl-ade20k | float16 | MLX before | 1441 | 6.62 | 5416 | – |
| | | MLX export | 961 | 8.12 | 2265 | MPS (first call) |
| | | MLX now | 812 | 0.88 | 3354 | MLX |
| | | best MPS | 1161 | 1.38 | 5559 | |
| | float32 | MLX before | 1588 | 7.56 | 6397 | – |
| | | MLX export | 1109 | 7.49 | 2802 | MPS (first call) |
| | | MLX now | 919 | 1.01 | 3838 | MLX |
| | | best MPS | 1127 | 1.49 | 9492 | |
| mask2former-swinl-ade20k | float16 | MLX before | 850 | 6.12 | 3614 | MPS (first call) |
| | | MLX now | 596 | 0.65 | 2907 | MLX |
| | | best MPS | 964 | 1.17 | 4503 | |
| | float32 | MLX before | 1033 | 6.50 | 4310 | MPS (first call) |
| | | MLX now | 672 | 0.72 | 3451 | MLX |
| | | best MPS | 909 | 1.21 | 6661 | |
| gmflow-scale1-things | float16 | MLX before | 3871 | 6.06 | 9100 | – |
| | | MLX now | 1934 | 3.51 | 1698 | MLX |
| | | best MPS | 2756 | 2.90 | 7112 | |
| | float32 | MLX before | 5909 | 8.08 | 8885 | – |
| | | MLX now | 2627 | 4.53 | 1706 | MLX |
| | | best MPS | 3780 | 4.05 | 8183 | |
| gmflow-scale2-mixdata | float16 | MLX before | 8276 | 12.78 | 11508 | – |
| | | MLX now | 4097 | 6.57 | 6369 | MLX |
| | | best MPS | 6039 | 5.90 | 9667 | |
| | float32 | MLX before | 12373 | 18.85 | 11132 | – |
| | | MLX now | 5903 | 8.84 | 6408 | MLX |
| | | best MPS | 8494 | 8.34 | 10716 | |
| gmflow-scale2-sintel | float16 | MLX before | 8283 | 12.25 | 11046 | – |
| | | MLX now | 4115 | 6.51 | 6357 | MLX |
| | | best MPS | 5725 | 5.89 | 9669 | |
| | float32 | MLX before | 13073 | 17.52 | 11100 | – |
| | | MLX now | 5917 | 8.64 | 6446 | MLX |
| | | best MPS | 8697 | 8.28 | 11737 | |
| gmflow-scale2-things | float16 | MLX before | 8261 | 12.74 | 11819 | – |
| | | MLX now | 4146 | 6.66 | 6363 | MLX |
| | | best MPS | 5982 | 6.28 | 9577 | |
| | float32 | MLX before | 13356 | 18.19 | 11126 | – |
| | | MLX now | 5932 | 8.44 | 6408 | MLX |
| | | best MPS | 8489 | 8.25 | 10738 | |
| gmflow-scale2-regrefine6-kitti | float16 | MLX before | 11251 | 16.38 | 11324 | – |
| | | MLX now | 7413 | 10.92 | 6732 | MLX |
| | | best MPS | 10998 | 10.95 | 9696 | |
| | float32 | MLX before | 16667 | 23.48 | 11033 | – |
| | | MLX now | 9992 | 13.30 | 6724 | MLX |
| | | best MPS | 14645 | 14.19 | 11791 | |
| gmflow-scale2-regrefine6-mixdata | float16 | MLX before | 11296 | 16.11 | 10699 | – |
| | | MLX now | 7455 | 10.73 | 6636 | MLX |
| | | best MPS | 11156 | 11.01 | 9694 | |
| | float32 | MLX before | 16974 | 22.77 | 11313 | – |
| | | MLX now | 10070 | 13.20 | 6754 | MLX |
| | | best MPS | 14901 | 14.09 | 10778 | |
| gmflow-scale2-regrefine6-sintelft | float16 | MLX before | 10975 | 15.59 | 12341 | – |
| | | MLX now | 7371 | 10.48 | 6700 | MLX |
| | | best MPS | 10867 | 10.99 | 9689 | |
| | float32 | MLX before | 16766 | 22.10 | 10885 | – |
| | | MLX now | 9905 | 13.06 | 6777 | MLX |
| | | best MPS | 14622 | 14.16 | 10775 | |
| gmflow-scale2-regrefine6-things | float16 | MLX before | 11280 | 15.47 | 10551 | – |
| | | MLX now | 7460 | 11.10 | 6663 | MLX |
| | | best MPS | 11300 | 11.28 | 9692 | |
| | float32 | MLX before | 16556 | 21.87 | 10591 | – |
| | | MLX now | 10122 | 13.19 | 6770 | MLX |
| | | best MPS | 14869 | 14.11 | 11645 | |
| sapiens-seg-1b-goliath | float16 | MLX before | 4110 | 36.67 | 6692 | – |
| | | MLX now | 2679 | 2.81 | 4541 | MLX |
| | | best MPS | 3113 | 3.24 | 6340 | |
| | float32 | MLX before | 5247 | 45.76 | 8762 | – |
| | | MLX now | 3594 | 3.76 | 7747 | MLX |
| | | best MPS | 5099 | 5.25 | 9794 | |

"MLX export" is the exported graph with the deformable-attention custom op (an intermediate state of this work);
"MLX now" for OneFormer / Mask2Former is the hand-written network. The gmflow-scale2 and sapiens-seg-1b "MLX now"
rows need no code change beyond the region-order twin and the hand-written Sapiens network below.

- Sapiens (`sapiens/mlx_net.py`): a hand-written network, so nothing is exported and the first call is the shader
  builds only. The ViT is `mlx_vit.patches` + `mlx_vit.block` compiled as one function. The decoder's 4x4
  stride-2 transposed convolutions are four phase GEMMs, interleaved. InstanceNorm + SiLU is one Metal kernel on the
  `channel_stats` statistics, and the 1x1 convolutions are `mx.addmm`. The labels come from
  `mlx_seg.resize_argmax`. sapiens-seg-1b shares the runner.
- OneFormer / Mask2Former (`shared/maskformer/mlx_net.py`, `MlxMaskFormer`): one hand-written network for both,
  so nothing is exported and the first call is the shader builds only (0.65-1.0 s against 6-8 s exported). Swin-L:
  each layer is one Metal kernel for LayerNorm + the gather into (shifted) windows by a cached index table, the
  window attention (`ops.window_attention` on the fused qkv in float16: the relative-position bias for all windows,
  the shift mask added only for the border windows, which come last; SDPA with an additive bias in float32), one Metal kernel
  for the inverse gather + residual + LayerNorm, and the GELU MLP. The pixel decoder's six deformable-attention
  layers use one GEMM for the offsets and weights, the softmax and sampling locations in float32 and `ops.msda`;
  GroupNorm is `ops.channel_stats` with a float32 affine. The transformer decoder computes the attention masks per
  level from the bilinearly resized mask features and passes a boolean mask to SDPA. OneFormer's task token (once) and
  query transformer run in float32 (the float32-stage trap does not apply: nothing is replayed).
  Inference stays `mlx_seg.mask_sigmoid_rows` + a GEMM with the class probabilities, the labels
  `mlx_seg.resize_argmax`. The custom op `fastforks::msda` and `shared/maskformer/mlx_ops.py` are gone.
- GMFlow (`_MlxTwin`, used by every GMFlow ref on MLX): the shifted-window Swin layers run in region order. The
  token permutations into plain and shifted windows are constants, built outside the export's fake mode
  (`unset_fake_temporarily`, because the lru cache would otherwise keep fake tensors). Each layer is one
  `scaled_dot_product_attention` per window group instead of a masked attention over all windows. The global
  matching attention pads the value to the query width so it also is one SDPA. Peak memory is 1.7 GB against 7-8 GB
  on MPS.

Left:

- OneFormer / Mask2Former peak is 0.6-1.1 GB above the exported graph (3.3-3.8 GB against 2.3-2.8 GB), still 35-60%
  below MPS. Part of it is the 1 GB MLX buffer cache the runner requests (`CACHE_MB`); lowering it was not tried.
- gmflow-scale2 / regrefine6: the first call is 0.5-1.3 s above MPS's (well inside the auto budget); peak memory is
  6.4-6.8 GB, the 1080x1920 global matching volume. Not profiled further.

### Done: LoFTR (2026-11-25)

M1 Pro, 768x1024 (matched at long side 840), all four MLX goldens pass (`tools/speed_survey.py`; MLX/MPS above is
the float16 row against the best MPS row):

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| loftr-indoor | MLX before, fp16 / fp32 | 1549 / 1806 | 3.60 / 3.98 | 3675 / 4296 | – |
| | MLX now, fp16 / fp32 | 442 / 1101 | 0.50 / 0.70 | 2129 / 3312 | MLX fp16 |
| | MPS portable, fp16 / fp32 (best MPS row fp16) | 701 / 877 | 0.83 / 1.03 | 3613 / 5768 | |
| loftr-outdoor | MLX before, fp16 / fp32 | 1526 / 1788 | 3.58 / 4.71 | 3703 / 4095 | – |
| | MLX now, fp16 / fp32 | 483 / 997 | 0.56 / 0.91 | 2228 / 2809 | MLX fp16 |
| | MPS portable, fp16 / fp32 (best MPS row fp16) | 723 / 875 | 0.86 / 1.04 | 3614 / 5767 | |

The machine was on 6 % battery while charging and swapping during these surveys: the same row came out anywhere
from 430 to 950 ms (fp16) and 620 to 1270 ms (fp32) between runs, so the recorded rows are a fast reading for fp16 and
a slow one for fp32 (fp32 measured 619 / 632 / 641 ms in the fast runs, which would beat MPS too). A same-process
control (MPS portable fp16 and MLX fp16 loaded together, alternating) gave MPS 1216-1381 ms against MLX 570-597 ms.
Re-survey both on a quiet, charged machine; only fp16 is auto-routed until then.

Before, `auto` stayed on MPS: the MLX port replayed an exported graph (3.6 s first call) and the fine matching ran in
PyTorch on the CPU (about 1 s of the call). Now the whole matcher is written in `mlx.core` (`loftr/mlx_net.py`),
nothing exported:

- `_dense`, compiled per pair of image sizes: the ResNet-FPN (batch norm folded, channels last, the 196-channel maps
  widened to 224 so MLX's Winograd 3x3 takes them), the coarse transformer, and the dual-softmax as
  `exp(2 s - logsumexp_rows - logsumexp_columns)` with every reduction along the last axis (the column statistics
  come from the transposed product); "mutual maximum" is the two argmaxes agreeing.
- one host read per call (which cells matched), then `_refine`, compiled per bucket of 512 matches (ids padded,
  the padding's offsets dropped, so the match count does not recompile): window gather, `FinePreprocess`, the fine
  transformer with both images' windows as one batch in the self layers, the expected offset.
- the fine attention over 25-token windows is one Metal kernel (`window_attention`, a threadgroup per window and
  head): the coarse stage's block-diagonal head-mask GEMMs were mostly zeros there (64 -> 12 ms per layer at 7400
  matches). The layer norms are one Metal kernel with the residual fused (`layer_norm`, a simdgroup per row, float32
  statistics, float16 in and out): `mx.fast.layer_norm` with float32 weights returns float32 and took 23 ms against
  2.4 ms on a `[185k, 128]` float16 input. Refine went 380 -> 264 ms (both under contention).
- float16: the convolutions and the transformers' GEMMs. Float32: the attention sums and normalisers, layer-norm
  statistics and weights, the position encoding's sum, the similarities, dual-softmax and outputs. Golden at float16:
  indoor recall 0.9984 (needs 0.998), mean 0.0018; outdoor 0.9994, mean 0.0022 (tolerance 0.0025). A float32 backbone
  cuts the error by 3-6x (indoor recall unchanged at 0.9985); not needed.

Left: the warm fp32 rows (and the fp16 ones, for a clean reading) on a quiet machine; the coarse attention still uses
the head-mask GEMMs (two `[L, 256] x [256, 256]` products per layer, cheap at 15k tokens).

## First calls over the auto budget (2026-11-25)

`auto` takes MLX only when its first call is at most the MPS first call plus twice the MPS warm time
(`tools/gen_auto_mode.py`, `cold=True`). 203 refs have an MLX row that wins warm (and on memory) against the MPS path
`auto` would replace, but miss that budget (2026-11-25 rows, M1 Pro 16 GB). Per family: refs, the MLX / MPS warm ratio,
the MLX first call, and how far over the budget it is:

| family | refs | warm MLX / MPS | first call s | over budget s |
|---|---|---|---|---|
| clip | 10 | 0.84-1.03 | 0.68-1.81 | 0.60-1.23 |
| coatnet | 10 | 0.37-0.57 | 0.86-2.86 | 0.51-2.38 |
| dfine | 10 | 0.72-0.97 | 1.52-9.11 | 1.17-8.68 |
| rfdetr | 10 | 0.57-0.74 | 1.82-2.56 | 1.57-1.91 |
| rtmpose | 10 | 0.90-1.05 | 1.25-1.65 | 0.73-1.05 |
| hgnetv2 | 7 | 0.45-0.70 | 0.73-1.76 | 0.59-1.53 |
| focalnet | 6 | 0.52-0.55 | 0.80-1.48 | 0.29-1.21 |
| siglip2 | 6 | 0.85-1.04 | 0.71-4.16 | 0.60-3.54 |
| lwdetr | 5 | 0.61-1.01 | 1.89-2.71 | 1.63-2.23 |
| rtdetrv2 | 5 | 0.84-1.03 | 1.54-2.12 | 1.18-1.58 |
| tinyvit | 5 | 0.54-0.78 | 0.97-1.18 | 0.83-1.09 |
| efficientformerv2 | 4 | 0.30-0.34 | 1.39-2.29 | 1.33-2.18 |
| grounding_dino | 4 | 0.69-0.89 | 5.64-9.16 | 2.08-5.45 |
| mmseg | 4 | 0.77-0.97 | 1.61-3.64 | 0.04-2.84 |
| segformer | 4 | 0.54-0.91 | 1.05-5.57 | 0.33-5.10 |
| shvit | 4 | 0.43-0.54 | 1.19-1.46 | 0.89-1.29 |
| starnet | 4 | 0.35-0.38 | 0.53-0.55 | 0.39-0.45 |
| wd_tagger | 4 | 0.85-0.89 | 0.68-2.24 | 0.48-1.49 |
| davit | 3 | 0.56-0.59 | 1.10-1.88 | 0.53-1.66 |
| efficientformer | 3 | 0.56-0.61 | 1.01-1.56 | 0.98-1.48 |
| fastvit | 3 | 0.54-0.55 | 0.53-1.03 | 0.43-0.89 |
| las2 | 3 | 0.82-0.91 | 1.86-5.41 | 0.81-3.31 |
| perception_encoder | 3 | 0.64-0.72 | 0.86-1.79 | 0.81-1.71 |
| restormer | 3 | 0.59-0.75 | 3.49-4.03 | 0.02-0.96 |
| rtmo | 3 | 0.72-1.03 | 1.39-1.54 | 0.67-1.21 |
| sam_hq | 3 | 0.76-0.81 | 2.36-7.57 | 0.29-0.47 |
| u2net | 3 | 0.56-0.83 | 2.22-4.97 | 1.58-4.36 |
| vitpose | 3 | 0.86-0.92 | 1.41-2.12 | 0.49-0.88 |
| birefnet | 2 | 0.81-0.91 | 5.18-7.28 | 3.30-3.39 |
| detr | 2 | 0.92-0.95 | 1.51-1.71 | 1.01-1.11 |
| foundationstereo | 2 | 0.93-1.01 | 59.93-127.25 | 1.78-48.38 |
| inspyrenet | 2 | 0.66-0.97 | 4.70-5.44 | 2.40-3.86 |
| isnet | 2 | 0.94-0.94 | 4.95-6.01 | 4.13-5.28 |
| loma | 2 | 0.90-1.02 | 4.82-5.86 | 1.83-2.43 |
| nafnet | 2 | 0.52-0.79 | 2.38-3.10 | 0.92-1.64 |
| parseq | 2 | 0.30-0.76 | 4.86-6.61 | 4.54-6.21 |
| pointdit | 2 | 0.94-1.03 | 8.72-23.79 | 7.57-20.70 |
| rtmdet | 2 | 0.91-0.98 | 1.50-1.93 | 1.17-1.22 |
| sea_raft | 2 | 0.74-0.83 | 1.62-2.02 | 0.79-0.85 |
| yunet | 2 | 0.91-1.03 | 1.30-1.54 | 1.12-1.35 |

One ref each (over budget, s): bridgedepth 4.23, clipseg 1.96, compact 1.28, deit 0.57, depthsplat 4.34, efficientvit 1.19, eomt 3.92, eva02 1.40, flowseek 4.90, geocalib 4.35, hrnet 1.57, inception_v3 0.94, lineart_anime 2.36, mask2former 3.09, megaloc 7.16, mlsd 1.72, mobilenetv4 0.75, nsfw_vit 0.58, oneformer 4.42, patchmatchnet 20.51, pidinet 0.70, raft 0.19, realesrgan 4.77, recover 0.50, rfdetr_keypoint 2.60, romav1 2.40, rt_monster_plusplus 2.57, sam 0.35, sam2 1.04, siglip 0.80, span 1.08, swin 1.90, teed 1.67, unik3d 0.39, vggt_omega 1.30, vit 0.54, yolox 0.99.

Owned elsewhere: dfine, detr, rtdetr (detection), oneformer, mask2former, geocalib. Not on the list: aliked-flexglue
(MLX fp32 938 ms, 4.2 s first call) has no MPS row with a passing golden to compare against.

Nearly in: mmseg deeplabv3 (0.04 s over), restormer-sidd (0.02 s), raft-small (0.19 s), the SAM / SAM-HQ ViTs
(0.29-0.47 s), unik3d (0.39 s). Most of the rest are exported graphs whose translation and first compile take
0.5-2 s on small classifiers; a hand-written network (no export) is the fix there.

### Done: SEA-RAFT and RAFT (2026-11-25)

M1 Pro, 1080x1920, all eleven MLX goldens pass; second of two surveys recorded (the two agreed within 3 %). `auto`
now takes MLX float16 for every ref: new for raft-small, sea-raft-s-spring and sea-raft-m-spring, whose first calls
were over the budget.

| ref | | warm ms | first call s | peak MB | auto |
|---|---|---|---|---|---|
| sea-raft-m (kitti, things alike) | MLX before, fp16 / fp32 | 1011 / 1330 | 2.72 / 2.99 | 2583 / 2719 | MLX fp16 |
| | MLX now, fp16 / fp32 | 928 / 1296 | 0.97 / 1.31 | 2002 / 2383 | MLX fp16 |
| | MPS kernels fp16 | 1221 | 1.31 | 4752 | |
| sea-raft-m-spring | MLX before | 281 / 345 | 2.02 / 2.04 | 1484 / 1448 | – |
| | MLX now | 265 / 319 | 0.30 / 0.36 | 1023 / 1203 | MLX fp16 |
| | MPS kernels fp16 | 340 | 0.49 | 1621 | |
| sea-raft-s (kitti alike) | MLX before | 665 / 957 | 2.10 / 2.42 | 2512 / 2716 | MLX fp16 |
| | MLX now | 617 / 832 | 0.65 / 0.84 | 2019 / 2274 | MLX fp16 |
| | MPS kernels fp16 | 833 | 0.91 | 4667 | |
| sea-raft-s-spring | MLX before | 172 / 229 | 1.62 / 1.76 | 1401 / 1382 | – |
| | MLX now | 167 / 218 | 0.20 / 0.24 | 985 / 1147 | MLX fp16 |
| | MPS kernels fp16 | 234 | 0.37 | 1527 | |
| raft-large (kitti, things alike) | MLX before | 996 / 1350 | 2.56 / 3.12 | 2004 / 2211 | MLX fp16 |
| | MLX now | 916 / 1348 | 0.96 / 1.40 | 1829 / 2051 | MLX fp16 |
| | MPS kernels fp16 | 1351 | 1.48 | 3647 | |
| raft-small | MLX before | 332 / 450 | 1.90 / 2.87 | 1645 / 1861 | – |
| | MLX now | 294 / 403 | 0.33 / 0.45 | 1092 / 1629 | MLX fp16 |
| | MPS kernels fp16 | 386 | 0.95 | 2482 | |

Before, both ports replayed an exported graph (`raft/mlx.py`, `sea_raft/mlx.py`, a PyTorch regroup for the export):
its translation and compiles were 1.5-3 s of each first call. Now each network is written in `mlx.core`
(`raft/mlx_net.py`, `sea_raft/mlx_net.py`), weights copied from the PyTorch twin built on the CPU, nothing exported:

- encoders channels last, batch norm folded, instance norm from float32 statistics (`ops.channel_stats`, the
  conv bias before it dropped). Evaluated block by block and one image at a time: built as one lazy graph over the
  pair, the 1/2-resolution convs' transients piled up to 3.4-3.8 GB of MLX peak (footprint 3274 MB on sea-raft-m,
  above the export's 2583); block by block the encoder peak is under 0.5 GB.
- no correlation volume: `graph_mlx.ops.corr_windows` on fmap1 and the 2x2-pooled fmap2 levels.
- one compiled update step. RAFT's GRUs convolve the unchanging context once per call (with the biases), z and r
  as one convolution; motion features zero-padded to multiples of 32 with the flow last, the motion conv's outputs
  padded to MLX's Winograd sizes. Float32: flow state, its 7x7 conv, the windows, the upsampling; SEA-RAFT's
  128 -> 128 encoder convs (MLX's half Winograd misses the golden there).

Left: the Spring variants' 512 MB buffer cache was tuned on the exported port; not re-tuned.

## 1x1 convs as a GEMM (2026-11-24)

The Restormer lead as a translator rule: a dense 1x1 stride-1 unpadded conv on the channels-last data is
`mx.addmm(bias, x.reshape(-1, C), W)` (`conv1x1` / `conv1x1_ok` in `shared/mlx_graph.py`, also used by
`libnet/mlx_cnn.py` and RoMa v2's `mlx_net.py`). Off with `FF_MLX_NO_MM1X1=1`, per port with
`mlx_wrap.port(..., gemm1x1=False)`; `FF_MLX_MM1X1_ALL=1` takes every width, `FF_MLX_MM1X1_NO_ADDMM=1` adds the bias
separately.

Microbenchmark, M1 Pro, MLX 0.32.4, one conv with bias, ms (`mx.conv2d` + bias through `_conv2d`'s padding / GEMM
with fused bias):

| in -> out | fp16 64x64 | fp16 128x128 | fp16 512x512 | fp32 64x64 | fp32 128x128 | fp32 512x512 |
|---|---|---|---|---|---|---|
| 24 -> 96 | 0.27 / 0.22 | 0.37 / 0.26 | 2.66 / 1.03 | 0.28 / 0.22 | 0.43 / 0.29 | 4.09 / 1.24 |
| 96 -> 288 | 0.42 / 0.36 | 0.60 / 0.47 | 6.94 / 4.36 | 0.45 / 0.34 | 0.83 / 0.51 | 9.92 / 5.16 |
| 96 -> 512 | 0.68 / 0.51 | 0.87 / 0.59 | 11.26 / 6.41 | 0.84 / 0.43 | 1.18 / 0.63 | 16.08 / 7.22 |
| 180 -> 180 | 0.46 / 0.31 | 1.06 / 0.51 | 10.25 / 5.18 | 0.47 / 0.30 | 1.05 / 0.58 | 12.66 / 6.22 |
| 256 -> 64 | 0.26 / 0.23 | 0.39 / 0.35 | 2.99 / 2.14 | 0.25 / 0.23 | 0.44 / 0.34 | 4.19 / 2.45 |
| 256 -> 96 | 0.37 / 0.29 | 0.57 / 0.46 | 5.66 / 4.45 | 0.33 / 0.28 | 0.67 / 0.52 | 7.54 / 5.30 |
| 256 -> 256 | 0.42 / 0.35 | 0.85 / 0.67 | 11.29 / 7.76 | 0.40 / 0.32 | 1.11 / 0.73 | 15.02 / 8.60 |
| 512 -> 512 | 0.88 / 0.67 | 2.69 / 2.05 | 39.97 / 29.48 | 0.95 / 0.71 | 3.37 / 2.23 | 50.56 / 32.09 |
| 256 -> 1 | 0.23 / 0.19 | 0.27 / 0.22 | 1.05 / 0.94 | 0.22 / 0.18 | 0.28 / 0.24 | 1.71 / 1.66 |
| 144 -> 24 | 0.26 / 0.25 | 0.61 / 0.51 | 1.91 / 1.64 | 0.24 / 0.23 | 0.36 / 0.31 | 2.49 / 1.95 |
| 128 -> 2 | 0.23 / 0.25 | 0.25 / 0.30 | 0.69 / 1.39 | 0.22 / 0.22 | 0.26 / 0.31 | 1.02 / 1.75 |
| 16 -> 16 | 0.23 / 0.22 | 0.27 / 0.23 | 0.62 / 0.51 | 0.24 / 0.15 | 0.24 / 0.26 | 0.70 / 0.60 |

The rule that came out of the full grid (8x8 to 512x512, 27 width pairs, with and without bias):

- 33 outputs and more, and 1 output: GEMM. Below 32x32 both are launch-bound and equal.
- 2 to 16 outputs: `mx.conv2d` has a narrow-output path, 1.1 to 2.2x faster than the GEMM from 16 inputs on (256 -> 8
  at 256x256: 1.70x fp16, 1.37x fp32). They stay convs.
- 17 to 32 outputs: GEMM only from 64 inputs and 128x128 positions on (0.46 to 0.93 there; 0.9 to 1.1 otherwise,
  48 -> 32 at 512x512 1.08).
- Stride 2: no gain (the conv reads a quarter of the positions). Grouped, padded: conv.
- A weight of 2^20 elements or more stays a conv. RegNetY-160's 1232 -> 1232 layers at 14x14 are 25% faster as GEMMs
  (the net 18.4 -> 16.3 ms fp16) but its peak footprint rose 40 MB each time (1060 -> 1104, 1160 -> 1213 MB): MLX's
  buffer cache holds about 40 MB more after a stack of such GEMMs (22 layers: cache 52 against 13 MB), with the fused
  bias or a separate add, from a contiguous matrix or a transposed view. At 448 channels there is no difference. With
  the limit: 18.0 ms, 1063 MB.
- The fused bias is worth keeping: with a separate add Restormer deblur is 655 / 1324 ms (fp16 / fp32) instead of
  599 / 1153, NAFNet GoPro 304 / 473 instead of 290 / 451. It is also the more accurate form: float16 against a
  float32 reference, `conv2d + b` and `matmul + b` have the same error to the digit (relative 2.0e-4 to 2.5e-4),
  `addmm` 1.76e-4 with about half the maximum error.

Survey, MLX, warm ms and peak footprint MB, without (`FF_MLX_NO_MM1X1=1`) and with the rule, same session. All
goldens pass on both sides.

| ref (input) | row | ms before | ms now | | peak before | peak now |
|---|---|---|---|---|---|---|
| restormer-deblur (512x512) | fp16 | 726.2 | 599.0 | -17.5% | 2067 | 2081 |
| | fp32 | 1539.5 | 1121.1 | -27.2% | 2666 | 2737 |
| restormer-sidd | fp16 | 794.7 | 665.2 | -16.3% | 2136 | 2136 |
| | fp32 | 1577.2 | 1235.5 | -21.7% | 2644 | 2570 |
| nafnet-gopro | portable fp16 | 359.6 | 289.7 | -19.4% | 1354 | 1424 |
| | portable fp32 | 527.8 | 423.0 | -19.9% | 1641 | 1577 |
| | kernels fp16 | 357.8 | 292.3 | -18.3% | 1355 | 1359 |
| | kernels fp32 | 533.4 | 427.3 | -19.9% | 1641 | 1578 |
| nafnet-sidd | portable fp16 | 279.3 | 231.8 | -17.0% | 1395 | 1395 |
| | portable fp32 | 445.9 | 383.0 | -14.1% | 1775 | 1710 |
| | kernels fp16 | 278.0 | 228.4 | -17.8% | 1395 | 1395 |
| | kernels fp32 | 446.2 | 380.4 | -14.7% | 1775 | 1712 |
| efficientnet_b0.tv_in1k | fp16 / fp32 | 4.4 / 4.7 | 3.8 / 4.8 | -14% / +2% | 356 / 364 | 360 / 367 |
| | kernels | 4.3 / 4.7 | 3.6 / 4.1 | -15% / -13% | 356 / 365 | 358 / 367 |
| mobilenet_v2.tv2_in1k | kernels | 3.1 / 3.1 | 2.5 / 2.7 | -20% / -14% | 327 / 337 | 330 / 341 |
| mobilenet_v3_large.tv2_in1k | kernels | 3.2 / 3.7 | 2.7 / 3.6 | -16% / -3% | 348 / 355 | 348 / 355 |
| repvit_m1_0.dist_300e_in1k | fp16 / fp32 | 5.1 / 6.1 | 4.4 / 4.6 | -13% / -24% | 423 / 440 | 424 / 444 |
| hgnetv2_b2.ssld_stage2_ft_in1k | fp16 / fp32 | 4.7 / 5.3 | 4.3 / 4.8 | -9% / -9% | 553 / 593 | 553 / 593 |
| resnet50.tv2_in1k | fp16 / fp32 | 6.9 / 7.9 | 6.5 / 7.4 | -6% / -6% | 518 / 549 | 518 / 554 |
| regnety_160.tv2_in1k | fp16 / fp32 | 18.4 / 19.2 | 18.0 / 18.8 | -2% / -2% | 1060 / 1160 | 1063 / 1165 |
| segnext-mscan-s-ade20k | kernels | 80.5 / 99.9 | 74.9 / 89.9 | -7% / -10% | 784 / 910 | 786 / 916 |
| deeplabv3plus-r101-d8-ade20k | fp16 / fp32 | 359.6 / 502.9 | 337.9 / 466.8 | -6% / -7% | 1392 / 1903 | 1395 / 1863 |
| dfine-s-coco | fp16 / fp32 | 32.3 / 36.0 | 29.5 / 34.6 | -9% / -4% | 668 / 703 | 652 / 699 |
| rtdetrv2-s-coco | fp16 / fp32 | 39.0 / 50.0 | 37.1 / 46.4 | -5% / -7% | 760 / 855 | 751 / 839 |
| rtmdet-s-coco | fp16 / fp32 | 25.9 / 35.3 | 25.9 / 33.0 | 0% / -7% | 719 / 797 | 721 / 800 |
| yolox-s | fp16 / fp32 | 22.9 / 28.6 | 22.3 / 26.9 | -3% / -6% | 605 / 663 | 616 / 666 |
| matanyone | fp16 / fp32 | 3691 / 5137 | 3513 / 4891 | -5% / -5% | 3563 / 4172 | 3562 / 4175 |
| raft-large | fp16 / fp32 | 1020 / 1385 | 996 / 1350 | -2% / -3% | 2004 / 2216 | 2004 / 2211 |
| sea-raft-m | fp16 / fp32 | 1037 / 1354 | 1011 / 1330 | -3% / -2% | 2608 / 2783 | 2583 / 2719 |
| hrnet-w32 | fp16 / fp32 | 111.3 / 149.4 | 109.0 / 144.9 | -2% / -3% | 867 / 1060 | 860 / 1072 |
| birefnet-lite | fp16 / fp32 | 389.7 / 637.9 | 382.1 / 617.6 | -2% / -3% | 1813 / 2703 | 1810 / 2695 |

Within +-2.6% in time (noise) and the same peak: segformer-b0 / b5, oneformer-swinl, zoedepth-nk, moge-2-vitl-normal,
romav2, realplksr (portable; kernels opted out), sam2.1-hiera-tiny, igev-stereo-sceneflow, neuflow-v2, rife-4.25,
u2net, swinir-m-x4, vit_base_patch16_224, convnext_tiny, depth-anything-v2-s, midas-dpt-beit-large-512 in time.

Peak footprint is the one open point. Repeats of the same row differ by up to about 70 MB on this machine, and on
three rows the higher value shows only with the rule: nafnet-gopro portable fp16 (1354 to 1357 MB in 3 runs without;
1359, 1360, 1424, 1424 with), restormer-deblur fp32 (2656, 2666 without; 2704, 2706, 2737 with), and
midas-dpt-beit-large-512 fp32 (3039 to 3085 in 5 runs without; 3071 to 3079 in 4 runs and 3221, 3231 in 2 with).
The other dtype of each, and the sibling refs (nafnet-sidd, restormer-sidd, zoedepth-nk), show none or a lower peak.
Not chased further: it is 1.5 to 5% for a 17 to 27% faster call on the first two. Midas gains nothing in time, so
its ports opt out (`gemm1x1=False` in `midas/model.py`). YOLOX opts out too: about 3% on yolox-s, and the yolox-x
kernels float16 golden drifted past its limit with the GEMMs (boxes mean |diff| 0.80 of 0.2; 0 failures without).

Goldens with the rule: 293 MLX goldens (every family above and every second ref of the rest) in seven batches, 289
pass. The four: realplksr kernels fp16 (p99 0.00181 of 0.0018, 0.00180 without the rule: opted out, passes),
rmbg-2.0 (gated weights, HTTP 401), owlv2-b16 (portable fp16 boxes mean 0.19) and sam2.1-hiera-base-plus (portable
fp16 masks mean 0.0028), the last two failing the same way with `FF_MLX_NO_MM1X1=1`.

**Left:** the 1232-wide-and-up GEMMs if the cache growth can be avoided (RegNetY-160 -11%); stride-2 1x1 convs as a
strided slice plus GEMM (equal in isolation); 1-D and 3-D convs with kernel 1; hand-written ports that call
`mx.conv2d` themselves (only `libnet/mlx_cnn.py` and RoMa v2 were switched).

## OWLv2 and SAM 2.1 float16 goldens: float32 stages that ran in float16 (2026-11-24)

`test_mlx[owlv2-b16@1]` (boxes mean |diff| 0.191 px, limit 0.15) and `test_mlx[sam2.1-hiera-base-plus@1]`
(pred_masks mean 0.0028 of the median, limit 0.0026) failed in portable float16, with and without
`FF_MLX_NO_MM1X1=1`. Float32 passed (2e-04 px, 1e-05).

**Since when**: 921afaa7 (GELU computed in float32 inside a float16 replay), found by bisect on owlv2-b16; at HEAD
with the old GELU sam2.1-hiera-base-plus is back at 0.0018. That GELU is not wrong: against a float64 reference it
is the correctly rounded float16 value (mean error 3.5e-05 at sigma 0.5, the old all-float16 one 5.4e-05, neither
biased). Both refs were inside their limits by luck before it (owlv2-b16 0.143 of 0.15, the survey row of
2026-11-23); a different rounding pattern moved them over.

**Cause**: stages PyTorch keeps in float32 ran in float16 on MLX.

- **OWLv2**: `Portable.heads` (post layer norm, the class-token merge, layer norm, box and class heads) is the
  float32 part after the float16 encoder, written as `t.float()`. An MLX replay is exported from a float32 twin, on
  which `.float()` records nothing, so the heads were replayed in float16: the box MLP's rounding goes straight into
  boxes (one float16 step of the sigmoid is 0.47 px at 960 px). They are `float32_call`s now (float32 inputs and
  weights in the replay; the same `module(x.float())` everywhere else). OWLv2 is routed to MLX float16 on `auto`, so
  this was a user-visible accuracy loss on `owlv2-b16`, not only a test failure.
- **SAM 2.1**: the float16 Hiera replay keeps a float32 residual stream (`fast._block_f32_stream`), but at the three
  stage transitions the shortcut is `pool(proj(norm1(x)))`, and `proj` ran in float16 on a float16 copy of the norm:
  the stream itself was rounded once per stage. `proj` is a `float32_call` on the float32 norm output now. Tried on
  base-plus and not needed (mean / p99, limits 0.0026 / 0.009): neck linears in float32 0.00257 / 0.0104, patch embed
  and position embedding in float32 0.00205 / 0.0098, the projection alone 0.00124 / 0.0056.

MLX portable float16 golden, mean / p99 before -> after (limits), M1 Pro; float32 is unchanged and passes everywhere:

| ref | before | after | limit |
|---|---|---|---|
| owlv2-b16 (boxes, px) | 0.1914 / 0.456 fail | 0.0871 / 0.288 | 0.15 / 0.39 |
| owlv2-l14 | 0.1334 / 0.356 | 0.0585 / 0.160 | 0.15 / 0.39 |
| sam2.1-hiera-tiny (pred_masks, relative) | 0.00111 | 0.00151 / 0.0067 | 0.0037 / 0.015 |
| sam2.1-hiera-small | 0.00580 | 0.00550 / 0.0167 | 0.017 / 0.051 |
| sam2.1-hiera-base-plus | 0.00280 / 0.0110 fail | 0.00124 / 0.0056 | 0.0026 / 0.009 |
| sam2.1-hiera-large | 0.00326 | 0.00362 / 0.0098 | 0.023 / 0.1 |

Cost (`tools/speed_survey.py`, float16, warm ms / peak MB, before -> after; the time differences are run-to-run
noise, the peak grows by the float32 head and projection weights and activations): owlv2-b16 294 / 991 -> 291 /
1043, owlv2-l14 1423 / 1755 -> 1427 / 1792, sam2.1 tiny 141 / 933 -> 145 / 949, small 164 / 951 -> 169 / 969,
base-plus 282 / 1099 -> 282 / 1099, large 597 / 1460 -> 598 / 1498. Routing is unchanged (OWLv2 `auto` on MLX
float16; SAM 2.1 `auto` stays on MPS, MLX float16 is the default of `device="mlx"`).

The same trap is possible wherever a portable runner marks its float32 part with `.float()` and has an MLX port
through `mlx_wrap.port`: on the float32 twin the cast vanishes.

### Audit across the MLX families (2026-11-25)

**Method.** `tools/mlx_dtype_report.py --float32-casts REF` lists every literal float32 cast (`.float()`,
`.to(torch.float32)`, `.type(torch.float32)`) that a module or function runs while the twin is being exported for
MLX, with its source line: each one records nothing in the replay. Not all of them matter. A constant made in float32
(`zeros`, `arange`, a coordinate grid) stays float32 in the replay and MLX promotes float16 with float32 to float32,
so a `.float()` on an accumulator is often already float32; a cast right before the output only changes the output
dtype. To screen, an A/B harness turned *every* vanishing cast into `fastforks::float32` (a monkeypatch, not
committed) and compared the float16 golden error and warm time with and without it, on the families whose report
listed casts (the float16 goldens of the 198 fastest MLX refs were swept first: all passed). A family was fixed when the strict run moved
its error materially and its runner marks a stage as float32 on PyTorch; the fix uses the existing helpers
(`to_float32`, `to_run_dtype`, `float32_call`) plus one new one, `mlx_cast.keep_float32(*modules)`, which runs
whole modules on float32 inputs and weights during the MLX export (the module form of `float32_call`, for heads that
are called from inside upstream code).

**Fixed** (MLX portable float16 golden, mean before -> after, M1 Pro; float32 passes unchanged everywhere, the
whole `test_mlx` golden set of each family passes float16 and float32):

| ref | float32 stage | before | after |
|---|---|---|---|
| nnfoundation-vit (`auto` MLX float16) | Primus residual stream, layer norms, layer-scale adds, RoPE | 0.0189 | 0.0052 (limit 0.03) |
| prompt-depth-anything-s / -l (-l `auto` MLX float16) | last DPT conv (`output_conv2[2]`) | 0.00049 | 0.00051 |
| any2full-s / -b / -l | depth embedding, last DPT conv | 0.000126 | 0.000088 |
| omni-dc | last DPT conv, the normal-equation solve, backbone and post stage | 0.000119 | 0.000087 |
| detr-r50 / -r101 | class and box heads | 0.0112 | 0.0058 |
| dfine-n..x | decoder heads, query selection, distribution projection | 0.0671 | 0.0623 |
| deim-n..x (D-FINE runner) | same | 0.106 | 0.0702 |
| rtdetrv2-s..x | encoder output projection and norm, decoder heads | 0.0371 | 0.0360 |
| geocalib-pinhole | NMF in float32 | 0.0355 | 0.0304 |
| anycalib-pinhole | tangent head, upsampling softmax | 0.050 | 0.071 (limit 0.094) |

anycalib is the one that got worse in float16 (p99 0.26 -> 0.41, inside its limits). It is kept: the float32
stages are what the PyTorch runner computes, the float16 replay is now the same computation, and anycalib's MLX
default is float32 anyway (`device="mlx"` float16 is opt-in). prompt-depth moved within noise; the fix is kept for
the same reason.

**Video Depth Anything is not changed.** Its float16 runner keeps the DPT output head (`output_conv2`) in float32,
and the replay does not. Keeping it float32 in the replay (`keep_float32`, tried) costs 12-14% warm time at the
standard 994x1482: metric-s 893 -> 1018 ms and s 900 -> 1009 ms, at the same peak (2978 MB). The error does not
improve consistently: metric-s 0.00204 -> 0.00159, s 0.00030 -> 0.00056, limits 0.0042 and 0.0055. Keeping only the
last conv in float32, as the other DPT heads do, gave 0.00181 at the same cost as the whole head. Neither ref is
routed to MLX on `auto`.

**Screened and left alone** (strict run vs. current, float16 mean / warm ms): raft-large 0.00082 / 125 -> 0.00074 /
148 and memflow-kitti 0.00102 / 488 -> 0.00096 / 543 (their `.float()` are on coordinate accumulators that are float32
in the replay already; the rest only costs time), sea-raft-s 0.00089 -> 0.00087, gimm-vfi-r-p 0.000144 -> 0.000142,
dedode-v2-g identical (3445 -> 3798 ms), monster-mix-all 0.00115 -> 0.00089 (limit 0.02), monster-plusplus-mix-all
0.0232 -> 0.0261 (worse), loftr-outdoor 0.00233 -> 0.00220 (a cast after a mean, not a float32 stage),
retinaface-mobilenet0.25 identical, gmflow-scale1-things 0.00245 -> 0.00241. lwdetr and rfdetr only cast their
outputs.

**Cost** (`tools/speed_survey.py` re-run of the changed refs on apple-m1-pro-mlx-16gb, MLX portable float16, warm
ms / first call s / peak MB). The baseline is the 2026-11-19 rows at an older commit, so the deltas also include
other work since then; the like-for-like A/B runs above showed no slowdown beyond noise for the fixed families.

| ref | 2026-11-19 | now |
|---|---|---|
| nnfoundation-vit | 13037 / 29.5 / 2469 | 13240 / 17.7 / 2519 |
| prompt-depth-anything-l | 1022 / 3.2 / 2862 | 1008 / 3.3 / 2710 |
| any2full-s | 160 / 3.4 / 1012 | 113 / 2.4 / 819 |
| omni-dc | 3873 / 11.9 / 5387 | 3352 / 9.5 / 5336 |
| detr-r50-coco | 97 / 1.5 / 1414 | 91 / 1.9 / 1397 |
| dfine-n-coco | 22.3 / 9.1 / 624 | 21.6 / 2.0 / 578 |
| deim-n-coco | 21.0 / 1.5 / 602 | 21.7 / 2.0 / 575 |
| rtdetrv2-s-coco | 37.1 / 1.8 / 751 | 37.1 / 1.8 / 768 |
| geocalib-pinhole | 352 / 6.0 / 1013 | 276 / 5.0 / 961 |
| anycalib-pinhole | 439 / 5.3 / 1495 | 295 / 2.4 / 1287 |

prompt-depth-anything-s and both Video Depth Anything refs had rows only at older standard inputs (756x1008,
480x640). Those rows were replaced by rows at today's 994x1482: prompt-depth-s 148 ms / 971 MB. `auto` routing is
unchanged: `tools/gen_auto_mode.py` keeps nnfoundation-vit and prompt-depth-anything-l on MLX float16. Two
secondary routes moved with the new rows. anycalib-pinhole's `device="mlx"` default is float16 now: 295 ms against
310 ms for float32, with the float16 golden passing at 0.071 of 0.094. geocalib-pinhole's explicit `dtype="float32"`
on `auto` now goes to MLX float32 (314 ms).

**Left**: the strict screen covered the families the report flagged, not every ref with an MLX port (the slow
ones beyond the 198 were only checked where flagged); monster-plusplus sits at 0.023 of 0.03 in float16 and
strict float32 did not help, so its margin comes from elsewhere. The report flag stays the way to check a new port:
`tools/mlx_dtype_report.py --float32-casts REF` should list nothing that the runner means as a float32 stage.
