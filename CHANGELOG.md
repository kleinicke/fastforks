# Changelog

## Unreleased

- Real-ESRGAN super-resolution (`realesrgan-x4plus`, `realesrgan-x2plus`, `realesrgan-x4plus-anime-6b`; new task
  `restoration`; BSD-3-Clause code and weights): RRDBNet as `RealESRGANer.enhance` (no tiles). 512x512 x4plus:
  402 ms kernels, 632 ms portable (354 ms compile, 273 ms TensorRT) vs 1209 ms original fp32 (631 ms fp16), peak
  1.4 GB vs 5.0 GB. DIV2K validation (100 images, bicubic LR) PSNR 26.64 in every tier, as the original.

- RIFE frame interpolation (`rife-4.25`, `rife-4.26`, `rife-4.25-lite`; new task `interpolation`; MIT code and
  weights): Practical-RIFE `IFNet_HDv3`, any timestep in [0, 1], frames zero-padded to a multiple of 128 as upstream.
  1080x1920 pair: 31.4 ms kernels, 36.0 ms portable (44.5 ms compile, 17.6 ms TensorRT) vs 45.3 ms original fp32
  (24.5 ms fp16), peak 1.0 GB vs 1.9 GB. DAVIS 2017 (180 triplets) PSNR 27.13 kernels / portable vs 27.13 original
  fp32 (27.03 fp16); the warps sample in float32.

- New task `fastforks.retrieval` (image retrieval / visual place recognition): one unit-length global descriptor per
  image, `top_k` (exact cosine top-k on the GPU, chunked) and `pairs` (image pairs for 3-D reconstruction).

- MegaLoc place recognition (`megaloc`, retrieval; MIT code and weights): DINOv2 ViT-B with optimal-transport
  (Sinkhorn) pooling, an 8448-d descriptor. 480x640 image: 4.26 ms kernels, 5.29 ms portable (4.27 ms compile, 3.21 ms
  TensorRT) vs 14.6 ms original fp32 (9.68 ms autocast; `half()` fails in the authors' code), peak 508 MB vs 944 MB.
  St Lucia recall@1 100.0 for all modes and the original.

- BoQ, Bag-of-Queries place recognition (`boq-dinov2`, retrieval; MIT code and weights): DINOv2 ViT-B with learned
  queries, a 12288-d descriptor. 480x640 image: 2.84 ms kernels, 5.73 ms portable (3.89 ms compile, 2.73 ms TensorRT)
  vs 13.5 ms original fp32 (7.44 ms `half()`), peak 256 MB vs 430 MB. St Lucia recall@1 99.86 for all modes and the
  original. The ResNet-50 variant is not ported.

- SAMURAI motion-aware tracking (`samurai` option of the `sam2.1-hiera-*` cards, `track`; Apache-2.0 code, the SAM 2.1
  weights): a Kalman filter on each object's mask box picks among the multimask outputs once stable (weighted with the
  predicted IoU, `kf_score_weight` 0.25, 0.15 for large) and the memory bank keeps the confident frames; several
  objects at once. LaSOT subset (6 videos, 10.5k frames, base-plus fp16): AUC 73.58 kernels / 76.93 portable vs 73.73
  original SAMURAI (portable recovers the target on mouse-8; without it 81.20 / 81.19 vs 81.37), 52.5 ms per frame
  kernels, 59.2 ms portable vs 90.9 ms original.

- Multi-object tracking (`tracking.ByteTracker`, `tracking.track_objects`; MIT, after ByteTrack): ByteTrack's two-stage
  association of per-frame detections (D-FINE / RT-DETR / RF-DETR results or `[N, 5]` arrays), batched Kalman filter
  (`shared/box_kalman`, also for SAMURAI) and an exact linear assignment without scipy. MOT17 train with D-FINE-X
  person detections: identical tracks to the official ByteTrack (0 id differences in 5316 frames, boxes within
  1.2e-4 px; HOTA 39.40 / MOTA 33.05 / IDF1 46.23 both), 1.10 ms per frame on the CPU (official 1.27 ms); with the
  detector 11.5 ms per 1080p frame kernels (20.0 ms portable).
- New task `fastforks.calibration` (single-image camera calibration): the pinhole intrinsics `[3, 3]` and the
  vertical field of view, plus gravity, roll and pitch where the model estimates them.

- GeoCalib (`geocalib-pinhole`, calibration; Apache-2.0 code, CC-BY-4.0 weights): MSCAN-L encoder with up / latitude field
  heads and a Levenberg-Marquardt fit of the camera and gravity. 1080x1920 image: 82.5 ms kernels, 82.3 ms portable
  (73.1 ms compile; TensorRT export fails on the LM fit) vs 124.9 ms original fp32 (106.2 ms autocast), peak 342 MB vs
  658 MB. LaMAR2k AUC@5 roll / pitch / vFoV 92.6 / 76.8 / 41.6 for all modes and the original. The Hamburger heads'
  NMF starts from fixed bases (upstream: new random ones per call); no uncertainty estimate.

- AnyCalib (`anycalib-pinhole`, calibration; Apache-2.0 code and weights): DINOv2 ViT-L/14 with a light DPT decoder
  predicting one ray per pixel, then a linear pinhole fit and 10 Gauss-Newton steps. 1080x1920 image: 21.3 ms kernels,
  29.4 ms portable (25.5 ms compile; TensorRT export fails on the fit) vs 58.5 ms original fp32 (38.4 ms autocast), peak
  816 MB vs 1372 MB. LaMAR2k vFoV AUC@5 51.5 for all modes and the original. The fit runs batched in float32; a failed
  fit gives NaN (no RANSAC); pinhole only, intrinsics and vertical FoV (no gravity).

- LW-DETR real-time detection (`lw-detr-tiny-coco` ... `lw-detr-xlarge-coco`, detection; Apache-2.0 code and
  weights): ViT encoder with a multi-level projector and a deformable DETR decoder, the authors' PostProcess.
  800x1344 image, tiny: 2.2 ms kernels, 7.4 ms portable (3.0 ms compile, 1.0 ms TensorRT) vs 20.4 ms original fp32
  (16.4 ms fp16); xlarge 9.4 ms kernels vs 45.7 ms. COCO val2017 (every 20th) box mAP tiny 49.0 kernels / 48.8
  portable vs 49.1 original, large 60.7 / 61.0 vs 60.7. BatchNorms are folded before float16 (some overflow it).

- MM-Grounding-DINO open-vocabulary detection (`mm-grounding-dino-tiny`, `mm-grounding-dino-base`, detection;
  Apache-2.0 code and weights): the Grounding DINO family with a scaled, biased class head and unshared decoder heads.
  800x1344 image: base 160 ms portable (132 ms compile) vs 345 ms original fp32 (233 ms fp16), peak 1.6 GB vs 2.7 GB;
  COCO val2017 (every 20th, zero-shot) box mAP tiny 54.9 vs 55.2, base 56.9 vs 57.1 original. TensorRT export fails.

- Checkpoint loads keep the checkpoint's value of every weight a model class ties (`untie_to_checkpoint` in
  `shared/seginfer`): transformers 5.19 ties MM-Grounding-DINO's per-layer decoder heads to layer 0 and silently
  dropped the trained heads of layers 1-5. A load now raises if any loaded value still differs from the checkpoint.

- RoMa dense matching (`romav1`, matching task; MIT code and weights, Apache-2.0 DINOv2 weights): RoMa outdoor,
  one direction per call, DINOv2 ViT-L/14 coarse features + VGG19 fine features, 560 / 864 squares as upstream's
  defaults. 768x1024 pair: 356 ms kernels, 376 ms portable (float16) vs 583 ms original fp32 (386 ms fp16), peak 1.7
  GB vs 4.5 GB; compile times out, the TensorRT export fails. MegaDepth-1500 (672 / 1344, upstream's benchmark)
  AUC@5/10/20 62.36/76.42/86.21 kernels, 62.61/76.63/86.37 portable vs 62.59/76.61/86.28 original fp16.

- StableNormal turbo surface normals (`stablenormal-turbo`, depth task, normals only; Apache-2.0 code and weights):
  one-step YOSO v0.3 on a Stable Diffusion 2 UNet with a ControlNet image encoder, seeded noise. 994x1482 image:
  180 ms kernels, 289 ms portable (238 ms compile) vs 613 ms original fp32 (299 ms fp16), peak 3.0 GB vs 7.9 GB;
  iBims-1 normals mean angle 17.46 deg for all modes and the original. TensorRT build times out.

- LoFTR semi-dense matching (`loftr-outdoor`, `loftr-indoor`, matching; Apache-2.0 code, the authors' weights
  inherit it): grayscale coarse-to-fine transformer matching, matches as keypoints plus index pairs. 768x1024 pair:
  242 ms portable (176 ms compile) vs 242 ms original fp32 (no fp16 in upstream), peak 1.7 GB vs 1.9 GB;
  MegaDepth-1500 (every 10th) AUC@5/10/20 52.1/71.0/83.3 vs 53.8/71.8/83.7 original. TensorRT export fails.

- DeDoDe v2 sparse matching (`dedode-v2-b`, `dedode-v2-g`, matching; MIT code and weights, DINOv2 of -G
  Apache-2.0): detector v2 + descriptor B or G, 10000 keypoints on a 784 x 784 square, dual-softmax matches as
  keypoints plus index pairs. 768x1024 pair: `dedode-v2-b` 247 ms portable (164 ms compile) vs 561 ms original fp32
  (318 ms fp16), peak 1.8 GB vs 3.3 GB; MegaDepth-1500 (every 10th) AUC@5/10/20 54.6/71.8/83.1 vs 57.3/71.6/82.5
  original fp16. TensorRT export fails (data-dependent match counts).

- AllTracker dense point tracking (`alltracker`, `alltracker-tiny`, tracking task; MIT code and weights): every pixel
  of a query frame tracked offline through the video as long-range flow (384 x 512 internally, as upstream's
  evaluation). 24-frame 480x854 video, 256 queries: `alltracker` 221 ms kernels, 391 ms portable (284 ms
  compile; the TensorRT export fails) vs 788 ms original fp32 (485 ms fp16), peak 1.1 GB vs 2.5 GB; TAP-Vid DAVIS
  (first) AJ 64.15 kernels, 64.14 portable vs 64.17 original.

- AnySplat feed-forward Gaussian splatting (`anysplat`, multiview; MIT code and weights, VGGT License for the VGGT
  copy): unposed, uncalibrated views to cameras, depth and `result.gaussians` (`.save("scene.ply")`), voxel-fused as
  upstream. 8 views 336x518: 364 ms kernels, 504 ms portable (427 ms compile, 314 ms TensorRT) vs 1769 ms original
  fp32 (741 ms bf16), peak 5.2 GB vs 7.6 GB. DTU (5 scans, median-scaled) AbsRel 0.0108 for kernels, portable and the original.

- DepthSplat feed-forward Gaussian splatting (`depthsplat-small-re10k`, multiview; MIT code and weights): posed,
  calibrated views with a depth range to depth and `result.gaussians`. The network is not scale invariant; the scene
  is scaled to near = 1 (upstream's RealEstate10K bounds) and back. 8 views 336x518: 120 ms kernels, 119 ms portable,
  82 ms compile, 86 ms TensorRT vs 177 ms original fp32 (127 ms fp16), peak 5.7 GB vs 5.8 GB. DTU (5 scans,
  calibrated) AbsRel 0.0974 kernels vs 0.0992 original fp32 (0.0974 fp16).

- TAPIR and BootsTAPIR point tracking (`bootstapir`, `tapir`, tracking task; Apache-2.0 code and weights): offline,
  the whole video per call (256 x 256 internally, as upstream). 24-frame 480x854 video, 256 queries: `bootstapir`
  191 ms kernels, 240 ms portable (float16) vs 396 ms original fp32 (243 ms fp16), peak 0.77 GB vs 0.83 GB;
  TAP-Vid DAVIS (first) AJ 62.27 kernels, 62.29 portable vs 62.27 original. Shared fused-conv ops now reject
  non-float16 operands instead of reading them as float16.

- TAPNext point tracking (`tapnext`, `tapnext-pp`, new `tracking` task; Apache-2.0 code and weights): `model(video,
  queries)` returns `tracks` `[T, Q, 2]` and `visible` `[T, Q]` for `(t, x, y)` queries, online one frame at a time
  as upstream. 24-frame 480x854 video, 256 queries: 382 ms kernels, 529 ms portable vs 1541 ms original fp32;
  TAP-Vid DAVIS (first) AJ 66.56 kernels vs 66.55 original.

- SAM 2.1 video (`track` of `sam2.1-hiera-*`; Apache-2.0 code and weights): objects prompted on the first frame
  (points, boxes or masks) are followed through a video with the original memory attention / memory encoder.
  Tiny on DAVIS 2017 val (480p): 59 ms/frame portable, 63 ms kernels vs 282 ms original fp32 (90 ms fp16), peak
  1.3 GB vs 2.1 GB; J&F 87.53 portable, 87.54 kernels and original.

- ViTMatte trimap matting (`vitmatte-s`, `vitmatte-b`; Apache-2.0 code and weights): `model(image, trimap=...)`
  returns the alpha; ViTDet backbone on image + trimap, conv detail-capture decoder. 1080x1920: `vitmatte-s` 84 ms
  kernels, 102 ms portable (55 ms TensorRT) vs 274 ms original fp32 (140 ms fp16), peak 1.4 GB vs 5.0 GB; `vitmatte-b`
  147 ms kernels (108 ms TensorRT) vs 552 ms (257 ms). AIM-500 SAD 13.72 / MSE 12.18 (`vitmatte-b` kernels, 500
  images), every 5th image equal to the original for all tiers.
- BEN2 matting / background removal (`ben2`; MIT code and weights): Swin v1 Base on the whole image plus four
  full-resolution patches, merged by multi-view cross-attention. 1080x1920: 173 ms kernels, 184 ms portable (58 ms
  TensorRT, 103 ms compile) vs 318 ms original fp32 (201 ms fp16). DIS-VD MAE 0.0403 / max F-beta 0.900 (kernels,
  470 images), every 2nd image equal to the original for all tiers.
- BiRefNet matting / background removal (`birefnet`, `birefnet-lite`, new `fastforks.matting` task; MIT code and
  weights): alpha `[H, W]` in [0, 1] at the input size; shared Swin v1 encoder (`fastforks.shared.swin_v1`). 1080x1920:
  `birefnet` 214 ms kernels (88 ms TensorRT) vs 382 ms original fp32 (234 ms fp16); `birefnet-lite` 59 ms vs 123 ms
  (75 ms). DIS-VD MAE 0.0325 / max F-beta 0.9074 (`birefnet` kernels, 470 images), every 2nd image equal to the
  original for all tiers.
- Video Depth Anything (`video-depth-anything-s`, `video-depth-anything-metric-s`, depth task; Apache-2.0 code and
  weights, Small only): a whole video per call (`[L, 3, H, W]`), windows of 32 frames with the upstream keyframe
  alignment. 88-frame 480x640 video: 9.9 ms per frame kernels vs 57.5 ms original fp32 (43.0 ms fp16); TUM RGB-D
  (3 sequences, first 110 frames) AbsRel 0.0549 kernels and original.

- Grounding DINO open-vocabulary detection (`grounding-dino-tiny`, `grounding-dino-base`; Apache-2.0 code and
  weights): class names as `text`, shared BERT tokenizer. Portable only. Tiny at 800x1344: 138 ms vs 274 ms original
  fp32 (199 ms fp16); COCO val2017 (every 10th) box mAP 51.14 vs 51.08 original.

- OWLv2 open-vocabulary detection (`owlv2-b16`, `owlv2-l14`; Apache-2.0 code and weights): `model(image,
  text=[...])` detects the given class names (default: the 80 COCO names; `model.open_vocabulary`); shared CLIP
  tokenizer. B/16 at 800x1344: 23.8 ms kernels vs 190 ms original fp32 (92 ms fp16); COCO val2017 (every 10th) box
  mAP 46.14 kernels and original.

- DEIM detection (`deim-{n,s,m,l,x}-coco`, in the `dfine` family; Apache-2.0 code and weights): the D-FINE
  network with SiLU decoder MLPs and the authors' checkpoints, on `portable` and `kernels`. L at 800x1344: 8.7 ms
  kernels vs 30.5 ms original fp32; COCO val2017 (every 10th) box mAP 54.67 kernels and original.

- Lotus normals (`lotus-normal-g-v1-1`, `lotus-normal-d-v1-1`, depth task, `normals` output; Apache-2.0): one step
  on the shared Stable Diffusion 2 UNet / VAE. 994x1482: 164 ms kernels vs 552 ms original fp32 (344 ms fp16);
  NYUv2 normals mean angular error 16.61 deg (Lotus-G kernels), first 100 images 16.815 kernels vs 16.815 original.
- Marigold normals LCM (`marigold-normals-lcm-v0-1`, depth task, `normals` output; Apache-2.0): one LCM step on the
  shared Stable Diffusion 2 UNet / VAE (`fastforks.shared.sd2`, also used by Marigold depth). 994x1482: 167 ms kernels vs
  557 ms original fp32 (280 ms fp16); NYUv2 normals mean angular error 18.46 deg kernels vs 18.45 original fp32.
- LightGlue sparse matching (`aliked-lightglue`, `disk-lightglue`; Apache-2.0, ALIKED BSD-3-Clause):
  `model(image1, image2)` returns `keypoints1`, `keypoints2`, index `pairs` and `scores` (the form
  `pycolmap.match_from_pairs` takes). Portable only (data-dependent keypoint counts). 768x1024: ALIKED 48.4 ms
  (29.5 ms with torch.compile) vs 41.2 ms original; DISK 116 vs 117 ms. MegaDepth-1500 AUC@5 50.7 (ALIKED).
- XFeat sparse matching (`xfeat`, `xfeat-lighterglue`; Apache-2.0): mutual nearest neighbours or LighterGlue
  (shared LightGlue). Portable only. 768x1024: 8.4 ms (5.7 ms compiled) vs 8.2 ms original; LighterGlue 22.5 vs
  22.4 ms. MegaDepth-1500 AUC@5 26.2 (`xfeat`), 39.7 (`xfeat-lighterglue`).

- MEMFOF optical flow (`memfof`, `memfof-kitti`, `memfof-sintel`, `memfof-spring`; BSD-3-Clause): portable and kernels (a 1024-channel
  `raft_flow` correlation lookup). 1080p: 427 ms kernels vs 1035 ms original fp32 (531 ms fp16); KITTI 2015 train
  EPE 10.89 / Fl 27.07 %. Two-frame API: the first frame is repeated as MEMFOF's previous frame.

- RF-DETR detection (N, S, M, L) and RF-DETR Seg (N, S, M, L, XL, 2XL; Apache-2.0 code and weights): boxes,
  labels, scores and instance masks without NMS, on `portable` and `kernels` (fastbb backbone). M at 800x1344:
  4.4 ms kernels vs 13.5 ms original fp32; COCO val2017 box mAP 54.51 kernels vs 54.53 original.

- New task `fastforks.matching`: dense matching. `model(image1, image2)` returns `warp` `[B, 2, H1, W1]` (each
  pixel of image1 in pixels of image2) and `certainty` `[B, H1, W1]`. RoMa v2 (`romav2`) on `portable` (bf16 on
  CUDA, as upstream) and `kernels` (one-pass local correlation): 412 ms kernels, 552 ms portable vs 741 ms original
  bf16 at 768x1024, with 2.9 instead of 10.1 GiB reserved.

- TensorRT export: bf16 transposed convolutions with stride == kernel (TensorRT has no bf16 deconvolution tactic)
  are rewritten as a 1x1 convolution plus DepthToSpace.

- Add FlowFormer Small optical flow: Chairs, Things and Sintel checkpoints with permissive code notices.

- Add FlowSeek-T optical flow: mixed-data, FlyingThings and FlyingThings without TartanAir checkpoints.

- Add GMFlow optical flow: nine scale-one, scale-two and recurrent-refinement checkpoints.

- Add NeuFlow v2 optical flow: mixed-data, Sintel and FlyingThings checkpoints with portable inference.

- RTMO bottom-up pose (t, s, m, l; Apache-2.0): `model(image)` finds people and their 17 COCO keypoints
  without person boxes, on `portable` and `kernels`. Includes measured original fp32/fp16, compile and TensorRT
  timings, GPU goldens, and full COCO val2017 accuracy with a matched original-code subset comparison.

- New task `fastforks.flow`: optical flow. `model(image1, image2)` returns `flow` `[2, H, W]` float32 (dx, dy in
  pixels of the input, from image1 to image2). RAFT (torchvision's weights: `raft-large`, `raft-large-things`,
  `raft-large-kitti`, `raft-small`) on `portable` and `kernels`, with the load option `iters` (refinement steps,
  default 12). RAFT on `kernels` builds no correlation volume (the lookup computes it on the fly from a pooled
  feature pyramid, O(HW) memory): about 2x portable fp16 at KITTI size, 3.6x at 1080p with 370 instead of 8400 MiB.
- New task `fastforks.pose`: top-down 2D human pose. `model(image, boxes=person_boxes)` returns the keypoints with
  scores per person (boxes from `fastforks.detection` or ground truth): 17 COCO body keypoints, or 133 whole-body
  ones (body, feet, face, hands) for DWPose.
- ViTPose and ViTPose++ (6 variants: base-simple, base, plus-small, plus-base, plus-large, plus-huge) and RTMPose
  (t, s, m, l at 256x192; m and l at 384x288), plus the DWPose whole-body weights for RTMPose (t, s, m, l at
  256x192; l at 384x288).
- Pose on `kernels` runs under CUDA graphs. RTMPose has the load option `flip_test` (average with the mirrored crop,
  two network passes).
- New detector RTMDet (OpenMMLab, Apache-2.0): `rtmdet-tiny-coco`, `-s-`, `-m-`, `-l-`, `-x-coco` (80 COCO classes,
  640 x 640) and the person detectors of the RTMPose project, `rtmdet-nano-person` (320 x 320) and `rtmdet-m-person`,
  on `portable` and `kernels`. Their boxes feed `fastforks.pose` directly.
- New classifier ConvNeXt (Meta's V1 weights from timm, Apache-2.0): tiny, small, base and large, each trained on
  ImageNet-1k (`convnext_base.fb_in1k`) and on ImageNet-22k fine-tuned on ImageNet-1k at 224
  (`convnext_base.fb_in22k_ft_in1k`) and 384 (`..._384`), on `portable` and `kernels`.
- New classifier RepViT (the authors' distilled ImageNet-1k weights from timm, Apache-2.0): `repvit_m0_9`, `m1_0`,
  `m1_1`, `m1_5` and `m2_3`, each with the 300- and the 450-epoch weights (`repvit_m1_1.dist_450e_in1k`), on
  `portable` and `kernels`. The re-parameterisable branches are fused at load.
- New classifier TinyViT (Microsoft's weights from timm, Apache-2.0): `tiny_vit_5m_224`, `tiny_vit_11m_224` and
  `tiny_vit_21m_224` (`.dist_in22k_ft_in1k`), the 21M one also at 384 and 512, on `portable` and `kernels`.
- New classifier SHViT (the authors' weights from timm, MIT): `shvit_s1.in1k` to `shvit_s4.in1k` (S4 at 256), on
  `portable`.
- New classifier HGNetV2 (PaddleClas' SSLD weights from timm, Apache-2.0): `hgnetv2_b0.ssld_stage2_ft_in1k` to
  `hgnetv2_b6.ssld_stage2_ft_in1k`, on `portable`.
- New classifier StarNet (the authors' weights from timm, Apache-2.0): `starnet_s1.in1k` to `starnet_s4.in1k`, on
  `portable` and `kernels`; explicit CNN dispatch folds BatchNorm despite timm naming its blocks `Block`.
- New classifier EfficientFormerV2 (Snap Research's weights from timm, Apache-2.0): `efficientformerv2_s0`, `_s1`,
  `_s2` and `_l` (`.snap_dist_in1k`), on `portable`.
- New classifier EfficientFormer (Snap Research's weights from timm, Apache-2.0): `efficientformer_l1`, `_l3` and
  `_l7` (`.snap_dist_in1k`), on `portable`.
- New classifier CoAtNet (timm's own `rw` weights, Apache-2.0): `coatnet_nano_rw_224.sw_in1k` to
  `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (10 checkpoints), on `portable`.
- New classifier FocalNet (the authors' weights from timm, MIT): `focalnet_tiny_srf.ms_in1k` to
  `focalnet_base_lrf.ms_in1k` (6 checkpoints), on `portable`.
- New classifier DaViT (the authors' weights from timm, Apache-2.0): `davit_tiny.msft_in1k`, `davit_small.msft_in1k` and
  `davit_base.msft_in1k`, on `portable`.
- New segmentation family SegNeXt (the MMSegmentation ADE20K checkpoints, Apache-2.0): `segnext-mscan-t-ade20k`, `-s`,
  `-b`, `-l`, on `portable`. The head's matrix factorisation starts from fixed bases, so results repeat (upstream
  draws random ones per call).
- New feature backbones: SigLIP 2, CLIP / OpenCLIP, Perception Encoder and C-RADIOv3.
- New feature backbone: TIPSv2 (B/14, L/14 image encoders; the DINOv2 kernel path).
- New feature backbone ConvNeXt (Meta's V1 ImageNet-22k weights from timm, Apache-2.0): `convnext_tiny.fb_in22k`,
  `convnext_small.fb_in22k`, `convnext_base.fb_in22k`, `convnext_large.fb_in22k`, on `portable` and `kernels`.
  `last_hidden_state` is the stride-32 map of the last stage as tokens (no prefix token), `pooler_output` the
  normalised mean. The maps of the earlier stages are not returned.
- New depth models: Depth Anything V2 Metric (Hypersim indoor, max 20 m; Virtual KITTI outdoor, max 80 m; each S, B,
  L), which return metric `depth` in metres, and Distill-Any-Depth (S, B, L; relative `inverse_depth`, Apache-2.0
  weights), on `portable` and `kernels`.
- New promptable-segmentation family EfficientViT-SAM (MIT Han lab, Apache-2.0 code and weights):
  `efficientvit-sam-l0`, `-l1`, `-l2` (512 input) and `-xl0`, `-xl1` (1024 input), on `portable` and `kernels`.
  Same call and prompts as SAM (points, boxes, masks). On the RTX 5060 Ti the full call with one point takes
  4.4 ms (L0) to 15.3 ms (XL1) on `kernels`, against 26.9 and 72.6 ms for the original in float32.
- Fix: the `kernels` conv optimizer no longer hands a depthwise convolution followed by GELU to the `fvm_r3`
  kernel, which has no GELU (EfficientViT-SAM is the first model with the pattern in its fused graph).
- New promptable-segmentation family HQ-SAM (ETH Zurich, Apache-2.0 code and weights): `sam-hq-vit-b`, `-l`, `-h`,
  on `portable` and `kernels`. SAM's ViT encoder and decoder plus a small HQ head; one call returns one mask per
  prompt (SAM's best mask plus the HQ mask, as upstream's default). Same call and prompts as SAM. On the RTX 5060 Ti
  the full call with one point takes 36.8 ms (B) to 143.5 ms (H) on `kernels`, against 172 and 738 ms for the
  original in float32.
- New promptable-segmentation family MobileSAM (Kyung Hee University, Apache-2.0 code and weights): `mobile-sam`, on
  `portable` and `kernels`. SAM's prompt encoder and mask decoder with a small TinyViT image encoder (1024 input).
  Same call and prompts as SAM. On the RTX 5060 Ti the full call with one point takes 6.9 ms on `kernels`, against
  30.3 ms for the original in float32; COCO mIoU is the original's (51.53 best of 3).
- Fix: the ONNX export of EfficientViT-SAM no longer produces NaN. The exporter dropped the `+ 1e-15` of the linear
  attention's division as a no-op, so zero sums gave 0 / 0 in ONNX Runtime and TensorRT; the epsilon is now a tensor
  the exporter keeps. Torch results are unchanged. The fp16 TensorRT engine is finite but misses the golden
  tolerance (fp16 sums in the wide 1x1 convolutions); an engine of the model loaded with `dtype="float32"` matches.

## 0.1.1

Fixes and faster kernels. No new models.

- `kernels` is faster on GPUs that run fp16 tensor-core math with fp32 accumulation at full rate (NVIDIA L4, A10,
  A10G, L40S, A100, RTX PRO 6000): the DPT kernels (new `dpt_fused` build) use larger tiles for the deep layers and
  a tanh GELU there. Depth Pro and UniK3D keep the exact GELU. GeForce cards run as before.
- Marigold on `kernels` runs GroupNorm and SiLU on the channels-last map, without layout copies. The MoGe-3 refiner
  runs in fp16 on CUDA. RT-MonSter++ and CFNet on `kernels` no longer fall into slow paths on datacenter cards.
- TensorRT: the FoundationStereo cost volumes export without per-disparity loops. `c-fast-foundationstereo` at
  994 x 1482 on the RTX 5060 Ti takes 156 ms instead of 449 ms; the output is within fp16 rounding of `portable`.
- The fp16 overflow guard of the attention kernels syncs with the host once per model call instead of once per
  launch, which shortens the first calls of every guarded model. It also checks the warm-up before a CUDA graph is
  captured, so a graph never records a variant that overflows.
- Fix: in models with several attention launchers (`vggt-1b`, `depth-pro`), a launcher could run out of checked
  launches after reruns and stay unchecked.
- Fix: `Model.compile()` on segmentation models compiled the resize into the class maximum, three times slower than
  eager (SegFormer). The label step now runs eagerly inside the compiled model.
- Fix on Apple MPS: `unproject` and the stereo cameras move points to the CPU before float64 maths, and `save()` /
  `save_scene()` apply the `max_depth` and mask filters on the points' device.
- SAM 3, DINOv3, DINOv3 ConvNeXt and VGGT-Omega are validated on Apple MPS in fp32 and fp16 (`portable`).
- `fastforks download` no longer stops at the first model or file it cannot fetch (a gated repo without access, a
  host that is down). It fetches everything else, lists the failures as `MISSING` lines and exits with code 1.
- Changed: the weight files of Depth Anything 3 and MoGe have unique names (`DA3-LARGE-1.1/model.safetensors`,
  `moge-3-vitl/model.pt`) instead of `model.safetensors` / `model.pt` for every variant. A local weights folder
  (`FASTFORKS_WEIGHTS_PATH`) can now hold all variants; files placed there by hand under the old names must be moved.
- Docs: speed surveys per device (Apple M1 Pro, NVIDIA L4, A10, A10G, L40S, A100, RTX PRO 6000) and accuracy
  against the original code for depth, stereo, segmentation, detection, promptable segmentation, features,
  classification and multi-view depth.

## 0.1.0

First release.

- 174 models in eight tasks: stereo, depth, classification, segmentation, features, promptable segmentation,
  detection and multi-view depth.
- Two backends per model: `portable` (plain PyTorch, runs on CPU, any CUDA GPU and Apple MPS) and `kernels` (custom CUDA kernels
  for NVIDIA GPUs of compute capability 8.0 or newer; Metal kernels for some families on Apple silicon). `mode="auto"` picks kernels where they run and the recorded measurements prefer them.
- Typical speed-up of kernels over portable: median 1.9x across the models and sizes measured, a quarter at 2.5x or more, best 10x (small classifiers at batch 1). Speed and memory per model are in the
  [benchmarks](docs/benchmarks.md) and the [overview](docs/overview.md).
- Weights are downloaded from the authors' hosts at pinned revisions and checked by SHA-256; every model carries its
  code licence, weights licence and commercial-use status.
- `pip install fastforks` installs everything (every model, timm, torchvision, transformers, ONNX export).
  `pip install fastforks-core` is the lean install with every model whose code is permissive; `[nc]` adds the models
  with non-commercial or GPL code, `[libs]` the optional libraries. Non-commercial weights are downloaded only with
  `FASTFORKS_NONCOMMERCIAL=1`, in every install.
- ONNX and TensorRT export of the portable backend, stereo geometry (rectification, point clouds), command line
  (`list`, `info`, `bench`, `export`, `download`, `doctor`, `env`, `stereo run`).
