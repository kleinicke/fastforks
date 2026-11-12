# Changelog

## Unreleased

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
