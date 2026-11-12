# Ports in progress: who does what, and where each network stands

Status (2026-11-10): working notes for the branch `work/ports`, which sits on top of 0.1.1. Two machines work on it:

- **CUDA machine (RTX 5060 Ti):** ports a network, tunes its CUDA kernels and records its rows. It owns the model
  code, the CUDA kernels and every file in `docs/data/` except the Mac speed file.
- **Mac (M1 Pro):** takes a network once the CUDA machine is done with it: MPS validation, Metal kernels where they
  pay, and the Mac rows in `docs/data/speed/apple-m1-pro-16gb.jsonl`. It does not change the portable model code
  beyond what MPS needs.

## Rules for the branch

- One commit per finished network, not before. Finished means: `ruff format --check`, `ruff check`, `ty check`,
  `tools/gen_docs.py --check` and `pytest -m "not gpu"` pass, the GPU goldens are recorded (`pytest -m gpu --record`),
  and the rows below exist.
- `git pull --rebase` before every push. Never force-push.
- A licence gate comes first: code and weights licences are checked against the source before any work starts, and
  a non-commercial weight licence stops the port (see [model-landscape.md](model-landscape.md)).
- Rows every network needs (the statistics pages reuse them): original fp32 and fp16, `portable`, `kernels`,
  `torch.compile` and TensorRT, each with peak memory, and accuracy against the original. A failed export is a row
  too.
- Notes on how to tune a network differently on datacenter GPUs go to
  [datacenter-tuning.md](datacenter-tuning.md).
- When the CUDA machine finishes a network it adds a line to "Waiting for the Mac" below. The Mac removes the line
  when its part is committed.

## Open on the CUDA machine

In this order, unless something more useful turns up:

1. CLIP: `clip-l14-dfn2b`, `clip-h14-dfn5b` and `clip-h14-378-dfn5b` have no `torch.compile` or TensorRT rows in
   `docs/data/speed_compile.jsonl`.
2. Next networks, from [model-landscape.md](model-landscape.md): MaxViT, `moge-2-vitl` (no normal head), and more
   flow and pose models. ConvNeXt features, TIPSv2, RepViT and MobileSAM are already merged.

## Waiting for the Mac

- AnySplat (`anysplat`, multiview with `result.gaussians`): portable float32 MPS goldens and Mac speed rows (8 views
  336x518). No Metal kernels runner. Weights: the pinned HF URL of the card.
- DepthSplat (`depthsplat-small-re10k`, multiview with `result.gaussians`): portable float32 MPS goldens and Mac speed
  rows (8 views 336x518, calibrated). No Metal kernels runner. Weights: the pinned HF URL of the card.
- TAPNext (`tapnext`, `tapnext-pp`): portable float32 MPS goldens and Mac speed rows (24-frame 480x854 video, 256
  queries). No Metal kernels runner. Weights: the pinned URLs of the card (also in `data/ff-weights/tapnext/`).
- TAPIR (`bootstapir`, `tapir`): portable float32 MPS goldens and Mac speed rows (24-frame 480x854 video, 256
  queries). No Metal kernels runner. Weights: the pinned URLs of the card (also in `data/ff-weights/tapir/`).
- SAM 2 video (`track` of `sam2.1-hiera-*`): portable float32 MPS video goldens (`video_*` keys of
  `tests/test_golden.py`); the memory features are stored in bfloat16 as upstream.
- ViTMatte (`vitmatte-s`, `vitmatte-b`, matting with a trimap input): portable float32 MPS goldens and Mac speed
  rows at 1080x1920. No Metal kernels runner. Weights: the pinned HF URLs of the card.
- BEN2 (`ben2`, matting): portable float32 MPS goldens and Mac speed rows at 1080x1920. No Metal kernels runner.
  Weights: the pinned HF URL of the card (also in `~/scratch_tests/port-p1/w/ben2/`).
- BiRefNet (`birefnet`, `birefnet-lite`, new matting task; new `shared/swin_v1`): portable float32 MPS goldens and
  Mac speed rows at 1080x1920. No Metal kernels runner (CUDA kernels: `deform_conv` in the ASPP branches). Weights: the
  pinned HF URLs of the card (also in `~/scratch_tests/port-p1/w/birefnet/`).
- Video Depth Anything (`video-depth-anything-s`, `video-depth-anything-metric-s`): portable float32 MPS goldens and
  Mac speed rows (an 88-frame 480x640 video). No Metal kernels runner. Weights: the pinned HF URLs of the card (also in
  `~/scratch_tests/port-p1/w/video_depth_anything/`).
- Grounding DINO (`grounding-dino-tiny`, `grounding-dino-base`): portable float32 MPS goldens and Mac speed rows at
  800x1344; needs transformers. No Metal kernels runner. Weights: the pinned HF URLs of the card (also in
  `data/ff-weights/grounding_dino/`).
- OWLv2 (`owlv2-b16`, `owlv2-l14`): portable float32 MPS goldens and Mac speed rows at 800x1344; needs transformers.
  No Metal kernels runner. Weights: the pinned HF URLs of the card (also in `data/ff-weights/owlv2/`).
- DEIM (`deim-{n,s,m,l,x}-coco`, in the dfine family): portable float32 MPS goldens and Mac speed rows at 800x1344.
  No Metal kernels runner. Weights: the pinned Google Drive URLs of the card (also in `data/ff-weights/dfine/`).
- Lotus normals (`lotus-normal-g-v1-1`, `lotus-normal-d-v1-1`): portable float32 MPS goldens and Mac speed rows
  (994x1482). No Metal kernels runner. Weights: the pinned HF URLs of the card (also in
  `~/scratch_tests/port-p1/w/lotus/`).
- Marigold normals LCM (`marigold-normals-lcm-v0-1`): portable float32 MPS golden and Mac speed rows (994x1482). No
  Metal kernels runner. Weights: the pinned HF URLs of the card (also in `~/scratch_tests/port-p1/w/marigold_normals/`).
- XFeat (`xfeat`, `xfeat-lighterglue`): portable float32 MPS goldens and Mac speed rows at 768x1024. No Metal kernels
  runner. Weights: the pinned GitHub URLs of the card (also in `~/scratch_tests/port-p1/w/xfeat/`).
- LightGlue (`aliked-lightglue`, `disk-lightglue`): portable float32 MPS goldens (sparse: keypoint recall and match
  agreement) and Mac speed rows at 768x1024. No Metal kernels runner. Weights: the pinned GitHub release URLs of the
  card (also in `~/scratch_tests/port-p1/w/lightglue/`).
- MEMFOF (`memfof`, `memfof-kitti`): portable float32 MPS goldens and Mac speed rows (375x1242, 1080p). No Metal
  kernels runner. Weights: the pinned HF URLs of the card (also in `~/scratch_tests/port-p1/w/memfof/`).
- RF-DETR detection / Seg (10 variants): portable float32 MPS goldens and Mac speed rows at 800x1344. No Metal
  kernels runner. Weights: the pinned Roboflow URLs of the card (also on CUDA in `~/scratch_tests/port-p1/w/rfdetr/`).
- RoMa v2 (`romav2`, new matching task): portable float32 MPS goldens (`warp` checked where the golden certainty
  is >= 0.5) and Mac timings at 768x1024. No Metal kernels runner. Weights: the pinned GitHub release URL (1.1 GB).
- RTMPose zoo (7 Halpe26, hand5, 3 face6, AP-10K): MPS golden checks and portable speed rows.
  COCO accuracy covers Halpe26 body joints; hand, face and animal validation datasets are not staged.
  Weights on CUDA in `~/scratch_tests/port-rtmo/w/`.

- RTMW whole-body (5 variants): MPS golden checks, especially the 384-pixel variants, and portable speed rows.
  Weights on CUDA in `~/scratch_tests/port-rtmo/w/` or its `cache/`, otherwise the pinned upstream URLs.
- HGNetV2 B0-B6: portable fp32 MPS goldens and speed at native 224 crops. CUDA portable goldens use
  measured mean/p99 bounds 0.0080/0.033; weights are downloadable from the pinned timm Hugging Face card URLs.
- EfficientFormerV2 S0/S1/S2/L: portable fp32 MPS goldens and speed at native 224 crops. CUDA portable
  mean/p99 bounds are 0.12/0.42 from measured fp16 errors; pinned timm Hugging Face weights are downloadable.
- CoAtNet: validate MPS float32 at native 224/384 crops. CUDA portable FP16 is checked; no custom kernels
  runner exists. Download weights from the pinned card URLs; owned CUDA caches are removed after validation
  to stay within the scratch budget.
- FocalNet: validate portable MPS float32 and record Mac timings. No custom kernels runner exists.
- DaViT: validate portable MPS float32 and record Mac timings. No custom kernels runner exists.
  Download weights from the pinned card URLs; owned CUDA caches are removed after validation to stay
  within the scratch budget.

- EfficientFormer L1/L3/L7: portable fp32 MPS goldens and speed at native 224 crops. CUDA portable passes
  measured mean/p99 bounds 0.0046/0.019; pinned timm Hugging Face weights are downloadable.

- SegNeXt: validate portable MPS float32 and record Mac timings. No custom kernels runner exists.
  Download weights from the pinned card URLs; owned CUDA caches are removed after validation to stay
  within the scratch budget.

- SHViT S1-S4: portable fp32 MPS goldens and speed at native 224 crops (S4: 256). CUDA portable goldens pass
  with mean/p99 bounds 0.0060/0.023; weights are downloadable from the pinned timm Hugging Face card URLs.
- FlowFormer Small, all three variants: MPS float32 goldens and portable timing/memory; weights in
  `~/scratch_tests/codex-flow-finish/weights/flowformer-small/` or the pinned authors' archive.

- FlowSeek-T, all three variants: MPS float32 goldens and portable speed/memory; pinned weights also in
  `~/scratch_tests/port-flow2/weights/`.

- GMFlow, all nine variants: MPS float32 goldens and portable speed/memory; CUDA weights are in
  `~/scratch_tests/port-flow2/weights/` or available through the pinned authors' URLs.

- NeuFlow v2, all three variants: MPS float32 goldens and portable speed/memory rows. CUDA weights are in
  `~/scratch_tests/port-flow2/weights/` (symlinks to its cache), or use the pinned authors' URLs.

- RTMO (4 bottom-up pose variants): MPS goldens, person counts near the score threshold, and portable speed.
  Weights in `~/scratch_tests/port-rtmo/w/` or its `cache/` on CUDA, otherwise the pinned upstream URLs.
- HRNet top-down pose (w32, w32-384, w48, w48-384): MPS goldens and portable speed rows. CUDA bounds
  mean/p99 0.32/4.8 px; weights in `~/scratch_tests/port-next/w/`, otherwise the pinned card URLs. The
  top-down crop code moved from rtmpose to `shared/topdown`: recheck the RTMPose MPS goldens once.
- StarNet S1-S4: portable fp32 MPS goldens and speed at native 224 crops after explicit CNN dispatch. CUDA
  portable/kernels pass measured mean/p99 bounds 0.0035/0.012; pinned timm Hugging Face weights are downloadable.

- TinyViT, five classifiers: run portable fp32 MPS goldens and speed at native 224/384/512 crops. CUDA GPU goldens
  passed for portable and kernels; weights are downloadable from the pinned timm Hugging Face URLs in the card.

- RF-DETR Keypoint Preview (`rfdetr-keypoint`, new `shared/rfdetr`): MPS golden (CUDA mean/p99 bounds 0.072/0.17 px
  portable) and portable speed at 576. Weights `~/scratch_tests/codex-pose/rf-reference/` on CUDA, else the pinned URL.

- CSPNeXt moved to `shared/cspnext` (RTMDet, RTMPose/RTMW/DWPose): rerun the rtmdet and rtmpose MPS goldens and Metal
  kernels speed rows on the Mac. CUDA: all 35 goldens pass unchanged, speed A/B equal to the previous commit.

- FastViT T8 / T12 (`fastvit_t8.apple_dist_in1k`, `fastvit_t12.apple_dist_in1k`): portable fp32 MPS goldens and speed at
  the native 256 crop. CUDA bounds mean/p99 T8 0.023/0.069, T12 0.032/0.055; pinned timm Hugging Face weights.

Done on 2026-11-10 for every family on the branch: Mac speed rows (`portable`, and `kernels` where Metal kernels
exist) and the MPS goldens, all passing. Still open:

- TIPSv2 `b14`, `l14` (new family, features): MPS goldens (cpu_atol 6.9e-05, gpu mean 0.019 / p99 0.069) and
  portable fp32/fp16 speed rows. Weights from the pinned HF URLs.
- MobileSAM `mobile-sam` (new, promptable segmentation; new `shared/sam_prompt`): MPS golden and portable speed.
  Weights `~/scratch_tests/port-mobilesam/weights/mobile_sam.pt` on the CUDA machine, or the pinned URL.
- RepViT, 10 variants (new family, classification; changes shared `_optimize/fold.py` reparam and
  `_core/deferred_init.py`): portable fp32 MPS goldens of all 10, check that fused weights load through
  `deferred_init`. Weights `~/scratch_tests/port-repvit/weights/` or the pinned timm URLs.
- ConvNeXt features `convnext_{tiny,small,base,large}.fb_in22k` (new family; the kernels trunk moved to
  `shared/convnext_trunk`): portable fp32 MPS goldens of the 4 refs, `tests/test_convnext_features.py::test_layout_cpu`,
  and classification ConvNeXt on MPS after the move. Weights `~/scratch_tests/port-convnext1/w/` or the pinned URLs.
- MoGe-2 `moge-2-vitb-normal` (new variant of the existing depth family `moge`): MPS golden (cpu_atol 0.005, gpu
  portable mean / p99 0.019, kernels 0.033) and portable fp32/fp16 speed rows. Weights from the pinned HF URL.
- Perception Encoder `g14`: the peak footprint (about 17.8 GB) is above the 16 GB of the machine, so the times
  include swapping.
- SEA-RAFT `spring` variants run about ten times faster than the others at the same labelled input; not looked into.
- Metal kernels for the families that have none (flow, the feature backbones except Perception Encoder, RTMDet,
  ConvNeXt, the SAM variants): not tried yet.

## Known gaps that wait for a release

- The `dpt_fused` build `48e0065bb7cf` for `sm_120` is not on Hugging Face yet; kernel builds are uploaded at
  release time.
- Licence questions still open: the Depth Anything V2 Metric B / L weights and Distill-Any-Depth L.
