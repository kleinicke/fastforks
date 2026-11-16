# Design: the next fastforks: new tasks, Windows, ComfyUI, customisation and training

Status (2026-11-09): design only. `fastforks.pose` (ViTPose, RTMPose) and the CLIP / SigLIP 2 / Perception Encoder /
C-RADIO feature backbones are in progress on `docs/device-benchmarks`; everything else here is not built. Which
models go into which task, in which order, is in [roadmap.md](roadmap.md); the Gaussian splatting library is in
[splat-library.md](splat-library.md).

Order of work: fastforks comes first, until we judge it good enough for now; then development moves to the splat
library. The milestones at the end are an order, not a schedule, and what is still open at the switch stays on the
list.

## What belongs in fastforks

A topic belongs in fastforks when all four hold:

1. **Inference of a fixed network**: no per-scene optimisation, no long-lived state beyond a video memory the
   model itself defines.
2. **Images or video in, tensors out**, checked against the authors' code (golden tests), with the authors' output
   as the contract.
3. **Published weights** whose code and weights licences can be stated per model (`commercial use` flag as today).
4. **Kernel reuse**: the network is mostly built from blocks our kernels already cover (ViT / DINO blocks, CNN
   convolutions, correlation volumes + GRU updates, DETR decoders, Swin / hierarchical transformers). A task that
   needs a new kernel family must say so in the roadmap; it costs more.

Out of scope, by decision: per-scene methods (Gaussian splatting, NeRF → the splat library), image / video
generation (diffusion; crowded and a different optimisation problem), vision-language models (served by LLM engines),
stateful systems (SLAM, SfM pipelines; they may *use* fastforks), face recognition (legal and ethical baggage; face
detection and landmarks are fine).

Medical imaging follows the same rules. 2D medical models (MedSAM, Cellpose-SAM, pathology and X-ray encoders) are
ordinary model cards in existing tasks, marked research use only. 3D CT / MR volume models (TotalSegmentator,
VISTA3D, nnU-Net checkpoints) need 3D kernels, sliding-window inference and NIfTI input, so they wait for a possible
`volume` task. The reasoning and the candidates are in
[model-landscape.md](model-landscape.md#medical-imaging-decision).

## New tasks and their API

Every new task follows the existing pattern: `task.load(model_id, mode=..., device=..., dtype=...)` returns a
callable model; the result object holds tensors in input pixel coordinates, `.backend`, and `to_list()` / `save()`
where useful; `module()` gives the traceable portable part; a `TaskSpec` in `_core/tasks/` records the standard
output keys, inputs and export size. Proposed contracts:

### `fastforks.flow`: optical flow

```python
from fastforks import flow

model = flow.load("sea-raft-m")
result = model("frame_0.png", "frame_1.png")
result.flow  # [2, H, W] float32, pixels, (dx, dy) from image 1 to image 2
result.uncertainty  # [H, W] if the model predicts it (SEA-RAFT), else None
result.save("flow.flo")  # also .png colour coding via fastforks.flow.visualize
```

- Video: `model.video(frames)` returns the flow between consecutive frames, reusing the previous frame's features
  (the same encoder output serves two pairs).
- Kernel reuse: RAFT-family correlation volumes, lookup and GRU update kernels from `raft`, `igev`, `crestereo`
  (stereo is flow restricted to one dimension); ViT encoders for WAFT-style models. This is the cheapest new task
  per model.
- Golden metric: EPE / Fl-all on Sintel and KITTI 2015 public frames (accuracy check, not a leaderboard, as for
  stereo).

### `fastforks.tracking`: point tracking

```python
from fastforks import tracking

model = tracking.load("tapnext")
result = model("clip.mp4", queries=[[0, 120.0, 340.0], [10, 400.0, 220.0]])  # (frame, x, y)
result = model("clip.mp4", grid=20)  # or a regular grid on frame 0
result.tracks  # [N, T, 2] pixels
result.visible  # [N, T] bool
result.confidence  # [N, T] if the model predicts it
```

- Online models (TAPNext, BootsTAPIR's causal mode) also expose `model.stream()` returning a stateful tracker fed
  frame by frame.
- Defaults are permissive (TAPNext and BootsTAPIR: Apache-2.0 code and weights; AllTracker: MIT). CoTracker3 is
  CC-BY-NC 4.0 and ships only as a `[nc]` option.
- Kernel reuse: correlation sampling and transformer blocks; video decoding from `_core/io.py`.

### `fastforks.matching`: keypoints and matches

One task for sparse and dense matchers, one result type:

```python
from fastforks import matching

model = matching.load("aliked-lightglue")  # sparse: detector + matcher
result = model("a.jpg", "b.jpg")
result.keypoints0, result.keypoints1  # [M, 2] matched pixel positions
result.scores  # [M]
features = model.detect("a.jpg")  # keypoints, descriptors, scores (reusable across pairs)

dense = matching.load("roma-outdoor")  # dense: warp + certainty
result = dense("a.jpg", "b.jpg", max_matches=5000)
result.warp, result.certainty  # [H, W, 2], [H, W]; keypoints0/1 are sampled from them
```

- Feeds `multiview` (calibrated MVS needs poses) and the splat library (poses without COLMAP).
- Kernel reuse: RoMa's DINOv2 encoder (kernels exist), attention (LightGlue), CNN backbones (ALIKED, DISK, XFeat).
- Licences: SuperPoint is non-commercial, so the defaults pair LightGlue with ALIKED or DISK; MASt3R / DUSt3R
  (CC BY-NC-SA) and UFM (non-commercial weights) are `[nc]` options at most.

### `fastforks.pose`: keypoints (in progress)

Already designed and in progress (top-down with `boxes=` from a detector). Additions planned:

- **Bottom-up / one-stage models** (RTMO): no boxes needed; `model(image)` returns every person.
- **Whole-body, hand, face and animal skeletons** through the card's `label_set` (COCO-WholeBody 133, hand 21,
  AP-10K 17), so the result type does not change.
- **Convenience**: `pose.load(id, detector="rtdetrv2-s-coco")` runs the person detector itself when no boxes are
  passed; explicit, logged, never a silent default for top-down models.
- Not in `pose`: camera pose (in `multiview`), 3D human mesh (not planned: SMPL and MANO forbid commercial use),
  6D object pose (refine-and-render loops, not a single forward pass; the strongest models are non-commercial).
- Licences: ViTPose and the RTMPose family are Apache-2.0 (the RTM models' body7 training-data terms still need a
  check); Sapiens is CC-BY-NC (`[nc]`); Ultralytics pose models are AGPL-3.0 and never added.

### `fastforks.matting`: alpha mattes and background removal

```python
from fastforks import matting

model = matting.load("birefnet-general")
result = model("product.jpg")
result.alpha  # [H, W] float32 in [0, 1]
result.cutout()  # RGBA image
vitmatte = matting.load("vitmatte-s")
vitmatte("portrait.jpg", trimap="trimap.png")  # trimap-based models take a trimap (or a SAM mask)
```

- Kernel reuse: Swin / PVT backbones (BiRefNet, BEN2), ViT (ViTMatte), segmentation decoders.
- Licences: BiRefNet, BEN2 and ViTMatte are MIT / Apache for code and weights. BRIA RMBG-2.0 has a custom, gated
  licence and MatAnyone (video) is non-commercial: neither is a default.

### Additions to existing tasks

- `detection`: newer real-time DETRs (RF-DETR N/S/M/L, DEIM, LW-DETR) and **open-vocabulary detection**
  (`model(image, text=["cup", "laptop"])`, Grounding DINO and OWLv2). Pairs with SAM 3 and the CLIP / SigLIP
  backbones; reuses the DETR decoder kernels. Not added: DEIMv2 (non-commercial), the RF-DETR Plus sizes (Roboflow
  licence), YOLO-World (GPL-3.0), YOLOE and Ultralytics (AGPL-3.0).
- `promptable_segmentation`: **SAM 2 video** (`model.track(video, prompts={0: ...})` → masks per frame per object),
  the memory attention as kernels.
- `depth`: **video depth** (`model(frames)` with temporal consistency; Video Depth Anything Small is Apache-2.0,
  Base / Large are CC-BY-NC), **normals** as a first-class output (`result.normals`): the existing models that
  predict them (MoGe, Metric3D) plus Lotus, Marigold normals and StableNormal.
- `multiview`: **feed-forward Gaussian splatting** (AnySplat, DepthSplat, MVSplat) returning the predicted
  Gaussians as tensors (`result.gaussians`: means, scales, rotations, opacities, colours) and a PLY writer;
  rendering stays in the splat library. That also drops the Inria-derived rasteriser most of these repos bundle.
  NoPoSplat depends on DUSt3R / CroCo (non-commercial).

### Optional tasks (only on demand)

`restoration` (super-resolution, denoising), `interpolation` (video frame interpolation), `ocr` (text detection +
recognition), `face` (detection + landmarks, no recognition), `video_classification`. These fit the contract but
serve other audiences; the ComfyUI node pack is the most likely reason to build `restoration` and `interpolation`.
The [model landscape](model-landscape.md#restoration-and-interpolation-roadmap-optional-tasks-raised-to-p2) raises
`restoration` and `interpolation` to P2, and adds a small `calibration` task (AnyCalib, GeoCalib) at P2.

## Customisation without losing speed

Today: `weights=PATH` (fine-tunes of the same architecture), `labels=` (renamed classifier heads), `patch()`
(own timm / torchvision / transformers classifiers), `module()` (plain PyTorch of the portable model, trainable and
exportable). Kernels have no backward pass.

Planned, in order of value per cost:

1. **Frozen kernel backbone + trainable head.** A documented pattern and small API: `features.load(...)` (and the
   encoders of depth / stereo / segmentation models via `model.encoder()`) run on kernels without gradients and
   return tensors; the user's head is ordinary PyTorch with autograd. Covers linear probes, segmentation / depth /
   pose heads on DINOv3, and task-specific adapters. Costs almost nothing: the backbone already runs on kernels.
2. **Resized heads.** `load(..., weights=..., num_classes=...)` (classification, segmentation, detection) accepts a
   head of another size. The kernels are in the backbone; the head is small and can run in its portable form or in a
   generic kernel. Today the head shape check rejects these checkpoints.
3. **`fastforks.ops`**: public, documented building blocks (ViT block, attention, fused GEMM + epilogue, the
   correlation sampler, the deformable convolution) with stable signatures, for people building their own networks.
   Forward only; each op raises if an input requires grad.
4. **A broader `patch()`**: recognise known blocks (ViT blocks, ConvNeXt / ResNet stages, attention) in *any*
   `nn.Module`, replace them with kernel versions, leave unknown parts as PyTorch. `.report` lists which modules run
   on kernels and why the others do not. This follows the no-silent-fallback rule: what is not replaced is reported,
   and `mode="kernels"` raises if nothing could be replaced.
5. **Pre- and post-processing parameters** stay options (iterations of recurrent models, thresholds, input size);
   they never change which kernels run.

Rule for deeper changes: changing the architecture inside a kernel region means that region runs portable, reported
in `.report`, never silently.

## Training and fine-tuning

Kernels are forward-only, and writing backward kernels for every family roughly doubles the kernel work. Staged:

1. **Now**: train the portable `module()` (optionally with `torch.compile`), load the result back with
   `weights=`. Documented in the usage guide.
2. **Fine-tuning recipes** (no new kernels): per task a data loader for the standard formats, the paper's loss
   (stereo sequence loss, depth scale-invariant log loss, keypoint heatmap loss, flow EPE), LoRA for ViT
   backbones, and an evaluation hook that runs the fastforks golden metric. `fastforks finetune MODEL --data ...`.
3. **Head training on frozen kernel backbones** (see customisation 1): fast training of heads at inference speed.
4. **Backward kernels** for the most-used blocks (ViT block, attention, GEMM epilogues), at the end of the roadmap,
   informed by the splat library, which needs backward kernels from day one.

## Windows

Most of the kernels need only loader work: 18 of 30 kernel libraries are cubins (driver-API modules), which do not
depend on the OS; 12 are shared libraries (`.so` through ctypes: the CUTLASS and cuDNN ones) and need `.dll`
builds.

Steps:

1. **Portable backend on native Windows**: run `pytest -m "not gpu"` and the portable GPU tests; fix path, cache and
   file-locking issues. Document.
2. **WSL2**: `_libcuda_path` already searches `/usr/lib/wsl/lib`; run the kernel test suite under WSL2 and document
   it as the supported Windows route until native kernels exist.
3. **Cubin libraries native**: load `nvcuda.dll` (from `System32`) instead of `libcuda.so.1`; drop the Linux-only
   check in `kernels.py` for cubin libraries; a Windows driver-version table (Windows drivers are numbered
   differently, e.g. 572.xx); replace the `memfd_create` path with a temp file for in-memory images; skip the POSIX
   group check.
4. **Shared libraries native**: build `.dll` files with MSVC + nvcc in CI on a Windows runner; extend `build.toml`
   / `manifest.toml` with an OS key (`files` per `os/target`); export a plain C ABI (`extern "C"`,
   `__declspec(dllexport)`).
5. **Validation**: on the Windows install of the Ubuntu machine (RTX 5060 Ti); the same golden tests and a device
   survey row (`docs/devices/`).

Setting up the Windows machine (once): current NVIDIA driver; `uv`; the CUDA build of torch; for kernel builds
Visual Studio 2022 Build Tools (C++ workload) and the CUDA toolkit; the OpenSSH server (Settings → Optional
features) so it can be driven from the Mac like the Ubuntu box; optionally WSL2.

Windows pitfalls to handle or document:

- **GPU timeout (TDR)**: on the GPU that drives the display, Windows resets the driver when one kernel runs longer
  than ~2 s. Test with the display on another GPU or raise `TdrDelay`; document for users with large stereo inputs.
- **Silent memory spill**: the "CUDA Sysmem Fallback Policy" moves memory to system RAM instead of failing with OOM;
  disable it for benchmarks, and note it in the troubleshooting guide (slow instead of OOM).
- **Hugging Face cache symlinks** need Developer Mode; otherwise files are copied (works, uses more disk).
- **Defender** slows installs and imports: exclude the project and cache directories.
- **Long paths**: enable long-path support if deep cache paths fail.
- **No Triton by default**: `torch.compile` tests are skipped on Windows.

## Jetson and ARM64 Linux

After the Windows `.dll` builds, because it reuses the per-OS / per-CPU build matrix they introduce. Why: stereo,
depth, detection, pose and SAM are what robots and drones run, and Jetson is where they run them. The competitor is
TensorRT, which is well optimised there; our angle is the same as in ComfyUI: no engine build per device and
resolution, any input size, and models that export badly to TensorRT (FoundationStereo-class stereo). Whether we
match TensorRT's speed on a Jetson is not measured.

Steps:

1. **Cubins**: current Jetsons (Orin Nano / NX / AGX) report `sm_87`; the loader falls back to the nearest older
   build of the same major version (`sm_86`, `sm_80`), so the cubin libraries should load as they are. Check on a
   device, then decide whether `sm_87` gets its own build. Newer Jetson generations get their own row when a device
   is available.
2. **Shared libraries**: ARM64 builds of the 12 `.so` libraries (CUTLASS and cuDNN ones), cross-compiled in CI;
   `build.toml` / `manifest.toml` get a CPU key next to the OS key from the Windows work.
3. **Driver check**: a Jetson table, since JetPack / L4T number their drivers differently (as Windows does).
4. **PyTorch**: NVIDIA's Jetson wheels, not PyPI; document the install and test against the current JetPack.
5. **Tuning**: a Jetson has a small GPU that shares memory with the CPU, so choices tuned on desktop cards (fp16
   paths, GEMM variants, batch sizes) may not be the best there. A working port first, a tuned one later.
6. **Validation**: golden tests and a device survey row (`docs/devices/`) on a real device (an Orin Nano developer
   kit is the cheapest), plus one benchmark against TensorRT for a model both can run.

**ARM64 Linux servers with NVIDIA GPUs** (for example AWS Graviton GPU instances) come almost for free from step 2:
the same ARM64 shared libraries with the ordinary driver numbering. Validated on a rented instance.

Not planned: AMD and Intel GPUs. The kernels are CUDA, so that would be a second kernel set, as Metal was, for a
much smaller ML audience.

## ComfyUI node pack

A separate repository (`ComfyUI-fastforks`), GPL-3.0 like ComfyUI, depending on `fastforks-core` so fastforks stays
permissive. Published to the ComfyUI Registry and installable from ComfyUI-Manager.

Why: ComfyUI has a large user base and is where people discover tools. The win is reach more than a speed moat: the
diffusion model dominates most workflows. Speed matters where models run on many frames (video workflows guided by
depth / pose / masks, batch background removal, SAM 2 video). Fast alternatives today are TensorRT node packs
(almost all from one author, yuvraj108c: Depth Anything, DWPose, RIFE, upscalers), which build an engine per GPU and
resolution; fastforks is fast at any size with no build step. SAM, background removal and the controlnet_aux
preprocessors have no fast option at all (landscape in [roadmap.md](roadmap.md#platforms-and-integrations)).

Nodes (first release):

| node | input → output | fastforks call |
|---|---|---|
| Load fastforks model | model id, mode, dtype → `FASTFORKS_MODEL` | `task.load` |
| Depth | `IMAGE` (batch = video frames) → `IMAGE` (depth map, normalised) + raw | `depth`, video depth |
| Normals | `IMAGE` → `IMAGE` | `depth` normals |
| Pose | `IMAGE` → `IMAGE` (skeleton drawing, OpenPose colours) + keypoints JSON | `pose` + `detection` |
| Segment (SAM 2 / SAM 3) | `IMAGE`, points / boxes / text → `MASK` (per frame for video) | `promptable_segmentation` |
| Remove background | `IMAGE` → `IMAGE` + `MASK` | `matting` |
| Stereo / Flow | `IMAGE` pairs → `IMAGE` visualisations + raw | `stereo`, `flow` |

Details:

- ComfyUI's `IMAGE` is `[B, H, W, C]` float in [0, 1], `MASK` is `[B, H, W]`; conversion is a permute.
- Integrate with `comfy.model_management` (load device, offload to CPU when VRAM is needed, free on request).
- Pose drawing must match the OpenPose colour convention that ControlNet models expect.
- Requires Windows kernels to be compelling (most ComfyUI users run the Windows portable build); until then the pack
  runs portable models and says so.
- Later nodes once the tasks exist: upscale (`restoration`), frame interpolation (`interpolation`), point tracks.

## Relation to the splat library

fastforks provides, as optional imports of the splat library: camera poses without COLMAP (`multiview`: VGGT,
MapAnything; `matching` + bundle adjustment), depth and normal priors (`depth`), masks (`promptable_segmentation`),
features for feature fields (`features`), and feed-forward Gaussians (`multiview`). The kernel runtime (loader,
manifests, target selection, Hugging Face publishing, Windows support) is extracted into a small shared package used
by both, so improvements land in both at once. The extraction is the first step of the splat library (its M0), not
a fastforks milestone. The first splat milestones need only what fastforks already ships (stereo and multi-view
depth, COLMAP poses from outside); `matching` and the other new tasks matter there later, for poses without COLMAP.

## Milestones

| milestone | contents |
|---|---|
| F1 | finish `pose` (ViTPose, RTMPose); `flow` (RAFT, SEA-RAFT, WAFT, MEMFOF); CLIP / SigLIP 2 / PE / C-RADIO features |
| F2 | `matching` (ALIKED / DISK + LightGlue, XFeat, RoMa); `tracking` (TAPNext, BootsTAPIR, AllTracker); RTMO and whole-body pose (RTMW / DWPose); WSL2 documented; native Windows for cubin libraries |
| F3 | `matting`; detection additions (RF-DETR, DEIM, Grounding DINO, OWLv2); SAM 2 video; video depth; normals; Windows `.dll` builds; ComfyUI node pack v1 |
| F4 | frozen-backbone heads, resized heads, `fastforks.ops`; fine-tuning recipes; feed-forward splatting in `multiview`; Jetson and ARM64 Linux builds |
| F5 | optional tasks on demand (restoration, interpolation, OCR, face); broader `patch()`; backward kernels for the most-used blocks |
