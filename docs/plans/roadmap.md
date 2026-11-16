# Roadmap: which networks and methods go into which library

Status (2026-11-09): plan only. `fastforks.pose` (ViTPose, RTMPose) and the CLIP / SigLIP 2 / Perception Encoder /
C-RADIO feature backbones are in progress; nothing else below is built. The design behind each part is in
[fastforks-next.md](fastforks-next.md) (new tasks, Windows, ComfyUI, customisation, training) and
[splat-library.md](splat-library.md) (the Gaussian splatting / radiance field library). A wider survey of open models
beyond this plan, with licences and priorities (incl. medical and remote sensing), is in
[model-landscape.md](model-landscape.md).

**Order of work.** The milestones (F1–F5, M0–M5) are an order, not a schedule; there are no dates. fastforks is
developed first, until we judge it good enough for now. Then development moves to the splat library, which starts
from our existing research project (rasteriser about 3x faster than gsplat's for standard splatting). fastforks
items still open at that point stay in this list and are picked up when the splat library or a user needs them.
M0–M2 of the splat library need nothing from F2–F5: they use COLMAP poses and the stereo / multi-view models
fastforks already ships. The one shared task is extracting the kernel runtime, done in M0.

Licences in this document were checked on 2026-11-09 against each repository's LICENSE file and the Hugging Face model
card (`gh api`, card `license` field). "unverified" means the item was looked for and not confirmed; check it before
the model ships. None of this is legal advice.

## How to read the tables

- **Code / weights**: the licence of the reference code and of the published checkpoint. They differ often, and the
  weights decide whether a fastforks model card can say `commercial use: yes`.
- **Commercial**: `yes` (both permissive), `no` (non-commercial, copyleft we will not ship, or a custom restriction),
  `check` (something unverified or an extra condition such as training-data terms or a gated backbone).
- **Kernels**: which existing fastforks kernel families the network reuses. `new` means a new kernel family is
  needed, which costs more than adding a model.
- **Priority**: `P1` build in that milestone, `P2` after the P1 models of the same task, `P3` only on demand.
- A model with `commercial: no` can still be added (fastforks already ships non-commercial models with the flag), but
  it is never the default or the README example of a task, and it is `P2` at best.

## Where each topic goes

| topic | where | why |
|---|---|---|
| optical flow | fastforks, new task `flow` | fixed network; reuses stereo correlation / GRU kernels |
| point tracking | fastforks, new task `tracking` | fixed network over a clip; correlation + transformer blocks |
| keypoints and matching (sparse and dense) | fastforks, new task `matching` | fixed networks; feeds `multiview` and the splat library |
| 2D human pose (top-down, one-stage, whole-body) | fastforks `pose` (in progress) | fixed network |
| 3D human mesh, hands (SMPL / MANO) | not planned | the body models forbid commercial use |
| 6D object pose | not planned | render-and-refine loops; strongest models non-commercial |
| matting / background removal | fastforks, new task `matting` | fixed network; large ComfyUI demand |
| detection: real-time DETRs, open-vocabulary | fastforks `detection` (additions) | DETR decoder kernels exist |
| SAM 2 video | fastforks `promptable_segmentation` (addition) | same weights as SAM 2; memory attention as kernels |
| video depth | fastforks `depth` (addition) | DINOv2 + DPT kernels exist |
| surface normals | fastforks `depth` (output `normals`) | same models or same backbones as depth |
| feed-forward Gaussian splatting | fastforks `multiview` (output `gaussians`) | a forward pass; rendering lives in the splat library |
| restoration / super-resolution, frame interpolation, OCR, face detection, video classification | fastforks, optional tasks (P3) | fit the contract, serve other audiences; ComfyUI is the main reason |
| Gaussian splatting, NeRF, per-scene optimisation | new library (working name `fastsplat`) | training loop per scene: a different product |
| feed-forward splat *rendering*, meshes from splats, compression, viewer | `fastsplat` | |
| SfM, RANSAC, bundle adjustment, target calibration | not built; called through `pycolmap` / PoseLib / OpenCV from `fastsplat` L2 (see [Geometry](#geometry-sfm-ransac-bundle-adjustment-calibration)) | COLMAP 4.x has GLOMAP, GPU BA and learned matchers; algorithms, not fixed networks |
| image retrieval / pair selection | fastforks, small new task `retrieval` (P2) | global descriptors from existing `features` kernels; removes O(n²) matching |
| medical 2D (MedSAM, Cellpose-SAM, pathology / X-ray encoders) | fastforks, model cards in existing tasks (P2) | SAM / ViT kernels exist; research-use-only cards; see [model-landscape.md](model-landscape.md#medical-imaging-decision) |
| medical 3D (CT / MR: TotalSegmentator, VISTA3D, nnU-Net) | maybe later, a `volume` task (P3) | 3D kernels, sliding window, NIfTI input: a new input type |
| remote sensing foundation models | fastforks `features`, on demand (P3) | ViT-MAE kernels exist; multispectral input is the work |
| camera calibration, line segments | fastforks, small new task (P2) | single-image fixed networks; feed `matching` / `multiview` |
| point cloud networks | not planned | different data type and kernel set |
| image / video generation, VLMs, face recognition | out of scope | see [fastforks-next.md](fastforks-next.md#what-belongs-in-fastforks) |

## fastforks

Milestones F1–F5 are defined in [fastforks-next.md](fastforks-next.md#milestones). Model ids below are proposals
following the existing naming (`family-variant`).

### Flow (`fastforks.flow`, F1)

| model id | family / paper | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `raft-things`, `raft-sintel`, `raft-kitti` | RAFT | princeton-vl/RAFT | BSD-3 | unverified | check | `raft` correlation + GRU (stereo) | P1 |
| `sea-raft-s`, `sea-raft-m`, `sea-raft-l` | SEA-RAFT | princeton-vl/SEA-RAFT | BSD-3 | unverified | check | as RAFT, ResNet encoder | P1 |
| `waft-*` | WAFT | princeton-vl/WAFT | BSD-3 | unverified | check | ViT (DINO) encoder + warping; `waft_stereo` exists | P1 |
| `memfof-*` | MEMFOF (multi-frame) | msu-video-group/memfof | BSD-3 | BSD-3 | yes | RAFT family | P1 |
| `neuflow-v2` | NeuFlow v2 (edge, fast) | neufieldrobotics/NeuFlow_v2 | Apache-2.0 | unverified | check | CNN + global matching | P2 |
| `unimatch-flow-*` | UniMatch / GMFlow | autonomousvision/unimatch | MIT | unverified | check | transformer + global correlation | P2 |
| `memflow-*` | MemFlow | DQiaole/MemFlow | Apache-2.0 | unverified | check | RAFT family + memory | P2 |
| `flowseek-*` | FlowSeek | — | Apache-2.0 | unverified | check | depth-foundation features + RAFT | P2 |
| `dpflow-*` | DPFlow | — | Apache-2.0 | non-commercial | no | CNN pyramid | P3 |
| — | FlowFormer++, VideoFlow | — | no licence file | — | no | — | skip until licensed |
| — | FreeFlow (2026) | — | CC-BY-NC-SA | — | no | — | skip |

The default and README example is SEA-RAFT (`sea-raft-m`) once its weights licence is confirmed; MEMFOF is the
fallback default because both its code and weights are verified BSD-3.

### Point tracking (`fastforks.tracking`, F2)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `tapnext-*` | TAPNext | google-deepmind/tapnet | Apache-2.0 | Apache-2.0 | yes | ViT + recurrent (SSM) blocks: `new` for the SSM part | P1 |
| `bootstapir`, `tapir` | TAPIR / BootsTAPIR | google-deepmind/tapnet | Apache-2.0 | Apache-2.0 | yes | ResNet, correlation, conv refinement | P1 |
| `alltracker` | AllTracker (dense) | aharley/alltracker | MIT | unverified | check | RAFT-like low-res flow + transformer | P1 |
| `locotrack-*` | LocoTrack | cvlab-kaist/locotrack | Apache-2.0 | unverified | check | 4D correlation, transformer | P2 |
| `track-on` | Track-On | gorkaydemir/track_on | MIT | unverified | check | DINOv2 + transformer memory | P2 |
| `track-on2` | Track-On2 | same | MIT | needs DINOv3 (gated, custom licence) | check | DINOv3 + transformer | P3 |
| `cotracker3-offline`, `cotracker3-online` | CoTracker3 | facebookresearch/co-tracker | CC-BY-NC 4.0 | CC-BY-NC 4.0 | no | correlation + transformer | P2 (marked `[nc]`) |

The README example uses `tapnext` or `bootstapir`; CoTracker3 is widely used and worth having, but only as a
non-commercial option.

### Matching (`fastforks.matching`, F2)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `aliked-lightglue` | ALIKED + LightGlue | Shiaoming/ALIKED, cvg/LightGlue | BSD-3, Apache-2.0 | ALIKED unverified, LightGlue Apache-2.0 | check | CNN (deformable conv), attention | P1 |
| `disk-lightglue` | DISK + LightGlue | cvlab-epfl/disk, cvg/LightGlue | Apache-2.0 | Apache-2.0 | yes | U-Net CNN, attention | P1 |
| `xfeat`, `xfeat-lighterglue` | XFeat | verlab/accelerated_features | Apache-2.0 | unverified | check | small CNN | P1 |
| `roma-outdoor`, `roma-indoor` | RoMa (dense) | Parskatt/RoMa | MIT | unverified | check | DINOv2 (kernels exist) + CNN + GP matcher | P1 |
| `roma-v2-*` | RoMa v2 | Parskatt/RoMa | MIT | DINOv3 backbone (custom licence) | check | DINOv3 (kernels exist) | P2 |
| `loftr-*` | LoFTR | zju3dv/LoFTR | Apache-2.0 | unverified | check | ResNet + linear attention | P2 |
| `efficient-loftr` | EfficientLoFTR | zju3dv/EfficientLoFTR | registration-gated, unclear | unclear | check | as LoFTR | P3 |
| `superpoint-lightglue` | SuperPoint + LightGlue | magicleap/SuperPointPretrainedNetwork | non-commercial | non-commercial | no | small CNN | P2 (marked `[nc]`) |
| `mast3r-*`, `dust3r-*` | MASt3R / DUSt3R | naver/mast3r, naver/dust3r | CC BY-NC-SA | CC BY-NC-SA | no | ViT (kernels exist) | P3 (`[nc]`; better as `multiview` models) |
| `ufm-*` | UFM | — | BSD-3 | non-commercial | no | ViT | P3 |

SuperPoint is the usual LightGlue partner but non-commercial: the defaults are ALIKED and DISK.

Compared with Kornia (`kornia.feature`) and COLMAP 4.x (ONNX): ALIKED, DISK, LightGlue, XFeat and LoFTR are the same
networks with the same upstream weights, so accuracy is identical and we win only on speed (to be measured against
Kornia eager PyTorch and COLMAP's ONNX Runtime path). Better than both: RoMa / RoMa v2, the dense matchers, which
neither ships (Kornia has DeDoDe from the same author). Kornia has DeDoDe, SOLD2, KeyNet / HardNet and AdaLAM that we
do not plan. Newer sparse candidates to check before F2 freezes: DeDoDe v2, DaD keypoints, LoMa (in COLMAP 4.2);
licences unverified.

Why the rest of Kornia's and COLMAP's matchers are or are not planned:
- **DeDoDe v2**: candidate (P2). Its descriptor runs on DINOv2, so it reuses our kernels; a LightGlue variant exists.
  Licence to verify.
- **LoMa** (COLMAP 4.2): candidate. Matching COLMAP's own learned matchers makes the benchmark like for like.
  Licence and architecture to verify.
- **SIFT** (COLMAP's default): not a network, and COLMAP already runs it on the GPU.
- **SOLD2** (lines): not planned. Lines are covered by DeepLSD and ScaleLSD (detection) and GlueStick (matches
  points and lines together), the newer models from the same field. SOLD2 is a P3 fallback at most.
- **SIFT on the GPU**: not planned. Fast GPU versions already exist (COLMAP's SiftGPU, AliceVision's PopSift). Its
  irregular steps (histograms per keypoint) don't reuse our network kernels, and extraction is rarely the slow part
  of a COLMAP run (matching and bundle adjustment are).
- **KeyNet / HardNet**: skipped. Small 2017-2019 CNNs, outperformed by ALIKED / DISK, and HardNet works on
  per-keypoint patches, so the patch cropping costs more than the network. Little to speed up.
- **AdaLAM**: skipped. A hand-written outlier filter, not a network; it belongs with RANSAC in geometry, which we
  call rather than build.

### Pose (`fastforks.pose`, F1 → F3)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `vitpose-*` (in progress) | ViTPose (top-down) | HF transformers port | Apache-2.0 | Apache-2.0 | yes | ViT | P1 (F1) |
| `rtmpose-*` (in progress) | RTMPose (top-down) | open-mmlab/mmpose, rtmlib | Apache-2.0 | Apache-2.0 | check (body7 training data) | CSPNeXt CNN + SimCC | P1 (F1) |
| `rtmo-*` | RTMO (one-stage, no boxes) | open-mmlab/mmpose | Apache-2.0 | Apache-2.0 | check (body7) | CSPNeXt + YOLO-style head | P1 (F2) |
| `rtmw-*`, `dwpose-*` | RTMW / DWPose (whole-body 133) | mmpose, IDEA-Research/DWPose | Apache-2.0 | Apache-2.0 | check (training data) | as RTMPose | P1 (F2; ComfyUI needs DWPose) |
| `vitpose-plus-*` | ViTPose+ (multi-dataset, animal AP-10K) | HF port | Apache-2.0 | Apache-2.0 | check (AP-10K terms unverified) | ViT + MoE heads | P2 |
| `sapiens-pose-*` | Sapiens | facebookresearch/sapiens | CC-BY-NC 4.0 | CC-BY-NC 4.0 | no | ViT (sapiens segmentation exists) | P3 (`[nc]`) |
| — | Ultralytics YOLO-pose | ultralytics/ultralytics | AGPL-3.0 | AGPL-3.0 | no | — | never (copyleft) |
| — | HMR2, HaMeR, WiLoR, Multi-HMR | — | MIT / NC | need SMPL / MANO | no | — | not planned |
| — | FoundationPose, MegaPose, GigaPose (6D) | — | NVIDIA NC / Apache / MIT | — | mixed | — | not planned |

### Matting (`fastforks.matting`, F3)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `birefnet-general`, `birefnet-lite`, `birefnet-hr`, `birefnet-portrait` | BiRefNet | ZhengPeng7/BiRefNet | MIT | MIT | yes | Swin-L / PVTv2 (`new` Swin family unless shared with mask2former / oneformer) + conv decoder | P1 |
| `ben2` | BEN2 | PramaLLC/BEN2 | MIT | MIT | yes | as BiRefNet | P1 |
| `vitmatte-s`, `vitmatte-b` | ViTMatte (trimap) | hustvl/ViTMatte | MIT | Apache-2.0 | yes | ViT + small conv head | P1 |
| `rmbg-2` | BRIA RMBG-2.0 | HF briaai/RMBG-2.0 | — | custom `bria-rmbg-2.0`, gated | check (treat as no) | BiRefNet architecture | P3 after reading the licence |
| `matanyone` | MatAnyone (video) | pq-yang/MatAnyone | S-Lab 1.0 (NC) | S-Lab 1.0 | no | CNN + memory | P3 (`[nc]`) |

### Detection additions (F3)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `rf-detr-n`, `-s`, `-m`, `-l`, `rf-detr-seg-*` | RF-DETR | roboflow/rf-detr | Apache-2.0 | Apache-2.0 (N/S/M/L, Seg) | yes | DINOv2 ViT + DETR decoder (both exist) | P1 |
| — | RF-DETR Atto / Femto / Pico / XL / 2XL | roboflow/rf-detr_plus | — | Roboflow PML 1.0 | no | — | skip |
| `lw-detr-*` | LW-DETR | Atten4Vis/LW-DETR | Apache-2.0 | Apache-2.0 | yes | ViT + DETR decoder | P2 |
| `deim-dfine-*` | DEIM (on D-FINE) | Intellindust-AI-Lab/DEIM | Apache-2.0 | unverified | check | HGNetv2 + D-FINE decoder (`dfine` exists) | P1 |
| `deimv2-*` | DEIMv2 | Intellindust-AI-Lab/DEIMv2 | custom, non-commercial | same | no | DINOv3 + DETR decoder | P3 (`[nc]`) |
| `co-detr-*` | Co-DETR | Sense-X/Co-DETR | MIT | unverified | check | Swin-L / ViT-L + DETR | P3 (very large) |
| `grounding-dino-t`, `-b` | Grounding DINO (open-vocabulary) | IDEA-Research/GroundingDINO, HF port | Apache-2.0 | Apache-2.0 | yes | Swin + BERT text encoder (`new`) + DETR decoder | P1 |
| `mm-grounding-dino-*` | MM-Grounding-DINO | open-mmlab/mmdetection | Apache-2.0 | Apache-2.0 (tiny checked) | check | as Grounding DINO | P2 |
| `owlv2-b16`, `owlv2-l14` | OWLv2 (open-vocabulary) | google-research/scenic, HF port | Apache-2.0 | Apache-2.0 | yes | CLIP ViT (shared with the `clip` features) | P1 |
| `yolox-*` | YOLOX | Megvii-BaseDetection/YOLOX | Apache-2.0 | unverified | check | CSP-Darknet CNN | P3 |
| — | Ultralytics YOLOv8 / 11 / 26, YOLOE, YOLO-World | — | AGPL-3.0 / GPL-3.0 | same | no | — | never (copyleft) |
| — | Grounding DINO 1.5 / 1.6 | — | API only | — | — | — | not available |

Open-vocabulary API: `model(image, text=["cup", "laptop"])`; the text encoder runs once per prompt list and is cached.
SAM 3 (already in `promptable_segmentation`) covers text-prompted masks; Grounding DINO / OWLv2 add boxes.

### Video modes of existing tasks (F3)

| addition | model | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| `promptable_segmentation` video: `model.track(video, prompts=...)` | SAM 2.1 video predictor | facebookresearch/sam2 | Apache-2.0 | Apache-2.0 | yes | Hiera (exists) + memory attention (`new`) | P1 |
| same, motion-aware memory | SAMURAI | yangchris11/samurai | Apache-2.0 | SAM 2 weights | yes | as SAM 2 + Kalman filter (Python) | P2 |
| SAM 3 video tracking | SAM 3 | facebookresearch/sam3 | per the existing `sam3` card | same | per card | as SAM 2 | P2 |
| `depth` video: `model(frames)` | Video Depth Anything Small | DepthAnything/Video-Depth-Anything | Apache-2.0 | Apache-2.0 | yes | DINOv2 + DPT (exist) + temporal attention | P1 |
| same | Video Depth Anything Base / Large (relative, metric) | same | Apache-2.0 | CC-BY-NC 4.0 | no | same | P2 (`[nc]`) |
| — | DepthCrafter | Tencent/DepthCrafter | Tencent research-only | `other` | no | video diffusion | skip |

### Normals (`depth` output `result.normals`, F3)

| model id | family | repo | code | weights | commercial | kernels | priority |
|---|---|---|---|---|---|---|---|
| existing `moge-*`, `metric3d-*` | models that already predict normals | existing cards | per card | per card | per card | exist | P1 (expose the output) |
| `lotus-normal-*`, `lotus-2-normal-*` | Lotus / Lotus-2 (one-step diffusion) | EnVision-Research/Lotus, Lotus-2 | Apache-2.0 | Apache-2.0 (`lotus-normal-g-v1-1` checked) | yes | SD UNet (`marigold` exists) | P1 |
| `stablenormal-turbo` | StableNormal (YOSO) | Stable-X/StableNormal | Apache-2.0 | Apache-2.0 (`yoso-normal-v1-8-1`) | yes | SD UNet | P2 |
| `marigold-normals-lcm` | Marigold normals LCM | prs-eth/Marigold | Apache-2.0 | Apache-2.0 | yes | `marigold` exists | P1 |
| `marigold-normals-v1-1` | Marigold normals v1.1 | same | Apache-2.0 | OpenRAIL++ | check | same | P2 |
| `dsine` | DSINE | baegwangbin/DSINE | custom Imperial licence | no statement | check (treat as no) | EfficientNet + ConvGRU | P3 |

### Feed-forward Gaussian splatting (`multiview` output `result.gaussians`, F4)

fastforks returns the predicted Gaussians as tensors and writes PLY; rendering them is the splat library's job. This
also removes the licence problem these repos share: most bundle a fork of Inria's rasteriser, which fastforks does
not need.

| model id | repo | code | weights | depends on | commercial | priority |
|---|---|---|---|---|---|---|
| `anysplat` | InternRobotics/AnySplat | MIT | unverified | VGGT (licence per the existing `vggt` card), gsplat | check | P1 |
| `depthsplat-*` | cvg/depthsplat | MIT | unverified | UniMatch, Depth Anything V2 (Small Apache, larger NC) | check | P1 |
| `mvsplat` | donydchen/mvsplat | MIT | unverified | UniMatch; Inria rasteriser fork (not needed here) | check | P2 |
| `noposplat` | cvg/NoPoSplat | MIT | unverified | DUSt3R / CroCo backbone (CC BY-NC-SA) | no | P3 (`[nc]`) |
| `pixelsplat` | dcharatan/pixelsplat | MIT | unverified | Inria rasteriser fork (not needed here) | check | P3 (superseded) |

### Features (F1, in progress)

`c_radio`, `clip`, `siglip2`, `perception_encoder` join `dinov2`, `dinov3`, `dinov3_convnext`. They also become the
text / image encoders for open-vocabulary detection (OWLv2 uses CLIP), feature fields in the splat library, and
frozen-backbone heads (customisation 1 in [fastforks-next.md](fastforks-next.md#customisation-without-losing-speed)).

### Optional tasks (F5, only on demand)

| task | model | repo | code | weights | commercial | note |
|---|---|---|---|---|---|---|
| `restoration` | Real-ESRGAN (x2, x4, anime) | xinntao/Real-ESRGAN | BSD-3 | unverified | check | the ComfyUI standard; community fine-tunes have their own licences |
| | SwinIR, HAT, DAT, Swin2SR | JingyunLiang/SwinIR, XPixelGroup/HAT, zhengchen1999/DAT, mv-lab/swin2sr | Apache-2.0 | unverified | check | Swin kernels shared with matting / segmentation |
| | DRCT, SPAN | ming053l/DRCT (MIT), hongyuanyu/SPAN (Apache-2.0) | MIT / Apache-2.0 | unverified | check | SPAN is light and fast |
| | NAFNet, Restormer (denoise, deblur) | megvii-research/NAFNet, swz30/Restormer | MIT | unverified | check | |
| | SeedVR2 3B / 7B (video SR) | ByteDance-Seed/SeedVR | Apache-2.0 | Apache-2.0 | yes | diffusion transformer, heavy; most popular ComfyUI video upscaler |
| | — SUPIR, ResShift | — | non-commercial | — | no | skip |
| `interpolation` | RIFE 4.25 / 4.26 | hzwer/Practical-RIFE | MIT | MIT (per README) | yes | flow kernels reuse; ComfyUI standard |
| | FILM | google-research/frame-interpolation | Apache-2.0 | unverified | check | archived repo |
| | GMFSS Fortuna (anime) | 98mxr/GMFSS_Fortuna | MIT | unverified | check | |
| `ocr` | PP-OCRv5 detection + recognition | PaddlePaddle/PaddleOCR | Apache-2.0 | Apache-2.0 | yes | Paddle weights need conversion to PyTorch |
| | PARSeq (recognition) | baudm/parseq | Apache-2.0 | unverified | check | |
| `face` | YuNet (detection + 5 landmarks) | opencv/opencv_zoo | MIT | MIT | yes | tiny; the default |
| | RetinaFace | serengil/retinaface (wrapper) | MIT | unverified | check | check which weights it ships |
| | — SCRFD / insightface | deepinsight/insightface | MIT | non-commercial | no | skip; no face recognition at all |
| `video_classification` | VideoMAE v2 | OpenGVLab/VideoMAEv2 | MIT | CC-BY-NC 4.0 | no | only `[nc]`; check `OpenGVLab/VideoMAE2` (card says Apache-2.0) first |

### Platforms and integrations

- **Windows** (F2 cubin libraries native, F3 `.dll` builds): plan in
  [fastforks-next.md](fastforks-next.md#windows); validated on the Windows install of the Ubuntu machine.
- **NVIDIA Jetson and ARM64 Linux** (F4, after the Windows `.dll` builds): plan in
  [fastforks-next.md](fastforks-next.md#jetson-and-arm64-linux). The audience is robotics (stereo, depth, detection,
  pose, SAM); the competitor there is TensorRT. Needs a device to test on; no support claim without one.
- **Not planned**: AMD and Intel GPUs (a second kernel set for a small ML audience); browser and mobile except for the
  splat viewer.
- **ComfyUI node pack** (F3, separate GPL-3.0 repo): plan in [fastforks-next.md](fastforks-next.md#comfyui-node-pack).
  First release nodes: depth (incl. video), normals, pose (DWPose / RTMW drawing), SAM 2 / SAM 3 (incl. video),
  background removal (BiRefNet, BEN2), stereo and flow. Later: upscale, interpolation, point tracks.

The current ComfyUI landscape for these tasks (checked 2026-11-09):

| need | most-used packs | backend | fast alternatives today |
|---|---|---|---|
| preprocessors (depth, normals, pose, flow, lineart) | Fannovel16/comfyui_controlnet_aux (4.3k stars, Apache-2.0) | PyTorch (some ONNX, e.g. DWPose) | yuvraj108c TensorRT packs for Depth Anything and DWPose |
| depth | kijai/ComfyUI-DepthAnythingV2, PozzettiAndrea/ComfyUI-DepthAnythingV3, yuvraj108c/ComfyUI-Video-Depth-Anything | PyTorch | yuvraj108c/ComfyUI-Depth-Anything-Tensorrt |
| SAM | kijai/ComfyUI-segment-anything-2 (1.2k), PozzettiAndrea/ComfyUI-SAM3, storyicon/comfyui_segment_anything | PyTorch | none found |
| background removal | 1038lab/ComfyUI-RMBG (2.2k, GPL-3.0; RMBG-2.0, BiRefNet, SAM) | PyTorch (+ onnxruntime) | none found |
| upscale | numz/ComfyUI-SeedVR2_VideoUpscaler (3.0k), ssitu/ComfyUI_UltimateSDUpscale | PyTorch | yuvraj108c/ComfyUI-Upscaler-Tensorrt |
| frame interpolation | Fannovel16/ComfyUI-Frame-Interpolation (1.1k, MIT; RIFE, FILM, GMFSS, …) | PyTorch | yuvraj108c/ComfyUI-Rife-Tensorrt |
| pose for video models | kijai/ComfyUI-WanAnimatePreprocess (ViTPose, YOLO) | ONNX | yuvraj108c/ComfyUI-Dwpose-Tensorrt |

Conclusion: the fast packs are TensorRT builds per GPU and resolution from one author; SAM, background removal and
the preprocessor suite have no fast option at all. Those are the first nodes to ship.

### Where usage is visible and speed saves money (checked 2026-11-09)

Usage signals worth re-reading every quarter. All include CI and mirror traffic, so read them as ranks rather than user
counts.

- **PyPI** (pypistats.org, last month): timm 10.4M, ultralytics 4.2M, rembg 2.5M, kornia 2.2M, open_clip_torch 1.8M,
  segment-anything 377k, basicsr 241k, transparent-background 134k, sam2 56k.
- **Hugging Face** (`/api/models/<id>`, last month):
  - CLIP: clip-vit-base-patch32 17.7M, clip-vit-large-patch14 6.3M.
  - DINOv2: base 3.4M, small 2.6M.
  - Depth Anything V2 Small: 1.5M.
  - SigLIP2 base: 1.4M.
  - BiRefNet: 1.0M.
  - VGGT-1B: 570k.
  - RMBG-2.0: 408k.
  - DINOv3 ViT-B: 406k.
  - ViTPose base: 298k.
  - RT-DETR R50: 234k.
  - SAM 2.1 large: 220k.
- **ComfyUI registry** (`api.comfy.org/nodes/<id>`, all-time):
  - Impact Pack (SAM + detectors): 3.6M.
  - controlnet_aux: 2.8M.
  - Frame-Interpolation: 2.0M.
  - UltimateSDUpscale: 1.9M.
  - RMBG: 964k.
  - SAM 2: 475k.
  - Depth Anything V2: 360k.
- **Replicate** shows run counts and per-second GPU prices (T4 $0.000225/s, L40S $0.000975/s). zsxkib/rife has 2M+ runs
  and Depth Anything V2 ~150k. Scraped counts are noisy. fal.ai publishes no per-model numbers.

Integration targets, ranked by how directly a faster kernel turns into saved time or money:

1. **ComfyUI nodes.** Pure PyTorch on user GPUs, so we replace no existing backend. Covered by the node pack above.
2. **Immich ML** (118k stars): CLIP / SigLIP embeddings over every photo, plus InsightFace. It runs on ONNX Runtime
   (CUDA, ROCm, OpenVINO), so it only pays if we beat ORT-CUDA. Measure that first, then contribute a backend.
3. **Video-generation data curation** (NeMo Curator, Open-Sora scoring, OpenVid, Panda-70M, Koala-36M): per clip they
   run shot detection (TransNetV2), CLIP aesthetic scoring, optical flow (UniMatch / RAFT class), OCR and CLIP / DINO /
   SigLIP embeddings. This is the largest GPU spend, but it is in-house and unpublished. Flow and embeddings are ours
   already; TransNetV2, the aesthetic head and an OCR detector would be small additions (see model-landscape.md).
4. **Hosted per-call inference**: Roboflow inference (SAM, CLIP, depth, RF-DETR) and Replicate / fal.ai model owners
   (RIFE, Real-ESRGAN, BiRefNet, Depth Anything). Billing is per GPU second, so a speedup is margin.
5. **Annotation tools**: CVAT (Nuclio serverless), Label Studio ML backends, X-AnyLabeling (ONNX), autodistill. They
   use SAM / SAM 2 and Grounding DINO, and click latency is what users notice.
6. **Background removal servers**: rembg and transparent-background (2.5M + 134k a month). BiRefNet through ORT or
   PyTorch, high volume.
7. **Medical 2D**: Cellpose-SAM, SAM in 3D Slicer, OHIF and napari, and pathology ViT encoders. See
   model-landscape.md, section "Kaapana, RACOON and where medical speed matters"; Kaapana and RACOON themselves are 3D
   nnU-Net shops.
8. **Research 3D**: Kornia, hloc, nerfstudio and MASt3R need LightGlue, RoMa, depth and VGGT. The users are
   researchers, and splat preprocessing feeds fastsplat.
9. **Out of reach for now**:
   - ultralytics owns its own export stack.
   - Frigate already ships TensorRT and OpenVINO.
   - Video2X and Flowframes run on ncnn / Vulkan, which is not our audience until Windows .dll builds and maybe Vulkan.
   - Isaac ROS is already TensorRT-optimised.

Action before any outreach: one reproducible benchmark per target against what it ships today. That means ORT-CUDA for
Immich and rembg, PyTorch for ComfyUI and Cellpose, and the TensorRT PR numbers for Cellpose-SAM.

### Geometry: SfM, RANSAC, bundle adjustment, calibration

Checked 2026-11-09. **Decision: no own SfM, solver or calibration library. Networks go into fastforks; geometry is
called from existing libraries in fastsplat L2.**

The state of the field:
- **COLMAP 4.x** (BSD-3, `pycolmap` CUDA wheels on PyPI) has largely closed the classical speed gap:
  - GLOMAP is merged as the global mapper (4.0, 2026-03); the `colmap/glomap` repo is archived.
  - Caspar GPU bundle adjustment (4.1). The README claims 1–2 orders of magnitude over Ceres CUDA; not
    independently measured.
  - ALIKED, LightGlue and LoMa run through ONNX Runtime (CoreML on macOS).
  - COLMAP published AnyCalib weights as a release (2026-09).
- **Robust estimation:**
  - PoseLib (BSD-3, `poselib` on PyPI, used inside COLMAP) and OpenCV USAC / MAGSAC++ are CPU-only and good enough.
  - No maintained GPU RANSAC exists.
  - Learned RANSAC (DSAC*, ESAC, NG-RANSAC) has stalled as research code. ACE / ACE0 is the practical successor, but
    it trains a network per scene (so it is not a fixed network) and is under a Niantic research licence (probably NC).
- **GPU solvers:** PyPose (Apache-2.0, pip) is maintained; Theseus is effectively unmaintained; MegBA has no wheel.
  InstantSfM is the only GPU-native SfM and is CC BY-NC.
- **Calibration targets:** OpenCV, mrcal (Apache-2.0, best accuracy), Kalibr and AprilTag are all fast on CPU.
  Undistortion is a `grid_sample`, so there is no kernel gap.
- **Licence traps:**
  - Non-commercial: MASt3R, DUSt3R, CUT3R, MASt3R-SLAM, InstantSfM, ACE (likely).
  - OpenMVS is AGPL, so only run it as a subprocess.
  - VGGT and VGGSfM use Meta custom terms.
  - GitHub shows NOASSERTION for COLMAP, Ceres and GTSAM, so read their LICENSE files.

Where each piece goes:

| piece | where | note |
|---|---|---|
| ALIKED / DISK / XFeat + LightGlue, RoMa | fastforks `matching` | must beat COLMAP's ONNX path; output in the shape `pycolmap.match_from_pairs` takes |
| global descriptors, top-k pairs | fastforks `retrieval` (new, P2) | DINOv3 / SigLIP2 / CLIP kernels exist; pairs for matching and loop closure |
| AnyCalib, GeoCalib | fastforks `calibration` (P2) | intrinsics + distortion as a COLMAP camera prior; keep it small, COLMAP may ship AnyCalib itself |
| VGGT, MapAnything (shipped; Apache weights `facebook/map-anything-apache`), π3 (BSD-3 code, BSD-2 weights, P1) | fastforks `multiview` | feed-forward poses and points |
| mapping, triangulation, BA, RANSAC | `pycolmap` (+ Caspar), PoseLib, OpenCV, called from fastsplat L2 | optional dependencies behind `poses="colmap" / "fastforks"` |
| COLMAP format I/O, undistortion, pose-quality checks | fastsplat L2 | thin glue |
| DROID-SLAM (BSD-3), DPVO (MIT), MegaSaM (Apache-2.0) | not planned | custom CUDA extensions with their own BA; revisit for video poses only if fastsplat needs it |
| GPU RANSAC / relative-pose verifier | only if a profile shows it dominates | after matching is fast |

First proof: fastforks matches + retrieval + the pycolmap global mapper against plain COLMAP on a few Mip-NeRF 360
scenes. Measure time and pose accuracy.

Not in a milestone yet: `retrieval`, `calibration`, π3 and this first proof. They belong to the COLMAP-free pose
path, which the splat library needs only in M3, so they are built when that path is.

### Customisation and training

Not model additions, but on the same milestones (design in
[fastforks-next.md](fastforks-next.md#customisation-without-losing-speed)):

| milestone | item |
|---|---|
| F4 | frozen kernel backbone + trainable head; resized heads (`num_classes=`); `fastforks.ops`; fine-tuning recipes per task (`fastforks finetune`) |
| F5 | broader `patch()` for any `nn.Module`; backward kernels for the most-used blocks (ViT block, attention, GEMM epilogues) |

## The splat library

Working name `fastsplat`; design in [splat-library.md](splat-library.md). It starts from our research project, and
whether the code base starts inside gsplat or as a new base inspired by it is decided after a short spike in M0
([splat-library.md](splat-library.md#starting-point)); the tables below hold for both routes. Every method below is **reimplemented** on
our own base unless the "route" column says its code may be adapted. "Inria text" means the repo's LICENSE is Inria's
non-commercial Gaussian-Splatting License, which binds derivatives (its §4.2 keeps the research-only limitation on
modified versions); for those repos we use the paper, the diff against Inria (to understand, not to copy) and the
authors' outputs.

Routes:

- **adapt**: the code is permissive and written by the authors (or gsplat); files may be taken with their headers
  and a `NOTICE` entry, after a per-file provenance check for Inria-derived parts.
- **reimplement**: write it from the paper on our layers; check against the paper's numbers and the authors' renders.

### M0–M1: base and fast 3DGS

| method / component | reference | licence | route | note |
|---|---|---|---|---|
| 3DGS (vanilla) | graphdeco-inria/gaussian-splatting | Inria text | reimplement | gsplat's `simple_trainer` (Apache-2.0) is the closest permissive reference |
| rasteriser (fused + composable) | the rasteriser of our research project, gsplat kernels | ours, Apache-2.0 | adapt | ~3x faster than gsplat's for standard splatting; what the number covers and each file's origin are recorded first; gsplat's as a parity reference |
| spike: our rasteriser in gsplat's `simple_trainer` | nerfstudio-project/gsplat | Apache-2.0 | throwaway | end-to-end number, profile of a training step, route decision |
| fused Adam, updates for visible Gaussians only | Taming 3DGS (idea) | — | own code | |
| fused SSIM | rahul-goel/fused-ssim | MIT | adapt | |
| tile culling / per-pixel sort ideas | StopThePop (r4dl) | Inria text | reimplement | if profiling shows popping or sort cost |
| gsplat-compatible API, `DefaultStrategy` | nerfstudio-project/gsplat | Apache-2.0 | adapt | signature compatibility, tested with gsplat's tests |

### M2: research core (geometry)

Geometry moves before the rest of gsplat parity because it is the core of our own research (see
[splat-library.md](splat-library.md#milestones)).

| method | reference | licence | route | note |
|---|---|---|---|---|
| 3DGS-MCMC | ubc-vision/3dgs-mcmc | Inria text | reimplement (gsplat's `MCMCStrategy` is Apache-2.0: adapt that) | |
| AbsGS (absolute gradient) | TY424/AbsGS | Inria text | reimplement (in gsplat: adapt) | |
| Revising Densification (Rota Bulò et al.) | no official code | — | reimplement from the paper | gsplat has a version |
| Mip-Splatting (3D filter + 2D mip filter) | autonomousvision/mip-splatting | Inria text | reimplement (in gsplat: adapt) | |
| 2DGS (surfels) | hbb1/2d-gaussian-splatting | Inria text | reimplement (gsplat's 2DGS rasteriser: adapt) | |
| depth / normal rendering | gsplat | Apache-2.0 | adapt | |
| Taming 3DGS (budget densification, fast backward) | humansensinglab/taming-3dgs | MIT file, Inria-derived code not audited | reimplement | |
| PGSR (planar / unbiased depth, multi-view NCC and geometry losses) | zju3dv/PGSR | ZJU custom, non-commercial | reimplement after a patent check | US 2025/0272943 "Planar Splatting" exists, assignee unknown |
| stereo / MVS depth priors with confidence, consistency check, TSDF fusion | fastforks `stereo`, `multiview` | per model card | own code | stronger than mono depth in our sparse-view results |
| `fastsplat.clean`, `fastsplat.mesh` (floater cull, hole fill, TSDF with settable weight threshold) | our research scripts | ours | own code | |
| our sparse-view pipeline (stereo prior, free-space pruning, cull / fill, mesh fusion) | our research | ours | port | first own method; the base methods it ran on are reimplemented, not copied |

### M3: breadth

| area | method | reference | licence | route | note |
|---|---|---|---|---|---|
| training speed | DashGaussian (resolution / count schedule) | YouyuChen0207/DashGaussian | CC BY-NC 4.0 | reimplement | |
| | Speedy-Splat (pruning, sparse pixels) | j-alex-hanson/speedy-splat | Inria text | reimplement | |
| | 3DGS-LM (Levenberg-Marquardt) | lukasHoel/3DGS-LM | Inria text + own terms | reimplement | P3 |
| compression | LightGaussian | VITA-Group/LightGaussian | Inria text | reimplement | |
| | Compact-3DGS | maincold2/Compact-3DGS | Inria text | reimplement | |
| | EAGLES | Sharath-girish/efficientgaussian | Inria text | reimplement | |
| | HAC / HAC++ | YihangChen-ee/HAC, HAC-plus | Inria text | reimplement | |
| formats | SPZ | nianticlabs/spz | MIT | adapt (or Rust port) | v4 uses ZSTD per-attribute streams, ~10x smaller than PLY |
| | SOG | playcanvas/splat-transform | MIT | adapt (or Rust port) | WebP images + `meta.json`, ~15–20x smaller; spec page states no licence |
| meshes | Gaussian Opacity Fields | autonomousvision/gaussian-opacity-fields | Inria text | reimplement | |
| | RaDe-GS | HKUST-SAIL/RaDe-GS | Inria text | reimplement **after a patent check** | US 2026/0024269 A1 "Rasterizing depth in gaussian splatting" (HKUST) |
| large scenes / LOD | Scaffold-GS, Octree-GS | city-super/Scaffold-GS, Octree-AnyGS | Inria text | reimplement | |
| appearance in the wild | Splatfacto-W | KevinXu02/splatfacto-w | Apache-2.0 | adapt | builds on gsplat |
| | WildGaussians | jkulhanek/wild-gaussians | MIT + Inria text appended | reimplement | |
| dynamic scenes | 4DGS (Wu et al.) | hustvl/4DGaussians | Apache-2.0 (rasteriser Inria-derived) | reimplement | |
| | Deformable 3DGS | ingra14m/Deformable-3D-Gaussians | MIT (rasteriser Inria-derived) | reimplement | |
| | SpacetimeGaussians | oppo-us-research/SpacetimeGaussians | MIT + Inria limitation clause | reimplement | |
| priors and poses | fastforks mono depth / normals / masks / features (stereo / MVS priors are in M2); VGGT, MapAnything, ALIKED / DISK + LightGlue, RoMa + bundle adjustment | fastforks | per model card | own code | MASt3R is non-commercial: not a default pose source |
| gsplat parity | bilateral-grid appearance | gsplat `lib_bilagrid`, harry7557558/fused-bilagrid | Apache-2.0, MIT | adapt | |
| | PPISP (bilateral-grid alternative) | in gsplat main | Apache-2.0 | adapt | not in a gsplat PyPI release yet |
| | pose refinement, per-image appearance embeddings | gsplat | Apache-2.0 | adapt | |
| | 3DGUT (unscented transform, distorted / rolling-shutter cameras) | nv-tlabs/3dgrut, gsplat | Apache-2.0 | adapt | |
| | nerfstudio plugin | nerfstudio-project/nerfstudio | Apache-2.0 | own code | nerfstudio's last release is v1.1.5 (2024-11); keep the plugin thin |
| platform | Windows native | — | — | — | shared kernel runtime with fastforks |

### M4: viewer and app

Rust formats crate (PLY / SPZ / SOG, PyO3), the existing 3D Visualizer as viewer (live training view through
`viz3d`), desktop app on its Tauri workspace (capture → poses → train with
live preview → cleanup with SAM → export / share). References to learn from: ArthurBrussee/brush (Apache-2.0, Rust +
wgpu, runs in the browser and on Android), playcanvas/supersplat (MIT, web editor), sparkjsdev/spark (MIT, web
renderer), nvpro-samples/vk_gaussian_splatting (Apache-2.0, Vulkan). LichtFeld Studio is GPL-3.0: look, do not copy.

### M5: more renderers and NeRF baselines

| method | reference | licence | route | note |
|---|---|---|---|---|
| 3DGRT (ray-traced Gaussians) | nv-tlabs/3dgrut | Apache-2.0 | adapt | needs OptiX; tiny-cuda-nn submodule |
| EVER (constant-density ellipsoids) | half-potato/ever_training, google/ever | Inria text; google/ever Apache-2.0 but uses an Inria-licensed renderer | reimplement | |
| Radiant Foam (Voronoi) | theialab/radfoam | Apache-2.0 | adapt | "research codebase" by its own description |
| Triangle Splatting, Triangle Splatting+ | trianglesplatting/triangle-splatting(2) | Apache-2.0 (uses Inria `simple-knn`) | adapt, replace `simple-knn` | |
| Deformable Beta Splatting | RongLiu-Leo/beta-splatting | Apache-2.0 (built on Inria, 3DGS-MCMC, gsplat) | adapt after a per-file check | |
| Instant-NGP (hash grid) | NVlabs/instant-ngp | NVIDIA non-commercial | reimplement **after a patent check** | a NVIDIA "multiresolution hash encoding" patent document was found, status unconfirmed |
| Zip-NeRF | google-research/multinerf (JAX), SuLvXiangXin/zipnerf-pytorch | Apache-2.0 | adapt | baseline for quality |
| Metal kernels | — | — | own | reference tier runs on MPS from M0 |

### Not planned in the splat library

| item | why |
|---|---|
| GS-SLAM (MonoGS: Imperial custom licence; Photo-SLAM: GPL-3.0; GS-ICP SLAM: Inria-derived rasteriser) | a different, stateful product; maybe later as methods once poses-from-fastforks works |
| generative 3D (text / image to 3D) | diffusion, a different field |
| human avatars (Apple patent application US 2025/0148678 on human-subject splatting) | SMPL licence, patents |

## Other libraries considered

| candidate | decision |
|---|---|
| SfM / SLAM library | not now. `fastsplat` L2 gets a small bundle adjustment on top of fastforks `matching` / `multiview`; a full SfM / SLAM library is reconsidered once that exists and its limits are known. |
| point cloud networks (PointTransformer, sparse convolutions) | not planned: different data and kernel set; the existing point-cloud viewer work (ply-visualizer) stays a viewer. |
| generation (image / video / 3D diffusion) | not planned: crowded, served by diffusers / ComfyUI, a different optimisation problem. |
| shared kernel runtime | yes, as a small package extracted from fastforks (loader, manifests, target selection, Hugging Face publishing, Windows support), used by fastforks and `fastsplat`. Built in M0, when work on `fastsplat` starts. |

## Ecosystem reference

Checked 2026-11-09 (`gh api`; product facts from secondary web sources, rough).

### Libraries and tools

| project | licence | last push | stars | releases / notes |
|---|---|---|---|---|
| graphdeco-inria/gaussian-splatting | Inria non-commercial | 2025-10 | 24.4k | the reference implementation |
| graphdeco-inria/diff-gaussian-rasterization | Inria non-commercial | 2024-10 | 1.5k | the rasteriser most paper repos fork |
| nerfstudio-project/gsplat | Apache-2.0 | 2026-09 | 5.9k | last release v1.5.3 (2025-07-04); main has LiDAR, PPISP, HiGS, SH updates "not yet on PyPI" |
| nerfstudio-project/nerfstudio | Apache-2.0 | 2025-07 | 12.2k | last release v1.1.5 (2024-11) |
| MrNeRF/LichtFeld-Studio | GPL-3.0 | 2026-11 | 4.0k | v0.5.4.2 (2026-11-01); C++23, CUDA 12.8+, NVIDIA only; MCMC, bilateral grid, 3DGUT; Python plugins |
| ArthurBrussee/brush | Apache-2.0 | 2026-10 | 5.3k | v0.3.0 (2025-09); Rust + wgpu; any GPU, browser, Android |
| nv-tlabs/3dgrut | Apache-2.0 | 2026-11 | 2.5k | v1.1.0 (2026-06); 3DGRT, 3DGUT; adopted code from gsplat |
| pierotofy/OpenSplat | AGPL-3.0 | 2026-10 | 2.3k | v1.2.2 (2026-09); C++ / libtorch; CUDA, ROCm, Metal, CPU |
| humansensinglab/taming-3dgs | MIT file (Inria fork, not audited) | 2025-06 | 368 | |
| rahul-goel/fused-ssim | MIT | 2026-02 | 246 | CUDA, Metal, ROCm |
| playcanvas/supersplat | MIT | 2026-11 | 10.6k | web editor, v3.7.0 |
| playcanvas/splat-transform | MIT | 2026-11 | 1.4k | converter (SOG) |
| sparkjsdev/spark | MIT | 2026-10 | 3.9k | web renderer, frequent releases |
| nianticlabs/spz | MIT | 2026-11 | 943 | SPZ format; C++, WASM, Python |
| nvpro-samples/vk_gaussian_splatting | Apache-2.0 | 2026-09 | 517 | Vulkan raster / ray-traced viewer |

### How gsplat ships CUDA

1. `pip install gsplat` (PyPI) is a source package; the CUDA code JIT-compiles with ninja on first use.
2. Prebuilt wheels for Linux and Windows per PyTorch × CUDA combination at `docs.gsplat.studio/whl/ptXXcuYYY`
   (v1.5.3 also attaches 11 wheels to the GitHub release).
3. `pip install git+https://github.com/nerfstudio-project/gsplat.git` builds from source.

`fastsplat` instead ships a pure-Python wheel and downloads prebuilt cubins / shared libraries by hash, independent of
the torch version, with an opt-in local source build for unpublished GPUs, air-gapped machines and kernel hacking
(see [splat-library.md](splat-library.md#l0-kernels-and-distribution)).

### Commercial products

| product | platforms | training | export |
|---|---|---|---|
| Scaniverse (Niantic) | iOS, Android, web | on-device (free); cloud for paid / 360 video | PLY (SPZ per reviews) |
| Postshot | Windows, NVIDIA (compute 7.5+) | local | PLY; free plan non-commercial, no splat export |
| Polycam | iOS, web (Android meshes only) | cloud (per reviews) | "Splat PLY"; splats on paid plans |
| Luma AI | — | cloud (per reviews) | unverified |
| KIRI Engine | mobile | cloud (per reviews) | unverified |

### Patents found (titles only, not exhaustive, no claims read)

| number | title | applicant | relevance |
|---|---|---|---|
| US 2024/0355047 A1 | 3DGS initialisation from trained NeRF representations | Intel (per Google Patents) | NeRF → 3DGS init |
| US 2026/0024269 A1 | Rasterizing depth in gaussian splatting | HKUST | RaDe-GS |
| US 2025/0272943 | Planar Splatting | unknown | PGSR-style planar methods |
| US 2025/0148678 | Human subject Gaussian splatting using machine learning | Apple | avatars |
| (USPTO doc, number 12711373 in URL) | Multiresolution hash encoding for neural networks | NVIDIA (unconfirmed) | Instant-NGP hash grid |

No patents found for Inria or the 3DGS authors. A freedom-to-operate search by a patent attorney is needed before any
commercial-use claim (query by assignee: Inria, NVIDIA, Google, Meta, Apple, Intel, Niantic, Adobe, Huawei).

## Still to verify

- Weights licences marked `unverified` above, before each model ships (most flow, matching, restoration and
  feed-forward splatting checkpoints; DEIM; ALIKED).
- RMBG-2.0 and DSINE licence texts; AP-10K terms; the body7 training-data terms behind RTMPose / RTMO / RTMW.
- What `OpenGVLab/VideoMAE2` (Apache-2.0 card) contains.
- Per-file provenance in permissive repos that build on Inria (Taming 3DGS, beta-splatting, Triangle Splatting,
  FlashGS, 4DGS, Deformable 3DGS) before any file is adapted.
- Patent claims and status for the list above; a broader search before the first commercial release.
- Official code for Revising Densification; whether multinerf contains Zip-NeRF.
