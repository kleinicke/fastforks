# ComfyUI coverage: which vision networks popular node packs run, and which fastforks has

Status (2026-11-13): survey only. Goal (user, 2026-11-13): networks that are very popular in ComfyUI belong in
fastforks when their licence allows commercial use. Complements the ComfyUI landscape in the roadmap
("Platforms and integrations", "Where usage is visible") and [model-landscape.md](model-landscape.md).

**Method.** All 5,875 packs of the Comfy registry (`https://api.comfy.org/nodes?page=N&limit=100`, all-time
downloads, fetched 2026-11-13), ranked by downloads; kept the packs that run a vision network other than the
diffusion model itself. Licences: `gh api repos/<repo>` SPDX id, the LICENSE text when that says NOASSERTION, and
the HF card `license` field. HF usage: `huggingface.co/api/models/<id>` monthly downloads (file-only repos such as
`lllyasviel/Annotators` or `yzd-v/DWPose` report 0, so pack downloads are the signal there). Downloads include CI
and mirrors: read them as ranks. Not legal advice; training-data terms not audited.

## Packs (registry rank, all-time downloads) and the networks they run

| # | pack | downloads | vision networks |
|---|---|---|---|
| 5 / 11 | ltdrdata Impact Pack / Impact Subpack | 3.64M / 2.38M | SAM, Ultralytics YOLOv8 face / hand / person (`Bingsu/adetailer`), CLIPSeg (optional) |
| 8 | Fannovel16/comfyui_controlnet_aux | 2.83M | DWPose (+ YOLOX), OpenPose, Depth Anything V1 / V2, MiDaS, Zoe, LeReS, Metric3D, BAE / DSINE normals, HED, PiDiNet, TEED, lineart (+ anime, manga), MLSD, OneFormer, UniFormer, DensePose, UniMatch flow, MediaPipe face, LaMa, AnimalPose (RTMPose), MeshGraphormer |
| 3 / 7 / 1 | Easy-Use / Essentials / KJNodes | 3.84M / 3.15M / 4.80M | rembg sessions (U²-Net, RMBG-1.4, InSPyReNet), MediaPipe / LIP human parsing, CLIPSeg |
| 12 / 83 | chflame163 LayerStyle / LayerStyle_Advance | 2.27M / 214k | BiRefNet, BEN, SAM / SAM 2, Grounding DINO, Florence-2, YOLOv8, ViTMatte, LaMa, MediaPipe, SegFormer clothes / fashion, JoyCaption |
| 15 | Fannovel16/ComfyUI-Frame-Interpolation | 1.99M | RIFE, FILM, GMFSS Fortuna, IFRNet, FLAVR, AMT, CAIN, STMFNet, M2M |
| 16 | ssitu/UltimateSDUpscale (+ core `UpscaleModelLoader`, spandrel) | 1.92M | ESRGAN family (Real-ESRGAN, community 4x-*), SwinIR, HAT, DAT, SPAN, OmniSR |
| 18 | kijai/ComfyUI-Florence2 | 1.70M | Florence-2 base / large (+ PromptGen fine-tunes) |
| 19 / 22 | WAS node suite (two ids) | 1.63M / 1.19M | BLIP caption, CLIPSeg, MiDaS, rembg U²-Net, SAM |
| 20 | cubiq IPAdapter_plus | 1.63M | CLIP vision ViT-H / bigG, InsightFace (FaceID) |
| 23 | numz SeedVR2_VideoUpscaler | 1.15M | SeedVR2 3B / 7B |
| 27 | comfy-mtb | 1.00M | FILM, GFPGAN, rembg, DeepBump |
| 28 | 1038lab/ComfyUI-RMBG | 969k | RMBG-2.0, BiRefNet, BEN / BEN2, InSPyReNet, SAM / SAM 2 / SAM 3, Grounding DINO, SegFormer clothes / fashion / face |
| 30 / 45 | Gourieff ReActor (two ids) | 948k / 598k | InsightFace (inswapper, buffalo_l), RetinaFace / YOLO face, GFPGAN, CodeFormer, GPEN, RestoreFormer++, ViT NSFW classifier |
| 42 | pythongosssss/ComfyUI-WD14-Tagger | 624k | WD v3 taggers (SwinV2, ViT, EVA02, ConvNeXt) |
| 49 / 143 | Acly inpaint-nodes / lama-remover | 533k / 108k | LaMa, MAT |
| 51 / 96 | AdvancedLivePortrait / LivePortraitKJ | 494k / 183k | LivePortrait (+ InsightFace or MediaPipe face) |
| 52 / 93 / 162 | segment-anything-2 / comfyui-sam2 / SAM3 packs | 477k / 185k / 84k (+63k, 31k) | SAM 2.1, SAM 3 |
| 56 / 81 / 59 | PuLID (two) / InstantID | 415k / 225k / 388k | InsightFace antelopev2, EVA-CLIP, facexlib parsing |
| 58 | SUPIR | 392k | SUPIR |
| 64 | kijai DepthAnythingV2 | 361k | Depth Anything V2 |
| 71 | Inspyrenet-Rembg | 279k | InSPyReNet |
| 74 | kijai WanAnimatePreprocess | 269k | ViTPose-L / H whole-body (ONNX), YOLOv10 person, SAM 2 |
| 109 | joy_caption_two | 154k | JoyCaption (Llama VLM) |
| 156 / 163 | face_parsing / person-mask-generator | 90k / 84k | SegFormer face parsing, MediaPipe selfie multiclass |
| 164 / 213 / 237 | DepthCrafter / Marigold / Video-Matting | 83k / 59k / 51k | DepthCrafter, Marigold, RVM |

Skipped as not vision networks: rgthree, VideoHelperSuite, GGUF, Crystools, WanVideo / Hunyuan / CogVideo wrappers,
LoRA managers, LLM / Ollama / Qwen-VL nodes.

## Networks, ranked by usage

Status: **have** = shipped family; **road** = in the roadmap or model-landscape; **miss** = in neither.
Reuse names existing fastforks families / kernels. Effort: S = weights onto an existing family, M = new head or
small new family, L = new large family.

| # | network | packs | usage signal | code / weights | commercial | fastforks | reuse | effort |
|---|---|---|---|---|---|---|---|---|
| 1 | SAM / SAM 2.1 / SAM 3 | Impact, SAM2 packs, RMBG, LayerStyle, WAS | Impact 3.6M; HF sam2.1-l 250k/mo | Apache / Apache (SAM 3 per card) | yes | have | — | — |
| 2 | YOLOv8 face / hand / person | Impact Subpack, LayerStyle | `Bingsu/adetailer` 8.5M/mo | AGPL-3.0 / AGPL ([ultralytics](https://github.com/ultralytics/ultralytics)) | no (copyleft) | never | substitute: YuNet face (road), RF-DETR / D-FINE person (have) | — |
| 3 | DWPose (+ YOLOX-L det) | controlnet_aux, WanAnimate, dwposedeluxe | controlnet_aux 2.8M | Apache / Apache ([DWPose](https://github.com/IDEA-Research/DWPose), [HF](https://huggingface.co/yzd-v/DWPose)) | check (training data) | road P1 (`rtmw-*`, `dwpose-*`) | `rtmpose` SimCC head, CSPNeXt, `rtmdet` | S |
| 4 | Depth Anything V2 / V3, Video DA, MiDaS, Zoe, Metric3D, Marigold | controlnet_aux, DA packs, Marigold | DA V2 S 1.4M/mo | per card | per card | have | — | — |
| 5 | RIFE 4.x (+ FILM, GMFSS, IFRNet, FLAVR) | Frame-Interpolation, mtb | 2.0M pack; Replicate rife 2M+ runs | MIT / MIT ([RIFE](https://github.com/hzwer/Practical-RIFE)); FILM, FLAVR Apache; GMFSS, IFRNet MIT | yes | road (opt. P2/P3) | `raft_flow` warp / conv kernels, `cutlass_conv` | S–M |
| 6 | ESRGAN family (Real-ESRGAN, RRDB / Compact), SwinIR, HAT, DAT, SPAN | UltimateSDUpscale, core loader | 1.9M pack + core | Real-ESRGAN BSD-3 ([repo](https://github.com/xinntao/Real-ESRGAN)); SwinIR, HAT, DAT Apache; community weights vary (4x-UltraSharp [CC BY-NC-SA](https://huggingface.co/Kim2091/UltraSharp)) | yes (official weights) | road (opt. P2) | `cutlass_conv`, Swin window attention (BiRefNet / swin) | M |
| 7 | Florence-2 base / large | Florence2, LayerStyle, RMBG, captioners | HF base 14.1M/mo, large 402k/mo; 1.7M pack | MIT / MIT ([HF card](https://huggingface.co/microsoft/Florence-2-base)) | yes | miss (model-landscape: skip as VLM decoder) | DaViT (`classification/davit`) + BART decoder | M |
| 8 | BiRefNet, BEN2, ViTMatte | RMBG, BiRefNet_ll, LayerStyle | BiRefNet 1.0M/mo, ViTMatte 305k/mo | MIT / MIT | yes | have | — | — |
| 9 | U²-Net / IS-Net (rembg default) | WAS, Essentials, Easy-Use, mtb | rembg PyPI 2.5M/mo | Apache / Apache ([U-2-Net](https://github.com/xuebinqin/U-2-Net)) | yes (weights unverified) | miss | `cutlass_conv` | S |
| 10 | InSPyReNet (transparent-background) | Inspyrenet-Rembg, RMBG, Easy-Use, LayerStyle | 279k pack; PyPI 134k/mo | MIT / MIT ([repo](https://github.com/plemeri/InSPyReNet), [pkg](https://github.com/plemeri/transparent-background)) | yes | miss | Swin-B (`matting/birefnet` Swin kernels) | S–M |
| 11 | InsightFace (inswapper, buffalo_l, antelopev2) | ReActor, IPAdapter FaceID, InstantID, PuLID, FaceAnalysis | ReActor 1.5M | MIT / non-commercial ([insightface](https://github.com/deepinsight/insightface)) | no | never (no face recognition) | — | — |
| 12 | OpenPose body / hand / face | controlnet_aux | controlnet_aux 2.8M | CMU [non-commercial](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE) | no | miss | — | `[nc]` S |
| 13 | LaMa (big-lama) | inpaint-nodes, lama-remover, LayerStyle, controlnet_aux | 533k + 108k packs | Apache / Apache ([repo](https://github.com/advimman/lama)) | yes (weights from repo releases) | miss | FFC = cuFFT + conv; `cutlass_conv` | M |
| 14 | CLIP vision ViT-H / bigG | IPAdapter, core CLIPVision | CLIP-H 537k/mo, bigG 94k/mo | MIT / MIT ([HF](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)) | yes | have H/14; bigG miss | `features/clip` | S |
| 15 | WD v3 taggers (SwinV2, ViT-L, EVA02-L) | WD14-Tagger | 624k pack; swinv2 561k/mo, vit-large 4.0M all-time | Apache / Apache ([HF](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)) | yes | miss | timm `vit`, `eva02`, `swin` (SwinV2 new) | S |
| 16 | GFPGAN v1.4 (+ facexlib RetinaFace) | ReActor, mtb | ReActor 1.5M | Apache + third-party ([LICENSE](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE)) / unverified | check | road P3 (model-landscape) | StyleGAN2 conv, `cutlass_conv` | M |
| 17 | BLIP caption base / large | WAS, Easy-Use interrogator | HF base 1.28M/mo, large 400k/mo | BSD-3 / BSD-3 ([HF](https://huggingface.co/Salesforce/blip-image-captioning-base)) | yes | miss | `classification/vit` + BERT decoder | M |
| 18 | CLIPSeg rd64 | KJNodes, WAS, Impact, clipseg | HF 425k/mo | MIT (code, "not weights", [LICENSE](https://github.com/timojl/clipseg/blob/master/LICENSE)) / Apache ([HF card](https://huggingface.co/CIDAS/clipseg-rd64-refined)) | yes | miss | `features/clip` B/16 | S |
| 19 | ViT NSFW classifiers | ReActor, nsfw packs | Falconsai 1.6M/mo, AdamCodd 706k/mo | Apache / Apache ([HF](https://huggingface.co/Falconsai/nsfw_image_detection)) | yes | miss | `classification/vit` (weights only) | S |
| 20 | Line / edge preprocessors: HED, TEED (Anyline), lineart (informative-drawings), MLSD, UniFormer | controlnet_aux, Anyline | controlnet_aux 2.8M; Anyline 48k | HED, UniFormer Apache ([ControlNet](https://github.com/lllyasviel/ControlNet)); TEED MIT; lineart MIT; MLSD Apache | yes (weights via `lllyasviel/Annotators`, card `other`: check) | road P3 (TEED, model-landscape) | small CNNs, `cutlass_conv`; little GPU time, small win | S each |
| 21 | ViTPose whole-body L / H | WanAnimatePreprocess | 269k pack | Apache / Apache | check (data) | have (ViTPose++; expose whole-body head) | `pose/vitpose` | S |
| 22 | LivePortrait | AdvancedLivePortrait, LivePortraitKJ, liveportrait | 494k + 183k + 38k | MIT / MIT ([LICENSE](https://github.com/KwaiVGI/LivePortrait/blob/main/LICENSE), [HF](https://huggingface.co/KlingTeam/LivePortrait)); upstream face detector InsightFace NC | yes (with a non-InsightFace detector) | miss | `deform_conv`-style warp, grid_sample, SPADE decoder | L |
| 23 | MediaPipe face mesh / selfie multiclass | controlnet_aux, person-mask-generator, Easy-Use, LayerStyle | 84k pack (+ inside larger ones) | Apache / Apache ([mediapipe](https://github.com/google-ai-edge/mediapipe)) | yes | road P3 (model-landscape) | TFLite to PyTorch conversion; tiny | M |
| 24 | SeedVR2 3B / 7B | SeedVR2 | 1.15M pack; HF `numz/SeedVR2_comfyUI` 790k/mo | Apache / Apache ([repo](https://github.com/ByteDance-Seed/SeedVR)) | yes | road (opt.) | DiT attention / RoPE from `vggt`, `depth_anything_3` | L |
| 25 | Depth Anything V1 | controlnet_aux | inside controlnet_aux | Apache / Apache ([HF](https://huggingface.co/LiheYoung/depth-anything-large-hf)) | yes | miss | `depth_anything_v2` DPT kernels (`dpt_fused`) | S |
| 26 | BAE normals | controlnet_aux | inside controlnet_aux | MIT ([repo](https://github.com/baegwangbin/surface_normal_uncertainty)) / unverified | check | miss (DSINE road P3) | `classification/efficientnet` B5 | S |
| 27 | DensePose R50 | controlnet_aux (MagicAnimate / Champ workflows) | inside controlnet_aux | Apache ([detectron2](https://github.com/facebookresearch/detectron2)) / model zoo | check | miss | `detection/rcnn` | M |
| 28 | UniMatch flow | controlnet_aux | inside controlnet_aux | MIT / unverified | check | road P2 | `flow/gmflow` (same authors) | S |
| 29 | SegFormer clothes / fashion / face parsing | RMBG, LayerStyle, face_parsing | clothes 401k/mo, face-parsing 156k/mo | NVIDIA SCL ([card](https://huggingface.co/mattmdjaga/segformer_b2_clothes) → NVlabs licence) / face-parsing card "non-commercial" ([card](https://huggingface.co/jonathandinu/face-parsing)) | no | arch have | `segmentation/segformer` | `[nc]` S |
| 30 | RMBG-2.0 / RMBG-1.4 | RMBG, Easy-Use | RMBG-2.0 393k/mo, 1.4 247k/mo | Bria licences, gated ([2.0](https://huggingface.co/briaai/RMBG-2.0)) | no | road P3 (`rmbg-2`) | `matting/birefnet` (2.0 is BiRefNet) | `[nc]` S |
| 31 | CodeFormer, GPEN, RestoreFormer++ | ReActor, GPEN | ReActor 1.5M | CodeFormer [S-Lab NC](https://github.com/sczhou/CodeFormer/blob/master/LICENSE); GPEN no licence file | no | miss | — | `[nc]` M |
| 32 | AMT, MAT, SUPIR, 4x-UltraSharp, CoTracker, DepthCrafter | Frame-Interpolation, inpaint-nodes, SUPIR, UltimateSDUpscale, cotracker, DepthCrafter | SUPIR 392k pack | CC BY-NC ([AMT](https://github.com/MCG-NKU/AMT/blob/main/LICENSE), [MAT](https://github.com/fenglinglwb/MAT/blob/main/LICENSE)), SUPIR / UltraSharp / CoTracker NC, DepthCrafter research-only | no | CoTracker road `[nc]`; rest skip | — | — |
| 33 | RVM (Robust Video Matting) | Video-Matting | 51k pack | GPL-3.0 ([repo](https://github.com/PeterL1n/RobustVideoMatting)) | no (copyleft) | never | — | — |
| — | JoyCaption, Qwen-VL, Moondream | caption / VLM packs | JoyCaption 154k + 33k | Llama / Qwen licences | — | out of scope (LLM VLMs) | — | — |

**Coverage.** Of the 33 rows, 6 are shipped (SAM family; the depth suite; BiRefNet / BEN2 / ViTMatte; CLIP-H;
ViTPose whole-body; SegFormer architecture), 9 are on the roadmap or model-landscape (DWPose, RIFE, ESRGAN family,
GFPGAN, edge preprocessors, MediaPipe, SeedVR2, UniMatch, RMBG-2.0), 18 are missing or never. Of the missing ones,
10 are commercially usable (2 more to check: BAE normals, DensePose).

## Add next (commercially usable, by usage)

1. **DWPose / RTMW** (Apache / Apache, data check): controlnet_aux 2.8M, WanAnimate 269k; already P1. S.
2. **Florence-2 base / large** (MIT / MIT): 14.1M HF/mo, 1.7M pack plus LayerStyle / RMBG; reverses the
   model-landscape "skip". DaViT exists, BART decoder new. M.
3. **RIFE 4.x** (MIT / MIT): 2.0M pack, ComfyUI standard; FILM (Apache) and IFRNet (MIT) as cheap extras. S–M.
4. **ESRGAN family** (Real-ESRGAN BSD-3; SwinIR / HAT / DAT Apache): 1.9M pack plus every core upscale workflow;
   load spandrel-format checkpoints so users bring their own weights (licence is theirs). M.
5. **U²-Net / IS-Net** (Apache): rembg 2.5M PyPI/mo, the default in WAS / Essentials / Easy-Use. S.
6. **InSPyReNet** (MIT / MIT): 279k pack + RMBG / Easy-Use; Swin-B reuse. S–M.
7. **WD v3 taggers** (Apache / Apache): 624k pack, 561k/mo; timm ViT / EVA02 exist, SwinV2 new. S.
8. **LaMa** (Apache / Apache): 641k across inpaint packs; FFC via cuFFT. M.
9. **CLIPSeg** (MIT code, Apache weights) and **ViT NSFW classifiers** (Apache): 425k/mo and 1.6M/mo; weights on
   existing `clip` / `vit`. S each.
10. **BLIP caption** (BSD-3 / BSD-3): 1.28M/mo, WAS / Easy-Use. M.

Then: CLIP bigG (S), Depth Anything V1 (S), ViTPose whole-body head (S), GFPGAN (check weights, M), LivePortrait
with a MediaPipe face detector (L), SeedVR2 (L, biggest compute win), line / edge preprocessors (S each, small win).

`[nc]` candidates (popular, non-commercial; only as `[nc]` cards): RMBG-2.0 (BiRefNet arch, S), OpenPose (S),
SegFormer clothes / face parsing (S), CodeFormer (M). Never: YOLOv8 / v10 / 11 (AGPL), InsightFace (NC, face
recognition), RVM (GPL).
