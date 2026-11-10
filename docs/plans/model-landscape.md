# Model landscape: open computer-vision models worth knowing, with licence and priority

Status (2026-11-09): survey only. It lists open models **outside** the ones fastforks ships today and outside the
ones already planned in [roadmap.md](roadmap.md) (flow, tracking, matching, pose, matting, detection additions,
video modes, normals, feed-forward splatting, optional tasks). Read the roadmap first; this file is the wider net
the roadmap draws from.

Licences were looked up on 2026-11-09 with `gh api` (repository SPDX id, the LICENSE file when that says
NOASSERTION) and the Hugging Face card `license` field. Weights that live on GitHub releases or Google Drive often
state no licence of their own; those are `unverified`. Training-data terms (ScanNet, CelebAMask-HQ, SMPL / MANO,
nuScenes, medical datasets) were not audited beyond the notes. None of this is legal advice.

Same conventions as [roadmap.md](roadmap.md#how-to-read-the-tables): `commercial` is yes / no / check, `[nc]` marks
non-commercial weights or code, and priority is

- **P1**: worth a slot in the next milestones (F2–F4), alongside the roadmap's P1 models;
- **P2**: add after the roadmap's P1 work of the same task, or when a user asks;
- **P3**: only on demand, or once a new task or input type exists for other reasons;
- **skip**: out of scope, API-only, copyleft we will not ship, or nothing fixed to ship.

## Summary

| area | decision | first models |
|---|---|---|
| backbones for `classification` / `features` | P1–P2, cheap: timm ports with Apache / MIT weights | ConvNeXt v1, MaxViT, RepViT, TIPSv2, V-JEPA 2 |
| depth and multiview additions | P1 for the clean licences | Depth Anything V2 Metric, Distill-Any-Depth, MoGe-2 |
| detection additions beyond the roadmap | P2 | RTMDet, DINO (DETR), LLMDet |
| segmentation additions | P1 for small SAMs, P2 open-vocabulary | EfficientViT-SAM, SAM-HQ, MobileSAM; FC-CLIP, CAT-Seg |
| restoration, interpolation (roadmap optional tasks) | raise to P2 | Real-ESRGAN, SwinIR, SPAN, NAFNet, RIFE |
| camera calibration, lines | P2, new small task | GeoCalib, AnyCalib, DeepLSD |
| video encoders, video object segmentation | P2 | V-JEPA 2, InternVideo2; Cutie |
| medical 2D (promptable, pathology, X-ray) | P2, model cards only, no new task | MedSAM, Cellpose-SAM, BiomedCLIP, permissive pathology encoders |
| medical 3D (CT / MR volumes) | P3, a separate `volume` task if ever | TotalSegmentator, VISTA3D, nnU-Net checkpoints |
| remote sensing foundation models | P3, needs multispectral input | Prithvi-EO 2.0, Clay, TerraMind |
| document layout / OCR | P3 (roadmap optional `ocr`) | PP-OCRv5, PP-DocLayout, TATR |
| weather restoration (dehaze, derain, low light) | P2, cards in `restoration` | DehazeFormer, AdaIR, Retinexformer, HVI-CIDNet |
| agriculture, ecology | P1–P2 cards in existing tasks, no new task | DeepForest, BioCLIP 2, Meta canopy height |
| image retrieval / pair selection | P2, new small task `retrieval` | DINOv3 / SigLIP2 global descriptors (see roadmap Geometry) |
| line segments, vanishing points | P2–P3, cards in `calibration` | ScaleLSD, DeepLSD; HAWP, NeurVPS |
| anomaly detection, IQA, ReID, point clouds, mono 3D detection, 3D generation, weather forecasting, shadow / reflection removal | skip | trained per dataset, NC zoos, privacy, other data types, generation, or unlicensed |

## Medical imaging: decision

**Yes for 2D medical models as ordinary model cards; no new library; 3D volumes later and only on demand.**

Why:

- **What fits today.** 2D medical models are the same networks fastforks already runs: MedSAM is a SAM ViT-B
  fine-tune, Cellpose-SAM is a SAM ViT-L encoder, pathology foundation models (Virchow, H-optimus-0, Prov-GigaPath)
  are plain ViT-L/H/g, BiomedCLIP is an open_clip ViT-B/16. They become cards in `promptable_segmentation`,
  `segmentation` and `features`, reuse the existing kernels and cost little.
- **What does not fit.** About 70% of the radiology models are 3D volumetric (TotalSegmentator, VISTA3D, nnU-Net,
  SwinUNETR, MedNeXt, SAM-Med3D, MedSAM2, nnInteractive). They need 3D convolution and attention kernels (a new
  kernel family), sliding-window inference with Gaussian blending, resampling, and NIfTI / DICOM input. That is a
  new input type and pipeline, not a new card.
- **nnU-Net is not one network.** Its `plans.json` sets patch size, spacing, stage count and normalisation per
  dataset and rebuilds the U-Net from it. In fastforks it would be a family of PlainConvUNet / ResEncUNet
  configurations, each shipped as a plans + weights pair (TotalSegmentator's tasks are exactly that).
- **Speed is not the pain point there.** TotalSegmentator v1 on an RTX 3090: 1:17–3:32 per CT at 1.5 mm,
  34 s–1:23 with the 3 mm `--fast` model (paper, arXiv 2208.05868); v2 is somewhat slower. How much of that is
  resampling, I/O and model load rather than network time has not been profiled. Our kernels would help batch
  processing but are not why people choose a tool.
- **Licences are per task and often restrictive.** TotalSegmentator mixes Apache-2.0 tasks with licence-key and
  CC-BY-NC tasks; most pathology encoders are CC-BY-NC-ND and gated; medical training data carries its own terms.
- **Regulation.** Every model below is research use only. Each medical card states "research use only, not a
  medical device", and the docs never claim clinical suitability.
- **Papers from the field.** Track them the same way as any other paper: a model gets a card when it meets the four
  rules in [fastforks-next.md](fastforks-next.md#what-belongs-in-fastforks). In practice that means 2D SAM / ViT /
  U-Net derivatives with permissive weights.

If a `volume` task is built later (P3), its first cards are TotalSegmentator's Apache-2.0 tasks and VISTA3D, and it
needs: 3D conv / attention kernels, a sliding-window runner, a NIfTI reader (nibabel, optional dependency) and
spacing-aware resampling.

### Kaapana, RACOON and where medical speed matters

Researched 2026-11; numbers marked "unconfirmed" still need a primary source.

- **Kaapana** (DKFZ, [kaapana/kaapana](https://github.com/kaapana/kaapana)) is open source under AGPL-3.0 and
  actively developed. It is a Kubernetes/Helm platform: Airflow workflows, dcm4chee PACS, OHIF viewer, OpenSearch,
  MinIO, Keycloak. Its shipped workflow extensions are nnU-Net predict/train/install-model, TotalSegmentator (v1 CT,
  v2 CT+MR), body-and-organ analysis (BOA), shapemodel organ segmentation, radiomics, a ResNet18
  classification train/infer, MITK Flow (manual segmentation) and DICOM registration. No MONAI or MedSAM in the
  documented list. Each model runs as a container per Airflow task (unconfirmed).
- **RACOON** (NUM, all 38 German university clinics) runs on Kaapana. The software is open; the data and the
  per-project models are not. Named subprojects: RACOON-MARDER (liver MRI, HCC), RACOON-CORE-PE (pulmonary
  embolism), PRECISE-MD (pancreas and liver, ~20 sites). Their engines are almost certainly nnU-Net and
  TotalSegmentator. Kaapana also serves JIP (11 DKTK clinics) and DART (7 Cancer Core Europe centres). RACOON was
  extended; the end date and funding of the extension are unconfirmed.
- **What this means.** Kaapana and RACOON are 3D conv U-Net shops (nnU-Net, TotalSegmentator, VIBESegmentator,
  MRSegmentator). Our 2D ViT kernels do not help them; only the P3 `volume` task would. Batch cost is small: ~1–3 min
  per CT on a GPU, so even 100k scans is ~2k GPU-hours. Per-job container start and model load is probably a larger
  fixed cost than the kernels (unconfirmed). A pitch to them needs the 3D path first and a profile of one
  TotalSegmentator v2 run (network vs resampling vs I/O vs load) before any speed claim.
- **Where we fit now: interactive and 2D.**
  - Click-to-mask in 3D Slicer, OHIF and napari: SAMM in Slicer needs a one-time SAM embedding of 163 s for a
    352×352×240 volume on an RTX 3090, then 0.6 s per click (arXiv 2304.05622). The embedding is slice-wise 2D
    ViT work, which is exactly our kernels.
  - nnInteractive (NC weights, so no card) reports ~0.3 s per click on an A100 and ~15 s on CPU (secondary source).
    It shows the latency users expect.
  - Cellpose-SAM: ~0.7 s per 1000×1000 image on an L40S. An open TensorRT PR to Cellpose shows 180 → 96 ms on an
    RTX 5090, which proves demand and gives us a number to beat. This is the best single medical target.
  - Pathology encoders (QuPath, slide-level feature extraction over millions of tiles) are plain ViTs, and batch
    throughput there does cost money.
- **Grand Challenge** reloads the model for every image job: jobs cost at least 20 credits and users get 1,000
  credits a month. Load time, not kernels, dominates there.

### Medical and microscopy models

| model | task | repo | code | weights | commercial | 2D/3D | priority | note |
|---|---|---|---|---|---|---|---|---|
| MedSAM | promptable seg | bowang-lab/MedSAM | Apache-2.0 | Apache-2.0 (wanglab/medsam-vit-base) | yes | 2D | P2 | SAM ViT-B weights swap; first medical card |
| Cellpose-SAM (Cellpose v4) | cell seg | MouseLand/cellpose | BSD-3 | BSD-3 (mouseland/cellpose-sam) | yes | 2D | P2 | SAM ViT-L encoder + flow decoding; big microscopy user base |
| Cellpose v3 | cell seg | MouseLand/cellpose | BSD-3 | BSD-3 | yes | 2D | P3 | small U-Net |
| BiomedCLIP | features | microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (HF) | — | MIT | yes | 2D | P2 | open_clip ViT-B/16 |
| Virchow (v1) | pathology features | paige-ai/Virchow (HF) | — | Apache-2.0, gated | yes | 2D | P2 | ViT-H; gating complicates golden download |
| H-optimus-0 | pathology features | bioptimus/H-optimus-0 (HF) | — | Apache-2.0, gated | yes | 2D | P2 | ViT-g |
| Prov-GigaPath (tile encoder) | pathology features | prov-gigapath/prov-gigapath | Apache-2.0 | Apache-2.0, gated | yes | 2D | P2 | slide encoder (LongNet) out of scope |
| Hibou-L, Midnight | pathology features | histai/hibou-L, kaiko-ai/midnight (HF) | — | Apache-2.0 (gated) / MIT | yes | 2D | P2 | permissive alternatives |
| Rad-DINO | chest X-ray features | microsoft/rad-dino (HF) | — | unverified (card says MSRLA) | check | 2D | P3 | read licence first |
| micro-SAM | microscopy SAM | computational-cell-analytics/micro-sam | MIT | unverified | check | 2D | P3 | SAM fine-tunes |
| StarDist | nuclei | stardist/stardist | BSD-3 | unverified | check | 2D/3D | P3 | TensorFlow; star-convex NMS |
| CellSAM | cell seg | vanvalenlab/cellSAM | Apache-2.0 | unverified | check | 2D | P3 | check deepcell token |
| SAM-Med2D | promptable seg | OpenGVLab/SAM-Med2D | Apache-2.0 | unverified | check | 2D | P3 | |
| TotalSegmentator (open tasks) | CT / MR organs | wasserth/TotalSegmentator | Apache-2.0 | Apache-2.0 | yes | 3D | P3 (`volume`) | nnU-Net; per-task licences |
| VISTA3D / NV-Segment-CTMR | CT / MR seg | Project-MONAI/VISTA, NVIDIA-Medtech/NV-Segment-CTMR | Apache-2.0 | NVIDIA Open Model Licence | check | 3D | P3 (`volume`) | read NVIDIA terms |
| CT-FM | CT features + seg | project-lighter/CT-FM | MIT | Apache-2.0 | yes | 3D | P3 (`volume`) | SegResNet |
| Merlin | CT features | StanfordMIMI/Merlin | MIT | MIT | yes | 3D | P3 (`volume`) | image encoder only |
| MedSAM2 | promptable 3D / video | bowang-lab/MedSAM2 | Apache-2.0 | CC-BY-SA-4.0 | check | 3D | P3 | share-alike on weights; SAM 2 kernels |
| MRSegmentator, STU-Net, SAM-Med3D | 3D seg | hhaentze/MRSegmentator, uni-medical/STU-Net, uni-medical/SAM-Med3D | Apache-2.0 | unverified | check | 3D | P3 | |
| nnU-Net v2, MONAI bundles, Auto3DSeg, MedNeXt | frameworks | MIC-DKFZ/nnUNet, Project-MONAI/* | Apache-2.0 | per dataset / per bundle | — | 3D | skip | frameworks, not fixed networks |
| [nc] TotalSegmentator licensed tasks, nnInteractive, BiomedParse v1, UNI / UNI2, CONCH, Virchow2, H-optimus-1 | various | — | Apache / Other | CC-BY-NC(-SA/-ND) or licence key | no | 2D/3D | skip | NC weights; UNI / CONCH / Virchow2 also no-derivatives |
| Phikon, RETFound, Google HAI-DEF (Path / CXR / Derm Foundation, MedSigLIP) | features | owkin, rmaphoh, google | Other | custom / unverified | check | 2D | skip until read | custom terms with clinical-use clauses |

## Remote sensing

Most Earth-observation foundation models are ViT-MAE variants (kernels exist) but take multispectral, multi-temporal
input with wavelength or sensor embeddings: a new input contract (`channels != 3`, band metadata). P3 until a user
asks; the work is in the input side, not the kernels.

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| Prithvi-EO 2.0 (100M / 300M / 600M) | NASA-IMPACT/Prithvi-EO-2.0 | MIT | Apache-2.0 | yes | P3 | HLS bands, temporal 3D patches |
| Clay v1 | Clay-foundation/model | Apache-2.0 | Apache-2.0 | yes | P3 | wavelength-conditioned patch embed |
| TerraMind 1.0 | ibm/terramind | Apache-2.0 | Apache-2.0 | yes | P3 | multimodal (S1, S2, DEM) |
| Galileo | nasaharvest/galileo | MIT | MIT | yes | P3 | |
| AnySat | gastruc/AnySat | MIT | MIT | yes | P3 | |
| DOFA | (TorchGeo) | unverified | CC-BY-4.0 | yes | P3 | |
| Copernicus-FM | zhu-xlab/Copernicus-FM | Other (unread) | CC-BY-4.0 | check | P3 | |
| DeepForest | weecology/DeepForest | MIT | MIT | yes | P3 | RetinaNet on RGB; fits `detection` today |
| SAMGeo (segment-geospatial) | opengeos/segment-geospatial | MIT | SAM | yes | skip | wrapper; works with our SAM directly |
| SatMAE, Scale-MAE, SkySense | — | Other / unverified | unverified | check | skip | likely NC or no weights |
| AlphaEarth Foundations | — | — | no weights | no | skip | embeddings only, in Earth Engine |

## Additions to existing tasks

### Classification and features backbones

timm ports with permissive weights are the cheapest additions in fastforks: they reuse the conv / ViT kernels and
timm is the golden reference.

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| ConvNeXt v1 | facebookresearch/ConvNeXt | MIT | Apache-2.0 (timm) | yes | P1 | sibling of shipped ConvNeXt v2 |
| MaxViT, CoAtNet | google-research/maxvit, timm | Apache-2.0 | Apache-2.0 (timm) | yes | P2 | hybrid conv + attention |
| RepViT, SHViT, StarNet, TinyViT, EfficientFormer, HGNetV2 | THU-MIG/RepViT, ysj9909/SHViT, timm | Apache-2.0 / MIT | Apache-2.0 / MIT (timm) | yes | P2 | mobile-latency families; reparam fuses well |
| FocalNet, DaViT | microsoft/FocalNet, dingmyu/davit | MIT | MIT / Apache-2.0 (timm) | yes | P3 | also detection backbones |
| InternImage | OpenGVLab/InternImage | MIT | MIT | yes | P3 | needs a DCNv3 kernel (new) |
| VMamba | MzeroMiko/VMamba | MIT | unverified | check | P3 | needs a selective-scan kernel (new) |
| TIPS / TIPSv2 | google-deepmind/tips | Apache-2.0 | Apache-2.0 | yes | P1 | text-image model with strong dense features |
| Franca | valeoai/Franca | Apache-2.0 | unverified | check | P2 | open-data DINOv2-class SSL |
| V-JEPA 2 | facebookresearch/vjepa2 | MIT | MIT / Apache-2.0 | yes | P2 | video encoder; first `features` video input |
| InternVideo2 (vision encoder) | OpenGVLab/InternVideo2 | MIT | MIT / Apache-2.0 per checkpoint | check | P3 | large; check each checkpoint |
| MobileNetV5 | timm | Apache-2.0 | Gemma terms | check | P3 | Gemma 3n encoder |
| [nc] Hiera (standalone), MambaVision, AIMv2, Web-SSL, MobileCLIP / 2, VideoMAE v2 | — | — | NC / research-only | no | skip | SAM 2's own Hiera weights stay Apache-2.0 |

### Depth and multiview

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| Depth Anything V2 Metric (Hypersim / VKITTI) | DepthAnything/Depth-Anything-V2 | Apache-2.0 | Apache-2.0 (incl. Large) | yes | P1 | same kernels; the relative V2-Large is NC, the metric Large is not |
| Distill-Any-Depth | Westlake-AGI-Lab/Distill-Any-Depth | MIT | MIT | yes | P1 | DA2-style, permissive |
| MoGe-2 | microsoft/MoGe | MIT | MIT | yes | P1 | metric scale + normals; extends the shipped `moge` |
| [nc] Pi3 / Pi3X | yyfz/Pi3 | BSD-3 | CC-BY-NC-4.0 (repo README at 9fa3ddb; the HF tag `bsd-2-clause` on Pi3 is stale) | no | P3 | weights non-commercial (checked 2026-11-10); commercial use only by asking the authors |
| VGGT-1B-Commercial | facebookresearch/vggt | Meta custom | VGGT AUP, gated | check | P2 | the checkpoint a commercial user needs; read terms |
| Prompt Depth Anything | DepthAnything/PromptDA | Apache-2.0 | no card licence | check | P2 | LiDAR prompt = new input |
| [nc] Pi3X, Fast3R, CUT3R, StreamVGGT, MUSt3R | — | NC / research | NC | no | P3 at best | streaming variants are interesting for video |
| GeoWizard, VGGT-X, PURE | — | none / Other | unverified | check | skip | diffusion, or not a standalone model |

### Detection

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| RTMDet | open-mmlab/mmdetection | Apache-2.0 | Apache-2.0 | yes | P2 | pairs with RTMPose as its person detector |
| DINO (DETR) | IDEA-Research/DINO | Apache-2.0 | unverified | check | P2 | baseline many DETRs build on |
| LLMDet | iSEE-Laboratory/LLMDet | Apache-2.0 | unverified | check | P2 | open-vocabulary, Grounding DINO lineage |
| PP-YOLOE+, PicoDet | PaddlePaddle/PaddleDetection | Apache-2.0 | Apache-2.0 | yes | P3 | Paddle weights need conversion |
| DAMO-YOLO, NanoDet, Gold-YOLO | tinyvision/DAMO-YOLO, RangiLyu/nanodet, huawei-noah | Apache-2.0 / unverified | unverified | check | P3 | permissive YOLO-style options |
| RT-DETRv3 | clxia12/RT-DETRv3 | Apache-2.0 | unverified | check | P3 | third-party port; find the authors' release |
| YOLO-NAS | Deci-AI/super-gradients | Apache-2.0 | separate Deci terms (unread) | check | skip until read | |
| [nc] YOLOv6 / v9 / v10 / Ultralytics, FastSAM | — | GPL / AGPL | GPL / AGPL | no | skip | copyleft |
| Grounding DINO 1.5, DINO-X, T-Rex | — | API only | — | — | skip | no weights |

### Segmentation and promptable segmentation

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| EfficientViT-SAM | mit-han-lab/efficientvit | Apache-2.0 | Apache-2.0 | yes | P1 | fast SAM; EfficientViT kernels already shipped |
| SAM-HQ | SysCV/sam-hq | Apache-2.0 | Apache-2.0 | yes | P1 | SAM weights + small HQ head; better mask edges |
| MobileSAM | ChaoningZhang/MobileSAM | Apache-2.0 | Apache-2.0 (`weights/mobile_sam.pt` is committed in the Apache-2.0 repository; MIT on the co-author's Hugging Face copy, same file) | yes | P2 | ported 2026-11-10; TinyViT encoder (MIT, microsoft/Cream); distilled on 1 % of SA-1B |
| EfficientSAM | yformer/EfficientSAM | Apache-2.0 | unverified | check | P3 | |
| FC-CLIP | bytedance/fc-clip | Apache-2.0 | unverified | check | P2 | open-vocabulary panoptic on frozen CLIP; Mask2Former kernels |
| CAT-Seg | cvlab-kaist/CAT-Seg | MIT | unverified | check | P2 | open-vocabulary semantic |
| Mask DINO | IDEA-Research/MaskDINO | Apache-2.0 | unverified | check | P3 | |
| X-Decoder, SEEM | UX-Decoder/* | MIT / Apache-2.0 | unverified | check | P3 | |
| SegNeXt | open-mmlab/mmsegmentation | Apache-2.0 | Apache-2.0 | yes | P3 | check whether `mmseg` already covers it |
| Panoptic-DeepLab, kMaX-DeepLab | — | Apache-2.0 | unverified | check | skip | superseded; kMaX is JAX/TF |
| Grounded-SAM | IDEA-Research/Grounded-Segment-Anything | Apache-2.0 | composes | yes | skip as a model | a recipe: Grounding DINO + SAM, both planned |
| Cutie (video object segmentation) | hkchengrex/Cutie | MIT | unverified | check | P2 | memory-based VOS; alternative to SAM 2 video |
| [nc] EdgeSAM, OVSeg, DEVA | — | S-Lab / CC-BY-NC | NC | no | skip | |

### Stereo and pose

| model | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|
| Lite Any Stereo (V1, V2) | TomTomTommi/LiteAnyStereo | MIT | unverified | check | P1 | efficient zero-shot stereo, CVPR 2026 |
| [nc] Fast-FoundationStereo, Stereo Anywhere | NVlabs/Fast-FoundationStereo, bartn8/stereoanywhere | NVIDIA NC / CC-BY-NC-SA | NC | no | P2 | |
| Sapiens2 | facebookresearch/sapiens2 | Sapiens2 licence | Sapiens2 licence | check | P2 | read terms; would replace Sapiens in pose / seg / depth / normals |
| PCT | Gengzigang/PCT | MIT | unverified | check | P3 | |
| [nc] ED-Pose | IDEA-Research/ED-Pose | IDEA licence | — | no | skip | |

## Other task areas

### Restoration and interpolation (roadmap optional tasks, raised to P2)

The strongest new area: one-shot image-to-image networks, many permissive, real demand (video upscaling, photo
apps, ComfyUI), and the speed-ups are real. Most weights sit on GitHub releases and need a licence check.

| model | task | repo | code | weights | commercial | priority |
|---|---|---|---|---|---|---|
| Real-ESRGAN | SR | xinntao/Real-ESRGAN | BSD-3 | BSD-3 (releases) | yes | P2 |
| SwinIR | SR / denoise / JPEG | JingyunLiang/SwinIR | Apache-2.0 | Apache-2.0 (repo) | yes | P2 |
| SPAN | efficient SR | hongyuanyu/SPAN | Apache-2.0 | unverified | check | P2 |
| NAFNet, Restormer | denoise / deblur | megvii-research/NAFNet, swz30/Restormer | MIT | unverified | check | P2 |
| RIFE (Practical-RIFE) | interpolation | hzwer/Practical-RIFE | MIT | MIT (README) | yes | P2 |
| HAT, DRCT, Retinexformer, Swin2SR, BSRGAN, DPIR | SR / low-light | various | Apache-2.0 / MIT | unverified | check | P3 |
| EMA-VFI, BasicVSR++ | interpolation / video SR | MCG-NJU/EMA-VFI, ckkelvinchan/BasicVSR_PlusPlus | Apache-2.0 | unverified | check | P3 (BasicVSR++ needs DCN) |
| VFIMamba | interpolation | MCG-NJU/VFIMamba | Apache-2.0 | Apache-2.0 | yes | P3 (scan kernel) |
| GFPGAN | face restoration | TencentARC/GFPGAN | Apache-2.0 + third-party | unverified | check | P3 |
| [nc] PromptIR, CodeFormer, RVRT, ResShift, SUPIR | — | NC | NC | no | skip |

### Camera calibration and lines (new small task, P2)

Single-image fixed networks that feed `matching`, `multiview` and the splat library. Candidate task name
`calibration` (intrinsics, gravity) with line segments as a `matching` output.

| model | task | repo | code | weights | commercial | priority |
|---|---|---|---|---|---|---|
| AnyCalib | intrinsics | javrtg/AnyCalib | Apache-2.0 | Apache-2.0 | yes | P2 |
| GeoCalib | intrinsics + gravity | cvg/GeoCalib | Apache-2.0 | unverified | check | P2 |
| DeepLSD, GlueStick | line detection / matching | cvg/DeepLSD, cvg/GlueStick | MIT | unverified | check | P2 |
| DexiNed, TEED | edge detection | xavysp/DexiNed, xavysp/TEED | MIT | unverified | check | P3 (ComfyUI line-art preprocessors) |

### Document, faces, aesthetics

| model | task | repo | code | weights | commercial | priority | note |
|---|---|---|---|---|---|---|---|
| PP-OCRv5, PP-DocLayout, SLANeXt, UVDoc | OCR, layout, tables, dewarp | PaddlePaddle/PaddleOCR, PaddleX | Apache-2.0 | Apache-2.0 | yes | P3 | the roadmap `ocr` task; Paddle conversion |
| Table Transformer (TATR) | table structure | microsoft/table-transformer | MIT | MIT | yes | P3 | DETR kernels |
| docTR | OCR | mindee/doctr | Apache-2.0 | unverified | check | P3 | |
| LAION / improved aesthetic predictor | aesthetics | LAION-AI/aesthetic-predictor | MIT / Apache-2.0 | MIT / unverified | yes / check | P2 | MLP on CLIP; trivial once CLIP ships; used by every video-curation pipeline (Open-Sora, OpenVid, NeMo Curator) |
| TransNetV2 | shot boundaries | soCzech/TransNetV2 | MIT | MIT (unverified) | check | P2 | small 3D-conv net; first stage of video-curation pipelines; pairs with flow and embeddings (roadmap "Where usage is visible") |
| face-alignment (FAN), MediaPipe face landmarks | landmarks | 1adrianb/face-alignment, google-ai-edge/mediapipe | BSD-3 / Apache-2.0 | unverified | check | P3 | roadmap `face` task |
| [nc] / [agpl] Surya, DocLayout-YOLO, MinerU | — | — | NC / AGPL / custom | — | no | skip | |
| GOT-OCR2.0, dots.ocr, Florence-2, Q-Align | — | — | — | — | — | skip | VLM-style decoders |

### Not planned

| area | why |
|---|---|
| anomaly detection (anomalib, PatchCore, EfficientAD, Dinomaly) | trained per customer dataset; nothing fixed to ship. The DINOv2 backbones they use are already in `features`. |
| image quality (pyiqa zoo: TOPIQ, MANIQA, MUSIQ) | the usable zoo is PolyForm-NC / CC-BY-NC-SA; MUSIQ is TF-only |
| AI-image / deepfake detectors | fragile, short-lived models with mixed licences |
| person re-identification, gaze, crowd counting, face parsing | privacy, and training data is mostly NC |
| multi-object tracking (ByteTrack, OC-SORT, BoT-SORT, MOTIP) | mostly association logic around a detector; a recipe on top of `detection`, not a model |
| 3D human / hand mesh (HMR2, HaMeR, WiLoR, Multi-HMR), 6D pose (FoundationPose, MegaPose) | SMPL / MANO / BFM are NC; FoundationPose code is NC (see [roadmap.md](roadmap.md#where-each-topic-goes)). Exception to check: SAM 3D Body uses Meta's MHR rig (Apache-2.0) instead of SMPL; weights are under the gated Meta SAM License (trade-control clauses, commercial use unconfirmed). P3 `pose` mesh output if the licence reads clean |
| image-to-3D generation (TRELLIS / TRELLIS.2 MIT, TripoSG MIT, Step1X-3D Apache, Hunyuan3D, SF3D / SPAR3D) | generation, multi-step sampling; Hunyuan3D's licence excludes the EU, UK and South Korea; Stability's is gated with a revenue threshold. Only TripoSR (MIT code and weights, one feed-forward pass to a triplane) would fit, P3 on demand |
| point clouds (PTv3, Sonata, Concerto, OpenScene), lidar / BEV 3D detection, occupancy | other data type and kernel set (sparse conv, serialisation); PTv3 LICENSE is CC-BY-NC while its card says MIT, Sonata / Concerto weights are CC-BY-NC, nuScenes / ScanNet / HM3D are NC. Belongs in neither fastforks nor fastsplat |
| intrinsic decomposition, relighting | diffusion-based, i.e. generation |
| weather forecasting and nowcasting (GraphCast / GenCast, Aurora, AIFS, FourCastNet 3, NeuralGCM, Prithvi-WxC, Pangu [nc]) | gridded atmospheric tensors (~70+ channels on 721×1440), JAX GNNs and spherical operators, autoregressive rollouts, ERA5 data access. NVIDIA earth2studio / physicsnemo and ECMWF anemoi already ship optimised runners. Aurora (MIT, Swin-3D U-Net) is the only vision-like one |
| shadow detection / removal, reflection removal, glass / mirror segmentation | small field, mostly no licence; ShadowFormer and DSRNet only on request (table above) |

## Weather, agriculture, shadows, reflections, lines

Checked 2026-11-09. Most weights in these areas sit on Google Drive or Baidu: all are `unverified` until their weights
licence has been read. Many repos have **no licence file at all** (all rights reserved): DRSformer, DEA-Net,
MB-TaylorFormer, Histoformer, OneRestore, MoCE-IR, Zero-DCE, SCI, SNR-Aware, most shadow and glass nets. PromptIR,
MPRNet and MIRNet are under the non-commercial Academic Public License.

| model | area | repo | code | weights | commercial | arch | priority | where |
|---|---|---|---|---|---|---|---|---|
| DehazeFormer | dehazing | IDKiro/DehazeFormer | MIT | unverified | check | Swin U-Net | P2 | `restoration` |
| AdaIR | all-in-one (rain, haze, noise, blur, low light) | c-yn/AdaIR | MIT | unverified | check | Restormer-style + frequency modulation | P2 | `restoration` |
| Retinexformer | low light | caiyuanhao1998/Retinexformer | MIT | unverified | check | Restormer-like (1.6k stars) | P2 | `restoration` |
| HVI-CIDNet | low light | Fediory/HVI-CIDNet | MIT | MIT (HF card) | yes | cross-attention U-Net | P2 | `restoration` |
| IDR, TransWeather, Uformer, DarkIR | all-in-one / weather / low light | zhangyi-3/IDR, jeya-maria-jose/TransWeather, ZhendongWang6/Uformer, cidautai/DarkIR | MIT | unverified | check | transformer U-Nets | P3 | `restoration` |
| FUnIE-GAN | underwater | IRVLab/FUnIE-GAN | MIT | unverified | check | tiny GAN | P3 | `restoration` |
| DeepForest (tree, bird, livestock) | aerial trees / animals | weecology/DeepForest | MIT | MIT (HF, 105k downloads a month) | yes | RetinaNet R50 | P1 | `detection` |
| BioCLIP, BioCLIP 2 | species (plants, animals, fungi) | Imageomics/bioclip, bioclip-2 | MIT | MIT (125k / 215k downloads) | yes | open_clip ViT-B/16, ViT-L/14 | P1 | `features` / `classification` with the CLIP port; TreeOfLife data terms not audited |
| HighResCanopyHeight | canopy height from satellite RGB | facebookresearch/HighResCanopyHeight | Apache-2.0 | Apache-2.0 (README) | yes | DINOv2 ViT-L + DPT | P2 | `depth` (satellite tiles) |
| Detectree2 | tree-crown instances | PatBall1/detectree2 | MIT | unverified | check | detectron2 Mask R-CNN | P3 | `segmentation` |
| CountGD, CounTR | class-agnostic counting (fruit, leaves) | niki-amini-naieni/CountGD, Verg-Avesta/CounTR | MIT | unverified | check | Grounding DINO / ViT + exemplars | P3 | `detection` |
| PlantNet-300K, PlantCLEF DINOv2 | plant species | PlantNet/PlantNet-300K, dsgt-arc/plantclef-2025 | BSD-2 / MIT | unverified | check | ResNet / DINOv2 heads | P3 | `classification` |
| ShadowFormer | shadow removal | GuoLanqing/ShadowFormer | MIT | unverified | check | window-attention U-Net | P3 only on request | `restoration` |
| DSRNet, ERRNet | reflection removal | mingcv/DSRNet, Vandermode/ERRNet | Apache-2.0 / MIT | unverified | check | dual-stream CNN | P3 only on request | `restoration` |
| ScaleLSD | line segments | ant-research/scalelsd | MIT | unverified | check | HAWP-style transformer | P2 | `calibration` (lines, with DeepLSD) |
| HAWP, LCNN, NeurVPS, LETR | wireframes, lines, vanishing points | cherubicXN/hawp, zhou13/lcnn, zhou13/neurvps, mlpc-ucsd/LETR | MIT / MIT / MIT / Apache-2.0 | unverified | check | hourglass / conic conv / DETR | P3 | `calibration` |
| [nc] PromptIR, MPRNet, MIRNet, RIDCP, Mask-ShadowGAN, Perspective Fields, Pangu-Weather | — | — | NC | NC | no | — | skip | — |

Decisions:
- **Weather restoration** becomes cards in the planned `restoration` task with a "weather" filter in the docs, not a
  new task. HF usage is negligible, so it comes after Real-ESRGAN, SwinIR and NAFNet.
- **Agriculture** gets no task: real farm work is per-site training, which the fixed-network rule excludes. The
  reusable networks are DeepForest, BioCLIP and Meta's canopy height model, and they are cards in existing tasks.
- **Shadow, reflection, glass and mirror**: not planned. The field is tiny and mostly unlicensed; there is no
  licensed glass / mirror segmenter.
- **Vanishing points and lines** become optional cards in `calibration`. GeoCalib already predicts gravity and the
  horizon.
- **Weather forecasting** (GraphCast, GenCast, Aurora, AIFS, FourCastNet 3, NeuralGCM, Prithvi-WxC): not computer
  vision in our sense; see "Not planned".

## U-Net coverage

Checked 2026-11-09 against `_optimize/conv_ops.py`, `_optimize/transformer.py` and `kernels/`.
**Partly covered: 2D conv U-Nets and one SD UNet yes; 3D U-Nets no.**

| block | status |
|---|---|
| 2D conv + BN / affine + ReLU / SiLU, depthwise, pooling | covered (`conv_fused`, `dwconv`, `maxpool`, `avgpool`) |
| conv + GroupNorm + SiLU (SD UNet in Marigold) | channels-last PyTorch fast path in `marigold/net/fast.py`, not a fused kernel |
| attention inside a UNet | SDPA; ViT / Swin block kernels exist but are not wired into UNets (unverified) |
| upsample / transposed conv + skip concat | `seg_up` for segmentation heads; no fused upsample + concat + conv |
| 3D conv | only the stereo cost-volume shapes (`conv3d`, `conv3d_subpixel` in the FoundationStereo / Fast-FoundationStereo CUTLASS sets) |
| 3D instance norm, depthwise 3D conv (MedNeXt), sliding window, NIfTI | missing; that is the `volume` task |

U-Net candidates by value per effort:
1. **SD UNet one-step models** (Marigold exists; Lotus, Lotus-2, StableNormal planned): a fused conv + GroupNorm +
   SiLU kernel and UNet attention on the ViT kernels speed all of them up at once.
2. **Plain 2D U-Nets**: U²-Net / ISNet (rembg's legacy models, Apache-2.0 code), DexiNed, NAFNet (U-shaped, conv +
   SimpleGate, very fusable), the DISK backbone. They need a fused upsample + concat + conv; Restormer additionally
   needs channel ("transposed") attention. U²-Net / ISNet are P3 because BiRefNet / BEN2 are better.
3. **3D U-Nets** (nnU-Net PlainConvUNet / ResEncUNet, MONAI SegResNet / SwinUNETR): the largest real user base
   (Kaapana, RACOON), but only with the P3 `volume` task, and speed hurts least there.

## Kornia

Kornia (Apache-2.0, 12.4k stars, 2.2M PyPI downloads a month, v0.9.0 2026-10) is a differentiable CV and geometry
library, not an inference library. Its own ROADMAP.md froze model-zoo growth (new models need a named maintainer;
wrappers are "usable or deleted"). Its planned work is correctness and conventions, `torch.compile` / ONNX export,
cutting augmentation orchestration overhead (78% of a flip's time), a few Triton kernels (median blur, Canny) and a
batched RANSAC.

| Kornia part | overlap with us | decision |
|---|---|---|
| `kornia.feature` pretrained (LoFTR, LightGlue, DISK, DeDoDe, ALIKED, XFeat, SOLD2, KeyNet, HardNet) | `matching` covers LightGlue, DISK, ALIKED, XFeat, LoFTR | compete on speed in `matching`; follow Kornia's keypoint and pixel-centre conventions; offer a small adapter so Kornia users can swap in our backend. DeDoDe v2 (MIT, weights unverified) is a P2 candidate; lines go through DeepLSD / GlueStick, SOLD2 is a P3 fallback at most |
| `kornia.models` / `contrib` wrappers (SAM, RT-DETR, YuNet, DexiNed, Depth Anything, RRDBNet, SigLIP2) | all exist or are planned in fastforks | nothing new; their wrappers run eager PyTorch |
| `kornia.geometry` (RANSAC, homography, epipolar, PnP, camera models, undistortion, Lie groups) | none; algorithms, not networks | ignore; see [roadmap.md](roadmap.md#geometry-sfm-ransac-bundle-adjustment-calibration) |
| augmentation, filters, enhance, colour, morphology, losses, metrics | none | ignore; the bottleneck is launch overhead, which Kornia is attacking with CUDA graphs |
| `kornia-rs` (Apache-2.0, 751 stars, v0.2.0 2026-09, 2.0M PyPI downloads a month, mostly as a dependency of `kornia`) | input side and geometry, not the networks | see below |

**kornia-rs** is more than image I/O (checked 2026-11-09 from its crates):
- `kornia-io`, `kornia-image`, `kornia-imgproc`: libjpeg-turbo, PNG, TIFF and WebP decoding, SIMD resize and
  colour ops, one `Image` type on CPU or CUDA, zero-copy to PyTorch through DLPack, and V4L2 / GStreamer capture with a
  fused NV12 / YUYV → normalised CHW CUDA kernel. Wheels for Linux (incl. Jetson), macOS and Windows; free-threaded
  Python supported.
- `kornia-3d`: bundle adjustment (Schur), pose-graph optimisation, PnP, RANSAC, ICP / registration, stereo.
- `kornia-calib`: board-based multi-shot calibration. `kornia-bow`: ORB-SLAM3 vocabularies. `kornia-apriltag`.
- `kornia-vlm`: PaliGemma and SmolVLM(2) in Rust.

What it means for us:
- **Input pipeline.** Decoding and resizing on the CPU is the ceiling for batch throughput once the network is fast.
  Benchmark kornia-rs decode + resize + normalise (and its fused camera kernel) against our current path. If it
  wins, use it as an optional fastforks input backend instead of writing our own, and as fastsplat L2's image
  loader and cache (thousands of frames per scene).
- **Rust formats crate (fastsplat M4).** kornia-rs covers images, not splat formats (PLY / SPZ / SOG), so no
  overlap; share the `Image` / DLPack conventions if both end up in one process.
- **Geometry.** `kornia-3d` is a second option next to pycolmap / PoseLib for RANSAC, PnP and small BA in fastsplat;
  pycolmap stays the default because it is what the benchmarks use.
- **No competition** on the networks: its VLM crate is out of our scope.

## Still to verify

- Every `unverified` weights licence above before the model ships, especially those on GitHub releases or Google
  Drive (SPAN, NAFNet, Restormer, GeoCalib, DeepLSD, FC-CLIP, CAT-Seg, Cutie, LLMDet, Lite Any Stereo).
- Custom terms not read in full: Sapiens2, VGGT-Commercial AUP, NVIDIA Open Model Licence (VISTA3D, C-RADIO), Gemma
  terms (MobileNetV5), Google HAI-DEF, Owkin (Phikon), RETFound, Rad-DINO (MSRLA).
- Conflicts between code and card: DocLayout-YOLO (AGPL code, Apache card), PTv3 (CC-BY-NC LICENSE, MIT card),
  FoundationPose (NC code, NVIDIA open-model card).
- Training-data terms for medical weights, and for any model trained on ScanNet, CelebAMask-HQ, nuScenes or AVA.
- Weights licences of the restoration, agriculture and line networks in "Weather, agriculture, shadows, reflections,
  lines" (Drive / Baidu hosted), and BioCLIP's TreeOfLife data terms.
- Not surveyed: radar nowcasting code (NowcastNet not located; DGMR in deepmind-research).
