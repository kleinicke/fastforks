# Datacenter tuning notes

Options that were noted while tuning on the RTX 5060 Ti (sm_120, GeForce) and could pay on datacenter GPUs
(A100 / L40S / H100 class), collected per model family. Nothing here is built or measured on those GPUs unless a
bullet says so; numbers quoted are from the 5060 Ti, and anything a source only suggests is marked as uncertain.

Already built and measured for full-rate cards (A10, L4, L40S, A100): [full-rate-gpus.md](full-rate-gpus.md). The
surveys themselves: [device-survey.md](device-survey.md) and [OTHER_GPUS.md](../OTHER_GPUS.md).

## What the existing plans already cover

- **Full-rate fp32 accumulation** (`full-rate-gpus.md`): GeForce runs fp16 tensor-core maths with fp32 accumulation at
  half rate, data-centre and workstation cards at full rate, so the fp16-accumulate kernels gain less there. The
  class is decided from the device name (`half_rate_fp32_acc`, `FASTFORKS_ACC=fp16|fp32` overrides it).
- **Shipped for those cards**: `DptFused` takes the tanh GELU and larger tiles (cfg 4) on full-rate cards. Depth-pro on
  the L4 went from 564 to 484 ms and on the A10 from 367 to 349 ms in the lab (experiment 4).
- **Not pursued there**: new large-warp tiles and cuBLASLt (cuBLASLt "stays an option only for Hopper").
- **Open there, Hopper / Blackwell datacenter (sm_90, sm_100)**: the binaries use sm_80-style `mma.sync`, without
  wgmma or TMA. The plan is to run the GEMM shoot-out on an H100 first; if the gap to cuBLAS is large, use cuBLASLt
  or a CUTLASS 3.x kernel with an epilogue visitor tree there.
- **Surveys** (`device-survey.md`, `OTHER_GPUS.md`): the A100, A10, L4, L40S and RTX PRO 6000 pages are speed
  surveys checked against the golden outputs. The survey numbers mix kernel builds; a full re-run is planned.
  An H100 is not among the surveyed cards.

## Optical flow: RAFT, SEA-RAFT

- **Tensor-core (mma) correlation lookup for large channel counts.** It would read f1 once per 8-pixel tile; the
  report expects it to help most on H100 / B200 at 1080p and above, where the lookup dominates. Not built. Source:
  agent report, 2026-11-09/10 (RAFT `kernels` runner).
- **Persistent fused update iterations** that keep the hidden state in shared memory, removing about 10 launches per
  iteration (the SEA-RAFT summary words it as running all 12 update iterations inside one kernel). Not built.
  Source: agent report, 2026-11-09/10.
- **Batching pairs into one pass.** The kernels runner takes a batch pair by pair (a B=2 batch matches single calls
  exactly). Batching would use a big GPU better. Not built. Source: agent report, 2026-11-09/10.
- **FP8 encoders on sm_90 and newer.** Not built; accuracy effect unknown. Source: agent report, 2026-11-09/10.
- **Turning CUDA graphs back on where the GPU outruns the host's launches.** On the 5060 Ti a whole-forward graph
  saves only 0.1-0.3 ms (about 0.5-2 %) because the pass is GPU-bound from KITTI size up, so there is no
  `SmallShapeGraphs` and no `cuda_graphs` option. Source: agent report, 2026-11-09/10.
- Measured for reference (5060 Ti, 12 iterations, kernels vs portable fp16): raft-large 20.9 vs 43.8 ms at 440x1024
  and 82.8 vs 294.6 ms at 1088x1920 (peak 367 MiB vs 8388 MiB); raft-small 11.7 vs 23.1 ms at 440x1024.
- **TensorRT at 1080 x 1920 needs more than 16 GB to build.** At the default builder level TensorRT fuses the
  all-pairs correlation into one node and asks for a 21 GB buffer while tuning it (all four RAFT variants; the rows in
  `docs/data/speed_compile.jsonl` stay failures on the 5060 Ti). A card with 24 GB or more should build it. At
  `builder_optimization_level = 0` it builds on 16 GB, but raft-large then takes 338 ms and 8.2 GB (torch.compile:
  195 ms, 5.9 GB; kernels: 83 ms), so that is not offered. Measured 2026-11-10, TensorRT 11.4.
- SEA-RAFT: no datacenter notes of its own were found beyond the RAFT-era ones above (the same ideas were
  repeated in the summary of that work).

## Pose: ViTPose, RTMPose, DWPose

- No datacenter notes found in the repo or in the session transcript.
- Related, from the RTMDet card (the usual detector in front of RTMPose): batching is the headroom. See the
  detection section.

## Feature backbones: SigLIP 2, CLIP, Perception Encoder, C-RADIO

- **`fp32_accumulate` option.** All four cards carry it, default off: every GEMM in the `kernels` mode goes through
  cuBLAS with fp32 accumulation (the portable precision, inside the CUDA graphs). On GeForce it costs 15-40 %
  (half-rate fp32 accumulation); on full-rate cards it should cost almost nothing, but this is untested and the
  cards say "to be measured on data-centre GPUs". Measured on the 5060 Ti: CLIP-L mean error drops from 0.022 to
  0.0126, about the portable level; for `clip-h14-378-dfn5b`, the outlier with the largest kernels error, it cuts the
  error by roughly a factor of 10. A per-device default (on for full-rate cards) is the open question. Sources:
  `src/fastforks/features/models/{clip,siglip2,perception_encoder,c_radio}/card.toml`; agent report, 2026-11-09/10.
- **CUTLASS fp16-accumulate mainloop with fp32 promotion every K-chunk.** A middle way between the fast GEMMs and
  fp32 accumulation. Needs a kernel rebuild. Not built. Source: agent report, 2026-11-09/10.
- **Small-row dispatch.** At small row counts inside CUDA graphs, cuBLAS (fp16 accumulate) plus a torch epilogue beat
  the fused GEMM; at 1025 rows and more the fused GEMM was 1-3 % faster. Where that crossover lies on a datacenter
  GPU is not measured. Source: agent report, 2026-11-09/10.

## Classification: ConvNeXt (and the other `fvm_conv` families)

The measurements below are from an RTX 5060 Ti. Datacenter suggestions are untested; profile on the target GPU.

- **Batching.** Batch 1 is launch bound on datacenter GPUs (about 100 launches per call). A graph per batch size exists
  up to 2^20 input pixels; raise that limit and re-measure the two row thresholds (cuBLAS for GEMMs of up to 600 rows,
  cuDNN depthwise conv for maps of up to 256 rows), which are tuned for the 5060 Ti. Source:
  `src/fastforks/classification/models/convnext/card.toml`.
- **TF32 / fp32 accumulation** may have a smaller relative cost on datacenter GPUs, but no measurements are
  available. On the 5060 Ti it did not improve top-1 accuracy, and an fp32-accumulate second GEMM for wide layers
  was slower (Large 3.45 to 3.88 ms). Source: same card.
- **bf16** has no fused conv kernels. **INT8** belongs to a TensorRT export. Source: same card.

- **ConvNeXt as a feature backbone** (`features`, `convnext_*.fb_in22k`) runs the same trunk
  (`src/fastforks/shared/convnext_trunk`), so the points above apply unchanged. The maps of stages 1-3 (strides 4, 8,
  16) are computed anyway; returning them needs a multi-scale output in the features API. Not built.

## Detection: RTMDet

- **Batching.** The net is convolutions only, so batching is where the headroom is: one graph per batch size
  (decoding and NMS are per image already). Not built. Source: `src/fastforks/detection/models/rtmdet/card.toml`.
- **TF32** buys nothing over fp16 and shifts boxes (cuDNN's default TF32 convolutions are already 3.1 px off on
  the S variant against the CPU golden; float32 without TF32 is within 0.002 px). Source: same card.
- **bf16** has no fused conv kernels. **FP8 / INT8** needs calibration and moves box AP, so it belongs to a TensorRT
  export, not to the `kernels` mode. Source: same card.
- Larger graphs are already done (the whole network plus decoding is one graph). Measured on the 5060 Ti: the
  kernels path went from about 95 to 20 ms and from 5-7 GB to under 1.2 GB at the same mAP (the report text this
  comes from is abridged; see `docs/models/detection.md` for the rows). Source: agent report, 2026-11-09/10.

## Depth: Depth Anything V2 (incl. Metric and Distill-Any-Depth), Marigold, DPT-based models

- **Larger batches.** The kernels path is tuned for batch 1. Source: agent report, 2026-11-09/10 (Depth Anything V2
  Metric / Distill-Any-Depth).
- **FP8 GEMMs on sm_89 / sm_90 and newer; bf16 or TF32 on A100.** Not tried; accuracy effect unknown. Source: same
  report.
- **CUDA graphs around the whole call.** Not tried for these models. Source: same report.
- **Fused sigmoid option in `head_dot`** for the metric head. Not worth it on the 5060 Ti: the separate sigmoid pass
  costs about 0.02 ms (metric-S 4.67 ms against DA2-S 4.64 ms). Source: same report.
- **Per-card tiles and GELU** for the DPT GEMMs: built, see `full-rate-gpus.md` and
  `src/fastforks/kernels/dpt_fused/__init__.py` (`DptFused.full_rate`, `DptFused.gelu`). The Metric and Distill
  variants share the DA2 network and get the same 1.7x over portable (S 4.63-4.67 ms, B 10.8 ms, L 31.7-31.9 ms
  kernels at 994x1482). Their tolerances come from 5060 Ti measurements only and may be tight on other GPUs.
- **Marigold: launch-bound on fast cards.** The channels-last group-norm helper keeps the launch count low (7 with
  SiLU) because on fast cards (L40S) the kernels forward is launch-bound; on the full-rate A10 the layout copies of
  `F.group_norm` made kernels slower than portable. Survey row on the L40S: 140 ms portable, 128 ms kernels (1.1x,
  [l40s-44gb.md](../devices/l40s-44gb.md)). Whether graphs or fewer launches would help further is uncertain. Source:
  `src/fastforks/depth/models/marigold/net/fast.py`.
- MoGe: no notes of its own. The `full-rate-gpus.md` survey mentions MoGe-2-vits at +4 % (L4) and +20 % (L40S) in
  kernels mode, called one-off small-model noise there.

## Stereo: FoundationStereo and the other stereo families

- No datacenter-specific notes found. Related facts in the cards:
  - CreStereo, HITNet and IGEV++ cards note that default TF32 convolutions in float32 change outputs against the CPU
    (about 0.06-0.2 px mean at 720x1280; HITNet matches the CPU to 0.00002 px with
    `torch.backends.cudnn.allow_tf32 = False`). Any TF32 option on datacenter GPUs would have to be checked against
    the goldens. Sources: `src/fastforks/stereo/models/{crestereo,hitnet,igev_plusplus}/card.toml`.
  - DEFOM-Stereo runs its Depth Anything V2 branch in bf16 because the DPT refinenet1 activations exceed the fp16
    range, and TensorRT cannot build its bf16 deconvolution. Source:
    `src/fastforks/stereo/models/defom_stereo/kernels/inference.py` and `net/dpt.py`.

## Promptable segmentation: EfficientViT-SAM

All numbers below are from the RTX 5060 Ti (the tuning of `efficientvit_sam`, 2026-11-10); nothing here was run on
an A100, L40S or H100.

- **Batch of prompts.** The image embedding is cached, so many prompts on one image only repeat the prompt decoder.
  It is GPU-bound from about four prompts on: about 1.5 ms and 52 MiB reserved per prompt in float32 (16 prompts 24.7 ms,
  64 prompts 98.4 ms with 3.3 GB reserved). A CUDA graph only removes launch overhead (1 prompt 4.02 -> 1.46 ms,
  4 prompts 6.06 -> 5.41 ms, 64 prompts no change), so the family captures graphs up to 4 prompts
  (`GRAPH_PROMPTS` in `model.py`). On a faster GPU the crossover moves up; raise it after measuring there.
- **fp16 prompt decoder.** Not adopted: under autocast one prompt is slower (4.09 -> 6.03 ms), but 16 prompts take
  24.8 -> 18.2 ms and 64 prompts 98.7 -> 72.9 ms with 2342 -> 1837 MiB peak, at a mask-logit error of 0.0015 mean /
  0.006 p99 relative. Worth it for automatic mask generation (a 32 x 32 point grid = 1024 prompts) once the masks'
  thresholded IoU is checked on COCO.
- **Low-resolution masks for dense prompting.** The 256 x 256 logits are upscaled to the image size per prompt; for
  point grids, filter by `iou_scores` first and upscale only the kept masks. Not built.
- **Batch of images.** The encoder is convolutions plus linear attention with fixed shapes (512 or 1024 input), so it
  batches without changes to the kernels; the runner takes one image per call. On 80 GB cards the XL1 encoder (13 ms
  per image here, under 1 GB) is the candidate for a batched `embed`. Not built.
- **Whole-encoder graph.** Dropped on `kernels` (L0 2.11 -> 2.07 ms, XL1 11.99 -> 11.72 ms: the fused kernels leave
  few launches). It would halve the `portable` encoder (L0 7.0 -> 3.7 ms), but `portable` stays plain torch.
- **Wide 1x1 convolutions.** Fused linear 1x1 convolutions from 1024 input channels run through cuDNN with float32
  accumulation instead of the fp16 kernel: this is an accuracy fix (L2 mask-logit error against the golden 0.20 -> 0.016 relative mean; portable 0.013), not a
  speed choice, and applies on every GPU.
- **TensorRT.** Two separate causes, both found on the RTX 5060 Ti with TensorRT 11.4 (2026-11-10).
  (1) NaN: the ONNX exporter's rewriter (onnxscript 0.7.2) removed the scalar `+ 1e-15` of the linear attention's
  division as a no-op, so 0 / 0 gave NaN in the engine and in ONNX Runtime alike (not an fp16 overflow). Fixed in the
  family: the epsilon is a one-element tensor (`tensor_eps` in `fast.py`), torch output bit-identical. A general fix
  (keep tiny scalar constants, or check for them) would belong in `_core/export.py`; other families were not
  checked for the pattern (`mvsanywhere` also records an all-NaN engine, cause not verified).
  (2) Accuracy: the fp16 engine is then finite but outside the golden tolerance for all five variants (mask logits,
  relative mean, tolerance in brackets: L0 0.012 (0.015, but `iou_scores` 0.017 against 0.0037), L1 0.053 (0.017),
  L2 0.165 (0.025), XL0 0.014 (0.0081), XL1 0.018 (0.0052)). The cause is the same as for the `kernels` mode: fp16
  accumulation in the wide 1x1 convolutions. With the 1x1 convolutions from 1024 input channels cast to float32 in
  the ONNX graph (15 to 57 convolutions) L1, L2 and XL0 pass and L0 and XL1 miss by under 10 %; from 256 channels L0,
  L2 and XL0 pass and L1 and XL1 miss on `iou_scores` by about 13 %. Engine time is that of the fp16 engine (L0 2.4 ms,
  XL1 17.4 ms at the golden size). A float32 engine passes for all five (mean 0.0006 to 0.008) and is what
  `speed_compile.jsonl` records: 5.6 ms (L0) to 38.4 ms (XL1) for the encoder at 1024 x 1024, slower than `kernels`
  for the whole call (4.4 to 15.3 ms). A mixed-precision engine needs an export option that marks layers as float32
  and a golden check of the engine in `tools/speed_check.py` (it only checks for NaN today). Not built.

Source: `src/fastforks/promptable_segmentation/models/efficientvit_sam/` (`model.py`, `fast.py`, `card.toml`).

## Promptable segmentation: HQ-SAM

All numbers are from the RTX 5060 Ti (2026-11-10); nothing here was run on a datacenter GPU.

- **Same encoder as SAM.** The ViT encoder is SAM's and takes 92 to 98 % of the call (ViT-B 34 of 37 ms, ViT-H 141
  of 144 ms on `kernels`), so the family follows SAM for the encoder; a batched `embed` for 80 GB cards is not built
  (one image per call). The HQ head adds about 1.2 ms to the embedding: two small transposed-convolution stacks, on
  the first global-attention block's output and on the final embedding, computed once per image and cached.
- **Prompt decoder graph.** Kept: a CUDA graph per prompt layout takes one prompt from 4.4 to 2.6 ms (a box 4.7 ->
  2.8 ms); the decoder is launch-bound at one prompt. The bound of 4 prompts (`GRAPH_PROMPTS`) is EfficientViT-SAM's
  measurement of the same decoder, not re-measured with the HQ convolutions; in steady state the graphs cost about 100 MiB
  allocated and 190 MiB reserved, and a capture doubles the reserved peak of that call (4 prompts: 2.3 against
  1.2 GB). In a workload that mixes prompt counts the reserved pool grows by 3 to 4 GB (see below). On a faster GPU
  the crossover moves up.
- **Many prompts.** The embedding and the HQ feature are cached, prompts only repeat the decoder. The decoder is
  memory-heavy per prompt, as upstream: it upscales the embedding and runs the HQ convolutions at 256 x 256 per prompt
  in float32. Measured on ViT-B with `kernels`: 4 point prompts 1.2 GB reserved, 20 prompts 3.7 GB, 60 prompts
  10.0 GB (about 165 MiB per prompt, transient; 250 MiB stay allocated). On COCO (up to 52 instances per image, one
  call per image) the peak is 4.5 GB allocated / 10.4 GB reserved on `portable`, 4.6 / 14.7 GB on `kernels` (the
  graphs' pools and the capture add to the reserved pool) and 4.7 / 8.6 GB for the original. Decoding the prompts in
  chunks would bound this and is the first thing to build for automatic mask generation; an fp16 decoder was not
  tried. Not built.
- **TensorRT.** The fp16 engine of the encoder (with the HQ feature output) builds and matches the golden through the
  model's own prompt path (mask-logit error relative mean B 0.0012, L 0.0017, H 0.0009), 28 / 68 / 125 ms.

Source: `src/fastforks/promptable_segmentation/models/sam_hq/` (`model.py`, `fast.py`, `fused.py`, `card.toml`).

## Promptable segmentation: MobileSAM

All numbers are from the RTX 5060 Ti (2026-11-10); nothing here was run on a datacenter GPU.

- **Launch-bound encoder.** The TinyViT-5M encoder takes 6.5 ms on `portable` and 5.1 ms on `kernels` (fp16,
  1024 x 1024), torch.compile 4.8 ms and the TensorRT fp16 engine 5.0 ms; the full call with one point is 10.5 /
  6.9 / 6.2 ms. With 5 M parameters the encoder is small kernels and launches, so a faster GPU gains little per image.
  Not built: a CUDA graph over the whole encoder (the prompt decoder already replays one, as EfficientViT-SAM) and a
  batched `embed` (several images per call), which would fill an 80 GB card; both are the first things to try there.
- **Prompt side.** The prompt encoder and mask decoder are SAM's (`shared/sam_prompt`), with EfficientViT-SAM's CUDA
  graph per prompt layout (up to 4 prompts). On COCO the prompts cost 8.5 ms per image (all instances) against
  5.6 ms for the embedding on `kernels`, so for many prompts the decoder dominates; chunked or fp16 decoding is
  not tried.
- **Memory.** 183 MiB allocated at 1920 x 1080 on `portable`; the COCO peak (up to 52 prompts per image) is 1.9 GB
  allocated / 4.0 GB reserved on `kernels`, set by the decoder.

Source: `src/fastforks/promptable_segmentation/models/mobile_sam/` (`model.py`, `fast.py`, `card.toml`).

## Cross-cutting

- **Batching.** The `kernels` paths are tuned for batch 1, which is launch-bound on datacenter GPUs. This is the most
  repeated note (ConvNeXt, RTMDet, Depth, RAFT). A graph per batch size, and re-measured thresholds, are the usual
  follow-up. Nothing is measured on a datacenter GPU.
- **TF32.** Noted as no gain over fp16 (ConvNeXt, RTMDet) and as shifting outputs (RTMDet boxes, stereo
  disparities); the library's own fold check disables it (`src/fastforks/_optimize/fold.py`). A100 is the card named
  for it; untested.
- **bf16.** No fused conv kernels exist for it (ConvNeXt, RTMDet). Suggested for A100 for the DPT models; untested.
- **Larger CUDA graphs.** Already done for RTMDet (whole network plus decoding); suggested for the whole call in
  Depth; for RAFT a graph pays only where the GPU outruns the host's launches.
- **FP8 / INT8 via TensorRT.** FP8 GEMMs on sm_89 / sm_90 and newer are suggested for Depth and RAFT encoders;
  RTMDet and ConvNeXt say FP8 / INT8 belong to a TensorRT export, because they need calibration and move accuracy.
- **Half-rate vs full-rate fp16 accumulation.** On GeForce, fp32 accumulation runs at half the fp16-accumulate rate,
  so our fp16-accumulate kernels gain most there; datacenter and workstation cards run both at full rate. The L4
  cuBLAS test showed 48-50 TFLOPS with fp32 accumulation against 48-52 with fp16 (error 0.009 against 0.149 vs an
  fp32 reference). Consequences already shipped: tanh GELU and larger tiles on full-rate cards. Suggested, not
  built: an `fp32_accumulate` default per device for the feature backbones, and fp32-accumulate variants of the
  `dpt_fused` tiles. Source: `full-rate-gpus.md`, `src/fastforks/_core/kernels.py`.
- **Hopper / Blackwell.** All kernel libraries ship sm_90 and sm_100 binaries, but they use `mma.sync`, not wgmma
  or TMA. cuBLASLt or CUTLASS 3.x for those cards is the open item in `full-rate-gpus.md`.
- **Where tuned numbers are stored.** Per-family notes live in card comments (`card.toml`) and in agent reports; this
  file is the collection point.

## Pose: RTMO

Measured on RTX 5060 Ti only; these suggestions are untested on datacenter GPUs. RTMO finds people without
a separate detector. Full-call timings include letterboxing, person filtering/NMS, and coordinate decoding.
The kernels backend optimizes the CSPDarknet backbone; neck and head remain torch. Try graph capture for
neck/head after measuring its share of total time. Check person counts around the 0.1 score threshold as well
as coordinates: fp16 score changes can change detection membership. Original-code rows include mmpose's CPU
OpenCV preprocessing and decode; compare full-call time and peak memory, not only backbone throughput.

## Classification: TinyViT

The CUDA path uses fused convolution kernels and replays a CUDA graph while batch x height x width is at most
1,048,576 pixels. This threshold is inherited from ConvNeXt, not tuned per device. The RTX 5060 Ti measurements
cover batch one; there are no datacenter measurements. On faster GPUs, first measure graph versus eager execution
across batch sizes before changing the threshold. The fixed crop makes a graph key a batch size; its four-entry
cache can retain memory for several batches, so report reserved as well as allocated memory.

Source: `src/fastforks/classification/models/tinyvit/model.py` and the family rows in `docs/data/speed.jsonl`.
## RTMW whole-body pose

The PAFPN neck and two-scale SimCC head increase memory traffic. Batch detected people into crop buckets,
then compare CUDA graphs and compiled/exported raw-network paths at the deployment's actual person count.
RTMW-X384 uses fp32-accumulating wide convolutions and depthwise kernels: half accumulation produced large
SimCC peak flips on confident joints. Retain this accuracy check when tuning tensor-core recipes on other GPUs.
The current recipe matches portable throughput on the 5060 Ti; larger GPUs may benefit from fused fp32
accumulation and epilogues. Accuracy rows evaluate the first 17 COCO body joints with GT boxes, without flip test;
whole-body/face/hand accuracy requires the corresponding datasets and is not claimed by those rows.

## RTMPose model zoo

Halpe, hand, face and animal checkpoints reuse the existing CSPNeXt/SimCC kernels. Batch the appropriate
object boxes and tune crop buckets for the deployment's distribution; hand/face/animal models use square crops.
Halpe26's COCO rows score only the first 17 body joints. Hand, face and AP-10K goldens establish upstream
parity on the fixture, not task accuracy on unstaged datasets. The S-Halpe fixture contains an out-of-frame
knee with competing edge peaks; preserve real-image accuracy comparisons when changing accumulation precision.

For S-Halpe26, knee 13 in the first (upper-body) golden crop has y-bin 511 at the crop edge, with bin 460
only 0.51% below it in upstream fp32. Folded fp16 swaps those peaks; the point is outside the observed body.
Measured portable mean/p99 is 1.322/25.604 px and kernels 1.413/27.364 px. Its initial per-mode bounds are
1.5 times those measurements rounded up to two significant figures, without changing the family bounds.
On the same 500 COCO images, original fp32/fp16 score 71.44/71.30 AP and portable/kernels 71.36/71.24.
No golden hash, input or output was changed for this calibration.


## Classification: SHViT

CUDA measurements cover portable float16 at batch one on the RTX 5060 Ti, with native 224 crops (S4: 256).
No dedicated kernels runner or datacenter measurements exist yet. Measure throughput at larger batches and profile
the convolution and single-head attention portions separately before choosing fusions or graph-cache limits.

Source: the SHViT rows in `docs/data/speed.jsonl` and `docs/data/speed_compile.jsonl`.

## HGNetV2

Only the RTX 5060 Ti has been measured. The portable path folds BatchNorm and uses channels-last fp16
convolutions; B0-B6 run at native 224. Measure larger batches on the target GPU before changing precision
or adding aggregation-block fusions. Compare the existing compile and TensorRT rows as well as originals,
and preserve the measured mean/p99 bounds 0.0080/0.033. No datacenter speed claim is made here.

## RF-DETR Keypoint Preview (2026-11-11)

Pinned Apache-2.0 preview, 40.7M parameters, 576 x 576 input. The original FP32 / authors native FP16
(inference with compile=False, dtype=float16, inplace=True) full-call times at the standard 1080p scene are
29.45 / 19.28 ms; portable 18.278 ms; kernels 9.476 ms. The custom path packs DINOv2 QKV and uses
fastbb's fp32-accumulating flash attention, then replays the authors' export forward in a fixed-input CUDA graph.
Resizing/normalization and dynamic person filtering remain outside capture. Graph replay, changed input sizes,
output lifetime and explicit release are tested. No new binary or reduced-precision accumulation was introduced.

On full COCO val2017 (5000 images), portable scores 71.82 keypoint AP and kernels 71.79.
On the same every-tenth-image subset (500), original FP32/native FP16 score 72.69/72.89, portable 72.87
and kernels 72.81. All tiers evaluate top 100 query/class pairs, retain the person class, use threshold zero,
no NMS or GT boxes, and preserve uncertainty-fused person confidence and per-joint findability.

The original CPU FP32 eval fixture passes at atol 1e-4. Initial mean/p99 keypoint drift is
0.0479/0.1108 px portable and 0.0792/0.2130 px kernels; per-mode bounds are 1.5x rounded up to two
significant figures. During initial fixture creation, the reference helper was corrected to call eval():
the public upstream predictor does this itself; its bare constructor stays in training mode and activates
13 query groups. Only the new uncommitted reference fixture was corrected, with the first fixture retained
in scratch for audit. No existing golden or revision changed.

torch.compile measures 7.995 ms for the full portable call; TensorRT measures 4.283 ms for the raw
576 x 576 module, including resize/normalization but excluding confidence fusion and person filtering.
TensorRT raw query slots may reorder: compare decoded people rather than equal array indices.
A decoded single-person fixture gives score 0.77647 vs 0.77667 and at most 0.2813 px drift on confident
joints at 576 px; full-dataset TensorRT accuracy has not been measured.

Vendored computation stays pinned. Local compatibility edits remove unused training builders and import-time
JIT aliases, use cycle-free positional/export method overrides, and keep optimized parameter-only paths
from bypassing fastforks class overrides. Transformers and torchvision remain explicit optional dependencies.
## StarNet

StarNet selects CNN optimization explicitly: timm calls its convolutional units `Block`, a name also used by
transformers. The portable runner folds BatchNorm; kernels reuse the existing convolution fusions and bounded
small-batch CUDA graph cache. On another GPU, measure batch scaling across the shared HACC pixel threshold,
convolution epilogues and depthwise kernels before changing dispatch. RTX 5060 Ti measurements do not establish
datacenter performance.
## EfficientFormerV2

RTX 5060 Ti measurements cover the native 224 crop. Profile feature-map attention, learned bias tables,
strided depthwise convolutions and BatchNorm folding on the target GPU before proposing new fusions.
The fp16 golden errors are larger than for the simpler CNNs, so compare dataset accuracy and the original
FP32/FP16 results before changing precision. Compile/TensorRT rows are separate measured alternatives;
no datacenter performance is inferred from the consumer GPU.
## EfficientFormer

RTX 5060 Ti measurements cover the native 224 crop. The shared folding pass clears FX proxies left in
forward attention-bias caches so later traces cannot reuse nodes from a different graph; real tensor
cache entries remain intact. Profile pooling/MLP blocks and the final attention blocks on the target GPU
before adding kernels. Compare original FP32/autocast FP16, portable, compile and TensorRT rows without
inferring datacenter performance from this consumer GPU.

## CoAtNet

Validate the native 224 and 384 crops separately: convolution and attention proportions vary across
timm rw variants. The RTX 5060 Ti portable FP16 goldens have worst mean/p99 absolute errors
0.00157/0.0068, with bounds 0.0024/0.011. Profile MBConv and relative-position attention on the target
GPU before adding a dedicated kernels runner; no datacenter performance is inferred here.
## FocalNet

Profile focal modulation, depthwise convolutions and gated sums separately on the target GPU.
Compare small and large receptive-field variants at their native 224 crop before choosing fusions;
no dedicated kernels runner or datacenter measurements exist yet.
## DaViT

Profile spatial window attention and channel attention separately on the target GPU, including
the depthwise position convolutions. Check batching and layout conversion costs at the native 224 crop
before choosing fusions; no dedicated kernels runner or datacenter measurements exist yet.
## SegNeXt

Profile strip convolutions and the matrix-factorisation decoder separately at 512x512 and 1024x2048.
Keep the documented seeded NMF basis and matching preprocessing when comparing the original and portable
paths. No dedicated kernels runner or datacenter measurements exist yet.

The large variant has FP16 golden logit mean/p99 error 0.0871/0.312; measured bounds are
0.14/0.47. Keeping only the NMF bases or decoder in FP32 barely changes this error; full FP32
reduces it to 0.002718/0.010204. Across 500 ADE20K images per variant, portable FP16 mIoU
differs from the seeded original FP32 by at most 0.01 percentage points (large: both 48.88).
## Flow: NeuFlow v2

- The portable implementation uses autocast convolutions and cross-attention with float32 matching, correlation
  and upsampling. There is no custom-kernel path: the existing kernels do not implement its LeakyReLU(0.1)
  convolutions and single-level radius-four lookup with 128 channels.
- For datacenter GPUs, measure batching before adding launch optimizations. A fused correlation lookup and
  convolution epilogues with the correct LeakyReLU slope are candidates, requiring fresh golden and accuracy
  checks. No datacenter measurements or custom-kernel speed claims are made here.

NeuFlow v2's explicit singleton-head SDPA fixes the ONNX translator's 3D-query rejection. CPU and GPU A/B outputs are identical on the measured inputs; warm portable speed does not improve. All three 1080p TensorRT engines now build and run (about 65 ms, 8532 MiB total device memory); numerical engine parity is checked separately. The torch allocator alone excludes TensorRT's memory, so use the rows' device_mb.

TensorRT numerical check on the stored 384x512 golden inputs (all outputs finite):

| checkpoint | mean error vs CPU (px) | p99 (px) | max (px) |
|---|---:|---:|---:|
| neuflow-v2 | 0.0018181 | 0.0078912 | 0.0347252 |
| neuflow-v2-sintel | 0.0016701 | 0.0075026 | 0.0180321 |
| neuflow-v2-things | 0.0017063 | 0.0072393 | 0.0194032 |

These are TensorRT measurements; the portable golden bounds remain 0.0014 mean / 0.0067 p99. Re-running portable KITTI2015 after the attention-shape fix reproduces all three pre-fix EPE and Fl-all scores exactly.

## Flow: GMFlow

- Portable CUDA uses autocast for the CNN, Transformer, upsampling and recurrent refinement while matching,
  propagation, warping and flow stay float32. Global matching and propagation use row chunks to bound attention
  matrices at 0.5 GiB; original fp32/fp16 rows must retain their actual precision labels.
- Profile those chunked global attentions before replacing them. An online-softmax implementation that directly
  reduces weighted coordinates/flow could reduce intermediate memory, but needs measured parity and end-to-end
  speed before adoption. Local correlation/warping is another candidate for the six-refinement variants.
- Initial 1080p scale-one measurements show portable about 800 ms, compile about 382 ms and TensorRT about
  187 ms. Scale-two TensorRT attempts currently fail in the builder's Myelin CHECK(is_const()); preserve
  actual failure rows and do not substitute another backend.
- Compilation roughly halves scale-two latency, but peak allocation rises from 6433–6443 MiB to 10865–10875 MiB;
  compare latency and memory together. No datacenter or custom-kernel measurements have been made for this family.
