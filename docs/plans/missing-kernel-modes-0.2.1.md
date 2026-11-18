# Families without a kernels mode

On 2026-11-18, all 578 registered variants across 205 model families passed their kernel-mode reference test on an RTX 5060 Ti (16 GB), with required golden files and the tolerances in the development branch.
Grounding DINO Base passed with the corrected upstream checkpoint heads and the pinned reference fixture.

The primary RTX 5060 Ti timing tables contain 427 variants with matched kernels and `torch.compile` measurements.
Another 73 have kernel timings without a matching compile workload, and 78 still need a kernel timing row.
Run `python tools/kernel_gaps.py` to see each variant's declared libraries, measurement status, GPU, dtype,
workload and recorded dates. The measurements below describe the code and workloads tested at the time.

The 2026-11-17 website snapshot listed 25 families and 73 variants. The historical sections record why those
modes were absent and the measurements used when adding them.

## RTX 5060 Ti follow-up

All 73 variants from the original list and all five additional LoMa variants passed their upstream reference checks.

The first three edge variants were measured separately. The table shows warm end-to-end latency on a 1920 × 1080 CUDA uint8 input (batch one, five warm-ups, median of 15 synchronized calls). Memory is peak allocated CUDA tensor memory during the timed calls. The portable and kernels modes passed the repository's upstream golden checks.

| Variant | Portable | Kernels | Speedup | Peak MiB, portable → kernels |
|---|---:|---:|---:|---:|
| HED | 10.87 ms | 8.16 ms | 1.33× | 194 → 200 |
| Lineart | 19.89 ms | 17.63 ms | 1.13× | 228 → 224 |
| Lineart Coarse | 19.90 ms | 17.63 ms | 1.13× | 228 → 224 |

At 512 × 512, HED improved from 5.52 to 4.15 ms. The two Lineart variants gained about 0.24 ms each at that size. The experimental Lineart Anime path regressed from 3.33 to 4.10 ms at 512 × 512, in the convolution experiment. The subsequent resampling results appear below.

### SAM 3 and DINOv3

These two families already had kernels modes. The table compares them before and after the change on the same RTX 5060 Ti, using the original weights, PyTorch 2.10.0+cu130, batch one, six warm-ups and the median of 17 synchronized API calls. The SAM 3 input was the reference coin photograph with the prompt `coin`. All three modes returned 24 masks. The DINOv3 inputs were seeded square uint8 images at the specified evaluation size.

| Model and input | Portable | Previous kernels | New kernels |
|---|---:|---:|---:|
| SAM 3, 303 × 384 photo | 194.95 ms | 125.32 ms | 116.98 ms |
| DINOv3 ViT-H+, 224 × 224 | 13.97 ms | 8.63 ms | 8.58 ms |
| DINOv3 ViT-H+, 512 × 512 | 55.83 ms | 34.44 ms | 33.60 ms |
| DINOv3 ViT-H+, 1024 × 1024 | 262.70 ms | 159.44 ms | 150.15 ms |

SAM 3 fuses the residual additions and MLP activation into existing encoder GEMMs. On a synthetic 1008 × 1008 input its peak allocation fell from 1957 to 1928 MiB. The repository's reference-image checks cover text, exemplar boxes and combined prompts.

DINOv3 ViT-H+ fuses SwiGLU for 1024 or more token rows. The short-sequence path keeps its original activation. At 1024 px, peak allocation fell from 1814 to 1806 MiB. The final hidden states matched the previous kernels exactly on the 224 and 1024 px test inputs. At 512 px the mean absolute difference from previous kernels was 0.00055 and the maximum was 0.01782. Against portable fp16 the mean absolute difference at that size was 0.00057.

For evaluation, set `FASTFORKS_KERNEL_PATH` to the local `build/kernels` directory containing `swiglu/`, or run `python tools/build_kernels.py swiglu` on the CUDA build host first. The new binary is staged locally. The five new LoMa references are available through `FASTFORKS_GOLDEN_PATH` pointing to the local `build/goldens` directory.

### Classification and dense prediction

The classification runs used one 480 × 640 input, each model's own resize and crop, six warm-ups and the median of 21 synchronized calls. All 37 variants passed the upstream reference checks. CUDA graph replay accounts for much of the batch-one speedup on repeated shapes. CoAtNet and SHViT retain their original convolutions.

| Family | Variants | Warm speedup range |
|---|---:|---:|
| coatnet | 10 | 1.54–6.59× |
| davit | 3 | 6.91–8.59× |
| efficientformer | 3 | 5.60–6.82× |
| efficientformerv2 | 4 | 6.61–8.11× |
| focalnet | 6 | 4.72–5.52× |
| hgnetv2 | 7 | 6.28–8.35× |
| shvit | 4 | 9.08–11.15× |

The following runs used the public inference API, 768 × 1024 inputs, three warm-ups and nine synchronized calls on the RTX 5060 Ti. Fixed-resolution models resize internally. Each variant passed its existing GPU golden thresholds. These medians measure repeated-shape latency. For example, PP-OCRv5 Mobile uses more memory when replaying captured graphs.

| Variant | Portable (ms) | Kernels (ms) | Speedup |
|---|---:|---:|---:|
| anyline | 35.03 | 25.75 | 1.36× |
| bisenet-face-parsing | 3.63 | 1.92 | 1.89× |
| mlsd | 11.53 | 10.68 | 1.08× |
| teed | 3.75 | 3.42 | 1.10× |
| parseq | 9.40 | 4.31 | 2.18× |
| parseq-tiny | 7.87 | 2.53 | 3.11× |
| lineart-anime | 6.10 | 4.82 | 1.26× |
| pidinet | 9.72 | 9.46 | 1.03× |
| inspyrenet | 105.55 | 103.56 | 1.02× |
| inspyrenet-fast | 51.37 | 15.02 | 3.42× |
| matanyone | 467.85 | 396.53 | 1.18× |
| aliked-flexglue | 136.01 | 128.55 | 1.06× |
| xfeat | 9.54 | 9.18 | 1.04× |
| xfeat-lighterglue | 28.23 | 27.95 | 1.01× |
| ppocrv5-mobile | 103.70 | 100.28 | 1.03× |
| ppocrv5-server | 181.98 | 164.56 | 1.11× |
| loma-b | 321.53 | 308.01 | 1.04× |
| loma-b64 | 134.11 | 124.31 | 1.08× |
| loma-b128 | 155.20 | 141.12 | 1.10× |
| loma-g | 369.91 | 360.07 | 1.03× |
| loma-l | 330.57 | 318.46 | 1.04× |
| loma-r | 321.16 | 308.96 | 1.04× |

The LoMa variants share DeDoDe descriptor kernels. MatAnyone reuses the first frame's encoder features during recurrent warm-up. Lineart Anime and PiDiNet use fused OpenCV-compatible preprocessing. XFeat retains float32 convolution precision for keypoint selection. Build new CUDA libraries with `python tools/build_kernels.py LIBRARY` and point `FASTFORKS_KERNEL_PATH` at the resulting local `build/kernels` directory.

### Optical flow and open-vocabulary detection

At 768 × 1024, all nine GMFlow configurations gained from supplying the head dimension explicitly to CUDA scaled dot-product attention. GMFlow Scale 1 Things peak allocation fell from 1443 to 760 MiB. FlowFormer Small keeps the recurrent state and motion aggregation in float32 and runs the four recurrent gate convolutions in float16. The following medians use the same nine-call protocol as the dense prediction runs.

| Variant | Portable (ms) | Kernels (ms) | Speedup |
|---|---:|---:|---:|
| gmflow-scale1-things | 224.73 | 97.97 | 2.29× |
| gmflow-scale1-mixdata | 224.46 | 98.22 | 2.29× |
| gmflow-scale2-things | 504.25 | 246.20 | 2.05× |
| gmflow-scale2-sintel | 506.04 | 247.09 | 2.05× |
| gmflow-scale2-mixdata | 504.46 | 247.22 | 2.04× |
| gmflow-scale2-regrefine6-things | 674.88 | 419.74 | 1.61× |
| gmflow-scale2-regrefine6-sintelft | 676.99 | 419.76 | 1.61× |
| gmflow-scale2-regrefine6-kitti | 677.40 | 418.11 | 1.62× |
| gmflow-scale2-regrefine6-mixdata | 676.22 | 419.54 | 1.61× |
| flowformer-small-chairs | 680.77 | 659.14 | 1.03× |
| flowformer-small-things | 682.60 | 659.71 | 1.03× |
| flowformer-small-sintel | 682.72 | 676.71 | 1.01× |
| grounding-dino-tiny | 100.28 | 80.68 | 1.24× |
| mm-grounding-dino-tiny | 100.38 | 80.01 | 1.25× |
| mm-grounding-dino-base | 115.19 | 96.90 | 1.19× |

Grounding DINO uses one CUDA launch to sample feature levels and accumulate their attention weights. Both MM checkpoints and both standard checkpoints passed. The initial Base comparison on the coffee fixture exceeded the 0.25 px mean box tolerance. Restoring the upstream checkpoint heads and pinning the verified reference resolved the discrepancy with the existing tolerance.

## Original snapshot

## anyline (1)

Variants: `anyline`.

Recorded reason: the MTEED CNN plus a GPU blend, a few ms; no custom kernel measured to pay off.

## bisenet (1)

Variants: `bisenet-face-parsing`.

Recorded reason: measured slower: cudnn_hacc fp16-accumulate convolutions 2.77 ms vs 2.41 ms portable (512x512).

## coatnet (10)

Variants: `coatnet_nano_rw_224.sw_in1k`, `coatnet_rmlp_nano_rw_224.sw_in1k`, `coatnet_0_rw_224.sw_in1k`, `coatnet_1_rw_224.sw_in1k`, `coatnet_rmlp_1_rw_224.sw_in1k`, `coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k`, `coatnet_2_rw_224.sw_in12k_ft_in1k`, `coatnet_rmlp_2_rw_224.sw_in1k`, `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`, `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`.

Recorded reason: not implemented: MBConv stages plus attention with a relative position bias: needs its own runner.

## davit (3)

Variants: `davit_tiny.msft_in1k`, `davit_small.msft_in1k`, `davit_base.msft_in1k`.

Recorded reason: not implemented: window attention and channel attention with depthwise position convolutions: needs its own runner.

## efficientformer (3)

Variants: `efficientformer_l1.snap_dist_in1k`, `efficientformer_l3.snap_dist_in1k`, `efficientformer_l7.snap_dist_in1k`.

Recorded reason: not built yet: pooling blocks with Conv-BN MLPs and LayerNorm transformer blocks with learned attention bias tables need a dedicated kernels runner.

## efficientformerv2 (4)

Variants: `efficientformerv2_s0.snap_dist_in1k`, `efficientformerv2_s1.snap_dist_in1k`, `efficientformerv2_s2.snap_dist_in1k`, `efficientformerv2_l.snap_dist_in1k`.

Recorded reason: not built yet: Conv-BN blocks with depthwise 3x3, feature-map attention with learned bias tables and strided depthwise convolutions need a dedicated kernels runner.

## flexglue (1)

Variants: `aliked-flexglue`.

Recorded reason: not implemented: data-dependent keypoint and match counts. Portable at 768x1024, 2048 keypoints, MegaDepth pair: 129.6 ms against 152.9 ms for the original with its float16 option (1.18x), peak allocated 1799 vs 1801 MB (reserved 1962 vs 1950), 760 vs 759 matches.

## flowformer (3)

Variants: `flowformer-small-chairs`, `flowformer-small-things`, `flowformer-small-sintel`.

Recorded reason: Measured 2026-11-15: time is cuDNN TF32 convolutions plus the float32 aggregation matmul; fp16 aggregation (NaN at scale 1024, worse EPE) and SDPA (slower, EPE changed) were tried and dropped..

## focalnet (6)

Variants: `focalnet_tiny_srf.ms_in1k`, `focalnet_small_srf.ms_in1k`, `focalnet_base_srf.ms_in1k`, `focalnet_tiny_lrf.ms_in1k`, `focalnet_small_lrf.ms_in1k`, `focalnet_base_lrf.ms_in1k`.

Recorded reason: not implemented: LayerNorm blocks with depthwise convolutions and gated sums: needs its own runner.

## gmflow (9)

Variants: `gmflow-scale1-things`, `gmflow-scale1-mixdata`, `gmflow-scale2-things`, `gmflow-scale2-sintel`, `gmflow-scale2-mixdata`, `gmflow-scale2-regrefine6-things`, `gmflow-scale2-regrefine6-sintelft`, `gmflow-scale2-regrefine6-kitti`, `gmflow-scale2-regrefine6-mixdata`.

Recorded reason: after the fused attention portable is faster than the original and uses less memory at every tested size (e.g. scale1 1080x1920 556 vs 750 ms, 4496 vs 8360 MiB); the rest is the attention itself (55 % of the CUDA time at 1080p, already the memory-efficient / flash SDPA kernels) and float32 elementwise ops, and the convolutions are a small share, so no library in the manifests (cudnn_hacc speeds up convolutions) would change the total (measured 2026-11-15).

## grounding_dino (4)

Variants: `grounding-dino-tiny`, `grounding-dino-base`, `mm-grounding-dino-tiny`, `mm-grounding-dino-base`.

Recorded reason: not built: the portable model now samples per level without copying (deformable attention: grid_sample 11 % of the time, 2.4 ms/call of cat/stack copies removed, portable -20 % time and -26 % peak memory); the remaining cost is the grid_sample gather itself, and a fused multi-scale deformable-attention cubin (Triton prototype 12.85 -> 0.84 ms per call) is the next step but is not built; fp16-accumulate GEMMs gain nothing (tiny 112.5 vs 112.5 ms, base 127.6 vs 131.3 ms at 480x640).

## hed (1)

Variants: `hed`.

Recorded reason: a VGG-style CNN: cuDNN convolutions are the whole time; no custom kernel measured to pay off.

## hgnetv2 (7)

Variants: `hgnetv2_b0.ssld_stage2_ft_in1k`, `hgnetv2_b1.ssld_stage2_ft_in1k`, `hgnetv2_b2.ssld_stage2_ft_in1k`, `hgnetv2_b3.ssld_stage2_ft_in1k`, `hgnetv2_b4.ssld_stage2_ft_in1k`, `hgnetv2_b5.ssld_stage2_ft_in1k`, `hgnetv2_b6.ssld_stage2_ft_in1k`.

Recorded reason: not built yet: the portable CNN path is validated; HGNetV2 aggregation blocks need a dedicated kernels runner.

## inspyrenet (2)

Variants: `inspyrenet`, `inspyrenet-fast`.

Recorded reason: measured slower: cudnn_hacc convolutions 189.9 ms vs 159.4 ms portable (1080x1920, size 1024; upstream views their outputs, so channels-last needs copies); fp16-accumulate GEMMs alone 3 %.

## lineart (2)

Variants: `lineart`, `lineart-coarse`.

Recorded reason: a small conv generator: cuDNN convolutions and instance norm are the whole time; no custom kernel measured to pay off.

## lineart_anime (1)

Variants: `lineart-anime`.

Recorded reason: a pix2pix U-Net: cuDNN convolutions and instance norm are the whole time; no custom kernel measured to pay off.

## loma (1)

Variants: `loma-b`.

Recorded reason: not implemented: data-dependent keypoint and match counts. Portable is 307.9 ms against 382.8 ms for the original fp16 (1.24x) and 409.1 ms fp32, peak allocated 1790 vs 1843 MB (reserved 2562 vs 3398), matches 1030 vs 1038-1039.

## matanyone (1)

Variants: `matanyone`.

Recorded reason: not implemented: ResNet encoders plus a memory readout and small object transformer per frame; portable runs 1.2 % faster than the original fp16 at 1080x1920 (1205 vs 1220 ms) with 9 % lower peak allocated memory (2719 vs 2978 MB) and identical task accuracy (YouTubeMatte MAD 2.636 both); against the original fp32 at 512x288 portable fp16 is 5 % slower per frame (20.5 vs 19.4 ms) with 24 % less memory, so the speed bar holds at 1080p and against fp16 only.

## mlsd (1)

Variants: `mlsd`.

Recorded reason: a MobileNetV2 CNN (cuDNN); decoding is a few small ops; no custom kernel measured to pay off.

## parseq (2)

Variants: `parseq`, `parseq-tiny`.

Recorded reason: the 26-step decoding loop is launch-bound, not kernel-bound, and portable already replays it as one CUDA graph (`cuda_graphs`, exports unchanged: `module()` is the plain network): 12.6 -> 3.6 ms (parseq) and 11.6 -> 2.1 ms (parseq-tiny) vs the original 19.6 / 18.0 ms fp16, same logits and memory; SDPA / cuBLAS are already the compute and no library in the manifests helps, so a kernels mode would only repeat the graph (measured 2026-11-15).

## pidinet (1)

Variants: `pidinet`.

Recorded reason: the converted network is plain convolutions (cuDNN); no custom kernel measured to pay off.

## ppocrv5 (2)

Variants: `ppocrv5-server`, `ppocrv5-mobile`.

Recorded reason: cuDNN convolutions are the networks; the time outside them is the per-line crop / resize and the CPU box geometry.

## shvit (4)

Variants: `shvit_s1.in1k`, `shvit_s2.in1k`, `shvit_s3.in1k`, `shvit_s4.in1k`.

Recorded reason: not built yet: convolutions with BatchNorm plus single-head attention on feature maps need a dedicated kernels runner.

## teed (1)

Variants: `teed`.

Recorded reason: a 58 k-parameter CNN that runs in a few ms; no custom kernel measured to pay off.

## xfeat (2)

Variants: `xfeat`, `xfeat-lighterglue`.

Recorded reason: not implemented: data-dependent keypoint counts (NMS, top-k, match filtering); the extractor is a small CNN.


## LoMa and RoMa v2 follow-up, November 18

RTX 5060 Ti 16 GB, PyTorch 2.10.0+cu130. Each row compares the kernels runner in `36a20781` with the revised runner on the same device. The input was a seeded pair of 768 × 1024 RGB images on CUDA, with the model’s default preprocessing and keypoint budget. Each timing is the median of seven synchronized calls after two warmups. Memory is peak allocated by PyTorch.

| Model | Portable (ms) | Previous kernels (ms) | Revised kernels (ms) | Portable (MiB) | Previous kernels (MiB) | Revised kernels (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| LoMa-B | 315.2 | 300.2 | 280.0 | 1752 | 1729 | 1433 |
| LoMa-B64 | 132.6 | 122.5 | 117.2 | 836 | 824 | 823 |
| LoMa-B128 | 153.0 | 138.9 | 127.8 | 861 | 838 | 836 |
| LoMa-L | 324.7 | 310.4 | 289.7 | 1892 | 1868 | 1572 |
| LoMa-G | 362.6 | 347.9 | 327.1 | 2431 | 2406 | 2109 |
| LoMa-R | 315.1 | 299.9 | 279.8 | 1752 | 1729 | 1433 |
| RoMa v2 | 520.0 | 343.8 | 342.0 | 1995 | 1913 | 1911 |

LoMa now projects the finest descriptor at the four neighbours of each detected keypoint and samples the preceding scale at those locations. RoMa v2 reuses rotary frequencies within each transformer forward and reuses the forward cosine similarities for reverse matching. All seven runners passed their existing GPU reference checks in both portable and kernels modes. The RoMa latency difference was about 0.5% on this workload.
