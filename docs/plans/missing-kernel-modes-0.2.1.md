# Families without a kernels mode

Snapshot of generated local website data on 2026-11-17: 25 families, 73 variants. These are family-level reasons from the model cards, not new independent validation of the claims. The list describes absent modes, not missing individual kernels inside existing modes.

## RTX 5060 Ti follow-up

Three variants from the snapshot now have a kernels mode. The table shows warm end-to-end latency on a 1920 × 1080 CUDA uint8 input (batch one, five warm-ups, median of 15 synchronized calls). Memory is peak allocated CUDA tensor memory during the timed calls. The portable and kernels modes passed the repository's upstream golden checks.

| Variant | Portable | Kernels | Speedup | Peak MiB, portable → kernels |
|---|---:|---:|---:|---:|
| HED | 10.87 ms | 8.16 ms | 1.33× | 194 → 200 |
| Lineart | 19.89 ms | 17.63 ms | 1.13× | 228 → 224 |
| Lineart Coarse | 19.90 ms | 17.63 ms | 1.13× | 228 → 224 |

At 512 × 512, HED improved from 5.52 to 4.15 ms. The two Lineart variants gained about 0.24 ms each at that size. The experimental Lineart Anime path regressed from 3.33 to 4.10 ms at 512 × 512, so that variant retains its portable implementation.

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

