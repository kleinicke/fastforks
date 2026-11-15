# LightGlue kernels mode (aliked-lightglue, disk-lightglue), 2026-11-15

Commit: ecac19a2 + this change (branch work/ports). GPU RTX 5060 Ti, torch 2.14.1+cu130, igev / cutlass_conv /
cudnn_hacc local builds. Rows: `results/raw/2026-11-15_lightglue_kernels/` (speed: `tools/speed_check.py worker`, first
call + 3 warm-up + peak reset, median of 3 x 5, synthetic pair 768x1024; accuracy: `tools/eval_matching.py match|score
--mode kernels --every 5`, MegaDepth-1500 every 5th pair = 300 pairs, long side 1600, 2048 keypoints).

Change: `matching/models/lightglue/kernels/inference.py` + `Kernels` runner.
- DISK: the U-Net (InstanceNorm + PReLU + 5x5 conv blocks) runs on fp16 NHWC `cnn_kernels.conv_ep` (CUTLASS fused conv),
  input padded to 8 channels, in-place instance norm; detection (NMS, top-k), descriptors and LightGlue stay as portable.
  Describe per 1600 px image 54.8 -> 23.3 ms.
- ALIKED: dense network under fp16 autocast + channels-last (launch bound: CPU 53 ms vs CUDA 45.6 ms per call on a real
  pair; deformable convs, SELU and the DKD / SDDH heads have no epilogue code, so no deeper path was built).

| 768x1024 synthetic | ms | peak alloc MB | first call |
|---|---|---|---|
| aliked original (upstream) | 41.17 | 1305 | |
| aliked portable | 40.61 | 1300 | 0.86 s |
| aliked kernels | 35.05 | 1238 (reserved 1324) | 0.90 s |
| disk original (upstream) | 116.7 | 1955 | |
| disk portable | 116.46 | 1950 | 0.73 s |
| disk kernels | 52.11 | 736 (reserved 778) | 0.30 s |

Kernels vs portable: aliked 1.16x, 0.95x memory; disk 2.23x, 0.38x memory.
MegaDepth-1500 every5 on real pairs (ms per pair, peak alloc MB): disk kernels 120.5 ms / 2193 vs portable 276.8 / 6137;
aliked kernels 80.1 / 3825 vs 121.5 / 4027.

| MegaDepth-1500 every5, AUC 5 / 10 / 20 | portable | upstream | upstream fp16 | kernels |
|---|---|---|---|---|
| disk-lightglue | 45.59 / 63.28 / 77.33 | same | 44.91 / 63.40 | 44.95 / 62.94 / 76.58 |
| aliked-lightglue | 52.10 / 69.18 / 81.38 | 51.69 / 68.24 / 81.04 | 51.37 / 68.84 | 51.01 / 68.33 / 80.94 |

**Accuracy is NOT strictly equal to portable**: -0.64 (disk) and -1.09 (aliked) AUC@5 on 300 pairs. The disk loss equals
that of upstream's own fp16 (44.91); aliked is 0.36 below upstream fp16. The fp16 keypoint heatmap moves a few keypoints
(DISK, 30 pairs: recall within 1 px 0.983, matches found in both 0.965). The gate "accuracy equal to portable" is
therefore open: decision for the coordinator (accept fp16 as a documented trade-off, or keep the heatmap in fp32).
