# WAFT (flow) kernels mode, 2026-11-15

Commit: 87c684d5 + this change (branch work/ports). GPU: RTX 5060 Ti, torch 2.14.1+cu130, kernels igev / cutlass_conv /
cudnn_hacc / fastbb / dpt_fused (local builds). Rows: `results/raw/2026-11-15_kernel_audit/waft_rows.jsonl`
(`tools/speed_check.py worker`, first call + 3 warm-up + peak reset, median of 3 x 5, synthetic pair, iters=5).

Change: `flow/models/waft/kernels/inference.py` (`prepare_kernels`, `forward_kernels`) and a `Kernels` runner; the card's
`status = "none"` is replaced by `kernels = [...]`. Depth Anything ViT-S as the fp16 fastbb ViT (`use_fast_vit`), every
conv through `cnn_kernels.conv_ep` (fp16 NHWC), `resconv` blocks as GELU + conv/bias/GELU + conv/bias/residual; flow
state, warp coordinates, grid sampling and the softmax upsampling stay fp32; only the last iteration is upsampled.
Portable is unchanged.

| input | original fp32 / fp16 ms | portable ms | kernels ms | kernels vs portable | portable peak alloc MB | kernels peak alloc MB |
|---|---|---|---|---|---|---|
| 375x1242 | 378 / 225 | 154.7 | 91.1 | 1.70x | 1148 | 604 |
| 1080x1920 | 2866 / 1533 | 790.8 | 518.7 | 1.52x | 3553 | 1851 |

First call 0.21 s / 0.52 s (portable 0.35 / 0.79 s). Accuracy: the three GPU goldens (`waft-a1`, `-sintel`, `-kitti`,
upstream fp32 CPU reference, `pytest -m gpu`) pass with a kernels bound `gpu_mean_kernels = 0.008` px (measured mean
|diff| 0.0056 px on the kitti golden; portable fp16 0.0015 px), `gpu_p99_kernels = 0.04`. Goldens stay outside git.
