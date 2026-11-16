# LightGlue kernels vs portable, full 1500 MegaDepth pairs (2026-11-15)
Command: tools/eval_matching.py match|score (chunks under gpulock-low), RTX 5060 Ti, base df0154ce + this change.

| variant | AUC@5 / @10 / @20 | ms/img | peak MB |
|---|---|---|---|
| ALIKED portable | 50.41 / 67.49 / 80.30 | 101.1 | 4033 |
| ALIKED kernels fp16 (before) | 49.07 / 66.07 / 79.16 | 79.7 | 3831 |
| ALIKED kernels, fp32 score head (shipped) | 50.93 / 67.79 / 80.45 | 82.4 | 3836 |
| DISK portable | 42.71 | - | - |
| DISK kernels (fp16 NHWC conv_ep) | 43.39 | 2.3x faster | 2.8x less |

Gate (AUC@5 within 0.3 of portable): met by both. Change: aliked.py runs the score head in float32.
GPU golden (local): aliked-lightglue portable+kernels and disk portable pass; disk-lightglue kernels fails the
keypoint recall check (0.9917 < 0.9971). The DISK kernels path is unchanged by this commit (the golden predates the
fp16 conv_ep path); task accuracy on 1500 pairs is above portable, so the golden tolerance is not re-recorded here.
