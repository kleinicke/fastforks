# LoFTR kernels (ResNetFPN backbone on fp16 NHWC conv_ep) vs portable, loftr-outdoor, 1500 MegaDepth pairs (2026-11-15)
Command: tools/eval_matching.py match|score (gpulock-low), RTX 5060 Ti, torch 2.14.1+cu130, base f1198b34 + this change.

| mode | AUC@5 / @10 / @20 | ms/img | peak alloc MB | peak reserved MB |
|---|---|---|---|---|
| portable | 50.60 / 67.58 / 80.31 | 73.9 | 905 | 3696 |
| kernels | 50.94 / 68.14 / 80.58 | 60.5 | 905 | 3764 |

Faster by 18 %, AUC@5 +0.34 (inside noise). Peak memory is equal (allocated) / +2 % (reserved): the peak sits in the
transformer / coarse matching on the 1/8 features, not in the backbone convolutions, so convolution kernels cannot lower it.
Local GPU golden: none recorded for loftr (no file in golden/).
