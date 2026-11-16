# DeDoDe kernels (VGG19 detector + ConvRefiner on fp16 NHWC conv_ep; DINOv2 stays fp16 as portable) vs portable (2026-11-15)
Command: tools/eval_matching.py match|score (gpulock-low), RTX 5060 Ti, torch 2.14.1+cu130, base f1198b34 + this change.

| ref (MegaDepth-1500) | mode | AUC@5 / @10 / @20 | ms/img | peak alloc MB | peak reserved MB |
|---|---|---|---|---|---|
| dedode-v2-g | portable | 53.21 / 69.63 / 81.54 | 378.8 | 1890 | 3016 |
| dedode-v2-g | kernels | 53.01 / 69.64 / 81.65 | 331.7 | 1828 | 2940 |

kernels: 12 % faster, peak -3 %, AUC@5 -0.20 (within the 0.3 gate; AUC@10/20 +0.0/+0.1).
dedode-b (300 pairs only, noise about 1): within 0.07 of portable. Local GPU golden: none recorded for dedode (no file in golden/).
The memory gain is small: the peak is in the DINOv2 descriptor and the matcher, which stay as portable.
