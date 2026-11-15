# CodeFormer kernels mode, 2026-11-15

Commit: bac5abe5/8f9c75c0 + this change (branch work/ports). GPU: RTX 5060 Ti, torch 2.14.1+cu130, kernels igev
(epilogue) / cutlass_conv / cudnn_hacc (local builds). Row: `results/raw/2026-11-15_kernel_audit/codeformer_rows.jsonl`
(`tools/speed_check.py worker`, first call + 3 warm-up + peak reset, median of 3 x 5; aligned 512x512 face crop).

Change: `restoration/models/codeformer/kernels/inference.py` and a `Kernels` runner (fp16 only); the card's
`status = "none"` is replaced by `kernels = [...]`. fp16 NHWC weights/activations, every VQGAN conv through
`cnn_kernels.conv_ep`; `ResBlock` = GroupNorm + SiLU as one per-channel epilogue (group moments from fp32-accumulated
channel means / sums of squares, no fp32 copy of the activation), conv + bias, conv + bias + residual in the epilogue;
the 16 x 16 transformer, logits head, code lookup and AdaIN statistics as in portable (fp32 where portable is).
Portable is unchanged. First attempt with torch GroupNorm + SiLU was not faster (50.3 vs 48.1 ms: channels-last torch
elementwise kernels took 45 % of the time) and the first epilogue version used an fp32 copy per norm (peak 650 MB).

| 512x512 | ms | peak alloc MB | first call |
|---|---|---|---|
| original (fp32, upstream) | 85.7 | 959 | |
| portable fp16 | 48.6 | 512 | |
| kernels | 40.4 | 432 (reserved 496) | 0.23 s |

Kernels vs portable 1.20x, 0.84x the memory. Accuracy: the GPU golden (`codeformer@1`, upstream fp32 CPU reference,
`pytest -m gpu`, goldens outside git) passes with the family's tolerance; kernels vs portable on a real face crop: mean
|diff| 0.00066, p99 0.0045 (image range 0..1).
