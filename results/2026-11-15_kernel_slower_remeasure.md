# Kernels vs portable re-measure and parseq graph test (2026-11-15)

Command: `tools/speed_check.py run --only REF` under gpulock-low (first call, 3 warm-up, median of 3 rounds x 5),
RTX 5060 Ti, torch 2.14.1+cu130, raw rows `results/raw/2026-11-15_kernel_slower/rows.jsonl`.

| ref | portable ms / MB | kernels ms / MB | verdict |
|---|---|---|---|
| geocalib-pinhole | 81.18 / 320 | 81.92 / 277 | tie (0.9 %), less memory |
| wd-vit-tagger-v3 | 4.00 / 235 | 4.06 / 234 | tie (1.5 %, noise) |
| depthsplat-small-re10k | 118.6 / 5432 | 111.7 / 5434 | kernels 5.8 % faster |
| rtmw-x-384 | 50.72 / 466 | 50.93 / 496 | tie |
| dat-light-x4 | 267.6 / 1535 | 268.1 / 1535 | tie (0.2 %) |
| restormer-sidd | 181.3 / 850 | 178.5 / 850 | kernels faster than portable and than the original 217.95 ms (audit's 224.3 was noise/load) |
| neuflow-v2-things | rows mix two sizes, no valid kernels row (rerun open) | | open |

The audit's "kernels slower" gaps were GPU-sharing noise (other agents' jobs run concurrently); none reproduces.
Parseq (ws/parseq_bench.py, A/B eager vs CUDA graph): parseq 29.8 -> 3.4 ms, parseq-tiny 29.8 -> 1.9 ms, logits equal
(max diff 0.008, argmax equal). Graph only, so the card keeps `kernels status=none` with this measured reason; a
portable-mode graph is the follow-up.
