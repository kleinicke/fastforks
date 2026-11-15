# LoMa-B portable vs original (2026-11-15)
Commit db0c625e, GPU RTX 5060 Ti 16 GB, one MegaDepth pair, bench via cpuslot + gpulock-low (scratchpad bench_loma.py, modes orig16/orig32).

| variant | ms | peak allocated MiB | peak reserved MiB | matches |
|---|---|---|---|---|
| original fp32 | 409.1 | 1843 | 3392 | 1039 |
| original fp16 | 382.8 | 1843 | 3398 | 1038 |
| portable (earlier measurement, loma-b speed row) | 308.1 | 1767 | n/a | - |

Portable is 1.24x faster than the original fp16 and 4 % lower in peak allocated memory. Match count agrees within 1. The portable run in this batch did not finish in the GPU queue, so the portable row is the earlier recorded measurement (compile mode 285.4 ms / 2870 MB).
