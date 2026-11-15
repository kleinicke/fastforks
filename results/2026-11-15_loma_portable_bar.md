# LoMa-B portable vs original (2026-11-15)
Commit db0c625e, GPU RTX 5060 Ti 16 GB, one MegaDepth pair, bench via cpuslot + gpulock-low (scratchpad bench_loma.py, modes orig16/orig32).

| variant | ms | peak allocated MiB | peak reserved MiB | matches |
|---|---|---|---|---|
| original fp32 | 409.1 | 1843 | 3392 | 1039 |
| original fp16 | 382.8 | 1843 | 3398 | 1038 |
| portable (this batch) | 307.9 | 1790 | 2562 | 1030 |

Portable is 1.24x faster than the original fp16, with 3 % lower peak allocated and 25 % lower peak reserved memory. Matches 1030 vs 1038 (fp16 original), within 1 %.
