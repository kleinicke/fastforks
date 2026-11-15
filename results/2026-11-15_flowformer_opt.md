# flowformer-small portable: time and memory reduction, exact accuracy
Date 2026-11-15, GPU RTX 5060 Ti 16 GB, base commit a3c3a18a (+ uncommitted-at-measure changes, committed right after), `tools/eval_flow.py`, KITTI 2015, 200 pairs, 375x1242.
Raw: results/raw/2026-11-15_flowformer_opt.jsonl

Changes: decoder upsamples only the last iteration's flow; PatchEmbed over the 7332 cost maps in chunks of 1024; SepConvGRU convzr fused.

| ref | EPE now | EPE original fp32 | before ms / MiB | now ms / MiB | original fp32 / fp16 |
|---|---|---|---|---|---|
| flowformer-small-things | 6.8841 | 6.8841 | 351 / 3165 | 338 / 1410 | 303 ms / 2382 MiB (fp16) |
| flowformer-small-chairs | 10.9521 | 10.9521 | | 338 / 1410 | |
| flowformer-small-sintel | 4.6574 | 4.6574 | | 338 / 1410 | |

Peak memory is 41 % lower than the original fp16. Time still misses the fp16 original (303 ms): cost is cuDNN TF32 convs plus the fp32 aggregation matmul; fp16 aggregation gave NaN at scale 1024 and worse EPE, SDPA was slower and changed EPE. Kernels: none, same reason (time is in cuDNN convs and matmuls that already run on tensor cores; no fused-kernel gap found).
