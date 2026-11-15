# MatAnyone portable vs original (2026-11-15)

Collected from existing rows (no new measurement; RTX 5060 Ti, torch 2.14.1+cu130). Sources: `docs/data/speed.jsonl` (1080x1920 video, rows at 6ec60fba) and `docs/data/matting_eval.jsonl` (YouTubeMatte 512x288 every 2nd clip, 3200 frames, `python tools/eval_matting.py matanyone --dataset youtubematte --tiers ...`, commits e97998ab / 706c6828).

| input | tier | ms | peak alloc MB | peak reserved MB | MAD |
|---|---|---|---|---|---|
| 1080x1920 | original fp32 | 1736.3 | 3810 | 4104 | |
| 1080x1920 | original fp16 | 1219.9 | 2978 | 3208 | |
| 1080x1920 | portable fp16 | 1205.2 | 2719 | 2908 | |
| 512x288 per frame | original fp32 | 19.42 | 574 | 692 | 2.623 |
| 512x288 per frame | original fp16 | 25.13 | 475 | 566 | 2.636 |
| 512x288 per frame | portable fp16 | 20.49 | 434 | 472 | 2.636 |
| 512x288 per frame | portable fp32 | 16.77 | 549 | 632 | 2.623 |

Verdict: portable is faster and smaller than the original at 1080p with the original's accuracy. At 512x288 it is faster than original fp16 but 5 % slower than original fp32 (fp16 autocast overhead on tiny frames); memory is lower in all rows. No portable change; kernels left as is (scope: no new kernels).
