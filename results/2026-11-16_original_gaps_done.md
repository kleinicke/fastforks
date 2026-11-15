# Original-time gaps closed (2026-11-16)

Commit base: work/ports + this change. GPU: RTX 5060 Ti 16 GB, torch 2.14.1+cu130. Command: `results/raw/2026-11-16_origgaps2/queue.sh`
(originals, `tools/speed_original.py`, 3 runs each) and `queue_kernels.sh` (`kernels_speed.py`, kernels rows). Rows: `docs/data/speed.jsonl`,
raw copies in `results/raw/2026-11-16_origgaps2/`. Input 994x1482 (depth), 480x640 (lingbot-map). Original = authors' code at the pinned commit, no xFormers.

| ref | original fp32 ms / peak MB | original fp16 (authors' flag) ms / MB | portable ms / MB |
|---|---|---|---|
| any2full-s | 47.5 / 646 | none | 12.4 / 274 |
| any2full-b | 108.3 / 1109 | none | 26.8 / 538 |
| any2full-l | 268.8 / 2436 | none | 67.7 / 1178 |
| lingbot-depth | 1199 / 2576 | 297.1 / 2267 | 280.3 / 1559 |
| lingbot-depth-dc | 1200 / 2576 | 297.3 / 2267 | 280.7 / 1559 |
| lingbot-map | 2996 / 10047 | 993 / 7141 | 729.7 / 4512 |
| depthfm-v1 | OOM (16 GB) | OOM (16 GB) | 5662 / 13651 |

New kernels rows (kernels built locally with `tools/build_kernels.py gsraster|raft_flow --cuda-home /usr/local/cuda`; the manifests were unchanged):
- memfof-sintel 1080x1920: 1993 ms, 7249 MB peak (original and portable: OOM). 375x1242: 379 ms.
- resplat-base-re10k 480x640: 1280 ms, 6944 MB (portable 7085 ms).

Notes
- No family is slower or larger than its original (lingbot-depth portable 280 ms vs the authors' fp16 297 ms is the closest).
- Original adapters (`tools/_original.py`): any2full, lingbot_depth, depthfm, lingbot_map (golden-recipe inputs; xFormers off, per-sample nested block).
- depthfm original OOMs in both precisions (ensemble 4 at 994x1482), recorded as `status: oom` rows.
- resplat original stays unmeasured: it needs the pointops and gsplat CUDA extensions (not built for sm_120); the reason in its card is kept.
