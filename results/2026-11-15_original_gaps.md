# Original-side gaps (2026-11-15)

Machine: RTX 5060 Ti 16 GB, torch 2.14.1+cu130, fastforks `work/ports`. Originals: `tools/speed_original.py`
(3 warm-up, median of 5, uint8 CUDA input, float32 tier, peak = max over first and timed calls), same input as the
portable row. Raw rows and queue scripts: `results/raw/2026-11-15_origgaps/`; rows merged into `docs/data/speed.jsonl`.
New adapters in `tools/_original.py`: `omni_dc`, `flexglue`, `loma` (checkouts at the cards' pins; `tools/golden/loma.py`
`build` got a `device` argument, default CPU as before).

## Measured originals

| ref | input | original ms | original peak alloc / reserved MiB | portable ms | portable peak alloc MiB |
|---|---|---|---|---|---|
| omni-dc | 994x1482 | 875.5 | 5966 / 6732 | 554.3 | 4958 |
| aliked-flexglue | 768x1024 | 347.6 | 1801 / 1892 | 128.7 | 1775 |
| loma-b | 768x1024 | 1036.3 | 4176 / 4390 | 308.1 | 1767 |

## resplat-base-re10k (portable row, not an original)

Weights are public (haofeixu/resplat) and cached in `data/ff-weights/resplat`; the earlier "gated weights" note was wrong.
The portable row was missing because the bootstrap ran under a 175 s lock limit and one call takes 7.1 s
(3 warm-up + 3x5 timed + first call > 175 s). Measured with a 290 s limit: 480x640, portable 7084.8 ms,
6951 / 8634 MiB (alloc / reserved). resplat-small-dl3dv portable: 1723.4 ms, 4995 MiB (already merged).
No kernels row for resplat-base: the `gsraster` fatbin for sm_120 is not in the kernel cache and offline mode forbids the download.

## No original timed (reason in the card, `[upstream] no_original_reason`, shown on the overview and model pages)

| ref | reason |
|---|---|
| any2full-s / -b / -l | authors' code (zhiyuandaily/Any2Full) not checked out; no clone while `/` is under the 30 GB floor |
| lingbot-depth, lingbot-depth-dc | authors' code (robbyant/lingbot-depth) not checked out; same floor |
| lingbot-map | authors' code (Robbyant/lingbot-map) not checked out; same floor |
| depthfm-v1 | authors' code (CompVis/depth-fm) not checked out; same floor |
| resplat-base-re10k, resplat-small-dl3dv | original needs the pointops and gsplat CUDA extensions, not built for sm_120; the golden reference replaces them by slow PyTorch loops, so its time is not the original's |

Five of these are runnable once a small source clone is allowed (to `/mnt/hdd`) and an adapter written.

## Why the website showed "-" for portable

1. `tools/gen_docs.py` dropped every speed row that had an `error` key, also when it carried `ms` (memfof-sintel 375x1242).
2. For multiview (own-input tables) the portable cell was looked up at the kernels' input only (resplat, lingbot-map).
3. The memfof-sintel 1080x1920 portable out-of-memory had no row; it is recorded now (`status: oom`) and shown as OOM.

Fixed in the generator and data; `results/2026-11-15_measurement_gaps.md` still said memfof lacks portable rows (stale, patched).

## seedvr2-7b "OOM at 512x512" for portable and kernels

`docs/data/speed.jsonl` held `status: oom` rows (before `stream=True` became the default, 5283f796); the timed rows with
`options {stream: true}` lived only in `speed_options.jsonl`. `_oom()` ignored dates and sides, so the stale rows won.
Now an OOM row only marks a cell while no timed row of the same (ref, mode, input) is as new (test
`test_oom_row_older_than_a_timed_row_is_stale`), and the default-option rows are in `speed.jsonl`: portable 2714.8 ms,
6419 MiB; kernels 2711.5 ms. The original stays OOM (needs a block-wise offload hook).
