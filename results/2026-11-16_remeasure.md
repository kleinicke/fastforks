# Re-measure of stale speed / memory rows (2026-11-16)

Branch off origin/work/ports (83165dcc), GPU RTX 5060 Ti, `results/raw/2026-11-16_remeasure/kernels_speed.py` (= `speed_check run --force`, protocol of the existing rows: first call, 3 warm-up, peak reset, median of 3 rounds x median of 5; peaks allocated / reserved), each ref in a fresh process under `gpulock-low`, queues `queue*.sh` in one capped unit. Raw rows: `results/raw/2026-11-16_remeasure/rows_*.jsonl`; merged into `docs/data/speed.jsonl` (latest row wins), docs regenerated. Tests (test_docs, test_policy, test_card_sync) pass.

## Staleness criteria
A kernels / portable row is stale when its model code changed after the row: (1) 3d1468f2 depth-pro, 3c377f23 hrnet, e701d00f / 080c8886 cotracker + tapnext (weights dropped); (2) 080c8886 `shared/cuda_graphs.py` (graphs of small shapes freed before a >= 2x larger shape): only families that use GraphCache / SmallShapeGraphs with benchmark inputs >= 2x apart, whose rows still predate the kmem re-measure (987ed1d1).

## Result (peak MiB allocated / reserved, kernels unless noted; old -> new)
| ref | old | new |
|---|---|---|
| depth-pro 518x770 / 994x1482 | 3614 / 3698, 3626 / 3938 | 3038 / 3198, 3050 / 3198 (portable 3285 / 3458) |
| hrnet-w48 | 343 / 378 | 222 / 262 (portable 262 / 302) |
| hrnet-w32 / w32-384 / w48-384 | 210 / 262, 305 / 380, 438 / 498 | 155 / 194, 250 / 294, 317 / 360 |
| cotracker3-offline / -online | 799 / 916, 798 / 922 | 757 / 882, 757 / 888 (portable 755 / 882) |
| nnfoundation-cnn | 2985 / 3710 | 2985 / 3710 (unchanged; portable 3011 / 3130, card reason stays true) |
| monster-sceneflow (kernels 994x1482) | 2216 / 2514 | 2216 / 2514; portable reserved 2454 -> 2574 (kernels now below portable) |
| resplat-base-re10k portable | 6951 / 8634 | 6951 / 8736 (kernels 6944 / 8706) |
| great-igev x4 (994x1482) | 1192 / 1564 | 1134 / 1302 |
| las2-l / las2-m (994x1482) | 397 / 600, 329 / 572 | 268 / 362, 235 / 336 |
| raft-stereo x3 (994x1482) | 765 / 802 | 722 / 738-758 |
| film (480x854) | 698 / 916 | 698 / 736 |
| segformer-b0 / b5 (1024x2048) | 148 / 238, 407 / 534 | 105 / 178, 294 / 432 |
| segnext t / s / b / l (1024x2048) | 147 / 206 ... 314 / 428 | 110 / 158 ... 277 / 320 |
monster-mix-all (alias) re-measured too: kernels 2216 / 2514, portable 2325 / 2492 at 994x1482.
Speed of all re-measured rows is within noise (< 1.5 %) of the old rows.

## Cards
No card reason contradicts a new number: depth-pro / hrnet / cotracker / great-igev / las2 had no "kernels above portable" reason; the resplat card (2.8x / 3.5x slower than the bf16 original, kernels 2.0x / 1.4x faster) matches 7086 ms / 1280 ms / original 2521 ms; the nnfoundation-cnn reason (+580 reserved) still holds.

## Not re-measured (outside the memory / graph commits; candidates for a later pass)
Rows older than a performance commit of their family: SEA-RAFT kernels (3284e0d0, in-place correlation pyramid; 7 refs), tipsv2 / waft-stereo / unidepth-v2 / vggt kernels (4c514873 shared DINOv2 blocks), vitpose portable (8b72fab0), SeedVR2-7B (5283f796 stream default), cotracker / tapnext-pp portable (unaffected). Single-shape graph families (rtmpose, detection, ...) are not changed by 080c8886.
