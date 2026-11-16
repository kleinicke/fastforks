# Kernels vs portable peak reserved memory (2026-11-16)

Branch attempt/kmem (off work/ports). GPU RTX 5060 Ti, `tools/kernels_speed.py` / `speed_check run --modes portable`, raw rows in `results/raw/2026-11-16_kmem/`. Rows merged into `docs/data/speed.jsonl` (latest row per ref/mode/input wins), docs regenerated.

## Fixed (general, shared code)
- `shared/cuda_graphs.py`: a CUDA graph's private pool stayed reserved while a later, much larger (>= 2x numel) shape ran next to it (cause of most +100..+900 MiB rows: fast_acvnet, igev++, ema_vfi, hitnet, mmseg, detection, sam*). `GraphCache` now frees all graphs and empties the cache when a >= 2x larger input arrives; `SmallShapeGraphs` releases them when an input above its pixel limit runs. Test: `tests/test_graph_cache.py::test_a_much_larger_shape_frees_the_small_graphs`. Same release in rt_monster_plusplus.
- tapnext (+93 MiB) and cotracker3-online (+43 MiB): the merged fp16 filters/closures kept the original parameters; the replaced weights are now dropped (`tests/test_no_duplicate_weights.py` known entries removed).
- Stale rows: many kernels rows dated <= 2026-11-04 predated earlier memory fixes; 373 kernels/portable rows re-measured.
- Part A: original resplat-base/small measured (base fp32 OOM, bf16 2521 ms; small bf16 493 ms); `no_original_reason` dropped. resplat portable is 2.8x (base) / 3.5x (small) slower than the bf16 original at 480x640 (recorded in the card limitations); kernels are 2.0x / 1.4x faster.

## Result
Of 274 comparable kernels-vs-portable rows still above portable: 223 by < 50 MiB (allocator rounding / CUDA-graph pool of the benchmark shape), 26 by 50-100, 25 by >= 100 MiB.

## Reasons for the remaining >= 100 MiB rows
- Direct CUDA-graph replay (ema-vfi +382/+122 at 1080x1920, yolox/rtmdet +70..100, owlv2, rf-detr-seg-2xl, mobile-sam): the graph pool of the benchmark shape itself is reserved but not counted as allocated (alloc is lower than portable by up to 2.4 GB there); speed-for-memory trade, `cuda_graphs=false` removes it.
- Real kernel-side buffers (alloc also higher): depth-pro (+329 MiB alloc; fused/merged weights), sam3 (+64), sam-hq (+40..45), marigold-v2 (+90), hrnet-w48 (+81), ema-vfi-t 480x854 (+51).
- Reserved only (alloc equal): pointdit-l (+246), anysplat (+260), eva02/wd-eva02 (+122), pe-spatial-g14, swin2sr: workspace/rounding of fused attention kernels, not yet traced.
- nnfoundation-cnn (+580): row dated 2026-11-14, predates a portable re-measure on 11-16 (portable now 3130); kernels not re-measured.

## Open
depth-pro, pointdit-l, anysplat, sam3 and nnfoundation-cnn deserve a per-kernel workspace trace; the 25 rows >= 100 MiB above carry no card text yet.
