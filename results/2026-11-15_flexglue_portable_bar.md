# FlexGlue (aliked-flexglue) portable vs original, 2026-11-15

GPU RTX 5060 Ti, torch 2.14.1+cu130, base commit 9c9e8a75 (worktree wt-ports-kernels7), median of 9 runs after 3 warm-ups, one MegaDepth1500 pair (scene 0022, 2854678489 / 2061699781) resized to 768x1024, 2048 ALIKED keypoints, threshold 0.1, max_distance_px 3.0, locked with gpulock-low inside cpuslot --gpu.
Command: `bench_fg.py <orig16|portable> out.pt` (scratch `k7/bench_fg.py`; original = cvg/FlexGlue match_pair(sparse=True, mp=float16) + LightGlue ALIKED).

| tier | ms | peak alloc MiB | peak reserved MiB | matches |
|---|---|---|---|---|
| original fp16 (authors' mp option) | 152.9 | 1801 | 1950 | 759 |
| portable | 129.6 | 1799 | 1962 | 760 |

Portable is 1.18x faster; allocated memory equal (-2 MiB), reserved +12 MiB (0.6 %, allocator noise). Match counts agree (759 vs 760). The original fp32 run was still queued behind other GPU jobs at commit time; the original fp16 row is the faster original tier. Kernels: none (scope: no new kernels).
