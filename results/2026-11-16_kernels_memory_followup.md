# Kernels memory follow-up (2026-11-16, branch attempt/kmem2, GPU RTX 5060 Ti, traces via scratch trace.py, expandable_segments)

Fixed (kernels vs portable, peak MiB):
- depth-pro: decoder resblocks free the relu input early and write in place: alloc 3614 -> 3038 (portable 3285), reserved 3698 -> 3198. GPU golden passes.
- hrnet-w48: Kernels kept the portable fp16 net (121 MiB) next to the optimized copy; now replaced: alloc before call 267 -> 146 (portable 145), peak 343 -> 222 (portable 262). GPU golden passes (forward actually run).

Reason on the card (limitations, tools/gen_docs.py):
- CUDA graph pool + static outputs, reserved only: ema-vfi (+670 / +3200), sam3 (+300, +64 alloc), sam-hq (+125), mobile-sam, efficientvit-sam (+85), owlv2 (+120), yolox, rtmdet, rf-detr, rtdetrv2, eva02, perception-encoder, anycalib, pointdit (b +85, l two pools). `cuda_graphs=false` / `release_graphs()` frees them.
- Allocator fragmentation around cuDNN workspaces (alloc equal or lower): anysplat (+260 reserved, 257 MiB conv output), nnfoundation-cnn (+580 reserved, alloc 2985 vs 3011), swin2sr (+80), alltracker (+31 packed weights, +70 reserved).
- resplat: portable is already fp16 autocast, so no bf16 fix. Profile (base 480x640): kNN topk ~28 %, compositing cumprod/pow ~30 %, rest fp32 GEMM; card names them.
- monster-sceneflow, resplat-base: in my traces kernels are at or below portable (-180 / -50 reserved); the table rows compare stale/older portable rows.

Open:
- Re-measure speed.jsonl rows of depth-pro, hrnet-w48 and the stale nnfoundation-cnn kernels row (no side/date) after merge; docs still show old numbers.
- Per-family forward smoke after weight-dropping changes is covered by tests/test_golden.py (GPU, ran for depth-pro and hrnet); no new parametrized smoke test added.
- The 50-100 MiB rows were covered by the same two causes (graph pool, fragmentation) but not each traced.
