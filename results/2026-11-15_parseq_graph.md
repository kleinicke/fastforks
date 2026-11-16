# NeuFlow v2 things re-measure and PARSeq CUDA graph (2026-11-15)

Commit: see git log of work/ports. GPU RTX 5060 Ti, torch 2.14.1+cu130, `tools/speed_check.py run` (first call, 3 warm-up,
median of 3 rounds x 5) under gpulock-low; raw rows in `results/raw/2026-11-15_neuflow_parseq/`. Kernels use
`FASTFORKS_KERNEL_PATH=wt-port-p1/build/kernels` (raft_flow a101a33607b6, local; not yet on HF).

## neuflow-v2-things (the earlier rows mixed two code states; one row per size now, floor rows replaced)

| size | original fp32 | original fp16 (authors) | portable | kernels |
|---|---|---|---|---|
| 375x1242 | 27.40 ms / 697 MB | 12.18 / 318 | 14.30 / 235 | 12.19 / 157 |
| 1080x1920 | 171.7 / 8694 | 75.77 / 4374 | 71.41 / 2366 | 58.88 / 779 |

Kernels are faster than or tied with the original at both sizes (12.19 vs 12.18 ms at KITTI size, -22 % at 1080p) with half the
memory; portable is 17 % slower than the authors' fp16 at 375x1242 (documented in speed_allowlist.json: fp32 correlation,
lookup and upsampling kept for accuracy; KITTI EPE 4.0946 vs 4.0975) and faster at 1080p. Accuracy of kernels:
KITTI-2015 200 pairs EPE 4.0953 / Fl 15.01 vs portable 4.0946 / 15.009, original 4.0991 (docs/data/flow_eval.jsonl).
The earlier "no valid kernels row" was the sandbox missing the raft_flow binary (404 on HF), not a model problem.

## parseq: CUDA graph in portable

`Portable` on CUDA replays the 26-step decoding (stop=False, same logits as the early exit) from a `GraphCache` graph from the
second call of a batch size on (`cuda_graphs` option, default true, speed only; false: eager). `module()` (export, compile)
is unchanged plain PyTorch; CPU/MPS never graph. Policy check: no policy or docs test restricts graphs to kernels; the
repo's other graph users are kernels-mode families, so this is the first portable graph, behind an option.
Test: `tests/test_ocr_parseq.py::test_graph_replay_matches_eager` (gpu) and `test_no_graph_on_cpu`; policy / contract / docs /
review tests pass (1946 passed).

| ref | original fp32 / fp16 ms | portable before | portable now | peak alloc / reserved MB before -> now |
|---|---|---|---|---|
| parseq | 15.94 / 19.56 | 12.57 | 3.63 | 79 / 98 -> 79 / 100 |
| parseq-tiny | 14.58 / 17.98 | 11.57 | 2.10 | 44 / 54 -> 44 / 54 |

Kernels stays `status = none` with the reason updated: the graph is already in portable and no library helps.
