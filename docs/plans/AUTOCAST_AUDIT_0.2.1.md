# Portable autocast audit

2026-11-17. Investigation only: no weight-storage or inference behavior was changed. An autocast call can be useful
even when weights are already stored in FP16, so its presence alone does not establish unnecessary weight casts.

## Candidates retaining FP32 parameters

The portable implementations of RAFT, SEA-RAFT, FlowSeek, GMFlow, WAFT, FlowFormer, NeuFlow v2, MemFlow and MatAnyone
load their networks without a half-weight conversion and run parts of inference under FP16 autocast. RAFT, SEA-RAFT,
FlowSeek and MemFlow enter autocast in an iterative update loop, making them useful candidates to investigate repeated
casts. S2M2, LAS2, WAFT-Stereo, AANet, HitNet, AllTracker, LocoTrack and Track-On also merit an audit. These are
source-code observations, not claims that casting their full networks would be safe or faster.

On an RTX 5060 Ti, torch 2.14.1+cu130, the existing portable paths were run on a synthetic 256x448 image pair. The
profiler observed the following; the ordinary warm latency is the median of three synchronized calls after one warmup,
while copy counts come from a separate profiled call. Profiler copy counts include activation, output and parameter
copies and do not measure their individual cost. Parameter sizes count registered parameters, not all buffers.

| Model | FP32 parameters | Warm latency | `aten::_to_copy` calls |
|---|---:|---:|---:|
| RAFT-small | 3.78 MiB | 23.28 ms | 379 |
| FlowSeek-T | 162.83 MiB | 36.87 ms | 498 |
| GMFlow scale 1 Things | 17.85 MiB | 12.54 ms | 157 |
| WAFT-A1 | 244.66 MiB | 80.51 ms | 1,627 |

For example, RAFT-small recorded 36 copies with shape `[96, 242, 3, 3]` and WAFT recorded 84 with shape
`[1152, 384]`; these shapes are consistent with weights, but shape alone does not prove the source or contribution
to runtime. Exact profiler output and commands are retained locally in
`results/raw/2026-11-17_autocast_audit/profile.log` and `results/2026-11-17_autocast_audit.py`.

## Already pre-cast or intentionally mixed

TAPNext explicitly stores selected GEMM weights in FP16 while retaining autocast and FP32 components. MoGe,
LingBot-Depth and Metric3D similarly pre-cast selected blocks with `half_weights`; MVSplat, DepthSplat, MVSAnywhere
and ReSplat pre-cast the network but still use autocast around some operations. Their autocast regions are not by
themselves evidence that FP32 weight copies remain. Depth Pro documents FP16 weights with autocast. MatAnyone, in
contrast, explicitly documents FP32 weights under autocast.

The existing `_core/half.py` helper can pre-cast selected Linear and Conv weights while leaving norms and other
numerically sensitive parameters in FP32. A future change should compare current and proposed paths on the same GPU,
at standard sizes, and check non-finite outputs, task accuracy, latency, allocated and reserved memory, loading peak,
and export behavior. No full-network FP16 conversion or elimination of all autocast is assumed to be safe.
