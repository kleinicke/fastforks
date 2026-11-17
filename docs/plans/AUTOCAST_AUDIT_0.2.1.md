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

A follow-up Python `TorchDispatchMode` attempt observed zero `_to_copy` calls at all on RAFT-small, despite the
profiler recording 379. That tracer therefore cannot distinguish parameter copies in this path; its zero matches
are not evidence that weight casts are absent. The per-parameter conversion count and performance impact remain
unmeasured (`results/raw/2026-11-17_autocast_audit/trace_diagnostic.log`).

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

## Static call-site inventory

This lists direct PyTorch autocast calls that are not unconditionally disabled, including conditional and vendored calls. It is a syntax inventory, not a claim that each path executes in a particular mode, keeps FP32 weights, or causes repeated weight casts. Calls routed through wrappers and aliases may require further inspection. Line numbers refer to the audited branch state.

- `src/fastforks/_core/pipeline.py`: 496
- `src/fastforks/depth/models/depth_pro/model.py`: 254
- `src/fastforks/depth/models/lingbot_depth/model.py`: 161
- `src/fastforks/depth/models/metric3d/model.py`: 146, 248
- `src/fastforks/depth/models/moge/model.py`: 294
- `src/fastforks/depth/models/unidepth_v2/model.py`: 298
- `src/fastforks/depth/models/unik3d/model.py`: 210, 404
- `src/fastforks/depth/models/zoedepth/model.py`: 128, 203
- `src/fastforks/flow/models/flowformer/model.py`: 67
- `src/fastforks/flow/models/flowseek/net.py`: 316
- `src/fastforks/flow/models/gmflow/model.py`: 55
- `src/fastforks/flow/models/memflow/net.py`: 240, 257, 264
- `src/fastforks/flow/models/memfof/net/model.py`: 309, 329, 334
- `src/fastforks/flow/models/neuflow_v2/net.py`: 311
- `src/fastforks/flow/models/raft/net.py`: 331
- `src/fastforks/flow/models/sea_raft/net.py`: 259
- `src/fastforks/flow/models/waft/model.py`: 58
- `src/fastforks/interpolation/models/gimm_vfi/model.py`: 352
- `src/fastforks/interpolation/models/gimm_vfi/net/raft/raft.py`: 15
- `src/fastforks/matching/models/flexglue/model.py`: 87
- `src/fastforks/matching/models/flexglue/net/flexglue.py`: 273
- `src/fastforks/matching/models/lightglue/net/disk.py`: 251
- `src/fastforks/matching/models/loma/model.py`: 147
- `src/fastforks/matching/models/romav1/net/roma.py`: 25
- `src/fastforks/matching/models/romav2/net/amp.py`: 11
- `src/fastforks/matting/models/matanyone/model.py`: 242
- `src/fastforks/matting/models/matanyone/net/utils/device.py`: 20, 32
- `src/fastforks/multiview/models/depthsplat/model.py`: 271
- `src/fastforks/multiview/models/mvsanywhere/model.py`: 449
- `src/fastforks/multiview/models/mvsplat/model.py`: 413
- `src/fastforks/multiview/models/resplat/model.py`: 432
- `src/fastforks/multiview/models/vggt_omega/net/models/vggt_omega.py`: 45
- `src/fastforks/shared/aliked/net/aliked.py`: 747
- `src/fastforks/shared/dedode/net/dedode.py`: 39
- `src/fastforks/shared/lightglue/net/lightglue.py`: 471
- `src/fastforks/stereo/models/aanet/model.py`: 92
- `src/fastforks/stereo/models/bridgedepth/net/inference.py`: 59
- `src/fastforks/stereo/models/crestereo/net/crestereo.py`: 104, 110, 197, 230, 263
- `src/fastforks/stereo/models/defom_stereo/kernels/inference.py`: 231
- `src/fastforks/stereo/models/defom_stereo/net/defom_stereo.py`: 12
- `src/fastforks/stereo/models/foundationstereo/net/core/foundation_stereo.py`: 29
- `src/fastforks/stereo/models/foundationstereo/net/inference.py`: 195
- `src/fastforks/stereo/models/hitnet/kernels.py`: 301, 334
- `src/fastforks/stereo/models/hitnet/model.py`: 97
- `src/fastforks/stereo/models/igev_plusplus/net/igev_stereo.py`: 13
- `src/fastforks/stereo/models/igev_stereo/net/igev_stereo.py`: 12
- `src/fastforks/stereo/models/las2/model.py`: 221
- `src/fastforks/stereo/models/mocha_stereo/net/mocha_stereo.py`: 14
- `src/fastforks/stereo/models/monster/net/monster.py`: 346, 409
- `src/fastforks/stereo/models/monster_plusplus/net/monster.py`: 346, 409
- `src/fastforks/stereo/models/nmrf/net/inference.py`: 49
- `src/fastforks/stereo/models/raft_stereo/net/raft_stereo.py`: 12
- `src/fastforks/stereo/models/rt_monster_plusplus/net/inference.py`: 103
- `src/fastforks/stereo/models/rt_monster_plusplus/net/monster.py`: 371, 472
- `src/fastforks/stereo/models/s2m2/model.py`: 83
- `src/fastforks/stereo/models/selective_igev/net/igev_stereo.py`: 13
- `src/fastforks/stereo/models/waft_stereo/model.py`: 79
- `src/fastforks/tracking/models/alltracker/model.py`: 127
- `src/fastforks/tracking/models/locotrack/model.py`: 62
- `src/fastforks/tracking/models/tapir/model.py`: 55
- `src/fastforks/tracking/models/tapnext/model.py`: 86
- `src/fastforks/tracking/models/track_on/model.py`: 104
- `src/fastforks/volume/_nnunet.py`: 581
