# Allocator fragmentation after load: scan and shared fix (2026-11-16)

Commit: see git log of work/ports (arena scan). GPU RTX 5060 Ti, torch 2.14.1+cu130. Tool: `tools/arena_scan.py`
(`load(ref, mode, device="cuda")`, then reserved - allocated in MB, nothing run). Rows: `results/arena_scan/before.jsonl`
(all 1054 floor (ref, mode) pairs, portable + kernels, 0 errors), `after.jsonl` (981 pairs re-measured with the fix).

## Finding
Before: 662 of 1054 loads had fragmentation > 16 MB (per task: classification 123/165, depth 72/85, features 88/116,
detection 59/100, stereo 84/142, multiview 19/23, ...; max 823 MB stablenormal-turbo, 648 MB pi3). Every family that
builds weights on the device then casts / folds / packs has the holes, not only dinov3 / INSID3 / track-on.

## Fix (root)
`shared/arena.compact_runner` runs in `_core/model._build` after the build (CUDA only): per network of the runner
(`runner.networks()`, else the `nn.Module` attributes of kernels runners) every parameter / buffer storage moves, whole,
into one arena per device (views, packed qkv and tied weights stay views of one storage; values, dtype, shape, strides unchanged).
Storages that a tensor outside the registries also views are skipped (copying them would duplicate); networks < 64 MB are
skipped (an intermediate version without this guard raised the reserved memory of 14 tiny nets by ~14 MB); skipped when the arena does not fit next to the
old weights. The dinov3-only call was removed. `tests/test_arena.py` is general (views, buffers, ties, holes).

## Result (981 pairs re-measured, intermediate version without the 64 MB guard)
Fragmentation > 16 MB: 607 -> 213 of 981. Reserved after load summed over the 981 pairs: -42.2 GB; 485 pairs saved >= 16 MB,
225 pairs >= 64 MB. Allocated memory unchanged or lower (no pair > +8 MB). Remaining: 173 kernels loads (kernels runners keep
plain-tensor aliases of their packed weights, e.g. stablenormal-turbo 780 MB, pi3 390, pointdit-b 252, megaflow 195, vggt 179,
efficientvit-sam, sam2, dlnr, da3-giant) - they need the runner to register those tensors (per family), not done here.
Not re-measured: 73 pairs (GPU queue blocked by another long job); the 64 MB guard version was not re-run on GPU.

## Open
Speed rows (docs/data/speed.jsonl): not updated. Refs with >= 64 MB saved: `results/arena_scan/speed_refs_todo.txt` (156 refs);
run `tools/speed_check.py run --force --only ...`, merge, regen docs when the GPU is free. Speed is expected unchanged (weights only moved).
