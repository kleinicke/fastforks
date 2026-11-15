# wd-convnext-tagger-v3 memory: what "340 -> 260 MB" meant (2026-11-15)

Setup: RTX 5060 Ti, torch 2.14.1+cu130, 448x448 synthetic, `tools/speed_check.py run --only wd-convnext-tagger-v3 --sides lib --modes portable kernels --no-profile --force`
(first call, 3 warm-up, peak reset, median of 3x5), one worktree per commit, `PYTHONPATH=<wt>/src`, under gpulock-low in a capped unit.
Persistent = `torch.cuda.memory_allocated()` after `load()` (weights only, fp16). Original: `tools/speed_original.py` (upstreamhalf = fp16, upstream = fp32).
Raw rows: `results/raw/2026-11-15_convnext_reconcile/`.

| commit | portable peak alloc / reserved | portable persistent | kernels peak alloc / reserved | kernels persistent |
|---|---|---|---|---|
| 9e927861^ (before the fold) | 340 / 374 | 269.0 | 318 / 358 | 270.6 |
| 9e927861 (fold into params, in place) | 260 / 292 | 189.0 | 238 / 276 | 190.7 |
| dc4c4c1d | 260 / 292 | 189.0 | 238 / 276 | 190.7 |
| d7d66a3d (no in-place fold; kernels own folded fc2 copy) | 260 / 292 | 189.0 | 318 / 358 | 270.6 |
| 956ea9ea (= work/ports head; kernels fold in place again) | 260 / 292 | 189.0 | 238 / 276 | 190.7 |
| original fp16 | 259 / 294 | - | - | - |
| original fp32 | 481 / 516 | - | - | - |

## What is true

- The commit message is correct for 9e927861: portable peak 340 -> 260 MB (persistent 269 -> 189 MB), measured now. Cause: a second, layer-scale-folded copy of every fc2 weight (80 MB).
- d7d66a3d did NOT regress portable: portable stayed 260 MB / 189 MB persistent, because it applies gamma in the residual `addcmul` (no copy, no in-place fold; `test_convnext_portable_weight_grads` passes). It regressed only kernels (238 -> 318 peak, 190.7 -> 270.6 persistent), fixed by 956ea9ea.
- Current head: portable 260 / 292 (original fp16 259 / 294, tie), kernels 238 / 276. The earlier recheck's "peak 408.6 MB" and "could not reproduce 340 -> 260" came from a different metric/setup: 340 is portable peak allocated BEFORE 9e927861 (not reproducible after it), 270.6 is kernels persistent at d7d66a3d.
- Metrics: 340/260/238 = peak allocated (max over first and timed calls); 374/292/276 = peak reserved; 190.7/189.0 = persistent weights. Not interchangeable.

## Wrong statements

- `results/2026-11-15_numbers_fixed.md` row for wd-convnext-tagger-v3: the figures (340 -> 260 portable, 238 kernels) are right, but its root-cause text ("folded into the parameters, `_w2` shares them") describes the code of 9e927861 only; since d7d66a3d / 956ea9ea portable applies gamma in the residual addcmul and only the kernels path folds in place. Corrected here, not hand-edited there.
- `docs/data/speed.jsonl` rows 3117/3118 (portable 260/292, kernels 238/276) and the docs generated from them match the measurements above: no change. (The row `commit` field says `42dd73cb+dirty`, a pre-d7d66a3d build; the numbers reproduce at 956ea9ea, so rows left as validated.)
- Nothing needed fixing at the root: portable is already at the original's memory without copying weights.
