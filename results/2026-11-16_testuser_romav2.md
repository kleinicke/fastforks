# Test user: RoMa v2 (dense matching), 2026-11-16

Setup: persona researcher, installed venv (`fastforks/.venv`) with the `work/ports` source on PYTHONPATH (the venv's editable
install is the older main checkout without `fastforks.matching`), RTX 5060 Ti, MegaDepth-1500 pairs (0015/0022). Docs read:
README, docs/models/matching.md, `help(fastforks.matching)`. Every run in `cpuslot --gpu systemd-run ... gpulock-low`.

## What worked (no problem)
- `matching.load("romav2")`, call with paths, PIL, numpy uint8/float32/float64, torch uint8/float CHW, cuda tensors, `[1|2,3,H,W]`
  batch (one result, leading batch dim), list of paths (list of results), grayscale (PIL / HW / 1xHxW), RGBA, 16x16 .. 4000x3000,
  two different sizes; `matches`, `verify`, `estimate_fundamental`, `match_pairs`, options (`coarse_size`, `fine_size`, `keep_descriptors`),
  `dtype="float32"`, `mode="portable"`, CPU.
- Error messages good: float images in 0..255 (says divide by 255), HWC torch tensor (says torch is CHW), missing file, None,
  wrong model name (did you mean), bad mode / option / unknown option / device / weights path / dtype; offline with empty cache
  (WeightsDownloadError names the fix; kernels fall back to portable with a log line in auto).
- Speed claims (1024x768, kernels / portable): 355 / 551 ms (docs 354 / 546); peak allocated 1956 / 2034 MiB (docs 1909 / 1989),
  reserved 3136 / 3206 (docs 2300 / 2596: higher because the process had already run bigger sizes).
- Plot of the verified matches (colonnade pair, little overlap) looks right.

## Problems
1. bug: `estimate_pose(p1, p2, K, K)` with the CUDA matches from `result.matches()` and CPU K (the natural call) ->
   `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`. Same for float64 points.
2. bug: numpy points / numpy K / lists -> `AttributeError: 'numpy.ndarray' object has no attribute 'to' / 'dim'` (also `estimate_fundamental`, `estimate_absolute_pose`).
3. bug: K of shape [2,3] -> raw numpy reshape error.
4. bug: `result.matches(0)` / `(-5)` -> "no pixel of the first image has a certainty above zero" (wrong); `matches(1.5)`, `matches("a")` -> raw torch/TypeErrors.
5. missing docs: no matching quickstart or visualisation example in docs/usage.md (README row only).
6. docs bug: `HF_TOKEN` row of the environment-variable table in docs/usage.md sat after the paragraph (table broken).
7. confusing docs (not fixed): `verify()` on two unrelated photos returns `ok=True` with 332 "inliers" (7-point F always fits); now stated in usage.md.
8. confusing (not fixed): `dtype="float16"` with mode auto -> "romav2 portable cannot compute in float16" although the user did not ask for portable
   (usage.md explains the fallback); `print(result)` dumps the full warp tensors.
9. missing feature (not done): no homography estimator (`estimate_homography`); `matches(10**9)` -> CUDA OOM (user error, torch message).
10. minor: `match_pairs` with an out-of-range index -> raw IndexError.
11. doc: card says input H,W >= 32 but 16x16 runs (tolerant; fine).

## Fixes (branch tu-romav2 -> work/ports)
- 1-4: `_tensor` / `_intrinsics` helpers in matching/relative_pose.py used by estimate_pose, estimate_fundamental, estimate_absolute_pose
  (numpy / list / float64 / any-device K and mask; InputError for bad K / types); `MatchingResult.matches` validates `n`.
  Tests (fast): tests/test_relative_pose.py `test_estimators_accept_numpy_lists_and_float64`, `..._reject_bad_intrinsics_and_types`, `test_matches_rejects_bad_n`.
- 5, 7: usage.md matching quickstart (verify, pose, PIL plot). 6: table row moved.
Light tests: test_relative_pose, test_absolute_pose, test_matching_pairs, test_docs, test_policy pass.

## Still open
Items 8-10 and the 7-point-F "ok" semantics.
