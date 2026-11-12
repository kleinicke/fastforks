# GPU pose solver (idea, after the quality/speed audit)

Status: planned, not started (user, 2026-11-12). Worker: a Fable agent (user asked for Fable 5.2).

## Why
Users of a dense matcher (RoMa v2, later others) almost always run RANSAC on the matches. Today that is OpenCV on the
CPU: matches are downloaded, then `findEssentialMat` + `recoverPose` run on a single core. A GPU solver keeps the whole
"images -> relative pose" pipeline on the GPU, with no host round trip.

## Scope
- Essential matrix (5-point) and fundamental matrix (7/8-point) minimal solvers, batched hypotheses per kernel.
- RANSAC scoring of all hypotheses against all matches in parallel; LO-RANSAC / local refinement; `recoverPose`.
- Batched API: many image pairs per call (the MegaDepth-1500 layout: 5000 matches x 5 samplings per pair).
- Portable version in plain PyTorch (batched SVD / polynomial solve) + a kernels version.

## Not for the benchmark
Evaluation stays on OpenCV `findEssentialMat` (0.5 px / mean focal, confidence 0.99999), as in the papers; a different
solver changes the AUC. The GPU solver is a product feature. Report it next to OpenCV on MegaDepth-1500 (AUC and
ms/pair), and say clearly which solver each number uses.

## Cheaper first step for the eval itself
If the CPU scoring step of `tools/eval_matching.py` is ever the bottleneck, spread the pairs over the 16 cores
(process pool). Matching on the GPU, not RANSAC, takes most of the eval time today.
