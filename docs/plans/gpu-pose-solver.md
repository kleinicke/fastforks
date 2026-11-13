# GPU pose solver (idea, after the quality/speed audit)

Status: done 2026-11-12: `fastforks.matching.estimate_pose` / `estimate_fundamental` (portable PyTorch + CUDA kernels
`fastforks.kernels.ransac`) and `MatchingResult.matches(n)`. MegaDepth-1500 x5, RoMa v2 matches, RTX 5060 Ti + Ryzen 7
5800X (follow-up 6 of `results/2026-11-12_gpu_ransac.md`, raw rows in `results/raw/2026-11-12_gpu_ransac/accuracy/`;
the threshold-0.5 rows are in `docs/data/relative_pose_eval.jsonl`): essential,
kernels at threshold 1.0 AUC@5/10/20 69.87/81.48/89.33 at 0.055 ms/pair batched (1.68 ms for one pair alone) vs
PoseLib at its best threshold 69.96/81.58/89.47 at 39-45 ms/pair (a tie within noise); fundamental, kernels 63.65 AUC@5
at 0.074 ms/pair vs PoseLib 64.00. Open: CUDA-graph single-pair latency, publishing the `ransac` kernel build. The
roadmap line "GPU RANSAC only if a profile shows it dominates" is superseded.

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
