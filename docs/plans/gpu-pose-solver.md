# GPU pose solver (idea, after the quality/speed audit)

Status: done 2026-11-12: `fastforks.matching.estimate_pose` / `estimate_fundamental` (portable PyTorch + CUDA kernels
`fastforks.kernels.ransac`) and `MatchingResult.matches(n)`. MegaDepth-1500 x5, RoMa v2 matches, RTX 5060 Ti + Ryzen 7
5800X (follow-up 6 of the 2026-11-12 GPU RANSAC measurement, raw accuracy rows kept with it;
the current rows, each solver at its best threshold, at commit 1ca41bd9 in `docs/data/relative_pose_eval.jsonl`): essential,
kernels at threshold 1.0 AUC@5/10/20 69.87/81.48/89.33 at 0.054 ms/pair batched (18,446 pairs/s; 1.69 ms for one pair alone) vs
PoseLib at threshold 1.0 69.96/81.58/89.47 at 325 pairs/s on 16 processes, 29 ms per pair on one core (a tie within
noise); OpenCV at 0.5 62.87 AUC@5 at 323 pairs/s, 24 ms per pair on one core; fundamental, kernels 63.65 AUC@5
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

## Apple Silicon baseline (M1 Pro 16 GB, 2026-11-27)
Measured before writing any MLX solver; data in `docs/data/relative_pose_apple_silicon.jsonl` (29 rows: accuracy and speed
per seed, plus two stage profiles). Read it as a rough baseline:

- Data: synthetic, not MegaDepth-1500 (that eval set is not on the Mac). 100 seeded pairs per seed, 5000 matches each,
  1200x900 images, focal 900-1500 px, inlier ratio 0.15-0.85, 0.5 px noise, random outliers, rotation up to 40 deg. Three
  seeds (1000+seed), mean over seeds. Thresholds as in the CUDA rows (essential 1.0, fundamental 0.5, absolute 2.0 px).
  AUC is `eval_matching.pose_auc`; the synthetic pairs are easier than MegaDepth, so compare methods, not to the RTX table.
- CPU rows use one core (`OMP_NUM_THREADS=1`); OpenCV 5.0 `findEssentialMat` / `findFundamentalMat` / `solvePnPRansac`
  (AP3P). PoseLib is not installed on the Mac, so it is not in the table. The machine is shared with other agents, so
  times vary by tens of percent between runs (the std over seeds in the JSONL shows it).
- Finding: the portable path does not run on MPS as it is. `_TorchScorer.hypotheses` casts the minimal-solver inputs to
  float64 (Metal has no fp64) and `eigvals` is not on MPS (CPU fallback). The MPS rows therefore use a scratch-only
  monkeypatch that runs the hypothesis solver in fp32 (not committed); `torch.linalg.eigvals` still falls back to the
  CPU for the fundamental solver. The fp32 solver loses no accuracy here.

| problem | method | AUC@5 | AUC@10 | AUC@20 | one pair (ms) | batched 100 (ms/pair) |
|---|---|---|---|---|---|---|
| essential | OpenCV, 1 core | 69.5 | 80.0 | 86.8 | 124 | 114 |
| essential | portable, CPU | 86.8 | 88.7 | 90.8 | 134 | 86 |
| essential | portable, MPS (fp32 patch) | 87.7 | 89.5 | 91.3 | 264 | 19.0 |
| fundamental | OpenCV, 1 core | 48.8 | 58.6 | 66.5 | 54 | 55 |
| fundamental | portable, CPU | 66.0 | 69.8 | 73.8 | 41 | 27 |
| fundamental | portable, MPS (fp32 patch) | 66.7 | 69.9 | 73.5 | 85 | 10.6 |
| absolute | OpenCV, 1 core | 99.9 | 100.0 | 100.0 | 15 | 27 |
| absolute | portable, CPU | 99.9 | 100.0 | 100.0 | 25 | 26 |
| absolute | portable, MPS (fp32 patch) | 99.5 | 99.7 | 99.9 | 47 | 11.3 |

(The std of AUC over seeds is 4-9 points for essential and fundamental, because the seeds differ in inlier ratio; the
rows with the same seed are paired. Absolute pose is saturated on this data, so it says little about accuracy.)

Profile of the portable essential path on MPS (synchronised stage timers; one more pass under a `TorchDispatchMode`
counts ops and host syncs):

| | one pair | 100 pairs |
|---|---|---|
| wall | 157 ms | 1886 ms |
| 5-point solver (`hypotheses`, 2 / 8 rounds of 128) | 68 ms (45%) | 1031 ms (62%) |
| scoring (`score`, 17 / 29 calls) | 17 ms | 307 ms |
| LM refine | 39 ms | 228 ms |
| refit SVD + recoverPose + inliers + polish | 21 ms | 49 ms |
| other | 6 ms | 42 ms |
| torch dispatches | 17,282 | 55,419 |
| host syncs (`.item()`, boolean index) | 3 | 9 |

So on one pair the MPS path is dispatch bound (17k tiny ops, 12k of them in the 5-point solver and 3.5k in LM, ~9 us
each) and loses to the CPU by 2x; with 100 pairs the same ops get wide enough to beat one CPU core by 4.5x for essential,
2.5x for fundamental. Host syncs are not the problem (3-9 per call); op count is.

### Plan: MLX kernels (`mx.fast.metal_kernel`), mapped from `kernels/ransac/ransac.cu`
Metal has no fp64 and `mx.linalg` svd/inv/eig are CPU only, so each kernel carries its own small fp32 linear algebra
(the CUDA file already has Jacobi eigensolver / `svd3` for the 3x3 cases). The fp32 solver result above says fp32 hypotheses
are accurate enough; the roots are polished by scoring and LM afterwards anyway.

1. `e5pt_hypotheses` (null space, constraint matrix, Gauss-Jordan, `roots10`; lines 59-340): one thread per hypothesis,
   128 per round, no cross-thread work. 62% of batched and 45% of single-pair time and 12k dispatches. Needs fp32 polynomial
   arithmetic (the CUDA path uses double for `m[10][20]` and the roots): keep the elimination in fp32 with partial
   pivoting and add two Newton steps on the degree-10 polynomial; check it against the fp64 CPU path on the same samples.
2. `score_many` / `score_few` (Sampson gain per match, MSAC cost; lines 425-560): one threadgroup per hypothesis, reduce over
   matches in threadgroup memory. 18% of batched time, easy, shared by all three problems (the `_abs` variants differ in the
   residual only).
3. `lm_refine` + `polish` (lines 565-1030: `lm_run`, `svd3`, `decompose`, `cheirality_pass`, `recover`): one threadgroup per
   refined hypothesis (top 8). Brings the 3.5k LM dispatches and recoverPose (the SVD decomposition) down to one launch.
   14% of batched, 26% of single-pair time.
4. Later: `f7pt_hypotheses` (cubic only, lines 346-420; also removes the eigvals CPU fallback) and `p3p_hypotheses`
   (quartic, lines 1140-1345). Absolute pose is already cheap, so it comes last.

Expected win (estimate, not measured): the whole pipeline collapses from ~17k dispatches to about 15 launches per RANSAC
round set, so a single pair should land in the low milliseconds instead of 157-264 ms, and 100 pairs should move from
19 ms/pair to a few tenths of a millisecond per pair. For scale, the CUDA kernels reach 0.054 ms/pair on an RTX 5060 Ti
(about 4.5x the fp32 throughput of an M1 Pro GPU), which suggests 0.3-0.5 ms/pair batched here. The target for the first
milestone: beat one CPU core of OpenCV on a single pair, which the portable path does not do on MPS today.

Start with the essential matrix: it is the slowest of the three (114 ms OpenCV, 134 ms portable CPU per pair), the 5-point
solver dominates its profile, and it is what `estimate_pose` users run. Order: kernel 1 first (largest share, and its fp32 accuracy is
the one real risk), then 2 and 3 so the whole path is in kernels, with the torch version kept as the reference in tests.
