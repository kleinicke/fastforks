# Numbers audit follow-up, 2026-11-15

Command (all lib rows): `tools/speed_check.py run --sides lib --modes portable kernels --only <ref> --force`, via `gpulock-verylow`, RTX 5060 Ti,
torch 2.14.1+cu130, base commit 8f9c75c0 (raw: `results/raw/2026-11-15_numbers/lib*_<ref>.jsonl`). Rows replaced in `docs/data/speed.jsonl` (portable and kernels).

## Fixed
| ref | before | after | reason |
|---|---|---|---|
| film (480x854, 1080x1920) | 112.9 / 527.6 ms, 1093 / 4747 MB (stale 2026-11-14 rows, d199e94c+dirty) | 87.1 / 416.9 ms, 698 / 2856 MB (original 103.9 / 463.8 ms, 711 / 2953 MB) | stale rows; earlier fixes already in the code. Floor-test blocker cleared |
| restormer-sidd 512x512 | 227.0 ms (original 218.0) | 181.4 ms portable, 178.7 kernels, 850 MB (original 1221 MB) | root cause: `PixelShuffle` returns NCHW strides, the 16 residual adds of the 96-channel blocks mixed NCHW with channels-last (non-vectorized kernel, 2.5 ms each = 34 % of the time). `_block` now makes the residual stream channels-last (all restormer variants) |
| wd-convnext-tagger-v3 448x448 | 340 MB (original 259) | 260 MB portable, 238 MB kernels | root cause in `_optimize/transformer.py::_cnx_prep`: the layer scale was folded into a separate copy of every fc2 weight (80 MB = a second copy of 40 % of the model). Now folded into the parameters, `_w2` shares them. All timm ConvNeXt (layer-scale) models; test `test_convnext_layer_scale_fold_adds_no_weight_copy` |
| flowformer (3 refs) | portable fp32 +16 % vs fp16 original | unchanged, reason written in the card | portable is the validated fp32; equals the fp32 original (351 vs 352 ms, 3165 vs 3122 MB); fp16 default needs a full re-validation |
| 18 more refs (c-foundationstereo-s, dat-x4, gmflow-scale1/2, locotrack-b, marigold-depth-v1-1, moge-2/3, neuflow-v2, patchmatchnet, swin2sr, track-on2, rife, pspnet, ...) | audit flags | rows re-measured and replaced; most gaps were stale rows | see below for those still open |

`speed_check.py static` and `tests/test_speed_floor.py test_policy.py test_docs.py test_contract.py`: green (1884 passed). One stale moge-3-vitl allowlist entry removed.

## Remaining (not fixed), with reason
| ref | state after re-measure | why open |
|---|---|---|
| patchmatchnet 1536x1024 | portable 1714 MB vs fp16 original 1228 (fp32 original 1330); time 202 vs 220 ms | root cause not found (no profile of the per-stage peak yet) |
| c-foundationstereo-s 375x1242 | portable 3074 MB vs 2781; time 1076 vs 1121 ms | not found; profile shows fp32 layer_norm and copy kernels (28 % of time), a candidate |
| track-on2 480x854 | 1423 MB vs 1236 (time 1194 vs 1953 ms) | not investigated |
| locotrack-b 480x854 | 771 MB vs 719 (+7 %; time 187 vs 203) | not investigated |
| gmflow-scale1, moge-3-vitl, neuflow-v2, rife-4.26, pspnet 1024x2048 | still 2-9 % slower than the strictest original (memory is below it) | already in `speed_allowlist.json` with a written reason |
| efficientvit_b2, fastvit_t8, squeezenet1_1, las2-h, shufflenet, restormer lib rows 1/2 | worker timeouts (15 min lock wait), weights download error (shufflenet), neuflow-v2 kernels unavailable | not measured: needs a quiet GPU |
| flowformer-small-chairs/things | CUDA OOM during the contended re-measure | old rows kept |
| loftr-outdoor | eval cannot run: `~/.cache/fastforks/data/matching/megadepth1500/pairs_calibrated.txt` missing | dataset file absent |
| cotracker3-offline/online | no `[validated]` entries: not recorded | golden record not run (GPU queue) |
| Marigold V2 (stream=True default), seedvr2 | not re-measured | GPU queue time |
| kernels above portable (c_radio, clip, siglip2, perception_encoder, fast_acvnet, mmseg, ben2) | not touched | kernel agent |

## Follow-up (same day, commit base 3a7e4880, RTX 5060 Ti, torch 2.14.1+cu130; raw rows `results/raw/2026-11-15_numbers2/lib_rows.jsonl`)
Command: `tools/speed_check.py run --sides lib --modes portable kernels --only <ref> --force --prefix gpulock-verylow`; peak = alloc MB (reserved in the speed floor).
| ref | before | after | root cause / reason |
|---|---|---|---|
| patchmatchnet 1536x1024 portable | 1714 MB (fp16 original 1228) | 1039 MB, 201 ms (original 220 ms) | `run_stage` kept the grid temporaries (p/px/py/pz/gx/gy, ~290 MB), the unnormalised weights and the MLP's relu copy alive during the warp. Freed with `del`, in-place weight normalisation, in-place relu. CPU golden unchanged |
| c-foundationstereo-s 375x1242 portable | 3074 MB (original 2781) | 2147 MB, 1068 ms (original 1121 ms); 994x1482 6301 MB (original 8338) | profile: the disparity-attention chunk (Q, K, V, out of 32768 sequences = 808 MB at the peak). `CHUNK` 1<<15 -> 1<<13 in `portable.py` (result independent of the chunk) |
| locotrack-b 480x854 portable | 771 MB (original 719) | 603 MB, 188 ms (original 203 ms); kernels 614 MB, 173 ms | `model.py` kept the per-chunk feature lists next to their concatenation through the whole query loop; `_join` frees them (no cat for one chunk) |
| dat-2-x4 512x512 | 2546 MB (original 2436) | 2276 MB, 1312 ms portable / 1221 kernels (original 4880 ms) | `_spatial_block` padded q/k/v with zero padding (a 414 MB copy at the peak) even when the size divides; now only when padding is needed. Output bit-identical (max abs diff 0.0 vs the old code, portable and kernels) |
| track-on2 480x854 | 1423 MB vs 1403 fp16 original (the 1236 MB original is the 2.4x slower variant) | unchanged: peak is the 16-frame ViT + FPN chunk (group_norm 288, fused map 255, normalize 127, features 127 MB, profile `n/p_to2`); equals the original's own chunk, 1.4 % above the faster original, 15 % below it in time (1194 vs 1953 ms) | reason, not fixed |
| squeezenet1_1 portable | 41 MB (4.1x original) | 7 MB alloc / 28 reserved (original 10 / 30), 1.8 ms vs 2.6 ms | the 2026-11-04 row was stale (first-call cost); no code issue |
| efficientvit_b2 224 / fastvit_t8 256 / las2-h / shufflenet | not measured | efficientvit_b2 13.6 ms (original 14.7), 85 MB (173); fastvit_t8 3.3 ms (4.9), 45 MB (57); las2-h 375x1242 35.6 ms / 365 MB (original 36.4 / 352), 994x1482 112.8 ms / 1002 MB (134.9 / 1022); shufflenet 5.0 ms / 39 MB (fp16 6.5 / 47, fp32 4.8 / 45) | las2-h 375x1242 is 13 MB (3.7 %) above the original in alloc, reserved 390 vs 386; not profiled. shufflenet portable vs the fp32 original +4.5 % is launch bound (0.44 ms CUDA of 3.8 ms wall), allowlisted with that measurement. shufflenet kernels 55 / 76 MB vs original 47 / 45: above the original, not investigated (kernel agent) |
| flowformer-small-* | "OOM during the contended re-measure" | not contention: at 1080x1920 one attention allocation is 15.76 GiB (> the 15.5 GiB card), the original fails the same way (no original row either); 375x1242 re-measured 351 ms, 3118 MB | deterministic size limit, not a bug |
| seedvr2-3b 512x512 | not measured | portable = kernels 2295 ms, 6140 MB (original 13150 ms, 7598 MB) | the 401 was the private HF repo; weights taken from `data/ff-weights` via `FASTFORKS_WEIGHTS_PATH` |
| Marigold V2 (stream=True default) | 9079-9785 MB | portable 518x770 1683 ms / 792 MB, 994x1482 5093 ms / 1408 MB; kernels 1088 ms / 882 MB, 3886 ms / 1498 MB | **The Mac session must re-measure Marigold V2: no Mac data exists for the stream=True path.** |
| neuflow-v2-sintel / things | flagged | portable 375x1242 14.7 / 15.1 ms vs original 12.2 (+20 %), memory below (235 vs 310-318 MB); 1080x1920 71 ms vs 76 | allowlisted already; kernels mode unavailable (`raft_f...` kernel library) |
| gmflow-scale1-mixdata, eva02_large_patch14_448 | flagged | gmflow 56.2 vs 53.5 ms (+5 %, allowlisted), memory below; eva02 18.7 vs 32.1 ms, 648 vs 1819 MB | |
| loftr-outdoor MegaDepth-1500 (n=1500) | n=150: fp32 portable auc5 52.11 vs original 53.82 (-3.2 %) | upstream fp32 50.14 / 67.46 / 80.29; portable fp16 50.60 / 67.58 / 80.31 (74 ms vs 152 ms, 905 vs 1719 MB); portable32 50.75 / 67.83 / 80.24 | the 150-pair subset gap was noise; on all 1500 pairs portable is not below the original. No kernels mode for loftr. Rows in `docs/data/matching_eval.jsonl` (dataset megadepth1500) |
| cotracker3-offline / online | no [validated] | `[validated."cotracker3-*@1"]` written by `pytest tests/test_golden.py -m gpu --record -k cotracker3` (mean 0.0018 / 0.0015 px) | |
