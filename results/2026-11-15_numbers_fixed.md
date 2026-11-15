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
