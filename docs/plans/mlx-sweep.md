# MLX sweep: every family gets an Apple MLX port and a kernels mode

Goal (user): for every family, an MLX implementation that needs less memory than the MPS one and is at least as fast,
with the same quality (goldens), plus a `kernels` mode that actually helps. A family that cannot be done gets a written
reason here (table at the bottom), not silence.

## Tools that exist

- `fastforks.shared.mlx_graph`: `GraphNet` / `MlxModule` export a `torch.nn.Module` (torch.export, decomposed) and replay
  it on MLX under `mx.compile`. The op table `OPS` maps aten ops to `mx`; an unmapped op raises `UnsupportedOp` naming it.
  Add the mapping there (keep it exact; fp32 accumulate for norms / softmax). Convs run NHWC.
- `fastforks.shared.mlx_wrap.port(Portable)`: the family's `portable_mlx` / `kernels_mlx` runner. The portable runner is built
  on CPU in float32, its pre/post-processing stays PyTorch, and the network attribute(s) (`attrs=("net",)`) run on MLX in
  `ctx.dtype`. See `restoration/models/nafnet/model.py` for the two-line wiring and the card keys:
  `[modes.portable] mlx_dtypes = [...]`, `[modes.kernels] mlx_kernels = ["graph_mlx"]` + `mlx_dtypes = [...]`.
  Where the generic wrapper cannot reach (the network needs python control flow, data-dependent shapes, a non-tensor
  argument that changes per call, a custom CUDA-only op), restructure: wrap only the exportable sub-module(s) (`attrs`,
  or `MlxModule(submodule)` by hand inside a custom runner) and keep the rest in torch on the CPU, or write the part in
  `mx` by hand (see `shared/mlx_vit.py`, `kernels/dpt_mlx`).
- Kernels mode: `kernels/graph_mlx` (depthwise conv, global average pool) switches on with `kernels=True`. A kernels mode must
  be measurably faster than the portable MLX mode on that family; if the family has no op a kernel can speed up, write one
  (new `kernels/<lib>_mlx/` with `ops.py` using `fastforks._core.mlx.metal_kernel`, `meta.toml`, `__init__.py` Library like
  `kernels/graph_mlx`), tune it, and only declare `mlx_kernels` where it wins. If after real tuning nothing helps, do not
  declare kernels for that family and note why in the table below.
- fp16: only declare `float16` in `mlx_dtypes` when the golden passes in fp16 (overflow → nan is the usual failure: try
  bfloat16 is NOT an option unless the card allows it; fp32-only is acceptable when fp16 fails after an honest attempt).

## Rules

- THE MAC IS SHARED AND HAS A 16 GB M1 Pro WITH ~20 GB FREE DISK. Never run two MLX/torch model jobs at once; never run
  big models in a loop in one process (one process per ref; peaks add up); watch `df -h /` (stop if internal disk < 10 GB
  free); keep caches / temp / weights on `/Volumes/Daten` (env below). Remove scratch files you create when done.
  Do not run CPU-heavy work that is not the model test itself.
- Edit shared files (`mlx_graph.py`, `mlx_wrap.py`) only to add op mappings / fixes; keep existing behaviour; rerun
  `tests/test_golden.py::test_mlx[nafnet-gopro@1]` and one classifier (`wd-vit-tagger-v3@1`) after any change to them.
- Do not edit goldens or tolerances. Quality must pass the existing `test_mlx` golden of each ref (every declared
  mode x dtype).
- Commit on the current branch (`codex/mlx-v021`) in small commits per family group, message ends with
  `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` only. No push, no stash, no branch switching.
- Format touched files with `ruff format`; `ruff check` the new code; keep docstrings in the repo's style.

## Environment

```
export FASTFORKS_GOLDEN_PATH=/Volumes/Daten/ff-golden-all FF_TEST_DEFAULT_CACHES=1 \
  FASTFORKS_CACHE=/Volumes/Daten/fastforks-cache FASTFORKS_OFFLINE=1 PYTHONPATH=src
PY=/Volumes/Daten/ff-mlx/venv/bin/python
$PY -m pytest "tests/test_golden.py::test_mlx[<ref>]" -q --basetemp=/Volumes/Daten/fastforks-tmp/bt -p no:cacheprovider -m mlx
```

Speed and memory (MLX vs MPS): rows of the same ref/mode/dtype/input are in `docs/data/speed/apple-m1-pro-16gb.jsonl` (mps) and
get written to `docs/data/speed/apple-m1-pro-mlx-16gb.jsonl` by
`$PY tools/speed_survey.py run --device apple-m1-pro-mlx-16gb --only <ref> --modes portable kernels` (ms, process footprint MB,
golden status). Compare `ms` and `peak_footprint_mb` of the two files. MLX must be <= MPS in both; if a mode is not, tune
(layout propagation, fused kernels, dtype, fewer transposes, compile) before giving up.

## Results

Fill one row per family: MLX portable / kernels vs MPS (ms and MB, at the survey's input), or the reason it is not possible.

| family | task | status | notes |
|---|---|---|---|
| compact (realesr-general-x4v3) | restoration | portable_mlx, fp32+fp16 pass | 512x512, ms / MB. fp16 232 / 953 vs MPS 132 / 1289; fp32 375 / 1534 vs 178 / 1288. Memory wins in fp16; speed loses (MLX 64ch 3x3 conv ~5.8 ms vs ~4.1 ms MPS). graph_mlx kernels (depthwise, pool) give no gain here, so no kernels mode. |
| span | restoration | portable_mlx, fp32+fp16 pass | x4 fp16 131 / 945 vs MPS 89 / 1296; fp32 172 / 1378 vs 103 / 2321. x2 fp16 126 / 844 vs 74 / 1296. Lower memory, slower. |
| realplksr | restoration | portable_mlx, fp32+fp16 pass | fp16 1879 / 1277 vs MPS 1092 / 1368; fp32 2374 / 1537 vs 1527 / 1384. Slower. |
| realesrgan | restoration | portable_mlx, fp32+fp16 pass | x4plus fp16 5914 / 3485 vs MPS 2066 / 2466; fp32 8130 / 5341 vs 3022 / 4546. Slower and heavier (conv-bound). |
| swinir | restoration | portable_mlx, fp32+fp16 pass | m-x2 fp16 4874 / 2696 vs MPS 5797 / 3500; l-x4-real fp16 7700 / 4181 vs 9874 / 4561, fp32 10332 / 6250 vs 12063 / 9130. Faster and lighter than MPS. |
| swin2sr | restoration | portable_mlx, fp32+fp16 pass | x4 fp16 4701 / 4823 vs MPS 9410 / 4580; x2 fp16 4474 / 2870 vs 9258 / 3501; x4-real fp16 4774 / 3750 vs 9531 / 4579. About 2x faster than MPS, memory equal or lower. |
| dat | restoration | portable_mlx, fp32+fp16 pass | dat-x4 fp16 9857 / 5105 vs MPS 10512 / 10620; fp32 12783 / 7905 vs 21047 / 13161. dat-light-x4 fp16 2922 / 2809 vs 2562 / 6851. dat-2-x4 fp16 11318 / 5116 vs 9404 / 10600 (MPS faster). Memory about half of MPS. |
| hat | restoration | portable_mlx, fp32+fp16 pass | MPS runs out of memory. MLX hat-s-x4 fp16 7004 / 7298, fp32 11073 / 10176; hat-x4 fp16 12245 / 8358, fp32 14403 / 10274; real-hat-gan-x4 fp16 10273 / 8358. All golden pass: MLX win by being the only backend that runs. fp16 needed fp32 accumulation in sum, mean and vector norm. |
| drct | restoration | portable_mlx, fp32+fp16 pass | MPS runs out of memory. drct-x4 fp16 4771 / 4826, fp32 6705 / 7790; drct-l-x4 fp16 9227 / 5024, fp32 12855 / 8118. MLX-only win. |
| restormer | restoration | portable_mlx, fp32+fp16 pass | deblur fp16 1839 / 3274 vs MPS 1476 / 3657; derain fp16 1838 / 3277 vs 1814 / 3655; sidd fp16 1683 / 3467 vs 2330 / 3683. Roughly at parity, lower memory. fp16 needed the gram bmm in float64 (fp32 accumulation). |
| lama | restoration | portable_mlx whole-module, fp32+fp16 pass | big-lama fp16 286 / 1656 vs MPS 214 / 1679; fp32 277 / 1758 vs 241 / 1797. Slightly slower, same memory. Needed FFT, empty and strided slice_scatter ops. |
| codeformer | restoration | portable_mlx whole-module, fp32+fp16 pass | fp16 368 / 2512 vs MPS 196 / 1866; fp32 467 / 3128 vs 260 / 3064. Slower. |
| gfpgan | restoration | portable_mlx whole-module, fp32 only | fp32 364 / 3161 vs MPS 153 / 1954. fp16 golden mean diff 0.157, so not declared. Slower and heavier. |
| seedvr2 | restoration | not ported, infeasible on this Mac | 7B bf16 weights are 16.5 GB, over the 16 GB RAM. The 3B is 6.8 GB bf16 but the port builds a CPU fp32 twin (~13.6 GB) plus the MLX copy. MPS already reports it unsupported. Needs a streaming loader (convert layer by layer to MLX) first. |
| deeplabv3 / deeplabv3plus / fcn / pspnet | segmentation | portable_mlx, fp32+fp16 pass | 1024x2048 fp16 ms / MB. deeplabv3 469 / 2804 vs MPS 528 / 2010; fcn 379 / 2772 vs 405 / 1875; pspnet 413 / 2849 vs 411 / 1945; deeplabv3plus 489 / 2882 vs 430 / 1891. Conv-bound: about a tie on speed, more memory. |
| hrnet | segmentation | portable_mlx, fp32+fp16 pass | fp16 303 / 2603 vs MPS 228 / 2044; fp32 382 / 2844 vs 263 / 3210. Slower. |
| segformer | segmentation | portable_mlx (+ graph_mlx trial on b0), fp32+fp16 pass | b0 fp16 110 / 2027 vs MPS 113 / 1521; b5 505 / 3097 vs 375 / 1974; b2-clothes (512x512) 67 / 1232 vs 50 / 1643 (less memory); b5-face 113 / 1619 vs 111 / 1899. The graph_mlx kernels gain 5-10 % only. |
| segnext | segmentation | portable_mlx (+ graph_mlx trial on t), fp32+fp16 pass | t fp16 143 / 1974 vs MPS 118 / 1445; s 183 / 2035 vs 134 / 1556; b 284 / 2150 vs 184 / 1659; l 453 / 2335 vs 268 / 1840. Slower: large depthwise convs. |
| bisenet | segmentation | portable_mlx, fp32+fp16 pass | 512x512 fp16 18 / 682 vs MPS 14 / 440. Tiny net, dispatch bound. |
| eomt | segmentation | portable_mlx, fp32+fp16 pass | dinov2 fp32 666 / 5998 vs MPS 608 / 6308; fp16 568 / 4630 vs 540 / 4262. dinov3 fp32 743 / 6225 vs 648 / 6310. Near tie. |
| mask2former | segmentation | portable_mlx, fp32+fp16 pass | fp32 1059 / 5322 vs MPS 909 / 6661 (less memory); fp16 1067 / 4800 vs 964 / 4503. |
| oneformer | segmentation | portable_mlx, fp32+fp16 pass | fp32 1742 / 6961 vs MPS 1127 / 9492 (less memory); fp16 1795 / 6454 vs 1161 / 5559. The survey golden shows "fail" on MPS and MLX alike (survey input quirk); `test_mlx` passes. |
| sapiens | segmentation | portable_mlx, fp32+fp16 pass | 0.3b fp32 2247 / 8940 vs MPS 2988 / 4766 (faster); fp16 1927 / 7268 vs 1517 / 3748. 1b fp32 4376 / 11843 vs 5099 / 9794 (faster); fp16 3524 / 8795 vs 3113 / 6340. |
| segmentation notes | segmentation | | The label path runs the logits' bilinear resize and argmax as one fused Metal kernel (`mlx_seg.upsample_argmax`); full logits are resized on the CPU. Pre-processing stays on the CPU. Memory is mostly above MPS. |
| any2full | depth | portable_mlx, fp32+fp16 pass | 994x1482 fp16 s 169 / 1696 vs MPS 118 / 1641; b 435 / 2816 vs 268 / 2198; l 1031 / 5345 vs 684 / 3956. Slower. |
| depth_anything_3 | depth | portable_mlx, fp32+fp16 pass | mono-large fp16 363 / 2516 vs MPS 280 / 2499; metric-large 367 / 2604 vs 280 / 2499; da3-large-1.1 393 / 2578 vs 318 / 2520; giant-1.1 fp16 963 / 4196 vs 759 / 7243 (less memory); nested giant+large fp16 1334 / 4880 vs 1184 / 8136 (less memory). Slower, but fp16 giants use 40 % less memory. |
| depth_anything_v2 (and metric, distill-any-depth) | depth | existing MLX runner, fp32+fp16 pass | fp16 s 72 / 898 vs MPS 81 / 1575; b 181 / 1410 vs 199 / 2061; l 519 / 2343 vs 579 / 3282. Faster and lighter than MPS for all 12 refs. The kernels rows are slower (l fp16 548 / 2788). |
| depth_pro | depth | portable_mlx (net via `fast`), fp32+fp16 pass | fp16 4491 / 11370 vs MPS 5011 / 7169 (faster, more memory); fp32 6473 / 15337 vs 6115 / 11267. |
| depthfm | depth | portable_mlx, fp32+fp16 pass | Survey: out of memory on MLX and on MPS alike at 994x1482 (a 66 GB attention buffer). Golden passes at the test size. |
| lingbot_depth (+dc) | depth | portable_mlx, fp32+fp16 pass | fp16 4589 / 8528 vs MPS 3376 / 3228. Slower and heavier. |
| lotus (normal g, d) | depth | portable_mlx, fp32+fp16 pass | g fp16 2089 / 9398 vs MPS 1553 / 6632; d fp16 2166 / 9159 vs 1459 / 6628. UNet and VAE run on MLX, loop and noise on the CPU. Slower and heavier. |
| marigold | depth | portable_mlx, fp32+fp16 pass | fp16 4145 / 9333 vs MPS 2498 / 6640. Slower and heavier. |
| marigold_normals | depth | portable_mlx, fp32+fp16 pass | fp16 2104 / 9797 vs MPS 1655 / 6671. |
| stablenormal | depth | portable_mlx, fp32+fp16 pass | fp16 2661 / 11417 vs MPS 1690 / 8985 (UNet, ControlNet and VAE on MLX). |
| marigold_v2 | depth | not ported, infeasible on this Mac | 20 B Qwen-Image transformer (9 GB NF4, about 40 GB dequantized in fp16 or 16 GB with 4-bit per-call expansion) does not fit 16 GB. Card keeps its NF4 bf16 mode only. |
| metric3d | depth | portable_mlx, fp32+fp16 pass | fp16 2370 / 5009 vs MPS 1861 / 3636. Slower. |
| midas | depth | portable_mlx, fp32+fp16 pass | fp16 668 / 4117 vs MPS 635 / 2407. The survey golden shows "fail" on MPS too (survey input quirk); `test_mlx` passes. |
| moge | depth | portable_mlx (`_trunk` graph), fp32+fp16 pass | vits fp16 1148 / 6234 vs MPS 857 / 1679; vitb 1394 / 6475 vs 1012 / 2265; vitl 2221 / 7987 vs 1732 / 3265; moge-3 6508 / 10192 vs 3913 / 5968. The data-dependent sparse refiner of moge-3 stays eager. Slower and heavier. |
| omni_dc | depth | portable_mlx (pre/post graphs, CG solve on the CPU), fp32+fp16 pass | fp16 3300 / 9841 vs MPS 2408 / 10081. |
| pointdit | depth | portable_mlx (`_encode` fp32, `_dit`), fp32+fp16 pass | b fp16 373 / 1702 vs MPS 337 / 1912; fp32 357 / 1759 vs 367 / 1812. l fp16 1212 / 3759 vs 1013 / 4548; fp32 1073 / 4343 vs 1086 / 4516. Less memory than MPS. The DINOv3-L backbone overflows float16 (hidden states up to 1.5e5), so the backbone graph always runs in float32 (`fp32_methods`). |
| prompt_depth_anything | depth | portable_mlx, fp32+fp16 pass | 756x1008 fp16 s 299 / 2138 vs MPS 195 / 1588; l 1931 / 7471 vs 1310 / 3301. |
| unidepth_v2 | depth | portable_mlx, fp32 only | fp32 1694 / 6215 vs MPS 1252 / 6077. fp16 not declared (golden). |
| unik3d | depth | portable_mlx, fp32+fp16 pass | fp16 1637 / 4803 vs MPS 1294 / 4038. |
| video_depth_anything | depth | portable_mlx (`_encode`/`_window`), fp32+fp16 pass | 480x640 fp16 s 1846 / 5538 vs MPS 926 / 6506 (less memory, 2x slower); metric-s 1822 / 5537 vs 933 / 6535. |
| zoedepth | depth | portable_mlx, fp32 only | fp32 1297 / 5897 vs MPS 778 / 3558. fp16 gives NaN, so not declared. |
| yolox | detection | portable_mlx + kernels_mlx (conv_mlx), fp32+fp16 pass | tiny 800x1344 fp16 13 / 438 vs MPS 20 / 1492; fp32 15 / 476 vs 20 / 1497. Winograd kernels 12 / 453 fp16, 14 / 511 fp32 (about 8 % faster, declared). l/x pass goldens. |
| rtmdet | detection | portable_mlx, fp32+fp16 pass | tiny fp16 23 / 577 vs MPS 27 / 1545; fp32 30 / 644 vs 29 / 1551. Winograd kernels 21-25 ms but the survey golden failed for fp16, so no kernels mode. |
| dfine / deim | detection | portable_mlx, fp32+fp16 pass | n fp16 23 / 643 vs MPS 25 / 1510; fp32 31 / 667 vs 41 / 1545. |
| rtdetrv2 | detection | portable_mlx, fp32+fp16 pass | s fp16 42 / 922 vs MPS 44 / 1631; fp32 48 / 933 vs 44 / 1650. Needed `avg_pool2d` ceil_mode / divisor_override in `mlx_graph.py`. |
| lwdetr | detection | portable_mlx, fp32+fp16 pass | tiny fp16 35 / 942 vs MPS 41 / 1742; fp32 36 / 970 vs 47 / 1776. Needed `aten.select.int` with negative index. |
| rfdetr (+ keypoint) | detection, pose | portable_mlx, fp32 only | n fp32 22 / 714 vs MPS 32 / 1718; keypoint fp32 90 / 969 vs 148 / 1830. All 8 detection and the keypoint ref pass fp32. fp16 passes n/s/m/l/seg-n/seg-l/seg-xl/seg-2xl but fails rf-detr-seg-s, seg-m (boxes mean 0.37-0.40) and keypoint (keypoint_scores mean 6e-4); `mlx_dtypes` is per card, so fp32 only. Keypoint fp16 77 / 946 was measured. |
| detr | detection | portable_mlx, fp32+fp16 pass | r50 fp16 116 / 1120 vs MPS 105 / 1954; fp32 147 / 1271 vs 125 / 1995 (ResNet-50 conv backbone, slower). Winograd kernels 216-285 ms: slower, not declared. |
| rcnn | detection | portable_mlx (`_dense`: backbone, FPN and RPN heads on MLX; proposals, ROI heads and NMS eager), fp32+fp16 pass | mask-rcnn fp16 412 / 1473 vs MPS 203 / 1931; fp32 486 / 1976 vs 222 / 1930. Speed loses (800x1344 conv net); Winograd 407-457 ms, no real gain, not declared. |
| grounding_dino | detection | portable_mlx (`image` graph), fp32 only | tiny fp32 1524 / 1627 vs MPS 846 / 4787 (3x less memory, 1.8x slower). fp16 passes grounding-dino-tiny and mm-base but fails mm-grounding-dino-tiny (scores p99 0.029); grounding-dino-base cannot be validated (its golden npz does not match golden.toml). |
| owlv2 | detection | portable_mlx (`image` graph), fp32+fp16 pass | b16 fp32 395 / 1227 vs MPS 394 / 1515; fp16 294 / 985 (MPS has no fp16 row). |
| ben2 | matting | portable_mlx, fp32 pass | 1945 / 5335 vs MPS 1148 / 6252 (swin: less memory, slower). |
| birefnet | matting | portable_mlx, fp32+fp16 pass | fp32 2421 / 6331 vs MPS 1333 / 9488; fp16 1736 / 6151 vs 1152 / 5433. |
| inspyrenet | matting | portable_mlx (`_net` method), fp32 only | fp32 1339 / 2759 vs MPS 1108 / 7284. fp16 gives NaN on inspyrenet (both refs pass fp16 on inspyrenet-fast, but the card is per-card). |
| isnet | matting | portable_mlx + kernels_mlx (fp16 only), fp32+fp16 pass | fp32 234 / 1649 vs MPS 157 / 1787. fp16 portable 153 / 1259 vs 130 / 1746; Winograd fp16 128 / 1445 (ties MPS, declared fp16 only). |
| u2net | matting | portable_mlx, fp32+fp16 pass | fp16 62 / 1201 vs MPS 79 / 1885; fp32 83 / 1526 vs 82 / 1910. |
| vitmatte | matting | portable_mlx (`_net` method), fp32+fp16 pass | s 1080x1920 fp32 1531 / 3035 vs MPS 2333 / 5185; fp16 1486 / 2821 vs 1360 / 2542. |
| matanyone | matting | not ported | stateful video model (memory bank, top-k readout, per-frame Python control flow), no fixed-shape graph to export; its golden is skipped in the MLX run. |
| rtmpose / dwpose / rtmw | pose | portable_mlx, fp32+fp16 pass | rtmpose-t fp16 24 / 607 vs MPS 26 / 1460; fp32 27 / 656 vs 26 / 1462. Goldens tried: t, l-384, dwpose-t, rtmw-m, halpe26, hand5, face6, ap10k. |
| rtmo | pose | portable_mlx, fp32 only | t fp32 17 / 658 vs MPS 21 / 1523. fp16 fails (t scores p99 0.0049, l keypoints p99 0.90). |
| hrnet | pose | portable_mlx, fp32+fp16 pass | w32 fp16 125 / 1049 vs MPS 122 / 1675; fp32 166 / 1247 vs 141 / 1748. Winograd: tie, not declared. |
| retinaface | pose | portable_mlx, fp32+fp16 pass | resnet50 fp16 190 / 1266 vs MPS 150 / 1913; fp32 284 / 1521 vs 210 / 1879. Kernels mode fails in float16-only options, not declared. |
| vitpose | pose | portable_mlx, fp32+fp16 pass | base-simple fp16 156 / 987 vs MPS 186 / 1910; fp32 204 / 1326 vs 196 / 2073. plus-small and plus-huge pass goldens. |
| yunet | pose | portable_mlx, fp32+fp16 pass | n fp16 21 / 719 vs MPS 22 / 1377; fp32 28 / 771 vs 25 / 1380. |
| anyline, lineart, lineart_anime, mlsd | edges | portable_mlx, fp32+fp16 pass | anyline fp32 2817 ms / 2045 MB vs MPS 130 / 2346 (memory lower, speed 20x slower: only the TEED net runs as an MLX graph, the line-art post-processing `remove_small` is a long chain of small torch ops that runs eagerly outside it, so it dominates; not fixable without porting that step); lineart-anime fp16 43 / 1032 vs MPS 51 / 1774; mlsd fp16 33 / 715 vs 45 / 1373. lineart fp16 136 / 1170 vs 67 / 1362 (speed loses; Winograd 133, no gain). |
| hed | edges | portable_mlx, fp32 only | fp32 75 / 1120 vs MPS 65 / 1415. fp16 gives NaN. Winograd slower. |
| pidinet | edges | portable_mlx + kernels_mlx (conv_mlx), fp32+fp16 pass | fp16 portable 163 / 759, Winograd 78 / 747 vs MPS 73 / 1349 (kernels declared: 2.1x faster than portable); fp32 200 -> 115 / 987 vs 91 / 1359. |
| teed | edges | portable_mlx, fp32+fp16 pass | fp32 17 ms / 534 MB vs MPS 20 / 1314; fp16 15 / 474 vs 17 / 1313. Wins on both. |
| flowformer | flow | portable_mlx, fp32+fp16 pass | 1080x1920 out of memory on MLX and on MPS (golden passes at the small size). |
| flowseek | flow | portable_mlx, fp32+fp16 pass | t 1080x1920 fp16 2667 / 5208, fp32 4083 / 9096; MPS out of memory. |
| gmflow | flow | portable_mlx (`_flow` method; `index_put` mask assignment added to `mlx_graph.py`), fp32+fp16 pass | scale1-things fp16 6371 / 10879 vs MPS 10676 / 12643; fp32 6846 / 10971 vs 3269 / 11902. |
| megaflow | flow | portable_mlx, fp32+fp16 pass (golden); needed bicubic antialias resize (`_upsample_bicubic2d_aa`, max err 5e-6 vs torch) | 1080x1920: the fp32 survey job exhausted memory and swap (MPS fp32 is out of memory too; MPS fp16 11196 / 8969), so it was stopped and no row is recorded. |
| memflow, memfof | flow | portable_mlx (`_flow` method), fp32+fp16 pass | memflow fp32 16607 / 9749 vs MPS 10016 / 12659; memfof fp32 4267 / 2757 vs 3054 / 5038. |
| neuflow_v2, recover | flow | portable_mlx, fp32 only | neuflow-v2 fp32 1552 / 8929 vs MPS 1039 / 13354 (fp16 mean diff 0.0017); recover-cx fp32 1600 / 1924 vs 1107 / 4781 (the survey's golden check failed at 1080p; pytest golden passes; recover-mn fp16 p99 0.032). |
| raft | flow | portable_mlx, fp32+fp16 pass | large 1080x1920 fp16 3271 / 8609 vs MPS 2710 / 11644; fp32 3592 / 9668 vs 2522 / 12889. Winograd: no gain, fp16 golden failed, not declared. |
| sea_raft, waft | flow | portable_mlx, fp32+fp16 pass | sea-raft-m fp16 1785 / 5111 vs MPS 1635 / 9732; fp32 2973 / 8955 vs 1820 / 12542. waft-a1 fp32 10308 / 3514 vs 8225 / 8183. |
| (summary) detection / matting / pose / edges / flow | all | all families wired except matanyone | Memory is below MPS in nearly every row. Speed is at or above MPS for the transformer and small conv detectors and keypoint nets; the heavy conv and iterative-flow nets stay 1.2-2x slower. Goldens were run on representative refs of each card (smallest, largest, special variants), not every ref; the detection group ran every ref (43 of 48 pass in both dtypes, the rest are covered by the fp32-only declarations above).
| clip | features | portable_mlx (`_forward` graph), fp32+fp16 pass | 224x224 ms / MB. b16-openai fp16 14.5 / 1467 vs MPS 16.1 / 1724; fp32 17.4 / 1605 vs 17.5 / 1563. l14-laion2b fp16 56.1 / 2943 vs 56.6 / 4424; fp32 64.6 / 3558 vs 60.6 / 4426. All 10 refs pass goldens (both dtypes). Needed `Rope` / table caches skipped under export (`libnet/features.py`). |
| siglip2 | features | portable_mlx, fp32+fp16 pass | b16-256 fp16 15.0 / 1298 vs MPS 15.8 / 1563; fp32 17.8 / 1447 vs 17.2 / 1405. so400m14-378 fp16 65.8 / 3844 vs 68.6 / 5553; fp32 74.9 / 4659 vs 75.9 / 5551. g16 (1B) not run: the fp32 twin plus MLX copy pushed swap over the 10 GB limit on the shared Mac; the other 8 refs pass. |
| perception_encoder | features | portable_mlx, fp32+fp16 pass | t16-384 fp16 4.8 / 463 vs MPS 7.9 / 388; fp32 6.2 / 456 vs 7.4 / 379. l14-336 fp32 69.8 / 3724 vs 68.6 / 4527; fp16 97 / 3237 vs 68.7 / 4526 (the interleaved RoPE in fp16 is slower than fp32 on MLX; not tuned). g14 (1.9B) not run, size. 8 of 10 refs pass. |
| anycalib | calibration | portable_mlx (`_rays` graph; fit eager), fp32+fp16 pass | 1080x1920 fp32 311 / 2077 vs MPS 266 / 3023 (less memory, 1.2x slower); fp16 337 / 1480 (MPS declares fp32 only). |
| geocalib | calibration | portable_mlx (`_fields` graph; LM fit eager), fp32+fp16 pass | 1080x1920 fp16 346 / 1079 vs MPS 427 / 1644; fp32 394 / 1226 vs 517 / 1717. Faster and lighter than MPS. |
| rife | interpolation | portable_mlx (`_forward` graph, export-safe cat instead of in-place writes), fp32+fp16 pass | 1080x1920 4.25 fp16 406 / 1707 vs MPS 106 / 2482; lite fp16 298 / 1730 vs 80 / 2447. Lighter than MPS but about 3.5x slower: bilinear grid_sample gathers and no mx.compile (activations above 80 MB). Winograd conv_mlx gave no gain, so no kernels mode. |
| film | interpolation | portable_mlx (`_run` graph), fp32+fp16 pass | 1080x1920 fp16 7046 / 9354, fp32 9944 / 12658 (no MPS row; memory-heavy at 1080p). |
| ema_vfi | interpolation | portable_mlx (`_run` graph), fp32+fp16 pass | small-t 1080x1920 fp16 1201 / 2499 vs MPS 513 / 3583; fp32 1361 / 2371 vs 792 / 6499. Less memory, 1.7-2.3x slower. |
| gimm_vfi | interpolation | portable_mlx (`_forward` graph; added scatter_add and conv3d ops), fp32+fp16 pass | gimm-vfi-r 1080x1920 fp16 30451 / 12185, fp32 58742 / 14977 (no MPS row). Works but very heavy at 1080p on 16 GB. |
| parseq | ocr | portable_mlx (`_decode`: all 26 decoding steps as one graph, no early exit), fp32+fp16 pass | 32x128 fp16 14 / 491 vs MPS 24 / 440; fp32 26 / 537 vs 23 / 496. fp16 faster; memory about 10% above MPS (tiny net, runtime floor). |
| ppocrv5 | ocr | portable_mlx (detector and recogniser graphs; boxes, crops, CTC eager; fp16 neck scaling applied to the twin), fp32+fp16 pass | 1080x1920 mobile fp16 772 / 1433 vs MPS 736 / 2987; server fp16 1955 / 2910 vs 1607 / 6545. Memory 2-2.3x lower, speed 1.05-1.4x slower. |
| boq | retrieval | portable_mlx (`_descriptor` graph over backbone + aggregator; resize eager), fp32+fp16 pass | 480x640 fp32 47 / 905 vs MPS 57 / 987; fp16 52 / 856 vs 48 / 849. About parity. |
| megaloc | retrieval | portable_mlx (`_descriptor` graph), fp32+fp16 pass | 480x640 fp32 53 / 1345 vs MPS 54 / 1926; fp16 83 / 1496 vs 45 / 2617 (fp16 slower than fp32 on MLX here: the Sinkhorn step casts to float32). Less memory; fp32 is the better MLX dtype. |
| tapir | tracking | portable_mlx: whole offline pass as one graph (adds `grid_sampler_3d`, `avg_pool3d` ops); fp32 only (fp16 misses the goldens' tolerances) | 480x854 ms / MB, fp32. tapir 2770 / 1720 vs MPS 1382 / 2630. All variants pass goldens. |
| alltracker | tracking | portable_mlx: per-query-frame group graph (`group` arg replaces data-dependent `unique`/`nonzero`); fp32 only | 480x854 fp32 5397 / 2597 vs MPS 3508 / 4571. |
| locotrack | tracking | portable_mlx: whole pass as one graph; fp32 only | locotrack-b fp32 2039 / 1679 vs MPS 2104 / 2474 (faster and lighter). |
| tapnext | tracking | portable_mlx: whole pass as one graph; fp32 only | tapnext fp32 8155 / 3639 vs MPS 5855 / 3754. |
| cotracker | tracking | portable_mlx: offline = whole pass as one graph; online = window loop eager (data-dependent, `torch.export` guards), CNN and update transformer on MLX; fp32 only | offline fp32 3577 / 3438 vs MPS 1430 / 4725; online 3340 / 3971 vs 1707 / 4689. |
| track_on | tracking | portable_mlx: per-frame loop eager (data-dependent), DINOv3 adapter backbone as graph; fp32+fp16 pass | track-on2 fp32 21453 / 4902 vs MPS 8354 / 6102; fp16 21190 / 4947. Eager per-frame steps dominate; slower than MPS. |
| nnfoundation (cnn) | volume | `mlx_wrap.port(Portable)`: 3D UNet exported whole; 5-D conv via `mx.conv3d` (transposed: matmul for stride == kernel, else `conv_transpose3d`) | 200x200 fp32 5421 ms / 10651 MB vs MPS 3645 / 9435; fp16 4721 / 10094 vs 2935 / 7824; goldens pass. Slower and heavier than MPS: MLX 3-D conv is not tuned for this size. |
| nnfoundation (vit) | volume | same | 192x192 fp32 18038 ms / 4313 MB vs MPS 21342 / 4987; fp16 13037 / 2469 vs 17636 / 3817; goldens pass. Faster and lighter than MPS. |
| nninteractive | volume | same | fp32 4782 ms / 7025 MB vs MPS 3525 / 11439; fp16 4564 / 6823 (MPS fp16 not run); goldens pass. Memory lower, time 1.35x slower. |
| totalsegmentator | volume | `_mlx.port`: each part's sub-nets wrapped with `MlxModule(release=True)`, sliding-window glue stays eager | total fp32 58555 ms / 6627 MB vs MPS 27804 / 10180; fp16 42971 / 6614 vs 23070 / 6058; 3mm fp32 1573 / 5085 vs 593 / 4385; fp16 1333 / 4913 vs 522 / 2588; all goldens pass. Memory lower only for fp32 total; time about 2x slower than MPS (3-D conv). |
| pi3 | multiview | `mlx_wrap.port` of the upstream net (`predict` = `net(x)`); adds `aten._linalg_svd` / `_linalg_det` (camera head rotation orthogonalization, CPU stream) | 336x518 fp32 12062 ms / 5600 MB vs MPS 10298 / 6201; fp16 11173 / 3874 vs 8341 / 6510; goldens pass. Memory lower, time 1.2-1.3x slower. |
| vggt | multiview | same (`_Upstream.predict`) | vggt-1b 336x518 fp32 9359 ms / 9387 MB vs MPS 7209 / 10654; fp16 8774 / 6853 vs 6638 / 8423. vggt-omega-1b-512 fp32 9167 / 7391 vs 7938 / 8221; fp16 8358 / 5294 vs 7242 / 8602. Goldens pass; memory lower, time 1.15-1.3x slower. |
| anysplat | multiview | same | 336x518 fp32 8786 ms / 10565 MB vs MPS 33262 / 10705 (MPS fp32 path is slow); fp16 6974 / 8297 vs 5444 / 8751; goldens pass. fp32 3.8x faster than MPS. |
| depthsplat | multiview | `port(Portable)`, adds `aten.clamp.Tensor` | 336x518 fp32 1707 ms / 2177 MB vs MPS 670 / 8880; golden passes in fp32 only (fp16 confidence mean diff 0.016 > 0.014, so only fp32 is declared). Memory 4x lower, time 2.5x slower. |
| mvsplat | multiview | `port(Portable)` | 336x518 fp32 4550 ms / 3460 MB vs MPS 1077 / 4523; fp16 4527 / 3737 vs 1112 / 4537; goldens pass. Memory lower; 4x slower (the 3-D cost-volume warp graph is large and mostly gathers). |
| resplat | multiview | only `net.encode` on MLX (the Gaussian-rendering refinement loop is data-dependent and cannot be exported; it stays eager on torch CPU) | base-re10k 480x640 fp32 112309 ms / 12071 MB; MPS row missing (MPS does not fit the survey input). Goldens (base, small) pass in fp32. Declared fp32 only; practically unusable for speed: the eager refine dominates. |
| lingbot-map | multiview | `methods=("run",)`: the whole streaming pass incl. KV cache unrolled into one exported graph | 480x640 fp32 11010 ms / 9090 MB vs MPS 8840 / 13055; fp16 10046 / 8220 vs 7125 / 8834; goldens pass. Memory lower, time 1.25-1.4x slower. |
| mapanything | multiview | `methods=("predict",)` (transformer, DPT, heads and edge mask in one graph) | mapanything-apache 336x518 fp32 9091 ms / 11207 MB vs MPS 7357 / 11638; fp16 8284 / 8402 vs 6369 / 7218 (+1.2 GB); goldens pass. |
| mvsanywhere | multiview | `methods=("encode_image", "matching", "cv_encoder", "decode")`; cost volume stays eager torch | 448x672 fp32 3661 ms / 2475 MB vs MPS 1051 / 3534; fp16 3609 / 2899 vs 1064 / 3801; goldens pass. Memory lower; 3.4x slower (eager CPU cost volume). |
| patchmatchnet | multiview | `methods=("prepare", "predict")`; seeded random hypotheses drawn outside the trace (constants); adds stride-1 `avg_pool3d` | 1536x1024 fp32 2841 ms / 3155 MB vs MPS 1010 / 7593; fp16 2856 / 3625 vs 1574 / 6591; goldens pass. Memory 2x lower, 2.8x slower. |
| sam / sam-hq | promptable_segmentation | `port(Portable, methods=("_enc",))`: the ViT image encoder (and HQ features) on MLX after the eager torch resize; the prompt encoder and mask decoder stay torch | sam-vit-b 1024x1024 fp32 659 ms / 2036 MB vs MPS 631 / 4362; fp16 640 / 1935 vs 600 / 3762. sam-hq-vit-b fp32 691 / 2372 vs 645 / 4374; fp16 689 / 2253 vs 683 / 4542. Goldens pass for all b/l/h variants. Memory about 2x lower, speed within 5 percent of MPS. |
| mobile-sam | promptable_segmentation | same (`_enc` = TinyViT trunk + norm) | fp32 100 ms / 947 MB vs MPS 81 / 1480. fp16 declared off: golden fails (pred_masks mean diff 0.49). Memory 36 percent lower, 1.2x slower. |
| efficientvit-sam | promptable_segmentation | same; `compile=False` (a new `port` option): the fused elementwise kernels exceed Metal's argument-buffer limit under `mx.compile` | l0 fp32 116 ms / 1064 MB vs MPS 51 / 1779. Goldens pass. Memory lower, 2.3x slower. |
| sam2 | promptable_segmentation | same (`_enc` = Hiera trunk, neck) | hiera-tiny fp32 219 ms / 1139 MB vs MPS 241 / 2581: faster and 2.3x lighter. fp32 only declared (hiera-base-plus fp16 misses its golden: pred_masks p99 diff 0.0091 > 0.009). |
| sam3 | promptable_segmentation | `port(Portable, methods=("embed",))`; also fixes `aten.upsample_bilinear2d` antialias on integer images (uint8 resize matmul) | 1024x1024 fp32 2774 ms / 5023 MB vs MPS 3620 / 5807; fp16 2423 / 4198 vs 3688 / 3523. Goldens pass. 25-35 percent faster than MPS; fp16 uses more memory than MPS (peak at load). |
| clipseg | promptable_segmentation | `methods=("embed","decode")` | 1024x1024 fp32 33 ms / 932 MB vs MPS 45 / 1532; fp16 39 / 984 vs 38 / 1561. Goldens pass. Faster and lighter. |
| insid3 | promptable_segmentation | `methods=("embed",)` (DINOv3 ViT-L backbone); the reference matching stays eager | vitl16 fp32 3244 ms / 2981 MB vs MPS 2953 / 2963 (parity). fp16 declared off: similarity map NaN. |
| dedode | matching | `net.detector` + `net.descriptor` on MLX, fp32 only (fp16 keypoints mean diff 0.0022 > 0.002) | v2-b 2915 ms / 4374 MB vs MPS 1464 / 8011; v2-g 4967 / 5662 vs 3767 / 10008. Memory lower, speed slower than MPS. |
| loftr | matching | backbone + coarse transformer on MLX, fine stage and matching eager; fp32 only (indoor fp16 recall 0.9976 < 0.998) | outdoor 1979 ms / 3802 MB vs MPS 875 / 5767. Memory lower, speed slower. |
| loma | matching | detector, descriptor, matcher on MLX, fp32 (golden only for loma-b) | loma-b 4152 ms / 6133 MB vs MPS 3037 / 8147. Memory lower, speed slower. |
| romav1 | matching | whole net on MLX, fp32; `cholesky` became `cholesky_ex(check_errors=False)` to avoid the with_effects op; new CPU-stream cholesky ops in mlx_graph | 4500 ms / 5242 MB vs MPS 3050 / 7979. Memory lower, speed slower. |
| xfeat | matching | `net.xfeat.net` on MLX, fp32 (LighterGlue stays eager); new `aten.unfold` mapping | 158 ms / 543 MB vs MPS 42 / 1471. Memory lower, speed slower (small net, eager post-processing dominates). |
| flexglue (aliked) | matching | descriptor net + matcher decode on MLX, fp32 + fp16; wrapping ALIKED dense map rejected (keypoint recall 0.963 < 0.99) | fp32 2249 ms / 3048 MB vs MPS 1124 / 3932; fp16 2255 / 2970 vs 1069 / 3088. Memory lower, speed about 2x slower. |
| lightglue | matching | not ported: data-dependent keypoint counts in the matcher, and the ALIKED dense map misses recall (as in flexglue) | n/a |

## Conv3x3 kernel experiments (M1 Pro, fp16, 1x512x512x64 -> 64, 3x3, pad 1)

Microbenchmarks against `mx.conv2d` (5.5 ms, about 3.5 TFLOPS; MPS 4.1 ms). None was wired into `kernels_mlx`.

| variant | ms | note |
|---|---|---|
| `mx.conv2d` | 5.5 | baseline |
| simdgroup_matrix implicit GEMM, 16 px x 64 oc per simdgroup, B loaded from device (transposed) | 9.4 | B reloaded by each simdgroup |
| same, weights staged per tap in threadgroup memory | 5.3 | needs a pre-padded input (extra pass, ~0.3 ms), so no net gain |
| 32 px x 64 oc per simdgroup | 464 | register spill |
| 32 px x 32 oc per simdgroup, 2x2 simdgroups | 5.4 | no gain |
| Winograd F(2,3) lower bound | >5.5 | the 16 batched matmuls alone take 2.4 ms; a transform pass over V takes 1.7 ms each, two needed unfused |

Conclusion: the direct kernels sit at the same ALU/bandwidth ceiling as MLX's conv, so they cannot reach MPS (likely Winograd-based). Only a fused Winograd kernel (input transform, 16 tile GEMMs and output transform kept in threadgroup memory) could win; it was not built. The conv-bound families (compact, span, realplksr, realesrgan, codeformer, gfpgan) stay slower than MPS on MLX and keep `device="mps"` routing to MPS.

### Winograd conv (`kernels/conv_mlx`, kernels mode)

Three dispatches per 3x3 stride-1 convolution: fused input transform (`B^T d B`, fp32 registers, NHWC), 16 (F(2,3)) or
36 (F(4,3)) batched `mx.matmul`s against weights transformed once at load, fused output transform with bias, ReLU /
PReLU and optional residual. The transforms need `UR` (full loop unrolling) to be fast and 2 channels per thread for
F(4,3) (4 spills). `FF_WINO_M=2|4` picks the tile (default 4). `_conv_core` takes it for groups 1, 3x3, stride 1,
dilation 1, even channels >= 16, h*w >= 48^2; everything else stays `mx.conv2d`. In kernels mode, a conv whose only
consumer is ReLU / decomposed PReLU applies the activation in the output transform.

Microbench (M1 Pro, 1 image, NHWC; `mx.conv2d` includes the unfused bias + activation), ms:

| shape | dtype | mx.conv2d | F(2,3) | F(4,3) |
|---|---|---|---|---|
| 512^2 64->64 | fp16 | 6.1-7.4 | 4.5 | 2.9-3.0 |
| 256^2 128->128 | fp16 | 3.8-4.2 | 3.2 | 2.1 |
| 128^2 256->256 | fp16 | 2.9-3.2 | 2.7 | 2.0 |
| 1024^2 16->16 | fp16 | 6.2 | 6.2 | 3.9 |
| 512^2 64->64 | fp32 | 7.8-9.5 | 7.2 | 4.6 |
| 256^2 128->128 | fp32 | 4.8-5.8 | 5.0 | 3.1 |
| 128^2 256->256 | fp32 | 3.5-4.0 | 3.3 | 2.1 |

MPS for 64->64 at 512^2 is 4.1 ms, so F(4,3) wins there too (torch MPS microbench for the other shapes not run).
3->64 and 64->3 have too few channels and stay `mx.conv2d`. Accuracy (relative RMS vs fp32): F(4,3) fp16 3.8e-3
(direct fp16 3.6e-4, F(2,3) 6.9e-4), fp32 1.7e-6. Stage split at 512^2/64 fp16 F(4,3): input transform 0.9 ms,
matmuls 1.4 ms, output transform 0.85 ms (V and M are 75 MB each: bandwidth bound).

End to end (512x512 unless noted, ms, fp32 / fp16; MPS from `apple-m1-pro-16gb.jsonl`):

| family | Winograd kernels | portable MLX | MPS | verdict |
|---|---|---|---|---|
| realesr-general-x4v3 (compact) | 176 / 103 | 340 / 229 | 178 / 132 | wins (fp16 1.28x faster than MPS, fp32 tie); declared |
| span-x4 | 184 / 103 (fp32 declared as portable: 216) | 172-216 / 131-158 | 103 / 89 | fp16 declared (`kernels_fp16_only` in `mlx_wrap.port`: fp32 runs the portable path, since a card's `mlx_dtypes` must start with float32); still slower than MPS |
| realplksr | 2974 / 1724 (F(2,3): 3697 / 2239, golden passes) | 2374 / 1879 | 1527 / 1092 | slower than portable with either tile; not declared |
| realesrgan-x4plus | 11460 / 5557 | 8130 / 5914 | 3022 / 2066 | V/M of 1024^2-2048^2 activations blow memory; not declared |
| big-lama | 337 / 322 | 277 / 286 | 241 / 214 | slower; not declared |
| deeplabv3-r101-d8 (1024x2048) | 654 / 456 | fp16 469 | fp16 528 | tie; not declared |

Small shapes (fp16 ms, mx.conv2d + ReLU vs Winograd): 64^2x256 1.70 vs 0.75, 32^2x512 1.93 vs 0.92, 64^2x128 0.76 vs 0.63,
128^2x64 0.76 vs 0.73, 96^2x128 0.79 vs 0.96, 256^2x32 0.72 vs 1.09, 512^2x32 2.21 vs 2.04: it needs >= 64 channels and
a large image to win clearly; 32-channel layers tie or lose.

Banding (`bands=`, tile rows processed in bands, concatenated): 2.8 -> 3.3-3.5 ms at 512^2/64 and slower for every
other shape, so it gives no cache benefit (the concat copy and extra dispatches cost more). It is only used when one
layer's transformed tensor would exceed 192 MB (bounds the peak of 2048^2-class images; not measured on realesrgan
because the shared Mac was at 18 GB swap).

Not attempted: the fused simdgroup GEMM + output transform. Each 8x8 tile of 36 accumulators reuses every loaded fragment
once, so the GEMM would run well below the 3.3 TFLOPS the batched `mx.matmul` reaches (1.44 ms for 4.8 GFLOP); the
0.75 ms it could save is likely lost.

Findings: Winograd wins where the per-layer activations are small enough for V and M to stay cache-friendly and where
convolutions dominate (compact, 64 channels at 512^2). Big activations are bound by the V/M round trip (about
0.75 ms per 512^2/64 layer). Remaining: a fused GEMM + output transform (simdgroup_matrix, accumulate the 36 frequencies
in registers) to drop M, band-splitting for huge activations, F(2,3) for the fp16 realplksr golden, leaky_relu and
residual fusion.


## Memory pass (commits 4dfb4ac5 .. c50d549e)

Causes found: the exported graph's weights were held twice (PyTorch twin plus MLX copy) and the twin's freed pages stayed resident; lazy replay without checkpoints kept far more than the live set; whole-graph `mx.compile` doubled the peak of megapixel dense-prediction graphs; the segmentation logits were resized in several full-size passes.

Changes (all in `shared/mlx_graph.py`, `shared/mlx_seg.py`, `shared/mlx_wrap.py`, `kernels/graph_mlx/ops.py`):
- twin parameters are replaced by zero stand-ins after the MLX copy exists, and storages no other tensor shares are handed back to the OS (`madvise(MADV_FREE_REUSABLE)`);
- constants and the BN fold are built lazily;
- segmented replay: compile per segment, `mx.eval` every 256 MB of new activations, dead values dropped by liveness; whole-graph compile only when no activation exceeds 80 MB;
- fused Metal bilinear resize, and a fused two-step resize plus argmax for segmentation (84 to 36 ms, 1678 to 37 MB peak on the largest case; 1 pixel in 2M differs on the 19-class case from FMA rounding).

Survey, fp16 portable, `ms / peak footprint MB` (before = earlier rows in the data file):

| model | before | after | MPS |
|---|---|---|---|
| midas-dpt-beit-large-512 | 668 / 4117 | 574 / 2699 | 635 / 2407 |
| moge-2-vits-normal | 1148 / 6234 | 949 / 2500 | 857 / 1679 |
| segformer-b5-ade20k | 505 / 3097 | 341 / 1518 | 375 / 1974 |
| deeplabv3-r101-d8-ade20k | 469 / 2804 | 493 / 2029 | 528 / 2010 |
| segnext-mscan-b-ade20k | 284 / 2150 | 264 / 1229 | 184 / 1659 |
| lotus-normal-g-v1-1 | 2089 / 9398 | 2757 / 6413 (survey before the large-weights checkpoint rule) | 1553 / 6632 |
| marigold-depth-v1-1 | 4145 / 9333 | 6347 / 6415 (probe after the rule: about 4800 / 6411) | 2498 / 6640 |
| lingbot-depth | 4588 / 8528 | 4502 / 4096 | 3376 / 3228 |

Still above MPS: moge (export-phase CPU transient and the ~500 MB Metal cache, conv-heavy head), midas memory (+300 MB), lingbot memory, segnext portable speed (depthwise-strip convs, 259 ms vs MPS 184; the kernels mode runs 128 ms / 1278 MB and beats MPS). Lotus and marigold now sit below MPS memory (their peak is the float32 twin's load, about 6.4 GB) but are 1.8x slower than MPS. Segment evals cost them speed, so nets with weights of 1 GB or more checkpoint every min(weights, 2 GB) instead of 256 MB (marigold probe 5840 to 4800 ms, same peak). Goldens (including nafnet-gopro, wd-vit-tagger-v3, swin, segformer, depth-anything-v2, mask2former) pass.

## Memory pass (part 2): lazy twin, small buffer cache, lean group norm

Changes (all goldens of the spread below pass; no `LazyTwinError` fallback seen):

- **Lazy fp32 twin** (`mlx_wrap._lazy_twin`): the twin's `load_state_dict` assigns the mmapped checkpoint tensors (clean file
  pages are not counted in the process footprint) and `torch.nn.init` is skipped. A tensor still skipped and unwritten at
  the end raises `LazyTwinError` and the runner is rebuilt the eager way. Lotus load peak 4002 -> 1544 MB.
- **MLX buffer cache limit** 512 -> 128 MB on 16 GB (`Tuning`: fraction 1/128, min 128): -250..-450 MB peak, speed within
  1% (swinir +8%, CREStereo kernels +12% for -870 MB; the stereo case can be re-raised with `FASTFORKS_MLX_TUNING`).
- **Group norm** through `mx.fast.layer_norm` (no float32 copy of the activation): lotus 2.7 -> 2.2 s per call.
- `MlxModule._release` does not `madvise` file-mapped storages.

Survey, portable, MLX before -> after vs MPS (ms / peak MB):

| ref | dtype | MLX before | MLX after | MPS |
|---|---|---|---|---|
| lotus-normal-g | fp16 | 2757 / 6413 | 2152 / 4734 | 1553 / 6632 |
| lotus-normal-g | fp32 | 5081 / 8890 | 3376 / 8003 | 2886 / 9700 |
| lotus-normal-d | fp16 | 2166 / 9159 | 2197 / 4729 | 1459 / 6628 |
| marigold-depth | fp16 | 6347 / 6415 | 4231 / 4254 | 2498 / 6640 |
| marigold-depth | fp32 | 7195 / 8258 | 5404 / 7509 | 2970 / 9709 |
| marigold-normals-lcm | fp16 | 2104 / 9797 | 2235 / 4255 | 1655 / 6671 |
| stablenormal-turbo | fp16 | 2661 / 11417 | 2527 / 7816 | 1690 / 8985 |
| lingbot-depth | fp16 | 4502 / 4096 | 4567 / 2968 | 3376 / 3228 |
| lingbot-depth | fp32 | 4704 / 4615 | 4375 / 4209 | 3363 / 4011 |
| midas-dpt-beit-large-512 | fp16 | 574 / 2699 | 562 / 1886 | 635 / 2407 |
| moge-2-vits-normal | fp16 | 949 / 2500 | 877 / 2046 | 857 / 1679 |
| moge-2-vits-normal | fp32 | 985 / 2509 | 944 / 1822 | 361 / 2634 |
| deeplabv3-r101-d8 | fp16 | 493 / 2029 | 467 / 1604 | 528 / 2010 |
| segformer-b5-ade20k | fp16 | 341 / 1518 | 343 / 1373 | 375 / 1974 |
| pointdit-b-512 | fp16 | 373 / 1702 | 355 / 1519 | 337 / 1912 |
| pointdit-l-512 | fp16 | 1212 / 3759 | 1084 / 3352 | 1013 / 4548 |

Still above MPS in memory: moge-2-vits fp16 (+370 MB: call-phase activations, mx peak 1.3 GB at 994x1482) and
lingbot-depth fp32 (+200 MB). The SD-family (lotus, marigold, stablenormal) is 1.3-1.7x slower than MPS (time, not memory):
their resident heap beyond MLX buffers is about 650 MB and the first call pays the mmap page-in. Not done: duplicate
Winograd / folded-BN weights, fp32 upcasts in `_layer_norm` / `_vector_norm`, segment-size policy.

## Stereo group (994x1482, portable)

All 27 families with a `portable_mlx` pass `test_mlx[<ref>@1]` (cfnet: fp32 only, fp16 misses the tolerance; defom fp32 only). Wrapped with `mlx_wrap.port` (`whole="disparity"`, or `methods=("_run",)` for psmnet / s2m2). Enablers: functional cost volumes under `torch.compiler.is_exporting()` (`stereo/_volume.py`, avoids the slice_scatter copy chain: psmnet 22 s -> 4.5 s), phase-wise `conv_transpose3d` (`mx.conv_transpose3d` is ~18x slower: 167 ms -> 9 ms per 3x3x3 s2 layer), grouped conv3d, ops min.dim / any / all / adaptive_max_pool2d / empty_permuted, unbatched max_pool2d, cond-free DLNR refinement (`EXPORTING_FOR_MLX`).

Result: peak memory is below MPS for nearly every family (fp32 mostly 40-60 percent lower), but speed is 1.1-2.6x slower than MPS, so the speed target is NOT met. Memory is above MPS for gwcnet fp16 and psmnet (the 5-D volume activations). Not done: a fused correlation / cost-volume Metal kernel, conv_mlx / graph_mlx kernels for the stereo nets (no `mlx_kernels` declared), Winograd for the 3-D nets; c-foundationstereo (60 s) and foundationstereo (36 s) are 12 GB peak and dominated by 3-D conv hourglasses.

| ref | dtype | MPS | MLX |
|---|---|---|---|
| aanet-sceneflow | 16 | 1141 ms / 3.5 GB | 2176 ms / 2.7 GB |
| aanet-sceneflow | 32 | 912 ms / 3.5 GB | 2348 ms / 2.7 GB |
| bridgedepth-sf | 16 | 3782 ms / 6.1 GB | 5540 ms / 3.9 GB |
| bridgedepth-sf | 32 | 4280 ms / 8.1 GB | 5425 ms / 5.3 GB |
| c-foundationstereo-s | 16 | - | 60417 ms / 11.8 GB |
| c-foundationstereo-s | 32 | - | 64038 ms / 11.9 GB |
| cas-gwcnet-gc-sceneflow | 16 | 2995 ms / 7.0 GB | 8431 ms / 4.8 GB |
| cas-gwcnet-gc-sceneflow | 32 | 5490 ms / 11.6 GB | 8673 ms / 4.8 GB |
| cfnet-sceneflow | 32 | - | 5051 ms / 4.2 GB |
| coex-sceneflow | 16 | 411 ms / 3.5 GB | 560 ms / 2.5 GB |
| coex-sceneflow | 32 | 784 ms / 5.1 GB | 763 ms / 2.6 GB |
| defom-stereo-vitl-sceneflow | 32 | 11105 ms / 12.8 GB | 16523 ms / 6.7 GB |
| dlnr-sceneflow | 16 | 2143 ms / 4.0 GB | 4474 ms / 3.0 GB |
| dlnr-sceneflow | 32 | 2699 ms / 6.0 GB | 5411 ms / 3.1 GB |
| fast-acvnet-generalization | 16 | 478 ms / 2.7 GB | 735 ms / 1.9 GB |
| fast-acvnet-generalization | 32 | 944 ms / 3.6 GB | 853 ms / 1.8 GB |
| foundationstereo-vit-s | 16 | 20584 ms / 13.3 GB | 36138 ms / 12.1 GB |
| foundationstereo-vit-s | 32 | - | 40293 ms / 12.1 GB |
| great-igev-sceneflow | 16 | 11062 ms / 9.7 GB | 20253 ms / 9.5 GB |
| great-igev-sceneflow | 32 | 32187 ms / 12.4 GB | 22071 ms / 9.7 GB |
| gwcnet-gc-sceneflow | 16 | 2455 ms / 6.9 GB | 4974 ms / 7.7 GB |
| gwcnet-gc-sceneflow | 32 | 4461 ms / 12.1 GB | 5990 ms / 11.6 GB |
| hitnet-eth3d | 16 | 501 ms / 5.4 GB | 957 ms / 2.6 GB |
| hitnet-eth3d | 32 | 484 ms / 5.4 GB | 1147 ms / 2.4 GB |
| igev-plusplus-sceneflow | 16 | 8602 ms / 5.9 GB | 15900 ms / 3.4 GB |
| igev-plusplus-sceneflow | 32 | 10424 ms / 8.9 GB | 17620 ms / 5.0 GB |
| igev-stereo-sceneflow | 16 | 5772 ms / 3.7 GB | 9039 ms / 3.0 GB |
| igev-stereo-sceneflow | 32 | 6423 ms / 5.7 GB | 10763 ms / 3.1 GB |
| las2-s | 16 | 192 ms / 3.5 GB | 670 ms / 1.4 GB |
| las2-s | 32 | 245 ms / 3.9 GB | 945 ms / 2.3 GB |
| mocha-v1-sceneflow | 16 | 7967 ms / 5.6 GB | 10631 ms / 3.3 GB |
| mocha-v1-sceneflow | 32 | 9065 ms / 8.5 GB | 12425 ms / 3.4 GB |
| monster-plusplus-mix-all | 16 | 11876 ms / 7.2 GB | 19222 ms / 5.4 GB |
| monster-plusplus-mix-all | 32 | 14519 ms / 10.8 GB | 17403 ms / 6.3 GB |
| monster-sceneflow | 16 | 11368 ms / 6.2 GB | 18585 ms / 5.6 GB |
| monster-sceneflow | 32 | 13426 ms / 8.7 GB | 16887 ms / 6.1 GB |
| nmrf-sceneflow | 16 | 1469 ms / 4.6 GB | 2939 ms / 1.8 GB |
| nmrf-sceneflow | 32 | 1560 ms / 5.6 GB | 3499 ms / 2.0 GB |
| psmnet-sceneflow | 16 | 2579 ms / 6.3 GB | 4461 ms / 7.5 GB |
| psmnet-sceneflow | 32 | 4615 ms / 11.0 GB | 5625 ms / 11.2 GB |
| raft-stereo-middlebury | 16 | 3940 ms / 4.6 GB | 9857 ms / 3.0 GB |
| raft-stereo-middlebury | 32 | 4395 ms / 4.6 GB | 11969 ms / 3.2 GB |
| rt-monster-plusplus-mix-all | 16 | 682 ms / 6.9 GB | 1326 ms / 3.5 GB |
| rt-monster-plusplus-mix-all | 32 | 663 ms / 6.8 GB | 1363 ms / 3.6 GB |
| s2m2-s | 16 | 1607 ms / 5.9 GB | 3386 ms / 2.9 GB |
| s2m2-s | 32 | 1707 ms / 7.9 GB | 3877 ms / 4.4 GB |
| selective-igev-sceneflow | 16 | 6968 ms / 3.8 GB | 13071 ms / 3.5 GB |
| selective-igev-sceneflow | 32 | 9010 ms / 6.8 GB | 15360 ms / 3.2 GB |
| waft-stereo-s | 16 | 6090 ms / 9.0 GB | 6635 ms / 4.2 GB |
| waft-stereo-s | 32 | 3360 ms / 9.4 GB | 6903 ms / 4.2 GB |
| crestereo-eth3d | 16 | 4484 ms / 2.7 GB | 4308 ms / 2.5 GB |
| crestereo-eth3d | 32 | 5502 ms / 3.7 GB | 5763 ms / 3.8 GB |

## Stereo speed and memory pass (MLX vs MPS)

Goal: MLX faster than MPS and lower memory on the stereo group. Profiled with
`FF_MLX_PROFILE=1` (per-op, no compile, so elementwise cost is overstated).

Changes: functional cost-volume builders under export (cfnet, fast-acvnet,
las2, cas-gwcnet), fused Metal kernels `grid_sample`, `trilinear`
(3D upsample), `gwc_volume`, `upsample_soft_argmax` (upsample + softmax +
soft-argmin in one pass), `channel_stats` (instance norm), and a `kernels_mlx`
runner for raft-stereo (`conv_mlx`).

| Model | dtype | MLX before | MLX after | MPS |
|---|---|---|---|---|
| psmnet-sceneflow | 16 | 4461 ms / 7.6 GB | 2606 ms / 3.0 GB | 2579 ms / 6.4 GB |
| psmnet-sceneflow | 32 | 5625 ms / 11.5 GB | 3595 ms / 5.5 GB | 4615 ms / 11.3 GB |
| gwcnet-gc-sceneflow | 16 | 4974 ms / 7.9 GB | 2960 ms / 2.7 GB | 2455 ms / 7.1 GB |
| gwcnet-gc-sceneflow | 32 | 5990 ms / 11.9 GB | 3956 ms / 4.8 GB | 4461 ms / 12.4 GB |
| cas-gwcnet-gc-sceneflow | 16 | 8431 ms / 4.9 GB | 4586 ms / 4.9 GB | 2995 ms / 7.2 GB |
| cas-gwcnet-gc-sceneflow | 32 | 8673 ms / 4.9 GB | 5088 ms / 4.8 GB | 5490 ms / 11.9 GB |
| las2-s | 16 | 670 ms / 1.4 GB | 242 ms / 1.0 GB | 192 ms / 3.6 GB |
| las2-s | 32 | 945 ms | 300 ms | 245 ms |
| fast-acvnet | 16 | 735 ms / 2.0 GB | 581 ms / 1.8 GB | 478 ms / 2.7 GB |
| fast-acvnet | 32 | 853 ms | 754 ms | 944 ms |
| raft-stereo (kernels) | 16 | 9586 ms | 7745 ms | 3637 ms |

MLX now wins on fp32 for psmnet, gwcnet, cas-gwcnet and fast-acvnet, and on
memory for every family listed. Still slower than MPS (fp16): the conv-bound
nets (aanet, hitnet, s2m2, raft-stereo, igev family, monster family, nmrf,
dlnr, cfnet). MLX 3x3 conv2d and conv3d plateau around 2-3 TFLOPS, below MPS,
and Winograd or conv2d decompositions for conv3d did not help. Remaining
profile hot spots: hitnet 1-D grid_sample on stride-2 tensors, aanet
deformable-conv `mul`, s2m2 GELU and convs, raft GRU gates and correlation
lookup (candidate fused kernels). `apple_mlx.json` routing is unchanged: only
crestereo-eth3d wins.

## Compile policy and kernels declarations (v0.21 follow-up)

`mx.compile` is now decided per segment instead of per graph: segments whose largest activation exceeds
`SEG_COMPILE_MAX_MB` (200) run eagerly, and a segment with an activation above `TUNE_MIN_MB` (16) is timed eagerly and
compiled once on its first call and keeps the compiled form unless it is more than 3 % slower or peaks 25 % higher.
`FF_MLX_NO_TUNE=1` disables the tuning, `FF_MLX_SEG_MB` moves the size gate.

Portable fp16, 994x1482 (ms / peak MB, this pass, MLX only; MPS values are in the table above):

| Model | portable | kernels (`conv_mlx`) | declared |
|---|---|---|---|
| aanet-sceneflow | 1533 / 1152 | 1800 / 2395 | no |
| s2m2-s | 2775 / 2047 | 2676 / 3594 | no |
| nmrf-sceneflow | 2674 / 1131 | 2474 / 1145 | no (memory equal) |
| dlnr-sceneflow | 4300 / 2068 | 3564 / 4040 | no |
| mocha-v1-sceneflow | 9795 / 2208 | 9484 / 3786 | no |
| rt-monster-plusplus-mix-all | 1063 / 2250 | 1013 / 2069 | yes |
| igev-stereo-sceneflow | 8848 / 2385 | 7682 / 2542 | no |
| bridgedepth-sf | 5079 / 3339 | 4790 / 3270 | no (golden fails in kernels) |
| coex-sceneflow | 522 / 1827 | 510 / 1831 | no (tie) |
| waft-stereo-s | 5708 / 3446 | 6206 / 3541 | no |
| hitnet-eth3d | 591 / 996 | 720 / 2010 | no |
| selective-igev-sceneflow | 12598 / 3613 | 11854 / 3309 | yes |
| igev-plusplus-sceneflow | 16198 / 2531 | 9871 / 2697 | yes (speed, memory +6 %) |
| monster-sceneflow | 17723 / 4332 | 16998 / 7166 | no |
| great-igev-sceneflow | 19870 / 8861 | 18778 / 8949 | no |

The compile policy alone gave about 10-20 % on aanet, s2m2, nmrf, hitnet and coex with lower memory (coex 525 ms /
1.8 GB, hitnet 592 ms / 1.0 GB). Winograd in kernels mode mostly doubles peak memory (transformed V/M tensors), so only
the rows above that win are declared. `apple_mlx.json` is unchanged. Not done: fused GRU gates and correlation lookup,
deformable-conv sample*mask kernel, hitnet 1-D grid_sample, sibling-conv merge.

## Iterative-refinement nets (RAFT-style): buffer cache, not kernels

Profile (`FF_MLX_PROFILE=1`, raft-stereo fp16 kernels, 994x1482): the per-iteration cost was 175 ms against about 40 ms
for the same convolutions micro-benchmarked alone (Winograd 384->128 at 256x376: 6.9 ms; the profile showed 30 ms). Top
ops: GRU 3x3 convs (384->128) 2.8 s, flow-head conv 0.6 s, elementwise 0.5 s, cat 0.5 s, encoder convs and batch-norm
stats 1.2 s. The cause was the MLX buffer cache (128 MB limit since the memory pass): the loop allocates the same
50-170 MB temporaries (Winograd V/M, cats) every step, and each one became a fresh Metal allocation with zero-filled
pages. Raising the limit was worth more than any fusion (raft-stereo kernels, ms by cache MB: 128 7300, 512 6700,
1024 5300, 1536 4800, 2048 4000). Layout (NCHW views of NHWC data) was checked and costs under 1.2 ms per op, and cat is
1.3 ms, so fused GRU gates / merged convz+convr were not written: `mx.compile` already fuses the elementwise part
and the convs sit at the conv ceiling (FLOP-bound, see the conv3x3 experiments).

Change: `mlx_wrap.port(..., cache_mb=2048)` raises the cache limit when the runner is built; declared for raft-stereo,
igev-stereo, raft, sea-raft, memflow, neuflow-v2. Goldens pass (also nafnet-gopro, wd-vit-tagger-v3). Survey, ms / peak MB, before -> after (MPS in brackets):

| ref | mode | dtype | before | after | MPS |
|---|---|---|---|---|---|
| raft-stereo-sceneflow | kernels | fp16 | 7745 / 2772 | 4181 / 4678 | 3637 / 4720 (kernels) |
| raft-stereo-sceneflow | kernels | fp32 | 10247 / 3947 | 5818 / 5590 | - |
| raft-stereo-sceneflow | portable | fp16 | 9586 / 2885 | 5772 / 4708 | 3938 / 4695 |
| raft-stereo-sceneflow | portable | fp32 | 11477 / 2989 | 8090 / 4729 | 4396 / 4713 |
| igev-stereo-sceneflow | portable | fp16 | 9039 / 3041 | 6026 / 4931 | 5772 / 3819 |
| igev-stereo-sceneflow | portable | fp32 | 10763 / 3195 | 6812 / 4599 | 6423 / 5872 |
| raft-large | portable | fp16 | 3271 / 8609 | 2147 / 11100 | 2710 / 11644 |
| raft-large | portable | fp32 | 3592 / 9668 | 2735 / 11424 | 2522 / 12889 |
| sea-raft-m | portable | fp16 | 1785 / 5111 | 1395 / 6741 | 1635 / 9732 |
| sea-raft-m | portable | fp32 | 2973 / 8955 | 2209 / 10723 | 1820 / 12542 |
| neuflow-v2 | portable | fp32 | 1552 / 8929 | 1104 / 10753 | 1039 / 13354 |
| memflow | portable | fp32 | 16607 / 9749 | 12980 / 11371 | 10016 / 12659 |

Result: raft-large fp16 and sea-raft-m fp16 now beat MPS in speed and memory; igev-stereo fp32 beats MPS in both;
raft-stereo kernels fp16 is within 15 % of MPS speed at equal memory. Still slower than MPS: raft-stereo fp32 and
portable, igev fp16 (also +1.1 GB memory), neuflow, memflow, sea-raft fp32 (1.2-1.5x). Memory rose 1.6-2 GB because
the cache is now resident. Remaining: the GRU convs are FLOP-bound at about 3 TFLOPS effective (MPS is faster per
conv), so only a faster conv (fused Winograd GEMM) or fewer allocations (smaller V/M via banding) would close the gap;
a per-family cache size below 2048 (igev, 768) was not tuned. For torch MPS: nothing trivial (its allocator caches).

## Buffer-cache sizing and allocator-bound check (v0.21 follow-up 2)

`FASTFORKS_MLX_CACHE_MB` sets the exact cache limit of a run (also below a declared `cache_mb`; sweeps only);
`FF_SURVEY_DTYPE=float16` limits `tools/speed_survey.py` to one dtype; `FASTFORKS_MLX_TUNING=cache_min_mb=1024` raises the
default limit for any family without a declaration.

Cache sweep of the nets declared at 2048 MB (fp16 except neuflow / memflow fp32; ms / peak MB; MPS in the last column):

| ref | 128 | 256 | 512 | 768 | 1024 | 1536 | 2048 | MPS | declared |
|---|---|---|---|---|---|---|---|---|---|
| raft-stereo-sceneflow kernels | 7745 / 2772 | 6751 / 3153 | 6722 / 3403 | 6371 / 3529 | 5235 / 3603 (4901 / 3870 rerun) | 4800 | 4067 / 4764 | 3637 / 4720 | 2048 (curve is a cliff: 1024 is 20-28 % slower) |
| igev-stereo-sceneflow | 9039 / 3041 | - | 8264 / 3517 | 7556 / 3810 | 7315 / 3990 | 6659 / 4474 | 6026 / 4931 | 5772 / 3819 | 2048 (1536 is 10 % slower) |
| raft-large | 3271 / 8609 | - | 2268 / 8986 | - | 2176 / 8988 | - | 2147 / 11100 | 2710 / 11644 | 1024 |
| sea-raft-m | 1785 / 5111 | - | 1450 / 5131 | - | 1433 / 5716 | - | 1395 / 6741 | 1635 / 9732 | 1024 (512 is 4 % slower) |
| neuflow-v2 (fp32) | 1552 / 8929 | 1176 / 9031 | 1101 / 9203 | - | 1095 / 9806 | - | 1104 / 10753 | 1039 / 13354 | 512 |
| memflow (fp32) | 16607 / 9749 | - | 13584 / 10104 | - | 13287 / 10616 | - | 12980 / 11371 | 10016 / 12659 | 1024 |

The curve is a cliff for the nets whose per-step working set is above the limit: the cache evicts least-recently-used
buffers, and a loop that cycles through more bytes than the limit misses on every allocation. Where the working set is
small (raft-large, sea-raft, neuflow, memflow) 512-1024 MB already gets within 3 % of the best speed.

Winograd bands (raft-stereo kernels fp16, cache 1024): `_BAND_BYTES` 192 MB (default, one band at 994x1482) 4900 ms,
48 MB 7010 ms, 16 MB 10830 ms, peak memory unchanged (3.8-3.9 GB). More bands are slower (one dispatch round per band)
and do not lower the peak: the peak is the encoder and the cached buffers, not the V/M tensors. Dead end; not kept.

Other families, `cache_min_mb=1024` against the committed rows (fp16 unless noted; ms / peak MB):

| ref | before | cache 1024 | MPS | declared |
|---|---|---|---|---|
| flowseek-t | 2667 / 5208 | 2256 / 5770 | no row | yes |
| lotus-normal-d | 2197 / 4729 | 1981 / 5671 | 1459 / 6628 | yes |
| stablenormal-turbo | 2527 / 7816 | 2281 / 8687 | 1690 / 8985 | yes |
| detr-r50-coco | 116 / 1120 | 105 / 1903 | 105 / 1954 | yes (speed = MPS, memory just below) |
| nmrf-sceneflow | 2939 / 1851 | 2297 / 2939 | 1469 / 4747 | yes |
| dlnr-sceneflow | 4474 / 3085 | 3791 / 3877 | 2143 / 4094 | yes |
| dedode-v2-b (fp32) | 2915 / 4374 | 2568 / 5319 | 1464 / 8011 | yes |
| selective-igev-sceneflow | 13071 / 3586 | 10846 / 4733 | 6968 / 3859 | no (-17 % time, but +1.1 GB, above MPS) |
| igev-plusplus-sceneflow | 15900 / 3507 | 14541 / 4427 | 8602 / 6037 | no (9 %) |
| gmflow-scale1-things | 6371 / 10879 | 6198 / 11187 | - | no (3 %) |
| marigold-depth-v1-1 | 4231 / 4254 | 4006 / 5142 | 2498 / 6640 | no (5 %) |
| loftr-outdoor (fp32) | 1979 / 3802 | 1858 / 4486 | - | no (6 %) |
| mask-rcnn-r50-fpn-3x-coco | 412 / 1473 | 416 / 2220 | - | no (0 %) |
| romav1 | - | failed in this run | - | not measured |

Final survey rows (portable, ms / peak MB, before -> after, MPS):

| ref | dtype | before | after | MPS |
|---|---|---|---|---|
| raft-large | fp16 | 2147 / 11100 | 2134 / 8987 | 2710 / 11644 (MLX wins both) |
| raft-large | fp32 | 2735 / 11424 | 2724 / 10567 | 2522 / 12889 |
| sea-raft-m | fp16 | 1395 / 6741 | 1427 / 5714 | 1635 / 9732 (MLX wins both) |
| sea-raft-m | fp32 | 2209 / 10723 | 2187 / 9721 | 1820 / 12542 |
| neuflow-v2 | fp32 | 1104 / 10753 | 1087 / 9252 | 1039 / 13354 |
| memflow | fp32 | 12980 / 11371 | 13101 / 10550 | 10016 / 12659 |
| lotus-normal-d | fp16 | 2197 / 4729 | 1973 / 5670 | 1459 / 6628 |
| stablenormal-turbo | fp16 | 2527 / 7816 | 2299 / 8683 | 1690 / 8985 |
| detr-r50-coco | fp16 | 116 / 1120 | 106 / 1924 | 105 / 1954 |
| nmrf-sceneflow | fp16 | 2939 / 1851 | 2246 / 3106 | 1469 / 4747 |
| dlnr-sceneflow | fp16 | 4474 / 3085 | 3762 / 3937 | 2143 / 4094 |
| dedode-v2-b | fp32 | 2915 / 4374 | 2534 / 5327 | 1464 / 8011 |
| flowseek-t | fp16 | 2667 / 5208 | 2160 / 5781 | no row |

All survey goldens pass; nafnet-gopro and wd-vit-tagger-v3 `test_mlx` pass. `apple_mlx.json` was regenerated in the memory pass below (see "Routing" there). Remaining: every net above except raft-large / sea-raft-m fp16 stays slower than MPS (the
GRU / attention convs are FLOP-bound at about 3 TFLOPS effective), the cache cliff of raft-stereo / igev cannot be
shrunk without fewer distinct large buffers per step (not found cheaply), and the memory gap to MPS closes only where
the cache fits.

## Dense 3x3 conv kernel (conv_mlx) investigation

Micro-benchmark (chained 8-16 layers per eval, fp16, channels-last, M1 Pro; probe scripts in `/Volumes/Daten/ff-mlx-probe`,
outside the repo). Result: the existing three-stage Winograd F(4,3) (input transform, batched `mx.matmul`, output
transform fused with bias/act/residual) already beats `mx.conv2d` and is at or above torch MPS `F.conv2d` in isolation for
C >= 64. The net-level gap to MPS is therefore launch overhead, small layers and transient memory, not conv throughput.

Ceiling: the batched GEMM reaches about 3 TFLOPS effective (direct-equivalent) of the M1 Pro fp16 peak (about 10 TFLOPS
for the 16-core GPU); `mx.matmul` is the limiter, and the transforms are bandwidth-bound.

Tried and not shipped:

- Fused simdgroup_matrix Winograd (GEMM fused with the output transform, input transform in the operand load,
  `fused.py`): not faster than the three-stage pipeline. Threadgroup staging of the 36 frequencies limits occupancy, and
  MLX's own GEMM is better tuned than a hand-written 8x8-fragment kernel.
- Banding the transients (concat or `slice_update`) does not lower peak memory: MLX keeps transients alive until the
  command buffer commits. `MLX_MAX_MB_PER_BUFFER` is the only lever (not re-tested on a full net).
- Direct implicit-GEMM with simdgroup_matrix: slower than Winograd for C >= 64 (9 taps vs 36/16 multiplies per tile).

Shipped (commit 933b4913 and follow-up):

- 3x3x3 conv3d Winograd: the three depth taps are stacked along K (K = 3C), so one GEMM covers the volume, then the
  per-slice output transform. 1.5-3x over `mx.conv3d` for >= 64 channels; capped at 192 MB of transients
  (`FF_BAND_MB`), larger volumes fall back to `mx.conv3d`. Wired into `mlx_graph._conv_core` under kernels mode.
- Work-based gate for small images: `h*w >= 48^2`, or channels >= 64 and `h*w*cin*cout >= 2^25`; groups=1, k=3,
  stride 1, even channels >= 16.

Survey and goldens: compact, realplksr, span, realesrgan, nafnet-gopro, wd-vit-tagger-v3 and the stereo goldens pass.
Declaring a kernels mode for hitnet-eth3d (994x1482): portable 800 / 592 ms (fp32 / fp16), kernels 1004 / 746 ms, so it
is slower and was not declared. aanet, s2m2, dlnr and the SR nets were not declared either: earlier runs showed memory
roughly doubling for the stereo group, and the portable path is already the faster one.

## Net-level graph runner pass (commit 9373e1d2)

Profiled realesrgan-x4plus (512x512), hitnet-eth3d (994x1482) and big-lama (512x512) with `FF_MLX_PROFILE=1`. Finding:
convs return `_nchw(y)`, a transposed view of NHWC data, so every following elementwise op ran a strided (3-8x slower)
kernel, wrote NCHW, and the next conv copied it back to NHWC; `cat` along dim 1 materialised NCHW. In realesrgan the
elementwise and cat ops cost about 2 s of a 6.9 s unfused forward.

Fix (`mlx_graph.py`): values that are conv outputs (or elementwise/cat/pad/view ops of such values) are tagged and those
ops run on the channels-last view (`CL_OPS`, `_channels_last_values`). A blanket rule was worse (hitnet 612 -> 715 ms),
since genuinely NCHW tensors got strided access, hence the tagging. `FF_MLX_NO_CL=1` disables it.

M1 Pro 16 GB, portable, survey numbers (goldens pass):

| Net | dtype | before ms | after ms | MPS ms |
|---|---|---|---|---|
| realesrgan-x4plus | fp16 | 6604 | 5578 | 2066 |
| realesrgan-x4plus | fp32 | n/a | 9413 | n/a |
| hitnet-eth3d | fp32 / fp16 | 800 / 612 (fp16 harness) | 723 / 572 | see survey |
| big-lama | fp32 / fp16 | 370 (harness) | 343 / 321 | see survey |

Allocator cache (`FASTFORKS_MLX_CACHE_MB`, default 128) on realesrgan portable fp16: 128 -> 5613 ms, 2048 -> 4591 ms
(peak GPU 1576 MB). Not shipped as a default: the gain is net-specific and costs resident memory.

Forced Winograd on realesrgan (fp16, post-CL): cache 1024 gives 2760 ms but footprint 4.5 GB (MPS 2.5 GB); cache <= 512
falls to about 4650 ms (a cliff). With `FF_MLX_CHUNK_MB=1024` 2509 ms (133 -> 37 evals). A per-eval trace shows peak GPU
5.9 GB comes from the final segment (the 2048x2048 upsampling convs), not the cache: Winograd transients at that size
dominate. `MLX_MAX_MB_PER_BUFFER=16` has no effect. Kernels mode for realesrgan stays undeclared (memory above MPS).

Not done: leaky-ReLU fusion into the conv act (`_fuse_activations` only matches ReLU/PReLU), incremental transform of
dense-block concats, layout propagation through norm/pool ops, a torch/MPS channels_last equivalent.

## realesrgan-x4plus: where the 2.7x came from, and what closed it

Per-layer profile (512x512, fp16, M1 Pro): MLX's direct `mx.conv2d` runs the 351 convolutions at about 1.55 TFLOPS,
where MPS reaches about 4.5 TFLOPS effective (Winograd-class). The unfused LeakyReLU / x0.2 / add / cat / nearest
up-sampling chain adds 1.5-2 s on top. In kernels mode the Winograd convolution fixes the first part; the remaining
time was allocator churn (cache below the transient sizes) and the 2048x2048 tail's peak memory.

Fixes (each golden-checked, bit-identical to the unbanded path where noted):

- `mlx_band.band_tail`: a row-local tail chain (convs, elementwise, integer nearest up-sampling) ending in the output
  runs per row band (48 MB slabs, halo rows), each band evaluated before the next; peak MLX memory 5.9 GB -> 1.28 GB.
  `FF_MLX_NO_BAND=1` disables it; `FF_MLX_BAND_MB` sets the slab size.
- Nearest up-sampling on the channels-last view (`_nearest_cl`).
- Kernels mode: LeakyReLU fused into the Winograd output transform (PReLU with a constant slope), and
  `conv * c + r` (the RDB residual) fused with `c` folded into the weights and bias, `r` added in the epilogue.
- realesrgan declares `mlx_kernels = ["conv_mlx"]` with `cache_mb=1024` (cache 512 -> 3.4 s, 128 -> 4.3 s: a cliff).

Survey (apple-m1-pro-mlx-16gb, goldens pass), process footprint:

| Mode | dtype | ms | footprint MB | MPS ms / MB |
|---|---|---|---|---|
| portable | fp16 | 5376 | 1447 | 2066 / 2466 |
| kernels | fp16 | 2290 | 2535 | 2066 / 2466 |
| portable | fp32 | 9196 | 2193 | 3022 / 4546 |
| kernels | fp32 | 7473 | 2834 | 3022 / 4546 |

fp16 kernels: 5578 -> 2290 ms (MPS 2066, 1.11x), footprint near MPS (the 1 GB allocator cache is resident; MLX's own
peak is 1284 MB). fp32 stays 2.5x behind MPS but needs 38 % less memory. Not done: removing the dense-block `cat`
(multi-source input transform), reusing the input transform across concats (the transform is about 45 % of conv time),
reducing the cache.

## Rollout of the Winograd kernels to the other restoration and segmentation nets

Regression first. `test_mlx[raft-stereo-eth3d@1]` kernels fp16 drifted to mean|diff| 0.0229 (limit 0.02). Bisect
(`git archive` trees): `933b4913` (work-based gate admitting small images) made the 32x40, 384->128 GRU convs take the
Winograd path. Those run in float32 activations (fp32 islands) with float16 weights and bias; the kernels read bias,
slope and residual as the activation dtype `T`, so the fp16 bias was reinterpreted, not converted. Fix (`9ea0aae9`):
`conv_mlx.ops._as` casts them to `x.dtype`. Result 0.0229 -> 0.00178 (portable 0.00206), no speed change. raft-large,
raft-small, raft-stereo-middlebury, igev (4 variants), memflow, neuflow-v2, sea-raft-m, rt-monster goldens pass.

Second finding: the 128 MB buffer-cache default (`cc3fce13`) halved the Winograd nets that declared no `cache_mb`:
realesr-general-x4v3 kernels fp16 103 -> 203 ms, span 85 -> 102. They now declare `cache_mb=512`.

Survey, 512x512 input, M1 Pro 16 GB, ms / process footprint MB, fp16 (MPS fp16 row for reference). Goldens pass.

| Net | MLX portable | MLX kernels | MPS | Declared |
|---|---|---|---|---|
| realesrgan-x2plus | 1207 / 971 | 538 / 2076 | 549 / 1440 | conv_mlx, cache 1024 (shared card) |
| realesrgan-x4plus-anime-6b | 1615 / 1319 | 826 / 2361 | 664 / 2351 | same, 1.24x behind MPS, memory equal |
| realesr-general-x4v3 (compact) | 246 / 737 | 104 / 1061 | 132 / 1289 | cache 512 (was 203 ms at 128) |
| span-x4 | 149 / 827 | 86 / 1210 | 89 / 1296 | cache 512 |
| span-x2 | 134 / 753 | 77 / 1073 | 74 / 1296 | cache 512 |
| codeformer | 285 / 1624 | 180 / 2901 | 196 / 1866 | conv_mlx, fp16 only, cache 1024 (512: 216 / 2543) |
| deeplabv3plus-r101 (1024x2048) | 444 / 1618 | 365 / 1638 | 430 / 1891 | conv_mlx |
| fcn / deeplabv3 / pspnet r101 | 358 / 452 / 388 | 349 / 452 / 376 | 405 / 528 / 411 | conv_mlx (within 3 %, shared card) |

fp32 kernels runs of the large nets stay 2-3x behind MPS fp32 (x2plus 1677 vs 709 ms, anime-6b 2207 vs 973).

Not declared, with the reason:

- big-lama: reflect padding keeps every 3x3 conv off the Winograd gate; kernels run the portable path (325 ms vs MPS 214).
- gfpgan: style-modulated conv weights are tracers inside the compiled graph; `_wino_weights` cannot eval them
  ("Attempting to eval an array during function transformations"). Would need a modulated-weight Winograd path.
- realplksr: 1x1 and large depthwise kernels, no 3x3 dense convs; kernels run 1910 ms vs portable 1991, golden fails.
- nafnet-gopro: fp32 only on MLX (1x1, depthwise, channel attention); kernels = portable (694 vs 649 ms; MPS fp32 409
  ms / 2894 MB, fp16 368 / 1715), memory 1745 MB is below MPS fp32. Golden passes.
- restormer, swinir, swin2sr: attention- and depthwise/1x1-dominated (swinir-m-x4 is seconds, not tens of ms); the
  dense 3x3 convs are a small share, and their tails are not row-local, so `band_tail` does not trigger. Not extended
  to pixel_shuffle tails here: no measured memory problem on these nets.

## Memory pass 3: kernels mode at or below MPS memory, and memory-aware routing

Where the footprint came from (fp16, 512x512; `footprint = ~775 MB base + MLX peak`, MLX peak = resident buffer cache +
live weights + transients):

- The buffer cache is resident: footprint tracks the declared `cache_mb` almost 1:1 (x2plus 2076 -> 1353 MB going 1024 -> 256).
  Too small a cache is a speed cliff (allocator churn from the transients), so the cache has to be sized per net.
- codeformer: 500 MB of F(4,3) Winograd weights (29 layers of 512 channels on 16x16 maps, 18 MB each, 4x the kernel) plus
  125 MB of the source 3x3 weights; the transients (about 840 MB) equal portable's. So it was the weights, not the transients.
- Dropping the source weights was not done: the compiled closures and the shared constants hold them.

What changed:

- `_wino_tile`: maps of at most 16x16 use F(2,3) (1.78x the kernel instead of 4x); on larger maps F(4,3) stays faster
  (microbench fp16 ms/layer, direct / F4 / F2: 16^2x512 0.49 / 0.36 / 0.20; 32^2x512 1.64 / 0.39 / 0.59; 64^2x256 0.99 / 0.42 / 0.65).
- Lazy weights: a layer whose transformed weights exceed `FF_WINO_LAZY_MB` (4 MB) is transformed per call inside the graph
  (`conv_ops.weights_graph`, float32 `kron(G,G)` matmul, max diff to the cached transform 0 for F2, 3e-5 for F4) instead of
  being cached. codeformer: 8 MB 223 ms / 1895 MB, 4 MB 246 / 1607 (cache 128), 2 MB 242 / 1551.
- `mlx_wrap.port(cache_mb=..., chunk_mb=...)` accept a per-variant mapping (key `""`: default). `chunk_mb` is the graph's
  checkpoint interval (default 256 MB of new activations); a larger one evaluates less often, which the big-activation
  nets need at a small cache. realesrgan: x2plus cache 128 / chunk 1024, x4plus 768 / 1024, anime-6b 512 / 1024;
  codeformer cache 256. compact and span keep 512 (already below MPS memory; compact fp16 256: 111 ms / 985 MB, 128: 195 ms).

Survey, 512x512 fp16, ms / footprint MB (rows in `apple-m1-pro-mlx-16gb.jsonl`; goldens pass):

| Net | kernels before | kernels now | MPS |
|---|---|---|---|
| realesrgan-x2plus | 538 / 2076 | 567 / 1222 | 549 / 1440 |
| realesrgan-x4plus | 2290 / 2535 | 2184 / 2452 | 2066 / 2466 |
| realesrgan-x4plus-anime-6b | 826 / 2361 | 849 / 2106 | 664 / 2351 |
| codeformer | 180 / 2901 | 239 / 1637 | 196 / 1866 |

Memory is now at or below MPS for all four; speed gave back 5 % (x2plus), -5 % (x4plus, noise), 3 % (anime-6b) and 33 %
(codeformer: 180 -> 239 ms; it was 40 % above portable's footprint). x2plus is 3 % behind MPS, x4plus 6 %, anime-6b 28 %
and codeformer 22 % behind, so none of them routes to MLX (they did not before either). Timing noise on codeformer is large
(the same setting measured 232 and 375 ms in different runs). The fp32 rows of these nets run the portable graph with the
same cache and chunk; x4plus fp32 kernels is 7380 ms vs MPS 3022.

Not fixed: raft-stereo fp32 kernels (5590 vs MPS 4713 MB; the MLX peak is 4224 MB of transients, a smaller cache only
slows it: cache 1024 7011 ms / 4607 MB vs 2048 5435 / 5416), crestereo fp16 (+2 %), realesr-general fp32 (+3 %, equal
speed), segformer / segnext graph_mlx (the dpt_mlx note was a stale-row artefact, see Memory pass 4).

### Routing (`apple_mlx.json`)

`tools/gen_auto_mode.py` now also requires that an MLX route for `auto` / `dtypes` / `kernels` has a peak footprint within
`MEM_SLACK` (1.02) of the fastest MPS path it replaces on every workload (it did not look at memory before). The mps_auto
(Metal kernel vs portable MPS) selection is unchanged.

The regenerated file has 317 models (166 before). The growth is not stale rows: it is the many families surveyed on MLX
since the last regeneration (197 new `mlx` / `mlx_default` entries, which the loader uses only where MPS has no
implementation at that dtype, and 37 new `auto` entries). New `auto` routes to MLX kernels with a speed and memory win:
realesr-general-x4v3 fp16 (104 ms / 1061 vs MPS 132 / 1289), deeplabv3 / deeplabv3plus / fcn / pspnet r101 fp16
(349-376 ms / 1.6 GB vs 405-528 ms / 1.9-2.0 GB), yolox-tiny (12 vs 20 ms, 453 vs 1492 MB). Removed or changed because MLX
needs more memory than MPS: the seven depth-anything-v2 refs (`auto` kernels fp16 -> portable fp16: the dpt_mlx kernels
use 1.9 GB vs 1.4 GB portable) and dinov2-vits14 / vitb14 / vits14-reg / dinov3-vitl16 / tipsv2-b14 (faster at fp16, more
memory; routing dropped), dinov2-vitb14-reg and dinov3-vitb16 now route at fp32 instead of fp16.
Goldens of the newly routed refs pass; tests/test_auto_mode.py covers the memory rule. Pre-existing failures in
tests/test_mlx.py (`test_card_declares_the_port_per_mode`, `test_family_without_a_port_raises`,
`TestTuning::test_cache_limit_scales_with_memory_between_its_bounds`) are unrelated and unchanged.

## Memory pass 4: transformer / attention nets against MPS

Finding: most of the "MLX uses more memory than MPS" gaps of the ViT / DPT families were stale survey rows (taken before
the 128 MB cache default), not kernel cost. After a forced re-survey the dpt_mlx kernels are leaner than portable and MPS.
The auto routing was regenerated from the fresh rows (`apple_mlx.json`): depth-anything-v2 s/b/l and the other
re-measured ViT refs route to MLX again where they are faster and within `MEM_SLACK` of MPS.

Real fix: `compile=False` for the moge-2 (`_trunk`) and lingbot-depth (`_graph`) MLX ports. The compiled fp16 segments of the
full-resolution decoder peaked high; the lazy replay is as fast (moge) or 7 % slower (lingbot) with far less memory.
moge-2-vits 994x1482 fp16 footprint 3.9 GB -> 2.0 GB, lingbot-depth 4.8 GB -> 3.0 GB, moge-2-vitb 4.2 -> 2.2 GB.

fp16 ms / footprint MB (portable; kernels where it exists), re-surveyed, goldens pass:

| Net | MLX before | MLX now | MPS |
|---|---|---|---|
| depth-anything-v2-s kernels | 79 / 911 | 66 / 698 | 81 / 1575 |
| depth-anything-v2-b kernels | 192 / 1931 | 168 / 1029 | 199 / 2061 |
| depth-anything-v2-l kernels | 548 / 2788 | 501 / 1768 | 579 / 3282 |
| dinov2-vitb14 | 19 / 807 | 19 / 607 | 25 / 778 |
| dinov3-vitl16 | 55 / 1597 | 46 / 1017 | 61 / 1473 |
| tipsv2-b14 | 19 / 797 | 18 / 598 | 24 / 777 |
| moge-2-vits-normal | 3.9 GB (probe) | 886 / 2024 | 857 / 1679 |
| moge-2-vitb-normal | 4.2 GB (probe) | 1144 / 2183 | 1012 / 2265 |
| lingbot-depth | 4.8 GB (probe) | 4577 / 2989 | 3376 / 3228 |

Still behind MPS: dinov2-vits14 (398 vs 346 MB; process baseline, route stays off), moge-2-vits (+345 MB, 3 % slower),
moge-2-vitb (13 % slower), lingbot-depth (36 % slower; memory is below MPS), sam3 fp16 (+678 MB: only the image encoder runs on
MLX, the concept/text/decoder stay CPU torch plus the twin; the MLX peak itself is 2.4 GB). Not re-surveyed: prompt-depth-anything,
moge-3, metric3d, zoedepth, video-depth-anything, unik3d, any2full, mask2former / oneformer, da3 (large-gap old rows,
may be stale or may need the same `compile=False`).

The three pre-existing tests/test_mlx.py failures were stale tests, now fixed (aliked-lightglue as the portless family,
cache limit 1/128 with 128 MB floor); `test_kernels_takes_mlx_kernels` follows the new depth-anything routing.

## Final MLX vs MPS table

Every ref whose MLX row predated cc3fce13 (128 MB cache default) or the later compile-policy/channels-last/cache changes was re-measured on the M1 Pro 16 GB (warm median; one job at a time; ratios are MLX/MPS, below 1 is better for MLX; the row shown is the float16 one where both exist, else float32; best MLX mode of portable/kernels). Verdict: WIN = faster and not leaner-losing (memory within 1.02x), TIE = within 3% speed with memory not above MPS, LOSE otherwise. 0 refs: 0 WIN, 0 TIE, 0 LOSE.

Not measurable under the 9 GB swap rule (the shared Mac held 10.6 GB stale swap from other apps; these jobs were killed and keep their old rows): depth-pro, vggt-1b, vggt-omega, anysplat, lingbot-map, resplat-base-re10k, c-foundationstereo-s, foundationstereo-vit-s, great-igev-sceneflow, waft-stereo-s, alltracker, cotracker3-offline/online, locotrack-b, flowformer-small-chairs, gimm-vfi-r; not reached: tapir, tapnext, track-on2, nnfoundation-*, nninteractive, totalsegmentator-*. Memory-fix sweeps (compile=False, cache_mb 128/256/512) for the losers below were not run for the same reason; `FASTFORKS_MLX_COMPILE=0` (sweep switch in `mlx_wrap.port`) and `FASTFORKS_MLX_CACHE_MB` are ready for them.

| ref | dtype | input | MLX mode | ms | MB | MPS ms | MPS MB | speed | mem | verdict | why |
|---|---|---|---|---|---|---|---|---|---|---|---|
