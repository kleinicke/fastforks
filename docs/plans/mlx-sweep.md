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
  FASTFORKS_CACHE=/Volumes/Daten/fastforks-cache FASTFORKS_OFFLINE=1 PYTHONPATH=src \
  FASTFORKS_WEIGHTS_PATH=/Volumes/Daten/fastforks-weights
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
| seedvr2 | restoration | portable_mlx (hand-written `mx` net), bf16+fp16 pass; 7B bf16 by block streaming, or quant="nf4" | 512x512 -> 1024x1024 ms / MB: 3b bf16 14307 / 5177, fp16 14023 / 5178 (was 18649 / 8830 and 21337 / 8847 fully resident); 7b bf16 25365 / 6547, fp16 32426 / 6549 (was MemoryError). MPS reports it unsupported, so MLX is the only Apple backend. Golden 3b bf16 mean 0.0042 / p99 0.032, fp16 0.0037 / 0.027. Weights come straight from the bf16 safetensors via the lazy loader `shared/mlx_weights.Weights` (no PyTorch fp32 twin). VAE convs run in 32 MB row bands with group norm, SiLU and residual fused per band (peak 11.7 -> 8.8 GB). `stream=True` (the card default) now also applies on MLX: the transformer blocks that fit in `MLX_RESIDENT_SHARE` = 25 % of unified memory stay resident, spread evenly over the depth, and the rest are re-read per call with `Weights.stream` (fresh lazy reads), the next streamed block queued with `mx.async_eval` while the current one computes (the marigold_v2 design). The weight disk reads ~0.85 GB/s (dd: 16.5 GB in 19.5 s, no page-cache reuse), so the 7b is read-bound: fully streamed 19.8 s per call at golden size with 1.4 GB peak, 9 of 36 blocks resident 15.2 s with 5.4 GB; the 3b keeps ~19 of 32 blocks and got faster as well (median 14.3 vs 18.6 s, min 14.3 vs 15.6 s: the old run had a noisy median; cause of the rest not isolated). The 7b survey run grew swap 3.5 -> 6.7 GB with 41k pageouts (~0.65 GB over ten calls): small against the 16 GB read per call, timing valid. fp16 7b is slower than bf16 (likely the per-call bf16 -> fp16 cast of the streamed blocks; not profiled). 7b with quant="nf4" (`mlx_weights.Nf4`, bit-exact with shared.nf4) 4.5 GB resident, golden bf16 0.0043 / 0.025, fp16 0.0039 / 0.024, ~7 s per call at test size; the 7b golden is the nf4 one (bf16 vs nf4 differ by ~26 dB PSNR, so bf16 7b is checked by the survey's golden and by the shared code path with the 3b golden). fp16 7b needed the text residual stream in fp32 (it grows to ~1e5). Open: a faster weight disk (internal SSD) would roughly halve the 7b time; not used because the 16.5 GB file must stay on Daten. |
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
| depth_pro | depth | portable_mlx (net via `fast`, released twin, chunk 256 pipelined), fp32+fp16 pass | fp16 4653 / 5490 vs MPS 5011 / 7169 (faster and leaner; routed; was 4491 / 11370); fp32 7541 / 10529 vs 6115 / 11267 (leaner, slower; was 6473 / 15337). The float16 graph comes from the checkpoint's float16 weights (mmapped twin, no float32 copy), `release=True` frees the twin, and a declared 256 MB chunk with `pipeline=True` replaces the 2 GB big-weights segments (chunk 512: 4406 / 6859, 1024: 4736 / 6300 unpipelined). |
| depthfm | depth | portable_mlx, fp32+fp16 pass | Survey: out of memory on MLX and on MPS alike at 994x1482 (a 66 GB attention buffer). Golden passes at the test size. |
| lingbot_depth (+dc) | depth | portable_mlx, fp32+fp16 pass | fp16 4589 / 8528 vs MPS 3376 / 3228. Slower and heavier. |
| lotus (normal g, d) | depth | portable_mlx, fp32+fp16 pass | g fp16 2089 / 9398 vs MPS 1553 / 6632; d fp16 2166 / 9159 vs 1459 / 6628. UNet and VAE run on MLX, loop and noise on the CPU. Slower and heavier. |
| marigold | depth | portable_mlx, fp32+fp16 pass | fp16 4145 / 9333 vs MPS 2498 / 6640. Slower and heavier. |
| marigold_normals | depth | portable_mlx, fp32+fp16 pass | fp16 2104 / 9797 vs MPS 1655 / 6671. |
| stablenormal | depth | portable_mlx, fp32+fp16 pass | fp16 2661 / 11417 vs MPS 1690 / 8985 (UNet, ControlNet and VAE on MLX). |
| marigold_v2 | depth | portable_mlx (`net/mlx_net.py`), bf16 pass (mean 0.0089 / p99 0.038, limits 0.01 / 0.05) | bf16 51319 ms / 2251 MB; MPS unsupported (cannot run). NF4 stays packed and is expanded per linear; the 60 blocks are read lazily from the file one at a time, the next block's read overlapped (`stream=True`, default; `stream=false` keeps ~9.5 GB resident). Per 994x1482 call: 60 blocks x ~650 ms (GEMM-bound, NF4 expand ~10 ms per linear), VAE encode 1.5 s + decode 2.4 s, ~8 s of cold reads not hidden. Nothing carries over to MPS (the gain is MLX's lazy file reads). fp16 not offered: Qwen-Image activations overflow fp16. |
| metric3d | depth | portable_mlx, fp32+fp16 pass | fp16 2370 / 5009 vs MPS 1861 / 3636. Slower. |
| midas | depth | portable_mlx, fp32+fp16 pass | fp16 668 / 4117 vs MPS 635 / 2407. The survey golden shows "fail" on MPS too (survey input quirk); `test_mlx` passes. |
| moge | depth | portable_mlx (`_trunk` graph), fp32+fp16 pass | vits fp16 1148 / 6234 vs MPS 857 / 1679; vitb 1394 / 6475 vs 1012 / 2265; vitl 2221 / 7987 vs 1732 / 3265; moge-3 fp16 3257 / 3170 vs 3913 / 5968 (fp32 3508 / 4115): its data-dependent sparse refiner runs as hand-written MLX ops + two Metal kernels (`moge/mlx_refine.py`: neighbour tables, gather implicit GEMM per conv, also at the finest level; was eager torch on the CPU, 3.8 s + ~4 GB), 1.5 s of the 3.3 s; fp16 routes to MLX. MoGe-2: slower and heavier. |
| omni_dc | depth | portable_mlx (pre/post graphs, CG solve on the CPU), fp32+fp16 pass | fp16 3300 / 9841 vs MPS 2408 / 10081. |
| pointdit | depth | portable_mlx (`_encode` fp32, `_dit`), fp32+fp16 pass | b fp16 373 / 1702 vs MPS 337 / 1912; fp32 357 / 1759 vs 367 / 1812. l fp16 1212 / 3759 vs 1013 / 4548; fp32 1073 / 4343 vs 1086 / 4516. Less memory than MPS. The DINOv3-L backbone overflows float16 (hidden states up to 1.5e5), so the backbone graph always runs in float32 (`fp32_methods`). |
| prompt_depth_anything | depth | portable_mlx, fp32+fp16 pass | 756x1008 fp16 s 299 / 2138 vs MPS 195 / 1588. l re-surveyed at the new standard 994x1482 (both devices; the stale 756x1008 rows, upstream's included, dropped): fp16 1082 / 2757 vs MPS 1143 / 3303 (routed); fp32 1404 / 4681 vs 1278 / 4944. |
| unidepth_v2 | depth | portable_mlx, fp32 only | fp32 1694 / 6215 vs MPS 1252 / 6077. fp16 not declared (golden). |
| unik3d | depth | portable_mlx, fp32+fp16 pass | 2026-11-20: reflection / replication pads kept from the export decomposition (they were index gathers) and lowered as slices + concatenation: fp16 1453 / 3185 -> 1458 / 2388 vs MPS 1294 / 4038, fp32 1385 / 4120 -> 1396 / 3000 vs 1217 / 4001. Same speed, ~25 % leaner; still 13-15 % slower (ViT-L SDPA / MLPs at 3129 tokens, then the decoder convs and bilinear resizes). |
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
| vitmatte | matting | portable_mlx (`_net` method), fp32+fp16 pass | s 1080x1920 fp32 1317 / 2451 vs MPS 2333 / 5185; fp16 1280 / 2283 vs 1360 / 2542 (was 1595 / 3923: D=256 global attention now query-blocked, tune trials drop their cache, matting tail in strips). b fp32 2502 / 2701 vs 4680 / 5737; fp16 2368 / 2474 vs 2740 / 2966. Routed fp16. |
| matanyone | matting | not ported (tried, slower) | Tried 2026-11-20: the per-frame stages (`_frame_features`, `_encode_mask`, `pixel_fusion`, `readout_query`, `_segment`, `pred_uncertainty`) exported as fixed-shape MLX graphs, with the memory bank and loop in PyTorch. It worked (golden pass; positional-encoding cache dropped, `aten.prod.dim_int` mapped, aux-mask `torch.where` made exportable) but 1080x1920 ran fp16 6335 ms / 3718 MB vs MPS 3091 / 3838, fp32 8946 / 4671 vs 3444 / 6010. The decoder (`_segment`, 64→16 and 128→64 3x3 convs, 1x1 convs and bilinear upsampling at full and half resolution) takes ~420 ms per step on MLX vs ~136 on MPS, and runs 11 times per call. That is 4.7 of the 6.3 s. Memory gain only 3% in fp16, so the runner was dropped. Revisit when MLX low-channel full-res convs get faster. Golden skipped in the MLX run. |
| rtmpose / dwpose / rtmw | pose | portable_mlx, fp32+fp16 pass | rtmpose-t fp16 24 / 607 vs MPS 26 / 1460; fp32 27 / 656 vs 26 / 1462. Goldens tried: t, l-384, dwpose-t, rtmw-m, halpe26, hand5, face6, ap10k. |
| rtmo | pose | portable_mlx, fp32 only | t fp32 17 / 658 vs MPS 21 / 1523. fp16 fails (t scores p99 0.0049, l keypoints p99 0.90). |
| hrnet | pose | portable_mlx, fp32+fp16 pass | w32 fp16 125 / 1049 vs MPS 122 / 1675; fp32 166 / 1247 vs 141 / 1748. Winograd: tie, not declared. |
| retinaface | pose | portable_mlx, fp32+fp16 pass | resnet50 fp16 190 / 1266 vs MPS 150 / 1913; fp32 284 / 1521 vs 210 / 1879. Kernels mode fails in float16-only options, not declared. |
| vitpose | pose | portable_mlx, fp32+fp16 pass | base-simple fp16 156 / 987 vs MPS 186 / 1910; fp32 204 / 1326 vs 196 / 2073. plus-small and plus-huge pass goldens. |
| yunet | pose | portable_mlx, fp32+fp16 pass | n fp16 21 / 719 vs MPS 22 / 1377; fp32 28 / 771 vs 25 / 1380. |
| anyline, lineart, lineart_anime, mlsd | edges | portable_mlx, fp32+fp16 pass | anyline fp16 201 ms / 1777 MB -> 197 / 1776 vs MPS 117 / 1319, fp32 243 / 2030 -> 223 / 1920 vs 130 / 2346 (the whole module replays as one MLX graph; 2026-11-20: the channel-multiplier depthwise conv as repeat + depthwise and the 1-channel transposed convs as multiply-adds; still slower: the full-res 24-channel depthwise convs and Smish chains at 1280^2, compile on or off the same, MLX's depthwise conv and our Metal one equal); lineart-anime fp16 43 / 1032 vs MPS 51 / 1774; mlsd fp16 33 / 715 vs 45 / 1373. lineart fp16 136 / 1170 vs 67 / 1362 (speed loses; Winograd 133, no gain). |
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
| rife | interpolation | portable_mlx (`_forward` graph, export-safe cat instead of in-place writes), fp32+fp16 pass | 1080x1920 4.25 fp16 406 / 1707 vs MPS 106 / 2482; lite fp16 298 / 1730 vs 80 / 2447. Survey rows: 4.25 fp16 211 / 1680 vs MPS 106 / 2482, lite 202 / 1636 vs 80 / 2447. Lighter than MPS but 2-2.5x slower. 2026-11-20 profile: conv-bound (MLX conv2d 2-3x slower than MPS on these shapes); merging the deconv into its neighbours would save ~3.5 %, so nothing cheap is left. Winograd conv_mlx gave no gain, so no kernels mode. |
| film | interpolation | portable_mlx (`_run` graph), fp32+fp16 pass | 1080x1920 fp16 7046 / 9354, fp32 9944 / 12658 (no MPS row; memory-heavy at 1080p). |
| ema_vfi | interpolation | portable_mlx (`_run` graph), fp32+fp16 pass | small-t 1080x1920 fp16 1201 / 2499 vs MPS 513 / 3583; fp32 1361 / 2371 vs 792 / 6499. Less memory, 1.7-2.3x slower. |
| gimm_vfi | interpolation | portable_mlx (`_forward` graph; added scatter_add and conv3d ops), fp32+fp16 pass | gimm-vfi-r 1080x1920 fp16 30451 / 12185, fp32 58742 / 14977 (no MPS row). Works but very heavy at 1080p on 16 GB. |
| parseq | ocr | portable_mlx (`_decode`: all 26 decoding steps as one graph, no early exit), fp32+fp16 pass | 32x128 fp16 14 / 491 vs MPS 24 / 440; fp32 26 / 537 vs 23 / 496. fp16 faster; memory about 10% above MPS (tiny net, runtime floor). |
| ppocrv5 | ocr | portable_mlx (detector and recogniser graphs; boxes, crops, CTC eager; fp16 neck scaling applied to the twin), fp32+fp16 pass | 1080x1920 mobile fp16 772 / 1433 vs MPS 736 / 2987; server fp16 1955 / 2910 vs 1607 / 6545. Memory 2-2.3x lower, speed 1.05-1.4x slower. |
| boq | retrieval | portable_mlx (`_descriptor` graph over backbone + aggregator; resize eager), fp32+fp16 pass | 480x640 fp32 47 / 905 vs MPS 57 / 987; fp16 52 / 856 vs 48 / 849. About parity. |
| megaloc | retrieval | portable_mlx (`_descriptor` graph), fp32+fp16 pass | 480x640 fp32 53 / 1345 vs MPS 54 / 1926; fp16 83 / 1496 vs 45 / 2617 (fp16 slower than fp32 on MLX here: the Sinkhorn step casts to float32). Less memory; fp32 is the better MLX dtype. |
| tapir | tracking | portable_mlx: whole offline pass as one graph (adds `grid_sampler_3d`, `avg_pool3d` ops); fp32 only (fp16 misses the goldens' tolerances) | 480x854 ms / MB, fp32. tapir 2770 / 1720 -> 1592 / 1717 vs MPS 1382 / 2630 (2026-11-20: depthwise conv1d of the PIPs mixer as shifted multiply-adds instead of MLX's grouped conv1d, which fuse with the GELU: 1.9 -> 0.8 ms per [64, 24, 2048] conv). 1.15x slower, 35% leaner: not routed. All variants pass goldens. |
| alltracker | tracking | portable_mlx: per-query-frame group graph (`group` arg replaces data-dependent `unique`/`nonzero`); fp32 only | 480x854 fp32 5397 / 2597 vs MPS 3508 / 4571. |
| locotrack | tracking | portable_mlx: whole pass as one graph; fp32 only | 2026-11-22: locotrack-b fp32 1482 / 1738 (first call 5.1 s), locotrack-s 1209 / 1653 (4.7 s) vs MPS 2104 / 2474 and 1781 / 2427. MLX export uses 128-query chunks; both auto-route to MLX. |
| tapnext | tracking | portable_mlx: per-frame step graph (pipelined, released); fp32 only | 2026-11-22: tapnext fp32 7477 / 3673 -> 4034 / 2443 (first call 77 -> 6.5 s), tapnext-pp 4036 / 2025, vs MPS 5855 / 3754 and 5610 / 3657. Both auto-route to MLX. |
| cotracker | tracking | portable_mlx: offline = whole pass as one graph; online = window loop eager (data-dependent, `torch.export` guards), CNN and update transformer on MLX; fp32 only | offline fp32 3577 / 3438 vs MPS 1430 / 4725 (bench after the shared fixes: ~3057 ms, not re-surveyed); online 3340 / 3971 vs 1707 / 4689. 2026-11-20 profile (offline): about half is the feature CNN's convs (MLX conv2d 2-3x slower than MPS), the rest correlation lerps and SDPA; no cheap win. |
| track_on | tracking | portable_mlx: per-frame loop eager (data-dependent: queries join at their frame), DINOv3 adapter backbone and the per-frame `step` (attention over memory, correlation, rerank, head; the frame's tensors as arguments) as graphs; fp32+fp16 pass | track-on2 fp32 21453 / 4902 -> 10755 / 3318 vs MPS 8354 / 6102; fp16 21190 / 4947 -> 7623 / 3056 (2026-11-20: the step ran eagerly on the CPU before). fp16 routed (`auto`): 9% faster than MPS fp32 (no MPS fp16 row) at half the memory. |
| nnfoundation (cnn) | volume | `mlx_wrap.port(Portable)`: 3D UNet exported whole; 5-D conv via `mx.conv3d` (transposed: matmul for stride == kernel, else `conv_transpose3d`) | 200x200 fp32 5421 ms / 10651 MB -> 4271 / 9100 vs MPS 3645 / 9435; fp16 4721 / 10094 -> 3520 / 8041 vs 2935 / 7824 (2026-11-20: channels-last instance-norm statistics, 1-channel stem conv3d as one conv2d over folded depth taps); goldens pass. Still 1.2x slower: the rest is `mx.conv3d` itself (~2.4 TFLOPS: 64@112^3 ~115 ms, 32@224^3 ~258 ms per conv). With the graph kernels forced (channel_stats, Winograd 3D) fp16 bench 3045 / 6469: 3.7% slower but 17% leaner than MPS, short of the route rule, so no kernels mode is declared. |
| nnfoundation (vit) | volume | same | 192x192 fp32 18038 ms / 4313 MB vs MPS 21342 / 4987; fp16 13037 / 2469 vs 17636 / 3817; goldens pass. Faster and lighter than MPS. |
| nninteractive | volume | same | fp32 4782 ms / 7025 MB -> 4005 / 5996 vs MPS 3525 / 11439; fp16 4564 / 6823 -> 3797 / 5587 (MPS fp16 not run); goldens pass. Same nnU-Net fixes as nnfoundation-cnn. Memory half of MPS, time 1.14x slower (conv3d-bound). |
| totalsegmentator | volume | `_mlx.port`: each part's sub-nets wrapped with `MlxModule(release=True)`, sliding-window glue stays eager | total fp32 58555 ms / 6627 MB -> 43030 / 5924 vs MPS 27804 / 10180; fp16 42971 / 6614 -> 31549 / 5163 vs 23070 / 6058; 3mm fp32 1573 / 5085 -> 1208 / 4834 vs 593 / 4385; fp16 1333 / 4913 -> 1111 / 4730 vs 522 / 2588; all goldens pass (2026-11-20: the nnU-Net fixes of nnfoundation-cnn plus the conv bias dropped before each instance norm and the head argmax on the GPU). Still 1.4-2.1x slower: in total fp16 the graph alone is 20 windows x 1.33 s = 26.5 s (> MPS 23 s in all); the rest is the CPU sliding-window glue (fp16 accumulate ~2 s, float32 logits copy ~1.8 s); 3mm: graph 495 ms of 1111, the CPU accumulate ~330 ms. |
| pi3 | multiview | `mlx_wrap.port` of the upstream net (`predict` = `net(x)`); adds `aten._linalg_svd` / `_linalg_det` (camera head rotation orthogonalization, CPU stream) | 336x518 fp32 12062 ms / 5600 MB vs MPS 10298 / 6201; fp16 11173 / 3874 vs 8341 / 6510; goldens pass. Memory lower, time 1.2-1.3x slower. |
| vggt | multiview | same (`_Upstream.predict`) | vggt-1b 336x518 fp32 9359 ms / 9387 MB vs MPS 7209 / 10654; fp16 8774 / 6853 vs 6638 / 8423. vggt-omega-1b-512 fp32 9167 / 7391 vs 7938 / 8221; fp16 8358 / 5294 vs 7242 / 8602. Goldens pass; memory lower, time 1.15-1.3x slower. |
| anysplat | multiview | same | 336x518 fp32 8786 ms / 10565 MB vs MPS 33262 / 10705 (MPS fp32 path is slow); fp16 6974 / 8297 vs 5444 / 8751; goldens pass. fp32 3.8x faster than MPS. |
| depthsplat | multiview | `port(Portable)`, adds `aten.clamp.Tensor` | 336x518 fp32 1707 ms / 2177 MB vs MPS 670 / 8880; golden passes in fp32 only (fp16 confidence mean diff 0.016 > 0.014, so only fp32 is declared). Memory 4x lower, time 2.5x slower. |
| mvsplat | multiview | `port(Portable)` + fused `fastforks::warp_dot` Metal kernel | 2026-11-22: fp16 2116 / 2706 -> 1258 / 2541, fp32 2573 / 3466 -> 1573 / 3456 vs MPS 1123 / 4323 and 1040 / 4540. Still 1.1-1.5x slower: no auto route. |
| resplat | multiview | `net.encode` as the exported graph, its kNN as one op (`fastforks::resplat_knn`) lowered to a Metal z-order tiled kNN; the refinement by hand (`net/mlx_refine.py`): Metal projection / tile-intersection / rasteriser kernels, Metal kNN and kNN-attention, ResNet / low-res attention / point transformer in MLX (compiled blocks) | base-re10k 256x256 (the multi-view floor's smallest) fp32 19441 ms / 9553 MB; fp16 14649 / 5639. At 480x640 (warm call, own timing): before 112309 ms / 12071 MB (survey row, refine eager on torch CPU), now fp32 16.3 s, fp16 10.6 s (MLX peak 3.8 GB). Goldens (base, small) pass in fp32 and fp16. Left: the point transformer's GEMMs (~2.3 s per refine step) and the encoder graph (~4 s); MPS rows are still at 480x640 (fp32 14422 / 12080). |
| lingbot-map | multiview | `methods=("run",)`: the whole streaming pass incl. KV cache unrolled into one exported graph | 480x640 fp32 11010 ms / 9090 MB vs MPS 8840 / 13055; fp16 10046 / 8220 vs 7125 / 8834; goldens pass. Memory lower, time 1.25-1.4x slower. |
| mapanything | multiview | `methods=("predict",)` (transformer, DPT, heads and edge mask in one graph) | mapanything-apache 336x518 fp32 9091 ms / 11207 MB vs MPS 7357 / 11638; fp16 8284 / 8402 vs 6369 / 7218 (+1.2 GB); goldens pass. |
| mvsanywhere | multiview | `methods=("encode_image", "matching", "cv_encoder", "decode")`; cost volume stays eager torch | 448x672 fp32 3661 ms / 2475 MB vs MPS 1051 / 3534; fp16 3609 / 2899 vs 1064 / 3801; goldens pass. Memory lower; 3.4x slower (eager CPU cost volume). |
| patchmatchnet | multiview | `methods=("prepare", "predict")`; seeded random hypotheses drawn outside the trace (constants); adds stride-1 `avg_pool3d` | 1536x1024 fp32 2841 ms / 3155 MB vs MPS 1010 / 7593; fp16 2856 / 3625 vs 1574 / 6591; goldens pass. Memory 2x lower, 2.8x slower. |
| sam / sam-hq | promptable_segmentation | `port(Portable, methods=("_enc",))`: the ViT image encoder (and HQ features) on MLX after the eager torch resize; the prompt encoder and mask decoder stay torch | sam-vit-b 1024x1024 fp32 659 ms / 2036 MB vs MPS 631 / 4362; fp16 640 / 1935 vs 600 / 3762. sam-hq-vit-b fp32 691 / 2372 vs 645 / 4374; fp16 689 / 2253 vs 683 / 4542. Goldens pass for all b/l/h variants. Memory about 2x lower, speed within 5 percent of MPS. |
| mobile-sam | promptable_segmentation | same (`_enc` = TinyViT trunk + norm) | fp32 100 ms / 947 MB vs MPS 81 / 1480. fp16 declared off: golden fails (pred_masks mean diff 0.49). Memory 36 percent lower, 1.2x slower. |
| efficientvit-sam | promptable_segmentation | same; `compile=False` (a new `port` option): the fused elementwise kernels exceed Metal's argument-buffer limit under `mx.compile` | l0 fp32 116 ms / 1064 MB vs MPS 51 / 1779. Goldens pass. Memory lower, 2.3x slower. |
| sam2 | promptable_segmentation | same (`_enc` = Hiera trunk, neck) | hiera-tiny fp32 219 ms / 1139 MB vs MPS 241 / 2581: faster and 2.3x lighter. fp32 only declared (hiera-base-plus fp16 misses its golden: pred_masks p99 diff 0.0091 > 0.009). |
| sam3 | promptable_segmentation | `MlxPortable`: the whole detector in hand-written `mx` (`sam3/net/mlx_net.py`: trunk, neck, CLIP text encoder, geometry / DETR encoder and decoder, scoring, mask decoder), weights read lazily from the checkpoint (no PyTorch twin), fp32 islands as `half_parts`; head-dim-32 attention zero-padded to 64 so MLX uses its fused SDPA (unfused: 0.8 GB scores in the DETR encoder); `mx.eval` per DETR layer / neck level / pixel-decoder stage (detector peak 1.1 -> 0.6 GB); `video`: MLX trunk + tracker neck, tracker heads CPU torch fp32 | 1024x1024 fp32 2062 ms / 4705 MB vs MPS 3620 / 5807; fp16 1657 / 2768 vs 3688 / 3523. Goldens pass. 55 % faster than MPS, 21 % leaner in fp16 (before: 2423 / 4198, +678 MB). `mx.compile` of the ViT layer: no gain (GEMM-bound). Nothing carries to MPS: its SDPA is already fused at head dim 32 (8.5 ms for 5184 tokens; padded to 64: 17 ms). |
| clipseg | promptable_segmentation | `methods=("embed","decode")` | 1024x1024 fp32 33 ms / 932 MB vs MPS 45 / 1532; fp16 39 / 984 vs 38 / 1561. Goldens pass. Faster and lighter. |
| insid3 | promptable_segmentation | `methods=("embed",)` (DINOv3 ViT-L backbone); the reference matching stays eager | vitl16 fp32 3244 ms / 2981 MB vs MPS 2953 / 2963 (parity). fp16 declared off: similarity map NaN. |
| dedode | matching | `net.detector` + `net.descriptor` on MLX, fp32 only (fp16 keypoints mean diff 0.0022 > 0.002) | v2-b 2915 ms / 4374 MB vs MPS 1464 / 8011; v2-g 4967 / 5662 vs 3767 / 10008. Memory lower, speed slower than MPS. |
| loftr | matching | backbone + coarse transformer on MLX, fine stage and matching eager; fp32 only (indoor fp16 recall 0.9976 < 0.998) | outdoor 1979 ms / 3802 MB vs MPS 875 / 5767. Memory lower, speed slower. |
| loma | matching | detector, descriptor, matcher on MLX, fp32 (golden only for loma-b) | loma-b 4152 ms / 6133 MB vs MPS 3037 / 8147. Memory lower, speed slower. |
| romav1 | matching | whole net on MLX, fp32; `cholesky` became `cholesky_ex(check_errors=False)` to avoid the with_effects op; new CPU-stream cholesky ops in mlx_graph | 4500 ms / 5242 MB vs MPS 3050 / 7979. Memory lower, speed slower. |
| xfeat | matching | `net.xfeat.net` on MLX, fp32 (LighterGlue stays eager); new `aten.unfold` mapping | 158 ms / 543 MB vs MPS 42 / 1471. Memory lower, speed slower (small net, eager post-processing dominates). |
| flexglue (aliked) | matching | descriptor net, matcher decode and ALIKED's dense maps + score NMS (`ALIKED.detection_maps`, no feature map without `describe`: FlexGlue reads keypoints only) on MLX, fp32 + fp16; keypoint selection and the match head eager (~5 + 2x17 ms). The old recall miss was the `aten.elu` SELU scale (fixed); golden passes | fp32 2249 ms / 3048 MB -> 1224 / 2054 vs MPS 1124 / 3932; fp16 2255 / 2970 -> 1100 / 1387 vs 1069 / 3088. fp16 2.9 % slower (outside the 1 % trade rule), 55 % leaner; not routed. Left: the decoder's 36 SDPA calls at 4096 tokens (~12 ms each, ~440 of its ~810 ms). |
| lightglue | matching | portable_mlx fp32: the extractor's fixed-shape dense part as one MLX graph (DISK's U-Net; ALIKED's dense maps + score NMS via `ALIKED.detection_maps`); keypoint selection, descriptors and LightGlue (data-dependent counts) stay eager on the CPU. The old ALIKED recall miss (0.963, also flexglue's) was the `aten.elu` lowering ignoring SELU's scale; fixed in mlx_graph, both goldens pass | disk 1178 ms / 2333 MB vs MPS 1994 / 3531: WIN, routed. aliked 461 / 2416 vs 206 / 2586: leaner but 2.2x slower (dense graph ~93 ms/image vs MPS 81, CPU SDDH + matcher ~140 ms); stays on MPS. |

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
| c-foundationstereo-s | 16 | - | 27235 ms / 4.5 GB (was 60417 / 11.8) |
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
| foundationstereo-vit-s | 16 | 20584 ms / 13.3 GB | 20504 ms / 4.8 GB (was 36138 / 12.1) |
| foundationstereo-vit-l | 16 | 25647 ms / 13.4 GB | 25847 ms / 4.8 GB |
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

## FoundationStereo family: float16 end to end (v0.21 follow-up)

Goal: peak memory well below MPS, golden passes, speed at least MPS. The twin is exported in float32, so literal
float32 casts and constants leaked into the float16 replay and doubled the 5-D volume and GRU activations. Changes:

- `fastforks::run_dtype` (`_core/mlx_cast.py`, `to_run_dtype`): a cast to the network's dtype that the replay lowers to
  its run dtype; used in the FS / C-FS geometry, the DINOv2 / EdgeNeXt positional encodings, the post-interpolate
  features, the motion encoder and the `torch.where` zero of the concat volume (all guarded by `EXPORTING_FOR_MLX`).
- C-FS: group-wise correlation through the `gwc_volume` Metal kernel (normalised features, left times channels per group,
  since the kernel returns the group mean), fused SDPA in the DINOv2 blocks, `_chunked` joins by `torch.cat` (the
  `empty_like` buffer with slice writes was recorded as a float32 buffer plus copies).
- `mlx_graph`: depthwise conv3d as strided tap sums, transposed conv3d in depth slabs, and batch-statistics norm from
  float32 partial sums with one fused affine pass (no float32 copy of the map: 3.7 GB -> 0.6 GB, 196 -> 88 ms per call).

Survey, float16 portable, 994x1482, ms / peak footprint MB (golden pass):

| ref | MPS portable | MPS kernels | MLX before | MLX after |
|---|---|---|---|---|
| foundationstereo-vit-s | 20584 / 13617 | 19496 / 12980 | 36138 / 12387 | 20504 / 4943 |
| foundationstereo-vit-l | 25647 / 13761 | 24480 / 13277 | - | 25847 / 4941 |
| c-foundationstereo-s | OOM | - | 60417 / 12058 | 27235 / 4625 |

Tried and dropped: `FASTFORKS_MLX_CACHE_MB` 1024 / 2048 (vit-s 20.7-23 s against 21.7-26 s at the default, within the
noise, +0.3-1.1 GB). `kernels_mlx` with `conv_mlx` (Winograd): vit-s 19.0 s / 4.9 GB at survey size, but kernels float16
fails the golden at 100x150 (mean |diff| 625; NaN with `FF_WINO_M=2`; vit-l passes), so it is not declared.
Left: the kernels float16 failure, the SDPA (2.8 GB) and conv3d (3.1-3.7 GB) segment peaks, the float32 rows (old).

## Compile policy and kernels declarations (v0.21 follow-up)

`mx.compile` is now decided per segment instead of per graph: segments whose largest activation exceeds
`SEG_COMPILE_MAX_MB` (200) run eagerly, and a segment with an activation above `TUNE_MIN_MB` (16) is timed eagerly and
compiled and keeps the compiled form unless it is more than 3 % slower or peaks 25 % higher. The trial is spread over
the segment's first four calls (see "Spread tuning" under Cold start); `FF_MLX_TUNE_FIRST=1` runs it all on the first
call as before. `FF_MLX_NO_TUNE=1` disables the tuning, `FF_MLX_SEG_MB` moves the size gate.

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
moge-2-vitb (13 % slower), lingbot-depth (36 % slower; memory is below MPS). sam3 is fixed: the whole detector now runs
in `mx` without a twin (fp16 1657 ms / 2768 MB vs MPS 3688 / 3523), so it routes to MLX in both dtypes. Not re-surveyed: prompt-depth-anything,
moge-3, metric3d, zoedepth, video-depth-anything, unik3d, any2full, mask2former / oneformer, da3 (large-gap old rows,
may be stale or may need the same `compile=False`).

The three pre-existing tests/test_mlx.py failures were stale tests, now fixed (aliked-lightglue as the portless family,
cache limit 1/128 with 128 MB floor); `test_kernels_takes_mlx_kernels` follows the new depth-anything routing.

## Cold start: the first call per input size (v0.21 follow-up)

`mlx_graph.GraphNet` builds its replay lazily on the first call of each input signature, in memory only:
`torch.export` + core-ATen decomposition (1 to 7 s), translation (BN folding, fusion, NHWC, constants), `mx.compile`
of the segments, and a tuning pass that times every compiled segment above 16 MB once eagerly and once compiled
(each trial is two calls of each form). MPS has none of it (its `first_call_s` is the warm time plus 0.1 to 0.6 s).

Measured (M1 Pro 16 GB, portable mode, the survey's standard input, one job at a time, process start to first output;
warm = steady call; "before" is the code at bf3ed8e8, "no cache" includes the tuning memo below, "hit" is a new process
with a warm plan cache; second shape = one call at 3/4 of the size after the first, no cache -> hit):

| ref (dtype) | warm s | first, before | first, no cache | first, hit | second shape | trials before -> now |
|---|---|---|---|---|---|---|
| depth-anything-v2-s (fp16, hand-written) | 0.07 | 0.10 | 0.10 | 0.09 | 0.07 | 0 |
| dinov2-vitl14 (fp16, hand-written) | 0.06 | 0.08 | 0.08 | 0.08 | 0.03 | 0 |
| yolox-tiny (fp16) | 0.014 | 1.5 | 1.5 | 1.08 | | 0 |
| segformer-b0-ade20k (fp16) | 0.08 | 1.65 | 1.65 | 0.78 | 0.14 -> 0.07 | 0 |
| rf-detr-n-coco (fp32) | 0.023 | 3.1 | 2.5 | 1.21 | 0.02 | 0 |
| dfine-n-coco (fp16) | 0.021 | 2.9 (8.9 file-cache cold) | 2.9 | 1.16 | 0.02 | 0 |
| parseq (fp16) | 0.016 | 6.0 | 6.0 | 2.84 | 0.02 | 0 |
| ppocrv5-mobile (fp32) | 1.19 | 9.8 | 9.8 | 4.18 | | |
| u2net (fp16) | 0.06 | 3.2 | 3.2 | 1.64 | | 1 |
| hrnet-w32 (fp16) | 0.11 | 3.0 | 3.0 | 1.8 | 2.19 -> 1.05 | 4 |
| geocalib-pinhole (fp16) | 0.30 | 5.2 | 5.0 | 2.42 | 3.82 -> 1.64 | 12 -> 12 |
| da3mono-large (fp32) | 0.30 | 9.1 | 9.1 | 1.66 | 0.31 | 1 |
| vitmatte-s (fp16) | 1.20 | 11.6 | 11.0 | 3.42 | 5.80 -> 2.03 | 39 -> 37 |
| prompt-depth-anything-l (fp16) | 1.09 | 12.4 | 9.3 | 2.84 | 3.89 -> 2.05 (was 6.32) | 78 -> 23 |
| psmnet-sceneflow (fp16) | 2.57 | 7.3 | 6.8 | 4.09 | 6.37 -> 2.20 | 45 -> 32 |
| patchmatchnet (fp16) | 0.76 | 23.6 | 10.2 | 4.12 | 11.97 -> 3.12 (was 15.6) | 132 -> 95 |
| depth-pro (fp16) | 4.72 | 33.5 | 21.2 | 8.36 | 4.74 | 154 -> 20 |
| swin2sr-x2 (fp16) | 7.5 | 39.6 | 16.2 | 11.89 | 12.23 -> 7.64 (was 26.7) | 1076 -> 71 |

Outputs: no cache vs cache miss vs cache hit compared leaf by leaf on two calls of every row above (17 families):
bit-identical everywhere. `tests/test_golden.py::test_mlx` passes with the cache on, as miss and as hit, for vitmatte-s,
rf-detr-n-coco, swin2sr-x2, patchmatchnet, prompt-depth-anything-l, hrnet-w32, parseq, dfine-n-coco and geocalib-pinhole
(the hit runs take 30 to 80 % less time), and without it for swin2sr-x2 and patchmatchnet after the memo; `tests/test_mlx_plan_cache.py`
checks a hit on a tiny conv + BN net (no export, bit-identical). Warm times unchanged by the memo (within 3 %). Second shape: most nets resize to a fixed
input (detectors, dinov2, da3, depth-pro), so the second size costs one warm call; hrnet, geocalib, vitmatte, psmnet,
patchmatchnet, swin2sr and prompt-depth-anything rebuild the graph for it. A cache miss costs 0.3 to 2.5 s more than no
cache (`torch.export.save`); entries take 1 to 25 MB. The file-cache-cold first process (weights and Python modules not
in the OS file cache: dfine 8.9 vs 2.9 s) is not touched by any of this.

What was built:

- `Model.warmup(sizes, *, views=8)` (`_core/model.py`): one call per `(height, width)` on the synthetic bench inputs,
  under `inference_mode`; returns the model. Backend-agnostic (on MPS / CUDA it warms kernels and allocators).
- `FASTFORKS_MLX_PLAN_CACHE=<dir>` (`shared/mlx_plan_cache.py`, off by default): per `MlxModule` and input signature,
  the decomposed `ExportedProgram` (`torch.export.save`, parameters and buffers as one-element stand-ins rebound to the
  module's own tensors by name, shape and dtype on load; the loaded graph is dead-code-eliminated, else the loader's
  extra getitems on batch norm break BN folding and change bits) and the tuning decisions. Key: the load (ref, mode,
  dtype, load options, user weight files; set by `Model._build_mlx`), module index and class, parameter shapes, input
  signature, fastforks / torch / mlx versions and a fingerprint of fastforks' sources. Not cached: modules built outside
  a model load, and exports that needed the eager forward first (lazy buffers). Empty and one-element tensors are
  stored as they are (an empty stand-in broke `torch.export.save` for rf-detr).
- Tuning memo (`mlx_graph._TUNE_MEMO`, always on): a trial's decision is keyed by the segment's op sequence and its
  biggest activation rounded to a power of two, shared by all input sizes of an `MlxModule`; a repeated block and the
  same segment at another size take it instead of timing again. Eager won 298 of swin2sr's 1076 trials, 13 of
  patchmatchnet's 132, 9 of psmnet's 45, 6 of vitmatte's 39, so dropping the trials outright was not an option; the memo
  keeps one trial per structure.
- Survey: rows record `load_peak_footprint_mb` (peak after loading) and `second_shape_s` / `second_shape_input` (one
  call at 3/4 size after the timed runs); `first_call_s` was there already. Existing rows stay valid.

Usage: a server that knows its sizes calls `model.warmup([...])` at start-up; a CLI or batch job that restarts often
sets `FASTFORKS_MLX_PLAN_CACHE` (both combine: the warm-up of a later process is then the fast "hit" column).

### `cold_ms` and routing

`cold_ms` = `first_call_s * 1000 - ms` of a survey row: the one-off cost of the first call at a size. Over the 115
`auto` routes of `apple_mlx.json` (rows from before the memo): median 30 ms, 42 above 1 s, 17 above 5 s, 8 above 10 s;
MPS stays at 0.1 to 1.2 s for the same refs. Calls until the MLX route has paid back its extra cold cost
(`(cold_mlx - cold_mps) / (ms_mps - ms_mlx)`): swin2sr-x2 / x4 / x4-real 23 to 32, depth-pro 73, patchmatchnet 35,
vitmatte-s / b 107 / 33, prompt-depth-anything-l 109, geocalib-pinhole 71, nnfoundation-vit 3, but dfine-n-coco 3084,
parseq 1103, u2net 210, hrnet-w32 613, rf-detr-n-coco 301, segformer-b5 344 to 526, isnet-general-use 567;
psmnet-sceneflow never (MLX is 0.7 % slower warm and routes on memory). A one-shot CLI call on these small, fast nets
is slower on MLX than on MPS end to end.

Proposal for `tools/gen_auto_mode.py` (not implemented): read `first_call_s` (and `second_shape_s` where present) from
the rows and keep the `auto` route of a ref on MLX only when its break-even is at most `COLD_CALLS` calls (say 100) or
its `cold_ms` is below a floor (say 1 s); otherwise write the MLX choice under a separate `auto_warm` key that the
loader uses only when the caller warms the model (`Model.warmup` called, or `FASTFORKS_MLX_PLAN_CACHE` set, whose hit
cold cost would then be the one to measure: survey it with the cache warm). The survey rows of the routed refs should
be re-taken first (their `first_call_s` predates the memo, and swin2sr / depth-pro / patchmatchnet more than halved).

Left: MLX's own compile and the translation still run on a hit (1 to 4 s on the mid-sized nets, 4 to 7 s on swin2sr /
depth-pro); `mx.export_function` could store the compiled segments too. The survey rows of the routed refs were not
re-taken with the memo. ppocrv5-mobile's survey first call is 35 s at fp16 against 9.8 s fp32 in this probe (not
investigated; likely one graph per text-crop shape).

### Big nets: export regions and spread tuning (v0.21 follow-up)

`FF_MLX_COLD_LOG=1` prints the first call's cost by phase (export, decomposition, translation) and segment kind (first
call against the fifth). On the big nets the first call was export plus tuning trials:

- igev-stereo fp16: export 29.1 s, of which about 15 s were torch.export's `replace_autocast_with_hop_pass` /
  `replace_set_grad_with_hop_pass`. They split the graph at every no-op `autocast(enabled=False)` and `set_grad_enabled`
  region and recompile the GraphModule each time (quadratic in the regions). `_no_op_regions()` skips autocast regions
  that change nothing (disabled, with autocast already off) and grad toggles to the current mode while exporting under
  `no_grad`: export 29.1 -> 14.6 s (decomposition 8.5 s of it, a full retrace, left as is).
- Tuning trials replayed each segment above 16 MB three more times on the first call (pi3 +28.7 s, igev +8 s). The trial
  is now spread over the segment's first four calls: call 1 compiled, call 2 eager (both untimed, they warm kernels),
  calls 3 and 4 timed (synchronized, peak memory reset), then the same decision rule. No call replays a segment, so the
  first call pays only one compile. Decisions still go to the memo and the plan cache (partial ones are saved as None).
  `FF_MLX_TUNE_FIRST=1` restores the old first-call trials.

Survey (portable, M1 Pro 16 GB, before = committed rows, after = this change):

| ref (dtype) | first_call_s before -> after | warm ms before -> after | peak MB before -> after |
|---|---|---|---|
| igev-stereo-sceneflow (fp16) | 63.3 -> 22.1 | 6026 -> 5678 | 4931 -> 4566 |
| igev-stereo-sceneflow (fp32) | 62.6 -> 26.1 | 6812 -> 6303 | 4599 -> 4404 |
| pi3 (fp16) | 80.9 -> 29.2 | 11593 -> 10309 | 3874 -> 3859 |
| dat-2-x4 (fp16) | 77.4 -> 25.4 | 17232 -> 15757 | 3768 -> 3512 |
| dat-2-x4 (fp32) | 88.9 -> 31.1 | 23853 -> 20494 | 5890 -> 5886 |
| great-igev-sceneflow (fp16) | 62.6 -> 38.5 | 20253 -> 14349 | 9722 -> 9356 |

Not committed: pi3 fp32 (85.1 -> 44.0 s, 11851 -> 10942 ms, but 5526 -> 5591 MB) and great-igev fp32 (118.2 -> 45.2 s,
23111 -> 16330 ms, 9355 -> 10080 MB). The great-igev fp32 peak is the same with `FF_MLX_TUNE_FIRST=1` (10081 MB), so
it comes from earlier changes on the branch, not from this one. Part of the warm gains comes from earlier commits too
(the old rows predate them). Left: the decomposition retrace (5 to 8 s), translation on a page-cache-cold weight file
(pi3: 17 to 22 s cold vs 1 s warm, mmapped safetensors read while converting constants), and memo hits for repeated
blocks whose segment cuts do not line up.

## Iterative stereo pass: half-precision convs, merged siblings, geometry casts (v0.21 follow-up)

Target: MLX at or below MPS on the iterative stereo nets (mvsanywhere, the IGEV / MonSter family, raft-stereo,
waft-stereo-s, s2m2-s). `FF_MLX_PROFILE=1` on the fp16 runs showed the same pattern on all of them: the recurrent
convs ran in float32 although the run dtype was float16, the ragged channel counts (`128 - 1`, `128 + 2 + ...`) fell off
MLX's fast GEMM tiles, and the GRU gates read the same input in separate convs.

Causes and changes (`shared/mlx_graph.py` unless noted; each graph change has an off switch for A/B runs):

- **float32 conv inputs at fp16.** The exported twin is float32, so values the net builds from float32 math (the
  correlation lookups, a `cat` of a float32 lookup with half features) reach the conv as float32 and MLX promotes the
  conv to float32. `_conv_input` casts a float32 input to the weight dtype (`FF_MLX_KEEP_F32_CONV=1`: off).
- **Half cats.** A `cat` whose users are all conv inputs (or other such cats) casts its float32 parts first, so the
  concatenated buffer is half too (`_conv_only_cats`, `_cat_in`): half the bytes for the 384-channel GRU inputs.
- **Geometry casts.** The IGEV / MonSter / Selective / Mocha / GREAT geometry lookups (`*/net/geometry.py`) cast the
  correlation to float32 with literal casts, which the twin records as real casts. They use `to_run_dtype`
  (`fastforks::run_dtype`), which the replay maps to its run dtype; CUDA and MPS are unchanged.
- **Channel alignment.** `_conv2d` zero-pads output channels to a multiple of 16 and wide 3x3 inputs to a multiple of
  32, then slices the output (M1 Pro fp16 256x376: 128 -> 126 3x3 11.3 -> 5.0 ms; 258 -> 128 17.3 -> 10.1 ms).
  `FF_MLX_NO_ALIGN=1`: off.
- **Merged sibling convs.** Convs that read the same value with the same geometry (a GRU's z / r gates, twin heads)
  become one conv over the stacked weights plus channel slices (`_merge_siblings`; `FF_MLX_NO_MERGE=1`: off).
- **avg_pool as strided slice adds.** `_window_sum` sums a separable window with strided slices in float32 instead of
  materialising the `[N, C, OH, OW, kh, kw]` window view (the IGEV pyramid pools).
- **mvsanywhere cost volume** as one Metal kernel instead of the eager CPU torch volume (3.77 s -> 0.51 s fp16).

Survey (M1 Pro 16 GB, before = rows at 64bd11f9, after = this pass; MPS = fastest MPS row of the same dtype group the
cmp script uses):

| ref | mode dtype | ms before -> after | MB before -> after | first call s before -> after | after / MPS |
|---|---|---|---|---|---|
| mvsanywhere | portable fp16 | 3772 -> 506 | 2054 -> 1425 | 12.4 -> 7.4 | 0.49 |
| mvsanywhere | portable fp32 | 3791 -> 635 | 2498 -> 2119 | 16.1 -> 7.4 | 0.61 |
| raft-stereo-sceneflow | kernels fp16 | 4181 -> 2986 | 4678 -> 4650 | 29.8 -> 14.2 | 0.82 |
| raft-stereo-sceneflow | portable fp16 | 5772 -> 4212 | 4708 -> 3926 | 38.5 -> 15.7 | 1.16 |
| raft-stereo-sceneflow | portable fp32 | 8090 -> 6093 | 4729 -> 4705 | 40.5 -> 20.2 | 1.39 |
| igev-plusplus-sceneflow | kernels fp16 | new: 5895 | 3184 | 28.1 | 0.96 |
| igev-plusplus-sceneflow | kernels fp32 | new: 10986 | 4977 | 33.5 | 1.14 |
| igev-plusplus-sceneflow | portable fp16 | 14463 -> 10068 | 3562 -> 2993 | 105.1 -> 31.3 | 1.64 |
| igev-plusplus-sceneflow | portable fp32 | 15882 -> 14770 | 5319 -> 4834 | 114.0 -> 44.7 | 1.53 |
| selective-igev-sceneflow | kernels fp16 | new: 6296 | 2735 | 24.7 | 1.10 |
| selective-igev-sceneflow | portable fp16 | 11925 -> 6659 | 3442 -> 2167 | 68.3 -> 24.3 | 1.17 |
| selective-igev-sceneflow | portable fp32 | 13766 -> 11452 | 3038 -> 2938 | 81.8 -> 29.3 | 1.40 |
| monster-sceneflow | portable fp16 | 19650 -> 10863 | 5189 -> 3633 | 143.7 -> 36.1 | 1.13 |
| monster-plusplus-mix-all | portable fp16 | 18357 -> 11097 (not committed) | 5477 -> 3639 | 137.6 -> 34.2 | 1.09 |
| rt-monster-plusplus-mix-all | kernels fp16 | new: 622 | 2046 | 13.1 | 1.02 |
| rt-monster-plusplus-mix-all | portable fp16 | 842 -> 713 | 2288 -> 2006 | 19.8 -> 13.4 | 1.16 |
| waft-stereo-s | portable fp16 | 6635 -> 3509 | 4330 -> 1835 | 14.2 -> 14.2 | 1.06 |
| waft-stereo-s | portable fp32 | 6903 -> 5474 | 4303 -> 2869 | 16.3 -> 18.9 | 1.69 |
| s2m2-s | portable fp16 | 2938 -> 1835 | 2790 -> 2779 | 21.2 -> 12.7 | 1.14 |

Not committed (old row kept):

- Heavier fp32 rows: raft-stereo kernels (5818 -> 4706 ms, 5590 -> 5852 MB), monster (16562 -> 13918 ms, 5687 -> 5874
  MB), monster++ (17687 -> 14572 ms, 5520 -> 5850 MB), s2m2-s (4036 -> 2897 ms, 3993 -> 4110 MB) and the igev++
  portable fp32 re-run (4834 -> 4911 MB). The extra memory does not come from this pass. On monster++ fp32 with
  `FF_MLX_NO_ALIGN=1 FF_MLX_NO_MERGE=1`, the MLX peak (5505 MB) and the process peak footprint (6.19 GB) are the same as
  with this pass on. The geometry casts and half cats do nothing at fp32. On s2m2-s fp32 the MLX peak is 3025 MB with
  every switch. So the extra memory comes from earlier commits since the 2026-11-20 rows, like the great-igev fp32 note
  above. monster++ fp32 would otherwise route to MLX (0.91x MPS, compile 30.5 s under 3 x 16.1 s).
- monster-plusplus-mix-all fp16: its golden fails (mean |diff| 0.0368 against tolerance 0.03). The pytest golden also
  fails at the head before this pass (0.0359) and with every new switch off (0.0349). The 2026-11-20 row passed at
  0.0248, so the drift came in between, before this pass. Fixed below ("monster++ fp16 golden").

Goldens: nafnet-gopro@1, wd-vit-tagger-v3@1, and every stereo golden in the survey except monster++ fp16 pass.

Routing: none of these rows flips `auto` yet. The runtimes are close, but the compile (first call minus a warm call) is
over `COLD_FACTOR` x the MPS runtime:

- raft-stereo kernels fp16: 11.2 s against the 10.9 s limit.
- igev++ kernels fp16: 22.2 s against 18.4 s; it is also only 4% faster, and `auto` needs 5%.
- mvsanywhere fp16: 6.9 s against 3.1 s.
- rt-monster++: 12.4 s against 1.8 s.

Left:

- **First call.** The iterative nets pay 13 to 36 s of export, decomposition and translation. Cutting that, for
  example with a stored export, is what flips raft-stereo and mvsanywhere.
- **IGEV-family geometry lookup.** It is still a chain of small elementwise ops on `(B*H*W, levels, 2r+2)` tensors,
  about 1 s of eager op time per monster call. A Metal kernel as a custom op, like `gwc_volume`, would remove most of
  them.
- **GRU convs.** The 384 -> 256 / 384 -> 128 3x3 convs at 1/4 resolution are about 1.8 s per monster call. These are
  MLX implicit-GEMM bound, so there is no graph-level fix.
- **ViT encoder.** The depth-anything attention and MLP of MonSter (about 3.5 s eager) are as fast as MLX's
  `scaled_dot_product_attention` and GEMMs go.
- **Not profiled in this pass:** defom, and the flow nets (memfof, memflow, gmflow fp32).

### monster++ fp16 golden: float32 islands on the float32 twin

Bisected (`git bisect --first-parent`, golden mean < 0.029) to the merge 92496fa8; on its side branch, to c9cd93b2
(C-FoundationStereo in float16 end to end). That commit routed DINOv2's interpolated position embedding through
`to_run_dtype`. Before it, the float32 `pos_embed` promoted the ViT tokens, and with them the whole MonSter++ net, to
float32 on the "float16" replay: the 2026-11-20 row (18357 ms / 5477 MB, golden 0.0248) was a float32 run. Since then the
net really runs in float16, and the mean error was 0.035-0.037.

The cause is the float32 islands of the portable path. The MLX runner exports the float32 twin, where `.float()` is a
no-op and is not recorded, so every island ran in float16 on MLX. MPS keeps them. Region by region (that region's
inputs in float32, `FF_MLX_KEEP_F32_CONV=1`), only `feat_transfer` / `feat_transfer_cnet` mattered (0.0215; the mono
branch, cost aggregation, update blocks and upsampling / REMP stayed at 0.032-0.037). Those are the modules with the
`InstanceNorm2d`s: `_in_forward` normalises in float32 per element on MPS and in float16 on MLX.

Fix: `fastforks._core.mlx_cast.to_float32` records `fastforks::float32`, which `mlx_graph` lowers to a float32 cast
(layout kept). MonSter++'s instance norm takes the float16 branch while exporting for MLX (float32 statistics and
normalisation, then `to_run_dtype`). The other `.float()` islands of `forward_fast` (disparity state, regression and
upsampling softmaxes, warps, depth_mono, REMP's residual) use `to_float32` too, as the docstring promises. Alone they did
not move the golden (0.0366).

| monster-plusplus-mix-all portable fp16 | ms | MB | first call s | golden mean |
|---|---|---|---|---|
| 2026-11-20 row (float32 by accident) | 18357 | 5477 | 137.6 | 0.0248 |
| head, float16 | 11097 | 3639 | 34.2 | 0.0368 (fail) |
| fixed | 11269 | 3643 | 32.6 | 0.0232 |

The MPS float16 rows are 0.0254 (portable) and 0.0230 (kernels). The fp32 re-run (14435 ms / 5957 MB) is heavier than
the old row, so it is not committed. monster (0.0011) and the other iterative stereo float16 goldens pass unchanged:
rt-monster++, igev++, selective-igev, raft-stereo, s2m2-s, waft-stereo-s, igev-stereo, great-igev and mvsanywhere.
MonSter has the same latent instance-norm island, but its golden passes with a wide margin, so it is unchanged.

### fp32 peak "regression": a measurement artifact, not code

The float32 MLX peaks of raft-stereo kernels, monster, monster++, s2m2-s, great-igev and pi3 looked 1-8 % above the
2026-11-19/20 rows. Re-surveying the old commit 85384184 today with the same tool:

| ref (float32 portable) | old row | 85384184 today | head today |
|---|---|---|---|
| monster++ | 5520 MB | 14898 ms / 6114 MB | 14435 ms / 5957 MB |
| s2m2-s | 3993 MB | 2938 ms / 4103 MB | 4110 MB |

Head is no heavier than the old code. The old rows were taken while about 10.6 GB of stale swap was resident (see above).
Those rows were also slower. Under that memory pressure the OS compresses or swaps pages, so `phys_footprint` reads lower.
The rows stay as they are until a clean full re-run.

The monster++ peak sits inside single lazy segments, not across them. Segment 57 is a conv-heavy encoder segment that
peaks at 5755 MB with only 2486 MB live on entry. Every GRU iteration segment also peaks at about 5440 MB. Shrinking only
the encoder segment would therefore gain nothing. `FF_MLX_CHUNK_MB=64` cuts the footprint to 5356 MB, but warm time goes
from 14.4 s to 19.4 s. `FF_MLX_COMPILE=0` and `FASTFORKS_MLX_CACHE_MB=1` leave the peak unchanged. So there is no cheap win.

## MPS lessons from MLX

The MPS rows that used much more memory than MLX were dominated by the allocator, not by the model. Allocations of
10-512 MB go into 1 GB MTLHeaps, and allocations above 512 MB get heaps of their own size. A freed buffer is reused
only after its command buffer completes. So a long queue of stages keeps every stage's temporaries alive, and one
100-500 MB temporary in the wrong place opens another 1 GB heap (sometimes one more on the first warm call). MLX
avoids both by evaluating graph segments and keeping a small buffer cache. The fixes ported to the portable code,
all MPS-gated unless noted:

- **`fence`** (`_core/mps.py`, a `torch.mps.synchronize` at a stage boundary): it lets the next stage reuse the last
  stage's memory. Use it only where profiling shows a new heap. The extra las2 fnet fences cost 4.6% and saved nothing.
- **Chunk the big temporaries** below about 64 MB: strips (vitmatte), selective-volume chunks (rt-monster++), one
  group / neighbour at a time (patchmatchnet correlation and evaluation weights), one warp position at a time plus
  smaller init-cost row chunks (hitnet), and token chunks in the feed-forwards (video-depth-anything). The smaller
  ops are often faster too (hitnet fp16 -37%).
- **Fold BatchNorm into the convolutions at load**, in float32 before the dtype cast (psmnet). Folding works on every
  device and is not MPS-gated.
- **fp16 GELU in place** for GEGLU (`F.gelu(gate).mul_(x)`), with no float32 gate copy (video-depth-anything, as on
  CUDA).

| ref (M1 Pro 16 GB, portable MPS) | before ms / MB | after ms / MB | golden |
|---|---|---|---|
| vitmatte-s fp16 | 1360 / 2542 | 1254 / 1530 | pass |
| vitmatte-s fp32 | 2333 / 5185 | 2195 / 2602 | pass |
| rt-monster-plusplus-mix-all fp16 | 682 / 7047 | 657 / 4062 | pass |
| rt-monster-plusplus-mix-all fp32 | 663 / 6986 | 669 / 4016 | pass |
| las2-s fp16 | 192 / 3590 | 189 / 2588 | pass |
| las2-s fp32 | 245 / 3983 | 176 / 2999 | pass |
| patchmatchnet fp16 | 1574 / 6591 | 1012 / 3612 | pass |
| patchmatchnet fp32 | 1010 / 7593 | 1002 / 3608 | pass |
| psmnet-sceneflow fp16 | 2579 / 6428 | 2442 / 4783 | pass |
| psmnet-sceneflow fp32 | 4615 / 11257 | 4150 / 8663 | pass |
| hitnet-eth3d fp16 | 501 / 5510 | 313 / 2463 | pass |
| hitnet-eth3d fp32 | 484 / 5503 | 307 / 4507 | pass |
| video-depth-anything-s fp16 480x640 (profiled, warm) | 906 / 5537 | 899 / 4491 | pass |
| video-depth-anything-s fp16 994x1482 (profiled, warm) | 1008 / 6661 | 1000 / 4618 | pass |

Route flips in `apple_mlx.json`: vitmatte-s fp16 and psmnet-sceneflow fp16 moved from MLX to MPS. patchmatchnet fp16
stays on MLX (920 ms there). depthsplat-small-re10k rows were stale with no code change: fp16 757 / 8900 -> 734 / 2763,
fp32 670 / 8880 -> 682 / 3781.

**Not improved, and why:**

- **Small classifiers and detectors** stay about 1 GB above MLX. This is the first 1 GB heap; model code cannot
  remove it.
- **video-depth-anything** was not re-surveyed. The survey plan now picks 994x1482, while the existing rows are
  480x640.
- **grounding-dino-tiny**: the fp32 row (4787 MB, from 11-12) looks stale. A current profile peaks at 3790 MB warm.
  The remaining growth is the transformers Swin `relative_position_bias + attention_mask` add, which is upstream code.

**Left:**

- **inspyrenet fp32** at 1080x1920: 6.3 GB lifetime peak against 2.5 GB allocated. Candidates:
  - the Swin-v1 `attn_mask` add
  - `torch.cat` / `F.interpolate` in `attention_module.py`
  - BN folding in `layers.py`
- **Not looked at:** sam3, depth-pro, disk-lightglue, dat x4, and foundationstereo (too close to 16 GB to profile
  safely).

## Final MLX vs MPS table

Every ref whose MLX row predated cc3fce13 (128 MB cache default) or the later compile-policy/channels-last/cache changes was re-measured on the M1 Pro 16 GB (warm median, one job at a time). Ratios are MLX/MPS, below 1 favours MLX. Row shown: float16 where both dtypes exist, else float32; best MLX mode of portable/kernels. WIN: faster and memory within 1.02x; TIE: within 3% speed, memory not above MPS; LOSE otherwise. 224 refs: 91 WIN, 5 TIE, 128 LOSE.

Not measurable under the 9 GB swap rule (the shared Mac held 10.6 GB stale swap from other apps, jobs were killed; they keep their old rows): depth-pro, vggt-1b, anysplat, lingbot-map, resplat-base-re10k, great-igev-sceneflow, waft-stereo-s, alltracker, cotracker3-offline/online, locotrack-b, flowformer-small-chairs, gimm-vfi-r. tapir, tapnext, track-on2, nnfoundation-*, nninteractive and totalsegmentator-* were re-surveyed on 2026-11-20 (rows above). The memory-fix sweeps (compile=False, cache_mb 128/256/512) for the memory losers were not run for the same reason; `FASTFORKS_MLX_COMPILE=0` (sweep switch in `mlx_wrap.port`) and `FASTFORKS_MLX_CACHE_MB` are ready for them. oneformer-swinl-ade20k and rt-monster-plusplus-mix-all reported golden FAIL on MLX; both pass now (rt-monster after the `_Normalize` trace fix) and stay unrouted because MLX is slower.

| ref | dtype | input | MLX mode | ms | MB | MPS ms | MPS MB | speed | mem | verdict | why |
|---|---|---|---|---|---|---|---|---|---|---|---|
| aanet-sceneflow | 16 | 994x1482 | portable | 1695.2 | 2131 | 630.844 | 2606 | 2.69 | 0.82 | LOSE | slower (2.69x: portable MLX replay, no fused attention/conv kernels) |
| aliked-flexglue | 16 | 768x1024 | portable | 1100.4 | 1387 | 1068.521 | 3088 | 1.03 | 0.45 | LOSE | 2.9 % slower, 55 % leaner (dense ALIKED maps on MLX now); the matcher decode is SDPA-bound |
| aliked-lightglue | 32 | 768x1024 | portable | 460.6 | 2416 | 205.897 | 2586 | 2.24 | 0.93 | LOSE | slower: the MLX dense maps + NMS alone take ~93 ms per image (MPS 81), and SDDH descriptors and the matcher run eager on the CPU (~140 ms); unrouted |
| alltracker | 32 | 480x854 | portable | 5397.2 | 2597 | 3507.681 | 4571 | 1.54 | 0.57 | LOSE | slower (1.54x: portable MLX replay, no fused attention/conv kernels) |
| any2full-b | 16 | 994x1482 | portable | 378.6 | 1583 | 267.954 | 2198 | 1.41 | 0.72 | LOSE | slower (1.41x: portable MLX replay, no fused attention/conv kernels) |
| any2full-l | 16 | 994x1482 | portable | 908.5 | 3298 | 684.288 | 3956 | 1.33 | 0.83 | LOSE | slower (1.33x: portable MLX replay, no fused attention/conv kernels) |
| any2full-s | 16 | 994x1482 | portable | 159.7 | 1012 | 117.84 | 1641 | 1.36 | 0.62 | LOSE | slower (1.36x: portable MLX replay, no fused attention/conv kernels) |
| anycalib-pinhole | 32 | 1080x1920 | portable | 341.8 | 2089 | 266.497 | 3023 | 1.28 | 0.69 | LOSE | slower (1.28x: portable MLX replay, no fused attention/conv kernels) |
| anyline | 16 | 1024x1024 | portable | 197.5 | 1776 | 116.786 | 1319 | 1.69 | 1.35 | LOSE | slower and more memory: the whole module (TEED + line-art post-processing) runs as one MLX graph; the cost is the full-res (1280^2) 24-channel depthwise convs and Smish chains; fp32 223 / 1920 vs MPS 130 / 2346 |
| anysplat | 16 | 336x518 | portable | 6973.7 | 8297 | 5444.081 | 8751 | 1.28 | 0.95 | LOSE | slower (1.28x: portable MLX replay, no fused attention/conv kernels) |
| ben2 | 32 | 1080x1920 | portable | 1705.7 | 5012 | 1147.957 | 6252 | 1.49 | 0.80 | LOSE | slower (1.49x: portable MLX replay, no fused attention/conv kernels) |
| birefnet | 16 | 1080x1920 | portable | 1203.1 | 2705 | 1151.581 | 5433 | 1.04 | 0.50 | LOSE | slower (1.04x: portable MLX replay, no fused attention/conv kernels) |
| bisenet-face-parsing | 16 | 512x512 | portable | 17.6 | 607 | 13.643 | 440 | 1.29 | 1.38 | LOSE | slower (1.29x: portable MLX replay, no fused attention/conv kernels); more memory (1.38x: MLX runtime/buffer cache above the MPS footprint); tiny net: ~100 MB fixed cost of the torch.export imports plus the MLX runtime, not fixable by cache/chunk/compile switches |
| boq-dinov2 | 16 | 480x640 | portable | 55.3 | 864 | 48.219 | 849 | 1.15 | 1.02 | LOSE | slower (1.15x: portable MLX replay, no fused attention/conv kernels) |
| bridgedepth-sf | 16 | 994x1482 | portable | 5661.3 | 4483 | 3693.197 | 6270 | 1.53 | 0.71 | LOSE | slower (1.53x: portable MLX replay, no fused attention/conv kernels) |
| c-radio-v3-b | 16 | 224x224 | portable | 16.4 | 764 | 16.494 | 2230 | 0.99 | 0.34 | WIN |  |
| c-radio-v3-g | 16 | 224x224 | portable | 158.0 | 4960 | 144.655 | 9394 | 1.09 | 0.53 | LOSE | slower (1.09x: portable MLX replay, no fused attention/conv kernels) |
| c-radio-v3-h | 16 | 224x224 | portable | 93.5 | 2983 | 84.991 | 5806 | 1.10 | 0.51 | LOSE | slower (1.10x: portable MLX replay, no fused attention/conv kernels) |
| c-radio-v3-l | 16 | 224x224 | portable | 47.9 | 1635 | 47.91 | 3322 | 1.00 | 0.49 | WIN |  |
| cas-gwcnet-gc-sceneflow | 16 | 994x1482 | portable | 3840.6 | 4754 | 2995.289 | 7182 | 1.28 | 0.66 | LOSE | slower (1.28x: portable MLX replay, no fused attention/conv kernels) |
| clip-b16-openai | 16 | 224x224 | portable | 15.3 | 1478 | 16.051 | 1724 | 0.95 | 0.86 | WIN |  |
| clip-l14-laion2b | 16 | 224x224 | portable | 70.8 | 2951 | 56.555 | 4424 | 1.25 | 0.67 | LOSE | slower (1.25x: portable MLX replay, no fused attention/conv kernels) |
| clipseg | 16 | 1024x1024 | portable | 38.5 | 969 | 37.732 | 1561 | 1.02 | 0.62 | TIE |  |
| coex-sceneflow | 16 | 994x1482 | portable | 608.3 | 3187 | 411.053 | 3576 | 1.48 | 0.89 | LOSE | slower (1.48x: portable MLX replay, no fused attention/conv kernels) |
| convnext_base.fb_in1k | 16 | 224x224 | portable | 16.2 | 1212 | 24.79 | 2307 | 0.65 | 0.53 | WIN |  |
| convnext_base.fb_in22k | 16 | 224x224 | portable | 18.1 | 1368 | 24.495 | 2473 | 0.74 | 0.55 | WIN |  |
| convnext_base.fb_in22k_ft_in1k | 16 | 224x224 | kernels | 16.2 | 1212 | 24.928 | 2308 | 0.65 | 0.53 | WIN |  |
| convnext_base.fb_in22k_ft_in1k_384 | 16 | 224x224 | portable | 34.1 | 1210 | 45.218 | 2309 | 0.75 | 0.52 | WIN |  |
| convnext_large.fb_in1k | 16 | 224x224 | kernels | 29.0 | 2340 | 37.956 | 3485 | 0.76 | 0.67 | WIN |  |
| convnext_large.fb_in22k | 16 | 224x224 | kernels | 29.2 | 2574 | 37.501 | 3704 | 0.78 | 0.69 | WIN |  |
| convnext_large.fb_in22k_ft_in1k | 16 | 224x224 | portable | 29.1 | 2340 | 39.022 | 3485 | 0.75 | 0.67 | WIN |  |
| convnext_large.fb_in22k_ft_in1k_384 | 16 | 224x224 | portable | 63.4 | 2340 | 79.609 | 3494 | 0.80 | 0.67 | WIN |  |
| convnext_small.fb_in1k | 16 | 224x224 | kernels | 12.3 | 813 | 18.214 | 1044 | 0.67 | 0.78 | WIN |  |
| convnext_small.fb_in22k | 16 | 224x224 | portable | 12.1 | 931 | 17.95 | 1166 | 0.68 | 0.80 | WIN |  |
| convnext_small.fb_in22k_ft_in1k | 16 | 224x224 | kernels | 12.2 | 812 | 18.683 | 1044 | 0.66 | 0.78 | WIN |  |
| convnext_small.fb_in22k_ft_in1k_384 | 16 | 224x224 | kernels | 25.0 | 811 | 30.999 | 1043 | 0.81 | 0.78 | WIN |  |
| convnext_tiny.fb_in1k | 16 | 224x224 | portable | 7.3 | 642 | 10.213 | 748 | 0.71 | 0.86 | WIN |  |
| convnext_tiny.fb_in22k | 16 | 224x224 | kernels | 7.5 | 761 | 9.957 | 871 | 0.75 | 0.87 | WIN |  |
| convnext_tiny.fb_in22k_ft_in1k | 16 | 224x224 | portable | 7.5 | 644 | 10.834 | 748 | 0.69 | 0.86 | WIN |  |
| convnext_tiny.fb_in22k_ft_in1k_384 | 16 | 224x224 | portable | 15.2 | 714 | 18.051 | 749 | 0.84 | 0.95 | WIN |  |
| convnextv2_base.fcmae_ft_in22k_in1k | 16 | 224x224 | kernels | 19.7 | 1192 | 35.792 | 2323 | 0.55 | 0.51 | WIN |  |
| cotracker3-offline | 32 | 480x854 | portable | 3577.5 | 3438 | 1430.353 | 4725 | 2.50 | 0.73 | LOSE | slower (2.50x: portable MLX replay, no fused attention/conv kernels) |
| cotracker3-online | 32 | 480x854 | portable | 3340.3 | 3971 | 1706.522 | 4689 | 1.96 | 0.85 | LOSE | slower (1.96x: portable MLX replay, no fused attention/conv kernels) |
| crestereo-eth3d | 16 | 994x1482 | kernels | 3557.9 | 1863 | 4126.635 | 2801 | 0.86 | 0.67 | WIN |  |
| da3-giant-1.1 | 16 | 994x1482 | portable | 919.9 | 3858 | 758.549 | 7243 | 1.21 | 0.53 | LOSE | slower (1.21x: portable MLX replay, no fused attention/conv kernels) |
| da3-large-1.1 | 16 | 994x1482 | portable | 361.0 | 1975 | 318.477 | 2520 | 1.13 | 0.78 | LOSE | slower (1.13x: portable MLX replay, no fused attention/conv kernels) |
| da3metric-large | 16 | 994x1482 | portable | 328.0 | 1899 | 279.886 | 2499 | 1.17 | 0.76 | LOSE | slower (1.17x: portable MLX replay, no fused attention/conv kernels) |
| da3mono-large | 16 | 994x1482 | portable | 325.1 | 1898 | 279.76 | 2499 | 1.16 | 0.76 | LOSE | slower (1.16x: portable MLX replay, no fused attention/conv kernels) |
| da3nested-giant-large-1.1 | 16 | 994x1482 | portable | 1268.9 | 4267 | 1184.373 | 8136 | 1.07 | 0.52 | LOSE | slower (1.07x: portable MLX replay, no fused attention/conv kernels) |
| dat-2-x4 | 16 | 512x512 | portable | 17231.6 | 3768 | 9404.292 | 10600 | 1.83 | 0.36 | LOSE | slower (1.83x: portable MLX replay, no fused attention/conv kernels) |
| dat-light-x4 | 16 | 512x512 | portable | 4091.9 | 2782 | 2562.37 | 6851 | 1.60 | 0.41 | LOSE | slower (1.60x: portable MLX replay, no fused attention/conv kernels) |
| dat-x4 | 16 | 512x512 | portable | 16649.6 | 3779 | 10511.755 | 10620 | 1.58 | 0.36 | LOSE | slower (1.58x: portable MLX replay, no fused attention/conv kernels) |
| dedode-v2-g | 32 | 768x1024 | portable | 5544.3 | 6499 | 3767.204 | 10008 | 1.47 | 0.65 | LOSE | slower (1.47x: portable MLX replay, no fused attention/conv kernels) |
| defom-stereo-vitl-sceneflow | 32 | 994x1482 | portable | 15943.9 | 6974 | 11105.073 | 13138 | 1.44 | 0.53 | LOSE | slower (1.44x: portable MLX replay, no fused attention/conv kernels) |
| depth-anything-v2-metric-hypersim-b | 16 | 994x1482 | kernels | 169.7 | 1027 | 199.296 | 2062 | 0.85 | 0.50 | WIN |  |
| depth-anything-v2-metric-hypersim-l | 16 | 994x1482 | portable | 526.8 | 1831 | 579.185 | 3283 | 0.91 | 0.56 | WIN |  |
| depth-anything-v2-metric-hypersim-s | 16 | 994x1482 | kernels | 68.5 | 710 | 79.524 | 1572 | 0.86 | 0.45 | WIN |  |
| depth-anything-v2-metric-vkitti-b | 16 | 994x1482 | portable | 188.1 | 1021 | 199.299 | 2061 | 0.94 | 0.50 | WIN |  |
| depth-anything-v2-metric-vkitti-l | 16 | 994x1482 | portable | 543.9 | 1728 | 579.971 | 3283 | 0.94 | 0.53 | WIN |  |
| depth-anything-v2-metric-vkitti-s | 16 | 994x1482 | kernels | 67.4 | 707 | 79.467 | 1572 | 0.85 | 0.45 | WIN |  |
| depth-pro | 16 | 994x1482 | portable | 4653.3 | 5490 | 5011.45 | 7169 | 0.93 | 0.77 | WIN | (was 4491 / 11370 LOSE: float32 twin kept, 2 GB segments) |
| depthsplat-small-re10k | 32 | 336x518 | portable | 1269.9 | 2671 | 669.5 | 8880 | 1.90 | 0.30 | LOSE | slower (1.90x: portable MLX replay, no fused attention/conv kernels) |
| dfine-n-coco | 16 | 800x1344 | portable | 22.3 | 624 | 25.174 | 1510 | 0.89 | 0.41 | WIN |  |
| dinov2-b-linear-ade20k | 16 | 1024x2048 | portable | 761.1 | 1611 | 360.569 | 2174 | 2.11 | 0.74 | LOSE | slower (2.11x: portable MLX replay, no fused attention/conv kernels) |
| dinov2-l-linear-ade20k | 16 | 1024x2048 | portable | 1277.9 | 2388 | 927.202 | 3167 | 1.38 | 0.75 | LOSE | slower (1.38x: portable MLX replay, no fused attention/conv kernels) |
| dinov2-s-linear-ade20k | 16 | 1024x2048 | portable | 600.3 | 1232 | 194.951 | 1532 | 3.08 | 0.80 | LOSE | slower (3.08x: portable MLX replay, no fused attention/conv kernels) |
| dinov2-vitg14 | 16 | 224x224 | portable | 192.8 | 2634 | 218.016 | 4504 | 0.88 | 0.58 | WIN |  |
| dinov2-vitg14-reg | 16 | 224x224 | portable | 203.4 | 2630 | 260.65 | 4505 | 0.78 | 0.58 | WIN |  |
| dinov2-vitl14-reg | 16 | 224x224 | portable | 58.2 | 1029 | 95.672 | 1566 | 0.61 | 0.66 | WIN |  |
| dinov2_vitl14_lc | 16 | 224x224 | portable | 57.6 | 3981 | 59.274 | 4443 | 0.97 | 0.90 | WIN |  |
| dinov3-convnext-base | 16 | 224x224 | kernels | 16.0 | 512 | 23.12 | 880 | 0.69 | 0.58 | WIN |  |
| dinov3-convnext-large | 16 | 224x224 | portable | 29.9 | 822 | 36.928 | 2088 | 0.81 | 0.39 | WIN |  |
| dinov3-convnext-small | 16 | 224x224 | portable | 11.9 | 388 | 18.49 | 612 | 0.64 | 0.63 | WIN |  |
| dinov3-convnext-tiny | 16 | 224x224 | portable | 7.2 | 315 | 10.645 | 431 | 0.68 | 0.73 | WIN |  |
| dinov3-vith16plus | 16 | 224x224 | portable | 119.9 | 2174 | 143.346 | 3467 | 0.84 | 0.63 | WIN |  |
| disk-lightglue | 32 | 768x1024 | portable | 1177.6 | 2333 | 1994.053 | 3531 | 0.59 | 0.66 | WIN | (new port: DISK U-Net as an MLX graph; keypoints and matcher eager on the CPU) |
| distill-any-depth-b | 16 | 994x1482 | kernels | 169.6 | 1020 | 199.366 | 2061 | 0.85 | 0.49 | WIN |  |
| distill-any-depth-l | 16 | 994x1482 | portable | 529.2 | 1847 | 579.665 | 3281 | 0.91 | 0.56 | WIN |  |
| distill-any-depth-s | 16 | 994x1482 | kernels | 66.9 | 698 | 79.298 | 1570 | 0.84 | 0.44 | WIN |  |
| efficientnet_b0.tv_in1k | 16 | 224x224 | portable | 4.4 | 345 | 13.55 | 503 | 0.32 | 0.69 | WIN |  |
| efficientnet_b4.tv_in1k | 16 | 224x224 | kernels | 16.0 | 649 | 38.529 | 683 | 0.42 | 0.95 | WIN |  |
| efficientnet_v2_s.tv_in1k | 16 | 224x224 | kernels | 16.7 | 552 | 36.052 | 661 | 0.46 | 0.84 | WIN |  |
| efficientvit-sam-l0 | 32 | 1024x1024 | portable | 112.7 | 1102 | 50.612 | 1779 | 2.23 | 0.62 | LOSE | slower (2.23x: portable MLX replay, no fused attention/conv kernels) |
| ema-vfi-small-t | 16 | 1080x1920 | portable | 876.2 | 2460 | 513.231 | 3583 | 1.71 | 0.69 | LOSE | slower (1.71x: portable MLX replay, no fused attention/conv kernels) |
| eomt-l-dinov2-ade20k | 16 | 1024x2048 | portable | 572.5 | 3710 | 540.048 | 4262 | 1.06 | 0.87 | LOSE | slower (1.06x: portable MLX replay, no fused attention/conv kernels) |
| eomt-l-dinov3-ade20k | 16 | 1024x2048 | portable | 599.3 | 3724 | 574.509 | 4259 | 1.04 | 0.87 | LOSE | slower (1.04x: portable MLX replay, no fused attention/conv kernels) |
| fast-acvnet-generalization | 16 | 994x1482 | portable | 549.2 | 1881 | 478.214 | 2737 | 1.15 | 0.69 | LOSE | slower (1.15x: portable MLX replay, no fused attention/conv kernels) |
| foundationstereo-vit-s | 16 | 994x1482 | portable | 20504.1 | 4943 | 19495.754 | 12980 | 1.05 | 0.38 | LOSE | slower than the MPS kernels (1.05x), at par with portable MPS (20584 ms); see the FoundationStereo pass |
| geocalib-pinhole | 16 | 1080x1920 | portable | 352.3 | 1013 | 427.123 | 1644 | 0.82 | 0.62 | WIN |  |
| gfpgan-v1.4 | 32 | 512x512 | portable | 318.4 | 1673 | 153.218 | 1954 | 2.08 | 0.86 | LOSE | slower (2.08x: portable MLX replay, no fused attention/conv kernels) |
| gmflow-scale1-things | 16 | 1080x1920 | portable | 6821.6 | 10465 | 10676.45 | 12643 | 0.64 | 0.83 | WIN |  |
| great-igev-sceneflow | 16 | 994x1482 | portable | 20253.2 | 9722 | 9861.251 | 10126 | 2.05 | 0.96 | LOSE | slower (2.05x: portable MLX replay, no fused attention/conv kernels) |
| grounding-dino-tiny | 32 | 800x1344 | portable | 1043.4 | 1645 | 845.995 | 4787 | 1.23 | 0.34 | LOSE | slower (1.23x: portable MLX replay, no fused attention/conv kernels) |
| gwcnet-gc-sceneflow | 16 | 994x1482 | portable | 2933.5 | 2876 | 2454.734 | 7108 | 1.20 | 0.40 | LOSE | slower (1.20x: portable MLX replay, no fused attention/conv kernels) |
| hed | 32 | 1024x1024 | portable | 81.9 | 1144 | 64.901 | 1415 | 1.26 | 0.81 | LOSE | slower (1.26x: portable MLX replay, no fused attention/conv kernels) |
| hrnet-w32 | 16 | 1080x1920 | portable | 116.9 | 911 | 121.77 | 1675 | 0.96 | 0.54 | WIN |  |
| hrnet-w48-fcn-ade20k | 16 | 1024x2048 | portable | 220.3 | 1112 | 228.267 | 2044 | 0.97 | 0.54 | WIN |  |
| igev-plusplus-sceneflow | 16 | 994x1482 | portable | 14463.2 | 3562 | 6135.163 | 4972 | 2.36 | 0.72 | LOSE | slower (2.36x: portable MLX replay, no fused attention/conv kernels) |
| insid3-vitl16 | 32 | 1024x1024 | portable | 3211.1 | 3003 | 2953.299 | 2963 | 1.09 | 1.01 | LOSE | slower (1.09x: portable MLX replay, no fused attention/conv kernels) |
| inspyrenet | 32 | 1080x1920 | portable | 1273.3 | 2332 | 1107.579 | 7284 | 1.15 | 0.32 | LOSE | slower (1.15x: portable MLX replay, no fused attention/conv kernels) |
| isnet-general-use | 16 | 1080x1920 | kernels | 122.2 | 1328 | 129.932 | 1746 | 0.94 | 0.76 | WIN |  |
| las2-s | 16 | 994x1482 | portable | 229.7 | 988 | 192.074 | 3590 | 1.20 | 0.28 | LOSE | slower (1.20x: portable MLX replay, no fused attention/conv kernels) |
| lineart | 16 | 1024x1024 | portable | 108.2 | 1099 | 66.53 | 1362 | 1.63 | 0.81 | LOSE | slower (1.63x: portable MLX replay, no fused attention/conv kernels) |
| lineart-anime | 16 | 1024x1024 | portable | 44.5 | 982 | 51.126 | 1774 | 0.87 | 0.55 | WIN |  |
| lingbot-depth-dc | 16 | 994x1482 | portable | 5338.5 | 2987 | 3373.689 | 3230 | 1.58 | 0.92 | LOSE | slower (1.58x: portable MLX replay, no fused attention/conv kernels) |
| lingbot-map | 16 | 480x640 | portable | 10046.2 | 8220 | 7124.932 | 8834 | 1.41 | 0.93 | LOSE | slower (1.41x: portable MLX replay, no fused attention/conv kernels) |
| locotrack-b | 32 | 480x854 | portable | 2039.2 | 1679 | 2104.282 | 2474 | 0.97 | 0.68 | WIN |  |
| loftr-outdoor | 32 | 768x1024 | portable | 2017.5 | 3685 | 875.487 | 5767 | 2.30 | 0.64 | LOSE | slower (2.30x: portable MLX replay, no fused attention/conv kernels) |
| loma-b | 32 | 768x1024 | portable | 3859.6 | 6150 | 3037.483 | 8147 | 1.27 | 0.75 | LOSE | slower (1.27x: portable MLX replay, no fused attention/conv kernels) |
| lotus-normal-g-v1-1 | 16 | 994x1482 | portable | 2004.6 | 5674 | 1552.612 | 6632 | 1.29 | 0.86 | LOSE | slower (1.29x: portable MLX replay, no fused attention/conv kernels) |
| lw-detr-tiny-coco | 16 | 800x1344 | portable | 34.9 | 935 | 41.22 | 1742 | 0.85 | 0.54 | WIN |  |
| mapanything-apache | 16 | 336x518 | portable | 8875.8 | 8371 | 6369.19 | 7218 | 1.39 | 1.16 | LOSE | slower (1.39x: portable MLX replay, no fused attention/conv kernels); more memory (1.16x: MLX runtime/buffer cache above the MPS footprint) |
| marigold-depth-v1-1 | 16 | 994x1482 | portable | 4701.8 | 4287 | 2497.717 | 6640 | 1.88 | 0.65 | LOSE | slower (1.88x: portable MLX replay, no fused attention/conv kernels) |
| marigold-normals-lcm-v0-1 | 16 | 994x1482 | portable | 2363.0 | 4318 | 1654.841 | 6671 | 1.43 | 0.65 | LOSE | slower (1.43x: portable MLX replay, no fused attention/conv kernels) |
| mask-rcnn-r50-fpn-3x-coco | 16 | 800x1344 | portable | 411.2 | 1533 | 202.766 | 1931 | 2.03 | 0.79 | LOSE | slower (2.03x: portable MLX replay, no fused attention/conv kernels) |
| mask2former-swinl-ade20k | 16 | 1024x2048 | portable | 1148.1 | 3795 | 963.996 | 4503 | 1.19 | 0.84 | LOSE | slower (1.19x: portable MLX replay, no fused attention/conv kernels) |
| megaloc | 16 | 480x640 | portable | 87.3 | 1609 | 45.229 | 2617 | 1.93 | 0.61 | LOSE | slower (1.93x: portable MLX replay, no fused attention/conv kernels) |
| memfof | 32 | 1080x1920 | portable | 5435.9 | 2716 | 3054.409 | 5038 | 1.78 | 0.54 | LOSE | slower (1.78x: portable MLX replay, no fused attention/conv kernels) |
| metric3d-vit-large | 16 | 994x1482 | portable | 2460.2 | 2769 | 1861.345 | 3636 | 1.32 | 0.76 | LOSE | slower (1.32x: portable MLX replay, no fused attention/conv kernels) |
| midas-dpt-beit-large-512 | 16 | 994x1482 | portable | 668.2 | 1994 | 634.656 | 2407 | 1.05 | 0.83 | LOSE | slower (1.05x: portable MLX replay, no fused attention/conv kernels) |
| mlsd | 16 | 1024x1024 | portable | 35.8 | 676 | 44.568 | 1373 | 0.80 | 0.49 | WIN |  |
| mobile-sam | 32 | 1024x1024 | portable | 101.6 | 947 | 80.819 | 1480 | 1.26 | 0.64 | LOSE | slower (1.26x: portable MLX replay, no fused attention/conv kernels) |
| mobilenet_v2.tv2_in1k | 16 | 224x224 | portable | 2.8 | 319 | 8.084 | 444 | 0.34 | 0.72 | WIN |  |
| mobilenet_v3_large.tv2_in1k | 16 | 224x224 | portable | 3.3 | 335 | 8.964 | 493 | 0.37 | 0.68 | WIN |  |
| mobilenetv1_100.ra4_e3600_r224_in1k | 16 | 224x224 | portable | 2.3 | 367 | 5.129 | 461 | 0.45 | 0.80 | WIN |  |
| mocha-v1-sceneflow | 16 | 994x1482 | portable | 9657.0 | 3382 | 6693.339 | 5695 | 1.44 | 0.59 | LOSE | slower (1.44x: portable MLX replay, no fused attention/conv kernels) |
| moge-2-vitl-normal | 16 | 994x1482 | portable | 2080.7 | 3009 | 1731.978 | 3265 | 1.20 | 0.92 | LOSE | slower (1.20x: portable MLX replay, no fused attention/conv kernels) |
| moge-3-vitl | 16 | 994x1482 | portable | 3256.8 | 3170 | 3913.491 | 5968 | 0.83 | 0.53 | WIN | the trunk at parity (1.7 s); the sparse refiner on MLX (`mlx_refine.py`: gather implicit GEMM Metal kernel per conv, binary-search neighbour tables), 1.5 s for 3 steps vs MPS 2.15 s |
| monster-plusplus-mix-all | 16 | 994x1482 | portable | 18356.8 | 5477 | 10202.192 | 7347 | 1.80 | 0.75 | LOSE | slower (1.80x: portable MLX replay, no fused attention/conv kernels) |
| monster-sceneflow | 16 | 994x1482 | portable | 19649.8 | 5189 | 9642.315 | 6299 | 2.04 | 0.82 | LOSE | slower (2.04x: portable MLX replay, no fused attention/conv kernels) |
| mvsanywhere | 16 | 448x672 | portable | 3772.5 | 2054 | 1064.367 | 3801 | 3.54 | 0.54 | LOSE | slower (3.54x: portable MLX replay, no fused attention/conv kernels) |
| mvsplat-re10k | 16 | 336x518 | portable | 2417.5 | 2796 | 1111.616 | 4537 | 2.17 | 0.62 | LOSE | slower (2.17x: portable MLX replay, no fused attention/conv kernels) |
| nnfoundation-cnn | 16 | 200x200 | portable | 3519.7 | 8041 | 2935.289 | 7824 | 1.20 | 1.03 | LOSE | slower (1.20x: `mx.conv3d` ~2.4 TFLOPS; was 1.61x / 1.29x before the channels-last norm statistics and the stem conv fold) |
| nnfoundation-vit | 16 | 192x192 | portable | 13036.5 | 2469 | 17636.415 | 3817 | 0.74 | 0.65 | WIN |  |
| nninteractive | 32 | 200x200 | portable | 4005.3 | 5996 | 3524.991 | 11439 | 1.14 | 0.52 | LOSE | slower (1.14x: `mx.conv3d`; was 1.36x) |
| omni-dc | 16 | 994x1482 | portable | 5978.0 | 5375 | 2408.429 | 10081 | 2.48 | 0.53 | LOSE | slower (2.48x: portable MLX replay, no fused attention/conv kernels) |
| oneformer-swinl-ade20k | 16 | 1024x2048 | portable | 2078.1 | 6033 | 1160.815 | 5559 | 1.79 | 1.09 | LOSE | slower (1.79x: portable MLX replay, no fused attention/conv kernels); more memory (1.09x: MLX runtime/buffer cache above the MPS footprint); golden passes (test_mlx and survey), unrouted |
| owlv2-b16 | 32 | 800x1344 | portable | 398.1 | 1299 | 393.679 | 1515 | 1.01 | 0.86 | TIE |  |
| parseq | 16 | 32x128 | portable | 17.9 | 495 | 23.603 | 440 | 0.76 | 1.12 | LOSE | more memory (1.12x: MLX runtime/buffer cache above the MPS footprint); tiny net: ~100 MB fixed cost of the torch.export imports plus the MLX runtime, not fixable by cache/chunk/compile switches |
| patchmatchnet | 16 | 1536x1024 | portable | 920.2 | 2247 | 1573.872 | 6591 | 0.58 | 0.34 | WIN |  |
| pe-core-l14-336 | 16 | 224x224 | portable | 95.7 | 3240 | 58.343 | 4528 | 1.64 | 0.72 | LOSE | slower (1.64x: portable MLX replay, no fused attention/conv kernels) |
| pe-core-t16-384 | 16 | 224x224 | portable | 5.1 | 469 | 7.253 | 388 | 0.70 | 1.21 | LOSE | more memory (1.21x: MLX runtime/buffer cache above the MPS footprint) |
| pi3 | 16 | 336x518 | portable | 11593.4 | 3874 | 8340.992 | 6510 | 1.39 | 0.60 | LOSE | slower (1.39x: portable MLX replay, no fused attention/conv kernels) |
| pidinet | 16 | 1024x1024 | kernels | 75.1 | 732 | 73.21 | 1349 | 1.03 | 0.54 | TIE |  |
| pointdit-b-512 | 16 | 994x1482 | portable | 356.0 | 1545 | 337.166 | 1912 | 1.06 | 0.81 | LOSE | slower (1.06x: portable MLX replay, no fused attention/conv kernels) |
| pointdit-l-512 | 16 | 994x1482 | portable | 1231.4 | 3360 | 1013.331 | 4548 | 1.22 | 0.74 | LOSE | slower (1.22x: portable MLX replay, no fused attention/conv kernels) |
| ppocrv5-mobile | 16 | 1080x1920 | portable | 869.1 | 1610 | 735.803 | 2987 | 1.18 | 0.54 | LOSE | slower (1.18x: portable MLX replay, no fused attention/conv kernels) |
| ppocrv5-server | 16 | 1080x1920 | portable | 1903.8 | 2666 | 1607.126 | 6545 | 1.18 | 0.41 | LOSE | slower (1.18x: portable MLX replay, no fused attention/conv kernels) |
| prompt-depth-anything-l | 16 | 994x1482 | portable | 1082.0 | 2757 | 1142.683 | 3303 | 0.95 | 0.83 | WIN |  |
| prompt-depth-anything-s | 16 | 756x1008 | portable | 244.0 | 1168 | 194.524 | 1588 | 1.25 | 0.74 | LOSE | slower (1.25x: portable MLX replay, no fused attention/conv kernels) |
| psmnet-sceneflow | 16 | 994x1482 | portable | 2597.3 | 2941 | 2578.896 | 6428 | 1.01 | 0.46 | TIE |  |
| raft-stereo-middlebury | 16 | 994x1482 | portable | 5704.4 | 4315 | 3633.675 | 4722 | 1.57 | 0.91 | LOSE | slower (1.57x: portable MLX replay, no fused attention/conv kernels) |
| recover-cx | 32 | 1080x1920 | portable | 1755.1 | 1933 | 1106.503 | 4781 | 1.59 | 0.40 | LOSE | slower (1.59x: portable MLX replay, no fused attention/conv kernels) |
| regnety_160.tv2_in1k | 16 | 224x224 | portable | 18.9 | 1047 | 33.084 | 2030 | 0.57 | 0.52 | WIN |  |
| repvit_m0_9.dist_300e_in1k | 16 | 224x224 | portable | 4.3 | 384 | 13.241 | 470 | 0.33 | 0.82 | WIN |  |
| repvit_m0_9.dist_450e_in1k | 16 | 224x224 | portable | 4.3 | 384 | 11.973 | 470 | 0.36 | 0.82 | WIN |  |
| repvit_m1_0.dist_300e_in1k | 16 | 224x224 | portable | 5.2 | 410 | 12.292 | 493 | 0.42 | 0.83 | WIN |  |
| repvit_m1_0.dist_450e_in1k | 16 | 224x224 | portable | 5.3 | 414 | 12.528 | 493 | 0.42 | 0.84 | WIN |  |
| repvit_m1_1.dist_300e_in1k | 16 | 224x224 | portable | 4.4 | 425 | 11.882 | 508 | 0.37 | 0.84 | WIN |  |
| repvit_m1_1.dist_450e_in1k | 16 | 224x224 | portable | 4.4 | 425 | 12.006 | 510 | 0.37 | 0.83 | WIN |  |
| repvit_m1_5.dist_300e_in1k | 16 | 224x224 | portable | 12.1 | 510 | 19.804 | 582 | 0.61 | 0.88 | WIN |  |
| repvit_m1_5.dist_450e_in1k | 16 | 224x224 | portable | 6.9 | 506 | 20.036 | 585 | 0.34 | 0.86 | WIN |  |
| repvit_m2_3.dist_300e_in1k | 16 | 224x224 | portable | 11.1 | 630 | 27.864 | 736 | 0.40 | 0.86 | WIN |  |
| repvit_m2_3.dist_450e_in1k | 16 | 224x224 | portable | 10.7 | 628 | 27.494 | 734 | 0.39 | 0.86 | WIN |  |
| resnet152.tv2_in1k | 16 | 224x224 | kernels | 17.0 | 728 | 24.271 | 1101 | 0.70 | 0.66 | WIN |  |
| resnet18.tv_in1k | 16 | 224x224 | kernels | 3.8 | 382 | 5.371 | 481 | 0.70 | 0.79 | WIN |  |
| resnet50.tv2_in1k | 16 | 224x224 | kernels | 6.8 | 500 | 10.153 | 667 | 0.67 | 0.75 | WIN |  |
| restormer-deblur | 16 | 512x512 | portable | 2531.8 | 2666 | 1475.684 | 3657 | 1.72 | 0.73 | LOSE | slower (1.72x: portable MLX replay, no fused attention/conv kernels) |
| restormer-derain | 16 | 512x512 | portable | 2814.9 | 2668 | 1813.619 | 3655 | 1.55 | 0.73 | LOSE | slower (1.55x: portable MLX replay, no fused attention/conv kernels) |
| restormer-sidd | 16 | 512x512 | portable | 2426.8 | 2672 | 2329.566 | 3683 | 1.04 | 0.73 | LOSE | slower (1.04x: portable MLX replay, no fused attention/conv kernels) |
| retinaface-resnet50 | 16 | 1080x1920 | portable | 184.8 | 1165 | 150.321 | 1913 | 1.23 | 0.61 | LOSE | slower (1.23x: portable MLX replay, no fused attention/conv kernels) |
| rf-detr-keypoint-preview | 32 | 1080x1920 | portable | 92.6 | 997 | 147.777 | 1830 | 0.63 | 0.54 | WIN |  |
| rf-detr-n-coco | 32 | 800x1344 | portable | 22.1 | 714 | 31.806 | 1718 | 0.69 | 0.42 | WIN |  |
| rife-4.25 | 16 | 1080x1920 | portable | 210.9 | 1680 | 105.552 | 2482 | 2.00 | 0.68 | LOSE | slower (2.00x: portable MLX replay, no fused attention/conv kernels) |
| rife-4.25-lite | 16 | 1080x1920 | portable | 202.2 | 1636 | 79.991 | 2447 | 2.53 | 0.67 | LOSE | slower (2.53x: portable MLX replay, no fused attention/conv kernels) |
| romav1 | 32 | 768x1024 | portable | 4439.8 | 4499 | 3050.476 | 7979 | 1.46 | 0.56 | LOSE | slower (1.46x: portable MLX replay, no fused attention/conv kernels) |
| romav2 | 32 | 768x1024 | kernels | 4042.6 | 3150 | 3980.808 | 6725 | 1.02 | 0.47 | TIE |  |
| rt-monster-plusplus-mix-all | 16 | 994x1482 | portable | 842.1 | 2288 | 682.191 | 7047 | 1.23 | 0.32 | LOSE | slower (1.23x); golden passes since `_Normalize` no longer caches its constants while traced (a second export at another shape found the first export's fake tensors); fp32 1038 / 2451 vs MPS 663 / 6986. Unrouted (speed). |
| rtdetrv2-s-coco | 16 | 800x1344 | portable | 39.5 | 904 | 44.377 | 1631 | 0.89 | 0.55 | WIN |  |
| rtmdet-tiny-coco | 16 | 800x1344 | portable | 24.4 | 618 | 26.979 | 1545 | 0.91 | 0.40 | WIN |  |
| rtmo-t | 32 | 1080x1920 | portable | 17.9 | 667 | 20.977 | 1523 | 0.85 | 0.44 | WIN |  |
| rtmpose-t | 16 | 1080x1920 | portable | 25.6 | 665 | 25.633 | 1460 | 1.00 | 0.46 | WIN |  |
| s2m2-s | 16 | 994x1482 | portable | 2938.3 | 2790 | 1607.273 | 5991 | 1.83 | 0.47 | LOSE | slower (1.83x: portable MLX replay, no fused attention/conv kernels) |
| sam-hq-vit-b | 16 | 1024x1024 | portable | 703.3 | 2588 | 682.56 | 4542 | 1.03 | 0.57 | LOSE |  |
| sam-vit-b | 16 | 1024x1024 | portable | 666.4 | 2571 | 600.197 | 3762 | 1.11 | 0.68 | LOSE | slower (1.11x: portable MLX replay, no fused attention/conv kernels) |
| sam2.1-hiera-tiny | 32 | 1024x1024 | portable | 184.1 | 1174 | 241.468 | 2581 | 0.76 | 0.45 | WIN |  |
| sapiens-seg-0.3b-goliath | 16 | 1024x2048 | portable | 2392.1 | 3921 | 1517.056 | 3748 | 1.58 | 1.05 | LOSE | slower (1.58x: portable MLX replay, no fused attention/conv kernels); more memory (1.05x: MLX runtime/buffer cache above the MPS footprint) |
| sapiens-seg-1b-goliath | 16 | 1024x2048 | portable | 4110.0 | 6692 | 3112.873 | 6340 | 1.32 | 1.06 | LOSE | slower (1.32x: portable MLX replay, no fused attention/conv kernels); more memory (1.06x: MLX runtime/buffer cache above the MPS footprint) |
| segformer-b5-ade20k | 16 | 1024x2048 | portable | 357.2 | 1382 | 375.167 | 1974 | 0.95 | 0.70 | WIN |  |
| segformer-b5-face-parsing | 16 | 512x512 | portable | 100.6 | 963 | 110.76 | 1899 | 0.91 | 0.51 | WIN |  |
| segnext-mscan-b-ade20k | 16 | 1024x2048 | portable | 263.7 | 894 | 184.121 | 1659 | 1.43 | 0.54 | LOSE | slower (1.43x: portable MLX replay, no fused attention/conv kernels) |
| segnext-mscan-l-ade20k | 16 | 1024x2048 | portable | 424.1 | 974 | 268.202 | 1840 | 1.58 | 0.53 | LOSE | slower (1.58x: portable MLX replay, no fused attention/conv kernels) |
| selective-igev-sceneflow | 16 | 994x1482 | portable | 11924.9 | 3442 | 5709.633 | 3907 | 2.09 | 0.88 | LOSE | slower (2.09x: portable MLX replay, no fused attention/conv kernels) |
| siglip2-b16-256 | 16 | 224x224 | portable | 16.6 | 1306 | 15.803 | 1563 | 1.05 | 0.84 | LOSE | slower (1.05x: portable MLX replay, no fused attention/conv kernels) |
| siglip2-so400m14-378 | 16 | 224x224 | portable | 70.8 | 3850 | 68.57 | 5553 | 1.03 | 0.69 | LOSE |  |
| starnet_s1.in1k | 16 | 224x224 | portable | 3.1 | 342 | 14.046 | 400 | 0.22 | 0.85 | WIN |  |
| starnet_s2.in1k | 16 | 224x224 | portable | 2.7 | 348 | 7.212 | 437 | 0.38 | 0.80 | WIN |  |
| starnet_s3.in1k | 16 | 224x224 | portable | 3.3 | 375 | 9.548 | 469 | 0.35 | 0.80 | WIN |  |
| starnet_s4.in1k | 16 | 224x224 | portable | 4.4 | 397 | 12.753 | 491 | 0.34 | 0.81 | WIN |  |
| swin2sr-x2 | 16 | 512x512 | portable | 8019.8 | 2849 | 9257.892 | 3501 | 0.87 | 0.81 | WIN |  |
| swin2sr-x4 | 16 | 512x512 | portable | 7994.1 | 3266 | 9409.594 | 4580 | 0.85 | 0.71 | WIN |  |
| swin2sr-x4-real | 16 | 512x512 | portable | 8554.5 | 2851 | 9530.53 | 4579 | 0.90 | 0.62 | WIN |  |
| swinir-l-x4-real | 16 | 512x512 | portable | 14530.2 | 3161 | 9873.92 | 4561 | 1.47 | 0.69 | LOSE | slower (1.47x: portable MLX replay, no fused attention/conv kernels) |
| swinir-m-x2 | 16 | 512x512 | portable | 7248.2 | 2443 | 5796.813 | 3500 | 1.25 | 0.70 | LOSE | slower (1.25x: portable MLX replay, no fused attention/conv kernels) |
| tapir | 32 | 480x854 | portable | 1592.5 | 1717 | 1382.473 | 2630 | 1.15 | 0.65 | LOSE | slower (1.15x; was 2.00x before the depthwise conv1d fast path) |
| tapnext | 32 | 480x854 | portable | 7477.0 | 3673 | 5855.372 | 3754 | 1.28 | 0.98 | LOSE | slower (1.28x: portable MLX replay) |
| teed | 16 | 1024x1024 | portable | 19.2 | 492 | 16.675 | 1313 | 1.15 | 0.37 | LOSE | slower (1.15x: portable MLX replay, no fused attention/conv kernels) |
| tipsv2-l14 | 16 | 224x224 | portable | 72.9 | 1038 | 61.697 | 1574 | 1.18 | 0.66 | LOSE | slower (1.18x: portable MLX replay, no fused attention/conv kernels) |
| totalsegmentator-total | 16 | 200x200 | portable | 31549.0 | 5163 | 23069.574 | 6058 | 1.37 | 0.85 | LOSE | slower (1.37x: `mx.conv3d` and the CPU window glue; was 1.86x / 1.09x) |
| totalsegmentator-total-3mm | 16 | 200x200 | portable | 1110.8 | 4730 | 521.63 | 2588 | 2.13 | 1.83 | LOSE | slower (2.13x: graph 495 ms alone, plus ~0.5 s CPU window glue; was 2.56x) |
| track-on2 | 32 | 480x854 | portable | 10754.8 | 3318 | 8353.935 | 6102 | 1.29 | 0.54 | LOSE | fp32 slower (1.29x; was 2.57x before the per-frame step graph); fp16 7623 / 3056 WIN, routed |
| u2net | 16 | 1080x1920 | portable | 62.9 | 1156 | 79.422 | 1885 | 0.79 | 0.61 | WIN |  |
| unidepth-v2-vitl14 | 32 | 994x1482 | portable | 1691.2 | 3855 | 1252.486 | 6077 | 1.35 | 0.63 | LOSE | slower (1.35x: portable MLX replay, no fused attention/conv kernels) |
| unik3d-vitl | 16 | 994x1482 | portable | 1457.6 | 2388 | 1293.645 | 4038 | 1.13 | 0.59 | LOSE | 13 % slower, 41 % leaner (edge pads as slices now); the ViT-L encoder dominates; fp32 1396 / 3000 vs MPS 1217 / 4001 |
| vggt-1b | 16 | 336x518 | portable | 8774.2 | 6853 | 6637.772 | 8423 | 1.32 | 0.81 | LOSE | slower (1.32x: portable MLX replay, no fused attention/conv kernels) |
| vggt-omega-1b-512 | 16 | 336x518 | portable | 8564.1 | 5137 | 7241.942 | 8602 | 1.18 | 0.60 | LOSE | slower (1.18x: portable MLX replay, no fused attention/conv kernels) |
| video-depth-anything-metric-s | 16 | 480x640 | portable | 871.6 | 2021 | 933.206 | 6535 | 0.93 | 0.31 | WIN |  |
| video-depth-anything-s | 16 | 480x640 | portable | 896.1 | 2027 | 926.453 | 6506 | 0.97 | 0.31 | WIN |  |
| vitmatte-b | 16 | 1080x1920 | portable | 2368.0 | 2474 | 2740.0 | 2966 | 0.86 | 0.83 | WIN |  |
| vitmatte-s | 16 | 1080x1920 | portable | 1280.5 | 2283 | 1360.354 | 2542 | 0.94 | 0.90 | WIN |  |
| vitpose-base-simple | 16 | 1080x1920 | portable | 168.6 | 1080 | 159.7 | 1977 | 1.06 | 0.55 | LOSE | slower (1.06x: portable MLX replay, no fused attention/conv kernels) |
| waft-a1 | 16 | 1080x1920 | portable | 9339.5 | 3559 | 7167.849 | 7185 | 1.30 | 0.50 | LOSE | slower (1.30x: portable MLX replay, no fused attention/conv kernels) |
| waft-stereo-s | 16 | 994x1482 | portable | 6634.9 | 4330 | 6089.657 | 9209 | 1.09 | 0.47 | LOSE | slower (1.09x: portable MLX replay, no fused attention/conv kernels) |
| xfeat | 32 | 768x1024 | portable | 164.4 | 538 | 41.694 | 1471 | 3.94 | 0.37 | LOSE | slower (3.94x: portable MLX replay, no fused attention/conv kernels) |
| yolox-tiny | 16 | 800x1344 | kernels | 12.6 | 470 | 20.062 | 1492 | 0.63 | 0.32 | WIN |  |
| yunet-n | 16 | 1080x1920 | portable | 22.3 | 717 | 21.593 | 1377 | 1.03 | 0.52 | LOSE |  |
| zoedepth-nk | 32 | 994x1482 | portable | 861.9 | 2895 | 777.684 | 3558 | 1.11 | 0.81 | LOSE | slower (1.11x: portable MLX replay, no fused attention/conv kernels) |

### Decision: `auto` takes MLX only where its compile is cheap (2026-11-21)

MLX compiles per input shape, so its first call is slower than MPS's. `tools/gen_auto_mode.py` promotes MLX in `auto` /
`dtypes` only if, on every surveyed workload, the compile (`first_call_s` minus one warm call) is under `COLD_FACTOR`
(3) times the MPS runtime it replaces, on top of the speed / memory rule. Depth Pro (26 s against 5 s) and the other
slow-compiling nets stay on MPS by default; small nets with a first call of a few ms qualify. Everything MLX is
available with `device="mlx"`. The `mlx` / `mlx_default` tables (models MPS cannot run, such as Marigold-v2) and the
explicit `kernels` table are unchanged. `FASTFORKS_MLX_PLAN_CACHE` and `Model.warmup` remain the way to take the
first-call cost out of a deployed model.

## Swin window attention pass (v0.21 follow-up, 2026-11-21)

SwinIR-L at 512x512 fp16 spent 13.8 s warm (MPS 9.9 s). Causes, in profile order:

- Masked attention at head size 32 (`[4225, 8, 64, 32]` per block): MLX fuses sdpa only for head sizes
  64/72/80/96/128 and zero-pads the rest to 64, and the summed `bias[None] + shift_mask[:, None]` mask was built at
  `[B, H, L, L]` each block. Now `ops.window_attention`: one threadgroup per (window, head), K and V staged in
  threadgroup memory, S = QK^T and PV by 8x8 simdgroup MMA, float32 softmax, the (at most two) mask parts read where
  they lie (`b % MB`, head-broadcast). 13 ms against 23.7 ms for MLX sdpa (fp16), 24 against 35-58 ms (fp32). Used only
  for unfused head sizes; `_sdpa_mask_parts` peels `expand(repeat(a + b))` so the mask is never materialised.
- q, k, v sliced out of `qkv.permute(2, 0, 3, 1, 4)`: MLX copies each slice of a transposed array (19 ms of 31 per
  block). `_sdpa_qkv_args` peels the slices and the kernel reads the `[B, L, 3, H, D]` projection in place; it writes
  `[B, L, H, D]` and returns a transposed view, so the following `transpose(1, 2).reshape` is free (154 ms per call).
- `aten.roll` decomposed to an index gather: kept (`mx.roll`), about 3x faster.
- GELU as five eager passes: `mx.compile(shapeless=True)` makes it one kernel.
- About half of the plan was constant work (relative-position bias gathers, shift masks, per-block lifted scalars):
  `_fold_constants` evaluates steps whose inputs are all constants once at build time (4959 -> 2448 steps on SwinIR-L;
  small constants are deduplicated by value; capped at 512 MB of folded values).

Off switches: `FF_MLX_NO_WIN_ATTN` (kernel and mask peeling), `FF_MLX_NO_WIN_QKV` (fused-qkv read),
`FF_MLX_NO_GELU_COMPILE`, `FF_MLX_NO_CONST_FOLD`. Goldens pass (swinir-l-x4-real, swinir-m-x2, swin2sr-x2,
swin2sr-x4-real, dat-2-x4, dat-light-x4, hat-s-x4, swin_base window7, mask2former-swinl, nafnet-gopro,
wd-vit-tagger-v3).

| ref (512x512) | dtype | before ms / MB / first s | after ms / MB / first s | MPS ms / MB |
|---|---|---|---|---|
| swinir-l-x4-real | fp16 | 14530 / 3161 / 45.8 | 8691 / 2437 / 17.4 | 9874 / 4561 |
| swinir-l-x4-real | fp32 | 26151 / 4087 / 38.5 | 14252 / 4075 / 23.2 | 12063 / - |
| swinir-m-x2 | fp16 | 7248 / 2443 / 31.9 | 4312 / 2423 / 10.4 | 5797 / 3500 |
| swinir-m-x2 | fp32 | 13788 / 3456 / 46.8 | 6857 / 2865 / 12.8 | - |
| swin2sr-x2 | fp16 | 7619 / 2875 / 17.1 | 5144 / 1980 / 11.9 | 9258 / 3501 |
| swin2sr-x2 | fp32 | 14183 / 3876 / 52.2 | 8277 / 2881 / 15.0 | - |
| swin2sr-x4-real | fp16 | 8555 / 2851 / 39.5 | 5522 / 2268 / 12.3 | 9531 / 4579 |
| swin2sr-x4-real | fp32 | 14296 / 4082 / 53.2 | 8813 / 3869 / 15.6 | - |
| swin2sr-x4 | fp16 | 7994 / 3266 / 39.7 | 5360 / 3322 / 12.4 (not recorded: +56 MB) | 9410 / 4580 |
| swin2sr-x4 | fp32 | 13955 / 5925 / 55.2 | 8631 / 5983 / 15.8 (not recorded: +58 MB) | - |

Left on SwinIR-L fp16 (eager profile, per call): linears 4.5 s (already ahead of MPS), attention 1.0 s, roll plus
window partition / reverse copies 0.9 s (a fused roll-and-window gather would save about 0.5 s), GELU 0.64 s, residual
adds 0.62 s, layer norm 0.51 s, convs 1.0 s.

### DAT: depthwise conv, narrow layer norm, half-precision mask sum

DAT-2 at 512x512 fp16 spent 15.8 s warm (MPS 9.4 s). Causes, in profile order:

- `mx.conv2d(groups=C)` falls off its fast path at some channel counts: DAT's 180-channel 3x3 depthwise conv at 512^2
  took 75 ms against 2.6 ms for `ops.depthwise` (302 against 9.7 ms at 1024^2), 72 per call, 5.7 s in all. The
  depthwise kernel, so far a kernels-mode kernel, now runs in portable mode too (never slower in the sweep).
- `mx.fast.layer_norm` is slow at narrow or odd widths (47M fp16 elements: C=30 118 ms, C=96 25 ms, C=180 8 ms,
  C=384 5.8 ms). `ops.row_layer_norm` (one simdgroup per row, the row in registers, float32 mean and centred
  variance) takes about 1.3 ms at every C up to 512; used for 1-D layer norms with C <= 512.
- DAT's d=30, L=256 attention stays on MLX sdpa (the window kernel spills registers at L=256: 119 against 23 ms), but
  its peeled mask parts were summed in float32 before `_sdpa` cast them: the eager tuning trial held a float32
  `[1024, 1, 256, 256]` (call peak 2807 -> 2208 MB). `_sdpa_parts` now casts each part to the run dtype first.

Off switches: `FF_MLX_NO_DW_KERNEL`, `FF_MLX_NO_ROW_LN`. `FF_MLX_CONST_FOLD_MB` sets the folded-constant cap (default
512): at 1024 DAT folds its two 256 MB float32 shift masks, 0.6 s faster per call but 512 MB heavier, so the default
stays. Goldens pass (dat-2-x4, dat-light-x4, hat-s-x4, swinir-l-x4-real, swin2sr-x2, swin_base window7, nafnet-gopro,
wd-vit-tagger-v3, convnext_base, depth-anything-v2-b, davit_small).

| ref (512x512) | dtype | before ms / MB / first s | after ms / MB / first s | MPS ms / MB |
|---|---|---|---|---|
| dat-2-x4 | fp16 | 15757 / 3512 / 25.4 | 8676 / 3515 / 18.0 (not recorded: +3 MB; another run 8681 / 3538) | 9404 / 10600 |
| dat-2-x4 | fp32 | 20494 / 5886 / 31.1 | 12210 / 5878 / 22.8 | - |

Left on DAT fp16 (eager profile, per call): linears 1.9 s, d=30 attention 1.7 s, convs 1.1 s, reshape copies 0.6 s,
`vector_norm` 0.6 s (eager abs and cast; one compiled kernel would take 40 % of that), global mean 0.5 s, the shift
mask's where / ne / eq 0.6 s (folded at `FF_MLX_CONST_FOLD_MB=1024`).

The window kernel keeps a row of scores per simdgroup in registers, which spill beyond about L = 96: at HAT's
16x16 windows (L = 256, D = 24, fp16) it took 212 against 46 ms for MLX sdpa (L = 64 / 96: 13 / 12 against
18 / 28 ms). Windows longer than `FF_MLX_WIN_ATTN_MAX_L` (default 96) now take MLX sdpa with the summed mask parts.
The dat-light and hat-s rows below come from the same code.

| ref (512x512) | dtype | before ms / MB / first s | after ms / MB / first s |
|---|---|---|---|
| dat-light-x4 | fp16 | 4092 / 2782 / 21.2 | 1859 / 2408 / 7.3 |
| dat-light-x4 | fp32 | 5319 / 3730 / 25.0 | 2678 / 3346 / 8.1 |
| hat-s-x4 | fp16 | 12676 / 5361 / 48.3 | 10618 / 4539 / 17.0 (window kernel at L = 256: 14866 / 3406) |
| hat-s-x4 | fp32 | 20203 / 7380 / 79.7 | 15572 / 6016 / 21.1 |

`test_mlx[hat-x4@1]` (fp16, mean |diff| 0.00092 against 0.00083) already failed at c15d5667, before this pass.

## Real-ESRGAN: 3x3 convs with 32 / 48 outputs

`mx.conv2d` picks a narrow, slow tile for 32 / 48 output channels. At 512x512, a 160 -> 32 3x3 conv takes 16.5 ms
(fp16) / 27.8 ms (fp32), and 13.2 / 15.6 ms with the output padded to 64. 64 -> 32 fp32: 11.2 -> 6.8 ms, 160 -> 48
fp32: 21.5 -> 15.8 ms. 32 -> 32, fp16 64 -> 32, 8-24 outputs and 1x1 kernels gain nothing or lose. `_conv2d` now pads
the stride-1, undilated 3x3 convs with 48 outputs, or 32 outputs from >= 128 inputs (>= 64 in fp32), to 64 and slices
the result. RRDB's dense blocks are made of these (160 / 128 / 96 / 64 -> 32). Off switch: `FF_MLX_NO_WIDE_OUT`.
Goldens pass (realesrgan-x4plus, realesrgan-x4plus-anime-6b, nafnet-gopro, wd-vit-tagger-v3, convnext_base).

| ref (512x512) | dtype | before ms / MB / first s | after ms / MB / first s |
|---|---|---|---|
| realesrgan-x4plus | fp16 | 5320 / 1390 / 24.3 | 4766 / 1463 / 8.5 (not recorded: heavier) |
| realesrgan-x4plus | fp32 | 9349 / 2186 / 37.9 | 7548 / 2180 / 11.5 |
| realesrgan-x2plus | fp16 | 1199 / 974 / 8.3 | 1139 / 992 / 4.8 (not recorded) |
| realesrgan-x2plus | fp32 | 1599 / 1047 / 10.1 | 1489 / 1064 / 5.4 (not recorded) |
| realesrgan-x4plus-anime-6b | fp16 | 1617 / 1319 / 7.3 | 1484 / 1336 / 3.2 (not recorded) |
| realesrgan-x4plus-anime-6b | fp32 | 2700 / 2040 / 10.8 | 2293 / 2046 / 4.0 (not recorded) |

The x2plus and anime-6b rows are 6-18 MB heavier, which is within run-to-run spread. x4plus fp16 is 73 MB heavier.
Same-session footprints of x4plus fp16 were 1467 MB on and 1440 MB off, so most of that gap predates this change.
Each padded 64-wide output is held until its slice is consumed.
Left on x4plus fp16: the dense-block `_cat_in` concats (0.93 s per call), the leaky-relu lambda (0.52 s), and the
banded upsample (0.33 s).

## Marigold: wide-input 3x3 convs in input slices

The SD UNet decoder's convs on concatenated skips (2560 / 1920 inputs at 12-48 px) are slow in `mx.conv2d`. On an
M1 Pro, a 3x3 at 24^2 with 2560 -> 1280 takes 26.5 ms in fp16 and 60.4 ms in fp32. The same conv as four 640-input
convs summed takes 12.2 / 14.5 ms. At 12^2 the times are 6.9 -> 3.8 ms (fp16) and 15.5 -> 5.6 ms (fp32). In fp32,
24^2 1280 -> 1280 goes from 12.7 to 7.4 ms. An im2col matmul or a per-tap matmul only wins at small sizes and loses
badly above 96 px. `_conv2d` now splits stride-1, undilated 3x3 convs with at most 48x48 positions into 640-input
slices when they have >= 1920 inputs, or >= 1280 inputs in fp32 at <= 24x24 positions. Off switch:
`FF_MLX_NO_SPLIT_IN`. Same-session marigold-depth at 512^2 (fp.py), on against off: fp16 4.73 against 4.90 s (+111 MB),
fp32 7.50 against 8.38 s (same MB). The table compares against rows from 85384184, so it also includes the earlier
passes. Goldens pass (marigold-depth-v1-1, marigold-normals-lcm-v0-1, nafnet-gopro, wd-vit-tagger-v3,
realesrgan-x4plus, convnext_base).

| ref (994x1482) | dtype | before ms / MB / first s | after ms / MB / first s | MPS ms / MB |
|---|---|---|---|---|
| marigold-depth-v1-1 | fp16 | 4702 / 4287 / 31.9 | 2867 / 4176 / 19.3 | 2498 / 6640 |
| marigold-depth-v1-1 | fp32 | 5831 / 7406 / 42.3 | 4441 / 7268 / 37.8 | 2970 / 9709 |
| marigold-normals-lcm-v0-1 | fp16 | 2363 / 4318 / 38.7 | 1770 / 4220 / 19.5 | 1655 / 6671 |
| marigold-normals-lcm-v0-1 | fp32 | 5109 / 7471 / 44.6 | 3025 / 7310 / 38.3 | 2075 / 9727 |

Measured, then left as is (see "Measured conv dispatch" below):
`mx.conv2d`'s 3x3 timings jump with the position count and channel width. These are M1 Pro numbers,
taken outside any net. In fp32, 1024 -> 768 at 63x64 takes 19.5 ms and at 64x64 34.2 ms (MLX appears to switch to
Winograd at 4096 positions); 1024 -> 1024 takes 27.6 ms against 50.9 ms. In fp16, 1024 -> 1024 takes 23.2 ms against
33.8 ms. 2560 -> 1280 at 63x64 takes 435 ms in fp32 and 164 ms in fp16. At 64^2, a 3x3 run as nine per-tap matmuls
(1024 -> 1024: 19.1 / 23.6 ms) or as 640-input slices (1024 -> 768 fp16: 10.5 ms) beats `mx.conv2d`. memfof's
1024-input convs at 1/8 scale and the SD UNet's middle levels sit in this band. A dispatch picked from a measured table
(per position count and channel pair) should replace these hand rules. Scripts: cv3.py-cv7.py in the session
scratchpad.

MLX's Winograd is also much less accurate in half precision: relative mean error 6e-3 against 3.5e-4 for the direct
conv in float16 (bfloat16 4.8e-2 against 2.8e-3; float32 4e-6), exactly where its dispatch switches (dense 3x3, stride
and dilation 1, channels multiples of 32 summing to >= 256, >= 4096 positions). `_conv2d`'s ragged-channel padding
(d7a21332) owed its padded-input gains to entering it, and pushed HAT's 180-channel convs over hat-x4's float16 golden
(mean 0.00058 -> 0.00092). Half precision outside kernels mode now never pads into Winograd: hat-x4 fp16 golden 0.00057,
11550 ms / 3651 MB; hat-s-x4 10010 / 3396, dat-x4 9358 / 3305, dat-light-x4 1870 / 2425 (no slower than the rows before).

## Measured conv dispatch: tried, not kept (2026-11-21)

Question: should a table measured per conv shape replace `_conv2d`'s hand rules? Those rules are the ragged-channel
padding, the 32 / 48 -> 64 output pad, the 640-input split and the half-precision Winograd guard. Answer: no. No real net
gains more than 5% from such a table, so the rules stay as they are.

Method: the shapes and call counts of every `_conv2d` call were logged at the survey inputs of memfof, memflow,
defom-stereo, raft-stereo, sea-raft-m, marigold-depth, realesrgan-x4plus, hat-x4 and nafnet (366 distinct keys over
fp32 / fp16). Each key was timed alone (warm median) as:

- a direct `mx.conv2d`
- the current rules
- output pads to 16 / 32 / 64
- an input pad to 32
- input slices of 256-1024
- 16 extra outputs, which keeps MLX off Winograd
- nine per-tap matmuls

A half-precision candidate counted as unsafe if it entered Winograd where the direct conv does not.

The winners, each beating the rules by at least 10% and 0.3 ms, became a 20-entry table. Its key was dtype, kernel,
stride, cin, cout and the position-count bucket round(2 log2(N H W)). Its biggest isolated wins:

| Conv | Net | Lowering | Before -> after (ms) |
|---|---|---|---|
| fp32 stride-2 3x3 stem 64 -> 96 at 1024x1504 | raft-stereo / defom | outputs padded to 128 | 52.0 -> 18.5 |
| the same stem at 540x960 | memflow / sea-raft | outputs padded to 128 | 17.2 -> 6.5 |
| fp32 1x1 256 -> 144 | raft-stereo / defom | outputs padded to 192 | 15.1 -> 3.7 |
| fp32 1024 -> 768 at 68x120 (Winograd band) | memfof | 320-input slices | 40.9 -> 27.0 |
| fp32 1280 -> 1280 at 32x48 | marigold | 320-input slices | 33.6 -> 15.8 |
| fp32 512 -> 512 at 64x96 | marigold | 256-input slices | 10.2 -> 5.5 |
| hat fp32 60 -> 180 | hat | both padded to 32 multiples | 19.6 -> 14.1 |
| fp16 960-1920 -> 640 / 1280 at 32x48 | marigold | per-tap matmuls | 15-22% faster |

The fp16 per-tap matmuls have a relative error of 4.8e-4, against 1.8e-4 for the direct conv; Winograd's is 6e-3. The
fp32 lowerings stay at about 1e-5. Summed over each net's call counts, the table cut the estimated conv time to between
0.69x (raft-stereo fp32) and 0.94x (defom) of the rules. All goldens passed with it.

Whole nets, on against off (`FF_MLX_NO_CONV_TABLE`, same session, warm median of 5):

| Net | dtype | Table on (ms / MB) | Table off (ms / MB) | Change |
|---|---|---|---|---|
| memfof | fp32 | 3427 / 3179 | 3572 / 3013 | -4.1%, +166 MB |
| raft-stereo-sceneflow | fp32 | 4456 / 5928 | 4640 / 5915 | -4.0% |
| marigold-depth-v1-1 | fp32 | 4297 / 7259 | 4427 / 7262 | -2.9% |
| marigold-depth-v1-1 | fp16 | 2845 / 4172 | 2875 / 4205 | -1.0% |
| defom-stereo-vitl-sceneflow | fp32 | 13773 / 6417 | 13889 / 6456 | -0.8% |
| hat-x4 | fp32 | 17384 / 6057 | 17341 / 6052 | 0 |
| raft-stereo-sceneflow | fp16 | 2794 / 4750 | 2775 / 4747 | 0 |

The survey rows confirm this. With the table on, against the rows below without it:

| Net | dtype | With table (ms) | Without table (ms) |
|---|---|---|---|
| memfof | fp32 | 3502 | 3576 |
| marigold-depth | fp32 | 4257 | 4442 |
| marigold-normals | fp32 | 2882 | 2981 |
| defom | fp32 | 13578 | 13878 |
| raft-stereo | fp32 | 5822 | 5754 (the table was slower here) |

The memfof slices also cost about 170 MB. The isolated wins mostly vanish inside a net. The warm call is dominated by
the rest of the graph (correlation, attention, GRU updates), and MLX overlaps a conv's dispatch with neighbouring
kernels, so a conv that is 3x faster alone hardly moves the end-to-end time. The table was 70 lines of shape-specific
entries, bound to one chip and one MLX version, for at most 4%. It was dropped.

Worth revisiting only:

- with a newer MLX whose conv kernel selection changes
- on another chip
- if a net turns out to be conv-bound end to end

A per-net conv-time estimate above 0.85x is a sign it will not pay. Scripts: collect.py, bench.py, analyze.py,
verify.py and probe.py in the session scratchpad.

Survey rows at the committed code, `--modes portable --force`. The "before" rows were taken at older commits, so these
deltas include the earlier passes. All goldens pass:

| ref | dtype | input | before ms / MB / first s | now ms / MB / first s |
|---|---|---|---|---|
| defom-stereo-vitl-sceneflow | fp32 | 994x1482 | 15944 / 6974 / 95.2 | 13878 / 6458 / 24.5 |
| hat-x4 | fp16 | 512x512 | 11550 / 3651 / 17.3 | 11318 / 3674 / 17.1 |
| hat-x4 | fp32 | 512x512 | 24576 / 7377 / 89.3 | 17231 / 6065 / 23.8 |
| marigold-depth-v1-1 | fp16 | 994x1482 | 2867 / 4176 / 19.3 | 2868 / 4172 / 17.3 |
| marigold-depth-v1-1 | fp32 | 994x1482 | 4441 / 7268 / 37.8 | 4442 / 7286 / 30.7 |
| marigold-normals-lcm-v0-1 | fp16 | 994x1482 | 1770 / 4220 / 19.5 | 1772 / 4212 / 14.9 |
| marigold-normals-lcm-v0-1 | fp32 | 994x1482 | 3025 / 7310 / 38.3 | 2981 / 7309 / 34.5 |
| memflow | fp32 | 1080x1920 | 13101 / 10550 / 66.8 | 10095 / 9232 / 18.9 |
| memfof | fp32 | 1080x1920 | 5436 / 2716 / 30.2 | 3576 / 3012 / 13.0 |
| raft-stereo-sceneflow | fp16 | 994x1482 | 4212 / 3926 / 15.7 | 4232 / 3955 / 15.5 |
| raft-stereo-sceneflow | fp32 | 994x1482 | 6093 / 4705 / 20.2 | 5754 / 4789 / 19.0 |
| sea-raft-m | fp16 | 1080x1920 | 1427 / 5714 / 7.6 | 1300 / 5656 / 5.1 |
| sea-raft-m | fp32 | 1080x1920 | 2187 / 9721 / 8.7 | 1853 / 9736 / 4.9 |

Three rows are slower than before, all within run-to-run noise: raft-stereo fp16 (+0.5%) and marigold-depth fp16 /
fp32 (+1 ms each). memfof's peak rose 2716 -> 3012 MB with this code and no table, so the rise comes from the earlier
passes (the 2716 row was taken at 85384184), not from this one. Regenerated routes: memflow's cheaper compile (18.9 s
first call) now routes `auto` to MLX, and raft-stereo's `device="mlx"` float32 now picks portable (5754 ms) over
kernels.

## Cold-start cut for the iterative nets: one-pass export (2026-11-21)

Goal: bring the MLX compile (`first_call_s` minus one warm call) of raft-stereo, mvsanywhere and rt-monster++ under
the routing limit of 3 MPS runtimes (`COLD_FACTOR`): 10.9 s for raft-stereo fp16 (MPS kernels 3636 ms), 3.1 s for
mvsanywhere (1034 ms), 1.84 s for rt-monster++ (612 ms).

Where the seconds went (`FF_MLX_COLD_LOG=1` plus phase timers; raft-stereo kernels fp16, 994x1482, 32 GRU iterations
unrolled into a 6378-node graph): export 9.4 s of a 12.9 s first call. Of that, 3.8 s was the `torch.export.export`
trace (the pre-dispatch training IR) and 5.6 s was `run_decompositions`, which traces that graph a second time through
AOT export. Translation took 0.3 s, the segment compiles and tuning about 0.3 s, and the first-time imports (sympy,
mpmath and so on, 1220 modules) about 0.7 s. Per-node Python stack traces were 10 to 20 % of each trace. mvsanywhere
has five exported nets of 0.3 to 0.7 s each. Its rest is `gc.collect` (5 x 70 ms), constants (0.2 s) and the
first-time imports. rt-monster++ is one 3.8 s export plus about 1 s of segment work.

What changed (`shared/mlx_graph.py`):

- `_export_decomposed`: one post-dispatch trace (`torch.export._trace._export(pre_dispatch=False)`) with
  `run_decompositions`' own table split, so the trace emits core ATen directly. Composite ops are overridden while
  tracing and the rest go to the AOT tracer's decomposition table. The program has the same ops and signature except
  for the no-op `_assert_tensor_metadata` nodes (5764 nodes instead of 6378 on raft-stereo). It is private torch API:
  any failure falls back to the old two-pass path, which is also used for modules that need the eager fill of lazy
  buffers. `FF_MLX_TWO_PASS_EXPORT=1` forces the old path.
- `torch.fx.config.do_not_emit_stack_traces` is set while exporting, because nothing reads the stack traces.
- The decomposition table and its split are built once per process (`_decomp_table`, `_split_decomp_table`) instead
  of once per export (40 ms each).

Outputs: raft-stereo kernels fp16 is bit-identical to the two-pass export (`torch.equal` on every leaf, with and
without the plan cache). Goldens pass for raft-stereo-sceneflow, mvsanywhere, rt-monster++, igev-stereo-sceneflow,
sea-raft-m-kitti, raft-small, nafnet-gopro and wd-vit-tagger-v3, along with `tests/test_mlx_plan_cache.py`.

Survey rows (ms / peak MB / first call s, so compile = first minus ms):

| ref | mode dtype | before | after | compile before -> after | limit |
|---|---|---|---|---|---|
| raft-stereo-sceneflow | kernels fp16 | 2986 / 4650 / 14.2 | 2798 / 4613 / 8.0 | 11.2 -> 5.2 s | 10.9 s |
| raft-stereo-sceneflow | portable fp16 | 4232 / 3955 / 15.5 | 4219 / 3863 / 10.5 | 11.3 -> 6.3 s | (slower than MPS) |
| raft-stereo-sceneflow | portable fp32 | 5754 / 4789 / 19.0 | 6216 / 4653 / 13.9 | 13.3 -> 7.7 s | (slower than MPS) |
| raft-stereo-sceneflow | kernels fp32 | 5818 / 5590 / 32.0 | 4662 / 5837 / 11.7 | 26.2 -> 7.0 s | (slower than MPS) |
| mvsanywhere | portable fp16 | 506 / 1425 / 7.4 | 470 / 1471 / 5.0 | 6.9 -> 4.6 s | 3.1 s |
| mvsanywhere | portable fp32 | 635 / 2119 / 7.4 | 616 / 2133 / 5.1 | 6.8 -> 4.5 s | 3.2 s |
| rt-monster++ | kernels fp16 | 622 / 2046 / 13.1 | 574 / 1979 / 9.1 | 12.4 -> 8.5 s | 1.84 s |
| rt-monster++ | portable fp16 | 713 / 2006 / 13.4 | 672 / 1910 / 9.5 | 12.7 -> 8.9 s | 1.84 s |
| rt-monster++ | portable fp32 | 1025 / 2394 / 16.1 | 950 / 2286 / 10.5 | 15.1 -> 9.5 s | 1.85 s |
| rt-monster++ | kernels fp32 | 1020 / 2503 / 14.4 | 934 / 2407 / 9.4 | 13.3 -> 8.4 s | 1.85 s |

Regenerated routes: raft-stereo-sceneflow now routes `auto` to MLX kernels fp16 (0.77x the MPS runtime, with less
memory than MPS and a compile of 5.2 s against 10.9 s). Its `device="mlx"` float32 now picks kernels (4662 ms). Neither
mvsanywhere nor rt-monster++ flips.

The survey's first call is noisy for one reason that has nothing to do with this code: the Metal shader cache. The
first process that runs a kernel set compiles its Metal sources, and later processes find them in the OS cache.
rt-monster++ portable fp16 took 12.8 s on its first probe and 5.0 s on the next. The survey worker re-run directly gave
kernels fp16 a 5.6 s first call against the 9.1 s survey row. mvsanywhere fp32's first survey run gave 9.7 s; the
re-run recorded above gave 5.1 s. The rows above are what the survey measured (one shader-cold run each for raft-stereo
and rt-monster++). The probe numbers with a warm shader cache are lower still: raft kernels fp16 4.9 s, mvsanywhere fp16
3.5 to 4.1 s, rt-monster++ 4.3 to 5.0 s.

Plan cache (`FASTFORKS_MLX_PLAN_CACHE`), measured separately with a warm shader cache:

- raft-stereo kernels fp16: compile 4.9 s with no cache, 9.5 s on a miss (`torch.export.save` of the 5.8k-node
  program) and 4.7 s on a hit. With the one-pass export, loading the program costs about as much as tracing it, so the
  cache no longer pays on this net.
- mvsanywhere fp16: 5.4 s with no cache (a noisy run; 4.1 s earlier), 5.5 s on a miss and 2.9 s on a hit. A hit is
  under its 3.1 s limit, but the routing rule measures a cold process.

What is left: rt-monster++'s export alone (3.8 s) is twice its 1.84 s budget, so only a stored program, a much cheaper
trace (exporting the GRU body once instead of unrolled) or `Model.warmup` can route it. mvsanywhere needs another
1 to 1.5 s: its five traces (2.5 s), `gc.collect` after each build (0.35 s) and the first-time imports (about 0.6 s)
are what remain.

## Cold-start cut, part 2: smaller traces (2026-11-21)

Three changes, all on the trace:

- `kth_smallest` (shared/quantile.py) sorts under an MLX export instead of running the 64 bisection passes that TensorRT
  needs. Each call was about 500 graph nodes. monster_fit calls it once per refinement iteration, so rt-monster++'s
  graph goes from 5142 to 2834 nodes, and its export from 4.3 to 3.1 s.
- Pruned exports (`mlx_wrap.port(prune=True)`, `GraphNet._export_used`). Every registered parameter becomes a lifted
  placeholder, used or not. Each costs about 0.45 ms of fakifying and functionalizing (500 unused `Linear`s add
  0.5 s to a toy export). mvsanywhere's four methods each lift all 515 tensors but read only 30 to 224 of them.
  - How it works: a fake-tensor run under a `TorchFunctionMode` (0.01 to 0.1 s after the first) records the
    parameters and buffers that are read. The unread ones become plain attributes for the export.
  - The module's `__dict__`, `_parameters` and `_buffers` are restored after the fake run, so a lazy cache cannot
    keep a fake tensor.
  - If a read was missed, export lifts that tensor as a constant on the same storage. That case is detected, and
    the method is exported again in full. A failed fake run, or one that reads more than 3/4 of the tensors, also
    gets the full export.
  - Result: mvsanywhere's exports go from 3.4 to 2.5 s.
  - Opt-in only. For a whole net it costs more than it saves: rt-monster++ reads 515 of its 757 tensors, so pruning
    would save about 0.1 s but the fake run costs 0.6 s.
- `gc.collect` after a build now runs only when a release follows, because the release hands back only tensors that
  the program no longer holds. Without a release, the twin keeps its tensors anyway. This saves 60 to 90 ms per graph,
  0.34 s on mvsanywhere's five.

Golden fix found along the way: DA3's `PositionGetter` cached the positions of the first export as a FakeTensor. The
one-pass export of the next branch then lifted that FakeTensor as a constant, and `.numpy()` failed on it. This was
already failing at b0fa8c02, `test_mlx[da3-large-1.1@1]`. The fix: the cache no longer stores traced tensors.

The same latent bug sat in every copy of the RoPE caches:

- The `PositionGetter` position cache and the `RotaryPositionEmbedding2D` frequency cache in vggt, anysplat and
  megaflow. lingbot-map uses vggt's aggregator.
- DA3's frequency cache. `shared/rope2d.rope_forward` asks it for the same key on every export.
- pi3's `RoPE2D.get_cos_sin` and `PositionGetter`.

At HEAD, `test_mlx` failed with `.numpy() is not supported for tensor subclasses` for pi3, vggt-1b, anysplat,
megaflow and lingbot-map. The cache filled while the first export traced, so the second export found a
FunctionalTensor in it. vggt-omega-1b-512 passed.

All five files now cache only a plain `torch.Tensor`. The fix was checked as follows:

- tests/test_mlx_rope_cache.py is a toy per family that exports one module at b=2, then at b=3, with the same keys.
  It failed for all five families before the fix.
- After the fix, `test_mlx` passes for pi3, megaflow, lingbot-map, vggt-1b, anysplat, vggt-omega-1b-512 and
  da3-large-1.1.
- `test_mps` (portable plus kernels) passes for pi3, megaflow, lingbot-map, vggt-1b and anysplat. da3-large-1.1 is
  not in the default MPS set, so it was checked with FASTFORKS_MPS_FULL=1.

Rows are ms / peak MB / first call s, each the second of two survey runs unless noted. Compile is the first call minus
the warm ms.

| ref | mode dtype | before | after | compile before -> after | limit |
|---|---|---|---|---|---|
| mvsanywhere | portable fp16 | 470 / 1471 / 5.0 | 467 / 1467 / 3.35 | 4.6 -> 2.89 s | 3.10 s |
| mvsanywhere | portable fp32 | 616 / 2133 / 5.1 | 619 / 2130 / 3.42 | 4.5 -> 2.80 s | 3.15 s |
| rt-monster++ | kernels fp16 | 574 / 1979 / 9.1 | 580 / 2124 / 4.42 | 8.5 -> 3.84 s | 1.84 s |
| rt-monster++ | portable fp16 | 672 / 1910 / 9.5 | 666 / 2055 / 6.62 | 8.9 -> 5.95 s | 1.84 s |
| rt-monster++ | portable fp32 | 950 / 2286 / 10.5 | 994 / 2224 / 6.25 | 9.5 -> 5.26 s | 1.85 s |
| rt-monster++ | kernels fp32 | 934 / 2407 / 9.4 | 946 / 2308 / 4.59 | 8.4 -> 3.64 s | 1.85 s |

Noise across runs:

- mvsanywhere fp16: 3.37 and 3.35 s in fp16-only surveys. It was 3.94 s when the fp32 job had run just before it in
  the same survey, a compile of 3.47 s, which is over the limit.
- mvsanywhere fp32: 3.98, 3.41 and 3.42 s.
- rt-monster++, first of the two runs:

  | mode dtype | first call | peak MB |
  |---|---|---|
  | portable fp16 | 7.43 s | 1834 |
  | portable fp32 | 8.63 s | 2205 |
  | kernels fp16 | 5.16 s | 1897 |
  | kernels fp32 | 4.68 s | 2323 |

  The peak swings by about 200 MB between runs.

Regenerated routes: mvsanywhere now routes `auto` to MLX portable fp16 (0.45x the MPS runtime). rt-monster++ does not
flip.

Where rt-monster++'s 3.8 s (kernels fp16, warm shader cache) goes:

- Steady trace: 1.83 s.
- First-time costs of the process: about 1.27 s. Imports of torch._dynamo, aot_autograd, sympy and networkx take
  about 0.65 s, and the first use of each op the rest.
- Translate: 0.28 s.
- Segment compile and tuning: 0.3 s. With a cold shader cache this is 0.8 s.

The forward unrolls four refinement iterations through three different update blocks (iters [1, 1, 2]), so there is
no single loop body to export once and reuse. The 12 ViT blocks are only 360 of the 2834 nodes. The 1.84 s budget is
smaller than the first-time costs plus a minimal trace, so only a stored program (the plan cache) or `Model.warmup`
can route it.

### Compile re-measured after the one-pass export (blocked and borderline `auto` rows)

These were the MLX rows that `auto` blocked on compile (it must be under 3x the fastest MPS runtime). All of them
predated b0fa8c02 and f7958e27.

Each job ran twice with `speed_survey.py run --force`, and the table gives the second run, with the OS Metal shader
cache warm. Cells read ms / peak MB / first call s, with the compile (first call minus warm ms) in brackets. Limit is
3x the fastest passing MPS ms, for any dtype / for the same dtype.

| ref | mode dtype | before | after (second run) | limit |
|---|---|---|---|---|
| igev-stereo-sceneflow | portable 16 | 5678 / 4566 / 22.08 (16.40) | 3951 / 3367 / 12.59 (8.63) | 12.00 / 12.00 s |
| igev-stereo-sceneflow | portable 32 | 6303 / 4404 / 26.07 (19.77) | 5702 / 4315 / 14.04 (8.34) | 12.00 / 19.12 s |
| swin2sr-x4 | portable 16 | 7994 / 3266 / 39.66 (31.66) | 4902 / 3313 / 9.65 (4.75) | 28.23 / 28.23 s |
| swin2sr-x4 | portable 32 | 13955 / 5925 / 55.22 (41.26) | 8163 / 5971 / 13.02 (4.86) | 28.23 / 28.82 s |
| igev-plusplus-sceneflow | kernels 16 | 5895 / 3184 / 28.11 (22.22) | 6039 / 2957 / 17.38 (11.34) | 18.41 / 18.41 s |
| igev-plusplus-sceneflow | kernels 32 | 10986 / 4977 / 33.46 (22.48) | 10647 / 4814 / 22.49 (11.84) | 18.41 / 29.03 s |
| deeplabv3-r101-d8-ade20k | portable 16 | 457 / 1571 / 2.92 (2.46) | 442 / 1404 / 1.73 (1.28) | 1.58 / 1.58 s |
| deeplabv3-r101-d8-ade20k | portable 32 | 654 / 2244 / 3.91 (3.25) | 668 / 1764 / 1.94 (1.27) | 1.58 / 2.16 s |
| deeplabv3-r101-d8-ade20k | kernels 16 | 452 / 1588 / 2.92 (2.46) | 441 / 1426 / 1.75 (1.30) | 1.58 / 1.58 s |
| deeplabv3-r101-d8-ade20k | kernels 32 | 645 / 2264 / 3.97 (3.33) | 675 / 1770 / 2.09 (1.41) | 1.58 / 2.16 s |
| deeplabv3plus-r101-d8-ade20k | kernels 16 | 365 / 1638 / 2.89 (2.52) | 370 / 1358 / 1.70 (1.33) | 1.29 / 1.29 s |
| deeplabv3plus-r101-d8-ade20k | kernels 32 | 511 / 2092 / 3.73 (3.22) | 505 / 1958 / 1.85 (1.34) | 1.29 / 1.55 s |
| fcn-r101-d8-ade20k | portable 16 | 358 / 1567 / 2.25 (1.89) | 359 / 1568 / 1.61 (1.25) | 1.22 / 1.22 s |
| fcn-r101-d8-ade20k | portable 32 | 496 / 2051 / 3.34 (2.84) | 491 / 2472 / 1.69 (1.20) | 1.22 / 1.40 s |
| fcn-r101-d8-ade20k | kernels 16 | 349 / 1642 / 3.23 (2.89) | 352 / 1608 / 1.63 (1.28) | 1.22 / 1.22 s |
| fcn-r101-d8-ade20k | kernels 32 | 492 / 2145 / 4.33 (3.84) | 495 / 2424 / 1.74 (1.25) | 1.22 / 1.40 s |
| pspnet-r101-d8-ade20k | portable 16 | 388 / 1582 / 2.48 (2.09) | 380 / 1441 / 1.70 (1.32) | 1.23 / 1.23 s |
| pspnet-r101-d8-ade20k | portable 32 | 517 / 2439 / 3.71 (3.20) | 510 / 2103 / 1.79 (1.28) | 1.23 / 1.46 s |
| pspnet-r101-d8-ade20k | kernels 16 | 376 / 1602 / 2.44 (2.06) | 375 / 1450 / 1.73 (1.35) | 1.23 / 1.23 s |
| pspnet-r101-d8-ade20k | kernels 32 | 521 / 2550 / 3.52 (3.00) | 511 / 2116 / 1.82 (1.31) | 1.23 / 1.46 s |
| vitmatte-b | portable 16 | 2368 / 2474 / 15.26 (12.89) | 2192 / 1946 / 5.88 (3.69) | 8.22 / 8.22 s |
| vitmatte-b | portable 32 | 2502 / 2701 / 16.45 (13.95) | 2430 / 3249 / 6.15 (3.72) | 8.22 / 14.04 s |
| vitmatte-s | portable 16 | 1280 / 2283 / 10.13 (8.85) | 1018 / 2128 / 4.61 (3.59) | 3.76 / 3.76 s |
| vitmatte-s | portable 32 | 1317 / 2451 / 14.01 (12.69) | 1204 / 3067 / 4.79 (3.59) | 3.76 / 6.58 s |
| locotrack-b | portable 32 | 2039 / 1679 / 12.78 (10.74) | 1484 / 1608 / 7.30 (5.81) | 6.31 / 6.31 s |
| prompt-depth-anything-l | portable 16 | 1082 / 2757 / 7.91 (6.83) | 1069 / 2833 / 3.19 (2.12) | 3.43 / 3.43 s |
| prompt-depth-anything-l | portable 32 | 1404 / 4681 / 8.05 (6.65) | 1391 / 4767 / 3.51 (2.12) | 3.43 / 3.83 s |
| owlv2-b16 | portable 16 | 295 / 1005 / 2.75 (2.45) | 294 / 991 / 1.56 (1.26) | no fp16 MPS row |
| owlv2-b16 | portable 32 | 398 / 1299 / 3.02 (2.63) | 390 / 1213 / 1.60 (1.21) | 1.18 / 1.18 s |

Compile fell by 35 to 85 %. igev-stereo, swin2sr, locotrack and vitmatte also got faster warm. swin2sr fp16 went from
7994 to 4902 ms.

vitmatte-b failed at f7958e27 with "Attempting to use FunctionalTensor on its own". Its `_Geometry` cache (the
position embedding and relative tables per token grid) had the same traced-tensor leak as the RoPE caches. 0957b9e7
fixes it the same way, and its rows above are from after that fix.

The other per-shape caches of the MLX-ported runners were checked with `test_mlx`:

- Two more failed with `.numpy()` at f7958e27 and are fixed the same way. They now pass on MLX and on MPS:
  - cotracker3-offline: `time_embed`.
  - mapanything-apache: `pos_embed`.
- These pass and were left alone: da3mono, da3metric, eomt-l (dinov2 and dinov3), lingbot-depth, moge-2-vits,
  patchmatchnet (it seeds its cache on real tensors) and vitmatte-s.
- pointdit's `_table` has the same pattern but was not run. Its DINOv3 weights are not on the Mac.

The other 560 or so MLX goldens have not been re-run since the one-pass export. A full `test_mlx` sweep would show
whether more caches leak.

Routes after the regeneration:

| ref | `auto` before -> after | `dtypes` after |
|---|---|---|
| igev-stereo-sceneflow | MPS -> MLX portable fp16 | fp16, fp32 portable |
| swin2sr-x4 | MPS -> MLX portable fp16 | fp16, fp32 portable |
| igev-plusplus-sceneflow | MPS -> MLX kernels fp16 | fp16 kernels |
| deeplabv3-r101-d8-ade20k | MPS -> MLX kernels fp16 | fp16 kernels, fp32 portable |
| locotrack-b | MPS -> MLX portable fp32 | fp32 portable |
| prompt-depth-anything-l | MPS -> MLX portable fp16 | fp16 portable |
| vitmatte-b | MPS -> MLX portable fp16 | fp16 portable (fp32 portable already) |
| deeplabv3plus-r101-d8-ade20k | stays MPS | fp32 kernels (new) |

igev-stereo-sceneflow also gains `kernels_default` MLX portable fp16, ahead of the MPS kernel.

These stay on MPS:

- On compile, just over the limit: fcn (1.25 vs 1.22 s), pspnet (1.32 vs 1.23 s), deeplabv3plus fp16 (1.33 vs
  1.29 s) and owlv2-b16 fp32 (1.21 vs 1.18 s). About 1.2 s is the fixed first-call cost of a process.
- On memory: vitmatte-s. MPS fp16 is at 1530 MB after the strips change, against 2128 MB on MLX.

video-depth-anything-s was not re-measured. The survey now runs it at 994x1482, but its MPS rows and its old MLX rows
are at 480x640, so the new rows were dropped.

Not candidates, because speed or memory blocks them and a shorter compile cannot change that:

- sea-raft-m, selective-igev, monster, monster++, mocha, great-igev, defom, raft-stereo portable.
- memflow and raft-stereo-sceneflow kernels fp16 are already routed.

The tiny classifiers and detectors are blocked at 5 to 100x the limit.

## Six fp16 golden misses (2026-11-21)

| ref | before (mean / tol) | after | fix | commit |
|---|---|---|---|---|
| omni-dc@1 | NaN / 0.015 | 0.00094 | RGB-D backbone as a float32 graph (`fp32_methods`); PVT features reach 3e5 | 32458853 |
| metric3d-vit-large@1 | 0.021 / 0.01 | 0.00011 | decoder as a float32 graph | a0820d13 |
| fast-acvnet-plus-sceneflow@1 | 0.135 / 0.02 | fp32 only | card `mlx_dtypes = ["float32"]`; fp16 flips a top-k cell (4x-eps noise flips it in 3 of 6 seeds) | 3b2909b5 |
| rtmpose-s-face6@1 | 0.065 / 0.061 | 0.0175 | SimCC head as a float32 graph | a116edaa |
| sea-raft-s-spring@1 | 0.0024 / 0.002 | 0.00164 | `port(half_wino_f32=True)`: fp16 3x3 convs MLX would run as Winograd run in fp32 | 45ea7cec |
| track-on2@1 | 0.046 / 0.04 | 0.0167 | `to_float32` for correlation, MSDA offsets/weights/locations, head outputs | 3d2ecc65 |

- The common cause is the float32 twin. `.float()` records nothing on it, so float32 islands vanish in an fp16
  replay. Since d7a21332 the replay also casts fp32 conv inputs to the half weights' dtype, which exposed
  omni-dc and metric3d. Use `to_float32` or `fp32_methods` for regions that must stay fp32.
- MLX 0.32's half Winograd has about 17x the direct conv's error. sea-raft's layer2 convs measured 2.8e-3 rel
  against 4e-4. The new opt-in `half_wino_f32` option (port / MlxModule / GraphNet) is off by default.
- gen_auto_mode now drops survey rows whose MLX dtype the card does not declare. fast-acvnet-generalization's
  `mlx_default` moved from fp16 to fp32.
- grounding-dino-base@1: the local npz is stale, not golden.toml. The local copies date from Nov 12 and hash to
  1cc379c1. cfe2c9b1 (Nov 18) deliberately re-pinned the ref to the published `grounding-dino-base@1-6054688b.npz`.
  `tools/fetch_goldens.py` fetches it.

## Re-survey after the fp16 islands (2026-11-21)

Portable MLX on the M1 Pro, second run of two. Cells are ms / peak MB / first call s (compile s); every golden passes.

| ref | dtype | before | after | MPS |
|---|---|---|---|---|
| rtmpose-t | fp16 | 26 / 665 / 1.74 (1.72) | 25 / 709 / 1.14 (1.11) | 26 / 1460; kernels fp32 20 / 1532 |
| rtmpose-t | fp32 | 30 / 742 / 1.75 | 30 / 785 / 1.07 (1.04) | 26 / 1462 |
| sea-raft-m | fp16 | 1300 / 5656 / 5.13 (3.83) | 1387 / 5600 / 3.53 (2.14) | 1635 / 9732; kernels 1220 / 4752 |
| sea-raft-m | fp32 | 1853 / 9736 / 4.91 | 2003 / 9715 / 3.82 (1.82) | 1820 / 12542 |
| fast-acvnet-generalization | fp32 | 761 / 1776 / 8.64 (7.88) | 487 / 1877 / 2.59 (2.10) | 557 / 5022; fp16 410 / 3819 |
| metric3d-vit-large | fp16 | 2460 / 2769 / 16.81 (14.35) | 2460 / 2922 / 5.83 (3.37) | 1861 / 3636 |
| metric3d-vit-large | fp32 | 2273 / 3902 / 23.03 | 2450 / 3527 / 5.95 (3.50) | 1822 / 4523 |
| omni-dc | fp16 | 5978 / 5375 / 28.02 | 3873 / 5387 / 11.94 (8.07) | 2408 / 10081 |
| omni-dc | fp32 | 3475 / 6632 / 23.95 | 4033 / 6041 / 11.80 (7.77) | 2349 / 12317 |
| track-on2 | fp16 | 7623 / 3056 / 18.83 (11.21) | 7718 / 3207 / 12.47 (4.75) | fp32 8354 / 6102 |
| track-on2 | fp32 | 10755 / 3318 | 9798 / 3298 / 13.95 (4.15) | 8354 / 6102 |

The islands were narrowed where they cost time. These are fp16 timings:

- sea-raft: the float32 correlation (`to_float32(fmaps)`) cost about 4 GB and 400 ms at 1080p (1835 / 9670). It is
  back to `.float()`, so it runs in fp16 on MLX. Spring stays at 0.00169 against 0.00164 (tol 0.002).
- sea-raft: `half_wino_f32` now takes `(cin, cout)` pairs, and sea-raft uses `((128, 128),)`. Those are the encoders'
  layer2 convs, the only Winograd convs at the golden's size. All Winograd convs in fp32 measured 1492 ms; the pairs
  measured 1426 ms. With no fp32 Winograd at all it runs at 1323 ms, but s-spring fails (0.0024).
- track-on2: MSDA offsets, weights and locations plus the head outputs are back to `.float()`. They had cost
  7714 -> 8479 ms, about 9%, and the MSDA part is nearly all of it (head only: 8308). Goldens are track-on2 0.0010
  and track-on-r 0.0059 (tol 0.04 / 0.05). The correlation island must stay: with it in fp16, track-on2's p99 is
  0.76.
- omni-dc: the fp32 `_backbone` stays. An fp16 backbone is slower (24.4 s, because of overflow) and fails.
  metric3d's and rtmpose's islands cost nothing. fast-acvnet got faster.

Route changes:

- track-on2@1 lost `auto` in 59f98089 at 8479 ms and has it back at 7718 ms (MPS 8354 ms).
- omni-dc@1's `mlx_default` moved from fp32 to fp16.
- Nothing else changed.

## Compile time: rt-monster++, mvsanywhere and the fixed per-process cost (2026-11-21)

The targets were rt-monster++ (compile 3.6-6 s against a limit of 1.84 s), mvsanywhere fp16 (its `auto` flip is
fragile), and the fixed ~1.2 s per process that keeps fcn, pspnet, deeplabv3plus fp16 and owlv2 just over their limits.

How the survey measures it: `first_call_s` is the first call after `task.load()`. MLX's load is only 0.14-0.2 s
because the export and translation run lazily inside that first call. `gen_auto_mode` takes compile as
`first_call_s - warm`. All numbers below are compile times on the M1 Pro, from the second process (the first process
after a Metal shader-cache flush is much slower: rt-monster++ took 12 s instead of 3.8 s).

Where the fixed cost goes (`python -X importtime`, first call):

- `torch.export._trace` imports take 0.555 s:
  - torch._dynamo 0.40
  - aot_autograd 0.15
  - partitioners/networkx 0.11
  - symbolic_shapes/sympy 0.13
  - torch.distributed fsdp/tensor about 0.1
- Importing these at `load()` or at package import would only move the cost out of the measured call. That games the
  metric, so it was not done. The cost goes away only if the export is skipped.

Plan cache (`FASTFORKS_MLX_PLAN_CACHE`, opt-in), new storage format:

- The old entry was a weightless `ExportedProgram`. `torch.export.load` took most of a hit: rt-monster++ was 5.1 s on
  a miss and 2.85 s on a hit.
- An entry is now `<key>.graph`, which pickles only what `GraphNet._build_graph` reads:
  - each node's op, name, `OpOverload` (namespace, name, overload), args/kwargs with node references, and output
    `meta["val"]` shape/dtype
  - the input specs
  - the constants the forward builds
- On a hit the graph is rebuilt on the `meta` device and bound to the module's own tensors by name, shape and dtype.
  Hits skip the `torch.export` import entirely.

| model / mode | no cache | miss | hit | limit |
|---|---|---|---|---|
| rt-monster++ kernels fp16 | 3.8 s | 3.99 s | 0.75 s | 1.84 s |
| mvsanywhere portable fp16 | 2.89 s | 3.11 s | 0.42 s | 3.10 s |
| fcn-r101-d8-ade20k portable fp16 | 1.19 s | 1.30 s | 0.26 s | 1.22 s |

- Outputs on a hit are bit-identical to no cache for all three (rt-monster++ disparity, mvsanywhere's 7 leaves, fcn).
- An entry is 13-490 KB (rt-monster++ 490 KB).
- Goldens pass on a miss and then on a hit: rt-monster-plusplus, mvsanywhere, fcn-r101, nafnet-gopro and
  wd-vit-tagger-v3. rt-monster++ also passes without the cache.
- `_versions()` (the source-file fingerprint in the key) costs about 0.145 s per process, and only with the cache on.

Not done, as a note for a decision: making the cache on by default. That would turn the survey's second run into a
hit and flip rt-monster++, and probably fcn/pspnet/deeplabv3plus, to `auto`. A user's first call at a new input size
would still pay the miss (no-cache time + 0.1-0.2 s). So a hit-based first call reflects a returning user at a known
size, not a new size. Keyed by shape, that is every new resolution. I left the cache opt-in and the survey uncached,
so the routes are unchanged. If default-on is wanted, these are also needed:

- a size cap or pruning, since the cache is never pruned
- test isolation, since conftest only isolates `FASTFORKS_CACHE`
- changes to `test_off_by_default` and docs/usage.md
- a survey column for the miss as well as the hit

No re-survey and no route change came out of this work. The survey path (cache off) is unchanged code.

## Multiview transformers: fp16 replays that ran in fp32 (2026-11-21)

`FF_MLX_PROFILE=1` showed the same cause for all six refs that lost to MPS: the "float16" replay ran its transformer in
float32. The twin is exported in float32, so `.to(dtype)` on a table is a no-op there. Any float32 table the forward
builds (RoPE cos / sin, a bicubic-resized position table, a KV buffer from `new_empty`) then promotes every op
downstream of it under MLX type promotion. That covered most addmm, SDPA and LayerNorm calls in the aggregator. fp16
GEMM and SDPA on MLX are on par with or faster than MPS (8456x1024x4096 GEMM 15.7 vs 16.3 ms, SDPA L8456 74.6 vs 88.6 ms).

The fix is `to_run_dtype` at each table, a no-op off MLX:

- vggt-omega: q / k after the float32 RoPE. With only that change, the fp16 golden was nan: an fc2 output of 46k times
  the LayerScale gamma overflowed. So on MLX the block keeps the residual stream and LayerScale in float32, with the
  branches in fp16 (`_mlx_forward`). This matches upstream bf16 autocast and the MPS fast path.
- `shared/rope2d.rope_forward` (vggt-1b, anysplat, da3) and pi3's `_rope_forward`: the output goes back to the run
  dtype, and the rotation stays in float32. Casting the tables to fp16 instead was tried first. AnySplat then missed
  its `gaussians.means` bound (mean 0.0038, limit 0.0031), and still missed it (p99 0.0124, limit 0.012) with a float32
  residual stream.
- AnySplat's `Block` keeps a float32 residual stream on MLX. Without it the golden misses (mean 0.0038).
- mapanything: `pos_embed`, the bicubic-resized float32 table.
- lingbot-map runs the VGGT fast path on MLX, not `rope_forward`. Its fix is in `vggt._Fast.rope_tables`, the 3D
  `rope3d` tables and the `_Cache` KV buffers.

Survey rows (warm median ms / peak MB; compile = first call - warm):

| ref | dtype | MLX before | MLX after | compile | MPS (fastest) | `auto` |
|---|---|---|---|---|---|---|
| vggt-omega-1b-512 | fp16 | 8564 / 5137 | 5853 / 4289 | 20.0 s | 7242 / 8602 | now MLX fp16 |
| vggt-omega-1b-512 | fp32 | 16674 / 7431 (outlier row) | 7746 / 7388 | | 7938 / 8221 | |
| vggt-1b | fp16 | 8774 / 6853 | 5711 / 5689 | 24.8 s (limit 19.9) | 6638 / 8423 | blocked by compile |
| vggt-1b | fp32 | 9359 / 9387 | 7826 / 8964 | | 7016 (kernels) | |
| pi3 | fp16 | 10309 / 3859 | 7752 / 3102 | 8.8 s | 8341 / 6510 | now MLX fp16 |
| pi3 | fp32 | 11851 / 5526 | 10305 / 5246 | | 10298 / 6201 | |
| anysplat | fp16 | 6974 / 8297 | 4824 / 6370 | 9.3 s | 5164 / 8809 | now MLX fp16 |
| anysplat | fp32 | 8786 / 10565 | 6109 / 9821 | | 5566 / 8719 | |
| lingbot-map (480x640) | fp16 | 10046 / 8220 | 7217 / 5409 | 22.7 s (limit 21.4) | 7125 / 8834 | 1.3 % slower; compile over |
| lingbot-map (480x640) | fp32 | 11010 / 9090 | 9852 / 9304 | | 8840 / 13055 | |
| mapanything-apache | fp16 | 8876 / 8371 | 5894 / 5495 | 26.1 s (limit 19.1) | 6369 / 7218 | blocked by compile |
| mapanything-apache | fp32 | 11871 / 11118 | 9075 / 10774 | | 7357 / 11638 | |

All goldens pass: `test_mlx` for every touched family, da3 (5 refs, which share `rope_forward`), nafnet-gopro and
wd-vit-tagger-v3, and tests/test_mlx*.py. The fp32 gains are run-to-run differences, not code: fp32 replays were
already float32. Routes only gain entries; none are lost.

Still open:

- Compile for vggt-1b, mapanything-apache and lingbot-map: the survey's no-cache first call is 30-32 s. The fp16 speed
  is already below MPS.
- MLX fp32 is still slower than MPS fp32 for vggt-1b, anysplat, lingbot-map and mapanything. fp32 GEMM on MLX is about
  10 % behind MPS, and fp16 is the default route.
- DA3 was not re-surveyed. Its fp16 rows probably improve too, since the same RoPE output cast applies.

## fp16 replays that ran in fp32: detector and sweep (2026-11-21)

`FF_MLX_DTYPE_REPORT=1` (`shared/mlx_dtype_report.py`; `tools/mlx_dtype_report.py <refs>` runs one quick forward
per ref) prints, per replay signature:

- the float32 share of the heavy-op FLOPs (mm / addmm / bmm / conv / SDPA / einsum);
- the first float32 heavy op;
- the modules that run in float32;
- the float32 origins, which can be a folded constant, an input, or a deliberate `to_float32` island.

The cause is the same as in the multiview section above. `.to(dtype)`, `new_zeros`, `torch.empty(dtype=x.dtype)` and
`.float()` are no-ops in the float32 twin, so one float32 table lifts everything downstream. Each table now goes
through `to_run_dtype` at its source.

Also fixed: in `mlx_graph`, `aten.where` with a 0-dim float32 `scalar_tensor` takes the other operand's dtype, as in
torch. MLX would promote to float32 (oneformer 95.9 -> 4.9 % with the Swin pad fix).

| ref | heavy fp32 % before | after | fp16 ms / MB before | after | `auto` |
|---|---|---|---|---|---|
| da3-large-1.1 | 74.2 | 6.9 | 361 / 1975 | 268 / 1763 | |
| da3-giant-1.1 | | | 920 / 3858 | 647 / 3454 | |
| da3nested-giant-large-1.1 | | | 1269 / 4267 | 878 / 4123 | |
| da3metric-large / da3mono-large | | | 328 / 1899, 322 / 1884 | 231 / 1555, 231 / 1557 | |
| metric3d-vit-large | 99.9 | 0.0 | 2460 / 2922 | 2337 / 2943 | |
| unik3d-vitl | 83.2 | 8.3 | 1458 / 2388 | 1032 / 2039 | now MLX fp16 |
| sam-vit-b | 98.4 | 0.0 | 666 / 2571 | 446 / 1205 | |
| sam-hq-vit-b | | | 703 / 2588 | 476 / 1510 | |
| waft-a1 | 36.9 | 0.0 | 9339 / 3559 | 6431 / 2697 | now MLX fp16 |
| megaflow | | | (no MLX fp16 row) | 8480 / 6561 | now MLX fp16 (MPS 11196) |
| bridgedepth-sf | 12.2 | 0.3 | 5661 / 4483 | 3977 / 2650 | |
| nmrf-sceneflow | 58.3 | 0.0 | 2246 / 3106 | 1438 / 2241 | |
| restormer-deblur | 29.8 | 5.4 | 2532 / 2666 | 1467 / 1317 | |
| oneformer-swinl-ade20k | 99.0 | 4.9 | 2078 / 6033 | 2077 / 5376 | |
| mask2former-swinl-ade20k | | 2.1 | 1148 / 3795 | 878 / 3657 | |
| vitmatte-b | 81.3 | 0.0 | 2192 / 1946 | 1744 / 3013 | |
| mvsplat-re10k | 26.3 | 0.0 | 2418 / 2796 | 2116 / 2706 | |
| gmflow-scale1-things | 69.7 | 0.1 | 6822 / 10465 | 3871 / 9100 | (MPS fp32 3269) |
| big-lama | 70.9 | 7.9 | 321 / 1094 | 225 / 990 | |

All MLX fp16 goldens pass. Also run: tests/test_mlx*.py, nafnet-gopro and wd-vit-tagger-v3. The fp32 rows changed
only by run-to-run noise. Routes only gained entries: da3 x5, bridgedepth, mask2former and metric3d now default to
MLX fp16. vitmatte-b's peak rose by 1 GB while it got 20 % faster; this is not investigated.

Deliberate float32 islands kept, and negative results:

- **A generic cast of every folded constant to the run dtype** was rejected. It risks `-inf` / overflow masks and
  loses precision in coordinate tables. The casts stay per source.
- **gmflow**: a fully half Transformer missed the golden (mean 0.0036, limit 0.0031). Casting the position table, or
  only the sum after the position add, did not help. What passes is the CUDA / MPS autocast layout: a float32
  residual stream, with the projections, the attention and the MLP in half (`TransformerLayer.forward`, export-only).
  Matching, propagation and the refinement correlation stay float32 through `to_float32`.
- **restormer / DAT gram matrices** (channel attention) and **the big-lama FFT** (`convg2g.fu`) stay float32. They are
  the remaining 5.4 % and 7.9 %.
- **da3 / unik3d**: the remaining 7-8 % is the RoPE rotation, which deliberately stays float32 in
  `shared/rope2d`.
- **unidepth-v2**: its card is MLX float32 only, so the DINOv2 edit was reverted.

### Native MPS: no silent promotion (negative result)

`tools/mlx_dtype_report.py <ref> --mps` counted the heavy-op FLOPs by input dtype on an eager MPS fp16 forward. In
eager torch, `.to(dtype)` and `new_zeros` keep their real dtype, so the twin's failure mode does not exist there:

| ref | MPS fp16 heavy fp32 % | note |
|---|---|---|
| gmflow-scale1-things | 0.5 | MPS fp16 10676 ms vs fp32 3269 ms is not promotion: memory pressure (see "MPS pass" below) |
| moge-3-vitl | 6.4 | sparse `subm_conv` addmm and a few linears |
| video-depth-anything-metric-s | 0.0 | |
| mvsanywhere | 0.0 | |
| raft-large | n/a | no heavy op was seen by the dispatch hook |

No MPS code was changed. The fp16 >= fp32 MPS rows (sam3, swin2sr, moge, raft-large, gmflow) need an op-level
profile instead.

### MLX findings that plausibly help MPS (not implemented here)

- Re-run `tools/mlx_dtype_report.py <ref> --mps` after MPS code changes. The check above found no promotion today.
- Cost volumes from `new_zeros` filled in a loop (nmrf, mocha, cfnet, psmnet, cas_gwcnet): one `torch.stack` /
  `cat` avoids the in-place slice writes and the extra zero fill.
- Sibling projection fusion (q / k / v, or the gmflow q_proj + k_proj(query) pair) as one GEMM, as the MLX graph pass
  does.
- Release intermediates early (`del` before the next stage; `fence` where a new heap opens). The MLX buffer cache
  shows that most peak gaps are lifetimes, not sizes.
- Keep a float32 residual stream with half branches only where the golden needs it (vggt-omega, anysplat, gmflow).
  Everywhere else a fully half stream halves the activation bytes.

## Multiview first call: the weights' disk read (2026-11-22)

vggt-1b, mapanything-apache and lingbot-map missed `auto` on compile alone (the survey's first call was 30-32 s at
fp16). Timers on each `GraphNet` phase (vggt-1b fp16, 8 views, 26 s first call):

| phase | s | note |
|---|---|---|
| `_export_decomposed` | 7.4 | fake-tensor tracing and pytree work, spread thin (no single hot spot) |
| `to_mx`, 1405 tensors | 11.4 (24 from a cold page cache) | the disk read, not the conversion: warm it is under 1 s |
| `mx.compile`, tuning | 0.4 | |
| first replay | 6.1 | about one warm call |

The lazy twin's parameters are the mmapped checkpoint (`torch.load(mmap=True)`, the safetensors mmap). `load_s` is
0.3-1.3 s because nothing is read at load. The first call's translation then faults 5 GB in page by page on one thread,
at about 0.2 GB/s. The 16 GB Mac does not keep the file in its page cache between runs. MPS reads the same bytes at
load instead: its `load_s` is 8-17 s, which the cold rule does not count.

Fix (`mlx_graph._prefetch`): when the first input signature is built, 8 threads read one byte per page of every CPU
weight while the export traces. They are joined before the translation, because they hold the storages and
`_exclusive` must not see them. After the export, the lifted constants that are not yet converted are touched the
same way. mapanything keeps its ViT blocks as plain tensor attributes (`_Blk`), so only the export sees them. The
touch covers each tensor's own byte range, cut into 64 MB pieces: a safetensors checkpoint is one storage for the
whole file, and a split by storage gave mapanything a single thread. `FF_MLX_NO_PREFETCH=1` turns it off. Nothing
moved out of the measured call: the export, the reads and the translation all stay in it.

Survey, second run (warm ms / first call s / peak MB):

| ref | dtype | before | after | compile (limit) | `auto` |
|---|---|---|---|---|---|
| vggt-1b | fp16 | 5711 / 30.5 / 5689 | 5725 / 15.6 / 5755 | 9.9 s (19.9) | now MLX fp16 |
| vggt-1b | fp32 | 7826 / 60.3 / 8964 | 8081 / 18.4 / 8829 | | |
| mapanything-apache | fp16 | 5894 / 32.0 / 5495 | 5993 / 18.2 / 5514 | 12.2 s (19.1) | now MLX fp16 |
| mapanything-apache | fp32 | 9075 / 44.8 / 10774 | 9055 / 19.4 / 10756 | | |
| lingbot-map (480x640) | fp16 | 7217 / 29.9 / 5409 | 7261 / 14.0 / 5438 | 6.8 s (21.4) | still MPS: 1.9 % slower warm |
| lingbot-map (480x640) | fp32 | 9852 / 43.0 / 9304 | 9970 / 18.7 / 8930 | | |

A control run of lingbot-map in the same session with `FF_MLX_NO_PREFETCH=1` gave 25.8 s (fp16) and 40.3 s (fp32).
How cold the page cache is varies from run to run, and the first runs after another model ran 1-4 s slower. One
lingbot-map run measured 8023 ms warm, but a third run gave 7227 ms (the control gave 7230), so the warm speed did not
change. Goldens pass: `test_mlx` for the three refs, nafnet-gopro and wd-vit-tagger-v3, and tests/test_mlx*.py.

Negative results:

- `MADV_WILLNEED` on the storages was as fast (5 GB in 5-9 s from 4 threads), but macOS counts read-ahead pages in
  the process's physical footprint. vggt-1b's survey peak went from 5.7 to 9.9 GB (fp32 13.3 GB). Faulted pages are
  not counted.
- One-thread sequential touching is no faster than the translation's own faults (23.5 s for vggt-1b).
- Cold-cache benchmarks need care: `msync(MS_KILLPAGES)` on a shared mapping evicted the file only some of the time,
  and `purge` needs root. The survey's own second run is the measurement here.
- The 7.4 s export (vggt-1b; mapanything 3.6 s) is the largest cost left. The profile shows no hot spot to cut.
  Only the opt-in plan cache (`FASTFORKS_MLX_PLAN_CACHE`) skips it.

For MPS (not implemented): the same threaded page touch before the `.to("mps")` copy would cut the 8-17 s `load_s`
of these checkpoints, since MPS reads every weight at load from the same mmapped files. The cold rule compares only
the MLX compile with MPS warm ms. If `load_s` counted, MLX (0.3-1.4 s) would win the cold start for all three refs.

## MPS pass: load time, gmflow, fp16 >= fp32 rows (2026-11-22)

All numbers: M1 Pro 16 GB, portable mode on MPS, survey second run (warm ms / peak MB) unless noted.

### Load time of the big multiview checkpoints

`load_state_dict(..., assign=True)` on MPS (vggt-1b, lingbot-map `_net`): the mmapped float32 checkpoint moves to the
GPU as it is, without the full CPU copy that `load_state_dict` makes first (about 2.8 s for vggt's 4.8 GB).

| ref | dtype | `load_s` before / after | warm ms before / after | peak MB before / after |
|---|---|---|---|---|
| vggt-1b | fp16 | 17.3 / 12.2 | 6638 / 6193 | 8423 / 8700 |
| vggt-1b | fp32 | 12.5 / 11.1 | 7209 / 7223 | 10654 / 11354 |
| lingbot-map (480x640) | fp16 | 14.1 / 8.5 | 7125 / 7124 | 8834 / 7956 |
| lingbot-map (480x640) | fp32 | 14.0 / (11.6, not recorded) | | |

vggt-1b's peak rose by 0.3-0.7 GB, likely because the assigned parameters keep the mapped pages alive until `.to("mps")` replaces
them. lingbot-map's fell by 0.9 GB (it halves on the CPU before the move, so the copy was its peak).
The lingbot-map fp32 re-survey hit the MPS memory ceiling in its golden (11.45 GiB allocated, swap 4-5 GB used) and
ran 11.0 s warm against 8.8 s before, a memory-pressure run. Its old row is kept. Goldens pass for both refs.

Negative results:

- The threaded page touch of `mlx_graph._prefetch` before `.to("mps")` gives nothing. The weights on /Volumes/Daten
  read at 0.3-0.5 GB/s with 1 or 8 threads, so the load is disk-bound: joined first, the touch alone took 10.7 s for
  4.8 GB (vggt fp16 `load_s` 20.9 / 17.9 s against 22.4 / 17.2 s without); overlapped with the net build it was
  16.9 / 17.6 s (lingbot 12.8 / 15.2 against 15.3 / 15.4 s). Not kept. mapanything-apache (`load_s` 7.6-9.0 s, 3.3 GB)
  showed no change either.
- During the MPS load the faulted pages of the mapped checkpoint add about 3 GB to the footprint (vggt-1b load peak
  8.4-8.5 GB for a 4.8 GB checkpoint that ends up as 2.4 GB of fp16 GPU weights). They are dropped once the state dict
  is released. Moving the weights in 256 MB chunks with a sync after each did not lower that peak.

### gmflow: MPS attention writes the whole score matrix

The fp16 slowdown was memory, not compute: at 544x960 fp16 already won (283 ms against 336 ms fp32). At 1080p the old
fp16 row had a 12.6 GB peak and `ms_min` 4170 ms against a 10676 ms median (pageouts). With the MPS watermark at 1.0,
both dtypes ran out of memory in the global matching / propagation: MPS SDPA allocates the full `[B, L, M]` scores,
4 GB for L = M = 32640. The shifted-window mask (`[4, 8160, 8160]`, 1 GB fp32) was also tiled over the batch in fp32
and then cast, because the CUDA-only branch did not cover MPS.

Fix (`gmflow/net/attention.py`): on MPS, `sdpa` chunks the queries to at most 2^27 scores per call, and the shift mask
is cast once and shared across the batch as on CUDA. Goldens pass, also with the chunk limit forced down to 2^20 (the
golden size does not reach the 2^27 limit).

| ref | dtype | before | after |
|---|---|---|---|
| gmflow-scale1-things | fp16 | 10676 / 12643 | 2756 / 7112 |
| gmflow-scale1-things | fp32 | 3269 / 11902 | 3780 / 8183 |
| gmflow-scale1-mixdata | fp16 | 9282 / 12270 | 2904 / 7113 |
| gmflow-scale1-mixdata | fp32 | 3269 / 12049 | 3845 / 8160 |

fp32 is 16 % slower: a chunk limit of 2^28 gave the same speed (3659 ms) for 2.5 GB more peak, 2^25 was slower
(4.0 s), and 2^29 ran out of memory at watermark 1.0. MPS fp16 now beats MLX fp16 (3871 ms), and `auto` takes MPS for
gmflow-scale1-things fp16 again. The scale2-regrefine6 variants, out of memory at 1080p before, now run (22-27 s) but at
13.7-14.2 GB peak. Those timings are pageout-bound, so the old `oom` rows are kept. Their memory is in the 1/4-scale
stage and the refinement, not in this attention.

### Other fp16 >= fp32 rows

- **moge (all four variants)**: on MPS the fp16 head convs ran channels-last and the replicate-padded 3x3 convs used
  `_rep_conv` (two `torch.cat` per conv, written for CUDA's channels-last copy). Each cost about 2x there. MPS now keeps
  the plain layout and the native replicate padding. Goldens pass.

  | ref (994x1482) | fp16 before | fp16 after | fp32 after |
  |---|---|---|---|
  | moge-2-vits-normal | 869 | 303 | 362 |
  | moge-2-vitb-normal | 1108 | 494 | 585 |
  | moge-2-vitl-normal | 1790 | 1176 | 1350 |
  | moge-3-vitl | 3913 | 3333 | 3692 |

- **neuflow-v2 (three variants), no code change**: the 2026-11-12/14 rows (fp16 2.8-4.6 s against fp32 0.95-1.08 s,
  12-13 GB peaks) were memory-pressure runs. A re-survey of the same code gives fp16 0.70-0.74 s and fp32 0.74-0.76 s.
  The fp16 `_lookup_half` gather is not slow on MPS (723 ms against 792 ms fp32 in an isolated run).
- **starnet_s1** (14 ms fp16 against 8 ms fp32 at 224x224): dispatch-bound at this size, and s2-s4 are equal or
  faster in fp16. Not touched.
- **c-fast-foundationstereo** and the 1.2-1.3x rows (tapir, aanet, las2-h, video-depth-anything-metric-s): see the
  next section.

### fp16 >= fp32, part 2: autocast on MPS (2026-11-22)

Common cause: most of these runners take fp16 as `torch.autocast(device.type)` around plain PyTorch, which on MPS
either did not run fp16 at all or cost more in casts than it saved. Autocast's norms return fp32, so the activations
between the convs stay fp32 and every conv / matmul casts its input again. Where it is safe, the fix is the whole
network in fp16 on MPS (`pure_half`, NCHW). CUDA paths are unchanged everywhere. MPS goldens pass for every touched
family (`FASTFORKS_MPS_FULL=1`, both dtypes).

Detector for the next ones: an MPS fp16 row with the same peak as fp32 is not running fp16 (c-fast-foundationstereo,
aanet). A lower peak but a slower time points to casts or a layout (las2, tapir, video-depth-anything).

Survey rows, second of two runs (ms warm median / peak footprint MB):

| ref | input | fp16 before | fp16 after | fp32 |
|---|---|---|---|---|
| c-fast-foundationstereo | 994x1482 | 18200 / 12830 | 2714 / 8484 | 4881 / 12882 (row of 2026-11-06) |
| fast-foundationstereo | 994x1482 | 3047 / 13515 | 2715 / 8478 | 3209 / 13359 (row of 2026-11-21) |
| las2-h | 994x1482 | 821 / 4422 | 588 / 3674 | 664 / 5108 |
| las2-l | 994x1482 | 401 / 3496 | 426 / 2818 (run 1: 386) | 500 / 4238 |
| las2-m | 994x1482 | 237 / 2634 | 232 / 2684 | 241 / 3041 |
| las2-s | 994x1482 | 193 / 2587 | 167 / 2564 | 182 / 2994 |
| aanet-sceneflow | 994x1482 | 1163 / 3632 | 723 / 2594 | 880 / 3622 |
| aanet-plus-sceneflow | 994x1482 | 1285 / 5031 | 936 / 3836 | 992 / 5109 |
| tapir | 480x854 | 1809 / 1611 | 1393 / 1610 | 1356 / 1604 |
| bootstapir | 480x854 | 2059 / 1750 | 1719 / 1845 | 1704 / 1743 |
| video-depth-anything-metric-s | 480x640 (probe) | 897 | 700 | 748 |
| video-depth-anything-metric-s | 994x1482 (new rows) | - | 780 / 4581 | 836 / 6522 |
| video-depth-anything-s | 994x1482 (new rows) | - | 779 / 4578 | 839 / 6523 |

- **c-fast-foundationstereo / fast-foundationstereo**: TAO's AMP blocks are `autocast('cuda', ...)`, so MPS "fp16" was
  the fp32 network, with pageouts on top of a 12.8 GB peak (min 5.7 s, median 18.2 s). Now the whole network is fp16
  on MPS (`c_foundationstereo.portable.mps_half`). The GRU lookups keep TAO's `.float()` and the update block and
  upsampling heads cast back to fp16. The fp32 rows of this survey (9.9 s and 6.2 s median against 4.8 s and 3.7 s
  min, 8 GB swap) were pressure runs and were not kept; the older fp32 rows stay.
  - Negative: making `AUTOCAST` follow the device (MPS autocast) gave nothing: 3.7-3.9 s, 12.6 GB, BatchNorm outputs
    fp32, one flaky MPS OOM. Reverted.
  - Fixed (2026-11-22): c-foundationstereo-s fp16 at 994x1482 no longer OOMs on MPS. See "C-FoundationStereo on MPS
    and MLX: strips, bmm attention, classifier taps" below.
- **las2**: MPS autocast with channels-last H ran 926 ms against 702 ms fp32 (probe). NCHW alone gave 761 ms; pure fp16
  gave 636 ms at 3.7 GB instead of 5.1 GB. All four sizes use the same path on MPS. las2-l moved between 386 and
  426 ms across the two runs (400 before), so it is about flat; its peak dropped 0.7 GB. With the faster portable row,
  the regenerated routes drop las2-h's `mps_auto` fp16 entry (the Metal kernels no longer beat portable by `MIN_GAIN`).
- **aanet**: autocast ran at the fp32 peak. Pure fp16 NCHW gave AANet 736 ms. AANet+ stayed slower (1095 ms against
  990 ms) until the shared deformable conv was fixed. Its full-resolution hourglass `conv_start` took 270 ms in fp16
  against 158 ms in fp32. The cause is `torch.mul(cols_fp32, mask_fp16_strided_view, out=fp16)`, which runs 4x slower
  on MPS than the fp32 product (130 against 32 ms per call). `shared.deform_conv` now multiplies by an fp32 copy of the
  mask slice and casts after that, on MPS only. The result is bit-identical. ALIKED shares the op; its goldens pass.
- **tapir / bootstapir**: the PIPs correlations (`einsum("bnfsc,bnfc->bnfs")`) take fp32 neighbourhoods. These are
  grid_sample outputs with channels as the slowest stride. Autocast cast them to fp16 on every call: 240 ms against
  59 ms in fp32. On MPS they now run in fp32 outside autocast, which brings fp16 to parity with fp32.
  - Negative: fp16 gives nothing else on MPS here. The ResNet took 159 ms in fp16 against 145 ms, and the mixer 881 ms
    against 858 ms.
  - Not done: a pure-fp16 TAPIR would put the sampling coordinates in fp16 and meet the mixer's explicit `.float()`.
- **video-depth-anything**: already `pure_half`, but channels-last. NCHW on MPS: the 32-frame head window took 630 ms
  against 827 ms. The survey now takes 994x1482 from the reference floor (new rows). The 480x640 rows of 2026-11-12/17
  stay as they were (the survey cannot re-measure them).
- **Iterative stereo / flow (sea-raft, selective-igev, monster, mocha, great-igev, igev, raft-stereo), scan only**: on
  MPS each fp16 portable row is already faster than fp32 (0.71-0.91x) at a lower peak, so autocast does run fp16
  there. raft-stereo already keeps NCHW on MPS, stores fp16 conv weights and avoids the fp32 promotion of the GRU input.
  Pure fp16 would also put the flow / disparity state and the lookup coordinates in fp16, so it was not tried.
  Suspect rows to re-survey alone: the fp32 rows of sea-raft-m-kitti (median 5026 / min 4287), sea-raft-m-things
  (4162 / 2641), sea-raft-s (2115 / 1849) and monster-plusplus-mix-all (16071 / 13884), all at 11.5-12.7 GB. Their
  0.3-0.5x fp16 ratios are most likely fp32 pressure, not fp16 gains. defom-stereo has no MPS fp16 row.


### fp16 >= fp32, part 3, and clean re-surveys (2026-11-22)

Ms are warm medians (second of two survey runs), MB the process-footprint peak. MPS goldens pass in both dtypes
(`FASTFORKS_MPS_FULL=1`); MLX goldens of the touched families pass too.

| ref | dtype | before ms / MB | after ms / MB |
|---|---|---|---|
| restormer-sidd | fp16 | 2330 / 3683 | 1132 / 3675 |
| restormer-sidd | fp32 | 2218 / 4761 | 1112 / 5777 |
| restormer-derain | fp16 | 1814 / 3655 | 988 / 3653 |
| restormer-derain | fp32 | 1678 / 5778 | 995 / 4728 |
| restormer-deblur | fp16 | 1476 / 3657 | 990 / 3651 |
| restormer-deblur | fp32 | 1376 / 4735 | 989 / 4726 |
| swin2sr-x2 | fp16 | 9258 / 3501 | 4607 / 3495 |
| swin2sr-x2 | fp32 | 11075 / 6215 | 5731 / 6213 |
| swin2sr-x4 | fp16 | 9410 / 4580 | 4690 / 4581 |
| swin2sr-x4 | fp32 | 9608 / 8391 | 5793 / 8393 |
| swin2sr-x4-real | fp16 | 9531 / 4579 | 4751 / 4576 |
| swin2sr-x4-real | fp32 | 11350 / 8389 | 5894 / 8390 |
| swinir-m-x2 | fp16 | 5797 / 3500 | 3674 / 3492 |
| swinir-m-x2 | fp32 | 6260 / 5198 | 4760 / 5196 |
| swinir-m-x4 | fp16 | 5893 / 5060 | 3750 / 5064 |
| swinir-m-x4 | fp32 | 6506 / 8392 | 4824 / 8390 |
| swinir-m-x4-real | fp16 | 5851 / 5060 | 3829 / 5059 |
| swinir-m-x4-real | fp32 | 6622 / 8388 | 5045 / 8383 |
| swinir-l-x4-real | fp16 | 9874 / 4561 | 7252 / 5594 |
| swinir-l-x4-real | fp32 | 12063 / 9130 | 10185 / 9135 |
| unik3d-vitl | fp16 | 1294 / 4038 | 1129 / 4036 |

- **restormer (MPS)**: fp16 was no faster than fp32 because of the layout, not the dtype. It now runs channels-last
  on every non-CPU device; it was CUDA only before. The MPS q / k L2 norms are `square().sum().sqrt()`: the norm kernel
  over the pixel axis of channels-last tokens is 3x slower (microbench `[262144, 192]` fp32: `vector_norm` 10.4 ms,
  `square().sum()` 3.3 ms). In half precision, q / k are cast to fp32 once in their stored order, because MPS has no
  fp32-output half GEMM for the Gram matrix. Negatives:
  - NCHW with the norm on the contiguous channel axis: sidd fp16 1295 ms, fp32 1861 ms.
  - NCHW with a contiguous WithBias `layer_norm` output: deblur fp16 1360 ms, but fp32 regressed from 1378 to 1954 ms.
  - fp32 `var_mean` over the NCHW channel axis: deblur fp16 1488 ms.
  - Channels-last while the PixelShuffle output stayed NCHW: mixed-stride residual adds, deblur fp16 1270 ms.
  - The fp32-accumulating half `vector_norm` over a strided axis is 3x slower than the fp32 one (12.3 ms).
- **swin2sr (MPS)**: the profile at 512x512 fp16 (8.1 s) had `F.normalize` of the strided q / k plus its division and
  the cosine product at 3.4 s, and 7 3x3 convs on the channels-last `PatchUnEmbed` view at 1.33 s.
  - Cosine attention now uses `F.rms_norm(x, (d,), d**-0.5 weight)`, which equals `x / |x|`. The weighted MPS kernel is
    fused, sums in fp32 and does not overflow (checked at 1000x input scale); without a weight it is 3x slower.
  - `shared.swin_sr.patch_unembed` returns a contiguous NCHW copy on MPS. The 3x3 conv took 187 ms on the view against
    44 ms contiguous in fp16, and 254 against 47 ms in fp32. HAT's upstream `PatchUnEmbed` is already contiguous.
  - SwinIR shares the helper. The SwinIR "before" rows are from 2026-11-15, so part of their gain comes from other
    commits. A same-commit A/B (opprof, fp16) gave swinir-m-x4 4776 -> 3809 ms and swinir-l-x4-real 7680 -> 7390 ms.
    swinir-l-x4-real's fp16 peak of 5.6 GB is the same with or without the copy (5605 / 5604 MB in the A/B), so the
    rise from the 4561 MB row predates this change. The swinir-l fp32 row ran with 6 GB swap (median 10185, min 9532).
  - Negatives: fp32 `square().sum()` norms written back in place (5.9 s, peak +480 MB); in-place or out-of-place half
    products by the `[..., 1]` factor on the strided slice (6.0 s). A broadcast multiply by a last-dim-1 factor costs
    15 ms per 100M elements on MPS whatever the layout.
  - Left: the per-head cosine product (0.42 s) cannot fold into the rms_norm weight, which is per channel.
- **sea-raft fp32 (clean re-survey, quiet machine)**: m-kitti 5026 / 12688 -> 1423 / 9778, m-things 4162 / 11697 ->
  1425 / 9778, s 2115 / 11645 -> 1049 / 9653. The old rows were pressure runs.
- **monster-plusplus-mix-all fp32**: not measurable cleanly. The median was 32.8 s against a 23.6 s min with 8.4 GB of
  swap, so the old row stays.
- **lingbot-map fp32 golden OOM**: not caused by `assign=True`. With assign True and False MPS memory is identical
  (4418 MB allocated, 5160 MB driver). The fp32 golden needs 11.64 of the 11.84 GiB MPS limit, so the old row (8840 ms,
  golden skipped) stays.
- **video-depth-anything(-metric)-s 480x640**: the stale rows were dropped; the survey takes 994x1482 now.
- **vitmatte-b MLX fp16 peak 1946 -> 3013 MB**: `_fold_constants` in `shared/mlx_graph.py` kept its environment of
  folded intermediates alive (1.1 GB resident; MLX active 1402 against 209 MB). It now clears them after folding.
  Another 140 MB came from zeros-concat padding constants, which are now `F.pad` in vitmatte `_augmented` and
  `sam_attention`.

  | ref (MLX) | dtype | before ms / MB | after ms / MB |
  |---|---|---|---|
  | vitmatte-b | fp16 | 1744 / 3013 | 1670 / 1824 |
  | vitmatte-b | fp32 | 2502 / 3267 | 2487 / 3270 |
  | vitmatte-s | fp16 | 1018 / 2128 | 806 / 1676 |
  | vitmatte-s | fp32 | 1204 / 3067 | 1231 / 3044 |
  | sam-vit-b | fp16 | 446 / 1205 | 441 / 1150 |
  | sam-vit-b | fp32 | 637 / 1646 | 600 / 1642 |
  | sam-hq-vit-b | fp16 | 476 / 1510 | 493 / 1464 |
  | sam-hq-vit-b | fp32 | 680 / 1953 | 648 / 1946 |
  | mask2former-swinl-ade20k | fp16 | 878 / 3657 | 850 / 3614 |

- **unik3d-vitl (MPS fp16)**: the decoder's one-pixel reflection pads were two `torch.cat` calls each (13 calls, 218 ms
  on the channels-last fp16 activations). On MPS they are now `F.pad(mode="reflect")`, which is 2-5x faster (pad plus
  conv at 64x736x1088 channels-last: 99 ms -> 32 ms). fp16 is now faster than fp32 (1129 against 1217 ms). In a
  same-commit opprof A/B the synced total went from 1564 to 1379 ms.
  - Negative: an NCHW decoder on MPS was not better (synced 1378 ms). NCHW pad plus conv is faster in isolation, but the
    `permute` into the MLP linears adds strided copies.
- **By design**:
  - lingbot-depth keeps an fp32 residual stream for NYUv2 AbsRel accuracy, so its fp16 peak stays near fp32.
  - omni-dc keeps the RGB-D backbone in fp32 because its PVT features reach ~1e6. Its full-resolution fp32 convs
    (1008x1488, 0.33 s) are that backbone.
- **Launch-bound, left**:
  - metric3d-vit-large: synced op sum 2.3 s against 1.8 s warm; 167 `cat` calls in the GRU refinement. Its convs cost
    the same in every layout combination (microbench: 262->128 at 154x266, 4.4-4.7 ms).
  - tapnext: 1280-token tensors, thousands of small ops; synced sum 12.0 s against 5.8 s warm. The fp32 residual
    stream and fp32 biases (`add f16 + f32`, `to` 1.06 s) are the card's design around the fp32 scan. A pure-fp16
    port would need new golden evidence.
- **Not done (time box)**: tapnext-pp and locotrack (same autocast-around-fp32-scan pattern as tapnext), mvsplat. Done 2026-11-22, see the section at the end.


## Later idea: fp16 checkpoints from the start (2026-11-22, not started)

Loading is disk-bound (0.3-0.5 GB/s from the external drive, 8-17 s for the big multiview checkpoints). Where the
upstream file is fp32 and every shipped mode that reads it runs fp16, a converted fp16 copy halves the bytes read and
downloaded and drops the cast at load. Per-tensor dtypes inside one safetensors file allow keeping fp32 only where a
golden needs it (fp32 islands, the fp32 mode). Open points: hosting and pinning the converted artifact with its
provenance, re-checking goldens against it, keeping or fetching the original for fp32 runs, and the alternative of a
local fp16 cache written on first load. First step: list stored dtype and size per checkpoint from the safetensors
headers to see which models gain.

### Inventory (2026-11-22, read-only)

Method: every `weights` entry of every card (665 entries, 654 distinct files), found locally by name and size;
safetensors read from the header only, `.pth` / `.pt` with `torch.load(mmap=True, weights_only=True)` (dtype and
`nbytes` only, no data read). Byte shares count floating tensors only.

Strictly, nothing gains: a card counts as fp16-only when no mode on any device lists `float32`, and only marigold-v2
and seedvr2 are such cards. They are already stored in bf16 (about 10 MB of fp32 residue in total). 640 of the 654
files belong to cards that ship float32 and float16, so an fp16 copy would sit next to the fp32 original, not
replace it.

Already half on disk: depth-pro, lotus, stablenormal, birefnet and hed (fp16); marigold-v2 and seedvr2 (bf16).
anysplat (62 % bf16), loma (80 %) and romav2 (55 %) are mixed. xfeat is the only fp32-only card.

What an fp16 copy would save on the fp16 runs, with the fp32 file still needed for `dtype="float32"` (largest 15):

| checkpoint | fp32 GB | fp16 copy saves GB | load s saved at 0.5 / 0.3 GB/s |
|---|---|---|---|
| perception-encoder pe-core-g14-448 | 7.5 | 3.8 | 7.5 / 12.5 |
| perception-encoder pe-spatial-g14-448 | 7.4 | 3.7 | 7.4 / 12.3 |
| da3nested-giant-large-1.1 | 6.8 | 3.4 | 6.8 / 11.3 |
| da3-giant-1.1 | 5.4 | 2.7 | 5.4 / 9.0 |
| vggt-1b | 5.0 | 2.5 | 5.0 / 8.4 |
| mapanything | 4.9 | 2.5 | 4.9 / 8.2 |
| siglip2-g16-384, siglip2-g16-256 (each) | 4.7 | 2.3 | 4.7 / 7.8 |
| c-radio-v3-g | 4.6 | 2.3 | 4.6 / 7.7 |
| lingbot-map | 4.6 | 2.3 | 4.6 / 7.7 |
| vggt-omega-1b-512 | 4.6 | 2.3 | 4.6 / 7.6 |
| dinov2 vitg14, vitg14-reg (each) | 4.5 | 2.3 | 4.5 / 7.6 |
| pi3 | 3.8 | 1.9 | 3.8 / 6.4 |
| pointdit-l-512 | 3.7 | 1.9 | 3.7 / 6.2 |

The top 15 save about 38 GB and 77-128 s of load in total; all 489 inspected fp32 files hold 281 GB of fp32 bytes (an
fp16 copy of everything: about 141 GB). So the gain is real only for the dozen multi-GB ViT checkpoints, and only as
an extra file (a local fp16 cache written on first fp16 load fits that best: no hosting, no second download).

Not covered: 153 local files (38.5 GB) that `weights_only` + mmap could not open (legacy-format pickles: rtmpose /
rtmdet / rtmo, hrnet, mmseg, rfdetr, any2full, bridgedepth, foundationstereo, c-foundationstereo, waft-stereo,
mvsanywhere, depthfm, lama and others), the non-torch formats (sapiens `.pt2` 6 GB, tapnext `.npz`, ppocrv5, crestereo
`.mge`, hitnet `.pb`), and 12 files not present locally (7.5 GB by card size: dinov3-convnext, the dinov3 ViT files of
pointdit / insid3, four loma variants).

## Full MLX golden sweep (2026-11-22)

All 578 `test_mlx[...]` tests: 575 pass (matanyone has no MLX port and skips). `pe-core-g14-448` and `pe-spatial-g14-448`
do not fit the 16 GB Mac (swap above 15 GB even alone); `siglip2-g16-384` passes alone but is killed inside a full run.
`tests/test_mlx*.py` and `tests/test_auto_mode.py`: 71 passed, 1 skipped (CUDA). The eight failures of the first pass were
local setup, not MLX: DINOv3 checkpoints sat in `dinov3/` while the dinov3_convnext, insid3 and pointdit families look in
their own folder or at the top level; the grounding-dino and loma goldens were stale or missing
(`tools/fetch_goldens.py --all <dir>` refreshes them); the five new loma `.pth` files were not downloaded.

Weights layout: a file is found as `<FASTFORKS_WEIGHTS_PATH>/<family>/<name>` or `<...>/<name>`. A checkpoint shared by
several families (the DINOv3 ones) needs a top-level entry (a symlink is enough); family folders hold only their own files.
Run the sweep in chunks, not as one process: the giant models spike swap and kill a long run.

## fp32 gap to MPS on the big transformers (2026-11-22)

MLX fp32 ran about 10 % behind MPS fp32 on the ViT-L-class GraphNet refs. FF_MLX_PROFILE (eager, per op) puts
about 58 % of the time in `addmm` (clip-l14 224: 72 of 126 ms; eomt-l 1024x2048: 407 of 701 ms). The rest is not
the cause: `mx.fast` SDPA beats MPS SDPA (L=4000: 22.0 vs 26.0 ms), manual attention is slower than both,
LayerNorm is faster on MLX and GELU is even.

**Cause found and fixed: transposed weight reads.** Every `nn.Linear` exports as `mm` / `addmm` of `permute(w,
[1, 0])`, so MLX ran the GEMM with a transposed `[out, in]` operand ("NT"). On ViT shapes MLX's NT GEMM is 4-5 %
slower than NN in fp32 and about 3 % in fp16 (bf16: no difference). GraphNet now stores such weights (2-D float
placeholders whose every user is `t` / `permute [1, 0]`) once as their contiguous transpose
(`_transposed_weights`, `_const(transpose=True)`; `FF_MLX_NO_WT=1` turns it off). The transpose runs on the GPU,
then `mx.clear_cache()` drops the untransposed copies (without it, fp16 CLIP-L's peak rose by 58 MB; a CPU
transpose instead added 0.3 s to the first call). The hand-written ports (mlx_vit, depth_anything_v2 mlx_net,
dinov2/dinov3, c_radio, tipsv2) already store `[in, out]` and are untouched.

Survey, second run, portable, ms median / min / peak MB (A = `FF_MLX_NO_WT=1`, B = the change):

| ref | dtype | A | B | MPS |
| --- | --- | --- | --- | --- |
| clip-l14-laion2b 224 | fp32 | 65.1 / 64.9 / 3564 | 63.0 / 62.8 / 3580 | 60.6 / 4426 |
| clip-l14-laion2b 224 | fp16 | 55.6 / 55.5 / 2948 | 54.5 / 54.3 / 2958 | 56.6 / 4424 |
| siglip2-so400m14-378 224 | fp32 | 74.8 / 74.6 / 4666 | 72.4 / 72.2 / 4682 | 75.9 / 5551 |
| siglip2-so400m14-378 224 | fp16 | 65.4 / 65.4 / 3847 | 66.4 / 63.4 / 3825 | 68.6 / 5553 |
| eomt-l-dinov2-ade20k 1024x2048 | fp32 | 691.5 / 690.9 / 4618 | 681.7 / 657.8 / 4504 | 608.2 / 6308 |
| eomt-l-dinov2-ade20k 1024x2048 | fp16 | 581.1 / 574.6 / 3673 | 582.9 / 576.6 / 3663 | 540.0 / 4262 |
| unik3d-vitl 994x1482 | fp32 | 1295.1 / 1293.7 / 3250 | 1229.7 / 1224.7 / 3465 | 1217.5 / 4001 |
| unik3d-vitl 994x1482 | fp16 | 999.2 / 997.8 / 2023 | 947.3 / 944.3 / 1830 | 1129.4 / 4036 |

Peaks stay below MPS everywhere (unik3d fp32 +215 MB over A, still 536 MB under MPS). The eomt B row is noisy
(min 658 vs median 682). Goldens pass: nafnet-gopro, wd-vit-tagger-v3, clip-l14-laion2b, sam-vit-b,
eomt-l-dinov2-ade20k, unik3d-vitl, siglip2-so400m14-378 (all @1).

**What is left, and why it stays.** The remaining fp32 gap (clip 4 %, eomt ~10 %) is MLX's fp32 GEMM itself on
specific shapes; there is no tf32-like mode on Metal to switch on. Microbenchmarks, MLX NN vs MPS `F.linear`:
- unaligned M (CLIP-L's 257 tokens) costs more per row than M=320;
- K=4096, N=1024 (the MLP down projection) at small M is 15 % slower than MPS (low occupancy);
- eomt's 2058-row GEMMs (`[2058,1024]x[1024,4096]` 113 ms x20, the down projection 101 ms x20) dominate its
  profile, then SDPA 101 ms and the ConvTranspose upsampler 52 ms.

Tried, not kept:
- padding M to 16/32/64 mostly loses (e.g. K1024 N4096 M4097 fp32: 8.5 -> 9.2-9.8 ms). The one win, fp32
  M=2035 -> 2048 (4.7 -> 4.0 ms), is a single-shape anomaly that is absent in fp16, and the pad and slice copies
  eat it in a whole net (see "Measured conv dispatch");
- split-K for K=4096 (2 or 4 batched chunks, then a sum): -6 % at M=257, a loss from M=577 up;
- merged QKV and SDPA were already in place, and LN/GELU are not behind.

Not checked: whether the NN layout transfers to MPS (torch's `F.linear` already picks its own layout, so there is
nothing comparable to change there).

## Remaining stereo / flow nets: volume build, pyramid pools, memflow fp16 (2026-11-22)

Scope: selective-igev, monster, monster++, mocha, great-igev, defom, memflow, memfof (M1 Pro 16 GB, 994x1482
stereo / 1080x1920 flow). Each net was profiled with `FF_MLX_PROFILE=1`. Three changes:

- **great-igev concat volume** (`great_igev/net/great_igev.py`, `ExcitiveAttentionVolume.cost_volume`). The chunked
  in-place plane writes exported as `slice_scatter` chains: each disparity copied the whole `[1, 192, 18, 256, 376]`
  chunk, plus `where` plane masks. Together that was 5.6 s of the 17 s eager fp16 call. Under
  `torch.compiler.is_exporting()` the chunk is now built functionally: left broadcast over the planes, right shifted
  and padded per plane, `torch.stack`, then `torch.cat` over the chunks. It keeps upstream's semantics, where the left
  half fills every plane, so `stacked_volume` / `concat_fn` (which zero the left half too) do not fit. CUDA / MPS eager
  is unchanged. The fp32 MLX peak is 9175 MB, against 8921 MB with the in-place build (A/B in one session), so the
  fp32 row's size is not this change.
- **`_window_sum` for non-overlapping windows** (`shared/mlx_graph.py`). The function cast the whole pool input to
  float32 before slicing. On memflow's 1080p all-pairs volume (`[8160, 1, 136, 240]`-shaped levels, ~1e9 elements),
  that was a 4.2 GB float32 copy at fp16 and the 2.1 GB first separable pass at fp32. For `k == s` without padding
  (the correlation pyramids), it now sums the kh x kw strided views with a per-view cast, which compiles to one kernel
  that writes only the pooled output. Pool 2x2 of `[4080, 1, 136, 240]`, compiled: fp16 762 -> 64 MB transient,
  12.0 -> 2.2 ms; fp32 381 -> 127 MB, 8.3 -> 3.9 ms; max diff 6e-5 / 2e-7.
  - Overlapping windows keep the separable path. The one-pass form over 9 views made selective-igev fp16 about 2.5%
    slower: 7.02-7.15 against 6.81-7.06 s per call, A/B in both orders.
- **memflow fp16 on MLX** (card `mlx_dtypes`). The fp16 MLX goldens pass (memflow, -kitti, -sintel).

Survey, second run, portable, ms median / peak MB / first call s (MLX before -> after; MPS = fastest MPS row):

| ref | dtype | MLX before | MLX after | MPS | route |
|---|---|---|---|---|---|
| great-igev-sceneflow | fp16 | 14349 / 9356 / 38.5 | 8028 / 5560 / 19.8 | 9090 / 11106 (portable) | new: MLX fp16 (0.88x, 0.50x memory) |
| great-igev-sceneflow | fp32 | 23111 / 9355 / 118 (11-20) | 12477 / 10685 / 24.9 | 10367 / 11801 | none (1.20x) |
| memflow | fp16 | - | 6997 / 8345 / 11.7 | no fp16 row (fp32 10016 / 12659) | new: auto -> MLX fp16 |
| memflow | fp32 | 10095 / 9232 / 18.9 | 9908 / 7333 / 14.5 | 10016 / 12659 | MLX (unchanged) |
| mocha-v1-sceneflow | fp16 | 9657 / 3382 / 68.1 (11-20) | 6508 / 2182 / 20.2 | 6693 / 5695 (kernels) | new: MLX fp16 (0.97x, 0.38x memory) |
| mocha-v1-sceneflow | fp32 | 11379 / 3750 (11-20) | 10109 / 3545 / 23.9 | 8362 / 8787 | none |
| mocha-v2-sceneflow | fp16 | - | 6354 / 2187 / 18.9 | 6501 / 4643 (kernels) | new: MLX fp16 (0.98x, 0.47x memory) |
| mocha-v2-sceneflow | fp32 | - | 9856 / 3547 / 22.6 | 8323 / 8771 | none |
| monster-plusplus-mix-all | fp16 | 11269 / 3643 / 32.6 | 10966 / 3294 / 21.9 | 10202 / 7347 (kernels) | none (1.07x) |
| monster-plusplus-mix-all | fp32 | 17687 / 5520 / 130.6 (11-20) | 13947 / 6575 / 25.4 | 15001 / 11991 (re-surveyed; min 13973) | new: MLX fp32 (0.93x, 0.55x memory) |
| monster-sceneflow | fp16 | 10863 / 3633 / 36.1 | 10411 / 3295 / 21.3 | 9642 / 6299 (kernels) | none (1.08x) |
| monster-sceneflow | fp32 | 16562 / 5687 / 127.7 (11-20) | 13236 / 6569 / 24.4 | 12916 / 8976 | none (1.02x) |
| selective-igev-sceneflow | fp16 | 6659 / 2167 / 24.3 | 6895 / 1814 / 16.0 | 5709 / 3907 (kernels) | none |
| selective-igev-sceneflow | fp32 | 11452 / 2938 / 29.3 | 11924 / 2678 / 21.8 | 8458 / 7023 | none |
| memfof | fp32 | 3576 / 3012 / 13.0 | 3531 / 2983 / 7.1 | 3054 / 5038 | none (1.16x) |

Notes:

- Most "before" rows are older commits, so the monster, mocha and selective deltas include earlier passes.
- The selective-igev runs vary by ±4% between runs on the shared Mac (fp16 7183 then 6895, fp32 11980 then 11924).
- The monster++ MPS fp32 row was re-surveyed. It still has a median well above its min at 12 GB (15001 / 13973).
  Even against that min, MLX is at 1.00x and 0.55x memory.
- Goldens pass: every MLX golden of great-igev, memflow, memfof, mocha (v1 / v2), monster, monster++, selective-igev
  and raft-stereo-sceneflow, plus nafnet-gopro@1 and wd-vit-tagger-v3@1.

Investigated, not changed:

- **memfof fp16 on MLX**: the goldens fail (flow mean |diff| 0.0018 against tolerance 0.00099). The card stays fp32.
  The fp32 profile is conv-bound: the ResNet-34 encoder and the 1024 -> 768 / 508 heads run at 2.3 TFLOPS, and the
  2048-wide linears at 3.7 TFLOPS. That is MLX's fp32 rate, and the conv table measured earlier already gave memfof
  only -4%.
- **defom fp32**: the profile is 8.6 s conv, 2.7 s SDPA (24 x `[2, 16, 6017, 64]`, 2.7 TFLOPS) and 2.1 s ViT-L
  linears (3.2 TFLOPS). fp16 is out, because DAv2's activations overflow (the runner docstring). bf16 is emulated on
  M1. A mixed run (DAv2 fp32, GRU fp16), as MPS's MetalKernels does, would need per-region dtypes in the replay. It
  was not tried.
- **IGEV-family geometry lookup kernel** (selective / monster / mocha / great): the lookup's elementwise ops are about
  0.6 s per eager call, less once compiled. A Metal custom op like `gwc_volume` would give a few percent at most. Not
  done.
- **Transfer to MPS**: the slice_scatter chain and the float32 pool copy are artefacts of the MLX export and replay.
  On MPS, the eager in-place volume writes and native `avg_pool2d` have no such copies, so nothing carries over.

## C-FoundationStereo on MPS and MLX: strips, bmm attention, classifier taps (2026-11-22)

Goal: c-foundationstereo-s fp16 OOMed on MPS at 994x1482 ("Tried to allocate 5.38 GiB"). The task was to find the
cause, then cut peak memory and time on MPS and MLX.

Profile (MPS fp16, synchronised module hooks, padded 1024x1504; the 1/4 volume is 28 x 104 x 256 x 376, 560 MB in fp16):

| stage | time | driver memory |
|---|---|---|
| cost_agg (hourglass) | 17-22 s | grows to 15.5-16.7 GB |
| merge_attention (inside cost_agg) | ~12 s | 6.8 -> 13.4 GB |
| conv_out (inside cost_agg) | 2.9 s | +2.1 GB in its (17, 1, 1) conv |
| classifier | 1.9-3.0 s | 1.5 s or more in the 14 -> 1, k7 `Conv3d` |
| update_block (22 iterations) | ~8 s | - |

The cause is TAO's disparity transformer. It runs at full 1/4 resolution: 96256 sequences of length 104, head dim 7.
`merge_attention` first interpolates the whole 1/4-resolution volume, then holds the permuted copies and the chunk
outputs next to it. The row strips that fix this (`_merge_attention`, `_hourglass_tail`) were CUDA-only.

Changes (`c_foundationstereo/portable.py`):

- **Row strips on MPS.** `_use_volume_strips` now takes MPS volumes, never under the MLX export. The Portable runner
  passes `mps=True`. The output is bit-identical to the unstripped path (736x1024 fp16, both stripped hooks ran: max
  |diff| 0).
- **Attention as `bmm`, softmax, `bmm`** (MPS and the MLX export), instead of `scaled_dot_product_attention`:
  - MPS, one 20-row strip: 551 ms against 912 ms, with 1.6 GB of driver growth against 3.7 GB.
  - MLX pads head dim 7 to 64 for its fused kernel: 54 ms per 8192-sequence chunk, against 38 ms for the matmul form.
- **Classifier taps.** The classifier's last `Conv3d` (14 -> 1, k7) runs as one `conv2d` whose 7 output channels are
  the depth taps, then a shift-add along depth (the research FS `_conv_k7_to1` idea). On MPS this is 409 ms against
  1504 ms for `mps.conv3d`, which runs one `conv2d` per tap. The MLX export uses it as well (`_PortableMlx`).

Hooked profile afterwards: driver peak 7.3-8.4 GB (was 15-16.7), total 24-26 s (was 34-43 s).

Results (994x1482, ms median / MB peak footprint, second survey run). The MLX first-call column is the compile time in
seconds:

| ref | dtype | MPS before | MPS after | MLX before (first call s) | MLX after (first call s) |
|---|---|---|---|---|---|
| c-foundationstereo-s | fp16 | OOM | 25783 / 12573 | 27235 / 4625 (130) | 18737 / 4010 (31) |
| c-foundationstereo-s | fp32 | OOM | OOM | 62607 / 8203 (217) | 29962 / 7023 (45) |
| c-fast-foundationstereo | fp16 | 2714 / 8484 | 2539 / 8483 | - | 2737 / 2858 (7.8) |
| c-fast-foundationstereo | fp32 | 4881 / 12882 (11-06) | 3184 / 13079 | - | 4565 / 4181 (10.5) |

Notes:

- The c-fast-foundationstereo fp32 "before" row is from 2026-11-06, so its delta includes earlier passes.
- A first survey pass ran without `FASTFORKS_GOLDEN_PATH` (golden "skip"), so `gen_auto_mode` dropped the rows. It was
  re-run with goldens and both runs passed.

Routes (`apple_mlx.json`):

- c-foundationstereo-s now has `auto` / `dtypes` fp16 -> MLX portable: 1.38x faster than MPS at 0.32x the memory, and a
  12.6 s compile against 25.8 s MPS.
- c-fast-foundationstereo gets an `mlx_default` entry. MLX is slower than MPS (2737 against 2539 ms), so there is no
  `auto` entry.

Goldens pass: test_mps (FASTFORKS_MPS_FULL=1) and test_mlx for c-foundationstereo-s@1 and c-fast-foundationstereo@1.

Not done, or rejected:

- **c-foundationstereo-s fp32 on MPS** still OOMs: 9.75 GiB allocated, then a 1.06 GiB request, which is one fp32
  full-resolution 1/4 volume. A synchronised hooked run completes at 14.5 GB driver memory and is swap-bound (120 s).
  The volumes are 1.1 GB each in fp32, so fp32 at this size does not fit 16 GB without stripping every 3-D stage.
  MLX fp32 is the route (30 s, 7.0 GB).
- **Fused QKV projection** (one 28 -> 84 linear plus split): 805 ms against 551 ms per strip on MPS. The split and
  reshape copies cost more than the two saved linears. Rejected.
- **Attention chunk size** (2048 to 16384 sequences) did not change the MPS time (912 ms each). Only the driver growth
  changed. `CHUNK` is unchanged.
- **Generic `mps.conv3d` taps path** for every small-output `Conv3d`: the IGEV-family 8 -> 1, k3 classifier goes from
  19 to 12 ms. That is too small for a shared `mps.py` change that would re-touch every family's goldens. The taps path
  stays local to C-FS.
- **The (17, 1, 1) convs** of `Conv3dNormActReduced` cost 0.5 s each through `mps.conv3d`, against 190 ms as a
  `(17, 1)` `conv2d` over `(C, D, H*W)`. That would be a shared `mps.py` change. Not done.
- **Research foundationstereo-vit-s / vit-l** use separate code (`forward_fast`). It already has strip-wise volumes, the
  k7 classifier trick, and attention over 4 heads (sequence length 4, math backend). None of these changes apply to
  it. Its 13.2 GB MPS fp16 peak was not re-profiled in this pass.
- **The remaining MPS time** is the attention (~7 s; it has to materialise the 104 x 104 scores without a fused kernel)
  and the 22 GRU iterations (~8 s, 2-D convs at 1/4 resolution).

## 2026-11-22: tapnext, tapnext-pp, locotrack, mvsplat (M1 Pro 16 GB)

Second survey run each, `--modes portable`; ms / MB peak footprint / first call s.

| ref | dtype | MLX before | MLX after | MPS |
| --- | --- | --- | --- | --- |
| tapnext | fp32 | 7477 / 3673 / 77.1 | 4034 / 2443 / 6.5 | 5855 / 3754 |
| tapnext-pp | fp32 | no row | 4036 / 2025 / 6.4 | 5680 / 3308 -> 5610 / 3657 |
| tapnext-pp | fp16 | - | - | 5785 / 3310 -> 5654 / 3552 |
| locotrack-b | fp32 | 1484 / 1608 / 7.3 | 1482 / 1738 / 5.1 | 2104 / 2474 |
| locotrack-s | fp32 | no row | 1209 / 1653 / 4.7 | 1781 / 2427 |
| mvsplat-re10k | fp16 | 2116 / 2706 / 8.7 | 1258 / 2541 / 5.8 | 1123 / 4323 |
| mvsplat-re10k | fp32 | 2573 / 3466 / 9.1 | 1573 / 3456 / 6.3 | 1040 / 4540 |

The MPS tapnext-pp rows replace stale golden-"skip" rows.

Changes:

- **tapnext / tapnext-pp**: the MLX port is a per-frame `Steps` graph (`_step`, pipelined, released) instead of one
  unrolled whole-video graph. A `first` flag replaces the `h=None` reset, so every frame uses the same graph. Compile
  drops from 77 to 6.5 s; warm time drops 46%, peak memory 33%.
- **mlx_graph**: the per-segment `drops` list is built with bisect instead of a quadratic scan. A `raw=` passthrough
  lets a step graph keep MLX arrays between calls.
- **locotrack**: the MLX export uses 128-query chunks (`MLX_QUERY_CHUNK`; eager stays at `QUERY_CHUNK`). Every chunk is
  unrolled in the export, so 64 -> 128 cuts export 5.4 -> 3.4 s for 4% more warm time and +130-180 MB. This lets
  locotrack-s pass the cold-factor rule (it missed with 5.76 s compile against 3 x 1.78 s).
- **mvsplat**: new custom op `fastforks::warp_dot` (plane-sweep warp plus channel dot, `_core/volume_ops.py`). Eagerly
  it is the grid_sample loop; the MLX replay lowers it to one Metal kernel (`kernels/graph_mlx/ops.py`, one SIMD group
  per view/depth/pixel, fp32 accumulation). It never materialises the `[N, C, D, H, W]` warped volume. It is only used
  under `EXPORTING_FOR_MLX`, so MPS and CPU are unchanged.
- **mvsplat `sh_rotation`**: cast the float64 constants on the CPU before upload. test_mps[mvsplat-re10k@1] failed at
  HEAD with a float64 MPS TypeError; it passes now.

Routes (`apple_mlx.json`): tapnext@1 gains `auto` / `dtypes` (fp32 portable). New entries for tapnext-pp@1 and
locotrack-s@1 (`auto`, `dtypes`, `mlx_default`, fp32 portable). No mvsplat auto route.

Investigated, not changed:

- **tapnext `FF_MLX_CHUNK_MB=2048`**: no gain. The step is GEMM-bound: the MLP and QKV linears are about 3.8 of
  10.2 s eager.
- **locotrack chunk 256**: export 2.3 s but 8% slower warm. Kept 128.
- **mvsplat remainder**: the full-resolution refine convs (163 -> 168 channels at 160x256, about 137 ms per call) and
  the instance-norm stats. The `channel_stats` fast path in `_batch_norm_batch_stats` is gated to kernels mode, and
  mvsplat has no MLX kernels mode. Left as is.
- **MPS warp_dot**: MPS keeps the grid_sample loop. A Metal kernel for torch MPS would be a separate job; it is a
  transfer idea from the MLX kernel.

Goldens pass: test_mlx for tapnext, tapnext-pp, locotrack-b/s, mvsplat-re10k, plus nafnet-gopro, wd-vit-tagger-v3 and
c-fast-foundationstereo (shared mlx_graph change); MPS/CPU goldens for tapnext, locotrack and mvsplat (`--slow`);
test_auto_mode, test_mlx, test_mlx_plan_cache, test_mlx_shape_caches (67 passed).

## 2026-11-22: lingbot-map MPS memory, cotracker3-online steps, nnU-Net glue (M1 Pro 16 GB)

Second survey run each, `--modes portable`; ms / MB peak footprint / first call s.

### lingbot-map (MPS)

| dtype | MPS before | MPS after | MLX (unchanged) |
| --- | --- | --- | --- |
| fp32 | 8840 / 13055, golden skip (survey: OOM in the golden reload) | 9234 (min 9021) / 9658, golden pass | 9970 / 8930 / 18.7 |
| fp16 | 7124 / 7956 | 7609 (min 7603) / 5962 | 7261 / 5438 / 14.0 |

Profile (MPS fp32, 8 views 480x640, one call about 9.2 s): global attention 24 x 112 ms = 2.7 s, depth head about
0.9 s, camera head 0.06 s. Weights 4.4 GB, PyTorch-allocated peak about 6.6 GB (KV cache 1.6 GB). The MPS driver
memory grew to 11.7 GB in the DPT depth head: its 8-view maps at 1/1 resolution (output_conv2 / refinenet, about
0.8 GB each in fp32) each took a heap of their own.

Changes:

- **lingbot-map depth head on MPS one view at a time** (`frames_chunk_size=1`, as AnySplat). The views are
  independent, so the output is the same. Footprint peak 13.1 -> 9.7 GB (fp32), 8.0 -> 6.0 GB (fp16). The eager
  profile showed no slowdown (8.5 s warm either way); the survey's second run is 4-7% slower than the old rows, the
  first run was not (fp32 8502, fp16 7367). CUDA and the MLX export (traced on the CPU) keep the 8-view chunk.
- **speed_survey / merge_check**: a golden with load options (lingbot-map: `scale_frames`, `window`) reloads the
  model. The timed copy is now dropped and the cache emptied before that reload; two fp32 copies (2 x 4.4 GB weights)
  ran the 16 GB Mac out of memory, which is why the fp32 golden was "skip".

Investigated, not changed:

- **Chunked global-attention queries**: no gain. MPS SDPA at head dim 64 is fused (8336 tokens fp32, no driver
  growth, 112 ms per layer); it is not the memory cause.
- **Per-view depth head on MLX**: not tried in this time box. The MLX rows already peak below MPS (8.9 / 5.4 GB).

No route change (gen_auto_mode: no diff). Goldens: test_cpu_portable and test_mps for lingbot-map pass; the survey's
fp32 and fp16 goldens pass.

### cotracker3-online (MLX)

| dtype | MLX before | MLX after | MPS |
| --- | --- | --- | --- |
| fp32 | 3340 / 3971 / 6.7 | 2314 (min 2269) / 2749 / 4.6 | 1707 / 4689, golden skip -> 1731 / 4767, golden pass |

Profile (24 frames, 256 queries, 2 windows): on MLX the feature CNN was about 1.5 s and the refinement about
1.5 s per call; on MPS 0.46 s and 1.43 s. The refinement ran its correlation in PyTorch on the CPU, only the update
transformer on MLX.

Changes:

- **Step graph** (as tapnext): one refinement iteration (correlation, its MLP, update transformer) is one exported
  graph (`_OnlineSteps._iteration`), run 6 times per window. The window's levels and support features go to MLX once
  per window and the track state stays on MLX (`raw`). One unrolled window graph was as fast warm but took 9.1 s on
  the first call (step graph 5.4 s).
- **One CNN graph**: the first window encodes its 16 frames as two 8-frame chunks, so the CNN is exported once.
  Peak 3800 -> 2750 MB, first call 5.4 -> 4.6 s.
- **`cache_mb=256`**: the default 128 MB cache made each iteration's temporaries fresh Metal allocations. At the same
  code: 3150 ms at 128 MB vs 2460 at 256 MB (512: 2363 ms / +240 MB; 1024: 2212 ms / +730 MB). 256 is the memory pick.
- **mlx_graph**: `aten.exp2` mapping (`posenc`'s scales; the offline whole-graph export folded it away).

Investigated, not changed:

- **The remaining gap is the feature CNN.** MLX fp32 conv2d is irregular in the output channel count: at 96x128,
  8 frames, 96 -> 96 takes 22.6 ms (MPS 3.6) but 96 -> 128 takes 7.0 and 128 -> 128 takes 6.6. Couts 80, 112 and
  144 hit the same slow path. cotracker's layer2 (96 channels) is about 106 ms of the 270 ms CNN per 8 frames.
  Zero-padding layer2 to 128 channels in the MLX export would save about 60 ms per 8 frames; a generic cout-padding
  rule in mlx_graph's conv lowering would need a sweep over the ported models (the cost is not monotone in cout).
  Left for a later job; it would also help cotracker3-offline.
- **MPS**: no transfer. The MPS refinement is already on the GPU (no CPU correlation), and step graphs only help the
  MLX export.

Route: no change (MLX stays 1.34x slower than MPS). Goldens: test_mlx for cotracker3-online, plus nafnet-gopro,
wd-vit-tagger-v3, c-fast-foundationstereo, tapnext, tapnext-pp (mlx_graph change) pass; survey goldens pass.

### nnU-Net glue: TotalSegmentator sliding-window merge (MLX)

ms / peak footprint MB / first call s (survey, second run):

| ref | dtype | MLX before | MLX after | MPS |
| --- | --- | --- | --- | --- |
| totalsegmentator-total-3mm | fp16 | 1111 / 4730 / 3.8 | 740 (min 731) / 3246 / 1.9 | 522 / 2588 |
| totalsegmentator-total-3mm | fp32 | 1208 / 4834 / 4.3 | 828 (min 811) / 3031 / 2.2 | 593 / 4385 |
| totalsegmentator-total | fp16 | 31549 / 5163 / 52.7 | 33946 (min 33703) / 3416 / 38.1 | 23070 / 6058 |
| totalsegmentator-total | fp32 | 43030 / 5924 / 58.1 | 42521 (min 41505) / 4198 / 47.6 | 27804 / 10180 |

Profile (3mm, fp16, 200^3, one call): MLX 1138 ms, of which the window networks (`_logits`) 615, the CPU adds of
the Gaussian-weighted window logits into the float16 sums about 330, and the argmax (upload of the sums to MLX) 155.
MPS 629 ms: networks 501, merge and argmax on the GPU. Preprocessing and resampling are under 5 ms on both (the
bench volume is already at the plan spacing; the resampled path is the same PyTorch code on both backends).

Change: `MlxPredictor._accumulate` keeps the window logits, the float16 ring buffer of sums (`[heads, patch depth,
H, W]`) and the argmax on MLX when there is one fold network, no mirroring, argmax labels and a plain `net(x)`
forward (TotalSegmentator's tasks). The rounding is `Predictor`'s (float32 logits times the float16 Gaussian, added to
the float16 sums in float32, rounded to float16), so the labels are bit-identical to the CPU merge. The 118-head
float32 window logits no longer come back to the CPU, which is the memory saving. Other cases (several folds,
mirroring, regions, resampled logits) keep the CPU merge.

On totalsegmentator-total (1.5 mm, 8x more windows) the merge is a small share of the call: peak memory drops by
about 1.7 GB, fp32 is unchanged within noise (-1 %), fp16 is 7 % slower in this run (33.9 s vs 31.5 s; the first
call is 14.5 s faster). The fp16 difference is network-dominated and was not re-measured within the time box; a
later quiet-machine re-run should confirm or refute it.

Investigated, not changed:

- **The rest of the gap is the network.** 3mm `_logits` is 615 ms on MLX vs 501 on MPS; 3D conv lowering is the
  next target, not the glue.
- **MPS**: no transfer. The MPS merge already runs on the GPU in the same PyTorch code.
- **nnfoundation, nninteractive**: they port the whole runner (`mlx_wrap.port(Portable)`), not `MlxPredictor`, so they
  keep their merge.

Route: no change (MLX stays slower than MPS on both refs). Goldens: test_mlx for totalsegmentator-total-3mm and totalsegmentator-total pass; survey goldens pass.

**Re-measure (quiet machine, swap 2.0 GB and flat, nothing else running): the fp16 slowdown was noise.** One process,
both paths interleaved (`MlxPredictor._accumulate` vs the CPU merge through a temporary env switch, a warm-up call of
each, then 3 alternating pairs): MLX merge median 32074 / min 31885 ms, CPU merge median 33068 / min 32658 ms, so the
MLX merge is about 3 % faster, as on 3mm. Two back-to-back survey runs gave 30669 (min 30600) and 33823 (min 33108)
ms at the same code: a 33-second call drifts by about 10 % with the machine's thermal state, more than the merge's
effect. Recorded row (second run, as the protocol says): 33823 / 3402 MB / 38.2 s, golden pass. No code change.

## 2026-11-22: MLX conv output-channel padding to a multiple of 64 (M1 Pro 16 GB)

Micro-benchmark (MLX 0.32.4, `mx.set_cache_limit(256 MB)`; without a cache limit the sweep pushed swap to 10 GB and
the timings were worthless), 8 x 96x128 inputs, output counts 24 ... 320:

- 3x3 conv2d, stride 2: padding the outputs up to a multiple of 64 (weights zero-padded, output sliced) wins almost
  everywhere, often 2-5x (64 -> 96 2.8 -> 1.4 ms). Stride 1: wins for cout % 64 in {16, 48} and for 96
  (128 -> 96 41 -> 7 ms, 96 -> 144 67 -> 6.4, 256 -> 112 83 -> 10.5); 160 / 224 / 288 are neutral or a bit slower.
  float16 behaves alike.
- 1x1 conv2d: float32 wins up to 4x (256 -> 208 19.5 -> 4.8 ms); float16 neutral.
- Many of the biggest raw wins came from the padded shape landing in MLX's Winograd path. Those are excluded: the
  rule never pads when the original or the padded count would be Winograd (`_mlx_winograd`), so the output stays
  bit-identical (288 shapes checked through `_conv2d` with the flag toggled: 0 differing outputs; sum 3255 -> 2517 ms).
  The first version without that guard made 256 -> 96 slower (15.3 -> 19.75 ms).
- matmul / Linear: only 5-15 %, not done.
- Conv3d: wins at 80 / 96 / 144 / 160 (fp16) and 80 (fp32), but no surveyed 3-D net uses those counts (nnU-Net is
  32 / 64 / 128 / 256 / 320), so not done.
- PyTorch MPS conv2d: no such effect, padding only adds cost (96 -> 96 fp32 3.32 -> 4.43 ms). No MPS transform.

Implemented in `shared/mlx_graph.py` `_conv2d` (groups 1, no dilation, cout > 64 and not a multiple of 64; 1x1 only
in float32 at stride 1; at stride 1 only for cout % 64 != 32 or 96). `FF_MLX_NO_OUT64=1` switches it off.

| ref (portable) | dtype | before ms / MB | OFF (same session) | after ms / MB |
| --- | --- | --- | --- | --- |
| raft-stereo-sceneflow | fp32 | 6216 / 4653 | 6272, 6625 | 5271, 5336 (min 5197) / 4661 |
| raft-stereo-sceneflow | fp16 | 4219 / 3863 | | 4163 / 3867 |
| cotracker3-online | fp32 | 2314 / 2749 | 2225 | 2135, 2233 / 2747 |

raft-stereo fp32 is about 15 % faster (OFF/ON interleaved, so not drift); fp16 and cotracker3-online are within noise
(cotracker's 96 -> 96 conv was 22.6 ms only in the slow, state-dependent regime; alone it is about 7 ms). Goldens
pass (test_mlx: nafnet-gopro, wd-vit-tagger-v3, c-fast-foundationstereo, tapnext, tapnext-pp, cotracker3-online;
survey goldens for the rows above). No route change (`gen_auto_mode --check` clean).

## 2026-11-22: float16 on MLX for float32-only families with mixed dtypes (M1 Pro 16 GB)

Families whose card had float16 off on MLX, now on with float32 islands. Goldens and tolerances are unchanged. The
times are the survey's (median of 5, the second run, golden pass, 994x1482 stereo / 1080p flow / 768x1024 matching).

| ref (portable MLX) | before fp32 ms / MB | after fp16 ms / MB | fp32 now | golden fp16 (tol) | route |
| --- | --- | --- | --- | --- | --- |
| memfof (+ spring) | 3531 / 2983 | 3381 / 2417 | 3684 / 2964 | mean 0.00068 (0.00099) | mlx_default fp16 |
| defom-stereo-vitl-sceneflow | 13878 / 6458 | 10693 / 5189 | 13339 / 7039 | mean 0.00165 (0.02) | mlx_default fp16 |
| dedode-v2-b | 2534 / 5327 | 1756 / 4273 | 2508 / 5450 | matches pass (0.986) | mlx_default fp16 |
| dedode-v2-g | 5544 / 6499 | 3329 / 4871 | 4161 / 6626 | matches pass (0.986) | **auto -> MLX fp16** (MPS 3767 / 10008) |

What each needed:

- memfof (`c83edfea`): the flow state, its accumulation and the convex-upsampling weights as `to_float32` islands
  (`.float()` records nothing on the float32 twin, so upstream's casts vanished in the replay), plus
  `half_wino_f32=True`. Islands alone: mean 0.0015; Winograd alone: 0.00113; both: 0.00068. The correlation stays
  float16. The one Winograd conv at 1080p (1024 -> 512, 8160 positions) costs about 500 ms in float32. A float16 direct
  conv in 3 row chunks was tried (24.3 ms vs float32 22.6 ms vs half Winograd 14.1 ms, error 3.6e-4 vs 6.1e-3) and is
  not cheaper.
- defom-stereo (`08ab387c`): the whole-graph port split into three graphs, as `kernels/metal.py` does on MPS:
  `_encode` (DAv2 ViT + depth head, float16), `_dfeat` (DPT feature head + fnet's `convd`, float32 via
  `fp32_methods`: the refinenet1 / refinenet2 activations exceed float16) and `_loop` (cnet / fnet, the correlation and
  32 updates in float16; the disparity state and the upsampling are `to_float32`). A `to_float32` island cannot make a
  conv float32 inside a float16 graph (`_conv_input` casts its input to the half weights), so a separate graph is the
  only way. The float32 run also goes through the split now: 13.3 s vs 13.9 s, but +580 MB peak (tensors between the
  graphs).
- dedode (`599c5a3b`): only `half_wino_f32=True`. The earlier float16 failure (keypoints mean 0.0022) was MLX's half
  Winograd on the VGG convs: without it the match agreement is 0.9853 < 0.986.

Not done or rejected:

- memfof MPS float16: upstream's `torch.autocast("cuda")` does nothing on MPS. A "float16" MPS run matched float32
  exactly, so the dtype would only be a label. It needs real MPS autocast or halved weights first.
- defom MPS portable float16: not tried. `kernels_mps` already runs the float16 recipe.
- romav1 / romav2, alltracker, loma-b: not reached in the time box. romav2's card notes float16 overflows in
  upstream's bfloat16 regions. Given dedode, try `half_wino_f32=True` first for the conv-heavy ones.

## 2026-11-22: float16 on MLX for romav1, alltracker, loma (M1 Pro 16 GB)

Continued from the section above, same rules (goldens and tolerances unchanged; survey second run, median of 5;
romav1 / loma 768x1024 matching, alltracker 480x854).

| ref (portable MLX) | before fp32 ms / MB | after fp16 ms / MB | fp32 now | golden fp16 (tol) | MPS best | route |
| --- | --- | --- | --- | --- | --- | --- |
| romav1 | 4440 / 4499 | 3135 / 2460 | 3733 / 4487 | warp mean 0.00049 (0.00079) | fp32 3050 / 7979 | mlx_default fp16 |
| alltracker | 5397 / 2597 | 2288 / 2232 | 3151 / 2864 | tracks mean 0.00076 (0.0019) | fp16 3200 / 3806 | **auto -> MLX fp16** |
| alltracker-tiny | (no row) | 2174 / 2245 | 2909 / 2861 | pass (vis p99 0.00126 / 0.0014) | fp16 2743 / 3640 | **auto -> MLX fp16** |
| loma-b | 3860 / 6150 | 2760 / 4561 | 3617 / 6334 | recall 0.9956 (0.96), agree 0.9907 (0.93) | fp32 3041 / 8137 | **auto -> MLX fp16** |

The fp32 "now" column also moved (romav1 4440 -> 3733, alltracker 5397 -> 3151). The old rows predate the conv
output-padding change and other replay work on this branch; that was not attributed further.

What each needed:

- romav1 (`70492635`): nothing but the card's dtype list.
- alltracker (`1698403a`): `to_float32` islands for the flow / visibility state and its accumulation, the window's
  zero state, the queries and the convex upsampling. Those islands brought tracks from 0.0158 to 0.0078, still
  4x the tolerance. The rest came from float constants: the `device_tensor` scale factors (grid normalisation,
  query / track pixel mapping) are graph constants, so the replay casts them to float16 (up to 0.25 px off at 512 px).
  `scale_xy` multiplies by Python floats instead, which stay exact in the replay: tracks 0.00087. alltracker-tiny
  also needed `half_wino_f32=True` (visibility p99 0.0023 -> 0.00126).
- loma (`153f9c95`): only `half_wino_f32=True` (its detector / descriptor are dedode's). All six goldens pass
  (b, b64, b128, g, l, r); only loma-b is surveyed. The MPS loma-b row had golden "skip", so it gave no route
  baseline. It was re-surveyed with the golden path (3041 / 8137, pass), and auto then takes MLX.

Ruled out on alltracker (each still about 0.0075-0.0079 before the constant fix):

- a float32 residual stream in `CNBlock2d`, also with `half_wino_f32`: no margin gain on tiny;
- `FF_MLX_KEEP_F32_CONV` (the whole update block in float32);
- the correlation matmul in float32;
- `FF_MLX_NO_DW_KERNEL`.

An encoder bisect showed MLX's float16 feature error is only about 1.45x CPU autocast's, so the encoder was not the
cause.

Not done:

- romav2: the `kernels_mlx` Engine already runs the selective FP16 recipe (`half=kernels` projections and convs)
  under the float32 label, and the card notes plain float16 overflows upstream's bfloat16 regions. A portable fp16
  dtype would duplicate that, so it was skipped.
- depthsplat-small-re10k: `half_wino_f32=True` alone still fails (depth mean 0.00351 > 0.0035, confidence 0.016 >
  0.014). The net's inputs include the normalised intrinsics, the scaled cam-to-world poses and near / far, and the
  replay casts all of them to float16 at graph entry. The projections in the cost volume then see float16 cameras
  (the alltracker lesson again). A `to_float32` island cannot undo that, because the values are rounded before it.
  The likely fix is a way to keep chosen graph inputs float32 (an `mlx_graph` change, which needs its own
  regression runs), so it was not done in this time box.
- Other float32-only MLX families (unidepth_v2, zoedepth, insid3, inspyrenet NaN / golden failures; sam2, grounding
  dino) were not reached. Before adding islands, check for float constants and float inputs that carry geometry
  (`device_tensor`, camera inputs): they are the cheapest failure to miss.

## 2026-11-22 (afternoon): float16 on MLX for depthsplat and the float32-only families (M1 Pro 16 GB)

Continued from the "Not done" list above, same rules (goldens and tolerances unchanged; second survey run, median
of 5). One attempt per family, cheapest first: card dtype only, then `to_float32` islands, float32 graph inputs,
`half_wino_f32`. All seven now pass float16.

| ref (portable MLX) | old fp32 ms / MB | fp16 ms / MB | fp32 now | golden fp16 (tol) | MPS best | route |
| --- | --- | --- | --- | --- | --- | --- |
| depthsplat-small-re10k | - | 709 / 1716 | 820 / 2133 | depth mean 0.00040 (0.0035), conf 0.0033 (0.014) | - | mlx_default fp16 |
| insid3 | 3211 / 3003 | 2512 / 2227 | 3216 / 2929 | similarity mean 0.00053 (0.002), p99 0.00144 (0.005) | fp16 2510 / 3414 | **auto -> MLX fp16** |
| grounding-dino-tiny | 1043 / 1645 | 753 / 1499 | 933 / 1452 | pass (all 4 variants) | fp32 846 / 4787 | mlx_default fp16 |
| grounding-dino-mm-tiny | - | 766 / 1507 | 935 / 1453 | scores p99 0.0063 (0.025) | fp32 1104 / 4804 | mlx_default fp16 |
| grounding-dino-base | - | 947 / 1705 | 1227 / 1952 | pass | fp32 1087 / 6001 | mlx_default fp16 |
| grounding-dino-mm-base | - | 916 / 1719 | 1254 / 1953 | pass | fp32 1273 / 4952 | mlx_default fp16 |
| zoedepth | 862 / 2895 | 759 / 1955 | 957 / 2958 | depth mean 0.00029 (0.01), p99 0.00093 (0.03) | fp32 778 / 3558 | mlx_default fp16 |
| inspyrenet (1080p) | 1273 / 2332 | 679 / 1741 | 1121 / 2441 | alpha p99 0.00064 (0.25) | fp16 702 / 4172 | mlx_default fp16 |
| inspyrenet-fast (1080p) | - | 96 / 1060 | 129 / 1303 | alpha p99 0.00048 (0.25) | fp16 145 / 2071 | mlx_default fp16 |
| sam2.1-hiera-tiny | 184 / 1174 | 141 / 933 | 176 / 1051 | pass | fp16 129 / 1514 | mlx_default fp16 (was fp32) |
| sam2.1-hiera-small | - | 164 / 951 | 204 / 1068 | pass | fp16 148 / 1550 | mlx_default fp16 |
| sam2.1-hiera-base-plus | - | 282 / 1099 | 352 / 1467 | pred_masks mean 0.0018 (0.0026), p99 0.0079 (0.009) | fp16 394 / 3988 | mlx_default fp16, **dtype=float32 -> MLX** |
| sam2.1-hiera-large | - | 597 / 1460 | 713 / 1932 | pass | fp16 544 / 2206 | mlx_default fp16, **dtype=float32 -> MLX** |
| unidepth-v2-vitl14 | 1691 / 3855 | 1173 / 1825 | 1254 / 3337 | depth mean 0.00062 (0.01), p99 0.00135 (0.02) | fp32 1252 / 6077 | **auto -> MLX fp16**, dtype=float32 -> MLX |

Where MLX float16 is faster but auto stays on MPS (grounding dino, zoedepth, inspyrenet, sam2), the first-call
compile is over the 3-MPS-runtime cold budget, or MPS float16 is simply faster (sam2 tiny / small / large).

What each needed:

- depthsplat (`d44ac91b`): a new `mlx_graph` mechanism, `fp32_inputs` (argument names passed through `MlxModule` /
  `mlx_wrap.port`). `GraphNet` keeps those inputs float32 instead of casting every floating input to the run dtype
  at graph entry. depthsplat keeps its cameras (intrinsics, poses, near / far) float32. `half_wino_f32` was not
  needed. Covered by a new `tests/test_mlx.py` case.
- insid3 (`4d004c0f`): the DINOv3 stream scale (`_stream_scale`) is now also applied when the MLX twin runs float16
  (`mlx_wrap.run_dtype(ctx) == "float16"`), not only for the MPS half net. The patch tokens leave through
  `to_float32`. The dinov3 MLX goldens still pass.
- grounding dino (`ba3735d1`): the `.float()` casts on the decoder hidden states, the text hidden states and the
  last reference points become `to_float32` (`.float()` records nothing on the float32 twin). mm-tiny scores p99
  0.029 -> 0.0063.
- zoedepth (`ec081b23`): the export decomposes `nn.Softplus` into `exp(x)`, which overflows float16 above about 11;
  the seed bin centres reach 21. In the float16 twin, every Softplus is swapped for `relu(x) + log1p(exp(-|x|))`. A
  remaining inf in the inverse attractor's `alpha * dx^2` is harmless, because it is divided to 0.
- inspyrenet (`81b8b3ce`): the SICA context `bmm` reaches 1.6e5, with BN variances up to 5.4e10. `to_float32` alone
  fails, because the next 1x1 conv casts its input back to float16 (inf). In the float16 twin, the context is scaled
  by 1/4096 and the following key / value BN `running_mean` / `running_var` by 1/4096 and 1/4096^2. The convs are
  bias-free 1x1, so the scaling is exact. The float16 SICA override now also applies to the twin.
- sam2 (`89b0e75b`): the MLX export runs each Hiera block in autocast's layout. The residual stream and its norms
  stay float32, while attention and the MLP run in the run dtype (`to_run_dtype` / `to_float32`). The neck's
  `.float()` casts become `to_float32`. Both parts are needed: the stream alone gives base-plus p99 0.0096, and the
  neck alone gives p99 0.0127.
- unidepth_v2 (`cf6ce2b0`): only the card's dtype list. The plain float16 replay passes, so the earlier "float32
  only" note no longer holds.

Rejected or not needed:

- `half_wino_f32` on depthsplat (not needed once the cameras were float32) and on sam2 (no change).
- A shared `mlx_graph` softplus -> `logaddexp` lowering: no effect, because softplus is decomposed before it reaches
  the op table. It was reverted in favour of the zoedepth-local module swap.
- The partial sam2 fixes above (stream only, neck only).

Lessons: float overflow on large activations (softplus, attention context) needs a value-level fix (a stable
formula or a folded scale), not an island, because the next op casts back to float16. Search the export for
`.float()` first: on the float32 twin it is a no-op.

Not done:

- MPS float16 for these families was not revisited. Of the MLX tricks, the inspyrenet context scale and the zoedepth
  softplus could transfer to MPS half nets if their MPS float16 paths ever overflow.

## 2026-11-22 (evening): MPS `(kd, 1, 1)` Conv3d as one `conv2d` (M1 Pro 16 GB)

`mps.conv3d` ran a depth-only kernel (`Conv3dNormActReduced`'s `(17, 1, 1)`, also 13 / 9 / 3 in Fast-FS) as 17
1 x 1 `conv2d`s plus partial-plane adds. Now a `(kd, 1, 1)` kernel with stride 1 / no padding in height and width
runs as one `(kd, 1)` `conv2d` over the contiguous `[N, C, D, H * W]` (`_depth_conv`). Microbenchmark, C -> C
`(17, 1, 1)` at C-FS's hourglass levels, taps / new ms (input layout copy included):

| level | fp16 | fp32 |
| --- | --- | --- |
| 28 @ 104 x 256 x 376 | 402 / 278 | 544 / 321 |
| 56 @ 52 x 128 x 188 | 151 / 108 | 163 / 151 |
| 112 @ 26 x 64 x 94 | 34 / 27 | 39 / 81 |
| 168 @ 13 x 32 x 47 | 11.4 / 12.6 | 12.0 / 13.3 |

Gates from this: more than 4 taps (IGEV++'s `(2, 1, 1)` / `(4, 1, 1)` stride-2 patch convs: taps 0.8 vs 1.5 ms, kept
on taps) and at most 256 bytes of channels per position (fp16 C <= 128, fp32 C <= 64; fp32 96 / 112 / 128 lose 1.1-2x,
reproducible). Tried, not kept: the same `conv2d` on the channels-last-3d view (no copy, but 1.2-3x slower than the
contiguous one at >= 56 channels: MPS's channels-last `(kd, 1)` conv is slow) and a `(1, kd)` conv over the
transposed `[N, C, H * W, D]` (as slow).

Survey, MPS portable, second run, ms median (min) / MB peak:

| ref | dtype | before | after |
| --- | --- | --- | --- |
| c-foundationstereo-s | fp16 | 25783 / 12573 | 23526 (23040) / 12489 |
| c-foundationstereo-s | fp32 | OOM | OOM (same 1.06 GiB request) |
| c-fast-foundationstereo | fp16 | 2539 / 8483 | 2472 (2470) / 8460 |
| c-fast-foundationstereo | fp32 | 3184 (2978) / 13079 | 3326 (2967) / 13001 (pageout-noisy: min equal) |
| fast-foundationstereo | fp16 | 2715 / 8478 | 2477 (2475) / 8462 |
| fast-foundationstereo | fp32 | 3209 (3093) / 13359 | 3471 (3035) / 13035 (pageout-noisy) |
| foundationstereo-vit-s | fp16 | 19700 (18671) / 13239 | 19059 (18554) / 13094 |
| foundationstereo-vit-s | fp32 | OOM (2026-11-06) | 22956 (22452) / 13425 (the old row predates other passes) |

Goldens pass (test_mps, FASTFORKS_MPS_FULL=1): c-foundationstereo-s, c-fast-foundationstereo, fast-foundationstereo,
foundationstereo-vit-s, foundationstereo-vit-l (all @1; a counting wrapper confirmed the new path runs in them, fp16
and fp32). test_mps_conv_cpu gains two depth-only cases. No route change (`gen_auto_mode --check` clean).

## 2026-11-22 (evening): MLX kernel == stride ConvTranspose2d as one GEMM; the CLIP-L / EoMT-L fp32 gap (M1 Pro 16 GB)

`mx.conv_transpose2d` is slow for a transposed conv whose kernel equals its stride (no overlap): `_conv_core` now
runs it as `[N*H*W, Cin] @ [Cin, kh*kw*Cout]` (bias tiled) plus a depth-to-space transpose
(`_conv_transpose2d_gemm`), bit-identical to the old path. EoMT-L's 2 x 2 upsampler, 1024 -> 1024 channels, fp32:
3 x 32 x 32 16.1 -> 6.9 ms, 3 x 64 x 64 44.0 -> 26.1 ms (fp16 alike). Same layer shape in DPT / SAM / Sapiens /
MoGe / Depth Pro necks and decoders.

Survey, MLX portable, second run, ms median (min) / MB / compile s:

| ref | dtype | before | after |
| --- | --- | --- | --- |
| eomt-l-dinov2-ade20k | fp32 | 681.7 (657.8) / 4504 / 2.34 | 666.6 (642.8) / 4518 / 2.32 |
| eomt-l-dinov2-ade20k | fp16 | 582.9 (576.6) / 3663 / 2.26 | 574.1 (558.5) / 3661 / 2.25 |

MPS fp32 is 608 ms: EoMT-L's gap shrinks from ~12 % to ~10 %. Goldens pass (test_mlx): eomt-l-dinov2-ade20k,
nafnet-gopro, wd-vit-tagger-v3, c-fast-foundationstereo, tapnext, cotracker3-online, alltracker, loma-b, memfof,
depthsplat-small-re10k, zoedepth-nk, depth-anything-v2-s, sam-vit-b, sapiens-seg-0.3b-goliath, moge-2-vits-normal,
depth-pro (all @1). New test_mlx case for the lowering. No route change.

The rest of the fp32 gap is GEMM speed, not graph overhead. EoMT-L's eager profile (FF_MLX_PROFILE=1): addmm 399 of
673 ms, SDPA 98, GELU 46, convs 35. CLIP-L (MLX 62.9 vs MPS 60.6 ms): GEMMs 73 of 130 ms eager, nothing
family-specific. Microbench, fp32 `[M, K] @ [K, N]`, MLX vs MPS: at EoMT's M = 2058, MLX is ~5 % slower (K 1024 N 4096:
4.56 vs 4.35 ms); padded to M = 2112 it is 4.29. Not kept: per-block padding would gain ~1 % end to end (a pad and a
slice per MLP, the earlier padding reject stands). CLIP's M = 257..320 shows no padding win at all.

## 2026-11-22 (evening): MVSplat Metal plane sweep on MPS (`mvsplat_metal`, kernels mode, M1 Pro 16 GB)

Portable MPS runs the plane sweep as `grid_sample` of 32 depths at a time plus a channel sum
(`[V, C, 32, h, w]` warped chunks). `kernels/mvsplat_metal` ports the MLX `warp_dot` kernel to `torch.mps`: one SIMD
group per (view, depth, pixel), 4-channel slices of the four bilinear taps on NHWC features, fp32 products, one
`simd_sum`; the other view is indexed (no `roll` copy), offset 1 writes the volume, later offsets add (the CUDA
`mvsplat_corr.accumulate` contract). `correlation(fused=...)` now takes the accumulate function (CUDA:
`cuda_accumulate`, the grid in the features' dtype as before). New `kernels_mps` runner, card `mps_kernels`,
`mps_dtype = "float32"` (portable fp32 beats fp16 on MPS here; the kernel costs the same in both).

Correlation alone, C = 128, D = 128, ms min (loop -> Metal): 2 views 64 x 64 fp32 25.9 -> 10.3, fp16 18.2 -> 10.3;
3 views 64 x 64 fp32 79.2 -> 28.2, fp16 52.6 -> 28.1. Max error against a float64 reference: equal in fp32
(1.4e-5); fp16 1.3e-5 instead of 6e-2 (the loop samples with an fp16 grid).

Survey, MPS, second run, ms median (min) / MB peak / first call s:

| ref | mode / dtype | ms | MB |
| --- | --- | --- | --- |
| mvsplat-re10k | portable fp32 | 1039.6 (1039.0) | 4532 |
| mvsplat-re10k | portable fp16 | 1110.0 (1108.5) | 3535 |
| mvsplat-re10k | kernels fp32 (new) | 896.3 (881.4) | 4293 |

-14 % time and -239 MB against portable fp32. Route change: `mps_auto` for mvsplat-re10k@1 now picks the Metal
kernels in float32. Goldens pass: test_mps[mvsplat-re10k@1] (FASTFORKS_MPS_FULL=1, portable fp32 / fp16 and kernels
fp32), test_mlx[mvsplat-re10k@1]. New test: `test_metal_plane_sweep` (2 and 3 views, fp32 / fp16, against float64).
MLX stays much slower than MPS here (portable fp32 1573 ms, fp16 1259 ms): not looked at in this pass.

## 2026-11-22 (night): CoTracker3 on MLX and MPS (M1 Pro 16 GB, 24 frames at 480 x 854, float32)

Start: 17 GB free disk, 7.5 GB swap; end: 30 GB free, 6.9 GB swap.

Where online MLX lost (2314 ms against MPS 1731): not in the refinement (MLX 1114 ms, MPS ~1430), but in the feature CNN
(MLX 887 ms over three 8-frame chunks, MPS ~460). The convs alone (`tools/mlx_conv_bench.py`, per 8-frame chunk) are
close to MPS (MLX 109 ms portable, MPS 89). The rest were copies and fresh allocations:

- **`aten.instance_norm` lowering** (`shared/mlx_graph.py`, kept out of the decomposition): ATen reshapes
  `[N, C, H, W]` to `[1, N * C, H, W]` for a batch norm and back, two transposing copies of the channels-last map per norm
  when N > 1. Now per-(sample, channel) block sums over the `[N, P, C]` view and one fused pass (also a CL op). float32
  only: in float16 the other summation order moves single ulps, and flowformer-small-sintel (p99 0.0153 against 0.013)
  and alltracker-tiny failed their goldens. N = 1 and kernels mode keep the batch-norm path. `FF_MLX_NO_INORM=1`
  restores the decomposition (sweep switch). cotracker3-online: fnet 887 -> 739 ms; final setup 1702 -> 1588 ms
  without / with.
- **Channels-last bilinear** (`_bilinear_cl`, new `graph_mlx.ops.bilinear_nhwc` kernel, exact against the NCHW one): the
  pyramid upsamples of conv outputs stay NHWC, so the 416-channel cat is `_cat_cl` and conv2 reads no transposed copy.
  Neutral on its own here (eager conv2 stays ~46 ms, the microbench's 22 ms is a warm buffer); kept because it saves a
  163 MB copy.
- **Frame chunks + buffer cache** (the main gain): the 8-frame CNN transients (`[8, 64, 192, 256]` float32, 100 MB each)
  are larger than what the 256 MB cache recycles, so every one is a fresh, zero-filled Metal allocation. The CNN now runs
  `FNET_FRAMES = 2` frames per call (`_chunk_encode`, online and offline, results identical: per-sample norms), and the
  cache is 1 GB online, 1.5 GB offline. Sweep, bench median ms / peak MB:

| setup | online | offline |
| --- | --- | --- |
| 8 / 16 frames, cache 256 MB / default (before) | 1926 / 2538 | 2417 / 3355 |
| 8 / 16 frames, cache 1 GB | 1720 / 3307 | 1654 / 3174 (8 fr.) |
| 8 frames, cache 2 / 3 GB | 1597 / 4070, 1509 / 4077 | |
| 4 frames, cache 256 MB / 1 GB | 1863 / 2178, 1631 / 2864 | 1592 / 2701 (1 GB) |
| 2 frames, cache 256 MB / default | 1775 / 1904 | 2411 / 1763 |
| 2 frames, cache 768 MB | 1616 / 2422 | |
| **2 frames, cache 1 GB** (online, kept) | **1588 / 2649** | 1541 / 2709 |
| **2 frames, cache 1.5 GB** (offline, kept) | 1516 / 2847 | **1455 / 3201** |
| 2 frames, cache 2 GB | | 1436 / 3722 |
| 1 frame, cache 256 MB | 1809 / 1915 | |

- **Transfer to MPS**: the same 2-frame chunks on portable MPS halve the peak at equal speed (`Portable.__init__` on
  `mps`; 4 frames: online 1700 / 2704, offline 1426 / 3829).

Survey, second run, ms median (min) / MB peak / first call s:

| ref | device | before | after |
| --- | --- | --- | --- |
| cotracker3-online | MLX | 2233 / 2747 / 4.59 | 1590.2 (1584.9) / 2641 / 4.36 |
| cotracker3-offline | MLX | 3577 / 3438 / 12.97 | 1446.3 (1442.0) / 3213 / 7.37 |
| cotracker3-online | MPS | 1731 (1698) / 4767 / 1.83 | 1704.1 (1702.8) / 2748 / 1.84 |
| cotracker3-offline | MPS | 1430 (1430) / 4725 / 2.07 | 1428.1 (1427.1) / 2751 / 1.50 |

Route change (`apple_mlx.json`): cotracker3-online@1 gains `auto` / `dtypes` (fp32 portable on MLX: 1590 against MPS
1704 ms, first call 4.4 s < 3 x 1.7 s). cotracker3-offline stays on MPS (1446 vs 1428 ms).

Investigated, not kept:

- **`channel_stats` for the instance norm**: 4.6 against 4.9 ms per `[8, 64, 192, 256]` norm (one-pass kernel vs block
  sums), mx.mean / mx.var 6.7 ms. Not worth a kernels-only path.
- **Bigger caches online** (2-3 GB): 1509-1597 ms but +1.4 GB peak.

Goldens pass (test_mlx): cotracker3-online/offline, nafnet-gopro, wd-vit-tagger-v3, tapnext, alltracker(-tiny), loma-b,
memfof, depthsplat-small-re10k, zoedepth, eomt-l-dinov2-ade20k, raft / sea-raft, locotrack, tapir, track-on, flowformer,
mvsplat, monster, nninteractive, nnunet refs (51 cases); test_mps cotracker3-online/offline (FASTFORKS_MPS_FULL=1).
New test: `test_channels_last_instance_norm_and_bilinear_match_torch`.

## 2026-11-22 (night): MVSplat on MLX, conv padding into Winograd and a kernels mode (M1 Pro 16 GB)

Profile (fp32 portable, 8 views 336x518, 1512 ms / 3463 MB): the to_gaussians 3x3 conv 163 -> 168 at 8 x 160x256
was 238 ms. `_conv2d` padded it to 192 -> 176, and MLX's implicit GEMM is pathological at output counts that are an
odd multiple of 16 (in-graph 275 ms against 70 ms for 192 -> 192). The OUT64 rule refused 192 because it lands in
MLX's Winograd.

- **`_conv2d`: odd multiples of 16 pad 16 more into MLX's Winograd** (float32 or kernels mode, `FF_MLX_NO_WINO32=1`
  switches it off). Micro-benchmarks (fp32 3x3): 8 x 96x128 256 -> 112 17.9 -> 9.9 ms, 256x376 256 -> 208
  122.9 -> 14.8, mvsplat 192 -> 176 85-275 -> 43-70; fp16 192 -> 176 107 -> 34. Multiples of 32 stay excluded
  (256 -> 96 into 128 was slower, see the OUT64 notes). Portable half precision still never enters MLX's Winograd.
- **`_conv2d`: batched MLX Winograd convs run in chunks** when their transforms (about 4 x (Cin + Cout) per pixel)
  exceed 256 MB: chunks of `FF_MLX_WINO_CHUNK_MB` (128) sequenced with `mx.depends`. The result is the same, per
  sample. The Winograd conv alone added about 500 MB of peak at batch 8; chunked, fp32 peak 3902 -> 3054 MB and
  1433 -> 1385 ms.
- **mvsplat `kernels` on MLX** (`mlx_kernels = ["conv_mlx"]`, `kernels_fp16_only`): F(4,3) Winograd for the
  even-channel 3x3s in float16; float32 kernels runs as portable.

Survey (second run, ms / peak MB / first call s):

| ref | mode | before | after | MPS |
| --- | --- | --- | --- | --- |
| mvsplat-re10k | portable fp32 | 1573 / 3456 / 6.3 | 1362.8 / 3098 / 6.5 | kernels fp32 896 / 4293 |
| mvsplat-re10k | portable fp16 | 1258 / 2541 / 5.8 | 1144.6 / 2342 / 5.0 | |
| mvsplat-re10k | kernels fp16 (new) | | 972.6 / 2848 / 4.7 | |

Route: `mlx_default` and `mlx.float16` -> kernels. MLX is still 1.09x slower than MPS kernels fp32 (at 34 % less
memory), so no auto route.

Rejected:

- **MLX sdpa as matmul + softmax in float32**: 21.5 -> 16 ms for mvsplat's (8, 4, 640, 4480, 128). It is slower on
  most other shapes (ViT 577 / 4096 tokens, 1024 x 8192: up to 1.5x), so not a general rule, and not worth a
  model-specific switch.
- **Bigger buffer cache** (1-2 GB): 1394-1442 ms fp32 but +900-1900 MB peak.
- **fp32 kernels**: 1430-1621 ms, no faster than portable, so it runs portable.
- **168 -> 84 conv** (about 70 ms): every padding variant is within 10 %.

Goldens pass: test_mlx[mvsplat-re10k@1] (portable fp32 / fp16, kernels fp32 / fp16), plus 51 test_mlx cases for
nafnet-gopro, wd-vit-tagger-v3, tapnext*, cotracker3-online, alltracker*, loma-b, memfof, depthsplat-small-re10k,
zoedepth, eomt-l-dinov2-ade20k, raft-stereo*, hat*, monster*, igev*, flowformer* (shared conv change). New test:
`test_conv_padded_into_winograd_matches_torch` (whole and per-sample chunks).

### 2026-11-22 (night): memfof / monster follow-up

memfof MLX re-surveyed after the odd-16 Winograd padding change (second run, ms / peak MB / first call s):
fp32 3683.7 / 2964 / 8.1 → 3645.7 / 2980 / 7.3; fp16 3380.6 / 2417 / 7.7 → 3278.9 / 2380 / 7.3. Small gain
from the shared conv change, no memfof-specific code; goldens pass in both dtypes (fp16 passed in this survey).
No route change. Monster(++) not investigated (time box spent on cotracker3 and mvsplat).

### 2026-11-22 (evening): IGEV / MonSter family lookup kernels

Two new custom ops, each one Metal kernel in the MLX replay (eager / MPS paths unchanged):

- `fastforks::linear_sample` (`_core/volume_ops.linear_sample`): the 1-D geometry / correlation lookup of the
  `FastGeoEncodingVolume` in igev-stereo, igev++, selective-igev, great-igev, mocha-stereo, monster, monster++.
  Compiled as a gather chain it took 9.9 ms per lookup; the kernel takes 1.25 ms.
- `fastforks::gwc_warp` (`gwc_warp`): RT-MonSter++'s `build_gwc_volume_selective` (grid_sample of the right
  features at per-pixel disparity samples, group-wise mean with the left). It no longer builds the
  `[B, C, D, H, W]` warped volume.

MLX survey (second run, ms / peak MB / first call s):

| ref | mode | before | after |
| --- | --- | --- | --- |
| igev-stereo | fp16 | 3951 | 3641 |
| igev-stereo | fp32 | 5702 / first 12.6 | 4609 / first 9.2 |
| monster | fp16 | 10411 / 3295 / 21.3 | 10064 / 2941 / 18.4 |
| monster | fp32 | 13236 / 6569 | 13099 / 5829 |
| monster++ | fp16 | 10966 / 3294 / 21.9 | 10647 (min 10479) / 3033 / 18.9 |
| monster++ | fp32 | 13947 / 6575 | 13670 / 5919 |
| rt-monster++ | kernels fp16 | 580.1 / 2124 / 4.4 | 510.8 / 1591 / 5.1 |
| rt-monster++ | kernels fp32 | 945.7 / 2308 | 852.1 / 2308 |
| rt-monster++ | portable fp16 | 666.4 / 2055 | 605.1 / 1453 |
| rt-monster++ | portable fp32 | 993.9 / 2224 | 890.8 / 2154 |

Routes are unchanged. MPS rt-monster++ portable fp16 runs at 612 ms / 4076 MB, so MLX kernels fp16 is 17 % faster
at 39 % of the memory, but its first call (5.1 s) is above 3x the MPS runtime, so it gets no auto route. The MPS
memory of monster(++) was not addressed. New tests: `test_linear_sample_kernel_matches_torch` (fp32 / fp16) and
`test_gwc_warp_kernel_matches_torch`. Goldens pass: test_mlx for the seven geometry families and rt-monster++.

### 2026-11-22 (evening): fp16 retry of fp32-only MLX cards with `half_wino_f32`

Cards whose MLX `mlx_dtypes` lacked float16 got one cheap attempt: add float16 and port with
`half_wino_f32=True` (fp16 Winograd convs computed in fp32), then run test_mlx in every mode and dtype.

Kept (MLX, second run, ms / peak MB / first call s):

| ref | dtype | before | after | MPS best |
| --- | --- | --- | --- | --- |
| hed | portable fp32 | 81.9 / 1144 / 1.26 | 74.9 / 1089 / 0.98 | fp16 56.9 / 1381 |
| hed | portable fp16 (new) | | 65.2 / 1174 / 0.96 | |
| nafnet-gopro | portable fp32 | 648.9 / 1745 / 8.8 | 524.3 / 1409 / 3.6 | fp16 368.3 / 1715 |
| nafnet-gopro | kernels fp16 (new) | | 358.2 / 1124 / 3.3 | |

The fp32 gains come from earlier shared changes (Winograd, conv padding), not from this item. `mlx_default` moves
to float16 for hed (portable) and nafnet-gopro (kernels). There is no auto route: hed MLX is slower than MPS, and
nafnet MLX fp16 is only 2.7 % faster than MPS fp16 (needs more than 5 %) with a 3.3 s first call. nafnet-sidd
shares the card and passes the fp16 golden as well.

Rejected (goldens fail in fp16 even with `half_wino_f32`; cards stay fp32-only):

- gfpgan-v1.4: image mean diff 0.157.
- rtmo: rtmo-t scores p99 0.0049 (s / m / l pass, but the card is shared).
- tapir: tracks mean 0.0098; bootstapir 0.0092.
- locotrack: s tracks mean 0.53, b 0.116.
- cfnet-sceneflow: disparity mean 1.20.
- xfeat and xfeat-lighterglue: keypoint recall 0.9952.
- loftr-indoor and loftr-outdoor pass the golden, but fp16 is slower (2224 vs 1839 ms fp32), so the card stays
  fp32. Side note: loftr-outdoor fp32 now measures 1839 ms / 4863 MB (11-19 row: 2017 / 3685). The peak growth
  was not investigated; the old row is kept.

Not tried (earlier notes still hold):

- fast-acvnet: a top-k flip on a decision boundary.
- mobile-sam: mean diff 0.49.
- neuflow and recover: fp16 is slower.
- rfdetr: some seg variants fail.
- coatnet, fastvit, wd-tagger: they use the libnet wrapper.
- cotracker, lightglue, ben2, efficientvit-sam: fp32-only on MPS too.

### 2026-11-22 (evening): MPS frame chunks for the tracker feature passes (cotracker transfer)

Cotracker's 2-frame feature-CNN chunks halved its MPS peak, so the other trackers got an MPS-only
`MPS_FRAME_CHUNK`. MLX and CUDA keep `FRAME_CHUNK`, so the exported MLX graphs do not change. Each A/B is two
survey runs on the same tree (second run, MPS portable, ms / peak MB):

| ref | dtype | old chunk | new chunk |
| --- | --- | --- | --- |
| alltracker | fp32 | 16: 3326.9 / 4912 | 2: 3218.1 / 4445 |
| alltracker | fp16 | 16: 3042.2 / 3806 | 2: 3066.8 / 3782 |
| locotrack-b | fp32 | 16: 1806.6 / 2482 | 4: 1786.5 / 1470 |
| locotrack-b | fp16 | 16: 2076.2 / 2487 | 4: 2084.4 / 1483 |
| track-on2 | fp32 | 8: 8266.5 / 4925 | 4: 8370.1 / 3823 |

Chunk sizes tried:

- locotrack: 2 gives the same memory as 4 (1497 / 1520 MB) at 2 % more time in fp16.
- track-on: 2 gives 8579 ms / 3828 MB, so 4 keeps the memory saving at less cost.
- alltracker: 4 gives 3238 / 5032 (fp32) and 3045 / 3787 (fp16).

Route change: locotrack-b `auto` drops MLX. MLX fp32 (1482 ms / 1738 MB) is still 17 % faster, but it now uses
more memory than MPS (1470 MB).

Rejected: **tapir** chunk 2. 1365 / 1651 (fp32) and 1382 / 1639 (fp16) against 1356 / 1604 and 1393 / 1610: no
memory saving, because its peak is not in the feature pass.

The other two tricks found no MPS sibling worth a try this window. The mvsplat plane-sweep Metal kernel is
mvsplat-specific, and the geometry lookup kernel above is MLX-only. On MPS that lookup is already the two-gather
eager path.

Goldens pass: test_mps for alltracker-tiny, locotrack-b and track-on-r, plus the survey goldens for alltracker,
locotrack-b and track-on2. tests/test_mps*.py pass.
