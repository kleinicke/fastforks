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
