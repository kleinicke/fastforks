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
