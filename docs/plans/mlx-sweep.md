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
