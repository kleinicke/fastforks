# Torch-free MLX runs

2026-11-27. Goal: a Mac install that runs MLX without PyTorch. Step 1 (this plan, done for the first family):
torch stays a dependency, but an MLX run on a warm conversion cache never imports it and can return NumPy arrays.
Step 2 (later): torch an optional extra on macOS, NumPy results the default without it, a torch-free `.pth` reader.

## Design (step 1)

- **Lazy torch** (`fastforks._core.lazy`): the modules on the path `fastforks.<task>` / `load` -> mode selection ->
  MLX port -> result refer to torch through `lazy("torch")` (a module stand-in that imports torch on first attribute
  use) under `if TYPE_CHECKING: import torch`. Torch modes pay the import where they used to (at load); module-level
  torch constants became dtype names, deferred class factories (`Forward`, result `nn.Module`s made on first use via a
  module `__getattr__`) or string aliases. `after_import("torch", step)` runs the `add_safe_globals` registrations of
  the result types when someone imports torch. Packages whose `__init__` re-exported torch modules (e.g.
  `fastforks.shared.depth_anything_v2`) are PEP 562 lazy: the torch half moved to `_torch.py`, the MLX half imports
  without it.
- **Array runners** (`Runner.arrays = "numpy"`): a runner that computes on NumPy arrays. Its load runs with
  `Context.device = fastforks._core.arrays.HOST`: no torch device, no `deferred_init` / default-dtype guard, no MLX
  plan cache (`mlx_plan_cache` is for exported graphs), and `Model._arrays` set. The task API then decodes inputs
  with `to_rgb_batch(image, HOST)` (float32 `[B, 3, H, W]` NumPy; torch tensors are accepted when torch is in
  `sys.modules`, never imported to check), and the shared helpers (`_check_size`, `_batched` via `arrays.cat`,
  `_check_outputs`, the result types) work on either container.
- **Outputs** (`fastforks._core.arrays`): `model(..., outputs="numpy" | "torch")` per call,
  `fastforks.set_outputs(...)` / `get_outputs()` process-wide, `FASTFORKS_OUTPUTS` at start-up. Every result type
  (`MapOutputs`, `NumpyOutputs`, instances, boxes, tracks, ...) converts field by field with `arrays.convert`; NumPy
  from an MLX array is `np.asarray(mx)` (zero copy). The default is `outputs_default()`, which returns `"torch"`;
  step 2 changes its one line to `"numpy" if imported/installed torch is missing else "torch"`.
  An array runner with torch outputs gives CPU tensors via `torch.from_numpy` (zero copy) or MPS tensors where the
  MLX port stands in for `device="mps"`, as before.
- **Bit-exact torch maths without torch** (`fastforks.shared.mlx_resize`): the ports that resized with torch's CPU
  `F.interpolate` around the MLX network now run the same arithmetic in Metal kernels (`metal::precise::fma`, taps
  and weights computed on the host in float32 as torch does): bicubic (`align_corners=False`, optionally fused with
  the mean / std normalisation) and bilinear `align_corners=True` (torch's generic kernel, and its channels-last
  kernel for outputs with `OH + OW <= 128`). `bicubic_host` covers tables (DINOv2's position embedding by scale
  factor). Checked bit for bit against torch on random inputs (C = 1, 3, 4, 5; up- and down-sampling; tables) and on
  the DAv2 input sizes. The cubic weights follow torch's rounding step by step: the fourth weight takes `1 - t`, then
  adds 1 (computing `2 - t` directly differs in the last bit for some `t`, which changed DAv2 inputs at 320 x 240).
  Not covered: torch's single-thread 3-channel bilinear case (`torch.get_num_threads() == 1`).
- **Guards** (`tests/test_torch_free.py`): a meta-path finder that fails and records every torch / torchvision /
  triton import (so an `except ImportError` cannot hide one). Fast, no GPU: importing `fastforks` and every task
  package; the load API (`set_outputs`, `list_models`, `model_card`); each registered ref's model module, whose MLX
  runners must all be array runners. GPU (`-m mlx`): each registered ref per MLX mode and dtype, warm cache, NumPy
  outputs, in a subprocess that cannot import torch, against its golden (`test_golden._check`). The registry
  `TORCH_FREE` (task -> refs) grows as families are ported.

## Families

Torch-free (warm cache, NumPy or torch outputs):

| Family | Refs | MLX modes | Notes |
|---|---|---|---|
| Depth Anything V2 (`depth/models/depth_anything_v2`) | all 12 variants (relative, metric Hypersim / VKITTI, Distill-Any-Depth) | portable, kernels | input bicubic + normalisation and output bilinear in `mlx_resize`, bit-identical to the torch-CPU resizes it replaced |

Not yet, by kind of MLX port (`fastforks.shared.mlx_port`, `mlx_wrap`):

- **`converted` hand ports** (the weights come from the conversion cache without torch; what remains is task-side
  torch pre/post-processing): Depth Anything 3, Depth Pro, R-CNN detection, DINOv2 / DINOv3 / C-RADIO / TIPSv2
  features (the features task resizes and normalises in torch before the runner, and `dinov2/mlx_net` imports the
  torch `model.py` for its position table: `mlx_vit.dinov2_table` already has the NumPy path), RIFE / EMA-VFI
  interpolation (torch padding and flow warps around the network), SAM 3 (`mlx_net`, `mlx_tracker`: prompt encoding
  and mask post-processing in torch), CREStereo (stereo padding / disparity post-processing), libnet `mlx_features` /
  `mlx_vit_cls` users, volume (`_mlx`). Each needs its pre/post-processing in NumPy or MLX (as `mlx_resize` did for
  DAv2), `arrays = "numpy"` on its MLX runners, a lazy-torch model module, and a ref in `TORCH_FREE`. These are the
  next step-1 candidates, most-used first: features DINOv2 (shared with many depth ports), SAM 3, Depth Pro / DA3,
  RIFE.
- **`recorded` hand ports** (DETR, OWLv2, RTMDet, YOLOX, Lineart, FlowSeek, MemFlow, MemFOF, NeuFlow v2, ReCoVEr,
  HRNet, RetinaFace, ViTPose, YuNet, SAM 2, the restoration ports Compact / DAT / DRCT / HAT / Real-ESRGAN /
  Restormer / SPAN / Swin2SR / SwinIR, EoMT, libnet `mlx_cnn`): on a cache hit `mlx_port.recorded` still builds the
  torch network on the meta device to replay the builder's tape, so torch is imported. Needs the builders to build
  from the stored `config` JSON alone (no torch skeleton), then the same task-side work as above.
- **`mlx_graph` / `mlx_wrap.port` graph runners** (151 model modules, plus every `GRAPH_RUNNERS` entry rebuilt under
  `FASTFORKS_MLX_GRAPH=1`): the network is an exported torch graph (`torch.export`, run op by op in MLX), torch is
  inherent. They become torch-free only by being replaced by hand ports (`docs/plans/mlx-coverage-gaps.md`).
- **Real-PLKSR**: builds its MLX network from the torch network at every load (no conversion cache yet).

## Numbers

Start-up to first result (a fresh process: `import fastforks`, `load(device="mlx")` on a warm cache, one 320 x 240
call) and peak RSS, Apple M1 Pro 16 GB, default mode (kernels, float16), median of 3:

| Ref | before (0fc5353d^): torch outputs | after: NumPy outputs, no torch | after: torch outputs | difference (NumPy vs before) |
|---|---|---|---|---|
| `depth-anything-v2-s@1` | 0.99 s, 376 MB | 0.40 s, 230 MB | 0.98 s, 351 MB | -0.59 s, -146 MB |
| `depth-anything-v2-metric-hypersim-s@1` | 1.01 s, 375 MB | 0.39 s, 230 MB | 0.97 s, 354 MB | -0.61 s, -145 MB |
| `depth-anything-v2-b@1` | 1.12 s, 625 MB | 0.55 s, 471 MB | 1.15 s, 472 MB | -0.57 s, -154 MB |
| `depth-anything-v2-l@1` | 1.55 s, 1541 MB | 1.01 s, 1390 MB | 1.62 s, 1389 MB | -0.54 s, -151 MB |

The saving is torch's import (about 0.6 s with a warm OS file cache, more from a cold one) and its resident memory
(about 150 MB), below the 1.2 s / 200 MB first estimated (measured here with a warm file cache). With torch outputs the start-up is
unchanged; RSS is lower for B / L because the input and output resizes no longer allocate torch CPU buffers.
Warm calls are unchanged (5-call means, ms, before / after NumPy / after torch: S 66 / 64 / 68, B 168 / 165 / 167,
L 494 / 486 / 494; the same MLX engine, NumPy results are a zero-copy `np.asarray`).

## Step 2: what it needs

1. `outputs_default()`: `"numpy"` when torch is not installed (one line in `fastforks._core.arrays`).
2. A torch-free `.pth` reader (zip + pickle of storages, restricted unpickler; `safetensors` checkpoints already need
   no torch) so a cold conversion needs no torch either, and the conversion code (`prepare` functions) on NumPy.
3. `recorded` ports built from the cached `config` alone (no meta-device torch skeleton).
4. Task-side pre/post-processing in NumPy / MLX for the families above (resizes, padding, prompt encoding, box and
   mask post-processing, flow warps), with bit-exact kernels where torch's arithmetic must be matched.
5. Packaging: torch as an extra on macOS (`fastforks[torch]`), a clear error naming the extra when a torch mode or a
   non-ported family is loaded without it (never a silent fallback), and the guard tests run in an environment
   without torch installed.
6. The graph runners stay torch-only until replaced by hand ports.
