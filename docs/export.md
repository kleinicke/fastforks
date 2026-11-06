# Export: torch.compile, torch.export, ONNX, TensorRT

Every task exports the same way. Everything starts from `model.module()`, the traceable part of the portable model as
an `nn.Module` (what it covers per task, and what stays in `model(...)`: the table in
[API](api/index.md#what-modulemodule-covers)); `model.export_onnx`, `model.export_tensorrt` and `fastforks export` trace
it at one fixed size and batch.

```python
# needs: pip install "fastforks-core[onnx]"
from fastforks import depth

model = depth.load("depth-anything-v2-s", mode="portable")
model.size_range()  # min size, pad multiple, largest size tested (see below)
model.export_onnx("dav2s.onnx", 518, 770)  # stereo / depth / segmentation need a size: export_onnx(path, H, W)
model.export_tensorrt("dav2s.engine", 518, 770)  # ONNX -> TensorRT engine for this GPU
engine = depth.TensorRTEngine("dav2s.engine")
inverse_depth = engine(*model.example_inputs(518, 770))
```

```
# needs: pip install "fastforks-core[nc]" "fastforks-core[onnx]" "fastforks-core[tensorrt]"
# S2M2's code and weights are non-commercial: its weights download only with FASTFORKS_NONCOMMERCIAL=1
fastforks export depth-anything-v2-s --format onnx --size 518x770  # float32 graph at this fixed size
fastforks export s2m2-s --format trt --size 384x1248 --fp16       # the model's tested precision (fp16 on CUDA), engine + check
```

`pip install "fastforks-core[onnx]"` installs the ONNX exporter, `"fastforks-core[tensorrt]"` TensorRT. Without `-o FILE` the file is `MODEL_HxW_dtype.onnx` (or `.engine`) in the current directory.

## Inputs, outputs and size range per task

| task | inputs (`[B, 3, H, W]` float32) | outputs (ONNX / TensorRT names) | size range / default |
|---|---|---|---|
| stereo | `left`, `right`: RGB in `[0, 1]` | `disparity` `[B, H, W]` | any H x W >= the model's minimum, padded to its multiple (per model: `model.size_range()`); no default, `--size` is required |
| depth | `image`: RGB in `[0, 1]` | the model's main output (`depth`, `inverse_depth`, ...) at the input size | any H x W >= the minimum; the net resizes or pads inside; no default, `--size` is required |
| classification | `image`: the transformed crop (resized, cropped, normalised; see `model.data_config`) | `logits` `[B, classes]` | fixed: the model's crop, e.g. 224 x 224 (the default; 224 is the eval crop, `--size` takes another fixed size) |
| segmentation | `image`: RGB in `[0, 1]` | the model's main output | per model, see `model.size_range()`; no default, `--size` is required |
| promptable segmentation | `image`: RGB in `[0, 1]` (image encoder only) | SAM / SAM 2: `image_embeddings` `[1, 256, 64, 64]`; SAM 3: the FPN levels `fpn0`, `fpn1`, `fpn2` | any H x W, resized inside to the fixed encoder input (SAM / SAM 2: 1024, SAM 3: 1008), which is also the default export size |
| detection | `image`: RGB in `[0, 1]` at the network size (normalised inside; R-CNN and DETR also pad inside, RT-DETRv2 takes multiples of 32 only, `InputError` otherwise) | R-CNN: the feature pyramid (backbone + FPN) `p2` .. `p6`; DETR / RT-DETR: `logits`, `boxes` | any H x W >= the minimum (RT-DETRv2: multiples of 32; COCO inference uses shortest edge 800, e.g. 800x1344); no default, `--size` is required |
| features | `image`: RGB in `[0, 1]` (resized inside to a multiple of the patch stride) | `last_hidden_state` `[B, N, C]` | any H x W; default 224 x 224 |
| multi-view depth | `images` `[V, 3, H, W]`: the V views of one scene, RGB in `[0, 1]`; then the cameras the model requires (`intrinsics` `[V, 3, 3]`, `cam_to_world` `[V, 4, 4]`, `depth_range` `[2]`) | the model's main output (`depth`, ...); a model returning several outputs keeps their names (PatchmatchNet: `depth`, `confidence`) | one fixed size per export (no size range); no default, `--size` is required |

`model.example_inputs(H, W)` gives synthetic inputs of exactly this form (what the export traces and checks with);
`model.export_size()` is the default size where the task has one (classification: the crop; features: 224; promptable
segmentation: the encoder side) and `None` otherwise; stereo, depth, segmentation, detection and multi-view depth have no default size: `export_onnx`, `export_tensorrt` and `fastforks export` without a size raise
`InputError` naming the model's range and an example (`--size 704x1280`). `model.size_range()` returns a `SizeRange`
(`min_size`: the smallest size that runs, from the model definition and a probe of every family; `multiple`; `native`; `encoder`;
`tested_max` with its peak memory); `fastforks info MODEL` and the model pages under
[models](models/stereo.md) print the same range, and the largest size we measured (portable
fp16 on one RTX 5060 Ti 16 GB; "not measured" where no size run exists). Resizing to a family's processing size, padding and cropping are inside
`module()`, so a depth engine built at 518x770 takes a 518 x 770 image and returns maps at 518 x 770.

Rules for every task:

- Only the portable backend exports. `kernels` raises `BackendUnavailableError` from `export_onnx` / `export_tensorrt`
  (ONNX cannot hold our kernels); load with `mode="portable"`. There is no silent fallback.
- An export has one fixed size and batch (`batch=` argument). A size below the model's minimum raises `InputError` (e.g. CoEx with `--size 4x4`: below 32).
- The graph keeps the model's compute dtype: `dtype="float32"` exports float32 (closest to the original reference); the
  default keeps the model's tested fp16 casts, and TensorRT builds it strongly typed, in exactly that precision.
  The CLI exports float32 unless `--fp16` is given.
- Graphs above 2 GB keep their weights in `<file>.data` next to the ONNX file.
- **An engine only runs on the GPU model and TensorRT version it was built with: build it on the target machine.**
  `TensorRTEngine(path)` checks the count, dtype, shape and device of its inputs (`InputError` otherwise) and runs on
  torch's current CUDA stream.
- `fastforks export --format trt` builds the engine, then compares it with portable on the synthetic input (mean /
  max |difference| of every output, the worst one shown; stereo also the share of pixels off by > 1 px) and times both.
  DETR / RT-DETR: the queries scoring > 0.3 are matched by box (RT-DETR's encoder top-k picks and orders the 300
  queries, so near ties swap them; per-index differences would be as large as from 1e-4 input noise).

## What works

Compile and TensorRT per family (one variant each, at the task's standard size, with the failure reasons):
[benchmarks](benchmarks.md#torchcompile-and-tensorrt). In short: `torch.compile` works for every family, the fp16
TensorRT engine for all but these four (each a TensorRT bug; run their ONNX file with onnxruntime instead):

| model | TensorRT error | fp32 engine (`load(..., dtype="float32")`) |
|---|---|---|
| BridgeDepth | Myelin "Inconsistent strides" | builds |
| DEFOM | Myelin "Must have costs" | fails the same way |
| LAS2-H (994x1482; LAS2-S / M / L build) | Myelin "Must have costs" | fails the same way |
| MVSAnywhere | builds, but outputs NaN (`export_tensorrt` removes the engine and raises `BackendUnavailableError`) | NaN too |

- Stereo `kernels` is one opaque custom op (`fastforks::stereo_disparity`): it
  compiles and `torch.export`s for this process, but ONNX and TensorRT raise `BackendUnavailableError`. The other tasks'
  `kernels` have no exportable module.
- Stereo: ONNX is checked against onnxruntime on the CPU for every family; FoundationStereo, C-FS and Selective-IGEV
  need torch >= 2.14.
- Depth: ZoeDepth NK does not compile or export (graph break in the vendored MiDaS core, its NYU / KITTI head is picked
  per image). Marigold is not in the CPU export suite (Stable Diffusion UNet + VAE, > 2 GB graph). The exported
  graphs take only the image, no camera intrinsics: Metric3D's exported metric depth assumes its canonical camera
  (focal length 1000 px at the input size); rescale it yourself by `fx * s / 1000` (`fx` your focal length in px,
  `s` the resize factor to the export size). UniDepth's export uses its own predicted camera.

The CPU suite (`tests/test_export.py`) runs compile (`fullgraph`, zero graph breaks), export and ONNX + onnxruntime
for the smallest variant of every family and compares each with eager.

Notes:

- Compiled or exported graphs can differ from eager by a few thousandths of a pixel (other kernels, other rounding).
  Eager outputs repeat bitwise except portable CoEx and Fast-ACVNet ([usage](usage.md#video-and-repeated-sizes)).

## torch.compile of portable

`model.compile()` (or `torch.compile(model.module())`) speeds the portable backend up after a long first call; `kernels`
itself does not get faster when compiled. Compile times, speed and memory next to TensorRT and `kernels`, one variant
per family: [benchmarks](benchmarks.md#torchcompile-and-tensorrt).

## TensorRT (stereo, 384x1248)

Checked engines at 384x1248 (2026-10-27), TensorRT 11.4, one RTX 5060 Ti (ms vs portable at the same precision;
|difference| vs portable on the synthetic pair, no pixel off by > 1 px):

| model | fp16 engine ms | portable fp16 ms | max px | fp32 engine ms | portable fp32 ms | max px |
|---|---|---|---|---|---|---|
| hitnet-eth3d | 8.4 | 22.3 | 0.23 | 14.0 | 35.7 | 0.15 |
| las2-s | 4.0 | 18.8 | 0.07 | 8.6 | 19.1 | 0.003 |
| s2m2-s | 36.7 | 94.6 | 0.07 | 88.1 | 154.2 | 0.02 |
| crestereo-eth3d | 86.3 | 205.9 | 0.009 | 266.2 | 385.1 | 0.01 |
| igev-stereo-sceneflow | 94.2 | 214.8 | 0.01 | 281.9 | 424.9 | 0.005 |
| raft-stereo-middlebury | 101.8 | 214.5 | 0.15 | 590.7 | 457.3 | 0.13 |

Engines have a fixed input shape (the size they were built for).

Dynamic shapes are optional and not recommended: `export_onnx(..., size_range=((min_h, min_w), (max_h, max_w)))` /
`fastforks export --size-range MINxMIN:MAXxMAX` mark height and width as dynamic axes (sizes must be multiples of the
model's padding, and max must exceed min), and `export_tensorrt` adds an optimization profile (opt = the given size).
Only families whose dynamic file was checked against eager `module()` across a range accept a size range: AANet,
CAS-GwcNet, CoEx, DETR, IGEV-Stereo, PSMNet, RAFT-Stereo, R-CNN (Faster / Mask), S2M2, Sapiens and SegFormer
(`fastforks._core.export.DYNAMIC_VERIFIED`). Every other model is refused up front with the reason (`InputError`):
fixed-input models (classification, features, promptable segmentation), families whose traced graph fixes the size,
fails or gives wrong values inside the range (Depth Anything V2, HITNet, MoGe, LAS2, RT-DETRv2, Fast-ACVNet,
DINOv2-linear, ...) and the families not verified yet. A dynamic ONNX file must have symbolic H / W inputs and (with
`onnxruntime` installed) runs at the four corners of the range, where its outputs must match eager `module()` (mean
|difference| at most 1 % of the output's max |value|, at most 1 % of the values off by more than 5 % of it); an engine
runs at opt and the four corners. A file or engine that fails there is removed and the export raises `InputError`.
A dynamic engine run at a size inside its profile that is not a multiple of `model.size_range().multiple` raises
`InputError`. Measured on LAS2-S at 704x1280, fp16, RTX 5060 Ti: fixed engine 6.2 ms, dynamic engine 6.7 ms (+8 %),
build 84 s vs 116 s, and the dynamic engine failed at another size inside its range (TensorRT: a shape-dependent node
of the traced graph is not consistent), so the fixed engine stays the default. HITNet's height is fixed by the
exporter even when requested dynamic, and Depth Anything V2 cannot be exported dynamically (its resize target is
computed with Python `round`). The measurement is one model (LAS2-S).

## TensorRT or kernels?

[Benchmarks](benchmarks.md#torchcompile-and-tensorrt) has the fp16 engine next to `kernels` (speed and memory, one
variant per family at its standard size, same run). At 994x1482 the engine is faster for LAS2-S (10.3 vs 14.3 ms),
the same for Depth Anything V2 S (4.5 ms) and slower for HITNet, CREStereo, IGEV, RAFT and S2M2 (`kernels` 1.2-1.3x
faster). `kernels` needs no build step, takes any input size (an engine has one fixed shape) and runs the same weights
on every supported GPU; an engine must be rebuilt per GPU model, TensorRT version and input size.
