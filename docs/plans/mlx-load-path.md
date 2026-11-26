# MLX load path

2026-11-26. Goal: fp16 MLX from `load()` to the result as fast and lean as possible, the goldens unchanged.

## Conversion cache format

`fastforks._core.mlx.convert_cached` (format and listing: `fastforks._core.mlx_cache`; neither imports torch):

    <FASTFORKS_CACHE>/mlx/<task>/<family>/<variant>-r<rev>-<sha256[:12]>-<layout>-<f32|f16|bf16>.safetensors

- Stored in the run dtype; the arrays a port computes with in float32 (RoPE tables, a recorded builder's float32
  arrays) stay float32 and are listed in the `fp32` metadata. Every load-time transform (BatchNorm folds, layout
  permutes, packed projections, tables) is applied before saving, so a hit only reads the file.
- A value outside float16's range is an error at conversion (store bf16 or keep the array float32), not a silent inf.
- Metadata (`mlx_cache.META_KEYS`): format `fastforks-mlx-1`, variant, task, family, revision, weight file name and
  SHA-256, layout (with the source hash), layout name, dtype, fp32 islands, fastforks and MLX versions, creation time,
  plus the port's `config` JSON (what a hit builds the network from instead of a meta-device network).
- `sources`: the files whose code shapes the arrays are hashed into the layout tag (8 hex), so editing a builder never
  replays an old file. The superseded file is then stale for `prune` (same variant, weight file, dtype, layout name,
  older `created`).
- A hit needs neither torch nor the checkpoint: safetensors only, never a pickle. `load(weights=...)` is converted on
  every load and never cached. Writes go to `.<pid>.<thread>.<name>` and are renamed into place; a failed write leaves
  nothing behind.
- `fastforks cache mlx` lists the files with model, dtype, size and `stale: <reason>`; `--prune [--dry-run]` deletes
  the stale ones. Stale: no provenance (every file of the earlier format, e.g. `...-vit-cls-v1.safetensors`), variant not
  installed, another revision or weight file, superseded. Only regular files below `<FASTFORKS_CACHE>/mlx/` are
  deleted (a symlink or a path resolving elsewhere is refused), then the empty directories there.

## Port API: `fastforks.shared.mlx_port`

For hand-written MLX networks whose builder walks the PyTorch modules:

- `tensor(t, dtype, convert)`: the builder reads every weight through it. Outside a recording it is `convert()`.
- `recorded(ctx, layout, make, *, real, skeleton, sources)`: a miss builds `real()` (the checkpoint under
  `lazy_build`), runs `make` on it and records the arrays in call order into the cache. A hit builds `skeleton()` on the
  meta device (structure only, no checkpoint read), runs the same `make` on it, and each `tensor` call takes the next
  cached array (shape and dtype checked; every array must be taken, else the layout tag is wrong).
- So a builder is cache-ready once its weight reads go through `tensor`; anything it computes from weight values in
  other ways must be routed through `tensor` too.
- A hit evaluates the loaded arrays at once: a lazy `mx.load` array captured by `mx.compile` is traced into the graph,
  which then keeps its own copy of the weights beside the dict (eva02-large: 1178 MB resident instead of 593).

Ports that build their configuration from the network instead (`mlx_vit_cls.cached`) store it as the `config` JSON
and build no network at all on a hit.

## Done (pilots)

- RTMDet, YOLOX (`portable_mlx`, `kernels_mlx`): `RUNNERS` maps to an MLX runner that no longer builds the PyTorch
  runner (`mlx_wrap.port` did, in float32); `shared/cspnext/mlx_net.arr` reads through `tensor`, and now converts
  from a C-contiguous array (a strided numpy view gave slightly different float32 kernel results: cold and warm now
  agree bit for bit).
- ViT classifiers (vit, deit, eva02, siglip, nsfw_vit, wd_tagger ViT/EVA): `mlx_vit_cls.cached`, layout `vit-cls-v2`,
  stored in the run dtype with the configuration in the metadata.
- ResNet, MobileNetV2 (`libnet.mlx_cnn`): recorded, with a weightless `libnet.skeleton`.
- `fastforks._core.mlx` imports torch only in `to_torch`.

Outputs were compared base vs cold vs warm cache, fp16 and fp32: identical for every pilot except RTMDet fp32 (6e-5 px
box difference from the contiguity fix), and the goldens pass.

## Measurements

M1 Pro, fp16, a fresh process per run, two runs each (a/b). Baseline: 7cc6b01c.

Network only (the MLX network callable on the identical recorded input, `mx.eval`, median of 20; peak = MLX
allocator peak of one call incl. the resident weights):

| model | net ms before | net ms after | net peak MB before | after | resident MB before | after |
|---|---|---|---|---|---|---|
| rtmdet-x 800x1344 | 104.9 / 104.9 | 104.9 / 104.8 | 465 | 421 | 229 | 207 |
| yolox-l 800x1344 | 48.2 / 48.6 | 47.8 / 47.9 | 406 | 402 | 141 / 122 | 137 |
| yolox-x 800x1344 | 92.0 / 91.9 | 87.4 / 87.2 | 700 | 700 | 242 | 242 |
| eva02-large 448 | 195.0 / 195.1 | 195.2 / 195.3 | 718 | 718 | 594 | 593 |
| resnet50 224 | 5.94 / 5.88 | 5.94 / 5.91 | 75 | 75 | 50 | 50 |

No pilot's network got slower or heavier (same MLX code; the detectors no longer keep a float32 PyTorch copy).

Around the network (warm cache unless marked; load peak = process footprint after `load()`):

| model | import s | load cold s before / after | load warm s before / after | load peak warm MB before / after | pre/post ms before / after |
|---|---|---|---|---|---|
| rtmdet-x | 0.65 / 0.62 | 1.06, 0.54 / 1.24, 0.83 | 0.54, 0.53 / 0.16, 0.16 | 1170 / 383 | 6.6 / 6.6 |
| yolox-l | 0.63 / 0.60 | 0.88, 0.40 / 0.55, 0.55 | 0.41, 0.41 / 0.16, 0.16 | 753 / 286 | 5.7 / 5.9 |
| yolox-x | 0.64 / 0.60 | 1.29, 0.51 / 0.82, 0.80 | 0.51, 0.52 / 0.16, 0.16 | 1155 / 373 | 5.3 / 5.5 |
| eva02-large | 0.63 / 0.59 | 6.33, 3.80 / 5.63, 2.88 | 1.28, 1.24 / 0.18 | 981 / 766 | 1.1 / 1.1 |
| segformer-b0 1024x2048 | 0.61 / 0.60 | 2.77, 2.18 / 2.50, 2.17 | 2.24, 2.09 / 2.13, 2.14 | 339 / 340 | (not ported) |
| resnet50 | 0.58 / 0.59 | 0.94, 0.67 / 1.04, 0.78 | 0.67, 0.66 / 0.62, 0.69 | 405 / 306 | 1.5 / 0.7 |

The first warm load after the cache file left the page cache reads it from disk (rtmdet 0.44 s, eva02 0.91 s once).
Cold loads of rtmdet and resnet50 are slower (one conversion that also writes the run-dtype file, and the recording);
every later load is faster.

End to end (warm cache):

| model | first call s before / after | warm ms before / after | run peak MB before / after | torch after import / call |
|---|---|---|---|---|
| rtmdet-x | 0.66 / 0.61-0.73 | 110.5 / 110.6 | 1468 / 947 | yes / yes |
| yolox-l | 0.16 / 0.17 | 54.6 / 53.6 | 1077 / 825 | yes / yes |
| yolox-x | 0.28 / 0.27 | 96.2 / 92.9 | 1488 / 1108 | yes / yes |
| eva02-large | 0.22 / 0.23 | 195.6 / 196.7 | 1139 / 954 | yes / yes |
| segformer-b0 | 1.07 / 1.06 | 59.2 / 58.1 | 860 / 865 | yes / yes |
| resnet50 | 0.022 / 0.035 | 6.6 / 6.6 | 440 / 359 | yes / yes |

resnet50's first call is 13 ms slower (0.022 -> 0.035 s), not explained yet (likely the replayed arrays' first use; not
verified). resnet50's before pre/post 1.5 ms is the mean of a noisy 2.4 and 0.7.

## Left

- A torch-free `load()`: the task APIs, `_core/model.py`, the runners and the image pre-processing (`cv2_linear`) are
  torch. Beyond the safe part done here, it needs numpy/MLX pre-processing per task matching the goldens, runners that
  return numpy, and `_core/model.py` importing torch lazily (auto routing, kernels checks). torch stays in
  `sys.modules` after `import fastforks.<task>` today.
- The other ~170 `mlx_wrap.port` families: each needs its builder's weight reads routed through `mlx_port.tensor` and
  a skeleton; the generic exported-graph translation (`mlx_graph.GraphNet`) and the other `mlx_cnn` families
  (ConvNeXt incl. the WD tagger's, EfficientNet, RegNet, ...) are not cached yet.
- SegFormer (in the measured set) is not converted: its warm load stays at about 2 s.
- The `converted` users (c_radio, dinov2/v3, crestereo, depth_anything, depth_pro, tipsv2, mlx_features) still cache
  float32 and cast at load; moving them to `convert_cached(dtype=ctx.dtype)` is a one-line change each but was out of
  scope (C-RADIO, DINOv2/v3 and SAM 2/3 networks untouched by the brief).
- Run `fastforks cache mlx --prune` once to drop the earlier-format files.
