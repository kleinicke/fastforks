# fastforks

_These pages are rendered by mkdocs (`uv run --group docs mkdocs serve`); in a plain viewer read the same text with `python -c "import fastforks; help(fastforks.list_models)"`._

Top-level functions and classes (the exceptions are on the [errors](errors.md) page); each task module (`fastforks.stereo`, `fastforks.depth`, ...) has its own page.

::: fastforks.list_models

::: fastforks.describe_models

::: fastforks.load

::: fastforks.save

::: fastforks.patch

::: fastforks.FastModel

::: fastforks.base_type

::: fastforks.Model

::: fastforks.Backend

::: fastforks.ModelCard

::: fastforks._core.contract.Contract

::: fastforks.SizeRange

::: fastforks.TensorRTEngine

## Task functions

Every task package (`fastforks.stereo`, `fastforks.depth`, ...) has `list_models`, `model_card` and `load` with these signatures. `fastforks.classification.load` also takes `labels=` (class names of a fine-tuned head).

::: fastforks._core.api.ListModelsFn

::: fastforks._core.api.ModelCardFn

::: fastforks._core.api.LoadFn

## What `model.module()` covers

`module()` is the part of `model(...)` that `torch.compile`, `torch.export` and ONNX see; the rest of the call stays in
Python. Shapes and sizes per task: [export](../export.md). Every task has it on `portable`; stereo `kernels` gives one
opaque custom op (compiles, exports for this process only), the other tasks' `kernels` raise `BackendUnavailableError`.

| task | inside `module()` | outside (only in `model(...)`) |
|---|---|---|
| stereo | `(left, right)` RGB `[B, 3, H, W]` in `[0, 1]` -> `disparity` `[B, H, W]`, with scaling, padding and cropping | image conversion, `rig=` rectification, depth / points, the result object |
| depth | image -> the main map at the input size, with the resize to the processing size, padding and cropping | other maps (points, normals, confidence, intrinsics), `intrinsics=` |
| classification | the timm network: the transformed crop -> `logits` `[B, classes]` | resize, crop and normalisation (`model.data_config`), top-k |
| segmentation | RGB `[B, 3, H, W]` in `[0, 1]` -> logits at the input size, with the input handling | the argmax to `labels`, image conversion |
| features | RGB `[B, 3, H, W]` in `[0, 1]` -> `last_hidden_state` `[B, N, C]`, with the resize to the patch stride | `pooler_output`, image conversion |
| promptable segmentation | the image encoder: `[1, 3, H, W]` -> `image_embeddings` (SAM 3: the FPN levels `fpn0`, `fpn1`, `fpn2`) | the prompt encoder, the mask decoder, mask thresholding |
| detection | R-CNN: backbone + FPN -> the pyramid levels; DETR / RT-DETR: the whole net -> class logits and boxes, at the network size | resize, proposals, NMS, `score_threshold`, box scaling, mask pasting |
| multi-view depth | views `[V, 3, H, W]`, then the cameras (required ones first) -> the dict of maps | image conversion, the result object |
