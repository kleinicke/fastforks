# fastforks

_These pages are rendered by mkdocs (`uv run --group docs mkdocs serve`); in a plain viewer read the same text with `python -c "import fastforks; help(fastforks.list_models)"`._

Top-level functions and classes (the exceptions are on the [errors](errors.md) page); each task module (`fastforks.stereo`, `fastforks.depth`, ...) has its own page.

::: fastforks.list_models

::: fastforks.describe_models

::: fastforks.load

::: fastforks.save

::: fastforks.patch

::: fastforks.read_images

::: fastforks.FastModel

::: fastforks.base_type

::: fastforks.Model

::: fastforks.Backend

::: fastforks.ModelCard

::: fastforks._core.contract.Contract

::: fastforks.SizeRange

::: fastforks.TensorRTEngine

## Task functions

Every task package (`fastforks.stereo`, `fastforks.depth`, ...) has `list_models`, `model_card` and `load` with these signatures. `fastforks.classification.load` also takes `class_names=` (the class names of a fine-tuned head).

::: fastforks._core.api.ListModelsFn

::: fastforks._core.api.ModelCardFn

::: fastforks._core.api.LoadFn

## Calling a model

Every model is called directly; there are no per-task verbs such as `predict` or `segment`. The table lists the call
and the task's extra methods; the options after `*` and the result classes are on each task's page.

| task | `model(...)` | returns | other methods |
|---|---|---|---|
| calibration | `image` | `CalibrationResult` | |
| classification | `image` | `ClassificationResult` | |
| depth | `image, *, intrinsics, prompt_depth` | `DepthResult` | `video(frames)` |
| detection | `image, *, score_threshold, text` | `DetectionResult` (a list for a batch) | |
| edges | `image` | `EdgesResult` | |
| features | `image, *, size, layers` | `FeaturesResult` | |
| flow | `image1, image2` | `FlowResult` | |
| interpolation | `image1, image2, *, timestep` | `InterpolationResult` | |
| matching | `image1, image2` | `MatchingResult` (dense) or `SparseMatchingResult` | `match_pairs(images, pairs)` |
| matting | `image, *, trimap, mask` | `MattingResult` | `video(frames, *, mask)` -> `VideoMattingResult` |
| multi-view | `images, *, intrinsics, cam_to_world, reference, ...` | `MultiViewResult` | |
| OCR | `image` | `OCRResult` (a list for a batch) | |
| pose | `image, *, boxes` | `PoseResult` (a list for a batch) | |
| promptable segmentation | `image, *, input_points, input_labels, input_boxes, ...` | `PromptableSegmentationResult`; concept models `ConceptSegmentationResult`, in-context models `InContextSegmentationResult` | `get_image_embeddings(image)`, `video(frames, ...)` |
| restoration | `image, *, mask, tile` | `RestorationResult` | |
| retrieval | `image` | `RetrievalResult` | |
| segmentation | `image, *, logits` | `SegmentationResult` | |
| stereo | `left, right, *, rig, views, size, focal` | `StereoResult` | `run_many(lefts, rights, ...)` |
| tracking | `video, queries` | `TrackingResult` | |
| volume | `volume, *, spacing, meta` | `VolumeResult` | `networks()` |

All models also have `module()`, `example_inputs()`, `compile()`, `export_onnx()`, `export_tensorrt()`, `warmup()`,
`size_range()`, `export_size()` and `release_graphs()` (see `fastforks.Model` above).

## What `model.module()` covers

`module()` is the part of `model(...)` that `torch.compile`, `torch.export` and ONNX see; the rest of the call stays in
Python. Shapes and sizes per task: [export](../export.md). Every task has it on `portable`; stereo `kernels` gives one
opaque custom op (compiles, exports for this process only), the other tasks' `kernels` raise `BackendUnavailableError`.

| task | inside `module()` | outside (only in `model(...)`) |
|---|---|---|
| stereo | `(left, right)` RGB `[B, 3, H, W]` in `[0, 1]` -> `disparity` `[B, H, W]`, with scaling, padding and cropping | image conversion, `rig=` rectification, depth / points, the result object |
| flow | `(image1, image2)` RGB `[B, 3, H, W]` in `[0, 1]` -> `flow` `[B, 2, H, W]`, with scaling, padding and cropping | image conversion, the result object |
| depth | image -> the main map at the input size, with the resize to the processing size, padding and cropping | other maps (points, normals, confidence, intrinsics), `intrinsics=` |
| classification | the timm network: the transformed crop -> `logits` `[B, classes]` | resize, crop and normalisation (`model.data_config`), top-k |
| segmentation | RGB `[B, 3, H, W]` in `[0, 1]` -> logits at the input size, with the input handling | the argmax to `labels`, image conversion |
| features | RGB `[B, 3, H, W]` in `[0, 1]` -> `last_hidden_state` `[B, N, C]`, with the resize to the patch stride | `pooler_output`, image conversion |
| promptable segmentation | the image encoder: `[1, 3, H, W]` -> `image_embeddings` (SAM 3: the FPN levels `fpn0`, `fpn1`, `fpn2`) | the prompt encoder, the mask decoder, mask thresholding |
| pose | the network on person crops `[B, 3, h, w]` in `[0, 1]` at the network size -> heatmaps (ViTPose) or SimCC logits (RTMPose) | cropping the boxes, keypoint decoding and mapping back to the image |
| detection | R-CNN: backbone + FPN -> the pyramid levels; DETR / RT-DETR: the whole net -> class logits and boxes, at the network size | resize, proposals, NMS, `score_threshold`, box scaling, mask pasting |
| multi-view depth | views `[V, 3, H, W]`, then the cameras (required ones first) -> the dict of maps | image conversion, the result object |
| volume | not available: `model.networks()` are the patch-sized U-Nets `[B, C, d, h, w]` -> logits | resampling, normalisation, sliding windows, mirroring, fold ensemble, argmax, resampling back |
