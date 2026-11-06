# Detection API

_These pages are rendered by mkdocs (`uv run --group docs mkdocs serve`); in a plain viewer read the same text with `python -c "import fastforks; help(fastforks.detection)"`._

Field names follow the dicts of Hugging Face `transformers` `post_process_object_detection` /
`post_process_instance_segmentation`, one `DetectionResult` per image:

| fastforks | Hugging Face | notes |
|---|---|---|
| `model(image)` | `model(**processor(image))` + `processor.post_process_object_detection(...)` | takes the raw image (the model resizes / normalises / pads as the original code evaluated it) |
| `DetectionResult.boxes` `[N, 4]` | `boxes` | `(x0, y0, x1, y1)` in input pixels |
| `DetectionResult.scores` `[N]` | `scores` | sorted descending |
| `DetectionResult.labels` `[N]` int64 | `labels` | index into `model.class_names` (`result.class_names`) |
| `DetectionResult.masks` `[N, H, W]` bool | `segmentation` / per-instance masks of `post_process_instance_segmentation` | mask models only (else `None`); at the input size |
| `model(image, score_threshold=)` | `threshold` of the post-processing | per call; default: the load option `score_threshold` |
| load option `max_detections` | `top_k` / model config | |

A batch (BHWC / BCHW) gives a `list[DetectionResult]`; lists of images are not accepted. `len(result)` is the number
of detections, `result.to_list()` one plain dict per detection, `result.cpu()` the result on the host (through
pinned buffers: faster than `.cpu()` on each field for full-image masks).

The NMS / ROIAlign operators come from torchvision (`pip install 'fastforks-core[torchvision]'`); without it `load` raises
`BackendUnavailableError`.

`model.module()` (portable backends) takes RGB `[B, 3, h, w]` in `[0, 1]` already at the network size (normalised and
padded inside). R-CNN models: **backbone + feature pyramid** only -> the pyramid levels; the resize, proposals, NMS and
mask pasting are data-dependent and not part of it. DETR / RT-DETR: the whole network -> class logits and boxes (the
post-processing stays in `model(...)`). Export needs a size (`--size`; COCO inference: 800x1344).

`load`, `list_models` and `model_card` have the signatures documented under [task functions](index.md#task-functions).

::: fastforks.detection
