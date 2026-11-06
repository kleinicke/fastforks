# Promptable segmentation API

_These pages are rendered by mkdocs (`uv run --group docs mkdocs serve`); in a plain viewer read the same text with `python -c "import fastforks; help(fastforks.promptable_segmentation)"`._

Argument and field names follow Hugging Face `transformers` (`SamModel` / `SamProcessor`), for one image per call:

| fastforks | Hugging Face | notes |
|---|---|---|
| `model.get_image_embeddings(image)` | `SamModel.get_image_embeddings(pixel_values)` | takes the raw image (the model resizes / pads); returns an `ImageEmbeddings`; for a batch one object holding every image, `emb[i]` per image |
| `model(image_or_embeddings, ...)` | `SamModel(pixel_values=... / image_embeddings=..., ...)` | |
| `input_points` `[P, N, 2]` | `input_points` `[B, P, N, 2]` | pixels of the input image (HF: the processor rescales them); no image batch `B` |
| `input_labels` `[P, N]` | `input_labels` `[B, P, N]` | 1 foreground, 0 background, -1 padding |
| `input_boxes` `[P, 4]` | `input_boxes` `[B, P, 4]` | `(x0, y0, x1, y1)` in input pixels |
| `input_masks` `[P, 1, h, w]` | `input_masks` | low-resolution logits (a previous `pred_masks`) |
| `multimask_output` | `multimask_output` | |
| `PromptableSegmentationResult.iou_scores` `[P, M]` | `iou_scores` `[B, P, M]` | |
| `PromptableSegmentationResult.pred_masks` `[P, M, h, w]` | `pred_masks` `[B, P, M, h, w]` | logits |
| `PromptableSegmentationResult.masks` `[P, M, H, W]` bool | `processor.post_process_masks(pred_masks, ...)` | upsampled to the input size and thresholded at 0 |

`P` is dropped from the result if no prompt had it (`input_points=[[x, y]]`, `input_labels=[1]` -> `masks [M, H, W]`).

Concept models (SAM 3, `"text"` in `model.prompts`; `load("sam3")` is typed as `ConceptSegmentationModel`) follow
`Sam3Model` / `Sam3Processor` and return every instance as a `ConceptSegmentationResult`:

| fastforks | Hugging Face | notes |
|---|---|---|
| `text` | `text` | the concept, e.g. `"yellow school bus"` |
| `input_boxes` `[N, 4]` | `input_boxes` `[B, N, 4]` | exemplar boxes of one prompt, `(x0, y0, x1, y1)` in input pixels; alone (original text `"visual"`) or with `text`; at most 256, `x1 > x0`, `y1 > y0`, within one image size of the image; an empty set needs `text` |
| `input_boxes_labels` `[N]` | `input_boxes_labels` `[B, N]` | 1 positive, 0 negative (default all 1); no padding label |
| `threshold` | `confidence_threshold` | default 0.5 |

`text`, `input_boxes_labels` and `threshold` on SAM / SAM 2 raise `InputError` (earlier `threshold` was ignored).

Embeddings remember the model, backend and load options; a prompt call with embeddings from another model or other
options raises `InputError`.

`model.module()` (portable backends, for `torch.compile` / `torch.export`) returns the **image encoder only**: `[1, 3, H, W]`
RGB in `[0, 1]` -> `image_embeddings` (SAM 3: the FPN levels `fpn0`, `fpn1`, `fpn2`). The prompt encoder and mask decoder are not part of it.

`list_models` and `model_card` have the signatures documented under [task functions](index.md#task-functions).

::: fastforks.promptable_segmentation
