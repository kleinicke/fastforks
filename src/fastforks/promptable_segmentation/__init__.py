"""Promptable segmentation (SAM-style): `load(model_id)` returns a `PromptableSegmentationModel`.

```python
from fastforks import promptable_segmentation as ps

model = ps.load("sam-vit-h", mode="kernels")
emb = model.get_image_embeddings("street.jpg")  # once per image (the expensive part)
r = model(emb, input_points=[[500, 375]], input_labels=[1])  # cheap per prompt
r.masks  # [3, H, W] bool candidates (multimask_output=True), r.iou_scores [3] ranks them
r = model(emb, input_boxes=[[425, 600, 700, 875], [1375, 550, 1650, 800]], multimask_output=False)
r.masks  # [2, 1, H, W]: one mask per box
embs = model.get_image_embeddings(batch)  # BHWC / BCHW: one object, embs[i] per image
```

SAM 3 segments every instance of a concept: a text prompt and / or exemplar boxes (1 positive, 0 negative);
`load` returns a `ConceptSegmentationModel` for it (`ConceptModelId`), whose calls give `ConceptSegmentationResult`:

```python
model = ps.load("sam3", mode="kernels")
r = model("kitchen.jpg", text="mug")  # ConceptSegmentationResult: r.boxes [N, 4] px, r.scores [N], r.masks [N, H, W]
r = model("kitchen.jpg", input_boxes=[[40, 60, 120, 150]])  # every object like the one in the box
r = model("kitchen.jpg", text="mug", input_boxes=[[40, 60, 120, 150], [300, 80, 380, 170]], input_boxes_labels=[1, 0])
```

SAM 2 also tracks objects through a video (`track`; prompts on the first frame, `VideoSegmentationResult`):

```python
model = ps.load("sam2.1-hiera-tiny", mode="kernels")
r = model.track(frames, input_points=[[[210, 350]], [[600, 300]]], input_labels=[[1], [1]])  # two objects
r.masks  # [T, 2, H, W] bool; or input_boxes=[[x0, y0, x1, y1]], input_masks=frame0_mask ([H, W])
```

`model.module()` (portable) is the image encoder only (image -> `image_embeddings`; SAM 3: `fpn0` .. `fpn2`), not the
prompt side.
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.promptable_segmentation._api import (
    Backend,
    ConceptModelId,
    ConceptSegmentationModel,
    ConceptSegmentationResult,
    ImageEmbeddings,
    PromptableSegmentationModel,
    PromptableSegmentationResult,
    PromptLike,
    VideoSegmentationResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "Backend",
    "ConceptModelId",
    "ConceptSegmentationModel",
    "ConceptSegmentationResult",
    "ImageEmbeddings",
    "ImageLike",
    "ModelCard",
    "PromptLike",
    "PromptableSegmentationModel",
    "PromptableSegmentationResult",
    "SizeRange",
    "TensorRTEngine",
    "VideoSegmentationResult",
    "list_models",
    "load",
    "model_card",
]
