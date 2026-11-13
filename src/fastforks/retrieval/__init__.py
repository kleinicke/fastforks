"""Image retrieval / place recognition: `load(model_id)` returns a callable `RetrievalModel` (a `RetrievalResult`).

`top_k` and `pairs` search the descriptors, e.g. for the image pairs of structure from motion.

```python
from fastforks import retrieval

model = retrieval.load("megaloc", mode="kernels")
desc = torch.stack([model(p).descriptor for p in paths])  # [N, D] float32, unit length
scores, index = retrieval.top_k(desc[:5], desc, k=10)  # the 10 most similar database images per query
pairs = retrieval.pairs(desc, k=20)  # [P, 2] image pairs to match (each image with its 20 nearest)
```
"""

from fastforks._core.cards import ModelCard
from fastforks._core.export import TensorRTEngine
from fastforks._core.image import ImageLike
from fastforks._core.sizes import SizeRange
from fastforks.retrieval._api import (
    Backend,
    RetrievalModel,
    RetrievalResult,
    list_models,
    load,
    model_card,
    pairs,
    top_k,
)

__all__ = [
    "Backend",
    "ImageLike",
    "ModelCard",
    "RetrievalModel",
    "RetrievalResult",
    "SizeRange",
    "TensorRTEngine",
    "list_models",
    "load",
    "model_card",
    "pairs",
    "top_k",
]
