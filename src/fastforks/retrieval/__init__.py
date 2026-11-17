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

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.retrieval._api import (
    RetrievalModel,
    RetrievalResult,
    list_models,
    load,
    model_card,
    pairs,
    top_k,
)

__all__ = [
    "RetrievalModel",
    "RetrievalResult",
    "list_models",
    "load",
    "model_card",
    "pairs",
    "top_k",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
