"""Backbone features: `load(model_id)` returns a callable `FeaturesModel` producing a `FeaturesResult`.

```python
from fastforks import features

model = features.load("dinov2-vitl14", mode="kernels")
r = model("photo.jpg", size=518, layers=(-1, 12))
r.last_hidden_state  # [N, C]: cls (+ registers) + patch tokens, as Hugging Face's `last_hidden_state`
r.pooler_output  # [C]
r.feature_map()  # [C, h, w] patch grid
r.hidden_states[12]  # [N, C] output of block 12 (HF indexing; unselected entries are None)
```
"""

from fastforks._core.arguments import renamed_attributes
from fastforks._core.shared_types import OLD_TASK_NAMES
from fastforks.features._api import (
    FeaturesModel,
    FeaturesResult,
    list_models,
    load,
    model_card,
)

__all__ = [
    "FeaturesModel",
    "FeaturesResult",
    "list_models",
    "load",
    "model_card",
]

__getattr__ = renamed_attributes(__name__, OLD_TASK_NAMES)  # the shared types: `fastforks.<Name>`
