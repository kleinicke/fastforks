"""The interface a feature-extraction (backbone) family implements: one `Runner` per backend, in `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = ...
        self.hidden_size, self.num_layers = 768, 12
        self.num_prefix_tokens, self.num_register_tokens = 5, 4  # cls + 4 registers in front of the patches

    def __call__(self, image, layers):  # [B, 3, h, w] resized + normalised; HF hidden_states indices
        return {"last_hidden_state": ..., "pooler_output": ..., "hidden_states": ...}
```

The core converts and checks the image, resizes it to the network size (each side a multiple of the card's
`[io] pad.multiple`, the patch stride; `[io] interpolation`, antialiased), normalises it (`[io] input`), batches,
checks the outputs, locks and guards the device. Nothing is padded or cropped.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from fastforks._core import runner as core
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    import torch

__all__ = ["Context", "Forward", "NoParams", "Runner"]


class Runner(core.Runner):
    """One backend of a feature-extraction family.

    Instance attributes (set in `__init__`):
        hidden_size: Channels `C` of every token.
        num_layers: Number of blocks `L` (HF `hidden_states` has `L + 1` entries: 0 = embeddings, i = after block i).
        num_prefix_tokens: Tokens before the patch tokens in `last_hidden_state` (cls + registers; ConvNeXt: the
            pooled token).
        num_register_tokens: Register tokens among the prefix (after the cls token).
    """

    hidden_size: int
    num_layers: int
    num_prefix_tokens: int = 1
    num_register_tokens: int = 0

    @abc.abstractmethod
    def __call__(self, image: torch.Tensor, layers: tuple[int, ...]) -> dict[str, torch.Tensor]:
        """Run a batch.

        Args:
            image: `[B, 3, h, w]` float32 on `ctx.device`, resized (sides multiples of the stride) and normalised.
            layers: Sorted, non-negative indices into HF's `hidden_states` (0 = embedding output, i = output of
                block i, un-normalised as HF); empty: no hidden states.

        Returns:
            `"last_hidden_state"` `[B, N, C]` (prefix tokens, then the patch grid row-major, final norm applied),
            `"pooler_output"` `[B, C]` and, if `layers`, `"hidden_states"` `[B, len(layers), N, C]`.
        """
