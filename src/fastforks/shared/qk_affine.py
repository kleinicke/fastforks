"""Stack query and key normalization parameters for fused rotary attention."""

from typing import Any

import torch


def cached_qk_affine(
    attention: Any,  # noqa: ANN401 - vendored attention modules
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked Q/K weights and biases, each `[2, 1, head_dim]`."""
    # Graph capture can wrap modules in temporary objects whose IDs may be reused by other blocks.
    key = None if torch.compiler.is_compiling() else id(attention)
    if key is not None and key in cache:
        return cache[key]
    affine = (
        torch.stack([attention.q_norm.weight, attention.k_norm.weight])[:, None],
        torch.stack([attention.q_norm.bias, attention.k_norm.bias])[:, None],
    )
    if key is not None:
        cache[key] = affine
    return affine
