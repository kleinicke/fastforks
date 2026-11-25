"""Apply VGGT's query/key LayerNorm affine and 2D rotary embedding in one pass."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

SIGNATURES = {"rotary": "ppppppil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)

_HEAD_DIM = 64


def apply(
    norm: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Rotate normalized queries and keys in dense [batch, tokens, 2, heads, 64] layout."""
    b, n, qk, heads, dim = norm.shape
    if (
        (qk, dim) != (2, _HEAD_DIM)
        or weight.shape != (qk, 1, dim)
        or bias.shape != weight.shape
        or cosine.shape != (b, n, 1, 1, dim)
        or sine.shape != cosine.shape
        or any(
            t.dtype != torch.float16 or t.device != norm.device or not t.is_contiguous()
            for t in (norm, weight, bias, cosine, sine)
        )
    ):
        msg = "Expected contiguous VGGT fp16 queries, keys, affine parameters and rotary tables on one device"
        raise ValueError(msg)
    out = torch.empty_like(norm)
    load(norm.device).launch(
        "rotary", *linear(norm.numel()), norm, weight, bias, cosine, sine, out, heads, norm.numel()
    )
    return out
