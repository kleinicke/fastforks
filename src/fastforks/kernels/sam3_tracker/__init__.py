"""In-place spatial rotary embeddings on SAM 3 video memory attention projections."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear

load = cubin_library(Path(__file__).with_name("manifest.toml"), {"rotate_pairs": "pppiiiil"})


def rotate_(x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor], spatial: int) -> torch.Tensor:
    """Rotate the first `spatial` projected tokens in each object, repeating the position table per memory frame."""
    b, tokens, d = x.shape
    cos, sin = rope
    if x.dtype != torch.float16 or not x.is_contiguous() or d % 2 or x.data_ptr() % 4:
        msg = "Expected contiguous aligned fp16 projection with even width"
        raise ValueError(msg)
    if any(
        y.shape != cos.shape
        or y.dtype != torch.float32
        or y.device != x.device
        or not y.is_contiguous()
        or y.data_ptr() % 8
        for y in rope
    ):
        msg = "Expected two aligned dense float32 rotation tables on the projection device"
        raise ValueError(msg)
    if cos.shape != (1, cos.size(1), d) or not 0 < spatial <= tokens or not cos.size(1) or spatial % cos.size(1):
        msg = "Expected one full rotation table per spatial frame and optional trailing object pointers"
        raise ValueError(msg)
    pairs = d // 2
    n = b * spatial * pairs
    load(x.device).launch("rotate_pairs", *linear(n), x, cos, sin, spatial, tokens, cos.shape[1], pairs, n)
    return x
