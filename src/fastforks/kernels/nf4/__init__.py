"""NF4 (bitsandbytes 4-bit NormalFloat) -> bf16 weight dequantization in one pass (`nf4.cu`).

`dequant` writes `(code[nibble] * absmax).to(bfloat16)` for every packed weight straight from the uint8 codes,
instead of the portable unpack / gather / multiply / cast chain, so a 20B-parameter NF4 transformer streams its
weights at memory speed. Used by `fastforks.depth.models.marigold_v2` (kernels mode).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core import kernels
from fastforks._core.kernels import cubin_library

if TYPE_CHECKING:
    import torch

# p: pointer, l: int64 (must match the extern "C" kernel in nf4.cu)
SIGNATURES = {"nf4_dequant": "ppppl"}
_WORD = 8  # bytes per 64-bit word (16 codes)
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def dequant(q: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Fill `out` (bf16, `2 * q.numel()` elements) from `q` (uint8 packed codes), `absmax` (bf16, per 64), `code`.

    All on one CUDA device; `q` contiguous with a size divisible by 8, `code` fp32 `[16]`.
    """
    n = q.numel() // _WORD
    lib = load(q.device)
    lib.launch("nf4_dequant", *kernels.linear(n), q, absmax, code, out, n)
    return out
