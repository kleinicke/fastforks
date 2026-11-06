"""Mask2Former / OneFormer kernels of fast-vision-models (`fvm_seg.cu`; fp16).

`msda_self_h`: multi-scale deformable self-attention of the pixel decoder in one pass (softmax over levels x points,
sampling locations from the query's own position, bilinear sampling with zero padding as `grid_sample`), fp32 math.
`ln_gather_h` / `gather_add_ln_h`: the Swin layer's LayerNorm + pad + roll + window partition, and window reverse +
roll back + crop + residual + LayerNorm, as gathers through precomputed index tables. Used by
`fastforks.shared.maskformer`.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, f: float32 (must match the extern "C" kernels in fvm_seg.cu)
SIGNATURES = {
    "msda_self_h": "pppppp" + "iiiiii",
    "ln_gather_h": "ppppp" + "ii" + "f",
    "gather_add_ln_h": "ppppppp" + "ii" + "f",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
