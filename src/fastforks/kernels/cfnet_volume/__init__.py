"""CFNet's coarse and disparity-sampled refinement volumes."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear
from fastforks._core.layout import nchw

SIGNATURES = dict.fromkeys(("coarse_half", "coarse_float", "refinement"), "ppppppiiiiiil")
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def volume(
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    groups: int,
    depth: int,
    dtype: torch.dtype,
    samples: torch.Tensor | None = None,
) -> torch.Tensor:
    """Form one NCDHW volume from four fp16 feature planes; samples are optional for uniform disparities."""
    gl, gr, cl, cr = features
    b, cg, h, w = gl.shape
    cc = cl.shape[1]
    if groups < 1 or cg % groups or gr.shape != gl.shape or cl.shape != (b, cc, h, w) or cr.shape != cl.shape:
        msg = "CFNet feature planes must share spatial dimensions and have whole correlation groups"
        raise ValueError(msg)
    if any(t.dtype != torch.float16 or t.device != gl.device for t in features):
        msg = "CFNet CUDA cost volumes require fp16 feature planes on a common device"
        raise ValueError(msg)
    name = {torch.float16: "coarse_half", torch.float32: "coarse_float"}[dtype]
    if samples is not None:
        if (
            samples.shape != (b, depth, h, w)
            or samples.dtype != torch.float32
            or samples.device != gl.device
            or dtype != torch.float16
        ):
            msg = "Refinement expects FP32 [B,D,H,W] disparity samples and an fp16 output"
            raise ValueError(msg)
        name = "refinement"
        samples = samples.contiguous()
    out = gl.new_empty((b, groups + 2 * cc + int(samples is not None), depth, h, w), dtype=dtype)
    load(gl.device).launch(
        name,
        *linear(out.numel()),
        *(nchw(v) for v in features),
        samples,
        out,
        groups,
        cg,
        cc,
        depth,
        h,
        w,
        out.numel(),
    )
    return out
