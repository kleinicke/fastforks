"""Fused CUDA cost volumes for CFNet and Cas-GwcNet."""

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library, linear
from fastforks._core.layout import nchw

SIGNATURES = dict.fromkeys(("coarse_half", "coarse_float", "refinement"), "ppppppiiiiiil")
SIGNATURES["cascade_half"] = "ppppppiiiiiilllllfff"
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
    name = {torch.float16: "coarse_half", torch.float32: "coarse_float"}.get(dtype)
    if name is None:
        msg = "CFNet CUDA cost volumes require a float16 or float32 output"
        raise ValueError(msg)
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


def cascade(
    features: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor],
    samples: torch.Tensor,
    groups: int,
    scale: float,
) -> torch.Tensor:
    """Sample fractional disparity costs and write the CNN's packed FP16 volume directly.

    The feature tuple contains the left and right correlation features or two `None` values, followed by the
    left and right concatenation features. `samples` is broadcastable to `[B, D, H, W]`.

    Raises:
        ValueError: Features or disparity samples do not have the expected CUDA dtype or geometry.
    """
    lg, rg, left, right = features
    b, cc, h, w = left.shape
    cg = 0 if lg is None else lg.shape[1]
    if (
        left.shape != right.shape
        or h <= 1
        or w <= 1
        or min(b, cc) < 1
        or groups < 0
        or any(x is not None and (x.dtype != torch.float16 or x.device != left.device) for x in features)
        or not left.is_cuda
        or samples.dtype != torch.float32
        or samples.device != left.device
    ):
        msg = "Cascade costs require CUDA FP16 feature pairs and FP32 disparity samples on the same device"
        raise ValueError(msg)
    if groups:
        if lg is None or rg is None or cg < groups or cg % groups or lg.shape != (b, cg, h, w) or rg.shape != lg.shape:
            msg = "Correlation feature pairs must have the same geometry and whole channel groups"
            raise ValueError(msg)
    elif lg is not None or rg is not None:
        msg = "Correlation features require at least one channel group"
        raise ValueError(msg)
    if (
        samples.ndim != 4  # noqa: PLR2004 - BDHW disparity planes
        or samples.shape[0] not in (1, b)
        or samples.shape[1] < 1
        or samples.shape[2] not in (1, h)
        or samples.shape[3] not in (1, w)
    ):
        msg = "Expected nonempty [B or 1,D,H or 1,W or 1] cascade disparity samples"
        raise ValueError(msg)
    d = samples.shape[1]
    samples = samples.expand(b, d, h, w)
    out = torch.empty(
        (b, groups + 2 * cc, d, h, w), device=left.device, dtype=left.dtype, memory_format=torch.channels_last_3d
    )
    planes = [x.contiguous(memory_format=torch.channels_last) if x is not None else None for x in features]
    count = b * d * h * w
    load(left.device).launch(
        "cascade_half", *linear(count * 32), *planes, samples, out,
        groups, cg, cc, d, h, w, *samples.stride(), count, 2 / (h - 1), 2 / (w - 1), scale
    )  # fmt: skip
    return out
