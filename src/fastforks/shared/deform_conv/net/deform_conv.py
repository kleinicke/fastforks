"""Deformable convolution (v1 and modulated v2) in plain PyTorch ops: `grid_sample` + a matmul per row chunk.

Same math as the mmdetection-v1 CUDA extension and `torchvision.ops.deform_conv2d`:

- `offset`: `[B, deformable_groups * 2 * kh * kw, Ho, Wo]`, (dy, dx) pairs per deformable group and kernel tap
  (tap index `i * kw + j`); `mask`: `[B, deformable_groups * kh * kw, Ho, Wo]`
- the input is sampled bilinearly at `(ho * stride - padding + i * dilation + dy, ...)`; samples outside the image
  contribute zero (a sample in (-1, 0) interpolates with a zero neighbour)
- the sampled values are scaled by the mask and contracted with the weight

The sampling is `grid_sample(align_corners=True, padding_mode="zeros")`. Used by AANet (stereo) and ALIKED (matching).
"""

from __future__ import annotations

import torch
from torch.nn.modules.utils import _pair

from fastforks._core.sampling import grid_sample

__all__ = ["modulated_deform_conv"]

# fp32 elements of one chunk's column buffer: bounds the working set (the full buffer is `C * 9 * Ho * Wo`, GBs at
# Middlebury size); 2^22-2^24 measured equally fast on AANet+
_CHUNK = 1 << 24


def _tracing() -> bool:
    return torch.compiler.is_compiling() or torch.jit.is_tracing()


def modulated_deform_conv(  # noqa: PLR0913, PLR0917 - the mmdet / torchvision signature
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    deformable_groups: int = 1,
) -> torch.Tensor:
    """Modulated deformable convolution (`mask=None`: plain deformable convolution, mmdet DCN v1).

    The bilinear sampling runs in float32 (an fp16 sampling grid is too coarse for pixel coordinates) on a
    contiguous copy of the input (cuDNN's grid sampler is about 2x faster on NCHW than on channels-last); the
    columns are sampled, masked, cast to the dtype of `x` and contracted in row chunks of at most `_CHUNK` elements
    (one chunk when traced for export). The result has the dtype of `x` and is channels-last.
    """
    if groups != 1:
        msg = "deformable convolution with groups > 1"
        raise NotImplementedError(msg)
    stride, padding, dilation = _pair(stride), _pair(padding), _pair(dilation)
    b, c, h, w = x.shape
    o, _, kh, kw = weight.shape
    k, dg = kh * kw, deformable_groups
    ho = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    dev = x.device
    tracing = _tracing()

    tap = torch.arange(k, device=dev)
    ty = (tap // kw * dilation[0]).view(1, k, 1, 1).float()
    tx = (tap % kw * dilation[1]).view(1, k, 1, 1).float()
    gx = (torch.arange(wo, device=dev) * stride[1] - padding[1]).view(1, 1, 1, wo)
    sy, sx = 2.0 / max(h - 1, 1), 2.0 / max(w - 1, 1)
    wt = weight.reshape(o, c * k).t().to(x.dtype)
    rows = ho if tracing else max(1, min(ho, _CHUNK // (c * k * wo)))
    # [B, Ho, Wo, O]: each chunk's [Ho*Wo, C*K] @ [C*K, O] lands in it; returned as a channels-last [B, O, Ho, Wo]
    # view (what the convolutions after it prefer on CUDA)
    out = None if tracing else torch.empty(b, ho, wo, o, device=dev, dtype=x.dtype)
    outs = []
    for i in range(b):
        src = x[i : i + 1].to(torch.float32, memory_format=torch.contiguous_format).reshape(dg, c // dg, h, w)
        off = offset[i].reshape(dg, k, 2, ho, wo)
        for r0 in range(0, ho, rows):
            n = min(ho, r0 + rows) - r0
            oc = off[:, :, :, r0 : r0 + n].float()
            gy = (torch.arange(r0, r0 + n, device=dev) * stride[0] - padding[0]).view(1, 1, n, 1)
            grid = torch.stack(((gx + tx + oc[:, :, 1]) * sx - 1.0, (gy + ty + oc[:, :, 0]) * sy - 1.0), dim=-1)
            cols = grid_sample(
                src, grid.reshape(dg, k * n, wo, 2), mode="bilinear", padding_mode="zeros", align_corners=True
            ).view(dg, c // dg, k, n, wo)
            m = None if mask is None else mask[i, :, r0 : r0 + n].reshape(dg, 1, k, n, wo)
            if tracing:
                cols = (cols if m is None else cols * m.float()).to(x.dtype)
                outs.append(torch.matmul(cols.reshape(c * k, n * wo).t(), wt))
                continue
            if m is None:
                cols = cols.to(x.dtype)
            else:  # mask and cast in one pass
                cols = torch.mul(cols, m, out=torch.empty(cols.shape, device=dev, dtype=x.dtype))
            torch.matmul(cols.view(c * k, n * wo).t(), wt, out=out[i, r0 : r0 + n].view(n * wo, o))
            del cols
    if tracing:
        out = torch.stack(outs).view(b, ho, wo, o)
    if bias is not None:
        out = out + bias.to(x.dtype) if tracing else out.add_(bias.to(x.dtype))
    return out.permute(0, 3, 1, 2)
