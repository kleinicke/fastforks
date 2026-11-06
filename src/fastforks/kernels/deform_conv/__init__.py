"""Fused (modulated) deformable convolution: bilinear sampling + mask + conv + epilogue in one kernel.

`deform_conv.cu`. fp16 channels-last in / out, fp32 sampling and accumulation, no column buffer; the problem size
is a runtime argument (nothing is compiled or tuned per input size).
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernel in deform_conv.cu)
SIGNATURES = {"deform_conv_fwd": "pppppp" + "i" * 21 + "fi"}
TM, TN = 128, 128  # output pixels / channels per block
DG_MAX = 4
ACT_NONE, ACT_RELU = 0, 1

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """`[O, C, kh, kw]` weights as the kernel's `[kh * kw * C, Op]` fp16 matrix (Op = O rounded up to 16)."""
    o, c, kh, kw = w.shape
    m = w.detach().permute(2, 3, 1, 0).reshape(kh * kw * c, o).half()
    op = -(-o // 16) * 16
    return torch.nn.functional.pad(m, (0, op - o)).contiguous()


def fit_reason(c: int, dg: int) -> str | None:
    """Why `c` input channels in `dg` deformable groups do not fit the kernel (None: they fit)."""
    if c % 16 or (c // dg) % 8 or c % dg or not 1 <= dg <= DG_MAX:
        return f"needs 16 | C, 8 | C / dg and dg <= {DG_MAX}; got C = {c}, dg = {dg}"
    return None


def _pixel_stride(t: torch.Tensor) -> int | None:
    """Pixel stride of a channels-last `[B, C, H, W]` tensor whose pixels are dense rows (channels may be padded)."""
    b, c, h, w = t.shape
    ps = t.stride(3)
    ok = t.stride(1) == 1 and ps >= c and t.stride(2) == w * ps and (b == 1 or t.stride(0) == h * w * ps)
    return ps if ok else None


def deform_conv(  # noqa: PLR0913, PLR0917 - conv geometry and epilogue
    x: torch.Tensor,
    om: torch.Tensor,
    w: torch.Tensor,
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
    deformable_groups: int,
    mask_scale: float | None,
    s: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    act: int = ACT_NONE,
) -> torch.Tensor:
    """`act(deform_conv(x, offsets, mask) * s + t)` as a channels-last fp16 `[B, O, Ho, Wo]` tensor.

    `x`: channels-last fp16 `[B, C, H, W]`; `om`: the offset conv output, channels-last fp16 `[B, *, Ho, Wo]`:
    `dg * 2 * K` offsets ((dy, dx) per group and tap), then with `mask_scale` set `dg * K` mask logits (mask =
    `mask_scale * sigmoid(logit)`); `w` from `pack_weight`; `s` / `t` fp32 `[O]` or None, all on `x`'s device.
    Sample positions are clamped to [-2, H + 1] x [-2, W + 1] (zeros outside), so non-finite offsets give zeros.

    Raises:
        ValueError: The tensors do not have this layout, or the geometry does not fit the kernel (`fit_reason`).
    """
    b, c, h, wd = x.shape
    kh, kw = kernel_size
    k, dg = kh * kw, deformable_groups
    ho = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    need = dg * (3 if mask_scale is not None else 2) * k
    xps, omps = _pixel_stride(x), _pixel_stride(om)
    op = -(-out_channels // 16) * 16
    if reason := fit_reason(c, dg):
        msg = f"deform_conv: {reason}"
        raise ValueError(msg)
    if (
        x.dtype != torch.float16
        or om.dtype != torch.float16
        or xps is None
        or xps % 8
        or x.data_ptr() % 16
        or omps is None
        or om.shape[0] != b
        or tuple(om.shape[2:]) != (ho, wo)
        or om.shape[1] < need
        or w.shape != (k * c, op)
        or w.dtype != torch.float16
        or not w.is_contiguous()
        or w.data_ptr() % 16
        or not all(
            v is None
            or (v.shape == (out_channels,) and v.dtype == torch.float32 and v.is_contiguous() and v.device == x.device)
            for v in (s, t)
        )
        or not x.device == om.device == w.device
    ):
        msg = (
            f"deform_conv: needs channels-last fp16 x / offsets and packed weights; got x {tuple(x.shape)} "
            f"{x.dtype}, offsets {tuple(om.shape)} {om.dtype} (need {need} channels at {ho} x {wo}), w {tuple(w.shape)}"
        )
        raise ValueError(msg)
    y = torch.empty((b, out_channels, ho, wo), device=x.device, dtype=torch.float16, memory_format=torch.channels_last)
    if y.numel():
        grid = (-(-b * ho * wo // TM), -(-op // TN), 1)
        load(x.device).launch(
            "deform_conv_fwd", grid, (256, 1, 1),
            x, om, w, s, t, y, b, h, wd, c, xps, ho, wo, omps, out_channels, op, out_channels,
            *stride, *padding, *dilation, kh, kw, dg,
            int(mask_scale is not None), float(mask_scale or 0.0), act,
        )  # fmt: skip
    return y
