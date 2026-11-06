"""Stereo building blocks (`stereo_ops.cu`): cost volume, line convs, soft-argmax, superpixel upsampling, image norm."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.kernels import cubin_library, linear

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in stereo_ops.cu)
SIGNATURES = {
    "corr_volume": "pppiiiillllf",
    "dw_line": "pppppliiiiii",
    "soft_argmax": "ppliiilllli",
    "spx_upsample": "pppliiillllf",
    "norm_pad_pair": "ppppliiiiiifi",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)

CL = torch.channels_last
_TILE = 32  # corr_volume: pixels x disparities per block (stereo_ops.cu CT)
_MAX_ROWS = 65535  # corr_volume: one grid row per feature row
_TOPK_MAX = 8  # soft_argmax (stereo_ops.cu TOPK_MAX)


def _nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float16 or x.dim() != 4:  # noqa: PLR2004 - NCHW
        msg = f"stereo_ops: expected a 4-D fp16 tensor, got {x.dtype} {tuple(x.shape)}"
        raise RuntimeError(msg)
    return x.contiguous(memory_format=CL)


def corr_volume(
    k: CubinModule,
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """`out[n, d, y, x] = scale * sum_c left[n, c, y, x] * right[n, c, y, x - d]` (0 for x < d), one kernel.

    `left` / `right`: fp16 `[N, C, H, W]` (made channels-last); `out`: any fp16 `[N, D, H, W]` view (its strides are
    used as they are, e.g. a squeezed `[N, 1, D, H, W]` volume).

    Raises:
        RuntimeError: Not fp16, shapes differ, or more than 65535 feature rows (`H`; batches with `N * H` above
            that run one launch per image).
    """
    left, right = _nhwc(left), _nhwc(right)
    n, c, h, w = left.shape
    if right.shape != left.shape or out.dtype != torch.float16 or out.shape[0] != n or tuple(out.shape[2:]) != (h, w):
        msg = f"corr_volume: views {tuple(left.shape)} / {tuple(right.shape)}, output {tuple(out.shape)} {out.dtype}"
        raise RuntimeError(msg)
    if h > _MAX_ROWS:
        msg = f"corr_volume: H = {h} feature rows, at most {_MAX_ROWS}"
        raise RuntimeError(msg)
    if n * h > _MAX_ROWS:  # one grid row per feature row: one launch per image
        for i in range(n):
            corr_volume(k, left[i : i + 1], right[i : i + 1], out[i : i + 1], scale)
        return out
    d = out.shape[1]
    sn, sd, sy, sx = out.stride()
    blocks = -(-w // _TILE) * -(-d // _TILE)
    k.launch("corr_volume", (blocks, n * h, 1), (256, 1, 1), left, right, out, h, w, c, d, sn, sy, sx, sd, scale)
    return out


def dw_line(  # noqa: PLR0913 - input, the conv's weights, axis and the optional residual / destination
    k: CubinModule,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
    axis: int,
    *,
    res: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise 1xK (`axis` 0) / Kx1 (`axis` 1) conv, zero padding K // 2 (odd K), plus `res`; `w` fp32 `[K, C]`.

    `x` fp16 (made channels-last); `res` / `out` dense channels-last tensors of `x`'s shape, `out` may be `res`
    (accumulate in place).
    """
    x = _nhwc(x)
    n, c, h, wd = x.shape
    y = torch.empty_like(x, memory_format=CL) if out is None else out
    for t in (y, res):
        if t is not None and (t.shape != x.shape or t.dtype != x.dtype or not t.is_contiguous(memory_format=CL)):
            msg = f"dw_line: residual / output must be dense channels-last {tuple(x.shape)} fp16"
            raise RuntimeError(msg)
    if w.dim() != 2 or w.shape[1] != c or not w.is_contiguous() or w.dtype != torch.float32:  # noqa: PLR2004 - [K, C]
        msg = f"dw_line: weights must be dense fp32 [K, {c}], got {w.dtype} {tuple(w.shape)}"
        raise RuntimeError(msg)
    vec = c % 8 == 0 and all(t is None or t.data_ptr() % 16 == 0 for t in (x, w, res, y))
    g = 8 if vec else 1
    k.launch("dw_line", *linear(n * h * wd * c // g), x, w, b, res, y, n * h * wd, h, wd, c, w.shape[0], axis, int(vec))
    return y


def soft_argmax(k: CubinModule, cost: torch.Tensor, topk: int = 0) -> torch.Tensor:
    """`sum_d softmax_d(cost) * d` over dim 1 of fp16 `cost` `[N, D, H, W]` (any strides), float32 `[N, 1, H, W]`.

    `topk` 1..8: only the `topk` largest costs per pixel take part (CoEx's top-k regression; lower `d` first on ties).

    Raises:
        RuntimeError: Not a 4-D fp16 tensor, no disparities, or `topk` above 8.
    """
    if cost.dtype != torch.float16 or cost.dim() != 4 or topk > _TOPK_MAX or cost.shape[1] < 1:  # noqa: PLR2004 - NCHW
        msg = (
            f"soft_argmax: expected a 4-D fp16 cost with D >= 1 and topk <= {_TOPK_MAX}, "
            f"got {cost.dtype} {tuple(cost.shape)}"
        )
        raise RuntimeError(msg)
    n, d, h, w = cost.shape
    out = torch.empty(n, 1, h, w, device=cost.device, dtype=torch.float32)
    sn, sd, sy, sx = cost.stride()
    k.launch("soft_argmax", *linear(n * h * w), cost, out, n * h * w, h, w, d, sn, sd, sy, sx, topk)
    return out


def spx_upsample(k: CubinModule, low: torch.Tensor, logits: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Superpixel upsampling: each output pixel mixes the 3x3 neighbourhood (zero padded) of its coarse pixel.

    `out[n, Y, X] = scale * sum_k softmax_k(logits[n, k, Y, X]) * low[n, Y // f + k // 3 - 1, X // f + k % 3 - 1]`,
    i.e. `context_upsample(low * scale, softmax(logits))` of LAS2 / `upfeat` of CoEx in one pass, without the
    unfolded and upsampled float32 intermediates. `low` float32 `[N, 1, h, w]`; `logits` fp16 `[N, 9, h f, w f]`
    (any strides, e.g. a channel slice); returns float32 `[N, h f, w f]`.

    Raises:
        RuntimeError: Wrong dtypes, not 9 logits, or the logits are not an integer multiple of `low`'s size.
    """
    n, _, h, w = low.shape
    f = logits.shape[2] // h if h else 0
    if (
        low.dtype != torch.float32
        or low.shape[1] != 1
        or logits.dtype != torch.float16
        or logits.shape[:2] != (n, 9)
        or f < 1
        or tuple(logits.shape[2:]) != (h * f, w * f)
    ):
        msg = f"spx_upsample: low {tuple(low.shape)} {low.dtype}, logits {tuple(logits.shape)} {logits.dtype}"
        raise RuntimeError(msg)
    low = low.contiguous()
    out = torch.empty(n, h * f, w * f, device=low.device, dtype=torch.float32)
    sn, sk, sy, sx = logits.stride()
    p = out.numel()
    k.launch("spx_upsample", *linear(p), low, logits, out, p, h, w, f, sn, sk, sy, sx, scale)
    return out


def norm_pad_pair(  # noqa: PLR0913 - images, statistics, padding and scale
    k: CubinModule,
    left: torch.Tensor,
    right: torch.Tensor,
    ms: torch.Tensor,
    *,
    pads: tuple[int, int, int, int] = (0, 0, 0, 0),
    mul: float = 1.0,
    fill: str = "replicate",
) -> torch.Tensor:
    """Pad, scale and normalize both fp32 `[N, 3, H, W]` images into one fp16 channels-last batch `[2N, 3, H', W']`.

    `(img * mul - mean) / std`, padded by `pads` (`F.pad` order: left, right, top, bottom) with `fill` `"replicate"`
    or `"zero"` (zeros after the normalisation); `ms`: fp32 `[2, 3]` (mean, std) on the images' device. Left images
    come first.

    Raises:
        RuntimeError: Not two fp32 RGB batches of one shape, `ms` not fp32 `[2, 3]`, negative padding or an unknown
            `fill`.
    """
    if fill not in {"replicate", "zero"}:
        msg = f"norm_pad_pair: fill must be 'replicate' or 'zero', got {fill!r}"
        raise RuntimeError(msg)
    if left.shape != right.shape or left.dim() != 4 or left.shape[1] != 3 or ms.shape != (2, 3) or min(pads) < 0:  # noqa: PLR2004
        got = f"{tuple(left.shape)} {tuple(right.shape)} {tuple(ms.shape)} {pads}"
        msg = f"norm_pad_pair: expected two [N, 3, H, W] images, [2, 3] statistics and padding >= 0, got {got}"
        raise RuntimeError(msg)
    if any(t.dtype != torch.float32 for t in (left, right, ms)):
        msg = f"norm_pad_pair: expected fp32, got {left.dtype} {right.dtype} {ms.dtype}"
        raise RuntimeError(msg)
    n, _, h, w = left.shape
    pl, pr, pt, pb = pads
    hp, wp = h + pt + pb, w + pl + pr
    out = torch.empty(2 * n, 3, hp, wp, device=left.device, dtype=torch.float16, memory_format=CL)
    lc, rc = left.contiguous(), right.contiguous()
    k.launch(
        "norm_pad_pair",
        *linear(2 * n * hp * wp),
        lc,
        rc,
        ms.contiguous(),
        out,
        n,
        h,
        w,
        hp,
        wp,
        pt,
        pl,
        mul,
        int(fill == "zero"),
    )
    return out
