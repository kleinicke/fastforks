"""Functional cost-volume builders for the MLX export.

Upstream fills a preallocated volume slice by slice (`volume[:, :, i, :, i:] = ...`); exported, that is a chain of
`slice_scatter` ops, each copying the whole volume. Here every disparity is one padded slab and the volume a single
`stack`, which an exported graph replays without the copies. Eager runs keep the in-place builders.

`correlation_band` is the eager CUDA correlation volume as batched GEMMs (one per image row).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable


def stacked_volume(
    left: torch.Tensor,
    right: torch.Tensor,
    disparities: int,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """`[B, C', D, H, W]`: slab `i` = `fn(left[..., i:], right[..., :w - i])` shifted right by `i` (zeros left of it).

    `fn` maps two `[B, C, H, W - i]` tensors to `[B, C', H, W - i]`; disparities >= width stay zero.
    """
    w = left.shape[-1]
    slabs = []
    zero = None
    for i in range(disparities):
        if i < w:
            slabs.append(F.pad(fn(left[..., i:], right[..., : w - i]), (i, 0)))
        else:
            if zero is None:
                zero = torch.zeros_like(slabs[0])
            slabs.append(zero)
    return torch.stack(slabs, dim=2)


def concat_fn(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Channel concatenation of the overlapping parts."""
    return torch.cat((left, right), dim=1)


_BAND_BYTES = 1 << 25  # float32 row products of one `correlation_band` GEMM: 32 MiB


def correlation_band(left: torch.Tensor, right: torch.Tensor, disparities: int, groups: int = 1) -> torch.Tensor:
    """`[B, groups, D, H, W]`: the channel mean, per group, of `left[..., x] * right[..., x - d]`, zero where `x < d`.

    Upstream's slab loop (`(left[..., i:] * right[..., :-i]).mean(1)` per disparity) is one strided elementwise pass
    and one reduction per disparity over the whole feature map. Here each image row is one GEMM of the right row
    (`disparities` zero pixels in front) with the left row, `[D + W, C] @ [C, W]`, whose band `x - d` is the volume:
    products accumulated in float32 and rounded once (the loop rounds every product to the feature dtype). Rows go
    in chunks of at most `_BAND_BYTES` of float32 products.
    """
    b, c, h, w = left.shape
    cg, d = c // groups, disparities
    # rows (b, g, y) of [W, C / G] features, the right rows with D zero pixels in front
    lr = left.view(b, groups, cg, h, w).permute(0, 1, 3, 4, 2).reshape(b * groups * h, w, cg)
    rr = F.pad(right.view(b, groups, cg, h, w).permute(0, 1, 3, 4, 2), (0, 0, d, 0)).reshape(b * groups * h, d + w, cg)
    out = left.new_empty(b, groups, d, h, w)
    step = max(1, _BAND_BYTES // ((d + w) * w * 4))
    for bg in range(b * groups):
        dst = out.view(b * groups, d, h, w)[bg]
        for y in range(0, h, step):
            n = min(step, h - y)
            rows = slice(bg * h + y, bg * h + y + n)
            with torch.autocast(left.device.type, enabled=False):  # the features' own dtype, like the loop
                # [n, D + W, W]: prod[r, D + x - i, x]
                if left.dtype == torch.float32:
                    prod = torch.bmm(rr[rows], lr[rows].transpose(1, 2))
                else:
                    prod = torch.bmm(rr[rows], lr[rows].transpose(1, 2), out_dtype=torch.float32)
            # disparity i at (D + x - i) * W + x = W + (D - 1 - i) * W + x * (W + 1): flipped, a strided view
            band = prod.as_strided((n, d, w), ((d + w) * w, w, w + 1), w).flip(1)
            torch.mul(band.transpose(0, 1), 1.0 / cg, out=dst[:, y : y + n])
    return out


def gwc_volume_blocks(
    left: torch.Tensor, right: torch.Tensor, depth: int, groups: int, block: int = 32
) -> torch.Tensor:
    """`correlation_band` for a small `depth` (kernels paths): channels-last-3d `[B, groups, depth, H, W]`.

    Each row is cut into `block`-pixel windows, and one `baddbmm` computes every window's dot products with the
    `block + depth - 1` `right` pixels it can pair with (`right` zero-padded by `depth - 1` in front), the mean's
    `1 / C_g` in the GEMM epilogue: about `(block + depth) / depth` times the needed products instead of
    `(W + depth) / depth`. Products are summed in float32 and rounded once to the features' dtype, as in
    `correlation_band` (994 x 1482 stereo, 96 channels, 8 groups, depth 48: slab loop 13.9 -> 5.6 ms).
    """
    b, c, h, w = left.shape
    cg = c // groups
    nb = -(-w // block)
    wp, span = nb * block, block + depth - 1

    def rows(x: torch.Tensor) -> torch.Tensor:  # [B, H, groups, W, C_g]
        return x.permute(0, 2, 3, 1).reshape(b, h, w, groups, cg).transpose(2, 3)

    lb = F.pad(rows(left), (0, 0, 0, wp - w)).reshape(-1, block, cg)
    rb = F.pad(rows(right), (0, 0, depth - 1, wp - w)).unfold(3, span, block).reshape(-1, cg, span)
    m = torch.baddbmm(lb.new_empty(()).expand(lb.shape[0], block, span), lb, rb, beta=0, alpha=1.0 / cg)
    band = m.as_strided((m.shape[0], block, depth), (block * span, span + 1, 1))  # [..., e]: disparity depth - 1 - e
    band = band.view(b, h, groups, nb, block, depth).flip(-1).view(b, h, groups, wp, depth)[:, :, :, :w]
    shape = (b, groups, depth, h, w)
    out = torch.empty(shape, dtype=left.dtype, device=left.device, memory_format=torch.channels_last_3d)
    return out.copy_(band.permute(0, 2, 4, 1, 3))
