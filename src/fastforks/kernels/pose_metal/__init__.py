"""Metal kernels for top-down pose on Apple MPS (`pose_kernels.metal`): ViTPose's decoder, DARK, RTMPose's crop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from fastforks._core.metal import metal_library

if TYPE_CHECKING:
    from fastforks._core.metal import MetalLibrary

load = metal_library("pose_metal", Path(__file__).with_name("pose_kernels.metal"))

GROUP = 256  # fixed: threads per group of every kernel here (dark_decode: one group per heatmap, 8 SIMD groups)


def deconv4s2_relu(lib: MetalLibrary, cols: torch.Tensor, bias: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """ReLU(ConvTranspose2d(kernel 4, stride 2, padding 1) + bias) from its GEMM `cols` (the overlap-add half).

    `cols` `[B, height * width, 16 * C]`: the input tokens times the weight `[in, 4, 4, C]` reshaped to `[in, 16 * C]`
    -> `[B, 4 * height * width, C]` (the `2 height x 2 width` output, channels last), in `cols`' dtype.
    """
    bsz, n, k = cols.shape
    c = k // 16
    if n != height * width or k != 16 * c or bias.shape != (c,):
        msg = f"deconv4s2_relu: cols {tuple(cols.shape)}, bias {tuple(bias.shape)} for a {height} x {width} input"
        raise ValueError(msg)
    out = cols.new_empty(bsz, 4 * n, c)
    lib.kernel("deconv4s2_relu", cols.dtype)(
        out,
        cols.contiguous(),
        bias.to(cols.dtype).contiguous(),
        bsz,
        height,
        width,
        c,
        threads=out.numel(),
        group_size=GROUP,
    )
    return out


def dark_decode(  # noqa: PLR0913 - the kernel's operands
    lib: MetalLibrary,
    hm: torch.Tensor,
    keypoints: int,
    *,
    channels_last: bool,
    sigma: float,
    radius: int,
    clip: tuple[float, float],
) -> torch.Tensor:
    """`[P, K, 3]` float32 (x, y in heatmap pixels after the DARK step, score) of float32 heatmaps.

    `hm`: `[P, h, w, K]` if `channels_last`, else `[P, K, h, w]`; `sigma`, `radius`, `clip`: the blur and the clip
    before the log (`pose.models.vitpose.model.decode`). A heatmap with a NaN gives NaN x, y and score, as PyTorch.

    Raises:
        ValueError: Not float32 `[P, ., ., .]`, another keypoint count, or a heatmap smaller than `radius` (the
            blur's reflected border needs `radius` pixels per axis; the PyTorch decode raises an index error there).
    """
    if hm.dtype != torch.float32 or hm.dim() != 4:  # noqa: PLR2004 - [P, ., ., .]
        msg = f"dark_decode takes float32 [P, h, w, K] / [P, K, h, w] heatmaps, got {hm.dtype} {tuple(hm.shape)}"
        raise ValueError(msg)
    p, k = hm.shape[0], keypoints
    h, w = hm.shape[1:3] if channels_last else hm.shape[2:]
    if hm.shape[3 if channels_last else 1] != k:
        msg = f"dark_decode: {k} keypoints, heatmaps {tuple(hm.shape)} (channels_last={channels_last})"
        raise ValueError(msg)
    if radius < 0 or min(h, w) < radius:
        msg = f"dark_decode: {h} x {w} heatmaps are smaller than the blur radius {radius} (its reflected border)"
        raise ValueError(msg)
    cs, ps = (1, k) if channels_last else (h * w, 1)
    out = hm.new_empty(p, k, 3)
    lib.kernel("dark_decode", torch.float32)(
        out, hm.contiguous(), h, w, k, sigma, radius, clip[0], clip[1], cs, ps, threads=p * k * GROUP, group_size=GROUP
    )
    return out


def warp_crop(lib: MetalLibrary, image: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """OpenCV-exact bilinear crops `[P, 3, h, w]` in 0..255 (integers, float32) of `image` `[1, 3, H, W]` in [0, 1].

    `xs` `[P, w]`, `ys` `[P, h]`: the source coordinates of the crop columns and rows (multiples of 1/32, OpenCV's
    fixed point: `shared.topdown.fixed`); pixels outside the image count as 0, a NaN or infinite
    coordinate gives NaN (as `shared.topdown.crop`).
    """
    ih, iw = image.shape[-2:]
    (p, w), h = xs.shape, ys.shape[1]
    if image.shape[:2] != (1, 3) or ys.shape[0] != p:
        msg = f"warp_crop: image {tuple(image.shape)} (want [1, 3, H, W]), xs {tuple(xs.shape)}, ys {tuple(ys.shape)}"
        raise ValueError(msg)
    out = xs.new_empty(p, 3, h, w, dtype=torch.float32)
    lib.kernel("warp_crop", torch.float32)(
        out,
        image[0].float().contiguous(),
        xs.float().contiguous(),
        ys.float().contiguous(),
        p,
        ih,
        iw,
        h,
        w,
        threads=p * h * w,
        group_size=GROUP,
    )
    return out
