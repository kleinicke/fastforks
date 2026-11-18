"""RoMa v2 kernels (`romav2_kernels.cu`): the refiners' local correlation and their blocks' depthwise conv.

Local correlation without the sampled window tensor: one warp per pixel (blockDim 256 = 8 pixels), the (2R+2)^2 fp32
dot products of the integer window around the warped position, combined with the shared bilinear weights (zeros
outside, as `grid_sample`). Entry points for the two refiners that correlate: C = 192, radius 3 and C = 48, radius 1.
Depthwise 5x5 conv + bias + ReLU of a channels-last bf16 map in one pass (cuDNN's rounding): one thread per 8 channels
x `DEPTHWISE_TX` columns of a row; C a multiple of 8.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cubin_library

PIXELS_PER_BLOCK = 8
LOCAL_CORR = {(192, 3): "romav2_local_corr_c192_r3", (48, 1): "romav2_local_corr_c48_r1"}  # (C, radius) -> kernel

DEPTHWISE = "romav2_dw5_bias_relu_bf16"
DEPTHWISE_TX = 4  # columns per thread (TX of the kernel)

if TYPE_CHECKING:
    import torch

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in romav2_kernels.cu)
ROMA_CORR = {(c, r): f"roma_local_corr_c{c}_r{r}" for c, r in ((512, 7), (512, 3), (256, 2))}
SIGNATURES = dict.fromkeys((*LOCAL_CORR.values(), *ROMA_CORR.values()), "ppppiiif") | {DEPTHWISE: "ppppiiii"}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def correlate_roma1(
    feature0: torch.Tensor, feature1: torch.Tensor, local_radius: int, warp: torch.Tensor
) -> torch.Tensor:
    """RoMa v1 window correlations with the reference fp16 rounding of the normalized left feature."""
    left, right, radius = feature0, feature1, local_radius
    b, c, h, w = left.shape
    # grid_sample promotes to float32 with the float32 warp. The left scaling
    # occurs before this promotion and rounds once in the original feature dtype.
    left = (left / c**0.5).permute(0, 2, 3, 1).float().contiguous()
    right = right.permute(0, 2, 3, 1).float().contiguous()
    out = left.new_empty(b, h, w, (2 * radius + 1) ** 2)
    load(left.device).launch(
        ROMA_CORR[c, radius],
        (-(-b * h * w // PIXELS_PER_BLOCK), 1, 1),
        (256, 1, 1),
        left,
        right,
        warp.float().contiguous(),
        out,
        b,
        h,
        w,
        1.0,
    )
    return out.permute(0, 3, 1, 2)
