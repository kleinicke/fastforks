"""RoMa v2 kernels (`romav2_kernels.cu`): the refiners' local correlation and their blocks' depthwise conv.

Local correlation without the sampled window tensor: one warp per pixel (blockDim 256 = 8 pixels), the (2R+2)^2 fp32
dot products of the integer window around the warped position, combined with the shared bilinear weights (zeros
outside, as `grid_sample`). Entry points for the two refiners that correlate: C = 192, radius 3 and C = 48, radius 1.
Depthwise 5x5 conv + bias + ReLU of a channels-last bf16 map in one pass (cuDNN's rounding): one thread per 8 channels
x `DEPTHWISE_TX` columns of a row; C a multiple of 8.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

PIXELS_PER_BLOCK = 8
LOCAL_CORR = {(192, 3): "romav2_local_corr_c192_r3", (48, 1): "romav2_local_corr_c48_r1"}  # (C, radius) -> kernel

DEPTHWISE = "romav2_dw5_bias_relu_bf16"
DEPTHWISE_TX = 4  # columns per thread (TX of the kernel)

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in romav2_kernels.cu)
SIGNATURES = dict.fromkeys(LOCAL_CORR.values(), "ppppiiif") | {DEPTHWISE: "ppppiiii"}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
