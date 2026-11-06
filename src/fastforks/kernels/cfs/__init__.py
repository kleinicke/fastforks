"""C-FoundationStereo-S disparity-transformer kernels (`cfs_kernels.cu`), launched through the driver API."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in cfs_kernels.cu)
SIGNATURES = {
    "tao_qkv": "plippppp",
    "tao_post": "pplip",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
