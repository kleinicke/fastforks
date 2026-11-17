"""Fused fp16 SwiGLU activation for a packed gate and up projection."""

from pathlib import Path

from fastforks._core.kernels import cubin_library

SIGNATURES = {"swiglu": "ppil"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
