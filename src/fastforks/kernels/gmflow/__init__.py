"""Reduce global two-channel flow estimates without writing the softmax matrix."""

from pathlib import Path

from fastforks._core.kernels import cubin_library

SIGNATURES = {"expectation2": "pppiii"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
