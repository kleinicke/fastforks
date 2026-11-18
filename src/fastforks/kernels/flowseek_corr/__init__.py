"""Native 384-channel batched correlation windows for FlowSeek."""

from pathlib import Path

from fastforks._core.kernels import cubin_library

SIGNATURES = {"lookup": "pppppppiif"}
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
