"""MoCha-Stereo kernels: the fused LSTM gate update (sigmoid / tanh gates, cell and hidden state) in one pass.

`mocha_kernels.cu`.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in mocha_kernels.cu)
SIGNATURES = {
    "lstm_gates": "ppppili",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
