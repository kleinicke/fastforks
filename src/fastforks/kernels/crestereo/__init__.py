"""CREStereo kernels: the fused adaptive group correlation (warp + 4-group local correlation) and the flow packing.

`crestereo_kernels.cu`.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in crestereo_kernels.cu)
SIGNATURES = {
    "agcl_corr": "pppppiiiii",
    "flow_pack": "ppppiii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
