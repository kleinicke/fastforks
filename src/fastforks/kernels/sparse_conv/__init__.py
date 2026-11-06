"""MoGe-3 sparse-refiner kernels (`sparse_conv.cu`): submanifold 3x3x3 conv, neighbour tables, LayerNorm + SiLU."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in sparse_conv.cu)
SIGNATURES = {
    "subm_conv_n32": "ppppppppiiiiii",
    "subm_conv_n64": "ppppppppiiiiii",
    "subm_conv_n64s3": "ppppppppiiiiii",
    "subm_conv_n128": "ppppppppiiiiii",
    "grid_nbr": "pppiii",
    "sparse_nbr": "ppppiiii",
    "ln_silu": "ppppiif",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
