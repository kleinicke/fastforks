"""IGEV-family kernels: conv epilogues, geometry / correlation lookups, ConvGRU gates, pooling / interpolation.

`igev_kernels.cu`; the correlation lookup of DEFOM-Stereo (`defom_lookup`) lives here too.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in igev_kernels.cu)
SIGNATURES = {
    "affine_act": "pipppilii",
    "affine_act8": "pipppilii",
    "affine_actx": "pipppilii",
    "affine_act8x": "pipppilii",
    "geo_lookup": "piipiippli",
    "gru_zr": "pppipili",
    "gru_update": "ppppili",
    "pool2x_into": "piiipiiii",
    "interp_into": "piiipiiii",
    "pool2x_into2": "piiippiiii",
    "interp_into2": "piiippiiii",
    "defom_lookup": "pipipipipilii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
