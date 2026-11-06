"""FoundationStereo kernels (`fs_kernels.cu`): NHWC fp16 epilogues, geometry lookups, ConvGRU gates, EdgeNeXt blocks."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in fs_kernels.cu)
SIGNATURES = {
    "affine_act": "pipppilii",
    "affine_act8": "pipppilii",
    "affine_act4": "pipppilii",
    "affine_act2": "pipppilii",
    "patch_dw": "piiipppiiiiii",
    "geo_lookup": "piipiippli",
    "geo_lookup_s": "piiipiippli",
    "geo_lookup_hwd": "piipiipplii",
    "gru_zr": "pppipili",
    "pool2x_into": "piiipiiii",
    "interp_into": "piiipiiii",
    "gru_update": "ppppili",
    "dwconv7_nhwc": "ppppiii",
    "mlp_gelu_res": "ppppppppl",
    "bilerp2_nhwc": "piiiipiii",
    "build_vol_hwd": "piiiippp",
    "trilerp4_add": "piiiipiiiii",
    "chan_stats8": "plipp",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
