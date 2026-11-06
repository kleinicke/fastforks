"""RAFT-Stereo kernels (`raft_kernels.cu`: normalization, GRU gates, correlation, resampling, upsampling)."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in raft_kernels.cu)
SIGNATURES = {
    "prep_image_chw": "ppiiiiii",
    "in_stats_partial": "pliip",
    "in_stats_final": "piilifp",
    "in_apply": "ppppplii",
    "corr_pool": "pplii",
    "corr_lookup": "ppppiiiippl",
    "convf1_direct": "piippp",
    "prep_context": "ppppppl",
    "init_coords": "pil",
    "upsample_out": "ppiiiiiipi",
    "corr_gemm": "pppif",
    "affine_act": "pipipipppiliipi",
    "gru_zr": "ppppil",
    "flow_head_out_br": "ppiippp",
    "pool2x_dual": "piiippiiii",
    "interp_dual": "piiippiiii",
    "gru_update": "ppppipl",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
