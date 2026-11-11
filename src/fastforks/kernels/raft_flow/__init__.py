"""RAFT optical-flow kernels (`raft_flow_kernels.cu`).

Normalization, the correlation lookup without a volume, GRU gates, the flow head output and the upsampling (RAFT);
the depthwise conv + LayerNorm of SEA-RAFT's ConvNeXt blocks; the one-level lookup of NeuFlow v2.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in raft_flow_kernels.cu)
SIGNATURES = {
    "prep_image": "ppl",
    "in_stats_partial": "pliip",
    "in_stats_final": "piilifp",
    "in_apply": "ppppplii",
    "affine_act": "pipipipppilii",
    "avgpool2": "piipiii",
    "corr_lookup_c256_r4": "pppppiippif",
    "corr_lookup_c128_r3": "pppppiippif",
    "corr_lookup_c128_r4_l1": "ppiippif",
    "init_coords": "pil",
    "write_flow": "pilppii",
    "convflow1": "piipppi",
    "gru_zr": "pipppili",
    "gru_update": "pppipipli",
    "flow_head_out_256": "piippp",
    "flow_head_out_128": "piippp",
    "upsample_convex": "ppiip",
    "upsample_bilinear": "piip",
    "dwconv7_ln_384": "piiippppfp",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
