"""PatchmatchNet kernels: fused cost, view weights, feature weights, aggregation and hypotheses (batch 1).

`patchmatchnet_kernels.cu`. Layouts: features NHWC fp16 (ref `[HW, C]`, sources `[S, HW, C]`), depth hypotheses and
costs `[D, HW]` fp32, neighbour coordinates `[N, HW]` fp32 in grid_sample pixel units (border-clipped). MLP weights
(1x1x1 Conv3d + folded BN) packed fp32 `[w0 16xG, b0 16, w1 8x16, b1 8, w2 8, b2 1]`.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

CONFIGS = ((16, 4), (32, 8), (64, 8))  # (channels, groups) of stages 1, 2, 3

# p: pointer, i: int32, f: float32 (must match the extern "C" kernels in patchmatchnet_kernels.cu)
SIGNATURES = {
    **{f"pmn_view_weight_{c}_{g}": "ppppppii" for c, g in CONFIGS},
    **{f"pmn_cost_{c}_{g}": "pppppppiii" for c, g in CONFIGS},
    **{f"pmn_feat_weight_{c}_{g}": "pppppii" for c, g in CONFIGS},
    "pmn_aggregate": "pppppppfffiiii",
    "pmn_hypo": "pppppiifffii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
