"""S2M2 elementwise / attention / optimal-transport kernels (`s2m2_kernels.cu`), launched through the driver API."""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in s2m2_kernels.cu)
SIGNATURES = {
    "bias_act": "ppplii",
    "bias_act8": "ppplii",
    "gate_mix8": "pppppli",
    "ot_col_lse": "pppiiff",
    "ot_row_lse": "pppiiiff",
    "ot_final": "pppiiifppp",
    "ln_rows": "pplif",
    "up2x_nhwc8": "ppiiiilii",
    "gate_mix8s": "ppppplil",
    "pool2x_nhwc8": "ppiiii",
    "gn_stats": "ppplii",
    "affine_nc8": "ppppliii",
    "convex_up": "pppiiii",
    "cv_lookup_otf": "pppppppiiifff",
    "attn_d384": "ppppllllllii" + "f",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
