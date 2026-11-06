"""IGEV++ / GREAT-IGEV kernels: correlation volume, geometry lookups + blend, LayerNorm, volume / outlook attention.

`igevpp_kernels.cu`.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in igevpp_kernels.cu)
SIGNATURES = {
    "gwc_volume96": "pipipiiiii",
    "geo_lookup8": "piipiipfpiliii",
    "geo_blend": "pppiipipiiipil",
    "layer_norm_rows": "pipipplif",
    **{f"vol_stats_c{c}": "pppiilf" for c in (8, 16, 32, 48)},
    **{f"vol_update_c{c}": "pppppiilfff" for c in (8, 16, 32, 48)},
    "vol_regress_c8": "pppiilf",
    **{f"outlook_k{k}_c{c}": "pipipppiiiff" for k, c in ((7, 8), (5, 16), (3, 32), (3, 48))},
    **{f"outlook_fold_k{k}_c{c}": "ppiiii" for k, c in ((7, 8), (5, 16), (3, 32), (3, 48))},
    "cc_stem0": "pipipppiiiii",
    "dw357": "pipppppppiiii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
