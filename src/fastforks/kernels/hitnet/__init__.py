"""HITNet kernels (`hitnet_kernels.cu`): level-init cost, tile / warp SAD, feature head, channels-last concatenation.

The warps follow TensorFlow `warp_1d` (clipped linear interpolation along the row) with float32 accumulation; feature
maps are NHWC float16.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in hitnet_kernels.cu)
SIGNATURES = {
    "init_cost_f16": "ppiiiiipp",
    "tile_sad_f16": "pllppiiiipl",
    "ref_sad_f16": "plppiiiifp",
    "img_conv3": "ppiiiiiffippipplll",
    "deconv2x2": "piiiippipllliipilll",
    "cat_cl": "piiii" + "piillll" * 4,
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
