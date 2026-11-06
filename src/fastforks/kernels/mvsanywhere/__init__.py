"""MVSAnywhere kernels: the fused view-agnostic MLP cost volume and the antialiased-ResNet stem pooling.

`mvsanywhere_kernels.cu`. The cost-volume MLP weights travel as a by-value kernel parameter (1890 floats, 7.6 KB;
kernel parameters up to 32 KB need CUDA >= 12.1 and a Volta or newer GPU), so they sit in the constant bank like
module constants, without per-module state: several models with different weights can share the module.
"""

from __future__ import annotations

from pathlib import Path

from fastforks._core.kernels import cubin_library

NH = 32  # hidden width of the cost-volume MLP
NDYN = 23  # view- and depth-dependent layer-1 inputs (warped features 16, mask, depth, dot, angle, source ray 3)
# MlpWeights in mvsanywhere_kernels.cu: w1 [NH][NDYN], w1pn [NH], w2 [NH][NH], b2 [NH], w3 [2][NH], b3 [2]
MLP_FLOATS = NH * NDYN + NH + NH * NH + NH + 2 * NH + 2
STRUCTS = {"MlpWeights": "f" * MLP_FLOATS}  # by-value kernel parameter structs (checked by tests/test_kernels.py)

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in mvsanywhere_kernels.cu)
SIGNATURES = {
    "mvsa_cv_fused": "ppppppppp" + "iii" + STRUCTS["MlpWeights"],
    "mvsa_stem_pool": "ppiiiiii",
}

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)
