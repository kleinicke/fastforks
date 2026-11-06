"""Round-3 classification kernels of fast-vision-models (`fvm_r3.cu`; fp16 NHWC).

Tensor-core grouped / small-C dense conv (gconv2), SE scale, depthwise conv with half2 accumulation, fused stem conv +
max pool, and the cuBLASLt 1x1 conv (bound at run time to the libcublasLt PyTorch loaded).
The tensor-level wrappers are in `fastforks._optimize.ops`.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device
from fastforks.kernels.fvm_conv import FvmLibrary

if TYPE_CHECKING:
    import torch

SIGNATURES = {
    "fvm_gconv2": "ppppiiiiiiiiiiiiiiip",
    "fvm_se_scale": "pppilip",
    "fvm_dwconv2": "ppppiiiiiiiiiiip",
    "fvm_stem_pool": "pipppiiiiiiiiiiiiiiiiip",
    "fvm_pw": "pppppliiiiiipzp",
}


@functools.cache
def _load(device: torch.device) -> FvmLibrary:
    return FvmLibrary(Path(__file__).with_name("manifest.toml"), device, SIGNATURES)


def load(device: torch.device) -> FvmLibrary:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
