"""MBConv-family classification kernels of fast-vision-models (`fvm_mb.cu`; fp16 NHWC).

Fused inverted-residual block (ShuffleNetV2 units), grouped conv, expand 1x1 + depthwise, depthwise with channel
multiplier, channel gather, EfficientViT ReLU linear attention and the direct stem conv.
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
    "fvm_mbconv": "pppppppppppp" + "i" * 24 + "p",
    "fvm_gconv": "ppppiiiiiiiiiiiiip",
    "fvm_dwmult": "ppppiiiiiiiiiiiip",
    "fvm_xdw": "pppppp" + "i" * 16 + "p",
    "fvm_chan_gather": "ppppliiip",
    "fvm_linattn2": "pppiiiifp",
    "fvm_stem_direct": "pipppiiiiiiiiiiiip",
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
