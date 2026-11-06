"""Transformer row ops of fast-vision-models (`fvm_extra.cu`): fused rotary q / k split and SwiGLU + LayerNorm.

(EVA-02).
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
    "fvm_rope_qk": "ppppiiiiip",
    "fvm_swiglu_ln": "ppppliifp",
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
