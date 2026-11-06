"""Classification conv kernels of fast-vision-models (`fvm_conv.cu`; fp16 NHWC, fp16 accumulation).

CUTLASS implicit-GEMM conv with fused bias / residual / activation (18 tile configs, picked by a fixed shape rule),
direct 8-channel stem conv, per-channel affine + activation, max / avg pool, depthwise conv and the fp32 NCHW ->
fp16 NHWC8 input conversion. The tensor-level wrappers (checks, allocation, fallbacks) are in
`fastforks._optimize.ops`; this module only loads the library and declares the C signatures.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device, shared_library

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch

_CTYPES = {
    "p": ctypes.c_void_p,
    "i": ctypes.c_int,
    "l": ctypes.c_longlong,
    "z": ctypes.c_size_t,
    "f": ctypes.c_float,
}

SIGNATURES = {
    "fvm_conv_fused": "ipppppiiiiiiiiiiiiiip",
    "fvm_stem_c8": "ppppiiiiiiiiiiip",
    "fvm_affine_act": "pppplilip",
    "fvm_maxpool": "ppiiiiiiiiiiiip",
    "fvm_dwconv": "ppppiiiiiiiiiiip",
    "fvm_avgpool": "ppiiiiiiiiiiip",
    "fvm_input_nhwc8": "ppiiilp",
}


class FvmLibrary:
    """A loaded fvm kernel library: the `extern "C"` functions as attributes with their argument types set.

    Attributes:
        tag: `name@build_id:target`.
    """

    def __init__(self, manifest: Path, device: torch.device, signatures: Mapping[str, str]) -> None:
        """Load the library for `device`.

        Raises:
            KernelUnavailableError: No usable binary for this GPU / driver.
        """
        self.lib, self.tag = shared_library(manifest, device)
        for name, sig in signatures.items():
            f = getattr(self.lib, name)
            f.argtypes = [_CTYPES[c] for c in sig]
            f.restype = ctypes.c_int
            setattr(self, name.removeprefix("fvm_"), f)

    def __getattr__(self, name: str) -> Callable[..., int]:
        """Typing hook only: the entry points are instance attributes set in `__init__`."""
        raise AttributeError(name)


@functools.cache
def _load(device: torch.device) -> FvmLibrary:
    return FvmLibrary(Path(__file__).with_name("manifest.toml"), device, SIGNATURES)


def load(device: torch.device) -> FvmLibrary:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
