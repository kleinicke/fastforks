"""Metal kernel libraries: kernels for Apple MPS, shipped as Metal source and compiled on the device on first use.

There is nothing to download or build ahead: `torch.mps.compile_shader` compiles a library's `.metal` source in well
under a second, once per process. The tag (`name@<source hash>:metal`) records which source a result came from.
"""

from __future__ import annotations

import functools
import hashlib
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core.errors import KernelUnavailableError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class MetalLibrary:
    """A compiled Metal library: its kernels are attributes, called as `lib.name(*tensors_and_scalars, threads=n)`."""

    def __init__(self, name: str, source: str) -> None:
        """Compile `source`.

        Raises:
            KernelUnavailableError: The Metal compiler rejected the source.
        """
        self.tag = f"{name}@{hashlib.sha256(source.encode()).hexdigest()[:12]}:metal"
        try:
            self._lib = torch.mps.compile_shader(source)
        except Exception as e:  # any compiler failure (RuntimeError, ValueError, a missing API) means unavailable
            raise KernelUnavailableError(name, f"Metal compile failed: {e}") from e

    def kernel(self, name: str, dtype: torch.dtype) -> Any:  # noqa: ANN401 - torch's untyped shader function
        """Kernel `name` instantiated for `dtype` (float16 -> `<name>_half`, float32 -> `<name>_float`).

        The returned function raises `ValueError` for a non-contiguous tensor argument.

        Raises:
            TypeError: `dtype` is neither float16 nor float32.
        """
        suffix = {torch.float16: "half", torch.float32: "float"}.get(dtype)
        if suffix is None:
            msg = f"Metal kernels take float16 or float32 tensors, not {dtype}"
            raise TypeError(msg)
        fn = getattr(self._lib, f"{name}_{suffix}")

        def call(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - torch's untyped shader call
            for i, a in enumerate(args):  # the kernels index raw buffers: a strided view would be read wrongly
                if isinstance(a, torch.Tensor) and not a.is_contiguous():
                    msg = f"Metal kernel {name}: argument {i} (shape {tuple(a.shape)}) is not contiguous"
                    raise ValueError(msg)
            return fn(*args, **kwargs)

        return call


def metal_library(name: str, source: Path) -> Callable[[torch.device], MetalLibrary]:
    """The `load(device)` function of a Metal kernel library (`source`: its `.metal` file)."""

    @functools.cache
    def _compile() -> MetalLibrary:
        return MetalLibrary(name, source.read_text())

    def load(device: torch.device) -> MetalLibrary:
        """Compile the library for `device` (once per process).

        Raises:
            KernelUnavailableError: `device` is not Apple MPS, or the source does not compile.
        """
        if device.type != "mps":
            raise KernelUnavailableError(name, f"Metal kernels run on Apple MPS only, not on {device.type}")
        if not torch.backends.mps.is_available():
            raise KernelUnavailableError(name, "Apple MPS is not available")
        return _compile()

    return load
