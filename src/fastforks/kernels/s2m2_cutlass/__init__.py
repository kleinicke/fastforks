"""CUTLASS implicit-GEMM convolutions with fused S2M2 epilogues (`s2m2_cutlass.cu`; fp16 accumulation).

The problem size is a runtime argument: nothing is compiled or tuned per input size. Linear layers run as 1x1 convs.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device, current_stream, note_fallback, shared_library

if TYPE_CHECKING:
    import torch


class S2m2EpiArgs(ctypes.Structure):
    """Epilogue arguments (`struct EpiArgs` in s2m2_cutlass.cu; modes documented there)."""

    _fields_ = (
        ("mode", ctypes.c_int),
        ("act", ctypes.c_int),
        ("b", ctypes.c_void_p),
        ("r", ctypes.c_void_p),
        ("rs", ctypes.c_longlong),
        ("y", ctypes.c_void_p),
        ("ys", ctypes.c_longlong),
        ("y2", ctypes.c_void_p),
        ("y2s", ctypes.c_longlong),
        ("h", ctypes.c_void_p),
        ("hs", ctypes.c_longlong),
        ("z", ctypes.c_void_p),
        ("zs", ctypes.c_longlong),
        ("split", ctypes.c_int),
    )


def _p(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def epi_args(  # noqa: PLR0913, PLR0917 - mirrors the C struct field order
    mode: int = 0,
    act: int = 0,
    b: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
    rs: int = 0,
    y: torch.Tensor | None = None,
    ys: int = 0,
    y2: torch.Tensor | None = None,
    y2s: int = 0,
    h: torch.Tensor | None = None,
    hs: int = 0,
    z: torch.Tensor | None = None,
    zs: int = 0,
    split: int = 0,
) -> S2m2EpiArgs:
    """Build `S2m2EpiArgs` from tensors (`None` -> null pointer)."""
    return S2m2EpiArgs(mode, act, _p(b), _p(r), rs, _p(y), ys, _p(y2), y2s, _p(h), hs, _p(z), zs, split)


class CutlassConvs:
    """The loaded shared library.

    Attributes:
        tag: `name@build_id:target`.
    """

    def __init__(self, device: torch.device) -> None:
        """Load the library for `device`.

        Raises:
            KernelUnavailableError: No usable binary for this GPU / driver.
        """
        self.lib, self.tag = shared_library(Path(__file__).with_name("manifest.toml"), device)
        self.lib.s2m2_conv.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *[ctypes.c_int] * 10,
            ctypes.POINTER(S2m2EpiArgs),
            ctypes.c_void_p,
        ]

    def conv(  # noqa: PLR0913, PLR0917 - mirrors the C entry point
        self,
        cfg: int,
        x: torch.Tensor,
        w: torch.Tensor,
        padding: tuple[int, int] | list[int],
        stride: int,
        e: S2m2EpiArgs,
    ) -> bool:
        """Run a conv; x `[N, C, H, W]` and w `[K, C, R, S]` channels-last fp16.

        Returns:
            `True` if it ran; `False` if the kernel does not support the geometry (the caller then runs its validated
            cuDNN path; counted in `fastforks._core.kernels.fallbacks()`).

        Raises:
            RuntimeError: The kernel failed (a CUDA error).
        """
        n, c, h, w_ = x.shape
        k, _, r, s = w.shape
        rc = self.lib.s2m2_conv(
            cfg,
            x.data_ptr(),
            w.data_ptr(),
            n,
            h,
            w_,
            c,
            k,
            r,
            s,
            padding[0],
            padding[1],
            stride,
            ctypes.byref(e),
            current_stream(),
        )
        if rc == 0:
            return True
        geometry = f"{n}x{c}x{h}x{w_} -> {k}, {r}x{s} stride {stride}"
        if rc == 1:
            note_fallback("s2m2_cutlass", f"cfg {cfg} conv {geometry}")
            return False
        msg = f"s2m2_cutlass: conv cfg {cfg} ({geometry}) failed with CUDA error {rc}"
        raise RuntimeError(msg)


@functools.cache
def _load(device: torch.device) -> CutlassConvs:
    return CutlassConvs(device)


def load(device: torch.device) -> CutlassConvs:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
