"""CUTLASS implicit-GEMM convolutions with fused GRU / affine epilogues (`cutlass_conv.cu`; fp16 accumulation).

Shared by the RAFT-Stereo and IGEV-family kernels backends. The problem size is a runtime argument: nothing is compiled
or tuned per input size.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device, current_stream, note_fallback, shared_library

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch


class ConvEpiArgs(ctypes.Structure):
    """Epilogue arguments (`struct EpiArgs` in cutlass_conv.cu; modes documented there). Null pointers are 0."""

    _fields_ = (
        ("mode", ctypes.c_int),
        ("act", ctypes.c_int),
        ("c", ctypes.c_void_p),
        ("cs", ctypes.c_int),
        ("z", ctypes.c_void_p),
        ("zs", ctypes.c_int),
        ("h", ctypes.c_void_p),
        ("hs", ctypes.c_int),
        ("rh", ctypes.c_void_p),
        ("h2", ctypes.c_void_p),
        ("s", ctypes.c_void_p),
        ("t", ctypes.c_void_p),
        ("coords", ctypes.c_void_p),
        ("W", ctypes.c_int),
        ("h2s", ctypes.c_int),  # row stride of h2; 0 = 128
    )


class FusedConvs:
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
        self.lib.fused_conv_rs.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *[ctypes.c_int] * 9,
            ctypes.POINTER(ConvEpiArgs),
            ctypes.c_void_p,
        ]

    def conv(  # noqa: PLR0913, PLR0917 - the kernel's argument list
        self,
        cfg: int,
        x: torch.Tensor,
        w: torch.Tensor,
        hw: tuple[int, int],
        e: ConvEpiArgs,
        stride: int = 1,
        padding: tuple[int, int] | None = None,
    ) -> bool:
        """Run a conv; x `[H, W, C]` fp16 (packed), w `[K, R, S, C]` fp16 (packed), padding default `(R//2, S//2)`.

        Returns:
            `True` if it ran; `False` if the kernel does not support the geometry (the caller then runs its validated
            cuDNN path; counted in `fastforks._core.kernels.fallbacks()`).

        Raises:
            RuntimeError: The kernel failed (a CUDA error).
        """
        return _result(self.lib.fused_conv_rs(*_args(cfg, x, w, hw, e, stride, padding), current_stream()), cfg, w, hw)

    def bind(  # noqa: PLR0913, PLR0917 - the kernel's argument list
        self,
        cfg: int,
        x: torch.Tensor,
        w: torch.Tensor,
        hw: tuple[int, int],
        e: ConvEpiArgs,
        stride: int = 1,
        padding: tuple[int, int] | None = None,
    ) -> Callable[[], None]:
        """A no-argument launcher (on the then-current stream) for repeated identical calls of `conv`.

        Raises (when called):
            RuntimeError: The kernel failed or does not support the geometry.
        """
        f, args = self.lib.fused_conv_rs, _args(cfg, x, w, hw, e, stride, padding)

        def call() -> None:
            if not _result(f(*args, current_stream()), cfg, w, hw):
                msg = f"fused_conv cfg {cfg}: geometry {_geometry(w, hw)} not supported"
                raise RuntimeError(msg)

        call.keep = (e, x, w)  # ty: ignore[unresolved-attribute] - the struct and buffers must outlive the launcher
        return call


def _geometry(w: torch.Tensor, hw: tuple[int, int]) -> str:
    k, r, s, c = w.shape
    return f"{hw[0]}x{hw[1]}x{c} -> {k}, {r}x{s}"


def _result(rc: int, cfg: int, w: torch.Tensor, hw: tuple[int, int]) -> bool:
    """0 -> ran, 1 -> unsupported geometry (counted), else raise the CUDA error."""
    if rc == 0:
        return True
    if rc == 1:
        note_fallback("cutlass_conv", f"cfg {cfg} conv {_geometry(w, hw)}")
        return False
    msg = f"cutlass_conv: fused_conv cfg {cfg} ({_geometry(w, hw)}) failed with CUDA error {rc}"
    raise RuntimeError(msg)


def _args(  # noqa: PLR0913, PLR0917 - the kernel's argument list
    cfg: int,
    x: torch.Tensor,
    w: torch.Tensor,
    hw: tuple[int, int],
    e: ConvEpiArgs,
    stride: int,
    padding: tuple[int, int] | None,
) -> tuple[object, ...]:
    k, r, s, c = w.shape
    ph, pw = (r // 2, s // 2) if padding is None else padding
    return cfg, x.data_ptr(), w.data_ptr(), hw[0], hw[1], c, k, r, s, ph, pw, stride, ctypes.byref(e)


def narrow_cfg(pixels: int, k: int, device: torch.device) -> int:
    """Tile config (4-7) for a conv with `k < 128` output channels.

    Fixed rule from the `fastbb.normconv` measurements (RAFT / IGEV context encoders at Midd-H): 128x64 tiles
    for k <= 64, 128x128 for k = 96 / 128, smaller tiles when there are fewer than two waves of CTAs.
    """
    sms = _sm_count(cuda_device(device))

    def tiles(tm: int, tn: int) -> int:
        return -(-pixels // tm) * -(-k // tn)

    if k <= 64:  # noqa: PLR2004 - 64-channel convs
        return 4 if tiles(128, 64) >= 2 * sms else 7
    if tiles(128, 128) >= 2 * sms:
        return 5
    return 6 if tiles(64, 128) >= 2 * sms else 7


@functools.cache
def _sm_count(device: torch.device) -> int:
    import torch  # noqa: PLC0415 - only needed at run time

    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _load(device: torch.device) -> FusedConvs:
    return FusedConvs(device)


def load(device: torch.device) -> FusedConvs:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
