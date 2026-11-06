"""CUTLASS implicit-GEMM convolutions with fused FoundationStereo epilogues (`fs_cutlass.cu`; fp16 accumulation).

2-D (any R x S filter) and 3-D (channels-last-3d) convolutions. The problem size is a runtime argument: nothing is
compiled or tuned per input size. Modes (`FsEpiArgs.mode`) are documented in the source.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device, current_stream, note_fallback, shared_library

_DIM2 = 4  # rank of a 2-D convolution weight

if TYPE_CHECKING:
    import torch


class FsEpiArgs(ctypes.Structure):
    """Epilogue arguments (`struct EpiArgs` in fs_cutlass.cu; modes documented there)."""

    _fields_ = (
        ("mode", ctypes.c_int),
        ("act", ctypes.c_int),
        ("b", ctypes.c_void_p),
        ("o", ctypes.c_void_p),
        ("os", ctypes.c_int),
        ("o2", ctypes.c_void_p),
        ("o2s", ctypes.c_int),
        ("h", ctypes.c_void_p),
        ("hs", ctypes.c_int),
        ("z", ctypes.c_void_p),
        ("zs", ctypes.c_int),
        ("g", ctypes.c_void_p),
        ("gs", ctypes.c_int),
        ("att", ctypes.c_void_p),
        ("disp", ctypes.c_void_p),
        ("c", ctypes.c_void_p),
        ("cs", ctypes.c_int),
    )


class FsCutlass:
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
        p = ctypes.POINTER(FsEpiArgs)
        self.lib.fused_conv_rs.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *[ctypes.c_int] * 9,
            p,
            ctypes.c_void_p,
        ]
        self.lib.fused_conv3d.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            p,
            ctypes.c_void_p,
        ]

    def conv3d(  # noqa: PLR0913, PLR0917 - mirrors the kernel argument list
        self,
        cfg: int,
        x: torch.Tensor,
        w: torch.Tensor,
        e: FsEpiArgs,
        padding: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        *,
        unsupported_ok: bool = False,
    ) -> bool:
        """Run a 3-D conv; x `[N, C, D, H, W]` and w `[K, C, T, R, S]` channels-last-3d fp16.

        The output geometry follows padding / stride; results go only through the epilogue arguments.

        Returns:
            `True` if it ran; `False` only with `unsupported_ok` if the kernel does not support the geometry (the
            caller then runs its validated cuDNN path; counted in `fastforks._core.kernels.fallbacks()`).

        Raises:
            RuntimeError: The kernel failed (a CUDA error), or does not support the geometry and not `unsupported_ok`.
        """
        n, c, d, h, w_ = x.shape
        k, _, t, r, s = w.shape
        g = (ctypes.c_int * 15)(n, d, h, w_, c, k, t, r, s, *padding, *stride)
        rc = self.lib.fused_conv3d(cfg, x.data_ptr(), w.data_ptr(), g, ctypes.byref(e), current_stream())
        return rc == 0 or _failed(rc, f"conv3d cfg {cfg} x {tuple(x.shape)} w {tuple(w.shape)}", unsupported_ok)

    def conv(  # noqa: PLR0913, PLR0917 - mirrors the C entry point
        self,
        cfg: int,
        x: torch.Tensor,
        w: torch.Tensor,
        hw: tuple[int, int],
        c: int,
        k: int,
        e: FsEpiArgs,
        padding: tuple[int, int] | None = None,
        stride: int = 1,
    ) -> None:
        """Run one 2-D conv on the dense NHWC buffer `x` (`hw` = its H, W) with w `[K, R, S, C]` (NHWC filters).

        Raises:
            RuntimeError: The kernel does not support the geometry, or failed (a CUDA error).
        """
        r, s = (w.shape[1], w.shape[2]) if w.dim() == _DIM2 and w.shape[3] == c else (w.shape[2], w.shape[3])
        ph, pw = (r // 2, s // 2) if padding is None else padding
        rc = self.lib.fused_conv_rs(
            cfg, x.data_ptr(), w.data_ptr(), hw[0], hw[1], c, k, r, s, ph, pw, stride, ctypes.byref(e), current_stream()
        )
        if rc:
            _failed(rc, f"conv cfg {cfg} {hw[0]}x{hw[1]}x{c} -> {k}, {r}x{s} (stride {stride}, pad {ph},{pw})")


_RC = {
    1: "geometry not supported",
    2: "could not set the kernel's dynamic shared memory size",
    3: "CUDA launch error",
    -1: "unknown tile config",
}


def _failed(rc: int, what: str, unsupported_ok: bool = False) -> bool:  # noqa: FBT001, FBT002 - internal helper
    """Status `rc` != 0 of `what`: False for an unsupported geometry if `unsupported_ok` (counted), else raise."""
    if rc == 1 and unsupported_ok:
        note_fallback("fs_cutlass", what)
        return False
    reason = _RC.get(rc, f"status {rc}")
    msg = f"fs_cutlass: {what}: {reason}"
    raise RuntimeError(msg)


@functools.cache
def _load(device: torch.device) -> FsCutlass:
    return FsCutlass(device)


def load(device: torch.device) -> FsCutlass:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
