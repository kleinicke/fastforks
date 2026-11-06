"""Fused CUTLASS convs of the segmentation CNNs (`seg_conv.cu`; fp16 NHWC, fp16 accumulation).

`act(conv(x, w) * s + t (+ res))` in one kernel, dilated convs included, the tile config picked by a fixed shape rule;
depthwise convs (any dilation) as a direct fp32-accumulate kernel with the same epilogue (`dwconv`).
The problem size is a runtime argument: nothing is compiled or tuned per input size.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path
from typing import TYPE_CHECKING

from fastforks._core.kernels import cuda_device, current_stream, note_fallback, shared_library

if TYPE_CHECKING:
    import torch

_ARGS = [ctypes.c_int, *[ctypes.c_void_p] * 6, *[ctypes.c_int] * 15, ctypes.c_void_p]
_DW_ARGS = [*[ctypes.c_void_p] * 5, *[ctypes.c_int] * 13, ctypes.c_void_p]


class SegConv:
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
        self.lib.seg_conv.argtypes = _ARGS
        self.lib.seg_conv.restype = ctypes.c_int
        self.lib.seg_dwconv.argtypes = _DW_ARGS
        self.lib.seg_dwconv.restype = ctypes.c_int

    def conv(  # noqa: PLR0913, PLR0917 - the kernel's argument list
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor | None,
        t: torch.Tensor | None,
        res: torch.Tensor | None,
        *,
        padding: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        act: int,
    ) -> bool:
        """`y = act(conv(x, w) * s + t (+ res))`; all dense channels-last fp16 (`s`, `t` fp32 `[K]`), act 0 / 1 (relu).

        Returns:
            `True` if it ran; `False` if CUTLASS does not support the geometry (counted in
            `fastforks._core.kernels.fallbacks()`; the caller then runs its validated cuDNN path).

        Raises:
            RuntimeError: The kernel failed (a CUDA error).
        """
        n, c, h, wd = x.shape
        k, _, r, sk = w.shape
        rc = self.lib.seg_conv(
            -1,
            x.data_ptr(),
            w.data_ptr(),
            None if s is None else s.data_ptr(),
            None if t is None else t.data_ptr(),
            None if res is None else res.data_ptr(),
            y.data_ptr(),
            n,
            h,
            wd,
            c,
            k,
            r,
            sk,
            *padding,
            *stride,
            *dilation,
            act,
            0,
            current_stream(),
        )
        if rc == 0:
            return True
        geometry = f"{h}x{wd}x{c} -> {k}, {r}x{sk} stride {stride} dilation {dilation}"
        if rc == 1:
            note_fallback("seg_conv", f"conv {geometry}")
            return False
        msg = f"seg_conv: conv {geometry} failed with status {rc}"
        raise RuntimeError(msg)

    def dwconv(  # noqa: PLR0913 - the kernel's argument list
        self,
        x: torch.Tensor,
        w_rsc: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor | None,
        t: torch.Tensor | None,
        *,
        padding: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        act: int,
    ) -> bool:
        """`y = act(dwconv(x, w) * s + t)`, depthwise (groups = C); dense channels-last fp16, `w_rsc` `[R, S, C]`.

        Returns:
            `True` if it ran; `False` for `C % 8 != 0` (counted in `fastforks._core.kernels.fallbacks()`).

        Raises:
            RuntimeError: The kernel failed (a CUDA error).
        """
        n, c, h, wd = x.shape
        r, sk = w_rsc.shape[:2]
        rc = self.lib.seg_dwconv(
            x.data_ptr(),
            w_rsc.data_ptr(),
            None if s is None else s.data_ptr(),
            None if t is None else t.data_ptr(),
            y.data_ptr(),
            n,
            h,
            wd,
            c,
            r,
            sk,
            *padding,
            *stride,
            *dilation,
            act,
            current_stream(),
        )
        if rc == 0:
            return True
        geometry = f"{h}x{wd}x{c}, {r}x{sk} stride {stride} dilation {dilation}"
        if rc == 1:
            note_fallback("seg_conv", f"dwconv {geometry}")
            return False
        msg = f"seg_conv: dwconv {geometry} failed with status {rc}"
        raise RuntimeError(msg)


@functools.cache
def _load(device: torch.device) -> SegConv:
    return SegConv(device)


def load(device: torch.device) -> SegConv:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
