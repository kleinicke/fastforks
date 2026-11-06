"""Fused CUTLASS convolutions for HITNet: conv + bias (+ residual) + LeakyReLU(0.2), with dilation (`hitnet_conv.cu`).

fp16 accumulation; the problem size is a runtime argument (nothing is compiled or tuned per input size).
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch

from fastforks._core.kernels import cuda_device, current_stream, shared_library

ACT_NONE = 0
ACT_LEAKY = 7  # LeakyReLU(0.2)
_MAX_BYTES = 1 << 31  # CUTLASS implicit GEMM: every tensor < 2 GiB (`can_implement`)


class HitnetConv:
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
        self.lib.hitnet_conv.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            *[ctypes.c_int] * 13,
            *[ctypes.c_void_p] * 3,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

    def conv(  # noqa: PLR0913, PLR0917 - the kernel's argument list
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
        res: torch.Tensor | None,
        y: torch.Tensor,
        padding: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        act: int,
    ) -> None:
        """Run `y = act(conv(x, w) + bias (+ res))`.

        `x` is channels-last fp16 `[N, C, H, W]`, `w` `[K, C, R, S]` (channels-last), `res` / `y` `[N, K, P, Q]`
        (channels-last); C and K must be multiples of 8. A batch whose `x` or `y` is >= 2 GiB runs one image per call.

        Raises:
            RuntimeError: The tensors are not laid out as above, the kernel does not support the geometry, or it
                failed (a CUDA error).
        """
        n, c, h, wd = x.shape
        k, _, r, s = w.shape
        cl = torch.channels_last
        tensors = (x, w, y) if res is None else (x, w, y, res)
        if (
            c % 8
            or k % 8
            or w.shape[1] != c
            or bias.shape != (k,)
            or y.shape[:2] != (n, k)
            or (res is not None and res.shape != y.shape)
            or not all(t.dtype == torch.float16 and t.is_contiguous(memory_format=cl) for t in tensors)
            or bias.dtype != torch.float16
            or not bias.is_contiguous()
        ):
            msg = f"hitnet_conv: needs channels-last fp16 with 8 | C, K; got {tuple(x.shape)} * {tuple(w.shape)}"
            raise RuntimeError(msg)
        if max(t.numel() for t in (x, y)) * 2 >= _MAX_BYTES:
            if n == 1:
                msg = (
                    f"hitnet_conv: {tuple(x.shape)} * {tuple(w.shape)} exceeds the kernel's 2 GiB per-tensor limit "
                    "(CUTLASS 32-bit offsets); use a smaller input"
                )
                raise RuntimeError(msg)
            for i in range(n):  # one image per call (each tensor < 2 GiB)
                res_i = None if res is None else res[i : i + 1]
                self.conv(x[i : i + 1], w, bias, res_i, y[i : i + 1], padding, stride, dilation, act)
            return
        rc = self.lib.hitnet_conv(
            -1, x.data_ptr(), w.data_ptr(), n, h, wd, c, k, r, s, *padding, *stride, *dilation,
            bias.data_ptr(), 0 if res is None else res.data_ptr(), y.data_ptr(), act, current_stream(),
        )  # fmt: skip
        if rc:
            msg = f"hitnet_conv: {tuple(x.shape)} * {tuple(w.shape)} (dilation {dilation}) failed with code {rc}"
            raise RuntimeError(msg)


@functools.cache
def _load(device: torch.device) -> HitnetConv:
    return HitnetConv(device)


def load(device: torch.device) -> HitnetConv:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
