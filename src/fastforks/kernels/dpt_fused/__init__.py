"""Fused fp16 GEMMs / convolutions for DINOv2 ViT + DPT head networks (`dpt_fused.cu`; fp16 or fp32 accumulation).

CUTLASS GEMM and implicit-GEMM convolution with a visitor epilogue (`act((acc + bias) * scale + res + res2)`, an
optional second `relu` output, a deconvolution pixel shuffle), plus the bicubic patch input, NHWC bilinear and output
tail kernels. The problem size is a runtime argument; the tile configuration follows a fixed rule (`gemm_cfg`,
`conv_cfg`). A problem the kernels cannot run raises `ValueError` (no fallback).
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch

from fastforks._core.kernels import cuda_device, current_stream, half_rate_fp32_acc, shared_library

ACT_NONE, ACT_RELU, ACT_GELU, ACT_GELU_TANH = 0, 1, 5, 6
"""Epilogue activations; `ACT_GELU` is the exact (erf) GELU, `ACT_GELU_TANH` its tanh form (`tanh.approx.f32`). Models
take `DptFused.gelu`, which picks one per card class (see there)."""
_P = ctypes.c_void_p
_I = ctypes.c_int


class EpiArgs(ctypes.Structure):
    """Epilogue arguments (`struct EpiArgs` in dpt_fused.cu)."""

    _fields_ = (
        ("bias", _P),
        ("scale", _P),
        ("res", _P),
        ("lres", _I),
        ("res2", _P),
        ("lres2", _I),
        ("y", _P),
        ("ldy", _I),
        ("y2", _P),
        ("ldy2", _I),
        ("act", _I),
        ("sh", _I),
        ("sw", _I),
        ("gh", _I),
        ("gw", _I),
        ("cout", _I),
    )


def _ptr(t: torch.Tensor | None) -> int | None:
    return None if t is None else t.data_ptr()


def _rows(t: torch.Tensor, m: int, n: int, what: str) -> int:
    """Row stride of a `[m, n]` fp16 matrix with unit column stride (any leading shape flattened)."""
    if t.dtype != torch.float16 or t.stride(-1) != 1 or t.numel() != m * n:
        msg = f"dpt_fused: {what} must be fp16 [{m}, {n}] with contiguous rows, got {tuple(t.shape)} {t.dtype}"
        raise ValueError(msg)
    try:
        return t.view(m, n).stride(0)
    except RuntimeError:
        msg = f"dpt_fused: {what} rows are not evenly strided: {tuple(t.shape)} / {t.stride()}"
        raise ValueError(msg) from None


class DptFused:
    """The loaded shared library.

    Attributes:
        tag: `name@build_id:target`.
        sms: Number of SMs of the device (tile rule).
        full_rate: The card runs fp32-accumulate tensor-core math at full rate (datacenter / pro cards, not GeForce).
        gelu: The GELU epilogue for this card: `ACT_GELU_TANH` on full-rate cards, which are power-capped and gain
            3-11 % end to end from its far fewer instructions (L4, A10); `ACT_GELU` (erf) on GeForce, where the tanh
            form gains ~1.5 % and quadruples the mean error vs float32 (RTX 5060 Ti, `tools/kernel_lab.py`). Its
            slight negative bias moves depth-pro's focal length and UniK3D's depth scale past their goldens: those
            two take `ACT_GELU` on every card.
    """

    def __init__(self, device: torch.device) -> None:
        """Load the library for `device`.

        Raises:
            KernelUnavailableError: No usable binary for this GPU / driver.
        """
        self.lib, self.tag = shared_library(Path(__file__).with_name("manifest.toml"), device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.full_rate = not half_rate_fp32_acc(device)
        self.gelu = ACT_GELU_TANH if self.full_rate else ACT_GELU
        e = ctypes.POINTER(EpiArgs)
        lib = self.lib
        lib.dpt_gemm.argtypes = [_I, _I, _I, _I, _P, _I, _P, e, _P]
        lib.dpt_conv.argtypes = [_I, _P, _P, ctypes.POINTER(_I), e, _P]
        lib.dpt_patches.argtypes = [_P, _I, _I, _I, _I, _I, ctypes.c_float, ctypes.c_float, _I, _I, _P, _P, _P]
        lib.dpt_bilinear.argtypes = [_P, _I, _I, _I, _P, _I, _I, _I, ctypes.c_float, ctypes.c_float, _I, _P]
        lib.dpt_head_dot.argtypes = [_P, ctypes.c_longlong, _I, _P, ctypes.c_float, _P, _P]

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc:
            reason = {1: "geometry / alignment not supported", -1: "unknown tile configuration"}.get(rc, "CUDA error")
            msg = f"dpt_fused: {what} failed ({reason}, status {rc})"
            raise ValueError(msg)

    def gemm_cfg(self, m: int, n: int, k: int, act: int) -> int:
        """Fixed tile rule (sustained sweeps of ViT-L blocks, `tools/kernel_lab.py`).

        erf GELU: 128x64. tanh GELU and the rest: on full-rate cards 128x256 for deep (K >= 1024) GEMMs of >= 4 waves
        of such tiles (the ViT blocks of many patches; slower on GeForce), else 128x128 for >= 2 waves, else narrower.
        """

        def tiles(tm: int, tn: int) -> int:
            return -(-m // tm) * -(-n // tn)

        if act == ACT_GELU:
            return 2
        if self.full_rate and act != ACT_GELU_TANH and k >= 1024 and tiles(128, 256) >= 4 * self.sms:  # noqa: PLR2004
            return 4
        if tiles(128, 128) >= 2 * self.sms:
            return 0
        return 2 if tiles(128, 64) >= self.sms else 3

    def conv_cfg(self, m: int, k: int) -> int:
        """Fixed tile rule (sweeps): 128x128 if K fills 128-wide tiles and the GPU, else narrower."""

        def tiles(tm: int, tn: int) -> int:
            return -(-m // tm) * -(-k // tn)

        if k <= 32:  # noqa: PLR2004 - narrow output (DPT tail)
            return 4 if tiles(128, 32) >= self.sms else 3
        if -(-k // 128) * 128 <= 1.1 * k:
            if tiles(128, 128) >= self.sms:
                return 0
            if tiles(64, 128) >= self.sms:
                return 1
            return 3
        return 2 if tiles(128, 64) >= self.sms else 3

    def _epi(  # noqa: PLR0913, PLR0917 - the epilogue terms
        self,
        m: int,
        n: int,
        y: torch.Tensor,
        bias: torch.Tensor | None,
        scale: torch.Tensor | None,
        res: torch.Tensor | None,
        res2: torch.Tensor | None,
        y2: torch.Tensor | None,
        act: int,
    ) -> EpiArgs:
        if bias is not None and (bias.dtype != torch.float16 or not bias.is_contiguous() or bias.numel() != n):
            msg = f"dpt_fused: bias must be contiguous fp16 [{n}]"
            raise ValueError(msg)
        if scale is not None and (scale.dtype != torch.float32 or not scale.is_contiguous() or scale.numel() != n):
            msg = f"dpt_fused: scale must be contiguous fp32 [{n}]"
            raise ValueError(msg)
        return EpiArgs(
            _ptr(bias),
            _ptr(scale),
            _ptr(res),
            0 if res is None else _rows(res, m, n, "res"),
            _ptr(res2),
            0 if res2 is None else _rows(res2, m, n, "res2"),
            y.data_ptr(),
            _rows(y, m, n, "out"),
            _ptr(y2),
            0 if y2 is None else _rows(y2, m, n, "y2"),
            act,
            0,
            0,
            0,
            0,
            0,
        )

    def gemm(  # noqa: PLR0913 - the epilogue terms
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        act: int = ACT_NONE,
        scale: torch.Tensor | None = None,
        res: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        shuffle: tuple[int, int, int, int] | None = None,
        acc32: bool = False,
    ) -> torch.Tensor:
        """`epilogue(x @ w^T)` for x `[..., K]` (rows contiguous) and w `[N, K]` (contiguous), fp16.

        `out` (optional, may alias `res`) gets the result, else a new `[..., N]` tensor. `shuffle = (gh, gw, sh, sw)`:
        x rows are a `[B, gh, gw]` grid and N = sh * sw * cout; the result is the NHWC `[B, gh * sh, gw * sw, cout]`
        image of the kernel == stride deconvolution. `acc32` accumulates in fp32 (precision-critical layers).
        """
        k = x.shape[-1]
        m = x.numel() // k
        n = w.shape[0]
        lda = _rows(x, m, k, "x")
        if w.dtype != torch.float16 or not w.is_contiguous() or w.shape[1] != k:
            msg = f"dpt_fused: weight must be contiguous fp16 [N, {k}], got {tuple(w.shape)}"
            raise ValueError(msg)
        if shuffle is None:
            y = out if out is not None else x.new_empty((*x.shape[:-1], n))
            e = self._epi(m, n, y, bias, scale, res, None, None, act)
        else:
            gh, gw, sh, sw = shuffle
            cout = n // (sh * sw)
            b = m // (gh * gw)
            y = out if out is not None else x.new_empty((b, gh * sh, gw * sw, cout))
            if not y.is_contiguous() or y.numel() != m * n:
                msg = "dpt_fused: shuffle output must be a contiguous NHWC tensor"
                raise ValueError(msg)
            e = self._epi(m, n, y, bias, scale, None, None, None, act)
            e.ldy, e.sh, e.sw, e.gh, e.gw, e.cout = cout, sh, sw, gh, gw, cout
        cfg = (5 if -(-m // 128) * -(-n // 64) >= self.sms else 6) if acc32 else self.gemm_cfg(m, n, k, act)
        self._check(self.lib.dpt_gemm(cfg, m, n, k, x.data_ptr(), lda, w.data_ptr(), e, current_stream()), "gemm")
        return y

    def conv(  # noqa: PLR0913 - the epilogue terms
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        padding: int = 0,
        stride: int = 1,
        act: int = ACT_NONE,
        res: torch.Tensor | None = None,
        res2: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        y2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convolution of a contiguous NHWC fp16 `x` `[B, H, W, C]` with w `[K, R, S, C]` -> NHWC `[B, P, Q, K]`.

        `y2` (optional, same shape as the result) also gets `relu(result)`.
        """
        if not x.is_contiguous() or x.dtype != torch.float16 or x.dim() != 4:  # noqa: PLR2004
            msg = f"dpt_fused: conv input must be contiguous NHWC fp16, got {tuple(x.shape)} {x.dtype}"
            raise ValueError(msg)
        b, h, w_, c = x.shape
        k, r, s, c2 = w.shape
        if c2 != c or not w.is_contiguous() or w.dtype != torch.float16:
            msg = f"dpt_fused: conv weight must be contiguous fp16 [K, R, S, {c}], got {tuple(w.shape)}"
            raise ValueError(msg)
        p = (h + 2 * padding - r) // stride + 1
        q = (w_ + 2 * padding - s) // stride + 1
        y = out if out is not None else x.new_empty((b, p, q, k))
        m = b * p * q
        e = self._epi(m, k, y, bias, None, res, res2, y2, act)
        g = (_I * 11)(b, h, w_, c, k, r, s, padding, padding, stride, stride)
        rc = self.lib.dpt_conv(self.conv_cfg(m, k), x.data_ptr(), w.data_ptr(), g, e, current_stream())
        self._check(rc, f"conv {tuple(x.shape)} * {tuple(w.shape)}")
        return y

    def patches(  # noqa: PLR0913, PLR0917 - geometry + normalisation
        self,
        image: torch.Tensor,
        size: tuple[int, int],
        patch: int,
        kp: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> torch.Tensor:
        """Bicubic resize of float `[B, 3, H, W]` to `size`, normalise, as fp16 patch rows `[B, gh * gw, kp]`."""
        image = image.float().contiguous()
        b, _, hh, ww = image.shape
        h, w = size
        out = torch.empty((b, (h // patch) * (w // patch), kp), device=image.device, dtype=torch.float16)
        ms = (ctypes.c_float * 6)(*mean, *std)
        scy, scx = (torch.tensor([hh, ww], dtype=torch.float32) / torch.tensor([h, w], dtype=torch.float32)).tolist()
        rc = self.lib.dpt_patches(
            image.data_ptr(), b, hh, ww, h, w, scy, scx, patch, kp, ms, out.data_ptr(), current_stream()
        )
        self._check(rc, "patches")
        return out

    def bilinear(self, x: torch.Tensor, size: tuple[int, int], *, align_corners: bool) -> torch.Tensor:
        """`F.interpolate(mode="bilinear")` of a contiguous NHWC fp16 tensor `[B, H, W, C]`."""
        if not x.is_contiguous() or x.dtype != torch.float16:
            msg = "dpt_fused: bilinear input must be contiguous NHWC fp16"
            raise ValueError(msg)
        b, hs, ws, c = x.shape
        hd, wd = size
        y = x.new_empty((b, hd, wd, c))
        f32 = torch.float32
        if align_corners:
            sh = (torch.tensor(hs - 1, dtype=f32) / (hd - 1)).item() if hd > 1 else 0.0
            sw = (torch.tensor(ws - 1, dtype=f32) / (wd - 1)).item() if wd > 1 else 0.0
        else:
            sh = (torch.tensor(hs, dtype=f32) / hd).item()
            sw = (torch.tensor(ws, dtype=f32) / wd).item()
        rc = self.lib.dpt_bilinear(
            x.data_ptr(), b, hs, ws, y.data_ptr(), hd, wd, c, sh, sw, int(align_corners), current_stream()
        )
        self._check(rc, "bilinear")
        return y

    def head_dot(self, x: torch.Tensor, w: torch.Tensor, b: float) -> torch.Tensor:
        """`relu(x @ w + b)` in fp32 for a contiguous NHWC fp16 `[B, H, W, C]` and fp32 `w` `[C]` -> `[B, H, W]`."""
        if not x.is_contiguous() or x.dtype != torch.float16 or w.dtype != torch.float32 or w.numel() != x.shape[-1]:
            msg = "dpt_fused: head_dot needs contiguous NHWC fp16 x and fp32 w [C]"
            raise ValueError(msg)
        out = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
        rc = self.lib.dpt_head_dot(
            x.data_ptr(), out.numel(), x.shape[-1], w.data_ptr(), b, out.data_ptr(), current_stream()
        )
        self._check(rc, "head_dot")
        return out


@functools.cache
def _load(device: torch.device) -> DptFused:
    return DptFused(device)


def load(device: torch.device) -> DptFused:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
