"""Fast-FoundationStereo kernels (`ffs_cutlass.cu`): fused CUTLASS convolutions and the network's own kernels.

- `conv2d` / `conv3d`: implicit GEMM with fp16 accumulation; bias, activation (`ACT_*`) and an optional residual in
  the epilogue; the output may be a channel slice of a wider channels-last buffer (no concatenation copies).
  `conv3d_subpixel` runs one parity class of a stride-2 transposed 3-D conv (written straight into its output voxels).
- feature kernels: input packing to 8 channels, a direct small-K conv for 8-channel inputs, depthwise conv, row
  LayerNorm / L2 normalization.
- cost volume (group-wise correlation + concatenation in one pass), classifier soft-argmax, trilinear upsample-add,
  geometry lookup (2 levels, level 1 pooled on the fly) into the motion encoder's input, GRU gates, context upsample.

The problem size is a runtime argument: nothing is compiled or tuned per input size. Tile configs follow fixed rules
in the source (no timing); the SM count is read once per process.
"""

from __future__ import annotations

import ctypes
import functools
from pathlib import Path

import torch

from fastforks._core.kernels import cuda_device, current_stream, shared_library

ACT_NONE, ACT_RELU, ACT_GELU, ACT_SIGMOID, ACT_TANH, ACT_LEAKY = 0, 1, 5, 7, 8, 9  # LeakyReLU(0.01); GELU (erf)
_V = 8  # channels per 16-byte fp16 vector
_LN_MAX = 512  # widest ln_rows row
_TAPS = 9  # 3x3 context-upsampling taps
_MAX_BYTES = 1 << 31  # CUTLASS 2-D implicit GEMM: every tensor < 2 GiB (32-bit offsets)
_MAX_ELEMS3 = 1 << 31  # 3-D convs: int64 offsets in the kernel, every tensor < 2^31 elements (4 GiB fp16)
CL, CL3 = torch.channels_last, torch.channels_last_3d
_I, _P, _L, _F = ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_float
_ARGS = {
    "ffs_conv2d": [_I, _P, _P, *[_I] * 11, _P, _P, _P, _I, _I, _P],
    "ffs_conv3d": [_I, _P, _P, *[_I] * 15, _P, _P, _P, _P, *[_I] * 9, _P],
    "ffs_input_nhwc8": [_P, _I, _P, _I, _I, _I, _I, _P],
    "ffs_stem_c8": [_P, _P, _P, _P, *[_I] * 11, _P],
    "ffs_dwconv": [_P, _P, _P, _P, *[_I] * 11, _P],
    "ffs_ln_rows": [_P, _P, _P, _P, _L, _I, _F, _I, _P],
    "ffs_build_volume": [_P, _P, _P, _P, _P, *[_I] * 8, _P],
    "ffs_up3_add": [_P, _P, _P, *[_I] * 8, _P],
    "ffs_cls_regress": [_P, _P, _I, _I, _L, _F, _P],
    "ffs_ctx_up": [_P, _L, _L, _L, _L, _P, _P, _I, _I, _I, _I, _I, _F, _P],
    "ffs_geo_lookup": [*[_P] * 8, *[_I] * 13, _P],
    "ffs_gru_rh": [*[_P] * 6, _L, _I, _I, _I, _P],
    "ffs_gru_out": [*[_P] * 7, _L, _I, _I, _I, _P],
}


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else t.data_ptr()


def _check(name: str, rc: int, what: str) -> None:
    if rc:
        msg = f"ffs_cutlass.{name}: {what} failed with code {rc} (1: unsupported geometry, 2 / 3: CUDA error)"
        raise RuntimeError(msg)


def _need(ok: bool, name: str, what: str) -> None:  # noqa: FBT001 - a condition
    if not ok:
        msg = f"ffs_cutlass.{name}: {what}"
        raise RuntimeError(msg)


def _big(*ts: torch.Tensor | None) -> bool:
    return any(t is not None and t.numel() * 2 >= _MAX_BYTES for t in ts)


def _big3(*ts: torch.Tensor | None) -> bool:
    return any(t is not None and t.numel() >= _MAX_ELEMS3 for t in ts)


def _ld(y: torch.Tensor, rank: int) -> int:
    """Pixel stride of `y` (`[N, K, ...]`, channel slice of a channels-last buffer), checked."""
    ld = y.stride(-1)
    ok = y.stride(1) == 1 and ld % 8 == 0 and y.data_ptr() % 16 == 0 and y.dtype == torch.float16
    expect = ld
    for d in range(rank - 1, 1, -1):
        ok = ok and y.stride(d) == expect
        expect *= y.shape[d]
    _need(ok and (y.shape[0] == 1 or y.stride(0) == expect), "conv", "out must be a channel slice (16-byte aligned)")
    return ld


class FfsCutlass:
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
        for name, args in _ARGS.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int

    # -------------------------------------------------------------------------------------------- convolutions
    def conv2d(  # noqa: PLR0913 - the kernel's arguments
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        *,
        pad: tuple[int, int] = (0, 0),
        stride: tuple[int, int] = (1, 1),
        act: int = ACT_NONE,
        res: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        cfg: int = -1,
    ) -> torch.Tensor:
        """`act(conv(x, w) + b (+ res))` into a new channels-last tensor or into `out`.

        `x` `[N, C, H, W]` channels-last fp16, `w` `[K, C, R, S]` channels-last fp16, `b` `[K]` fp16, `res` like the
        output (channels-last), `out` a `[N, K, P, Q]` channel slice of a channels-last buffer; 8 | C, K. `cfg` < 0:
        tile config by the fixed rule. A batch with a tensor >= 2 GiB runs one image per call.

        Raises:
            RuntimeError: Bad layout, unsupported geometry, or a CUDA error.
        """
        n, c, h, wd = x.shape
        k, _, r, s = w.shape
        ph, pw = pad
        sh, sw = stride
        p, q = (h + 2 * ph - r) // sh + 1, (wd + 2 * pw - s) // sw + 1
        _need(
            c % 8 == 0
            and k % 8 == 0
            and w.shape[1] == c
            and x.dtype == w.dtype == torch.float16
            and x.is_contiguous(memory_format=CL)
            and w.is_contiguous(memory_format=CL)
            and (res is None or (res.shape == (n, k, p, q) and res.is_contiguous(memory_format=CL))),
            "conv2d",
            f"needs channels-last fp16 with 8 | C, K; got {tuple(x.shape)} * {tuple(w.shape)}",
        )
        if out is None:
            out = torch.empty((n, k, p, q), device=x.device, dtype=x.dtype, memory_format=CL)
        _need(out.shape == (n, k, p, q), "conv2d", f"out {tuple(out.shape)} != {(n, k, p, q)}")
        ld = _ld(out, 4)
        if _big(x, res) or out.storage_offset() + n * p * q * ld >= _MAX_BYTES // 2:
            _need(n > 1, "conv2d", f"{tuple(x.shape)} exceeds the 2 GiB per-tensor limit; use a smaller input")
            for i in range(n):
                ri = None if res is None else res[i : i + 1]
                self.conv2d(x[i : i + 1], w, b, pad=pad, stride=stride, act=act, res=ri, out=out[i : i + 1], cfg=cfg)
            return out
        rc = self.lib.ffs_conv2d(
            cfg, x.data_ptr(), w.data_ptr(), n, h, wd, c, k, r, s, ph, pw, sh, sw,
            _ptr(b), _ptr(res), out.data_ptr(), ld, act, current_stream(),
        )  # fmt: skip
        _check("conv2d", rc, f"{tuple(x.shape)} * {tuple(w.shape)} (cfg {cfg})")
        return out

    def stem_c8(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, x: torch.Tensor, wt: torch.Tensor, b: torch.Tensor | None, k: int, rs: tuple[int, int],
        pad: tuple[int, int], act: int = ACT_NONE,
    ) -> torch.Tensor:  # fmt: skip
        """Direct stride-1 conv of an 8-channel channels-last fp16 input to K <= 16 outputs (8 | K).

        `wt`: the `[K, 8, R, S]` weight as `[K / 8][R][S][8 in][8 out]` (`stem_c8_weight`).

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        n, c, h, wd = x.shape
        r, s = rs
        _need(c == _V and x.is_contiguous(memory_format=CL) and x.dtype == torch.float16, "stem_c8", "8-channel input")
        p, q = h + 2 * pad[0] - r + 1, wd + 2 * pad[1] - s + 1
        y = torch.empty((n, k, p, q), device=x.device, dtype=x.dtype, memory_format=CL)
        rc = self.lib.ffs_stem_c8(
            x.data_ptr(), wt.data_ptr(), _ptr(b), y.data_ptr(), n, h, wd, k, r, s, *pad, 1, 1, act, current_stream()
        )
        _check("stem_c8", rc, f"{tuple(x.shape)} -> {k}")
        return y

    def conv3d(  # noqa: PLR0913 - the kernel's arguments
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor | None,
        *,
        pad: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        act: int = ACT_NONE,
        res: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
        cfg: int = -1,
    ) -> torch.Tensor:
        """`act(conv3d(x, w) + b (+ res)) (* gate)`; channels-last-3d fp16 in and out, 8 | C, K.

        `res` is channels-last-3d; `gate` `[N, K, P, Q]` channels-last fp16 multiplies every depth slice of the output
        after the activation (TAO's `FeatureAtt`).

        A batch with a tensor of >= 2^31 elements (4 GiB) runs one volume per call.

        Raises:
            RuntimeError: Bad layout, unsupported geometry, or a CUDA error.
        """
        n, c, d, h, wd = x.shape
        k, _, t, r, s = w.shape
        z, p, q = (
            (i + 2 * pp - kk) // st + 1 for i, pp, kk, st in zip((d, h, wd), pad, (t, r, s), stride, strict=True)
        )
        _need(
            c % 8 == 0
            and k % 8 == 0
            and w.shape[1] == c
            and x.dtype == w.dtype == torch.float16
            and x.is_contiguous(memory_format=CL3)
            and w.is_contiguous(memory_format=CL3)
            and (res is None or (res.shape == (n, k, z, p, q) and res.is_contiguous(memory_format=CL3)))
            and (
                gate is None
                or (gate.shape == (n, k, p, q) and gate.dtype == x.dtype and gate.is_contiguous(memory_format=CL))
            ),
            "conv3d",
            f"needs channels-last-3d fp16 with 8 | C, K; got {tuple(x.shape)} * {tuple(w.shape)}",
        )
        y = torch.empty((n, k, z, p, q), device=x.device, dtype=x.dtype, memory_format=CL3)
        if _big3(x, y, res):
            _need(n > 1, "conv3d", f"{tuple(x.shape)} exceeds the 2^31-element per-tensor limit; use a smaller input")
            for i in range(n):
                ri = None if res is None else res[i : i + 1]
                gi = None if gate is None else gate[i : i + 1]
                y[i : i + 1] = self.conv3d(
                    x[i : i + 1], w, b, pad=pad, stride=stride, act=act, res=ri, gate=gi, cfg=cfg
                )
            return y
        rc = self.lib.ffs_conv3d(
            cfg, x.data_ptr(), w.data_ptr(), n, d, h, wd, c, k, t, r, s, *pad, *stride,
            _ptr(b), _ptr(res), _ptr(gate), y.data_ptr(), k, act, 0, 0, 0, 0, 1, 1, 1, current_stream(),
        )  # fmt: skip
        _check("conv3d", rc, f"{tuple(x.shape)} * {tuple(w.shape)} (cfg {cfg})")
        return y

    def conv3d_subpixel(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, act: int, out: torch.Tensor,
        offset: tuple[int, int, int],
    ) -> None:  # fmt: skip
        """One parity class of a stride-2 transposed conv: a 2x2x2 conv (padding 1) of `x` into part of `out`.

        Writes the output voxels `(2 z + offset)` of `out` (`[N, K, Do, Ho, Wo]` channels-last-3d; out-of-range
        voxels skipped).

        Raises:
            RuntimeError: Bad layout, unsupported geometry, or a CUDA error.
        """
        n, c, d, h, wd = x.shape
        k = w.shape[0]
        _need(
            c % 8 == 0
            and k % 8 == 0
            and x.is_contiguous(memory_format=CL3)
            and out.is_contiguous(memory_format=CL3)
            and out.shape[:2] == (n, k),
            "conv3d_subpixel",
            f"needs channels-last-3d fp16 with 8 | C, K; got {tuple(x.shape)} * {tuple(w.shape)}",
        )
        if _big3(x, out):
            _need(n > 1, "conv3d_subpixel", f"{tuple(x.shape)} exceeds the 2^31-element per-tensor limit")
            for i in range(n):
                self.conv3d_subpixel(x[i : i + 1], w, b, act, out[i : i + 1], offset)
            return
        rc = self.lib.ffs_conv3d(
            -1, x.data_ptr(), w.data_ptr(), n, d, h, wd, c, k, 2, 2, 2, 1, 1, 1, 1, 1, 1,
            _ptr(b), 0, 0, out.data_ptr(), k, act, 1, *offset, *out.shape[2:], current_stream(),
        )  # fmt: skip
        _check("conv3d_subpixel", rc, f"{tuple(x.shape)} * {tuple(w.shape)}")

    # -------------------------------------------------------------------------------------------- features
    def input_nhwc8(self, x: torch.Tensor) -> torch.Tensor:
        """`[N, C <= 8, H, W]` fp32 / fp16 (contiguous) -> `[N, 8, H, W]` channels-last fp16, zero channels appended.

        Raises:
            RuntimeError: More than 8 channels or a CUDA error.
        """
        x = x.contiguous()
        n, c, h, w = x.shape
        _need(c <= _V and x.dtype in {torch.float32, torch.float16}, "input_nhwc8", "C <= 8, fp32 / fp16")
        y = torch.empty((n, 8, h, w), device=x.device, dtype=torch.float16, memory_format=CL)
        rc = self.lib.ffs_input_nhwc8(
            x.data_ptr(), int(x.dtype == torch.float32), y.data_ptr(), n, c, h, w, current_stream()
        )
        _check("input_nhwc8", rc, str(tuple(x.shape)))
        return y

    def dwconv(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, pad: tuple[int, int]) -> torch.Tensor:
        """Stride-1 depthwise conv; `x` channels-last fp16 (8 | C), `w` `[R, S, C]` fp16, `b` `[C]` fp16.

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        n, c, h, wd = x.shape
        r, s, _ = w.shape
        _need(c % 8 == 0 and x.is_contiguous(memory_format=CL) and w.shape[2] == c, "dwconv", "channels-last, 8 | C")
        p, q = h + 2 * pad[0] - r + 1, wd + 2 * pad[1] - s + 1
        y = torch.empty((n, c, p, q), device=x.device, dtype=x.dtype, memory_format=CL)
        rc = self.lib.ffs_dwconv(
            x.data_ptr(), w.data_ptr(), _ptr(b), y.data_ptr(), n, h, wd, c, r, s, *pad, 1, 1, 0, current_stream()
        )
        _check("dwconv", rc, str(tuple(x.shape)))
        return y

    def ln_rows(self, x: torch.Tensor, w: torch.Tensor | None, b: torch.Tensor | None, eps: float) -> torch.Tensor:
        """LayerNorm over the last dim of contiguous fp16 rows (8 | C <= 512, fp32 statistics); `w` None: L2 norm.

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        c = x.shape[-1]
        _need(
            x.is_contiguous() and x.dtype == torch.float16 and c % _V == 0 and c <= _LN_MAX, "ln_rows", "8 | C <= 512"
        )
        y = torch.empty_like(x)
        rc = self.lib.ffs_ln_rows(
            x.data_ptr(), _ptr(w), _ptr(b), y.data_ptr(), x.numel() // c, c, eps, int(w is None), current_stream()
        )
        _check("ln_rows", rc, str(tuple(x.shape)))
        return y

    # -------------------------------------------------------------------------------------------- volume
    def build_volume(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, nl: torch.Tensor, nr: torch.Tensor, lp: torch.Tensor, rp: torch.Tensor, d: int, groups: int, *,
        normalize: bool,
    ) -> torch.Tensor:  # fmt: skip
        """`cat([group-wise correlation, left projection, shifted right projection])`, channels-last-3d fp16.

        Output `[B, G + 2 Cc, D, H, W]`; `nl`, `nr` `[B, H, W, CF]`, `lp`, `rp` `[B, H, W, Cc]` contiguous fp16.

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        bsz, h, w, cf = nl.shape
        cc = lp.shape[3]
        _need(
            all(t.is_contiguous() and t.dtype == torch.float16 for t in (nl, nr, lp, rp)) and cf % groups == 0,
            "build_volume",
            "contiguous fp16 [B, H, W, C]",
        )
        y = torch.empty((bsz, groups + 2 * cc, d, h, w), device=nl.device, dtype=nl.dtype, memory_format=CL3)
        rc = self.lib.ffs_build_volume(
            nl.data_ptr(), nr.data_ptr(), lp.data_ptr(), rp.data_ptr(), y.data_ptr(),
            bsz, h, w, d, groups, cf // groups, cc, int(normalize), current_stream(),
        )  # fmt: skip
        _check("build_volume", rc, f"{tuple(nl.shape)} D {d}")
        return y

    def up3_add(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """`trilinear_upsample(x, size of r, align_corners=False) + r`, channels-last-3d fp16, 8 | C.

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        _need(
            x.is_contiguous(memory_format=CL3) and r.is_contiguous(memory_format=CL3) and x.shape[:2] == r.shape[:2],
            "up3_add",
            "channels-last-3d, same N, C",
        )
        y = torch.empty_like(r, memory_format=CL3)
        rc = self.lib.ffs_up3_add(x.data_ptr(), r.data_ptr(), y.data_ptr(), *x.shape, *r.shape[2:], current_stream())
        _check("up3_add", rc, f"{tuple(x.shape)} -> {tuple(r.shape)}")
        return y

    def cls_regress(self, o: torch.Tensor, b: int, d: int, bias: float) -> torch.Tensor:
        """Soft-argmax disparity `[B, H, W]` fp32 of the classifier taps `o` (`[B * D, 8, H, W]` channels-last fp16).

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        _need(o.is_contiguous(memory_format=CL) and o.shape[1] == _V and o.shape[0] == b * d, "cls_regress", "layout")
        h, w = o.shape[2:]
        y = torch.empty((b, h, w), device=o.device, dtype=torch.float32)
        rc = self.lib.ffs_cls_regress(o.data_ptr(), y.data_ptr(), b, d, h * w, bias, current_stream())
        _check("cls_regress", rc, str(tuple(o.shape)))
        return y

    def ctx_up(self, logits: torch.Tensor, disp: torch.Tensor, scale: float) -> torch.Tensor:
        """Context upsample: `sum softmax(logits) * 3x3 neighbourhood of scale * disp`, `[B, H4, W4]` fp32.

        `logits` `[B, 9, H4, W4]` fp16 (any strides), `disp` `[B, h, w]` fp32 contiguous.

        Raises:
            RuntimeError: Bad layout or a CUDA error.
        """
        b, c, h4, w4 = logits.shape
        _need(c == _TAPS and logits.dtype == torch.float16 and disp.is_contiguous(), "ctx_up", "9 fp16 logits")
        y = torch.empty((b, h4, w4), device=disp.device, dtype=torch.float32)
        rc = self.lib.ffs_ctx_up(
            logits.data_ptr(), *logits.stride(), disp.data_ptr(), y.data_ptr(), b, h4, w4, *disp.shape[1:], scale,
            current_stream(),
        )  # fmt: skip
        _check("ctx_up", rc, str(tuple(logits.shape)))
        return y

    # -------------------------------------------------------------------------------------------- GRU
    def geo_lookup(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, g0: torch.Tensor, c0: torch.Tensor, disp: torch.Tensor, out: torch.Tensor, d8: torch.Tensor,
        mf: torch.Tensor, *, k: int, cw: int, mfoff: int,
    ) -> None:  # fmt: skip
        """2-level geometry lookup (level 1 pooled on the fly) into `out` `[P, ldo]`.

        Layout `[geo l0 (k, c) | geo l1 | corr l0 (k) | corr l1]`; also writes `disp` (fp16) to `d8[:, 0]` and
        `mf[:, mfoff]`.

        `g0` `[B, D, H, W, Cg]` (first `cw` channels used), `c0` `[B, H, W, W]`, `disp` `[B, H, W]` fp32.

        Raises:
            RuntimeError: Bad layout (D, W must be even) or a CUDA error.
        """
        b, d, h, w, cg = g0.shape
        w0 = c0.shape[3]
        _need(
            g0.is_contiguous() and c0.is_contiguous() and disp.is_contiguous() and disp.dtype == torch.float32
            and d % 2 == 0 and w0 % 2 == 0 and out.is_contiguous() and mf.is_contiguous() and d8.shape[-1] == _V,
            "geo_lookup",
            "contiguous inputs, even D and W",
        )  # fmt: skip
        rc = self.lib.ffs_geo_lookup(
            g0.data_ptr(), 0, c0.data_ptr(), 0, disp.data_ptr(), out.data_ptr(), d8.data_ptr(), mf.data_ptr(),
            b, h, w, d, d // 2, w0, w0 // 2, cg, cw, k, out.shape[-1], mf.shape[-1], mfoff, current_stream(),
        )  # fmt: skip
        _check("geo_lookup", rc, str(tuple(g0.shape)))

    def gru_rh(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, zrs: torch.Tensor, zrl: torch.Tensor, h: torch.Tensor, hx: torch.Tensor, qs: torch.Tensor,
        ql: torch.Tensor, cx: int,
    ) -> None:  # fmt: skip
        """`qs = [r_small * h | x]`, `ql = [r_large * h | x]` (`zr*` `[P, 2 Hd]` = z | r, `x` = `hx[:, :cx]`).

        Raises:
            RuntimeError: Bad sizes or a CUDA error.
        """
        hd = h.shape[1]
        rc = self.lib.ffs_gru_rh(
            zrs.data_ptr(), zrl.data_ptr(), h.data_ptr(), hx.data_ptr(), qs.data_ptr(), ql.data_ptr(),
            h.shape[0], hd, cx, hx.shape[1], current_stream(),
        )  # fmt: skip
        _check("gru_rh", rc, f"Hd {hd}, Cx {cx}")

    def gru_out(  # noqa: PLR0913, PLR0917 - the kernel's arguments
        self, zrs: torch.Tensor, zrl: torch.Tensor, qs: torch.Tensor, ql: torch.Tensor, att: torch.Tensor,
        h: torch.Tensor, hx: torch.Tensor, hxoff: int,
    ) -> None:  # fmt: skip
        """`h = att ((1 - zs) h + zs qs) + (1 - att) ((1 - zl) h + zl ql)` in place, copied to `hx[:, hxoff:]`.

        Raises:
            RuntimeError: Bad sizes or a CUDA error.
        """
        rc = self.lib.ffs_gru_out(
            zrs.data_ptr(), zrl.data_ptr(), qs.data_ptr(), ql.data_ptr(), att.data_ptr(), h.data_ptr(),
            hx.data_ptr(), h.shape[0], h.shape[1], hx.shape[1], hxoff, current_stream(),
        )  # fmt: skip
        _check("gru_out", rc, f"Hd {h.shape[1]}")


def stem_c8_weight(w: torch.Tensor) -> torch.Tensor:
    """`[K, 8, R, S]` -> `[K / 8][R][S][8 in][8 out]` (`FfsCutlass.stem_c8`)."""
    k, c, r, s = w.shape
    return w.contiguous().view(k // 8, 8, c, r, s).permute(0, 3, 4, 2, 1).contiguous()


@functools.cache
def _load(device: torch.device) -> FfsCutlass:
    return FfsCutlass(device)


def load(device: torch.device) -> FfsCutlass:
    """Load the library for `device` (once per device).

    Raises:
        KernelUnavailableError: No usable binary for this GPU / driver.
    """
    return _load(cuda_device(device))
