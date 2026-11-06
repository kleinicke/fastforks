"""fastbb kernels: flash attention (head_dim 64), NHWC instance norm, window attention, Restormer kernels, DINOv3 RoPE.

Flash attention with optional fp16 accumulation (`Flash64`: the one launcher, with the overflow fallback), window
attention with relative-position embeddings. `fastbb_kernels.cu`, from the `fastbb` backbone package and our NMRF
fork. Used by the DINOv2 ViT fast path (`fastforks.shared.depth_anything_v2`), the fp16 instance-norm statistics /
apply passes, the NMRF / BridgeDepth window attention (`win_rpe_attn`), DLNR's Restormer extractor (`ln_lastdim_v*`,
`dwconv3_nhwc`, `dwgate_v*`) and the in-place DINOv3 RoPE on the packed qkv (`rope_qk_h`,
`fastforks.features.models.dinov3`).
"""

from __future__ import annotations

import functools
import logging
import math
from pathlib import Path

import torch

from fastforks._core.kernels import cubin_library

# p: pointer, i: int32, l: int64, f: float32 (must match the extern "C" kernels in fastbb_kernels.cu)
_FLASH = "pppp" + "ii" + "lii" * 4 + "f"
SIGNATURES = {
    **{f"flash64_r{r}_{v}": _FLASH for r in (1, 2) for v in ("hh", "fh", "ff")},
    "in_stats_h": "ppiii",
    "in_stats_b": "ppiii",
    "in_stats_final": "ppiiiiifpp",
    "in_apply_h": "pppppp" + "iii" + "f" + "i",
    "in_apply_b": "pppppp" + "iii" + "f" + "i",
    "win_rpe_attn": "pii" + "pi" + "p" + "i" * 8 + "f" + "p" + "iiii",
    **{f"ln_lastdim_v{v}": "pppplifii" for v in (1, 2, 4, 8)},
    **{f"dwgate_v{v}": "ppppiiii" for v in (1, 2, 4)},
    "dwconv3_nhwc": "ppppiiii",
    "rope_qk_h": "ppiiiif",
}
LN_MAX_C = 2048  # ln_lastdim: 32 threads x 8 vectors x 8 channels
IN_STATS_SMEM = 256 * 16 * 4  # dynamic shared memory of in_stats_*

load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def _aligned(*ts: torch.Tensor | None, to: int) -> bool:
    return all(t is None or t.data_ptr() % to == 0 for t in ts)


def check_flash64(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor | None = None) -> None:
    """Check the `flash64_*` launch contract: fp16 CUDA `[B, N, heads, 64]` views with unit last stride.

    Raises:
        ValueError: A dtype, head dim, shape, device or stride the kernel cannot read.
    """
    ts = (("q", q), ("k", k), ("v", v)) + ((("o", o),) if o is not None else ())
    for name, t in ts:
        if t.dtype != torch.float16 or not t.is_cuda or t.dim() != 4 or t.shape[-1] != 64 or t.stride(-1) != 1:  # noqa: PLR2004
            msg = (
                f"flash64: {name} must be a float16 CUDA [B, N, heads, 64] tensor with unit last stride, "
                f"got {tuple(t.shape)} {t.dtype} {t.device} strides {t.stride()}"
            )
            raise ValueError(msg)
    if any(t.shape != q.shape for _, t in ts):
        msg = f"flash64: q, k, v, o shapes differ: {[tuple(t.shape) for _, t in ts]}"
        raise ValueError(msg)


FLASH_VARIANTS = ("hh", "fh", "ff")  # fp16 q k^T + fp16 p v, fp32 q k^T + fp16 p v, fp32 both (increasing range)
FLASH_PROBE = 64  # launches per `Flash64` whose output is checked for overflow
_LOG2E = math.log2(math.e)
_log = logging.getLogger(__name__)


class Flash64:
    """The `flash64_*` attention of one model, with a fallback when its fp16 accumulators overflow.

    `"hh"` accumulates the unscaled q k^T in fp16, `"hh"` / `"fh"` a 64-key tile of p v in fp16: past 65504 the sum
    is inf and the online softmax gives NaN (DINOv3 ViT-S / B: q k^T up to 36x the fp16 range). The first `probe`
    eager launches of this object check their output (one sync each); a non-finite output is recomputed with the
    next variant of `FLASH_VARIANTS`, and the first variant giving a finite output is kept for every later call.
    Non-finite in every variant (NaN inputs): returned as is, the variant is kept. After the probe launches, and
    while a CUDA graph is captured, nothing is checked (no sync, no extra work). The outliers that overflow come from
    the weights (fixed channels), so the first images of a model show them.
    """

    def __init__(self, variant: str = "hh", *, probe: int = FLASH_PROBE) -> None:
        """`variant`: one of `FLASH_VARIANTS` (the fastest the model's accuracy allows); `probe`: checked launches."""
        if variant not in FLASH_VARIANTS:
            msg = f"flash64 variant {variant!r}: expected one of {FLASH_VARIANTS}"
            raise ValueError(msg)
        self.variant, self.probe = variant, probe

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        """softmax(q k^T * scale) v of fp16 CUDA `[B, N, heads, 64]` views (unit last stride) -> `[B, N, heads, 64]`."""
        o = self._run(self.variant, q, k, v, scale)
        if self.probe <= 0 or torch.cuda.is_current_stream_capturing():
            return o
        self.probe -= 1
        if bool(torch.isfinite(o).all()):
            return o
        for variant in FLASH_VARIANTS[FLASH_VARIANTS.index(self.variant) + 1 :]:
            o = self._run(variant, q, k, v, scale)
            if bool(torch.isfinite(o).all()):
                _log.info("flash64: fp16 overflow in %r (non-finite output), using %r", self.variant, variant)
                self.variant = variant
                break
        return o

    @staticmethod
    def _run(variant: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
        b, n, h, d = q.shape
        o = torch.empty((b, n, h, d), device=q.device, dtype=q.dtype)
        rm = 2 if n * b * h >= 36 * 2 * 128 * 4 else 1  # small problems: 64-row CTAs fill the GPU better
        br = 64 * rm
        load(q.device).launch(
            f"flash64_r{rm}_{variant}", ((n + br - 1) // br, b * h, 1), (128, 1, 1), q, k, v, o, n, h,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2),
            float(scale * _LOG2E),
        )  # fmt: skip
        return o


@functools.cache
def win_rpe_attn_reason(device: torch.device, *, ws: int, n: int) -> str | None:
    """Why `win_rpe_attn` cannot run windows of `ws x ws` cells with `n` tokens each on `device`, or None.

    One thread per token and the head's table plus K / V of the CTA's windows in shared memory: T = ws^2 n must fit the
    kernel's block limit and the shared memory the device's opt-in limit.

    Raises:
        KernelUnavailableError: The library cannot be loaded for `device`.
    """
    t = ws * ws * n
    wpc = max(1, 128 // t)
    threads = (wpc * t + 31) // 32 * 32
    smem = ((2 * ws - 1) ** 2 * 96 + 2 * wpc * t * 32) * 2
    limit = load(device).max_threads("win_rpe_attn")
    smem_limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    if threads > limit:
        return f"win_rpe_attn: {t} tokens per window ({ws} x {ws} cells x {n}) need {threads} threads, limit {limit}"
    if smem > smem_limit:
        return f"win_rpe_attn: window {ws} x {ws} x {n} needs {smem} B shared memory, device limit {smem_limit} B"
    return None


def win_rpe_attn(  # noqa: PLR0913 - the window geometry
    qkv: torch.Tensor,
    table: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    heads: int,
    ws: int,
    n: int,
    grid: tuple[int, int],
    shift: int,
    scale: float,
) -> torch.Tensor:
    """Window attention with relative-position embeddings on q / k / v, straight on the packed qkv projection.

    Args:
        qkv: fp16 `[B, Hc, Wc, n, 3C]` token grid (or `[windows, T, 3C]` with `grid=(ws, ws)`), contiguous; the
            windows are taken after a cyclic shift by `-shift` (Swin), the output is written back unshifted.
        table: fp16 `[(2 ws - 1)^2, 3C]` relative-position table (per head: Eq | Ek | Ev, 32 channels each).
        mask: fp32 `[nW, T, T]` additive mask (window w uses `mask[w % nW]`), or None. Every row needs a finite
            entry: a fully masked row gives NaN, as softmax does (the NMRF / BridgeDepth masks keep the diagonal).
        heads: Heads (`C / heads` must be 32).
        ws: Window size in cells.
        n: Tokens per cell.
        grid: `(Hc, Wc)` cells of the token grid (multiples of `ws`).
        shift: Cyclic shift in cells (0: no shift).
        scale: Scale of q.

    Returns:
        fp16 `[..., C]` in the token layout of `qkv`.

    Raises:
        ValueError: Shapes, dtypes, alignment or a window geometry (`win_rpe_attn_reason`) the kernel cannot run.
    """
    c3 = qkv.shape[-1]
    hc, wc = grid
    t = ws * ws * n
    if (
        qkv.dtype != torch.float16
        or not qkv.is_contiguous()
        or c3 != 96 * heads
        or hc % ws
        or wc % ws
        or qkv.numel() % (hc * wc * n * c3)
    ):
        msg = f"win_rpe_attn: fp16 contiguous qkv with head_dim 32 on a grid of whole windows, got {tuple(qkv.shape)}"
        raise ValueError(msg)
    r = (2 * ws - 1) ** 2
    if (
        table.dtype != torch.float16
        or table.shape != (r, c3)
        or table.stride(1) != 1
        or table.stride(0) % 2
        or not _aligned(qkv, table, to=4)  # read as half2
    ):
        msg = f"win_rpe_attn: fp16 [{r}, {c3}] table with even row stride, 4-byte aligned qkv / table"
        raise ValueError(msg)
    if not 0 <= shift < min(hc, wc):  # the kernel wraps the shifted window once
        msg = f"win_rpe_attn: shift {shift} outside [0, {min(hc, wc)})"
        raise ValueError(msg)
    if mask is not None and (
        mask.dtype != torch.float32 or not mask.is_contiguous() or mask.shape[0] < 1 or mask.shape[1:] != (t, t)
    ):
        msg = f"win_rpe_attn: fp32 contiguous [nW >= 1, {t}, {t}] mask"
        raise ValueError(msg)
    if reason := win_rpe_attn_reason(qkv.device, ws=ws, n=n):
        raise ValueError(reason)
    c = c3 // 3
    out = qkv.new_empty((*qkv.shape[:-1], c))
    nwin = qkv.numel() // (c3 * t)
    wpc = max(1, 128 // t)
    smem = (r * 96 + 2 * wpc * t * 32) * 2
    k = load(qkv.device)
    k.launch(
        "win_rpe_attn",
        ((nwin + wpc - 1) // wpc, heads, 1),
        ((wpc * t + 31) // 32 * 32, 1, 1),
        qkv, c3, c, table, table.stride(0), mask, 1 if mask is None else mask.shape[0], heads, t, ws, n,
        hc, wc, shift, scale, out, c, nwin, wpc, 1,
        smem=smem,
    )  # fmt: skip
    return out


def _vec(t: torch.Tensor, numel: int, dtype: torch.dtype) -> bool:
    return t.dtype == dtype and t.numel() == numel and t.is_contiguous() and t.device.type == "cuda"


def ln_lastdim(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, eps: float, *, bias_free: bool
) -> torch.Tensor:
    """LayerNorm over the last dim of a contiguous fp16 tensor (C % 8 == 0, C <= 2048), fp32 `w` / `b`.

    bias_free: Restormer's BiasFree variant `x * rstd * w`.
    """
    ch = x.shape[-1]
    if (
        x.dtype != torch.float16
        or not x.is_contiguous()
        or ch % 8
        or ch > LN_MAX_C
        or not _aligned(x, to=16)  # read as uint4
        or not _vec(w, ch, torch.float32)
        or (b is not None and not _vec(b, ch, torch.float32))
    ):
        msg = (
            f"ln_lastdim: contiguous 16-byte aligned fp16 [..., C % 8 == 0, C <= {LN_MAX_C}], contiguous fp32 [C]"
            f" weight / bias; got {x.dtype} {tuple(x.shape)}"
        )
        raise ValueError(msg)
    y = torch.empty_like(x)
    m = x.numel() // ch
    g = 1  # threads per row (up to a warp), 8 channels per vector
    while g < 32 and g * 8 < ch:  # noqa: PLR2004 - warp size
        g *= 2
    v = next(v for v in (1, 2, 4, 8) if g * 8 * v >= ch)  # vectors per thread (template variant)
    load(x.device).launch(
        f"ln_lastdim_v{v}", ((m * g + 255) // 256, 1, 1), (256, 1, 1), x, w, b, y, m, ch, eps, int(bias_free), g
    )
    return y


def dwgate(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """Restormer GDFN core `gelu(dw3x3(x)[:h]) * dw3x3(x)[h:]`.

    On NHWC fp16 `x [N, H, W, 2h]` (contiguous), fp32 weight `[9, 2h]` and bias `[2h]`; returns `[N, H, W, h]`.
    """
    n, hh, ww, c2 = x.shape
    h = c2 // 2
    v = 4 if h % 4 == 0 else 2 if h % 2 == 0 else 1
    if (
        x.dtype != torch.float16
        or not x.is_contiguous()
        or c2 % 2
        or not _aligned(x, to=2 * v)  # read as v halves
        or w.shape != (9, c2)
        or not _vec(w, 9 * c2, torch.float32)
        or (b is not None and not _vec(b, c2, torch.float32))
    ):
        msg = (
            f"dwgate: contiguous fp16 [N, H, W, 2h] ({2 * v}-byte aligned), fp32 [9, 2h] weight and [2h] bias,"
            f" got {tuple(x.shape)} {tuple(w.shape)}"
        )
        raise ValueError(msg)
    y = x.new_empty((n, hh, ww, h))
    total = n * hh * ((ww + 3) // 4) * (h // v)
    load(x.device).launch(f"dwgate_v{v}", ((total + 127) // 128, 1, 1), (128, 1, 1), x, w, b, y, n, hh, ww, h)
    return y


def dwconv3(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """Depthwise 3x3 conv (stride 1, zero pad 1) on NHWC fp16 `x [N, H, W, C]`.

    `x` contiguous with C % 8 == 0, fp16 weight `[3, 3, C]` and bias `[C]`.
    """
    n, hh, ww, c = x.shape
    if (
        x.dtype != torch.float16
        or not x.is_contiguous()
        or c % 8
        or w.shape != (3, 3, c)
        or not _vec(w, 9 * c, torch.float16)
        or (b is not None and not _vec(b, c, torch.float16))
        or not _aligned(x, w, b, to=16)  # read as uint4
    ):
        msg = (
            f"dwconv3: contiguous 16-byte aligned fp16 [N, H, W, C % 8 == 0], fp16 [3, 3, C] weight and [C] bias,"
            f" got {tuple(x.shape)} {tuple(w.shape)}"
        )
        raise ValueError(msg)
    y = torch.empty_like(x)
    total = n * hh * ((ww + 1) // 2) * (c // 8)
    load(x.device).launch("dwconv3_nhwc", ((total + 127) // 128, 1, 1), (128, 1, 1), x, w, b, y, n, hh, ww, c // 8)
    return y
