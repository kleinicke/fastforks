"""SeedVR2 on Apple MLX: `dit.Transformer` and `vae.VAE` in `mx` operations, weights read straight from the checkpoints.

The PyTorch modules are never built: the bf16 safetensors files are opened lazily (`fastforks.shared.mlx_weights`)
and each transformer block / VAE tensor is read, cast to the run dtype and kept resident one group at a time, so
memory holds the weights once in the run dtype (3B: 6.8 GB in bf16 / fp16) plus the activations.

The arithmetic follows the PyTorch net step for step, with `bf16` replaced by the run dtype `dt` (bfloat16 as upstream,
or float16): GEMMs, attention, residual stream and convolutions in `dt`; RMS statistics, RoPE, the AdaSingle sums and
group norms in float32, rounded to `dt` where the PyTorch net rounds to bf16. The load-time constants (time
embedding, text tokens after `txt_in`, AdaSingle sums) are computed in bf16 as upstream and then cast.

The VAE runs one frame (the image API restores one frame, a one-frame clip): its causal 3-D convolutions are then 2-D
convolutions with the temporal taps summed (`_vae_layout`). The window partition and the RoPE tables are made by the
PyTorch code (`dit.make_plan`) on the CPU and copied in.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import mx
from fastforks.shared.mlx_weights import Nf4, Weights, nf4_quantize

from . import dit
from .vae import CHANNELS, GROUPS, LATENT, SCALE, TEMPORAL_DOWN, TEMPORAL_UP
from .vae import EPS as VAE_EPS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from .dit import Config

F32 = mx.float32
BF = mx.bfloat16
_PLANS = 8  # sizes whose index plans are kept (the least recently used one is dropped)


def _mx(t: torch.Tensor) -> mx.array:
    """A CPU tensor (float32 / int64) as an `mx.array` (int64 indices -> int32)."""
    t = t.detach().contiguous()
    if t.dtype == torch.int64:
        t = t.to(torch.int32)
    return mx.array(t.numpy())


def linear(x: mx.array, w: mx.array, b: mx.array | None = None) -> mx.array:
    """`F.linear` in the dtype of `x` / `w` (bias added in the GEMM's epilogue, as cuBLAS does)."""
    if b is None:
        return mx.matmul(x, w.T)
    return mx.addmm(b, x.reshape(-1, x.shape[-1]), w.T).reshape(*x.shape[:-1], w.shape[0])


def rms(x: mx.array) -> mx.array:
    """RMS norm without affine: float32 statistics, one rounding to the dtype of `x`."""
    return mx.fast.rms_norm(x, None, dit.EPS)


def modulate(x: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    """`x.mul_(scale).add_(shift)` on a `dt` tensor with float32 vectors: float32 products, a rounding after each."""
    dt = x.dtype
    return ((x * scale).astype(dt) + shift).astype(dt)


def rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`dit.rope`: float32 `x * cos + rotate_half(x) * sin` (interleaved pairs) -> the dtype of `x`."""
    xf = x.astype(F32)
    pairs = xf.reshape(*xf.shape[:-1], -1, 2)
    rot = mx.stack((-pairs[..., 1], pairs[..., 0]), axis=-1).reshape(xf.shape)
    return (xf * cos + rot * sin).astype(x.dtype)


class Group:
    """Windows of one shape: indices `[n_win, n]` and the RoPE tables `[n, 1, 128]`."""

    def __init__(self, g: dit.Group) -> None:
        """Copied from the PyTorch plan."""
        self.idx = _mx(g.idx)
        self.cos, self.sin = _mx(g.cos), _mx(g.sin)
        self.nw, self.n = g.idx.shape


class Plan:
    """`dit.Plan` in MLX, plus the permutation that puts the concatenated window outputs back in token order."""

    def __init__(self, p: dit.Plan) -> None:
        """Copied from the PyTorch plan."""
        self.groups = [Group(g) for g in p.groups]
        self.count = p.count
        order = torch.cat([g.idx.reshape(-1) for g in p.groups])
        self.inverse = _mx(torch.argsort(order))


class Transformer:
    """`dit.Transformer` on MLX: constants at load, then 32 / 36 blocks resident in the run dtype, or streamed.

    `quant`: the blocks' linear weights in NF4 (`quant = "nf4"`, as `net.read_dit`), quantised one tensor at a time
    and expanded per call (`mlx_weights.Nf4`). `stream` (without `quant`): only the blocks that fit in `resident`
    bytes are kept (spread evenly over the depth); each call reads the others from the file (`Weights.stream`), the
    next streamed block's read overlapped with the compute of the blocks before it. Memory then holds the resident
    blocks, two streamed blocks (7B bf16: 0.45 GB each) and the activations; a call takes about the file read time of
    the streamed blocks (the resident blocks' compute hides under it).
    """

    def __init__(  # noqa: PLR0913 - the checkpoint and the run's settings
        self, w: Weights, cfg: Config, dt: mx.Dtype, *, quant: bool = False, stream: bool = False, resident: int = 0
    ) -> None:
        """Read the transformer's checkpoint block by block (`w` keys as in the checkpoint)."""
        self.cfg, self.dt, self.w = cfg, dt, w
        emb = self._time_embedding(w)
        txt = linear(w.peek("txt"), w.peek("txt_in.weight"), w.peek("txt_in.bias"))
        self.txt = txt.astype(dt)
        self.consts: dict[tuple[int, str, int], tuple[mx.array, mx.array, mx.array]] = {}
        e = emb.reshape(cfg.dim, 2, 3)  # "b (d l g)"
        for i in range(cfg.layers):
            for branch in ("vid", "txt") if i < cfg.mm_layers else ("all",):
                for layer, name in enumerate(("attn", "mlp")):
                    p = f"blocks.{i}.ada.{branch}.{name}_"
                    self.consts[i, branch, layer] = tuple(
                        e[:, layer, j].astype(F32) + w.peek(p + g) for j, g in enumerate(("shift", "scale", "gate"))
                    )
        if cfg.out_norm:
            self.out_shift = e[:, 0, 0].astype(F32) + w.peek("vid_out_ada.out_shift")
            self.out_scale = e[:, 0, 1].astype(F32) + w.peek("vid_out_ada.out_scale")
            self.out_norm = w.peek("vid_out_norm.weight").astype(dt)
        mx.eval(self.txt, self.consts, *((self.out_shift, self.out_scale, self.out_norm) if cfg.out_norm else ()))
        top = ("vid_in.proj.weight", "vid_in.proj.bias", "vid_out.proj.weight", "vid_out.proj.bias")
        self.io = w.take(top, dt)
        self.keys = [[k for k in w.prefixed(f"blocks.{i}.") if ".ada." not in k] for i in range(cfg.layers)]
        kept = range(cfg.layers)
        if stream and not quant:
            size = max(sum(w.peek(k).size for k in keys) for keys in self.keys) * dt.size
            n = min(cfg.layers, resident // size)
            kept = [i for i in range(cfg.layers) if (i + 1) * n // cfg.layers > i * n // cfg.layers]
        self.blocks: list[dict[str, mx.array | Nf4] | None] = [None] * cfg.layers  # None: streamed
        for i in kept:
            p, keys = f"blocks.{i}.", self.keys[i]
            nf4 = {k for k in keys if quant and k.endswith(".weight") and w.peek(k).ndim == 2}  # noqa: PLR2004
            block: dict[str, mx.array | Nf4] = dict(w.take([k for k in keys if k not in nf4], dt))
            for k in sorted(nf4):
                block[k] = nf4_quantize(w.take([k])[k])
            self.blocks[i] = {k[len(p) :]: v for k, v in block.items()}
        a = dit._angles(torch.arange(self.txt.shape[0])).repeat(1, 3)[:, None]  # noqa: SLF001 - the shared RoPE angles
        pad = dit.HEAD_DIM - 3 * dit.ROPE_AXIS
        self.txt_rope = (
            mx.pad(_mx(a.cos()), [(0, 0), (0, 0), (0, pad)], constant_values=1.0),
            mx.pad(_mx(a.sin()), [(0, 0), (0, 0), (0, pad)]),
        )
        self.plans: dict[tuple[int, int, int], tuple[Plan, Plan]] = {}

    @staticmethod
    def _time_embedding(w: Weights) -> mx.array:
        """`dit.time_embedding` in bf16 -> `[6 * dim]` bf16."""
        half = dit.SINUSOID // 2
        exponent = -math.log(10000) * mx.arange(half, dtype=F32) / half
        arg = mx.array([dit.TIMESTEP], dtype=F32)[:, None] * mx.exp(exponent)[None]
        emb = mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1).astype(BF)
        for name in ("proj_in", "proj_hid"):
            emb = _silu(linear(emb, w.peek(f"emb_in.{name}.weight"), w.peek(f"emb_in.{name}.bias")))
        return linear(emb, w.peek("emb_in.proj_out.weight"), w.peek("emb_in.proj_out.bias"))[0]

    def _plans(self, size: tuple[int, int, int]) -> tuple[Plan, Plan]:
        if size not in self.plans:
            n = self.txt.shape[0] if self.cfg.mm_rope else None
            cpu = torch.device("cpu")
            self.plans[size] = (
                Plan(dit.make_plan(size, shifted=False, txt_len=n, device=cpu)),
                Plan(dit.make_plan(size, shifted=True, txt_len=n, device=cpu)),
            )
            while len(self.plans) > _PLANS:  # a video of every size would otherwise keep each size's index tables
                del self.plans[next(iter(self.plans))]
        else:
            self.plans[size] = self.plans.pop(size)  # most recently used last
        return self.plans[size]

    def _block_weights(self, i: int) -> dict[str, mx.array | Nf4]:
        """Block `i`: resident, or fresh lazy reads of the file in the run dtype (`stream`)."""
        if (t := self.blocks[i]) is not None:
            return t
        p = f"blocks.{i}."
        return {k[len(p) :]: v for k, v in self.w.stream(self.keys[i], self.dt).items()}

    def _const(self, i: int, branch: str, layer: int) -> tuple[mx.array, mx.array, mx.array]:
        return self.consts[i, "all" if i >= self.cfg.mm_layers else branch, layer]

    @staticmethod
    def _name(t: dict[str, mx.array], name: str, branch: str) -> str:
        n = f"{name}.{branch}"
        return n if n + ".weight" in t else f"{name}.all"

    def _lin(self, t: dict[str, mx.array | Nf4], name: str, x: mx.array) -> mx.array:
        wt = t[name + ".weight"]
        return linear(x, wt.expand(self.dt) if isinstance(wt, Nf4) else wt, t.get(name + ".bias"))

    def attention(
        self, t: dict[str, mx.array], vid: mx.array, txt: mx.array, plan: Plan, *, last: bool
    ) -> tuple[mx.array, mx.array | None]:
        """`dit.Transformer.attention`: window attention over `[window tokens; text tokens]`."""
        heads, d = self.cfg.heads, dit.HEAD_DIM
        nm = self._name
        qkv = self._lin(t, nm(t, "attn.proj_qkv", "vid"), vid).reshape(-1, 3, heads, d)
        tqkv = self._lin(t, nm(t, "attn.proj_qkv", "txt"), txt).reshape(-1, 3, heads, d)
        q = t[nm(t, "attn.norm_q", "vid") + ".weight"] * rms(qkv[:, 0])
        k = t[nm(t, "attn.norm_k", "vid") + ".weight"] * rms(qkv[:, 1])
        v = qkv[:, 2]
        tq = t[nm(t, "attn.norm_q", "txt") + ".weight"] * rms(tqkv[:, 0])
        tk = t[nm(t, "attn.norm_k", "txt") + ".weight"] * rms(tqkv[:, 1])
        if self.cfg.mm_rope:
            tq, tk = rope(tq, *self.txt_rope), rope(tk, *self.txt_rope)
        tv = tqkv[:, 2]
        n_txt = txt.shape[0]
        pieces = []
        txt_sum = None if last else mx.zeros((n_txt, heads, d), dtype=F32)
        for g in plan.groups:
            nw, n = g.nw, g.n
            gq, gk = rope(q[g.idx], g.cos, g.sin), rope(k[g.idx], g.cos, g.sin)
            gk = mx.concatenate([gk, mx.broadcast_to(tk[None], (nw, n_txt, heads, d))], axis=1)
            gv = mx.concatenate([v[g.idx], mx.broadcast_to(tv[None], (nw, n_txt, heads, d))], axis=1)
            if not last:
                gq = mx.concatenate([gq, mx.broadcast_to(tq[None], (nw, n_txt, heads, d))], axis=1)
            o = mx.fast.scaled_dot_product_attention(
                gq.transpose(0, 2, 1, 3), gk.transpose(0, 2, 1, 3), gv.transpose(0, 2, 1, 3), scale=d**-0.5
            ).transpose(0, 2, 1, 3)  # [nw, n (+ 58), H, D]
            pieces.append(o[:, :n].reshape(nw * n, heads * d))
            if txt_sum is not None:
                txt_sum = txt_sum + o[:, n:].astype(F32).sum(0)
        out = mx.concatenate(pieces, axis=0)[plan.inverse]
        vid_out = self._lin(t, nm(t, "attn.proj_out", "vid"), out)
        if txt_sum is None:
            return vid_out, None
        txt_mean = (txt_sum / plan.count).astype(self.dt).reshape(n_txt, heads * d)
        return vid_out, self._lin(t, nm(t, "attn.proj_out", "txt"), txt_mean)

    def mlp(self, t: dict[str, mx.array], x: mx.array, branch: str) -> mx.array:
        """SwiGLU (3B) / tanh-GELU (7B)."""
        n = f"mlp.{branch}" if f"mlp.{branch}.proj_in.weight" in t else "mlp.all"
        if self.cfg.swiglu:
            return self._lin(
                t, n + ".proj_out", _silu(self._lin(t, n + ".proj_in_gate", x)) * self._lin(t, n + ".proj_in", x)
            )
        return self._lin(t, n + ".proj_out", _gelu_tanh(self._lin(t, n + ".proj_in", x)))

    def block(
        self, i: int, t: dict[str, mx.array | Nf4], vid: mx.array, txt: mx.array, plan: Plan
    ) -> tuple[mx.array, mx.array]:
        """`dit.Transformer.block` with the block's weights `t`."""
        last = i == self.cfg.layers - 1
        vs, vc, vg = self._const(i, "vid", 0)
        va = modulate(rms(vid), vc, vs)
        ta = rms(txt).astype(self.dt)  # the text stream may be float32 (`__call__`)
        if not (last and self.cfg.last_vid_only):
            ts, tc, _ = self._const(i, "txt", 0)
            ta = modulate(ta, tc, ts)
        vo, to = self.attention(t, va, ta, plan, last=last)
        vid = (vo * vg).astype(self.dt) + vid
        if to is not None:
            txt = (to * self._const(i, "txt", 0)[2]).astype(txt.dtype) + txt
        vs, vc, vg = self._const(i, "vid", 1)
        vid = (self.mlp(t, modulate(rms(vid), vc, vs), "vid") * vg).astype(self.dt) + vid
        if not last:
            ts, tc, tg = self._const(i, "txt", 1)
            txt = (self.mlp(t, modulate(rms(txt).astype(self.dt), tc, ts), "txt") * tg).astype(txt.dtype) + txt
        return vid, txt

    def __call__(self, x: mx.array) -> mx.array:
        """`[T, H, W, 33]` (noise, latent, ones) -> the predicted velocity `[T, H, W, 16]`, evaluated per block."""
        tt, hh, ww, c = x.shape
        size = (tt, hh // 2, ww // 2)
        vid = x.reshape(tt, size[1], 2, size[2], 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 4 * c)
        vid = linear(vid, self.io["vid_in.proj.weight"], self.io["vid_in.proj.bias"])
        # The 7B text stream grows to ~1e5 from block 3 on (its gated MLP output to ~9e4): over float16's range, so in
        # float16 the text residual stream and its gated updates stay float32 (58 tokens)
        txt = self.txt.astype(F32) if self.dt == mx.float16 else self.txt
        plans = self._plans(size)
        streamed = [i for i, t in enumerate(self.blocks) if t is None]
        reads: dict[int, dict[str, mx.array | Nf4]] = {}
        for i in range(self.cfg.layers):
            t = reads.pop(i) if i in reads else self._block_weights(i)
            # the next streamed block's read runs while this block (and any resident ones after it) compute
            j = next((j for j in streamed if j > i), None)
            if j is not None and j not in reads:
                reads[j] = self._block_weights(j)
                mx.async_eval(reads[j])
            vid, txt = self.block(i, t, vid, txt, plans[i % 2])
            mx.eval(vid, txt)
            del t
        if self.cfg.out_norm:
            vid = modulate(self.out_norm * rms(vid), self.out_scale, self.out_shift)
        out = linear(vid, self.io["vid_out.proj.weight"], self.io["vid_out.proj.bias"])
        co = out.shape[-1] // 4
        return out.reshape(tt, size[1], size[2], 2, 2, co).transpose(0, 1, 3, 2, 4, 5).reshape(tt, hh, ww, co)


# The elementwise chains below are compiled: one fused kernel, no float32 intermediates in memory.


@mx.compile
def _silu(x: mx.array) -> mx.array:
    """SiLU in float32, rounded once (PyTorch's bf16 kernel)."""
    xf = x.astype(F32)
    return (xf * mx.sigmoid(xf)).astype(x.dtype)


@mx.compile
def _gelu_tanh(x: mx.array) -> mx.array:
    """tanh-approximated GELU in float32, rounded once."""
    xf = x.astype(F32)
    return (0.5 * xf * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (xf + 0.044715 * xf**3)))).astype(x.dtype)


def _group_apply(x: mx.array, mean: mx.array, rstd: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """`(x - mean) * rstd * w + b` in float32 per group (`x` `[B, N, G, C/G]`), rounded once to `x`'s dtype."""
    c = w.shape[-1]
    y = ((x.astype(F32) - mean) * rstd).reshape(*x.shape[:-2], c)
    return (y * w.astype(F32) + b.astype(F32)).astype(x.dtype)


_group_norm = mx.compile(_group_apply)
_group_norm_silu = mx.compile(lambda *a: _silu(_group_apply(*a)))

_BAND_BYTES = 32 << 20  # rows of a VAE activation handled at once (`VAE.conv`, `VAE.norm_fn`)


# --- VAE (one frame, NHWC) -------------------------------------------------------------------------------------------


def _vae_layout(name: str, a: mx.array) -> mx.array:
    """PyTorch conv weights `[O, I, kt, kh, kw]` -> the 2-D convolution of one frame, MLX `[O, kh, kw, I]`.

    On a one-frame clip every causal convolution sees the frame `kt` times (the frame repeated in front of itself), so
    it is the 2-D convolution with the temporal taps summed (in float32, then rounded to the run dtype): one third of
    the work of the 3-D form, and no repeated frames in memory. A temporal upsampler's 1x1x1 convolution writes
    channels `(x, y, z, c)` and the second frame (z = 1) is dropped, so only its z = 0 outputs are kept.
    """
    if a.ndim != 5:  # noqa: PLR2004 - 3-D convolution weights
        return a[_first_frame(a.shape[0])] if name in _TEMPORAL_UPSCALE_BIAS else a
    if name in _TEMPORAL_UPSCALE:
        a = a[_first_frame(a.shape[0])]
    return a.astype(F32).sum(axis=2).transpose(0, 2, 3, 1)


def _first_frame(out: int) -> mx.array:
    """Output channels `(x, y, z, c)` (z = 2) of a temporal upsampler -> the indices with z = 0."""
    c = out // 8
    return (mx.arange(4)[:, None] * 2 * c + mx.arange(c)[None]).reshape(-1)


_TEMPORAL_UPSCALE = {f"decoder.up_blocks.{i}.upsamplers.0.upscale_conv.weight" for i, t in enumerate(TEMPORAL_UP) if t}
_TEMPORAL_UPSCALE_BIAS = {k[: -len("weight")] + "bias" for k in _TEMPORAL_UPSCALE}


class VAE:
    """`vae.VAE` of one frame on MLX, channels last: `encode` `[1, H, W, 3]` -> `[1, h, w, 16]`, `decode` the inverse.

    The image API restores one frame, a one-frame clip: the causal 3-D convolutions are 2-D (`_vae_layout`), temporal
    downsampling keeps the frame and temporal upsampling's first frame is the only one kept.
    """

    def __init__(self, w: Weights, dt: mx.Dtype) -> None:
        """Read every VAE tensor in `dt` (conv weights as one-frame 2-D convolutions)."""
        self.dt = dt
        self.p: dict[str, mx.array] = {}
        for k in w.prefixed("encoder.") + w.prefixed("decoder."):  # one at a time: the float32 tap sums are transient
            self.p |= w.take([k], dt, _vae_layout)

    def conv(
        self,
        name: str,
        x: mx.array,
        stride: int = 1,
        pad: tuple[tuple[int, int], tuple[int, int]] | int = 0,
        *,
        pre: Callable[[mx.array], mx.array] | None = None,
        res: mx.array | None = None,
    ) -> mx.array:
        """`CausalConv3d` of a one-frame clip, in bands of output rows: `conv(pre(x)) + res`.

        `pad` is zero padding `((top, bottom), (left, right))` (an int: all four) of `pre(x)`. `pre` (a group norm) is
        applied per band and `res` added per band, so neither `pre(x)` nor the convolution's workspace ever exists for
        the whole frame: a 1024 x 1024 x 128 convolution needs ~0.7 GB of workspace in one piece.
        """
        w, b = self.p[name + ".weight"], self.p[name + ".bias"]
        (pt, pb), (pl, pr) = ((pad, pad), (pad, pad)) if isinstance(pad, int) else pad
        _, h, wd, c = x.shape
        k, s = w.shape[1], stride
        rows_out = (h + pt + pb - k) // s + 1
        rows = max(1, _BAND_BYTES // (wd * c * x.itemsize * s))
        outs = []
        for r0 in range(0, rows_out, rows):
            r1 = min(rows_out, r0 + rows)
            a, e = s * r0 - pt, s * (r1 - 1) + k - pt  # input rows [a, e), of which [lo, hi) are inside the frame
            lo, hi = max(a, 0), min(e, h)
            band = x[:, lo:hi] if (lo, hi) != (0, h) else x
            if pre is not None:
                band = pre(band)
            top, bottom = lo - a, e - hi
            if top != bottom:  # a band at the top or bottom edge; others leave the padding to the convolution
                band = mx.pad(band, [(0, 0), (top, bottom), (0, 0), (0, 0)])
                top = 0
            if pl != pr:
                band = mx.pad(band, [(0, 0), (0, 0), (pl, pr), (0, 0)])
            # its own padding selects other (faster, here more accurate in bf16) kernels than a padded input
            y = mx.conv2d(band, w, stride=s, padding=(top, pl if pl == pr else 0)) + b
            if res is not None:
                y = y + (res[:, r0:r1] if rows < rows_out else res)
            if rows < rows_out:
                mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def norm_fn(self, name: str, x: mx.array, *, silu: bool) -> Callable[[mx.array], mx.array]:
        """Group norm in float32 (`causal_norm_wrapper`), rounded once, then SiLU, as a function of rows of `x`.

        The statistics are those of the whole `x`, summed in float32 band by band (no float32 copy of `x`).
        """
        _, h, wd, c = x.shape
        g = GROUPS
        rows = max(1, _BAND_BYTES // (wd * c * 4))

        def bands() -> Iterator[mx.array]:
            for r in range(0, h, rows):
                yield x[:, r : r + rows].reshape(1, -1, g, c // g).astype(F32)

        total = mx.zeros((1, 1, g, 1), dtype=F32)
        for xb in bands():
            total = total + xb.sum(axis=(1, 3), keepdims=True)
            mx.eval(total)
        mean = total / (h * wd * (c // g))
        total = mx.zeros_like(total)
        for xb in bands():
            total = total + mx.square(xb - mean).sum(axis=(1, 3), keepdims=True)
            mx.eval(total)
        rstd = mx.rsqrt(total / (h * wd * (c // g)) + VAE_EPS)
        weight, bias = self.p[name + ".weight"], self.p[name + ".bias"]
        apply = _group_norm_silu if silu else _group_norm

        def f(xb: mx.array) -> mx.array:
            n, hb, wb, _ = xb.shape
            return apply(xb.reshape(n, hb * wb, g, c // g), mean, rstd, weight, bias).reshape(n, hb, wb, c)

        return f

    def norm(self, name: str, x: mx.array, *, silu: bool) -> mx.array:
        """`norm_fn` applied to the whole `x`."""
        return self.norm_fn(name, x, silu=silu)(x)

    def resnet(self, p: str, x: mx.array) -> mx.array:
        """Each convolution in bands with its norm folded in (`conv`), evaluated per convolution: one lazy graph of
        a whole stage keeps most of its intermediates alive at once (a 208 x 304 decode: 1.8 GB instead of 0.2 GB)."""
        h = self.conv(p + ".conv1", x, pad=1, pre=self.norm_fn(p + ".norm1", x, silu=True))
        mx.eval(h)
        if p + ".conv_shortcut.weight" in self.p:
            x = self.conv(p + ".conv_shortcut", x)
            mx.eval(x)
        h = self.conv(p + ".conv2", h, pad=1, pre=self.norm_fn(p + ".norm2", h, silu=True), res=x)
        mx.eval(h)
        return h

    def attention(self, p: str, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.norm(p + ".group_norm", x, silu=False).reshape(b, 1, h * w, c)
        q, k, v = (linear(y, self.p[f"{p}.{m}.weight"], self.p[f"{p}.{m}.bias"]) for m in ("to_q", "to_k", "to_v"))
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)[:, 0]
        y = linear(y, self.p[p + ".to_out.0.weight"], self.p[p + ".to_out.0.bias"])
        return x + y.reshape(b, h, w, c)

    def mid(self, p: str, x: mx.array) -> mx.array:
        x = self.resnet(p + ".resnets.0", x)
        x = self.attention(p + ".attentions.0", x)
        return self.resnet(p + ".resnets.1", x)

    def encode(self, x: mx.array, eps: mx.array) -> mx.array:
        """`[1, H, W, 3]` in [-1, 1] -> posterior sample `[1, h, w, 16]` times the scaling factor."""
        x = self.conv("encoder.conv_in", x, pad=1)
        for i in range(len(CHANNELS)):
            p = f"encoder.down_blocks.{i}"
            for j in range(2):
                x = self.resnet(f"{p}.resnets.{j}", x)
            if i < len(TEMPORAL_DOWN):
                x = self.conv(f"{p}.downsamplers.0.conv", x, 2, ((0, 1), (0, 1)))
            mx.eval(x)
        x = self.mid("encoder.mid_block", x)
        h = self.conv("encoder.conv_out", x, pad=1, pre=self.norm_fn("encoder.conv_norm_out", x, silu=True))
        mean, logvar = h[..., :LATENT], h[..., LATENT:]
        dt = h.dtype
        std = mx.exp((0.5 * mx.clip(logvar, -30.0, 20.0)).astype(dt)).astype(dt)
        z = (mean + (std * eps.astype(dt)).astype(dt)).astype(dt)
        return (z * SCALE).astype(dt)

    def decode(self, z: mx.array) -> mx.array:
        """Scaled latent `[1, h, w, 16]` -> `[1, 8h, 8w, 3]` (unclamped)."""
        x = (z / SCALE).astype(z.dtype)
        x = self.mid("decoder.mid_block", self.conv("decoder.conv_in", x, pad=1))
        mx.eval(x)
        for i in range(len(CHANNELS)):
            p = f"decoder.up_blocks.{i}"
            for j in range(3):
                x = self.resnet(f"{p}.resnets.{j}", x)
            if i < len(TEMPORAL_UP):
                x = self.upsample(f"{p}.upsamplers.0", x)
            mx.eval(x)
        return self.conv("decoder.conv_out", x, pad=1, pre=self.norm_fn("decoder.conv_norm_out", x, silu=True))

    def upsample(self, p: str, x: mx.array) -> mx.array:
        """`Upsample3D` of one frame: 1x1 conv, depth-to-space, 3x3 conv (temporal: the first frame's channels)."""
        b, h, w, c = x.shape
        y = self.conv(p + ".upscale_conv", x)
        mx.eval(y)
        y = y.reshape(b, h, w, 2, 2, c).transpose(0, 1, 3, 2, 4, 5).reshape(b, 2 * h, 2 * w, c)
        return self.conv(p + ".conv", y, pad=1)


class Net:
    """`net.Net` on MLX: VAE encode, one transformer step at t = 1000, VAE decode."""

    def __init__(
        self,
        dit_path: Path,
        vae_path: Path,
        cfg: Config,
        dtype: mx.Dtype,
        *,
        quant: bool = False,
        stream: bool = False,
        resident: int = 0,
    ) -> None:
        """Read both checkpoints into the run dtype `dtype` (`quant`: the blocks' linear weights in NF4; `stream`
        without `quant`: blocks beyond `resident` bytes are read from the file per call, see `Transformer`)."""
        self.dt = dtype
        self.dit = Transformer(Weights(dit_path), cfg, dtype, quant=quant, stream=stream, resident=resident)
        self.vae = VAE(Weights(vae_path), dtype)

    def __call__(self, x: torch.Tensor, eps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` in [-1, 1] (CPU), posterior noise `[1, 16, 1, h, w]`, diffusion noise `[1, h, w, 16]` ->
        the restored frame `[1, 3, H, W]` float32 (unclamped)."""
        dt = self.dt
        xm = _mx(x.float()).astype(BF).astype(dt).transpose(0, 2, 3, 1)  # [1, H, W, 3]
        e = _mx(eps.float()).astype(BF)[:, :, 0].transpose(0, 2, 3, 1)  # [1, h, w, 16]
        latent = self.vae.encode(xm, e)
        mx.eval(latent)
        z = _mx(noise.float()).astype(BF).astype(dt)
        pred = self.dit(mx.concatenate([z, latent, mx.ones_like(latent[..., :1])], axis=-1))
        x0 = (z.astype(F32) - pred.astype(F32)).astype(dt)
        y = self.vae.decode(x0).transpose(0, 3, 1, 2).astype(F32)
        mx.eval(y)
        return torch.from_numpy(np.array(y))
