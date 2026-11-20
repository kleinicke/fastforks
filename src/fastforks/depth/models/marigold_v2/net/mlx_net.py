"""Marigold V2 on Apple MLX: `Net` (VAE, folded transformer) in `mx` operations, weights read from the lean checkpoint.

The PyTorch modules are never built: the safetensors file is opened lazily (`fastforks.shared.mlx_weights`). The 60
transformer blocks keep their NF4 storage (packed nibbles + bf16 absmax, 141 MB per block, 8.5 GB in all) and each
linear's weight is expanded to bf16 for its call (`mlx_weights.Nf4`, bit-exact with bitsandbytes), as the CUDA path
does. With `stream` (the default) the blocks are not kept: each call reads them from the file one block ahead of the
one computing (`Weights.stream`), so memory holds two blocks plus the activations (the 8.5 GB of a 16 GB Mac's
unified memory would otherwise push the rest of the system to swap); `stream=False` keeps them resident.

Precision follows the PyTorch net (`dit.py`, `nf4.py`, `vae.py`): GEMMs, attention and convolutions in bf16; the
residual stream, layer / RMS norms, RoPE, GELU and the LoRA sums in float32, rounded to bf16 where the PyTorch net
rounds. Scalar factors of bf16 tensors are applied in float32 (PyTorch's opmath) and rounded once. The VAE runs NHWC,
its large convolutions in row bands (norm and residual folded in per band, evaluated per band) and its mid-block
single-head attention (head dim 384, which the fused MLX attention does not take) in float32 query chunks.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.mlx import mx
from fastforks.shared.mlx_weights import Nf4, Weights

from . import dit

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

F32, BF = mx.float32, mx.bfloat16
PATCH = 2  # latent pixels per transformer token
EPS = 1e-6  # layer / RMS norms of the transformer
_BAND_BYTES = 32 << 20  # bytes of a VAE activation handled at once (`VAE.conv`)
_ATTN_SCORES = 256 << 20  # bytes of float32 scores per query chunk of the VAE's mid-block attention


def to_mx(t: torch.Tensor) -> mx.array:
    """A CPU tensor as an `mx.array` of its dtype (float32 / int64 -> int32)."""
    t = t.detach().contiguous()
    if t.dtype == torch.int64:
        t = t.to(torch.int32)
    return mx.array(t.numpy())


# --- transformer ------------------------------------------------------------------------------------------------------


def _linear(t: dict[str, mx.array], name: str, x: mx.array) -> tuple[mx.array, mx.array | None]:
    """`NF4Linear._parts`: the bf16 GEMM with bias, and the bf16 LoRA term (None without LoRA)."""
    b = t[name + ".bias"]
    w = Nf4(t[name + ".q"], t[name + ".absmax"].reshape(b.shape[0], -1)).expand(BF)
    y = mx.addmm(b, x, w.T)
    a = t.get(name + ".lora_A")
    return y, None if a is None else mx.matmul(mx.matmul(x, a.T), t[name + ".lora_B"].T)


@mx.compile
def _sum32(y: mx.array, lora: mx.array) -> mx.array:
    return y.astype(F32) + lora.astype(F32)


@mx.compile
def _gelu(y: mx.array, lora: mx.array) -> mx.array:
    """`NF4Linear.gelu`: tanh-GELU of the float32 sum, rounded to bf16."""
    x = y.astype(F32) + lora.astype(F32)
    return (0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))).astype(BF)


@mx.compile
def _gate_res(y: mx.array, lora: mx.array, gate: mx.array, res: mx.array) -> mx.array:
    """`NF4Linear.gate_res`: `res + (y + lora) * gate` in float32."""
    return res + (y.astype(F32) + lora.astype(F32)) * gate.astype(F32)


@mx.compile
def _modulate(x: mx.array, scale1p: mx.array, shift: mx.array) -> mx.array:
    """`Block._mod`: float32 layer norm (no affine), `* (1 + scale) + shift` in float32, rounded to bf16."""
    y = mx.fast.layer_norm(x, None, None, EPS)
    return (y * scale1p.astype(F32) + shift.astype(F32)).astype(BF)


@mx.compile
def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """`dit.rope`: float32 `[T, H, D]` rotated as complex pairs by `cos + i sin` `[T, 1, D / 2]` -> bf16."""
    p = x.reshape(*x.shape[:-1], -1, 2)
    a, b = p[..., 0], p[..., 1]
    return mx.stack([a * cos - b * sin, a * sin + b * cos], axis=-1).reshape(x.shape).astype(BF)


def _mods(v: mx.array) -> list[tuple[mx.array, mx.array, mx.array]]:
    """`Block._mods`: `[6 * dim]` bf16 -> `(shift, 1 + scale, gate)` for attention and MLP (`1 + scale` in bf16)."""
    d = v.shape[0] // dit.N_MOD
    return [(v[o : o + d], 1 + v[o + d : o + 2 * d], v[o + 2 * d : o + 3 * d]) for o in (0, 3 * d)]


class Transformer:
    """`dit.Transformer` on MLX: `img_in`, the 60 blocks (resident or streamed), the folded output norm, `proj_out`."""

    def __init__(self, w: Weights, *, stream: bool) -> None:
        """Read the constants (and with `stream=False` every block) from `w`."""
        self.w, self.stream = w, stream
        self.txt = w.take(["txt"])["txt"]  # [338, 3072] float32: the text tokens after `txt_in`
        scale, shift = mx.split(w.take(["norm_out.linear"])["norm_out.linear"], 2)
        self.out_scale1p, self.out_shift = 1 + scale, shift
        mx.eval(self.out_scale1p, self.out_shift)
        self.io = w.take([*w.prefixed("img_in."), *w.prefixed("proj_out.")])
        self.n = 1 + max(int(k.split(".")[1]) for k in w.prefixed("transformer_blocks."))
        self.keys = [w.prefixed(f"transformer_blocks.{i}.") for i in range(self.n)]
        self.blocks = None if stream else [self._strip(i, w.take(self.keys[i])) for i in range(self.n)]
        pos, neg = dit.rope_tables()
        self.tables = (pos, neg)

    @staticmethod
    def _strip(i: int, t: dict[str, mx.array]) -> dict[str, mx.array]:
        p = f"transformer_blocks.{i}."
        return {k[len(p) :]: v for k, v in t.items()}

    def _block_weights(self, i: int) -> dict[str, mx.array]:
        """Block `i`: resident, or fresh lazy reads of the file."""
        if self.blocks is not None:
            return self.blocks[i]
        return self._strip(i, self.w.stream(self.keys[i]))

    def _freqs(self, h: int, w: int) -> tuple[mx.array, mx.array]:
        f = dit.freqs(*self.tables, self.txt.shape[0], h, w)  # complex64 [T, 1, 64]
        return to_mx(f.real.float()), to_mx(f.imag.float())

    @staticmethod
    def _qk(yt: mx.array, yi: mx.array, wt: mx.array, wi: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """`Block._qk`: float32 RMS norm of each stream's projection, `[text; image]`, RoPE -> bf16 `[1, H, T, D]`."""
        heads = (dit.HEADS, dit.HEAD_DIM)
        xt = mx.fast.rms_norm(yt.reshape(-1, *heads), wt.astype(F32), EPS)
        xi = mx.fast.rms_norm(yi.reshape(-1, *heads), wi.astype(F32), EPS)
        return _rope(mx.concatenate([xt, xi]), cos, sin).transpose(1, 0, 2)[None]

    @staticmethod
    def block(
        t: dict[str, mx.array], img: mx.array, txt: mx.array, cos: mx.array, sin: mx.array, *, last: bool
    ) -> tuple[mx.array, mx.array]:
        """`dit.Block.forward` on `img [T_img, dim]`, `txt [T_txt, dim]` (float32)."""
        im, tm = _mods(t["img_mod"].astype(BF)), _mods(t["txt_mod"].astype(BF))
        n_txt = txt.shape[0]
        xi, xt = _modulate(img, im[0][1], im[0][0]), _modulate(txt, tm[0][1], tm[0][0])

        def f32(name: str, x: mx.array) -> mx.array:
            return _sum32(*_linear(t, name, x))

        def bf(name: str, x: mx.array) -> mx.array:
            y, lora = _linear(t, name, x)
            return y + lora

        q = Transformer._qk(f32("attn.add_q_proj", xt), f32("attn.to_q", xi), t["attn.norm_added_q.weight"], t["attn.norm_q.weight"], cos, sin)
        k = Transformer._qk(f32("attn.add_k_proj", xt), f32("attn.to_k", xi), t["attn.norm_added_k.weight"], t["attn.norm_k.weight"], cos, sin)
        v = mx.concatenate([bf("attn.add_v_proj", xt), bf("attn.to_v", xi)])
        v = v.reshape(-1, dit.HEADS, dit.HEAD_DIM).transpose(1, 0, 2)[None]
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=dit.HEAD_DIM**-0.5)
        a = a[0].transpose(1, 0, 2).reshape(-1, dit.HEADS * dit.HEAD_DIM)
        img = _gate_res(*_linear(t, "attn.to_out.0", a[n_txt:]), im[0][2], img)
        mx.eval(img, a)  # the attention's q / k / v and expanded weights go before the MLP's
        hidden = _gelu(*_linear(t, "img_mlp.net.0.proj", _modulate(img, im[1][1], im[1][0])))
        img = _gate_res(*_linear(t, "img_mlp.net.2", hidden), im[1][2], img)
        if not last:
            mx.eval(img)
            txt = _gate_res(*_linear(t, "attn.to_add_out", a[:n_txt]), tm[0][2], txt)
            hidden = _gelu(*_linear(t, "txt_mlp.net.0.proj", _modulate(txt, tm[1][1], tm[1][0])))
            txt = _gate_res(*_linear(t, "txt_mlp.net.2", hidden), tm[1][2], txt)
        return img, txt

    def __call__(self, packed: mx.array, h: int, w: int) -> mx.array:
        """Packed latent tokens `[h * w, 64]` bf16 -> the predicted packed velocity (bf16), evaluated per block."""
        cos, sin = self._freqs(h, w)
        y, lora = _linear(self.io, "img_in", packed)
        img = (y + lora).astype(F32)  # keep_bf: the sum in bf16
        txt = self.txt
        nxt = self._block_weights(0)
        for i in range(self.n):
            t = nxt
            if i + 1 < self.n:
                nxt = self._block_weights(i + 1)
                if self.blocks is None:
                    mx.async_eval(nxt)  # read the next block while this one computes
            img, txt = self.block(t, img, txt, cos, sin, last=i == self.n - 1)
            mx.eval(img, txt)
            del t
        x = mx.fast.layer_norm(img, None, None, EPS) * self.out_scale1p + self.out_shift
        return mx.addmm(self.io["proj_out.bias"], x.astype(BF), self.io["proj_out.weight"].T)


# --- VAE (one frame, NHWC) ---------------------------------------------------------------------------------------------


@mx.compile
def _l2norm(x: mx.array, gamma: mx.array) -> mx.array:
    """`VAE2D.norm`: channel L2 normalisation in float32 -> bf16, `* sqrt(C)` (float32 opmath) -> bf16, `* gamma`."""
    xf = x.astype(F32)
    n = mx.maximum(mx.sqrt((xf * xf).sum(axis=-1, keepdims=True)), 1e-12)
    y = (xf / n).astype(BF)
    y = (y.astype(F32) * math.sqrt(x.shape[-1])).astype(BF)
    return y * gamma


@mx.compile
def _l2norm_silu(x: mx.array, gamma: mx.array) -> mx.array:
    y = _l2norm(x, gamma).astype(F32)
    return (y * mx.sigmoid(y)).astype(BF)


class VAE:
    """`vae.VAE2D` on MLX, channels last: `encode` `[1, H, W, 3]` -> `[1, h, w, 16]`, `decode` the inverse."""

    def __init__(self, w: Weights) -> None:
        """Read the `vae.*` tensors (bf16; convolutions OHWI, 1 x 1 convolutions as linears `[O, I]`)."""

        def layout(k: str, a: mx.array) -> mx.array:
            if a.ndim == 4:  # noqa: PLR2004 - convolution weights
                return a.reshape(a.shape[0], -1) if a.shape[-1] == 1 else a.transpose(0, 2, 3, 1)
            if k.endswith(".gamma"):
                return a.reshape(-1)
            return a

        self.p = {k[4:]: v for k, v in w.take(w.prefixed("vae."), None, layout).items()}
        self.mean = self.p.pop("latents_mean").astype(BF)
        self.std_inv = 1.0 / self.p.pop("latents_std").astype(BF)
        mx.eval(self.mean, self.std_inv)

    def __contains__(self, name: str) -> bool:
        return name + ".weight" in self.p

    def conv(
        self,
        name: str,
        x: mx.array,
        *,
        pre: Callable[[mx.array], mx.array] | None = None,
        res: mx.array | None = None,
        up: bool = False,
        stride: int = 1,
    ) -> mx.array:
        """Convolution `name` of `pre(x)` (upsampled 2x nearest first if `up`) `+ res`, in bands of output rows.

        Same padding at stride 1; at stride 2 the PyTorch net's (0, 1, 0, 1) padding. `pre` and `res` are applied per
        band, so neither `pre(x)`, the upsampled input nor the convolution's workspace exists for the whole frame.
        """
        w, b = self.p[name + ".weight"], self.p[name + ".bias"]
        if w.ndim == 2:  # 1 x 1  # noqa: PLR2004
            y = pre(x) if pre is not None else x
            out = mx.addmm(b, y.reshape(-1, y.shape[-1]), w.T).reshape(*y.shape[:-1], w.shape[0])
            return out if res is None else out + res
        k = w.shape[1]
        n, h, wd, c = x.shape
        if up:
            h, wd = 2 * h, 2 * wd
        if stride == 2:  # noqa: PLR2004 - the downsampling resample: pad right / bottom, no bands (quarter output)
            y = pre(x) if pre is not None else x
            y = mx.pad(y, [(0, 0), (0, 1), (0, 1), (0, 0)])
            return mx.conv2d(y, w, stride=2) + b
        pad = k // 2
        rows = max(8, _BAND_BYTES // max(1, wd * max(c, w.shape[0]) * 2))
        outs = []
        for r0 in range(0, h, rows):
            r1 = min(h, r0 + rows)
            lo, hi = max(0, r0 - pad), min(h, r1 + pad)  # input rows with the halo, inside the frame
            if up:
                band = x[:, lo // 2 : (hi + 1) // 2]
                band = mx.repeat(mx.repeat(band, 2, axis=1), 2, axis=2)[:, lo - 2 * (lo // 2) :][:, : hi - lo]
            else:
                band = x[:, lo:hi] if (lo, hi) != (0, h) else x
            if pre is not None:
                band = pre(band)
            top, bottom = lo - (r0 - pad), (r1 + pad) - hi
            if top or bottom:
                band = mx.pad(band, [(0, 0), (top, bottom), (0, 0), (0, 0)])
            y = mx.conv2d(band, w, padding=(0, pad)) + b
            if res is not None:
                y = y + (res[:, r0:r1] if (r0, r1) != (0, h) else res)
            if rows < h:
                mx.eval(y)
            outs.append(y)
        return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)

    def _norm(self, name: str, *, silu: bool) -> Callable[[mx.array], mx.array]:
        g = self.p[name + ".gamma"]
        return (lambda x: _l2norm_silu(x, g)) if silu else (lambda x: _l2norm(x, g))

    def res(self, p: str, x: mx.array) -> mx.array:
        """`VAE2D.res`, each convolution banded with its norm folded in, evaluated per convolution."""
        h = self.conv(p + ".conv_shortcut", x) if p + ".conv_shortcut" in self else x
        y = self.conv(p + ".conv1", x, pre=self._norm(p + ".norm1", silu=True))
        mx.eval(y, h)
        y = self.conv(p + ".conv2", y, pre=self._norm(p + ".norm2", silu=True), res=h)
        mx.eval(y)
        return y

    def attn(self, p: str, x: mx.array) -> mx.array:
        """`VAE2D.attn`: single-head attention over the pixels, float32 in query chunks (head dim 384)."""
        n, h, w, c = x.shape
        qkv = self.conv(p + ".to_qkv", x, pre=self._norm(p + ".norm", silu=False)).reshape(h * w, 3 * c)
        q, k, v = (a.astype(F32) for a in mx.split(qkv, 3, axis=-1))
        kt = k.T
        rows = max(1, _ATTN_SCORES // (4 * h * w))
        outs = []
        for r in range(0, h * w, rows):
            s = mx.matmul(q[r : r + rows] * (c**-0.5), kt)
            o = mx.matmul(mx.softmax(s, axis=-1), v).astype(BF)
            mx.eval(o)
            outs.append(o)
        y = mx.concatenate(outs).reshape(n, h, w, c)
        return self.conv(p + ".proj", y, res=x)

    def mid(self, p: str, x: mx.array) -> mx.array:
        """Mid block: residual, attention, residual."""
        x = self.res(p + ".resnets.0", x)
        x = self.attn(p + ".attentions.0", x)
        mx.eval(x)
        return self.res(p + ".resnets.1", x)

    def encode(self, x: mx.array, noise: mx.array) -> mx.array:
        """`VAE2D.encode`: `[1, H, W, 3]` bf16 in `[-1, 1]`, noise `[1, h, w, 16]` -> normalised latent (bf16)."""
        x = self.conv("encoder.conv_in", x)
        mx.eval(x)
        j = 0
        while True:
            p = f"encoder.down_blocks.{j}"
            if p + ".resample.1" in self:
                x = self.conv(p + ".resample.1", x, stride=2)
                mx.eval(x)
            elif p + ".conv1" in self:
                x = self.res(p, x)
            else:
                break
            j += 1
        x = self.mid("encoder.mid_block", x)
        x = self.conv("encoder.conv_out", x, pre=self._norm("encoder.norm_out", silu=True))
        x = self.conv("quant_conv", x)
        mean, logvar = x[..., :16], x[..., 16:]
        std = mx.exp(0.5 * mx.clip(logvar, -30.0, 20.0))
        return (mean + std * noise.astype(BF) - self.mean) * self.std_inv

    def decode(self, z: mx.array) -> mx.array:
        """`VAE2D.decode`: normalised latent -> `[1, H, W, 3]` in `[-1, 1]` (bf16)."""
        x = self.conv("post_quant_conv", z / self.std_inv + self.mean)
        x = self.conv("decoder.conv_in", x)
        x = self.mid("decoder.mid_block", x)
        for i in range(4):
            p = f"decoder.up_blocks.{i}"
            for r in range(3):
                x = self.res(f"{p}.resnets.{r}", x)
            if p + ".upsamplers.0.resample.1" in self:
                x = self.conv(p + ".upsamplers.0.resample.1", x, up=True)
                mx.eval(x)
        x = self.conv("decoder.conv_out", x, pre=self._norm("decoder.norm_out", silu=True))
        return mx.clip(x, -1.0, 1.0)


class Net:
    """`net.Net` on MLX: VAE posterior sample, 2 x 2 packing, one velocity prediction, `latent - velocity`, decode."""

    def __init__(self, path: str | Path, *, stream: bool = True) -> None:
        """Open the lean checkpoint; read the VAE and the transformer's constants (and its blocks unless `stream`)."""
        w = Weights(path)
        self.vae = VAE(w)
        self.dit = Transformer(w, stream=stream)

    def __call__(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """`[1, 3, H, W]` in `[-1, 1]` (CPU float32), noise `[1, 16, H / 8, W / 8]` -> log depth `[1, 1, H, W]` float32."""
        xm = to_mx(x.float()).transpose(0, 2, 3, 1).astype(BF)
        lat = self.vae.encode(xm, to_mx(noise.float()).transpose(0, 2, 3, 1))
        mx.eval(lat)
        _, h, w, c = lat.shape
        hp, wp = h // PATCH, w // PATCH
        packed = lat.reshape(hp, PATCH, wp, PATCH, c).transpose(0, 2, 4, 1, 3).reshape(hp * wp, c * PATCH * PATCH)
        pred = self.dit(packed, hp, wp)
        pred = pred.reshape(hp, wp, c, PATCH, PATCH).transpose(0, 3, 1, 4, 2).reshape(1, h, w, c)
        y = self.vae.decode(lat - pred)
        d = y.astype(F32).mean(axis=-1).astype(BF).astype(F32)[:, None]  # bf16 mean over the three channels
        mx.eval(d)
        return torch.from_numpy(np.array(d))
