"""ViTMatte runners: weights -> network and the portable / kernels forward passes.

Inference as the transformers `VitMatteImageProcessor` + `VitMatteForImageMatting`: the image normalised to [-1, 1],
the trimap in [0, 1] as a 4th channel, both zero-padded bottom-right to a multiple of 32; the network's alpha is
cropped back to the image (no resize).

`portable`: the backbone runs channels-last (`[B, H, W, C]` throughout, as upstream's blocks do inside) with the
attention's decomposed relative-position bias folded into an augmented dot product (`q' = [q * scale, rel_h, rel_w]`,
`k' = [k, onehot(kh), onehot(kw)]`, padded to a multiple of 8; `v` zero-padded to the same head dim) so SDPA (flash)
never materialises the N x N matrix (8160 tokens at 1080p); the residual blocks' channel LayerNorms through
`F.layer_norm` (float32 statistics in float16). Interpolated position embeddings and relative tables are cached per
token grid. The decoder is upstream's modules. fp32 or fp16 weights and activations.

`kernels`: fp16; the backbone's GEMMs on `dpt_fused` (fused bias / GELU / residual epilogues, fp16 accumulation; the
patch embedding with fp32 accumulation) and its 3 x 3 convs as NHWC implicit GEMMs; the decoder's conv + BatchNorm +
ReLU as one cuDNN fp16-accumulate conv with an fp32 affine epilogue each (`shared.cnn_kernels`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch
import torch.nn.functional as F

from fastforks.kernels.dpt_fused import ACT_GELU
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, prepare_convs
from fastforks.shared.cnn_kernels import load as load_cnn

from .net.vitmatte import Config, VitMatteForImageMatting, get_rel_pos

if TYPE_CHECKING:
    from fastforks._core.cards import Io
    from fastforks.kernels.dpt_fused import DptFused
    from fastforks.matting._runner import Context

_PAD = 32  # fixed: the processor's `size_divisibility`
Rel = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        hidden_size: Backbone width (S: 384, B: 768).
        num_heads: Attention heads (head dim 64).
    """

    hidden_size: int
    num_heads: int


def _cpu_net(ctx: Context) -> VitMatteForImageMatting:
    p = ctx.params
    net = VitMatteForImageMatting(Config(hidden_size=p.hidden_size, num_attention_heads=p.num_heads))
    sd = {k: v for k, v in ctx.state_dict().items() if not k.endswith("num_batches_tracked")}
    net.load_state_dict(sd)
    return net.eval()


def pixel_values(image: torch.Tensor, trimap: torch.Tensor, io: Io) -> torch.Tensor:
    """`[B, 3, H, W]` RGB and `[B, 1, H, W]` trimap in `[0, 1]` -> the padded `[B, 4, H', W']` network input."""
    x = torch.cat([scale(image, io), trimap], 1)
    h, w = x.shape[-2:]
    return F.pad(x, (0, -w % _PAD, 0, -h % _PAD))


# ------------------------------------------------------------------------------------------- augmented attention
def _onehot_keys(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`[h * w, h + w]`: row `(kh, kw)` has a 1 in column `kh` and in column `h + kw`."""
    eye_h = torch.eye(h, device=device, dtype=dtype)
    eye_w = torch.eye(w, device=device, dtype=dtype)
    return torch.cat([eye_h.repeat_interleave(w, 0), eye_w.repeat(h, 1)], 1)


def _augmented(qkv: torch.Tensor, nh: int, rel: Rel, hw: tuple[int, int]) -> tuple[torch.Tensor, ...]:
    """Packed qkv `[B, N, 3C]` -> SDPA operands `q', k', v'` `[B, heads, N, Dp]` (to run with `scale=1`)."""
    rh, rw, onehot = rel
    b, n, c3 = qkv.shape
    d = c3 // 3 // nh
    h, w = hw
    q, k, v = qkv.view(b, n, 3, nh, d).unbind(2)
    r_q = q.reshape(b, h, w, nh, d)
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", r_q, rh).reshape(b, nh, n, h)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", r_q, rw).reshape(b, nh, n, w)
    pad = -(d + h + w) % 8
    qt = q.transpose(1, 2)
    zeros = qt.new_zeros(b, nh, n, pad)
    qa = torch.cat([qt * d**-0.5, rel_h, rel_w, zeros], -1)
    ka = torch.cat([k.transpose(1, 2), onehot.expand(b, nh, n, h + w), zeros], -1)
    va = torch.cat([v.transpose(1, 2), qt.new_zeros(b, nh, n, h + w + pad)], -1)
    return qa, ka, va


def _attend(qkv: torch.Tensor, heads: int, rel: Rel, hw: tuple[int, int]) -> torch.Tensor:
    """Attention of packed qkv `[B, N, 3C]` -> `[B, N, C]` (before the output projection)."""
    b, n, c3 = qkv.shape
    qa, ka, va = _augmented(qkv, heads, rel, hw)
    o = F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., : c3 // 3 // heads]
    return o.transpose(1, 2).reshape(b, n, c3 // 3)


def _partition(x: torch.Tensor, ws: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """`[B, H, W, C]` -> zero-padded `[B * windows, ws, ws, C]` and the padded size (upstream `window_partition`)."""
    b, h, w, c = x.shape
    ph, pw = -h % ws, -w % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    hp, wp = h + ph, w + pw
    x = x.view(b, hp // ws, ws, wp // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c)
    return x, (hp, wp)


def _unpartition(x: torch.Tensor, ws: int, pad_hw: tuple[int, int], hw: tuple[int, int]) -> torch.Tensor:
    """Inverse of `_partition` (upstream `window_unpartition`)."""
    (hp, wp), (h, w) = pad_hw, hw
    b = x.shape[0] // (hp * wp // ws // ws)
    x = x.view(b, hp // ws, wp // ws, ws, ws, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    return x[:, :h, :w] if (hp > h or wp > w) else x


class _Geometry:
    """Per token grid: the interpolated position embedding and every layer's relative tables (cached)."""

    def __init__(self, net: VitMatteForImageMatting, dtype: torch.dtype, device: torch.device) -> None:
        self.net, self.dtype, self.device = net, dtype, device
        self.cache: dict[tuple[int, int], tuple[torch.Tensor, list[Rel]]] = {}

    def __call__(self, h: int, w: int) -> tuple[torch.Tensor, list[Rel]]:
        if (h, w) not in self.cache:
            with torch.no_grad():
                emb = self.net.backbone.embeddings
                pos = emb.get_absolute_positions(emb.position_embeddings.float(), True, h, w)  # noqa: FBT003
                rel = [self._rel(layer, h, w) for layer in self.net.backbone.encoder.layer]
            self.cache[h, w] = pos.to(self.device, self.dtype).contiguous(), rel
        return self.cache[h, w]

    def _rel(self, layer: Any, h: int, w: int) -> Rel:  # noqa: ANN401 - upstream VitDetLayer
        a, ws = layer.attention, int(layer.window_size)
        sh, sw = (ws, ws) if ws else (h, w)
        rh = get_rel_pos(sh, sh, a.rel_pos_h.float()).to(self.device, self.dtype)
        rw = get_rel_pos(sw, sw, a.rel_pos_w.float()).to(self.device, self.dtype)
        return rh, rw, _onehot_keys(sh, sw, self.device, self.dtype)


# ------------------------------------------------------------------------------------------------ runners
class _Matting(Runner):
    """Pad, network (image + trimap), crop."""

    Params = Params
    trimap: ClassVar = "required"
    compiled_methods = ("_net",)
    net: VitMatteForImageMatting
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB and its `[B, 1, H, W]` trimap in `[0, 1]`."""
        if trimap is None:  # trimap = "required": the task checks it before
            msg = "ViTMatte needs a trimap"
            raise RuntimeError(msg)
        h, w = image.shape[-2:]
        return {"alpha": self._net(pixel_values(image, trimap, self.ctx.io))[:, 0, :h, :w].float()}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: image `[B, 3, H, W]`, trimap `[B, 1, H, W]` -> `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, trimap: self(image, trimap)["alpha"])


class Portable(_Matting):
    """Channels-last backbone with augmented-SDPA attention, upstream decoder; float32 or float16."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx).to(ctx.device, self.dtype)
        self.geometry = _Geometry(self.net, self.dtype, ctx.device)

    def _block(self, layer: Any, x: torch.Tensor, rel: Rel) -> torch.Tensor:  # noqa: ANN401 - upstream VitDetLayer
        a, ws = layer.attention, int(layer.window_size)
        y = layer.norm1(x)
        if ws:
            hw = (y.shape[1], y.shape[2])
            y, pad_hw = _partition(y, ws)
            n = y.shape[0]
            y = a.proj(_attend(a.qkv(y.view(n, ws * ws, -1)), a.num_heads, rel, (ws, ws)))
            y = _unpartition(y.view(n, ws, ws, -1), ws, pad_hw, hw)
        else:
            b, h, w, c = y.shape
            y = a.proj(_attend(a.qkv(y.view(b, h * w, c)), a.num_heads, rel, (h, w))).view(b, h, w, c)
        x = x + y
        x = x + layer.mlp(layer.norm2(x))
        if layer.use_residual_block:  # conv1x1, LN, GELU, conv3x3, LN, GELU, conv1x1, LN; channels-last
            r = layer.residual
            y = F.linear(x, r.conv1.weight.flatten(1))
            y = F.gelu(F.layer_norm(y, (y.shape[-1],), r.norm1.weight, r.norm1.bias, r.norm1.eps))
            y = r.conv2(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            y = F.gelu(F.layer_norm(y, (y.shape[-1],), r.norm2.weight, r.norm2.bias, r.norm2.eps))
            y = F.linear(y, r.conv3.weight.flatten(1))
            x = x + F.layer_norm(y, (y.shape[-1],), r.norm3.weight, r.norm3.bias, r.norm3.eps)
        return x

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype)
        emb = self.net.backbone.embeddings
        h = emb.projection(x).permute(0, 2, 3, 1)
        pos, rel = self.geometry(h.shape[1], h.shape[2])
        h = h + pos
        for layer, r in zip(self.net.backbone.encoder.layer, rel, strict=True):
            h = self._block(layer, h, r)
        return self.net.decoder(h.permute(0, 3, 1, 2), x)


def _half(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.detach().half().contiguous().to(device)


class _FusedLayer:
    """One backbone layer's fp16 weights in the `dpt_fused` layouts."""

    def __init__(self, layer: Any, device: torch.device) -> None:  # noqa: ANN401 - upstream VitDetLayer
        a = layer.attention
        self.ws, self.heads = int(layer.window_size), int(a.num_heads)
        self.n1 = (_half(layer.norm1.weight, device), _half(layer.norm1.bias, device), float(layer.norm1.eps))
        self.n2 = (_half(layer.norm2.weight, device), _half(layer.norm2.bias, device), float(layer.norm2.eps))
        self.qkv = (_half(a.qkv.weight, device), _half(a.qkv.bias, device))
        self.proj = (_half(a.proj.weight, device), _half(a.proj.bias, device))
        self.fc1 = (_half(layer.mlp.fc1.weight, device), _half(layer.mlp.fc1.bias, device))
        self.fc2 = (_half(layer.mlp.fc2.weight, device), _half(layer.mlp.fc2.bias, device))
        self.res = None
        if layer.use_residual_block:
            r = layer.residual
            norms = [
                (_half(n.weight, device), _half(n.bias, device), float(n.eps)) for n in (r.norm1, r.norm2, r.norm3)
            ]
            self.res = (
                _half(r.conv1.weight.flatten(1), device),
                _half(r.conv2.weight.permute(0, 2, 3, 1), device),  # [K, R, S, C]
                _half(r.conv3.weight.flatten(1), device),
                norms,
            )


class Kernels(_Matting):
    """fp16: backbone on `dpt_fused` GEMMs / convs with SDPA (flash) attention, decoder on fused cuDNN convs."""

    def __init__(self, ctx: Context) -> None:
        """Convert the backbone (on the CPU) and the decoder (on the device)."""
        super().__init__(ctx)
        dev = ctx.device
        net = _cpu_net(ctx)
        self.k: DptFused = load_dpt_fused(dev)
        load_cnn(dev)
        emb = net.backbone.embeddings
        conv = emb.projection
        self.ps = int(conv.kernel_size[0])
        self.patch = (_half(conv.weight.flatten(1), dev), _half(cast("torch.Tensor", conv.bias), dev))
        self.layers = [_FusedLayer(layer, dev) for layer in net.backbone.encoder.layer]
        self.dtype = torch.float16
        self.geometry = _Geometry(net, self.dtype, dev)  # position embedding, relative tables: from the CPU weights
        self.net = net  # the backbone's weights stay on the CPU (fp16 copies above); the decoder runs on the device
        dec = net.decoder.to(dev)
        self.convs: list[Any] = [*dec.convstream.convs, *(f.conv for f in dec.fusion_blocks)]
        self.affine = [bn_affine(c.batch_norm) for c in self.convs]
        head = dec.matting_head.matting_convs
        self.head_affine = bn_affine(head[1], head[0])
        prepare_convs(dec)  # fp16, channels-last; the >= 128-channel convs on the CUTLASS fused kernel

    def _attention(self, b: _FusedLayer, x: torch.Tensor, rel: Rel) -> torch.Tensor:
        """`[B, h, w, C]` (normed, partitioned) -> attention output before proj, rows `[B * h * w, C]`."""
        bw, h, w, c = x.shape
        qkv = self.k.gemm(x.reshape(-1, c), *b.qkv).view(bw, h * w, 3 * c)
        return _attend(qkv, b.heads, rel, (h, w)).reshape(-1, c)

    def _layer(self, b: _FusedLayer, x: torch.Tensor, rel: Rel) -> torch.Tensor:
        k = self.k
        bs, th, tw, c = x.shape
        y = F.layer_norm(x, (c,), *b.n1[:2], b.n1[2])
        if b.ws:
            y, pad_hw = _partition(y, b.ws)
            o = k.gemm(self._attention(b, y, rel), *b.proj).view(-1, b.ws, b.ws, c)
            x = x + _unpartition(o, b.ws, pad_hw, (th, tw))
            rows = x.view(-1, c)
        else:
            rows = k.gemm(self._attention(b, y, rel), *b.proj, res=x.view(-1, c))
        hid = k.gemm(F.layer_norm(rows, (c,), *b.n2[:2], b.n2[2]), *b.fc1, act=ACT_GELU)
        rows = k.gemm(hid, *b.fc2, res=rows, out=rows)
        if b.res is not None:
            w1, w2, w3, (n1, n2, n3) = b.res
            y = F.gelu(F.layer_norm(k.gemm(rows, w1), (w1.shape[0],), *n1[:2], n1[2]))
            y = k.conv(y.view(bs, th, tw, -1).contiguous(), w2, padding=1)
            y = F.gelu(F.layer_norm(y, (w2.shape[0],), *n2[:2], n2[2])).view(-1, w2.shape[0])
            y = F.layer_norm(k.gemm(y, w3), (c,), *n3[:2], n3[2])
            rows = rows.add_(y)
        return rows.view(bs, th, tw, c)

    def _decoder(self, feats: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        relu = ACT["relu"]
        n = len(self.net.decoder.convstream.convs)
        details = [x]
        for conv, (s, t) in zip(self.convs[:n], self.affine[:n], strict=True):
            details.append(conv_ep(conv.conv, details[-1], s, t, relu))
        y = feats
        for i, (conv, (s, t)) in enumerate(zip(self.convs[n:], self.affine[n:], strict=True)):
            up = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = conv_ep(conv.conv, torch.cat([details[n - i], up], 1).contiguous(memory_format=CL), s, t, relu)
        head = self.net.decoder.matting_head.matting_convs
        y = conv_ep(head[0], y, *self.head_affine, relu)
        return conv_ep(head[3], y).float().sigmoid()

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        k, ps = self.k, self.ps
        b, cin, hh, ww = x.shape
        th, tw = hh // ps, ww // ps
        x = x.half()
        p = x.view(b, cin, th, ps, tw, ps).permute(0, 2, 4, 1, 3, 5).reshape(b * th * tw, cin * ps * ps)
        pos, rel = self.geometry(th, tw)
        h = k.gemm(p, *self.patch, acc32=True).view(b, th, tw, -1).add_(pos)
        for layer, r in zip(self.layers, rel, strict=True):
            h = self._layer(layer, h, r)
        return self._decoder(h.permute(0, 3, 1, 2), x.contiguous(memory_format=CL))


RUNNERS = {"portable": Portable, "kernels": Kernels}
