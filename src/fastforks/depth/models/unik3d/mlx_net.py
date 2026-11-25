"""UniK3D on Apple MLX, written in `mlx.core` operations (`MlxUniK3D(net, dtype)`): network input -> radius, rays.

`UniK3DNet.forward`, channels last. Written out instead of replayed from an exported graph: the export was most of a
3.9 s first call at 994 x 1482 whose network runs in ~0.95 s.

- the encoder: the shared DINOv2 blocks (`fastforks.shared.mlx_vit`, LayerScale folded; upstream's eps 1e-5), the
  position table as upstream's (bicubic by output size, float32 on the host, kept per grid);
- the angular module, the camera rays and the ray embedding in float32: 18 camera tokens, then per pixel the
  trigonometry and the spherical harmonics; the antialiased downsampling of the rays to the token grid is two matrix
  products (`mlx_graph._aa_matrix`'s weights);
- the attention blocks with their LayerNorms' affines folded into the projections after them, LayerScale into the
  last ones;
- the kernel == stride transposed convolutions as one GEMM and a depth-to-space, the residual units' gamma folded into
  their second conv, 1x1 convs as GEMMs, the 3x3 ones through `mlx_graph._conv2d` (its channel padding onto MLX's fast
  tiles), the bilinear resizes on the shared kernel (`bilinear_nhwc`);
- the two heads' LayerNorm + Linear as one GEMM of the shared normalised features.

The run dtype covers the encoder and the radial module (the torch fp16 runner's pure-fp16 encoder and decoder).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fastforks._core.mlx import mx, to_mx, to_torch
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_graph as mg
from fastforks.shared.mlx_vit import block, dinov2_params, gelu, patches

from . import net as un

if TYPE_CHECKING:
    from torch import nn

__all__ = ["MlxUniK3D"]

ONE, ZERO = (1, 1), (0, 0)
EPS = 1e-5  # every LayerNorm (the encoder's set by upstream, the decoder's torch's default)
SLOPE = 0.01  # nn.LeakyReLU default
TABLES = 4  # position tables kept (one per token grid, least recently used dropped)

leaky = mx.compile(lambda x: mx.maximum(x, SLOPE * x), shapeless=True)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().double().numpy()


def _arr(v: np.ndarray, dtype: mx.Dtype) -> mx.array:
    """`v` in `dtype`, evaluated (cast one by one: only the run-dtype copy accumulates)."""
    a = mx.array(np.ascontiguousarray(v, np.float32))
    if dtype != mx.float32:
        a = a.astype(dtype)
    mx.eval(a)
    return a


def _folded(norm: nn.LayerNorm, lin: nn.Linear, dtype: mx.Dtype, gamma: np.ndarray | float = 1.0) -> tuple:
    """`lin(norm(x)) * gamma` as `(weight [C, O], bias [O])` on the affine-free normalised `x`."""
    w = _np(lin.weight) * np.reshape(gamma, (-1, 1))
    b = w @ _np(norm.bias) + (0.0 if lin.bias is None else _np(lin.bias) * gamma)
    return _arr((w * _np(norm.weight)[None]).T, dtype), _arr(b, dtype)


def _ls(m: nn.Module) -> np.ndarray | float:
    return _np(m.gamma) if isinstance(m, un.LayerScale) else 1.0


def _block(blk: un.AttentionBlock, dtype: mx.Dtype) -> dict[str, Any]:
    """An `AttentionBlock`'s weights: norms folded into q / kv / proj1, LayerScale into out / proj2."""
    g1, g2 = _ls(blk.ls1_1), _ls(blk.ls2)
    m = blk.mlp
    return {
        "q": _folded(blk.norm_attnx, blk.q, dtype),
        "kv": _folded(blk.norm_attnctx, blk.kv, dtype),
        "out": _arr((_np(blk.out.weight) * np.reshape(g1, (-1, 1))).T, dtype),
        "mlp1": _folded(m.norm, m.proj1, dtype),
        "mlp2": (_arr((_np(m.proj2.weight) * np.reshape(g2, (-1, 1))).T, dtype), _arr(_np(m.proj2.bias) * g2, dtype)),
    }


def _ln(x: mx.array) -> mx.array:
    return mx.fast.layer_norm(x, None, None, EPS)


def _attend(p: dict[str, Any], x: mx.array, context: mx.array, pos: mx.array | None = None) -> mx.array:
    """`AttentionBlock.forward(x, context, pos_embed)` on tokens `[B, N, C]`."""
    b, n, c = x.shape
    d = c // un.HEADS
    kv = mx.addmm(p["kv"][1], _ln(context), p["kv"][0])
    kv = kv.reshape(b, context.shape[1], 2, un.HEADS, d).transpose(2, 0, 3, 1, 4)
    q = mx.addmm(p["q"][1], _ln(x), p["q"][0])
    if pos is not None:
        q = q + pos
    q = q.reshape(b, n, un.HEADS, d).transpose(0, 2, 1, 3)
    o = mx.fast.scaled_dot_product_attention(q, kv[0], kv[1], scale=d**-0.5)
    x = x + o.transpose(0, 2, 1, 3).reshape(b, n, c) @ p["out"]
    return x + mx.addmm(p["mlp2"][1], gelu(mx.addmm(p["mlp1"][1], _ln(x), p["mlp1"][0])), p["mlp2"][0])


def _mlp(p: tuple, x: mx.array) -> mx.array:
    """`MLP.forward` (the angular module's output MLPs)."""
    w1, b1, w2, b2 = p
    return mx.addmm(b2, gelu(mx.addmm(b1, _ln(x), w1)), w2)


def camera_rays(fov: mx.array, sh: mx.array, h: int, w: int) -> mx.array:
    """`net.camera_rays` (float32): field of view / principal point `[B, 4]`, SH `[B, 15, 3]` -> rays `[B, H W, 3]`."""
    u = mx.linspace(0.5, w - 0.5, w)[None, None]
    v = mx.linspace(0.5, h - 0.5, h)[None, :, None]
    lon = (u - fov[:, 2, None, None]) / w * fov[:, 0, None, None]  # [B, 1, W]
    lat = (v - fov[:, 3, None, None]) / h * fov[:, 1, None, None]  # [B, H, 1]
    cl = mx.cos(lat)
    sx, sy, sz = cl * mx.sin(lon), mx.broadcast_to(-mx.sin(lat), (fov.shape[0], h, w)), cl * mx.cos(lon)
    norm = mx.maximum(mx.sqrt(sx * sx + sy * sy + sz * sz), 1e-5)
    sx, sy, sz = sx / norm, sy / norm, sz / norm
    rays = mx.stack(un.rsh_terms(sx, sy, sz), -1).reshape(fov.shape[0], h * w, 15) @ sh
    return rays / mx.maximum(mx.linalg.norm(rays, axis=-1, keepdims=True), 1e-5)


def embed_rays(rays: mx.array, size: tuple[int, int], grid: tuple[int, int]) -> mx.array:
    """`net.embed_rays` (float32): rays `[B, H W, 3]` -> Fourier embedding `[B, h w, 512]` of the token grid."""
    b, (hh, ww), (h, w) = rays.shape[0], size, grid
    if size != grid:  # antialiased bilinear, one axis per matrix product
        wy = mx.array(mg._aa_matrix(hh, h, align=False, scale=None))  # noqa: SLF001 - the translator's resize weights
        wx = mx.array(mg._aa_matrix(ww, w, align=False, scale=None))  # noqa: SLF001
        t = (wy @ rays.reshape(b, hh, ww * 3)).reshape(b, h, ww, 3)
        rays = mx.swapaxes(mx.swapaxes(t, 2, 3) @ wx.T, 2, 3).reshape(b, h * w, 3)
    rays = rays / mx.maximum(mx.linalg.norm(rays, axis=-1, keepdims=True), 1e-4)
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    x_clipped = mx.maximum(mx.abs(x), 1e-3) * mx.where(x >= 0, 1.0, -1.0)
    angles = mx.stack([mx.arccos(z), mx.arctan2(y, x_clipped)], -1)
    bands = un.HIDDEN // 2
    scales = to_mx(2.0 ** torch.linspace(0.0, math.log2(max(grid) // 2), steps=bands))  # upstream's float32 bands
    return mx.sin(angles[..., None] * scales * math.pi).reshape(b, h * w, un.HIDDEN)


def _conv_w(c: nn.Conv2d, dtype: mx.Dtype, gamma: torch.Tensor | None = None) -> tuple[mx.array, mx.array]:
    """A conv's `(weight OHWI, bias)`, scaled per output channel by `gamma`."""
    g = np.ones(c.out_channels) if gamma is None else _np(gamma).reshape(-1)
    w = _np(c.weight) * g[:, None, None, None]
    return _arr(w.transpose(0, 2, 3, 1), dtype), _arr(_np(c.bias) * g, dtype)


def _conv(x: mx.array, wb: tuple[mx.array, mx.array], *, reflect: bool = False) -> mx.array:
    """3x3 conv with bias: zero padding, or upstream's one-pixel reflection (`reflect`)."""
    if reflect:
        x = mx.concatenate([x[:, 1:2], x, x[:, -2:-1]], axis=1)
        x = mx.concatenate([x[:, :, 1:2], x, x[:, :, -2:-1]], axis=2)
    return mg._conv2d(x, wb[0], ONE, ZERO if reflect else ONE, ONE, 1) + wb[1]  # noqa: SLF001 - its conv dispatch


class MlxUniK3D:
    """The network: normalised `[B, 3, H, W]` (sides multiples of 14) -> float32 log-radius, log-confidence, rays."""

    def __init__(self, net: un.UniK3DNet, dtype: str) -> None:
        """The weights of `net` (float32) in `dtype`; the camera part in float32."""
        self.dtype = dt = getattr(mx, dtype)
        f32 = mx.float32
        vit: Any = net.pixel_encoder
        self.pe = vit.pos_embed.detach().float().clone()
        self.prefix = (vit.cls_token.detach().float()[0, 0], vit.patch_embed.proj.bias.detach().float().clone())
        self.heads = vit.blocks[0].attn.num_heads
        self.layers = tuple(i + 1 for i in un.LAYERS)  # 1-based block outputs
        self.vit = dinov2_params(dict(vit.state_dict()), un.PATCH, dt)
        for k in ("pos.cls", "pos.grid", "patch.bias"):  # `table` (float32 on the host)
            del self.vit[k]
        self.tables: OrderedDict[tuple[int, int], mx.array] = OrderedDict()
        dec: Any = net.pixel_decoder
        lin = lambda m, d: (_arr(_np(m.weight).T, d), _arr(_np(m.bias), d))  # noqa: E731
        self.adapters = [lin(m, dt) for m in dec.input_adapter.input_adapters]
        self.camera_adapters = [lin(m, f32) for m in dec.camera_token_adapter.input_adapters]
        ang = dec.angular_module
        self.angular = [_block(b, f32) for b in (ang.aggregate1, ang.aggregate2)]
        self.latents_pos = _arr(_np(ang.latents_pos)[0], f32)
        projs = (ang.project_pin, ang.project_deg1, ang.project_deg2, ang.project_deg3)
        self.project = [_arr(_np(m.weight).T, f32) for m in projs]
        self.outs = []
        for m in (ang.out_pinhole, ang.out_deg1, ang.out_deg2, ang.out_deg3):
            self.outs.append((*_folded(m.norm, m.proj1, f32), *lin(m.proj2, f32)))
        rm = dec.radial_module
        self.prompts = [_block(p.layers[0], dt) for p in rm.prompt_camera]
        self.latents = lin(rm.to_latents, dt)
        self.pf = []
        for t in rm.process_features:  # [Cin, Cout, s, s] -> [Cin, (a, b, Cout)]
            s, w = t.stride[0], _np(t.weight)
            self.pf.append(
                (s, _arr(w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1), dt), _arr(np.tile(_np(t.bias), s * s), dt))
            )
        self.ups = []
        for u in rm.ups:
            rcus = [(_conv_w(r.conv1, dt), _conv_w(r.conv2, dt, r.gamma)) for r in u.convs]
            self.ups.append((rcus, lin_1x1(u.up[0], dt)))
        heads = [
            (rm.depth_mlp[-1], rm.to_depth_lr, rm.to_depth_hr),
            (rm.confidence_mlp, rm.to_confidence_lr, rm.to_confidence_hr),
        ]
        mlps = [_folded(mlp[0], mlp[1], dt) for mlp, _, _ in heads]  # both heads: one GEMM of the normed features
        self.head_mlp = (mx.concatenate([w for w, _ in mlps], axis=1), mx.concatenate([b for _, b in mlps]))
        self.head_convs = [(_conv_w(lr, dt), _conv_w(hr[0], dt), lin_1x1(hr[2], dt)) for _, lr, hr in heads]

    def table(self, h: int, w: int) -> mx.array:
        """`[1 + h w, C]` in the run dtype: row 0 = cls token + its position, the rest = positions + patch bias."""
        key = (h, w)
        if key not in self.tables:
            t = un.pos_table(self.pe, h, w)[0].clone()  # float32, upstream's bicubic by output size
            t[0] += self.prefix[0]
            t[1:] += self.prefix[1]
            self.tables[key] = to_mx(t).astype(self.dtype)
            mx.eval(self.tables[key])
            while len(self.tables) > TABLES:
                self.tables.popitem(last=False)
        self.tables.move_to_end(key)
        return self.tables[key]

    def _encode(self, x: mx.array, h: int, w: int) -> list[mx.array]:
        """Normalised `[B, 3, H, W]` -> the normed tokens `[B, 1 + h w, C]` of `LAYERS`."""
        p, table = self.vit, self.table(h, w)
        t = patches(x.astype(self.dtype), un.PATCH) @ p["patch.weight"] + table[1:]
        t = mx.concatenate([mx.broadcast_to(table[:1], (x.shape[0], 1, t.shape[-1])), t], axis=1)
        outs = []
        for i in range(max(self.layers)):
            t = block(p, i, t, self.heads, EPS)
            if i + 1 in self.layers:
                outs.append(mx.fast.layer_norm(t, p["norm.weight"], p["norm.bias"], EPS))
            mx.async_eval(t, *outs)
        return outs

    def _angular(self, cls: list[mx.array], size: tuple[int, int]) -> tuple[mx.array, mx.array]:
        """`AngularModule.forward` (float32): class tokens of the levels -> fov / centre `[B, 4]`, SH `[B, 15, 3]`."""
        b = cls[0].shape[0]
        adapted = [
            mx.addmm(bias, c[:, 0].astype(mx.float32), w)
            for c, (w, bias) in zip(cls, self.camera_adapters, strict=True)
        ]
        tokens = mx.concatenate(
            [(a @ w).reshape(b, n, un.HIDDEN) for a, w, n in zip(adapted, self.project, un.SH_PARAMS, strict=True)],
            axis=1,
        )
        for p in self.angular:
            tokens = _attend(p, tokens, tokens, self.latents_pos)
        ends = np.cumsum(un.SH_PARAMS)
        parts = [tokens[:, e - n : e] for e, n in zip(ends, un.SH_PARAMS, strict=True)]
        x = _mlp(self.outs[0], parts[0])[..., 0]
        sh = mx.concatenate([_mlp(o, t) for o, t in zip(self.outs[1:], parts[1:], strict=True)], axis=1)
        h, w = size
        hfov = mx.sigmoid(x[:, 0] - 1.1)
        k = mx.stack([hfov, hfov * (h / w), mx.sigmoid(x[:, 1]), mx.sigmoid(x[:, 2])], 1)
        return mx.array([2 * math.pi, 2 * math.pi, w, h], dtype=mx.float32) * k, sh

    def _radial(
        self, feats: list[mx.array], rays: mx.array, size: tuple[int, int], grid: tuple[int, int], *, confidence: bool
    ) -> list[mx.array]:
        """`RadialModule.forward` -> the float32 log maps `[B, 1, H, W]` (depth, then confidence)."""
        b, (h, w) = feats[0].shape[0], grid
        emb = embed_rays(rays, size, grid).astype(self.dtype)
        cond = []
        for p, f in zip(self.prompts, feats, strict=True):
            cond.append(_attend(p, f, emb))
            mx.async_eval(cond[-1])
        x = mx.addmm(self.latents[1], cond[0], self.latents[0]).reshape(b, h, w, -1)
        for (s, wg, bg), (rcus, (wu, bu)), c in zip(self.pf, self.ups, cond[1:], strict=True):
            y = mx.addmm(bg, c, wg).reshape(b, h, w, s, s, -1).transpose(0, 1, 3, 2, 4, 5)
            x = x + y.reshape(x.shape)
            for c1, c2 in rcus:
                x = x + _conv(leaky(_conv(leaky(x), c1)), c2)
            x = mx.addmm(bu, x, wu)
            x = ops.bilinear_nhwc(x, (2 * x.shape[1], 2 * x.shape[2]), (0.5, 0.5), align=False)
            mx.async_eval(x)
        n = 2 if confidence else 1
        c = self.head_mlp[0].shape[1] // 2
        z = mx.addmm(self.head_mlp[1][: n * c], _ln(x), self.head_mlp[0][:, : n * c])
        hh, ww = size
        ratios = ((x.shape[1] - 1) / (hh - 1), (x.shape[2] - 1) / (ww - 1))
        outs = []
        for i, (lr, hr, (wo, bo)) in enumerate(self.head_convs[:n]):
            t = _conv(z[..., i * c : (i + 1) * c], lr, reflect=True)
            t = ops.bilinear_nhwc(t, size, ratios, align=True)
            t = mx.addmm(bo, leaky(_conv(t, hr, reflect=True)), wo)
            outs.append(t.astype(mx.float32).reshape(b, 1, hh, ww))
        return outs

    def __call__(
        self, x: torch.Tensor, rays: torch.Tensor | None, *, confidence: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """`UniK3DNet.forward` of a float32 CPU input: (log-radius, log-confidence or None, rays), float32 CPU."""
        mg._KERNELS, mg._GEMM1X1 = False, True  # noqa: SLF001 - the translator's conv dispatch, as the export ran
        mg._HALF_WINO_F32 = mg._HALF_WINO_PAD = False  # noqa: SLF001
        size = (x.shape[-2], x.shape[-1])
        grid = (size[0] // un.PATCH, size[1] // un.PATCH)
        toks = self._encode(to_mx(x), *grid)
        feats = [mx.addmm(bias, t[:, 1:], w) for t, (w, bias) in zip(toks, self.adapters, strict=True)]
        if rays is None:
            fov, sh = self._angular([t[:, :1] for t in toks], size)
            r = camera_rays(fov, sh, *size)
        else:
            r = to_mx(rays.float())
        del toks
        outs = self._radial(feats, r, size, grid, confidence=confidence)
        return to_torch(outs[0]), to_torch(outs[1]) if confidence else None, to_torch(r)


def lin_1x1(c: nn.Conv2d, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """A 1x1 conv as `(weight [I, O], bias)`."""
    return _arr(_np(c.weight)[:, :, 0, 0].T, dtype), _arr(_np(c.bias), dtype)
