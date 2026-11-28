"""WAFT a1 on Apple MLX, written in `mlx.core` operations: `MlxWAFT(sd, dtype, iters)(image1, image2)`.

The maths of `net.ViTWarpV8.forward` (final flow only), channels last, built from the checkpoint's state dict:

- the images normalised and zero-padded to multiples of 112 (upstream's `Padder`, 'sintel' mode) in float32;
- Depth Anything V2 Small on each padded image: the DINOv2 blocks (`mlx_vit.Stepped`, LayerScale folded, the patch
  embedding one matmul, the position table bicubic per grid size on the host, as DINOv2's interpolation) and the DPT
  head's fusion path (`depth_anything_v2.mlx_sd`), `output_conv1` and its resize to the image: 32 feature channels;
- the ResNet18 feature encoder with its batch norms folded, each block one compiled step; the transposed 2 x 2
  convs of its decoder one matmul and a pixel shuffle each;
- the five warping iterations, each a few compiled steps, one in flight: the warp of image 2's features by the
  float32 flow (`graph_mlx.ops.grid_sample`), `warp_linear` with image 1's share computed once, the refinement ViT-S
  (patch 8; its 8 x 8 position table bicubic per grid size, timm blocks) and its DPT head, `refine_transform` and
  the flow head's two flow rows (float32 flow state);
- the convex 2x upsampling in float32, the mask head's 0.25 folded in, then the padding cropped.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps
from fastforks.shared.depth_anything_v2 import mlx_net as da2
from fastforks.shared.depth_anything_v2 import mlx_sd as sdh
from fastforks.shared.mlx_resize import bicubic_host
from fastforks.shared.mlx_vit import Stepped, depth_of, dinov2_table, patches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxWAFT"]

LAYOUT = "waft-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__), Path(da2.__file__), Path(sdh.__file__))

FACTOR, PAD = 2, 112
DA = "da_feature.depth_anything"
DA_PATCH, HEADS, EPS = 14, 6, 1e-6
LAYERS = (3, 6, 9, 12)  # 1-based: Depth Anything V2 Small's and the refinement ViT's (upstream idx 2, 5, 8, 11)
VIT_PATCH = 8
TABLES = 8  # position tables kept per encoder (one per grid size, least recently used dropped)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_FLOW = (0, 1)  # the flow head's flow rows (the rest is the uncertainty)


def _gelu(x: mx.array) -> mx.array:
    return mf.gelu(x.astype(mx.float32)).astype(x.dtype)


class _Up:
    """`ConvTranspose2d(kernel=stride=2)` of NHWC maps: one matmul to `[2, 2, out]` per pixel and a pixel shuffle."""

    def __init__(self, sd: Mapping[str, Any], name: str, dt: mx.Dtype) -> None:
        """Convert `name`'s `[in, out, 2, 2]` weight."""
        i, o, k, _ = (int(n) for n in sd[f"{name}.weight"].shape)
        self.k = k
        self.w = mf.put((i, k * k * o), dt, lambda: mf.np64(sd[f"{name}.weight"]).transpose(0, 2, 3, 1).reshape(i, -1))
        self.b = mf.put((k * k * o,), dt, lambda: np.tile(mf.np64(sd[f"{name}.bias"]), k * k))

    def __call__(self, x: mx.array) -> mx.array:
        """`[N, H, W, in]` -> `[N, 2 H, 2 W, out]`."""
        n, h, w, _ = x.shape
        k = self.k
        y = mx.addmm(self.b, x.reshape(-1, x.shape[-1]), self.w)
        return y.reshape(n, h, w, k, k, -1).transpose(0, 1, 3, 2, 4, 5).reshape(n, h * k, w * k, -1)


class _ResConv:
    """WAFT's `resconv`: conv(GELU(conv(GELU(x)))) + skip (a 1 x 1 conv, or the input)."""

    def __init__(self, sd: Mapping[str, Any], name: str, dt: mx.Dtype, stride: int = 1) -> None:
        """Convert `name`."""
        self.c1 = mf.conv(sd, f"{name}.conv.1", dt, stride=stride)
        self.c2 = mf.conv(sd, f"{name}.conv.3", dt)
        self.skip = mf.conv(sd, f"{name}.skip_conv", dt, stride=stride) if f"{name}.skip_conv.weight" in sd else None

    def __call__(self, x: mx.array) -> mx.array:
        """The block on `[N, H, W, C]`."""
        y = self.c2(_gelu(self.c1(_gelu(x))))
        return y + (self.skip(x) if self.skip is not None else x)


def _basic(c1: Conv, c2: Conv, ds: Conv | None) -> Callable[[mx.array], mx.array]:
    def run(x: mx.array) -> mx.array:
        y = c2(mf.relu(c1(x)))
        return mf.relu((ds(x) if ds is not None else x) + y)

    return run


class _Features:
    """`ResNet18Deconv`: `resconv` stem (stride 2), timm ResNet18 stages (batch norms folded), the deconv decoder."""

    def __init__(self, sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> None:
        """Convert the encoder `p` of `sd`."""
        stem = _ResConv(sd, f"{p}.ds1", dt, stride=2)
        self.stages: list[list[Callable[..., mx.array]]] = []
        for layer in (1, 2, 3, 4):
            stage = [stepped(stem)] if layer == 1 else []
            i = 0
            while f"{p}.conv{layer}.{i}.conv1.weight" in sd:
                b = f"{p}.conv{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1 = mf.conv(sd, f"{b}.conv1", dt, stride=stride, bn=f"{b}.bn1")
                c2 = mf.conv(sd, f"{b}.conv2", dt, bn=f"{b}.bn2")
                ds = (
                    mf.conv(sd, f"{b}.downsample.0", dt, stride=stride, padding=0, bn=f"{b}.downsample.1")
                    if f"{b}.downsample.0.weight" in sd
                    else None
                )
                stage.append(stepped(_basic(c1, c2, ds)))
                i += 1
            self.stages.append(stage)
        self.up = [_Up(sd, f"{p}.up_{i}", dt) for i in (4, 3, 2)]
        self.proj = [_ResConv(sd, f"{p}.proj_{i}", dt) for i in (3, 2, 1)]
        self._dec = [stepped(self._decode(up, proj)) for up, proj in zip(self.up, self.proj, strict=True)]

    @staticmethod
    def _decode(up: _Up, proj: _ResConv) -> Callable[[mx.array, mx.array], mx.array]:
        return lambda skip, x: proj(skip + up(x))

    def __call__(self, x: mx.array) -> mx.array:
        """`[1, H, W, 35]` -> the half-resolution features `[1, H / 2, W / 2, 64]` (`out_1`)."""
        outs = []
        for stage in self.stages:
            for f in stage:
                x = f(x)
            outs.append(x)
        x = outs[3]
        for skip, dec in zip(outs[2::-1], self._dec, strict=True):
            x = dec(skip, x)
        return x


class MlxWAFT:
    """WAFT a1 from the state dict `sd` in `dtype` (the name), `iters` warping iterations; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.da = sdh.vit(sd, f"{DA}.pretrained", dt, table=False)
        self.da_head = sdh.head(sd, f"{DA}.depth_head", dt, depth=False)
        self.fnet = _Features(sd, "fnet", dt)
        self.fmap_conv = mf.conv(sd, "fmap_conv", dt)
        self.hidden_conv = mf.conv(sd, "hidden_conv", dt)
        d = int(sd["hidden_conv.weight"].shape[0])  # the network dim (64)
        # warp_linear(cat[fmap1, warp, net, flow]): image 1's share (with the bias) once, the rest per iteration
        self.warp_f1 = mf.conv(sd, "warp_linear", dt, inputs=lambda w: w[:, :d], cin=d)
        cin = int(sd["warp_linear.weight"].shape[1])
        self.warp_rest = mf.conv(sd, "warp_linear", dt, inputs=lambda w: w[:, d:], cin=cin - d, bias=False)
        r = "refine_net"
        self.patch = mf.conv(sd, f"{r}.patch_embed.proj", dt, stride=VIT_PATCH, padding=0)
        n, c = (int(k) for k in sd[f"{r}.pos_embed"].shape[1:])
        self.vit_pos = mf.put((n, c), mx.float32, lambda: mf.np64(sd[f"{r}.pos_embed"])[0])
        vit = sdh.blocks(sd, f"{r}.blks", dt)
        self.vit = Stepped(vit, depth_of(vit), HEADS, EPS)
        self.vit_head = sdh.head(sd, f"{r}.dpt_head", dt, depth=False)
        self.refine_transform = mf.conv(sd, "refine_transform", dt)
        self.head1 = mf.conv(sd, "flow_head.0", dt)
        self.head2 = mf.conv(sd, "flow_head.2", dt, rows=_FLOW)
        self.up1 = mf.conv(sd, "upsample_weight.0", dt)
        self.up2 = mf.conv(sd, "upsample_weight.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self.da_blocks = Stepped(self.da, depth_of(self.da), HEADS, EPS)
        self._tables: OrderedDict[tuple[str, int, int], mx.array] = OrderedDict()
        self._embed = stepped(self._embed_fn)
        self._depth = stepped(self._depth_fn)
        self._fmap = stepped(self._fmap_fn)
        self._hidden = stepped(self._hidden_fn)
        self._warp = stepped(self._warp_fn)
        self._refine = stepped(self._refine_fn)
        self._update = stepped(self._update_fn)
        self._mask = stepped(lambda net: self.up2(mf.relu(self.up1(net))))

    # position tables (float32 on the host, as torch interpolates them; cast once) ---------------------------------

    def _table(self, kind: str, h0: int, w0: int) -> mx.array:
        key = (kind, h0, w0)
        if (t := self._tables.get(key)) is None:
            if kind == "da":
                p = self.da
                t = dinov2_table(np.array(p["pos.cls"]), np.array(p["pos.grid"]), np.array(p["pos.bias"]), h0, w0)
            else:  # the refinement ViT: bicubic by the scale factors (h0 / 8, w0 / 8), no offset
                g = np.array(self.vit_pos)
                side = math.isqrt(g.shape[0])
                g = np.ascontiguousarray(g.reshape(1, side, side, -1).transpose(0, 3, 1, 2))
                g = bicubic_host(g, (h0, w0), (h0 / side, w0 / side))
                t = g[0].transpose(1, 2, 0).reshape(h0 * w0, -1)
            t = mx.array(t).astype(self.dtype)
            self._tables[key] = t
            while len(self._tables) > 2 * TABLES:
                self._tables.popitem(last=False)
        else:
            self._tables.move_to_end(key)
        return t

    # image features ------------------------------------------------------------------------------------------------

    def _embed_fn(self, img: mx.array, table: mx.array) -> mx.array:
        """Padded, normalised `[1, H, W, 3]` -> DINOv2 tokens `[1, 1 + N, C]` (cls first)."""
        x = img.transpose(0, 3, 1, 2).astype(table.dtype)
        t = patches(x, DA_PATCH) @ self.da["patch.weight"] + table[1:]
        return mx.concatenate([table[None, :1], t], axis=1)

    def _depth_fn(self, img: mx.array, *hs: mx.array) -> list[mx.array]:
        """The DPT head's 32 features at the image size and the encoder input `cat[features, image]`."""
        p, (_, h, w, _) = self.da, img.shape
        h0, w0 = h // DA_PATCH, w // DA_PATCH
        feats = [mx.fast.layer_norm(t[:, 1:], p["norm.weight"], p["norm.bias"], EPS) for t in hs]
        path = sdh.fusion(self.da_head, feats, h0, w0)
        out = sdh.bilinear(da2._conv(self.da_head, "head.out1", path), h, w, align=True)  # noqa: SLF001
        return [out, mx.concatenate([out, img.astype(out.dtype)], axis=-1)]

    def _fmap_fn(self, f: mx.array, depth: mx.array) -> mx.array:
        d2 = sdh.bilinear(depth, f.shape[1], f.shape[2], align=True)
        return self.fmap_conv(mx.concatenate([f, d2], axis=-1))

    def _image(self, img: mx.array) -> mx.array:
        """One padded, normalised float32 image `[1, H, W, 3]` -> `fmap_conv`'s `[1, H / 2, W / 2, 64]`."""
        _, h, w, _ = img.shape
        t = self._embed(img, self._table("da", h // DA_PATCH, w // DA_PATCH))
        _, hs = self.da_blocks(t, layers=LAYERS)
        depth, x = self._depth(img, *hs)
        return self._fmap(self.fnet(x), depth)

    # the warping iterations ----------------------------------------------------------------------------------------

    def _hidden_fn(self, f1: mx.array, f2: mx.array) -> list[mx.array]:
        return [self.hidden_conv(mx.concatenate([f1, f2], axis=-1)), self.warp_f1(f1)]

    def _warp_fn(self, f1w: mx.array, f2: mx.array, net: mx.array, flow: mx.array, table: mx.array) -> mx.array:
        """The refinement ViT's embedded tokens `[1, N, C]` of `warp_linear(cat[fmap1, warp, net, flow])`."""
        _, h, w, _ = flow.shape
        coords = mf.grid(h, w) + flow
        scale = mx.array([2 / (w - 1), 2 / (h - 1)], dtype=mx.float32)
        warp = ops.grid_sample(f2, coords * scale - 1, align_corners=True, border=False)
        x = mx.concatenate([warp, net, flow.astype(net.dtype)], axis=-1)
        x = f1w + self.warp_rest(x)
        t = self.patch(x)
        return t.reshape(1, -1, t.shape[-1]) + table

    def _refine_fn(self, net: mx.array, *hs: mx.array) -> mx.array:
        """The refinement ViT's DPT features at the half resolution, then `refine_transform(cat[out, net])`."""
        _, h, w, _ = net.shape
        h0, w0 = h // VIT_PATCH, w // VIT_PATCH
        path = sdh.fusion(self.vit_head, list(hs), h0, w0)
        out = da2._conv(self.vit_head, "head.out1", path)  # noqa: SLF001
        out = sdh.bilinear(out, h0 * DA_PATCH, w0 * DA_PATCH, align=True)
        out = sdh.bilinear(out, h, w, align=True)
        return self.refine_transform(mx.concatenate([out, net], axis=-1))

    def _update_fn(self, net: mx.array, flow: mx.array) -> mx.array:
        return flow + self.head2(mf.relu(self.head1(net))).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        _, _, h, w = image1.shape
        ph = ((h + 8) // PAD + 1) * PAD - h
        pw = ((w + 8) // PAD + 1) * PAD - w
        top, left = ph // 2, pw // 2
        pads = ((0, 0), (top, ph - top), (left, pw - left), (0, 0))
        mean = mx.array(MEAN, dtype=mx.float32)
        std = mx.array(STD, dtype=mx.float32)

        def prep(x: mx.array) -> mx.array:
            return mx.pad((x.transpose(0, 2, 3, 1) / 255.0 - mean) / std, pads)

        f1 = self._image(prep(image1))
        f2 = self._image(prep(image2))
        net, f1w = self._hidden(f1, f2)
        _, h2, w2, _ = f1.shape
        flow = mx.zeros((1, h2, w2, 2), dtype=mx.float32)
        table = self._table("vit", h2 // VIT_PATCH, w2 // VIT_PATCH)
        for _ in range(self.iters):
            t = self._warp(f1w, f2, net, flow, table)
            _, hs = self.vit(t, layers=LAYERS)
            net = self._refine(net, *hs)
            flow = self._update(net, flow)
        up = mf.convex_upsample(flow, self._mask(net), FACTOR)
        up = up[:, :, top : top + h, left : left + w]
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of an unpadded float32 pair `[1, 3, H, W]` in [0, 255]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
