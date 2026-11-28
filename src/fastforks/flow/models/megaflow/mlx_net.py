"""MegaFlow on Apple MLX, written in `mlx.core` operations: `MlxMegaFlow(sd, dtype, iters)(image1, image2)`.

The maths of `model._Fast.fast` (upstream `MegaFlow.forward` on a pair), channels last, built from the state dict:

- the ResNet CNN (batch norms folded) on each full-size padded frame, each block one compiled step: its 1/2 and 1/4
  features enter the head as their `concat_project` shares (`pixel_unshuffle` + 1 x 1 conv = a patch conv, before
  the bilinear resize, which commutes with it);
- the frames resized to width 952 (bilinear, align corners); DINOv2 ViT-L/14 with 4 registers (`mlx_vit.Stepped`,
  LayerScale folded, the patch embedding one matmul, the position table resized on the host by PyTorch's
  antialiased bicubic matrices, as upstream's `interpolate_pos_encoding`), its final norm on the patch tokens;
- the 24 frame / global block pairs (`mlx_vggt.Aggregator`, 2D RoPE folded into permuted q / k channels);
- the DPT head on both frames: per layer LayerNorm, projection, the sine position embedding (0.1), the transposed
  2 x 2 convs as one matmul and a pixel shuffle, the fusion blocks (ReLU'd residuals, as upstream's in-place ReLU);
- the global softmax correlation as one float32 attention call (values: the centred pixel grid, padded to 128);
- `iters` updates at 1/4 resolution, each a compiled step: the 9 x 9 correlation windows (`graph_mlx.ops`, the
  `convc1` input channels transposed to the kernel's x-major order), the motion encoder, the two ConvNeXt blocks
  with the temporal attention of a length-1 sequence (`proj(v)`, folded into one linear), the float32 flow update;
- the convex 4x upsampling in float32, the mask head's 0.25 folded in.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared import mlx_vggt
from fastforks.shared.cspnext.mlx_net import stepped, steps
from fastforks.shared.depth_anything_v2 import mlx_sd as sdh
from fastforks.shared.mlx_vit import Stepped, depth_of, patches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastforks.shared.mlx_flow import Conv

__all__ = ["LAYOUT", "SOURCES", "MlxMegaFlow"]

LAYOUT = "megaflow-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__), Path(sdh.__file__), Path(mlx_vggt.__file__))

PATCH, WIDTH = 14, 952
LAYERS = (4, 11, 17, 23)  # 0-based aggregator blocks the head reads
REGISTERS = 4  # DINOv2's register tokens; the aggregator's 4 register tokens per frame (its patch start)
HEADS, DINO_EPS = 16, 1e-6
RADIUS, UP = 4, 4
TABLES = 8  # position tables kept (one per grid size, least recently used dropped)


# host tables ---------------------------------------------------------------------------------------------------


def _cubic(t: np.ndarray) -> np.ndarray:
    """PyTorch's bicubic filter (a = -0.5)."""
    t = np.abs(t)
    near = ((1.5 * t - 2.5) * t) * t + 1.0
    far = (((-0.5 * t + 2.5) * t) - 4.0) * t + 2.0
    return np.where(t < 1.0, near, np.where(t < 2.0, far, 0.0))  # noqa: PLR2004 - the filter's support


def _aa_bicubic(inp: int, out: int) -> np.ndarray:
    """`[out, inp]` weights of `F.interpolate(size=out, mode="bicubic", antialias=True)` along one axis."""
    s = inp / out
    support = max(s, 1.0) * 2.0
    inv = 1.0 / s if s >= 1.0 else 1.0
    m = np.zeros((out, inp))
    for i in range(out):
        c = s * (i + 0.5)
        lo, hi = max(int(c - support + 0.5), 0), min(int(c + support + 0.5), inp)
        wt = _cubic((np.arange(lo, hi) - c + 0.5) * inv)
        m[i, lo:hi] = wt / wt.sum()
    return m


def _sincos(d: int, pos: np.ndarray) -> np.ndarray:
    """`make_sincos_pos_embed`: `[M, d]` = `[sin(pos w), cos(pos w)]`, `w = 1 / 100^(k / (d / 2))`."""
    omega = 1.0 / 100.0 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
    out = pos.reshape(-1, 1) * omega
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _uv_embed(d: int, h: int, w: int, aspect: float) -> np.ndarray:
    """0.1 x upstream's `position_grid_to_embed(create_uv_grid(w, h, aspect), d)`: `[h, w, d]`."""
    diag = math.sqrt(aspect**2 + 1.0)
    sx, sy = aspect / diag, 1.0 / diag
    xs = np.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w)
    ys = np.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h)
    uu, vv = np.meshgrid(xs, ys, indexing="xy")  # [h, w]
    emb = np.concatenate([_sincos(d // 2, uu), _sincos(d // 2, vv)], axis=1)
    return 0.1 * emb.reshape(h, w, d)


# network pieces ------------------------------------------------------------------------------------------------


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


def _unshuffle_conv(sd: Mapping[str, Any], name: str, dt: mx.Dtype, k: int, cin: int) -> Conv:
    """`concat_project`'s share of `pixel_unshuffle(x, k)` (`cin` channels of x) as a `k` x `k` patch conv, no bias."""
    o = int(sd[f"{name}.weight"].shape[0])

    def wt() -> np.ndarray:
        w = mf.np64(sd[f"{name}.weight"])[:, : cin * k * k, 0, 0].reshape(o, cin, k, k)
        return w.transpose(0, 2, 3, 1).reshape(o, -1)

    return mf.Conv(mf.put((o, k * k * cin), dt, wt), mx.zeros((o,), dt), k, 0, patch=k)


def _rcu(c1: Conv, c2: Conv, x: mx.array) -> mx.array:
    """Upstream's `ResidualConvUnit` with its in-place ReLU: the residual is `relu(x)`."""
    x = mf.relu(x)
    return c2(mf.relu(c1(x))) + x


def _basic(c1: Conv, c2: Conv, ds: Conv | None) -> Callable[[mx.array], mx.array]:
    """MegaFlow's `BasicBlock`: `relu(ds(x) + relu(bn2(conv2(relu(bn1(conv1(x)))))))`."""

    def run(x: mx.array) -> mx.array:
        y = mf.relu(c2(mf.relu(c1(x))))
        return mf.relu((ds(x) if ds is not None else x) + y)

    return run


class _CNN:
    """`backbone` (`ResNetFPN`, `return_all_feat`): the stem's 1/2 features and `final_conv`'s 1/4 features."""

    def __init__(self, sd: Mapping[str, Any], dt: mx.Dtype) -> None:
        """Convert `backbone.*`."""
        p = "backbone"
        stem = mf.conv(sd, f"{p}.conv1", dt, stride=2, bn=f"{p}.bn1")
        self.stem = stepped(lambda x: mf.relu(stem(x)))
        self.blocks = []
        for layer in (1, 2):
            i = 0
            while f"{p}.layer{layer}.{i}.conv1.weight" in sd:
                b = f"{p}.layer{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1 = mf.conv(sd, f"{b}.conv1", dt, stride=stride, bn=f"{b}.bn1")
                c2 = mf.conv(sd, f"{b}.conv2", dt, bn=f"{b}.bn2")
                ds = (
                    mf.conv(sd, f"{b}.downsample.0", dt, stride=stride, padding=0, bn=f"{b}.downsample.1")
                    if f"{b}.downsample.0.weight" in sd
                    else None
                )
                self.blocks.append(_basic(c1, c2, ds))
                i += 1
        final, last = mf.conv(sd, f"{p}.final_conv", dt), self.blocks[-1]
        self.blocks[-1] = lambda x: final(last(x))
        self.blocks = [stepped(f) for f in self.blocks]

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """`[1, H, W, 3]` -> (`[1, H / 2, W / 2, 64]`, `[1, H / 4, W / 4, 128]`)."""
        x = half = self.stem(x)
        for f in self.blocks:
            x = f(x)
        return half, x


class _Head:
    """`flow_head` (`FlowFeature`, `fuse_cnn`): four aggregator layers -> 128-channel features at twice the grid."""

    def __init__(self, sd: Mapping[str, Any], dt: mx.Dtype) -> None:
        """Convert `flow_head.*`."""
        p = "flow_head"
        c = int(sd[f"{p}.norm.weight"].shape[0])
        self.nw = mf.put((c,), dt, lambda: mf.np64(sd[f"{p}.norm.weight"]))
        self.nb = mf.put((c,), dt, lambda: mf.np64(sd[f"{p}.norm.bias"]))
        self.proj = [mf.conv(sd, f"{p}.projects.{i}", dt) for i in range(4)]
        self.up = [_Up(sd, f"{p}.resize_layers.{i}", dt) for i in (0, 1)]
        self.down = mf.conv(sd, f"{p}.resize_layers.3", dt, stride=2, padding=1)
        cp = [f"{p}.concat_project.{i}" for i in (0, 1)]
        self.cnn = [_unshuffle_conv(sd, cp[0], dt, 4, 64), _unshuffle_conv(sd, cp[1], dt, 2, 128)]
        self.cat = [
            mf.conv(sd, cp[0], dt, inputs=lambda w: w[:, 64 * 16 :], cin=self.proj[0].out),
            mf.conv(sd, cp[1], dt, inputs=lambda w: w[:, 128 * 4 :], cin=self.proj[1].out),
        ]
        s = f"{p}.scratch"
        self.rn = [mf.conv(sd, f"{s}.layer{i + 1}_rn", dt, bias=False) for i in range(4)]
        self.units: list[list[tuple[Conv, Conv] | None]] = []
        self.outs = []
        for i in range(4):
            r = f"{s}.refinenet{i + 1}"
            self.units.append(
                [
                    (mf.conv(sd, f"{r}.resConfUnit{u}.conv1", dt), mf.conv(sd, f"{r}.resConfUnit{u}.conv2", dt))
                    if f"{r}.resConfUnit{u}.conv1.weight" in sd
                    else None
                    for u in (1, 2)
                ]
            )
            self.outs.append(mf.conv(sd, f"{r}.out_conv", dt))
        self.out1 = mf.conv(sd, f"{s}.output_conv1", dt)
        self.dims = [c.out for c in self.proj]

    def layer(self, i: int, t: mx.array, pos: mx.array, h0: int, w0: int) -> mx.array:
        """Layer `i`'s patch tokens `[S, h0 w0, 2 C]` -> its resized map (before `layer<i>_rn`)."""
        s = t.shape[0]
        x = mx.fast.layer_norm(t, self.nw, self.nb, 1e-5).reshape(s, h0, w0, -1)
        x = self.proj[i](x) + pos
        if i < len(self.up):
            return self.up[i](x)
        return self.down(x) if i == len(self.proj) - 1 else x

    def fuse(self, i: int, x: mx.array | None, skip: mx.array | None, size: tuple[int, int]) -> mx.array:
        """`refinenet<i + 1>(x, skip, size=size)` (`x` None: the coarsest block, `skip` its only input)."""
        u1, u2 = self.units[i]
        out = skip if x is None else x if u1 is None or skip is None else x + _rcu(*u1, skip)
        assert u2 is not None  # noqa: S101 - every block has its second unit
        assert out is not None  # noqa: S101 - a block has an input
        out = _rcu(*u2, out)
        if (out.shape[1], out.shape[2]) != size:
            out = sdh.bilinear(out, *size, align=True)
        return self.outs[i](out)


class MlxMegaFlow:
    """MegaFlow from the state dict `sd` in `dtype` (the name), `iters` refinement updates; one pair per call."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.cnn = _CNN(sd, dt)
        self.dino = sdh.vit(sd, "patch_embed", dt, table=False)
        c = int(sd["patch_embed.cls_token"].shape[-1])
        self.dino_reg = mf.put((REGISTERS, c), mx.float32, lambda: mf.np64(sd["patch_embed.register_tokens"])[0])
        self.dino_blocks = Stepped(self.dino, depth_of(self.dino), HEADS, DINO_EPS)
        self.reg = mf.put((2, REGISTERS, c), dt, lambda: mf.np64(sd["register_token"])[0])
        depth = max(LAYERS) + 1
        self.agg = mlx_vggt.Aggregator(sd, dt, heads=HEADS, depth=depth)
        self.head = _Head(sd, dt)
        self.refine_proj = mf.conv(sd, "refine_proj", dt)
        e = "refine.encoder"
        side = 2 * RADIUS + 1
        self.convc1 = mf.conv(  # the windows are x-offset major (upstream: y major)
            sd, f"{e}.convc1", dt, inputs=lambda w: w.reshape(-1, side, side, 1, 1).transpose(0, 2, 1, 3, 4).reshape(
                w.shape)
        )  # fmt: skip
        self.convc2 = mf.conv(sd, f"{e}.convc2", dt)
        self.convf1 = mf.conv(sd, f"{e}.convf1", dt)
        self.convf2 = mf.conv(sd, f"{e}.convf2", dt)
        self.conv = mf.conv(sd, f"{e}.conv", dt)
        self.convnext = mf.blocks(sd, "refine.refine", dt)
        self.temporal = [self._temporal(sd, f"refine.temporal_attn_refine.{i}", dt) for i in range(len(self.convnext))]
        self.fh1 = mf.conv(sd, "refine.flow_head.conv1", dt)
        self.fh2 = mf.conv(sd, "refine.flow_head.conv2", dt)
        self.mask1 = mf.conv(sd, "refine.mask.0", dt)
        self.mask2 = mf.conv(sd, "refine.mask.2", dt, rows=mf.convex_rows(UP), scale=0.25)
        self._tables: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._embed = stepped(self._embed_fn)
        self._tokens = stepped(self._tokens_fn)
        self._layers = [stepped(lambda t, pos, i=i: self._layer_fn(i, t, pos)) for i in range(4)]
        self._fuse = stepped(self._fuse_fn)
        self._global = stepped(self._global_fn)
        self._update = stepped(self._update_fn)
        self._mask = stepped(lambda net: self.mask2(mf.relu(self.mask1(net))))

    @staticmethod
    def _temporal(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> dict[str, mx.array]:
        """The temporal `Block` on a length-1 sequence: `t + proj(v(LN1 t))`, then the MLP; `proj . v` one linear."""
        c = int(sd[f"{p}.norm1.weight"].shape[0])
        hidden = int(sd[f"{p}.mlp.fc1.weight"].shape[0])

        def take(k: str) -> np.ndarray:
            return mf.np64(sd[f"{p}.{k}"])

        def wv() -> np.ndarray:
            return (take("attn.proj.weight") @ take("attn.qkv.weight")[2 * c :]).T

        def bv() -> np.ndarray:
            return take("attn.proj.weight") @ take("attn.qkv.bias")[2 * c :] + take("attn.proj.bias")

        out = {"wv": mf.put((c, c), dt, wv), "bv": mf.put((c,), dt, bv)}
        for name in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                out[f"{name}.{kind}"] = mf.put((c,), dt, lambda k=f"{name}.{kind}": take(k))
        out["fc1.weight"] = mf.put((c, hidden), dt, lambda: take("mlp.fc1.weight").T)
        out["fc1.bias"] = mf.put((hidden,), dt, lambda: take("mlp.fc1.bias"))
        out["fc2.weight"] = mf.put((hidden, c), dt, lambda: take("mlp.fc2.weight").T)
        out["fc2.bias"] = mf.put((c,), dt, lambda: take("mlp.fc2.bias"))
        return out

    # host tables (cached per size, least recently used dropped) -------------------------------------------------

    def _cached(self, key: tuple[Any, ...], make: Callable[[], Any]) -> Any:  # noqa: ANN401 - arrays or tuples of them
        if (t := self._tables.get(key)) is None:
            t = self._tables[key] = make()
            while len(self._tables) > TABLES:
                self._tables.popitem(last=False)
        else:
            self._tables.move_to_end(key)
        return t

    def _dino_table(self, h0: int, w0: int) -> mx.array:
        """`[1 + registers + h0 w0, C]` in the run dtype: cls + its position, the registers, patch bias + position."""

        def make() -> mx.array:
            p = self.dino
            g = np.array(p["pos.grid"], dtype=np.float64)
            side = math.isqrt(g.shape[0])
            g = g.reshape(side, side, -1)
            g = np.einsum("ia,abc->ibc", _aa_bicubic(side, h0), g)
            g = np.einsum("jb,ibc->ijc", _aa_bicubic(side, w0), g).reshape(h0 * w0, -1)
            g = g.astype(np.float32) + np.array(p["pos.bias"])
            rows = [np.array(p["pos.cls"]), np.array(self.dino_reg), g]
            return mx.array(np.concatenate(rows).astype(np.float32)).astype(self.dtype)

        return self._cached(("dino", h0, w0), make)

    def _pos(self, d: int, h: int, w: int, aspect: float) -> mx.array:
        return self._cached(("pos", d, h, w, aspect), lambda: mx.array(_uv_embed(d, h, w, aspect).astype(np.float32)))

    def _rope(self, h0: int, w0: int) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        def make() -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
            frame = mlx_vggt.rope(h0, w0, REGISTERS, 1)
            return frame, (mx.tile(frame[0], (2, 1)), mx.tile(frame[1], (2, 1)))

        return self._cached(("rope", h0, w0), make)

    # steps -----------------------------------------------------------------------------------------------------

    def _embed_fn(self, x: mx.array, table: mx.array, rh: int) -> mx.array:
        """Normalised frames `[2, H, W, 3]` -> DINOv2 tokens `[2, 1 + registers + N, C]` at `(rh, WIDTH)`."""
        xr = sdh.bilinear(x, rh, WIDTH, align=True).transpose(0, 3, 1, 2).astype(table.dtype)
        t = patches(xr, PATCH) @ self.dino["patch.weight"] + table[1 + REGISTERS :]
        lead = mx.broadcast_to(table[None, : 1 + REGISTERS], (t.shape[0], 1 + REGISTERS, t.shape[-1]))
        return mx.concatenate([lead, t], axis=1)

    def _tokens_fn(self, t: mx.array) -> mx.array:
        """The final norm on DINOv2's patch tokens, the aggregator's registers before them: `[2, 4 + N, C]`."""
        p = self.dino
        pt = mx.fast.layer_norm(t[:, 1 + REGISTERS :], p["norm.weight"], p["norm.bias"], DINO_EPS)
        return mx.concatenate([self.reg.astype(pt.dtype), pt], axis=1)

    def _layer_fn(self, i: int, t: mx.array, pos: mx.array) -> mx.array:
        _, h, w, _ = pos.shape
        return self.head.layer(i, t[:, REGISTERS:], pos.astype(t.dtype)[0], h, w)

    def _fuse_fn(self, maps: list[mx.array], c1: mx.array, c2: mx.array, pos: mx.array) -> mx.array:
        """The fusion path -> the head's features `[2, 2 h0, 2 w0, 128]` (cnn shares `c1`, `c2` at 1/8)."""
        hd = self.head
        l1, l2, l3, l4 = maps
        size1, size2 = (l1.shape[1], l1.shape[2]), (l2.shape[1], l2.shape[2])
        l1 = hd.cat[0](l1) + sdh.bilinear(c1, *size1, align=True)
        l2 = hd.cat[1](l2) + sdh.bilinear(c2, *size2, align=True)
        rn = [hd.rn[i](x) for i, x in enumerate((l1, l2, l3, l4))]
        size = [(m.shape[1], m.shape[2]) for m in rn]
        x = hd.fuse(3, None, rn[3], size[2])
        x = hd.fuse(2, x, rn[2], size[1])
        x = hd.fuse(1, x, rn[1], size[0])
        x = hd.fuse(0, x, rn[0], size[0])
        return hd.out1(x) + pos.astype(x.dtype)

    def _global_fn(self, feats: mx.array, h4: int, w4: int) -> list[mx.array]:
        """`global_correlation_softmax` at the head's grid, resized to 1/4 (flow float32) and `refine_proj`."""
        _, hf, wf, c = feats.shape
        q = feats[:1].reshape(1, 1, hf * wf, c).astype(mx.float32)
        k = feats[1:].reshape(1, 1, hf * wf, c).astype(mx.float32)
        g = (mf.grid(hf, wf) - mx.array([(wf - 1) / 2, (hf - 1) / 2], mx.float32)).reshape(1, 1, hf * wf, 2)
        v = mx.pad(g, ((0, 0), (0, 0), (0, 0), (0, c - 2)))
        o = attention(q, k, v, scale=c**-0.5)[..., :2]
        flow = (o - g).reshape(1, hf, wf, 2)
        flow = sdh.bilinear(flow, h4, w4, align=True) * mx.array([w4 / wf, h4 / hf], mx.float32)
        ref = sdh.bilinear(self.refine_proj(feats[:1]), h4, w4, align=True)
        d = ref.shape[-1] // 2
        return [flow, ref[..., :d], ref[..., d:]]

    def _block(self, i: int, net: mx.array) -> mx.array:
        p = self.temporal[i]
        _, h, w, c = net.shape
        t = net.reshape(h * w, c)
        t = t + mx.addmm(p["bv"], mx.fast.layer_norm(t, p["norm1.weight"], p["norm1.bias"], 1e-5), p["wv"])
        y = mx.fast.layer_norm(t, p["norm2.weight"], p["norm2.bias"], 1e-5)
        y = mf.gelu(mx.addmm(p["fc1.bias"], y, p["fc1.weight"]))
        return (t + mx.addmm(p["fc2.bias"], y, p["fc2.weight"])).reshape(net.shape)

    def _update_fn(
        self, net: mx.array, inp: mx.array, flow: mx.array, f0: mx.array, f1: mx.array
    ) -> list[mx.array]:  # fmt: skip
        """One `RAFTUpdateBlock` step on the float32 flow `[1, h, w, 2]` -> (hidden state, flow)."""
        _, h, w, _ = flow.shape
        dt = net.dtype
        coords = (mf.grid(h, w) + flow).transpose(0, 3, 1, 2)
        corr = ops.corr_windows(f0, [f1], coords, RADIUS, f0.shape[-1] ** -0.5).transpose(0, 2, 3, 1).astype(dt)
        fl = flow.astype(dt)
        cor = mf.relu(self.convc2(mf.relu(self.convc1(corr))))
        flo = mf.relu(self.convf2(mf.relu(self.convf1(fl))))
        motion = mx.concatenate([mf.relu(self.conv(mx.concatenate([cor, flo], axis=-1))), fl], axis=-1)
        for i, blk in enumerate(self.convnext):
            net = self._block(i, blk(mx.concatenate([net, inp, motion], axis=-1)))
        return [net, flow + self.fh2(mf.relu(self.fh1(net))).astype(mx.float32)]

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        _, _, h, w = image1.shape
        frames = [x.transpose(0, 2, 3, 1) for x in (image1, image2)]
        cnn = [self.cnn(x.astype(self.dtype)) for x in frames]
        hd = self.head
        c1 = mx.concatenate([hd.cnn[0](half) for half, _ in cnn])  # [2, H / 8, W / 8, 256]
        c2 = mx.concatenate([hd.cnn[1](quarter) for _, quarter in cnn])
        f0, f1 = cnn[0][1], cnn[1][1]
        del cnn
        rh = round(h * WIDTH / w / PATCH) * PATCH
        h0, w0 = rh // PATCH, WIDTH // PATCH
        aspect = WIDTH / rh
        t = self._embed(mx.concatenate(frames), self._dino_table(h0, w0), rh)
        t, _ = self.dino_blocks(t)
        t = self._tokens(t)
        frame_rope, global_rope = self._rope(h0, w0)
        hs = self.agg(t, frame_rope, global_rope, LAYERS)
        maps = [
            self._layers[i](x, self._pos(d, h0, w0, aspect)[None])
            for i, (x, d) in enumerate(zip(hs, hd.dims, strict=True))
        ]
        del hs
        feats = self._fuse(maps, c1, c2, self._pos(128, 2 * h0, 2 * w0, aspect)[None])
        del maps
        h4, w4 = h // UP, w // UP
        flow, net, inp = self._global(feats, h4, w4)
        del feats
        for _ in range(self.iters):
            net, flow = self._update(net, inp, flow, f0, f1)
        up = mf.convex_upsample(flow, self._mask(net), UP)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded, normalised float32 pair `[1, 3, H, W]` (H, W multiples of 8)."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
