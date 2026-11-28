"""FlowFormer Small on Apple MLX, written in `mlx.core` operations: `MlxFlowFormer(sd, dtype, iters)(image1, image2)`.

The maths of upstream's `FlowFormer.forward` (the Small configuration), channels last, from the checkpoint's state dict
(`shared.mlx_flow`, `shared.mlx_twins`):

- the two instance-norm `BasicEncoder`s (float32 statistics; the features of both frames one batch), each residual
  block one compiled step; their last 1 x 1 conv folded with the 1 x 1 after it (`memory_decoder.proj`, the
  `channel_convertor`);
- no cost volume: the cost encoder makes the all-pairs maps `EMBED_MAPS` query pixels at a time, patch-embeds them
  (the position half of `ffn_with_coord.0` one bias per grid) and keeps only their latent tokens: the input layer's
  latent queries are constant, so its keys are a `[128, heads x tokens]` GEMM of the patch tokens (their bias drops out
  of the softmax) and its values the softmax-weighted tokens through `v`;
- the 4-token self-attention, then the Twins local and global blocks over the latents' maps (`mlx_twins`);
- the context and GMA's matrix after the cost memory, each update four compiled steps (decoder, motion, GRU halves): the
  decoder's 81 costs from `graph_mlx.ops.corr_windows` (unscaled dot products), its cross-attention to each pixel's 4
  cost tokens as broadcast products, the GMA readout's softmax matrix computed once while its float32 size is at most
  `MATRIX_BYTES` (else fused attention in each update), its `gamma` folded into `to_v`, the SepConvGRU's z and r convs
  stacked; the flow state, the flow encoder's first conv, the windows and the convex upsampling in float32; the mask
  head's 0.25 folded in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.mlx import attention, mx
from fastforks.kernels.graph_mlx import ops
from fastforks.shared import mlx_flow as mf
from fastforks.shared import mlx_graph as mg
from fastforks.shared import mlx_twins as tw
from fastforks.shared.cspnext.mlx_net import stepped, steps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["LAYOUT", "SOURCES", "MlxFlowFormer"]

LAYOUT = "flowformer-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__), Path(tw.__file__))

FACTOR, RADIUS, PATCH, HIDDEN = 8, 4, 8, 128
EMBED_MAPS = 512  # cost maps patch-embedded per step
MATRIX_BYTES = 1 << 30  # float32 GMA matrix from which fused attention recomputes it in each update


def _instance(x: mx.array) -> mx.array:
    """`InstanceNorm2d` (no affine, eps 1e-5) of `[N, H, W, C]`, float32 statistics."""
    y = x.astype(mx.float32)
    mean = y.mean(axis=(1, 2), keepdims=True)
    var = mx.square(y - mean).mean(axis=(1, 2), keepdims=True)
    return ((y - mean) * mx.rsqrt(var + 1e-5)).astype(x.dtype)


def _layer_norm_np(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5) * w + b


class _Encoder:
    """Instance-norm `BasicEncoder`: 7 x 7 stride-2 stem, residual stages of 64 / 96 / 128 channels (to H / 8).

    Each conv is its own step, evaluated at once (`mlx_flow.eager`): pipelined blocks held two blocks' float32
    Winograd transforms and instance-norm statistics (the pair at 1/2 resolution, 544 x 960 float16: +0.5 GB).
    """

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        def conv(name: str, **kw: Any) -> Callable[[mx.array], mx.array]:  # noqa: ANN401 - conv arguments
            c = mf.conv(sd, name, dtype, **kw)
            return lambda x: _instance(c(x))

        stem = conv(f"{p}.conv1", stride=2)
        self._stem = mf.eager(lambda x: mf.relu(stem(x)))
        self._blocks: list[tuple[Callable, Callable]] = []
        for layer in (1, 2, 3):
            for i in (0, 1):
                b = f"{p}.layer{layer}.{i}"
                stride = 2 if i == 0 and layer > 1 else 1
                c1, c2 = conv(f"{b}.conv1", stride=stride), conv(f"{b}.conv2")
                ds = conv(f"{b}.downsample.0", stride=stride, padding=0) if stride > 1 else None
                self._blocks.append((mf.eager(lambda x, c1=c1: mf.relu(c1(x))), mf.eager(self._second(c2, ds))))

    @staticmethod
    def _second(c2: Callable, ds: Callable | None) -> Callable[[mx.array, mx.array], mx.array]:
        def run(x: mx.array, y: mx.array) -> mx.array:
            return mf.relu((ds(x) if ds is not None else x) + mf.relu(c2(y)))

        return run

    def __call__(self, x: mx.array) -> mx.array:
        """`[N, H, W, 3]` -> `[N, H / 8, W / 8, 128]`."""
        x = self._stem(x)
        for first, second in self._blocks:
            x = second(x, first(x))
        return x


def _folded(sd: Mapping[str, Any], conv2: str, after: str, dtype: mx.Dtype, *, bias: bool) -> mf.Conv:
    """The 1 x 1 `conv2` followed by the 1 x 1 `after` (with its bias, `bias`) as one 1 x 1 conv."""
    o, i = int(sd[f"{after}.weight"].shape[0]), int(sd[f"{conv2}.weight"].shape[1])

    def a() -> np.ndarray:
        return mf.np64(sd[f"{after}.weight"])[:, :, 0, 0]

    def b() -> np.ndarray:
        out = a() @ mf.np64(sd[f"{conv2}.bias"])
        return out + mf.np64(sd[f"{after}.bias"]) if bias else out

    w = mf.put((o, i), dtype, lambda: a() @ mf.np64(sd[f"{conv2}.weight"])[:, :, 0, 0])
    return mf.Conv(w, mf.put((o,), dtype, b), 1, 0)


def _linear(x: mx.array, wb: tuple[mx.array, mx.array]) -> mx.array:
    """`Linear` of `[..., in]`."""
    shape = x.shape
    return (x.reshape(-1, shape[-1]) @ wb[0].T + wb[1]).reshape(*shape[:-1], -1)


def _ln(x: mx.array, p: tuple[mx.array, mx.array]) -> mx.array:
    return mx.fast.layer_norm(x, p[0], p[1], 1e-5)


class _CostEncoder:
    """`CostPerceiverEncoder` without the volume: maps, patch embedding and input layer per chunk, then the latents."""

    def __init__(self, sd: Mapping[str, Any], p: str, dtype: mx.Dtype) -> None:
        self.dtype = dt = dtype
        pe = f"{p}.patch_embed"
        self.convs = [mf.conv(sd, f"{pe}.proj.{i}", dt, stride=2, padding=2) for i in (0, 2, 4)]
        c = self.c = int(sd[f"{pe}.proj.4.weight"].shape[0])  # 64; tokens 2 C
        w0 = f"{pe}.ffn_with_coord.0"
        self.f0x = mf.put((2 * c, c), dt, lambda: mf.np64(sd[f"{w0}.weight"])[:, :c, 0, 0])
        self.f0p = mf.put((2 * c, c), dt, lambda: mf.np64(sd[f"{w0}.weight"])[:, c:, 0, 0])
        self.f0b = mf.put((2 * c,), dt, lambda: mf.np64(sd[f"{w0}.bias"]))
        w2 = f"{pe}.ffn_with_coord.2"
        self.f2 = (
            mf.put((2 * c, 2 * c), dt, lambda: mf.np64(sd[f"{w2}.weight"])[:, :, 0, 0]),
            mf.put((2 * c,), dt, lambda: mf.np64(sd[f"{w2}.bias"])),
        )
        self.pnorm = tw.norm(sd, f"{pe}.norm", dt)
        # the input layer: constant latent queries
        il = f"{p}.input_layer"
        k_, d_ = (int(n) for n in sd[f"{p}.latent_tokens"].shape[1:])  # 4 tokens of 32
        self.tokens, self.dim, self.heads = k_, d_, 8
        hd = d_ // self.heads

        def latent() -> np.ndarray:
            return mf.np64(sd[f"{p}.latent_tokens"])[0]

        def scores() -> np.ndarray:  # [2 C, heads x tokens]: keys' weight times the scaled per-head queries
            q = _layer_norm_np(latent(), mf.np64(sd[f"{il}.norm1.weight"]), mf.np64(sd[f"{il}.norm1.bias"]))
            q = (q @ mf.np64(sd[f"{il}.q.weight"]).T + mf.np64(sd[f"{il}.q.bias"])) * hd**-0.5
            wk = mf.np64(sd[f"{il}.k.weight"]).reshape(self.heads, hd, 2 * c)
            return np.einsum("hdc,ihd->chi", wk, q.reshape(k_, self.heads, hd)).reshape(2 * c, -1)

        self.score = mf.put((2 * c, self.heads * k_), dt, scores)
        self.wv = mf.put(
            (self.heads, 2 * c, hd),
            dt,
            lambda: mf.np64(sd[f"{il}.v.weight"]).reshape(self.heads, hd, -1).transpose(0, 2, 1),
        )
        self.bv = mf.put((d_,), dt, lambda: mf.np64(sd[f"{il}.v.bias"]))
        self.latent = mf.put((k_, d_), dt, latent)
        self.proj = mf.linear(sd, f"{il}.proj", dt)
        self.n2 = tw.norm(sd, f"{il}.norm2", dt)
        self.ffn = (mf.linear(sd, f"{il}.ffn.0", dt), mf.linear(sd, f"{il}.ffn.3", dt))
        # the self-attention layer over each pixel's tokens
        el = f"{p}.encoder_layers.0"
        self.e1, self.e2 = tw.norm(sd, f"{el}.norm1", dt), tw.norm(sd, f"{el}.norm2", dt)

        def qkv(part: str) -> np.ndarray:
            s = np.array([hd**-0.5, 1, 1])
            return np.concatenate([mf.np64(sd[f"{el}.{n}.{part}"]) * f for n, f in zip("qkv", s, strict=True)])

        self.eqkv = (mf.put((3 * d_, d_), dt, lambda: qkv("weight")), mf.put((3 * d_,), dt, lambda: qkv("bias")))
        self.eproj = mf.linear(sd, f"{el}.proj", dt)
        self.effn = (mf.linear(sd, f"{el}.ffn.0", dt), mf.linear(sd, f"{el}.ffn.3", dt))
        vl = f"{p}.vertical_encoder_layers.0"
        self.local = tw.LocalBlock(sd, f"{vl}.local_block", dt)
        self.glob = tw.GlobalBlock(sd, f"{vl}.global_block", dt)
        self._chunk = stepped(self._embed)
        self._latents = stepped(self._encode)

    def position(self, gh: int, gw: int) -> mx.array:
        """`ffn_with_coord.0`'s position half and bias for a `gh` x `gw` patch grid: `[gh gw, 2 C]`."""
        xy = mf.grid(gh, gw).reshape(-1, 2) * PATCH + PATCH / 2
        return tw.sine(xy, self.c).astype(self.dtype) @ self.f0p.T + self.f0b

    def _embed(self, f1: mx.array, f2t: mx.array, pos: mx.array, h: int, w: int) -> mx.array:
        """The latent tokens `[M, 4, 32]` of the cost maps of the query pixels `f1` `[M, C]` (`f2t` `[C, h w]`)."""
        m = f1.shape[0]
        x = (f1 @ f2t).reshape(m, h, w, 1)
        x = mx.pad(x, ((0, 0), (0, -h % PATCH), (0, -w % PATCH), (0, 0)))
        a, b, c = self.convs
        x = c(mf.relu(b(mf.relu(a(x)))))
        j = x.shape[1] * x.shape[2]
        t = mf.relu(x.reshape(m * j, -1) @ self.f0x.T + mx.tile(pos, (m, 1)))
        t = _ln(t @ self.f2[0].T + self.f2[1], self.pnorm).reshape(m, j, -1)  # [M, J, 2 C]
        heads, k_ = self.heads, self.tokens
        s = mx.softmax((t.reshape(m * j, -1) @ self.score).reshape(m, j, -1), axis=1, precise=True)
        agg = (s.transpose(0, 2, 1) @ t).reshape(m, heads, k_, -1)  # [M, heads, tokens, 2 C]
        o = (agg @ self.wv).transpose(0, 2, 1, 3).reshape(m, k_, -1) + self.bv  # [M, tokens, 32]
        x = self.latent + _linear(o, self.proj)
        return x + _linear(mf.gelu(_linear(_ln(x, self.n2), self.ffn[0])), self.ffn[1])

    def _encode(self, x: mx.array, h: int, w: int) -> mx.array:
        """The latents `[h w, 4, 32]` after the self-attention and the Twins blocks, plus the shortcut."""
        n, k_, d_ = x.shape
        heads = self.heads
        y = _linear(_ln(x, self.e1), self.eqkv).reshape(n, k_, 3, heads, d_ // heads).transpose(2, 0, 3, 1, 4)
        o = attention(y[0], y[1], y[2], scale=1.0).transpose(0, 2, 1, 3).reshape(n, k_, d_)
        z = x + _linear(o, self.eproj)
        z = z + _linear(mf.gelu(_linear(_ln(z, self.e2), self.effn[0])), self.effn[1])
        z = z.reshape(h, w, k_, d_).transpose(2, 0, 1, 3)  # [tokens, h, w, 32]
        z = self.glob(self.local(z))
        return z.transpose(1, 2, 0, 3).reshape(n, k_, d_) + x

    def __call__(self, f1: mx.array, f2: mx.array) -> mx.array:
        """The cost memory `[h w, 4, 32]` of the features `[1, h, w, C]`."""
        _, h, w, c = f1.shape
        f1 = f1.reshape(h * w, c)
        f2t = f2.reshape(h * w, c).T
        pos = self.position(-(-h // PATCH), -(-w // PATCH))
        parts = [self._chunk(f1[s : s + EMBED_MAPS], f2t, pos, h, w) for s in range(0, h * w, EMBED_MAPS)]
        return self._latents(mx.concatenate(parts), h, w)


def _sep_conv(sd: Mapping[str, Any], names: tuple[str, ...], dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """The `Conv2d`s `names` (same input) stacked: OHWI weight, bias."""
    o = sum(int(sd[f"{n}.weight"].shape[0]) for n in names)
    _, i, kh, kw = (int(n) for n in sd[f"{names[0]}.weight"].shape)
    return (
        mf.put(
            (o, kh, kw, i),
            dtype,
            lambda: np.concatenate([mf.np64(sd[f"{n}.weight"]) for n in names]).transpose(0, 2, 3, 1),
        ),
        mf.put((o,), dtype, lambda: np.concatenate([mf.np64(sd[f"{n}.bias"]) for n in names])),
    )


class MlxFlowFormer:
    """FlowFormer Small from the state dict `sd` in `dtype` (the name), `iters` updates."""

    def __init__(self, sd: Mapping[str, Any], dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.cnet = _Encoder(sd, "context_encoder", dt)
        self.cout = _folded(sd, "context_encoder.conv2", "memory_decoder.proj", dt, bias=True)
        self.fnet = _Encoder(sd, "memory_encoder.feat_encoder", dt)
        self.fout = _folded(sd, "memory_encoder.feat_encoder.conv2", "memory_encoder.channel_convertor", dt, bias=False)
        self.cost = _CostEncoder(sd, "memory_encoder.cost_perceiver_encoder", dt)
        d = "memory_decoder"
        self.tok = (mf.conv(sd, f"{d}.flow_token_encoder.0", dt), mf.conv(sd, f"{d}.flow_token_encoder.2", dt))
        ca = f"{d}.decoder_layer.cross_attend"
        self.qd = int(sd[f"{ca}.q.weight"].shape[0])  # 64
        hd = self.qd // 8
        self.cn1, self.cn2 = tw.norm(sd, f"{ca}.norm1", dt), tw.norm(sd, f"{ca}.norm2", dt)
        self.cq = (
            mf.put((self.qd, self.qd), dt, lambda: mf.np64(sd[f"{ca}.q.weight"]) * hd**-0.5),
            mf.put((self.qd,), dt, lambda: mf.np64(sd[f"{ca}.q.bias"]) * hd**-0.5),
        )
        self.ckv = (
            mf.put((2 * self.qd, 32), dt, lambda: np.concatenate([mf.np64(sd[f"{ca}.{n}.weight"]) for n in "kv"])),
            mf.put((2 * self.qd,), dt, lambda: np.concatenate([mf.np64(sd[f"{ca}.{n}.bias"]) for n in "kv"])),
        )
        self.cproj = mf.linear(sd, f"{ca}.proj", dt)
        self.cffn = (mf.linear(sd, f"{ca}.ffn.0", dt), mf.linear(sd, f"{ca}.ffn.3", dt))
        self.qk = mf.conv(sd, f"{d}.att.to_qk", dt, bias=False)
        u = f"{d}.update_block"
        e = f"{u}.encoder"
        self.convc1, self.convc2 = mf.conv(sd, f"{e}.convc1", dt), mf.conv(sd, f"{e}.convc2", dt)
        self.convf1 = mf.conv(sd, f"{e}.convf1", mx.float32)
        self.convf2, self.conv = mf.conv(sd, f"{e}.convf2", dt), mf.conv(sd, f"{e}.conv", dt)
        a = f"{u}.aggregator"
        self.to_v = mf.put(
            (HIDDEN, HIDDEN), dt, lambda: mf.np64(sd[f"{a}.to_v.weight"])[:, :, 0, 0] * mf.np64(sd[f"{a}.gamma"])
        )
        g = f"{u}.gru"
        self.zr1, self.q1 = _sep_conv(sd, (f"{g}.convz1", f"{g}.convr1"), dt), _sep_conv(sd, (f"{g}.convq1",), dt)
        self.zr2, self.q2 = _sep_conv(sd, (f"{g}.convz2", f"{g}.convr2"), dt), _sep_conv(sd, (f"{g}.convq2",), dt)
        self.h1, self.h2 = mf.conv(sd, f"{u}.flow_head.conv1", dt), mf.conv(sd, f"{u}.flow_head.conv2", dt)
        self.m1 = mf.conv(sd, f"{u}.mask.0", dt)
        self.m2 = mf.conv(sd, f"{u}.mask.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self._context = stepped(self._context_fn)
        self._features = stepped(self.fout)
        self._memory = stepped(lambda mem: _linear(mem, self.ckv))
        # an update in four steps: one compiled update held all its intermediates to the end (+0.34 GB at 544 x 960
        # float32), and two halves in flight together 0.35 GB; four pieces pipeline at the same speed (0.2 GB)
        self._update = [stepped(f) for f in (self._decode, self._motion, self._gru1, self._gru2)]
        self._mask = stepped(lambda net: self.m2(mf.relu(self.m1(net))))

    def _context_fn(self, x: mx.array) -> list[mx.array]:
        """Hidden state, context and GMA's matrix (or its scaled query and key)."""
        c = self.cout(x)
        _, h, w, _ = c.shape
        net, inp = mx.tanh(c[..., :HIDDEN]), mf.relu(c[..., HIDDEN:])
        n = h * w
        qk = self.qk(inp).reshape(n, 2 * HIDDEN)
        q, k = qk[:, :HIDDEN] * HIDDEN**-0.5, qk[:, HIDDEN:]
        if n * n * 4 <= MATRIX_BYTES:
            return [net, inp, mx.softmax(q @ k.T, axis=-1, precise=True)]
        return [net, inp, q[None, None], k[None, None]]

    @staticmethod
    def _gru_half(h: mx.array, x: mx.array, zr: tuple, q: tuple, pad: tuple[int, int]) -> mx.array:
        def conv(a: mx.array, wb: tuple) -> mx.array:
            return mg._conv2d(a, wb[0], (1, 1), pad, (1, 1), 1) + wb[1]  # noqa: SLF001 - the translator's dispatch

        z, r = mx.split(mx.sigmoid(conv(mx.concatenate([h, x], axis=-1), zr)), 2, axis=-1)
        cand = mx.tanh(conv(mx.concatenate([r * h, x], axis=-1), q))
        return h + z * (cand - h)

    def _decode(self, flow: mx.array, kv: mx.array, f1: mx.array, f2: mx.array) -> mx.array:
        """An update's cost windows decoded against the cost memory `kv`, with the windows: `[1, h, w, 64 + 81]`."""
        _, h, w, _ = flow.shape
        dt, n, qd = self.dtype, h * w, self.qd
        coords = mf.grid(h, w) + flow
        cost = ops.corr_windows(f1, [f2], coords.transpose(0, 3, 1, 2), RADIUS, 1.0).transpose(0, 2, 3, 1)
        cost = cost.astype(dt)  # [1, h, w, 81]
        query = self.tok[1](mf.gelu(self.tok[0](cost))).reshape(n, qd)
        pos = tw.sine(coords.reshape(n, 2), qd).astype(dt)
        q = _linear(_ln(query, self.cn1) + pos, self.cq).reshape(n, 1, 8, qd // 8)
        kv4 = kv.reshape(n, -1, 2, 8, qd // 8)
        wts = mx.softmax((q * kv4[:, :, 0]).sum(-1), axis=1, precise=True)  # [n, tokens, heads]
        o = (wts[..., None] * kv4[:, :, 1]).sum(1).reshape(n, qd)
        x = query + _linear(mx.concatenate([o, query], axis=-1), self.cproj)
        x = x + _linear(mf.gelu(_linear(_ln(x, self.cn2), self.cffn[0])), self.cffn[1])
        return mx.concatenate([x.reshape(1, h, w, qd), cost], axis=-1)

    def _motion(self, flow: mx.array, corr: mx.array) -> mx.array:
        """The motion features of the decoded windows `corr` and the flow."""
        dt = self.dtype
        cor = mf.relu(self.convc2(mf.relu(self.convc1(corr))))
        flo = mf.relu(self.convf2(mf.relu(self.convf1(flow)).astype(dt)))
        return mx.concatenate([mf.relu(self.conv(mx.concatenate([cor, flo], axis=-1))), flow.astype(dt)], axis=-1)

    def _gru1(self, net: mx.array, motion: mx.array, inp: mx.array, *att: mx.array) -> tuple:
        """GMA's aggregation of `motion` and the GRU's horizontal half: the hidden state and the GRU's input."""
        n = net.shape[1] * net.shape[2]
        v = motion.reshape(n, HIDDEN) @ self.to_v.T
        readout = att[0] @ v if len(att) == 1 else attention(att[0], att[1], v[None, None], scale=1.0)[0, 0]
        xg = mx.concatenate([inp, motion, motion + readout.reshape(motion.shape)], axis=-1)
        return self._gru_half(net, xg, self.zr1, self.q1, (0, 2)), xg

    def _gru2(self, net: mx.array, xg: mx.array, flow: mx.array) -> tuple:
        """The GRU's vertical half and the flow head."""
        net = self._gru_half(net, xg, self.zr2, self.q2, (2, 0))
        return net, flow + self.h2(mf.relu(self.h1(net))).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        dt = self.dtype
        x1 = (image1.transpose(0, 2, 3, 1) * (2 / 255) - 1).astype(dt)
        x2 = (image2.transpose(0, 2, 3, 1) * (2 / 255) - 1).astype(dt)
        f = self._features(self.fnet(mx.concatenate([x1, x2], axis=0)))
        f1, f2 = f[:1], f[1:]
        kv = self._memory(self.cost(f1, f2))
        # the context after the cost memory: GMA's matrix (0.27 GB at 544 x 960 float32) was live through both peaks
        net, inp, *att = self._context(self.cnet(x1))
        _, h, w, _ = net.shape
        flow = mx.zeros((1, h, w, 2), dtype=mx.float32)
        for _ in range(self.iters):
            decode, motion, gru1, gru2 = self._update
            m = motion(flow, decode(flow, kv, f1, f2))
            net, flow = gru2(*gru1(net, m, inp, *att), flow)
        up = mf.convex_upsample(flow, self._mask(net), FACTOR)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [0, 255]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
