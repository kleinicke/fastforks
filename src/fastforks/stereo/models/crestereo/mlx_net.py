"""CREStereo on Apple MLX (`device="mlx"`): the maths of the portable forward (`portable.forward_fast`).

NHWC throughout. Everything is a pure function of arrays and a flat parameter dict; the pieces that differ between
portable (MLX ops only: `mlx_ops`) and kernels (Metal kernels: `fastforks.kernels.crestereo_mlx.ops`) come in
through an `ops` namespace: `instance_norm_relu`, `corr_iter`, `corr_offset`, `gru`, `convex_upsample`. `Engine`
is what the runners hold.

Prepared once at load (`prepare`, exact in real arithmetic):
- conv weights (O, I, H, W) -> (O, H, W, I); MegEngine's `[1, C, 1, 1]` biases flattened.
- biases of the convolutions that feed an instance norm dropped (the norm removes a per-channel constant).
- the motion encoder's last conv padded from 126 to 128 outputs (two zero filters): MLX's Winograd 3x3 path needs
  channel counts that are multiples of 32, and is twice as fast here as the general path.
- the encoder's 96-channel stage (layer2) padded to 128 channels with zero filters: MLX's 3x3 convolution with 96
  outputs is 3-6x slower than with 128 on the M1 (a zero channel stays zero through the instance norm and the relu).
  Tunable `conv_pad`.
- GRU: z and r share one convolution; the context (`inp`) columns of all six GRU convolutions are split off into one
  conv per half that runs once per stage (with the biases), so each iteration convolves hidden + motion only.

float16: convolutions and their activations fp16. Kept in fp32: the flow and everything derived from it (coordinates,
offsets, the 7x7 flow convolution's input), instance-norm statistics, correlation accumulation, attention at 1/16,
the upsampling masks' softmax, and the Winograd-domain GRU weights of the kernels mode.
"""
# ruff: noqa: ANN001, ANN201, ANN202, D103, D205, PLR0913, PLR0917, PLR2004, FBT002 - array-in / array-out maths
# functions (positional arrays and flags, as the torch forward they mirror); shapes are in the docstrings

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from fastforks._core.checkpoint import NUMPY_ARRAYS
from fastforks._core.mlx import converted, mx, to_mx, to_torch, tuning

if TYPE_CHECKING:
    from types import ModuleType

    import torch

    from fastforks.stereo._runner import Context

HDIM = 128  # GRU hidden channels
ATTENTION = ("self_att_fn.", "cross_att_fn.")
NORMED = ("fnet.conv1", "fnet.layer")  # convolutions followed by an instance norm
NARROW, WIDE = 96, 128  # the encoder's 96-channel stage is run 128 wide (tunable `conv_pad`)
LAYOUT = "v1"  # version of `prepare`'s parameter layout (part of the conversion cache's file name)


# --------------------------------------------------------------------------- weights
def prepare(state: dict[str, np.ndarray], *, pad: bool) -> dict[str, mx.array]:
    """MegEngine checkpoint arrays -> the float32 arrays the forward uses (see module docstring).

    `pad`: widen the encoder's 96-channel stage to 128.
    """
    s = {k: np.asarray(v, np.float32) for k, v in state.items()}
    out: dict[str, np.ndarray] = {}
    for k, v in s.items():
        if k.startswith("update_block.gru."):
            continue
        if k.startswith(ATTENTION):
            out[k] = v.T if v.ndim == 2 else v  # linear weights as [in, out]
        elif k.endswith(".bias"):
            if not k.startswith(NORMED):
                out[k] = v.reshape(-1)
        else:
            out[k] = v.transpose(0, 2, 3, 1)
    w, b = out["update_block.encoder.conv.weight"], out["update_block.encoder.conv.bias"]
    out["update_block.encoder.conv.weight"] = np.concatenate([w, np.zeros((2, *w.shape[1:]), np.float32)])
    out["update_block.encoder.conv.bias"] = np.concatenate([b, np.zeros(2, np.float32)])
    wide = [k for k in out if k.startswith(("fnet.layer2.", "fnet.layer3.0.")) and k.endswith(".weight")]
    for k in wide if pad else ():
        v = out[k]  # layer2's 96 channels -> 128 (zero filters / zero taps)
        grow = (
            WIDE - v.shape[0] if k.startswith("fnet.layer2.") else 0,
            WIDE - v.shape[3] if v.shape[3] == NARROW else 0,
        )
        out[k] = np.pad(v, [(0, grow[0]), (0, 0), (0, 0), (0, grow[1])])
    for half in (1, 2):
        wz, wr, wq = (s[f"update_block.gru.conv{t}{half}.weight"].transpose(0, 2, 3, 1) for t in "zrq")
        bias = np.concatenate([s[f"update_block.gru.conv{t}{half}.bias"].reshape(-1) for t in "zrq"])
        own = lambda a: np.concatenate([a[..., :HDIM], a[..., 2 * HDIM :]], axis=-1)  # noqa: E731 - hidden + motion
        out[f"gru{half}.zr.weight"] = own(np.concatenate([wz, wr]))  # [256, kh, kw, 256]
        out[f"gru{half}.q.weight"] = own(wq)  # [128, kh, kw, 256]
        out[f"gru{half}.ctx.weight"] = np.concatenate([wz, wr, wq])[..., HDIM : 2 * HDIM]  # [384, kh, kw, 128]
        out[f"gru{half}.ctx.bias"] = bias
    return {k: mx.array(np.ascontiguousarray(v)) for k, v in out.items()}


def fp32(k: str) -> bool:
    """Parameters that stay float32 in a float16 run: the attention, the flow convolution (fp32 input) and the
    Winograd-domain GRU weights (the conversion cache stores them so).
    """
    return k.startswith((*ATTENTION, "update_block.encoder.convf1")) or k.endswith(".wino")


def cast(p32: dict[str, mx.array], dtype: mx.Dtype) -> dict[str, mx.array]:
    """Parameters in the run dtype, but `fp32`'s (a no-op on arrays the cache stored in the run dtype)."""
    return {k: v if fp32(k) else v.astype(dtype) for k, v in p32.items()}


# --------------------------------------------------------------------------- small ops
def conv(x, p, name, stride=1, pad=0):
    y = mx.conv2d(x, p[name + ".weight"], stride=stride, padding=pad)
    b = p.get(name + ".bias")
    return y if b is None else y + b


def avg_pool(x, k):
    n, h, w, c = x.shape
    return x.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4))


@functools.lru_cache(maxsize=64)
def _interp_axis(n_in: int, n_out: int):
    src = np.arange(n_out, dtype=np.float64) * ((n_in - 1) / (n_out - 1) if n_out > 1 else 0.0)
    i0 = np.clip(np.floor(src).astype(np.int32), 0, n_in - 1)
    return mx.array(i0), mx.array(np.clip(i0 + 1, 0, n_in - 1)), mx.array((src - i0).astype(np.float32))


def resize(x, size):
    """Bilinear, align_corners=True, NHWC."""
    _, h, w, _ = x.shape
    oh, ow = size
    if (oh, ow) == (h, w):
        return x
    i0, i1, w1 = _interp_axis(h, oh)
    w1 = w1[None, :, None, None]
    x = x[:, i0] * (1 - w1) + x[:, i1] * w1
    i0, i1, w1 = _interp_axis(w, ow)
    w1 = w1[None, None, :, None]
    return x[:, :, i0] * (1 - w1) + x[:, :, i1] * w1


def resize_flow(flow, hw):
    """Stage hand-over `-scale * interpolate(flow)` (sign flip included)."""
    return (-hw[0] / flow.shape[1]) * resize(flow, hw)


# --------------------------------------------------------------------------- encoder
def _res_block(ops, x, p, name, stride):
    y = ops.instance_norm_relu(mx.conv2d(x, p[f"{name}.conv1.weight"], stride=stride, padding=1))
    y = ops.instance_norm_relu(mx.conv2d(y, p[f"{name}.conv2.weight"], padding=1))
    d = mx.conv2d(x, p[f"{name}.downsample.0.weight"], stride=stride)
    return ops.instance_norm_relu(d, y)  # relu(norm(d) + y)


def fnet(ops, p, x):
    """BasicEncoder (instance norm, per image): `[N, H, W, 3]` -> `[N, H/4, W/4, 256]`."""
    x = ops.instance_norm_relu(mx.conv2d(x, p["fnet.conv1.weight"], stride=2, padding=3))
    for layer, stride in (("layer1", 1), ("layer2", 2), ("layer3", 1)):
        x = _res_block(ops, x, p, f"fnet.{layer}.0", stride)
        x = _res_block(ops, x, p, f"fnet.{layer}.1", 1)
    return conv(x, p, "fnet.conv2")


# --------------------------------------------------------------------------- attention at 1/16, float32
@functools.lru_cache(maxsize=8)
def position_encoding(c, h, w):
    """`PositionEncodingSine.pe` as `[1, h, w, c]`, with the original's `-log(1e4) / d_model // 2` (a floor
    division: the frequency step is exactly -1).
    """
    step = (-math.log(10000.0) / c) // 2
    div = np.exp(np.arange(0, c // 2, 2, dtype=np.float32) * np.float32(step))
    ypos = np.cumsum(np.ones((h, w), np.float32), 0)[..., None]
    xpos = np.cumsum(np.ones((h, w), np.float32), 1)[..., None]
    pe = np.zeros((h, w, c), np.float32)
    pe[..., 0::4], pe[..., 1::4] = np.sin(xpos * div), np.cos(xpos * div)
    pe[..., 2::4], pe[..., 3::4] = np.sin(ypos * div), np.cos(ypos * div)
    return mx.array(pe[None])


def _linear_attention(q, k, v, eps=1e-6):
    q = mx.where(q > 0, q + 1, mx.exp(q))  # elu(x) + 1
    k = mx.where(k > 0, k + 1, mx.exp(k))
    length = v.shape[1]
    kv = mx.einsum("nshd,nshv->nhdv", k, v / length)
    z = 1 / (mx.einsum("nlhd,nhd->nlh", q, k.sum(axis=1)) + eps)
    return mx.einsum("nlhd,nhdv,nlh->nlhv", q, kv, z) * length


def _loftr_layer(p, name, x, source, nhead=8):
    n, length, c = x.shape
    d = c // nhead
    q = (x @ p[f"{name}.q_proj.weight"]).reshape(n, length, nhead, d)
    k = (source @ p[f"{name}.k_proj.weight"]).reshape(n, -1, nhead, d)
    v = (source @ p[f"{name}.v_proj.weight"]).reshape(n, -1, nhead, d)
    msg = _linear_attention(q, k, v).reshape(n, length, c) @ p[f"{name}.merge.weight"]
    msg = mx.fast.layer_norm(msg, p[f"{name}.norm1.weight"], p[f"{name}.norm1.bias"], 1e-5)
    msg = mx.maximum(mx.concatenate([x, msg], axis=2) @ p[f"{name}.mlp.0.weight"], 0) @ p[f"{name}.mlp.2.weight"]
    return x + mx.fast.layer_norm(msg, p[f"{name}.norm2.weight"], p[f"{name}.norm2.bias"], 1e-5)


def attention(p, f1, f2):
    """Position encoding, self attention (per image), then cross attention."""
    n, h, w, c = f1.shape
    pe = position_encoding(c, h, w)
    s1, s2 = ((x + pe).reshape(n, h * w, c) for x in (f1, f2))
    s1 = _loftr_layer(p, "self_att_fn.layers.0", s1, s1)
    s2 = _loftr_layer(p, "self_att_fn.layers.0", s2, s2)
    s1 = _loftr_layer(p, "cross_att_fn.layers.0", s1, s2)
    s2 = _loftr_layer(p, "cross_att_fn.layers.0", s2, s1)  # the updated s1, as upstream
    return s1.reshape(n, h, w, c), s2.reshape(n, h, w, c)


# --------------------------------------------------------------------------- update block
def context(p, inp):
    """The six GRU convolutions' context terms (+ biases), once per stage: `[(zr, q), (zr, q)]`."""
    out = []
    for half, pad in ((1, (0, 2)), (2, (2, 0))):
        c = mx.conv2d(inp, p[f"gru{half}.ctx.weight"], padding=pad) + p[f"gru{half}.ctx.bias"]
        out.append((c[..., : 2 * HDIM], c[..., 2 * HDIM :]))
    return out


def motion_features(p, flow, corr, dtype, raw=False):
    """Motion encoder up to its last conv: `[N, H, W, 128]`, channels 126-127 are zero (the flow's place).
    `raw`: without the last conv's bias and relu (the kernels variant's `gru_pack` applies them).
    """
    e = "update_block.encoder."
    cor = mx.maximum(conv(corr, p, e + "convc1"), 0)
    cor = mx.maximum(conv(cor, p, e + "convc2", 1, 1), 0)
    flo = mx.maximum(conv(flow, p, e + "convf1", 1, 3), 0).astype(dtype)  # fp32 conv on the fp32 flow
    flo = mx.maximum(conv(flo, p, e + "convf2", 1, 1), 0)
    x = mx.concatenate([cor, flo], -1)
    if raw:
        return mx.conv2d(x, p[e + "conv.weight"], padding=1)
    return mx.maximum(conv(x, p, e + "conv", 1, 1), 0)


def flow_head(p, hidden):
    h = mx.maximum(conv(hidden, p, "update_block.flow_head.conv1", 1, 1), 0)
    return conv(h, p, "update_block.flow_head.conv2", 1, 1).astype(mx.float32)


def mask_head(p, hidden):
    m = conv(mx.maximum(conv(hidden, p, "update_block.mask.0", 1, 1), 0), p, "update_block.mask.2")
    return 0.25 * m.astype(mx.float32)


def make_steps(ops, p, dtype, compile_steps=True):
    """The update steps `(hidden, flow, f1, f2, offs, czr1, cq1, czr2, cq2) -> (hidden, flow)` keyed by
    `(kind, small_patch)`; kind "off" = correlation at learned offsets, "iter" = warp + local window. Plus the
    encoder under "fnet". Compiled (one trace per input shape) unless `compile_steps` is off.
    """
    raw = getattr(ops, "RAW_MOTION", False)

    def build(kind, small):
        def step(hidden, flow, f1, f2, offs, czr1, cq1, czr2, cq2):
            corr = ops.corr_offset(f1, f2, offs, flow, small) if kind == "off" else ops.corr_iter(f1, f2, flow, small)
            motion = motion_features(p, flow, corr.astype(dtype), dtype, raw)
            hidden = ops.gru(p, hidden, motion, flow, czr1, cq1, czr2, cq2)
            return hidden, flow + flow_head(p, hidden)

        return mx.compile(step) if compile_steps else step

    steps: dict[Any, Any] = {(k, s): build(k, s) for k in ("off", "iter") for s in (False, True)}
    encoder = lambda x: fnet(ops, p, x)  # noqa: E731
    steps["fnet"] = mx.compile(encoder) if compile_steps else encoder
    return steps


def stage(p, steps, kind, hidden, inp, f1, f2, offs, flow, iters):
    """One cascade stage: `iters` updates (3x3 window on odd iterations), then the upsampling mask."""
    (czr1, cq1), (czr2, cq2) = context(p, inp)
    mx.eval(czr1, cq1, czr2, cq2, f1, f2, hidden, flow, offs)
    for i in range(iters):
        hidden, flow = steps[kind, i % 2 == 1](hidden, flow, f1, f2, offs, czr1, cq1, czr2, cq2)
        mx.eval(hidden, flow)
    return flow, mask_head(p, hidden)


def run_pass(ops, p, steps, images, iters, flow_init, dtype, pair=False):
    """`CREStereo.forward` (test mode) on the stacked BGR pair `[2, H, W, 3]` (0-255): `(flow, mask)` at 1/4.

    `pair` (tunable `batch_pair`): both images through the encoder in one batch; else one at a time, which is as
    fast on the M1 and halves the convolutions' temporaries.
    """
    x = (2 * (images / 255.0) - 1.0).astype(dtype)
    if pair:
        fmaps = steps["fnet"](x)
        fmap1, fmap2 = fmaps[:1], fmaps[1:]
        mx.eval(fmap1, fmap2)
    else:
        fmap1 = steps["fnet"](x[:1])
        mx.eval(fmap1)
        fmap2 = steps["fnet"](x[1:])
        mx.eval(fmap2)
    hidden, inp = mx.tanh(fmap1[..., :HDIM]), mx.maximum(fmap1[..., HDIM:], 0)
    if flow_init is None:
        f1_8, f2_8, f1_16, f2_16 = avg_pool(fmap1, 2), avg_pool(fmap2, 2), avg_pool(fmap1, 4), avg_pool(fmap2, 4)
        off8 = (mx.sigmoid(conv(f1_8, p, "conv_offset_8", 1, 1).astype(mx.float32)) - 0.5) * 2.0
        off16 = (mx.sigmoid(conv(f1_16, p, "conv_offset_16", 1, 1).astype(mx.float32)) - 0.5) * 2.0
        a1, a2 = attention(p, f1_16.astype(mx.float32), f2_16.astype(mx.float32))
        flow = mx.zeros((1, *a1.shape[1:3], 2), mx.float32)
        flow, mask = stage(p, steps, "off", avg_pool(hidden, 4), avg_pool(inp, 4), a1, a2, off16, flow, iters // 2)
        flow = resize_flow(ops.convex_upsample(flow, mask), f1_8.shape[1:3])
        flow, mask = stage(p, steps, "off", avg_pool(hidden, 2), avg_pool(inp, 2), f1_8, f2_8, off8, flow, iters // 2)
        flow = resize_flow(ops.convex_upsample(flow, mask), fmap1.shape[1:3])
    else:
        flow = resize_flow(flow_init, fmap1.shape[1:3])
    return stage(p, steps, "iter", hidden, inp, fmap1, fmap2, mx.zeros((1,), mx.float32), flow, iters)


def forward(ops, p, steps, left, right, iters, dtype, pair=False):
    """Padded RGB `[1, 3, H, W]` 0-255 (H, W multiples of 32) -> disparity `[1, H, W]` float32."""
    x = mx.concatenate([left, right]).astype(mx.float32).transpose(0, 2, 3, 1)[..., ::-1]  # BGR, as cv2.imread
    h, w = x.shape[1:3]
    flow, mask = run_pass(ops, p, steps, resize(x, (h // 2, w // 2)), iters, None, dtype, pair)
    flow, mask = run_pass(ops, p, steps, x, iters, -ops.convex_upsample(flow, mask), dtype, pair)
    return -ops.convex_upsample(flow, mask)[..., 0]


class Engine:
    """The MLX network of one loaded model: converted weights, the ops of its mode, the compiled steps."""

    def __init__(self, ctx: Context, ops: ModuleType) -> None:
        """Convert (or read the cached) weights for `ctx` and build the steps with `ops`."""
        tune = tuning()
        extend = getattr(ops, "extend", None)  # the kernels mode's extra parameters (Winograd-domain GRU weights)

        def build() -> dict[str, Any]:
            state = ctx.state_dict(key="state_dict", strip="", extra=NUMPY_ARRAYS)  # MegEngine pickle of numpy arrays
            p32 = prepare({k: v.numpy() for k, v in state.items()}, pad=tune.conv_pad)
            return p32 if extend is None else p32 | extend(p32)

        layout = f"{LAYOUT}-w{WIDE if tune.conv_pad else NARROW}"
        if extend is not None:
            layout += f"-{ctx.mode}-wino{int(tune.winograd)}"
        self.ops = ops
        self.dtype = getattr(mx, ctx.dtype)
        self.p = cast(converted(ctx, layout, build, dtype=ctx.dtype, fp32=fp32), self.dtype)
        mx.eval(self.p)
        self.steps = make_steps(ops, self.p, self.dtype, tune.compile_steps)
        self.pair = tune.batch_pair

    def __call__(self, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
        """Padded `[1, 3, H, W]` 0-255 CPU tensors -> disparity `[1, H, W]` float32 on the CPU."""
        out = forward(self.ops, self.p, self.steps, to_mx(left), to_mx(right), iters, self.dtype, self.pair)
        return to_torch(out)
