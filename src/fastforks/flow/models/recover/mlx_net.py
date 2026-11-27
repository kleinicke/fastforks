"""ReCoVEr on Apple MLX, written in `mlx.core` operations: `MlxReCoVEr(sd, arch, dtype, iters)(image1, image2)` -> flow.

The maths of `net.ReCoVEr.forward`, channels last, built from the checkpoint's state dict (`shared.mlx_flow`):

- the context encoder on the six-channel pair, its batch norms folded, each block one compiled step
  (`cspnext.stepped`): the ResNet34 stages (`rn`), the MobileNetV3-Large features (`mn`: depthwise convs through
  `graph_mlx.ops.depthwise`), or the ConvNeXt-T stages (`cx`: the patch convs as GEMMs, the layer scales folded);
- the update step (motion encoder, SEA-RAFT's ConvNeXt refinement, the flow head reduced to its 2 flow outputs) one
  compiled step, one update in flight; upstream's all-zero correlation enters as `corr_term`, computed once per
  feature size on a 5 x 5 canvas and expanded (as `BasicMotionEncoder.corr_term`);
- the upsampling head with its 0.25 folded in, the convex upsampling in float32 (`mlx_flow.convex_upsample`);
- float32: the flow state and the upsampling; in float16 the convs MLX would run as half-precision Winograd run in
  float32 (`mlx_flow.options`), as in the graph translation.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared import mlx_flow as mf
from fastforks.shared.cspnext.mlx_net import stepped, steps

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

__all__ = ["LAYOUT", "SOURCES", "MlxReCoVEr"]

LAYOUT = "recover-mlx-1"
SOURCES = (Path(__file__), Path(mf.__file__))

FACTOR = 8
DIM = 128
_MOBILENET = (  # upstream size "l": (kernel, expanded, SE, hard swish, stride) of features.1-15, `net.MOBILENET_L`
    (3, 16, False, False, 1), (3, 64, False, False, 2), (3, 72, False, False, 1), (5, 72, True, False, 2),
    (5, 120, True, False, 1), (5, 120, True, False, 1), (3, 240, False, True, 1), (3, 200, False, True, 1),
    (3, 184, False, True, 1), (3, 184, False, True, 1), (3, 480, True, True, 1), (3, 672, True, True, 1),
    (5, 672, True, True, 1), (5, 960, True, True, 1), (5, 960, True, True, 1),
)  # fmt: skip


def _hardswish(x: mx.array) -> mx.array:
    return x * mx.clip(x + 3, 0, 6) * (1 / 6)


def _hardsigmoid(x: mx.array) -> mx.array:
    return mx.clip(x * (1 / 6) + 0.5, 0, 1)


# --- the context encoders: lists of steps -----------------------------------------------------------------------------


def _resnet(sd: Mapping[str, Any], dt: mx.Dtype) -> list[Callable[[mx.array], mx.array]]:
    """`ResNetFPN`: 7 x 7 stride-2 stem, ResNet34 stages (relu before the residual sum, as upstream), 1 x 1 conv."""
    stem = mf.conv(sd, "cnet.conv1", dt, stride=2, bn="cnet.bn1")
    out: list[Callable[[mx.array], mx.array]] = [lambda x: mf.relu(stem(x))]
    for layer, n in (("layer1", 3), ("layer2", 4), ("layer3", 6)):
        for i in range(n):
            p = f"cnet.{layer}.{i}"
            stride = 2 if i == 0 and layer != "layer1" else 1
            c1 = mf.conv(sd, f"{p}.conv1", dt, stride=stride, bn=f"{p}.bn1")
            c2 = mf.conv(sd, f"{p}.conv2", dt, bn=f"{p}.bn2")
            ds = None
            if f"{p}.downsample.0.weight" in sd:
                ds = mf.conv(sd, f"{p}.downsample.0", dt, stride=stride, padding=0, bn=f"{p}.downsample.1")

            def block(x: mx.array, c1: mf.Conv = c1, c2: mf.Conv = c2, ds: mf.Conv | None = ds) -> mx.array:
                y = mf.relu(c2(mf.relu(c1(x))))
                return mf.relu(y + (ds(x) if ds is not None else x))

            out.append(block)
    final = mf.conv(sd, "cnet.final_conv", dt)
    out[-1] = _then(out[-1], final)
    return out


def _then(f: Callable[[mx.array], mx.array], g: Callable[[mx.array], mx.array]) -> Callable[[mx.array], mx.array]:
    return lambda x: g(f(x))


def _inverted(  # noqa: PLR0913 - the block's layers
    x: mx.array,
    *,
    expand: mf.Conv | None,
    dw: mf.Depthwise,
    gate: tuple[mf.Conv, mf.Conv] | None,
    project: mf.Conv,
    act: Callable[[mx.array], mx.array],
    residual: bool,
) -> mx.array:
    """The torchvision MobileNetV3 `InvertedResidual` (batch norms folded)."""
    y = act(expand(x)) if expand is not None else x
    y = act(dw(y))
    if gate is not None:
        s = mf.relu(gate[0](y.mean(axis=(1, 2), keepdims=True)))
        y = y * _hardsigmoid(gate[1](s))
    y = project(y)
    return x + y if residual else y


def _mobilenet(sd: Mapping[str, Any], dt: mx.Dtype) -> list[Callable[[mx.array], mx.array]]:
    """MobileNetV3-Large features (batch norms eps 1e-3 folded), then the 1 x 1 `final`."""
    eps = 1e-3
    stem = mf.conv(sd, "cnet.features.0.0", dt, stride=2, bn="cnet.features.0.1", eps=eps)
    out: list[Callable[[mx.array], mx.array]] = [lambda x: _hardswish(stem(x))]
    cin = 16
    for i, (_k, exp, se, hs, stride) in enumerate(_MOBILENET, start=1):
        p, j = f"cnet.features.{i}.block", 0
        act = _hardswish if hs else mf.relu
        expand = None
        if exp != cin:
            expand = mf.conv(sd, f"{p}.0.0", dt, bn=f"{p}.0.1", eps=eps)
            j = 1
        dw = mf.depthwise(sd, f"{p}.{j}.0", dt, stride=stride, bn=f"{p}.{j}.1", eps=eps)
        gate = None
        if se:
            gate = (mf.conv(sd, f"{p}.{j + 1}.fc1", dt), mf.conv(sd, f"{p}.{j + 1}.fc2", dt))
            j += 1
        project = mf.conv(sd, f"{p}.{j + 1}.0", dt, bn=f"{p}.{j + 1}.1", eps=eps)
        cout = project.out
        block = partial(
            _inverted, expand=expand, dw=dw, gate=gate, project=project, act=act, residual=stride == 1 and cin == cout
        )
        out.append(block)
        cin = cout
    last = mf.conv(sd, "cnet.features.16.0", dt, bn="cnet.features.16.1", eps=eps)
    final = mf.conv(sd, "cnet.final", dt)
    out.append(lambda x: final(_hardswish(last(x))))
    return out


def _cn_block(  # noqa: PLR0913 - the block's layers
    x: mx.array,
    *,
    dw: mf.Depthwise,
    ln: Callable[[mx.array], mx.array],
    w1: mx.array,
    b1: mx.array,
    w2: mx.array,
    b2: mx.array,
) -> mx.array:
    """The torchvision `CNBlock`, the layer scale folded into the second Linear."""
    n, h, w, c = x.shape
    y = mf.gelu(mx.addmm(b1, ln(dw(x)).reshape(n * h * w, c), w1.T))
    return x + mx.addmm(b2, y, w2.T).reshape(x.shape)


def _layer_norm(sd: Mapping[str, Any], p: str, dt: mx.Dtype) -> Callable[[mx.array], mx.array]:
    c = int(sd[f"{p}.weight"].shape[0])
    w = mf.put((c,), dt, lambda: mf.np64(sd[f"{p}.weight"]))
    b = mf.put((c,), dt, lambda: mf.np64(sd[f"{p}.bias"]))
    return lambda x: mx.fast.layer_norm(x, w, b, 1e-6)


def _convnext(sd: Mapping[str, Any], dt: mx.Dtype) -> list[Callable[[mx.array], mx.array]]:
    """ConvNeXt-T at stride 8: torchvision's `CNBlock`s (layer scale folded) and upstream's (pad +) 2 x 2 convs."""
    f = "cnet.features"
    stem, stem_ln = mf.conv(sd, f"{f}.0.0", dt, stride=4, padding=0), _layer_norm(sd, f"{f}.0.1", dt)
    out: list[Callable[[mx.array], mx.array]] = [lambda x: stem_ln(stem(x))]
    for stage, n in ((1, 3), (3, 3), (5, 9), (7, 3)):
        for i in range(n):
            p = f"{f}.{stage}.{i}"
            dw = mf.depthwise(sd, f"{p}.block.0", dt)
            ln = _layer_norm(sd, f"{p}.block.2", dt)
            w1, b1 = mf.linear(sd, f"{p}.block.3", dt)
            w2, b2 = mf.linear(sd, f"{p}.block.5", dt, scale=f"{p}.layer_scale")

            out.append(partial(_cn_block, dw=dw, ln=ln, w1=w1, b1=b1, w2=w2, b2=b2))
        ln = _layer_norm(sd, f"{f}.{stage + 1}.0", dt)
        if stage == 1:
            down = mf.conv(sd, f"{f}.2.1", dt, stride=2, padding=0)
            out[-1] = _then(out[-1], _then(ln, down))
        else:
            down = mf.conv(sd, f"{f}.{stage + 1}.2", dt, padding=0)  # after ZeroPad2d((0, 1, 0, 1))

            def pad_conv(x: mx.array, ln: Callable[[mx.array], mx.array] = ln, down: mf.Conv = down) -> mx.array:
                return down(mx.pad(ln(x), ((0, 0), (0, 1), (0, 1), (0, 0))))

            out[-1] = _then(out[-1], pad_conv)
    return out


_ENCODERS = {"rn": _resnet, "mn": _mobilenet, "cx": _convnext}


class MlxReCoVEr:
    """ReCoVEr (`arch`: `rn`, `mn`, `cx`) from the state dict `sd` in `dtype` (the name), `iters` updates."""

    def __init__(self, sd: Mapping[str, Any], arch: str, dtype: str, iters: int) -> None:
        """Convert the weights (through the conversion cache, `mlx_port.recorded`)."""
        dt = self.dtype = mf.DTYPES[dtype]
        self.iters = iters
        self.encoder = [stepped(f) for f in _ENCODERS[arch](sd, dt)]
        self.init_conv = mf.conv(sd, "init_conv", dt)
        self.head1 = mf.conv(sd, "flow_head.0", dt)
        self.head2 = mf.conv(sd, "flow_head.2", dt, rows=(0, 1))
        e = "update_block.encoder"
        corr = int(sd[f"{e}.convc2.weight"].shape[0])  # the correlation half of `conv`'s inputs
        flo = int(sd[f"{e}.convf2.weight"].shape[0])
        self.convc1_b = mf.put((2 * DIM,), dt, lambda: mf.np64(sd[f"{e}.convc1.bias"]))
        self.convc2 = mf.conv(sd, f"{e}.convc2", dt)
        self.conv_c = mf.conv(sd, f"{e}.conv", dt, inputs=lambda w: w[:, :corr], cin=corr)
        self.conv_f = mf.conv(sd, f"{e}.conv", dt, inputs=lambda w: w[:, corr:], cin=flo, bias=False)
        self.convf1 = mf.conv(sd, f"{e}.convf1", dt)
        self.convf2 = mf.conv(sd, f"{e}.convf2", dt)
        self.refine = mf.blocks(sd, "update_block.refine", dt)
        self.up1 = mf.conv(sd, "upsample_weight.0", dt)
        self.up2 = mf.conv(sd, "upsample_weight.2", dt, rows=mf.convex_rows(FACTOR), scale=0.25)
        self._corr: dict[tuple[int, int], mx.array] = {}
        self._step = stepped(self._update)

    def corr_term(self, h: int, w: int) -> mx.array:
        """`BasicMotionEncoder.corr_term`: `conv`'s correlation half and bias for a zero correlation, `[1, h, w, C]`."""
        if (h, w) not in self._corr:
            cor = mx.broadcast_to(mf.relu(self.convc1_b), (1, 5, 5, self.convc1_b.shape[0]))
            small = self.conv_c(mf.relu(self.convc2(cor)))
            iy = mx.array([0, 1] + [2] * (h - 4) + [3, 4])
            ix = mx.array([0, 1] + [2] * (w - 4) + [3, 4])
            term = small[:, iy][:, :, ix]
            mx.eval(term)
            self._corr = {(h, w): term}
        return self._corr[h, w]

    def _update(self, net: mx.array, flow: mx.array, context: mx.array, corr: mx.array) -> tuple[mx.array, mx.array]:
        fl = flow.astype(self.dtype)
        flo = mf.relu(self.convf2(mf.relu(self.convf1(fl))))
        motion = mf.relu(self.conv_f(flo) + corr)
        net = mf.refine(self.refine, net, [context, motion, fl])
        return net, flow + self.head2(mf.relu(self.head1(net))).astype(mx.float32)

    def _run(self, image1: mx.array, image2: mx.array) -> mx.array:
        x = mx.concatenate([image1, image2], axis=1).transpose(0, 2, 3, 1).astype(self.dtype)
        for f in self.encoder:
            x = f(x)
        cmap = self.init_conv(x)
        net, context = cmap[..., :DIM], cmap[..., DIM:]
        flow = self.head2(mf.relu(self.head1(net))).astype(mx.float32)
        _, h, w, _ = net.shape
        corr = self.corr_term(h, w)
        for _ in range(self.iters):
            net, flow = self._step(net, flow, context, corr)
        mask = self.up2(mf.relu(self.up1(net)))
        up = mf.convex_upsample(flow, mask, FACTOR)
        mx.eval(up)
        return up

    def __call__(self, image1: mx.array, image2: mx.array) -> mx.array:
        """Flow `[1, 2, H, W]` (float32) of a padded float32 pair `[1, 3, H, W]` in [-1, 1]."""
        mf.options(half_wino_f32=self.dtype == mx.float16)
        return steps(self._run, image1, image2)
