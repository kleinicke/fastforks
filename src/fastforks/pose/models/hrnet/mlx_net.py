"""HRNet (top-down pose) on Apple MLX, hand-written (`fastforks.shared.cspnext.mlx_net`'s convs): NHWC, the run dtype.

The network of `model.Portable.forward`: normalised crops `[P, 3, h, w]` float32 -> float32 heatmaps
`[P, K, h / 4, w / 4]`. The convs (BatchNorm folded) dispatch as the graph translator's (`graph="portable"`: 1 x 1
ones as GEMMs, the others through `mlx_graph._conv2d`); bias, ReLU and residual fuse into one compiled pass per piece.
The stem convs are one step, each bottleneck of the first stage another, each transition, each branch of an `HRModule`
and its exchange (1 x 1 conv + nearest upsampling upwards, stride-2 3 x 3 convs downwards) another
(`cspnext.mlx_net.stepped`).
"""

# ruff: noqa: ANN401 - the network's torch modules

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.shared.cspnext import mlx_net as csp

from . import net as tnet

if TYPE_CHECKING:
    from collections.abc import Callable

    from .net import HRNet

    Layer = Callable[[mx.array], mx.array]

LAYOUT = "hrnet-pose-v1"
SOURCES = (Path(__file__), Path(tnet.__file__), Path(csp.__file__))
GRAPH = "portable"  # the translator's conv dispatch (`csp.conv`)


def _relu(x: mx.array) -> mx.array:
    return mx.maximum(x, 0)


def _up(x: mx.array, s: int) -> mx.array:
    """Nearest `s`x upsampling of an NHWC map."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None], (n, h, s, w, s, c)).reshape(n, h * s, w * s, c)


def _chain(seq: Any, dtype: mx.Dtype) -> Layer:
    """A `Sequential` of convs and ReLUs (nested ones flattened, `Identity` skipped)."""
    ops: list[Layer] = []

    def walk(m: Any) -> None:
        name = type(m).__name__
        if name == "Conv2d":
            ops.append(csp.conv(m, dtype, GRAPH))
        elif name == "ReLU":
            ops.append(_relu)
        elif name == "Sequential":
            for c in m.children():
                walk(c)
        elif name != "Identity":
            msg = f"HRNet MLX: unexpected module {name}"
            raise NotImplementedError(msg)

    walk(seq)
    return csp.chain(*ops)


def _block(blk: Any, dtype: mx.Dtype) -> Layer:
    """`BasicBlock` / `Bottleneck`: the convs with ReLUs between, the shortcut, the residual ReLU."""
    names = ("conv1", "conv2", "conv3") if hasattr(blk, "conv3") else ("conv1", "conv2")
    convs = [csp.conv(getattr(blk, n), dtype, GRAPH) for n in names]
    ds = getattr(blk, "downsample", None)  # `BasicBlock`: an identity shortcut
    down = None if ds is None or type(ds).__name__ == "Identity" else _chain(ds, dtype)

    def apply(x: mx.array) -> mx.array:
        idt = x if down is None else down(x)
        for c in convs[:-1]:
            x = _relu(c(x))
        return _relu(convs[-1](x) + idt)

    return apply


def _module(m: Any, dtype: mx.Dtype) -> Callable[[list[mx.array]], list[mx.array]]:
    """`HRModule.forward`: per-branch blocks (a step each), then the exchange (a step), ReLU of the resampled sums."""
    branches = [csp.stepped(csp.chain(*(_block(b, dtype) for b in br))) for br in m.branches]
    fuse: list[list[Any]] = []
    for i, row in enumerate(m.fuse_layers):
        terms: list[Any] = []
        for j, f in enumerate(row):
            if j == i:
                terms.append(None)
            elif j > i:  # 1 x 1 conv + nearest 2^(j - i)
                terms.append((csp.conv(f[0], dtype, GRAPH), 1 << (j - i)))
            else:
                terms.append(_chain(f, dtype))
        fuse.append(terms)

    def exchange(xs: list[mx.array]) -> list[mx.array]:
        out = []
        for terms in fuse:
            y = None
            for j, f in enumerate(terms):
                if f is None:
                    t = xs[j]
                elif isinstance(f, tuple):
                    t = _up(f[0](xs[j]), f[1])
                else:
                    t = f(xs[j])
                y = t if y is None else y + t
            out.append(_relu(y))
        return out

    fused = csp.stepped(exchange)
    return lambda xs: fused([b(x) for b, x in zip(branches, xs, strict=True)])


class Network:
    """`HRNet.forward`: normalised crops `[P, 3, h, w]` float32 -> heatmaps `[P, K, h / 4, w / 4]` float32."""

    def __init__(self, net: HRNet, dtype: str) -> None:
        """The arrays of `net` (float32, or the meta skeleton on a cache hit) in `dtype`."""
        dt = getattr(mx, dtype)
        self.dtype = dt
        c1, c2 = csp.conv(net.conv1, dt, GRAPH), csp.conv(net.conv2, dt, GRAPH)
        self.stem = [csp.stepped(lambda x: _relu(c2(_relu(c1(x.transpose(0, 2, 3, 1).astype(dt))))))]
        self.stem += [csp.stepped(_block(b, dt)) for b in net.layer1]  # a step each: the widest maps (256 at 1 / 4)
        self.stages: list[tuple[list[Layer], list[Callable[..., Any]]]] = []
        for n in range(1, len(tnet.MODULES) + 1):
            trans = [_chain(t, dt) for t in getattr(net, f"transition{n}")]
            mods = [_module(m, dt) for m in getattr(net, f"stage{n + 1}")]
            self.stages.append((trans, mods))
        head = csp.conv(net.head, dt, GRAPH)
        self.head = csp.stepped(lambda y: head(y).astype(mx.float32).transpose(0, 3, 1, 2))

    def forward(self, x: mx.array) -> mx.array:
        """The network on `[P, 3, h, w]` float32 (the steps)."""
        for step in self.stem:
            x = step(x)
        ys = [x]
        for trans, mods in self.stages:
            # a branch that exists passes through (stage 2: the stem's map for both); the new one reads the coarsest
            ys = [t(ys[min(b, len(ys) - 1)]) for b, t in enumerate(trans)]
            for m in mods:
                ys = m(ys)
        return self.head(ys[0])

    def __call__(self, x: mx.array) -> mx.array:
        """`forward` with its steps, evaluated."""
        csp.options(GRAPH)
        out = csp.steps(self.forward, x)
        mx.eval(out)
        return out


def network(net: Any, dtype: str) -> Network:
    """The MLX network of an `HRNet` (float32 or meta skeleton) in `dtype`."""
    return Network(net, dtype)
