"""GwcNet on Apple MLX, written in `mlx.core` operations (`MlxGwcNet(net, dtype)`): a padded pair -> disparity.

`GwcNet.forward` (eval) of the vendored network as one compiled function, channels last, the batch norms folded into
the convolutions (`fastforks.shared.mlx_stereo`); every weight-derived array is read through the conversion cache.

- the group-wise correlation volume is the shared `gwc_volume` kernel (40 groups); the concatenation volume of the
  `gc` variant is the two 12-channel feature maps shifted per disparity (zeros where the shift leaves the image);
- the transposed 3-D convolutions run as one small convolution per output phase (`mlx_graph`'s);
- the head is `upsample_soft_argmax`: the trilinear upsampling, the softmax and the regression in one Metal kernel.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import gwc_volume, upsample_soft_argmax
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_graph import _conv_transpose3d_phases
from fastforks.shared.mlx_stereo import Weights, relu

if TYPE_CHECKING:
    from .net.gwcnet import GwcNet

__all__ = ["LAYOUT", "SOURCES", "MlxGwcNet"]

LAYOUT = "gwcnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))

_Conv = tuple[mx.array, mx.array, int, int, int]  # weight, bias, stride, padding, dilation


def _seq(w: Weights, seq: Any) -> _Conv:  # noqa: ANN401 - a vendored `convbn(...)` Sequential: conv, batch norm
    c = w.conv(seq[0], seq[1])
    return (*c, int(seq[0].dilation[0]))


def _volume_shifted(left: mx.array, right: mx.array, depth: int) -> mx.array:
    """The concatenation volume `[1, D, H, W, 2C]` of `[1, H, W, C]` features (upstream `build_concat_volume`)."""
    wd = left.shape[2]
    rp = mx.pad(right, [(0, 0), (0, 0), (depth, 0), (0, 0)])
    cols = mx.arange(wd)[None, :, None]
    zero = mx.zeros((), left.dtype)
    slabs = []
    for i in range(depth):
        lt = mx.where(cols >= i, left[0], zero)
        rt = rp[0, :, depth - i : depth - i + wd]  # zeros where the shifted column leaves the image (the padding)
        slabs.append(mx.concatenate([lt, rt], axis=-1))
    return mx.stack(slabs)[None]


class MlxGwcNet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: GwcNet, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        w = Weights(self.dtype)
        self.depth = int(net.maxdisp) // 4
        self.groups = int(net.num_groups)
        self.max_disp = int(net.maxdisp)
        self.concat = bool(net.use_concat_volume)
        f = net.feature_extraction
        self.first = [_seq(w, s) for s in (f.firstconv[0], f.firstconv[2], f.firstconv[4])]
        self.layers = [
            [self._block(w, b) for b in getattr(f, name)] for name in ("layer1", "layer2", "layer3", "layer4")
        ]
        if self.concat:
            self.last = (_seq(w, f.lastconv[0]), (*w.conv(f.lastconv[2]), 1))
        self.dres0 = [self._c3(w, s) for s in (net.dres0[0], net.dres0[2])]
        self.dres1 = [self._c3(w, s) for s in (net.dres1[0], net.dres1[2])]
        self.hg = [self._hourglass(w, h) for h in (net.dres2, net.dres3, net.dres4)]
        self.classif = (self._c3(w, net.classif3[0]), w.conv(net.classif3[2]))
        mx.eval(self._arrays())
        self._compiled = mx.compile(self._forward)

    # weights ---------------------------------------------------------------------------------------------------------

    def _arrays(self) -> list[mx.array]:
        found: list[mx.array] = []

        def walk(v: object) -> None:
            if isinstance(v, mx.array):
                found.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)

        walk(list(vars(self).values()))
        return found

    @staticmethod
    def _c3(w: Weights, seq: Any) -> mlx_stereo.Conv:  # noqa: ANN401
        return w.conv(seq[0], seq[1])

    def _block(self, w: Weights, b: Any) -> tuple[Any, ...]:  # noqa: ANN401
        """`BasicBlock`: (conv1, conv2, downsample or None)."""
        down = None if b.downsample is None else (*w.conv(b.downsample[0], b.downsample[1]), 1)
        return _seq(w, b.conv1[0]), _seq(w, b.conv2), down

    def _hourglass(self, w: Weights, h: Any) -> dict[str, Any]:  # noqa: ANN401
        out: dict[str, Any] = {k: self._c3(w, getattr(h, k)[0]) for k in ("conv1", "conv2", "conv3", "conv4")}
        for k in ("conv5", "conv6"):
            seq = getattr(h, k)
            m = type("M", (), {"conv": seq[0], "bn": seq[1], "use_bn": True, "relu": False})
            out[k] = w.deconv3(m)
        out["redir1"] = self._c3(w, h.redir1)
        out["redir2"] = self._c3(w, h.redir2)
        return out

    # layers ----------------------------------------------------------------------------------------------------------

    @staticmethod
    def _c(x: mx.array, c: _Conv) -> mx.array:
        w, b, stride, padding, dilation = c
        return mx.conv2d(x, w, stride=stride, padding=padding, dilation=dilation) + b

    @staticmethod
    def _v(x: mx.array, c: mlx_stereo.Conv, *, act: bool = True) -> mx.array:
        w, b, stride, padding = c
        y = mx.conv3d(x, w, stride=stride, padding=padding) + b
        return relu(y) if act else y

    @staticmethod
    def _up(x: mx.array, up: tuple[mx.array, mx.array, bool]) -> mx.array:
        w, b, _ = up
        y = _conv_transpose3d_phases(x, w, (2, 2, 2), (1, 1, 1), (1, 1, 1))
        if y is None:  # pragma: no cover - the pair is padded to 32s, every volume level halves exactly
            msg = f"MlxGwcNet: the cost volume {tuple(x.shape[1:4])} does not halve evenly"
            raise ValueError(msg)
        return y + b

    def _hg(self, x: mx.array, m: dict[str, Any]) -> mx.array:
        low = self._v(self._v(x, m["conv1"]), m["conv2"])
        bottom = self._v(self._v(low, m["conv3"]), m["conv4"])
        up = relu(self._up(bottom, m["conv5"]) + self._v(low, m["redir2"], act=False))
        return relu(self._up(up, m["conv6"]) + self._v(x, m["redir1"], act=False))

    def _features(self, x: mx.array) -> tuple[mx.array, mx.array | None]:
        for c in self.first:
            x = relu(self._c(x, c))
        layers = []
        for blocks in self.layers:
            for c1, c2, down in blocks:
                y = self._c(relu(self._c(x, c1)), c2)
                x = y + (x if down is None else self._c(x, down))
            layers.append(x)
        gwc = mx.concatenate(layers[1:], axis=-1)
        if not self.concat:
            return gwc, None
        return gwc, self._c(relu(self._c(gwc, self.last[0])), self.last[1])

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        gwc, cat = self._features(both)
        volume = gwc_volume(gwc[:1], gwc[1:], self.depth, self.groups)
        if cat is not None:
            volume = mx.concatenate([volume, _volume_shifted(cat[:1], cat[1:], self.depth)], axis=-1)
        cost = self._v(self._v(volume, self.dres0[0]), self.dres0[1])
        cost = self._v(self._v(cost, self.dres1[0]), self.dres1[1], act=False) + cost
        for h in self.hg:
            cost = self._hg(cost, h)
        cost = self._v(self._v(cost, self.classif[0]), self.classif[1], act=False)  # [1, D, h, w, 1]
        height, width = left.shape[2], left.shape[3]
        return upsample_soft_argmax(cost.transpose(0, 4, 1, 2, 3), self.max_disp, height, width)

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        return self._compiled(left, right)
