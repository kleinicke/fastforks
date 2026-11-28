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
from fastforks.shared.mlx_stereo import Weights, concat_volume, conv2d, conv3d, deconv3d, relu, resnet

if TYPE_CHECKING:
    from .net.gwcnet import GwcNet

__all__ = ["LAYOUT", "SOURCES", "MlxGwcNet"]

LAYOUT = "gwcnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))


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
        self.first = [w.dilated(f.firstconv[i]) for i in (0, 2, 4)]
        self.layers = [[w.residual(b) for b in getattr(f, f"layer{i}")] for i in (1, 2, 3, 4)]
        if self.concat:
            self.last = (w.dilated(f.lastconv[0]), (*w.conv(f.lastconv[2]), 1))
        self.dres0 = [w.cb(net.dres0[i]) for i in (0, 2)]
        self.dres1 = [w.cb(net.dres1[i]) for i in (0, 2)]
        self.hg = [w.hourglass3(h) for h in (net.dres2, net.dres3, net.dres4)]
        self.classif = (w.cb(net.classif3[0]), w.conv(net.classif3[2]))
        mx.eval(self._arrays())
        self._compiled = mx.compile(self._forward)

    def _arrays(self) -> list[mx.array]:
        found: list[mx.array] = []

        def walk(v: object) -> None:
            if isinstance(v, mx.array):
                found.append(v)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                walk(list(v.values()))

        walk(list(vars(self).values()))
        return found

    def _hg(self, x: mx.array, m: dict[str, Any]) -> mx.array:
        low = conv3d(conv3d(x, m["conv1"]), m["conv2"])
        bottom = conv3d(conv3d(low, m["conv3"]), m["conv4"])
        up = relu(deconv3d(bottom, m["conv5"]) + conv3d(low, m["redir2"], act=False))
        return relu(deconv3d(up, m["conv6"]) + conv3d(x, m["redir1"], act=False))

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        gwc = mx.concatenate(resnet(both, self.first, self.layers)[1:], axis=-1)
        volume = gwc_volume(gwc[:1], gwc[1:], self.depth, self.groups)
        if self.concat:
            cat = conv2d(relu(conv2d(gwc, self.last[0])), self.last[1])
            volume = mx.concatenate([volume, concat_volume(cat[:1], cat[1:], self.depth)], axis=-1)
        cost = conv3d(conv3d(volume, self.dres0[0]), self.dres0[1])
        cost = conv3d(conv3d(cost, self.dres1[0]), self.dres1[1], act=False) + cost
        for h in self.hg:
            cost = self._hg(cost, h)
        cost = conv3d(conv3d(cost, self.classif[0]), self.classif[1], act=False)  # [1, D, h, w, 1]
        return upsample_soft_argmax(cost.transpose(0, 4, 1, 2, 3), self.max_disp, left.shape[2], left.shape[3])

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]` (multiples of 32)."""
        return self._compiled(left, right)
