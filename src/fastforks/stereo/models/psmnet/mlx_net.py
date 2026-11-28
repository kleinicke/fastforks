"""PSMNet on Apple MLX, written in `mlx.core` operations (`MlxPSMNet(net, dtype)`): a padded pair -> disparity.

`PSMNet.forward` (eval) of the vendored network as one compiled function, channels last, the batch norms folded into
the convolutions (`fastforks.shared.mlx_stereo`; the feature trunk and the 3-D blocks are the ones GwcNet uses); every
weight-derived array is read through the conversion cache.

- the spatial pyramid pools the 1/4 features in 64 / 32 / 16 / 8 px cells (floor, as `AvgPool2d`), a 1 x 1 convolution
  each, bilinearly resized back; the concatenation volume is the two 32-channel feature maps shifted per disparity;
- three stacked hourglasses (with the previous one's `pre` / `post` skips), the three classifiers summed, and the
  head `upsample_soft_argmax` (trilinear upsampling, softmax and regression in one Metal kernel).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastforks._core.mlx import mx
from fastforks.kernels.graph_mlx.ops import upsample_soft_argmax
from fastforks.shared import mlx_stereo
from fastforks.shared.mlx_seg import resize
from fastforks.shared.mlx_stereo import Weights, concat_volume, conv2d, conv3d, deconv3d, relu, resnet

if TYPE_CHECKING:
    from .net.stackhourglass import PSMNet

__all__ = ["LAYOUT", "SOURCES", "MlxPSMNet"]

LAYOUT = "psmnet-v1"
SOURCES = (Path(__file__), Path(mlx_stereo.__file__))
_POOLS = (("branch4", 8), ("branch3", 16), ("branch2", 32), ("branch1", 64))  # concatenation order of upstream


def _pool(x: mx.array, k: int) -> mx.array:
    """`AvgPool2d(k, stride=k)` of NHWC `x` (partial cells at the border dropped)."""
    n, h, w, c = x.shape
    x = x[:, : h // k * k, : w // k * k]
    return x.reshape(n, h // k, k, w // k, k, c).mean(axis=(2, 4))


class MlxPSMNet:
    """The network `net` (eval, float32 weights) in `dtype` (the name): one compiled function per image size."""

    def __init__(self, net: PSMNet, dtype: str) -> None:
        """Copy the weights of `net` to MLX."""
        self.dtype = getattr(mx, dtype)
        w = Weights(self.dtype)
        self.depth = int(net.maxdisp) // 4
        self.max_disp = int(net.maxdisp)
        f = net.feature_extraction
        self.first = [w.dilated(f.firstconv[i]) for i in (0, 2, 4)]
        self.layers = [[w.residual(b) for b in getattr(f, f"layer{i}")] for i in (1, 2, 3, 4)]
        self.branches = [w.dilated(getattr(f, name)[1]) for name, _ in _POOLS]
        self.last = (w.dilated(f.lastconv[0]), (*w.conv(f.lastconv[2]), 1))
        self.dres0 = [w.cb(net.dres0[i]) for i in (0, 2)]
        self.dres1 = [w.cb(net.dres1[i]) for i in (0, 2)]
        self.hg = [w.hourglass3(h) for h in (net.dres2, net.dres3, net.dres4)]
        self.classif = [(w.cb(c[0]), w.conv(c[2])) for c in (net.classif1, net.classif2, net.classif3)]
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

    def _features(self, x: mx.array) -> mx.array:
        outs = resnet(x, self.first, self.layers)
        raw, skip = outs[1], outs[3]
        h, w = skip.shape[1:3]
        branches = [
            resize(relu(conv2d(_pool(skip, k), c)), (h, w)) for (_, k), c in zip(_POOLS, self.branches, strict=True)
        ]
        feature = mx.concatenate([raw, skip, *branches], axis=-1)
        return conv2d(relu(conv2d(feature, self.last[0])), self.last[1])

    @staticmethod
    def _hg(
        x: mx.array, pre_in: mx.array | None, post_in: mx.array | None, m: dict[str, Any]
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Upstream `hourglass(x, presqu, postsqu)`: the output, `pre` and `post`."""
        pre = conv3d(conv3d(x, m["conv1"]), m["conv2"], act=False)
        pre = relu(pre if post_in is None else pre + post_in)
        out = conv3d(conv3d(pre, m["conv3"]), m["conv4"])
        post = relu(deconv3d(out, m["conv5"]) + (pre if pre_in is None else pre_in))
        return deconv3d(post, m["conv6"]), pre, post

    def _classif(self, x: mx.array, k: int) -> mx.array:
        return conv3d(conv3d(x, self.classif[k][0]), self.classif[k][1], act=False)

    def _forward(self, left: mx.array, right: mx.array) -> mx.array:
        both = mx.concatenate([left, right], axis=0).transpose(0, 2, 3, 1).astype(self.dtype)
        feature = self._features(both)
        cost0 = conv3d(conv3d(concat_volume(feature[:1], feature[1:], self.depth), self.dres0[0]), self.dres0[1])
        cost0 = conv3d(conv3d(cost0, self.dres1[0]), self.dres1[1], act=False) + cost0
        out1, pre1, post1 = self._hg(cost0, None, None, self.hg[0])
        out1 = out1 + cost0
        out2, _, post2 = self._hg(out1, pre1, post1, self.hg[1])
        out2 = out2 + cost0
        out3 = self._hg(out2, pre1, post2, self.hg[2])[0] + cost0
        cost = self._classif(out3, 2) + (self._classif(out2, 1) + self._classif(out1, 0))
        return upsample_soft_argmax(cost.transpose(0, 4, 1, 2, 3), self.max_disp, left.shape[2], left.shape[3])

    def __call__(self, left: mx.array, right: mx.array) -> mx.array:
        """Disparity `[1, H, W]` (float32, lazy) of a padded float32 pair `[1, 3, H, W]`."""
        return self._compiled(left, right)
