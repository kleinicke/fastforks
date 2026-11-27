"""IS-Net runners: weights -> network and the portable / kernels forward passes.

Inference as upstream's `IS-Net/Inference.py`: the image resized to `size x size` (bilinear, no anti-aliasing; upstream
`F.upsample` on the uint8 tensor, so it truncates to 8 bits; here it stays float), normalised with mean 0.5 and std 1,
the first side output `d1` (sigmoid), resized back to the image size (bilinear) and min-max normalised.

`portable`: upstream's modules in plain PyTorch, fp32 or fp16; in fp16 each BatchNorm is folded into its conv (as in
U²-Net, unnormalised conv outputs can exceed the float16 range). `kernels`: fp16 channels-last; each conv ->
BatchNorm -> ReLU block (`REBNCONV`) is one cuDNN fp16-accumulate conv (`cudnn_hacc`), the BatchNorm folded into it,
with the bias and the ReLU in an fp32 epilogue (`shared.cnn_kernels`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, fold_bn, prepare_generic

from .net import ISNetDIS
from .net.isnet import REBNCONV

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        size: Network input side (square).
    """

    size: int


def _cpu_net(ctx: Context, *, fold: bool) -> torch.nn.Module:
    net = ISNetDIS(3, 1)
    net.load_state_dict(ctx.state_dict())
    if fold:
        for m in net.modules():
            if isinstance(m, REBNCONV):
                fold_bn(m.conv_s1, m.bn_s1)
    return net.eval()


_TINY = 1e-8  # fixed: our floor of the normalisation divisors (constant image / map: not 0 / 0)


class _Matting(Runner):
    """Resize, normalise, network (`d1`), resize back, min-max."""

    Params = Params
    net: torch.nn.Module
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0][0]

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False)
        y = self._net(scale(x, self.ctx.io).to(self.dtype)).float()
        y = F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
        return {"alpha": (y - lo) / (hi - lo).clamp_min(_TINY)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


class Portable(_Matting):
    """Upstream's modules in plain PyTorch, float32 or float16."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx, fold=self.dtype == torch.float16).to(ctx.device, self.dtype)
        if ctx.device.type == "cuda" and self.dtype == torch.float16:
            # channels-last weights: cuDNN's fp16 kernels skip the NCHW<->NHWC conversion around each conv
            # (1080p 32.9 -> 26.6 ms, peak 571 -> 506 MB RTX 5060 Ti, Windows); float32 stays NCHW
            self.net.to(memory_format=torch.channels_last)


def _rebnconv(m: REBNCONV, x: torch.Tensor) -> torch.Tensor:
    """`relu(bn(conv(x)))` as one conv (the BatchNorm folded in) with an fp32 bias + ReLU epilogue."""
    return conv_ep(m.conv_s1, x, act=ACT["relu"])


class Kernels(_Matting):
    """fp16 channels-last network, cuDNN fp16-accumulate convs with BatchNorm + ReLU epilogues."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold each BatchNorm into its conv and the ReLU into the conv's epilogue."""
        super().__init__(ctx)
        net = _cpu_net(ctx, fold=True).to(ctx.device)
        for m in net.modules():
            if isinstance(m, REBNCONV):
                override(m, _rebnconv)
        prepare_generic(net, ctx.device)
        self.dtype = torch.float16
        self.net = net

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.contiguous(memory_format=CL))[0][0]


class _MlxPortable(_Matting):
    """Apple MLX: the network written in MLX operations (`mlx_net`), the pre- and post-processing on MLX too."""

    module = Runner.module  # the MLX network is not exportable
    dtypes = ("float32", "float16")
    KERNELS = False
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        kernels = self.KERNELS and ctx.dtype == "float16"  # float32 runs as portable (as the graph port's kernels)
        net = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda n: mlx_net.network(n, ctx.dtype, kernels=kernels),
            real=lambda: _cpu_net(ctx, fold=False),
            skeleton=lambda: ISNetDIS(3, 1),
            sources=mlx_net.SOURCES,
        )
        self.mlx = net
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx
        from fastforks.shared import mlx_matting as mm  # noqa: PLC0415 - imports mlx

        # `_Matting.__call__` on the GPU: resize, normalise, network, resize back, min-max
        x = to_mx(image.float())
        s, (h, w) = self.ctx.params.size, x.shape[2:]
        y = self.mlx(mm.normalise(mm.resize(x, (s, s)), self.ctx.io))
        return {"alpha": to_torch(mm.min_max(mm.resize(y, (h, w))[:, 0], (1, 2)))}


class _MlxKernels(_MlxPortable):
    """Apple MLX, float16: the 3 x 3 convs through `conv_mlx`'s Winograd kernel, the stem its direct conv."""

    KERNELS = True


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
    "kernels_mlx": _MlxKernels,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph ports the hand-written network replaced
GRAPH_RUNNERS = {
    "portable_mlx": mlx_wrap.port(Portable),
    "kernels_mlx": mlx_wrap.port(Portable, kernels_fp16_only=True),
}
