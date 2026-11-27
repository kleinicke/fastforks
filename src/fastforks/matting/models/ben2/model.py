"""BEN2 runners: weights -> network and the portable / kernels forward passes.

Inference as the authors' `BEN_Base.inference`: the image resized to `size x size` (upstream: PIL Lanczos on the
8-bit image; here bicubic, antialiased, on the float image), ImageNet normalisation, the network's mask, resized back
to the image size (bilinear) and min-max normalised per image (upstream then rounds it to 8 bits).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16 weights and activations (the authors run fp16 autocast
on CUDA), SDPA attention (Swin windows and the decoder's cross-attention). `kernels`: fp16 channels-last; every conv
runs as a cuDNN fp16-accumulate graph (`cudnn_hacc`) with an fp32 bias epilogue, the GEMMs accumulate in fp16 and the
InstanceNorms run as one pass each (`shared.cnn_kernels`); the two nearest 2x upsamplings and the convs after them
run as stride-2 transposed convs (`up_deconv`); the fixed-size network replays a CUDA graph (one or two images per
call) when `cuda_graphs` is enabled. It is opt-in because the modest replay benefit at 1024x1024 comes with a
larger reserved memory footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from fastforks._core.half import fp16_accumulate
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap, swin_v1
from fastforks.shared.cnn_kernels import CL, prepare_generic, up_deconv
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net.ben2 import BEN_Base

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        size: Network input side (square).
        cuda_graphs: Replay the fixed-size network in CUDA kernels mode.
    """

    size: int
    cuda_graphs: bool


def _cpu_net(ctx: Context) -> BEN_Base:
    net = BEN_Base()  # the skeleton of the MLX conversion cache too
    net.load_state_dict(ctx.state_dict())
    return net.eval()


# the Swin windows (144 tokens, a bias mask) on the memory-efficient kernel, the decoder's attention on the runner's
# cuDNN-first order (1080p, RTX 5060 Ti, Windows: windows 9.3 -> 7.6 ms; the decoder on memory-efficient 8.9 -> 22 ms)
_SWIN_SDPA = "efficient,cudnn,math"
_TINY = 1e-8  # fixed: our floor of the normalisation divisors (constant image / map: not 0 / 0)


class _Matting(Runner):
    """Resize, normalise, network, sigmoid, resize back, min-max."""

    Params = Params
    net: BEN_Base
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bicubic", align_corners=False, antialias=True)
        y = self._net(scale(x, self.ctx.io).to(self.dtype)).float().sigmoid()
        y = F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        lo, hi = y.amin((1, 2), keepdim=True), y.amax((1, 2), keepdim=True)
        return {"alpha": (y - lo) / (hi - lo).clamp_min(_TINY)}

    def module(self) -> torch.nn.Module:
        """The network with its pre- and post-processing: `[B, 3, H, W]` in `[0, 1]` -> alpha `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image, None)["alpha"])


class Portable(_Matting):
    """Upstream's modules in plain PyTorch, float32 or float16 (SDPA attention)."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.dtype = getattr(torch, ctx.dtype)
        self.net = _cpu_net(ctx).to(ctx.device, self.dtype)


class Kernels(_Matting):
    """fp16 channels-last network, cuDNN fp16-accumulate convs and GEMMs, InstanceNorm kernels, a CUDA graph."""

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert its convs and norms."""
        super().__init__(ctx)
        net = _cpu_net(ctx).to(ctx.device)
        # the two nearest 2x + conv stages as transposed convs (1024 input, RTX 5060 Ti: 88.2 -> 81.7 ms)
        net.up_deconv1, net.up_deconv2 = up_deconv(net.upsample1[0]), up_deconv(net.upsample2[0])
        prepare_generic(net, ctx.device)  # no BatchNorm to fuse: convs + InstanceNorms
        swin_v1.fast(net.backbone, sdpa_auto=_SWIN_SDPA)
        self.dtype = torch.float16
        self.net = net
        self.graphs = SmallShapeGraphs(self._eager, pixels=2**21, enabled=ctx.params.cuda_graphs)

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.contiguous(memory_format=CL))

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():  # the Swin and decoder GEMMs (1080p 102 -> 96 ms, golden error unchanged)
            return self.graphs(x)

    def release_graphs(self) -> None:
        """Release captured activation buffers."""
        self.graphs.release()


class _MlxPortable(_Matting):
    """Apple MLX: the network written in MLX operations (`mlx_net`), the pre- and post-processing on MLX too."""

    module = Runner.module  # the MLX network is not exportable
    dtypes = ("float32", "float16")
    CACHE_MB: ClassVar[dict[str, int]] = {}  # the buffer cache this mode asks for, by variant ('*': all)

    def __init__(self, ctx: Context) -> None:
        """The MLX network from the conversion cache (the first load converts the checkpoint's weights)."""
        Runner.__init__(self, ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_port.recorded(
            ctx,
            mlx_net.LAYOUT,
            lambda n: mlx_net.network(n, ctx.dtype),
            real=lambda: _cpu_net(ctx),
            skeleton=BEN_Base,
            sources=mlx_net.SOURCES,
        )
        if mb := self.CACHE_MB.get(ctx.variant.id, self.CACHE_MB.get("*", 0)):  # held while this runner lives
            from fastforks._core.mlx import request_cache  # noqa: PLC0415 - MLX only

            request_cache(self, mb)

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        from fastforks._core.mlx import mx, to_mx, to_torch  # noqa: PLC0415 - imports mlx
        from fastforks.shared import mlx_matting as mm  # noqa: PLC0415 - imports mlx

        # `_Matting.__call__` on the GPU: antialiased bicubic resize, normalise, net, sigmoid, bilinear back, min-max
        x = to_mx(image.float())
        s, (h, w) = self.ctx.params.size, x.shape[2:]
        x = mm.normalise(mm.resize_aa(x, (s, s), cubic=True), self.ctx.io).transpose(0, 2, 3, 1)
        y = mx.concatenate([mx.sigmoid(self.mlx(x[i : i + 1])) for i in range(x.shape[0])])
        y = mm.resize(y.transpose(0, 3, 1, 2), (h, w))
        return {"alpha": to_torch(mm.min_max(y, (2, 3))[:, 0])}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": _MlxPortable,
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable)}
