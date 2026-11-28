"""GwcNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.layout import nchw
from fastforks._core.tracing import exporting_for_mlx
from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner
from fastforks.stereo._volume import stacked_volume

from .net.gwcnet import GwcNet
from .net.submodule import groupwise_correlation
from .regress import upsample_softmax_regress

if TYPE_CHECKING:
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    concat_volume: bool
    max_disp: int
    cuda_graphs: bool


def _skeleton(p: Params) -> GwcNet:
    return GwcNet(p.max_disp, use_concat_volume=p.concat_volume).eval()


def _net(ctx: Context) -> GwcNet:
    net = _skeleton(ctx.params)
    net.load_state_dict(ctx.state_dict(key="model"))
    return net.eval().to(ctx.device)


def cost_volume(net: GwcNet, fl: dict[str, torch.Tensor], fr: dict[str, torch.Tensor]) -> torch.Tensor:
    """Upstream's group-wise correlation (+ concatenation) volume `[B, G (+ 2C), D, H, W]`, built in place.

    Same values as `torch.cat((build_gwc_volume(...), build_concat_volume(...)), 1)`, without the two separate volumes
    alive next to the concatenated one.
    """
    # NCHW features: the group mean over a channels-last product is a strided reduction, 8x slower than contiguous
    gl, gr = nchw(fl["gwc_feature"]), nchw(fr["gwc_feature"])
    cl, cr = (fl["concat_feature"], fr["concat_feature"]) if net.use_concat_volume else (gl[:, :0], gr[:, :0])
    b, _, h, w = gl.shape
    g, c = net.num_groups, cl.shape[1]
    if exporting_for_mlx():
        nc = gl.shape[1]

        def fn(a: torch.Tensor, bb: torch.Tensor) -> torch.Tensor:
            parts = [groupwise_correlation(a[:, :nc], bb[:, :nc], g)]
            if c:
                parts += [a[:, nc:], bb[:, nc:]]
            return torch.cat(parts, dim=1)

        return stacked_volume(torch.cat((gl, cl), 1), torch.cat((gr, cr), 1), net.maxdisp // 4, fn)
    volume = gl.new_zeros([b, g + 2 * c, net.maxdisp // 4, h, w])
    for i in range(min(net.maxdisp // 4, w)):  # disparities >= width: no overlap, stays zero
        v = volume[:, :, i, :, i:]
        v[:, :g] = groupwise_correlation(gl[..., i:], gr[..., : w - i], g)
        if c:
            v[:, g : g + c] = cl[..., i:]
            v[:, g + c :] = cr[..., : w - i]
    return volume


def forward(net: GwcNet, left: torch.Tensor, right: torch.Tensor, *, fused: bool = False) -> torch.Tensor:
    """Upstream's test forward with the row-chunked regression tail; disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); the regression runs in float32. `fused` (kernels):
    the residual adds and their ReLUs run in the conv epilogues (`cnn_kernels.conv_residual`).
    """
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    fl, fr = net.feature_extraction(left), net.feature_extraction(right)
    volume = cost_volume(net, fl, fr)
    del fl, fr
    cost0 = net.dres0(volume)
    del volume
    if fused:
        from fastforks.shared.cnn_kernels import conv_residual, stereo_hourglass  # noqa: PLC0415 - kernels only

        cost0 = conv_residual(net.dres1, cost0, cost0)
        for hourglass in (net.dres2, net.dres3, net.dres4):
            cost0 = stereo_hourglass(hourglass, cost0)
    else:
        cost0 = net.dres1(cost0) + cost0
        for hourglass in (net.dres2, net.dres3, net.dres4):
            cost0 = hourglass(cost0)
    logits = net.classif3(cost0)
    del cost0
    return upsample_softmax_regress(logits, net.maxdisp, left.shape[2], left.shape[3])


class Portable(Runner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights, NHWC (CUDA), regression in float32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.torch_dtype)
        if ctx.device.type == "cuda":  # NHWC for the 2D feature network (the 3D cost-volume convs stay NCDHW)
            self.net.feature_extraction.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


class Kernels(Runner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + activation fused into one epilogue, fp16."""

    Params = Params
    _GRAPH_PIXELS = 384 * 640  # fixed: cap the pool at the measured launch-bound sizes

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import hacc_padded_deconvs, prepare_generic  # noqa: PLC0415 - kernel libraries

        self.net = _net(ctx)
        prepare_generic(self.net, ctx.device)
        hacc_padded_deconvs(self.net)  # the hourglasses' upsampling convs
        self._graphs = SmallShapeGraphs(
            self._disparity,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
            one_shape=True,
        )

    def _disparity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward(self.net, left, right, fused=True)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()


class _MlxPortable(Runner):
    """The network in MLX operations (`mlx_net`: no exported graph, so no trace on a first call), one pair per call."""

    batch = 1
    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the MLX network from the conversion cache (the checkpoint only on a miss); no PyTorch one kept."""
        super().__init__(ctx)
        from fastforks.shared import mlx_port  # noqa: PLC0415 - imports mlx

        from . import mlx_net  # noqa: PLC0415 - imports mlx

        run = dataclasses.replace(ctx, dtype=mlx_wrap.run_dtype(ctx))  # `mlx_wrap.port` builds this in float32
        self.net = mlx_port.recorded(
            run,
            mlx_net.LAYOUT,
            lambda net: mlx_net.MlxGwcNet(net, run.dtype),
            real=lambda: _net(ctx),
            skeleton=lambda: _skeleton(ctx.params),
            sources=mlx_net.SOURCES,
        )

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"disparity": to_torch(self.net(to_mx(left.float()), to_mx(right.float())))}


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable, whole="disparity")}
