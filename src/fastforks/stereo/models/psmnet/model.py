"""PSMNet runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from fastforks._core.mps import fence
from fastforks._core.tracing import exporting_for_mlx
from fastforks.kernels import cudnn_hacc
from fastforks.shared import mlx_wrap
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Forward, Runner
from fastforks.stereo._volume import concat_fn, stacked_volume

from .net.stackhourglass import PSMNet
from .regress import upsample_softmax_regress

if TYPE_CHECKING:
    from fastforks.shared.cnn_kernels import Patched
    from fastforks.stereo._runner import Context


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]` + variant `params`)."""

    max_disp: int
    cuda_graphs: bool


def _net(ctx: Context, *, fold: bool) -> PSMNet:
    """The network with the checkpoint, eval, on `ctx.device`; `fold`: BatchNorms folded into the convs (portable)."""
    net = PSMNet(ctx.params.max_disp)
    net.load_state_dict(ctx.state_dict(key="state_dict"))
    if fold:
        _fold_batchnorms(net.eval())
    return net.eval().to(ctx.device)


def _fold_batchnorms(module: nn.Module) -> None:
    """Fold every BatchNorm that directly follows a convolution in an `nn.Sequential` into it (float32, before casts).

    fastforks: no separate normalisation pass over the 3-D volumes (MPS: their outputs fragmented the 1 GB heaps).
    """
    for seq in module.modules():
        if not isinstance(seq, nn.Sequential):
            continue
        for i in range(len(seq) - 1):
            conv, bn = seq[i], seq[i + 1]
            if isinstance(conv, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)) and isinstance(
                bn, (nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                seq[i] = fuse_conv_bn_eval(conv, bn, transpose=isinstance(conv, nn.ConvTranspose3d))
                seq[i + 1] = nn.Identity()


def cost_volume(
    net: PSMNet, ref: torch.Tensor, tgt: torch.Tensor, memory_format: torch.memory_format = torch.contiguous_format
) -> torch.Tensor:
    """Upstream's concatenation volume `[B, 2C, D, H, W]` in the features' dtype (upstream: float32).

    `memory_format=torch.channels_last_3d` (kernels: the layout the `cudnn_hacc` convs take) writes the channels-last
    features without a transpose and spares the first conv its conversion.
    """
    if exporting_for_mlx():
        return stacked_volume(ref, tgt, net.maxdisp // 4, concat_fn)
    b, c, h, w = ref.shape
    shape = (b, 2 * c, net.maxdisp // 4, h, w)
    volume = torch.empty(shape, dtype=ref.dtype, device=ref.device, memory_format=memory_format).zero_()
    for i in range(min(net.maxdisp // 4, w)):  # disparities >= width: no overlap, stays zero
        volume[:, :c, i, :, i:] = ref[..., i:]
        volume[:, c:, i, :, i:] = tgt[..., : w - i]
    return volume


def _hourglass_kernels(
    hg: Patched, x: torch.Tensor, presqu: torch.Tensor | None, postsqu: torch.Tensor | None, skip: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upstream's `hourglass(x, presqu, postsqu)` plus `skip` on its output, every add / ReLU in a conv epilogue."""
    from fastforks.shared.cnn_kernels import ACT, conv_residual  # noqa: PLC0415 - kernels only

    relu = ACT["relu"]
    pre = conv_residual(hg.conv2, hg.conv1(x), postsqu, relu)
    post = conv_residual(hg.conv5, hg.conv4(hg.conv3(pre)), pre if presqu is None else presqu, relu)
    return conv_residual(hg.conv6, post, skip), pre, post


def _forward_kernels(net: PSMNet, cost0: torch.Tensor) -> torch.Tensor:
    """`forward`'s 3D stack from `dres1` on, the residual adds and ReLUs in the conv epilogues."""
    from fastforks.shared.cnn_kernels import conv_residual  # noqa: PLC0415 - kernels only

    cost0 = conv_residual(net.dres1, cost0, cost0)
    out1, pre1, post1 = _hourglass_kernels(net.dres2, cost0, None, None, cost0)
    out2, _, post2 = _hourglass_kernels(net.dres3, out1, pre1, post1, cost0)
    del post1
    cost1 = net.classif1(out1)
    del out1
    out3 = _hourglass_kernels(net.dres4, out2, pre1, post2, cost0)[0]
    del pre1, post2, cost0
    cost2 = net.classif2(out2)
    del out2
    return net.classif3(out3) + (cost2 + cost1)  # upstream's summation order


def forward(net: PSMNet, left: torch.Tensor, right: torch.Tensor, *, fused: bool = False) -> torch.Tensor:
    """Upstream's test forward (`pred3`) with the row-chunked regression tail; disparity `[B, H, W]` float32.

    Inputs are cast to the network's dtype (channels-last on CUDA); the regression runs in float32. `fused` (kernels):
    the residual adds and their ReLUs run in the conv epilogues (`cnn_kernels.conv_residual`).
    """
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    vf = torch.channels_last_3d if fused else torch.contiguous_format
    volume = cost_volume(net, net.feature_extraction(left), net.feature_extraction(right), vf)
    cost0 = net.dres0(volume)
    del volume
    if fused:
        return upsample_softmax_regress(_forward_kernels(net, cost0), net.maxdisp, left.shape[2], left.shape[3])
    cost0 = fence(net.dres1(cost0) + cost0)  # fastforks (MPS): the freed volume is reusable only once the GPU is done
    out1, pre1, post1 = net.dres2(cost0, None, None)
    out1 = out1 + cost0
    out2, pre2, post2 = net.dres3(out1, pre1, post1)
    del pre2, post1
    cost1 = net.classif1(out1)
    del out1
    out2 = out2 + cost0
    out3 = net.dres4(out2, pre1, post2)[0]
    del pre1, post2
    cost2 = net.classif2(out2)
    del out2
    out3 = out3 + cost0
    del cost0
    cost3 = net.classif3(out3) + (cost2 + cost1)  # upstream's summation order
    del out3, cost2, cost1
    fence(cost3)
    return upsample_softmax_regress(cost3, net.maxdisp, left.shape[2], left.shape[3])


class Portable(Runner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights, NHWC features (CUDA), regression in float32."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx, fold=True).to(ctx.torch_dtype)
        if ctx.device.type == "cuda":  # NHWC for the 2D feature network (the 3D cost-volume convs stay NCDHW)
            self.net.feature_extraction.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": self._run(left, right)}

    def _run(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """The network's forward pass (the part `portable_mlx` runs as one graph)."""
        return forward(self.net, left, right)

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


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
            lambda net: mlx_net.MlxPSMNet(net, run.dtype),
            real=lambda: _net(ctx, fold=False),
            skeleton=lambda: PSMNet(ctx.params.max_disp).eval(),
            sources=mlx_net.SOURCES,
        )

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        from fastforks._core.mlx import to_mx, to_torch  # noqa: PLC0415 - imports mlx

        return {"disparity": to_torch(self.net(to_mx(left.float()), to_mx(right.float())))}


class Kernels(Runner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + activation fused into one epilogue, fp16."""

    Params = Params
    _GRAPH_PIXELS = 384 * 640  # fixed: cap the pool at the measured launch-bound sizes

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared.cnn_kernels import prepare_generic  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx, fold=False)  # prepare_generic fuses each conv + BatchNorm run into the epilogue itself
        if not prepare_generic(self.net, ctx.device):
            msg = "kernels: no conv + BatchNorm run was fused (the module tree changed)"
            raise RuntimeError(msg)
        self._graphs = SmallShapeGraphs(
            self._disparity,
            pixels=self._GRAPH_PIXELS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
            one_shape=True,
            capture_warmup=cudnn_hacc.BUILD_AFTER > 0,
        )

    def _disparity(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return forward(self.net, left, right, fused=True)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded pair."""
        return {"disparity": self._graphs(left, right)}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        self._graphs.release()


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_MlxPortable, attrs=()),
}
# FASTFORKS_MLX_GRAPH=1 (diagnostics, tools/mlx_core_ab.py): the exported-graph port the hand-written network replaced
GRAPH_RUNNERS = {"portable_mlx": mlx_wrap.port(Portable, attrs=(), methods=("_run",))}
