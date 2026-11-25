"""U²-Net runners: weights -> network and the portable / kernels forward passes.

Inference as upstream's `u2net_test.py` (and rembg): the image resized to `size x size` (upstream: scikit-image
`resize`, bilinear with Gaussian anti-aliasing; here bilinear, antialiased), divided by its maximum (`ToTensorLab`),
ImageNet normalisation, the fused sigmoid map `d0`, min-max normalised (`normPRED`) and resized back to the image size
(bilinear; upstream: PIL on the 8-bit map).

`portable`: upstream's modules in plain PyTorch, fp32 or fp16; in fp16 each BatchNorm is folded into its conv (some
unnormalised conv outputs exceed the float16 range, e.g. 87 548 in `stage2d`). `kernels`: fp16 channels-last; each
conv -> BatchNorm -> ReLU block (`REBNCONV`) is one cuDNN fp16-accumulate conv (`cudnn_hacc`), the BatchNorm folded
into it, with the bias and the ReLU in an fp32 epilogue (`shared.cnn_kernels`).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.patch import override
from fastforks.kernels import cudnn_hacc
from fastforks.kernels.resize_cat import resize_cat
from fastforks.matting._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, fold_bn, prepare_generic
from fastforks.shared.cuda_graphs import SmallShapeGraphs

from .net import U2NET, U2NETP
from .net.u2net import REBNCONV

if TYPE_CHECKING:
    from fastforks.matting._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params`.

    Attributes:
        size: Network input side (square).
        arch: `"u2net"` (full, 44 M parameters) or `"u2netp"` (small, 1.1 M).
    """

    size: int
    arch: str
    cuda_graphs: bool = True


def _cpu_net(ctx: Context, *, fold: bool) -> torch.nn.Module:
    net = {"u2net": U2NET, "u2netp": U2NETP}[ctx.params.arch](3, 1)
    net.load_state_dict(ctx.state_dict())
    if fold:
        for m in net.modules():
            if isinstance(m, REBNCONV):
                fold_bn(m.conv_s1, m.bn_s1)
    return net.eval()


_TINY = 1e-8  # fixed: our floor of the normalisation divisors (constant image / map: not 0 / 0)


class _Matting(Runner):
    """Resize, divide by the maximum, normalise, network (`d0`), min-max, resize back."""

    Params = Params
    batch = None
    # network pixels (images x size^2) per call: the eager call is launch bound (RTX 5060 Ti, Windows, fp16, 8 images:
    # u2netp 14.1 -> 3.7 ms per image, u2net 14.1 -> 6.2; kernels 2.9 -> 2.5, 4.9 -> 3.9)
    net_pixels = 1_300_000
    net: torch.nn.Module
    dtype: torch.dtype

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)[0]

    def _nets(self, x: torch.Tensor) -> torch.Tensor:
        step = max(1, self.net_pixels // (x.shape[-2] * x.shape[-1]))
        return torch.cat([self._net(x[i : i + step]) for i in range(0, len(x), step)])

    def __call__(self, image: torch.Tensor, trimap: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Alpha `[B, H, W]` of `[B, 3, H, W]` RGB in `[0, 1]`."""
        s = self.ctx.params.size
        x = F.interpolate(image, (s, s), mode="bilinear", align_corners=False, antialias=True)
        x = x / x.amax((1, 2, 3), keepdim=True).clamp_min(_TINY)  # a black image stays black, not 0 / 0
        y = self._nets(scale(x, self.ctx.io).to(self.dtype)).float()
        lo, hi = y.amin((1, 2, 3), keepdim=True), y.amax((1, 2, 3), keepdim=True)
        y = (y - lo) / (hi - lo).clamp_min(_TINY)  # a constant map stays 0, not 0 / 0
        return {"alpha": F.interpolate(y, image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]}

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
        self.net = _cpu_net(ctx, fold=self.dtype == torch.float16)
        if ctx.device.type == "cuda" and self.dtype == torch.float16:
            # The affine and running statistics are already folded into conv_s1.
            for block in self.net.modules():
                if isinstance(block, REBNCONV):
                    block.set_submodule("bn_s1", torch.nn.Identity())
            self.net.to(memory_format=CL)
        self.net.to(ctx.device, self.dtype)


def _rebnconv(m: REBNCONV, x: torch.Tensor) -> torch.Tensor:
    """`relu(bn(conv(x)))` as one conv (the BatchNorm folded in) with an fp32 bias + ReLU epilogue."""
    return conv_ep(m.conv_s1, x, act=ACT["relu"])


class _DecoderTracer(torch.fx.Tracer):
    """Keep each fused convolution as a leaf and expose the decoder's skip connections."""

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """Preserve REBNCONV instances for the kernel epilogue."""
        return isinstance(m, REBNCONV) or super().is_leaf_module(m, module_qualified_name)


def _resized_for(up: object, skip: object) -> bool:
    if not isinstance(up, torch.fx.Node) or up.target is not F.interpolate:
        return False
    size = up.kwargs.get("size")
    if not isinstance(size, torch.fx.Node) or size.target is not operator.getitem or size.args[1] != slice(2, None):
        return False
    shape = size.args[0]
    return bool(
        isinstance(shape, torch.fx.Node)
        and shape.target is getattr
        and shape.args == (skip, "shape")
        and up.kwargs["mode"] == "bilinear"
        and not up.kwargs["align_corners"]
        and not up.kwargs["antialias"]
    )


def _alpha_only(net: torch.nn.Module) -> torch.fx.GraphModule:
    """Fuse resize followed by concatenation and discard the six auxiliary side-map sigmoids."""
    graph = _DecoderTracer().trace(net)
    for node in graph.nodes:
        if node.op == "output":
            node.args = (node.args[0][0],)  # U2NET and U2NETP return seven maps
        elif node.target == torch.cat:
            operands = node.args[0]
            if isinstance(operands, (tuple, list)) and len(operands) == 2:  # noqa: PLR2004 - resized map, skip
                up, skip = operands
                if isinstance(up, torch.fx.Node) and _resized_for(up, skip):
                    node.target = resize_cat
                    node.args = (up.args[0], skip)
    traced = torch.fx.GraphModule(net, graph)
    graph.eliminate_dead_code()
    graph.lint()
    traced.recompile()
    return traced


class Kernels(_Matting):
    """fp16 channels-last network, cuDNN fp16-accumulate convs with BatchNorm + ReLU epilogues."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold each BatchNorm into its conv and the ReLU into the conv's epilogue."""
        super().__init__(ctx)
        net = _alpha_only(_cpu_net(ctx, fold=True)).to(ctx.device)
        for m in net.modules():
            if isinstance(m, REBNCONV):
                override(m, _rebnconv)
        prepare_generic(net, ctx.device)
        self.dtype = torch.float16
        self.net = net
        self.graphs = SmallShapeGraphs(
            self.net,
            pixels=self.net_pixels,
            graphs=2,  # a batch's full chunks and its last one
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Release the fixed-resolution network graph and its workspace."""
        self.graphs.release()

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        return self.graphs(x.contiguous(memory_format=CL))


class _OneImage(Portable):
    """Portable for the MLX port: one image per call, as before batching (not measured on Apple GPUs)."""

    batch = 1


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(_OneImage),
}
