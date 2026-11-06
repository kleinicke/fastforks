"""CoEx runners: weights -> network and the portable / kernels forward passes."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import Inert
from fastforks._core.patch import override
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD
from fastforks.kernels import cudnn_hacc, stereo_ops
from fastforks.shared.cuda_graphs import SmallShapeGraphs
from fastforks.stereo._runner import Crop, Forward, Runner, crop, pad_widths

from .net.coex import BasicConv, CoEx, CostVolume

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule
    from fastforks.shared.cnn_kernels import Patched
    from fastforks.stereo._runner import Context

# The checkpoint is a Lightning file: its hyper-parameter records pickle these two classes (their values are unused).
_EXTRA = {("collections", "Counter"): collections.Counter, ("ruamel.yaml.scalarfloat", "ScalarFloat"): Inert}


@dataclass(frozen=True)
class Params:
    """Architecture of a variant (card `[params]`)."""

    max_disp: int
    top_k: int
    cuda_graphs: bool  # option: kernels replays CUDA graphs for small repeated shapes


def _net(ctx: Context) -> CoEx:
    net = CoEx(ctx.params.max_disp, ctx.params.top_k)
    net.load_state_dict(ctx.state_dict(key="state_dict", strip="stereo.", extra=_EXTRA))
    return net.eval().to(ctx.device)


def forward(net: CoEx, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Disparity `[B, H, W]` float32; inputs are cast to the network's dtype (channels-last on CUDA)."""
    p = next(net.parameters())
    fmt = torch.channels_last if p.is_cuda else torch.contiguous_format
    left, right = (x.to(p.dtype).contiguous(memory_format=fmt) for x in (left, right))
    return net(left, right)


def forward_kernels(net: CoEx, pair: torch.Tensor, k: CubinModule) -> torch.Tensor:
    """`CoEx.forward` with the top-k regression and the superpixel upsampling as two `stereo_ops` kernels.

    `pair`: the scaled fp16 channels-last `[2B, 3, H, W]` pair, left images first.
    """
    left, right = pair.chunk(2)
    x, y, stem_2x = net.features(left, right)
    cost = net.cost_agg(x, net.cost_volume(x[0], y[0])[:, :, :-1])  # [B, 1, D, h, w] fp16
    disp_4 = stereo_ops.soft_argmax(k, cost[:, 0], net.top_k)
    logits = net.spx(net.spx_2(net.spx_4(x[0]), stem_2x))  # the 9 superpixel logits (a channel slice)
    return stereo_ops.spx_upsample(k, disp_4, logits, 4.0)


def _cost_volume_forward(self: Patched, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """`CostVolume.forward` (pad + unfold + multiply + sum + flip) as one `stereo_ops` kernel."""
    k: CubinModule = self._ops
    b, _, h, w = x.shape
    out = torch.empty((b, 1, self.maxdisp, h, w), device=x.device, dtype=torch.float16)
    stereo_ops.corr_volume(k, x, y, out[:, 0], 1.0)
    return out


class _CoExRunner(Runner):
    Params = Params


class Portable(_CoExRunner):
    """Plain PyTorch: float32 (CPU) or pure fp16 weights (CUDA, NHWC); float32 volume statistics and regression."""

    dtypes = ("float32", "float16")

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx).to(ctx.torch_dtype)
        if ctx.device.type == "cuda":
            for m in self.net.modules():  # 4-D weights only: the 3-D convolutions have 5-D ones
                if isinstance(m, torch.nn.Conv2d):
                    m.to(memory_format=torch.channels_last)

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a padded batch."""
        return {"disparity": forward(self.net, left, right)}

    def module(self) -> torch.nn.Module:
        """The network; returns the disparity."""
        return Forward(self.net, forward)


_GRAPH_PIXELS = 500_000  # fixed: launch-bound up to KITTI 375x1242 (batch x H x W of the raw input), measured
_GRAPHS = 2  # fixed: graph cache policy


class Kernels(_CoExRunner):
    """Our kernels: fp16-accumulate convolutions, conv + BatchNorm + LeakyReLU fused into one epilogue, fp16.

    A repeated input shape up to `_GRAPH_PIXELS` replays a CUDA graph (`fastforks.shared.cuda_graphs`): the eager
    pass is launch-bound there. Input scaling and padding are one `stereo_ops` kernel inside the pass
    (`io = "raw"`).
    """

    io = "raw"

    def __init__(self, ctx: Context) -> None:
        """Build the network and the kernel state."""
        super().__init__(ctx)
        from fastforks.shared import cnn_kernels  # noqa: PLC0415 - needs the kernel libraries

        self.net = _net(ctx)
        cnn_kernels.load(ctx.device)
        self.ops = ops = stereo_ops.load(ctx.device)
        for m in self.net.modules():  # BasicConv is conv -> BN -> LeakyReLU(0.01): one fused kernel
            if isinstance(m, BasicConv):
                fused = (
                    cnn_kernels.FusedConvBN(m.conv, m.bn, cnn_kernels.ACT["leaky"] if m.relu else 0)
                    if m.use_bn
                    else None
                )
                if fused is not None:
                    m.fused = fused
                    m.forward = fused.forward  # ty: ignore[invalid-assignment]  # per-instance override
            elif isinstance(m, CostVolume):
                cv: Patched = m
                cv._ops = ops  # noqa: SLF001
                override(cv, _cost_volume_forward)
        self.net.spx[0] = cnn_kernels.pad_deconv_out(self.net.spx[0])  # 9 superpixel weights
        cnn_kernels.prepare_generic(self.net, ctx.device)
        io = ctx.io
        if io.pad_fill not in {"replicate", "zero"}:
            msg = f"coex kernels: unsupported pad fill {io.pad_fill!r}"
            raise ValueError(msg)
        mean, std = {"imagenet": (IMAGENET_MEAN, IMAGENET_STD), "mean_std": (io.mean, io.std)}.get(
            io.input, ((0.0,) * 3, (1.0,) * 3)
        )  # as _core.preprocess.scale, fused into the first kernel
        self._ms = torch.tensor((mean, std), device=ctx.device)
        self._mul = 255.0 if io.input == "0-255" else 1.0
        self._graphs = SmallShapeGraphs(
            self._forward,
            pixels=_GRAPH_PIXELS,
            graphs=_GRAPHS,
            enabled=ctx.params.cuda_graphs,
            capture_after=cudnn_hacc.capture_after(),
        )

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool on the device."""
        self._graphs.release()

    def __call__(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        """Disparity of a raw `[0, 1]` pair (`io = "raw"`: padding and scaling run inside the pass)."""
        return {"disparity": self._graphs(left, right)}

    def _forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        h, w = left.shape[-2:]
        io = self.ctx.io
        pads = pad_widths(h, w, io)
        pair = stereo_ops.norm_pad_pair(self.ops, left, right, self._ms, pads=pads, mul=self._mul, fill=io.pad_fill)
        disp = forward_kernels(self.net, pair, self.ops)
        return crop(disp, Crop(pads[2], pads[0], h, w))


RUNNERS = {"portable": Portable, "kernels": Kernels}
