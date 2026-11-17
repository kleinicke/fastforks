"""OMNI-DC runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `OGNIDC.forward` with the released demo configuration: the ImageNet-normalised image and the
sparse depth are zero-padded below and on the right to a multiple of 16 (the image in normalised space, i.e. with the
ImageNet mean in [0, 1]) and the output is cropped back. Missing depth (0, negative, NaN, inf) enters as 0. The
conjugate-gradient solve runs in float32 in chunks of steps with one host sync each (upstream syncs every step); it
returns the same iterate as upstream's loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.half import pure_half
from fastforks._core.preprocess import IMAGENET_MEAN
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.depth.models.depth_anything_v2.fused import FusedHead, FusedViT
from fastforks.kernels.dpt_fused import load as load_dpt_fused
from fastforks.shared.depth_anything_v2 import fold_layer_scale, sdpa_attention

from .net import (
    MULTIPLE,
    CGState,
    OmniDCNet,
    mono_size,
    quantile_normalise,
    solve,
    sparse_inputs,
    system,
)

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

_LAYERS = (4, 11, 17, 23)  # fixed: Depth Anything V2 Large's intermediate layers


def _graph_solve(cg: GraphCG, d: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    cg.d.copy_(d)
    cg.b.copy_(b)
    return cg.run()


def _net(ctx: Context) -> OmniDCNet:
    net = OmniDCNet()
    net.load_state_dict(net.checkpoint_keys(ctx.state_dict()))
    net.eval()
    net.fold_bn()
    return net


def prepare(image: torch.Tensor, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Image `[B, 3, H, W]` in [0, 1] and depth `[B, 1, h, w]` -> both padded to a multiple of 16 (image: the mean)."""
    h, w = image.shape[-2:]
    d = prompt.float()
    if tuple(d.shape[-2:]) != (h, w):
        d = F.interpolate(d, (h, w), mode="nearest")
    d = torch.where(torch.isfinite(d) & (d > 0), d, 0)
    pads = (0, -w % MULTIPLE, 0, -h % MULTIPLE)
    if any(pads):
        mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
        image = F.pad(image - mean, pads) + mean
        d = F.pad(d, pads)
    return image, d


def _forward(
    net: OmniDCNet,
    x: torch.Tensor,
    dep: torch.Tensor,
    dav2: torch.Tensor,
) -> torch.Tensor:
    """Normalised padded image, padded sparse depth, Depth Anything V2 map -> depth `[B, 1, H, W]` (float32)."""
    net_in, small, mask = sparse_inputs(dep)
    guide, conf_spn, context, conf_in = net.backbone(x, torch.cat([net_in, dav2], 1))
    grad, up = net.update_block(context)
    d, b = system(grad.float(), small, mask, conf_in.float())
    log_depth = solve(d, b)
    depth = torch.exp(net.upsample(log_depth, up))
    return net.spn(depth, guide, dep, conf_spn)


class Portable(Runner):
    """Plain PyTorch with fused SDPA attention; BatchNorms folded. float32, or on CUDA fp16 networks without autocast.

    In fp16 the RGB-D backbone (its PVT features exceed the fp16 range; upstream's fp16 autocast overflows there),
    the convex upsampling, the DySPN propagation and the conjugate-gradient solve stay float32, as does the Depth
    Anything V2 head's last conv.
    """

    dtypes = ("float32", "float16")
    batch = None
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        net = _net(ctx).to(ctx.device)
        sdpa_attention(net.depth_module.pretrained)
        fold_layer_scale(net.depth_module.pretrained)
        if ctx.dtype == "float16":
            head = net.depth_module.depth_head
            pure_half(net.depth_module, fp32=[head.scratch.output_conv2[2]])
            pure_half(net.update_block)  # the RGB-D backbone stays fp32: its PVT features reach ~1e6
            net.prop_layer.float()
        self.net = net

    def _run(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        x, dep = prepare(image, prompt)
        x = scale(x, self.ctx.io)
        out = _forward(self.net, x, dep, self.net.mono(x))
        return out[:, 0, :h, :w]

    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, H, W]` (0 / NaN: missing)."""
        return {"depth": self._run(image, prompt_depth)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `([B, 3, H, W] in [0, 1], depth [B, 1, H, W])` -> depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image, prompt: self._run(image, prompt))


class GraphCG(CGState):
    """`CGState` whose chunk of steps is a CUDA graph (captured per shape on first use)."""

    CHUNK = 32  # fixed: steps per graph replay (one host sync each)

    def __init__(self, d: torch.Tensor, b: torch.Tensor) -> None:
        """Allocate the state for `b`'s shape and capture the graph."""
        super().__init__(d.clone(), b.clone())
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(self.CHUNK):
                self.step()

    def run(self) -> torch.Tensor:
        """Solve for the current `d` / `b`."""
        self.reset()
        while True:
            self.graph.replay()
            if bool(self.done):
                return self.x.clone()


class Kernels(Runner):
    """dpt_fused kernels for Depth Anything V2 Large (fused-epilogue GEMMs, flash attention, NHWC fp16 DPT head).

    The GRU step is the portable fp16 module, the RGB-D backbone float32 (its PVT features exceed the fp16 range); the
    conjugate-gradient solve is float32 replayed as CUDA graphs of 32 steps. The input resize for Depth Anything V2 is
    PyTorch's (bicubic, align_corners, as upstream), so the kernels' patch resize is the identity.
    """

    batch = None
    prompt_depth = "required"
    prompt_sparse = True

    def __init__(self, ctx: Context) -> None:
        """Build the network on the CPU, convert it, move it to the device."""
        super().__init__(ctx)
        dev = ctx.device
        k = load_dpt_fused(dev)  # before the weights: a kernel build / load failure costs no weight load
        net = _net(ctx)
        self.vit = FusedViT(net.depth_module.pretrained, dev, k, "hh")
        self.head = FusedHead(net.depth_module.depth_head, dev, k)
        del net.depth_module
        net.to(dev)
        pure_half(net.update_block)  # the RGB-D backbone stays fp32: its PVT features reach ~1e6
        self.net = net
        self.cg: dict[tuple[int, ...], GraphCG] = {}

    def _mono(self, image01: torch.Tensor) -> torch.Tensor:
        h, w = image01.shape[-2:]
        size = mono_size(h, w)
        x = F.interpolate(image01, size, mode="bicubic", align_corners=True)
        feats = self.vit.features(x, size, _LAYERS)
        d = self.head(feats, size[0] // 14, size[1] // 14)[:, None]
        return quantile_normalise(F.interpolate(d, (h, w), mode="bilinear", align_corners=True))

    @torch.inference_mode()
    def __call__(  # ty: ignore[invalid-method-override]
        self,
        image: torch.Tensor,
        _intrinsics: torch.Tensor | None,
        prompt_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Metric depth `[B, H, W]` of a batch and its sparse depth maps `[B, 1, H, W]` (0 / NaN: missing)."""
        h, w = image.shape[-2:]
        with torch.autocast("cuda", enabled=False):
            x01, dep = prepare(image.float(), prompt_depth)
            dav2 = self._mono(x01)
            x = scale(x01, self.ctx.io)
            net = self.net
            net_in, small, mask = sparse_inputs(dep)
            guide, conf_spn, context, conf_in = net.backbone(x, torch.cat([net_in, dav2], 1))
            grad, up = net.update_block(context)
            d, b = system(grad.float(), small, mask, conf_in.float())
            key = tuple(b.shape)
            if key not in self.cg:
                self.cg[key] = GraphCG(d, b)
            log_depth = _graph_solve(self.cg[key], d, b)
            depth = torch.exp(net.upsample(log_depth, up))
            out = net.spn(depth, guide, dep, conf_spn)
        return {"depth": out[:, 0, :h, :w]}


RUNNERS = {"portable": Portable, "kernels": Kernels}
