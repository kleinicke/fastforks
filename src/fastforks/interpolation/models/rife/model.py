"""RIFE runners: Practical-RIFE's `IFNet_HDv3` (releases 4.25 / 4.26) and its `inference` call.

As upstream `RIFE_HDv3.Model.inference`: the two frames in `[0, 1]` are concatenated, five IFBlocks refine the
bidirectional flow and the fusion mask from 1/16 (lite: 1/32) to full resolution (`scale` 1), and the frame is the
mask-weighted blend of both frames warped to the time. Upstream's `inference_video.py` pads to a multiple of 128 with
zeros at the right and bottom; the card's `[io] pad` does the same.

`_interpolate` is `IFNet.forward` reordered, not changed: an IFBlock only reads its input bilinearly downscaled by an
even `s`, i.e. the mean of the 2 x 2 pixels around each output pixel. So the warps and the upsampled mask / features
are evaluated at those pixels only (full resolution only for the last two blocks), each side's image and features are
warped by one `grid_sample`, and the flow is accumulated in float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import base_type, override
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import IFNet
from .net.IFNet_HDv3 import IFBlock, ResConv

if TYPE_CHECKING:
    from fastforks.interpolation._runner import Context

_TRAINING_ONLY = ("teacher.", "caltime.")  # in the checkpoints, unused at inference (upstream loads with strict=False)


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    scales: tuple[int, ...]  # the five IFBlocks' downscale factors (upstream `scale_list` at `scale` 1)
    c4: int  # width of the last IFBlock
    cuda_graphs: bool


def _points(n: int, s: int, device: torch.device) -> torch.Tensor:
    """Pixel indices (of `n`) a bilinear 1/`s` downscale reads: 2 per output pixel (all `n` for `s` <= 2)."""
    if s == 1:
        return torch.arange(n, device=device, dtype=torch.float32)
    i = torch.arange(n // s, device=device, dtype=torch.float32)[:, None] * s + (s // 2 - 1)
    return (i + torch.arange(2, device=device)).flatten()


def _at_points(x: torch.Tensor, s: int) -> torch.Tensor:
    """Full-resolution `x` `[B, C, H, W]` at the `_points` rows and columns: `[B, C, 2H/s, 2W/s]` (`x` for `s` <= 2)."""
    if s <= 2:  # noqa: PLR2004 - every pixel
        return x
    b, c, h, w = x.shape
    k = s // 2 - 1
    return x.view(b, c, h // s, s, w // s, s)[:, :, :, k : k + 2, :, k : k + 2].reshape(b, c, 2 * h // s, 2 * w // s)


def _pool(x: torch.Tensor, s: int) -> torch.Tensor:
    """The 1/`s` bilinear downscale of the full-resolution field sampled at the `_points` (the same 2 x 2 means)."""
    return x if s == 1 else F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


def _block(block: IFBlock, x: torch.Tensor) -> torch.Tensor:
    """IFBlock on its already downscaled input, before its upsampling: `[B, 13, H/s, W/s]` (flow, mask, features)."""
    return block.lastconv(block.convblock(block.conv0(x)))


def _grid(py: torch.Tensor, px: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`grid_sample` grid `[len(py), len(px), 2]` of the pixels `py` x `px` of an `h` x `w` image (upstream `warp`)."""
    gx = torch.linspace(-1.0, 1.0, w, device=px.device)[px.long()]
    gy = torch.linspace(-1.0, 1.0, h, device=py.device)[py.long()]
    return torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)


def _warp(src: torch.Tensor, flow: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """Upstream `warp` of both sides `[2B, C, H, W]` (side-major) by their flows `[B, 4, h, w]` at `base`.

    One `grid_sample` (bilinear, border, align_corners) -> `[2B, C, h, w]`.
    """
    h, w = src.shape[-2:]
    b, _, fh, fw = flow.shape
    d = torch.empty(2, device=flow.device)  # pixels per grid unit (filled on the device: CUDA-graph safe)
    d[0], d[1] = (w - 1.0) / 2.0, (h - 1.0) / 2.0
    fp = flow.view(b, 2, 2, fh, fw).transpose(0, 1).reshape(2 * b, 2, fh, fw).permute(0, 2, 3, 1)
    return F.grid_sample(src, torch.addcdiv(base, fp, d), "bilinear", "border", align_corners=True)


def _upsampled(x: torch.Tensor, s: int, pts: tuple[torch.Tensor, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
    """`x` bilinearly upsampled to `size` (upstream `IFBlock`'s output resize) at the pixels `pts` (rows, columns)."""
    (py, px), (h, w) = pts, size
    if s <= 2:  # noqa: PLR2004 - every pixel: the resize itself
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    gy, gx = torch.meshgrid((py + 0.5) * (2 / h) - 1, (px + 0.5) * (2 / w) - 1, indexing="ij")
    g = torch.stack((gx, gy), -1).expand(x.shape[0], -1, -1, -1)
    return F.grid_sample(x.float(), g, "bilinear", "border", align_corners=False)  # border = the resize's edge clamp


def _interpolate(
    net: IFNet, x: torch.Tensor, timestep: float | torch.Tensor, scales: list[int], *, channels_last: bool = False
) -> torch.Tensor:
    """`IFNet.forward(x, timestep, scales)[2][-1]` (fastmode, no ensemble), evaluated where the blocks read it."""
    dtype, dev = x.dtype, x.device
    b, _, h, w = x.shape
    feat = torch.cat((net.encode(x[:, :3]), net.encode(x[:, 3:6])), 0)  # per image, as upstream (faster than batched)
    nf = feat.shape[1]
    src = torch.empty(2 * b, 3 + nf, h, w, device=dev)  # both sides' image + features, side-major, float32
    src[:, :3].copy_(torch.cat((x[:, :3], x[:, 3:6]), 0))
    src[:, 3:].copy_(feat)
    flow = tmp = None
    prev = 1
    blocks = [net.block0, net.block1, net.block2, net.block3, net.block4]
    for block, s in zip(blocks, scales, strict=True):
        py, px = _points(h, s, dev), _points(w, s, dev)
        hs, ws = h // s, w // s
        warped = src if flow is None else _warp(src, _at_points(flow, s), _grid(py, px, h, w))
        side = _pool(_at_points(warped, s) if flow is None else warped, s).view(2, b, 3 + nf, hs, ws).transpose(0, 1)
        extra = 0 if tmp is None else tmp.shape[1]  # mask + features (9), flow (4)
        inp = torch.empty(b, 7 + 2 * nf + extra, hs, ws, device=dev, dtype=dtype)  # the block input, cast once
        inp[:, :6].view(b, 2, 3, hs, ws).copy_(side[:, :, :3])  # image 1, image 2
        inp[:, 6 : 6 + 2 * nf].view(b, 2, nf, hs, ws).copy_(side[:, :, 3:])  # features 1, features 2
        c = 6 + 2 * nf
        if isinstance(timestep, torch.Tensor):
            inp[:, c : c + 1].copy_(timestep.expand(b, 1, hs, ws))
        else:
            inp[:, c : c + 1].fill_(timestep)
        if tmp is not None and flow is not None:  # mask and features of the last block, its flow (block scale units)
            inp[:, c + 1 : c + extra - 3].copy_(_pool(_upsampled(tmp[:, 4:], s, (py, px), (h, w)), s))
            inp[:, c + extra - 3 :].copy_(_pool(_at_points(flow, s), s).div_(s))
        tmp = _block(block, inp.contiguous(memory_format=torch.channels_last) if channels_last else inp)
        up = tmp[:, :4].float()
        if s > 1:
            up = F.interpolate(up, scale_factor=s, mode="bilinear", align_corners=False)
        flow = (up * s).contiguous() if flow is None else flow.add_(up, alpha=s)
        prev = s
    if tmp is None or flow is None:
        msg = "RIFE needs at least one block scale"
        raise ValueError(msg)
    mask = tmp[:, 4:5].float()
    if prev > 1:
        mask = F.interpolate(mask, scale_factor=prev, mode="bilinear", align_corners=False)
    warped = _warp(src[:, :3], flow, _grid(_points(h, 1, dev), _points(w, 1, dev), h, w))
    return torch.lerp(warped[b:], warped[:b], torch.sigmoid(mask)).to(dtype)


class Portable(Runner):
    """Plain PyTorch in the card's dtype (the whole network; the warps sample in float32)."""

    dtypes = ("float32", "float16")
    Params = Params
    channels_last = False  # the blocks' input layout

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = IFNet(c4=ctx.params.c4)
        sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith(_TRAINING_ONLY)}
        check_fits(self.net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
        self.net.load_state_dict(sd)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))

    def _forward(self, x: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        return _interpolate(self.net, x, timestep, list(self.ctx.params.scales), channels_last=self.channels_last)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch."""
        dtype = getattr(torch, self.ctx.dtype)
        return {"frame": self._forward(torch.cat((image1, image2), 1).to(dtype), timestep)}

    def module(self) -> torch.nn.Module:
        """The network; returns the middle frame (time 0.5)."""
        scales, dtype = list(self.ctx.params.scales), getattr(torch, self.ctx.dtype)
        return Forward(
            self.net,
            lambda net, image1, image2: _interpolate(net, torch.cat((image1, image2), 1).to(dtype), 0.5, scales),
        )


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


def _hacc_deconv(deconv: torch.nn.ConvTranspose2d, x: torch.Tensor) -> torch.Tensor:
    y = cudnn_hacc.deconv_half_acc(x, deconv.weight, deconv.padding, deconv.stride)  # ty: ignore[invalid-argument-type]
    return y.add_(deconv.bias.view(1, -1, 1, 1))  # ty: ignore[unresolved-attribute]


def _res_conv(m: ResConv, x: torch.Tensor) -> torch.Tensor:
    return m.relu(m.conv(x) + x)  # beta folded into the conv


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs and transposed convs (`cudnn_hacc`), one CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators
    channels_last = True

    def __init__(self, ctx: Context) -> None:
        """Build the network, fold the residual scales, route the convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, ResConv):
                beta = m.beta.data.view(-1)
                m.conv.weight.data.mul_(beta.view(-1, 1, 1, 1))
                m.conv.bias.data.mul_(beta)  # ty: ignore[unresolved-attribute]
                override(m, _res_conv)
            elif isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)
            elif isinstance(m, torch.nn.ConvTranspose2d) and base_type(m) is torch.nn.ConvTranspose2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_deconv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self._forward(x, timestep)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        x = torch.cat((image1, image2), 1).to(torch.float16).contiguous(memory_format=torch.channels_last)
        t = torch.full((x.shape[0], 1, 1, 1), timestep, dtype=x.dtype, device=x.device)
        out = None if self.graphs is None else self.graphs.run((x.device, tuple(x.shape)), x, t)
        return {"frame": self._eager(x, t) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
