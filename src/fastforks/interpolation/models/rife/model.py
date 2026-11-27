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
from fastforks._core.sampling import grid_sample
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.cuda_graphs import GraphCache

from .net import IFNet
from .net.IFNet_HDv3 import ResConv

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


def _grid(py: torch.Tensor, px: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`grid_sample` grid `[len(py), len(px), 2]` of the pixels `py` x `px` of an `h` x `w` image (upstream `warp`)."""
    gx = torch.linspace(-1.0, 1.0, w, device=px.device)[px.long()]
    gy = torch.linspace(-1.0, 1.0, h, device=py.device)[py.long()]
    return torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)


def _units(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Pixels per `grid_sample` unit (x, y) of an `h` x `w` image (upstream `warp`'s flow scale)."""
    # filled on the device: `d[0] = x` copies a CPU scalar, which a CUDA graph capture rejects
    d = torch.empty(2, device=device)
    d[0].fill_((w - 1.0) / 2.0)
    d[1].fill_((h - 1.0) / 2.0)
    return d


def _warp(src: torch.Tensor, flow: torch.Tensor, base: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Upstream `warp` of both sides `[2B, C, H, W]` (side-major) by their flows `[B, 4, h, w]` at `base` (`_units` d).

    One `grid_sample` (bilinear, border, align_corners) -> `[2B, C, h, w]`.
    """
    b, _, fh, fw = flow.shape
    fp = flow.view(b, 2, 2, fh, fw).transpose(0, 1).reshape(2 * b, 2, fh, fw).permute(0, 2, 3, 1)
    return grid_sample(src, torch.addcdiv(base, fp, d), mode="bilinear", padding_mode="border", align_corners=True)


def _up_grid(py: torch.Tensor, px: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`grid_sample` grid `[len(py), len(px), 2]` of the pixels `py` x `px` of an `h` x `w` resize."""
    gy, gx = torch.meshgrid((py + 0.5) * (2 / h) - 1, (px + 0.5) * (2 / w) - 1, indexing="ij")
    return torch.stack((gx, gy), -1)


def _upsampled(x: torch.Tensor, s: int, grid: torch.Tensor | None, size: tuple[int, int]) -> torch.Tensor:
    """`x` bilinearly upsampled to `size` (upstream `IFBlock`'s output resize) at the `_up_grid` pixels."""
    if s <= 2 or grid is None:  # noqa: PLR2004 - every pixel: the resize itself
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    g = grid.expand(x.shape[0], -1, -1, -1)
    return grid_sample(
        x.float(), g, mode="bilinear", padding_mode="border", align_corners=False
    )  # border = the resize's edge clamp


def _sources(x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    """Both sides' image + features, side-major, float32."""
    images = torch.cat((x[:, :3], x[:, 3:6]), 0)
    src = torch.empty(images.shape[0], 3 + feat.shape[1], *x.shape[2:], device=x.device)
    src[:, :3].copy_(images)
    src[:, 3:].copy_(feat)
    return src


Grids = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]  # warp grid, upsample grid (s > 2), pixel units


def _grids(h: int, w: int, s: int, device: torch.device, cache: dict | None) -> Grids:
    """The 1/`s` block's warp and upsample grids of an `h` x `w` image and its pixel units (cached in `cache`)."""
    key = (h, w, s, device)
    if cache is not None and key in cache:
        return cache[key]
    py, px = _points(h, s, device), _points(w, s, device)
    out = (_grid(py, px, h, w), _up_grid(py, px, h, w) if s > 2 else None, _units(h, w, device))  # noqa: PLR2004
    if cache is not None:
        cache[key] = out
    return out


def _interpolate(  # noqa: PLR0913 - upstream's forward, one loop over the blocks
    net: IFNet,
    x: torch.Tensor,
    timestep: float | torch.Tensor,
    scales: list[int],
    *,
    channels_last: bool = False,
    cache: dict | None = None,
) -> torch.Tensor:
    """`IFNet.forward(x, timestep, scales)[2][-1]` (fastmode, no ensemble), evaluated where the blocks read it.

    `cache` keeps the shape-only grids between calls (eager only: not while exporting or capturing a CUDA graph).
    """
    dtype, dev = x.dtype, x.device
    b, _, h, w = x.shape
    feat = torch.cat((net.encode(x[:, :3]), net.encode(x[:, 3:6])), 0)  # per image, as upstream (faster than batched)
    nf = feat.shape[1]
    if torch.compiler.is_exporting():  # no fake tensors in the cache
        cache = None
    src = _sources(x, feat)  # both sides' image + features, side-major, float32
    del feat
    flow = tmp = None
    prev = 1
    blocks = [net.block0, net.block1, net.block2, net.block3, net.block4]
    for block, s in zip(blocks, scales, strict=True):
        grid, up_grid, units = _grids(h, w, s, dev, cache)
        hs, ws = h // s, w // s
        warped = src if flow is None else _warp(src, _at_points(flow, s), grid, units)
        side = _pool(_at_points(warped, s) if flow is None else warped, s).view(2, b, 3 + nf, hs, ws).transpose(0, 1)
        extra = 0 if tmp is None else tmp.shape[1]  # mask + features (9), flow (4)
        inp = torch.empty(b, 7 + 2 * nf + extra, hs, ws, device=dev, dtype=dtype)  # the block input, cast once
        inp[:, :6].view(b, 2, 3, hs, ws).copy_(side[:, :, :3])  # image 1, image 2
        inp[:, 6 : 6 + 2 * nf].view(b, 2, nf, hs, ws).copy_(side[:, :, 3:])  # features 1, features 2
        del side, warped
        c = 6 + 2 * nf
        if isinstance(timestep, torch.Tensor):
            inp[:, c : c + 1].copy_(timestep.expand(b, 1, hs, ws))
        else:
            inp[:, c : c + 1].fill_(timestep)
        if tmp is not None and flow is not None:  # mask and features of the last block, its flow (block scale units)
            inp[:, c + 1 : c + extra - 3].copy_(_pool(_upsampled(tmp[:, 4:], s, up_grid, (h, w)), s))
            inp[:, c + extra - 3 :].copy_(_pool(_at_points(flow, s), s).div_(s))
        # Release the full-resolution inputs once the strided stem has consumed them.
        block_input = block.conv0(inp.contiguous(memory_format=torch.channels_last) if channels_last else inp)
        del inp
        block_input = block.convblock(block_input)
        tmp = block.lastconv(block_input)
        del block_input
        up = tmp[:, :4].float()
        if s > 1:
            up = F.interpolate(up, scale_factor=s, mode="bilinear", align_corners=False)
        flow = (up * s).contiguous() if flow is None else flow.add_(up, alpha=s)
        del up
        prev = s
    if tmp is None or flow is None:
        msg = "RIFE needs at least one block scale"
        raise ValueError(msg)
    mask = tmp[:, 4:5].float()
    if prev > 1:
        mask = F.interpolate(mask, scale_factor=prev, mode="bilinear", align_corners=False)
    grid, _, units = _grids(h, w, 1, dev, cache)
    warped = _warp(src[:, :3], flow, grid, units)
    return torch.lerp(warped[b:], warped[:b], torch.sigmoid(mask)).to(dtype)


def _net(ctx: Context) -> IFNet:
    """Upstream's network with the variant's checkpoint (float32, CPU)."""
    net = IFNet(c4=ctx.params.c4)
    sd = {k: v for k, v in ctx.state_dict().items() if not k.startswith(_TRAINING_ONLY)}
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net


class Portable(Runner):
    """Plain PyTorch in the card's dtype (the whole network; the warps sample in float32)."""

    dtypes = ("float32", "float16")
    Params = Params
    batch = None
    # pixels (pairs x H x W) per network call: the eager call is launch bound at small sizes (RTX 5060 Ti, Windows,
    # fp16, 8 pairs 480 x 640: 9.6 -> 3.8 ms per pair, 4 a call), not from 720p on (11.8 ms either way, 2.2 GB)
    batch_pixels = 1_300_000
    channels_last = False  # the blocks' input layout
    # cuDNN's default transposed convs accumulate with atomics, and the warps amplify the last bits: two calls on the
    # same 720 x 1280 pair differed by up to 0.009 (smooth frames; 0.95 on noise). Deterministic costs ~3 % there.
    deterministic_cudnn = True

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        if ctx.device.type == "cuda":
            # residual scales folded into their convs (in float32, before the cast): one launch fewer per ResConv,
            # 40 per frame; the eager call is launch bound (RTX 5060 Ti, Windows, fp16 384 x 384: 10.0 -> 9.3 ms)
            _fold_res_convs(self.net)
        self.net.eval().to(ctx.device, getattr(torch, ctx.dtype))
        if ctx.device.type == "cuda" and ctx.dtype == "float16":
            # channels-last blocks and weights: cuDNN otherwise converts NCHW <-> NHWC around each fp16 conv (~120
            # extra kernels per call). RTX 5060 Ti 384 x 384 11.0 -> 10.0 ms, 720p 14.1 -> 12.2, 1080p 27.4 -> 25.1
            # (82 dB from NCHW, less memory); float32 is slower and larger channels-last, so it stays NCHW
            self.channels_last = True
            self.net.to(memory_format=torch.channels_last)
        self._grids: dict | None = {}  # shape -> `_grids` (portable: eager, 2 shapes)

    def _forward(self, x: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        if self._grids is not None and len(self._grids) > 2 * len(self.ctx.params.scales) + 2:  # 2 shapes' worth
            self._grids.clear()
        return _interpolate(
            self.net, x, timestep, list(self.ctx.params.scales), channels_last=self.channels_last, cache=self._grids
        )

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
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation, fuse_bias=True)  # ty: ignore[invalid-argument-type]


def _hacc_deconv(deconv: torch.nn.ConvTranspose2d, x: torch.Tensor) -> torch.Tensor:
    y = cudnn_hacc.deconv_half_acc(x, deconv.weight, deconv.padding, deconv.stride)  # ty: ignore[invalid-argument-type]
    return y.add_(deconv.bias.view(1, -1, 1, 1))  # ty: ignore[unresolved-attribute]


def _res_conv(m: ResConv, x: torch.Tensor) -> torch.Tensor:
    return m.relu(m.conv(x) + x)  # beta folded into the conv


def _fold_res_convs(net: torch.nn.Module) -> None:
    """Fold each `ResConv`'s `beta` into its conv's weight and bias (`conv(x) * beta` -> `conv'(x)`)."""
    for m in net.modules():
        if isinstance(m, ResConv):
            beta = m.beta.data.view(-1)
            m.conv.weight.data.mul_(beta.view(-1, 1, 1, 1))
            m.conv.bias.data.mul_(beta)  # ty: ignore[unresolved-attribute]
            override(m, _res_conv)


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs and transposed convs (`cudnn_hacc`), one CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators
    batch = 1  # one pair per graph replay: batched pairs are not faster (480 x 640: 3.3 vs 3.1 ms, 720p 8.6 vs 10.4)
    channels_last = True
    deterministic_cudnn = False  # its convolutions are `cudnn_hacc` plans, not cuDNN's algorithm choice

    def __init__(self, ctx: Context) -> None:
        """Build the network (residual scales folded), route the convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_conv)
            elif isinstance(m, torch.nn.ConvTranspose2d) and base_type(m) is torch.nn.ConvTranspose2d:
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                override(m, _hacc_deconv)
        self._grids = None  # a captured graph holds its own grids
        after = cudnn_hacc.capture_after()
        self.graphs = (
            GraphCache(self._eager, graphs=1, capture_after=after, capture_warmup=cudnn_hacc.BUILD_AFTER > 0)
            if ctx.params.cuda_graphs
            else None
        )

    def _eager(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self._forward(x, timestep)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch. One graph handles every timestep of the same shape."""
        b, _, h, w = image1.shape
        x = torch.empty((b, 6, h, w), dtype=torch.float16, device=image1.device, memory_format=torch.channels_last)
        torch.cat((image1, image2), dim=1, out=x)
        t = torch.full((x.shape[0], 1, 1, 1), timestep, dtype=x.dtype, device=x.device)
        out = None if self.graphs is None else self.graphs.run((x.device, tuple(x.shape)), x, t)
        return {"frame": self._eager(x, t) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


class MlxPortable(Runner):
    """Apple MLX (`device="mlx"`): the convolutions in MLX, the warps and resizes between them as Metal kernels.

    `mlx_net`: float32, or float16 convolutions with the flow, the warps and the blend in float32 (as `Portable`).
    """

    Params = Params
    batch = 1

    def __init__(self, ctx: Context) -> None:
        """Convert or read the cached weights and build the MLX network."""
        super().__init__(ctx)
        from fastforks._core.mlx import converted  # noqa: PLC0415 - needs the mlx extra

        from . import mlx_net  # noqa: PLC0415

        build = lambda: mlx_net.prepare(_net(ctx).eval())  # noqa: E731
        p = converted(ctx, mlx_net.LAYOUT, build, dtype=ctx.dtype, fp32=mlx_net.fp32)
        self.net = mlx_net.Net(p, ctx.params.scales, ctx.dtype)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of one padded pair."""
        return {"frame": self.net(image1, image2, timestep)}


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": MlxPortable}
