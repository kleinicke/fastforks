"""EMA-VFI runners: `MultiScaleFlow` on the `MotionFormer` backbone and its `Model.inference` call.

As upstream `Trainer.Model.inference` (called as `benchmark/MiddleBury.py`): the two frames in `[0, 1]` (BGR, as
`cv2.imread`; replicate-padded to a multiple of 32, centred) go through the backbone together; its window-attention
stages (1/8 and 1/16) let each frame attend to the other and return appearance and motion features; two heads refine
the flow to both frames and the fusion mask from 1/16 to full size, the U-Net adds a residual to the blend of both
warped frames. `tta` = upstream's `fast_TTA`: the pair and its 180-degree rotation in one batch, averaged.

`_interpolate` is `MultiScaleFlow.forward` without its unused outputs (the per-stage flows, masks and blends): the
window masks and position windows are built once per shape (upstream builds the shift mask on the CPU every call and
caches it in buffers registered during forward), the flow and mask are accumulated and warped in float32, and the
features and convolutions run in the runner's dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import check_fits
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import base_type, override
from fastforks.interpolation._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc, ema_warp
from fastforks.shared.cuda_graphs import GraphCache
from fastforks.shared.swin_sr import roll

from .net import MotionFormer, MultiScaleFlow
from .net.feature_extractor import window_partition, window_reverse

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from fastforks.interpolation._runner import Context

    from .net.feature_extractor import InterFrameAttention, MotionFormerBlock
    from .net.flow_estimation import Head

Cache = dict[tuple[Any, ...], Any]


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    width: int  # config.init_model_config F
    depths: tuple[int, ...]
    window: int  # attention window of both transformer stages
    tta: bool  # average with the pair rotated by 180 degrees (upstream fast_TTA)
    cuda_graphs: bool  # kernels: replay the network from a CUDA graph


def _config(p: Params) -> tuple[dict[str, Any], dict[str, Any]]:
    """config.py `init_model_config(F=width, W=window, depth=depths)`: the backbone's and the flow model's kwargs."""
    f, d, w = p.width, list(p.depths), p.window
    common = {
        "embed_dims": [f, 2 * f, 4 * f, 8 * f, 16 * f],
        "motion_dims": [0, 0, 0, 8 * f // d[-2], 16 * f // d[-1]],
        "num_heads": [8 * f // 32, 16 * f // 32],
        "depths": d,
        "window_sizes": [w, w],
    }
    backbone = {**common, "mlp_ratios": [4, 4], "qkv_bias": True, "norm_layer": partial(torch.nn.LayerNorm, eps=1e-6)}
    return backbone, {**common, "scales": [4, 8, 16], "hidden_dims": [4 * f, 4 * f], "c": f}


def _regions(h: int, w: int, hs: tuple[slice, ...], ws: tuple[slice, ...]) -> torch.Tensor:
    """`[1, h, w, 1]` region ids: `cnt` over the `hs` x `ws` slices (upstream's `img_mask` / `shift_mask`)."""
    m = torch.zeros((1, h, w, 1))
    for i, a in enumerate(hs):
        for j, b in enumerate(ws):
            m[:, a, b, :] = i * len(ws) + j
    return m


def _pairs(regions: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    """`[nW, N, N]` 0 where two tokens of a window share a region, else 1."""
    mw = window_partition(regions, window).squeeze(-1)
    return (mw.unsqueeze(1) - mw.unsqueeze(2) != 0).float()


def _mask(h: int, w: int, window: tuple[int, int], shift: tuple[int, int]) -> torch.Tensor | None:
    """The additive `[nW, N, N]` attention mask of a block (-100 across padding / shift regions) or None.

    Upstream `pad_if_needed` (centred padding) and `MotionFormerBlock.forward`'s shift mask, merged as there.
    """
    ph, pw = -h % window[0], -w % window[1]
    hp, wp = h + ph, w + pw
    pad = None
    if ph or pw:
        hs = (slice(0, ph // 2), slice(ph // 2, h + ph // 2), slice(h + ph // 2, None))
        ws = (slice(0, pw // 2), slice(pw // 2, w + pw // 2), slice(w + pw // 2, None))
        pad = _pairs(_regions(hp, wp, hs, ws), window)
    if not (shift[0] or shift[1]):
        return None if pad is None else pad * -100.0
    hs = (slice(0, -window[0]), slice(-window[0], -shift[0]), slice(-shift[0], None))
    ws = (slice(0, -window[1]), slice(-window[1], -shift[1]), slice(-shift[1], None))
    m = _pairs(_regions(hp, wp, hs, ws), window)
    if pad is not None:
        m = torch.maximum(m, pad)
    return m * -100.0


def _padded(x: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    """`[B, H, W, C]` zero-padded (centred) to multiples of the window (upstream `pad_if_needed`)."""
    h, w = x.shape[1:3]
    ph, pw = -h % window[0], -w % window[1]
    if not (ph or pw):
        return x
    return F.pad(x, (0, 0, pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))


def _depadded(x: torch.Tensor, h: int, w: int, window: tuple[int, int]) -> torch.Tensor:
    ph, pw = -h % window[0], -w % window[1]
    return x[:, ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w, :]


def _rolled(x: torch.Tensor, shift: tuple[int, int], sign: int) -> torch.Tensor:
    return roll(x, (sign * shift[0], sign * shift[1]), (1, 2)) if shift[0] or shift[1] else x


def _eager_cuda(x: torch.Tensor) -> bool:
    """Whether inference can reuse owned CUDA activations."""
    return x.is_cuda and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled())


def _attention(
    m: InterFrameAttention, x1: torch.Tensor, x2: torch.Tensor, cor: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`InterFrameAttention.forward`: one softmax for the appearance (`attn @ v`) and the motion (`attn @ cor`)."""
    b, n, c = x1.shape
    heads, md = m.num_heads, m.motion_dim
    q = m.q(x1).reshape(b, n, heads, c // heads).transpose(1, 2)
    k, v = m.kv(x2).reshape(b, n, 2, heads, c // heads).permute(2, 0, 3, 1, 4)
    cor_ = m.cor_embed(cor)
    ce = cor_.reshape(b, n, heads, md // heads).transpose(1, 2)
    attn = q @ k.transpose(-2, -1)
    del q, k
    reuse = _eager_cuda(attn)
    attn = attn.mul_(m.scale) if reuse else attn * m.scale
    if mask is not None:
        nw = mask.shape[0]
        grouped = attn.view(b // nw, nw, heads, n, n)
        bias = mask[None, :, None].to(attn.dtype)
        attn = (grouped.add_(bias) if reuse else grouped + bias).view(b, heads, n, n)
        del grouped, bias
    attn = attn.softmax(dim=-1)
    out = attn @ torch.cat((v, ce), -1)  # one product for both (upstream: two)
    x = m.proj(out[..., : c // heads].transpose(1, 2).reshape(b, n, c))
    motion = m.motion_proj(out[..., c // heads :].transpose(1, 2).reshape(b, n, md) - cor_)
    return x, motion


def _sizes(blk: MotionFormerBlock) -> tuple[tuple[int, int], tuple[int, int]]:
    """The block's window and shift (upstream `to_2tuple`s)."""
    return cast("tuple[int, int]", blk.window_size), cast("tuple[int, int]", blk.shift_size)


def _former_block(  # noqa: PLR0913, PLR0917 - upstream's arguments + the per-shape constants
    blk: MotionFormerBlock, x: torch.Tensor, cor_win: torch.Tensor, mask: torch.Tensor | None, h: int, w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`MotionFormerBlock.forward` with the shape's mask and position windows given (`[2B, H*W, C]` in and out)."""
    win, shift = _sizes(blk)
    b2, _, c = x.shape
    xp = _rolled(_padded(x.view(b2, h, w, c), win), shift, -1)
    hp, wp = xp.shape[1:3]
    xn = blk.norm1(window_partition(xp, win))
    half = xn.shape[0] // 2
    app, motion = _attention(blk.attn, xn, torch.cat((xn[half:], xn[:half])), cor_win, mask)
    xn = xn.add_(app) if _eager_cuda(xn) else xn + app  # upstream: the residual on the normed tokens
    del app, xp
    back = _depadded(_rolled(window_reverse(xn, win, hp, wp), shift, 1), h, w, win).reshape(b2, h * w, c)
    motion = _depadded(_rolled(window_reverse(motion, win, hp, wp), shift, 1), h, w, win).reshape(b2, h * w, -1)
    return back + blk.mlp(blk.norm2(back), h, w), motion


def _constants(blk: MotionFormerBlock, shape: tuple[int, int, int], x: torch.Tensor, cache: Cache | None) -> tuple:
    """The block's attention mask and position windows (upstream `get_cor`, padded and rolled), per `(B, H, W)`."""
    b2, h, w = shape
    win, shift = _sizes(blk)
    key = (b2, h, w, win, shift, x.device, x.dtype)
    if cache is not None and key in cache:
        return cache[key]
    gx = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, w).expand(b2, h, w)
    gy = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, h, 1).expand(b2, h, w)
    cor = _rolled(_padded(torch.stack((gx, gy), -1), win), shift, -1)
    mask = _mask(h, w, win, shift)
    out = (window_partition(cor, win).to(x.dtype), None if mask is None else mask.to(x.device))
    if cache is not None and type(out[0]) is torch.Tensor:  # not a traced one (an MLX export's fake tensor)
        cache[key] = out
    return out


def _backbone(
    m: MotionFormer, x1: torch.Tensor, x2: torch.Tensor, cache: Cache | None
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """`MotionFormer.forward`: appearance features of all 5 stages and motion features of the attention stages."""
    x = torch.cat((x1, x2), 0)
    b2 = x.shape[0]
    af: list[torch.Tensor] = []
    mf: list[torch.Tensor | None] = []
    xs: list[torch.Tensor] = []
    for i in range(m.num_stages):
        embed: Any = getattr(m, f"patch_embed{i + 1}", None)
        block = getattr(m, f"block{i + 1}")
        if i < m.conv_stages:
            x = block(x if embed is None else embed(x))
            xs.append(x)
            mf.append(None)
        else:
            x, h, w = embed(xs) if i == m.conv_stages else embed(x)
            motions = []
            for blk in block:
                cor_win, mask = _constants(blk, (b2, h, w), x, cache)
                x, xm = _former_block(blk, x, cor_win, mask, h, w)
                motions.append(xm.reshape(b2, h, w, -1).permute(0, 3, 1, 2))
            x = getattr(m, f"norm{i + 1}")(x).reshape(b2, h, w, -1).permute(0, 3, 1, 2).contiguous()
            mf.append(torch.cat(motions, 1))
        af.append(x)
    return af, mf


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Upstream `warp` (bilinear, border, align_corners) in float32: `x` `[B, C, H, W]` by `flow` `[B, 2, H, W]`."""
    h, w = flow.shape[-2:]
    gx = torch.linspace(-1.0, 1.0, w, device=flow.device).view(1, 1, w)
    gy = torch.linspace(-1.0, 1.0, h, device=flow.device).view(1, h, 1)
    f = flow.float()
    g = torch.stack((gx + f[:, 0] / ((w - 1.0) / 2.0), gy + f[:, 1] / ((h - 1.0) / 2.0)), -1)
    return F.grid_sample(x.float(), g, mode="bilinear", padding_mode="border", align_corners=True)


def _warp_pair(x0: torch.Tensor, x1: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _warp(x0, flow[:, :2]), _warp(x1, flow[:, 2:4])


def _head(
    hd: Head, motion: torch.Tensor, xs: tuple[torch.Tensor, ...], flow: torch.Tensor | None, fmt: torch.memory_format
) -> tuple[torch.Tensor, torch.Tensor]:
    """`Head.forward(motion, cat(xs), flow)`: the flow and mask update at full size in float32.

    Its convs run in the features' dtype. Bilinear resizing works per channel, so the pieces are resized one by one
    and concatenated at the head's size instead of concatenating them at full size first (the same values).
    """
    s = hd.scale
    motion = hd.upsample(motion)
    if s != 4:  # noqa: PLR2004 - upstream's 1/4 head
        xs = tuple(F.interpolate(x, scale_factor=4.0 / s, mode="bilinear", align_corners=False) for x in xs)
    if flow is not None:
        if s != 4:  # noqa: PLR2004
            flow = F.interpolate(flow, scale_factor=4.0 / s, mode="bilinear", align_corners=False) * 4.0 / s
        xs = (*xs, flow)
    y = hd.conv(torch.cat((motion, *(x.to(motion.dtype, memory_format=fmt) for x in xs)), 1)).float()
    if s != 4:  # noqa: PLR2004
        y = F.interpolate(y, scale_factor=s // 4, mode="bilinear", align_corners=False)
        return y[:, :4] * (s // 4), y[:, 4:5]
    return y[:, :4], y[:, 4:5]


def _refine_warped(
    net: MultiScaleFlow,
    parts: tuple[torch.Tensor, ...],
    features: list[torch.Tensor],
    stack: Callable[[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run each U-Net block as soon as its pair of warped features is available, releasing that scale's inputs."""
    unet = net.unet
    flow = parts[-1]
    inputs = parts
    skips = []
    decoder = (unet.up1, unet.up2, unet.up3)
    features.reverse()  # pop in fine-to-coarse order so already consumed backbone activations can be freed
    for index, block in enumerate((unet.down0, unet.down1, unet.down2, unet.down3, unet.up0)):
        if index:
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        stacked = stack(inputs, features.pop(), flow)
        y = block(stacked)
        del stacked
        if index < len(decoder):  # down0, down1 and down2 feed the matching decoder skip
            skips.append(y)
        inputs = (y,)
    y = inputs[0]
    for block in decoder:
        y = block(torch.cat((y, skips.pop()), 1))
    return torch.sigmoid(unet.conv(y))


def _interpolate(  # noqa: PLR0913
    net: MultiScaleFlow,
    img0: torch.Tensor,
    img1: torch.Tensor,
    t: torch.Tensor,
    cache: Cache | None,
    *,
    fused: bool = False,
    stream_warps: bool = False,
) -> torch.Tensor:
    """`MultiScaleFlow.forward(cat(img0, img1), t)[3]` for float32 `[B, 3, H, W]` frames, `t` `[B, 1, 1, 1]`."""
    b = img0.shape[0]
    dtype = next(net.parameters()).dtype
    cl = net.unet.conv.weight.is_contiguous(memory_format=torch.channels_last) and img0.is_cuda
    fmt = torch.channels_last if cl else torch.contiguous_format  # the convolutions' inputs in NHWC on CUDA
    img0, img1 = (x.contiguous(memory_format=fmt) for x in (img0, img1))
    af, mf = _backbone(net.feature_bone, img0.to(dtype), img1.to(dtype), cache)
    flow = mask = warped0 = warped1 = None
    for i in range(net.flow_num_stage):
        m = mf[-1 - i]
        if m is None:
            msg = "EMA-VFI's heads read attention stages"
            raise ValueError(msg)
        tt = t.to(m.dtype)
        feat = torch.cat((tt * m[:b], (1 - tt) * m[b:], af[-1 - i][:b], af[-1 - i][b:]), 1)
        hd = cast("Head", net.block[i])
        if flow is None or mask is None or warped0 is None or warped1 is None:
            flow, mask = _head(hd, feat, (img0, img1), None, fmt)
        else:
            flow_, mask_ = _head(hd, feat, (img0, img1, warped0, warped1, mask), flow, fmt)
            flow, mask = flow + flow_, mask + mask_
            del flow_, mask_
        del m, feat
        warped0, warped1 = (ema_warp.pair if fused else _warp_pair)(img0, img1, flow)
    if flow is None or mask is None or warped0 is None or warped1 is None:
        msg = "EMA-VFI needs at least one head"
        raise ValueError(msg)
    parts = (img0, img1, warped0, warped1, mask, flow)
    if fused:
        del mf
        tmp = _refine_warped(net, parts, af, ema_warp.stack)
    elif stream_warps:
        del mf

        def stack(inputs: tuple[torch.Tensor, ...], features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
            left = _warp(features[:b], coords[:, :2]).to(dtype, memory_format=fmt)
            right = _warp(features[b:], coords[:, 2:4]).to(dtype, memory_format=fmt)
            normalized = tuple(p.to(dtype, memory_format=fmt) for p in inputs)
            return torch.cat((*normalized, left, right), 1)

        tmp = _refine_warped(net, parts, af, stack)
    else:
        c0, c1, fl = [], [], flow
        for i, x in enumerate(af):  # MultiScaleFlow.warp_features
            if i:
                fl = F.interpolate(fl, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
            c0.append(_warp(x[:b], fl[:, :2]).to(dtype, memory_format=fmt))
            c1.append(_warp(x[b:], fl[:, 2:4]).to(dtype, memory_format=fmt))
        tmp = net.unet(*(p.to(dtype, memory_format=fmt) for p in parts), c0, c1)
    res = tmp[:, :3].float() * 2 - 1
    merged = torch.lerp(warped1, warped0, torch.sigmoid(mask))  # warped0 * mask + warped1 * (1 - mask)
    return torch.clamp(merged + res, 0, 1)


def _frame(  # noqa: PLR0913
    net: MultiScaleFlow,
    pair: tuple[torch.Tensor, torch.Tensor],
    t: torch.Tensor,
    cache: Cache | None,
    *,
    tta: bool,
    fused: bool = False,
    stream_warps: bool = False,
) -> torch.Tensor:
    """`Model.inference` on RGB frames: BGR in and out; `tta` = `fast_TTA` (the pair turned by 180 degrees, mean)."""
    i0, i1 = pair[0].flip(1), pair[1].flip(1)
    if not tta:
        return _interpolate(net, i0, i1, t, cache, fused=fused, stream_warps=stream_warps).flip(1)
    b = i0.shape[0]
    both0, both1 = torch.cat((i0, i0.flip(2, 3))), torch.cat((i1, i1.flip(2, 3)))
    pred = _interpolate(net, both0, both1, t.repeat(2, 1, 1, 1), cache, fused=fused, stream_warps=stream_warps)
    return ((pred[:b] + pred[b:].flip(2, 3)) / 2).flip(1)


def _loaded(ctx: Context) -> MultiScaleFlow:
    """Upstream's network with the variant's checkpoint (float32, CPU)."""
    backbone, flow = _config(ctx.params)
    net = MultiScaleFlow(MotionFormer(**backbone), **flow)
    sd = {  # Trainer.Model.load_model's convert: DDP prefixes, no cached shift masks
        k.removeprefix("module."): v for k, v in ctx.state_dict().items() if "attn_mask" not in k and "HW" not in k
    }
    check_fits(net.state_dict(), sd, f"{ctx.variant.ref} checkpoint")
    net.load_state_dict(sd)
    return net.eval()


def _net(ctx: Context) -> MultiScaleFlow:
    net = _loaded(ctx).to(ctx.device, getattr(torch, ctx.dtype))
    return net.to(memory_format=torch.channels_last) if net.unet.conv.weight.is_cuda else net


class Portable(Runner):
    """Plain PyTorch in the card's dtype (the flow, the mask and the warps in float32)."""

    dtypes = ("float32", "float16")
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.cache: Cache = {}

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch."""
        t = torch.full((image1.shape[0], 1, 1, 1), timestep, device=image1.device)
        return {"frame": self._run(image1, image2, t)}

    def _run(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return _frame(
            self.net, (image1, image2), t, self.cache, tta=self.ctx.params.tta, stream_warps=_eager_cuda(image1)
        )

    def module(self) -> torch.nn.Module:
        """The network; returns the middle frame (time 0.5)."""
        tta = self.ctx.params.tta

        def run(net: MultiScaleFlow, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
            t = torch.full((image1.shape[0], 1, 1, 1), 0.5, device=image1.device)
            return _frame(net, (image1, image2), t, None, tta=tta)

        return Forward(self.net, run)


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation, fuse_bias=True)  # ty: ignore[invalid-argument-type]


def _hacc_deconv(deconv: torch.nn.ConvTranspose2d, x: torch.Tensor) -> torch.Tensor:
    y = cudnn_hacc.deconv_half_acc(x, deconv.weight, deconv.padding, deconv.stride)  # ty: ignore[invalid-argument-type]
    return y.add_(deconv.bias.view(1, -1, 1, 1))  # ty: ignore[unresolved-attribute]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convs and transposed convs (`cudnn_hacc`), one CUDA graph per shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the network, route the dense convolutions through `cudnn_hacc`."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        ema_warp.load(ctx.device)
        for m in self.net.modules():
            if isinstance(m, torch.nn.Conv2d) and base_type(m) is torch.nn.Conv2d and m.groups == 1:
                override(m, _hacc_conv)
            elif isinstance(m, torch.nn.ConvTranspose2d) and base_type(m) is torch.nn.ConvTranspose2d:
                override(m, _hacc_deconv)  # Unet's up blocks: kernel 4, stride 2, padding 1, groups 1
        after = cudnn_hacc.capture_after()
        self.graphs = (
            GraphCache(self._eager, graphs=1, capture_after=after, capture_warmup=False)
            if ctx.params.cuda_graphs
            else None
        )

    def _eager(self, image1: torch.Tensor, image2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return _frame(self.net, (image1, image2), t, self.cache, tta=self.ctx.params.tta, fused=True)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of a padded batch (the time is a graph input: one graph for every time)."""
        t = torch.full((image1.shape[0], 1, 1, 1), timestep, device=image1.device)
        key = (image1.device, tuple(image1.shape))
        out = None if self.graphs is None else self.graphs.run(key, image1, image2, t)
        return {"frame": self._eager(image1, image2, t) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graph and its memory pool."""
        if self.graphs is not None:
            self.graphs.release()


def _window_constants(window: int, h: int, w: int, shift: int) -> tuple[np.ndarray, np.ndarray | None]:
    """`_constants` of one frame for `mlx_net`: the position windows `[nW, N, 2]` and the mask `[nW, N, N]` or None."""
    win, sh = (window, window), (shift, shift)
    gx = torch.linspace(-1.0, 1.0, w).view(1, 1, w).expand(1, h, w)
    gy = torch.linspace(-1.0, 1.0, h).view(1, h, 1).expand(1, h, w)
    cor = window_partition(_rolled(_padded(torch.stack((gx, gy), -1), win), sh, -1), win)
    mask = _mask(h, w, win, sh)
    return cor.numpy(), None if mask is None else mask.numpy()


class MlxPortable(Runner):
    """Apple MLX (`device="mlx"`): the network in `mlx.core` operations and two Metal kernels (`mlx_net`).

    float32, or float16 features and convolutions with the flow, the mask and the warps in float32 (as `Portable`).
    `tta` runs its two passes one after the other.
    """

    Params = Params
    batch = 1

    def __init__(self, ctx: Context) -> None:
        """Convert or read the cached weights and build the MLX network."""
        super().__init__(ctx)
        from fastforks._core.mlx import converted  # noqa: PLC0415 - needs the mlx extra

        from . import mlx_net  # noqa: PLC0415

        p = converted(ctx, mlx_net.LAYOUT, lambda: mlx_net.prepare(_loaded(ctx)))
        params = ctx.params
        constants = partial(_window_constants, params.window)
        self.net = mlx_net.Net(p, params.depths, params.window, ctx.dtype, constants, tta=params.tta)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, timestep: float) -> dict[str, torch.Tensor]:
        """The frame at `timestep` of one padded pair."""
        return {"frame": self.net(image1, image2, timestep)}


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": MlxPortable}
