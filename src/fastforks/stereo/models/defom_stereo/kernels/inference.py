"""Kernels inference path for DEFOM-Stereo (general: any input size, no per-size compilation / tuning).

Port of our DEFOM fork's `cuda-kernels` branch (core/defom_fast.py @ 0bd7847, DEFOM_FAST=max) on the shared kernel
libraries.

`prepare_kernels(model, device)`, then `forward_kernels(model, left, right, iters, scale_iters)` (images `[1, 3, H, W]`,
0..255, multiple of 32):
  - Depth Anything V2 ViT-L + DPT in bf16 (DPT refinenet1 activations exceed the fp16 range); the ViT runs as the
    fastbb fp16 ViT (flash kernel, fp16-accumulate GEMMs) and hands bf16 tokens on
  - fnet / cnet as fp16 NHWC encoders (`encoder.py`: folded BatchNorm, fastbb instance norm with fused ReLU / residual
    tails, fp16-accumulate cuDNN convs); only the convs on the depth features stay bf16
  - everything from the context outputs on in pure fp16 NHWC: update + scale-update blocks and the context convs on
    `cnn_kernels` convs, merged z|r GRU convs with CUTLASS epilogues (gates and state update fused), pool / interp
    written straight into the GRU inputs, motion conv written into the GRU buffers
  - the correlation volume as one fp32 batched GEMM, its lookup (2 levels x 9 taps, or 8 scales x 5 taps) as one
    kernel writing the fp16 NHWC input of the motion encoder (no grid_sample / cat / permute)
  - fp32 disparity state, convex upsampling only after the last iteration, fnet one view at a time for large inputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks.kernels import dpt_fused
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, conv_ep, conv_fused, cutlass, fuse_ok, fused_cfg
from fastforks.shared.depth_anything_v2 import use_fast_vit

from ..net.utils import get_danv2_io_size
from .encoder import prepare_encoders

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SEQ_FNET_PIXELS = 2_000_000  # fnet runs one view at a time above this input size (lower peak memory, same result)


def _h16(x: torch.Tensor) -> torch.Tensor:
    return x.half().contiguous(memory_format=CL)


# ------------------------------------------------------------------------------------------------ GRU
def _conv_gru_fast(g: torch.Tensor, h: int, ctx: tuple[torch.Tensor, torch.Tensor], parts: list[torch.Tensor]) -> int:
    """ConvGRU with fused gates. ctx = (czr, cq): context + conv biases (prepared once); parts: the x inputs as
    tensors, ("pool2x", src) / ("interp", src) (written straight into the [h | x] conv input: no cat, no temporaries)
    or ("fill", fn, width). Updates h in place. The z|r conv reads [h | x] and writes z and r*h from its epilogue
    into a second buffer [r*h | x]; the q conv updates h in place from its epilogue."""
    czr, cq = ctx
    b, hd, hh, ww = h.shape
    k = cnn_kernels.kernels(h.device)
    fz = b == 1 and hd == 128
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[2] if p[0] == "fill" else p[1].shape[1] for p in parts]
    ctot = hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    bufs = [hx, torch.empty_like(hx)] if fz else [hx]
    hx[:, :hd].copy_(h)
    c0 = hd
    for p, cw in zip(parts, widths, strict=True):
        if not isinstance(p, torch.Tensor) and p[0] == "fill":  # writes its channels into all buffers itself
            p[1](bufs, c0)
            c0 += cw
            continue
        if isinstance(p, torch.Tensor):
            for buf in bufs:
                buf[:, c0 : c0 + cw].copy_(p)
        else:
            kind, src = p
            if not (b == 1 and src.is_contiguous(memory_format=CL) and cw % 8 == 0):
                msg = f"fused GRU input needs batch 1, channels-last, channels % 8 == 0 (got {cw})"
                raise RuntimeError(msg)
            n = linear(hh * ww * cw // 8)
            if fz:  # one pass, both buffers
                k.launch(
                    kind + "_into2", *n, src, cw, src.shape[2], src.shape[3], hx.data_ptr() + 2 * c0,
                    bufs[1].data_ptr() + 2 * c0, ctot, hh, ww, cw,
                )  # fmt: skip
            else:
                k.launch(
                    kind + "_into", *n, src, cw, src.shape[2], src.shape[3], hx.data_ptr() + 2 * c0, ctot, hh, ww, cw
                )
        c0 += cw
    n_pix = h.numel() // hd
    if fz:
        rhx = bufs[1]
        zr = torch.empty((b, 2 * hd, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
        e0 = ConvEpiArgs(0, 0, czr.data_ptr(), 2 * hd, zr.data_ptr(), 2 * hd, hx.data_ptr(), ctot, rhx.data_ptr())
        e1 = ConvEpiArgs(1, 0, cq.data_ptr(), hd, zr.data_ptr(), 2 * hd, h.data_ptr(), hd)
        if not (
            cutlass(0, hx, g.convzr.weight, (hh, ww), e0)
            and cutlass(fused_cfg(n_pix), rhx, g.convq.weight, (hh, ww), e1)
        ):
            msg = "cutlass_conv: GRU conv geometry not supported"
            raise RuntimeError(msg)
        return h
    zr = cnn_kernels.conv_raw(g.convzr, hx)
    k.launch("gru_zr", *linear(n_pix * hd), zr, czr, h, hd, hx, ctot, n_pix, hd)  # r*h -> hx[:, :hd]
    q = cnn_kernels.conv_raw(g.convq, hx)
    k.launch("gru_update", *linear(n_pix * hd), q, cq, zr, h, hd, n_pix, hd)
    return h


def _motion_fill(enc: nn.Module, disp_h: torch.Tensor, corr: torch.Tensor) -> tuple[str, Callable[..., None], int]:
    """Motion encoder; its final conv (127 outputs + 1 zero pad) writes from the epilogue into the x slot of the GRU's
    [h | x] / [r*h | x] buffers, disp overwrites the pad channel. Without the fused conv: a tensor with the same
    channels."""
    _, _, hh, ww = disp_h.shape
    cor = conv_ep(enc.convc1, corr, act=ACT["relu"])
    d = conv_ep(enc.convd1, disp_h, act=ACT["relu"])
    cd = torch.empty((1, 128, hh, ww), device=cor.device, dtype=cor.dtype, memory_format=CL)
    for m, x, c0 in ((enc.convc2, cor, 0), (enc.convd2, d, 64)):
        cd[:, c0 : c0 + 64].copy_(conv_ep(m, x, act=ACT["relu"]))
    del cor, d
    if not fuse_ok(enc.conv, hh, ww):
        out = conv_ep(enc.conv, cd, act=ACT["relu"], keep_pad=True)
        out[:, 127:128].copy_(disp_h)
        return out

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        out2 = (bufs[1], c0) if len(bufs) > 1 else None
        if conv_fused(enc.conv, cd, None, enc.conv._bias32, ACT["relu"], out=(bufs[0], c0), out2=out2) is None:
            msg = "fused motion conv unsupported"
            raise RuntimeError(msg)
        for buf in bufs:
            buf[:, c0 + 127 : c0 + 128].copy_(disp_h)

    return ("fill", fill, 128)


def _update(
    blk: nn.Module, net: nn.Module, ctx: Sequence[Any], corr: torch.Tensor, disp_h: torch.Tensor
) -> torch.Tensor:
    _conv_gru_fast(blk.gru32, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(blk.gru16, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    mf = _motion_fill(blk.encoder, disp_h, corr)
    _conv_gru_fast(blk.gru08, net[0], ctx[0], [mf, ("interp", net[1])])
    d = conv_ep(blk.disp_head.conv1, net[0], act=ACT["relu"])
    return conv_ep(blk.disp_head.conv2, d)


# ------------------------------------------------------------------------------------------------ correlation
class Corr:
    """CorrBlock1D (reg, fp32): all-pairs volume per row (fp32 batched GEMM = upstream einsum) + 2-level pyramid; the
    lookups for the normal (2 levels x (2r+1)) and the scale (len(scale_list) x (2sr+1)) iterations are one kernel."""

    def __init__(
        self, fmap1: torch.Tensor, fmap2: torch.Tensor, radius: int, scale_list: list[torch.Tensor], scale_radius: float
    ) -> None:
        _, d, h, w = fmap1.shape
        f1 = fmap1[0].float().permute(1, 2, 0)  # [H, W, D]
        f2 = fmap2[0].float().permute(1, 0, 2)  # [H, D, W]
        self.c0 = torch.bmm(f1, f2).div_(float(d) ** 0.5)  # [H, W1, W2]
        self.c1 = F.avg_pool2d(self.c0.view(h * w, 1, 1, w), [1, 2], stride=[1, 2]).view(h, w, w // 2)
        self.h, self.w, self.r, self.sr = h, w, radius, scale_radius
        self.scales = torch.tensor(scale_list, device=fmap1.device, dtype=torch.float32)
        nch = (2 * (2 * radius + 1), len(scale_list) * (2 * scale_radius + 1))
        self.cpad = tuple((c + 7) // 8 * 8 for c in nch)  # lookup buffer widths: input channels of convc1, padded

    def __call__(self, disp: torch.Tensor, *, scaling: bool) -> torch.Tensor:
        c = self.cpad[1 if scaling else 0]
        out = torch.empty((1, c, self.h, self.w), device=disp.device, dtype=torch.float16, memory_format=CL)
        p = self.h * self.w
        cnn_kernels.kernels(disp.device).launch(
            "defom_lookup", *linear(p * c), self.c0, self.w, self.c1, self.w // 2, disp.contiguous(), self.w,
            self.scales, self.scales.numel(), out, c, p, self.sr if scaling else self.r, 1 if scaling else 0,
        )  # fmt: skip
        return out


# ------------------------------------------------------------------------------------------------ prepare / forward
def _prep_block(blk: nn.Module, cpad: torch.Tensor) -> None:
    for g in (blk.gru08, blk.gru16, blk.gru32):
        zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, 3, padding=1, bias=False)
        zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data], 0)
        g.convzr = zr.to(g.convz.weight.device)
        g._bzr = torch.cat([g.convz.bias, g.convr.bias]).float().contiguous()
        g._bq = g.convq.bias.float().contiguous()
    c1 = blk.encoder.convc1  # 1x1 on the lookup: input channels zero-padded to the lookup buffer width
    if c1.weight.shape[1] != cpad:
        w = c1.weight.data
        c1.weight.data = torch.cat([w, w.new_zeros(w.shape[0], cpad - w.shape[1], 1, 1)], 1)
        c1.in_channels = cpad


def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (on `device`, eval) for `forward_kernels`."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    model.eval()
    a = model.args
    if not (a.n_gru_layers == 3 and a.n_downsample == 2 and a.corr_levels == 2 and list(a.hidden_dims) == [128] * 3):
        msg = "DEFOM kernels path: 3 GRU levels, n_downsample 2, 2 correlation levels, hidden dims 128 only"
        raise ValueError(msg)
    pretrained = model.defomencoder.depth_anything.pretrained
    use_fast_vit(
        pretrained, out_dtype=torch.bfloat16, fused=dpt_fused.load(device)
    )  # fp16 ViT, bf16 tokens for the bf16 DPT heads
    prepare_encoders(model)  # fnet / cnet: fp16 NHWC, folded BN, fastbb instance norm, hacc convs (convs on DPT: bf16)
    nch = (2 * (2 * a.corr_radius + 1), len(a.scale_list) * (2 * a.scale_corr_radius + 1))
    cpad = tuple((c + 7) // 8 * 8 for c in nch)
    _prep_block(model.update_block, cpad[0])
    _prep_block(model.scale_update_block, cpad[1])
    cnn_kernels.prepare_convs(model.update_block)
    cnn_kernels.prepare_convs(model.scale_update_block)
    cnn_kernels.prepare_convs(model.context_zqr_convs)


def _ctx(c_list: list[torch.Tensor], blk: nn.Module) -> torch.Tensor:
    out = []
    for c, g in zip(c_list, (blk.gru08, blk.gru16, blk.gru32), strict=True):
        czr = (c[:, :256].float() + g._bzr.view(1, -1, 1, 1)).half().contiguous(memory_format=CL)
        cq = (c[:, 256:].float() + g._bq.view(1, -1, 1, 1)).half().contiguous(memory_format=CL)
        out.append((czr, cq))
    return out


@torch.no_grad()
def forward_kernels(
    self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32, scale_iters: int = 8
) -> torch.Tensor:
    """DEFOMStereo.forward (test mode) with the kernels: returns the disparity `[1, 1, H, W]`."""
    a = self.args
    image1 = ((image1 - self.mean) / self.std).contiguous()
    image2 = ((image2 - self.mean) / self.std).contiguous()
    _, _, h, w = image1.shape
    sizes = get_danv2_io_size(h, w, a.n_downsample)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d_features, dfeat1, dfeat2, disp = self.defomencoder([image1, image2], sizes)
        cnet_list = self.cnet(image1, d_features)
        del d_features
        if h * w > SEQ_FNET_PIXELS:  # instance norm is per image: the same result, a lower peak
            fmap1 = self.fnet([image1], [dfeat1])[0]
            del dfeat1
            fmap2 = self.fnet([image2], [dfeat2])[0]
            del dfeat2
        else:
            fmap1, fmap2 = self.fnet([image1, image2], [dfeat1, dfeat2])
            del dfeat1, dfeat2
    net = [_h16(torch.tanh(x[0].float())) for x in cnet_list]
    inp = [_h16(torch.relu(x[1].float())) for x in cnet_list]
    del cnet_list
    c_list = [conv_ep(conv, x) for x, conv in zip(inp, self.context_zqr_convs, strict=True)]
    del inp
    ctx_u, ctx_s = _ctx(c_list, self.update_block), _ctx(c_list, self.scale_update_block)
    del c_list
    corr = Corr(fmap1, fmap2, a.corr_radius, a.scale_list, a.scale_corr_radius)
    del fmap1, fmap2
    disp = disp.float().contiguous()
    lim = 2 ** (a.corr_levels - 1) * a.corr_radius
    for itr in range(iters):
        scaling = itr < scale_iters
        look = corr(disp, scaling=scaling)
        disp_h = _h16(disp)
        if scaling:
            x = _update(self.scale_update_block, net, ctx_s, look, disp_h)
            disp = F.relu6(torch.exp(0.25 * x.float())) * disp
        else:
            x = _update(self.update_block, net, ctx_u, look, disp_h)
            disp = disp + torch.clamp(x.float(), -lim, lim)
    del corr
    m = (self.scale_update_block if iters <= scale_iters else self.update_block).mask
    mask = conv_ep(m[2], conv_ep(m[0], net[0], act=ACT["relu"]))
    return self.upsample_flow(disp, 0.25 * mask.float())
