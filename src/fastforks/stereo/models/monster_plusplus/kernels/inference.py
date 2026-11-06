"""Kernels inference path for MonSter++ (general: any input size, no per-size compilation / tuning).

Port of our MonSter-plusplus fork's `cuda-kernels` branch (MonSter++/core/monster_fast.py `prepare_monster` /
`forward_fast` @ 9b8fd95, the stereo-zoo configuration: MONSTER_REMP=fast, FASTBB_VIT=hh-g1; the same code as our
MonSter fork's) on the shared kernel libraries.

`prepare_kernels(model, device)`, then `forward_kernels(model, left, right, iters)` (images `[1, 3, H, W]`, 0..255, multiple
of 32):
  - the IGEV kernels machinery (`fastforks.shared.cnn_kernels`) for the whole model incl. the Depth Anything V2 DPT heads:
    pure fp16 NHWC weights, BatchNorm as fp32 affine in the conv epilogue, fp16-accumulate cuDNN convs, conv +
    epilogue as one CUTLASS kernel for >= 128 output channels
  - the DINOv2 ViT-L as the fastbb fp16 ViT (flash attention kernel with fp16 accumulation, fp16-accumulate GEMMs)
  - ConvGRUs with merged z|r convs and fused gates for the IGEV update block and both mixed update blocks; the mixed
    motion encoders read one [geometry lookup | flaw | 0] buffer per view and write straight into the GRU buffers
  - geometry / correlation lookup as one kernel on the fp16 level-0 volumes (level 1 pooled on the fly)
  - REMP in fp16 NHWC with BatchNorm / activations in the conv epilogues; warps, flaws and the final residual add fp32
  - fp32 disparity states, convex upsampling only where the result is used

The group-wise correlation volume is one `igevpp` kernel (fp32 sums, NDHWC output; as the fork's `igev` kernel).
Explicit PyTorch op (not a fallback; the kernel libraries have no such kernel): REMP's LeakyReLU(0.2) after
the conv epilogue (the fork fused it as epilogue activation `leaky02`). Dilated convs run as PyTorch fp16 convs
(cuDNN's fp16-accumulate graphs do not support dilation), as in the fork.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, override
from fastforks.kernels import igevpp
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, conv_fused, cutlass, epilogue, fused_cfg
from fastforks.shared.depth_anything_v2 import use_fast_vit

from ..net import submodule
from ..net.inference import warp
from ..net.monster import compute_scale_shift
from ..net.submodule import build_gwc_volume_fast, context_upsample, disparity_regression


def _h16(x: torch.Tensor) -> torch.Tensor:
    return x.half().contiguous(memory_format=CL)


def _gwc_volume(left: torch.Tensor, right: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """build_gwc_volume_fast(left, right, maxdisp, 8) as one `igevpp` kernel: [B, 8, D, H, W] channels-last-3d fp16."""
    b, c, h, w = left.shape
    if c != 96 or left.dtype != torch.float16:
        return build_gwc_volume_fast(left, right, maxdisp, 8, True)
    left, right = left.contiguous(memory_format=CL), right.contiguous(memory_format=CL)
    out = torch.empty((b, 8, maxdisp, h, w), device=left.device, dtype=left.dtype, memory_format=torch.channels_last_3d)
    dchunk = 32
    n = b * ((maxdisp + dchunk - 1) // dchunk) * h * w
    igevpp.load(left.device).launch("gwc_volume96", *linear(n), left, c, right, c, out, b, h, w, maxdisp, dchunk)
    return out


def _basic_conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # BasicConv: conv -> BN -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _basic_conv_in_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # BasicConv_IN: conv -> InstanceNorm -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    y = conv_ep(self.conv, x)
    if not self.use_in:
        return epilogue(y, None, None, act) if act else y
    return cnn_kernels.instance_norm(y, self.IN.eps, act)


# ------------------------------------------------------------------------------------------------ geometry lookup
class GeoLookup:
    """FastGeoEncodingVolume (num_levels 2) as one kernel: fp16 level-0 geometry and correlation volumes only, the
    kernel pools level 1 on the fly. Output [B, 2 * (C + 1) * (2r + 1), H, W] fp16 as an NHWC view."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, geo_volume: torch.Tensor, radius: int) -> None:
        b, c, d, h, w = geo_volume.shape
        self.shape, self.radius = (b, h, w), radius
        # correlation: [B*H, W1, D] @ [B*H, D, W2] (fp16 in, fp32 accumulate, fp16 out)
        f1 = fmap1.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        f2 = fmap2.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        self.geo = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d).half().contiguous()
        self.corr = torch.bmm(f1.half(), f2.half()).view(b * h * w, w)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        b, h, w = self.shape
        P, C, D = self.geo.shape
        out = torch.empty((P, 2 * (C + 1) * (2 * self.radius + 1)), device=disp.device, dtype=torch.float16)
        dsp = disp.reshape(-1).float().contiguous()
        cnn_kernels.kernels(disp.device).launch(
            "geo_lookup", *linear(P * 2 * (C + 1)), self.geo, C, D, self.corr, self.corr.shape[1], w, dsp, out, P,
            self.radius,
        )  # fmt: skip
        return out.view(b, h, w, -1).permute(0, 3, 1, 2)


# ------------------------------------------------------------------------------------------------ update blocks
def _conv_gru_fast(self: nn.Module, h: int, ctx: tuple[torch.Tensor, torch.Tensor], parts: list[torch.Tensor]) -> int:
    """ConvGRU with fused gates. ctx = (czr, cq): context + conv biases (prepared once); parts: the x inputs as
    tensors, ("pool2x", src) / ("interp", src) (written straight into the [h | x] conv input: no cat, no temporaries)
    or ("fill", fn, width). Updates h in place. The z|r conv reads [h | x] and writes z and r*h from its epilogue
    into a second buffer [r*h | x]; the q conv updates h in place from its epilogue."""
    czr, cq = ctx
    b, Hd, hh, ww = h.shape
    k = cnn_kernels.kernels(h.device)
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[2] if p[0] == "fill" else p[1].shape[1] for p in parts]
    ctot = Hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    bufs = [hx, torch.empty_like(hx)]
    hx[:, :Hd].copy_(h)
    c0 = Hd
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
            if not (src.is_contiguous(memory_format=CL) and cw % 8 == 0):
                raise RuntimeError(f"fused GRU input needs channels-last, channels % 8 == 0 (got {cw})")
            k.launch(
                kind + "_into2", *linear(hh * ww * cw // 8), src, cw, src.shape[2], src.shape[3],
                hx.data_ptr() + 2 * c0, bufs[1].data_ptr() + 2 * c0, ctot, hh, ww, cw,
            )  # fmt: skip
        c0 += cw
    P = h.numel() // Hd
    rhx = bufs[1]
    zr = torch.empty((b, 2 * Hd, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    e0 = ConvEpiArgs(0, 0, czr.data_ptr(), 2 * Hd, zr.data_ptr(), 2 * Hd, hx.data_ptr(), ctot, rhx.data_ptr())
    e1 = ConvEpiArgs(1, 0, cq.data_ptr(), Hd, zr.data_ptr(), 2 * Hd, h.data_ptr(), Hd)
    if not (
        cutlass(0, hx, self.convzr.weight, (hh, ww), e0) and cutlass(fused_cfg(P), rhx, self.convq.weight, (hh, ww), e1)
    ):
        raise RuntimeError("cutlass_conv: GRU conv geometry not supported")
    return h


def _ctx_block(c_list: list[torch.Tensor], blk: nn.Module) -> torch.Tensor:
    """(czr, cq) per level for an update block from the shared context conv outputs (+ that block's GRU biases)."""
    out = []
    for c, g in zip(c_list, (blk.gru04, blk.gru08, blk.gru16), strict=True):
        Hd = c.shape[1] // 3
        bz, br, bq = g.convz.bias.float(), g.convr.bias.float(), g.convq.bias.float()
        czr = epilogue(c[:, : 2 * Hd].contiguous(memory_format=CL), None, torch.cat([bz, br]).contiguous(), 0)
        cq = epilogue(c[:, 2 * Hd :].contiguous(memory_format=CL), None, bq.contiguous(), 0)
        out.append((czr, cq))
    return out


def _motion_fill_fused(enc: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> tuple[str, torch.Tensor, int]:
    """IGEV motion encoder fused: the motion conv (127 outputs + 1 zero pad) writes from its epilogue into the x slot
    of the GRU's [h | x] and [r*h | x] buffers; disp then overwrites the pad channel."""
    _, _, hh, ww = disp.shape
    cor = conv_ep(enc.convc1, corr, act=ACT["relu"])
    d = conv_ep(enc.convd1, disp, act=ACT["relu"])
    cd = torch.empty((1, 128, hh, ww), device=cor.device, dtype=cor.dtype, memory_format=CL)
    for m, x, c0 in ((enc.convc2, cor, 0), (enc.convd2, d, 64)):
        cd[:, c0 : c0 + 64].copy_(conv_ep(m, x, act=ACT["relu"]))
    del cor, d

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        if conv_fused(enc.conv, cd, None, enc.conv._bias32, ACT["relu"], out=(bufs[0], c0), out2=(bufs[1], c0)) is None:
            raise RuntimeError("fused motion conv unsupported")
        for b in bufs:
            b[:, c0 + 127 : c0 + 128].copy_(disp)

    return ("fill", fill, 128)


def _mix_fill(
    enc: nn.Module, d_own: torch.Tensor, c_own: torch.Tensor, d_oth: torch.Tensor, c_oth: torch.Tensor
) -> tuple[str, torch.Tensor, int]:
    """BasicMotionEncoder_mix2 on the [corr | flaw | 0] buffers; the conv / conv_mono outputs and both disparities
    written straight into the GRU's [h | x] / [r*h | x] buffers (no cats)."""
    R = ACT["relu"]
    _, _, hh, ww = d_own.shape
    outs = []
    for c1, c2, dd1, dd2, cv, corr, d in (
        (enc.convc1, enc.convc2, enc.convd1, enc.convd2, enc.conv, c_own, d_own),
        (enc.convc1_mono, enc.convc2_mono, enc.convd1_mono, enc.convd2_mono, enc.conv_mono, c_oth, d_oth),
    ):
        cd = torch.empty((1, 128, hh, ww), device=d.device, dtype=d.dtype, memory_format=CL)
        cd[:, :64].copy_(conv_ep(c2, conv_ep(c1, corr, act=R), act=R))
        cd[:, 64:].copy_(conv_ep(dd2, conv_ep(dd1, d, act=R), act=R))
        outs.append((conv_ep(cv, cd, act=R, keep_pad=True), d))

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        for b in bufs:
            for j, (o, d) in enumerate(outs):
                b[:, c0 + 64 * j : c0 + 64 * j + 63].copy_(o[:, :63])
                b[:, c0 + 64 * j + 63 : c0 + 64 * j + 64].copy_(d)

    return ("fill", fill, 128)


def _heads(blk: nn.Module, net: nn.Module, compute_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
    d = conv_ep(blk.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(blk.disp_head.conv2, d)
    mask = conv_ep(blk.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return mask, delta


def _update_fast(
    blk: nn.Module, net: nn.Module, ctx: Sequence[Any], corr: torch.Tensor, disp: torch.Tensor, compute_mask: bool
) -> torch.Tensor:
    """BasicMultiUpdateBlock.forward (3 levels, inference; net updated in place)."""
    _conv_gru_fast(blk.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(blk.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    _conv_gru_fast(blk.gru04, net[0], ctx[0], [_motion_fill_fused(blk.encoder, disp, corr), ("interp", net[1])])
    return _heads(blk, net, compute_mask)


def _update_mix_fast(
    blk: nn.Module,
    net: nn.Module,
    ctx: Sequence[Any],
    d_own: torch.Tensor,
    c_own: torch.Tensor,
    d_oth: torch.Tensor,
    c_oth: torch.Tensor,
    compute_mask: bool,
) -> torch.Tensor:
    """BasicMultiUpdateBlock_mix2.forward (3 levels, inference; net updated in place)."""
    _conv_gru_fast(blk.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(blk.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    _conv_gru_fast(blk.gru04, net[0], ctx[0], [_mix_fill(blk.encoder, d_own, c_own, d_oth, c_oth), ("interp", net[1])])
    return _heads(blk, net, compute_mask)


def _corr_flaw(
    geo: torch.Tensor, fr0f: torch.Tensor, fl0f: torch.Tensor, disp: torch.Tensor, buf: torch.Tensor
) -> torch.Tensor:
    """[geo lookup | warp(fr0, disp) - fl0 | 0] as one fp16 NHWC buffer (channels padded to a multiple of 8)."""
    g = geo.shape[1]
    if buf is None:
        c = (g + fl0f.shape[1] + 7) // 8 * 8
        buf = torch.zeros((1, c, geo.shape[2], geo.shape[3]), device=geo.device, dtype=torch.float16).contiguous(
            memory_format=CL
        )
    buf[:, :g].copy_(geo)
    buf[:, g : g + fl0f.shape[1]].copy_(warp(fr0f, disp) - fl0f)
    return buf


# ------------------------------------------------------------------------------------------------ REMP
def _cbr(seq_or_basic: nn.Module, x: torch.Tensor, act: int) -> torch.Tensor:
    """conv (no bias) + BN + activation: BN and ReLU in the conv epilogue; LeakyReLU(0.2) (refinement.conv2d) as a
    separate in-place pass."""
    if isinstance(seq_or_basic, nn.Sequential):  # refinement.conv2d: Conv2d, BN, LeakyReLU(0.2)
        c, bn = seq_or_basic[0], seq_or_basic[1]
        return F.leaky_relu_(conv_ep(c, x, bn._s, bn._t, 0), 0.2)
    return conv_ep(seq_or_basic.conv, x, seq_or_basic.bn._s, seq_or_basic.bn._t, act)  # BasicConv_now


def _c2x(m: nn.Module, x: torch.Tensor, rem: int) -> torch.Tensor:
    """Conv2x_now (concat): conv1 (stride 2 / deconv 4x4 s2) + BN + ReLU, cat with rem, conv2 + BN + ReLU."""
    x = _cbr(m.conv1, x, ACT["relu"])
    return _cbr(m.conv2, torch.cat([x, rem], 1), ACT["relu"])


def remp_fast(
    R: int, disp_mono: torch.Tensor, disp_stereo: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    """REMP.forward with fp16 NHWC convs; warps, flaws and the residual disparity add in fp32 (images in [-1, 1])."""
    relu = ACT["relu"]
    fm = _h16(torch.cat([warp(right, disp_mono) - left, left], 1))
    fs = _h16(torch.cat([warp(right, disp_stereo) - left, left], 1))
    x = torch.cat(
        [
            _cbr(R.conv1_mono, fm, 0), _cbr(R.conv2_mono, _h16(disp_mono), 0),
            _cbr(R.conv1_stereo, fs, 0), _cbr(R.conv2_stereo, _h16(disp_stereo), 0),
        ],
        1,
    )  # fmt: skip
    del fm, fs
    x = _cbr(R.conv_start, x, relu)
    u = R.RefinementBlock
    rem0 = x
    x = rem1 = _cbr(u.conv1a, x, relu)
    x = rem2 = _cbr(u.conv2a, x, relu)
    x = rem3 = _cbr(u.conv3a, x, relu)
    x = rem4 = _cbr(u.conv4a, x, relu)
    x = rem3 = _c2x(u.deconv4a, x, rem3)
    x = rem2 = _c2x(u.deconv3a, x, rem2)
    x = rem1 = _c2x(u.deconv2a, x, rem1)
    x = rem0 = _c2x(u.deconv1a, x, rem0)
    x = rem1 = _c2x(u.conv1b, x, rem1)
    x = rem2 = _c2x(u.conv2b, x, rem2)
    x = rem3 = _c2x(u.conv3b, x, rem3)
    x = _c2x(u.conv4b, x, rem4)
    x = _c2x(u.deconv4b, x, rem3)
    x = _c2x(u.deconv3b, x, rem2)
    x = _c2x(u.deconv2b, x, rem1)
    x = _c2x(u.deconv1b, x, rem0)
    del rem0, rem1, rem2, rem3, rem4
    low = R.LFE(x.float().mean((2, 3), keepdim=True).half())  # AP + LFE (1 x 1)
    l0, l1, l2 = R.LMC[0], R.LMC[1], R.LMC[3]  # conv, conv + ReLU, conv + sigmoid
    motif = conv_ep(l2, conv_ep(l1, conv_ep(l0, x), act=relu), act=ACT["sigmoid"])
    x = torch.lerp(low.expand_as(x), x, motif)  # (1 - motif) * low + motif * x
    x = conv_ep(R.final_conv, x.contiguous(memory_format=CL))
    return F.leaky_relu(disp_stereo + x.float(), 0.01)


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_kernels(self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """Monster test-mode forward with the kernels (see the module docstring); returns the disparity [1, 1, H, W]."""
    if image1.shape[0] != 1:
        raise ValueError("MonSter++ kernels path: batch 1 only (fused GRU buffers)")
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = _h16(im1), _h16(im2)

    depth_mono, fml, fmr = self.infer_mono(i1, i2)
    size = (int(depth_mono.shape[-2] * 0.25), int(depth_mono.shape[-1] * 0.25))
    disp_mono_4x = F.interpolate(depth_mono.float(), size=size, mode="bilinear", align_corners=False)
    del depth_mono

    # right view first: only its 1/4 features are kept
    features_right = self.feat_transfer(fmr)
    del fmr
    fr0 = torch.cat((features_right[0], self.stem_4(self.stem_2(i2))), 1)
    del features_right
    match_right = self.desc(self.conv(fr0))
    features_left = self.feat_transfer(fml)
    stem_2x = self.stem_2(i1)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)
    match_left = self.desc(self.conv(features_left[0]))

    gwc_volume = _gwc_volume(match_left, match_right, D)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
    del gwc_volume
    prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob
    fl0 = features_left[0]
    del features_left

    cnet_list = self.feat_transfer_cnet(fml, [stem_16x, stem_8x, stem_4x])
    del fml, stem_8x, stem_16x, stem_4x
    net_list = [torch.tanh(x[0]).contiguous(memory_format=CL) for x in cnet_list]
    c_list = [conv_ep(conv, torch.relu(x[1])) for x, conv in zip(cnet_list, self.context_zqr_convs, strict=True)]
    del cnet_list
    ctx = _ctx_block(c_list, self.update_block)
    ctx_ms = _ctx_block(c_list, self.update_block_mix_stereo)
    ctx_mm = _ctx_block(c_list, self.update_block_mix_mono)
    del c_list
    net_mono = [x.clone() for x in net_list]

    geo_fn = GeoLookup(match_left, match_right, geo_encoding_volume, self.args.corr_radius)
    del geo_encoding_volume, match_left, match_right
    fl0f, fr0f = fl0.float(), fr0.float()  # warps in fp32 (fp16 sampling coordinates are too coarse)
    del fl0, fr0

    b, _, h, w = init_disp.shape
    disp = init_disp.float()
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16)
    dm_h = torch.empty_like(disp_h)
    mask_feat_4 = mask_mono = cs = cm = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        last = itr == iters - 1
        if itr <= iters - 8:
            mask_feat_4, delta_disp = _update_fast(self.update_block, net_list, ctx, geo_feat, disp_h, last)
        else:
            if itr == iters - 7:
                for i in range(b):
                    scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1), disp[i].clone().squeeze(1))
                    disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
            dm_h.copy_(disp_mono_4x)
            cs = _corr_flaw(geo_feat, fr0f, fl0f, disp, cs)
            cm = _corr_flaw(geo_fn(disp_mono_4x), fr0f, fl0f, disp_mono_4x, cm)
            mask_feat_4, delta_disp = _update_mix_fast(
                self.update_block_mix_stereo, net_list, ctx_ms, disp_h, cs, dm_h, cm, last
            )
            mask_mono, delta_mono = _update_mix_fast(
                self.update_block_mix_mono, net_mono, ctx_mm, dm_h, cm, disp_h, cs, last
            )
            disp_mono_4x = disp_mono_4x + delta_mono.float()
        disp = disp + delta_disp.float()

    def up(dl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        xspx = self.spx_2_gru(mask, stem_2x)
        return context_upsample(dl * 4.0, F.softmax(self.spx_gru(xspx).float(), 1)).unsqueeze(1)

    disp_up = up(disp, mask_feat_4)
    disp_mono_up = up(disp_mono_4x, mask_mono)
    return disp_up + remp_fast(self.REMP, disp_mono_up, disp_up, im1, im2)


# ------------------------------------------------------------------------------------------------ prepare
def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (on `device`, eval) for forward_kernels."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    model.eval()
    a = model.args
    if a.n_gru_layers != 3 or a.corr_levels != 2 or list(a.hidden_dims) != [128] * 3:
        raise ValueError("MonSter++ kernels path: 3 GRU levels, 2 correlation levels, hidden dims 128 only")
    # merged z|r conv weights of all GRUs (before the dtype cast; biases go into the context, see _ctx_block)
    for blk in (model.update_block, model.update_block_mix_stereo, model.update_block_mix_mono):
        for g in (blk.gru04, blk.gru08, blk.gru16):
            zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, 3, padding=1, bias=False)
            zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data], 0)
            g.convzr = zr.to(g.convz.weight.device)
    for m in model.modules():
        t = base_type(m)
        if t is submodule.BasicConv:
            if m.use_bn:
                m._s, m._t = bn_affine(m.bn, m.conv)
            override(m, _basic_conv_forward)
        elif t is submodule.BasicConv_IN:
            override(m, _basic_conv_in_forward)
    # the mixed motion encoders read [corr | flaw | 0] buffers padded to a multiple of 8 channels: pad convc1's input
    cin = (2 * (8 + 1) * (2 * a.corr_radius + 1) + 96 + 7) // 8 * 8
    for blk in (model.update_block_mix_stereo, model.update_block_mix_mono):
        for c in (blk.encoder.convc1, blk.encoder.convc1_mono):
            w = c.weight.data
            c.weight.data = torch.cat([w, w.new_zeros(w.shape[0], cin - w.shape[1], *w.shape[2:])], 1)
            c.in_channels = cin
    use_fast_vit(model.mono_encoder, attn="hh")
    cnn_kernels.patch_norms(model)  # remaining BatchNorms (DPT-free; REMP's) and bare InstanceNorms
    cnn_kernels.prepare_convs(model)
    for blk in (model.update_block,):  # the fused motion conv needs 128 (127 + pad) CUTLASS outputs
        if not getattr(blk.encoder.conv, "_fuse", False) or blk.encoder.conv.weight.shape[0] != 128:
            raise RuntimeError("MonSter++ kernels path: motion conv not CUTLASS-fusable")
