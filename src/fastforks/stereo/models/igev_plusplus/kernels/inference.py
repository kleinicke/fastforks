"""Kernels inference path for IGEV++ (general: any input size, no per-size compilation / tuning).

prepare_kernels(model, device), then forward_kernels(model, left, right, iters) (images [1, 3, H, W], 0..255, /32). Built
from our IGEV-Stereo kernels path (`igev_stereo/kernels/inference.py`, copied: families do not import each other):
  - pure fp16 weights once; BatchNorm (context encoder, MobileNetV2) kept as an fp32 per-channel affine in the conv
    epilogue; InstanceNorm (feature decoder, the whole cost aggregation) with fp32 statistics applied by the epilogue
    kernel; NHWC / NDHWC throughout
  - convs (2-D and 3-D, transposed too) as fp16-accumulate cuDNN graphs (`cudnn_hacc`) + one `igev` epilogue kernel,
    or one CUTLASS kernel (`cutlass_conv`) with the epilogue for the wide 2-D convs
  - the group-wise correlation volume (192 disparities) as one `igevpp` kernel (fp32 sums, NDHWC output)
  - the three geometry lookups (small range + its pooled level + the all-pairs correlation, medium range at disp / 2,
    large range at disp / 4) as `igevpp` geo_lookup8 kernels on fp16 volumes: the small range's 162 rows zero-padded
    to 168 feed geometry encoder 0's first conv directly (weights zero-padded / re-ordered to the kernel's channel
    order), the other two ranges write only the 72 geometry rows their encoders read
  - the selective blend of the three geometry encodings + the correlation rows of the motion-encoder input as one
    `igevpp` kernel
  - the motion encoder's input [blended geometry 96 | correlation 18 | 6 zero] built in one buffer (CUTLASS-fusable
    convc1), its last conv written by the CUTLASS epilogue straight into the ConvGRU input buffers
  - ConvGRU with fused gates (merged z|r conv, pool / interp written into the conv input buffer)
  - context encoder full-resolution stage in row strips with a halo (exact)
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, override
from fastforks.kernels import igevpp as igevpp_kernels
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, conv_fused, cutlass, epilogue, fuse_ok, fused_cfg
from fastforks.shared.igev_blocks.net import extractor
from fastforks.shared.igev_blocks.net import submodule as shared_submodule

from ..net import submodule
from ..net.submodule import build_gwc_volume_fast, context_upsample, disparity_regression

_TAIL = 3  # cutlass_conv affine act 3: relu(res + relu(conv * s + t))
STRIP_PIXELS = 2 * 2**20  # context-encoder full-res stage processed in row strips of about this many pixels
STRIP_HALO = 16  # full-res rows of receptive field of conv1(7x7) + layer1 + layer2 (14), kept even
CONVC1_IN = 120  # motion encoder convc1 input: 96 geometry + 18 correlation channels, zero-padded to a multiple of 8


def _basic_conv_in_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # BasicConv_IN: conv -> InstanceNorm -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    y = conv_ep(self.conv, x)
    if not self.use_in:
        return epilogue(y, None, None, act) if act else y
    return cnn_kernels.instance_norm(y, self.IN.eps, act)


def _resblock_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # batch-norm ResidualBlock (context encoder): relu(x + relu(bn2(conv2(relu(bn1(conv1 x))))))
    y = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    if self.downsample is not None:
        x = conv_ep(self.downsample[0], x, self.downsample[1]._s, self.downsample[1]._t, 0)
    if getattr(self.conv2, "_fuse", False) and x.shape[0] == 1 and x.is_contiguous(memory_format=CL):
        o = conv_fused(self.conv2, y, self.norm2._s, self.norm2._t, _TAIL, res=x)  # relu(x + relu(bn(conv)))
        if o is not None:
            return o
    o = conv_ep(self.conv2, y, self.norm2._s, self.norm2._t, ACT["relu"])
    del y
    return epilogue(o, None, None, ACT["relu"], r=x)


def _stem(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    x = self.layer1(x)
    return self.layer2(x)


def _encoder_stem_forward(
    self: nn.Module, x: torch.Tensor, dual_inp: bool = False, num_layers: int = 3
) -> tuple[torch.Tensor, ...]:
    """MultiBasicEncoder with conv1 + norm1 + relu fused. The full-resolution part (conv1, layer1: 64 channels at
    input resolution, then layer2 to 1/2) runs in row strips with a halo: all ops there are local (eval BatchNorm),
    so the result is the same, but only one strip of full-res activations is alive (Midd-F: 3 x 0.78 GB -> ~0.8 GB)."""
    H, W = x.shape[-2:]
    rows = max(64, (STRIP_PIXELS // W) // 2 * 2)
    if rows + 2 * STRIP_HALO >= H or self.downsample != 2:
        x = _stem(self, x)
    else:
        out = None
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            a, b = max(0, r0 - STRIP_HALO), min(H, r1 + STRIP_HALO)
            y = _stem(self, x[:, :, a:b])
            if out is None:
                out = torch.empty(
                    (y.shape[0], y.shape[1], (H + 1) // 2, y.shape[3]), device=y.device, dtype=y.dtype, memory_format=CL
                )
            t = (r0 - a) // 2
            out[:, :, r0 // 2 : (r1 + 1) // 2] = y[:, :, t : t + (r1 + 1) // 2 - r0 // 2]
            del y
        x = out
    x = self.layer3(x)
    outputs04 = [f(x) for f in self.outputs04]
    if num_layers == 1:
        return (outputs04,)
    y = self.layer4(x)
    outputs08 = [f(y) for f in self.outputs08]
    if num_layers == 2:
        return (outputs04, outputs08)
    z = self.layer5(y)
    outputs16 = [f(z) for f in self.outputs16]
    return (outputs04, outputs08, outputs16)


def _feature_att_forward(
    self: nn.Module, cv: torch.Tensor, feat: torch.Tensor
) -> torch.Tensor:  # FeatureAtt (IN): cv * sigmoid(conv1x1(BasicConv(feat))), in place
    a = conv_ep(self.feat_att[1], self.feat_att[0](feat), act=ACT["sigmoid"])
    return cv.mul_(a.unsqueeze(2))


# ------------------------------------------------------------------------------------------------ geometry lookup
class GeoLookup:
    """Combined_Geo_Encoding_Volume as three `igevpp` geo_lookup8 kernels on fp16 [P, C, D] volumes: the small range
    at disp as rows [geo l0 C*K | corr l0 K | geo l1 C*K | corr l1 K | zero pad] (a multiple of 8 channels: the
    geometry encoder's first conv runs with 8-aligned loads), the medium / large ranges at disp / 2, disp / 4 as their
    geo l0 C*K rows only (all their encoders read)."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, vols: list[torch.Tensor], radius: int) -> None:
        b, c, d, h, w = vols[0].shape
        self.shape, self.radius = (b, h, w), radius
        f1 = fmap1.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        f2 = fmap2.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        self.geo = [v.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, v.shape[2]).half().contiguous() for v in vols]
        self.corr = torch.bmm(f1.half(), f2.half()).view(b * h * w, w)

    def __call__(self, disp: torch.Tensor) -> torch.Tensor:
        b, h, w = self.shape
        k = igevpp_kernels.load(disp.device)
        dsp = disp.reshape(-1).float().contiguous()
        K = 2 * self.radius + 1
        outs = []
        for i, g in enumerate(self.geo):
            P, C, D = g.shape
            nl, wc = (2, 1) if i == 0 else (1, 0)
            rows = nl * (C + wc) * K
            rows += -rows % 8
            out = torch.empty((P, rows), device=disp.device, dtype=torch.float16)
            k.launch(
                "geo_lookup8", *linear(P * (nl * (C + wc) + 1)), g, C, D, self.corr, self.corr.shape[1], w, dsp,
                0.5**i, out, rows, P, self.radius, nl, wc,
            )  # fmt: skip
            outs.append(out.view(b, h, w, -1).permute(0, 3, 1, 2))
        return outs


def gwc_volume(left: torch.Tensor, right: torch.Tensor, maxdisp: int) -> torch.Tensor:
    """build_gwc_volume_fast(left, right, maxdisp, 8) as one `igevpp` kernel: [B, 8, D, H, W] channels-last-3d fp16."""
    b, c, h, w = left.shape
    if c != 96 or left.dtype != torch.float16:
        return build_gwc_volume_fast(left, right, maxdisp, 8, True)
    left, right = left.contiguous(memory_format=CL), right.contiguous(memory_format=CL)
    out = torch.empty((b, 8, maxdisp, h, w), device=left.device, dtype=left.dtype, memory_format=torch.channels_last_3d)
    dchunk = 32
    n = b * ((maxdisp + dchunk - 1) // dchunk) * h * w
    igevpp_kernels.load(left.device).launch(
        "gwc_volume96", *linear(n), left, c, right, c, out, b, h, w, maxdisp, dchunk
    )
    return out


# ------------------------------------------------------------------------------------------------ update block
def _conv_gru_fast(self: nn.Module, h: int, ctx: tuple[torch.Tensor, torch.Tensor], parts: list[torch.Tensor]) -> int:
    """ConvGRU with fused gates. ctx = (czr, cq): context + conv biases (prepared once); parts: the x inputs as
    tensors, ("pool2x", src) / ("interp", src) (written straight into the [h | x] conv input: no cat, no temporaries)
    or ("fill", fn, width). Updates h in place. The z|r conv reads [h | x] and writes z and r*h from its epilogue
    into a second buffer [r*h | x]; the q conv updates h in place from its epilogue (v8 layout)."""
    czr, cq = ctx
    b, Hd, hh, ww = h.shape
    k = cnn_kernels.kernels(h.device)
    fz = b == 1 and Hd % 8 == 0
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[2] if p[0] == "fill" else p[1].shape[1] for p in parts]
    ctot = Hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    bufs = [hx, torch.empty_like(hx)] if fz else [hx]
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
            if not (b == 1 and src.is_contiguous(memory_format=CL) and cw % 8 == 0):
                raise RuntimeError(f"fused GRU input needs batch 1, channels-last, channels % 8 == 0 (got {cw})")
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
    P = h.numel() // Hd
    if fz:
        rhx = bufs[1]
        zr = torch.empty((b, 2 * Hd, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
        e0 = ConvEpiArgs(0, 0, czr.data_ptr(), 2 * Hd, zr.data_ptr(), 2 * Hd, hx.data_ptr(), ctot, rhx.data_ptr())
        e1 = ConvEpiArgs(1, 0, cq.data_ptr(), Hd, zr.data_ptr(), 2 * Hd, h.data_ptr(), Hd)
        if not (
            cutlass(0, hx, self.convzr.weight, (hh, ww), e0)
            and cutlass(fused_cfg(P), rhx, self.convq.weight, (hh, ww), e1)
        ):
            raise RuntimeError("cutlass_conv: GRU conv geometry not supported")
        return h
    zr = cnn_kernels.conv_raw(self.convzr, hx)
    k.launch("gru_zr", *linear(P * Hd), zr, czr, h, Hd, hx, ctot, P, Hd)  # r*h -> hx[:, :Hd]
    q = cnn_kernels.conv_raw(self.convq, hx)
    k.launch("gru_update", *linear(P * Hd), q, cq, zr, h, Hd, P, Hd)
    return h


def prepare_context(model: nn.Module, inp_list: list[torch.Tensor]) -> torch.Tensor:
    """inp_list[i] = relu(cnet output) -> per level (czr, cq) with the GRU conv biases added once."""
    out = []
    grus = [model.update_block.gru04, model.update_block.gru08, model.update_block.gru16]
    for i, (x, conv) in enumerate(zip(inp_list, model.context_zqr_convs, strict=False)):
        c = conv_ep(conv, x)  # [B, 3*Hd, h, w]
        Hd = c.shape[1] // 3
        g = grus[i]
        bz, br, bq = g.convz.bias.float(), g.convr.bias.float(), g.convq.bias.float()
        czr = epilogue(c[:, : 2 * Hd].contiguous(memory_format=CL), None, torch.cat([bz, br]).contiguous(), 0)
        cq = epilogue(c[:, 2 * Hd :].contiguous(memory_format=CL), None, bq.contiguous(), 0)
        out.append((czr, cq))
    return out


def _motion_encoder_fast(self: nn.Module, disp: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    cor = conv_ep(self.convc1, x, act=ACT["relu"])
    cor = conv_ep(self.convc2, cor, act=ACT["relu"])
    d = conv_ep(self.convd1, disp, act=ACT["relu"])
    d = conv_ep(self.convd2, d, act=ACT["relu"])
    out = conv_ep(self.conv, torch.cat([cor, d], 1), act=ACT["relu"])
    return torch.cat([out, disp], 1).contiguous(memory_format=CL)


def _motion_fill_fused(
    self: nn.Module, disp: torch.Tensor, x: torch.Tensor
) -> tuple[str, torch.Tensor, torch.Tensor] | None:
    """Fused motion encoder: the motion conv (127 outputs + 1 zero pad) writes from its epilogue into the x slot of
    the GRU's [h | x] and [r*h | x] buffers; disp then overwrites the pad channel. Returns ("fill", fn, 128) for
    _conv_gru_fast, or None (geometry unsupported)."""
    enc = self
    _, _, hh, ww = disp.shape
    n = enc.conv.weight.shape[0]
    if not fuse_ok(enc.conv, hh, ww) or n != enc.conv.out_channels + 1:
        return None
    cor = conv_ep(enc.convc1, x, act=ACT["relu"])
    d = conv_ep(enc.convd1, disp, act=ACT["relu"])
    c2 = enc.convc2.out_channels
    cd = torch.empty((1, enc.conv.in_channels, hh, ww), device=cor.device, dtype=cor.dtype, memory_format=CL)
    cd[:, :c2].copy_(conv_ep(enc.convc2, cor, act=ACT["relu"]))
    cd[:, c2:].copy_(conv_ep(enc.convd2, d, act=ACT["relu"]))
    del cor, d

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        out2 = (bufs[1], c0) if len(bufs) > 1 else None
        if conv_fused(enc.conv, cd, None, enc.conv._bias32, ACT["relu"], out=(bufs[0], c0), out2=out2) is None:
            raise RuntimeError("fused motion conv unsupported")
        for b in bufs:
            b[:, c0 + n - 1 : c0 + n].copy_(disp)

    return ("fill", fill, n)


def _blend_into(e: nn.Module, sw: torch.Tensor, lk0: torch.Tensor, cl: bool, K: int, xbuf: torch.Tensor) -> None:
    """xbuf[:, :96] = sum_i e[i] * sw[:, i] (the selective blend of the geometry encodings), then the two correlation
    levels of the small-range lookup rows (cl geometry channels per level): one `igevpp` kernel."""
    ce = e[0].shape[1]
    b, _, h, w = xbuf.shape
    ts = (*e, sw, lk0, xbuf)
    rows = [t.stride(3) for t in ts]  # NHWC row strides: pixel p of the batch at p * row stride
    dense = all(
        t.stride(1) == 1
        and (h == 1 or t.stride(2) == w * t.stride(3))
        and (b == 1 or t.stride(0) == h * w * t.stride(3))
        for t in ts
    )
    if ce % 8 or rows[0] % 8 or rows[5] % 8 or not dense or len({rows[0], rows[1], rows[2]}) != 1:
        gf = e[0] * sw[:, 0:1] + e[1] * sw[:, 1:2] + e[2] * sw[:, 2:3]
        xbuf[:, :ce].copy_(gf)
        xbuf[:, ce : ce + K].copy_(lk0[:, cl : cl + K])
        xbuf[:, ce + K : ce + 2 * K].copy_(lk0[:, 2 * cl + K : 2 * cl + 2 * K])
        return
    P = b * h * w
    igevpp_kernels.load(xbuf.device).launch(
        "geo_blend", *linear(P * (ce // 8 + 1)), e[0], e[1], e[2], rows[0], ce, sw, rows[3], lk0, rows[4], cl, K,
        xbuf, rows[5], P,
    )  # fmt: skip


def update_block_fast(
    self: nn.Module,
    net: nn.Module,
    ctx: Sequence[Any],
    lk: torch.Tensor,
    sw: torch.Tensor,
    xbuf: torch.Tensor,
    disp: torch.Tensor,
    compute_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BasicMultiUpdateBlock.forward for 3 GRU levels, inference (net updated in place). lk: the three GeoLookup
    outputs; sw: the selective weights [B, 3, H, W]; xbuf: the [B, CONVC1_IN, H, W] motion-encoder input buffer."""
    _conv_gru_fast(self.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(self.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    enc = (self.geo_encoder0, self.geo_encoder1, self.geo_encoder2)
    e = [conv_ep(g.convg2, conv_ep(g.convg1, x, act=ACT["relu"])) for g, x in zip(enc, lk, strict=True)]
    _blend_into(e, sw, lk[0], self._ck, 2 * self.args.corr_radius + 1, xbuf)
    del e
    mf = _motion_fill_fused(self.encoder, disp, xbuf) if net[0].shape[0] == 1 else None
    if mf is None:
        mf = _motion_encoder_fast(self.encoder, disp, xbuf)
    _conv_gru_fast(self.gru04, net[0], ctx[0], [mf, ("interp", net[1])])
    d = conv_ep(self.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(self.disp_head.conv2, d)
    mask = conv_ep(self.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return net, mask, delta


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_kernels(self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """IGEVStereo.forward (test mode) with the kernels: fp16 NHWC, disparity state + lookup coordinates + softmax /
    regression / upsampling in fp32, no host syncs in the loop, upsampling features only after the last iteration,
    each range volume aggregated in turn, encoder temporaries freed before the next stage."""
    a = self.args
    im1 = (2 * (image1 / 255.0) - 1.0).half().contiguous(memory_format=CL)
    im2 = (2 * (image2 / 255.0) - 1.0).half().contiguous(memory_format=CL)

    # right view first: only its 1/4 matching features are kept
    fr = self.feature(im2)
    stem_4y = self.stem_4(self.stem_2(im2))
    match_right = self.desc(self.conv(torch.cat((fr[0], stem_4y), 1)))
    del fr, stem_4y, im2
    features_left = self.feature(im1)
    stem_2x = self.stem_2(im1)
    stem_4x = self.stem_4(stem_2x)
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)
    del stem_4x
    match_left = self.desc(self.conv(features_left[0]))

    cl3 = torch.channels_last_3d
    volume = gwc_volume(match_left, match_right, a.max_disp // 4)
    ranges = (
        (self.cost_agg0, lambda v: v[:, :, : a.s_disp_range].contiguous(memory_format=cl3), a.s_disp_range,
         a.s_disp_interval),
        (self.cost_agg1, lambda v: self.patch0(v[:, :, : a.m_disp_range].contiguous(memory_format=cl3)),
         a.m_disp_range, a.m_disp_interval),
        (self.cost_agg2, self.patch1, a.l_disp_range, a.l_disp_interval),
    )  # fmt: skip
    geo, agg = [], []
    for cost_agg, select, depth, interval in ranges:
        g = cost_agg(select(volume), features_left)
        prob = F.softmax(self.classifier(g).squeeze(1).float(), dim=1)
        agg.append(disparity_regression(prob, depth, interval))
        geo.append(g)
        del prob
    del volume
    disp_feature = self.disp_conv(torch.cat(agg, 1).half().contiguous(memory_format=CL))
    sc = self.selective_conv
    y = sc[1](sc[0](torch.cat((features_left[0], disp_feature), 1)))
    sw = conv_ep(sc[2], y, act=ACT["sigmoid"])
    del disp_feature, features_left, y

    geo_fn = GeoLookup(match_left, match_right, geo, a.corr_radius)
    del geo, match_left, match_right

    cnet_list = self.cnet(im1, num_layers=a.n_gru_layers)
    del im1
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    ctx = prepare_context(self, inp_list)

    disp = agg[0]
    b, _, h, w = disp.shape
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16, memory_format=CL)
    xbuf = torch.empty((b, CONVC1_IN, h, w), device=disp.device, dtype=torch.float16, memory_format=CL).zero_()
    mask_feat_4 = None
    for itr in range(iters):
        lk = geo_fn(disp)
        disp_h.copy_(disp)
        net_list, mask_feat_4, delta_disp = update_block_fast(
            self.update_block, net_list, ctx, lk, sw, xbuf, disp_h, itr == iters - 1
        )
        disp = disp + delta_disp.float()

    xspx = self.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(self.spx_gru(xspx).float(), 1)
    return context_upsample(disp * 4.0, spx_pred)


# ------------------------------------------------------------------------------------------------ prepare
def _scatter_in(m: nn.Module, n_in: int, spans: list[torch.Tensor]) -> None:
    """Give 1x1 conv `m` `n_in` input channels: zero weights except spans (dst, src, width) copied from the old ones."""
    w = m.weight.data
    nw = w.new_zeros((w.shape[0], n_in, *w.shape[2:]))
    for dst, src, n in spans:
        nw[:, dst : dst + n] = w[:, src : src + n]
    m.weight = nn.Parameter(nw)
    m.in_channels = n_in


def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (on `device`, eval) for forward_kernels."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    igevpp_kernels.load(device)
    model.eval()
    ub = model.update_block
    # merged z|r conv weights of the GRUs (before the dtype cast; biases go into the context, see prepare_context)
    for g in (ub.gru04, ub.gru08, ub.gru16):
        zr = nn.Conv2d(g.convz.in_channels, 2 * g.convz.out_channels, 3, padding=1, bias=False)
        zr.weight.data = torch.cat([g.convz.weight.data, g.convr.weight.data], 0)
        g.convzr = zr.to(g.convz.weight.device)
    # geometry encoders read the geo_lookup rows [geo l0 C*K | corr l0 K | geo l1 C*K | corr l1 K] directly
    K = 2 * model.args.corr_radius + 1
    ck = model.classifier.in_channels * K
    rows = 2 * (ck + K)
    _scatter_in(ub.geo_encoder0.convg1, rows + (-rows % 8), [(0, 0, ck), (ck + K, ck, ck)])
    ub._ck = ck  # geometry channels per level in the small-range lookup rows (_blend_into)
    _scatter_in(ub.encoder.convc1, CONVC1_IN, [(0, 0, ub.encoder.convc1.in_channels)])
    for m in model.modules():
        t = base_type(m)
        if t in (submodule.BasicConv, shared_submodule.BasicConv_IN):
            override(m, _basic_conv_in_forward)
        elif t is submodule.FeatureAtt:
            override(m, _feature_att_forward)
        elif t is extractor.ResidualBlock and isinstance(m.norm1, nn.BatchNorm2d):
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            m.norm2._s, m.norm2._t = bn_affine(m.norm2, m.conv2)
            if m.downsample is not None:
                m.downsample[1]._s, m.downsample[1]._t = bn_affine(m.downsample[1], m.downsample[0])
            override(m, _resblock_forward)
        elif t is extractor.MultiBasicEncoder:
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            override(m, _encoder_stem_forward)
    cnn_kernels.patch_norms(model)  # remaining BatchNorms (incl. the backbone's BatchNormAct2d) and bare InstanceNorm2d
    cnn_kernels.prepare_convs(model)
    cnn_kernels.fuse_narrow(
        model.cnet
    )  # 64 / 96-channel context convs as CUTLASS narrow tiles (+ the ResidualBlock tail)
