"""Kernels inference path for IGEV-Stereo (general: any input size, no per-size compilation / tuning).

Port of our IGEV fork's `cuda-kernels` branch (core/inference.py, core/geometry.py `use_kernel`, igev_stereo.py
`forward_fast` kernel branch @ 25aeee1) in its only validated configuration: half=True, half_acc=True,
channels_last=True, kernels=True.

prepare_kernels(model, device), then forward_kernels(model, left, right, iters) (images [1, 3, H, W], 0..255, /32):
  - pure fp16 weights once; BatchNorm kept as an fp32 per-channel affine applied in the conv epilogue (folding it into
    the fp16 weights loses precision); NHWC / NDHWC throughout, layout-preserving norms
  - convs: fp16-accumulate cuDNN graphs (`cudnn_hacc`, also for ConvTranspose2d/3d as conv_dgrad); output channels
    zero-padded to a multiple of 8; bias / BN affine / residual / activation as one `igev` epilogue kernel
  - conv + epilogue as one CUTLASS kernel (`cutlass_conv`) for >= 128 output channels (64/96-channel convs are as
    fast or faster as cuDNN + a separate pass); a geometry the CUTLASS kernel does not support runs on cuDNN
  - ConvGRU without the 3 cats + 6 elementwise passes per level: merged z|r conv whose epilogue writes z and r*h into
    a second [r*h | x] buffer, the q conv updates h in place; pool / interp inputs written straight into the buffers
  - geometry-volume lookup as one kernel on the fp16 level-0 volumes (level 1 pooled on the fly): Midd-F geometry +
    correlation pyramids 2.9 GB -> 0.86 GB
  - context encoder full-resolution stage in row strips with a halo (exact; ~0.8 GB less at Midd-F)

The generic conv / norm machinery (the first three points) is `fastforks.shared.cnn_kernels`; this file holds the
IGEV-specific modules.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, override
from fastforks.kernels.cutlass_conv import ConvEpiArgs
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, conv_fused, cutlass, epilogue, fuse_ok, fused_cfg
from fastforks.shared.igev_blocks.net import extractor, submodule
from fastforks.shared.igev_blocks.net.submodule import build_gwc_volume_fast, context_upsample, disparity_regression

_TAIL = 3  # cutlass_conv affine act 3: relu(res + relu(conv * s + t))
STRIP_PIXELS = 2 * 2**20  # context-encoder full-res stage processed in row strips of about this many pixels
STRIP_HALO = 16  # full-res rows of receptive field of conv1(7x7) + layer1 + layer2 (14), kept even


def _basic_conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # BasicConv: conv -> BN -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _in_stats(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    return mean, var


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


def _motion_encoder_fast(self: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    cor = conv_ep(self.convc1, corr, act=ACT["relu"])
    cor = conv_ep(self.convc2, cor, act=ACT["relu"])
    d = conv_ep(self.convd1, disp, act=ACT["relu"])
    d = conv_ep(self.convd2, d, act=ACT["relu"])
    out = conv_ep(self.conv, torch.cat([cor, d], 1), act=ACT["relu"], keep_pad=True)  # 127 outputs padded to 128
    if out.shape[1] == self.conv.out_channels + 1:
        out[:, -1:].copy_(disp)  # == torch.cat([out, disp], 1) without the copy
        return out
    return torch.cat([out[:, : self.conv.out_channels], disp], 1).contiguous(memory_format=CL)


def _motion_fill_fused(self: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> tuple[str, torch.Tensor, int] | None:
    """Fused motion encoder: the motion conv (127 outputs + 1 zero pad) writes from its epilogue into the x slot of
    the GRU's [h | x] and [r*h | x] buffers; disp then overwrites the pad channel. Returns ("fill", fn, 128) for
    _conv_gru_fast, or None (geometry unsupported)."""
    enc = self
    _, _, hh, ww = disp.shape
    if not fuse_ok(enc.conv, hh, ww) or enc.conv.weight.shape[0] != 128:
        return None
    cor = conv_ep(enc.convc1, corr, act=ACT["relu"])
    d = conv_ep(enc.convd1, disp, act=ACT["relu"])
    cd = torch.empty((1, 128, hh, ww), device=cor.device, dtype=cor.dtype, memory_format=CL)
    # 64-channel convs stay cuDNN + epilogue (+ copy into the cat buffer): fused into the buffer was 1-4 ms slower
    for m, x, c0 in ((enc.convc2, cor, 0), (enc.convd2, d, 64)):
        cd[:, c0 : c0 + 64].copy_(conv_ep(m, x, act=ACT["relu"]))
    del cor, d

    def fill(bufs: list[torch.Tensor], c0: int) -> None:
        out2 = (bufs[1], c0) if len(bufs) > 1 else None
        if conv_fused(enc.conv, cd, None, enc.conv._bias32, ACT["relu"], out=(bufs[0], c0), out2=out2) is None:
            raise RuntimeError("fused motion conv unsupported")
        for b in bufs:
            b[:, c0 + 127 : c0 + 128].copy_(disp)

    return ("fill", fill, 128)


def update_block_fast(
    self: nn.Module, net: nn.Module, ctx: Sequence[Any], corr: torch.Tensor, disp: torch.Tensor, compute_mask: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BasicMultiUpdateBlock.forward for 3 GRU levels, inference (net updated in place)."""
    _conv_gru_fast(self.gru16, net[2], ctx[2], [("pool2x", net[1])])
    _conv_gru_fast(self.gru08, net[1], ctx[1], [("pool2x", net[0]), ("interp", net[2])])
    mf = _motion_fill_fused(self.encoder, disp, corr) if net[0].shape[0] == 1 else None
    if mf is None:
        mf = _motion_encoder_fast(self.encoder, disp, corr)
    _conv_gru_fast(self.gru04, net[0], ctx[0], [mf, ("interp", net[1])])
    d = conv_ep(self.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(self.disp_head.conv2, d)
    mask = conv_ep(self.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None
    return net, mask, delta


# ------------------------------------------------------------------------------------------------ forward
@torch.no_grad()
def forward_kernels(self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """IGEVStereo.forward_fast with the kernels (cuda-kernels branch): fp16 NHWC, disparity state + lookup coordinates
    + softmax / regression / upsampling in fp32, no host syncs in the loop, upsampling features only after the last
    iteration, encoder temporaries freed before the next stage."""
    D = self.args.max_disp // 4
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

    gwc_volume = build_gwc_volume_fast(match_left, match_right, D, 8, True)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
    del gwc_volume, features_left
    prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob

    geo_fn = GeoLookup(match_left, match_right, geo_encoding_volume, self.args.corr_radius)
    del geo_encoding_volume, match_left, match_right

    cnet_list = self.cnet(im1, num_layers=self.args.n_gru_layers)
    del im1
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    ctx = prepare_context(self, inp_list)

    b, _, h, w = init_disp.shape
    disp = init_disp.float()
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16, memory_format=CL)
    mask_feat_4 = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        net_list, mask_feat_4, delta_disp = update_block_fast(
            self.update_block, net_list, ctx, geo_feat, disp_h, itr == iters - 1
        )
        disp = disp + delta_disp.float()

    xspx = self.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(self.spx_gru(xspx).float(), 1)
    return context_upsample(disp * 4.0, spx_pred).unsqueeze(1)


# ------------------------------------------------------------------------------------------------ prepare
def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (on `device`, eval) for forward_kernels."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    model.eval()
    # merged z|r conv weights of the GRUs (before the dtype cast; biases go into the context, see prepare_context)
    for g in (model.update_block.gru04, model.update_block.gru08, model.update_block.gru16):
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
        elif t is extractor.ResidualBlock and isinstance(m.norm1, nn.BatchNorm2d):
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            m.norm2._s, m.norm2._t = bn_affine(m.norm2, m.conv2)
            if m.downsample is not None:
                m.downsample[1]._s, m.downsample[1]._t = bn_affine(m.downsample[1], m.downsample[0])
            override(m, _resblock_forward)
        elif t is extractor.MultiBasicEncoder:
            m.norm1._s, m.norm1._t = bn_affine(m.norm1, m.conv1)
            override(m, _encoder_stem_forward)
    cnn_kernels.patch_norms(model)  # remaining BatchNorms (incl. the backbone's BatchNormAct2d) and bare InstanceNorms
    cnn_kernels.prepare_convs(model)
    cnn_kernels.fuse_narrow(
        model.cnet
    )  # 64 / 96-channel context convs as CUTLASS narrow tiles (+ the ResidualBlock tail)
