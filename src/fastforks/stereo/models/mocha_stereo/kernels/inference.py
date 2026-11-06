"""Kernels inference path for MoCha-Stereo (general: any input size, no per-size compilation / tuning).

The portable forward (`net/inference.py`) on the shared kernel libraries, after our MoCha fork's `cuda-kernels` tier:

prepare_kernels(model, device), then forward_kernels(model, left, right, iters) (images [1, 3, H, W], 0..255, /32):
  - pure fp16 weights once; BatchNorm kept as an fp32 per-channel affine in the conv epilogue; NHWC / NDHWC
  - convs: fp16-accumulate cuDNN graphs (`cudnn_hacc`) or CUTLASS conv + epilogue (`cutlass_conv`); bias / BN affine /
    InstanceNorm / residual / activation as `igev` epilogue kernels (`fastforks.shared.cnn_kernels`)
  - LSTM: the [h | x] gate-conv input assembled in one buffer (`igev` pool2x_into / interp_into), one merged 4-gate
    conv per level, then the `mocha` lstm_gates kernel (context + gate biases, sigmoid / tanh, state update in place);
    Mogrifier gates as conv + sigmoid epilogue
  - BatchNormAct2d SiLU (EfficientNet backbone) and feature.BasicConv ReLU in the BN / conv epilogue
  - geometry / correlation lookup as one kernel (`igev` geo_lookup) on the fp16 level-0 volumes
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.kernels import linear
from fastforks._core.patch import base_type, override
from fastforks.kernels import mocha as mocha_kernels
from fastforks.shared import cnn_kernels
from fastforks.shared.cnn_kernels import ACT, CL, bn_affine, conv_ep, conv_fused, epilogue

from ..net import extractor, feature, submodule
from ..net.inference import context_upsample, mccv, remp
from ..net.mccv import Mca_Camp
from ..net.submodule import build_gwc_volume_fast, disparity_regression
from ..net.update import pool2x

_TAIL = 3  # cutlass_conv affine act 3: relu(res + relu(conv * s + t))
SIG = ACT["sigmoid"]


def _basic_conv_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:  # BasicConv: conv -> BN -> LeakyReLU(0.01)
    act = ACT["leaky"] if self.relu else 0
    if self.use_bn:
        return conv_ep(self.conv, x, self._s, self._t, act)
    return conv_ep(self.conv, x, None, None, act)


def _feature_conv_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # feature.BasicConv 2-D: conv -> BN -> ReLU
    act = ACT["relu"] if self.relu else 0
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


def _resblock_forward(
    self: nn.Module, x: torch.Tensor
) -> torch.Tensor:  # batch-norm ResidualBlock: relu(x + relu(bn2(conv2(relu(bn1(conv1 x))))))
    y = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    if self.downsample is not None:
        x = conv_ep(self.downsample[0], x, self.downsample[1]._s, self.downsample[1]._t, 0)
    if getattr(self.conv2, "_fuse", False) and x.shape[0] == 1 and x.is_contiguous(memory_format=CL):
        o = conv_fused(self.conv2, y, self.norm2._s, self.norm2._t, _TAIL, res=x)
        if o is not None:
            return o
    o = conv_ep(self.conv2, y, self.norm2._s, self.norm2._t, ACT["relu"])
    del y
    return epilogue(o, None, None, ACT["relu"], r=x)


def _cnet_forward(
    self: nn.Module, x: torch.Tensor, num_layers: int = 3
) -> torch.Tensor:  # MultiBasicEncoder with conv1 + norm1 + relu fused
    x = conv_ep(self.conv1, x, self.norm1._s, self.norm1._t, ACT["relu"])
    x = self.layer3(self.layer2(self.layer1(x)))
    out = [[f(x) for f in self.outputs04]]
    if num_layers > 1:
        y = self.layer4(x)
        out.append([f(y) for f in self.outputs08])
        if num_layers > 2:
            z = self.layer5(y)
            out.append([f(z) for f in self.outputs16])
    return out


class GeoLookup:
    """FastGeoEncodingVolume (num_levels 2) as one kernel: fp16 level-0 geometry and correlation volumes only, the
    kernel pools level 1 on the fly. Output [B, 2 * (C + 1) * (2r + 1), H, W] fp16 as an NHWC view."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, geo_volume: torch.Tensor, radius: int) -> None:
        b, c, d, h, w = geo_volume.shape
        self.shape, self.radius = (b, h, w), radius
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


def _hx_buf(h: torch.Tensor, parts: Sequence[Any]) -> torch.Tensor:
    """[h | parts...] channels-last gate-conv input; a part is a tensor or ("pool2x_into" | "interp_into", src), written straight
    into its channel slice by the `igev` pool2x_into / interp_into kernels (no cat / avg_pool / upsample tensors)."""
    b, hd, hh, ww = h.shape
    widths = [p.shape[1] if isinstance(p, torch.Tensor) else p[1].shape[1] for p in parts]
    ctot = hd + sum(widths)
    hx = torch.empty((b, ctot, hh, ww), device=h.device, dtype=h.dtype, memory_format=CL)
    hx[:, :hd].copy_(h)
    lib = cnn_kernels.kernels(h.device)
    c0 = hd
    for p, cw in zip(parts, widths, strict=True):
        if isinstance(p, torch.Tensor):
            hx[:, c0 : c0 + cw].copy_(p)
        else:
            kind, src = p
            src = src.contiguous(memory_format=CL)
            for i in range(b):
                lib.launch(
                    kind, *linear(hh * ww * cw // 8), src[i], cw, src.shape[2], src.shape[3],
                    hx[i].data_ptr() + 2 * c0, ctot, hh, ww, cw,
                )  # fmt: skip
        c0 += cw
    return hx


def _lstm(g: nn.Module, h: torch.Tensor, ctx: torch.Tensor, parts: Sequence[Any]) -> torch.Tensor:
    """update.LSTM(c=h, h, *ctx, x) in place on h: Mogrifier (lstm16), merged 4-gate conv, `mocha` lstm_gates kernel
    (ctx: context + gate biases; cell input = previous h, as upstream's aliased netC / netH lists)."""
    b, hd, hh, ww = h.shape
    hin = h
    if len(parts) == 1 and parts[0][1].shape[1] == hd:  # Mogrifier: x = 2 sigmoid(Q h) x, h = 2 sigmoid(R x) h
        x = pool2x(parts[0][1]).contiguous(memory_format=CL)
        hin = h.clone()
        for ind in range(g.m.iters):
            if ind % 2 == 0:
                x.mul_(conv_ep(g.m.Q, hin, act=SIG)).mul_(2)
            else:
                hin.mul_(conv_ep(g.m.R, x, act=SIG)).mul_(2)
        parts = [x]
    gates = conv_ep(g.conv4, _hx_buf(hin, parts))
    mocha_kernels.load(h.device).launch(
        "lstm_gates", *linear(b * hh * ww * hd), gates, ctx, None, h, hd, b * hh * ww, hd
    )
    return h


def _motion_encoder(self: nn.Module, disp: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    cor = conv_ep(self.convc1, corr, act=ACT["relu"])
    cor = conv_ep(self.convc2, cor, act=ACT["relu"])
    d = conv_ep(self.convd1, disp, act=ACT["relu"])
    d = conv_ep(self.convd2, d, act=ACT["relu"])
    out = conv_ep(self.conv, torch.cat([cor, d], 1), act=ACT["relu"], keep_pad=True)  # 127 outputs padded to 128
    if out.shape[1] == self.conv.out_channels + 1:
        out[:, -1:].copy_(disp)  # == torch.cat([out, disp], 1) without the copy
        return out
    return torch.cat([out[:, : self.conv.out_channels], disp], 1).contiguous(memory_format=CL)


def update_block(
    blk: nn.Module, net: nn.Module, ctx: Sequence[Any], corr: torch.Tensor, disp: torch.Tensor, compute_mask: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """LSTMMultiUpdateBlock.forward (3 levels, test mode); net updated in place."""
    _lstm(blk.lstm16, net[2], ctx[2], [("pool2x_into", net[1])])
    _lstm(blk.lstm08, net[1], ctx[1], [("pool2x_into", net[0]), ("interp_into", net[2])])
    mf = _motion_encoder(blk.encoder, disp, corr)
    _lstm(blk.lstm04, net[0], ctx[0], [mf, ("interp_into", net[1])])
    del mf
    d = conv_ep(blk.disp_head.conv1, net[0], act=ACT["relu"])
    delta = conv_ep(blk.disp_head.conv2, d)
    return delta, conv_ep(blk.mask_feat_4[0], net[0], act=ACT["relu"]) if compute_mask else None


@torch.no_grad()
def forward_kernels(self: nn.Module, image1: torch.Tensor, image2: torch.Tensor, iters: int = 32) -> torch.Tensor:
    """forward_fast (net/inference.py) on the kernels; returns the disparity [B, 1, H, W] fp32."""
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = (x.half().contiguous(memory_format=CL) for x in (im1, im2))

    fr = self.feature(i2)  # right view: 1/4 matching features, 1/8 and 1/16 for the MCCV
    fr0 = torch.cat((fr[0], self.stem_4(self.stem_2(i2))), 1)
    match_right = self.desc(self.conv(fr0))
    del fr0, i2
    fl = self.feature(i1)
    stem_2x = self.stem_2(i1)
    stem_4x = self.stem_4(stem_2x)
    fl[0] = torch.cat((fl[0], stem_4x), 1)
    del stem_4x
    match_left = self.desc(self.conv(fl[0]))

    gwc_volume = build_gwc_volume_fast(match_left, match_right, D, 8, True)
    gwc_volume = self.corr_stem(gwc_volume)
    gwc_volume = self.corr_feature_att(gwc_volume, fl[0])
    ccv = mccv(self.mccv, fl, fr, CL) if isinstance(self.mccv, Mca_Camp) else self.mccv(fl, fr)
    del fr, fl
    volume = self.cost_agg(gwc_volume, ccv)
    del gwc_volume, ccv
    prob = F.softmax(self.classifier(volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, D)
    del prob
    geo_fn = GeoLookup(match_left, match_right, volume, self.args.corr_radius)
    del volume, match_left, match_right

    cnet_list = self.cnet(i1, num_layers=self.args.n_gru_layers)
    del i1
    net = [torch.tanh(x[0]).contiguous(memory_format=CL) for x in cnet_list]  # hidden states, updated in place
    lstms = (self.update_block.lstm04, self.update_block.lstm08, self.update_block.lstm16)
    ctx = [
        conv_ep(conv, torch.relu(x[1]), None, g._tc)
        for x, conv, g in zip(cnet_list, self.context_zqr_convs, lstms, strict=True)
    ]
    del cnet_list

    disp = init_disp.float()
    b, _, h, w = disp.shape
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=torch.float16, memory_format=CL)
    mask_feat_4 = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        delta, mask_feat_4 = update_block(self.update_block, net, ctx, geo_feat, disp_h, itr == iters - 1)
        disp = disp + delta.float()

    xspx = self.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(self.spx_gru(xspx).float(), 1)
    disp_up = context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
    return disp_up + remp(self.REMP, disp_up, im1, im2, torch.float16, CL)


@torch.no_grad()
def prepare_kernels(model: nn.Module, device: torch.device) -> None:
    """Load the kernel libraries for `device` and patch `model` (fp32, on `device`, eval) for forward_kernels."""
    cnn_kernels.load(device)  # raises if a library or cuDNN is unusable: no silent fallback to PyTorch convs
    mocha_kernels.load(device)
    model.eval()
    for g, conv in zip(
        (model.update_block.lstm04, model.update_block.lstm08, model.update_block.lstm16),
        model.context_zqr_convs,
        strict=True,
    ):
        convs = (g.conv_it, g.conv_ft, g.conv_c_t, g.conv_ot)
        c4 = nn.Conv2d(g.conv_it.in_channels, 4 * g.conv_it.out_channels, 3, padding=1, bias=False)
        c4.weight.data = torch.cat([c.weight.data for c in convs], 0)
        g.conv4 = c4.to(g.conv_it.weight.device)
        g._tc = (conv.bias.float() + torch.cat([c.bias.float() for c in convs])).contiguous()  # context + gate biases
    for m in model.modules():
        t = base_type(m)
        if t is submodule.BasicConv:
            if m.use_bn:
                m._s, m._t = bn_affine(m.bn, m.conv)
            override(m, _basic_conv_forward)
        elif t is feature.BasicConv and not isinstance(m.conv, (nn.Conv3d, nn.ConvTranspose3d)):
            if m.use_bn:
                m._s, m._t = bn_affine(m.bn, m.conv)
            override(m, _feature_conv_forward)
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
            override(m, _cnet_forward)
    cnn_kernels.patch_norms(model)  # remaining BatchNorms (incl. the backbone's BatchNormAct2d) and bare InstanceNorms
    # BatchNormAct2d + SiLU: the activation in the epilogue too ("affine_act8x" / "affine_actx")
    for m in model.modules():
        if base_type(getattr(m, "_post", None)) is nn.SiLU:
            m._act, m._post = ACT["silu"], None
    cnn_kernels.prepare_convs(model)
    cnn_kernels.fuse_narrow(model.cnet)
