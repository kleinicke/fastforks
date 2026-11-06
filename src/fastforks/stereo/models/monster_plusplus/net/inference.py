"""Portable inference path for MonSter++ (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our MonSter-plusplus fork's upstream-friendly tier (`MonSter++/core/monster_fast.py` `prepare_friendly` /
`forward_friendly`, the same code as our MonSter fork's) on the IGEV-family `fold_for_inference`:

fold_for_inference(model, half, channels_last):
  - pure fp16 (or fp32) weights once; BatchNorm statistics kept as an fp32 per-channel affine applied to the conv
    output (folding them into fp16 weights loses precision); layout-preserving (NHWC / NDHWC) norms
  - the Depth Anything V2 ViT attention as fused `scaled_dot_product_attention` (same maths, no N x N matrix)
forward_fast(model, image1, image2, iters): upstream's test-mode schedule (IGEV updates for itr <= iters - 8, then
  mixed stereo / mono updates, mono scale / shift fit once at itr == iters - 7, REMP once at the end) with
  - disparity states, lookup coordinates, warps, flaws, softmax / regression / upsampling and REMP's residual add in
    fp32 (fp16 sampling coordinates are too coarse)
  - the geometry / correlation lookup by gathers on fp32 / fp16 volumes instead of grid_sample (+ its host sync)
  - upsampling only where the result is used (upstream upsamples the mono disparity in every mixed iteration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforks._core.patch import base_type, override
from fastforks.shared.depth_anything_v2 import use_sdpa_attention
from fastforks.shared.monster_fit import fit

from . import submodule
from .geometry import FastGeoEncodingVolume
from .monster import compute_scale_shift
from .submodule import build_gwc_volume_fast, context_upsample, disparity_regression


def _bn_affine(m):
    scale = m.weight.float() * torch.rsqrt(m.running_var.float() + m.eps)
    shift = m.bias.float() - m.running_mean.float() * scale
    return scale, shift


def _bn_forward(self, x):
    shp = (1, -1) + (1,) * (x.dim() - 2)
    return x.mul_(self._s.to(x.dtype).view(shp)).add_(self._t.to(x.dtype).view(shp))


def _in_forward(self, x):
    # InstanceNorm2d(affine=False): fp32 statistics, applied in place for fp32 (keeps NHWC)
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    if x.dtype == torch.float32:
        return x.sub_(mean).mul_(torch.rsqrt(var + self.eps))
    return ((x.float() - mean) * torch.rsqrt(var + self.eps)).to(x.dtype)  # fp16: fp32 per element, as F.instance_norm


def _basic_conv_forward(self, x):  # BasicConv / BasicConv_IN: conv -> norm -> LeakyReLU(0.01), in place
    x = self.conv(x)
    if getattr(self, "use_bn", False):
        x = self.bn(x)
    if getattr(self, "use_in", False):
        x = self.IN(x)
    if self.relu:
        x = F.leaky_relu_(x)
    return x


@torch.no_grad()  # the folded affines are constants, not part of an autograd graph
def fold_for_inference(model, half=True, channels_last=True):
    model.eval()
    dt = torch.float16 if half else torch.float32
    for m in model.modules():
        t = base_type(m)
        if t in (nn.BatchNorm2d, nn.BatchNorm3d):
            m._s, m._t = _bn_affine(m)
            override(m, _bn_forward)
        elif t is nn.InstanceNorm2d and not m.affine and not m.track_running_stats:
            override(m, _in_forward)
        elif t in (submodule.BasicConv, submodule.BasicConv_IN):
            override(m, _basic_conv_forward)
    use_sdpa_attention(model.mono_encoder)
    model.to(dt)
    for m in model.modules():  # BN affine stays fp32 (cast to the activation dtype at use, tiny)
        if hasattr(m, "_s"):
            m._s, m._t = m._s.float(), m._t.float()
    if channels_last:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    model.inf_dtype, model.inf_cl = dt, channels_last
    return model


def warp(img, disp):
    """warp.disp_warp(img, disp)[0] (border padding, same coordinate formula, fp32) without the unused valid mask."""
    b, _, h, w = img.shape
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, w)
    gx = 2 * ((xs - disp[:, 0].float()) / (w - 1)) - 1
    gy = (2 * (torch.arange(h, device=img.device, dtype=torch.float32) / (h - 1)) - 1).view(1, h, 1).expand_as(gx)
    return F.grid_sample(
        img.float(), torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border", align_corners=False
    )  # upstream: the default (False), not passed


def remp(R, disp_mono, disp_stereo, left, right, dt, mf):
    """REMP.forward through the module forwards in `dt`; warps, flaws and the residual add in fp32."""
    fm = torch.cat([warp(right, disp_mono) - left, left], 1).to(dt).contiguous(memory_format=mf)
    fs = torch.cat([warp(right, disp_stereo) - left, left], 1).to(dt).contiguous(memory_format=mf)
    x = torch.cat(
        [R.conv1_mono(fm), R.conv2_mono(disp_mono.to(dt)), R.conv1_stereo(fs), R.conv2_stereo(disp_stereo.to(dt))], 1
    )
    del fm, fs
    x = R.RefinementBlock(R.conv_start(x))
    low = R.LFE(R.AP(x))
    motif = R.LMC(x)
    x = R.final_conv((1 - motif) * low + motif * x)
    return F.leaky_relu(disp_stereo + x.float(), 0.01)


@torch.no_grad()
def forward_fast(self, image1, image2, iters=32):
    """Test-mode forward after fold_for_inference (see the module docstring); returns the disparity [B, 1, H, W]."""
    dt = self.inf_dtype
    mf = torch.channels_last if self.inf_cl else torch.contiguous_format
    D = self.args.max_disp // 4
    im1 = 2 * (image1 / 255.0) - 1.0
    im2 = 2 * (image2 / 255.0) - 1.0
    i1, i2 = (x.to(dt).contiguous(memory_format=mf) for x in (im1, im2))
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

    gwc_volume = build_gwc_volume_fast(match_left, match_right, D, 8, self.inf_cl)
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
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]  # (upstream applies relu twice: idempotent)
    del cnet_list
    inp_list = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]
    net_mono = [x.clone() for x in net_list]

    # fastforks: the kernels runner on Apple MPS sets `geo_volume` (the lookups as Metal kernels)
    geo_fn = getattr(self, "geo_volume", FastGeoEncodingVolume)(
        match_left,
        match_right,
        geo_encoding_volume,
        radius=self.args.corr_radius,
        num_levels=self.args.corr_levels,
        out_dtype=dt,
    )
    del geo_encoding_volume, match_left, match_right
    fl0f, fr0f = fl0.float(), fr0.float()
    del fl0, fr0

    disp = init_disp.float()
    mask_feat_4 = mask_mono = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h = disp.to(dt)
        if itr <= iters - 8:
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp_h)
        else:
            if itr == iters - 7 and torch.compiler.is_exporting():  # fastforks: no linalg.solve / .item() in ONNX
                disp_mono_4x = fit(disp_mono_4x, disp)
            elif itr == iters - 7:
                for i in range(disp.shape[0]):
                    scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1), disp[i].clone().squeeze(1))
                    disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
            flaw_mono = (warp(fr0f, disp_mono_4x) - fl0f).to(dt)
            flaw_stereo = (warp(fr0f, disp) - fl0f).to(dt)
            geo_feat_mono = geo_fn(disp_mono_4x)
            dm_h = disp_mono_4x.to(dt)
            net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(
                net_list, inp_list, flaw_stereo, disp_h, geo_feat, flaw_mono, dm_h, geo_feat_mono
            )
            net_mono, mask_mono, delta_mono = self.update_block_mix_mono(
                net_mono, inp_list, flaw_mono, dm_h, geo_feat_mono, flaw_stereo, disp_h, geo_feat
            )
            disp_mono_4x = disp_mono_4x + delta_mono.float()
        disp = disp + delta_disp.float()

    def up(dl, mask):
        xspx = self.spx_2_gru(mask, stem_2x)
        return context_upsample(dl * 4.0, F.softmax(self.spx_gru(xspx).float(), 1)).unsqueeze(1)

    disp_up = up(disp, mask_feat_4)
    disp_mono_up = up(disp_mono_4x, mask_mono)
    return disp_up + remp(self.REMP, disp_mono_up, disp_up, im1, im2, dt, mf)
