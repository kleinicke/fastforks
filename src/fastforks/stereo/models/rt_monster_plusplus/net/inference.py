"""Portable inference path for RT-MonSter++ (fastforks; plain PyTorch ops only, exports to ONNX / TensorRT).

Port of our MonSter-plusplus fork's general path (`RT-MonSter++/core/rt_fast.py`, attempt/general-fused 3baec33):

prepare(model, half, channels_last): fp16 weights once (CUDA; autocast would re-cast every fp32 weight in every
  call), NHWC weights (cuDNN NHWC convolutions), the Depth Anything V2 ViT attention as fused
  `scaled_dot_product_attention` (same maths, no N x N matrix).
forward_fast(model, image1, image2): upstream's test-mode forward (coarse-to-fine 1/16 -> 1/8 -> 1/4, 1 / 1 / 2
  iterations) under fp16 autocast (fp16 only), including the mono branch (upstream forces fp32 there); disparity
  states, the geometry / correlation lookups, the mono scale / shift fit, the warps and the upsampling softmax stay
  fp32. Only the outputs used in test mode are computed. For export and `torch.compile` without graph breaks, the
  scale / shift fit is the same least-squares problem written with masked sums and a closed-form 2 x 2 solve (no
  boolean indexing, no `.item()`), and the lookups call `grid_sample` without upstream's host-side `torch.unique`
  assertion; both are the same maths (rounding-level differences).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforks.shared.depth_anything_v2 import use_sdpa_attention
from fastforks.shared.monster_fit import fit as _fit

from .geometry import Combined_Geo_Encoding_Volume
from .monster import get_cur_disp_range_samples
from .submodule import build_gwc_volume, build_gwc_volume_selective, context_upsample, context_upsample_2x, \
    disparity_regression


@torch.no_grad()
def prepare(model, half=True, channels_last=True):
    model.eval()
    use_sdpa_attention(model.mono_encoder)
    if half:
        model.half()
    if channels_last:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    model.inf_half, model.inf_cl = half, channels_last
    return model


def _sample(vol, x):
    """bilinear_sampler(vol, [x, 0]) of utils.py: vol [N, C, 1, W], x [N, 1, K, 1] pixels; border, align_corners."""
    gx = 2 * x / (vol.shape[-1] - 1) - 1
    return F.grid_sample(vol, torch.cat([gx, torch.zeros_like(gx)], -1), padding_mode='border', align_corners=True)


def _lookup(geo, disp, coords, ndisps=None):
    """Combined_Geo_Encoding_Volume.__call__ (same maths, without the per-call host-side unique() check)."""
    r = geo.radius
    b, _, h, w = disp.shape
    dx = torch.linspace(-r, r, 2 * r + 1, device=disp.device).view(1, 1, 2 * r + 1, 1)
    if ndisps is not None:  # local volume: around its centre bin
        x0 = (dx + float(ndisps // 2)).expand(b * h * w, 1, 2 * r + 1, 1)
    else:
        x0 = dx + disp.reshape(b * h * w, 1, 1, 1)
    g = _sample(geo.geo_volume, x0).view(b, h, w, -1)
    c = _sample(geo.init_corr, coords.reshape(b * h * w, 1, 1, 1) - disp.reshape(b * h * w, 1, 1, 1) + dx)
    return torch.cat([g, c.view(b, h, w, -1)], -1).permute(0, 3, 1, 2).contiguous().float()


def _nchw32(x):
    """fp32 NCHW copy: `grid_sample` / `repeat` read a channels-last map uncoalesced (the kernels path keeps the
    features channels-last; portable's InstanceNorm returns NCHW). Several times slower on an A10 otherwise."""
    return x.to(torch.float32, memory_format=torch.contiguous_format)


def warp(img, disp):
    """warp.disp_warp(img, disp)[0] (border padding, same coordinate formula, fp32) without the unused valid mask."""
    b, _, h, w = img.shape
    xs = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, w)
    gx = 2 * ((xs - disp[:, 0].float()) / (w - 1)) - 1
    gy = (2 * (torch.arange(h, device=img.device, dtype=torch.float32) / (h - 1)) - 1).view(1, h, 1).expand_as(gx)
    return F.grid_sample(img.float(), torch.stack([gx, gy], -1), mode="bilinear", padding_mode="border",
                         align_corners=False)  # upstream: the default (False), not passed


def _f32(dev):
    return torch.autocast(dev, enabled=False)


def _up4(self, disp, mask_feat, stem_2x):
    xspx = self.spx_2_gru(mask_feat, stem_2x)
    spx = F.softmax(self.spx_gru(xspx).float(), 1)
    with _f32(disp.device.type):
        return context_upsample(disp.float() * 4., spx).unsqueeze(1)


def _up2(self, disp, mask_feat, stem, i):
    xspx = self.spx_gru_2x[i](mask_feat, stem)
    spx = F.softmax(self.spx_gru_list[i](xspx).float(), 1)
    with _f32(disp.device.type):
        return context_upsample_2x(disp.float() * 2., spx).unsqueeze(1)


@torch.no_grad()
def forward_fast(self, image1, image2):
    """Test-mode forward after `prepare` (see the module docstring); returns the disparity [B, 1, H, W] (fp32)."""
    mf = torch.channels_last if self.inf_cl else torch.contiguous_format
    with torch.autocast(image1.device.type, dtype=torch.float16, enabled=self.inf_half):
        return _forward(self, image1.contiguous(memory_format=mf), image2.contiguous(memory_format=mf))


def _forward(self, image1, image2):
    dev = image1.device.type
    depth_mono, features_mono_left, features_mono_right = self.infer_mono(image1, image2)
    depth_mono = depth_mono.float()
    image1 = (2 * (image1 / 255.0) - 1.0)
    image2 = (2 * (image2 / 255.0) - 1.0)
    size = (image1.shape[-2], image1.shape[-1])
    disp_mono_list = [F.interpolate(depth_mono, size=tuple(x // s for x in size), mode='bilinear', align_corners=False)
                      for s in (16, 8, 4)]

    features_left = self.feat_transfer(features_mono_left)
    features_right = self.feat_transfer(features_mono_right)
    image = torch.cat([image1, image2], 0)
    stem_2x = self.stem_2(image)
    stem_4x = self.stem_4(stem_2x)
    stem_8x = self.stem_8(stem_4x)
    stem_16x = self.stem_16(stem_8x)
    stem_2x, stem_2y = torch.chunk(stem_2x, 2, dim=0)
    stem_4x, stem_4y = torch.chunk(stem_4x, 2, dim=0)
    stem_8x, stem_8y = torch.chunk(stem_8x, 2, dim=0)
    stem_16x, stem_16y = torch.chunk(stem_16x, 2, dim=0)
    stem_x_list = [stem_16x, stem_8x, stem_4x, stem_2x]
    stem_y_list = [stem_16y, stem_8y, stem_4y, stem_2y]
    iters_list = [1, 1, 2]  # upstream's test-mode schedule (independent of `iters`)

    cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]  # (upstream applies relu twice: idempotent)
    inp_list = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, self.context_zqr_convs)]

    disp_next = mask_feat = disp = None
    for i in range(3):
        feat_left = torch.cat((features_left[-1 - i], stem_x_list[i]), 1)
        feat_right = torch.cat((features_right[-1 - i], stem_y_list[i]), 1)
        match_left = self.desc(self.conv_list[i](feat_left))
        match_right = self.desc(self.conv_list[i](feat_right))
        if i == 0:
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp // 16, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
            geo_encoding_volume = self.cost_agg[i](gwc_volume)
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
            cur_disp = disparity_regression(prob, self.args.max_disp // 16)
            del prob
        else:
            with _f32(dev):
                drs = get_cur_disp_range_samples(cur_disp=disp_next.squeeze(1), ndisp=self.ndisps[i - 1],
                                                 disp_interval_pixel=self.disp_interval_pixel[i - 1],
                                                 shape=[match_left.shape[0], match_left.shape[2], match_left.shape[3]])
                gwc_volume = build_gwc_volume_selective(_nchw32(match_left), _nchw32(match_right), drs,
                                                        self.ndisps[i - 1], 8)
            gwc_volume = self.corr_stem(gwc_volume.to(match_left.dtype))
            gwc_volume = self.corr_feature_att_list[i](gwc_volume, feat_left)
            geo_encoding_volume = self.cost_agg[i](gwc_volume)
            cur_disp = disp_next
        del gwc_volume
        with _f32(dev):
            geo_fn = Combined_Geo_Encoding_Volume(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                                                  radius=self.args.corr_radius[i])
        b, c, h, w = match_left.shape
        coords = torch.arange(w, device=match_left.device, dtype=torch.float32).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        nd = self.ndisps[i - 1] if i > 0 else None
        disp = cur_disp.float()
        for itr in range(iters_list[i]):
            with _f32(dev):
                geo_feat = _lookup(geo_fn, disp, coords, nd)
                disp_mono_list[i] = _fit(disp_mono_list[i].float(), disp)
            kw = dict(iter16=(self.args.n_gru_layers == 3), iter08=(self.args.n_gru_layers >= 2 and i >= 1),
                      iter04=(self.args.n_gru_layers >= 1 and i >= 2))
            if i == 2 and itr == iters_list[2] - 1:  # last 1/4 iteration: dual-branch update (its mono half is unused)
                with _f32(dev):
                    fr, fl = _nchw32(feat_right), feat_left.float()
                    flaw_mono = warp(fr, disp_mono_list[i]) - fl
                    flaw_stereo = warp(fr, disp) - fl
                    geo_feat_mono = _lookup(geo_fn, disp_mono_list[i], coords, nd)
                net_list, mask_feat, delta_disp = self.update_block_mix_stereo(
                    net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_list[i], geo_feat_mono,
                    iter16=False, iter08=False)
            else:
                net_list, mask_feat, delta_disp = self.update_block_list[i](net_list, inp_list, geo_feat, disp,
                                                                            disp_mono_list[i], **kw)
            disp = disp + delta_disp.float()
        disp_next = _up2(self, disp, mask_feat, stem_x_list[i + 1], i) if i < 2 else None
    return _up4(self, disp, mask_feat, stem_2x)
