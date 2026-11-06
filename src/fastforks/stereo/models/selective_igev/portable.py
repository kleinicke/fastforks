"""Plain-PyTorch Selective-IGEV forward (no custom kernels): the exportable inference path.

Mirrors `IGEVStereo.forward` (test mode) on the network prepared by `net.inference.fold_for_inference`: pure fp16 or
fp32 weights and activations, NHWC on CUDA, disparity state, lookup coordinates, softmax / regression and upsampling in
fp32, no host syncs in the update loop, encoder temporaries freed before the next stage. The selective part is
upstream's `BasicSelectiveMultiUpdateBlock` (channel / spatial attention on the context, attention-blended small and
large ConvGRUs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks.shared.igev_blocks.net.submodule import build_gwc_volume_fast, context_upsample, disparity_regression

from .net.geometry import FastGeoEncodingVolume

if TYPE_CHECKING:
    from .net.igev_stereo import IGEVStereo


@torch.no_grad()
def forward_fast(net: IGEVStereo, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
    """Disparity `[B, 1, H, W]` float32 of a padded pair (`left`, `right`: RGB in 0-255)."""
    dt = cast("torch.dtype", net.inf_dtype)  # set by fold_for_inference
    mf: torch.memory_format = torch.channels_last if cast("bool", net.inf_cl) else torch.contiguous_format
    depth = net.args.max_disp // 4
    im1 = (2 * (left / 255.0) - 1.0).to(dt).contiguous(memory_format=mf)
    im2 = (2 * (right / 255.0) - 1.0).to(dt).contiguous(memory_format=mf)

    # right view first: only its 1/4 matching features are kept
    fr = net.feature(im2)
    stem_4y = net.stem_4(net.stem_2(im2))
    match_right = net.desc(net.conv(torch.cat((fr[0], stem_4y), 1)))
    del fr, stem_4y, im2
    features_left = net.feature(im1)
    stem_2x = net.stem_2(im1)
    stem_4x = net.stem_4(stem_2x)
    features_left[0] = torch.cat((features_left[0], stem_4x), 1)
    del stem_4x
    match_left = net.desc(net.conv(features_left[0]))

    gwc_volume = build_gwc_volume_fast(match_left, match_right, depth, 8, cast("bool", net.inf_cl))
    gwc_volume = net.corr_stem(gwc_volume)
    gwc_volume = net.corr_feature_att(gwc_volume, features_left[0])
    geo_encoding_volume = net.cost_agg(gwc_volume, features_left)
    del gwc_volume, features_left
    prob = F.softmax(net.classifier(geo_encoding_volume).squeeze(1).float(), dim=1)
    init_disp = disparity_regression(prob, depth)
    del prob

    # the kernels runner on Apple MPS sets `geo_volume` (the lookups as Metal kernels)
    geo_fn = getattr(net, "geo_volume", FastGeoEncodingVolume)(
        match_left, match_right, geo_encoding_volume,
        radius=net.args.corr_radius, num_levels=net.args.corr_levels, out_dtype=dt,
    )  # fmt: skip
    del geo_encoding_volume, match_left, match_right

    cnet_list = net.cnet(im1, num_layers=net.args.n_gru_layers)
    del im1
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    inp_list = [net.cam(x) * x for x in inp_list]
    att = [net.sam(x) for x in inp_list]

    b, _, h, w = init_disp.shape
    disp = init_disp.float()
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=dt, memory_format=mf)
    mask_feat_4 = None
    for _ in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        net_list, mask_feat_4, delta_disp = net.update_block(net_list, inp_list, geo_feat, disp_h, att)
        disp = disp + delta_disp.float()

    xspx = net.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(net.spx_gru(xspx).float(), 1)
    return context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
