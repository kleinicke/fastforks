"""Plain-PyTorch GREAT-IGEV forward (no custom kernels): the exportable inference path.

Mirrors `GREATStereo.forward` (test mode) on the network prepared by `net.inference.fold_for_inference`: pure fp16 or
fp32 weights and activations, NHWC / NDHWC on CUDA, disparity state, lookup coordinates, softmax / regression and
upsampling in fp32, no host syncs in the update loop, encoder temporaries freed before the next stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from fastforks.shared.igev_blocks.net.submodule import context_upsample

from .net.geometry import FastGeoEncodingVolume

if TYPE_CHECKING:
    from .net.great_igev import GREATStereo


def encode(net: GREATStereo, im1: torch.Tensor, im2: torch.Tensor) -> tuple:
    """Matching features (1/4), the excitive-attention volume and the initial disparity; `stem_2x` for upsampling."""
    eav = net.excitive_attention_volume
    # right view first: only its refined 1/4 matching features are kept
    fr = net.feature(im2)
    fr0 = torch.cat((fr[0], net.stem_4(net.stem_2(im2))), 1)
    del fr
    match_right = eav.global_spatial_refiner(fr0)
    del fr0
    feat_left = net.feature(im1)
    stem_2x = net.stem_2(im1)
    feat_left[0] = torch.cat((feat_left[0], net.stem_4(stem_2x)), 1)
    match_left = eav.global_spatial_refiner(feat_left[0])
    pos = eav.matching_pos(match_left).to(match_left.dtype).contiguous(memory_format=torch.channels_last)
    match_left, match_right = eav.global_matching_refiner(match_left + pos, match_right + pos)
    del pos
    context = eav.global_context_refiner(feat_left)
    del feat_left
    volume = eav.cost_volume(match_left, match_right, context[0])
    volume, init_disp = eav.eav_agg.disparity(volume, context)
    del context
    return match_left, match_right, volume, init_disp, stem_2x


@torch.no_grad()
def forward_fast(net: GREATStereo, left: torch.Tensor, right: torch.Tensor, iters: int) -> torch.Tensor:
    """Disparity `[B, 1, H, W]` float32 of a padded pair (`left`, `right`: RGB in 0-255)."""
    dt = cast("torch.dtype", net.inf_dtype)  # set by fold_for_inference
    cl = cast("bool", net.inf_cl)
    mf: torch.memory_format = torch.channels_last if cl else torch.contiguous_format
    a = net.args
    im1 = (2 * (left / 255.0) - 1.0).to(dt).contiguous(memory_format=mf)
    im2 = (2 * (right / 255.0) - 1.0).to(dt).contiguous(memory_format=mf)

    match_left, match_right, volume, disp, stem_2x = encode(net, im1, im2)
    del im2
    # the kernels runner on Apple MPS sets `geo_volume` (the lookups as Metal kernels)
    geo_fn = getattr(net, "geo_volume", FastGeoEncodingVolume)(
        match_left, match_right, volume, radius=a.corr_radius, num_levels=a.corr_levels, out_dtype=dt
    )
    del volume, match_left, match_right

    cnet_list = net.cnet(im1, num_layers=a.n_gru_layers)
    del im1
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    del cnet_list
    zqr: Any = net.context_zqr_convs
    inp_list = [list(conv(i).split(conv.out_channels // 3, dim=1)) for i, conv in zip(inp_list, zqr, strict=True)]

    b, _, h, w = disp.shape
    disp_h = torch.empty((b, 1, h, w), device=disp.device, dtype=dt, memory_format=mf)
    mask_feat_4 = None
    for itr in range(iters):
        geo_feat = geo_fn(disp)
        disp_h.copy_(disp)
        net_list, mask_feat_4, delta_disp = net.update_block(
            net_list, inp_list, geo_feat, disp_h, compute_mask=itr == iters - 1
        )  # 3 GRU levels: iter16 / iter08 on (the defaults)
        disp = disp + delta_disp.float()

    xspx = net.spx_2_gru(mask_feat_4, stem_2x)
    spx_pred = F.softmax(net.spx_gru(xspx).float(), 1)
    return context_upsample(disp * 4.0, spx_pred).unsqueeze(1)
