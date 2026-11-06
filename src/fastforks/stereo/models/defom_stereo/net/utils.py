# fastforks: only the helpers the model uses are kept (get_danv2_io_size, bilinear_sampler, coords_grid, upflow); the
# rest of upstream core/utils/utils.py (InputPadder, forward_interpolate, gauss_blur, ...) needs scipy / is training-only.
import torch
import torch.nn.functional as F
import numpy as np
from fastforks._core.sampling import grid_sample


def get_danv2_io_size(h, w, nds, max_i_size=2688, multiple_of=14): 
    """compute the input and output sizes of danv2 network"""
    danv2_oh, danv2_ow = h//2**nds, w//2**nds
    danv2_io_factor = 3.5  # more precise, 14/8=3.5
    ih, iw = danv2_io_factor*danv2_oh, danv2_io_factor*danv2_ow
    ih = int(np.ceil(ih / multiple_of) * multiple_of)
    iw = int(np.ceil(iw / multiple_of) * multiple_of)

    max_i_size = int(np.floor(max_i_size / multiple_of) * multiple_of)

    if ih <= max_i_size and iw <= max_i_size:
        danv2_ih, danv2_iw = ih, iw
    else:
        factor_h = max_i_size/ih
        factor_w = max_i_size/iw

        if factor_w > factor_h:
            danv2_ih = max_i_size
            danv2_iw = int(np.ceil(factor_h * iw / multiple_of) * multiple_of)
        else:
            danv2_iw = max_i_size
            danv2_ih = int(np.ceil(factor_w * ih / multiple_of) * multiple_of)

    return danv2_ih, danv2_iw, danv2_oh, danv2_ow


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    if H > 1:
        ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = grid_sample(img, grid, align_corners=True)
    # img = bilinear_grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing="ij")  # fastforks: explicit (default, required by recent torch)
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow, factor=8, mode='bilinear', sacle=True):
    new_size = (factor * flow.shape[2], factor * flow.shape[3])
    if sacle:
        return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
    else:
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
