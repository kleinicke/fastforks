import torch
import torch.nn.functional as F

# fastforks: only bilinear_sampler of RT-MonSter++'s core/utils/utils.py (the rest needs scipy and is not used at
# inference). Unlike MonSter's copy it samples with border padding.


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, padding_mode='border', align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img
