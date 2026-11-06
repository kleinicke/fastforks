# Modified by fastforks: translated from MegEngine to PyTorch (see net/__init__.py).
# Original: https://github.com/megvii-research/CREStereo/blob/ad3a1613bdedd88b93247e5f002cb7c80799762d/nets/utils/utils.py
import torch
import torch.nn.functional as F
import numpy as np
from fastforks._core.sampling import grid_sample


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    img = grid_sample(img, torch.cat([xgrid, ygrid], dim=-1), mode=mode, padding_mode="zeros", align_corners=True)

    if mask:
        mask = (
            (coords[:, :, :, 0:1] < 0)
            | (coords[:, :, :, 0:1] > W - 1)
            | (coords[:, :, :, 1:2] < 0)
            | (coords[:, :, :, 1:2] > H - 1)
        )
        mask = torch.logical_not(mask)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    y_grid, x_grid = torch.tensor(y_grid, dtype=torch.float32), torch.tensor(
        x_grid, dtype=torch.float32
    )
    coords = torch.stack([x_grid, y_grid], dim=0)
    coords = torch.repeat_interleave(torch.unsqueeze(coords, dim=0), batch, dim=0)
    return coords


def manual_pad(x, pady, padx):
    if pady > 0:
        u = torch.repeat_interleave(x[:, :, 0:1, :], pady, dim=2)
        d = torch.repeat_interleave(x[:, :, -1:, :], pady, dim=2)
        x = torch.cat([u, x, d], dim=2)
    if padx > 0:
        l = torch.repeat_interleave(x[:, :, :, 0:1], padx, dim=3)
        r = torch.repeat_interleave(x[:, :, :, -1:], padx, dim=3)
        x = torch.cat([l, x, r], dim=3)
    return x
