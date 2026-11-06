# Modified by fastforks: translated from MegEngine to PyTorch (see net/__init__.py).
# Original: https://github.com/megvii-research/CREStereo/blob/ad3a1613bdedd88b93247e5f002cb7c80799762d/nets/corr.py
import numpy as np
import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid


class AGCL:
    """
    Implementation of Adaptive Group Correlation Layer (AGCL).
    """

    def __init__(self, fmap1, fmap2, att=None):
        self.fmap1 = fmap1
        self.fmap2 = fmap2

        self.att = att

        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3]).to(
            fmap1.device
        )

    def __call__(self, flow, extra_offset, small_patch=False, iter_mode=False):
        if iter_mode:
            corr = self.corr_iter(self.fmap1, self.fmap2, flow, small_patch)
        else:
            corr = self.corr_att_offset(
                self.fmap1, self.fmap2, flow, extra_offset, small_patch
            )
        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.shape

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = F.pad(right_feature, (padx, padx, pady, pady), mode="replicate")

        right_slid = right_pad.unfold(2, H, di_y).unfold(3, W, di_x)
        right_slid = right_slid.reshape(N, C, -1, H, W)
        right_slid = right_slid.permute(0, 2, 1, 3, 4)
        right_slid = right_slid.reshape(-1, C, H, W)

        corr_mean = torch.mean(left_feature * right_slid, dim=1, keepdim=True)
        corr_final = corr_mean.reshape(1, -1, H, W)

        return corr_final

    def corr_iter(self, left_feature, right_feature, flow, small_patch):

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = left_feature.shape
        lefts = torch.chunk(left_feature, 4, dim=1)
        rights = torch.chunk(right_feature, 4, dim=1)

        corrs = []
        for i in range(len(psize_list)):
            corr = self.get_correlation(
                lefts[i], rights[i], psize_list[i], dilate_list[i]
            )
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def corr_att_offset(
        self, left_feature, right_feature, flow, extra_offset, small_patch
    ):

        N, C, H, W = left_feature.shape

        if self.att is not None:
            left_feature = torch.reshape(
                left_feature.permute(0, 2, 3, 1), (N, H * W, C)
            )  # 'n c h w -> n (h w) c'
            right_feature = torch.reshape(
                right_feature.permute(0, 2, 3, 1), (N, H * W, C)
            )  # 'n c h w -> n (h w) c'
            left_feature, right_feature = self.att(left_feature, right_feature)
            # 'n (h w) c -> n c h w'
            left_feature, right_feature = [
                torch.reshape(x, (N, H, W, C)).permute(0, 3, 1, 2)
                for x in [left_feature, right_feature]
            ]

        lefts = torch.chunk(left_feature, 4, dim=1)
        rights = torch.chunk(right_feature, 4, dim=1)

        C = C // 4

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        search_num = 9
        extra_offset = torch.reshape(extra_offset, (N, search_num, 2, H, W)).permute(
            0, 1, 3, 4, 2
        )  # [N, search_num, 1, 1, 2]

        corrs = []
        for i in range(len(psize_list)):
            left_feature, right_feature = lefts[i], rights[i]
            psize, dilate = psize_list[i], dilate_list[i]

            psizey, psizex = psize[0], psize[1]
            dilatey, dilatex = dilate[0], dilate[1]

            ry = psizey // 2 * dilatey
            rx = psizex // 2 * dilatex

            x_grid, y_grid = np.meshgrid(
                np.arange(-rx, rx + 1, dilatex), np.arange(-ry, ry + 1, dilatey)
            )
            y_grid, x_grid = torch.tensor(y_grid, device=self.fmap1.device), torch.tensor(
                x_grid, device=self.fmap1.device
            )
            offsets = torch.reshape(torch.stack((x_grid, y_grid)), (2, -1)).permute(
                1, 0
            )  # [search_num, 2]
            offsets = offsets[None, :, None, None]
            offsets = offsets + extra_offset

            coords = self.coords + flow  # [N, 2, H, W]
            coords = coords.permute(0, 2, 3, 1)  # [N, H, W, 2]
            coords = torch.unsqueeze(coords, 1) + offsets
            coords = torch.reshape(coords, (N, -1, W, 2))  # [N, search_num*H, W, 2]

            right_feature = bilinear_sampler(
                right_feature, coords
            )  # [N, C, search_num*H, W]
            right_feature = torch.reshape(
                right_feature, (N, C, -1, H, W)
            )  # [N, C, search_num, H, W]

            left_feature = torch.unsqueeze(left_feature, 2)
            corr = torch.mean(left_feature * right_feature, dim=1)

            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr
