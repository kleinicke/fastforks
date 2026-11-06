# fastforks: only CorrBlock1D (the "reg" implementation the demo uses) is kept; the CorrSampler / CorrBlockFast1D /
# PytorchAlternateCorrBlock1D / AlternateCorrBlock variants need the compiled corr_sampler / alt_cuda_corr extensions.
import torch
import torch.nn.functional as F
from .utils import bilinear_sampler


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, coords, num_levels=4, radius=4,
                 scale_list=[0.25, 0.5, 2.0, 4.0], scale_corr_radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.scale_list = scale_list
        self.scale_corr_radius = scale_corr_radius
        self.corr_pyramid = []
        self.coords_pyramid = []
        dx = torch.linspace(-radius, radius, 2*radius+1)
        self.dx = dx[:, None].to(coords.device)

        sdx = torch.linspace(-scale_corr_radius, scale_corr_radius, 2*scale_corr_radius+1)
        self.sdx = sdx[:, None].to(coords.device)

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        self.batch = batch
        self.h1 = h1
        self.w1 = w1
        self.w2 = w2
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)
        self.coords = coords.reshape(batch*h1*w1, 1, 1, 1)

        self.corr_pyramid.append(corr)
        for i in range(1, self.num_levels):
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])
            self.corr_pyramid.append(corr)

    def __call__(self, disp, scaling=False):
        batch, _, h1, w1 = disp.shape

        disp = disp.reshape(self.batch*self.h1*self.w1, 1, 1, 1)
        out_pyramid = []

        if scaling:
            corr = self.corr_pyramid[0]
            for scale in self.scale_list:
                x0 = self.sdx + self.coords - scale * disp
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)
        else:
            coords = self.coords - disp
            for i in range(self.num_levels):
                corr = self.corr_pyramid[i]
                x0 = self.dx + coords / 2**i
                y0 = torch.zeros_like(x0)
                coords_lvl = torch.cat([x0, y0], dim=-1)
                corr_s = bilinear_sampler(corr, coords_lvl)
                corr_s = corr_s.view(self.batch, self.h1, self.w1, -1)
                out_pyramid.append(corr_s)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())
