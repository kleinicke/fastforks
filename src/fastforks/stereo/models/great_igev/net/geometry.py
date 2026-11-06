import torch
import torch.nn.functional as F
from fastforks._core.sampling import grid_sample


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
            out_pyramid.append(init_corr)
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
        return corr

def _pool2_last(x):
    """== F.avg_pool1d(x, 2, 2) over the last axis as the mean of even / odd slices (TensorRT 11's Myelin finds no
    tactic for a 1-D average pool over millions of rows)."""
    n = x.shape[-1] // 2
    return 0.5 * (x[..., 0:2 * n:2] + x[..., 1:2 * n:2])


class FastGeoEncodingVolume:
    """Inference version of Combined_Geo_Encoding_Volume (same math as grid_sample bilinear, align_corners=True,
    zero padding), without per-lookup host syncs / H2D copies:
    - geometry volume kept fp32 ([P, C, D], small), all-pairs correlation from an fp16 GEMM stored fp16
      ([P, W2], the big one: 8x less memory than upstream's fp32 einsum + contiguous copy at Middlebury-F),
    - 1-D linear interpolation by two gathers instead of grid_sample on [P, C, 1, D] (+ upstream's torch.unique sync),
    - output written as NHWC [B, H, W, 162] (channels_last view), lookup coordinates fp32."""

    def __init__(self, fmap1, fmap2, geo_volume, num_levels=2, radius=4, out_dtype=torch.float16):
        self.num_levels, self.radius, self.out_dtype = num_levels, radius, out_dtype
        b, c, d, h, w = geo_volume.shape
        self.shape = (b, h, w)
        g = geo_volume.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, d).float()
        # correlation: [B*H, W1, D] @ [B*H, D, W2] (fp16 in, fp32 accumulate, fp16 out; fp32 for fp32 inference)
        f1 = fmap1.permute(0, 2, 3, 1).reshape(b * h, w, -1)
        f2 = fmap2.permute(0, 2, 1, 3).reshape(b * h, -1, w)
        ct = torch.float32 if out_dtype == torch.float32 else torch.float16
        corr = torch.bmm(f1.to(ct), f2.to(ct)).view(b * h * w, w)
        self.geo, self.corr = [g], [corr]
        for _ in range(num_levels - 1):
            g = _pool2_last(g)
            self.geo.append(g)
            corr = _pool2_last(corr)
            self.corr.append(corr)
        self.dx = torch.arange(-radius, radius + 1, device=geo_volume.device, dtype=torch.float32).view(1, -1)
        self.coords = torch.arange(w, device=geo_volume.device, dtype=torch.float32).repeat(b * h).view(-1, 1)

    @staticmethod
    def _sample(vol, x):
        """vol [P, C, D] or [P, D]; x [P, K] fp32 positions -> [P, C*K] or [P, K] (fp32 math)."""
        D = vol.shape[-1]
        x0 = torch.floor(x)
        w1 = x - x0
        i0 = x0.long()
        i1 = i0 + 1
        v0 = ((i0 >= 0) & (i0 < D)).float() * (1 - w1)
        v1 = ((i1 >= 0) & (i1 < D)).float() * w1
        idx = torch.cat([i0, i1], 1).clamp_(0, D - 1)
        wgt = torch.cat([v0, v1], 1)
        K = x.shape[1]
        if vol.dim() == 3:
            P, C, _ = vol.shape
            s = vol.gather(2, idx.unsqueeze(1).expand(P, C, 2 * K)).float() * wgt.unsqueeze(1)
            return (s[..., :K] + s[..., K:]).reshape(P, C * K)
        s = vol.gather(1, idx).float() * wgt
        return s[:, :K] + s[:, K:]

    def __call__(self, disp):
        b, h, w = self.shape
        dsp = disp.reshape(-1, 1).float()
        out = []
        for i in range(self.num_levels):
            sc = 1.0 / 2 ** i
            out.append(self._sample(self.geo[i], dsp * sc + self.dx))
            out.append(self._sample(self.corr[i], self.coords * sc - dsp * sc + self.dx))
        out = torch.cat(out, 1).to(self.out_dtype)
        return out.view(b, h, w, -1).permute(0, 3, 1, 2)
