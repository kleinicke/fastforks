import torch


# fastforks: our RAFT-Stereo fork's exportable lookup (reg_gather) instead of upstream's reg / reg_cuda (same values);
# DLNR passes fp32 feature maps (upstream's reg casts them, reg_cuda in fp32 in our fork).
class CorrBlockGather1D:
    """Same lookup as reg / reg_cuda, in plain PyTorch ops that export to ONNX / TensorRT (no grid_sample, no extension):
    all-pairs volume as one batched matmul in the feature dtype (fp16 under AMP, as reg_cuda), pyramid by 1-D average
    pooling, linear interpolation of the 2r+2 integer taps around each lookup position via torch.gather (ONNX
    GatherElements). Taps outside the row read 0 (= reg_cuda / grid_sample zero padding). Output channels are
    level-major (level * (2r+1) + tap), as reg_cuda."""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        B, D, H, W1 = fmap1.shape
        # [B, H, W1, D] @ [B, H, D, W2] -> [B, H, W1, W2]
        corr = torch.matmul(fmap1.permute(0, 2, 3, 1), fmap2.permute(0, 2, 1, 3)) * (D**-0.5)
        corr = corr.reshape(B * H * W1, -1)
        self.shape = (B, H, W1)
        self.corr_pyramid = [corr]
        for _ in range(1, num_levels):
            # == F.avg_pool1d(corr, 2, 2) as the mean of even / odd slices (TensorRT 11's Myelin finds no tactic for
            # a 1-D average pool over B*H*W rows)
            n = corr.shape[-1] // 2
            corr = corr[:, : 2 * n].reshape(corr.shape[0], n, 2).mean(dim=-1)
            self.corr_pyramid.append(corr)
        self.taps = torch.arange(-radius, radius + 2, device=fmap1.device)

    def __call__(self, coords):
        B, H, W1 = self.shape
        x = coords[:, 0].reshape(B * H * W1, 1).float()
        out = []
        for i, vol in enumerate(self.corr_pyramid):
            w2 = vol.shape[-1]
            xi = x / 2**i
            x0 = torch.floor(xi)
            a = (xi - x0).to(vol.dtype)
            idx = x0.long() + self.taps  # [N, 2r+2]
            valid = (idx >= 0) & (idx < w2)
            v = torch.gather(vol, 1, idx.clamp(0, w2 - 1)) * valid.to(vol.dtype)
            out.append(v[:, :-1] * (1 - a) + v[:, 1:] * a)  # [N, 2r+1]
        out = torch.cat(out, dim=1).view(B, H, W1, -1)
        return out.permute(0, 3, 1, 2)  # NHWC storage (channels_last view)
