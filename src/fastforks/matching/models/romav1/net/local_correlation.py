# fastforks: upstream `romatch/utils/local_correlation.py` calls the `local_corr` CUDA extension when it is installed
# and otherwise samples a [C, H, W * K] window tensor per image. This plain-PyTorch version loops over the K window
# offsets instead: the same bilinear samples and products, one [B, C, H, W] sample at a time.
import torch
import torch.nn.functional as F


def local_correlation(
    feature0: torch.Tensor,  # (B x C x H x W)
    feature1: torch.Tensor,  # (B x C x H x W)
    local_radius: int,
    warp: torch.Tensor,  # (B x H x W x 2), normalized coordinates in image 1
) -> torch.Tensor:
    """`[B, (2r + 1)^2, H, W]` scaled dot products of `feature0` with `feature1` around `warp` (window rows y-major)."""
    r = local_radius
    _, c, h, w = feature0.shape
    # one step of the window = one feature pixel (the grid spans 2 / size per pixel)
    ys = torch.linspace(-2 * r / h, 2 * r / h, 2 * r + 1, device=warp.device)
    xs = torch.linspace(-2 * r / w, 2 * r / w, 2 * r + 1, device=warp.device)
    f0 = feature0 / (c**0.5)
    corr = []
    for dy in ys:
        for dx in xs:
            offset = torch.stack((dx, dy)).to(warp.dtype)
            window = F.grid_sample(feature1, warp + offset, padding_mode="zeros", align_corners=False, mode="bilinear")
            corr.append((f0 * window).sum(dim=1))
    return torch.stack(corr, dim=1)
