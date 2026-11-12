# fastforks: the grid helpers of upstream `romav2/geometry.py` used at inference (the depth warps, pose and
# precision-matrix helpers belong to training / evaluation); the device comes from the inputs.
import torch
import torch.nn.functional as F


def get_normalized_grid(B: int, H: int, W: int, device: torch.device | None = None) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def bhwc_interpolate(
    x: torch.Tensor,
    size: tuple[int, int],
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode=mode, align_corners=align_corners).permute(0, 2, 3, 1)


def bhwc_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool | None = None,
) -> torch.Tensor:
    return F.grid_sample(x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners).permute(0, 2, 3, 1)
