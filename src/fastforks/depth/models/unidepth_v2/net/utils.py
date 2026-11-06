"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

# fastforks: the helpers UniDepth V2 inference uses from unidepth/utils/{misc,coordinate,geometric,
# positional_embedding,camera}.py, verbatim except the camera classes (see PinholeCamera)
from math import log2, pi
from typing import Tuple

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------- utils/misc.py
def last_stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return tensors[-1]


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


# ---------------------------------------------------------------- utils/coordinate.py
def coords_grid(b, h, w, homogeneous=False, device=None, noisy=False):
    pixel_coords_x = torch.linspace(0.5, w - 0.5, w, device=device)
    pixel_coords_y = torch.linspace(0.5, h - 0.5, h, device=device)
    if noisy:  # \pm 0.5px noise
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5

    stacks = [pixel_coords_x.repeat(h, 1), pixel_coords_y.repeat(w, 1).t()]
    if homogeneous:
        ones = torch.ones_like(stacks[0])  # [H, W]
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]
    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]
    if device is not None:
        grid = grid.to(device)

    return grid


# ---------------------------------------------------------------- utils/geometric.py
def flat_interpolate(
    flat_tensor: torch.Tensor,
    old: Tuple[int, int],
    new: Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> torch.Tensor:
    if old[0] == new[0] and old[1] == new[1]:
        return flat_tensor
    tensor = flat_tensor.view(flat_tensor.shape[0], old[0], old[1], -1).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = F.interpolate(
        tensor,
        size=(new[0], new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, new[0] * new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()


# ---------------------------------------------------------------- utils/positional_embedding.py
def generate_fourier_features(
    x: torch.Tensor,
    dim: int = 512,
    max_freq: int = 64,
    use_cos: bool = False,
    use_log: bool = False,
    cat_orig: bool = False,
):
    x_orig = x
    device, dtype, input_dim = x.device, x.dtype, x.shape[-1]
    num_bands = dim // (2 * input_dim) if use_cos else dim // input_dim

    if use_log:
        scales = 2.0 ** torch.linspace(
            0.0, log2(max_freq), steps=num_bands, device=device, dtype=dtype
        )
    else:
        scales = torch.linspace(
            1.0, max_freq / 2, num_bands, device=device, dtype=dtype
        )

    x = x.unsqueeze(-1)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat(
        (
            [x.sin(), x.cos()]
            if use_cos
            else [
                x.sin(),
            ]
        ),
        dim=-1,
    )
    x = x.flatten(-2)
    if cat_orig:
        return torch.cat((x, x_orig), dim=-1)
    return x


# ---------------------------------------------------------------- utils/camera.py
# fastforks: the pinhole path of `BatchCamera.from_camera(Pinhole(K=K))` as one small class: `crop` / `resize` /
# `get_rays` with upstream's maths (Camera.crop, Camera.resize, Camera.get_rays + Pinhole.unproject); upstream edits
# K in place, this copies it first
class PinholeCamera:
    def __init__(self, K):
        self.K = K.clone()

    def crop(self, left, top, right=None, bottom=None):
        self.K[..., 0, 2] -= left
        self.K[..., 1, 2] -= top
        return self

    def resize(self, sx, sy=None):  # fastforks: per-axis scales (upstream: one factor)
        self.K[..., 0, :] *= sx
        self.K[..., 1, :] *= sx if sy is None else sy
        return self

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def unproject(self, uv):
        b, _, h, w = uv.shape
        uv_flat = uv.reshape(b, 2, -1)  # [B, 2, H*W]
        uv_homogeneous = torch.cat(
            [uv_flat, torch.ones(b, 1, h * w, device=uv.device)], dim=1
        )  # [B, 3, H*W]
        K_inv = torch.inverse(self.K.float())
        xyz = K_inv @ uv_homogeneous
        xyz = xyz / xyz[:, -1:].clip(min=1e-4)
        xyz = xyz.reshape(-1, 3, h, w)  # fastforks: -1 (upstream b): the batch comes from K, as BatchCamera's
        return xyz

    def get_rays(self, shapes, noisy=False):
        b, h, w = shapes
        uv = coords_grid(1, h, w, device=self.K.device, noisy=noisy)
        rays = self.unproject(uv)
        return rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)
