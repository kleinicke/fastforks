import contextlib

import torch
import torch.nn.functional as F

from fastforks._core.upload import to_device  # fastforks
from fastforks.multiview._runner import invert  # fastforks: closed-form inverses (torch.inverse has no ONNX op)
from fastforks._core.sampling import grid_sample


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")  # [H, W]  # fastforks: the default indexing, explicit

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = to_device(grid, device)  # fastforks: to_device (a pageable upload syncs the host)

    return grid


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    grid_sample_disable_cudnn=False,
    *,
    rays=None,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    grid = projected_grid(intrinsics, pose, depth, h, w, clamp_min_depth, rays=rays)

    # sample features
    # ref: https://github.com/pytorch/pytorch/issues/88380
    # print(feature1.shape, grid.shape)
    # hardcoded workaround
    if feature1.numel() > 1000000:
        grid_sample_disable_cudnn = True
    # fastforks: the cuDNN flag has no effect on CPU and breaks Dynamo capture there.
    cudnn = torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn) if feature1.is_cuda else contextlib.nullcontext()
    with cudnn:
        warped_feature = grid_sample(
            feature1,
            grid.view(b, d * h, w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(
            b, c, d, h, w
        )  # [B, C, D, H, W]

    return warped_feature


@torch.no_grad()
def projected_rays(intrinsics, pose, h, w):
    b = intrinsics.shape[0]
    grid = coords_grid(b, h, w, homogeneous=True, device=intrinsics.device)
    points = invert(intrinsics).bmm(grid.view(b, 3, -1))
    return torch.bmm(pose[:, :3, :3], points)


def projected_grid(intrinsics, pose, depth, h, w, clamp_min_depth=1e-3, *, rays=None):
    # fastforks: share upstream's projection and autocast rounding with the fused cost-volume kernel.
    b, d = depth.shape[:2]
    with torch.no_grad():
        if rays is None:
            rays = projected_rays(intrinsics, pose, h, w)
        points = rays.unsqueeze(2) * depth.reshape(b, 1, d, -1)  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    return grid


def reference_cost_volume(
    ref, targets, intrinsics, pose, depth_candidates, grid_sample_disable_cudnn, *, rays=None, reuse_warp=False
):
    # fastforks: extracted from MultiViewUniMatch.forward so a runner can replace only this operation.
    b, neighbors, c, h, w = targets.shape
    depth = depth_candidates.unsqueeze(1).expand(-1, neighbors, -1, h, w).reshape(-1, depth_candidates.shape[1], h, w)
    k = intrinsics.unsqueeze(1).expand(-1, neighbors, -1, -1).reshape(-1, 3, 3)
    warped = warp_with_pose_depth_candidates(
        targets.flatten(0, 1), k, pose.flatten(0, 1), depth, grid_sample_disable_cudnn=grid_sample_disable_cudnn, rays=rays
    ).unflatten(0, (b, neighbors))
    ref = ref.unsqueeze(-3).unsqueeze(1)
    if reuse_warp and torch.promote_types(warped.dtype, ref.dtype) == warped.dtype:
        product = warped.mul_(ref)
    else:
        product = ref * warped
    return (product.sum(2) / (c ** 0.5)).mean(1)
