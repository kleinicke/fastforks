"""Group-wise correlation cost volume as one op, so the MLX replay can run it as one Metal kernel.

`gwc_volume(left, right, depth, groups)[b, g, d, y, x]` is the channel mean, per group, of `left[x] * right[x - d]`
(zero where `x < d`) -- upstream's `build_gwc_volume` family. Eagerly it is the slab loop; an exported graph keeps the op
and `fastforks.shared.mlx_graph` lowers it to `fastforks.kernels.graph_mlx.ops.gwc_volume`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.library.custom_op("fastforks::gwc_volume", mutates_args=())
def gwc_volume(left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:
    """`[B, groups, depth, H, W]` group-wise correlation of `[B, C, H, W]` features (`C % groups == 0`)."""
    b, c, h, w = left.shape
    slabs = []
    for i in range(depth):
        if i < w:
            prod = (left[..., i:] * right[..., : w - i]).view(b, groups, c // groups, h, w - i).mean(dim=2)
            slabs.append(F.pad(prod, (i, 0)))
        else:
            slabs.append(left.new_zeros((b, groups, h, w)))
    return torch.stack(slabs, dim=2)


@gwc_volume.register_fake
def _(left: torch.Tensor, right: torch.Tensor, depth: int, groups: int) -> torch.Tensor:  # noqa: ARG001 - same schema
    b, _, h, w = left.shape
    return left.new_empty((b, groups, depth, h, w))


def normalized_groups(x: torch.Tensor, groups: int, eps: float = 1e-5) -> torch.Tensor:
    """`x / (||x|| + eps)` over each group's channels (upstream's `groupwise_correlation_norm`), `[B, C, H, W]`."""
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w)
    return (x / (torch.norm(x, 2, 2, True) + eps)).view(b, c, h, w)


@torch.library.custom_op("fastforks::upsample_soft_argmax", mutates_args=())
def upsample_soft_argmax(cost: torch.Tensor, max_disp: int, height: int, width: int) -> torch.Tensor:
    """Trilinear upsample of `[B, 1, d, h, w]` to `[max_disp, height, width]`, softmax over disparity, soft-argmin.

    Returns `[B, height, width]` float32 (upstream PSMNet / GwcNet test tail; `align_corners=False`).
    """
    up = F.interpolate(cost.float(), size=(max_disp, height, width), mode="trilinear", align_corners=False)
    values = torch.arange(max_disp, device=cost.device, dtype=torch.float32).view(1, max_disp, 1, 1)
    return torch.sum(F.softmax(up.squeeze(1), dim=1) * values, 1)


@upsample_soft_argmax.register_fake
def _(cost: torch.Tensor, max_disp: int, height: int, width: int) -> torch.Tensor:  # noqa: ARG001 - same schema
    return cost.new_empty((cost.shape[0], height, width), dtype=torch.float32)
