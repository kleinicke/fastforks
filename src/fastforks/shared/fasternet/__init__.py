"""FasterNet feature stages with timm's module names, so timm-trained checkpoints load unchanged.

LiteAnyStereo V2 builds its encoder from `timm.create_model("fasternet_t0", features_only=True, out_indices=(0, 1, 2,
3))` and keeps `patch_embed` and `stages_N` in its state dict. This module rebuilds that structure without timm: same
module tree, parameter names and eval-mode math (`split_cat` partial convolution, no layer scale). The layout follows
timm's `fasternet_t0` definition (Apache-2.0; FasterNet itself is MIT, Microsoft Corporation; see LICENSE).
"""

from __future__ import annotations

import torch
from torch import nn


class PartialConv3(nn.Module):
    """3x3 convolution on the first `dim // n_div` channels; the other channels pass through."""

    def __init__(self, dim: int, n_div: int) -> None:
        """Create the partial convolution."""
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve the first channel group and concatenate the rest unchanged."""
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv3(x1), x2), 1)


class MLPBlock(nn.Module):
    """Partial-conv spatial mixing followed by a residual 1x1 MLP (conv, BN, GELU, conv)."""

    def __init__(self, dim: int, n_div: int, mlp_ratio: float) -> None:
        """Create the block."""
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
        )
        self.spatial_mixing = PartialConv3(dim, n_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        return x + self.mlp(self.spatial_mixing(x))


class PatchMerging(nn.Module):
    """2x2 stride-2 convolution doubling the channels, then BatchNorm."""

    def __init__(self, dim: int, patch_size: int = 2) -> None:
        """Create the merge from `dim` to `2 * dim` channels."""
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, patch_size, patch_size, bias=False)
        self.norm = nn.BatchNorm2d(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample."""
        return self.norm(self.reduction(x))


class Stage(nn.Module):
    """Optional patch merging followed by `depth` MLP blocks."""

    def __init__(self, dim: int, depth: int, n_div: int, mlp_ratio: float, *, merge: bool) -> None:
        """Create a stage at width `dim` (the merge, if any, comes from `dim // 2`)."""
        super().__init__()
        self.blocks = nn.Sequential(*[MLPBlock(dim, n_div, mlp_ratio) for _ in range(depth)])
        self.downsample = PatchMerging(dim // 2) if merge else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample, then run the blocks."""
        return self.blocks(self.downsample(x))


class FasterNetFeatures(nn.Module):
    """FasterNet backbone returning the output of every stage (strides 4, 8, 16, 32).

    Defaults are `fasternet_t0` (embed_dim 40, depths 1-2-8-2, GELU, `mlp_ratio` 2, `n_div` 4).
    """

    def __init__(
        self,
        embed_dim: int = 40,
        depths: tuple[int, ...] = (1, 2, 8, 2),
        mlp_ratio: float = 2.0,
        n_div: int = 4,
        patch_size: int = 4,
    ) -> None:
        """Create the backbone."""
        super().__init__()
        self.patch_embed = _PatchEmbed(3, embed_dim, patch_size)
        for i, depth in enumerate(depths):
            stage = Stage(embed_dim * 2**i, depth, n_div, mlp_ratio, merge=i > 0)
            self.add_module(f"stages_{i}", stage)
        self.num_stages = len(depths)
        self.channels = [embed_dim * 2**i for i in range(len(depths))]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the feature map after each stage."""
        x = self.patch_embed(x)
        out = []
        for i in range(self.num_stages):
            x = getattr(self, f"stages_{i}")(x)
            out.append(x)
        return out


class _PatchEmbed(nn.Module):
    """Non-overlapping patch convolution and BatchNorm."""

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int) -> None:
        """Create the stem."""
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patches."""
        return self.norm(self.proj(x))
