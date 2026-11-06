"""Depth Pro network (apple/ml-depth-pro 9e65e4d), with upstream's parameter names.

The three ViT-L/16 backbones (patch encoder, image encoder, FOV encoder) are timm DINOv2 ViTs at 384x384; their state
dicts load into the shared DINOv2 module unchanged. Upstream's encoder crops 35 overlapping 384x384 windows from a
3-level image pyramid (5x5 at 1536, 3x3 at 768, the whole 384 image) and merges their token maps again; here the
windows are gathered as token rows of the per-level patch embedding (every window starts on a multiple of 16 pixels,
so its patches are patches of the full grid) and merged with a precomputed row index (`tables`): same values.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from fastforks._core.checkpoint import load_strict
from fastforks.shared.depth_anything_v2 import vit_large

SIZE = 1536  # network input side
WINDOW = 384  # ViT input side
PATCH = 16
GRID = WINDOW // PATCH  # 24 tokens per window side
HOOKS = (5, 11)  # patch encoder blocks whose outputs feed the two highest-resolution levels
DIMS = (256, 512, 1024, 1024)  # encoder feature dims
DEC = 256  # decoder features


def vit() -> Any:  # noqa: ANN401 - the DINOv2 module (duck-typed attributes)
    """The timm `vit_large_patch14_dinov2` as Depth Pro configures it (patch 16, 384x384)."""
    return vit_large(patch_size=PATCH, img_size=WINDOW, init_values=1.0, block_chunks=0)


def _project_upsample(dim_in: int, dim_out: int, layers: int, dim_int: int | None = None) -> nn.Sequential:
    dim_int = dim_out if dim_int is None else dim_int
    up = [nn.ConvTranspose2d(dim_int if i == 0 else dim_out, dim_out, 2, 2, bias=False) for i in range(layers)]
    return nn.Sequential(nn.Conv2d(dim_in, dim_int, 1, bias=False), *up)


class Encoder(nn.Module):
    """`DepthProEncoder`'s modules."""

    def __init__(self) -> None:
        """Upstream's layout for `dinov2l16_384`."""
        super().__init__()
        c = 1024
        self.patch_encoder = vit()
        self.image_encoder = vit()
        self.upsample_latent0 = _project_upsample(c, DEC, 3, DIMS[0])
        self.upsample_latent1 = _project_upsample(c, DIMS[0], 2)
        self.upsample0 = _project_upsample(c, DIMS[1], 1)
        self.upsample1 = _project_upsample(c, DIMS[2], 1)
        self.upsample2 = _project_upsample(c, DIMS[3], 1)
        self.upsample_lowres = nn.ConvTranspose2d(c, DIMS[3], 2, 2)
        self.fuse_lowres = nn.Conv2d(2 * DIMS[3], DIMS[3], 1)


class ResidualBlock(nn.Module):
    """`x + conv(relu(conv(relu(x))))`."""

    def __init__(self) -> None:
        """Two 3x3 convs with bias."""
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(), nn.Conv2d(DEC, DEC, 3, padding=1), nn.ReLU(), nn.Conv2d(DEC, DEC, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        return x + self.residual(x)


class FusionBlock(nn.Module):
    """`FeatureFusionBlock2d` (no batch norm)."""

    def __init__(self, *, deconv: bool) -> None:
        """Two residual blocks, an optional 2x2 / stride 2 deconvolution, a 1x1 output conv."""
        super().__init__()
        self.resnet1 = ResidualBlock()
        self.resnet2 = ResidualBlock()
        self.deconv = nn.ConvTranspose2d(DEC, DEC, 2, 2, bias=False) if deconv else None
        self.out_conv = nn.Conv2d(DEC, DEC, 1)


class Decoder(nn.Module):
    """`MultiresConvDecoder`."""

    def __init__(self) -> None:
        """Level 0 needs no projection (256 = DEC channels); levels 1-4 a 3x3 conv each."""
        super().__init__()
        dims = (DEC, *DIMS)
        self.convs = nn.ModuleList([nn.Identity()] + [nn.Conv2d(d, DEC, 3, padding=1, bias=False) for d in dims[1:]])
        self.fusions = nn.ModuleList([FusionBlock(deconv=i != 0) for i in range(len(dims))])


class FOV(nn.Module):
    """`FOVNetwork` with its own ViT encoder."""

    def __init__(self) -> None:
        """Encoder (ViT + linear), downsampling conv for the decoder features, conv head."""
        super().__init__()
        self.encoder = nn.Sequential(vit(), nn.Linear(1024, DEC // 2))
        self.downsample = nn.Sequential(nn.Conv2d(DEC, DEC // 2, 3, 2, 1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(DEC // 2, DEC // 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(DEC // 4, DEC // 8, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(DEC // 8, 1, 6),
        )


class DepthProNet(nn.Module):
    """`DepthPro` with the FOV head (the released checkpoint)."""

    def __init__(self) -> None:
        """Encoder, decoder, head, FOV network."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = nn.Sequential(
            nn.Conv2d(DEC, DEC // 2, 3, padding=1),
            nn.ConvTranspose2d(DEC // 2, DEC // 2, 2, 2),
            nn.Conv2d(DEC // 2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )
        self.fov = FOV()

    def vits(self) -> list[nn.Module]:
        """The three ViTs."""
        return [self.encoder.patch_encoder, self.encoder.image_encoder, self.fov.encoder[0]]

    def load(self, sd: dict[str, torch.Tensor], what: str = "Depth Pro checkpoint") -> DepthProNet:
        """Load upstream's state dict strictly, assigned in its float16.

        It has no DINOv2 `mask_token`, which inference does not use.

        Raises:
            WeightsError: The state dict does not fit (names `what` and the first keys).
        """
        load_strict(self, sd, what, not_in_checkpoint=(".mask_token",), assign=True, dtype=torch.float16)
        return self


def _window_rows(steps: int, stride: int, grid: int) -> torch.Tensor:
    """`[steps^2, GRID^2]` rows of a `grid x grid` token map covered by each window, in upstream `split` order."""
    out = []
    for j in range(steps):
        for i in range(steps):
            r = torch.arange(GRID) + j * stride
            c = torch.arange(GRID) + i * stride
            out.append((r[:, None] * grid + c[None, :]).flatten())
    return torch.stack(out)


def _merge(x: torch.Tensor, padding: int) -> torch.Tensor:
    """Upstream `DepthProEncoder.merge` for one image: `[steps^2, ..., h, w]` windows -> one map."""
    steps = math.isqrt(x.shape[0])
    rows = []
    for j in range(steps):
        row = []
        for i in range(steps):
            t = x[j * steps + i]
            t = t[..., padding if j else 0 : -padding if j != steps - 1 else None, :]
            row.append(t[..., padding if i else 0 : -padding if i != steps - 1 else None])
        rows.append(torch.cat(row, -1))
    return torch.cat(rows, -2)


def tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Index tables `(tok, m0, m1, m2)` of the window gather and the merges.

    `tok [35, 576]`: window token rows into the stacked per-level patch embeddings; `m0 [96, 96]`, `m1 [48, 48]`,
    `m2 [24, 24]`: merged maps as rows of the `[35 * 577, C]` encoder output (cls tokens included).
    """
    g0, g1, g2 = SIZE // PATCH, SIZE // 2 // PATCH, SIZE // 4 // PATCH
    w0 = _window_rows(5, WINDOW * 3 // 4 // PATCH, g0)  # overlap 0.25
    w1 = _window_rows(3, WINDOW // 2 // PATCH, g1) + g0 * g0  # overlap 0.5
    w2 = _window_rows(1, 0, g2) + g0 * g0 + g1 * g1
    n = len(w0) + len(w1) + len(w2)
    rows = torch.arange(n * (GRID * GRID + 1)).view(n, GRID * GRID + 1)[:, 1:].reshape(n, GRID, GRID)
    m0 = _merge(rows[: len(w0)], 3)
    m1 = _merge(rows[len(w0) : len(w0) + len(w1)], 6)
    return torch.cat([w0, w1, w2]), m0, m1, rows[-1]
