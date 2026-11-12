"""MoGe networks, vendored from upstream (microsoft/MoGe 74fbce0, MIT, see `LICENSE`) with the same state-dict names.

Changes: only the layer types the released checkpoints use (no normalisation in the residual blocks, transposed-conv
and bilinear resamplers); no gradient checkpointing, ONNX mode or hub loading; the DINOv2 backbone is
`fastforks.shared.depth_anything_v2`'s (same module as MoGe's hub `dinov2_vit*14`); the MoGe-3 refiner's FlexGEMM
(Triton) sparse ops are re-implemented in PyTorch (`sparse.py`).
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2 import DINOv2

from .sparse import SparseUNet

PATCH = 14


class ResidualConvBlock(nn.Module):
    """`x + conv(relu(conv(relu(x))))`, 3x3 replicate-padded convs (upstream with `in_norm = hidden_norm = "none"`)."""

    def __init__(self, channels: int) -> None:
        """Layers at upstream's `Sequential` indices (0 / 3: the identity norms)."""
        super().__init__()
        conv = lambda: nn.Conv2d(channels, channels, 3, padding=1, padding_mode="replicate")  # noqa: E731
        self.layers = nn.Sequential(nn.Identity(), nn.ReLU(), conv(), nn.Identity(), nn.ReLU(), conv())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual block."""
        return self.layers(x) + x


class Resampler(nn.Sequential):
    """2x upsampling: transposed conv (k = s = 2) or bilinear, then a 3x3 replicate-padded conv."""

    def __init__(self, c_in: int, c_out: int, kind: str) -> None:
        """`kind`: `"conv_transpose"` or `"bilinear"`."""
        if kind == "conv_transpose":
            first: nn.Module = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
            c_in = c_out
        elif kind == "bilinear":
            first = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        else:
            msg = f"resampler {kind!r}"
            raise ValueError(msg)
        super().__init__(first, nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="replicate"))


class ConvStack(nn.Module):
    """Upstream `ConvStack`: per level an input 1x1 conv (added), residual blocks, an output 1x1 conv, a resampler.

    The neck's output convs are identities (its outputs are the levels' features); the heads use only the last level.
    """

    def __init__(
        self,
        dim_in: list[int],
        dim_res_blocks: list[int],
        dim_out: list[int | None],
        resamplers: list[str],
        num_res_blocks: list[int],
    ) -> None:
        """Same arguments as upstream (the released configs)."""
        super().__init__()
        conv1 = lambda i, o: nn.Conv2d(i, o, 1) if o is not None else nn.Identity()  # noqa: E731
        self.input_blocks = nn.ModuleList(conv1(i, r) for i, r in zip(dim_in, dim_res_blocks, strict=True))
        self.resamplers = nn.ModuleList(
            Resampler(a, b, k) for a, b, k in zip(dim_res_blocks[:-1], dim_res_blocks[1:], resamplers, strict=True)
        )
        self.res_blocks = nn.ModuleList(
            nn.Sequential(*(ResidualConvBlock(c) for _ in range(n)))
            for c, n in zip(dim_res_blocks, num_res_blocks, strict=True)
        )
        self.output_blocks = nn.ModuleList(conv1(r, o) for o, r in zip(dim_out, dim_res_blocks, strict=True))

    def forward(self, feats: list[torch.Tensor], *, fp32_out: bool = False) -> list[torch.Tensor]:
        """Every level's output (upstream); `fp32_out`: the output convs in fp32 (upstream `fp32_output_projection`)."""
        outs = []
        x = feats[0]
        n = len(self.res_blocks)
        for i in range(n):
            f = self.input_blocks[i](feats[i])
            x = f if i == 0 else x + f
            x = self.res_blocks[i](x)
            ob = self.output_blocks[i]
            if fp32_out and isinstance(ob, nn.Conv2d):
                with torch.autocast(x.device.type, enabled=False):
                    outs.append(F.conv2d(x.float(), ob.weight.float(), None if ob.bias is None else ob.bias.float()))
            else:
                outs.append(ob(x))
            if i < n - 1:
                x = self.resamplers[i](x)
        return outs


class MLP(nn.Sequential):
    """Linear layers with ReLU in between."""

    def __init__(self, dims: list[int]) -> None:
        """`dims`: widths, input first."""
        layers: list[nn.Module] = []
        for a, b in itertools.pairwise(dims[:-1]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        super().__init__(*layers, nn.Linear(dims[-2], dims[-1]))


class Encoder(nn.Module):
    """Upstream `DINOv2Encoder`: DINOv2 tokens of some blocks, each projected by a 1x1 conv, summed."""

    def __init__(self, arch: str, layers: list[int], dim_out: int) -> None:
        """`arch`: `"vits"`, `"vitb"` or `"vitl"`."""
        super().__init__()
        self.layers = layers
        self.backbone = DINOv2(arch)
        dim = self.backbone.embed_dim
        self.output_projections = nn.ModuleList(nn.Conv2d(dim, dim_out, 1) for _ in layers)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`[B, 3, H, W]` in `[0, 1]` -> features `[B, dim_out, rows, cols]` and the last class token."""
        x = F.interpolate(image, (rows * PATCH, cols * PATCH), mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.get_buffer("image_mean")) / self.get_buffer("image_std")
        feats = self.backbone.get_intermediate_layers(x, n=self.layers, return_class_token=True)
        out = sum(
            proj(t.permute(0, 2, 1).unflatten(2, (rows, cols)).contiguous())
            for proj, (t, _) in zip(self.output_projections, feats, strict=True)
        )
        return out, feats[-1][1]  # sum of tensors


def _stack(dims: list[int], blocks: list[int], dim_in: list[int], dim_out: list[int | None]) -> ConvStack:
    return ConvStack(dim_in, dims, dim_out, ["conv_transpose"] * 3 + ["bilinear"], blocks)


class MoGeNet(nn.Module):
    """MoGe-2 / MoGe-3 network (upstream `MoGeModel` modules); the runners drive the forward pass."""

    def __init__(self, arch: str, version: int) -> None:
        """`arch`: `"vits"` / `"vitb"` (MoGe-2 ViT-S / ViT-B, same layout) or `"vitl"`; `version` 3 adds the refiner."""
        super().__init__()
        small = arch != "vitl"
        width = {"vits": 384, "vitb": 768, "vitl": 1024}[arch]
        layers = [5, 11] if small else [5, 11, 17, 23]
        dims = [width, 256, 128, 64, 32]
        self.encoder = Encoder(arch, layers, width)
        self.neck = _stack(dims, [0, 1, 1, 1, 0] if small else [0, 2, 2, 2, 0], [width + 2, 2, 2, 2, 2], [None] * 5)
        head = lambda c: _stack(dims, [0, 1, 1, 1, 0], list(dims), [None] * 4 + [c])  # noqa: E731
        self.points_head, self.normal_head, self.mask_head = head(3), head(3), head(1)
        self.scale_head = MLP([width, width, width, 1])
        self.refiner = SparseUNet(3, 1, width + 2, [32, 64, 128, 256, 512]) if version == 3 else None  # noqa: PLR2004
