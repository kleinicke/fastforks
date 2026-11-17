"""Network pieces shared by the RT-DETR-style detection families (RT-DETRv2, D-FINE).

Convolution blocks with folded BatchNorm, the AIFI encoder layer, MLP, anchors and the position embedding.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.detection._attention import Attention
from fastforks.detection._convs import Convs

BN_EPS = 1e-5  # fixed: `nn.BatchNorm2d` / `FrozenBatchNorm2d` default
LN_EPS = 1e-5  # fixed: `nn.LayerNorm` default
POS_TEMPERATURE = 10000.0  # fixed: HybridEncoder `pe_temperature`
ANCHOR_SIZE = 0.05  # fixed: decoder `_generate_anchors` grid_size (doubles per level)
VALID_EPS = 1e-2  # fixed: decoder `eps` of the anchor validity mask
HEADS = 8


class ConvBn(nn.Module):
    """Conv (+ folded BatchNorm bias) + activation (`ConvNormLayer`); `cv` runs it (the kernels runner swaps it)."""

    cv: Convs = Convs()

    def __init__(  # noqa: PLR0913, PLR0917 - a plain convolution's arguments
        self, cin: int, cout: int, k: int, stride: int = 1, act: str | None = None, groups: int = 1
    ) -> None:
        """Create the convolution; `act` is `"relu"`, `"silu"` or `None`."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2, groups=groups)
        self.act = act

    def forward(self, x: torch.Tensor, res: torch.Tensor | None = None) -> torch.Tensor:
        """`act(conv(x))`, or `relu(conv(x) + res)` with a residual (no activation of its own)."""
        if res is not None:
            return self.cv.conv_add_relu(self.conv, x, res)
        if self.act == "relu":
            return self.cv.conv_relu(self.conv, x)
        y = self.cv.conv(self.conv, x)
        return F.silu(y) if self.act == "silu" else y


class RepVgg(nn.Module):
    """RepVGG block in its deployed form: one 3x3 convolution (the 1x1 branch merged at load) and SiLU."""

    cv: Convs = Convs()

    def __init__(self, c: int) -> None:
        """Create the merged convolution."""
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`silu(conv(x))`."""
        return F.silu(self.cv.conv(self.conv, x))


class CspRep(nn.Module):
    """`CSPRepLayer`: two 1x1 branches, RepVGG blocks on the first, summed (and projected if narrower)."""

    def __init__(self, cin: int, cout: int, expansion: float, blocks: int = 3) -> None:
        """Create the branches (`blocks` RepVGG blocks)."""
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = ConvBn(cin, hidden, 1, 1, "silu")
        self.conv2 = ConvBn(cin, hidden, 1, 1, "silu")
        self.bottlenecks = nn.Sequential(*[RepVgg(hidden) for _ in range(blocks)])
        self.conv3 = ConvBn(hidden, cout, 1, 1, "silu") if hidden != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`conv3(bottlenecks(conv1(x)) + conv2(x))`."""
        y = self.bottlenecks(self.conv1(x)) + self.conv2(x)
        return y if self.conv3 is None else self.conv3(y)


class AifiLayer(nn.Module):
    """Post-norm transformer encoder layer with GELU, position added to queries / keys."""

    def __init__(self, d: int, ffn: int) -> None:
        """Create the layer."""
        super().__init__()
        self.self_attn = Attention(d, HEADS)
        self.linear1, self.linear2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.norm1, self.norm2 = nn.LayerNorm(d, LN_EPS), nn.LayerNorm(d, LN_EPS)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """`[B, S, d]` -> `[B, S, d]`."""
        qk = x + pos
        x = self.norm1(x + self.self_attn(qk, qk, x))
        return self.norm2(x + self.linear2(F.gelu(self.linear1(x))))


def sincos_position(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """HybridEncoder `build_2d_sincos_position_embedding` (rows first): `[1, h * w, dim]` float32."""
    gy, gx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    omega = 1.0 / (POS_TEMPERATURE ** (torch.arange(dim // 4, dtype=torch.float32, device=device) / (dim // 4)))
    oy, ox = gy.flatten()[:, None] @ omega[None], gx.flatten()[:, None] @ omega[None]
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], dim=1)[None]


class Mlp(nn.Module):
    """`layers.N` linears with ReLU (or `act = "silu"`: SiLU) between."""

    def __init__(self, cin: int, hidden: int, cout: int, n: int, act: str = "relu") -> None:
        """Create `n` linears."""
        super().__init__()
        dims = [cin] + [hidden] * (n - 1) + [cout]
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in itertools.pairwise(dims))
        self.act = F.silu if act == "silu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layers."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
        return x


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RT-DETR `inverse_sigmoid` (clipped)."""
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def anchors(shapes: list[tuple[int, int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Decoder `_generate_anchors`: log-odds anchors `[1, S, 4]` (inf where invalid) and the validity mask."""
    out = []
    for lvl, (h, w) in enumerate(shapes):
        gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        xy = torch.stack([(gx + 0.5) / w, (gy + 0.5) / h], dim=-1).unsqueeze(0)  # no host tensor: graph-capturable
        wh = torch.ones_like(xy) * ANCHOR_SIZE * 2.0**lvl
        out.append(torch.cat([xy, wh], dim=-1).reshape(-1, h * w, 4))
    a = torch.cat(out, dim=1)
    valid = ((a > VALID_EPS) & (a < 1 - VALID_EPS)).all(-1, keepdim=True)
    return torch.where(valid, torch.log(a / (1 - a)), torch.inf), valid
