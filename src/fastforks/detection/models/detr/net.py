"""DETR (ResNet backbone + encoder-decoder transformer) for inference, BatchNorm folded into the convolutions.

State-dict names follow the official checkpoint, with `backbone.0.body.` shortened to `backbone.` and every
`bnN` / `downsample.1` merged into the preceding convolution's bias by `fold_frozen_bn`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.detection._attention import Attention
from fastforks.detection._fold import fold_bn_into

FROZEN_BN_EPS = 1e-5  # fixed: DETR `FrozenBatchNorm2d` (torchvision's default is 0 in older releases)
POS_TEMPERATURE = 10000.0  # fixed: DETR `PositionEmbeddingSine`
LN_EPS = 1e-5  # fixed: `nn.LayerNorm` default


@dataclass(frozen=True)
class Arch:
    """Network shape (card `[params]`)."""

    blocks: list[int]  # bottleneck blocks per ResNet stage
    d_model: int
    heads: int
    enc_layers: int
    dec_layers: int
    ffn: int
    num_queries: int
    num_classes: int  # class ids incl. the unused ones (91); the head has one more output (no object)


class Bottleneck(nn.Module):
    """torchvision `Bottleneck` (stride on the 3x3), BatchNorm folded into the conv biases."""

    def __init__(self, cin: int, mid: int, stride: int) -> None:
        """Create the parameters."""
        super().__init__()
        self.conv1 = nn.Conv2d(cin, mid, 1)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1)
        self.conv3 = nn.Conv2d(mid, mid * 4, 1)
        self.downsample = nn.Sequential(nn.Conv2d(cin, mid * 4, 1, stride)) if cin != mid * 4 or stride != 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`relu(conv3(relu(conv2(relu(conv1(x))))) + shortcut(x))`."""
        out = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNet(nn.Module):
    """ResNet-50 / 101 up to `layer4` (stride 32)."""

    def __init__(self, blocks: list[int]) -> None:
        """Create the parameters."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        layers, cin = [], 64
        for i, n in enumerate(blocks):
            mid = 64 << i
            layers.append(
                nn.Sequential(
                    *[Bottleneck(cin if j == 0 else mid * 4, mid, 2 if i and j == 0 else 1) for j in range(n)]
                )
            )
            cin = mid * 4
        self.layer1, self.layer2, self.layer3, self.layer4 = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised image -> stride-32 features."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 3, 2, 1)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class EncoderLayer(nn.Module):
    """Post-norm encoder layer; the positional embedding is added to the queries and keys only."""

    def __init__(self, d: int, heads: int, ffn: int) -> None:
        """Create the parameters."""
        super().__init__()
        self.self_attn = Attention(d, heads)
        self.linear1, self.linear2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.norm1, self.norm2 = nn.LayerNorm(d, LN_EPS), nn.LayerNorm(d, LN_EPS)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """`[B, S, d]` -> `[B, S, d]`."""
        qk = x + pos
        x = self.norm1(x + self.self_attn(qk, qk, x))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    """Post-norm decoder layer: self-attention over the queries, cross-attention over the memory, FFN."""

    def __init__(self, d: int, heads: int, ffn: int) -> None:
        """Create the parameters."""
        super().__init__()
        self.self_attn, self.multihead_attn = Attention(d, heads), Attention(d, heads)
        self.linear1, self.linear2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d, LN_EPS) for _ in range(3))

    def forward(self, x: torch.Tensor, mem: torch.Tensor, pos: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """`[B, Q, d]` queries over `[B, S, d]` memory -> `[B, Q, d]`."""
        qk = x + qpos
        x = self.norm1(x + self.self_attn(qk, qk, x))
        x = self.norm2(x + self.multihead_attn(x + qpos, mem + pos, mem))
        return self.norm3(x + self.linear2(F.relu(self.linear1(x))))


class Transformer(nn.Module):
    """Encoder and decoder stacks (checkpoint names `encoder.layers.N`, `decoder.layers.N`, `decoder.norm`)."""

    def __init__(self, a: Arch) -> None:
        """Create the parameters."""
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(EncoderLayer(a.d_model, a.heads, a.ffn) for _ in range(a.enc_layers))
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(DecoderLayer(a.d_model, a.heads, a.ffn) for _ in range(a.dec_layers))
        self.decoder.norm = nn.LayerNorm(a.d_model, LN_EPS)


class Mlp(nn.Module):
    """The box head: `layers.N` linears with ReLU between."""

    def __init__(self, d: int, out: int, n: int = 3) -> None:
        """Create the parameters."""
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d, d if i < n - 1 else out) for i in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[..., d]` -> `[..., out]`."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


def sine_position(h: int, w: int, d: int, device: torch.device) -> torch.Tensor:
    """DETR `PositionEmbeddingSine` (normalised, no padding mask): `[1, h * w, d]` float32, rows first."""
    half = d // 2
    scale = 2 * math.pi
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device) / (h + 1e-6) * scale
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device) / (w + 1e-6) * scale
    dim_t = torch.arange(half, dtype=torch.float32, device=device)
    dim_t = POS_TEMPERATURE ** (2 * (dim_t // 2) / half)
    py, px = y[:, None] / dim_t, x[:, None] / dim_t  # [h, half], [w, half]
    py = torch.stack((py[:, 0::2].sin(), py[:, 1::2].cos()), dim=2).flatten(1)
    px = torch.stack((px[:, 0::2].sin(), px[:, 1::2].cos()), dim=2).flatten(1)
    pos = torch.cat((py[:, None, :].expand(h, w, half), px[None, :, :].expand(h, w, half)), dim=2)
    return pos.reshape(1, h * w, d)


class Detr(nn.Module):
    """DETR: image -> class logits `[B, Q, classes + 1]` and boxes `[B, Q, 4]` (sigmoid, centre x/y, w/h)."""

    def __init__(self, a: Arch) -> None:
        """Create the parameters."""
        super().__init__()
        self.backbone = ResNet(a.blocks)
        self.input_proj = nn.Conv2d(2048, a.d_model, 1)
        self.transformer = Transformer(a)
        self.query_embed = nn.Embedding(a.num_queries, a.d_model)
        self.class_embed = nn.Linear(a.d_model, a.num_classes + 1)
        self.bbox_embed = Mlp(a.d_model, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalised `[B, 3, h, w]` -> (logits, boxes)."""
        return self.head(self.input_proj(self.backbone(x)))

    def head(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Projected backbone features `[B, d, h, w]` -> (logits, boxes): the transformer and the prediction heads."""
        b, d, h, w = src.shape
        pos = sine_position(h, w, d, src.device).to(src.dtype)
        mem = src.flatten(2).transpose(1, 2)
        for layer in self.transformer.encoder.layers:  # ty: ignore[not-iterable]
            mem = layer(mem, pos)  # ty: ignore[call-non-callable]
        qpos = self.query_embed.weight.to(mem.dtype).unsqueeze(0).expand(b, -1, -1)
        out = torch.zeros_like(qpos)
        for layer in self.transformer.decoder.layers:  # ty: ignore[not-iterable]
            out = layer(out, mem, pos, qpos)  # ty: ignore[call-non-callable]
        out = self.transformer.decoder.norm(out).to(self.class_embed.weight.dtype)  # ty: ignore[call-non-callable]
        return self.class_embed(out), self.bbox_embed(out).sigmoid()


def fold_frozen_bn(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint state dict -> the same for `Detr`: `backbone.0.body.` -> `backbone.`, BatchNorm folded into the convs.

    `y = (conv(x) - mean) * w / sqrt(var + eps) + b` (computed in float64). A missing or reshaped convolution is left
    as is (the bias still folded), so the strict load names it in the network's keys.
    """
    sd = {k.replace("backbone.0.body.", "backbone."): v for k, v in sd.items()}
    bns = [k.removesuffix(".running_var") for k in sd if k.endswith(".running_var")]
    out = {k: v for k, v in sd.items() if k.rsplit(".", 1)[0] not in bns}
    for bn in bns:
        conv = bn.replace("bn", "conv") if bn.rsplit(".", 1)[-1].startswith("bn") else bn.removesuffix(".1") + ".0"
        out.update(fold_bn_into(sd, bn, conv, FROZEN_BN_EPS))
    return out
