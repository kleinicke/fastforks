"""Heads: the DB probability map (`det_db_head.py`; `PFHeadLocal` for the server detector) and the CTC recognition
head (`rec_multi_head.py` CTC branch: `EncoderWithSVTR` + `CTCHead`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ConvBN, LegacyBatchNorm, up


class _DBBranch(nn.Module):
    """`Head`: conv 3x3 + BN + ReLU, 2x deconv + BN + ReLU (-> `f`), 2x deconv to one channel, sigmoid."""

    def __init__(self, c: int) -> None:
        super().__init__()
        q = c // 4
        self.conv1 = nn.Conv2d(c, q, 3, padding=1, bias=False)
        self.conv_bn1 = LegacyBatchNorm(q)
        self.conv2 = nn.ConvTranspose2d(q, q, 2, 2)
        self.conv_bn2 = LegacyBatchNorm(q)
        self.conv3 = nn.ConvTranspose2d(q, 1, 2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = F.relu(self.conv_bn2(self.conv2(F.relu(self.conv_bn1(self.conv1(x))))))
        return torch.sigmoid(self.conv3(f)), f


class DBHead(nn.Module):
    """Shrink map `[B, 1, H, W]` in [0, 1]; `local` adds PFHeadLocal's refinement (mode large): the mean of both."""

    def __init__(self, c: int, local: bool) -> None:
        super().__init__()
        self.binarize = _DBBranch(c)  # upstream also creates `thresh` (training only; not exported)
        if local:
            q = c // 4
            self.last_3 = ConvBN(q + 1, q, 3, act=nn.ReLU(), bn=LegacyBatchNorm)
            self.last_1 = nn.Conv2d(q, 1, 1)
        self.local = local

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shrink, f = self.binarize(x)
        if not self.local:
            return shrink
        cbn = torch.sigmoid(self.last_1(self.last_3(torch.cat([shrink, up(f, 2)], 1))))
        return 0.5 * (shrink + cbn)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int = 8, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.mixer = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.fc2(F.silu(self.fc1(self.norm2(x))))


class CTCHead(nn.Module):
    """`EncoderWithSVTR` (dims 120, depth 2, kernel 1x3) + `Im2Seq` + `CTCHead`: `[B, C, 1, T]` -> probs `[B, T, K]`."""

    def __init__(self, cin: int, classes: int, dims: int = 120, hidden: int = 120, depth: int = 2) -> None:
        super().__init__()
        r = cin // 8
        self.conv1 = ConvBN(cin, r, (1, 3), act=nn.SiLU())
        self.conv2 = ConvBN(r, hidden, 1, act=nn.SiLU())
        self.blocks = nn.ModuleList(_Block(hidden) for _ in range(depth))
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.conv3 = ConvBN(hidden, cin, 1, act=nn.SiLU())
        self.conv4 = ConvBN(2 * cin, r, (1, 3), act=nn.SiLU())
        self.conv1x1 = ConvBN(r, dims, 1, act=nn.SiLU())
        self.fc = nn.Linear(dims, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv2(self.conv1(x))
        b, c, h, w = z.shape
        t = z.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            t = blk(t)
        z = self.norm(t).reshape(b, h, w, c).permute(0, 3, 1, 2)
        z = self.conv1x1(self.conv4(torch.cat([x, self.conv3(z)], 1)))
        return self.fc(z.squeeze(2).transpose(1, 2)).softmax(-1)
