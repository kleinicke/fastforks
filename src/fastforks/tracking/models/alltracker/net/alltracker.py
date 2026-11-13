# MIT License
#
# Copyright (c) 2025 Adam W. Harley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# fastforks: upstream `nets/alltracker.py` and the blocks of `nets/blocks.py` it uses with its released configurations
# (ConvNeXt encoder with split flow / context features; the tiny model's BasicEncoder with shared features; attention
# over time), inference only: the window loop lives in the runner, only the last refinement step is upsampled (the
# earlier ones are discarded upstream at inference), einops / torchvision removed. Parameter names as upstream (its
# state dicts load unchanged).
"""AllTracker (Harley et al., ICCV 2025): dense point tracking as multi-frame optical flow at 1/8 resolution."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

DIM = 128  # flow / context feature channels (upstream dim = hdim)
CORR_LEVELS, CORR_RADIUS = 5, 4
WINDOW = 16  # frames per window (upstream seqlen / window_len); the time embedding has this length
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of NCHW."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(
            0, 3, 1, 2
        )


class CNBlock2d(nn.Module):
    """ConvNeXt block: 7 x 7 depthwise conv, LayerNorm, MLP (x4, GELU), layer scale, residual."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.block = nn.Sequential(  # indices as upstream (1, 4 and 6 hold its Permute / GELU modules)
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.Identity(),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        b = self.block
        y = b[5](b[4](b[3](b[2](b[0](x).permute(0, 2, 3, 1))))).permute(0, 3, 1, 2)
        return x + self.layer_scale * y


def _down(cin: int, cout: int, *, dilated: bool) -> nn.Sequential:
    """LayerNorm, then 2 x 2 stride-2 conv, or (stage 3, kept at 1/8) its 3 x 3 dilation-2 replacement."""
    conv = nn.Conv2d(cin, cout, 3, padding=2, dilation=2) if dilated else nn.Conv2d(cin, cout, 2, stride=2)
    return nn.Sequential(LayerNorm2d(cin, eps=1e-6), conv)


class ConvNeXt(nn.Module):
    """ConvNeXt-Tiny's first three stages (96 / 192 / 384 channels, 3 / 3 / 9 blocks) at 1/4, 1/8, 1/8."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.features = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, 96, 4, stride=4), LayerNorm2d(96, eps=1e-6)),
            nn.Sequential(*[CNBlock2d(96) for _ in range(3)]),
            _down(96, 192, dilated=False),
            nn.Sequential(*[CNBlock2d(192) for _ in range(3)]),
            _down(192, 384, dilated=True),
            nn.Sequential(*[CNBlock2d(384) for _ in range(9)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.features(x)


class ResidualBlock(nn.Module):
    """Two 3 x 3 convs with instance norm and ReLU, residual (1 x 1 strided projection when downsampling)."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:  # noqa: D107
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1, self.norm2 = nn.InstanceNorm2d(cout), nn.InstanceNorm2d(cout)
        self.downsample = (
            None if stride == 1 else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride), nn.InstanceNorm2d(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = F.relu(self.norm2(self.conv2(F.relu(self.norm1(self.conv1(x))))))
        return F.relu((x if self.downsample is None else self.downsample(x)) + y)


class BasicEncoder(nn.Module):
    """The tiny model's encoder: a RAFT-style ResNet whose four stages are resized to 1/8 and fused."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim // 2, 7, stride=2, padding=3)
        self.norm1, self.norm2 = nn.InstanceNorm2d(dim // 2), nn.InstanceNorm2d(dim * 2)
        chans = (dim // 2, dim // 2, dim // 4 * 3, dim, dim)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            nn.Sequential(
                ResidualBlock(chans[i], chans[i + 1], 1 if i == 0 else 2), ResidualBlock(chans[i + 1], chans[i + 1], 1)
            )
            for i in range(4)
        )
        self.conv2 = nn.Conv2d(dim * 3 + dim // 4, dim * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        size = (x.shape[-2] // 8, x.shape[-1] // 8)
        x = F.relu(self.norm1(self.conv1(x)))
        feats = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            feats.append(F.interpolate(x, size, mode="bilinear", align_corners=True))
        return self.conv3(F.relu(self.norm2(self.conv2(torch.cat(feats, 1)))))


class Attention(nn.Module):
    """Multi-head self-attention (8 heads)."""

    def __init__(self, dim: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, 2 * dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        n, s, c = x.shape
        q = self.to_q(x).view(n, s, self.heads, -1).transpose(1, 2)
        k, v = (t.reshape(n, s, self.heads, -1).transpose(1, 2) for t in self.to_kv(x).chunk(2, -1))
        # two batched GEMMs and a softmax: at 16 tokens of head size 16, 6x faster than the flash / memory-efficient
        # SDPA kernels (fp16, RTX 5060 Ti)
        a = ((q * q.shape[-1] ** -0.5) @ k.transpose(-1, -2)).softmax(-1) @ v
        return self.to_out(a.transpose(1, 2).reshape(n, s, c))


class Mlp(nn.Module):
    """Linear, GELU-tanh, Linear."""

    def __init__(self, dim: int, hidden: int) -> None:  # noqa: D107
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class AttnBlock(nn.Module):
    """Pre-norm transformer block (affine-free LayerNorms)."""

    def __init__(self, dim: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CNBlock1d(nn.Module):
    """Attention over the `S` frames of every 1/8 pixel, layer scale, residual."""

    def __init__(self, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.block = AttnBlock(dim, heads=8)
        self.layer_scale = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor, s: int) -> torch.Tensor:  # noqa: D102
        bs, c, h, w = x.shape
        t = x.reshape(bs // s, s, c, h * w).permute(0, 3, 1, 2).reshape(-1, s, c)  # (b h w) s c
        t = t + self.layer_scale[:, 0] * self.block(t)
        return t.view(bs // s, h * w, s, c).permute(0, 2, 3, 1).reshape(bs, c, h, w)


class MotionEncoder(nn.Module):
    """Correlation and flow encoders, fused; the flow is appended (upstream BasicMotionEncoder, pdim 2)."""

    def __init__(self, corr_channels: int, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.convc1 = nn.Conv2d(corr_channels, dim * 4, 1)
        self.convc2 = nn.Conv2d(dim * 4, dim + dim // 2, 3, padding=1)
        self.convf1 = nn.Conv2d(2, dim * 2, 5, padding=2)
        self.convf2 = nn.Conv2d(dim * 2, dim // 2, 3, padding=1)
        self.conv = nn.Conv2d(dim * 2, dim - 2, 3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:  # noqa: D102
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        return torch.cat([F.relu(self.conv(torch.cat([cor, flo], 1))), flow], 1)


class UpdateBlock(nn.Module):
    """Upstream RelUpdateBlock: motion features, a 1 x 1 compressor, 3 x (attention over time, ConvNeXt block)."""

    def __init__(self, corr_channels: int, dim: int, blocks: int) -> None:  # noqa: D107
        super().__init__()
        self.motion_encoder = MotionEncoder(corr_channels, dim)
        self.compressor = nn.Conv2d(3 * dim + 2, dim, 1)
        self.refine = nn.ModuleList(m for _ in range(blocks) for m in (CNBlock1d(dim), CNBlock2d(dim)))
        self.final_conv = nn.Conv2d(dim, dim, 1)

    def forward(  # noqa: PLR0913, PLR0917 - upstream's arguments
        self, flowfeat: torch.Tensor, ctxfeat: torch.Tensor, visconf: torch.Tensor, corr: torch.Tensor,
        flow: torch.Tensor, s: int,
    ) -> torch.Tensor:  # fmt: skip
        """The new flow features `[B * S, C, H, W]`."""
        x = self.compressor(torch.cat([flowfeat, ctxfeat, self.motion_encoder(flow, corr), visconf], 1))
        for blk in self.refine:
            x = blk(x, s) if isinstance(blk, CNBlock1d) else blk(x)
        return self.final_conv(x)


def _head(dim: int, out: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(dim, 2 * dim, 3, padding=1), nn.ReLU(), nn.Conv2d(2 * dim, out, k, padding=k // 2))


class CorrPyramid:
    """All-pairs correlation of one anchor frame against `S` frames, area-pooled over 5 levels; windowed lookups."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor) -> None:
        """`fmap1` `[C, H, W]` (the anchor), `fmap2` `[S, C, H, W]`."""
        c, h, w = fmap1.shape
        a = fmap1.reshape(c, h * w).T
        self.levels = []
        for _ in range(CORR_LEVELS):
            s, _, h2, w2 = fmap2.shape
            corr = (a @ fmap2.reshape(s, c, h2 * w2)) / torch.tensor(float(c)).sqrt()  # as upstream: divide after
            self.levels.append(corr.reshape(s * h * w, 1, h2, w2))
            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode="area")
        r = torch.linspace(-CORR_RADIUS, CORR_RADIUS, 2 * CORR_RADIUS + 1, device=fmap1.device)
        self.delta = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1)  # (dy, dx) added to (x, y), as upstream

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """`[S, 5 * 81, H, W]` float32 correlations around `coords` `[S, 2, H, W]` (x, y at 1/8)."""
        s, _, h, w = coords.shape
        centre = coords.permute(0, 2, 3, 1).reshape(-1, 1, 1, 2)
        out = []
        for i, corr in enumerate(self.levels):
            h2, w2 = corr.shape[-2:]
            xy = centre / 2**i + self.delta
            scale = torch.tensor((2 / max(w2 - 1, 1), 2 / max(h2 - 1, 1)), device=xy.device)
            g = F.grid_sample(corr, (xy * scale - 1).to(corr.dtype), align_corners=True, padding_mode="border")
            out.append(g.view(s, h, w, -1))
        return torch.cat(out, -1).permute(0, 3, 1, 2).float()


class AllTracker(nn.Module):
    """The network: per-frame encoder and the windowed iterative refinement of flow and visibility at 1/8."""

    def __init__(self, *, tiny: bool) -> None:
        """`tiny`: the BasicEncoder with shared flow / context features; else ConvNeXt with split ones."""
        super().__init__()
        self.tiny = tiny
        if tiny:
            self.fnet = BasicEncoder(DIM)
        else:
            self.cnn = ConvNeXt()
            self.dot_conv = nn.Conv2d(384, 2 * DIM, 1)
        self.upsample_weight = _head(DIM, 64 * 9, 1)
        self.flow_head = _head(DIM, 2, 3)
        self.visconf_head = _head(DIM, 2, 3)
        self.update_block = UpdateBlock(CORR_LEVELS * (2 * CORR_RADIUS + 1) ** 2, DIM, 3)
        self.register_buffer("time_emb", torch.zeros(1, WINDOW, DIM))  # sin-cos, from the checkpoint

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """`[N, C, H/8, W/8]` features of `[N, 3, H, W]` images in `[0, 1]` (H, W multiples of 8)."""
        mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(STD).view(1, 3, 1, 1)
        x = (images - mean) / std
        return self.fnet(x) if self.tiny else self.dot_conv(self.cnn(x))

    def time_embed(self, s: int) -> torch.Tensor:
        """`[S, C]` time embedding of a window of `s` frames (upstream fetch_time_embed at inference)."""
        e = self.time_emb[0]
        if s == WINDOW:
            return e
        if s == 1:
            return e[1:2]
        return F.interpolate(e.float().T[None], size=s, mode="linear")[0].T

    def window(  # noqa: PLR0913, PLR0917 - upstream forward_window's state
        self, anchor: torch.Tensor, fmaps: torch.Tensor, flows8: torch.Tensor, visconfs8: torch.Tensor, iters: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine flow and visibility logits `[S, 2, h, w]` of `fmaps` `[S, C, h, w]` against `anchor` `[C, h, w]`.

        Returns:
            The new `flows8`, `visconfs8` and the upsampling weights `[S, 576, h, w]` of the last step.
        """
        s, _, h, w = fmaps.shape
        corr_fn = CorrPyramid(anchor, fmaps)
        ys, xs = torch.meshgrid(
            torch.arange(h, device=anchor.device, dtype=torch.float32),
            torch.arange(w, device=anchor.device, dtype=torch.float32),
            indexing="ij",
        )
        coords1 = torch.stack((xs, ys))[None]  # the flow state stays float32
        f1 = anchor[None].expand(s, -1, -1, -1)
        flowfeat, ctxfeat = (f1, f1) if self.tiny else f1.split(DIM, 1)
        ctxfeat = ctxfeat + self.time_embed(s).to(ctxfeat.dtype)[:, :, None, None]
        for _ in range(iters):
            corr = corr_fn(coords1 + flows8).to(anchor.dtype)
            flowfeat = self.update_block(flowfeat, ctxfeat, visconfs8, corr, flows8, s)
            flows8 = flows8 + self.flow_head(flowfeat)
            visconfs8 = visconfs8 + self.visconf_head(flowfeat)
        return flows8, visconfs8, 0.25 * self.upsample_weight(flowfeat)


def upsample(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`[N, 2, 8h, 8w]`: convex 3 x 3 combination of `8 * data` `[N, 2, h, w]` with the weights `[N, 576, h, w]`."""
    n, c, h, w = data.shape
    m = mask.view(n, 1, 9, 8, 8, h, w).softmax(2)
    up = F.unfold(8 * data, [3, 3], padding=1).view(n, c, 9, 1, 1, h, w)
    return (m * up).sum(2).permute(0, 1, 4, 2, 5, 3).reshape(n, c, 8 * h, 8 * w).to(data.dtype)
