"""OMNI-DC network (upstream `src/model`, model OGNIDC with the released demo configuration).

Upstream (BSD-3-Clause): an RGB-D backbone (convs, the first two ResNet-34 stages, a four-stage PVT whose blocks add a
CBAM residual branch, a transposed-conv decoder) predicts, at 1/4 resolution, a GRU context, a confidence for the
sparse depth, and at full resolution a DySPN guide and confidence. Depth Anything V2 Large gives a relative depth map as
an extra input channel. One GRU step predicts log-depth gradients at three resolutions; a conjugate-gradient solve
integrates them with the sparse depth (`solve`), a convex 4x upsampling and six DySPN propagation steps refine it.

Same parameter names as upstream (`backbone.*`, `depth_module.*`, `update_block.*`, `prop_layer.*`); the unused
learnable-weights head and the fixed DySPN offsets of the checkpoint are dropped. Configuration fixed to upstream
`demo.sh` (3 resolutions, uniform gradient weights, exp depth, median whitening, one GRU iteration, min-pooled depth).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.depth_anything_v2.net.dpt import DepthAnythingV2

RES = 3  # fixed: upstream num_resolution (gradients at 1, 1/2, 1/4 of the 1/4-resolution grid)
DOWN = 4  # fixed: upstream backbone_output_downsample_rate
MULTIPLE = DOWN * 2 ** (RES - 1)  # fixed: upstream pads both sides to a multiple of this (16)
HIDDEN = 64  # fixed: upstream gru_hidden_dim / gru_context_dim
PROP = 6  # fixed: upstream prop_time (DySPN steps)
NUM = 5  # fixed: DySPN samples per step
OFFSETS = ((0, -1), (-1, 0), (0, 0), (1, 0), (0, 1))  # fixed: upstream DySPN offset_3x3 for num = 5, (x, y)
LAMDA = 5.0  # fixed: upstream integration_alpha (weight of the sparse-depth term)
CLAMP = 1.0  # fixed: upstream optim_layer_input_clamp
GRAD_W = 1 / RES  # fixed: upstream "uniform" gradient weights (clamped to >= 1e-4 by the solver: no-op)
RTOL = 1e-5  # fixed: upstream solver rtol
MAXITER = 5000  # fixed: upstream solver maxiter
STALL_TOL, STALL_STEPS = 1e-2, 10  # fixed: upstream cg_batch early stopping (improvement below tol for 10 steps)
INVALID = 1e5  # fixed: upstream min-pooling fill for missing depth
DAV2_SIZE = 518  # fixed: Depth Anything V2 input (shorter side, rounded to a multiple of 14)
QUANT = (0.02, 0.98)  # fixed: upstream normalises the Depth Anything V2 map by these quantiles


def _conv(i: int, o: int, k: int = 3, *, bias: bool = True, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(i, o, k, stride, k // 2, bias=bias)


def _cbr(i: int, o: int, *, bn: bool = True, relu: bool = True) -> nn.Sequential:
    """Upstream `conv_bn_relu` (3x3)."""
    layers: list[nn.Module] = [_conv(i, o, bias=not bn)]
    if bn:
        layers.append(nn.BatchNorm2d(o))
    if relu:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _ChannelAttention(nn.Module):
    def __init__(self, c: int, ratio: int) -> None:
        """Build the layers."""
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // ratio, 1, bias=False), nn.ReLU(), nn.Conv2d(c // ratio, c, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        return torch.sigmoid(self.fc(x.mean((2, 3), keepdim=True)) + self.fc(x.amax((2, 3), keepdim=True)))


class _SpatialAttention(nn.Module):
    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        return torch.sigmoid(self.conv1(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))


class CBAMBlock(nn.Module):
    """Upstream `resnet_cbam.BasicBlock` (stride 1): conv-BN-ReLU-conv-BN, channel and spatial attention, residual."""

    def __init__(self, c: int, ratio: int) -> None:
        """Build the layers."""
        super().__init__()
        self.conv1 = _conv(c, c, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = _conv(c, c, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.ca = _ChannelAttention(c, ratio)
        self.sa = _SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return F.relu(out + x)


class ResBlock(nn.Module):
    """torchvision ResNet `BasicBlock` (the ResNet-34 stages upstream copies)."""

    def __init__(self, i: int, o: int, stride: int = 1) -> None:
        """Build the layers."""
        super().__init__()
        self.conv1 = _conv(i, o, bias=False, stride=stride)
        self.bn1 = nn.BatchNorm2d(o)
        self.conv2 = _conv(o, o, bias=False)
        self.bn2 = nn.BatchNorm2d(o)
        self.downsample = (
            nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o)) if stride != 1 or i != o else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class Attention(nn.Module):
    """PVT spatial-reduction attention (keys / values from a strided conv of the tokens)."""

    def __init__(self, dim: int, heads: int, sr: int) -> None:
        """Build the layers."""
        super().__init__()
        self.heads, self.sr_ratio = heads, sr
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        if sr > 1:
            self.sr = nn.Conv2d(dim, dim, sr, sr)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply the module."""
        b, n, c = x.shape
        q = self.q(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.sr_ratio > 1:
            x = self.norm(self.sr(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2))
        kv = self.kv(x).view(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(o.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    """Transformer MLP (GELU)."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Build the layers."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Upstream PVT `Block`: a transformer block plus a CBAM branch on its input, fused by a 3x3 conv."""

    def __init__(self, dim: int, heads: int, mlp: int, sr: int) -> None:
        """Build the layers."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads, sr)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * mlp)
        self.resblock = CBAMBlock(dim, 16)
        self.concat_conv = _conv(2 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Tokens `[B, h * w, C]` -> `[B, C, h, w]`."""
        b, _, c = x.shape
        img = x.transpose(1, 2).reshape(b, c, h, w)
        x = x + self.attn(self.norm1(x), h, w)
        x = x + self.mlp(self.norm2(x))
        return self.concat_conv(torch.cat([x.transpose(1, 2).reshape(b, c, h, w), self.resblock(img)], 1))


class PatchEmbed(nn.Module):
    """Stride-2 2x2 conv patch embedding + LayerNorm -> tokens and grid size."""

    def __init__(self, i: int, o: int) -> None:
        """Build the layers."""
        super().__init__()
        self.proj = nn.Conv2d(i, o, 2, 2)
        self.norm = nn.LayerNorm(o)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Apply the module."""
        x = self.proj(x)
        h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


LAST = 3  # fixed: upstream's last PVT stage has a cls position (dropped)


class PVT(nn.Module):
    """Upstream `PVT` (in_chans 64, patch 2): ResNet-34 layer1 / layer2, then four PVT stages; six feature maps."""

    DIMS = (64, 128, 320, 512)
    HEADS = (1, 2, 5, 8)
    MLP = (8, 8, 4, 4)
    SR = (8, 4, 2, 1)
    DEPTHS = (3, 4, 6, 3)
    GRIDS = (112, 28, 14, 7)  # fixed: the position tables' grids (upstream img_size 224)

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.embed_layer1 = nn.Sequential(*[ResBlock(64, 64) for _ in range(3)])
        self.embed_layer2 = nn.Sequential(ResBlock(64, 128, 2), *[ResBlock(128, 128) for _ in range(3)])
        c = 128
        for i, d in enumerate(self.DIMS):
            setattr(self, f"patch_embed{i + 1}", PatchEmbed(c, d))
            setattr(self, f"pos_embed{i + 1}", nn.Parameter(torch.zeros(1, self.GRIDS[i] ** 2 + (i == LAST), d)))
            blocks = [Block(d, self.HEADS[i], self.MLP[i], self.SR[i]) for _ in range(self.DEPTHS[i])]
            setattr(self, f"block{i + 1}", nn.ModuleList(blocks))
            c = d

    def _pos(self, i: int, h: int, w: int) -> torch.Tensor:
        pos = getattr(self, f"pos_embed{i + 1}")
        if i == LAST:
            pos = pos[:, 1:]
        if h * w == self.GRIDS[0] ** 2:  # upstream compares every stage with the first one's patch count
            return pos
        g = self.GRIDS[i]
        grid = pos.reshape(1, g, g, -1).permute(0, 3, 1, 2)
        return F.interpolate(grid, (h, w), mode="bilinear").flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply the module."""
        outs = [self.embed_layer1(x)]
        outs.append(self.embed_layer2(outs[0]))
        x = outs[1]
        for i in range(4):
            t, h, w = getattr(self, f"patch_embed{i + 1}")(x)
            t = t + self._pos(i, h, w).to(t.dtype)
            for blk in getattr(self, f"block{i + 1}"):
                x = blk(t, h, w)
                t = x.flatten(2).transpose(1, 2)
            outs.append(x)
        return outs


def _dec(i: int, o: int, ratio: int) -> nn.Sequential:
    """Upstream decoder stage: ConvTranspose (x2) + BN + ReLU, then a CBAM block."""
    up = nn.Sequential(nn.ConvTranspose2d(i, o, 3, 2, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
    return nn.Sequential(up, CBAMBlock(o, ratio))


def _cat(fd: torch.Tensor, fe: torch.Tensor) -> torch.Tensor:
    """Upstream `Backbone._concat`: the decoder feature resized (bilinear, align_corners) to the encoder's + concat."""
    if fd.shape[-2:] != fe.shape[-2:]:
        fd = F.interpolate(fd, fe.shape[-2:], mode="bilinear", align_corners=True)
    return torch.cat([fd, fe], 1)


class Backbone(nn.Module):
    """Upstream `Backbone` (mode rgbd, cformer): guide, DySPN confidence, GRU context, sparse-depth confidence."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.conv1_rgb = _cbr(3, 48, bn=False)
        self.conv1_dep = _cbr(2, 16, bn=False)
        self.conv1 = _cbr(64, 64, bn=False)
        self.former = PVT()
        self.dec6 = _dec(512, 256, 16)
        self.dec5 = _dec(256 + 320, 128, 8)
        self.dec4 = _dec(128 + 128, 64, 4)
        self.dec3 = _dec(64 + 64, 64, 4)
        self.dec2 = _dec(64 + 128, 64, 4)
        self.gd_dec1 = _cbr(128, 64)
        self.gd_dec0 = _cbr(128, NUM * PROP, bn=False, relu=False)
        self.cf_dec1 = _cbr(128, 32)
        self.cf_dec0 = _conv(32 + 64, 1)
        self.ctx_dec1 = _cbr(128, 64)
        self.ctx_dec0 = _cbr(64, 2 * HIDDEN, bn=False, relu=False)
        self.cfi_dec1 = _cbr(128, 32)
        self.cfi_dec0 = nn.Sequential(_conv(32, 1), nn.Sigmoid())

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Image `[B, 3, H, W]`, depth input `[B, 2, H, W]` -> guide, DySPN confidence, context, input confidence."""
        fe1 = self.conv1(torch.cat([self.conv1_rgb(rgb), self.conv1_dep(depth)], 1))
        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        fd5 = self.dec5(_cat(self.dec6(fe7), fe6))
        fd4 = self.dec4(_cat(fd5, fe5))
        fd2 = self.dec2(_cat(self.dec3(_cat(fd4, fe4)), fe3))
        f2 = _cat(fd2, fe2)
        guide = self.gd_dec0(_cat(self.gd_dec1(f2), fe1))
        conf_spn = self.cf_dec0(_cat(self.cf_dec1(f2), fe1))
        f4 = _cat(fd4, fe4)
        return guide, conf_spn, self.ctx_dec0(self.ctx_dec1(f4)), self.cfi_dec0(self.cfi_dec1(f4))


class _GradBlock(nn.Module):
    def __init__(self, i: int, o: int) -> None:
        """Build the layers."""
        super().__init__()
        self.block = nn.Sequential(_conv(i, o, bias=False), nn.ReLU(), _conv(o, o, bias=False))
        self.skip = nn.Sequential(nn.Conv2d(i, o, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        return F.relu(self.block(x) + self.skip(x))


class DepthGradHead(nn.Module):
    """Upstream `DepthGradHead`: residual conv blocks, one 1x1 decoder (x / y gradient) per resolution."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.blocks = nn.ModuleList([_GradBlock(HIDDEN, 128)] + [_GradBlock(128, 128) for _ in range(RES - 1)])
        self.decoders = nn.ModuleList([nn.Conv2d(128, 2, 1) for _ in range(RES)])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Apply the module."""
        out = []
        for blk, dec in zip(self.blocks, self.decoders, strict=True):
            x = blk(x)
            out.append(dec(x))
            x = F.max_pool2d(x, 2)
        return out


class DepthEncoder(nn.Module):
    """Upstream `BasicDepthEncoder`."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.convd1 = nn.Conv2d(1, 64, 7, padding=3)
        self.convd2 = _conv(64, 32)
        self.convg1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convg2 = _conv(64, 32)
        self.conv = _conv(64, 64 - 3)

    def forward(self, depth: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        dep = F.relu(self.convd2(F.relu(self.convd1(depth))))
        gra = F.relu(self.convg2(F.relu(self.convg1(grad))))
        return torch.cat([F.relu(self.conv(torch.cat([dep, gra], 1))), depth, grad], 1)


class SepConvGRU(nn.Module):
    """Upstream `SepConvGRU`: a horizontal (1x5), then a vertical (5x1) GRU update."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        c = HIDDEN + 64 + HIDDEN
        for d in ("z", "r", "q"):
            setattr(self, f"conv{d}1", nn.Conv2d(c, HIDDEN, (1, 5), padding=(0, 2)))
            setattr(self, f"conv{d}2", nn.Conv2d(c, HIDDEN, (5, 1), padding=(2, 0)))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        for i in (1, 2):
            hx = torch.cat([h, x], 1)
            z = torch.sigmoid(getattr(self, f"convz{i}")(hx))
            r = torch.sigmoid(getattr(self, f"convr{i}")(hx))
            q = torch.tanh(getattr(self, f"convq{i}")(torch.cat([r * h, x], 1)))
            h = (1 - z) * h + z * q
        return h


class UpdateBlock(nn.Module):
    """Upstream `BasicUpdateBlock`: one GRU step from zero -> log-depth gradients `[B, 6, h, w]`, upsample mask."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.encoder = DepthEncoder()
        self.gru = SepConvGRU()
        self.depth_grad_head = DepthGradHead()
        self.mask = nn.Sequential(_conv(HIDDEN, 256), nn.ReLU(), nn.Conv2d(256, DOWN * DOWN * 9, 1))

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Context `[B, 128, h, w]` -> clamped gradients (corner-aligned per resolution), mask `[B, 144, h, w]`."""
        net, inp = torch.split(context, [HIDDEN, HIDDEN], 1)
        b, _, h, w = context.shape
        zero = context.new_zeros((b, 3, h, w))  # the whitened log depth and its gradients start at zero
        net = self.gru(torch.tanh(net), torch.cat([F.relu(inp), self.encoder(zero[:, :1], zero[:, 1:])], 1))
        raw = self.depth_grad_head(net)
        grads = [F.pad(g, (0, w - g.shape[-1], 0, h - g.shape[-2])) for g in raw]
        return torch.cat(grads, 1).clamp(-CLAMP, CLAMP), self.mask(net)


class _Prop(nn.Module):
    """Upstream `DySPN_Module`'s learned part: offsets and affinities from the guide."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_offset_aff = _conv(NUM * PROP, 3 * NUM * PROP)


class OmniDCNet(nn.Module):
    """The checkpoint's modules; the steps are `mono`, `backbone`, `update_block`, `solve`, `upsample`, `spn`."""

    def __init__(self) -> None:
        """Build the layers."""
        super().__init__()
        self.backbone = Backbone()
        self.depth_module = DepthAnythingV2("vitl", 256, [256, 512, 1024, 1024])
        self.update_block = UpdateBlock()
        self.prop_layer = _Prop()

    @staticmethod
    def checkpoint_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Upstream keys minus the unused gradient-weights head and the fixed DySPN offsets."""
        drop = ("update_block.grad_weights_head.", "prop_layer.offset_3x3")
        return {k: v for k, v in sd.items() if not k.startswith(drop)}

    def fold_bn(self) -> None:
        """Fold every inference BatchNorm into the conv before it (the BN becomes an identity)."""
        from torch.nn.utils.fusion import fuse_conv_bn_eval  # noqa: PLC0415

        def fold(parent: nn.Module, conv: str, bn: str) -> None:
            c, n = getattr(parent, conv), getattr(parent, bn)
            setattr(parent, conv, fuse_conv_bn_eval(c, n, transpose=isinstance(c, nn.ConvTranspose2d)))
            setattr(parent, bn, nn.Identity())

        for m in list(self.backbone.modules()):
            if isinstance(m, (CBAMBlock, ResBlock)):
                fold(m, "conv1", "bn1")
                fold(m, "conv2", "bn2")
            if isinstance(m, nn.Sequential) and len(m) > 1 and isinstance(m[1], nn.BatchNorm2d):
                fold(m, "0", "1")

    def mono(self, x: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalised padded image `[B, 3, H, W]` -> Depth Anything V2 map scaled to [0, 1] (2 % / 98 %)."""
        h, w = x.shape[-2:]
        y = F.interpolate(x, mono_size(h, w), mode="bicubic", align_corners=True)
        d = self.depth_module(y.to(self.depth_module.pretrained.cls_token.dtype))[:, None].float()
        return quantile_normalise(F.interpolate(d, (h, w), mode="bilinear", align_corners=True))

    def upsample(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Upstream `upsample_depth`: convex 4x upsampling of `[B, 1, h, w]` with the mask `[B, 144, h, w]`."""
        b, _, h, w = x.shape
        m = mask.float().view(b, 1, 9, DOWN, DOWN, h, w).softmax(2)
        u = F.unfold(x, 3, padding=1).view(b, 1, 9, 1, 1, h, w)
        return (m * u).sum(2).permute(0, 1, 4, 2, 5, 3).reshape(b, 1, DOWN * h, DOWN * w)

    def spn(self, depth: torch.Tensor, guide: torch.Tensor, sparse: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
        """Upstream `DySPN_Module` (mode yx): six steps of 5-sample deformable propagation, sparse depth re-injected."""
        b, _, h, w = depth.shape
        oa = self.prop_layer.conv_offset_aff(guide.float())
        off = oa[:, : 2 * NUM * PROP].view(b, PROP, NUM, 2, h, w)
        aff = oa[:, 2 * NUM * PROP :].view(b, PROP, NUM, h, w).softmax(2)
        o3 = torch.tensor(OFFSETS, dtype=torch.float32, device=depth.device)
        ys = torch.arange(h, dtype=torch.float32, device=depth.device)
        xs = torch.arange(w, dtype=torch.float32, device=depth.device)
        # grid_sample (align_corners False) at pixel (x + dx + ox, y + dy + oy): normalised (2 p + 1) / size - 1
        gx = ((off[:, :, :, 1] + o3[:, 0, None, None]) * 2 + (2 * xs - w + 1)) / w
        gy = ((off[:, :, :, 0] + o3[:, 1, None, None]) * 2 + (2 * ys[:, None] - h + 1)) / h
        grid = torch.stack([gx, gy], -1).view(b, PROP, NUM * h, w, 2)
        c = torch.sigmoid(conf.float()) * sparse.sign()
        cs = c * sparse
        x = depth.float()
        for i in range(PROP):
            s = F.grid_sample(x, grid[:, i], mode="bilinear", padding_mode="zeros", align_corners=False)
            x = (1 - c) * (s.view(b, NUM, h, w) * aff[:, i]).sum(1, keepdim=True) + cs
        return x


def mono_size(h: int, w: int) -> tuple[int, int]:
    """Upstream `resize_image`: the shorter side to 518, both sides rounded (half to even) to a multiple of 14."""
    s = max(DAV2_SIZE / h, DAV2_SIZE / w)
    return round(s * h / 14) * 14, round(s * w / 14) * 14


def quantile_normalise(d: torch.Tensor) -> torch.Tensor:
    """`relu(d)` mapped so its 2 % / 98 % quantiles (per image) are 0 / 1."""
    d = F.relu(d)
    q = torch.quantile(d.flatten(1), torch.tensor(QUANT, device=d.device), dim=1)
    lo, hi = q[0].view(-1, 1, 1, 1), q[1].view(-1, 1, 1, 1)
    return (d - lo) / (hi - lo)


def sparse_inputs(dep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse depth `[B, 1, H, W]` (0: missing) -> network log depth (over the median), 1/4 log depth, 1/4 mask."""
    valid = dep > 0
    n = valid.flatten(1).sum(1)
    v = torch.where(valid, dep, torch.inf).flatten(1).sort(1).values
    med = v.gather(1, ((n - 1).clamp_min(0) // 2)[:, None])  # torch.median: the lower middle value
    med = torch.where(n[:, None] > 0, med, 1).view(-1, 1, 1, 1)
    net_in = torch.where(valid, torch.log(dep / med), 0)
    small = -F.max_pool2d(-torch.where(valid, dep, INVALID), DOWN)
    mask = F.max_pool2d(valid.float(), DOWN) > 0
    return net_in, torch.where(mask, torch.log(small), 0), mask.float()


def _at(y_u: list[torch.Tensor], y_v: list[torch.Tensor]) -> torch.Tensor:
    """A^T of the multi-resolution finite differences (upstream `FastFiniteDiffMatrix.bmm_transposed`), image layout."""
    out = None
    for i in range(RES):
        r = 2**i
        u = y_u[i].repeat_interleave(r, 2).repeat_interleave(r, 3) / r**2
        v = y_v[i].repeat_interleave(r, 2).repeat_interleave(r, 3) / r**2
        t = F.pad(u, (r, 0)) - F.pad(u, (0, r)) + F.pad(v, (0, 0, r, 0)) - F.pad(v, (0, 0, 0, r))
        out = t if out is None else out + t
    assert out is not None  # noqa: S101 - RES > 0
    return out


def _a(x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The multi-resolution finite differences (upstream `FastFiniteDiffMatrix.bmm`), per resolution u / v maps."""
    us, vs = [], []
    for i in range(RES):
        r = 2**i
        du, dv = x[..., r:] - x[..., :-r], x[..., r:, :] - x[..., :-r, :]
        us.append(F.avg_pool2d(du, r) if r > 1 else du)
        vs.append(F.avg_pool2d(dv, r) if r > 1 else dv)
    return us, vs


def system(
    grad: torch.Tensor, dep: torch.Tensor, mask: torch.Tensor, conf_in: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream `DepthGradOptimLayer`'s normal equations `(A^T C A + diag(d)) x = b`: returns `d` and `b`.

    `grad` `[B, 6, h, w]` (resolution i's x / y gradients in the top-left `h / 2^i x w / 2^i` corner; the first column /
    row of each is no equation), `dep` / `mask` the 1/4 log depth and its mask, `conf_in` the sparse-depth confidence.
    """
    h, w = grad.shape[-2:]
    rhs_u = [grad[:, 2 * i : 2 * i + 1, : h >> i, 1 : w >> i] * GRAD_W for i in range(RES)]
    rhs_v = [grad[:, 2 * i + 1 : 2 * i + 2, 1 : h >> i, : w >> i] * GRAD_W for i in range(RES)]
    d = LAMDA * conf_in * mask
    return d, _at(rhs_u, rhs_v) + d * dep


def normal_op(p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """`(A^T C A + diag(d)) p` (C: the uniform gradient weights)."""
    us, vs = _a(p)
    return _at([u * GRAD_W for u in us], [v * GRAD_W for v in vs]) + d * p


class CGState:
    """Upstream `cg_batch` (X0 = 0, no preconditioner) as fixed-shape in-place steps: no host sync inside a step.

    The batch stays coupled as upstream's: early stopping looks at the largest relative improvement over the batch, the
    tolerance stop needs every image. `x` holds the iterate upstream returns once `done` is set (it stops changing).
    """

    def __init__(self, d: torch.Tensor, b: torch.Tensor) -> None:
        """Build the layers."""
        self.d, self.b = d, b
        z = torch.zeros_like(b)
        n = b.shape[0]
        self.x, self.ax, self.r, self.p = z, z.clone(), z.clone(), z.clone()
        self.rr, self.rr_old, self.last, self.stop = (b.new_zeros(n) for _ in range(4))
        self.k = torch.zeros((), dtype=torch.int32, device=b.device)
        self.stall = torch.zeros_like(self.k)
        self.done = torch.zeros((), dtype=torch.bool, device=b.device)
        self.reset()

    def reset(self) -> None:
        """Start from X0 = 0 for the current `d` / `b` (in place)."""
        b = self.b
        self.x.zero_()
        self.ax.zero_()  # upstream A(X0) = 0
        self.p.zero_()
        self.r.copy_(b)
        self.rr.copy_((b * b).flatten(1).sum(1))
        self.rr_old.fill_(1)
        self.last.copy_(b.flatten(1).norm(dim=1))
        self.stop.copy_(RTOL * self.last)
        self.k.zero_()
        self.stall.zero_()
        self.done.zero_()

    def step(self) -> None:
        """One CG iteration (in place)."""
        s = (-1, 1, 1, 1)
        self.k += 1
        beta = self.rr / torch.where(self.rr_old == 0, 1e-8, self.rr_old)
        self.p.copy_(torch.where(self.k == 1, self.r, self.r + beta.view(s) * self.p))
        ap = normal_op(self.p, self.d)
        den = (self.p * ap).flatten(1).sum(1)
        alpha = (self.rr / torch.where(den == 0, 1e-8, den)).view(s)
        self.x.copy_(torch.where(self.done, self.x, self.x + alpha * self.p))
        self.r.sub_(alpha * ap)
        self.ax.add_(alpha * ap)
        self.rr_old.copy_(self.rr)
        self.rr.copy_((self.r * self.r).flatten(1).sum(1))
        res = (self.ax - self.b).flatten(1).norm(dim=1)
        imp = ((self.last - res) / self.last).amax()
        self.last.copy_(res)
        self.stall.copy_(torch.where(imp < STALL_TOL, self.stall + 1, 0))
        stop = ((self.k > STALL_STEPS) & (self.stall >= STALL_STEPS)) | (res <= self.stop).all() | (self.k >= MAXITER)
        self.done |= stop


def solve(d: torch.Tensor, b: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """Run `CGState` steps in chunks with one host sync per chunk; the same iterate as upstream's loop."""
    st = CGState(d, b)
    while True:
        for _ in range(chunk):
            st.step()
        if bool(st.done):
            return st.x
