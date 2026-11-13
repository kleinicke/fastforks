"""LoFTR (Sun et al., CVPR 2021): upstream zju3dv/LoFTR `src/loftr` (Apache-2.0), inference only.

fastforks changes: the modules of `backbone/resnet_fpn.py` (ResNetFPN_8_2), `utils/position_encoding.py`,
`loftr_module/{linear_attention,transformer,fine_preprocess}.py`, `utils/{coarse,fine}_matching.py` and `loftr.py` in
one file; einops and kornia (`dsnt.spatial_expectation2d`, `create_meshgrid`) replaced by the same torch operations;
the `data` dict replaced by arguments / return values; training-only code (sinkhorn, padding masks, supervision,
coarse sampling, full attention) removed; the sine position encoding is computed for the feature map's size instead of
sliced from a 256 x 256 table (the same values; inputs larger than 2048 px no longer fail).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            None if stride == 1 else nn.Sequential(conv1x1(in_planes, planes, stride=stride), nn.BatchNorm2d(planes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class ResNetFPN_8_2(nn.Module):  # noqa: N801 - upstream name
    """ResNet + FPN, outputs at 1/8 and 1/2."""

    def __init__(self, initial_dim: int = 128, block_dims: tuple[int, int, int] = (128, 196, 256)) -> None:
        super().__init__()
        self.in_planes = initial_dim
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block_dims[0], stride=1)  # 1/2
        self.layer2 = self._make_layer(block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block_dims[2], stride=2)  # 1/8
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

    def _make_layer(self, dim: int, stride: int) -> nn.Sequential:
        layers = (BasicBlock(self.in_planes, dim, stride=stride), BasicBlock(dim, dim, stride=1))
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x3_out = self.layer3_outconv(x3)
        x3_out_2x = F.interpolate(x3_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x2_out = self.layer2_outconv2(self.layer2_outconv(x2) + x3_out_2x)
        x2_out_2x = F.interpolate(x2_out, scale_factor=2.0, mode="bilinear", align_corners=True)
        x1_out = self.layer1_outconv2(self.layer1_outconv(x1) + x2_out_2x)
        return x3_out, x1_out


def sine_encoding(d_model: int, h: int, w: int, *, temp_bug_fix: bool, device: torch.device) -> torch.Tensor:
    """Upstream `PositionEncodingSine.pe[:, :, :h, :w]` (`temp_bug_fix=False`: the released outdoor model's buggy
    frequencies, upstream issue #41)."""
    y = torch.ones(h, w, device=device).cumsum(0)[None]
    x = torch.ones(h, w, device=device).cumsum(1)[None]
    i = torch.arange(0, d_model // 2, 2, device=device).float()
    if temp_bug_fix:
        div = torch.exp(i * (-math.log(10000.0) / (d_model // 2)))
    else:  # upstream's buggy precedence: (-log(10000) / d_model) // 2
        div = torch.exp(i * (-math.log(10000.0) / d_model // 2))
    div = div[:, None, None]
    pe = torch.zeros(d_model, h, w, device=device)
    pe[0::4] = torch.sin(x * div)
    pe[1::4] = torch.cos(x * div)
    pe[2::4] = torch.sin(y * div)
    pe[3::4] = torch.cos(y * div)
    return pe[None]


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Upstream `LinearAttention` (no masks): `[N, L, H, D]` queries, `[N, S, H, D]` keys / values."""
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    s = v.size(1)
    v = v / s  # upstream: prevents fp16 overflow
    kv = torch.einsum("nshd,nshv->nhdv", k, v)
    z = 1 / (torch.einsum("nlhd,nhd->nlh", q, k.sum(dim=1)) + eps)
    return torch.einsum("nlhd,nhdv,nlh->nlhv", q, kv, z) * s


class LoFTREncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        q = self.q_proj(x).view(bs, -1, self.nhead, self.dim)
        k = self.k_proj(source).view(bs, -1, self.nhead, self.dim)
        v = self.v_proj(source).view(bs, -1, self.nhead, self.dim)
        message = self.merge(linear_attention(q, k, v).reshape(bs, -1, self.nhead * self.dim))
        message = self.norm1(message)
        message = self.norm2(self.mlp(torch.cat([x, message], dim=2)))
        return x + message


class LocalFeatureTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, layer_names: tuple[str, ...]) -> None:
        super().__init__()
        self.layer_names = layer_names
        self.layers = nn.ModuleList([LoFTREncoderLayer(d_model, nhead) for _ in layer_names])

    def forward(self, feat0: torch.Tensor, feat1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.layer_names, strict=True):
            if name == "self":
                feat0, feat1 = layer(feat0, feat0), layer(feat1, feat1)
            else:  # cross: feat1 attends to the updated feat0, as upstream
                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)
        return feat0, feat1


class FinePreprocess(nn.Module):
    def __init__(self, d_model_c: int = 256, d_model_f: int = 128, window: int = 5) -> None:
        super().__init__()
        self.W = window
        self.d_model_f = d_model_f
        self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
        self.merge_feat = nn.Linear(2 * d_model_f, d_model_f, bias=True)

    def forward(
        self,
        feat_f0: torch.Tensor,
        feat_f1: torch.Tensor,
        feat_c0: torch.Tensor,
        feat_c1: torch.Tensor,
        i_ids: torch.Tensor,
        j_ids: torch.Tensor,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.W
        if i_ids.shape[0] == 0:
            empty = feat_f0.new_empty(0, w * w, self.d_model_f)
            return empty, empty
        # fastforks: the window of each selected cell only (upstream unfolds every cell, then selects)
        f0 = self._windows(feat_f0, i_ids, stride)
        f1 = self._windows(feat_f1, j_ids, stride)
        c = self.down_proj(torch.cat([feat_c0[0, i_ids], feat_c1[0, j_ids]], 0))  # [2n, cf]
        merged = self.merge_feat(torch.cat([torch.cat([f0, f1], 0), c[:, None].expand(-1, w * w, -1)], -1))
        return merged.chunk(2, dim=0)

    def _windows(self, feat: torch.Tensor, ids: torch.Tensor, stride: int) -> torch.Tensor:
        """`F.unfold(feat, W, stride, padding=W // 2)[:, :, ids]` as `[n, W * W, C]` (batch 1)."""
        w, r = self.W, self.W // 2
        _, c, _, wf = feat.shape
        wc = (wf + 2 * r - w) // stride + 1
        x = F.pad(feat[0], (r, r, r, r))  # [C, H + 2r, W + 2r]
        oy = (ids // wc) * stride
        ox = (ids % wc) * stride
        d = torch.arange(w, device=feat.device)
        rows = (oy[:, None] + d)[:, :, None].expand(-1, w, w)  # [n, W, W]
        cols = (ox[:, None] + d)[:, None, :].expand(-1, w, w)
        flat = (rows * x.shape[-1] + cols).reshape(-1)
        return x.reshape(c, -1)[:, flat].reshape(c, -1, w * w).permute(1, 2, 0)


class LoFTR(nn.Module):
    """Upstream `LoFTR` with the default config (ResNetFPN 8/2, 4 x self/cross coarse layers, dual-softmax)."""

    def __init__(self, *, temp_bug_fix: bool, thr: float = 0.2, border_rm: int = 2, temperature: float = 0.1) -> None:
        super().__init__()
        self.temp_bug_fix = temp_bug_fix
        self.thr, self.border_rm, self.temperature = thr, border_rm, temperature
        self.backbone = ResNetFPN_8_2()
        self.loftr_coarse = LocalFeatureTransformer(256, 8, ("self", "cross") * 4)
        self.fine_preprocess = FinePreprocess()
        self.loftr_fine = LocalFeatureTransformer(128, 8, ("self", "cross"))

    def features(self, image0: torch.Tensor, image1: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Backbone: coarse (1/8) and fine (1/2) maps of both grayscale `[1, 1, H, W]` images."""
        if image0.shape == image1.shape:  # upstream: one batch when the shapes match
            c, f = self.backbone(torch.cat([image0, image1]))
            return c[:1], c[1:], f[:1], f[1:]
        (c0, f0), (c1, f1) = self.backbone(image0), self.backbone(image1)
        return c0, c1, f0, f1

    def coarse(self, feat_c0: torch.Tensor, feat_c1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Position encoding + coarse transformer: `[1, L, C]`, `[1, S, C]`."""
        seqs = []
        for f in (feat_c0, feat_c1):
            pe = sine_encoding(f.shape[1], *f.shape[-2:], temp_bug_fix=self.temp_bug_fix, device=f.device)
            seqs.append((f + pe.to(f.dtype)).flatten(2).transpose(1, 2))
        return self.loftr_coarse(*seqs)

    def confidence(self, feat_c0: torch.Tensor, feat_c1: torch.Tensor) -> torch.Tensor:
        """Upstream dual-softmax confidence matrix `[1, L, S]`."""
        c = feat_c0.shape[-1]
        sim = torch.einsum("nlc,nsc->nls", feat_c0 / c**0.5, feat_c1 / c**0.5) / self.temperature
        return F.softmax(sim, 1) * F.softmax(sim, 2)

    def coarse_matches(
        self, conf: torch.Tensor, hw0: tuple[int, int], hw1: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upstream `get_coarse_match` (inference): threshold, border removal, mutual maximum -> i, j, confidence."""
        mask = (conf > self.thr).view(1, *hw0, *hw1)
        b = self.border_rm
        if b > 0:
            mask[:, :b] = False
            mask[:, :, :b] = False
            mask[:, :, :, :b] = False
            mask[:, :, :, :, :b] = False
            mask[:, -b:] = False
            mask[:, :, -b:] = False
            mask[:, :, :, -b:] = False
            mask[:, :, :, :, -b:] = False
        mask = mask.view(conf.shape)
        mask = (
            mask
            & (conf == conf.max(dim=2, keepdim=True)[0])
            & (conf == conf.max(dim=1, keepdim=True)[0])
        )
        mask_v, all_j = mask.max(dim=2)
        i_ids = torch.where(mask_v[0])[0]
        j_ids = all_j[0, i_ids]
        return i_ids, j_ids, conf[0, i_ids, j_ids]

    def fine(self, feat_f0: torch.Tensor, feat_f1: torch.Tensor) -> torch.Tensor:
        """Upstream `FineMatching`: the expected offset in the window, normalised to [-1, 1] (x, y), `[M, 2]`."""
        m, ww, c = feat_f0.shape
        w = math.isqrt(ww)
        sim = torch.einsum("mc,mrc->mr", feat_f0[:, ww // 2], feat_f1)
        heat = torch.softmax(sim / c**0.5, dim=1)  # [M, WW]
        g = torch.linspace(-1, 1, w, device=heat.device, dtype=heat.dtype)
        gx, gy = g[None].expand(w, w).reshape(-1), g[:, None].expand(w, w).reshape(-1)
        return torch.stack([heat @ gx, heat @ gy], -1)

    def forward(self, image0: torch.Tensor, image1: torch.Tensor) -> dict[str, torch.Tensor]:
        """Grayscale `[1, 1, H, W]` pair (sides multiples of 8) -> matched points (upstream `mkpts0_f`, `mkpts1_f`)
        and their coarse confidence (`mconf`)."""
        feat_c0, feat_c1, feat_f0, feat_f1 = self.features(image0, image1)
        hw0, hw1 = tuple(feat_c0.shape[-2:]), tuple(feat_c1.shape[-2:])
        feat_c0, feat_c1 = self.coarse(feat_c0, feat_c1)
        conf = self.confidence(feat_c0, feat_c1)
        i_ids, j_ids, mconf = self.coarse_matches(conf.float(), hw0, hw1)
        scale = image0.shape[2] / hw0[0]
        mk0 = torch.stack([i_ids % hw0[1], i_ids // hw0[1]], 1) * scale
        mk1 = torch.stack([j_ids % hw1[1], j_ids // hw1[1]], 1) * scale
        stride = feat_f0.shape[2] // hw0[0]
        f0, f1 = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, i_ids, j_ids, stride)
        if f0.shape[0] > 0:
            f0, f1 = self.loftr_fine(f0, f1)
            w = self.fine_preprocess.W
            mk1 = mk1 + self.fine(f0, f1).float() * (w // 2) * (image0.shape[2] / feat_f0.shape[2])
        return {"keypoints0": mk0.float(), "keypoints1": mk1.float(), "confidence": mconf}
