# MIT License
#
# Copyright (c) 2025 Görkay Aydemir
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

"""Track-On2 inference path in plain PyTorch, with the checkpoint's parameter names.

From gorkaydemir/track_on `model/`: `trackon.py` (`Track_On2.track_frame`, `multiscale_correlation`,
`memory_extension`), `backbone.py`, `modules.py` (`MHA_Block`, `DMSMHA_Block`, `SimpleFPN`), `reranking.py`,
`prediction_head.py` and `vit_adapter/dinov3_adapter/` (ViT-Adapter around the frozen DINOv3 ViT-S+/16; its blocks run
from `net/dinov3.py`). mmcv's `MultiScaleDeformableAttention` is `MSDA` with its PyTorch path (`grid_sample`,
bilinear, zero padding, `align_corners=False`); mmcv adds the query it was given (its `identity`) to the output, the
callers here add it explicitly. Dropouts and training paths (random memory masking, the uncertainty / top-k score
heads' outputs) are dropped; their parameters stay for strict loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from .dinov3 import CONFIGS, PATCH, REGISTERS, DINOv3ViT

D = 256  # fixed: upstream args.D
HEADS = 4  # fixed: Track_On2 nhead
TOPK = 16  # fixed: upstream args.K: re-ranking candidates
LAYERS = 3  # fixed: decoder / re-ranking / prediction-head layers
STRIDE = 4  # fixed: feature stride of the finest level
POINTS = 4  # fixed: deformable attention points per head and level
ADAPTER_HEADS = 6  # fixed: ViTAdapter deform_num_heads (ViT-S)
INTERACTIONS = ((0, 2), (3, 5), (6, 8), (9, 11))  # fixed: ViT blocks of each adapter interaction
ADAPTER_EPS = 1e-6  # fixed: adapter LayerNorm eps
_MEAN = (0.485, 0.456, 0.406)  # fixed: ImageNet normalisation
_STD = (0.229, 0.224, 0.225)  # fixed: ImageNet normalisation

Shapes = list[tuple[int, int]]


def msda_sample(value: torch.Tensor, shapes: Shapes, loc: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """mmcv `multi_scale_deformable_attn_pytorch`: `value` `[B, S, h, dh]`, `loc` `[B, Q, h, L, P, 2]` in `[0, 1]`,
    `weights` `[B, Q, h, L, P]` -> `[B, Q, h * dh]`."""  # noqa: D205
    b, _, h, dh = value.shape
    q, p = loc.shape[1], loc.shape[4]
    grids = 2 * loc - 1
    out = None
    start = 0
    for lvl, (hl, wl) in enumerate(shapes):
        v = value[:, start : start + hl * wl].permute(0, 2, 3, 1).reshape(b * h, dh, hl, wl)
        start += hl * wl
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1).to(v.dtype)  # [B h, Q, P, 2]
        s = F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)  # [B h, dh, Q, P]
        t = (s * weights[:, :, :, lvl].transpose(1, 2).reshape(b * h, 1, q, p).to(s.dtype)).sum(-1)
        out = t if out is None else out + t
    return cast("torch.Tensor", out).view(b, h * dh, q).transpose(1, 2)


class MSDA(nn.Module):
    """mmcv `MultiScaleDeformableAttention` (batch first, value_proj_ratio 1) without its identity residual."""

    def __init__(self, c: int, heads: int, levels: int, points: int = POINTS) -> None:  # noqa: D107
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.sampling_offsets = nn.Linear(c, heads * levels * points * 2)
        self.attention_weights = nn.Linear(c, heads * levels * points)
        self.value_proj = nn.Linear(c, c)
        self.output_proj = nn.Linear(c, c)

    def forward(self, query: torch.Tensor, value: torch.Tensor, ref: torch.Tensor, shapes: Shapes) -> torch.Tensor:
        """`query` `[B, Q, C]`, `value` `[B, S, C]` (levels flattened), `ref` `[B, Q, 1 or L, 2]` (x, y) in `[0, 1]`."""
        b, q, _ = query.shape
        h, nl, p = self.heads, self.levels, self.points
        v = self.value_proj(value).view(b, value.shape[1], h, -1)
        off = self.sampling_offsets(query).view(b, q, h, nl, p, 2).float()
        w = self.attention_weights(query).view(b, q, h, nl * p).float().softmax(-1).view(b, q, h, nl, p)
        norm = torch.tensor([(wl, hl) for hl, wl in shapes], dtype=torch.float32, device=off.device)
        loc = ref.float()[:, :, None, :, None, :] + off / norm[:, None, :]
        return self.output_proj(msda_sample(v, shapes, loc, w))


class MHA(nn.Module):
    """`nn.MultiheadAttention` (batch first, eval) with its parameter names, as scaled-dot-product attention."""

    def __init__(self, c: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * c, c))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * c))
        self.out_proj = nn.Linear(c, c)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`[B, S, C]` queries, keys, values; `mask` `[B, Sk]` True = ignore (`key_padding_mask`)."""
        c = q.shape[-1]
        w, b = self.in_proj_weight, self.in_proj_bias
        if q is k and k is v:
            q, k, v = F.linear(q, w, b).chunk(3, -1)
        elif q is k:
            q, k = F.linear(q, w[: 2 * c], b[: 2 * c]).chunk(2, -1)
            v = F.linear(v, w[2 * c :], b[2 * c :])
        elif k is v:
            q = F.linear(q, w[:c], b[:c])
            k, v = F.linear(k, w[c:], b[c:]).chunk(2, -1)
        else:
            q, k, v = (F.linear(t, w[i * c : (i + 1) * c], b[i * c : (i + 1) * c]) for i, t in enumerate((q, k, v)))
        q, k, v = (t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in (q, k, v))
        attn = None if mask is None else ~mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn)
        return self.out_proj(out.transpose(1, 2).flatten(2))


class MHABlock(nn.Module):
    """`MHA_Block`: post-norm attention + ReLU MLP (4x)."""

    def __init__(self, c: int = D, heads: int = HEADS) -> None:  # noqa: D107
        super().__init__()
        self.mha = MHA(c, heads)
        self.linear1, self.linear2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.norm1, self.norm2 = nn.LayerNorm(c), nn.LayerNorm(c)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Upstream forward (no key row is fully masked here: the query's own slot never is)."""
        q = self.norm1(q + self.mha(q, k, v, mask))
        return self.norm2(q + self.linear2(F.relu(self.linear1(q))))


class DMSMHABlock(nn.Module):
    """`DMSMHA_Block`: deformable attention over the 4 feature levels + ReLU MLP (4x), post-norm."""

    def __init__(self, c: int = D, heads: int = HEADS, levels: int = 4) -> None:  # noqa: D107
        super().__init__()
        self.mha = MSDA(c, heads, levels)
        self.linear1, self.linear2 = nn.Linear(c, 4 * c), nn.Linear(4 * c, c)
        self.norm1, self.norm2 = nn.LayerNorm(c), nn.LayerNorm(c)

    def forward(self, q: torch.Tensor, value: torch.Tensor, ref: torch.Tensor, shapes: Shapes) -> torch.Tensor:
        """`q + (mmcv output + identity)`: the query enters the first residual twice, as upstream."""
        q = self.norm1(q + q + self.mha(q, value, ref, shapes))
        return self.norm2(q + self.linear2(F.relu(self.linear1(q))))


def _cbr(cin: int, cout: int, stride: int) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class SpatialPriorModule(nn.Module):
    """The ViT-Adapter's CNN: features at 1/4, 1/8, 1/16, 1/32."""

    def __init__(self, c: int, inplanes: int = 64) -> None:  # noqa: D107
        super().__init__()
        i = inplanes
        self.stem = nn.Sequential(*_cbr(3, i, 2), *_cbr(i, i, 1), *_cbr(i, i, 1), nn.MaxPool2d(3, 2, 1))
        self.conv2 = nn.Sequential(*_cbr(i, 2 * i, 2))
        self.conv3 = nn.Sequential(*_cbr(2 * i, 4 * i, 2))
        self.conv4 = nn.Sequential(*_cbr(4 * i, 4 * i, 2))
        self.fc1, self.fc2 = nn.Conv2d(i, c, 1), nn.Conv2d(2 * i, c, 1)
        self.fc3, self.fc4 = nn.Conv2d(4 * i, c, 1), nn.Conv2d(4 * i, c, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`c1` map `[B, C, H/4, W/4]`, `c2..c4` tokens `[B, n, C]`."""
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return (
            self.fc1(c1),
            self.fc2(c2).flatten(2).transpose(1, 2),
            self.fc3(c3).flatten(2).transpose(1, 2),
            self.fc4(c4).flatten(2).transpose(1, 2),
        )


class _DWConv(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(c, c, 3, 1, 1, groups=c)


class ConvFFN(nn.Module):
    """fc1, the 3 x 3 depthwise conv on each of the 3 adapter levels, GELU, fc2."""

    def __init__(self, c: int, hidden: int) -> None:  # noqa: D107
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, hidden), nn.Linear(hidden, c)
        self.dwconv = _DWConv(hidden)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """`x` `[B, 4hw + hw + hw/4, C]`: the adapter levels at `(2h, 2w)`, `(h, w)`, `(h/2, w/2)`."""
        x = self.fc1(x)
        b, _, c = x.shape
        out = []
        start = 0
        for lh, lw in ((2 * h, 2 * w), (h, w), (h // 2, w // 2)):
            m = x[:, start : start + lh * lw].transpose(1, 2).reshape(b, c, lh, lw)
            start += lh * lw
            out.append(self.dwconv.dwconv(m).flatten(2).transpose(1, 2))
        return self.fc2(F.gelu(torch.cat(out, 1)))


class Injector(nn.Module):
    """Adapter features -> ViT patch tokens (deformable attention over the 3 adapter levels, scaled by `gamma`)."""

    def __init__(self, c: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.query_norm, self.feat_norm = nn.LayerNorm(c, eps=ADAPTER_EPS), nn.LayerNorm(c, eps=ADAPTER_EPS)
        self.attn = MSDA(c, heads, 3)
        self.gamma = nn.Parameter(torch.empty(c))

    def forward(self, query: torch.Tensor, ref: torch.Tensor, feat: torch.Tensor, shapes: Shapes) -> torch.Tensor:
        """`query + gamma * (attn + normed query)` (mmcv's identity is the normed query)."""
        qn = self.query_norm(query)
        return query + self.gamma * (self.attn(qn, self.feat_norm(feat), ref, shapes) + qn)


class Extractor(nn.Module):
    """ViT patch tokens -> adapter features (deformable attention over the ViT grid), then the ConvFFN."""

    def __init__(self, c: int, heads: int) -> None:  # noqa: D107
        super().__init__()
        self.query_norm, self.feat_norm = nn.LayerNorm(c, eps=ADAPTER_EPS), nn.LayerNorm(c, eps=ADAPTER_EPS)
        self.attn = MSDA(c, heads, 1)
        self.ffn = ConvFFN(c, c // 4)
        self.ffn_norm = nn.LayerNorm(c, eps=ADAPTER_EPS)

    def forward(  # noqa: PLR0913, PLR0917 - upstream's arguments
        self, query: torch.Tensor, ref: torch.Tensor, feat: torch.Tensor, shapes: Shapes, h: int, w: int
    ) -> torch.Tensor:
        """`query + attn + normed query`, then the ConvFFN residual."""
        qn = self.query_norm(query)
        query = query + self.attn(qn, self.feat_norm(feat), ref, shapes) + qn
        return query + self.ffn(self.ffn_norm(query), h, w)


class Interaction(nn.Module):
    """`InteractionBlockWithCls` (the ViT blocks run from the adapter)."""

    def __init__(self, c: int, heads: int, *, extra: bool) -> None:  # noqa: D107
        super().__init__()
        self.injector = Injector(c, heads)
        self.extractor = Extractor(c, heads)
        self.extra_extractors = nn.ModuleList(Extractor(c, heads) for _ in range(2)) if extra else None


def reference_points(shapes: Shapes, device: torch.device) -> torch.Tensor:
    """Cell centres `[1, sum(h w), 1, 2]` (x, y) in `[0, 1]` of the levels (upstream `get_reference_points`)."""
    refs = []
    for h, w in shapes:
        ys, xs = torch.meshgrid(
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device) / h,
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device) / w,
            indexing="ij",
        )
        refs.append(torch.stack((xs.reshape(-1), ys.reshape(-1)), -1))
    return torch.cat(refs)[None, :, None]


class ViTAdapter(nn.Module):
    """ViT-Adapter around the frozen DINOv3 ViT: 4 feature maps `[B, C, H/4 .. H/32, ...]` of a normalised image."""

    def __init__(self, arch: str, vit_scale: float) -> None:  # noqa: D107
        super().__init__()
        cfg = CONFIGS[arch]
        c = cfg.hidden_size
        self.vit_scale = vit_scale
        self.dinov3 = DINOv3ViT(cfg)
        self.level_embed = nn.Parameter(torch.empty(3, c))
        self.spm = SpatialPriorModule(c)
        n = len(INTERACTIONS)
        self.interactions = nn.ModuleList(Interaction(c, ADAPTER_HEADS, extra=i == n - 1) for i in range(n))
        self.up = nn.ConvTranspose2d(c, c, 2, 2)
        self.norm1, self.norm2 = nn.BatchNorm2d(c), nn.BatchNorm2d(c)
        self.norm3, self.norm4 = nn.BatchNorm2d(c), nn.BatchNorm2d(c)

    def vit_block(self, i: int, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ViT block `i` (HF `DINOv3ViTLayer`); the residual stream keeps the dtype of `x`."""
        vit = self.dinov3
        blk = vit.blocks()[i]
        x = x + vit.attention(blk, blk.norm1(x), rope) * blk.layer_scale1.lambda1
        return x + blk.mlp(blk.norm2(x)) * blk.layer_scale2.lambda1

    def vit_rope(self, h: int, w: int, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The padded RoPE tables of an `h x w` patch grid (float32: applied in place to the q / k)."""
        return self.dinov3.rope(h, w, torch.float32, x.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalised `[B, 3, H, W]` (H, W multiples of 32) -> BatchNormed maps at 1/4, 1/8, 1/16, 1/32."""
        b, _, hh, ww = x.shape
        hv, wv = int(hh * self.vit_scale) // PATCH * PATCH, int(ww * self.vit_scale) // PATCH * PATCH
        x_vit = F.interpolate(x, (hv, wv), mode="bilinear", align_corners=False)
        gh, gw = hv // PATCH, wv // PATCH
        ha, wa = hh // 16, ww // 16
        adapter = [(hh // 8, ww // 8), (ha, wa), (hh // 32, ww // 32)]
        ref1, ref2 = reference_points([(gh, gw)], x.device), reference_points(adapter, x.device)
        c1, c2, c3, c4 = self.spm(x)
        le = self.level_embed
        c = torch.cat([c2 + le[0], c3 + le[1], c4 + le[2]], 1)
        t = self.dinov3.tokens(x_vit)
        rope = self.vit_rope(gh, gw, t)
        pre = 1 + REGISTERS
        prefix, xp = t[:, :pre], t[:, pre:]
        for inter, (first, last) in zip(self.interactions, INTERACTIONS, strict=True):
            it = cast("Interaction", inter)
            xp = it.injector(xp, ref1, c, adapter)
            t = torch.cat([prefix.to(xp.dtype), xp], 1)
            for i in range(first, last + 1):
                t = self.vit_block(i, t, rope)
            prefix, xp = t[:, :pre], t[:, pre:]
            c = it.extractor(c, ref2, xp, [(gh, gw)], ha, wa)
            for e in it.extra_extractors or ():
                c = e(c, ref2, xp, [(gh, gw)], ha, wa)
        n2, n3 = 4 * ha * wa, ha * wa
        ch = c.shape[-1]
        m2 = c[:, :n2].transpose(1, 2).reshape(b, ch, 2 * ha, 2 * wa)
        m3 = c[:, n2 : n2 + n3].transpose(1, 2).reshape(b, ch, ha, wa)
        m4 = c[:, n2 + n3 :].transpose(1, 2).reshape(b, ch, ha // 2, wa // 2)
        m1 = self.up(m2) + c1
        return self.norm1(m1), self.norm2(m2), self.norm3(m3), self.norm4(m4)


class Backbone(nn.Module):
    """ImageNet normalisation, the adapter, per-level projections to `D` plus the learned position embedding."""

    def __init__(self, arch: str, vit_scale: float, size: tuple[int, int]) -> None:  # noqa: D107
        super().__init__()
        self.vit_encoder = ViTAdapter(arch, vit_scale)
        c = CONFIGS[arch].hidden_size
        self.projections = nn.ModuleList(nn.Sequential(nn.Linear(c, D)) for _ in range(4))
        self.pos_embedding = nn.Parameter(torch.empty(1, D, size[0] // STRIDE, size[1] // STRIDE))
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, video: torch.Tensor) -> list[torch.Tensor]:
        """`[B, 3, H, W]` in `[0, 1]` -> 4 maps `[B, D, H/4 .. H/32, ...]`."""
        maps = self.vit_encoder((video - self.mean) / self.std)
        out = []
        for i, (m, proj) in enumerate(zip(maps, self.projections, strict=True)):
            pos = self.pos_embedding
            if i:
                pos = F.interpolate(pos, scale_factor=0.5**i, mode="bilinear", align_corners=False)
            out.append(proj(m.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + pos)
        return out


def _lateral() -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(D, D, 1), nn.GroupNorm(32, D), nn.ReLU())


class SimpleFPN(nn.Module):
    """Top-down sum of the 4 levels into the 1/4 map."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.lateral32, self.lateral16, self.lateral8 = _lateral(), _lateral(), _lateral()
        self.output = nn.Sequential(nn.Conv2d(D, D, 3, padding=1), nn.GroupNorm(32, D), nn.ReLU(), nn.Conv2d(D, D, 3, padding=1))

    def forward(self, f4: torch.Tensor, f8: torch.Tensor, f16: torch.Tensor, f32: torch.Tensor) -> torch.Tensor:
        """Maps -> fused 1/4 map."""
        lat = self.lateral32(f32)
        lat = self.lateral16(f16) + F.interpolate(lat, size=f16.shape[-2:], mode="nearest")
        lat = self.lateral8(f8) + F.interpolate(lat, size=f8.shape[-2:], mode="nearest")
        return self.output(f4 + F.interpolate(lat, size=f4.shape[-2:], mode="nearest"))


@dataclass
class Frame:
    """One frame's features: tokens `[1, P_l, D]` per level, their concatenation, the fused tokens."""

    levels: list[torch.Tensor]
    value: torch.Tensor  # the 4 levels concatenated (deformable attention value)
    unit: list[torch.Tensor]  # L2-normalised levels `[P_l, D]` (correlation)
    fused: torch.Tensor


class RerankModule(nn.Module):
    """`Rerank_Module`: deformable decoding of the top-K correlation peaks, fused into the query."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.local_decoder = nn.ModuleList(DMSMHABlock() for _ in range(LAYERS))
        self.fusion = MHABlock()
        self.fusion_layer = nn.Linear(2 * D, D)
        self.final_projection_layer = nn.Linear(2 * D, D)
        self.certainty_layer = nn.Linear(D, 1)  # training only
        self.score_layer = nn.Linear(D, 1)  # training only


class PredictionHead(nn.Module):
    """`Prediction_Head`: deformable offset refinement around the patch and the visibility logit."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.offset_transformer = nn.ModuleList(DMSMHABlock() for _ in range(LAYERS))
        self.offset_layer = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 2), nn.Tanh())
        self.vis_layer = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))
        self.unc_layer = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1))  # training only


def _layers(n: int) -> nn.ModuleList:
    return nn.ModuleList(MHABlock() for _ in range(n))


class TrackOn2(nn.Module):
    """`Track_On2` (online): `features` of a frame, then `step` of the active queries with their memory."""

    def __init__(self, arch: str, vit_scale: float, size: tuple[int, int], memory: int) -> None:  # noqa: D107
        super().__init__()
        self.size = size
        self.hf, self.wf = size[0] // STRIDE, size[1] // STRIDE
        self.shapes: Shapes = [(self.hf >> i, self.wf >> i) for i in range(4)]
        self.memory = memory
        self.backbone = Backbone(arch, vit_scale, size)
        self.fpn = SimpleFPN()
        self.reranking_head = RerankModule()
        self.prediction_head = PredictionHead()
        self.feature_attention, self.query_attention, self.memory_attention = _layers(LAYERS), _layers(LAYERS), _layers(LAYERS)
        self.projection1, self.projection2 = nn.Linear(D, D), nn.Linear(D, D)
        self.ms_corr_proj = nn.Conv2d(4, 1, 1)
        self.t_embedding = nn.Parameter(torch.empty(1, memory + 1, D))

    @staticmethod
    def extend_memory(t_embedding: torch.Tensor, memory: int) -> torch.Tensor:
        """Upstream `memory_extension`: the past slots interpolated linearly (align_corners) to `memory`."""
        past = F.interpolate(t_embedding[:, :-1].transpose(1, 2), size=memory, mode="linear", align_corners=True)
        return torch.cat([past.transpose(1, 2), t_embedding[:, -1:]], 1)

    def features(self, video: torch.Tensor) -> list[Frame]:
        """Per-frame features of `[B, 3, H, W]` in `[0, 1]` (resized to the network size, bilinear)."""
        if tuple(video.shape[-2:]) != self.size:
            video = F.interpolate(video, self.size, mode="bilinear", align_corners=False)
        maps = self.backbone(video)
        fused = self.fpn(*maps).flatten(2).transpose(1, 2)
        levels = [m.flatten(2).transpose(1, 2) for m in maps]
        out = []
        for b in range(video.shape[0]):
            lv = [lvl[b : b + 1] for lvl in levels]
            unit = [F.normalize(lvl[0].float(), dim=-1) for lvl in lv]
            out.append(Frame(lv, torch.cat(lv, 1), unit, fused[b : b + 1]))
        return out

    def sample(self, frame: Frame, xy: torch.Tensor) -> torch.Tensor:
        """Query features `[N, D]` at `xy` `[N, 2]` in `[0, 1]` (edge-aligned) of the fused map (bilinear, border)."""
        f = frame.fused.transpose(1, 2).reshape(1, D, self.hf, self.wf)
        grid = (xy * 2 - 1)[None, :, None].to(f.dtype)
        return F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T

    def correlation(self, q: torch.Tensor, frame: Frame) -> torch.Tensor:
        """`multiscale_correlation`: cosine similarity per level, upsampled to 1/4, mixed by a 1 x 1 conv -> `[N, P]`."""
        n = q.shape[0]
        qn = F.normalize(q.float(), dim=-1)
        cs = []
        for i, (u, (h, w)) in enumerate(zip(frame.unit, self.shapes, strict=True)):
            c = (qn @ u.T).view(n, 1, h, w)
            cs.append(F.interpolate(c, size=(self.hf, self.wf), mode="bilinear", align_corners=False) if i else c)
        conv = self.ms_corr_proj
        return F.conv2d(torch.cat(cs, 1), conv.weight.float(), cast("torch.Tensor", conv.bias).float()).view(n, -1)

    def coords(self, idx: torch.Tensor) -> torch.Tensor:
        """`indices_to_coords`: cell centres `(x, y)` in network pixels of flat 1/4 indices."""
        return torch.stack(((idx % self.wf).float() + 0.5, (idx // self.wf).float() + 0.5), -1) * STRIDE

    def _norm(self, p: torch.Tensor) -> torch.Tensor:
        return (p / p.new_tensor((self.size[1], self.size[0]))).clamp(0, 1)

    def rerank(self, q: torch.Tensor, frame: Frame, c1: torch.Tensor) -> torch.Tensor:
        """`Rerank_Module.forward` (inference outputs): the re-ranked query `[N, D]`."""
        n = q.shape[0]
        rh = self.reranking_head
        ref = self._norm(self.coords(c1.topk(TOPK, -1).indices)).view(1, n * TOPK, 1, 2)
        qs = q[:, None].expand(n, TOPK, D)
        qk = qs.reshape(1, n * TOPK, D)
        for blk in rh.local_decoder:
            qk = blk(qk, frame.value, ref, self.shapes)
        qk = rh.fusion_layer(torch.cat([qk.view(n, TOPK, D), qs.to(qk.dtype)], -1))
        fused = rh.fusion(q[:, None], qk, qk)
        return rh.final_projection_layer(torch.cat([fused, q[:, None].to(fused.dtype)], -1)).view(n, D)

    def head(self, q: torch.Tensor, frame: Frame, p_patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`Prediction_Head.forward` (inference outputs): last-layer offset `[N, 2]` (pixels), visibility logit `[N]`."""
        ph = self.prediction_head
        ref = self._norm(p_patch)[None, :, None]
        qo = q[None]
        for blk in ph.offset_transformer:
            qo = blk(qo, frame.value, ref, self.shapes)
        return ph.offset_layer(qo)[0].float() * STRIDE, ph.vis_layer(q)[:, 0].float()

    def step(
        self, q_init: torch.Tensor, mask: torch.Tensor, memory: torch.Tensor, frame: Frame
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`track_frame`: positions `[N, 2]` (network pixels, edge-aligned), visibility logits `[N]`, the new memory
        entry `[N, D]` of the `N` active queries; `mask` `[N, M]` True = empty memory slot, `memory` `[N, M, D]`."""  # noqa: D205
        n = q_init.shape[0]
        q = q_init[None]
        m = torch.cat([mask, mask.new_zeros(n, 1)], 1)
        qkv = memory
        temb = self.t_embedding
        for fa, qa, ma in zip(self.feature_attention, self.query_attention, self.memory_attention, strict=True):
            q = fa(q, frame.fused, frame.fused)
            q = qa(q, q, q)
            qkv = torch.cat([qkv[:, :-1] if qkv.shape[1] > memory.shape[1] else qkv, q.view(n, 1, D).to(qkv.dtype)], 1)
            x = qkv + temb
            qkv = ma(x, x, qkv, m)
            q = qkv[:, -1][None]
        q = self.projection1(q[0])
        q = self.projection2(self.rerank(q, frame, self.correlation(q, frame)))
        p_patch = self.coords(self.correlation(q, frame).argmax(-1))
        off, vis = self.head(q, frame, p_patch)
        return p_patch + off, vis, q
