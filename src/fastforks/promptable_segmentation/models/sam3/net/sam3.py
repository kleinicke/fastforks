"""SAM 3 image detector (text and exemplar-box prompts), ported from Hugging Face transformers `models/sam3`.

Inference only, plain PyTorch. Module and parameter names match the checkpoint's `detector_model.*` keys (the tracker
is left out: see `UNUSED`). Differences from the transformers `modeling_sam3.py` forward, all exact for one image and
one prompt:

- The text encoder runs on the real tokens only (`[sot, ..., eot]`, no padding): the CLIP encoder is causal, so padding
  never reaches them, and every consumer masks the padding out anyway. So no attention mask is needed by default;
  `detect(..., text_valid=)` takes padded tokens (max's CUDA graphs per token bucket) and masks them out of the
  prompt cross-attentions and the scoring mean, as transformers does with its attention mask.
- The geometry encoder always runs, as in upstream `Sam3Image._encode_prompt` (transformers skips it without boxes):
  without boxes it encodes its CLS token only, which cross-attends to the image. Its tokens are the boxes, then the
  CLS token; `Exemplars.valid` masks padding boxes (max's box buckets) as `text_valid` masks padding tokens.
- ROI align (torchvision `roi_align`, output 7 x 7, adaptive sampling, `aligned=False`) is separable: `roi_weights`
  folds it into one `[7, h]` and one `[7, w]` weight matrix per box (computed on the CPU, as torchvision computes
  the sample points in float32), so pooling is two small matmuls with fixed shapes (CUDA-graph safe).
- Only the three FPN levels the detector reads are built (the 0.5x level is unused); the neck's position encoding is
  computed for the 72 x 72 level only (the only one read).
- The decoder returns the last layer only (the others are training outputs).

Copyright 2025 the HuggingFace Team and Meta Platforms, Inc. (Apache-2.0); port for fastforks.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

GRID = 72  # fixed: SAM 3 patch grid (1008 / 14); the processor always resizes to 1008 x 1008
HEAD = 64  # fixed: ViT head dim (1024 / 16 heads)
DIM = 256  # fixed: detector width
DETR_HEADS = 8  # fixed: heads of every 256-wide attention
UNUSED = (  # checkpoint keys (after `detector_model.`) this port does not load
    "vision_encoder.neck.fpn_layers.3.",  # the 0.5x FPN level (dropped by Sam3Model.forward)
    "text_encoder.text_projection.",  # CLIP's own projection (SAM 3 projects the hidden states)
    "mask_decoder.semantic_projection.",  # semantic segmentation head (not an output here)
    "mask_decoder.pixel_decoder.conv_layers.2.",  # third upsampling stage: only two run for three levels
    "mask_decoder.pixel_decoder.norms.2.",
)
ROI = 7  # fixed: exemplar-box ROI-align output side (transformers `Sam3GeometryEncoderConfig.roi_size`)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """transformers `inverse_sigmoid`."""
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    """Boxes `(cx, cy, w, h)` -> `(x0, y0, x1, y1)`."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _sine(x: torch.Tensor, features: int = DIM // 2, temperature: int = 10000) -> torch.Tensor:
    """`[..., ] -> [..., features]` interleaved sin / cos (transformers `Sam3SinePositionEmbedding`)."""
    dim_t = torch.arange(features, dtype=torch.int64, device=x.device).to(x.dtype)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / features)
    p = x[..., None] / dim_t
    return torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2)


def grid_position_encoding(h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
    """Normalised sine encoding of an `h x w` grid, `[1, 256, h, w]` float32 (the neck's `position_encoding`)."""
    eps, scale = 1e-6, 2 * math.pi
    y = torch.arange(1, h + 1, dtype=torch.float32, device=device)[:, None].expand(h, w)
    x = torch.arange(1, w + 1, dtype=torch.float32, device=device)[None, :].expand(h, w)
    y = y / (y[-1:, :] + eps) * scale
    x = x / (x[:, -1:] + eps) * scale
    return torch.cat((_sine(y), _sine(x)), dim=-1).permute(2, 0, 1)[None]


def encode_boxes(boxes: torch.Tensor) -> torch.Tensor:
    """`[B, Q, 4]` `(cx, cy, w, h)` -> `[B, Q, 512]` (`encode_boxes`: order y, x, w, h; scale 2 pi, unnormalised)."""
    s = boxes * (2 * math.pi)
    return torch.cat((_sine(s[..., 1]), _sine(s[..., 0]), _sine(s[..., 2]), _sine(s[..., 3])), dim=-1)


class Exemplars(NamedTuple):
    """Exemplar boxes of a prompt (transformers `input_boxes`, `input_boxes_labels`), image batch first.

    Attributes:
        boxes: `[B, N, 4]` normalised `(cx, cy, w, h)` float32.
        labels: `[B, N]` int64: 1 positive, 0 negative.
        pool_y: `[B, N, 7, h]` ROI-align row weights (`roi_weights`).
        pool_x: `[B, N, 7, w]` ROI-align column weights.
        valid: `[B, N]` float32, 0 for a padding box; `None`: all real.
    """

    boxes: torch.Tensor
    labels: torch.Tensor
    pool_y: torch.Tensor
    pool_x: torch.Tensor
    valid: torch.Tensor | None = None


def _roi_axis(start: torch.Tensor, end: torch.Tensor, size: int, out: int) -> torch.Tensor:
    """One axis of torchvision `roi_align` (`aligned=False`, adaptive sampling): `[out, size]` weights, mean folded in.

    The sample points are those of torchvision's kernel, in float32 and in its order of operations; bilinear
    weights go to the two neighbouring rows, points outside `[-1, size]` count as zero.
    """
    length = torch.clamp(end - start, min=1.0)
    step = length / out
    grid = max(math.ceil(float(step)), 1)
    p = torch.arange(out, dtype=torch.float32)[:, None]
    i = torch.arange(grid, dtype=torch.float32)[None]
    pos = start + p * step + (i + 0.5) * step / grid  # [out, grid]
    inside = (pos >= -1.0) & (pos <= size)
    y = pos.clamp(min=0.0)
    low = y.floor()
    edge = low >= size - 1
    low = torch.where(edge, torch.full_like(low, size - 1), low)
    y = torch.where(edge, low, y)
    hi_w = (y - low) * inside
    lo_w = (1.0 - (y - low)) * inside
    high = torch.where(edge, low, low + 1)
    w = torch.zeros(out, size)
    w.scatter_add_(1, low.long(), lo_w)
    w.scatter_add_(1, high.long(), hi_w)
    return w / grid


def roi_weights(boxes: torch.Tensor, h: int, w: int, out: int = ROI) -> tuple[torch.Tensor, torch.Tensor]:
    """ROI-align weights of `[..., 4]` normalised `(cx, cy, w, h)` boxes on an `h x w` map (float32, on the CPU).

    `pool[c, p, q] = sum_yx rows[p, y] * feature[c, y, x] * cols[q, x]` is torchvision
    `roi_align(feature, boxes * [w, h, w, h] as xyxy, out)` (transformers `Sam3GeometryEncoder._encode_boxes`).

    Returns:
        Row weights `[..., out, h]` and column weights `[..., out, w]`.
    """
    xyxy = (cxcywh_to_xyxy(boxes.float().cpu()) * torch.tensor([w, h, w, h], dtype=torch.float32)).reshape(-1, 4)
    rows = [_roi_axis(b[1], b[3], h, out) for b in xyxy]
    cols = [_roi_axis(b[0], b[2], w, out) for b in xyxy]
    lead = boxes.shape[:-1]
    if not rows:
        return torch.zeros((*lead, out, h)), torch.zeros((*lead, out, w))
    return torch.stack(rows).reshape(*lead, out, h), torch.stack(cols).reshape(*lead, out, w)


def mask_bias(valid: torch.Tensor) -> torch.Tensor:
    """`[B, T]` validity (0 = padding) -> additive attention bias `[B, 1, 1, T]` (-inf on padding)."""
    return torch.zeros_like(valid).masked_fill(valid == 0, -math.inf)[:, None, None]


def rope_tables(side: int, scale: float, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Axial RoPE `cos`, `sin` `[side * side, 64]` float32 for a `side x side` grid, positions times `scale`.

    transformers `Sam3ViTLayer.precompute_positions` + `Sam3ViTRotaryEmbedding` (theta 10000, pairwise layout).
    """
    hh, ww = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    pos = torch.stack([ww.flatten(), hh.flatten()], dim=-1).to(device, torch.float32) * scale
    inv = 1.0 / (10000 ** (torch.arange(0, HEAD // 2, 2, dtype=torch.float32, device=device) / (HEAD // 2)))
    f = pos[..., None] * inv  # [N, 2, 16]
    f = torch.cat([f[:, 0], f[:, 1]], dim=-1).repeat_interleave(2, dim=-1)
    return f.cos(), f.sin()


def rope_pairs(cos: torch.Tensor, sin: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """`rope_tables` -> `(cos, cos)`, `(-sin, sin)` `[N, 1, 1, 32, 2]` in `dtype` for `RopeAttention` (pairwise layout)."""
    c, s = cos[:, 0::2], sin[:, 0::2]  # the tables repeat each angle twice
    return torch.stack([c, c], -1).to(dtype)[:, None, None], torch.stack([-s, s], -1).to(dtype)[:, None, None]


# --- vision ---------------------------------------------------------------------------------------------------------


class Mlp(nn.Module):
    """`fc2(act(fc1(x)))`."""

    def __init__(self, dim: int, hidden: int, act: str) -> None:
        """`act`: `"gelu"` (exact) or `"relu"`."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = F.gelu if act == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply."""
        return self.fc2(self.act(self.fc1(x)))


class RopeAttention(nn.Module):
    """ViT self-attention with axial RoPE, q / k / v as one packed projection after `pack()` (checkpoint: three)."""

    def __init__(self, dim: int, heads: int) -> None:
        """Projections with bias."""
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.o_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)

    @torch.no_grad()
    def pack(self) -> None:
        """Replace `q_proj`, `k_proj`, `v_proj` by one `qkv` projection (one GEMM)."""
        q, k, v = self.q_proj, self.k_proj, self.v_proj
        self.qkv = nn.Linear(q.in_features, 3 * q.out_features, device=q.weight.device, dtype=q.weight.dtype)
        self.qkv.weight.copy_(torch.cat([q.weight, k.weight, v.weight]))
        self.qkv.bias.copy_(torch.cat([q.bias, k.bias, v.bias]))
        self.qkv.train(q.training).requires_grad_(q.weight.requires_grad)  # a new module: keep the parent's state
        del self.q_proj, self.k_proj, self.v_proj

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """`[B, N, C]` -> `[B, N, C]`; `cos`, `sin`: `rope_pairs` (rotation in the compute dtype)."""
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, -1, 2)
        qk = qkv[:, :, :2]
        qk = torch.addcmul(qk * cos, qk.flip(-1), sin).flatten(-2)  # (x0, x1) -> (x0 c - x1 s, x1 c + x0 s)
        v = qkv[:, :, 2].flatten(-2)
        o = F.scaled_dot_product_attention(qk[:, :, 0].transpose(1, 2), qk[:, :, 1].transpose(1, 2), v.transpose(1, 2))
        return self.o_proj(o.transpose(1, 2).reshape(b, n, c))


class ViTLayer(nn.Module):
    """Pre-LN block, windowed (`window > 0`, windows of `window x window` tokens) or global attention."""

    def __init__(self, dim: int, heads: int, mlp: int, window: int) -> None:
        """LayerNorm eps 1e-6, exact GELU."""
        super().__init__()
        self.window = window
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = RopeAttention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp, "gelu")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Tokens `[B, N, C]` (windowed layers: one window per batch entry, `ViT.forward` partitions)."""
        x = x + self.attention(self.layer_norm1(x), cos, sin)
        return x + self.mlp(self.layer_norm2(x))


class PatchEmbeddings(nn.Module):
    """14 x 14 patches, no bias."""

    def __init__(self, dim: int, patch: int) -> None:
        """Conv `3 -> dim`."""
        super().__init__()
        self.projection = nn.Conv2d(3, dim, kernel_size=patch, stride=patch, bias=False)


class Embeddings(nn.Module):
    """Patch embedding + a 24 x 24 position embedding tiled over the grid (no cls token)."""

    def __init__(self, dim: int, patch: int, pretrain_tokens: int) -> None:
        """`pretrain_tokens`: side of the stored position grid (336 / 14)."""
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(dim, patch)
        self.position_embeddings = nn.Parameter(torch.zeros(1, pretrain_tokens * pretrain_tokens, dim))

    def tiled(self, h: int, w: int) -> torch.Tensor:
        """Position embedding `[1, h, w, C]` (transformers `_tile_position_embeddings`)."""
        p = self.position_embeddings
        s = math.isqrt(p.shape[1])
        g = p.reshape(1, s, s, -1)
        return g.tile(1, h // s + 1, w // s + 1, 1)[:, :h, :w]


class ViT(nn.Module):
    """SAM 3 ViT-L-like trunk (`Sam3ViTModel`): 32 layers, windows of 24, global attention every 8th layer."""

    def __init__(  # noqa: PLR0913 - the config of the trunk
        self, dim: int, depth: int, heads: int, mlp: int, patch: int, window: int, globals_: tuple[int, ...]
    ) -> None:
        """Untrained trunk."""
        super().__init__()
        self.patch = patch
        self.embeddings = Embeddings(dim, patch, window)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.layers = nn.ModuleList(
            ViTLayer(dim, heads, mlp, 0 if i in globals_ else window) for i in range(depth)
        )

    def pack(self) -> None:
        """`RopeAttention.pack` in every layer."""
        for m in self.modules():
            if isinstance(m, RopeAttention):
                m.pack()

    def forward(self, pixels: torch.Tensor, ropes: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """`[B, 3, H, W]` normalised -> `[B, H/14, W/14, C]` (`ropes`: window size or 0 -> `rope_pairs`)."""
        proj = self.embeddings.patch_embeddings.projection
        x = proj(pixels.to(proj.weight.dtype)).permute(0, 2, 3, 1)
        x = self.layer_norm(x + self.embeddings.tiled(x.shape[1], x.shape[2]))
        b, h, w, c = x.shape
        s = next(layer.window for layer in self.layers if layer.window)
        k, j = h // s, w // s

        def part(z: torch.Tensor) -> torch.Tensor:  # [B, H * W, C] -> [B k j, s s, C]
            return z.view(b, k, s, j, s, c).permute(0, 1, 3, 2, 4, 5).reshape(b * k * j, s * s, c)

        def unpart(z: torch.Tensor) -> torch.Tensor:  # [B k j, s s, C] -> [B, H * W, C]
            return z.view(b, k, j, s, s, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h * w, c)

        x, windowed = x.reshape(b, h * w, c), False  # the residual stays in window order between global layers
        for layer in self.layers:
            if bool(layer.window) != windowed:
                x, windowed = (part(x) if layer.window else unpart(x)), bool(layer.window)
            x = layer(x, *ropes[layer.window])
        return (unpart(x) if windowed else x).view(b, h, w, c)


class FpnLayer(nn.Module):
    """One neck level: upsampling (4x: two transposed convs with GELU; 2x: one; 1x: none), 1 x 1, 3 x 3."""

    def __init__(self, dim: int, out: int, scale: int) -> None:
        """`scale`: 4, 2 or 1."""
        super().__init__()
        layers: list[nn.Module] = []
        c = dim
        for i in range({4: 2, 2: 1, 1: 0}[scale]):
            if i:
                layers.append(nn.GELU())
            layers.append(nn.ConvTranspose2d(c, c // 2, kernel_size=2, stride=2))
            c //= 2
        self.scale_layers = nn.ModuleList(layers)
        self.proj1 = nn.Conv2d(c, out, kernel_size=1)
        self.proj2 = nn.Conv2d(out, out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, h, w]` -> `[B, out, s h, s w]`."""
        for layer in self.scale_layers:
            x = layer(x)
        return self.proj2(self.proj1(x))


class Neck(nn.Module):
    """FPN levels 4x, 2x and 1x of the trunk output (288², 144², 72² for 1008 input)."""

    def __init__(self, dim: int, out: int) -> None:
        """Three levels."""
        super().__init__()
        self.fpn_layers = nn.ModuleList(FpnLayer(dim, out, s) for s in (4, 2, 1))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Trunk output `[B, C, h, w]` -> the levels fine to coarse."""
        return [layer(x) for layer in self.fpn_layers]


class VisionEncoder(nn.Module):
    """`Sam3VisionModel`: trunk + neck."""

    def __init__(self, backbone: ViT, neck: Neck) -> None:
        """Wrap."""
        super().__init__()
        self.backbone = backbone
        self.neck = neck


# --- text -----------------------------------------------------------------------------------------------------------


class ClipAttention(nn.Module):
    """CLIP causal self-attention."""

    def __init__(self, dim: int, heads: int) -> None:
        """Projections with bias."""
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.out_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, T, C]` -> `[B, T, C]`."""
        b, t, c = x.shape

        def split(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.heads, c // self.heads).transpose(1, 2)

        o = F.scaled_dot_product_attention(
            split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x)), is_causal=True
        )
        return self.out_proj(o.transpose(1, 2).reshape(b, t, c))


class ClipLayer(nn.Module):
    """Pre-LN CLIP block (eps 1e-5, exact GELU)."""

    def __init__(self, dim: int, heads: int, mlp: int) -> None:
        """Untrained block."""
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(dim)
        self.self_attn = ClipAttention(dim, heads)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp, "gelu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply."""
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class ClipEmbeddings(nn.Module):
    """Token + learned position embeddings."""

    def __init__(self, vocab: int, context: int, dim: int) -> None:
        """Untrained tables."""
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, dim)
        self.position_embedding = nn.Embedding(context, dim)


class ClipEncoder(nn.Module):
    """The block stack."""

    def __init__(self, layers: list[ClipLayer]) -> None:
        """Wrap."""
        super().__init__()
        self.layers = nn.ModuleList(layers)


class ClipTextModel(nn.Module):
    """CLIP text transformer (`CLIPTextTransformer`), returning the final-LayerNorm hidden states."""

    def __init__(self, vocab: int, context: int, dim: int, depth: int, heads: int, mlp: int) -> None:  # noqa: PLR0913 - the config
        """Untrained encoder."""
        super().__init__()
        self.embeddings = ClipEmbeddings(vocab, context, dim)
        self.encoder = ClipEncoder([ClipLayer(dim, heads, mlp) for _ in range(depth)])
        self.final_layer_norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """`[B, T]` token ids -> `[B, T, C]`."""
        e = self.embeddings
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = e.token_embedding(ids) + e.position_embedding(pos)[None]
        for layer in self.encoder.layers:
            x = layer(x)
        return self.final_layer_norm(x)


class TextEncoder(nn.Module):
    """`CLIPTextModelWithProjection` without its projection."""

    def __init__(self, text_model: ClipTextModel) -> None:
        """Wrap."""
        super().__init__()
        self.text_model = text_model


# --- detector -------------------------------------------------------------------------------------------------------


class Attention(nn.Module):
    """`Sam3Attention`: multi-head attention with separate q / k / v / o projections and an optional additive bias."""

    def __init__(self, dim: int = DIM, heads: int = DETR_HEADS) -> None:
        """Projections with bias."""
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.v_proj, self.o_proj = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`q [B, Nq, C]`, `k`, `v [B, Nk, C]`, `bias [B, heads, Nq, Nk]` (or broadcastable) -> `[B, Nq, C]`.

        Runs in the dtype of the weights (the output in that of `q`).
        """
        out_dtype, dt = q.dtype, self.q_proj.weight.dtype
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
        bias = None if bias is None else bias.to(dt)
        b, nq, c = q.shape
        nk, d = k.shape[1], c // self.heads
        qh = self.q_proj(q).view(b, nq, self.heads, d).transpose(1, 2)
        kh = self.k_proj(k).view(b, nk, self.heads, d).transpose(1, 2)
        vh = self.v_proj(v).view(b, nk, self.heads, d).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        return self.o_proj(o.transpose(1, 2).reshape(b, nq, c)).to(out_dtype)


class DecoderMLP(nn.Module):
    """`Sam3DecoderMLP`: 2 or 3 linear layers with ReLU between."""

    def __init__(self, dims: tuple[int, ...]) -> None:
        """`dims`: `(in, hidden, out)` (2 layers) or `(in, hidden, hidden, out)` (3)."""
        super().__init__()
        self.layer1 = nn.Linear(dims[0], dims[1])
        self.layer2 = nn.Linear(dims[1], dims[2])
        self.layer3 = nn.Linear(dims[2], dims[3]) if len(dims) == 4 else None  # noqa: PLR2004 - 3 layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply."""
        x = F.relu(self.layer1(x))
        if self.layer3 is None:
            return self.layer2(x)
        return self.layer3(F.relu(self.layer2(x)))


class PromptVisionLayer(nn.Module):
    """Pre-LN block of the geometry encoder (prompt tokens) and the DETR encoder (image tokens).

    Self-attention (DETR encoder: query / key with the image position encoding), cross-attention, ReLU MLP.
    """

    def __init__(self) -> None:
        """Untrained block (LayerNorm eps 1e-5)."""
        super().__init__()
        self.layer_norm1, self.layer_norm2, self.layer_norm3 = nn.LayerNorm(DIM), nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.self_attn, self.cross_attn = Attention(), Attention()
        self.mlp = Mlp(DIM, 2048, "relu")


class GeometryEncoder(nn.Module):
    """`Sam3GeometryEncoder`: exemplar boxes (if any) and a CLS token, attending to the 72 x 72 image level."""

    def __init__(self, depth: int) -> None:
        """Untrained encoder."""
        super().__init__()
        self.label_embed = nn.Embedding(2, DIM)
        self.cls_embed = nn.Embedding(1, DIM)
        self.boxes_direct_project = nn.Linear(4, DIM)
        self.boxes_pool_project = nn.Conv2d(DIM, DIM, ROI)
        self.boxes_pos_enc_project = nn.Linear(DIM + 2, DIM)
        self.vision_layer_norm = nn.LayerNorm(DIM)
        self.final_proj = nn.Linear(DIM, DIM)
        self.prompt_layer_norm = nn.LayerNorm(DIM)
        self.layers = nn.ModuleList(PromptVisionLayer() for _ in range(depth))
        self.output_layer_norm = nn.LayerNorm(DIM)

    def embed_boxes(self, vision: torch.Tensor, ex: Exemplars) -> torch.Tensor:
        """`vision [B, h * w, C]`, the boxes -> box tokens `[B, N, C]` (`_encode_boxes`: direct + ROI + position + label).

        Runs in float32.
        """
        b, n, _ = ex.boxes.shape
        h, w = ex.pool_y.shape[-1], ex.pool_x.shape[-1]
        f = self.vision_layer_norm(vision.float().unflatten(1, (h, w)))  # [B, h, w, C]
        rows = torch.einsum("bnpy,byxc->bnpxc", ex.pool_y, f)
        pooled = torch.einsum("bnpxc,bnqx->bncpq", rows, ex.pool_x).flatten(0, 1)  # [B N, C, 7, 7]
        e = self.boxes_direct_project(ex.boxes) + self.boxes_pool_project(pooled).view(b, n, DIM)
        cx, cy, bw, bh = ex.boxes.unbind(-1)
        s = 2 * math.pi
        enc = torch.cat((_sine(cy * s), _sine(cx * s), bh[..., None], bw[..., None]), dim=-1)
        return self.label_embed(ex.labels) + (e + self.boxes_pos_enc_project(enc))

    def forward(self, vision: torch.Tensor, pos: torch.Tensor, ex: Exemplars | None = None) -> torch.Tensor:
        """`vision`, `pos` `[B, h * w, C]` -> the prompt tokens `[B, N + 1, C]` (the boxes, then CLS)."""
        b = vision.shape[0]
        p = self.cls_embed.weight[None].expand(b, -1, -1)
        bias = None
        if ex is not None:
            p = torch.cat([self.embed_boxes(vision, ex), p], dim=1)
            if ex.valid is not None:  # CLS is always valid: no row is fully masked
                bias = mask_bias(F.pad(ex.valid, (0, 1), value=1.0))
        p = self.prompt_layer_norm(self.final_proj(p))
        key = vision + pos
        for layer in self.layers:
            h = layer.layer_norm1(p)
            p = p + layer.self_attn(h, h, h, bias)
            p = p + layer.cross_attn(layer.layer_norm2(p), key, vision)
            p = p + layer.mlp(layer.layer_norm3(p))
        return self.output_layer_norm(p)


class DetrEncoder(nn.Module):
    """`Sam3DetrEncoder` on the 72 x 72 level: image tokens attend to themselves and to the prompt."""

    def __init__(self, depth: int) -> None:
        """Untrained encoder."""
        super().__init__()
        self.layers = nn.ModuleList(PromptVisionLayer() for _ in range(depth))

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, prompt: torch.Tensor, prompt_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`x`, `pos` `[B, N, C]`, `prompt [B, T, C]`, `prompt_bias [B, 1, 1, T]` (padding mask) -> `[B, N, C]`."""
        for layer in self.layers:
            h = layer.layer_norm1(x)
            hp = h + pos
            x = x + layer.self_attn(hp, hp, h)
            x = x + layer.cross_attn(layer.layer_norm2(x), prompt, prompt, prompt_bias)
            x = x + layer.mlp(layer.layer_norm3(x))
        return x


class DetrDecoderLayer(nn.Module):
    """Post-LN decoder block: self-attention, text cross-attention, image cross-attention (box bias), MLP."""

    def __init__(self) -> None:
        """Untrained block."""
        super().__init__()
        self.self_attn, self.text_cross_attn, self.vision_cross_attn = Attention(), Attention(), Attention()
        self.self_attn_layer_norm = nn.LayerNorm(DIM)
        self.text_cross_attn_layer_norm = nn.LayerNorm(DIM)
        self.vision_cross_attn_layer_norm = nn.LayerNorm(DIM)
        self.mlp = Mlp(DIM, 2048, "relu")
        self.mlp_layer_norm = nn.LayerNorm(DIM)

    def forward(  # noqa: PLR0913 - the block's inputs
        self,
        h: torch.Tensor,
        qpos: torch.Tensor,
        prompt: torch.Tensor,
        vision: torch.Tensor,
        vision_key: torch.Tensor,
        bias: torch.Tensor,
        prompt_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """`h [B, 1 + Q, C]` (presence token first), `qpos [B, 1 + Q, C]` (zero for presence) -> `[B, 1 + Q, C]`."""
        q = h + qpos
        h = self.self_attn_layer_norm(h + self.self_attn(q, q, h))
        h = self.text_cross_attn_layer_norm(h + self.text_cross_attn(h + qpos, prompt, prompt, prompt_bias))
        h = self.vision_cross_attn_layer_norm(h + self.vision_cross_attn(h + qpos, vision_key, vision, bias))
        return self.mlp_layer_norm(h + self.mlp(h))


class DetrDecoder(nn.Module):
    """`Sam3DetrDecoder`: 200 queries + a presence token, iterative box refinement, box relative-position bias."""

    def __init__(self, depth: int, queries: int) -> None:
        """Untrained decoder."""
        super().__init__()
        self.layers = nn.ModuleList(DetrDecoderLayer() for _ in range(depth))
        self.output_layer_norm = nn.LayerNorm(DIM)
        self.box_head = DecoderMLP((DIM, DIM, DIM, 4))
        self.query_embed = nn.Embedding(queries, DIM)
        self.reference_points = nn.Embedding(queries, 4)
        self.presence_token = nn.Embedding(1, DIM)
        self.presence_head = DecoderMLP((DIM, DIM, DIM, 1))
        self.presence_layer_norm = nn.LayerNorm(DIM)
        self.ref_point_head = DecoderMLP((2 * DIM, DIM, DIM))
        self.box_rpb_embed_x = DecoderMLP((2, DIM, DETR_HEADS))
        self.box_rpb_embed_y = DecoderMLP((2, DIM, DETR_HEADS))

    def rpb(self, ref: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Box relative-position bias `[B, heads, 1 + Q, h * w]` (zero row for the presence token; log mode)."""
        b, q, _ = ref.shape
        xyxy = cxcywh_to_xyxy(ref)
        ch = torch.arange(0, h, device=ref.device, dtype=ref.dtype) / h
        cw = torch.arange(0, w, device=ref.device, dtype=ref.dtype) / w
        dy = ch.view(1, 1, -1, 1) - xyxy[..., None, 1:4:2]  # [B, Q, h, 2]
        dx = cw.view(1, 1, -1, 1) - xyxy[..., None, 0:3:2]

        def log(d: torch.Tensor) -> torch.Tensor:
            d = d * 8
            return torch.sign(d) * torch.log2(torch.abs(d) + 1.0) / math.log2(8)

        ey = self.box_rpb_embed_y(log(dy)).permute(0, 3, 1, 2)  # [B, heads, Q, h]
        ex = self.box_rpb_embed_x(log(dx)).permute(0, 3, 1, 2)
        m = torch.zeros(b, ey.shape[1], 1 + q, h * w, device=ref.device, dtype=ref.dtype)  # dense: SDPA's fused kernels
        torch.add(ey[..., None], ex[..., None, :], out=m[:, :, 1:].unflatten(-1, (h, w)))
        return m

    def forward(
        self,
        vision: torch.Tensor,
        pos: torch.Tensor,
        prompt: torch.Tensor,
        hw: tuple[int, int],
        prompt_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Last-layer query states `[B, Q, C]` (output-normed), boxes `[B, Q, 4]` (cxcywh), presence logit `[B, 1]`."""
        b = vision.shape[0]
        ref = self.reference_points.weight[None].expand(b, -1, -1).sigmoid()
        h = torch.cat([self.presence_token.weight, self.query_embed.weight])[None].expand(b, -1, -1)
        key = vision + pos
        out = ref
        for layer in self.layers:
            qpos = F.pad(self.ref_point_head(encode_boxes(ref)), (0, 0, 1, 0))
            h = layer(h, qpos, prompt, vision, key, self.rpb(ref, *hw), prompt_bias)
            out = self.output_layer_norm(h[:, 1:])
            ref = (self.box_head(out) + inverse_sigmoid(ref)).sigmoid()
        presence = self.presence_head(self.presence_layer_norm(h[:, :1])).squeeze(-1).clamp(-10.0, 10.0)
        return out, ref, presence


class DotProductScoring(nn.Module):
    """`Sam3DotProductScoring`: query logits from the mean-pooled prompt."""

    def __init__(self) -> None:
        """Untrained head."""
        super().__init__()
        self.text_mlp = DecoderMLP((DIM, 2048, DIM))
        self.text_mlp_out_norm = nn.LayerNorm(DIM)
        self.text_proj = nn.Linear(DIM, DIM)
        self.query_proj = nn.Linear(DIM, DIM)

    def forward(self, queries: torch.Tensor, prompt: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        """`queries [B, Q, C]`, `prompt [B, T, C]`, `valid [B, T]` (1 = token, 0 = padding; all valid if `None`).

        Returns:
            Logits `[B, Q]`.
        """
        t = self.text_mlp_out_norm(self.text_mlp(prompt) + prompt)
        pooled = t.mean(dim=1) if valid is None else (t * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        pt = self.text_proj(pooled)
        s = (self.query_proj(queries) @ pt[:, :, None]).squeeze(-1) / math.sqrt(DIM)
        return s.clamp(-12.0, 12.0)


class PixelDecoder(nn.Module):
    """`Sam3PixelDecoder`: coarse-to-fine nearest upsampling + skip, 3 x 3 conv, GroupNorm(8), ReLU."""

    def __init__(self, stages: int) -> None:
        """Untrained decoder."""
        super().__init__()
        self.conv_layers = nn.ModuleList(nn.Conv2d(DIM, DIM, 3, padding=1) for _ in range(stages))
        self.norms = nn.ModuleList(nn.GroupNorm(8, DIM) for _ in range(stages))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Levels fine to coarse `[B, C, h_i, w_i]` -> `[B, C, h_0, w_0]`."""
        x = feats[-1]
        for i, f in enumerate(reversed(feats[:-1])):
            x = F.interpolate(x, size=f.shape[-2:], mode="nearest") + f
            x = group_norm_relu(self.conv_layers[i](x), self.norms[i])
        return x


def group_norm_relu(x: torch.Tensor, norm: nn.GroupNorm) -> torch.Tensor:
    """`relu(norm(x))` for `[B, C, H, W]` with one strided reduction for the statistics (torch's GroupNorm kernels
    are slow at 288 x 288, and slower channels-last)."""
    b, c, h, w = x.shape
    g = norm.num_groups
    if x.is_contiguous():
        v, m = torch.var_mean(x.view(b, g, -1), dim=2, correction=0)
    else:  # channels-last
        v, m = torch.var_mean(x.permute(0, 2, 3, 1).reshape(b, h * w, g, c // g), dim=(1, 3), correction=0)
    r = torch.rsqrt(v.float() + norm.eps)  # [B, G]
    scale = (norm.weight.float().view(1, g, -1) * r[..., None]).view(b, c)
    shift = norm.bias.float() - (m.float()[..., None] * scale.view(b, g, -1)).view(b, c)
    return torch.addcmul(shift.to(x.dtype)[..., None, None], x, scale.to(x.dtype)[..., None, None]).relu_()


class MaskEmbedder(nn.Module):
    """Three linear layers with ReLU between."""

    def __init__(self) -> None:
        """Untrained MLP."""
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(DIM, DIM) for _ in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply."""
        for i, layer in enumerate(self.layers):
            x = layer(x) if i == len(self.layers) - 1 else F.relu(layer(x))
        return x


class MaskDecoder(nn.Module):
    """`Sam3MaskDecoder`: the encoder output attends to the prompt, pixel decoder, per-query dot products."""

    def __init__(self) -> None:
        """Untrained decoder (two of three upsampling stages are used)."""
        super().__init__()
        self.pixel_decoder = PixelDecoder(2)
        self.mask_embedder = MaskEmbedder()
        self.instance_projection = nn.Conv2d(DIM, DIM, 1)
        self.prompt_cross_attn = Attention()
        self.prompt_cross_attn_norm = nn.LayerNorm(DIM)

    def forward(
        self,
        queries: torch.Tensor,
        fpn: list[torch.Tensor],
        encoded: torch.Tensor,
        prompt: torch.Tensor,
        prompt_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mask logits `[B, Q, h_0, w_0]` (`fpn` fine to coarse; `encoded` replaces the coarsest level)."""
        e = encoded + self.prompt_cross_attn(self.prompt_cross_attn_norm(encoded), prompt, prompt, prompt_bias)
        b, _, c = e.shape
        h, w = fpn[-1].shape[-2:]
        pix = self.pixel_decoder([*fpn[:-1], e.transpose(1, 2).reshape(b, c, h, w)])
        inst = self.instance_projection(pix)
        return torch.einsum("bqc,bchw->bqhw", self.mask_embedder(queries), inst)


class Sam3(nn.Module):
    """The SAM 3 image detector (`Sam3Model`, `detector_model.*` keys)."""

    def __init__(self, vision: VisionEncoder, text: TextEncoder, *, geometry_depth: int = 3) -> None:
        """Untrained detector around a trunk and a text encoder."""
        super().__init__()
        self.vision_encoder = vision
        self.text_encoder = text
        self.text_projection = nn.Linear(text.text_model.final_layer_norm.normalized_shape[0], DIM)
        self.geometry_encoder = GeometryEncoder(geometry_depth)
        self.detr_encoder = DetrEncoder(6)
        self.detr_decoder = DetrDecoder(6, 200)
        self.mask_decoder = MaskDecoder()
        self.dot_product_scoring = DotProductScoring()

    def prompt(self, ids: torch.Tensor) -> torch.Tensor:
        """Token ids `[B, T]` (real tokens only) -> projected text features `[B, T, 256]`."""
        return self.text_projection(self.text_encoder.text_model(ids))

    def detect(
        self,
        fpn: list[torch.Tensor],
        pos: torch.Tensor,
        text: torch.Tensor,
        text_valid: torch.Tensor | None = None,
        exemplars: Exemplars | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Logits `[B, Q]`, presence `[B, 1]`, boxes `[B, Q, 4]` (normalised xyxy), mask logits `[B, Q, 288, 288]`.

        Each head runs in the dtype of its weights (the DETR encoder and the mask decoder may be float16); the
        outputs are float32.

        Args:
            fpn: Neck levels fine to coarse, `[B, 256, h_i, w_i]`.
            pos: Position encoding of the coarsest level, `[1, 256, h, w]` float32.
            text: `prompt` output, `[B, T, 256]` float32.
            text_valid: `[B, T]` float32, 1 for a real token and 0 for padding after `eot`; `None`: no padding.
            exemplars: Exemplar boxes (transformers `input_boxes`), or `None`.
        """
        top = fpn[-1]
        b, _, h, w = top.shape
        vision = top.flatten(2).transpose(1, 2).float()
        p = pos.flatten(2).transpose(1, 2).expand(b, -1, -1)
        geometry = self.geometry_encoder(vision, p, exemplars)
        prompt = torch.cat([text, geometry], dim=1)
        valid = bias = None
        box_valid = None if exemplars is None else exemplars.valid
        if text_valid is not None or box_valid is not None:  # the CLS token is always valid: no row is fully masked
            ones = text.new_ones(b, prompt.shape[1])
            parts = [ones[:, : text.shape[1]] if text_valid is None else text_valid]
            if exemplars is not None:
                parts.append(ones[:, : exemplars.boxes.shape[1]] if box_valid is None else box_valid)
            valid = torch.cat([*parts, ones[:, :1]], dim=1)
            bias = mask_bias(valid)
        dt = self.detr_encoder.layers[0].mlp.fc1.weight.dtype
        encoded = self.detr_encoder(vision.to(dt), p.to(dt), prompt.to(dt), bias).float()
        queries, boxes, presence = self.detr_decoder(encoded, p, prompt, (h, w), bias)
        logits = self.dot_product_scoring(queries, prompt, valid)
        dt = self.mask_decoder.instance_projection.weight.dtype
        fpn = [f.to(dt) for f in fpn]
        masks = self.mask_decoder(queries.to(dt), fpn, encoded.to(dt), prompt.to(dt), bias).float()
        return logits, presence, cxcywh_to_xyxy(boxes), masks
