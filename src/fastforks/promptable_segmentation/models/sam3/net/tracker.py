"""SAM 3 video tracker heads and memory modules (transformers `modeling_sam3_tracker_video.py`, v5.19.0, Apache-2.0).

The modules of `Sam3TrackerVideoModel` without the vision encoder, with the same attribute names, so the checkpoint's
`tracker_model.*` tensors load without renaming. The released configuration is fixed in the code (no config object);
dropout and training branches are dropped, attention runs through `F.scaled_dot_product_attention` (transformers'
default "sdpa"). Objects run as the batch dimension (transformers runs them one by one; the same maths).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DIM = 256  # fixed: tracker width
MEM_DIM = 64  # fixed: memory feature width
TOKENS = 72  # fixed: image-embedding side (1008 / 14)
SIZE = 1008  # fixed: input side
NUM_MASKMEM = 7  # fixed: memory frames (frame 0 + 6 previous)
STABILITY_DELTA, STABILITY_THRESH = 0.05, 0.98  # fixed: dynamic_multimask_stability_delta / _thresh


class LayerNorm2d(nn.LayerNorm):
    """`Sam3TrackerVideoLayerNorm` with `data_format="channels_first"`: LayerNorm over C of `[B, C, H, W]`."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Width `dim`."""
        super().__init__(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, C, H, W]` -> normalised over C."""
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FeedForward(nn.Module):
    """`Sam3TrackerVideoFeedForward`: `num_layers` linear layers with ReLU between, optional sigmoid."""

    def __init__(self, din: int, hidden: int, dout: int, num_layers: int, *, sigmoid: bool = False) -> None:
        """Widths and depth."""
        super().__init__()
        self.proj_in = nn.Linear(din, hidden)
        self.proj_out = nn.Linear(hidden, dout)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(num_layers - 2))
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[..., din]` -> `[..., dout]`."""
        x = F.relu(self.proj_in(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.proj_out(x)
        return x.sigmoid() if self.sigmoid else x


class Attention(nn.Module):
    """`Sam3TrackerVideoAttention`: 8 heads on `256 / downsample` channels, `[B, P, N, C]` inputs."""

    def __init__(self, downsample: int = 2, heads: int = 8) -> None:
        """Mask-decoder attention (`attention_downsample_rate` 2, 1 for the token self-attention)."""
        super().__init__()
        inner = DIM // downsample
        self.heads = heads
        self.q_proj = nn.Linear(DIM, inner)
        self.k_proj = nn.Linear(DIM, inner)
        self.v_proj = nn.Linear(DIM, inner)
        self.o_proj = nn.Linear(inner, DIM)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Attention of `q [B, P, Nq, C]` over `k`, `v [B, P, Nk, C]`."""
        b, p = q.shape[:2]

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b * p, x.shape[2], self.heads, -1).transpose(1, 2)

        o = F.scaled_dot_product_attention(split(self.q_proj(q)), split(self.k_proj(k)), split(self.v_proj(v)))
        return self.o_proj(o.transpose(1, 2).reshape(b, p, q.shape[2], -1))


class TwoWayAttentionBlock(nn.Module):
    """`Sam3TrackerVideoTwoWayAttentionBlock`."""

    def __init__(self, *, skip_first_layer_pe: bool) -> None:
        """Self-attention, token -> image, MLP 2048, image -> token."""
        super().__init__()
        self.self_attn = Attention(downsample=1)
        self.layer_norm1 = nn.LayerNorm(DIM)
        self.cross_attn_token_to_image = Attention()
        self.layer_norm2 = nn.LayerNorm(DIM)
        self.mlp = FeedForward(DIM, 2048, DIM, 2)
        self.layer_norm3 = nn.LayerNorm(DIM)
        self.layer_norm4 = nn.LayerNorm(DIM)
        self.cross_attn_image_to_token = Attention()
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens `[B, P, T, C]`, image `[B, P, HW, C]` -> both updated."""
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q, q, queries)
        queries = self.layer_norm1(queries)
        q, k = queries + query_pe, keys + key_pe
        queries = self.layer_norm2(queries + self.cross_attn_token_to_image(q, k, keys))
        queries = self.layer_norm3(queries + self.mlp(queries))
        q, k = queries + query_pe, keys + key_pe
        keys = self.layer_norm4(keys + self.cross_attn_image_to_token(k, q, queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    """`Sam3TrackerVideoTwoWayTransformer`: 2 blocks + final token -> image attention."""

    def __init__(self) -> None:
        """Released depth 2."""
        super().__init__()
        self.layers = nn.ModuleList(TwoWayAttentionBlock(skip_first_layer_pe=i == 0) for i in range(2))
        self.final_attn_token_to_image = Attention()
        self.layer_norm_final_attn = nn.LayerNorm(DIM)

    def forward(
        self, tokens: torch.Tensor, image: torch.Tensor, image_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens `[B, P, T, C]`, image / PE `[B, C, H, W]` -> tokens, image `[B, P, HW, C]`."""
        keys = image.flatten(2).transpose(1, 2)[:, None]
        key_pe = image_pe.flatten(2).transpose(1, 2)[:, None]
        queries = tokens
        for layer in self.layers:
            queries, keys = layer(queries, keys, tokens, key_pe)
        q, k = queries + tokens, keys + key_pe
        queries = self.layer_norm_final_attn(queries + self.final_attn_token_to_image(q, k, keys))
        return queries, keys


class MaskDecoder(nn.Module):
    """`Sam3TrackerVideoMaskDecoder` (4 mask tokens, object-score token, dynamic multimask via stability)."""

    def __init__(self) -> None:
        """Released configuration."""
        super().__init__()
        self.iou_token = nn.Embedding(1, DIM)
        self.mask_tokens = nn.Embedding(4, DIM)
        self.transformer = TwoWayTransformer()
        self.upscale_conv1 = nn.ConvTranspose2d(DIM, DIM // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(DIM // 4, DIM // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = LayerNorm2d(DIM // 4)
        self.output_hypernetworks_mlps = nn.ModuleList(FeedForward(DIM, DIM, DIM // 8, 3) for _ in range(4))
        self.iou_prediction_head = FeedForward(DIM, 256, 4, 3, sigmoid=True)
        self.conv_s0 = nn.Conv2d(DIM, DIM // 8, kernel_size=1)
        self.conv_s1 = nn.Conv2d(DIM, DIM // 4, kernel_size=1)
        self.obj_score_token = nn.Embedding(1, DIM)
        self.pred_obj_score_head = FeedForward(DIM, DIM, 1, 3)

    def forward(  # noqa: PLR0913 - upstream arguments
        self,
        image: torch.Tensor,
        image_pe: torch.Tensor,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        hr: list[torch.Tensor],
        *,
        multimask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Image `[B, 256, 72, 72]`, sparse `[B, 1, N, 256]`, dense `[B or 1, 256, 72, 72]`, `hr` (s0, s1).

        Returns:
            Mask logits `[B, M, 288, 288]`, IoU `[B, M]`, mask tokens `[B, M, 256]`, object score `[B, 1]`; `M` 3
            (multimask) or 1 (token 0, or the best multimask output when token 0 is unstable).
        """
        b, c, h, w = image.shape
        out = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight])
        tokens = torch.cat((out.expand(b, 1, -1, -1).to(sparse.dtype), sparse), dim=2)
        q, keys = self.transformer(tokens, image + dense, image_pe.expand(b, -1, -1, -1))
        iou_out, mask_out = q[:, 0, 1], q[:, 0, 2:6]
        img = keys[:, 0].transpose(1, 2).reshape(b, c, h, w)
        feat_s0, feat_s1 = hr
        up = F.gelu(self.upscale_layer_norm(self.upscale_conv1(img) + feat_s1))
        up = F.gelu(self.upscale_conv2(up) + feat_s0)
        hyper = torch.stack([mlp(mask_out[:, i]) for i, mlp in enumerate(self.output_hypernetworks_mlps)], 1)
        masks = (hyper @ up.flatten(2)).view(b, 4, *up.shape[2:])
        iou = self.iou_prediction_head(iou_out)
        obj = self.pred_obj_score_head(q[:, 0, 0])
        if multimask:
            return masks[:, 1:], iou[:, 1:], mask_out[:, 1:], obj
        best = iou[:, 1:].argmax(-1)
        rows = torch.arange(b, device=masks.device)
        single = masks[:, 0].flatten(1)
        area_i = (single > STABILITY_DELTA).sum(-1).float()
        area_u = (single > -STABILITY_DELTA).sum(-1).float()
        stable = torch.where(area_u > 0, area_i / area_u, 1.0) >= STABILITY_THRESH
        m = torch.where(stable[:, None, None], masks[:, 0], masks[rows, best + 1])[:, None]
        s = torch.where(stable, iou[:, 0], iou[rows, best + 1])[:, None]
        return m, s, mask_out[:, :1], obj


class PositionalEmbedding(nn.Module):
    """`Sam3TrackerVideoPositionalEmbedding`: random Fourier features of `[0, 1]` coordinates."""

    def __init__(self) -> None:
        """`positional_embedding [2, 128]` (a checkpoint buffer)."""
        super().__init__()
        self.register_buffer("positional_embedding", torch.zeros(2, DIM // 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """`[..., 2]` in `[0, 1]` -> `[..., 256]`."""
        x = (2 * coords - 1).to(self.positional_embedding.dtype) @ self.positional_embedding
        x = 2 * math.pi * x
        return torch.cat([x.sin(), x.cos()], dim=-1)


class MaskEmbedding(nn.Module):
    """`Sam3TrackerVideoMaskEmbedding`: `[B, 1, 288, 288]` mask logits -> `[B, 256, 72, 72]`."""

    def __init__(self) -> None:
        """Two 2 x 2 stride-2 convs (4, 16 channels) with LayerNorm + GELU, then 1 x 1 to 256."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, DIM, kernel_size=1)
        self.layer_norm1 = LayerNorm2d(4)
        self.layer_norm2 = LayerNorm2d(16)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Embed."""
        x = F.gelu(self.layer_norm1(self.conv1(masks)))
        x = F.gelu(self.layer_norm2(self.conv2(x)))
        return self.conv3(x)


class PromptEncoder(nn.Module):
    """`Sam3TrackerVideoPromptEncoder` for points (boxes come as two points with labels 2, 3) and masks."""

    def __init__(self) -> None:
        """Released configuration."""
        super().__init__()
        self.shared_embedding = PositionalEmbedding()
        self.mask_embed = MaskEmbedding()
        self.no_mask_embed = nn.Embedding(1, DIM)
        self.point_embed = nn.Embedding(4, DIM)
        self.not_a_point_embed = nn.Embedding(1, DIM)

    def forward(
        self, points: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Points `[B, 1, N, 2]` (1008-px input coordinates), labels `[B, 1, N]`, masks `[B, 1, 288, 288]` or None.

        Returns:
            Sparse `[B, 1, N + 1, 256]` (a not-a-point token appended, upstream `pad`), dense `[B or 1, 256, 72, 72]`.
        """
        points = F.pad(points + 0.5, (0, 0, 0, 1))
        labels = F.pad(labels, (0, 1), value=-1)
        emb = self.shared_embedding(points / SIZE)
        emb = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, emb)  # noqa: PLR2004
        emb = torch.where(labels[..., None] != -10, emb, torch.zeros_like(emb))  # noqa: PLR2004
        emb = emb + self.point_embed(labels.clamp(min=0)) * (labels >= 0)[..., None]
        if masks is not None:
            return emb, self.mask_embed(masks)
        return emb, self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(1, -1, TOKENS, TOKENS)


def rope_tables(device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """`Sam3TrackerVideoVisionRotaryEmbedding` on the 72 x 72 grid: cos, sin `[1, 5184, 256]` float32."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, DIM // 2, 2, dtype=torch.float32, device=device) / (DIM // 2)))
    h, w = torch.meshgrid(torch.arange(TOKENS, device=device), torch.arange(TOKENS, device=device), indexing="ij")
    ids = torch.stack([w.flatten(), h.flatten()], dim=-1).float()  # transformers' order: (w, h)
    freqs = ids[..., None] * inv  # [N, 2, 64]
    freqs = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)[None].repeat_interleave(2, dim=-1)
    return freqs.cos(), freqs.sin()


def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    """`(x0, x1, ...)` pairs -> `(-x1, x0, ...)`."""
    x1, x2 = x.view(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPEAttention(nn.Module):
    """`Sam3TrackerVideoRoPEAttention`: one head of 256, axial RoPE on q and the spatial keys (float32)."""

    def __init__(self, kv_in_dim: int = DIM, *, rope_k_repeat: bool = False) -> None:
        """Self-attention (`kv_in_dim` 256) or memory cross-attention (64, keys repeat the RoPE per memory frame)."""
        super().__init__()
        self.q_proj = nn.Linear(DIM, DIM)
        self.k_proj = nn.Linear(kv_in_dim, DIM)
        self.v_proj = nn.Linear(kv_in_dim, DIM)
        self.o_proj = nn.Linear(DIM, DIM)
        self.rope_k_repeat = rope_k_repeat

    def forward(  # noqa: PLR0913 - upstream arguments
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """`q [B, Nq, C]`, `k`, `v [B, Nk, Ck]` -> `[B, Nq, C]`; the last `num_k_exclude_rope` keys without RoPE."""
        cos, sin = rope
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        qf = q.float()
        q = (qf * cos + rotate_pairwise(qf) * sin).to(v.dtype)
        n = k.shape[1] - num_k_exclude_rope
        k_rot, k_pass = k[:, :n], k[:, n:]
        if self.rope_k_repeat and n != q.shape[1]:
            cos, sin = cos.repeat(1, n // q.shape[1], 1), sin.repeat(1, n // q.shape[1], 1)
        kf = k_rot.float()
        k = torch.cat([(kf * cos + rotate_pairwise(kf) * sin).to(v.dtype), k_pass], dim=1)
        return self.o_proj(F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0])


class MemoryAttentionLayer(nn.Module):
    """`Sam3TrackerVideoMemoryAttentionLayer` (pre-norm self, cross to memory, ReLU MLP 2048)."""

    def __init__(self) -> None:
        """Released configuration."""
        super().__init__()
        self.self_attn = RoPEAttention()
        self.cross_attn_image = RoPEAttention(MEM_DIM, rope_k_repeat=True)
        self.linear1 = nn.Linear(DIM, 2048)
        self.linear2 = nn.Linear(2048, DIM)
        self.layer_norm1 = nn.LayerNorm(DIM)
        self.layer_norm2 = nn.LayerNorm(DIM)
        self.layer_norm3 = nn.LayerNorm(DIM)

    def forward(  # noqa: PLR0913 - upstream arguments
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        num_k_exclude_rope: int,
    ) -> torch.Tensor:
        """`x [B, HW, C]`, memory / pos `[B, M, 64]`."""
        q = self.layer_norm1(x)
        x = x + self.self_attn(q, q, q, rope)
        q = self.layer_norm2(x)
        x = x + self.cross_attn_image(q, memory + memory_pos, memory, rope, num_k_exclude_rope)
        return x + self.linear2(F.relu(self.linear1(self.layer_norm3(x))))


class MemoryAttention(nn.Module):
    """`Sam3TrackerVideoMemoryAttention`: 4 layers + LayerNorm, the input plus 0.1 x its position encoding."""

    def __init__(self) -> None:
        """Released configuration; the RoPE tables are not in the checkpoint (`rope_tables`)."""
        super().__init__()
        self.layers = nn.ModuleList(MemoryAttentionLayer() for _ in range(4))
        self.layer_norm = nn.LayerNorm(DIM)

    def forward(  # noqa: PLR0913 - upstream arguments
        self,
        curr: torch.Tensor,
        curr_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        num_obj_ptr_tokens: int,
    ) -> torch.Tensor:
        """Batch first: `curr`, `curr_pos [B, HW, 256]`, `memory`, `memory_pos [B, M, 64]` -> `[B, HW, 256]`."""
        x = curr + 0.1 * curr_pos
        for layer in self.layers:
            x = layer(x, memory, memory_pos, rope, num_obj_ptr_tokens)
        return self.layer_norm(x)


class CXBlock(nn.Module):
    """`Sam3TrackerVideoMemoryFuserCXBlock` (ConvNeXt block: dw 7 x 7, LN, 1024 GELU, layer scale)."""

    def __init__(self) -> None:
        """Released configuration."""
        super().__init__()
        self.depthwise_conv = nn.Conv2d(DIM, DIM, kernel_size=7, padding=3, groups=DIM)
        self.layer_norm = LayerNorm2d(DIM)
        self.pointwise_conv1 = nn.Linear(DIM, 1024)
        self.pointwise_conv2 = nn.Linear(1024, DIM)
        self.scale = nn.Parameter(torch.ones(DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 256, H, W]` -> same."""
        y = self.layer_norm(self.depthwise_conv(x)).permute(0, 2, 3, 1)
        y = self.scale * self.pointwise_conv2(F.gelu(self.pointwise_conv1(y)))
        return x + y.permute(0, 3, 1, 2)


class MemoryFuser(nn.Module):
    """`Sam3TrackerVideoMemoryFuser`: 2 CX blocks."""

    def __init__(self) -> None:
        """Released depth."""
        super().__init__()
        self.layers = nn.ModuleList(CXBlock() for _ in range(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the blocks."""
        for layer in self.layers:
            x = layer(x)
        return x


class MaskDownSamplerLayer(nn.Module):
    """`Sam3TrackerVideoMaskDownSamplerLayer`: 3 x 3 stride-2 conv, LayerNorm, GELU."""

    def __init__(self, cin: int, cout: int) -> None:
        """Channels."""
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1)
        self.layer_norm = LayerNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample 2x."""
        return F.gelu(self.layer_norm(self.conv(x)))


class MaskDownSampler(nn.Module):
    """`Sam3TrackerVideoMaskDownSampler`: 16x down (1 -> 4 -> 16 -> 64 -> 256 channels), 1 x 1 to 256."""

    def __init__(self) -> None:
        """Released configuration (total stride 16)."""
        super().__init__()
        self.layers = nn.ModuleList(MaskDownSamplerLayer(4**i, 4 ** (i + 1)) for i in range(4))
        self.final_conv = nn.Conv2d(256, DIM, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 1, 1152, 1152]` -> `[B, 256, 72, 72]`."""
        for layer in self.layers:
            x = layer(x)
        return self.final_conv(x)


class MemoryEncoder(nn.Module):
    """`Sam3TrackerVideoMemoryEncoder` without its constant position encoding (`memory_position_encoding`)."""

    def __init__(self) -> None:
        """Released configuration."""
        super().__init__()
        self.mask_downsampler = MaskDownSampler()
        self.feature_projection = nn.Conv2d(DIM, DIM, kernel_size=1)
        self.memory_fuser = MemoryFuser()
        self.projection = nn.Conv2d(DIM, MEM_DIM, kernel_size=1)

    def forward(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Features `[B, 256, 72, 72]`, mask logits `[B, 1, 1152, 1152]` -> memory `[B, 64, 72, 72]`."""
        x = self.feature_projection(features) + self.mask_downsampler(masks)
        return self.projection(self.memory_fuser(x))


def memory_position_encoding(device: torch.device | None = None) -> torch.Tensor:
    """The memory encoder's normalised sine encoding (32 features per axis), `[1, 64, 72, 72]` float32."""
    eps, scale, n = 1e-6, 2 * math.pi, MEM_DIM // 2
    y = torch.arange(1, TOKENS + 1, dtype=torch.float32, device=device)[:, None].expand(TOKENS, TOKENS)
    x = torch.arange(1, TOKENS + 1, dtype=torch.float32, device=device)[None, :].expand(TOKENS, TOKENS)
    y, x = y / (y[-1:] + eps) * scale, x / (x[:, -1:] + eps) * scale
    dim_t = 10000 ** (2 * torch.div(torch.arange(n, device=device), 2, rounding_mode="floor").float() / n)

    def sine(p: torch.Tensor) -> torch.Tensor:
        p = p[..., None] / dim_t
        return torch.stack((p[..., 0::2].sin(), p[..., 1::2].cos()), dim=-1).flatten(-2)

    return torch.cat((sine(y), sine(x)), dim=-1).permute(2, 0, 1)[None]


def get_1d_sine_pe(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformers `get_1d_sine_pe` (temperature 10000)."""
    half = dim // 2
    dim_t = 10000 ** (2 * (torch.arange(half, dtype=torch.float32, device=pos.device) // 2) / half)
    p = pos[..., None] / dim_t
    return torch.cat([p.sin(), p.cos()], dim=-1)


class TrackerModel(nn.Module):
    """`Sam3TrackerVideoModel` without the vision encoder (checkpoint prefix `tracker_model.`)."""

    def __init__(self) -> None:
        """Released configuration (untrained)."""
        super().__init__()
        self.shared_image_embedding = PositionalEmbedding()
        self.prompt_encoder = PromptEncoder()
        self.mask_decoder = MaskDecoder()
        self.no_memory_embedding = nn.Parameter(torch.zeros(1, 1, DIM))
        self.memory_attention = MemoryAttention()
        self.memory_encoder = MemoryEncoder()
        self.no_memory_positional_encoding = nn.Parameter(torch.zeros(1, 1, DIM))  # unused (as upstream)
        self.memory_temporal_positional_encoding = nn.Parameter(torch.zeros(NUM_MASKMEM, 1, 1, MEM_DIM))
        self.no_object_pointer = nn.Parameter(torch.zeros(1, DIM))
        self.mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.object_pointer_proj = FeedForward(DIM, DIM, DIM, 3)
        self.temporal_positional_encoding_projection_layer = nn.Linear(DIM, MEM_DIM)
        self.occlusion_spatial_embedding_parameter = nn.Parameter(torch.zeros(1, MEM_DIM))

    def image_pe(self) -> torch.Tensor:
        """`get_image_wide_positional_embeddings`: `[1, 256, 72, 72]`."""
        pe = self.shared_image_embedding.positional_embedding
        c = (torch.arange(TOKENS, dtype=pe.dtype, device=pe.device) + 0.5) / TOKENS
        grid = torch.stack([c[None, :].expand(TOKENS, TOKENS), c[:, None].expand(TOKENS, TOKENS)], dim=-1)
        return self.shared_image_embedding(grid).permute(2, 0, 1)[None]
