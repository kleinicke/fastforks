# MIT License, Copyright (c) 2026 Computer Vision and Geometry Lab (LICENSE)
"""FlexGlue's matcher, sparse mode: upstream cvg/FlexGlue `flexglue/flexglue.py` (`decode_pair`,
`keypoint_correspondences` without the cycle readout, `gate_points`, `sparse_matches` with `gemm_nn`),
`flexglue/utils/modules.py` (transformer layers, `MatchHead`, `OffsetHead`) and `flexglue/utils/ops.py`.

fastforks changes: inference of the sparse mode only, batch 1 (no refiner, no dense / sampled modes, no early stop,
no `torch.compile` padding); only the head of the last layer is built (the one sparse matching reads), without its
covisibility layer (`disambiguate`: with upstream's `min_covisibility = 0` the score gates nothing).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

ENTMAX_K = 512  # fixed: upstream `entmax15_window_readout` top-k
WINDOW_RADIUS = 1  # fixed: upstream `entmax15_window_readout` radius (a 3 x 3 window around the maximum)
QUERY_CHUNK = 4096  # fixed: upstream `query_chunk_size` (bounds the [chunk, h * w] logits)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pairs (x1, x2) -> (-x2, x1) along the last axis."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), -1).flatten(-2)


def apply_rotary(enc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotary encoding `enc` (cos, sin) applied to `t`."""
    return t * enc[0] + rotate_half(t) * enc[1]


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention on token-major (B, N, H, D) tensors, returns (B, N, H * D)."""
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return out.transpose(1, 2).flatten(-2)


def mlp(dims: list[int]) -> nn.Sequential:
    """Linear layers with LayerNorm + ReLU between them."""
    layers: list[nn.Module] = []
    for i in range(1, len(dims)):
        layers.append(nn.Linear(dims[i - 1], dims[i]))
        if i < len(dims) - 1:
            layers += [nn.LayerNorm(dims[i]), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def ffn(dim: int) -> nn.Sequential:
    """The transformer blocks' feed-forward net on [x, message]."""
    return nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))


class LearnableFourierPositionalEncoding(nn.Module):
    """(cos, sin) of a learned projection of the cell centres, repeated per pair of channels."""

    def __init__(self, m: int, dim: int) -> None:  # noqa: D107
        super().__init__()
        self.Wr = nn.Linear(m, dim // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        projected = self.Wr(x)
        emb = torch.stack([projected.cos(), projected.sin()], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class SelfBlock(nn.Module):
    """Self-attention with q / k LayerNorms and rotary encoding (Wqkv rows ordered (H, D, 3) as the checkpoint)."""

    def __init__(self, dim: int, num_heads: int) -> None:  # noqa: D107
        super().__init__()
        self.num_heads = num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = ffn(dim)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:  # noqa: D102
        qkv = self.Wqkv(x)  # token-major (B, N, H, D) q, k, v
        q, k, v = qkv.unflatten(-1, (self.num_heads, -1, 3)).unbind(-1)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        encoding = encoding.transpose(-3, -2)  # broadcast over (N, H) instead of (H, N)
        q, k = apply_rotary(encoding, q), apply_rotary(encoding, k)
        message = self.out_proj(attention(q, k, v.contiguous()))
        return x + self.ffn(torch.cat([x, self.message_norm(message)], -1))


class CrossBlock(nn.Module):
    """Symmetric cross-attention with shared q / k projections."""

    def __init__(self, dim: int, num_heads: int) -> None:  # noqa: D107
        super().__init__()
        self.num_heads = num_heads
        self.to_qk = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.qk_norm = nn.LayerNorm(dim // num_heads)
        self.message_norm = nn.LayerNorm(dim)
        self.ffn = ffn(dim)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D102
        qk0, qk1, v0, v1 = (
            t.unflatten(-1, (self.num_heads, -1))
            for t in (self.to_qk(x0), self.to_qk(x1), self.to_v(x0), self.to_v(x1))
        )
        qk0, qk1 = self.qk_norm(qk0).to(v0.dtype), self.qk_norm(qk1).to(v1.dtype)
        m0 = self.to_out(attention(qk0, qk1, v1))
        m1 = self.to_out(attention(qk1, qk0, v0))
        x0 = x0 + self.ffn(torch.cat([x0, self.message_norm(m0)], -1))
        x1 = x1 + self.ffn(torch.cat([x1, self.message_norm(m1)], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    """Self-attention in each image, then cross-attention between them."""

    def __init__(self, dim: int, num_heads: int) -> None:  # noqa: D107
        super().__init__()
        self.self_attn = SelfBlock(dim, num_heads)
        self.cross_attn = CrossBlock(dim, num_heads)

    def forward(  # noqa: D102
        self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor, encoding1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cross_attn(self.self_attn(x0, encoding0), self.self_attn(x1, encoding1))


def cell_centers(h: int, w: int, scale: tuple[float, float], device: torch.device) -> torch.Tensor:
    """Normalized (x, y) centers (h*w, 2) of an h x w grid, divided by (sx, sy)."""
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    xy = torch.stack([(x + 0.5) / scale[0], (y + 0.5) / scale[1]], -1)
    return xy.flatten(0, 1) * 2 - 1


def sample_features(features: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample (B, C, H, W) features at points (B, N, 2) in [-1, 1]."""
    out = F.grid_sample(features, pts[:, None], "bilinear", padding_mode="border", align_corners=False)
    return out[:, :, 0].transpose(1, 2)


def sorted_entmax15_threshold(ys: torch.Tensor) -> torch.Tensor:
    """Exact 1.5-entmax threshold from scores sorted in descending order."""
    rho = torch.arange(1, ys.shape[-1] + 1, device=ys.device, dtype=ys.dtype)
    mean = ys.cumsum(-1) / rho
    mean_sq = ys.square().cumsum(-1) / rho
    tau = mean - ((1 - rho * (mean_sq - mean.square())) / rho).clamp_min(0).sqrt()
    support = (tau <= ys).sum(-1, keepdim=True)
    return tau.gather(-1, support - 1)


def entmax15_window_readout(
    logits: torch.Tensor, grid: torch.Tensor, hw: tuple[int, int], radius: int = WINDOW_RADIUS, k: int = ENTMAX_K
) -> tuple[torch.Tensor, torch.Tensor]:
    """1.5-entmax expected position (B, N, 2) and mass (B, N) in the window around the argmax."""
    if logits.shape[-1] > k:  # maximum and threshold from the same (sorted) top-k
        topv, topi = logits.topk(k, dim=-1)
        vmax, idx = topv[..., :1], topi[..., :1]
        y = (logits - vmax) / 2
        tau = sorted_entmax15_threshold((topv - vmax) / 2)
    else:  # upstream `entmax15_threshold` on <= k scores: the full sort
        vmax, idx = logits.max(-1, keepdim=True)
        y = (logits - vmax) / 2
        tau = sorted_entmax15_threshold(y.sort(-1, descending=True).values)
    h, w = hw
    r = torch.arange(-radius, radius + 1, device=y.device)
    ny = (idx // w)[..., None] + r[:, None]
    nx = (idx % w)[..., None] + r[None, :]
    inside = ((ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)).flatten(-2)
    nidx = (ny.clamp(0, h - 1) * w + nx.clamp(0, w - 1)).flatten(-2)
    p = (y.gather(-1, nidx) - tau).clamp_min(0).square()  # entmax sums to 1
    position = (p[..., None] * grid[nidx]).sum(-2) / p.sum(-1, keepdim=True)
    return position, (p * inside).sum(-1)  # clamped border cells count once


class OffsetHead(nn.Module):
    """Sub-cell offset as a soft-argmax over bins spanning +-max_offset cells."""

    def __init__(self, dim: int, hidden: int = 256, bins: int = 16, max_offset: float = 1.0) -> None:  # noqa: D107
        super().__init__()
        self.bins, self.max_offset = bins, max_offset
        self.query_encoder = mlp([dim, hidden, hidden])
        self.match_encoder = mlp([dim, hidden, hidden])
        self.head = mlp([2 * hidden, 2 * hidden, 2 * (bins + 2)])  # last bin unused

    def forward(self, f_query: torch.Tensor, f_match: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:  # noqa: D102
        x = torch.cat([self.query_encoder(f_query), self.match_encoder(f_match)], -1)
        logits = self.head(x).unflatten(-1, (2, self.bins + 2))[..., : self.bins + 1]
        bins = torch.arange(self.bins + 1, device=x.device, dtype=x.dtype)
        delta = ((logits.softmax(-1) * bins).sum(-1) / self.bins - 0.5) * 2
        delta = delta * self.max_offset
        return torch.stack([delta[..., 0] * 2 / hw[1], delta[..., 1] * 2 / hw[0]], -1)


class MatchHead(nn.Module):
    """Regress the correspondence of each point in the other image's feature map (a cross-view head)."""

    def __init__(self, dim: int, search_dim: int) -> None:  # noqa: D107
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(10.0))
        self.to_search = nn.Linear(dim, search_dim)
        self.matchability = nn.Linear(dim, 1)
        self.offset_head = OffsetHead(dim)

    def forward(
        self, points_ni: torch.Tensor, cf_i: torch.Tensor, cf_j: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized points (B, N, 2) of i -> predicted points in j and confidence."""
        search = F.normalize(self.to_search(cf_j.flatten(2).mT), dim=-1)
        chunks = [self.match_points(points, cf_i, cf_j, search) for points in points_ni.split(QUERY_CHUNK, dim=1)]
        if len(chunks) == 1:
            return chunks[0]
        p2di_nj, confidence = zip(*chunks, strict=True)
        return torch.cat(p2di_nj, 1), torch.cat(confidence, 1)

    def match_points(
        self, points_ni: torch.Tensor, cf_i: torch.Tensor, cf_j: torch.Tensor, search: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`forward` on one chunk of points, with the normalized search features of j."""
        hw = (cf_j.shape[-2], cf_j.shape[-1])
        f_query = sample_features(cf_i, points_ni)
        query = F.normalize(self.to_search(f_query), dim=-1)
        logits = query @ search.mT * self.temperature
        grid = cell_centers(*hw, scale=(hw[1], hw[0]), device=logits.device)
        p2di_nj, window_mass = entmax15_window_readout(logits, grid, hw)
        f_match = sample_features(cf_j, p2di_nj)
        p2di_nj = p2di_nj + self.offset_head(f_query, f_match, hw)
        matchability = torch.sigmoid(self.matchability(f_query).squeeze(-1))
        return p2di_nj, matchability * window_mass


def sq_dist_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared dist (B, M, N) of x (B, M, D) and y (B, N, D) in one GEMM, no sqrt."""
    ones_x, ones_y = x.new_ones(*x.shape[:-1], 1), y.new_ones(*y.shape[:-1], 1)
    xa = torch.cat([-2 * x, x.square().sum(-1, keepdim=True), ones_x], -1)
    ya = torch.cat([y, ones_y, y.square().sum(-1, keepdim=True)], -1)
    return xa @ ya.mT


def mutual_nn(dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual nearest neighbors of a (B, M, N) distance matrix, -1 if not mutual."""
    m0, m1 = dist.argmin(2), dist.argmin(1)
    mutual0 = torch.arange(m0.shape[1], device=m0.device) == m1.gather(1, m0)
    mutual1 = torch.arange(m1.shape[1], device=m1.device) == m0.gather(1, m1)
    return torch.where(mutual0, m0, -1), torch.where(mutual1, m1, -1)


def inside(pts: torch.Tensor, size: torch.Tensor, border: float) -> torch.Tensor:
    """Points (B, N, 2) in pixels at least `border` pixels inside a (w, h) `size` image (upstream `inside_border`)."""
    return ((pts > border) & (pts < size - border)).all(-1)


class FlexGlueNet(nn.Module):
    """Input projection, positional encoding, `n_layers` transformer layers and the last layer's match head."""

    def __init__(self, input_dim: int = 256, dim: int = 512, search_dim: int = 64, n_layers: int = 9, heads: int = 8):
        """Upstream `FlexGlue.default_conf` (input_dim, descriptor_dim, search_dim, n_layers, num_heads)."""
        super().__init__()
        self.input_proj = nn.Linear(input_dim, dim)
        self.posenc = LearnableFourierPositionalEncoding(2, dim // heads)
        self.transformers = nn.ModuleList([TransformerLayer(dim, heads) for _ in range(n_layers)])
        self.head = MatchHead(dim, search_dim)  # upstream heads[n_layers]

    def decode(
        self, f0: torch.Tensor, f1: torch.Tensor, amp: torch.dtype | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross features `[B, D, h, w]` (float32) of both dense feature maps; the layers under `amp` autocast."""
        hw0, hw1 = (f0.shape[-2], f0.shape[-1]), (f1.shape[-2], f1.shape[-1])
        enc0, enc1 = (self.posenc(cell_centers(*hw, scale=(max(hw),) * 2, device=f0.device)) for hw in (hw0, hw1))
        x0, x1 = self.input_proj(f0.flatten(2).mT), self.input_proj(f1.flatten(2).mT)
        with torch.autocast(f0.device.type, amp or torch.float16, enabled=amp is not None):
            for block in self.transformers:
                x0, x1 = block(x0, x1, enc0, enc1)
        x0, x1 = x0.float(), x1.float()
        return x0.mT.unflatten(-1, hw0), x1.mT.unflatten(-1, hw1)
