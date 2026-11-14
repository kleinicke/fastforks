import torch
import torch.nn as nn
import torch.nn.functional as F

from fastforks.shared.depth_anything_v2 import Block, SdpaAttention  # fastforks: upstream's dinov2 Block (SDPA)


def _morton(p):  # fastforks: z-order of 10-bit cells of the bounding box (compact chunks of query rows)
    lo, hi = p.min(0).values, p.max(0).values
    v = ((p - lo) / (hi - lo).clamp_min(1e-12) * 1023).long()
    for shift, mask in ((16, 0x030000FF), (8, 0x0300F00F), (4, 0x030C30C3), (2, 0x09249249)):
        v = (v | (v << shift)) & mask
    return v[:, 0] | (v[:, 1] << 1) | (v[:, 2] << 2)


def _knn_pooled(k, p, qi, idx, rows, pool, margin, budget, g):  # fastforks: one level of knn_query
    n = p.shape[0]
    s, m = min(n, max(pool, k)), min(n, max(pool, k), k + margin)
    rows = min(rows, len(qi))
    nc = -(-len(qi) // rows)
    qi = torch.cat((qi, qi[-1:].expand(nc * rows - len(qi)))).view(nc, rows)  # last chunk padded by repeats
    bad = torch.zeros(n, dtype=torch.bool, device=p.device)
    grp = max(1, budget // (rows * s + n * 3))
    for i in range(0, nc, grp):
        rid = qi[i : i + grp]
        q = p[rid]  # [G, rows, 3]
        c = q.mean(1, keepdim=True)
        if s < n:
            r2, near = (p - c).square().sum(-1).topk(s + 1, dim=1, largest=False)  # sorted: [:, s] = nearest outside
            cand = near[:, :s]
        else:
            cand = torch.arange(n, device=p.device).expand(len(rid), n)
        qc, pc = q - c, p[cand] - c
        a = qc.square().sum(-1)
        d = torch.baddbmm(pc.square().sum(-1)[:, None], qc, pc.transpose(1, 2), alpha=-2).add_(a[..., None])
        dm, j = d.topk(m, dim=-1, largest=False, sorted=False)
        cand = cand.gather(1, j.flatten(1)).view_as(j)
        dk, j = (p[cand] - q[..., None, :]).square().sum(-1).topk(k, dim=-1, largest=False, sorted=False)
        idx[rid] = cand.gather(-1, j)
        # |error| <= g (|q - c|^2 + |p - c|^2), |p - c|^2 <= 2 |q - c|^2 + 2 d:
        # every non-candidate is >= (d_m - 3 g a) / (1 + 2 g)
        dk = dk.amax(-1) * (1 + 1e-6)
        out = (dm.amax(-1) - 3 * g * a) / (1 + 2 * g) <= dk if m < s else torch.zeros_like(dk, dtype=torch.bool)
        if s < n:
            far = r2[:, s, None].sqrt() * (1 - 1e-6) - a.sqrt() * (1 + 1e-6)
            out |= (far <= 0) | (dk >= far.square())
        bad[rid] = out
    return bad.nonzero()[:, 0]  # one host sync per level


def knn_query(k, p, rows=128, pool=4096, margin=16, budget=1 << 25):  # fastforks: replaces pointops.knn_query
    """Indices `[N, k]` of the `k` nearest points (including the point itself) of every point of `p` `[N, 3]`.

    Exact squared distances as pointops (a CUDA extension; one batch), without the all-pairs matrix: z-ordered chunks of
    `rows` queries search the `pool` points nearest to the chunk's centre `c`, preselect `k + margin` of them by an
    fp32 matmul about `c` and rank those exactly. A query is redone (4x fewer rows, 4x more pool, then over all
    points) unless its k-th distance is below the bounds of the matmul error and of `rho - |q - c|` (`rho`: the
    nearest point outside the pool). Chunks run in groups of about `budget` distances. The neighbours' order does not
    matter (the attention sums over them).
    """
    n = p.shape[0]
    g = 1e-6 if torch.get_float32_matmul_precision() == "highest" else 1e-2  # relative matmul error (TF32 / bf16)
    with torch.autocast(p.device.type, enabled=False):
        p = p.float()
        idx = torch.empty(n, k, dtype=torch.long, device=p.device)
        qi = _morton(p).argsort()
        while len(qi) and pool < n:
            qi = _knn_pooled(k, p, qi, idx, rows, pool, margin, budget, g)  # bad rows stay in z-order
            rows, pool = max(1, rows // 4), pool * 4
        step = max(1, budget // (3 * n))
        for t in range(0, len(qi), step):
            b = qi[t : t + step]
            idx[b] = (p[None] - p[b, None]).square().sum(-1).topk(k, dim=1, largest=False, sorted=False)[1]
    return idx


class KNNAttention(nn.Module):
    def __init__(self, channels, knn_samples=16,
        num_heads=1,
        proj_channels=None,
        ):
        super().__init__()
        self.proj_channels = proj_channels

        self.knn_samples = knn_samples
        self.num_heads = num_heads
        assert self.num_heads == 1

        if self.proj_channels is not None:
            self.qkv = nn.Linear(channels, self.proj_channels * 3, bias=False)
            self.proj = nn.Linear(self.proj_channels, channels)
        else:
            self.qkv = nn.Linear(channels, channels * 3, bias=False)
            self.proj = nn.Linear(channels, channels)

    def forward(self, pxo, knn_idx=None):
        # [N, 3], [N, C], [B]
        p, x, o = pxo
        c = x.size(1)

        if self.proj_channels is not None:
            c = self.proj_channels

        assert c % self.num_heads == 0
        head_dim = c // self.num_heads
        scale_factor = head_dim ** -0.5

        qkv = self.qkv(x)
        # [N, C]
        x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)

        # fastforks: pointops.knn_query_and_group(x, p, o, idx=knn_idx, with_xyz=False) -> x[knn_idx]
        # [N, K, C]
        x_k = x_k[knn_idx]

        # [N, K, C]
        x_v = x_v[knn_idx]

        n, k, c = x_k.shape

        # [N, 1, K]
        scores = torch.matmul(x_q.unsqueeze(1), x_k.permute(0, 2, 1)) * scale_factor
        # [N, C]
        out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)

        out = self.proj(out)

        return out


class MLP(nn.Module):
    def __init__(
        self,
        channels
    ):
        super().__init__()

        expansion = 4

        self.fc1 = nn.Linear(channels, channels * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion, channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, knn_samples=16,
        num_heads=1,
        attn_proj_channels=None,
        ):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = KNNAttention(channels, knn_samples=knn_samples,
            num_heads=num_heads,
            proj_channels=attn_proj_channels,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(self, pxo, knn_idx=None):
        p, x, o = pxo

        x = x + self.attn((p, self.norm1(x), o), knn_idx=knn_idx)
        x = x + self.mlp(self.norm2(x))

        return x


class PlainPointTransformer(nn.Module):
    # fastforks: without the local kNN, gradient checkpointing and non-lowres multi-view options (unused at eval)
    def __init__(self, channels, knn_samples=16, num_blocks=4,
        num_heads=1,
        attn_proj_channels=None,
        with_mv_attn=False,
        mvattn_down_factor=4,
        ):
        super().__init__()

        self.knn_samples = knn_samples

        self.with_mv_attn = with_mv_attn

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(TransformerBlock(channels, knn_samples=knn_samples,
                num_heads=num_heads,
                attn_proj_channels=attn_proj_channels,
                ))

        # multi-view attention
        if self.with_mv_attn:
            self.mv_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                self.mv_blocks.append(
                    MultViewLowresAttn(
                        channels,
                        down_factor=mvattn_down_factor,
                    )
                )

    def forward(self, pxo, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        # compute knn idx once and reuse across blocks (positions don't change)
        knn_idx = knn_query(self.knn_samples, p)  # fastforks: was pointops.knn_query(k, p, o, p, o)

        if self.with_mv_attn:
            assert b is not None and v is not None and h is not None and w is not None
            for i in range(len(self.mv_blocks)):
                x = self.blocks[i]([p, x, o], knn_idx=knn_idx)
                # global multi-view attention
                x = x.reshape(b, v * h * w, -1)  # fastforks: einops.rearrange "(b v h w) c -> b (v h w) c"
                x = self.mv_blocks[i](x, v=v, h=h, w=w)
                x = x.flatten(0, 1)  # fastforks: einops.rearrange "b (v h w) c -> (b v h w) c"
        else:
            for blk in self.blocks:
                x = blk((p, x, o), knn_idx=knn_idx)

        return x


def _to_maps(x, v, h, w):  # fastforks: einops.rearrange "b (v h w) c -> (b v) c h w"
    return x.unflatten(1, (v, h, w)).flatten(0, 1).permute(0, 3, 1, 2)


def _to_tokens(x, v):  # fastforks: einops.rearrange "(b v) c h w -> b (v h w) c"
    return x.unflatten(0, (-1, v)).permute(0, 1, 3, 4, 2).flatten(1, 3)


class MultViewLowresAttn(nn.Module):
    def __init__(self, channels,
        down_factor=4,
        attn_proj_channels=None,
        ):
        super().__init__()

        self.down_factor = down_factor

        self.attn_proj_channels = attn_proj_channels

        if attn_proj_channels:
            ori_channels = channels
            self.proj0 = nn.Linear(channels, attn_proj_channels)
            channels = attn_proj_channels

        if self.down_factor == 8:
            down_factor = 4
        else:
            down_factor = self.down_factor

        self.proj1 = nn.Linear(channels * down_factor ** 2, channels)
        self.norm1 = nn.LayerNorm(channels)

        self.proj2 = nn.Linear(channels, channels * down_factor ** 2)
        self.norm2 = nn.LayerNorm(channels * down_factor ** 2)

        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if attn_proj_channels:
            self.proj3 = nn.Linear(channels, ori_channels)

        num_heads = 1 if self.attn_proj_channels else 4

        if channels % 32 != 0 and channels % 16 == 0:
            # flash attention 3 head_size should be a multiple of 8
            num_heads = 2

        if channels == 48 or channels == 3:
            # ablation: rgb error instead of feature error
            num_heads = 1

        self.attn = Block(channels, num_heads, attn_class=SdpaAttention)  # fastforks: upstream MemEffAttention = SDPA

    def forward(self, x, v=None, h=None, w=None):  # fastforks: without the cross-attention variant (unused)
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)

        x = _to_maps(x, v, h, w)

        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
            down_factor = 4
        else:
            down_factor = self.down_factor

        x = F.pixel_unshuffle(x, down_factor)

        x = _to_tokens(x, v)
        x = self.proj1(x)
        x = self.norm1(x)

        x = self.attn(x)

        x = self.proj2(x)
        x = self.norm2(x)

        x = _to_maps(x, v, h // self.down_factor, w // self.down_factor)
        x = F.pixel_shuffle(x, down_factor)
        x = self.conv(x)
        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = _to_tokens(x, v)
        if self.attn_proj_channels:
            x = self.proj3(x)
        x = x + residual

        return x


class PointLinearWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, pxo, b=None, v=None, h=None, w=None):
        p, x, o = pxo
        x = self.linear(x)

        return [p, x, o]
