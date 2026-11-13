"""LightGlue on many pairs per forward pass: the tokens of all pairs packed along one dimension.

Upstream matches one pair per forward (about 830 launches and a host sync per layer for the adaptive depth and
width; launch bound at 2048 keypoints). Here the keypoints of `B` pairs are one packed token sequence per side
(`[T, D]`), so the linear layers, LayerNorm, GELU and the rotary encoding are per-token and run once per layer for
all pairs; the attention is variable-length flash attention over the packed sequences (`cu_seqlens`). The adaptive
depth and width keep upstream's per-pair semantics: a pair that stops at layer `i` leaves the packed set and gets its
assignment head `log_assignment[i]` there; pruning drops its tokens from the packed sequences. The assignment head
runs padded over the pairs that finish at the same layer, without forming the `(M + 1) x (N + 1)` log-assignment
matrix: the row / column maxima of `sim - lse` are the same matches.

Results agree with the per-pair forward up to float rounding (other GEMM shapes and attention tiles).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .net.lightglue import CrossBlock, SelfBlock, apply_cached_rotary_emb

if TYPE_CHECKING:
    from .net.lightglue import LightGlue

_HEAD_PAIRS = 4  # pairs per padded assignment head call (3 `[M, N]` float32 temporaries each: 48 MB at 2048)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sq: _Side, sk: _Side) -> torch.Tensor:
    """Attention of every pair's packed queries `[Tq, H, d]` over its keys / values `[Tk, H, d]` -> `[Tq, H, d]`.

    CUDA: variable-length flash attention on float16 (as upstream's SDPA on half). Elsewhere: upstream's SDPA per
    pair, float32.
    """
    if q.device.type == "cuda":
        q, k, v = (x.half().contiguous() for x in (q, k, v))
        out = torch.ops.aten._flash_attention_forward(  # noqa: SLF001 - the varlen SDPA (no public API)
            q,
            k,
            v,
            sq.cu,
            sk.cu,
            max(sq.lens),
            max(sk.lens),
            0.0,
            False,  # noqa: FBT003 - dropout, is_causal, return_debug_mask: positional-only
            False,  # noqa: FBT003
        )
        return out[0].float()
    outs = []
    for (a, b), (c, d) in zip(sq.spans(), sk.spans(), strict=True):
        qi, ki, vi = (x.transpose(0, 1).contiguous() for x in (q[a:b], k[c:d], v[c:d]))
        outs.append(F.scaled_dot_product_attention(qi, ki, vi).transpose(0, 1))
    return torch.cat(outs)


class _Side:
    """One side of the active pairs: packed tokens, their rotary encodings, keypoint indices and pair lengths."""

    def __init__(self, desc: torch.Tensor, enc: torch.Tensor, lens: list[int]) -> None:
        self.desc = desc  # [T, D]
        self.enc = enc  # [2, T, 1, d]
        self.lens = lens  # tokens per pair
        self.ind = torch.arange(desc.shape[0], device=desc.device) - self.offsets()[self.seg()]  # keypoint index
        self.cu = self.cu_seqlens()

    def offsets(self) -> torch.Tensor:
        return torch.tensor([0, *self.lens[:-1]], device=self.desc.device).cumsum(0)

    def seg(self) -> torch.Tensor:
        """The pair (position) of every token."""
        lens = torch.tensor(self.lens, device=self.desc.device)
        return torch.arange(len(self.lens), device=self.desc.device).repeat_interleave(lens)

    def spans(self) -> list[tuple[int, int]]:
        """`(start, end)` of every pair's tokens."""
        cu = [0]
        for n in self.lens:
            cu.append(cu[-1] + n)
        return list(itertools.pairwise(cu))

    def cu_seqlens(self) -> torch.Tensor:
        cu = [0, *(end for _, end in self.spans())]
        return torch.tensor(cu, device=self.desc.device, dtype=torch.int32)

    def select(self, keep: torch.Tensor, lens: list[int]) -> None:
        """Keep the tokens `keep` (sorted indices); `lens`: the new tokens per pair."""
        self.desc = self.desc.index_select(0, keep)
        self.enc = self.enc.index_select(1, keep)
        self.ind = self.ind.index_select(0, keep)
        self.lens = lens
        self.cu = self.cu_seqlens()

    def prune(self, keep: torch.Tensor, continuing: list[int]) -> None:
        """Keep the tokens with `keep` (`[T]` bool) of the pairs `continuing` (positions, sorted), drop the others."""
        seg = self.seg()
        stays = torch.zeros(len(self.lens), dtype=torch.bool, device=self.desc.device)
        stays[continuing] = True
        keep = keep & stays[seg]
        lens = torch.bincount(seg[keep], minlength=len(self.lens))[continuing].tolist()
        self.select(keep.nonzero().squeeze(1), lens)


def _self_block(block: SelfBlock, side: _Side) -> torch.Tensor:
    qkv = block.Wqkv(side.desc).unflatten(-1, (block.num_heads, -1, 3))  # [T, H, d, 3]
    q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
    q = apply_cached_rotary_emb(side.enc, q)
    k = apply_cached_rotary_emb(side.enc, k)
    context = _attention(q, k, v, side, side)
    message = block.out_proj(context.flatten(1))
    return side.desc + block.ffn(torch.cat([side.desc, message], -1))


def _cross_block(
    block: CrossBlock, x0: torch.Tensor, x1: torch.Tensor, s0: _Side, s1: _Side
) -> tuple[torch.Tensor, torch.Tensor]:
    h = block.heads
    qk0, qk1 = block.to_qk(x0).unflatten(-1, (h, -1)), block.to_qk(x1).unflatten(-1, (h, -1))
    v0, v1 = block.to_v(x0).unflatten(-1, (h, -1)), block.to_v(x1).unflatten(-1, (h, -1))
    m0 = _attention(qk0, qk1, v1, s0, s1)
    m1 = _attention(qk1, qk0, v0, s1, s0)
    m0, m1 = block.to_out(m0.flatten(1)), block.to_out(m1.flatten(1))
    x0 = x0 + block.ffn(torch.cat([x0, m0], -1))
    x1 = x1 + block.ffn(torch.cat([x1, m1], -1))
    return x0, x1


def _posenc(net: LightGlue, kpts: torch.Tensor) -> torch.Tensor:
    """Upstream `LearnableFourierPositionalEncoding` per token: `[T, 2]` -> `[2, T, 1, d]`."""
    projected = net.posenc.Wr(kpts)
    emb = torch.stack([torch.cos(projected), torch.sin(projected)], 0)
    return emb.repeat_interleave(2, dim=-1)[:, :, None, :]


def _pad_index(offsets: list[int], lens: list[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """`[G, L]` packed-token index of the padded rows and the valid mask."""
    n = max(lens)
    ar = torch.arange(n, device=device)
    idx = torch.tensor(offsets, device=device)[:, None] + ar[None]
    mask = ar[None] < torch.tensor(lens, device=device)[:, None]
    return idx * mask, mask


Match = tuple[torch.Tensor, torch.Tensor]  # `[S, 2]` keypoint index pairs, `[S]` scores


def _head(net: LightGlue, layer: int, s0: _Side, s1: _Side, pairs: list[int]) -> list[Match]:
    """`log_assignment[layer]` + `filter_matches` of the given pairs (positions)."""
    out: list[Match] = []
    head: Any = net.log_assignment[layer]  # MatchAssignment
    d = net.conf.descriptor_dim
    off0, off1 = s0.offsets().tolist(), s1.offsets().tolist()
    for start in range(0, len(pairs), _HEAD_PAIRS):
        group = pairs[start : start + _HEAD_PAIRS]
        i0, mask0 = _pad_index([off0[b] for b in group], [s0.lens[b] for b in group], s0.desc.device)
        i1, mask1 = _pad_index([off1[b] for b in group], [s1.lens[b] for b in group], s1.desc.device)
        desc0, desc1 = s0.desc[i0], s1.desc[i1]  # [G, M, D], [G, N, D]
        mdesc0, mdesc1 = head.final_proj(desc0) / d**0.25, head.final_proj(desc1) / d**0.25
        sim = torch.bmm(mdesc0, mdesc1.transpose(1, 2))
        valid = mask0[:, :, None] & mask1[:, None, :]
        sim.masked_fill_(~valid, -float("inf"))
        lz0 = F.logsigmoid(head.matchability(desc0).squeeze(-1))  # [G, M]
        lz1 = F.logsigmoid(head.matchability(desc1).squeeze(-1))  # [G, N]
        lse0 = torch.logsumexp(sim, 2).masked_fill_(~mask0, 0)  # rows
        lse1 = torch.logsumexp(sim, 1).masked_fill_(~mask1, 0)  # columns
        # upstream `sigmoid_log_double_softmax`: scores[i, j] = 2 sim - lse0[i] - lse1[j] + lz0[i] + lz1[j] (both
        # log-softmaxes contain sim); the row argmax ignores the terms in i and vice versa
        sim.mul_(2)
        max0 = (sim - lse1[:, None, :] + lz1[:, None, :]).max(2)
        m0 = max0.indices
        m1 = (sim - lse0[:, :, None] + lz0[:, :, None]).max(1).indices
        del sim
        mscores0 = (max0.values - lse0 + lz0).exp()
        mutual0 = torch.arange(m0.shape[1], device=m0.device)[None] == m1.gather(1, m0)
        valid0 = mutual0 & (mscores0 > net.conf.filter_threshold) & mask0
        nz = valid0.nonzero()
        g, i = nz[:, 0], nz[:, 1]
        j = m0[g, i]
        counts = torch.bincount(g, minlength=len(group)).tolist()
        kp0 = s0.ind[i0[g, i]]
        kp1 = s1.ind[i1[g, j]]
        matches = torch.stack([kp0, kp1], -1)
        scores = mscores0[g, i]
        for k, c in enumerate(counts):
            a = sum(counts[:k])
            out.append((matches[a : a + c], scores[a : a + c]))
    return out


Features = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]  # upstream's `image0`, `image1` dicts


def match_many(net: LightGlue, features: list[Features]) -> list[dict[str, torch.Tensor]]:
    """Upstream's forward on many pairs at once: a dict with `matches` `[S, 2]` and `scores` `[S]` per pair.

    `features`: `(image0, image1)` feature dicts (`keypoints` `[1, M, 2]`, `descriptors` `[1, M, C]`, `image_size`
    `[1, 2]`) per pair, as upstream's `data`. Adaptive depth (`depth_confidence`) and width (`width_confidence`)
    decided per pair as upstream.
    """
    device = features[0][0]["keypoints"].device
    empty = (torch.empty((0, 2), dtype=torch.long, device=device), torch.empty(0, device=device))
    results: list[Match] = [empty] * len(features)
    active = [p for p, (f0, f1) in enumerate(features) if f0["keypoints"].shape[1] and f1["keypoints"].shape[1]]
    if active:
        _layers(net, features, active, results, empty)
    return [{"matches": m, "scores": s} for m, s in results]


class _Batch:
    """The active pairs: both packed sides, the pairs' positions in `features` and their original keypoint counts."""

    def __init__(self, net: LightGlue, features: list[Features], active: list[int]) -> None:
        self.net = net
        self.active = active
        self.s0, self.s1 = (self._side(features, k) for k in (0, 1))
        self.num_points = [m + n for m, n in zip(self.s0.lens, self.s1.lens, strict=True)]

    def _side(self, features: list[Features], k: int) -> _Side:
        feats = [features[p][k] for p in self.active]
        lens = [f["keypoints"].shape[1] for f in feats]
        kpts = torch.cat([f["keypoints"][0] for f in feats])
        size = torch.cat([f["image_size"].expand(n, 2) for f, n in zip(feats, lens, strict=True)]).to(kpts)
        kpts = (kpts - size / 2) / (size.max(-1).values / 2)[:, None]  # upstream `normalize_keypoints`
        desc = self.net.input_proj(torch.cat([f["descriptors"][0] for f in feats]).contiguous())
        return _Side(desc, _posenc(self.net, kpts), lens)

    def layer(self, i: int) -> None:
        layer: Any = self.net.transformers[i]  # TransformerLayer
        x0 = _self_block(layer.self_attn, self.s0)
        x1 = _self_block(layer.self_attn, self.s1)
        self.s0.desc, self.s1.desc = _cross_block(layer.cross_attn, x0, x1, self.s0, self.s1)

    def stopped(self, i: int, token0: torch.Tensor, token1: torch.Tensor) -> list[int]:
        """Upstream `check_if_stop` per pair: the positions of the pairs that stop after layer `i`."""
        thresholds: Any = self.net.confidence_thresholds  # buffer [n_layers]
        cth: torch.Tensor = thresholds[i]
        low = torch.zeros(len(self.active), device=cth.device)
        low.index_add_(0, self.s0.seg(), (token0 < cth).float())
        low.index_add_(0, self.s1.seg(), (token1 < cth).float())
        ratio = 1.0 - low / torch.tensor(self.num_points, device=cth.device, dtype=torch.float32)
        return (ratio > self.net.conf.depth_confidence).nonzero().squeeze(1).tolist()

    def head(self, i: int, pairs: list[int], results: list[Match]) -> None:
        for p, r in zip(
            [self.active[k] for k in pairs],
            _head(self.net, i, self.s0, self.s1, pairs),
            strict=True,
        ):
            results[p] = r

    def keep(self, i: int, continuing: list[int], tokens: tuple[torch.Tensor | None, torch.Tensor | None]) -> None:
        """Drop the pairs not in `continuing`; prune the tokens of the others as upstream (`get_pruning_mask`)."""
        net = self.net
        pruning_th = net.pruning_min_kpts(self.s0.desc.device)
        for side, token in zip((self.s0, self.s1), tokens, strict=True):
            keep = torch.ones(side.desc.shape[0], dtype=torch.bool, device=side.desc.device)
            if net.conf.width_confidence > 0 and any(side.lens[k] > pruning_th for k in continuing):
                head: Any = net.log_assignment[i]  # MatchAssignment
                keep = net.get_pruning_mask(token, head.get_matchability(side.desc), i)  # ty: ignore[invalid-argument-type] - upstream accepts None
                big = torch.tensor([n > pruning_th for n in side.lens], device=keep.device)
                keep |= ~big[side.seg()]
            side.prune(keep, continuing)
        self.active = [self.active[k] for k in continuing]
        self.num_points = [self.num_points[k] for k in continuing]


def _layers(net: LightGlue, features: list[Features], active: list[int], results: list[Match], empty: Match) -> None:
    """The transformer layers with per-pair early stop and pruning; the head result of every pair into `results`."""
    conf = net.conf
    batch = _Batch(net, features, active)
    for i in range(conf.n_layers):
        batch.layer(i)
        b = len(batch.active)
        if i == conf.n_layers - 1:  # no early stopping or adaptive width at the last layer
            batch.head(i, list(range(b)), results)
            break
        tokens: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        done: list[int] = []
        if conf.depth_confidence > 0:
            tokens = net.token_confidence[i](batch.s0.desc, batch.s1.desc)
            done = batch.stopped(i, *tokens)
            batch.head(i, done, results)
        if len(done) == b:
            break
        if done or conf.width_confidence > 0:
            batch.keep(i, [k for k in range(b) if k not in set(done)], tokens)
        if 0 in batch.s0.lens + batch.s1.lens:  # a side pruned to nothing: no matches (upstream breaks)
            continuing = [k for k, (m, n) in enumerate(zip(batch.s0.lens, batch.s1.lens, strict=True)) if m and n]
            for k in set(range(len(batch.active))) - set(continuing):
                results[batch.active[k]] = empty
            if not continuing:
                break
            batch.keep(i, continuing, (None, None))
