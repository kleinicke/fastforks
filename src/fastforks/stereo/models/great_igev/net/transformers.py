"""GREAT attention modules (upstream `models/great_stereo/{transformers,positions,attentions}.py`), inference forms.

Same module tree and parameter names as upstream (the checkpoints load strictly); the forwards are rewritten for
inference with the same math:
  - token layouts are permuted views of NHWC tensors (LayerNorm / attention over the contiguous channel axis), no
    einops,
  - attention through `F.scaled_dot_product_attention`; upstream's "sink competition" (softmax + eps, renormalised)
    is `(softmax(s) @ v + eps * sum(v)) / (1 + n * eps)`,
  - outlook attention in pixel chunks (upstream materialises k^4 = 2401 logits per 1/4-resolution pixel at once),
    keeping upstream's flat reshape of the NCHW logits (logits of "pixel" n are the flat elements n*k^4 ...),
  - window attention: the windows are the SDPA heads; with shift only the edge windows get a (boolean) mask,
  - outlook unfold / fold as shifted slices (exportable), volume attention reassociated (no D x C q tensor).
Only the configurations GREAT-IGEV uses are implemented (one head, stride 1, 2D windows, `receptive_aug="conv"`).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

OUTLOOK_CHUNK = 2**24  # outlook logits per chunk (elements)


def tokens(x):
    """[B, C, H, W] (any memory format) -> [B, H, W, C] contiguous (a view for channels-last input)."""
    return x.permute(0, 2, 3, 1).contiguous()


def planes(t):
    """[B, H, W, C] -> [B, C, H, W] view (channels-last memory)."""
    return t.permute(0, 3, 1, 2)


def layer_norm_2d(norm, x):
    """nn.LayerNorm over the channels of [B, C, H, W]; returns a channels-last [B, C, H, W]."""
    return planes(norm(tokens(x)))


def sink_attention(q, k, v, scale, eps, mask=None):
    """softmax(q k^T * scale [masked]) + eps, renormalised, @ v; q/k/v [..., N, C].

    Upstream adds eps to every weight, masked ones included: eps * sum over all keys either way.
    """
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    vs = v.float().sum(-2, keepdim=True)
    return ((o.float() + eps * vs) / (1 + k.shape[-2] * eps)).to(q.dtype)


class PositionalEmbeddingCosine2D(nn.Module):
    """DETR-style 2D sine / cosine embedding, normalised coordinates (`normalize=True`)."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, feats):
        """[1, C, H, W] fp32 embedding for `feats` [B, C, H, W] (broadcasts over the batch)."""
        _, _, h, w = feats.shape
        dev = feats.device
        y = torch.arange(1, h + 1, device=dev, dtype=torch.float32).view(1, h, 1).expand(1, h, w)
        x = torch.arange(1, w + 1, device=dev, dtype=torch.float32).view(1, 1, w).expand(1, h, w)
        if self.normalize:
            y = y / (h + 1e-6) * self.scale
            x = x / (w + 1e-6) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=dev)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)
        px = x[..., None] / dim_t
        py = y[..., None] / dim_t
        px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=4).flatten(3)
        py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((py, px), dim=3).permute(0, 3, 1, 2)


def shift_regions(height, width, window_height, window_width, shift_height, shift_width, device):
    """Region id map [H, W] of upstream's shifted-window mask (tokens of different regions do not attend)."""

    def axis(n, win, shift):
        i = torch.arange(n, device=device)
        return (i >= n - win).long() + (i >= n - shift).long()

    return axis(height, window_height, shift_height)[:, None] * 3 + axis(width, window_width, shift_width)[None, :]


class FeedForwardLayer(nn.Module):
    """LayerNorm -> 1x1 expand -> SiLU -> x + SiLU(1x1(cat(dw3, dw5, dw7))) -> 1x1 (`receptive_aug="conv"`)."""

    def __init__(self, in_channels, out_channels, expansion=2, receptive_aug="conv"):
        super().__init__()
        if receptive_aug != "conv":
            raise NotImplementedError(receptive_aug)
        e = expansion * in_channels
        self.norm = nn.LayerNorm(in_channels)
        self.in_conv = nn.Conv2d(in_channels, e, kernel_size=1)
        self.act = nn.SiLU()
        self.dw_conv_3 = nn.Conv2d(e, e, kernel_size=3, stride=1, padding=1, groups=e)
        self.dw_conv_5 = nn.Conv2d(e, e, kernel_size=5, stride=1, padding=2, groups=e)
        self.dw_conv_7 = nn.Conv2d(e, e, kernel_size=7, stride=1, padding=3, groups=e)
        self.pointwise_conv = nn.Sequential(nn.Conv2d(3 * e, e, kernel_size=1), nn.SiLU())
        self.out_conv = nn.Conv2d(e, out_channels, kernel_size=1)

    def forward(self, x):
        y = F.silu(self.in_conv(layer_norm_2d(self.norm, x)))
        z = torch.cat([self.dw_conv_3(y), self.dw_conv_5(y), self.dw_conv_7(y)], dim=1)
        y = y + self.pointwise_conv(z)
        del z
        return self.out_conv(y)


class AttentionLayer(nn.Module):
    """Cross attention along image rows (full width), pre-norm, one head, sink competition."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, eps=1e-6):
        super().__init__()
        if num_heads != 1:
            raise NotImplementedError("num_heads != 1")
        self.eps = eps
        self.scale = hidden_channels**-0.5
        self.norm = nn.LayerNorm(in_channels, eps=eps)
        self.norm_context = nn.LayerNorm(in_channels, eps=eps)
        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.to_kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        b, _, h, w = x1.shape
        q = tokens(self.to_q(layer_norm_2d(self.norm, x1)))
        kv = tokens(self.to_kv(layer_norm_2d(self.norm_context, x2)))
        c = q.shape[-1]
        k, v = kv[..., :c].reshape(b * h, 1, w, c), kv[..., c:].reshape(b * h, 1, w, c)
        o = sink_attention(q.view(b * h, 1, w, c), k, v, self.scale, self.eps)
        return self.to_out(planes(o.view(b, h, w, c)))


class TransformerBlock(nn.Module):
    """Bidirectional cross attention (left <- right and right <- left as one batch) + feed-forward."""

    def __init__(self, channels, num_heads, receptive_aug="conv"):
        super().__init__()
        self.cross_attn = AttentionLayer(channels, channels, channels, num_heads=num_heads)
        self.feedforward = FeedForwardLayer(channels, channels, receptive_aug=receptive_aug)

    def forward(self, x1, x2):
        a = torch.cat([x1, x2], dim=0)
        a = a + self.cross_attn(a, torch.cat([x2, x1], dim=0))
        a = a + self.feedforward(a)
        return a.chunk(2, dim=0)


class TransformerFeatureRefiner(nn.Module):
    def __init__(self, channels, num_heads, depth, receptive_aug="conv"):
        super().__init__()
        self.depth = depth
        for i in range(depth):
            setattr(self, f"refinement_{i + 1}", TransformerBlock(channels, num_heads, receptive_aug))

    def forward(self, x1, x2):
        for i in range(self.depth):
            x1, x2 = getattr(self, f"refinement_{i + 1}")(x1, x2)
        return x1, x2


class OutlookAttentionLayer(nn.Module):
    """VOLO-style outlook attention (stride 1, one head), pre-norm, sink competition."""

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, padding=1, stride=1, eps=1e-6):
        super().__init__()
        if stride != 1:
            raise NotImplementedError("stride != 1")
        self.eps = eps
        self.kernel_size, self.padding = kernel_size, padding
        self.scale = hidden_channels**-0.5
        self.norm = nn.LayerNorm(in_channels, eps=eps)
        self.to_v = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.to_attn = nn.Conv2d(in_channels, kernel_size**4, kernel_size=1)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        k, kk, n, pad = self.kernel_size, self.kernel_size**2, h * w, self.padding
        x = layer_norm_2d(self.norm, x)
        # value windows [B, N, k*k, C] (upstream: unfold), window position t = i * k + j reads pixel (y + i, x + j)
        vp = F.pad(self.to_v(x), (pad, pad, pad, pad)).permute(0, 2, 3, 1)
        v = torch.stack([vp[:, i : i + h, j : j + w] for i in range(k) for j in range(k)], dim=3).view(b, n, kk, c)
        vp = None  # frees the padded copy; not `del`: the comprehension's cell (Python 3.11) breaks torch 2.8's Dynamo
        # logits: upstream reshapes the NCHW [B, k^4, H, W] conv output flat to [B, N, k*k, k*k]; "pixel" m reads
        # flat elements [m * k^4, (m + 1) * k^4), i.e. output channel f // N at pixel f % N
        a = self.to_attn
        wt = a.weight[: kk * kk].reshape(kk * kk, c)  # [: k^4]: the max path pads output channels to 8
        xf = x.contiguous().view(b, c, n)
        out = torch.empty((b, n, kk, c), device=x.device, dtype=x.dtype)
        step = max(1, OUTLOOK_CHUNK // (kk * kk))
        for m0 in range(0, n, step):
            m1 = min(n, m0 + step)
            f0, f1 = m0 * kk * kk, m1 * kk * kk
            c0, c1 = f0 // n, (f1 - 1) // n + 1
            lg = torch.matmul(wt[c0:c1], xf) + a.bias[c0:c1, None]  # [B, c1-c0, N]
            lg = lg.reshape(b, -1)[:, f0 - c0 * n : f1 - c0 * n].reshape(b, m1 - m0, kk, kk).float()
            if torch.compiler.is_compiling() and not torch.compiler.is_exporting():
                # fastforks: Inductor lowers softmax / amax of x * scale with a per-row any(non-finite) check, which
                # on these 49-wide rows crashes Triton's LLVM ("Cannot BITCAST between types of different sizes");
                # exp(scale * (x - max x)) is the same softmax (scale > 0) without that pattern
                p = ((lg - lg.amax(-1, keepdim=True)) * self.scale).exp_()
                p = p.div_(p.sum(-1, keepdim=True)).add_(self.eps)
            else:
                p = torch.softmax(lg * self.scale, dim=-1).add_(self.eps)
            p = p.div_(p.sum(-1, keepdim=True))
            out[:, m0:m1] = torch.matmul(p.to(v.dtype), v[:, m0:m1])
            del lg, p
        del v
        # fold (upstream: F.fold): window position t of pixel (y, x) adds to pixel (y + i - pad, x + j - pad)
        out = out.view(b, h, w, kk, c)
        acc = torch.zeros((b, h + 2 * pad, w + 2 * pad, c), device=x.device, dtype=torch.float32)
        for i in range(k):
            for j in range(k):
                acc[:, i : i + h, j : j + w] += out[:, :, :, i * k + j]
        del out
        o = acc[:, pad : pad + h, pad : pad + w].to(x.dtype)
        return self.to_out(planes(o))


class WindowAttentionLayer(nn.Module):
    """Swin-style 2D window self / cross attention (one head), pre-norm, sink competition."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, eps=1e-6):
        super().__init__()
        if num_heads != 1:
            raise NotImplementedError("num_heads != 1")
        self.eps = eps
        self.scale = hidden_channels**-0.5
        self.norm = nn.LayerNorm(in_channels, eps=eps)
        self.norm_context = nn.LayerNorm(in_channels, eps=eps)
        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.to_kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2, num_splits, with_shift):
        b, _, h, w = x1.shape
        s = num_splits
        wh, ww = h // s, w // s
        q = tokens(self.to_q(layer_norm_2d(self.norm, x1)))
        kv = tokens(self.to_kv(layer_norm_2d(self.norm_context, x2)))
        c = q.shape[-1]
        k, v = kv[..., :c], kv[..., c:]
        sh, sw = wh // 2, ww // 2
        if with_shift:
            q, k, v = (torch.roll(t, shifts=(-sh, -sw), dims=(1, 2)) for t in (q, k, v))

        def win(t):  # [B, H, W, C'] -> [B, s*s, wh*ww, C'] (window index i * s + j, as upstream's split_feature)
            return t.reshape(b, s, wh, s, ww, -1).permute(0, 1, 3, 2, 4, 5).reshape(b, s * s, wh * ww, t.shape[-1])

        q, k, v = win(q), win(k), win(v)
        o = sink_attention(q, k, v, self.scale, self.eps)
        if with_shift:
            # only the last window row / column straddles the shift regions (upstream: -100 on cross-region logits,
            # weight ~e-44): recompute those windows with a boolean mask, the others need none
            ids = win(shift_regions(h, w, wh, ww, sh, sw, x1.device).view(1, h, w, 1))[0, :, :, 0]
            # last column, then last row (aranges, no boolean index: no host sync, CUDA-graph capturable)
            dev = x1.device
            edge = torch.cat([torch.arange(s - 1, s * s - 1, s, device=dev), torch.arange(s * s - s, s * s, device=dev)])
            ie = ids[edge]
            mask = (ie[:, :, None] == ie[:, None, :]).unsqueeze(0)
            o[:, edge] = sink_attention(q[:, edge], k[:, edge], v[:, edge], self.scale, self.eps, mask)
        o = o.view(b, s, s, wh, ww, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
        if with_shift:
            o = torch.roll(o, shifts=(sh, sw), dims=(1, 2))
        return self.to_out(planes(o.contiguous()))


VOL_CHUNK = 2**25  # volume tokens x channels per chunk (volume attention / softmax over disparity)


def d_chunks(d, per_plane, budget):
    step = max(1, budget // max(1, per_plane))
    return [(d0, min(d, d0 + step)) for d0 in range(0, d, step)]


def volume_softmax(t, g, eps, ln_eps=None):
    """softmax over D of the per-token logits `t . g` (+ eps, renormalised), fp32 [B, D, H, W].

    t [B, D, H, W, C] tokens, g [B, H, W, C] fp32 per-pixel weights; with `ln_eps` the tokens are layer-normalised
    (no affine) first.
    """
    b, d, h, w, c = t.shape
    s = torch.empty((b, d, h, w), device=t.device, dtype=torch.float32)
    for d0, d1 in d_chunks(d, h * w * c, VOL_CHUNK):
        x = t[:, d0:d1].float()
        if ln_eps is not None:
            x = F.layer_norm(x, (c,), eps=ln_eps)
        s[:, d0:d1] = (x * g.unsqueeze(1)).sum(-1)
        del x
    p = torch.softmax(s, dim=1).add_(eps)
    return p.div_(p.sum(1, keepdim=True))


class VolumeAttentionLayer(nn.Module):
    """Per pixel: the D volume tokens attend to the one feature token (softmax over D), pre-norm, sink competition.

    Parameters only; `VolumeTransformerBlock.forward` evaluates it reassociated.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, eps=1e-6):
        super().__init__()
        if num_heads != 1:
            raise NotImplementedError("num_heads != 1")
        self.eps = eps
        self.scale = hidden_channels**-0.5
        self.norm = nn.LayerNorm(in_channels, eps=eps)
        self.norm_context = nn.LayerNorm(in_channels, eps=eps)
        self.to_q = nn.Linear(in_channels, hidden_channels)
        self.to_kv = nn.Linear(in_channels, hidden_channels * 2)
        self.to_out = nn.Linear(hidden_channels, out_channels)


class VolumeTransformerBlock(nn.Module):
    def __init__(self, cv_channels, num_heads, expansion=2):
        super().__init__()
        self.volume_attn = VolumeAttentionLayer(cv_channels, cv_channels, cv_channels, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.LayerNorm(cv_channels),
            nn.Linear(cv_channels, expansion * cv_channels),
            nn.SiLU(),
            nn.Linear(expansion * cv_channels, cv_channels),
        )

    def forward(self, volume, feats):
        """volume [B, C, D, H, W] -> same (channels-last-3d; a channels-last-3d input is updated in place).

        Volume attention reassociated: logits LayerNorm(vol_d) . (scale * gamma * W_q^T k) (the bias / beta terms are
        constant over D and cancel in the softmax), update p_d * (W_o v) + b_o; in fp32, disparity chunks.
        """
        t = volume.permute(0, 2, 3, 4, 1)
        if not t.is_contiguous():
            t = t.contiguous()
        a = self.volume_attn
        c = t.shape[-1]
        kv = a.to_kv(a.norm_context(tokens(feats))).float()
        k, v = kv[..., :c], kv[..., c:]
        g = (k @ a.to_q.weight.float()) * (a.scale * a.norm.weight.float())
        vo = v @ a.to_out.weight.float().t()
        del kv, k, v
        p = volume_softmax(t, g, a.eps, ln_eps=a.norm.eps)
        del g
        bo = a.to_out.bias.float()
        _, d, h, w, _ = t.shape
        for d0, d1 in d_chunks(d, h * w * c, VOL_CHUNK):
            tc = t[:, d0:d1]
            tc.copy_(tc.float().add_(p[:, d0:d1].unsqueeze(-1) * vo.unsqueeze(1)).add_(bo))
            tc.add_(self.feedforward(tc))
        return t.permute(0, 4, 1, 2, 3)
