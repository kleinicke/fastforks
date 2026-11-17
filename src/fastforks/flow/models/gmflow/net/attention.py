import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import split_feature, merge_splits, split_feature_1d, merge_splits_1d


def sdpa(q, k, v, attn_mask=None):
    """softmax(q k^T / sqrt(C) + mask) v as one fused kernel (fastforks: replaces upstream's matmul / softmax / matmul,
    which wrote the [L, L] scores; same maths). Value channels are padded to a multiple of 8 (2-channel flow values)."""
    # fastforks: preserve upstream's float32 operation order on CPU. Chunk
    # queries in eager mode so large images do not materialize an N x N matrix;
    # capture uses one dynamic matmul instead of unrolling a shape-dependent loop.
    if q.device.type == "cpu":
        if torch.compiler.is_compiling():
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            if attn_mask is not None:
                scores = scores + attn_mask
            return torch.matmul(torch.softmax(scores, dim=-1), v)
        step = 256
        out = []
        for start in range(0, q.size(-2), step):
            scores = torch.matmul(q[..., start:start + step, :], k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            if attn_mask is not None:
                mask = attn_mask if attn_mask.size(-2) == 1 else attn_mask[..., start:start + step, :]
                scores = scores + mask
            out.append(torch.matmul(torch.softmax(scores, dim=-1), v))
        return torch.cat(out, dim=-2)
    pad = -v.size(-1) % 8
    if pad:
        v = F.pad(v, (0, pad))
    if attn_mask is not None:
        attn_mask = attn_mask.to(q.dtype)
    # ONNX requires an explicit attention head dimension for rank-three inputs.
    export = torch.compiler.is_exporting() and q.dim() == 3
    if export:
        q, k, v = (t.unsqueeze(1) for t in (q, k, v))
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
    # contiguous: a transposed token layout (last stride != 1) drops the fused kernel to the math path
    out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), attn_mask=attn_mask)
    if export:
        out = out.squeeze(1)
    return out[..., :-pad] if pad else out


def single_head_full_attention(q, k, v):
    assert q.dim() == k.dim() == v.dim() == 3

    return sdpa(q, k, v)  # [B, L, C]


def single_head_full_attention_1d(q, k, v,
                                  h=None,
                                  w=None,
                                  ):
    # q, k, v: [B, L, C]

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale_factor  # [B, H, W, W]

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v).view(b, -1, c)  # [B, H*W, C]

    return out


def single_head_split_window_attention(q, k, v,
                                       num_splits=1,
                                       with_shift=False,
                                       h=None,
                                       w=None,
                                       attn_mask=None,
                                       attention=sdpa,  # fastforks: per-runner CUDA attention layout
                                       ):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)

    mask = attn_mask.repeat(b, 1, 1) if with_shift else None  # [B*K*K, H/K*W/K, H/K*W/K]
    out = attention(q.view(b_new, -1, c), k.view(b_new, -1, c), v.view(b_new, -1, c), mask)  # [B*K*K, H/K*W/K, C]

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


def single_head_split_window_attention_1d(q, k, v,
                                          relative_position_bias=None,
                                          num_splits=1,
                                          with_shift=False,
                                          h=None,
                                          w=None,
                                          attn_mask=None,
                                          ):
    # q, k, v: [B, L, C]

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * h

    window_size_w = w // num_splits

    q = q.view(b * h, w, c)  # [B*H, W, C]
    k = k.view(b * h, w, c)
    v = v.view(b * h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=-shift_size_w, dims=1)
        k = torch.roll(k, shifts=-shift_size_w, dims=1)
        v = torch.roll(v, shifts=-shift_size_w, dims=1)

    q = split_feature_1d(q, num_splits=num_splits)  # [B*H*K, W/K, C]
    k = split_feature_1d(k, num_splits=num_splits)
    v = split_feature_1d(v, num_splits=num_splits)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  # [B*H*K, W/K, W/K]

    if with_shift:
        # attn_mask: [K, W/K, W/K]
        scores += attn_mask.repeat(b * h, 1, 1)  # [B*H*K, W/K, W/K]

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*H*K, W/K, C]

    out = merge_splits_1d(out, h, num_splits=num_splits)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=shift_size_w, dims=2)

    out = out.view(b, -1, c)

    return out


class SelfAttnPropagation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        out = sdpa(query, key, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        assert flow.size(1) == 2 or flow.size(1) == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, value_channel, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, value_channel)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, value_channel
                                                   ).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out
