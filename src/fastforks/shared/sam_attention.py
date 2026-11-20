"""Relative-position attention operands for SAM and SAM-HQ."""

from __future__ import annotations

import torch


def augmented(
    qkv: torch.Tensor, nh: int, rel: tuple[torch.Tensor, torch.Tensor, torch.Tensor], hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Packed qkv `[B, N, 3C]` -> SDPA operands `q', k', v'` `[B, heads, N, Dp]` (to run with `scale=1`), head dim.

    `rel`: the relative-position tables `Rh [h, h, d]`, `Rw [w, w, d]` and `onehot_keys(h, w)`.
    """
    rh, rw, onehot = rel
    b, n, c3 = qkv.shape
    d = c3 // 3 // nh
    h, w = hw
    q, k, v = qkv.view(b, n, 3, nh, d).unbind(2)
    r_q = q.reshape(b, h, w, nh, d)
    pad = -(d + h + w) % 8
    qt = q.transpose(1, 2)
    if (
        qkv.is_cuda
        and qkv.dtype == torch.float16
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled())
    ):
        # Write each relative plane directly, so the two projections and the value padding need no shared temporary.
        shape = (b, nh, n, d + h + w + pad)
        qa = qt.new_empty(shape)
        qa[..., :d].copy_(qt * d**-0.5)
        qa[..., d : d + h].copy_(torch.einsum("bhwnc,hkc->bnhwk", r_q, rh).reshape(b, nh, n, h))
        qa[..., d + h : d + h + w].copy_(torch.einsum("bhwnc,wkc->bnhwk", r_q, rw).reshape(b, nh, n, w))
        qa[..., d + h + w :].zero_()
        ka = qt.new_empty(shape)
        ka[..., :d].copy_(k.transpose(1, 2))
        ka[..., d : d + h + w].copy_(onehot)
        ka[..., d + h + w :].zero_()
        va = qt.new_empty(shape)
        va[..., :d].copy_(v.transpose(1, 2))
        va[..., d:].zero_()
        return qa, ka, va, d
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", r_q, rh).reshape(b, nh, n, h)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", r_q, rw).reshape(b, nh, n, w)
    zeros = qt.new_zeros(b, nh, n, pad)
    qa = torch.cat([qt * d**-0.5, rel_h, rel_w, zeros], -1)
    ka = torch.cat([k.transpose(1, 2), onehot.expand(b, nh, n, h + w), zeros], -1)
    va = torch.cat([v.transpose(1, 2), qt.new_zeros(b, nh, n, h + w + pad)], -1)
    return qa, ka, va, d
