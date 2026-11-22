"""Relative-position attention for SAM, SAM-HQ and ViTMatte (decomposed bias `rel_h[q, kh] + rel_w[q, kw]`).

Two exact forms of upstream's `attn = (q * scale) @ k^T + rel_h + rel_w`:

- `augmented` (default): the bias folded into a dot product of head dim `d + h + w` (padded to a multiple of 8), so
  flash never materialises the N x N matrix (SAM global attention: 64 + 64 + 64 = 192).
- `biased`: SDPA at head dim `d` with the bias as a dense float mask, built in query-row / head chunks of at most
  `MASK_BYTES`. Used on CUDA when the SDPA policy has no flash (`shared.sdpa`: PyTorch's Windows wheels) and the
  augmented head dim exceeds cuDNN's 128: there `augmented` runs the memory-efficient backend (fp16 `[1, 12, 4096,
  192]` on an RTX 5060 Ti: 10.8 ms) while the masked call runs cuDNN (3.1 ms; Linux flash on the augmented form:
  3.4 ms). The mask is `rel_h + rel_w` in the activation dtype, as upstream adds it.

`use_biased` makes the choice (logged once per process); `attend` applies it.
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

from fastforks.shared import sdpa

CUDNN_MAX_HEAD_DIM = 128  # fixed: PyTorch's cuDNN SDPA head-dim limit
MASK_BYTES = 128 * 1024**2  # memory bound of one dense-mask chunk; the result does not depend on it

Rel = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def augmented_dim(d: int, hw: tuple[int, int]) -> int:
    """Head dim of the `augmented` operands: `d + h + w`, padded to a multiple of 8."""
    n = d + hw[0] + hw[1]
    return n + -n % 8


def use_biased(qkv: torch.Tensor, d: int, hw: tuple[int, int]) -> bool:
    """Whether this call runs `biased` rather than `augmented` (see the module docstring)."""
    if not qkv.is_cuda or torch.compiler.is_compiling() or torch.jit.is_tracing() or torch.is_grad_enabled():
        return False
    return needs_bias(d, hw)


def needs_bias(d: int, hw: tuple[int, int]) -> bool:
    """The rule of `use_biased` for a CUDA call: the SDPA policy excludes flash and the augmented head dim > 128."""
    chosen = sdpa.backends()
    if chosen is None or SDPBackend.FLASH_ATTENTION in chosen or augmented_dim(d, hw) <= CUDNN_MAX_HEAD_DIM:
        return False
    _log_biased(augmented_dim(d, hw))
    return True


@functools.cache
def _log_biased(dim: int) -> None:
    logging.getLogger("fastforks").info(
        "SAM relative-position attention: the SDPA policy has no flash and the augmented head dim %d exceeds cuDNN's "
        "%d, so it runs as SDPA with a dense bias mask",
        dim,
        CUDNN_MAX_HEAD_DIM,
    )


def biased(qkv: torch.Tensor, nh: int, rel: Rel, hw: tuple[int, int]) -> torch.Tensor:
    """Packed qkv `[B, N, 3C]` -> attention output `[B, heads, N, d]`: SDPA at head dim `d` with a dense bias mask."""
    rh, rw, _ = rel
    b, n, c3 = qkv.shape
    d = c3 // 3 // nh
    h, w = hw
    q, k, v = qkv.view(b, n, 3, nh, d).unbind(2)
    r_q = q.reshape(b, h, w, nh, d)
    rel_h = torch.einsum("bhwnc,hkc->bnhwk", r_q, rh).reshape(b, nh, n, h, 1)
    rel_w = torch.einsum("bhwnc,wkc->bnhwk", r_q, rw).reshape(b, nh, n, 1, w)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = qt.new_empty(b, nh, n, d)
    per_head = n * n * qkv.element_size()
    heads = max(1, min(nh, MASK_BYTES // per_head))
    rows = n if heads > 1 else max(1, min(n, MASK_BYTES // (n * qkv.element_size())))
    for i in range(b):
        bs = slice(i, i + 1)
        for j in range(0, nh, heads):
            hs = slice(j, j + heads)
            for r in range(0, n, rows):
                rs = slice(r, r + rows)
                mask = (rel_h[bs, hs, rs] + rel_w[bs, hs, rs]).flatten(-2)
                out[bs, hs, rs] = F.scaled_dot_product_attention(
                    qt[bs, hs, rs], kt[bs, hs], vt[bs, hs], attn_mask=mask, scale=d**-0.5
                )
    return out


def attend(qkv: torch.Tensor, nh: int, rel: Rel, hw: tuple[int, int]) -> torch.Tensor:
    """Packed qkv `[B, N, 3C]` -> attention output `[B, heads, N, d]` (before the output projection)."""
    if use_biased(qkv, qkv.shape[-1] // 3 // nh, hw):
        return biased(qkv, nh, rel, hw)
    qa, ka, va, d = augmented(qkv, nh, rel, hw)
    return F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., :d]


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
    # zero columns by padding: a zeros tensor is a constant the MLX twin would fold and keep resident
    qa = F.pad(torch.cat([qt * d**-0.5, rel_h, rel_w], -1), (0, pad))
    ka = F.pad(torch.cat([k.transpose(1, 2), onehot.expand(b, nh, n, h + w)], -1), (0, pad))
    va = F.pad(v.transpose(1, 2), (0, h + w + pad))
    return qa, ka, va, d
