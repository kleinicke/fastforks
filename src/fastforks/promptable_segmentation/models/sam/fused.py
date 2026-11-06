"""Kernels-mode SAM image encoder on the `dpt_fused` kernels (GEMMs moved to fused-epilogue kernels).

Same maths as upstream `ImageEncoderViT.forward`, fp16 weights and activations:

- patch embedding: the 16 x 16 stride-16 conv as one GEMM over the patch rows (fp32 accumulation, K = 768), the
  epilogue adds the bias and the position embedding
- blocks: qkv + bias, proj + bias (+ the residual for global blocks; windowed blocks add it after un-partitioning),
  lin1 + bias + GELU (erf), lin2 + bias + residual as fused GEMMs with fp16 accumulation; LayerNorm in PyTorch (fp32
  statistics); attention with the decomposed relative-position bias folded into an augmented q / k (`fast.augmented`)
  through SDPA (flash). The input is always padded to the square `image_size`, so every geometry is fixed: the
  relative-position tables and the one-hot key columns are built once.
- neck: 1 x 1 conv as a GEMM, LayerNorm, 3 x 3 conv as an NHWC implicit-GEMM conv, LayerNorm

Anything else (another block layout) raises `BackendUnavailableError` (no PyTorch fallback; `auto` uses portable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks.kernels.dpt_fused import ACT_GELU

from .fast import augmented, rel_tables, window_partition, window_unpartition

if TYPE_CHECKING:
    from fastforks.kernels.dpt_fused import DptFused

_Ln = tuple[torch.Tensor, torch.Tensor, float]


def _half(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.detach().half().contiguous().to(device)


def _ln(m: Any, device: torch.device) -> _Ln:  # noqa: ANN401 - upstream LayerNorm / LayerNorm2d
    return _half(m.weight, device), _half(m.bias, device), float(m.eps)


class _Block:
    def __init__(self, blk: Any, tokens: int, device: torch.device) -> None:  # noqa: ANN401 - upstream Block
        a = blk.attn
        if not (a.use_rel_pos and a.qkv.bias is not None and isinstance(blk.mlp.act, torch.nn.GELU)):
            msg = "dpt_fused SAM: expected relative positions, a qkv bias and a GELU MLP"
            raise BackendUnavailableError(msg)  # mode="auto" falls back to portable
        self.ws = int(blk.window_size)
        self.heads = int(a.num_heads)
        self.n1, self.n2 = _ln(blk.norm1, device), _ln(blk.norm2, device)
        self.qkv = (_half(a.qkv.weight, device), _half(a.qkv.bias, device))
        self.proj = (_half(a.proj.weight, device), _half(a.proj.bias, device))
        self.lin1 = (_half(blk.mlp.lin1.weight, device), _half(blk.mlp.lin1.bias, device))
        self.lin2 = (_half(blk.mlp.lin2.weight, device), _half(blk.mlp.lin2.bias, device))
        rh, rw, onehot = rel_tables(blk, tokens, torch.float16)
        self.rel = (rh.to(device), rw.to(device), onehot.to(device))


class FusedEncoder:
    """SAM `ImageEncoderViT` on dpt_fused: `[1, 3, S, S]` normalised float32 -> float32 `[1, 256, S/16, S/16]`."""

    def __init__(self, enc: Any, device: torch.device, k: DptFused) -> None:  # noqa: ANN401 - upstream ImageEncoderViT
        """Check and convert the encoder (on the CPU), move it to `device`."""
        conv = enc.patch_embed.proj
        ps = conv.kernel_size[0]
        if conv.kernel_size != (ps, ps) or conv.stride != (ps, ps) or enc.pos_embed is None:
            msg = "dpt_fused SAM: patch embedding must be a stride == kernel conv with a position embedding"
            raise BackendUnavailableError(msg)  # mode="auto" falls back to portable
        self.k, self.ps, self.dim = k, ps, int(enc.pos_embed.shape[-1])
        self.tokens = int(enc.pos_embed.shape[1])
        self.patch = (_half(conv.weight.flatten(1), device), _half(conv.bias, device))
        self.pos = _half(enc.pos_embed.flatten(0, 2), device)
        self.blocks = [_Block(b, self.tokens, device) for b in enc.blocks]
        n = enc.neck
        self.neck1 = _half(n[0].weight.flatten(1), device)
        self.neck_ln1, self.neck_ln2 = _ln(n[1], device), _ln(n[3], device)
        self.neck3 = _half(n[2].weight.permute(0, 2, 3, 1), device)  # [K, R, S, C]

    def _attention(self, b: _Block, x: torch.Tensor) -> torch.Tensor:
        """`[B, h, w, C]` (normed, partitioned) -> attention output before proj, rows `[B * h * w, C]`."""
        bw, h, w, c = x.shape
        qkv = self.k.gemm(x.view(-1, c), *b.qkv).view(bw, h * w, 3 * c)
        qa, ka, va, d = augmented(qkv, b.heads, b.rel, (h, w))
        o = F.scaled_dot_product_attention(qa, ka, va, scale=1.0)[..., :d]
        return o.transpose(1, 2).reshape(-1, c)

    def _block(self, b: _Block, x: torch.Tensor) -> torch.Tensor:
        k, t, c = self.k, self.tokens, self.dim
        rows = x.view(-1, c)
        y = F.layer_norm(x, (c,), *b.n1[:2], b.n1[2])
        if b.ws:
            y, pad_hw = window_partition(y, b.ws)
            o = k.gemm(self._attention(b, y), *b.proj).view(-1, b.ws, b.ws, c)
            x = x + window_unpartition(o, b.ws, pad_hw, (t, t))
            rows = x.view(-1, c)
        else:
            rows = k.gemm(self._attention(b, y), *b.proj, res=rows)
        hid = k.gemm(F.layer_norm(rows, (c,), *b.n2[:2], b.n2[2]), *b.lin1, act=ACT_GELU)
        return k.gemm(hid, *b.lin2, res=rows, out=rows).view(1, t, t, c)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one padded, normalised image."""
        k, t, ps, c = self.k, self.tokens, self.ps, self.dim
        if x.shape != (1, 3, t * ps, t * ps):
            msg = f"dpt_fused SAM: expected a [1, 3, {t * ps}, {t * ps}] input, got {tuple(x.shape)}"
            raise ValueError(msg)
        with torch.autocast("cuda", enabled=False):
            p = x.half().view(3, t, ps, t, ps).permute(1, 3, 0, 2, 4).reshape(t * t, 3 * ps * ps)
            h = k.gemm(p, *self.patch, res=self.pos, acc32=True).view(1, t, t, c)
            for b in self.blocks:
                h = self._block(b, h)
            y = k.gemm(h.view(-1, c), self.neck1)
            y = F.layer_norm(y, (y.shape[-1],), *self.neck_ln1[:2], self.neck_ln1[2]).view(1, t, t, -1)
            y = k.conv(y.contiguous(), self.neck3, padding=1)
            y = F.layer_norm(y, (y.shape[-1],), *self.neck_ln2[:2], self.neck_ln2[2])
        return y.float().permute(0, 3, 1, 2).contiguous()
