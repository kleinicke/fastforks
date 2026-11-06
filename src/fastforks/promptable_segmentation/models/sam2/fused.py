"""Kernels-mode SAM 2 image encoder on the `dpt_fused` kernels (GEMMs moved to fused-epilogue kernels).

Same maths as upstream `Hiera.forward` + the image path of the neck (`fast.neck`), fp16 weights and activations:

- patch embedding: the 7 x 7 stride-4 conv in PyTorch (channels-last fp16), plus the precomputed position embedding
- blocks: LayerNorm in PyTorch (fp32 statistics); qkv + bias, proj + bias (+ the residual for global blocks; windowed
  blocks add it after un-partitioning), mlp fc1 + bias + GELU (erf), fc2 + bias + residual as fused GEMMs with fp16
  accumulation; the stage-change blocks' shortcut projection as a GEMM, its 2 x 2 max pooling and the query pooling in
  PyTorch; attention through SDPA (flash)
- neck: the four 1 x 1 maps as GEMMs (bias in the epilogue), the top-down sum in float32

Anything else (another block layout) raises `BackendUnavailableError` (no PyTorch fallback; `auto` uses portable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from fastforks._core.errors import BackendUnavailableError
from fastforks.kernels.dpt_fused import ACT_GELU

from .fast import Neck, neck, pos_embed
from .net.backbone_utils import window_partition, window_unpartition

if TYPE_CHECKING:
    from fastforks.kernels.dpt_fused import DptFused

_Ln = tuple[torch.Tensor, torch.Tensor, float]
_Lin = tuple[torch.Tensor, torch.Tensor]


def _half(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.detach().half().contiguous().to(device)


def _lin(m: torch.nn.Linear, device: torch.device) -> _Lin:
    return _half(m.weight, device), _half(m.bias, device)


def _pool(x: torch.Tensor) -> torch.Tensor:
    """2 x 2 stride-2 max pooling of a channels-last `[B, H, W, C]` map (upstream `do_pool`, floor)."""
    return F.max_pool2d(x.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1)


class _Block:
    def __init__(self, blk: Any, device: torch.device) -> None:  # noqa: ANN401 - upstream MultiScaleBlock
        a, mlp = blk.attn, blk.mlp
        pooled = blk.q_stride is not None
        if (
            pooled != (blk.dim != blk.dim_out)
            or (pooled and tuple(blk.q_stride) != (2, 2))
            or len(mlp.layers) != 2  # noqa: PLR2004 - fc1, fc2
            or not isinstance(mlp.act, torch.nn.GELU)
            or mlp.act.approximate != "none"
        ):
            msg = "dpt_fused SAM 2: expected 2 x 2 query pooling exactly at the width changes and a GELU MLP"
            raise BackendUnavailableError(msg)  # mode="auto" falls back to portable
        self.ws, self.heads, self.pooled, self.dim_out = (
            int(blk.window_size),
            int(a.num_heads),
            pooled,
            int(blk.dim_out),
        )
        self.n1 = (_half(blk.norm1.weight, device), _half(blk.norm1.bias, device), float(blk.norm1.eps))
        self.n2 = (_half(blk.norm2.weight, device), _half(blk.norm2.bias, device), float(blk.norm2.eps))
        self.qkv, self.proj = _lin(a.qkv, device), _lin(a.proj, device)
        self.fc1, self.fc2 = _lin(mlp.layers[0], device), _lin(mlp.layers[1], device)
        self.short = _lin(blk.proj, device) if pooled else None


class FusedEncoder:
    """SAM 2 image encoder on dpt_fused: `[1, 3, S, S]` normalised float32 -> the `fast.neck` maps."""

    def __init__(self, image_encoder: Any, n: Neck, size: int, device: torch.device, k: DptFused) -> None:  # noqa: ANN401 - upstream ImageEncoder
        """Check and convert the encoder for `size x size` inputs (on the CPU), move it to `device`."""
        t = image_encoder.trunk
        conv = t.patch_embed.proj
        self.k, self.size = k, size
        self.pos = _half(pos_embed(t, size // int(conv.stride[0])), device)
        self.patch = conv.to(device, torch.float16).to(memory_format=torch.channels_last)
        self.stage_ends = set(t.stage_ends)
        self.blocks = [_Block(b, device) for b in t.blocks]
        self.neck = n.to(device, torch.float16)

    def _attention(self, b: _Block, y: torch.Tensor) -> torch.Tensor:
        """Normed (partitioned) `[B, h, w, C]` -> attention output before proj, `[B, h', w', C_out]` contiguous."""
        bw, h, w, c = y.shape
        co = b.dim_out
        qkv = self.k.gemm(y.reshape(-1, c), *b.qkv).view(bw, h * w, 3, b.heads, co // b.heads)
        q, kk, v = qkv.unbind(2)
        if b.pooled:
            q = _pool(q.reshape(bw, h, w, co))
            h, w = q.shape[1:3]
            q = q.reshape(bw, h * w, b.heads, co // b.heads)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), kk.transpose(1, 2), v.transpose(1, 2))
        return o.transpose(1, 2).reshape(bw, h, w, co)

    def _block(self, b: _Block, x: torch.Tensor) -> torch.Tensor:
        k = self.k
        c, co = x.shape[-1], b.dim_out
        y = F.layer_norm(x, (c,), *b.n1[:2], b.n1[2])
        shortcut = x if b.short is None else _pool(k.gemm(y, *b.short)).contiguous()
        h, w = shortcut.shape[1:3]
        if b.ws:
            y, _ = window_partition(y, b.ws)
            o = k.gemm(self._attention(b, y), *b.proj)
            ws = b.ws // 2 if b.pooled else b.ws
            pad_hw = (h + (ws - h % ws) % ws, w + (ws - w % ws) % ws)
            x = shortcut + window_unpartition(o, ws, pad_hw, (h, w))
        else:
            x = k.gemm(self._attention(b, y), *b.proj, res=shortcut)
        rows = x.view(-1, co)
        hid = k.gemm(F.layer_norm(rows, (co,), *b.n2[:2], b.n2[2]), *b.fc1, act=ACT_GELU)
        return k.gemm(hid, *b.fc2, res=rows, out=rows).view(x.shape)

    def __call__(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode one normalised `[1, 3, S, S]` image."""
        if x.shape != (1, 3, self.size, self.size):
            msg = f"dpt_fused SAM 2: expected a [1, 3, {self.size}, {self.size}] input, got {tuple(x.shape)}"
            raise ValueError(msg)
        with torch.autocast("cuda", enabled=False):
            p = self.patch(x.half().contiguous(memory_format=torch.channels_last)).permute(0, 2, 3, 1)
            h = (p + self.pos).contiguous()
            feats = []
            for i, b in enumerate(self.blocks):
                h = self._block(b, h)
                if i in self.stage_ends:
                    feats.append(h)
            return neck(self.neck, feats, self._linear)

    def _linear(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.k.gemm(x, w, b)
