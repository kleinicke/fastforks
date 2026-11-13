"""Kernel ViT blocks of the Track-On2 adapter (as `features.models.dinov3` `Kernels._block`).

fp16 weights, the q / k / v and gate / up GEMMs packed, dpt_fused GEMMs with fused bias / LayerScale / residual
epilogues, RoPE in place on the packed q / k (fastbb `rope_qk_h`, q also takes the softmax scale) and fastbb flash
attention with fp32 `q k^T` ("fh": ViT-S q / k outliers exceed the fp16 range).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core import kernels
from fastforks._core.errors import BackendUnavailableError, InputError
from fastforks.kernels.dpt_fused import DEEP_K, DptFused
from fastforks.kernels.fastbb import Flash64
from fastforks.kernels.fastbb import load as load_kernels
from fastforks.shared.depth_anything_v2.fast_vit import HEAD_DIM, _flash

from .net.dinov3 import rope_table

if TYPE_CHECKING:
    from .net.track_on import TrackOn2

_INT32 = 2**31 - 1  # fixed: `rope_qk_h` indexes in int32


def install(net: TrackOn2) -> None:
    """Replace the adapter's `vit_block` / `vit_rope` with the kernel versions (the ViT in fp16, packed)."""
    enc = net.backbone.vit_encoder
    device = enc.level_embed.device
    vit = enc.dinov3.pack()
    scales = [(b.layer_scale1.lambda1.detach().float(), b.layer_scale2.lambda1.detach().float()) for b in vit.blocks()]
    vit.half()
    blocks, cfg = vit.blocks(), vit.cfg
    if cfg.hidden_size // cfg.num_heads != HEAD_DIM:
        msg = f"fastbb flash attention needs head_dim {HEAD_DIM}"
        raise BackendUnavailableError(msg)
    k, kern, attn = DptFused(device), load_kernels(device), Flash64("fh")
    tables: dict[tuple[int, int], torch.Tensor] = {}

    def vit_rope(h: int, w: int, x: torch.Tensor) -> tuple:
        if (h, w) not in tables:
            cos, sin = rope_table(h, w, HEAD_DIM, cfg.rope_theta, x.device)
            tables[h, w] = torch.stack((cos[:, : HEAD_DIM // 2], sin[:, : HEAD_DIM // 2]), 1).contiguous()
        cs = tables[h, w]
        b, n = x.shape[0], x.shape[1]
        total = b * n * 2 * cfg.num_heads * (HEAD_DIM // 4)  # q / k half2 pairs per half head
        grid = kernels.linear(total)
        if total + grid[0][0] * grid[1][0] > _INT32:
            msg = f"batch too large for the fastbb RoPE kernel ({b} x {n} tokens): use fewer frames per chunk"
            raise InputError(msg)
        return cs, grid, total

    @torch.autocast("cuda", enabled=False)  # fp16 LayerNorm outputs for the GEMMs (float32 statistics)
    def vit_block(i: int, x: torch.Tensor, rope: tuple) -> torch.Tensor:
        cs, grid, total = rope
        x = x.half().contiguous()
        blk = blocks[i]
        b, n, _ = x.shape
        a, m = blk.attention, blk.mlp
        s1, s2 = scales[i]
        w, bias = cast("torch.Tensor", a.qkv_weight), cast("torch.Tensor", a.qkv_bias)
        qkv = k.gemm(blk.norm1(x), w, bias).view(b, n, 3, cfg.num_heads, HEAD_DIM)
        kern.launch("rope_qk_h", *grid, qkv, cs, n, n - cs.shape[0], cfg.num_heads, total, HEAD_DIM**-0.5)
        x = k.gemm(_flash(qkv, 1.0, attn), a.o_proj.weight, a.o_proj.bias, scale=s1, res=x)
        y = blk.norm2(x)
        if hasattr(m, "gate_up_weight"):  # SwiGLU (ViT-S+): the gate in place, the down GEMM reads the strided half
            w, bias = cast("torch.Tensor", m.gate_up_weight), cast("torch.Tensor", m.gate_up_bias)
            g, u = k.gemm(y, w, bias).chunk(2, -1)
            hid = F.silu(g, inplace=True).mul_(u)
        else:
            hid = k.gemm(y, m.up_proj.weight, m.up_proj.bias, act=k.gelu)
        if m.down_proj.in_features >= DEEP_K:
            return x.addcmul_(m.down_proj(hid), blk.layer_scale2.lambda1)
        return k.gemm(hid, m.down_proj.weight, m.down_proj.bias, scale=s2, res=x, out=x)

    enc.vit_block = vit_block  # per-instance override
    enc.vit_rope = vit_rope  # ty: ignore[invalid-assignment]  # per-instance override
