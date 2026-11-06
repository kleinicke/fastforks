"""Kernels Restormer TransformerBlock of DLNR's extractor (port of our `fastbb.restormer` kernel path).

The portable block (`net/extractor.py` `TransformerBlock.forward`, NHWC) with its non-GEMM work on `fastbb` kernels:
  - both LayerNorms as `ln_lastdim` (one read, one write, fp32 statistics)
  - the MDTA depthwise 3x3 on the NHWC qkv as `dwconv3` (no NCHW round trip)
  - q / k L2 norms over the pixels as fp32 vector norms and q^T k as a chunked batched GEMM with fp32 output, the
    norms divided out of the [c, c] logits afterwards (same maths as normalising q and k first; no normalised copies)
  - the GDFN's depthwise 3x3 + GELU gate as one `dwgate` kernel; its hidden width padded to a multiple of 8 with
    zero weights (layout [a | 0 | b | 0]; padded channels are gelu(0) * 0 = 0) so that all GEMMs and loads are aligned
The GEMMs stay cuBLAS (fp32 accumulation) as portable; attn @ v + project_out stay folded into one baddbmm.
"""

import torch
import torch.nn.functional as F
from torch import nn

from fastforks._core.errors import BackendUnavailableError
from fastforks.kernels import fastbb

from ..net.extractor import TransformerBlock, WithBias_LayerNorm

CHUNK = 1024  # rows per q^T k partial GEMM (one long-K GEMM per sample leaves the GPU mostly idle)


def _w2(conv: nn.Conv2d) -> torch.Tensor:
    return conv.weight.reshape(conv.out_channels, -1)


def _pad(t: torch.Tensor, h: int, hp: int) -> torch.Tensor:
    """[2h, ...] -> [2hp, ...] as [a | 0 | b | 0]."""
    z = t.new_zeros((hp - h, *t.shape[1:]))
    return torch.cat([t[:h], z, t[h:], z]).contiguous()


def _gram(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """q^T k over the rows of [B, HW, C] views -> [B, C, C] fp32."""
    b, hw, c = q.shape
    s = hw // CHUNK
    r0 = s * CHUNK if s > 1 else 0
    out = []
    for i in range(b):
        qb, kb = q[i], k[i]
        if r0 < hw:
            g = torch.mm(qb[r0:].t(), kb[r0:], out_dtype=torch.float32)
        else:
            g = q.new_zeros((c, c), dtype=torch.float32)
        if s > 1:
            qs, ks = qb[:r0].unflatten(0, (s, CHUNK)), kb[:r0].unflatten(0, (s, CHUNK))
            g = g + torch.bmm(qs.transpose(1, 2), ks, out_dtype=torch.float32).sum(0)
        out.append(g)
    return torch.stack(out)


class KernelsTransformerBlock(nn.Module):
    """Built from a float16 TransformerBlock (after `model.half()`; the fp32 parameters derived here stay fp32)."""

    ln1w: torch.Tensor
    ln2w: torch.Tensor
    wqkv: torch.Tensor
    wdw: torch.Tensor
    wp_t: torch.Tensor
    temp: torch.Tensor
    win: torch.Tensor
    wg: torch.Tensor
    wout_t: torch.Tensor
    ln1b: torch.Tensor | None
    ln2b: torch.Tensor | None
    bqkv: torch.Tensor | None
    bdw: torch.Tensor | None
    bp: torch.Tensor | None
    bin: torch.Tensor | None
    bg: torch.Tensor | None
    bout: torch.Tensor | None

    def __init__(self, blk: TransformerBlock) -> None:
        super().__init__()
        at, ff = blk.attn, blk.ffn
        c = at.project_out.out_channels
        if c % 8 or c > fastbb.LN_MAX_C or blk.norm1.body.weight.dtype != torch.float16:
            msg = f"DLNR kernels Restormer: float16 blocks with C % 8 == 0, C <= {fastbb.LN_MAX_C} only, got C = {c}"
            raise BackendUnavailableError(msg)
        self.heads = at.num_heads
        buf = lambda n, t: self.register_buffer(n, None if t is None else t.contiguous(), persistent=False)
        for i, nm in ((1, blk.norm1), (2, blk.norm2)):
            body = nm.body
            buf(f"ln{i}w", body.weight.float())
            buf(f"ln{i}b", body.bias.float() if isinstance(body, WithBias_LayerNorm) else None)
        self.bias_free = not isinstance(blk.norm1.body, WithBias_LayerNorm)
        buf("wqkv", _w2(at.qkv))
        buf("bqkv", at.qkv.bias)
        buf("wdw", _w2(at.qkv_dwconv).reshape(3 * c, 3, 3).permute(1, 2, 0))  # [3, 3, 3C]
        buf("bdw", at.qkv_dwconv.bias)
        buf("wp_t", _w2(at.project_out).t().float().reshape(self.heads, c // self.heads, c))  # [heads, c, C]
        buf("bp", at.project_out.bias)
        buf("temp", at.temperature.float().reshape(1, self.heads, 1, 1))
        win = _w2(ff.project_in)
        h = win.shape[0] // 2
        hp = (h + 7) // 8 * 8
        buf("win", _pad(win, h, hp))
        buf("bin", None if ff.project_in.bias is None else _pad(ff.project_in.bias, h, hp))
        buf("wg", _pad(_w2(ff.dwconv).float(), h, hp).t())  # [9, 2hp]
        buf("bg", None if ff.dwconv.bias is None else _pad(ff.dwconv.bias.float(), h, hp))
        wout = _w2(ff.project_out)  # [C, h]
        buf("wout_t", torch.cat([wout.t(), wout.new_zeros((hp - h, c))]))  # [hp, C]
        buf("bout", ff.project_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, C, H, W] float16 (channels-last) -> channels-last NCHW view."""
        b, c, hh, ww = x.shape
        heads, cc = self.heads, c // self.heads
        xh = x.permute(0, 2, 3, 1).contiguous()  # free for channels-last input
        x3 = xh.reshape(b, hh * ww, c)

        # MDTA
        y = fastbb.ln_lastdim(xh, self.ln1w, self.ln1b, 1e-5, bias_free=self.bias_free)
        t = fastbb.dwconv3(F.linear(y, self.wqkv, self.bqkv), self.wdw, self.bdw).reshape(b, hh * ww, 3 * c)
        nrm = torch.linalg.vector_norm(t[:, :, : 2 * c], dim=1, dtype=torch.float32).clamp_(min=1e-12)  # [B, 2C]
        g = _gram(t[:, :, :c], t[:, :, c : 2 * c])  # [B, C, C]
        g = torch.diagonal(g.view(b, heads, cc, heads, cc), dim1=1, dim2=3).permute(0, 3, 1, 2)  # [B, heads, c, c]
        g = g / (nrm[:, :c].reshape(b, heads, cc, 1) * nrm[:, c:].reshape(b, heads, 1, cc))
        attn = torch.softmax(g * self.temp, dim=-1)
        m = torch.matmul(attn.transpose(-1, -2), self.wp_t).reshape(b, c, c).to(x.dtype)
        x3 = torch.baddbmm(x3, t[:, :, 2 * c :], m)
        if self.bp is not None:
            x3 = x3 + self.bp

        # GDFN
        y = fastbb.ln_lastdim(x3.reshape(b, hh, ww, c), self.ln2w, self.ln2b, 1e-5, bias_free=self.bias_free)
        gg = fastbb.dwgate(F.linear(y, self.win, self.bin), self.wg, self.bg)  # [B, H, W, hp]
        out = torch.addmm(x3.reshape(-1, c), gg.reshape(-1, gg.shape[-1]), self.wout_t)
        if self.bout is not None:
            out = out + self.bout
        return out.reshape(b, hh, ww, c).permute(0, 3, 1, 2)


def swap_blocks(mod: nn.Module) -> int:
    """Replace every TransformerBlock below `mod` by a KernelsTransformerBlock; returns the count."""
    n = 0
    for name, ch in mod.named_children():
        if isinstance(ch, TransformerBlock):
            setattr(mod, name, KernelsTransformerBlock(ch))
            n += 1
        else:
            n += swap_blocks(ch)
    return n
