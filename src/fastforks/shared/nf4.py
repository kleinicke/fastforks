"""NF4 (bitsandbytes 4-bit NormalFloat, blocksize 64) weights in plain PyTorch: quantise, expand to bf16.

Storage as bitsandbytes' `Linear4bit`: per block of 64 weights an `absmax` (here bf16, the largest |w|), each weight
the nearest of the 16 `NF4_CODE` values times it (as a nibble, two per byte, high nibble first). `dequant_lut` expands
a weight to bf16 (`bf16(code * absmax)`, round to nearest even, as bitsandbytes dequantises) by a gather from a
256-entry pair table; `fastforks.kernels.nf4.dequant` is the same as one fused kernel. Used by Marigold V2 (weights
stored in NF4) and SeedVR2 (`quant = "nf4"`: bf16 weights quantised at load).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

BF = torch.bfloat16
BLOCK = 64  # weights per absmax
NF4_CODE = (  # bitsandbytes `functional.get_4bit_type("nf4")`, as float32
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)  # fmt: skip


def code(device: torch.device) -> torch.Tensor:
    """`NF4_CODE` as a float32 tensor `[16]` on `device`."""
    return torch.tensor(NF4_CODE, dtype=torch.float32, device=device)


def pair_lut(code: torch.Tensor) -> torch.Tensor:
    """`[256, 2]` float32: the two code values of each packed byte (high nibble first)."""
    b = torch.arange(256, device=code.device)
    return torch.stack([code[b >> 4], code[b & 15]], dim=1)


def dequant_lut(q: torch.Tensor, absmax: torch.Tensor, lut: torch.Tensor, n_out: int) -> torch.Tensor:
    """Packed bytes `[N * K / 2]` + `absmax [N * K / 64]` -> bf16 weight `[N, K]` (plain PyTorch).

    An embedding lookup of the byte in the pair table (float32), then one multiply by the absmax that rounds straight
    to bf16 (computed in float32, as `(code * absmax).to(bfloat16)`). Index tensors must be int32 or int64; the
    embedding is ~2x faster than advanced indexing, per-block product tables were slower still.
    """
    g = F.embedding(q.int(), lut).view(-1, BLOCK)
    w = torch.empty(g.shape, dtype=BF, device=g.device)
    return torch.mul(g, absmax.reshape(-1)[:, None], out=w).view(n_out, -1)


def quantize(w: torch.Tensor, rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight `[N, K]` (K a multiple of 64) -> packed bytes `[N * K / 2]` (uint8) and `absmax [N, K / 64]` (bf16).

    Each weight becomes the code value nearest to `w / absmax` (bf16 absmax, ties to the lower code), as
    bitsandbytes' `quantize_4bit` with the absmax rounded to bf16. `rows` rows at a time bound the float32 scratch.
    """
    n, k = w.shape
    if k % BLOCK:
        msg = f"NF4 needs the input width to be a multiple of {BLOCK}, got {k}"
        raise ValueError(msg)
    c = code(w.device)
    mids = (c[1:] + c[:-1]) / 2
    q = torch.empty(n * k // 2, dtype=torch.uint8, device=w.device)
    absmax = torch.empty(n, k // BLOCK, dtype=BF, device=w.device)
    for r in range(0, n, rows):
        blk = w[r : r + rows].float().view(-1, k // BLOCK, BLOCK)
        am = blk.abs().amax(-1).to(BF)
        absmax[r : r + rows] = am
        x = blk / am.float().clamp_min(torch.finfo(torch.float32).tiny)[..., None]
        idx = torch.bucketize(x, mids).to(torch.uint8).view(-1, 2)
        q[r * k // 2 : (r + rows) * k // 2] = (idx[:, 0] << 4) | idx[:, 1]
    return q, absmax
