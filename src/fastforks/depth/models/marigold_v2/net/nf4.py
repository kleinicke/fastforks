"""NF4 linear layers of the lean checkpoint: `y = dequant(W) x + b + B (A x)`.

Storage follows bitsandbytes' NF4 (`Linear4bit`, blocksize 64): per block of 64 weights an `absmax` (bf16, the largest
|w|), each weight the nearest of the 16 `NF4_CODE` values (as a nibble, two per byte, high nibble first). The weight
is expanded to bf16 for each call (`bf16(code * absmax)`, round to nearest even, as bitsandbytes dequantises) and
dropped again, so the resident model is the packed size. The expansion is a gather from a 256-entry pair table
(`lut`) in plain PyTorch; the kernels backend sets `NF4Linear.dequant` to a fused kernel (`fastforks.kernels.nf4`).

The dtype choreography is upstream's under autocast: a bf16 GEMM with the bf16 bias (cuBLAS), the LoRA term as two
bf16 GEMMs, both summed in float32 (`keep_bf` keeps the sum in bf16, as `img_in` does).
"""

from __future__ import annotations

from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

BF = torch.bfloat16
BLOCK = 64  # weights per absmax


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
    return torch.mul(g, absmax[:, None], out=w).view(n_out, -1)


class NF4Linear(nn.Module):
    """One NF4 linear with its bias and optional LoRA factors (buffers `q`, `absmax`, `bias`, `lora_a`, `lora_b`)."""

    lut: ClassVar[torch.Tensor | None] = None  # pair table of the device (set by the network)
    code: ClassVar[torch.Tensor | None] = None  # the 16 code values, float32, on the device
    dequant: ClassVar[Callable[[NF4Linear], torch.Tensor] | None] = None  # fused kernel: self -> bf16 weight

    def __init__(self, t: dict[str, torch.Tensor], name: str, *, fused: bool = False) -> None:
        """Take the tensors `name.q`, `.absmax`, `.bias` and, if present, `.lora_A`, `.lora_B` from `t`.

        `fused`: expand with `NF4Linear.dequant` (the kernels backend) instead of the table gather.
        """
        super().__init__()
        self.fused = fused
        self.register_buffer("q", t[name + ".q"])
        self.register_buffer("absmax", t[name + ".absmax"])
        self.register_buffer("bias", t[name + ".bias"])
        self.register_buffer("lora_a", t.get(name + ".lora_A"))
        self.register_buffer("lora_b", t.get(name + ".lora_B"))

    def weight(self) -> torch.Tensor:
        """The bf16 weight `[N, K]`, expanded for this call."""
        if self.fused and NF4Linear.dequant is not None:
            return NF4Linear.dequant(self)
        lut = NF4Linear.lut
        if lut is None or lut.device != self.q.device:
            msg = "NF4Linear.lut is not set for this device"
            raise RuntimeError(msg)
        return dequant_lut(self.q, self.absmax, lut, self.bias.numel())

    def forward(self, xb: torch.Tensor, *, keep_bf: bool = False) -> torch.Tensor:
        """bf16 `xb` -> float32 (bf16 GEMM + bf16 LoRA term summed in float32), or bf16 with `keep_bf`."""
        y = F.linear(xb, self.weight(), self.bias)
        if self.lora_a is None:
            return y if keep_bf else y.float()
        lora = F.linear(F.linear(xb, self.lora_a), self.lora_b)
        if keep_bf:
            return y + lora
        out = y.float()
        del y
        return out.add_(lora)  # float32 + bf16 in float32, as `y.float() + lora`, without a second float32 copy
