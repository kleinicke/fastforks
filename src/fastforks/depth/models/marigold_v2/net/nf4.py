"""NF4 linear layers of the lean checkpoint: `y = dequant(W) x + b + B (A x)`.

Storage follows bitsandbytes' NF4 (`Linear4bit`, blocksize 64): per block of 64 weights an `absmax` (bf16, the largest
|w|), each weight the nearest of the 16 `NF4_CODE` values (as a nibble, two per byte, high nibble first). The weight
is expanded to bf16 for each call (`bf16(code * absmax)`, round to nearest even, as bitsandbytes dequantises) and
dropped again, so the resident model is the packed size. The expansion is a gather from a 256-entry pair table
(`lut`) in plain PyTorch; the kernels backend sets `NF4Linear.dequant` to a fused kernel (`fastforks.kernels.nf4`).

The dtype choreography is upstream's under autocast: a bf16 GEMM with the bf16 bias (cuBLAS), the LoRA term as two
bf16 GEMMs, both summed in float32 (`keep_bf` keeps the sum in bf16, as `img_in` does). `bf`, `gelu` and `gate_res`
consume that sum directly (to bf16, tanh-GELU to bf16, gated residual add); with the kernels backend each is one fused
epilogue kernel (`NF4Linear.epi`), bit-equal to the PyTorch chain.
"""

from __future__ import annotations

from typing import Callable, ClassVar, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from fastforks.shared.nf4 import BF, BLOCK, dequant_lut, pair_lut

if TYPE_CHECKING:
    from types import ModuleType

__all__ = ["BF", "BLOCK", "NF4Linear", "dequant_lut", "pair_lut"]


class NF4Linear(nn.Module):
    """One NF4 linear with its bias and optional LoRA factors (buffers `q`, `absmax`, `bias`, `lora_a`, `lora_b`)."""

    lut: ClassVar[torch.Tensor | None] = None  # pair table of the device (set by the network)
    code: ClassVar[torch.Tensor | None] = None  # the 16 code values, float32, on the device
    dequant: ClassVar[Callable[[NF4Linear], torch.Tensor] | None] = None  # fused kernel: self -> bf16 weight
    epi: ClassVar[ModuleType | None] = None  # fused LoRA epilogues (`fastforks.kernels.nf4`), kernels backend

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

    def _parts(self, xb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The bf16 GEMM output (with bias) and the bf16 LoRA term (None without LoRA)."""
        y = F.linear(xb, self.weight(), self.bias)
        return y, None if self.lora_a is None else F.linear(F.linear(xb, self.lora_a), self.lora_b)

    def _epi(self, lora: torch.Tensor | None) -> ModuleType | None:
        """The fused epilogue kernels, if this layer uses them."""
        return NF4Linear.epi if self.fused and lora is not None else None

    def forward(self, xb: torch.Tensor, *, keep_bf: bool = False) -> torch.Tensor:
        """bf16 `xb` -> float32 (bf16 GEMM + bf16 LoRA term summed in float32), or bf16 with `keep_bf`."""
        y, lora = self._parts(xb)
        if lora is None:
            return y if keep_bf else y.float()
        if (e := self._epi(lora)) is not None:
            return e.lora_bf16(y, lora) if keep_bf else e.lora_f32(y, lora)
        if keep_bf:
            return y + lora
        out = y.float()
        del y
        return out.add_(lora)  # float32 + bf16 in float32, as `y.float() + lora`, without a second float32 copy

    def bf(self, xb: torch.Tensor) -> torch.Tensor:
        """`self(xb).to(bfloat16)`."""
        y, lora = self._parts(xb)
        if (e := self._epi(lora)) is not None:
            return e.lora_bf16(y, lora)
        return y if lora is None else y.add_(lora)

    def gelu(self, xb: torch.Tensor) -> torch.Tensor:
        """`F.gelu(self(xb), approximate="tanh").to(bfloat16)`."""
        y, lora = self._parts(xb)
        if (e := self._epi(lora)) is not None:
            return e.lora_gelu_bf16(y, lora)
        x = y.float() if lora is None else y.float().add_(lora)
        del y
        return torch.ops.aten.gelu_(x, approximate="tanh").to(BF)

    def gate_res(self, xb: torch.Tensor, gate: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """`res += self(xb) * gate` in place (`res` float32, `gate` bf16 over the last dim)."""
        y, lora = self._parts(xb)
        if (e := self._epi(lora)) is not None:
            return e.lora_gate_res(y, lora, gate, res)
        x = y.float() if lora is None else y.float().add_(lora)
        del y
        return res.add_(x.mul_(gate))
