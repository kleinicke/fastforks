"""NF4 linear kernels (`nf4.cu`): bitsandbytes 4-bit NormalFloat -> bf16 dequantization and LoRA epilogues.

`dequant` writes `(code[nibble] * absmax).to(bfloat16)` for every packed weight straight from the uint8 codes,
instead of the portable unpack / gather / multiply / cast chain, so a 20B-parameter NF4 transformer streams its
weights at memory speed. The LoRA epilogues (`lora_*`) sum the bf16 GEMM output and the bf16 LoRA term in fp32 and
consume the sum in the same pass (fp32 out, bf16 out, tanh-GELU to bf16, gate + residual in place); `mod_bf16` is the
block's modulation (`x * scale + shift` to bf16). All are bit-equal to the PyTorch chains they replace. Used by
`fastforks.depth.models.marigold_v2` (kernels mode).
"""

from __future__ import annotations

from pathlib import Path

import torch

from fastforks._core import kernels
from fastforks._core.kernels import cubin_library

# p: pointer, l: int64 (must match the extern "C" kernels in nf4.cu)
SIGNATURES = {
    "nf4_dequant": "ppppl",
    "nf4_lora_f32": "pppl",
    "nf4_lora_bf16": "pppl",
    "nf4_lora_gelu_bf16": "pppl",
    "nf4_lora_gate_res": "ppppll",
    "nf4_mod_bf16": "ppppll",
    "nf4_rms_rope_bf16": "pppppliil",
}
_WORD = 8  # bytes per 64-bit word (16 codes)
load = cubin_library(Path(__file__).with_name("manifest.toml"), SIGNATURES)


def dequant(q: torch.Tensor, absmax: torch.Tensor, code: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Fill `out` (bf16, `2 * q.numel()` elements) from `q` (uint8 packed codes), `absmax` (bf16, per 64), `code`.

    All on one CUDA device; `q` contiguous with a size divisible by 8, `code` fp32 `[16]`.
    """
    n = q.numel() // _WORD
    lib = load(q.device)
    lib.launch("nf4_dequant", *kernels.linear(n), q, absmax, code, out, n)
    return out


def _n8(*ts: torch.Tensor) -> int:
    n = ts[0].numel()
    if n % 8 or any(not t.is_contiguous() or t.numel() != n for t in ts):
        msg = "nf4 epilogues need contiguous tensors of one size, a multiple of 8 elements"
        raise ValueError(msg)
    return n // 8


def _vec(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if v.numel() != x.shape[-1] or not v.is_contiguous() or v.dtype != torch.bfloat16:
        msg = "per-column vectors must be contiguous bf16 as long as the last dim"
        raise ValueError(msg)
    return v


def _run(name: str, n8: int, x: torch.Tensor, *args: torch.Tensor | float | None) -> None:
    load(x.device).launch(name, *kernels.linear(n8), x, *args)


def lora_f32(y: torch.Tensor, lora: torch.Tensor) -> torch.Tensor:
    """`y.float() + lora` (both bf16, same shape) as a new fp32 tensor."""
    n8 = _n8(y, lora)
    out = torch.empty_like(y, dtype=torch.float32)
    _run("nf4_lora_f32", n8, y, lora, out, n8)
    return out


def lora_bf16(y: torch.Tensor, lora: torch.Tensor) -> torch.Tensor:
    """`(y.float() + lora).to(bfloat16)` (= `y + lora` in bf16)."""
    n8 = _n8(y, lora)
    out = torch.empty_like(y)
    _run("nf4_lora_bf16", n8, y, lora, out, n8)
    return out


def lora_gelu_bf16(y: torch.Tensor, lora: torch.Tensor) -> torch.Tensor:
    """`F.gelu(y.float() + lora, approximate="tanh").to(bfloat16)`."""
    n8 = _n8(y, lora)
    out = torch.empty_like(y)
    _run("nf4_lora_gelu_bf16", n8, y, lora, out, n8)
    return out


def lora_gate_res(y: torch.Tensor, lora: torch.Tensor, gate: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
    """`res += (y.float() + lora) * gate` in place (`res` fp32, `gate` bf16 over the last dim)."""
    n8 = _n8(y, lora, res)
    _run("nf4_lora_gate_res", n8, y, lora, _vec(gate, y), res, n8, y.shape[-1] // 8)
    return res


def mod_bf16(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """`(x * scale + shift).to(bfloat16)` (`x` fp32, `scale` / `shift` bf16 over the last dim)."""
    n8 = _n8(x)
    out = torch.empty_like(x, dtype=torch.bfloat16)
    _run("nf4_mod_bf16", n8, x, _vec(scale, x), _vec(shift, x), out, n8, x.shape[-1] // 8)
    return out


def rms_rope_bf16(x: torch.Tensor, w: torch.Tensor, f: torch.Tensor, out: torch.Tensor, t0: int) -> torch.Tensor:
    """`out[:, t0 : t0 + T] = rope(x * rsqrt(x.pow(2).mean(-1) + 1e-6) * w, f).to(bfloat16)`: RMS norm + RoPE of q / k.

    `x` fp32 `[1, T, H, D]`, `w` fp32 `[D]`, `f` complex64 `[T_all, 1, D / 2]` (the joint sequence), `out` bf16
    `[1, T_all, H, D]`; all contiguous. The mean stays PyTorch's (its reduction order), the rest is one kernel.
    """
    if not all(t.is_contiguous() for t in (x, w, f, out)) or x.shape[0] != 1 or out.shape[2:] != x.shape[2:]:
        msg = "rms_rope_bf16 needs contiguous tensors, batch 1, matching heads"
        raise ValueError(msg)
    n2, half = x.numel() // 2, x.shape[-1] // 2
    v = x.pow(2).mean(-1)
    _run("nf4_rms_rope_bf16", n2, x, v, w, torch.view_as_real(f), out, n2, x.shape[2], half, t0)
    return out
