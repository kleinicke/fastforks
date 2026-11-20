"""The Qwen-Image VAE (`AutoencoderKLQwenImage`) for one frame, with Marigold V2's fine-tuned decoder.

Upstream is a causal 3-D video VAE; on a single frame every causal 3-D conv sees only its last temporal tap (the
earlier taps meet the zero padding), so the checkpoint keeps that tap as a 2-D conv (`[:, :, -1]`) and drops the
temporal `time_conv` layers (identity on one frame). The RMS norms (`F.normalize` x sqrt(C) x gamma) run in float32.

The latent is `(z - latents_mean) / latents_std` as upstream's pipeline normalises it; `encode` draws the posterior
sample with the given noise (upstream: `latent_dist.sample()`); `decode` maps back and clamps to `[-1, 1]`.

`bands`: large same-padded convolutions (the full-resolution layers) run in row bands with a one-row halo, the 2x
upsample per band too, so neither the cuDNN workspace nor the upsampled tensor is ever full size. cuDNN may pick
another kernel for a band than for the whole image (another summation order): the output can differ in the last
bf16 bit, so `exact_mask` turns bands off.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .nf4 import BF

_SEP = "__"  # buffer names cannot contain dots
_CHUNKS = 8  # channel chunks of the float32 division in `norm`
_BAND = 32 << 20  # bytes of conv output per row band (`bands`)


class VAE2D(nn.Module):
    """Encoder and decoder from the `vae.*` tensors (bf16 weights; `latents_mean` / `latents_std` float32)."""

    def __init__(self, t: dict[str, torch.Tensor], *, bands: bool = False) -> None:
        """Take the `vae.*` tensors from `t`; `bands`: see the module docstring."""
        super().__init__()
        self.bands = bands
        self.keys: set[str] = set()
        for k, v in t.items():
            if k.startswith("vae."):
                name = k[4:]
                self.keys.add(name)
                self.register_buffer(name.replace(".", _SEP), v if v.dtype == BF else v.to(BF))
        self.register_buffer("mean", t["vae.latents_mean"].to(BF).view(1, -1, 1, 1))
        self.register_buffer("std_inv", 1.0 / t["vae.latents_std"].to(BF).view(1, -1, 1, 1))

    def t(self, name: str) -> torch.Tensor:
        """The tensor `name` (dotted checkpoint name)."""
        return getattr(self, name.replace(".", _SEP))

    def conv(self, x: torch.Tensor, p: str, stride: int = 1, *, up: bool = False) -> torch.Tensor:
        """Conv `p` (same padding; the stride-2 resample is padded by the caller); `up`: 2x nearest upsample first."""
        w, b = self.t(p + ".weight"), self.t(p + ".bias")
        n, _, h, wd = x.shape
        if up:
            h, wd = 2 * h, 2 * wd
        rows = _BAND // max(1, w.shape[0] * wd * x.element_size())
        if stride == 1 and self.bands and rows < h:
            return self._conv_bands(x, w, b, h, max(rows, 8), up=up)
        if up:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")  # bf16: a copy, equal to upstream's float32
        return F.conv2d(x, w, b, stride=stride, padding=0 if stride == 2 else w.shape[-1] // 2)

    @staticmethod
    def _conv_bands(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, h: int, rows: int, *, up: bool) -> torch.Tensor:
        """Same-padded conv of `x` (upsampled 2x first if `up`, `h` output rows) in bands of `rows` output rows."""
        k = w.shape[-1] // 2
        out = x.new_empty(x.shape[0], w.shape[0], h, 2 * x.shape[-1] if up else x.shape[-1])
        for r0 in range(0, h, rows):
            r1 = min(h, r0 + rows)
            lo, hi = max(0, r0 - k), min(h, r1 + k)  # input rows of the band with its halo
            if up:  # nearest-exact 2x: output row r reads input row r // 2
                i0 = lo // 2
                xb = F.interpolate(x[:, :, i0 : (hi + 1) // 2], scale_factor=2.0, mode="nearest-exact")
                xb = xb[:, :, lo - 2 * i0 : hi - 2 * i0]
            else:
                xb = x[:, :, lo:hi]
            if lo > r0 - k or hi < r1 + k:  # image border: the zero padding of the full conv
                xb = F.pad(xb, (0, 0, lo - (r0 - k), r1 + k - hi))
            out[:, :, r0:r1] = F.conv2d(xb, w, b, padding=(0, k))
        return out

    def norm(self, x: torch.Tensor, p: str, *, silu: bool = False) -> torch.Tensor:
        """Upstream `QwenImageRMS_norm`: channel-wise L2 normalisation in float32 x sqrt(C) x gamma (+ SiLU).

        As `F.normalize(x.float(), dim=1).to(x.dtype) * sqrt(C) * gamma`, bit for bit. The norm reduces the
        bf16 input in float32. Eager CUDA divides in float32 and writes directly to the bf16 output.
        """
        n = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)
        out = torch.empty_like(x)
        if x.is_cuda and not x.requires_grad and not torch.compiler.is_compiling():
            torch.div(x, n, out=out)
        else:
            step = max(1, x.shape[1] // _CHUNKS)
            for c in range(0, x.shape[1], step):
                out[:, c : c + step] = x[:, c : c + step].float().div_(n)
        out.mul_(x.shape[1] ** 0.5).mul_(self.t(p + ".gamma"))
        return F.silu(out, inplace=True) if silu else out

    def res(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """Residual block (each full-size temporary dropped as soon as it is consumed)."""
        h = self.conv(x, p + ".conv_shortcut") if p + ".conv_shortcut.weight" in self.keys else x
        x = self.conv(self.norm(x, p + ".norm1", silu=True), p + ".conv1")
        x = self.norm(x, p + ".norm2", silu=True)
        return self.conv(x, p + ".conv2").add_(h)

    def attn(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """Single-head attention over the pixels of the mid block."""
        b, c, h, w = x.shape
        qkv = self.conv(self.norm(x, p + ".norm"), p + ".to_qkv")
        q, k, v = qkv.reshape(b, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
        return self.conv(y, p + ".proj") + x

    def mid(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """Mid block: residual, attention, residual."""
        x = self.res(x, p + ".resnets.0")
        x = self.attn(x, p + ".attentions.0")
        return self.res(x, p + ".resnets.1")

    def encode_params(self, x: torch.Tensor) -> torch.Tensor:
        """`[B, 3, H, W]` in `[-1, 1]` -> posterior mean and log-variance `[B, 32, H / 8, W / 8]`."""
        x = self.conv(x.to(BF), "encoder.conv_in")
        j = 0
        while True:
            p = f"encoder.down_blocks.{j}"
            if p + ".resample.1.weight" in self.keys:
                x = self.conv(F.pad(x, (0, 1, 0, 1)), p + ".resample.1", stride=2)
            elif p + ".conv1.weight" in self.keys:
                x = self.res(x, p)
            else:
                break
            j += 1
        x = self.mid(x, "encoder.mid_block")
        x = self.conv(self.norm(x, "encoder.norm_out", silu=True), "encoder.conv_out")
        return self.conv(x, "quant_conv")

    def encode(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Normalised posterior sample `[B, 16, H / 8, W / 8]` with the standard-normal `noise` of that shape."""
        mean, logvar = self.encode_params(x).chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        return (mean + std * noise.to(mean.dtype) - self.mean) * self.std_inv

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised latent -> `[B, 3, H, W]` in `[-1, 1]` (bf16)."""
        x = self.conv(z / self.std_inv + self.mean, "post_quant_conv")
        x = self.conv(x, "decoder.conv_in")
        x = self.mid(x, "decoder.mid_block")
        for i in range(4):
            p = f"decoder.up_blocks.{i}"
            for r in range(3):
                x = self.res(x, f"{p}.resnets.{r}")
            if p + ".upsamplers.0.resample.1.weight" in self.keys:
                if not self.bands:  # here, so the half-size input is freed before the conv
                    x = F.interpolate(x, scale_factor=2.0, mode="nearest-exact")  # bf16: a copy, equal to float32
                x = self.conv(x, p + ".upsamplers.0.resample.1", up=self.bands)
        x = self.conv(self.norm(x, "decoder.norm_out", silu=True), "decoder.conv_out")
        return x.clamp(-1.0, 1.0)
