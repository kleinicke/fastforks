"""The Qwen-Image VAE (`AutoencoderKLQwenImage`) for one frame, with Marigold V2's fine-tuned decoder.

Upstream is a causal 3-D video VAE; on a single frame every causal 3-D conv sees only its last temporal tap (the
earlier taps meet the zero padding), so the checkpoint keeps that tap as a 2-D conv (`[:, :, -1]`) and drops the
temporal `time_conv` layers (identity on one frame). The RMS norms (`F.normalize` x sqrt(C) x gamma) run in float32.

The latent is `(z - latents_mean) / latents_std` as upstream's pipeline normalises it; `encode` draws the posterior
sample with the given noise (upstream: `latent_dist.sample()`); `decode` maps back and clamps to `[-1, 1]`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .nf4 import BF

_SEP = "__"  # buffer names cannot contain dots


class VAE2D(nn.Module):
    """Encoder and decoder from the `vae.*` tensors (bf16 weights; `latents_mean` / `latents_std` float32)."""

    def __init__(self, t: dict[str, torch.Tensor]) -> None:
        """Take the `vae.*` tensors from `t`."""
        super().__init__()
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

    def conv(self, x: torch.Tensor, p: str, stride: int = 1) -> torch.Tensor:
        """Conv `p` (same padding; the stride-2 resample is padded by the caller)."""
        w = self.t(p + ".weight")
        return F.conv2d(x, w, self.t(p + ".bias"), stride=stride, padding=0 if stride == 2 else w.shape[-1] // 2)

    def norm(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """Upstream `QwenImageRMS_norm`: channel-wise L2 normalisation in float32 x sqrt(C) x gamma."""
        n = F.normalize(x.float(), dim=1).to(x.dtype)
        return n * (x.shape[1] ** 0.5) * self.t(p + ".gamma")

    def res(self, x: torch.Tensor, p: str) -> torch.Tensor:
        """Residual block."""
        h = self.conv(x, p + ".conv_shortcut") if p + ".conv_shortcut.weight" in self.keys else x
        x = self.conv(F.silu(self.norm(x, p + ".norm1")), p + ".conv1")
        x = self.conv(F.silu(self.norm(x, p + ".norm2")), p + ".conv2")
        return x + h

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
        x = self.conv(F.silu(self.norm(x, "encoder.norm_out")), "encoder.conv_out")
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
                x = F.interpolate(x.float(), scale_factor=2.0, mode="nearest-exact").to(x.dtype)
                x = self.conv(x, p + ".upsamplers.0.resample.1")
        x = self.conv(F.silu(self.norm(x, "decoder.norm_out")), "decoder.conv_out")
        return x.clamp(-1.0, 1.0)
