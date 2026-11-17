"""Marigold runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `MarigoldDepthPipeline.__call__` with `ensemble_size=1`: resize the longest side to
`long_side` (bilinear, antialiased, rounded to 8-bit as upstream resizes the uint8 image), map to `[-1, 1]`, encode
with the VAE (posterior mean x 0.18215), denoise a Gaussian latent in `steps` DDIM steps (v-prediction, trailing
timesteps, zero terminal SNR) with the UNet conditioned on the image latent and the empty-prompt CLIP embedding, decode,
average the three channels, clip to `[-1, 1]`, map to `[0, 1]`, resize to the input size (bilinear) and clip.

The initial noise is `torch.randn` from a CPU generator seeded with `seed` (float32, then moved to the device), so a
given seed and input size give the same noise on every device and in every mode. Upstream draws it from the generator
the caller passes (`marigold_depth_pipeline.py`, `single_infer`), on the pipeline's device and dtype: the values equal
upstream's with a seeded CPU generator, but differ from upstream's CUDA generator (and fp16 noise) for the same seed.
The noise is drawn eagerly (`Noise`) and kept for the last latent shape, outside the traced graph, so `module()`
compiles and exports; a traced or exported module reuses the noise of the shape it was traced with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.sd2 import LATENT_SCALE, Noise, SD2Net, load_net
from fastforks.shared.sd2.net.fast import Fast

from .net.ddim import DDIM

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

_MIN_SIDE = 8  # fixed: smallest network input side (the VAE encoder fails below 8: measured 3..7 fail, 8 works)


@dataclass(frozen=True)
class Params:
    """Load options."""

    steps: int  # DDIM denoising steps
    long_side: int  # longest side of the network input; 0 = the input size
    seed: int  # initial-noise seed


def network_size(h: int, w: int, res: int) -> tuple[int, int]:
    """Upstream `resize_max_res`: scale the longest side to `res` (sides truncated).

    A side below `_MIN_SIDE` (aspect ratio above ~100, where the VAE fails) is `_MIN_SIDE` (stretched).
    """
    f = min(res / w, res / h)
    return max(_MIN_SIDE, int(h * f)), max(_MIN_SIDE, int(w * f))


def prepare(image: torch.Tensor, res: int) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> network input in `[-1, 1]` (float32)."""
    rgb = image * 255.0
    if res > 0:
        size = network_size(*image.shape[-2:], res)
        rgb = F.interpolate(rgb, size, mode="bilinear", align_corners=False, antialias=True).round_().clamp_(0, 255)
    return rgb / 255.0 * 2.0 - 1.0


def finish(d: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[B, 3, h, w]` -> depth `[B, H, W]` in `[0, 1]` (float32)."""
    d = (d.mean(dim=1, keepdim=True).clip(-1.0, 1.0) + 1.0) / 2.0
    return F.interpolate(d.float(), size, mode="bilinear", align_corners=False, antialias=True)[:, 0].clip(0, 1)


def infer(net: SD2Net, image: torch.Tensor, params: Params, noise: Noise, ddim: DDIM) -> torch.Tensor:
    """Upstream `single_infer` on one image (the plain modules, in the dtype of `net`)."""
    dtype = net.empty_text.dtype
    x = prepare(image, params.long_side).to(dtype)
    rgb = net.vae.encode_mean(x) * LATENT_SCALE
    (z,) = noise(rgb.shape, rgb.device, dtype)
    ctx = net.empty_text.to(dtype).expand(rgb.shape[0], -1, -1)
    for t in ddim.timesteps:
        v = net.unet(torch.cat([rgb, z], dim=1), torch.tensor(t, device=rgb.device), ctx)
        z = ddim.step(v, t, z)
    return finish(net.vae.decode(z / LATENT_SCALE), image.shape[-2:])


class Portable(Runner):
    """Plain PyTorch (SDPA attention): float32, or float16 weights and activations (upstream `half_precision`)."""

    dtypes = ("float32", "float16")
    batch = 1  # the seeded noise is per image
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = load_net(ctx)
        self.dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.net.to(self.dtype)
        self.noise = Noise(ctx.params.seed)
        with torch.inference_mode(mode=False):  # constants of the traced module()
            self.ddim = DDIM(ctx.params.steps)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return infer(self.net, image, self.ctx.params, self.noise, self.ddim)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant depth of one image."""
        return {"depth": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> depth `[1, H, W]` in `[0, 1]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.net, lambda _net, image: self._run(image))


class Kernels(Runner):
    """float16 channels-last, cuDNN fp16-accumulate convs, constants of the prompt and the timesteps computed once."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert it and precompute the constants."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        self.net = load_net(ctx).half().to(memory_format=torch.channels_last)
        self.ddim = DDIM(ctx.params.steps)
        with torch.inference_mode():
            self.fast = Fast(self.net.unet, self.net.empty_text.half(), self.ddim.timesteps)
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant depth of one image."""
        p, f = self.ctx.params, self.fast
        x = prepare(image, p.long_side).half()
        rgb = f.encode_mean(self.net.vae, x) * LATENT_SCALE
        (z,) = self.noise(rgb.shape, rgb.device, torch.float16)
        for t in self.ddim.timesteps:
            z = self.ddim.step(f.unet_forward(torch.cat([rgb, z], dim=1), t), t, z)
        return {"depth": finish(f.decode(self.net.vae, z / LATENT_SCALE), image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
