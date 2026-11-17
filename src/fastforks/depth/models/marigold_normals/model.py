"""Marigold normals LCM runners: weights -> network and the portable / kernels forward passes.

Both follow diffusers `MarigoldNormalsPipeline.__call__` with `ensemble_size=1` and `use_full_z_range`: map to
`[-1, 1]`, resize the longest side to `long_side` (bilinear, antialiased, sides truncated), pad the bottom / right to a
multiple of 8 (replicate), encode with the VAE (posterior mean x 0.18215), denoise a Gaussian latent in `steps` LCM
steps (v-prediction) with the UNet conditioned on the image latent and the empty-prompt CLIP embedding, decode, clip to
`[-1, 1]`, normalise, remove the padding, resize to the input size (bilinear, not antialiased) and normalise again.
Marigold's normals point x right, y up, z towards the camera; they are returned in the camera frame (y down, z forward).

The initial latent and the noise of every step but the last are drawn in that order from one CPU generator seeded with
`seed` (float32, then moved to the device): the values equal diffusers' with a seeded CPU generator (float32), but
differ from a CUDA generator's. The noise is drawn eagerly (`Noise`) and kept for the last latent shape, outside the
traced graph, so `module()` compiles and exports; a traced or exported module reuses the noise of its traced shape.
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

from .net.lcm import LCM

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

_ALIGN = 8  # fixed: the VAE downsampling factor; the input is padded to a multiple of it
_CAMERA = (1.0, -1.0, -1.0)  # fixed: Marigold's (x right, y up, z to the viewer) -> camera (y down, z forward)


@dataclass(frozen=True)
class Params:
    """Load options."""

    steps: int  # LCM denoising steps
    long_side: int  # longest side of the network input; 0 = the input size
    seed: int  # noise seed


def network_size(h: int, w: int, res: int) -> tuple[int, int]:
    """The diffusers `resize_to_max_edge`: the longest side becomes `res` (sides truncated)."""
    new_h, new_w = h * res // max(h, w), w * res // max(h, w)
    if not (new_h and new_w):
        msg = f"extreme aspect ratio of the input image: {w} x {h}"
        raise ValueError(msg)
    return new_h, new_w


def prepare(image: torch.Tensor, res: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """`[B, 3, H, W]` in `[0, 1]` -> the padded network input in `[-1, 1]` (float32) and the padding (bottom, right)."""
    x = image * 2.0 - 1.0
    if res > 0:
        x = F.interpolate(x, network_size(*image.shape[-2:], res), mode="bilinear", antialias=True)
    ph, pw = -x.shape[-2] % _ALIGN, -x.shape[-1] % _ALIGN
    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (ph, pw)


def _unit(n: torch.Tensor) -> torch.Tensor:
    """The diffusers `normalize_normals` (norm clamped at 1e-6)."""
    return n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp(min=1e-6)


def finish(n: torch.Tensor, pad: tuple[int, int], size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[B, 3, h, w]` -> unit normals `[B, H, W, 3]` in the camera frame (float32)."""
    ph, pw = pad
    n = _unit(n.float().clip(-1.0, 1.0))[:, :, : -ph or None, : -pw or None]
    if tuple(n.shape[-2:]) != tuple(size):
        n = F.interpolate(n, size, mode="bilinear")
    n = _unit(n) * n.new_tensor(_CAMERA)[:, None, None]
    return n.permute(0, 2, 3, 1)


def infer(net: SD2Net, image: torch.Tensor, params: Params, noise: Noise, lcm: LCM) -> torch.Tensor:
    """The pipeline on one image (the plain modules, in the dtype of `net`)."""
    dtype = net.empty_text.dtype
    x, pad = prepare(image, params.long_side)
    rgb = net.vae.encode_mean(x.to(dtype)) * LATENT_SCALE
    z, *step_noise = noise(rgb.shape, rgb.device, dtype)
    ctx = net.empty_text.to(dtype).expand(rgb.shape[0], -1, -1)
    for i, t in enumerate(lcm.timesteps):
        v = net.unet(torch.cat([rgb, z], dim=1), torch.tensor(t, device=rgb.device), ctx)
        z = lcm.step(v, t, z, step_noise[i] if i < len(step_noise) else None)
    return finish(net.vae.decode(z / LATENT_SCALE), pad, image.shape[-2:])


class Portable(Runner):
    """Plain PyTorch (SDPA attention): float32, or float16 weights and activations."""

    dtypes = ("float32", "float16")
    batch = 1  # the seeded noise is per image
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = load_net(ctx)
        self.dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.net.to(self.dtype)
        self.noise = Noise(ctx.params.seed, draws=ctx.params.steps)
        with torch.inference_mode(mode=False):  # constants of the traced module()
            self.lcm = LCM(ctx.params.steps)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return infer(self.net, image, self.ctx.params, self.noise, self.lcm)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        return {"normals": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> normals `[1, H, W, 3]`.

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
        self.lcm = LCM(ctx.params.steps)
        with torch.inference_mode():
            self.fast = Fast(self.net.unet, self.net.empty_text.half(), self.lcm.timesteps)
        self.noise = Noise(ctx.params.seed, draws=ctx.params.steps)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        p, f = self.ctx.params, self.fast
        x, pad = prepare(image, p.long_side)
        rgb = f.encode_mean(self.net.vae, x.half()) * LATENT_SCALE
        z, *step_noise = self.noise(rgb.shape, rgb.device, torch.float16)
        for i, t in enumerate(self.lcm.timesteps):
            v = f.unet_forward(torch.cat([rgb, z], dim=1), t)
            z = self.lcm.step(v, t, z, step_noise[i] if i < len(step_noise) else None)
        return {"normals": finish(f.decode(self.net.vae, z / LATENT_SCALE), pad, image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
