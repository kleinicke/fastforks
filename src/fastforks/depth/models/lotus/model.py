"""Lotus normals runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `LotusGPipeline` / `LotusDPipeline.__call__` with `infer.py`'s settings (one step at timestep 999,
the empty prompt, the normals task switcher): map to `[-1, 1]`, resize the longest side to `long_side` (bilinear,
antialiased, sides truncated as upstream's float arithmetic does), encode with the VAE (a posterior sample x 0.18215),
run the UNet once (sample prediction) on the image latent (Lotus-D) or on the image latent next to a Gaussian latent
(Lotus-G), decode, map to `[0, 1]` (clamped), resize to the input size (nearest-exact) and return `2 x - 1` normalised.
Lotus' normals point x left, y up, z towards the camera (its Hypersim training normals: `x *= -1` only); they are
returned in the camera frame (x right, y down, z forward).

The noise is one tensor from a CPU generator seeded with `seed` (float32, then moved to the device), used both as the
Lotus-G latent and as the posterior sample's noise: upstream draws the latent from its seeded generator and the
posterior noise from the global RNG, both seeded with the same seed by `infer.py --seed` (`seed_all`), so for the first
image they are the same tensor (CPU). A CUDA generator's values differ. The noise is drawn eagerly (`Noise`) outside the
traced graph, so `module()` compiles and exports; a traced or exported module reuses the noise of its traced shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.sd2 import LATENT_SCALE, Noise, SD2Net, load_net
from fastforks.shared.sd2.net.fast import Fast
from fastforks.shared.sd2.net.vae import sample

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

TIMESTEP = 999  # fixed: upstream infer.py / eval.py `--timestep` default, the one the models were trained at
TASK = (math.sin(1.0), math.sin(0.0), math.cos(1.0), math.cos(0.0))  # fixed: upstream's normals task switcher
_CAMERA = (-1.0, -1.0, -1.0)  # fixed: Lotus' (x left, y up, z to the viewer) -> camera (x right, y down, z forward)


@dataclass(frozen=True)
class Params:
    """Load options."""

    generative: bool  # Lotus-G: the UNet also gets a noise latent (8 input channels); Lotus-D: the image latent only
    long_side: int  # longest side of the network input; 0 = the input size
    seed: int  # noise seed


def network_size(h: int, w: int, res: int) -> tuple[int, int]:
    """Upstream `resize_max_res`: the longest side becomes `res`, sides truncated (in its float arithmetic)."""
    f = min(res / w, res / h)
    new_h, new_w = int(h * f), int(w * f)
    if not (new_h and new_w):
        msg = f"extreme aspect ratio of the input image: {w} x {h}"
        raise ValueError(msg)
    return new_h, new_w


def prepare(image: torch.Tensor, res: int) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> the network input in `[-1, 1]` (not padded: the VAE floors odd sizes)."""
    x = image * 2.0 - 1.0
    if res > 0:
        x = F.interpolate(x, network_size(*image.shape[-2:], res), mode="bilinear", antialias=True)
    return x


def finish(n: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[B, 3, h, w]` -> unit normals `[B, H, W, 3]` in the camera frame (float32)."""
    n = (n.float() / 2 + 0.5).clamp(0.0, 1.0)  # upstream's image postprocessing
    if tuple(n.shape[-2:]) != tuple(size):
        n = F.interpolate(n, size, mode="nearest-exact")  # upstream maps "nearest" to NEAREST_EXACT
    n = n * 2.0 - 1.0
    n = n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp(min=1e-6) * n.new_tensor(_CAMERA)[:, None, None]
    return n.permute(0, 2, 3, 1)


def _net(ctx: Context) -> SD2Net:
    return load_net(ctx, cin=8 if ctx.params.generative else 4, class_embed=len(TASK))


def infer(net: SD2Net, image: torch.Tensor, params: Params, noise: Noise) -> torch.Tensor:
    """The pipeline on one image (the plain modules, in the dtype of `net`)."""
    dtype = net.empty_text.dtype
    moments = net.vae.encode_moments(prepare(image, params.long_side).to(dtype))
    (eps,) = noise(torch.Size((moments.shape[0], moments.shape[1] // 2, *moments.shape[2:])), moments.device, dtype)
    rgb = sample(moments, eps) * LATENT_SCALE
    x = torch.cat([rgb, eps], dim=1) if params.generative else rgb
    ctx = net.empty_text.to(dtype).expand(rgb.shape[0], -1, -1)
    labels = torch.tensor([TASK], device=rgb.device, dtype=dtype).expand(rgb.shape[0], -1)
    pred = net.unet(x, torch.tensor(TIMESTEP, device=rgb.device), ctx, labels)
    return finish(net.vae.decode(pred / LATENT_SCALE), image.shape[-2:])


class Portable(Runner):
    """Plain PyTorch (SDPA attention): float32, or float16 weights and activations."""

    dtypes = ("float32", "float16")
    batch = 1  # the seeded noise is per image
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        self.dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.net.to(self.dtype)
        self.noise = Noise(ctx.params.seed)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return infer(self.net, image, self.ctx.params, self.noise)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        return {"normals": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> normals `[1, H, W, 3]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.net, lambda _net, image: self._run(image))


class Kernels(Runner):
    """float16 channels-last, cuDNN fp16-accumulate convs, constants of the prompt, timestep and task computed once."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert it and precompute the constants."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        self.net = _net(ctx).half().to(memory_format=torch.channels_last)
        with torch.inference_mode():
            labels = torch.tensor([TASK], device=ctx.device)
            self.fast = Fast(self.net.unet, self.net.empty_text.half(), [TIMESTEP], labels)
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        p, f = self.ctx.params, self.fast
        moments = f.encode_moments(self.net.vae, prepare(image, p.long_side).half())
        (eps,) = self.noise(
            torch.Size((moments.shape[0], moments.shape[1] // 2, *moments.shape[2:])), moments.device, torch.half
        )
        rgb = sample(moments, eps) * LATENT_SCALE
        pred = f.unet_forward(torch.cat([rgb, eps], dim=1) if p.generative else rgb, TIMESTEP)
        return {"normals": finish(f.decode(self.net.vae, pred / LATENT_SCALE), image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
