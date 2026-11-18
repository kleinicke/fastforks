"""Marigold V2 runners: the lean checkpoint -> network and the portable / kernels forward passes.

Both follow the authors' `run.py` (depth, `Log-stage2`): the image in `[-1, 1]`, each side enlarged to the next
multiple of 16 (upstream: OpenCV Lanczos; here bicubic, antialiased: the rare enlargement is at most 15 px per
side), VAE posterior sample (upstream `latent_dist.sample()`; the noise comes from a CPU generator seeded with `seed`,
float32, so a seed and input size give the same noise on every device and in every mode), one rectified-flow step at
the checkpoint's fixed timestep, `latent - velocity`, VAE decode, mean of the three channels, resized back to the input
size (bilinear). The output is log depth up to an unknown scale and shift (`kind = "affine_invariant_log"`), in
`[-1, 1]` as the decoder clamps it.

`long_side` (ours) scales the longest side first; `stream` (default on, CUDA only) keeps the 60 transformer blocks
pinned on the host and uploads one at a time (~1-3 GB of device memory instead of ~9.5 GB, ~0.4 % time); printed when
on, off with `stream=False`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.runner import host
from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import nf4

from .net import load as build_net
from .net.nf4 import NF4Linear

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.depth._runner import Context

MULTIPLE = 16  # fixed: the VAE's 8 x the 2 x 2 latent patching of the transformer
LATENT_CHANNELS = 16  # fixed: the Qwen-Image VAE latent
VAE_SCALE = 8  # fixed: the Qwen-Image VAE downsampling


@dataclass(frozen=True)
class Params:
    """Load options."""

    seed: int  # seed of the VAE posterior sample
    long_side: int  # longest side of the network input before rounding up to a multiple of 16; 0 = the input size
    stream: bool  # transformer blocks on the host, uploaded one at a time (small device memory)
    exact_mask: bool  # upstream's all-true attention mask (bit-equal to the authors' code, slower)


def network_size(h: int, w: int, long_side: int) -> tuple[int, int]:
    """The network input size: the longest side scaled to `long_side` (if > 0), both sides rounded up to 16."""
    if long_side > 0:
        f = long_side / max(h, w)
        h, w = max(1, round(h * f)), max(1, round(w * f))
    return math.ceil(h / MULTIPLE) * MULTIPLE, math.ceil(w / MULTIPLE) * MULTIPLE


def prepare(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> network input in `[-1, 1]` (float32)."""
    size = network_size(*image.shape[-2:], long_side)
    if size != tuple(image.shape[-2:]):
        image = F.interpolate(image, size, mode="bicubic", align_corners=False, antialias=True).clamp_(0.0, 1.0)
    return image * 2.0 - 1.0


def finish(d: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[B, 1, h, w]` -> log depth `[B, H, W]` (float32)."""
    d = d.float()
    if tuple(d.shape[-2:]) != tuple(size):
        d = F.interpolate(d, size, mode="bilinear", align_corners=False)
    return d[:, 0]


class Noise:
    """Seeded standard-normal latent noise from a CPU generator (float32), redrawn when the shape changes."""

    def __init__(self, seed: int) -> None:
        """No noise drawn yet."""
        self.seed = seed
        self.key: tuple[tuple[int, ...], torch.device] | None = None
        self.z = torch.empty(0)

    @torch.compiler.disable
    def __call__(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """The noise of `shape` on `device`."""
        if (shape, device) != self.key:
            with torch.inference_mode(mode=False):
                g = torch.Generator().manual_seed(self.seed)
                self.z = torch.randn(shape, generator=g, dtype=torch.float32).to(device)
            self.key = (shape, device)
        return self.z


def infer(
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], image: torch.Tensor, params: Params, noise: Noise
) -> torch.Tensor:
    """The authors' single-step inference on `image [B, 3, H, W]` in `[0, 1]` -> log depth `[B, H, W]`."""
    x = prepare(image, params.long_side)
    b, h, w = x.shape[0], x.shape[-2] // VAE_SCALE, x.shape[-1] // VAE_SCALE
    z = noise((b, LATENT_CHANNELS, h, w), x.device)
    return finish(predict(x, z), image.shape[-2:])


def _announce(ctx: Context) -> None:
    if ctx.params.stream and ctx.device.type == "cuda":
        print(f"{ctx.variant.ref}: stream=True, transformer blocks are uploaded per block from pinned host memory")  # noqa: T201


class Portable(Runner):
    """Plain PyTorch: NF4 weights expanded per linear by a table gather, bf16 GEMMs, float32 glue."""

    dtypes = ("bfloat16",)
    compiled_methods = ("_encode", "_decode")
    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        _announce(ctx)
        self.net = build_net(ctx, stream=ctx.params.stream, exact_mask=ctx.params.exact_mask)
        self.noise = Noise(ctx.params.seed)

    def _encode(self, image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.net.vae.encode(image, noise)

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net.vae.decode(latent)

    def _predict(self, image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.net.forward_with_vae(image, noise, encode=self._encode, decode=self._decode)

    @host
    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant log depth of one image."""
        return {"depth": infer(self._predict, image, self.ctx.params, self.noise)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> log depth `[1, H, W]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.net, lambda net, image: infer(net, image, self.ctx.params, self.noise))


class Kernels(Runner):
    """Portable with the NF4 expansion as one fused kernel (`nf4`): one read of the packed bytes, one bf16 write."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and install the kernel."""
        super().__init__(ctx)
        _announce(ctx)
        nf4.load(ctx.device)
        NF4Linear.dequant, NF4Linear.epi = _dequant, nf4
        self.net = build_net(ctx, stream=ctx.params.stream, fused=True, exact_mask=ctx.params.exact_mask)
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant log depth of one image."""
        return {"depth": infer(self.net, image, self.ctx.params, self.noise)}


def _dequant(lin: NF4Linear) -> torch.Tensor:
    q, absmax, bias = (cast("torch.Tensor", t) for t in (lin.q, lin.absmax, lin.bias))  # registered buffers
    # A linear consumes the expansion on the current stream. PyTorch's allocator
    # reuses the storage after the GEMM and isolates concurrent inference streams.
    out = torch.empty(q.numel() * 2, dtype=torch.bfloat16, device=q.device)
    nf4.dequant(q, absmax, cast("torch.Tensor", NF4Linear.code), out)
    return out.view(bias.numel(), -1)


RUNNERS = {"portable": Portable, "kernels": Kernels}
