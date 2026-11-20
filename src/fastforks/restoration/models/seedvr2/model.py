"""SeedVR2 runners: the converted checkpoints -> network and the authors' one-step restoration of one frame.

The call follows `projects/inference_seedvr2_{3b,7b}.py` with `sp_size = 1` and a one-frame clip: the image resized
by `scale` (bicubic, antialiased, clamped; upstream: torchvision `Resize` of the frame to `scale x` its area, the same
filter), mapped to `[-1, 1]`, VAE posterior sample, one transformer step from `[noise, latent, 1]` at t = 1000,
`noise - velocity`, VAE decode, clamped and mapped back to `[0, 1]`. Upstream then centre-crops the resized frame to
multiples of 16; here the core pads the input on the bottom and right (replicate) so that `scale x` the size is a
multiple of 16 and crops the output back, so every pixel is restored. Both noises (the VAE posterior sample, then the
diffusion noise) come from one CPU generator seeded with `seed` (upstream's default 666, drawn in the same order and
shapes), so a seed and size give the same noise on every device and in every mode.

`stream` (default on, CUDA) keeps the transformer blocks in page-locked host memory and uploads one at a time
(`fastforks.shared.host_blocks`); `quant = "nf4"` quantises the blocks' linear weights to NF4 at load (bitsandbytes'
format, `fastforks.shared.nf4`) and expands each to bf16 per call: ~3.5x smaller transformer, a small quality cost
(measured 2026-11-14). Both are explicit options, printed when used, never silent fallbacks. The kernels
backend expands NF4 weights with one fused kernel (`fastforks.kernels.nf4`); without `quant` it is the portable path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.kernels import nf4
from fastforks.restoration._runner import Forward, Runner
from fastforks.shared.nf4 import code as nf4_code

from .net import BF, CONFIGS, LATENT, SPATIAL, Net
from .net import load as build_net

if TYPE_CHECKING:
    from fastforks.restoration._runner import Context


@dataclass(frozen=True)
class Params:
    """Card `[params]` + variant `params` + `[options]`."""

    scale: int  # output size / input size
    variant: str  # "3b" or "7b": the transformer configuration (`net.dit.CONFIGS`)
    seed: int  # seed of the CPU generator of both noises (upstream's default 666)
    stream: bool  # transformer blocks on the host, uploaded one at a time (small device memory)
    quant: str  # "none" (bf16, as upstream) or "nf4" (the blocks' linear weights in 4 bit)


class Noise:
    """The posterior noise `[1, 16, t, h, w]` and the diffusion noise `[t, h, w, 16]` (bf16), redrawn per shape.

    One CPU generator seeded with `seed` draws both in float32, the posterior noise first, as upstream does with
    its seeded global generator.
    """

    def __init__(self, seed: int) -> None:
        """No noise drawn yet."""
        self.seed = seed
        self.key: tuple[tuple[int, int, int], torch.device] | None = None
        self.eps = self.z = torch.empty(0)

    @torch.compiler.disable
    def __call__(self, size: tuple[int, int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The two noises of latent size `(t, h, w)` on `device`."""
        if (size, device) != self.key:
            with torch.inference_mode(mode=False):
                g = torch.Generator().manual_seed(self.seed)
                self.eps = torch.randn(1, LATENT, *size, generator=g).to(device, BF)
                self.z = torch.randn(*size, LATENT, generator=g).to(device, BF)
            self.key = (size, device)
        return self.eps, self.z


def infer(net: Net, image: torch.Tensor, scale: int, noise: Noise) -> torch.Tensor:
    """One frame `[1, 3, H, W]` in `[0, 1]` (`scale x` H and W multiples of 16) -> `[1, 3, scale H, scale W]`."""
    h, w = image.shape[-2] * scale, image.shape[-1] * scale
    x = F.interpolate(image, (h, w), mode="bicubic", align_corners=False, antialias=True).clamp_(0.0, 1.0)
    eps, z = noise((1, h // SPATIAL, w // SPATIAL), x.device)
    y = net((x * 2.0 - 1.0).to(BF), eps, z)
    return y.float().clamp_(-1.0, 1.0).mul_(0.5).add_(0.5)


def _announce(ctx: Context) -> None:
    if ctx.params.stream and ctx.device.type == "cuda":
        print(f"{ctx.variant.ref}: stream=True, transformer blocks are uploaded per block from pinned host memory")  # noqa: T201
    if ctx.params.quant != "none":
        print(f"{ctx.variant.ref}: quant={ctx.params.quant!r}, transformer block weights quantised at load")  # noqa: T201


def _build(ctx: Context) -> Net:
    dit, vae = ctx.weights()
    p = ctx.params
    return build_net(dit, vae, ctx.device, p.variant, stream=p.stream, quant=p.quant)


class Portable(Runner):
    """Plain PyTorch: bf16 GEMMs, SDPA window attention and convolutions, float32 norms, RoPE and AdaSingle glue."""

    dtypes = ("bfloat16",)
    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        _announce(ctx)
        self.net = _build(ctx)
        self.noise = Noise(ctx.params.seed)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return infer(self.net, image, self.ctx.params.scale, self.noise)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored frame, `scale x` the (padded) input size."""
        return {"image": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> `[1, 3, scale H, scale W]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.net, lambda _net, image: self._run(image))


class Kernels(Runner):
    """Portable with the NF4 expansion (`quant = "nf4"`) as one fused kernel (`nf4`): one read of the packed bytes."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and install the kernel."""
        super().__init__(ctx)
        _announce(ctx)
        self.net = _build(ctx)
        if ctx.params.quant == "nf4":
            nf4.load(ctx.device)
            self.net.dit.dequant = _Dequant(ctx.device)
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored frame, `scale x` the (padded) input size."""
        return {"image": infer(self.net, image, self.ctx.params.scale, self.noise)}


class MlxPortable(Runner):
    """Apple MLX (`device="mlx"`): the transformer and VAE in `mx` operations (`net.mlx_net`).

    Weights are read block by block from the bf16 checkpoints into the run dtype (no PyTorch twin), or with
    `quant = "nf4"` the blocks' linear weights quantised to NF4 as on CUDA; the resize and the noise stay PyTorch.
    """

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Read the checkpoints into MLX."""
        super().__init__(ctx)
        _announce(ctx)
        from fastforks.shared.mlx_weights import DTYPES  # noqa: PLC0415 - needs the mlx extra

        from .net import mlx_net  # noqa: PLC0415

        dit, vae = ctx.weights()
        if ctx.params.quant == "none":
            _require_mlx_ram(ctx.variant.ref, (dit, vae), DTYPES[ctx.dtype].size)
        cfg = CONFIGS[ctx.params.variant]
        self.net = mlx_net.Net(dit, vae, cfg, DTYPES[ctx.dtype], quant=ctx.params.quant == "nf4")
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The restored frame, `scale x` the (padded) input size."""
        return {"image": infer(self.net, image, self.ctx.params.scale, self.noise)}


def _require_mlx_ram(ref: str, paths: tuple[str, ...], itemsize: int) -> None:
    """Fail clearly when the resident weights (the bf16 checkpoints in the run dtype) exceed 80 % of unified memory.

    Without the check a 7 B transformer in bf16 (16.5 GB) on a 16 GB Mac pages the weights to swap on every call.
    """
    from fastforks._core.mlx import mx  # noqa: PLC0415

    need = sum(Path(p).stat().st_size for p in paths) * itemsize // 2
    ram = int(mx.device_info()["memory_size"])
    if need > 0.8 * ram:
        msg = (
            f"{ref}: the weights need {need / 2**30:.1f} GiB resident in MLX, but this Mac has {ram / 2**30:.0f} GiB "
            'of unified memory: pass quant="nf4" (the transformer blocks in 4 bit, ~3.5x smaller) or use a Mac with '
            "more memory"
        )
        raise MemoryError(msg)


class _Dequant:
    """`fastforks.kernels.nf4.dequant` into one reused bf16 buffer per weight size (no allocator churn).

    A buffer is overwritten by the next weight of its size; the GEMM that read it was queued before on the same stream.
    """

    def __init__(self, device: torch.device) -> None:
        self.code = nf4_code(device)
        self.pool: dict[int, torch.Tensor] = {}

    def __call__(self, q: torch.Tensor, absmax: torch.Tensor) -> torch.Tensor:
        n = q.numel() * 2
        out = self.pool.get(n)
        if out is None:
            out = self.pool[n] = torch.empty(n, dtype=BF, device=q.device)
        nf4.dequant(q, absmax, self.code, out)
        return out.view(absmax.shape[0], -1)


RUNNERS = {"portable": Portable, "kernels": Kernels, "portable_mlx": MlxPortable}
