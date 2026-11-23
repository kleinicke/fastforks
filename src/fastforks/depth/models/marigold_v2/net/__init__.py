"""Marigold V2 depth without diffusers, bitsandbytes and peft: the lean checkpoint's tensors as plain modules.

`load(ctx, stream=...)` reads the one safetensors file straight to the device (no host copy of the 8.4 GB), or, when
streaming, packs the transformer blocks into one page-locked host buffer (`fastforks.shared.host_blocks.HostBlocks`). `Net` runs the authors' single-step inference:
VAE posterior sample -> 2 x 2 patch packing -> one velocity prediction at the fixed timestep -> `latent - velocity` ->
VAE decode -> mean of the three channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from .dit import Block, Transformer, block_keys
from .nf4 import BF, NF4Linear, pair_lut
from fastforks._core.checkpoint import SafetensorsReader
from fastforks.shared.host_blocks import READERS, HostBlocks
from .vae import VAE2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks.depth._runner import Context

__all__ = ["BF", "Block", "NF4Linear", "Net", "Transformer", "load"]

VAE_SCALE = 8  # pixels per latent
PATCH = 2  # latent pixels per transformer token


class Net(nn.Module):
    """VAE + transformer; `forward(x, noise)` on `[B, 3, H, W]` in `[-1, 1]` (H, W multiples of 16) -> `[B, 1, H, W]`."""

    def __init__(
        self,
        t: dict[str, torch.Tensor],
        *,
        host: HostBlocks | None = None,
        fused: bool = False,
        exact_mask: bool = False,
    ) -> None:
        """Build from the checkpoint tensors.

        `host`: the streamed transformer blocks, see `load`; `fused`: expand the NF4 weights with the kernel
        `NF4Linear.dequant` (the kernels backend sets it); `exact_mask`: see `dit.Block` (also keeps the VAE's
        full-size convolutions, see `VAE2D`).
        """
        super().__init__()
        self.register_buffer("nf4_code", t["nf4_code"].float())
        self.vae = VAE2D(t, bands=not exact_mask)
        self.dit = Transformer(t, host=host, fused=fused, exact_mask=exact_mask)

    def lut(self) -> None:
        """Set the NF4 code tables of this device (after moving the module)."""
        if NF4Linear.code is None or NF4Linear.code.device != self.nf4_code.device:
            NF4Linear.code = self.nf4_code
            NF4Linear.lut = pair_lut(self.nf4_code)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Log depth (affine-invariant) in `[-1, 1]`, bf16, at the network size."""
        return self.forward_with_vae(x, noise, encode=self.vae.encode, decode=self.vae.decode)

    def forward_with_vae(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        *,
        encode: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decode: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Run the streamed transformer between the supplied VAE encoder and decoder."""
        self.lut()
        lat = encode(x, noise)
        b, c, h, w = lat.shape
        hp, wp = h // PATCH, w // PATCH
        packed = lat.view(b, c, hp, PATCH, wp, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, hp * wp, c * PATCH * PATCH)
        pred = self.dit(packed, hp, wp)
        pred = pred.view(b, hp, wp, c, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return decode(lat - pred).mean(dim=1, keepdim=True)


def _block_index(k: str) -> int | None:
    return int(k.split(".")[1]) if k.startswith("transformer_blocks.") else None


def _host_blocks(
    keys: list[str], meta: dict[str, tuple[torch.dtype, torch.Size]], fill: Callable[[str, torch.Tensor], object],
    *, workers: int = 1,
) -> HostBlocks:
    n = 1 + max(i for k in keys if (i := _block_index(k)) is not None)
    specs = [{k: meta[k] for k in block_keys(keys, i)} for i in range(n)]
    return HostBlocks(specs, fill, workers=workers)


def _read(ctx: Context, device: torch.device, *, stream: bool) -> tuple[dict[str, torch.Tensor], HostBlocks | None]:
    if ctx.user_state is not None:
        t = dict(ctx.user_state)
        host = None
        if stream:
            keys = list(t)
            host = _host_blocks(keys, {k: (v.dtype, v.shape) for k, v in t.items()}, lambda k, dst: dst.copy_(t[k]))
        rest = {k: v.to(device) for k, v in t.items() if not (stream and _block_index(k) is not None)}
        return rest, host
    path = ctx.weights()[0]
    with SafetensorsReader(path) as f:  # plain reads: no copy-on-write mapping of the whole file (Windows commit)
        keys = f.keys()
        t = {k: f.get_tensor(k).to(device) for k in keys if not (stream and _block_index(k) is not None)}
        if not stream:
            return t, None
        meta = {k: (f.dtype(k), f.shape(k)) for k in keys if _block_index(k) is not None}
        return t, _host_blocks(keys, meta, f.read_into, workers=READERS)  # straight into the host block buffer



def load(ctx: Context, *, stream: bool, fused: bool = False, exact_mask: bool = False) -> Net:
    """The network on `ctx.device`, in eval mode; `stream` keeps the transformer blocks on the host (page-locked)."""
    stream = stream and ctx.device.type == "cuda"  # page-locked streaming is CUDA only; other devices stay resident
    t, host = _read(ctx, ctx.device, stream=stream)
    net = Net(t, host=host, fused=fused, exact_mask=exact_mask).to(ctx.device)
    del t
    return net.eval()
