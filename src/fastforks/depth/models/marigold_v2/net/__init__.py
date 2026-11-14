"""Marigold V2 depth without diffusers, bitsandbytes and peft: the lean checkpoint's tensors as plain modules.

`load(ctx, stream=...)` reads the one safetensors file straight to the device (no host copy of the 8.4 GB), or, when
streaming, keeps the transformer blocks pinned on the host. `Net` runs the authors' single-step inference:
VAE posterior sample -> 2 x 2 patch packing -> one velocity prediction at the fixed timestep -> `latent - velocity` ->
VAE decode -> mean of the three channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn

from .dit import Block, Transformer, block_keys
from .nf4 import BF, NF4Linear, pair_lut
from .vae import VAE2D

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

__all__ = ["BF", "Block", "NF4Linear", "Net", "Transformer", "load"]

VAE_SCALE = 8  # pixels per latent
PATCH = 2  # latent pixels per transformer token


class Net(nn.Module):
    """VAE + transformer; `forward(x, noise)` on `[B, 3, H, W]` in `[-1, 1]` (H, W multiples of 16) -> `[B, 1, H, W]`."""

    def __init__(
        self, t: dict[str, torch.Tensor], *, host: list[dict[str, torch.Tensor]] | None = None, fused: bool = False
    ) -> None:
        """Build from the checkpoint tensors.

        `host`: the transformer blocks' tensors (streaming), see `load`; `fused`: expand the NF4 weights with the
        kernel `NF4Linear.dequant` (the kernels backend sets it).
        """
        super().__init__()
        self.register_buffer("nf4_code", t["nf4_code"].float())
        self.vae = VAE2D(t)
        self.dit = Transformer(t, host=host, fused=fused)

    def lut(self) -> None:
        """Set the NF4 code tables of this device (after moving the module)."""
        if NF4Linear.code is None or NF4Linear.code.device != self.nf4_code.device:
            NF4Linear.code = self.nf4_code
            NF4Linear.lut = pair_lut(self.nf4_code)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Log depth (affine-invariant) in `[-1, 1]`, bf16, at the network size."""
        self.lut()
        lat = self.vae.encode(x, noise)
        b, c, h, w = lat.shape
        hp, wp = h // PATCH, w // PATCH
        packed = lat.view(b, c, hp, PATCH, wp, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, hp * wp, c * PATCH * PATCH)
        pred = self.dit(packed, hp, wp)
        pred = pred.view(b, hp, wp, c, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)
        return self.vae.decode(lat - pred).mean(dim=1, keepdim=True)


def _read(ctx: Context, device: torch.device, *, stream: bool) -> tuple[dict[str, torch.Tensor], list[dict] | None]:
    if ctx.user_state is not None:
        t = dict(ctx.user_state)
        if stream:
            host, rest = [], {}
            n = 1 + max(int(k.split(".")[1]) for k in t if k.startswith("transformer_blocks."))
            for i in range(n):
                host.append({k: t[k].pin_memory() for k in block_keys(list(t), i)})
            rest = {k: v.to(device) for k, v in t.items() if not k.startswith("transformer_blocks.")}
            return rest, host
        return {k: v.to(device) for k, v in t.items()}, None
    path = ctx.weights()[0]
    t, host = {}, {}
    with safe_open(path, "pt", device="cpu" if stream else str(device)) as f:
        for k in f.keys():  # noqa: SIM118 - safetensors handle, not a dict
            if stream and k.startswith("transformer_blocks."):
                host.setdefault(int(k.split(".")[1]), {})[k] = f.get_tensor(k).pin_memory()
            else:
                t[k] = f.get_tensor(k).to(device)
    return t, [host[i] for i in range(len(host))] if stream else None


def load(ctx: Context, *, stream: bool, fused: bool = False) -> Net:
    """The network on `ctx.device`, in eval mode; `stream` keeps the transformer blocks on the host (pinned)."""
    if stream and ctx.device.type != "cuda":
        msg = "stream=True needs a CUDA device"
        raise ValueError(msg)
    t, host = _read(ctx, ctx.device, stream=stream)
    net = Net(t, host=host, fused=fused).to(ctx.device)
    del t
    return net.eval()
