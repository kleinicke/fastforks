"""SeedVR2 without its training framework, flash-attn and apex: the converted checkpoints as plain modules.

`load(ctx, ...)` reads the transformer file straight to the device or, when streaming, packs its blocks into one
page-locked host buffer (`fastforks.shared.host_blocks.HostBlocks`), and the VAE file to the device. `Net` runs the
authors' one-step restoration (`projects/inference_seedvr2_*.py` at cfg 1): VAE posterior sample of the (resized) input
-> one velocity prediction from `[noise, latent, 1]` at t = 1000 -> `noise - velocity` -> VAE decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn

from fastforks._core.checkpoint import SafetensorsReader
from fastforks.shared.host_blocks import READERS, HostBlocks
from fastforks.shared.nf4 import BLOCK, dequant_lut, pair_lut, quantize
from fastforks.shared.nf4 import code as nf4_code

from .dit import BF, CONFIGS, Config, Transformer, block_keys
from .vae import VAE

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ALIGN", "BF", "CONFIGS", "LATENT", "SPATIAL", "TEMPORAL", "Net", "load"]

LATENT = 16
SPATIAL = 8  # pixels per latent
TEMPORAL = 4  # frames per latent frame after the first
ALIGN = SPATIAL * 2  # pixels per transformer token


class Net(nn.Module):
    """VAE + transformer on one clip: `[T, 3, H, W]` in `[-1, 1]` (T = 4n + 1, H and W multiples of 16)."""

    def __init__(self, dit: dict[str, torch.Tensor], vae: dict[str, torch.Tensor], cfg: Config, *, host: HostBlocks | None = None) -> None:
        """Build from the two checkpoints' tensors; `host`: the streamed transformer blocks, see `load`."""
        super().__init__()
        self.dit = Transformer(dit, cfg, host=None if host is None else host.run)
        self.vae = VAE(vae)

    def encode(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Scaled latent `[t, h, w, 16]` (posterior noise `eps` `[1, 16, t, h, w]`)."""
        return self.vae.encode(x.transpose(0, 1)[None], eps)

    def denoise(self, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """The one Euler step from t = 1000: `noise - velocity` (float32 subtraction, rounded to bf16)."""
        ones = torch.ones_like(latent[..., :1])
        pred = self.dit(torch.cat((noise, latent, ones), -1))
        return (noise.float() - pred.float()).to(BF)

    def decode(self, x0: torch.Tensor) -> torch.Tensor:
        """`[t, h, w, 16]` -> `[4t - 3, 3, 8h, 8w]` bf16 (unclamped)."""
        return self.vae.decode(x0)[0].transpose(0, 1)

    def forward(self, x: torch.Tensor, eps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Restored clip, same shape as `x`, bf16 (unclamped); `eps` / `noise`: the posterior and diffusion noise."""
        return self.decode(self.denoise(self.encode(x, eps), noise))



def _is_block(k: str) -> bool:
    return k.startswith("blocks.") and ".ada." not in k


def _quantised(k: str, shape: torch.Size) -> bool:
    """A block's linear weight: NF4 with `quant = "nf4"` (norm weights, biases and the outer layers stay bf16)."""
    return _is_block(k) and k.endswith(".weight") and len(shape) == 2


def _nf4_meta(k: str, shape: torch.Size) -> dict[str, tuple[torch.dtype, torch.Size]]:
    base, (n, kk) = k[: -len(".weight")], shape
    return {base + ".q": (torch.uint8, torch.Size((n * kk // 2,))), base + ".absmax": (BF, torch.Size((n, kk // BLOCK)))}


def read_dit(
    path: Path, device: torch.device, cfg: Config, *, stream: bool, quant: bool
) -> tuple[dict[str, torch.Tensor], HostBlocks | None]:
    """The transformer's tensors on `device`; with `stream` its blocks (not their AdaSingle vectors) on the host.

    `quant`: the blocks' linear weights quantised to NF4 (`fastforks.shared.nf4.quantize`) one tensor at a time on
    `device`, so the bf16 transformer is never resident as a whole.
    """
    with SafetensorsReader(path) as f:  # plain reads: no copy-on-write mapping of the whole file (Windows commit)
        keys = f.keys()
        shapes = {k: f.shape(k) for k in keys}
        t: dict[str, torch.Tensor] = {}
        for k in keys:
            if stream and _is_block(k):
                continue
            v = f.get_tensor(k).to(device)
            if quant and _quantised(k, shapes[k]):
                q, absmax = quantize(v)
                del v
                t[k[: -len(".weight")] + ".q"], t[k[: -len(".weight")] + ".absmax"] = q, absmax
            else:
                t[k] = v
        if not stream:
            return t, None
        meta: dict[str, tuple[torch.dtype, torch.Size]] = {}
        for k in keys:
            if _is_block(k):
                if quant and _quantised(k, shapes[k]):
                    meta |= _nf4_meta(k, shapes[k])
                else:
                    meta[k] = (f.dtype(k), shapes[k])
        last: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def fill(k: str, dst: torch.Tensor) -> None:
            base, part = k.rsplit(".", 1)
            if part not in ("q", "absmax") or k in shapes:
                f.read_into(k, dst)  # straight into the host block buffer
                return
            if base not in last:  # `.q` and `.absmax` of one weight are filled one after the other
                last.clear()
                last[base] = quantize(f.get_tensor(base + ".weight").to(device))
            dst.copy_(last[base][part == "absmax"])

        specs = [{k: meta[k] for k in block_keys(meta, i)} for i in range(cfg.layers)]
        # NF4: `fill` keeps the last quantised weight (not thread-safe); bf16: plain `read_into`, read in parallel
        return t, HostBlocks(specs, fill, workers=1 if quant else READERS)


def read_vae(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """The VAE's tensors on `device`."""
    with safe_open(path, "pt", device=str(device)) as f:
        return {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 - safe_open has no iteration


def load(
    dit: Path, vae: Path, device: torch.device, variant: str, *, stream: bool, quant: str = "none", keep: int = 0
) -> Net:
    """The network on `device`, in eval mode.

    `stream` keeps the transformer blocks on the host (page-locked); `quant = "nf4"` stores the blocks' linear weights
    in NF4 and expands each per call with the plain-PyTorch table gather (`fastforks.shared.nf4.dequant_lut`).
    `keep` (bytes, streaming only): blocks held on the device instead, uploaded once (`HostBlocks.keep`).
    """
    stream = stream and device.type == "cuda"  # page-locked streaming is CUDA only; other devices stay resident
    cfg = CONFIGS[variant]
    if quant not in ("none", "nf4"):
        msg = f"quant must be 'none' or 'nf4', got {quant!r}"
        raise ValueError(msg)
    if keep < 0 or (keep and not stream):
        msg = f"keep must be 0, or positive with stream=True on CUDA (blocks streamed from the host), got {keep} bytes"
        raise ValueError(msg)
    t, host = read_dit(dit, device, cfg, stream=stream, quant=quant == "nf4")
    if host is not None and keep:
        host.keep(keep)
    net = Net(t, read_vae(vae, device), cfg, host=host)
    del t
    if quant == "nf4":
        lut = pair_lut(nf4_code(device))
        net.dit.dequant = lambda q, absmax: dequant_lut(q, absmax, lut, absmax.shape[0])
    return net.eval().requires_grad_(False)
