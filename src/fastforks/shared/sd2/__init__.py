"""Stable Diffusion 2 UNet, VAE and CLIP text encoder, re-implemented without diffusers (Marigold, Lotus).

`SD2Net` holds the UNet, the VAE and the empty-prompt text embedding; `load_net` builds it from a checkpoint's UNet
(file 0), VAE (file 1) and text encoder (file 2, run once on the CPU in float32). `Noise` draws the seeded initial
latent (and any per-step noise) from a CPU generator outside the traced graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .net.clip_text import CLIPTextTransformer
from .net.unet import UNet2DConditionModel
from .net.vae import AutoencoderKL

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

LATENT_SCALE = 0.18215  # fixed: the Stable Diffusion VAE latent scale, hard-coded upstream
EMPTY_PROMPT = [49406, 49407]  # CLIPTokenizer("", padding="do_not_pad").input_ids: <|startoftext|>, <|endoftext|>


class SD2Net(torch.nn.Module):
    """UNet + VAE + the empty-prompt text embedding `[1, 2, 1024]` (a buffer)."""

    empty_text: torch.Tensor

    def __init__(self, **unet: Any) -> None:  # noqa: ANN401 - UNet2DConditionModel arguments
        """Random weights."""
        super().__init__()
        self.unet = UNet2DConditionModel(**unet)
        self.vae = AutoencoderKL()
        self.register_buffer("empty_text", torch.zeros(1, len(EMPTY_PROMPT), 1024), persistent=False)


def load_net(ctx: Context, **unet: Any) -> SD2Net:  # noqa: ANN401 - UNet2DConditionModel arguments
    """The checkpoint's UNet (file 0) and VAE (file 1); the text encoder (file 2) runs once, on the CPU, in float32."""
    text = CLIPTextTransformer()
    text.load_state_dict(
        {k: v for k, v in ctx.state_dict(2, strip="text_model.").items() if k != "embeddings.position_ids"}
    )
    with torch.no_grad():
        embed = text.eval()(torch.tensor([EMPTY_PROMPT]))
    del text
    with torch.device(ctx.device):
        net = SD2Net(**unet)
    net.unet.load_state_dict(ctx.state_dict(0, strip=""))
    net.vae.load_state_dict(ctx.state_dict(1, strip=""))
    net.empty_text.copy_(embed)
    return net.eval()


class Noise:
    """Seeded noise: `draws` tensors in turn from one CPU generator (float32), drawn when shape, device or dtype change.

    The tensors of the last key are kept and returned as is (callers must not modify them). They are drawn outside
    `torch.inference_mode`, so a module traced after an eager call can hold them as constants.
    """

    def __init__(self, seed: int, draws: int = 1) -> None:
        """No noise drawn yet."""
        self.seed, self.draws = seed, draws
        self.key: tuple[tuple[int, ...], torch.device, torch.dtype] | None = None
        self.z: tuple[torch.Tensor, ...] = ()

    def __call__(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        """The `draws` noise tensors of `shape` on `device` in `dtype`."""
        key = (tuple(shape), device, dtype)
        if key != self.key:
            self.z = self._draw(shape, device, dtype)
            self.key = key
        return self.z

    @torch.compiler.disable
    def _draw(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        with torch.inference_mode(mode=False):
            g = torch.Generator().manual_seed(self.seed)
            return tuple(
                torch.randn(shape, generator=g, dtype=torch.float32).to(device, dtype) for _ in range(self.draws)
            )
