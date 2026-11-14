"""DepthFM runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `DepthFM.forward` as `inference.py` calls it: resize the input (the longest side to `long_side`,
or the input size; both sides rounded to multiples of 64; Pillow's 8-bit bilinear, bit-exact, as upstream resizes the
PIL image), map to `[-1, 1]`, encode with the Stable Diffusion 1.5 VAE (posterior mode x 0.18215), noise the latent to
the checkpoint's `noising_step` (cosine schedule), integrate the flow from t = 0 to 1 in `steps` Euler steps with the
UNet conditioned on the image latent (concatenated) and the checkpoint's empty-prompt embedding, decode, average the
three channels and the `ensemble` members, take `exp` and min-max normalise to `[0, 1]`, resize to the input size
(bilinear, antialiased).

Upstream encodes the image once per ensemble member; the latent is the same, so it is encoded once here. The noise is
`torch.randn` `[ensemble, 4, h/8, w/8]` from a CPU generator seeded with `seed` (float32), as `Noise` draws it; upstream
draws `torch.randn_like` on the device. The flow state stays float32 in every mode (upstream: the autocast dtype).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import NUMPY_ARRAYS, config_globals
from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.pil_resize import pil_bilinear
from fastforks.shared.sd2 import LATENT_SCALE, Noise, SD2Net, old_attention_names
from fastforks.shared.sd2.net.fast import Fast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.depth._runner import Context

_MULTIPLE = 64  # fixed: upstream `resize_max_res` rounds both sides to multiples of 64
_DIFFUSION_STEPS = 1000  # fixed: upstream `q_sample(n_diffusion_timesteps=1000)`
_HEADS = (8, 8, 8, 8)  # fixed: the checkpoint's `ldm_hparams` (num_heads 8 at every level)
_RESNET = {  # fixed: LDM `ResBlock` -> diffusers `ResnetBlock2D` names
    "in_layers.0.": "norm1.",
    "in_layers.2.": "conv1.",
    "emb_layers.1.": "time_emb_proj.",
    "out_layers.0.": "norm2.",
    "out_layers.3.": "conv2.",
    "skip_connection.": "conv_shortcut.",
}


@dataclass(frozen=True)
class Params:
    """Load options."""

    steps: int  # Euler steps of the flow
    ensemble: int  # noise samples averaged
    long_side: int  # longest side of the network input before rounding to 64; 0 = the input size
    seed: int  # initial-noise seed


def _resnet(rest: str) -> str:
    for old, new in _RESNET.items():
        if rest.startswith(old):
            return new + rest.removeprefix(old)
    return rest


def _ldm_key(part: list[str]) -> str:
    head, rest = part[0], _resnet(".".join(part[3:]))
    if head == "input_blocks":
        level, j = divmod(int(part[1]) - 1, 3)
        if j == 2:  # noqa: PLR2004 - the third block of a level is the downsampler
            return f"down_blocks.{level}.downsamplers.0.conv.{part[-1]}"
        return f"down_blocks.{level}.{'resnets' if part[2] == '0' else 'attentions'}.{j}.{rest}"
    if head == "output_blocks":
        level, j = divmod(int(part[1]), 3)
        if part[2] != "0" and part[3] == "conv":  # the upsampler (after the resnet, or after the attention)
            return f"up_blocks.{level}.upsamplers.0.conv.{part[-1]}"
        return f"up_blocks.{level}.{'resnets' if part[2] == '0' else 'attentions'}.{j}.{rest}"
    sub = {"0": "resnets.0", "1": "attentions.0", "2": "resnets.1"}[part[1]]  # middle_block
    return f"mid_block.{sub}.{_resnet('.'.join(part[2:]))}"


def ldm_unet_keys(sd: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """An LDM `openaimodel.UNetModel` state dict (SD2 layout: 4 levels of 2 resnets) -> `UNet2DConditionModel` names."""
    out = {}
    for key, v in sd.items():
        part = key.split(".")
        if part[0] == "time_embed":
            name = f"time_embedding.linear_{1 + int(part[1]) // 2}.{part[2]}"
        elif part[0] == "out":
            name = f"{'conv_norm_out' if part[1] == '0' else 'conv_out'}.{part[2]}"
        elif part[0] == "input_blocks" and part[1] == "0":
            name = f"conv_in.{part[3]}"
        else:
            name = _ldm_key(part)
        out[name] = v
    return out


def build(ctx: Context) -> tuple[SD2Net, float]:
    """The checkpoint's UNet and empty-prompt embedding (file 0) and the VAE (file 1); also sqrt(alpha_bar)."""
    ckpt = cast("dict[str, object]", ctx.state_dict(0, strip="", extra={**NUMPY_ARRAYS, **config_globals()}))
    text = cast("torch.Tensor", ckpt["empty_text_embedding"])
    with torch.device(ctx.device):
        net = SD2Net(text.shape[1], heads=_HEADS)
    unet = cast("dict[str, torch.Tensor]", ckpt["state_dict"])
    net.unet.load_state_dict(ldm_unet_keys(unet))
    net.vae.load_state_dict({old_attention_names(k): v for k, v in ctx.state_dict(1, strip="").items()})
    net.empty_text.copy_(text)
    t = cast("int", ckpt["noising_step"]) / _DIFFUSION_STEPS
    alpha_bar = 1 / (1 + math.exp(2 * math.log(math.tan(math.pi * t / 2) + 1e-5)))  # upstream `cosine_alpha_bar`
    return net.eval(), alpha_bar


def network_size(h: int, w: int, res: int) -> tuple[int, int]:
    """Upstream `resize_max_res`: the longest side to `res` (input size for 0; truncated), both rounded to 64."""
    f = min(res / w, res / h) if res > 0 else 1.0
    return max(_MULTIPLE, round(int(h * f) / _MULTIPLE) * _MULTIPLE), max(
        _MULTIPLE, round(int(w * f) / _MULTIPLE) * _MULTIPLE
    )


def prepare(image: torch.Tensor, res: int) -> torch.Tensor:
    """`[1, 3, H, W]` in `[0, 1]` -> network input in `[-1, 1]` (float32)."""
    size = network_size(*image.shape[-2:], res)
    rgb = image.float().mul(255).round_()  # upstream reads 8-bit images
    if size != tuple(image.shape[-2:]):
        rgb = pil_bilinear(rgb, size)
    return rgb / 127.5 - 1.0


def times(steps: int) -> list[float]:
    """The Euler grid of upstream's `odeint(method="euler", step_size=1 / steps)` on [0, 1], end point included."""
    return [k / steps for k in range(steps)] + [1.0]


def finish(d: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[E, 3, h, w]` -> depth `[1, H, W]` in `[0, 1]` (float32)."""
    d = d.float().mean(dim=1, keepdim=True).mean(dim=0, keepdim=True).exp()
    lo, hi = d.amin(), d.amax()
    d = (d - lo) / (hi - lo)
    return F.interpolate(d, size, mode="bilinear", align_corners=False, antialias=True)[:, 0].clip(0, 1)


class Portable(Runner):
    """Plain PyTorch (SDPA attention): float32, or float16 weights and activations (upstream's fp16 autocast)."""

    dtypes = ("float32", "float16")
    batch = 1  # the ensemble and its seeded noise are per image
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net, self.alpha_bar = build(ctx)
        self.dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.net.to(self.dtype)
        self.noise = Noise(ctx.params.seed)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        p, net, dt = self.ctx.params, self.net, self.dtype
        x = prepare(image, p.long_side).to(dt)
        z = net.vae.encode_mean(x).float() * LATENT_SCALE
        (noise,) = self.noise(torch.Size((p.ensemble, *z.shape[1:])), z.device, torch.float32)
        y = math.sqrt(self.alpha_bar) * z + math.sqrt(1 - self.alpha_bar) * noise
        cond = z.to(dt).expand(p.ensemble, -1, -1, -1)
        text = net.empty_text.to(dt).expand(p.ensemble, -1, -1)
        grid = times(p.steps)
        for t, t1 in itertools.pairwise(grid):
            v = net.unet(torch.cat([y.to(dt), cond], dim=1), torch.tensor(t, device=z.device), text)
            y = y + (t1 - t) * v.float()
        return finish(net.vae.decode((y / LATENT_SCALE).to(dt)), image.shape[-2:])

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant depth of one image."""
        return {"depth": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[1, 3, H, W]` in `[0, 1]` -> depth `[1, H, W]` in `[0, 1]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.net, lambda _net, image: self._run(image))


class Kernels(Runner):
    """float16 channels-last, cuDNN fp16-accumulate convs, constants of the prompt and the flow times computed once."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network, convert it and precompute the constants."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        net, self.alpha_bar = build(ctx)
        self.net = net.half().to(memory_format=torch.channels_last)
        self.grid = times(ctx.params.steps)
        with torch.inference_mode():
            self.fast = Fast(self.net.unet, self.net.empty_text.half(), self.grid[:-1])
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Affine-invariant depth of one image."""
        p, f, grid = self.ctx.params, self.fast, self.grid
        x = prepare(image, p.long_side).half()
        z = f.encode_mean(self.net.vae, x).float() * LATENT_SCALE
        (noise,) = self.noise(torch.Size((p.ensemble, *z.shape[1:])), z.device, torch.float32)
        y = math.sqrt(self.alpha_bar) * z + math.sqrt(1 - self.alpha_bar) * noise
        cond = z.half().expand(p.ensemble, -1, -1, -1)
        for t, t1 in itertools.pairwise(grid):
            y = y + (t1 - t) * f.unet_forward(torch.cat([y.half(), cond], dim=1), t).float()
        return {"depth": finish(f.decode(self.net.vae, (y / LATENT_SCALE).half()), image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
