"""StableNormal turbo runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `YOSONormalsPipeline.__call__` as hubconf's `StableNormal_turbo` sets it up (`t_start=0`, one
step, the default `processing_resolution=768`, `ensemble_size=1`): map to `[-1, 1]`, resize the longest side to
`long_side` (bilinear, antialiased, sides truncated), pad the bottom / right to a multiple of 8 (replicate), encode with
the VAE (posterior mode x 0.18215), run the ControlNet encoder (a UNet encoder, `ControlNetVAEModel`: its zero
convolutions are skipped upstream) on the image latent at timestep 0, run the UNet at timestep 0 on a Gaussian latent
with the ControlNet's outputs added to its skips and mid block, decode, normalise, remove the padding, resize to the
input size (bilinear, not antialiased) and normalise again. Both networks see the empty prompt as diffusers'
`encode_prompt` makes it (padded to 77 tokens). StableNormal's normals point x left, y up, z towards the camera (as
Lotus'; measured: the x flip takes the mean angular error on iBims-1 from 66 to 18 degrees); they are returned in the
camera frame (x right, y down, z forward).

The Gaussian latent comes from a CPU generator seeded with `seed` (float32, then moved to the device); upstream draws it
from the unseeded global RNG. The noise is drawn eagerly (`Noise`) outside the traced graph, so `module()` compiles and
exports; a traced or exported module reuses the noise of its traced shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.depth._runner import Forward, Runner
from fastforks.kernels import cudnn_hacc
from fastforks.shared.sd2 import EMPTY_PROMPT_77, LATENT_SCALE, Noise, SD2Net, load_net
from fastforks.shared.sd2.net.fast import Fast
from fastforks.shared.sd2.net.unet import ControlNetEncoder

if TYPE_CHECKING:
    from fastforks.depth._runner import Context

TIMESTEP = 0  # fixed: hubconf StableNormal_turbo `t_start=0`, the ControlNet's and the UNet's timestep
_ALIGN = 8  # fixed: the VAE downsampling factor; the input is padded to a multiple of it
_CAMERA = (-1.0, -1.0, -1.0)  # fixed: StableNormal's (x left, y up, z to the viewer) -> camera (y down, z forward)
_UNUSED = ("controlnet_cond_embedding.", "controlnet_down_blocks.", "controlnet_mid_block.")  # skipped upstream


@dataclass(frozen=True)
class Params:
    """Load options."""

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
    """Upstream `normalize_normals` (norm clamped at 1e-6)."""
    return n / torch.linalg.vector_norm(n, dim=1, keepdim=True).clamp(min=1e-6)


def finish(n: torch.Tensor, pad: tuple[int, int], size: tuple[int, ...]) -> torch.Tensor:
    """Decoded `[B, 3, h, w]` -> unit normals `[B, H, W, 3]` in the camera frame (float32)."""
    ph, pw = pad
    n = _unit(n.float())[:, :, : -ph or None, : -pw or None]
    if tuple(n.shape[-2:]) != tuple(size):
        n = F.interpolate(n, size, mode="bilinear")
    n = _unit(n) * n.new_tensor(_CAMERA)[:, None, None]
    return n.permute(0, 2, 3, 1)


def _nets(ctx: Context) -> tuple[SD2Net, ControlNetEncoder]:
    """UNet + VAE + the 77-token empty prompt (files 0-2) and the ControlNet encoder (file 3)."""
    net = load_net(ctx, prompt=EMPTY_PROMPT_77, cin=4)
    with torch.device(ctx.device):
        control = ControlNetEncoder(cin=4)
    control.load_state_dict({k: v for k, v in ctx.state_dict(3, strip="").items() if not k.startswith(_UNUSED)})
    return net, control.eval()


def infer(net: SD2Net, control: ControlNetEncoder, image: torch.Tensor, params: Params, noise: Noise) -> torch.Tensor:
    """The pipeline on one image (the plain modules, in the dtype of `net`)."""
    dtype = net.empty_text.dtype
    x, pad = prepare(image, params.long_side)
    rgb = net.vae.encode_mean(x.to(dtype)) * LATENT_SCALE
    (z,) = noise(rgb.shape, rgb.device, dtype)
    ctx = net.empty_text.to(dtype).expand(rgb.shape[0], -1, -1)
    t = torch.tensor(TIMESTEP, device=rgb.device)
    pred = net.unet(z, t, ctx, residuals=control(rgb, t, ctx))
    return finish(net.vae.decode(pred / LATENT_SCALE), pad, image.shape[-2:])


class _Both(torch.nn.Module):
    """The SD2 network and the ControlNet encoder as one module (`Forward` traces one)."""

    def __init__(self, net: SD2Net, control: ControlNetEncoder) -> None:
        super().__init__()
        self.net, self.control = net, control


class Portable(Runner):
    """Plain PyTorch (SDPA attention): float32, or float16 weights and activations."""

    dtypes = ("float32", "float16")
    batch = 1  # the seeded noise is per image
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks."""
        super().__init__(ctx)
        self.dtype = torch.float16 if ctx.dtype == "float16" else torch.float32
        self.both = _Both(*_nets(ctx)).to(self.dtype)
        self.noise = Noise(ctx.params.seed)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return infer(self.both.net, self.both.control, image, self.ctx.params, self.noise)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        return {"normals": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The networks with their input handling: `[1, 3, H, W]` in `[0, 1]` -> normals `[1, H, W, 3]`.

        Call it once eagerly per input size before tracing: the seeded noise is drawn outside the graph (`Noise`).
        """
        return Forward(self.both, lambda _net, image: self._run(image))


class Kernels(Runner):
    """float16 channels-last, cuDNN fp16-accumulate convs, constants of the prompt and the timestep computed once."""

    batch = 1
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the networks, convert them and precompute the constants."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        net, control = _nets(ctx)
        self.net = net.half().to(memory_format=torch.channels_last)
        control = control.half().to(memory_format=torch.channels_last)
        text = self.net.empty_text.half()
        with torch.inference_mode():
            self.fast = Fast(self.net.unet, text, [TIMESTEP])
            self.fast_control = Fast(control, text, [TIMESTEP])
        self.noise = Noise(ctx.params.seed)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Surface normals of one image."""
        f = self.fast
        x, pad = prepare(image, self.ctx.params.long_side)
        rgb = f.encode_mean(self.net.vae, x.half()) * LATENT_SCALE
        (z,) = self.noise(rgb.shape, rgb.device, torch.float16)
        pred = f.unet_forward(z, TIMESTEP, self.fast_control.control(rgb, TIMESTEP))
        return {"normals": finish(f.decode(self.net.vae, pred / LATENT_SCALE), pad, image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
