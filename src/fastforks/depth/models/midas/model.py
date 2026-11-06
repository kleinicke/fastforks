"""MiDaS v3.1 runners: weights -> network and the portable / kernels forward passes.

Both follow upstream `run.py` for `dpt_beit_large_512`: keep the aspect ratio and resize by the scale closer to 1 of
`512 / H` and `512 / W` ("minimal"), sides rounded to multiples of 32 (bicubic), normalise with mean = std = 0.5, run,
and upsample the relative inverse depth back to the input size (bicubic, `align_corners=False`). Upstream resizes with
OpenCV's `INTER_CUBIC` on the float image; PyTorch's bicubic uses the same kernel (a = -0.75, replicated borders).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.half import pure_half
from fastforks.depth._runner import Forward, Runner, scale
from fastforks.kernels import cudnn_hacc
from fastforks.shared.midas_beit import HEAD_SCALE, DPTDepthModel, dpt_beit_large, scale_head, use_sdpa_attention
from fastforks.shared.midas_beit import load as load_state
from fastforks.shared.midas_beit.fast import decoder, encode, fp16_accumulate, to_kernels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastforks.depth._runner import Context

MULTIPLE = 32  # fixed: the BEiT-L/512 input multiple, hard-coded upstream


@dataclass(frozen=True)
class Params:
    """Variant `params` + load options."""

    input_size: int  # target side of the "minimal" resize


def network_size(h: int, w: int, input_size: int) -> tuple[int, int]:
    """Upstream `Resize(keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal")` for `h x w`.

    A side that upstream rounds to 0 (aspect ratio above ~32) is `MULTIPLE` (stretched).
    """
    sh, sw = input_size / h, input_size / w
    s = sw if abs(1 - sw) < abs(1 - sh) else sh
    return (
        max(MULTIPLE, int(np.round(s * h / MULTIPLE)) * MULTIPLE),
        max(MULTIPLE, int(np.round(s * w / MULTIPLE)) * MULTIPLE),
    )


def _net(ctx: Context) -> DPTDepthModel:
    with torch.device(ctx.device):
        net = dpt_beit_large(512)
    load_state(net, ctx.state_dict())
    scale_head(net)  # fp16 head activations reach ~1.1e5 otherwise; `_restore` undoes it in float32
    return net.eval()


def _prepare(ctx: Context, image: torch.Tensor) -> torch.Tensor:
    """`[B, 3, H, W]` in `[0, 1]` -> resized, normalised network input (float32)."""
    h, w = network_size(*image.shape[-2:], ctx.params.input_size)
    return scale(F.interpolate(image, (h, w), mode="bicubic", align_corners=False), ctx.io)


def _restore(d: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Network output `[B, h, w]` (`HEAD_SCALE` x inverse depth) -> float32 `[B, H, W]` at the input size."""
    return F.interpolate(d[:, None].float() * (1 / HEAD_SCALE), size, mode="bicubic", align_corners=False)[:, 0]


class Portable(Runner):
    """Plain PyTorch, SDPA attention with the relative position bias as mask: float32, or on CUDA the network in fp16.

    The fp16 network has no autocast (weights cast once, channels-last convs, head scaled by `HEAD_SCALE`).
    """

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = _net(ctx)
        use_sdpa_attention(self.net)
        self.half = ctx.dtype == "float16"
        if self.half:
            pure_half(self.net)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        return _restore(self.net(_prepare(self.ctx, image)), image.shape[-2:])

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Relative inverse depth of a batch."""
        return {"inverse_depth": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> inverse depth `[B, H, W]`."""
        return Forward(self.net, lambda _net, image: self._run(image))


class Kernels(Runner):
    """fp16 BEiT (SDPA attention with the bias mask, fp16-accumulating GEMMs) and a cuDNN fp16-accumulate DPT head."""

    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network and convert it."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        self.net = _net(ctx)
        to_kernels(self.net)

    def __call__(self, image: torch.Tensor, intrinsics: torch.Tensor | None) -> dict[str, torch.Tensor]:  # noqa: ARG002 - unused
        """Relative inverse depth of a batch."""
        x = _prepare(self.ctx, image)
        h, w = x.shape[-2:]
        with fp16_accumulate():
            y = decoder(self.net, encode(self.net, x), h, w)[0]
        last = self.net.scratch.output_conv[4]
        d = F.relu(F.conv2d(y.float(), last.weight.float(), last.bias.float()))  # 32 -> 1 channel in fp32
        return {"inverse_depth": _restore(d[:, 0], image.shape[-2:])}


RUNNERS = {"portable": Portable, "kernels": Kernels}
