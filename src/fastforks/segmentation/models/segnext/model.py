"""SegNeXt runner: the vendored network (`net.py`) with the mmsegmentation ADE20K checkpoints.

Test-time protocol of the configs (mmseg `test_pipeline`, `SegDataPreProcessor` with `size_divisor=32`,
`EncoderDecoder.whole_inference`, `postprocess_result`): resize keeping the aspect ratio to fit `2048 x 512`
(bilinear), ImageNet normalisation, zero padding on the right and bottom to multiples of 32, one forward pass, logits
(1/8 resolution) resized bilinearly to the padded size, the padding cut off, then resized to the image size.

One deviation from upstream: the head's matrix factorisation starts from fixed bases (`net.Hamburger`), so the same
image always gives the same logits; upstream draws random ones on every call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.segmentation.models.segnext import net as sn
from fastforks.shared.seginfer import resize, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

_DIVISOR = 32  # fixed: the configs' `test_cfg=dict(size_divisor=32)`


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: Network (`net.ARCHS`): `mscan-t`, `mscan-s`, `mscan-b`, `mscan-l`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["mmseg", 2048, 512]`.
    """

    arch: str
    resize: tuple[str, int, int]


def network(ctx: Context) -> sn.EncoderDecoder:
    """The variant's network with its checkpoint (CPU, float32, eval).

    Raises:
        CheckpointMismatchError: Any missing, unexpected or reshaped tensor.
    """
    model = sn.build(ctx.params.arch)
    load_strict(model, ctx.state_dict(0, key="state_dict", strip=""), f"{ctx.variant.ref} checkpoint")
    return model.eval().requires_grad_(requires_grad=False)


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 channels-last on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = network(ctx).to(ctx.device, ctx.torch_dtype)
        if ctx.device.type == "cuda":
            self.net = self.net.to(memory_format=torch.channels_last)

    def _scores(self, image: torch.Tensor) -> torch.Tensor:
        """Logits at the resized input's size (before the last resize to the image size)."""
        size = target_size(*image.shape[-2:], self.ctx.params.resize)
        x = scale(resize(image, size), self.ctx.io)
        x = F.pad(x, (0, -size[1] % _DIVISOR, 0, -size[0] % _DIVISOR))
        if x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        s = self.net(x.to(next(self.net.parameters()).dtype)).contiguous()  # NCHW: faster bilinear upsampling
        return resize(s, x.shape[-2:])[..., : size[0], : size[1]]

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 150, H, W]` of a batch."""
        return {"logits": upsample(self._scores(image), (image.shape[-2:],))}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        return upsample_argmax(self._scores(image), (image.shape[-2:],))

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self(image)["logits"])


RUNNERS = {"portable": Portable}
