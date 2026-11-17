"""BiSeNet face parsing runner: the vendored network (`net`) with facexlib's `parsing_bisenet.pth`.

Test protocol of face-parsing.PyTorch's `test.py`: the image resized to 512 x 512 (PIL `BILINEAR`), ImageNet
normalisation, `net(x)[0]` (the main head's logits, bilinearly upsampled to 512 x 512 with `align_corners=True` inside
the network). fastforks then resizes the logits to the image size (bilinear, half-pixel centres), as for the other
fixed-size face parsers.

`portable` folds the BatchNorms into the convs, skips the two auxiliary heads (training-only outputs that `test.py`
discards) and runs fp16 channels-last on CUDA. No kernels runner: fp16-accumulate convolutions (`cudnn_hacc`) were
slower (2.77 vs 2.41 ms at 512 x 512).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.checkpoint import load_strict
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.segmentation.models.bisenet.net import BiSeNet
from fastforks.shared.seginfer import resize, target_size, upsample, upsample_argmax

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context

_CLASSES = 19  # fixed: the CelebAMask-HQ classes of the checkpoint


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), `["fixed", 512, 512]`.
    """

    resize: tuple[str, int, int]


def _fold(net: torch.nn.Module) -> None:
    """Fold every `conv -> BatchNorm2d` pair into the conv (float32, in place) and drop the BatchNorm."""
    for parent in list(net.modules()):
        pairs = (("conv", "bn"), ("conv1", "bn1"), ("conv2", "bn2"), ("conv_atten", "bn_atten"), ("0", "1"))
        for c, b in pairs:
            conv, bn = getattr(parent, c, None), getattr(parent, b, None)
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                s = bn.weight * torch.rsqrt(bn.running_var + bn.eps)
                t = bn.bias - bn.running_mean * s
                if conv.bias is not None:
                    t = t + conv.bias * s
                conv.weight.data.mul_(s.view(-1, 1, 1, 1))
                conv.bias = torch.nn.Parameter(t)
                setattr(parent, b, torch.nn.Identity())
    left = [n for n, m in net.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if left:
        msg = f"BiSeNet: BatchNorms without a conv to fold into: {left}"
        raise AssertionError(msg)


def network(ctx: Context) -> BiSeNet:
    """The checkpoint's network (CPU, float32, eval), BatchNorms folded.

    Raises:
        CheckpointMismatchError: Any missing, unexpected or reshaped tensor.
    """
    net = BiSeNet(num_class=_CLASSES)
    load_strict(net, ctx.state_dict(), f"{ctx.variant.ref} checkpoint")
    net.eval().requires_grad_(requires_grad=False)
    _fold(net)
    return net


def _main_head(net: BiSeNet, x: torch.Tensor) -> torch.Tensor:
    """`net(x)[0]` without the auxiliary heads: logits `[B, 19, h, w]` at the input size."""
    feat8, cp8, _ = net.cp(x)
    out, _ = net.conv_out(net.ffm(feat8, cp8))
    return F.interpolate(out, x.shape[-2:], mode="bilinear", align_corners=True)


class Portable(Runner):
    """Plain PyTorch: float32, or fp16 channels-last on CUDA."""

    dtypes = ("float32", "float16")
    batch = None
    Params = Params

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = network(ctx).to(ctx.device, ctx.torch_dtype)
        self.fmt = torch.channels_last if ctx.device.type == "cuda" else torch.contiguous_format
        self.net = self.net.to(memory_format=self.fmt)

    def _scores(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
        """Logits at the network size (NCHW) and the resize step to the image size."""
        size = target_size(*image.shape[-2:], self.ctx.params.resize)
        x = scale(resize(image, size, pil=True), self.ctx.io).contiguous(memory_format=self.fmt)
        s = _main_head(self.net, x.to(next(self.net.parameters()).dtype)).contiguous()
        return s, ((image.shape[-2], image.shape[-1]),)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Logits `[B, 19, H, W]` of a batch."""
        return {"logits": upsample(*self._scores(image))}

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the logits, class chunk by chunk (no full-size logits)."""
        return upsample_argmax(*self._scores(image))

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 19, H, W]`."""
        return Forward(self.net, lambda _net, image: self(image)["logits"])


RUNNERS = {"portable": Portable}
