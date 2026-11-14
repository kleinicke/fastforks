"""LaMa (big-lama) runners: `FFCResNetGenerator` and the call of upstream `bin/predict.py`.

As upstream (`predict.py` with `configs/prediction/default.yaml`: no refinement, `pad_out_to_modulo` 8): the RGB
image and the binary mask (nonzero = hole) are padded symmetrically at the bottom / right to a multiple of 8, the
network sees the masked image plus the mask (`DefaultInpaintingTrainingModule`, `concat_mask`), and its output is
pasted into the hole (`mask * predicted + (1 - mask) * image`), then cropped. Upstream runs float32.

fastforks folds the eval BatchNorms into the convolutions and runs them in the card's dtype (channels-last on CUDA),
pads each FFC input once (reflect, keeping channels-last) and transforms the NHWC view in the Fourier units, which
stay float32 (cuFFT's half precision needs power-of-two sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fastforks._core.checkpoint import Inert, config_globals, load_strict
from fastforks._core.half import fp16_accumulate
from fastforks._core.patch import is_base, override
from fastforks._core.preprocess import Crop, pad_widths
from fastforks.kernels import cudnn_hacc
from fastforks.restoration._runner import Forward, Runner
from fastforks.shared.cuda_graphs import GraphCache

from .net import FFCResNetGenerator
from .net.ffc import FourierUnit, fold_batchnorm

if TYPE_CHECKING:
    from fastforks.restoration._runner import Context

# Lightning checkpoint: the hyper-parameters are OmegaConf containers, the callbacks Lightning's
_GLOBALS = {**config_globals(), ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"): Inert}


@dataclass(frozen=True)
class Params:
    """Card `[params]` and `[options]`."""

    scale: int  # 1: inpainting keeps the size
    cuda_graphs: bool  # kernels: replay the network from a CUDA graph


def _symmetric(x: torch.Tensor, bottom: int, right: int) -> torch.Tensor:
    """As numpy `pad(mode="symmetric")` at the bottom / right (the edge row / column repeated, then mirrored)."""
    if bottom:
        x = torch.cat([x, x[..., x.shape[-2] - bottom :, :].flip(-2)], -2)
    if right:
        x = torch.cat([x, x[..., x.shape[-1] - right :].flip(-1)], -1)
    return x


class Portable(Runner):
    """Plain PyTorch, BatchNorms folded, convolutions in the card's dtype (channels-last on CUDA), FFTs in float32."""

    dtypes = ("float32", "float16")
    Params = Params
    mask = "required"

    def __init__(self, ctx: Context) -> None:
        """Build the generator, fold its BatchNorms."""
        super().__init__(ctx)
        self.net = FFCResNetGenerator()
        sd = ctx.state_dict(key="state_dict", strip="", extra=_GLOBALS)
        gen = {k.removeprefix("generator."): v for k, v in sd.items() if k.startswith("generator.")}
        load_strict(self.net, gen, f"{ctx.variant.ref} checkpoint")
        fold_batchnorm(self.net.eval())
        self.fmt = torch.channels_last if torch.device(ctx.device).type == "cuda" else torch.contiguous_format
        self.dtype = getattr(torch, ctx.dtype)
        self.net.to(ctx.device, self.dtype).to(memory_format=self.fmt)
        for m in self.net.modules():
            if isinstance(m, FourierUnit):
                m.float()

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, Crop]:
        """Symmetric padding at the bottom / right to a multiple of 8 (upstream `pad_img_to_modulo`)."""
        h, w = x.shape[-2:]
        left, right, top, bottom = pad_widths(h, w, self.ctx.io)
        return _symmetric(x, bottom, right), Crop(top, left, h, w)

    def _inpaint(self, x: torch.Tensor) -> torch.Tensor:
        image, mask = x[:, :3], x[:, 3:]
        inp = torch.cat([image * (1 - mask), mask], 1).to(self.dtype).contiguous(memory_format=self.fmt)
        pred = self.net(inp).float()
        return mask * pred + (1 - mask) * image

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The inpainted batch (RGB + mask channels in, RGB out)."""
        return {"image": self._inpaint(image)}

    def module(self) -> torch.nn.Module:
        """The generator with the masking and the paste-back."""
        return Forward(self.net, lambda _net, image: self._inpaint(image))


def _hacc_conv(conv: torch.nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    return cudnn_hacc.conv_half_acc(x, conv.weight, conv.bias, conv.padding, conv.stride, conv.dilation)  # ty: ignore[invalid-argument-type]


class Kernels(Portable):
    """float16, channels-last, fp16-accumulate convolutions (`cudnn_hacc`), a CUDA graph per input shape."""

    module = Runner.module  # fp16-accumulate convolutions are cuDNN plans, not exportable operators

    def __init__(self, ctx: Context) -> None:
        """Build the generator, route its half-precision convolutions through `cudnn_hacc`, set up the graphs."""
        super().__init__(ctx)
        cudnn_hacc.load(ctx.device)
        for m in self.net.modules():
            if is_base(m, torch.nn.Conv2d) and m.weight.dtype == torch.float16 and m.groups == 1:
                override(m, _hacc_conv)
        after = cudnn_hacc.capture_after()
        self.graphs = GraphCache(self._eager, graphs=1, capture_after=after) if ctx.params.cuda_graphs else None

    def _eager(self, x: torch.Tensor) -> torch.Tensor:
        with fp16_accumulate():
            return self._inpaint(x)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """The inpainted batch."""
        out = None if self.graphs is None else self.graphs.run((image.device, image.dtype, tuple(image.shape)), image)
        return {"image": self._eager(image) if out is None else out}

    def release_graphs(self) -> None:
        """Free the CUDA graphs and their memory pool."""
        if self.graphs is not None:
            self.graphs.release()


RUNNERS = {"portable": Portable, "kernels": Kernels}
