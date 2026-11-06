"""Mask2Former runners: the transformers network with the authors' weights, and its portable / kernels forward passes.

Test-time protocol of the ADE20K checkpoints (detectron2, the authors' `semantic_inference`), per image: resize the
shorter side to 640 (longer side at most 2560; bilinear), ImageNet normalisation, zero-pad bottom / right to a
multiple of 32, one forward pass; the mask logits (1/4 resolution) are resized bilinearly to the padded size, the
semantic logits `sum_q softmax(class_q)[:-1] * sigmoid(mask_q)` are formed there
(`SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False`), cropped to the resized image and resized to the image size.

The network runs through `fastforks.shared.maskformer` (gather-based Swin windows, SDPA): `portable` in plain PyTorch
(fp16 with channels-last convs on CUDA), `kernels` adds the `fvm_seg` kernels (Swin LayerNorm + window gathers, fused
multi-scale deformable attention) and fp16-accumulate GEMMs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared.maskformer import MaskFormer
from fastforks.shared.seginfer import (
    mask_semantic,
    pad_bottom_right,
    pretrained,
    resize,
    target_size,
    upsample,
    upsample_argmax,
)

if TYPE_CHECKING:
    from fastforks.segmentation._runner import Context


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: transformers class, `"Mask2FormerForUniversalSegmentation"`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), e.g. `["short", 640, 2560]`.
        pad: The padded input's sides are multiples of this.
    """

    arch: str
    resize: tuple[str, int, int]
    pad: int


class Portable(Runner):
    """Plain PyTorch (gather-based Swin windows, SDPA): float32, or fp16 with channels-last convs on CUDA."""

    dtypes = ("float32", "float16")
    probabilities = True  # class softmax x mask sigmoid, summed over the queries
    batch = None
    Params = Params
    kernels = False

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = MaskFormer(pretrained(ctx), ctx.torch_dtype, kernels=self.kernels).to(ctx.device)

    def _scores(self, image: torch.Tensor) -> torch.Tensor:
        """Semantic scores of one image `[1, 150, rh, rw]` at the resized image size, in the network dtype."""
        h, w = image.shape[-2:]
        rh, rw = target_size(h, w, self.ctx.params.resize)
        x = pad_bottom_right(scale(resize(image, (rh, rw), pil=True), self.ctx.io), self.ctx.params.pad)
        masks, classes = self.net(x)
        masks = F.interpolate(masks, x.shape[-2:], mode="bilinear", align_corners=False)
        # fp16 on CUDA from the product on (its dtype) to the labels: no float32 image-size scores
        return mask_semantic(masks, classes, None)[..., :rh, :rw]

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        size = (tuple(image.shape[-2:]),)  # `resize` to the image size, fused by `seg_up` in kernels
        out = [upsample(self._scores(im), size, fused=self.kernels) for im in image.split(1)]
        return out[0] if len(out) == 1 else torch.cat(out)

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the scores, class chunk by chunk (no image-size scores)."""
        size = (tuple(image.shape[-2:]),)
        out = [upsample_argmax(self._scores(im), size, fused=self.kernels) for im in image.split(1)]
        return out[0] if len(out) == 1 else torch.cat(out)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Semantic logits `[B, 150, H, W]` of a batch (one forward pass per image)."""
        return {"logits": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """`fvm_seg` kernels (Swin LayerNorm + window gathers, deformable attention), fp16-accumulate GEMMs, `seg_up`."""

    kernels = True
    module = Runner.module


RUNNERS = {"portable": Portable, "kernels": Kernels}
