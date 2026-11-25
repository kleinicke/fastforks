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
    from fastforks._core.mlx import mx
    from fastforks.segmentation._runner import Context

_ROW_CHUNK = 1 << 26  # fixed: mask-logit elements per row chunk of the MLX scores (100 queries: 640 x 2560 in 1)


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
        # The resized masks are disposable, so CUDA inference can activate them in place.
        inplace = masks.is_cuda and torch.is_inference_mode_enabled() and not torch.compiler.is_compiling()
        return mask_semantic(masks, classes, None, inplace=inplace)[..., :rh, :rw]

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


class Mlx(Portable):
    """Apple MLX: the network written in MLX operations (`shared.maskformer.mlx_net`), the semantic inference too.

    The scores at the resized image size are one kernel (the mask resize to the padded size, the crop and the
    sigmoid, `mlx_seg.mask_sigmoid_rows`) and one matrix product with the class probabilities per row chunk; the
    labels' resize and argmax are one kernel.
    """

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Build the transformers network on the checkpoint and copy its weights to MLX."""
        Runner.__init__(self, ctx)
        from fastforks._core.mlx import request_cache  # noqa: PLC0415 - imports mlx
        from fastforks.shared.maskformer import mlx_net  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_net.MlxMaskFormer(pretrained(ctx), ctx.dtype)
        request_cache(self, mlx_net.cache_mb(ctx.dtype))

    def _scores_mlx(self, image: torch.Tensor) -> mx.array:
        """`_scores` on MLX: `[1, rh, rw, 150]` float32 (computed in the run dtype, as `mask_semantic`)."""
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - MLX only (imports mlx)

        h, w = image.shape[-2:]
        rh, rw = target_size(h, w, self.ctx.params.resize)
        x = pad_bottom_right(scale(resize(image, (rh, rw), pil=True), self.ctx.io), self.ctx.params.pad)
        low, classes = self.mlx.raw(to_mx(x))
        dtype = getattr(mx, self.ctx.dtype)
        probs = mx.softmax(classes, axis=-1)[:, :-1].astype(dtype)
        low = low.astype(mx.float32)  # `mask_sigmoid_rows` reads float32 logits
        mx.eval(low, probs)
        step = max(1, _ROW_CHUNK // (low.shape[-1] * rw))
        sizes = (x.shape[-2:], (rh, rw), (rh, rw))
        parts = []
        for r0 in range(0, rh, step):
            parts.append(mlx_seg.mask_sigmoid_rows(low, sizes, (r0, min(r0 + step, rh)), dtype) @ probs)
            mx.eval(parts[-1])
        return mx.concatenate(parts).reshape(1, rh, rw, -1).astype(mx.float32)

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - MLX only
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - MLX only (imports mlx)

        out = [
            to_torch(mlx_seg.resize(self._scores_mlx(im), im.shape[-2:]).transpose(0, 3, 1, 2))
            for im in image.split(1)
        ]  # fmt: skip
        return out[0] if len(out) == 1 else torch.cat(out)

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the scores stay on MLX up to the argmax (one kernel with the resize)."""
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - MLX only (imports mlx)

        out = [mlx_seg.resize_argmax(self._scores_mlx(im), im.shape[-2:]) for im in image.split(1)]
        return out[0] if len(out) == 1 else torch.cat(out)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": Mlx,
    "kernels_mlx": Mlx,
}
