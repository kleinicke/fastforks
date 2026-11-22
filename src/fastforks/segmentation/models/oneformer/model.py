"""OneFormer runners (semantic task): the transformers network with the authors' weights, portable and kernels passes.

Test-time protocol of the ADE20K checkpoints (detectron2, the authors' `semantic_inference`), per image: resize the
shorter side to 640 (longer side at most 2560; bilinear), ImageNet normalisation, zero-pad bottom / right to a
multiple of 32, one forward pass; the mask logits (1/4 resolution) are resized bilinearly to the padded size, cropped
to the resized image, resized to the image size, and the semantic logits are `sum_q softmax(class_q)[:-1] *
sigmoid(mask_q)`. The task input is the tokenised "the task is semantic" (the authors' `task_inputs`).

The network runs through `fastforks.shared.maskformer` (gather-based Swin windows, SDPA): `portable` in plain PyTorch
(fp16 with channels-last convs on CUDA; the query transformer stays float32, its task token computed once in
float32), `kernels` adds the `fvm_seg` kernels (Swin LayerNorm + window gathers, fused multi-scale deformable attention)
and fp16-accumulate GEMMs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastforks._core.tracing import in_onnx_export
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared import mlx_wrap
from fastforks.shared.maskformer import MaskFormer
from fastforks.shared.seginfer import pad_bottom_right, pretrained, resize, target_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastforks.segmentation._runner import Context

# fixed: mask-logit elements per row chunk of the semantic scores (fp16: 128 MB; 1024 x 2048, 250 queries: 8 chunks)
_ROW_CHUNK = 1 << 26
_MASK_RESIZE_CHUNK = 1 << 25


@dataclass(frozen=True)
class Params:
    """Variant `params`.

    Attributes:
        arch: transformers class, `"OneFormerForUniversalSegmentation"`.
        resize: Test-time resize rule (`fastforks.shared.seginfer.target_size`), e.g. `["short", 640, 2560]`.
        pad: The padded input's sides are multiples of this.
        task_tokens: Token ids of the task text (CLIP tokenizer), zero-padded to the config's `task_seq_len`.
    """

    arch: str
    resize: tuple[str, int, int]
    pad: int
    task_tokens: tuple[int, ...]


class Portable(Runner):
    """Plain PyTorch (gather-based Swin windows, SDPA): float32, or fp16 with channels-last convs on CUDA."""

    dtypes = ("float32", "float16")
    probabilities = True  # class softmax x mask sigmoid, summed over the queries
    batch = None
    Params = Params
    kernels = False
    # Swin window attention (a bias mask): cuDNN builds a graph per new shape, so the first 1024 x 2048 call took
    # 0.95 s (memory-efficient: 0.45 s) and it is no faster after (kernels 112.4 -> 111.0 ms; `fastforks.shared.sdpa`)
    sdpa_auto = "efficient,cudnn,math"

    def __init__(self, ctx: Context) -> None:
        """Build the network."""
        super().__init__(ctx)
        self.net = MaskFormer(
            pretrained(ctx), ctx.torch_dtype, kernels=self.kernels, task_inputs=ctx.params.task_tokens
        ).to(ctx.device)

    def _rows(self, image: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        """Semantic scores of one image at its size, row chunk by row chunk: `(row, [1, 150, rows, W])`, net dtype.

        The image-size mask logits are the one image-size tensor (no float32 or full sigmoid copy); `__call__` and
        `labels` run the same chunks, so the labels are exactly the argmax of the logits.
        """
        h, w = image.shape[-2:]
        rh, rw = target_size(h, w, self.ctx.params.resize)
        x = pad_bottom_right(scale(resize(image, (rh, rw), pil=True), self.ctx.io), self.ctx.params.pad)
        masks, classes = self.net(x)
        hp, wp = x.shape[-2:]
        probs = classes.float().softmax(-1)[..., :-1].to(masks.dtype)  # as `mask_semantic`
        step = max(1, _ROW_CHUNK // (masks.shape[1] * w))
        if (
            masks.is_cuda
            and not torch.is_grad_enabled()
            and not (torch.compiler.is_compiling() or torch.jit.is_tracing() or in_onnx_export())
            and masks.shape[1] * max(h * w, hp * wp) > _ROW_CHUNK
        ):
            # Store the activated masks in the contiguous row blocks consumed by the class projection.
            # Each query needs only its own pixels to resize, so the full intermediate grid is never allocated.
            query_step = max(1, _MASK_RESIZE_CHUNK // max(h * w, hp * wp))
            strips = deque(
                (start, masks.new_empty((1, masks.shape[1], min(step, h - start), w))) for start in range(0, h, step)
            )
            for index, queries in enumerate(masks.split(query_step, dim=1)):
                up = F.interpolate(queries, (hp, wp), mode="bilinear", align_corners=False)[..., :rh, :rw]
                activated = resize(up, (h, w)).sigmoid_()
                q0 = index * query_step
                for start, stripe in strips:
                    stripe[:, q0 : q0 + queries.shape[1]].copy_(activated[..., start : start + stripe.shape[2], :])
                del up, activated
            del masks
        else:
            masks = F.interpolate(masks, (hp, wp), mode="bilinear", align_corners=False)[..., :rh, :rw]
            masks = resize(masks, (h, w))
            for r0 in range(0, h, step):
                yield r0, torch.einsum("bqc,bqhw->bchw", probs, masks[:, :, r0 : r0 + step].sigmoid())
            return
        while strips:
            r0, stripe = strips.popleft()
            scores = torch.einsum("bqc,bqhw->bchw", probs, stripe)
            del stripe
            yield r0, scores

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        out = None
        for r0, s in self._rows(image):
            if out is None:
                out = s.new_empty((*s.shape[:2], *image.shape[-2:]))
            out[:, :, r0 : r0 + s.shape[2]] = s
        assert out is not None  # noqa: S101 - at least one row
        return out

    def _run(self, image: torch.Tensor) -> torch.Tensor:
        out = [self._one(im) for im in image.split(1)]
        return out[0] if len(out) == 1 else torch.cat(out)

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of the scores, row chunk by row chunk (no image-size scores)."""
        out = torch.empty((image.shape[0], *image.shape[-2:]), device=image.device, dtype=torch.int64)
        for i, im in enumerate(image.split(1)):
            for r0, s in self._rows(im):
                out[i, r0 : r0 + s.shape[2]] = s[0].argmax(0)
        return out

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Semantic logits `[B, 150, H, W]` of a batch (one forward pass per image)."""
        return {"logits": self._run(image)}

    def module(self) -> torch.nn.Module:
        """The network with its input handling: `[B, 3, H, W]` in `[0, 1]` -> logits `[B, 150, H, W]`."""
        return Forward(self.net.eval(), lambda _net, image: self._run(image))


class Kernels(Portable):
    """`fvm_seg` kernels (Swin LayerNorm + window gathers, multi-scale deformable attention), fp16-accumulate GEMMs."""

    kernels = True
    module = Runner.module


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": mlx_wrap.port(Portable),
    "kernels_mlx": mlx_wrap.port(Portable),
}
