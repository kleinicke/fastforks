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

import numpy as np
import torch
import torch.nn.functional as F

from fastforks._core.tracing import in_onnx_export
from fastforks.segmentation._runner import Forward, Runner, scale
from fastforks.shared.maskformer import MaskFormer
from fastforks.shared.seginfer import pad_bottom_right, pretrained, resize, target_size

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastforks._core.mlx import mx
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
                logits = resize(up, (h, w))
                q0 = index * query_step
                for start, stripe in strips:  # the sigmoid writes the stripe: no separate copy pass
                    torch.sigmoid(
                        logits[..., start : start + stripe.shape[2], :], out=stripe[:, q0 : q0 + queries.shape[1]]
                    )
                del up, logits
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


class Mlx(Portable):
    """Apple MLX: the network written in MLX operations (`shared.maskformer.mlx_net`), the semantic inference too.

    The scores of a row chunk are one kernel (both mask resizes and the sigmoid, `mlx_seg.mask_sigmoid_rows`) and one
    matrix product with the class probabilities, so the masks never leave MLX and are never image-size.
    """

    module = Runner.module

    def __init__(self, ctx: Context) -> None:
        """Build the transformers network on the checkpoint and copy its weights to MLX."""
        Runner.__init__(self, ctx)
        from fastforks._core.mlx import request_cache  # noqa: PLC0415 - imports mlx
        from fastforks.shared.maskformer import mlx_net  # noqa: PLC0415 - imports mlx

        self.mlx = mlx_net.MlxMaskFormer(pretrained(ctx), ctx.dtype, task_inputs=ctx.params.task_tokens)
        request_cache(self, mlx_net.CACHE_MB)

    def _rows_mlx(self, image: torch.Tensor) -> Iterator[tuple[int, mx.array]]:
        """`_rows` on MLX: `(row, [rows * W, 150])` scores in the run dtype (evaluated chunk by chunk by the caller)."""
        from fastforks._core.mlx import mx, to_mx  # noqa: PLC0415 - MLX only
        from fastforks.shared import mlx_seg  # noqa: PLC0415 - MLX only (imports mlx)

        h, w = image.shape[-2:]
        rh, rw = target_size(h, w, self.ctx.params.resize)
        x = pad_bottom_right(scale(resize(image, (rh, rw), pil=True), self.ctx.io), self.ctx.params.pad)
        low, classes = self.mlx.raw(to_mx(x))
        dtype = getattr(mx, self.ctx.dtype)
        probs = mx.softmax(classes, axis=-1)[:, :-1].astype(dtype)  # as `mask_semantic`
        low = low.astype(mx.float32)  # `mask_sigmoid_rows` reads float32 logits
        mx.eval(low, probs)
        del classes
        step = max(1, _ROW_CHUNK // (low.shape[-1] * w))
        sizes = (x.shape[-2:], (rh, rw), (h, w))
        for r0 in range(0, h, step):
            r1 = min(r0 + step, h)
            yield r0, mlx_seg.mask_sigmoid_rows(low, sizes, (r0, r1), dtype) @ probs

    def _one(self, image: torch.Tensor) -> torch.Tensor:
        from fastforks._core.mlx import to_torch  # noqa: PLC0415 - MLX only

        h, w = image.shape[-2:]
        out = None
        for r0, s in self._rows_mlx(image):
            rows = to_torch(s).view(-1, w, s.shape[-1])
            if out is None:
                out = rows.new_empty((1, s.shape[-1], h, w))
            out[0, :, r0 : r0 + rows.shape[0]] = rows.permute(2, 0, 1)
        assert out is not None  # noqa: S101 - at least one row
        return out

    def labels(self, image: torch.Tensor) -> torch.Tensor:
        """Labels `[B, H, W]` of a batch: the argmax of each row chunk's scores on MLX."""
        from fastforks._core.mlx import mx  # noqa: PLC0415 - MLX only

        h, w = image.shape[-2:]
        out = []
        for im in image.split(1):
            parts = []
            for _, s in self._rows_mlx(im):
                parts.append(mx.argmax(s, axis=-1))
                mx.eval(parts[-1])  # the chunk's masks and scores are freed before the next chunk's are made
            out.append(mx.concatenate(parts).reshape(h, w))
        labels = mx.stack(out)
        mx.eval(labels)
        return torch.from_numpy(np.array(labels)).to(torch.int64)


RUNNERS = {
    "portable": Portable,
    "kernels": Kernels,
    "portable_mlx": Mlx,
    "kernels_mlx": Mlx,
}
