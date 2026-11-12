"""Matching API: `list_models`, `model_card`, `load`, `MatchingModel`, `MatchingResult`, `SparseMatchingResult`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.errors import CardError, InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batches, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.matching._runner import scale

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.matching._runner import Runner

__all__ = ["Backend", "MatchingModel", "MatchingResult", "SparseMatchingResult", "list_models", "load", "model_card"]

TASK = "matching"


@dataclass(frozen=True)
class MatchingResult(NumpyOutputs):
    """Output of a dense-matching model: float32 on the model's device, aligned with the first image.

    Attributes:
        warp: Where every pixel of the first image lies in the second, `[2, H1, W1]` (`[B, 2, H1, W1]` for batches):
            channel 0 is `x`, channel 1 is `y`, in pixels of the second image at its input resolution (pixel centres
            at integer coordinates). Meaningful only where `certainty` is high.
        certainty: `[H1, W1]` in `[0, 1]`: the model's probability that the pixel is visible in the second image
            (unthresholded), `None` if the card does not list it.
        model: Versioned model id, e.g. `"romav2@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    warp: torch.Tensor
    certainty: torch.Tensor | None
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, n: int = 5000, *, threshold: float = 0.05, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `n` keypoint matches from the warp, for `estimate_pose`.

        Pixels are drawn without replacement with a probability proportional to their certainty, clipped to 1 above
        `threshold` (upstream RoMa's sampling without its density balancing); without a certainty map, uniformly.

        Args:
            n: Matches per pair (fewer if fewer pixels have a non-zero certainty).
            threshold: Certainty from which pixels are drawn with equal probability.
            seed: Of the sampling.

        Returns:
            Points in the first and in the second image, `[n, 2]` float32 `(x, y)` pixels (`[B, n, 2]` for batches),
            on the warp's device.
        """
        warp = self.warp.unsqueeze(0) if self.warp.dim() == 3 else self.warp  # noqa: PLR2004 - [2, H, W]
        b, _, h, w = warp.shape
        if self.certainty is None:
            weight = torch.ones(b, h * w, device=warp.device)
        else:
            c = self.certainty.reshape(b, h * w)
            weight = torch.where(c > threshold, 1.0, c)
        n = min(n, int((weight > 0).sum(1).min()))
        if n < 1:
            msg = "no pixel of the first image has a certainty above zero"
            raise InputError(msg)
        gen = torch.Generator(warp.device).manual_seed(seed)
        ind = torch.multinomial(weight, n, replacement=False, generator=gen)  # [B, n]
        xy1 = torch.stack((ind % w, ind // w), -1).float()
        xy2 = torch.gather(warp.flatten(2), 2, ind[:, None].expand(b, 2, n)).transpose(1, 2)
        if self.warp.dim() == 3:  # noqa: PLR2004
            return xy1[0], xy2[0]
        return xy1, xy2


@dataclass(frozen=True)
class SparseMatchingResult(NumpyOutputs):
    """Output of a sparse-matching model for one image pair, on the model's device.

    `pairs` index the two keypoint sets: the form COLMAP / `pycolmap.match_from_pairs` take.

    Attributes:
        keypoints1: `[N, 2]` float32 `(x, y)` keypoints detected in the first image, in its input pixels (pixel
            centres at integer coordinates).
        keypoints2: `[M, 2]` float32 keypoints of the second image.
        pairs: `[K, 2]` int64: index into `keypoints1`, index into `keypoints2` of every match.
        scores: `[K]` float32 match confidence in `[0, 1]`.
        model: Versioned model id, e.g. `"aliked-lightglue@1"`.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults (they change the numerics).
    """

    keypoints1: torch.Tensor
    keypoints2: torch.Tensor
    pairs: torch.Tensor
    scores: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """The matched points, for `estimate_pose`.

        Args:
            threshold: Keep the matches with a score above it (the model already dropped those below its own).

        Returns:
            Points in the first and in the second image, `[K, 2]` float32 `(x, y)` pixels, row `i` matched to row `i`.
        """
        keep = self.pairs[self.scores > threshold]
        return self.keypoints1[keep[:, 0]], self.keypoints2[keep[:, 1]]


class MatchingModel(core.Model):
    """A loaded matching model. Call it with two images (or two batches of the same length) of any sizes.

    Dense models return a `MatchingResult`, sparse ones (whose card lists `pairs`) a `SparseMatchingResult` per pair
    (a list for batches: the keypoint counts differ).

    It is deliberately not an `nn.Module`: calls take images of any supported type and size and do the resizing.
    `module()` is the same model as an `nn.Module` for `torch.compile` / `torch.export`, and `export_onnx` writes it
    to ONNX (for TensorRT).

    An instance is not reentrant: calls from several threads are serialised (use one instance per thread to run them
    in parallel).

    Has the attributes of `fastforks.Model`.
    """

    _runner: Runner

    def module(self) -> torch.nn.Module:
        """The whole model as an `nn.Module`, for `torch.compile` and `torch.export`.

        `forward(image1, image2)` takes RGB `[B, 3, H, W]` float32 tensors in `[0, 1]` on the model's device and
        returns `{"warp": [B, 2, H, W], "certainty": [B, H, W]}` float32, as a `MatchingModel` call does for a batch.
        Call it under `torch.inference_mode()` (or `no_grad`).

        Raises:
            BackendUnavailableError: The backend has no exportable module.
        """
        return _PortableModule(self._runner, super().module())

    @property
    def sparse(self) -> bool:
        """Whether the model returns keypoint matches (`SparseMatchingResult`) instead of a dense warp."""
        return "pairs" in self.variant.outputs

    def __call__(
        self, image1: ImageLike, image2: ImageLike
    ) -> MatchingResult | SparseMatchingResult | list[SparseMatchingResult]:
        """Match `image1` into `image2`: every pixel (dense models) or detected keypoints (sparse models).

        ```python
        result = model("view_a.jpg", "view_b.jpg")
        x, y = result.warp  # [H1, W1] each, pixels of view_b
        ```

        Args:
            image1: First image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`); a
                batch gives `[B, 2, H1, W1]` warps.
            image2: Second image, any size; a batch of the same length if `image1` is one.

        Returns:
            The `MatchingResult`; for sparse models a `SparseMatchingResult` (a list of them for batches).

        Raises:
            InputError: Unsupported or too small inputs, or batches of different lengths.
        """
        t1, t2 = to_rgb_batches((image1, image2), self.device)
        if is_batch(image1) != is_batch(image2) or t1.shape[0] != t2.shape[0]:
            msg = f"image1 and image2 must be two images or equal-length batches, got {t1.shape[0]}, {t2.shape[0]}"
            raise InputError(msg)
        self._check_size(t1)
        self._check_size(t2)
        if self.sparse:
            with self._session():
                results = [self._run_sparse(a[None], b[None]) for a, b in zip(t1, t2, strict=True)]
            return results if is_batch(image1) else results[0]
        maps = unbatch(self._batched(self._run, t1, t2), image1)
        return MatchingResult(maps["warp"], maps.get("certainty"), self.variant.ref, self.backend, self.options)

    def _run(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scale, run and check one chunk."""
        io = self.variant.io
        out = self._runner(scale(image1, io), scale(image2, io))
        b, (h, w) = image1.shape[0], image1.shape[-2:]
        return self._check_outputs(out, {"warp": (b, 2, h, w), "certainty": (b, h, w)})

    def _run_sparse(self, image1: torch.Tensor, image2: torch.Tensor) -> SparseMatchingResult:
        """Scale, run and check one pair `[1, 3, H, W]`."""
        io = self.variant.io
        out = self._runner(scale(image1, io), scale(image2, io))
        n, m, k = (len(out[key]) for key in ("keypoints1", "keypoints2", "pairs"))
        shapes = {"keypoints1": (n, 2), "keypoints2": (m, 2), "pairs": (k, 2), "scores": (k,)}
        maps = self._check_outputs(out, shapes)
        pairs = maps["pairs"].long()
        if k and (pairs.min() < 0 or pairs[:, 0].max() >= n or pairs[:, 1].max() >= m):
            msg = f"{self.variant.ref}: runner returned pairs outside the keypoint sets ({n}, {m})"
            raise CardError(msg)
        return SparseMatchingResult(
            maps["keypoints1"], maps["keypoints2"], pairs, maps["scores"], self.variant.ref, self.backend, self.options
        )


class _PortableModule(torch.nn.Module):
    """Scaling and the portable runner (its network, resizing and precision)."""

    def __init__(self, runner: Runner, net: torch.nn.Module) -> None:
        """Wrap a portable runner; `net` (its `Runner.module()`) holds the parameters."""
        super().__init__()
        self.net = net.eval().requires_grad_(requires_grad=False)  # inference only
        self._runner = runner
        self.eval()

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> dict[str, torch.Tensor]:
        """`{"warp": [B, 2, H, W], "certainty": [B, H, W]}` float32 of RGB `[B, 3, H, W]` pairs in `[0, 1]`."""
        io = self._runner.ctx.io
        return {
            k: v.float() if v.is_floating_point() else v
            for k, v in self.net(scale(image1, io), scale(image2, io)).items()
        }


def bench_inputs(model: MatchingModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: a synthetic textured pair `[1, 3, H, W]` (the second image shifted by 8 px)."""
    return synthetic_pair(size.height, size.width, model.device), {}


list_models, model_card, load = task_api(TASK, MatchingModel)
