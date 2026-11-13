"""Matching API: `list_models`, `model_card`, `load`, `MatchingModel`, `MatchingResult`, `SparseMatchingResult`."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_pair
from fastforks._core.errors import CardError, InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch, to_rgb_batches, unbatch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs
from fastforks.matching._pairs import DecodedImages, PairCache
from fastforks.matching._runner import scale
from fastforks.matching.relative_pose import FundamentalMatrix, RelativePose, estimate_fundamental, estimate_pose

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from fastforks.matching._runner import Runner

__all__ = [
    "Backend",
    "MatchingModel",
    "MatchingResult",
    "SparseMatchingResult",
    "VerifiedMatches",
    "list_models",
    "load",
    "model_card",
]

TASK = "matching"
Mode = Literal["auto", "portable", "kernels"]
CACHE_MB = 512.0  # `match_pairs`: 26 RoMa v2 descriptors (19.5 MB each at coarse_size 800)
DECODED_MB = 1024.0  # `match_pairs`: host memory for decoded image files (300 images of 1 MP)
_AT_ONCE_WINDOW = 16  # `Runner.match_streams` > 1: pairs per session (their results are yielded when all are done)


@dataclass(frozen=True)
class VerifiedMatches:
    """Matches of an image pair with their geometric verification (`verify` of a matching result).

    Attributes:
        points1: `[N, 2]` float32 `(x, y)` pixels in the first image at its input resolution.
        points2: `[N, 2]` the matching points in the second image.
        scores: `[N]` float32 confidence in `[0, 1]`: the certainty at the pixel (dense), the match score (sparse).
        inliers: `[N]` bool: the matches that agree with `geometry`.
        geometry: The `RelativePose` (`R`, `t`, `E`, `ok`) if intrinsics were given, else the `FundamentalMatrix`
            (`F`, `ok`).
    """

    points1: torch.Tensor
    points2: torch.Tensor
    scores: torch.Tensor
    inliers: torch.Tensor
    geometry: RelativePose | FundamentalMatrix


def _verify(  # noqa: PLR0913, PLR0917 - the matches and the estimators' arguments
    points1: torch.Tensor,
    points2: torch.Tensor,
    scores: torch.Tensor,
    k1: torch.Tensor | None,
    k2: torch.Tensor | None,
    threshold: float | None,
    mode: Mode,
) -> VerifiedMatches:
    """`estimate_pose` (with intrinsics) or `estimate_fundamental` of the matches."""
    if (k1 is None) != (k2 is None):
        msg = "give both intrinsics k1 and k2 (relative pose) or neither (fundamental matrix)"
        raise InputError(msg)
    options: dict[str, Any] = {"mode": mode} | ({} if threshold is None else {"threshold": threshold})
    if k1 is None or k2 is None:
        geometry = estimate_fundamental(points1, points2, **options)
    else:
        geometry = estimate_pose(points1, points2, k1, k2, **options)
    return VerifiedMatches(points1, points2, scores, geometry.inliers, geometry)


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

    def matches(
        self, n: int = 5000, *, threshold: float = 0.05, seed: int = 0, balanced: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample `n` keypoint matches from the warp, for `estimate_pose`.

        Pixels are drawn without replacement with a probability proportional to their certainty, clipped to 1 above
        `threshold`; without a certainty map, uniformly. `balanced` (upstream RoMa's sampling) draws `4 n` candidates
        this way and `n` of them with a probability inverse to their density (a Gaussian kernel of 0.1 of the image
        size over the match coordinates, candidates with fewer than about 10 close neighbours excluded), which
        spreads the matches over the image.

        Args:
            n: Matches per pair (fewer if fewer pixels have a non-zero certainty).
            threshold: Certainty from which pixels are drawn with equal probability.
            seed: Of the sampling.
            balanced: Density balancing of the candidates (see above).

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
        if balanced:
            m = min(4 * n, int((weight > 0).sum(1).min()))
            cand = _draw(weight, m, gen)  # [B, 4 n]
            xy = torch.cat((torch.stack((cand % w, cand // w), -1).float(), _at(warp, cand)), -1)
            density = _density(xy / torch.tensor([w, h, w, h], device=warp.device) * 2 - 1)
            p = torch.where(density < _MIN_DENSITY, 1e-7, 1 / (density + 1))
            ind = torch.gather(cand, 1, _draw(p, n, gen))
        else:
            ind = _draw(weight, n, gen)  # [B, n]
        xy1 = torch.stack((ind % w, ind // w), -1).float()
        xy2 = _at(warp, ind)
        if self.warp.dim() == 3:  # noqa: PLR2004
            return xy1[0], xy2[0]
        return xy1, xy2

    def verify(  # noqa: PLR0913 - the intrinsics and the options of `matches` and the estimator
        self,
        k1: torch.Tensor | None = None,
        k2: torch.Tensor | None = None,
        *,
        n: int = 5000,
        seed: int = 0,
        threshold: float | None = None,
        mode: Mode = "auto",
    ) -> VerifiedMatches:
        """Sampled matches and their geometric verification in one call: `matches` + `estimate_pose` / `_fundamental`.

        ```python
        v = model(image_a, image_b).verify()  # or verify(K1, K2): relative pose instead of a fundamental matrix
        keep = v.inliers & (v.scores > 0.5)
        v.points1[keep], v.points2[keep], v.geometry.F  # verified matches in pixels of the two input images
        ```

        Args:
            k1: Intrinsics `[3, 3]` of the first image (pixels); with both, the relative pose is estimated
                (`estimate_pose`), without, the fundamental matrix (`estimate_fundamental`).
            k2: Intrinsics of the second image.
            n: Matches to sample (`matches`).
            seed: Of the sampling.
            threshold: Inlier threshold of the estimator in pixels; `None` is its default.
            mode: Backend of the estimator.

        Returns:
            The matches, their certainty, the inlier mask and the geometry (batched like the result).
        """
        points1, points2 = self.matches(n, seed=seed)
        if self.certainty is None:
            scores = torch.ones(points1.shape[:-1], device=points1.device)
        else:
            flat = (points1[..., 1] * self.warp.shape[-1] + points1[..., 0]).long()
            scores = self.certainty.flatten(-2).gather(-1, flat)
        return _verify(points1, points2, scores, k1, k2, threshold, mode)


_MIN_DENSITY = 10.0  # upstream RoMa: candidates with fewer close neighbours are (nearly) never drawn
_DENSITY_REFERENCES = 2048  # points the density is estimated from


def _draw(weight: torch.Tensor, n: int, gen: torch.Generator) -> torch.Tensor:
    """`n` indices per row of `weight` `[B, K]` without replacement, with a probability proportional to the weight.

    Gumbel top-k: the same distribution as `torch.multinomial`, whose first CUDA call costs 40-130 ms.
    """
    u = torch.rand(weight.shape, device=weight.device, generator=gen).clamp_min(1e-20)
    return (weight.log() - (-u.log()).log()).topk(n, dim=1).indices


def _at(warp: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """The warp `[B, 2, H, W]` at the flat pixel indices `ind` `[B, n]`: `[B, n, 2]`."""
    b, n = ind.shape
    return torch.gather(warp.flatten(2), 2, ind[:, None].expand(b, 2, n)).transpose(1, 2)


def _density(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """Upstream RoMa's `kde`: Gaussian kernel density `[B, n]` of `x` `[B, n, 4]`, from `_DENSITY_REFERENCES` of them.

    The kernel sum runs over every `n / _DENSITY_REFERENCES`-th point (the points come in random order) and is scaled
    back: linear in `n` instead of upstream's `n^2` (4 ms and 175 MB instead of 29 ms and 320 MB for 20 000 points on
    an RTX 5060 Ti, the same pose accuracy).
    """
    x = x.half() if x.is_cuda else x.float()
    step = max(1, x.shape[1] // _DENSITY_REFERENCES)
    return (-(torch.cdist(x, x[:, ::step]) ** 2) / (2 * std**2)).exp().sum(-1) * step


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

    def verify(
        self,
        k1: torch.Tensor | None = None,
        k2: torch.Tensor | None = None,
        *,
        threshold: float | None = None,
        mode: Mode = "auto",
    ) -> VerifiedMatches:
        """All matches and their geometric verification; see `MatchingResult.verify` (`threshold`: the estimator's)."""
        return _verify(*self.matches(), self.scores, k1, k2, threshold, mode)


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

    def match_pairs(
        self, images: Sequence[ImageLike], pairs: Iterable[tuple[int, int]], *, cache_mb: float = CACHE_MB
    ) -> Iterator[MatchingResult | SparseMatchingResult]:
        """Match many pairs of an image set, computing each image's own part once (RoMa v2: its DINOv3 descriptor).

        ```python
        for (i, j), result in zip(pairs, model.match_pairs(paths, pairs)):
            v = result.verify()  # points, scores, RANSAC inliers, fundamental matrix
        ```

        The results equal `model(images[i], images[j])` bit for bit and come in the order of `pairs`. The per-image
        parts are kept on the device while later pairs need them, within `cache_mb`: over it, the image whose next
        pair is furthest away is dropped and computed again when that pair comes (list each image's pairs close
        together to avoid it). Families without a per-image part just run the pairs.

        Args:
            images: The image set (any `ImageLike`, single images; files are decoded once, in background threads
                ahead of their pairs, and kept in host memory while later pairs need them).
            pairs: `(i, j)` indices into `images`: match image `i` into image `j`.
            cache_mb: Device memory for the kept per-image parts, in MB (RoMa v2: 19.5 MB per image; the two images
                of the running pair come on top).

        Yields:
            The result of each pair, as a call returns it.

        Raises:
            InputError: Unsupported or too small images.
        """
        pairs = [(int(i), int(j)) for i, j in pairs]
        cache = PairCache(pairs, int(cache_mb * 2**20))
        lazy = self.sparse and not self._runner.match_reads_images
        with DecodedImages(images, pairs, int(DECODED_MB * 2**20), lambda i: lazy and i in cache.kept) as decoded:
            yield from self._match_pairs(images, pairs, cache, decoded, lazy=lazy)

    def _match_pairs(
        self,
        images: Sequence[ImageLike],
        pairs: list[tuple[int, int]],
        cache: PairCache,
        decoded: DecodedImages,
        *,
        lazy: bool,
    ) -> Iterator[MatchingResult | SparseMatchingResult]:
        """`lazy`: an image whose `describe` result is kept is not read (its size was checked when it was described)."""
        streams = _match_streams(self, lazy=lazy)
        if streams > 1 or (lazy and self._runner.matches_many):
            yield from self._match_pairs_at_once(pairs, cache, decoded, streams)
            return
        for pair in pairs:
            image1 = images[pair[0]]
            t1, t2 = (None if lazy and i in cache.kept else self._checked(decoded.get(i)) for i in pair)
            decoded.done(pair)
            with self._session():
                if self.sparse:
                    result = self._run_sparse(t1, t2, cache, pair)
                else:
                    assert t1 is not None  # noqa: S101 - `lazy` is sparse only
                    assert t2 is not None  # noqa: S101
                    result = self._run(t1, t2, cache, pair)
                cache.done(pair)
            if isinstance(result, dict):
                maps = unbatch(result, image1)
                result = MatchingResult(
                    maps["warp"], maps.get("certainty"), self.variant.ref, self.backend, self.options
                )
            yield result

    def _match_pairs_at_once(
        self, pairs: list[tuple[int, int]], cache: PairCache, decoded: DecodedImages, streams: int
    ) -> Iterator[SparseMatchingResult]:
        """`_match_pairs` (lazy), `streams` pairs at a time.

        `describe` runs here in pair order and each pair's `match` starts on one of `streams` threads with a CUDA
        stream each (`Runner.match_streams`) as soon as its images are described; one session per
        `_AT_ONCE_WINDOW` pairs. The same kernels: equal results. A runner with `match_many` gets the window's
        pairs in one call instead.
        """
        cuda = self.device.type == "cuda" and not self._runner.matches_many
        local = threading.local()

        def match(described: tuple[Any, Any], ready: torch.cuda.Event | None) -> dict[str, torch.Tensor]:
            if not cuda:
                with torch.inference_mode():
                    return self._runner.match(None, None, described)  # ty: ignore[invalid-argument-type]
            if not hasattr(local, "stream"):
                local.stream = torch.cuda.Stream(self.device)
            with torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(local.stream):
                local.stream.wait_event(ready)
                out = self._runner.match(None, None, described)  # ty: ignore[invalid-argument-type]
            local.stream.synchronize()  # before the inputs or outputs are freed or read on another stream
            return out

        io = self.variant.io
        with ThreadPoolExecutor(streams) as pool:
            for start in range(0, len(pairs), _AT_ONCE_WINDOW):
                futures = []
                many = []
                with self._session():
                    for pair in pairs[start : start + _AT_ONCE_WINDOW]:
                        images = [None if i in cache.kept else self._checked(decoded.get(i)) for i in pair]
                        decoded.done(pair)
                        described = tuple(
                            cache.get(i, lambda x=x: self._runner.describe(scale(x, io)))  # ty: ignore[invalid-argument-type] - not kept
                            for i, x in zip(pair, images, strict=True)
                        )
                        cache.done(pair)
                        if self._runner.matches_many:
                            many.append(described)
                            continue
                        ready = None
                        if cuda:
                            ready = torch.cuda.Event()
                            ready.record()
                        futures.append(pool.submit(match, described, ready))
                    outs = self._runner.match_many(many) if many else [f.result() for f in futures]
                for out in outs:
                    yield self._sparse_result(out)

    def _matched(
        self, image1: torch.Tensor | None, image2: torch.Tensor | None, cache: PairCache | None, pair: tuple[int, int]
    ) -> dict[str, torch.Tensor]:
        """Scale and run one chunk; `cache` (`match_pairs`) holds the runner's per-image parts of the images `pair`.

        An image is `None` (`match_pairs`, a runner whose `match` does not read the images) if its part is kept.
        """
        io = self.variant.io
        image1, image2 = (None if x is None else scale(x, io) for x in (image1, image2))
        if cache is None:
            return self._runner(image1, image2)  # ty: ignore[invalid-argument-type] - images without a cache
        d1 = cache.get(pair[0], lambda: self._runner.describe(image1))  # ty: ignore[invalid-argument-type] - not kept
        d2 = cache.get(pair[1], lambda: self._runner.describe(image2))  # ty: ignore[invalid-argument-type]
        if d1 is None:
            return self._runner(image1, image2)  # ty: ignore[invalid-argument-type] - not lazy without `describe`
        return self._runner.match(image1, image2, (d1, d2))  # ty: ignore[invalid-argument-type]

    def _checked(self, image: ImageLike) -> torch.Tensor:
        """`image` on the device as `[1, 3, H, W]`, its size checked."""
        t = to_rgb_batch(image, self.device)
        self._check_size(t)
        return t

    def _run(
        self, image1: torch.Tensor, image2: torch.Tensor, cache: PairCache | None = None, pair: tuple[int, int] = (0, 0)
    ) -> dict[str, torch.Tensor]:
        """Scale, run and check one chunk."""
        out = self._matched(image1, image2, cache, pair)
        b, (h, w) = image1.shape[0], image1.shape[-2:]
        return self._check_outputs(out, {"warp": (b, 2, h, w), "certainty": (b, h, w)})

    def _run_sparse(
        self,
        image1: torch.Tensor | None,
        image2: torch.Tensor | None,
        cache: PairCache | None = None,
        pair: tuple[int, int] = (0, 0),
    ) -> SparseMatchingResult:
        """Scale, run and check one pair `[1, 3, H, W]`."""
        return self._sparse_result(self._matched(image1, image2, cache, pair))

    def _sparse_result(self, out: dict[str, torch.Tensor]) -> SparseMatchingResult:
        """The checked result of a sparse runner's output."""
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


def _match_streams(model: MatchingModel, *, lazy: bool) -> int:
    """Pairs `match_pairs` matches at once: `Runner.match_streams` on CUDA if `match` does not read the images."""
    return model._runner.match_streams if lazy and model.device.type == "cuda" else 1  # noqa: SLF001 - its runner


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
