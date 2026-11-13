"""`PairCache`: the per-image stage of a matcher (`Runner.describe`) over a known list of image pairs.

`DecodedImages`: the image files of such a list, decoded once in background threads ahead of their pairs.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import torch

from fastforks._core.image import _FILE, decode_image

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from fastforks._core.image import ImageLike


def nbytes(value: object) -> int:
    """Bytes of the tensors in `value` (a tensor, or lists / tuples / dict values of them)."""
    if isinstance(value, torch.Tensor):
        return value.nbytes
    if isinstance(value, dict):
        value = list(value.values())
    return sum(nbytes(v) for v in value) if isinstance(value, (list, tuple)) else 0


class PairCache:
    """Keeps per-image results while later pairs need them, within `limit` bytes.

    The whole pair list is known, so the image whose next pair is furthest away is dropped first (Belady's rule: the
    fewest recomputations any cache of this size can reach) and an image is freed after its last pair. A dropped
    image is computed again when its next pair comes: the limit changes the time, never the results.
    """

    def __init__(self, pairs: Sequence[tuple[int, int]], limit: int, size: Callable[[Any], int] = nbytes) -> None:
        """`pairs`: image indices in the order `done` is called; `size`: the bytes of a result."""
        self.limit, self.size = limit, size
        self.uses: dict[int, list[int]] = {}  # image -> positions of its remaining pairs, the next one last
        for position in range(len(pairs) - 1, -1, -1):
            for image in set(pairs[position]):
                self.uses.setdefault(image, []).append(position)
        self.kept: dict[int, Any] = {}
        self.bytes = self.hits = self.misses = 0

    def get(self, image: int, compute: Callable[[], Any]) -> Any:  # noqa: ANN401 - whatever `compute` returns
        """The kept result of `image`, or `compute()` (kept at least until `done`)."""
        if image in self.kept:
            self.hits += 1
            return self.kept[image]
        self.misses += 1
        value = self.kept[image] = compute()
        self.bytes += self.size(value)
        return value

    def done(self, pair: tuple[int, int]) -> None:
        """After the next pair of the list: free its images without later pairs, then down to the limit."""
        for image in set(pair):
            self.uses[image].pop()
            if not self.uses[image]:
                self._drop(image)
        while self.bytes > self.limit and self.kept:
            self._drop(max(self.kept, key=lambda image: self.uses[image][-1]))

    def _drop(self, image: int) -> None:
        if image in self.kept:
            self.bytes -= self.size(self.kept.pop(image))


class DecodedImages:
    """The files among `images` decoded in background threads ahead of `pairs`, each kept while later pairs need it.

    Decoded arrays stay in host memory within `limit` bytes (the `PairCache` rule; a dropped file is decoded again);
    other inputs (arrays, tensors) are passed through. Use as a context manager; `done` after each pair, in order.
    """

    def __init__(
        self,
        images: Sequence[ImageLike],
        pairs: Sequence[tuple[int, int]],
        limit: int,
        skip: Callable[[int], bool] = lambda _: False,
        ahead: int = 8,
    ) -> None:
        """`skip(image)`: not worth decoding ahead now; `ahead`: files decoding at most at once, for the next pairs."""
        self.images, self.pairs, self.skip, self.ahead, self.position = images, pairs, skip, ahead, 0
        self.cache = PairCache(pairs, limit, lambda a: a.nbytes if isinstance(a, np.ndarray) else 0)
        self.pending: dict[int, Future[np.ndarray]] = {}
        self.pool = ThreadPoolExecutor(4, thread_name_prefix="fastforks-read")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    def get(self, image: int) -> ImageLike:
        """An image of the next pair, decoded if it is a file."""
        for later in self.pairs[self.position : self.position + self.ahead]:
            for i in later:
                source = self.images[i]
                if len(self.pending) < self.ahead and isinstance(source, _FILE) and not self._known(i):
                    self.pending[i] = self.pool.submit(decode_image, source)
        source = self.images[image]
        if not isinstance(source, _FILE):
            return source
        future = self.pending.pop(image, None)
        return self.cache.get(image, future.result if future else lambda: decode_image(source))

    def done(self, pair: tuple[int, int]) -> None:
        """After each pair (with or without `get`): free files without later pairs, then down to the limit."""
        self.cache.done(pair)
        self.position += 1

    def _known(self, image: int) -> bool:
        return image in self.cache.kept or image in self.pending or self.skip(image)
