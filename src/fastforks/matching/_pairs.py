"""`PairCache`: the per-image stage of a matcher (`Runner.describe`) over a known list of image pairs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def nbytes(value: object) -> int:
    """Bytes of the tensors in `value` (a tensor, or lists / tuples of them)."""
    if isinstance(value, torch.Tensor):
        return value.nbytes
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
