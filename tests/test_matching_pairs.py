"""`MatchingModel.match_pairs` and its `PairCache`: results independent of the cache limit, Belady eviction."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import torch

from fastforks import matching
from fastforks._core import registry
from fastforks.matching._pairs import PairCache
from fastforks.matching._runner import Runner


def run(pairs: list[tuple[int, int]], limit: int, size: int = 10) -> tuple[PairCache, list[tuple[int, int]], int]:
    """The cache over `pairs` with results of `size` bytes: it, what every pair got, the most bytes after a pair."""
    cache = PairCache(pairs, limit, size=lambda _: size)
    got, peak = [], 0
    for pair in pairs:
        got.append((cache.get(pair[0], lambda i=pair[0]: i), cache.get(pair[1], lambda j=pair[1]: j)))
        cache.done(pair)
        peak = max(peak, cache.bytes)
        assert cache.bytes == size * len(cache.kept) <= max(limit, 0)
    assert not cache.kept  # every image is freed after its last pair
    return cache, got, peak


def lru_misses(pairs: list[tuple[int, int]], count: int) -> int:
    """Misses of a least-recently-used cache of `count` images."""
    kept: list[int] = []
    misses = 0
    for pair in pairs:
        for i in pair:
            misses += i not in kept
            kept = [*[k for k in kept if k != i], i][-count:]
    return misses


def test_pair_cache_frees_after_the_last_pair():
    """Sequential pairs need two descriptors at a time, whatever the limit allows."""
    pairs = [(i, i + 1) for i in range(50)]
    cache, got, peak = run(pairs, limit=1000)
    assert got == pairs
    assert (cache.misses, cache.hits) == (51, 49)  # every image once
    assert peak == 10  # after a pair only its second image has a later pair


@pytest.mark.parametrize("limit", [0, 10, 30, 100, 10**6])
def test_pair_cache_results_do_not_depend_on_the_limit(limit: int):
    gen = np.random.default_rng(0)
    pairs = [(int(a), int(b)) for a, b in gen.integers(0, 40, (300, 2))]
    cache, got, _ = run(pairs, limit)
    assert got == pairs
    assert cache.hits + cache.misses == 2 * len(pairs)
    if limit == 10**6:
        assert cache.misses == len({i for p in pairs for i in p})


def test_pair_cache_drops_the_furthest_next_use():
    """Belady: with room for two, image 0 (needed again at once) stays, 1 and 2 (needed late) go first."""
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (3, 4), (1, 2)]
    cache, _, _ = run(pairs, limit=20)
    assert cache.misses == 7  # 0-4 once, 1 and 2 again; an LRU of two images recomputes more
    assert lru_misses(pairs, 2) > cache.misses
    gen = np.random.default_rng(1)  # retrieval-style: 200 images, 5 neighbours each
    pairs = [(i, int(j)) for i in range(200) for j in gen.integers(0, 200, 5)]
    assert run(pairs, limit=80)[0].misses < lru_misses(pairs, 8)


@dataclasses.dataclass(frozen=True)
class Params:
    coarse_size: int
    fine_size: int
    keep_descriptors: int = 0


class Dummy(Runner):
    """Warp = the pixel grid shifted by the two images' red means; the mean of an image is its per-image part."""

    Params = Params
    described = 0

    def __call__(self, image1, image2):
        return self.match(image1, image2, (image1.mean(), image2.mean()))

    def describe(self, image):
        Dummy.described += 1
        return image.mean()

    def match(self, image1, image2, described):  # noqa: ARG002 - test stub
        b, _, h, w = image1.shape
        ys, xs = torch.meshgrid(torch.arange(h) + described[0], torch.arange(w) + described[1], indexing="ij")
        return {"warp": torch.stack((xs, ys)).expand(b, 2, h, w), "certainty": torch.ones(b, h, w)}


@pytest.mark.parametrize("cache_mb", [0.0, 512.0])
def test_match_pairs_equals_single_calls(monkeypatch, cache_mb: float):
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": Dummy}))
    model = matching.load("romav2", device="cpu", mode="portable")
    images = [np.full((16 + i, 24, 3), 20 * i, np.uint8) for i in range(5)]
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (2, 0), (4, 4)]
    Dummy.described = 0
    results = list(model.match_pairs(images, pairs, cache_mb=cache_mb))
    assert Dummy.described == (5 if cache_mb else 11)  # (4, 4): one image
    for (i, j), result in zip(pairs, results, strict=True):
        single = model(images[i], images[j])
        assert torch.equal(result.warp, single.warp)
        assert torch.equal(result.certainty, single.certainty)
        assert result.warp.shape == (2, 16 + i, 24)
