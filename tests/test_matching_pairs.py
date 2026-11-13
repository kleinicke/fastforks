"""`MatchingModel.match_pairs` and its `PairCache`: results independent of the cache limit, Belady eviction."""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import torch
from PIL import Image

from fastforks import matching
from fastforks._core import registry
from fastforks._core.image import decode_image
from fastforks.matching import _api as matching_api
from fastforks.matching import _pairs
from fastforks.matching._pairs import PairCache, nbytes
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


def test_nbytes_counts_dict_values():
    value = {"keypoints": torch.zeros(4, 2), "descriptors": [torch.zeros(4, 8)], "name": "x"}
    assert nbytes(value) == 4 * (8 + 32)


@dataclasses.dataclass(frozen=True)
class SparseParams:
    extractor: str
    max_keypoints: int
    detection_threshold: float
    long_side: int


class SparseDummy(Runner):
    """Keypoints = the image corners scaled by its red mean; `match` does not read the images."""

    Params = SparseParams
    match_reads_images = False

    def __call__(self, image1, image2):
        return self.match(image1, image2, (self.describe(image1), self.describe(image2)))

    def describe(self, image):
        return {"keypoints": torch.tensor([[0.0, 0.0], [1.0, 1.0]]) * image[0, 0].mean() + image.shape[-1]}

    def match(self, image1, image2, described):  # noqa: ARG002 - test stub
        return {
            "keypoints1": described[0]["keypoints"],
            "keypoints2": described[1]["keypoints"],
            "pairs": torch.tensor([[0, 0], [1, 1]]),
            "scores": torch.ones(2),
        }


@pytest.mark.parametrize(
    ("runner", "decoded_mb", "once"), [(Dummy, 1024.0, True), (Dummy, 0.0, False), (SparseDummy, 0.0, True)]
)
def test_match_pairs_decodes_files_once(monkeypatch, tmp_path, runner, decoded_mb: float, *, once: bool):
    """Files are decoded ahead and kept while later pairs need them (within `DECODED_MB`); a runner whose `match`
    does not read the images needs no file again once its `describe` result is kept."""
    card = "romav2" if runner is Dummy else "aliked-lightglue"
    monkeypatch.setattr(registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": runner}))
    monkeypatch.setattr(matching_api, "DECODED_MB", decoded_mb)
    model = matching.load(card, device="cpu", mode="portable")
    arrays = [np.full((64 + i, 72, 3), 20 * i, np.uint8) for i in range(5)]
    paths = [tmp_path / f"{i}.png" for i in range(5)]
    for path, array in zip(paths, arrays, strict=True):
        Image.fromarray(array).save(path)
    calls = []
    monkeypatch.setattr(_pairs, "decode_image", lambda p: calls.append(p) or decode_image(p))
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (2, 0), (4, 4), (0, 3)]
    results = list(model.match_pairs(paths, pairs))
    assert (len(calls) == len(paths)) == once
    for result, expected in zip(results, model.match_pairs(arrays, pairs), strict=True):
        for name in ("keypoints1", "keypoints2") if model.sparse else ("warp",):
            assert torch.equal(getattr(result, name), getattr(expected, name))


@pytest.mark.parametrize("streams", [2, 3])
def test_match_pairs_at_once_equals_one_at_a_time(monkeypatch, streams: int):
    """`Runner.match_streams` pairs at a time (CUDA streams; plain threads here): the same results in pair order."""
    monkeypatch.setattr(
        registry, "import_model", lambda _card: types.SimpleNamespace(RUNNERS={"portable": SparseDummy})
    )
    model = matching.load("aliked-lightglue", device="cpu", mode="portable")
    images = [np.full((64 + i, 72, 3), 20 * i, np.uint8) for i in range(5)]
    pairs = [(0, 1), (1, 2), (0, 2), (3, 4), (2, 0), (4, 4), (0, 3)]
    expected = list(model.match_pairs(images, pairs, cache_mb=0.0))
    monkeypatch.setattr(matching_api, "_match_streams", lambda _model, *, lazy: streams if lazy else 1)
    for cache_mb in (0.0, 512.0):
        results = list(model.match_pairs(images, pairs, cache_mb=cache_mb))
        assert len(results) == len(pairs)
        for result, want in zip(results, expected, strict=True):
            assert torch.equal(result.keypoints1, want.keypoints1)
            assert torch.equal(result.keypoints2, want.keypoints2)
            assert torch.equal(result.pairs, want.pairs)
