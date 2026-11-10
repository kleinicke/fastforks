"""Kernels CUDA runners of the top-down pose families (ViTPose, RTMPose) away from the golden's single call.

A graph capture that fails marks its shape eager without a word (`GraphCache._failed`), and the goldens make one call
with two people and the default options, so they cannot see: whether the graph is really captured and replayed with
new crops, the zero crops padding a bucket, more people than the largest bucket, and RTMPose's `flip_test`.

Keypoint comparisons run on the golden's photo and person boxes (noise crops have no keypoints: their argmax jumps
between equal peaks) and, as the goldens, on the keypoints the fp32 portable runner scores at least `gpu_min_score`,
within the family's golden.toml tolerance (mean and 99th percentile, px).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fastforks import pose
from fastforks._core.errors import WeightsError
from fastforks._core.half import graph_sized
from fastforks.pose._runner import BUCKETS

REFS = ["vitpose-base-simple@1", "vitpose-plus-small@1", "rtmpose-t@1"]  # both ViTPose decoders, ViTPose++, SimCC
BOXES = torch.tensor([[10.0, 5.0, 200.0, 300.0], [150.0, 40.0, 330.0, 290.0], [0.0, 0.0, 320.0, 320.0]])

pytestmark = [pytest.mark.model, pytest.mark.gpu]


def _load(ref: str, mode: str = "kernels", **kw: Any) -> pose.PoseModel:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    try:
        return pose.load(ref, mode=mode, device="cuda", **kw)
    except WeightsError as e:
        pytest.skip(f"{ref} weights not available: {e}")


def _photo(ref: str) -> tuple[np.ndarray, np.ndarray]:
    """The golden's input photo (RGB uint8) and its person boxes `[2, 4]`."""
    root = os.environ.get("FASTFORKS_GOLDEN_PATH")
    path = Path(root or ".") / f"{ref}.npz"
    if not root or not path.is_file():
        pytest.skip(f"golden {ref} not available (set FASTFORKS_GOLDEN_PATH)")
    with np.load(path) as z:
        return z["input"].round().astype(np.uint8), z["boxes"].astype(np.float32)


def _people(boxes: np.ndarray, count: int) -> np.ndarray:
    """`count` person boxes: the golden's, then copies moved by up to 6 px."""
    rng = np.random.default_rng(0)
    more = [boxes + rng.uniform(-6, 6, boxes.shape).astype(np.float32) for _ in range(-(-count // len(boxes)) - 1)]
    return np.concatenate([boxes, *more])[:count]


def _assert_same_pose(model: pose.PoseModel, got: pose.PoseResult, want: pose.PoseResult, what: str) -> None:
    """`got` within the family's GPU golden tolerance of `want` on the keypoints `want` is sure of."""
    _assert_close(model, (got.keypoints, got.keypoint_scores), (want.keypoints, want.keypoint_scores), what)


Pose = tuple[torch.Tensor, torch.Tensor]  # keypoints `[P, K, 2]`, keypoint scores `[P, K]`


def _assert_close(model: pose.PoseModel, got: Pose, want: Pose, what: str) -> None:
    with (Path(pose.__file__).parent / "models" / model.card.family / "golden.toml").open("rb") as f:
        tol = tomllib.load(f)["tolerance"]
    (kp, score), (want_kp, want_score) = got, want
    sure = want_score >= tol["gpu_min_score"]
    assert float(sure.float().mean()) >= 1 / 3, f"{what}: the score mask keeps {float(sure.float().mean()):.0%}"
    err = (kp - want_kp).abs()[sure].float().cpu().numpy()
    assert err.mean() < tol["gpu_mean"], f"{what}: mean |diff| {err.mean():.4f} px"
    assert np.percentile(err, 99) < tol["gpu_p99"], f"{what}: p99 |diff| {np.percentile(err, 99):.4f} px"
    extra = {k: (mean, p99) for k, mean, p99 in tol["gpu_extra"]}
    e = ((score - want_score).abs() / want_score.abs().median()).cpu().numpy()
    mean, p99 = extra["keypoint_scores"]
    assert e.mean() < mean, f"{what}: keypoint score mean |diff| {e.mean():.4f}"
    assert np.percentile(e, 99) < p99, f"{what}: keypoint score p99 |diff| {np.percentile(e, 99):.4f}"


def _eager(r: Any, img: torch.Tensor, boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    """An eager call with the GEMM dispatch of a graph-sized call (`graph_sized`), the graph cache off."""
    cache, r.graphs.cache = r.graphs.cache, None
    try:
        with graph_sized():
            return r(img, boxes)
    finally:
        r.graphs.cache = cache


@pytest.mark.parametrize("ref", REFS)
def test_replay_matches_eager(ref: str) -> None:
    model = _load(ref)
    r = model._runner
    cache = r.graphs.cache
    assert cache is not None
    gen = torch.Generator("cuda").manual_seed(0)
    x, y = (torch.rand(1, 3, 320, 336, device="cuda", generator=gen) for _ in range(2))
    boxes = BOXES.cuda()
    with torch.inference_mode():
        want_x = r(x, boxes)  # first call: eager, with the GEMM dispatch of a capture (`graph_sized`)
        first = r(x, boxes)  # second call: captures (3 people -> the 4 bucket)
        assert len(cache.graphs) == 1, f"no graph captured (failed: {list(cache._failed)})"
        second = r(y, boxes)  # replay with new data
        want_y = _eager(r, y, boxes)
        for got, want in ((first, want_x), (second, want_y)):
            for k in ("keypoints", "keypoint_scores"):
                torch.testing.assert_close(got[k], want[k], rtol=0, atol=0)
    model.release_graphs()
    assert not cache.graphs


@pytest.mark.parametrize("ref", REFS)
def test_zero_crops_of_a_bucket_leave_the_people_alone(ref: str) -> None:
    """3 people run in the 4 bucket beside a zero crop, each of them alone in the 1 bucket: the same pose.

    Not bit-equal: the fp16 kernels differ per batch size (cuDNN / GEMM picks).
    """
    model = _load(ref)
    img, boxes = _photo(ref)
    boxes = _people(boxes, 3)
    together = model(img, boxes=boxes)
    alone = [model(img, boxes=boxes[i : i + 1]) for i in range(3)]
    got = (torch.cat([p.keypoints for p in alone]), torch.cat([p.keypoint_scores for p in alone]))
    # the three together: one of the golden's two boxes alone has too few sure keypoints for the percentiles
    _assert_close(model, got, (together.keypoints, together.keypoint_scores), f"{ref} alone")
    model.release_graphs()


@pytest.mark.parametrize("ref", ["vitpose-base-simple@1", "rtmpose-t@1"])
def test_more_people_than_the_largest_bucket(ref: str) -> None:
    """70 people: 32 + 32 + 6 (in the 8 bucket), each chunk through its graph, against the fp32 portable runner."""
    people = 2 * BUCKETS[-1] + 6
    img, boxes = _photo(ref)
    boxes = torch.from_numpy(_people(boxes, people)).cuda()
    want = _load(ref, "portable", dtype="float32")(img, boxes=boxes)
    model = _load(ref)
    cache = model._runner.graphs.cache
    assert cache is not None
    for call in range(3):  # eager, capture, replay
        got = model(img, boxes=boxes)
        assert got.keypoints.shape[0] == people
        _assert_same_pose(model, got, want, f"{ref} call {call}")
    assert len(cache.graphs) == 2, f"graphs {list(cache.graphs)}, failed {list(cache._failed)}"  # the 32 and 8 buckets
    assert not cache._failed
    kept = got.boxes.clone()
    assert got.boxes is not boxes
    boxes.zero_()  # the result owns its boxes
    assert torch.equal(got.boxes, kept)
    model.release_graphs()


@pytest.mark.parametrize("ref", ["rtmpose-t@1", "dwpose-l-384@1"])
def test_flip_test_matches_portable(ref: str) -> None:
    """RTMPose `flip_test=True` (off in the goldens): the Kernels runner against the fp32 portable one."""
    img, boxes = _photo(ref)
    flip = {"flip_test": True}
    want = _load(ref, "portable", dtype="float32", options=flip)(img, boxes=boxes)
    plain = _load(ref, "portable", dtype="float32")(img, boxes=boxes)
    assert not torch.equal(want.keypoint_scores, plain.keypoint_scores)  # the option reaches the runner
    model = _load(ref, options=flip)
    for call in range(3):  # eager, capture, replay
        _assert_same_pose(model, model(img, boxes=boxes), want, f"{ref} flip_test call {call}")
    cache = model._runner.graphs.cache
    assert cache is not None
    assert len(cache.graphs) == 1, f"no graph captured (failed: {list(cache._failed)})"
    model.release_graphs()
