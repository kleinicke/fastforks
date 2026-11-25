"""Top-down pose on a batch: `TopDown.estimate_batch` (every person at once) and how the core groups the images."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

from fastforks.pose import _api
from fastforks.pose._runner import PEOPLE, TopDown


class _Fake(TopDown):
    """Crops: the image's value at the box corner; keypoints: crop + centre, so each person shows where it came from."""

    def __init__(self) -> None:  # no checkpoint
        self.calls: list[int] = []

    def regions(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return boxes[:, :2], boxes[:, 2:] - boxes[:, :2]

    def crops(self, image: torch.Tensor, centre: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        del size
        return image[0, 0, centre[:, 1].long(), centre[:, 0].long()][:, None]  # [P, 1]

    def people(self, crops: torch.Tensor, centre: torch.Tensor, size: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls.append(crops.shape[0])
        kp = (crops[:, :, None] + centre[:, None]).expand(-1, 3, -1)  # [P, 3, 2]
        return {"keypoints": kp, "keypoint_scores": kp[..., 0], "scores": size[:, 0]}


def _boxes(n: int, seed: int) -> torch.Tensor:
    xy = torch.randint(0, 8, (n, 2), generator=torch.Generator().manual_seed(seed)).float()
    return torch.cat([xy, xy + 1 + seed], 1)


def test_estimate_batch_is_one_people_call_and_equals_per_image_calls() -> None:
    runner = _Fake()
    images = torch.rand(3, 3, 8, 8)
    boxes = [_boxes(2, 0), _boxes(1, 1), _boxes(3, 2)]
    got = runner.estimate_batch(images, boxes)
    assert runner.calls == [6]
    for i, (g, b) in enumerate(zip(got, boxes, strict=True)):
        want = runner(images[i : i + 1], b)
        assert set(g) == set(want)
        for k in want:
            assert torch.equal(g[k], want[k])


def test_the_core_groups_images_up_to_people_boxes_and_skips_the_empty_ones() -> None:
    runner = _Fake()
    model = SimpleNamespace(
        _runner=runner,
        _session=contextlib.nullcontext,
        _empty=lambda x: {
            "keypoints": x.new_zeros(0, 3, 2),
            "keypoint_scores": x.new_zeros(0, 3),
            "scores": x.new_zeros(0),
        },
    )
    counts = [PEOPLE - 1, 0, 2, PEOPLE + 3, 1]
    images = torch.rand(len(counts), 3, 8, 8)
    boxes = [_boxes(n, i) for i, n in enumerate(counts)]
    outs = _api.PoseModel._raw_batch(model, images, boxes)  # ty: ignore[invalid-argument-type]
    assert runner.calls == [PEOPLE - 1, 2, PEOPLE + 3, 1]  # an image never splits; a full group starts a new call
    assert [o["keypoints"].shape[0] for o in outs] == counts
    for i, (o, b) in enumerate(zip(outs, boxes, strict=True)):
        if b.shape[0]:
            assert torch.equal(o["keypoints"], runner(images[i : i + 1], b)["keypoints"])
