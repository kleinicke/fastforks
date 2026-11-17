"""Weight-free checks of RetinaFace's anchors, merged heads, upsampling, decoding and NMS against upstream's rules."""

from __future__ import annotations

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from fastforks.pose.models.retinaface import model as rf
from fastforks.pose.models.retinaface.net import HEAD, _up


def _prior_box(h: int, w: int) -> torch.Tensor:
    """facexlib `PriorBox.forward` (normalised cx, cy, w, h), loop for loop."""
    anchors = []
    for step, sizes in zip(rf.STEPS, rf.MIN_SIZES, strict=True):
        for i, j in itertools.product(range(math.ceil(h / step)), range(math.ceil(w / step))):
            for s in sizes:
                anchors += [(j + 0.5) * step / w, (i + 0.5) * step / h, s / w, s / h]
    return torch.tensor(anchors).view(-1, 4)


@pytest.mark.parametrize(("h", "w"), [(64, 96), (70, 45)])
def test_anchors_are_upstream_priors_in_pixels(h: int, w: int):
    got = rf.anchors(h, w, torch.device("cpu"))
    want = _prior_box(h, w) * torch.tensor([w, h, w, h])
    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=1e-6, atol=1e-4)


def test_split_is_upstream_heads_permute_view():
    g = torch.Generator().manual_seed(0)
    maps = tuple(torch.randn(1, HEAD, n, n + 1, generator=g) for n in (4, 2, 1))
    logits, box, lms = rf.split(maps)

    def head(c0: int, c1: int, k: int) -> torch.Tensor:  # upstream ClassHead / BboxHead / LandmarkHead.forward
        return torch.cat([m[:, c0:c1].permute(0, 2, 3, 1).reshape(1, -1, k) for m in maps], 1)[0]

    assert torch.equal(logits, head(0, 4, 2))
    assert torch.equal(box, head(4, 12, 4))
    assert torch.equal(lms, head(12, 32, 10))


@pytest.mark.parametrize("size", [(9, 7), (8, 8), (5, 12)])
def test_upsampling_is_interpolate_to_size(size: tuple[int, int]):
    x = torch.randn(1, 2, (size[0] + 1) // 2, (size[1] + 1) // 2)
    like = torch.empty(1, 2, *size)
    assert torch.equal(_up(x, like), F.interpolate(x, size=size, mode="nearest"))


def test_one_confident_anchor_decodes_to_its_box_and_landmarks():
    h, w = 32, 32
    maps = tuple(torch.zeros(1, HEAD, math.ceil(h / s), math.ceil(w / s)) for s in rf.STEPS)
    maps[0][0, :4] = torch.tensor([5.0, -5.0, 5.0, -5.0])[:, None, None]  # every anchor: background
    maps[0][0, 2:4, 1, 2] = torch.tensor([-5.0, 5.0])  # cell (1, 2), anchor 1 (side 32): a face
    maps[0][0, 8:12, 1, 2] = torch.tensor([1.0, -1.0, 0.0, math.log(2) / 0.2])  # shift (3.2, -3.2), height x 2
    maps[0][0, 22:32, 1, 2] = torch.arange(10.0)
    for m in maps[1:]:
        m[0, 0:4] = torch.tensor([5.0, -5.0, 5.0, -5.0])[:, None, None]
    out = rf.faces(maps, h, w, 0.5, 0.4)
    cx, cy = 2.5 * 8 + 3.2, 1.5 * 8 - 3.2
    assert torch.allclose(out["boxes"], torch.tensor([[cx - 16, cy - 32, cx + 16, cy + 32]]), atol=1e-4)
    want = torch.tensor([20.0, 12.0]) + torch.arange(10.0).view(5, 2) * 0.1 * 32
    assert torch.allclose(out["keypoints"][0], want, atol=1e-4)
    assert out["keypoint_scores"].shape == (1, 5)
    assert torch.allclose(out["scores"], torch.sigmoid(torch.tensor([10.0])))


def test_nms_drops_boxes_a_kept_one_overlaps():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [8.0, 0.0, 18.0, 10.0]])
    assert rf.nms(boxes, 0.4).tolist() == [0, 2]
    assert rf.nms(boxes[:0], 0.4).tolist() == []
