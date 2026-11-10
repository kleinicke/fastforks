"""Weight-free checks of the pose pre- and post-processing: crop indexing, box validation, flip tables and the flip
test, skeletons, person buckets."""

from __future__ import annotations

from importlib import resources
from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import InputError
from fastforks.pose import _api
from fastforks.pose._runner import BUCKETS, in_buckets
from fastforks.pose.models.rtmpose import model as rtmpose
from fastforks.pose.models.rtmpose.net import RTMPose


def test_crop_reads_the_right_pixels_above_2_pow_24_pixels():
    # 4100 x 4100 > 2^24 pixels: a float32 flat index would land on neighbouring pixels in the last rows
    h = w = 4100
    g = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (1, 3, h, w), generator=g, dtype=torch.uint8).float() / 255
    boxes = torch.tensor([[w - 200.0, h - 264.0, w - 8.0, h - 8.0]])  # integer box, scale 1: taps are whole pixels
    centre, size = rtmpose.crop_boxes(boxes, 256, 192)
    got = rtmpose.crop(image, centre, size, 256, 192)
    x0 = int(centre[0, 0] - size[0, 0] / 2)
    y0 = int(centre[0, 1] - size[0, 1] / 2)
    step = float(size[0, 0]) / 192
    assert step == 1.25  # the 1.25 padding of a 192 x 256 box
    ys = torch.arange(0, 256, 4)  # every 4th crop pixel sits on a whole source pixel (4 * 1.25 = 5)
    xs = torch.arange(0, 192, 4)
    sy, sx = y0 + ys * 5 // 4, x0 + xs * 5 // 4
    inside = ((sy >= 0) & (sy < h))[:, None] & ((sx >= 0) & (sx < w))[None, :]
    want = (image[0, :, sy.clamp(0, h - 1)][:, :, sx.clamp(0, w - 1)] * 255).round() * inside
    assert torch.equal(got[0][:, ys][:, :, xs], want)


def test_boxes_whose_size_overflows_float32_are_rejected():
    with pytest.raises(InputError, match="non-finite"):
        _api._boxes_tensor([[-3e38, -3e38, 3e38, 3e38]], torch.device("cpu"))


@pytest.mark.parametrize(("count", "labels"), [(17, "coco-keypoints-17.txt"), (133, "coco-wholebody-133.txt")])
def test_flip_table_swaps_left_and_right(count, labels):
    names = (resources.files("fastforks._core") / "labels" / labels).read_text().split()
    flip = rtmpose.FLIP[count]
    assert len(names) == len(flip) == count
    assert [flip[i] for i in flip] == list(range(count))  # an involution
    for i, name in enumerate(names):
        if name.startswith("left_"):
            assert names[flip[i]] == "right_" + name[5:], name
        elif name.startswith("right_"):
            assert names[flip[i]] == "left_" + name[6:], name


def _logits(net: RTMPose, params: rtmpose.Params, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`Portable.logits` on a network with random weights (no checkpoint, no runner context)."""
    runner = SimpleNamespace(
        net=net,
        mean=torch.tensor(rtmpose.MEAN).view(1, 3, 1, 1),
        std=torch.tensor(rtmpose.STD).view(1, 3, 1, 1),
        flip=torch.tensor(rtmpose.FLIP[params.keypoints]),
        ctx=SimpleNamespace(params=params, torch_dtype=torch.float32),
    )
    with torch.inference_mode():
        return rtmpose.Portable.logits(runner, crops)  # `runner`: a stand-in


@pytest.mark.parametrize("keypoints", [17, 133])
def test_flip_test_of_the_mirrored_crop_gives_the_mirrored_logits(keypoints):
    torch.manual_seed(0)
    arch = {"deepen": 0.167, "widen": 0.375, "keypoints": keypoints, "height": 256, "width": 192}
    on, off = (rtmpose.Params(**arch, flip_test=flag) for flag in (True, False))
    net = RTMPose(on).eval()
    crops = (torch.rand(2, 3, 256, 192) * 255).round()
    flip = torch.tensor(rtmpose.FLIP[keypoints])

    def mirrored(sx: torch.Tensor, sy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sx.index_select(1, flip).flip(-1), sy.index_select(1, flip)  # left <-> right, the x bins reversed

    sx, sy = _logits(net, on, crops)
    assert sx.shape == (2, keypoints, 2 * 192)
    assert sy.shape == (2, keypoints, 2 * 256)
    for got, want in zip(_logits(net, on, crops.flip(-1)), mirrored(sx, sy), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=1e-4 * float(want.abs().max()))
    # ... which a random network alone does not do: without the flip test the mirrored crop gives other logits
    plain = _logits(net, off, crops)
    gap = [float((g - w).abs().max()) for g, w in zip(_logits(net, off, crops.flip(-1)), mirrored(*plain), strict=True)]
    assert min(gap) > 1e-2 * float(plain[0].abs().max()), gap


@pytest.mark.parametrize(("count", "labels"), [(17, "coco-keypoints-17"), (133, "coco-wholebody-133")])
def test_skeleton_joins_named_neighbours(count, labels):
    names = (resources.files("fastforks._core") / "labels" / f"{labels}.txt").read_text().split()
    limbs = _api.SKELETONS[labels]
    assert len(set(limbs)) == len(limbs) == {17: 19, 133: 65}[count]  # mmpose `skeleton_info` of the dataset
    assert all(0 <= a < count and 0 <= b < count and a != b for a, b in limbs)
    assert _api.SKELETONS["coco-wholebody-133"][:19] == _api.SKELETONS["coco-keypoints-17"]
    joined = {i for limb in limbs for i in limb}
    assert not any(names[i].startswith("face-") for i in joined)
    assert joined == {i for i, n in enumerate(names) if not n.startswith("face-")}  # every body / foot / hand point
    for a, b in limbs:
        if "hand_root" in names[a] or "finger" in names[a] or "thumb" in names[a]:
            assert names[a].split("_")[0] == names[b].split("_")[0], (names[a], names[b])  # one hand per limb


@pytest.mark.parametrize("people", [1, 3, 32, 33, 70])
def test_in_buckets_runs_bucket_sized_batches_in_order(people):
    seen = []

    def forward(x):
        seen.append(x.shape[0])
        return x * 2

    crops = torch.arange(1, people + 1, dtype=torch.float32)[:, None].expand(people, 3).contiguous()
    assert torch.equal(in_buckets(forward, crops), crops * 2)
    assert set(seen) <= set(BUCKETS)
    assert sum(seen) - people < BUCKETS[-1]
