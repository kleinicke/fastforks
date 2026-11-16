"""Weight-free checks of YuNet's merged heads, cell points and decoding against upstream's rules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fastforks._core import registry
from fastforks.pose.models.yunet import model as yn
from fastforks.pose.models.yunet.net import HEAD, LEVELS, PARTS, Arch, YuNet, convert

NAMES = ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias")
WIDTHS = {"cls": 1, "obj": 1, "bbox": 4, "kps": 10}  # upstream head channels per part


def test_card_variants():
    cards = {v.id: {**c.params, **v.params} for c, v in registry.cards("pose").values() if c.family == "yunet"}
    assert sorted(cards) == ["yunet-n", "yunet-s"]
    assert (cards["yunet-n"]["stage1"], cards["yunet-s"]["stage1"]) == (64, 32)


def _upstream_sd(net: YuNet) -> dict[str, torch.Tensor]:
    """The network's state dict in the checkpoint layout: per level one head unit per part, `module.` prefixed."""
    sd = {k: v for k, v in net.state_dict().items() if not k.startswith("bbox_head.heads.")}
    for level in range(LEVELS):
        for name in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"):
            merged = net.state_dict()[f"bbox_head.heads.{level}.{name}"]
            for p, part in zip(PARTS, merged.split(list(WIDTHS.values())), strict=True):
                sd[f"bbox_head.multi_level_{p}.{level}.{name}"] = part
    return {f"module.{k}": v for k, v in sd.items()}


def test_merged_heads_are_the_four_upstream_units():
    torch.manual_seed(0)
    net = YuNet(Arch(stage1=16, shared_convs=1)).eval()
    for p in net.parameters():
        p.data.normal_(0, 0.3)
    sd = _upstream_sd(net)
    merged = convert(sd)
    assert merged.keys() == net.state_dict().keys()
    assert all(torch.equal(merged[k], v) for k, v in net.state_dict().items())
    head = net.bbox_head.heads[0]
    x = torch.randn(1, head.conv1.in_channels, 5, 6)
    got = head(x)
    want = []
    for p in PARTS:  # upstream Unit: 1 x 1 conv, depthwise 3 x 3 conv (no BN / ReLU in the heads)
        w1, b1, w2, b2 = (sd[f"module.bbox_head.multi_level_{p}.0.{n}"] for n in NAMES)
        want.append(F.conv2d(F.conv2d(x, w1, b1), w2, b2, padding=1, groups=w2.shape[0]))
    torch.testing.assert_close(got, torch.cat(want, 1))
    assert got.shape[1] == HEAD


def test_points_are_upstream_cell_corners():
    maps = tuple(torch.zeros(1, HEAD, 64 // s, 96 // s) for s in yn.STRIDES)
    pts = yn.points(maps)
    assert pts.shape == (sum(m.shape[2] * m.shape[3] for m in maps), 3)
    assert pts[:3].tolist() == [[0, 0, 8], [8, 0, 8], [16, 0, 8]]  # MlvlPointGenerator(offset=0): row by row
    assert pts[12].tolist() == [0, 8, 8]
    assert pts[-1].tolist() == [64, 32, 32]


def test_one_confident_cell_decodes_to_its_box_and_landmarks():
    maps = tuple(torch.full((1, HEAD, 64 // s, 64 // s), -10.0) for s in yn.STRIDES)
    cell = maps[1][0, :, 2, 1]  # stride 16, point (16, 32)
    cell[:2] = 5.0
    cell[2:6] = torch.tensor([0.5, -0.25, math.log(2), 0.0])
    cell[6:] = torch.arange(10.0) / 10
    out = yn.faces(maps, 0.5, 0.3)
    cx, cy, w, h = 16 + 8, 32 - 4, 32, 16
    torch.testing.assert_close(out["boxes"], torch.tensor([[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]]))
    want = torch.tensor([16.0, 32.0]) + (torch.arange(10.0) / 10).view(5, 2) * 16
    torch.testing.assert_close(out["keypoints"][0], want)
    torch.testing.assert_close(out["scores"], torch.sigmoid(torch.tensor([5.0])) ** 2)
    assert out["keypoint_scores"].shape == (1, 5)
