"""`save` of the new result types: restoration / interpolation images, flow, instances / pose JSON."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image

import fastforks
from fastforks import InputError
from fastforks._core.instances import Instances
from fastforks.flow import FlowResult
from fastforks.interpolation import InterpolationResult
from fastforks.pose import PoseResult
from fastforks.restoration import RestorationResult

if TYPE_CHECKING:
    from pathlib import Path


def test_image_results(tmp_path: Path) -> None:
    """PNG 8-bit (rounded), `.npy` float32 as is, a batch needs `{i}`."""
    img = torch.rand(3, 5, 7)
    for r in (RestorationResult(img, "m", "portable"), InterpolationResult(img, "m", "portable")):
        r.save(tmp_path / "a.png")
        png = np.asarray(Image.open(tmp_path / "a.png"))
        assert png.shape == (5, 7, 3)
        assert np.abs(png / 255.0 - img.permute(1, 2, 0).numpy()).max() <= 0.5 / 255 + 1e-6
        fastforks.save(tmp_path / "a.npy", r)
        assert np.array_equal(np.load(tmp_path / "a.npy"), img.numpy())
    batch = RestorationResult(torch.rand(2, 3, 4, 4), "m", "portable")
    with pytest.raises(InputError, match=r"\{i\}"):
        batch.save(tmp_path / "b.png")
    batch.save(tmp_path / "b_{i}.jpg")
    assert sorted(p.name for p in tmp_path.glob("b_*")) == ["b_0.jpg", "b_1.jpg"]
    with pytest.raises(InputError, match=r"cannot write '\.tif'"):
        batch.save(tmp_path / "b.tif")


def test_flow_result(tmp_path: Path) -> None:
    """`.flo` is Middlebury's layout (tag, width, height, `[H, W, 2]`); `.npy` keeps `[2, H, W]`."""
    flow = torch.randn(2, 3, 4)
    FlowResult(flow, "m", "portable").save(tmp_path / "f.flo")
    raw = (tmp_path / "f.flo").read_bytes()
    assert np.frombuffer(raw[:4], "<f4")[0] == 202021.25
    assert np.frombuffer(raw[4:12], "<i4").tolist() == [4, 3]
    assert np.array_equal(np.frombuffer(raw[12:], "<f4").reshape(3, 4, 2), flow.permute(1, 2, 0).numpy())
    FlowResult(flow, "m", "portable").save(tmp_path / "f.npy")
    assert np.array_equal(np.load(tmp_path / "f.npy"), flow.numpy())


def test_instances_and_pose_json(tmp_path: Path) -> None:
    """`save` writes `to_list()` without masks; `PoseResult` is `Instances` and indexing keeps its keypoints."""
    det = Instances(
        torch.tensor([[0.0, 0, 2, 2], [1, 1, 3, 3]]),
        torch.tensor([0.9, 0.4]),
        torch.tensor([1, 0]),
        torch.ones(2, 4, 4, dtype=torch.bool),
        ("cat", "dog"),
        "m",
        "portable",
    )
    det.save(tmp_path / "d.json")
    rows = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert rows == [{k: v for k, v in d.items() if k != "mask"} for d in det.to_list()]
    assert rows[0]["class_name"] == "dog"
    with pytest.raises(InputError, match=r"writes \.json"):
        det.save(tmp_path / "d.png")
    pose = PoseResult(
        boxes=det.boxes,
        scores=det.scores,
        keypoints=torch.rand(2, 3, 2),
        keypoint_scores=torch.rand(2, 3),
        keypoint_names=("a", "b", "c"),
        skeleton=((0, 1),),
        model="m",
        backend="portable",
    )
    assert isinstance(pose, Instances)
    top = pose[pose.scores > 0.5]
    assert len(top) == 1
    assert torch.equal(top.keypoints, pose.keypoints[:1])
    assert torch.equal(top.keypoint_scores, pose.keypoint_scores[:1])
    fastforks.save(tmp_path / "p.json", pose)
    got = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert got[1]["keypoints"]["c"] == list(pose.to_list()[1]["keypoints"]["c"])
