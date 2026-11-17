"""RF-DETR COCO label mapping, threshold, detection cap and empty-output contracts without model weights."""

from types import SimpleNamespace

import pytest
import torch

from fastforks.detection.models.rfdetr.model import Portable, class_map
from fastforks.shared.rfdetr.net.models.postprocess import PostProcess


def _runner(monkeypatch, max_detections=100, *, masks=False):
    runner = object.__new__(Portable)
    runner.ctx = SimpleNamespace(params=SimpleNamespace(max_detections=max_detections))
    runner.postprocess = PostProcess(num_select=3)
    runner.ids = class_map(91)
    logits = torch.full((1, 3, 91), -9.0)
    logits[0, 0, 1] = 3.0  # COCO id 1 (person) -> 0, score 0.95
    logits[0, 1, 12] = 4.0  # unused COCO id 12 -> dropped although it scores best
    logits[0, 2, 13] = 1.0  # COCO id 13 (stop sign) -> 11, score 0.73
    raw = {"pred_logits": logits, "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]] * 3])}
    if masks:
        raw["pred_masks"] = torch.ones(1, 3, 4, 4)
    monkeypatch.setattr(runner, "_raw", lambda _image: raw)
    return runner


def test_class_map_drops_background_and_unused_ids():
    ids = class_map(91)
    assert ids[0] == -1
    assert ids[1] == 0
    assert ids[90] == 79  # toothbrush
    assert int((ids >= 0).sum()) == 80  # coco-80


def test_labels_threshold_and_input_pixels(monkeypatch):
    got = _runner(monkeypatch)(torch.zeros(1, 3, 100, 200), 0.5)
    assert got["labels"].tolist() == [0, 11]
    torch.testing.assert_close(got["scores"], torch.tensor([3.0, 1.0]).sigmoid())
    torch.testing.assert_close(got["boxes"], torch.tensor([[0.0, 0.0, 200.0, 100.0]] * 2))
    assert _runner(monkeypatch)(torch.zeros(1, 3, 100, 200), 0.8)["labels"].tolist() == [0]


def test_max_detections_and_seg_masks(monkeypatch):
    got = _runner(monkeypatch, max_detections=1, masks=True)(torch.zeros(1, 3, 100, 200), 0.5)
    assert got["labels"].tolist() == [0]
    assert got["masks"].shape == (1, 100, 200)
    assert got["masks"].dtype == torch.bool


@pytest.mark.parametrize("masks", [False, True])
def test_no_detection_keeps_shapes(monkeypatch, masks):
    got = _runner(monkeypatch, masks=masks)(torch.zeros(1, 3, 100, 200), 1.0)
    assert got["boxes"].shape == (0, 4)
    assert got["scores"].shape == (0,)
    assert got["labels"].shape == (0,)
    if masks:
        assert got["masks"].shape == (0, 100, 200)
