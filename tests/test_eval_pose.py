"""COCO pose evaluation preserves false positives and distinguishes predicted from GT boxes."""

from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import _original_coco
import eval_coco


def test_bottomup_evaluates_predictions_without_ground_truth(tmp_path, monkeypatch):
    path = tmp_path / "1.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(eval_coco, "_sync", lambda: 0.0)
    coco = SimpleNamespace(getAnnIds=lambda **_kw: [], loadAnns=lambda _ids: [])

    class Model:
        needs_boxes = False

        def __call__(self, _img):
            return SimpleNamespace(
                keypoints=torch.zeros(2, 17, 2),
                keypoint_scores=torch.ones(2, 17),
                scores=torch.tensor([0.8, 0.2]),
            )

    rec = {}
    rows, ms = eval_coco._pose(Model(), coco, [path], rec)
    assert len(rows) == 2
    assert rec["pose_gt_boxes"] is False
    assert np.isclose(rows[0]["score"], 0.8)
    assert ms == [0.0]


def test_topdown_preserves_ground_truth_crops_and_keypoint_scores(tmp_path, monkeypatch):
    path = tmp_path / "1.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(eval_coco, "_sync", lambda: 0.0)
    coco = SimpleNamespace(
        getAnnIds=lambda **_kw: [1],
        loadAnns=lambda _ids: [{"num_keypoints": 17, "bbox": [1, 2, 3, 4]}],
    )

    class Model:
        needs_boxes = True

        def __call__(self, _img, *, boxes):
            np.testing.assert_array_equal(boxes, [[1, 2, 4, 6]])
            return SimpleNamespace(keypoints=torch.zeros(1, 17, 2), keypoint_scores=torch.full((1, 17), 0.5))

    rec = {}
    rows, _ = eval_coco._pose(Model(), coco, [path], rec)
    assert len(rows) == 1
    assert rows[0]["score"] == 0.5
    assert rec["pose_gt_boxes"]


def test_bottomup_empty_reference_result(tmp_path, monkeypatch):
    path = tmp_path / "1.png"
    Image.new("RGB", (16, 16)).save(path)
    monkeypatch.setattr(eval_coco, "_sync", lambda: 0.0)
    coco = SimpleNamespace(getAnnIds=lambda **_kw: [], loadAnns=lambda _ids: [])
    model = object.__new__(_original_coco._Pose)
    model.family = "rtmo"  # bottom-up
    model.original = lambda _img: []
    rec = {}
    rows, ms = eval_coco._pose(model, coco, [path], rec)
    assert rows == []
    assert ms == [0.0]
    assert rec["people"] == 0


def test_bottomup_reference_uses_person_scores():
    model = object.__new__(_original_coco._Pose)
    model.family = "rtmo"  # bottom-up
    pred = SimpleNamespace(
        keypoints=np.zeros((2, 17, 2), dtype=np.float32),
        keypoint_scores=np.ones((2, 17), dtype=np.float32),
        bbox_scores=np.array([0.9, 0.1], dtype=np.float32),
    )
    model.original = lambda _img: [SimpleNamespace(pred_instances=pred)]
    result = model(np.zeros((16, 16, 3), dtype=np.uint8))
    np.testing.assert_allclose(result.scores, [0.9, 0.1])
