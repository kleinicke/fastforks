"""LW-DETR: the card, the COCO label mapping, the PostProcess (threshold, cap, empty output) and the square resize."""

from types import SimpleNamespace

import torch

from fastforks import detection
from fastforks._core import registry
from fastforks.detection.models.lwdetr.model import Portable, class_map, square


def _runner(max_detections: int = 100) -> Portable:
    runner = object.__new__(Portable)
    runner.ctx = SimpleNamespace(params=SimpleNamespace(size=4, num_select=3, max_detections=max_detections))
    runner.ids = class_map(91)
    runner.dt = torch.float32
    runner.mean, runner.std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    logits = torch.full((1, 3, 91), -9.0)
    logits[0, 0, 1] = 3.0  # COCO id 1 (person) -> 0, score 0.95
    logits[0, 1, 12] = 4.0  # unused COCO id 12 -> dropped although it scores best
    logits[0, 2, 13] = 1.0  # COCO id 13 (stop sign) -> 11, score 0.73
    boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]] * 3])
    runner._forward = lambda _x: (logits, boxes)
    return runner


def test_card() -> None:
    names = detection.list_models()
    for size in ("tiny", "small", "medium", "large", "xlarge"):
        assert f"lw-detr-{size}-coco" in names
    _, variant = registry.resolve("detection", "lw-detr-tiny-coco")
    assert variant.params["num_select"] == 100


def test_class_map_drops_background_and_unused_ids() -> None:
    ids = class_map(91)
    assert ids[0] == -1
    assert ids[1] == 0
    assert ids[12] == -1
    assert ids[90] == 79  # toothbrush
    assert int((ids >= 0).sum()) == 80  # coco-80


def test_labels_threshold_cap_and_input_pixels() -> None:
    image = torch.zeros(1, 3, 100, 200)
    got = _runner()(image, 0.5)
    assert got["labels"].tolist() == [0, 11]
    torch.testing.assert_close(got["scores"], torch.tensor([3.0, 1.0]).sigmoid())
    torch.testing.assert_close(got["boxes"], torch.tensor([[0.0, 0.0, 200.0, 100.0]] * 2))
    assert _runner()(image, 0.8)["labels"].tolist() == [0]
    assert _runner(max_detections=1)(image, 0.5)["labels"].tolist() == [0]
    empty = _runner()(image, 1.0)
    assert empty["boxes"].shape == (0, 4)
    assert empty["labels"].shape == (0,)


def test_square_resize_values() -> None:
    """uint8 values on the CPU (torch's integer kernel); the same size passes through rounded."""
    x = torch.rand(1, 3, 30, 50)
    out = square(x, 16)
    assert out.shape == (1, 3, 16, 16)
    assert torch.equal(out, out.round())
    same = square(x[..., :30], 30)
    assert torch.equal(same, (x[..., :30] * 255).round())
