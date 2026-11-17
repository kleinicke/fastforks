"""OWLv2: the processor's input path, the post-processing and the CLIP tokenizer against transformers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks.detection._api import DetectionModel
from fastforks.detection.models.owlv2.model import MEAN, STD, Portable
from fastforks.shared.clip_tokenizer import ClipTokenizer


def _runner(size: int = 64, max_detections: int = 100) -> Portable:
    runner = object.__new__(Portable)
    runner.ctx = SimpleNamespace(params=SimpleNamespace(max_detections=max_detections))
    runner.size = size
    runner.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    runner.std = torch.tensor(STD).view(1, 3, 1, 1)
    return runner


@pytest.mark.slow  # imports transformers
@pytest.mark.parametrize("hw", [(150, 130), (40, 30), (64, 64)])  # blur + downscale, upscale, same size
def test_pixels_match_the_transformers_processor(hw: tuple[int, int]) -> None:
    transformers = pytest.importorskip("transformers")
    img = np.random.default_rng(0).integers(0, 256, (*hw, 3), dtype=np.uint8)
    want = transformers.Owlv2ImageProcessor(size={"height": 64, "width": 64})(images=img, return_tensors="pt")
    got = _runner().pixels(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255)
    torch.testing.assert_close(got, want["pixel_values"], atol=1e-5, rtol=0)


def test_postprocess_threshold_order_cap_and_square_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(max_detections=2)
    runner.default = ("a", "b")
    monkeypatch.setattr(runner, "queries", lambda names: torch.eye(len(names), 4))
    emb = torch.tensor([[[0.0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])  # patch -> query
    scale = torch.tensor([[[1.0], [3.0], [2.0], [-1.0]]])  # logits 1, 3, 2, -1 (score of the last: below 0.5)
    boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.5, 0.5], [0.5, 0.25, 1.0, 0.5], [0.5, 0.5, 0.1, 0.1]]])
    monkeypatch.setattr(runner, "pixels", lambda image: image)
    monkeypatch.setattr(runner, "_image", lambda _x: (emb, torch.zeros(1, 4, 1), scale, boxes))
    got = runner(torch.zeros(1, 3, 100, 200), 0.5)
    torch.testing.assert_close(got["scores"], torch.tensor([3.0, 2.0]).sigmoid())  # sorted, cut at max_detections
    assert got["labels"].tolist() == [0, 1]
    torch.testing.assert_close(got["boxes"], torch.tensor([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 200.0, 100.0]]))
    assert runner(torch.zeros(1, 3, 100, 200), float(torch.tensor(3.0).sigmoid()))["scores"].numel() == 0  # strict


@pytest.mark.slow
@pytest.mark.model
def test_clip_tokenizer_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    card, variant = registry.resolve("detection", "owlv2-b16")
    _, _, vocab, merges = registry.weights(card, variant, allow_download=False)
    ours = ClipTokenizer(vocab, merges, context=16)
    theirs = transformers.CLIPTokenizer(str(vocab), str(merges))
    texts = ["dining table", "a photo of a  Hot-Dog!", "teddy bear's 2nd toy", "ein Ölgemälde", " ".join(["cat"] * 30)]
    for text in texts:
        want = theirs(text, max_length=16, truncation=True)["input_ids"]
        assert ours(text) == want, text


def test_text_input_checks() -> None:
    m = object.__new__(DetectionModel)
    m.variant = SimpleNamespace(ref="x")
    m._runner = SimpleNamespace(open_vocabulary=True)
    assert m._queries(None) is None
    assert m._queries(["cat", "remote control"]) == ("cat", "remote control")
    assert m._queries("cat") == ("cat",)
    for bad in ([], [""], ["cat", "  "], ["cat", 3]):
        with pytest.raises(InputError, match="non-empty"):
            m._queries(bad)
    m._runner = SimpleNamespace(open_vocabulary=False)
    with pytest.raises(InputError, match="fixed label set"):
        m._queries(["cat"])
