"""M-LSD: the card, the segment decoding and the `cv2.line` rasterizer vs OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from fastforks._core import registry
from fastforks.edges.models.mlsd.model import _clip, decode, draw


def test_card() -> None:
    cards = {v.id: c for c, v in registry.cards("edges").values() if c.family == "mlsd"}
    assert sorted(cards) == ["mlsd"]
    assert cards["mlsd"].licence.code == "Apache-2.0"
    assert cards["mlsd"].options["thr_v"] == 0.1


def test_draw_matches_cv2_line() -> None:
    """Exact `cv2.line(thickness=1)` incl. segments crossing or outside the border, batched with a keep mask."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        h, w = (int(v) for v in rng.integers(5, 50, 2))
        seg = rng.integers(-40, 90, (2, 6, 4))
        keep = rng.random((2, 6)) < 0.8
        got = draw(torch.from_numpy(seg), torch.from_numpy(keep), h, w).numpy()
        for b in range(2):
            want = np.zeros((h, w), np.uint8)
            for (x0, y0, x1, y1), k in zip(seg[b].tolist(), keep[b], strict=True):
                if k:
                    cv2.line(want, (x0, y0), (x1, y1), 255, 1)
            np.testing.assert_array_equal(got[b], want / 255)


def test_decode() -> None:
    """One center: `2 * (center + displacement)`, kept only above both thresholds."""
    maps = torch.full((1, 9, 8, 8), -10.0)
    maps[0, 0, 3, 5] = 2.0  # sigmoid 0.88
    maps[0, 1:5, 3, 5] = torch.tensor([-1.0, 0.5, 2.0, -0.5])
    seg, keep = decode(maps, 0.1, 0.1)
    assert keep.sum() == 1
    assert seg[keep].tolist() == [[8.0, 7.0, 14.0, 5.0]]
    _, keep = decode(maps, 0.9, 0.1)
    assert not keep.any()
    _, keep = decode(maps, 0.1, 4.0)
    assert not keep.any()


@pytest.mark.slow
def test_clip_onnx_matches_eager_without_integer_bitwise(tmp_path) -> None:
    """Line clipping exports correctly without operators rejected by TensorRT."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    class Clip(torch.nn.Module):
        def forward(self, segments: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return _clip(segments, 17, 23)

    segments = torch.tensor([[-5, 2, 30, 15], [-4, -3, -1, 8], [10, -5, 10, 30], [1, 1, 20, 15]])
    model = Clip().eval()
    path = tmp_path / "clip.onnx"
    torch.onnx.export(model, (segments,), path, dynamo=True)
    graph = onnx.load(str(path)).graph
    assert not any(node.op_type.startswith("Bitwise") for node in graph.node)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: segments.numpy()})
    for got, want in zip(actual, model(segments), strict=True):
        np.testing.assert_array_equal(got, want.numpy())
