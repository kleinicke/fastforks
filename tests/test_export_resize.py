"""`export_resize`: antialiased resizes as matmuls in exported graphs (TensorRT has no antialiased Resize)."""

from __future__ import annotations

import importlib.util
import threading

import pytest
import torch
import torch.nn.functional as F

from fastforks._core import export_resize
from fastforks._core.export_resize import matmul_antialias

CASES = [  # (input H, W), interpolate kwargs
    ((37, 53), {"size": (16, 24)}),  # downscale (the antialiasing case)
    ((16, 24), {"size": (37, 53)}),  # upscale
    ((40, 30), {"size": (40, 12)}),  # one axis unchanged
    ((50, 70), {"scale_factor": 0.37}),  # the scale is passed to the filter
    ((50, 70), {"scale_factor": (0.5, 0.25), "recompute_scale_factor": True}),
    ((33, 47), {"size": (10, 9), "align_corners": True}),
]


@pytest.mark.parametrize("mode", ["bilinear", "bicubic"])
@pytest.mark.parametrize(("hw", "kw"), CASES)
def test_matmul_equals_antialiased_interpolate(mode, hw, kw):
    x = torch.rand(2, 3, *hw, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    want = F.interpolate(x, mode=mode, antialias=True, **kw)
    with matmul_antialias():
        got = F.interpolate(x, mode=mode, antialias=True, **kw)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)


def test_patch_is_scoped_to_the_context_and_thread():
    assert F.interpolate is export_resize._ORIGINAL
    seen = []

    def other_thread() -> None:  # the patched function, but this thread did not enter the context
        x = torch.rand(1, 1, 20, 20)
        orig = export_resize._ORIGINAL(x, (7, 7), mode="bilinear", antialias=True)
        seen.append(torch.equal(F.interpolate(x, (7, 7), mode="bilinear", antialias=True), orig))

    with matmul_antialias():
        assert F.interpolate is not export_resize._ORIGINAL
        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
    assert F.interpolate is export_resize._ORIGINAL
    assert seen == [True]  # the other thread ran the original op (bit-equal)


class _Resize(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.interpolate(x, (24, 20), mode="bicubic", align_corners=False, antialias=True)
        return a + F.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True).mean()


def test_export_has_no_antialiased_op():
    x = torch.rand(1, 3, 61, 47)
    want = _Resize()(x)
    with matmul_antialias():
        ep = torch.export.export(_Resize().eval(), (x,), strict=False)
    ops = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
    assert not any("_aa" in o for o in ops), ops
    torch.testing.assert_close(ep.module()(x), want, rtol=1e-5, atol=1e-6)


def test_dynamic_sizes_keep_the_original_op():
    x = torch.rand(1, 3, 64, 48)
    dim = torch.export.Dim("h", min=32, max=128)
    with matmul_antialias():
        ep = torch.export.export(_Resize().eval(), (x,), dynamic_shapes=({2: dim},), strict=False)
    assert any("_aa" in str(n.target) for n in ep.graph.nodes if n.op == "call_function")


@pytest.mark.skipif(
    not (importlib.util.find_spec("onnxscript") and importlib.util.find_spec("onnxruntime")),
    reason="needs onnxscript + onnxruntime (dependency group export)",
)
def test_onnx_has_no_antialiased_resize(tmp_path):
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    x = torch.rand(1, 3, 61, 47)
    with matmul_antialias():
        torch.onnx.export(_Resize().eval(), (x,), str(tmp_path / "m.onnx"), dynamo=True, input_names=["image"])
    graph = onnx.load(str(tmp_path / "m.onnx")).graph
    assert not [n for n in graph.node if n.op_type == "Resize"]
    got = ort.InferenceSession(str(tmp_path / "m.onnx")).run(None, {"image": x.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(got), _Resize()(x), rtol=1e-5, atol=1e-5)
