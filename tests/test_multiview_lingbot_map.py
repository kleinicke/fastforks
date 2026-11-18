"""LingBot-Map's fixed-shape export tables."""

from types import SimpleNamespace

import pytest
import torch

from fastforks._core.errors import OptionsError
from fastforks.multiview.models.lingbot_map.model import Portable, _export_rope3d, _Fast, _rope3d_tables


def test_host_stream_requires_cuda_before_loading() -> None:
    ctx = SimpleNamespace(device=torch.device("cpu"), params=SimpleNamespace(stream=True))
    with pytest.raises(OptionsError, match="stream=True requires CUDA"):
        Portable(ctx)  # ty: ignore[invalid-argument-type] - reject before reading the checkpoint


class _RopeModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.runner = SimpleNamespace(ctx=SimpleNamespace(device=torch.device("cpu")), dt=torch.float32)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = _Fast.rope3d(self.runner, range(1, 3), 2, 3)
        return cos + x, sin + x


@pytest.mark.parametrize("start", [0, 40, 1022])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_precomputed_tables_match_float64_math(start: int, dtype: torch.dtype) -> None:
    expected = _rope3d_tables(range(start, start + 2), 7, 9, torch.device("cpu"), dtype)
    actual = _export_rope3d(start, start + 2, 7, 9, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_exported_rope_matches_float64_tables() -> None:
    module = _RopeModule().eval()
    x = torch.tensor(0.0)
    eager = module(x)
    expected = _rope3d_tables(range(1, 3), 2, 3, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
    exported = torch.export.export(module, (x,), strict=False).module()(x)
    torch.testing.assert_close(exported, eager, rtol=0, atol=0)


@pytest.mark.slow
def test_onnx_rope_uses_precomputed_float64_tables(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    module = _RopeModule().eval()
    x = torch.tensor(0.0)
    path = tmp_path / "rope.onnx"
    torch.onnx.export(module, (x,), path, dynamo=True, input_names=["x"])
    assert all(node.op_type not in {"Sin", "Cos"} for node in onnx.load(path).graph.node)
    actual = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})
    torch.testing.assert_close(tuple(torch.from_numpy(y) for y in actual), module(x), rtol=0, atol=0)
