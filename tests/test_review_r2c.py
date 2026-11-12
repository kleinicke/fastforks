"""Review R2 pass 3 fixes (CPU, no weights): dynamic export allowlist and value check, save formats, torch dtypes,
`python -m fastforks`, `load` annotations, the docs' time rounding."""

from __future__ import annotations

import importlib
import runpy
import sys
import types
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

import fastforks
from fastforks import InputError, OptionsError
from fastforks._core import export, registry
from fastforks._core import io as ffio
from fastforks._core.model import _check_args
from fastforks._core.sizes import size_range
from fastforks.depth import DepthResult
from fastforks.promptable_segmentation import PromptableSegmentationResult
from fastforks.segmentation import SegmentationResult
from fastforks.shared.depth_anything_v2.net.dinov2 import _bicubic_matrix
from gen_docs import _ms_str

if TYPE_CHECKING:
    from pathlib import Path


# R2-28 / R2-29: dynamic export


def test_dynamic_export_only_for_verified_families() -> None:
    """A size range is refused up front outside `DYNAMIC_VERIFIED`, with the reason."""
    card, v = registry.resolve_any("raft-stereo-middlebury")
    rng = size_range(v, card)
    assert card.family in export.DYNAMIC_VERIFIED
    assert not export.fixed_size_reason(card, rng)
    reason = export.fixed_size_reason(types.SimpleNamespace(family="toy", task="stereo"), rng)
    assert "not verified for the toy family" in reason
    assert not export.DYNAMIC_VERIFIED & set(export.DYNAMIC_UNSUPPORTED)
    card, v = registry.resolve_any("dinov2-b-linear-ade20k")
    assert "sliding-window" in export.fixed_size_reason(card, size_range(v, card))


def test_mismatch() -> None:
    ref = torch.linspace(-4, 4, 1000).reshape(1, 10, 100)
    a = ref.numpy()
    assert not export._mismatch((ref,), [a + 1e-3])  # float noise
    assert "shape" in export._mismatch((ref,), [a[:, :5]])
    assert "not finite" in export._mismatch((ref,), [np.where(a > 3, np.nan, a)])
    assert "mean |difference|" in export._mismatch((ref,), [a + 0.2])  # 5 % of the max everywhere
    rows = a.copy()
    rows[:, :2] = 0  # 20 % of the rows wrong, as a graph that keeps the traced height
    assert "% of the values off" in export._mismatch((ref,), [rows])
    assert not export._mismatch((ref, ref.new_full((3,), float("nan"))), [a, np.zeros(3, np.float32)])


def _identity_onnx(path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    h = onnx.helper
    image = h.make_tensor_value_info("image", onnx.TensorProto.FLOAT, ["b", 3, "h", "w"])
    out = h.make_tensor_value_info("out", onnx.TensorProto.FLOAT, None)
    graph = h.make_graph([h.make_node("Identity", ["image"], ["out"])], "g", [image], [out])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10), str(path))
    return path


class _Scaled(torch.nn.Module):
    def __init__(self, at: int) -> None:
        super().__init__()
        self.at = at

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2 if x.shape[-1] == self.at else x  # eager differs from the graph at one width


def _model(at: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        variant=types.SimpleNamespace(ref="stub"),
        device=torch.device("cpu"),
        example_inputs=lambda h, w, batch=1: (torch.rand(batch, 3, h, w),),
        module=lambda: _Scaled(at),
    )


def test_dynamic_onnx_wrong_values_are_removed(tmp_path: Path) -> None:
    """The graph must match eager `module()` at the corners; wrong values remove the file."""
    good = _identity_onnx(tmp_path / "good.onnx")
    export._check_dynamic_onnx(_model(-1), good, ((32, 32), (64, 96)), 1, run=True)
    assert good.exists()
    bad = _identity_onnx(tmp_path / "bad.onnx")
    with pytest.raises(InputError, match="wrong values at 64x96"):
        export._check_dynamic_onnx(_model(96), bad, ((32, 32), (64, 96)), 1, run=True)
    assert not bad.exists()


@pytest.mark.parametrize("family", ["psmnet", "gwcnet"])
def test_regress_export_branch_matches_the_row_blocks(monkeypatch: pytest.MonkeyPatch, family: str) -> None:
    """The one-block regression used while exporting gives the row-blocked values."""
    regress = importlib.import_module(f"fastforks.stereo.models.{family}.regress")
    cost = torch.randn(1, 1, 12, 37, 20)
    want = regress.upsample_softmax_regress(cost, 48, 148, 80, rows=16)
    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    torch.testing.assert_close(regress.upsample_softmax_regress(cost, 48, 148, 80, rows=16), want)


def test_cas_regress_export_branch_matches_the_row_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastforks.stereo.models.cas_gwcnet import regress  # noqa: PLC0415

    cost, values = torch.randn(1, 1, 12, 37, 20), torch.rand(1, 48, 1, 1) * 48
    want = regress.upsample_softmax_regress(cost, values, 148, 80, rows=16)
    monkeypatch.setattr(torch.compiler, "is_exporting", lambda: True)
    torch.testing.assert_close(regress.upsample_softmax_regress(cost, values, 148, 80, rows=16), want)


@pytest.mark.parametrize(("n_out", "n_in", "offset"), [(23, 37, 0.1), (74, 37, 0.1), (37, 37, 0.0), (9, 16, 0.1)])
def test_bicubic_matrix_matches_scale_factor(n_out: int, n_in: int, offset: float) -> None:
    """The ViT position-embedding resize used while exporting matches `F.interpolate(scale_factor=...)`."""
    grid = torch.randn(1, 5, n_in, n_in)
    s = (n_out + offset) / n_in
    want = F.interpolate(grid, scale_factor=(s, s), mode="bicubic")
    m = _bicubic_matrix(n_out, n_in, offset, grid.device)
    torch.testing.assert_close(torch.einsum("ip,cpq,jq->cij", m, grid[0], m)[None], want, atol=1e-4, rtol=1e-4)


# R2-30: save


def test_write_map_png(tmp_path: Path) -> None:
    mask = np.arange(12).reshape(3, 4) % 3 == 0
    ffio.write_map(tmp_path / "m.png", mask)
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "m.png")), mask)
    for labels in (np.arange(12).reshape(3, 4) * 20, np.arange(12).reshape(3, 4) * 5000):
        ffio.write_map(tmp_path / "l.png", torch.from_numpy(labels))
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "l.png")), labels)
    with pytest.raises(InputError, match=r"integer or bool maps.*\.tif"):
        ffio.write_map(tmp_path / "f.png", np.zeros((3, 4), np.float32))
    with pytest.raises(InputError, match=r"0\.\.65535"):
        ffio.write_map(tmp_path / "f.png", np.full((3, 4), -1))
    with pytest.raises(InputError, match=r"\.tif, \.tiff, \.npy, \.pfm, \.png"):
        ffio.write_map(tmp_path / "f.jpg", mask)
    fastforks.save(tmp_path / "s.png", torch.from_numpy(mask))  # one mask through the free function
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "s.png")), mask)


def test_segmentation_and_masks_save_png(tmp_path: Path) -> None:
    labels = torch.arange(20).reshape(4, 5) % 3
    SegmentationResult(labels, None, ("a", "b", "c"), "m@1", None).save(tmp_path / "s.png")  # type: ignore[arg-type]
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "s.png")), labels.numpy())
    masks = torch.rand(1, 1, 4, 5) > 0.5
    p = PromptableSegmentationResult(masks, torch.rand(1, 1), torch.rand(1, 1, 2, 2), "m@1", None)  # type: ignore[arg-type]
    p.save(tmp_path / "p.png")  # [1, 1, H, W] -> [H, W]
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "p.png")), masks[0, 0].numpy())
    masks = torch.rand(1, 3, 4, 5) > 0.5
    three = PromptableSegmentationResult(masks, torch.rand(1, 3), torch.rand(1, 3, 2, 2), "m@1", None)  # type: ignore[arg-type]
    with pytest.raises(InputError, match=r"must be \[H, W\]; 'masks' has shape \[3, 4, 5\].*masks\[\.\.\.\]"):
        three.save(tmp_path / "t.png")
    with pytest.raises(InputError, match=r"SegmentationResult.save cannot write '.jpg': use .tif, .tiff"):
        SegmentationResult(labels, None, ("a",), "m@1", None).save(tmp_path / "s.jpg")  # type: ignore[arg-type]


def _depth(**kw: object) -> DepthResult:
    fields = dict.fromkeys(("depth", "inverse_depth", "points", "intrinsics", "confidence", "mask", "normals"))
    return DepthResult(**(fields | {"extras": {}, "kind": "metric", "model": "m", "backend": "portable"} | kw))  # type: ignore[arg-type]


def test_depth_save_point_cloud(tmp_path: Path) -> None:
    points, mask = torch.rand(4, 5, 3), torch.ones(4, 5, dtype=torch.bool)
    mask[0] = False
    _depth(depth=points[..., 2], points=points, mask=mask).save(tmp_path / "c.ply")
    text = (tmp_path / "c.ply").read_bytes()
    assert b"element vertex 15" in text
    _depth(depth=points[..., 2], points=points[None], mask=mask[None]).save(tmp_path / "b.ply")
    assert (tmp_path / "b.ply").read_bytes() == text
    with pytest.raises(InputError, match="no points"):
        _depth(depth=points[..., 2]).save(tmp_path / "c.ply")
    with pytest.raises(InputError, match="one image per call"):
        _depth(points=points.expand(2, 4, 5, 3)).save(tmp_path / "c.ply")
    with pytest.raises(InputError, match="key= and bits="):
        _depth(points=points).save(tmp_path / "c.ply", bits=10)
    with pytest.raises(InputError, match=r"DepthResult.save cannot write '.xyz': use .*\.ply"):
        _depth(points=points).save(tmp_path / "c.xyz")


# R2-32: torch dtypes


def test_torch_dtypes_are_accepted() -> None:
    assert _check_args("auto", torch.float16) == _check_args("auto", "float16") == (False, "float16")
    assert _check_args("portable", torch.float32) == (True, None)
    assert _check_args("auto", torch.bfloat16) == (False, "bfloat16")
    with pytest.raises(OptionsError, match="or the torch dtype"):
        _check_args("auto", torch.float64)
    from fastforks import patch  # noqa: PLC0415

    with pytest.raises(OptionsError, match="dtype must be None"):  # torch.float32 is 'float32', so it gets this far
        patch(torch.nn.Linear(2, 2), mode="kernels", dtype=torch.float32)
    with pytest.raises(OptionsError, match="None or 'float32'"):
        patch(torch.nn.Linear(2, 2), dtype=torch.float16)


# R2-34 / R2-35 / R2-33


def test_python_m_fastforks(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["fastforks", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("fastforks", run_name="__main__")
    assert e.value.code == 0
    assert "usage" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("task", "cls"),
    [
        ("stereo", "StereoModel"),
        ("depth", "DepthModel"),
        ("multiview", "MultiViewModel"),
        ("segmentation", "SegmentationModel"),
        ("features", "FeaturesModel"),
        ("flow", "FlowModel"),
        ("matching", "MatchingModel"),
        ("matting", "MattingModel"),
        ("tracking", "TrackingModel"),
    ],
)
def test_load_return_annotation(task: str, cls: str) -> None:
    mod = importlib.import_module(f"fastforks.{task}")
    assert mod.load.__annotations__["return"] is getattr(mod, cls)


def test_ms_rounding() -> None:
    assert [_ms_str(x) for x in (3.4812, 20.14, 214.4, 1234.6, 0.04567, 99.96)] == [
        "3.48",
        "20.1",
        "214",
        "1235",
        "0.0457",
        "100",
    ]
    assert _ms_str("n/a") == "n/a"


# TA-2: max_depth on cloud saves


def test_max_depth_cloud_saves(tmp_path: Path) -> None:
    from fastforks.stereo.geometry.ply import write_cloud  # noqa: PLC0415

    pts = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 9.0], [1.0, 1.0, 7e9]])
    assert write_cloud(tmp_path / "a.ply", pts) == 3  # default unchanged
    assert write_cloud(tmp_path / "b.ply", pts, max_depth=10.0) == 2
    assert write_cloud(tmp_path / "c.ply", pts.numpy(), max_depth=5.0) == 1
    fastforks.save(tmp_path / "d.ply", pts, max_depth=10.0)
    assert (tmp_path / "d.ply").stat().st_size == (tmp_path / "b.ply").stat().st_size
    d = _depth(depth=torch.ones(2, 2), points=torch.tensor([[[0, 0, 1], [0, 0, 20.0]], [[0, 0, 3], [0, 0, 4]]]))
    d.save(tmp_path / "e.ply", max_depth=10.0)
    assert b"element vertex 3" in (tmp_path / "e.ply").read_bytes()[:400]
    with pytest.raises(InputError, match="max_depth= is for point clouds"):
        d.save(tmp_path / "e.tif", max_depth=10.0)


def test_max_depth_run_many_needs_rig(tmp_path: Path) -> None:
    from fastforks.stereo.pipeline import _within  # noqa: PLC0415

    a = np.array([1.0, 50.0], np.float32)
    assert _within(a, None) is a
    assert np.isnan(_within(a, 10.0)).tolist() == [False, True]


def test_vit_dynamic_export_follows_the_input_size() -> None:
    """A dynamic-shape export of the shared ViT gives the eager values at another size (R2-28 b)."""
    from fastforks.shared.depth_anything_v2.net.dinov2 import DinoVisionTransformer  # noqa: PLC0415

    vit = DinoVisionTransformer(img_size=56, patch_size=14, embed_dim=16, depth=1, num_heads=2).eval()
    vit.pos_embed.data.normal_()

    def run(x: torch.Tensor) -> torch.Tensor:
        return vit.prepare_tokens_with_masks(x)

    class Wrap(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return run(x)

    n, m = torch.export.Dim("n", min=2, max=10), torch.export.Dim("m", min=2, max=10)
    ep = torch.export.export(
        Wrap(), (torch.rand(1, 3, 70, 98),), dynamic_shapes=({2: 14 * n, 3: 14 * m},), strict=False
    )
    for size in ((28, 42), (112, 140)):
        x = torch.rand(1, 3, *size)
        with torch.no_grad():
            torch.testing.assert_close(ep.module()(x), run(x), rtol=1e-4, atol=1e-4)
