"""Sort-free quantiles in exported graphs: `shared.quantile`, MonSter's mono fit, Depth Anything 3."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from fastforks.shared.monster_fit import _sorted_at, fit, scale_shift
from fastforks.shared.quantile import kth_smallest, quantile
from fastforks.stereo.models.monster.net.monster import compute_scale_shift


def _data(seed: int, shape=(1, 23, 31)) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mono = torch.rand(shape, generator=g) * 40 - 2
    mono[..., :3] = mono[..., 3:4]  # ties at the quantile
    gt = 1.7 * mono + 3 + torch.randn(shape, generator=g) * 0.1
    gt[..., 5] = 0  # gt <= 0 is masked
    return mono, gt


class _Kth(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _sorted_at(x.reshape(-1), int(0.2 * x.numel()))


class _Fit(torch.nn.Module):
    def forward(self, mono: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
        return fit(mono, disp)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
def test_exported_quantile_is_the_sorted_value(seed):
    mono, _ = _data(seed)
    ep = torch.export.export(_Kth().eval(), (mono,), strict=False)
    assert not any("sort" in str(n.target) or "topk" in str(n.target) for n in ep.graph.nodes)
    for s in range(5):  # the bisection is not specialized to the traced values
        x, _ = _data(100 + s)
        assert ep.module()(x).item() == torch.sort(x.reshape(-1))[0][int(0.2 * x.numel())].item()


@pytest.mark.parametrize("seed", range(5))
def test_scale_shift_matches_upstream(seed):
    mono, gt = _data(seed)
    want = compute_scale_shift(mono.clone(), gt.clone())
    got = scale_shift(mono, gt)
    assert got[0].item() == pytest.approx(want[0], rel=1e-4)
    assert got[1].item() == pytest.approx(want[1], abs=2e-3)  # px; upstream solves the fp32 normal equations


@pytest.mark.slow
def test_exported_fit_matches_eager():
    mono, gt = _data(0, (2, 1, 23, 31))
    want = fit(mono, gt)
    ep = torch.export.export(_Fit().eval(), (mono, gt), strict=False)
    ops = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
    assert not any("sort" in o or "solve" in o for o in ops), ops
    torch.testing.assert_close(ep.module()(mono, gt), want, rtol=1e-6, atol=1e-5)


@pytest.mark.skipif(
    not (importlib.util.find_spec("onnxscript") and importlib.util.find_spec("onnxruntime")),
    reason="needs onnxscript + onnxruntime (dependency group export)",
)
def test_onnx_fit_has_no_topk(tmp_path):
    import onnx  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    mono, gt = _data(1, (1, 1, 23, 31))
    path = str(tmp_path / "m.onnx")
    torch.onnx.export(_Fit().eval(), (mono, gt), path, dynamo=True, input_names=["mono", "disp"])
    assert not [n for n in onnx.load(path).graph.node if n.op_type in {"TopK", "Sort"}]
    got = ort.InferenceSession(path).run(None, {"mono": mono.numpy(), "disp": gt.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(got), fit(mono, gt), rtol=1e-5, atol=1e-4)


class _Quantile(torch.nn.Module):
    def __init__(self, q: float) -> None:
        super().__init__()
        self.q = q

    def forward(self, v: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        from fastforks.depth.models.depth_anything_3.model import _quantile  # noqa: PLC0415

        return _quantile(v, valid, self.q)


@pytest.mark.slow
@pytest.mark.parametrize("q", [0.5, 0.99, 0.0, 1.0, 0.37])
def test_da3_exported_quantile_matches_sort(q):
    g = torch.Generator().manual_seed(0)
    v = torch.rand(4, 999, generator=g) * 80
    v[:, :40] = v[:, 40:41]  # ties
    valid = torch.rand(4, 999, generator=g) > 0.3
    valid[2] = False  # no valid entry
    valid[3] = False
    valid[3, 7] = True  # one valid entry
    want = _Quantile(q)(v, valid)
    ep = torch.export.export(_Quantile(q).eval(), (v, valid), strict=False)
    assert not any("sort" in str(n.target) or "topk" in str(n.target) for n in ep.graph.nodes)
    torch.testing.assert_close(ep.module()(v, valid), want, rtol=0, atol=0, equal_nan=True)


def test_kth_smallest_rows():
    g = torch.Generator().manual_seed(1)
    v = torch.randn(3, 5, 257, generator=g).half()
    k = torch.randint(0, 257, (3, 5), generator=g)
    want = v.sort(-1).values.gather(-1, k[..., None])[..., 0]
    assert torch.equal(kth_smallest(v, k), want)
    assert kth_smallest(v, 0).dtype == torch.float16


@pytest.mark.parametrize("q", [0.05, 0.95, 0.5, 0.0, 1.0, 0.37])
def test_quantile_matches_torch(q):
    g = torch.Generator().manual_seed(3)
    v = torch.rand(3, 257, generator=g) * 9 + 0.5
    v[0, :40] = v[0, 40]  # ties
    v[1] = torch.round(v[1])  # many ties
    torch.testing.assert_close(quantile(v, q), torch.quantile(v, q, dim=-1), rtol=1e-6, atol=1e-6)


@pytest.mark.slow
def test_mvsanywhere_exported_range_matches_eager():
    from fastforks.multiview.models.mvsanywhere.model import refine_range  # noqa: PLC0415

    class M(torch.nn.Module):
        def forward(self, depth: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
            return refine_range(depth, rng)

    g = torch.Generator().manual_seed(4)
    depth = torch.rand(1, 1, 48, 64, generator=g) * 7 + 0.3  # 3072 + pixels: more than TensorRT's TopK takes
    rng = torch.tensor([0.25, 9.0])
    ep = torch.export.export(M().eval(), (depth, rng), strict=False)
    assert not [n for n in ep.graph.nodes if any(o in str(n.target) for o in ("sort", "quantile", "topk"))]
    torch.testing.assert_close(ep.module()(depth, rng), M()(depth, rng), rtol=1e-6, atol=1e-6)
