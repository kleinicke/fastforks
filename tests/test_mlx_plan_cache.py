"""On-disk MLX plan cache (`FASTFORKS_MLX_PLAN_CACHE`, `shared/mlx_plan_cache.py`) on a tiny conv net."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("mlx.core")

from fastforks.shared import mlx_graph, mlx_plan_cache

pytestmark = pytest.mark.mlx


class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(8)
        self.register_buffer("empty", torch.zeros(0, 0))  # an empty buffer must not break the save

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y = torch.relu(self.bn(self.conv(x)))
        return {"y": y, "mean": y.mean((2, 3))}


def _net() -> Net:
    torch.manual_seed(0)
    net = Net().eval()
    with torch.no_grad():
        net.bn.running_mean.uniform_(-1, 1)
        net.bn.running_var.uniform_(0.5, 2)
    return net


def _run(x: torch.Tensor, scope: str | None = "tiny") -> dict[str, torch.Tensor]:
    if scope is None:
        return mlx_graph.MlxModule(_net())(x)
    with mlx_plan_cache.scope(scope):
        m = mlx_graph.MlxModule(_net())
    return m(x)


def test_hit_skips_the_export_and_is_bit_identical(tmp_path, monkeypatch):
    x = torch.randn(1, 3, 32, 48)
    plain = _run(x)
    monkeypatch.setenv(mlx_plan_cache.ENV, str(tmp_path))
    miss = _run(x)
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".pkl", ".pt2"]

    def no_export(*args, **kwargs):
        msg = "a cache hit must not export"
        raise AssertionError(msg)

    monkeypatch.setattr(torch.export, "export", no_export)
    hit = _run(x)
    for k in plain:
        assert torch.equal(plain[k], miss[k])
        assert torch.equal(plain[k], hit[k])


def test_other_inputs_scopes_and_unscoped_modules_miss(tmp_path, monkeypatch):
    monkeypatch.setenv(mlx_plan_cache.ENV, str(tmp_path))
    _run(torch.randn(1, 3, 32, 48))
    _run(torch.randn(1, 3, 32, 64))
    _run(torch.randn(1, 3, 32, 48), scope="other")
    assert len(list(tmp_path.glob("*.pt2"))) == 3
    _run(torch.randn(1, 3, 32, 48), scope=None)  # built outside a model load: nothing cached
    assert len(list(tmp_path.glob("*.pt2"))) == 3


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(mlx_plan_cache.ENV, raising=False)
    assert mlx_plan_cache.directory() is None
    assert mlx_plan_cache.Entry.make("tiny#0", _net(), ()) is None
