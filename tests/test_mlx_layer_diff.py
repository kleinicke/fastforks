"""`mlx_port.tap` / `taps` and `tools/mlx_layer_diff.py`'s matching and comparison, on small made-up data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import mlx_layer_diff as ld


class _T:
    def __init__(self, name: str, value: np.ndarray | None, ms: float = 1.0) -> None:
        self.name, self.value, self.ms = name, value, ms
        self.shape = () if value is None else tuple(value.shape)


def test_split() -> None:
    assert ld.split("blocks.3") == ("blocks.3", None)
    assert ld.split("self_att_fn[1]") == ("self_att_fn", 1)


def test_align_layouts() -> None:
    rng = np.random.default_rng(0)
    nchw = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
    got = ld.align(nchw, nchw.transpose(0, 2, 3, 1))  # NHWC
    assert got is not None
    np.testing.assert_array_equal(got, nchw)
    ncdhw = rng.standard_normal((1, 2, 3, 4, 5)).astype(np.float32)
    got = ld.align(ncdhw, ncdhw.transpose(0, 2, 3, 4, 1))
    assert got is not None
    np.testing.assert_array_equal(got, ncdhw)
    tokens = rng.standard_normal((2, 7, 6)).astype(np.float32)
    np.testing.assert_array_equal(ld.align(tokens, tokens), tokens)  # NOT read as NHWC
    fmap = rng.standard_normal((2, 6, 4, 5)).astype(np.float32)  # tokens [B, N, C] of a [B, C, h, w] map
    got = ld.align(fmap, fmap.reshape(2, 6, 20).transpose(0, 2, 1))
    assert got is not None
    np.testing.assert_array_equal(got, fmap)
    assert ld.align(nchw, rng.standard_normal((2, 3)).astype(np.float32)) is None


def test_compare_per_call_and_tuple() -> None:
    a, b = np.ones((1, 2, 2, 3), np.float32), np.full((1, 2, 2, 3), 2.0, np.float32)
    taps = [_T("blk", a.transpose(0, 2, 3, 1)), _T("blk", b.transpose(0, 2, 3, 1) + 0.5), _T("att[1]", b), _T("x", a)]
    recorded = {"net.blk": [a, b], "net.att": [(a, b)]}
    rows = ld.compare(taps, recorded, {"blk": "net.blk", "att": "net.att", "x": None}, times=[1.0, 2.0, 3.0, 4.0])
    assert rows[0].max == 0.0
    assert rows[1].max == pytest.approx(0.5)
    assert rows[1].rel == pytest.approx(0.25)
    assert rows[2].max == 0.0
    assert rows[3].torch == "no torch module"
    assert rows[3].rel is None
    assert [r.ms for r in rows] == [1.0, 2.0, 3.0, 4.0]
    assert ld.first_diverging(rows, 0.1) is rows[1]
    assert ld.first_diverging(rows, 0.3) is None
    assert "!" in ld.table(rows, 0.1).splitlines()[2]


def test_compare_extra_calls_and_shape() -> None:
    a = np.ones((1, 2, 2, 3), np.float32)
    rows = ld.compare(
        [_T("blk", a), _T("blk", a), _T("m", np.ones((5,), np.float32))],
        {"blk": [a], "m": [a]},
        {"blk": "blk", "m": "m"},
    )
    assert rows[1].torch == "blk: called 1x"
    assert rows[2].torch.startswith("m: shape")


def test_match_exact_then_shortest_suffix() -> None:
    torch = pytest.importorskip("torch")
    net = torch.nn.Module()
    net.pretrained = torch.nn.Module()
    net.pretrained.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
    net.head = torch.nn.Module()
    net.head.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    runner = type("R", (), {})()
    part = type("P", (), {})()
    part.nets = [net]  # an instance attribute, as a runner sets it
    runner.parts = [part]
    mods = ld.roots(runner)
    assert [p for p, _ in mods] == ["parts.0.nets.0"]
    found = ld.match(["blocks.1", "head.blocks.0", "pretrained", "nope"], mods)
    assert found["blocks.1"] is not None
    assert found["blocks.1"][0] == "parts.0.nets.0.pretrained.blocks.1"
    assert found["head.blocks.0"] is not None
    assert found["head.blocks.0"][0] == "parts.0.nets.0.head.blocks.0"
    assert found["pretrained"] is not None
    assert found["nope"] is None


def test_match_takes_the_path_in_every_root() -> None:
    torch = pytest.importorskip("torch")

    def net() -> object:
        n = torch.nn.Module()
        n.enc = torch.nn.Module()
        n.enc.conv = torch.nn.Identity()
        return n

    a, b = net(), net()
    found = ld.match(["conv", "enc.conv", "p1.enc.conv"], [("p0", a), ("p1", b)])
    assert found["conv"] == ("p0.enc.conv (+1 roots)", [a.enc.conv, b.enc.conv])
    assert found["enc.conv"] == found["conv"]
    assert found["p1.enc.conv"] == ("p1.enc.conv", [b.enc.conv])


def test_taps_record_only_inside() -> None:
    mx = pytest.importorskip("mlx.core")
    from fastforks.shared import mlx_port  # noqa: PLC0415 - needs mlx

    x = mx.arange(6, dtype=mx.float16).reshape(2, 3)
    assert mlx_port.tap("a", x) is x  # identity, nothing recorded
    with mlx_port.taps() as rows:
        y = mlx_port.tap("a", x * 2)
        mlx_port.tap("b[0]", y + 1)
    assert [r.name for r in rows] == ["a", "b[0]"]
    assert rows[0].shape == (2, 3)
    assert rows[0].value is not None
    assert rows[0].value.dtype == np.float32
    np.testing.assert_array_equal(rows[1].value, np.arange(6).reshape(2, 3) * 2 + 1)
    assert all(r.ms >= 0 for r in rows)
    with mlx_port.taps(keep=False) as rows:
        mlx_port.tap("a", x)
    assert rows[0].value is None
    assert mlx_port.tap("a", x) is x
