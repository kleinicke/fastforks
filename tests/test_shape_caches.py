"""Per-input-size caches (position / view-plane / RoPE tables): keyed by everything they depend on, and bounded.

A model that ran size A and then size B must give the same output at B as a fresh model: a cache key that misses an
input of the table (MoGe's view-plane tables were keyed by the token grid, not the aspect) reuses A's table at B. A and
B share MoGe's token grid at different aspects.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from fastforks._core.bench import BenchSize, bench_inputs
from fastforks.shared.depth_anything_v2 import fast_vit
from test_golden import API

A, B = (384, 576), (400, 596)  # MoGe: both 49 x 73 tokens (resolution_level 9), aspects 1.5 and 1.49
REFS = [
    ("depth", "moge-2-vits-normal", "kernels"),
    ("depth", "moge-2-vits-normal", "portable"),
    ("depth", "da3-large-1.1", "kernels"),
    ("depth", "depth-anything-v2-s", "kernels"),
    ("depth", "unik3d-vitl", "kernels"),
    ("depth", "unidepth-v2-vitl14", "kernels"),
    ("features", "dinov2-vits14", "kernels"),
    ("features", "dinov3-vits16", "kernels"),
    ("multiview", "vggt-omega-1b-512", "kernels"),
    ("multiview", "mapanything-apache", "kernels"),
    ("multiview", "patchmatchnet", "kernels"),
]


def _maps(r: Any) -> dict[str, torch.Tensor]:
    fields = {f.name: getattr(r, f.name) for f in dataclasses.fields(r)} | dict(getattr(r, "extras", None) or {})
    return {k: v.float() for k, v in fields.items() if isinstance(v, torch.Tensor)}


def _call(m: Any, size: tuple[int, int]) -> dict[str, torch.Tensor]:
    args, kw = bench_inputs(m, BenchSize(*size, 2))
    return _maps(m(*args, **kw))


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.parametrize(("task", "ref", "mode"), REFS, ids=[f"{r}-{m}" for _, r, m in REFS])
def test_size_after_other_size_matches_fresh(task, ref, mode):
    """B after A == B on a fresh model (both B calls are the first call at B: same cuDNN plans / graph state)."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    m = API[task].load(ref, mode=mode, device="cuda")
    _call(m, A)
    got = _call(m, B)
    del m
    want = _call(API[task].load(ref, mode=mode, device="cuda"), B)
    assert got.keys() == want.keys()
    for k, w in want.items():
        assert torch.equal(got[k], w), f"{k}: max |diff| {(got[k] - w).abs().nan_to_num().max().item():.3g}"


def test_fast_vit_pos_cache_bounded():
    vit = SimpleNamespace(pos_embed=torch.zeros(1), interpolate_offset=0.1, interpolate_antialias=False)
    f = object.__new__(fast_vit.FastViT)
    f.vit, f.pos_fp32, f._pos, f.cache_pos = vit, torch.randn(1, 1 + 4 * 4, 8), {}, True
    for side in range(2, 2 + fast_vit.POS_SIZES + 3):  # every patch grid adds a table
        f.pos_embed(side, side + 1)
    assert len(f._pos) == fast_vit.POS_SIZES
    assert (2, 3) not in f._pos  # the oldest grids went first
    torch.testing.assert_close(f.pos_embed(4, 4), f.pos_fp32.half(), rtol=0, atol=0)  # the table's own grid


@pytest.mark.parametrize(("nh", "nw"), [(1, 1), (1, 5), (4, 1), (3, 4), (11, 7)])
def test_swin_window_order_is_the_stable_border_sort(nh, nw):
    """Swin's shifted-window geometry: interior windows first, as the stable sort it replaces (ONNX has none)."""
    from fastforks.shared.maskformer import _interior_first  # noqa: PLC0415

    wy, wx = (g.flatten() for g in torch.meshgrid(torch.arange(nh), torch.arange(nw), indexing="ij"))
    want = torch.argsort(((wy == nh - 1) | (wx == nw - 1)).to(torch.uint8), stable=True)
    assert torch.equal(_interior_first(nh, nw), want)


@pytest.mark.parametrize(("h", "w", "ws", "s"), [(7, 9, 4, 2), (8, 8, 4, 2), (5, 13, 7, 0), (12, 10, 4, 0)])
def test_swin_inverse_index_fits_tensorrt_scatter(h, w, ws, s):
    """The exported inverse-window scatter has no more indices than data (TensorRT's ScatterElements); same values."""
    from fastforks.shared import maskformer  # noqa: PLC0415

    class M(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + maskformer._geom(h, w, ws, s, x.device).inv

    maskformer._geom.cache_clear()
    ep = torch.export.export(M(), (torch.zeros(h * w, dtype=torch.long),), strict=False)
    maskformer._geom.cache_clear()
    scatters = [n for n in ep.graph.nodes if "scatter" in str(n.target)]
    assert scatters
    for n in scatters:
        assert n.args[2].meta["val"].numel() <= n.args[0].meta["val"].numel()
    # the reference: zeros(n + 1).scatter_ (indices may outnumber the data in torch)
    hp, wp = -(-h // ws) * ws, -(-w // ws) * ws
    g = maskformer._geom(h, w, ws, s, torch.device("cpu"))
    idx = g.idx
    want = torch.zeros(h * w + 1, dtype=torch.long).scatter_(0, idx, torch.arange(idx.numel()))[: h * w]
    assert idx.numel() == hp * wp
    assert torch.equal(g.inv, want)
