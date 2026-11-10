"""The shared `grid_sample` (`fastforks._core.sampling`): `F.grid_sample` without cuDNN's batch limit.

CPU: the limit fit and the fall-through to `F.grid_sample`. GPU: batches over the limit (where `F.grid_sample` raises
CUDNN_STATUS_NOT_SUPPORTED on cuDNN 9.24) match the native kernel, also under autocast and `torch.compile`; batches
under it stay bit-identical to `F.grid_sample`. Source: every call that can reach cuDNN uses the shared sampler.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastforks._core.sampling import cudnn_max_rows, grid_sample

SRC = Path(__file__).parents[1] / "src" / "fastforks"


@pytest.mark.parametrize(
    ("c", "h_out", "w_out", "rows"),
    [
        (64, 1, 9, 65535),
        (28, 1, 9, 196605),
        (8, 1, 9, 655350),
        (1, 1, 9, 1376235),
        (2, 1, 1, 4194240),
        (8, 4, 9, 65535),
    ],
)
def test_limit_fit(c, h_out, w_out, rows):
    """The measured N_max (cuDNN 9.24, RTX 5060 Ti; N_max + 1 raises)."""
    assert cudnn_max_rows(c, h_out, w_out) == rows


@pytest.mark.parametrize(
    "kw",
    [
        {"align_corners": True},
        {"mode": "nearest", "align_corners": True},
        {"padding_mode": "border", "align_corners": False},
    ],
)
def test_cpu_is_f_grid_sample(kw):
    g = torch.Generator().manual_seed(0)
    img, grid = torch.randn(70000, 2, 1, 8, generator=g), torch.rand(70000, 1, 3, 2, generator=g) * 2 - 1
    torch.testing.assert_close(grid_sample(img, grid, **kw), F.grid_sample(img, grid, **kw), rtol=0, atol=0)


def _native(img, grid):
    return torch.grid_sampler_2d(img, grid, interpolation_mode=0, padding_mode=0, align_corners=True)


def _lookup(n, dtype=torch.float32, c=64, w=16, k=9):
    g = torch.Generator().manual_seed(0)
    img = torch.randn(n, c, 1, w, generator=g).to("cuda", dtype)
    x = torch.rand(n, 1, k, 1, generator=g) * 2 - 1
    return img, torch.cat([x, torch.zeros_like(x)], -1).to("cuda", dtype)


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_over_the_limit_matches_native(cuda, dtype):
    img, grid = _lookup(65536, dtype)  # N_max + 1 for C = 64, W_out = 9
    out = grid_sample(img, grid, mode="bilinear", align_corners=True)
    ref = _native(img.float(), grid.float())
    torch.testing.assert_close(out.float(), ref, rtol=0, atol=2e-3 if dtype == torch.float16 else 1e-6)


@pytest.mark.gpu
def test_under_the_limit_is_f_grid_sample(cuda):
    img, grid = _lookup(65535)
    ref = F.grid_sample(img, grid, mode="bilinear", align_corners=True)
    torch.testing.assert_close(grid_sample(img, grid, align_corners=True), ref, rtol=0, atol=0)


@pytest.mark.gpu
def test_over_the_limit_under_autocast_is_fp32(cuda):
    img, grid = _lookup(65536, torch.float16)
    with torch.autocast("cuda", dtype=torch.float16):
        out = grid_sample(img, grid, align_corners=True)
    assert out.dtype == torch.float32  # autocast runs grid_sample in fp32
    torch.testing.assert_close(out, _native(img.float(), grid.float()), rtol=0, atol=0)


@pytest.mark.gpu
def test_over_the_limit_compiles(cuda):
    """Dynamo's fake-tensor trace of `F.grid_sample` fails on these batches as well."""
    img, grid = _lookup(65536)
    torch._dynamo.reset()
    f = torch.compile(lambda a, g: grid_sample(a, g, align_corners=True), backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(f(img, grid), _native(img, grid), rtol=0, atol=1e-6)


_UNKNOWN = object()


def _may_use_cudnn(call: ast.Call) -> bool:
    """cuDNN takes bilinear, zeros, align_corners=True (F.grid_sample defaults: bilinear, zeros, False)."""
    if len(call.args) > 2:
        return True
    kw = {k.arg: k.value for k in call.keywords}

    def arg(name, default=None):
        v = kw.get(name)
        if v is None:
            return default
        return v.value if isinstance(v, ast.Constant) else _UNKNOWN

    return (
        arg("align_corners") not in (False, None)
        and arg("mode", "bilinear") in ("bilinear", _UNKNOWN)
        and arg("padding_mode", "zeros") in ("zeros", _UNKNOWN)
    )


@pytest.mark.slow
def test_cudnn_reachable_calls_use_the_shared_sampler():
    own = SRC / "_core" / "sampling.py"
    bad = [
        f"{path.relative_to(SRC)}:{node.lineno}"
        for path in sorted(SRC.rglob("*.py"))
        if path != own
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "grid_sample"
        and _may_use_cudnn(node)
    ]
    assert not bad, f"F.grid_sample that can hit cuDNN's batch limit (use fastforks._core.sampling): {bad}"
