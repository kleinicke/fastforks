"""`mps.adapt`'s transposed convolutions against torch's, on the CPU (the pure-PyTorch maths needs no MPS)."""

from __future__ import annotations

import itertools
import subprocess
import sys

import pytest
import torch
from torch import nn

from fastforks._core import mps


def _pair(ref: nn.Module, x: torch.Tensor, output_size: list[int] | None = None, tol: float = 1e-9) -> None:
    ref = ref.double()
    m = type(ref)(
        ref.in_channels,
        ref.out_channels,
        ref.kernel_size,
        ref.stride,
        ref.padding,
        ref.output_padding,
        ref.groups,
        ref.bias is not None,
        ref.dilation,
    ).double()
    m.load_state_dict(ref.state_dict())
    mps.adapt(m)
    kw = {} if output_size is None else {"output_size": output_size}
    want, got = ref(x, **kw), m(x, **kw)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "cfg",
    [
        c
        for c in itertools.product((2, 3), (1, 2, 3, 4), (1, 2, 3, 4), (0, 1), (0, 1), (1, 2), (1, 2))
        if c[4] < c[2] and c[3] < c[1] and (c[0] == 2 or c[1] < 4)  # (nd, kernel, stride, padding, op, dil, groups)
    ],
)
def test_matches_torch(cfg: tuple[int, ...]):
    nd, kernel, stride, padding, output_padding, dilation, groups = cfg
    torch.manual_seed(0)
    cls = nn.ConvTranspose2d if nd == 2 else nn.ConvTranspose3d
    ref = cls(4, 4, kernel, stride, padding, output_padding, groups, bias=True, dilation=dilation)
    # a 3-D layer off the polyphase path runs in float32 (MPS has no fp16 one): float32 tolerance
    _pair(ref, torch.randn(2, 4, *([4, 3, 5][:nd])).double(), tol=1e-9 if nd == 2 else 1e-4)


@pytest.mark.parametrize(("stride", "kernel", "nd"), [(2, 3, 3), (2, 4, 2), (4, 4, 2), (3, 3, 2), (2, 3, 2)])
def test_output_size_per_axis(stride: int, kernel: int, nd: int):
    """An odd `output_size` is per-axis output padding: only stride 2 may take the polyphase split."""
    torch.manual_seed(0)
    cls = nn.ConvTranspose2d if nd == 2 else nn.ConvTranspose3d
    p = 1 if stride == 2 else 0
    base = [(3 - 1) * stride - 2 * p + kernel] * nd
    size = [*base[:-1], base[-1] + 1]
    _pair(cls(2, 2, kernel, stride, p), torch.randn(1, 2, *([3] * nd)).double(), size, tol=1e-9 if nd == 2 else 1e-4)


def test_per_axis_output_padding_stride4():
    torch.manual_seed(0)
    _pair(nn.ConvTranspose2d(2, 2, 4, 4, 0, (1, 0)), torch.randn(1, 2, 3, 3).double())


def test_unbatched():
    torch.manual_seed(0)
    _pair(nn.ConvTranspose2d(2, 2, 3, 2, 1, 1), torch.randn(2, 4, 4).double())
    _pair(nn.ConvTranspose2d(2, 2, 2, 2), torch.randn(2, 4, 4).double())


def test_autocast_dtype():
    torch.manual_seed(0)
    ref = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)
    m = nn.ConvTranspose2d(2, 2, 3, 2, 1, 1)
    m.load_state_dict(ref.state_dict())
    mps.adapt(m)
    x = torch.randn(1, 2, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(x).dtype == ref(x).dtype == torch.bfloat16


def test_metal_runners_adapt():
    """Every `MetalKernels` runner that subclasses a portable one adapts its transposed convs itself."""
    from pathlib import Path  # noqa: PLC0415

    import fastforks  # noqa: PLC0415

    found = 0
    for path in Path(fastforks.__file__).parent.rglob("model.py"):
        src = path.read_text()
        if "class MetalKernels(Portable)" in src:
            found += 1
            assert "mps.adapt(self.net)" in src, path
    assert found >= 3


@pytest.mark.parametrize("exc", [RuntimeError, ValueError, AttributeError])
def test_metal_compile_failure_is_unavailable(monkeypatch: pytest.MonkeyPatch, exc: type[Exception]):
    from fastforks._core.errors import KernelUnavailableError  # noqa: PLC0415
    from fastforks._core.metal import MetalLibrary  # noqa: PLC0415

    def boom(_src: str) -> None:
        msg = "bad shader"
        raise exc(msg)

    monkeypatch.setattr(torch.mps, "compile_shader", boom, raising=False)
    with pytest.raises(KernelUnavailableError, match="Metal compile failed"):
        MetalLibrary("x", "src")


def test_metal_sources_have_no_infinity_compare():
    """Metal compiles with fast math (no infinities): a mask test against INFINITY can be optimised away."""
    from pathlib import Path  # noqa: PLC0415

    import fastforks  # noqa: PLC0415

    for path in (Path(fastforks.__file__).parent / "kernels").rglob("*.metal"):
        src = path.read_text()
        assert "== -INFINITY" not in src, path
        assert "m = -INFINITY" not in src, path


def test_subm_conv_dtype_guards():
    from fastforks.kernels import sparse_metal  # noqa: PLC0415

    conv = sparse_metal.SubmConv(None, torch.zeros(32, 3, 3, 3, 32), torch.zeros(32))
    nbr = torch.full((27, 4), -1, dtype=torch.int32)
    with pytest.raises(TypeError, match="must match"):
        conv(torch.zeros(4, 32, dtype=torch.float16), nbr)
    with pytest.raises(ValueError, match="int32"):
        conv(torch.zeros(4, 32), nbr.long())


def test_kernels_module_imports_without_grp_pwd():
    """Windows has no `grp` / `pwd`: `_core.kernels` must import (and its group check say no) without them."""
    code = (
        "import sys\n"
        "sys.modules['grp'] = None; sys.modules['pwd'] = None\n"
        "from fastforks._core import kernels\n"
        "assert kernels._own_group(0) is False\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("family", ["nmrf", "bridgedepth"])
def test_cuda_kernels_module_does_not_import_metal(family: str):
    code = (
        f"import sys, fastforks.stereo.models.{family}.kernels.inference\n"
        "bad = [m for m in sys.modules if 'stereo_metal' in m or m == 'fastforks._core.metal']\n"
        "assert not bad, bad\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
