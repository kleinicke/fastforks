from __future__ import annotations

import types

import pytest
import torch

from fastforks._core import model as core_model
from fastforks._core.cards import Io
from fastforks._core.errors import CardError
from fastforks.stereo._runner import Crop, Runner, crop, pad_to, pad_widths, scale
from fastforks.stereo.models.s2m2.model import Portable as S2M2Portable
from fastforks.stereo.models.s2m2.model import image_pad


class R(Runner):
    dtypes = ("float32", "float16")

    def __call__(self, left, right):  # noqa: ARG002 - test stub
        return {"disparity": left[:, 0]}


def card(dtype, kernels_dtype="float16"):
    return types.SimpleNamespace(
        family="fam", portable_cuda_dtype=dtype, kernels_dtype=kernels_dtype, kernels_dtype_on=lambda _: kernels_dtype
    )


def test_dtype_pick():
    cpu, cuda = torch.device("cpu"), torch.device("cuda:0")
    assert core_model.compute_dtype(card("float16"), R, "portable", cpu) == "float32"
    assert core_model.compute_dtype(card("float16"), R, "portable", cuda) == "float16"
    assert core_model.compute_dtype(card("float16"), R, "kernels", cuda) == "float16"
    with pytest.raises(CardError, match="fam: portable cuda_dtype 'bfloat16' not implemented"):
        core_model.compute_dtype(card("bfloat16"), R, "portable", cuda)


@pytest.mark.parametrize(("h", "w"), [(32, 32), (33, 47), (1, 1)])
@pytest.mark.parametrize("mode", ["centre", "top_right", "bottom_right"])
@pytest.mark.parametrize("fill", ["replicate", "zero"])
def test_pad_roundtrip(h, w, mode, fill):
    io = Io(pad_multiple=32, pad_mode=mode, pad_fill=fill)
    x = torch.randn(1, 2, h, w)
    y, region = pad_to(x, io)
    assert y.shape[-2] % 32 == 0
    assert y.shape[-1] % 32 == 0
    assert torch.equal(crop(y, region), x)


def test_pad_matches_upstream_padder():
    # upstream InputPadder('sintel'): [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    assert pad_widths(33, 47, Io()) == (8, 9, 15, 16)
    assert pad_widths(33, 47, Io(pad_mode="top_right")) == (0, 17, 31, 0)
    assert pad_widths(33, 47, Io(pad_mode="bottom_right")) == (0, 17, 0, 31)


def test_scale():
    x = torch.full((1, 3, 2, 2), 0.5)
    assert torch.equal(scale(x, Io(input="0-255")), x * 255.0)
    assert torch.equal(scale(x, Io(input="0-1")), x)
    want = (0.5 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(scale(x, Io(input="imagenet"))[0, :, 0, 0], want)


@pytest.mark.parametrize(("h", "w"), [(5, 7), (31, 64), (40, 33)])
def test_s2m2_image_pad_small(h, w):
    x = torch.rand(1, 3, h, w)
    runner = types.SimpleNamespace(ctx=types.SimpleNamespace(io=Io(pad_multiple=32)))
    y, region = S2M2Portable.pad(runner, x)  # pad reads only ctx.io
    assert y.shape[-2] % 32 == 0
    assert y.shape[-1] % 32 == 0
    assert torch.equal(y, image_pad(x, 32))
    assert torch.equal(crop(y, region), x)
    assert isinstance(region, Crop)


def test_scale_constants_ignore_the_default_dtype():
    x = torch.full((1, 3, 2, 2), 0.5)
    io = Io(input="imagenet")
    want = scale(x, io)
    torch.set_default_dtype(torch.float64)
    try:
        during = scale(x, io)
    finally:
        torch.set_default_dtype(torch.float32)
    after = scale(x, io)
    assert want.dtype == during.dtype == after.dtype == torch.float32
    assert torch.equal(want, during)
    assert torch.equal(want, after)
