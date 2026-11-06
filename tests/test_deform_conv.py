"""The `deform_conv` kernel library: weight packing, fit rules and argument checks (CPU), parity (GPU)."""

import pytest
import torch

from fastforks.kernels import deform_conv as dk

CL = torch.channels_last


def test_pack_weight_layout():
    w = torch.randn(40, 32, 3, 3)
    p = dk.pack_weight(w)
    assert p.shape == (9 * 32, 48)
    assert p.dtype == torch.float16
    assert p.is_contiguous()
    # row tap * C + c, column o; zero padding columns
    assert torch.equal(p[(1 * 3 + 2) * 32 + 5, :40], w[:, 5, 1, 2].half())
    assert not p[:, 40:].any()


@pytest.mark.parametrize(
    ("c", "dg", "fits"),
    [
        (32, 2, True),
        (16, 2, True),
        (256, 2, True),
        (48, 1, True),
        (24, 1, False),
        (32, 8, False),
        (40, 1, False),
        (48, 2, True),
    ],
)
def test_fit_reason(c, dg, fits):
    assert (dk.fit_reason(c, dg) is None) == fits


def _args(c=32, o=32, h=12, w=14, dg=2, om_channels=None):  # noqa: PLR0913, PLR0917 - geometry
    x = torch.zeros(1, c, h, w, dtype=torch.float16).contiguous(memory_format=CL)
    om = torch.zeros(1, om_channels or dg * 27, h, w, dtype=torch.float16).contiguous(memory_format=CL)
    return x, om, dk.pack_weight(torch.zeros(o, c, 3, 3)), o, (3, 3), (1, 1), (2, 2), (2, 2), dg, 2.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"c": 24}, r"16 \| C"),
        ({"om_channels": 36}, "need 54 channels"),
    ],
)
def test_rejects_misfit_arguments(change, match):
    with pytest.raises(ValueError, match=match):
        dk.deform_conv(*_args(**change))


def test_rejects_nchw_input():
    x, *rest = _args()
    with pytest.raises(ValueError, match="channels-last fp16"):
        dk.deform_conv(x.contiguous(), *rest)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("b", "c", "o", "h", "w", "stride", "dil", "dg", "modulated"),
    [
        (1, 32, 32, 40, 52, 1, 2, 2, True),
        (1, 64, 96, 25, 35, 2, 2, 2, True),
        (1, 192, 96, 17, 23, 1, 2, 2, True),
        (1, 16, 16, 15, 21, 1, 2, 2, True),
        (2, 48, 40, 17, 23, 1, 1, 1, False),
        (1, 64, 200, 13, 19, 1, 2, 4, True),
    ],
)
def test_matches_reference(b, c, o, h, w, stride, dil, dg, modulated):  # noqa: PLR0913, PLR0917 - geometry
    from fastforks.stereo.models.aanet.net.deform_conv import modulated_deform_conv  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    k = 9
    ho, wo = (h - 1) // stride + 1, (w - 1) // stride + 1
    x = torch.randn(b, c, h, w, generator=g).half().to(dev).contiguous(memory_format=CL)
    off = torch.randn(b, dg * 2 * k, ho, wo, generator=g) * 3
    logit = torch.randn(b, dg * k, ho, wo, generator=g)
    om = (torch.cat([off, logit], 1) if modulated else off).half().to(dev).contiguous(memory_format=CL)
    wt = (torch.randn(o, c, 3, 3, generator=g) / (c * 9) ** 0.5).to(dev)
    s, t = (torch.rand(o, generator=g) + 0.5).to(dev), torch.randn(o, generator=g).to(dev)
    mask = om[:, dg * 2 * k :].sigmoid() * 2 if modulated else None
    ref = modulated_deform_conv(x, om[:, : dg * 2 * k], mask, wt.half(), None, stride, dil, dil, 1, dg).float()
    ref = torch.relu(ref * s.view(1, -1, 1, 1) + t.view(1, -1, 1, 1))
    got = dk.deform_conv(
        x, om, dk.pack_weight(wt), o, (3, 3), (stride, stride), (dil, dil), (dil, dil), dg,
        2.0 if modulated else None, s, t, dk.ACT_RELU,
    )  # fmt: skip
    assert got.shape == ref.shape
    assert got.is_contiguous(memory_format=CL)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=1e-2)
