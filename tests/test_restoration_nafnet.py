"""NAFNet: the card's variants, the folded blocks and the TLC box means against the vendored upstream, `load(path)`."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pytest
import torch

from fastforks import restoration
from fastforks._core import registry
from fastforks._core.bench import synthetic_image
from fastforks._core.errors import WeightsError
from fastforks.restoration._api import _checkpoint_family
from fastforks.restoration.models.nafnet.net import NAFNet, NAFNetLocal

if TYPE_CHECKING:
    from pathlib import Path

KW = {"width": 8, "enc_blk_nums": [1, 2], "middle_blk_num": 1, "dec_blk_nums": [1, 1]}


def test_card_variants() -> None:
    cards = {v.id: (c, v) for c, v in registry.cards("restoration").values() if c.family == "nafnet"}
    assert sorted(cards) == ["nafnet-gopro", "nafnet-sidd"]
    assert {**cards["nafnet-gopro"][0].params, **cards["nafnet-gopro"][1].params}["tlc"] == 256
    assert {**cards["nafnet-sidd"][0].params, **cards["nafnet-sidd"][1].params}["tlc"] == 0


def _trained_like(net: torch.nn.Module) -> torch.nn.Module:
    """Random norm affines and residual scales (init: ones / zeros, which would hide folding errors)."""
    torch.manual_seed(1)
    for name, p in net.named_parameters():
        if name.endswith(("beta", "gamma")) or ".norm" in name:
            p.data.add_(torch.randn_like(p) * 0.5)
    return net.eval()


@pytest.mark.parametrize("tlc", [0, 32])
def test_matches_vendored_upstream(tmp_path: Path, tlc: int) -> None:
    """`load(path)` finds the family; the folded blocks (and with `tlc` the local means) match upstream."""
    torch.manual_seed(0)
    net = _trained_like(NAFNetLocal(**KW, train_size=(1, 3, tlc, tlc)) if tlc else NAFNet(**KW))
    torch.save({"params": net.state_dict()}, tmp_path / "w.pth")
    assert _checkpoint_family(tmp_path / "w.pth") == "nafnet"
    m = restoration.load(tmp_path / "w.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == "nafnet-sidd"  # not the deblurring structure: global attention
    if tlc:  # the GoPro path (`NAFNetLocal`): the same runner with a small training size
        ctx = m._runner.ctx
        params = dataclasses.replace(ctx.params, tlc=tlc)
        runner = type(m._runner)(dataclasses.replace(ctx, params=params, user_state=net.state_dict()))
    else:
        runner = m._runner
    x = synthetic_image(41, 57, torch.device("cpu"))  # padded to 44 x 60: box means at every level for tlc 32
    with torch.inference_mode():
        want, got = net(x), runner(x)["image"]
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_rejects_non_rgb(tmp_path: Path) -> None:
    torch.save(NAFNet(img_channel=1, **KW).state_dict(), tmp_path / "w.pth")
    with pytest.raises(WeightsError, match="input channels"):
        restoration.load(tmp_path / "w.pth", mode="portable", device="cpu", dtype="float32")


@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("mode", ["portable", "kernels"])
@pytest.mark.parametrize("model", ["nafnet-gopro", "nafnet-sidd"])
def test_float16_finite_on_stress_inputs(model: str, mode: str) -> None:
    """float16 stays finite on low-contrast noise (the stream once overflowed), constant and saturated images."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    gen = torch.Generator().manual_seed(1)
    noise = torch.rand(1, 3, 256, 320, generator=gen)
    inputs = {
        "noise10": (0.5 + 0.1 * (noise - 0.5)).clamp(0, 1),
        "noise_full": noise,
        "synthetic": synthetic_image(256, 320, torch.device("cpu")),
        "const0": torch.zeros(1, 3, 256, 320),
        "const1": torch.ones(1, 3, 256, 320),
        "binary": (noise > 0.5).float(),
    }
    m = restoration.load(model, mode=mode)
    for name, x in inputs.items():
        out = m._runner(x.cuda())["image"]
        assert torch.isfinite(out).all(), f"{model} [{mode}] {name}: NaN / inf in float16"


@pytest.mark.gpu
@pytest.mark.parametrize(("n", "c", "h", "w"), [(2, 7, 3, 5), (1, 64, 1, 1), (1, 64, 13, 11)])
def test_fused_depthwise_gate(n: int, c: int, h: int, w: int) -> None:
    from fastforks.kernels.nafnet_gate import depthwise_gate  # noqa: PLC0415

    torch.manual_seed(5)
    x = torch.randn(n, 2 * c, h, w, device="cuda", dtype=torch.float16)
    weight = torch.randn(2 * c, 1, 3, 3, device="cuda", dtype=x.dtype)
    bias = torch.randn(2 * c, device="cuda", dtype=x.dtype)
    # Float reference with the model's fp16 rounding between convolution and gate.
    with torch.backends.cudnn.flags(allow_tf32=False):
        a, b = (
            torch.nn.functional.conv2d(x.float(), weight.float(), bias.float(), padding=1, groups=2 * c)
            .half()
            .chunk(2, 1)
        )
    actual = depthwise_gate(x, weight.flatten(1).T.contiguous(), bias)
    torch.testing.assert_close(actual, a * b, atol=0.015625, rtol=0.001)
    assert actual.is_contiguous(memory_format=torch.channels_last)
