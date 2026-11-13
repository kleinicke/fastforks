"""`restoration.load_file`: architecture detection from the state-dict keys, key mapping, clear unsupported errors."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import pytest
import torch
from safetensors.torch import save_file

from fastforks import restoration
from fastforks._core.bench import synthetic_image
from fastforks._core.checkpoint import WeightsError
from fastforks.restoration.models.dat.net import DAT
from fastforks.restoration.models.hat.net import HAT
from fastforks.restoration.models.realesrgan.net import RRDBNet
from fastforks.restoration.models.span.net import SPAN
from fastforks.restoration.models.swinir.net import SwinIR

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_NEW = {"conv_first.": "model.0.", "conv_up1.": "model.3.", "conv_up2.": "model.6.", "conv_hr.": "model.8."}


def _old_esrgan_keys(sd: dict[str, torch.Tensor], blocks: int) -> dict[str, torch.Tensor]:
    """RRDBNet's keys in the old ESRGAN `model.N` layout (the community upscalers' format)."""
    names = {**_NEW, "conv_body.": f"model.1.sub.{blocks}.", "conv_last.": "model.10."}
    out = {}
    for k, v in sd.items():
        if m := re.match(r"body\.(\d+)\.rdb(\d)\.conv(\d)\.", k):
            out[f"model.1.sub.{m.group(1)}.RDB{m.group(2)}.conv{m.group(3)}.0.{k[m.end() :]}"] = v
        else:
            prefix = next(p for p in names if k.startswith(p))
            out[names[prefix] + k[len(prefix) :]] = v
    return out


def _check(m: restoration.RestorationModel, net: torch.nn.Module, scale: int) -> None:
    img = synthetic_image(16, 24, torch.device("cpu"))
    with torch.inference_mode():
        want = net(img)
        got = m._runner(img)["image"]  # unclamped: random weights saturate the clamp
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4 * float(want.abs().max()))
    assert m.scale == scale


def test_old_esrgan_keys(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=2, num_grow_ch=32).eval()
    torch.save(_old_esrgan_keys(net.state_dict(), 2), tmp_path / "4x-community.pth")
    assert restoration.checkpoint_family(tmp_path / "4x-community.pth") == "realesrgan"
    m = restoration.load_file(tmp_path / "4x-community.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == "realesrgan-x4plus"
    _check(m, net, 4)


def test_span_safetensors(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = SPAN(3, 3, feature_channels=52, upscale=2).eval()
    save_file({k: v.contiguous() for k, v in net.state_dict().items()}, tmp_path / "2x-span.safetensors")
    m = restoration.load_file(tmp_path / "2x-span.safetensors", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == "span-x2"
    _check(m, net, 2)


def test_swinir_small(tmp_path: Path) -> None:
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
        net = SwinIR(
            upscale=2, in_chans=3, img_size=16, window_size=8, img_range=1.0, depths=[2], embed_dim=12,
            num_heads=[2], mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
        ).eval()  # fmt: skip
    torch.save({"params": net.state_dict()}, tmp_path / "swinir.pth")
    m = restoration.load_file(tmp_path / "swinir.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == "swinir-m-x2"
    _check(m, net, 2)


def _hat() -> torch.nn.Module:
    return HAT(
        upscale=2, in_chans=3, img_size=16, window_size=8, compress_ratio=3, squeeze_factor=6, conv_scale=0.01,
        overlap_ratio=0.5, img_range=1.0, depths=[2], embed_dim=24, num_heads=[2], mlp_ratio=2,
        upsampler="pixelshuffle", resi_connection="1conv",
    )  # fmt: skip


def _dat() -> torch.nn.Module:
    return DAT(
        upscale=3, in_chans=3, img_size=16, img_range=1.0, split_size=[2, 4], depth=[3], embed_dim=32,
        num_heads=[2], expansion_factor=2, resi_connection="3conv", upsampler="pixelshuffledirect",
    )  # fmt: skip


@pytest.mark.parametrize(
    ("build", "family", "variant", "scale"), [(_hat, "hat", "hat-s-x4", 2), (_dat, "dat", "dat-light-x4", 3)]
)
def test_swin_families(
    tmp_path: Path, build: Callable[[], torch.nn.Module], family: str, variant: str, scale: int
) -> None:
    """HAT and DAT files (their keys also match SwinIR's block pattern) load as their own family, any size."""
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "torch.meshgrid", UserWarning)
        net = build().eval()
    torch.save({"params_ema": net.state_dict()}, tmp_path / "net.pth")
    assert restoration.checkpoint_family(tmp_path / "net.pth") == family
    m = restoration.load_file(tmp_path / "net.pth", mode="portable", device="cpu", dtype="float32")
    assert m.variant.id == variant
    _check(m, net, scale)


def test_unsupported(tmp_path: Path) -> None:
    srvgg = {f"body.{i}.weight": torch.zeros(64, 3 if i == 0 else 64, 3, 3) for i in range(0, 6, 2)}
    torch.save(srvgg, tmp_path / "compact.pth")
    assert restoration.checkpoint_family(tmp_path / "compact.pth") is None
    with pytest.raises(WeightsError, match=r"compact\.pth: unsupported architecture"):
        restoration.load_file(tmp_path / "compact.pth")
    narrow = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=32, num_block=1, num_grow_ch=16)
    torch.save(narrow.state_dict(), tmp_path / "narrow.pth")
    with pytest.raises(WeightsError, match=r"features 32 \(fastforks: 64\).*not supported"):
        restoration.load_file(tmp_path / "narrow.pth")
    with pytest.raises(WeightsError, match="no such file"):
        restoration.load_file(tmp_path / "missing.pth")
